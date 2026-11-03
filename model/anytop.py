import torch
import torch.nn as nn
import numpy as np
from model.motion_transformer import (
    GraphMotionDecoderLayer,
    GraphMotionDecoder,
    circular_phase_embedding,
    run_in_fp32,
)
from model.joint_mask_utils import sample_subtree_joint_mask_batch
from utils.device_transfer import check_on_device, host_to_device
from data_loaders.truebones.truebones_utils.joint_struct_features import JOINT_STRUCT_DIM
from data_loaders.truebones.truebones_utils.canonical_features import MODEL_CHANNELS
from data_loaders.truebones.truebones_utils.action_label_conditioning_contract import (
    ACTION_LABEL_SLOTS,
    HEAD_SLOT_PRIMARY_WEIGHT,
    SLOT_DIRECTION,
    SLOT_MODIFIER,
    SLOT_HEAD,
    ActionConditioningError,
    projection_input_dim,
    slot_channel_widths,
    slot_source_rank_report,
    validate_action_conditioning_metadata,
    word_table_sha256,
)
from data_loaders.truebones.truebones_utils.motion_labels import (
    ACTION_LABEL_MAX_WORDS,
    CONTROLLED_VOCAB,
)
from data_loaders.truebones.truebones_utils.dataset_tags import species_code_dim
from data_loaders.reference import REST_GEOM_DIM
from model.joint_name_noise import JointNameNoise


def create_sin_embedding(positions: torch.Tensor, dim: int, max_period: float = 10000,
                         dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create sinusoidal positional embedding, with shape `[B, T, C]`.

    Args:
        positions (torch.Tensor): LongTensor of positions.
        dim (int): Dimension of the embedding.
        max_period (float): Maximum period of the cosine/sine functions.
        dtype (torch.dtype or str): dtype to use to generate the embedding.
    Returns:
        torch.Tensor: Sinusoidal positional embedding.
    """
    # We aim for BTC format
    assert dim % 2 == 0
    half_dim = dim // 2
    positions = positions.to(dtype)
    adim = torch.arange(half_dim, device=positions.device, dtype=dtype).view(1, 1, -1)
    max_period_tensor = torch.full([], max_period, device=positions.device, dtype=dtype)  # avoid sync point
    phase = positions / (max_period_tensor ** (adim / (half_dim - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)

def _zero_init_mlp(width):
    """Two-layer MLP whose last layer starts at zero (a no-op head)."""
    mlp = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
    nn.init.zeros_(mlp[-1].weight)
    nn.init.zeros_(mlp[-1].bias)
    return mlp


class AnyTop(nn.Module):
    def __init__(self, max_joints, feature_len,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 activation="gelu", t5_out_dim = 512,
                 **kargs):
        super().__init__()

        self.max_joints = max_joints
        self.feature_len = feature_len
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        # feature_len is the tensor width; the model reads and writes only the
        # channels canonical_features keeps (velocity is not a model channel).
        self.input_feats = len(MODEL_CHANNELS)
        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.value_emb=kargs.get('value_emb', False)
        self.cross_limb=kargs.get('cross_limb', True)
        self.cross_limb_latents=kargs.get('cross_limb_latents', 8)
        self.cross_limb_dim=kargs.get('cross_limb_dim', 64)
        self.cross_limb_last_n=kargs.get('cross_limb_last_n', 0)
        self.temporal_rope=bool(kargs.get('temporal_rope', False))
        self.temporal_rope_ref_drop_prob=float(kargs.get('temporal_rope_ref_drop_prob', 0.0))
        self.temporal_rope_drop_prob=float(kargs.get('temporal_rope_drop_prob', 0.0))
        for name in ('temporal_rope_ref_drop_prob', 'temporal_rope_drop_prob'):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'{name} must be in [0, 1]')
        # Hidden widths of conditioning heads; 0 selects latent_dim.
        self.action_adaln_bottleneck=int(kargs.get('action_adaln_bottleneck', 0))
        self.species_film_bottleneck=int(kargs.get('species_film_bottleneck', 128))
        self.last_layer_ff=int(kargs.get('last_layer_ff', 0))
        if self.species_film_bottleneck < 0:
            raise ValueError(
                f"species_film_bottleneck must be >= 0, got {self.species_film_bottleneck}"
            )
        self.joint_mask_prob=float(kargs.get('joint_mask_prob', 0.5))
        self.temporal_span_mask_prob=float(kargs.get('temporal_span_mask_prob', 0.0))
        self.temporal_span_mask_min_frames=int(kargs.get('temporal_span_mask_min_frames', 4))
        self.temporal_span_mask_max_frames=int(kargs.get('temporal_span_mask_max_frames', 20))
        self.unreliable_mask_drop_prob=float(kargs.get('unreliable_mask_drop_prob', 0.0))
        self.tempo_cfg_drop_prob=float(kargs.get('tempo_cfg_drop_prob', 0.1))
        if not 0.0 <= self.tempo_cfg_drop_prob <= 1.0:
            raise ValueError(
                f"tempo_cfg_drop_prob must be in [0, 1], got {self.tempo_cfg_drop_prob}"
            )
        # --species_cond: the species code (body plan + variant) FiLM-modulates
        # each joint-name embedding in InputProcess; that is its only entry point.
        # It is on exactly on the rows without an active reference, so it has
        # no CFG drop and no null mode of its own.
        self.species_cond=bool(kargs.get('species_cond', False))
        # Whole-joint name dropout (see InputProcess). It defaults to 0.0 so an
        # existing checkpoint rebuilds bit-identically and gains no state_dict key.
        self.joint_name_drop_prob=float(kargs.get('joint_name_drop_prob', 0.0))
        if not 0.0 <= self.joint_name_drop_prob <= 1.0:
            raise ValueError(
                f"joint_name_drop_prob must be in [0, 1], got {self.joint_name_drop_prob}"
            )
        # Training-only noise on the name embeddings (model/joint_name_noise.py); the
        # basis is attached after construction by set_joint_name_noise_basis.
        self.joint_name_noise_std=float(kargs.get('joint_name_noise_std', 0.0))
        # Rest-geometry drop (see _sample_rest_geom_drop): whole sample, then
        # single joints, drawn independently of the name drop.
        self.rest_geom_drop_prob=float(kargs.get('rest_geom_drop_prob', 0.0))
        self.rest_geom_joint_drop_prob=float(kargs.get('rest_geom_joint_drop_prob', 0.0))
        for key in ('rest_geom_drop_prob', 'rest_geom_joint_drop_prob'):
            if not 0.0 <= getattr(self, key) <= 1.0:
                raise ValueError(f"{key} must be in [0, 1], got {getattr(self, key)}")
        # Training-only; validated by GraphMotionDecoder.
        self.mirror_twin_drop_prob=float(kargs.get('mirror_twin_drop_prob', 0.0))
        self.lateral_drop_prob=float(kargs.get('lateral_drop_prob', 0.0))
        # Action-label conditioning: a single pathway -- the frozen T5 vectors of
        # the label's WORDS, pooled into one channel per slot (head /
        # direction / modifier) and concatenated. A channel reads its own slot
        # only, so a label's head and direction axes are literally unchanged by
        # however many modifiers it also spells, and unseen (action x direction)
        # combinations compose out of word vectors the model has already seen.
        self.action_label_cond = bool(kargs.get('action_label_cond', False))
        self.action_label_cfg_drop_prob = float(kargs.get('action_label_cfg_drop_prob', 0.2))
        if not 0.0 <= self.action_label_cfg_drop_prob <= 1.0:
            raise ValueError(
                f"action_label_cfg_drop_prob must be in [0, 1], got {self.action_label_cfg_drop_prob}"
            )
        # Slot dropout: with these probabilities a training sample keeps its
        # label but loses the words of ONE slot, so that slot's empty row is
        # trained as the marginal over its words rather than as a content key.
        #   direction: a bare "attack, swat" at inference draws one side rather
        #              than a blend of both.
        #   modifier:  a bare "attack" draws SOME attack (bite, cast, swat ...)
        #              rather than the handful of clips annotated without a
        #              modifier, which is what an undropped empty slot learns.
        # The two draws are independent per row.
        self.direction_slot_drop_prob = float(kargs.get('direction_slot_drop_prob', 0.0))
        if not 0.0 <= self.direction_slot_drop_prob <= 1.0:
            raise ValueError(
                f"direction_slot_drop_prob must be in [0, 1], got {self.direction_slot_drop_prob}"
            )
        self.modifier_slot_drop_prob = float(kargs.get('modifier_slot_drop_prob', 0.0))
        if not 0.0 <= self.modifier_slot_drop_prob <= 1.0:
            raise ValueError(
                f"modifier_slot_drop_prob must be in [0, 1], got {self.modifier_slot_drop_prob}"
            )
        # The action token also gets a multiplicative pathway through the
        # decoder on top of the additive one (GraphMotionDecoder owns the head).
        # It exists exactly when there is a label to drive it.
        self.action_label_adaln = self.action_label_cond
        if not 0.0 <= self.joint_mask_prob <= 1.0:
            raise ValueError(f"joint_mask_prob must be in [0, 1], got {self.joint_mask_prob}")
        if not 0.0 <= self.temporal_span_mask_prob <= 1.0:
            raise ValueError(
                f"temporal_span_mask_prob must be in [0, 1], got {self.temporal_span_mask_prob}"
            )
        if not 0.0 <= self.unreliable_mask_drop_prob <= 1.0:
            raise ValueError(
                f"unreliable_mask_drop_prob must be in [0, 1], got {self.unreliable_mask_drop_prob}"
            )
        if self.temporal_span_mask_min_frames < 1:
            raise ValueError(
                "temporal_span_mask_min_frames must be >= 1, got "
                f"{self.temporal_span_mask_min_frames}"
            )
        if self.temporal_span_mask_max_frames <= 0:
            raise ValueError("temporal_span_mask_max_frames must be >= 1")
        if 0 < self.temporal_span_mask_max_frames < self.temporal_span_mask_min_frames:
            raise ValueError(
                "temporal_span_mask_max_frames must be >= temporal_span_mask_min_frames "
                f"(got min={self.temporal_span_mask_min_frames}, max={self.temporal_span_mask_max_frames})"
            )

        self.input_process = InputProcess(
            self.input_feats, self.latent_dim, t5_out_dim,
            joint_name_noise_std=self.joint_name_noise_std, species_cond=self.species_cond,
            species_film_bottleneck=self.species_film_bottleneck,
        )
        # Token-type embedding for "this (frame, joint) is being re-drawn",
        # added to the input tokens right after InputProcess so the WHOLE trunk
        # (spatial / temporal attention, FFN, and through them the cross-limb
        # values and queries) can see the unreliable map. The cross-limb
        # reliability biases alone are logit shifts, which softmax cancels the
        # moment every valid joint of a frame is flagged -- whole-frame
        # inpainting and temporal spans were invisible. Zero-init, so the
        # path starts as a no-op.
        self.unreliable_embedding = nn.Parameter(torch.zeros(self.latent_dim))
        # y['is_loop'] (0/1, whether the window is a closed cycle) is NOT summed
        # into the condition embedding. It selects the loop time encoding
        # (_frame_position_embedding, _tempo_frame_embedding): a closed window
        # is a statement about time topology, and the circular table is the
        # place that can express it.
        # A global 0/1 token was tried and
        # removed: in the corpus the flag is confounded with content (two in
        # three loop clips are stationary idles, and within a species-tag
        # cluster the loop-authored gaits differ from the one-shot ones), so
        # the token became a content key -- a loop-conditioned run borrowed
        # the low-lift, phase-scrambled gait of the cluster's other loops
        # instead of the species' own one-shot runs, while the phase table
        # alone closed the window just as well (docs/anytop_model_architecture.md §6).
        # resample_speed_cond (source frames per window, L / T) is NOT summed
        # into the condition embedding either. It enters per frame
        # (_tempo_frame_embedding): an open window gets each frame's physical
        # time, a loop window the physical length of one cycle, L / k. A
        # statement about the window's time scale, instead of a global token
        # beside the action label where the frame count became a content key
        # that could outvote the label. The two get separate heads because
        # the same input value means a frame time in one and a period in the
        # other. Zero-init last layers, so a fresh model starts tempo-blind;
        # dropped in training with tempo_cfg_drop_prob as a regularizer.
        self.tempo_projection = _zero_init_mlp(self.latent_dim)
        self.loop_period_projection = _zero_init_mlp(self.latent_dim)
        # The loop window's cycle count k (y['loop_tile_count'] on loop rows)
        # has no token of its own: it enters only as the period of the
        # circular table (T / k) and of the loop tempo (L / k), so the model
        # can read it as time only.
        if self.action_label_cond:
            # Project the concatenated slot channels and add the result to the
            # timestep token. One
            # Linear over the concatenation IS one Linear per channel block,
            # summed, so each channel's relative scale is learned rather than
            # being some offline budget we picked. Additive and linear is enough:
            # direction is linearly readable out of the T5 vectors. If a trained
            # model still under-follows the prompt, raise --action_label_cfg_scale
            # (no retrain needed).
            # The direction slot is a code slot and is read on its
            # SYNTHETIC_CODE_DIM axes only: see compact_slot_channels.
            self._init_action_conditioning(kargs.get('action_conditioning'), t5_out_dim)
            self.action_label_projection = nn.Sequential(
                nn.Linear(projection_input_dim(t5_out_dim), self.latent_dim),
                nn.GELU(),
                nn.Linear(self.latent_dim, self.latent_dim),
            )
            # Learned unconditional embedding substituted when the condition is
            # hard-dropped. Zero-init so the unconditional mode starts identical
            # to a model with no action conditioning.
            self.action_label_null_emb = nn.Parameter(torch.zeros(self.latent_dim))
        else:
            self.action_label_projection = None
            self.action_label_null_emb = None
            self.action_conditioning_metadata = None

        seqTransDecoderLayer = GraphMotionDecoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)
        self.seqTransDecoder = GraphMotionDecoder(seqTransDecoderLayer,
                                                        num_layers=self.num_layers, value_emb=self.value_emb,
                                                        cross_limb=self.cross_limb,
                                                        cross_limb_latents=self.cross_limb_latents,
                                                        cross_limb_dim=self.cross_limb_dim,
                                                        cross_limb_last_n=self.cross_limb_last_n,
                                                        action_label_adaln=self.action_label_adaln,
                                                        action_adaln_bottleneck=self.action_adaln_bottleneck,
                                                        last_layer_ff=self.last_layer_ff,
                                                        mirror_twin_drop_prob=self.mirror_twin_drop_prob,
                                                        lateral_drop_prob=self.lateral_drop_prob,
                                                        temporal_rope=self.temporal_rope)
            
        
        self.output_process = OutputProcess(
            self.feature_len, self.input_feats, self.max_joints, self.latent_dim
        )
        self.ref_encoder = bool(kargs.get('ref_encoder', False))
        self.ref_cfg_drop_prob = float(kargs.get('ref_cfg_drop_prob', 0.4))
        if not 0 <= self.ref_cfg_drop_prob <= 1:
            raise ValueError('ref_cfg_drop_prob must be in [0, 1]')
        self.motion_encoder = None
        if self.ref_encoder:
            from model.motion_encoder import MotionEncoder
            ref_dim = int(kargs.get('ref_dim', 8))
            ref_width = int(kargs.get('ref_width', 64))
            if ref_width >= self.latent_dim:
                raise ValueError('ref_width must be smaller than the shared decoder latent_dim')
            self.motion_encoder = MotionEncoder(
                t5_out_dim, slots=int(kargs.get('ref_slots', 2)), ref_dim=ref_dim,
                width=ref_width, layers=int(kargs.get('ref_layers', 2)),
                dropout=self.dropout, joint_name_noise_std=self.joint_name_noise_std,
            )
            for layer in self.seqTransDecoder.layers:
                layer.enable_reference(ref_dim)

    def set_joint_name_noise_basis(self, basis, std):
        """Attach the name-noise basis (model/joint_name_noise.joint_name_noise_basis) to
        the target and reference name paths alike."""
        device = next(self.parameters()).device
        self.input_process.joint_name_noise.set_basis(basis, std, device)
        if self.motion_encoder is not None:
            self.motion_encoder.joint_name_noise.set_basis(basis, std, device)

    def reference_parameter_budget(self):
        encoder = sum(p.numel() for p in self.motion_encoder.parameters()) if self.motion_encoder else 0
        branches = sum(p.numel() for n, p in self.seqTransDecoder.named_parameters()
                       if '.ref_' in n)
        total = sum(p.numel() for p in self.parameters())
        return dict(total=total, shared=total - encoder - branches,
                    reference_encoder=encoder, reference_branches=branches)

    def _reference_condition(self, y, batch_size, frames, device):
        if self.motion_encoder is None:
            return None, None
        active = torch.as_tensor(y['ref_active'], device=device, dtype=torch.bool).reshape(-1)
        if active.numel() != batch_size:
            raise ValueError('ref_active must have shape [B]')
        active = active & y['ref_mask'].to(device).any(-1)
        ref_y = {k: v.to(device) for k, v in y.items() if k.startswith('ref_') and torch.is_tensor(v)}
        if ref_y['ref_feats'].shape[-1] != frames:
            raise ValueError('Reference and output windows must be frame aligned')
        if not self.training or self.ref_cfg_drop_prob == 0:
            return self.motion_encoder(ref_y), active
        # Reference dropout keeps a fixed number of randomly chosen rows per
        # batch, so the encoder runs on those rows only (static shape for
        # compile); a dropped row is hard-bypassed and its z is never read.
        keep = batch_size - round(batch_size * self.ref_cfg_drop_prob)
        encoder = self.motion_encoder
        if keep == 0:
            z = ref_y['ref_feats'].new_zeros(batch_size, frames, encoder.slots, encoder.ref_dim)
            return z, torch.zeros_like(active)
        rows = torch.randperm(batch_size, device=device)[:keep]
        z = encoder({k: v[rows] for k, v in ref_y.items()})
        z = z.new_zeros((batch_size,) + z.shape[1:]).index_copy(0, rows, z)
        kept = torch.zeros_like(active).index_fill(0, rows, True)
        return z, active & kept

    @staticmethod
    def _prepare_unreliable_mask(raw_mask, bs, nframes, njoints, device, dtype):
        """(nframes, B, J) float map from the (B, nframes, J) input; None when absent.

        Padding joints are never flagged by either producer (the samplers draw
        from valid joints only; inference maps derive from the inpaint mask over
        real joints), so nothing is re-zeroed here.
        """
        if raw_mask is None:
            return None
        mask = raw_mask.to(device=device, dtype=dtype)
        expected_shape = (bs, nframes, njoints)
        if mask.shape != expected_shape:
            raise ValueError(
                "y['cross_limb_unreliable_mask'] must have shape "
                f"{expected_shape}, got {tuple(mask.shape)}"
            )
        return mask.transpose(0, 1).contiguous()

    @staticmethod
    def _build_joint_key_padding_mask(njoints, n_joints, device):
        """Return the padding-only joint key mask used by attention.

        Training-time subtree perturbation deliberately stays out of the
        attention masks. Only structurally padded joints are masked here.
        """
        return torch.arange(njoints, device=device)[None, :] >= n_joints[:, None]

    def _coerce_resample_speed_cond(self, raw_resample_speed_cond, batch_size, device, dtype):
        if raw_resample_speed_cond is None:
            raw_resample_speed_cond = torch.ones(batch_size, device=device, dtype=dtype)
        elif not torch.is_tensor(raw_resample_speed_cond):
            raw_resample_speed_cond = torch.as_tensor(raw_resample_speed_cond, device=device)
        raw_resample_speed_cond = raw_resample_speed_cond.to(device=device)
        if raw_resample_speed_cond.dim() == 0:
            raw_resample_speed_cond = raw_resample_speed_cond.reshape(1)
        raw_resample_speed_cond = raw_resample_speed_cond.reshape(-1)
        if raw_resample_speed_cond.numel() == 1 and batch_size != 1:
            raw_resample_speed_cond = raw_resample_speed_cond.expand(batch_size)
        elif raw_resample_speed_cond.numel() != batch_size:
            raise ValueError(
                "resample_speed_cond batch dimension must match the motion batch size, got "
                f"{raw_resample_speed_cond.numel()} for batch {batch_size}"
            )
        # `.all()` in a python `if` is a data-dependent value that forces a
        # torch.compile graph break every step, so skip the finiteness guard
        # under compilation (eager eval/inference still validates).
        if not torch.compiler.is_compiling():
            check_on_device(torch.isfinite(raw_resample_speed_cond).all(),
                             "resample_speed_cond must be finite")
        return raw_resample_speed_cond.to(dtype=dtype)

    @staticmethod
    def _per_sample(raw, batch_size, device, dtype, name):
        value = torch.as_tensor(raw, device=device).to(dtype).reshape(-1)
        if value.numel() == 1 and batch_size != 1:
            return value.expand(batch_size)
        if value.numel() != batch_size:
            raise ValueError(
                f"{name} batch dimension must match the motion batch size, got "
                f"{value.numel()} for batch {batch_size}"
            )
        return value

    def _loop_condition(self, y, batch_size, device, ref_active=None):
        """``(is_loop, cycles)``, each ``[B]``.

        ``cycles`` is the loop window's cycle count k, from
        y['loop_tile_count'] (the loader's tile count, 1 when absent), and
        reads 1 on an open row and on a reference-active one (``ref_active``),
        so it can feed the period of the circular table directly. A reference
        supplies the motion, its cycles included, so like the action label and
        the species FiLM the count is bypassed there, in training and at
        inference alike: its table and tempo take the window as one cycle.
        Branch-free on tensor values (compile).
        """
        raw_loop = y.get('is_loop')
        if raw_loop is None:
            is_loop = torch.zeros(batch_size, device=device, dtype=torch.bool)
        else:
            is_loop = self._per_sample(raw_loop, batch_size, device, torch.bool, 'is_loop')
        raw_cycles = y.get('loop_tile_count')
        if raw_cycles is None:
            cycles = torch.ones(batch_size, device=device, dtype=torch.long)
        else:
            cycles = self._per_sample(
                raw_cycles, batch_size, device, torch.long, 'loop_tile_count',
            ).clamp(min=1)
        counted = is_loop if ref_active is None else is_loop & ~ref_active
        cycles = torch.where(counted, cycles, torch.ones_like(cycles))
        return is_loop, cycles

    def _frame_position_embedding(self, nframes, is_loop, cycles, device):
        """``[T, B, D]`` fp32 frame-position signal: the absolute frame PE for
        an open window, the circular table of period ``T / k`` for a loop one.

        A loop window gets no absolute PE: it would hand the model a frame
        position that bypasses the circular table and jumps at the seam.
        """
        positions = torch.arange(nframes, device=device).view(1, -1, 1)
        absolute = create_sin_embedding(positions, self.latent_dim)[0].unsqueeze(1)  # [T, 1, D]
        circular = circular_phase_embedding(
            nframes, self.latent_dim, device, torch.float32, cycles=cycles,
        )                                                                          # [T, B, D]
        return torch.where(is_loop.view(1, -1, 1), circular, absolute)

    def _tempo_frame_embedding(self, y, batch_size, nframes, device, is_loop=None, cycles=None):
        """``[nframes, B, D]`` tempo embedding.

        An open window: each frame's physical time, frame ``i`` at source frame
        ``i * (L - 1) / (T - 1)``. A loop window: the physical length of one
        cycle, ``L / k``, the same at every frame (``cycles`` is 1 on a
        reference-active row, so that row reads ``L``) -- frame positions inside
        the cycle are the circular table's job, and a ramp would jump at the
        seam. ``L = resample_speed_cond * T``. A dropped row (``tempo_active``
        False) gets zeros (the training-time tempo drop). ``is_loop`` /
        ``cycles`` default to :meth:`_loop_condition` of ``y``.
        """
        if is_loop is None or cycles is None:
            is_loop, cycles = self._loop_condition(y, batch_size, device)
        speed = self._coerce_resample_speed_cond(
            y.get('resample_speed_cond'), batch_size=batch_size, device=device, dtype=torch.float32,
        )
        source_frames = speed * float(nframes)
        if nframes > 1:
            open_step = ((source_frames.clamp_min(1.0) - 1.0) / float(nframes - 1)).clamp_min(0.0)
        else:
            open_step = torch.ones_like(speed)
        frame_times = (
            torch.arange(nframes, device=device, dtype=torch.float32).view(1, -1, 1)
            * open_step.view(-1, 1, 1)
        )
        open_emb = run_in_fp32(
            self.tempo_projection, create_sin_embedding(frame_times, self.latent_dim)
        )                                                                          # [B, T, D]
        period = (source_frames / cycles.to(torch.float32)).view(-1, 1, 1)
        loop_emb = run_in_fp32(
            self.loop_period_projection, create_sin_embedding(period, self.latent_dim)
        )                                                                          # [B, 1, D]
        tempo_emb = torch.where(is_loop.view(-1, 1, 1), loop_emb, open_emb)
        active = self._resolve_tempo_active(y.get('tempo_active'), batch_size, device)
        tempo_emb = tempo_emb * active.view(-1, 1, 1).to(tempo_emb.dtype)
        return tempo_emb.transpose(0, 1)

    def _init_action_conditioning(self, bundle, t5_out_dim):
        """Freeze the word table into a model buffer.

        A persistent buffer, so a checkpoint carries the exact vectors its weights
        were fitted on and inference never has to pick a sidecar out of a data
        directory. With no bundle (the inference path) it is a placeholder that
        ``load_state_dict`` overwrites; ``validate_loaded_action_conditioning``
        is what certifies what landed in it.
        """
        vocab_size = len(CONTROLLED_VOCAB)
        if bundle is None:
            table = torch.zeros(vocab_size, int(t5_out_dim))
            self.action_conditioning_metadata = None
        else:
            if bundle.embedding_dim != int(t5_out_dim):
                raise ValueError(
                    f"the action word table is {bundle.embedding_dim}-dimensional but "
                    f"t5_out_dim is {t5_out_dim}. It was encoded with a different T5 "
                    "than cond.npy's joints_names_embs; rebuild it with --t5-model "
                    "matching cond.npy."
                )
            # The first Linear has to be wide enough to stay injective on the
            # direct sum of the slot source spaces, or labels that differ only in
            # slot membership can collide before any weight is trained.
            report = bundle.slot_source_rank_report(self.latent_dim)
            if not report['full_rank']:
                raise ValueError(
                    f"{bundle.source}: the frozen word table does not have full "
                    f"slot-source rank ({report['slots']}), so distinct labels are not "
                    "guaranteed to reach distinct conditions. Rebuild it with "
                    "tools/build_action_label_embeddings.py --force."
                )
            if not report['fits_projection']:
                raise ValueError(
                    f"latent_dim {self.latent_dim} is smaller than the total slot source "
                    f"rank {report['total_rank']}, so action_label_projection cannot be "
                    "injective on the reachable condition space -- some pairs of labels "
                    f"would be indistinguishable by construction. Train with --latent_dim "
                    f"{report['total_rank']} or more."
                )
            table = torch.from_numpy(
                np.array(bundle.word_embeddings, dtype=np.float32, copy=True)
            )
            self.action_conditioning_metadata = bundle.checkpoint_metadata()
        self.register_buffer('action_word_embeddings', table, persistent=True)

    def validate_loaded_action_conditioning(self, metadata, source: str):
        """Bind loaded weights to the contract they were trained under.

        Called right after ``load_state_dict``: the checkpoint's word-table
        buffer is the authoritative material for THOSE weights, so it is
        certified against the hash the checkpoint's own contract commits to,
        never assumed to equal what a sidecar on this host holds today.
        """
        if not self.action_label_cond:
            return None
        validated = validate_action_conditioning_metadata(metadata, source=source)
        # "The weights loaded but the buffer did not" is the failure a reader has
        # to see, so the placeholder is ruled out before anything hashes it.
        table = self.action_word_embeddings.detach().cpu().numpy()
        if table.shape[0] != len(CONTROLLED_VOCAB):
            raise ActionConditioningError(
                f"{source}: its word table has {table.shape[0]} rows, this code's "
                f"vocabulary has {len(CONTROLLED_VOCAB)}"
            )
        if not np.isfinite(table).all() or not np.any(table):
            raise ActionConditioningError(
                f"{source}: its word table did not load (all zero or non-finite)"
            )
        # The vectors that arrived in the buffers, against the hash the
        # checkpoint's own contract commits to. Shape and finiteness say the
        # table loaded; only this says it is the table these weights were fitted
        # on -- and it is the check that makes the loading ORDER matter, so it
        # has to run on buffers that already hold the checkpoint's values.
        declared_table_hash = validated['embedding_contract'].get('word_table_sha256')
        actual_table_hash = word_table_sha256(table)
        if declared_table_hash != actual_table_hash:
            raise ActionConditioningError(
                f"{source}: its word table hashes to {actual_table_hash}, but the "
                f"embedding_contract it was saved with commits to "
                f"{declared_table_hash!r}. These weights were fitted on different "
                "word vectors than the ones now in the buffers."
            )
        report = slot_source_rank_report(table, self.latent_dim)
        if not report['full_rank'] or not report['fits_projection']:
            raise ActionConditioningError(
                f"{source}: its word table and latent_dim {self.latent_dim} do not "
                f"certify separable slot channels (total slot source rank "
                f"{report['total_rank']}, full_rank={report['full_rank']})"
            )
        self.action_conditioning_metadata = validated
        return validated

    def _coerce_action_slot_field(self, value, name, batch_size, device, dtype):
        """Normalize one loader field to ``[B, W]``, broadcasting a single row."""
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() != 2:
            raise ValueError(
                f"{name} must have shape (W,) or (B, W), got {tuple(tensor.shape)}"
            )
        if tensor.shape[1] > ACTION_LABEL_MAX_WORDS:
            raise ValueError(
                f"{name} has {tensor.shape[1]} word columns, over the contract cap "
                f"{ACTION_LABEL_MAX_WORDS}"
            )
        if tensor.shape[0] == 1 and batch_size != 1:
            tensor = tensor.expand(batch_size, -1)
        elif tensor.shape[0] != batch_size:
            raise ValueError(
                f"{name} batch dimension must match the motion batch size, got "
                f"{tensor.shape[0]} for batch {batch_size}"
            )
        return tensor

    def _action_slot_inputs(self, y, batch_size, device):
        """The loader's word-level condition, or ``None`` when y carries none.

        The loader emits ids and masks, never assembled vectors, so the frozen
        table lives in exactly one place -- this model. All three fields travel
        together: a label's meaning is the (word, slot) set, and accepting a
        partial one would silently condition on a different label.
        """
        raw_word_ids = y.get('action_word_ids')
        if raw_word_ids is None:
            return None
        word_ids = self._coerce_action_slot_field(
            raw_word_ids, 'action_word_ids', batch_size, device, torch.long
        )
        fields = {}
        for name, dtype in (
            ('action_slot_ids', torch.long),
            ('action_word_mask', torch.bool),
        ):
            raw = y.get(name)
            if raw is None:
                raise ValueError(
                    f"y carries action_word_ids but no {name}. The word ids alone do "
                    "not say which slot a word feeds, so the condition cannot be "
                    "assembled from them."
                )
            fields[name] = self._coerce_action_slot_field(
                raw, name, batch_size, device, dtype
            )
        mismatched = {
            name: tuple(tensor.shape)
            for name, tensor in fields.items()
            if tensor.shape != word_ids.shape
        }
        if mismatched:
            raise ValueError(
                "action_word_ids, action_slot_ids and action_word_mask must all "
                f"have the same shape; action_word_ids is {tuple(word_ids.shape)} "
                f"but {mismatched}"
            )
        if not torch.compiler.is_compiling():
            vocab_size = int(self.action_word_embeddings.shape[0])
            live = fields['action_word_mask']
            check_on_device(
                ~(((word_ids < 0) | (word_ids >= vocab_size)) & live).any(),
                f"action_word_ids holds an id outside 0..{vocab_size - 1}; the "
                "loader's ordered vocabulary is not this model's.")
        return word_ids, fields['action_slot_ids'], fields['action_word_mask']

    def _action_slot_member_weights(self, slot_ids, word_mask, dtype):
        """Tensor mirror of ``slot_member_weights``: ``[B, W]``.

        Every word weighs 1.0 except the head slot's first member, which weighs
        ``HEAD_SLOT_PRIMARY_WEIGHT``. "First" is the earliest live column of the
        row assigned to the head slot, found with a cumulative sum rather than a
        search so the shapes stay static for compile/cudagraph, and read off the
        same written order the numpy side uses.
        """
        head_member = word_mask & (slot_ids == SLOT_HEAD)
        first_head = head_member & (head_member.to(torch.int32).cumsum(dim=1) == 1)
        return torch.where(
            first_head,
            torch.full_like(first_head, HEAD_SLOT_PRIMARY_WEIGHT, dtype=dtype),
            torch.ones_like(first_head, dtype=dtype),
        )

    def _assemble_action_slot_channels(self, word_ids, slot_ids, word_mask, dtype):
        """Tensor mirror of ``assemble_slot_channels``: ``[B, S * D]``.

        Same rule, same slot ids, one channel per slot: the weighted mean of
        that slot's member word vectors, L2-normalised, and a zero row for an
        absent slot. Only the head slot has a weight other than 1.0, and only on
        its first member, so the other three are plain means of a set. Because a
        channel is a function of its own slot's members only, appending a
        non-head word moves the head and direction channels by exactly zero.
        """
        vectors = self.action_word_embeddings.to(dtype)[word_ids]
        weights = self._action_slot_member_weights(slot_ids, word_mask, dtype)
        channels = []
        for slot in range(len(ACTION_LABEL_SLOTS)):
            member = (
                (word_mask & (slot_ids == slot)).to(dtype) * weights
            ).unsqueeze(-1)
            total = member.sum(dim=1)
            # min=1.0 rather than an epsilon: every weight is >= 1.0, so a
            # non-empty slot's total is never below it and the clamp only ever
            # bites on the empty one -- whose numerator is the zero vector. An
            # epsilon divisor would underflow to zero under fp16 autocast and
            # make an absent slot NaN in BOTH branches of the where below.
            mean = (vectors * member).sum(dim=1) / total.clamp(min=1.0)
            norm = torch.linalg.vector_norm(mean, dim=-1, keepdim=True)
            # An absent slot is a zero row, never a renormalisation of the others:
            # "this label spells no direction" has to stay distinguishable from
            # "this label spells one".
            channels.append(torch.where(total > 0, mean / norm.clamp(min=1e-9), mean * 0.0))
        return torch.cat(channels, dim=-1)

    def _compact_action_slot_channels(self, channels):
        """Tensor mirror of ``compact_slot_channels``: ``[B, S * D]`` ->
        ``[B, projection_input_dim(D)]``.

        A code slot's channel is a renormalised mean of one-hot code rows, so
        outside its SYNTHETIC_CODE_DIM axes it is zero for every legal label;
        the projection reads only that prefix. Static slices, so the graph is
        the same for every batch.
        """
        dim = self.action_word_embeddings.shape[1]
        blocks = []
        for slot, width in enumerate(slot_channel_widths(dim)):
            start = slot * dim
            blocks.append(channels[:, start:start + width])
        return torch.cat(blocks, dim=-1)

    def _drop_slot_words(self, word_mask, slot_ids, slot, prob, batch_size, device):
        """Training-only: blank the words of *slot* on a random subset of rows.

        A pure mask operation, written like the CFG keep mask so it compiles
        the same way: a dropped row's members of that slot leave ``word_mask``,
        the slot's channel pools to its zero row, and every other slot is
        untouched. Rows with no word in the slot are unaffected, and the label
        stays valid (a head word is never dropped and every label carries one,
        so ``action_label_valid`` cannot flip). Eval and inference never drop.
        """
        if not self.training or prob <= 0.0:
            return word_mask
        drop = torch.rand(batch_size, device=device) < prob
        return word_mask & ~(drop[:, None] & (slot_ids == slot))

    def _drop_direction_slot(self, word_mask, slot_ids, batch_size, device):
        return self._drop_slot_words(
            word_mask, slot_ids, SLOT_DIRECTION, self.direction_slot_drop_prob, batch_size, device
        )

    def _drop_modifier_slot(self, word_mask, slot_ids, batch_size, device):
        return self._drop_slot_words(
            word_mask, slot_ids, SLOT_MODIFIER, self.modifier_slot_drop_prob, batch_size, device
        )

    def _resolve_action_label_active(self, raw_action_label_active, batch_size, device):
        """Per-sample CFG mask for the action condition (True == conditional).

        An explicit ``y['action_label_active']`` wins (used at inference to force
        the unconditional branch). Otherwise, training samples a Bernoulli keep
        per sample so the condition is hard-dropped with probability
        ``action_label_cfg_drop_prob``; eval/inference keeps every sample.
        """
        if raw_action_label_active is not None:
            active = torch.as_tensor(raw_action_label_active, device=device, dtype=torch.bool).reshape(-1)
            if active.numel() == 1 and batch_size != 1:
                active = active.expand(batch_size)
            elif active.numel() != batch_size:
                raise ValueError(
                    "action_label_active batch dimension must match the motion batch size, got "
                    f"{active.numel()} for batch {batch_size}"
                )
            return active
        if self.training and self.action_label_cfg_drop_prob > 0.0:
            return torch.rand(batch_size, device=device) >= self.action_label_cfg_drop_prob
        return torch.ones(batch_size, device=device, dtype=torch.bool)

    def _resolve_action_label_valid(self, y, batch_size, device, word_mask):
        """Which rows actually carry a label (an empty label = no condition).

        Prefers the loader's explicit ``y['action_label_valid']``; falls back to
        "did this row carry any word at all", which is the same statement made by
        the mask.
        """
        raw_valid = y.get('action_label_valid')
        if raw_valid is not None:
            valid = torch.as_tensor(raw_valid, device=device, dtype=torch.bool).reshape(-1)
            if valid.numel() == 1 and batch_size != 1:
                valid = valid.expand(batch_size)
            elif valid.numel() != batch_size:
                raise ValueError(
                    "action_label_valid batch dimension must match the motion batch size, got "
                    f"{valid.numel()} for batch {batch_size}"
                )
            return valid
        if word_mask is None:
            return torch.zeros(batch_size, device=device, dtype=torch.bool)
        return word_mask.any(dim=-1)

    def _action_condition(self, y, batch_size, device, dtype, ref_active=None):
        """``(channels [B, S*D], active [B])`` for the action condition, or ``None``.

        ``active`` folds together the CFG hard-drop mask and "does this row carry
        a label at all": a row with no label carries no action information, and
        routing it through the same unconditional path as a dropped row is what
        makes omitting ``--action_label`` at inference land on the learned
        unconditional mode automatically.

        Every row is conditioned on its label as written; there is no longer a
        class of borrowed rows to route differently (auxiliary groups are
        retired -- with --action_group all every clip is simply in the corpus).
        """
        if not self.action_label_cond:
            return None
        resolved = self._action_slot_inputs(y, batch_size, device)
        if resolved is None:
            channels = torch.zeros(
                batch_size, self.action_label_projection[0].in_features,
                device=device, dtype=dtype,
            )
            word_mask = None
        else:
            word_ids, slot_ids, word_mask = resolved
            word_mask = self._drop_direction_slot(word_mask, slot_ids, batch_size, device)
            word_mask = self._drop_modifier_slot(word_mask, slot_ids, batch_size, device)
            channels = self._compact_action_slot_channels(
                self._assemble_action_slot_channels(word_ids, slot_ids, word_mask, dtype)
            )
        active = self._resolve_action_label_active(
            y.get('action_label_active'), batch_size, device
        )
        active = active & self._resolve_action_label_valid(
            y, batch_size, device, word_mask
        )
        # A reference supplies the motion content, so the action label is
        # bypassed on every reference-active row, in training and at inference.
        if ref_active is not None:
            active = active & ~ref_active
        return channels, active

    def _build_action_label_token(self, y, batch_size, device, dtype, ref_active=None):
        """The action token added to the timestep embedding, or ``None`` when off."""
        if self.action_label_projection is None:
            return None
        resolved = self._action_condition(y, batch_size, device, dtype, ref_active)
        if resolved is None:
            return None
        channels, action_active = resolved
        # An inactive row goes to the learned null rather than pushing the zero
        # vector through the projection, whose output there is an untrained region.
        null_emb = self.action_label_null_emb.to(device=device, dtype=dtype)
        return torch.where(
            action_active.view(batch_size, 1),
            run_in_fp32(self.action_label_projection, channels),
            null_emb.unsqueeze(0).expand(batch_size, -1),
        )

    def sample_subtree_joint_mask_train(self, y, njoints, device):
        """Select subtrees of joints to perturb during training.

        Returns a bool tensor of shape ``[B, njoints]`` (True = joint selected),
        or ``None`` if not in training mode or if ``joint_mask_prob == 0`` -- so eval-mode
        loss reports a clean diffusion objective.

        When the feature is on but the draw selects nothing, the return is an
        all-False mask, NOT ``None``: ``None`` makes
        ``training_losses`` drop ``y['cross_limb_unreliable_mask']``, and a key
        set that varies between steps forces torch.compile to recompile the
        forward the first time an empty batch comes up. An all-False mask is a
        no-op for both the re-noise and the cross-limb reliability bias, so the
        step is unchanged.

        Called from ``GaussianDiffusion.training_losses`` AFTER ``q_sample``
        to decide which joints' x_t slice should be re-noised at an independent
        timestep drawn from ``[t, T)`` with fresh noise, so that those joints
        carry strictly less information about x_0 than the rest of the sample. This trains
        the cross-joint pathway to denoise robustly against per-joint
        timestep mismatch -- the regime inpaint clamping produces at
        inference. The model's forward itself stays vanilla.
        """
        if (not self.training) or self.joint_mask_prob <= 0.0:
            return None
        n_joints_cpu = torch.as_tensor(
            y.get('n_joints_cpu', y['n_joints']), device='cpu', dtype=torch.int64
        ).reshape(-1)
        subtree_joint_mask = self._sample_subtree_joint_mask(y, n_joints_cpu, njoints, device)
        if subtree_joint_mask is None:
            subtree_joint_mask = torch.zeros(
                (int(n_joints_cpu.shape[0]), int(njoints)), dtype=torch.bool, device=device
            )
        return subtree_joint_mask

    def _sample_subtree_joint_mask(self, y, n_joints, njoints, device):
        parents_batch = y.get('parents')
        if parents_batch is None:
            return None
        limb_roots_batch = y.get('joint_mask_limb_roots')
        if torch.is_tensor(n_joints):
            n_joints_np = n_joints.detach().to(device='cpu', dtype=torch.int64).numpy()
        else:
            n_joints_np = np.asarray(n_joints, dtype=np.int64)

        if torch.is_tensor(parents_batch):
            parents_batch_np = parents_batch.detach().to(device='cpu', dtype=torch.int64).numpy()
        else:
            parents_batch_np = [np.asarray(parents, dtype=np.int64) for parents in parents_batch]

        if limb_roots_batch is None:
            limb_roots_np = None
        elif torch.is_tensor(limb_roots_batch):
            limb_roots_np = limb_roots_batch.detach().to(device='cpu').numpy()
        else:
            limb_roots_np = np.asarray(limb_roots_batch, dtype=np.bool_)

        subtree_joint_mask_np = sample_subtree_joint_mask_batch(
            parents_batch=parents_batch_np,
            limb_root_mask_batch=limb_roots_np,
            n_joints=n_joints_np,
            max_joints=njoints,
            joint_mask_prob=self.joint_mask_prob,
            rng=np.random,
        )
        if subtree_joint_mask_np is None:
            return None
        return host_to_device(subtree_joint_mask_np, device)

    def sample_temporal_span_mask_train(self, y, njoints, nframes, device):
        """Select contiguous temporal spans to perturb during training.

        Returns a bool tensor of shape ``[B, njoints, nframes]`` (True =
        re-noise that joint/frame cell), or ``None`` if not in training mode or
        if ``temporal_span_mask_prob == 0``.

        As with ``sample_subtree_joint_mask_train``, a draw that selects no span
        returns an all-False mask rather than ``None``, so ``y``'s key set --
        and therefore the compiled graph -- stays identical across steps. It
        also keeps the sampler sync-free: no ``.any()`` readback per step.

        Full-clip spans are not supported: the sampled span always leaves at
        least one frame at the start or end unmasked so seam detection never
        triggers on a clip-span boundary.  ``temporal_span_mask_max_frames``
        is therefore capped at ``min(config_max, nframes - 1)`` at runtime.
        Every real joint in the sample shares the same masked frames, while
        padded joints stay False throughout.

        A k-tiled loop window (``y['loop_tile_count']``, from the loader's
        ``_tile_loop_motion``) is k copies of one cycle, so one span drawn
        anywhere in it has a clean twin one period away and the model could
        fill it by copying. For those samples the span is drawn within one
        period and repeated at every period -- the same phase is re-drawn in
        every copy -- and capped at ``period - 1`` so each cycle keeps an
        unmasked frame. An untiled window has ``period == nframes`` and gets
        the single contiguous span as before.
        """
        if (not self.training) or self.temporal_span_mask_prob <= 0.0:
            return None

        # Prefer the device copy: building the mask on the device from the host
        # counts would be a blocking upload every step.
        n_joints_batch = y.get('n_joints', y.get('n_joints_cpu'))
        if n_joints_batch is None:
            return None

        # Fully vectorized, on-device sampling: no host<->device sync and no
        # Python loop, so this stays compile/cudagraph friendly. Shapes are
        # static across batches (frames resampled to nframes, joints padded to
        # njoints). Randomness uses torch RNG, which is checkpointed alongside
        # the numpy RNG state, so resume reproducibility is preserved (the
        # exact sample sequence differs from the previous numpy path).
        n_joints_t = torch.as_tensor(n_joints_batch, device=device, dtype=torch.long).reshape(-1)
        batch_size = n_joints_t.shape[0]
        if batch_size == 0 or nframes <= 0:
            return None

        min_span = min(self.temporal_span_mask_min_frames, nframes)
        max_span = max(min_span, min(self.temporal_span_mask_max_frames, nframes - 1))

        valid_joints = n_joints_t.clamp(min=0, max=njoints)                       # [B]
        active = (torch.rand(batch_size, device=device) < self.temporal_span_mask_prob) \
            & (valid_joints > 0)                                                  # [B]
        # Period of the window's content in frames: nframes / tile count. Float
        # on purpose -- a tile count that does not divide nframes makes copies
        # that are interpolated rather than bit-identical, and the mask must
        # still land on the same phase of each one.
        tile_count = y.get('loop_tile_count')
        if tile_count is None:
            period = torch.full((batch_size,), float(nframes), device=device)
        else:
            tile_count = torch.as_tensor(tile_count, device=device).reshape(-1)
            period = float(nframes) / tile_count.clamp(min=1).to(torch.float32)  # [B]
        # Per-sample span range: the configured one, capped so each period
        # keeps at least one unmasked frame (period - 1, which for an untiled
        # window is the nframes - 1 cap above).
        span_hi = torch.minimum(
            torch.full_like(period, float(max_span)), period.floor() - 1.0
        ).clamp(min=1.0)                                                          # [B]
        span_lo = torch.minimum(torch.full_like(period, float(min_span)), span_hi)  # [B]
        # randint(lo, hi + 1) per sample: scale [0,1) by (hi-lo+1) then floor,
        # clamping guards the rare rand()==~1.0 rounding past hi.
        span_length = (
            span_lo + torch.rand(batch_size, device=device) * (span_hi - span_lo + 1.0)
        ).floor()
        span_length = torch.minimum(span_length, span_hi)                         # [B]
        # Start inside the first period, so every repeat lies inside its own.
        start_hi = (period - span_length).clamp(min=0.0).floor()                  # [B]
        span_start = (torch.rand(batch_size, device=device) * (start_hi + 1.0)).floor()
        span_start = torch.minimum(span_start, start_hi)                          # [B]

        frame_idx = torch.arange(nframes, device=device, dtype=torch.float32)    # [T]
        # Frame t is masked when its phase past the span start, taken modulo
        # the period, falls inside the span. Frames before the start in the
        # first period wrap to >= period - start >= span length: never masked.
        phase = torch.remainder(frame_idx[None, :] - span_start[:, None], period[:, None])  # [B, T]
        frame_mask = phase < span_length[:, None]                                # [B, T]
        joint_idx = torch.arange(njoints, device=device)                         # [J]
        joint_mask = joint_idx[None, :] < valid_joints[:, None]                   # [B, J]

        mask = (
            active[:, None, None]
            & joint_mask[:, :, None]
            & frame_mask[:, None, :]
        )                                                                         # [B, J, T]
        return mask

    def sample_unreliable_mask_cond_drop_train(self, batch_size, device):
        """Decide, per sample, whether to HIDE the training-time unreliable mask.

        Returns a bool tensor of shape ``[B]`` (True = do not tell the model
        which joints/frames were re-noised), or ``None`` when the feature is off
        or we are not training.

        The corruption itself is unaffected: the selected joints / frames still
        get their independent-timestep re-noise, only the
        ``y['cross_limb_unreliable_mask']`` handed to the forward is zeroed for
        the dropped rows. Without this, the model only ever sees mixed
        reliability together with a map of exactly where it is, and at inference
        without a mask (plain generation, or repairing a motion whose damaged
        joints are not known) it has never had to localize the damage itself.
        Dropping the map on a fraction of steps trains that pathway too, in the
        same way CFG dropout trains the unconditional branch.

        Compile-safety: the returned row gate only zeroes an existing tensor, so
        ``y``'s key set and every tensor shape are identical on every step;
        the draw is one on-device ``rand`` with no host sync.
        """
        if (not self.training) or self.unreliable_mask_drop_prob <= 0.0:
            return None
        batch_size = int(batch_size)
        if batch_size <= 0:
            return None
        return torch.rand(batch_size, device=device) < self.unreliable_mask_drop_prob

    def _coerce_species_code(self, y, batch_size, device, dtype):
        """Pull y['species_code'] and broadcast it to [B, species_code_dim]."""
        raw_species_code = y.get('species_code') if y is not None else None
        if raw_species_code is None:
            raise ValueError(
                "species conditioning is enabled but y['species_code'] is missing; it is "
                "bound from the species descriptor table (bind_cond_species_embs)."
            )
        species_code = torch.as_tensor(raw_species_code, device=device, dtype=dtype)
        if species_code.dim() == 1:
            species_code = species_code.unsqueeze(0)
        if species_code.shape[0] == 1 and batch_size != 1:
            species_code = species_code.expand(batch_size, -1)
        elif species_code.shape[0] != batch_size:
            raise ValueError(
                "species_code batch dimension must match the motion batch size, got "
                f"{species_code.shape[0]} for batch {batch_size}"
            )
        return species_code

    def _resolve_tempo_active(self, raw_tempo_active, batch_size, device):
        """Per-sample keep mask for the tempo condition. An explicit
        y['tempo_active'] overrides;
        otherwise training samples a Bernoulli keep so the condition is hard-dropped
        with probability tempo_cfg_drop_prob; eval/inference keeps every sample."""
        if raw_tempo_active is not None:
            active = torch.as_tensor(raw_tempo_active, device=device, dtype=torch.bool).reshape(-1)
            if active.numel() == 1 and batch_size != 1:
                active = active.expand(batch_size)
            elif active.numel() != batch_size:
                raise ValueError(
                    "tempo_active batch dimension must match the motion batch size, got "
                    f"{active.numel()} for batch {batch_size}"
                )
            return active
        if self.training and self.tempo_cfg_drop_prob > 0.0:
            return torch.rand(batch_size, device=device) >= self.tempo_cfg_drop_prob
        return torch.ones(batch_size, device=device, dtype=torch.bool)

    def forward(self, x, timesteps, y=None, train_step=None, **unused_kwargs):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        joints_padding_mask = y['joints_padding_mask'].to(x.device)
        rest_geom = y['rest_geom'].to(x.device)

        bs, njoints, nfeats, nframes = x.shape
        ref_z, ref_active = self._reference_condition(y, bs, nframes, x.device)
        n_joints = torch.as_tensor(y['n_joints'], device=x.device).reshape(-1)
        joint_key_padding_mask = self._build_joint_key_padding_mask(njoints, n_joints, x.device)
        # joint-mask subtree perturbation is applied OUTSIDE this
        # forward, in diffusion.training_losses, by re-noising the selected
        # joints' x_t with q_sample(x_0, t_random, fresh_noise). The model
        # itself stays vanilla -- selected joints DO NOT enter the
        # key-padding masks and continue to participate in attention normally,
        # just with mismatched noise levels, which trains the
        # cross-joint pathway to denoise robustly against per-joint timestep
        # disagreement (matching inpaint clamp behavior at inference).
        timesteps_emb = create_sin_embedding(timesteps.view(1, -1, 1), self.latent_dim)[0]
        action_label_token = self._build_action_label_token(y, bs, x.device, x.dtype, ref_active)
        if action_label_token is not None:
            timesteps_emb = timesteps_emb + action_label_token

        species_code = (
            self._coerce_species_code(y, bs, x.device, x.dtype) if self.species_cond else None
        )
        # A reference supplies the motion content, so the target species FiLM is
        # bypassed on every reference-active row and applied on every other row,
        # in training and at inference alike.
        species_active = (
            ~ref_active if self.species_cond and ref_active is not None else None
        )
        # Per-joint validity for the whole-joint name dropout: joints_padding_mask is
        # the outer product valid[i]*valid[j] over max_joints+1 slots whose slot 0 is
        # the always-present leading token, so joint j lives at slot j+1 and the
        # diagonal of the [1:, 1:] block is the per-joint flag. Read from the mask
        # rather than from all-zero name rows -- 256 live joints across the corpus
        # (locators, prop sockets, Bip01) carry a deliberately blank name.
        joint_valid = torch.diagonal(
            joints_padding_mask[:, 0, 0, 1:, 1:], dim1=-2, dim2=-1
        ) > 0.5
        if y.get('joint_struct') is None:
            raise ValueError(
                "y['joint_struct'] is missing. The loader and sample/generate.py both "
                "build it from the cond entry via build_joint_struct_features; a batch "
                "without it comes from a caller that assembles model_kwargs by hand."
            )
        joint_name_drop = self._sample_joint_name_drop(bs, njoints, x.device)
        rest_geom_drop = self._sample_rest_geom_drop(bs, njoints, x.device)
        is_loop, loop_cycles = self._loop_condition(y, bs, x.device, ref_active)
        x = self.input_process(x, rest_geom, y['joints_names_embs'], joint_valid, y['joint_struct'],
                               joint_name_drop=joint_name_drop, rest_geom_drop=rest_geom_drop,
                               species_code=species_code,
                               species_active=species_active,
                               frame_pos_emb=self._frame_position_embedding(
                                   nframes, is_loop, loop_cycles, x.device,
                               )) # applies linear layer on each frame to convert it to latent dim
        tempo_emb = self._tempo_frame_embedding(y, bs, nframes, x.device, is_loop, loop_cycles)
        x = x + tempo_emb.unsqueeze(2).to(x.dtype)
        spatial_mask = (1.0 - joints_padding_mask[:, 0, 0, 1:, 1:].float()) * -1e4
        spatial_mask = spatial_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        # Temporal self-attention is unmasked: every frame token attends over the
        # whole window.

        # One (nframes, B, J) unreliable map, 1 == being re-drawn, shared by the
        # input embedding below and every cross-limb block. Training loss and
        # inference both hand over the (B, nframes, J) layout.
        cross_limb_unreliable_mask = self._prepare_unreliable_mask(
            y.get('cross_limb_unreliable_mask'), bs, nframes, njoints, x.device, x.dtype
        )
        if cross_limb_unreliable_mask is not None:
            x = x + cross_limb_unreliable_mask.unsqueeze(-1) * self.unreliable_embedding

        # The decoder re-adds the same circular table (period T / k) per layer
        # on loop rows.
        output = self.seqTransDecoder(
            temporal_rope_off=self._temporal_rope_off(y, bs, x.device, ref_active),
            tgt=x,
            timesteps_embs=timesteps_emb,
            memory=None,
            spatial_mask=spatial_mask,
            tgt_key_padding_mask=joint_key_padding_mask,
            y=y,
            cross_limb_unreliable_mask=cross_limb_unreliable_mask,
            loop_phase_mask=is_loop,
            loop_cycles=loop_cycles,
            # Same token the additive path summed in above, so a CFG-dropped row
            # modulates by the null embedding rather than skipping the head.
            action_adaln_cond=action_label_token if self.action_label_adaln else None,
            ref_z=ref_z,
            ref_active=ref_active,
        )
        output = self.output_process(output) # Applies linear layer on each frame to convert it back to feature len dim
        return output

    def _temporal_rope_off(self, y, batch_size, device, ref_active):
        """``[B]`` bool, True = the row's temporal attention runs without RoPE,
        or None when every row keeps it.

        Training drops RoPE per row: ``temporal_rope_ref_drop_prob`` on
        reference-active rows, ``temporal_rope_drop_prob`` on the rest. With
        the reference drop on, inference runs every reference-active row
        without RoPE; the reference CFG null pass keeps RoPE on.
        y['temporal_rope_off'] can additionally disable it on explicit rows.
        """
        ref_p, other_p = self.temporal_rope_ref_drop_prob, self.temporal_rope_drop_prob
        if not self.temporal_rope or (ref_p == 0 and other_p == 0):
            return None
        if ref_active is None:
            ref_active = torch.zeros(batch_size, device=device, dtype=torch.bool)
        if self.training:
            prob = torch.where(ref_active, ref_p, other_p)
            return torch.rand(batch_size, device=device) < prob
        if ref_p == 0:
            return None
        off = ref_active
        null_off = y.get('temporal_rope_off')
        if null_off is not None:
            off = off | torch.as_tensor(null_off, device=device, dtype=torch.bool).reshape(-1)
        return off

    def _sample_joint_name_drop(self, batch_size, joint_count, device):
        """Whole-joint name dropout mask, ``[B, J]`` bool, or None when off.

        In training, a per-(sample, joint) Bernoulli at ``joint_name_drop_prob``
        (some names unknown, as when ``process_new_skeleton`` blanks unseen
        texts). InputProcess zeroes the selected name rows. Padding rows are zero
        already, so they need no exclusion. Eval keeps every name.

        Branch-free on tensor values (``torch.rand`` + compare, no ``.any()`` in a
        python ``if``) so it does not break torch.compile or cudagraph capture.
        """
        if not self.training:
            return None
        if self.joint_name_drop_prob <= 0.0:
            return None
        return torch.rand(batch_size, joint_count, device=device) < self.joint_name_drop_prob

    def _sample_rest_geom_drop(self, batch_size, joint_count, device):
        """Rest-geometry drop mask, ``[B, J]`` bool, or None when off.

        In training, a joint's rest geometry is hidden when its whole sample
        draws ``rest_geom_drop_prob`` or the joint itself draws
        ``rest_geom_joint_drop_prob``. InputProcess zeroes the selected rows,
        present flag included. Eval keeps every row. Branch-free on tensor
        values, like _sample_joint_name_drop.
        """
        if not self.training:
            return None
        if self.rest_geom_drop_prob <= 0.0 and self.rest_geom_joint_drop_prob <= 0.0:
            return None
        sample_drop = torch.rand(batch_size, 1, device=device) < self.rest_geom_drop_prob
        joint_drop = torch.rand(batch_size, joint_count, device=device) < self.rest_geom_joint_drop_prob
        return sample_drop | joint_drop


    def _apply(self, fn):
        super()._apply(fn)


    def train(self, *args, **kwargs):
        # Return self like nn.Module does: eval() is `return self.train(False)`,
        # so dropping the value made `model.eval()` evaluate to None.
        return super().train(*args, **kwargs)

# in the case of GMDM, the input process is as follows:
# embed each joint of each frame of each motion in batch by the same MLP, separately !
class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim, t5_output_dim, joint_name_noise_std=0.0,
                 species_cond=False, species_film_bottleneck=128):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        # One embedding for every joint, the root included: its channels mean
        # the same (canonical_features, "Root"), and its name tells it apart.
        self.joint_embedding = nn.Linear(self.input_feats, self.latent_dim)
        # Training-time noise in the names' principal subspace (model/joint_name_noise.py):
        # nearby names become interchangeable, so a name carries coarse semantics
        # rather than an exact per-name identity.
        self.joint_name_noise = JointNameNoise(joint_name_noise_std)
        # Whole-joint name dropout. Noise blurs a name but never hides the joint's
        # identity, so on its own the model is never trained on "I do not know what
        # this joint is" and never learns to fall back on rest geometry / graph_dist /
        # joints_relations. The name embedding
        # is the only per-joint identity signal in the token (it is added straight
        # onto x below), which is why a single rare token -- Mixamo's "LeftLeg"
        # landing next to "Left Arm" in T5 space -- can flip a whole limb's motion
        # prior. Dropping the entire row for a random subset of joints demotes the
        # name from sole signal to one signal among several.
        # A dropped name is an all-zero row, the same "no name" the padding rows
        # carry. Padding never reaches a live joint (the attention mask and the
        # loss both exclude it), so sharing the zero row costs nothing, and it
        # leaves inference a parameter-free way to feed a joint with no name.
        # The mask is drawn by AnyTop (_sample_joint_name_drop).
        # When --species_cond, the species code FiLM-modulates each per-joint
        # name embedding: gamma/beta are produced from the *concatenation* of
        # that joint's embedding and the species code, so the modulation is a
        # function of the species x joint interaction.
        #
        # The concat (rather than a sum) is load-bearing: text_embedding is a
        # bare Linear with no nonlinearity in between, so an additive
        # `joints_emb + species_proj(species)` fusion collapses to
        # `E.j_i + const(s)` -- the same per-species constant for every joint,
        # i.e. zero per-joint differentiation. With concat that additive form
        # stays representable (gamma_residual=0, beta=W(s)), so the FiLM optimum
        # is no worse.
        #
        # The final linear is zero-initialized so the head starts at identity
        # (gamma=1, beta=0).
        species_hidden = species_film_bottleneck or self.latent_dim
        self.species_film_j = nn.Sequential(
            nn.Linear(t5_output_dim + species_code_dim(t5_output_dim), species_hidden),
            nn.GELU(),
            nn.Linear(species_hidden, 2 * t5_output_dim),
        ) if species_cond else None
        if self.species_film_j is not None:
            nn.init.zeros_(self.species_film_j[-1].weight)
            nn.init.zeros_(self.species_film_j[-1].bias)
        self.text_embedding = nn.Linear(t5_output_dim, self.latent_dim)
        # Structural channel: the joint's place in the topology (its position
        # along a branch-free run, whether it is a leaf, its subtree size, its
        # sibling rank) and its side, projected into every frame's token
        # alongside the name. Ordinary init, NOT zero: this is meant to carry
        # signal from step one, and it is a per-joint identity left when
        # joint_name_drop_prob blanks a joint's name.
        self.struct_embedding = nn.Sequential(
            nn.Linear(JOINT_STRUCT_DIM, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        # Rest-geometry channel: rest_geometry's REST_GEOM_COLUMNS (rest bone
        # direction, height and bone length over L) plus the present flag,
        # projected into every frame's token. The only path by which the rest
        # geometry reaches the denoiser, so rest_geom_drop hides all of it.
        self.rest_geom_embedding = nn.Sequential(
            nn.Linear(REST_GEOM_DIM, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

    def forward(self, x, rest_geom, joints_embedded_names, joint_valid=None,
                joint_struct=None, joint_name_drop=None, rest_geom_drop=None,
                species_code=None, species_active=None, frame_pos_emb=None):
        """Embed every (frame, joint) with its static per-joint channels.

        ``rest_geom`` is ``[B, J, REST_GEOM_DIM]`` (y['rest_geom']).
        ``joint_name_drop`` ([B, J] bool) zeroes those joints' whole name rows,
        leaving rest geometry and the topology as their identity signals. No
        1/(1-p) rescale: a hidden row is not a thinned vector.
        ``rest_geom_drop`` ([B, J] bool) zeroes those joints' rest geometry,
        present flag included.

        ``species_code`` ([B, species_code_dim]) drives the species FiLM and is
        required exactly when it exists; a row whose ``species_active`` is False
        bypasses the FiLM entirely.

        ``frame_pos_emb`` ([T, B, D], AnyTop._frame_position_embedding) is the
        frame-position signal; None adds the absolute frame PE to every sample.
        """
        # x.shape = [batch_size, joints, feature_len, frames]
        x = x.permute(3, 0, 1, 2) # [frames, batch_size, n_joints, features_len]
        x = self.joint_embedding(x[..., :len(MODEL_CHANNELS)])
        joints_clean = joints_embedded_names.to(x.device)
        if joint_valid is None:
            # AnyTop.forward always supplies the real mask; a bare InputProcess
            # call (tests, ablations) treats every joint as live.
            joint_valid = torch.ones(
                joints_clean.shape[:2], device=joints_clean.device, dtype=torch.bool
            )
        else:
            joint_valid = joint_valid.to(device=joints_clean.device, dtype=torch.bool)
        # Whole-joint dropout runs FIRST, so everything downstream -- the noise,
        # the FiLM condition, the projection -- sees the zeroed row and the hidden
        # name cannot leak back in through the species pathway. The noise skips
        # all-zero rows, so a dropped or unknown name stays exactly zero.
        if joint_name_drop is not None:
            joints_clean = joints_clean.masked_fill(
                joint_name_drop.to(joints_clean.device).unsqueeze(-1), 0.0
            )
        joints_embedded_names = self.joint_name_noise(joints_clean)
        if self.species_film_j is not None:
            if species_code is None:
                raise ValueError(
                    "species_cond is enabled but species_code was not passed to "
                    "InputProcess (expected y['species_code'])."
                )
            # The FiLM condition reads the same noised names as the projection: the
            # noise has no train-time rescale, and FiLM should see the same coarse
            # name the token does.
            # joints_embedded_names: [B, J, t5]; species_code: [B, code] -> [B, J, code]
            species_broadcast = species_code.to(device=x.device, dtype=joints_embedded_names.dtype).unsqueeze(1).expand(-1, joints_embedded_names.shape[1], -1)
            gamma_residual, beta = run_in_fp32(
                self.species_film_j, torch.cat([joints_embedded_names, species_broadcast], dim=-1)
            ).chunk(2, dim=-1)
            modulated = (1.0 + gamma_residual) * joints_embedded_names + beta
            # Drop the entire modulation, including its joint-dependent component.
            # Zeroing only the species input would leave a learned conditional path.
            if species_active is not None:
                active = species_active.to(device=x.device, dtype=torch.bool).view(-1, 1, 1)
                modulated = torch.where(active, modulated, joints_embedded_names)
            joints_embedded_names = modulated
        elif species_code is not None:
            raise ValueError("species_code was passed but this InputProcess has no species FiLM.")
        joints_embedded_names = run_in_fp32(self.text_embedding, joints_embedded_names)
        x = x + joints_embedded_names[None, ...]# [frames, batch_size, n_joints, d]
        if joint_struct is None:
            raise ValueError(
                "no joint_struct was passed; see build_joint_struct_features."
            )
        # The name-dropout probabilities deliberately do NOT reach here: the
        # point of the structural channel is to be the signal that survives a
        # missing name.
        struct_latent = run_in_fp32(
            self.struct_embedding,
            joint_struct.to(device=x.device, dtype=x.dtype),
        )
        rest_geom = rest_geom.to(device=x.device, dtype=x.dtype)
        if rest_geom_drop is not None:
            rest_geom = rest_geom.masked_fill(rest_geom_drop.to(x.device)[..., None], 0.0)
        rest_latent = run_in_fp32(self.rest_geom_embedding, rest_geom)
        # Re-zero AFTER the projections. The padded rows arrive as zeros, but
        # the MLPs have biases, so a zero row does not stay zero through them.
        # A dropped rest row is a live joint and keeps the bias: present=0 is
        # its "geometry unknown" input.
        valid = joint_valid.unsqueeze(-1).to(struct_latent.dtype)
        x = x + ((struct_latent + rest_latent) * valid)[None, ...]
        # fp32 on purpose: this add promotes the bf16 residual stream back to
        # fp32 before the decoder.
        if frame_pos_emb is None:
            positions = torch.arange(x.shape[0], device=x.device).view(1, -1, 1)
            frame_pos_emb = create_sin_embedding(positions, self.latent_dim)[0].unsqueeze(1)
        x = x + frame_pos_emb.to(device=x.device, dtype=torch.float32).unsqueeze(2)
        return x


class OutputProcess(nn.Module):
    """Predicts the kept channels of every joint and writes them into a
    ``feature_len``-wide tensor whose velocity channels are zero
    (canonical_features, "Velocity")."""

    def __init__(self, feature_len, joint_feature_len, max_joints, latent_dim):
        super().__init__()
        self.feature_len = feature_len
        self.max_joints = max_joints
        self.latent_dim = latent_dim
        self.joint_feature_len = joint_feature_len
        self.joint_dembedding = nn.Linear(self.latent_dim, self.joint_feature_len)

    def forward(self, output):
        # output shape [frames, batch_size, joints, latent_dim]
        # The x0 prediction must not be quantized to bf16: its per-frame rounding is
        # white noise that time derivatives amplify, inflating jerk/snap/spectral
        # flatness (docs/bf16_precision_issues.md). A small linear, so fp32 is free.
        with torch.autocast(device_type=output.device.type, enabled=False):
            joints_kept = self.joint_dembedding(output.float())
        output = nn.functional.pad(joints_kept, (0, self.feature_len - self.joint_feature_len))
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output
