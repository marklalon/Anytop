from model.motion_conditioning import MotionConditioningMixin, create_sin_embedding, _zero_init_mlp
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


class AnyTop(MotionConditioningMixin, nn.Module):
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
        # Per-joint geometry drop, independent of name drop.
        self.rest_geom_joint_drop_prob = float(kargs.get('rest_geom_joint_drop_prob', 0.0))
        if not 0.0 <= self.rest_geom_joint_drop_prob <= 1.0:
            raise ValueError('rest_geom_joint_drop_prob must be in [0, 1]')
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
            species_cond=self.species_cond,
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
                                                        temporal_rope=self.temporal_rope)
            
        
        self.output_process = OutputProcess(
            self.feature_len, self.input_feats, self.max_joints, self.latent_dim
        )
        self.ref_encoder = bool(kargs.get('ref_encoder', False))
        self.ref_cfg_drop_prob = float(kargs.get('ref_cfg_drop_prob', 0.4))
        if not 0 <= self.ref_cfg_drop_prob <= 1:
            raise ValueError('ref_cfg_drop_prob must be in [0, 1]')
        self.reference_encoder = None
        if self.ref_encoder:
            from model.reference_encoder import ReferenceEncoder
            ref_dim = int(kargs.get('ref_dim', 8))
            ref_width = int(kargs.get('ref_width', 64))
            if ref_width >= self.latent_dim:
                raise ValueError('ref_width must be smaller than the shared decoder latent_dim')
            self.reference_encoder = ReferenceEncoder(
                t5_out_dim, slots=int(kargs.get('ref_slots', 2)), ref_dim=ref_dim,
                width=ref_width, layers=int(kargs.get('ref_layers', 2)),
                dropout=self.dropout,
            )
            for layer in self.seqTransDecoder.layers:
                layer.enable_reference(ref_dim)

    def reference_parameter_budget(self):
        encoder = sum(p.numel() for p in self.reference_encoder.parameters()) if self.reference_encoder else 0
        branches = sum(p.numel() for n, p in self.seqTransDecoder.named_parameters()
                       if '.ref_' in n)
        total = sum(p.numel() for p in self.parameters())
        return dict(total=total, shared=total - encoder - branches,
                    reference_encoder=encoder, reference_branches=branches)


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


    def _apply(self, fn):
        super()._apply(fn)


    def train(self, *args, **kwargs):
        # Return self like nn.Module does: eval() is `return self.train(False)`,
        # so dropping the value made `model.eval()` evaluate to None.
        return super().train(*args, **kwargs)

# in the case of GMDM, the input process is as follows:
# embed each joint of each frame of each motion in batch by the same MLP, separately !
class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim, t5_output_dim,
                 species_cond=False, species_film_bottleneck=128):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        # One embedding for every joint, the root included: its channels mean
        # the same (canonical_features, "Root"), and its name tells it apart.
        self.joint_embedding = nn.Linear(self.input_feats, self.latent_dim)
        # Whole-joint name dropout encourages fallback to rest geometry,
        # graph distances and joint relations when a name is unknown.
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
        names, geometry = self.static_components(
            rest_geom, joints_embedded_names, x.device, x.dtype, joint_valid,
            joint_struct, joint_name_drop, rest_geom_drop, species_code, species_active)
        x = x + names[None]
        x = x + geometry[None]
        # fp32 on purpose: this add promotes the bf16 residual stream back to
        # fp32 before the decoder.
        if frame_pos_emb is None:
            positions = torch.arange(x.shape[0], device=x.device).view(1, -1, 1)
            frame_pos_emb = create_sin_embedding(positions, self.latent_dim)[0].unsqueeze(1)
        x = x + frame_pos_emb.to(device=x.device, dtype=torch.float32).unsqueeze(2)
        return x

    def static_components(self, rest_geom, joints_embedded_names, device, dtype,
                          joint_valid=None, joint_struct=None, joint_name_drop=None,
                          rest_geom_drop=None, species_code=None, species_active=None):
        """Shared static skeleton channels, without motion or frame positions."""
        joints_clean = joints_embedded_names.to(device)
        if joint_valid is None:
            # AnyTop.forward always supplies the real mask; a bare InputProcess
            # call (tests, ablations) treats every joint as live.
            joint_valid = torch.ones(
                joints_clean.shape[:2], device=joints_clean.device, dtype=torch.bool
            )
        else:
            joint_valid = joint_valid.to(device=joints_clean.device, dtype=torch.bool)
        # Whole-joint dropout runs before FiLM and projection, so the hidden
        # name cannot leak back in through the species pathway.
        if joint_name_drop is not None:
            joints_clean = joints_clean.masked_fill(
                joint_name_drop.to(joints_clean.device).unsqueeze(-1), 0.0
            )
        joints_embedded_names = joints_clean
        if self.species_film_j is not None:
            if species_code is None:
                raise ValueError(
                    "species_cond is enabled but species_code was not passed to "
                    "InputProcess (expected y['species_code'])."
                )
            # FiLM and projection read the same names after whole-joint dropout.
            # joints_embedded_names: [B, J, t5]; species_code: [B, code] -> [B, J, code]
            species_broadcast = species_code.to(device=device, dtype=joints_embedded_names.dtype).unsqueeze(1).expand(-1, joints_embedded_names.shape[1], -1)
            gamma_residual, beta = run_in_fp32(
                self.species_film_j, torch.cat([joints_embedded_names, species_broadcast], dim=-1)
            ).chunk(2, dim=-1)
            modulated = (1.0 + gamma_residual) * joints_embedded_names + beta
            # Drop the entire modulation, including its joint-dependent component.
            # Zeroing only the species input would leave a learned conditional path.
            if species_active is not None:
                active = species_active.to(device=device, dtype=torch.bool).view(-1, 1, 1)
                modulated = torch.where(active, modulated, joints_embedded_names)
            joints_embedded_names = modulated
        elif species_code is not None:
            raise ValueError("species_code was passed but this InputProcess has no species FiLM.")
        joints_embedded_names = run_in_fp32(self.text_embedding, joints_embedded_names)
        # The original motion path adds names before converting structure/rest:
        # that add promotes its AMP residual to fp32. Preserve the conversion
        # precision when evaluating the static components separately.
        dtype = torch.promote_types(dtype, joints_embedded_names.dtype)
        if joint_struct is None:
            raise ValueError(
                "no joint_struct was passed; see build_joint_struct_features."
            )
        # The name-dropout probabilities deliberately do NOT reach here: the
        # point of the structural channel is to be the signal that survives a
        # missing name.
        struct_latent = run_in_fp32(
            self.struct_embedding,
            joint_struct.to(device=device, dtype=dtype),
        )
        rest_geom = rest_geom.to(device=device, dtype=dtype)
        if rest_geom_drop is not None:
            rest_geom = rest_geom.masked_fill(rest_geom_drop.to(device)[..., None], 0.0)
        rest_latent = run_in_fp32(self.rest_geom_embedding, rest_geom)
        # Re-zero AFTER the projections. The padded rows arrive as zeros, but
        # the MLPs have biases, so a zero row does not stay zero through them.
        # A dropped rest row is a live joint and keeps the bias: present=0 is
        # its "geometry unknown" input.
        valid = joint_valid.unsqueeze(-1).to(struct_latent.dtype)
        return joints_embedded_names, (struct_latent + rest_latent) * valid


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
