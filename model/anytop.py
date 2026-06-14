import torch
torch.cuda.empty_cache()
import torch.nn as nn
import numpy as np
from model.motion_transformer import GraphMotionDecoderLayer, GraphMotionDecoder
from model.joint_mask_utils import sample_subtree_joint_mask_batch
from model.morphology_expert import (
    resolve_morphology_ids,
    object_types_to_group_id_tensor,
    validate_morphology_registry,
    validate_object_type_to_group_id,
    MORPHOLOGY_GROUPS,
)


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

class AnyTop(nn.Module):
    def __init__(self, max_joints, feature_len,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 activation="gelu", t5_out_dim = 512, root_input_feats=13,
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
        self.input_feats = self.feature_len
        self.root_input_feats = root_input_feats
        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.value_emb=kargs.get('value_emb', False)
        self.cross_limb=kargs.get('cross_limb', True)
        self.cross_limb_latents=kargs.get('cross_limb_latents', 8)
        self.cross_limb_dim=kargs.get('cross_limb_dim', 64)
        self.cross_limb_last_n=kargs.get('cross_limb_last_n', 0)
        self.joint_mask_prob=float(kargs.get('joint_mask_prob', 0.5))
        self.joint_mask_budget=float(kargs.get('joint_mask_budget', 0.15))
        self.temporal_span_mask_prob=float(kargs.get('temporal_span_mask_prob', 0.0))
        self.temporal_span_mask_min_frames=int(kargs.get('temporal_span_mask_min_frames', 4))
        self.temporal_span_mask_max_frames=int(kargs.get('temporal_span_mask_max_frames', 12))
        self.global_energy_cond=bool(kargs.get('global_energy_cond', False))
        self.species_cond=bool(kargs.get('species_cond', False))
        # Morphology (form-class) expert: per-group bottleneck adapters routed
        # by object_type. Structural condition (always on, not part of CFG).
        self.morphology_expert=bool(kargs.get('morphology_expert', False))
        self.morphology_expert_bottleneck=int(kargs.get('morphology_expert_bottleneck', 64))
        self.morphology_expert_layers=str(kargs.get('morphology_expert_layers', 'last4'))
        self.morphology_expert_dropout=float(kargs.get('morphology_expert_dropout', 0.05))
        self.morphology_tags_path=kargs.get('morphology_tags_path', None)
        # A checkpoint freezes its object_type -> group_id table (and registry
        # order) into args.json. When that frozen mapping is supplied, use it
        # verbatim -- so the live species_tags.jsonl can be edited/moved without
        # silently re-routing an existing checkpoint -- and validate the saved
        # registry is still a prefix of the (append-only) MORPHOLOGY_GROUPS
        # constant. Only a fresh run with no frozen mapping reads the tags file.
        saved_groups = kargs.get('morphology_groups', None)
        saved_mapping = kargs.get('morphology_object_type_to_group_id', None)
        if self.morphology_expert:
            if saved_mapping:
                self.morphology_groups = validate_morphology_registry(
                    saved_groups if saved_groups else MORPHOLOGY_GROUPS
                )
                self.object_type_to_group_id = validate_object_type_to_group_id(
                    saved_mapping, len(self.morphology_groups)
                )
            else:
                self.morphology_groups, self.object_type_to_group_id = resolve_morphology_ids(
                    self.morphology_tags_path
                )
        else:
            self.morphology_groups, self.object_type_to_group_id = (), {}
        self.loop_cond_prob=float(kargs.get('loop_cond_prob', 1.0))
        self.global_energy_stats_momentum = 0.01
        # CFG drop probability for global energy conditioning during training.
        # When > 0, randomly replaces the energy condition with the running mean
        # so the model learns to actually respond to it (otherwise zero-init
        # FiLM has no incentive to move away from identity).
        self.global_energy_cfg_drop_prob = float(kargs.get('global_energy_cfg_drop_prob', 0.1))
        # Action-tag conditioning. A multi-hot over the canonical action-tag
        # vocabulary is projected and added to the timestep token (same additive
        # pathway as loop/playspeed). When enabled, each sample's action
        # condition is hard-dropped with probability action_tag_cfg_drop_prob
        # during training — replaced by a learned null embedding so the model
        # also learns an unconditional mode, enabling classifier-free guidance.
        self.action_tag_cond = bool(kargs.get('action_tag_cond', False))
        self.action_tag_cfg_drop_prob = float(kargs.get('action_tag_cfg_drop_prob', 0.3))
        if not 0.0 <= self.action_tag_cfg_drop_prob <= 1.0:
            raise ValueError(
                f"action_tag_cfg_drop_prob must be in [0, 1], got {self.action_tag_cfg_drop_prob}"
            )
        if not 0.0 <= self.joint_mask_prob <= 1.0:
            raise ValueError(f"joint_mask_prob must be in [0, 1], got {self.joint_mask_prob}")
        if not 0.0 <= self.joint_mask_budget <= 1.0:
            raise ValueError(f"joint_mask_budget must be in [0, 1], got {self.joint_mask_budget}")
        if not 0.0 <= self.temporal_span_mask_prob <= 1.0:
            raise ValueError(
                f"temporal_span_mask_prob must be in [0, 1], got {self.temporal_span_mask_prob}"
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

        self.input_process = InputProcess(self.input_feats, self.root_input_feats, self.latent_dim, t5_out_dim, dropout_prob=self.dropout)
        if self.global_energy_cond:
            # NOTE: do NOT prepend nn.LayerNorm(1) here. LayerNorm over a
            # size-1 feature dim maps every scalar input to a constant (mean=x
            # => x-mean=0), which silently collapses the energy condition so
            # --global_energy has no effect. The input is already Z-scored
            # against the running stats in _build_global_energy_token, so no
            # normalization layer is needed.
            self.global_energy_projection = nn.Sequential(
                nn.Linear(1, self.latent_dim),
                nn.GELU(),
                nn.Linear(self.latent_dim, self.latent_dim),
            )
            self.register_buffer('global_energy_running_mean', torch.zeros(1, dtype=torch.float32))
            self.register_buffer('global_energy_running_var', torch.ones(1, dtype=torch.float32))
            self.register_buffer('global_energy_running_count', torch.zeros((), dtype=torch.long))
        else:
            self.global_energy_projection = None
        if self.loop_cond_prob > 0.0:
            self.loop_condition_projection = nn.Sequential(
                nn.Linear(1, self.latent_dim),
                nn.GELU(),
                nn.Linear(self.latent_dim, self.latent_dim),
            )
        else:
            self.loop_condition_projection = None
        self.playspeed_projection = nn.Sequential(
            nn.Linear(1, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        if self.action_tag_cond:
            # Import the canonical vocabulary lazily so the model module does not
            # pull in the data-loader stack unless action conditioning is used.
            from data_loaders.truebones.truebones_utils.motion_labels import ACTION_TAGS
            self.action_tag_vocab = list(ACTION_TAGS)
            self.action_tag_to_index = {tag: i for i, tag in enumerate(self.action_tag_vocab)}
            n_action_tags = len(self.action_tag_vocab)
            # Linear over the multi-hot == sum of per-tag learned embeddings; the
            # MLP lets active tags interact. The all-zero multi-hot ("no tag")
            # maps to the projection's bias, which is intentionally distinct from
            # the hard-dropped (null) state below.
            self.action_tag_projection = nn.Sequential(
                nn.Linear(n_action_tags, self.latent_dim),
                nn.GELU(),
                nn.Linear(self.latent_dim, self.latent_dim),
            )
            # Learned unconditional embedding substituted when the condition is
            # hard-dropped. Zero-init so the unconditional mode starts identical
            # to a model with no action conditioning.
            self.action_tag_null_emb = nn.Parameter(torch.zeros(self.latent_dim))
        else:
            self.action_tag_vocab = []
            self.action_tag_to_index = {}
            self.action_tag_projection = None

        # Per-species condition: a clean, T5-derived species descriptor
        # (separate from the per-joint name embeddings) added to the timestep
        # embedding, which every decoder layer re-injects via embed_timesteps
        # -> deep, per-layer modulation. The final linear is zero-initialized so
        # the species condition starts at 0 (identity to the baseline), giving a
        # near-no-regret addition that only deviates if it lowers the loss.
        if self.species_cond:
            self.species_projection = nn.Sequential(
                nn.Linear(t5_out_dim, self.latent_dim),
                nn.GELU(),
                nn.Linear(self.latent_dim, self.latent_dim),
            )
            nn.init.zeros_(self.species_projection[-1].weight)
            nn.init.zeros_(self.species_projection[-1].bias)
        else:
            self.species_projection = None

        seqTransDecoderLayer = GraphMotionDecoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation,
                                                            global_energy_cond=self.global_energy_cond)
        self.seqTransDecoder = GraphMotionDecoder(seqTransDecoderLayer,
                                                        num_layers=self.num_layers, value_emb=self.value_emb,
                                                        cross_limb=self.cross_limb,
                                                        cross_limb_latents=self.cross_limb_latents,
                                                        cross_limb_dim=self.cross_limb_dim,
                                                        cross_limb_last_n=self.cross_limb_last_n,
                                                        morphology_expert=self.morphology_expert,
                                                        morphology_num_groups=len(self.morphology_groups),
                                                        morphology_expert_bottleneck=self.morphology_expert_bottleneck,
                                                        morphology_expert_layers=self.morphology_expert_layers,
                                                        morphology_expert_dropout=self.morphology_expert_dropout)
            
        
        self.output_process = OutputProcess(self.feature_len, self.root_input_feats, self.max_joints, self.latent_dim)

    @staticmethod
    def _build_joint_key_padding_mask(njoints, n_joints, device):
        """Return the padding-only joint key mask used by attention.

        Training-time subtree perturbation deliberately stays out of the
        attention masks. Only structurally padded joints are masked here.
        """
        return torch.arange(njoints, device=device)[None, :] >= n_joints[:, None]

    def _update_global_energy_running_stats(self, raw_global_energy_cond, energy_active=None):
        if not self.global_energy_cond or raw_global_energy_cond.numel() == 0:
            return

        batch_stats = raw_global_energy_cond.detach().to(dtype=torch.float32)
        # CFG-dropped samples are unconditional this step: they must not leak
        # into the running stats (the model sees no energy for them). Apply the
        # mask as 0/1 weights rather than `batch_stats[energy_active]` -- boolean
        # masked-select lowers to aten.nonzero, a data-dependent dynamic-shape op
        # that forces a torch.compile graph break every step. The weighted
        # moments below are identical to mean/var(unbiased=False) over the active
        # rows, but keep static shapes.
        if energy_active is None:
            weight = torch.ones(batch_stats.shape[0], device=batch_stats.device, dtype=torch.float32)
        else:
            weight = energy_active.to(device=batch_stats.device, dtype=torch.float32).reshape(-1)
        w = weight.unsqueeze(1)
        count = weight.sum()
        safe_count = count.clamp_min(1.0)
        batch_mean = (batch_stats * w).sum(dim=0) / safe_count
        batch_var = (((batch_stats - batch_mean.unsqueeze(0)) ** 2) * w).sum(dim=0) / safe_count
        batch_var = batch_var.clamp_min(1e-6)
        with torch.no_grad():
            # Keep this update entirely in tensor space so torch.compile does
            # not graph-break/recompile on a changing Python scalar count. When
            # no sample is active this step (count == 0), update_weight is forced
            # to 0 so the running stats and count stay untouched.
            has_obs = count > 0
            is_first_batch = self.global_energy_running_count.eq(0)
            momentum = torch.where(
                is_first_batch,
                torch.ones_like(batch_mean),
                torch.full_like(batch_mean, self.global_energy_stats_momentum),
            )
            update_weight = torch.where(has_obs, momentum, torch.zeros_like(batch_mean))
            self.global_energy_running_mean.copy_(
                torch.lerp(self.global_energy_running_mean, batch_mean, update_weight)
            )
            self.global_energy_running_var.copy_(
                torch.lerp(self.global_energy_running_var, batch_var, update_weight)
            )
            self.global_energy_running_count.add_(count.to(self.global_energy_running_count.dtype))

    def _coerce_global_energy_condition(self, raw_global_energy_cond, batch_size, device, dtype):
        # raw_global_energy_cond is never None here: _build_global_energy_token
        # short-circuits to the unconditional (no-token) path before calling
        # this. We intentionally do NOT fall back to the running mean -- a
        # missing condition means "no energy token at all", not "average energy".
        if not torch.is_tensor(raw_global_energy_cond):
            raw_global_energy_cond = torch.as_tensor(raw_global_energy_cond)
        raw_global_energy_cond = raw_global_energy_cond.to(device=device, dtype=dtype)
        if raw_global_energy_cond.dim() == 1:
            raw_global_energy_cond = raw_global_energy_cond.unsqueeze(0)
        elif raw_global_energy_cond.dim() != 2:
            raise ValueError(
                "global_energy_cond must have shape (1,) or (B, 1), got "
                f"{tuple(raw_global_energy_cond.shape)}"
            )
        if raw_global_energy_cond.shape[1] != 1:
            raise ValueError(
                "global_energy_cond must provide [energy], got "
                f"shape {tuple(raw_global_energy_cond.shape)}"
            )
        if raw_global_energy_cond.shape[0] == 1 and batch_size != 1:
            raw_global_energy_cond = raw_global_energy_cond.expand(batch_size, -1)
        elif raw_global_energy_cond.shape[0] != batch_size:
            raise ValueError(
                "global_energy_cond batch dimension must match the motion batch size, got "
                f"{raw_global_energy_cond.shape[0]} for batch {batch_size}"
            )
        # Finiteness is a cheap sanity guard, but `.all()` in a python `if` is a
        # data-dependent value that forces a torch.compile graph break every
        # step. Skip it under compilation (eager eval/inference still validates;
        # a non-finite condition would surface as a NaN loss anyway).
        if not torch.compiler.is_compiling() and not torch.isfinite(raw_global_energy_cond).all():
            raise ValueError("global_energy_cond must be finite")
        return raw_global_energy_cond

    def _build_global_energy_token(self, raw_global_energy_cond, batch_size, device, dtype, energy_active=None):
        if not self.global_energy_cond or self.global_energy_projection is None:
            return None
        if raw_global_energy_cond is None:
            # True unconditional path: emit no energy token at all, so the FiLM
            # sublayer is bypassed downstream (byte-identical to a model built
            # with global_energy_cond=False). Hit at inference when
            # --global_energy is omitted. Nothing is observed, so running stats
            # are left untouched.
            return None

        raw_global_energy_cond = self._coerce_global_energy_condition(
            raw_global_energy_cond,
            batch_size,
            device,
            dtype,
        )
        if self.training:
            self._update_global_energy_running_stats(raw_global_energy_cond, energy_active)
        running_mean = self.global_energy_running_mean.to(device=device, dtype=dtype)
        running_std = torch.sqrt(self.global_energy_running_var.to(device=device, dtype=dtype).clamp_min(1e-6))
        normalized_global_energy = (raw_global_energy_cond - running_mean.unsqueeze(0)) / running_std.unsqueeze(0)
        return self.global_energy_projection(normalized_global_energy)

    def _coerce_energy_active(self, raw_energy_active, batch_size, device):
        # Per-sample CFG mask: True == conditional (apply FiLM), False == this
        # sample is unconditional this step and must bypass the energy sublayer
        # entirely. None means "all samples conditional" (e.g. inference with an
        # explicit --global_energy, where no per-sample drop occurs).
        if raw_energy_active is None:
            return None
        energy_active = torch.as_tensor(raw_energy_active, device=device, dtype=torch.bool).reshape(-1)
        if energy_active.numel() == 1 and batch_size != 1:
            energy_active = energy_active.expand(batch_size)
        elif energy_active.numel() != batch_size:
            raise ValueError(
                "global_energy_active batch dimension must match the motion batch size, got "
                f"{energy_active.numel()} for batch {batch_size}"
            )
        return energy_active

    def _coerce_loop_condition(self, raw_loop_cond, batch_size, device, dtype):
        if raw_loop_cond is None:
            raw_loop_cond = torch.zeros(batch_size, device=device, dtype=dtype)
        elif not torch.is_tensor(raw_loop_cond):
            raw_loop_cond = torch.as_tensor(raw_loop_cond, device=device)
        raw_loop_cond = raw_loop_cond.to(device=device)
        if raw_loop_cond.dim() == 0:
            raw_loop_cond = raw_loop_cond.reshape(1)
        raw_loop_cond = raw_loop_cond.reshape(-1)
        if raw_loop_cond.numel() == 1 and batch_size != 1:
            raw_loop_cond = raw_loop_cond.expand(batch_size)
        elif raw_loop_cond.numel() != batch_size:
            raise ValueError(
                "is_loop batch dimension must match the motion batch size, got "
                f"{raw_loop_cond.numel()} for batch {batch_size}"
            )
        return raw_loop_cond.to(dtype=dtype).view(batch_size, 1)

    def _coerce_playspeed_cond(self, raw_playspeed_cond, batch_size, device, dtype):
        if raw_playspeed_cond is None:
            raw_playspeed_cond = torch.ones(batch_size, device=device, dtype=dtype)
        elif not torch.is_tensor(raw_playspeed_cond):
            raw_playspeed_cond = torch.as_tensor(raw_playspeed_cond, device=device)
        raw_playspeed_cond = raw_playspeed_cond.to(device=device)
        if raw_playspeed_cond.dim() == 0:
            raw_playspeed_cond = raw_playspeed_cond.reshape(1)
        raw_playspeed_cond = raw_playspeed_cond.reshape(-1)
        if raw_playspeed_cond.numel() == 1 and batch_size != 1:
            raw_playspeed_cond = raw_playspeed_cond.expand(batch_size)
        elif raw_playspeed_cond.numel() != batch_size:
            raise ValueError(
                "playspeed_cond batch dimension must match the motion batch size, got "
                f"{raw_playspeed_cond.numel()} for batch {batch_size}"
            )
        # See _coerce_global_energy_condition: skip the data-dependent finiteness
        # guard under torch.compile to avoid a per-step graph break.
        if not torch.compiler.is_compiling() and not torch.isfinite(raw_playspeed_cond).all():
            raise ValueError("playspeed_cond must be finite")
        return raw_playspeed_cond.to(dtype=dtype).view(batch_size, 1)

    def _build_action_tag_multihot(self, raw_action_tags, batch_size, device, dtype):
        """Multi-hot encode per-sample action tags over the canonical vocabulary.

        ``raw_action_tags`` is the ``y['action_tags']`` entry: a length-B list
        where each element is a list/set/str of normalized tag strings (or
        ``None``). Unknown tags and ``None`` rows yield all-zero rows.
        """
        multihot = torch.zeros(batch_size, len(self.action_tag_vocab), device=device, dtype=dtype)
        if raw_action_tags is None:
            return multihot
        for i in range(min(batch_size, len(raw_action_tags))):
            tags = raw_action_tags[i]
            if tags is None:
                continue
            if isinstance(tags, str):
                tags = [tags]
            for tag in tags:
                idx = self.action_tag_to_index.get(str(tag).strip().lower())
                if idx is not None:
                    multihot[i, idx] = 1.0
        return multihot

    def _coerce_action_tag_multihot(self, raw_action_tag_multihot, batch_size, device, dtype):
        """Normalize pre-batched action-tag multi-hot tensors to ``[B, V]``.

        The data loader can precompute this tensor so torch.compile sees a
        stable tensor input instead of a Python ``list[list[str]]`` whose
        structure/content varies across batches and triggers recompiles.
        """
        if raw_action_tag_multihot is None:
            return None
        action_tag_multihot = torch.as_tensor(raw_action_tag_multihot, device=device, dtype=dtype)
        if action_tag_multihot.dim() == 1:
            action_tag_multihot = action_tag_multihot.unsqueeze(0)
        elif action_tag_multihot.dim() != 2:
            raise ValueError(
                "action_tag_multihot must have shape (V,) or (B, V), got "
                f"{tuple(action_tag_multihot.shape)}"
            )
        expected_vocab_size = len(self.action_tag_vocab)
        if action_tag_multihot.shape[1] != expected_vocab_size:
            raise ValueError(
                "action_tag_multihot vocab dimension must match the canonical action-tag vocabulary, got "
                f"{action_tag_multihot.shape[1]} for vocab size {expected_vocab_size}"
            )
        if action_tag_multihot.shape[0] == 1 and batch_size != 1:
            action_tag_multihot = action_tag_multihot.expand(batch_size, -1)
        elif action_tag_multihot.shape[0] != batch_size:
            raise ValueError(
                "action_tag_multihot batch dimension must match the motion batch size, got "
                f"{action_tag_multihot.shape[0]} for batch {batch_size}"
            )
        return action_tag_multihot

    def _resolve_action_tag_active(self, raw_action_tag_active, batch_size, device):
        """Per-sample CFG mask for action tags (True == conditional).

        An explicit ``y['action_tag_active']`` wins (used at inference to force
        the unconditional branch). Otherwise, training samples a Bernoulli keep
        per sample so the condition is hard-dropped with probability
        ``action_tag_cfg_drop_prob``; eval/inference keeps every sample.
        """
        if raw_action_tag_active is not None:
            active = torch.as_tensor(raw_action_tag_active, device=device, dtype=torch.bool).reshape(-1)
            if active.numel() == 1 and batch_size != 1:
                active = active.expand(batch_size)
            elif active.numel() != batch_size:
                raise ValueError(
                    "action_tag_active batch dimension must match the motion batch size, got "
                    f"{active.numel()} for batch {batch_size}"
                )
            return active
        if self.training and self.action_tag_cfg_drop_prob > 0.0:
            return torch.rand(batch_size, device=device) >= self.action_tag_cfg_drop_prob
        return torch.ones(batch_size, device=device, dtype=torch.bool)

    def _build_action_tag_token(self, y, batch_size, device, dtype):
        if not self.action_tag_cond or self.action_tag_projection is None:
            return None
        action_tag_multihot = self._coerce_action_tag_multihot(
            y.get('action_tag_multihot'), batch_size, device, dtype
        )
        if action_tag_multihot is None:
            action_tag_multihot = self._build_action_tag_multihot(
                y.get('action_tags'), batch_size, device, dtype
            )
        action_tag_emb = self.action_tag_projection(action_tag_multihot)
        action_tag_active = self._resolve_action_tag_active(
            y.get('action_tag_active'), batch_size, device
        )
        # A row with no recognized tags (missing/None/empty/all-out-of-vocab)
        # carries no action information. Route it to the learned null embedding
        # rather than feeding the all-zero multi-hot through the projection,
        # whose bias output is an untrained region (every training clip carries
        # >=1 tag, and hard-dropped rows already go to null_emb). This unifies
        # "no tags" with the hard-dropped state, so omitting action_tags at
        # inference yields the learned unconditional mode automatically.
        has_tags = action_tag_multihot.any(dim=1)
        action_tag_active = action_tag_active & has_tags
        null_emb = self.action_tag_null_emb.to(device=device, dtype=dtype)
        return torch.where(
            action_tag_active.view(batch_size, 1),
            action_tag_emb,
            null_emb.unsqueeze(0).expand(batch_size, -1),
        )

    def sample_subtree_joint_mask_train(self, y, njoints, device):
        """Select subtrees of joints to perturb during training.

        Returns a bool tensor of shape ``[B, njoints]`` (True = joint selected)
        or ``None`` if no joint was selected, or if not in training mode, or
        if ``joint_mask_prob == 0`` or ``joint_mask_budget == 0`` -- so
        eval-mode loss reports a clean diffusion objective.

        Called from ``GaussianDiffusion.training_losses`` AFTER ``q_sample``
        to decide which joints' x_t slice should be re-noised with an
        independent random timestep and fresh noise, so that those joints'
        noise level disagrees with the rest of the batch sample. This trains
        the cross-joint pathway to denoise robustly against per-joint
        timestep mismatch -- the regime inpaint clamping produces at
        inference. The model's forward itself stays vanilla.
        """
        if (not self.training) or self.joint_mask_prob <= 0.0 or self.joint_mask_budget <= 0.0:
            return None
        n_joints_cpu = torch.as_tensor(y['n_joints'], device='cpu', dtype=torch.int64).reshape(-1)
        return self._sample_subtree_joint_mask(y, n_joints_cpu, njoints, device)

    def _sample_subtree_joint_mask(self, y, n_joints, njoints, device):
        parents_batch = y.get('parents')
        if parents_batch is None:
            return None
        candidate_roots_batch = y.get('joint_mask_candidate_roots')
        if torch.is_tensor(n_joints):
            n_joints_np = n_joints.detach().to(device='cpu', dtype=torch.int64).numpy()
        else:
            n_joints_np = np.asarray(n_joints, dtype=np.int64)

        if torch.is_tensor(parents_batch):
            parents_batch_np = parents_batch.detach().to(device='cpu', dtype=torch.int64).numpy()
        else:
            parents_batch_np = [np.asarray(parents, dtype=np.int64) for parents in parents_batch]

        if candidate_roots_batch is None:
            candidate_roots_np = None
        elif torch.is_tensor(candidate_roots_batch):
            candidate_roots_np = candidate_roots_batch.detach().to(device='cpu').numpy()
        else:
            candidate_roots_np = np.asarray(candidate_roots_batch, dtype=np.bool_)

        subtree_joint_mask_np = sample_subtree_joint_mask_batch(
            parents_batch=parents_batch_np,
            candidate_root_mask_batch=candidate_roots_np,
            n_joints=n_joints_np,
            max_joints=njoints,
            joint_mask_prob=self.joint_mask_prob,
            joint_mask_budget=self.joint_mask_budget,
            rng=np.random,
        )
        if subtree_joint_mask_np is None:
            return None
        return torch.from_numpy(subtree_joint_mask_np).to(device=device)

    def sample_temporal_span_mask_train(self, y, njoints, nframes, device):
        """Select contiguous temporal spans to perturb during training.

        Returns a bool tensor of shape ``[B, njoints, nframes]`` (True =
        re-noise that joint/frame cell) or ``None`` if no span was selected,
        if not in training mode, or if ``temporal_span_mask_prob == 0``.

        Full-clip spans are not supported: the sampled span always leaves at
        least one frame at the start or end unmasked so seam detection never
        triggers on a clip-span boundary.  ``temporal_span_mask_max_frames``
        is therefore capped at ``min(config_max, nframes - 1)`` at runtime.
        Every real joint in the sample shares the same contiguous masked frame
        interval, while padded joints stay False throughout.
        """
        if (not self.training) or self.temporal_span_mask_prob <= 0.0:
            return None

        n_joints_batch = y.get('n_joints_cpu', y.get('n_joints'))
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
        # randint(min_span, max_span + 1) per sample
        span_length = torch.randint(min_span, max_span + 1, (batch_size,), device=device)  # [B]
        start_hi = (nframes - span_length).clamp(min=0)                           # [B]
        # randint(0, start_hi + 1): scale [0,1) by (start_hi+1) then floor,
        # clamping guards the rare rand()==~1.0 rounding to start_hi+1.
        span_start = (torch.rand(batch_size, device=device) * (start_hi + 1).float()).long()
        span_start = torch.minimum(span_start, start_hi)                          # [B]

        frame_idx = torch.arange(nframes, device=device)                         # [T]
        frame_mask = (frame_idx[None, :] >= span_start[:, None]) & \
            (frame_idx[None, :] < (span_start + span_length)[:, None])           # [B, T]
        joint_idx = torch.arange(njoints, device=device)                         # [J]
        joint_mask = joint_idx[None, :] < valid_joints[:, None]                   # [B, J]

        mask = (
            active[:, None, None]
            & joint_mask[:, :, None]
            & frame_mask[:, None, :]
        )                                                                         # [B, J, T]
        if not bool(mask.any()):
            return None
        return mask

    def _build_species_token(self, y, batch_size, device, dtype):
        """Project the per-species T5 descriptor into a [B, latent_dim] token
        added to the timestep embedding. Returns None when species
        conditioning is disabled."""
        if self.species_projection is None:
            return None
        raw_species_emb = y.get('species_emb') if y is not None else None
        if raw_species_emb is None:
            raise ValueError(
                "species_cond is enabled but y['species_emb'] is missing. "
                "Regenerate cond.npy so each species carries 'species_emb'."
            )
        species_emb = torch.as_tensor(raw_species_emb, device=device, dtype=dtype)
        if species_emb.dim() == 1:
            species_emb = species_emb.unsqueeze(0)
        if species_emb.shape[0] == 1 and batch_size != 1:
            species_emb = species_emb.expand(batch_size, -1)
        elif species_emb.shape[0] != batch_size:
            raise ValueError(
                "species_emb batch dimension must match the motion batch size, got "
                f"{species_emb.shape[0]} for batch {batch_size}"
            )
        return self.species_projection(species_emb)

    def forward(self, x, timesteps, y=None, train_step=None, **unused_kwargs):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """

        joints_padding_mask = y['joints_padding_mask'].to(x.device)
        temp_mask = y['mask'].to(x.device)
        tpos_first_frame = y['tpos_first_frame'].to(x.device).unsqueeze(0)

        bs, njoints, nfeats, nframes = x.shape
        n_joints = torch.as_tensor(y['n_joints'], device=x.device).reshape(-1)
        joint_key_padding_mask = self._build_joint_key_padding_mask(njoints, n_joints, x.device)
        global_energy_condition = None
        # joint-mask subtree perturbation is applied OUTSIDE this
        # forward, in diffusion.training_losses, by re-noising the selected
        # joints' x_t with q_sample(x_0, t_random, fresh_noise). The model
        # itself stays vanilla -- selected joints DO NOT enter the
        # key-padding masks and continue to participate in attention normally,
        # just with mismatched noise levels, which trains the
        # cross-joint pathway to denoise robustly against per-joint timestep
        # disagreement (matching inpaint clamp behavior at inference).
        timesteps_emb = create_sin_embedding(timesteps.view(1, -1, 1), self.latent_dim)[0]
        playspeed_condition = self._coerce_playspeed_cond(
            y.get('playspeed_cond'),
            batch_size=bs,
            device=x.device,
            dtype=x.dtype,
        )
        timesteps_emb = timesteps_emb + self.playspeed_projection(playspeed_condition)
        if self.loop_cond_prob > 0.0 and self.loop_condition_projection is not None:
            loop_condition = self._coerce_loop_condition(
                y.get('is_loop'),
                batch_size=bs,
                device=x.device,
                dtype=x.dtype,
            )
            timesteps_emb = timesteps_emb + self.loop_condition_projection(loop_condition)
        action_tag_token = self._build_action_tag_token(y, bs, x.device, x.dtype)
        if action_tag_token is not None:
            timesteps_emb = timesteps_emb + action_tag_token
        species_token = self._build_species_token(y, bs, x.device, x.dtype)
        if species_token is not None:
            timesteps_emb = timesteps_emb + species_token

        loop_phase_mask = None
        raw_loop_phase_mask = y.get('is_loop')
        if raw_loop_phase_mask is not None:
            loop_phase_mask = torch.as_tensor(raw_loop_phase_mask, device=x.device, dtype=torch.bool).reshape(-1)
            if loop_phase_mask.numel() == 1 and bs != 1:
                loop_phase_mask = loop_phase_mask.expand(bs)
            elif loop_phase_mask.numel() != bs:
                raise ValueError(
                    "is_loop batch dimension must match the motion batch size, got "
                    f"{loop_phase_mask.numel()} for batch {bs}"
                )
            raw_loop_full_cycle = y.get('loop_full_cycle')
            if raw_loop_full_cycle is not None:
                loop_full_cycle_mask = torch.as_tensor(raw_loop_full_cycle, device=x.device, dtype=torch.bool).reshape(-1)
                if loop_full_cycle_mask.numel() == 1 and bs != 1:
                    loop_full_cycle_mask = loop_full_cycle_mask.expand(bs)
                elif loop_full_cycle_mask.numel() != bs:
                    raise ValueError(
                        "loop_full_cycle batch dimension must match the motion batch size, got "
                        f"{loop_full_cycle_mask.numel()} for batch {bs}"
                    )
                loop_phase_mask = loop_phase_mask & loop_full_cycle_mask

        x = self.input_process(x, tpos_first_frame, y['joints_names_embs']) # applies linear layer on each frame to convert it to latent dim
        spatial_mask = (1.0 - joints_padding_mask[:, 0, 0, 1:, 1:].float()) * -1e4
        spatial_mask = spatial_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        temporal_template = (1.0 - temp_mask.reshape(bs, -1, nframes + 1, nframes + 1)[:, :1].float()) * -1e4
        temporal_template = temporal_template.expand(-1, self.num_heads, -1, -1)
        temporal_mask = temporal_template.unsqueeze(1).expand(-1, njoints, -1, -1, -1).reshape(-1, nframes + 1, nframes + 1)

        # Cross-limb temporal pathway needs (1) the windowed temporal mask
        # WITHOUT the per-joint repeat -> (bs*H, T, T), which the block expands
        # per-latent itself, and (2) a (bs, njoints) bool key-padding mask
        # (True == padded) derived from the real joint count.
        if self.cross_limb:
            assert 'n_joints' in y, "cross_limb requires y['n_joints'] in the batch"
            temporal_template = temporal_template.reshape(-1, nframes + 1, nframes + 1)
        else:
            temporal_template = None

        global_energy_active = None
        if self.global_energy_cond:
            global_energy_active = self._coerce_energy_active(
                y.get('global_energy_active'),
                batch_size=bs,
                device=x.device,
            )
            global_energy_condition = self._build_global_energy_token(
                y.get('global_energy_cond'),
                batch_size=bs,
                device=x.device,
                dtype=x.dtype,
                energy_active=global_energy_active,
            )

        cross_limb_unreliable_mask = None
        if self.cross_limb:
            raw_cross_limb_unreliable_mask = y.get('cross_limb_unreliable_mask')
            if raw_cross_limb_unreliable_mask is not None:
                cross_limb_unreliable_mask = raw_cross_limb_unreliable_mask.to(device=x.device, dtype=x.dtype)
                raw_expected_shape = (bs, nframes, njoints)
                prepared_expected_shape = (nframes + 1, bs, njoints)
                if cross_limb_unreliable_mask.shape == raw_expected_shape:
                    reliable_tpose = torch.zeros((bs, 1, njoints), device=x.device, dtype=x.dtype)
                    cross_limb_unreliable_mask = torch.cat([reliable_tpose, cross_limb_unreliable_mask], dim=1)
                    cross_limb_unreliable_mask = cross_limb_unreliable_mask.transpose(0, 1).contiguous()
                elif cross_limb_unreliable_mask.shape != prepared_expected_shape:
                    raise ValueError(
                        "y['cross_limb_unreliable_mask'] must have shape "
                        f"{raw_expected_shape} or {prepared_expected_shape}, got "
                        f"{tuple(cross_limb_unreliable_mask.shape)}"
                    )

        loop_phase_lengths = y.get('loop_phase_lengths', y.get('lengths'))

        group_ids = None
        if self.morphology_expert:
            # Prefer a precomputed tensor (compile-friendly); otherwise resolve
            # from the per-sample object_type strings via the fixed registry.
            raw_group_ids = y.get('group_ids')
            if raw_group_ids is not None:
                group_ids = torch.as_tensor(raw_group_ids, device=x.device, dtype=torch.long).reshape(-1)
            else:
                object_types = y.get('object_type')
                if object_types is None:
                    raise ValueError(
                        "morphology_expert is enabled but y['object_type'] is missing; "
                        "cannot route to a morphology group."
                    )
                group_ids = object_types_to_group_id_tensor(
                    object_types, self.object_type_to_group_id, x.device
                )
            if group_ids.numel() == 1 and bs != 1:
                group_ids = group_ids.expand(bs)
            elif group_ids.numel() != bs:
                raise ValueError(
                    "group_ids batch dimension must match the motion batch size, got "
                    f"{group_ids.numel()} for batch {bs}"
                )

        output = self.seqTransDecoder(
            tgt=x,
            timesteps_embs=timesteps_emb,
            memory=None,
            spatial_mask=spatial_mask,
            temporal_mask=temporal_mask,
            tgt_key_padding_mask=joint_key_padding_mask,
            y=y,
            global_energy_condition=global_energy_condition,
            global_energy_active=global_energy_active,
            temporal_template=temporal_template,
            cross_limb_unreliable_mask=cross_limb_unreliable_mask,
            loop_phase_mask=loop_phase_mask,
            lengths=loop_phase_lengths,
            group_ids=group_ids,
        )
        output = self.output_process(output) # Applies linear layer on each frame to convert it back to feature len dim
        return output


    def _apply(self, fn):
        super()._apply(fn)


    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)

# in the case of GMDM, the input process is as follows: 
# embed each joint of each frame of each motion in batch by the same MLP, separately ! 
class InputProcess(nn.Module):
    def __init__(self, input_feats, root_input_feats, latent_dim, t5_output_dim, dropout_prob=0):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.root_input_feats = root_input_feats
        self.root_embedding = nn.Linear(self.root_input_feats, self.latent_dim)
        self.tpos_root_embedding = nn.Linear(self.root_input_feats, self.latent_dim)
        self.joint_embedding = nn.Linear(self.input_feats, self.latent_dim)
        self.tpos_joint_embedding = nn.Linear(self.input_feats, self.latent_dim)
        self.joints_names_dropout = nn.Dropout(p=dropout_prob)
        self.text_embedding = nn.Linear(t5_output_dim, self.latent_dim)
    def forward(self, x, tpos_first_frame, joints_embedded_names):
        # x.shape = [batch_size, joints, 13, frames]
        x = x.permute(3, 0, 1, 2) # [frames, batch_size, n_joints, features_len]
        tpos_all_joints_except_root = self.tpos_joint_embedding(tpos_first_frame[:, :, 1:])
        tpos_root_data = self.tpos_root_embedding(tpos_first_frame[:, :, 0:1])
        all_joints_except_root = self.joint_embedding(x[:, :, 1:])
        root_data = self.root_embedding(x[:, :, 0:1])
        tpos_embedded = torch.cat([tpos_root_data, tpos_all_joints_except_root], dim=2)
        x_embedded = torch.cat([root_data, all_joints_except_root], dim=2)
        x = torch.cat([tpos_embedded, x_embedded], dim=0)
        joints_embedded_names = self.text_embedding(self.joints_names_dropout(joints_embedded_names.to(x.device)))
        x = x + joints_embedded_names[None, ...]# [frames, batch_size, n_joints, d]
        positions = torch.arange(x.shape[0], device=x.device).view(1, -1, 1).repeat(x.shape[1], 1, 1)
        pos_emb = create_sin_embedding(positions, self.latent_dim)[0]
        return x + pos_emb.unsqueeze(1).unsqueeze(1)


class GlobalEnergyExtractor:
    """Phase-invariant clip-level global energy statistics.

    Standalone helper retained after the reference/controlnet conditioning
    path was removed; the ``global_energy_cond`` feature consumes these
    statistics to derive a clip-level [energy_mean, energy_std] condition.
    Assumes the canonical 13-D motion schema: pos[0:3], rot6d[3:9],
    vel[9:12], contact[12].
    """

    @staticmethod
    def _masked_mean_and_std(values, weights, eps=1e-6):
        weights_sum = weights.sum(dim=2, keepdim=True).clamp_min(eps)
        mean = (values * weights.unsqueeze(-1)).sum(dim=2) / weights_sum
        second_moment = ((values * values) * weights.unsqueeze(-1)).sum(dim=2) / weights_sum
        variance = (second_moment - mean * mean).clamp_min(0.0)
        std = torch.sqrt(variance + eps)
        return mean, std

    @staticmethod
    def _coerce_global_energy_playspeed_cond(raw_playspeed_cond, batch_size, device, dtype):
        if raw_playspeed_cond is None:
            return None
        if not torch.is_tensor(raw_playspeed_cond):
            raw_playspeed_cond = torch.as_tensor(raw_playspeed_cond, device=device)
        raw_playspeed_cond = raw_playspeed_cond.to(device=device)
        if raw_playspeed_cond.dim() == 0:
            raw_playspeed_cond = raw_playspeed_cond.reshape(1)
        raw_playspeed_cond = raw_playspeed_cond.reshape(-1)
        if raw_playspeed_cond.numel() == 1 and batch_size != 1:
            raw_playspeed_cond = raw_playspeed_cond.expand(batch_size)
        elif raw_playspeed_cond.numel() != batch_size:
            raise ValueError(
                "playspeed_cond batch dimension must match the motion batch size, got "
                f"{raw_playspeed_cond.numel()} for batch {batch_size}"
            )
        if not torch.isfinite(raw_playspeed_cond).all():
            raise ValueError("playspeed_cond must be finite")
        if bool((raw_playspeed_cond <= 0).any()):
            raise ValueError("playspeed_cond must be positive")
        return raw_playspeed_cond.to(dtype=dtype).view(batch_size, 1, 1, 1)

    @staticmethod
    def _resample_motion_time_axis(motion, target_frame_count):
        """Linear resample along the time axis.  Input: (J, F, T)."""
        source_frame_count = int(motion.shape[-1])
        target_frame_count = int(target_frame_count)
        if target_frame_count <= 0:
            raise ValueError(f"target_frame_count must be positive, got {target_frame_count}")
        if source_frame_count == target_frame_count:
            return motion

        # (J, F, T) → (T, J, F)
        motion_tjf = motion.permute(2, 0, 1)
        src = torch.linspace(
            0.0,
            float(source_frame_count - 1),
            target_frame_count,
            device=motion.device,
            dtype=motion.dtype,
        )
        lo = torch.floor(src).to(dtype=torch.long).clamp(0, source_frame_count - 1)
        hi = torch.minimum(lo + 1, torch.full_like(lo, source_frame_count - 1))
        w = (src - torch.floor(src)).view(-1, 1, 1).to(dtype=motion_tjf.dtype)
        resampled = motion_tjf.index_select(0, lo) * (1.0 - w) + motion_tjf.index_select(0, hi) * w

        if motion.shape[1] >= 13:
            nearest = torch.round(src).to(dtype=torch.long).clamp(0, source_frame_count - 1)
            resampled[..., 12] = (motion_tjf.index_select(0, nearest)[..., 12] >= 0.5).to(dtype=resampled.dtype)

        return resampled.permute(1, 2, 0).contiguous()

    @staticmethod
    def _build_joint_motion_frame_features(vel, rot_delta_norm, contact):
        """Return phase-invariant per-frame energy cues.

        Drops velocity sign by using ``vel_norm`` so the energy statistic is
        driven by stable motion magnitude rather than instantaneous swing
        direction.
        """
        vel_norm = torch.linalg.norm(vel, dim=-1, keepdim=True)
        energy = torch.sqrt(vel_norm.square() + rot_delta_norm.square() + 1e-6)
        joint_motion_frame_features = torch.cat([vel_norm, rot_delta_norm, energy, contact], dim=-1)
        return joint_motion_frame_features, vel_norm, energy

    @classmethod
    def _extract_joint_motion_inputs(cls, motion, n_joints):
        if motion.dim() != 4:
            raise ValueError(f"motion must have shape (B, J, F, T), got {tuple(motion.shape)}")

        batch_size, max_joints, feature_dim, frame_count = motion.shape
        if feature_dim < 13:
            raise ValueError(
                "global energy extraction expects at least the 13-dim feature schema "
                f"[pos(3), rot6d(6), vel(3), contact(1)], got feature_dim={feature_dim}"
            )

        device = motion.device
        dtype = motion.dtype
        n_joints = torch.as_tensor(n_joints, device=device, dtype=torch.long).reshape(batch_size)
        if bool(((n_joints < 0) | (n_joints > max_joints)).any()):
            raise ValueError(
                f"n_joints must be in [0, {max_joints}], got {n_joints.tolist()}"
            )

        valid_joints = torch.arange(max_joints, device=device).unsqueeze(0) < n_joints.unsqueeze(1)
        motion_btjf = motion.permute(0, 3, 1, 2)
        vel = motion_btjf[..., 9:12]
        rot = motion_btjf[..., 3:9]
        rot_delta = torch.zeros_like(rot)
        if frame_count > 1:
            rot_delta[:, 1:] = rot[:, 1:] - rot[:, :-1]
        rot_delta_norm = torch.linalg.norm(rot_delta, dim=-1, keepdim=True)
        contact = motion_btjf[..., 12:13]
        joint_motion_frame_features, vel_norm, energy = cls._build_joint_motion_frame_features(
            vel,
            rot_delta_norm,
            contact,
        )
        joint_motion_frame_features = joint_motion_frame_features * valid_joints[:, None, :, None].to(dtype)
        return {
            'dtype': dtype,
            'frame_count': frame_count,
            'valid_joints': valid_joints,
            'joint_motion_frame_features': joint_motion_frame_features,
        }

    @classmethod
    def compute_global_energy_condition(cls, motion, n_joints, playspeed_cond=None):
        if playspeed_cond is not None:
            if motion.dim() != 4:
                raise ValueError(f"motion must have shape (B, J, F, T), got {tuple(motion.shape)}")
            batch_size = motion.shape[0]
            source_length_ratio = cls._coerce_global_energy_playspeed_cond(
                playspeed_cond,
                batch_size,
                motion.device,
                motion.dtype,
            ).reshape(batch_size)
            inferred_source_frames = torch.round(source_length_ratio * float(motion.shape[-1])).to(dtype=torch.long).clamp_min(1)
            if bool((inferred_source_frames != motion.shape[-1]).any()):
                # Batched resample + pad-to-max + masked statistics.
                # Each sample may have a different target frame count, so we
                # resample independently, pad to the maximum, and use a frame
                # mask so the weighted mean/std ignores padding.
                #
                # Because _extract_joint_motion_inputs computes rot_delta as
                # rot[:, 1:] - rot[:, :-1], the boundary between the last valid
                # frame and the first zero-padded frame produces a spurious large
                # delta. We mask out that boundary frame as well.
                max_target = int(inferred_source_frames.max().item())
                original_T = motion.shape[-1]

                # Resample + pad in a single pass: resample to each sample's
                # target frame count, replicate the last frame into the padding
                # region (so rot_delta at boundary is zero), and build a frame
                # mask that zeros padding during statistics.
                padded_list = []
                frame_mask_list = []
                for i in range(batch_size):
                    t_i = int(inferred_source_frames[i].item())
                    s = (
                        cls._resample_motion_time_axis(motion[i], t_i)
                        if t_i != original_T
                        else motion[i]
                    )
                    pad = max_target - t_i
                    if pad > 0:
                        last_frame = s[:, :, -1:].expand(-1, -1, pad)
                        s = torch.cat([s, last_frame], dim=2)
                        fm = torch.cat([
                            torch.ones(t_i, device=s.device, dtype=s.dtype),
                            torch.zeros(pad, device=s.device, dtype=s.dtype),
                        ])
                    else:
                        fm = torch.ones(max_target, device=s.device, dtype=s.dtype)
                    padded_list.append(s)
                    frame_mask_list.append(fm)

                motion_padded = torch.stack(padded_list, dim=0)  # (B, J, F, max_T)
                frame_mask = torch.stack(frame_mask_list, dim=0)  # (B, max_T)

                # Compute energy statistics on padded motion with frame mask.
                motion_inputs = cls._extract_joint_motion_inputs(motion_padded, n_joints)
                dtype = motion_inputs['dtype']
                valid_joints = motion_inputs['valid_joints']  # (B, J)
                joint_motion_frame_features = motion_inputs['joint_motion_frame_features']  # (B, max_T, J, 4)

                # Step 1: Apply joint mask via _masked_mean_and_std (dim=2 = joint).
                # Result: (B, max_T, 4) — per-frame energy features averaged over joints.
                global_mean, _ = cls._masked_mean_and_std(
                    joint_motion_frame_features,
                    valid_joints.unsqueeze(1).expand(-1, max_target, -1).to(dtype),
                )

                # Step 2: Apply frame mask via weighted mean over time (dim=1 = time).
                # frame_mask: (B, max_T) — 1 for valid frames, 0 for padding.
                energy_profile = global_mean[..., 2:3]  # (B, max_T, 1)
                frame_weights = frame_mask.unsqueeze(-1).to(dtype)  # (B, max_T, 1)
                frame_weights_sum = frame_weights.sum(dim=1).clamp_min(1e-6)  # (B, 1)
                result = (energy_profile * frame_weights).sum(dim=1) / frame_weights_sum  # (B, 1)
                return result

        motion_inputs = cls._extract_joint_motion_inputs(motion, n_joints)
        dtype = motion_inputs['dtype']
        frame_count = motion_inputs['frame_count']
        valid_joints = motion_inputs['valid_joints']
        joint_motion_frame_features = motion_inputs['joint_motion_frame_features']
        global_mean, _ = cls._masked_mean_and_std(
            joint_motion_frame_features,
            valid_joints[:, None, :].expand(-1, frame_count, -1).to(dtype),
        )
        global_energy_profile = global_mean[..., 2:3]
        return global_energy_profile.mean(dim=1)

class OutputProcess(nn.Module):
    def __init__(self, feature_len, root_feature_len, max_joints, latent_dim):
        super().__init__()
        self.feature_len = feature_len
        self.max_joints = max_joints
        self.latent_dim = latent_dim
        self.root_feature_len = root_feature_len
        self.root_dembedding = nn.Linear(self.latent_dim, self.root_feature_len)
        self.joint_dembedding = nn.Linear(self.latent_dim, self.feature_len)

    def forward(self, output):
        # output shape [frames, batch_size, joints, latent_dim]
        root_data = self.root_dembedding(output[:, :, 0])
        all_joints = self.joint_dembedding(output[:, :, 1:])
        output = torch.cat([root_data.unsqueeze(2), all_joints], dim=-2)
        output = output.permute(1, 2, 3, 0)[..., 1:]  # [bs, njoints, nfeats, nframes]
        return output


