import torch
import torch.nn as nn
import numpy as np
from model.motion_transformer import GraphMotionDecoderLayer, GraphMotionDecoder
from model.joint_mask_utils import sample_subtree_joint_mask_batch
from data_loaders.truebones.truebones_utils.action_label_conditioning_contract import (
    ACTION_LABEL_SLOTS,
    ROLE_B_ARTIFACT_SCHEMA_VERSION,
    ROLE_HEAD_1,
    ActionConditioningError,
    slot_source_rank_report,
    validate_action_conditioning_metadata,
    validate_role_b_payload,
    word_table_sha256,
)
from data_loaders.truebones.truebones_utils.motion_labels import (
    ACTION_LABEL_MAX_WORDS,
    CONTROLLED_VOCAB,
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
        self.species_cond=bool(kargs.get('species_cond', False))
        self.species_cfg_drop_prob=float(kargs.get('species_cfg_drop_prob', 0.15))
        if not 0.0 <= self.species_cfg_drop_prob <= 1.0:
            raise ValueError(
                f"species_cfg_drop_prob must be in [0, 1], got {self.species_cfg_drop_prob}"
            )
        self.species_joint_cond=bool(kargs.get('species_joint_cond', False))
        self.loop_cond_prob=float(kargs.get('loop_cond_prob', 1.0))
        # Action-label conditioning: a single pathway -- the frozen T5 vectors of
        # the label's WORDS, pooled into one channel per role slot (head /
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

        self.input_process = InputProcess(self.input_feats, self.root_input_feats, self.latent_dim, t5_out_dim, dropout_prob=self.dropout, species_joint_cond=self.species_joint_cond)
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
        if self.action_label_cond:
            # Project the concatenated slot channels and add the result to the
            # timestep token, alongside loop / playspeed / canonical frame. One
            # Linear over the concatenation IS one Linear per channel block,
            # summed, so each channel's relative scale is learned rather than
            # being some offline budget we picked. Additive and linear is enough:
            # direction is linearly readable out of the T5 vectors. If a trained
            # model still under-follows the prompt, raise --action_label_cfg_scale
            # (no retrain needed).
            self._init_action_conditioning(kargs.get('action_conditioning'), t5_out_dim)
            self.action_label_projection = nn.Sequential(
                nn.Linear(len(ACTION_LABEL_SLOTS) * t5_out_dim, self.latent_dim),
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

        # Per-species condition: a T5-derived species descriptor that FiLM-modulates
        # the timestep embedding (which every decoder layer re-injects via
        # embed_timesteps -> deep, per-layer modulation). The head emits
        # (gamma_residual, beta); the effective scale is gamma = 1 + gamma_residual.
        # The final linear is zero-initialized so the condition starts at *identity*
        # (gamma=1, beta=0) -- a no-regret start. Unlike a zero-init *additive* token
        # (which collapses to dead when the signal is recoverable from the per-joint
        # name path -- see species_cond_no_lsimple_effect), the multiplicative form can
        # express scalings the additive joint path cannot, so the two stay orthogonal
        # even when --species_joint_cond also injects the descriptor per joint.
        # CFG-droppable: hard-dropped samples bypass to identity (NOT a running-mean
        # substitute), so the model learns a true unconditional mode for guidance.
        if self.species_cond:
            self.species_film = nn.Sequential(
                nn.Linear(t5_out_dim, self.latent_dim),
                nn.GELU(),
                nn.Linear(self.latent_dim, 2 * self.latent_dim),
            )
            nn.init.zeros_(self.species_film[-1].weight)
            nn.init.zeros_(self.species_film[-1].bias)
        else:
            self.species_film = None

        # Output-coordinate-frame condition: the per-object_subset canonical
        # (mean, std) 13-vectors the features are written in, projected and added
        # to the timestep token. These statistics define which of the seven affine
        # canonical spaces this sample lives in, and before this projection NOTHING
        # in the model read them -- the only trace was the object_subset word buried
        # inside the mean-pooled species descriptor, which --species_cfg_drop_prob
        # then dropped 15% of the time. A model that cannot tell which space it is
        # writing into has to guess the gain, and a wrong guess is a deformation,
        # not an offset.
        #
        # UNCONDITIONAL, on purpose -- there is no flag for it. Every cond.npy in
        # the canonical_motion_v3 feature space carries these two vectors and the
        # dataset refuses to load without them, so "off" would only ever mean
        # "blind to the output space", which is the defect this fixes. It is also
        # NOT CFG-droppable and not gated behind species conditioning: this is not
        # a semantic condition to guide toward or away from, it is the definition
        # of the output space. Zero-initialized final linear, so the condition
        # starts at exact identity (contributes 0 to the timestep token).
        self.canonical_frame_projection = nn.Sequential(
            nn.Linear(2 * self.feature_len, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        nn.init.zeros_(self.canonical_frame_projection[-1].weight)
        nn.init.zeros_(self.canonical_frame_projection[-1].bias)

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
                                                        cross_limb_last_n=self.cross_limb_last_n)
            
        
        self.output_process = OutputProcess(self.feature_len, self.root_input_feats, self.max_joints, self.latent_dim)

    @staticmethod
    def _build_joint_key_padding_mask(njoints, n_joints, device):
        """Return the padding-only joint key mask used by attention.

        Training-time subtree perturbation deliberately stays out of the
        attention masks. Only structurally padded joints are masked here.
        """
        return torch.arange(njoints, device=device)[None, :] >= n_joints[:, None]

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
        # `.all()` in a python `if` is a data-dependent value that forces a
        # torch.compile graph break every step, so skip the finiteness guard
        # under compilation (eager eval/inference still validates).
        if not torch.compiler.is_compiling() and not torch.isfinite(raw_playspeed_cond).all():
            raise ValueError("playspeed_cond must be finite")
        return raw_playspeed_cond.to(dtype=dtype).view(batch_size, 1)

    def _init_action_conditioning(self, bundle, t5_out_dim):
        """Freeze the word table and the role transform into model buffers.

        Persistent buffers, so a checkpoint carries the exact vectors its weights
        were fitted on and inference never has to pick a sidecar out of a data
        directory. With no bundle (the inference path) they are placeholders that
        ``load_state_dict`` overwrites; ``validate_loaded_action_conditioning``
        is what certifies what landed in them.
        """
        vocab_size = len(CONTROLLED_VOCAB)
        if bundle is None:
            table = torch.zeros(vocab_size, int(t5_out_dim))
            perm = torch.arange(int(t5_out_dim), dtype=torch.long)
            sign = torch.ones(int(t5_out_dim))
            self.action_conditioning_metadata = None
        else:
            if bundle.embedding_dim != int(t5_out_dim):
                raise ValueError(
                    f"the action word table is {bundle.embedding_dim}-dimensional but "
                    f"t5_out_dim is {t5_out_dim}. It was encoded with a different T5 "
                    "than cond.npy's joints_names_embs; rebuild it with --t5-model "
                    "matching cond.npy."
                )
            # The gate the geometry preflight cannot enforce on its own: the
            # first Linear has to be wide enough to stay injective on the direct
            # sum of the three slot source spaces, or labels that differ only in
            # slot membership can collide before any weight is trained.
            report = bundle.slot_source_rank_report(self.latent_dim)
            if not report['full_rank']:
                raise ValueError(
                    f"{bundle.source}: the frozen word table does not have full "
                    f"slot-source rank ({report['slots']}), so distinct labels are not "
                    "guaranteed to reach distinct conditions. Rebuild it and re-run "
                    "tools/evaluate_action_label_geometry.py."
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
            perm = torch.as_tensor(bundle.role_b_perm, dtype=torch.long)
            sign = torch.as_tensor(bundle.role_b_sign, dtype=torch.float32)
            self.action_conditioning_metadata = bundle.checkpoint_metadata()
        self.register_buffer('action_word_embeddings', table, persistent=True)
        self.register_buffer('action_role_b_perm', perm, persistent=True)
        self.register_buffer('action_role_b_sign', sign, persistent=True)

    def validate_loaded_action_conditioning(self, metadata, source: str):
        """Bind loaded weights to the contract they were trained under.

        Called right after ``load_state_dict``. The checkpoint's ``perm``/``sign``
        buffers are the authoritative material for THOSE weights, so they are
        re-validated as an external payload rather than assumed to equal what
        this code derives today -- and the recorded material hash is what says
        whether they are the same transform.
        """
        if not self.action_label_cond:
            return None
        validated = validate_action_conditioning_metadata(metadata, source=source)
        # The table first: "the weights loaded but the buffers did not" is the
        # failure a reader has to see, and it would otherwise surface as a role
        # hash mismatch on the untouched placeholder material.
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
        contract = validated['conditioning_contract']
        try:
            validate_role_b_payload(
                {
                    'schema_version': ROLE_B_ARTIFACT_SCHEMA_VERSION,
                    'embedding_dim': int(self.action_role_b_perm.numel()),
                    'perm': self.action_role_b_perm.detach().cpu().tolist(),
                    'sign': self.action_role_b_sign.detach().cpu().to(torch.int64).tolist(),
                    'material_sha256': contract.get('role_b_material_sha256'),
                },
                expected_dim=int(self.action_word_embeddings.shape[1]),
                source=f"{source} role transform buffers",
            )
        except ValueError as exc:
            # One error type out of the whole bind: this is a refused checkpoint
            # like every other failure here, and callers that catch
            # ActionConditioningError should not miss it because the shared
            # payload validator speaks plain ValueError.
            raise ActionConditioningError(str(exc)) from exc
        report = slot_source_rank_report(
            table,
            self.action_role_b_perm.detach().cpu().tolist(),
            self.action_role_b_sign.detach().cpu().tolist(),
            self.latent_dim,
        )
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
        table lives in exactly one place -- this model. All four fields travel
        together: a label's meaning is the (word, role, slot) set, and accepting
        a partial one would silently condition on a different label.
        """
        raw_word_ids = y.get('action_word_ids')
        if raw_word_ids is None:
            return None
        word_ids = self._coerce_action_slot_field(
            raw_word_ids, 'action_word_ids', batch_size, device, torch.long
        )
        fields = {}
        for name, dtype in (
            ('action_role_ids', torch.long),
            ('action_slot_ids', torch.long),
            ('action_word_mask', torch.bool),
        ):
            raw = y.get(name)
            if raw is None:
                raise ValueError(
                    f"y carries action_word_ids but no {name}. The word ids alone do "
                    "not say which slot a word feeds or which role it plays, so the "
                    "condition cannot be assembled from them."
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
                "action_word_ids, action_role_ids, action_slot_ids and "
                "action_word_mask must all have the same shape; "
                f"action_word_ids is {tuple(word_ids.shape)} but {mismatched}"
            )
        if not torch.compiler.is_compiling():
            vocab_size = int(self.action_word_embeddings.shape[0])
            live = fields['action_word_mask']
            if bool(((word_ids < 0) | (word_ids >= vocab_size))[live].any()):
                raise ValueError(
                    f"action_word_ids holds an id outside 0..{vocab_size - 1}; the "
                    "loader's ordered vocabulary is not this model's."
                )
        return (
            word_ids,
            fields['action_role_ids'],
            fields['action_slot_ids'],
            fields['action_word_mask'],
        )

    def _assemble_action_slot_channels(self, word_ids, role_ids, slot_ids, word_mask, dtype):
        """Tensor mirror of ``assemble_slot_channels``: ``[B, S * D]``.

        Same rule, same slot ids, one channel per slot: the mean of that slot's
        member word vectors, L2-normalised, with ``R_B`` applied to the
        ``ROLE_HEAD_1`` word first, and a zero row for an absent slot. Because a
        channel is a function of its own slot's members only, appending modifiers
        moves the head and direction channels by exactly zero.
        """
        vectors = self.action_word_embeddings.to(dtype)[word_ids]
        role_vectors = (
            vectors[..., self.action_role_b_perm] * self.action_role_b_sign.to(dtype)
        )
        vectors = torch.where(
            (role_ids == ROLE_HEAD_1).unsqueeze(-1), role_vectors, vectors
        )
        channels = []
        for slot in range(len(ACTION_LABEL_SLOTS)):
            member = (word_mask & (slot_ids == slot)).unsqueeze(-1).to(dtype)
            count = member.sum(dim=1)
            mean = (vectors * member).sum(dim=1) / count.clamp(min=1.0)
            norm = torch.linalg.vector_norm(mean, dim=-1, keepdim=True)
            # An absent slot is a zero row, never a renormalisation of the others:
            # "this label spells no direction" has to stay distinguishable from
            # "this label spells one".
            channels.append(torch.where(count > 0, mean / norm.clamp(min=1e-9), mean * 0.0))
        return torch.cat(channels, dim=-1)

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

    def _action_condition(self, y, batch_size, device, dtype):
        """``(channels [B, S*D], active [B])`` for the action condition, or ``None``.

        ``active`` folds together the CFG hard-drop mask and "does this row carry
        a label at all": a row with no label carries no action information, and
        routing it through the same unconditional path as a dropped row is what
        makes omitting ``--action_label`` at inference land on the learned
        unconditional mode automatically.
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
            word_ids, role_ids, slot_ids, word_mask = resolved
            channels = self._assemble_action_slot_channels(
                word_ids, role_ids, slot_ids, word_mask, dtype
            )
        active = self._resolve_action_label_active(
            y.get('action_label_active'), batch_size, device
        )
        active = active & self._resolve_action_label_valid(
            y, batch_size, device, word_mask
        )
        return channels, active

    def _build_action_label_token(self, y, batch_size, device, dtype):
        """The action token added to the timestep embedding, or ``None`` when off."""
        if self.action_label_projection is None:
            return None
        resolved = self._action_condition(y, batch_size, device, dtype)
        if resolved is None:
            return None
        channels, action_active = resolved
        # An inactive row goes to the learned null rather than pushing the zero
        # vector through the projection, whose output there is an untrained region.
        null_emb = self.action_label_null_emb.to(device=device, dtype=dtype)
        return torch.where(
            action_active.view(batch_size, 1),
            self.action_label_projection(channels),
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

    def _coerce_species_emb(self, y, batch_size, device, dtype):
        """Pull y['species_emb'] and broadcast it to [B, t5_out_dim]. Shared by the
        FiLM head (--species_cond) and the per-joint fusion (--species_joint_cond)."""
        raw_species_emb = y.get('species_emb') if y is not None else None
        if raw_species_emb is None:
            raise ValueError(
                "species conditioning is enabled but y['species_emb'] is missing. "
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
        return species_emb

    def _resolve_species_active(self, raw_species_active, batch_size, device):
        """Per-sample keep mask for the species FiLM condition. An explicit
        y['species_active'] (e.g. the unconditional branch at sampling) overrides;
        otherwise training samples a Bernoulli keep so the condition is hard-dropped
        with probability species_cfg_drop_prob; eval/inference keeps every sample."""
        if raw_species_active is not None:
            active = torch.as_tensor(raw_species_active, device=device, dtype=torch.bool).reshape(-1)
            if active.numel() == 1 and batch_size != 1:
                active = active.expand(batch_size)
            elif active.numel() != batch_size:
                raise ValueError(
                    "species_active batch dimension must match the motion batch size, got "
                    f"{active.numel()} for batch {batch_size}"
                )
            return active
        if self.training and self.species_cfg_drop_prob > 0.0:
            return torch.rand(batch_size, device=device) >= self.species_cfg_drop_prob
        return torch.ones(batch_size, device=device, dtype=torch.bool)

    def _apply_species_film(self, timesteps_emb, y, batch_size, device, dtype):
        """FiLM-modulate the timestep embedding by the species descriptor:
        out = gamma * t + beta with gamma = 1 + gamma_residual (zero-init -> identity
        at start). Hard-dropped/unconditional samples bypass to identity (gamma=1,
        beta=0) rather than a running-mean substitute. No-op when disabled."""
        if self.species_film is None:
            return timesteps_emb
        species_emb = self._coerce_species_emb(y, batch_size, device, dtype)
        gamma_residual, beta = self.species_film(species_emb).chunk(2, dim=-1)
        gamma = 1.0 + gamma_residual
        active = self._resolve_species_active(
            y.get('species_active') if y is not None else None, batch_size, device
        ).view(batch_size, 1)
        gamma = torch.where(active, gamma, torch.ones_like(gamma))
        beta = torch.where(active, beta, torch.zeros_like(beta))
        return gamma * timesteps_emb + beta

    def _coerce_canonical_frame_stat(self, raw_stat, name, batch_size, device, dtype):
        """Shape one canonical stat vector to ``[B, feature_len]``.

        The collate stacks these per-sample (``[B, F]``) because a mixed-species
        batch spans several object_subsets; a single-species caller may pass the
        bare ``[F]`` vector, which is broadcast.
        """
        stat = torch.as_tensor(raw_stat, device=device, dtype=dtype)
        if stat.dim() == 1:
            stat = stat.unsqueeze(0)
        stat = stat.reshape(stat.shape[0], -1)[:, :self.feature_len]
        if stat.shape[1] != self.feature_len:
            raise ValueError(
                f"y['{name}'] must carry at least {self.feature_len} channels, got "
                f"{stat.shape[1]}"
            )
        if stat.shape[0] == 1 and batch_size != 1:
            stat = stat.expand(batch_size, -1)
        elif stat.shape[0] != batch_size:
            raise ValueError(
                f"{name} batch dimension must match the motion batch size, got "
                f"{stat.shape[0]} for batch {batch_size}"
            )
        return stat

    def _build_canonical_frame_token(self, y, batch_size, device, dtype):
        """Project the sample's output coordinate frame into the timestep token.

        The frame is ``[canonical_feature_mean || canonical_feature_std]`` --
        exactly the vectors ``canonical_to_physical_hml`` will de-standardize the
        output with. Always built and never dropped: see the constructor.
        """
        raw_mean = y.get('canonical_feature_mean') if y is not None else None
        raw_std = y.get('canonical_feature_std') if y is not None else None
        if raw_mean is None or raw_std is None:
            raise ValueError(
                "y['canonical_feature_mean'] / y['canonical_feature_std'] are "
                "missing. They define the output coordinate frame and are always "
                "read. Regenerate cond.npy so each species carries its "
                "object_subset's canonical standardization stats."
            )
        mean = self._coerce_canonical_frame_stat(
            raw_mean, 'canonical_feature_mean', batch_size, device, dtype)
        std = self._coerce_canonical_frame_stat(
            raw_std, 'canonical_feature_std', batch_size, device, dtype)
        return self.canonical_frame_projection(torch.cat([mean, std], dim=-1))

    def forward(self, x, timesteps, y=None, train_step=None, **unused_kwargs):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """

        joints_padding_mask = y['joints_padding_mask'].to(x.device)
        rest_pose = y['rest_pose'].to(x.device).unsqueeze(0)

        bs, njoints, nfeats, nframes = x.shape
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
        # Species FiLM modulates the base time signal *before* the additive
        # condition tokens (action/loop/playspeed) are summed, so each conditioning
        # channel stays independent and the additive tokens are not scaled by it.
        timesteps_emb = self._apply_species_film(timesteps_emb, y, bs, x.device, x.dtype)
        playspeed_condition = self._coerce_playspeed_cond(
            y.get('playspeed_cond'),
            batch_size=bs,
            device=x.device,
            dtype=x.dtype,
        )
        timesteps_emb = timesteps_emb + self.playspeed_projection(playspeed_condition)
        timesteps_emb = timesteps_emb + self._build_canonical_frame_token(
            y, bs, x.device, x.dtype)
        if self.loop_cond_prob > 0.0 and self.loop_condition_projection is not None:
            loop_condition = self._coerce_loop_condition(
                y.get('is_loop'),
                batch_size=bs,
                device=x.device,
                dtype=x.dtype,
            )
            timesteps_emb = timesteps_emb + self.loop_condition_projection(loop_condition)
        action_label_token = self._build_action_label_token(y, bs, x.device, x.dtype)
        if action_label_token is not None:
            timesteps_emb = timesteps_emb + action_label_token

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

        species_emb_for_joints = (
            self._coerce_species_emb(y, bs, x.device, x.dtype) if self.species_joint_cond else None
        )
        x = self.input_process(x, rest_pose, y['joints_names_embs'], species_emb_for_joints) # applies linear layer on each frame to convert it to latent dim
        spatial_mask = (1.0 - joints_padding_mask[:, 0, 0, 1:, 1:].float()) * -1e4
        spatial_mask = spatial_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        # Temporal self-attention is unmasked: every frame token attends over the
        # whole window, including the T-pose token at index 0 and, symmetrically,
        # that token over every frame.

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

        output = self.seqTransDecoder(
            tgt=x,
            timesteps_embs=timesteps_emb,
            memory=None,
            spatial_mask=spatial_mask,
            tgt_key_padding_mask=joint_key_padding_mask,
            y=y,
            cross_limb_unreliable_mask=cross_limb_unreliable_mask,
            loop_phase_mask=loop_phase_mask,
            lengths=loop_phase_lengths,
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
    def __init__(self, input_feats, root_input_feats, latent_dim, t5_output_dim, dropout_prob=0, species_joint_cond=False):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.root_input_feats = root_input_feats
        self.species_joint_cond = species_joint_cond
        self.root_embedding = nn.Linear(self.root_input_feats, self.latent_dim)
        self.tpos_root_embedding = nn.Linear(self.root_input_feats, self.latent_dim)
        self.joint_embedding = nn.Linear(self.input_feats, self.latent_dim)
        self.tpos_joint_embedding = nn.Linear(self.input_feats, self.latent_dim)
        self.joints_names_dropout = nn.Dropout(p=dropout_prob)
        # When --species_joint_cond, the species descriptor FiLM-modulates each
        # per-joint name embedding: gamma/beta are produced from the *concatenation*
        # of that joint's embedding and the species descriptor, so the modulation is
        # a function of the species x joint interaction.
        #
        # The concat (rather than a sum) is load-bearing: this replaced a purely
        # additive `joints_emb + species_proj(species_emb)` fusion, and since
        # text_embedding is a bare Linear with no nonlinearity in between, that
        # additive form collapsed to `E.j_i + const(s)` -- the same per-species
        # constant for every joint, i.e. exactly zero per-joint differentiation.
        # With concat, the old behaviour stays trivially representable
        # (gamma_residual=0, beta=W(s)), so the FiLM optimum is no worse.
        #
        # The final linear is zero-initialized so the head starts at identity
        # (gamma=1, beta=0); the latent_dim bottleneck matches the house style of
        # the timestep species_film head and keeps the parameter cost down.
        text_in_dim = t5_output_dim
        self.species_film_j = nn.Sequential(
            nn.Linear(2 * t5_output_dim, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, 2 * t5_output_dim),
        ) if species_joint_cond else None
        if self.species_film_j is not None:
            nn.init.zeros_(self.species_film_j[-1].weight)
            nn.init.zeros_(self.species_film_j[-1].bias)
        self.text_embedding = nn.Linear(text_in_dim, self.latent_dim)
    def forward(self, x, rest_pose, joints_embedded_names, species_emb=None):
        # x.shape = [batch_size, joints, 13, frames]
        x = x.permute(3, 0, 1, 2) # [frames, batch_size, n_joints, features_len]
        rest_pose_all_joints_except_root = self.tpos_joint_embedding(rest_pose[:, :, 1:])
        rest_pose_root_data = self.tpos_root_embedding(rest_pose[:, :, 0:1])
        all_joints_except_root = self.joint_embedding(x[:, :, 1:])
        root_data = self.root_embedding(x[:, :, 0:1])
        tpos_embedded = torch.cat([rest_pose_root_data, rest_pose_all_joints_except_root], dim=2)
        x_embedded = torch.cat([root_data, all_joints_except_root], dim=2)
        x = torch.cat([tpos_embedded, x_embedded], dim=0)
        joints_clean = joints_embedded_names.to(x.device)
        joints_embedded_names = self.joints_names_dropout(joints_clean)
        if self.species_joint_cond:
            if species_emb is None:
                raise ValueError(
                    "species_joint_cond is enabled but species_emb was not passed to "
                    "InputProcess (expected y['species_emb'])."
                )
            # The FiLM condition reads the *pre-dropout* joint embedding on purpose.
            # joints_names_dropout zeroes entries and rescales by 1/(1-p) at train
            # time only; feeding that to the head would make gamma/beta themselves
            # train/eval-mismatched. Dropout applies to the modulated copy alone.
            # joints_clean: [B, J, t5]; species_emb: [B, t5] -> broadcast to [B, J, t5]
            species_broadcast = species_emb.to(device=x.device, dtype=joints_clean.dtype).unsqueeze(1).expand(-1, joints_clean.shape[1], -1)
            gamma_residual, beta = self.species_film_j(
                torch.cat([joints_clean, species_broadcast], dim=-1)
            ).chunk(2, dim=-1)
            joints_embedded_names = (1.0 + gamma_residual) * joints_embedded_names + beta
        joints_embedded_names = self.text_embedding(joints_embedded_names)
        x = x + joints_embedded_names[None, ...]# [frames, batch_size, n_joints, d]
        positions = torch.arange(x.shape[0], device=x.device).view(1, -1, 1).repeat(x.shape[1], 1, 1)
        pos_emb = create_sin_embedding(positions, self.latent_dim)[0]
        return x + pos_emb.unsqueeze(1).unsqueeze(1)


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
