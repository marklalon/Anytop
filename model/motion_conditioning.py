import torch
import torch.nn as nn
import numpy as np
from model.motion_transformer import (
    GraphMotionDecoderLayer,
    GraphMotionDecoder,
    circular_phase_embedding,
    run_in_fp32,
)
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


class MotionConditioningMixin:
    """Shared action slots, reference priority and physical-time semantics."""
    def _reference_condition(self, y, batch_size, frames, device):
        if self.reference_encoder is None:
            return None, None
        active = torch.as_tensor(y['ref_active'], device=device, dtype=torch.bool).reshape(-1)
        if active.numel() != batch_size:
            raise ValueError('ref_active must have shape [B]')
        active = active & y['ref_mask'].to(device).any(-1)
        ref_y = {k: v.to(device) for k, v in y.items() if k.startswith('ref_') and torch.is_tensor(v)}
        if ref_y['ref_feats'].shape[-1] != frames:
            raise ValueError('Reference and output windows must be frame aligned')
        if not self.training or self.ref_cfg_drop_prob == 0:
            return self.reference_encoder(ref_y), active
        # Reference dropout keeps a fixed number of randomly chosen rows per
        # batch, so the encoder runs on those rows only (static shape for
        # compile); a dropped row is hard-bypassed and its z is never read.
        keep = batch_size - round(batch_size * self.ref_cfg_drop_prob)
        encoder = self.reference_encoder
        if keep == 0:
            z = ref_y['ref_feats'].new_zeros(batch_size, frames, encoder.slots, encoder.ref_dim)
            return z, torch.zeros_like(active)
        rows = torch.randperm(batch_size, device=device)[:keep]
        z = encoder({k: v[rows] for k, v in ref_y.items()})
        z = z.new_zeros((batch_size,) + z.shape[1:]).index_copy(0, rows, z)
        kept = torch.zeros_like(active).index_fill(0, rows, True)
        return z, active & kept


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
