"""Canonical AnyTop motion feature conversion helpers.

V3 keeps the HML-like physical feature layout but converts to a *canonical*
model feature space via three prior-free, exactly-invertible steps:

  1. Position re-centering: the position channel is expressed relative to the
     skeleton rest-pose feature position (residual-from-rest).
  2. Per-skeleton size normalization (L-scaling): the position and velocity
     channels are divided by the skeleton's own geometric length ``L`` so that
     every species lands in a size-free space (cross-species fair). Rotation
     (6d) is size-independent and is left untouched here. ``L`` is floored at
     ``REST_LENGTH_SCALE_FLOOR``: a rig of a few clustered joints (MU04_Pollen,
     4 joints, L=0.03) has almost no joint spread, and dividing its body-scale
     root trajectory by that spread blew a 1.7-unit fall up to ~70 std.
  3. Global standardization: subtract a per-channel mean and divide by a
     *block-collapsed* std (12-vectors). One pair of vectors for the whole
     corpus, computed at preprocessing time in the L-normalized space of step
     (2) by :func:`pool_global_stats`: clips are first pooled per object_subset,
     then the subsets are combined with equal weight so the largest subsets do
     not set the scale for the rest. Every species is written in the same
     canonical frame, so the frame carries no body-plan information and the
     same physical motion has the same feature amplitude on every skeleton.
     The vectors are stored on every cond entry (``canonical_feature_mean`` /
     ``canonical_feature_std``) and are identical across entries; a new
     species copies them from any entry of the training cond.

     Each block (position / rotation-6d / velocity) of ``std`` is one scalar,
     so ``l_simple`` weights every joint and axis of a block uniformly.

Channel layout per joint (n_feats == 12):
    0:3   position   (rest-centered residual)
    3:9   rotation 6d
    9:12  local velocity

Every channel belongs to exactly one of the three blocks, so ``pool_global_stats``
covers the whole vector.

Derived velocity. A joint's world step is its own RIC position step plus the
translation root's XZ step, so the only velocity the positions do not already
carry is the root's X/Z. The canonical space therefore keeps velocity on the
root (joint 0) X/Z only (``ROOT_VEL_CHANNELS``): every other velocity channel is
zero after encode and after decode, the model neither reads nor writes it, and
the velocity block's statistics are the root X/Z's. Consumers of a decoded
sample that need the full physical rows call :func:`rebuild_derived_velocity`.

"""

from __future__ import annotations

import numpy as np

from data_loaders.truebones.truebones_utils.param_utils import HML_REF_REST_LENGTH_SCALE

try:
    import torch
except Exception:  # pragma: no cover - torch is present in training/runtime.
    torch = None


CANONICAL_FEATURE_SPACE = "canonical_motion_v4"
PHYSICAL_FEATURE_SPACE = "hml_like_v_current"

# Keys under which the global per-channel standardization statistics are stored
# on each cond entry. Every entry carries the same 12-vectors. Computed at
# preprocessing time over the L-normalized training distribution -- see
# ``pool_global_stats``.
CANONICAL_MEAN_KEY = "canonical_feature_mean"
CANONICAL_STD_KEY = "canonical_feature_std"
# RMS of one frame's L-normalized position step, pooled like the block stds. The
# position-step loss divides by it so its weight means the same on every corpus.
CANONICAL_DPOS_STD_KEY = "canonical_dpos_std"

# Per-species weights of the root-velocity and position-step losses: a species
# whose own energy in that term exceeds ``AUX_LOSS_ENERGY_CAP`` (in the global
# canonical units) is weighted by sqrt(cap / energy), never up. This softens
# the influence of fast movers without capping their weighted loss energy.
ROOT_VEL_LOSS_WEIGHT_KEY = "root_vel_loss_weight"
DPOS_LOSS_WEIGHT_KEY = "dpos_loss_weight"
AUX_LOSS_ENERGY_CAP = 2.0

# The model's channel contract (see "Derived velocity" above). Joint 0 is the
# translation root of every cond skeleton.
ROOT_VEL_CHANNELS = (9, 11)
JOINT_MODEL_CHANNELS = tuple(range(0, 9))
ROOT_MODEL_CHANNELS = JOINT_MODEL_CHANNELS + ROOT_VEL_CHANNELS

# Per-sample geometric length ``L`` (``[B]``) that the training collate
# precomputes from each sample's unpadded rest and stores in ``y``; see
# ``_length_scale_from_cond``. Never stored in cond.npy (species entries derive
# it from ``rest_pos_ric_hml`` on the fly).
REST_LENGTH_SCALE_KEY = "rest_length_scale"

# Lower bound on ``L``: half the reference skeleton's own ``L``. Preprocessing
# has already scaled every character to the reference body size (its extent
# folds in the root's elevation, which the joint spread never sees), so an ``L``
# far below the reference means clustered joints, not a small body. Of 260
# species it lifts the seven 2-6 joint effect/drifting rigs (0.030-0.106, up to
# 4.3x) and grazes two 8-joint rigs (0.126, 0.127 -> 0.130).
REST_LENGTH_SCALE_FLOOR = 0.5 * HML_REF_REST_LENGTH_SCALE

# Channels whose std falls below this floor are treated as unit-variance so the
# encode divide never explodes (mirrors the old std_safe floor).
_STD_FLOOR = 1e-5

_POS_SLICE = slice(0, 3)
_ROT_SLICE = slice(3, 9)
_VEL_SLICE = slice(9, 12)


def _is_torch_tensor(value) -> bool:
    return torch is not None and torch.is_tensor(value)


def _kept_channel_mask(n_joints, n_feats):
    """``[J, F]`` float mask: 0 on the derived velocity channels, 1 elsewhere."""
    mask = np.ones((int(n_joints), int(n_feats)), dtype=np.float32)
    if n_feats >= _VEL_SLICE.stop:
        mask[:, _VEL_SLICE] = 0.0
        if n_joints > 0:
            mask[0, list(ROOT_VEL_CHANNELS)] = 1.0
    return mask


_KEPT_MASK_TENSOR_CACHE = {}


def zero_derived_velocity(feature):
    """Zero every velocity channel except the root's X/Z.

    Accepts ``[B, J, F, T]`` (torch or numpy) or channels-last ``[..., J, F]``.
    """
    if _is_torch_tensor(feature):
        if feature.dim() == 4:
            n_joints, n_feats = feature.shape[1], feature.shape[2]
        else:
            n_joints, n_feats = feature.shape[-2], feature.shape[-1]
        key = (int(n_joints), int(n_feats), str(feature.device), feature.dtype)
        mask = _KEPT_MASK_TENSOR_CACHE.get(key)
        if mask is None:
            mask = torch.as_tensor(
                _kept_channel_mask(n_joints, n_feats), device=feature.device, dtype=feature.dtype
            )
            _KEPT_MASK_TENSOR_CACHE[key] = mask
        if feature.dim() == 4:
            return feature * mask.view(1, n_joints, n_feats, 1)
        return feature * mask

    out = np.asarray(feature, dtype=np.float32)
    if out.ndim == 4:
        mask = _kept_channel_mask(out.shape[1], out.shape[2])[None, :, :, None]
    else:
        mask = _kept_channel_mask(out.shape[-2], out.shape[-1])
    return (out * mask).astype(np.float32, copy=False)


def rebuild_derived_velocity(motion, *, is_loop, step_scale=1.0):
    """Fill the derived velocity rows of a decoded physical ``[T, J, F]`` clip.

    ``vel[t] * step_scale`` is the world step from frame ``t`` to ``t + 1``: the
    joint's RIC position step plus the root's XZ step (RIC removes the root's
    world XZ from every joint, the root included). The terminal row follows
    preprocessing: a loop's wrap step, otherwise the last step repeated. The
    root's X/Z rows are the model's own and are kept.
    """
    out = np.array(motion, dtype=np.float32, copy=True)
    n_frames = out.shape[0]
    if out.ndim != 3 or out.shape[-1] < _VEL_SLICE.stop or n_frames == 0:
        return out
    step = float(step_scale) if step_scale and step_scale > 0 else 1.0
    pos = out[..., 0:3].astype(np.float64)
    root_xz_step = np.zeros((n_frames, 1, 3), dtype=np.float64)
    root_xz_step[:, 0, 0] = out[:, 0, 9] * step
    root_xz_step[:, 0, 2] = out[:, 0, 11] * step
    world_step = np.zeros(pos.shape, dtype=np.float64)
    world_step[:-1] = pos[1:] - pos[:-1] + root_xz_step[:-1]
    if is_loop or n_frames == 1:
        world_step[-1] = pos[0] - pos[-1] + root_xz_step[-1]
    else:
        world_step[-1] = world_step[-2]
    root_xz = out[:, 0, list(ROOT_VEL_CHANNELS)].copy()
    out[..., _VEL_SLICE] = (world_step / step).astype(np.float32)
    out[:, 0, list(ROOT_VEL_CHANNELS)] = root_xz
    return out


def rest_pos_from_cond(cond_entry, *, like=None):
    """The skeleton's physical rest-pose positions, ``[J, 3]``.

    Public: the reference encoder's bone-direction channel reads this same
    position channel, so it is part of the feature contract, not an internal
    helper.
    """
    if cond_entry is None:
        raise KeyError("cond_entry is required for canonical feature conversion.")

    if "rest_pos_ric_hml" in cond_entry:
        rest_pos = cond_entry["rest_pos_ric_hml"]
    elif "rest_pose_physical" in cond_entry:
        rest_pos = cond_entry["rest_pose_physical"][..., 0:3]
    elif "rest_pose" in cond_entry:
        rest_pos = cond_entry["rest_pose"][..., 0:3]
    else:
        raise KeyError(
            "cond entry is missing rest_pos_ric_hml/rest_pose_physical/rest_pose; "
            "cannot convert canonical feature space."
        )

    if _is_torch_tensor(like):
        return torch.as_tensor(rest_pos, device=like.device, dtype=like.dtype)
    if _is_torch_tensor(rest_pos):
        return rest_pos.detach().cpu().numpy()
    return np.asarray(rest_pos, dtype=np.float32)


def _joint_count_batch(n_counts: int, rest_batch: int, rest_shape) -> int:
    """Batch size of ``L`` for ``n_counts`` joint counts against a rest batch.

    Broadcasts like numpy: one count covers every rest row, and one rest row
    (an unbatched ``[J, 3]`` rest is one row) is shared by every count -- the
    homogeneous batch whose loader hands a single species rest alongside the
    collated per-row ``n_joints``. Each row then gets its own ``L`` from its own
    count, so unequal counts are honoured rather than silently collapsed.
    """
    if n_counts == rest_batch or rest_batch == 1:
        return n_counts
    if n_counts == 1:
        return rest_batch
    raise ValueError(
        f"n_joints has {n_counts} entries but rest_pos has shape {tuple(rest_shape)}: "
        "pass one count, one per batch row, or a single shared rest."
    )


def _length_scale_from_rest(rest_pos, n_joints=None):
    """Per-skeleton geometric length L = RMS spread of rest-pose joint positions.

    Accepts ``[J, 3]`` (returns a scalar) or ``[B, J, 3]`` (returns ``[B]``).
    ``n_joints`` gives the real joint count(s) so the zero-padded rows of a
    collated mixed-skeleton batch are left out; without it every row counts.
    Counts and rest rows broadcast (see ``_joint_count_batch``): the result is a
    scalar only for an unbatched rest with a single count, otherwise ``[B]``.
    Derived purely from skeleton geometry -- no motion stats. Floored at
    ``REST_LENGTH_SCALE_FLOOR``; a collapsed or non-finite rest lands on the
    floor too.
    """
    if _is_torch_tensor(rest_pos):
        rp = rest_pos.to(dtype=torch.float32)
        if n_joints is None:
            centered = rp - rp.mean(dim=-2, keepdim=True)
            L = torch.sqrt((centered ** 2).mean(dim=(-2, -1)))
        else:
            if rp.dim() not in (2, 3):
                raise ValueError(
                    f"rest_pos must have shape [J, 3] or [B, J, 3], got {tuple(rp.shape)}."
                )
            rp3 = rp.unsqueeze(0) if rp.dim() == 2 else rp
            max_joints = rp3.shape[1]
            counts = torch.as_tensor(n_joints, device=rp3.device, dtype=torch.long).reshape(-1)
            batch = _joint_count_batch(counts.numel(), rp3.shape[0], rp.shape)
            rp3 = rp3.expand(batch, -1, -1)
            counts = counts.expand(batch)
            # Counts from a collated y live on CUDA. Validating them there would
            # force a device synchronization on every decode, so only CPU callers
            # get the fail-fast (the training decode reads rest_length_scale and
            # does not reach this path at all).
            if (
                counts.device.type == "cpu"
                and bool(((counts <= 0) | (counts > max_joints)).any())
            ):
                raise ValueError(
                    f"n_joints must be in [1, {max_joints}], got {counts.tolist()}."
                )

            valid = torch.arange(max_joints, device=rp3.device).unsqueeze(0) < counts.unsqueeze(1)
            valid_f = valid.to(dtype=rp3.dtype).unsqueeze(-1)
            counts_f = counts.to(dtype=rp3.dtype)
            mean = (rp3 * valid_f).sum(dim=-2, keepdim=True) / counts_f.view(-1, 1, 1)
            squared = ((rp3 - mean) ** 2) * valid_f
            L = torch.sqrt(squared.sum(dim=(-2, -1)) / (counts_f * rp3.shape[-1]))
            if rp.dim() == 2 and counts.numel() == 1:
                L = L[0]
        # Mirrors the numpy branch below.
        return torch.where(torch.isfinite(L), L, torch.zeros_like(L)).clamp_min(
            REST_LENGTH_SCALE_FLOOR
        )

    rp = np.asarray(rest_pos, dtype=np.float64)
    if n_joints is None:
        centered = rp - rp.mean(axis=-2, keepdims=True)
        L = np.sqrt((centered ** 2).mean(axis=(-2, -1)))
    else:
        if rp.ndim not in (2, 3):
            raise ValueError(
                f"rest_pos must have shape [J, 3] or [B, J, 3], got {rp.shape}."
            )
        rp3 = rp[None, ...] if rp.ndim == 2 else rp
        max_joints = rp3.shape[1]
        counts = np.asarray(n_joints, dtype=np.int64).reshape(-1)
        batch = _joint_count_batch(counts.size, rp3.shape[0], rp.shape)
        rp3 = np.broadcast_to(rp3, (batch,) + rp3.shape[1:])
        counts = np.broadcast_to(counts, (batch,))
        if np.any((counts <= 0) | (counts > max_joints)):
            raise ValueError(f"n_joints must be in [1, {max_joints}], got {counts.tolist()}.")

        valid = np.arange(max_joints)[None, :] < counts[:, None]
        valid_f = valid[..., None].astype(rp3.dtype)
        mean = (rp3 * valid_f).sum(axis=-2, keepdims=True) / counts.reshape(-1, 1, 1)
        squared = ((rp3 - mean) ** 2) * valid_f
        L = np.sqrt(squared.sum(axis=(-2, -1)) / (counts * rp3.shape[-1]))
        if rp.ndim == 2 and counts.size == 1:
            L = L[0]
    return np.maximum(np.where(np.isfinite(L), L, 0.0), REST_LENGTH_SCALE_FLOOR)


def _spatial_L_vectors(n_feats):
    """Return a per-channel ``uses_L`` vector of length ``n_feats``.

    ``uses_L[c] == 1`` for the position and velocity groups (size-dependent) and
    ``0`` for rotation (size-independent). The L-scaling divides those channels
    by the per-skeleton length ``L`` (encode) / multiplies (decode).
    """
    uses_L = np.zeros(n_feats, dtype=np.float64)
    uses_L[_POS_SLICE] = 1.0
    if n_feats >= _VEL_SLICE.stop:
        uses_L[_VEL_SLICE] = 1.0
    return uses_L


def _length_scale_from_cond(cond_entry, *, like=None):
    """Resolve the per-skeleton length ``L`` for a cond entry / collated ``y``.

    The collate precomputes ``rest_length_scale`` (``[B]``) from each sample's
    UNPADDED rest, so the training aux-loss decode reads it instead of spending
    a dozen kernel launches re-deriving it from the padded rest every step.
    Without it, ``L`` is derived from the rest positions, honouring ``n_joints``
    so the zero-padded rows of a collated batch are left out.
    """
    L = cond_entry.get(REST_LENGTH_SCALE_KEY) if hasattr(cond_entry, "get") else None
    if L is not None:
        return L
    n_joints = cond_entry.get("n_joints") if hasattr(cond_entry, "get") else None
    return _length_scale_from_rest(
        rest_pos_from_cond(cond_entry, like=like), n_joints=n_joints
    )


_SPATIAL_L_TENSOR_CACHE = {}


def _spatial_L_vector_tensor(n_feats, device, dtype):
    """``_spatial_L_vectors`` as a cached tensor: rebuilding it on the device on
    every decode was a blocking host copy (a stream sync) per call."""
    key = (int(n_feats), str(device), dtype)
    cached = _SPATIAL_L_TENSOR_CACHE.get(key)
    if cached is None:
        cached = torch.as_tensor(_spatial_L_vectors(n_feats), device=device, dtype=dtype)
        _SPATIAL_L_TENSOR_CACHE[key] = cached
    return cached


def _apply_L_scale(feature, cond_entry, inverse: bool):
    """Divide (encode) or multiply (decode) the position/velocity channels by the
    per-skeleton length ``L``. Exact inverse of itself with flipped ``inverse``.
    Rotation channels are left unchanged. ``L`` may be a scalar or ``[B]``.
    """
    L = _length_scale_from_cond(cond_entry, like=feature)

    if _is_torch_tensor(feature):
        out = feature
        ndim = out.dim()
        n_feats = out.shape[2] if ndim == 4 else out.shape[-1]
        uses_L = _spatial_L_vector_tensor(n_feats, out.device, out.dtype)
        Lt = (L if _is_torch_tensor(L) else torch.as_tensor(np.asarray(L))) \
            .to(device=out.device, dtype=out.dtype).reshape(-1)

        if ndim == 4:
            # [B, J, F, T]; L is scalar or [B].
            if Lt.numel() == 1:
                scale = (Lt ** uses_L).view(1, 1, n_feats, 1)
            else:
                scale = (Lt.view(-1, 1) ** uses_L.view(1, -1)).view(out.shape[0], 1, n_feats, 1)
        else:
            # channels on the last axis ([J, F] / [T, J, F]); L is scalar.
            if Lt.numel() != 1:
                raise ValueError(
                    f"a per-sample L of {Lt.numel()} entries needs a [B, J, F, T] feature, "
                    f"got rank {ndim}; slice the collated y per sample when decoding one."
                )
            scale = Lt[0] ** uses_L
        return out * scale if inverse else out / scale

    out = np.asarray(feature, dtype=np.float32)
    ndim = out.ndim
    n_feats = out.shape[2] if ndim == 4 else out.shape[-1]
    uses_L = _spatial_L_vectors(n_feats)
    Lnp = np.asarray(L.detach().cpu() if _is_torch_tensor(L) else L, dtype=np.float64)

    if ndim == 4:
        if Lnp.ndim == 0:
            scale = (Lnp ** uses_L).reshape(1, 1, n_feats, 1)
        else:
            scale = (Lnp.reshape(-1, 1) ** uses_L.reshape(1, -1)).reshape(out.shape[0], 1, n_feats, 1)
    else:
        scale = Lnp ** uses_L
    out = out * scale if inverse else out / scale
    return out.astype(np.float32, copy=False)


def get_canonical_global_stats(cond_entry):
    """Return ``(mean, std)`` global standardization vectors from a
    cond entry, or ``None`` when absent (cond predates the stats, e.g. tests).
    """
    if cond_entry is None:
        return None
    mean = cond_entry.get(CANONICAL_MEAN_KEY) if hasattr(cond_entry, "get") else None
    std = cond_entry.get(CANONICAL_STD_KEY) if hasattr(cond_entry, "get") else None
    if mean is None or std is None:
        return None
    return mean, std


def get_canonical_dpos_std(cond_entry):
    """The stored position-step RMS, or ``None`` when absent."""
    if cond_entry is None or not hasattr(cond_entry, "get"):
        return None
    return cond_entry.get(CANONICAL_DPOS_STD_KEY)


def set_canonical_global_stats(cond_entry, mean, std, dpos_std):
    """Store the global mean/std (12-vectors) and the position-step RMS on a
    cond entry, flooring near-constant channels to unit std so the encode
    divide is safe.
    """
    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    std = np.asarray(std, dtype=np.float32).reshape(-1)
    std = np.where(std < _STD_FLOOR, 1.0, std).astype(np.float32, copy=False)
    dpos_std = float(dpos_std)
    if not np.isfinite(dpos_std) or dpos_std < _STD_FLOOR:
        dpos_std = 1.0
    cond_entry[CANONICAL_MEAN_KEY] = mean
    cond_entry[CANONICAL_STD_KEY] = std
    cond_entry[CANONICAL_DPOS_STD_KEY] = np.float32(dpos_std)
    return cond_entry


def set_aux_loss_weights(cond_entry, root_vel_weight=1.0, dpos_weight=1.0):
    """Store a species' root-velocity / position-step loss weights."""
    cond_entry[ROOT_VEL_LOSS_WEIGHT_KEY] = np.float32(root_vel_weight)
    cond_entry[DPOS_LOSS_WEIGHT_KEY] = np.float32(dpos_weight)
    return cond_entry


def _check_per_sample_stat(stat_rows: int, feature_shape):
    """A per-sample ``[B, F]`` stat must line up with a ``[B, J, F, T]`` feature.

    A ``[B, F]`` stat broadcasts silently against a single-sample ``[1, J, F, T]``
    feature (the result grows to ``B`` and ``[0]`` reads row 0's stats for
    every sample), so a row count that matches neither 1 nor the feature
    batch is an error, not a broadcast.
    """
    if len(feature_shape) != 4:
        raise ValueError(
            "per-sample [B, F] canonical stats are only supported for 4D "
            f"[B, J, F, T] features, got feature rank {len(feature_shape)}."
        )
    if stat_rows not in (1, feature_shape[0]):
        raise ValueError(
            f"per-sample canonical stats carry {stat_rows} rows but the feature batch "
            f"has {feature_shape[0]} samples; slice the collated y per sample "
            "(stats[i]) when decoding one sample out of a batch."
        )


def _view_stat_torch(stat, feature_shape, n_feats: int):
    """Shape a standardization stat tensor for broadcasting against a feature.

    ``stat`` is either 1D ``[F]`` (broadcast over the whole batch) or 2D
    ``[B, F]`` (per-sample). For a 4D ``[B, J, F, T]`` feature the result is
    ``[1, 1, F, 1]`` (1D) or ``[B, 1, F, 1]`` (per-sample). For lower-rank
    single-sample features the 1D vector broadcasts over the trailing axis as-is.
    """
    if stat.dim() >= 2:
        _check_per_sample_stat(stat.shape[0], feature_shape)
        stat2 = stat.reshape(stat.shape[0], -1)[:, :n_feats]
        return stat2.view(stat2.shape[0], 1, n_feats, 1)
    stat1 = stat.reshape(-1)[:n_feats]
    return stat1.view(1, 1, n_feats, 1) if len(feature_shape) == 4 else stat1


def _view_stat_numpy(stat, feature_shape, n_feats: int):
    """Numpy counterpart of _view_stat_torch (same 1D / per-sample contract)."""
    if stat.ndim >= 2:
        _check_per_sample_stat(stat.shape[0], feature_shape)
        stat2 = stat.reshape(stat.shape[0], -1)[:, :n_feats]
        return stat2.reshape(stat2.shape[0], 1, n_feats, 1)
    stat1 = stat.reshape(-1)[:n_feats]
    return stat1.reshape(1, 1, n_feats, 1) if len(feature_shape) == 4 else stat1


def _apply_global_stats(feature, cond_entry, inverse: bool):
    """Standardize (encode: ``(x - mean) / std``) or de-standardize
    (decode: ``x * std + mean``) every channel by the global statistics
    stored on ``cond_entry``.

    Raises if the statistics are absent. Silently skipping standardization would
    leave the features in the L-normalized space (wrong scale) without any error
    -- exactly the failure mode that produced broken inference when a caller
    passed a decode dict missing the stats. Callers must always thread the stats
    through (cond entry, or the collated ``y`` dict). The stats-free
    L-normalized space is reachable only via physical_hml_to_lnorm(), which is
    used solely to *compute* these statistics at preprocessing time.
    """
    stats = get_canonical_global_stats(cond_entry)
    if stats is None:
        raise KeyError(
            f"cond entry is missing {CANONICAL_MEAN_KEY!r}/{CANONICAL_STD_KEY!r}; "
            "cannot (de)standardize canonical features. Pass the global stats "
            "(from the cond entry or the collated y dict), or regenerate "
            "cond.npy. A freshly built new-skeleton cond copies these stats from "
            "the dataset cond.npy."
        )
    mean, std = stats

    if _is_torch_tensor(feature):
        out = feature
        ndim = out.dim()
        n_feats = out.shape[2] if ndim == 4 else out.shape[-1]
        # Stats may arrive as numpy (per-species cond entry) or as torch tensors,
        # possibly on CUDA (collated into y for the training aux-loss decode).
        # Convert without round-tripping through numpy so a CUDA tensor is safe.
        # A 1D ``[F]`` vector is broadcast over the whole batch (single-sample
        # decode). A 2D ``[B, F]`` vector is per-sample, as the collate stacks
        # each item's stats in batch order.
        mean_t = (mean if _is_torch_tensor(mean) else torch.as_tensor(np.asarray(mean, dtype=np.float32))) \
            .to(device=out.device, dtype=out.dtype)
        std_t = (std if _is_torch_tensor(std) else torch.as_tensor(np.asarray(std, dtype=np.float32))) \
            .to(device=out.device, dtype=out.dtype)
        mean_v = _view_stat_torch(mean_t, out.shape, n_feats)
        std_v = _view_stat_torch(std_t, out.shape, n_feats)
        return (out * std_v) + mean_v if inverse else (out - mean_v) / std_v

    out = np.asarray(feature, dtype=np.float32)
    ndim = out.ndim
    n_feats = out.shape[2] if ndim == 4 else out.shape[-1]
    mean_v = _view_stat_numpy(np.asarray(mean, dtype=np.float32), out.shape, n_feats)
    std_v = _view_stat_numpy(np.asarray(std, dtype=np.float32), out.shape, n_feats)
    out = (out * std_v) + mean_v if inverse else (out - mean_v) / std_v
    return out.astype(np.float32, copy=False)


def _apply_rest_pos(feature, cond_entry, sign: float):
    rest_pos = rest_pos_from_cond(cond_entry, like=feature)
    out = feature.clone() if _is_torch_tensor(feature) else np.asarray(feature, dtype=np.float32).copy()

    if _is_torch_tensor(out):
        if out.dim() == 4:
            # [B, J, F, T]
            if rest_pos.dim() == 2:
                rest_pos = rest_pos.unsqueeze(0).expand(out.shape[0], -1, -1)
            out[:, :, 0:3, :] = out[:, :, 0:3, :] + (float(sign) * rest_pos[:, :, :, None])
            return out
        if out.dim() == 3:
            # [T, J, F] or [J, F]. Broadcasting covers both common no-batch cases.
            out[..., 0:3] = out[..., 0:3] + (float(sign) * rest_pos)
            return out
        if out.dim() == 2:
            out[:, 0:3] = out[:, 0:3] + (float(sign) * rest_pos)
            return out
        raise ValueError(f"Unsupported torch feature rank {out.dim()} for canonical conversion.")

    if out.ndim in (2, 3):
        out[..., 0:3] = out[..., 0:3] + (float(sign) * rest_pos)
        return out.astype(np.float32, copy=False)
    if out.ndim == 4:
        if rest_pos.ndim == 2:
            rest_pos = rest_pos[None, :, :, None]
        elif rest_pos.ndim == 3:
            rest_pos = rest_pos[:, :, :, None]
        out[:, :, 0:3, :] = out[:, :, 0:3, :] + (float(sign) * rest_pos)
        return out.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported numpy feature rank {out.ndim} for canonical conversion.")


def physical_hml_to_lnorm(feature, cond_entry):
    """Encode HML-like physical features into the *L-normalized* intermediate
    space (steps 1-2): subtract rest from the position channel, then divide the
    position/velocity channels by the per-skeleton length ``L``. This is the
    space in which the GLOBAL standardization statistics are calibrated.
    """
    centered = _apply_rest_pos(feature, cond_entry, sign=-1.0)
    return _apply_L_scale(centered, cond_entry, inverse=False)


def physical_hml_to_canonical(feature, cond_entry):
    """Encode HML-like physical features into canonical model features.

    Encode order: subtract rest from the position channel, divide the
    position/velocity channels by ``L``, standardize every channel by the
    global mean/std, then zero the derived velocity channels.
    """
    lnorm = physical_hml_to_lnorm(feature, cond_entry)
    return zero_derived_velocity(_apply_global_stats(lnorm, cond_entry, inverse=False))


def canonical_to_physical_hml(feature, cond_entry):
    """Decode canonical model features back to HML-like physical features.

    Decode order (inverse of encode on every kept channel): de-standardize by
    the global mean/std, multiply the position/velocity channels by ``L``, then
    add rest back onto the position channel. The derived velocity channels come
    back as zero; :func:`rebuild_derived_velocity` fills them.
    """
    lnorm = _apply_global_stats(feature, cond_entry, inverse=True)
    unscaled = _apply_L_scale(lnorm, cond_entry, inverse=True)
    return zero_derived_velocity(_apply_rest_pos(unscaled, cond_entry, sign=1.0))


def build_canonical_rest_feature(cond_entry):
    """Build the rest-pose token in canonical feature space.

    The rest pose has no motion, so its physical feature is the rest rotation
    with the position channel equal to the rest position and the velocity
    channels zeroed. Encoding it through the same forward transform
    keeps the rest token in exactly the canonical space the model sees (the
    position residual collapses to 0 before standardization).
    """
    if "rest_pose_physical" in cond_entry:
        rest = cond_entry["rest_pose_physical"]
    elif "rest_pose" in cond_entry:
        rest = cond_entry["rest_pose"]
    else:
        raise KeyError("cond entry is missing rest_pose/rest_pose_physical.")

    if _is_torch_tensor(rest):
        rest_phys = rest.clone()
        if rest_phys.shape[-1] >= _VEL_SLICE.stop:
            rest_phys[..., _VEL_SLICE] = 0.0
    else:
        rest_phys = np.asarray(rest, dtype=np.float32).copy()
        if rest_phys.shape[-1] >= _VEL_SLICE.stop:
            rest_phys[..., _VEL_SLICE] = 0.0

    return physical_hml_to_canonical(rest_phys, cond_entry)


def _new_term_acc(n_feats):
    return {
        "root_sum": np.zeros(n_feats), "root_sumsq": np.zeros(n_feats), "root_count": 0,
        "dpos_sumsq": 0.0, "dpos_count": 0,
    }


def _accumulate_terms(acc, lnorm):
    """Root-row and position-step sums of one finite ``[T, J, F]`` clip."""
    root = lnorm[:, 0, :]
    acc["root_sum"] += root.sum(axis=0)
    acc["root_sumsq"] += (root ** 2).sum(axis=0)
    acc["root_count"] += root.shape[0]
    if lnorm.shape[0] > 1:
        steps = lnorm[1:, :, _POS_SLICE] - lnorm[:-1, :, _POS_SLICE]
        acc["dpos_sumsq"] += float((steps ** 2).sum())
        acc["dpos_count"] += int(steps.size)


def accumulate_lnorm_stats(feature, cond_entry, acc=None, species=None):
    """Accumulate the L-normalized statistics of one physical ``[T, J, F]`` clip.

    Per-channel sum / sum-of-squares over every joint and frame (position and
    rotation blocks), the same over the root row alone (the velocity block, see
    "Derived velocity"), and the sum of squared position steps. With
    ``species`` the root and step sums are also kept per species for
    :func:`species_aux_loss_weights`. ``acc`` is a mutable dict carried across
    clips; finalize_lnorm_stats() turns it into ``(mean, std, dpos_std)``.
    """
    lnorm = np.asarray(physical_hml_to_lnorm(feature, cond_entry), dtype=np.float64)
    n_feats = lnorm.shape[-1]
    if acc is None:
        acc = {"sum": np.zeros(n_feats), "sumsq": np.zeros(n_feats), "count": 0,
               "species": {}, **_new_term_acc(n_feats)}
    if lnorm.ndim != 3 or not np.isfinite(lnorm).all():
        # A clip with a non-finite value is left out whole: the step and root
        # sums need its frames contiguous.
        return acc
    flat = lnorm.reshape(-1, n_feats)
    acc["sum"] += flat.sum(axis=0)
    acc["sumsq"] += (flat ** 2).sum(axis=0)
    acc["count"] += flat.shape[0]
    _accumulate_terms(acc, lnorm)
    if species is not None:
        _accumulate_terms(acc["species"].setdefault(species, _new_term_acc(n_feats)), lnorm)
    return acc


def finalize_lnorm_stats(acc):
    """Turn an accumulator from accumulate_lnorm_stats() into
    ``(mean, std, dpos_std)``: 12-vectors whose velocity block is the root's X/Z
    (the derived channels get mean 0 and a zero std, which pooling skips and
    set_canonical_global_stats() floors to one), and the position-step RMS.
    """
    if acc is None or acc["count"] <= 0 or acc["root_count"] <= 0:
        raise ValueError("No finite samples accumulated for canonical stats.")
    mean = acc["sum"] / acc["count"]
    var = np.maximum(acc["sumsq"] / acc["count"] - mean ** 2, 0.0)
    n_feats = mean.shape[0]
    if n_feats >= _VEL_SLICE.stop:
        root_mean = acc["root_sum"] / acc["root_count"]
        root_var = np.maximum(acc["root_sumsq"] / acc["root_count"] - root_mean ** 2, 0.0)
        mean[_VEL_SLICE] = 0.0
        var[_VEL_SLICE] = 0.0
        channels = list(ROOT_VEL_CHANNELS)
        mean[channels] = root_mean[channels]
        var[channels] = root_var[channels]
    dpos_std = np.sqrt(acc["dpos_sumsq"] / acc["dpos_count"]) if acc["dpos_count"] > 0 else 0.0
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32), np.float32(dpos_std)


def _term_energy(term_acc, mean, std, dpos_std):
    """A species' mean squared root X/Z velocity and position step, both in
    the global canonical units."""
    channels = list(ROOT_VEL_CHANNELS)
    n = term_acc["root_count"]
    root_energy = None
    if n > 0:
        m = np.asarray(mean, dtype=np.float64)[channels]
        s = np.asarray(std, dtype=np.float64)[channels]
        centered_sq = (
            term_acc["root_sumsq"][channels] - 2.0 * m * term_acc["root_sum"][channels] + n * m ** 2
        ) / n
        root_energy = float(np.mean(centered_sq / np.maximum(s, _STD_FLOOR) ** 2))
    step_energy = None
    if term_acc["dpos_count"] > 0:
        step_energy = term_acc["dpos_sumsq"] / term_acc["dpos_count"] / max(float(dpos_std), _STD_FLOOR) ** 2
    return root_energy, step_energy


def species_aux_loss_weights(accs, mean, std, dpos_std):
    """``{species: (root_vel_weight, dpos_weight)}`` from the per-species sums
    of one or more accumulators: ``min(1, sqrt(AUX_LOSS_ENERGY_CAP / energy))``."""
    def weight(energy):
        if energy is None or not np.isfinite(energy) or energy <= AUX_LOSS_ENERGY_CAP:
            return 1.0
        return float(np.sqrt(AUX_LOSS_ENERGY_CAP / energy))

    weights = {}
    for acc in accs:
        for species, term_acc in acc.get("species", {}).items():
            root_energy, step_energy = _term_energy(term_acc, mean, std, dpos_std)
            weights[species] = (weight(root_energy), weight(step_energy))
    return weights


def _block_std_scalar(std, block):
    """Mean of one block's finite, non-degenerate per-channel stds.

    Returns ``None`` when the block holds nothing usable (empty slice, or every
    channel constant), in which case the caller leaves that block untouched and
    the ``_STD_FLOOR`` path in set_canonical_global_stats() still applies.
    """
    values = np.asarray(std, dtype=np.float64).reshape(-1)[block]
    values = values[np.isfinite(values) & (values > _STD_FLOOR)]
    if values.size == 0:
        return None
    return float(values.mean())


def pool_global_stats(subset_stats):
    """Combine per-object_subset L-normalized stats into the one global frame.

    ``subset_stats``: ``{object_subset: (mean12, std12, dpos_std)}`` ->
    ``(mean12, std12, dpos_std)``.

    Every subset gets equal weight, whatever its clip count:

    * ``mean`` is the average of the subset means.
    * ``std``: within each subset, each block (position / rotation-6d /
      velocity) is collapsed to the mean of its per-channel stds, and the block
      scalar is the geometric mean of those subset scalars (a gain is
      multiplicative, so the geometric mean spreads every subset's resulting
      variance symmetrically around one in log space). ``dpos_std`` is pooled
      the same way.

    ``l_simple`` is computed in this standardized space, so ``1 / std`` is an
    implicit per-channel loss weight; a single scalar per block weights every
    joint and axis of the block uniformly.
    """
    if not subset_stats:
        raise ValueError("pool_global_stats needs at least one object_subset's stats.")

    means = [np.asarray(stats[0], dtype=np.float64).reshape(-1) for stats in subset_stats.values()]
    if len({m.shape for m in means}) != 1:
        raise ValueError("object_subset stats disagree on the feature width.")
    mean_out = np.mean(np.stack(means, axis=0), axis=0)

    std_out = np.ones_like(mean_out)
    n_feats = std_out.shape[0]
    for block in (_POS_SLICE, _ROT_SLICE, _VEL_SLICE):
        if block.start >= n_feats:
            continue
        scalars = [
            value for value in (_block_std_scalar(stats[1], block) for stats in subset_stats.values())
            if value is not None
        ]
        if not scalars:
            continue
        std_out[block.start:min(block.stop, n_feats)] = _geometric_mean(scalars)
    dpos_scalars = [float(stats[2]) for stats in subset_stats.values() if float(stats[2]) > _STD_FLOOR]
    dpos_out = _geometric_mean(dpos_scalars) if dpos_scalars else 1.0
    return mean_out.astype(np.float32), std_out.astype(np.float32), np.float32(dpos_out)


def _geometric_mean(values):
    return float(np.exp(np.mean(np.log(np.asarray(values, dtype=np.float64)))))


def assert_single_canonical_frame(cond_dict, source="cond.npy"):
    """Raise unless every entry of ``cond_dict`` carries the same mean/std and
    position-step RMS.

    The model is trained in one global canonical frame; a cond whose entries
    disagree was built by a per-object_subset (or otherwise stale) stats pass and
    would silently write some species in a frame the model never saw. Entries
    without stats are left to the callers' own missing-stats checks.
    """
    frames = {}
    for key, entry in cond_dict.items():
        stats = get_canonical_global_stats(entry) if isinstance(entry, dict) else None
        if stats is None:
            continue
        frame = tuple(
            np.asarray(stat, dtype=np.float32).reshape(-1).tobytes()
            for stat in (*stats, get_canonical_dpos_std(entry) if get_canonical_dpos_std(entry) is not None else np.nan)
        )
        frames.setdefault(frame, []).append(key)
    if len(frames) > 1:
        examples = ", ".join(sorted(keys)[0] for keys in frames.values())
        raise ValueError(
            f"{source} holds {len(frames)} different canonical_feature_mean/std frames "
            f"(e.g. {examples}); every species must share the one global frame. "
            "Regenerate it (tools/regenerate_dataset_artifacts.py for a dataset, "
            "tools/merge_dataset_cond.py for the merged training cond)."
        )


def mark_canonical_cond_entry(cond_entry):
    cond_entry["feature_space"] = CANONICAL_FEATURE_SPACE
    cond_entry["physical_feature_space"] = PHYSICAL_FEATURE_SPACE
    if "rest_pose" in cond_entry:
        cond_entry["rest_pose"] = np.asarray(cond_entry["rest_pose"], dtype=np.float32)
        cond_entry["rest_pos_ric_hml"] = cond_entry["rest_pose"][:, 0:3].astype(np.float32, copy=True)
    return cond_entry


def validate_feature_space(cond_entry, expected=CANONICAL_FEATURE_SPACE):
    actual = cond_entry.get("feature_space")
    if actual != expected:
        raise ValueError(
            f"Expected cond feature_space={expected!r}, got {actual!r}. "
            "Regenerate cond.npy with canonical motion features."
        )
    if get_canonical_global_stats(cond_entry) is None:
        raise ValueError(
            f"cond entry is missing {CANONICAL_MEAN_KEY!r}/{CANONICAL_STD_KEY!r}. "
            "Regenerate cond.npy (regenerate_dataset_artifacts) to compute the "
            "global canonical standardization statistics."
        )
    if get_canonical_dpos_std(cond_entry) is None:
        raise ValueError(
            f"cond entry is missing {CANONICAL_DPOS_STD_KEY!r}: it was built before velocity "
            "was kept on the root alone. Regenerate cond.npy (tools/merge_dataset_cond.py "
            "for the merged training cond, regenerate_dataset_artifacts for a dataset)."
        )
    return True
