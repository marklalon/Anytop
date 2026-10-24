"""Canonical AnyTop motion feature conversion helpers.

The physical feature layout is HML-like (RIC position, each joint's own local
rotation relative to the rest pose, local velocity). The *canonical* model
feature space is reached via three prior-free, exactly-invertible steps. They
read the rest only as topology, bone lengths and per-joint bone frames, which a
re-posed rest turns along with the motion, so one world motion encodes to the
same canonical features whichever rest convention (T-pose, A-pose, ...) the rig
was authored in:

  1. Rotation re-expression: each joint's local rotation becomes its
     bone-aligned global rotation ``G_j @ A_j`` (see "Rotation channel").
  2. Per-skeleton size normalization (L-scaling): the position channels are
     divided by the skeleton's own geometric length ``L`` (the
     joints' RMS spread measured along the bone tree, so bone lengths and
     topology decide it and the rest pose does not) so that every species lands
     in a size-free space (cross-species fair). Rotation (6d) is
     size-independent and is left untouched here. ``L`` is floored at
     ``REST_LENGTH_SCALE_FLOOR``: a rig of a few clustered joints (MU04_Pollen)
     has almost no joint spread, and dividing its body-scale root trajectory by
     that spread blew a fall up to tens of std.
  3. Global standardization: subtract a per-channel mean and divide by a
     *block-collapsed* std (see "Statistics layout"). One pair of vectors for the whole
     corpus, computed at preprocessing time in the L-normalized space of step
     (2) by :func:`pool_global_stats`: clips are first pooled per object_subset,
     then the subsets are combined with equal weight so the largest subsets do
     not set the scale for the rest. Every species is written in the same
     canonical frame, so the frame carries no body-plan information and the
     same physical motion has the same feature amplitude on every skeleton.
     The vectors are stored on every cond entry (``canonical_feature_mean`` /
     ``canonical_feature_std``) and are identical across entries; a new
     species copies them from any entry of the training cond.

     Each block (position / rotation-6d) of ``std`` is one scalar, so
     ``l_simple`` weights every joint and axis of a block uniformly.

Channel layout per joint (n_feats == 12):
    0:3   position   (RIC: world position minus the translation root's XZ;
                      the root itself holds its world position, see "Root")
    3:9   rotation 6d (bone-aligned global rotation)
    9:12  velocity   (zero in the working layout, see "Velocity")

Root. A stored clip (preprocessing's ``.npy``) keeps the translation root's
X/Z at zero and its world XZ path in the velocity rows. The loader turns it
into the working layout before any temporal stage (:func:`stored_to_working`):
the root's ch0/ch2 become its world XZ, integrated from frame 0, which
preprocessing centres on the origin. Every other joint stays RIC, so its world
position is its RIC position plus the root's XZ (:func:`root_relative_positions`
is the all-RIC view). The model reads and writes the root's position like any
joint's; only its standardization differs: the root's world XZ spreads far less
than the joints' RIC positions, so the root's position takes its own mean and
std, one scalar shared by X and Z (the horizontal plane stays isotropic) and
one for Y.

Statistics layout. ``canonical_feature_mean`` / ``canonical_feature_std`` are
``FEATS_LEN + 3`` vectors: the per-channel statistics of every joint, then the
root position's (``ROOT_STAT_SLICE``), which replace channels 0:3 on joint 0.

Velocity. Every velocity is a position step, so the working layout carries
none: its velocity channels are zero, every temporal stage interpolates
positions and rotations only, and the velocity statistics are the identity.
:func:`working_to_stored` writes a working clip back in the stored layout.

Rotation channel. A physical rotation is identity AT THE REST POSE, so the same
world motion stored on a T-pose rig and on an A-pose rig of the same skeleton
carries different rotations. The canonical channel stores
``G_j @ A_j`` instead, where ``G_j`` is the joint's global rotation (the local
rotations composed along the chain; the rest offsets are world-aligned) and
``A_j`` is the joint's rest bone frame (:func:`rest_bone_frames`): column 0 the
rest bone direction ``d_j``, column 1 a twist reference ``p_j`` orthogonal to
it, column 2 their cross product. The first 6d column is therefore the joint's
current world bone direction and the second its current twist reference. A
rest that turns joint ``j``'s frame by ``Q`` turns ``d_j`` by ``Q`` and turns
``G_j`` by ``Q^T``; whenever ``Q`` also carries ``p_j`` onto the new rest's
``p_j`` (any turn in the plane of ``d_j`` and the twist reference axis, e.g.
an arm lowered from T to A about Z, a leg swung about X) the canonical rotation
is exactly unchanged. Decoding reads the TARGET rig's frames, so a decoded clip
is re-expressed against that rig's own rest.

"""

from __future__ import annotations

import functools

import numpy as np

from data_loaders.truebones.truebones_utils.param_utils import (
    HML_REF_REST_LENGTH_SCALE,
    ancestor_matrix,
    joint_depths,
    rest_tree_spread,
)

try:
    import torch
except Exception:  # pragma: no cover - torch is present in training/runtime.
    torch = None


CANONICAL_FEATURE_SPACE = "canonical_motion_v6"
PHYSICAL_FEATURE_SPACE = "hml_like_v_current"

# Keys under which the global per-channel standardization statistics are stored
# on each cond entry (see "Statistics layout"). Every entry carries the same
# vectors. Computed at
# preprocessing time over the L-normalized training distribution -- see
# ``pool_global_stats``.
CANONICAL_MEAN_KEY = "canonical_feature_mean"
CANONICAL_STD_KEY = "canonical_feature_std"
# RMS of one frame's L-normalized position step, pooled like the block stds. The
# position-step loss divides by it so its weight means the same on every corpus.
CANONICAL_DPOS_STD_KEY = "canonical_dpos_std"

# Per-species weight of the position-step loss: a species whose own energy in
# that term exceeds ``AUX_LOSS_ENERGY_CAP`` (in the global canonical units) is
# weighted by sqrt(cap / energy), never up. This softens the influence of fast
# movers without capping their weighted loss energy.
DPOS_LOSS_WEIGHT_KEY = "dpos_loss_weight"
AUX_LOSS_ENERGY_CAP = 2.0

# Cond keys measured in the canonical space; they are only valid under the
# feature space stamped on the same entry.
_FEATURE_SPACE_BOUND_KEYS = (
    CANONICAL_MEAN_KEY,
    CANONICAL_STD_KEY,
    CANONICAL_DPOS_STD_KEY,
    DPOS_LOSS_WEIGHT_KEY,
)

# The model's channel contract (see "Velocity" above), the same on every joint.
# Joint 0 is the translation root of every cond skeleton.
MODEL_CHANNELS = tuple(range(0, 9))
# The root's world X/Z channels (see "Root").
ROOT_XZ_CHANNELS = (0, 2)

# Per-sample geometric length ``L`` (``[B]``) that the training collate
# precomputes from each sample's unpadded rig and stores in ``y``; see
# ``_length_scale_from_cond``. Never stored in cond.npy (species entries derive
# it from ``rest_pos_ric_hml`` and ``parents`` on the fly).
REST_LENGTH_SCALE_KEY = "rest_length_scale"

# Lower bound on ``L``: a fraction of the reference skeleton's own ``L``.
# Preprocessing has already scaled every character to the reference body size
# (its extent folds in the root's elevation, which the joint spread never sees),
# so an ``L`` far below the reference means clustered joints, not a small body.
# The fraction is set so that only the few-joint effect / drifting rigs reach it;
# short-boned bodies such as fish stay above.
REST_LENGTH_SCALE_FLOOR = 0.35 * HML_REF_REST_LENGTH_SCALE

# Per-sample rest bone frames ``[B, J, 3, 3]`` and parent indices ``[B, J]``
# (-1 for a root and for padding) that the training collate precomputes from
# each sample's unpadded rig, so the training-time decode of the rotation
# channel never rebuilds them; see ``_rotation_context``. Never stored in
# cond.npy (species entries derive them from ``parents`` and the rest).
REST_BONE_FRAME_KEY = "rest_bone_frame"
PARENT_INDEX_KEY = "parent_index"

# Twist reference axes of ``rest_bone_frames`` in order of preference: the first
# one not within ``_TWIST_REFERENCE_MAX_COS`` of the bone direction is used.
_TWIST_REFERENCE_AXES = ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
_TWIST_REFERENCE_MAX_COS = 0.95
# A subtree-weighted mean of child directions shorter than this has cancelled
# out (opposed children of equal weight) and names no direction.
_MIN_CHILD_DIRECTION_NORM = 0.3
# Rest bones shorter than this fraction of ``L`` carry no direction.
_MIN_BONE_FRAC_OF_L = 1e-3
# Rigs whose ``L`` and rest bone frames stay cached per process: every stored
# rig plus the recent augmented ones.
_RIG_GEOMETRY_CACHE_SIZE = 2048

# Channels whose std falls below this floor are treated as unit-variance so the
# encode divide never explodes (mirrors the old std_safe floor).
_STD_FLOOR = 1e-5

_POS_SLICE = slice(0, 3)
_ROT_SLICE = slice(3, 9)
_VEL_SLICE = slice(9, 12)
# Where the root position's statistics sit in the stat vectors, and their length.
ROOT_STAT_SLICE = slice(_VEL_SLICE.stop, _VEL_SLICE.stop + 3)
CANONICAL_STAT_LEN = ROOT_STAT_SLICE.stop


def _is_torch_tensor(value) -> bool:
    return torch is not None and torch.is_tensor(value)


_MASK_TENSOR_CACHE = {}


def _channel_mask(n_joints, n_feats, which):
    """``[J, F]`` float mask, 0 on the channels ``which`` names, 1 elsewhere:
    ``"velocity"`` is every joint's velocity block, ``"root_xz"`` the root's
    world X/Z."""
    mask = np.ones((int(n_joints), int(n_feats)), dtype=np.float32)
    if which == "velocity":
        if n_feats >= _VEL_SLICE.stop:
            mask[:, _VEL_SLICE] = 0.0
    elif n_joints > 0:
        mask[0, list(ROOT_XZ_CHANNELS)] = 0.0
    return mask


def _apply_channel_mask(feature, which):
    """``feature`` times :func:`_channel_mask`. Accepts ``[B, J, F, T]`` (torch
    or numpy) or channels-last ``[..., J, F]``; never writes the input."""
    four_d = (feature.dim() if _is_torch_tensor(feature) else np.ndim(feature)) == 4
    shape = feature.shape
    n_joints, n_feats = (shape[1], shape[2]) if four_d else (shape[-2], shape[-1])
    if _is_torch_tensor(feature):
        key = (int(n_joints), int(n_feats), which, str(feature.device), feature.dtype)
        mask = _MASK_TENSOR_CACHE.get(key)
        if mask is None:
            mask = torch.as_tensor(
                _channel_mask(n_joints, n_feats, which), device=feature.device, dtype=feature.dtype
            )
            _MASK_TENSOR_CACHE[key] = mask
        return feature * (mask.view(1, n_joints, n_feats, 1) if four_d else mask)

    out = np.asarray(feature, dtype=np.float32)
    mask = _channel_mask(n_joints, n_feats, which)
    return (out * (mask[None, :, :, None] if four_d else mask)).astype(np.float32, copy=False)


def zero_velocity(feature):
    """Zero every velocity channel (see "Velocity")."""
    return _apply_channel_mask(feature, "velocity")


def root_relative_positions(feature):
    """The all-RIC view of a working feature: the root's world X/Z zeroed, so
    every joint's position is relative to the root's XZ (see "Root"). Parent
    to child differences and per-joint heights read this view."""
    return _apply_channel_mask(feature, "root_xz")


def stored_to_working(motion):
    """A stored ``[T, J, F]`` clip in the working layout (see "Root").

    The root's ch0/ch2 take its world XZ: the sum of the root velocity rows
    before each frame, zero at frame 0. Every velocity channel is then zeroed.
    """
    out = np.array(motion, dtype=np.float32, copy=True)
    if out.ndim != 3 or out.shape[-1] < _VEL_SLICE.stop or out.shape[0] == 0:
        return out
    vel_xz = [_VEL_SLICE.start + c for c in ROOT_XZ_CHANNELS]
    steps = out[:-1, 0, vel_xz].astype(np.float64)
    world_xz = np.zeros((out.shape[0], 2), dtype=np.float64)
    world_xz[1:] = np.cumsum(steps, axis=0)
    out[:, 0, list(ROOT_XZ_CHANNELS)] = world_xz
    out[..., _VEL_SLICE] = 0.0
    return out


def working_to_stored(motion, *, is_loop):
    """A working ``[T, J, F]`` clip in the stored layout, one row per frame.

    The root's position goes back to RIC (ch0/ch2 zero) and its velocity rows
    take its world steps: row ``t`` is the step from frame ``t`` to ``t + 1``,
    the terminal row a loop's wrap step, otherwise the last step repeated.
    That is the only velocity a stored-clip reader uses (the exporter
    integrates it into the trajectory, the loop detector reads its closure);
    every other joint's velocity row stays zero.
    """
    out = np.array(motion, dtype=np.float32, copy=True)
    n_frames = out.shape[0]
    if out.ndim != 3 or out.shape[-1] < _VEL_SLICE.stop or n_frames == 0:
        return out
    root = np.asarray(out[:, 0, _POS_SLICE], dtype=np.float64)
    step = np.zeros_like(root)
    step[:-1] = root[1:] - root[:-1]
    if is_loop or n_frames == 1:
        step[-1] = root[0] - root[-1]
    else:
        step[-1] = step[-2]
    out[..., _VEL_SLICE] = 0.0
    out[:, 0, _VEL_SLICE] = step.astype(np.float32)
    out[:, 0, list(ROOT_XZ_CHANNELS)] = 0.0
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


def _length_scale_from_rest(rest_pos, parents):
    """Per-skeleton geometric length ``L`` of one rig: its tree RMS spread
    (``param_utils.rest_tree_spread``), floored at ``REST_LENGTH_SCALE_FLOOR``.

    ``rest_pos`` is ``[J, 3]`` (rows past ``len(parents)`` are padding and
    ignored). Derived purely from skeleton geometry -- no motion stats. A
    collapsed or non-finite rest lands on the floor.
    """
    return _rig_geometry(*_rig_key(rest_pos, parents))[0]


def _rig_key(rest_pos, parents):
    """Hashable ``(rest bytes, parents bytes)`` of one rig for ``_rig_geometry``:
    the unpadded rest as float64, so float32 and float64 copies of it agree."""
    if _is_torch_tensor(rest_pos):
        rest_pos = rest_pos.detach().cpu().numpy()
    parents = np.ascontiguousarray(np.asarray(parents, dtype=np.int64).reshape(-1))
    rest_pos = np.ascontiguousarray(np.asarray(rest_pos, dtype=np.float64)[:parents.size, :3])
    return rest_pos.tobytes(), parents.tobytes()


# A training sample derives its rig's geometry several times (target and
# reference encode, collate), and most samples use their species' stored rig.
@functools.lru_cache(maxsize=_RIG_GEOMETRY_CACHE_SIZE)
def _rig_geometry(rest_bytes, parents_bytes):
    """``(L, rest bone frames [J, 3, 3] read-only)`` of the rig a ``_rig_key`` names."""
    parents = np.frombuffer(parents_bytes, dtype=np.int64)
    rest_pos = np.frombuffer(rest_bytes, dtype=np.float64).reshape(parents.size, 3)
    L = rest_tree_spread(rest_pos, parents)
    L = max(L if np.isfinite(L) else 0.0, REST_LENGTH_SCALE_FLOOR)
    frames = _bone_frames(rest_pos, parents, L)
    frames.setflags(write=False)
    return L, frames


def _spatial_L_vectors(n_feats):
    """Return a per-channel ``uses_L`` vector of length ``n_feats``.

    ``uses_L[c] == 1`` for the position group (size-dependent) and ``0``
    elsewhere (rotation is size-independent, velocity is zero). The L-scaling
    divides those channels by the per-skeleton length ``L`` (encode) /
    multiplies (decode).
    """
    uses_L = np.zeros(n_feats, dtype=np.float64)
    uses_L[_POS_SLICE] = 1.0
    return uses_L


def _length_scale_from_cond(cond_entry, *, like=None):
    """Resolve the per-skeleton length ``L`` for a cond entry / collated ``y``.

    The collate precomputes ``rest_length_scale`` (``[B]``) from each sample's
    unpadded rig, so the training aux-loss decode reads it instead of
    re-deriving it every step. Without it, ``L`` is derived from the rest
    positions and ``parents``: one rig's array gives a scalar, a collated
    per-sample list gives ``[B]`` over the padded ``[B, J, 3]`` rest.
    """
    L = cond_entry.get(REST_LENGTH_SCALE_KEY) if hasattr(cond_entry, "get") else None
    if L is not None:
        return L
    parents = cond_entry.get("parents") if hasattr(cond_entry, "get") else None
    if parents is None:
        raise KeyError(
            "cond entry is missing 'parents' (and no precomputed rest_length_scale): "
            "L is measured along the skeleton's bone tree."
        )
    rest = rest_pos_from_cond(cond_entry)
    rest = np.asarray(rest.detach().cpu() if _is_torch_tensor(rest) else rest, dtype=np.float64)
    if isinstance(parents, (list, tuple)) and len(parents) > 0 and np.ndim(parents[0]) > 0:
        rest_rows = rest if rest.ndim == 3 else np.broadcast_to(rest, (len(parents),) + rest.shape)
        L = np.array([
            _length_scale_from_rest(rest_rows[b], np.asarray(p).reshape(-1))
            for b, p in enumerate(parents)
        ])
    else:
        L = _length_scale_from_rest(rest[0] if rest.ndim == 3 else rest, parents)
    if _is_torch_tensor(like):
        return torch.as_tensor(L, device=like.device, dtype=torch.float32)
    return L


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
    """Divide (encode) or multiply (decode) the position channels by the
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
    """Store the global mean/std (see "Statistics layout") and the
    position-step RMS on a cond entry, flooring near-constant channels to unit
    std so the encode divide is safe.
    """
    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    std = np.asarray(std, dtype=np.float32).reshape(-1)
    _check_root_stats(min(mean.size, std.size))
    std = np.where(std < _STD_FLOOR, 1.0, std).astype(np.float32, copy=False)
    dpos_std = float(dpos_std)
    if not np.isfinite(dpos_std) or dpos_std < _STD_FLOOR:
        dpos_std = 1.0
    cond_entry[CANONICAL_MEAN_KEY] = mean
    cond_entry[CANONICAL_STD_KEY] = std
    cond_entry[CANONICAL_DPOS_STD_KEY] = np.float32(dpos_std)
    return cond_entry


def set_aux_loss_weights(cond_entry, dpos_weight=1.0):
    """Store a species' position-step loss weight."""
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
        _check_root_stats(mean_t.shape[-1])
        mean_v = _view_stat_torch(mean_t, out.shape, n_feats)
        std_v = _view_stat_torch(std_t, out.shape, n_feats)
        out = (out * std_v) + mean_v if inverse else (out - mean_v) / std_v
        root_mean = _view_root_stat(mean_t, ndim)
        root_std = _view_root_stat(std_t, ndim)
        if ndim == 4:
            root = feature[:, 0:1, _POS_SLICE]
            root = root * root_std + root_mean if inverse else (root - root_mean) / root_std
            return torch.cat([torch.cat([root, out[:, 0:1, _POS_SLICE.stop:]], dim=2), out[:, 1:]], dim=1)
        root = feature[..., 0:1, _POS_SLICE]
        root = root * root_std + root_mean if inverse else (root - root_mean) / root_std
        return torch.cat([torch.cat([root, out[..., 0:1, _POS_SLICE.stop:]], dim=-1), out[..., 1:, :]], dim=-2)

    out = np.asarray(feature, dtype=np.float32)
    ndim = out.ndim
    n_feats = out.shape[2] if ndim == 4 else out.shape[-1]
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    _check_root_stats(mean.shape[-1])
    mean_v = _view_stat_numpy(mean, out.shape, n_feats)
    std_v = _view_stat_numpy(std, out.shape, n_feats)
    result = (out * std_v) + mean_v if inverse else (out - mean_v) / std_v
    root_mean = _view_root_stat(mean, ndim)
    root_std = _view_root_stat(std, ndim)
    root = out[:, 0:1, _POS_SLICE] if ndim == 4 else out[..., 0:1, _POS_SLICE]
    root = root * root_std + root_mean if inverse else (root - root_mean) / root_std
    if ndim == 4:
        result[:, 0:1, _POS_SLICE] = root
    else:
        result[..., 0:1, _POS_SLICE] = root
    return result.astype(np.float32, copy=False)


def _check_root_stats(width):
    if width < ROOT_STAT_SLICE.stop:
        raise ValueError(
            f"canonical stats have {width} entries, the root position's sit at "
            f"{ROOT_STAT_SLICE.start}:{ROOT_STAT_SLICE.stop} (see 'Statistics layout'). "
            "Regenerate cond.npy."
        )


def _view_root_stat(stat, feature_ndim):
    """The root position's stat (``ROOT_STAT_SLICE``), shaped to broadcast
    against joint 0's position slice of a feature of rank ``feature_ndim``:
    ``[N, 1, 3, 1]`` for ``[B, J, F, T]``, ``[1, 3]`` for channels-last."""
    root = stat[..., ROOT_STAT_SLICE]
    if feature_ndim == 4:
        rows = root.reshape(-1, 3)
        return rows.reshape(rows.shape[0], 1, 3, 1)
    return root.reshape(1, 3)


def _twist_references(directions):
    """``[K, 3]`` unit twist references ``p`` for unit directions ``[K, 3]``."""
    axes = np.asarray(_TWIST_REFERENCE_AXES, dtype=np.float64)
    cos = directions @ axes.T
    usable = np.abs(cos) <= _TWIST_REFERENCE_MAX_COS
    # The first usable axis; the last one when none is.
    choice = np.where(usable.any(axis=1), usable.argmax(axis=1), len(axes) - 1)
    reference = axes[choice] - cos[np.arange(choice.size), choice][:, None] * directions
    return reference / np.linalg.norm(reference, axis=-1, keepdims=True)


def _frames_from_directions(directions):
    """``[K, 3, 3]`` frames with columns ``d``, twist reference ``p``, ``d x p``
    for unit directions ``[K, 3]``."""
    reference = _twist_references(directions)
    return np.stack([directions, reference, np.cross(directions, reference)], axis=-1)


def rest_bone_frames(rest_pos, parents):
    """``[J, 3, 3]`` rest bone frame ``A_j`` of every joint (see "Rotation channel").

    Column 0 is the rest bone direction ``d_j``: the mean of the directions to
    the joint's children, each weighted by the size of its subtree, so a
    mirrored pair of children cancels sideways and a chain's continuation
    outweighs a short branch. A leaf, or a joint whose children cancel out,
    takes the direction of its own bone from its parent; a joint with neither
    (a zero-length nub, a lone root) inherits its parent's frame, identity at a
    root. Column 1 is the first ``_TWIST_REFERENCE_AXES`` axis not within
    ``_TWIST_REFERENCE_MAX_COS`` of ``d_j``, made orthogonal to it; column 2 is
    ``d_j x p_j``.

    The weights are topology and the directions are rest offsets, which a
    joint's rest frame turns rigidly -- that is what keeps the canonical
    rotation independent of the rest pose.
    """
    return _rig_geometry(*_rig_key(rest_pos, parents))[1].copy()


def _bone_frames(rest_pos, parents, length_scale):
    """:func:`rest_bone_frames` of an unpadded float64 rest whose ``L`` is known."""
    n_joints = parents.size
    ancestors, depths = ancestor_matrix(parents)
    has_parent = parents >= 0
    subtree = ancestors.sum(axis=0).astype(np.float64)

    offsets = np.zeros_like(rest_pos)
    offsets[has_parent] = rest_pos[has_parent] - rest_pos[parents[has_parent]]
    lengths = np.linalg.norm(offsets, axis=-1)
    min_len = _MIN_BONE_FRAC_OF_L * length_scale
    live_bone = has_parent & (lengths > min_len)
    unit = np.zeros_like(offsets)
    unit[live_bone] = offsets[live_bone] / lengths[live_bone, None]

    weighted = np.zeros_like(offsets)
    weight = np.zeros(n_joints, dtype=np.float64)
    live = np.flatnonzero(live_bone)
    np.add.at(weighted, parents[live], subtree[live, None] * unit[live])
    np.add.at(weight, parents[live], subtree[live])

    mean = np.divide(weighted, weight[:, None], out=np.zeros_like(weighted), where=weight[:, None] > 0.0)
    mean_norm = np.linalg.norm(mean, axis=-1)
    from_children = (weight > 0.0) & (mean_norm >= _MIN_CHILD_DIRECTION_NORM)
    has_direction = from_children | live_bone
    direction = np.where(
        from_children[:, None], mean / np.maximum(mean_norm, _MIN_CHILD_DIRECTION_NORM)[:, None], unit
    )
    own = np.repeat(np.eye(3)[None], n_joints, axis=0)
    own[has_direction] = _frames_from_directions(direction[has_direction])

    # A joint without a direction takes the frame of its deepest ancestor that
    # has one, identity when none does.
    holders = ancestors & has_direction[None, :]
    source = np.where(holders, depths[None, :], -1).argmax(axis=1)
    frames = np.where(holders.any(axis=1)[:, None, None], own[source], np.eye(3))
    return frames.astype(np.float32)


def padded_rotation_context(rest_pos, parents, n_slots):
    """``(frames [n_slots, 3, 3], parent_index [n_slots])`` of one rig.

    Slots past the rig's joints are padding: identity frame, parent -1.
    """
    parents = np.asarray(parents, dtype=np.int64).reshape(-1)
    n_joints = parents.size
    if n_joints > n_slots:
        raise ValueError(f"rig has {n_joints} joints but the feature has {n_slots} slots.")
    frames = np.repeat(np.eye(3, dtype=np.float32)[None], n_slots, axis=0)
    frames[:n_joints] = rest_bone_frames(rest_pos, parents)
    parent_index = np.full(n_slots, -1, dtype=np.int64)
    parent_index[:n_joints] = parents
    return frames, parent_index


def _rotation_context(cond_entry, like, n_joints):
    """Rest bone frames ``[N, J, 3, 3]`` and parent indices ``[N, J]`` for a
    feature with ``n_joints`` slots, numpy or tensors on ``like``'s device.

    Reads the collate's ``REST_BONE_FRAME_KEY`` / ``PARENT_INDEX_KEY`` when
    present. Otherwise ``parents`` is either one rig's array (a cond entry,
    ``N == 1``) or a collated per-sample list over a padded ``[B, J, 3]`` rest.
    """
    getter = cond_entry.get if hasattr(cond_entry, "get") else (lambda key: None)
    frames, parent_index = getter(REST_BONE_FRAME_KEY), getter(PARENT_INDEX_KEY)
    if frames is None or parent_index is None:
        parents = getter("parents")
        if parents is None:
            raise KeyError(
                "cond entry is missing 'parents': the canonical rotation channel is "
                "the bone-aligned global rotation and needs the skeleton's chain."
            )
        rest = rest_pos_from_cond(cond_entry)
        rest = np.asarray(rest.detach().cpu() if _is_torch_tensor(rest) else rest, dtype=np.float32)
        if isinstance(parents, (list, tuple)) and len(parents) > 0 and np.ndim(parents[0]) > 0:
            rest_rows = rest if rest.ndim == 3 else np.broadcast_to(rest, (len(parents),) + rest.shape)
            pairs = [
                padded_rotation_context(rest_rows[b], np.asarray(p).reshape(-1), n_joints)
                for b, p in enumerate(parents)
            ]
            frames = np.stack([pair[0] for pair in pairs])
            parent_index = np.stack([pair[1] for pair in pairs])
        else:
            frames, parent_index = padded_rotation_context(
                rest[0] if rest.ndim == 3 else rest, parents, n_joints
            )

    if _is_torch_tensor(like):
        frames = torch.as_tensor(frames).to(device=like.device, dtype=like.dtype)
        parent_index = torch.as_tensor(parent_index).to(device=like.device, dtype=torch.long)
        if frames.dim() == 3:
            frames, parent_index = frames.unsqueeze(0), parent_index.reshape(1, -1)
        return frames[:, :n_joints], parent_index[:, :n_joints]
    if _is_torch_tensor(frames):
        frames, parent_index = frames.detach().cpu().numpy(), parent_index.detach().cpu().numpy()
    frames = np.asarray(frames, dtype=np.float64)
    parent_index = np.asarray(parent_index, dtype=np.int64)
    if frames.ndim == 3:
        frames, parent_index = frames[None], parent_index.reshape(1, -1)
    return frames[:, :n_joints], parent_index[:, :n_joints]


def _rotation_view(feature):
    """The rotation channels of any feature rank as ``[N, T, J, 6]``."""
    ndim = feature.dim() if _is_torch_tensor(feature) else feature.ndim
    if ndim == 4:
        rot = feature[:, :, _ROT_SLICE, :]
        return rot.permute(0, 3, 1, 2) if _is_torch_tensor(rot) else rot.transpose(0, 3, 1, 2)
    if ndim == 3:
        return feature[None, :, :, _ROT_SLICE]
    if ndim == 2:
        return feature[None, None, :, _ROT_SLICE]
    raise ValueError(f"Unsupported feature rank {ndim} for canonical conversion.")


def _write_rotation_view(out, rot6d):
    """Inverse of ``_rotation_view``: store ``[N, T, J, 6]`` back into ``out``."""
    ndim = out.dim() if _is_torch_tensor(out) else out.ndim
    if ndim == 4:
        out[:, :, _ROT_SLICE, :] = rot6d.permute(0, 2, 3, 1) if _is_torch_tensor(rot6d) \
            else rot6d.transpose(0, 2, 3, 1)
    elif ndim == 3:
        out[..., _ROT_SLICE] = rot6d[0]
    else:
        out[..., _ROT_SLICE] = rot6d[0, 0]
    return out


def _gather_parent(matrices, parent_index):
    """``matrices[n, t, parent_index[n, j]]`` of ``[N, T, J, 3, 3]`` matrices; a
    root slot reads itself."""
    if _is_torch_tensor(matrices):
        own = torch.arange(parent_index.shape[1], device=parent_index.device).expand_as(parent_index)
        index = torch.where(parent_index >= 0, parent_index, own)
        n_batch = max(matrices.shape[0], index.shape[0])
        index = index.expand(n_batch, -1).reshape(n_batch, 1, -1, 1, 1)
        index = index.expand(-1, matrices.shape[1], -1, 3, 3)
        return torch.gather(matrices.expand(n_batch, -1, -1, -1, -1), 2, index)
    own = np.arange(parent_index.shape[1])[None]
    index = np.where(parent_index >= 0, parent_index, own)
    return np.take_along_axis(matrices, index[:, None, :, None, None], axis=2)


def _apply_rotation_frames(feature, cond_entry, inverse: bool):
    """Physical local rotation (encode) <-> canonical bone-aligned global
    rotation (decode, ``inverse=True``).

    Encode composes the local rotations down each chain into ``G_j``, one step
    per chain depth, and right-multiplies the rest bone frame. Decode removes
    the frame and reads each joint's local rotation off its parent's global
    one, in parallel over joints.
    """
    from utils.rotation_conversions import rotation_6d_to_matrix_np, rotation_6d_to_matrix_safe

    is_torch = _is_torch_tensor(feature)
    out = feature.clone() if is_torch else np.asarray(feature, dtype=np.float32).copy()
    n_feats = out.shape[2] if out.ndim == 4 else out.shape[-1]
    if n_feats < _ROT_SLICE.stop:
        return out
    # A copy, not a view: ``out`` is written in place below, and the rotation
    # conversion keeps its input for the backward pass.
    rot6d = _rotation_view(out)
    rot6d = rot6d.clone() if is_torch else rot6d.copy()
    frames, parent_index = _rotation_context(cond_entry, out, rot6d.shape[2])
    if frames.shape[0] not in (1, rot6d.shape[0]):
        raise ValueError(
            f"rotation context has {frames.shape[0]} rigs but the feature batch has "
            f"{rot6d.shape[0]} samples; slice the collated y per sample when decoding one."
        )
    frames = frames[:, None]                                            # [N, 1, J, 3, 3]
    if is_torch:
        matrices = rotation_6d_to_matrix_safe(rot6d)
        transpose, where = (lambda m: m.transpose(-1, -2)), torch.where
    else:
        matrices = rotation_6d_to_matrix_np(rot6d.astype(np.float64))
        transpose, where = (lambda m: np.swapaxes(m, -1, -2)), np.where

    if inverse:
        global_rot = matrices @ transpose(frames)
        local = transpose(_gather_parent(global_rot, parent_index)) @ global_rot
        result = where((parent_index < 0)[:, None, :, None, None], global_rot, local)
    else:
        parents_host = parent_index.detach().cpu().numpy() if is_torch else parent_index
        depths = np.stack([joint_depths(row) for row in parents_host])
        if is_torch:
            global_rot = matrices
            for level in range(1, int(depths.max(initial=0)) + 1):
                at_level = torch.as_tensor(depths == level, device=out.device)
                composed = _gather_parent(global_rot, parent_index) @ matrices
                global_rot = where(at_level[:, None, :, None, None], composed, global_rot)
        else:
            # In place, one chain level at a time, touching only that level's
            # joints; each rig row composes its own chains.
            global_rot = matrices.copy()
            shared = depths.shape[0] == 1
            for row in range(depths.shape[0]):
                rows = slice(None) if shared else slice(row, row + 1)
                for level in range(1, int(depths[row].max(initial=0)) + 1):
                    joints = np.flatnonzero(depths[row] == level)
                    global_rot[rows, :, joints] = (
                        global_rot[rows, :, parents_host[row, joints]] @ matrices[rows, :, joints]
                    )
        result = global_rot @ frames

    if is_torch:
        result6d = torch.cat([result[..., :, 0], result[..., :, 1]], dim=-1).to(out.dtype)
    else:
        result6d = np.concatenate([result[..., :, 0], result[..., :, 1]], axis=-1).astype(np.float32)
    return _write_rotation_view(out, result6d)


def physical_hml_to_lnorm(feature, cond_entry):
    """Encode working-layout physical features into the *L-normalized*
    intermediate space (steps 1-2): zero the velocity, re-express the rotation
    channel as the bone-aligned global rotation, then divide the position
    channels by the per-skeleton length ``L``. This is the space in which the
    GLOBAL standardization statistics are calibrated.
    """
    rotated = _apply_rotation_frames(zero_velocity(feature), cond_entry, inverse=False)
    return _apply_L_scale(rotated, cond_entry, inverse=False)


def physical_hml_to_canonical(feature, cond_entry):
    """Encode working-layout physical features into canonical model features.

    Encode order: zero the velocity, re-express the rotation channel
    (bone-aligned global), divide the position channels by ``L``, standardize
    every channel by the global mean/std, then zero the velocity again.
    """
    lnorm = physical_hml_to_lnorm(feature, cond_entry)
    return zero_velocity(_apply_global_stats(lnorm, cond_entry, inverse=False))


def canonical_to_physical_hml(feature, cond_entry):
    """Decode canonical model features back to working-layout physical features.

    Decode order (inverse of encode on every kept channel): de-standardize by
    the global mean/std, multiply the position channels by ``L``, then turn the
    rotation channel back into local rotations against ``cond_entry``'s own
    rest. The velocity channels come back as zero;
    :func:`working_to_stored` writes the stored layout.
    """
    lnorm = _apply_global_stats(feature, cond_entry, inverse=True)
    unscaled = _apply_L_scale(lnorm, cond_entry, inverse=True)
    return zero_velocity(_apply_rotation_frames(unscaled, cond_entry, inverse=True))


def _new_step_acc():
    return {"dpos_sumsq": 0.0, "dpos_count": 0}


def _accumulate_steps(acc, lnorm):
    """Position-step sums of one finite ``[T, J, F]`` clip."""
    if lnorm.shape[0] > 1:
        steps = lnorm[1:, :, _POS_SLICE] - lnorm[:-1, :, _POS_SLICE]
        acc["dpos_sumsq"] += float((steps ** 2).sum())
        acc["dpos_count"] += int(steps.size)


def accumulate_lnorm_stats(feature, cond_entry, acc=None, species=None):
    """Accumulate the L-normalized statistics of one working-layout physical
    ``[T, J, F]`` clip.

    Per-channel sum / sum-of-squares over every joint and frame, the same over
    the root's position alone (see "Root"), and the sum of squared position
    steps. With ``species`` the step sums are also kept per species for
    :func:`species_aux_loss_weights`. ``acc`` is a mutable dict carried across
    clips; finalize_lnorm_stats() turns it into ``(mean, std, dpos_std)``.
    """
    lnorm = np.asarray(physical_hml_to_lnorm(feature, cond_entry), dtype=np.float64)
    n_feats = lnorm.shape[-1]
    if acc is None:
        acc = {"sum": np.zeros(n_feats), "sumsq": np.zeros(n_feats), "count": 0,
               "root_sum": np.zeros(3), "root_sumsq": np.zeros(3), "root_count": 0,
               "species": {}, **_new_step_acc()}
    if lnorm.ndim != 3 or not np.isfinite(lnorm).all():
        # A clip with a non-finite value is left out whole: the step sums need
        # its frames contiguous.
        return acc
    flat = lnorm.reshape(-1, n_feats)
    acc["sum"] += flat.sum(axis=0)
    acc["sumsq"] += (flat ** 2).sum(axis=0)
    acc["count"] += flat.shape[0]
    root = lnorm[:, 0, _POS_SLICE]
    acc["root_sum"] += root.sum(axis=0)
    acc["root_sumsq"] += (root ** 2).sum(axis=0)
    acc["root_count"] += root.shape[0]
    _accumulate_steps(acc, lnorm)
    if species is not None:
        _accumulate_steps(acc["species"].setdefault(species, _new_step_acc()), lnorm)
    return acc


def finalize_lnorm_stats(acc):
    """Turn an accumulator from accumulate_lnorm_stats() into
    ``(mean, std, dpos_std)``: stat vectors in the "Statistics layout" (the
    velocity block gets mean 0 and a zero std, which pooling skips and
    set_canonical_global_stats() floors to one), and the position-step RMS.
    """
    if acc is None or acc["count"] <= 0 or acc["root_count"] <= 0:
        raise ValueError("No finite samples accumulated for canonical stats.")
    mean = acc["sum"] / acc["count"]
    var = np.maximum(acc["sumsq"] / acc["count"] - mean ** 2, 0.0)
    mean[_VEL_SLICE] = 0.0
    var[_VEL_SLICE] = 0.0
    root_mean = acc["root_sum"] / acc["root_count"]
    root_var = np.maximum(acc["root_sumsq"] / acc["root_count"] - root_mean ** 2, 0.0)
    mean = np.concatenate([mean, root_mean])
    var = np.concatenate([var, root_var])
    dpos_std = np.sqrt(acc["dpos_sumsq"] / acc["dpos_count"]) if acc["dpos_count"] > 0 else 0.0
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32), np.float32(dpos_std)


def species_aux_loss_weights(accs, dpos_std):
    """``{species: dpos_weight}`` from the per-species step sums of one or more
    accumulators: ``min(1, sqrt(AUX_LOSS_ENERGY_CAP / energy))``, where the
    energy is the species' mean squared position step in units of
    ``dpos_std``."""
    weights = {}
    for acc in accs:
        for species, step_acc in acc.get("species", {}).items():
            weight = 1.0
            if step_acc["dpos_count"] > 0:
                energy = step_acc["dpos_sumsq"] / step_acc["dpos_count"] / max(float(dpos_std), _STD_FLOOR) ** 2
                if np.isfinite(energy) and energy > AUX_LOSS_ENERGY_CAP:
                    weight = float(np.sqrt(AUX_LOSS_ENERGY_CAP / energy))
            weights[species] = weight
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

    ``subset_stats``: ``{object_subset: (mean, std, dpos_std)}`` ->
    ``(mean, std, dpos_std)``, stat vectors in the "Statistics layout".

    Every subset gets equal weight, whatever its clip count:

    * ``mean`` is the average of the subset means.
    * ``std``: within each subset, each block (position / rotation-6d, and the
      root's X/Z and its Y) is collapsed to the mean of its per-channel stds,
      and the block scalar is the geometric mean of those subset scalars (a
      gain is multiplicative, so the geometric mean spreads every subset's
      resulting variance symmetrically around one in log space). ``dpos_std``
      is pooled the same way. The velocity block stays at one.

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

    _check_root_stats(mean_out.shape[0])
    std_out = np.ones_like(mean_out)
    root = ROOT_STAT_SLICE.start
    blocks = (_POS_SLICE, _ROT_SLICE, [root + c for c in ROOT_XZ_CHANNELS], [root + 1])
    for block in blocks:
        scalars = [
            value for value in (_block_std_scalar(stats[1], block) for stats in subset_stats.values())
            if value is not None
        ]
        if not scalars:
            continue
        std_out[block] = _geometric_mean(scalars)
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
    without stats are left to the callers' own missing-stats checks. An entry
    whose stats were measured in another feature space raises too.
    """
    frames = {}
    for key, entry in cond_dict.items():
        stats = get_canonical_global_stats(entry) if isinstance(entry, dict) else None
        if stats is None:
            continue
        if _is_other_feature_space(entry):
            raise ValueError(
                f"{source} entry {key!r} was built in feature_space={entry['feature_space']!r}, the code "
                f"expects {CANONICAL_FEATURE_SPACE!r}. Regenerate it "
                "(tools/regenerate_dataset_artifacts.py for a dataset, "
                "tools/merge_dataset_cond.py for the merged training cond)."
            )
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


def _is_other_feature_space(cond_entry):
    actual = cond_entry.get("feature_space")
    return actual is not None and actual != CANONICAL_FEATURE_SPACE


def mark_canonical_cond_entry(cond_entry, *, restamp=False):
    """Stamp ``cond_entry`` with the current feature space and derive its rest
    geometry.

    An entry built in another feature space carries stats measured in that
    space, and the model would be trained or decoded with them silently. Such an
    entry raises, unless ``restamp`` is set by a writer that re-measures the
    stats afterwards: the stale space-bound keys are then dropped so none of them
    survive under the new stamp.
    """
    if _is_other_feature_space(cond_entry):
        if not restamp:
            raise ValueError(
                f"cond entry was built in feature_space={cond_entry['feature_space']!r}, the code expects "
                f"{CANONICAL_FEATURE_SPACE!r}; its canonical stats belong to the old space. "
                "Regenerate cond.npy (preprocess_all_datasets.bat --regenerate-side-artifacts, "
                "then tools/merge_dataset_cond.py for the merged training cond)."
            )
        for key in _FEATURE_SPACE_BOUND_KEYS:
            cond_entry.pop(key, None)
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
