"""Canonical AnyTop motion feature conversion helpers.

V3 keeps the HML-like physical feature layout but converts to a *canonical*
model feature space via three prior-free, exactly-invertible steps:

  1. Position re-centering: the position channel is expressed relative to the
     skeleton rest-pose feature position (residual-from-rest).
  2. Per-skeleton size normalization (L-scaling): the position and velocity
     channels are divided by the skeleton's own geometric length ``L`` so that
     every species lands in a size-free space (cross-species fair). Rotation
     (6d) and contact are size-independent and are left untouched here.
  3. Per-object_subset standardization: subtract a per-channel mean and divide
     by a *block-collapsed* std (13-vectors). These statistics are computed at
     preprocessing time in the L-normalized space of step (2), pooled across all
     joints / frames / clips / species *within each object_subset* (quadruped /
     biped / multiped / serpentine / aquatic / winged / drifting). They are
     therefore a
     cross-species constant *per object_subset* (NOT a per-species motion prior):
     a held-out species inherits the stats of its object_subset, so the
     standardization generalizes while giving each object_subset its own
     zero-mean / unit-std calibration (winged flapping and quadruped locomotion
     have very different velocity / rotation scales). They are stored in ``cond``
     per species (``canonical_feature_mean`` / ``canonical_feature_std``); species
     sharing an object_subset carry the same 13-vectors.

     The raw per-channel std is passed through :func:`collapse_stat_blocks`
     before it is stored: each block (position / rotation-6d / velocity) is
     collapsed to one scalar, and the position scalar is shared by every
     object_subset. Bone lengths are a function of the position channel alone,
     so a globally shared position gain makes an object_subset mismatch
     structurally unable to deform a skeleton -- see
     ``docs/canonical_frame_and_label_transfer.md``. ``mean`` stays per-channel
     and per-subset (a mean mismatch is a rigid translation, bone-length exact).

Channel layout per joint (n_feats == 13):
    0:3   position   (rest-centered residual)
    3:9   rotation 6d
    9:12  local velocity
    12    foot contact (binary)

"""

from __future__ import annotations

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch is present in training/runtime.
    torch = None


CANONICAL_FEATURE_SPACE = "canonical_motion_v3"
PHYSICAL_FEATURE_SPACE = "hml_like_v_current"

# Keys under which the per-object_subset per-channel standardization statistics
# are stored in each cond entry. The 13-vectors are shared by species within the
# same object_subset (quadruped / winged / ...). Computed at preprocessing time
# over the L-normalized training distribution -- see
# ``_compute_canonical_stats_per_object_subset``.
CANONICAL_MEAN_KEY = "canonical_feature_mean"
CANONICAL_STD_KEY = "canonical_feature_std"

# Channels whose std falls below this floor are treated as unit-variance so the
# encode divide never explodes (mirrors the old std_safe floor).
_STD_FLOOR = 1e-5

_POS_SLICE = slice(0, 3)
_ROT_SLICE = slice(3, 9)
_VEL_SLICE = slice(9, 12)
_CONTACT_IDX = 12


def _is_torch_tensor(value) -> bool:
    return torch is not None and torch.is_tensor(value)


def _rest_pos_from_cond(cond_entry, *, like=None):
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


def _length_scale_from_rest(rest_pos):
    """Per-skeleton geometric length L = RMS spread of rest-pose joint positions.

    Accepts ``[J, 3]`` (returns a scalar) or ``[B, J, 3]`` (returns ``[B]``).
    Reduces over the joint and xyz axes (the trailing two), so a leading batch
    axis is preserved. Derived purely from skeleton geometry -- no motion stats.
    """
    if _is_torch_tensor(rest_pos):
        rp = rest_pos.to(dtype=torch.float32)
        centered = rp - rp.mean(dim=-2, keepdim=True)
        L = torch.sqrt((centered ** 2).mean(dim=(-2, -1)))
        # Degenerate (single joint / collapsed) skeleton -> fall back to 1.0 so
        # the encode divide never explodes. Mirrors the numpy branch below.
        return torch.where(torch.isfinite(L) & (L > 1e-6), L, torch.ones_like(L))
    rp = np.asarray(rest_pos, dtype=np.float64)
    centered = rp - rp.mean(axis=-2, keepdims=True)
    L = np.sqrt((centered ** 2).mean(axis=(-2, -1)))
    return np.where(np.isfinite(L) & (L > 1e-6), L, 1.0)


def _spatial_L_vectors(n_feats):
    """Return a per-channel ``uses_L`` vector of length ``n_feats``.

    ``uses_L[c] == 1`` for the position and velocity groups (size-dependent) and
    ``0`` for rotation/contact (size-independent). The L-scaling divides those
    channels by the per-skeleton length ``L`` (encode) / multiplies (decode).
    """
    uses_L = np.zeros(n_feats, dtype=np.float64)
    uses_L[_POS_SLICE] = 1.0
    if n_feats >= _VEL_SLICE.stop:
        uses_L[_VEL_SLICE] = 1.0
    return uses_L


def _apply_L_scale(feature, cond_entry, inverse: bool):
    """Divide (encode) or multiply (decode) the position/velocity channels by the
    per-skeleton length ``L``. Exact inverse of itself with flipped ``inverse``.
    Rotation/contact channels are left unchanged. ``L`` may be a scalar or ``[B]``.
    """
    L = _length_scale_from_rest(_rest_pos_from_cond(cond_entry, like=feature))

    if _is_torch_tensor(feature):
        out = feature
        ndim = out.dim()
        n_feats = out.shape[2] if ndim == 4 else out.shape[-1]
        uses_L = torch.as_tensor(_spatial_L_vectors(n_feats), device=out.device, dtype=out.dtype)
        Lt = L.to(device=out.device, dtype=out.dtype) if _is_torch_tensor(L) \
            else torch.as_tensor(float(L), device=out.device, dtype=out.dtype)

        if ndim == 4:
            # [B, J, F, T]; L is scalar or [B].
            if Lt.dim() == 0:
                scale = (Lt ** uses_L).view(1, 1, n_feats, 1)
            else:
                scale = (Lt.view(-1, 1) ** uses_L.view(1, -1)).view(out.shape[0], 1, n_feats, 1)
        else:
            # channels on the last axis ([J, F] / [T, J, F]); L is scalar.
            scale = Lt ** uses_L
        return out * scale if inverse else out / scale

    out = np.asarray(feature, dtype=np.float32)
    ndim = out.ndim
    n_feats = out.shape[2] if ndim == 4 else out.shape[-1]
    uses_L = _spatial_L_vectors(n_feats)
    Lnp = np.asarray(L, dtype=np.float64)

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
    """Return ``(mean, std)`` per-object_subset standardization vectors from a
    cond entry, or ``None`` when absent (cond predates the stats, e.g. tests).
    """
    if cond_entry is None:
        return None
    mean = cond_entry.get(CANONICAL_MEAN_KEY) if hasattr(cond_entry, "get") else None
    std = cond_entry.get(CANONICAL_STD_KEY) if hasattr(cond_entry, "get") else None
    if mean is None or std is None:
        return None
    return mean, std


def set_canonical_global_stats(cond_entry, mean, std):
    """Store the per-object_subset mean/std (13-vectors) on a cond entry,
    flooring near-constant channels to unit std so the encode divide is safe.
    """
    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    std = np.asarray(std, dtype=np.float32).reshape(-1)
    std = np.where(std < _STD_FLOOR, 1.0, std).astype(np.float32, copy=False)
    cond_entry[CANONICAL_MEAN_KEY] = mean
    cond_entry[CANONICAL_STD_KEY] = std
    return cond_entry


def _view_stat_torch(stat, ndim: int, n_feats: int):
    """Shape a standardization stat tensor for broadcasting against a feature.

    ``stat`` is either 1D ``[F]`` (broadcast over the whole batch) or 2D
    ``[B, F]`` (per-sample). For a 4D ``[B, J, F, T]`` feature the result is
    ``[1, 1, F, 1]`` (1D) or ``[B, 1, F, 1]`` (per-sample). For lower-rank
    single-sample features the 1D vector broadcasts over the trailing axis as-is.
    """
    if stat.dim() >= 2:
        if ndim != 4:
            raise ValueError(
                "per-sample [B, F] canonical stats are only supported for 4D "
                f"[B, J, F, T] features, got feature rank {ndim}."
            )
        stat2 = stat.reshape(stat.shape[0], -1)[:, :n_feats]
        return stat2.view(stat2.shape[0], 1, n_feats, 1)
    stat1 = stat.reshape(-1)[:n_feats]
    return stat1.view(1, 1, n_feats, 1) if ndim == 4 else stat1


def _view_stat_numpy(stat, ndim: int, n_feats: int):
    """Numpy counterpart of _view_stat_torch (same 1D / per-sample contract)."""
    if stat.ndim >= 2:
        if ndim != 4:
            raise ValueError(
                "per-sample [B, F] canonical stats are only supported for 4D "
                f"[B, J, F, T] features, got feature rank {ndim}."
            )
        stat2 = stat.reshape(stat.shape[0], -1)[:, :n_feats]
        return stat2.reshape(stat2.shape[0], 1, n_feats, 1)
    stat1 = stat.reshape(-1)[:n_feats]
    return stat1.reshape(1, 1, n_feats, 1) if ndim == 4 else stat1


def _apply_global_stats(feature, cond_entry, inverse: bool):
    """Standardize (encode: ``(x - mean) / std``) or de-standardize
    (decode: ``x * std + mean``) every channel by the per-object_subset
    statistics stored on ``cond_entry``.

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
            "cannot (de)standardize canonical features. Pass the per-object_subset "
            "stats (from the cond entry or the collated y dict), or regenerate "
            "cond.npy. A freshly built new-skeleton cond inherits these stats from "
            "a same-object_subset species in the dataset cond.npy."
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
        # decode, or a homogeneous batch). A 2D ``[B, F]`` vector is per-sample:
        # each batch element carries its own object_subset's stats (the collate stacks
        # them in batch order), which a mixed-species batch requires.
        mean_t = (mean if _is_torch_tensor(mean) else torch.as_tensor(np.asarray(mean, dtype=np.float32))) \
            .to(device=out.device, dtype=out.dtype)
        std_t = (std if _is_torch_tensor(std) else torch.as_tensor(np.asarray(std, dtype=np.float32))) \
            .to(device=out.device, dtype=out.dtype)
        mean_v = _view_stat_torch(mean_t, ndim, n_feats)
        std_v = _view_stat_torch(std_t, ndim, n_feats)
        return (out * std_v) + mean_v if inverse else (out - mean_v) / std_v

    out = np.asarray(feature, dtype=np.float32)
    ndim = out.ndim
    n_feats = out.shape[2] if ndim == 4 else out.shape[-1]
    mean_v = _view_stat_numpy(np.asarray(mean, dtype=np.float32), ndim, n_feats)
    std_v = _view_stat_numpy(np.asarray(std, dtype=np.float32), ndim, n_feats)
    out = (out * std_v) + mean_v if inverse else (out - mean_v) / std_v
    return out.astype(np.float32, copy=False)


def _apply_rest_pos(feature, cond_entry, sign: float):
    rest_pos = _rest_pos_from_cond(cond_entry, like=feature)
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
    position/velocity channels by ``L``, then standardize every channel by the
    per-object_subset mean/std.
    """
    lnorm = physical_hml_to_lnorm(feature, cond_entry)
    return _apply_global_stats(lnorm, cond_entry, inverse=False)


def canonical_to_physical_hml(feature, cond_entry):
    """Decode canonical model features back to HML-like physical features.

    Decode order (exact inverse of encode): de-standardize by the
    per-object_subset mean/std, multiply the position/velocity channels by
    ``L``, then add rest back onto the position channel.
    """
    lnorm = _apply_global_stats(feature, cond_entry, inverse=True)
    unscaled = _apply_L_scale(lnorm, cond_entry, inverse=True)
    return _apply_rest_pos(unscaled, cond_entry, sign=1.0)


def build_canonical_rest_feature(cond_entry):
    """Build the rest-pose token in canonical feature space.

    The rest pose has no motion, so its physical feature is the rest rotation
    with the position channel equal to the rest position and the velocity /
    contact channels zeroed. Encoding it through the same forward transform
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
        if rest_phys.shape[-1] > _CONTACT_IDX:
            rest_phys[..., _CONTACT_IDX] = 0.0
    else:
        rest_phys = np.asarray(rest, dtype=np.float32).copy()
        if rest_phys.shape[-1] >= _VEL_SLICE.stop:
            rest_phys[..., _VEL_SLICE] = 0.0
        if rest_phys.shape[-1] > _CONTACT_IDX:
            rest_phys[..., _CONTACT_IDX] = 0.0

    return physical_hml_to_canonical(rest_phys, cond_entry)


def accumulate_lnorm_stats(feature, cond_entry, acc=None):
    """Accumulate per-channel sum / sum-of-squares / count over the L-normalized
    encoding of one physical clip, pooled across all joints and frames. ``acc``
    is a mutable dict carried across clips; pass the same dict for every clip.
    Use finalize_lnorm_stats() to turn the accumulator into (mean, std).
    """
    lnorm = np.asarray(physical_hml_to_lnorm(feature, cond_entry), dtype=np.float64)
    flat = lnorm.reshape(-1, lnorm.shape[-1])
    flat = flat[np.isfinite(flat).all(axis=1)]
    n_feats = flat.shape[1]
    if acc is None:
        acc = {"sum": np.zeros(n_feats), "sumsq": np.zeros(n_feats), "count": 0}
    acc["sum"] += flat.sum(axis=0)
    acc["sumsq"] += (flat ** 2).sum(axis=0)
    acc["count"] += flat.shape[0]
    return acc


def finalize_lnorm_stats(acc):
    """Turn an accumulator from accumulate_lnorm_stats() into ``(mean, std)``
    13-vectors. Near-constant channels are floored to unit std downstream by
    set_canonical_global_stats().
    """
    if acc is None or acc["count"] <= 0:
        raise ValueError("No finite samples accumulated for canonical stats.")
    mean = acc["sum"] / acc["count"]
    var = np.maximum(acc["sumsq"] / acc["count"] - mean ** 2, 0.0)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


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


def collapse_stat_blocks(subset_stats):
    """Collapse each object_subset's std inside feature blocks and share the
    position gain across every subset.

    ``subset_stats``: ``{object_subset: (mean13, std13)}`` -> a new dict of the
    same keys and shapes. Only ``std`` is touched; ``mean`` is returned as-is.

    Two changes, both motivated in ``docs/canonical_frame_and_label_transfer.md``:

    1. **Block collapse.** Within each block (position / rotation-6d / velocity)
       the per-channel stds are averaged into one scalar. ``l_simple`` is
       computed in this standardized space, so ``1 / std`` is an implicit
       per-channel loss weight: an anisotropic block systematically under-
       penalizes whichever axis has the largest std (measured up to 4.1x on the
       vertical position axis). This restores the invariant the pre-v3
       ``get_mean_std`` held (it collapsed each block with ``.mean()``), so
       ``l_simple`` again weights every joint *and* axis uniformly.

    2. **Shared position gain.** The position block scalar is then shared by
       every subset, as the geometric mean of the per-subset scalars (a gain is
       multiplicative, so the geometric mean is the average that is not dragged
       by the largest subset). Bone lengths are decided by the position channel
       alone (the exporter overrides the FK joint placement with the RIC
       position channel), and the decode is ``x * std + mean``: a ``mean``
       mismatch translates the whole skeleton rigidly and leaves every bone
       length exact, so once the position *gain* is a single global constant,
       decoding a clip through the wrong subset's statistics cannot deform it at
       all. Rotation / velocity gains stay per-subset -- they cannot change bone
       lengths, and each body plan keeps its own unit-variance calibration where
       it costs nothing.

    ``contact`` (index 12) belongs to no block and stays per-subset untouched:
    the subsets whose contact channel is identically zero (aquatic / serpentine)
    keep falling through to the ``_STD_FLOOR`` -> 1.0 path in
    set_canonical_global_stats(), exactly as before.

    The 13-dim shape, the two cond keys and ``CANONICAL_FEATURE_SPACE`` are all
    unchanged -- this only changes the *numbers* written into the table, so the
    encode/decode contract is untouched (a regenerated cond.npy is still
    ``canonical_motion_v3``).
    """
    if not subset_stats:
        return {}

    blocks = (_POS_SLICE, _ROT_SLICE, _VEL_SLICE)
    per_subset_scalars = {
        subset: [_block_std_scalar(std, block) for block in blocks]
        for subset, (_mean, std) in subset_stats.items()
    }

    pos_scalars = [
        scalars[0] for scalars in per_subset_scalars.values() if scalars[0] is not None
    ]
    shared_pos_gain = (
        float(np.exp(np.mean(np.log(np.asarray(pos_scalars, dtype=np.float64)))))
        if pos_scalars else None
    )

    collapsed = {}
    for subset, (mean, std) in subset_stats.items():
        std_out = np.asarray(std, dtype=np.float64).reshape(-1).copy()
        n_feats = std_out.shape[0]
        block_values = list(per_subset_scalars[subset])
        if shared_pos_gain is not None:
            block_values[0] = shared_pos_gain
        for block, value in zip(blocks, block_values):
            if value is None or block.start >= n_feats:
                continue
            std_out[block.start:min(block.stop, n_feats)] = value
        collapsed[subset] = (
            np.asarray(mean, dtype=np.float32).reshape(-1),
            std_out.astype(np.float32),
        )
    return collapsed


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
            "per-object_subset canonical standardization statistics."
        )
    return True
