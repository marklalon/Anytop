"""Canonical AnyTop motion feature conversion helpers.

V3 keeps the HML-like physical feature layout but converts to a *canonical*
model feature space via three prior-free, exactly-invertible steps:

  1. Position re-centering: the position channel is expressed relative to the
     skeleton rest-pose feature position (residual-from-rest).
  2. Per-skeleton size normalization (L-scaling): the position and velocity
     channels are divided by the skeleton's own geometric length ``L`` so that
     every species lands in a size-free space (cross-species fair). Rotation
     (6d) and contact are size-independent and are left untouched here.
  3. Global per-channel standardization: subtract a GLOBAL per-channel mean and
     divide by a GLOBAL per-channel std (13-vectors). These statistics are
     computed ONCE over the whole training set in the L-normalized space of
     step (2) and pooled across all joints / frames / clips / species, so they
     are a single cross-species constant (NOT a per-species motion prior) and
     generalize to held-out species. They are stored in ``cond`` at
     preprocessing time (``canonical_feature_mean`` / ``canonical_feature_std``).

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

# Keys under which the global per-channel standardization statistics are stored
# in each cond entry. The same 13-vectors are written to every species (they are
# a single cross-species constant). Computed at preprocessing time over the
# L-normalized training distribution -- see compute_global_canonical_stats().
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
    """Return ``(mean, std)`` global per-channel standardization vectors from a
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
    """Store the global per-channel mean/std (13-vectors) on a cond entry,
    flooring near-constant channels to unit std so the encode divide is safe.
    """
    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    std = np.asarray(std, dtype=np.float32).reshape(-1)
    std = np.where(std < _STD_FLOOR, 1.0, std).astype(np.float32, copy=False)
    cond_entry[CANONICAL_MEAN_KEY] = mean
    cond_entry[CANONICAL_STD_KEY] = std
    return cond_entry


def _apply_global_stats(feature, cond_entry, inverse: bool):
    """Standardize (encode: ``(x - mean) / std``) or de-standardize
    (decode: ``x * std + mean``) every channel by the GLOBAL per-channel
    statistics stored on ``cond_entry``.

    Raises if the statistics are absent. Silently skipping standardization would
    leave the features in the L-normalized space (wrong scale) without any error
    -- exactly the failure mode that produced broken inference when a caller
    passed a decode dict missing the stats. Callers must always thread the global
    stats through (cond entry, or the collated ``y`` dict). The stats-free
    L-normalized space is reachable only via physical_hml_to_lnorm(), which is
    used solely to *compute* these statistics at preprocessing time.
    """
    stats = get_canonical_global_stats(cond_entry)
    if stats is None:
        raise KeyError(
            f"cond entry is missing {CANONICAL_MEAN_KEY!r}/{CANONICAL_STD_KEY!r}; "
            "cannot (de)standardize canonical features. Pass the global stats "
            "(from the cond entry or the collated y dict), or regenerate cond.npy."
        )
    mean, std = stats

    if _is_torch_tensor(feature):
        out = feature
        ndim = out.dim()
        n_feats = out.shape[2] if ndim == 4 else out.shape[-1]
        # Stats may arrive as numpy (per-species cond entry) or as torch tensors,
        # possibly on CUDA (collated into y for the training aux-loss decode).
        # Convert without round-tripping through numpy so a CUDA tensor is safe.
        mean_t = (mean if _is_torch_tensor(mean) else torch.as_tensor(np.asarray(mean, dtype=np.float32))) \
            .to(device=out.device, dtype=out.dtype).reshape(-1)
        std_t = (std if _is_torch_tensor(std) else torch.as_tensor(np.asarray(std, dtype=np.float32))) \
            .to(device=out.device, dtype=out.dtype).reshape(-1)
        mean_t = mean_t[:n_feats]
        std_t = std_t[:n_feats]
        if ndim == 4:
            mean_v = mean_t.view(1, 1, n_feats, 1)
            std_v = std_t.view(1, 1, n_feats, 1)
        else:
            mean_v = mean_t
            std_v = std_t
        return (out * std_v) + mean_v if inverse else (out - mean_v) / std_v

    out = np.asarray(feature, dtype=np.float32)
    ndim = out.ndim
    n_feats = out.shape[2] if ndim == 4 else out.shape[-1]
    mean_np = np.asarray(mean, dtype=np.float32).reshape(-1)[:n_feats]
    std_np = np.asarray(std, dtype=np.float32).reshape(-1)[:n_feats]
    if ndim == 4:
        mean_v = mean_np.reshape(1, 1, n_feats, 1)
        std_v = std_np.reshape(1, 1, n_feats, 1)
    else:
        mean_v = mean_np
        std_v = std_np
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
    GLOBAL per-channel mean/std.
    """
    lnorm = physical_hml_to_lnorm(feature, cond_entry)
    return _apply_global_stats(lnorm, cond_entry, inverse=False)


def canonical_to_physical_hml(feature, cond_entry):
    """Decode canonical model features back to HML-like physical features.

    Decode order (exact inverse of encode): de-standardize by the GLOBAL
    per-channel mean/std, multiply the position/velocity channels by ``L``, then
    add rest back onto the position channel.
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
    return True
