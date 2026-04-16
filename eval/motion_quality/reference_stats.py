"""
Per-skeleton reference statistics used to calibrate quality scores.

Statistics are computed from the truebones processed dataset and cached
in-memory (and optionally on disk) so they are only computed once per
skeleton type per process.
"""

from __future__ import annotations

import glob
import os
import pickle
from typing import Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Channel layout (13 features per joint, root excluded from joint array)
# ---------------------------------------------------------------------------
#  0-2  : local RIC position (3)
#  3-8  : 6D rotation (6)  — two orthonormal basis vectors
#  9-11 : linear velocity (3)
#  12   : foot contact flag (1)

CH_POS   = slice(0, 3)
CH_ROT   = slice(3, 9)
CH_ROT_A = slice(3, 6)   # first basis vector
CH_ROT_B = slice(6, 9)   # second basis vector
CH_VEL   = slice(9, 12)
CH_CONT  = 12


class PerSkeletonReferenceStats:
    """Statistics extracted from all reference motions for one skeleton type."""

    def __init__(
        self,
        object_type: str,
        per_channel_mean: np.ndarray,    # (13,)  mean over frames×joints
        per_channel_std: np.ndarray,     # (13,)  std  over frames×joints
        per_joint_vel_std: np.ndarray,   # (J,)   velocity magnitude std per joint
        jerk_rot_p25: float,             # 25th percentile of per-frame rot jerk
        jerk_rot_p75: float,             # 75th percentile
        jerk_vel_p25: float,
        jerk_vel_p75: float,
        temporal_var_p25: float,         # 25th percentile of temporal variance
        temporal_var_p75: float,
        n_motions: int,
    ):
        self.object_type = object_type
        self.per_channel_mean = per_channel_mean
        self.per_channel_std = per_channel_std
        self.per_joint_vel_std = per_joint_vel_std
        self.jerk_rot_p25 = jerk_rot_p25
        self.jerk_rot_p75 = jerk_rot_p75
        self.jerk_vel_p25 = jerk_vel_p25
        self.jerk_vel_p75 = jerk_vel_p75
        self.temporal_var_p25 = temporal_var_p25
        self.temporal_var_p75 = temporal_var_p75
        self.n_motions = n_motions


# Module-level cache: {object_type -> PerSkeletonReferenceStats}
_CACHE: Dict[str, PerSkeletonReferenceStats] = {}


def _compute_jerk(motion: np.ndarray, ch: slice) -> float:
    """Mean squared jerk of channels `ch` across time."""
    r = motion[:, :, ch]          # (T, J, C)
    a = np.diff(np.diff(r, axis=0), axis=0)   # 2nd diff = acceleration
    j = np.diff(a, axis=0)                    # 3rd diff = jerk
    return float((j ** 2).mean())


def _compute_for_motion(motion: np.ndarray):
    """Return (jerk_rot, jerk_vel, temporal_var) for one motion clip."""
    jerk_rot = _compute_jerk(motion, CH_ROT)
    jerk_vel = _compute_jerk(motion, CH_VEL)
    temporal_var = float(motion.var(axis=0).mean())
    return jerk_rot, jerk_vel, temporal_var


def compute_reference_stats(
    object_type: str,
    dataset_dir: str,
    max_motions: int = 200,
) -> Optional[PerSkeletonReferenceStats]:
    """
    Compute reference statistics from all *.npy motion files that match
    ``<dataset_dir>/motions/<object_type>_*.npy``.

    Returns None if no reference motions are found.
    """
    pattern = os.path.join(dataset_dir, "motions", f"{object_type}_*.npy")
    paths = sorted(glob.glob(pattern))[:max_motions]
    if not paths:
        return None

    all_frames: list[np.ndarray] = []
    jerk_rots, jerk_vels, temporal_vars = [], [], []
    per_joint_vel_sq_sum = None
    n_frames_total = 0

    for path in paths:
        try:
            m = np.load(path).astype(np.float32)  # (T, J, 13)
        except Exception:
            continue
        if m.ndim != 3 or m.shape[-1] != 13:
            continue

        # Flatten to (T*J, 13) for channel statistics
        all_frames.append(m.reshape(-1, 13))

        jr, jv, tv = _compute_for_motion(m)
        jerk_rots.append(jr)
        jerk_vels.append(jv)
        temporal_vars.append(tv)

        # Per-joint velocity magnitude
        vel_mag = np.sqrt((m[:, :, CH_VEL] ** 2).sum(-1))  # (T, J)
        if per_joint_vel_sq_sum is None:
            per_joint_vel_sq_sum = np.zeros(m.shape[1], dtype=np.float64)
        per_joint_vel_sq_sum += vel_mag.var(axis=0)
        n_frames_total += m.shape[0]

    if not jerk_rots:
        return None

    all_data = np.concatenate(all_frames, axis=0)   # (N, 13)
    per_channel_mean = all_data.mean(0)
    per_channel_std  = all_data.std(0) + 1e-8

    n_motion = len(jerk_rots)
    per_joint_vel_std = np.sqrt(np.maximum(per_joint_vel_sq_sum / n_motion, 0))

    return PerSkeletonReferenceStats(
        object_type=object_type,
        per_channel_mean=per_channel_mean,
        per_channel_std=per_channel_std,
        per_joint_vel_std=per_joint_vel_std,
        jerk_rot_p25=float(np.percentile(jerk_rots, 25)),
        jerk_rot_p75=float(np.percentile(jerk_rots, 75)),
        jerk_vel_p25=float(np.percentile(jerk_vels, 25)),
        jerk_vel_p75=float(np.percentile(jerk_vels, 75)),
        temporal_var_p25=float(np.percentile(temporal_vars, 25)),
        temporal_var_p75=float(np.percentile(temporal_vars, 75)),
        n_motions=n_motion,
    )


def get_reference_stats(
    object_type: str,
    dataset_dir: str,
    cache_path: Optional[str] = None,
    max_motions: int = 200,
) -> Optional[PerSkeletonReferenceStats]:
    """
    Return (possibly cached) reference statistics for ``object_type``.

    If ``cache_path`` is given, a pickle file is read/written there so that
    stats survive across processes.
    """
    if object_type in _CACHE:
        return _CACHE[object_type]

    # Try disk cache first
    if cache_path and os.path.isfile(cache_path):
        try:
            with open(cache_path, "rb") as fh:
                stats: Dict[str, PerSkeletonReferenceStats] = pickle.load(fh)
            if object_type in stats:
                _CACHE[object_type] = stats[object_type]
                return _CACHE[object_type]
        except Exception:
            pass

    stats_obj = compute_reference_stats(object_type, dataset_dir, max_motions)
    if stats_obj is not None:
        _CACHE[object_type] = stats_obj

        # Write/update disk cache
        if cache_path:
            existing: Dict[str, PerSkeletonReferenceStats] = {}
            if os.path.isfile(cache_path):
                try:
                    with open(cache_path, "rb") as fh:
                        existing = pickle.load(fh)
                except Exception:
                    pass
            existing[object_type] = stats_obj
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            with open(cache_path, "wb") as fh:
                pickle.dump(existing, fh)

    return stats_obj
