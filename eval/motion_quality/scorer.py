"""
Low-Shot Weighted-Reference Motion Quality Scorer
=================================================

Scores one or more query motions by comparing them against a weighted
reference prior assembled from dataset motions that share the requested
semantic action tags.

Reference construction
----------------------
- Resolve the query species in cond.npy.
- Find the Top-K nearest species in semantic joint-name embedding space.
- Filter dataset motions by action_tags.
- Distribute each selected species weight across its reference motions in
  proportion to motion frame count.

Scoring dimensions
------------------
- Local joint naturalness
    Per-group spectral and smoothness summaries are compared to weighted
    reference priors using the same robust-deviation scheme.

Motion format  (T × J × 13  float32, normalised):
    ch 0-2  : local RIC position
    ch 3-8  : 6-D rotation
    ch 9-11 : linear velocity
    ch 12   : foot-contact flag  (not evaluated)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.signal

from data_loaders.truebones.offline_reference_dataset import load_cond_dict

from .bone_length_drift import compute_bone_length_drift, resolve_comparison_edges
from .reference_bank import ReferenceClip, WeightedReferenceBank, build_weighted_reference_bank
from .reference_stats import CH_POS, CH_ROT

_MIN_CLIP_FRAMES = 8
_GROUP_ORDER = ("root", "axial", "limbs")


@dataclass
class DistributionEvalReport:
    """Full low-shot weighted-reference quality report."""

    object_type: Optional[str]
    action_tags: Optional[str]
    n_input: int
    n_reference: int
    input_total_frames: int
    reference_total_frames: int
    scoring_mode: str
    top_k_species: int
    reference_species: List[Dict[str, Any]]

    overall_score: float
    spectral_flatness_score: float
    jerk_score: float
    snap_score: float
    bone_length_score: float

    raw: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "overall_score": round(self.overall_score, 4),
            "detail": {
                "jerk_score (w=0.328)": round(self.jerk_score, 4),
                "snap_score (w=0.228)": round(self.snap_score, 4),
                "spectral_flatness_score (w=0.228)": round(self.spectral_flatness_score, 4),
                "bone_length_score (w=0.228)": round(self.bone_length_score, 4),
            },
            "reference": {
                "scoring_mode": self.scoring_mode,
                "top_k_species": self.top_k_species,
                "selected_species": self.reference_species,
                "n_reference": self.n_reference,
                "reference_total_frames": self.reference_total_frames,
            },
            "meta": {
                "object_type": self.object_type,
                "action_tags": self.action_tags,
                "n_input": self.n_input,
                "input_total_frames": self.input_total_frames,
            },
            "raw": self.raw,
        }

    def __str__(self) -> str:
        width = 46
        lines = [
            "Low-Shot Weighted-Reference Motion Quality Report",
            f"  Object type : {self.object_type or 'unknown'}",
            f"  Action tags : {self.action_tags or 'unknown'}",
            f"  Inputs      : {self.n_input} clip(s) / {self.input_total_frames} frames",
            f"  Reference   : {self.n_reference} clip(s) / {self.reference_total_frames} frames",
            "",
            f"  +{'-' * width}+--------+",
            f"  | {'Dimension':<{width}}| Score  |",
            f"  +{'-' * width}+--------+",
            f"  | {'Joint naturalness':<{width}}| {self.overall_score:6.4f} |",
            f"  +{'-' * width}+--------+",
            "",
            f"  Reference species : {self.reference_species}",
        ]
        return "\n".join(lines)


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("weights must not be empty")
    if np.any(arr < 0.0):
        raise ValueError("weights must be non-negative")
    total = float(arr.sum())
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    return arr / total


def _weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    values_arr = np.asarray(values, dtype=np.float64).reshape(-1)
    weight_arr = _normalize_weights(np.asarray(weights, dtype=np.float64).reshape(-1))
    if values_arr.size != weight_arr.size:
        raise ValueError("values and weights must have the same size")
    return float(np.sum(values_arr * weight_arr))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    values_arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if values_arr.size == 0:
        raise ValueError("values must not be empty")
    if values_arr.size == 1:
        return float(values_arr[0])
    weight_arr = _normalize_weights(np.asarray(weights, dtype=np.float64).reshape(-1))
    if weight_arr.size != values_arr.size:
        raise ValueError("values and weights must have the same size")
    order = np.argsort(values_arr)
    sorted_values = values_arr[order]
    sorted_weights = weight_arr[order]
    cumulative = np.cumsum(sorted_weights)
    return float(np.interp(float(quantile), cumulative, sorted_values))


def _weighted_iqr(values: np.ndarray, weights: np.ndarray) -> float:
    q25 = _weighted_quantile(values, weights, 0.25)
    q75 = _weighted_quantile(values, weights, 0.75)
    return float(q75 - q25)


def _metric_abs_floor(name) -> float:
    if name == "psd_bin":
        return 0.01
    if name in {"spectral_flatness", "acf_peak"}:
        return 0.03
    if name == "spectral_centroid":
        return 0.015
    if name in {"jerk_norm", "snap_norm"}:
        return 0.05
    return 0.02


def _distance_to_score(distance: float) -> float:
    return float(1.0 / (1.0 + distance * distance))


_LIMB_KEYWORDS = (
    "thigh",
    "calf",
    "leg",
    "foot",
    "toe",
    "hoof",
    "paw",
    "arm",
    "forearm",
    "hand",
    "finger",
    "clavicle",
    "shoulder",
    "horse link",
    "wing",
    "flipper",
)
_AXIAL_KEYWORDS = (
    "spine",
    "pelvis",
    "neck",
    "head",
    "tail",
    "torso",
    "chest",
    "abdomen",
    "back",
    "body",
    "center",
    "ctrl",
    "control",
    "saddle",
    "handle",
    "hair",
    "ear",
    "jaw",
    "snout",
    "muzzle",
)


def _normalise_joint_label(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _looks_like_limb_joint(label: str) -> bool:
    return any(keyword in label for keyword in _LIMB_KEYWORDS)


def _looks_like_axial_joint(label: str) -> bool:
    return any(keyword in label for keyword in _AXIAL_KEYWORDS)


def _coerce_index_array(indices: Sequence[int] | np.ndarray | None, n_joints: int) -> np.ndarray:
    if indices is None:
        return np.zeros(0, dtype=np.int64)
    arr = np.asarray(list(indices), dtype=np.int64)
    if arr.size == 0:
        return arr
    return np.unique(arr[(arr >= 0) & (arr < n_joints)])


def _build_joint_groups_from_cond(
    object_cond: Mapping[str, object],
    n_joints: int,
) -> Tuple[Dict[str, np.ndarray], str]:
    parents = np.asarray(object_cond.get("parents", []), dtype=np.int64)
    if len(parents) != n_joints:
        raise ValueError(f"Expected {len(parents)} joints from metadata, got {n_joints}")

    labels_source = object_cond.get("canonical_joint_names") or object_cond.get("joints_names") or []
    labels = [_normalise_joint_label(labels_source[idx]) if idx < len(labels_source) else "" for idx in range(n_joints)]
    contact_indices = _coerce_index_array(object_cond.get("contact_joints"), n_joints)

    limb_mask = np.zeros(n_joints, dtype=bool)
    axial_name_mask = np.zeros(n_joints, dtype=bool)
    children: list[list[int]] = [[] for _ in range(n_joints)]
    for child_index in range(1, n_joints):
        parent_index = int(parents[child_index])
        if 0 <= parent_index < n_joints:
            children[parent_index].append(child_index)

    for joint_index in range(1, n_joints):
        label = labels[joint_index]
        axial_name_mask[joint_index] = _looks_like_axial_joint(label)
        if _looks_like_limb_joint(label):
            limb_mask[joint_index] = True
    if contact_indices.size:
        limb_mask[contact_indices] = True

    changed = True
    while changed:
        changed = False
        for joint_index in range(1, n_joints):
            parent_index = int(parents[joint_index])
            if parent_index > 0 and limb_mask[parent_index] and not axial_name_mask[joint_index] and not limb_mask[joint_index]:
                limb_mask[joint_index] = True
                changed = True
        for joint_index in range(n_joints - 1, 0, -1):
            if limb_mask[joint_index] or axial_name_mask[joint_index]:
                continue
            child_indices = children[joint_index]
            if child_indices and all(limb_mask[child] for child in child_indices):
                limb_mask[joint_index] = True
                changed = True

    axial_mask = np.zeros(n_joints, dtype=bool)
    axial_mask[1:] = ~limb_mask[1:]

    groups = {
        "root": np.asarray([0], dtype=np.int64),
        "axial": np.flatnonzero(axial_mask),
        "limbs": np.flatnonzero(limb_mask),
    }
    return groups, "cond_semantics"


def _welch_psd(sig: np.ndarray, nperseg: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(sig)
    nps = min(nperseg, n)
    if nps < 2:
        return np.array([0.0]), np.array([max(float(np.var(sig)), 1e-30)])
    freqs, psd = scipy.signal.welch(sig, nperseg=nps)
    return freqs, psd + 1e-30


def _compute_features(motion: np.ndarray, nperseg: int) -> Dict[str, np.ndarray]:
    """Compute per-joint features for a single motion (T, J, 13)."""
    t_len, joint_count, _ = motion.shape
    pos = motion[:, :, CH_POS].astype(np.float64)

    # Spectral flatness — batched Welch PSD over all joints×channels at once
    spectral_flatness = np.zeros(joint_count)
    nps = min(nperseg, t_len)
    if nps >= 2:
        pos_flat = pos.reshape(t_len, -1)  # (T, J*3)
        _, psd_all = scipy.signal.welch(pos_flat, nperseg=nps, axis=0)  # (n_freq, J*3)
        psd_all = psd_all + 1e-30
        psd_all = psd_all.reshape(-1, joint_count, 3)  # (n_freq, J, 3)
        log_mean = np.mean(np.log(psd_all), axis=0)       # (J, 3)
        arith_mean = np.mean(psd_all, axis=0)             # (J, 3)
        spectral_flatness = np.exp(log_mean) / np.maximum(arith_mean, 1e-30)  # (J, 3)
        spectral_flatness = spectral_flatness.mean(axis=-1)  # (J,)

    # Jerk (3rd derivative) — already vectorized
    jerk_norm = np.zeros(joint_count)
    if t_len >= 4:
        jerk3 = np.diff(np.diff(np.diff(pos, axis=0), axis=0), axis=0)
        per_joint_jerk = (jerk3 ** 2).mean(axis=(0, 2))
        per_joint_var = pos.var(axis=0).mean(axis=-1)
        jerk_norm = per_joint_jerk / (per_joint_var + 1e-10)

    # Snap (4th derivative) — already vectorized
    snap_norm = np.zeros(joint_count)
    if t_len >= 5:
        snap = np.diff(np.diff(np.diff(np.diff(pos, axis=0), axis=0), axis=0), axis=0)
        snap_rms = np.sqrt((snap ** 2).mean(axis=(0, 2)))
        pos_rms = np.sqrt((pos ** 2).mean(axis=(0, 2))) + 1e-10
        snap_norm = snap_rms / pos_rms

    return {
        "spectral_flatness": spectral_flatness,
        "jerk_norm": jerk_norm,
        "snap_norm": snap_norm,
    }


def _compute_features_batch(motions: List[np.ndarray], nperseg: int) -> List[Dict[str, np.ndarray]]:
    """Compute per-joint features for multiple motions, batching by frame count.

    Groups motions with the same T together so Welch PSD can process them
    in a single call instead of one-per-motion. Large groups are further
    split into chunks of at most 32 to bound peak memory.
    """
    # Group by (T, J) so we can stack
    groups: Dict[Tuple[int, int], List[int]] = {}
    for idx, m in enumerate(motions):
        key = (m.shape[0], m.shape[1])
        groups.setdefault(key, []).append(idx)

    results: List[Optional[Dict[str, np.ndarray]]] = [None] * len(motions)

    for (t_len, joint_count), indices in groups.items():
        # Chunk into batches of at most 32 to bound memory
        chunk_size = 32
        for chunk_start in range(0, len(indices), chunk_size):
            chunk_indices = indices[chunk_start : chunk_start + chunk_size]
            n_chunk = len(chunk_indices)

            # Stack to (N, T, J, 3) — float32 is sufficient; Welch promotes internally
            pos = np.stack([motions[i][:, :, CH_POS] for i in chunk_indices], axis=0).astype(np.float32)

            # Spectral flatness — batched Welch PSD over all motions×joints×channels
            sf = np.zeros((n_chunk, joint_count), dtype=np.float64)
            nps = min(nperseg, t_len)
            if nps >= 2:
                pos_flat = np.transpose(pos, (1, 0, 2, 3)).reshape(t_len, -1)  # (T, N*J*3)
                _, psd_all = scipy.signal.welch(pos_flat, nperseg=nps, axis=0)  # (n_freq, N*J*3)
                psd_all = psd_all + 1e-30
                psd_all = psd_all.reshape(-1, n_chunk, joint_count, 3)  # (n_freq, N, J, 3)
                log_mean = np.mean(np.log(psd_all), axis=0)       # (N, J, 3)
                arith_mean = np.mean(psd_all, axis=0)             # (N, J, 3)
                sf = (np.exp(log_mean) / np.maximum(arith_mean, 1e-30)).mean(axis=-1)

            # Jerk — fully vectorized over batch
            jerk = np.zeros((n_chunk, joint_count), dtype=np.float64)
            if t_len >= 4:
                jerk3 = np.diff(np.diff(np.diff(pos, axis=1), axis=1), axis=1)  # (N, T-3, J, 3)
                per_joint_jerk = (jerk3 ** 2).mean(axis=(1, 3))                 # (N, J)
                per_joint_var = pos.var(axis=1).mean(axis=-1)                   # (N, J)
                jerk = per_joint_jerk / (per_joint_var + 1e-10)

            # Snap — fully vectorized over batch
            snap = np.zeros((n_chunk, joint_count), dtype=np.float64)
            if t_len >= 5:
                snap4 = np.diff(np.diff(np.diff(np.diff(pos, axis=1), axis=1), axis=1), axis=1)  # (N, T-4, J, 3)
                snap_rms = np.sqrt((snap4 ** 2).mean(axis=(1, 3)))                               # (N, J)
                pos_rms = np.sqrt((pos ** 2).mean(axis=(1, 3))) + 1e-10                          # (N, J)
                snap = snap_rms / pos_rms

            for k, idx in enumerate(chunk_indices):
                results[idx] = {
                    "spectral_flatness": sf[k],
                    "jerk_norm": jerk[k],
                    "snap_norm": snap[k],
                }

    return results  # type: ignore[return-value]


def _active_joint_groups(joint_groups: Mapping[str, np.ndarray]) -> List[Tuple[str, np.ndarray]]:
    groups: List[Tuple[str, np.ndarray]] = []
    for name in _GROUP_ORDER:
        indices = np.asarray(joint_groups.get(name, np.zeros(0, dtype=np.int64)), dtype=np.int64)
        if indices.size:
            groups.append((name, indices))
    return groups


def _group_scalar_means(values: np.ndarray, joint_groups: Mapping[str, np.ndarray]) -> Dict[str, float]:
    grouped: Dict[str, float] = {}
    for group_name, group_indices in _active_joint_groups(joint_groups):
        grouped[group_name] = float(np.mean(np.asarray(values[group_indices], dtype=np.float64)))
    return grouped


def _group_psd_means(psd: np.ndarray, joint_groups: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    grouped: Dict[str, np.ndarray] = {}
    for group_name, group_indices in _active_joint_groups(joint_groups):
        grouped[group_name] = np.asarray(psd[group_indices], dtype=np.float64).mean(axis=0)
    return grouped


def _group_bone_time_series(values: np.ndarray, joint_groups: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    grouped: Dict[str, np.ndarray] = {}
    value_arr = np.asarray(values, dtype=np.float64)
    if value_arr.ndim != 2 or value_arr.shape[1] == 0:
        return grouped

    for group_name, group_indices in _active_joint_groups(joint_groups):
        bone_indices = np.asarray(group_indices, dtype=np.int64)
        bone_indices = bone_indices[bone_indices > 0] - 1
        bone_indices = bone_indices[(bone_indices >= 0) & (bone_indices < value_arr.shape[1])]
        if bone_indices.size == 0:
            continue
        group_values = np.asarray(value_arr[:, bone_indices], dtype=np.float64)
        finite_mask = np.isfinite(group_values)
        if not np.any(finite_mask):
            continue
        valid_counts = finite_mask.sum(axis=1)
        finite_values = np.where(finite_mask, group_values, 0.0)
        grouped[group_name] = np.divide(
            finite_values.sum(axis=1),
            valid_counts,
            out=np.full(value_arr.shape[0], np.nan, dtype=np.float64),
            where=valid_counts > 0,
        )
    return grouped


def _flatten_group_series_samples(
    grouped_samples: Sequence[Mapping[str, np.ndarray]],
    sample_weights: np.ndarray,
    group_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    values: List[np.ndarray] = []
    weights: List[np.ndarray] = []
    sample_weight_arr = np.asarray(sample_weights, dtype=np.float64).reshape(-1)

    for sample_index, grouped in enumerate(grouped_samples):
        if sample_index >= sample_weight_arr.size or group_name not in grouped:
            continue
        series = np.asarray(grouped[group_name], dtype=np.float64).reshape(-1)
        if series.size == 0:
            continue
        finite_mask = np.isfinite(series)
        if not np.any(finite_mask):
            continue
        series = series[finite_mask]
        values.append(series)
        weights.append(np.full(series.shape, sample_weight_arr[sample_index] / float(series.size), dtype=np.float64))

    if not values:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    return np.concatenate(values), np.concatenate(weights)


def _weighted_mean_vector(vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    vector_arr = np.asarray(vectors, dtype=np.float64)
    weight_arr = _normalize_weights(weights)
    return np.sum(vector_arr * weight_arr[:, None], axis=0)


def _score_query_against_reference(
    query_values: np.ndarray,
    query_weights: np.ndarray,
    reference_values: np.ndarray,
    reference_weights: np.ndarray,
    abs_floor: float,
) -> Dict[str, float]:
    reference_median = _weighted_quantile(reference_values, reference_weights, 0.5)
    scale = max(_weighted_iqr(reference_values, reference_weights), abs_floor)
    normalized_deviation = np.abs(np.asarray(query_values, dtype=np.float64) - reference_median) / scale
    scores = np.asarray([_distance_to_score(value) for value in normalized_deviation], dtype=np.float64)
    return {
        "reference_median": float(reference_median),
        "scale": float(scale),
        "normalized_deviation": _weighted_average(normalized_deviation, query_weights),
        "score": _weighted_average(scores, query_weights),
    }




# ---------------------------------------------------------------------------
# FK-based bone-length drift scoring (scorer-specific)
# ---------------------------------------------------------------------------
SIGMOID_THRESHOLD = 200.0  # drift percentage at which score = 0.5
SIGMOID_K = 0.01636

def _sigmoid_score(drift_pct: float) -> float:
    """Score a single drift percentage value.

    Distribution: 0%->~0.96, 20%->~0.95, 100%->~0.84, 200%->0.50, 300%->~0.16
    """
    return float(1.0 / (1.0 + math.exp(SIGMOID_K * (drift_pct - SIGMOID_THRESHOLD))))


def _compute_bone_length_drift_from_motion(
    motion: np.ndarray,   # [T, J, 13]  motion features
    parents: np.ndarray,  # [J]
    offsets: np.ndarray,  # [J, 3]
) -> np.ndarray:
    """Compute bone-length drift from a motion feature array.

    Uses recover_from_features + positions_global to produce consistent
    world-space positions, then computes per-edge drift relative to frame 0.

    Returns (T, E) drift array.
    """
    parents = np.asarray(parents, dtype=np.int32)
    offsets = np.asarray(offsets, dtype=np.float64)

    try:
        from utils.npy_roundtrip_utils import recover_from_features
        from motion_lib.Animation import positions_global

        recovered_anim, _has_animated_pos = recover_from_features(
            motion,
            parents,
            offsets,
        )
        world_pos = np.asarray(positions_global(recovered_anim), dtype=np.float64)
    except Exception:
        return np.full((motion.shape[0], 0), np.nan, dtype=np.float64)

    edge_parent_idx, edge_child_idx = resolve_comparison_edges(parents, offsets)

    if edge_parent_idx.size == 0:
        return np.full((motion.shape[0], 0), np.nan, dtype=np.float64)

    return compute_bone_length_drift(world_pos, edge_parent_idx, edge_child_idx)


def _summarize_drift_array(drift: np.ndarray) -> Dict[str, float]:
    """Summarize a (T, E) drift array into core percentage statistics."""
    abs_drift_pct = np.abs(drift * 100.0)
    abs_drift_pct_finite = abs_drift_pct[np.isfinite(abs_drift_pct)]

    if abs_drift_pct_finite.size == 0:
        return {
            "median_abs_drift_pct": 0.0,
            "mean_abs_drift_pct": 0.0,
            "max_abs_drift_pct": 0.0,
        }

    return {
        "median_abs_drift_pct": float(np.median(abs_drift_pct_finite)),
        "mean_abs_drift_pct": float(np.mean(abs_drift_pct_finite)),
        "max_abs_drift_pct": float(np.max(abs_drift_pct_finite)),
    }


def _score_bone_length_from_drift(
    motions: List[np.ndarray],
    motion_weights: np.ndarray,
    parents: np.ndarray,
    offsets: np.ndarray,
) -> Dict[str, float]:
    """Score bone length stability using FK-based drift statistics.

    For each motion, compute frame-to-frame bone length drift (relative to frame 0)
    using world-space FK positions. Extract median_abs, mean_abs, max_abs drift
    percentages and score each using sigmoid mapping.

    Returns dict with 'score' (0-1) and raw drift statistics.
    """
    all_stats: List[Dict[str, float]] = []
    selected_motion_weights: List[float] = []

    motion_weight_arr = np.asarray(motion_weights, dtype=np.float64).reshape(-1)

    for motion_idx, motion in enumerate(motions):
        if motion_idx >= motion_weight_arr.size:
            continue

        drift = _compute_bone_length_drift_from_motion(
            motion, parents, offsets
        )  # (T, E)

        if drift.size == 0:
            continue

        if not np.isfinite(np.asarray(drift, dtype=np.float64)).any():
            continue

        stats = _summarize_drift_array(drift)
        all_stats.append(stats)
        selected_motion_weights.append(float(motion_weight_arr[motion_idx]))

    if not all_stats:
        return {
            "score": 0.0,
            "median_abs_drift_pct": 0.0,
            "mean_abs_drift_pct": 0.0,
            "max_abs_drift_pct": 0.0,
        }

    median_abs = _weighted_average(
        np.asarray([s["median_abs_drift_pct"] for s in all_stats]),
        np.asarray(selected_motion_weights, dtype=np.float64),
    )
    mean_abs = _weighted_average(
        np.asarray([s["mean_abs_drift_pct"] for s in all_stats]),
        np.asarray(selected_motion_weights, dtype=np.float64),
    )
    max_abs = _weighted_average(
        np.asarray([s["max_abs_drift_pct"] for s in all_stats]),
        np.asarray(selected_motion_weights, dtype=np.float64),
    )

    score_median = _sigmoid_score(median_abs)
    score_mean = _sigmoid_score(mean_abs)
    score_max = _sigmoid_score(max_abs)

    score = (score_median + score_mean + score_max) / 3.0

    return {
        "score": float(np.clip(score, 0.0, 1.0)),
        "median_abs_drift_pct": median_abs,
        "mean_abs_drift_pct": mean_abs,
        "max_abs_drift_pct": max_abs,
        "score_median_abs": score_median,
        "score_mean_abs": score_mean,
        "score_max_abs": score_max,
    }



class DistributionMotionQualityScorer:
    """Low-shot weighted-reference motion quality scorer."""

    def __init__(self, dataset_root: Optional[str] = None):
        self.dataset_root = dataset_root
        self._cond_lookup = load_cond_dict(dataset_root)
        self._query_cond_lookup = dict(self._cond_lookup)
        self._custom_cond_keys: set[str] = set()
        self._joint_group_cache: Dict[Tuple[str, int], Tuple[Dict[str, np.ndarray], str]] = {}

    def register_cond(self, cond_dict: dict) -> None:
        """Register query skeleton metadata from a custom cond.npy.

        Custom entries are used to interpret query skeletons, but the dataset
        reference baseline remains the default cond.npy loaded by the scorer.
        """
        for key, entry in cond_dict.items():
            key_str = str(key)
            lowered = key_str.strip().lower()
            target_key = key_str
            for existing_key in self._query_cond_lookup:
                if str(existing_key).strip().lower() == lowered:
                    target_key = str(existing_key)
                    break
            self._query_cond_lookup[target_key] = entry
            self._custom_cond_keys.add(target_key)
            for cache_key in list(self._joint_group_cache):
                if cache_key[0] == target_key:
                    del self._joint_group_cache[cache_key]

    def evaluate(
        self,
        motions: List[np.ndarray],
        object_type: str,
        action_tags: str,
        top_k_species: int = 5,
    ) -> DistributionEvalReport:
        query_motions = [
            motion.astype(np.float32)
            for motion in motions
            if motion.ndim == 3 and motion.shape[-1] == 13 and motion.shape[0] >= _MIN_CLIP_FRAMES
        ]
        if not query_motions:
            raise ValueError(f"Need at least one valid query motion with shape (T, J, 13) and T >= {_MIN_CLIP_FRAMES}")

        query_joint_counts = {motion.shape[1] for motion in query_motions}
        if len(query_joint_counts) != 1:
            raise ValueError("All input motions must share the same joint count")

        object_key = self._resolve_object_type_key(object_type)
        query_joint_groups, joint_group_source = self._resolve_joint_groups(object_key, next(iter(query_joint_counts)))
        reference_kwargs: Dict[str, object] = {}
        if object_key in self._custom_cond_keys:
            reference_kwargs["cond_lookup"] = self._cond_lookup
            reference_kwargs["query_cond"] = self._query_cond_lookup[object_key]
        reference_bank = build_weighted_reference_bank(
            object_type=object_key,
            action_tags=action_tags,
            dataset_root=self.dataset_root,
            top_k_species=top_k_species,
            min_frames=_MIN_CLIP_FRAMES,
            **reference_kwargs,
        )

        min_t = min(
            min(motion.shape[0] for motion in query_motions),
            min(clip.motion.shape[0] for clip in reference_bank.clips),
        )
        nperseg = max(4, min(64, min_t))
        query_weights = _normalize_weights(np.asarray([motion.shape[0] for motion in query_motions], dtype=np.float64))
        reference_weights = reference_bank.clip_weights

        local = self._compute_low_shot(
            query_motions,
            query_weights,
            reference_bank.clips,
            reference_weights,
            nperseg,
            query_joint_groups,
            object_key,
        )

        return DistributionEvalReport(
            object_type=object_key,
            action_tags=str(action_tags or "").strip(),
            n_input=len(query_motions),
            n_reference=len(reference_bank.clips),
            input_total_frames=int(sum(motion.shape[0] for motion in query_motions)),
            reference_total_frames=reference_bank.total_reference_frames,
            scoring_mode="low_shot_weighted_reference",
            top_k_species=reference_bank.top_k_species,
            reference_species=[
                {
                    "object_type": species.object_type,
                    "cosine_distance": round(species.cosine_distance, 4),
                    "species_weight": round(species.species_weight, 4),
                    "clip_count": species.clip_count,
                    "total_frames": species.total_frames,
                }
                for species in reference_bank.species
            ],
            overall_score=local["score"],
            spectral_flatness_score=local["component_scores"]["spectral_flatness"],
            jerk_score=local["component_scores"]["jerk_norm"],
            snap_score=local["component_scores"]["snap_norm"],
            bone_length_score=local["component_scores"]["bone_length"],
            raw={
                "nperseg": nperseg,
                "joint_group_source": joint_group_source,
                "effective_reference_mass": reference_bank.effective_reference_mass,
                **local.get("raw", {}),
            },
        )

    def _resolve_object_type_key(self, object_type: str) -> str:
        if object_type in self._query_cond_lookup:
            return object_type
        lowered = str(object_type).strip().lower()
        for key in self._query_cond_lookup:
            if str(key).lower() == lowered:
                return str(key)
        raise KeyError(f"Unknown object_type {object_type!r} in cond.npy")

    def _resolve_joint_groups(self, object_type: str, n_joints: int) -> Tuple[Dict[str, np.ndarray], str]:
        cache_key = (object_type, n_joints)
        if cache_key in self._joint_group_cache:
            return self._joint_group_cache[cache_key]

        cond = self._query_cond_lookup.get(object_type)
        if cond is not None and len(cond.get("parents", [])) != n_joints:
            raise ValueError(
                f"Object type {object_type!r} expects {len(cond.get('parents', []))} joints, got {n_joints}"
            )

        if cond is None:
            raise KeyError(
                f"Object type {object_type!r} not found in cond.npy. "
                f"Joint group resolution requires a matching skeleton definition."
            )

        result = _build_joint_groups_from_cond(cond, n_joints)
        self._joint_group_cache[cache_key] = result
        return result

    def _compute_low_shot(
        self,
        query_motions: List[np.ndarray],
        query_weights: np.ndarray,
        reference_clips: List[ReferenceClip],
        reference_weights: np.ndarray,
        nperseg: int,
        query_joint_groups: Mapping[str, np.ndarray],
        object_key: str,
    ) -> dict:
        query_active_groups = [name for name, _ in _active_joint_groups(query_joint_groups)]

        # Batch feature extraction for query motions
        query_features_list = _compute_features_batch(query_motions, nperseg)
        query_local = []
        for motion, features in zip(query_motions, query_features_list):
            query_local.append({
                key: _group_scalar_means(values, query_joint_groups)
                for key, values in features.items()
            })

        # Batch feature extraction for reference clips
        ref_motions = [clip.motion for clip in reference_clips]
        ref_features_list = _compute_features_batch(ref_motions, nperseg)
        reference_local = []
        for clip, features in zip(reference_clips, ref_features_list):
            clip_joint_groups, _ = self._resolve_joint_groups(clip.object_type, clip.motion.shape[1])
            reference_local.append({
                key: _group_scalar_means(values, clip_joint_groups)
                for key, values in features.items()
            })

        # Bone length scoring: use FK-based drift directly (no reference comparison)
        query_cond = self._query_cond_lookup.get(object_key)
        if query_cond is None:
            raise KeyError(
                f"Object type {object_key!r} not found in cond.npy. "
                f"Bone-length drift scoring requires a matching skeleton definition. "
                f"Available types: {sorted(self._query_cond_lookup.keys())}"
            )

        q_parents = np.asarray(query_cond["parents"], dtype=np.int32)
        q_offsets = np.asarray(query_cond["offsets"], dtype=np.float64)
        bone_length_result = _score_bone_length_from_drift(
            query_motions, query_weights, q_parents, q_offsets
        )

        metric_group_scores: Dict[str, Dict[str, float]] = {}
        metric_group_dev: Dict[str, Dict[str, float]] = {}
        metric_group_scale: Dict[str, Dict[str, float]] = {}
        metric_group_tolerance: Dict[str, Dict[str, float]] = {}
        metric_group_penalty: Dict[str, Dict[str, float]] = {}

        for metric_name in ["spectral_flatness", "jerk_norm", "snap_norm"]:
            metric_group_scores[metric_name] = {}
            metric_group_dev[metric_name] = {}
            metric_group_scale[metric_name] = {}
            metric_group_tolerance[metric_name] = {}
            metric_group_penalty[metric_name] = {}
            for group_name in query_active_groups:
                reference_pairs = [
                    (sample[metric_name][group_name], reference_weights[index])
                    for index, sample in enumerate(reference_local)
                    if group_name in sample[metric_name]
                ]
                if not reference_pairs:
                    continue
                reference_values = np.asarray([value for value, _ in reference_pairs], dtype=np.float64)
                reference_group_weights = _normalize_weights(np.asarray([weight for _, weight in reference_pairs], dtype=np.float64))
                query_values = np.asarray([sample[metric_name][group_name] for sample in query_local], dtype=np.float64)
                detail = _score_query_against_reference(
                    query_values,
                    query_weights,
                    reference_values,
                    reference_group_weights,
                    _metric_abs_floor(metric_name),
                )
                metric_group_scores[metric_name][group_name] = detail["score"]
                metric_group_dev[metric_name][group_name] = detail["normalized_deviation"]
                metric_group_scale[metric_name][group_name] = detail["scale"]
                metric_group_penalty[metric_name][group_name] = 1.0 - detail["score"]



        # Bone length: single overall score (no per-group breakdown)
        bone_length_score = bone_length_result["score"]

        metric_weights = {
            "jerk_norm": 0.328,
            "snap_norm": 0.228,
            "spectral_flatness": 0.228,
            "bone_length": 0.228,
        }
        # Build component_scores in weight-descending order (Python 3.7+ preserves dict insertion order)
        component_scores = {}
        for metric_name in ["jerk_norm", "snap_norm", "spectral_flatness"]:
            component_scores[metric_name] = (
                float(np.mean(list(metric_group_scores[metric_name].values())))
                if metric_group_scores[metric_name] else 0.0
            )
        component_scores["bone_length"] = bone_length_score
        group_scores = {
            group_name: float(np.mean([
                metric_group_scores[metric_name][group_name]
                for metric_name in metric_weights
                if group_name in metric_group_scores.get(metric_name, {})
            ]))
            for group_name in query_active_groups
            if any(group_name in metric_group_scores.get(metric_name, {}) for metric_name in metric_weights)
        }
        score = float(sum(metric_weights[name] * component_scores[name] for name in metric_weights))

        return {
            "score": float(np.clip(score, 0.0, 1.0)),
            "component_scores": component_scores,
            "raw": {
                "component_scores": component_scores,
                "joint_group_scores": group_scores,
                "metric_group_scores": metric_group_scores,
                "metric_group_normalized_deviation": metric_group_dev,
                "metric_group_scale": metric_group_scale,
                "metric_group_tolerance": metric_group_tolerance,
                "metric_group_penalty": metric_group_penalty,
                "active_joint_groups": query_active_groups,
                "bone_length_drift_stats": {
                    "median_abs_drift_pct": round(bone_length_result.get("median_abs_drift_pct", 0.0), 4),
                    "mean_abs_drift_pct": round(bone_length_result.get("mean_abs_drift_pct", 0.0), 4),
                    "max_abs_drift_pct": round(bone_length_result.get("max_abs_drift_pct", 0.0), 4),
                    "score_median_abs": round(bone_length_result.get("score_median_abs", 0.0), 4),
                    "score_mean_abs": round(bone_length_result.get("score_mean_abs", 0.0), 4),
                    "score_max_abs": round(bone_length_result.get("score_max_abs", 0.0), 4),
                },
            },
        }