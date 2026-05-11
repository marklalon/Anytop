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

    naturalness_score: float
    spectral_flatness_score: float
    jerk_score: float
    snap_score: float
    bone_length_score: float
    bone_rotation_score: float

    raw: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "naturalness_score": round(self.naturalness_score, 4),
            "detail": {
                "jerk_score (w=0.328)": round(self.jerk_score, 4),
                "snap_score (w=0.228)": round(self.snap_score, 4),
                "spectral_flatness_score (w=0.228)": round(self.spectral_flatness_score, 4),
                "bone_length_score (w=0.168)": round(self.bone_length_score, 4),
                "bone_rotation_score (w=0.06)": round(self.bone_rotation_score, 4),
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
            f"  | {'Joint naturalness':<{width}}| {self.naturalness_score:6.4f} |",
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
    if name == "bone_rotation":
        return 0.02
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
    t_len, joint_count, _ = motion.shape
    pos = motion[:, :, CH_POS].astype(np.float64)

    spectral_flatness = np.zeros(joint_count)
    jerk_norm = np.zeros(joint_count)
    snap_norm = np.zeros(joint_count)

    for joint_index in range(joint_count):
        flatness_values = []
        for channel_index in range(3):
            freqs_j, psd = _welch_psd(pos[:, joint_index, channel_index], nperseg)
            log_mean = float(np.mean(np.log(psd)))
            arith_mean = float(np.mean(psd))
            flatness_values.append(math.exp(log_mean) / max(arith_mean, 1e-30))
        spectral_flatness[joint_index] = float(np.mean(flatness_values))

    if t_len >= 4:
        jerk3 = np.diff(np.diff(np.diff(pos, axis=0), axis=0), axis=0)
        per_joint_jerk = (jerk3 ** 2).mean(axis=(0, 2))
        per_joint_var = pos.var(axis=0).mean(axis=-1)
        jerk_norm = per_joint_jerk / (per_joint_var + 1e-10)

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
# Bone rotation excess scoring
# ---------------------------------------------------------------------------
_BONE_ROTATION_PENALTY_K = 5.0


def _score_bone_rotation_excess(
    query_values: np.ndarray,
    query_weights: np.ndarray,
    reference_series_with_weights: Sequence[Tuple[np.ndarray, float]],
    abs_floor: float = 1e-6,
    penalty_k: float = _BONE_ROTATION_PENALTY_K,
) -> Dict[str, float]:
    """Score bone rotation by measuring how much query exceeds a reference max threshold.

    1. Build ``max_ref`` = weighted median of per-clip maximum rotation angles.
    2. For each query sample, excess = max(0, value - max_ref).
    3. Normalise excess by max_ref (or abs_floor) and apply logistic-style penalty.

    Returns score in [0, 1] where 1 means no excess at all.
    """
    # Step 1: per-clip max → weighted median
    clip_maxes: List[float] = []
    clip_max_weights: List[float] = []
    for series, clip_weight in reference_series_with_weights:
        s = np.asarray(series, dtype=np.float64).reshape(-1)
        finite = s[np.isfinite(s)]
        if finite.size == 0:
            continue
        clip_maxes.append(float(np.max(finite)))
        clip_max_weights.append(float(clip_weight))

    if not clip_maxes:
        return {
            "max_ref": 0.0,
            "score": 0.0,
            "normalized_excess": 0.0,
            "penalty": 1.0,
        }

    max_ref = _weighted_quantile(
        np.asarray(clip_maxes, dtype=np.float64),
        np.asarray(clip_max_weights, dtype=np.float64),
        0.5,
    )
    normalizer = max(max_ref, abs_floor)

    # Step 2 & 3: per-sample excess → score
    query_arr = np.asarray(query_values, dtype=np.float64).reshape(-1)
    weight_arr = _normalize_weights(np.asarray(query_weights, dtype=np.float64).reshape(-1))

    excess = np.maximum(0.0, query_arr - max_ref)
    excess_norm = excess / normalizer
    scores = 1.0 / (1.0 + excess_norm * penalty_k)

    return {
        "max_ref": float(max_ref),
        "normalized_excess": float(_weighted_average(excess_norm, weight_arr)),
        "penalty": float(_weighted_average(1.0 - scores, weight_arr)),
        "score": float(_weighted_average(scores, weight_arr)),
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


def _compute_bone_rotation_angle(
    motion: np.ndarray,
    parents: np.ndarray,
) -> np.ndarray:
    """Return per-frame bone rotation angle (change in direction)  (T-1, J_bones).

    For each non-root joint, the bone direction is the normalised local position
    vector.  We measure the angle between consecutive-frame direction vectors.
    """
    pos = motion[:, :, CH_POS].astype(np.float64)  # (T, J, 3)
    t_len, j_count, _ = pos.shape
    if t_len < 2:
        return np.zeros((0, j_count - 1), dtype=np.float64)

    norms = np.linalg.norm(pos, axis=-1, keepdims=True)
    directions = pos / (norms + 1e-10)  # (T, J, 3)

    bone_indices = np.arange(1, j_count, dtype=np.int64)
    dir_bones = directions[:, bone_indices, :]  # (T, B, 3)

    dot = np.sum(dir_bones[:-1] * dir_bones[1:], axis=-1).clip(-1.0, 1.0)
    angles = np.arccos(dot)  # (T-1, B)  radians
    return angles


class DistributionMotionQualityScorer:
    """Low-shot weighted-reference motion quality scorer."""

    def __init__(self, fps: int = 30, dataset_root: Optional[str] = None):
        self.fps = fps
        self.dataset_root = dataset_root
        self._cond_lookup = load_cond_dict(dataset_root)
        self._joint_group_cache: Dict[Tuple[str, int], Tuple[Dict[str, np.ndarray], str]] = {}

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
        reference_bank = build_weighted_reference_bank(
            object_type=object_key,
            action_tags=action_tags,
            dataset_root=self.dataset_root,
            top_k_species=top_k_species,
            min_frames=_MIN_CLIP_FRAMES,
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
            naturalness_score=local["score"],
            spectral_flatness_score=local["component_scores"]["spectral_flatness"],
            jerk_score=local["component_scores"]["jerk_norm"],
            snap_score=local["component_scores"]["snap_norm"],
            bone_length_score=local["component_scores"]["bone_length"],
            bone_rotation_score=local["component_scores"]["bone_rotation"],
            raw={
                "nperseg": nperseg,
                "joint_group_source": joint_group_source,
                "effective_reference_mass": reference_bank.effective_reference_mass,
                **local.get("raw", {}),
            },
        )

    def _resolve_object_type_key(self, object_type: str) -> str:
        if object_type in self._cond_lookup:
            return object_type
        lowered = str(object_type).strip().lower()
        for key in self._cond_lookup:
            if str(key).lower() == lowered:
                return str(key)
        raise KeyError(f"Unknown object_type {object_type!r} in cond.npy")

    def _resolve_joint_groups(self, object_type: str, n_joints: int) -> Tuple[Dict[str, np.ndarray], str]:
        cache_key = (object_type, n_joints)
        if cache_key in self._joint_group_cache:
            return self._joint_group_cache[cache_key]

        cond = self._cond_lookup.get(object_type)
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
        query_bone_groups = [name for name in query_active_groups if name != "root"]
        query_local = []
        query_bone_rotations = []
        for motion in query_motions:
            features = _compute_features(motion, nperseg)
            query_local.append({
                key: _group_scalar_means(values, query_joint_groups)
                for key, values in features.items()
            })
            query_bone_rotations.append(
                _group_bone_time_series(_compute_bone_rotation_angle(motion, np.zeros(0, dtype=np.int64)), query_joint_groups)
            )

        reference_local = []
        reference_bone_rotations = []
        for clip in reference_clips:
            clip_joint_groups, _ = self._resolve_joint_groups(clip.object_type, clip.motion.shape[1])
            features = _compute_features(clip.motion, nperseg)
            reference_local.append({
                key: _group_scalar_means(values, clip_joint_groups)
                for key, values in features.items()
            })
            reference_bone_rotations.append(
                _group_bone_time_series(_compute_bone_rotation_angle(clip.motion, np.zeros(0, dtype=np.int64)), clip_joint_groups)
            )

        # Bone length scoring: use FK-based drift directly (no reference comparison)
        query_cond = self._cond_lookup.get(object_key)
        if query_cond is None:
            raise KeyError(
                f"Object type {object_key!r} not found in cond.npy. "
                f"Bone-length drift scoring requires a matching skeleton definition. "
                f"Available types: {sorted(self._cond_lookup.keys())}"
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



        # Bone rotation: excess-based scoring (how much rotation exceeds reference max)
        metric_group_scores["bone_rotation"] = {}
        metric_group_dev["bone_rotation"] = {}
        metric_group_scale["bone_rotation"] = {}
        metric_group_tolerance["bone_rotation"] = {}
        metric_group_penalty["bone_rotation"] = {}
        for group_name in query_bone_groups:
            query_values, query_value_weights = _flatten_group_series_samples(
                query_bone_rotations,
                query_weights,
                group_name,
            )
            if query_values.size == 0:
                continue
            # Gather per-clip series with their original clip weights.
            reference_series_with_weights: List[Tuple[np.ndarray, float]] = []
            for clip_index, clip_sample in enumerate(reference_bone_rotations):
                if clip_index >= reference_weights.size or group_name not in clip_sample:
                    continue
                reference_series_with_weights.append(
                    (
                        np.asarray(clip_sample[group_name], dtype=np.float64),
                        float(reference_weights[clip_index]),
                    )
                )
            if not reference_series_with_weights:
                continue
            detail = _score_bone_rotation_excess(
                query_values,
                query_value_weights,
                reference_series_with_weights,
                abs_floor=_metric_abs_floor("bone_rotation"),
            )
            metric_group_scores["bone_rotation"][group_name] = detail["score"]
            metric_group_dev["bone_rotation"][group_name] = detail["normalized_excess"]
            metric_group_scale["bone_rotation"][group_name] = detail["max_ref"]
            metric_group_tolerance["bone_rotation"][group_name] = detail["max_ref"]
            metric_group_penalty["bone_rotation"][group_name] = detail["penalty"]

        # Bone length: single overall score (no per-group breakdown)
        bone_length_score = bone_length_result["score"]

        metric_weights = {
            "jerk_norm": 0.328,
            "snap_norm": 0.228,
            "spectral_flatness": 0.228,
            "bone_length": 0.168,
            "bone_rotation": 0.06,
        }
        # Build component_scores in weight-descending order (Python 3.7+ preserves dict insertion order)
        component_scores = {}
        for metric_name in ["jerk_norm", "snap_norm", "spectral_flatness"]:
            component_scores[metric_name] = (
                float(np.mean(list(metric_group_scores[metric_name].values())))
                if metric_group_scores[metric_name] else 0.0
            )
        component_scores["bone_length"] = bone_length_score
        component_scores["bone_rotation"] = (
            float(np.mean(list(metric_group_scores["bone_rotation"].values())))
            if metric_group_scores["bone_rotation"] else 0.0
        )
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