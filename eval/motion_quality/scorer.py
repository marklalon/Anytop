"""
DistributionMotionQualityScorer
================================

Evaluates motion quality at the **distribution level**.  Single-clip scoring
is not supported; both the generated and clean sets must contain ≥32 clips.

Two evaluation dimensions are combined with equal 50/50 weight:

Macro distribution fidelity  [0, 1]
    Robust per-feature similarity between fixed-size feature vectors extracted
    from each clip.  Raw 1-D Wasserstein distances are normalised by a
    combined-set IQR with a feature-family absolute floor, then aggregated with
    equal weight across feature families and joint groups (root / axial /
    limbs).  Macro aggregation is therefore sensitive to trunk and spine motion
    instead of averaging it away into all-joint statistics.

Local joint naturalness  [0, 1]
    Distribution comparison of per-joint spectral and smoothness features:
    spectral flatness (periodicity vs noise), PSD shape (Jensen-Shannon
    divergence), autocorrelation peak, activity-normalised jerk, and snap
    (4th temporal derivative).  Each metric is compared per semantic joint
    group (root / axial / limbs) via 1-D Wasserstein distance, normalised by
    a combined-set IQR with semantic absolute floors, then aggregated.

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
import scipy.stats

from data_loaders.truebones.offline_reference_dataset import load_cond_dict

from .reference_stats import CH_POS, CH_ROT, CH_VEL

MIN_BATCH_SIZE = 32
_MIN_CLIP_FRAMES = 8


# ─────────────────────────────────────────────────────────────────────────────
# Report dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DistributionEvalReport:
    """Full distribution-level quality report."""

    # -- Identification --
    object_type: Optional[str]
    n_generated: int
    n_clean: int

    # -- Macro-level distribution fidelity --
    macro_fidelity_score: float           # [0, 1], higher = better
    macro_feature_group_scores: Dict[str, float]
    macro_joint_group_scores: Dict[str, float]
    macro_joint_group_sizes: Dict[str, int]
    macro_top_deviating_features: List[Tuple[str, float]]  # top-5 worst normalized distances

    # -- Local joint naturalness --
    local_naturalness_score: float        # [0, 1], higher = better
    local_psd_jsd_root: float
    local_psd_jsd_limbs: float
    local_spectral_flatness_w1: float
    local_jerk_w1: float
    local_acf_peak_w1: float
    local_spectral_centroid_w1: float
    local_snap_w1: float

    # -- Combined --
    overall_score: float                  # 0.5 * macro + 0.5 * local

    # -- Diagnostics --
    raw: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def as_dict(self) -> dict:
        return {
            "overall_score":           round(self.overall_score, 4),
            "macro_fidelity_score":    round(self.macro_fidelity_score, 4),
            "local_naturalness_score": round(self.local_naturalness_score, 4),
            "macro": {
                "feature_group_scores": {
                    key: round(value, 4)
                    for key, value in self.macro_feature_group_scores.items()
                },
                "joint_group_scores": {
                    key: round(value, 4)
                    for key, value in self.macro_joint_group_scores.items()
                },
                "joint_group_sizes": dict(self.macro_joint_group_sizes),
                "top_deviating_features": [
                    {"feature": n, "normalized_distance": round(v, 6)}
                    for n, v in self.macro_top_deviating_features
                ],
            },
            "local": {
                "psd_jsd_root":           round(self.local_psd_jsd_root, 6),
                "psd_jsd_limbs":          round(self.local_psd_jsd_limbs, 6),
                "spectral_flatness_w1":   round(self.local_spectral_flatness_w1, 6),
                "jerk_w1":                round(self.local_jerk_w1, 6),
                "acf_peak_w1":            round(self.local_acf_peak_w1, 6),
                "spectral_centroid_w1":   round(self.local_spectral_centroid_w1, 6),
                "snap_w1":                round(self.local_snap_w1, 6),
            },
            "meta": {
                "object_type": self.object_type,
                "n_generated": self.n_generated,
                "n_clean":     self.n_clean,
            },
            "raw": self.raw,
        }

    def __str__(self) -> str:
        W = 46
        lines = [
            "Distribution Motion Quality Report",
            f"  Object type : {self.object_type or 'unknown'}",
            f"  Samples     : {self.n_generated} generated  /  {self.n_clean} clean",
            "",
            f"  +{'-'*W}+--------+",
            f"  | {'Dimension':<{W}}| Score  |",
            f"  +{'-'*W}+--------+",
            f"  | {'Macro distribution fidelity  (w=0.50)':<{W}}| {self.macro_fidelity_score:5.3f}  |",
            f"  +{'-'*W}+--------+",
            f"  | {'Local joint naturalness      (w=0.50)':<{W}}| {self.local_naturalness_score:5.3f}  |",
            f"  +{'-'*W}+--------+",
            f"  | {'OVERALL SCORE':<{W}}| {self.overall_score:5.3f}  |",
            f"  +{'-'*W}+--------+",
            "",
            f"  Macro feature groups : {self.macro_feature_group_scores}",
            f"  Macro joint groups   : {self.macro_joint_group_scores}",
            f"  Macro joint sizes    : {self.macro_joint_group_sizes}",
            "",
            "  Top macro-feature deviations (generated vs clean):",
        ]
        for name, val in self.macro_top_deviating_features:
            lines.append(f"    {name:<50s}  ND = {val:.5f}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Statistical helpers
# ─────────────────────────────────────────────────────────────────────────────


def _psd_jsd(psd1: np.ndarray, psd2: np.ndarray) -> float:
    """Jensen-Shannon divergence between two non-negative arrays (treated as PSDs)."""
    p = psd1 / (psd1.sum() + 1e-30)
    q = psd2 / (psd2.sum() + 1e-30)
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log((a[mask] + 1e-30) / (b[mask] + 1e-30))))

    return max(0.0, 0.5 * (_kl(p, m) + _kl(q, m)))


_LOCAL_GROUP_ORDER = ("root", "axial", "limbs")


def _local_metric_abs_floor(name: str) -> float:
    if name == "psd_bin":
        return 0.01
    if name in {"spectral_flatness", "acf_peak"}:
        return 0.03
    if name == "spectral_centroid":
        return 0.015
    if name in {"jerk_norm", "snap_norm"}:
        return 0.05
    return 0.02


def _active_joint_groups(joint_groups: Mapping[str, np.ndarray]) -> List[Tuple[str, np.ndarray]]:
    groups: List[Tuple[str, np.ndarray]] = []
    for name in _LOCAL_GROUP_ORDER:
        indices = np.asarray(joint_groups.get(name, np.zeros(0, dtype=np.int64)), dtype=np.int64)
        if indices.size:
            groups.append((name, indices))
    return groups


def _score_distribution_distance(
    generated: np.ndarray,
    clean: np.ndarray,
    abs_floor: float,
) -> Dict[str, float]:
    raw_w1 = float(scipy.stats.wasserstein_distance(generated, clean))
    scale = max(_combined_iqr(generated, clean), abs_floor)
    normalized_distance = raw_w1 / scale
    return {
        "raw_w1": raw_w1,
        "scale": float(scale),
        "normalized_distance": float(normalized_distance),
        "score": _macro_distance_to_score(normalized_distance),
    }


def _macro_feature_group(name: str) -> str:
    if name.startswith("pos_"):
        return "pos"
    if name.startswith("rot_"):
        return "rot"
    if name.startswith("vel_") or name.startswith("vel_mag_"):
        return "vel"
    return "freq"


def _macro_joint_bucket(name: str) -> str:
    if "_axial_" in name:
        return "axial"
    if "_limbs_" in name:
        return "limbs"
    return "root"


def _macro_feature_abs_floor(name: str) -> float:
    """Semantic tolerance floor for macro features.

    Combined-IQR prevents near-zero-variance explosions, but zero-centred
    features can still produce an over-large normalized distance from a tiny
    absolute shift when both sets are narrow.  These floors encode a minimum
    practically-meaningful delta per feature family.
    """
    if "_lag1_" in name:
        return 0.10
    if name.startswith("freq_dom_ratio_"):
        return 0.03
    if name.startswith("spectral_centroid_"):
        return 0.02
    if name.startswith("vel_mag_"):
        return 0.02
    if name.startswith("vel_"):
        return 0.01
    if name.startswith("pos_") or name.startswith("rot_"):
        return 0.01
    return 0.01


def _combined_iqr(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.concatenate([a, b], axis=0)
    q25, q75 = np.percentile(pooled, [25, 75])
    return float(q75 - q25)


def _macro_distance_to_score(distance: float) -> float:
    """Smoothly map a normalized per-feature distance to [0, 1].

    A distance of 1.0 means the two sets differ by roughly one robust scale.
    That should reduce confidence, but not collapse the whole macro score.
    """
    return float(1.0 / (1.0 + distance * distance))


_MACRO_LIMB_KEYWORDS = (
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
_MACRO_AXIAL_KEYWORDS = (
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
    return any(keyword in label for keyword in _MACRO_LIMB_KEYWORDS)


def _looks_like_axial_joint(label: str) -> bool:
    return any(keyword in label for keyword in _MACRO_AXIAL_KEYWORDS)


def _coerce_index_array(indices: Sequence[int] | np.ndarray | None, n_joints: int) -> np.ndarray:
    if indices is None:
        return np.zeros(0, dtype=np.int64)
    arr = np.asarray(list(indices), dtype=np.int64)
    if arr.size == 0:
        return arr
    return np.unique(arr[(arr >= 0) & (arr < n_joints)])


def _build_macro_joint_groups_from_cond(
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


def _fallback_macro_joint_groups(n_joints: int) -> Tuple[Dict[str, np.ndarray], str]:
    if n_joints <= 0:
        raise ValueError("Motion must contain at least one joint")
    return {
        "root": np.asarray([0], dtype=np.int64),
        "axial": np.arange(1, n_joints, dtype=np.int64),
        "limbs": np.zeros(0, dtype=np.int64),
    }, "fallback_non_root_axial"


def _append_joint_group_vector_features(
    feats: List[float],
    names: List[str],
    prefix: str,
    values: np.ndarray,
    joint_groups: Mapping[str, np.ndarray],
) -> None:
    root_index = int(joint_groups["root"][0])
    aggregates: list[tuple[str, np.ndarray]] = [("root", np.asarray(values[root_index], dtype=np.float64))]
    for group_name in ("axial", "limbs"):
        group_indices = joint_groups[group_name]
        if len(group_indices) == 0:
            zero = np.zeros(values.shape[1], dtype=np.float64)
            aggregates.append((f"{group_name}_mean", zero))
            aggregates.append((f"{group_name}_std", zero))
        else:
            group_values = np.asarray(values[group_indices], dtype=np.float64)
            aggregates.append((f"{group_name}_mean", group_values.mean(axis=0)))
            aggregates.append((f"{group_name}_std", group_values.std(axis=0)))
    for aggregate_name, aggregate_values in aggregates:
        for channel_index in range(aggregate_values.shape[0]):
            feats.append(float(aggregate_values[channel_index]))
            names.append(f"{prefix}_{aggregate_name}_{channel_index}")


def _append_joint_group_scalar_features(
    feats: List[float],
    names: List[str],
    prefix: str,
    values: np.ndarray,
    joint_groups: Mapping[str, np.ndarray],
) -> None:
    root_index = int(joint_groups["root"][0])
    feats.append(float(values[root_index]))
    names.append(f"{prefix}_root")
    for group_name in ("axial", "limbs"):
        group_indices = joint_groups[group_name]
        if len(group_indices) == 0:
            feats.append(0.0)
            names.append(f"{prefix}_{group_name}_mean")
            feats.append(0.0)
            names.append(f"{prefix}_{group_name}_std")
        else:
            group_values = np.asarray(values[group_indices], dtype=np.float64)
            feats.append(float(group_values.mean()))
            names.append(f"{prefix}_{group_name}_mean")
            feats.append(float(group_values.std()))
            names.append(f"{prefix}_{group_name}_std")


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction (module-level, stateless)
# ─────────────────────────────────────────────────────────────────────────────

def _welch_psd(sig: np.ndarray, nperseg: int) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD with fallback for very short signals.  Returns (freqs, psd)."""
    n = len(sig)
    nps = min(nperseg, n)
    if nps < 2:
        return np.array([0.0]), np.array([max(float(np.var(sig)), 1e-30)])
    freqs, psd = scipy.signal.welch(sig, nperseg=nps)
    return freqs, psd + 1e-30


def _compute_macro_features(
    motion: np.ndarray,
    nperseg: int,
    joint_groups: Mapping[str, np.ndarray],
) -> Tuple[np.ndarray, List[str]]:
    """Extract a fixed-size macro feature vector from one motion clip.

    Feature groups
    --------------
    A. Kinematic statistics:
       Per-channel temporal mean, std, and lag-1 autocorrelation for POS (3ch),
       ROT (6ch), VEL (3ch). Each statistic is aggregated by semantic joint
       groups: root, axial, and limbs, with within-group mean/std for the
       non-root groups.

    B. Velocity magnitude profile:
       5 percentiles (p10/p25/p50/p75/p90) of per-joint velocity magnitude,
       aggregated by semantic joint groups.

    C. Frequency-domain:
       Dominant-frequency energy ratio and spectral centroid for position
       channels, aggregated by semantic joint groups.

    Parameters
    ----------
    motion   : (T, J, 13) float32
    nperseg  : Welch segment length (same across all clips in a batch)

    Returns
    -------
    features : (D,) float64
    names    : list[str] of length D
    """
    T, J, _ = motion.shape
    feats: List[float] = []
    names: List[str] = []

    # ── Group A: Kinematic statistics ────────────────────────────────────────
    for ch_name, ch in [("pos", CH_POS), ("rot", CH_ROT), ("vel", CH_VEL)]:
        x = motion[:, :, ch].astype(np.float64)   # (T, J, C)
        C = x.shape[-1]

        mu = x.mean(axis=0)
        _append_joint_group_vector_features(feats, names, f"{ch_name}_mu", mu, joint_groups)

        sigma = x.std(axis=0)
        _append_joint_group_vector_features(feats, names, f"{ch_name}_sig", sigma, joint_groups)

        if T > 1:
            x0, x1 = x[:-1], x[1:]                    # (T-1, J, C)
            mu0, mu1 = x0.mean(0), x1.mean(0)
            num  = ((x0 - mu0) * (x1 - mu1)).mean(0)  # (J, C)
            denom = x0.std(0) * x1.std(0) + 1e-10
            acorr = (num / denom).mean(-1)             # (J,)
        else:
            acorr = np.zeros(J)

        _append_joint_group_scalar_features(feats, names, f"{ch_name}_lag1", acorr, joint_groups)

    # ── Group B: Velocity magnitude profile ──────────────────────────────────
    vel = motion[:, :, CH_VEL].astype(np.float64)     # (T, J, 3)
    vel_mag = np.sqrt((vel ** 2).sum(axis=-1))         # (T, J)
    pcts = np.percentile(vel_mag, [10, 25, 50, 75, 90], axis=0)  # (5, J)
    for pi, pname in enumerate(["p10", "p25", "p50", "p75", "p90"]):
        _append_joint_group_scalar_features(feats, names, f"vel_mag_{pname}", pcts[pi], joint_groups)

    # ── Group C: Frequency domain (position channels) ────────────────────────
    pos = motion[:, :, CH_POS].astype(np.float64)     # (T, J, 3)
    dom_ratios = np.zeros(J)
    centroids  = np.zeros(J)

    for j in range(J):
        r_vals, c_vals = [], []
        for ci in range(3):
            freqs_j, psd = _welch_psd(pos[:, j, ci], nperseg)
            total = psd.sum()
            top3  = np.sort(psd)[::-1][:min(3, len(psd))].sum()
            r_vals.append(float(top3 / total))
            c_vals.append(float((freqs_j * psd).sum() / total))
        dom_ratios[j] = float(np.mean(r_vals))
        centroids[j]  = float(np.mean(c_vals))

    _append_joint_group_scalar_features(feats, names, "freq_dom_ratio", dom_ratios, joint_groups)
    _append_joint_group_scalar_features(feats, names, "spectral_centroid", centroids, joint_groups)

    return np.array(feats, dtype=np.float64), names


def _compute_local_features(
    motion: np.ndarray,
    nperseg: int,
) -> Dict[str, np.ndarray]:
    """Extract per-joint local naturalness features.

    Returns a dict of metric_name → np.ndarray of shape (J,), one scalar per
    joint.  The following metrics are computed on position channels (ch 0-2):

    spectral_flatness
        exp(mean(log(PSD))) / mean(PSD).  Near 0 = tonal/sinusoidal;
        near 1 = white-noise-like.

    spectral_centroid
        Frequency centre-of-mass of PSD.  Low = low-frequency dominated
        (over-smooth); high = high-frequency dominated (jittery).

    acf_peak
        Maximum normalized autocorrelation for lags in (T/8, T/2].
        High values indicate strong periodicity (e.g. walking gait).

    jerk_norm
        Activity-normalised jerk (3rd temporal derivative), per joint
        without the global root upweighting used by _compute_normalised_jerk.

    snap_norm
        RMS of 4th temporal derivative / RMS of position.  Catches
        high-frequency micro-jitter that jerk alone may miss.
    """
    T, J, _ = motion.shape
    pos = motion[:, :, CH_POS].astype(np.float64)   # (T, J, 3)

    spectral_flatness = np.zeros(J)
    spectral_centroid = np.zeros(J)
    acf_peak          = np.zeros(J)
    jerk_norm         = np.zeros(J)
    snap_norm         = np.zeros(J)

    # Spectral features (per joint, mean across 3 position channels)
    for j in range(J):
        sf_vals, sc_vals = [], []
        for ci in range(3):
            freqs_j, psd = _welch_psd(pos[:, j, ci], nperseg)
            log_mean   = float(np.mean(np.log(psd)))
            arith_mean = float(np.mean(psd))
            sf_vals.append(math.exp(log_mean) / max(arith_mean, 1e-30))
            sc_vals.append(float((freqs_j * psd).sum() / psd.sum()))
        spectral_flatness[j] = float(np.mean(sf_vals))
        spectral_centroid[j] = float(np.mean(sc_vals))

    # Autocorrelation peak (requires at least MIN_CLIP_FRAMES)
    if T >= _MIN_CLIP_FRAMES:
        min_lag = max(1, T // 8)
        max_lag = T // 2
        for j in range(J):
            peaks = []
            for ci in range(3):
                sig = pos[:, j, ci]
                sig_c = sig - sig.mean()
                if sig_c.std() < 1e-8:
                    peaks.append(0.0)
                    continue
                acf_full = np.correlate(sig_c, sig_c, mode="full")
                acf_half = acf_full[T - 1:]                 # lags 0, 1, 2, …
                acf_norm = acf_half / (acf_half[0] + 1e-30)
                if max_lag > min_lag:
                    peaks.append(float(acf_norm[min_lag : max_lag + 1].max()))
                else:
                    peaks.append(0.0)
            acf_peak[j] = float(np.mean(peaks))

    # Per-joint jerk (no root upweighting)
    if T >= 4:
        jerk3 = np.diff(np.diff(np.diff(pos, axis=0), axis=0), axis=0)  # (T-3, J, 3)
        pjj   = (jerk3 ** 2).mean(axis=(0, 2))            # (J,)
        pjv   = pos.var(axis=0).mean(axis=-1)              # (J,)
        jerk_norm = pjj / (pjv + 1e-10)

    # Per-joint snap (4th derivative)
    if T >= 5:
        snap   = np.diff(np.diff(np.diff(np.diff(pos, axis=0), axis=0), axis=0), axis=0)
        s_rms  = np.sqrt((snap ** 2).mean(axis=(0, 2)))    # (J,)
        p_rms  = np.sqrt((pos  ** 2).mean(axis=(0, 2))) + 1e-10
        snap_norm = s_rms / p_rms

    return {
        "spectral_flatness": spectral_flatness,
        "spectral_centroid": spectral_centroid,
        "acf_peak":          acf_peak,
        "jerk_norm":         jerk_norm,
        "snap_norm":         snap_norm,
    }


def _compute_joint_psd(motion: np.ndarray, nperseg: int) -> np.ndarray:
    """Average normalized PSD per joint (position channels).

    Returns
    -------
    psd : (J, n_freqs)  normalized (sums to 1 per joint)
    """
    T, J, _ = motion.shape
    pos    = motion[:, :, CH_POS].astype(np.float64)
    n_freq = nperseg // 2 + 1
    out    = np.zeros((J, n_freq))

    for j in range(J):
        for ci in range(3):
            _, psd = _welch_psd(pos[:, j, ci], nperseg)
            if len(psd) == n_freq:
                out[j] += psd
            else:
                # Length mismatch only if nperseg was clamped for this clip
                out[j] += np.interp(
                    np.linspace(0, 1, n_freq),
                    np.linspace(0, 1, len(psd)),
                    psd,
                )
        out[j] /= 3.0
        s = out[j].sum()
        if s > 1e-30:
            out[j] /= s

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Scorer
# ─────────────────────────────────────────────────────────────────────────────

class DistributionMotionQualityScorer:
    """Evaluate motion quality by comparing distributions of generated vs clean.

    Parameters
    ----------
    fps : int
        Frame rate of motion data (used for scaling frequency features).
    min_batch_size : int
        Minimum number of clips required in each set (default 32).
    """

    def __init__(self, fps: int = 30, min_batch_size: int = MIN_BATCH_SIZE):
        self.fps = fps
        self.min_batch_size = min_batch_size
        
        # Fixed macro cell weights: boost pos, rot, and axial
        self.macro_cell_weights = {}
        for feature in ["pos", "rot", "freq", "vel"]:
            for joint in ["root", "axial", "limbs"]:
                f_weight = 1.5 if feature in ["pos", "rot"] else 1.0
                j_weight = 1.5 if joint == "axial" else 1.0
                self.macro_cell_weights[(feature, joint)] = f_weight * j_weight
        
        self._cond_lookup = load_cond_dict()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        generated: List[np.ndarray],
        clean: List[np.ndarray],
        object_type: Optional[str] = None,
    ) -> DistributionEvalReport:
        """Compare the distribution of generated motions against clean GT.

        Parameters
        ----------
        generated : list of np.ndarray  shape (T_i, J, 13)
            Variable clip lengths are supported; joint count J must match
            across all clips and across both sets.
        clean     : list of np.ndarray  shape (T_i, J, 13)
        object_type : str, optional
            Skeleton type name stored as metadata.

        Returns
        -------
        DistributionEvalReport

        Raises
        ------
        ValueError
            Fewer than min_batch_size clips, bad array shapes, or joint count
            mismatch between the two sets.
        """
        generated = [m.astype(np.float32) for m in generated
                     if m.ndim == 3 and m.shape[-1] == 13 and m.shape[0] >= _MIN_CLIP_FRAMES]
        clean     = [m.astype(np.float32) for m in clean
                     if m.ndim == 3 and m.shape[-1] == 13 and m.shape[0] >= _MIN_CLIP_FRAMES]

        if len(generated) < self.min_batch_size:
            raise ValueError(
                f"Need ≥{self.min_batch_size} valid generated clips, got {len(generated)}"
            )
        if len(clean) < self.min_batch_size:
            raise ValueError(
                f"Need ≥{self.min_batch_size} valid clean clips, got {len(clean)}"
            )

        generated_joint_counts = {m.shape[1] for m in generated}
        clean_joint_counts = {m.shape[1] for m in clean}
        if len(generated_joint_counts) != 1 or len(clean_joint_counts) != 1:
            raise ValueError("All clips within each set must share the same joint count")
        if generated_joint_counts != clean_joint_counts:
            raise ValueError("Generated and clean clips must have the same joint count")

        n_joints = next(iter(generated_joint_counts))
        joint_groups, joint_group_source = self._resolve_macro_joint_groups(object_type, n_joints)

        # Consistent FFT window across the whole batch
        min_T = min(
            min(m.shape[0] for m in generated),
            min(m.shape[0] for m in clean),
        )
        nperseg = max(4, min(64, min_T))

        macro = self._compute_macro(generated, clean, nperseg, joint_groups)
        local = self._compute_local(generated, clean, nperseg, joint_groups)

        overall = float(np.clip(0.6 * macro["score"] + 0.4 * local["score"], 0.0, 1.0))

        return DistributionEvalReport(
            object_type=object_type,
            n_generated=len(generated),
            n_clean=len(clean),
            macro_fidelity_score=macro["score"],
            macro_feature_group_scores=macro["feature_group_scores"],
            macro_joint_group_scores=macro["joint_group_scores"],
            macro_joint_group_sizes={name: int(len(indices)) for name, indices in joint_groups.items()},
            macro_top_deviating_features=macro["top_features"],
            local_naturalness_score=local["score"],
            local_psd_jsd_root=local["psd_jsd_root"],
            local_psd_jsd_limbs=local["psd_jsd_limbs"],
            local_spectral_flatness_w1=local["sf_w1"],
            local_jerk_w1=local["jerk_w1"],
            local_acf_peak_w1=local["acf_w1"],
            local_spectral_centroid_w1=local["sc_w1"],
            local_snap_w1=local["snap_w1"],
            overall_score=overall,
            raw={
                "nperseg": nperseg,
                "macro_joint_group_source": joint_group_source,
                **macro.get("raw", {}),
                **local.get("raw", {}),
            },
        )

    def _resolve_macro_joint_groups(
        self,
        object_type: Optional[str],
        n_joints: int,
    ) -> Tuple[Dict[str, np.ndarray], str]:
        cond = None
        if object_type is not None:
            key = str(object_type)
            cond = self._cond_lookup.get(key)
            if cond is None:
                cond = next((value for name, value in self._cond_lookup.items() if str(name).lower() == key.lower()), None)
            if cond is not None and len(cond.get("parents", [])) != n_joints:
                raise ValueError(
                    f"Object type {object_type!r} expects {len(cond.get('parents', []))} joints, got {n_joints}"
                )

        if cond is None:
            candidates = [
                value
                for value in self._cond_lookup.values()
                if len(value.get("parents", [])) == n_joints
            ]
            if len(candidates) == 1:
                cond = candidates[0]

        if cond is None:
            return _fallback_macro_joint_groups(n_joints)

        return _build_macro_joint_groups_from_cond(cond, n_joints)

    # ------------------------------------------------------------------
    # Internal: macro fidelity
    # ------------------------------------------------------------------

    def _compute_macro(
        self,
        generated: List[np.ndarray],
        clean: List[np.ndarray],
        nperseg: int,
        joint_groups: Mapping[str, np.ndarray],
    ) -> dict:
        feat_names: Optional[List[str]] = None
        gen_feats, cln_feats = [], []

        for m in generated:
            f, names = _compute_macro_features(m, nperseg, joint_groups)
            gen_feats.append(f)
            if feat_names is None:
                feat_names = names
        for m in clean:
            f, _ = _compute_macro_features(m, nperseg, joint_groups)
            cln_feats.append(f)

        assert feat_names is not None
        F_gen = np.stack(gen_feats, axis=0)   # (N_gen, D)
        F_cln = np.stack(cln_feats, axis=0)   # (N_cln, D)

        # Robust feature scaling: use the combined-set IQR to avoid exploding
        # distances on clean-only near-constant features, with an absolute
        # family floor so tiny zero-centred offsets are not treated as a full
        # "one-scale" error.
        center_ref = np.median(F_cln, axis=0)
        scales = np.zeros(F_cln.shape[1], dtype=np.float64)
        per_feat_raw_w1: Dict[str, float] = {}
        per_feat_scale: Dict[str, float] = {}
        per_feat_nd: Dict[str, float] = {}
        feature_group_values: Dict[str, List[float]] = {}
        joint_group_values: Dict[str, List[float]] = {}
        cell_values: Dict[Tuple[str, str], List[float]] = {}

        for d, name in enumerate(feat_names):
            raw_w1 = float(scipy.stats.wasserstein_distance(F_gen[:, d], F_cln[:, d]))
            scale = max(_combined_iqr(F_gen[:, d], F_cln[:, d]), _macro_feature_abs_floor(name))
            norm_distance = raw_w1 / scale
            feat_score = _macro_distance_to_score(norm_distance)
            feature_group = _macro_feature_group(name)
            joint_bucket = _macro_joint_bucket(name)

            scales[d] = scale
            per_feat_raw_w1[name] = raw_w1
            per_feat_scale[name] = scale
            per_feat_nd[name] = norm_distance
            feature_group_values.setdefault(feature_group, []).append(feat_score)
            joint_group_values.setdefault(joint_bucket, []).append(feat_score)
            cell_values.setdefault((feature_group, joint_bucket), []).append(feat_score)

        macro_feature_group_scores = {
            group: float(np.mean(scores))
            for group, scores in feature_group_values.items()
        }
        macro_joint_group_scores = {
            group: float(np.mean(scores))
            for group, scores in joint_group_values.items()
        }
        macro_cell_scores = {
            f"{feature_group}:{joint_bucket}": float(np.mean(scores))
            for (feature_group, joint_bucket), scores in cell_values.items()
        }
        
        # Compute weighted macro score
        cell_weights = {}
        for (feature_group, joint_bucket) in cell_values.keys():
            weight = 1.0
            # Check exact match first
            if (feature_group, joint_bucket) in self.macro_cell_weights:
                weight = self.macro_cell_weights[(feature_group, joint_bucket)]
            # Check wildcard matches
            elif (feature_group, "*") in self.macro_cell_weights:
                weight = self.macro_cell_weights[(feature_group, "*")]
            elif ("*", joint_bucket) in self.macro_cell_weights:
                weight = self.macro_cell_weights[("*", joint_bucket)]
            cell_weights[f"{feature_group}:{joint_bucket}"] = weight
        
        # Weighted average of cell scores
        weighted_scores = [
            cell_scores * cell_weights[cell_key]
            for cell_key, cell_scores in macro_cell_scores.items()
        ]
        total_weight = sum(cell_weights.values())
        macro_score = float(np.sum(weighted_scores) / total_weight) if total_weight > 0 else 0.0

        top_features = sorted(per_feat_nd.items(), key=lambda kv: -kv[1])[:5]

        return {
            "score":        float(np.clip(macro_score, 0.0, 1.0)),
            "feature_group_scores": macro_feature_group_scores,
            "joint_group_scores": macro_joint_group_scores,
            "top_features": top_features,
            "raw":          {
                "macro_feature_group_scores": macro_feature_group_scores,
                "macro_joint_group_scores": macro_joint_group_scores,
                "macro_cell_scores": macro_cell_scores,
                "macro_cell_weights": cell_weights,
                "macro_per_feature_w1": per_feat_raw_w1,
                "macro_per_feature_scale": per_feat_scale,
                "macro_per_feature_normalized_distance": per_feat_nd,
            },
        }

    # ------------------------------------------------------------------
    # Internal: local naturalness
    # ------------------------------------------------------------------

    def _compute_local(
        self,
        generated: List[np.ndarray],
        clean: List[np.ndarray],
        nperseg: int,
        joint_groups: Mapping[str, np.ndarray],
    ) -> dict:
        gen_local = [_compute_local_features(m, nperseg) for m in generated]
        cln_local = [_compute_local_features(m, nperseg) for m in clean]
        gen_psds  = [_compute_joint_psd(m, nperseg) for m in generated]
        cln_psds  = [_compute_joint_psd(m, nperseg) for m in clean]

        def _concat_group_scalars(
            feats_list: List[Dict[str, np.ndarray]],
            key: str,
            group_indices: np.ndarray,
        ) -> np.ndarray:
            pooled = [
                np.asarray(sample[key][group_indices], dtype=np.float64).reshape(-1)
                for sample in feats_list
            ]
            return np.concatenate(pooled, axis=0)

        def _stack_group_psds(psds_list: List[np.ndarray], group_indices: np.ndarray) -> np.ndarray:
            return np.stack([
                np.asarray(psd[group_indices], dtype=np.float64).mean(axis=0)
                for psd in psds_list
            ], axis=0)

        active_groups = _active_joint_groups(joint_groups)
        metric_group_scores: Dict[str, Dict[str, float]] = {}
        metric_group_raw_w1: Dict[str, Dict[str, float]] = {}
        metric_group_scale: Dict[str, Dict[str, float]] = {}
        metric_group_nd: Dict[str, Dict[str, float]] = {}

        psd_group_scores: Dict[str, float] = {}
        psd_group_raw_w1: Dict[str, float] = {}
        psd_group_scale: Dict[str, float] = {}
        psd_group_nd: Dict[str, float] = {}
        psd_jsd_by_group: Dict[str, float] = {}

        for group_name, group_indices in active_groups:
            gen_group_psd = _stack_group_psds(gen_psds, group_indices)
            cln_group_psd = _stack_group_psds(cln_psds, group_indices)
            psd_jsd_by_group[group_name] = _psd_jsd(
                gen_group_psd.mean(axis=0),
                cln_group_psd.mean(axis=0),
            )

            bin_details = [
                _score_distribution_distance(
                    gen_group_psd[:, freq_index],
                    cln_group_psd[:, freq_index],
                    _local_metric_abs_floor("psd_bin"),
                )
                for freq_index in range(gen_group_psd.shape[1])
            ]
            psd_group_scores[group_name] = float(np.mean([detail["score"] for detail in bin_details]))
            psd_group_raw_w1[group_name] = float(np.mean([detail["raw_w1"] for detail in bin_details]))
            psd_group_scale[group_name] = float(np.mean([detail["scale"] for detail in bin_details]))
            psd_group_nd[group_name] = float(np.mean([detail["normalized_distance"] for detail in bin_details]))

        metric_group_scores["psd_jsd"] = psd_group_scores
        metric_group_raw_w1["psd_jsd"] = psd_group_raw_w1
        metric_group_scale["psd_jsd"] = psd_group_scale
        metric_group_nd["psd_jsd"] = psd_group_nd

        for key in ["spectral_flatness", "spectral_centroid", "acf_peak", "jerk_norm", "snap_norm"]:
            group_scores: Dict[str, float] = {}
            group_raw_w1: Dict[str, float] = {}
            group_scale: Dict[str, float] = {}
            group_nd: Dict[str, float] = {}

            for group_name, group_indices in active_groups:
                gen_values = _concat_group_scalars(gen_local, key, group_indices)
                cln_values = _concat_group_scalars(cln_local, key, group_indices)
                detail = _score_distribution_distance(
                    gen_values,
                    cln_values,
                    _local_metric_abs_floor(key),
                )
                group_scores[group_name] = detail["score"]
                group_raw_w1[group_name] = detail["raw_w1"]
                group_scale[group_name] = detail["scale"]
                group_nd[group_name] = detail["normalized_distance"]

            metric_group_scores[key] = group_scores
            metric_group_raw_w1[key] = group_raw_w1
            metric_group_scale[key] = group_scale
            metric_group_nd[key] = group_nd

        weights = {
            "spectral_flatness": 0.25,
            "psd_jsd":           0.25,
            "jerk_norm":         0.20,
            "acf_peak":          0.15,
            "spectral_centroid": 0.10,
            "snap_norm":         0.05,
        }
        component_scores = {
            key: float(np.mean(list(metric_group_scores[key].values())))
            for key in weights
        }
        local_group_scores = {
            group_name: float(np.mean([
                metric_group_scores[key][group_name]
                for key in weights
                if group_name in metric_group_scores[key]
            ]))
            for group_name, _ in active_groups
        }
        local_score = sum(weights[k] * component_scores[k] for k in weights)

        return {
            "score":         float(np.clip(local_score, 0.0, 1.0)),
            "psd_jsd_root":  float(psd_jsd_by_group.get("root", 0.0)),
            "psd_jsd_limbs": float(psd_jsd_by_group.get("limbs", 0.0)),
            "sf_w1":         float(np.mean(list(metric_group_raw_w1["spectral_flatness"].values()))),
            "jerk_w1":       float(np.mean(list(metric_group_raw_w1["jerk_norm"].values()))),
            "acf_w1":        float(np.mean(list(metric_group_raw_w1["acf_peak"].values()))),
            "sc_w1":         float(np.mean(list(metric_group_raw_w1["spectral_centroid"].values()))),
            "snap_w1":       float(np.mean(list(metric_group_raw_w1["snap_norm"].values()))),
            "raw": {
                "local_component_scores": component_scores,
                "local_joint_group_scores": local_group_scores,
                "local_metric_group_scores": metric_group_scores,
                "local_metric_group_w1": metric_group_raw_w1,
                "local_metric_group_scale": metric_group_scale,
                "local_metric_group_normalized_distance": metric_group_nd,
                "local_psd_jsd_by_group": psd_jsd_by_group,
                "local_active_joint_groups": [name for name, _ in active_groups],
            },
        }
