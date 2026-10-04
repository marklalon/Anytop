"""Per-joint rotation statistics: DOF class, principal axes, flex sign, speed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from motion_edit.rotations import (
    quat_angle,
    quat_inv,
    quat_mul,
    robust_mean_quat,
    rotvec_from_quat,
    weighted_quantile,
)

FIXED_STD_DEG = 1.0        # total angular std below which a joint is "fixed"
HINGE_RATIO = 0.85         # first principal component's share of the variance
PLANAR_RATIO = 0.90        # first two components' share
# Below this spread of the fold distance (relative to its mean) the limb never
# folds in the data and the flex direction is left undetermined.
FOLD_SPREAD_MIN = 1e-3


@dataclass
class JointStats:
    mean: np.ndarray                 # (4,) robust mean local rotation
    axes: np.ndarray                 # (3, 3) rows = principal axes, parent frame
    variance_ratio: np.ndarray       # (3,) descending
    angle_std_deg: float
    dof_class: str
    flex_sign: int                   # hinge only, else 0
    ang_speed: tuple[float, float]   # q50, q95 in rad/s
    main_child: Optional[int]


def _canonical_axes(vectors: np.ndarray) -> np.ndarray:
    """Eigenvector rows with a deterministic sign (largest-|.| component positive)."""
    out = vectors.copy()
    for i in range(out.shape[0]):
        k = int(np.argmax(np.abs(out[i])))
        if out[i, k] < 0:
            out[i] = -out[i]
    return out


def classify_dof(variance_ratio: np.ndarray, angle_std_deg: float) -> str:
    if angle_std_deg < FIXED_STD_DEG:
        return "fixed"
    if variance_ratio[0] >= HINGE_RATIO:
        return "hinge"
    if variance_ratio[0] + variance_ratio[1] >= PLANAR_RATIO:
        return "planar"
    return "ball"


def main_child(children: list[int], bone_length: np.ndarray) -> Optional[int]:
    if not children:
        return None
    best = max(children, key=lambda c: bone_length[c])
    return best if bone_length[best] > 1e-6 else None


def data_flex_sign(angle: np.ndarray, fold_distance: np.ndarray, weights: np.ndarray) -> int:
    """+1 when turning the hinge towards +axis folds the limb, read from the data.

    ``angle`` is the joint's projection on its hinge axis, ``fold_distance`` the
    distance from its anchor (nearest ancestor at a distinct position) to its
    child.  Flexion is the direction in which that distance shrinks: the sign of
    their weighted covariance, over every frame rather than around one pose (a
    nearly straight mean pose sits at the distance maximum, where a local probe
    cannot tell the two directions apart).
    """
    w = np.asarray(weights, dtype=np.float64)
    a_mean = np.average(angle, weights=w)
    d_mean = np.average(fold_distance, weights=w)
    spread = np.sqrt(np.average((fold_distance - d_mean) ** 2, weights=w))
    if d_mean <= 0 or spread < FOLD_SPREAD_MIN * d_mean:
        return 0
    cov = np.sum(w * (angle - a_mean) * (fold_distance - d_mean))
    return -1 if cov > 0 else 1


def joint_stats(
    rotations: list[np.ndarray],
    weights: list[np.ndarray],
    fps: list[float],
    child_index: Optional[int],
    *,
    fold_distances: Optional[list[np.ndarray]] = None,
) -> JointStats:
    """Statistics of one joint over clips.

    ``rotations`` / ``weights`` hold one ``(F, 4)`` / ``(F,)`` array per clip,
    ``fold_distances`` the matching anchor-to-child distances that decide the
    flex direction of a hinge (see :func:`data_flex_sign`).
    """
    q = np.concatenate(rotations, axis=0)
    w = np.concatenate(weights, axis=0)
    mean = robust_mean_quat(q, w)
    deviation = rotvec_from_quat(quat_mul(q, quat_inv(mean)[None]))   # parent frame

    total = w.sum()
    centre = (w[:, None] * deviation).sum(axis=0) / total
    centred = deviation - centre
    cov = np.einsum("n,ni,nj->ij", w, centred, centred) / total
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 0.0)
    axes = _canonical_axes(evecs[:, order].T)
    trace = float(evals.sum())
    ratio = evals / trace if trace > 0 else np.array([1.0, 0.0, 0.0])
    angle_std_deg = float(np.degrees(np.sqrt(trace)))
    dof = classify_dof(ratio, angle_std_deg)

    flex = 0
    if dof == "hinge" and fold_distances is not None:
        flex = data_flex_sign(deviation @ axes[0], np.concatenate(fold_distances), w)

    speeds, speed_w = [], []
    for rot, wt, rate in zip(rotations, weights, fps):
        if rot.shape[0] < 2:
            continue
        step = quat_angle(quat_mul(rot[1:], quat_inv(rot[:-1])))
        speeds.append(step * rate)
        speed_w.append(wt[1:])
    if speeds:
        s50, s95 = weighted_quantile(np.concatenate(speeds), np.concatenate(speed_w), (0.5, 0.95))
    else:
        s50 = s95 = 0.0

    return JointStats(
        mean=mean, axes=axes, variance_ratio=ratio, angle_std_deg=angle_std_deg,
        dof_class=dof, flex_sign=int(flex),
        ang_speed=(float(s50), float(s95)), main_child=child_index,
    )


def axis_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two axes, ignoring sign."""
    cos = abs(float(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return float(np.degrees(np.arccos(min(1.0, cos))))
