"""Quaternion helpers shared by the profile extractor and the edit runtime.

Quaternions are ``(..., 4)`` arrays in ``wxyz`` order, the layout
``motion_lib.Quaternions`` stores.  Rotation vectors are full-angle
(``|v|`` is the rotation angle in radians); ``Quaternions.log`` returns the
half-angle vector, so every conversion goes through the two helpers here.
"""

from __future__ import annotations

import numpy as np

from motion_lib.Quaternions import Quaternions


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.broadcast_arrays(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    return (Quaternions(a) * Quaternions(b)).qs


def quat_inv(q: np.ndarray) -> np.ndarray:
    return np.asarray(q, dtype=np.float64) * np.array([1.0, -1.0, -1.0, -1.0])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors ``v`` (..., 3) by unit quaternions ``q`` (..., 4), broadcasting."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = q[..., :1]
    u = q[..., 1:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def rotvec_from_quat(q: np.ndarray) -> np.ndarray:
    """Shortest-arc rotation vector of each quaternion."""
    return 2.0 * Quaternions(np.asarray(q, dtype=np.float64)).log()


def nearest_rotvec_branch(v: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The rotation vector equivalent to ``v`` (same rotation, angle shifted by
    a multiple of 2 pi along its axis) closest to ``target``."""
    v = np.asarray(v, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    angle = np.linalg.norm(v, axis=-1, keepdims=True)
    target_norm = np.linalg.norm(target, axis=-1, keepdims=True)
    # at the identity the axis is free: take the target's
    axis = np.where(angle > 1e-9, v / np.maximum(angle, 1e-12),
                    target / np.maximum(target_norm, 1e-12))
    along = np.sum(target * axis, axis=-1, keepdims=True)
    turns = np.round((along - angle) / (2.0 * np.pi))
    return axis * (angle + 2.0 * np.pi * turns)


def unwrap_rotvec(v: np.ndarray) -> np.ndarray:
    """Rotation vectors (F, ...) continued along axis 0 so consecutive frames
    never jump between branches; frame 0 keeps its shortest-arc value."""
    out = np.array(v, dtype=np.float64)
    for f in range(1, out.shape[0]):
        out[f] = nearest_rotvec_branch(out[f], out[f - 1])
    return out


def quat_from_rotvec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    angle = np.linalg.norm(v, axis=-1, keepdims=True)
    half = 0.5 * angle
    scale = np.where(angle > 1e-12, np.sin(half) / np.maximum(angle, 1e-12), 0.5)
    return np.concatenate([np.cos(half), v * scale], axis=-1)


def quat_angle(q: np.ndarray) -> np.ndarray:
    """Rotation angle of each quaternion in ``[0, pi]``."""
    w = np.abs(np.clip(np.asarray(q, dtype=np.float64)[..., 0], -1.0, 1.0))
    return 2.0 * np.arccos(w)


def weighted_mean_quat(q: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted chordal mean of ``q`` (N, 4): the top eigenvector of sum w q q^T."""
    q = np.asarray(q, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    m = np.einsum("n,ni,nj->ij", w, q, q)
    _, vecs = np.linalg.eigh(m)
    mean = vecs[:, -1]
    return mean if mean[0] >= 0 else -mean


def robust_mean_quat(q: np.ndarray, weights: np.ndarray, iterations: int = 3) -> np.ndarray:
    """Weighted mean that down-weights poses far from the bulk (Huber on the angle).

    The Huber knee is twice the weighted median deviation, so a clip that holds
    an extreme pose for a few frames does not drag the reference pose.
    """
    w = np.asarray(weights, dtype=np.float64)
    mean = weighted_mean_quat(q, w)
    for _ in range(iterations):
        angle = quat_angle(quat_mul(q, quat_inv(mean)[None]))
        knee = max(2.0 * weighted_quantile(angle, w, 0.5), np.radians(2.0))
        robust = np.where(angle <= knee, 1.0, knee / np.maximum(angle, 1e-12))
        mean = weighted_mean_quat(q, w * robust)
    return mean


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q) -> np.ndarray:
    """Quantile(s) ``q`` of ``values`` under ``weights`` (step CDF, midpoint rule)."""
    values = np.asarray(values, dtype=np.float64).ravel()
    weights = np.asarray(weights, dtype=np.float64).ravel()
    keep = weights > 0
    values, weights = values[keep], weights[keep]
    if values.size == 0:
        return np.full(np.shape(q), np.nan)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cdf = (np.cumsum(weights) - 0.5 * weights) / weights.sum()
    return np.interp(q, cdf, values)


def quat_slerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Shortest-arc slerp of ``a`` to ``b`` at ``t`` (broadcast over leading axes).

    ``t == 0`` returns ``a`` bit-exactly, so sampling a clip at integer frame
    times reproduces its keys.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)[..., None]
    dot = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(dot < 0.0, -b, b)
    dot = np.abs(dot)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    small = sin_theta < 1e-6
    safe = np.where(small, 1.0, sin_theta)
    w0 = np.where(small, 1.0 - t, np.sin((1.0 - t) * theta) / safe)
    w1 = np.where(small, t, np.sin(t * theta) / safe)
    out = w0 * a + w1 * b
    return np.where(t == 0.0, a, out)


def rotation_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Shortest-arc quaternion turning direction ``u`` onto ``v`` (identity when either is
    zero; a half turn about an axis perpendicular to ``u`` when they are opposite)."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    nu = np.linalg.norm(u, axis=-1, keepdims=True)
    nv = np.linalg.norm(v, axis=-1, keepdims=True)
    ok = (nu > 1e-9) & (nv > 1e-9)
    u = u / np.where(ok, nu, 1.0)
    v = v / np.where(ok, nv, 1.0)
    axis = np.cross(u, v)
    sin = np.linalg.norm(axis, axis=-1, keepdims=True)
    cos = np.sum(u * v, axis=-1, keepdims=True)
    angle = np.arctan2(sin, cos)
    turns = sin > 1e-12
    axis = axis / np.where(turns, sin, 1.0)
    # opposite directions: any perpendicular axis; cross with the basis vector u is least aligned with
    basis = np.eye(3)[np.argmin(np.abs(u), axis=-1)]
    half = np.cross(u, basis)
    half = half / np.maximum(np.linalg.norm(half, axis=-1, keepdims=True), 1e-12)
    opposite = ~turns & (cos < 0.0)
    rotvec = np.where(ok & turns, axis * angle, np.where(ok & opposite, half * np.pi, 0.0))
    return quat_from_rotvec(rotvec)
