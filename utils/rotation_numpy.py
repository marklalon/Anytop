from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def quat_conjugate_wxyz_np(quaternions: np.ndarray) -> np.ndarray:
    quat_array = np.asarray(quaternions)
    if quat_array.shape[-1] != 4:
        raise ValueError(f"Expected quaternions with shape (..., 4), got {quat_array.shape}")
    out = quat_array.copy()
    out[..., 1:] *= -1
    return out


def quat_multiply_wxyz_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q1_array = np.asarray(q1)
    q2_array = np.asarray(q2)
    if q1_array.shape[-1] != 4 or q2_array.shape[-1] != 4:
        raise ValueError(
            f"Expected quaternions with shape (..., 4), got {q1_array.shape} and {q2_array.shape}"
        )
    w1, x1, y1, z1 = q1_array[..., 0], q1_array[..., 1], q1_array[..., 2], q1_array[..., 3]
    w2, x2, y2, z2 = q2_array[..., 0], q2_array[..., 1], q2_array[..., 2], q2_array[..., 3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.stack([w, x, y, z], axis=-1)


def quat_rotate_wxyz_np(quaternions: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    quat_array = np.asarray(quaternions)
    vec_array = np.asarray(vectors)
    if quat_array.shape[-1] != 4:
        raise ValueError(f"Expected quaternions with shape (..., 4), got {quat_array.shape}")
    if vec_array.shape[-1] != 3:
        raise ValueError(f"Expected vectors with shape (..., 3), got {vec_array.shape}")
    quat_conj = quat_conjugate_wxyz_np(quat_array)
    qv = np.concatenate([np.zeros_like(vec_array[..., :1]), vec_array], axis=-1)
    return quat_multiply_wxyz_np(quat_multiply_wxyz_np(quat_array, qv), quat_conj)[..., 1:]


def _quat_wxyz_to_xyzw(flat_quaternions: np.ndarray) -> np.ndarray:
    return np.concatenate([flat_quaternions[:, 1:], flat_quaternions[:, :1]], axis=-1)


def _quat_xyzw_to_wxyz(flat_quaternions: np.ndarray) -> np.ndarray:
    return np.concatenate([flat_quaternions[:, 3:4], flat_quaternions[:, :3]], axis=-1)


def _float_dtype_or_default(array: np.ndarray) -> np.dtype:
    return array.dtype if np.issubdtype(array.dtype, np.floating) else np.float64


def quat_to_matrix_wxyz_np(quaternions: np.ndarray) -> np.ndarray:
    quat_array = np.asarray(quaternions)
    if quat_array.shape[-1] != 4:
        raise ValueError(f"Expected quaternions with shape (..., 4), got {quat_array.shape}")
    orig_dtype = _float_dtype_or_default(quat_array)
    flat_quats = quat_array.astype(np.float64, copy=False).reshape(-1, 4)
    flat_mats = Rotation.from_quat(_quat_wxyz_to_xyzw(flat_quats)).as_matrix()
    return flat_mats.reshape(quat_array.shape[:-1] + (3, 3)).astype(orig_dtype, copy=False)


def matrix_to_quat_wxyz_np(matrices: np.ndarray) -> np.ndarray:
    matrix_array = np.asarray(matrices)
    if matrix_array.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrices with shape (..., 3, 3), got {matrix_array.shape}")
    orig_dtype = _float_dtype_or_default(matrix_array)
    flat_mats = matrix_array.astype(np.float64, copy=False).reshape(-1, 3, 3)
    flat_quats = Rotation.from_matrix(flat_mats).as_quat()
    flat_wxyz = _quat_xyzw_to_wxyz(flat_quats)
    return flat_wxyz.reshape(matrix_array.shape[:-2] + (4,)).astype(orig_dtype, copy=False)


def quaternion_angle_degrees_wxyz_np(
    q_a: np.ndarray,
    q_b: np.ndarray,
) -> np.ndarray:
    """Per-element quaternion angular difference in degrees.

    Returns angle in [0, 180] for each (..., 4) pair.
    Normalizes both quaternions before computing dot product to avoid
    errors from non-unit quaternions (e.g., due to floating-point drift).
    """
    norm_a = np.maximum(np.linalg.norm(q_a, axis=-1, keepdims=True), 1e-12)
    norm_b = np.maximum(np.linalg.norm(q_b, axis=-1, keepdims=True), 1e-12)
    q_a_n = q_a / norm_a
    q_b_n = q_b / norm_b

    dots = np.sum(q_a_n * q_b_n, axis=-1)
    dots = np.clip(np.abs(dots), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))


def apply_rotation_to_quaternions_wxyz_np(quaternions: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    quat_array = np.asarray(quaternions)
    if quat_array.shape[-1] != 4:
        raise ValueError(f"Expected quaternions with shape (..., 4), got {quat_array.shape}")
    rotation_array = np.asarray(rotation_matrix)
    if rotation_array.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 rotation matrix, got {rotation_array.shape}")
    orig_dtype = _float_dtype_or_default(quat_array)
    flat_quats = quat_array.astype(np.float64, copy=False).reshape(-1, 4)
    base_rot = Rotation.from_quat(_quat_wxyz_to_xyzw(flat_quats))
    applied_rot = Rotation.from_matrix(rotation_array.astype(np.float64, copy=False)) * base_rot
    flat_wxyz = _quat_xyzw_to_wxyz(applied_rot.as_quat())
    return flat_wxyz.reshape(quat_array.shape).astype(orig_dtype, copy=False)