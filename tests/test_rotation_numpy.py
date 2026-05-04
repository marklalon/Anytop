import os
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation as SciPyRotation


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ANYTOP_ROOT = os.path.dirname(_SCRIPT_DIR)
_REPO_ROOT = os.path.dirname(_ANYTOP_ROOT)

for _path in [_REPO_ROOT, _ANYTOP_ROOT]:
    if _path not in sys.path:
        sys.path.insert(0, _path)


from Anytop.utils.rotation_numpy import (
    apply_rotation_to_quaternions_wxyz_np,
    matrix_to_quat_wxyz_np,
    quat_conjugate_wxyz_np,
    quat_multiply_wxyz_np,
    quat_rotate_wxyz_np,
    quat_to_matrix_wxyz_np,
)


def _xyzw_to_wxyz(quaternions_xyzw: np.ndarray) -> np.ndarray:
    return np.concatenate([quaternions_xyzw[:, 3:4], quaternions_xyzw[:, :3]], axis=-1)


def _wxyz_to_xyzw(quaternions_wxyz: np.ndarray) -> np.ndarray:
    return np.concatenate([quaternions_wxyz[:, 1:], quaternions_wxyz[:, :1]], axis=-1)


def _quat_angle_deg_wxyz(q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
    dots = np.abs(np.sum(q_a * q_b, axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(dots, 0.0, 1.0)))


def _random_quaternions_wxyz(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    quaternions_xyzw = SciPyRotation.random(count, random_state=rng).as_quat().astype(np.float64)
    return _xyzw_to_wxyz(quaternions_xyzw)


class RotationNumpyTest(unittest.TestCase):
    def test_quat_to_matrix_matches_scipy(self):
        quaternions_wxyz = _random_quaternions_wxyz(4096, seed=1234)
        reference_mats = SciPyRotation.from_quat(_wxyz_to_xyzw(quaternions_wxyz)).as_matrix()
        converted_mats = quat_to_matrix_wxyz_np(quaternions_wxyz)
        np.testing.assert_allclose(converted_mats, reference_mats, atol=1e-12)

    def test_matrix_to_quat_matches_scipy_up_to_sign(self):
        quaternions_wxyz = _random_quaternions_wxyz(4096, seed=2345)
        matrices = SciPyRotation.from_quat(_wxyz_to_xyzw(quaternions_wxyz)).as_matrix()
        recovered_wxyz = matrix_to_quat_wxyz_np(matrices)
        max_angle_deg = float(_quat_angle_deg_wxyz(recovered_wxyz, quaternions_wxyz).max())
        self.assertLess(max_angle_deg, 1e-5)

    def test_apply_rotation_to_quaternions_matches_scipy(self):
        quaternions_wxyz = _random_quaternions_wxyz(4096, seed=3456)
        left_rotation_xyzw = SciPyRotation.random(1, random_state=np.random.default_rng(4567)).as_quat()[0]
        left_rotation_matrix = SciPyRotation.from_quat(left_rotation_xyzw).as_matrix()

        rotated_wxyz = apply_rotation_to_quaternions_wxyz_np(
            quaternions_wxyz.reshape(64, 64, 4),
            left_rotation_matrix,
        ).reshape(-1, 4)

        reference_rot = SciPyRotation.from_matrix(left_rotation_matrix) * SciPyRotation.from_quat(
            _wxyz_to_xyzw(quaternions_wxyz)
        )
        reference_wxyz = _xyzw_to_wxyz(reference_rot.as_quat())
        max_angle_deg = float(_quat_angle_deg_wxyz(rotated_wxyz, reference_wxyz).max())
        self.assertLess(max_angle_deg, 1e-5)

    def test_quat_multiply_matches_scipy_composition(self):
        q1_wxyz = _random_quaternions_wxyz(4096, seed=5678)
        q2_wxyz = _random_quaternions_wxyz(4096, seed=6789)
        product_wxyz = quat_multiply_wxyz_np(q1_wxyz, q2_wxyz)

        reference_rot = SciPyRotation.from_quat(_wxyz_to_xyzw(q1_wxyz)) * SciPyRotation.from_quat(
            _wxyz_to_xyzw(q2_wxyz)
        )
        reference_wxyz = _xyzw_to_wxyz(reference_rot.as_quat())
        max_angle_deg = float(_quat_angle_deg_wxyz(product_wxyz, reference_wxyz).max())
        self.assertLess(max_angle_deg, 1e-5)

    def test_quat_rotate_and_conjugate_match_scipy(self):
        quaternions_wxyz = _random_quaternions_wxyz(2048, seed=7890)
        vectors = np.random.default_rng(8901).normal(size=(2048, 3)).astype(np.float64)

        rotated_vectors = quat_rotate_wxyz_np(quaternions_wxyz, vectors)
        reference_vectors = SciPyRotation.from_quat(_wxyz_to_xyzw(quaternions_wxyz)).apply(vectors)
        np.testing.assert_allclose(rotated_vectors, reference_vectors, atol=1e-12)

        conjugated_wxyz = quat_conjugate_wxyz_np(quaternions_wxyz)
        reference_inverse_wxyz = _xyzw_to_wxyz(
            SciPyRotation.from_quat(_wxyz_to_xyzw(quaternions_wxyz)).inv().as_quat()
        )
        max_angle_deg = float(_quat_angle_deg_wxyz(conjugated_wxyz, reference_inverse_wxyz).max())
        self.assertLess(max_angle_deg, 1e-5)


if __name__ == "__main__":
    unittest.main()