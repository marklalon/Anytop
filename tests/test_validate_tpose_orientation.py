import os
import sys
import unittest

import numpy as np


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ANYTOP_ROOT = os.path.dirname(_SCRIPT_DIR)
_REPO_ROOT = os.path.dirname(_ANYTOP_ROOT)

for _path in [_REPO_ROOT, _ANYTOP_ROOT, os.path.join(_ANYTOP_ROOT, 'utils')]:
    if _path not in sys.path:
        sys.path.insert(0, _path)


from motion_lib.Quaternions import Quaternions
from validate_anytop_dataset import summarize_tpose_orientation_axis_alignment


class ValidateTPoseOrientationTest(unittest.TestCase):
    def test_axis_aligned_orientation_maps_to_cardinal_axis(self):
        orientation_quat = Quaternions.between(
            np.array([[-1.0, 0.0, 0.0]], dtype=np.float64),
            np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
        )[0]

        best_axis, delta_deg = summarize_tpose_orientation_axis_alignment(
            'BuzzardLike',
            {'orientation_quat': orientation_quat.qs},
        )

        self.assertEqual(best_axis, '-x')
        self.assertLess(delta_deg, 1e-3)

    def test_diagonal_orientation_reports_nearest_axis_delta(self):
        diagonal_forward = np.array([[1.0, 0.0, 1.0]], dtype=np.float64)
        diagonal_forward /= np.linalg.norm(diagonal_forward, axis=-1, keepdims=True)
        orientation_quat = Quaternions.between(
            diagonal_forward,
            np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
        )[0]

        best_axis, delta_deg = summarize_tpose_orientation_axis_alignment(
            'DiagonalLike',
            {'orientation_quat': orientation_quat.qs},
        )

        self.assertIn(best_axis, {'+x', '+z'})
        self.assertAlmostEqual(delta_deg, 45.0, places=5)


if __name__ == '__main__':
    unittest.main()
