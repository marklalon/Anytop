import os
import sys
import unittest
from unittest.mock import patch

import numpy as np


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ANYTOP_ROOT = os.path.dirname(_SCRIPT_DIR)
_REPO_ROOT = os.path.dirname(_ANYTOP_ROOT)

for _path in [_REPO_ROOT, _ANYTOP_ROOT]:
    if _path not in sys.path:
        sys.path.insert(0, _path)


from data_loaders.truebones.truebones_utils import face_orientation
from data_loaders.truebones.truebones_utils.face_orientation import (
    _choose_facing_forward,
    _get_facing_candidates,
    resolve_forward_reference_joints,
)


class FaceOrientationChainForwardTest(unittest.TestCase):
    def setUp(self):
        face_orientation._EMITTED_DEGENERATE_FACING_WARNINGS.clear()

    def test_forward_reference_falls_back_to_tail_spine_generically(self):
        joint_names = [
            'Hips',
            'Bip01_Spine',
            'Bip01_Pelvis',
            'Bip01_Tail',
            'Bip01_Tail1',
            'Bip01_R_Thigh_1',
            'Bip01_L_Thigh_4',
            'Bip01_R_Clavicle',
            'Bip01_L_Clavicle',
        ]
        parents = np.array([-1, 0, 0, 0, 3, 0, 0, 1, 1], dtype=np.int64)

        with patch('builtins.print') as mock_print:
            forward_joint_index, forward_base_joint_index = resolve_forward_reference_joints(
                joint_names,
                parents,
                object_type='FallbackBug',
            )

        self.assertEqual(forward_joint_index, 1)
        self.assertEqual(forward_base_joint_index, 3)
        mock_print.assert_called_once_with(
            '[WARN] FallbackBug: no head/neck forward reference was found; falling back to tail->spine body-axis orientation.'
        )

    def test_tail_spine_reference_axis_produces_forward_candidate(self):
        joints = np.zeros((1, 63, 3), dtype=np.float64)
        joints[:, 3] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        joints[:, 1] = np.array([1.0, 0.0, 1.0], dtype=np.float64)

        candidates = _get_facing_candidates(
            joints,
            'FallbackBug',
            face_joint_indx=[],
            forward_joint_index=1,
            forward_base_joint_index=3,
        )

        self.assertEqual(set(candidates.keys()), {'tail_spine'})

        expected = np.array([[1.0, 0.0, 1.0]], dtype=np.float64)
        expected /= np.linalg.norm(expected, axis=-1, keepdims=True)
        np.testing.assert_allclose(candidates['tail_spine'], expected, atol=1e-8)

    def test_priority_prefers_torso_head_over_tail_spine_and_across(self):
        torso_head = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
        tail_spine = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
        across = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)

        candidate_name, candidate_forward = _choose_facing_forward(
            {
                'across': across,
                'tail_spine': tail_spine,
                'torso_head': torso_head,
            },
            object_type='PriorityBug',
        )

        self.assertEqual(candidate_name, 'torso_head')
        np.testing.assert_allclose(candidate_forward, torso_head, atol=1e-8)

    def test_near_vertical_primary_falls_back_to_across(self):
        torso_head = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
        across = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)

        candidate_name, candidate_forward = _choose_facing_forward(
            {
                'torso_head': torso_head,
                'across': across,
            },
            object_type='VerticalBug',
            near_y_candidates={'torso_head': True},
        )

        self.assertEqual(candidate_name, 'across')
        np.testing.assert_allclose(candidate_forward, across, atol=1e-8)

    def test_forward_reference_ignores_head_nub(self):
        joint_names = [
            'Hips',
            'Bip01_Neck2',
            'Bip01_Head',
            'Bip01_HeadNub',
        ]
        parents = np.array([-1, 0, 1, 2], dtype=np.int64)

        forward_joint_index, forward_base_joint_index = resolve_forward_reference_joints(
            joint_names,
            parents,
            object_type='BuzzardLike',
        )

        self.assertEqual(forward_joint_index, 2)
        self.assertIsNone(forward_base_joint_index)

    def test_across_forward_is_projected_to_xz(self):
        joints = np.zeros((1, 4, 3), dtype=np.float64)
        joints[:, 0] = np.array([1.0, 2.0, 0.0], dtype=np.float64)
        joints[:, 1] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        joints[:, 2] = np.array([1.0, 3.0, 0.0], dtype=np.float64)
        joints[:, 3] = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        candidates = _get_facing_candidates(
            joints,
            'AcrossBug',
            face_joint_indx=[0, 1, 2, 3],
            forward_joint_index=None,
            forward_base_joint_index=None,
        )

        np.testing.assert_allclose(candidates['across'][0, 1], 0.0, atol=1e-8)
        np.testing.assert_allclose(candidates['across'][0], np.array([0.0, 0.0, -1.0], dtype=np.float64), atol=1e-8)

    def test_across_selection_emits_warning_once(self):
        across = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)

        with patch('builtins.print') as mock_print:
            first_name, _first_forward = _choose_facing_forward(
                {
                    'across': across,
                },
                object_type='FallbackBug',
            )
            second_name, _second_forward = _choose_facing_forward(
                {
                    'across': across,
                },
                object_type='FallbackBug',
            )

        self.assertEqual(first_name, 'across')
        self.assertEqual(second_name, 'across')
        mock_print.assert_called_once_with(
            '[WARN] FallbackBug: orientation calculation fell back to the across-vector heuristic because higher-priority forward references were unavailable or near-parallel to the Y axis.'
        )


if __name__ == '__main__':
    unittest.main()