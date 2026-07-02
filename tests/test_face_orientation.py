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
    resolve_face_joints,
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

    def test_degenerate_neck_reference_skipped_in_favor_of_tail_spine(self):
        # A zero-length 'Neck' coincident with the hips (e.g. Scorpion's
        # placeholder Bip01_Neck1) carries no directional info. With rest
        # positions supplied it must be skipped so the forward reference falls
        # through to the reliable tail->spine body axis instead of silently
        # reversing the facing.
        joint_names = [
            'Hips',
            'Bip01_Neck1',
            'Bip01_Spine',
            'Bip01_Pelvis',
            'Bip01_Tail',
            'Bip01_Tail1',
        ]
        parents = np.array([-1, 0, 0, 2, 3, 4], dtype=np.int64)
        rest_positions = np.array([[
            [0.0, 0.0, 0.0],   # Hips
            [0.0, 0.0, 0.0],   # Neck1 coincident with Hips -> degenerate
            [0.0, 0.0, -0.1],  # Spine (toward front)
            [0.0, 0.0, 0.1],   # Pelvis
            [0.0, 0.0, 0.3],   # Tail
            [0.0, 0.0, 0.5],   # Tail1
        ]], dtype=np.float64)

        # Without positions the degenerate neck is (wrongly) selected.
        forward_no_pos, base_no_pos = resolve_forward_reference_joints(
            joint_names, parents, object_type='DegenerateNeck',
        )
        self.assertEqual(forward_no_pos, 1)
        self.assertIsNone(base_no_pos)

        # With positions the neck is skipped and the tail->spine axis is used.
        forward_idx, base_idx = resolve_forward_reference_joints(
            joint_names, parents, object_type='DegenerateNeck', rest_positions=rest_positions,
        )
        self.assertEqual(forward_idx, 2)   # Bip01_Spine
        self.assertEqual(base_idx, 4)      # Bip01_Tail

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

    def test_tail_spine_fallback_does_not_emit_torso_head_candidate(self):
        joints = np.zeros((1, 9, 3), dtype=np.float64)
        joints[:, 3] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        joints[:, 1] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        joints[:, 5] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        joints[:, 6] = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        joints[:, 7] = np.array([1.0, 1.0, 0.0], dtype=np.float64)
        joints[:, 8] = np.array([-1.0, 1.0, 0.0], dtype=np.float64)

        candidates = _get_facing_candidates(
            joints,
            'FallbackBug',
            face_joint_indx=[5, 6, 7, 8],
            forward_joint_index=1,
            forward_base_joint_index=3,
        )

        self.assertEqual(set(candidates.keys()), {'tail_spine', 'across'})

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

    def test_numeric_zoo_rigs_use_chain_forward_joints(self):
        jaws_positions = np.zeros((1, 16, 3), dtype=np.float64)
        jaws_positions[:, 15] = np.array([-0.8, 0.0, 0.0], dtype=np.float64)
        jaws_positions[:, 3] = np.array([0.3, 0.0, 0.0], dtype=np.float64)
        jaws_candidates = _get_facing_candidates(jaws_positions, 'Jaws')

        crow_positions = np.zeros((1, 24, 3), dtype=np.float64)
        crow_positions[:, 8] = np.array([-0.4, 0.0, 0.0], dtype=np.float64)
        crow_positions[:, 22] = np.array([0.6, 0.0, 0.0], dtype=np.float64)
        crow_candidates = _get_facing_candidates(crow_positions, 'Crow')

        np.testing.assert_allclose(jaws_candidates['chain'][0], np.array([1.0, 0.0, 0.0]), atol=1e-8)
        np.testing.assert_allclose(crow_candidates['chain'][0], np.array([1.0, 0.0, 0.0]), atol=1e-8)

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

    def test_resolve_face_joints_prefers_homologous_crab_pairs(self):
        joint_names = [
            'Hips',
            'BN_Bip01_Pelvis',
            'BN_leg_R_01',
            'BN_leg_R_06',
            'BN_Leg_R_11',
            'BN_Leg_L_11',
            'BN_leg_L_06',
            'BN_leg_L_01',
            'BN_Arm_L_01',
            'BN_Arm_R_01',
        ]
        parents = np.array([-1, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int64)

        face_joints = resolve_face_joints('CrabLike', joint_names, parents)

        self.assertEqual(face_joints, [2, 7, 9, 8])

    def test_homologous_crab_pairs_keep_across_aligned_with_heading(self):
        joints = np.zeros((1, 10, 3), dtype=np.float64)
        joints[:, 2] = np.array([8.997085, 22.944474, -3.344254], dtype=np.float64)
        joints[:, 7] = np.array([-6.384427, 22.558812, -6.23502], dtype=np.float64)
        joints[:, 8] = np.array([-5.843798, 25.862572, -10.677262], dtype=np.float64)
        joints[:, 9] = np.array([8.728792, 24.642908, -8.238453], dtype=np.float64)

        candidates = _get_facing_candidates(
            joints,
            'CrabLike',
            face_joint_indx=[2, 7, 9, 8],
            forward_joint_index=None,
            forward_base_joint_index=None,
        )

        self.assertGreater(candidates['across'][0, 0], 0.0)
        self.assertLess(candidates['across'][0, 2], 0.0)


    def test_fills_missing_hip_slot_from_generic_pair(self):
        # Mirrors a renamer output where the left thigh was left "Unknown": the
        # hip keyword search finds only the right side, but calves/arms are
        # symmetric. The lateral axis must still resolve (no blind +Z warning).
        joint_names = [
            'Hips', 'Spine', 'Head',
            'LeftClavicle', 'RightClavicle',
            'LeftCalf', 'RightCalf',
            'Unknown.004', 'RightThigh',
        ]
        parents = np.array([-1, 0, 1, 1, 1, 7, 8, 0, 0], dtype=np.int64)

        with patch('builtins.print') as mock_print:
            face_joints = resolve_face_joints('Horse', joint_names, parents)

        self.assertEqual(len(face_joints), 4)
        # Upper slot keeps the semantic clavicle pair; hip slot is filled by a
        # generic homologous pair (calf), not left empty.
        self.assertEqual((face_joints[2], face_joints[3]), (4, 3))
        self.assertNotIn(
            '[WARN] Horse: no left-right joint pairs found; using default +Z orientation. '
            'Provide --face-joints-names explicitly if a different orientation is needed.',
            [call.args[0] for call in mock_print.call_args_list],
        )

    def test_single_upper_pair_alone_resolves(self):
        joint_names = ['Hips', 'Spine', 'LeftClavicle', 'RightClavicle']
        parents = np.array([-1, 0, 1, 1], dtype=np.int64)

        face_joints = resolve_face_joints('OneGirdle', joint_names, parents)

        self.assertEqual(face_joints, [3, 2, 3, 2])

    def test_geometric_mirror_fallback_for_unnamed_skeleton(self):
        # No L/R tokens anywhere; the lateral axis must come from rest-pose
        # bilateral symmetry. Limbs spread along X, head forward along +Z.
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.8, 0.0, 0.2], [-0.8, 0.0, 0.2],
            [0.6, 0.0, -0.5], [-0.6, 0.0, -0.5],
            [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
        ], dtype=np.float64)
        joint_names = [f'j{i}' for i in range(len(positions))]
        parents = np.array([-1, 0, 0, 0, 0, 0, 2, 3], dtype=np.int64)

        with patch('builtins.print') as mock_print:
            face_joints = resolve_face_joints(
                'Unnamed', joint_names, parents, rest_positions=positions
            )

        # Strongest mirror pair is the widest one (the hands at +/-1.0 on X).
        self.assertEqual(face_joints, [6, 7, 6, 7])
        warnings = [call.args[0] for call in mock_print.call_args_list]
        self.assertTrue(any('mirror symmetry' in w for w in warnings))

    def test_truly_asymmetric_skeleton_still_warns_and_returns_empty(self):
        joint_names = ['Root', 'LegA', 'LegB', 'LegC']
        parents = np.array([-1, 0, 1, 2], dtype=np.int64)

        with patch('builtins.print') as mock_print:
            face_joints = resolve_face_joints('LegsOnly', joint_names, parents)

        self.assertEqual(face_joints, [])
        mock_print.assert_called_once_with(
            '[WARN] LegsOnly: no left-right joint pairs found; using default +Z orientation. '
            'Provide --face-joints-names explicitly if a different orientation is needed.'
        )


if __name__ == '__main__':
    unittest.main()