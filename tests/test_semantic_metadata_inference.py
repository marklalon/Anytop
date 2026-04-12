from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


from data_loaders.truebones.truebones_utils.motion_process import (  # noqa: E402
    _build_semantic_metadata,
    _is_idle_reference_path,
    _is_walk_reference_path,
    find_orientation_reference_path,
)


class SemanticMetadataInferenceTests(unittest.TestCase):
    def test_contact_joints_drop_upper_limb_ball_chain(self):
        joint_names = [
            'Hips',
            'NPC_LLeg',
            'NPC_LLegAnkle',
            'NPC_LLegBall1',
            'NPC_L_Toe',
            'NPC_RLeg',
            'NPC_RLegAnkle',
            'NPC_RLegBall1',
            'NPC_R_Toe',
            'NPC_LArm',
            'NPC_LArmBall1',
            'NPC_LIndex01',
            'NPC_RArm',
            'NPC_RArmBall1',
            'NPC_RIndex01',
        ]
        parents = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 0, 12, 13], dtype=np.int64)
        offsets = np.array([
            [0.0, 0.0, 0.0],
            [-0.2, -0.4, 0.0],
            [0.0, -0.4, 0.0],
            [0.0, -0.1, 0.1],
            [0.0, -0.05, 0.15],
            [0.2, -0.4, 0.0],
            [0.0, -0.4, 0.0],
            [0.0, -0.1, 0.1],
            [0.0, -0.05, 0.15],
            [-0.35, 0.4, 0.0],
            [0.0, -0.05, 0.1],
            [0.0, -0.02, 0.08],
            [0.35, 0.4, 0.0],
            [0.0, -0.05, 0.1],
            [0.0, -0.02, 0.08],
        ], dtype=np.float64)

        semantic_metadata = _build_semantic_metadata('Bear', joint_names, parents, offsets)

        self.assertIn('NPC_LLegAnkle', semantic_metadata['contact_joint_names'])
        self.assertIn('NPC_RLegAnkle', semantic_metadata['contact_joint_names'])
        self.assertNotIn('NPC_LArmBall1', semantic_metadata['contact_joint_names'])
        self.assertNotIn('NPC_RArmBall1', semantic_metadata['contact_joint_names'])
        self.assertNotIn('NPC_LIndex01', semantic_metadata['contact_joint_names'])
        self.assertNotIn('NPC_RIndex01', semantic_metadata['contact_joint_names'])

    def test_end_effectors_drop_accessories_and_cosmetics(self):
        joint_names = [
            'Hips',
            'BN_Tail_05',
            'Bip01_R_Toe0Nub',
            'Bip01_L_Toe0Nub',
            'Saddle',
            'BN_hair04_03',
            'Bip01_R_Finger0Nub',
            'Bip01_L_Finger0Nub',
            'Bip01_HeadNub',
            'BN_Halter_R_02',
        ]
        parents = np.array([-1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
        offsets = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.2, -0.8],
            [0.3, -0.8, 0.3],
            [-0.3, -0.8, 0.3],
            [0.0, 0.1, 0.2],
            [0.0, 0.5, -0.1],
            [0.5, -0.2, 0.5],
            [-0.5, -0.2, 0.5],
            [0.0, 0.8, 0.7],
            [0.2, 0.5, 0.4],
        ], dtype=np.float64)

        semantic_metadata = _build_semantic_metadata('Horse', joint_names, parents, offsets)

        self.assertIn('BN_Tail_05', semantic_metadata['end_effector_names'])
        self.assertIn('Bip01_R_Toe0Nub', semantic_metadata['end_effector_names'])
        self.assertIn('Bip01_L_Toe0Nub', semantic_metadata['end_effector_names'])
        self.assertIn('Bip01_HeadNub', semantic_metadata['end_effector_names'])
        self.assertNotIn('Saddle', semantic_metadata['end_effector_names'])
        self.assertNotIn('BN_hair04_03', semantic_metadata['end_effector_names'])
        self.assertNotIn('BN_Halter_R_02', semantic_metadata['end_effector_names'])

    def test_contact_fallback_prefers_low_leg_leaves(self):
        joint_names = [
            'Root',
            'RightLeg',
            'LeftLeg',
            'Tail01',
            'Head',
            'LeftForeArm',
            'RightForeArm',
        ]
        parents = np.array([-1, 0, 0, 0, 0, 0, 0], dtype=np.int64)
        offsets = np.array([
            [0.0, 0.0, 0.0],
            [0.2, -1.0, 0.1],
            [-0.2, -1.0, 0.1],
            [0.0, -0.2, -0.5],
            [0.0, 0.9, 0.7],
            [-0.7, 0.4, 0.3],
            [0.7, 0.4, 0.3],
        ], dtype=np.float64)

        semantic_metadata = _build_semantic_metadata('Pigeon', joint_names, parents, offsets)

        self.assertEqual(sorted(semantic_metadata['contact_joint_names']), ['LeftLeg', 'RightLeg'])

    def test_contact_chain_keeps_foot_with_grounded_claws(self):
        joint_names = [
            'Root',
            'jt_Foot_L',
            'jt_ToeMiddle_L',
            'jt_ClawMiddle_L',
            'jt_Foot_R',
            'jt_ToeMiddle_R',
            'jt_ClawMiddle_R',
        ]
        parents = np.array([-1, 0, 1, 2, 0, 4, 5], dtype=np.int64)
        offsets = np.array([
            [0.0, 0.0, 0.0],
            [-0.2, -0.7, 0.1],
            [0.0, -0.15, 0.08],
            [0.0, -0.08, 0.1],
            [0.2, -0.7, 0.1],
            [0.0, -0.15, 0.08],
            [0.0, -0.08, 0.1],
        ], dtype=np.float64)

        semantic_metadata = _build_semantic_metadata('Pteranodon', joint_names, parents, offsets)

        self.assertIn('jt_Foot_L', semantic_metadata['contact_joint_names'])
        self.assertIn('jt_Foot_R', semantic_metadata['contact_joint_names'])
        self.assertIn('jt_ClawMiddle_L', semantic_metadata['contact_joint_names'])
        self.assertIn('jt_ClawMiddle_R', semantic_metadata['contact_joint_names'])

    def test_quadruped_forelegs_named_as_fingers_keep_distal_contacts_only(self):
        joint_names = [
            'Hips',
            'Bip01_R_Calf',
            'Bip01_R_HorseLink',
            'Bip01_R_Foot',
            'Bip01_R_Toe0',
            'Bip01_R_Toe0Nub',
            'Bip01_L_Calf',
            'Bip01_L_HorseLink',
            'Bip01_L_Foot',
            'Bip01_L_Toe0',
            'Bip01_L_Toe0Nub',
            'Bip01_R_Hand',
            'Bip01_R_Finger0',
            'Bip01_R_Finger0Nub',
            'Bip01_L_Hand',
            'Bip01_L_Finger0',
            'Bip01_L_Finger0Nub',
            'Bip01_HeadNub',
        ]
        parents = np.array([-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 0, 14, 15, 0], dtype=np.int64)
        offsets = np.array([
            [0.0, 0.0, 0.0],
            [-0.12, 0.12, -0.64],
            [0.0, -0.12, 0.06],
            [0.0, -0.10, 0.04],
            [0.0, -0.05, 0.08],
            [0.0, 0.0, 0.09],
            [0.12, 0.12, -0.64],
            [0.0, -0.12, 0.06],
            [0.0, -0.10, 0.04],
            [0.0, -0.05, 0.08],
            [0.0, 0.0, 0.09],
            [-0.08, 0.04, 0.40],
            [0.0, -0.10, 0.16],
            [0.0, -0.03, 0.09],
            [0.08, 0.04, 0.40],
            [0.0, -0.10, 0.16],
            [0.0, -0.03, 0.09],
            [0.0, -0.08, 0.08],
            [0.0, 0.85, 0.72],
        ], dtype=np.float64)

        semantic_metadata = _build_semantic_metadata('Horse', joint_names, parents, offsets)

        self.assertNotIn('Bip01_R_Calf', semantic_metadata['contact_joint_names'])
        self.assertNotIn('Bip01_R_HorseLink', semantic_metadata['contact_joint_names'])
        self.assertNotIn('Bip01_L_Calf', semantic_metadata['contact_joint_names'])
        self.assertNotIn('Bip01_L_HorseLink', semantic_metadata['contact_joint_names'])
        self.assertNotIn('Bip01_R_Hand', semantic_metadata['contact_joint_names'])
        self.assertIn('Bip01_R_Finger0Nub', semantic_metadata['contact_joint_names'])
        self.assertNotIn('Bip01_L_Hand', semantic_metadata['contact_joint_names'])
        self.assertIn('Bip01_L_Finger0Nub', semantic_metadata['contact_joint_names'])
        self.assertIn('Bip01_R_Toe0Nub', semantic_metadata['contact_joint_names'])
        self.assertIn('Bip01_L_Toe0Nub', semantic_metadata['contact_joint_names'])
        self.assertTrue(
            {'Bip01_R_Finger0Nub', 'Bip01_L_Finger0Nub', 'Bip01_R_Toe0Nub', 'Bip01_L_Toe0Nub'}.issubset(
                set(semantic_metadata['end_effector_names'])
            )
        )

    def test_contact_chain_respects_cumulative_offset_limit(self):
        joint_names = [
            'Hips',
            'Bip01_R_Ankle',
            'Bip01_R_Foot',
            'Bip01_R_Toe0',
            'Bip01_R_Toe0Nub',
            'Bip01_L_Ankle',
            'Bip01_L_Foot',
            'Bip01_L_Toe0',
            'Bip01_L_Toe0Nub',
            'Bip01_HeadNub',
        ]
        parents = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0], dtype=np.int64)
        offsets = np.array([
            [0.0, 0.0, 0.0],
            [-0.14, 0.26, -0.55],
            [0.0, -0.12, 0.11],
            [0.0, -0.11, 0.12],
            [0.0, -0.08, 0.12],
            [0.14, 0.26, -0.55],
            [0.0, -0.12, 0.11],
            [0.0, -0.11, 0.12],
            [0.0, -0.08, 0.12],
            [0.0, 0.9, 0.7],
        ], dtype=np.float64)

        semantic_metadata = _build_semantic_metadata('Horse', joint_names, parents, offsets)

        self.assertIn('Bip01_R_Foot', semantic_metadata['contact_joint_names'])
        self.assertIn('Bip01_L_Foot', semantic_metadata['contact_joint_names'])
        self.assertNotIn('Bip01_R_Ankle', semantic_metadata['contact_joint_names'])
        self.assertNotIn('Bip01_L_Ankle', semantic_metadata['contact_joint_names'])


class OrientationReferencePathTests(unittest.TestCase):
    def test_idle_reference_requires_pure_reference_tail(self):
        self.assertTrue(_is_idle_reference_path('Lion/__Idle.bvh'))
        self.assertTrue(_is_idle_reference_path('Camel/__IdleLoop.bvh'))
        self.assertTrue(_is_idle_reference_path('Elephant/__Idle2.bvh'))

        self.assertFalse(_is_idle_reference_path('Lion/__DeathIdle.bvh'))
        self.assertFalse(_is_idle_reference_path('Lion/__SlowIdle.bvh'))
        self.assertFalse(_is_idle_reference_path('Cat/CAT_IdlePurr.bvh'))
        self.assertFalse(_is_idle_reference_path('Puppy/Puppy_IdleLayDown.bvh'))
        self.assertFalse(_is_idle_reference_path('Trex/__idle_investigate.bvh'))

    def test_walk_reference_requires_pure_reference_tail(self):
        self.assertTrue(_is_walk_reference_path('Lion/__Walk.bvh'))
        self.assertTrue(_is_walk_reference_path('Buffalo/__WalkLoop.bvh'))
        self.assertTrue(_is_walk_reference_path('Deer/__WalkForward.bvh'))
        self.assertTrue(_is_walk_reference_path('Coyote/__Walking.bvh'))

        self.assertFalse(_is_walk_reference_path('Lion/__SlowWalk.bvh'))
        self.assertFalse(_is_walk_reference_path('SabreToothTiger/__Startwalk.bvh'))
        self.assertFalse(_is_walk_reference_path('Trex/__walk_bite.bvh'))
        self.assertFalse(_is_walk_reference_path('Trex/__walk_slow_loop.bvh'))

    def test_reference_selection_skips_compound_idle_and_uses_walk(self):
        bvh_files = [
            'Lion/__Attack.bvh',
            'Lion/__DeathIdle.bvh',
            'Lion/__SlowIdle.bvh',
            'Lion/__Walk.bvh',
        ]

        selected, source = find_orientation_reference_path(list(bvh_files))

        self.assertEqual(selected, 'Lion/__Walk.bvh')
        self.assertEqual(source, 'walk')

    def test_reference_selection_keeps_idle_loop_priority_over_walk(self):
        bvh_files = [
            'Camel/__IdleLoop.bvh',
            'Camel/__Walk.bvh',
        ]

        selected, source = find_orientation_reference_path(list(bvh_files))

        self.assertEqual(selected, 'Camel/__IdleLoop.bvh')
        self.assertEqual(source, 'idle')


if __name__ == '__main__':
    unittest.main()