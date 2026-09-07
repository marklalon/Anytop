"""The structural joint channel: what it encodes, and that a rename cannot move it.

The channel exists because the joint-name embedding used to be the only
per-joint identity in the token, so respelling a leg could change how the knee
bends. These pin the two properties that make it a usable fallback -- it is a
pure function of geometry and topology, and it separates joints the slim name
text deliberately collapses -- plus the contracts the builder refuses to guess
at (a forest, a cycle, a contact index off the end).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_loaders.tensors import truebones_batch_collate  # noqa: E402
from data_loaders.truebones.truebones_utils.joint_struct_features import (  # noqa: E402
    JOINT_STRUCT_DIM,
    JOINT_STRUCT_FEATURE_NAMES,
    JointStructFeatureError,
    build_joint_struct_features,
)
from model.anytop import InputProcess  # noqa: E402


def _channel(features, name):
    return features[:, JOINT_STRUCT_FEATURE_NAMES.index(name)]


#   0 Hips
#   +- 1 Spine -- 2 Head
#   +- 3 LThigh -- 4 LShin -- 5 LFoot
#   +- 6 RThigh -- 7 RShin -- 8 RFoot
_BIPED_PARENTS = [-1, 0, 1, 0, 3, 4, 0, 6, 7]
_BIPED_REST = np.array(
    [
        [0.0, 1.00, 0.0],
        [0.0, 1.40, 0.0],
        [0.0, 1.70, 0.0],
        [-0.2, 0.90, 0.0],
        [-0.2, 0.50, 0.0],
        [-0.2, 0.00, 0.1],
        [0.2, 0.90, 0.0],
        [0.2, 0.50, 0.0],
        [0.2, 0.00, 0.1],
    ],
    dtype=np.float32,
)


def _biped(**overrides):
    entry = {
        'parents': np.asarray(_BIPED_PARENTS, dtype=np.int64),
        'rest_pos_ric_hml': _BIPED_REST.copy(),
        'contact_joints': [5, 8],
        'contact_joint_source': 'geometry',
        'joints_names': ['Hips', 'Spine', 'Head', 'LThigh', 'LShin', 'LFoot',
                         'RThigh', 'RShin', 'RFoot'],
    }
    entry.update(overrides)
    return entry


class BranchFreeRuns(unittest.TestCase):
    def test_root_is_its_own_run_and_every_child_opens_a_new_one(self):
        features = build_joint_struct_features(_biped())
        depth = _channel(features, 'depth_norm')
        run_len_inv = _channel(features, 'run_len_inv')
        # The root is a run of one whatever its degree.
        self.assertAlmostEqual(float(depth[0]), 1.0)
        self.assertAlmostEqual(float(run_len_inv[0]), 1.0)
        # Spine -> Head is one run of two; each leg is one run of three.
        np.testing.assert_allclose(depth[[1, 2]], [0.5, 1.0], atol=1e-6)
        np.testing.assert_allclose(depth[[3, 4, 5]], [1 / 3, 2 / 3, 1.0], atol=1e-6)
        np.testing.assert_allclose(run_len_inv[[3, 4, 5]], [1 / 3] * 3, atol=1e-6)

    def test_a_branch_point_ends_its_own_run_and_its_children_start_new_ones(self):
        #  0 -- 1 -- 2 -+- 3
        #                +- 4
        entry = {
            'parents': np.asarray([-1, 0, 1, 2, 2], dtype=np.int64),
            'rest_pos_ric_hml': np.array(
                [[0, 0, 0], [0, 1, 0], [0, 2, 0], [-1, 3, 0], [1, 3, 0]], dtype=np.float32
            ),
            'contact_joints': [],
            'contact_joint_source': 'none',
        }
        run_len_inv = _channel(build_joint_struct_features(entry), 'run_len_inv')
        # 1 and 2 share a run of two; the fork's children are runs of one each.
        np.testing.assert_allclose(run_len_inv, [1.0, 0.5, 0.5, 1.0, 1.0], atol=1e-6)

    def test_attach_height_is_the_run_start_not_the_joint(self):
        features = build_joint_struct_features(_biped())
        height = _channel(features, 'height_n')
        attach = _channel(features, 'attach_h_n')
        # Every joint of the left leg reports the thigh's height.
        np.testing.assert_allclose(attach[[3, 4, 5]], [height[3]] * 3, atol=1e-6)
        self.assertGreater(float(attach[5]), float(height[5]))


class ContactAndShape(unittest.TestCase):
    def test_the_grounded_limb_is_marked_and_the_neck_is_not(self):
        features = build_joint_struct_features(_biped())
        ends_contact = _channel(features, 'run_ends_contact')
        np.testing.assert_allclose(ends_contact[[3, 4, 5, 6, 7, 8]], 1.0)
        np.testing.assert_allclose(ends_contact[[0, 1, 2]], 0.0)

    def test_a_contact_one_joint_below_the_run_still_marks_it(self):
        # A toe annotated as the contact opens its own run; the foot's run must
        # still read as grounded.
        entry = {
            'parents': np.asarray([-1, 0, 1, 2, 2], dtype=np.int64),
            'rest_pos_ric_hml': np.array(
                [[0, 1, 0], [0, .5, 0], [0, 0, 0], [0, 0, .1], [0, 0, -.1]], dtype=np.float32
            ),
            'contact_joints': [3],
            'contact_joint_source': 'names',
        }
        ends_contact = _channel(build_joint_struct_features(entry), 'run_ends_contact')
        self.assertEqual(float(ends_contact[1]), 1.0)  # the run ending at joint 2
        self.assertEqual(float(ends_contact[2]), 1.0)

    def test_contact_known_separates_no_contact_from_no_annotation(self):
        annotated = build_joint_struct_features(_biped())
        unannotated = build_joint_struct_features(
            _biped(contact_joints=[], contact_joint_source='none')
        )
        np.testing.assert_allclose(_channel(annotated, 'contact_known'), 1.0)
        np.testing.assert_allclose(_channel(unannotated, 'contact_known'), 0.0)
        np.testing.assert_allclose(_channel(unannotated, 'is_contact'), 0.0)

    def test_leaves_heights_and_subtree_sizes(self):
        features = build_joint_struct_features(_biped())
        np.testing.assert_allclose(
            _channel(features, 'is_leaf'), [0, 0, 1, 0, 0, 1, 0, 0, 1], atol=1e-6
        )
        height = _channel(features, 'height_n')
        self.assertAlmostEqual(float(height[2]), 1.0)  # Head is the top
        self.assertAlmostEqual(float(height[5]), 0.0)  # LFoot is the bottom
        subtree = _channel(features, 'subtree_n')
        self.assertAlmostEqual(float(subtree[0]), 1.0)  # the root's subtree is all of it
        self.assertAlmostEqual(float(subtree[2]), 1 / 9)
        self.assertAlmostEqual(float(subtree[3]), 3 / 9)

    def test_lateral_is_signed_so_left_and_right_never_coincide(self):
        features = build_joint_struct_features(_biped())
        lateral = _channel(features, 'lateral_signed')
        self.assertLess(float(lateral[3]), 0.0)
        self.assertGreater(float(lateral[6]), 0.0)
        self.assertAlmostEqual(float(lateral[0]), 0.0)

    def test_sibling_rank_separates_a_mirrored_pair(self):
        features = build_joint_struct_features(_biped())
        rank = _channel(features, 'sib_rank')
        n_inv = _channel(features, 'sib_n_inv')
        # Hips has three children (Spine, LThigh, RThigh) ordered by (z, x).
        np.testing.assert_allclose(n_inv[[1, 3, 6]], [1 / 3] * 3, atol=1e-6)
        np.testing.assert_allclose(rank[[3, 1, 6]], [0.0, 0.5, 1.0], atol=1e-6)
        # An only child ranks 0 and reports a sibling count of one.
        self.assertAlmostEqual(float(rank[4]), 0.0)
        self.assertAlmostEqual(float(n_inv[4]), 1.0)

    def test_a_repeated_sibling_pair_is_told_apart(self):
        """What the retired 'Instance Second Of 4' ordinal used to do -- four
        interchangeable necks off one parent share a slim text and must not share
        a structural row."""
        entry = {
            'parents': np.asarray([-1, 0, 1, 1, 1, 1], dtype=np.int64),
            'rest_pos_ric_hml': np.array(
                [[0, 0, 0], [0, 1, 0], [-1, 2, 0], [-.3, 2, 0], [.3, 2, 0], [1, 2, 0]],
                dtype=np.float32,
            ),
            'contact_joints': [],
            'contact_joint_source': 'none',
        }
        necks = build_joint_struct_features(entry)[2:]
        self.assertEqual(len({tuple(row) for row in necks.tolist()}), 4)


class NameInvarianceAndDeterminism(unittest.TestCase):
    def test_renaming_every_joint_changes_nothing(self):
        baseline = build_joint_struct_features(_biped())
        renamed = build_joint_struct_features(_biped(
            joints_names=['q'] * 9,
            canonical_joint_names=['LeftArm'] * 9,
        ))
        self.assertTrue(np.array_equal(baseline, renamed))

    def test_repeated_calls_are_bit_identical(self):
        entry = _biped()
        self.assertTrue(np.array_equal(
            build_joint_struct_features(entry), build_joint_struct_features(entry)
        ))

    def test_shape_and_dtype(self):
        features = build_joint_struct_features(_biped())
        self.assertEqual(features.shape, (9, JOINT_STRUCT_DIM))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.isfinite(features).all())

    def test_rest_pose_is_the_documented_fallback(self):
        rest_pose = np.zeros((9, 13), dtype=np.float32)
        rest_pose[:, 0:3] = _BIPED_REST
        entry = _biped()
        del entry['rest_pos_ric_hml']
        entry['rest_pose'] = rest_pose
        self.assertTrue(np.array_equal(
            build_joint_struct_features(entry), build_joint_struct_features(_biped())
        ))


class MalformedInputIsRefused(unittest.TestCase):
    def _expect(self, **overrides):
        with self.assertRaises(JointStructFeatureError):
            build_joint_struct_features(_biped(**overrides))

    def test_two_roots(self):
        self._expect(parents=np.asarray([-1, 0, 1, -1, 3, 4, 0, 6, 7], dtype=np.int64))

    def test_root_is_not_joint_zero(self):
        self._expect(parents=np.asarray([1, -1, 1, 0, 3, 4, 0, 6, 7], dtype=np.int64))

    def test_a_joint_parented_to_itself(self):
        self._expect(parents=np.asarray([-1, 1, 1, 0, 3, 4, 0, 6, 7], dtype=np.int64))

    def test_a_cycle(self):
        self._expect(parents=np.asarray([-1, 2, 1, 0, 3, 4, 0, 6, 7], dtype=np.int64))

    def test_a_parent_index_past_the_end(self):
        self._expect(parents=np.asarray([-1, 0, 99, 0, 3, 4, 0, 6, 7], dtype=np.int64))

    def test_a_contact_index_past_the_end(self):
        self._expect(contact_joints=[5, 99])

    def test_non_finite_rest_positions(self):
        broken = _BIPED_REST.copy()
        broken[4, 1] = np.nan
        self._expect(rest_pos_ric_hml=broken)

    def test_a_rest_pose_of_the_wrong_length(self):
        self._expect(rest_pos_ric_hml=_BIPED_REST[:5].copy())

    def test_missing_parents(self):
        with self.assertRaises(JointStructFeatureError):
            build_joint_struct_features({'rest_pos_ric_hml': _BIPED_REST.copy()})

    def test_a_degenerate_rig_stays_finite(self):
        """Every joint at one point: the span floors instead of dividing by zero."""
        features = build_joint_struct_features(
            _biped(rest_pos_ric_hml=np.zeros((9, 3), dtype=np.float32))
        )
        self.assertTrue(np.isfinite(features).all())


class PaddingSurvivesTheProjection(unittest.TestCase):
    def _item(self, n_joints, max_joints, n_feats=13):
        motion = np.zeros((4, n_joints, n_feats), dtype=np.float32)
        return [
            motion, 4,
            np.asarray(_BIPED_PARENTS[:n_joints], dtype=np.int64),
            np.zeros((n_joints, n_feats), dtype=np.float32),
            np.zeros((n_joints, 3), dtype=np.float32),
            np.zeros((n_joints, n_joints), dtype=np.int64),
            np.zeros((n_joints, n_joints), dtype=np.int64),
            'truebones/zoo/Test',
            np.zeros((n_joints, 8), dtype=np.float32),
            max_joints,
            {'translation_root_index': 0},
            'clip',
            {'joint_struct': np.arange(n_joints * JOINT_STRUCT_DIM, dtype=np.float32)
                .reshape(n_joints, JOINT_STRUCT_DIM) + 1.0},
        ]

    def test_collate_pads_the_rows_past_n_joints_with_zeros(self):
        _motion, cond = truebones_batch_collate([self._item(4, 6), self._item(6, 6)])
        joint_struct = cond['y']['joint_struct']
        self.assertEqual(tuple(joint_struct.shape), (2, 6, JOINT_STRUCT_DIM))
        self.assertTrue(torch.equal(joint_struct[0, 4:], torch.zeros(2, JOINT_STRUCT_DIM)))
        self.assertTrue(bool((joint_struct[0, :4] != 0).all()))

    def test_collate_refuses_a_descriptor_of_the_wrong_width(self):
        item = self._item(4, 6)
        item[-1]['joint_struct'] = np.zeros((4, JOINT_STRUCT_DIM + 1), dtype=np.float32)
        with self.assertRaises(ValueError):
            truebones_batch_collate([item])

    def test_the_projected_padding_latent_is_exactly_zero(self):
        """The MLP has biases, so a zero input row does NOT stay zero through it;
        the mask has to be reapplied on the output side."""
        torch.manual_seed(0)
        process = InputProcess(13, 13, 16, 32, dropout_prob=0.0).eval()
        valid = torch.tensor([[True, True, False, False]])
        struct = torch.randn(1, 4, JOINT_STRUCT_DIM) * valid.unsqueeze(-1)
        with torch.no_grad():
            raw = process.struct_embedding(struct)
            masked = raw * valid.unsqueeze(-1)
        self.assertGreater(float(raw[0, 2:].abs().max()), 0.0)
        self.assertEqual(float(masked[0, 2:].abs().max()), 0.0)

    def test_input_process_refuses_to_run_without_the_descriptors(self):
        process = InputProcess(13, 13, 16, 32, dropout_prob=0.0).eval()
        with self.assertRaises(ValueError):
            process(
                torch.randn(1, 4, 13, 3), torch.randn(1, 1, 4, 13),
                torch.randn(1, 4, 32), None, torch.ones(1, 4, dtype=torch.bool),
            )


if __name__ == '__main__':
    unittest.main()
