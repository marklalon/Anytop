from __future__ import annotations

import math
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


from diffusion.gaussian_diffusion import GaussianDiffusion, LossType, ModelMeanType, ModelVarType  # noqa: E402
from data_loaders.truebones.data.dataset import _limb_tip_mask  # noqa: E402
from data_loaders.truebones.truebones_utils.canonical_features import REST_LENGTH_SCALE_KEY  # noqa: E402
from data_loaders.truebones.truebones_utils import joint_parts  # noqa: E402
from data_loaders.truebones.truebones_utils.joint_parts import (  # noqa: E402
    bone_path_lengths,
    chain_tips,
    foot_chains,
    limb_end_chains,
    limb_tips,
)
from data_loaders.truebones.truebones_utils.leaf_drop import drop_joints_from_cond  # noqa: E402


N_FRAMES = 60
PERIOD = 20
# root, spine, then one tip per leg
TIPS = (2, 3, 4, 5)


def _diffusion() -> GaussianDiffusion:
    return GaussianDiffusion(
        betas=np.array([0.001, 0.002, 0.003], dtype=np.float64),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
        lambda_limb_corr=1.0,
    )


def _gait(phases, amplitude=0.2, shift=0):
    """[1, J, 3, T] physical positions: each tip swings fore/aft and lifts at
    its own phase; root and spine stay put."""
    t = torch.arange(N_FRAMES, dtype=torch.float64) + shift
    pos = torch.zeros(1, 6, 3, N_FRAMES, dtype=torch.float64)
    pos[:, 0, 1] = 1.0
    pos[:, 1, 1] = 1.0
    for tip, phase in zip(TIPS, phases):
        if phase is None:
            continue
        angle = 2 * math.pi * t / PERIOD + phase
        pos[0, tip, 2] = amplitude * torch.sin(angle)
        pos[0, tip, 1] = 0.5 * amplitude * torch.cos(angle).clamp_min(0.0)
    return pos


TROT = (0.0, math.pi, math.pi, 0.0)


def _y():
    tips = torch.zeros(1, 6, dtype=torch.bool)
    tips[0, list(TIPS)] = True
    return {'limb_tip_mask': tips, REST_LENGTH_SCALE_KEY: torch.tensor([1.0], dtype=torch.float64)}


def _loss(pred, target, y=None):
    lengths = torch.tensor([N_FRAMES], dtype=torch.float64)
    return _diffusion().limb_correlation_loss(pred, target, _y() if y is None else y, lengths)


class LimbCorrelationLossTests(unittest.TestCase):
    def test_matching_coordination_is_zero(self):
        target = _gait(TROT)
        self.assertAlmostEqual(_loss(target.clone(), target).item(), 0.0, places=10)

    def test_one_leg_out_of_phase_is_penalized(self):
        target = _gait(TROT)
        swapped = _gait((0.0, 0.0, math.pi, 0.0))
        self.assertGreater(_loss(swapped, target).item(), 0.1)

    def test_global_time_shift_is_nearly_free(self):
        target = _gait(TROT)
        shifted = _gait(TROT, shift=PERIOD // 4)
        swapped = _gait((0.0, 0.0, math.pi, 0.0))
        self.assertLess(_loss(shifted, target).item(), 0.05 * _loss(swapped, target).item())

    def test_amplitude_scale_is_nearly_free(self):
        target = _gait(TROT)
        blurred = _gait(TROT, amplitude=0.06)
        swapped = _gait((0.0, 0.0, math.pi, 0.0))
        self.assertLess(_loss(blurred, target).item(), 0.05 * _loss(swapped, target).item())

    def test_limb_still_in_target_drops_out(self):
        # A one-sided strike: one leg moves, the others hold.
        target = _gait((0.0, None, None, None))
        noisy = target.clone()
        noisy[0, 3, 2] = 0.05 * torch.randn(N_FRAMES, dtype=torch.float64)
        self.assertLess(_loss(noisy, target).item(), 1e-3)

    def test_still_clip_and_still_prediction_have_finite_gradient(self):
        target = _gait((None, None, None, None))
        pred = torch.zeros_like(target, requires_grad=True)
        loss = _loss(pred, target)
        loss.sum().backward()
        self.assertTrue(torch.isfinite(loss).all())
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertAlmostEqual(loss.item(), 0.0, places=10)

    def test_no_limbs_is_zero(self):
        target = _gait(TROT)
        y = _y()
        y['limb_tip_mask'] = torch.zeros_like(y['limb_tip_mask'])
        self.assertEqual(_loss(_gait((0.0, 0.0, 0.0, 0.0)), target, y).item(), 0.0)


class LimbTipMaskTests(unittest.TestCase):
    def _cond(self):
        # root -> hip -> knee -> foot -> toe, twice
        parents = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7])
        rest = np.array([
            [0, 1.0, 0],
            [0.2, 0.9, 0], [0.2, 0.5, 0], [0.2, 0.1, 0], [0.2, 0.0, 0.1],
            [-0.2, 0.9, 0], [-0.2, 0.5, 0], [-0.2, 0.1, 0], [-0.2, 0.0, 0.1],
        ])
        offsets = np.where(parents[:, None] >= 0, rest - rest[np.maximum(parents, 0)], 0.0)
        return {
            'parents': parents,
            'offsets': offsets,
            'limb_reach': bone_path_lengths(parents, offsets),
            'limb_chain_id': np.array([-1, -1, -1, 0, 0, -1, -1, 1, 1]),
            'limb_wing': np.zeros(9, dtype=bool),
            'joints_names': [f'j{i}' for i in range(9)],
        }

    def test_unbranched_chain_runs_to_its_leaf(self):
        self.assertEqual(np.flatnonzero(_limb_tip_mask(self._cond())).tolist(), [4, 8])

    def test_dropped_toe_falls_back_to_the_foot(self):
        _keep, dropped = drop_joints_from_cond(self._cond(), [4])
        self.assertEqual(np.flatnonzero(_limb_tip_mask(dropped)).tolist(), [3, 7])

    def test_tip_runs_from_the_knee_to_the_leaf(self):
        cond = self._cond()
        cond['limb_chain_id'] = np.array([-1, -1, 0, 0, 0, -1, 1, 1, 1])
        self.assertEqual(np.flatnonzero(_limb_tip_mask(cond)).tolist(), [4, 8])

    def test_foot_that_splits_into_toes_is_the_tip(self):
        # ankle -> foot -> two toes (each two joints long); the second foot has one toe
        parents = [-1, 0, 1, 2, 3, 2, 5, 0, 7, 8, 9]
        chains = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10]]
        self.assertEqual(chain_tips(parents, chains), [2, 10])

    def test_limb_chains_keep_hands_that_do_not_stand(self):
        # root -> hand -> finger, root -> foot -> toe
        parents = [-1, 0, 1, 0, 3]
        parts = ['trunk', 'hand', 'hand', 'foot', 'foot']
        self.assertEqual(limb_end_chains(parents, parts), [[1, 2], [3, 4]])
        self.assertEqual(foot_chains(parents, parts), [[3, 4]])

    def test_wing_tip_is_its_span_end_past_the_feathers(self):
        # shoulder -> upper arm -> forearm -> hand, a feather off the upper arm
        parents = [-1, 0, 1, 2, 3, 2]
        offsets = np.array([[0, 1, 0], [0.1, 0, 0], [0.3, 0, 0], [0.3, 0, 0], [0.2, 0, 0], [0, -0.1, -0.2]])
        wing = [1, 2, 3, 4, 5]
        self.assertEqual(chain_tips(parents, [wing]), [2])
        reach = bone_path_lengths(parents, offsets)
        self.assertEqual(limb_tips(parents, [wing, [0]], [True, False], reach), [4, 0])

    def test_wing_tip_holds_when_the_wing_folds(self):
        # The hand folds back onto the shoulder: in rest the forearm is the
        # farthest joint, along the bones the hand still is.
        parents = [-1, 0, 1, 2, 3]
        folded = np.array([[0, 1, 0], [0.1, 0, 0], [0.3, 0, 0], [-0.25, 0, 0.05], [-0.1, 0, 0]])
        reach = bone_path_lengths(parents, folded)
        self.assertEqual(limb_tips(parents, [[1, 2, 3, 4], [0]], [True, False], reach), [4, 0])

    def test_single_limb_gets_no_tips(self):
        self.assertEqual(limb_tips([-1, 0, 1], [[1, 2]], [False], np.zeros(3)), [])

    def test_midline_wing_root_splits_into_left_and_right_wings(self):
        # spine -> wing root (center) -> left wing, right wing
        parents = [-1, 0, 1, 2, 1, 4]
        parts = ['trunk', 'wing', 'wing', 'wing', 'wing', 'wing']
        sides = ['center', 'center', 'left', 'left', 'right', 'right']
        self.assertEqual(limb_end_chains(parents, parts), [[1, 2, 3, 4, 5]])
        self.assertEqual(limb_end_chains(parents, parts, sides), [[2, 3], [4, 5]])

    def test_fluttering_species_keeps_legs_only(self):
        entry = {'parents': np.array([-1, 0, 0]), 'joints_names': ['root', 'wing_L', 'leg_L'],
                 'joint_side_labels': ['center', 'left', 'left']}
        prefill = {'root': {'part': 'trunk'}, 'wing_L': {'part': 'wing'}, 'leg_L': {'part': 'foot'}}
        with patch.object(joint_parts, 'prefill_joint_parts', return_value=prefill):
            self.assertEqual(joint_parts.prefill_limb_end_chains(entry), ([[1], [2]], [True, False]))
            entry['species_tags'] = ('Winged', joint_parts.FLUTTERING_TAG)
            self.assertEqual(joint_parts.prefill_limb_end_chains(entry), ([[2]], [False]))

    def test_single_limb_flags_none(self):
        cond = self._cond()
        cond['limb_chain_id'] = np.array([-1, -1, -1, 0, 0, -1, -1, -1, -1])
        self.assertFalse(_limb_tip_mask(cond).any())


if __name__ == '__main__':
    unittest.main()
