"""Precision policy from docs/fp16_vs_bf16_precision.md (B + C + D).

B: the fp16 GradScaler's loss scale is capped, so it stops probing the overflow
   wall and manufacturing skipped steps.
C: a loss-scale overflow is classified apart from a real gradient spike.
D: the broadcast conditioning heads run in fp32 (largest weight gradients in the
   network, hence the first to overflow).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from diffusion.fp16_util import GRAD_SCALER_MAX_SCALE, MixedPrecisionTrainer  # noqa: E402
from model.motion_diffusion import run_in_fp32  # noqa: E402
from data_loaders.truebones.truebones_utils.joint_struct_features import JOINT_STRUCT_DIM  # noqa: E402
from train.training_loop import AMP_OVERFLOW_MAX_DUMPS, classify_grad_event  # noqa: E402


class LossScaleCapTests(unittest.TestCase):
    """B: the scaler is not allowed to climb into the overflow wall."""

    class _FakeScaler:
        def __init__(self, scale):
            self._scale = float(scale)
            self.update_calls = []

        def is_enabled(self):
            return True

        def get_scale(self):
            return self._scale

        def _get_scale_async(self):
            return torch.tensor(self._scale)

        def update(self, new_scale=None):
            if torch.is_tensor(new_scale):
                new_scale = float(new_scale)
            self.update_calls.append(new_scale)
            if new_scale is not None:
                self._scale = float(new_scale)

    def _trainer(self, scale):
        model = nn.Linear(3, 3)
        trainer = MixedPrecisionTrainer(model=model, amp_enabled=False, log_norms=False)
        trainer.scaler = self._FakeScaler(scale)
        return trainer

    def test_scale_above_the_cap_is_pulled_back(self):
        trainer = self._trainer(GRAD_SCALER_MAX_SCALE * 8)
        trainer._cap_loss_scale()
        self.assertEqual(trainer.scaler.get_scale(), float(GRAD_SCALER_MAX_SCALE))
        self.assertEqual(trainer.scaler.update_calls, [float(GRAD_SCALER_MAX_SCALE)])

    def test_scale_below_the_cap_is_left_alone(self):
        trainer = self._trainer(GRAD_SCALER_MAX_SCALE / 4)
        trainer._cap_loss_scale()
        self.assertEqual(trainer.scaler.get_scale(), GRAD_SCALER_MAX_SCALE / 4)

    def test_real_scaler_is_capped_on_device(self):
        model = nn.Linear(3, 3)
        trainer = MixedPrecisionTrainer(model=model, amp_enabled=False, log_norms=False)
        trainer.scaler = torch.amp.GradScaler('cpu', init_scale=GRAD_SCALER_MAX_SCALE * 8)
        trainer.scaler.scale(torch.ones(()))  # creates the scale tensor
        trainer._cap_loss_scale()
        self.assertEqual(trainer.scaler.get_scale(), float(GRAD_SCALER_MAX_SCALE))

    def test_disabled_scaler_is_untouched(self):
        model = nn.Linear(3, 3)
        trainer = MixedPrecisionTrainer(model=model, amp_enabled=False, log_norms=False)
        self.assertFalse(trainer.scaler.is_enabled())
        trainer._cap_loss_scale()  # must not raise on the fp32 / bf16 paths

    def test_cap_sits_inside_the_measured_flat_window(self):
        # The measured window is 2^14..2^22 flat, wall at 2^23 on a typical
        # batch; the cap has to leave room for outlier batches.
        self.assertGreaterEqual(GRAD_SCALER_MAX_SCALE, 2 ** 14)
        self.assertLessEqual(GRAD_SCALER_MAX_SCALE, 2 ** 18)

    def test_amp_trainer_caps_loss_scale(self):
        model = nn.Linear(3, 3)
        trainer = MixedPrecisionTrainer(model=model, amp_enabled=False, log_norms=True)
        trainer.scaler = self._FakeScaler(GRAD_SCALER_MAX_SCALE * 4)
        trainer._cap_loss_scale()
        self.assertEqual(trainer.scaler.get_scale(), float(GRAD_SCALER_MAX_SCALE))


class GradEventClassificationTests(unittest.TestCase):
    """C: an overflow is not a spike."""

    def test_non_finite_with_scaler_is_an_overflow(self):
        self.assertEqual(classify_grad_event(float('inf'), True, 50.0), 'overflow')
        self.assertEqual(classify_grad_event(float('nan'), True, 50.0), 'overflow')

    def test_non_finite_without_scaler_is_a_real_failure(self):
        # bf16 / fp32 have no loss scale, so non-finite gradients there are not
        # routine and must still be dumped as spikes.
        self.assertEqual(classify_grad_event(float('inf'), False, 50.0), 'spike')

    def test_large_finite_norm_is_a_spike_either_way(self):
        self.assertEqual(classify_grad_event(120.0, True, 50.0), 'spike')
        self.assertEqual(classify_grad_event(120.0, False, 50.0), 'spike')

    def test_ordinary_step_is_not_an_event(self):
        self.assertIsNone(classify_grad_event(0.8, True, 50.0))
        self.assertIsNone(classify_grad_event(None, True, 50.0))

    def test_overflow_dump_budget_is_small_and_separate(self):
        self.assertGreaterEqual(AMP_OVERFLOW_MAX_DUMPS, 1)
        self.assertLessEqual(AMP_OVERFLOW_MAX_DUMPS, 3)


class ConditioningHeadFp32Tests(unittest.TestCase):
    """D: the broadcast conditioning heads keep their fp32 weight gradients."""

    def test_run_in_fp32_matches_the_plain_fp32_call(self):
        torch.manual_seed(0)
        head = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 8))
        x = torch.randn(4, 8)
        expected = head(x)
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            actual = run_in_fp32(head, x)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.equal(actual, expected))

    def test_run_in_fp32_upcasts_a_reduced_precision_input(self):
        torch.manual_seed(0)
        head = nn.Linear(8, 8)
        x = torch.randn(4, 8)
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            actual = run_in_fp32(head, x.to(torch.bfloat16))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.equal(actual, head(x.to(torch.bfloat16).float())))

    def test_autocast_without_the_helper_would_downcast(self):
        # Guards the premise: these heads really are inside an autocast region.
        torch.manual_seed(0)
        head = nn.Linear(8, 8)
        x = torch.randn(4, 8)
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            self.assertEqual(head(x).dtype, torch.bfloat16)


if __name__ == '__main__':
    unittest.main()
