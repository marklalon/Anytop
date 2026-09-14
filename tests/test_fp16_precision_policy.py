"""Precision policy from docs/fp16_vs_bf16_precision.md (A + B + C + D).

A: the cross-K sub-path is an fp32 island (it lives behind a zero-init gate, so
   its gradients underflow fp16 while that gate is closed).
B: the fp16 GradScaler's loss scale is capped, so it stops probing the overflow
   wall and manufacturing skipped steps.
C: a loss-scale overflow is classified apart from a real gradient spike.
D: the broadcast conditioning heads run in fp32 (largest weight gradients in the
   network, hence the first to overflow).
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from diffusion import fp16_util  # noqa: E402
from diffusion.fp16_util import GRAD_SCALER_MAX_SCALE, MixedPrecisionTrainer  # noqa: E402
from model.anytop import run_in_fp32  # noqa: E402
from model.motion_transformer import CrossLimbTemporalBlock  # noqa: E402
from train.training_loop import AMP_OVERFLOW_MAX_DUMPS, classify_grad_event  # noqa: E402


class CrossKFp32IslandTests(unittest.TestCase):
    """A: cross-K stays fp32 no matter what autocast the trunk runs in."""

    def _block(self):
        torch.manual_seed(0)
        block = CrossLimbTemporalBlock(d_model=32, nhead=4, num_latents=4,
                                       dropout=0.0, latent_width=16)
        block.eval()
        return block

    def _run_and_record(self, block, dtype):
        seen = {}

        def record(name):
            def hook(module, args, output):
                tensor = output[0] if isinstance(output, tuple) else output
                seen[name] = tensor.dtype
            return hook

        handles = [
            block.cross_k_norm.register_forward_hook(record('cross_k_norm')),
            block.cross_k_attn.register_forward_hook(record('cross_k_attn')),
            block.proj_in.register_forward_hook(record('proj_in')),
        ]
        x = torch.randn(3, 2, 5, 32)
        mask = torch.zeros(2, 5, dtype=torch.bool)
        try:
            with torch.autocast(device_type='cpu', dtype=dtype):
                out = block(x, mask)
        finally:
            for handle in handles:
                handle.remove()
        return out, seen

    def test_cross_k_runs_in_fp32_under_autocast(self):
        block = self._block()
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                _, seen = self._run_and_record(block, dtype)
                self.assertEqual(seen['cross_k_norm'], torch.float32)
                self.assertEqual(seen['cross_k_attn'], torch.float32)

    def test_the_island_is_local_to_cross_k(self):
        # The rest of the block must stay in autocast: pinning the whole block
        # costs ~4% of a step, the island costs nothing.
        block = self._block()
        _, seen = self._run_and_record(block, torch.bfloat16)
        self.assertEqual(seen['proj_in'], torch.bfloat16)

    def test_no_autocast_is_unchanged(self):
        block = self._block()
        x = torch.randn(3, 2, 5, 32)
        mask = torch.zeros(2, 5, dtype=torch.bool)
        expected = block(x, mask)
        again = block(x, mask)
        self.assertTrue(torch.equal(expected, again))
        self.assertEqual(expected.dtype, torch.float32)


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

        def update(self, new_scale=None):
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
        self.assertEqual(trainer.scaler.update_calls, [])

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

    def test_amp_optimize_caps_and_logs(self):
        logged = {}
        model = nn.Linear(3, 3)
        trainer = MixedPrecisionTrainer(model=model, amp_enabled=False, log_norms=True)
        trainer.scaler = self._FakeScaler(GRAD_SCALER_MAX_SCALE * 4)
        original = fp16_util.logger.logkv_mean
        fp16_util.logger.logkv_mean = lambda key, value: logged.__setitem__(key, value)
        try:
            trainer._cap_loss_scale()
        finally:
            fp16_util.logger.logkv_mean = original
        self.assertEqual(trainer.scaler.get_scale(), float(GRAD_SCALER_MAX_SCALE))
        self.assertAlmostEqual(logged['loss_scale_log2'],
                               math.log2(GRAD_SCALER_MAX_SCALE))


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
