import os
import sys
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion.fp16_util import MixedPrecisionTrainer, inspect_optimizer_state, sanitize_optimizer_state
from model.motion_transformer import GraphMultiHeadAttention, SelectiveMultiheadAttention


class MixedPrecisionTrainerTests(unittest.TestCase):
    def test_sanitize_optimizer_state_clears_nonfinite_entries(self):
        model = torch.nn.Linear(4, 2)
        opt = AdamW(model.parameters(), lr=1e-3)

        data = torch.randn(3, 4)
        target = torch.randn(3, 2)
        loss = torch.nn.functional.mse_loss(model(data), target)
        loss.backward()
        opt.step()

        first_state = next(iter(opt.state.values()))
        first_state["exp_avg_sq"].view(-1)[0] = float("inf")
        first_state["exp_avg"].view(-1)[0] = float("nan")

        stats = sanitize_optimizer_state(opt)

        self.assertTrue(stats["found"])
        self.assertGreater(stats["inf"], 0)
        self.assertGreater(stats["nan"], 0)
        for state in opt.state.values():
            for value in state.values():
                if torch.is_tensor(value) and torch.is_floating_point(value):
                    self.assertTrue(torch.isfinite(value).all())

    def test_inspect_optimizer_state_reports_nonfinite_entries_without_mutation(self):
        model = torch.nn.Linear(4, 2)
        opt = AdamW(model.parameters(), lr=1e-3)

        data = torch.randn(3, 4)
        target = torch.randn(3, 2)
        loss = torch.nn.functional.mse_loss(model(data), target)
        loss.backward()
        opt.step()

        first_state = next(iter(opt.state.values()))
        first_state["exp_avg_sq"].view(-1)[0] = float("inf")

        stats = inspect_optimizer_state(opt)

        self.assertTrue(stats["found"])
        self.assertGreater(stats["inf"], 0)
        self.assertTrue(torch.isinf(first_state["exp_avg_sq"]).any())

    def test_optimize_amp_skips_nonfinite_gradients(self):
        model = torch.nn.Linear(4, 2)
        trainer = MixedPrecisionTrainer(
            model=model,
            use_fp16=False,
            amp_dtype="bf16",
            amp_enabled=True,
            device_type="cpu",
            log_norms=False,
        )
        opt = AdamW(trainer.master_params, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)

        trainer.zero_grad()
        model.weight.grad = torch.full_like(model.weight, float("inf"))
        model.bias.grad = torch.zeros_like(model.bias)
        before_weight = model.weight.detach().clone()

        took_step = trainer.optimize(opt, scheduler)

        self.assertFalse(took_step)
        self.assertTrue(torch.equal(before_weight, model.weight.detach()))
        self.assertEqual(len(opt.state), 0)
        self.assertEqual(scheduler.last_epoch, 0)

    def test_optimize_normal_raises_on_large_finite_gradients(self):
        model = torch.nn.Linear(4, 2)
        trainer = MixedPrecisionTrainer(
            model=model,
            use_fp16=False,
            amp_enabled=False,
            log_norms=False,
        )
        opt = AdamW(trainer.master_params, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)

        trainer.zero_grad()
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 4e11)
        before_params = [parameter.detach().clone() for parameter in model.parameters()]

        with self.assertRaisesRegex(RuntimeError, "Detected abnormal finite grad_norm"):
            trainer.optimize(opt, scheduler)

        self.assertEqual(scheduler.last_epoch, 0)
        self.assertEqual(len(opt.state), 0)
        for before_param, parameter in zip(before_params, model.parameters()):
            self.assertTrue(torch.equal(before_param, parameter.detach()))

    def test_optimize_amp_raises_on_large_finite_gradients(self):
        model = torch.nn.Linear(4, 2)
        trainer = MixedPrecisionTrainer(
            model=model,
            use_fp16=False,
            amp_dtype="bf16",
            amp_enabled=True,
            device_type="cpu",
            log_norms=False,
        )
        opt = AdamW(trainer.master_params, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)

        trainer.zero_grad()
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 4e11)
        before_params = [parameter.detach().clone() for parameter in model.parameters()]

        with self.assertRaisesRegex(RuntimeError, "Detected abnormal finite grad_norm"):
            trainer.optimize(opt, scheduler)

        self.assertEqual(scheduler.last_epoch, 0)
        self.assertEqual(len(opt.state), 0)
        for before_param, parameter in zip(before_params, model.parameters()):
            self.assertTrue(torch.equal(before_param, parameter.detach()))

    def test_optimize_amp_finite_step_uses_clip_without_nonfinite_scan(self):
        model = torch.nn.Linear(4, 2)
        trainer = MixedPrecisionTrainer(
            model=model,
            use_fp16=False,
            amp_dtype="bf16",
            amp_enabled=True,
            device_type="cpu",
            log_norms=False,
        )
        opt = AdamW(trainer.master_params, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)

        trainer.zero_grad()
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        before_params = [parameter.detach().clone() for parameter in model.parameters()]
        original_clip_grad_norm = torch.nn.utils.clip_grad_norm_

        with patch(
            "diffusion.fp16_util.count_nonfinite_gradients",
            side_effect=AssertionError("finite step should not trigger non-finite scan"),
        ), patch(
            "diffusion.fp16_util.th.nn.utils.clip_grad_norm_",
            wraps=original_clip_grad_norm,
        ) as clip_grad_norm:
            took_step = trainer.optimize(opt, scheduler)

        self.assertTrue(took_step)
        clip_grad_norm.assert_called_once()
        self.assertEqual(scheduler.last_epoch, 1)
        self.assertEqual(len(opt.state), 2)
        self.assertTrue(
            any(
                not torch.equal(before_param, parameter.detach())
                for before_param, parameter in zip(before_params, model.parameters())
            )
        )

    def test_optimize_fp16_raises_on_large_finite_gradients_when_log_norms_disabled(self):
        model = _MinimalFP16Model()
        trainer = MixedPrecisionTrainer(
            model=model,
            use_fp16=True,
            amp_enabled=False,
            log_norms=False,
        )
        opt = AdamW(trainer.master_params, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)

        trainer.zero_grad()
        scaled_grad_value = 4e11 * (2 ** trainer.lg_loss_scale)
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, scaled_grad_value)
        before_params = [parameter.detach().clone() for parameter in model.parameters()]

        with self.assertRaisesRegex(RuntimeError, "Detected abnormal finite grad_norm"):
            trainer.optimize(opt, scheduler)

        self.assertEqual(scheduler.last_epoch, 0)
        self.assertEqual(len(opt.state), 0)
        for before_param, parameter in zip(before_params, model.parameters()):
            self.assertTrue(torch.equal(before_param, parameter.detach()))

    def test_optimize_fp16_skips_nonfinite_gradients_when_log_norms_disabled(self):
        model = _MinimalFP16Model()
        trainer = MixedPrecisionTrainer(
            model=model,
            use_fp16=True,
            amp_enabled=False,
            log_norms=False,
        )
        opt = AdamW(trainer.master_params, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)

        trainer.zero_grad()
        model.linear.weight.grad = torch.full_like(model.linear.weight, float("inf"))
        model.linear.bias.grad = torch.zeros_like(model.linear.bias)
        before_params = [parameter.detach().clone() for parameter in model.parameters()]

        took_step = trainer.optimize(opt, scheduler)

        self.assertFalse(took_step)
        self.assertEqual(scheduler.last_epoch, 0)
        self.assertEqual(len(opt.state), 0)
        self.assertLess(trainer.lg_loss_scale, 20.0)
        for before_param, parameter in zip(before_params, model.parameters()):
            self.assertTrue(torch.equal(before_param, parameter.detach()))


class _MinimalFP16Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, input):
        return self.linear(input)

    def convert_to_fp16(self):
        return None

if __name__ == "__main__":
    unittest.main()