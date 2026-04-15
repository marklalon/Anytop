import unittest

import torch
from torch.optim import AdamW

from diffusion.fp16_util import MixedPrecisionTrainer, inspect_optimizer_state, sanitize_optimizer_state


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


if __name__ == "__main__":
    unittest.main()