"""TrainLoop.evaluate() is a plain val-loss pass.

It runs the same ``training_losses`` as a training step over the whole val
split, for the online model and the EMA copy, under fixed RNG streams so two
passes at the same weights report the same number, and it leaves the training
RNG streams exactly where it found them.
"""

import random
import sys
from pathlib import Path

import torch

ANYTOP_ROOT = Path(__file__).resolve().parents[1]
if str(ANYTOP_ROOT) not in sys.path:
    sys.path.insert(0, str(ANYTOP_ROOT))

from train.training_loop import TrainLoop, _fixed_validation_rng  # noqa: E402


class _Model(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale


class _Diffusion:
    """``loss = scale * |x| * (1 + t)`` plus the noise drawn inside, so the loss
    depends on the model, the timestep draw and the torch RNG stream."""

    def __init__(self):
        self.seen_training_flags = []

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        self.seen_training_flags.append(model.training)
        noise = torch.randn_like(x_start)
        per_sample = model.scale * x_start.abs().flatten(1).mean(1) * (1 + t.float()) + noise.flatten(1).mean(1)
        return {"loss": per_sample, "l_simple": per_sample * 0.5, "note": "ignored"}


class _Sampler:
    def sample(self, batch_size, device):
        t = torch.randint(0, 10, (batch_size,), device=device)
        return t, torch.ones(batch_size, device=device)


class _Platform:
    def __init__(self):
        self.scalars = []

    def report_scalar(self, name, value, iteration, group_name):
        self.scalars.append((group_name, name, value, iteration))


def _stub(eval_batches, use_ema):
    stub = TrainLoop.__new__(TrainLoop)
    stub.device = torch.device("cpu")
    stub.non_blocking = False
    stub.amp_enabled = False
    stub.model = _Model(1.0)
    stub.model_avg = _Model(2.0) if use_ema else None
    stub.diffusion = _Diffusion()
    stub.schedule_sampler = _Sampler()
    stub.train_platform = _Platform()
    stub.eval_data = eval_batches
    stub.step, stub.resume_step = 41, 0
    stub.HOST_ONLY_COND_KEYS = TrainLoop.HOST_ONLY_COND_KEYS
    return stub


def _batches():
    # Two batches of unequal size: the mean must be per clip, not per batch.
    return [
        (torch.full((4, 3, 2, 5), 1.0), {"y": {"n_joints": torch.tensor([3, 3, 3, 3])}}),
        (torch.full((2, 3, 2, 5), 3.0), {"y": {"n_joints": torch.tensor([3, 3])}}),
    ]


def test_evaluate_logs_per_clip_mean_for_online_and_ema_models():
    stub = _stub(_batches(), use_ema=True)
    stub.evaluate()

    logged = {name: value for group, name, value, _ in stub.train_platform.scalars if group == "Val"}
    assert set(logged) == {"loss", "l_simple", "ema_loss", "ema_l_simple"}
    assert all(it == 42 for _, _, _, it in stub.train_platform.scalars)
    # Under fixed RNG the pass is reproducible bit-for-bit.
    with _fixed_validation_rng(stub.device):
        expected = {}
        count = 0
        for motion, cond in _batches():
            for prefix, model in (("", stub.model), ("ema_", stub.model_avg)):
                for key, value in stub._validation_losses(model, motion, cond).items():
                    expected[prefix + key] = expected.get(prefix + key, 0.0) + float(value) * motion.shape[0]
            count += motion.shape[0]
    for name, total in expected.items():
        assert abs(logged[name] - total / count) < 1e-6
    # The EMA copy is a different model, so its loss is a different number.
    assert logged["ema_loss"] != logged["loss"]
    assert abs(logged["l_simple"] - 0.5 * logged["loss"]) < 1e-6


def test_evaluate_runs_in_eval_mode_and_restores_training_mode():
    stub = _stub(_batches(), use_ema=True)
    stub.model.train()
    stub.model_avg.eval()
    stub.evaluate()
    assert stub.diffusion.seen_training_flags and not any(stub.diffusion.seen_training_flags)
    assert stub.model.training is True
    assert stub.model_avg.training is False


def test_evaluate_is_deterministic_and_leaves_training_rng_untouched():
    stub = _stub(_batches(), use_ema=False)
    random.seed(7)
    torch.manual_seed(7)
    stub.evaluate()
    first = dict((n, v) for g, n, v, _ in stub.train_platform.scalars)
    after_py = random.random()
    after_torch = torch.rand(3)

    # Same seeds, no evaluate(): the training streams must be at the same point.
    random.seed(7)
    torch.manual_seed(7)
    assert random.random() == after_py
    assert torch.equal(torch.rand(3), after_torch)

    # A second pass at different training RNG positions reports the same losses.
    random.seed(99)
    torch.manual_seed(99)
    stub.train_platform.scalars.clear()
    stub.evaluate()
    second = dict((n, v) for g, n, v, _ in stub.train_platform.scalars)
    assert first == second


def test_evaluate_is_a_noop_without_a_val_loader():
    stub = _stub([], use_ema=False)
    stub.eval_data = None
    stub.evaluate()
    assert stub.train_platform.scalars == []
    assert stub._should_validate(1000) is False
