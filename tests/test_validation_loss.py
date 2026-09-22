"""TrainLoop.evaluate() is a plain val-loss pass.

It runs the same ``training_losses`` as a training step over the whole val
split, for the online model and the EMA copy, under seeded RNG streams that
reduce validation variance.  The synchronous fixtures here also verify that
their caller RNG states are restored; production background prefetch does not
promise bit-for-bit reproducibility.  Each clip is scored once per timestep
stratum (--val_t_strata), and both models see the same in-pass draws.
"""

import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ANYTOP_ROOT = Path(__file__).resolve().parents[1]
if str(ANYTOP_ROOT) not in sys.path:
    sys.path.insert(0, str(ANYTOP_ROOT))

from train.training_loop import TrainLoop  # noqa: E402


class _Model(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale


class _Diffusion:
    """``loss = scale * |x| * (1 + t)`` plus the noise handed in, so the loss
    depends on the model, the timestep draw and the torch RNG stream."""

    num_timesteps = 100

    def __init__(self):
        self.seen_training_flags = []
        self.calls = []  # (model.scale, t, noise) per training_losses call

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        self.seen_training_flags.append(model.training)
        assert noise is not None and noise.shape == x_start.shape
        self.calls.append((model.scale, t.clone(), noise.clone()))
        per_sample = model.scale * x_start.abs().flatten(1).mean(1) * (1 + t.float()) + noise.flatten(1).mean(1)
        return {"loss": per_sample, "l_simple": per_sample * 0.5, "note": "ignored"}


class _Platform:
    def __init__(self):
        self.scalars = []

    def report_scalar(self, name, value, iteration, group_name):
        self.scalars.append((group_name, name, value, iteration))


def _stub(eval_batches, use_ema, val_t_strata=4):
    stub = TrainLoop.__new__(TrainLoop)
    stub.args = SimpleNamespace(val_t_strata=val_t_strata)
    stub.device = torch.device("cpu")
    stub.non_blocking = False
    stub.amp_enabled = False
    stub.model = _Model(1.0)
    stub.model_avg = _Model(2.0) if use_ema else None
    stub.diffusion = _Diffusion()
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
    # Per clip and per stratum, the mean over clips and strata: recompute it
    # from the (t, noise) draws the pass actually made.
    expected = {}
    count = 0
    # Snapshot: the recomputation below goes through the same recording stub.
    for scale, t, noise in list(stub.diffusion.calls):
        motion = torch.full(noise.shape, 1.0 if noise.shape[0] == 4 else 3.0)
        prefix = "" if scale == 1.0 else "ema_"
        for key, value in stub.diffusion.training_losses(_Model(scale), motion, t, noise=noise).items():
            if torch.is_tensor(value):
                expected[prefix + key] = expected.get(prefix + key, 0.0) + float(value.mean()) * noise.shape[0]
        if prefix == "":
            count += noise.shape[0]
    for name, total in expected.items():
        assert abs(logged[name] - total / count) < 1e-5 * abs(total / count)
    # The EMA copy is a different model, so its loss is a different number.
    assert logged["ema_loss"] != logged["loss"]
    assert abs(logged["l_simple"] - 0.5 * logged["loss"]) < 1e-6


def test_evaluate_scores_each_clip_once_per_timestep_stratum_with_paired_draws():
    strata = 4
    stub = _stub(_batches(), use_ema=True, val_t_strata=strata)
    stub.evaluate()
    calls = stub.diffusion.calls
    # Per batch: `strata` calls for the online model, then the same for EMA.
    assert len(calls) == 2 * strata * len(_batches())
    width = _Diffusion.num_timesteps / strata
    for batch_start in range(0, len(calls), 2 * strata):
        live = calls[batch_start:batch_start + strata]
        ema = calls[batch_start + strata:batch_start + 2 * strata]
        assert all(scale == 1.0 for scale, _, _ in live)
        assert all(scale == 2.0 for scale, _, _ in ema)
        # Both models see identical (t, noise) draws: a paired comparison.
        for (_, t_live, n_live), (_, t_ema, n_ema) in zip(live, ema):
            assert torch.equal(t_live, t_ema) and torch.equal(n_live, n_ema)
        # Each clip lands in stratum k on pass k, at the same offset in every
        # stratum, and the noise differs between strata.
        offsets = (live[0][1].float() - 0) / width
        for k, (_, t, _) in enumerate(live):
            assert torch.all(t >= int(k * width)) and torch.all(t < int((k + 1) * width))
            assert torch.allclose((t.float() - k * width) / width, offsets, atol=1 / width + 1e-6)
        assert not torch.equal(live[0][2], live[1][2])


def test_single_stratum_is_a_plain_uniform_draw():
    stub = _stub(_batches(), use_ema=False, val_t_strata=1)
    stub.evaluate()
    assert len(stub.diffusion.calls) == len(_batches())
    for _, t, _ in stub.diffusion.calls:
        assert torch.all(t >= 0) and torch.all(t < _Diffusion.num_timesteps)


def test_evaluate_runs_in_eval_mode_and_restores_training_mode():
    stub = _stub(_batches(), use_ema=True)
    stub.model.train()
    stub.model_avg.eval()
    stub.evaluate()
    assert stub.diffusion.seen_training_flags and not any(stub.diffusion.seen_training_flags)
    assert stub.model.training is True
    assert stub.model_avg.training is False


def test_synchronous_evaluate_uses_fixed_draws_and_restores_caller_rng():
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

    # With this synchronous fixture, a second pass at a different caller RNG
    # position uses the same seeded draws. Background prefetch is deliberately
    # outside the scope of this unit test.
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
