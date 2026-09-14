import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.sample_loss_limit import SampleLossLimiter  # noqa: E402


def _warm(limiter, t_value, loss_value, samples=SampleLossLimiter.WARMUP_SAMPLES):
    t = torch.full((samples,), t_value, dtype=torch.long)
    limiter.weights(t, torch.full((samples,), loss_value))


def test_nothing_is_limited_before_a_timestep_warms_up():
    limiter = SampleLossLimiter(num_timesteps=10, ratio=8.0, device='cpu')
    t = torch.tensor([3, 3, 3])
    w = limiter.weights(t, torch.tensor([0.1, 1000.0, 5.0]))
    torch.testing.assert_close(w, torch.ones(3))


def test_weight_is_one_below_the_cap_and_sqrt_cap_over_loss_above_it():
    limiter = SampleLossLimiter(num_timesteps=10, ratio=8.0, device='cpu')
    _warm(limiter, 4, 0.5)
    assert math.exp(limiter.log_ref[4].item()) == pytest.approx(0.5)

    losses = torch.tensor([0.5, 3.9, 400.0])
    w = limiter.weights(torch.tensor([4, 4, 4]), losses)
    cap = 8.0 * 0.5
    torch.testing.assert_close(w[:2], torch.ones(2))
    assert w[2].item() == pytest.approx(math.sqrt(cap / 400.0), rel=1e-5)


def test_caps_are_per_timestep():
    limiter = SampleLossLimiter(num_timesteps=100, ratio=8.0, device='cpu')
    _warm(limiter, 1, 0.005)
    _warm(limiter, 99, 0.5)
    # 0.3 is ordinary at t=99 but 60x the reference at t=1.
    w = limiter.weights(torch.tensor([1, 99]), torch.tensor([0.3, 0.3]))
    assert w[0].item() < 1.0
    assert w[1].item() == 1.0


def test_limited_sample_gradient_stops_growing_at_the_cap():
    """The point of the limiter: past the cap a sample's gradient norm is that of a
    sample sitting on the cap, however far out its target is."""
    cap = 4.0

    def grad_norm(residual_scale):
        # A fresh limiter per call: weights() also updates the reference.
        limiter = SampleLossLimiter(num_timesteps=4, ratio=4.0, device='cpu')
        _warm(limiter, 2, 1.0)
        # l = mean(residual^2) with a uniform residual, so l = residual_scale^2.
        pred = torch.zeros(16, requires_grad=True)
        target = torch.full((16,), residual_scale)
        l_simple = ((pred - target) ** 2).mean().reshape(1)
        w = limiter.weights(torch.tensor([2]), l_simple)
        loss = l_simple + (w - 1.0) * l_simple
        loss.sum().backward()
        return pred.grad.norm().item()

    at_cap = grad_norm(math.sqrt(cap))
    assert grad_norm(math.sqrt(cap) * 10.0) == pytest.approx(at_cap, rel=1e-4)
    assert grad_norm(math.sqrt(cap) * 1000.0) == pytest.approx(at_cap, rel=1e-4)
    # Below the cap the gradient is untouched.
    assert grad_norm(1.0) == pytest.approx(at_cap / 2.0, rel=1e-4)


def test_rare_outliers_barely_move_the_reference():
    """A capped sample enters the reference at the cap, so a 2% stream of 1e6-scale
    outliers shifts it by about p*log(ratio)/(1-p) instead of dragging it toward
    log(1e6). A sustained shift of the whole distribution still moves it, at most
    rate*log(ratio) per sample."""
    limiter = SampleLossLimiter(num_timesteps=4, ratio=8.0, device='cpu')
    _warm(limiter, 0, 1.0)
    for step in range(2000):
        loss = 1e6 if step % 50 == 0 else 1.0
        limiter.weights(torch.tensor([0]), torch.tensor([loss]))
    assert limiter.log_ref[0].item() < 0.2

    shifted = SampleLossLimiter(num_timesteps=4, ratio=8.0, device='cpu')
    _warm(shifted, 0, 1.0)
    before = shifted.log_ref[0].item()
    shifted.weights(torch.tensor([0]), torch.tensor([1e6]))
    assert shifted.log_ref[0].item() - before <= (1.0 / 21.0) * math.log(8.0) + 1e-6


def test_duplicate_timesteps_in_one_batch_update_as_a_mean():
    limiter = SampleLossLimiter(num_timesteps=4, ratio=8.0, device='cpu')
    limiter.weights(torch.tensor([1, 1, 2]), torch.tensor([1.0, 100.0, 3.0]))
    assert limiter.count.tolist() == [0.0, 2.0, 1.0, 0.0]
    assert limiter.log_ref[1].item() == pytest.approx(math.log(10.0), rel=1e-5)
    assert limiter.log_ref[2].item() == pytest.approx(math.log(3.0), rel=1e-5)
    assert limiter.log_ref[0].item() == 0.0


def test_state_dict_roundtrip_and_timestep_mismatch():
    limiter = SampleLossLimiter(num_timesteps=10, ratio=8.0, device='cpu')
    _warm(limiter, 7, 0.25)
    restored = SampleLossLimiter(num_timesteps=10, ratio=8.0, device='cpu')
    assert restored.load_state_dict(limiter.state_dict())
    torch.testing.assert_close(restored.log_ref, limiter.log_ref)
    torch.testing.assert_close(restored.count, limiter.count)
    assert not SampleLossLimiter(num_timesteps=20, ratio=8.0, device='cpu').load_state_dict(
        limiter.state_dict()
    )


def test_ratio_must_exceed_one():
    with pytest.raises(ValueError):
        SampleLossLimiter(num_timesteps=10, ratio=1.0, device='cpu')
