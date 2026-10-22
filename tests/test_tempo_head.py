"""Tempo head contract: integer-frame bins, two-hot target, the loss, and the
window <-> physical velocity units the output length converts between."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loaders.truebones.data.dataset import (  # noqa: E402
    resample_motion_features,
    resample_step_scale,
    velocity_to_source_units,
    velocity_to_window_units,
)
from diffusion.gaussian_diffusion import GaussianDiffusion, LossType, ModelMeanType, ModelVarType  # noqa: E402
from model.tempo_head import TempoHead, expected_frames, tempo_bin_frames, two_hot_targets  # noqa: E402
from sample.output_lengths import (  # noqa: E402
    _resample_window_to_output,
    predict_window_source_frames,
    window_to_physical,
)

WINDOW = 60


def test_bins_run_every_five_frames_from_min_length_to_the_budget():
    assert tempo_bin_frames(WINDOW, 20).tolist() == list(range(20, 121, 5))
    # A budget off the 5-frame grid is still the last bin.
    bins = tempo_bin_frames(WINDOW, 22).tolist()
    assert bins[:2] == [22, 27] and bins[-2:] == [117, 120]
    with pytest.raises(ValueError):
        tempo_bin_frames(WINDOW, 120)


def test_two_hot_target_is_exact_in_expectation_and_clamped():
    bins = torch.as_tensor(tempo_bin_frames(WINDOW, 20))
    frames = torch.tensor([20, 37, 63, 120, 7, 300])
    targets = two_hot_targets(frames, bins)
    assert torch.allclose(targets.sum(dim=-1), torch.ones(6))
    assert int((targets > 0).sum(dim=-1).max()) <= 2
    assert torch.allclose(targets @ bins, torch.tensor([20.0, 37.0, 63.0, 120.0, 20.0, 120.0]))


def test_expected_frames_reads_a_confident_bin():
    bins = torch.as_tensor(tempo_bin_frames(WINDOW, 20))
    logits = torch.full((1, bins.shape[0]), -50.0)
    logits[0, 3] = 50.0
    assert torch.allclose(expected_frames(logits, bins), bins[3:4])


def test_tempo_head_ignores_padded_joints_and_the_rest_token():
    torch.manual_seed(0)
    head = TempoHead(16, tempo_bin_frames(WINDOW, 20)).eval()
    hidden = torch.randn(WINDOW + 1, 2, 5, 16)
    valid = torch.tensor([[True, True, True, False, False], [True] * 5])
    logits = head(hidden, valid)
    moved = hidden.clone()
    moved[:, 0, 3:] += 100.0  # padded joints of sample 0
    moved[0] += 100.0  # the rest-pose token of every sample
    assert torch.allclose(head(moved, valid), logits, atol=1e-5)
    assert logits.shape == (2, 21) and logits.dtype == torch.float32


class _FakeModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.tempo_head = TempoHead(4, tempo_bin_frames(WINDOW, 20))
        self.logits = logits

    def forward(self, x, timesteps, y=None, return_tempo_logits=False, **_):
        assert return_tempo_logits and int(timesteps.max()) == 0
        return x, self.logits


def _diffusion(lambda_tempo=0.1):
    return GaussianDiffusion(
        betas=np.linspace(1e-4, 2e-2, 4),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
        lambda_tempo=lambda_tempo,
    )


def test_tempo_loss_is_cross_entropy_on_the_two_hot_target():
    bins = torch.as_tensor(tempo_bin_frames(WINDOW, 20))
    frames = torch.tensor([37, 90])
    target = two_hot_targets(frames, bins)
    exact = _FakeModel(torch.log(target + 1e-9))
    flat = _FakeModel(torch.zeros(2, bins.shape[0]))
    y = {'window_source_frames': frames}
    good = _diffusion().tempo_loss(exact, exact.logits, y)
    bad = _diffusion().tempo_loss(flat, flat.logits, y)
    entropy = -(target * torch.log(target.clamp_min(1e-12))).sum(dim=-1)
    assert torch.allclose(good['tempo_loss'], entropy, atol=1e-4)
    assert torch.all(bad['tempo_loss'] > good['tempo_loss'])
    assert torch.allclose(good['tempo_frames_error'], torch.zeros(2), atol=1e-3)
    with pytest.raises(KeyError):
        _diffusion().tempo_loss(exact, exact.logits, {})
    with pytest.raises(ValueError):
        _diffusion(lambda_tempo=-1.0)


def test_predict_window_source_frames_rounds_the_expected_count():
    bins = torch.as_tensor(tempo_bin_frames(WINDOW, 20))
    model = _FakeModel(torch.log(two_hot_targets(torch.tensor([37, 101]), bins) + 1e-9))
    frames = predict_window_source_frames(model, torch.zeros(2, 3, 12, WINDOW), {'y': {}})
    assert frames.tolist() == [37, 101]


def test_window_units_round_trip_touches_velocity_only():
    rng = np.random.default_rng(0)
    clip = rng.normal(size=(WINDOW, 3, 12)).astype(np.float32)
    window = velocity_to_window_units(clip, 1.5)
    np.testing.assert_array_equal(window[..., :9], clip[..., :9])
    np.testing.assert_allclose(window[..., 9:], clip[..., 9:] * 1.5, rtol=1e-6)
    np.testing.assert_allclose(velocity_to_source_units(window, 1.5), clip, rtol=1e-6)


def _walking_clip(frames, velocity_xz=(0.1, 0.05)):
    """A root walking at a constant physical XZ velocity per source frame."""
    clip = np.zeros((frames, 2, 12), dtype=np.float32)
    clip[:, :, 1] = 1.0
    clip[:, 1, 0] = 0.2
    clip[:, :, 3] = 1.0
    clip[:, :, 7] = 1.0
    clip[:, 0, 9] = velocity_xz[0]
    clip[:, 0, 11] = velocity_xz[1]
    return clip


@pytest.mark.parametrize('source_frames, loop', [(90, False), (37, False), (120, True), (25, True)])
def test_loader_window_comes_back_physical_at_the_source_length(source_frames, loop):
    clip = _walking_clip(source_frames)
    step = resample_step_scale(source_frames, WINDOW, periodic=loop)
    # The loader's window: resampled into T, velocity per window frame.
    window = velocity_to_window_units(resample_motion_features(clip, WINDOW, periodic=loop), step)
    np.testing.assert_allclose(window[:, 0, 9], 0.1 * step, rtol=1e-5)
    # The window's XZ path is its own, independent of the source length.
    np.testing.assert_allclose(window[:, 0, 9].sum(), 0.1 * step * WINDOW, rtol=1e-5)

    physical = window_to_physical(window, source_frames, loop)
    np.testing.assert_allclose(physical[:, 0, 9], 0.1, rtol=1e-5)
    np.testing.assert_allclose(physical[:, 0, 11], 0.05, rtol=1e-5)
    exported = _resample_window_to_output(physical, source_frames, WINDOW, loop)
    assert exported.shape[0] == source_frames
    np.testing.assert_allclose(exported[:, 0, 9], 0.1, rtol=1e-4)


def test_reference_bundle_hands_the_model_window_units():
    from data_loaders.truebones.truebones_utils.canonical_features import (
        canonical_to_physical_hml,
    )
    from sample.reference_motion import _prepare_img2img_reference_bundle

    n_joints = 2
    cond = {
        "parents": np.array([-1, 0]),
        "canonical_feature_mean": np.zeros(12, dtype=np.float32),
        "canonical_feature_std": np.ones(12, dtype=np.float32),
        "rest_pose": np.zeros((n_joints, 12), dtype=np.float32),
    }
    source_frames = 40
    bundle = _prepare_img2img_reference_bundle(
        None, "Walker", cond, max_joints=n_joints, target_feature_len=12, batch_size=1,
        requested_output_frame_count=WINDOW, preloaded_features=_walking_clip(source_frames),
    )
    step = resample_step_scale(source_frames, WINDOW)
    np.testing.assert_allclose(bundle["reference_physical"][:, 0, 9], 0.1 * step, rtol=1e-5)
    window = canonical_to_physical_hml(bundle["reference_motion"][0].permute(2, 0, 1).numpy(), cond)
    np.testing.assert_allclose(window[:, 0, 9], 0.1 * step, rtol=1e-5)
    assert bundle["reference_source_frame_count"] == source_frames
