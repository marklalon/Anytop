"""High-frequency clips: inverse window resample and the never-downsample loader rule."""
import random
import types

import numpy as np
import pytest

from data_loaders.truebones.data.dataset import (
    MotionDataset,
    invert_resample_motion_features,
    resample_motion_features,
)
from data_loaders.truebones.truebones_utils.high_frequency import (
    HIGH_FREQUENCY_SCORE_THRESHOLD,
    high_frequency_score,
    is_high_frequency,
)
from sample.output_lengths import _resample_window_to_output

WINDOW = 60


def _clip(frames, joints=3, seed=0):
    rng = np.random.default_rng(seed)
    clip = rng.normal(size=(frames, joints, 12)).astype(np.float32)
    # A period-2 swing on one rotation channel: the worst case for interpolation.
    clip[:, 1, 3] += np.where(np.arange(frames) % 2 == 0, 1.0, -1.0).astype(np.float32)
    return clip


@pytest.mark.parametrize('periodic', [False, True])
@pytest.mark.parametrize('frames', [20, 26, 31, 51, 59])
def test_inverse_recovers_every_channel(frames, periodic):
    clip = _clip(frames)
    window = resample_motion_features(clip, WINDOW, periodic=periodic)
    recovered = invert_resample_motion_features(window, frames, periodic=periodic)
    assert recovered.shape == clip.shape
    np.testing.assert_allclose(recovered[:, :, :9], clip[:, :, :9], atol=1e-4)
    # An open clip's terminal velocity row is not a step: the forward resample
    # never reads it, and the inverse repeats the last step there.
    visible = slice(None) if periodic else slice(0, -1)
    np.testing.assert_allclose(recovered[visible, :, 9:12], clip[visible, :, 9:12], atol=1e-4)
    if not periodic:
        np.testing.assert_allclose(recovered[-1, :, 9:12], recovered[-2, :, 9:12])


def test_inverse_beats_linear_on_a_wing_beat():
    beat = np.zeros((26, 1, 12), dtype=np.float32)
    beat[:, 0, 3] = np.minimum(np.arange(26) % 4, 4 - np.arange(26) % 4)
    window = resample_motion_features(beat, WINDOW, periodic=True)
    linear = resample_motion_features(window, 26, periodic=True)
    inverse = invert_resample_motion_features(window, 26, periodic=True)
    assert np.abs(linear - beat).max() > 0.1
    np.testing.assert_allclose(inverse, beat, atol=1e-5)


def test_window_to_output_inverts_only_when_shorter():
    clip = _clip(26)
    window = resample_motion_features(clip, WINDOW, periodic=True)
    np.testing.assert_allclose(
        _resample_window_to_output(window, 26, WINDOW, True)[:, :, :9], clip[:, :, :9], atol=1e-4
    )
    longer = _resample_window_to_output(window, 90, WINDOW, True)
    np.testing.assert_array_equal(longer, resample_motion_features(window, 90, periodic=True))
    assert _resample_window_to_output(window, WINDOW, WINDOW, True) is window


def test_score_flags_a_frame_locked_beat_not_a_slow_swing():
    frames = np.arange(40)
    beat = np.zeros((40, 2, 12))
    beat[:, 1, 3] = 1.6 * (np.minimum(frames % 4, 4 - frames % 4) / 2.0 - 0.5)
    slow = np.zeros((40, 2, 12))
    slow[:, 1, 3] = 0.8 * np.sin(2 * np.pi * frames / 20.0)
    assert is_high_frequency(high_frequency_score(beat))
    assert high_frequency_score(slow) < HIGH_FREQUENCY_SCORE_THRESHOLD
    assert high_frequency_score(beat[:5]) == 0.0


@pytest.mark.parametrize('frames', [40, 41])
@pytest.mark.parametrize('period', [2, 3, 4])
def test_score_is_the_rms_of_a_pure_high_frequency_signal(frames, period):
    # Covers the Nyquist bin of an even-length clip (period 2, 40 frames),
    # which has no mirror and must not be counted twice.
    t = np.arange(frames)
    signal = np.where(t % 2 == 0, 1.0, -1.0) if period == 2 else np.sin(2 * np.pi * t / period)
    motion = np.zeros((frames, 1, 12))
    motion[:, 0, 3] = signal
    rms = np.sqrt(((signal - signal.mean()) ** 2).mean())
    assert high_frequency_score(motion) == pytest.approx(rms, rel=0.02)


def _sampler(ratio=1.3):
    return types.SimpleNamespace(
        opt=types.SimpleNamespace(motion_speed_aug=ratio, motion_speed_aug_prob=1.0),
        min_length=20,
    )


@pytest.mark.parametrize('length,is_loop', [(20, True), (26, False), (45, True), (59, False), (90, True), (151, False)])
def test_slow_only_never_shortens_nor_slows_past_the_window(length, is_loop):
    random.seed(0)
    sampler = _sampler()
    draws = {
        MotionDataset._sample_motion_speed_target_length(
            sampler, length, is_loop, WINDOW, slow_only=True
        )
        for _ in range(500)
    }
    assert min(draws) >= length
    if length <= WINDOW:
        assert max(draws) <= WINDOW
    assert len(draws) > 1 or length >= WINDOW


def test_slow_only_excludes_fit_budget():
    with pytest.raises(ValueError):
        MotionDataset._sample_motion_speed_target_length(
            _sampler(), 30, False, WINDOW, fit_budget=True, slow_only=True
        )
