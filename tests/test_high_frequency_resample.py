"""High-frequency clips: inverse window resample and the never-downsample loader rule."""
import random
import types

import numpy as np
import pytest

from data_loaders.truebones.data.dataset import (
    MotionDataset,
    _circular_roll_motion,
    _tile_loop_motion,
    invert_resample_motion_features,
    resample_motion_features,
    resample_time_scaled_window,
    time_scale_motion_features,
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
    np.testing.assert_allclose(recovered, clip, atol=1e-4)


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


@pytest.mark.parametrize('length', [20, 26, 45, 59, 90, 151])
def test_slow_only_never_shortens_nor_slows_past_the_window(length):
    random.seed(0)
    sampler = _sampler()
    draws = {
        MotionDataset._sample_motion_speed_target_length(
            sampler, length, WINDOW, slow_only=True
        )
        for _ in range(500)
    }
    if length > WINDOW:
        # Over the window: left at the recorded tempo, cropped later.
        assert draws == {length}
        return
    assert min(draws) >= length
    assert max(draws) <= WINDOW
    assert len(draws) > 1 or length == WINDOW


# ── One interpolation for time-scale + shift + window resample ──

def _reference_pose_window(clip, scaled, window_frames, *, source_periodic, window_periodic, shift):
    """Pose channels by np.interp on the source's piecewise-linear curve, at the
    composed times computed in floating point."""
    source = clip.shape[0]
    t = np.arange(WINDOW, dtype=np.float64)
    if window_periodic:
        u = t * window_frames / WINDOW + shift
    else:
        u = t * (window_frames - 1) / (WINDOW - 1) + shift
    if source_periodic:
        s = (u % scaled) * source / scaled
        grid = np.arange(source + 1)
        curve = np.concatenate([clip, clip[:1]], axis=0)
    else:
        s = u * (source - 1) / (scaled - 1)
        grid = np.arange(source)
        curve = clip
    flat = curve[:, :, :9].reshape(curve.shape[0], -1)
    out = np.stack([np.interp(s, grid, flat[:, k]) for k in range(flat.shape[1])], axis=1)
    return out.reshape(WINDOW, clip.shape[1], 9)


@pytest.mark.parametrize('source,scaled,tiles,shift', [
    (26, 31, 1, 0), (26, 31, 1, 7), (20, 23, 2, 5), (31, 24, 2, 11), (35, 41, 1, 40),
])
def test_composed_loop_window_matches_the_continuous_curve(source, scaled, tiles, shift):
    clip = _clip(source)
    out = resample_time_scaled_window(
        clip, scaled, scaled * tiles, WINDOW,
        source_periodic=True, window_periodic=True, shift=shift,
    )
    ref = _reference_pose_window(
        clip, scaled, scaled * tiles, source_periodic=True, window_periodic=True, shift=shift,
    )
    np.testing.assert_allclose(out[:, :, :9], ref, atol=1e-5)


@pytest.mark.parametrize('source,scaled,window,shift,source_periodic', [
    (51, 66, 60, 3, False), (40, 33, 30, 2, False), (45, 52, 120, 9, True),
])
def test_composed_open_window_matches_the_continuous_curve(source, scaled, window, shift, source_periodic):
    # An open window: a non-loop clip's crop, or a loop downgraded by its crop.
    clip = _clip(source)
    out = resample_time_scaled_window(
        clip, scaled, window, WINDOW,
        source_periodic=source_periodic, window_periodic=False, shift=shift,
    )
    ref = _reference_pose_window(
        clip, scaled, window, source_periodic=source_periodic, window_periodic=False, shift=shift,
    )
    np.testing.assert_allclose(out[:, :, :9], ref, atol=1e-5)


@pytest.mark.parametrize('periodic', [False, True])
def test_composed_window_equals_two_steps_when_the_speed_step_is_exact(periodic):
    # Doubling a periodic clip (or (L-1)*2+1 an open one) puts every source key
    # on a scaled frame, so the second interpolation cuts no corner and the two
    # pipelines must agree on every channel -- roll and tiles included.
    source = 26
    clip = _clip(source)
    scaled = 2 * source if periodic else 2 * (source - 1) + 1
    two_step, _speed = time_scale_motion_features(clip, scaled, periodic=periodic)
    if periodic:
        shift, tiles = 9, 2
        two_step = _tile_loop_motion(_circular_roll_motion(two_step, shift), tiles)
        window = scaled * tiles
    else:
        shift, window = 4, scaled - 6
        two_step = two_step[shift:shift + window]
    two_step = resample_motion_features(two_step, WINDOW, periodic=periodic)
    composed = resample_time_scaled_window(
        clip, scaled, window, WINDOW, source_periodic=periodic, window_periodic=periodic, shift=shift,
    )
    np.testing.assert_allclose(composed, two_step, atol=1e-5)


def test_composed_window_keeps_a_wing_beat_the_two_step_pipeline_flattens():
    # A period-4 triangle beat, slowed 1.15x into a single-cycle window: both
    # steps upsample, yet two interpolations in a row cut the peaks.
    source = 26
    beat = np.minimum(np.arange(source) % 4, 4 - np.arange(source) % 4) * 22.5
    clip = np.zeros((source, 1, 12))
    clip[:, 0, 3] = beat
    scaled = 30
    composed = resample_time_scaled_window(
        clip, scaled, scaled, WINDOW, source_periodic=True, window_periodic=True,
    )[:, 0, 3]
    two_step = resample_motion_features(
        time_scale_motion_features(clip, scaled, periodic=True)[0], WINDOW, periodic=True,
    )[:, 0, 3]
    direct = resample_motion_features(clip, WINDOW, periodic=True)[:, 0, 3]
    # A loop played slower into the same window is the same content.
    np.testing.assert_allclose(composed, direct, atol=1e-5)
    # The two-step window misses individual peaks by several degrees.
    assert np.abs(two_step - composed).max() > 3.0


def test_composed_window_rejects_an_open_crop_past_the_clip():
    with pytest.raises(ValueError):
        resample_time_scaled_window(
            _clip(30), 40, 30, WINDOW, source_periodic=False, window_periodic=False, shift=11,
        )
