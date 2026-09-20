"""Root height: when the detrend reaches the vertical channel and when it does not.

The XZ detrend makes a gait run in place. A climb, a dive or a swim ascent is
the same transport pointing up, so the vertical channel takes the same operator
-- but behind its own, far looser threshold, because a clip's height is measured
from the floor rather than from an arbitrary origin and a hop's arc is content,
not travel.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion_lib.Animation import Animation, positions_global
from motion_lib.Quaternions import Quaternions

from data_loaders.truebones.truebones_utils.animation_utils import (
    ROOT_Y_DRIFT_THRESHOLD,
    _transport_carrier_index,
    flatten_root_y_drift,
    root_y_drift_correction,
    set_translation_root_y,
)
from data_loaders.truebones.truebones_utils.features import (
    extract_motion_features_from_aligned_anims,
)
from data_loaders.truebones.truebones_utils.motion_process import (
    recover_root_quat_and_pos_np,
)
from data_loaders.truebones.truebones_utils.param_utils import ROOT_Y_MIN_HEIGHT


def _anim(path_xyz: np.ndarray) -> Animation:
    """A two-joint skeleton whose root follows ``path_xyz``."""
    path_xyz = np.asarray(path_xyz, dtype=np.float64)
    n_frames = path_xyz.shape[0]
    parents = np.array([-1, 0], dtype=np.int64)
    offsets = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    rotations = Quaternions(
        np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_frames, len(parents), 1))
    )
    positions = np.zeros((n_frames, len(parents), 3), dtype=np.float64)
    positions[:, 0] = path_xyz
    positions[:, 1] = offsets[1]
    return Animation(rotations, positions, Quaternions.id(len(parents)), offsets, parents)


def _climb(n_frames: int, rise: float, bob: float = 0.0, cycle: int = 8) -> np.ndarray:
    """A steady ascent with an optional wingbeat bob riding on it."""
    t = np.arange(n_frames, dtype=np.float64)
    y = 1.0 + rise * t / (n_frames - 1) + bob * np.sin(2.0 * np.pi * t / cycle)
    return np.stack([np.zeros(n_frames), y, np.zeros(n_frames)], axis=-1)


def _hop(n_frames: int, height: float) -> np.ndarray:
    """Up and back down: net offset 0, peak ``height``."""
    t = np.linspace(0.0, np.pi, num=n_frames)
    y = 1.0 + height * np.sin(t)
    return np.stack([np.zeros(n_frames), y, np.zeros(n_frames)], axis=-1)


def _extract(anim, locomotion=True, clamp=False):
    (features, _mj, motion_anim, _export, _loop, xz_flattened,
     y_flattened) = extract_motion_features_from_aligned_anims(
        anim,
        anim,
        object_type='TestSkeleton',
        max_joints=8,
        orientation_quat=Quaternions.id(1).qs[0],
        translation_root_index=0,
        flatten_root_travel=locomotion,
        clamp_root_extent=clamp,
    )
    return features, xz_flattened, y_flattened, motion_anim


def _root_height(features, root_index=0):
    """The root height the decoder reads back out of a feature tensor."""
    _rot, r_pos = recover_root_quat_and_pos_np(
        features, translation_root_index=root_index
    )
    return np.asarray(r_pos[:, 1], dtype=np.float64)


# -- the operator ---------------------------------------------------------


def test_the_correction_is_the_ramp_between_the_endpoints():
    """Not a least-squares line: only the endpoints separate a climb from a bob."""
    y = np.array([1.0, 3.0, 1.2, 1.4, 2.0], dtype=np.float64)

    correction, drift = root_y_drift_correction(y)

    assert correction[0] == pytest.approx(0.0)
    assert correction[-1] == pytest.approx(1.0)
    assert np.allclose(correction, np.linspace(0.0, 1.0, num=5))
    assert drift == pytest.approx(1.0)


def test_flattening_keeps_frame_zero_and_returns_to_it():
    # A whole number of bob cycles, so the endpoints differ by the rise alone.
    y = _climb(41, rise=2.0, bob=0.1, cycle=8)[:, 1]

    flattened, drift = flatten_root_y_drift(y)

    assert drift == pytest.approx(2.0, abs=1e-9)
    assert flattened[0] == pytest.approx(y[0])
    assert flattened[-1] == pytest.approx(y[0])


def test_flattening_keeps_the_bob_the_climb_was_riding_on():
    """What is removed is the transport, not the cycle -- the same contract as XZ."""
    y = _climb(41, rise=2.0, bob=0.25, cycle=8)[:, 1]

    flattened, _drift = flatten_root_y_drift(y)

    assert float(flattened.max() - flattened.min()) == pytest.approx(0.5, abs=0.05)


def test_a_hop_nets_out_and_is_kept_whatever_its_height():
    """An out-and-back excursion has no transport, however far it reached."""
    y = _hop(40, height=3.0)[:, 1]

    flattened, drift = flatten_root_y_drift(y)

    assert drift == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(flattened, y)


def test_a_single_frame_track_is_a_no_op():
    flattened, drift = flatten_root_y_drift(np.array([1.5]))

    assert drift == 0.0
    assert np.allclose(flattened, [1.5])


# -- the gate -------------------------------------------------------------


def test_a_gait_that_climbs_past_the_threshold_is_flattened():
    rise = ROOT_Y_DRIFT_THRESHOLD * 3.0
    features, _xz, y_flattened, _anim_out = _extract(_anim(_climb(40, rise, bob=0.1)))

    assert y_flattened is True
    height = _root_height(features)
    assert height[-1] == pytest.approx(height[0], abs=1e-4)


def test_a_gait_that_climbs_inside_the_threshold_keeps_its_height_verbatim():
    """Below the threshold the vertical channel is bit-for-bit untouched."""
    rise = ROOT_Y_DRIFT_THRESHOLD * 0.5
    source = _anim(_climb(40, rise, bob=0.1))

    features, _xz, y_flattened, _anim_out = _extract(source)

    assert y_flattened is False
    source_height = positions_global(source)[:, 0, 1]
    assert np.allclose(_root_height(features), source_height, atol=1e-6)


def test_a_hop_is_kept_however_high_it_reaches():
    """The threshold is on net offset, so a jump that lands where it took off stays."""
    source = _anim(_hop(40, height=ROOT_Y_DRIFT_THRESHOLD * 5.0))

    features, _xz, y_flattened, _anim_out = _extract(source)

    assert y_flattened is False
    assert np.allclose(
        _root_height(features), positions_global(source)[:, 0, 1], atol=1e-6
    )


def test_a_clip_outside_the_policy_keeps_its_climb():
    """Stationary clips enter neither detrend; the gate is the caller's, as for XZ."""
    source = _anim(_climb(40, rise=ROOT_Y_DRIFT_THRESHOLD * 4.0))

    features, _xz, y_flattened, _anim_out = _extract(source, locomotion=False)

    assert y_flattened is False
    assert np.allclose(
        _root_height(features), positions_global(source)[:, 0, 1], atol=1e-6
    )


def test_the_two_channels_are_gated_separately():
    """A pure climb detrends vertically and leaves the horizontal channel alone."""
    path = _climb(40, rise=ROOT_Y_DRIFT_THRESHOLD * 3.0)
    path[:, 0] = 0.01 * np.sin(np.linspace(0.0, 4.0 * np.pi, 40))  # sway, no travel

    features, xz_flattened, y_flattened, _anim_out = _extract(_anim(path))

    assert y_flattened is True
    assert xz_flattened is False
    _rot, r_pos = recover_root_quat_and_pos_np(features, translation_root_index=0)
    assert np.allclose(r_pos[:, 0] - r_pos[0, 0], path[:, 0] - path[0, 0], atol=1e-5)


def test_a_travelling_climb_loses_both():
    path = _climb(40, rise=ROOT_Y_DRIFT_THRESHOLD * 3.0)
    path[:, 0] = np.linspace(0.0, 4.0, num=40)

    features, xz_flattened, y_flattened, _anim_out = _extract(_anim(path))

    assert xz_flattened is True and y_flattened is True
    _rot, r_pos = recover_root_quat_and_pos_np(features, translation_root_index=0)
    assert float(abs(r_pos[-1, 0] - r_pos[0, 0])) < 1e-3
    assert float(abs(r_pos[-1, 1] - r_pos[0, 1])) < 1e-3


# -- where the correction lands -------------------------------------------


def _wrapper_anim(n_frames: int, wrapper_climbs: bool) -> Animation:
    """Root -> wrapper -> body. The wrapper either carries the climb or sits still."""
    parents = np.array([-1, 0, 1], dtype=np.int64)
    offsets = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    rotations = Quaternions(
        np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_frames, len(parents), 1))
    )
    positions = np.zeros((n_frames, len(parents), 3), dtype=np.float64)
    positions[:, 2] = offsets[2]
    rise = np.linspace(0.0, 2.0, num=n_frames)
    if wrapper_climbs:
        positions[:, 1, 1] = rise
    else:
        positions[:, 2, 1] = offsets[2][1] + rise
    return Animation(rotations, positions, Quaternions.id(len(parents)), offsets, parents)


def test_the_carrier_is_asked_per_axis():
    """A chain static in XZ and moving in Y answers the two questions differently."""
    anim = _wrapper_anim(24, wrapper_climbs=True)

    assert _transport_carrier_index(anim, 2, axes=(0, 2)) == 2
    assert _transport_carrier_index(anim, 2, axes=(1,)) == 1


def test_the_vertical_correction_lands_on_the_joint_that_climbs():
    """The wrapper carries the climb, so it takes the correction and nothing tears."""
    anim = _wrapper_anim(24, wrapper_climbs=True)
    before = positions_global(anim)
    target = np.full(24, before[0, 2, 1], dtype=np.float64)

    corrected = set_translation_root_y(anim, 2, target)
    after = positions_global(corrected)

    assert np.allclose(after[:, 2, 1], target, atol=1e-9)
    # The body stays rigidly seated on the wrapper it was seated on.
    assert np.allclose(after[:, 2] - after[:, 1], before[:, 2] - before[:, 1], atol=1e-9)


def test_a_static_wrapper_is_not_made_to_climb():
    """Nothing above the climbing joint moves, so a still wrapper stays still."""
    anim = _wrapper_anim(24, wrapper_climbs=False)
    before = positions_global(anim)
    target = np.full(24, before[0, 2, 1], dtype=np.float64)

    corrected = set_translation_root_y(anim, 2, target)
    after = positions_global(corrected)

    assert np.allclose(after[:, 2, 1], target, atol=1e-9)
    assert np.allclose(after[:, 1], before[:, 1], atol=1e-9)


# -- the validator reads the same invariant back --------------------------


def test_validator_warns_when_a_selected_clip_still_climbs(capsys):
    from utils.validate_anytop_dataset import _validate_root_motion_drift

    features, _xz, y_flattened, _anim_out = _extract(
        _anim(_climb(40, rise=ROOT_Y_DRIFT_THRESHOLD * 4.0)), locomotion=False
    )
    assert y_flattened is False  # the clip kept its climb

    _validate_root_motion_drift(
        features, 'TestSkeleton', 'TestSkeleton_FlyUp.npy', 0.08, 0,
        parents=np.array([-1, 0], dtype=np.int64),
    )

    assert 'root height still climbs' in capsys.readouterr().out


def test_validator_is_quiet_for_a_flattened_climb(capsys):
    from utils.validate_anytop_dataset import _validate_root_motion_drift

    features, _xz, y_flattened, _anim_out = _extract(
        _anim(_climb(40, rise=ROOT_Y_DRIFT_THRESHOLD * 4.0, bob=0.1))
    )
    assert y_flattened is True

    _validate_root_motion_drift(
        features, 'TestSkeleton', 'TestSkeleton_FlyUp.npy', 0.08, 0,
        parents=np.array([-1, 0], dtype=np.int64),
    )

    assert 'root height still climbs' not in capsys.readouterr().out


# -- the vertical clamp runs after the detrend, exactly once ---------------


def test_process_anim_can_opt_out_of_the_vertical_clamp():
    """The motion path opts out and is bounded later; everything else clamps here."""
    from data_loaders.truebones.truebones_utils.features import process_anim

    source = _anim(_climb(24, rise=0.0))
    source.positions[:, 0, 1] = -2.0  # far below ROOT_Y_MIN_HEIGHT

    clamped, _c, _s = process_anim(
        source, 'TestSkeleton', Quaternions.id(1), scale_factor=1.0,
        translation_root_index=0,
    )
    raw, _c2, _s2 = process_anim(
        source, 'TestSkeleton', Quaternions.id(1), scale_factor=1.0,
        translation_root_index=0, clamp_vertical=False,
    )

    assert positions_global(clamped)[:, 0, 1].min() > ROOT_Y_MIN_HEIGHT
    assert positions_global(raw)[:, 0, 1].min() == pytest.approx(-2.0)


def test_the_bound_applies_to_what_the_detrend_left():
    """A dive deep enough to break the floor comes back inside it once detrended.

    The whole reason the clamp moved: run before the detrend it bounded an
    intermediate trajectory, and the detrend could then walk back out of the
    bound. Run after, the bound holds on what ships.
    """
    descent = _climb(40, rise=-(ROOT_Y_DRIFT_THRESHOLD * 8.0))
    descent[:, 1] -= 0.6  # ends well past the floor

    features, _xz, y_flattened, _anim_out = _extract(
        _anim(descent), locomotion=True, clamp=True,
    )

    assert y_flattened is True
    assert _root_height(features).min() > ROOT_Y_MIN_HEIGHT


def test_an_unbounded_extract_leaves_the_height_alone():
    """``clamp_root_extent`` is opt-in: recovery and retarget re-extract untouched."""
    descent = _climb(40, rise=0.0)
    descent[:, 1] = -1.5

    (features, _mj, _a, _e, _loop, _xz,
     _y) = extract_motion_features_from_aligned_anims(
        _anim(descent), _anim(descent),
        object_type='TestSkeleton', max_joints=8,
        orientation_quat=Quaternions.id(1).qs[0], translation_root_index=0,
        flatten_root_travel=True, clamp_root_extent=False,
    )

    assert _root_height(features).min() == pytest.approx(-1.5, abs=1e-6)


def test_the_pipeline_never_clamps_the_height_twice():
    """Every dataset-build alignment opts out, because extract bounds it later.

    A structural check rather than a numeric one: the band is not idempotent, so
    an alignment call that forgot ``clamp_vertical=False`` would compress the
    excursion once on the way in and again after the detrend, and the only
    symptom would be quietly flattened flight.
    """
    import inspect

    from data_loaders.truebones.truebones_utils import dataset_pipeline

    source = inspect.getsource(dataset_pipeline)
    calls = source.count('get_hml_aligned_anim(')
    opted_out = source.count('clamp_vertical=False')
    assert calls > 0
    assert opted_out == calls, (
        f"{calls} alignment call(s) in dataset_pipeline but {opted_out} opted out "
        "of the vertical clamp"
    )
