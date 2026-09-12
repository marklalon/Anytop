"""Root XZ: what a clip keeps and what it loses.

See docs/root_xz_motion_refactor.md. No clip's root XZ is zeroed. A LOCOMOTION
clip whose cycle-window baseline ends somewhere other than where it started
loses that travel and keeps the within-cycle surge and sway; everything else --
a lunge, a death, a dodge, a swing -- keeps its root motion, bounded by a soft
clamp that is the identity within 0.6 of the origin and asymptotic to 0.8.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion_lib.Animation import Animation, positions_global, rotations_global
from motion_lib.Quaternions import Quaternions

from data_loaders.truebones.truebones_utils.motion_process import (
    ROOT_XZ_DRIFT_THRESHOLD,
    ROOT_XZ_LOCOMOTION_KNEE,
    ROOT_XZ_LOCOMOTION_LIMIT,
    ROOT_XZ_SOFT_CLAMP_KNEE,
    ROOT_XZ_SOFT_CLAMP_LIMIT,
    flatten_root_xz_drift,
    scale_root_xz_extent,
    soft_clamp_extent,
    soft_clamp_root_xz,
)
from data_loaders.truebones.truebones_utils.animation_utils import (
    _frame_correction,
    _transport_carrier_index,
    collapse_translation_root_chain,
    promote_translation_root_to_hierarchy_root,
    root_xz_drift_correction,
    root_xz_heading,
    root_xz_trajectory,
    set_translation_root_xz,
)
from data_loaders.truebones.truebones_utils.features import (
    extract_motion_features_from_aligned_anims,
)
from data_loaders.truebones.truebones_utils.motion_process import (
    move_xz_to_origin,
)
from data_loaders.truebones.truebones_utils.param_utils import FEATS_LEN
from data_loaders.truebones.data.dataset import _tile_loop_motion


def _straight_line_anim(n_frames: int, path_xz: np.ndarray) -> Animation:
    """A two-joint skeleton whose root walks ``path_xz`` while the child bobs."""
    parents = np.array([-1, 0], dtype=np.int64)
    offsets = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    rotations = Quaternions(
        np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_frames, len(parents), 1))
    )
    positions = np.zeros((n_frames, len(parents), 3), dtype=np.float64)
    positions[:, 0, 0] = path_xz[:, 0]
    positions[:, 0, 2] = path_xz[:, 1]
    positions[:, 1] = offsets[1]
    return Animation(rotations, positions, Quaternions.id(len(parents)), offsets, parents)


def _turning_anim(n_frames: int, path_xz: np.ndarray, turn_deg: float) -> Animation:
    """``_straight_line_anim`` with the root yawing steadily across the clip.

    The shape every RunLeft/GlideLeft in the shipped data has: travel whose
    direction rotates with the character. A frame fitted end to end cannot
    follow it; the root's own heading can.
    """
    anim = _straight_line_anim(n_frames, path_xz)
    yaw = np.radians(turn_deg) * np.arange(n_frames) / max(n_frames - 1, 1)
    anim.rotations[:, 0] = Quaternions.from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))
    return anim


def _feature_heading(features, parents, root_index):
    """The heading the dataset validator reads back off a feature tensor.

    Rotations are recovered straight from the 6D channels, so the offsets only
    place joints in space and cannot move this; zeros are enough.
    """
    from data_loaders.truebones.truebones_utils.motion_process import (
        recover_from_bvh_rot_np,
    )

    parents = np.asarray(parents, dtype=np.int64)
    _positions, recovered = recover_from_bvh_rot_np(
        features,
        parents,
        np.zeros((parents.shape[0], 3), dtype=np.float64),
        translation_root_index=root_index,
    )
    return root_xz_heading(recovered, root_index)


def _flat_heading(n_frames: int) -> np.ndarray:
    """The heading of a rig that never turns -- what the straight fixtures have."""
    return np.zeros(n_frames, dtype=np.float64)


def _turning_path(n_frames: int, radius: float, turn_deg: float) -> np.ndarray:
    """A constant-speed arc: travel that curves through ``turn_deg``."""
    ang = np.radians(turn_deg) * np.arange(n_frames) / max(n_frames - 1, 1)
    return np.stack([radius * (1.0 - np.cos(ang)), radius * np.sin(ang)], axis=-1)


def _extract(anim, locomotion=True, clamp=False):
    features, _max_joints, motion_anim, _e, is_loop, flattened = extract_motion_features_from_aligned_anims(
        anim,
        anim,
        object_type='TestSkeleton',
        max_joints=8,
        orientation_quat=Quaternions.id(1).qs[0],
        translation_root_index=0,
        flatten_root_travel=locomotion,
        clamp_root_xz_extent=clamp,
    )
    return features, is_loop, flattened, motion_anim


def _reach(features, root_index=0):
    return float(np.linalg.norm(_root_path(features, root_index), axis=1).max())


def _root_path(features, root_index=0):
    """Integrate the root XZ back out of the features exactly as the decoder does.

    T frames starting at the origin, not T-1 velocity steps: the dataset
    validator reads the same reconstruction, and fitting a baseline over a path
    one frame short of the clip is enough to disagree about a curve.
    """
    from data_loaders.truebones.truebones_utils.motion_process import (
        recover_root_quat_and_pos_np,
    )
    _rot, r_pos = recover_root_quat_and_pos_np(features, translation_root_index=root_index)
    return r_pos[:, [0, 2]]


def _closed_excursion(n_frames: int, amplitude: float) -> np.ndarray:
    """Go out and come back: net displacement 0, extent ``amplitude``."""
    t = np.linspace(0.0, np.pi, num=n_frames, endpoint=True)
    return np.stack([amplitude * np.sin(t), np.zeros_like(t)], axis=-1)


def _travelling_path(n_frames: int, distance: float = 4.0) -> np.ndarray:
    return np.stack(
        [np.linspace(0.0, distance, num=n_frames), np.zeros(n_frames)], axis=-1
    )


def _gait_path(n_frames: int, cycle: int, stride: float, surge: float) -> np.ndarray:
    """Steady travel with a within-cycle surge riding on top of it."""
    t = np.arange(n_frames, dtype=np.float64)
    return np.stack(
        [stride * t / cycle + surge * np.sin(2.0 * np.pi * t / cycle), np.zeros(n_frames)],
        axis=-1,
    )


def _least_squares_heading_fit(heading: np.ndarray) -> np.ndarray:
    """The rejected estimator, kept as the reference the tests measure against.

    A straight line through EVERY frame of the heading, which is what the detrend
    fitted before the endpoint ramp replaced it.
    """
    heading = np.asarray(heading, dtype=np.float64)
    times = np.arange(heading.shape[0], dtype=np.float64)
    dev = times - times.mean()
    slope = float((dev * (heading - heading.mean())).sum() / float(dev @ dev))
    return heading.mean() + slope * dev


def _residual_extent(traj: np.ndarray, correction: np.ndarray) -> float:
    """How far the corrected path still ranges from its own frame 0."""
    flat = np.asarray(traj, dtype=np.float64) - correction
    return float(np.linalg.norm(flat - flat[0], axis=1).max())


# ── the gate: a path that closes is kept, at any amplitude ─────────────────

@pytest.mark.parametrize('amplitude', [0.3, 0.6, 2.0])
def test_a_closed_excursion_is_kept_however_far_it_reaches(amplitude):
    """Strikes, dodges and sways. The excursion IS the action.

    The old extent gate zeroed everything past 0.6 wholesale; measured on the
    shipped datasets those clips' net displacement had a median of 0.00 -- they
    were out-and-back all along.
    """
    features, _loop, flattened, _anim = _extract(
        _straight_line_anim(40, _closed_excursion(40, amplitude))
    )

    assert flattened is False
    travelled = np.abs(_root_path(features)[:, 0]).max()
    assert travelled == pytest.approx(amplitude, rel=0.05)


def test_sustained_travel_is_flattened():
    features, _loop, flattened, _anim = _extract(
        _straight_line_anim(40, _travelling_path(40))
    )

    assert flattened is True
    _flat, drift = flatten_root_xz_drift(_root_path(features), _flat_heading(40))
    assert drift <= ROOT_XZ_DRIFT_THRESHOLD


def test_flattening_keeps_the_within_cycle_surge():
    """The point of the whole change: in place, but not welded to the floor."""
    n_frames, cycle, surge = 60, 20, 0.05
    path = _gait_path(n_frames, cycle, stride=0.5, surge=surge)
    features, _loop, flattened, _anim = _extract(_straight_line_anim(n_frames, path))

    assert flattened is True
    residual = _root_path(features)[:, 0]
    assert residual.max() - residual.min() > surge
    # ...and what is left is the oscillation, not a fraction of the travel.
    assert np.abs(residual).max() < 0.25 * path[-1, 0]


def test_nothing_is_ever_written_as_an_exact_zero():
    """There is no fabricated value left for a conditioning flag to warn about."""
    features, _loop, flattened, _anim = _extract(
        _straight_line_anim(60, _gait_path(60, 20, stride=0.5, surge=0.05))
    )

    assert flattened is True
    assert np.abs(features[:, 0, [9, 11]]).max() > 0.0


def test_frame_zero_stays_at_the_origin():
    """The pipeline centred the clip there and every downstream consumer assumes it."""
    correction, _drift = root_xz_drift_correction(_travelling_path(40), _flat_heading(40))
    np.testing.assert_allclose(correction[0], 0.0, atol=1e-12)


def test_an_authored_in_place_clip_is_untouched():
    n_frames = 30
    features, _loop, flattened, _anim = _extract(
        _straight_line_anim(n_frames, np.zeros((n_frames, 2)))
    )

    assert flattened is False
    assert np.abs(features[:, 0, [9, 11]]).max() == pytest.approx(0.0, abs=1e-9)


def test_root_ric_xz_is_structurally_zero():
    """The exporter's unconditional RIC cleanup rests on this identity."""
    features, _loop, _flattened, _anim = _extract(
        _straight_line_anim(30, _closed_excursion(30, 0.4))
    )
    np.testing.assert_array_equal(features[:, 0, [0, 2]], 0.0)


def test_flattening_is_idempotent():
    """Re-measuring a flattened path must not find travel to remove again."""
    path = _gait_path(60, 20, stride=0.5, surge=0.05)
    heading = _flat_heading(60)
    flat, drift = flatten_root_xz_drift(path, heading)
    assert drift > ROOT_XZ_DRIFT_THRESHOLD

    _again, second_drift = flatten_root_xz_drift(flat, heading)
    assert second_drift <= ROOT_XZ_DRIFT_THRESHOLD


# ── only a gait's travel is removed ────────────────────────────────────────

def test_a_one_shot_action_keeps_its_displacement():
    """A death, a knockdown, a leap: the character ends up somewhere else and
    that IS the action.

    No measurement separates these from a gait take -- both are one closed cycle
    that ends displaced -- so the caller's action group decides. On the shipped
    datasets a pure drift test would have flattened 393 death and knockdown
    clips (Monkey_Die drifts 0.91, TNR_Archer_DeathA 0.82).
    """
    path = _travelling_path(40, distance=0.8)
    features, _loop, flattened, _anim = _extract(
        _straight_line_anim(40, path), locomotion=False
    )

    assert flattened is False
    np.testing.assert_allclose(_root_path(features)[-1, 0], path[-1, 0], atol=1e-6)


def test_a_locomotion_one_shot_is_flattened_and_then_bounded():
    """MB_Unka_GroundDodgeLeft: one accelerate-decelerate move, 2.0 across.

    Labelled locomotion, so its transport comes off like any gait's -- and what a
    detrend leaves of it is large, 0.210, past the group's own 0.200 limit. That
    residual is not exempted from the bound, it is what the bound is for: the
    scale takes it to 0.152. The assertions pin the behaviour rather than the 5%
    margin, which is the real clip's.
    """
    n_frames = 61
    t = np.linspace(0.0, np.pi, n_frames)
    path = np.stack([1.0 - np.cos(t), np.zeros(n_frames)], axis=-1)

    flattened_path, drift = flatten_root_xz_drift(path, _flat_heading(n_frames))
    residual = float(np.linalg.norm(flattened_path, axis=1).max())
    assert drift == pytest.approx(2.0, rel=1e-6)
    # Past the knee, so the bound has something to act on.
    assert residual > ROOT_XZ_LOCOMOTION_KNEE

    features, _loop, flattened, _anim = _extract(
        _straight_line_anim(n_frames, path), clamp=True
    )

    reach = _reach(features)
    assert flattened is True
    assert ROOT_XZ_LOCOMOTION_KNEE < reach < ROOT_XZ_LOCOMOTION_LIMIT
    # Strictly inside the residual: the scale is what bounded it, not the detrend
    # having already left something small enough.
    assert reach < residual


def test_a_gait_that_was_authored_in_place_is_left_alone():
    """Permission is not an instruction: an in-place gait has no travel to take,
    so it must not be pushed through the operator for float noise."""
    n_frames = 40
    features, _loop, flattened, _anim = _extract(
        _straight_line_anim(n_frames, _closed_excursion(n_frames, 0.05)), locomotion=True
    )

    assert flattened is False


def test_the_pipeline_reads_the_gate_from_the_action_labels_sidecar(tmp_path):
    """And a dataset with no sidecar fails fast: preprocessing never flattens
    on a guess, the sidecar is a prerequisite."""
    from data_loaders.truebones.truebones_utils.dataset_pipeline import (
        load_locomotion_clip_names,
    )

    rows = [
        {"clip": "Wolf_Walk", "action_group": "locomotion", "action_label": "walk"},
        {"clip": "Wolf_Die", "action_group": "transition", "action_label": "die"},
    ]
    (tmp_path / 'action_labels.jsonl').write_text(
        os.linesep.join(json.dumps(row) for row in rows), encoding='utf-8'
    )
    assert load_locomotion_clip_names(tmp_path) == {'Wolf_Walk'}
    with pytest.raises(FileNotFoundError):
        load_locomotion_clip_names(tmp_path / 'nope')


def test_the_validator_only_checks_locomotion_clips():
    import inspect

    from utils import validate_anytop_dataset

    source = inspect.getsource(validate_anytop_dataset.validate_motion_files)
    assert 'if motion_path.name in locomotion_clips:' in source


# ── a curve is travel too, and the heading follows it ─────────────────────

@pytest.mark.parametrize('turn_deg', [30.0, 60.0, 90.0, 180.0])
def test_a_turning_gait_is_flattened_as_completely_as_a_straight_one(turn_deg):
    """The reported bug: RunLeft/GlideLeft kept a bow of residual travel.

    An end-to-end line fit cannot represent an arc, so it left the sagitta
    behind -- 0.371 for ``MB_Unka_GlideLeft`` and 0.732 for
    ``IAC_Cavewoman_RunTurnRight``, 27% and 53% of a body span. In the root's
    own heading frame the same motion is a constant forward speed, and a
    constant is what the detrend removes.
    """
    n_frames = 40
    path = _turning_path(n_frames, radius=1.6, turn_deg=turn_deg)
    features, _loop, flattened, _anim = _extract(
        _turning_anim(n_frames, path, turn_deg)
    )

    assert flattened is True
    travelled = np.linalg.norm(path - path[0], axis=1).max()
    residual = np.linalg.norm(_root_path(features), axis=1).max()
    assert residual < 0.05 * travelled


def test_a_heading_that_wanders_cannot_bend_the_frame():
    """MB_TigerDrago_RunJump: the clip travels dead straight while the pelvis
    yaws and comes back.

    A least-squares line through that heading reports a turn that never happened
    and leaves 0.46 of lateral displacement in a path that has none. A ramp
    between the endpoints is moved not at all by a yaw that returns to its start.
    """
    n_frames = 120
    traj = _travelling_path(n_frames, distance=4.5)
    u = np.linspace(0.0, 1.0, n_frames)
    heading = 0.5 - np.sin(np.pi * u ** 2)      # wanders out and back, net turn 0

    line = _least_squares_heading_fit(heading[:-1])
    assert _residual_extent(traj, _frame_correction(traj, line)) > 0.4

    correction, drift = root_xz_drift_correction(traj, heading)

    assert _residual_extent(traj, correction) < 0.05
    # ...and the travel it was carrying is still removed in full.
    assert drift == pytest.approx(np.linalg.norm(traj[-1] - traj[0]), rel=1e-2)


def test_a_turning_gait_still_turns_when_the_pelvis_wobbles_too():
    """KI_Soldier_Crawling01TurnRight01: a real 0.73 rad turn under 11.40 rad of
    crawling yaw.

    Both failure modes are live at once, so it pins the frame against both:
    holding it constant leaves the arc's sagitta (10% of the path), and fitting
    every frame lets the wobble tilt the line. The net turn is neither -- it is
    the turn, and only the turn.
    """
    n_frames = 120
    path = _turning_path(n_frames, radius=1.6, turn_deg=45.0)
    u = np.linspace(0.0, 1.0, n_frames)
    turn = np.radians(45.0) * np.arange(n_frames) / (n_frames - 1)
    heading = turn + 0.8 * (0.5 - np.sin(np.pi * u ** 2))

    travelled = float(np.linalg.norm(path - path[0], axis=1).max())
    held_constant = _frame_correction(path, np.zeros(n_frames - 1))
    fitted = _frame_correction(path, _least_squares_heading_fit(heading[:-1]))
    assert _residual_extent(path, held_constant) > 0.09 * travelled
    assert _residual_extent(path, fitted) > 0.08 * travelled

    correction, _drift = root_xz_drift_correction(path, heading)

    assert _residual_extent(path, correction) < 0.01 * travelled


def test_the_end_to_end_line_fit_is_what_the_heading_frame_replaces():
    """Pin the failure the fix exists for: same arc, heading withheld.

    97.7% of the flattened clips in the shipped datasets are a single stride
    long, so the old cycle estimator found no period and the baseline collapsed
    to one straight line across the whole clip. This is that line.
    """
    n_frames = 40

    def reach(path):
        return float(np.linalg.norm(path - path[0], axis=1).max())

    line_fit = []
    for turn_deg in (30.0, 60.0, 90.0, 180.0):
        arc = _turning_path(n_frames, radius=1.6, turn_deg=turn_deg)
        straight_frame, _drift = flatten_root_xz_drift(arc, _flat_heading(n_frames))
        heading_frame, _drift = flatten_root_xz_drift(
            arc, root_xz_heading(_turning_anim(n_frames, arc, turn_deg), 0)
        )

        # What the line fit cannot reach is the arc's sagitta, and it grows with
        # the turn: 6.6% of the path at 30 degrees, 50% at 180.
        assert reach(straight_frame) > 0.05 * reach(arc)
        line_fit.append(reach(straight_frame) / reach(arc))

        # In the heading frame a constant-rate turn is a constant forward speed,
        # so there is nothing left over at all.
        assert reach(heading_frame) < 1e-9

    assert line_fit == sorted(line_fit)


def test_a_constant_heading_offset_cannot_change_the_result():
    """What lets the dataset validator re-measure drift off the features.

    Its recovered animation carries identity orients where the pipeline's rest
    pose carried real ones, so its heading can sit a constant away from the one
    preprocessing used. Rotating into a frame, removing a mean and rotating back
    is equivariant under a constant rotation of that frame, so it cancels.
    """
    n_frames = 40
    arc = _turning_path(n_frames, radius=1.6, turn_deg=75.0)
    heading = root_xz_heading(_turning_anim(n_frames, arc, 75.0), 0)

    base, base_drift = flatten_root_xz_drift(arc, heading)
    offset, offset_drift = flatten_root_xz_drift(arc, heading + 1.234)

    np.testing.assert_allclose(base, offset, atol=1e-12)
    assert base_drift == pytest.approx(offset_drift, abs=1e-12)


def test_the_heading_is_read_through_the_seated_wrapper():
    """Rigs that keep the turn on an inert ``CG``/``All`` node above the root.

    ``collapse_translation_root_chain`` seats those onto the root before the
    flatten runs, so the root's world rotation carries their yaw too -- which is
    the only reason the heading is read off the collapsed animation.
    """
    n_frames = 30
    anim = _static_wrapper_anim(n_frames, _travelling_path(n_frames))
    yaw = np.radians(60.0) * np.arange(n_frames) / (n_frames - 1)
    anim.rotations[:, 0] = Quaternions.from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))

    seated = collapse_translation_root_chain(anim, 1)
    heading = root_xz_heading(seated, 1)

    assert np.degrees(heading[-1] - heading[0]) == pytest.approx(60.0, abs=1e-6)


# ── the re-seat itself ─────────────────────────────────────────────────────

def _intermediate_root_anim(n_frames: int, path_xz: np.ndarray) -> Animation:
    """A rig whose locomotion lives on joint 1, under a wrapper root that moves.

    The KI_* characters are shaped like this. It matters because the re-seat has
    to solve joint 1's translation in its PARENT's frame, so "the root follows
    this path" is a non-linear property of the channels that carry it.
    """
    parents = np.array([-1, 0, 1], dtype=np.int64)
    offsets = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    t = np.linspace(0.0, 1.0, num=n_frames)
    rotations = Quaternions(
        np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_frames, len(parents), 1))
    )
    rotations.qs[:, 0] = Quaternions.from_euler(
        np.stack([0.4 * t, 1.1 * t, -0.3 * t], axis=-1)
    ).qs
    positions = np.zeros((n_frames, len(parents), 3))
    positions[:, 0, 0] = 0.2 * t                 # the wrapper root drifts too
    positions[:, 0, 1] = 0.5
    positions[:, 0, 2] = -0.15 * t
    positions[:, 1, 0] = path_xz[:, 0]
    positions[:, 1, 2] = path_xz[:, 1]
    positions[:, 1, 1] = 1.0 + 0.05 * np.sin(2.0 * np.pi * t)
    positions[:, 2] = offsets[2]
    return Animation(rotations, positions, Quaternions.id(len(parents)), offsets, parents)


def _static_wrapper_anim(n_frames: int, path_xz: np.ndarray) -> Animation:
    """Horse / Jaws / Bear / Crow / Pirrana: joint 0 sits still at the origin
    while the effective root walks away from it on its own channel."""
    parents = np.array([-1, 0, 1], dtype=np.int64)
    offsets = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rotations = Quaternions(
        np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_frames, len(parents), 1))
    )
    positions = np.zeros((n_frames, len(parents), 3))
    positions[:, 0, 1] = 0.5
    positions[:, 1, 0] = path_xz[:, 0]
    positions[:, 1, 2] = path_xz[:, 1]
    positions[:, 1, 1] = 1.0
    positions[:, 2] = offsets[2]
    return Animation(rotations, positions, Quaternions.id(len(parents)), offsets, parents)


def _riding_root_anim(n_frames: int, path_xz: np.ndarray) -> Animation:
    """Dog / Dog-2: joint 0 carries the travel and the effective root rides it.

    The two joints keep a constant offset, so nothing may come between them.
    """
    parents = np.array([-1, 0, 1], dtype=np.int64)
    offsets = np.array([[0.0, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 1.0, 0.0]])
    rotations = Quaternions(
        np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_frames, len(parents), 1))
    )
    positions = np.zeros((n_frames, len(parents), 3))
    positions[:, 0, 0] = path_xz[:, 0]
    positions[:, 0, 2] = path_xz[:, 1]
    positions[:, 1] = offsets[1]
    positions[:, 2] = offsets[2]
    return Animation(rotations, positions, Quaternions.id(len(parents)), offsets, parents)


def test_a_static_wrapper_keeps_its_place_and_the_root_takes_the_correction():
    anim = _static_wrapper_anim(24, _travelling_path(24))
    target = _closed_excursion(24, 0.3)

    assert _transport_carrier_index(anim, 1) == 1
    reseated = set_translation_root_xz(anim, 1, target)

    np.testing.assert_allclose(root_xz_trajectory(reseated, 1), target, atol=1e-9)
    # Shifting a joint that never moved would invent travel for it.
    np.testing.assert_allclose(
        positions_global(reseated)[:, 0], positions_global(anim)[:, 0], atol=1e-9
    )


def test_a_travelling_ancestor_takes_the_correction_with_the_root():
    """Dog_Running: Hips carries the travel, Spine0 (the effective root) rides
    along. Re-seating Spine0 alone left Hips travelling and tore the two apart
    by 5.27 -- all of it landing in Hips' RIC, which is the reported symptom.
    """
    anim = _riding_root_anim(24, _travelling_path(24))
    target = _closed_excursion(24, 0.3)

    assert _transport_carrier_index(anim, 1) == 0
    reseated = set_translation_root_xz(anim, 1, target)

    np.testing.assert_allclose(root_xz_trajectory(reseated, 1), target, atol=1e-9)
    # The ancestor moved by exactly the same delta: no relative motion invented.
    before, after = positions_global(anim), positions_global(reseated)
    delta_root = (after - before)[:, 1][:, [0, 2]]
    delta_ancestor = (after - before)[:, 0][:, [0, 2]]
    np.testing.assert_allclose(delta_ancestor, delta_root, atol=1e-9)


def test_the_carrier_is_the_highest_joint_that_actually_moves():
    static = _static_wrapper_anim(24, _travelling_path(24))
    riding = _riding_root_anim(24, _travelling_path(24))

    assert _transport_carrier_index(static, 1) == 1
    assert _transport_carrier_index(riding, 1) == 0
    # A root-0 rig has nothing above it to consider.
    assert _transport_carrier_index(_straight_line_anim(24, _travelling_path(24)), 0) == 0


def test_set_translation_root_xz_rejects_a_mismatched_target():
    anim = _straight_line_anim(12, _travelling_path(12))
    with pytest.raises(ValueError, match='target_xz'):
        set_translation_root_xz(anim, 0, np.zeros((11, 2)))


def test_reextraction_after_a_resample_runs_from_the_source_anims():
    """The resample branch must not feed pass 1's output back in.

    Re-measuring a flattened trajectory would look for drift that was just
    removed, so the branch resamples the anims as AUTHORED and extracts once
    more from those.
    """
    import inspect

    from data_loaders.truebones.truebones_utils import dataset_pipeline

    source = inspect.getsource(dataset_pipeline._prepare_object_outputs)
    assert "result['source_new_anim'] = _resample_animation(" in source
    assert "result['source_export_anim'] = _resample_animation(" in source
    assert 'force_strip' not in source


def test_resampling_the_source_and_reextracting_agrees_with_the_first_pass():
    from data_loaders.truebones.truebones_utils.dataset_pipeline import _resample_animation

    source = _intermediate_root_anim(18, _travelling_path(18))
    first, _mj, _m, _e, _loop, first_flattened = extract_motion_features_from_aligned_anims(
        source, source, 'TestSkeleton', 8,
        Quaternions.id(1).qs[0], translation_root_index=1,
        flatten_root_travel=True,
    )
    assert first_flattened is True

    resampled = _resample_animation(source, 20)
    second, _mj, _m, _e, _loop2, second_flattened = extract_motion_features_from_aligned_anims(
        resampled, resampled, 'TestSkeleton', 8,
        Quaternions.id(1).qs[0], translation_root_index=1,
        flatten_root_travel=True,
    )

    assert second_flattened is True
    assert first.shape[0] == 18 and second.shape[0] == 20
    # Measured the way the dataset validator measures it: the decoder's own
    # reconstruction, in the heading read back off the same features. This rig
    # keeps its turn on the wrapper above the root, so a flat heading would not
    # do -- which is the point.
    _flat, drift = flatten_root_xz_drift(
        _root_path(second, root_index=1),
        _feature_heading(second, np.array([-1, 0, 1]), 1),
    )
    assert drift <= ROOT_XZ_DRIFT_THRESHOLD


def test_a_riding_ancestor_does_not_accumulate_the_removed_travel():
    """The reported bug, at the feature level.

    Dog / Dog-2 are ``Hips(0) -> Spine0(1) -> Pelvis(2)`` with the species
    translation root frozen at Spine0, while Hips is what actually travels. The
    body read as in place because the features de-root on Spine0 -- and the
    whole trajectory reappeared in Hips' RIC channel (Dog_Running 5.27,
    Dog-2_RunFast 8.68).
    """
    n_frames = 40
    anim = _riding_root_anim(n_frames, _travelling_path(n_frames, distance=4.0))

    features, _mj, _m, _e, _loop, flattened = extract_motion_features_from_aligned_anims(
        anim, anim, 'TestSkeleton', 8,
        Quaternions.id(1).qs[0], translation_root_index=1,
        flatten_root_travel=True,
    )

    assert flattened is True
    ancestor_ric = features[:, 0][:, [0, 2]]
    excursion = float(np.linalg.norm(ancestor_ric - ancestor_ric[0], axis=1).max())
    assert excursion < 0.05, f'the ancestor kept {excursion:.3f} of the removed travel'


# ── §1.4: a travelling loop still tiles without a seam ─────────────────────

def test_travelling_loop_tiles_without_a_jump():
    """Position is the cumulative sum of the velocity rows, and the terminal row
    holds the cycle's wrap delta (its last step), so joining two copies is just
    concatenating velocities -- the seam step must match the stride.
    """
    n_frames = 24
    step = 0.05
    motion = np.zeros((n_frames, 2, FEATS_LEN), dtype=np.float32)
    motion[:, 1, 9] = step           # a constant stride in +X, terminal row included

    tiled = _tile_loop_motion(motion, 2)
    assert tiled.shape[0] == 2 * n_frames

    path = np.cumsum(tiled[:-1, 1, 9])
    steps = np.diff(path)
    np.testing.assert_allclose(steps, step, atol=1e-6)
    assert path[-1] == pytest.approx(step * (2 * n_frames - 1), rel=1e-5)


# ── the conditioning channel is gone ───────────────────────────────────────

def test_the_model_has_no_root_xz_strip_channel():
    """Nothing is fabricated any more, so there is no fabrication to declare.

    The flag existed to tell the model "this root trajectory is a zero this
    pipeline wrote, do not average it into the honest samples". No clip carries
    such a zero now.
    """
    from model.anytop import AnyTop

    model = AnyTop(
        max_joints=4, feature_len=12, latent_dim=8, ff_size=32, num_layers=1,
        num_heads=2, dropout=0.0, cross_limb=True, t5_out_dim=512,
    )
    assert not hasattr(model, 'root_xz_strip_projection')


def test_the_collate_no_longer_carries_the_flag():
    import torch
    from data_loaders.tensors import truebones_collate

    def _item():
        return {
            'inp': torch.zeros(3, 12, 5),
            'rest_pose': torch.zeros(3, 12),
            'n_joints': 3,
            'lengths': 5,
            'parents': np.array([-1, 0, 1]),
            'graph_dist': torch.zeros(4, 4),
            'joints_relations': torch.zeros(4, 4),
            'object_type': 'TestSkeleton',
            'joints_names_embs': torch.zeros(4, 512),
            'joints_padding_mask': torch.ones(1, 1, 4, 4),
            'is_loop': False,
        }

    _motion, cond = truebones_collate([_item(), _item()])
    assert 'root_xz_stripped' not in cond['y']


def test_generation_declares_no_root_xz_flag():
    import inspect

    from sample import generate as generate_module

    source = inspect.getsource(generate_module.create_condition)
    assert 'root_xz_stripped' not in source


def test_checkpoint_version_rejects_the_previous_generation():
    from utils.parser_util import CKPT_VERSION

    assert CKPT_VERSION >= 6


# ── the inert joints above the root ────────────────────────────────────────

def test_the_root_subtree_is_untouched_when_the_wrapper_is_seated():
    anim = _static_wrapper_anim(24, _travelling_path(24))

    seated = collapse_translation_root_chain(anim, 1)

    before, after = positions_global(anim), positions_global(seated)
    # The fixture is wrapper -> root -> child, so the root's subtree is [1, 2].
    np.testing.assert_allclose(after[:, 1:], before[:, 1:], atol=1e-9)


def test_the_wrapper_stops_carrying_the_negated_root_trajectory():
    """``Horse_Attack`` 0.718, ``Pirrana_Jump2`` 0.783: a wrapper that stands
    still while the root walks away reads, after de-rooting, as a joint moving
    backwards by exactly the trajectory."""
    anim = _static_wrapper_anim(40, _closed_excursion(40, 2.0))

    def derooted_wrapper_xz(global_pos):
        return global_pos[:, 0][:, [0, 2]] - global_pos[:, 1][:, [0, 2]]

    before = derooted_wrapper_xz(positions_global(anim))
    after = derooted_wrapper_xz(positions_global(collapse_translation_root_chain(anim, 1)))

    assert np.linalg.norm(before - before[0], axis=1).max() > 1.0
    assert np.abs(after - after[0]).max() < 1e-9


def test_seating_the_wrapper_is_idempotent():
    anim = _static_wrapper_anim(24, _travelling_path(24))

    once = collapse_translation_root_chain(anim, 1)
    twice = collapse_translation_root_chain(once, 1)

    np.testing.assert_allclose(twice.positions, once.positions, atol=1e-12)


def test_a_rig_whose_hierarchy_root_already_carries_the_travel_is_left_alone():
    """Dog / Dog-2: Hips travels and the effective root rides it at a fixed
    offset, so there is nothing to re-seat."""
    anim = _riding_root_anim(24, _travelling_path(24))

    seated = collapse_translation_root_chain(anim, 1)

    np.testing.assert_allclose(seated.positions, anim.positions, atol=1e-9)


def test_a_branch_above_the_root_blocks_the_re_seat():
    """The guard. Moving a joint that carries something else would drag it too.

    Every root this pipeline picks comes off the unbranched candidate chain, so
    this protects a hand-set or stale index rather than a case in the data.
    """
    parents = np.array([-1, 0, 0], dtype=np.int64)
    offsets = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    rotations = Quaternions(np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (12, 3, 1)))
    positions = np.zeros((12, 3, 3))
    positions[:, 1, 0] = np.linspace(0.0, 3.0, num=12)
    positions[:, 2] = offsets[2]
    anim = Animation(rotations, positions, Quaternions.id(3), offsets, parents)

    seated = collapse_translation_root_chain(anim, 1)

    np.testing.assert_array_equal(seated.positions, anim.positions)


def test_the_wrapper_ric_no_longer_depends_on_where_the_clip_was_authored():
    """The per-clip constant.

    Centring subtracts the effective root's initial XZ from joint 0, so the
    wrapper lands at minus wherever the animator parked the character -- same
    species, same rig, a different answer per clip (MB_TigerDrago: DogdeLeftG
    1.546, Run 0.193, Idle 0.102, GetHitR 0.000).
    """
    reach = []
    for start in (0.0, 4.0, -9.0):
        path = _closed_excursion(30, 0.3) + np.array([start, 0.0])
        anim = _static_wrapper_anim(30, path)
        # Centre on the effective root the way process_anim does.
        centred, _ = move_xz_to_origin(anim, translation_root_index=1)

        features, _mj, _m, _e, _loop, _flat = extract_motion_features_from_aligned_anims(
            centred, centred, 'TestSkeleton', 8,
            Quaternions.id(1).qs[0], translation_root_index=1,
        )
        wrapper = features[:, 0][:, [0, 2]]
        reach.append(float(np.linalg.norm(wrapper, axis=1).max()))

    np.testing.assert_allclose(reach, 0.0, atol=1e-6)


# ── dropping the wrapper outright ──────────────────────────────────────────

def _yawing_wrapper_anim(n_frames: int, path_xz: np.ndarray) -> Animation:
    """``Wrapper(0) -> Root(1) -> Tip(2)``, with the WRAPPER carrying the yaw.

    Nine of the 34 wrapper species author the character's turn on the wrapper
    rather than on the joint it holds -- ``SabreToothTiger_180RIght``,
    ``KI_Human_RunTurn02Right``, ``Dog_TurnRight``. Dropping the joint has to
    take that rotation with it.
    """
    parents = np.array([-1, 0, 1], dtype=np.int64)
    offsets = np.array([[0.0, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 1.0, 0.0]])
    yaw = np.linspace(0.0, np.pi, num=n_frames)
    rotations = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_frames, len(parents), 1))
    rotations[:, 0, 0] = np.cos(yaw / 2.0)
    rotations[:, 0, 2] = np.sin(yaw / 2.0)
    positions = np.zeros((n_frames, len(parents), 3))
    positions[:, 0, 0] = path_xz[:, 0]
    positions[:, 0, 2] = path_xz[:, 1]
    positions[:, 1] = offsets[1]
    positions[:, 2] = offsets[2]
    return Animation(
        Quaternions(rotations), positions, Quaternions.id(len(parents)), offsets, parents
    )


def test_dropping_the_wrapper_keeps_every_remaining_joint_where_it_was():
    anim = _yawing_wrapper_anim(24, _travelling_path(24))

    promoted, names, keep = promote_translation_root_to_hierarchy_root(
        anim, ['Wrapper', 'Root', 'Tip'], 1,
    )

    assert names == ['Root', 'Tip']
    assert keep == [1, 2]
    np.testing.assert_allclose(
        positions_global(promoted), positions_global(anim)[:, keep], atol=1e-9
    )


def test_dropping_the_wrapper_takes_its_yaw_with_it():
    """Without the rotation fold the turn would simply vanish."""
    anim = _yawing_wrapper_anim(24, _travelling_path(24))

    promoted, _names, keep = promote_translation_root_to_hierarchy_root(
        anim, ['Wrapper', 'Root', 'Tip'], 1,
    )

    before = rotations_global(anim).qs[:, keep]
    after = rotations_global(promoted).qs
    # Quaternion double cover: q and -q are the same rotation.
    assert np.minimum(np.abs(after - before), np.abs(after + before)).max() < 1e-9
    # ...and the yaw really is there to lose.
    assert np.abs(after[-1] - after[0]).max() > 0.5


def test_dropping_nothing_is_a_no_op():
    anim = _yawing_wrapper_anim(12, _travelling_path(12))

    promoted, names, keep = promote_translation_root_to_hierarchy_root(
        anim, ['Wrapper', 'Root', 'Tip'], 0,
    )

    assert keep is None
    assert names == ['Wrapper', 'Root', 'Tip']
    assert promoted is anim


def test_a_branch_above_the_root_refuses_the_drop():
    """Moving a joint that carries something else would drag it along."""
    parents = np.array([-1, 0, 0], dtype=np.int64)
    offsets = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    rotations = Quaternions(np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (8, 3, 1)))
    positions = np.zeros((8, 3, 3))
    positions[:, 1] = offsets[1]
    positions[:, 2] = offsets[2]
    anim = Animation(rotations, positions, Quaternions.id(3), offsets, parents)

    with pytest.raises(ValueError, match='more than one child'):
        promote_translation_root_to_hierarchy_root(anim, ['A', 'B', 'C'], 1)


# ── the ceiling: an excursion is bounded, never zeroed ─────────────────────

def test_the_clamp_leaves_everything_inside_the_knee_alone():
    traj = _closed_excursion(40, ROOT_XZ_SOFT_CLAMP_KNEE)

    np.testing.assert_array_equal(soft_clamp_root_xz(traj), traj)


@pytest.mark.parametrize('amplitude', [0.7, 2.0, 8.0, 100.0])
def test_the_clamp_bounds_any_reach_strictly_under_the_ceiling(amplitude):
    clamped = soft_clamp_root_xz(_closed_excursion(40, amplitude))

    assert np.linalg.norm(clamped, axis=1).max() < ROOT_XZ_SOFT_CLAMP_LIMIT


def test_the_clamp_has_no_step_in_value_or_speed_at_the_knee():
    """A kink at 0.6 would be a property of the dataset, not of any motion.

    The old hard gate had the worst version of this: everything past the
    threshold fell to zero. A clamp with a slope discontinuity is milder and
    still teaches the model that 0.6 is a real place.
    """
    knee = ROOT_XZ_SOFT_CLAMP_KNEE
    eps = 1e-5
    radii = np.array([knee - eps, knee, knee + eps])
    mapped = soft_clamp_extent(radii)

    # Continuous in value, and the identity right up to the knee.
    np.testing.assert_allclose(mapped[:2], radii[:2], atol=1e-12)
    slope_below = (mapped[1] - mapped[0]) / eps
    slope_above = (mapped[2] - mapped[1]) / eps
    assert slope_below == pytest.approx(1.0, abs=1e-4)
    assert slope_above == pytest.approx(slope_below, abs=1e-3)


def test_the_clamp_keeps_a_bigger_excursion_bigger():
    """Strictly increasing, which is what a hard clamp gives up.

    Capping at the ceiling would map a 1.0 lunge and a 10.0 fall onto the same
    root motion and leave the model no way to tell them apart.
    """
    radii = np.linspace(0.0, 40.0, num=2000)
    mapped = soft_clamp_extent(radii)

    assert np.all(np.diff(mapped) > 0.0)


def test_the_clamp_keeps_the_bearing_the_origin_and_the_closure():
    t = np.linspace(0.0, 2.0 * np.pi, num=61)
    # A circle of radius 2 that passes through the origin at both ends.
    traj = np.stack([2.0 * np.cos(t) - 2.0, 2.0 * np.sin(t)], axis=-1)

    clamped = soft_clamp_root_xz(traj)

    # Frame 0 is where the pipeline centred the clip, and it has to stay there.
    np.testing.assert_allclose(clamped[0], 0.0, atol=1e-12)
    # A loop that closed still closes, so tiling gains no seam.
    np.testing.assert_allclose(clamped[-1], clamped[0], atol=1e-9)
    away = np.linalg.norm(traj, axis=1) > 1e-9
    cosines = (clamped[away] * traj[away]).sum(axis=1) / (
        np.linalg.norm(clamped[away], axis=1) * np.linalg.norm(traj[away], axis=1)
    )
    np.testing.assert_allclose(cosines, 1.0, atol=1e-12)


def test_footwork_near_the_origin_survives_a_lunge_elsewhere_in_the_clip():
    """Why the map is per frame and not one scale factor for the whole clip.

    Scaling the trajectory by ``ceiling / extent`` would shrink these first
    thirty frames by the same 0.27 the lunge needs, and a single outlier frame
    would resize every clip around it.
    """
    near_origin = np.zeros((30, 2))
    near_origin[:, 0] = 0.2 * np.sin(np.linspace(0.0, 2.0 * np.pi, num=30))
    lunge = np.zeros((30, 2))
    lunge[:, 0] = 3.0 * np.sin(np.linspace(0.0, np.pi, num=30))
    traj = np.concatenate([near_origin, lunge], axis=0)

    clamped = soft_clamp_root_xz(traj)

    np.testing.assert_array_equal(clamped[:30], near_origin)
    assert np.linalg.norm(clamped, axis=1).max() < ROOT_XZ_SOFT_CLAMP_LIMIT


def test_a_lunge_is_bounded_rather_than_zeroed():
    features, _loop, flattened, _anim = _extract(
        _straight_line_anim(40, _closed_excursion(40, 2.5)),
        locomotion=False,
        clamp=True,
    )

    assert flattened is False
    assert ROOT_XZ_SOFT_CLAMP_KNEE < _reach(features) < ROOT_XZ_SOFT_CLAMP_LIMIT


def test_a_travelling_gait_is_flattened_before_it_is_clamped():
    """Order matters. This walk covers 1.48, far past the knee.

    Clamped first it would arrive at the flattener with every stride compressed
    into the ceiling and no longer looking like a gait; flattened first there is
    nothing left for the clamp to do at all.
    """
    features, _loop, flattened, _anim = _extract(
        _straight_line_anim(60, _gait_path(60, 20, stride=0.5, surge=0.05)),
        locomotion=True,
        clamp=True,
    )

    assert flattened is True
    assert _reach(features) < ROOT_XZ_SOFT_CLAMP_KNEE


def test_the_clamp_is_opt_in_so_re_extraction_cannot_compress_twice():
    """The clamp is not idempotent, which is why recovery paths do not run it.

    Retarget, the resample branch's second pass and the NPy round trip all
    re-extract features from an animation that has already been through it.
    """
    anim = _straight_line_anim(40, _closed_excursion(40, 2.0))
    once, _loop, _flattened, clamped_anim = _extract(anim, locomotion=False, clamp=True)

    again, _loop2, _flattened2, _anim2 = _extract(clamped_anim, locomotion=False)
    np.testing.assert_allclose(_root_path(again), _root_path(once), atol=1e-6)

    compressed_twice, _loop3, _flattened3, _anim3 = _extract(
        clamped_anim, locomotion=False, clamp=True,
    )
    assert _reach(compressed_twice) < _reach(once)


# ── locomotion's own extent bound ──────────────────────────────────────────

def test_the_locomotion_bound_scales_by_a_single_factor():
    """One factor for the whole clip, so the cycle keeps its shape.

    A per-frame radial map would compress the far half of every stride harder
    than the near half, changing what the surge looks like rather than its size.
    """
    t = np.linspace(0.0, np.pi, 40)
    path = np.stack([0.3 * np.sin(t), 0.1 * np.sin(4.0 * t)], axis=-1)
    # Pin one component exactly at zero, so the ratio below has to mask rather
    # than divide to nan -- otherwise the mask is only tested by luck.
    path[20, 0] = 0.0

    scaled = scale_root_xz_extent(path)

    assert float(np.linalg.norm(scaled, axis=1).max()) < ROOT_XZ_LOCOMOTION_LIMIT
    # One factor means every NONZERO component scales alike. A component sitting
    # exactly at zero would divide to nan and swallow the comparison, so the mask
    # is what makes the ratio well defined rather than a property of the fixture.
    nonzero = path[1:] != 0.0
    assert nonzero.any()
    ratios = scaled[1:][nonzero] / path[1:][nonzero]
    assert float(np.ptp(ratios)) < 1e-12
    np.testing.assert_allclose(scaled[0], path[0], atol=1e-12)


def test_the_locomotion_bound_leaves_a_small_gait_bit_for_bit_alone():
    """Below the knee it is the identity: most gaits detrend to well under it
    (p50 0.013 over the shipped locomotion clips) and must not be touched."""
    path = _closed_excursion(40, ROOT_XZ_LOCOMOTION_KNEE * 0.5)

    np.testing.assert_array_equal(scale_root_xz_extent(path), path)


def test_the_locomotion_bound_applies_even_when_nothing_was_flattened():
    """The bound is a property of BEING locomotion, not of having travelled.

    An in-place gait authored with a wide excursion never trips the drift gate,
    so no detrend runs -- and it is still held to the group's limit, which is
    what lets the validator state one extent invariant for every locomotion clip.
    """
    n_frames = 40
    path = _closed_excursion(n_frames, 0.30)

    features, _loop, flattened, _anim = _extract(
        _straight_line_anim(n_frames, path), locomotion=True, clamp=True
    )

    assert flattened is False
    assert ROOT_XZ_LOCOMOTION_KNEE < _reach(features) < ROOT_XZ_LOCOMOTION_LIMIT


def test_a_non_locomotion_clip_keeps_the_wide_bound():
    """The tighter bound is locomotion's alone: a lunge is still allowed to reach
    for the dataset-wide ceiling."""
    n_frames = 40
    path = _closed_excursion(n_frames, 3.0)

    features, _loop, _flattened, _anim = _extract(
        _straight_line_anim(n_frames, path), locomotion=False, clamp=True
    )

    assert ROOT_XZ_SOFT_CLAMP_KNEE < _reach(features) < ROOT_XZ_SOFT_CLAMP_LIMIT


# ── validator: the invariant preprocessing establishes ─────────────────────

def _run_drift_validator(motion, root_index, capsys, threshold=ROOT_XZ_DRIFT_THRESHOLD):
    from utils.validate_anytop_dataset import _validate_root_motion_drift

    parents = np.arange(-1, motion.shape[1] - 1, dtype=np.int64)
    _validate_root_motion_drift(
        motion, 'TestSkeleton', 'Clip_Test.npy', threshold, root_index,
        parents=parents, offsets=np.zeros((motion.shape[1], 3), dtype=np.float64),
    )
    return capsys.readouterr().out


def _motion_with_root_path(path_xz, root_index=1, joints=3):
    motion = np.zeros((path_xz.shape[0], joints, FEATS_LEN), dtype=np.float32)
    # Identity rotations: the validator reads the heading out of these channels,
    # and an all-zero block is not a rotation matrix.
    motion[:, :, 3:9] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    motion[:-1, root_index, 9] = np.diff(path_xz[:, 0])
    motion[:-1, root_index, 11] = np.diff(path_xz[:, 1])
    return motion


def test_validator_accepts_a_closed_excursion(capsys):
    motion = _motion_with_root_path(_closed_excursion(40, 1.5))
    assert _run_drift_validator(motion, 1, capsys) == ''


def test_validator_flags_a_clip_that_still_travels(capsys):
    motion = _motion_with_root_path(_travelling_path(40, distance=1.5))
    out = _run_drift_validator(motion, 1, capsys)
    assert 'still carries' in out


def test_validator_accepts_a_clip_the_pipeline_flattened(capsys):
    path = _gait_path(60, 20, stride=0.5, surge=0.05)
    flat, _drift = flatten_root_xz_drift(path, _flat_heading(60))
    assert _run_drift_validator(_motion_with_root_path(flat), 1, capsys) == ''


def test_validator_reports_a_root_index_the_features_disagree_with(capsys):
    from utils.validate_anytop_dataset import _validate_translation_root_feature_alignment

    motion = np.zeros((10, 3, 12), dtype=np.float32)
    motion[:, 1, 0] = 0.25

    assert not _validate_translation_root_feature_alignment(
        motion,
        'Clip_Test.npy',
        1,
    )
    assert 'not the joint the features were built around' in capsys.readouterr().out


def _run_ceiling_validator(motion, root_index, capsys):
    from utils.validate_anytop_dataset import _validate_root_xz_ceiling

    _validate_root_xz_ceiling(motion, 'Clip_Test.npy', root_index)
    return capsys.readouterr().out


def test_validator_accepts_a_clip_the_clamp_bounded(capsys):
    clamped = soft_clamp_root_xz(_closed_excursion(40, 3.0))

    assert _run_ceiling_validator(_motion_with_root_path(clamped), 1, capsys) == ''


def test_validator_flags_a_clip_that_never_went_through_the_clamp(capsys):
    out = _run_ceiling_validator(
        _motion_with_root_path(_closed_excursion(40, 3.0)), 1, capsys
    )

    assert 'past the soft clamp ceiling' in out


def test_validator_holds_a_locomotion_clip_to_the_tighter_bound(capsys):
    """The invariant the locomotion scale establishes, read straight off the
    tensor -- no decoder and no heading, unlike the drift check."""
    from utils.validate_anytop_dataset import _validate_root_xz_ceiling

    # Inside the dataset-wide ceiling, past what a locomotion clip may keep.
    motion = _motion_with_root_path(_closed_excursion(40, 0.5))

    assert _run_ceiling_validator(motion, 1, capsys) == ''

    _validate_root_xz_ceiling(
        motion, 'Clip_Test.npy', 1,
        limit=ROOT_XZ_LOCOMOTION_LIMIT,
        bound_name='locomotion extent bound',
    )
    assert 'locomotion extent bound' in capsys.readouterr().out


def test_validator_accepts_a_locomotion_clip_the_scale_bounded(capsys):
    from utils.validate_anytop_dataset import _validate_root_xz_ceiling

    motion = _motion_with_root_path(scale_root_xz_extent(_closed_excursion(40, 0.5)))

    _validate_root_xz_ceiling(
        motion, 'Clip_Test.npy', 1,
        limit=ROOT_XZ_LOCOMOTION_LIMIT,
        bound_name='locomotion extent bound',
    )
    assert capsys.readouterr().out == ''


def test_validator_checks_the_ceiling_on_every_clip_not_just_gaits(capsys):
    """The drift check is gated on the locomotion label; this one is not.

    A death that ends face down is allowed to drift and is still not allowed to
    reach past the ceiling, because nothing preprocessing produces ever does.
    """
    import inspect

    from utils import validate_anytop_dataset

    source = inspect.getsource(validate_anytop_dataset.validate_motion_files)
    ceiling_at = source.index('_validate_root_xz_ceiling(')
    gate_at = source.index('if motion_path.name in locomotion_clips:')
    assert ceiling_at < gate_at
