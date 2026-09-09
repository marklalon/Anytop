"""Root XZ: what a clip keeps, what it loses, and when the contacts are read.

See docs/root_xz_motion_refactor.md. No clip's root XZ is zeroed. A LOCOMOTION
clip whose cycle-window baseline ends somewhere other than where it started
loses that travel and keeps the within-cycle surge and sway; everything else --
a lunge, a death, a dodge, a swing -- keeps its root motion, bounded by a soft
clamp that is the identity within 0.6 of the origin and asymptotic to 0.8. Foot
contact is read off the motion as authored, ahead of any of it.
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
    ROOT_XZ_SOFT_CLAMP_KNEE,
    ROOT_XZ_SOFT_CLAMP_LIMIT,
    flatten_root_xz_drift,
    soft_clamp_extent,
    soft_clamp_root_xz,
)
from data_loaders.truebones.truebones_utils.animation_utils import (
    _transport_carrier_index,
    collapse_translation_root_chain,
    promote_translation_root_to_hierarchy_root,
    translation_root_subtree_mask,
    estimate_pose_cycle_length,
    root_xz_drift_correction,
    root_xz_trajectory,
    set_translation_root_xz,
)
from data_loaders.truebones.truebones_utils.features import (
    extract_motion_features_from_aligned_anims,
    get_contact_state,
)
from data_loaders.truebones.truebones_utils.motion_process import (
    move_xz_to_origin,
    root_xz_relative_pose,
)
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


def _extract(anim, foot_indices=(1,), vel_thresh=0.01, locomotion=True, clamp=False):
    features, _max_joints, motion_anim, _e, is_loop, flattened = extract_motion_features_from_aligned_anims(
        anim,
        anim,
        foot_contact_vel_thresh=vel_thresh,
        object_type='TestSkeleton',
        max_joints=8,
        foot_indices=list(foot_indices),
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


def _cyclic_pose(n_frames: int, cycle: int) -> np.ndarray:
    t = np.arange(n_frames, dtype=np.float64)
    return np.stack(
        [np.sin(2.0 * np.pi * t / cycle), np.cos(2.0 * np.pi * t / cycle), np.zeros(n_frames)],
        axis=-1,
    )[:, None, :]


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
    _flat, drift, _window = flatten_root_xz_drift(_root_path(features))
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
    correction, _drift = root_xz_drift_correction(_travelling_path(40), 40)
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
    pose = _cyclic_pose(60, 20)
    flat, drift, _window = flatten_root_xz_drift(path, pose=pose)
    assert drift > ROOT_XZ_DRIFT_THRESHOLD

    _again, second_drift, _w = flatten_root_xz_drift(flat, pose=pose)
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
        {"clip": "Wolf_Walk.npy", "action_group": "locomotion", "action_label": "walk"},
        {"clip": "Wolf_Die.npy", "action_group": "transition", "action_label": "die"},
    ]
    (tmp_path / 'action_labels.jsonl').write_text(
        os.linesep.join(json.dumps(row) for row in rows), encoding='utf-8'
    )
    assert load_locomotion_clip_names(tmp_path) == {'Wolf_Walk.npy'}
    with pytest.raises(FileNotFoundError):
        load_locomotion_clip_names(tmp_path / 'nope')


def test_the_validator_only_checks_locomotion_clips():
    import inspect

    from utils import validate_anytop_dataset

    source = inspect.getsource(validate_anytop_dataset.validate_motion_files)
    assert 'if motion_path.name in locomotion_clips:' in source


# ── the window: a curve is travel too ──────────────────────────────────────

def test_a_curved_path_is_flattened_far_better_than_a_line_fit_manages():
    """A WalkTurn arc has no straight trend to subtract.

    The cycle-window baseline follows the curve; a single end-to-end line fit
    leaves the whole arc behind, which is why the window exists at all.
    """
    n_frames = 60
    ang = np.linspace(0.0, np.pi, num=n_frames)
    arc = np.stack([1.2 * np.sin(ang / 2), 1.2 * (1.0 - np.cos(ang / 2))], axis=-1)

    flat, drift, window = flatten_root_xz_drift(arc, pose=_cyclic_pose(n_frames, 15))
    assert drift > ROOT_XZ_DRIFT_THRESHOLD
    assert window < n_frames

    line_fit_residual = np.linalg.norm(
        arc - np.outer(np.arange(n_frames) / (n_frames - 1), arc[-1]), axis=1
    ).max()
    windowed_residual = np.linalg.norm(flat - flat[0], axis=1).max()
    assert windowed_residual < 0.2 * line_fit_residual


def test_the_cycle_estimator_finds_a_period_and_declines_when_there_is_none():
    assert estimate_pose_cycle_length(_cyclic_pose(60, 20)) == 20

    aperiodic = np.linspace(0.0, 1.0, num=60)[:, None, None] * np.ones((1, 3, 3))
    assert estimate_pose_cycle_length(aperiodic) == 60


def test_a_window_is_never_shorter_than_a_quarter_of_the_clip():
    """A mis-estimate must not high-pass the motion itself away."""
    assert estimate_pose_cycle_length(_cyclic_pose(80, 8)) >= 20


# ── foot contact is read before the root is touched ────────────────────────

def _gait_rig(n_frames: int, cycle: int, stride: float, travels: bool) -> Animation:
    """A root and one foot: planted for half a cycle, swinging for the other half.

    ``travels`` chooses between the same gait authored travelling and authored
    in place. The foot's world-space behaviour is identical relative to the body
    either way, which is exactly why their contact labels must match.
    """
    t = np.arange(n_frames)
    phase = (t % cycle) / cycle
    plant = stride * (t // cycle)
    world_foot_x = plant + np.where(phase < 0.5, 0.0, stride * (phase - 0.5) * 2.0)
    root_x = stride * t / cycle

    parents = np.array([-1, 0], dtype=np.int64)
    offsets = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    rotations = Quaternions(np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_frames, 2, 1)))
    positions = np.zeros((n_frames, 2, 3), dtype=np.float64)
    if travels:
        positions[:, 0, 0] = root_x
    positions[:, 1, 0] = world_foot_x - root_x
    positions[:, 1, 1] = np.where(phase < 0.5, 0.0, 0.25)
    return Animation(rotations, positions, Quaternions.id(2), offsets, parents)


def test_contacts_survive_a_gait_fast_enough_to_break_them():
    """Flattening subtracts the gait speed from every joint.

    The median clip the old pipeline zeroed travelled 1.44 over 25 frames --
    0.058/frame against the 0.0447/frame the contact threshold allows -- so a
    planted foot stopped being planted and the stance phase lost its label.
    """
    n_frames, cycle, stride = 60, 20, 1.4
    anim = _gait_rig(n_frames, cycle, stride, travels=True)

    features, _loop, flattened, motion_anim = _extract(anim, vel_thresh=0.002)
    assert flattened is True
    assert features[:, 1, 12].sum() > 0

    measured_after = get_contact_state(positions_global(motion_anim), [1], 0.002)
    assert measured_after[:, 1].sum() == 0      # what the old order produced


def test_contacts_are_the_source_motions_own_labels():
    """Read off the motion as authored, not off what the re-seat leaves behind.

    That is also what makes a flattened gait agree with the ones an artist
    authored in place, which is the majority of this dataset's locomotion and
    does NOT slide its feet: measured per joint, authored-in-place locomotion
    carries 0.773 mean contact (truebones 0.755) against 0.820 (0.716) for the
    clips authored travelling. Labelling the travelling ones off the re-seated
    motion is what pulled them away from that.
    """
    n_frames, cycle, stride = 60, 20, 1.4
    anim = _gait_rig(n_frames, cycle, stride, travels=True)

    features, _loop, flattened, motion_anim = _extract(anim, vel_thresh=0.002)
    assert flattened is True

    expected = get_contact_state(positions_global(anim), [1], 0.002)
    np.testing.assert_array_equal(features[:-1, 1, 12], expected[:, 1])
    assert expected[:, 1].sum() > 0

    # And emphatically not the re-seated motion's labels, which are all zero.
    assert get_contact_state(positions_global(motion_anim), [1], 0.002)[:, 1].sum() == 0


def test_the_cycle_window_survives_an_intermediate_root():
    """The wrapper above an intermediate root does not move when the root is
    re-seated, so its de-rooted position carries the whole negated trajectory
    beforehand and nothing afterwards. Restricting the signal to the root's own
    subtree is what lets both sides of the edit read the same period.
    """
    from data_loaders.truebones.truebones_utils.animation_utils import (
        root_xz_relative_pose,
        translation_root_subtree_mask,
    )

    # The static-wrapper shape: there the re-seat really does move the root out
    # from under a joint that stays put, so the two sides of the edit disagree
    # about that joint unless it is masked out.
    anim = _static_wrapper_anim(40, _travelling_path(40))
    before = positions_global(anim)
    reseated = set_translation_root_xz(anim, 1, _closed_excursion(40, 0.2))
    after = positions_global(reseated)

    mask = translation_root_subtree_mask(anim.parents, 1)
    np.testing.assert_array_equal(mask, [False, True, True])

    np.testing.assert_allclose(
        root_xz_relative_pose(before, 1, parents=anim.parents),
        root_xz_relative_pose(after, 1, parents=anim.parents),
        atol=1e-9,
    )
    # Without the mask the wrapper joint alone moves the signal by the trajectory.
    unmasked_delta = np.abs(
        root_xz_relative_pose(before, 1) - root_xz_relative_pose(after, 1)
    ).max()
    assert unmasked_delta > 1.0


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
    removed and would read contact off feet that are already sliding, so the
    branch resamples the anims as AUTHORED and extracts once more from those.
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
        source, source, 0.01, 'TestSkeleton', 8, [2],
        Quaternions.id(1).qs[0], translation_root_index=1,
        flatten_root_travel=True,
    )
    assert first_flattened is True

    resampled = _resample_animation(source, 20)
    second, _mj, _m, _e, _loop2, second_flattened = extract_motion_features_from_aligned_anims(
        resampled, resampled, 0.01, 'TestSkeleton', 8, [2],
        Quaternions.id(1).qs[0], translation_root_index=1,
        flatten_root_travel=True,
    )

    assert second_flattened is True
    assert first.shape[0] == 18 and second.shape[0] == 20
    # Measured the way the dataset validator measures it: the decoder's own
    # reconstruction and the same pose signal, hence the same window.
    _flat, drift, _w = flatten_root_xz_drift(
        _root_path(second, root_index=1), pose=second[:, :, 0:3]
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
        anim, anim, 0.01, 'TestSkeleton', 8, [2],
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
    motion = np.zeros((n_frames, 2, 13), dtype=np.float32)
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
        max_joints=4, feature_len=13, latent_dim=8, ff_size=32, num_layers=1,
        num_heads=2, dropout=0.0, cross_limb=True, t5_out_dim=512,
    )
    assert not hasattr(model, 'root_xz_strip_projection')


def test_the_collate_no_longer_carries_the_flag():
    import torch
    from data_loaders.tensors import truebones_collate

    def _item():
        return {
            'inp': torch.zeros(3, 13, 5),
            'rest_pose': torch.zeros(3, 13),
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
    subtree = translation_root_subtree_mask(anim.parents, 1)
    np.testing.assert_allclose(after[:, subtree], before[:, subtree], atol=1e-9)


def test_the_wrapper_stops_carrying_the_negated_root_trajectory():
    """``Horse_Attack`` 0.718, ``Pirrana_Jump2`` 0.783: a wrapper that stands
    still while the root walks away reads, after de-rooting, as a joint moving
    backwards by exactly the trajectory."""
    anim = _static_wrapper_anim(40, _closed_excursion(40, 2.0))

    before = root_xz_relative_pose(positions_global(anim), 1)[:, 0][:, [0, 2]]
    after = root_xz_relative_pose(
        positions_global(collapse_translation_root_chain(anim, 1)), 1
    )[:, 0][:, [0, 2]]

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
            centred, centred, 0.01, 'TestSkeleton', 8, [2],
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


# ── validator: the invariant preprocessing establishes ─────────────────────

def _run_drift_validator(motion, root_index, capsys, threshold=ROOT_XZ_DRIFT_THRESHOLD):
    from utils.validate_anytop_dataset import _validate_root_motion_drift

    _validate_root_motion_drift(motion, 'TestSkeleton', 'Clip_Test.npy', threshold, root_index)
    return capsys.readouterr().out


def _motion_with_root_path(path_xz, root_index=1, joints=3):
    motion = np.zeros((path_xz.shape[0], joints, 13), dtype=np.float32)
    motion[:-1, root_index, 9] = np.diff(path_xz[:, 0])
    motion[:-1, root_index, 11] = np.diff(path_xz[:, 1])
    return motion


def test_validator_accepts_a_closed_excursion(capsys):
    motion = _motion_with_root_path(_closed_excursion(40, 1.5))
    assert _run_drift_validator(motion, 1, capsys) == ''


def test_validator_flags_a_clip_that_still_travels(capsys):
    motion = _motion_with_root_path(_travelling_path(40, distance=1.5))
    out = _run_drift_validator(motion, 1, capsys)
    assert 'baseline drifts' in out


def test_validator_accepts_a_clip_the_pipeline_flattened(capsys):
    path = _gait_path(60, 20, stride=0.5, surge=0.05)
    flat, _drift, _window = flatten_root_xz_drift(path, pose=_cyclic_pose(60, 20))
    assert _run_drift_validator(_motion_with_root_path(flat), 1, capsys) == ''


def test_validator_reports_a_root_index_the_features_disagree_with(capsys):
    from utils.validate_anytop_dataset import _validate_translation_root_feature_alignment

    motion = np.zeros((10, 3, 13), dtype=np.float32)
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
