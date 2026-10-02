"""Decomposer + EditRuntime mechanical invariants (section 6.1: T1, T1b, T2, T4-T7).

Usage:
    pytest tests/test_motion_edit_runtime.py

The clip-level tests decompose Truebones Horse clips and skip when that
dataset is not on disk.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion_edit.decompose import (
    ContactSource,
    chain_layer,
    decompose_motion,
    profile_subset,
    redecompose,
    split_root,
    with_contact_joints,
    with_contact_mask,
)
from motion_edit.ik import Limb, LimbSolver
from motion_edit.package import EditPackage, PackageVersionError, decode_json
from motion_edit.rotations import quat_from_rotvec, quat_inv, quat_mul, rotvec_from_quat
from utils.fullbody_ik import trunk_joint_indices
from motion_edit.runtime import (
    IMPLEMENTED,
    PARAM_SPECS,
    EditRuntime,
    UnsupportedParameterError,
    blend_between_keys,
    forward_kinematics,
    ground_displacement,
    ramped_envelope,
    sample_linear,
    soft_scale,
    yaw_quat,
)

ANYTOP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HORSE_ROOT = os.path.join(ANYTOP, "dataset", "truebones", "zoo", "truebones_processed")
HORSE_KEY = "truebones/zoo/Horse"

ROT_TOL = 1e-5   # rad
POS_TOL = 1e-6
BONE_TOL = 1e-6


def rotation_error(a: np.ndarray, b: np.ndarray) -> float:
    """Largest angle between corresponding rotations; atan2 stays exact for
    the slightly non-unit float32 quaternions a decode returns."""
    d = quat_mul(a, quat_inv(b))
    return float((2.0 * np.arctan2(np.linalg.norm(d[..., 1:], axis=-1), np.abs(d[..., 0]))).max())


# ── layer helpers (synthetic) ────────────────────────────────────────────────

def test_split_root_recombines_exactly():
    rng = np.random.default_rng(0)
    frames = 40
    pos = np.cumsum(rng.normal(size=(frames, 3)), axis=0)
    rot = quat_from_rotvec(rng.normal(size=(frames, 3)))
    for periodic, window in ((True, 13), (False, 15)):
        trend, osc, yaw, tilt = split_root(pos, rot, window=window, periodic=periodic)
        assert np.allclose(trend + osc, pos, atol=1e-12)
        assert rotation_error(quat_mul(yaw_quat(yaw), tilt), rot) < 1e-9
        # the tilt carries no rotation about world Y
        assert np.allclose(tilt[:, 2], 0.0, atol=1e-9)


def test_chain_layer_offsets_rebuild_rotations_and_stay_continuous():
    rng = np.random.default_rng(1)
    frames = 30
    rot = quat_from_rotvec(rng.normal(scale=0.4, size=(frames, 5, 3)))
    # joint 3 swings to 170 deg and back, joint 4 winds a whole turn per period
    swing = np.radians(170.0) * np.sin(np.linspace(0.0, np.pi, frames))
    rot[:, 3] = quat_from_rotvec(np.stack([np.zeros(frames), swing, np.zeros(frames)], -1))
    turn = 2.0 * np.pi * np.arange(frames) / frames
    rot[:, 4] = quat_from_rotvec(np.stack([np.zeros(frames), np.zeros(frames), turn], -1))
    for periodic in (True, False):
        reference, offsets, locked = chain_layer(rot, root=0, periodic=periodic)
        rebuilt = quat_mul(quat_from_rotvec(offsets), reference[None])
        assert rotation_error(rebuilt[:, 1:], rot[:, 1:]) < 1e-9
        assert np.all(offsets[:, 0] == 0.0)
        # half gain never steps further than the source does per frame
        half = quat_mul(quat_from_rotvec(0.5 * offsets), reference[None])
        step = lambda q: np.linalg.norm(rotvec_from_quat(quat_mul(q[1:], quat_inv(q[:-1]))), axis=-1)
        assert np.all(step(half)[:, 1:4] <= step(rot)[:, 1:4] + 1e-6)
        assert list(np.flatnonzero(locked)) == ([4] if periodic else [])


def test_trunk_is_the_ancestors_of_the_contact_limbs():
    #   0 hips - 1 spine - 2 chest - 3 neck - 4 head
    #            1 - 5 L thigh - 6 L foot      2 - 7 L arm - 8 L hand - 9 L finger
    #            1 - 10 R thigh - 11 R foot    3 - 12 jaw
    parents = [-1, 0, 1, 2, 3, 1, 5, 2, 7, 8, 1, 10, 3]
    sides = ["center"] * 5 + ["left", "left", "left", "left", "left", "right", "right", "center"]
    assert trunk_joint_indices(parents, sides, [6, 11]).tolist() == [0, 1]
    assert trunk_joint_indices(parents, sides, [6, 11, 9]).tolist() == [0, 1, 2]
    # a centre-line contact is its own limb root
    assert trunk_joint_indices(parents, sides, [12]).tolist() == [0, 1, 2, 3]
    assert trunk_joint_indices(parents, None, [6]).size == 0
    assert trunk_joint_indices(parents, sides, []).size == 0


def test_contact_source_provenance():
    source = ContactSource(cond=[4, 5, 8], species_add=[12], species_remove=[5],
                           package_add=[5, 13], package_remove=[8])
    assert source.used == [4, 5, 12, 13]
    assert ContactSource.from_dict(source.as_dict()).used == source.used


# ── clip-level invariants ────────────────────────────────────────────────────

requires_horse = pytest.mark.skipif(
    not os.path.isfile(os.path.join(HORSE_ROOT, "cond.npy")), reason="Truebones zoo dataset not on disk")


def _decompose(clip: str, is_loop: bool, action_group: str, action_label: str, stretch: float = 0.1):
    from motion_edit.profile.data import load_profiles

    cond = np.load(os.path.join(HORSE_ROOT, "cond.npy"), allow_pickle=True).item()[HORSE_KEY]
    features = np.load(os.path.join(HORSE_ROOT, "motions", clip + ".npy"))
    if is_loop:
        from data_loaders.truebones.data.dataset import _drop_loop_closing_frame
        features = _drop_loop_closing_frame(features)
    profile = profile_subset(load_profiles(HORSE_ROOT).get(HORSE_KEY), cond, action_label)
    return decompose_motion(features, cond, object_type=HORSE_KEY, clip_name=clip, is_loop=is_loop,
                            action_group=action_group, action_label=action_label,
                            stretch_factor=stretch, profile=profile)


def _reference(package: EditPackage):
    from motion_lib.Animation import positions_global
    from utils.npy_restore import build_skeleton_only_context, restore_animation_from_features

    cond = decode_json(package["source_cond"])
    restored = restore_animation_from_features(
        package["source_features"], build_skeleton_only_context(cond), restore_space="hml",
        fullbody_ik=package.manifest["fullbody_ik"], stretch_factor=package.manifest["stretch_factor"],
        fps=package.fps)
    return restored.animation.rotations.qs, positions_global(restored.animation)


@pytest.fixture(scope="module")
def horse_packages():
    return {
        "loop": _decompose("Horse_RunLoop", True, "locomotion", "run, forward"),
        "one_shot": _decompose("Horse_Jumping", False, "stationary", "attack, jump, smash"),
    }


@requires_horse
@pytest.mark.parametrize("kind", ["loop", "one_shot"])
@pytest.mark.parametrize("compose", [False, True], ids=["T1_replay", "T1b_roundtrip"])
def test_default_params_replay_the_decode(horse_packages, kind, compose):
    package = horse_packages[kind]
    ref_rot, ref_pos = _reference(package)
    result = EditRuntime(package).apply(compose=compose)
    assert result.composed == compose
    assert rotation_error(result.animation.rotations.qs, ref_rot) < ROT_TOL
    assert np.abs(result.global_positions - ref_pos).max() < POS_TOL


@requires_horse
@pytest.mark.parametrize("compose", [False, True])
def test_T2_deterministic(horse_packages, compose):
    runtime = EditRuntime(horse_packages["loop"])
    a = runtime.apply(compose=compose)
    b = runtime.apply(compose=compose)
    assert np.array_equal(a.animation.rotations.qs, b.animation.rotations.qs)
    assert np.array_equal(a.animation.positions, b.animation.positions)
    assert np.array_equal(a.global_positions, b.global_positions)


@requires_horse
@pytest.mark.parametrize("compose", [False, True])
def test_T5_bone_lengths_match_package(horse_packages, compose):
    package = horse_packages["one_shot"]
    result = EditRuntime(package).apply(compose=compose)
    parents = package["parents"]
    child = np.flatnonzero(parents >= 0)
    bones = np.linalg.norm(result.global_positions[:, child] - result.global_positions[:, parents[child]], axis=-1)
    expected = np.linalg.norm(package["base_pos"][:, child], axis=-1)
    assert np.abs(bones - expected).max() < BONE_TOL


@requires_horse
def test_package_round_trips_through_disk(horse_packages, tmp_path):
    package = horse_packages["loop"]
    package.save(str(tmp_path / "clip.edit"))
    loaded = EditPackage.load(str(tmp_path / "clip.edit"))
    for compose in (False, True):
        a = EditRuntime(package).apply(compose=compose)
        b = EditRuntime(loaded).apply(compose=compose)
        assert np.array_equal(a.global_positions, b.global_positions)
    loaded.manifest["runtime_version"] = -1
    loaded.save(str(tmp_path / "old.edit"))
    with pytest.raises(PackageVersionError):
        EditPackage.load(str(tmp_path / "old.edit"))


@requires_horse
def test_redecompose_with_zero_stretch_is_rigid(horse_packages):
    package = horse_packages["one_shot"]
    rigid = redecompose(package, 0.0)
    assert rigid.manifest["stretch_factor"] == 0.0
    assert rigid.manifest["contacts"] == package.manifest["contacts"]
    # the trunk and the limbs' attachment offsets keep the decode; every other bone is rigid
    parents = rigid["parents"]
    cond = decode_json(rigid["source_cond"])
    kept = {rigid.root, *trunk_joint_indices(parents, cond["joint_side_labels"],
                                             cond["contact_joints"]).tolist()}
    child = np.flatnonzero((parents >= 0) & ~np.isin(parents, sorted(kept)))
    rest = np.linalg.norm(rigid["anim_offsets"][child], axis=-1)
    lengths = np.linalg.norm(rigid["base_pos"][:, child], axis=-1)
    assert np.abs(lengths - rest).max() < 1e-5
    ref_rot, ref_pos = _reference(rigid)
    result = EditRuntime(rigid).apply(compose=True)
    assert rotation_error(result.animation.rotations.qs, ref_rot) < ROT_TOL
    assert np.abs(result.global_positions - ref_pos).max() < POS_TOL


@requires_horse
def test_redecompose_without_fullbody_ik(horse_packages):
    package = horse_packages["loop"]
    free = redecompose(package, fullbody_ik=False)
    assert free.manifest["fullbody_ik"] is False
    assert free.manifest["stretch_factor"] == package.manifest["stretch_factor"]
    assert free.manifest["diagnostics"]["ik_error"] is None
    # the decode's own per-frame translations: bones are not held near their rest length
    child = np.flatnonzero(free["parents"] >= 0)
    rest = np.linalg.norm(free["anim_offsets"][child], axis=-1)
    lengths = np.linalg.norm(free["base_pos"][:, child], axis=-1)
    assert np.abs(lengths / rest - 1).max() > package.manifest["stretch_factor"] + 0.1
    ref_rot, ref_pos = _reference(free)
    for compose in (False, True):
        result = EditRuntime(free).apply(compose=compose)
        assert rotation_error(result.animation.rotations.qs, ref_rot) < ROT_TOL
        assert np.abs(result.global_positions - ref_pos).max() < POS_TOL
    # a stretch-only re-decompose keeps the package's IK choice
    assert redecompose(free, 0.3).manifest["fullbody_ik"] is False



@requires_horse
def test_contacts_and_events_are_detected(horse_packages):
    loop = horse_packages["loop"]
    assert loop.manifest["profile"]["status"] == "ok"
    assert loop.manifest["events"]["period"] == pytest.approx(loop.frame_count, abs=1)
    assert len(loop["plant_intervals"]) > 0
    mask = loop["contact_mask"]
    for k, start, end in loop["plant_intervals"]:
        frames = np.arange(start, end) % loop.frame_count
        assert mask[frames, k].all()
        assert (loop["plant_id"][frames, k] >= 0).all()
    assert "stride" in loop.manifest["available_params"]
    assert "force" not in loop.manifest["available_params"]
    assert "force" in horse_packages["one_shot"].manifest["available_params"]


@requires_horse
def test_params_are_refused_not_ignored(horse_packages):
    runtime = EditRuntime(horse_packages["loop"])
    assert not runtime.apply({"tempo": 1.0, "foot_lock": False}).composed
    with pytest.raises(ValueError):
        runtime.apply({"no_such_param": 1.0})
    with pytest.raises(UnsupportedParameterError):
        runtime.apply({"force": 1.5})   # one-shot only
    with pytest.raises(UnsupportedParameterError):
        EditRuntime(horse_packages["one_shot"]).apply({"force": 1.5})   # not implemented yet
    with pytest.raises(UnsupportedParameterError):
        runtime.apply({"jump_height": 1.5})   # not on a loop


# ── edits (M3): T4-T7 ────────────────────────────────────────────────────────

def _edit_cases(package) -> list[dict]:
    """Every implemented slider of the clip at both ends of its range, and foot_lock."""
    cases = []
    for name in package.manifest["available_params"]:
        if name not in IMPLEMENTED:
            continue
        spec = PARAM_SPECS[name]
        cases += [{name: True}] if spec.toggle else [{name: spec.low}, {name: spec.high}]
    return cases


def _plant_drift(runtime: EditRuntime, result, *, pivots_only: bool = False,
                 pivot_runs: bool = False) -> dict[tuple, tuple]:
    """Per package plant interval: ``(drift, baseline)``, the largest XZ distance from the mean
    in the ground frame, of the result and of the original clip over the same source times
    (the planted frames IK could reach).  ``pivot_runs`` measures each unbroken run of a
    plant's pivot frames on its own (keyed ``(plant, first frame)``)."""
    pkg = runtime.package
    disp = ground_displacement(pkg["ground_velocity"], pkg.fps, pkg.is_loop)
    original = runtime.original_positions()
    reached = ~result.unreached.any(axis=1)
    out = {}
    for k, joint in enumerate(pkg["contact_joints"]):
        on = (result.plant_id[:, k] >= 0) & reached
        if pivots_only:
            on &= result.pivot[:, k]
        if not on.any():
            continue
        u = result.plant_time[on, k]
        d_u = sample_linear(disp, u, False)
        edited = result.global_positions[on, joint][:, [0, 2]] + result.stride_factor * d_u
        source = sample_linear(original[:, joint], u, pkg.is_loop)[:, [0, 2]] + d_u
        frames = np.flatnonzero(on)
        for plant in np.unique(result.plant_id[on, k]):
            sel = result.plant_id[on, k] == plant
            groups = [(int(plant), sel)]
            if pivot_runs:
                idx = np.flatnonzero(sel)
                breaks = np.flatnonzero(np.diff(frames[idx]) > 1) + 1
                groups = [((int(plant), int(frames[run[0]])), np.isin(np.arange(len(sel)), run))
                          for run in np.split(idx, breaks)]
            for key, sel in groups:
                spread = [float(np.linalg.norm(x[sel] - x[sel].mean(axis=0), axis=-1).max())
                          for x in (edited, source)]
                out[key] = tuple(spread)
    return out


def _locking_pivot_baseline(runtime: EditRuntime, result) -> dict[int, float]:
    """Per package plant: the largest original drift of a plant that pivoted its foot on
    a frame it was planted."""
    pivots = _plant_drift(runtime, result, pivots_only=True)
    columns_of = {c: limb.columns for limb in runtime.limbs()[0] for c in limb.columns}
    out = {}
    for k in range(result.plant_id.shape[1]):
        for f in np.flatnonzero(result.plant_id[:, k] >= 0):
            plant = int(result.plant_id[f, k])
            for c in columns_of.get(k, [k]):
                if result.pivot[f, c]:
                    base = pivots.get(int(result.plant_id[f, c]), (0.0, 0.0))[1]
                    out[plant] = max(out.get(plant, 0.0), base)
    return out


@requires_horse
@pytest.mark.parametrize("kind", ["loop", "one_shot"])
def test_T5_T6_edits_keep_bones_and_plants(horse_packages, kind):
    package = horse_packages[kind]
    runtime = EditRuntime(package)
    leg = runtime.leg
    parents = package["parents"]
    child = np.flatnonzero(parents >= 0)
    legs = sorted({j for limb in runtime.limbs()[0] for j in limb.chain[1:] + [limb.foot]})
    rigid = np.setdiff1d(child, legs)
    for params in _edit_cases(package):
        result = runtime.apply(params)
        # T5: every bone is as long as the package's (time-sampled) local translation, except
        # the leg chains soft_stretch scales, which stay within its limit of it
        sampled = sample_linear(np.asarray(package["base_pos"], dtype=np.float64), result.source_time,
                                package.is_loop)
        ratio = np.linalg.norm(result.global_positions[:, child]
                               - result.global_positions[:, parents[child]], axis=-1) / np.maximum(
            np.linalg.norm(sampled[:, child], axis=-1), 1e-12)
        limit = result.params["soft_stretch"]
        assert np.abs(ratio - 1.0).max() <= limit + BONE_TOL, params
        assert np.abs(ratio[:, np.isin(child, rigid)] - 1.0).max() < BONE_TOL, params
        off = runtime.apply({**params, "soft_stretch": 0.0})
        off_bones = np.linalg.norm(off.global_positions[:, child] - off.global_positions[:, parents[child]], axis=-1)
        assert np.abs(off_bones - np.linalg.norm(sampled[:, child], axis=-1)).max() < BONE_TOL, params
        # T6: no plant slides more than the original did.  IK pins the pivot; the foot's other
        # contacts follow it rigidly, and an amplitude edit that flattens the toes' own roll
        # (amp.legs = 0) leaves them a little slip of their own, one that amplifies it
        # (amp.legs > 1) carries a joint rolling over the pivot proportionally further.
        # foot_lock moves a foot's contacts by its pivot's slip, so a contact rolling over
        # the pivot may spread by as much as that slip on top of its own.
        roll = max(1.0, result.params.get("amp.legs", 1.0))
        locked = _locking_pivot_baseline(runtime, result) if result.params.get("foot_lock") else {}
        for plant, (drift, baseline) in _plant_drift(runtime, result).items():
            allowed = roll * baseline + locked.get(plant, 0.0) + 0.015 * leg
            assert drift <= allowed, (params, plant, drift, baseline)
        for plant, (drift, baseline) in _plant_drift(runtime, result, pivots_only=True).items():
            assert drift <= baseline + 1e-4 * leg, (params, plant, drift, baseline)


@requires_horse
@pytest.mark.parametrize("kind", ["loop", "one_shot"])
def test_T6_foot_lock_pins_the_pivots(horse_packages, kind):
    """The pivot is the foot's deepest planted contact; it holds still while it stays the
    pivot.  A heel that is the pivot again after its toe lifts has rolled about the toe
    in between, so each unbroken run is measured on its own."""
    runtime = EditRuntime(horse_packages[kind])
    for extra in ({}, {"bounce": 1.5}, {"amp.legs": 1.4}):
        result = runtime.apply({"foot_lock": True, **extra})
        drift = _plant_drift(runtime, result, pivots_only=True, pivot_runs=True)
        assert drift and max(d for d, _ in drift.values()) < 1e-4 * runtime.leg, extra


@requires_horse
@pytest.mark.parametrize("kind", ["loop", "one_shot"])
def test_T6_foot_lock_keeps_the_foot_shape(horse_packages, kind):
    """foot_lock removes one slip per foot: every planted contact keeps its original offset
    from the foot's reference contact, so a heel peeling up about a planted toe still peels."""
    runtime = EditRuntime(horse_packages[kind])
    package = runtime.package
    original = runtime.original_positions()
    result = runtime.apply({"foot_lock": True})
    peeled = 0
    for limb in runtime.limbs()[0]:
        cols = limb.columns
        for f in range(result.plant_id.shape[0]):
            planted = [c for c in cols if result.plant_id[f, c] >= 0]
            if len(planted) < 2:
                continue
            ref = max(planted, key=lambda c: limb.depth[limb.contacts[cols.index(c)]])
            u = result.plant_time[f, ref]
            base = sample_linear(original[:, package["contact_joints"][ref]], np.array([u]), package.is_loop)[0]
            for c in planted:
                offset = sample_linear(original[:, package["contact_joints"][c]],
                                       np.array([result.plant_time[f, c]]), package.is_loop)[0] - base
                assert np.allclose(result.plant_target[f, c] - result.plant_target[f, ref], offset,
                                   atol=1e-9), (f, c)
                peeled += int(abs(offset[1]) > 0.0)
    assert peeled


@requires_horse
def test_T4_loop_seam_is_an_ordinary_step(horse_packages):
    package = horse_packages["loop"]
    runtime = EditRuntime(package)
    for params in [{}] + _edit_cases(package):
        pos = runtime.apply(params, compose=True).global_positions
        steps = np.linalg.norm(np.diff(pos, axis=0), axis=-1).max(axis=1)
        wrap = np.linalg.norm(pos[0] - pos[-1], axis=-1).max()
        assert wrap <= steps.max() + 1e-9, params


@requires_horse
@pytest.mark.parametrize("kind", ["loop", "one_shot"])
def test_T7_sliders_are_continuous_at_their_defaults(horse_packages, kind):
    package = horse_packages[kind]
    runtime = EditRuntime(package)
    reference = runtime.apply().global_positions
    for name in package.manifest["available_params"]:
        spec = PARAM_SPECS[name]
        if name not in IMPLEMENTED or spec.toggle:
            continue
        eps = 1e-3 * (spec.high - spec.low)
        for value in (spec.default - eps, spec.default + eps):
            pos = runtime.apply({name: value}).global_positions
            n = min(len(pos), len(reference))
            change = np.abs(pos[:n] - reference[:n]).max() / runtime.leg
            assert change <= 20.0 * eps, (name, value, change)


@requires_horse
def test_T2_deterministic_with_edits(horse_packages):
    runtime = EditRuntime(horse_packages["loop"])
    params = {"stride": 1.3, "bounce": 1.4, "amp.legs": 1.2, "tempo": 1.25, "foot_lock": True}
    a, b = runtime.apply(params), runtime.apply(params)
    assert np.array_equal(a.animation.rotations.qs, b.animation.rotations.qs)
    assert np.array_equal(a.global_positions, b.global_positions)


@requires_horse
def test_tempo_keeps_whole_loop_periods(horse_packages):
    package = horse_packages["loop"]
    runtime = EditRuntime(package)
    for tempo in (0.5, 0.8, 1.3, 2.0):
        result = runtime.apply({"tempo": tempo})
        count = len(result.source_time)
        assert count == max(2, round(package.frame_count / tempo))
        assert np.allclose(np.diff(result.source_time), package.frame_count / count)
    one_shot = EditRuntime(horse_packages["one_shot"]).apply({"tempo": 2.0})
    assert one_shot.source_time[-1] <= horse_packages["one_shot"].frame_count - 1
    assert np.allclose(np.diff(one_shot.source_time), 2.0)


@requires_horse
def test_tempo_and_stride_scale_the_ground_speed(horse_packages):
    package = horse_packages["loop"]
    runtime = EditRuntime(package)
    base = np.linalg.norm(runtime.apply().ground_velocity, axis=-1).mean()
    stride = runtime.apply({"stride": 1.3})
    assert stride.stride_factor == pytest.approx(1.3)
    assert np.linalg.norm(stride.ground_velocity, axis=-1).mean() == pytest.approx(1.3 * base, rel=1e-9)
    both = runtime.apply({"tempo": 1.2, "stride": 1.3})
    rate = package.frame_count / len(both.source_time)         # whole-frame loop period
    assert both.stride_factor == pytest.approx(1.3)
    assert np.linalg.norm(both.ground_velocity, axis=-1).mean() == pytest.approx(
        1.3 * rate * base, rel=0.02)
    with pytest.raises(ValueError):
        runtime.apply({"speed": 1.5})   # ground speed is tempo x stride, not a parameter of its own


@requires_horse
def test_amplitude_and_root_layers_act_where_named(horse_packages):
    package = horse_packages["loop"]
    runtime = EditRuntime(package)
    base = runtime.apply(compose=True)
    groups = np.asarray([str(g) for g in package["chain_group"]])
    tail = groups == "tail"
    moved = np.abs(runtime.apply({"amp.tail": 1.8}).global_positions - base.global_positions).max(axis=(0, 2))
    assert moved[tail].max() > 1e-3
    assert moved[~tail].max() < 1e-9          # the tail carries no foot: nothing else moves
    # bounce 0 leaves the root on its trend, lowered only where a foot could not reach
    result = runtime.apply({"bounce": 0.0})
    lowered = package["root_trend"][:, 1] - result.animation.positions[:, package.root, 1]
    assert lowered.min() > -1e-9
    if not any(d["kind"] == "ground" for d in result.diagnostics):
        assert lowered.max() < 1e-9


def test_limb_solver_reaches_and_keeps_bone_lengths():
    # a 3-bone planar leg hanging down -Y; ask the foot to move forward and up
    parents = np.array([-1, 0, 1, 2, 3])
    limb = Limb(root=1, foot=4, chain=[1, 2, 3], contacts=[4], columns=[0], depth={4: 0})
    solver = LimbSolver(limb, parents)
    frames = 5
    ident = np.tile([1.0, 0.0, 0.0, 0.0], (frames, 3, 1))
    bend = quat_from_rotvec(np.array([0.3, 0.0, 0.0]))
    local_rot = ident.copy()
    local_rot[:, 1] = bend
    local_pos = np.tile(np.array([[0.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, -0.5, 0.0], [0.0, -0.4, 0.0]]),
                        (frames, 1, 1))
    parent_rot = np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1))
    parent_pos = np.zeros((frames, 3))
    _, _, foot = solver._fk(parent_rot, parent_pos, local_rot, local_pos)
    target = foot + np.linspace(0.0, 0.15, frames)[:, None] * np.array([0.0, 1.0, 0.5])
    solved, error = solver.solve(parent_rot, parent_pos, local_rot, local_pos, target, 1.4)
    assert error.max() < 1e-6
    _, poss, foot_after = solver._fk(parent_rot, parent_pos, solved, local_pos)
    assert np.allclose(np.linalg.norm(poss[1] - poss[0], axis=-1), 0.5)
    assert np.allclose(np.linalg.norm(foot_after - poss[2], axis=-1), 0.4)
    assert np.allclose(solved[0], local_rot[0])      # no change asked, none made


def test_limb_solver_settles_on_targets_out_of_reach():
    """A target past full extension leaves the leg straight toward it, the same pose
    whatever the iteration budget, missing by exactly the distance it is out of reach."""
    import motion_edit.ik as ik

    parents = np.array([-1, 0, 1, 2, 3])
    limb = Limb(root=1, foot=4, chain=[1, 2, 3], contacts=[4], columns=[0], depth={4: 0})
    solver = LimbSolver(limb, parents)
    frames = 4
    local_rot = np.tile([1.0, 0.0, 0.0, 0.0], (frames, 3, 1))
    local_rot[:, 1] = quat_from_rotvec(np.array([0.4, 0.0, 0.0]))
    local_pos = np.tile(np.array([[0.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, -0.5, 0.0], [0.0, -0.4, 0.0]]),
                        (frames, 1, 1))
    parent_rot = np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1))
    parent_pos = np.zeros((frames, 3))
    direction = np.array([0.0, -0.8, 0.6])
    distance = np.array([1.4, 1.5, 1.7, 2.2])               # chain length is 1.4
    target = distance[:, None] * direction
    reach = ik.STRAIGHT_RATIO * 1.4
    feet = []
    for iterations in (40, 100, 400):
        old, ik.ITERATIONS = ik.ITERATIONS, iterations
        try:
            solved, error = solver.solve(parent_rot, parent_pos, local_rot, local_pos, target, 1.4)
        finally:
            ik.ITERATIONS = old
        assert np.allclose(error, distance - reach, atol=1e-5), iterations
        feet.append(solver._fk(parent_rot, parent_pos, solved, local_pos)[2])
    assert np.allclose(feet[0], feet[1], atol=1e-5) and np.allclose(feet[1], feet[2], atol=1e-5)
    assert np.allclose(feet[1], reach * direction, atol=1e-5)


def test_blend_between_keys_wraps_for_loops():
    values = np.zeros((8, 1))
    keyed = np.zeros(8, dtype=bool)
    keyed[[1, 5]] = True
    values[1], values[5] = 1.0, 3.0
    loop = blend_between_keys(values, keyed, True)[:, 0]
    assert loop[3] == pytest.approx(2.0) and loop[7] == pytest.approx(2.0)
    held = blend_between_keys(values, keyed, False)[:, 0]
    assert held[0] == 1.0 and held[7] == 3.0


def test_soft_scale_band_and_limits():
    own = np.full(7, 0.9)
    need = np.array([0.5, 0.75, 0.81, 0.9, 0.96, 1.05, 1.5])
    scale = soft_scale(need, own, 0.1)
    assert (scale[2:5] == 1.0).all()                     # comfortable band: untouched
    assert scale[0] < scale[1] < 1.0 < scale[5] < scale[6]
    assert scale.min() > 0.9 and scale.max() < 1.1        # eased into the limit
    assert (soft_scale(need, own, 0.0) == 1.0).all()
    # a pose already straighter than the stretch start is its own band edge
    assert soft_scale(np.array([0.99]), np.array([0.99]), 0.1)[0] == 1.0


@requires_horse
def test_soft_stretch_takes_over_from_lowering(horse_packages):
    """A raised posture is reached by lengthening the legs first; the body is lowered only
    for what the limit cannot cover, and a lowered posture shortens them."""
    runtime = EditRuntime(horse_packages["loop"])
    package = runtime.package

    def lowered(result):
        return max([float(d["message"].split("up to ")[1].split(" ")[0])
                    for d in result.diagnostics if d["kind"] == "ground"] or [0.0])

    def leg_ratio(result):
        legs = [limb.chain[1] for limb in runtime.limbs()[0] if len(limb.chain) > 1]
        sampled = sample_linear(np.asarray(package["base_pos"], dtype=np.float64), result.source_time,
                                package.is_loop)
        return (np.linalg.norm(result.animation.positions[:, legs], axis=-1)
                / np.linalg.norm(sampled[:, legs], axis=-1))

    up = runtime.apply({"posture": 0.2, "soft_stretch": 0.2})
    up_rigid = runtime.apply({"posture": 0.2, "soft_stretch": 0.0})
    assert lowered(up) < 0.5 * lowered(up_rigid)
    assert leg_ratio(up).max() > 1.05
    down = runtime.apply({"posture": -0.3})
    assert leg_ratio(down).min() < 0.95
    assert any(d["kind"] == "stretch" for d in down.diagnostics)


def test_ramped_envelope_covers_and_eases():
    values = np.zeros(20)
    values[10] = 1.0
    out = ramped_envelope(values, 4, False)
    assert (out >= values).all() and out[10] == 1.0
    assert out[5] == 0.0 and out[15] == 0.0 and 0.0 < out[6] < out[8] < out[9] < 1.0
    assert np.abs(np.diff(out)).max() < 0.5          # no one-frame drop
    loop = ramped_envelope(np.roll(values, 9), 4, True)    # peak on frame 19 wraps to frame 0
    assert loop[0] == pytest.approx(out[11]) and loop[3] == pytest.approx(out[14])


def test_forward_kinematics_matches_motion_lib():
    from motion_lib.Animation import Animation, positions_global
    from motion_lib.Quaternions import Quaternions

    rng = np.random.default_rng(3)
    parents = np.array([-1, 0, 1, 1, 3])
    rot = quat_from_rotvec(rng.normal(scale=0.5, size=(6, 5, 3)))
    pos = rng.normal(size=(6, 5, 3))
    _, ours = forward_kinematics(parents, rot, pos)
    anim = Animation(Quaternions(rot), pos, Quaternions.id(0), pos[0].copy(), parents)
    assert np.allclose(ours, positions_global(anim), atol=1e-12)


# ── contact edits ─────────────────────────────────────────────────────────────

@requires_horse
def test_contact_mask_edit_rebuilds_plants(horse_packages):
    package = horse_packages["one_shot"]
    same = with_contact_mask(package, package["contact_mask"])
    assert same.manifest["contacts"]["intervals_edited"]
    assert np.array_equal(same["plant_intervals"], package["plant_intervals"])
    mask = np.array(package["contact_mask"], copy=True)
    mask[:, 0] = False
    mask[10:20, 0] = True
    edited = with_contact_mask(package, mask)
    rows = edited["plant_intervals"]
    assert [tuple(r) for r in rows if r[0] == 0] == [(0, 10, 20)]
    result = EditRuntime(edited).apply({"foot_lock": True})
    drift = _plant_drift(EditRuntime(edited), result, pivots_only=True, pivot_runs=True)
    assert max(d for d, _ in drift.values()) < 1e-4 * EditRuntime(edited).leg
    with pytest.raises(ValueError):
        with_contact_mask(package, mask[:, 1:])


@requires_horse
@pytest.mark.parametrize("kind", ["loop", "one_shot"])
def test_edits_without_contact_joints_skip_the_plants(horse_packages, kind):
    edited = with_contact_joints(horse_packages[kind], [])
    runtime = EditRuntime(edited)
    for params in ({"amp.legs": 0.5}, {"bounce": 1.5}):
        result = runtime.apply(params)
        assert result.composed and result.plant_id.shape[1] == 0
    with pytest.raises(UnsupportedParameterError):
        runtime.apply({"foot_lock": True})


@requires_horse
def test_contact_joint_edit_records_provenance(horse_packages):
    package = horse_packages["one_shot"]
    joints = list(package["contact_joints"])
    removed = joints[0]
    names = [str(n) for n in package["names"]]
    added = names.index("Bip01_Head") if "Bip01_Head" in names else int(package["parents"][joints[0]])
    edited = with_contact_joints(package, joints[1:] + [added])
    source = edited.manifest["contacts"]["source"]
    assert source["package_remove"] == [removed]
    assert source["package_add"] == [added]
    assert added in list(edited["contact_joints"]) and removed not in list(edited["contact_joints"])
    # writing the change into the species override moves it out of the package's own edits
    species = with_contact_joints(package, joints[1:] + [added],
                                  species_add=[added], species_remove=[removed])
    s2 = species.manifest["contacts"]["source"]
    assert s2["package_add"] == [] and s2["package_remove"] == []
    assert list(species["contact_joints"]) == list(edited["contact_joints"])


@requires_horse
def test_input_notes_survive_rebuilds(horse_packages):
    package = horse_packages["one_shot"]
    note = {"kind": "contacts", "message": "override row is stale"}
    package = EditPackage(dict(package.manifest, input_notes=[note]), package.arrays)
    for rebuilt in (redecompose(package), with_contact_mask(package, package["contact_mask"]),
                    with_contact_joints(package, list(package["contact_joints"]))):
        assert note in rebuilt.manifest["diagnostics"]["items"]
        assert rebuilt.manifest["input_notes"] == [note]
    # a rewritten species override voids the notes about the old one
    joints = list(package["contact_joints"])
    rewritten = with_contact_joints(package, joints, species_add=[], species_remove=[])
    assert note not in rewritten.manifest["diagnostics"]["items"]


def test_serve_refuses_non_json_posts(tmp_path):
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer

    from motion_edit.ui.serve import Handler, PackageStore

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.store = PackageStore(str(tmp_path))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        def post(ctype):
            request = urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}/api/contacts", method="POST",
                data=b'{"id": "x", "joints": [0], "species": true}', headers={"Content-Type": ctype})
            try:
                return urllib.request.urlopen(request).status
            except urllib.error.HTTPError as exc:
                return exc.code
        assert post("text/plain") == 415
        assert post("application/json; charset=utf-8") == 404     # parsed; no such package
    finally:
        server.shutdown()
        server.server_close()
