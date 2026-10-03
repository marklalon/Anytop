"""Decomposer + EditRuntime mechanical invariants (section 6.1: T1, T1b, T2, T4-T7).

Usage:
    pytest tests/test_motion_edit_runtime.py

The clip-level tests decompose Truebones Horse and Trex clips and skip when
that dataset is not on disk.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.truebones.truebones_utils.param_utils import FPS
from motion_edit.decompose import (
    TREND_SECONDS,
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
    FORCE_SLOW_EXPONENT,
    IMPLEMENTED,
    PARAM_SPECS,
    STRIKE_LEAN,
    STRIKE_SHIFT,
    EditRuntime,
    UnsupportedParameterError,
    blend_between_keys,
    forward_kinematics,
    ground_displacement,
    pchip,
    ramped_envelope,
    sample_linear,
    soft_scale,
    strike_shapes,
    yaw_quat,
)

ANYTOP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HORSE_ROOT = os.path.join(ANYTOP, "dataset", "truebones", "zoo", "truebones_processed")
HORSE_KEY = "truebones/zoo/Horse"
TREX_KEY = "truebones/zoo/Trex"

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


def test_loop_root_trend_keeps_a_lunge():
    # a loop that lunges out and back: with the trend window of a stationary loop the
    # lunge stays in the trend, so sway / bounce leave it alone
    frames = 60
    pos = np.zeros((frames, 3))
    pos[:, 0] = 0.5 * np.clip(1.5 * np.sin(np.pi * np.arange(frames) / frames), 0.0, 1.0)
    rot = np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1))
    trend, osc, _, _ = split_root(pos, rot, window=round(TREND_SECONDS * FPS), periodic=True)
    assert np.ptp(trend[:, 0]) > 0.6 * np.ptp(pos[:, 0])
    # the whole loop as the window would average every frame alike
    trend, _, _, _ = split_root(pos, rot, window=frames, periodic=True)
    assert np.ptp(trend[:, 0]) < 1e-9


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


def _decompose(clip: str, is_loop: bool, action_group: str, action_label: str, stretch: float = 0.1,
               key: str = HORSE_KEY):
    from motion_edit.profile.data import load_profiles

    cond = np.load(os.path.join(HORSE_ROOT, "cond.npy"), allow_pickle=True).item()[key]
    features = np.load(os.path.join(HORSE_ROOT, "motions", clip + ".npy"))
    if is_loop:
        from data_loaders.truebones.data.dataset import _drop_loop_closing_frame
        features = _drop_loop_closing_frame(features)
    profile = profile_subset(load_profiles(HORSE_ROOT).get(key), cond, action_label)
    return decompose_motion(features, cond, object_type=key, clip_name=clip, is_loop=is_loop,
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
def packages():
    return {
        "loop": _decompose("Horse_RunLoop", True, "locomotion", "run, forward"),
        "one_shot": _decompose("Horse_Jumping", False, "stationary", "attack, jump, smash"),
        "attack": _decompose("Trex_HumanBite", False, "stationary", "attack, bite", key=TREX_KEY),
        "hurt": _decompose("Trex_HitTorsoLeft", False, "stationary", "hurt", key=TREX_KEY),
        "loop_attack": _decompose("Horse_Attack", True, "stationary", "attack, charge"),
    }


@requires_horse
@pytest.mark.parametrize("kind", ["loop", "one_shot"])
@pytest.mark.parametrize("compose", [False, True], ids=["T1_replay", "T1b_roundtrip"])
def test_default_params_replay_the_decode(packages, kind, compose):
    package = packages[kind]
    ref_rot, ref_pos = _reference(package)
    result = EditRuntime(package).apply(compose=compose)
    assert result.composed == compose
    assert rotation_error(result.animation.rotations.qs, ref_rot) < ROT_TOL
    assert np.abs(result.global_positions - ref_pos).max() < POS_TOL


@requires_horse
@pytest.mark.parametrize("compose", [False, True])
def test_T2_deterministic(packages, compose):
    runtime = EditRuntime(packages["loop"])
    a = runtime.apply(compose=compose)
    b = runtime.apply(compose=compose)
    assert np.array_equal(a.animation.rotations.qs, b.animation.rotations.qs)
    assert np.array_equal(a.animation.positions, b.animation.positions)
    assert np.array_equal(a.global_positions, b.global_positions)


@requires_horse
@pytest.mark.parametrize("compose", [False, True])
def test_T5_bone_lengths_match_package(packages, compose):
    package = packages["one_shot"]
    result = EditRuntime(package).apply(compose=compose)
    parents = package["parents"]
    child = np.flatnonzero(parents >= 0)
    bones = np.linalg.norm(result.global_positions[:, child] - result.global_positions[:, parents[child]], axis=-1)
    expected = np.linalg.norm(package["base_pos"][:, child], axis=-1)
    assert np.abs(bones - expected).max() < BONE_TOL


@requires_horse
def test_package_round_trips_through_disk(packages, tmp_path):
    package = packages["loop"]
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
def test_redecompose_with_zero_stretch_is_rigid(packages):
    package = packages["one_shot"]
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
def test_redecompose_without_fullbody_ik(packages):
    package = packages["loop"]
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
def test_contacts_and_events_are_detected(packages):
    loop = packages["loop"]
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
    assert "force" in packages["one_shot"].manifest["available_params"]


@requires_horse
def test_params_are_refused_not_ignored(packages):
    runtime = EditRuntime(packages["loop"])
    assert not runtime.apply({"tempo": 1.0, "foot_lock": False}).composed
    with pytest.raises(ValueError):
        runtime.apply({"no_such_param": 1.0})
    with pytest.raises(UnsupportedParameterError):
        runtime.apply({"force": 1.5})   # one-shot only
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
@pytest.mark.parametrize("kind", ["loop", "one_shot", "attack", "hurt", "loop_attack"])
def test_T5_T6_edits_keep_bones_and_plants(packages, kind):
    package = packages[kind]
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
            allowed = roll * baseline + locked.get(plant, 0.0) + 0.02 * leg
            assert drift <= allowed, (params, plant, drift, baseline)
        # foot_lock re-pins a heel that pivots again after rolling over its toe: per run
        runs = bool(result.params.get("foot_lock"))
        for plant, (drift, baseline) in _plant_drift(runtime, result, pivots_only=True, pivot_runs=runs).items():
            assert drift <= baseline + 1e-4 * leg, (params, plant, drift, baseline)


@requires_horse
@pytest.mark.parametrize("kind", ["loop", "one_shot"])
def test_T6_foot_lock_pins_the_pivots(packages, kind):
    """The pivot is the foot's deepest planted contact; it holds still while it stays the
    pivot.  A heel that is the pivot again after its toe lifts has rolled about the toe
    in between, so each unbroken run is measured on its own."""
    runtime = EditRuntime(packages[kind])
    for extra in ({}, {"bounce": 1.5}, {"amp.legs": 1.4}):
        result = runtime.apply({"foot_lock": True, **extra})
        drift = _plant_drift(runtime, result, pivots_only=True, pivot_runs=True)
        assert drift and max(d for d, _ in drift.values()) < 1e-4 * runtime.leg, extra


@requires_horse
@pytest.mark.parametrize("kind", ["loop", "one_shot"])
def test_T6_foot_lock_keeps_the_foot_shape(packages, kind):
    """foot_lock removes one slip per foot: every planted contact keeps its original offset
    from the foot's reference contact, so a heel peeling up about a planted toe still peels."""
    runtime = EditRuntime(packages[kind])
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
@pytest.mark.parametrize("kind", ["loop", "loop_attack"])
def test_T4_loop_seam_is_an_ordinary_step(packages, kind):
    package = packages[kind]
    runtime = EditRuntime(package)
    for params in [{}] + _edit_cases(package):
        pos = runtime.apply(params, compose=True).global_positions
        steps = np.linalg.norm(np.diff(pos, axis=0), axis=-1).max(axis=1)
        wrap = np.linalg.norm(pos[0] - pos[-1], axis=-1).max()
        assert wrap <= steps.max() + 1e-9, params


@requires_horse
@pytest.mark.parametrize("kind", ["loop", "one_shot", "attack", "hurt", "loop_attack"])
def test_T7_sliders_are_continuous_at_their_defaults(packages, kind):
    package = packages[kind]
    runtime = EditRuntime(package)
    reference = runtime.apply().global_positions

    def change(name, value):
        pos = runtime.apply({name: value}).global_positions
        n = min(len(pos), len(reference))
        return np.abs(pos[:n] - reference[:n]).max() / runtime.leg

    for name in package.manifest["available_params"]:
        spec = PARAM_SPECS[name]
        if name not in IMPLEMENTED or spec.toggle:
            continue
        eps = 1e-3 * (spec.high - spec.low)
        for sign in (-1.0, 1.0):
            moved = change(name, spec.default + sign * eps)
            if moved <= 20.0 * eps:
                continue
            # a timing slider shifts every later frame, so its constant grows with the clip's
            # length and speed: past the bound it must still be linear (half the step, half
            # the change), which a layer switching on all at once is not
            assert change(name, spec.default + sign * eps / 2) <= 0.55 * moved, (name, sign, moved)


@requires_horse
def test_T2_deterministic_with_edits(packages):
    runtime = EditRuntime(packages["loop"])
    params = {"stride": 1.3, "bounce": 1.4, "amp.legs": 1.2, "tempo": 1.25, "foot_lock": True}
    a, b = runtime.apply(params), runtime.apply(params)
    assert np.array_equal(a.animation.rotations.qs, b.animation.rotations.qs)
    assert np.array_equal(a.global_positions, b.global_positions)


@requires_horse
def test_tempo_keeps_whole_loop_periods(packages):
    package = packages["loop"]
    runtime = EditRuntime(package)
    for tempo in (0.5, 0.8, 1.3, 2.0):
        result = runtime.apply({"tempo": tempo})
        count = len(result.source_time)
        assert count == max(2, round(package.frame_count / tempo))
        assert np.allclose(np.diff(result.source_time), package.frame_count / count)
    one_shot = EditRuntime(packages["one_shot"]).apply({"tempo": 2.0})
    assert one_shot.source_time[-1] <= packages["one_shot"].frame_count - 1
    assert np.allclose(np.diff(one_shot.source_time), 2.0)


@requires_horse
def test_tempo_and_stride_scale_the_ground_speed(packages):
    package = packages["loop"]
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
def test_amplitude_and_root_layers_act_where_named(packages):
    package = packages["loop"]
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
def test_soft_stretch_takes_over_from_lowering(packages):
    """A raised posture is reached by lengthening the legs first; the body is lowered only
    for what the limit cannot cover, and a lowered posture shortens them."""
    runtime = EditRuntime(packages["loop"])
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
def test_contact_mask_edit_rebuilds_plants(packages):
    package = packages["one_shot"]
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
def test_edits_without_contact_joints_skip_the_plants(packages, kind):
    edited = with_contact_joints(packages[kind], [])
    runtime = EditRuntime(edited)
    for params in ({"amp.legs": 0.5}, {"bounce": 1.5}):
        result = runtime.apply(params)
        assert result.composed and result.plant_id.shape[1] == 0
    with pytest.raises(UnsupportedParameterError):
        runtime.apply({"foot_lock": True})


@requires_horse
def test_contact_joint_edit_records_provenance(packages):
    package = packages["one_shot"]
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
def test_input_notes_survive_rebuilds(packages):
    package = packages["one_shot"]
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


# ── one-shot events + force (M4) ─────────────────────────────────────────────

def test_pchip_is_monotone_through_its_nodes():
    x = np.array([0.0, 4.0, 6.0, 15.0, 20.0])
    y = np.array([0.0, 10.0, 12.0, 14.0, 30.0])
    q = np.linspace(0.0, 20.0, 401)
    assert np.allclose(pchip(x, y, x), y)
    assert np.all(np.diff(pchip(x, y, q)) >= -1e-12)
    # collinear nodes give the straight line
    assert np.allclose(pchip(x, 2.0 * x + 1.0, q), 2.0 * q + 1.0)


def test_strike_shapes_hold_and_ease():
    t = np.arange(0.0, 60.0, 0.25)
    hold, push = strike_shapes(t, 20, 30, 36)
    at = lambda c, f: c[np.searchsorted(t, f)]
    assert at(hold, 5) == 0.0 and at(hold, 59) == 0.0 and at(push, 5) == 0.0 and at(push, 59) == 0.0
    assert at(hold, 20) == 1.0 and 0.0 < at(hold, 25) < 1.0 and at(hold, 30) == 0.0
    assert at(push, 25) == 0.0 and at(push, 32) == pytest.approx(1.0) and at(push, 36) == 0.0
    assert at(push, 30) > 0.0                       # the drive starts before contact
    assert np.abs(np.diff(hold)).max() < 0.1 and np.abs(np.diff(push)).max() < 0.15   # no step


@requires_horse
@pytest.mark.parametrize("kind", ["attack", "hurt", "one_shot"])
def test_strike_events_are_ordered_on_the_active_chain(packages, kind):
    package = packages[kind]
    strike = package.manifest["events"]["strike"]
    assert 0 <= strike["windup"] < strike["swing"] <= strike["impact"] < strike["recover"] <= package.frame_count - 1
    chain = next(c for c in strike["candidates"] if c["joint"] == strike["chain"])
    assert strike["effector"] in chain["joints"]
    assert all(0.0 <= v <= 1.0 for v in strike["confidence"].values())
    assert package.manifest["facts"]["strike"]


@requires_horse
def test_strike_bite_is_delivered_by_the_neck(packages):
    package = packages["attack"]
    strike = package.manifest["events"]["strike"]
    groups = [str(g) for g in package["chain_group"]]
    assert strike["label_group"] == "axial" and groups[strike["chain"]] == "axial"


@requires_horse
def test_strike_events_can_be_moved(packages):
    runtime = EditRuntime(packages["attack"])
    strike = runtime.strike()
    moved = runtime.strike({"impact": strike["impact"] + 1})
    assert moved["impact"] == strike["impact"] + 1 and moved["moved"] == ["impact"]
    other = next(c for c in strike["candidates"] if c["joint"] != strike["chain"])
    switched = runtime.strike({"chain": other["joint"]})
    assert switched["joints"] == other["joints"]
    # the effector follows the chain: the lean's lever is the new chain's own leaf
    assert switched["effector"] == other["effector"] and switched["effector"] in other["joints"]
    for bad in ({"impact": strike["windup"]}, {"recover": strike["impact"]},
                {"chain": runtime.package.root}, {"apex": 3}):
        with pytest.raises(ValueError):
            runtime.strike(bad)
    with pytest.raises(ValueError):
        EditRuntime(with_contact_joints(packages["one_shot"], [])).strike({"impact": 3})
    # moving events alone changes nothing: they only key the force parameters
    base = runtime.apply(compose=True).global_positions
    again = runtime.apply(compose=True, events={"impact": strike["impact"] + 1}).global_positions
    assert np.array_equal(again, base)


@requires_horse
def test_force_moves_the_whole_strike_body(packages):
    from motion_edit.rotations import quat_rotate

    runtime = EditRuntime(packages["attack"])
    package = runtime.package
    strike = runtime.strike()
    w, i = strike["windup"], strike["impact"]
    trunk, body = runtime.strike_body(strike)
    assert trunk and set(strike["joints"]) <= set(body) | {package.root}
    base = runtime.apply(compose=True)
    direction = runtime.strike_direction(strike)
    flat = lambda v: v[..., [0, 2]] @ direction

    deep = runtime.apply({"windup_depth": 2.0})
    assert np.array_equal(deep.source_time, base.source_time)
    # the body's weight goes back against the strike in the windup...
    root = package.root
    assert flat(deep.global_positions[w, root] - base.global_positions[w, root]) < -0.5 * STRIKE_SHIFT * runtime.leg
    # ...the trunk leans back with it: the joint the chain hangs on falls behind further than the root
    top = trunk[-1]
    assert flat(deep.global_positions[w, top] - base.global_positions[w, top]) < flat(
        deep.global_positions[w, root] - base.global_positions[w, root])
    # ...and the body is back at its contact pose by contact
    assert np.abs(deep.global_positions[i, body] - base.global_positions[i, body]).max() < 1e-3 * runtime.leg

    far = runtime.apply({"overshoot": 2.0})
    crest = int(round(i + (strike["recover"] - i) / 3.0))
    # past contact the body drives on along the strike, and the active chain's reach with it
    assert flat(far.global_positions[crest, root] - base.global_positions[crest, root]) > 0.5 * STRIKE_SHIFT * runtime.leg
    tip = strike["effector"]
    assert flat(far.global_positions[crest, tip] - base.global_positions[crest, tip]) > 0.5 * STRIKE_SHIFT * runtime.leg
    # joints off the strike body move only rigidly with it or through the plants' IK
    legs = {j for limb in runtime.limbs()[0] for j in limb.chain + [limb.foot]}
    moved = np.abs(far.animation.rotations.qs - base.animation.rotations.qs).max(axis=(0, 2)) > 1e-9
    assert set(np.flatnonzero(moved)) <= set(body) | legs | {root}
    # the lean swings the root -> effector lever toward the strike, never away from it
    axis = runtime.strike_lean_axis(strike, direction)
    lever = runtime.original_positions()[i, tip] - runtime.original_positions()[i, root]
    assert np.dot(np.cross(axis, lever), [direction[0], 0.0, direction[1]]) >= 0.0

    # force is the preset over the rest: depth, strike speed and overshoot up, windup and
    # recover speeds down
    a = runtime.apply({"force": 1.4}).global_positions
    slowed = 1.0 / 1.4 ** FORCE_SLOW_EXPONENT
    b = runtime.apply({"windup_depth": 1.4, "strike_speed": 1.4, "overshoot": 1.4,
                       "windup_speed": slowed, "recover_speed": slowed}).global_positions
    assert np.array_equal(a, b)
    slow = runtime.apply({"force": 2.0, "recover_speed": 0.5})
    assert any(d["kind"] == "clamped" and d["param"] == "recover_speed" for d in slow.diagnostics)
    clamped = runtime.apply({"force": 2.0, "windup_depth": 1.5})
    assert any(d["kind"] == "clamped" and d["param"] == "windup_depth" for d in clamped.diagnostics)
    assert "impact_shift" not in PARAM_SPECS


@requires_horse
def test_segment_speeds_retime_the_events(packages):
    runtime = EditRuntime(packages["attack"])
    strike = runtime.strike()
    w, i, r = strike["windup"], strike["impact"], strike["recover"]
    last = runtime.package.frame_count - 1

    def frame_of(result, source):
        return float(np.interp(source, result.source_time, np.arange(len(result.source_time))))

    for name, (a, b) in (("windup_speed", (0, w)), ("strike_speed", (w, i)), ("recover_speed", (i, r))):
        fast = runtime.apply({name: 2.0})
        assert np.all(np.diff(fast.source_time) > 0), name
        assert frame_of(fast, b) - frame_of(fast, a) == pytest.approx((b - a) / 2.0, abs=0.15), name
        for c, d in ((0, w), (w, i), (i, r)):
            if (c, d) != (a, b):
                assert frame_of(fast, d) - frame_of(fast, c) == pytest.approx(d - c, abs=0.15), (name, c, d)
        assert len(fast.source_time) == int(np.floor(last - (b - a) / 2.0 + 1e-9)) + 1
    # the warped timeline drives the implied ground speed
    fast = runtime.apply({"strike_speed": 2.0})
    expected = (sample_linear(runtime.package["ground_velocity"], fast.source_time, False)
                * np.gradient(fast.source_time)[:, None])
    assert np.allclose(fast.ground_velocity, expected)


@requires_horse
def test_loop_strike_events_wrap_within_one_period(packages):
    package = packages["loop_attack"]
    frames = package.frame_count
    strike = package.manifest["events"]["strike"]
    assert 0 <= strike["impact"] < frames
    assert strike["windup"] < strike["swing"] <= strike["impact"] < strike["recover"] <= strike["windup"] + frames - 1
    runtime = EditRuntime(package)
    # an impact dragged across the seam is the same strike a period on
    moved = runtime.strike({k: strike[k] + frames for k in ("windup", "impact", "recover")})
    assert [moved[k] for k in ("windup", "impact", "recover")] == [strike[k] for k in ("windup", "impact", "recover")]
    with pytest.raises(ValueError):
        runtime.strike({"recover": strike["windup"] + frames})       # longer than a period


@requires_horse
def test_loop_segment_speeds_keep_whole_periods(packages):
    runtime = EditRuntime(packages["loop_attack"])
    frames = runtime.package.frame_count
    strike = runtime.strike()
    w, i, r = strike["windup"], strike["impact"], strike["recover"]
    for params in ({"strike_speed": 2.0}, {"recover_speed": 0.5}, {"windup_speed": 1.5}, {"force": 2.0, "tempo": 1.3}):
        result = runtime.apply(params)
        unwrapped = np.unwrap(result.source_time * 2 * np.pi / frames) * frames / (2 * np.pi)
        assert np.all(np.diff(unwrapped) > 0), params
        assert result.source_time[0] == pytest.approx(0.0, abs=1e-9), params      # frame 0 samples frame 0
        # one whole period: the step across the seam is an ordinary one
        seam = (result.source_time[0] + frames) - result.source_time[-1]
        assert 0.0 < seam < 3.0 * np.diff(unwrapped).max(), params
    fast = runtime.apply({"strike_speed": 2.0})
    length = frames - (i - w) / 2.0
    assert len(fast.source_time) == max(2, int(round(length)))
