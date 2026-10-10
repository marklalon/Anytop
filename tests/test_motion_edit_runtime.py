"""Decomposer + EditRuntime mechanical invariants (section 6.1: T1, T1b, T2, T4-T7).

Usage:
    pytest tests/test_motion_edit_runtime.py

The clip-level tests decompose Truebones Horse and Trex clips and skip when
that dataset is not on disk.
"""

from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.truebones.truebones_utils.param_utils import FPS
from motion_edit.decompose import (
    TREND_SECONDS,
    PartSource,
    airborne_spans,
    chain_layer,
    decompose_motion,
    profile_subset,
    redecompose,
    split_root,
    with_contact_mask,
    with_parts,
)
from motion_edit.ik import Limb, LimbSolver
from motion_edit.package import EditPackage, PackageVersionError, decode_json
from motion_edit.rotations import quat_from_rotvec, quat_inv, quat_mul, rotvec_from_quat
from utils.fullbody_ik import trunk_joint_indices
from motion_edit.runtime import (
    FORCE_SLOW_EXPONENT,
    IMPLEMENTED,
    PARAM_SPECS,
    SPREAD_PLANTED_REACH,
    SPRING_G_GRAV,
    STRIKE_LEAN,
    STRIKE_SHIFT,
    EditRuntime,
    UnsupportedParameterError,
    available_params,
    blend_between_keys,
    forward_kinematics,
    ground_displacement,
    pchip,
    ramped_envelope,
    sample_linear,
    soft_scale,
    spring_response,
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


def test_trunk_is_the_ancestors_of_the_limbs():
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
    # by default every sided limb bounds the trunk: the arms hang off the chest
    assert trunk_joint_indices(parents, sides).tolist() == [0, 1, 2]


def test_part_source_round_trips_and_refuses_unknown_parts():
    source = PartSource(species={"Tail": {"part": "soft"}}, package={"Toe": {"contact": 1}})
    assert PartSource.from_dict(source.as_dict()) == source
    assert PartSource.from_dict(None) == PartSource()
    with pytest.raises(ValueError, match="unknown part"):
        PartSource.from_dict({"package": {"Tail": {"part": "tentacle"}}})


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
        package["source_features"], build_skeleton_only_context(cond),
        restore_space="hml",
        fullbody_ik=package.manifest["fullbody_ik"], stretch_factor=package.manifest["stretch_factor"],
        fps=package.fps)
    return restored.animation.rotations.qs, positions_global(restored.animation)


def _with_passive(package: EditPackage, names: list[str]) -> EditPackage:
    """``package`` with exactly ``names`` and their subtrees soft, so passive (this package's
    own edit); any other soft joint becomes a helper, which no slider or spring reaches."""
    from motion_edit.profile.skeleton import subtree_closure

    index = [str(n) for n in package["names"]]
    soft = subtree_closure(package["parents"], [index.index(n) for n in names])
    parts = [str(p) for p in package["joint_part"]]
    joints = {index[j]: {"part": "soft" if j in soft else "helper"}
              for j in range(len(index)) if j in soft or parts[j] == "soft"}
    return with_parts(package, joints)


def _with_contacts(package: EditPackage, joints) -> EditPackage:
    """``package`` with exactly ``joints`` as its contacts (this package's own edit)."""
    names = [str(n) for n in package["names"]]
    return with_parts(package, {name: {"contact": int(j in set(joints))} for j, name in enumerate(names)})


def test_an_edit_cannot_make_a_helper_a_contact(packages):
    package = packages["loop"]
    names = [str(n) for n in package["names"]]
    contacts = set(int(j) for j in package["contact_joints"])
    contact_name = names[min(contacts)]
    free_name = names[next(j for j in range(len(names)) if j not in contacts and j != package.root)]
    helper = with_parts(package, {free_name: {"part": "helper"}})
    with pytest.raises(ValueError, match="helper joint cannot be a contact"):
        with_parts(helper, {free_name: {"contact": 1}})
    with pytest.raises(ValueError, match="helper joint cannot be a contact"):
        with_parts(package, {free_name: {"part": "helper", "contact": 1}})
    # making a contact joint a helper drops its contact
    dropped = with_parts(package, {contact_name: {"part": "helper"}})
    assert names.index(contact_name) not in set(int(j) for j in dropped["contact_joints"])


HORSE_TAIL = ["BN_Tail_01"]       # with its subtree
TREX_TAIL = ["jt_Tail1_C"]


@pytest.fixture(scope="module")
def packages():
    out = {
        "loop": _decompose("Horse_RunLoop", True, "locomotion", "run, forward"),
        "one_shot": _decompose("Horse_Jumping", False, "stationary", "attack, jump, smash"),
        "attack": _decompose("Trex_HumanBite", False, "stationary", "attack, bite", key=TREX_KEY),
        "hurt": _decompose("Trex_HitTorsoLeft", False, "stationary", "hurt", key=TREX_KEY),
        "loop_attack": _decompose("Horse_Attack", True, "stationary", "attack, charge"),
    }
    out["loop_tail"] = _with_passive(out["loop"], HORSE_TAIL)
    out["loop_bare"] = _with_passive(out["loop"], [])
    out["attack_tail"] = _with_passive(out["attack"], TREX_TAIL)
    return out


@requires_horse
@pytest.mark.parametrize("kind", ["loop", "one_shot", "attack_tail"])
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
    kept = {rigid.root, *trunk_joint_indices(parents, cond["joint_side_labels"]).tolist()}
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
        runtime.apply({"jump_height": 1.5})   # the label names no jump


def test_airborne_spans_wrap_the_loop_seam():
    mask = np.ones((10, 2), dtype=bool)
    mask[[0, 1, 8, 9]] = False
    assert airborne_spans(mask) == []                      # touches both ends: not between plants
    assert airborne_spans(mask, periodic=True) == [(8, 12)]
    assert airborne_spans(np.zeros((10, 2), dtype=bool), periodic=True) == []


def test_jump_height_follows_the_jump_label():
    facts = {"is_loop": True, "locomotion": True, "has_plants": True, "turning": False,
             "airborne": True, "jump": True, "has_passive": False, "chain_groups": []}
    assert "jump_height" in available_params(facts)
    assert "jump_height" not in available_params({**facts, "jump": False, "is_loop": False})
    assert "jump_height" not in available_params({**facts, "airborne": False})


@requires_horse
def test_jump_height_scales_the_lift_without_sinking_feet():
    package = _decompose("Horse_RunJump", False, "stationary", "jump, forward")
    runtime = EditRuntime(package)
    assert "jump_height" in runtime.available
    contacts = np.asarray(package["contact_joints"])
    mask = np.asarray(package["contact_mask"], dtype=bool)
    heights = runtime.original_positions()[:, contacts, 1]
    floor = np.array([heights[mask[:, c], c].min() if mask[:, c].any() else package.manifest["ground_height"]
                      for c in range(len(contacts))])
    spans = package.manifest["events"]["airborne"]
    jump = max(spans, key=lambda span: span[1] - span[0])

    def clearance(k):
        positions = runtime.apply({"jump_height": k}, compose=True).global_positions
        return (positions[:, contacts, 1] - floor[None]).min(axis=1)

    source, low, high = clearance(1.0), clearance(0.5), clearance(1.8)
    frames = np.arange(*jump)
    assert low[frames].max() < 0.6 * source[frames].max()
    assert high[frames].max() > 1.6 * source[frames].max()
    for start, end in spans:
        frames = np.arange(start, end)
        assert (low[frames] >= np.minimum(source[frames], 0.0) - POS_TOL).all()

# ── edits (M3): T4-T7 ────────────────────────────────────────────────────────

def _edit_cases(package) -> list[dict]:
    """Every implemented slider of the clip at both ends of its range, and every toggle flipped."""
    cases = []
    available = EditRuntime(package).available
    for name in (n for n in PARAM_SPECS if n in available):
        if name not in IMPLEMENTED:
            continue
        spec = PARAM_SPECS[name]
        cases += [{name: not spec.default}] if spec.toggle else [{name: spec.low}, {name: spec.high}]
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
@pytest.mark.parametrize("kind", ["loop", "one_shot", "attack", "hurt", "loop_attack", "loop_tail"])
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
@pytest.mark.parametrize("kind", ["loop", "loop_attack", "loop_tail"])
def test_T4_loop_seam_is_an_ordinary_step(packages, kind):
    package = packages[kind]
    runtime = EditRuntime(package)
    for params in [{}] + _edit_cases(package):
        pos = runtime.apply(params, compose=True).global_positions
        steps = np.linalg.norm(np.diff(pos, axis=0), axis=-1).max(axis=1)
        wrap = np.linalg.norm(pos[0] - pos[-1], axis=-1).max()
        assert wrap <= steps.max() + 1e-9, params


@requires_horse
@pytest.mark.parametrize("kind", ["loop", "one_shot", "attack", "hurt", "loop_attack", "loop_tail",
                                  "attack_tail"])
def test_T7_sliders_are_continuous_at_their_defaults(packages, kind):
    package = packages[kind]
    runtime = EditRuntime(package)
    reference = runtime.apply().global_positions

    def change(name, value):
        pos = runtime.apply({name: value}).global_positions
        n = min(len(pos), len(reference))
        return np.abs(pos[:n] - reference[:n]).max() / runtime.leg

    for name in (n for n in PARAM_SPECS if n in runtime.available):
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
    moved = np.abs(runtime.apply({"tail_weight": 1.8}).global_positions - base.global_positions).max(axis=(0, 2))
    assert moved[tail].max() > 1e-3
    assert moved[~tail].max() < 1e-9          # the tail carries no foot: nothing else moves
    # bounce 0 leaves the root on its trend, lowered only where a foot could not reach
    result = runtime.apply({"bounce": 0.0})
    lowered = package["root_trend"][:, 1] - result.animation.positions[:, package.root, 1]
    assert lowered.min() > -1e-9
    if not any(d["kind"] == "ground" for d in result.diagnostics):
        assert lowered.max() < 1e-9


@requires_horse
@pytest.mark.parametrize("kind, group", [("attack", "arms"), ("attack", "legs"), ("loop", "legs")])
def test_spread_moves_the_limb_tips_outward(packages, kind, group):
    """A spread turns paired limbs only, never a passive part (one may hang off a limb and
    ride along, like a T-rex's thigh muscle)."""
    runtime = EditRuntime(packages[kind])
    base = runtime.apply(compose=True).global_positions
    value = 0.3
    result = runtime.apply({f"spread.{group}": value})
    lateral = runtime._spread_lateral(group, result.source_time)
    limbs = [l for l in runtime.spread_limbs() if l.group == group]
    assert {l.sign for l in limbs} == {1.0, -1.0}
    passive = [bool(c) for c in runtime.channels()]
    parents = packages[kind]["parents"]

    def chain(limb):
        j, out = limb.tip, []
        while j != limb.pivot:
            out.append(j)
            j = int(parents[j])
        return out + [limb.pivot]

    assert not any(passive[j] for l in limbs for j in chain(l))
    for limb in limbs:
        outward = limb.sign * np.sum((result.global_positions[:, limb.tip] - base[:, limb.tip]) * lateral, axis=-1)
        assert np.median(outward) == pytest.approx(value * limb.reach * limb.length, rel=0.05)
    if group == "arms":
        # nothing but the arms moves: the T-rex's arms carry no foot
        arms = np.zeros(packages[kind].joint_count, dtype=bool)
        for limb in limbs:
            arms[_subtree(packages[kind], limb.pivot)] = True
        moved = np.abs(result.global_positions - base).max(axis=(0, 2))
        assert moved[~arms].max() < 1e-9


def test_groups_follow_the_parts():
    # a skirt hanging off the pelvis beside the legs is soft: it groups with the small
    # attachments; the same chain labelled as arms is an arm
    from motion_edit.decompose import chain_groups
    from motion_edit.profile.skeleton import SkeletonStructure

    names = ["Pelvis", "L Thigh", "L Foot", "R Thigh", "R Foot", "Skirt_L1", "Skirt_L2", "Skirt_R1", "Skirt_R2"]
    cond = {"joints_names": names, "parents": [-1, 0, 1, 0, 3, 0, 5, 0, 7],
            "offsets": [[0, 1, 0], [0.1, 0, 0], [0, -0.9, 0], [-0.1, 0, 0], [0, -0.9, 0],
                        [0.15, 0, 0], [0, -0.4, 0], [-0.15, 0, 0], [0, -0.4, 0]],
            "joint_side_labels": ["center", "left", "left", "right", "right", "left", "left", "right", "right"]}
    structure = SkeletonStructure(cond, [2, 4])
    legs = ["trunk", "leg", "foot", "leg", "foot"]
    assert list(chain_groups(structure, legs + ["soft"] * 4)) == ["root"] + ["legs"] * 4 + ["other"] * 4
    assert list(chain_groups(structure, legs + ["arm", "hand"] * 2)[5:]) == ["arms"] * 4
    assert list(chain_groups(structure, legs + ["fin", "helper", "wing", "tail"])[5:]) == ["fins", "other", "wings", "tail"]


@requires_horse
def test_spread_is_offered_for_paired_limbs_only(packages):
    # the horse's forelegs are arms and its hind legs legs; all four stand on the ground, so
    # each turns from its IK limb's root by the planted reach and moves its contacts
    runtime = EditRuntime(packages["loop"])
    limbs = runtime.spread_limbs()
    assert sorted(l.group for l in limbs) == ["arms", "arms", "legs", "legs"]
    assert all(l.columns and l.reach == SPREAD_PLANTED_REACH for l in limbs)
    for group in ("arms", "legs"):
        assert f"spread.{group}" in runtime.available
        assert packages["loop"].manifest["params"][f"spread.{group}"]["available"]
    # a pair broken by a part edit is no longer offered
    package = packages["loop"]
    names = [str(n) for n in package["names"]]
    right_arm = [names[j] for j in range(len(names))
                 if str(package["chain_group"][j]) == "arms" and str(package["sides"][j]) == "right"]
    lopsided = EditRuntime(with_parts(package, {n: {"part": "helper"} for n in right_arm}))
    assert "spread.arms" not in lopsided.available
    with pytest.raises(UnsupportedParameterError):
        lopsided.apply({"spread.arms": 0.5})


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
    edited = _with_contacts(packages[kind], [])
    runtime = EditRuntime(edited)
    for params in ({"amp.legs": 0.5}, {"bounce": 1.5}):
        result = runtime.apply(params)
        assert result.composed and result.plant_id.shape[1] == 0
    with pytest.raises(UnsupportedParameterError):
        runtime.apply({"foot_lock": True})


@requires_horse
def test_part_edit_records_its_layer(packages):
    package = packages["one_shot"]
    joints = list(package["contact_joints"])
    removed = joints[0]
    names = [str(n) for n in package["names"]]
    added = names.index("Bip01_Head") if "Bip01_Head" in names else int(package["parents"][joints[0]])
    edited = _with_contacts(package, joints[1:] + [added])
    # the package layer holds exactly what differs from the prefill
    assert edited.manifest["parts"]["package"] == {names[added]: {"contact": 1}, names[removed]: {"contact": 0}}
    assert added in list(edited["contact_joints"]) and removed not in list(edited["contact_joints"])
    rows = edited.manifest["parts"]["joints"]
    assert rows[added]["src"] == rows[removed]["src"] == "package"
    # written into the species row, the change leaves the package's own edits
    species = with_parts(edited, {}, species=True)
    assert species.manifest["parts"]["package"] == {}
    assert species.manifest["parts"]["species"] == edited.manifest["parts"]["package"]
    assert list(species["contact_joints"]) == list(edited["contact_joints"])
    # a part edit that leaves the contact set alone keeps hand-edited intervals
    mask = np.array(package["contact_mask"], copy=True)
    mask[:2] = True
    kept = with_parts(with_contact_mask(package, mask), {names[added]: {"part": "soft"}})
    assert np.array_equal(kept["contact_mask"], mask) and kept.manifest["contacts"]["intervals_edited"]


@requires_horse
def test_input_notes_survive_rebuilds(packages):
    package = packages["one_shot"]
    note = {"kind": "parts", "message": "override row is stale"}
    package = EditPackage(dict(package.manifest, input_notes=[note]), package.arrays)
    for rebuilt in (redecompose(package), with_contact_mask(package, package["contact_mask"]),
                    with_parts(package, {})):
        assert note in rebuilt.manifest["diagnostics"]["items"]
        assert rebuilt.manifest["input_notes"] == [note]
    # a rewritten species row voids the notes about the old one
    rewritten = with_parts(package, {}, species=True)
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
    parts = [str(p) for p in package["joint_part"]]
    assert strike["label_parts"] == ["head", "neck"] and parts[strike["chain"]] in ("head", "neck")


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
        EditRuntime(_with_contacts(packages["one_shot"], [])).strike({"impact": 3})
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
    # the passive tail swings with the body it hangs on
    passive = set(np.flatnonzero(package["profile_passive"]).tolist())
    assert set(np.flatnonzero(moved)) <= set(body) | legs | passive | {root}
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


# ── secondary motion (M5) ────────────────────────────────────────────────────

def test_spring_response_matches_the_steady_state_and_closes_a_loop():
    fps, k, c, frames = 30.0, 150.0, 4.0, 60
    omega = 2.0 * np.pi * fps / frames * 3                     # three cycles per loop
    t = np.arange(frames) / fps
    drive = np.stack([np.sin(omega * t), np.cos(omega * t), np.zeros(frames)], axis=1)
    theta = spring_response(drive, k, c, fps, periodic=True)
    gain = 1.0 / abs(k - omega ** 2 + 1j * omega * c)
    assert np.linalg.norm(theta[:, :2], axis=1) == pytest.approx(np.full(frames, gain), rel=0.02)
    assert np.abs(theta[:, 2]).max() == 0.0
    # periodic: one more step past the last frame lands on the first
    again = spring_response(np.concatenate([drive, drive]), k, c, fps, periodic=True)
    assert np.abs(again[frames:] - theta).max() < 1e-9
    # a one-shot starts at rest, and a stiff spring stays stable
    assert np.abs(spring_response(np.zeros((10, 3)), k, c, fps, periodic=False)).max() == 0.0
    stiff = spring_response(drive, 4.0 * (2.0 * np.pi * 8.0) ** 2, 1.0, fps, periodic=False)
    assert np.isfinite(stiff).all() and np.abs(stiff).max() < 1.0


def _subtree(package: EditPackage, j: int) -> list[int]:
    from motion_edit.profile.skeleton import subtree_closure

    return sorted(subtree_closure(package["parents"], [j]))


def _swing(result, reference) -> np.ndarray:
    """Per frame and joint, the parent-frame rotation vector taking ``reference`` to ``result``."""
    return rotvec_from_quat(quat_mul(result.animation.rotations.qs, quat_inv(reference.animation.rotations.qs)))


@requires_horse
@pytest.mark.parametrize("kind", ["loop_tail", "attack_tail"])
def test_secondary_scales_and_swings_the_passive_joints(packages, kind):
    package = packages[kind]
    runtime = EditRuntime(package)
    passive = np.flatnonzero(package["profile_passive"])
    passive = passive[~np.asarray(package["chain_gain_locked"], dtype=bool)[passive]]
    others = np.setdiff1d(np.arange(package.joint_count), passive)
    reference = np.asarray(package["chain_reference"])
    offsets = np.asarray(package["chain_offsets"])
    one = runtime.apply(compose=True)
    for value in (0.0, 0.5):
        rot = runtime.apply({"passive_weight": value}).animation.rotations.qs
        # below 1 the passive joints' own curves scale like an amplitude: rigid at 0
        expected = quat_mul(quat_from_rotvec(value * offsets[:, passive]), reference[None, passive])
        assert rotation_error(rot[:, passive], expected) < ROT_TOL, value
        assert rotation_error(rot[:, others], one.animation.rotations.qs[:, others]) < ROT_TOL, value
    # past 1 they swing with the body, each bone by an angle in proportion to the slider
    full, half = runtime.apply({"passive_weight": 2.0}), runtime.apply({"passive_weight": 1.5})
    swing = _swing(full, one)
    assert np.abs(swing[:, others]).max() < ROT_TOL
    assert np.abs(full.global_positions[:, others] - one.global_positions[:, others]).max() < POS_TOL
    assert np.linalg.norm(swing[:, passive], axis=-1).max() > np.radians(1.0)

    def largest(result):
        item = next(d for d in result.diagnostics if d["kind"] == "secondary" and "joint" in d)
        return float(re.search(r"up to ([0-9.]+) deg", item["message"]).group(1))

    assert largest(half) == pytest.approx(0.5 * largest(full), abs=0.1)


@requires_horse
def test_secondary_swings_with_the_edited_body(packages):
    """The swing answers to the body as edited; a constant posture offset carries no acceleration."""
    runtime = EditRuntime(packages["loop_tail"])
    passive = np.flatnonzero(runtime.package["profile_passive"])
    alone = _swing(runtime.apply({"passive_weight": 2.0}), runtime.apply(compose=True))[:, passive]
    for params, moves in (({"amp.axial": 1.5}, True), ({"posture": -0.2}, False)):
        swing = _swing(runtime.apply({**params, "passive_weight": 2.0}), runtime.apply(params))[:, passive]
        assert (np.abs(swing - alone).max() > np.radians(1.0)) == moves, params


@requires_horse
def test_stiffness_tunes_only_its_own_swing(packages):
    """A channel's stiffness acts only on its swing (weight past 1): a stiffer spring swings
    less; the other channel's stiffness leaves it alone."""
    runtime = EditRuntime(packages["loop_tail"])
    passive = np.flatnonzero(runtime.package["profile_passive"])
    one = runtime.apply(compose=True)
    still = runtime.apply({"passive_stiffness": 2.0})
    assert np.array_equal(still.animation.rotations.qs, one.animation.rotations.qs)
    swing = {hard: np.linalg.norm(_swing(runtime.apply({"passive_weight": 2.0, "passive_stiffness": hard}),
                                         one)[:, passive], axis=-1).max()
             for hard in (0.5, 1.0, 2.0)}
    assert swing[0.5] > swing[1.0] > swing[2.0] > 0.0
    other = runtime.apply({"passive_weight": 2.0, "tail_stiffness": 2.0}
                          if "tail_stiffness" in runtime.available else {"passive_weight": 2.0})
    assert np.linalg.norm(_swing(other, one)[:, passive], axis=-1).max() == pytest.approx(swing[1.0])


@requires_horse
def test_gravity_toggles_the_springs_gravity_term(packages):
    """gravity keeps the springs' gravity term; it acts only on a swing (weight past 1)."""
    runtime = EditRuntime(packages["loop_tail"])
    passive = np.flatnonzero(runtime.package["profile_passive"])
    one = runtime.apply(compose=True)
    assert "gravity" in runtime.available
    assert np.array_equal(runtime.apply({"gravity": False}).animation.rotations.qs, one.animation.rotations.qs)
    for spring in runtime.springs()[0]:
        spring["spring"] = np.array(spring["spring"], copy=True)
        spring["spring"][SPRING_G_GRAV] = 20.0
    on, off = runtime.apply({"passive_weight": 2.0}), runtime.apply({"passive_weight": 2.0, "gravity": False})
    assert np.linalg.norm(_swing(on, off)[:, passive], axis=-1).max() > np.radians(1.0)
    for spring in runtime.springs()[0]:
        spring["spring"][SPRING_G_GRAV] = 0.0
    zero = runtime.apply({"passive_weight": 2.0})
    assert rotation_error(zero.animation.rotations.qs, off.animation.rotations.qs) < ROT_TOL


@requires_horse
def test_T2_deterministic_with_secondary(packages):
    runtime = EditRuntime(packages["attack_tail"])
    params = {"force": 1.6, "amp.axial": 1.3, "passive_weight": 1.7}
    a, b = runtime.apply(params), runtime.apply(params)
    assert np.array_equal(a.animation.rotations.qs, b.animation.rotations.qs)
    assert np.array_equal(a.global_positions, b.global_positions)


# ── passive joint set ────────────────────────────────────────────────────────

@requires_horse
def test_passive_defaults_are_the_soft_parts(packages):
    package = packages["loop"]
    m = package.manifest["passive"]
    names = set(m["names"])
    parts = [str(p) for p in package["joint_part"]]
    # the mane is soft with its whole subtree, leaves included; the tail stays with tail_weight,
    # the ears and the jaw with the head
    assert {"BN_hair04_01", "BN_hair04_03"} <= names
    assert not names & {"BN_Tail_01", "BN_Tail_05", "Bip01_Jaw", "Bip01_R_Ear_01", "Handle"}
    passive = set(m["joints"])
    assert all(j in passive for j in passive for j in _subtree(package, j))
    assert passive == {j for j, p in enumerate(parts) if p == "soft"} & set(m["candidates"])
    # hair answers to passive_weight, the tail to tail_weight: tuned apart
    facts = package.manifest["facts"]
    assert facts["has_passive"] and "passive_weight" in package.manifest["available_params"]
    assert "tail_weight" in package.manifest["available_params"]
    # a tail made passive leaves tail_weight with nothing to scale
    assert "tail_weight" not in packages["loop_tail"].manifest["available_params"]
    bare = packages["loop_bare"]
    assert bare.manifest["passive"]["joints"] == [] and not bare.manifest["facts"]["has_passive"]
    assert "passive_weight" not in bare.manifest["available_params"]


@requires_horse
def test_soft_edits_make_parts_passive_and_survive_rebuilds(packages):
    package = packages["one_shot"]
    names = [str(n) for n in package["names"]]
    jaw = _subtree(package, names.index("Bip01_Jaw"))
    # a part may stop partway down: the jaw's tip left as it is follows the jaw rigidly
    edited = with_parts(package, {names[j]: {"part": "soft"} for j in jaw[:-1]})
    passive = edited.manifest["passive"]["joints"]
    assert set(jaw[:-1]) <= set(passive) and jaw[-1] not in passive
    # a soft joint with a support joint below it is reported, not made passive
    spine = names.index("Bip01_Spine")
    blocked = with_parts(package, {names[spine]: {"part": "soft"}})
    assert spine not in blocked.manifest["passive"]["joints"]
    assert any(d["kind"] == "passive" and d.get("joint") == names[spine]
               for d in blocked.manifest["diagnostics"]["items"])
    assert redecompose(edited).manifest["passive"]["joints"] == passive


@requires_horse
def test_serve_writes_a_species_parts_override(packages, tmp_path):
    from motion_edit.profile.parts import JOINT_PARTS_OVERRIDES_FILE
    from motion_edit.ui.serve import Handler, PackageStore

    package = packages["one_shot"]
    root = tmp_path / "dataset"
    root.mkdir()
    store_root = tmp_path / "packages"
    package = EditPackage(dict(package.manifest, dataset_root=str(root)), package.arrays)
    package.save(str(store_root / "clip.edit"))
    handler = Handler.__new__(Handler)
    handler.server = type("Server", (), {"store": PackageStore(str(store_root))})()
    names = [str(n) for n in package["names"]]
    jaw = _subtree(package, names.index("Bip01_Jaw"))
    # a package edit first, then the same joints written for the species
    handler._parts("clip.edit", {"joints": {names[j]: {"part": "soft"} for j in jaw}})
    payload = handler._parts("clip.edit", {"joints": {}, "species": True})
    rows = json.loads((root / JOINT_PARTS_OVERRIDES_FILE).read_text(encoding="utf-8"))
    row = rows[package.manifest["object_type"]]
    assert row["joints"] == {names[j]: {"part": "soft"} for j in jaw}
    assert row["skeleton_hash"] == package.manifest["skeleton_hash"]
    assert set(jaw) <= set(payload["passive"]["joints"])
    assert payload["parts"]["package"] == {} and payload["parts"]["species"] == row["joints"]
    assert {payload["parts"]["joints"][j]["src"] for j in jaw} == {"species"}
    with pytest.raises(ValueError, match="unknown part"):
        handler._parts("clip.edit", {"joints": {names[jaw[0]]: {"part": "antenna"}}})


@requires_horse
def test_passive_part_hinges_at_its_parent(packages):
    """A passive part swings from its non-passive parent, so even a lone leaf moves; bones keep
    their lengths and nothing outside the part moves."""
    runtime = EditRuntime(_with_passive(packages["loop"], ["Bip01_R_Ear_Nub"]))
    package = runtime.package
    leaf = list(package["names"]).index("Bip01_R_Ear_Nub")
    assert [(s["joint"], s["virtual"]) for s in runtime.springs()[0] if s["channel"] == "passive"] == [(leaf, True)]
    base, swung = runtime.apply(compose=True), runtime.apply({"passive_weight": 2.0})
    moved = np.linalg.norm(swung.global_positions - base.global_positions, axis=-1).max(axis=0)
    assert moved[leaf] > 1e-3 * runtime.leg
    assert np.delete(moved, leaf).max() < POS_TOL
    assert np.abs(np.linalg.norm(swung.animation.positions, axis=-1)
                  - np.linalg.norm(base.animation.positions, axis=-1)).max() < BONE_TOL


@requires_horse
def test_tail_and_passive_parts_share_one_treatment(packages):
    """tail_weight on the tail is what passive_weight is on the same joints made passive: two channels,
    one treatment (scaled curves up to 1, a hinged spring swing past it)."""
    tail_runtime = EditRuntime(packages["loop_bare"])
    passive_runtime = EditRuntime(packages["loop_tail"])
    tail = [j for j, c in enumerate(tail_runtime.channels()) if c == "tail"]
    assert tail and tail == list(np.flatnonzero(packages["loop_tail"]["profile_passive"]))
    for value in (0.0, 0.5, 1.5, 2.0):
        a = tail_runtime.apply({"tail_weight": value})
        b = passive_runtime.apply({"passive_weight": value})
        assert np.abs(a.global_positions - b.global_positions).max() < POS_TOL, value
        assert rotation_error(a.animation.rotations.qs, b.animation.rotations.qs) < ROT_TOL, value
    # the first tail joint moves: its bone hinges at the pelvis
    base, swung = tail_runtime.apply(compose=True), tail_runtime.apply({"tail_weight": 2.0})
    assert np.linalg.norm(swung.global_positions[:, tail[0]] - base.global_positions[:, tail[0]], axis=-1).max()         > 1e-3 * tail_runtime.leg


# ── export (M6) ────────────────────────────────────────────────────────────

def _serve(root: str):
    import threading
    from http.server import ThreadingHTTPServer

    from motion_edit.ui.serve import Handler, PackageStore

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.store = PackageStore(root)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _call(server, path: str, body=None):
    import urllib.request

    request = urllib.request.Request(
        f"http://127.0.0.1:{server.server_port}{path}", method="GET" if body is None else "POST",
        data=None if body is None else json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        raw = response.read()
        return json.loads(raw) if response.headers.get_content_type() == "application/json" else raw


@requires_horse
def test_ui_export_matches_the_command_line(packages, tmp_path):
    """The page's export and ``apply_edit --sidecar <sidecar>`` write the same bytes, with
    moved strike events and root motion; all defaults export the package's own decode."""
    from motion_edit.apply_edit import export_result
    from motion_edit.apply_edit import main as apply_main

    package = packages["attack"]
    package.save(str(tmp_path / "attack.edit"))
    strike = EditRuntime(package).strike()
    server = _serve(str(tmp_path))
    try:
        cases = {"edit": ({"force": 1.5, "tempo": 1.2}, {"impact": strike["impact"] + 1}, True),
                 "default": ({}, None, False)}
        for name, (params, events, root_motion) in cases.items():
            out = _call(server, "/api/export", {"id": "attack.edit", "params": params, "events": events,
                                                "root_motion": root_motion, "name": name})
            with open(out["sidecar"], "r", encoding="utf-8") as handle:
                assert json.load(handle)["events"] == events
            downloaded = _call(server, out["url"])
            with open(out["path"], "rb") as handle:
                assert handle.read() == downloaded
            cli = str(tmp_path / f"cli_{name}.glb")
            argv = [str(tmp_path / "attack.edit"), "--sidecar", out["sidecar"], "--glb", cli]
            assert apply_main(argv + (["--root_motion"] if root_motion else [])) == 0
            with open(cli, "rb") as handle:
                assert handle.read() == downloaded, name
        # all defaults: the runtime's replay of the package, not the composed round trip
        replay = str(tmp_path / "replay.glb")
        export_result(package, EditRuntime(package).apply(), replay, fmt="glb")
        with open(replay, "rb") as a, open(str(tmp_path / "attack.edit" / "exports" / "default.glb"), "rb") as b:
            assert a.read() == b.read()
    finally:
        server.shutdown()
        server.server_close()


@requires_horse
def test_root_motion_travels_along_the_ground_velocity(packages):
    from motion_edit.apply_edit import root_motion_animation

    package = packages["loop"]
    result = EditRuntime(package).apply({"tempo": 1.3})
    moved = root_motion_animation(package, result)
    disp = ground_displacement(result.ground_velocity, result.fps, False)
    shift = moved.positions - result.animation.positions
    roots = np.flatnonzero(package["parents"] < 0)
    assert np.allclose(shift[:, roots][..., [0, 2]], disp[:, None], atol=1e-12)
    assert np.abs(np.delete(shift, roots, axis=1)).max() == 0 and np.abs(shift[..., 1]).max() == 0
    # a run covers ground: the export is not the in-place clip
    assert np.linalg.norm(disp[-1]) > 0.5 * EditRuntime(package).leg


HORSE_TPOSE = os.path.join(ANYTOP, "dataset", "truebones", "zoo", "Truebone_Z-OO", "Horse", "HorseALL-TPOSE.glb")
requires_horse_mesh = pytest.mark.skipif(not os.path.isfile(HORSE_TPOSE), reason="Horse T-pose mesh not on disk")


@requires_horse
@requires_horse_mesh
def test_mesh_preview_drives_bones_like_the_skinned_export(packages, tmp_path):
    """A bone at ``W_joint(t) · C`` (the page's skinning) is where the skinned export puts it,
    up to the export's fixed similarity; the UI's skinned export is the command line's."""
    from motion_edit.apply_edit import main as apply_main
    from motion_edit.mesh import _glb_skin_worlds, _quat_matrix, attach_mesh, mesh_source

    package = packages["loop"]
    directory = str(tmp_path / "run.edit")
    package.save(directory)
    calibration = attach_mesh(directory, package, HORSE_TPOSE)
    runtime = EditRuntime(package)
    assert calibration["rest_residual"] < 1e-4 * runtime.leg
    assert not calibration["unskinned_joints"] and mesh_source(directory) == os.path.abspath(HORSE_TPOSE)
    # re-decomposing rewrites the package, not its mesh
    redecompose(package, 0.0).save(directory)
    assert mesh_source(directory) is not None

    server = _serve(str(tmp_path))
    try:
        payload = _call(server, "/api/package/run.edit")["mesh"]
        assert payload["exportable"] and set(payload["bones"]) == set(calibration["bones"])
        params = {"tempo": 1.2, "amp.legs": 1.5}
        out = _call(server, "/api/export", {"id": "run.edit", "params": params,
                                            "mesh": True, "name": "skinned"})
        assert "--mesh" in out["command"]
    finally:
        server.shutdown()
        server.server_close()
    cli = str(tmp_path / "cli.glb")
    assert apply_main([directory, "--sidecar", out["sidecar"], "--mesh", "--glb", cli]) == 0
    with open(cli, "rb") as a, open(out["path"], "rb") as b:
        assert a.read() == b.read()

    # bone worlds of the export's first key against the page's W · C (export space -> package space)
    exported = _glb_skin_worlds(out["path"])
    result = EditRuntime(EditPackage.load(directory)).apply(params)
    rotations, positions = forward_kinematics(package["parents"], np.asarray(result.animation.rotations.qs, float),
                                              np.asarray(result.animation.positions, float))
    driven = {}
    for name, bone in calibration["bones"].items():
        j = bone["joint"]
        if j < 0:
            continue
        world = np.eye(4)
        q = rotations[0, j]
        world[:3, :3] = _quat_matrix([q[1], q[2], q[3], q[0]])
        world[:3, 3] = positions[0, j]
        driven[name] = world @ np.array(bone["matrix"]).reshape(4, 4).T
    names = sorted(driven)
    a = np.array([exported[n][:3, 3] for n in names])
    b = np.array([driven[n][:3, 3] for n in names])
    from motion_edit.mesh import _similarity
    similarity = _similarity(a, b)
    worst = max(np.abs(similarity @ exported[n] - driven[n]).max() for n in names)
    assert worst < 1e-4 * runtime.leg


@requires_horse
@requires_horse_mesh
def test_skinned_export_matches_restore_glb(packages, tmp_path):
    """The skinned export at defaults is ``tools/restore_glb_from_npy.py``'s skinned GLB of the
    same features: every bone's world rotation agrees (IK runs on the cond skeleton here and on
    the mesh rig there, so not to the bit)."""
    from data_loaders.truebones.truebones_utils.cond_schema import load_cond
    from motion_edit.mesh import _glb_skin_worlds, export_skinned_glb, mesh_context
    from utils.exporter import AnimationExporter, animation_to_exporter_inputs
    from utils.npy_restore import build_mesh_restore_context, restore_animation_from_features

    package = packages["loop"]
    mine = export_skinned_glb(package, EditRuntime(package).apply().animation, package.fps,
                              str(tmp_path / "mine.glb"), HORSE_TPOSE)
    cond = load_cond(os.path.join(HORSE_ROOT, "cond.npy"))[HORSE_KEY]
    restored = restore_animation_from_features(
        package["source_features"],
        build_mesh_restore_context(cond, HORSE_TPOSE, HORSE_KEY),
        restore_space="native", fullbody_ik=True, stretch_factor=package.manifest["stretch_factor"],
        fps=package.fps)
    inputs = animation_to_exporter_inputs(restored.animation, restored.skeleton)
    reference = str(tmp_path / "reference.glb")
    AnimationExporter(restored.skeleton, fps=package.fps).export_glb(
        *inputs[:3], reference, mesh_path=HORSE_TPOSE, bone_translations=inputs[3], export_mesh=True)
    a, b = _glb_skin_worlds(mine), _glb_skin_worlds(reference)
    assert set(a) == set(b)
    for name in a:
        ra = a[name][:3, :3] / np.linalg.norm(a[name][:3, 0])
        rb = b[name][:3, :3] / np.linalg.norm(b[name][:3, 0])
        angle = np.degrees(np.arccos(np.clip((np.trace(ra.T @ rb) - 1) / 2, -1, 1)))
        assert angle < 0.5, (name, angle)

    # a package from before its cond subset carried root_promote_depth cannot be skinned
    old = EditPackage(package.manifest, dict(package.arrays))
    cond_subset = decode_json(package["source_cond"])
    del cond_subset["root_promote_depth"]
    from motion_edit.package import encode_json
    old.arrays["source_cond"] = encode_json(cond_subset)
    with pytest.raises(ValueError, match="root_promote_depth"):
        mesh_context(old, HORSE_TPOSE)
