"""Skeleton Profile extractor (motion_edit/profile, motion_edit/contacts).

Usage:
    pytest tests/test_motion_edit_profile.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion_edit.contacts import contact_intervals, detect_contacts
from motion_edit.profile import gait as gait_stats
from motion_edit.profile.build import _split_halves, apply_fallback
from motion_edit.profile.data import Clip, skeleton_hash
from motion_edit.profile.joints import classify_dof, joint_stats
from motion_edit.profile.skeleton import SkeletonStructure, resolve_contacts
from motion_edit.profile.spring import clip_rows, default_spring, fit_spring
from motion_edit.rotations import (
    quat_from_rotvec,
    quat_mul,
    quat_rotate,
    rotation_between,
    rotvec_from_quat,
    weighted_mean_quat,
)

FPS = 30.0


# ── rotations ────────────────────────────────────────────────────────────────

def test_rotvec_round_trip_and_rotation_direction():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(50, 3))
    v *= (rng.uniform(0.0, 3.0, size=(50, 1)) / np.linalg.norm(v, axis=1, keepdims=True))
    assert np.allclose(rotvec_from_quat(quat_from_rotvec(v)), v, atol=1e-9)
    q = quat_from_rotvec(np.array([0.0, 0.0, np.pi / 2]))
    assert np.allclose(quat_rotate(q, np.array([1.0, 0.0, 0.0])), [0.0, 1.0, 0.0], atol=1e-12)



def test_rotation_between_turns_opposite_directions():
    for u in (np.array([1.0, 0.0, 0.0]), np.array([0.3, -2.0, 0.5])):
        q = rotation_between(u, -u)
        assert np.allclose(quat_rotate(q, u), -u, atol=1e-9)
    assert np.allclose(rotation_between(np.zeros(3), np.array([1.0, 0.0, 0.0])), [1.0, 0.0, 0.0, 0.0])

def test_weighted_mean_quat_ignores_sign_and_follows_weights():
    a = quat_from_rotvec(np.array([0.2, 0.0, 0.0]))
    b = quat_from_rotvec(np.array([0.4, 0.0, 0.0]))
    mean = weighted_mean_quat(np.stack([a, -a, b]), np.array([1.0, 1.0, 1e-9]))
    assert np.allclose(rotvec_from_quat(mean), [0.2, 0.0, 0.0], atol=1e-6)


# ── skeleton structure ───────────────────────────────────────────────────────

def _biped_cond():
    # root(0) - pelvis(1) - L hip(2) knee(3) ankle(4) toe(5); R hip(6) knee(7) ankle(8) toe(9);
    # spine(10) - head(11); tail(12) - tail tip(13)
    parents = [-1, 0, 1, 2, 3, 4, 1, 6, 7, 8, 1, 10, 1, 12]
    offsets = np.array([
        [0, 1.0, 0], [0, 0, 0],
        [0.2, -0.05, 0], [0, -0.45, 0], [0, -0.45, 0], [0, -0.05, 0.1],
        [-0.2, -0.05, 0], [0, -0.45, 0], [0, -0.45, 0], [0, -0.05, 0.1],
        [0, 0.3, 0], [0, 0.3, 0],
        [0, 0, -0.3], [0, 0, -0.3],
    ], dtype=np.float64)
    sides = ["center", "center"] + ["left"] * 4 + ["right"] * 4 + ["center"] * 4
    names = ["root", "pelvis", "l_hip", "l_knee", "l_ankle", "l_toe",
             "r_hip", "r_knee", "r_ankle", "r_toe", "spine", "head", "tail", "tail_tip"]
    return {
        "joints_names": names, "parents": parents, "offsets": offsets,
        "joint_side_labels": sides, "contact_joints": [4, 5, 8, 9],
        "translation_root_index": 0, "axial_avg_len": 0.3, "scale_factor": 1.0,
        "species_tags": ("Biped", "Walking"),
        "canonical_joint_names": ["Root", "Hips", "Left Thigh", "Left Knee", "Left Foot", "Left Toe",
                                  "Right Thigh", "Right Knee", "Right Foot", "Right Toe",
                                  "Spine", "Head", "Tail 01", "Tail 02"],
    }


def test_contact_limbs_leg_length_and_roles():
    cond = _biped_cond()
    s = SkeletonStructure(cond, cond["contact_joints"])
    assert s.contact_limbs() == {2: [5, 4], 6: [9, 8]}       # lowest-at-rest first
    # hip -> toe: knee 0.45 + ankle 0.45 + toe |(0,-0.05,0.1)|
    expected = 0.9 + np.hypot(0.05, 0.1)
    assert s.leg_length == pytest.approx(expected)
    roles = s.base_roles()
    assert roles[0] == "root"
    assert all(roles[j] == "support" for j in (2, 3, 4, 5, 6, 7, 8, 9))
    assert roles[10] == roles[11] == roles[12] == "axial"
    # leaf chains off the legs: spine -> head and the tail (tips have no bone to swing)
    assert s.passive_candidates() == [10, 12]


def test_resolve_contacts_applies_name_overrides():
    cond = _biped_cond()
    contacts = resolve_contacts(cond, {"add": ["tail_tip", "nope"], "remove": ["l_ankle"]})
    assert contacts.used == [5, 8, 9, 13]
    assert contacts.unknown_names == ["nope"]


def test_skeleton_hash_tracks_offsets():
    cond = _biped_cond()
    h = skeleton_hash(cond["parents"], cond["offsets"])
    moved = cond["offsets"].copy()
    moved[3, 1] += 1e-3
    assert h == skeleton_hash(cond["parents"], cond["offsets"] + 1e-9)
    assert h != skeleton_hash(cond["parents"], moved)


# ── contact intervals ───────────────────────────────────────────────────────

def _treadmill_walk(frames=60, speed=1.0, leg=1.0, periodic_phase=0.0):
    """Two feet on a treadmill: stance slides back at ``speed`` on the floor, swing lifts."""
    t = np.arange(frames)
    pos = np.zeros((frames, 2, 3))
    duty = 0.6
    period = frames / 2.0                        # two cycles per clip
    stride = speed * duty * period / FPS         # distance a planted foot slides back
    for k, offset in enumerate((0.0, 0.5)):
        phase = ((t / period) + offset + periodic_phase) % 1.0
        stance = phase < duty
        s = np.where(stance, phase / duty, (phase - duty) / (1 - duty))
        z = np.where(stance, stride / 2 - stride * s, -stride / 2 + stride * s)
        y = np.where(stance, 0.0, 0.15 * leg * np.sin(np.pi * s))
        pos[:, k, 2] = z
        pos[:, k, 1] = y
        pos[:, k, 0] = 0.2 if k == 0 else -0.2
    truth = np.stack([(((t / period) + o + periodic_phase) % 1.0) < duty for o in (0.0, 0.5)], axis=1)
    return pos, truth


def test_detect_contacts_on_treadmill_walk():
    pos, truth = _treadmill_walk()
    result = detect_contacts(pos, [0, 1], 1.0, FPS, periodic=True)
    agreement = (result.mask == truth).mean()
    assert agreement > 0.85
    stance = truth.any(axis=1)
    speed = np.linalg.norm(result.ground_velocity[stance], axis=1).mean()
    assert speed == pytest.approx(1.0, rel=0.2)


def test_tucked_feet_never_plant():
    pos, _ = _treadmill_walk()
    pos[:, 1, 1] += 0.8                                   # second foot held far above the ground
    result = detect_contacts(pos, [0, 1], 1.0, FPS, periodic=True)
    assert not result.mask[:, 1].any()


def test_hovering_feet_plant_only_on_their_ground():
    pos, truth = _treadmill_walk()
    pos[..., 1] += 1.0                                    # the whole clip a leg above y = 0
    assert not detect_contacts(pos, [0, 1], 1.0, FPS, periodic=True).mask.any()
    result = detect_contacts(pos, [0, 1], 1.0, FPS, periodic=True, ground=1.0)
    assert (result.mask == truth).mean() > 0.85


def test_two_frame_stance_of_a_fast_run_plants():
    # Every stance frame borders a touchdown or lift-off frame: only a
    # one-sided vertical difference reads it as still.
    frames, period = 20, 10
    pos = np.zeros((frames, 1, 3))
    phase = np.arange(frames) % period
    pos[:, 0, 1] = np.where(phase < 2, 0.0, 0.3 * np.sin(np.pi * (phase - 1) / (period - 1)))
    pos[:, 0, 2] = np.where(phase < 2, -0.15 * (phase - 0.5), 0.0)   # 4.5 leg/s backwards
    result = detect_contacts(pos, [0], 1.0, FPS, periodic=True)
    assert contact_intervals(result.mask[:, 0], periodic=True) == [(0, 2), (10, 12)]


def _gait_clip(pos: np.ndarray) -> Clip:
    frames, joints = pos.shape[:2]
    rot = np.zeros((frames, joints, 4))
    rot[..., 0] = 1.0
    return Clip(name="gait", action_group="locomotion", action_label="walk", is_loop=True, fps=FPS,
                local_rotations=rot, global_rotations=rot, global_positions=pos)


def test_gait_needs_repeated_steps():
    pos, _ = _treadmill_walk()
    gait = gait_stats.clip_gait(_gait_clip(pos), [0, 1], {0: [0], 1: [1]}, 1.0)
    assert gait is not None and gait.period == pytest.approx(30.0)
    # feet resettling at irregular times: touchdown spacings 8, 32, 20 and 60 frames
    frames = 60
    pos = np.zeros((frames, 2, 3))
    pos[:, :, 1] = 0.3
    for foot, stances in ((0, [(0, 5), (8, 13), (40, 45)]), (1, [(25, 30)])):
        for start, end in stances:
            pos[start:end, foot, 1] = 0.0
    assert gait_stats.clip_gait(_gait_clip(pos), [0, 1], {0: [0], 1: [1]}, 1.0) is None


def test_gait_ignores_a_flicker_without_lift_off():
    mask = np.zeros(24, dtype=bool)
    mask[1:9] = mask[11:13] = True                        # one stance broken at frames 9-10
    height = np.where(mask, 0.0, 0.3)
    height[9:11] = 0.02                                   # the foot never left the floor
    filled = gait_stats._fill_unlifted_gaps(mask, height, periodic=True)
    assert contact_intervals(filled, periodic=True) == [(1, 13)]
    height[9:11] = 0.3                                    # a real swing stays a new step
    assert (gait_stats._fill_unlifted_gaps(mask, height, periodic=True) == mask).all()


def test_periodic_seam_plant_is_one_interval():
    mask = np.zeros(20, dtype=bool)
    mask[:4] = True
    mask[15:] = True
    assert contact_intervals(mask, periodic=True) == [(15, 24)]
    assert contact_intervals(mask, periodic=False) == [(0, 4), (15, 20)]


# ── joint statistics ────────────────────────────────────────────────────────

def test_classify_dof_thresholds():
    assert classify_dof(np.array([0.9, 0.08, 0.02]), 0.5) == "fixed"
    assert classify_dof(np.array([0.9, 0.08, 0.02]), 10) == "hinge"
    assert classify_dof(np.array([0.6, 0.35, 0.05]), 10) == "planar"
    assert classify_dof(np.array([0.4, 0.35, 0.25]), 10) == "ball"


def _knee(angles_deg, seed=1):
    """A knee bending about +X: rotations, and the hip-to-ankle fold distance."""
    rng = np.random.default_rng(seed)
    n = len(angles_deg)
    rv = np.zeros((n, 3))
    rv[:, 0] = np.radians(angles_deg)
    rv[:, 1:] = np.radians(rng.normal(0.0, 0.5, (n, 2)))
    q = quat_from_rotvec(rv)
    own = np.array([0.0, -0.5, 0.0])                      # knee below the hip
    child = np.array([0.0, -0.5, 0.0])                    # ankle below the knee
    fold = np.linalg.norm(own + quat_rotate(q, child[None]), axis=-1)
    return q, fold


def _folds_towards_plus_x(stats) -> bool:
    return stats.flex_sign * stats.axes[0, 0] > 0


def test_joint_stats_recovers_hinge_axis_and_flex_sign():
    q, fold = _knee(np.random.default_rng(1).uniform(0.0, 90.0, 4000))
    stats = joint_stats([q], [np.full(4000, 1.0)], [FPS], 1, fold_distances=[fold])
    assert stats.dof_class == "hinge"
    assert abs(stats.axes[0, 0]) > 0.99
    assert _folds_towards_plus_x(stats)


def test_flex_sign_of_a_mostly_straight_leg():
    # Straight most of the time, occasionally folding: the mean pose is the
    # straight leg, where both directions shorten the limb equally.
    angles = np.where(np.random.default_rng(2).uniform(size=4000) < 0.8, 0.0,
                      np.random.default_rng(3).uniform(10.0, 70.0, 4000))
    q, fold = _knee(angles)
    stats = joint_stats([q], [np.full(4000, 1.0)], [FPS], 1, fold_distances=[fold])
    assert stats.dof_class == "hinge"
    assert _folds_towards_plus_x(stats)


def test_flex_sign_undetermined_without_folding():
    q, fold = _knee(np.zeros(500))
    stats = joint_stats([q], [np.full(500, 1.0)], [FPS], 1,
                        fold_distances=[np.full(500, fold.mean())])
    assert stats.flex_sign == 0


# ── spring fit ──────────────────────────────────────────────────────────────

def _driven_spring(k=40.0, c=2.0, g=1.0, frames=600, seed=0):
    """A tail joint driven by its parent's yaw acceleration, integrated finely."""
    rng = np.random.default_rng(seed)
    sub = 20
    dt = 1.0 / (FPS * sub)
    n = frames * sub
    knots = rng.normal(0.0, 0.6, frames // 15 + 2)
    yaw = np.interp(np.arange(n) / (15 * sub), np.arange(len(knots)), knots)
    yaw = np.convolve(yaw, np.ones(sub * 4) / (sub * 4), mode="same")
    alpha = np.gradient(np.gradient(yaw, dt), dt)
    theta = np.zeros(n)
    vel = 0.0
    for i in range(1, n):
        acc = -k * theta[i - 1] - c * vel - g * alpha[i - 1]
        vel += acc * dt
        theta[i] = theta[i - 1] + vel * dt
    keep = slice(None, None, sub)
    parent_rot = quat_from_rotvec(np.stack([np.zeros(n), yaw, np.zeros(n)], axis=1))[keep]
    joint_rv = np.stack([np.zeros(n), theta, np.zeros(n)], axis=1)[keep]
    return joint_rv, parent_rot


def test_spring_fit_accepts_a_driven_spring():
    theta, parent_rot = _driven_spring()
    rows = [clip_rows(theta, parent_rot, np.zeros((theta.shape[0], 3)),
                      np.array([0.0, 0.0, -1.0]), FPS, False)]
    fit = fit_spring(rows)
    assert fit is not None and fit.passed, fit.reason
    assert fit.k == pytest.approx(40.0, rel=0.3)


def test_spring_fit_rejects_an_undriven_keyed_wag():
    frames = 600
    t = np.arange(frames) / FPS
    theta = np.zeros((frames, 3))
    theta[:, 1] = 0.4 * np.sin(2 * np.pi * 1.0 * t)           # hand-keyed wag
    rng = np.random.default_rng(3)
    yaw = np.cumsum(rng.normal(0, 0.01, frames))               # unrelated parent motion
    parent_rot = quat_from_rotvec(np.stack([np.zeros(frames), yaw, np.zeros(frames)], axis=1))
    rows = [clip_rows(theta, parent_rot, np.zeros((frames, 3)), np.array([0.0, 0.0, -1.0]), FPS, False)]
    fit = fit_spring(rows)
    assert fit is not None and not fit.passed
    assert fit.r2 > 0.6                                         # fits well ...
    assert fit.delta_r2 < 0.2                                   # ... without the drive


def test_default_spring_slows_with_chain_length():
    short = default_spring(0.2, 1.0)
    long = default_spring(2.0, 1.0)
    assert short["source"] == long["source"] == "default"
    assert short["natural_hz"] > long["natural_hz"]
    assert long["natural_hz"] == pytest.approx(1.5 / np.sqrt(2.0), rel=1e-3)
    for spring in (short, long):
        zeta = spring["c"] / (2.0 * np.sqrt(spring["k"]))
        assert zeta == pytest.approx(0.3, rel=1e-3)
        assert spring["g_ang"] == 1.0 and spring["g_grav"] == 0.0
    assert default_spring(1e-4, 1.0)["natural_hz"] == 4.0          # clamped
    assert short["g_lin"] == pytest.approx(1.5 / 0.2)


def test_swing_length_and_passive_chains():
    cond = _biped_cond()
    s = SkeletonStructure(cond, cond["contact_joints"])
    chains = {tuple(c): tuple(k) for c, k in s.passive_chains()}
    assert chains == {(10, 11): (10,), (12, 13): (12,)}
    assert s.swing_length([12, 13], 12) == pytest.approx(0.3)


# ── build helpers ───────────────────────────────────────────────────────────

def _clip(name, group, label, frames=10):
    z = np.zeros((frames, 1, 4))
    z[..., 0] = 1.0
    return Clip(name, group, label, False, FPS, z, z, np.zeros((frames, 1, 3)))


def test_split_halves_spreads_every_family():
    clips = ([_clip(f"w{i}", "locomotion", "walk") for i in range(4)]
             + [_clip(f"a{i}", "stationary", "attack") for i in range(4)])
    a, b = _split_halves(clips, seed=7)
    assert sorted(a + b) == list(range(8))
    for half in (a, b):
        assert {clips[i].family for i in half} == {"locomotion|walk", "stationary|attack"}


def test_fallback_borrows_by_canonical_name_within_species_tags():
    cond = _biped_cond()
    donor_profile = {"joints": [
        {"index": j, "dof_class": "hinge" if j in (3, 7) else "ball", "hinge_flex_sign": 1,
         "principal_axes": np.eye(3).tolist()}
        for j in range(14)]}
    other_tags = dict(cond, species_tags=("Quadruped", "Walking"))
    profile = {"contacts": {"used": cond["contact_joints"]}, "needs_fallback": True}
    apply_fallback(profile, cond, [(cond, donor_profile), (other_tags, {"joints": []})])
    knee = profile["joints"][3]
    assert knee["dof_class"] == "hinge" and knee["fallback_donors"] == 1
    assert knee["confidence"] == 0.0 and profile["fallback"]

    profile = {"contacts": {"used": cond["contact_joints"]}, "needs_fallback": True}
    apply_fallback(profile, cond, [(other_tags, donor_profile)])
    assert profile["joints"][3]["dof_class"] == "ball"


# ── end to end on a real species (skipped without the dataset) ─────────────

def test_build_alligator_profile_from_dataset():
    from motion_edit.profile.data import discover_sources, load_cond
    sources = {s.namespace: s for s in discover_sources()}
    source = sources.get("truebones/zoo")
    if source is None or not os.path.isfile(os.path.join(source.root, "cond.npy")):
        pytest.skip("truebones/zoo dataset not present")
    from data_loaders.truebones.truebones_utils.motion_labels import load_motion_metadata
    from motion_edit.profile.build import build_species_profile, cond_subset
    from motion_edit.profile.data import species_motion_names

    cond = load_cond(source)
    key = "truebones/zoo/Alligator"
    entry = cond_subset(cond[key])
    metadata = load_motion_metadata(source.root)
    rows = {n: metadata[n] for n in species_motion_names(metadata, "Alligator")}
    from motion_edit.profile.data import SpeciesOverride
    confirm = SpeciesOverride(entries={"confirmed": ["sippo03", "R_ashi"]})
    profile, findings = build_species_profile(source, key, entry, rows, passive_override=confirm)
    assert findings.clips == len(rows) > 0 and not findings.decode_failures
    assert len(profile["joints"]) == len(entry["parents"])
    assert profile["contacts"]["used"] == sorted(entry["contact_joints"])
    elbows = [j for j in profile["joints"] if j["name"] in ("R_hiji", "L_hiji")]
    assert all(j["dof_class"] == "hinge" and j["hinge_flex_sign"] != 0 for j in elbows)
    assert "locomotion|walk" in profile["source"]["action_families"]
    assert profile["gait"], "the walk clip should yield a gait entry"
    tail = next(j for j in profile["joints"] if j["name"] == "sippo03")
    assert tail["role"] == "passive" and tail["passive_confirmed"]
    assert tail["spring"]["source"] in ("fit", "default") and tail["spring"]["k"] > 0
    unconfirmed = next(j for j in profile["joints"] if j["name"] == "sippo02")
    assert unconfirmed["role"] == "axial" and not unconfirmed["passive_confirmed"]
    # a foot is not a leaf-chain candidate: the confirmation is reported, not applied
    assert any("R_ashi" in note for note in findings.override_notes)
