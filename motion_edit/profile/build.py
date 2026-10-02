"""Build one species' Skeleton Profile and its report findings."""

from __future__ import annotations

import re
import zlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from motion_edit.profile import gait as gait_stats
from motion_edit.profile.data import Clip, DatasetSource, decode_clip, skeleton_hash
from motion_edit.profile.joints import (
    JointStats,
    axis_angle_deg,
    joint_stats,
    main_child,
)
from motion_edit.profile.skeleton import SkeletonStructure, resolve_contacts
from motion_edit.profile.spring import clip_rows, default_spring, fit_spring, fitted_spring
from motion_edit.rotations import quat_inv, quat_mul, quat_rotate, rotvec_from_quat

SCHEMA_VERSION = 2

# Split-half stability (section 3.7)
HINGE_AXIS_TOL_DEG = 15.0
# DOF classes are thresholds on the variance ratios; two estimates disagree on
# the DOF only when those ratios differ by more than this, not when they sit
# on either side of a class boundary.
DOF_RATIO_MARGIN = 0.1
# Left/right symmetry
SYMMETRY_AXIS_TOL_DEG = 20.0
SYMMETRY_STRONG_AXIS_DEG = 30.0   # reported by name; milder disagreements stay in the JSON
# Confidence
FULL_COVERAGE_CLIPS = 8
FULL_COVERAGE_FAMILIES = 3
UNSTABLE_FACTOR = 0.5
LOW_CONFIDENCE = 0.5
# A contact joint planted in less than this share of locomotion frames
RARE_CONTACT_SHARE = 0.05
# Joints whose name says they bend one way
HINGE_NAME = re.compile(r"knee|elbow|hiza|hiji", re.IGNORECASE)

COND_FIELDS = (
    "joints_names", "parents", "offsets", "scale_factor", "orientation_quat",
    "translation_root_index", "contact_joints", "joint_side_labels", "symmetry_partner_indices",
    "kinematic_chains", "species_tags", "axial_avg_len", "canonical_joint_names",
    "forward_joint_index", "forward_base_joint_index",
)


def cond_subset(entry: dict) -> dict:
    """The cond fields a profile build reads (keeps worker payloads small)."""
    return {k: entry[k] for k in COND_FIELDS if k in entry}


@dataclass
class Findings:
    """What the report lists for one species."""

    species: str
    clips: int = 0
    decode_failures: list = field(default_factory=list)
    override_notes: list = field(default_factory=list)
    low_confidence: list = field(default_factory=list)
    unstable: list = field(default_factory=list)
    passive_chains: list = field(default_factory=list)
    symmetry: list = field(default_factory=list)
    symmetry_axis: list = field(default_factory=list)
    name_dof: list = field(default_factory=list)
    rare_contacts: list = field(default_factory=list)
    fallback: bool = False


def _frame_weights(clips: list[Clip]) -> list[np.ndarray]:
    """Each clip sums to 1 / (clips in its family), each family to 1 / families."""
    families = Counter(c.family for c in clips)
    n_families = len(families)
    return [np.full(c.frame_count, 1.0 / (c.frame_count * families[c.family] * n_families))
            for c in clips]


def _split_halves(clips: list[Clip], seed: int) -> tuple[list[int], list[int]]:
    """Two halves with every action family spread across both (a random split
    of a few clips would otherwise compare attacks against idles)."""
    rng = np.random.default_rng(seed)
    by_family: dict[str, list[int]] = {}
    for i, c in enumerate(clips):
        by_family.setdefault(c.family, []).append(i)
    halves: tuple[list[int], list[int]] = ([], [])
    for family in sorted(by_family):
        members = [by_family[family][k] for k in rng.permutation(len(by_family[family]))]
        for i in members:
            # the smaller half takes the next clip; ties alternate by family order
            target = 0 if len(halves[0]) < len(halves[1]) else 1 if len(halves[1]) < len(halves[0]) else len(halves[0]) % 2
            halves[target].append(i)
    return sorted(halves[0]), sorted(halves[1])


def fold_anchor(structure, j: int) -> Optional[int]:
    """Nearest ancestor of ``j`` at a distinct rest position (zero-length bones
    put a parent on top of its child, which then measures nothing)."""
    anchor = int(structure.parents[j])
    while anchor >= 0 and np.linalg.norm(structure.rest[anchor] - structure.rest[j]) <= 1e-6:
        anchor = int(structure.parents[anchor])
    return anchor if anchor >= 0 else None


def _stats_for(clips, weights, structure, j) -> JointStats:
    child = main_child(structure.children[j], structure.bone_length)
    anchor = fold_anchor(structure, j)
    folds = None
    if child is not None and anchor is not None:
        folds = [np.linalg.norm(c.global_positions[:, child] - c.global_positions[:, anchor], axis=-1)
                 for c in clips]
    return joint_stats(
        [c.local_rotations[:, j] for c in clips], weights, [c.fps for c in clips], child,
        fold_distances=folds,
    )


def _dof_differs(a: JointStats, b: JointStats) -> bool:
    if (a.dof_class == "fixed") != (b.dof_class == "fixed"):
        return True
    if a.dof_class == "fixed":
        return False
    ra, rb = a.variance_ratio, b.variance_ratio
    return (abs(ra[0] - rb[0]) > DOF_RATIO_MARGIN
            or abs(ra[0] + ra[1] - rb[0] - rb[1]) > DOF_RATIO_MARGIN)


def _stability(full: JointStats, halves: list[JointStats]) -> list[str]:
    problems = []
    if full.dof_class == "fixed":
        return problems
    if _dof_differs(halves[0], halves[1]):
        problems.append("dof " + "/".join(
            f"{h.dof_class}({h.variance_ratio[0]:.2f})" for h in halves))
    if full.dof_class == "hinge" and all(h.dof_class == "hinge" for h in halves):
        angle = axis_angle_deg(halves[0].axes[0], halves[1].axes[0])
        if angle >= HINGE_AXIS_TOL_DEG:
            problems.append(f"hinge axis {angle:.0f}°")
    return problems


def _mirror(v: np.ndarray) -> np.ndarray:
    """Rotation axis seen in the mirror across the sagittal (X = 0) plane."""
    return v * np.array([1.0, -1.0, -1.0])


def _symmetry(structure, stats: dict, partners) -> tuple[list[str], list[str]]:
    """All left/right disagreements, and the strong axis mismatches among them
    (the ones a wrong left/right rig definition would produce)."""
    issues, strong = [], []
    for j, p in enumerate(partners):
        p = int(p)
        if p <= j:
            continue
        a, b = stats[j], stats[p]
        name = f"{structure.names[j]} ↔ {structure.names[p]}"
        if _dof_differs(a, b):
            issues.append(f"{name}: dof {a.dof_class}({a.variance_ratio[0]:.2f}) "
                          f"vs {b.dof_class}({b.variance_ratio[0]:.2f})")
            continue
        if a.dof_class != b.dof_class:
            continue
        # The defining axis: a hinge's rotation axis, a planar joint's plane normal.
        axis = {"hinge": 0, "planar": 2}.get(a.dof_class)
        if axis is None:
            continue
        angle = axis_angle_deg(a.axes[axis], _mirror(b.axes[axis]))
        if angle >= SYMMETRY_AXIS_TOL_DEG:
            label = "rotation axis" if axis == 0 else "plane normal"
            issues.append(f"{name}: mirrored {label} off by {angle:.0f}°")
            if angle >= SYMMETRY_STRONG_AXIS_DEG:
                strong.append(issues[-1])
    return issues, strong


def _spring_rows(clips, structure, j, mean, child):
    parent = int(structure.parents[j])
    bone = quat_rotate(mean, structure.offsets[child])
    bone = bone / (np.linalg.norm(bone) + 1e-12)
    rows = []
    for c in clips:
        theta = rotvec_from_quat(quat_mul(c.local_rotations[:, j], quat_inv(mean)[None]))
        rows.append(clip_rows(theta, c.global_rotations[:, parent], c.global_positions[:, parent],
                              bone, c.fps, c.is_loop))
    return rows


def _joint_json(structure, j, stats: JointStats, role, confidence, problems, spring):
    row = {
        "index": j,
        "name": structure.names[j],
        "role": role,
        "dof_class": stats.dof_class,
        "principal_axes": np.round(stats.axes, 5).tolist(),
        "variance_ratio": np.round(stats.variance_ratio, 4).tolist(),
        "angle_std_deg": round(stats.angle_std_deg, 3),
        "hinge_flex_sign": stats.flex_sign,
        "ang_speed": {"q50": round(stats.ang_speed[0], 4), "q95": round(stats.ang_speed[1], 4)},
        "confidence": round(confidence, 3),
    }
    if problems:
        row["unstable"] = problems
    if spring is not None:
        row.update(spring)
    return row


def build_species_profile(
    source: DatasetSource,
    cond_key: str,
    cond_entry: dict,
    motion_rows: dict,
    contact_override=None,
    passive_override=None,
) -> tuple[dict, Findings]:
    """Profile of one species from its clips (``motion_rows``: motion file -> metadata row)."""
    findings = Findings(species=cond_key)
    current_hash = skeleton_hash(cond_entry["parents"], cond_entry["offsets"])

    override_entries = None
    if contact_override is not None:
        if contact_override.stale(current_hash):
            findings.override_notes.append(
                "contact_overrides.json row is stale (skeleton_hash changed); ignored")
        else:
            override_entries = contact_override.entries
    contacts = resolve_contacts(cond_entry, override_entries)
    for name in contacts.unknown_names:
        findings.override_notes.append(f"contact override names unknown joint '{name}'")
    structure = SkeletonStructure(cond_entry, contacts.used)

    confirmed_names = set()
    if passive_override is not None:
        if passive_override.stale(current_hash):
            findings.override_notes.append(
                "passive_confirmations.json row is stale (skeleton_hash changed); ignored")
        else:
            confirmed_names = set(passive_override.entries.get("confirmed", []))
    candidate_names = {structure.names[j] for j in structure.passive_candidates()}
    for name in sorted(confirmed_names - candidate_names):
        findings.override_notes.append(
            f"passive_confirmations.json confirms '{name}', which is not a leaf-chain candidate")

    clips: list[Clip] = []
    for motion_name, row in sorted(motion_rows.items()):
        try:
            clips.append(decode_clip(source, motion_name, cond_entry, row, cond_key=cond_key))
        except Exception as exc:   # one broken clip must not lose the species
            findings.decode_failures.append(f"{motion_name}: {exc}")
    findings.clips = len(clips)

    leg = structure.leg_length
    profile = {
        "skeleton_hash": current_hash,
        "source": {"clips": len(clips),
                   "action_families": dict(sorted(Counter(c.family for c in clips).items()))},
        "global": {
            "leg_length": round(leg, 6),
            "has_legs": structure.has_legs,
            "hip_height": None if structure.hip_height is None else round(structure.hip_height / leg, 4),
            "axial_length": round(structure.axial_length / leg, 4) if leg > 0 else None,
        },
        "contacts": {"cond": contacts.cond, "override_add": sorted(contacts.added),
                     "override_remove": sorted(contacts.removed), "used": contacts.used},
    }
    if not clips:
        profile["joints"] = []
        profile["needs_fallback"] = True
        findings.fallback = True
        return profile, findings

    weights = _frame_weights(clips)
    n_families = len({c.family for c in clips})
    coverage = (min(1.0, len(clips) / FULL_COVERAGE_CLIPS)
                * (0.5 + 0.5 * min(1.0, n_families / FULL_COVERAGE_FAMILIES)))

    stats = [_stats_for(clips, weights, structure, j) for j in range(structure.joint_count)]

    halves_idx = _split_halves(clips, zlib.crc32(cond_key.encode("utf-8")))
    split_ok = all(len(h) > 0 for h in halves_idx)
    problems = [[] for _ in range(structure.joint_count)]
    if split_ok:
        for j in range(structure.joint_count):
            halves = []
            for idx in halves_idx:
                sub = [clips[i] for i in idx]
                halves.append(_stats_for(sub, _frame_weights(sub), structure, j))
            problems[j] = _stability(stats[j], halves)

    # Secondary motion: every leaf-chain candidate gets a spring -- the fitted
    # one when the data shows real driven motion, a default one otherwise --
    # and is simulated only once the user confirms it.
    roles = structure.base_roles()
    base_roles = list(roles)
    springs = {}
    for chain, candidates in structure.passive_chains():
        passed, confirmed = [], []
        for j in candidates:
            name = structure.names[j]
            fit = None
            child = stats[j].main_child
            if child is not None and stats[j].dof_class != "fixed":
                fit = fit_spring(_spring_rows(clips, structure, j, stats[j].mean, child))
            if fit is not None and fit.passed:
                spring = fitted_spring(fit)
                passed.append(name)
            else:
                spring = default_spring(structure.swing_length(chain, j), leg)
            spring["swing_length"] = round(structure.swing_length(chain, j) / leg, 4) if leg > 0 else None
            springs[j] = {"spring": spring,
                          "spring_fit": None if fit is None else fit.as_json(),
                          "passive_confirmed": name in confirmed_names}
            if name in confirmed_names:
                roles[j] = "passive"
                confirmed.append(name)
        findings.passive_chains.append({
            "joints": [structure.names[j] for j in candidates],
            "role": base_roles[candidates[0]],
            "fit_passed": passed,
            "confirmed": confirmed,
        })

    joints = []
    for j in range(structure.joint_count):
        confidence = coverage * (1.0 if (split_ok and not problems[j]) else UNSTABLE_FACTOR)
        joints.append(_joint_json(structure, j, stats[j], roles[j], confidence, problems[j],
                                  springs.get(j)))
        if problems[j]:
            findings.unstable.append(structure.names[j])
        if confidence < LOW_CONFIDENCE and stats[j].dof_class != "fixed":
            findings.low_confidence.append(structure.names[j])
    profile["joints"] = joints

    # Left and right only have to agree over symmetric motion: a one-armed
    # attack or a turn makes the two sides differ by design.
    partners = cond_entry.get("symmetry_partner_indices")
    loco = [c for c in clips if c.action_group == "locomotion"]
    if partners is not None and loco:
        loco_weights = _frame_weights(loco)
        paired = {j for j, p in enumerate(partners) if int(p) >= 0}
        loco_stats = {j: _stats_for(loco, loco_weights, structure, j) for j in paired}
        findings.symmetry, findings.symmetry_axis = _symmetry(structure, loco_stats, partners)

    canonical = cond_entry.get("canonical_joint_names") or [""] * structure.joint_count
    for j in range(structure.joint_count):
        label = f"{structure.names[j]} {canonical[j]}"
        if HINGE_NAME.search(label) and stats[j].dof_class != "hinge":
            findings.name_dof.append(f"{structure.names[j]} ({canonical[j]}): {stats[j].dof_class}")

    gaits, planted, loco_frames = [], np.zeros(len(contacts.used)), 0
    limbs = structure.contact_limbs()
    for c in clips:
        if c.action_group != "locomotion" or not contacts.used:
            continue
        result = gait_stats.clip_contacts(c, contacts.used, leg)
        planted += result.mask.sum(axis=0)
        loco_frames += c.frame_count
        g = gait_stats.clip_gait(c, contacts.used, limbs, leg, result)
        if g is not None:
            gaits.append(g)
    if loco_frames:
        for k, joint in enumerate(contacts.used):
            share = planted[k] / loco_frames
            if share < RARE_CONTACT_SHARE:
                findings.rare_contacts.append(f"{structure.names[joint]} ({share:.1%})")
    profile["gait"] = gait_stats.aggregate_gait(gaits)
    profile["stride_speed_fit"] = gait_stats.stride_speed_fit(gaits)
    vertical = [v for v in (gait_stats.clip_vertical(c, structure.root, leg) for c in clips) if v]
    profile["vertical"] = gait_stats.aggregate_vertical(vertical)
    return profile, findings


# ── no-motion fallback (section 3.9) ─────────────────────────────────────────

def _fallback_joint(structure, j, donors_by_name, role):
    rows = donors_by_name.get(structure.canonical[j], [])
    if rows:
        dof = Counter(r["dof_class"] for r in rows).most_common(1)[0][0]
        same = [r for r in rows if r["dof_class"] == dof]
        flex = Counter(r["hinge_flex_sign"] for r in same).most_common(1)[0][0]
        return {"index": j, "name": structure.names[j], "role": role, "dof_class": dof,
                "principal_axes": same[0]["principal_axes"], "hinge_flex_sign": flex,
                "confidence": 0.0, "fallback_donors": len(same)}
    return {"index": j, "name": structure.names[j], "role": role, "dof_class": "ball",
            "principal_axes": np.eye(3).tolist(), "hinge_flex_sign": 0,
            "confidence": 0.0, "fallback_donors": 0}


def apply_fallback(profile: dict, cond_entry: dict, donors: list[tuple[dict, dict]]) -> None:
    """Fill a clip-less profile from same-``species_tags`` profiles, by canonical joint name.

    ``donors`` is ``[(cond_entry, profile), ...]`` of species that have clips.
    """
    contacts = profile["contacts"]["used"]
    structure = SkeletonStructure(cond_entry, contacts)
    structure.canonical = list(cond_entry.get("canonical_joint_names") or structure.names)
    tags = tuple(cond_entry.get("species_tags") or ())
    by_name: dict[str, list[dict]] = {}
    for donor_cond, donor_profile in donors:
        if tuple(donor_cond.get("species_tags") or ()) != tags:
            continue
        donor_names = list(donor_cond.get("canonical_joint_names") or donor_cond["joints_names"])
        for row in donor_profile.get("joints", []):
            by_name.setdefault(donor_names[row["index"]], []).append(row)
    roles = structure.base_roles()
    profile["joints"] = [_fallback_joint(structure, j, by_name, roles[j])
                         for j in range(structure.joint_count)]
    profile["needs_fallback"] = False
    profile["fallback"] = True
