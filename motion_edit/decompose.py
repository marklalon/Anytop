"""Decomposer: a generation result (F, J, 12) -> Edit Package (section 4).

1. Decode with full-body IK (``restore_animation_from_features``,
   ``restore_space="hml"``, ``fullbody_ik=True``, ``stretch_factor=s``), so
   the position channels are carried by rotations and bones stay within
   ``[1 - s, 1 + s]`` of their rest length.
2. Detect the contact intervals of the clip's contact set
   (``motion_edit.contacts``, shared with the profile).
3. Read the events: the loop's gait period and per-limb touchdowns, and the
   airborne spans of a clip with real vertical motion.
4. Split into layers: ``base`` (the decoded clip itself), ``root`` (trend /
   oscillation of the root translation, yaw), ``chains`` (each joint's
   rotation vector off a reference pose), ``plants`` (per contact interval:
   ground-frame anchor, the root-relative anchor, the slip residual).

The package also keeps the source features, the cond subset and the profile
subset, so the server can re-decompose it with another ``stretch_factor``
without the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.ndimage import uniform_filter1d

from data_loaders.truebones.truebones_utils.param_utils import FPS
from motion_edit.contacts import (
    ContactParams,
    ContactResult,
    contact_intervals,
    detect_contacts,
    ground_velocity_from_mask,
)
from motion_edit.package import (
    NEAR_PI,
    RUNTIME_VERSION,
    EditPackage,
    decode_json,
    encode_json,
)
from motion_edit.profile import gait as gait_stats
from motion_edit.profile.build import COND_FIELDS, SCHEMA_VERSION as PROFILE_SCHEMA
from motion_edit.profile.data import Clip, skeleton_hash
from motion_edit.profile.skeleton import CENTER, SkeletonStructure, resolve_contacts
from motion_edit.rotations import (
    quat_inv,
    quat_mul,
    rotvec_from_quat,
    weighted_mean_quat,
)
from motion_edit.runtime import (
    available_params,
    forward_kinematics,
    ground_displacement,
    param_manifest,
    yaw_quat,
)

PACKAGE_COND_FIELDS = tuple(dict.fromkeys(COND_FIELDS + ("canonical_bvh_joint_names", "end_effector_joints")))

# Full-body IK bone stretch / compress bound (fraction of bone length) used to
# decode an edit package.
DEFAULT_STRETCH_FACTOR = 0.2
# Low-pass window of the root trend for a one-shot clip; a loop uses its period.
ONE_SHOT_TREND_SECONDS = 0.5
# Shortest span with no planted contact that counts as airborne.
MIN_AIRBORNE_FRAMES = 2
# Root yaw range above which a clip counts as turning: stride assumes a
# straight walk and is not offered (section 8).
TURNING_YAW_DEG = 25.0

ROLE_CODES = ("root", "axial", "support", "swing", "passive", "other")
DOF_CODES = ("fixed", "hinge", "planar", "ball")
SPRING_FIELDS = ("k", "c", "g_ang", "g_lin", "g_grav")


class StaleProfileError(ValueError):
    pass


# ── inputs ───────────────────────────────────────────────────────────────────

@dataclass
class ContactSource:
    """Where the clip's contact set came from (manifest provenance, joint indices)."""

    cond: list[int]
    species_add: list[int] = field(default_factory=list)
    species_remove: list[int] = field(default_factory=list)
    package_add: list[int] = field(default_factory=list)
    package_remove: list[int] = field(default_factory=list)
    unknown_names: list[str] = field(default_factory=list)

    @property
    def used(self) -> list[int]:
        joints = (set(self.cond) | set(self.species_add)) - set(self.species_remove)
        return sorted((joints | set(self.package_add)) - set(self.package_remove))

    def as_dict(self) -> dict:
        return {"cond": self.cond, "species_add": self.species_add,
                "species_remove": self.species_remove, "package_add": self.package_add,
                "package_remove": self.package_remove}

    @classmethod
    def from_dict(cls, row: dict) -> "ContactSource":
        return cls(**{k: [int(j) for j in row.get(k, [])]
                      for k in ("cond", "species_add", "species_remove", "package_add", "package_remove")})


def contact_source(cond_entry: dict, species_override: Optional[dict] = None) -> ContactSource:
    """cond's contacts with a species ``contact_overrides.json`` row (already hash-checked)."""
    resolved = resolve_contacts(cond_entry, species_override)
    return ContactSource(cond=list(resolved.cond), species_add=sorted(resolved.added),
                         species_remove=sorted(resolved.removed),
                         unknown_names=list(resolved.unknown_names))


@dataclass
class ProfileSubset:
    """What the package keeps of a species profile."""

    status: str                      # ok / fallback / missing
    leg_length: Optional[float] = None
    gait: Optional[dict] = None      # the profile's gait row for this action_label
    stride_speed_fit: Optional[dict] = None
    arrays: Optional[dict] = None    # profile_* arrays, None without a profile


def check_profile(profile: Optional[dict], cond_entry: dict) -> str:
    """``ok`` / ``fallback`` / ``missing``; a profile of another skeleton raises."""
    if not profile or not profile.get("joints"):
        return "missing"
    current = skeleton_hash(cond_entry["parents"], cond_entry["offsets"])
    if profile.get("skeleton_hash") != current:
        raise StaleProfileError(
            "skeleton profile was built for another skeleton (skeleton_hash differs); "
            "rebuild it with python -m motion_edit.build_profiles")
    return "fallback" if profile.get("fallback") else "ok"


def profile_subset(profile: Optional[dict], cond_entry: dict, action_label: str = "") -> ProfileSubset:
    status = check_profile(profile, cond_entry)
    if status == "missing":
        return ProfileSubset(status)
    joint_count = len(cond_entry["parents"])
    role = np.full(joint_count, "", dtype="<U8")
    dof = np.full(joint_count, "ball", dtype="<U8")
    axes = np.tile(np.eye(3), (joint_count, 1, 1))
    flex = np.zeros(joint_count, dtype=np.int8)
    spring = np.full((joint_count, len(SPRING_FIELDS)), np.nan)
    passive = np.zeros(joint_count, dtype=bool)
    confidence = np.zeros(joint_count)
    for row in profile["joints"]:
        j = int(row["index"])
        role[j] = row.get("role", "")
        dof[j] = row.get("dof_class", "ball")
        if row.get("principal_axes") is not None:
            axes[j] = np.asarray(row["principal_axes"], dtype=np.float64)
        flex[j] = int(row.get("hinge_flex_sign", 0) or 0)
        confidence[j] = float(row.get("confidence", 0.0) or 0.0)
        if row.get("spring"):
            spring[j] = [float(row["spring"].get(k, np.nan)) for k in SPRING_FIELDS]
        passive[j] = row.get("role") == "passive"
    leg = float(profile.get("global", {}).get("leg_length") or 0.0)
    return ProfileSubset(
        status=status,
        leg_length=leg if leg > 0 else None,
        gait=(profile.get("gait") or {}).get(action_label),
        stride_speed_fit=profile.get("stride_speed_fit"),
        arrays={"profile_role": role, "profile_dof": dof, "profile_axes": axes,
                "profile_flex_sign": flex, "profile_spring": spring,
                "profile_passive": passive, "profile_confidence": confidence},
    )


def cond_subset(cond_entry: dict) -> dict:
    return {k: cond_entry[k] for k in PACKAGE_COND_FIELDS if k in cond_entry}


# ── layers ───────────────────────────────────────────────────────────────────

def chain_groups(structure: SkeletonStructure, roles: list[str], canonical_names: list[str]) -> np.ndarray:
    """Amplitude group of every joint (``CHAIN_GROUPS`` codes).

    Support chains are legs; other sided limbs are arms, or wings by name;
    center joints are axial, or tail by name.  A joint inherits its parent's
    tail / wing group, so an unnamed tip stays with its chain.
    """
    groups = np.full(structure.joint_count, "other", dtype="<U8")
    for j in range(structure.joint_count):        # parents precede children
        name = canonical_names[j].lower()
        parent = int(structure.parents[j])
        inherited = groups[parent] if parent >= 0 else ""
        role = roles[j]
        if role == "root":
            groups[j] = "root"
        elif role == "support":
            groups[j] = "legs"
        elif role in ("swing", "passive") and structure.sides[j] != CENTER:
            groups[j] = "wings" if ("wing" in name or inherited == "wings") else "arms"
        elif structure.sides[j] == CENTER and role in ("axial", "passive"):
            groups[j] = "tail" if ("tail" in name or inherited == "tail") else "axial"
        else:
            groups[j] = "other"
    return groups


def split_root(root_pos: np.ndarray, root_rot: np.ndarray, *, window: int, periodic: bool):
    """Root translation as low-pass trend + oscillation, rotation as world-yaw x tilt."""
    window = max(1, min(int(window), root_pos.shape[0]))
    trend = uniform_filter1d(root_pos.astype(np.float64), size=window, axis=0,
                             mode="wrap" if periodic else "nearest")
    osc = root_pos - trend
    w, y = root_rot[:, 0], root_rot[:, 2]
    yaw = np.unwrap(2.0 * np.arctan2(y, w))
    tilt = quat_mul(quat_inv(yaw_quat(yaw)), root_rot)
    return trend, osc, yaw, tilt


def chain_layer(rotations: np.ndarray, root: int, *, periodic: bool):
    """Reference pose (loop: per-joint mean; one-shot: first frame) and rotation-vector offsets."""
    frame_count, joint_count = rotations.shape[:2]
    if periodic:
        reference = np.stack([weighted_mean_quat(rotations[:, j], np.ones(frame_count))
                              for j in range(joint_count)])
    else:
        reference = np.array(rotations[0], dtype=np.float64)
    offsets = rotvec_from_quat(quat_mul(rotations, quat_inv(reference)[None]))
    offsets[:, root] = 0.0
    near_pi = np.linalg.norm(offsets, axis=-1) > NEAR_PI
    return reference, offsets, near_pi


def plant_layer(positions: np.ndarray, contacts: list[int], mask: np.ndarray,
                ground_velocity: np.ndarray, fps: float, root: int, periodic: bool):
    """Per contact interval: anchor in the ground frame, root-relative anchor, slip residual."""
    frame_count = positions.shape[0]
    displacement = ground_displacement(ground_velocity, fps, periodic)
    intervals, anchors, root_offsets, drift = [], [], [], []
    residual = np.zeros((frame_count, len(contacts), 3))
    plant_id = np.full((frame_count, len(contacts)), -1, dtype=np.int32)
    for k, joint in enumerate(contacts):
        for start, end in contact_intervals(mask[:, k], periodic=periodic):
            frames = np.arange(start, end)
            wrapped = frames % frame_count
            ground = positions[wrapped, joint].copy()
            ground[:, [0, 2]] += displacement[frames]
            anchor = ground.mean(axis=0)
            mid = int(wrapped[len(frames) // 2])
            index = len(intervals)
            intervals.append((k, start, end))
            anchors.append(anchor)
            root_offsets.append(positions[mid, joint] - positions[mid, root])
            residual[wrapped, k] = ground - anchor
            plant_id[wrapped, k] = index
            drift.append(float(np.linalg.norm((ground - anchor)[:, [0, 2]], axis=-1).max()))
    return {
        "plant_intervals": np.asarray(intervals, dtype=np.int32).reshape(-1, 3),
        "plant_anchor": np.asarray(anchors, dtype=np.float64).reshape(-1, 3),
        "plant_root_offset": np.asarray(root_offsets, dtype=np.float64).reshape(-1, 3),
        "plant_residual": residual,
        "plant_id": plant_id,
        "plant_drift": np.asarray(drift, dtype=np.float64),
    }


def airborne_spans(mask: np.ndarray, min_frames: int = MIN_AIRBORNE_FRAMES) -> list[tuple[int, int]]:
    """``(takeoff, landing)`` frames of each span with no planted contact between two plants."""
    planted = mask.any(axis=1)
    spans = []
    for start, end in contact_intervals(~planted):
        if start > 0 and end < len(planted) and end - start >= min_frames:
            spans.append((start, end))
    return spans


def _optional_index(value) -> Optional[int]:
    return None if value is None else int(value)


def _circular_delta(a: float, b: float) -> float:
    d = (a - b) % 1.0
    return d - 1.0 if d > 0.5 else d


# ── decomposition ────────────────────────────────────────────────────────────

@dataclass
class Decoded:
    """The decoded, rigidified clip every layer is cut from."""

    base_rot: np.ndarray            # (F, J, 4)
    base_pos: np.ndarray            # (F, J, 3) local translations
    orients: np.ndarray
    anim_offsets: np.ndarray
    parents: np.ndarray
    skeleton_offsets: np.ndarray
    skeleton_rest_rotations: np.ndarray
    root: int
    fps: float
    ik_error: Optional[tuple[float, float]]

    @classmethod
    def from_package(cls, package: EditPackage) -> "Decoded":
        a = package.arrays
        ik = package.manifest["diagnostics"].get("ik_error")
        return cls(a["base_rot"], a["base_pos"], a["orients"], a["anim_offsets"], a["parents"],
                   a["skeleton_offsets"], a["skeleton_rest_rotations"], package.root, package.fps,
                   tuple(ik) if ik is not None else None)


@dataclass
class ClipInfo:
    object_type: str
    clip_name: str
    is_loop: bool
    action_group: str = ""
    action_label: str = ""
    stretch_factor: float = DEFAULT_STRETCH_FACTOR
    dataset_root: Optional[str] = None
    fullbody_ik: bool = True
    # diagnostics about the package's inputs (a stale species override); a rebuild keeps them
    notes: list = field(default_factory=list)


def decode_features(features: np.ndarray, cond_entry: dict, object_type: str, fps: float,
                    stretch_factor: float, fullbody_ik: bool = True) -> Decoded:
    """``fullbody_ik`` off keeps the decode's per-frame local translations, so bones
    stretch and swing freely and ``stretch_factor`` has no effect."""
    from utils.npy_restore import build_skeleton_only_context, restore_animation_from_features

    ctx = build_skeleton_only_context(cond_entry, object_type=object_type,
                                      feature_joint_count=features.shape[1])
    restored = restore_animation_from_features(
        features, ctx, restore_space="hml", fullbody_ik=bool(fullbody_ik),
        stretch_factor=float(stretch_factor), fps=fps)
    anim = restored.animation
    return Decoded(
        base_rot=np.asarray(anim.rotations.qs), base_pos=np.asarray(anim.positions),
        orients=np.asarray(anim.orients.qs), anim_offsets=np.asarray(anim.offsets),
        parents=np.asarray(anim.parents, dtype=np.int32),
        skeleton_offsets=np.asarray(ctx.export_offsets, dtype=np.float32),
        skeleton_rest_rotations=np.asarray(ctx.export_rest_rotations, dtype=np.float32),
        root=int(restored.translation_root_index), fps=float(restored.fps),
        ik_error=restored.ik_error,
    )


def decompose_motion(
    features: np.ndarray,
    cond_entry: dict,
    *,
    object_type: str,
    clip_name: str,
    is_loop: bool,
    action_group: str = "",
    action_label: str = "",
    fps: float = FPS,
    stretch_factor: float = DEFAULT_STRETCH_FACTOR,
    fullbody_ik: bool = True,
    profile: Optional[ProfileSubset] = None,
    contacts: Optional[ContactSource] = None,
    dataset_root: Optional[str] = None,
    notes: Optional[list] = None,
    contact_params: ContactParams = ContactParams(),
) -> EditPackage:
    """``notes``: diagnostic items about the inputs, kept through later rebuilds."""
    features = np.asarray(features)
    decoded = decode_features(features, cond_entry, object_type, fps, stretch_factor, fullbody_ik)
    info = ClipInfo(object_type, clip_name, bool(is_loop), action_group, action_label,
                    float(stretch_factor), dataset_root, bool(fullbody_ik), list(notes or []))
    return assemble_package(decoded, features, cond_entry, info,
                            profile or ProfileSubset("missing"),
                            contacts or contact_source(cond_entry), contact_params=contact_params)


def assemble_package(
    decoded: Decoded,
    features: np.ndarray,
    cond_entry: dict,
    info: ClipInfo,
    profile: ProfileSubset,
    contacts: ContactSource,
    *,
    contact_mask: Optional[np.ndarray] = None,
    contact_params: ContactParams = ContactParams(),
) -> EditPackage:
    """Steps 2-4 of section 4.1 on a decoded clip.

    ``contact_mask`` (F, K) replaces the detected contact intervals (a hand
    edit); the ground velocity is then re-estimated from it.
    """
    is_loop = info.is_loop
    used = contacts.used
    base_rot = decoded.base_rot
    rotations = np.asarray(base_rot, dtype=np.float64)
    global_rot, positions = forward_kinematics(decoded.parents, rotations,
                                               np.asarray(decoded.base_pos, dtype=np.float64))
    frame_count, joint_count = base_rot.shape[:2]
    root = decoded.root
    fps = decoded.fps

    structure = SkeletonStructure(cond_entry, used)
    leg = profile.leg_length or structure.leg_length
    diagnostics: list[dict] = [dict(item) for item in info.notes]
    if profile.status == "missing":
        diagnostics.append({"kind": "profile", "message": "no skeleton profile; roles from cond only"})
    for name in contacts.unknown_names:
        diagnostics.append({"kind": "contacts", "message": f"contact override names unknown joint '{name}'"})
    if not leg or leg <= 0:
        diagnostics.append({"kind": "contacts", "message": "no leg or axial length; contact detection skipped"})

    # 2. contacts
    if contact_mask is None:
        detected = detect_contacts(positions, used, leg or 0.0, fps, periodic=is_loop, params=contact_params)
    else:
        mask = np.asarray(contact_mask, dtype=bool).reshape(frame_count, len(used))
        detected = ContactResult(
            mask, ground_velocity_from_mask(positions, used, mask, fps, periodic=is_loop, params=contact_params),
            float("nan"))
    mask = detected.mask
    ground_velocity = detected.ground_velocity

    # 3. events
    clip = Clip(name=info.clip_name, action_group=info.action_group, action_label=info.action_label,
                is_loop=is_loop, fps=fps, local_rotations=rotations,
                global_rotations=global_rot, global_positions=positions)
    events: dict = {"period": None, "touchdowns": {}, "airborne": []}
    limbs = structure.contact_limbs()
    if is_loop and used and leg:
        gait = gait_stats.clip_gait(clip, used, limbs, leg, detected)
        masks = gait_stats.limb_masks(detected, used, limbs, positions, True)
        events["touchdowns"] = {str(foot): sorted(s for s, _ in contact_intervals(m, periodic=True))
                                for foot, m in masks.items()}
        if gait is not None:
            events["period"] = gait.period
            events["duty"] = gait.duty
            events["phase"] = gait.phase
            if profile.gait:
                ref_period = float(profile.gait.get("period") or 0.0)
                phase_dev = {foot: round(_circular_delta(p, float(profile.gait["phase"][foot])), 4)
                             for foot, p in gait.phase.items() if foot in profile.gait.get("phase", {})}
                events["profile_deviation"] = {
                    "period_ratio": round(gait.period / ref_period, 4) if ref_period else None,
                    "phase": phase_dev,
                }
                worst = max((abs(v) for v in phase_dev.values()), default=0.0)
                if worst > 0.15:
                    diagnostics.append({"kind": "gait", "message":
                                        f"touchdown phase differs from the profile's '{info.action_label}' "
                                        f"gait by up to {worst:.2f} cycle"})
    vertical = gait_stats.clip_vertical(clip, root, leg) if leg else None
    if used:
        events["airborne"] = [list(span) for span in airborne_spans(mask)]
    if vertical is not None:
        events["vertical"] = {"net": round(vertical["net"], 4), "peak": round(vertical["peak"], 4)}

    # 4. layers
    window = round(events["period"]) if (is_loop and events["period"]) else (
        frame_count if is_loop else round(ONE_SHOT_TREND_SECONDS * fps))
    root_trend, root_osc, root_yaw, root_tilt = split_root(
        np.asarray(decoded.base_pos[:, root], dtype=np.float64), rotations[:, root],
        window=window, periodic=is_loop)
    reference, chain_offsets, near_pi = chain_layer(rotations, root, periodic=is_loop)
    plants = plant_layer(positions, used, mask, ground_velocity, fps, root, is_loop)

    roles = (list(profile.arrays["profile_role"]) if profile.arrays is not None
             else structure.base_roles())
    canonical = [str(n) for n in cond_entry.get("canonical_joint_names", structure.names)]
    groups = chain_groups(structure, roles, canonical)
    passive = (profile.arrays["profile_passive"] if profile.arrays is not None
               else np.zeros(joint_count, dtype=bool))

    near_pi_frames = {structure.names[j]: int(near_pi[:, j].sum())
                      for j in np.flatnonzero(near_pi.any(axis=0))}
    for name, count in near_pi_frames.items():
        frame = int(np.flatnonzero(near_pi[:, structure.names.index(name)])[0])
        diagnostics.append({"kind": "near_pi", "joint": name, "frame": frame,
                            "message": f"{name}: chain offset near pi on {count} frame(s); gain held at 1"})
    if decoded.ik_error is not None:
        diagnostics.append({"kind": "ik", "message":
                            f"full-body IK residual mean {decoded.ik_error[0]:.4g}, "
                            f"max {decoded.ik_error[1]:.4g}"})

    yaw_range = float(np.degrees(root_yaw.max() - root_yaw.min()))
    facts = {
        "is_loop": is_loop,
        "locomotion": info.action_group == "locomotion",
        "has_plants": len(plants["plant_intervals"]) > 0,
        "turning": yaw_range > TURNING_YAW_DEG,
        "airborne": bool(events["airborne"]),
        "has_passive": bool(passive.any()),
        "chain_groups": sorted(set(groups.tolist())),
    }
    available = available_params(facts)
    if facts["locomotion"] and facts["has_plants"] and facts["turning"]:
        diagnostics.append({"kind": "turning", "message":
                            f"root yaw spans {yaw_range:.0f} deg; stride assumes a straight walk "
                            "and is not offered"})

    canonical_bvh = [str(n) for n in cond_entry.get("canonical_bvh_joint_names", structure.names)]
    arrays = {
        "parents": np.asarray(decoded.parents, dtype=np.int32),
        "names": np.asarray(structure.names),
        "bvh_names": np.asarray(canonical_bvh),
        "sides": np.asarray(structure.sides),
        "orients": decoded.orients,
        "anim_offsets": decoded.anim_offsets,
        "skeleton_offsets": decoded.skeleton_offsets,
        "skeleton_rest_rotations": decoded.skeleton_rest_rotations,
        "base_rot": base_rot,
        "base_pos": decoded.base_pos,
        "root_trend": root_trend,
        "root_osc": root_osc,
        "root_yaw": root_yaw,
        "root_tilt": root_tilt,
        "chain_reference": reference,
        "chain_offsets": chain_offsets,
        "chain_near_pi": near_pi,
        "chain_group": groups,
        "contact_joints": np.asarray(used, dtype=np.int32).reshape(-1),
        "contact_mask": mask,
        "ground_velocity": ground_velocity,
        **plants,
        "source_features": features,
        "source_cond": encode_json(cond_subset(cond_entry)),
    }
    if profile.arrays is not None:
        arrays.update(profile.arrays)
    else:
        arrays["profile_role"] = np.asarray(roles, dtype="<U8")

    drift = plants["plant_drift"]
    manifest = {
        "runtime_version": RUNTIME_VERSION,
        "profile_schema": PROFILE_SCHEMA,
        "skeleton_hash": skeleton_hash(cond_entry["parents"], cond_entry["offsets"]),
        "object_type": info.object_type,
        "clip": info.clip_name,
        "dataset_root": info.dataset_root,
        "fps": fps,
        "frame_count": int(frame_count),
        "joint_count": int(joint_count),
        "is_loop": bool(is_loop),
        "action_group": info.action_group,
        "action_label": info.action_label,
        "translation_root_index": root,
        "forward_joint_index": _optional_index(cond_entry.get("forward_joint_index")),
        "forward_base_joint_index": _optional_index(cond_entry.get("forward_base_joint_index")),
        "stretch_factor": float(info.stretch_factor),
        "fullbody_ik": bool(info.fullbody_ik),
        "input_notes": info.notes,
        "leg_length": float(leg or 0.0),
        "profile": {"status": profile.status, "leg_length": profile.leg_length, "gait": profile.gait,
                    "stride_speed_fit": profile.stride_speed_fit},
        "contacts": {
            "joints": used,
            "names": [structure.names[j] for j in used],
            "source": contacts.as_dict(),
            "intervals_edited": contact_mask is not None,
        },
        "events": events,
        "facts": facts,
        "params": param_manifest(available),
        "available_params": available,
        "diagnostics": {
            "ik_error": list(decoded.ik_error) if decoded.ik_error is not None else None,
            "plants": int(len(drift)),
            "slip_baseline_max": float(drift.max()) if len(drift) else 0.0,
            "slip_baseline_mean": float(drift.mean()) if len(drift) else 0.0,
            "near_pi_frames": near_pi_frames,
            "items": diagnostics,
        },
    }
    return EditPackage(manifest, arrays)


# ── edits of a package's own inputs ──────────────────────────────────────────

def _stored_inputs(package: EditPackage):
    m = package.manifest
    arrays = package.arrays
    stored = m.get("profile") or {}
    has_profile = stored.get("status") in ("ok", "fallback")
    profile = ProfileSubset(
        status=stored.get("status", "missing"),
        leg_length=stored.get("leg_length"),
        gait=stored.get("gait"),
        stride_speed_fit=stored.get("stride_speed_fit"),
        arrays={k: arrays[k] for k in arrays if k.startswith("profile_")} if has_profile else None,
    )
    info = ClipInfo(m["object_type"], m["clip"], bool(m["is_loop"]), m.get("action_group", ""),
                    m.get("action_label", ""), float(m["stretch_factor"]), m.get("dataset_root"),
                    bool(m.get("fullbody_ik", True)), list(m.get("input_notes", [])))
    return decode_json(arrays["source_cond"]), profile, info


def redecompose(package: EditPackage, stretch_factor: Optional[float] = None, *,
                fullbody_ik: Optional[bool] = None) -> EditPackage:
    """The same clip decomposed again from its stored features with another
    ``stretch_factor`` and / or ``fullbody_ik`` (``None`` keeps the package's).

    The contact set and the profile subset are carried over; contact intervals
    are detected afresh (manual interval edits do not survive).
    """
    cond, profile, info = _stored_inputs(package)
    if stretch_factor is not None:
        info.stretch_factor = float(stretch_factor)
    if fullbody_ik is not None:
        info.fullbody_ik = bool(fullbody_ik)
    features = package["source_features"]
    decoded = decode_features(features, cond, info.object_type, package.fps, info.stretch_factor,
                              info.fullbody_ik)
    return assemble_package(decoded, features, cond, info, profile,
                            ContactSource.from_dict(package.manifest["contacts"]["source"]))


def with_contact_joints(package: EditPackage, joints, *, species_add=None,
                        species_remove=None) -> EditPackage:
    """The package with another contact joint set (intervals detected afresh).

    The set is recorded against cond and the species override as this
    package's own additions and removals; ``species_add`` / ``species_remove``
    first replace the species override (after it was written to
    ``contact_overrides.json``).
    """
    cond, profile, info = _stored_inputs(package)
    source = ContactSource.from_dict(package.manifest["contacts"]["source"])
    if species_add is not None:
        source.species_add = sorted(int(j) for j in species_add)
    if species_remove is not None:
        source.species_remove = sorted(int(j) for j in species_remove)
    if species_add is not None or species_remove is not None:
        # the override was just rewritten against this skeleton: input notes about it are void
        info.notes = [n for n in info.notes if n.get("kind") != "contacts"]
    base = (set(source.cond) | set(source.species_add)) - set(source.species_remove)
    wanted = {int(j) for j in joints}
    source.package_add = sorted(wanted - base)
    source.package_remove = sorted(base - wanted)
    return assemble_package(Decoded.from_package(package), package["source_features"], cond, info,
                            profile, source)


def with_contact_mask(package: EditPackage, mask) -> EditPackage:
    """The package with hand-edited contact intervals (``mask`` (F, K) over its contact joints)."""
    cond, profile, info = _stored_inputs(package)
    mask = np.asarray(mask, dtype=bool)
    expected = (package.frame_count, len(package["contact_joints"]))
    if mask.shape != expected:
        raise ValueError(f"contact mask has shape {mask.shape}, expected {expected}")
    return assemble_package(Decoded.from_package(package), package["source_features"], cond, info,
                            profile, ContactSource.from_dict(package.manifest["contacts"]["source"]),
                            contact_mask=mask)
