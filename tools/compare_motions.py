"""
compare_motions.py — Cross-format motion comparison tool.

Loads two motion files (BVH / GLB / FBX) via bpy, auto-detects
and aligns spatial differences (translation, coordinate system, scale, time
offset), then reports per-bone world-space position and rotation errors.

All formats go through bpy importers for a consistent world-space pipeline:
  - .bvh  → bpy.ops.import_anim.bvh   (Blender's Z-up → Y-up conversion)
  - .glb  → bpy.ops.import_scene.gltf  (Y-up, right-handed)
  - .fbx  → bpy.ops.import_scene.fbx   (Y-up, right-handed)
  - .gltf → bpy.ops.import_scene.gltf

Usage (requires bpy installed in your Python environment):
    python Anytop/tools/compare_motions.py --motion_a path/to/a.bvh --motion_b path/to/b.glb

    With optional JSON report:
    python Anytop/tools/compare_motions.py --motion_a path/to/a.bvh --motion_b path/to/b.glb --json-out report.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from Anytop.utils.rotation_numpy import apply_rotation_to_quaternions_wxyz_np


# ── Path setup ────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANYTOP_ROOT = os.path.realpath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.dirname(ANYTOP_ROOT)

for _p in [REPO_ROOT, ANYTOP_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class MotionData:
    """Unified container for a loaded motion."""
    file_path: str
    file_format: str  # "bvh" | "glb" | "fbx"
    bone_names: list[str]
    parents: np.ndarray           # (J,) int32, -1 = root
    offsets: np.ndarray           # (J, 3) rest-pose offsets
    world_positions: np.ndarray   # (F, J, 3) world-space bone-head positions
    world_rotations: np.ndarray   # (F, J, 4) world-space quaternions (w, x, y, z)
    sample_times: list[float]     # (F,) sample times in seconds, relative to clip start
    fps: float
    num_frames: int
    num_joints: int


@dataclass
class AlignmentResult:
    """Detected alignment corrections applied to motion B."""
    time_offset: float           # B.sample_times[0] - A.sample_times[0] in seconds
    translation_offset: np.ndarray  # (3,) mean root pos of A - mean root pos of B
    rotation_matrix: np.ndarray     # (3, 3) applied to B positions & rotations
    rotation_label: str             # human-readable name of the rotation
    scale: float                    # mean_bone_len_A / mean_bone_len_B
    applied: bool                   # True if any non-identity correction was applied


# ── Blender helpers (inlined, no external dependency beyond bpy) ──────────────

def _clear_scene() -> None:
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _remove_lights_and_cameras() -> None:
    import bpy
    for obj in list(bpy.data.objects):
        if obj.type in {"LIGHT", "CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)


def _iter_action_fcurves(action):
    """Iterate fcurves from an action (handles layered/stripped actions)."""
    if action is None:
        return []
    if hasattr(action, "fcurves"):
        return list(action.fcurves)
    all_fcurves = []
    if hasattr(action, "layers"):
        for layer in action.layers:
            for strip in layer.strips:
                if hasattr(strip, "channelbags"):
                    for channelbag in strip.channelbags:
                        all_fcurves.extend(channelbag.fcurves)
    return all_fcurves


def _set_scene_time(scene, sample_time: float) -> None:
    frame = math.floor(sample_time)
    subframe = float(sample_time - frame)
    scene.frame_set(frame, subframe=subframe)


def _get_action_sample_frames(armature) -> list[float]:
    """Extract sorted unique keyframe frame values from an armature's action."""
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        return [0.0]
    key_times = sorted({
        round(float(keyframe.co[0]), 6)
        for fcurve in _iter_action_fcurves(action)
        for keyframe in fcurve.keyframe_points
    })
    return key_times or [0.0]


def _infer_sample_fps(scene, sample_times: list[float]) -> float:
    """Infer effective FPS from scene FPS and median sample delta."""
    scene_fps = scene.render.fps / scene.render.fps_base
    if len(sample_times) < 2:
        return float(scene_fps)
    deltas = np.diff(np.asarray(sample_times, dtype=np.float64))
    positive_deltas = deltas[deltas > 1e-6]
    if positive_deltas.size == 0:
        return float(scene_fps)
    return float(scene_fps / np.median(positive_deltas))


def _sample_frames_to_relative_seconds(sample_frames: list[float], fps: float) -> list[float]:
    """Convert Blender frame values to relative seconds from clip start."""
    if not sample_frames:
        return [0.0]
    if fps <= 1e-8:
        return [0.0 for _ in sample_frames]
    first_frame = float(sample_frames[0])
    return [float(frame - first_frame) / float(fps) for frame in sample_frames]


def _read_bvh_frame_rate(file_path: str) -> float:
    """Read the FPS from a BVH file's 'Frame Time' field.

    Falls back to 30.0 if the field is not found or invalid.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                stripped = line.strip()
                if stripped.lower().startswith("frame time"):
                    # Format: "Frame Time: 0.0333333"
                    parts = stripped.split(":")
                    if len(parts) >= 2:
                        frame_time = float(parts[-1].strip())
                        if frame_time > 0:
                            return 1.0 / frame_time
    except Exception:
        pass
    return 30.0


def _extract_armature_skeleton_data(armature):
    """Extract bone names, parents, offsets, rest rotations via BFS traversal.

    Returns
    -------
    bone_names : list[str]
        BFS-ordered bone names.
    parents : np.ndarray[int32, (J,)]
        Parent index per bone (-1 = root).
    offsets : np.ndarray[float64, (J, 3)]
        Rest-pose local offsets.
    rest_rotations : np.ndarray[float64, (J, 4)]
        Rest-pose local quaternions (w, x, y, z).
    """
    armature_bones = armature.data.bones
    all_roots = [bone for bone in armature_bones if bone.parent is None]
    if not all_roots:
        raise RuntimeError("No root bone found in armature")

    # Pick the root with the largest subtree
    def _subtree_size(root_bone) -> int:
        count = 0
        queue = deque([root_bone])
        while queue:
            bone = queue.popleft()
            count += 1
            queue.extend(bone.children)
        return count

    root_bone = max(all_roots, key=_subtree_size)

    ordered_bones = []
    queue = deque([root_bone])
    while queue:
        bone = queue.popleft()
        ordered_bones.append(bone)
        queue.extend(bone.children)

    joint_count = len(ordered_bones)
    bone_names = [bone.name for bone in ordered_bones]
    parents = np.full(joint_count, -1, dtype=np.int32)
    offsets = np.zeros((joint_count, 3), dtype=np.float64)
    rest_rotations = np.zeros((joint_count, 4), dtype=np.float64)
    name_to_idx = {name: idx for idx, name in enumerate(bone_names)}

    for joint_idx, bone in enumerate(ordered_bones):
        if bone.parent is not None and bone.parent.name in name_to_idx:
            parent_idx = name_to_idx[bone.parent.name]
            parents[joint_idx] = parent_idx
            rest_local = bone.parent.matrix_local.inverted_safe() @ bone.matrix_local
        else:
            rest_local = bone.matrix_local.copy()

        rest_translation = rest_local.translation
        rest_quat = rest_local.to_quaternion()
        offsets[joint_idx] = (rest_translation.x, rest_translation.y, rest_translation.z)
        rest_rotations[joint_idx] = (rest_quat.w, rest_quat.x, rest_quat.y, rest_quat.z)

    return bone_names, parents, offsets, rest_rotations


def _collect_armature_world_pose_data(armature, sample_frames: list[float],
                                       bone_names: list[str]) -> dict[str, Any]:
    """Collect bone-head world positions and world quaternions at each sample time.

    Uses the provided bone_names order (from _extract_armature_skeleton_data)
    to ensure consistent indexing between skeleton metadata and pose data.

    Returns dict with keys:
        bone_names, sample_frames, head_positions (F, J, 3), world_rotations (F, J, 4)
    """
    import bpy
    from mathutils import Vector

    scene = bpy.context.scene
    num_frames = len(sample_frames)
    num_joints = len(bone_names)
    head_positions = np.zeros((num_frames, num_joints, 3), dtype=np.float64)
    world_rotations = np.zeros((num_frames, num_joints, 4), dtype=np.float64)

    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="OBJECT")

    for frame_idx, sample_frame in enumerate(sample_frames):
        _set_scene_time(scene, float(sample_frame))
        bpy.context.view_layer.update()
        for bone_idx, bone_name in enumerate(bone_names):
            pose_bone = armature.pose.bones.get(bone_name)
            if pose_bone is None:
                continue
            head_local = pose_bone.head
            head_world = armature.matrix_world @ Vector((head_local.x, head_local.y, head_local.z))
            world_quat = (armature.matrix_world @ pose_bone.matrix).to_quaternion()
            head_positions[frame_idx, bone_idx] = (head_world.x, head_world.y, head_world.z)
            world_rotations[frame_idx, bone_idx] = (world_quat.w, world_quat.x, world_quat.y, world_quat.z)

    return {
        "bone_names": bone_names,
        "sample_frames": np.asarray(sample_frames, dtype=np.float64),
        "head_positions": head_positions,
        "world_rotations": world_rotations,
    }


# ── Import dispatcher ─────────────────────────────────────────────────────────

def _load_motion(file_path: str) -> MotionData:
    """Load a motion file (BVH/GLB/FBX) and return MotionData.

    All formats are imported via Blender (bpy) for consistent world-space output.
    """
    import bpy

    ext = os.path.splitext(file_path)[1].lower()
    ext_map = {".bvh": "bvh", ".glb": "glb", ".gltf": "glb", ".fbx": "fbx"}

    file_format = ext_map.get(ext)
    if file_format is None:
        raise ValueError(f"Unsupported format: {ext} (supported: .bvh, .glb, .gltf, .fbx)")

    _clear_scene()

    if file_format == "bvh":
        # Suppress Blender's verbose importer output (e.g. "zero length node found")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            bpy.ops.import_anim.bvh(filepath=file_path)
    elif file_format == "fbx":
        from Anytop.utils.fbx import import_fbx
        import_fbx(file_path)
    elif file_format == "glb":
        bpy.ops.import_scene.gltf(filepath=file_path)

    _remove_lights_and_cameras()

    armature = next((obj for obj in bpy.data.objects if obj.type == "ARMATURE"), None)
    if armature is None:
        raise RuntimeError(f"No armature found in {file_path}")

    # Extract skeleton metadata
    bone_names, parents, offsets, _rest_rotations = _extract_armature_skeleton_data(armature)
    num_joints = len(bone_names)

    # Determine sample times and FPS
    sample_frames = _get_action_sample_frames(armature)
    scene = bpy.context.scene

    # For BVH, read FPS from the file header's "Frame Time" field directly,
    # because Blender's BVH importer places keyframes on integer frames and
    # _infer_sample_fps would incorrectly use the scene's default FPS (24).
    if file_format == "bvh":
        fps = _read_bvh_frame_rate(file_path)
    else:
        fps = _infer_sample_fps(scene, sample_frames)
    sample_times = _sample_frames_to_relative_seconds(sample_frames, fps)
    num_frames = len(sample_frames)

    # Collect world-space pose data (pass BFS-ordered bone names for consistency)
    pose_data = _collect_armature_world_pose_data(armature, sample_frames, bone_names)

    # Double-check bone name consistency
    assert pose_data["bone_names"] == bone_names, (
        f"Bone name mismatch between skeleton extraction and pose collection: "
        f"{bone_names} vs {pose_data['bone_names']}"
    )

    return MotionData(
        file_path=file_path,
        file_format=file_format,
        bone_names=bone_names,
        parents=parents,
        offsets=offsets,
        world_positions=pose_data["head_positions"],
        world_rotations=pose_data["world_rotations"],
        sample_times=sample_times,
        fps=fps,
        num_frames=num_frames,
        num_joints=num_joints,
    )


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_compatible(motion_a: MotionData, motion_b: MotionData) -> None:
    """Check FPS and frame count match exactly; abort on mismatch."""
    ok = True

    fps_tol = 0.001 * max(motion_a.fps, motion_b.fps)
    if abs(motion_a.fps - motion_b.fps) > fps_tol:
        print(f"[ERROR] FPS mismatch: A={motion_a.fps:.4f}, B={motion_b.fps:.4f}", file=sys.stderr)
        ok = False

    if motion_a.num_frames != motion_b.num_frames:
        print(f"[ERROR] Frame count mismatch: A={motion_a.num_frames}, B={motion_b.num_frames}", file=sys.stderr)
        ok = False

    if not ok:
        sys.exit(1)


# ── Alignment helpers ─────────────────────────────────────────────────────────

def _canonical_bone_name(name: str) -> str:
    """Normalize bone name for cross-format matching."""
    return name.replace(" ", "_").lower()


def _quaternion_angle_degrees(q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
    """Per-element quaternion angular difference in degrees.

    Returns angle in [0, 180] for each (..., 4) pair.
    Normalizes both quaternions before computing dot product to avoid
    errors from non-unit quaternions (e.g., due to floating-point drift).
    """
    # Normalize to handle non-unit quaternions
    norm_a = np.maximum(np.linalg.norm(q_a, axis=-1, keepdims=True), 1e-12)
    norm_b = np.maximum(np.linalg.norm(q_b, axis=-1, keepdims=True), 1e-12)
    q_a_n = q_a / norm_a
    q_b_n = q_b / norm_b

    dots = np.sum(q_a_n * q_b_n, axis=-1)
    dots = np.clip(np.abs(dots), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))


def _compute_mean_bone_length_from_rest(offsets: np.ndarray, parents: np.ndarray) -> float:
    """Compute mean bone length from rest-pose offsets (child bones only)."""
    child_mask = parents >= 0
    if not np.any(child_mask):
        return 0.0
    lengths = np.linalg.norm(offsets[child_mask], axis=-1)
    return float(lengths.mean())


def _compute_mean_bone_length_from_positions(
    world_positions: np.ndarray,
    parents: np.ndarray,
    canonical_to_idx: dict[str, int],
    ref_canonical_names: list[str],
    ref_parents: np.ndarray,
) -> float:
    """Compute mean bone length from world positions using reference parent relationships.

    This matches bones by canonical name and computes parent→child lengths
    using the reference parent structure. Useful for comparing motions with
    different bone hierarchies.
    """
    lengths = []
    for child_idx, parent_idx in enumerate(ref_parents):
        if parent_idx < 0:
            continue
        child_name = ref_canonical_names[child_idx]
        parent_name = ref_canonical_names[parent_idx]
        if child_name not in canonical_to_idx or parent_name not in canonical_to_idx:
            continue
        child_pos = world_positions[:, canonical_to_idx[child_name], :]
        parent_pos = world_positions[:, canonical_to_idx[parent_name], :]
        lengths.append(np.linalg.norm(child_pos - parent_pos, axis=-1))

    if not lengths:
        return 0.0
    return float(np.stack(lengths, axis=0).mean())


def _compute_common_bone_reindex(
    names_a: list[str], names_b: list[str],
) -> tuple[list[str], list[int], list[int]]:
    """Find common bones by canonical name and return reindex mappings.

    Returns
    -------
    common_names : list[str]
        Sorted canonical names common to both.
    idx_a : list[int]
        Indices into A's original arrays for each common bone.
    idx_b : list[int]
        Indices into B's original arrays for each common bone.
    """
    canon_a = {_canonical_bone_name(n): i for i, n in enumerate(names_a)}
    canon_b = {_canonical_bone_name(n): i for i, n in enumerate(names_b)}
    common = sorted(set(canon_a) & set(canon_b))
    idx_a = [canon_a[c] for c in common]
    idx_b = [canon_b[c] for c in common]
    return common, idx_a, idx_b


# ── Coordinate system candidates ──────────────────────────────────────────────

def _generate_coordinate_candidates() -> list[tuple[str, np.ndarray]]:
    """Generate candidate 3x3 rotation/reflection matrices for auto-detection.

    Returns list of (label, matrix) pairs.
    """
    I = np.eye(3, dtype=np.float64)

    def R_x(deg: float) -> np.ndarray:
        rad = np.deg2rad(deg)
        c, s = math.cos(rad), math.sin(rad)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)

    def R_y(deg: float) -> np.ndarray:
        rad = np.deg2rad(deg)
        c, s = math.cos(rad), math.sin(rad)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)

    def R_z(deg: float) -> np.ndarray:
        rad = np.deg2rad(deg)
        c, s = math.cos(rad), math.sin(rad)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

    candidates = [
        ("identity", I),
        ("R_x(+90°)", R_x(90)),
        ("R_x(-90°)", R_x(-90)),
        ("R_y(+90°)", R_y(90)),
        ("R_y(-90°)", R_y(-90)),
        ("R_z(+90°)", R_z(90)),
        ("R_z(-90°)", R_z(-90)),
        ("R_x(+180°)", R_x(180)),     # diag(1,-1,-1)
        ("R_z(+180°)", R_z(180)),     # diag(-1,-1,1)
        ("flip_X", np.diag([-1, 1, 1])),
        ("flip_Y", np.diag([1, -1, 1])),
        ("flip_Z", np.diag([1, 1, -1])),
    ]
    return candidates


# ── Alignment detection ───────────────────────────────────────────────────────

def _detect_and_align(
    motion_a: MotionData,
    motion_b: MotionData,
) -> tuple[MotionData, AlignmentResult]:
    """Auto-detect and apply time offset, translation, coordinate, scale alignment.

    Returns (motion_b_aligned, alignment).
    """
    # ── Step 1: Detect time offset ──────────────────────────────────────────
    time_offset = motion_b.sample_times[0] - motion_a.sample_times[0]

    # ── Step 2: Find common bones for alignment diagnostics ─────────────────
    common_names, idx_a, idx_b = _compute_common_bone_reindex(
        motion_a.bone_names, motion_b.bone_names,
    )
    if not common_names:
        raise RuntimeError("No common bones found between the two motions")

    pos_a = motion_a.world_positions[:, idx_a, :]  # (F, K, 3)
    pos_b = motion_b.world_positions[:, idx_b, :]  # (F, K, 3)
    rot_a = motion_a.world_rotations[:, idx_a, :]  # (F, K, 4)
    rot_b = motion_b.world_rotations[:, idx_b, :]  # (F, K, 4)

    # ── Step 3: Detect scale FIRST (before translation) ────────────────────
    # Use world-position bone lengths for only the parent→child pairs where
    # both bones are in the common set (shared by A and B). This ensures the
    # bone length comparison is valid regardless of differing hierarchies.
    ref_canonical_a = [_canonical_bone_name(motion_a.bone_names[i]) for i in idx_a]
    reindexed_bone_names_a = [motion_a.bone_names[i] for i in idx_a]
    reindexed_canonical_a = {_canonical_bone_name(n): i for i, n in enumerate(reindexed_bone_names_a)}

    # Map A's parent indices from its original index space to the common-bone
    # index space (0..K-1).  A parent that is not in the common set is set to
    # -1 so that _compute_mean_bone_length_from_positions skips that bone.
    orig_a_to_common = {orig_idx: ci for ci, orig_idx in enumerate(idx_a)}
    common_parents = np.array(
        [orig_a_to_common.get(p, -1) if p >= 0 else -1 for p in motion_a.parents],
        dtype=np.int32,
    )
    # Subset to only the common bones
    common_parents_subset = common_parents[idx_a]

    mean_len_a = _compute_mean_bone_length_from_positions(
        pos_a, common_parents_subset, reindexed_canonical_a, ref_canonical_a, common_parents_subset,
    )
    mean_len_b = _compute_mean_bone_length_from_positions(
        pos_b, common_parents_subset, reindexed_canonical_a, ref_canonical_a, common_parents_subset,
    )

    if mean_len_b > 1e-8 and mean_len_a > 1e-8:
        scale = mean_len_a / mean_len_b
    else:
        scale = 1.0

    # Skip scale alignment if difference is negligible (< 0.001)
    if abs(scale - 1.0) < 0.001:
        scale = 1.0

    # ── Step 4: Detect translation offset (after scaling B) ────────────────
    # Scale B first, then compute translation from root positions.
    # Use the first common root bone (parents == -1) for robust alignment.
    pos_b_scaled = pos_b * scale
    root_in_common = None
    for ci, orig_idx in enumerate(idx_a):
        if motion_a.parents[orig_idx] == -1:
            root_in_common = ci
            break
    if root_in_common is None:
        root_in_common = 0  # fallback to first common bone
    translation_offset = pos_a[0, root_in_common, :] - pos_b_scaled[0, root_in_common, :]  # (3,)

    # Skip translation alignment if difference is negligible (< 1e-5)
    if np.linalg.norm(translation_offset) < 1e-5:
        translation_offset = np.zeros(3, dtype=np.float64)

    # Apply scale + translation to B
    pos_b_aligned = pos_b_scaled + translation_offset[np.newaxis, np.newaxis, :]

    # ── Step 5: Detect coordinate system ────────────────────────────────────
    candidates = _generate_coordinate_candidates()

    best_label = "identity"
    best_R = np.eye(3, dtype=np.float64)
    best_error = float("inf")

    for label, R in candidates:
        pos_candidate = pos_b_aligned @ R.T
        err = float(np.mean(np.linalg.norm(pos_a - pos_candidate, axis=-1)))
        if err < best_error:
            best_error = err
            best_label = label
            best_R = R

    # ── Step 6: Build aligned B ─────────────────────────────────────────────
    is_identity_transform = bool(
        abs(time_offset) < 1e-6
        and np.allclose(best_R, np.eye(3), atol=1e-8)
        and abs(scale - 1.0) < 1e-8
        and np.allclose(translation_offset, 0.0, atol=1e-8)
    )

    if is_identity_transform:
        # Skip quat→matrix→quat round-trip to avoid floating-point drift
        aligned_positions = motion_b.world_positions.copy()
        aligned_rotations = motion_b.world_rotations.copy()
    else:
        # Scale is already baked into pos_b_aligned; apply rotation on top
        aligned_positions = (
            scale * motion_b.world_positions + translation_offset[np.newaxis, np.newaxis, :]
        ) @ best_R.T
        aligned_rotations = apply_rotation_to_quaternions_wxyz_np(motion_b.world_rotations, best_R)

    motion_b_aligned = MotionData(
        file_path=motion_b.file_path,
        file_format=motion_b.file_format,
        bone_names=motion_b.bone_names,
        parents=motion_b.parents.copy(),
        offsets=motion_b.offsets.copy(),
        world_positions=aligned_positions,
        world_rotations=aligned_rotations,
        sample_times=motion_b.sample_times,
        fps=motion_b.fps,
        num_frames=motion_b.num_frames,
        num_joints=motion_b.num_joints,
    )

    alignment = AlignmentResult(
        time_offset=time_offset,
        translation_offset=translation_offset,
        rotation_matrix=best_R,
        rotation_label=best_label,
        scale=scale,
        applied=not is_identity_transform,
    )

    return motion_b_aligned, alignment


# ── Comparison ────────────────────────────────────────────────────────────────

def _compare_motions(
    motion_a: MotionData,
    motion_b: MotionData,
    alignment: AlignmentResult,
) -> dict[str, Any]:
    """Compare two motions on common bones and produce a detailed report dict.

    Both motions must have the same frame count (validated earlier).
    """
    common_names, idx_a, idx_b = _compute_common_bone_reindex(
        motion_a.bone_names, motion_b.bone_names,
    )
    if not common_names:
        raise RuntimeError("No common bones found between the two motions")

    num_common = len(common_names)
    num_frames = motion_a.num_frames  # same as motion_b.num_frames

    # Extract aligned sub-arrays
    pos_a = motion_a.world_positions[:, idx_a, :]          # (F, K, 3)
    pos_b = motion_b.world_positions[:, idx_b, :]          # (F, K, 3)
    rot_a = motion_a.world_rotations[:, idx_a, :]          # (F, K, 4)
    rot_b = motion_b.world_rotations[:, idx_b, :]          # (F, K, 4)

    # ── Position errors ────────────────────────────────────────────────────
    pos_diffs = pos_a - pos_b                                # (F, K, 3)
    pos_errors = np.linalg.norm(pos_diffs, axis=-1)          # (F, K)

    # Per-bone max/mean
    per_bone_max = pos_errors.max(axis=0)                    # (K,)
    per_bone_mean = pos_errors.mean(axis=0)                  # (K,)
    per_frame_max = pos_errors.max(axis=1)                   # (F,)

    # Global stats
    mean_bone_len_a = _compute_mean_bone_length_from_rest(motion_a.offsets, motion_a.parents)
    if mean_bone_len_a <= 0.0:
        mean_bone_len_a = _compute_mean_bone_length_from_positions(
            motion_a.world_positions, motion_a.parents,
            {_canonical_bone_name(n): i for i, n in enumerate(motion_a.bone_names)},
            [_canonical_bone_name(n) for n in motion_a.bone_names],
            motion_a.parents,
        )

    pos_errors_all = pos_errors.ravel()
    pos_max = float(pos_errors.max())
    pos_mean = float(pos_errors.mean())
    pos_median = float(np.median(pos_errors))
    pos_std = float(pos_errors.std())

    # Percentage relative to A's mean bone length
    pct_normalizer = max(mean_bone_len_a, 1e-8)
    pos_max_pct = pos_max / pct_normalizer * 100.0
    pos_mean_pct = pos_mean / pct_normalizer * 100.0

    # Character overall size: bounding box diagonal of root-relative positions (first frame only)
    root_idx_a = int(np.where(motion_a.parents == -1)[0][0])
    root_in_common = None
    for ci, oi in enumerate(idx_a):
        if oi == root_idx_a:
            root_in_common = ci
            break
    pos_a_first = motion_a.world_positions[0, idx_a, :]  # (K, 3), frame 0
    if root_in_common is not None:
        root_pos = pos_a_first[root_in_common:root_in_common+1]  # (1, 3)
        pos_a_rel = pos_a_first - root_pos  # (K, 3), root-relative
    else:
        pos_a_rel = pos_a_first
    char_size = float(np.linalg.norm(pos_a_rel.max(axis=0) - pos_a_rel.min(axis=0)))
    pos_max_pct_char = pos_max / max(char_size, 1e-8) * 100.0
    pos_mean_pct_char = pos_mean / max(char_size, 1e-8) * 100.0

    # Worst bone / frame (position)
    worst_flat_idx = int(np.argmax(pos_errors))
    worst_frame_idx, worst_bone_idx = np.unravel_index(worst_flat_idx, pos_errors.shape)

    # Top 1 worst bone
    bone_worst_order = np.argsort(per_bone_max)[::-1]
    top1_bones_pos = [
        {"name": common_names[bi], "max_error": float(per_bone_max[bi]), "mean_error": float(per_bone_mean[bi])}
        for bi in bone_worst_order[:1]
    ]

    # ── Rotation errors (bone direction comparison) ────────────────────────
    # Compare bone direction vectors (parent→child) in world space.
    # This is format-independent and avoids quaternion representation issues
    # that arise from different Blender importer conventions (Euler vs quat).
    # Bones with very short parent-child distance (< min_bone_len * 1e-4)
    # are skipped as their direction is numerically unstable.
    num_frames = motion_a.num_frames

    # Determine minimum bone length for stable direction computation
    mean_len_for_dir = max(mean_bone_len_a, 1e-8)
    min_dir_len = mean_len_for_dir * 1e-4

    # Build: original index A → common bone index
    orig_a_to_common = {oi: ci for ci, oi in enumerate(idx_a)}
    # Build: canonical name → common bone index (for A's bones)
    common_canon_to_ci = {_canonical_bone_name(motion_a.bone_names[oi]): ci
                          for ci, oi in enumerate(idx_a)}

    dir_errors = np.full((num_frames, num_common), np.nan, dtype=np.float64)  # (F, K)

    for ci in range(num_common):
        orig_idx_a = idx_a[ci]
        parent_orig_a = motion_a.parents[orig_idx_a]
        if parent_orig_a < 0:
            continue  # root bone, skip

        # Is A's parent also in the common bone set?
        parent_canon = _canonical_bone_name(motion_a.bone_names[parent_orig_a])
        if parent_canon not in common_canon_to_ci:
            continue
        parent_ci = common_canon_to_ci[parent_canon]

        # Direction vectors in world space
        child_a = pos_a[:, ci, :]       # (F, 3)
        parent_a = pos_a[:, parent_ci, :]
        child_b = pos_b[:, ci, :]
        parent_b = pos_b[:, parent_ci, :]

        dir_a = child_a - parent_a
        dir_b = child_b - parent_b

        # Per-frame norms
        na = np.linalg.norm(dir_a, axis=-1)  # (F,)
        nb = np.linalg.norm(dir_b, axis=-1)

        # Skip frames where either direction is too short
        valid_frame = (na > min_dir_len) & (nb > min_dir_len)
        if not np.any(valid_frame):
            continue

        # Compute angle only on valid frames
        dir_a_n = dir_a / np.maximum(na, 1e-12)[:, None]
        dir_b_n = dir_b / np.maximum(nb, 1e-12)[:, None]
        dots = np.clip(np.sum(dir_a_n * dir_b_n, axis=-1), -1.0, 1.0)
        angles = np.degrees(np.arccos(dots))

        dir_errors[valid_frame, ci] = angles[valid_frame]

    # Only compute stats over valid (non-NaN) entries
    valid_mask = ~np.isnan(dir_errors)
    valid_errors = dir_errors[valid_mask] if np.any(valid_mask) else np.zeros((0,))

    rot_per_bone_max = np.full(num_common, np.nan, dtype=np.float64)
    rot_per_bone_mean = np.full(num_common, np.nan, dtype=np.float64)
    for ci in range(num_common):
        vals = dir_errors[:, ci]
        m = ~np.isnan(vals)
        if np.any(m):
            rot_per_bone_max[ci] = float(vals[m].max())
            rot_per_bone_mean[ci] = float(vals[m].mean())

    rot_per_frame_max = np.full(num_frames, np.nan, dtype=np.float64)
    for fi in range(num_frames):
        vals = dir_errors[fi, :]
        m = ~np.isnan(vals)
        if np.any(m):
            rot_per_frame_max[fi] = float(vals[m].max())

    rot_max = float(np.nanmax(dir_errors)) if np.any(valid_mask) else 0.0
    rot_mean = float(valid_errors.mean()) if valid_errors.size > 0 else 0.0
    rot_median = float(np.nanmedian(dir_errors)) if np.any(valid_mask) else 0.0
    rot_std = float(valid_errors.std()) if valid_errors.size > 0 else 0.0

    if np.any(valid_mask):
        rot_worst_flat = int(np.nanargmax(dir_errors))
        rot_worst_frame, rot_worst_bone = np.unravel_index(rot_worst_flat, dir_errors.shape)
    else:
        rot_worst_frame, rot_worst_bone = 0, 0

    bone_worst_order_rot = np.argsort(rot_per_bone_max)[::-1]
    # Filter out NaN from worst-bone ranking
    bone_worst_order_rot = [bi for bi in bone_worst_order_rot if not np.isnan(rot_per_bone_max[bi])]
    top1_bones_rot = [
        {"name": common_names[bi], "max_error_deg": float(rot_per_bone_max[bi]), "mean_error_deg": float(rot_per_bone_mean[bi])}
        for bi in bone_worst_order_rot[:1]
    ]

    return {
        "comparison": {
            "common_bones": num_common,
            "common_bone_names": common_names,
            "compared_frames": num_frames,
            "a_bone_count": motion_a.num_joints,
            "b_bone_count": motion_b.num_joints,
        },
        "position": {
            "max_error": pos_max,
            "mean_error": pos_mean,
            "median_error": pos_median,
            "std_error": pos_std,
            "max_error_pct": pos_max_pct,
            "mean_error_pct": pos_mean_pct,
            "max_error_pct_char": pos_max_pct_char,
            "mean_error_pct_char": pos_mean_pct_char,
            "normalizer_mean_bone_len": float(mean_bone_len_a),
            "character_size": char_size,
            "worst_bone": common_names[worst_bone_idx],
            "worst_frame": int(worst_frame_idx),
            "worst_value": float(pos_errors[worst_frame_idx, worst_bone_idx]),
            "per_bone": {
                "max": per_bone_max.tolist(),
                "mean": per_bone_mean.tolist(),
                "names": common_names,
            },
            "per_frame_max": per_frame_max.tolist(),
            "top1_worst_bones": top1_bones_pos,
        },
        "rotation": {
            "max_error_deg": rot_max,
            "mean_error_deg": rot_mean,
            "median_error_deg": rot_median,
            "std_error_deg": rot_std,
            "num_bones_with_parent": int(valid_mask.any(axis=0).sum()),
            "worst_bone": common_names[rot_worst_bone] if np.any(valid_mask) and rot_worst_bone < len(common_names) else "N/A",
            "worst_frame": int(rot_worst_frame),
            "worst_value_deg": float(dir_errors[rot_worst_frame, rot_worst_bone]) if np.any(valid_mask) else 0.0,
            "per_bone": {
                "max_deg": np.where(np.isnan(rot_per_bone_max), 0.0, rot_per_bone_max).tolist(),
                "mean_deg": np.where(np.isnan(rot_per_bone_mean), 0.0, rot_per_bone_mean).tolist(),
                "names": common_names,
            },
            "per_frame_max_deg": np.where(np.isnan(rot_per_frame_max), 0.0, rot_per_frame_max).tolist(),
            "top1_worst_bones": top1_bones_rot,
        },
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_summary(
    motion_a: MotionData,
    motion_b: MotionData,
    alignment: AlignmentResult,
    result: dict[str, Any],
) -> None:
    """Pretty-print comparison summary to console."""
    cmp = result["comparison"]
    pos = result["position"]
    rot = result["rotation"]

    def _fmt_path(p: str) -> str:
        if not os.path.isabs(p):
            return p
        try:
            return os.path.relpath(p, REPO_ROOT)
        except ValueError:
            return p

    print(f"[Motion A] {_fmt_path(motion_a.file_path)}")
    print(f"           format={motion_a.file_format.upper()}  frames={motion_a.num_frames}  "
          f"bones={motion_a.num_joints}  fps={motion_a.fps:.2f}")
    print(f"[Motion B] {_fmt_path(motion_b.file_path)}")
    print(f"           format={motion_b.file_format.upper()}  frames={motion_b.num_frames}  "
          f"bones={motion_b.num_joints}  fps={motion_b.fps:.2f}")
    print()

    # Alignment summary
    align_parts = []
    if abs(alignment.time_offset) > 1e-6:
        align_parts.append(f"time_offset={alignment.time_offset:+.6f}s")
    if np.any(np.abs(alignment.translation_offset) > 1e-8):
        tx, ty, tz = alignment.translation_offset
        align_parts.append(f"translation=({tx:.6f}, {ty:.6f}, {tz:.6f})")
    if alignment.rotation_label != "identity":
        R = alignment.rotation_matrix
        align_parts.append(f"coordinate={alignment.rotation_label}")
        # Also show matrix if non-trivial
        if not np.allclose(R, np.eye(3), atol=1e-6):
            align_parts.append(f"  R=[[{R[0,0]:+.4f} {R[0,1]:+.4f} {R[0,2]:+.4f}],"
                               f"[{R[1,0]:+.4f} {R[1,1]:+.4f} {R[1,2]:+.4f}],"
                               f"[{R[2,0]:+.4f} {R[2,1]:+.4f} {R[2,2]:+.4f}]]")
    if abs(alignment.scale - 1.0) > 1e-6:
        align_parts.append(f"scale={alignment.scale:.6f}")

    if alignment.applied:
        print(f"[Alignment] {' | '.join(align_parts)}")
    else:
        print("[Alignment] None needed (motions are already aligned)")
    print()

    print(f"[Comparison] common_bones={cmp['common_bones']}/{cmp['a_bone_count']},{cmp['b_bone_count']}  "
          f"frames={cmp['compared_frames']}")
    print()

    print(f"  Position error (world-space, char_size={pos['character_size']:.4f}):")
    print(f"    max  = {pos['max_error']:.6f}  ({pos['max_error_pct_char']:.4f}%)")
    print(f"    mean = {pos['mean_error']:.6f}  ({pos['mean_error_pct_char']:.4f}%)")
    print(f"    median = {pos['median_error']:.6f}  std = {pos['std_error']:.6f}")
    print(f"    worst_bone={pos['worst_bone']}  frame={pos['worst_frame']}  "
          f"value={pos['worst_value']:.6f}")
    if pos['max_error'] > 1e-8:
        print()
        print("    Top 1 bone (position):")
        for b in pos["top1_worst_bones"]:
            print(f"      {b['name']:<35} max={b['max_error']:.6f}  mean={b['mean_error']:.6f}")
        print()

    print(f"  Rotation error (bone direction angle, parent→child):")
    print(f"    max  = {rot['max_error_deg']:.6f} deg")
    print(f"    mean = {rot['mean_error_deg']:.6f} deg")
    print(f"    median = {rot['median_error_deg']:.6f} deg  std = {rot['std_error_deg']:.6f} deg")
    print(f"    worst_bone={rot['worst_bone']}  frame={rot['worst_frame']}  "
          f"value={rot['worst_value_deg']:.6f} deg")
    if rot['max_error_deg'] > 1e-8:
        print()
        print("    Top 1 bone (rotation):")
        for b in rot["top1_worst_bones"]:
            print(f"      {b['name']:<35} max={b['max_error_deg']:.6f} deg  mean={b['mean_error_deg']:.6f} deg")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two motion files (BVH/GLB/FBX) and report per-bone differences.",
    )
    parser.add_argument("--motion_a", required=True, help="Path to first motion file")
    parser.add_argument("--motion_b", required=True, help="Path to second motion file")
    parser.add_argument("--json-out", default=None, help="Optional path to write JSON report")
    args = parser.parse_args()

    print(f"[Info] Loading {args.motion_a} ...")
    motion_a = _load_motion(args.motion_a)
    print(f"[Info] Loading {args.motion_b} ...")
    motion_b = _load_motion(args.motion_b)
    print()

    print("[Validate] Checking compatibility ...")
    _validate_compatible(motion_a, motion_b)
    print("[Validate] OK — FPS and frame count match")
    print()

    print("[Align] Detecting and applying spatial alignment ...")
    motion_b_aligned, alignment = _detect_and_align(motion_a, motion_b)
    print("[Align] Done")
    print()

    print("[Compare] Computing per-bone position and rotation errors ...")
    result = _compare_motions(motion_a, motion_b_aligned, alignment)
    print("[Compare] Done")
    print()

    _print_summary(motion_a, motion_b, alignment, result)
    print()

    if args.json_out:
        report = {
            "motion_a": {
                "path": os.path.abspath(motion_a.file_path),
                "format": motion_a.file_format,
                "frames": motion_a.num_frames,
                "bones": motion_a.num_joints,
                "fps": motion_a.fps,
            },
            "motion_b": {
                "path": os.path.abspath(motion_b.file_path),
                "format": motion_b.file_format,
                "frames": motion_b.num_frames,
                "bones": motion_b.num_joints,
                "fps": motion_b.fps,
            },
            "alignment": {
                "time_offset_s": float(alignment.time_offset),
                "translation_offset": alignment.translation_offset.tolist(),
                "rotation_matrix": alignment.rotation_matrix.tolist(),
                "rotation_label": alignment.rotation_label,
                "scale": float(alignment.scale),
                "applied": alignment.applied,
            },
            "result": result,
        }

        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[Report] JSON saved -> {args.json_out}")


if __name__ == "__main__":
    main()
