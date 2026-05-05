"""
Shared helpers for FBX/GLB/NPY/BVH roundtrip tests.

Extracted from test_fbx_glb_npy_roundtrip.py to avoid duplication
between NPY and BVH roundtrip tests.
"""
from __future__ import annotations

import math
import os
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch


# ── Skeleton helpers for exporter ────────────────────────────────────────────

@dataclass
class _SimpleBone:
    id: int
    name: str
    parent_id: Optional[int]
    rest_offset: torch.Tensor
    rest_rotation: torch.Tensor


class _SimpleSkeleton:
    """Minimal skeleton that matches the API expected by AnimationExporter."""

    def __init__(self, bones: list[_SimpleBone]):
        self.bones = bones
        self.rest_offsets = torch.stack([bone.rest_offset for bone in bones], dim=0)
        self._build_depth_levels()

    @property
    def num_joints(self) -> int:
        return len(self.bones)

    def _build_depth_levels(self):
        joint_count = len(self.bones)
        parents = torch.tensor(
            [bone.parent_id if bone.parent_id is not None else -1 for bone in self.bones],
            dtype=torch.long,
        )
        depths = torch.zeros(joint_count, dtype=torch.long)
        for joint_idx in range(1, joint_count):
            parent_idx = parents[joint_idx].item()
            if parent_idx >= 0:
                depths[joint_idx] = depths[parent_idx] + 1

        max_depth = depths.max().item()
        device = self.rest_offsets.device
        self.depth_levels: list[tuple] = []
        self.root_bone_ids: list[int] = []

        for depth in range(max_depth + 1):
            ids = [bone.id for bone in self.bones if depths[bone.id].item() == depth]
            if depth == 0:
                self.root_bone_ids = ids
                parent_ids = [-1] * len(ids)
            else:
                parent_ids = [self.bones[bone_id].parent_id for bone_id in ids]
            self.depth_levels.append((
                torch.tensor(ids, dtype=torch.long, device=device),
                torch.tensor(parent_ids, dtype=torch.long, device=device),
            ))


def _build_skeleton(
    bone_names, offsets, parents, rest_rotations=None, device=None,
) -> _SimpleSkeleton:
    if device is None:
        device = torch.device("cpu")

    bones = []
    for joint_idx, bone_name in enumerate(bone_names):
        parent_idx = parents[joint_idx]
        if rest_rotations is None:
            rest_rotation = [1.0, 0.0, 0.0, 0.0]
        else:
            rest_rotation = rest_rotations[joint_idx]
        bones.append(_SimpleBone(
            id=joint_idx,
            name=bone_name,
            parent_id=None if parent_idx < 0 else int(parent_idx),
            rest_offset=torch.tensor(offsets[joint_idx], dtype=torch.float32, device=device),
            rest_rotation=torch.tensor(rest_rotation, dtype=torch.float32, device=device),
        ))
    return _SimpleSkeleton(bones)


# ── FBX loading helpers ──────────────────────────────────────────────────────

def _load_fbx_scene(fbx_path: str):
    """Import an FBX file into a fresh Blender scene and return the armature."""
    import bpy
    from Anytop.utils.fbx import clear_scene, import_fbx, remove_lights_and_cameras

    clear_scene()
    import_fbx(fbx_path, ignore_leaf_bones=False)
    remove_lights_and_cameras()

    armature = next((obj for obj in bpy.data.objects if obj.type == "ARMATURE"), None)
    if armature is None:
        raise RuntimeError(f"No armature found in {fbx_path}")
    return armature


def _load_fbx_skeleton_metadata(
    fbx_path: str,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    armature = _load_fbx_scene(fbx_path)
    return _extract_armature_skeleton_data(armature)


# ── GLB loading helpers ──────────────────────────────────────────────────────

def _load_glb_scene(glb_path: str):
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=glb_path)

    for obj in list(bpy.data.objects):
        if obj.type in {"LIGHT", "CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)

    armature = next((obj for obj in bpy.data.objects if obj.type == "ARMATURE"), None)
    if armature is None:
        raise RuntimeError(f"No armature found in {glb_path}")
    return armature


def _load_glb_skeleton_metadata(
    glb_path: str,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Load a GLB file and extract bone names, parents, offsets, and rest rotations."""
    armature = _load_glb_scene(glb_path)
    return _extract_armature_skeleton_data(armature)


# ── Skeleton data extraction ─────────────────────────────────────────────────

def _extract_armature_skeleton_data(
    armature,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Extract bone names, parents, rest offsets, and rest rotations from an armature."""
    armature_bones = armature.data.bones
    all_roots = [bone for bone in armature_bones if bone.parent is None]
    if not all_roots:
        raise RuntimeError("No root bone found in armature")

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


# ── Animation extraction helpers ─────────────────────────────────────────────

def _get_action_frame_range(armature) -> tuple[int, int, int]:
    if armature.animation_data and armature.animation_data.action:
        action = armature.animation_data.action
        frame_start = int(round(action.frame_range[0]))
        frame_end = int(round(action.frame_range[1]))
    else:
        frame_start = 0
        frame_end = 0
    return frame_start, frame_end, frame_end - frame_start + 1


def _iter_action_fcurves(action):
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


def _get_action_sample_times(armature) -> list[float]:
    action = armature.animation_data.action if armature.animation_data else None
    key_times = sorted({
        round(float(keyframe.co[0]), 6)
        for fcurve in _iter_action_fcurves(action)
        for keyframe in fcurve.keyframe_points
    })
    return key_times or [0.0]


def _infer_sample_fps(scene, sample_times: list[float]) -> float:
    scene_fps = scene.render.fps / scene.render.fps_base
    if len(sample_times) < 2:
        return float(scene_fps)
    deltas = np.diff(np.asarray(sample_times, dtype=np.float64))
    positive_deltas = deltas[deltas > 1e-6]
    if positive_deltas.size == 0:
        return float(scene_fps)
    return float(scene_fps / np.median(positive_deltas))


def _set_scene_time(scene, sample_time: float) -> None:
    frame = math.floor(sample_time)
    subframe = float(sample_time - frame)
    scene.frame_set(frame, subframe=subframe)


# ── FBX → Animation ─────────────────────────────────────────────────────────

def _fbx_to_animation(fbx_path: str) -> tuple[Any, list[str], float]:
    """Load FBX via Blender and return (Animation, joint_names, fps)."""
    import bpy
    from motion_lib.Animation import Animation as ATopAnim
    from motion_lib.Quaternions import Quaternions

    armature = _load_fbx_scene(fbx_path)

    bone_names, parents, offsets, _rest_rotations = _extract_armature_skeleton_data(armature)

    joint_count = len(bone_names)
    orients = Quaternions.id(joint_count)

    scene = bpy.context.scene
    sample_times = _get_action_sample_times(armature)
    fps = _infer_sample_fps(scene, sample_times)
    num_frames = len(sample_times)

    rot_qs = np.zeros((num_frames, joint_count, 4), dtype=np.float64)
    pos_np = np.zeros((num_frames, joint_count, 3), dtype=np.float64)

    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="POSE")

    # Pre-build ordered pose_bone list to avoid repeated dict lookups
    pose_bones = armature.pose.bones
    ordered_pose_bones = [pose_bones.get(bone_name) for bone_name in bone_names]

    for frame_idx, sample_time in enumerate(sample_times):
        _set_scene_time(scene, sample_time)
        bpy.context.view_layer.update()

        for joint_idx, pose_bone in enumerate(ordered_pose_bones):
            if pose_bone is None:
                rot_qs[frame_idx, joint_idx] = [1.0, 0.0, 0.0, 0.0]
                pos_np[frame_idx, joint_idx] = offsets[joint_idx]
                continue

            rot = pose_bone.rotation_quaternion
            rot_qs[frame_idx, joint_idx] = [rot.w, rot.x, rot.y, rot.z]
            loc = pose_bone.location
            pos_np[frame_idx, joint_idx] = [loc.x, loc.y, loc.z]

    bpy.ops.object.mode_set(mode="OBJECT")

    anim = ATopAnim(Quaternions(rot_qs), pos_np, orients, offsets, parents)
    return anim, bone_names, fps


# ── GLB → Animation ─────────────────────────────────────────────────────────

def _glb_to_animation(glb_path: str) -> tuple[Any, list[str], float]:
    """Load GLB via Blender and return (Animation, joint_names, fps)."""
    import bpy
    from motion_lib.Animation import Animation as ATopAnim
    from motion_lib.Quaternions import Quaternions

    armature = _load_glb_scene(glb_path)

    bone_names, parents, offsets, _rest_rotations = _extract_armature_skeleton_data(armature)

    joint_count = len(bone_names)
    orients = Quaternions.id(joint_count)

    scene = bpy.context.scene
    sample_times = _get_action_sample_times(armature)
    fps = _infer_sample_fps(scene, sample_times)
    num_frames = len(sample_times)

    rot_qs = np.zeros((num_frames, joint_count, 4), dtype=np.float64)
    pos_np = np.zeros((num_frames, joint_count, 3), dtype=np.float64)

    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="POSE")

    # Pre-build ordered pose_bone list to avoid repeated dict lookups
    pose_bones = armature.pose.bones
    ordered_pose_bones = [pose_bones.get(bone_name) for bone_name in bone_names]

    for frame_idx, sample_time in enumerate(sample_times):
        _set_scene_time(scene, sample_time)
        bpy.context.view_layer.update()

        for joint_idx, pose_bone in enumerate(ordered_pose_bones):
            if pose_bone is None:
                rot_qs[frame_idx, joint_idx] = [1.0, 0.0, 0.0, 0.0]
                pos_np[frame_idx, joint_idx] = offsets[joint_idx]
                continue

            rot = pose_bone.rotation_quaternion
            rot_qs[frame_idx, joint_idx] = [rot.w, rot.x, rot.y, rot.z]
            loc = pose_bone.location
            pos_np[frame_idx, joint_idx] = [loc.x, loc.y, loc.z]

    bpy.ops.object.mode_set(mode="OBJECT")

    anim = ATopAnim(Quaternions(rot_qs), pos_np, orients, offsets, parents)
    return anim, bone_names, fps


# ── Animation → GLB export ──────────────────────────────────────────────────

def _export_animation_to_glb(
    animation,
    skeleton,
    output_glb: str,
    mesh_path: str,
    fps: float,
) -> None:
    from Anytop.utils.exporter import AnimationExporter

    exporter = AnimationExporter(skeleton, fps=fps)
    joint_rotations = torch.from_numpy(animation.rotations.qs.astype(np.float32))
    root_translation = torch.from_numpy(animation.positions[:, 0, :].astype(np.float32))
    root_rotation = torch.zeros((animation.shape[0], 4), dtype=torch.float32)
    root_rotation[:, 0] = 1.0
    bone_translations_np = animation.positions.astype(np.float32).copy()
    bone_translations_np[:, 0, :] = 0.0
    bone_translations = torch.from_numpy(bone_translations_np)
    exporter.export(
        joint_rotations,
        root_translation,
        root_rotation,
        output_glb,
        mesh_path=mesh_path,
        bone_translations=bone_translations,
    )


def _export_animation_to_bvh(
    animation,
    skeleton,
    output_bvh: str,
    fps: float,
    source_fbx: str | None = None,
) -> None:
    """Export Animation to BVH without rest-quaternion baking.

    When ``source_fbx`` is provided, this helper uses Blender's native BVH
    exporter on the imported FBX armature so the standalone BVH matches the
    same armature basis as manual FBX -> BVH export from Blender.

    Otherwise it falls back to ``motion_lib.BVH.save`` with the Animation's raw
    rotations, bypassing ``AnimationExporter._export_bvh`` which bakes
    ``rest_quat`` into rotation channels.

    Position channels are enabled automatically when any non-root joint's
    local translation differs from its static rest offset.  This matches the
    repo's internal Animation semantics where ``anim.positions`` carries the
    full local translation for every joint, not just the root.

    When loading back, NO unbaking is needed — the rotations are already the
    original local rotations.
    """
    if source_fbx is not None:
        import bpy

        armature = _load_fbx_scene(source_fbx)
        frame_start, frame_end, _ = _get_action_frame_range(armature)
        scene = bpy.context.scene
        scene.frame_start = frame_start
        scene.frame_end = frame_end
        bpy.context.view_layer.objects.active = armature
        for obj in bpy.data.objects:
            obj.select_set(False)
        armature.select_set(True)
        bpy.ops.export_anim.bvh(
            filepath=output_bvh,
            frame_start=frame_start,
            frame_end=frame_end,
            rotate_mode='NATIVE',
            root_transform_only=False,
            global_scale=1.0,
        )
        return

    from motion_lib.BVH import save as bvh_save

    # ── Build joint_names, sanitize whitespace ───────────────────────
    joint_names = [b.name.replace(" ", "_") for b in skeleton.bones]

    # ── Decide whether non-root BVH position channels are required ───
    positions_required = False
    if animation.positions.shape[1] > 1:
        nonroot_positions = np.asarray(animation.positions[:, 1:, :], dtype=np.float64)
        rest_offsets = np.asarray(animation.offsets[1:], dtype=np.float64)[None, :, :]
        positions_required = bool(np.any(np.abs(nonroot_positions - rest_offsets) > 1e-4))

    # ── Build a clean Animation with raw (unbaked) rotations ─────────
    #   orients = identity (already the case)
    #   positions = either full local translations for all joints,
    #               or root-only when the animation truly does not need
    #               non-root position channels.
    from motion_lib.Animation import Animation as ATopAnim
    from motion_lib.Quaternions import Quaternions

    F = animation.shape[0]
    J = animation.shape[1]

    if positions_required:
        positions_np = np.asarray(animation.positions, dtype=np.float64).copy()
    else:
        positions_np = np.zeros((F, J, 3), dtype=np.float64)
        positions_np[:, 0, :] = animation.positions[:, 0, :]

    # Use raw rotations (no rest_quat baking)
    rotations_raw = animation.rotations

    clean_anim = ATopAnim(
        rotations_raw, positions_np,
        Quaternions.id(J), animation.offsets, animation.parents,
    )

    bvh_save(
        output_bvh, clean_anim, names=joint_names,
        frametime=1.0 / fps, order='xyz',
        positions=positions_required,
        orients=True,              # multiply identity orients → no-op
        all_joints_as_names=True,  # preserve all joint names
    )


# ── FBX → GLB direct ────────────────────────────────────────────────────────

def _fbx_to_glb(fbx_path: str, output_glb: str) -> None:
    """Export FBX motion to GLB via AnimationExporter to preserve local channels."""
    bone_names, parents, offsets, rest_rotations = _load_fbx_skeleton_metadata(fbx_path)
    animation, anim_bone_names, fps = _fbx_to_animation(fbx_path)
    if anim_bone_names != bone_names:
        raise AssertionError("FBX animation bone order does not match extracted skeleton metadata")
    skeleton = _build_skeleton(bone_names, offsets, parents, rest_rotations)
    _export_animation_to_glb(animation, skeleton, output_glb, mesh_path=fbx_path, fps=fps)


# ── Comparison helpers ───────────────────────────────────────────────────────

def _collect_bone_world_positions(armature, bone_names, sample_times):
    """Collect bone-head world positions at each sample time.

    Optimized: frame-outer / bone-inner loop so that view_layer.update()
    is called once per frame instead of once per (frame, bone) pair.
    Additionally, pose_bone lookups are pre-computed outside the loop.
    """
    import bpy
    from mathutils import Vector

    scene = bpy.context.scene
    num_frames = len(sample_times)
    result = {name: np.zeros((num_frames, 3), dtype=np.float64) for name in bone_names}

    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="OBJECT")

    # Pre-build pose_bone lookup list to avoid repeated dict lookups
    ordered_pose_bones = [armature.pose.bones.get(name) for name in bone_names]

    for frame_idx, sample_time in enumerate(sample_times):
        _set_scene_time(scene, sample_time)
        bpy.context.view_layer.update()

        for joint_idx, bone_name in enumerate(bone_names):
            pose_bone = ordered_pose_bones[joint_idx]
            if pose_bone is None:
                result[bone_name][frame_idx] = (0.0, 0.0, 0.0)
                continue

            head_local = pose_bone.head
            head_world = armature.matrix_world @ Vector((head_local.x, head_local.y, head_local.z))
            result[bone_name][frame_idx] = (head_world.x, head_world.y, head_world.z)

    return result


def _load_fbx_world_pose_data(fbx_path: str, sample_times: Optional[list[float]] = None) -> dict[str, Any]:
    import bpy

    armature = _load_fbx_scene(fbx_path)

    bone_names, parents, offsets, _rest_rotations = _extract_armature_skeleton_data(armature)

    actual_sample_times = _get_action_sample_times(armature)
    compare_times = actual_sample_times if sample_times is None else sample_times
    positions_by_name = _collect_bone_world_positions(armature, bone_names, compare_times)
    positions = np.stack([positions_by_name[name] for name in bone_names], axis=1)

    scene = bpy.context.scene
    fps = _infer_sample_fps(scene, actual_sample_times)
    return {
        "bone_names": bone_names,
        "parents": parents,
        "offsets": offsets,
        "sample_times": compare_times,
        "actual_sample_times": actual_sample_times,
        "num_frames": len(compare_times),
        "actual_num_frames": len(actual_sample_times),
        "fps": fps,
        "positions": positions,
    }


def _load_glb_world_pose_data(glb_path: str, sample_times: Optional[list[float]] = None) -> dict[str, Any]:
    import bpy

    armature = _load_glb_scene(glb_path)

    bone_names, parents, offsets, _rest_rotations = _extract_armature_skeleton_data(armature)

    actual_sample_times = _get_action_sample_times(armature)
    compare_times = actual_sample_times if sample_times is None else sample_times
    positions_by_name = _collect_bone_world_positions(armature, bone_names, compare_times)
    positions = np.stack([positions_by_name[name] for name in bone_names], axis=1)

    scene = bpy.context.scene
    fps = _infer_sample_fps(scene, actual_sample_times)
    return {
        "bone_names": bone_names,
        "parents": parents,
        "offsets": offsets,
        "sample_times": compare_times,
        "actual_sample_times": actual_sample_times,
        "num_frames": len(compare_times),
        "actual_num_frames": len(actual_sample_times),
        "fps": fps,
        "positions": positions,
    }


def _compute_child_centroid_directions(
    positions: np.ndarray,
    parents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a twist-invariant world-space direction per bone from child heads.

    For each bone, use the vector from that bone's head to the centroid of its
    child bone heads. This remains comparable across FBX/glTF imports even when
    the importer chooses different local bone rolls or tail semantics. The
    returned baseline lengths let callers reject degenerate cases where a child
    centroid collapses onto the parent head in one format, making the direction
    angle numerically meaningless.
    """
    frame_count, joint_count, _ = positions.shape
    directions = np.zeros((frame_count, joint_count, 3), dtype=np.float64)
    valid = np.zeros((frame_count, joint_count), dtype=bool)
    lengths = np.zeros((frame_count, joint_count), dtype=np.float64)

    children_by_parent = [[] for _ in range(joint_count)]
    for child_idx, parent_idx in enumerate(parents):
        if parent_idx >= 0:
            children_by_parent[int(parent_idx)].append(child_idx)

    for parent_idx, child_indices in enumerate(children_by_parent):
        if not child_indices:
            continue
        child_centroid = positions[:, child_indices, :].mean(axis=1)
        vectors = child_centroid - positions[:, parent_idx, :]
        vector_lengths = np.linalg.norm(vectors, axis=-1)
        lengths[:, parent_idx] = vector_lengths
        nonzero = vector_lengths > 1e-8
        if np.any(nonzero):
            directions[nonzero, parent_idx, :] = vectors[nonzero] / vector_lengths[nonzero, None]
            valid[nonzero, parent_idx] = True

    return directions, valid, lengths


def _quaternion_angle_degrees(source_q: np.ndarray, target_q: np.ndarray) -> np.ndarray:
    dots = np.sum(source_q * target_q, axis=-1)
    dots = np.clip(np.abs(dots), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))


def _rotate_positions_about_x(positions: np.ndarray, degrees: float) -> np.ndarray:
    if abs(degrees) < 1e-8:
        return positions
    radians = np.deg2rad(degrees)
    cos_theta = np.cos(radians)
    sin_theta = np.sin(radians)
    rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_theta, -sin_theta],
            [0.0, sin_theta, cos_theta],
        ],
        dtype=np.float64,
    )
    return positions @ rotation.T


def _measure_fbx_glb_error(
    fbx_path: str,
    glb_path: str,
    undo_target_global_x_deg: float = 0.0,
) -> dict[str, Any]:
    """Compare FBX source against GLB target by raw sample index.

    This metric is diagnostic only. Blender's FBX and glTF importers can keep
    visually equivalent animation while producing different armature wrapper
    transforms and different bone-head semantics, so cross-format bone-head and
    local-channel errors are not a reliable roundtrip oracle.

    When both assets are re-imported into Blender for comparison, keep
    ``undo_target_global_x_deg`` at ``0.0`` unless a separate probe proves an
    extra correction is still needed. Blender's glTF importer already converts
    the exported Y-up asset back into Blender world space, so manually undoing
    that axis change here can fabricate a large false error.
    """
    source_meta = _load_fbx_world_pose_data(fbx_path)
    target_meta = _load_glb_world_pose_data(glb_path)

    if source_meta["bone_names"] != target_meta["bone_names"]:
        raise AssertionError(
            f"Bone order mismatch between {fbx_path} and {glb_path}: "
            f"{len(source_meta['bone_names'])} vs {len(target_meta['bone_names'])} bones"
        )

    bone_names = source_meta["bone_names"]
    source_sample_times = np.asarray(source_meta["actual_sample_times"], dtype=np.float64)
    target_sample_times = np.asarray(target_meta["actual_sample_times"], dtype=np.float64)
    compare_frame_count = min(len(source_sample_times), len(target_sample_times))
    compare_indices = np.arange(compare_frame_count, dtype=np.int32)

    source_pose = _load_fbx_world_pose_data(
        fbx_path,
        sample_times=source_sample_times[:compare_frame_count].tolist(),
    )
    target_pose = _load_glb_world_pose_data(
        glb_path,
        sample_times=target_sample_times[:compare_frame_count].tolist(),
    )

    target_positions = _rotate_positions_about_x(
        target_pose["positions"],
        -undo_target_global_x_deg,
    )

    position_errors = np.linalg.norm(
        source_pose["positions"] - target_positions, axis=-1,
    )
    per_bone_error = position_errors.max(axis=0)
    per_frame_error = position_errors.max(axis=1)
    worst_flat_idx = int(np.argmax(position_errors))
    worst_frame_idx, worst_bone_idx = np.unravel_index(worst_flat_idx, position_errors.shape)

    source_directions, source_valid, source_lengths = _compute_child_centroid_directions(
        source_pose["positions"],
        source_meta["parents"],
    )
    target_directions, target_valid, target_lengths = _compute_child_centroid_directions(
        target_positions,
        target_meta["parents"],
    )
    valid_direction_mask = source_valid & target_valid
    length_denominator = np.maximum(source_lengths, target_lengths)
    stable_length_ratio = np.zeros_like(source_lengths)
    nonzero_length_mask = length_denominator > 1e-8
    stable_length_ratio[nonzero_length_mask] = (
        np.minimum(source_lengths[nonzero_length_mask], target_lengths[nonzero_length_mask])
        / length_denominator[nonzero_length_mask]
    )
    valid_direction_mask &= stable_length_ratio >= 0.1
    rotation_diag = None
    if np.any(valid_direction_mask):
        direction_dots = np.clip(
            np.sum(source_directions * target_directions, axis=-1),
            -1.0,
            1.0,
        )
        direction_errors = np.zeros_like(direction_dots)
        direction_errors[valid_direction_mask] = np.degrees(
            np.arccos(direction_dots[valid_direction_mask]),
        )
        direction_errors[~valid_direction_mask] = -1.0
        rot_worst_flat_idx = int(np.argmax(direction_errors))
        rot_worst_frame_idx, rot_worst_bone_idx = np.unravel_index(
            rot_worst_flat_idx,
            direction_errors.shape,
        )
        rotation_diag = {
            "label": "World child-direction diagnostic",
            "max_deg": float(direction_errors[rot_worst_frame_idx, rot_worst_bone_idx]),
            "worst_frame": int(rot_worst_frame_idx),
            "worst_bone": bone_names[rot_worst_bone_idx],
        }

    return {
        "bone_names": bone_names,
        "num_frames": compare_frame_count,
        "compare_times": compare_indices.tolist(),
        "source_num_frames": int(len(source_sample_times)),
        "target_num_frames": int(len(target_sample_times)),
        "max_error": float(position_errors.max()),
        "worst_frame": int(worst_frame_idx),
        "worst_time": float(compare_indices[worst_frame_idx]),
        "worst_bone": bone_names[worst_bone_idx],
        "per_bone_error": per_bone_error,
        "per_frame_error": per_frame_error,
        "rotation_diag": rotation_diag,
    }


def _measure_glb_pair_error(source_glb: str, target_glb: str) -> dict[str, Any]:
    source_meta = _load_glb_world_pose_data(source_glb)
    target_meta = _load_glb_world_pose_data(target_glb)

    if source_meta["bone_names"] != target_meta["bone_names"]:
        raise AssertionError(
            f"Bone order mismatch between {source_glb} and {target_glb}: "
            f"{len(source_meta['bone_names'])} vs {len(target_meta['bone_names'])} bones"
        )
    bone_names = source_meta["bone_names"]
    source_offset = source_meta["actual_sample_times"][0]
    target_offset = target_meta["actual_sample_times"][0]
    compare_times = sorted({
        *(round(sample_time - source_offset, 6) for sample_time in source_meta["actual_sample_times"]),
        *(round(sample_time - target_offset, 6) for sample_time in target_meta["actual_sample_times"]),
    })
    if not compare_times:
        raise AssertionError(f"No comparable sample times between {source_glb} and {target_glb}")

    source_pose = _load_glb_world_pose_data(
        source_glb,
        sample_times=[sample_time + source_offset for sample_time in compare_times],
    )
    target_pose = _load_glb_world_pose_data(
        target_glb,
        sample_times=[sample_time + target_offset for sample_time in compare_times],
    )

    position_errors = np.linalg.norm(
        source_pose["positions"] - target_pose["positions"], axis=-1,
    )
    per_bone_error = position_errors.max(axis=0)
    per_frame_error = position_errors.max(axis=1)
    worst_flat_idx = int(np.argmax(position_errors))
    worst_frame_idx, worst_bone_idx = np.unravel_index(worst_flat_idx, position_errors.shape)

    source_anim, source_anim_bones, _source_fps = _glb_to_animation(source_glb)
    target_anim, target_anim_bones, _target_fps = _glb_to_animation(target_glb)
    rotation_diag = None
    if source_anim_bones == target_anim_bones:
        compared_rot_frames = min(source_anim.rotations.qs.shape[0], target_anim.rotations.qs.shape[0])
        compared_rot_joints = min(source_anim.rotations.qs.shape[1], target_anim.rotations.qs.shape[1])
        angle_errors = _quaternion_angle_degrees(
            source_anim.rotations.qs[:compared_rot_frames, :compared_rot_joints],
            target_anim.rotations.qs[:compared_rot_frames, :compared_rot_joints],
        )
        rot_worst_flat_idx = int(np.argmax(angle_errors))
        rot_worst_frame_idx, rot_worst_bone_idx = np.unravel_index(rot_worst_flat_idx, angle_errors.shape)
        rotation_diag = {
            "max_deg": float(angle_errors.max()),
            "worst_frame": int(rot_worst_frame_idx),
            "worst_bone": source_anim_bones[rot_worst_bone_idx],
        }

    return {
        "bone_names": bone_names,
        "num_frames": len(compare_times),
        "compare_times": compare_times,
        "source_num_frames": source_meta["actual_num_frames"],
        "target_num_frames": target_meta["actual_num_frames"],
        "max_error": float(position_errors.max()),
        "worst_frame": int(worst_frame_idx),
        "worst_time": float(compare_times[worst_frame_idx]),
        "worst_bone": bone_names[worst_bone_idx],
        "per_bone_error": per_bone_error,
        "per_frame_error": per_frame_error,
        "rotation_diag": rotation_diag,
    }


def _measure_glb_pair_error_on_display_frames(source_glb: str, target_glb: str) -> dict[str, Any]:
    source_meta = _load_glb_world_pose_data(source_glb)
    target_meta = _load_glb_world_pose_data(target_glb)

    if source_meta["bone_names"] != target_meta["bone_names"]:
        raise AssertionError(
            f"Bone order mismatch between {source_glb} and {target_glb}: "
            f"{len(source_meta['bone_names'])} vs {len(target_meta['bone_names'])} bones"
        )

    bone_names = source_meta["bone_names"]
    source_offset = float(source_meta["actual_sample_times"][0])
    target_offset = float(target_meta["actual_sample_times"][0])
    source_display_frames = int(round(source_meta["actual_sample_times"][-1] - source_offset)) + 1
    target_display_frames = int(round(target_meta["actual_sample_times"][-1] - target_offset)) + 1
    compare_frame_count = min(source_display_frames, target_display_frames)
    compare_times = np.arange(compare_frame_count, dtype=np.float64)

    source_pose = _load_glb_world_pose_data(
        source_glb,
        sample_times=(compare_times + source_offset).tolist(),
    )
    target_pose = _load_glb_world_pose_data(
        target_glb,
        sample_times=(compare_times + target_offset).tolist(),
    )

    position_errors = np.linalg.norm(
        source_pose["positions"] - target_pose["positions"], axis=-1,
    )
    per_bone_error = position_errors.max(axis=0)
    per_frame_error = position_errors.max(axis=1)
    worst_flat_idx = int(np.argmax(position_errors))
    worst_frame_idx, worst_bone_idx = np.unravel_index(worst_flat_idx, position_errors.shape)

    source_anim, source_anim_bones, _source_fps = _glb_to_animation(source_glb)
    target_anim, target_anim_bones, _target_fps = _glb_to_animation(target_glb)
    rotation_diag = None
    if source_anim_bones == target_anim_bones:
        compared_rot_frames = min(source_anim.rotations.qs.shape[0], target_anim.rotations.qs.shape[0])
        compared_rot_joints = min(source_anim.rotations.qs.shape[1], target_anim.rotations.qs.shape[1])
        angle_errors = _quaternion_angle_degrees(
            source_anim.rotations.qs[:compared_rot_frames, :compared_rot_joints],
            target_anim.rotations.qs[:compared_rot_frames, :compared_rot_joints],
        )
        rot_worst_flat_idx = int(np.argmax(angle_errors))
        rot_worst_frame_idx, rot_worst_bone_idx = np.unravel_index(rot_worst_flat_idx, angle_errors.shape)
        rotation_diag = {
            "max_deg": float(angle_errors.max()),
            "worst_frame": int(rot_worst_frame_idx),
            "worst_bone": source_anim_bones[rot_worst_bone_idx],
        }

    return {
        "bone_names": bone_names,
        "num_frames": compare_frame_count,
        "compare_times": compare_times.tolist(),
        "source_num_frames": source_display_frames,
        "target_num_frames": target_display_frames,
        "max_error": float(position_errors.max()),
        "worst_frame": int(worst_frame_idx),
        "worst_time": float(compare_times[worst_frame_idx]),
        "worst_bone": bone_names[worst_bone_idx],
        "per_bone_error": per_bone_error,
        "per_frame_error": per_frame_error,
        "rotation_diag": rotation_diag,
    }


def _print_comparison_report(label: str, metrics: dict[str, Any]) -> None:
    print(f"  [{label}] Full-frame comparison:")
    print(
        f"    Compared samples: {metrics['num_frames']} "
        f"(source={metrics['source_num_frames']}, target={metrics['target_num_frames']})"
    )
    print(
        f"    Max bone-head world error: {metrics['max_error']:.6f}m  "
        f"(bone={metrics['worst_bone']}, sample={metrics['worst_frame']}, time={metrics['worst_time']:.6f})"
    )

    top_bones = np.argsort(metrics["per_bone_error"])[::-1][:5]
    print("    Top 5 bones:")
    for bone_idx in top_bones:
        bone_name = metrics["bone_names"][bone_idx]
        print(f"      {bone_name:<35} {metrics['per_bone_error'][bone_idx]:.6f}")

    top_frames = np.argsort(metrics["per_frame_error"])[::-1][:5]
    print("    Top 5 frames:")
    for frame_idx in top_frames:
        print(
            f"      sample {int(frame_idx):<4} time={metrics['compare_times'][frame_idx]:.6f} "
            f"{metrics['per_frame_error'][frame_idx]:.6f}"
        )

    rotation_diag = metrics.get("rotation_diag")
    if rotation_diag is not None:
        diag_label = rotation_diag.get("label", "Rotation diagnostic")
        print(
            f"    {diag_label}: {rotation_diag['max_deg']:.6f}deg "
            f"(bone={rotation_diag['worst_bone']}, frame={rotation_diag['worst_frame']})"
        )
