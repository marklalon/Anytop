"""
Shared helpers for FBX/GLB/NPY/BVH roundtrip tests.

Extracted from test_fbx_glb_npy_roundtrip.py to avoid duplication
between NPY and BVH roundtrip tests.
"""
from __future__ import annotations

import contextlib
import io
import math
from collections import deque
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
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
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
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
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



