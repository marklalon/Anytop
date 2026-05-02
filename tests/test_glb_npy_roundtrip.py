"""
GLB -> NPY -> GLB roundtrip test.

Loads a source GLB animation through Blender, verifies Blender's own GLB
roundtrip by exporting `original2.glb`, exports the animation to AnyTop's
13-channel NPY motion features, recovers an Animation using a T-pose GLB,
exports `recovered_export.glb`, and compares both exported GLBs against the
source GLB on every frame and every bone in Blender world space.

Requires bpy (Blender as Python module) in the current Python environment.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch


# ── ensure repo root is on sys.path ──────────────────────────────────────────
_ANYTOP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_ANYTOP_ROOT)
sys.path.insert(0, _REPO_ROOT)

# ── Resolve utils namespace conflict ────────────────────────────────────────
import utils
import utils.geometry
import utils.quaternion

import importlib.machinery
import importlib.util

_rotconv_path = os.path.join(_ANYTOP_ROOT, "utils", "rotation_conversions.py")
if os.path.isfile(_rotconv_path) and "utils.rotation_conversions" not in sys.modules:
    _loader = importlib.machinery.SourceFileLoader(
        "utils.rotation_conversions", _rotconv_path,
    )
    _spec = importlib.util.spec_from_loader(
        "utils.rotation_conversions", _loader, origin=_rotconv_path,
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["utils.rotation_conversions"] = _mod
    _spec.loader.exec_module(_mod)

_npy_rt_path = os.path.join(_ANYTOP_ROOT, "utils", "npy_roundtrip_utils.py")
if os.path.isfile(_npy_rt_path) and "utils.npy_roundtrip_utils" not in sys.modules:
    _loader = importlib.machinery.SourceFileLoader(
        "utils.npy_roundtrip_utils", _npy_rt_path,
    )
    _spec = importlib.util.spec_from_loader(
        "utils.npy_roundtrip_utils", _loader, origin=_npy_rt_path,
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["utils.npy_roundtrip_utils"] = _mod
    _spec.loader.exec_module(_mod)

if _ANYTOP_ROOT not in sys.path:
    sys.path.insert(1, _ANYTOP_ROOT)


from utils.npy_roundtrip_utils import (
    build_roundtrip_feature_payload,
    coerce_feature_payload,
    recover_from_features,
    extract_raw_features,
    compute_rest_positions,
    get_cont6d_params_own,
    detect_motion_loop,
    compute_terminal_local_velocity,
)


def _extract_armature_skeleton_data(armature) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Extract bone names, parents, and rest offsets from an armature."""
    from collections import deque

    edit_bones = armature.data.edit_bones
    all_roots = [edit_bone for edit_bone in edit_bones if edit_bone.parent is None]
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

    ordered_edit_bones = []
    queue = deque([root_bone])
    while queue:
        edit_bone = queue.popleft()
        ordered_edit_bones.append(edit_bone)
        queue.extend(edit_bone.children)

    joint_count = len(ordered_edit_bones)
    bone_names = [edit_bone.name for edit_bone in ordered_edit_bones]
    parents = np.full(joint_count, -1, dtype=np.int32)
    offsets = np.zeros((joint_count, 3), dtype=np.float64)
    name_to_idx = {name: idx for idx, name in enumerate(bone_names)}

    for joint_idx, edit_bone in enumerate(ordered_edit_bones):
        if edit_bone.parent is not None and edit_bone.parent.name in name_to_idx:
            parent_idx = name_to_idx[edit_bone.parent.name]
            parents[joint_idx] = parent_idx
            offsets[joint_idx] = np.array(edit_bone.head) - np.array(edit_bone.parent.head)
        else:
            offsets[joint_idx] = np.array(edit_bone.head)

    return bone_names, parents, offsets


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


def _glb_to_animation(glb_path: str) -> tuple[Any, list[str], float]:
    """Load GLB via Blender and return (Animation, joint_names, fps)."""
    import bpy

    from motion_lib.Animation import Animation as ATopAnim
    from motion_lib.Quaternions import Quaternions

    armature = _load_glb_scene(glb_path)

    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="EDIT")
    bone_names, parents, offsets = _extract_armature_skeleton_data(armature)
    bpy.ops.object.mode_set(mode="OBJECT")

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

    pose_bones = armature.pose.bones
    for frame_idx, sample_time in enumerate(sample_times):
        _set_scene_time(scene, sample_time)
        bpy.context.view_layer.update()

        for joint_idx, bone_name in enumerate(bone_names):
            pose_bone = pose_bones.get(bone_name)
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


def _roundtrip_glb_via_blender(source_glb: str, output_glb: str) -> None:
    import bpy

    armature = _load_glb_scene(source_glb)
    scene = bpy.context.scene
    sample_times = _get_action_sample_times(armature)
    first_time = sample_times[0]
    source_fps = scene.render.fps
    source_fps_base = scene.render.fps_base

    action = armature.animation_data.action if armature.animation_data else None
    for fcurve in _iter_action_fcurves(action):
        for keyframe in fcurve.keyframe_points:
            keyframe.co[0] = keyframe.co[0] - first_time
            keyframe.handle_left[0] = keyframe.handle_left[0] - first_time
            keyframe.handle_right[0] = keyframe.handle_right[0] - first_time
            keyframe.interpolation = "LINEAR"

    scene.render.fps = source_fps
    scene.render.fps_base = source_fps_base
    scene.frame_start = 0
    scene.frame_end = max(int(round(sample_times[-1] - first_time)), 0)

    os.makedirs(os.path.dirname(output_glb) or ".", exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=output_glb,
        export_format="GLB",
        export_animations=True,
        export_force_sampling=True,
        export_frame_range=True,
        export_apply=False,
    )


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


def _build_skeleton(bone_names, offsets, parents, device=None):
    if device is None:
        device = torch.device("cpu")

    bones = []
    for joint_idx, bone_name in enumerate(bone_names):
        parent_idx = parents[joint_idx]
        bones.append(_SimpleBone(
            id=joint_idx,
            name=bone_name,
            parent_id=None if parent_idx < 0 else int(parent_idx),
            rest_offset=torch.tensor(offsets[joint_idx], dtype=torch.float32, device=device),
            rest_rotation=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device),
        ))
    return _SimpleSkeleton(bones)


def _collect_bone_world_positions(armature, bone_names, sample_times):
    import bpy
    from mathutils import Vector

    scene = bpy.context.scene
    result: dict[str, np.ndarray] = {}

    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="OBJECT")

    positions = np.zeros((len(sample_times), 3), dtype=np.float64)
    for bone_name in bone_names:
        for frame_idx, sample_time in enumerate(sample_times):
            _set_scene_time(scene, sample_time)
            bpy.context.view_layer.update()

            pose_bone = armature.pose.bones.get(bone_name)
            if pose_bone is None:
                positions[frame_idx] = (0.0, 0.0, 0.0)
                continue

            head_local = pose_bone.head
            head_world = armature.matrix_world @ Vector((head_local.x, head_local.y, head_local.z))
            positions[frame_idx] = (head_world.x, head_world.y, head_world.z)
        result[bone_name] = positions.copy()

    return result


def _load_glb_world_pose_data(glb_path: str, sample_times: Optional[list[float]] = None) -> dict[str, Any]:
    import bpy

    armature = _load_glb_scene(glb_path)

    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="EDIT")
    bone_names, parents, offsets = _extract_armature_skeleton_data(armature)
    bpy.ops.object.mode_set(mode="OBJECT")

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


def _quaternion_angle_degrees(source_q: np.ndarray, target_q: np.ndarray) -> np.ndarray:
    dots = np.sum(source_q * target_q, axis=-1)
    dots = np.clip(np.abs(dots), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))


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
    print(f"  [{label}] Full-frame GLB comparison:")
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
        print(
            f"    Rotation diagnostic: {rotation_diag['max_deg']:.6f}deg "
            f"(bone={rotation_diag['worst_bone']}, frame={rotation_diag['worst_frame']})"
        )


def test_glb_npy_roundtrip(
    tpose_glb: str,
    anim_glb: str,
    object_type: str = "Alligator",
    output_dir: str | None = None,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    for file_path in [tpose_glb, anim_glb]:
        assert os.path.isfile(file_path), f"Missing required file: {file_path}"

    temp_context = nullcontext(output_dir) if output_dir else tempfile.TemporaryDirectory()
    with temp_context as work_dir:
        assert work_dir is not None
        os.makedirs(work_dir, exist_ok=True)

        original2_glb = os.path.join(work_dir, "original2.glb")
        recovered_glb = os.path.join(work_dir, "recovered_export.glb")
        npy_path = os.path.join(work_dir, "roundtrip_features.npy")

        print(f"[GLB Roundtrip] T-pose GLB : {tpose_glb}")
        print(f"[GLB Roundtrip] Source GLB : {anim_glb}")
        print(f"[GLB Roundtrip] Output dir : {work_dir}")

        print("  [Phase A] Loading T-pose GLB for skeleton metadata...")
        tpose_anim, tpose_bone_names, tpose_fps = _glb_to_animation(tpose_glb)
        offsets = tpose_anim.offsets.copy()
        parents = tpose_anim.parents.copy()
        print(f"    Joints: {len(tpose_bone_names)}, FPS: {tpose_fps:.1f}")

        print("  [Phase B] Loading source GLB and extracting motion...")
        source_anim, source_bone_names, source_fps = _glb_to_animation(anim_glb)
        print(f"    Frames: {len(source_anim)}, Joints: {source_anim.shape[1]}, FPS: {source_fps:.1f}")

        if source_bone_names != tpose_bone_names:
            raise AssertionError(
                "Source GLB and T-pose GLB do not share the same BFS bone order"
            )

        print("  [Phase C] Blender GLB roundtrip baseline -> original2.glb...")
        _roundtrip_glb_via_blender(anim_glb, original2_glb)

        print("  [Phase D] Extracting raw NPY features...")
        feature_payload = build_roundtrip_feature_payload(
            source_anim, object_type, offsets, parents, source_bone_names,
        )
        np.save(npy_path, feature_payload, allow_pickle=True)
        print(f"    NPY shape: {feature_payload['features'].shape}")
        print(f"    Saved NPY features to {npy_path}")

        print("  [Phase E] Recovering Animation from NPY features...")
        recovered_anim, has_animated_pos = recover_from_features(
            feature_payload, parents, offsets,
        )
        print(f"    Recovered frames: {len(recovered_anim)}")
        if has_animated_pos:
            print("    (has non-root animated position channels)")

        from motion_lib.Animation import positions_global

        source_global = positions_global(source_anim)
        recovered_global = positions_global(recovered_anim)
        npy_position_error = np.abs(source_global - recovered_global).max(axis=(0, 2))
        npy_worst_idx = int(np.argmax(npy_position_error))
        npy_worst_bone = source_bone_names[npy_worst_idx] if npy_worst_idx < len(source_bone_names) else "?"
        print(
            "  [Diag] Animation-domain source-vs-recovered max per-joint error: "
            f"{npy_position_error.max():.6f} ({npy_worst_bone})"
        )
        print("    Note: this is diagnostic only because recovery is built on the T-pose GLB skeleton.")

        recovered_skeleton = _build_skeleton(tpose_bone_names, offsets, parents)
        from postprocessing.exporter import AnimationExporter

        recovered_exporter = AnimationExporter(recovered_skeleton, fps=source_fps)
        jq_t = torch.from_numpy(recovered_anim.rotations.qs.astype(np.float32))
        rt_t = torch.from_numpy(recovered_anim.positions[:, 0, :].astype(np.float32))
        rr_t = torch.from_numpy(recovered_anim.rotations.qs[:, 0, :].astype(np.float32))
        bt_t = torch.from_numpy(recovered_anim.positions.astype(np.float32))

        print("  [Phase F] Exporting NPY-recovered animation -> recovered_export.glb...")
        recovered_exporter.export(
            jq_t,
            rt_t,
            rr_t,
            recovered_glb,
            mesh_path=tpose_glb,
            bone_translations=bt_t,
        )

        original2_metrics = _measure_glb_pair_error_on_display_frames(anim_glb, original2_glb)
        _print_comparison_report("Experiment A", original2_metrics)

        recovered_metrics = _measure_glb_pair_error(anim_glb, recovered_glb)
        _print_comparison_report("Experiment C", recovered_metrics)

        assert original2_metrics["max_error"] < tolerance, (
            f"original.glb -> original2.glb max error {original2_metrics['max_error']:.6f} exceeds "
            f"{tolerance} (worst bone={original2_metrics['worst_bone']}, "
            f"sample={original2_metrics['worst_frame']}, time={original2_metrics['worst_time']:.6f})"
        )
        assert recovered_metrics["max_error"] < tolerance, (
            f"original.glb -> recovered_export.glb max error {recovered_metrics['max_error']:.6f} exceeds "
            f"{tolerance} (worst bone={recovered_metrics['worst_bone']}, "
            f"sample={recovered_metrics['worst_frame']}, time={recovered_metrics['worst_time']:.6f})"
        )

        print("\n  PASS  GLB roundtrip checks passed")
        return {
            "npy_error": float(npy_position_error.max()),
            "original2_error": float(original2_metrics["max_error"]),
            "original2_worst_bone": original2_metrics["worst_bone"],
            "original2_worst_frame": int(original2_metrics["worst_frame"]),
            "original2_worst_time": float(original2_metrics["worst_time"]),
            "recovered_error": float(recovered_metrics["max_error"]),
            "recovered_worst_bone": recovered_metrics["worst_bone"],
            "recovered_worst_frame": int(recovered_metrics["worst_frame"]),
            "recovered_worst_time": float(recovered_metrics["worst_time"]),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GLB -> NPY -> GLB roundtrip smoke test",
    )
    parser.add_argument(
        "--tpose-glb",
        default=os.path.join(_ANYTOP_ROOT, "outputs", "glb_npy_roundtrip", "tpose.glb"),
        help="Path to T-pose GLB file used as skeleton metadata and export container.",
    )
    parser.add_argument(
        "--anim-glb",
        default=os.path.join(_ANYTOP_ROOT, "outputs", "glb_npy_roundtrip", "original.glb"),
        help="Path to source animation GLB file.",
    )
    parser.add_argument(
        "--object-type",
        default="Horse",
        help="Character type for contact inference (default: Horse).",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(_ANYTOP_ROOT, "outputs", "glb_npy_roundtrip"),
        help="Directory to save roundtrip artifacts.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Max allowed GLB comparison error in meters (default: 1e-4).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"T-pose GLB : {args.tpose_glb}")
    print(f"Anim GLB   : {args.anim_glb}")
    print(f"Output dir : {args.output_dir}")
    print(f"Object type: {args.object_type}")
    print(f"Tolerance  : {args.tolerance}")
    print()

    result = test_glb_npy_roundtrip(
        tpose_glb=args.tpose_glb,
        anim_glb=args.anim_glb,
        object_type=args.object_type,
        output_dir=args.output_dir,
        tolerance=args.tolerance,
    )

    print("\nSummary:")
    print(f"  NPY encoding error         : {result['npy_error']:.6f}")
    print(
        f"  original -> original2      : {result['original2_error']:.6f} "
        f"(bone={result['original2_worst_bone']}, sample={result['original2_worst_frame']}, "
        f"time={result['original2_worst_time']:.6f})"
    )
    print(
        f"  original -> recovered      : {result['recovered_error']:.6f} "
        f"(bone={result['recovered_worst_bone']}, sample={result['recovered_worst_frame']}, "
        f"time={result['recovered_worst_time']:.6f})"
    )