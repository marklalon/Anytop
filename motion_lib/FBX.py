"""
FBX.load — drop-in replacement for BVH.load that sources animation from an FBX file
loaded through Blender (bpy).  The public interface is intentionally identical to
BVH.load so that all downstream call sites in motion_process.py can swap the two
with a one-line change.

Note: loading requires a live Blender (bpy) session.  Because bpy is single-threaded
and stateful (clear_scene affects the whole process), callers must NOT invoke
FBX.load concurrently from multiple threads.
"""

from __future__ import annotations

import contextlib
import io
import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


# ── FBX import utilities (merged from utils/fbx.py) ─────────────────────────

def patch_fbx_light_import():
    """Monkey-patch the FBX importer's blen_read_light to handle Blender 5.0."""
    import sys
    import importlib

    mod = sys.modules.get("io_scene_fbx.import_fbx")
    if mod is None:
        try:
            mod = importlib.import_module("io_scene_fbx.import_fbx")
        except ImportError:
            return
    if mod is None or not hasattr(mod, "blen_read_light"):
        return

    original_fn = mod.blen_read_light

    def _patched_blen_read_light(fbx_tmpl, fbx_obj, settings, _orig=original_fn):
        try:
            return _orig(fbx_tmpl, fbx_obj, settings)
        except AttributeError as exc:
            if "cast_shadow" in str(exc):
                return None
            raise

    mod.blen_read_light = _patched_blen_read_light


def import_fbx(filepath: str, ignore_leaf_bones: bool = True) -> None:
    """Import an FBX file into the current Blender scene."""
    import bpy

    patch_fbx_light_import()
    bpy.ops.import_scene.fbx(
        filepath=filepath,
        ignore_leaf_bones=ignore_leaf_bones,
        force_connect_children=False,
        automatic_bone_orientation=True,
        bake_space_transform=False,
        use_custom_normals=False,
        use_image_search=False,
    )


def clear_scene() -> None:
    """Reset Blender to a fresh empty scene."""
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)


def remove_lights_and_cameras() -> None:
    """Remove all LIGHT and CAMERA objects from the current scene."""
    import bpy

    for obj in list(bpy.data.objects):
        if obj.type in {"LIGHT", "CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)


# ── FBX loading helpers ──────────────────────────────────────────────────────

def _load_fbx_scene(fbx_path: str):
    """Import an FBX file into a fresh Blender scene and return the armature."""
    import bpy

    clear_scene()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import_fbx(fbx_path, ignore_leaf_bones=False)
    remove_lights_and_cameras()

    armature = next((obj for obj in bpy.data.objects if obj.type == "ARMATURE"), None)
    if armature is None:
        raise RuntimeError(f"No armature found in {fbx_path}")
    return armature


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


# ── Public API ───────────────────────────────────────────────────────────────

def load(filepath, start=None, end=None, order=None, world=True):
    """Load an FBX file and return (Animation, joint_names, frametime).

    Parameters
    ----------
    filepath : str | Path
        Path to the FBX file.
    start : int, optional
        First frame index to include (0-based).  ``None`` means beginning.
    end : int, optional
        One-past-last frame index.  ``None`` means end.
    order : ignored
        Accepted for signature compatibility with BVH.load; has no effect.
    world : ignored
        Accepted for signature compatibility with BVH.load; has no effect.

    Returns
    -------
    anim : Animation
        Joint rotations (.rotations), local positions (.positions),
        orientations (.orients), rest offsets (.offsets), and parent
        indices (.parents) — same structure as BVH.load output.
    joint_names : list[str]
        BFS-ordered joint names extracted from the FBX armature.
    frametime : float
        Seconds per frame (1 / fps).
    """
    anim, names, fps = _fbx_to_animation(str(filepath))

    frametime = 1.0 / fps if fps > 0 else (1.0 / 30.0)

    if start is not None or end is not None:
        anim = anim[start:end]

    return anim, names, frametime
