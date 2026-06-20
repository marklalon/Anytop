"""
FBX.load — drop-in replacement for BVH.load that sources animation from an FBX
or GLB/GLTF file loaded through Blender (bpy).  The public interface is
intentionally identical to BVH.load so that all downstream call sites in
motion_process.py can swap the two with a one-line change.

Note: loading requires a live Blender (bpy) session.  Because bpy is single-threaded
and stateful (clear_scene affects the whole process), callers must NOT invoke
FBX.load concurrently from multiple threads.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .root_collapse import collapse_root_skeleton
except ImportError:
    from root_collapse import collapse_root_skeleton


# ── FBX import utilities (merged from utils/fbx.py) ─────────────────────────

def import_fbx(filepath: str, use_image_search: bool = False) -> None:
    """Import an FBX file into the current Blender scene.

    Always imports with ``ignore_leaf_bones=False`` so that leaf bones carrying
    animation (tail tips, hair, halter, etc.) are preserved.

    When ``use_image_search`` is true, run Blender's missing-file search after
    import so external textures in sibling folders such as ``tex/`` are
    resolved for the new ``wm.fbx_import`` operator, which does not expose the
    old add-on importer's ``use_image_search`` parameter directly, then repair
    any image datablocks whose pixels never loaded (see
    :func:`_reload_unloaded_images`).
    """
    import bpy

    with _silence_os_std():
        bpy.ops.wm.fbx_import(
            filepath=filepath,
            ignore_leaf_bones=False,
            use_custom_normals=False,
            use_anim=True,
        )
    if use_image_search:
        bpy.ops.file.find_missing_files(
            directory=os.path.dirname(os.path.abspath(filepath))
        )
        _reload_unloaded_images(bpy)


def _reload_unloaded_images(bpy) -> None:
    """Force-load image datablocks whose pixels failed to populate at import.

    ``wm.fbx_import`` creates image datablocks pointing at the FBX's embedded
    ``.fbm`` cache path; when that cache is absent the pixels never load
    (``size == (0, 0)``).  ``find_missing_files`` relinks the *filepath* to the
    real texture on disk but does **not** repopulate the pixels, and
    ``Image.reload()`` cannot revive such a datablock.  The glTF exporter embeds
    images by their loaded pixel data, so an unloaded image is silently dropped
    from the GLB — the restored mesh loses its diffuse (texture wiring is
    otherwise correct).

    For each image that has a readable on-disk filepath but no loaded pixels,
    load a fresh datablock (``check_existing=False`` — a matching path would
    otherwise hand back the same broken datablock) and remap every user
    (material nodes, etc.) onto it.  Healthy images (already-loaded or packed)
    are left untouched, so the normal path is unaffected.
    """
    for image in list(bpy.data.images):
        if tuple(image.size) != (0, 0):
            continue
        if image.packed_file is not None:
            continue
        abspath = bpy.path.abspath(image.filepath_raw or image.filepath)
        if not abspath or not os.path.isfile(abspath):
            continue
        try:
            fresh = bpy.data.images.load(abspath, check_existing=False)
        except RuntimeError:
            continue
        if tuple(fresh.size) == (0, 0):
            bpy.data.images.remove(fresh)
            continue
        # Preserve the original colorspace so base-color vs. data (normal/
        # alpha) textures keep their correct interpretation after the swap.
        try:
            fresh.colorspace_settings.name = image.colorspace_settings.name
        except (RuntimeError, TypeError):
            pass
        name = image.name
        image.user_remap(fresh)
        bpy.data.images.remove(image)
        fresh.name = name


def import_gltf(filepath: str) -> None:
    """Import a GLB/GLTF file into the current Blender scene."""
    import bpy

    bpy.ops.import_scene.gltf(filepath=filepath)


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


# ── FBX/GLB loading helpers ──────────────────────────────────────────────────

@contextlib.contextmanager
def _silence_os_std():
    """Context manager that redirects OS-level fd 1 & 2 to /dev/null.

    bpy's C-level perfmon writes directly to OS file descriptors, bypassing
    Python's sys.stdout/stderr.  Use this as a complement to
    ``contextlib.redirect_stdout`` when both Python-level and OS-level output
    must be suppressed.
    """
    _devnull_fd = os.open(os.devnull, os.O_WRONLY)
    _saved_out = os.dup(1)
    _saved_err = os.dup(2)
    try:
        os.dup2(_devnull_fd, 1)
        os.dup2(_devnull_fd, 2)
        yield
    finally:
        os.dup2(_saved_out, 1)
        os.dup2(_saved_err, 2)
        os.close(_saved_out)
        os.close(_saved_err)
        os.close(_devnull_fd)


def _load_scene(filepath: str):
    """Import an FBX/GLB/GLTF file into a fresh Blender scene and return the armature."""
    import bpy

    path = str(filepath)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Animation file not found: {path}\n"
            f"Please verify the path exists and the file is accessible."
        )
    suffix = Path(path).suffix.lower()

    clear_scene()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        if suffix == ".fbx":
            import_fbx(path)
        elif suffix in {".glb", ".gltf"}:
            import_gltf(path)
        else:
            raise ValueError(
                f"Unsupported animation format: {suffix} (supported: .fbx, .glb, .gltf)"
            )
    remove_lights_and_cameras()

    armature = next((obj for obj in bpy.data.objects if obj.type == "ARMATURE"), None)
    if armature is None:
        raise RuntimeError(f"No armature found in {path}")
    return armature


def load_fbx_scene(fbx_path: str):
    """Import an FBX file into a fresh Blender scene and return the armature."""
    return _load_scene(fbx_path)

def extract_armature_skeleton_data(
    armature,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Extract bone names, parents, rest offsets, and rest rotations from an armature.

    Bones are returned in depth-first pre-order so the joint indexing matches the
    hierarchy order produced by BVH.save/BVH.load for the same skeleton.
    """
    armature_bones = armature.data.bones
    all_roots = [bone for bone in armature_bones if bone.parent is None]
    if not all_roots:
        raise RuntimeError("No root bone found in armature")

    # Some FBX exports wrap the real skeleton in a synthetic "null" root.
    # Skip that wrapper bone itself, but promote its child chains as root
    # candidates so we still import the actual skeleton.
    candidate_roots = []
    for bone in all_roots:
        if bone.name.lower() == "null":
            candidate_roots.extend(bone.children)
        else:
            candidate_roots.append(bone)
    if not candidate_roots:
        raise RuntimeError(
            "No valid root bone found in armature after skipping 'null' roots"
        )

    def _subtree_size(root_bone) -> int:
        count = 0
        queue = deque([root_bone])
        while queue:
            bone = queue.popleft()
            count += 1
            queue.extend(bone.children)
        return count

    root_bone = max(candidate_roots, key=_subtree_size)

    ordered_bones = []
    def _append_preorder(bone) -> None:
        ordered_bones.append(bone)

        for child in bone.children:
            # Skip mesh-binding bones (e.g. "Mesh") that some Truebones rigs
            # attach as a sibling of the skeleton chain under the root.
            # These carry the character mesh, not skeletal animation.
            if "mesh" in child.name.lower():
                continue
            _append_preorder(child)

    _append_preorder(root_bone)

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

def iter_action_fcurves(action):
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


def get_action_sample_times(armature) -> list[float]:
    action = armature.animation_data.action if armature.animation_data else None
    key_times = sorted({
        round(float(keyframe.co[0]), 6)
        for fcurve in iter_action_fcurves(action)
        for keyframe in fcurve.keyframe_points
    })
    return key_times or [0.0]


def infer_sample_fps(scene, sample_times: list[float]) -> float:
    scene_fps = scene.render.fps / scene.render.fps_base
    if len(sample_times) < 2:
        return float(scene_fps)
    deltas = np.diff(np.asarray(sample_times, dtype=np.float64))
    positive_deltas = deltas[deltas > 1e-6]
    if positive_deltas.size == 0:
        return float(scene_fps)
    return float(scene_fps / np.median(positive_deltas))


def set_scene_time(scene, sample_time: float) -> None:
    frame = math.floor(sample_time)
    subframe = float(sample_time - frame)
    scene.frame_set(frame, subframe=subframe)


# ── FBX/GLB → Animation ─────────────────────────────────────────────────────

def _scene_to_animation(scene_path: str, collapse_root: bool = True) -> tuple[Any, list[str], float]:
    """Load FBX/GLB/GLTF via Blender and return (Animation, joint_names, fps).

    Parameters
    ----------
    collapse_root : bool, default True
        When True (default), redundant root joints and zero-offset wrapper
        roots are collapsed.  Set to False for exporter/restore paths where
        the skeleton hierarchy must remain exactly as-is.
    """
    import bpy
    from motion_lib.Animation import Animation as ATopAnim
    from motion_lib.Quaternions import Quaternions

    armature = _load_scene(scene_path)
    bone_names, parents, offsets, rest_rotations = extract_armature_skeleton_data(armature)

    joint_count = len(bone_names)
    orients = Quaternions(rest_rotations)

    scene = bpy.context.scene
    sample_times = get_action_sample_times(armature)
    fps = infer_sample_fps(scene, sample_times)
    num_frames = len(sample_times)

    rot_qs = np.zeros((num_frames, joint_count, 4), dtype=np.float64)
    pos_np = np.zeros((num_frames, joint_count, 3), dtype=np.float64)

    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="POSE")

    # Pre-build ordered pose_bone list to avoid repeated dict lookups
    pose_bones = armature.pose.bones
    ordered_pose_bones = [pose_bones.get(bone_name) for bone_name in bone_names]
    ordered_parent_indices = parents.tolist()

    for frame_idx, sample_time in enumerate(sample_times):
        set_scene_time(scene, sample_time)

        pose_matrices = [
            pose_bone.matrix.copy() if pose_bone is not None else None
            for pose_bone in ordered_pose_bones
        ]
        parent_inverse_matrices = [
            pose_matrices[parent_idx].inverted_safe()
            if parent_idx >= 0 and pose_matrices[parent_idx] is not None
            else None
            for parent_idx in ordered_parent_indices
        ]

        for joint_idx, pose_bone in enumerate(ordered_pose_bones):
            if pose_bone is None:
                rot_qs[frame_idx, joint_idx] = [1.0, 0.0, 0.0, 0.0]
                pos_np[frame_idx, joint_idx] = offsets[joint_idx]
                continue

            pose_matrix = pose_matrices[joint_idx]
            parent_inverse = parent_inverse_matrices[joint_idx]
            if parent_inverse is None:
                local_matrix = pose_matrix
            else:
                local_matrix = parent_inverse @ pose_matrix

            t = local_matrix.translation
            q = local_matrix.to_quaternion()
            pos_np[frame_idx, joint_idx] = [t.x, t.y, t.z]
            rot_qs[frame_idx, joint_idx] = [q.w, q.x, q.y, q.z]

    bpy.ops.object.mode_set(mode="OBJECT")

    if collapse_root:
        bone_names, parents, offsets, rot_qs, pos_np, orients = collapse_root_skeleton(
            bone_names,
            parents,
            offsets,
            rot_qs,
            pos_np,
            orients,
            warn_path=scene_path,
        )

    anim = ATopAnim(Quaternions(rot_qs), pos_np, orients, offsets, parents)
    return anim, bone_names, fps


def fbx_to_animation(fbx_path: str, collapse_root: bool = True) -> tuple[Any, list[str], float]:
    """Load FBX via Blender and return (Animation, joint_names, fps)."""
    return _scene_to_animation(fbx_path, collapse_root=collapse_root)


# ── Public API ───────────────────────────────────────────────────────────────

def load(filepath, start=None, end=None, order=None, world=True, collapse_root=True):
    """Load an FBX/GLB/GLTF file and return (Animation, joint_names, frametime).

    Parameters
    ----------
    filepath : str | Path
        Path to the FBX/GLB/GLTF file.
    start : int, optional
        First frame index to include (0-based).  ``None`` means beginning.
    end : int, optional
        One-past-last frame index.  ``None`` means end.
    order : ignored
        Accepted for signature compatibility with BVH.load; has no effect.
    world : ignored
        Accepted for signature compatibility with BVH.load; has no effect.
    collapse_root : bool, default True
        When False, skips both the redundant root joint removal and the
        post-processing root collapse.  Use ``False`` for exporter/restore
        workflows where the skeleton hierarchy must remain unchanged.

    Returns
    -------
    anim : Animation
        Joint rotations (.rotations), local positions (.positions),
        orientations (.orients), rest offsets (.offsets), and parent
        indices (.parents) — same structure as BVH.load output.
    joint_names : list[str]
        Depth-first pre-order joint names extracted from the imported armature.
    frametime : float
        Seconds per frame (1 / fps).
    """
    anim, names, fps = _scene_to_animation(str(filepath), collapse_root=collapse_root)

    frametime = 1.0 / fps if fps > 0 else (1.0 / 30.0)

    if start is not None or end is not None:
        anim = anim[start:end]

    return anim, names, frametime
