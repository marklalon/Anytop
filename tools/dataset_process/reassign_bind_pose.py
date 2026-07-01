"""
reassign_bind_pose.py

Reassign every GLB in a directory to use either one animation frame or the
bind/rest pose from another GLB as the armature bind/rest pose, then re-export
the files.

This is useful for Truebones characters whose exported GLB bind pose is not the
standing/symmetric pose expected by downstream preprocessing.  The script uses
Blender's "Apply Pose as Rest Pose" operation, then re-bakes the original
animation channels against the new rest pose so the motion's total local joint
transforms are preserved while the bind/rest pose changes.

Requires bpy (Blender as a Python module) -- run with the project's .venv:

    .venv/Scripts/python.exe Anytop/tools/dataset_process/reassign_bind_pose.py \
        --dir Anytop/dataset/truebones/zoo/Truebone_Z-OO/Camel \
        --bind-pose Anytop/dataset/truebones/zoo/Truebone_Z-OO/Camel/Camel-IdleLoop.glb \
        --frame 0 \
        --overwrite
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Put the Anytop package root on sys.path so data_loaders / motion_lib / utils
# imports resolve to the Anytop copies (NOT the top-level pcvg utils).
_ANYTOP_DIR = Path(__file__).resolve().parent.parent.parent
if str(_ANYTOP_DIR) not in sys.path:
    sys.path.insert(0, str(_ANYTOP_DIR))


MatrixRows = list[list[float]]
BoneTopology = tuple[list[str], list[int]]

_WORKER_REFERENCE_POSE_BASIS: dict[str, MatrixRows] | None = None
_WORKER_REFERENCE_BONE_NAMES: list[str] | None = None
_WORKER_REFERENCE_BONE_TOPOLOGY: BoneTopology | None = None
_WORKER_REFERENCE_FIRST_SAMPLE_TIME: float | None = None
_WORKER_REFERENCE_IS_REST_POSE: bool = False


def _list_glb_files(directory: str) -> list[str]:
    """Return sorted absolute paths of all ``.glb`` files in *directory*."""
    return sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.lower().endswith(".glb")
    )


def _confirm_yes_no(prompt: str) -> bool:
    reply = input(prompt).strip().lower()
    return reply in ("y", "yes")


def _is_tpose_glb(path: str) -> bool:
    """Return True when the filename stem contains a T-pose suffix/hint."""
    stem = Path(path).stem.lower()
    compact = stem.replace("-", "").replace("_", "").replace(" ", "")
    return "tpose" in compact


def _matrix_to_rows(matrix) -> MatrixRows:
    return [[float(matrix[row][col]) for col in range(4)] for row in range(4)]


def _is_identity_rows(rows: MatrixRows, tol: float = 1e-3) -> bool:
    """Return True when *rows* is (near-)equal to the 4x4 identity matrix.

    The tolerance is deliberately loose (matching the project's bind-pose
    validation tolerance) so glTF quantization noise in a near-identity static
    bind-pose action still counts as "no pose", while a real animation frame
    (basis deviations of order 0.1-1.0) does not.
    """
    for row in range(4):
        for col in range(4):
            expected = 1.0 if row == col else 0.0
            if abs(rows[row][col] - expected) > tol:
                return False
    return True


def _find_armature(bpy, path: str):
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError(f"No armature found in GLB: {path}")
    # Truebones files should only have one armature.  If helper armatures exist,
    # use the one with the largest bone count as the main character rig.
    return max(armatures, key=lambda obj: len(obj.data.bones))


def _extract_bone_topology(armature) -> BoneTopology:
    """Return bone names and parent indices in deterministic hierarchy order."""
    data_bones = list(armature.data.bones)
    ordered_bones = []
    seen: set[int] = set()

    def _append_preorder(bone) -> None:
        marker = id(bone)
        if marker in seen:
            return
        seen.add(marker)
        ordered_bones.append(bone)
        for child in bone.children:
            _append_preorder(child)

    for bone in data_bones:
        if bone.parent is None:
            _append_preorder(bone)
    for bone in data_bones:
        _append_preorder(bone)

    names = [bone.name for bone in ordered_bones]
    name_to_index = {name: index for index, name in enumerate(names)}
    parents = [
        name_to_index.get(bone.parent.name, -1) if bone.parent is not None else -1
        for bone in ordered_bones
    ]
    return names, parents


def _build_topology_rename_pairs(
    armature,
    reference_topology: BoneTopology,
) -> list[tuple[str, str]]:
    """Build source-to-reference bone-name pairs after exact topology checking."""
    source_names, source_parents = _extract_bone_topology(armature)
    reference_names, reference_parents = reference_topology

    if len(source_names) != len(reference_names):
        raise RuntimeError(
            "cannot fix bone names; topology differs from reference "
            f"(source bones={len(source_names)}, reference bones={len(reference_names)})"
        )
    if source_parents != reference_parents:
        raise RuntimeError(
            "cannot fix bone names; topology differs from reference "
            "(parent hierarchy is not identical)"
        )

    return list(zip(source_names, reference_names))


def _unique_temp_name(existing_names: set[str], prefix: str, index: int) -> str:
    candidate = f"{prefix}_{index}__"
    suffix = 0
    while candidate in existing_names:
        suffix += 1
        candidate = f"{prefix}_{index}_{suffix}__"
    existing_names.add(candidate)
    return candidate


def _rename_animation_data_paths(bpy, rename_map: dict[str, str]) -> None:
    """Best-effort update for existing pose-bone FCurve paths."""
    if not rename_map:
        return

    rename_items = list(rename_map.items())
    for action in getattr(bpy.data, "actions", []):
        for fcurve in getattr(action, "fcurves", []):
            data_path = getattr(fcurve, "data_path", "")
            if not data_path:
                continue
            for index, (old_name, _new_name) in enumerate(rename_items):
                temp_name = f"__fix_bone_names_tmp_path_{index}__"
                data_path = data_path.replace(
                    f'pose.bones["{old_name}"]',
                    f'pose.bones["{temp_name}"]',
                )
                data_path = data_path.replace(
                    f"pose.bones['{old_name}']",
                    f"pose.bones['{temp_name}']",
                )
            for index, (_old_name, new_name) in enumerate(rename_items):
                temp_name = f"__fix_bone_names_tmp_path_{index}__"
                data_path = data_path.replace(
                    f'pose.bones["{temp_name}"]',
                    f'pose.bones["{new_name}"]',
                )
                data_path = data_path.replace(
                    f"pose.bones['{temp_name}']",
                    f"pose.bones['{new_name}']",
                )
            fcurve.data_path = data_path


def _rename_armature_bones_to_reference(
    bpy,
    armature,
    name_pairs: list[tuple[str, str]],
) -> dict[str, str]:
    """Rename bones and matching skin vertex groups to reference names."""
    rename = [(old, new) for old, new in name_pairs if old != new]
    if not rename:
        return {}
    rename_map = dict(rename)

    data_bones = armature.data.bones
    existing_bone_names = {bone.name for bone in data_bones}
    pending_bones: list[tuple[str, str]] = []
    for index, (old_name, new_name) in enumerate(rename):
        bone = data_bones.get(old_name)
        if bone is None:
            continue
        existing_bone_names.discard(old_name)
        temp_name = _unique_temp_name(
            existing_bone_names,
            "__fix_bone_names_tmp_bone",
            index,
        )
        bone.name = temp_name
        pending_bones.append((temp_name, new_name))
    for temp_name, new_name in pending_bones:
        bone = data_bones.get(temp_name)
        if bone is not None:
            bone.name = new_name

    for obj in bpy.data.objects:
        if obj.parent == armature and obj.parent_type == "BONE":
            new_parent_bone = rename_map.get(obj.parent_bone)
            if new_parent_bone is not None:
                world_matrix = obj.matrix_world.copy()
                obj.parent_bone = new_parent_bone
                obj.matrix_world = world_matrix

    for mesh in _armature_bound_meshes(bpy, armature):
        vertex_groups = mesh.vertex_groups
        existing_group_names = {group.name for group in vertex_groups}
        pending_groups: list[tuple[str, str]] = []
        for index, (old_name, new_name) in enumerate(rename):
            group = vertex_groups.get(old_name)
            if group is None:
                continue
            existing_group_names.discard(old_name)
            temp_name = _unique_temp_name(
                existing_group_names,
                "__fix_bone_names_tmp_group",
                index,
            )
            group.name = temp_name
            pending_groups.append((temp_name, new_name))
        for temp_name, new_name in pending_groups:
            group = vertex_groups.get(temp_name)
            if group is not None:
                group.name = new_name

    _rename_animation_data_paths(bpy, rename_map)
    return rename_map


def _load_reference_pose(
    bind_pose_glb: str,
    frame_index: int = 0,
) -> tuple[dict[str, MatrixRows], list[str], BoneTopology, float, bool]:
    """Load *bind_pose_glb* and return the requested reference pose matrices.

    ``frame_index >= 0`` reads a sampled animation frame and returns pose-bone
    ``matrix_basis`` values. ``frame_index == -1`` reads the GLB's bind/rest
    pose and returns armature-space bone ``matrix_local`` values.
    """
    import bpy
    from motion_lib.FBX import (
        _silence_os_std,
        clear_scene,
        get_action_sample_times,
        import_gltf,
        remove_lights_and_cameras,
        set_scene_time,
    )

    clear_scene()
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()), \
         _silence_os_std():
        import_gltf(bind_pose_glb)
    remove_lights_and_cameras()

    armature = _find_armature(bpy, bind_pose_glb)
    bone_topology = _extract_bone_topology(armature)
    scene = bpy.context.scene

    def _read_rest_pose() -> tuple[dict[str, MatrixRows], list[str], BoneTopology, float, bool]:
        bone_names = [pose_bone.name for pose_bone in armature.pose.bones]
        data_bones = armature.data.bones
        rest_pose = {
            bone_name: _matrix_to_rows(data_bones.get(bone_name).matrix_local)
            for bone_name in bone_names
            if data_bones.get(bone_name) is not None
        }
        clear_scene()
        return rest_pose, bone_names, bone_topology, -1.0, True

    if frame_index < -1:
        raise ValueError(f"--frame must be -1 or >= 0, got {frame_index}")
    if frame_index == -1:
        return _read_rest_pose()

    sample_times = get_action_sample_times(armature)
    if frame_index >= len(sample_times):
        frame_index = len(sample_times) - 1
    sample_time = float(sample_times[frame_index])
    set_scene_time(scene, sample_time)
    bpy.context.view_layer.update()

    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="POSE")

    bone_names = [pose_bone.name for pose_bone in armature.pose.bones]
    pose_basis = {
        pose_bone.name: _matrix_to_rows(pose_bone.matrix_basis)
        for pose_bone in armature.pose.bones
    }

    bpy.ops.object.mode_set(mode="OBJECT")

    # The sampled ``matrix_basis`` is the pose *relative to the rest pose*.  When
    # every bone's basis is (near-)identity -- e.g. a rest-pose-only TPOSE export
    # (whose ``get_action_sample_times`` still yields ``[0.0]``), or a GLB that
    # only carries a static identity bind-pose action -- applying it would be a
    # no-op that keeps each source file's own rest pose.  Fall back to the
    # reference's actual bind/rest pose so the reassignment really retargets.
    if all(_is_identity_rows(rows) for rows in pose_basis.values()):
        return _read_rest_pose()

    clear_scene()
    return pose_basis, bone_names, bone_topology, sample_time, False


def _init_worker_reference_pose(bind_pose_glb: str, frame_index: int = 0) -> None:
    """Load the reference pose once inside each worker process."""
    global _WORKER_REFERENCE_POSE_BASIS
    global _WORKER_REFERENCE_BONE_NAMES
    global _WORKER_REFERENCE_BONE_TOPOLOGY
    global _WORKER_REFERENCE_FIRST_SAMPLE_TIME
    global _WORKER_REFERENCE_IS_REST_POSE

    (
        _WORKER_REFERENCE_POSE_BASIS,
        _WORKER_REFERENCE_BONE_NAMES,
        _WORKER_REFERENCE_BONE_TOPOLOGY,
        _WORKER_REFERENCE_FIRST_SAMPLE_TIME,
        _WORKER_REFERENCE_IS_REST_POSE,
    ) = _load_reference_pose(bind_pose_glb, frame_index=frame_index)


def _export_glb_safely(bpy, output_path: str, *, in_place: bool) -> None:
    """Export the current scene to *output_path*, using a temp file for in-place writes."""
    from motion_lib.FBX import _silence_os_std

    output_abs = os.path.abspath(output_path)
    export_path = output_abs
    temp_path = None
    if in_place:
        fd, temp_path = tempfile.mkstemp(
            prefix=f".{Path(output_abs).stem}.",
            suffix=".tmp.glb",
            dir=str(Path(output_abs).parent),
        )
        os.close(fd)
        export_path = temp_path

    try:
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()), \
             _silence_os_std():
            bpy.ops.export_scene.gltf(
                filepath=export_path,
                export_format="GLB",
                export_animations=True,
                export_animation_mode="ACTIVE_ACTIONS",
                export_force_sampling=True,
                export_frame_range=True,
                export_apply=False,
            )
        if temp_path is not None:
            os.replace(temp_path, output_abs)
            temp_path = None
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.remove(temp_path)


def _armature_bound_meshes(bpy, armature) -> list:
    """Return meshes skinned to or parented under *armature*."""
    meshes = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if obj.parent == armature or any(
            mod.type == "ARMATURE" and mod.object == armature
            for mod in obj.modifiers
        ):
            meshes.append(obj)
    return meshes


def _bake_current_deformed_meshes_to_bind_data(bpy, armature) -> None:
    """Write each bound mesh's current evaluated deformation into mesh.data.

    ``pose.armature_apply`` changes the armature rest pose, but it does not
    make the mesh's object-space vertices become the current posed/bind shape.
    For a real bind-pose reassignment, the mesh data must be rebased too;
    otherwise the exported skin has the new rest bones with the old bind mesh.
    """
    from mathutils import Vector

    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for mesh_obj in _armature_bound_meshes(bpy, armature):
        eval_obj = mesh_obj.evaluated_get(depsgraph)
        eval_mesh = eval_obj.to_mesh()
        try:
            if len(eval_mesh.vertices) != len(mesh_obj.data.vertices):
                raise RuntimeError(
                    f"{mesh_obj.name}: evaluated vertex count "
                    f"{len(eval_mesh.vertices)} differs from source "
                    f"{len(mesh_obj.data.vertices)}"
                )
            inv_world = mesh_obj.matrix_world.inverted_safe()
            new_coords = [
                inv_world @ (eval_obj.matrix_world @ vert.co)
                for vert in eval_mesh.vertices
            ]
        finally:
            eval_obj.to_mesh_clear()

        for vertex, co in zip(mesh_obj.data.vertices, new_coords):
            vertex.co = co

        # If a mesh has shape keys, keep the Basis key in sync with the rebased
        # mesh data.  Truebones character meshes usually do not have shape keys,
        # but this avoids an invisible stale basis if one appears.
        shape_keys = mesh_obj.data.shape_keys
        if shape_keys is not None and shape_keys.key_blocks:
            basis = shape_keys.key_blocks[0]
            for point, co in zip(basis.data, new_coords):
                point.co = Vector(co)

        mesh_obj.data.update()


def _rebake_action_against_new_rest(
    bpy,
    armature,
    original_anim,
    original_names: list[str],
    original_frametime: float,
) -> None:
    """Rebuild pose-bone animation so total local transforms stay unchanged."""
    import numpy as np
    from mathutils import Quaternion
    from motion_lib.FBX import extract_armature_skeleton_data, iter_action_fcurves
    from utils.rotation_numpy import (
        quat_conjugate_wxyz_np,
        quat_multiply_wxyz_np,
        quat_rotate_wxyz_np,
    )

    new_names, _new_parents, new_offsets, new_rest_rotations = (
        extract_armature_skeleton_data(armature)
    )
    original_name_to_idx = {name: idx for idx, name in enumerate(original_names)}
    missing = [name for name in new_names if name not in original_name_to_idx]
    if missing:
        preview = ", ".join(missing[:8])
        raise RuntimeError(f"cannot rebake animation; missing original bones: {preview}")

    source_indices = np.asarray(
        [original_name_to_idx[name] for name in new_names],
        dtype=np.int64,
    )
    total_rotations = np.asarray(original_anim.rotations.qs, dtype=np.float64)[:, source_indices, :]
    total_positions = np.asarray(original_anim.positions, dtype=np.float64)[:, source_indices, :]
    frame_count = int(total_rotations.shape[0])
    if frame_count == 0:
        return

    rest_rot = np.asarray(new_rest_rotations, dtype=np.float64)
    rest_off = np.asarray(new_offsets, dtype=np.float64)
    rest_rot_inv = quat_conjugate_wxyz_np(
        np.repeat(rest_rot[None, :, :], frame_count, axis=0)
    )
    pose_rotations = quat_multiply_wxyz_np(rest_rot_inv, total_rotations)
    pose_locations = quat_rotate_wxyz_np(
        rest_rot_inv,
        total_positions - rest_off[None, :, :],
    )

    if armature.animation_data:
        armature.animation_data_clear()
    armature.animation_data_create()
    action = bpy.data.actions.new(name="RebasedBindPoseAnimation")
    armature.animation_data.action = action

    fps = 1.0 / original_frametime if original_frametime and original_frametime > 0 else 30.0
    scene = bpy.context.scene
    scene.render.fps = int(round(fps))
    scene.render.fps_base = 1.0
    scene.frame_start = 0
    scene.frame_end = max(frame_count - 1, 0)

    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="POSE")

    for frame_idx in range(frame_count):
        scene.frame_set(frame_idx)
        for joint_idx, bone_name in enumerate(new_names):
            pose_bone = armature.pose.bones.get(bone_name)
            if pose_bone is None:
                continue
            pose_bone.rotation_mode = "QUATERNION"
            pose_bone.location = tuple(float(v) for v in pose_locations[frame_idx, joint_idx])
            q = pose_rotations[frame_idx, joint_idx]
            q_norm = float(np.linalg.norm(q))
            if q_norm > 1e-8:
                quat = Quaternion(tuple(float(v / q_norm) for v in q))
            else:
                quat = Quaternion((1.0, 0.0, 0.0, 0.0))
            pose_bone.rotation_quaternion = quat
            pose_bone.scale = (1.0, 1.0, 1.0)
            pose_bone.keyframe_insert(data_path="location", frame=frame_idx)
            pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx)
            pose_bone.keyframe_insert(data_path="scale", frame=frame_idx)

    for fcurve in iter_action_fcurves(action):
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = "LINEAR"
        fcurve.update()

    bpy.ops.object.mode_set(mode="OBJECT")


def _write_static_bind_pose_action(
    bpy,
    armature,
    fps: float = 30.0,
) -> None:
    """Create a one-frame identity pose action, which samples as the bind pose."""
    from motion_lib.FBX import iter_action_fcurves

    if armature.animation_data:
        armature.animation_data_clear()
    armature.animation_data_create()
    action = bpy.data.actions.new(name="StaticBindPoseAnimation")
    armature.animation_data.action = action

    scene = bpy.context.scene
    scene.render.fps = int(round(fps)) if fps and fps > 0 else 30
    scene.render.fps_base = 1.0
    scene.frame_start = 0
    scene.frame_end = 0
    scene.frame_set(0)

    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="POSE")

    for pose_bone in armature.pose.bones:
        pose_bone.rotation_mode = "QUATERNION"
        pose_bone.location = (0.0, 0.0, 0.0)
        pose_bone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        pose_bone.scale = (1.0, 1.0, 1.0)
        pose_bone.keyframe_insert(data_path="location", frame=0)
        pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=0)
        pose_bone.keyframe_insert(data_path="scale", frame=0)

    for fcurve in iter_action_fcurves(action):
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = "LINEAR"
        fcurve.update()

    bpy.ops.object.mode_set(mode="OBJECT")


def _apply_reference_rest_pose_to_armature(armature, reference_rest_pose: dict[str, MatrixRows]) -> None:
    """Set pose-bone bases so the final pose equals *reference_rest_pose*."""
    from mathutils import Matrix

    data_bones = armature.data.bones
    desired = {
        name: Matrix(rows)
        for name, rows in reference_rest_pose.items()
        if data_bones.get(name) is not None
    }
    for pose_bone in armature.pose.bones:
        target_rest_bone = data_bones.get(pose_bone.name)
        desired_matrix = desired.get(pose_bone.name)
        if target_rest_bone is None or desired_matrix is None:
            continue

        target_rest = target_rest_bone.matrix_local
        parent_bone = target_rest_bone.parent
        if parent_bone is not None and parent_bone.name in desired:
            parent_rest = parent_bone.matrix_local
            parent_desired = desired[parent_bone.name]
            basis = (
                target_rest.inverted_safe()
                @ parent_rest
                @ parent_desired.inverted_safe()
                @ desired_matrix
            )
        else:
            basis = target_rest.inverted_safe() @ desired_matrix
        pose_bone.matrix_basis = basis


def _reassign_one_glb(
    glb_path: str,
    output_path: str,
    ignore_missing_bones: bool,
    preserve_animation: bool,
    fix_bone_names: bool,
) -> tuple[str, str]:
    """Worker entry point.  Returns ``(basename, status)``."""
    import bpy
    from mathutils import Matrix
    from motion_lib import FBX
    from motion_lib.FBX import (
        _silence_os_std,
        clear_scene,
        import_gltf,
        remove_lights_and_cameras,
    )

    basename = os.path.basename(glb_path)
    try:
        if (
            _WORKER_REFERENCE_POSE_BASIS is None
            or _WORKER_REFERENCE_BONE_NAMES is None
            or _WORKER_REFERENCE_BONE_TOPOLOGY is None
        ):
            raise RuntimeError(
                "worker reference pose is not initialized; "
                "call _init_worker_reference_pose first"
            )

        force_static_bind_pose = _is_tpose_glb(glb_path)
        original_anim = original_names = original_frametime = None
        if preserve_animation and not force_static_bind_pose:
            original_anim, original_names, original_frametime = FBX.load(
                glb_path,
                collapse_root=False,
            )

        clear_scene()
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()), \
             _silence_os_std():
            import_gltf(glb_path)
        remove_lights_and_cameras()

        armature = _find_armature(bpy, glb_path)
        source_bone_names = [pose_bone.name for pose_bone in armature.pose.bones]
        source_set = set(source_bone_names)
        reference_set = set(_WORKER_REFERENCE_BONE_NAMES)
        rename_map: dict[str, str] = {}
        if fix_bone_names and source_set != reference_set:
            name_pairs = _build_topology_rename_pairs(
                armature,
                _WORKER_REFERENCE_BONE_TOPOLOGY,
            )
            rename_map = _rename_armature_bones_to_reference(
                bpy,
                armature,
                name_pairs,
            )
            if original_names is not None:
                original_names = [
                    rename_map.get(name, name)
                    for name in original_names
                ]
            source_bone_names = [pose_bone.name for pose_bone in armature.pose.bones]
            source_set = set(source_bone_names)

        missing_in_source = sorted(reference_set - source_set)
        missing_in_reference = sorted(source_set - reference_set)
        if (missing_in_source or missing_in_reference) and not ignore_missing_bones:
            details = []
            if missing_in_source:
                preview = ", ".join(missing_in_source[:8])
                details.append(f"missing source bones: {preview}")
            if missing_in_reference:
                preview = ", ".join(missing_in_reference[:8])
                details.append(f"missing reference bones: {preview}")
            raise RuntimeError("; ".join(details))

        bpy.context.view_layer.objects.active = armature
        armature.select_set(True)
        armature.data.pose_position = "POSE"
        bpy.ops.object.mode_set(mode="POSE")

        if _WORKER_REFERENCE_IS_REST_POSE:
            _apply_reference_rest_pose_to_armature(
                armature,
                _WORKER_REFERENCE_POSE_BASIS,
            )
        else:
            for pose_bone in armature.pose.bones:
                rows = _WORKER_REFERENCE_POSE_BASIS.get(pose_bone.name)
                if rows is None:
                    continue
                pose_bone.matrix_basis = Matrix(rows)

        bpy.context.view_layer.update()
        _bake_current_deformed_meshes_to_bind_data(bpy, armature)
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()), \
             _silence_os_std():
            bpy.ops.pose.armature_apply(selected=False)
        bpy.ops.object.mode_set(mode="OBJECT")

        if force_static_bind_pose:
            _write_static_bind_pose_action(bpy, armature)
        elif preserve_animation:
            _rebake_action_against_new_rest(
                bpy,
                armature,
                original_anim,
                original_names,
                original_frametime,
            )

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        _export_glb_safely(
            bpy,
            output_path,
            in_place=os.path.abspath(glb_path) == os.path.abspath(output_path),
        )
        clear_scene()
        return basename, "reassigned"
    except Exception as exc:
        try:
            clear_scene()
        except Exception:
            pass
        return basename, f"failed: {exc}"


def reassign_bind_pose_directory(
    directory: str,
    bind_pose_glb: str,
    output_dir: str | None = None,
    overwrite: str = "prompt",
    workers: int = 16,
    ignore_missing_bones: bool = False,
    preserve_animation: bool = True,
    fix_bone_names: bool = False,
    frame_index: int = 0,
) -> int:
    """Reassign GLB bind poses in *directory* from one frame of *bind_pose_glb*.

    Args:
        directory: Folder containing the GLB files to process.
        bind_pose_glb: Reference GLB whose selected animation frame becomes
            the new bind/rest pose.
        output_dir: Where to write the GLBs. Defaults to *directory* (in-place,
            same-named output).
        overwrite: "prompt" (ask once when outputs exist), "force" (always),
            or "skip".
        workers: Number of Blender worker processes.
        ignore_missing_bones: If true, apply matching bones and ignore name
            mismatches.  Defaults to strict one-to-one bone-name checking.
        preserve_animation: Re-bake animation channels after changing the rest
            pose so total local joint transforms stay unchanged.
        fix_bone_names: If true and bone names differ from the reference,
            require identical topology and rename source bones/skin groups to
            the reference names before applying the bind pose.
        frame_index: Zero-based sampled frame index from *bind_pose_glb* to use
            as the new bind/rest pose.  Use -1 to use *bind_pose_glb*'s own
            bind/rest pose instead of an animation frame.

    Returns:
        Number of GLB files successfully written.
    """
    directory_abs = os.path.abspath(directory)
    bind_pose_abs = os.path.abspath(bind_pose_glb)
    output_dir_abs = os.path.abspath(output_dir or directory)

    if not os.path.isdir(directory_abs):
        raise NotADirectoryError(f"Input directory not found: {directory_abs}")
    if not os.path.isfile(bind_pose_abs):
        raise FileNotFoundError(f"--bind-pose GLB not found: {bind_pose_abs}")
    if frame_index < -1:
        raise ValueError(f"--frame must be -1 or >= 0, got {frame_index}")
    os.makedirs(output_dir_abs, exist_ok=True)

    glb_files = _list_glb_files(directory_abs)
    if not glb_files:
        raise FileNotFoundError(f"No .glb files found in {directory_abs}")

    print(f"Input dir      : {directory_abs}")
    print(f"Bind-pose GLB  : {bind_pose_abs}")
    print(f"Bind-pose frame: {frame_index}")
    print(f"Output dir     : {output_dir_abs}")
    print(f"GLB files      : {len(glb_files)}")
    print(f"Fix bone names : {fix_bone_names}")

    all_tasks = [
        (glb_path, os.path.join(output_dir_abs, os.path.basename(glb_path)))
        for glb_path in glb_files
    ]
    existing_tasks = [
        (glb_path, output_path)
        for glb_path, output_path in all_tasks
        if os.path.exists(output_path)
    ]

    if overwrite == "prompt" and existing_tasks:
        preview = ", ".join(
            os.path.basename(path) for _glb_path, path in existing_tasks[:5]
        )
        if len(existing_tasks) > 5:
            preview += ", ..."
        print(f"Existing output: {len(existing_tasks)} file(s) ({preview})")
        if _confirm_yes_no(f"Overwrite existing output GLB file(s)? [y/N] "):
            tasks = all_tasks
        else:
            tasks = [
                (glb_path, output_path)
                for glb_path, output_path in all_tasks
                if not os.path.exists(output_path)
            ]
            print(f"[skip] {len(existing_tasks)} existing GLB file(s)")
    else:
        tasks = all_tasks

    if not tasks:
        print("\nDone. No GLB files were written.")
        return 0

    written = 0
    failed = 0
    n_workers = max(1, min(int(workers), len(tasks)))
    print(f"\nStarting bind-pose reassignment with {n_workers} worker(s) ...\n")

    if n_workers == 1:
        print("Loading reference pose ...")
        _init_worker_reference_pose(bind_pose_abs, frame_index)
        if _WORKER_REFERENCE_IS_REST_POSE:
            print("Reference source: bind/rest pose")
        else:
            print(f"Reference sample time: {_WORKER_REFERENCE_FIRST_SAMPLE_TIME:g}")
        print(f"Reference bones: {len(_WORKER_REFERENCE_BONE_NAMES or [])}\n")

        for index, (glb_path, output_path) in enumerate(tasks, start=1):
            basename, status = _reassign_one_glb(
                glb_path,
                output_path,
                ignore_missing_bones,
                preserve_animation,
                fix_bone_names,
            )
            if status == "reassigned":
                written += 1
                print(f"[{index}/{len(tasks)}] [OK]   {basename}")
            else:
                failed += 1
                print(f"[{index}/{len(tasks)}] [FAIL] {basename}: {status}")
    else:
        print("Reference pose will be loaded once inside each worker process.\n")
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker_reference_pose,
            initargs=(bind_pose_abs, frame_index),
        ) as executor:
            future_to_item = {
                executor.submit(
                    _reassign_one_glb,
                    glb_path,
                    output_path,
                    ignore_missing_bones,
                    preserve_animation,
                    fix_bone_names,
                ): (glb_path, output_path)
                for glb_path, output_path in tasks
            }
            completed = 0
            for future in as_completed(future_to_item):
                completed += 1
                glb_path, _output_path = future_to_item[future]
                try:
                    basename, status = future.result()
                except Exception as exc:
                    basename = os.path.basename(glb_path)
                    status = f"failed: {exc}"
                if status == "reassigned":
                    written += 1
                    print(f"[{completed}/{len(tasks)}] [OK]   {basename}")
                else:
                    failed += 1
                    print(f"[{completed}/{len(tasks)}] [FAIL] {basename}: {status}")

    print(f"\nDone. Wrote {written} GLB file(s) to {output_dir_abs}")
    if failed:
        print(f"Failed: {failed}")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reassign all GLB bind poses in a directory to a selected animation "
            "frame, or the bind/rest pose, of a reference GLB."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="Directory containing GLB files to process.",
    )
    parser.add_argument(
        "--bind-pose",
        required=True,
        help=(
            "Reference GLB. One sampled animation frame, or its bind/rest pose "
            "with --frame -1, is applied as the new bind pose."
        ),
    )
    parser.add_argument(
        "--frame",
        default=0,
        type=int,
        help=(
            "Zero-based sampled frame index from --bind-pose to use. "
            "Use -1 to use the GLB's bind/rest pose. Default 0."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write GLBs. Defaults to --dir (in-place, same-named output).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing GLB files without prompting.",
    )
    parser.add_argument(
        "--workers",
        "-j",
        default=16,
        type=int,
        help="Number of parallel Blender worker processes. Default 16.",
    )
    parser.add_argument(
        "--ignore-missing-bones",
        action="store_true",
        help="Apply matching bones and ignore source/reference bone-name mismatches.",
    )
    parser.add_argument(
        "--fix-bone-names",
        action="store_true",
        help=(
            "When source bone names differ from --bind-pose, require identical "
            "bone topology and rename source bones/skin groups to reference names."
        ),
    )
    parser.add_argument(
        "--no-preserve-animation",
        action="store_true",
        help=(
            "Only apply the new rest pose and do not re-bake animation channels. "
            "By default, animations are re-baked to preserve total joint motion."
        ),
    )
    args = parser.parse_args()

    overwrite = "force" if args.overwrite else "prompt"

    try:
        reassign_bind_pose_directory(
            directory=args.dir,
            bind_pose_glb=args.bind_pose,
            output_dir=args.output_dir,
            overwrite=overwrite,
            workers=args.workers,
            ignore_missing_bones=args.ignore_missing_bones,
            preserve_animation=not args.no_preserve_animation,
            fix_bone_names=args.fix_bone_names,
            frame_index=args.frame,
        )
        return 0
    except Exception as exc:
        print(f"ERROR: bind-pose reassignment failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
