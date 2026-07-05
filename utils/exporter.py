"""
Animation export to GLB and BVH motion formats.

GLB export requires Blender (bpy) to be available in the Python
environment. BVH export is written through Anytop motion_lib so the
round-trip path stays consistent with motion_lib.BVH.load.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from motion_lib.FBX import extract_armature_skeleton_data, import_fbx, remove_lights_and_cameras
from data_loaders.truebones.truebones_utils.animation_utils import refresh_joint_metadata_in_object_cond
from .rotation_numpy import (
    quat_conjugate_wxyz_np,
    quat_multiply_wxyz_np,
    quat_rotate_wxyz_np,
)
from .retarget import (
    _batch_internal_pose_fk_np,
    retarget_world_space_np,
)
from .texture_resolve import resolve_main_character_textures


__all__ = [
    "animation_to_exporter_inputs",
    "AnimationExporter",
]


def _build_canonical_name_variants(
    joint_names: list[str],
    parents: np.ndarray,
    offsets: np.ndarray,
    *,
    log_hint: str,
) -> tuple[list[str], list[str]]:
    """Derive canonical joint names from raw skeleton data.

    Returns ``(match_names, bvh_names)``:

    * ``match_names`` — ``canonical_joint_names`` (e.g. ``"Tail 01"``), the
      semantic names used for src↔tgt retarget matching.
    * ``bvh_names`` — ``canonical_bvh_joint_names`` (e.g. ``"Tail01"``), the
      BVH-compatible variant used to name the exported GLB's bones so the GLB
      and the processed BVH share one joint-name set.

    *log_hint* is only used in error messages to identify which skeleton
    failed to canonicalize; it does not affect the matching logic.
    """
    object_cond = {
        "object_type": log_hint,
        "joints_names": list(joint_names),
        "parents": np.asarray(parents, dtype=np.int32),
        "offsets": np.asarray(offsets, dtype=np.float64),
    }
    refresh_joint_metadata_in_object_cond(object_cond)
    match_names = object_cond.get("canonical_joint_names")
    bvh_names = object_cond.get("canonical_bvh_joint_names")
    if match_names is None or bvh_names is None:
        raise ValueError(
            f"Unable to derive canonical joint names ({log_hint})"
        )
    match_names = list(match_names)
    bvh_names = list(bvh_names)
    if len(match_names) != len(joint_names) or len(bvh_names) != len(joint_names):
        raise ValueError(
            f"Derived canonical joint name length mismatch ({log_hint}): "
            f"match={len(match_names)} bvh={len(bvh_names)} vs {len(joint_names)}"
        )
    return match_names, bvh_names


def _build_canonical_match_names(
    joint_names: list[str],
    parents: np.ndarray,
    offsets: np.ndarray,
    *,
    log_hint: str,
) -> list[str]:
    """Return the semantic canonical match names (``canonical_joint_names``)."""
    return _build_canonical_name_variants(
        joint_names, parents, offsets, log_hint=log_hint
    )[0]


def animation_to_exporter_inputs(animation, skeleton) -> tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
    """Convert motion_lib Animation local transforms into exporter inputs.

    ``BVH.load`` and ``FBX.load`` both return per-joint total local transforms:
        local_rot_total = animation.rotations
        local_pos_total = animation.positions

    ``AnimationExporter`` expects those same transforms split into:
      - ``joint_rotations``: animated local rotation after removing each joint's
        static ``rest_rotation``
      - ``root_translation``: extra world-space wrapper translation applied
        before the root joint's static rest offset
      - ``bone_translations``: Blender pose-bone ``location`` channels for
        non-root joints, expressed in the joint's rest-rotated local basis
    """
    joint_count = len(skeleton.bones)
    if animation.shape[1] != joint_count:
        raise ValueError(
            f"Animation joint count {animation.shape[1]} does not match skeleton joint count {joint_count}"
        )

    total_rotations_np = np.asarray(animation.rotations.qs, dtype=np.float64)
    total_positions_np = np.asarray(animation.positions, dtype=np.float64)
    frame_count = total_rotations_np.shape[0]

    rest_offsets_np = np.empty((joint_count, 3), dtype=np.float64)
    rest_rotations_np = np.empty((joint_count, 4), dtype=np.float64)
    for bone in skeleton.bones:
        rest_offsets_np[bone.id] = bone.rest_offset.detach().cpu().to(torch.float64).numpy()
        rest_rotations_np[bone.id] = bone.rest_rotation.detach().cpu().to(torch.float64).numpy()

    pose_rotations_np = np.empty_like(total_rotations_np)
    pose_translations_np = np.zeros_like(total_positions_np)
    for joint_idx in range(joint_count):
        rest_q = np.repeat(rest_rotations_np[joint_idx:joint_idx + 1], frame_count, axis=0)
        rest_q_conj = quat_conjugate_wxyz_np(rest_q)
        pose_rotations_np[:, joint_idx, :] = quat_multiply_wxyz_np(
            rest_q_conj,
            total_rotations_np[:, joint_idx, :],
        )
        if joint_idx == 0:
            continue
        pose_translations_np[:, joint_idx, :] = quat_rotate_wxyz_np(
            rest_q_conj,
            total_positions_np[:, joint_idx, :] - rest_offsets_np[joint_idx:joint_idx + 1, :],
        )

    root_translation_np = total_positions_np[:, 0, :] - rest_offsets_np[0:1, :]
    root_rotation_np = np.zeros((frame_count, 4), dtype=np.float32)
    root_rotation_np[:, 0] = 1.0

    joint_rotations = torch.from_numpy(pose_rotations_np.astype(np.float32, copy=False))
    root_translation = torch.from_numpy(root_translation_np.astype(np.float32, copy=False))
    root_rotation = torch.from_numpy(root_rotation_np)

    if joint_count > 1 and np.any(np.abs(pose_translations_np[:, 1:, :]) > 1e-6):
        bone_translations = torch.from_numpy(pose_translations_np.astype(np.float32, copy=False))
    else:
        bone_translations = None

    return joint_rotations, root_translation, root_rotation, bone_translations


def _normalize_imported_armature_and_meshes(bpy, armature) -> None:
    """Normalize imported mesh-source wrappers while preserving the axis wrapper.

    FBX import commonly leaves a +90deg X axis wrapper and a 0.01 object scale
    on the armature object. GLB/GLTF T-pose assets can also carry a stale
    object-level translation (for example a character parked half a metre above
    the origin). We want to remove those object-level wrappers and stale
    parent-inverse state, but we must keep the axis wrapper rotation: the
    imported bone and mesh local data still live in the FBX armature's Y-up
    object space, so stripping the rotation would roll the character 90 degrees.

    Dropping the armature's object scale re-interprets its armature-local bone
    data (centimetre units) as world units, which is what aligns the exported
    skeleton with the centimetre-scale NPY animation. Dropping the armature's
    object translation is equally important for mesh-source restores: the
    restored motion should determine world placement, not an arbitrary static
    offset baked into the T-pose asset. The skinned meshes must move with those
    wrapper removals so they stay glued to the skeleton — but they do NOT
    necessarily share the armature's object scale: a mesh parented *under* the
    armature inherits an extra scale factor (e.g. armature scale 0.01 + mesh
    basis 0.01 -> mesh world scale 0.0001). Resetting every object independently
    therefore detaches such a mesh.

    Instead, compute the world-space delta that maps the armature's old world
    frame onto its normalized (scale-dropped) frame, then apply that *same*
    delta to every bound mesh. This preserves each mesh's transform relative to
    the armature exactly, so the skin stays aligned regardless of how the mesh's
    own object scale differs from the armature's.
    """
    from mathutils import Matrix

    related_meshes = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if obj.parent == armature or any(
            mod.type == "ARMATURE" and mod.object == armature for mod in obj.modifiers
        ):
            related_meshes.append(obj)

    arm_world_old = armature.matrix_world.copy()
    # Normalized armature frame: keep only the world rotation wrapper. Any
    # object-level translation/scale on a mesh-source armature is static asset
    # placement, not part of the recovered motion.
    arm_rotation = arm_world_old.to_quaternion().to_matrix().to_4x4()
    arm_world_new = arm_rotation

    # World-space delta carrying the old armature frame onto the normalized one.
    # Applying it to a bound mesh preserves arm_world_new.inverted() @ mesh_world
    # == arm_world_old.inverted() @ mesh_world_old, i.e. the mesh-in-armature
    # bind transform the armature modifier relies on.
    delta = arm_world_new @ arm_world_old.inverted()

    mesh_worlds_old = {m.name: m.matrix_world.copy() for m in related_meshes}

    armature.matrix_parent_inverse = Matrix.Identity(4)
    armature.matrix_world = arm_world_new

    for mesh in related_meshes:
        mesh.matrix_parent_inverse = Matrix.Identity(4)
        mesh.matrix_world = delta @ mesh_worlds_old[mesh.name]


def _apply_gltf_output_space_similarity(
    bpy,
    armature,
    scale_factor: float | None,
    orientation_quat_wxyz: Optional[np.ndarray],
) -> None:
    """Reverse-align an imported rig into HML/npy space (skinned restore mode 2).

    The skinned-GLB output lives in glTF output space (Y-up), which is the same
    space the source FBX/GLB rig occupies after export. The preprocessing that
    produced the NPY mapped that raw rig space into HML space with a single
    global similarity ``G = Scale(scale_factor) · Rot(orientation_quat)`` (the
    XZ-centering is a per-clip root translation already carried by the recovered
    animation, and the descendant-Y bake is present in both spaces). Applying
    ``G`` here therefore reverse-aligns the *rig* onto the NPY/HML frame instead
    of aligning the animation onto the rig, so the exported GLB keeps the NPY's
    orientation, scale, and centered placement (like the corresponding BVH).

    ``G`` is expressed in glTF output space; the armature object lives in
    Blender's Z-up space, so it is conjugated by the ``export_yup`` axis change
    ``C`` (Blender Z-up -> glTF Y-up): ``D_blender = C⁻¹ · G · C``. The same
    rigid + uniform-scale delta is applied to the armature and every bound mesh,
    which preserves each mesh's armature-relative bind transform exactly.
    """
    from mathutils import Matrix, Quaternion

    # Blender (Z-up) -> glTF (Y-up): (x, y, z) -> (x, z, -y). Matches export_yup=True.
    C = Matrix((
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ))

    s = float(scale_factor) if scale_factor else 1.0
    if s <= 0.0:
        raise ValueError(f"scale_factor must be positive, got {s}")
    G = Matrix.Scale(s, 4)
    if orientation_quat_wxyz is not None:
        q = np.asarray(orientation_quat_wxyz, dtype=np.float64).reshape(-1)
        if q.shape != (4,):
            raise ValueError(f"orientation_quat must have shape (4,), got {q.shape}")
        rotation = Quaternion((float(q[0]), float(q[1]), float(q[2]), float(q[3]))).to_matrix().to_4x4()
        G = G @ rotation

    if s == 1.0 and (orientation_quat_wxyz is None):
        return

    delta = C.inverted() @ G @ C

    related_meshes = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if obj.parent == armature or any(
            mod.type == "ARMATURE" and mod.object == armature for mod in obj.modifiers
        ):
            related_meshes.append(obj)

    mesh_worlds_old = {m.name: m.matrix_world.copy() for m in related_meshes}
    armature.matrix_world = delta @ armature.matrix_world
    for mesh in related_meshes:
        mesh.matrix_world = delta @ mesh_worlds_old[mesh.name]


def _remove_mesh_objects_for_skeleton_only_export(bpy) -> int:
    """Remove mesh objects while leaving the imported armature/animation intact."""
    removed = 0
    for obj in list(bpy.data.objects):
        if obj.type == "MESH":
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
    return removed


def _clear_imported_animation_data(bpy) -> int:
    """Discard animation imported from a source asset before writing NPY motion."""
    datablocks = []
    for collection_name in (
        "objects",
        "armatures",
        "meshes",
        "materials",
        "cameras",
        "lights",
        "curves",
    ):
        datablocks.extend(list(getattr(bpy.data, collection_name, [])))

    for mesh in list(getattr(bpy.data, "meshes", [])):
        shape_keys = getattr(mesh, "shape_keys", None)
        if shape_keys is not None:
            datablocks.append(shape_keys)

    cleared = 0
    seen: set[int] = set()
    for datablock in datablocks:
        key = id(datablock)
        if key in seen or getattr(datablock, "animation_data", None) is None:
            continue
        seen.add(key)
        if hasattr(datablock, "animation_data_clear"):
            datablock.animation_data_clear()
            cleared += 1

    for action in list(getattr(bpy.data, "actions", [])):
        bpy.data.actions.remove(action)

    return cleared


def _prune_unmapped_armature_bones(
    bpy,
    armature,
    keep_bone_names,
) -> list[str]:
    """Remove armature bones that are not part of the kept (AnyTop) joint set.

    The restore imports the full source rig (for its mesh + skin weights), which
    typically carries wrapper bones above the AnyTop root and dead side-branches
    that AnyTop's preprocessing collapses away (so they are absent from the
    NPY / processed BVH). Those bones leak into the exported GLB. This removes
    every bone whose name is not in *keep_bone_names*.

    Called at rest, before any animation is keyed: Blender's edit-mode
    reparenting preserves each kept bone's world rest, so no per-frame transform
    baking is needed. A removed bone's children are reparented to the nearest
    kept ancestor (or made roots), and any skin weight it carries is merged into
    that ancestor's vertex group so deformation is preserved.

    Returns the list of bone names actually removed.
    """
    keep = set(keep_bone_names)
    data_bones = armature.data.bones
    all_names = [b.name for b in data_bones]
    remove_names = [name for name in all_names if name not in keep]
    if not remove_names:
        return []

    # Map every bone to its nearest kept ancestor (None → becomes a root).
    parent_of = {b.name: (b.parent.name if b.parent else None) for b in data_bones}

    def _nearest_kept_ancestor(name: str):
        cursor = parent_of.get(name)
        while cursor is not None and cursor not in keep:
            cursor = parent_of.get(cursor)
        return cursor

    # Objects can be parented directly to a bone, independently of armature
    # deform vertex groups. If the bone is removed, Blender's dependency graph
    # repeatedly warns about missing "Bone Parent" relations. Redirect those
    # object parents before deleting the bones, preserving their rest transform.
    for obj in bpy.data.objects:
        if obj.parent != armature or obj.parent_type != "BONE":
            continue
        if obj.parent_bone not in remove_names:
            continue
        world_matrix = obj.matrix_world.copy()
        target_name = _nearest_kept_ancestor(obj.parent_bone)
        if target_name is None:
            obj.parent_type = "OBJECT"
            obj.parent_bone = ""
        else:
            obj.parent_type = "BONE"
            obj.parent_bone = target_name
        obj.matrix_world = world_matrix

    # Merge skin weights of removed bones into their nearest kept ancestor so
    # deformation is preserved when a removed bone happened to carry weight.
    related_meshes = [
        obj
        for obj in bpy.data.objects
        if obj.type == "MESH"
        and (
            obj.parent == armature
            or any(
                mod.type == "ARMATURE" and mod.object == armature
                for mod in obj.modifiers
            )
        )
    ]
    for mesh in related_meshes:
        vertex_groups = mesh.vertex_groups
        for remove_name in remove_names:
            source_group = vertex_groups.get(remove_name)
            if source_group is None:
                continue
            target_name = _nearest_kept_ancestor(remove_name)
            if target_name is not None:
                target_group = vertex_groups.get(target_name) or vertex_groups.new(
                    name=target_name
                )
                source_index = source_group.index
                for vert in mesh.data.vertices:
                    weight = next(
                        (g.weight for g in vert.groups if g.group == source_index),
                        0.0,
                    )
                    if weight > 0.0:
                        target_group.add([vert.index], weight, "ADD")
            vertex_groups.remove(source_group)

    # Reparent kept children of removed bones, then delete the removed bones.
    # Editing edit_bones keeps each remaining bone's world head/tail fixed.
    previous_active = bpy.context.view_layer.objects.active
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = armature.data.edit_bones
    for name in all_names:
        if name in keep:
            edit_bone = edit_bones.get(name)
            if edit_bone is None:
                continue
            ancestor_name = _nearest_kept_ancestor(name)
            new_parent = edit_bones.get(ancestor_name) if ancestor_name else None
            if edit_bone.parent is not new_parent:
                edit_bone.use_connect = False
                edit_bone.parent = new_parent
    for name in remove_names:
        edit_bone = edit_bones.get(name)
        if edit_bone is not None:
            edit_bones.remove(edit_bone)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.objects.active = previous_active

    return remove_names


def _rename_armature_bones_to_canonical(
    bpy,
    armature,
    name_pairs: list[tuple[str, str]],
) -> None:
    """Rename imported armature bones (and matching mesh vertex groups) to canonical names.

    *name_pairs* is a list of ``(original_name, canonical_name)`` in any order.
    Used for HML restore so the exported GLB carries canonical AnyTop joint
    names instead of the source rig's native bone names. Canonical names are
    already disambiguated to be unique upstream.

    Vertex groups are renamed alongside the bones because Blender's armature
    deform binds vertex group name → bone name; renaming bones without their
    vertex groups would silently break skinning on export. A two-pass rename
    (via unique temporary names) avoids transient collisions when a target
    canonical name still belongs to a not-yet-renamed bone/group.
    """
    rename = [(old, new) for old, new in name_pairs if old != new]
    if not rename:
        return
    rename_map = dict(rename)

    data_bones = armature.data.bones
    pending_bones: list[tuple[str, str]] = []
    for index, (old_name, new_name) in enumerate(rename):
        bone = data_bones.get(old_name)
        if bone is None:
            continue
        temp_name = f"__canon_tmp_{index}"
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

    related_meshes = [
        obj
        for obj in bpy.data.objects
        if obj.type == "MESH"
        and (
            obj.parent == armature
            or any(
                mod.type == "ARMATURE" and mod.object == armature
                for mod in obj.modifiers
            )
        )
    ]
    for mesh in related_meshes:
        vertex_groups = mesh.vertex_groups
        pending_groups: list[tuple[str, str]] = []
        for index, (old_name, new_name) in enumerate(rename):
            group = vertex_groups.get(old_name)
            if group is None:
                continue
            temp_name = f"__canon_tmp_{index}"
            group.name = temp_name
            pending_groups.append((temp_name, new_name))
        for temp_name, new_name in pending_groups:
            group = vertex_groups.get(temp_name)
            if group is not None:
                group.name = new_name


class AnimationExporter:
    """Export optimised joint rotations to GLB, BVH."""

    def __init__(self, skeleton, fps: float = 30.0):
        self.skeleton = skeleton
        self.fps      = fps

        # Defensive: ensure skeleton.bones list index == bone.id
        # (Silent index mismatch would cause completely wrong exports.)
        for idx, b in enumerate(self.skeleton.bones):
            assert b.id == idx, (
                f"skeleton.bones[{idx}].id must equal {idx}, got {b.id}. "
                "The exporter indexes by list position but reads bone.id — "
                "a mismatch will produce silently corrupted output."
            )

    # ------------------------------------------------------------------
    # BVH export (motion_lib.BVH.save)
    # ------------------------------------------------------------------
    def export_bvh(
        self,
        joint_rotations: Tensor,
        root_translation: Tensor,
        root_rotation: Tensor,
        output_path: str,
        bone_translations: Optional[Tensor] = None,
    ) -> None:
        """Write a BVH file using the unified exporter parameter semantics.

        Args:
            joint_rotations: Animated local joint quaternions with shape ``[F, J, 4]``.
            root_translation: Extra wrapper world translation with shape ``[F, 3]``.
            root_rotation: Extra wrapper world rotation with shape ``[F, 4]``.
            output_path: Destination ``.bvh`` path.
            bone_translations: Optional pose-bone local translations with shape
                ``[F, J, 3]``. Non-root entries are exported as explicit BVH
                position channels when provided.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        import numpy as np
        import sys
        _anytop_root = os.path.dirname(os.path.dirname(__file__))
        if _anytop_root not in sys.path:
            sys.path.insert(0, _anytop_root)

        from motion_lib.BVH import save as bvh_save
        from motion_lib.Animation import Animation
        from motion_lib.Quaternions import Quaternions

        F, J = joint_rotations.shape[:2]

        # ── Build joint_names (Skeleton already uses topological order) ──
        # Sanitize: replace whitespace with '_' so BVHView/Anytop loaders
        # that use \S+ regex can re-import the file.
        joint_names = [b.name.replace(" ", "_") for b in self.skeleton.bones]

        # ── Rest-pose attributes ────────────────────────────────────
        offsets_np = np.empty((J, 3), dtype=np.float64)
        rest_rots_np = np.empty((J, 4), dtype=np.float64)
        orients_np = np.empty((J, 4), dtype=np.float64)
        parents_np = np.empty((J,), dtype=np.int32)
        for b in self.skeleton.bones:
            offsets_np[b.id] = b.rest_offset.detach().cpu().to(torch.float64).numpy()
            rest_rots_np[b.id] = b.rest_rotation.detach().cpu().to(torch.float64).numpy()
            orients_np[b.id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            parents_np[b.id] = b.parent_id if b.parent_id is not None else -1

        joint_rot_np = joint_rotations.detach().cpu().to(torch.float64).numpy()
        root_translation_np = root_translation.detach().cpu().to(torch.float64).numpy()
        root_rotation_np = root_rotation.detach().cpu().to(torch.float64).numpy()
        if bone_translations is not None:
            pose_locations_np = bone_translations.detach().cpu().to(torch.float64).numpy()
        else:
            pose_locations_np = None

        # ── Bake rest_rotation into per-frame channel rotations ─────
        # BVH file format has no per-joint rest-rotation slot; the loader
        # rebuilds the rest pose with identity local rotation on every joint.
        # To make the loaded animation reproduce the original total local
        # rotation (rest_q ⊗ pose_q), pre-multiply each channel by rest_q.
        # The wrapper transform `root_rotation` has no BVH equivalent above
        # the root joint, so it is folded into the root channel rotation
        # (root_rotation ⊗ rest_q[0] ⊗ pose_q[0]).
        rotations_np = np.empty_like(joint_rot_np)
        for j in range(J):
            rest_q_repeat = np.repeat(rest_rots_np[j:j + 1], F, axis=0)
            rotations_np[:, j, :] = quat_multiply_wxyz_np(
                rest_q_repeat, joint_rot_np[:, j, :],
            )
        if J > 0:
            rotations_np[:, 0, :] = quat_multiply_wxyz_np(
                root_rotation_np, rotations_np[:, 0, :],
            )
        rotations = Quaternions(rotations_np)

        # ── World-space FK for root position channels ───────────────
        joint_positions_np, _ = _batch_internal_pose_fk_np(
            joint_rot_np,
            root_translation_np,
            root_rotation_np,
            pose_locations_np,
            parents_np,
            offsets_np,
            rest_rots_np,
        )

        # ── Per-frame position channels ─────────────────────────────
        # For non-root j the BVH motion position channel must equal the bone
        # head position in parent's local rest frame, since the loader applies
        # `parent_world_rot @ pose_loc` to chain world positions and parent's
        # world rest rot is already baked into the channel rotations above.
        # That position is `rest_offset[j] + rest_q[j].rotate(pose_loc[j])`.
        positions_np = np.repeat(offsets_np[np.newaxis, :, :], F, axis=0)
        positions_np[:, 0, :] = joint_positions_np[:, 0, :]
        if pose_locations_np is not None and J > 1:
            for joint_idx in range(1, J):
                rest_q = np.repeat(rest_rots_np[joint_idx:joint_idx + 1], F, axis=0)
                positions_np[:, joint_idx, :] = (
                    np.repeat(offsets_np[joint_idx:joint_idx + 1], F, axis=0)
                    + quat_rotate_wxyz_np(rest_q, pose_locations_np[:, joint_idx, :])
                )

        offsets_np[0] = 0.0  # root offset is always zero in BVH

        # ── Reindex into DFS order before writing ───────────────────
        # motion_lib.BVH.save writes the HIERARCHY block via DFS recursion
        # but the MOTION block via `for j in range(n_joints)` (index order).
        # Both BVH.load and the BVH spec require channels in HIERARCHY (DFS)
        # order. When we feed it a BFS-indexed skeleton (the FBX path) the two
        # orders disagree and per-joint channels land on the wrong bones.
        # Remap every per-joint array into DFS order so index == DFS index.
        children_lists = [[] for _ in range(J)]
        for j in range(J):
            p = int(parents_np[j])
            if p >= 0:
                children_lists[p].append(j)

        dfs_order: list[int] = []
        stack = [j for j in range(J) if int(parents_np[j]) < 0]
        stack.reverse()  # so the first root is popped first
        while stack:
            node = stack.pop()
            dfs_order.append(node)
            stack.extend(reversed(children_lists[node]))

        if len(dfs_order) != J:
            raise RuntimeError(
                f"DFS traversal covered {len(dfs_order)} of {J} joints; "
                "skeleton has detached joints not reachable from any root."
            )

        old_to_new = np.empty(J, dtype=np.int64)
        for new_id, old_id in enumerate(dfs_order):
            old_to_new[old_id] = new_id
        perm = np.array(dfs_order, dtype=np.int64)

        joint_names_dfs = [joint_names[i] for i in dfs_order]
        rotations_dfs = Quaternions(rotations_np[:, perm, :])
        positions_dfs = positions_np[:, perm, :]
        offsets_dfs = offsets_np[perm]
        orients_dfs = Quaternions(orients_np[perm])
        parents_dfs = np.array([
            old_to_new[int(parents_np[old_id])] if int(parents_np[old_id]) >= 0 else -1
            for old_id in dfs_order
        ], dtype=np.int32)

        anim = Animation(rotations_dfs, positions_dfs, orients_dfs,
                         offsets_dfs, parents_dfs)
        bvh_save(output_path, anim, names=joint_names_dfs,
                 frametime=1.0 / self.fps, order='xyz',
                 positions=True)

    # ------------------------------------------------------------------
    # GLB export (via Blender glTF exporter)
    # ------------------------------------------------------------------

    def export_glb(
        self,
        joint_rotations: Tensor,
        root_translation: Tensor,
        root_rotation: Tensor,
        output_path: str,
        mesh_path: Optional[str] = None,
        bone_translations: Optional[Tensor] = None,
        rotation_channel_mask: Optional[Tensor | np.ndarray] = None,
        global_similarity: Optional[tuple[float | None, Optional[np.ndarray]]] = None,
        use_image_search: bool = False,
        export_mesh: bool = True,
        rename_bones_to_canonical: bool = False,
        prune_unmapped_bones: bool = False,
    ) -> None:
        """Export GLB directly through bpy in the current Python process.

        When *mesh_path* is provided, imports the external asset for its
        mesh + armature and keyframes animation on it. Set *export_mesh* to
        ``False`` to keep that imported armature path but omit meshes from the
        final GLB. When *mesh_path* is not provided, a fresh armature is created
        from the input skeleton (skeleton-only GLB, no mesh or skinning).

        Args:
            joint_rotations: Animated local joint quaternions with shape ``[F, J, 4]``.
            root_translation: Extra wrapper world translation with shape ``[F, 3]``.
            root_rotation: Extra wrapper world rotation with shape ``[F, 4]``.
            output_path: Destination ``.glb`` path.
            mesh_path: Optional source rig/mesh asset used for skinned export or
                retargeting. When omitted, exports a skeleton-only GLB.
            bone_translations: Optional pose-bone local translations with shape
                ``[F, J, 3]``. Non-root entries follow Blender
                ``pose_bone.location`` semantics.
            rotation_channel_mask: Optional per-joint boolean mask. ``True`` keeps
                writing animated rotation channels for that joint; ``False`` leaves
                the imported/created rest rotation unkeyed for that joint.
            global_similarity: Optional ``(scale_factor, orientation_quat_wxyz)``
                forward HML similarity. When provided alongside *mesh_path*, the
                imported rig (armature + bound meshes) is reverse-aligned into
                HML/npy space so the exported GLB keeps the NPY's orientation,
                scale, and centered placement (skinned restore mode 2). See
                :func:`_apply_gltf_output_space_similarity`.
            use_image_search: When ``True`` (and *mesh_path* is an FBX), import
                textures by recursively searching directories near the source
                mesh (bpy's own resolver), then fall back to
                :func:`_auto_resolve_main_character_textures` for any main
                character mesh still left without a usable base-color texture.
                When ``False`` (default), neither texture-resolution path runs
                and the importer's behavior is unchanged.
            export_mesh: When ``False`` and *mesh_path* is provided, import and
                animate the source armature exactly like a skinned export, then
                remove mesh objects before writing the GLB. This is useful for a
                skeleton-only companion that must preserve the source rig's
                node-scale / local-offset decomposition.
            rename_bones_to_canonical: When ``True`` (and *mesh_path* is given),
                rename the exported armature's bones — and the bound meshes'
                vertex groups — to the canonical BVH joint names (e.g. ``Tail01``)
                so the GLB shares one joint-name set with the processed BVH
                instead of the source rig's native bone names. Used by HML
                restore.
            prune_unmapped_bones: When ``True`` (and *mesh_path* is given), remove
                armature bones the source rig carries above/outside the AnyTop
                skeleton (wrapper roots, dead side-branches) that the AnyTop
                preprocessing collapses away, so the exported GLB's joint set
                matches the NPY / processed BVH. Bones are pruned at rest (no
                per-frame baking); any skin weight is merged into the nearest
                kept ancestor.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        try:
            import bpy
        except ImportError as exc:
            raise RuntimeError(
                "GLB export requires bpy. Install with: pip install bpy"
            ) from exc

        bone_names = [b.name for b in self.skeleton.bones]
        num_frames = joint_rotations.shape[0]
        jr = joint_rotations.detach().cpu().tolist()
        rt = root_translation.detach().cpu().tolist()
        rr = root_rotation.detach().cpu().tolist()
        if rotation_channel_mask is not None:
            if isinstance(rotation_channel_mask, torch.Tensor):
                rotation_channel_mask_np = rotation_channel_mask.detach().cpu().numpy().astype(bool, copy=False)
            else:
                rotation_channel_mask_np = np.asarray(rotation_channel_mask, dtype=bool)
            if rotation_channel_mask_np.shape != (len(bone_names),):
                raise ValueError(
                    f"rotation_channel_mask must have shape ({len(bone_names)},), got {rotation_channel_mask_np.shape}"
                )
        else:
            rotation_channel_mask_np = None
        if bone_translations is not None:
            bt = bone_translations.detach().cpu().tolist()
        else:
            bt = None

        bpy.ops.wm.read_factory_settings(use_empty=True)

        armature = None
        export_skeleton = self.skeleton
        mesh_path_lower = mesh_path.lower() if mesh_path else None
        yup = False
        if mesh_path:
            # 外部 mesh (FBX/GLB) 为 Y-up 坐标系，导出时需 yup=True 以保持一致
            yup = True
            if mesh_path_lower.endswith(".fbx"):
                # FBX import often leaves object-level wrapper rotation/scale on
                # the armature and skinned meshes. Normalize those wrappers back
                # to identity so pose-bone keyframes operate in armature space
                # instead of double-counting the importer transform.
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    import_fbx(mesh_path, use_image_search=use_image_search)
            elif mesh_path_lower.endswith((".glb", ".gltf")):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    bpy.ops.import_scene.gltf(filepath=mesh_path)
            else:
                raise ValueError(
                    f"Unsupported mesh source for GLB export: {mesh_path}"
                )
            remove_lights_and_cameras()
            armature = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
            if armature is None:
                raise RuntimeError(f"No armature found after importing mesh_path: {mesh_path}")
            # When texture resolution is requested, bpy's image search above
            # handles the common case; fall back to our own resolver for any
            # main-character mesh still missing a usable base-color texture.
            # Best-effort and gated so default exports are unchanged.
            if use_image_search:
                resolve_main_character_textures(bpy, armature, mesh_path)
            # GLB/GLTF T-poses can also carry Blender's 0.01 armature wrapper
            # scale. In HML restore we need raw rig units before applying the
            # dataset scale_factor, otherwise the output is 100x too small.
            if mesh_path_lower.endswith(".fbx") or (
                global_similarity is not None
                and mesh_path_lower.endswith((".glb", ".gltf"))
            ):
                _normalize_imported_armature_and_meshes(bpy, armature)
            if global_similarity is not None:
                _apply_gltf_output_space_similarity(bpy, armature, *global_similarity)
        elif global_similarity is not None:
            raise ValueError(
                "global_similarity (HML reverse-alignment) requires a mesh_path; "
                "it has no effect on skeleton-only GLB export."
            )
        else:
            armature = self._create_armature_from_skeleton(bpy, skeleton=export_skeleton)

        # ────────────────────────────────────────────────────────────────
        # Retargeting: convert input-skeleton animation to FBX armature
        # local space via world-space alignment. The numpy core lives in
        # ``Anytop.utils.retarget`` so non-Blender callers can share it.
        # ────────────────────────────────────────────────────────────────
        if mesh_path:
            fbx_names, fbx_parents, fbx_offsets, fbx_rest_rots = extract_armature_skeleton_data(armature)
            J_fbx = len(fbx_names)

            rest_rot_input = np.array([
                b.rest_rotation.detach().cpu().numpy().astype(np.float64)
                for b in self.skeleton.bones
            ])
            parents_input = np.array([
                b.parent_id if b.parent_id is not None else -1
                for b in self.skeleton.bones
            ], dtype=np.int32)
            rest_offsets_input = np.array([
                b.rest_offset.detach().cpu().numpy().astype(np.float64)
                for b in self.skeleton.bones
            ])

            # Imported GLB rigs already carry the glTF wrapper/object space that
            # Blender will re-emit on export. Running the internal rest-pose
            # coordinate search there can spuriously add an extra rigid basis
            # rotation (Horse picked R_y(+90°), which re-imports as a visible
            # whole-character Z rotation), so plain GLB->GLB retargeting keeps
            # the rig's existing basis.
            #
            # HML restore is different: ``global_similarity`` has already
            # reverse-aligned the imported GLB rig into NPY/HML space while the
            # recovered source animation still arrives in raw export space. In
            # that case disabling coordinate search pins the alignment to
            # identity and cancels the intended 90-degree facing change, so we
            # re-enable the search only for the reverse-aligned path.
            is_gltf_mesh = bool(
                mesh_path_lower and mesh_path_lower.endswith((".glb", ".gltf"))
            )
            coordinate_search = (not is_gltf_mesh) or (global_similarity is not None)
            src_match_names = _build_canonical_match_names(
                bone_names,
                parents_input,
                rest_offsets_input,
                log_hint="export source skeleton",
            )

            def _retarget_to_armature(names, parents, offsets, rest_rots, verbose):
                tgt_names, tgt_bvh_names = _build_canonical_name_variants(
                    names,
                    parents,
                    offsets,
                    log_hint=os.path.basename(mesh_path) if mesh_path else "export target armature",
                )
                result = retarget_world_space_np(
                    src_parents=parents_input,
                    src_rest_offsets=rest_offsets_input,
                    src_rest_rotations=rest_rot_input,
                    tgt_parents=parents,
                    tgt_rest_offsets=offsets,
                    tgt_rest_rotations=rest_rots,
                    src_joint_rotations=np.array(jr, dtype=np.float64),
                    src_root_translation=np.array(rt, dtype=np.float64),
                    src_root_rotation=np.array(rr, dtype=np.float64),
                    src_match_names=src_match_names,
                    tgt_match_names=tgt_names,
                    src_bone_translations=np.array(bt, dtype=np.float64) if bt is not None else None,
                    coordinate_search=coordinate_search,
                    verbose=verbose,
                )
                return result, tgt_bvh_names

            # Prune wrapper / dead-branch bones the source rig carries above or
            # outside the AnyTop skeleton (absent from the NPY / processed BVH).
            # A first mapping-only retarget identifies which armature bones the
            # AnyTop joints actually map to; the rest are pruned at rest, then we
            # retarget against the cleaned-up armature so indices stay consistent.
            if prune_unmapped_bones:
                mapping_result, _ = _retarget_to_armature(
                    fbx_names, fbx_parents, fbx_offsets, fbx_rest_rots, verbose=False
                )
                src_to_tgt = mapping_result["src_to_tgt"]
                keep_bone_names = {
                    fbx_names[int(t)] for t in src_to_tgt if int(t) >= 0
                }
                removed = _prune_unmapped_armature_bones(bpy, armature, keep_bone_names)
                if removed:
                    preview = ", ".join(removed[:10]) + ("..." if len(removed) > 10 else "")
                    print(f"Pruned {len(removed)} unmapped bone(s): {preview}")
                    fbx_names, fbx_parents, fbx_offsets, fbx_rest_rots = (
                        extract_armature_skeleton_data(armature)
                    )
                    J_fbx = len(fbx_names)

            retarget_result, tgt_bvh_names = _retarget_to_armature(
                fbx_names, fbx_parents, fbx_offsets, fbx_rest_rots, verbose=True
            )

            fbx_pose_rot = retarget_result["joint_rotations"]
            fbx_pose_loc = retarget_result["bone_translations"]
            input_to_fbx = retarget_result["src_to_tgt"]

            root_mask = fbx_parents < 0
            root_indices = np.flatnonzero(root_mask)

            if rotation_channel_mask_np is not None:
                fbx_rotation_channel_mask = np.zeros((J_fbx,), dtype=bool)
                for ii, fi in enumerate(input_to_fbx):
                    if fi >= 0 and rotation_channel_mask_np[ii]:
                        fbx_rotation_channel_mask[int(fi)] = True
                if root_indices.size > 0:
                    fbx_rotation_channel_mask[root_indices[0]] = True
                rotation_channel_mask_np = fbx_rotation_channel_mask

            jr = fbx_pose_rot.tolist()
            rr = retarget_result["root_rotation"].tolist()
            rt = retarget_result["root_translation"].tolist()
            bone_names = fbx_names
            bt = fbx_pose_loc.tolist() if fbx_pose_loc is not None else None

            if rename_bones_to_canonical:
                # Rename the imported rig's bones (and matching vertex groups) to
                # the canonical BVH joint names so the GLB shares one joint-name
                # set with the processed BVH (e.g. "Tail01") instead of the source
                # rig's native names. The animation channels below are keyed by
                # name, so update the lookup list to the canonical names too.
                _rename_armature_bones_to_canonical(
                    bpy, armature, list(zip(fbx_names, tgt_bvh_names))
                )
                bone_names = list(tgt_bvh_names)

        # ── Drop imported animation, create fresh NPY action ─────────
        _clear_imported_animation_data(bpy)
        armature.animation_data_create()
        action = bpy.data.actions.new(name="PCVGAnimation")
        armature.animation_data.action = action
        armature.rotation_mode = "QUATERNION"

        scene = bpy.context.scene
        scene.render.fps = int(self.fps)
        scene.render.fps_base = 1.0
        scene.frame_start = 0
        scene.frame_end = max(num_frames - 1, 0)

        bpy.context.view_layer.objects.active = armature
        armature.select_set(True)
        bpy.ops.object.mode_set(mode="POSE")

        # ── Build per-channel value arrays (vectorised over frames) ───
        # Instead of setting properties + keyframe_insert F×J×3 times (tens of
        # thousands of Python→C calls), we establish the fcurve structure with a
        # single frame-0 keyframe per channel (so masking / root handling stays
        # byte-for-byte identical to the per-frame path), then bulk-fill every
        # frame in one ``foreach_set`` per fcurve. ``data_path_value_arrays``
        # maps each fcurve's (data_path, array_index) to its length-F samples.
        F = num_frames
        jr_np = np.asarray(jr, dtype=np.float64) if F else None
        rt_np = np.asarray(rt, dtype=np.float64) if F else None
        rr_np = np.asarray(rr, dtype=np.float64) if F else None
        bt_np = np.asarray(bt, dtype=np.float64) if (bt is not None and F) else None
        zeros_f3 = np.zeros((F, 3), dtype=np.float64) if F else None
        ones_f3 = np.ones((F, 3), dtype=np.float64) if F else None

        data_path_value_arrays: dict[tuple[str, int], np.ndarray] = {}

        def _record(base_path: str, arr: np.ndarray) -> None:
            # arr has shape (F, C); register one entry per component column.
            for c in range(arr.shape[1]):
                data_path_value_arrays[(base_path, c)] = np.ascontiguousarray(arr[:, c])

        if F:
            if not mesh_path:
                armature.rotation_mode = "QUATERNION"
                armature.location = tuple(rt_np[0])
                armature.rotation_quaternion = tuple(rr_np[0])
                armature.scale = (1.0, 1.0, 1.0)
                armature.keyframe_insert(data_path="location", frame=0)
                armature.keyframe_insert(data_path="rotation_quaternion", frame=0)
                armature.keyframe_insert(data_path="scale", frame=0)
                _record(armature.path_from_id("location"), rt_np)
                _record(armature.path_from_id("rotation_quaternion"), rr_np)
                _record(armature.path_from_id("scale"), ones_f3)

            for j, bname in enumerate(bone_names):
                pbone = armature.pose.bones.get(bname)
                if pbone is None:
                    continue

                if mesh_path:
                    # After retargeting, bones are FBX bones and parent comes from FBX
                    parent_id = fbx_parents[j] if j < len(fbx_names) else -1
                else:
                    parent_id = self.skeleton.bones[j].parent_id if self.skeleton.bones[j].parent_id is not None else -1
                is_root = parent_id < 0
                if is_root and mesh_path:
                    loc_arr = rt_np
                    rot_arr = rr_np
                else:
                    rot_arr = jr_np[:, j, :]
                    if bt_np is not None and not is_root:
                        loc_arr = bt_np[:, j, :]
                    else:
                        loc_arr = zeros_f3

                write_rotation_channel = (
                    True if rotation_channel_mask_np is None else bool(rotation_channel_mask_np[j])
                )
                pbone.rotation_mode = "QUATERNION"
                pbone.location = tuple(loc_arr[0])
                if write_rotation_channel:
                    pbone.rotation_quaternion = tuple(rot_arr[0])
                else:
                    pbone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
                pbone.scale = (1.0, 1.0, 1.0)
                pbone.keyframe_insert(data_path="location", frame=0)
                if write_rotation_channel:
                    pbone.keyframe_insert(data_path="rotation_quaternion", frame=0)
                pbone.keyframe_insert(data_path="scale", frame=0)

                _record(pbone.path_from_id("location"), loc_arr)
                if write_rotation_channel:
                    _record(pbone.path_from_id("rotation_quaternion"), rot_arr)
                _record(pbone.path_from_id("scale"), ones_f3)

        bpy.ops.object.mode_set(mode="OBJECT")

        if mesh_path and not export_mesh:
            _remove_mesh_objects_for_skeleton_only_export(bpy)

        # ── Bulk-fill every frame on each fcurve via foreach_set ──────
        if F and action is not None and data_path_value_arrays:
            frames = np.arange(F, dtype=np.float64)
            channelbags = []
            for layer in getattr(action, "layers", []):
                for strip in layer.strips:
                    channelbags.extend(getattr(strip, "channelbags", []))
            # LINEAR interpolation enum maps to integer 1 (CONSTANT=0, BEZIER=2).
            linear_codes = np.ones(F, dtype=np.int32)
            for cb in channelbags:
                for fc in cb.fcurves:
                    values = data_path_value_arrays.get((fc.data_path, fc.array_index))
                    if values is None:
                        continue
                    existing = len(fc.keyframe_points)
                    if existing < F:
                        fc.keyframe_points.add(F - existing)
                    co = np.empty(2 * F, dtype=np.float64)
                    co[0::2] = frames
                    co[1::2] = values
                    fc.keyframe_points.foreach_set("co", co)
                    # Force LINEAR (replaces the former per-keyframe Python loop).
                    fc.keyframe_points.foreach_set("interpolation", linear_codes)
                    fc.update()

        # ── Export GLB ────────────────────────────────────────────────
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            bpy.ops.export_scene.gltf(
                filepath=output_path,
                export_format='GLB',
                export_animations=True,
                export_animation_mode='ACTIVE_ACTIONS',
                export_force_sampling=True,
                export_frame_range=True,
                export_apply=False,
                export_yup=yup,
            )

    def _create_armature_from_skeleton(self, bpy, skeleton=None):
        from mathutils import Quaternion, Vector

        skeleton = self.skeleton if skeleton is None else skeleton

        bpy.ops.object.armature_add()
        armature_obj = bpy.context.active_object
        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode="EDIT")

        edit_bones = armature_obj.data.edit_bones
        if edit_bones:
            edit_bones.remove(edit_bones[0])

        J = len(skeleton.bones)
        # Compute children from parent_ids (works for SimpleSkeleton and any
        # skeleton class that doesn't have children_ids precomputed).
        children = [[] for _ in range(J)]
        for b in skeleton.bones:
            pid = b.parent_id
            if pid is not None and pid >= 0:
                children[pid].append(b.id)

        world_heads = {}
        world_rotations = {}
        for bone in skeleton.bones:
            local_rotation = Quaternion(tuple(float(v) for v in bone.rest_rotation.tolist()))
            local_offset = Vector(tuple(float(v) for v in bone.rest_offset.tolist()))
            if bone.parent_id is None:
                head = local_offset
                world_rotation = local_rotation
            else:
                parent_rotation = world_rotations[bone.parent_id]
                head = world_heads[bone.parent_id] + (parent_rotation @ local_offset)
                world_rotation = parent_rotation @ local_rotation
            world_heads[bone.id] = head
            world_rotations[bone.id] = world_rotation

        created = {}
        for bone in skeleton.bones:
            eb = edit_bones.new(bone.name)
            head = world_heads[bone.id]
            eb.head = head
            child_ids = children[bone.id]
            default_length = max(float(bone.rest_offset.norm().item()), 0.1)
            if child_ids:
                child_id = child_ids[0]
                child_offset = Vector(tuple(float(v) for v in skeleton.bones[child_id].rest_offset.tolist()))
                bone_length = max(child_offset.length, 0.1)
            else:
                bone_length = default_length
            world_rotation = world_rotations[bone.id]
            eb.tail = head + (world_rotation @ Vector((0.0, bone_length, 0.0)))
            eb.align_roll(world_rotation @ Vector((0.0, 0.0, 1.0)))
            created[bone.id] = eb

        for bone in skeleton.bones:
            if bone.parent_id is not None:
                created[bone.id].parent = created[bone.parent_id]

        bpy.ops.object.mode_set(mode="OBJECT")
        return armature_obj
