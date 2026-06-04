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


__all__ = [
    "animation_to_exporter_inputs",
    "AnimationExporter",
]


def _build_canonical_match_names(
    joint_names: list[str],
    parents: np.ndarray,
    offsets: np.ndarray,
    *,
    log_hint: str,
) -> list[str]:
    """Derive canonical joint names from raw skeleton data.

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
    canonical_joint_names = object_cond.get("canonical_joint_names")
    if canonical_joint_names is None:
        raise ValueError(
            f"Unable to derive canonical_joint_names ({log_hint})"
        )
    canonical_joint_names = list(canonical_joint_names)
    if len(canonical_joint_names) != len(joint_names):
        raise ValueError(
            f"Derived canonical_joint_names length mismatch ({log_hint}): "
            f"{len(canonical_joint_names)} vs {len(joint_names)}"
        )
    return canonical_joint_names


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
    """Normalize imported FBX object scale while preserving the axis wrapper.

    FBX import commonly leaves a +90deg X axis wrapper and a 0.01 object scale
    on the armature object. We want to remove the importer scale and stale
    parent-inverse state, but we must keep the axis wrapper rotation: the
    imported bone and mesh local data still live in the FBX armature's Y-up
    object space, so stripping the rotation would roll the character 90 degrees.

    Dropping the armature's object scale re-interprets its armature-local bone
    data (centimetre units) as world units, which is what aligns the exported
    skeleton with the centimetre-scale NPY animation. The skinned meshes must
    move with that change so they stay glued to the skeleton — but they do NOT
    necessarily share the armature's object scale: a mesh parented *under* the
    armature inherits an extra scale factor (e.g. armature scale 0.01 + mesh
    basis 0.01 -> mesh world scale 0.0001). Resetting every object to unit scale
    independently therefore detaches such a mesh (it ends up 100x too large).

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
    # Normalized armature frame: keep world translation + rotation, drop scale/shear.
    arm_position = arm_world_old.translation.copy()
    arm_rotation = arm_world_old.to_quaternion().to_matrix().to_4x4()
    arm_world_new = Matrix.Translation(arm_position) @ arm_rotation

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
    ) -> None:
        """Export GLB directly through bpy in the current Python process.

        When *mesh_path* is provided, imports the external asset for its
        mesh + armature and keyframes animation on it.  When not provided,
        only the armature is exported (skeleton-only GLB, no mesh or skinning).

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
                    import_fbx(mesh_path)
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
            if mesh_path_lower.endswith(".fbx"):
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
            # whole-character Z rotation). Keep GLB/GTLF retargeting in the
            # rig's existing basis and reserve the broader coordinate search
            # for FBX sources.
            coordinate_search = not (
                mesh_path_lower and mesh_path_lower.endswith((".glb", ".gltf"))
            )
            src_match_names = _build_canonical_match_names(
                bone_names,
                parents_input,
                rest_offsets_input,
                log_hint="export source skeleton",
            )
            tgt_match_names = _build_canonical_match_names(
                fbx_names,
                fbx_parents,
                fbx_offsets,
                log_hint=os.path.basename(mesh_path) if mesh_path else "export target armature",
            )

            retarget_result = retarget_world_space_np(
                src_parents=parents_input,
                src_rest_offsets=rest_offsets_input,
                src_rest_rotations=rest_rot_input,
                tgt_parents=fbx_parents,
                tgt_rest_offsets=fbx_offsets,
                tgt_rest_rotations=fbx_rest_rots,
                src_joint_rotations=np.array(jr, dtype=np.float64),
                src_root_translation=np.array(rt, dtype=np.float64),
                src_root_rotation=np.array(rr, dtype=np.float64),
                src_match_names=src_match_names,
                tgt_match_names=tgt_match_names,
                src_bone_translations=np.array(bt, dtype=np.float64) if bt is not None else None,
                coordinate_search=coordinate_search,
                verbose=True,
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

        # ── Clear existing animation, create fresh action ─────────────
        if armature.animation_data:
            armature.animation_data_clear()
        armature.animation_data_create()
        # Remove any old actions to avoid name collisions
        for a in list(bpy.data.actions):
            if a.name.startswith("PCVGAnimation"):
                bpy.data.actions.remove(a)
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

        for f in range(num_frames):
            scene.frame_set(f)
            if not mesh_path:
                armature.location = (rt[f][0], rt[f][1], rt[f][2])
                armature.rotation_mode = "QUATERNION"
                armature.rotation_quaternion = (rr[f][0], rr[f][1], rr[f][2], rr[f][3])
                armature.scale = (1.0, 1.0, 1.0)
                armature.keyframe_insert(data_path="location", frame=f)
                armature.keyframe_insert(data_path="rotation_quaternion", frame=f)
                armature.keyframe_insert(data_path="scale", frame=f)
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
                    loc_val = rt[f]
                    rot_val = rr[f]
                    pbone.location = (loc_val[0], loc_val[1], loc_val[2])
                else:
                    rot_val = jr[f][j]
                    if bt is not None:
                        loc_val = bt[f][j] if not is_root else (0.0, 0.0, 0.0)
                        pbone.location = (loc_val[0], loc_val[1], loc_val[2])
                    else:
                        pbone.location = (0.0, 0.0, 0.0)

                write_rotation_channel = (
                    True if rotation_channel_mask_np is None else bool(rotation_channel_mask_np[j])
                )
                pbone.rotation_mode = "QUATERNION"
                if write_rotation_channel:
                    pbone.rotation_quaternion = (rot_val[0], rot_val[1], rot_val[2], rot_val[3])
                else:
                    pbone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
                pbone.scale = (1.0, 1.0, 1.0)
                pbone.keyframe_insert(data_path="location", frame=f)
                if write_rotation_channel:
                    pbone.keyframe_insert(data_path="rotation_quaternion", frame=f)
                pbone.keyframe_insert(data_path="scale", frame=f)

        bpy.ops.object.mode_set(mode="OBJECT")

        # ── Force LINEAR interpolation ────────────────────────────────
        if action is not None:
            all_fcurves = []
            if hasattr(action, "fcurves"):
                all_fcurves = list(action.fcurves)
            elif hasattr(action, "layers"):
                for layer in action.layers:
                    for strip in layer.strips:
                        if hasattr(strip, "channelbags"):
                            for channelbag in strip.channelbags:
                                all_fcurves.extend(channelbag.fcurves)
            for fcurve in all_fcurves:
                for kp in fcurve.keyframe_points:
                    kp.interpolation = "LINEAR"

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
