"""
Animation export to GLB and BVH formats.

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

from Anytop.utils.fbx import import_fbx, remove_lights_and_cameras
from Anytop.utils.rotation_numpy import (
    apply_rotation_to_quaternions_wxyz_np,
    quat_conjugate_wxyz_np,
    quat_multiply_wxyz_np,
    quat_rotate_wxyz_np,
)


# ---------------------------------------------------------------------------
# Retargeting helpers (numpy-based, for world-space alignment & FBX-local
# conversion in the mesh_path GLB export path)
# ---------------------------------------------------------------------------

def _canonical_bone_name(name: str) -> str:
    """Normalize bone name for cross-format matching."""
    return name.replace(" ", "_").lower()


def _generate_coordinate_candidates_np():
    """Generate candidate 3x3 rotation/flip matrices for auto-detection."""
    I = np.eye(3, dtype=np.float64)

    def R_x(deg):
        c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)

    def R_y(deg):
        c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)

    def R_z(deg):
        c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

    return [
        ("identity", I),
        ("R_x(+90°)", R_x(90)),
        ("R_x(-90°)", R_x(-90)),
        ("R_y(+90°)", R_y(90)),
        ("R_y(-90°)", R_y(-90)),
        ("R_z(+90°)", R_z(90)),
        ("R_z(-90°)", R_z(-90)),
        ("R_x(+180°)", R_x(180)),
        ("R_z(+180°)", R_z(180)),
        ("flip_X", np.diag([-1, 1, 1])),
        ("flip_Y", np.diag([1, -1, 1])),
        ("flip_Z", np.diag([1, 1, -1])),
    ]


def _batch_forward_kinematics_np(
    local_rotations: np.ndarray,
    local_positions: np.ndarray,
    parents: np.ndarray,
    rest_rotations: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute world-space positions & rotations from local data.

    Args:
        local_rotations: (F, J, 4)  animated local quaternions
        local_positions: (F, J, 3)  local (parent-relative) translations
        parents:         (J,) int32 parent indices (-1 = root)
        rest_rotations:  (J, 4) or None — total local rot = rest_rot ⊗ local_rot

    Returns:
        world_positions: (F, J, 3)
        world_rotations: (F, J, 4)
    """
    F, J = local_rotations.shape[:2]

    # Compose total local rotation
    if rest_rotations is not None:
        total_local = np.zeros((F, J, 4), dtype=np.float64)
        for j in range(J):
            total_local[:, j] = quat_multiply_wxyz_np(
                rest_rotations[j:j+1], local_rotations[:, j]
            )
    else:
        total_local = local_rotations.copy()

    world_pos = np.zeros((F, J, 3), dtype=np.float64)
    world_rot = np.zeros((F, J, 4), dtype=np.float64)

    for j in range(J):
        p = parents[j]
        if p < 0:
            world_pos[:, j] = local_positions[:, j]
            world_rot[:, j] = total_local[:, j]
        else:
            world_pos[:, j] = world_pos[:, p] + quat_rotate_wxyz_np(
                world_rot[:, p], local_positions[:, j]
            )
            world_rot[:, j] = quat_multiply_wxyz_np(world_rot[:, p], total_local[:, j])

    return world_pos, world_rot


def _batch_pose_fk_np(
    pose_rotations: np.ndarray,
    pose_locations: np.ndarray,
    parents: np.ndarray,
    rest_offsets: np.ndarray,
    rest_rotations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute world transforms using Blender pose-bone semantics.

    Local bone transform is modeled as:
        T_local = T(rest_offset) * R(rest_rotation) * T(pose_location) * R(pose_rotation)

    This matches how the exporter drives external FBX/GLB armatures through
    pose bone `location` and `rotation_quaternion` channels.
    """
    F, J = pose_rotations.shape[:2]
    world_pos = np.zeros((F, J, 3), dtype=np.float64)
    world_rot = np.zeros((F, J, 4), dtype=np.float64)

    for j in range(J):
        rest_q = np.repeat(rest_rotations[j:j+1], F, axis=0)
        total_local_rot = quat_multiply_wxyz_np(rest_q, pose_rotations[:, j])
        pose_loc_in_parent = rest_offsets[j:j+1] + quat_rotate_wxyz_np(rest_q, pose_locations[:, j])

        p = parents[j]
        if p < 0:
            world_pos[:, j] = pose_loc_in_parent
            world_rot[:, j] = total_local_rot
        else:
            world_pos[:, j] = world_pos[:, p] + quat_rotate_wxyz_np(world_rot[:, p], pose_loc_in_parent)
            world_rot[:, j] = quat_multiply_wxyz_np(world_rot[:, p], total_local_rot)

    return world_pos, world_rot


def _batch_internal_pose_fk_np(
    joint_rotations: np.ndarray,
    root_translation: np.ndarray,
    root_rotation: np.ndarray,
    pose_locations: np.ndarray | None,
    parents: np.ndarray,
    rest_offsets: np.ndarray,
    rest_rotations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute world pose with unified exporter semantics.

    External caller semantics:
      - ``joint_rotations`` carries animated local joint quaternions for all joints,
        including the root joint.
      - ``root_translation`` / ``root_rotation`` form an extra world-space wrapper
        transform applied before the skeleton hierarchy.
      - ``pose_locations`` carries optional Blender-style pose-bone location channels
        for non-root joints. The root entry is ignored; root world translation always
        comes from ``root_translation``.
    """
    F, J = joint_rotations.shape[:2]
    world_pos = np.zeros((F, J, 3), dtype=np.float64)
    world_rot = np.zeros((F, J, 4), dtype=np.float64)

    zero_loc = np.zeros((F, 3), dtype=np.float64)

    for j in range(J):
        rest_q = np.repeat(rest_rotations[j:j+1], F, axis=0)
        total_local_rot = quat_multiply_wxyz_np(rest_q, joint_rotations[:, j])

        if pose_locations is None or parents[j] < 0:
            pose_loc = zero_loc
        else:
            pose_loc = pose_locations[:, j]

        local_pos = np.repeat(rest_offsets[j:j+1], F, axis=0) + quat_rotate_wxyz_np(
            rest_q,
            pose_loc,
        )

        p = parents[j]
        if p < 0:
            world_pos[:, j] = root_translation + quat_rotate_wxyz_np(root_rotation, local_pos)
            world_rot[:, j] = quat_multiply_wxyz_np(root_rotation, total_local_rot)
        else:
            world_pos[:, j] = world_pos[:, p] + quat_rotate_wxyz_np(world_rot[:, p], local_pos)
            world_rot[:, j] = quat_multiply_wxyz_np(world_rot[:, p], total_local_rot)

    return world_pos, world_rot


def _batch_internal_fk_np(
    joint_rotations: np.ndarray,
    root_translation: np.ndarray,
    root_rotation: np.ndarray,
    parents: np.ndarray,
    rest_offsets: np.ndarray,
    rest_rotations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match Anytop forward_kinematics root semantics in numpy.

    The pipeline skeleton uses a separate root world transform:
        world_root = T(root_translation) * R(root_rotation)
    and then applies each bone's parent-local rest transform followed by the
    animated local joint rotation. This differs from Blender pose-bone channel
    semantics and must be preserved before retargeting to an imported FBX rig.
    """
    return _batch_internal_pose_fk_np(
        joint_rotations,
        root_translation,
        root_rotation,
        pose_locations=None,
        parents=parents,
        rest_offsets=rest_offsets,
        rest_rotations=rest_rotations,
    )


def _extract_fbx_skeleton_data(armature):
    """Extract bone names, parents, offsets, rest rotations from a Blender armature.

    Uses BFS traversal (largest-root-subtree heuristic to find the primary
    skeleton when multiple roots exist).
    """
    from collections import deque
    armature_bones = armature.data.bones
    all_roots = [b for b in armature_bones if b.parent is None]
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

    J = len(ordered_bones)
    bone_names = [b.name for b in ordered_bones]
    parents = np.full(J, -1, dtype=np.int32)
    offsets = np.zeros((J, 3), dtype=np.float64)
    rest_rotations = np.zeros((J, 4), dtype=np.float64)
    name_to_idx = {n: i for i, n in enumerate(bone_names)}

    for joint_idx, bone in enumerate(ordered_bones):
        if bone.parent is not None and bone.parent.name in name_to_idx:
            parents[joint_idx] = name_to_idx[bone.parent.name]
            rest_local = bone.parent.matrix_local.inverted_safe() @ bone.matrix_local
        else:
            rest_local = bone.matrix_local.copy()
        t = rest_local.translation
        q = rest_local.to_quaternion()
        offsets[joint_idx] = (t.x, t.y, t.z)
        rest_rotations[joint_idx] = (q.w, q.x, q.y, q.z)

    return bone_names, parents, offsets, rest_rotations


def _normalize_imported_armature_and_meshes(bpy, armature) -> None:
    """Remove FBX importer object-level rotation/scale from armature + meshes.

    FBX import commonly leaves a +90deg X rotation and 0.01 object scale on
    the armature object. The skinned meshes inherit the same world transform via
    parenting / armature modifiers. That object-level transform is not part of
    the pose retargeting math and can corrupt the exported glTF skin bind.

    Normalize the imported objects back to identity rotation + unit scale while
    preserving translation and keeping armature-local mesh/bone data unchanged.
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

    objects_to_normalize = [armature] + related_meshes
    world_positions = {
        obj.name: obj.matrix_world.translation.copy()
        for obj in objects_to_normalize
    }

    for obj in objects_to_normalize:
        obj.matrix_parent_inverse = Matrix.Identity(4)
        obj.matrix_world = Matrix.Translation(world_positions[obj.name])


class AnimationExporter:
    """Export optimised joint rotations to GLB or BVH."""

    def __init__(self, skeleton, fps: float = 30.0):
        self.skeleton = skeleton
        self.fps      = fps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export(self, joint_rotations: Tensor, root_translation: Tensor,
               root_rotation: Tensor, output_path: str,
               mesh_path: Optional[str] = None,
               bone_translations: Optional[Tensor] = None) -> None:
        """Export animation to the format inferred from *output_path* extension.

        Args:
            joint_rotations:  [F, J, 4]  local quaternions for all joints,
                              including the root joint's animated local rotation
            root_translation: [F, 3]     world translation of an extra wrapper
                              transform applied before the skeleton hierarchy
            root_rotation:    [F, 4]     world rotation of that extra wrapper
                              transform; use identity when the motion already
                              lives entirely in joint_rotations
            output_path:      destination file (*.glb or *.bvh)
            mesh_path:        source mesh/rig for GLB export (e.g. T-pose GLB/FBX)
            bone_translations: [F, J, 3] optional Blender-style pose-bone local
                               translations. Non-root entries match
                               pose_bone.location semantics; the root entry is
                               ignored because root world translation is always
                               carried by root_translation.
        """
        ext = os.path.splitext(output_path)[1].lower()
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        if ext == ".bvh":
            self._export_bvh(joint_rotations, root_translation,
                             root_rotation, output_path,
                             bone_translations=bone_translations)
        elif ext == ".glb":
            self._export_glb(joint_rotations, root_translation,
                             root_rotation, output_path, mesh_path,
                             bone_translations=bone_translations)
        else:
            raise ValueError(f"Unsupported export format: {ext!r}")

    # ------------------------------------------------------------------
    # BVH export (motion_lib.BVH.save)
    # ------------------------------------------------------------------
    def _export_bvh(self, joint_rotations: Tensor, root_translation: Tensor,
                    root_rotation: Tensor, output_path: str,
                    bone_translations: Optional[Tensor] = None) -> None:
        """Write a BVH file using the unified exporter parameter semantics."""
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
                 positions=True,
                 all_joints_as_names=True)

    # ------------------------------------------------------------------
    # GLB export (via Blender glTF exporter)
    # ------------------------------------------------------------------

    def _export_glb(self, joint_rotations: Tensor, root_translation: Tensor,
                    root_rotation: Tensor, output_path: str,
                    mesh_path: Optional[str],
                    bone_translations: Optional[Tensor] = None) -> None:
        """Export GLB directly through bpy in the current Python process.

        When *mesh_path* is provided, imports the external asset for its
        mesh + armature and keyframes animation on it.  When not provided,
        only the armature is exported (skeleton-only GLB, no mesh or skinning).
        """
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
                # Preserve the FBX import wrapper transform (+90deg X, 0.01 scale)
                # so the re-imported GLB lands in the same object space.
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    import_fbx(mesh_path, ignore_leaf_bones=False)
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
        else:
            armature = self._create_armature_from_skeleton(bpy, skeleton=export_skeleton)

        # ────────────────────────────────────────────────────────────────
        # Retargeting: convert input-skeleton animation to FBX armature
        # local space via world-space alignment.
        # ────────────────────────────────────────────────────────────────
        if mesh_path:
            # ── A) Extract FBX skeleton metadata ───────────────────────
            fbx_names, fbx_parents, fbx_offsets, fbx_rest_rots = _extract_fbx_skeleton_data(armature)
            J_fbx = len(fbx_names)
            fbx_canon_to_idx = {_canonical_bone_name(n): i for i, n in enumerate(fbx_names)}

            # ── B) Map input→FBX bone indices ─────────────────────────
            input_to_fbx = np.full(len(bone_names), -1, dtype=np.int32)
            for i, name in enumerate(bone_names):
                canon = _canonical_bone_name(name)
                if canon in fbx_canon_to_idx:
                    input_to_fbx[i] = fbx_canon_to_idx[canon]

            # ── C) Build numpy arrays for FK ──────────────────────────
            jr_np = np.array(jr, dtype=np.float64)           # (F, J, 4)
            rt_np = np.array(rt, dtype=np.float64)           # (F, 3)
            rr_np = np.array(rr, dtype=np.float64)           # (F, 4)

            # Rest rotations from input skeleton
            rest_rot_input = np.array([
                b.rest_rotation.detach().cpu().numpy().astype(np.float64)
                for b in self.skeleton.bones
            ])

            # Parents from input skeleton
            parents_input = np.array([
                b.parent_id if b.parent_id is not None else -1
                for b in self.skeleton.bones
            ], dtype=np.int32)

            # ── D) Compute input world-space (FK) ─────────────────────
            rest_offsets_input = np.array([
                b.rest_offset.detach().cpu().numpy().astype(np.float64)
                for b in self.skeleton.bones
            ])

            if bt is not None:
                pose_locations_np = np.array(bt, dtype=np.float64)
            else:
                pose_locations_np = None

            input_wpos, input_wrot = _batch_internal_pose_fk_np(
                jr_np,
                rt_np,
                rr_np,
                pose_locations_np,
                parents_input,
                rest_offsets_input,
                rest_rot_input,
            )

            # ── E) Compute FBX rest-pose world-space ─────────────────
            # One-frame rest pose: identity pose rotations + zero pose translations
            identity_rot = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
            fbx_rest_local_rot = np.tile(identity_rot, (J_fbx, 1))  # (J_fbx, 4)
            fbx_rest_local_pos = np.zeros((1, J_fbx, 3), dtype=np.float64)

            fbx_rest_wpos, fbx_rest_wrot = _batch_pose_fk_np(
                fbx_rest_local_rot[None],          # (1, J, 4)
                fbx_rest_local_pos,                # (1, J, 3)
                fbx_parents,
                fbx_offsets,
                fbx_rest_rots,
            )

            # ── F) Auto-detect alignment on common bones ─────────────────
            # Use the INPUT REST POSE (identity rotations) — NOT frame 0 of
            # the animation, which may be in a running pose.
            input_rest_local_rot = np.tile(
                np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64), (len(bone_names), 1)
            )
            input_rest_local_pos = np.zeros((1, len(bone_names), 3), dtype=np.float64)

            input_rest_wpos, _ = _batch_pose_fk_np(
                input_rest_local_rot[None], input_rest_local_pos,
                parents_input, rest_offsets_input, rest_rot_input,
            )

            common_input_idx = [i for i in range(len(bone_names)) if input_to_fbx[i] >= 0]
            common_fbx_idx = [int(input_to_fbx[i]) for i in common_input_idx]

            if not common_input_idx:
                raise RuntimeError(
                    "No common bones between input skeleton and FBX armature.\n"
                    f"  Input bones: {bone_names[:10]}...\n"
                    f"  FBX bones:   {fbx_names[:10]}..."
                )

            pos_input_rest = input_rest_wpos[:, common_input_idx, :]  # (1, K, 3)
            pos_fbx_rest = fbx_rest_wpos[:, common_fbx_idx, :]         # (1, K, 3)

            def _mean_bone_len_on_common(pos, local_fbx_idx):
                """Compute mean parent→child length using FBX parent structure."""
                lengths = []
                for ci, fi in enumerate(local_fbx_idx):
                    p = fbx_parents[fi]
                    if p < 0:
                        continue
                    for ci2, fi2 in enumerate(local_fbx_idx):
                        if fi2 == p:
                            diff = pos[0, ci] - pos[0, ci2]
                            lengths.append(float(np.linalg.norm(diff)))
                            break
                return float(np.mean(lengths)) if lengths else 1.0

            mean_len_input = _mean_bone_len_on_common(pos_input_rest, common_fbx_idx)
            mean_len_fbx = _mean_bone_len_on_common(pos_fbx_rest, common_fbx_idx)

            scale = mean_len_fbx / mean_len_input if mean_len_input > 1e-8 else 1.0
            if abs(scale - 1.0) < 0.001:
                scale = 1.0

            root_fbx_idx = int(np.flatnonzero(fbx_parents == -1)[0])
            root_in_common = None
            for ci, fi in enumerate(common_fbx_idx):
                if fi == root_fbx_idx:
                    root_in_common = ci
                    break
            if root_in_common is None:
                root_in_common = 0

            t_align = pos_fbx_rest[0, root_in_common] - pos_input_rest[0, root_in_common] * scale
            pos_input_rest_st = pos_input_rest * scale + t_align[np.newaxis, np.newaxis, :]

            # Imported GLB rigs already carry the glTF wrapper/object space that
            # Blender will re-emit on export. Running the internal rest-pose
            # coordinate search here can spuriously add an extra rigid basis
            # rotation (Horse picked R_y(+90°), which re-imports as a visible
            # whole-character Z rotation). Keep GLB/GTLF retargeting in the
            # rig's existing basis and reserve the broader coordinate search
            # for FBX sources.
            if mesh_path_lower and mesh_path_lower.endswith((".glb", ".gltf")):
                candidates = [("identity", np.eye(3, dtype=np.float64))]
            else:
                candidates = _generate_coordinate_candidates_np()
            best_R = np.eye(3, dtype=np.float64)
            best_label = "identity"
            best_err = float("inf")

            for label, R in candidates:
                pos_candidate = pos_input_rest_st @ R.T
                err = float(np.mean(np.linalg.norm(pos_fbx_rest - pos_candidate, axis=-1)))
                if err < best_err:
                    best_err = err
                    best_label = label
                    best_R = R

            print(f"  [Retarget] common={len(common_input_idx)}/{len(bone_names)}, "
                  f"FBX={J_fbx} bones, alignment error={best_err:.6f}")
            print(f"  [Retarget] alignment: scale={scale:.6f}, "
                  f"rot={best_label}, "
                  f"trans=({t_align[0]:.4f}, {t_align[1]:.4f}, {t_align[2]:.4f})")

            target_wpos = np.repeat(fbx_rest_wpos, num_frames, axis=0)
            target_wrot = np.repeat(fbx_rest_wrot, num_frames, axis=0)
            aligned_input_wpos = (
                input_wpos * scale + t_align[np.newaxis, np.newaxis, :]
            ) @ best_R.T
            aligned_input_wrot = apply_rotation_to_quaternions_wxyz_np(input_wrot, best_R)

            for ii, fi in enumerate(input_to_fbx):
                if fi >= 0:
                    target_wpos[:, fi] = aligned_input_wpos[:, ii]
                    target_wrot[:, fi] = aligned_input_wrot[:, ii]

            fbx_pose_rot = np.zeros((num_frames, J_fbx, 4), dtype=np.float64)
            fbx_pose_rot[:] = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
            fbx_pose_loc = np.zeros((num_frames, J_fbx, 3), dtype=np.float64)

            root_mask = fbx_parents < 0
            root_indices = np.flatnonzero(root_mask)
            identity_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

            for j in range(J_fbx):
                parent_j = int(fbx_parents[j])
                if parent_j < 0:
                    parent_world_rot = np.repeat(identity_q[np.newaxis], num_frames, axis=0)
                    parent_world_pos = np.zeros((num_frames, 3), dtype=np.float64)
                else:
                    parent_world_rot = target_wrot[:, parent_j]
                    parent_world_pos = target_wpos[:, parent_j]

                rel_world_rot = quat_multiply_wxyz_np(
                    quat_conjugate_wxyz_np(parent_world_rot),
                    target_wrot[:, j],
                )
                fbx_pose_rot[:, j] = quat_multiply_wxyz_np(
                    quat_conjugate_wxyz_np(np.repeat(fbx_rest_rots[j:j+1], num_frames, axis=0)),
                    rel_world_rot,
                )

                rel_world_pos = target_wpos[:, j] - parent_world_pos
                rel_parent_pos = quat_rotate_wxyz_np(
                    quat_conjugate_wxyz_np(parent_world_rot),
                    rel_world_pos,
                )
                fbx_pose_loc[:, j] = quat_rotate_wxyz_np(
                    quat_conjugate_wxyz_np(np.repeat(fbx_rest_rots[j:j+1], num_frames, axis=0)),
                    rel_parent_pos - np.repeat(fbx_offsets[j:j+1], num_frames, axis=0),
                )

            jr = fbx_pose_rot.tolist()
            rr = fbx_pose_rot[:, root_indices[0], :].tolist() if len(root_indices) > 0 else rr
            rt = fbx_pose_loc[:, root_indices[0], :].tolist() if len(root_indices) > 0 else rt
            bone_names = fbx_names

            if bt is not None or np.any(np.abs(fbx_pose_loc[:, ~root_mask, :]) > 1e-6):
                bt = fbx_pose_loc.tolist()
            else:
                bt = None

            print(f"  [Retarget] Conversion complete: {num_frames} frames, {J_fbx} bones")

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

                pbone.rotation_mode = "QUATERNION"
                pbone.rotation_quaternion = (rot_val[0], rot_val[1], rot_val[2], rot_val[3])
                pbone.scale = (1.0, 1.0, 1.0)
                pbone.keyframe_insert(data_path="location", frame=f)
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
