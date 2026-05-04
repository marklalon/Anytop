"""
Animation export to GLB and BVH formats.

GLB export requires Blender (bpy) to be available in the Python
environment. BVH export delegates to the Anytop motion_lib.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from Anytop.utils.fbx import import_fbx, remove_lights_and_cameras


# ---------------------------------------------------------------------------
# Retargeting helpers (numpy-based, for world-space alignment & FBX-local
# conversion in the mesh_path GLB export path)
# ---------------------------------------------------------------------------

def _canonical_bone_name(name: str) -> str:
    """Normalize bone name for cross-format matching."""
    return name.replace(" ", "_").lower()


def _quat_multiply_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 ⊗ q2.  (..., 4) each.  Handles broadcasting."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.stack([w, x, y, z], axis=-1)


def _quat_conjugate_np(q: np.ndarray) -> np.ndarray:
    """Conjugate (w, -x, -y, -z)."""
    out = q.copy()
    out[..., 1:] *= -1
    return out


def _quat_rotate_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) v by quaternion(s) q.  q (..., 4), v (..., 3)."""
    q_conj = _quat_conjugate_np(q)
    qv = np.concatenate([np.zeros_like(v[..., :1]), v], axis=-1)  # (..., 4)
    return _quat_multiply_np(_quat_multiply_np(q, qv), q_conj)[..., 1:]


def _quat_to_matrix_np(qs: np.ndarray) -> np.ndarray:
    """(..., 4) quaternions (w,x,y,z) → (..., 3, 3) rotation matrices."""
    qw, qx, qy, qz = qs[..., 0], qs[..., 1], qs[..., 2], qs[..., 3]
    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, yy, wx = qx * x2, qy * y2, qw * x2
    xy, yz, wy = qx * y2, qy * z2, qw * y2
    xz, zz, wz = qx * z2, qz * z2, qw * z2

    mat = np.zeros(qs.shape[:-1] + (3, 3), dtype=qs.dtype)
    mat[..., 0, 0] = 1.0 - (yy + zz)
    mat[..., 0, 1] = xy - wz
    mat[..., 0, 2] = xz + wy
    mat[..., 1, 0] = xy + wz
    mat[..., 1, 1] = 1.0 - (xx + zz)
    mat[..., 1, 2] = yz - wx
    mat[..., 2, 0] = xz - wy
    mat[..., 2, 1] = yz + wx
    mat[..., 2, 2] = 1.0 - (xx + yy)
    return mat


def _matrix_to_quat_np(mats: np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotation matrices → (..., 4) quaternions (w,x,y,z)."""
    orig_shape = mats.shape[:-2]
    m = mats.reshape(-1, 3, 3)
    N = m.shape[0]
    quats = np.zeros((N, 4), dtype=mats.dtype)

    m00, m01, m02 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    m10, m11, m12 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    m20, m21, m22 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]
    trace = m00 + m11 + m22

    mask1 = trace > 0.0
    if np.any(mask1):
        s = 0.5 / np.sqrt(trace[mask1] + 1.0)
        quats[mask1, 0] = 0.25 / s
        quats[mask1, 1] = (m21[mask1] - m12[mask1]) * s
        quats[mask1, 2] = (m02[mask1] - m20[mask1]) * s
        quats[mask1, 3] = (m10[mask1] - m01[mask1]) * s

    mask2 = (~mask1) & (m00 > m11) & (m00 > m22)
    if np.any(mask2):
        s = 2.0 * np.sqrt(1.0 + m00[mask2] - m11[mask2] - m22[mask2])
        quats[mask2, 0] = (m21[mask2] - m12[mask2]) / s
        quats[mask2, 1] = 0.25 * s
        quats[mask2, 2] = (m01[mask2] + m10[mask2]) / s
        quats[mask2, 3] = (m02[mask2] + m20[mask2]) / s

    mask3 = (~mask1) & (~mask2) & (m11 > m22)
    if np.any(mask3):
        s = 2.0 * np.sqrt(1.0 + m11[mask3] - m00[mask3] - m22[mask3])
        quats[mask3, 0] = (m02[mask3] - m20[mask3]) / s
        quats[mask3, 1] = (m01[mask3] + m10[mask3]) / s
        quats[mask3, 2] = 0.25 * s
        quats[mask3, 3] = (m12[mask3] + m21[mask3]) / s

    mask4 = (~mask1) & (~mask2) & (~mask3)
    if np.any(mask4):
        s = 2.0 * np.sqrt(1.0 + m22[mask4] - m00[mask4] - m11[mask4])
        quats[mask4, 0] = (m10[mask4] - m01[mask4]) / s
        quats[mask4, 1] = (m02[mask4] + m20[mask4]) / s
        quats[mask4, 2] = (m12[mask4] + m21[mask4]) / s
        quats[mask4, 3] = 0.25 * s

    norms = np.maximum(np.linalg.norm(quats, axis=-1, keepdims=True), 1e-12)
    quats = quats / norms
    return quats.reshape(orig_shape + (4,))


def _apply_rotation_to_positions_np(positions: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Apply 3x3 rotation R to (F, J, 3) positions: pos @ R.T"""
    return positions @ R.T


def _apply_rotation_to_quaternions_np(rotations: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Apply 3x3 rotation R to (..., 4) quaternions → (..., 4)."""
    mats = _quat_to_matrix_np(rotations.astype(np.float64))  # (..., 3, 3)
    rotated_mats = R.astype(np.float64) @ mats               # R @ mats with broadcast
    return _matrix_to_quat_np(rotated_mats).astype(rotations.dtype)


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
            total_local[:, j] = _quat_multiply_np(
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
            world_pos[:, j] = world_pos[:, p] + _quat_rotate_np(
                world_rot[:, p], local_positions[:, j]
            )
            world_rot[:, j] = _quat_multiply_np(world_rot[:, p], total_local[:, j])

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
        total_local_rot = _quat_multiply_np(rest_q, pose_rotations[:, j])
        pose_loc_in_parent = rest_offsets[j:j+1] + _quat_rotate_np(rest_q, pose_locations[:, j])

        p = parents[j]
        if p < 0:
            world_pos[:, j] = pose_loc_in_parent
            world_rot[:, j] = total_local_rot
        else:
            world_pos[:, j] = world_pos[:, p] + _quat_rotate_np(world_rot[:, p], pose_loc_in_parent)
            world_rot[:, j] = _quat_multiply_np(world_rot[:, p], total_local_rot)

    return world_pos, world_rot


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


@dataclass
class InternalGlbConfig:
    """Configuration for internal GLB export (no external mesh source)."""

    render_vertices: Optional[Tensor] = None
    render_faces: Optional[Tensor] = None
    render_skin_weights: Optional[Tensor] = None
    unit_scale: float = 1.0

    @property
    def has_mesh_payload(self) -> bool:
        return all(
            x is not None
            for x in (self.render_vertices, self.render_faces, self.render_skin_weights)
        )


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
               bone_translations: Optional[Tensor] = None,
               internal_glb_config: Optional[InternalGlbConfig] = None) -> None:
        """Export animation to the format inferred from *output_path* extension.

        Args:
            joint_rotations:  [F, J, 4]  local quaternions for all joints
            root_translation: [F, 3]     world translation for root joint
            root_rotation:    [F, 4]     world rotation for root joint
            output_path:      destination file (*.glb or *.bvh)
            mesh_path:        source mesh/rig for GLB export (e.g. T-pose GLB/FBX)
            bone_translations: [F, J, 3] optional per-bone local translation.
                               Needed when non-root bones have animated local
                               positions (e.g. IK control bones in complex
                               rigs like Horse).  If None, non-root bones
                               keep their rest-pose local position.
            internal_glb_config: configuration for internal GLB export.
                                 When it has a complete mesh payload, creates both
                                 armature and skinned mesh from the internal skeleton
                                 and vertex/face/skin data instead of importing an
                                 external asset.  When *mesh_path* is provided,
                                 imports the external asset for its mesh + armature
                                 and keyframes animation on it.  When neither is
                                 provided, only the armature is exported
                                 (skeleton-only GLB, no mesh or skinning).

                                 In all cases, *unit_scale* (if set on
                                 *internal_glb_config*) is applied to the skeleton
                                 before armature creation.
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
                             bone_translations=bone_translations,
                             internal_glb_config=internal_glb_config)
        else:
            raise ValueError(f"Unsupported export format: {ext!r}")

        print(f"[Exporter] Saved animation → {output_path}")

    # ------------------------------------------------------------------
    # BVH export  (delegates to Anytop's battle-tested BVH.save)
    # ------------------------------------------------------------------

    def _export_bvh(self, joint_rotations: Tensor, root_translation: Tensor,
                    root_rotation: Tensor, output_path: str,
                    bone_translations: Optional[Tensor] = None) -> None:
        """Write a BVH file by constructing an Anytop Animation and calling BVH.save."""
        import numpy as np
        import sys
        from Anytop.kinematics import forward_kinematics
        from Anytop.utils.quaternion import quat_multiply
        _cwd = os.getcwd()
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

        base_quat = joint_rotations.detach().clone()
        base_quat[:, 0, :] = root_rotation.detach()
        rest_quat = torch.stack([b.rest_rotation for b in self.skeleton.bones], dim=0).to(
            device=base_quat.device,
            dtype=base_quat.dtype,
        )
        baked_quat = quat_multiply(
            rest_quat.unsqueeze(0).expand(F, -1, -1),
            base_quat,
        )
        baked_quat[:, 0, :] = quat_multiply(
            root_rotation.detach(),
            rest_quat[0].unsqueeze(0).expand(F, -1),
        )
        rotations = Quaternions(baked_quat.cpu().to(torch.float64).numpy())

        # ── Build positions: root always carries translation; non-root
        # ── bones get animated positions only when bone_translations is set ──
        _, joint_positions = forward_kinematics(
            joint_rotations.detach(),
            root_translation.detach(),
            root_rotation.detach(),
            self.skeleton,
        )
        has_bone_positions = bone_translations is not None
        if has_bone_positions:
            bt_np = bone_translations.detach().cpu().to(torch.float64).numpy()
            positions_np = bt_np.copy()
            positions_np[:, 0, :] = joint_positions[:, 0, :].detach().cpu().to(torch.float64).numpy()
        else:
            positions_np = np.zeros((F, J, 3), dtype=np.float64)
            positions_np[:, 0, :] = joint_positions[:, 0, :].detach().cpu().to(torch.float64).numpy()

        # ── Rest-pose attributes ────────────────────────────────────
        offsets_np = np.empty((J, 3), dtype=np.float64)
        orients_np = np.empty((J, 4), dtype=np.float64)
        parents_np = np.empty((J,), dtype=np.int32)
        for b in self.skeleton.bones:
            offsets_np[b.id] = b.rest_offset.detach().cpu().to(torch.float64).numpy()
            orients_np[b.id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            parents_np[b.id] = b.parent_id if b.parent_id is not None else -1

        offsets_np[0] = 0.0  # root offset is always zero in BVH
        orients = Quaternions(orients_np)

        anim = Animation(rotations, positions_np, orients, offsets_np, parents_np)
        bvh_save(output_path, anim, names=joint_names,
                 frametime=1.0 / self.fps, order='xyz',
                 positions=has_bone_positions,
                 all_joints_as_names=True)

    # ------------------------------------------------------------------
    # GLB export (via Blender glTF exporter)
    # ------------------------------------------------------------------

    def _export_glb(self, joint_rotations: Tensor, root_translation: Tensor,
                    root_rotation: Tensor, output_path: str,
                    mesh_path: Optional[str],
                    bone_translations: Optional[Tensor] = None,
                    internal_glb_config: Optional[InternalGlbConfig] = None) -> None:
        """Export GLB directly through bpy in the current Python process.

        When *internal_glb_config* has a complete mesh payload, creates both
        armature and skinned mesh from the internal skeleton and vertex/face/skin
        data.  When *mesh_path* is provided, imports the external asset for its
        mesh + armature and keyframes animation on it.  When neither is provided,
        only the armature is exported (skeleton-only GLB, no mesh or skinning).

        In all cases, *unit_scale* (if set on *internal_glb_config*) is applied
        to the skeleton before armature creation.
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
        export_scale = float(internal_glb_config.unit_scale) if internal_glb_config else 1.0
        export_skeleton = self._scale_skeleton_for_export(export_scale)
        use_internal_armature = bool(internal_glb_config and internal_glb_config.has_mesh_payload)
        mesh_path_lower = mesh_path.lower() if mesh_path else None
        yup = False
        if use_internal_armature:
            armature = self._create_armature_from_skeleton(bpy, skeleton=export_skeleton)
            self._create_skinned_mesh_from_payload(
                bpy=bpy,
                armature=armature,
                vertices=internal_glb_config.render_vertices * export_scale,
                faces=internal_glb_config.render_faces,
                skin_weights=internal_glb_config.render_skin_weights,
            )
        elif mesh_path:
            # 外部 mesh (FBX/GLB) 为 Y-up 坐标系，导出时需 yup=True 以保持一致
            yup = True
            if mesh_path_lower.endswith(".fbx"):
                import_fbx(mesh_path, ignore_leaf_bones=False)
            elif mesh_path_lower.endswith((".glb", ".gltf")):
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
        if mesh_path and not use_internal_armature:
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

            input_local_rot_np = jr_np.copy()
            for j in range(len(bone_names)):
                if self.skeleton.bones[j].parent_id is None:
                    input_local_rot_np[:, j] = rr_np

            # Pose-bone translation deltas.
            local_pos_np = np.zeros((num_frames, len(bone_names), 3), dtype=np.float64)
            for j in range(len(bone_names)):
                if self.skeleton.bones[j].parent_id is None:
                    local_pos_np[:, j] = rt_np
                elif bt is not None:
                    local_pos_np[:, j] = np.array([bt[f][j] for f in range(num_frames)], dtype=np.float64)
                else:
                    local_pos_np[:, j] = 0.0

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

            input_wpos, input_wrot = _batch_pose_fk_np(
                input_local_rot_np, local_pos_np, parents_input,
                rest_offsets_input, rest_rot_input,
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

            candidates = _generate_coordinate_candidates_np()
            best_R = np.eye(3, dtype=np.float64)
            best_label = "identity"
            best_err = float("inf")

            for label, R in candidates:
                pos_candidate = _apply_rotation_to_positions_np(pos_input_rest_st, R)
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
            aligned_input_wpos = _apply_rotation_to_positions_np(
                input_wpos * scale + t_align[np.newaxis, np.newaxis, :],
                best_R,
            )
            aligned_input_wrot = _apply_rotation_to_quaternions_np(input_wrot, best_R)

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

                rel_world_rot = _quat_multiply_np(
                    _quat_conjugate_np(parent_world_rot),
                    target_wrot[:, j],
                )
                fbx_pose_rot[:, j] = _quat_multiply_np(
                    _quat_conjugate_np(np.repeat(fbx_rest_rots[j:j+1], num_frames, axis=0)),
                    rel_world_rot,
                )

                rel_world_pos = target_wpos[:, j] - parent_world_pos
                rel_parent_pos = _quat_rotate_np(
                    _quat_conjugate_np(parent_world_rot),
                    rel_world_pos,
                )
                fbx_pose_loc[:, j] = _quat_rotate_np(
                    _quat_conjugate_np(np.repeat(fbx_rest_rots[j:j+1], num_frames, axis=0)),
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
            if use_internal_armature:
                loc_val = [export_scale * value for value in rt[f]]
                rot_val = rr[f]
                armature.location = (loc_val[0], loc_val[1], loc_val[2])
                armature.rotation_quaternion = (rot_val[0], rot_val[1], rot_val[2], rot_val[3])
                armature.keyframe_insert(data_path="location", frame=f)
                armature.keyframe_insert(data_path="rotation_quaternion", frame=f)
            for j, bname in enumerate(bone_names):
                pbone = armature.pose.bones.get(bname)
                if pbone is None:
                    continue

                if mesh_path and not use_internal_armature:
                    # After retargeting, bones are FBX bones and parent comes from FBX
                    parent_id = fbx_parents[j] if j < len(fbx_names) else -1
                else:
                    parent_id = self.skeleton.bones[j].parent_id if self.skeleton.bones[j].parent_id is not None else -1
                is_root = parent_id < 0
                if use_internal_armature:
                    # The armature object carries the root transform;
                    # the root pose bone must be identity to avoid double-applying.
                    if is_root:
                        rot_val = [1.0, 0.0, 0.0, 0.0]  # identity — armature handles root
                        pbone.location = (0.0, 0.0, 0.0)
                    else:
                        rot_val = jr[f][j]
                        if bt is not None:
                            loc_val = bt[f][j]
                            pbone.location = (loc_val[0], loc_val[1], loc_val[2])
                        else:
                            pbone.location = (0.0, 0.0, 0.0)
                elif is_root:
                    loc_val = rt[f]
                    rot_val = rr[f]
                    pbone.location = (loc_val[0], loc_val[1], loc_val[2])
                else:
                    rot_val = jr[f][j]
                    if bt is not None:
                        loc_val = bt[f][j]
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
        bpy.ops.export_scene.gltf(
            filepath=output_path,
            export_format='GLB',
            export_animations=True,
            export_animation_mode='ACTIVE_ACTIONS',
            export_force_sampling=False,
            export_frame_range=True,
            export_apply=False,
            export_yup=yup,
        )

    def _create_skinned_mesh_from_payload(
        self,
        bpy,
        armature,
        vertices: Tensor,
        faces: Tensor,
        skin_weights: Tensor,
    ):
        scene = bpy.context.scene
        mesh_data = bpy.data.meshes.new("PCVGMesh")
        vertex_array = vertices.detach().cpu().numpy()
        face_array = faces.detach().cpu().numpy()
        mesh_data.from_pydata(
            [tuple(float(v) for v in vertex) for vertex in vertex_array],
            [],
            [tuple(int(index) for index in face) for face in face_array],
        )
        mesh_data.update()

        mesh_obj = bpy.data.objects.new("PCVGMesh", mesh_data)
        scene.collection.objects.link(mesh_obj)

        weight_array = skin_weights.detach().cpu().numpy()
        vertex_groups = [mesh_obj.vertex_groups.new(name=bone.name) for bone in self.skeleton.bones]
        for vertex_idx, weights in enumerate(weight_array):
            nonzero_indices = np.nonzero(weights > 1e-8)[0]
            for bone_idx in nonzero_indices:
                vertex_groups[int(bone_idx)].add([vertex_idx], float(weights[bone_idx]), "REPLACE")

        modifier = mesh_obj.modifiers.new(name="Armature", type="ARMATURE")
        modifier.object = armature
        mesh_obj.parent = armature
        return mesh_obj

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

    def _scale_skeleton_for_export(self, unit_scale: float):
        from Anytop.kinematics.skeleton import Bone, Skeleton

        if abs(unit_scale - 1.0) < 1e-8:
            return self.skeleton

        scaled_bones = []
        for bone in self.skeleton.bones:
            scaled_bones.append(Bone(
                id=bone.id,
                name=bone.name,
                parent_id=bone.parent_id,
                rest_offset=bone.rest_offset.detach().clone() * unit_scale,
                rest_rotation=bone.rest_rotation.detach().clone(),
            ))
        return Skeleton(scaled_bones)
