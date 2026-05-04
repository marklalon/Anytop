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
        else:
            armature = self._create_armature_from_skeleton(bpy, skeleton=export_skeleton)

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

                parent_id = self.skeleton.bones[j].parent_id
                if use_internal_armature:
                    rot_val = jr[f][j]
                    pbone.location = (0.0, 0.0, 0.0)
                elif parent_id is None:
                    loc_val = rt[f]
                    rot_val = rr[f]
                    pbone.location = (loc_val[0], loc_val[1], loc_val[2])
                else:
                    rot_val = jr[f][j]
                    if bt is not None:
                        loc_val = bt[f][j]
                        pbone.location = (loc_val[0], loc_val[1], loc_val[2])

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
