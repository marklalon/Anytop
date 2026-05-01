"""
restore_fbx_motion.py

Take a restored BVH (already matching the T-pose skeleton in topology,
rotation, and scale) and a T-pose skinned FBX, and produce a skinned FBX
with the restored animation.

Pipeline:
  1. Import the source FBX (mesh + armature + skin weights).
  2. Import the restored BVH as an armature animation.
  3. Match bones by name and transfer the pose to the source armature frame by frame.
  4. Export the result as a skinned FBX with animation.

Usage:
    python tools/restore_fbx_motion.py --input_bvh restored.bvh --source_fbx character_tpose.fbx --output_fbx character_animated.fbx
"""

import argparse
import os
import sys

# Add Anytop and project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANYTOP_DIR = os.path.join(SCRIPT_DIR, '..')
PROJECT_ROOT = os.path.join(ANYTOP_DIR, '..')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, ANYTOP_DIR)

from motion_lib.BVH import load as bvh_load


def _patch_fbx_light_import_inline():
    """Inline copy of postprocessing.exporter._patch_fbx_light_import to avoid import chain."""
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
    def _patched(fbx_tmpl, fbx_obj, settings, _orig=original_fn):
        try:
            return _orig(fbx_tmpl, fbx_obj, settings)
        except AttributeError as exc:
            if "cast_shadow" in str(exc):
                return None
            raise
    mod.blen_read_light = _patched


def _find_matching_bone(pbone_name, bvh_bone_names):
    """Find the index of a BVH bone that matches the given pose bone name.

    Tries exact match first, then case-insensitive, then substring.
    Returns bvh_index or None.
    """
    pb_lower = pbone_name.lower().replace(" ", "_")

    # Exact match
    for i, bname in enumerate(bvh_bone_names):
        if bname.lower().replace(" ", "_") == pb_lower:
            return i

    # Substring / contains match
    for i, bname in enumerate(bvh_bone_names):
        bname_lower = bname.lower().replace(" ", "_")
        if pb_lower in bname_lower or bname_lower in pb_lower:
            return i

    return None


def _export_fbx_with_blender(restored_bvh, source_fbx, output_fbx, fps=30.0):
    """Use Blender to:
    1. Import source FBX (mesh + armature + skin)
    2. Import restored BVH as animation
    3. Transfer animation from BVH armature to source armature by bone name matching
    4. Export as FBX
    """
    import bpy
    from mathutils import Quaternion

    # Clean scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # --- Import source FBX ---
    print(f"[Blender] Importing source FBX: {source_fbx}")
    _patch_fbx_light_import_inline()

    bpy.ops.import_scene.fbx(
        filepath=source_fbx,
        ignore_leaf_bones=True,
        force_connect_children=True,
        automatic_bone_orientation=True,
        bake_space_transform=False,
        use_custom_normals=False,
        use_image_search=False,
    )

    # Remove lights and cameras
    for obj in list(bpy.data.objects):
        if obj.type in {"LIGHT", "CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)

    source_armature = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    if source_armature is None:
        print(f"[ERROR] No armature found in source FBX: {source_fbx}")
        sys.exit(1)

    source_bone_names = [b.name for b in source_armature.data.bones]
    print(f"[Blender] Source armature has {len(source_bone_names)} bones")
    if len(source_bone_names) <= 10:
        print(f"  Bones: {source_bone_names}")
    else:
        print(f"  Bones: {source_bone_names[:10]}...")

    # --- Import restored BVH as a separate armature ---
    print(f"[Blender] Importing restored BVH: {restored_bvh}")
    bpy.ops.import_anim.bvh(
        filepath=restored_bvh,
        target='ARMATURE',
    )

    bvh_armature = bpy.context.active_object
    if bvh_armature is None or bvh_armature.type != "ARMATURE":
        print(f"[ERROR] Failed to import BVH as armature")
        sys.exit(1)

    bvh_bone_names = [b.name for b in bvh_armature.data.bones]
    print(f"[Blender] BVH armature has {len(bvh_bone_names)} bones")
    if len(bvh_bone_names) <= 10:
        print(f"  Bones: {bvh_bone_names}")
    else:
        print(f"  Bones: {bvh_bone_names[:10]}...")

    # --- Compute object-level transform compensation ---
    # The source FBX armature may have a non-identity object rotation due to
    # coordinate system differences between the authoring tool and Blender.
    #
    # bvh_pbone.matrix is in Blender world space.
    # source bone.matrix_local is in source armature local space.
    # Only compensate the pure rotation (no translation, no scale) so that
    # rest_mats and desired_mats stay in the same reference frame.
    loc, rot, scale = source_armature.matrix_world.decompose()
    source_obj_inv = rot.to_matrix().to_4x4().inverted()

    # --- Load BVH data for frame-by-frame transfer ---
    bvh_anim, bvh_names, frametime = bvh_load(restored_bvh)
    num_frames = bvh_anim.shape[0]
    detected_fps = round(1.0 / frametime) if frametime else 30.0
    if abs(detected_fps - fps) > 0.5:
        print(f"  Note: BVH frametime={frametime:.4f} ({detected_fps} fps), using --fps={fps}")

    scene = bpy.context.scene
    scene.render.fps = int(fps)
    scene.frame_start = 0
    scene.frame_end = max(num_frames - 1, 0)

    # --- Setup source armature for animation ---
    source_armature.animation_data_create()
    source_armature.animation_data.action = bpy.data.actions.new(name="RestoredAnimation")
    source_armature.rotation_mode = "QUATERNION"

    # --- Enter pose mode on source armature ---
    bpy.context.view_layer.objects.active = source_armature
    source_armature.select_set(True)
    bpy.ops.object.mode_set(mode="POSE")

    # --- Also enter pose mode on BVH armature to read transforms ---
    bpy.context.view_layer.objects.active = bvh_armature
    bvh_armature.select_set(True)
    bpy.ops.object.mode_set(mode="POSE")
    bvh_armature.rotation_mode = "QUATERNION"
    bpy.context.view_layer.objects.active = source_armature
    source_armature.select_set(True)
    bpy.ops.object.mode_set(mode="POSE")

    # --- Build bone name mapping (source -> BVH index) ---
    bone_map = {}
    for sb_name in source_bone_names:
        bvh_idx = _find_matching_bone(sb_name, bvh_bone_names)
        if bvh_idx is not None:
            bone_map[sb_name] = bvh_idx
            print(f"  Mapped: {sb_name} -> BVH[{bvh_idx}] ({bvh_bone_names[bvh_idx]})")

    print(f"  Mapped {len(bone_map)}/{len(source_bone_names)} bones")
    if len(bone_map) == 0:
        print("[ERROR] No bones matched between source FBX and restored BVH!")
        sys.exit(1)

    # --- Get rest matrices for source armature ---
    rest_mats = {
        bone.name: bone.matrix_local.copy()
        for bone in source_armature.data.bones
    }

    # --- Build parent map for source armature ---
    source_bone_parent = {}
    for bone in source_armature.data.bones:
        source_bone_parent[bone.name] = bone.parent.name if bone.parent else None

    # --- Transfer animation frame by frame ---
    for f in range(num_frames):
        scene.frame_set(f)

        # Compute desired matrices in source armature local space.
        # bvh_pbone.matrix is in Blender world space.
        # Convert to source armature local space using source_obj_inv.
        desired_pose_mats = {}
        for sb_name, bvh_idx in bone_map.items():
            bvh_pbone = bvh_armature.pose.bones[bvh_bone_names[bvh_idx]]
            desired_pose_mats[sb_name] = source_obj_inv @ bvh_pbone.matrix.copy()

        # Apply to source armature pose bones
        for sb_name in source_bone_names:
            pbone = source_armature.pose.bones.get(sb_name)
            if pbone is None:
                continue

            rest_mat = rest_mats.get(sb_name)
            desired_mat = desired_pose_mats.get(sb_name)
            if rest_mat is None or desired_mat is None:
                continue

            parent_name = source_bone_parent.get(sb_name)
            if parent_name is None:
                # Root bone
                basis = rest_mat.inverted() @ desired_mat
            else:
                parent_rest = rest_mats.get(parent_name)
                parent_desired = desired_pose_mats.get(parent_name)
                if parent_rest is None or parent_desired is None:
                    basis = rest_mat.inverted() @ desired_mat
                else:
                    basis = rest_mat.inverted() @ parent_rest @ parent_desired.inverted() @ desired_mat

            loc, rot, scale = basis.decompose()
            pbone.location = loc
            pbone.rotation_mode = "QUATERNION"
            pbone.rotation_quaternion = rot if isinstance(rot, Quaternion) else rot.to_quaternion()
            pbone.scale = scale
            pbone.keyframe_insert(data_path="location", frame=f)
            pbone.keyframe_insert(data_path="rotation_quaternion", frame=f)
            pbone.keyframe_insert(data_path="scale", frame=f)

    # --- Exit pose mode ---
    bpy.ops.object.mode_set(mode="OBJECT")

    # --- Set interpolation to LINEAR ---
    action = source_armature.animation_data.action
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

    # --- Cleanup: remove BVH armature ---
    bpy.data.objects.remove(bvh_armature, do_unlink=True)

    # --- Export FBX ---
    print(f"[Blender] Exporting FBX: {output_fbx}")
    os.makedirs(os.path.dirname(output_fbx) or ".", exist_ok=True)
    bpy.ops.export_scene.fbx(
        filepath=output_fbx,
        use_selection=False,
        bake_anim=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_simplify_factor=0.0,
        add_leaf_bones=False,
    )
    print(f"[Blender] Done: {output_fbx}")


def main():
    parser = argparse.ArgumentParser(
        description='Transfer a restored BVH animation onto a T-pose skinned FBX and export as animated FBX'
    )

    parser.add_argument('--input_bvh', type=str, required=True,
                        help='Restored BVH file (bone names, topology, rotation, and scale '
                             'must already match the T-pose armature)')
    parser.add_argument('--source_fbx', type=str, required=True,
                        help='T-pose FBX file with mesh, armature, and skin weights')
    parser.add_argument('--output_fbx', type=str, required=True,
                        help='Output skinned FBX file path')
    parser.add_argument('--fps', type=float, default=30.0,
                        help='Output FPS (default: 30)')

    args = parser.parse_args()

    if not os.path.isfile(args.input_bvh):
        parser.error(f"Input BVH not found: {args.input_bvh}")
    if not os.path.isfile(args.source_fbx):
        parser.error(f"Source FBX not found: {args.source_fbx}")

    _export_fbx_with_blender(args.input_bvh, args.source_fbx, args.output_fbx, args.fps)


if __name__ == '__main__':
    main()
