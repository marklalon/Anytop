"""
restore_fbx_motion.py

Take a processed or model-generated BVH and an original skinned FBX,
and produce a skinned FBX with the restored animation.

Pipeline:
  1. Call restore_bvh_bones.py (same directory) to restore original bone
     names, scale, and orientation from the BVH.
  2. Import the original FBX (mesh + armature + skin weights).
  3. Import the restored BVH as an armature animation.
  4. Match bones by name and transfer the pose to the original armature.
  5. Export the result as a skinned FBX with animation.

Usage:
    python tools/restore_fbx_motion.py --input_bvh generated.bvh --source_fbx character.fbx --output_fbx character_animated.fbx
    python tools/restore_fbx_motion.py --input_bvh generated.bvh --source_fbx character.fbx --output_fbx character_animated.fbx --object_type Hound
    python tools/restore_fbx_motion.py --input_dir bvhs/ --source_fbx character.fbx --output_dir fbx_output/
    python tools/restore_fbx_motion.py --input_bvh generated.bvh --source_fbx character.fbx --output_fbx character_animated.fbx --raw_bvh raw_character.bvh
"""

import argparse
import numpy as np
import os
import sys
import subprocess
import tempfile

# Add Anytop and project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANYTOP_DIR = os.path.join(SCRIPT_DIR, '..')
PROJECT_ROOT = os.path.join(ANYTOP_DIR, '..')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, ANYTOP_DIR)

from motion_lib.BVH import load as bvh_load
from motion_lib.Quaternions import Quaternions


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

# Default cond.npy path (same as restore_bvh_bones.py)
_DEFAULT_COND_NPY = os.path.realpath(os.path.join(
    ANYTOP_DIR,
    "dataset/truebones/zoo/truebones_processed/cond.npy"
))

# Path to sibling script
_RESTORE_BVH_SCRIPT = os.path.join(SCRIPT_DIR, 'restore_bvh_bones.py')


def _restore_bvh(input_bvh, output_bvh, cond_npy, object_type=None,
                 raw_bvh=None, scale_factor=None):
    """Call restore_bvh_bones.py as a subprocess and return the restored BVH path."""
    cmd = [
        sys.executable, _RESTORE_BVH_SCRIPT,
        '--input_bvh', input_bvh,
        '--output_bvh', output_bvh,
        '--cond_npy', cond_npy,
    ]
    if object_type:
        cmd.extend(['--object_type', object_type])
    if raw_bvh:
        cmd.extend(['--raw_bvh', raw_bvh])
    if scale_factor is not None:
        cmd.extend(['--scale_factor', str(scale_factor)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] restore_bvh_bones.py failed:")
        print(result.stderr)
        sys.exit(1)
    print(result.stdout)
    return output_bvh


def _get_object_type_from_filename(bvh_path, cond):
    """Auto-detect object_type from BVH filename (same logic as restore_bvh_bones.py)."""
    basename = os.path.splitext(os.path.basename(bvh_path))[0]
    sep = "___"
    if sep in basename:
        obj_type = basename.split(sep)[0]
        if obj_type in cond:
            return obj_type
    if "_" in basename:
        parts = basename.split("_")
        for i in range(1, len(parts)):
            candidate = "_".join(parts[:i])
            if candidate in cond:
                return candidate
    return None


def _load_cond(cond_npy):
    """Load cond.npy and return the dict."""
    return np.load(cond_npy, allow_pickle=True).item()


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
    from mathutils import Quaternion, Vector, Matrix

    # Clean scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # --- Import source FBX ---
    print(f"[Blender] Importing source FBX: {source_fbx}")
    # Inline _patch_fbx_light_import to avoid postprocessing/__init__ import chain
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
    bone_map = {}  # source_bone_name -> bvh_index
    for sb_name in source_bone_names:
        bvh_idx = _find_matching_bone(sb_name, bvh_bone_names)
        if bvh_idx is not None:
            bone_map[sb_name] = bvh_idx
            print(f"  Mapped: {sb_name} -> BVH[{bvh_idx}] ({bvh_bone_names[bvh_idx]})")
    
    print(f"  Mapped {len(bone_map)}/{len(source_bone_names)} bones")
    if len(bone_map) == 0:
        print("[ERROR] No bones matched between source FBX and restored BVH!")
        print("  Try specifying --object_type to ensure correct bone name restoration.")
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

    # --- BVH armature bone parent map ---
    bvh_bone_parent = {}
    bvh_name_to_idx = {name: i for i, name in enumerate(bvh_bone_names)}
    for i, bone in enumerate(bvh_armature.data.bones):
        if bone.parent:
            bvh_bone_parent[i] = bvh_name_to_idx.get(bone.parent.name)
        else:
            bvh_bone_parent[i] = None

    # --- Transfer animation frame by frame ---
    for f in range(num_frames):
        scene.frame_set(f)

        # Compute desired world matrices from BVH armature pose bones
        desired_pose_mats = {}
        for sb_name, bvh_idx in bone_map.items():
            bvh_pbone = bvh_armature.pose.bones[bvh_bone_names[bvh_idx]]
            desired_pose_mats[sb_name] = bvh_pbone.matrix.copy()

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


def restore_fbx_motion(input_bvh, source_fbx, output_fbx, cond_npy,
                       object_type=None, raw_bvh=None, scale_factor=None,
                       fps=30.0):
    """Main entry: restore BVH bones then export to skinned FBX."""
    # Step 1: Restore BVH (create temp file for intermediate)
    with tempfile.NamedTemporaryFile(suffix='.bvh', delete=False) as tmp:
        tmp_bvh = tmp.name
    
    try:
        print("=" * 60)
        print("Step 1: Restoring BVH bone names and scale")
        print("=" * 60)
        _restore_bvh(input_bvh, tmp_bvh, cond_npy, object_type, raw_bvh, scale_factor)
        
        print()
        print("=" * 60)
        print("Step 2: Transferring animation to source FBX")
        print("=" * 60)
        _export_fbx_with_blender(tmp_bvh, source_fbx, output_fbx, fps)
    finally:
        if os.path.exists(tmp_bvh):
            os.remove(tmp_bvh)


def main():
    parser = argparse.ArgumentParser(
        description='Restore BVH bones and export animation to a skinned FBX file'
    )

    parser.add_argument('--cond_npy', type=str, default=None,
                        help=f'Path to cond.npy (default: {_DEFAULT_COND_NPY})')

    # Single file mode
    parser.add_argument('--input_bvh', type=str, default=None,
                        help='Input BVH file (processed or model-generated)')
    parser.add_argument('--output_fbx', type=str, default=None,
                        help='Output skinned FBX file path')

    # Batch mode
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Input directory containing BVH files (batch mode)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for FBX files (batch mode)')

    # Common args
    parser.add_argument('--source_fbx', type=str, required=True,
                        help='Source FBX file with mesh, armature, and skin weights')
    parser.add_argument('--object_type', type=str, default=None,
                        help='Object type in cond.npy (e.g., "Hound"). Auto-detected if omitted.')
    parser.add_argument('--raw_bvh', type=str, default=None,
                        help='Raw BVH for scale computation (if cond.npy lacks scale_factor)')
    parser.add_argument('--scale_factor', type=float, default=None,
                        help='Explicit scale factor override')
    parser.add_argument('--fps', type=float, default=30.0,
                        help='Output FPS (default: 30)')

    args = parser.parse_args()

    # Resolve cond_npy
    cond_npy_path = args.cond_npy or _DEFAULT_COND_NPY
    if not os.path.isfile(cond_npy_path):
        parser.error(f"cond.npy not found at {cond_npy_path}. Use --cond_npy to specify.")

    if not os.path.isfile(args.source_fbx):
        parser.error(f"Source FBX not found: {args.source_fbx}")

    if args.input_bvh and args.input_dir:
        parser.error("Provide either --input_bvh or --input_dir, not both")
    if not args.input_bvh and not args.input_dir:
        parser.error("Provide either --input_bvh or --input_dir")
    if args.input_bvh and not args.output_fbx:
        parser.error("--output_fbx is required with --input_bvh")
    if args.input_dir and not args.output_dir:
        parser.error("--output_dir is required with --input_dir")

    cond = _load_cond(cond_npy_path)

    if args.input_bvh:
        # Single file mode
        if not os.path.isfile(args.input_bvh):
            parser.error(f"Input BVH not found: {args.input_bvh}")
        
        obj_type = args.object_type
        if obj_type is None:
            obj_type = _get_object_type_from_filename(args.input_bvh, cond)
            if obj_type is None:
                print(f"WARNING: Cannot detect object_type from '{os.path.basename(args.input_bvh)}'")
                print(f"  Will attempt auto-detection from FBX bone names in Blender.")
        
        restore_fbx_motion(
            args.input_bvh, args.source_fbx, args.output_fbx,
            cond_npy_path, obj_type, args.raw_bvh, args.scale_factor, args.fps
        )
    else:
        # Batch mode
        os.makedirs(args.output_dir, exist_ok=True)
        bvh_files = sorted([f for f in os.listdir(args.input_dir) if f.lower().endswith('.bvh')])
        if not bvh_files:
            print(f"No .bvh files found in {args.input_dir}")
            return
        
        for bvh_file in bvh_files:
            input_path = os.path.join(args.input_dir, bvh_file)
            output_name = os.path.splitext(bvh_file)[0] + '.fbx'
            output_path = os.path.join(args.output_dir, output_name)
            
            obj_type = _get_object_type_from_filename(input_path, cond)
            if obj_type is None:
                print(f"\n[{bvh_file}] SKIP: Cannot detect object_type")
                continue
            
            print(f"\n{'=' * 60}")
            print(f"Processing: {bvh_file}")
            print(f"{'=' * 60}")
            try:
                restore_fbx_motion(
                    input_path, args.source_fbx, output_path,
                    cond_npy_path, obj_type, args.raw_bvh, args.scale_factor, args.fps
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()


if __name__ == '__main__':
    main()
