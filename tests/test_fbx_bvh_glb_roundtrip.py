"""
FBX -> BVH -> GLB roundtrip test.

Loads a source FBX animation through Blender, exports the same motion as
clean standard BVH, imports that BVH back through Blender, copies the BVH
pose channels onto the target FBX rig, exports `recovered_export.glb`, and
compares the final GLB against the source FBX on every frame and every bone
in Blender world space.

Requires bpy (Blender as Python module) in the current Python environment.

Usage examples:
    # Single FBX as both T-pose and animation source:
    python tests/test_fbx_bvh_glb_roundtrip.py \
        --fbx outputs/fbx_npy_roundtrip/original.fbx

    # Separate T-pose FBX:
    python tests/test_fbx_bvh_glb_roundtrip.py \
        --tpose-fbx outputs/tpose.fbx \\
        --fbx outputs/fbx_npy_roundtrip/original.fbx

    # Custom output directory and tolerance:
    python tests/test_fbx_bvh_glb_roundtrip.py \\
        --fbx outputs/fbx_npy_roundtrip/original.fbx \\
        --output-dir outputs/my_bvh_roundtrip \\
        --tolerance 0.002
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from contextlib import nullcontext
from typing import Any

import numpy as np


# ── ensure parent of Anytop is on sys.path (so `import Anytop` works) ───────
_ANYTOP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_ANYTOP_ROOT)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _ANYTOP_ROOT not in sys.path:
    sys.path.insert(1, _ANYTOP_ROOT)

# ── Resolve utils namespace conflict ────────────────────────────────────────
import importlib.machinery
import importlib.util

_rotconv_path = os.path.join(_ANYTOP_ROOT, "utils", "rotation_conversions.py")
if os.path.isfile(_rotconv_path) and "utils.rotation_conversions" not in sys.modules:
    _loader = importlib.machinery.SourceFileLoader(
        "utils.rotation_conversions", _rotconv_path,
    )
    _spec = importlib.util.spec_from_loader(
        "utils.rotation_conversions", _loader, origin=_rotconv_path,
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["utils.rotation_conversions"] = _mod
    _spec.loader.exec_module(_mod)


from _roundtrip_common import (
    _export_animation_to_bvh,
    _fbx_to_animation,
    _get_action_frame_range,
    _get_action_sample_times,
    _measure_fbx_glb_error,
    _print_comparison_report,
    _set_scene_time,
)


# ── BVH-specific helpers ─────────────────────────────────────────────────────

def _bvh_to_glb_via_blender_bridge(
    bvh_path: str,
    tpose_fbx: str,
    output_glb: str,
    fps: float,
) -> None:
    """Import BVH in Blender, bind the FBX mesh to the BVH armature, export GLB."""
    import bpy
    from Anytop.utils.fbx import clear_scene, import_fbx, remove_lights_and_cameras

    clear_scene()
    bpy.ops.import_anim.bvh(
        filepath=bvh_path,
        rotate_mode='NATIVE',
        target='ARMATURE',
        global_scale=1.0,
        frame_start=1,
        use_fps_scale=False,
        update_scene_fps=True,
        update_scene_duration=True,
        use_cyclic=False,
    )
    source_armature = next((obj for obj in bpy.data.objects if obj.type == "ARMATURE"), None)
    if source_armature is None:
        raise RuntimeError(f"No armature found after importing BVH: {bvh_path}")
    source_armature.name = "BVH_Source_Armature"
    source_armature.data.name = "BVH_Source_Armature_Data"

    existing_armatures = {obj.name for obj in bpy.data.objects if obj.type == "ARMATURE"}
    import_fbx(tpose_fbx, ignore_leaf_bones=False)
    remove_lights_and_cameras()
    target_armature = next(
        (obj for obj in bpy.data.objects if obj.type == "ARMATURE" and obj.name not in existing_armatures),
        None,
    )
    if target_armature is None:
        raise RuntimeError(f"No target FBX armature found in {tpose_fbx}")

    scene = bpy.context.scene
    scene.render.fps = int(round(fps))
    scene.render.fps_base = 1.0
    frame_start, frame_end, _ = _get_action_frame_range(source_armature)
    scene.frame_start = frame_start
    scene.frame_end = frame_end

    mesh_objects = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    for mesh_obj in mesh_objects:
        if mesh_obj.parent == target_armature:
            mesh_obj.parent = source_armature
        for modifier in mesh_obj.modifiers:
            if modifier.type == "ARMATURE" and modifier.object == target_armature:
                modifier.object = source_armature

    bpy.data.objects.remove(target_armature, do_unlink=True)
    bpy.ops.export_scene.gltf(
        filepath=output_glb,
        export_format='GLB',
        export_animations=True,
        export_animation_mode='ACTIVE_ACTIONS',
        export_force_sampling=False,
        export_frame_range=True,
        export_apply=False,
        export_yup=True,
    )


# ── Main test function ───────────────────────────────────────────────────────

def test_fbx_bvh_glb_roundtrip(
    tpose_fbx: str,
    anim_fbx: str,
    output_dir: str | None = None,
    tolerance: float = 3e-2,
) -> dict[str, Any]:
    """FBX -> BVH -> GLB roundtrip test.

    Pipeline:
        1. Load source FBX -> source_anim, bone_names, fps
        2. Export source motion as clean standard BVH
        3. Import the BVH in Blender and transfer its pose channels to the T-pose FBX rig
        4. Export recovered GLB
        5. Compare: recovered GLB vs source FBX
    """
    for file_path in [tpose_fbx, anim_fbx]:
        assert os.path.isfile(file_path), f"Missing required file: {file_path}"

    temp_context = nullcontext(output_dir) if output_dir else tempfile.TemporaryDirectory()
    with temp_context as work_dir:
        assert work_dir is not None
        os.makedirs(work_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(anim_fbx))[0]
        recovered_glb = os.path.join(work_dir, f"{base_name}_recovered.glb")
        bvh_path = os.path.join(work_dir, f"{base_name}.bvh")

        print(f"[BVH Roundtrip] T-pose FBX : {tpose_fbx}")
        print(f"[BVH Roundtrip] Source FBX  : {anim_fbx}")
        print(f"[BVH Roundtrip] Output dir  : {work_dir}")

        # ── Phase A: Load source FBX and extract motion ──────────────
        print("  [Phase A] Loading source FBX and extracting motion...")
        source_anim, source_bone_names, source_fps = _fbx_to_animation(anim_fbx)
        print(
            f"    Frames: {len(source_anim)}, Joints: {source_anim.shape[1]}, "
            f"FPS: {source_fps:.1f}"
        )

        # ── Phase B: Export clean standard BVH ───────────────────────
        print("  [Phase B] Exporting source animation -> clean BVH...")
        _export_animation_to_bvh(
            source_anim, None, bvh_path, fps=source_fps,
            source_fbx=anim_fbx,
        )
        print(f"    Saved BVH to {bvh_path}")

        # ── Phase C: Recover GLB from the exported BVH ───────────────
        print("  [Phase C] Importing BVH and exporting recovered GLB...")
        _bvh_to_glb_via_blender_bridge(
            bvh_path=bvh_path,
            tpose_fbx=tpose_fbx,
            output_glb=recovered_glb,
            fps=source_fps,
        )

        # ── Phase D: Compare recovered GLB vs source FBX ─────────────
        recovered_metrics = _measure_fbx_glb_error(anim_fbx, recovered_glb)
        _print_comparison_report("Comparison (FBX vs recovered GLB)", recovered_metrics)

        if recovered_metrics["max_error"] >= tolerance:
            print(
                "  [WARN] FBX -> recovered GLB max error exceeds tolerance: "
                f"{recovered_metrics['max_error']:.6f} >= {tolerance:.6f} "
                f"(worst bone={recovered_metrics['worst_bone']}, "
                f"sample={recovered_metrics['worst_frame']}, "
                f"time={recovered_metrics['worst_time']:.6f})"
            )

        print("\n  DONE  BVH reconstruction completed and FBX/GLB difference was measured")
        return {
            "recovered_error": float(recovered_metrics["max_error"]),
            "recovered_worst_bone": recovered_metrics["worst_bone"],
            "recovered_worst_frame": int(recovered_metrics["worst_frame"]),
            "recovered_worst_time": float(recovered_metrics["worst_time"]),
        }


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FBX -> BVH -> GLB roundtrip test",
    )
    parser.add_argument(
        "--tpose-fbx",
        default=None,
        help=(
            "Path to T-pose FBX used as skeleton metadata and export container. "
            "Defaults to --fbx if not specified."
        ),
    )
    parser.add_argument(
        "--fbx",
        default=os.path.join(
            _ANYTOP_ROOT, "dataset", "truebones", "zoo",
            "Truebone_Z-OO", "Horse", "HorseALL-RunToStop.fbx",
        ),
        help="Path to source animation FBX file.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(_ANYTOP_ROOT, "outputs", "fbx_bvh_roundtrip"),
        help="Directory to save roundtrip artifacts.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=3e-2,
        help="Max allowed FBX -> recovered GLB error in meters (default: 3e-2).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.tpose_fbx is None:
        args.tpose_fbx = args.fbx

    print(f"T-pose FBX : {args.tpose_fbx}")
    print(f"Anim FBX   : {args.fbx}")
    print(f"Output dir : {args.output_dir}")
    print(f"Tolerance  : {args.tolerance}")
    print()

    result = test_fbx_bvh_glb_roundtrip(
        tpose_fbx=args.tpose_fbx,
        anim_fbx=args.fbx,
        output_dir=args.output_dir,
        tolerance=args.tolerance,
    )

    print("\nSummary:")
    print(
        f"  FBX -> recovered GLB       : "
        f"{result['recovered_error']:.6f} "
        f"(bone={result['recovered_worst_bone']}, "
        f"sample={result['recovered_worst_frame']}, "
        f"time={result['recovered_worst_time']:.6f})"
    )
