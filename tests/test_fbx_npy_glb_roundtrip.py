"""
FBX -> NPY -> GLB roundtrip test.

Loads a source FBX animation through Blender, extracts AnyTop's 13-channel
NPY motion features, recovers an Animation, exports `recovered.glb`, and
compares the final GLB directly against the source FBX on every frame and
every bone in Blender world space.

Requires bpy (Blender as Python module) in the current Python environment.

Usage examples:
    # Use a single FBX as both T-pose source and animation source:
    python tests/test_fbx_npy_glb_roundtrip.py \
        --fbx outputs/fbx_npy_roundtrip/original.fbx \
        --object-type Horse

    # Specify a separate T-pose FBX for skeleton metadata:
    python tests/test_fbx_npy_glb_roundtrip.py \
        --tpose-fbx outputs/tpose.fbx \
        --fbx outputs/fbx_npy_roundtrip/original.fbx \
        --object-type Horse

    # Custom output directory and tolerance:
    python tests/test_fbx_npy_glb_roundtrip.py \
        --fbx outputs/fbx_npy_roundtrip/original.fbx \\
        --output-dir outputs/my_roundtrip \\
        --tolerance 0.05
"""
from __future__ import annotations

import argparse
import os
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

_npy_rt_path = os.path.join(_ANYTOP_ROOT, "utils", "npy_roundtrip_utils.py")
if os.path.isfile(_npy_rt_path) and "utils.npy_roundtrip_utils" not in sys.modules:
    _loader = importlib.machinery.SourceFileLoader(
        "utils.npy_roundtrip_utils", _npy_rt_path,
    )
    _spec = importlib.util.spec_from_loader(
        "utils.npy_roundtrip_utils", _loader, origin=_npy_rt_path,
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["utils.npy_roundtrip_utils"] = _mod
    _spec.loader.exec_module(_mod)

if _ANYTOP_ROOT not in sys.path:
    sys.path.insert(1, _ANYTOP_ROOT)


from utils.npy_roundtrip_utils import (
    build_roundtrip_feature_payload,
    coerce_feature_payload,
    recover_from_features,
    extract_raw_features,
    compute_rest_positions,
    get_cont6d_params_own,
    detect_motion_loop,
    compute_terminal_local_velocity,
)
from Anytop.utils.fbx import clear_scene, import_fbx, remove_lights_and_cameras


from _roundtrip_common import (
    _build_skeleton,
    _fbx_to_animation,
    _export_animation_to_glb,
    _load_fbx_scene,
    _load_glb_scene,
    _load_fbx_skeleton_metadata,
)

from tools.compare_motions import (
    _load_motion,
    _validate_compatible,
    _detect_and_align,
    _compare_motions,
    _print_summary,
)


# ── Main test function ───────────────────────────────────────────────────────

def _pick_primary_mesh():
    import bpy

    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("No mesh found in scene")

    return max(meshes, key=lambda obj: len(obj.data.polygons))


def _sample_world_mesh_points(mesh_obj, frame_idx: int) -> np.ndarray:
    import bpy

    scene = bpy.context.scene
    scene.frame_set(frame_idx)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = mesh_obj.evaluated_get(depsgraph)
    mesh_eval = obj_eval.to_mesh()
    matrix_world = np.array(obj_eval.matrix_world, dtype=np.float64)
    try:
        points = np.array([
            (matrix_world @ np.array([vertex.co[0], vertex.co[1], vertex.co[2], 1.0], dtype=np.float64))[:3]
            for vertex in mesh_eval.vertices
        ], dtype=np.float64)
    finally:
        obj_eval.to_mesh_clear()
    return points


def _nearest_surface_stats(points_a: np.ndarray, points_b: np.ndarray) -> dict[str, float]:
    from mathutils import kdtree

    def _one_way(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        tree = kdtree.KDTree(len(dst))
        for idx, point in enumerate(dst):
            tree.insert(tuple(float(v) for v in point), idx)
        tree.balance()

        distances = np.empty((len(src),), dtype=np.float64)
        for idx, point in enumerate(src):
            _co, _dst_idx, dist = tree.find(tuple(float(v) for v in point))
            distances[idx] = dist
        return distances

    a_to_b = _one_way(points_a, points_b)
    b_to_a = _one_way(points_b, points_a)
    both = np.concatenate([a_to_b, b_to_a], axis=0)
    return {
        "mean": float(both.mean()),
        "max": float(both.max()),
        "p99": float(np.quantile(both, 0.99)),
    }


def _compare_mesh_surfaces(source_fbx: str, recovered_glb: str, alignment, sample_frames: list[int]) -> dict[str, Any]:
    _load_fbx_scene(source_fbx)
    source_mesh = _pick_primary_mesh()
    source_samples = {
        frame_idx: _sample_world_mesh_points(source_mesh, frame_idx)
        for frame_idx in sample_frames
    }

    _load_glb_scene(recovered_glb)
    recovered_mesh = _pick_primary_mesh()

    rotation = np.asarray(alignment.rotation_matrix, dtype=np.float64)
    translation = np.asarray(alignment.translation_offset, dtype=np.float64)
    scale = float(alignment.scale)

    per_frame = []
    for frame_idx in sample_frames:
        recovered_points = _sample_world_mesh_points(recovered_mesh, frame_idx)
        aligned_recovered = (scale * recovered_points + translation[np.newaxis, :]) @ rotation.T
        stats = _nearest_surface_stats(source_samples[frame_idx], aligned_recovered)
        stats["frame"] = int(frame_idx)
        per_frame.append(stats)

    return {
        "per_frame": per_frame,
        "mean": float(np.mean([item["mean"] for item in per_frame])),
        "max": float(np.max([item["max"] for item in per_frame])),
        "p99": float(np.max([item["p99"] for item in per_frame])),
    }

def test_fbx_npy_glb_roundtrip(
    tpose_fbx: str,
    anim_fbx: str,
    object_type: str = "Alligator",
    output_dir: str | None = None,
    tolerance: float = 0.05,
) -> dict[str, Any]:
    """FBX -> NPY -> GLB roundtrip test.

    Pipeline:
        1. Load source FBX, extract animation
        2. Extract NPY features from the animation
        3. Recover Animation from NPY features
        4. Export recovered animation as GLB
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
        npy_path = os.path.join(work_dir, f"{base_name}_features.npy")

        # Phase A: Load T-pose FBX for skeleton metadata
        print("[Phase A] Loading T-pose FBX for skeleton metadata...")
        tpose_meta_bone_names, parents, offsets, rest_rotations = _load_fbx_skeleton_metadata(tpose_fbx)
        tpose_anim, tpose_bone_names, tpose_fps = _fbx_to_animation(tpose_fbx)
        print(f"Joints: {len(tpose_bone_names)}, FPS: {tpose_fps:.1f}")
        if tpose_meta_bone_names != tpose_bone_names:
            raise AssertionError("T-pose FBX animation bone order does not match extracted skeleton metadata")

        tpose_skeleton = _build_skeleton(tpose_bone_names, offsets, parents, rest_rotations)

        # Phase B: Load source FBX and extract motion
        print("[Phase B] Loading source FBX and extracting motion...")
        source_anim, source_bone_names, source_fps = _fbx_to_animation(anim_fbx)
        print(f"Frames: {len(source_anim)}, Joints: {source_anim.shape[1]}, FPS: {source_fps:.1f}")

        if source_bone_names != tpose_bone_names:
            raise AssertionError(
                "Source FBX and T-pose FBX do not share the same BFS bone order"
            )

        # Phase C: Extract raw NPY features
        print("[Phase C] Extracting raw NPY features...")
        feature_payload = build_roundtrip_feature_payload(
            source_anim, object_type, offsets, parents, source_bone_names,
        )
        np.save(npy_path, feature_payload, allow_pickle=True)
        print(f"  NPY shape: {feature_payload['features'].shape}")
        print(f"Saved NPY features to {npy_path}")

        # Phase D: Recover Animation from NPY features
        print("[Phase D] Recovering Animation from NPY features...")
        recovered_anim, has_animated_pos = recover_from_features(
            feature_payload, parents, offsets,
        )
        print(f"  Recovered frames: {len(recovered_anim)}")
        if has_animated_pos:
            print("(has non-root animated position channels)")

        from motion_lib.Animation import positions_global

        source_global = positions_global(source_anim)
        recovered_global = positions_global(recovered_anim)
        npy_position_error = np.abs(source_global - recovered_global).max(axis=(0, 2))
        npy_worst_idx = int(np.argmax(npy_position_error))
        npy_worst_bone = source_bone_names[npy_worst_idx] if npy_worst_idx < len(source_bone_names) else "?"
        print(
            "[Diag] Animation-domain source-vs-recovered max per-joint error: "
            f"{npy_position_error.max():.6f} ({npy_worst_bone})"
        )
        print("Note: this is diagnostic only because recovery is built on the T-pose FBX skeleton.")

        # Phase E: Export NPY-recovered animation -> recovered.glb
        print(f"[Phase E] Exporting NPY-recovered animation -> {os.path.basename(recovered_glb)}...")
        _export_animation_to_glb(
            recovered_anim,
            tpose_skeleton,
            recovered_glb,
            mesh_path=tpose_fbx,
            fps=source_fps,
        )

        # Phase F: Compare recovered GLB vs source FBX using compare_motions.py
        print("[Phase F] Comparing recovered GLB vs source FBX via compare_motions...")
        motion_a = _load_motion(anim_fbx)
        motion_b = _load_motion(recovered_glb)
        _validate_compatible(motion_a, motion_b)
        motion_b_aligned, _alignment = _detect_and_align(motion_a, motion_b)
        result = _compare_motions(motion_a, motion_b_aligned, _alignment)
        print(f"{'Compare Motions':=^{70}}")
        _print_summary(motion_a, motion_b, _alignment, result)

        sample_frames = sorted({0, max(motion_a.num_frames // 2, 0), max(motion_a.num_frames - 1, 0)})
        mesh_result = _compare_mesh_surfaces(anim_fbx, recovered_glb, _alignment, sample_frames)
        char_size = max(float(result["position"]["character_size"]), 1e-8)
        mesh_mean_pct_char = mesh_result["mean"] / char_size * 100.0
        mesh_p99_pct_char = mesh_result["p99"] / char_size * 100.0
        print(
            "[Mesh] nearest-surface error: "
            f"mean={mesh_result['mean']:.6f} ({mesh_mean_pct_char:.4f}%), "
            f"p99={mesh_result['p99']:.6f} ({mesh_p99_pct_char:.4f}%), "
            f"max={mesh_result['max']:.6f}"
        )

        pos_result = result["position"]
        assert pos_result["max_error"] < tolerance, (
            f"FBX -> recovered GLB max error {pos_result['max_error']:.6f} exceeds "
            f"{tolerance} (worst bone={pos_result['worst_bone']}, "
            f"frame={pos_result['worst_frame']})"
        )
        assert mesh_mean_pct_char < 3.5, (
            f"FBX -> recovered GLB mesh mean surface error {mesh_result['mean']:.6f} "
            f"({mesh_mean_pct_char:.4f}% of character size) exceeds 3.5%"
        )
        assert mesh_p99_pct_char < 12.0, (
            f"FBX -> recovered GLB mesh p99 surface error {mesh_result['p99']:.6f} "
            f"({mesh_p99_pct_char:.4f}% of character size) exceeds 12.0%"
        )

        # Map worst_frame to sample time for the return dict
        worst_frame = pos_result["worst_frame"]
        recovered_worst_time = float(motion_a.sample_times[worst_frame]) \
            if worst_frame < len(motion_a.sample_times) else 0.0

        print("\nPASS  FBX -> NPY -> GLB roundtrip checks passed")
        return {
            "npy_error": float(npy_position_error.max()),
            "npy_worst_bone": npy_worst_bone,
            "npy_worst_frame": npy_worst_idx,
            "recovered_error": float(pos_result["max_error"]),
            "recovered_error_pct_char": float(pos_result["max_error_pct_char"]),
            "recovered_worst_bone": pos_result["worst_bone"],
            "recovered_worst_frame": int(worst_frame),
            "recovered_worst_time": recovered_worst_time,
            "mesh_mean_error": float(mesh_result["mean"]),
            "mesh_p99_error": float(mesh_result["p99"]),
            "rot_max": float(result["rotation"]["max_error_deg"]),
            "rot_worst_bone": result["rotation"]["worst_bone"],
            "rot_worst_frame": int(result["rotation"]["worst_frame"]),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FBX -> NPY -> GLB roundtrip smoke test",
    )
    parser.add_argument(
        "--tpose-fbx",
        default=None,
        help="Path to T-pose FBX file used as skeleton metadata and export container. Defaults to --fbx if not specified.",
    )
    parser.add_argument(
        "--fbx",
        default=os.path.join(_ANYTOP_ROOT, "dataset", "truebones", "zoo", "Truebone_Z-OO", "Horse", "HorseALL-RunToStop.fbx"),
        help="Path to source animation FBX file.",
    )
    parser.add_argument(
        "--object-type",
        default=None,
        help="Character type for contact inference. Inferred from the FBX filename (first segment before '_') if not specified.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(_ANYTOP_ROOT, "outputs", "fbx_npy_roundtrip"),
        help="Directory to save roundtrip artifacts.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Max allowed comparison error in meters (default: 0.05).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.tpose_fbx is None:
        args.tpose_fbx = args.fbx

    if args.object_type is None:
        args.object_type = os.path.splitext(os.path.basename(args.fbx))[0].split("_", 1)[0]

    print(f"T-pose FBX : {args.tpose_fbx}")
    print(f"Anim FBX   : {args.fbx}")
    print(f"Output dir : {args.output_dir}")
    print(f"Object type: {args.object_type}")
    print(f"Tolerance  : {args.tolerance}")
    print()

    result = test_fbx_npy_glb_roundtrip(
        tpose_fbx=args.tpose_fbx,
        anim_fbx=args.fbx,
        object_type=args.object_type,
        output_dir=args.output_dir,
        tolerance=args.tolerance,
    )

    print("\nSummary:")
    print(f"  NPY encoding error   : {result['npy_error']:.6f}  "
          f"(bone={result['npy_worst_bone']}, frame={result['npy_worst_frame']})")
    print(f"  FBX -> recovered GLB : pos_max={result['recovered_error']:.6f} ({result['recovered_error_pct_char']:.4f}%) "
          f"(bone={result['recovered_worst_bone']}, frame={result['recovered_worst_frame']}, "
          f"time={result['recovered_worst_time']:.6f})")
    print(f"  Mesh surface error   : mean={result['mesh_mean_error']:.6f}  p99={result['mesh_p99_error']:.6f}")
    print(f"  Rotation error       : max={result['rot_max']:.6f}°  "
          f"(bone={result['rot_worst_bone']}, frame={result['rot_worst_frame']})")
