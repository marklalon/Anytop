"""
End-to-end exporter test against a source GLB.

The test loads one GLB animation, converts it into AnimationExporter inputs,
exports three variants via utils.exporter.AnimationExporter, then uses
tools.compare_motions to compare each exported file back to the source GLB.

Pass criteria:
  - bones-only GLB vs source passes compare_motions thresholds
  - skinned GLB vs source passes compare_motions thresholds
  - bones-only GLB is verified to contain no skinned mesh
  - skinned GLB is verified to contain a skinned mesh and passes mesh checks
  - BVH vs source passes compare_motions thresholds, if --no-skip-bvh is set

Usage:
        python tests/test_exporter.py

        python tests/test_exporter.py \
        --fbx dataset/truebones/zoo/Truebone_Z-OO/Horse/HorseALL-RunToStop.glb \
        --output-dir outputs/exporter_compare_motions
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch


_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ANYTOP_ROOT = os.path.dirname(_TESTS_DIR)
_REPO_ROOT = os.path.dirname(_ANYTOP_ROOT)

for _path in [_REPO_ROOT, _ANYTOP_ROOT]:
    if _path not in sys.path:
        sys.path.insert(0, _path)


from utils.roundtrip_common import build_skeleton, load_fbx_skeleton_metadata
from motion_lib import FBX
from motion_lib.FBX import fbx_to_animation
from utils.exporter import AnimationExporter, animation_to_exporter_inputs
from tools.compare_motions import compare_motions, compute_mesh_surface_error, detect_and_align, load_motion, print_summary


_DEFAULT_GLB = os.path.join(
    _ANYTOP_ROOT,
    "dataset",
    "truebones",
    "zoo",
    "Truebone_Z-OO",
    "Horse",
    "HorseALL-RunToStop.glb",
)
_DEFAULT_FBX = _DEFAULT_GLB


_COLOR_RED = "\033[31m"
_COLOR_RESET = "\033[0m"


def _fmt_path(path: str) -> str:
    if not os.path.isabs(path):
        return path
    try:
        return os.path.relpath(path, _REPO_ROOT)
    except ValueError:
        return path


def _assert_compatible(label: str, motion_a, motion_b) -> None:
    fps_tol = 0.001 * max(motion_a.fps, motion_b.fps)
    assert abs(motion_a.fps - motion_b.fps) <= fps_tol, (
        f"{label}: FPS mismatch: A={motion_a.fps:.6f}, B={motion_b.fps:.6f}"
    )
    assert motion_a.num_frames == motion_b.num_frames, (
        f"{label}: frame count mismatch: A={motion_a.num_frames}, B={motion_b.num_frames}"
    )


def _export_variants_from_fbx(fbx_path: str, work_dir: str, skip_bvh: bool = False) -> dict[str, Any]:
    meta_bone_names, parents, offsets, rest_rotations = load_fbx_skeleton_metadata(fbx_path)
    animation, anim_bone_names, fps = fbx_to_animation(fbx_path)
    assert meta_bone_names == anim_bone_names, (
        "FBX animation bone order does not match extracted skeleton metadata"
    )

    skeleton = build_skeleton(meta_bone_names, offsets, parents, rest_rotations)
    exporter = AnimationExporter(skeleton, fps=fps)

    joint_rotations, root_translation, root_rotation, bone_translations = (
        animation_to_exporter_inputs(animation, skeleton)
    )

    base_name = os.path.splitext(os.path.basename(fbx_path))[0]
    outputs = {
        "FBX": os.path.abspath(fbx_path),
        "BonesOnlyGLB": os.path.join(work_dir, f"{base_name}_bones_only.glb"),
        "SkinnedGLB": os.path.join(work_dir, f"{base_name}_skinned.glb"),
        "fps": float(fps),
        "num_frames": int(animation.shape[0]),
        "num_joints": int(animation.shape[1]),
    }

    if not skip_bvh:
        outputs["BVH"] = os.path.join(work_dir, f"{base_name}_exported.bvh")
        exporter.export_bvh(
            joint_rotations,
            root_translation,
            root_rotation,
            outputs["BVH"],
            bone_translations=bone_translations,
        )
    exporter.export_glb(
        joint_rotations,
        root_translation,
        root_rotation,
        outputs["BonesOnlyGLB"],
        bone_translations=bone_translations,
    )
    exporter.export_glb(
        joint_rotations,
        root_translation,
        root_rotation,
        outputs["SkinnedGLB"],
        mesh_path=fbx_path,
        bone_translations=bone_translations,
    )
    return outputs


def _assert_loaded_animation_matches(
    source_path: str,
    converted_path: str,
    *,
    offset_atol_pct: float = 0.01,
    position_atol_pct: float = 0.01,
    rotation_dot_atol: float = 1e-4,
    frame_time_atol: float = 1e-4,
) -> None:
    """Assert FBX.load returns identical data for the same animation in different formats.

    Compares raw return values of FBX.load (offsets, rest rotations, pose rotations,
    local positions) to verify that loading the same animation from FBX vs GLB
    produces consistent results.

    Does NOT compute world-space FK, because the goal is to verify that the
    FBX.load loader itself is format-agnostic, not to verify FK correctness.

    Args:
        offset_atol_pct: max allowed offset error as a percentage of the source offset magnitude.
        position_atol_pct: max allowed position error as a percentage of the source position magnitude.
        rotation_dot_atol: max allowed deviation from 1.0 for quaternion dot products
            (both rest rotations and pose rotations).
        frame_time_atol: max allowed absolute error for frame time (seconds per frame).
    """
    source_anim, source_names, source_frame_time = FBX.load(source_path)
    converted_anim, converted_names, converted_frame_time = FBX.load(converted_path)

    assert source_names == converted_names
    assert source_anim.parents.tolist() == converted_anim.parents.tolist()
    assert source_anim.shape == converted_anim.shape
    assert abs(float(source_frame_time) - float(converted_frame_time)) < frame_time_atol

    # ── Rest offsets (relative error) ──
    source_offsets = np.asarray(source_anim.offsets, dtype=np.float64)
    converted_offsets = np.asarray(converted_anim.offsets, dtype=np.float64)
    offset_diff = np.abs(source_offsets - converted_offsets)
    offset_magnitude = np.max(np.abs(source_offsets))
    if offset_magnitude > 0:
        offset_pct = np.max(offset_diff) / offset_magnitude * 100.0
    else:
        offset_pct = 0.0 if np.max(offset_diff) == 0 else float("inf")
    assert offset_pct <= offset_atol_pct, (
        f"Rest offset mismatch: max_abs_diff={np.max(offset_diff):.6e}, "
        f"source_magnitude={offset_magnitude:.6e}, "
        f"error={offset_pct:.4f}%, atol={offset_atol_pct:.4f}%"
    )

    # ── Rest rotations (orient) ──
    source_orients = np.asarray(source_anim.orients.qs, dtype=np.float64)
    converted_orients = np.asarray(converted_anim.orients.qs, dtype=np.float64)
    orient_dots = np.abs(np.sum(source_orients * converted_orients, axis=-1))
    orient_min = np.min(orient_dots)
    assert orient_min >= 1.0 - rotation_dot_atol, (
        f"Rest rotation mismatch: min_quat_dot={orient_min:.10f}, atol={rotation_dot_atol:.6e}"
    )

    # ── Pose rotations (local quaternion per frame) ──
    source_rots = np.asarray(source_anim.rotations.qs, dtype=np.float64)
    converted_rots = np.asarray(converted_anim.rotations.qs, dtype=np.float64)
    rot_dots = np.abs(np.sum(source_rots * converted_rots, axis=-1))
    rot_min = np.min(rot_dots)
    assert rot_min >= 1.0 - rotation_dot_atol, (
        f"Pose rotation mismatch: min_quat_dot={rot_min:.10f}, atol={rotation_dot_atol:.6e}"
    )

    # ── Local positions (relative error) ──
    source_positions = np.asarray(source_anim.positions, dtype=np.float64)
    converted_positions = np.asarray(converted_anim.positions, dtype=np.float64)
    pos_diff = np.abs(source_positions - converted_positions)
    pos_magnitude = np.max(np.abs(source_positions))
    if pos_magnitude > 0:
        pos_pct = np.max(pos_diff) / pos_magnitude * 100.0
    else:
        pos_pct = 0.0 if np.max(pos_diff) == 0 else float("inf")
    assert pos_pct <= position_atol_pct, (
        f"Local position mismatch: max_abs_diff={np.max(pos_diff):.6e}, "
        f"source_magnitude={pos_magnitude:.6e}, "
        f"error={pos_pct:.4f}%, atol={position_atol_pct:.4f}%"
    )


def _compare_export_to_fbx(
    source_fbx: str,
    exported_path: str,
    label: str,
    pos_tolerance_pct_char: float,
    rot_tolerance_deg: float,
    mesh_mean_tolerance_pct_char: float,
    mesh_p99_tolerance_pct_char: float,
) -> dict[str, Any]:
    motion_a = load_motion(source_fbx)
    motion_b = load_motion(exported_path)

    _assert_compatible(label, motion_a, motion_b)

    # Mesh presence checks
    errors: list[str] = []
    if label == "BonesOnlyGLB" and motion_b.has_skinned_mesh:
        errors.append(f"{label}: expected skeleton-only GLB without skinned mesh")
    if label == "SkinnedGLB" and not motion_b.has_skinned_mesh:
        errors.append(f"{label}: expected exported GLB to include skinned mesh")

    motion_b_aligned, alignment = detect_and_align(motion_a, motion_b)
    result = compare_motions(motion_a, motion_b_aligned, alignment)

    print(f"\n{'=' * 78}")
    print(f"[Compare] FBX vs {label}: {_fmt_path(exported_path)}")
    print(f"{'=' * 78}")
    print_summary(motion_a, motion_b, alignment, result)

    pos_result = result["position"]
    rot_result = result["rotation"]
    world_quat_result = result["world_quaternion"]

    # Position check
    if pos_result["max_error_pct_char"] > pos_tolerance_pct_char:
        errors.append(
            f"{label}: position max error {pos_result['max_error']:.6f} "
            f"({pos_result['max_error_pct_char']:.4f}% char) exceeds "
            f"{pos_tolerance_pct_char:.4f}% on bone={pos_result['worst_bone']} "
            f"frame={pos_result['worst_frame']}"
        )

    # Rotation check
    if rot_result["max_error_deg"] > rot_tolerance_deg:
        errors.append(
            f"{label}: rotation max error {rot_result['max_error_deg']:.6f} deg exceeds "
            f"{rot_tolerance_deg:.6f} on bone={rot_result['worst_bone']} "
            f"frame={rot_result['worst_frame']}"
        )

    # World quaternion check
    if world_quat_result["max_error_deg"] > rot_tolerance_deg:
        errors.append(
            f"{label}: world quaternion max error {world_quat_result['max_error_deg']:.6f} deg exceeds "
            f"{rot_tolerance_deg:.6f} on bone={world_quat_result['worst_bone']} "
            f"frame={world_quat_result['worst_frame']}"
        )

    # Mesh surface checks
    mesh_result = None
    mesh_mean_pct_char = None
    mesh_p99_pct_char = None
    if motion_a.has_skinned_mesh and motion_b.has_skinned_mesh:
        mesh_result = compute_mesh_surface_error(motion_a, motion_b_aligned, alignment)
        if mesh_result is None:
            errors.append(f"{label}: expected mesh surface stats")
        else:
            char_size = max(float(pos_result["character_size"]), 1e-8)
            mesh_mean_pct_char = mesh_result["mean"] / char_size * 100.0
            mesh_p99_pct_char = mesh_result["p99"] / char_size * 100.0

            if mesh_mean_pct_char > mesh_mean_tolerance_pct_char:
                errors.append(
                    f"{label}: mesh mean surface error {mesh_result['mean']:.6f} "
                    f"({mesh_mean_pct_char:.4f}% char) exceeds {mesh_mean_tolerance_pct_char:.4f}%"
                )
            if mesh_p99_pct_char > mesh_p99_tolerance_pct_char:
                errors.append(
                    f"{label}: mesh p99 surface error {mesh_result['p99']:.6f} "
                    f"({mesh_p99_pct_char:.4f}% char) exceeds {mesh_p99_tolerance_pct_char:.4f}%"
                )

    # Print PASS/FAIL
    passed = len(errors) == 0
    status = "PASS" if passed else f"{_COLOR_RED}FAIL{_COLOR_RESET}"
    if errors:
        for err in errors:
            print(f"\n  [{status}] {_COLOR_RED}{err}{_COLOR_RESET}")
    else:
        print(f"\n  [{status}]")

    return {
        "path": os.path.abspath(exported_path),
        "format": motion_b.file_format,
        "has_skinned_mesh": bool(motion_b.has_skinned_mesh),
        "passed": passed,
        "errors": errors,
        "position_max_error": float(pos_result["max_error"]),
        "position_max_error_pct_char": float(pos_result["max_error_pct_char"]),
        "position_worst_bone": pos_result["worst_bone"],
        "position_worst_frame": int(pos_result["worst_frame"]),
        "rotation_max_error_deg": float(rot_result["max_error_deg"]),
        "rotation_worst_bone": rot_result["worst_bone"],
        "rotation_worst_frame": int(rot_result["worst_frame"]),
        "world_quaternion_max_error_deg": float(world_quat_result["max_error_deg"]),
        "world_quaternion_worst_bone": world_quat_result["worst_bone"],
        "world_quaternion_worst_frame": int(world_quat_result["worst_frame"]),
        "mesh_mean_error_pct_char": mesh_mean_pct_char,
        "mesh_p99_error_pct_char": mesh_p99_pct_char,
    }


def test_fbx_load_matches_skinned_glb_local_channels(tmp_path) -> None:
    outputs = _export_variants_from_fbx(_DEFAULT_FBX, str(tmp_path), skip_bvh=True)
    _assert_loaded_animation_matches(_DEFAULT_FBX, outputs["SkinnedGLB"])


def _run_test_exporter_compare_motions(
    fbx_path: str | None = None,
    output_dir: str | None = None,
    pos_tolerance_pct_char: float = 0.1,
    rot_tolerance_deg: float = 0.1,
    mesh_mean_tolerance_pct_char: float = 0.1,
    mesh_p99_tolerance_pct_char: float = 1.0,
    skip_bvh: bool = True,
) -> dict[str, Any]:
    if fbx_path is None:
        fbx_path = _DEFAULT_FBX
    assert os.path.isfile(fbx_path), f"Missing required GLB: {fbx_path}"

    temp_context = nullcontext(output_dir) if output_dir else tempfile.TemporaryDirectory(prefix="exporter_compare_")
    with temp_context as work_dir:
        assert work_dir is not None
        os.makedirs(work_dir, exist_ok=True)

        print(f"[Source] GLB: {_fmt_path(fbx_path)}")
        exports = _export_variants_from_fbx(fbx_path, work_dir, skip_bvh=skip_bvh)

        if not skip_bvh:
            print(f"  [Exported] BVH: {_fmt_path(exports['BVH'])}")
        print(f"  [Exported] Bones-only GLB: {_fmt_path(exports['BonesOnlyGLB'])}")
        print(f"  [Exported] Skinned GLB: {_fmt_path(exports['SkinnedGLB'])}")

        results = {}
        if not skip_bvh:
            results["BVH"] = _compare_export_to_fbx(
                exports["FBX"],
                exports["BVH"],
                "BVH",
                pos_tolerance_pct_char,
                rot_tolerance_deg,
                mesh_mean_tolerance_pct_char,
                mesh_p99_tolerance_pct_char,
            )
        results["BonesOnlyGLB"] = _compare_export_to_fbx(
            exports["FBX"],
            exports["BonesOnlyGLB"],
            "BonesOnlyGLB",
            pos_tolerance_pct_char,
            rot_tolerance_deg,
            mesh_mean_tolerance_pct_char,
            mesh_p99_tolerance_pct_char,
        )
        results["SkinnedGLB"] = _compare_export_to_fbx(
            exports["FBX"],
            exports["SkinnedGLB"],
            "SkinnedGLB",
            pos_tolerance_pct_char,
            rot_tolerance_deg,
            mesh_mean_tolerance_pct_char,
            mesh_p99_tolerance_pct_char,
        )

        # Summary
        all_passed = all(r["passed"] for r in results.values())
        overall_status = "PASS" if all_passed else f"{_COLOR_RED}FAIL{_COLOR_RESET}"
        print(f"\n{'=' * 78}")
        print(f"[{overall_status}] exporter outputs vs source")
        print(f"{'=' * 78}")
        return {
            "source_fbx": os.path.abspath(fbx_path),
            "output_dir": os.path.abspath(work_dir),
            "fps": exports["fps"],
            "num_frames": exports["num_frames"],
            "num_joints": exports["num_joints"],
            "tolerances": {
                "position_pct_char": float(pos_tolerance_pct_char),
                "rotation_deg": float(rot_tolerance_deg),
                "mesh_mean_pct_char": float(mesh_mean_tolerance_pct_char),
                "mesh_p99_pct_char": float(mesh_p99_tolerance_pct_char),
            },
            "results": results,
        }


# ── pytest entry point ────────────────────────────────────────────────────────

def test_exporter_compare_motions() -> None:
    """Pytest-compatible entry point. Asserts all exporter comparisons pass."""
    report = _run_test_exporter_compare_motions()
    assert all(r["passed"] for r in report["results"].values()), (
        "Exporter comparison failed: "
        + ", ".join(
            f"{k}={v['errors']}"
            for k, v in report["results"].items()
            if not v["passed"]
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export BVH / bones-only GLB / skinned GLB from one GLB and compare each output to the source.",
    )
    parser.add_argument(
        "--fbx",
        default=_DEFAULT_FBX,
        help="Path to source GLB animation.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(_ANYTOP_ROOT, "outputs", "exporter_compare_motions"),
        help="Directory used for generated exporter artifacts.",
    )
    parser.add_argument(
        "--pos-tolerance",
        type=float,
        default=0.1,
        help="Max allowed position error as percent of character size.",
    )
    parser.add_argument(
        "--rot-tolerance",
        type=float,
        default=0.1,
        help="Max allowed rotation error in degrees.",
    )
    parser.add_argument(
        "--mesh-mean-tolerance",
        type=float,
        default=0.1,
        help="Max allowed mesh mean nearest-surface error as percent of character size.",
    )
    parser.add_argument(
        "--mesh-p99-tolerance",
        type=float,
        default=1.0,
        help="Max allowed mesh p99 nearest-surface error as percent of character size.",
    )
    parser.add_argument(
        "--no-skip-bvh",
        action="store_true",
        default=False,
        help="Include BVH export and comparison (default: skip BVH).",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write a JSON report.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    skip_bvh = not args.no_skip_bvh
    report = _run_test_exporter_compare_motions(
        fbx_path=args.fbx,
        output_dir=args.output_dir,
        pos_tolerance_pct_char=args.pos_tolerance,
        rot_tolerance_deg=args.rot_tolerance,
        mesh_mean_tolerance_pct_char=args.mesh_mean_tolerance,
        mesh_p99_tolerance_pct_char=args.mesh_p99_tolerance,
        skip_bvh=skip_bvh,
    )

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"[Report] JSON saved -> {args.json_out}")
