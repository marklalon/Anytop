"""
Test compare_motions.py with self and pairwise cross-format comparison.

This script takes a single FBX animation, uses Blender to export temporary
BVH and GLB files from that FBX, then compares all three formats against each
other using the existing compare_motions pipeline.

Expected: errors should be very small (near-zero for self-comparison, small
for cross-format due to import/export precision).

Usage:
    # Run inside Blender Python environment:
    blender -b -P tests/test_compare_motions.py -- \
        --fbx dataset/truebones/zoo/Truebone_Z-OO/Horse/HorseALL-RunToStop.fbx \
        --pos-tolerance 0.05 \
        --rot-tolerance 1.0

    # Or with an absolute path:
    blender -b -P tests/test_compare_motions.py -- \
        --fbx "D:\AI\pcvg-skeleton-animation\Anytop\dataset\truebones\zoo\Truebone_Z-OO\Horse\HorseALL-RunToStop.fbx"
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
from typing import Any


# ANSI colors
_COLOR_RED = "\033[31m"
_COLOR_RESET = "\033[0m"


_COLOR_RED = "\033[31m"
_COLOR_RESET = "\033[0m"


# ── Path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ANYTOP_ROOT = os.path.dirname(_SCRIPT_DIR)
_REPO_ROOT = os.path.dirname(_ANYTOP_ROOT)

for _p in [_REPO_ROOT, _ANYTOP_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── Import compare_motions module ─────────────────────────────────────────────
from tools.compare_motions import (
    _load_motion,
    _validate_compatible,
    _detect_and_align,
    _compare_motions,
    _print_summary,
    MotionData,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_path(p: str) -> str:
    if not os.path.isabs(p):
        return p
    try:
        return os.path.relpath(p, _REPO_ROOT)
    except ValueError:
        return p


def _get_action_frame_range(armature) -> tuple[int, int]:
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        return 0, 0
    return int(round(action.frame_range[0])), int(round(action.frame_range[1]))


def _select_armature_and_meshes(armature) -> None:
    import bpy

    for obj in bpy.data.objects:
        obj.select_set(False)

    armature.select_set(True)
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if obj.parent == armature:
            obj.select_set(True)
            continue
        if any(mod.type == "ARMATURE" and mod.object == armature for mod in obj.modifiers):
            obj.select_set(True)


def _export_temp_motion_files(fbx_path: str, temp_dir: str) -> dict[str, str]:
    import bpy
    from utils.fbx import clear_scene, import_fbx, remove_lights_and_cameras

    clear_scene()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import_fbx(fbx_path)
    remove_lights_and_cameras()

    armature = next((obj for obj in bpy.data.objects if obj.type == "ARMATURE"), None)
    if armature is None:
        raise RuntimeError(f"No armature found in {_fmt_path(fbx_path)}")

    frame_start, frame_end = _get_action_frame_range(armature)
    scene = bpy.context.scene
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    bpy.context.view_layer.objects.active = armature

    base_name = os.path.splitext(os.path.basename(fbx_path))[0]
    glb_path = os.path.join(temp_dir, f"{base_name}.glb")
    bvh_path = os.path.join(temp_dir, f"{base_name}.bvh")

    _select_armature_and_meshes(armature)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        bpy.ops.export_scene.gltf(
            filepath=glb_path,
            export_format='GLB',
            export_animations=True,
            export_animation_mode='ACTIVE_ACTIONS',
            # Bake each frame so the FBX->GLB test measures importer/exporter parity,
            # not sparse glTF key reduction differences.
            export_force_sampling=True,
            export_frame_range=True,
            export_apply=False,
            export_yup=True,
        )

    for obj in bpy.data.objects:
        obj.select_set(False)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.export_anim.bvh(
        filepath=bvh_path,
        frame_start=frame_start,
        frame_end=frame_end,
        rotate_mode='NATIVE',
        root_transform_only=False,
        global_scale=1.0,
    )

    return {
        "FBX": os.path.abspath(fbx_path),
        "GLB": glb_path,
        "BVH": bvh_path,
    }


def _run_comparison(name_a: str, motion_a: MotionData,
                    name_b: str, motion_b: MotionData) -> dict[str, Any] | None:
    """Run full alignment + comparison pipeline, return result dict or None on incompatibility."""
    print(f"\n{'='*70}")
    print(f"  Comparing: {name_a} vs {name_b}")
    print(f"{'='*70}")

    # Check FPS compatibility before calling _validate_compatible (which calls sys.exit)
    fps_tol = 0.001 * max(motion_a.fps, motion_b.fps)
    if abs(motion_a.fps - motion_b.fps) > fps_tol:
        print(f"  [SKIP] FPS mismatch: A={motion_a.fps:.4f}, B={motion_b.fps:.4f}")
        return None
    if motion_a.num_frames != motion_b.num_frames:
        print(f"  [SKIP] Frame count mismatch: A={motion_a.num_frames}, B={motion_b.num_frames}")
        return None

    _validate_compatible(motion_a, motion_b)
    motion_b_aligned, alignment = _detect_and_align(motion_a, motion_b)
    result = _compare_motions(motion_a, motion_b_aligned, alignment)

    _print_summary(motion_a, motion_b, alignment, result)
    return result


def _check_pass(result: dict[str, Any], pos_tol: float, rot_tol: float,
                label: str) -> bool:
    """Check if position max_error_pct_char <= pos_tol and rotation max_error_deg <= rot_tol."""
    pos_max = result["position"]["max_error"]
    pos_max_pct_char = result["position"]["max_error_pct_char"]
    rot_max = result["rotation"]["max_error_deg"]
    world_quat_max = result["world_quaternion"]["max_error_deg"]
    passed = pos_max_pct_char <= pos_tol and rot_max <= rot_tol and world_quat_max <= rot_tol

    status = f"{_COLOR_RED}FAIL{_COLOR_RESET}" if not passed else "PASS"
    print(f"\n  [{status}] {label}: pos_max={pos_max:.6f} ({pos_max_pct_char:.4f}%), "
          f"rot_max={rot_max:.6f}°, world_quat_max={world_quat_max:.6f}°")
    return passed


# ── Default test files ────────────────────────────────────────────────────────

_DEFAULT_FBX  = os.path.join(_ANYTOP_ROOT, "dataset", "truebones", "zoo",
                              "Truebone_Z-OO", "Horse", "HorseALL-RunToStop.fbx")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test compare_motions.py by exporting temporary BVH/GLB from one FBX.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--fbx", default=_DEFAULT_FBX, help="Path to FBX file")
    parser.add_argument("--pos-tolerance", type=float, default=1.0,
                        help="Max position error tolerance (%% of character size, default=1.0)")
    parser.add_argument("--rot-tolerance", type=float, default=1.0,
                        help="Max rotation error tolerance (degrees, default=1.0)")
    parser.add_argument("--json-out", default=None,
                        help="Optional path to write full JSON report")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="compare_motions_") as temp_dir:
        generated_paths = _export_temp_motion_files(args.fbx, temp_dir)

        print(f"[Source] FBX: {_fmt_path(generated_paths['FBX'])}")
        print(f"[Generated] BVH: {_fmt_path(generated_paths['BVH'])}")
        print(f"[Generated] GLB: {_fmt_path(generated_paths['GLB'])}")

        # ── Load all three motions ──────────────────────────────────────────
        motions = {}
        for label in ["BVH", "FBX", "GLB"]:
            path = generated_paths[label]
            print(f"[Loading] {label}: {_fmt_path(path)}")
            motions[label] = _load_motion(path)

        for label, m in motions.items():
            print(f"  {label}: frames={m.num_frames}, bones={m.num_joints}, "
                  f"fps={m.fps:.2f}, format={m.file_format.upper()}")

        print(f"\n[Tolerances] position={args.pos_tolerance:.2f}% char  rotation={args.rot_tolerance:.4f}°  world_quaternion={args.rot_tolerance:.4f}°")

        # ── Self-comparisons (expect near-zero error) ──────────────────────
        print("\n\n### Self-comparisons (expect near-zero error) ###\n")
        results = {}
        all_passed = True
        SELF_TOL = 1e-5

        for label, motion in motions.items():
            motion_b_aligned, alignment = _detect_and_align(motion, motion)
            result = _compare_motions(motion, motion_b_aligned, alignment)
            results[f"{label}_self"] = result

            pos_max = result["position"]["max_error"]
            rot_max = result["rotation"]["max_error_deg"]
            passed = pos_max <= SELF_TOL and rot_max <= SELF_TOL

            status = f"{_COLOR_RED}FAIL{_COLOR_RESET}" if not passed else "PASS"
            print(f"  [{status}] {label} self: pos_max={pos_max:.2e}, rot_max={rot_max:.2e} (tol={SELF_TOL:.0e})")
            all_passed = all_passed and passed

        # ── Pairwise comparisons ────────────────────────────────────────────
        print("\n\n### Pairwise cross-format comparisons ###\n")
        pairs = [
            ("BVH", "FBX"),
            ("BVH", "GLB"),
            ("FBX", "GLB"),
        ]

        for a_label, b_label in pairs:
            r = _run_comparison(
                a_label, motions[a_label],
                b_label, motions[b_label],
            )
            key = f"{a_label}_vs_{b_label}"
            results[key] = r
            if r is None:
                print(f"  [SKIP] {a_label} vs {b_label} — incompatible, skipping")
                continue
            passed = _check_pass(r, pos_tol=args.pos_tolerance,
                                 rot_tol=args.rot_tolerance,
                                 label=f"{a_label} vs {b_label}")
            all_passed = all_passed and passed

        # ── Summary ─────────────────────────────────────────────────────────
        print(f"\n\n{'#'*70}")
        overall = "ALL TESTS PASSED" if all_passed else f"{_COLOR_RED}SOME TESTS FAILED{_COLOR_RESET}"
        print(f"  {overall}")
        print(f"{'#'*70}\n")

        # ── JSON report ─────────────────────────────────────────────────────
        if args.json_out:
            report = {
                "files": {
                    label: {
                        "path": os.path.abspath(m.file_path),
                        "format": m.file_format,
                        "frames": m.num_frames,
                        "bones": m.num_joints,
                        "fps": m.fps,
                    }
                    for label, m in motions.items()
                },
                "tolerances": {
                    "position": args.pos_tolerance,
                    "rotation_deg": args.rot_tolerance,
                },
                "results": {},
            }
            for key, r in results.items():
                if r is None:
                    report["results"][key] = {"skipped": True}
                    continue
                report["results"][key] = {
                    "position_max_error": r["position"]["max_error"],
                    "position_mean_error": r["position"]["mean_error"],
                    "position_max_error_pct_char": r["position"]["max_error_pct_char"],
                    "position_mean_error_pct_char": r["position"]["mean_error_pct_char"],
                    "character_size": r["position"]["character_size"],
                    "rotation_max_error_deg": r["rotation"]["max_error_deg"],
                    "rotation_mean_error_deg": r["rotation"]["mean_error_deg"],
                }

            os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"[Report] JSON saved -> {args.json_out}")

        sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
