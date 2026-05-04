"""
Test compare_motions.py with cross-format self-comparison and pairwise comparison.

Loads three motion files (BVH, FBX, GLB) representing the same Horse animation,
then compares each file against itself and all pairwise combinations.

Expected: errors should be very small (near-zero for self-comparison, small
for cross-format due to import/export precision).

Usage:
    # Run inside Blender Python environment:
    blender -b -P tests/test_compare_motions.py -- \\
        --bvh  dataset/truebones/zoo/Truebone_Z-OO/Horse/__RunToStop.bvh \\
        --fbx  dataset/truebones/zoo/Truebone_Z-OO/Horse/HorseALL-RunToStop.fbx \\
        --glb  outputs/fbx_npy_roundtrip/HorseALL-RunToStop_recovered.glb \\
        --pos-tolerance 0.05 \\
        --rot-tolerance 1.0

    # Or with absolute paths:
    blender -b -P tests/test_compare_motions.py -- \\
        --bvh  "D:\AI\pcvg-skeleton-animation\Anytop\dataset\truebones\zoo\Truebone_Z-OO\Horse\__RunToStop.bvh" \\
        --fbx  "D:\AI\pcvg-skeleton-animation\Anytop\dataset\truebones\zoo\Truebone_Z-OO\Horse\HorseALL-RunToStop.fbx" \\
        --glb  "D:\AI\pcvg-skeleton-animation\Anytop\outputs\fbx_npy_roundtrip\HorseALL-RunToStop_recovered.glb"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np


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
    AlignmentResult,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_path(p: str) -> str:
    return os.path.relpath(p, _REPO_ROOT) if os.path.isabs(p) else p


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
    passed = pos_max_pct_char <= pos_tol and rot_max <= rot_tol

    status = "PASS" if passed else "FAIL"
    print(f"\n  [{status}] {label}: pos_max={pos_max:.6f} ({pos_max_pct_char:.4f}%), "
          f"rot_max={rot_max:.6f}°")
    return passed


# ── Default test files ────────────────────────────────────────────────────────

_DEFAULT_BVH  = os.path.join(_ANYTOP_ROOT, "dataset", "truebones", "zoo",
                              "Truebone_Z-OO", "Horse", "__RunToStop.bvh")
_DEFAULT_FBX  = os.path.join(_ANYTOP_ROOT, "dataset", "truebones", "zoo",
                              "Truebone_Z-OO", "Horse", "HorseALL-RunToStop.fbx")
_DEFAULT_GLB  = os.path.join(_ANYTOP_ROOT, "outputs", "fbx_npy_roundtrip",
                              "HorseALL-RunToStop_recovered.glb")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test compare_motions.py with self & pairwise comparisons.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bvh", default=_DEFAULT_BVH, help="Path to BVH file")
    parser.add_argument("--fbx", default=_DEFAULT_FBX, help="Path to FBX file")
    parser.add_argument("--glb", default=_DEFAULT_GLB, help="Path to GLB file")
    parser.add_argument("--pos-tolerance", type=float, default=1.0,
                        help="Max position error tolerance (% of character size, default=1.0)")
    parser.add_argument("--rot-tolerance", type=float, default=1.0,
                        help="Max rotation error tolerance (degrees, default=1.0)")
    parser.add_argument("--json-out", default=None,
                        help="Optional path to write full JSON report")
    args = parser.parse_args()

    # ── Load all three motions ──────────────────────────────────────────────
    motions = {}
    for label, path in [("BVH", args.bvh), ("FBX", args.fbx), ("GLB", args.glb)]:
        print(f"[Loading] {label}: {_fmt_path(path)}")
        motions[label] = _load_motion(path)

    for label, m in motions.items():
        print(f"  {label}: frames={m.num_frames}, bones={m.num_joints}, "
              f"fps={m.fps:.2f}, format={m.file_format.upper()}")

    print(f"\n[Tolerances] position={args.pos_tolerance:.2f}% char  rotation={args.rot_tolerance:.4f}°")

    # ── Self-comparisons (expect near-zero error) ──────────────────────────
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

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label} self: pos_max={pos_max:.2e}, rot_max={rot_max:.2e} (tol={SELF_TOL:.0e})")
        all_passed = all_passed and passed

    # ── Pairwise comparisons ────────────────────────────────────────────────
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

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n\n{'#'*70}")
    overall = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"
    print(f"  {overall}")
    print(f"{'#'*70}\n")

    # ── JSON report ─────────────────────────────────────────────────────────
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
