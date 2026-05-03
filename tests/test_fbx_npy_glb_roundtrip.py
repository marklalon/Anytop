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
    _load_fbx_skeleton_metadata,
    _measure_fbx_glb_error,
    _print_comparison_report,
)


# ── Main test function ───────────────────────────────────────────────────────

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
        recovered_glb = os.path.join(work_dir, f"{base_name}.glb")
        npy_path = os.path.join(work_dir, f"{base_name}_features.npy")

        print(f"[FBX Roundtrip] T-pose FBX : {tpose_fbx}")
        print(f"[FBX Roundtrip] Source FBX  : {anim_fbx}")
        print(f"[FBX Roundtrip] Output dir  : {work_dir}")

        # Phase A: Load T-pose FBX for skeleton metadata
        print("  [Phase A] Loading T-pose FBX for skeleton metadata...")
        tpose_meta_bone_names, parents, offsets, rest_rotations = _load_fbx_skeleton_metadata(tpose_fbx)
        tpose_anim, tpose_bone_names, tpose_fps = _fbx_to_animation(tpose_fbx)
        print(f"    Joints: {len(tpose_bone_names)}, FPS: {tpose_fps:.1f}")
        if tpose_meta_bone_names != tpose_bone_names:
            raise AssertionError("T-pose FBX animation bone order does not match extracted skeleton metadata")

        tpose_skeleton = _build_skeleton(tpose_bone_names, offsets, parents, rest_rotations)

        # Phase B: Load source FBX and extract motion
        print("  [Phase B] Loading source FBX and extracting motion...")
        source_anim, source_bone_names, source_fps = _fbx_to_animation(anim_fbx)
        print(f"    Frames: {len(source_anim)}, Joints: {source_anim.shape[1]}, FPS: {source_fps:.1f}")

        if source_bone_names != tpose_bone_names:
            raise AssertionError(
                "Source FBX and T-pose FBX do not share the same BFS bone order"
            )

        # Phase C: Extract raw NPY features
        print("  [Phase C] Extracting raw NPY features...")
        feature_payload = build_roundtrip_feature_payload(
            source_anim, object_type, offsets, parents, source_bone_names,
        )
        np.save(npy_path, feature_payload, allow_pickle=True)
        print(f"    NPY shape: {feature_payload['features'].shape}")
        print(f"    Saved NPY features to {npy_path}")

        # Phase D: Recover Animation from NPY features
        print("  [Phase D] Recovering Animation from NPY features...")
        recovered_anim, has_animated_pos = recover_from_features(
            feature_payload, parents, offsets,
        )
        print(f"    Recovered frames: {len(recovered_anim)}")
        if has_animated_pos:
            print("    (has non-root animated position channels)")

        from motion_lib.Animation import positions_global

        source_global = positions_global(source_anim)
        recovered_global = positions_global(recovered_anim)
        npy_position_error = np.abs(source_global - recovered_global).max(axis=(0, 2))
        npy_worst_idx = int(np.argmax(npy_position_error))
        npy_worst_bone = source_bone_names[npy_worst_idx] if npy_worst_idx < len(source_bone_names) else "?"
        print(
            "  [Diag] Animation-domain source-vs-recovered max per-joint error: "
            f"{npy_position_error.max():.6f} ({npy_worst_bone})"
        )
        print("    Note: this is diagnostic only because recovery is built on the T-pose FBX skeleton.")

        # Phase E: Export NPY-recovered animation -> recovered.glb
        print("  [Phase E] Exporting NPY-recovered animation -> recovered.glb...")
        _export_animation_to_glb(
            recovered_anim,
            tpose_skeleton,
            recovered_glb,
            mesh_path=tpose_fbx,
            fps=source_fps,
        )

        # Phase F: Compare recovered GLB vs source FBX
        recovered_metrics = _measure_fbx_glb_error(anim_fbx, recovered_glb)
        _print_comparison_report("Roundtrip (FBX vs recovered GLB)", recovered_metrics)

        assert recovered_metrics["max_error"] < tolerance, (
            f"FBX -> recovered GLB max error {recovered_metrics['max_error']:.6f} exceeds "
            f"{tolerance} (worst bone={recovered_metrics['worst_bone']}, "
            f"sample={recovered_metrics['worst_frame']}, time={recovered_metrics['worst_time']:.6f})"
        )

        print("\n  PASS  FBX -> NPY -> GLB roundtrip checks passed")
        return {
            "npy_error": float(npy_position_error.max()),
            "npy_worst_bone": npy_worst_bone,
            "npy_worst_frame": npy_worst_idx,
            "recovered_error": float(recovered_metrics["max_error"]),
            "recovered_worst_bone": recovered_metrics["worst_bone"],
            "recovered_worst_frame": int(recovered_metrics["worst_frame"]),
            "recovered_worst_time": float(recovered_metrics["worst_time"]),
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
    print(
        f"  NPY encoding error   : {result['npy_error']:.6f}  "
        f"(bone={result['npy_worst_bone']}, frame={result['npy_worst_frame']})"
    )
    print(
        f"  FBX -> recovered GLB : {result['recovered_error']:.6f} "
        f"(bone={result['recovered_worst_bone']}, sample={result['recovered_worst_frame']}, "
        f"time={result['recovered_worst_time']:.6f})"
    )
