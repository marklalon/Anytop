"""
FBX -> NPY -> GLB roundtrip test.

Loads a source FBX animation, converts it into AnyTop's production preprocessed
13-channel NPY feature space, saves the bare `(F, J, 13)` tensor exactly like
the real generation pipeline, restores it through `tools/restore_glb_from_npy`,
and compares the recovered GLB directly against the original source FBX.

This keeps the test aligned with the real sample/generate environment instead
of the self-contained own-rotation payload helpers used by narrower debugging
roundtrips.

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
        --tolerance 0.01
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
for _p in [_REPO_ROOT, _ANYTOP_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Resolve utils namespace conflict ────────────────────────────────────────
import importlib.util

def _load_utils_module(module_name: str) -> None:
    """Load a utils submodule under its full dotted name to avoid namespace collision."""
    _path = os.path.join(_ANYTOP_ROOT, "utils", f"{module_name.rsplit('.', 1)[-1]}.py")
    if os.path.isfile(_path) and module_name not in sys.modules:
        _spec = importlib.util.spec_from_file_location(module_name, _path)
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules[module_name] = _mod
        _spec.loader.exec_module(_mod)

_load_utils_module("utils.rotation_conversions")
_load_utils_module("utils.npy_roundtrip_utils")


from Anytop.motion_lib import FBX
from data_loaders.truebones.truebones_utils.motion_process import (
    FOOT_CONTACT_VEL_THRESH,
    get_common_features_from_T_pose,
    get_motion,
)
from tools.restore_glb_from_npy import restore_glb

from tools.compare_motions import (
    _canonical_bone_name,
    _compute_mesh_surface_error,
    _load_motion,
    _validate_compatible,
    _detect_and_align,
    _compare_motions,
    _print_summary,
)


# ── Main test function ───────────────────────────────────────────────────────

_COLOR_RED = "\033[31m"
_COLOR_RESET = "\033[0m"

_EXPORT_POS_TOLERANCE_PCT_CHAR = 1.5
_EXPORT_ROT_TOLERANCE_DEG = 1.0
_EXPORT_MESH_MEAN_TOLERANCE_PCT_CHAR = 3.5
_EXPORT_MESH_P99_TOLERANCE_PCT_CHAR = 12.0


_DEFAULT_FBX = os.path.join(
    _ANYTOP_ROOT, "dataset", "truebones", "zoo", "Truebone_Z-OO", "Horse", "HorseALL-RunToStop.fbx",
)
_DEFAULT_TPOSE_FBX = os.path.join(
    _ANYTOP_ROOT, "dataset", "truebones", "zoo", "Truebone_Z-OO", "Horse", "HorseALL-TPOSE.fbx",
)
_DEFAULT_OBJECT_TYPE = "Horse"

def _compare_export_to_source(
    source_fbx: str,
    exported_path: str,
    label: str,
) -> dict[str, Any]:
    motion_a = _load_motion(source_fbx)
    motion_b = _load_motion(exported_path)
    _validate_compatible(motion_a, motion_b)

    motion_b_aligned, alignment = _detect_and_align(motion_a, motion_b)
    result = _compare_motions(motion_a, motion_b_aligned, alignment)
    print(f"{'Compare Motions':=^{70}}")
    print(f"[Compare] {label}")
    _print_summary(motion_a, motion_b, alignment, result)

    errors: list[str] = []
    if alignment.rotation_label != "identity":
        errors.append(
            f"{label}: unexpected rigid coordinate alignment {alignment.rotation_label}; "
            "export should preserve the source-facing world basis"
        )

    pos_result = result["position"]
    rot_result = result["rotation"]
    world_quat_result = result["world_quaternion"]
    common_names = list(result["comparison"]["common_bone_names"])

    canon_to_orig_idx = {
        _canonical_bone_name(name): index for index, name in enumerate(motion_a.bone_names)
    }
    child_counts = np.zeros(len(motion_a.parents), dtype=np.int32)
    for parent_idx in motion_a.parents:
        if parent_idx >= 0:
            child_counts[int(parent_idx)] += 1
    encoded_joint_mask = np.array(
        [child_counts[canon_to_orig_idx[name]] > 0 for name in common_names],
        dtype=bool,
    )
    encoded_world_quat_per_bone = np.asarray(
        world_quat_result["per_bone"]["max_deg"],
        dtype=np.float64,
    )
    encoded_world_quat_max = float(
        encoded_world_quat_per_bone[encoded_joint_mask].max()
        if np.any(encoded_joint_mask)
        else 0.0
    )

    if pos_result["max_error_pct_char"] > _EXPORT_POS_TOLERANCE_PCT_CHAR:
        errors.append(
            f"{label}: position max error {pos_result['max_error']:.6f} "
            f"({pos_result['max_error_pct_char']:.4f}% char) exceeds "
            f"{_EXPORT_POS_TOLERANCE_PCT_CHAR:.4f}% "
            f"(worst bone={pos_result['worst_bone']}, frame={pos_result['worst_frame']})"
        )
    if rot_result["max_error_deg"] > _EXPORT_ROT_TOLERANCE_DEG:
        errors.append(
            f"{label}: rotation max error {rot_result['max_error_deg']:.6f} deg exceeds "
            f"{_EXPORT_ROT_TOLERANCE_DEG:.6f} "
            f"(worst bone={rot_result['worst_bone']}, frame={rot_result['worst_frame']})"
        )
    if encoded_world_quat_max > _EXPORT_ROT_TOLERANCE_DEG:
        errors.append(
            f"{label}: encoded-joint world quaternion max error {encoded_world_quat_max:.6f} deg exceeds "
            f"{_EXPORT_ROT_TOLERANCE_DEG:.6f}"
        )

    mesh_mean_pct_char = None
    mesh_p99_pct_char = None
    if motion_a.has_skinned_mesh and motion_b.has_skinned_mesh:
        mesh_result = _compute_mesh_surface_error(motion_a, motion_b_aligned, alignment)
        if mesh_result is None:
            errors.append(f"{label}: expected mesh surface stats")
        else:
            char_size = max(float(pos_result["character_size"]), 1e-8)
            mesh_mean_pct_char = mesh_result["mean"] / char_size * 100.0
            mesh_p99_pct_char = mesh_result["p99"] / char_size * 100.0
            if mesh_mean_pct_char > _EXPORT_MESH_MEAN_TOLERANCE_PCT_CHAR:
                errors.append(
                    f"{label}: mesh mean error {mesh_result['mean']:.6f} "
                    f"({mesh_mean_pct_char:.4f}% char) exceeds {_EXPORT_MESH_MEAN_TOLERANCE_PCT_CHAR:.4f}%"
                )
            if mesh_p99_pct_char > _EXPORT_MESH_P99_TOLERANCE_PCT_CHAR:
                errors.append(
                    f"{label}: mesh p99 error {mesh_result['p99']:.6f} "
                    f"({mesh_p99_pct_char:.4f}% char) exceeds {_EXPORT_MESH_P99_TOLERANCE_PCT_CHAR:.4f}%"
                )

    return {
        "label": label,
        "path": os.path.abspath(exported_path),
        "errors": errors,
        "position_max_error": float(pos_result["max_error"]),
        "position_max_error_pct_char": float(pos_result["max_error_pct_char"]),
        "position_worst_bone": pos_result["worst_bone"],
        "position_worst_frame": int(pos_result["worst_frame"]),
        "rotation_max_error_deg": float(rot_result["max_error_deg"]),
        "rotation_worst_bone": rot_result["worst_bone"],
        "rotation_worst_frame": int(rot_result["worst_frame"]),
        "world_quaternion_max_error_deg": float(world_quat_result["max_error_deg"]),
        "encoded_world_quaternion_max_error_deg": encoded_world_quat_max,
        "world_quaternion_worst_bone": world_quat_result["worst_bone"],
        "world_quaternion_worst_frame": int(world_quat_result["worst_frame"]),
        "mesh_mean_error_pct_char": mesh_mean_pct_char,
        "mesh_p99_error_pct_char": mesh_p99_pct_char,
    }


def _run_test_fbx_npy_glb_roundtrip(
    tpose_fbx: str | None = None,
    anim_fbx: str | None = None,
    object_type: str = "Horse",
    output_dir: str | None = None,
    tolerance: float = 0.01,
) -> dict[str, Any]:
    """FBX -> NPY -> GLB roundtrip test.

    Pipeline:
        1. Load T-pose metadata and derive AnyTop preprocessing parameters
        2. Convert the source FBX into the same preprocessed feature-space animation
        3. Save a metadata payload with the 13-channel NPY features
        4. Recover Animation from that payload
        5. Invert T-pose reparameterization back to processed/source-rig semantics
        6. Export baseline + recovered animations as GLB
        7. Compare both exports against the original source FBX
    """
    if tpose_fbx is None:
        tpose_fbx = _DEFAULT_TPOSE_FBX if os.path.isfile(_DEFAULT_TPOSE_FBX) else _DEFAULT_FBX
    if anim_fbx is None:
        anim_fbx = _DEFAULT_FBX

    for file_path in [tpose_fbx, anim_fbx]:
        assert os.path.isfile(file_path), f"Missing required file: {file_path}"

    temp_context = nullcontext(output_dir) if output_dir else tempfile.TemporaryDirectory()
    with temp_context as work_dir:
        assert work_dir is not None
        os.makedirs(work_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(anim_fbx))[0]
        recovered_glb = os.path.join(work_dir, f"{base_name}_recovered.glb")
        npy_path = os.path.join(work_dir, f"{base_name}_features.npy")

        # Phase A: Load T-pose metadata used by the production preprocessing path
        print("[Phase A] Loading T-pose FBX preprocessing metadata...")
        (
            _root_pose_init_xz,
            scale_factor,
            offsets_hml,
            _foot_indices,
            tpose_rotations,
            tpose_bone_names,
            feature_tpose_anim,
            face_joints,
            orientation_quat,
            forward_joint_index,
            forward_base_joint_index,
            _contact_joint_source,
        ) = get_common_features_from_T_pose(tpose_fbx, object_type)
        print(f"Joints: {len(tpose_bone_names)}, scale_factor: {scale_factor:.6f}")

        # Phase B: Load source FBX and build the production preprocessed motion/features
        print("[Phase B] Loading source FBX and building production preprocessed motion...")
        source_anim, source_bone_names, source_frame_time = FBX.load(anim_fbx)
        source_fps = 1.0 / source_frame_time if source_frame_time > 0 else 30.0
        print(f"Frames: {len(source_anim)}, Joints: {source_anim.shape[1]}, FPS: {source_fps:.1f}")

        squared_positions_error: dict[str, float] = {}
        features, feature_parents, _max_joints, feature_anim, _baseline_export_anim, _is_loop = get_motion(
            anim_fbx,
            FOOT_CONTACT_VEL_THRESH,
            object_type,
            len(tpose_bone_names),
            _root_pose_init_xz,
            scale_factor,
            offsets_hml,
            _foot_indices,
            tpose_rotations,
            squared_positions_error,
            face_joints=face_joints,
            orientation_quat=orientation_quat,
            forward_joint_index=forward_joint_index,
            forward_base_joint_index=forward_base_joint_index,
            preloaded=(source_anim, source_bone_names),
        )

        if feature_anim is None or _baseline_export_anim is None:
            raise AssertionError("Production preprocessing failed to build feature/export animations")
        if feature_anim.shape[1] != len(tpose_bone_names):
            raise AssertionError(
                "Production preprocessing joint count does not match the T-pose feature skeleton"
            )
        if squared_positions_error:
            worst_sq_error = max(squared_positions_error.values())
            print(f"  T-pose reparameterization MSE: {worst_sq_error:.8f}")

        # Phase C: Save the production bare NPY tensor exactly like sample/generate.py
        print("[Phase C] Saving production bare NPY feature tensor...")
        features = np.asarray(features, dtype=np.float32)
        np.save(npy_path, features, allow_pickle=False)
        print(f"  NPY shape: {features.shape}")
        print(f"Saved NPY features to {npy_path}")

        # Phase D: Restore the production bare tensor through the real restore tool
        print(f"[Phase D] Restoring production NPY via restore_glb_from_npy.py -> {os.path.basename(recovered_glb)}...")
        restore_glb(
            npy_path=npy_path,
            tpose_mesh=tpose_fbx,
            output_glb=recovered_glb,
            object_type=object_type,
            fps=source_fps,
        )

        # Phase E: Compare recovered GLB vs original source FBX using compare_motions.py
        recovered_report = _compare_export_to_source(anim_fbx, recovered_glb, "RecoveredGLB")

        errors: list[str] = list()
        errors.extend(recovered_report["errors"])

        # Map worst_frame to sample time for the return dict
        worst_frame = recovered_report["position_worst_frame"]
        source_motion = _load_motion(anim_fbx)
        recovered_worst_time = float(source_motion.sample_times[worst_frame]) \
            if worst_frame < len(source_motion.sample_times) else 0.0

        # Print PASS/FAIL
        passed = len(errors) == 0
        status = "PASS" if passed else f"{_COLOR_RED}FAIL{_COLOR_RESET}"
        if errors:
            for err in errors:
                print(f"\n  [{status}] {_COLOR_RED}{err}{_COLOR_RESET}")
        else:
            print(f"\n  [{status}] FBX -> NPY -> GLB roundtrip checks passed")
        return {
            "passed": passed,
            "errors": errors,
            "recovered_error": float(recovered_report["position_max_error"]),
            "recovered_error_pct_char": float(recovered_report["position_max_error_pct_char"]),
            "recovered_worst_bone": recovered_report["position_worst_bone"],
            "recovered_worst_frame": int(worst_frame),
            "recovered_worst_time": recovered_worst_time,
            "rot_max": float(recovered_report["rotation_max_error_deg"]),
            "rot_worst_bone": recovered_report["rotation_worst_bone"],
            "rot_worst_frame": int(recovered_report["rotation_worst_frame"]),
            "world_quat_max": float(recovered_report["world_quaternion_max_error_deg"]),
            "encoded_world_quat_max": float(recovered_report["encoded_world_quaternion_max_error_deg"]),
        }


# ── pytest entry point ────────────────────────────────────────────────────────

def test_fbx_npy_glb_roundtrip() -> None:
    """Pytest-compatible entry point. Asserts FBX->NPY->GLB roundtrip passes."""
    result = _run_test_fbx_npy_glb_roundtrip()
    assert result["passed"], "FBX->NPY->GLB roundtrip failed: " + "; ".join(result["errors"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FBX -> NPY -> GLB roundtrip smoke test",
    )
    parser.add_argument(
        "--tpose-fbx",
        default=None,
        help="Path to the T-pose FBX used for preprocessing metadata and as the export container.",
    )
    parser.add_argument(
        "--fbx",
        default=None,
        help="Path to source animation FBX file.",
    )
    parser.add_argument(
        "--object-type",
        default=None,
        help="Character type for contact inference. Inferred from the FBX filename if not specified.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(_ANYTOP_ROOT, "outputs", "fbx_npy_roundtrip"),
        help="Directory to save roundtrip artifacts.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Max allowed comparison error in meters (default: 0.01).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.tpose_fbx is None:
        args.tpose_fbx = _DEFAULT_TPOSE_FBX if os.path.isfile(_DEFAULT_TPOSE_FBX) else args.fbx
        args.object_type = _DEFAULT_OBJECT_TYPE

    if args.object_type is None:
        from utils.misc import infer_object_type_from_filename
        args.object_type = infer_object_type_from_filename(args.fbx) or os.path.basename(args.fbx)

    print(f"T-pose FBX : {args.tpose_fbx}")
    print(f"Anim FBX   : {args.fbx}")
    print(f"Output dir : {args.output_dir}")
    print(f"Object type: {args.object_type}")
    print(f"Tolerance  : {args.tolerance}")
    print()

    result = _run_test_fbx_npy_glb_roundtrip(
        tpose_fbx=args.tpose_fbx,
        anim_fbx=args.fbx,
        object_type=args.object_type,
        output_dir=args.output_dir,
        tolerance=args.tolerance,
    )

    print("\nSummary:")
    print(
        f"  Recovered -> source   : pos_max={result['recovered_error']:.6f} "
        f"({result['recovered_error_pct_char']:.4f}%) "
        f"(bone={result['recovered_worst_bone']}, frame={result['recovered_worst_frame']}, "
        f"time={result['recovered_worst_time']:.6f})"
    )
    print(
        f"  Rotation / wquat      : max_rot={result['rot_max']:.6f}°  "
        f"max_wquat_encoded={result['encoded_world_quat_max']:.6f}°  "
        f"max_wquat_raw={result['world_quat_max']:.6f}°  "
        f"(bone={result['rot_worst_bone']}, frame={result['rot_worst_frame']})"
    )
