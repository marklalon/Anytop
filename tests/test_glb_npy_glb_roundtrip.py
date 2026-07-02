"""
GLB -> NPY -> GLB roundtrip test.

Loads a source GLB animation, converts it into AnyTop's production preprocessed
13-channel NPY feature space, saves the bare `(F, J, 13)` tensor exactly like
the real generation pipeline, restores it through `tools/restore_glb_from_npy`,
and compares the recovered GLB directly against the original source GLB.

This keeps the test aligned with the real sample/generate environment instead
of the self-contained own-rotation payload helpers used by narrower debugging
roundtrips.

Requires bpy (Blender as Python module) in the current Python environment.

Usage examples:
    # Use a single GLB as both T-pose source and animation source:
    python tests/test_glb_npy_glb_roundtrip.py \
        --glb outputs/fbx_npy_roundtrip/original.glb \
        --object-type Horse

    # Specify a separate T-pose GLB for skeleton metadata:
    python tests/test_glb_npy_glb_roundtrip.py \
        --tpose-mesh outputs/tpose.glb \
        --glb outputs/fbx_npy_roundtrip/original.glb \
        --object-type Horse

    # Custom output directory and tolerance:
    python tests/test_glb_npy_glb_roundtrip.py \
        --glb outputs/fbx_npy_roundtrip/original.glb \\
        --output-dir outputs/glb_npy_roundtrip \\
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
import pytest


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
from data_loaders.truebones.offline_reference_dataset import load_cond_dict
from data_loaders.truebones.truebones_utils.param_utils import MAX_JOINTS
from data_loaders.truebones.truebones_utils.motion_process import (
    FOOT_CONTACT_VEL_THRESH,
    get_common_features_from_T_pose,
    TPoseFeatures,
    get_motion,
)
from tools.restore_glb_from_npy import restore_glb

from tools.compare_motions import (
    compute_mesh_surface_error,
    load_motion,
    _validate_compatible,
    detect_and_align,
    compare_motions,
    print_summary,
)


# ── Main test function ───────────────────────────────────────────────────────

_COLOR_RED = "\033[31m"
_COLOR_RESET = "\033[0m"

_EXPORT_POS_TOLERANCE_PCT_CHAR = 1.5
_EXPORT_ROT_TOLERANCE_DEG = 1.0
_EXPORT_MESH_MEAN_TOLERANCE_PCT_CHAR = 3.5
_EXPORT_MESH_P99_TOLERANCE_PCT_CHAR = 12.0


_DEFAULT_GLB = os.path.join(
    _ANYTOP_ROOT, "dataset", "truebones", "zoo", "Truebone_Z-OO", "Buffalo", "Buffalo-WalkLoop.glb",
)
_DEFAULT_TPOSE_MESH = os.path.join(
    _ANYTOP_ROOT, "dataset", "truebones", "zoo", "Truebone_Z-OO", "Buffalo", "Buffalo-TPOSE.glb",
)
_DEFAULT_OBJECT_TYPE = "Buffalo"


from Anytop.utils.misc import normalize_bone_key as _normalize_bone_key


def _require_dataset_cond_entry_or_skip(cond_entry: dict[str, Any] | None, object_type: str) -> dict[str, Any]:
    if cond_entry is None:
        message = f"cond.npy is missing an entry for {object_type}; regenerate or point the test at a matching dataset"
        if "PYTEST_CURRENT_TEST" in os.environ:
            pytest.skip(message)
        raise AssertionError(message)
    return cond_entry

def _compare_export_to_source(
    source_glb: str,
    exported_path: str,
    label: str,
) -> dict[str, Any]:
    motion_a = load_motion(source_glb)
    motion_b = load_motion(exported_path)
    _validate_compatible(motion_a, motion_b)

    motion_b_aligned, alignment = detect_and_align(motion_a, motion_b)
    result = compare_motions(motion_a, motion_b_aligned, alignment)
    print(f"{'Compare Motions':=^{70}}")
    print(f"[Compare] {label}")
    print_summary(motion_a, motion_b, alignment, result)

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
        _normalize_bone_key(name): index for index, name in enumerate(motion_a.bone_names)
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
        mesh_result = compute_mesh_surface_error(motion_a, motion_b_aligned, alignment)
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


def _run_test_glb_npy_glb_roundtrip(
    tpose_mesh: str | None = None,
    anim_glb: str | None = None,
    object_type: str = _DEFAULT_OBJECT_TYPE,
    output_dir: str | None = None,
    tolerance: float = 0.01,
) -> dict[str, Any]:
    """GLB -> NPY -> GLB roundtrip test.

    Pipeline:
        1. Load T-pose metadata and derive AnyTop preprocessing parameters
        2. Convert the source GLB into the same preprocessed feature-space animation
        3. Save a metadata payload with the 13-channel NPY features
        4. Recover Animation from that payload
        5. Invert T-pose reparameterization back to processed/source-rig semantics
        6. Export baseline + recovered animations as GLB
        7. Compare both exports against the original source GLB
    """
    if tpose_mesh is None:
        tpose_mesh = _DEFAULT_TPOSE_MESH if os.path.isfile(_DEFAULT_TPOSE_MESH) else _DEFAULT_GLB
    if anim_glb is None:
        anim_glb = _DEFAULT_GLB

    for file_path in [tpose_mesh, anim_glb]:
        assert os.path.isfile(file_path), f"Missing required file: {file_path}"

    temp_context = nullcontext(output_dir) if output_dir else tempfile.TemporaryDirectory()
    with temp_context as work_dir:
        assert work_dir is not None
        os.makedirs(work_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(anim_glb))[0]
        recovered_glb = os.path.join(work_dir, f"{base_name}_recovered.glb")
        npy_path = os.path.join(work_dir, f"{base_name}_features.npy")

        # Phase A: Load T-pose metadata used by the production preprocessing path
        print("[Phase A] Loading T-pose GLB preprocessing metadata...")
        cond_entry = load_cond_dict().get(object_type)
        cond_entry = _require_dataset_cond_entry_or_skip(cond_entry, object_type)
        preprocess_max_joints = (
            len(cond_entry["parents"])
            if cond_entry is not None and cond_entry.get("parents") is not None
            else MAX_JOINTS
        )
        tp: TPoseFeatures = get_common_features_from_T_pose(
            tpose_mesh,
            object_type,
            max_joints=preprocess_max_joints,
        )
        if cond_entry is None or "scale_factor" not in cond_entry:
            print(f"[WARN] scale_factor missing from cond.npy for {object_type}; falling back to T-pose metadata")
            scale_factor = float(tp.scale_factor)
        else:
            scale_factor = float(cond_entry["scale_factor"])
        print(f"Joints: {len(tp.names)}, scale_factor: {scale_factor:.6f}")

        # Phase B: Load source GLB and build the production preprocessed motion/features
        print("[Phase B] Loading source GLB and building production preprocessed motion...")
        source_anim, source_bone_names, source_frame_time = FBX.load(anim_glb)
        source_fps = 1.0 / source_frame_time if source_frame_time > 0 else 30.0
        print(f"Frames: {len(source_anim)}, Joints: {source_anim.shape[1]}, FPS: {source_fps:.1f}")

        squared_positions_error: dict[str, float] = {}
        features, feature_parents, _max_joints, feature_anim, _baseline_export_anim, _is_loop, _motion_translation_root_index, motion_root_translation_xz = get_motion(
            anim_glb,
            FOOT_CONTACT_VEL_THRESH,
            object_type,
            preprocess_max_joints,
            tp.offsets,
            tp.foot_indices,
            tp.tpos_rots,
            squared_positions_error,
            scale_factor=scale_factor,
            orientation_quat=tp.orientation_quat,
            preloaded=(source_anim, source_bone_names),
        )

        if feature_anim is None or _baseline_export_anim is None:
            raise AssertionError("Production preprocessing failed to build feature/export animations")
        if motion_root_translation_xz is None:
            raise AssertionError("Production preprocessing did not report a per-clip root_translation_xz")
        if feature_anim.shape[1] != len(tp.names):
            raise AssertionError(
                "Production preprocessing joint count does not match the T-pose feature skeleton"
            )

        # Phase C: Save the production bare NPY tensor exactly like sample/generate.py
        print("[Phase C] Saving production bare NPY feature tensor...")
        features = np.asarray(features, dtype=np.float32)
        if features.shape[1] != len(tp.names):
            raise AssertionError(
                "Production bare NPY joint count does not match the helper-aware T-pose feature skeleton"
            )
        np.save(npy_path, features, allow_pickle=False)
        print(f"Saved NPY features to {npy_path}")

        # Phase D: Restore the production bare tensor through the real restore tool
        print(f"[Phase D] Restoring production NPY via restore_glb_from_npy.py -> {os.path.basename(recovered_glb)}...")
        restore_glb(
            npy_path=npy_path,
            tpose_mesh=tpose_mesh,
            output_glb=recovered_glb,
            object_type=object_type,
            fps=source_fps,
            root_translation_xz=motion_root_translation_xz,
        )

        # Phase E: Compare recovered GLB vs original source GLB using compare_motions.py
        recovered_report = _compare_export_to_source(anim_glb, recovered_glb, "RecoveredGLB")

        errors: list[str] = list()
        errors.extend(recovered_report["errors"])

        # Map worst_frame to sample time for the return dict
        worst_frame = recovered_report["position_worst_frame"]
        source_motion = load_motion(anim_glb)
        recovered_worst_time = float(source_motion.sample_times[worst_frame]) \
            if worst_frame < len(source_motion.sample_times) else 0.0

        # Print PASS/FAIL
        passed = len(errors) == 0
        status = "PASS" if passed else f"{_COLOR_RED}FAIL{_COLOR_RESET}"
        if errors:
            for err in errors:
                print(f"\n  [{status}] {_COLOR_RED}{err}{_COLOR_RESET}")
        else:
            print(f"\n  [{status}] GLB -> NPY -> GLB roundtrip checks passed")
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

def test_glb_npy_glb_roundtrip() -> None:
    """Pytest-compatible entry point. Asserts GLB->NPY->GLB roundtrip passes."""
    result = _run_test_glb_npy_glb_roundtrip()
    assert result["passed"], "GLB->NPY->GLB roundtrip failed: " + "; ".join(result["errors"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GLB -> NPY -> GLB roundtrip smoke test",
    )
    parser.add_argument(
        "--tpose-mesh",
        default=None,
        help="Path to the T-pose GLB used for preprocessing metadata and as the export container.",
    )
    parser.add_argument(
        "--glb",
        default=None,
        help="Path to source animation GLB file.",
    )
    parser.add_argument(
        "--object-type",
        default=None,
        help="Character type for contact inference. Inferred from the GLB filename if not specified.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(_ANYTOP_ROOT, "outputs", "glb_npy_roundtrip"),
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

    if args.tpose_mesh is None:
        args.tpose_mesh = _DEFAULT_TPOSE_MESH if os.path.isfile(_DEFAULT_TPOSE_MESH) else args.glb
        args.object_type = _DEFAULT_OBJECT_TYPE

    if args.object_type is None:
        from utils.misc import infer_object_type_from_filename
        args.object_type = infer_object_type_from_filename(args.glb) or os.path.basename(args.glb)

    print(f"T-pose GLB : {args.tpose_mesh}")
    print(f"Anim GLB   : {args.glb}")
    print(f"Output dir : {args.output_dir}")
    print(f"Object type: {args.object_type}")
    print(f"Tolerance  : {args.tolerance}")
    print()

    result = _run_test_glb_npy_glb_roundtrip(
        tpose_mesh=args.tpose_mesh,
        anim_glb=args.glb,
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
