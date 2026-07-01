"""
Retarget roundtrip test: GLB -> retarget NPY -> restore GLB -> compare vs source.

Locks in the cond-free file-retarget path
(``utils.auto_retarget.retarget_animation_file_to_target``): a raw source
animation is retargeted onto a target skeleton purely from the file (no source
cond entry), restored to a GLB through the production ``restore_glb_from_npy``
tool, and compared against the original source with ``tools/compare_motions``.

The default case is a Buffalo-RunLoop **self-retarget** (source and target are
the same Buffalo skeleton), so the restored clip must reproduce the source
motion. The retarget canonicalizes facing via an on-the-fly ``orientation_quat``
computed from the source bind pose; the restore inverts the target's
canonicalization, so a correct pipeline lands the restored GLB back in the
source's native world basis. That makes the rigid alignment between restore and
source **identity** — the primary regression guard: a broken facing step would
leave the clip in its native (e.g. +X) authoring basis relative to the +Z
canonical features fed to the model, which surfaces here as a non-identity
coordinate alignment and a large position error.

What this test asserts:
  * identity rigid alignment  -- facing canonicalization is correct;
  * joint-position error      -- the reproduced motion matches the source;
    * local bone-direction error-- the per-joint pose channels match.

The raw file path is intentionally source-cond-free and, when a sibling reference
file is present, first normalizes the source into the same rest-pose feature base as
matching NPY inputs. That means this GLB smoke test gates on rigid alignment,
joint positions, and bone directions. Full raw-rig twist / mesh-surface identity
is covered by export/restore-specific tests, not by this feature-space retarget
path; the exact retarget NPY equivalence is locked below by
``test_retarget_raw_glb_matches_matching_npy_reference``.

Requires bpy (Blender as a Python module).
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

# ── Resolve utils namespace conflict (mirrors test_fbx_npy_glb_roundtrip) ────
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
from data_loaders.truebones.truebones_utils.motion_process import (
    get_common_features_from_T_pose,
    tpose_features_from_cond,
    TPoseFeatures,
)
from utils.auto_retarget import retarget_animation_file_to_target, retarget_features_npy_to_target
from utils.misc import normalize_bone_key as _normalize_bone_key
from tools.restore_glb_from_npy import restore_glb
from tools.compare_motions import (
    load_motion,
    _validate_compatible,
    detect_and_align,
    compare_motions,
    print_summary,
)


# ── Tolerances ──────────────────────────────────────────────────────────────
# Measured on the Buffalo-RunLoop self-retarget: position max ~1.37% char,
# bone-direction max ~0.17 deg. Tolerances keep headroom over the observed
# values while staying tight enough to catch a regressed facing/transfer.
_POS_TOLERANCE_PCT_CHAR = 2.5
_ROT_TOLERANCE_DEG = 1.5
_COLOR_RED = "\033[31m"
_COLOR_RESET = "\033[0m"

_DEFAULT_SOURCE = os.path.join(
    _ANYTOP_ROOT, "dataset", "truebones", "zoo", "Truebone_Z-OO", "Buffalo", "Buffalo-RunLoop.glb",
)
_DEFAULT_TPOSE_MESH = os.path.join(
    _ANYTOP_ROOT, "dataset", "truebones", "zoo", "Truebone_Z-OO", "Buffalo", "Buffalo-TPOSE.glb",
)
_DEFAULT_TARGET_TYPE = "Buffalo"
_DEFAULT_MATCHING_NPY = os.path.join(
    _ANYTOP_ROOT,
    "dataset",
    "truebones",
    "zoo",
    "truebones_processed",
    "motions",
    "Buffalo_RunLoop_1.npy",
)
_DEFAULT_SOURCE_TYPE = "Buffalo"
_DEFAULT_CROSS_TARGET_TYPE = "Deer"


def _require_or_skip(condition: bool, message: str) -> None:
    if condition:
        return
    if "PYTEST_CURRENT_TEST" in os.environ:
        pytest.skip(message)
    raise AssertionError(message)


def _run_retarget_glb_roundtrip(
    source_motion: str | None = None,
    tpose_mesh: str | None = None,
    target_type: str = _DEFAULT_TARGET_TYPE,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """GLB -> retarget NPY -> restore GLB -> compare vs source.

    Pipeline:
        1. Build the target rest-pose features + resolve the target cond entry.
        2. Retarget the raw source animation onto the target skeleton straight
           from the file (cond-free source) -> (F, J, 13) feature tensor.
        3. Save the bare NPY exactly like the generation pipeline.
        4. Restore it to a GLB through the production restore tool.
        5. Compare the restored GLB against the original source motion.
    """
    source_motion = source_motion or _DEFAULT_SOURCE
    tpose_mesh = tpose_mesh or _DEFAULT_TPOSE_MESH

    _require_or_skip(os.path.isfile(source_motion), f"Missing source motion: {source_motion}")
    _require_or_skip(os.path.isfile(tpose_mesh), f"Missing target rest-pose reference: {tpose_mesh}")

    target_cond = load_cond_dict().get(target_type)
    _require_or_skip(
        target_cond is not None,
        f"cond.npy is missing a '{target_type}' entry; regenerate or point the test at a matching dataset",
    )

    max_joints = (
        len(np.asarray(target_cond["parents"]))
        if target_cond.get("parents") is not None
        else None
    )
    _require_or_skip(max_joints is not None, f"cond entry for '{target_type}' has no parents/joint count")

    # Source fps (for the restored GLB frame timing only).
    _src_anim, _src_names, src_frame_time = FBX.load(source_motion)
    source_fps = 1.0 / src_frame_time if src_frame_time and src_frame_time > 0 else 30.0

    temp_context = nullcontext(output_dir) if output_dir else tempfile.TemporaryDirectory()
    with temp_context as work_dir:
        assert work_dir is not None
        os.makedirs(work_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(source_motion))[0]
        npy_path = os.path.join(work_dir, f"{base_name}_retargeted_to_{target_type}.npy")
        recovered_glb = os.path.join(work_dir, f"{base_name}_retargeted_to_{target_type}.glb")

        # Phase A: target rest-pose features (helper-augmented, like generation).
        print(f"[Phase A] Building target rest-pose features for {target_type}...")
        target_tp: TPoseFeatures = get_common_features_from_T_pose(
            tpose_mesh,
            target_type,
            max_joints=max_joints,
        )

        # Phase B: cond-free file retarget -> feature tensor.
        print(f"[Phase B] Retargeting {os.path.basename(source_motion)} onto {target_type} (cond-free source)...")
        features = retarget_animation_file_to_target(
            source_motion,
            target_tp,
            target_type,
            max_joints,
            target_cond,
        )
        _require_or_skip(features is not None, "retarget_animation_file_to_target returned None")
        features = np.asarray(features, dtype=np.float32)
        if features.shape[1] != len(target_tp.names):
            raise AssertionError(
                f"Retarget joint count {features.shape[1]} does not match target feature "
                f"skeleton {len(target_tp.names)}"
            )

        # Phase C: save the bare NPY exactly like sample/generate.py.
        print("[Phase C] Saving bare NPY feature tensor...")
        np.save(npy_path, features, allow_pickle=False)

        # Phase D: restore to GLB through the production restore tool.
        print(f"[Phase D] Restoring NPY -> {os.path.basename(recovered_glb)}...")
        restore_glb(
            npy_path=npy_path,
            tpose_mesh=tpose_mesh,
            output_glb=recovered_glb,
            object_type=target_type,
            fps=source_fps,
        )

        # Phase E: compare restored GLB vs source.
        print("[Phase E] Comparing restored GLB against source motion...")
        motion_a = load_motion(source_motion)
        motion_b = load_motion(recovered_glb)
        _validate_compatible(motion_a, motion_b)
        motion_b_aligned, alignment = detect_and_align(motion_a, motion_b)
        result = compare_motions(motion_a, motion_b_aligned, alignment)
        print_summary(motion_a, motion_b, alignment, result)

        pos_result = result["position"]
        rot_result = result["rotation"]
        world_quat_result = result["world_quaternion"]

        # Restrict the world-quat gate to "encoded" joints (those with at least
        # one child), matching test_fbx_npy_glb_roundtrip: leaf-only joints carry
        # no encoded local rotation, so their world frame about the bone axis is
        # genuinely undetermined and excluded.
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
            world_quat_result["per_bone"]["max_deg"], dtype=np.float64
        )
        encoded_world_quat_max = float(
            encoded_world_quat_per_bone[encoded_joint_mask].max()
            if np.any(encoded_joint_mask)
            else 0.0
        )

        errors: list[str] = []
        if alignment.rotation_label != "identity":
            errors.append(
                f"unexpected rigid coordinate alignment {alignment.rotation_label!r}; the retarget "
                "should reproduce the source-facing world basis (a non-identity alignment means the "
                "facing canonicalization regressed and the features are OOD)"
            )
        if pos_result["max_error_pct_char"] > _POS_TOLERANCE_PCT_CHAR:
            errors.append(
                f"position max error {pos_result['max_error']:.6f} "
                f"({pos_result['max_error_pct_char']:.4f}% char) exceeds {_POS_TOLERANCE_PCT_CHAR:.4f}% "
                f"(worst bone={pos_result['worst_bone']}, frame={pos_result['worst_frame']})"
            )
        if rot_result["max_error_deg"] > _ROT_TOLERANCE_DEG:
            errors.append(
                f"local rotation max error {rot_result['max_error_deg']:.6f} deg exceeds "
                f"{_ROT_TOLERANCE_DEG:.6f} "
                f"(worst bone={rot_result['worst_bone']}, frame={rot_result['worst_frame']})"
            )
        if encoded_world_quat_max > _ROT_TOLERANCE_DEG:
            errors.append(
                f"encoded-joint world quaternion max error {encoded_world_quat_max:.6f} deg exceeds "
                f"{_ROT_TOLERANCE_DEG:.6f} "
                f"(worst bone={world_quat_result['worst_bone']})"
            )

        mesh_mean_pct_char = None
        mesh_p99_pct_char = None
        mesh_result = result.get("mesh_surface")
        if mesh_result is not None:
            char_size = max(float(pos_result["character_size"]), 1e-8)
            mesh_mean_pct_char = mesh_result["mean"] / char_size * 100.0
            mesh_p99_pct_char = mesh_result["p99"] / char_size * 100.0
            if mesh_mean_pct_char > _POS_TOLERANCE_PCT_CHAR:
                errors.append(
                    f"mesh mean error {mesh_result['mean']:.6f} "
                    f"({mesh_mean_pct_char:.4f}% char) exceeds {_POS_TOLERANCE_PCT_CHAR:.4f}%"
                )
            if mesh_p99_pct_char > _POS_TOLERANCE_PCT_CHAR * 4:
                errors.append(
                    f"mesh p99 error {mesh_result['p99']:.6f} "
                    f"({mesh_p99_pct_char:.4f}% char) exceeds {_POS_TOLERANCE_PCT_CHAR * 4:.4f}%"
                )

        passed = len(errors) == 0
        status = "PASS" if passed else f"{_COLOR_RED}FAIL{_COLOR_RESET}"
        if errors:
            for err in errors:
                print(f"\n  [{status}] {_COLOR_RED}{err}{_COLOR_RESET}")
        else:
            print(f"\n  [{status}] retarget GLB -> NPY -> GLB roundtrip checks passed")

        return {
            "passed": passed,
            "errors": errors,
            "alignment_label": alignment.rotation_label,
            "position_max_error_pct_char": float(pos_result["max_error_pct_char"]),
            "position_worst_bone": pos_result["worst_bone"],
            "rotation_max_error_deg": float(rot_result["max_error_deg"]),
            "rotation_worst_bone": rot_result["worst_bone"],
            "encoded_world_quaternion_max_error_deg": encoded_world_quat_max,
            "world_quaternion_worst_bone": world_quat_result["worst_bone"],
            "mesh_mean_error_pct_char": mesh_mean_pct_char,
            "mesh_p99_error_pct_char": mesh_p99_pct_char,
        }


# ── pytest entry point ────────────────────────────────────────────────────────

def test_retarget_glb_roundtrip() -> None:
    """Buffalo-RunLoop self-retarget roundtrip: restore must match the source."""
    result = _run_retarget_glb_roundtrip()
    assert result["passed"], "retarget GLB->NPY->GLB roundtrip failed: " + "; ".join(result["errors"])


@pytest.mark.parametrize("target_type", [_DEFAULT_SOURCE_TYPE, _DEFAULT_CROSS_TARGET_TYPE])
def test_retarget_raw_glb_matches_matching_npy_reference(target_type: str) -> None:
    """Matching GLB and NPY references must retarget to the same target features."""
    _require_or_skip(os.path.isfile(_DEFAULT_SOURCE), f"Missing source GLB: {_DEFAULT_SOURCE}")
    _require_or_skip(os.path.isfile(_DEFAULT_MATCHING_NPY), f"Missing source NPY: {_DEFAULT_MATCHING_NPY}")

    cond_dict = load_cond_dict()
    source_cond = cond_dict.get(_DEFAULT_SOURCE_TYPE)
    target_cond = cond_dict.get(target_type)
    _require_or_skip(source_cond is not None, f"cond.npy is missing '{_DEFAULT_SOURCE_TYPE}'")
    _require_or_skip(target_cond is not None, f"cond.npy is missing '{target_type}'")
    _require_or_skip(
        'tpose_rest_rotations' in target_cond,
        "cond.npy predates the baked 'tpose_rest_rotations' field; regenerate cond",
    )

    max_joints = max(
        len(np.asarray(source_cond["parents"])),
        len(np.asarray(target_cond["parents"])),
    )
    # Target rest-pose features come straight from cond (no T-pose mesh access).
    target_tp = tpose_features_from_cond(target_cond, target_type)

    raw_file_features = retarget_animation_file_to_target(
        _DEFAULT_SOURCE,
        target_tp,
        target_type,
        max_joints,
        target_cond,
    )
    source_features = np.load(_DEFAULT_MATCHING_NPY).astype(np.float32)
    npy_features = retarget_features_npy_to_target(
        source_features,
        source_cond,
        _DEFAULT_SOURCE_TYPE,
        target_tp,
        target_type,
        max_joints,
        target_cond=target_cond,
    )

    _require_or_skip(raw_file_features is not None, "retarget_animation_file_to_target returned None")
    _require_or_skip(npy_features is not None, "retarget_features_npy_to_target returned None")
    raw_file_features = np.asarray(raw_file_features, dtype=np.float32)
    npy_features = np.asarray(npy_features, dtype=np.float32)
    assert raw_file_features.shape == npy_features.shape

    diff = np.abs(raw_file_features - npy_features)
    assert float(diff.max()) <= 1e-5, (
        "GLB cond-free retarget and matching NPY retarget diverged: "
        f"max={float(diff.max()):.8g}, mean={float(diff.mean()):.8g}, "
        f"argmax={np.unravel_index(int(np.argmax(diff)), diff.shape)}"
    )

    if target_type == _DEFAULT_SOURCE_TYPE:
        self_diff = np.abs(npy_features - source_features)
        assert float(self_diff.max()) <= 1e-3, (
            "NPY self-retarget should preserve all feature channels: "
            f"max={float(self_diff.max()):.8g}, mean={float(self_diff.mean()):.8g}, "
            f"argmax={np.unravel_index(int(np.argmax(self_diff)), self_diff.shape)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retarget GLB -> NPY -> GLB roundtrip test")
    parser.add_argument("--source", default=None, help="Source animation GLB to retarget.")
    parser.add_argument("--tpose-mesh", default=None, help="Target rest-pose reference GLB (skeleton + restore container).")
    parser.add_argument("--target-type", default=_DEFAULT_TARGET_TYPE, help="Target object type (must be in cond.npy).")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(_ANYTOP_ROOT, "outputs", "retarget_glb_roundtrip"),
        help="Directory to save roundtrip artifacts.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Source     : {args.source or _DEFAULT_SOURCE}")
    print(f"Reference  : {args.tpose_mesh or _DEFAULT_TPOSE_MESH}")
    print(f"Target type: {args.target_type}")
    print(f"Output dir : {args.output_dir}")
    print()

    result = _run_retarget_glb_roundtrip(
        source_motion=args.source,
        tpose_mesh=args.tpose_mesh,
        target_type=args.target_type,
        output_dir=args.output_dir,
    )

    mesh_p99 = result.get("mesh_p99_error_pct_char")
    mesh_str = f"{mesh_p99:.4f}%" if mesh_p99 is not None else "n/a"
    print("\nSummary:")
    print(
        f"  align={result['alignment_label']}  "
        f"pos_max={result['position_max_error_pct_char']:.4f}% (bone={result['position_worst_bone']})  "
        f"rot_max={result['rotation_max_error_deg']:.4f}deg (bone={result['rotation_worst_bone']})  "
        f"wquat_max={result['encoded_world_quaternion_max_error_deg']:.4f}deg "
        f"(bone={result['world_quaternion_worst_bone']})  "
        f"mesh_p99={mesh_str}"
    )
    sys.exit(0 if result["passed"] else 1)
