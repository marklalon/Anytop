"""test_debug_preprocessing.py

Processes all BVH files found under dataset/truebones/zoo/debug/<Animal>/,
writes processed motion .npy files alongside the sources, then validates
that --- with ROOT_XZ_STRIP_THRESHOLD = 100 (effectively disabled) ---
the processed motion faithfully preserves the original skeleton dynamics
up to the expected transforms:
  1. HML face-orientation rotation   (global rigid rotation `qs_rot`)
  2. XZ centering of the first frame  (shift by `root_pose_init_xz`)
  3. Uniform skeletal scaling         (factor `scale_factor`)

Validation strategy
-------------------
Let R be the 3x3 rotation matrix for the HML orientation quaternion, t the
3D centering shift (y=0), and s the scale factor.  Then for every joint j
and frame f:

    processed_global[f,j] ≈ R @ raw_global[f,j] * s - t * s

Inverse:

    raw_global[f,j] ≈ R^T @ (processed_global[f,j] / s + t)

_bake_descendant_y_into_translation_root() may introduce small Y discrepancies
on certain rigs (Horse, Bear, Camel).  The script reports these separately so
unexpected large errors are clearly visible.

Usage (from Anytop directory with venv active):
    python tests/test_debug_preprocessing.py
"""

import sys
import os

# Ensure imports resolve correctly when run from the Anytop directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
from pathlib import Path

from motion_lib import BVH, Animation
from motion_lib.Animation import positions_global
from data_loaders.truebones.truebones_utils.motion_process import (
    get_common_features_from_T_pose,
    get_hml_aligned_anim,
    _find_translation_root,
    _xz_locomotion_extent,
    ROOT_XZ_STRIP_THRESHOLD,
    find_orientation_reference_path,
)
from data_loaders.truebones.truebones_utils.param_utils import (
    FOOT_CONTACT_VEL_THRESH,
    MOTION_DIR,
    BVHS_DIR,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEBUG_DIR = Path(__file__).parent.parent / "dataset/truebones/zoo/debug"
OUTPUT_SUBDIR = "processed"   # created inside each animal directory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rotation_matrix_from_quat(orientation_quat):
    """Return a (3, 3) numpy rotation matrix from an orientation Quaternions object."""
    R = orientation_quat.transforms()      # (..., 3, 3)
    return R.reshape(3, 3)


def _inverse_transform(proc_global, R, scale_factor, root_pose_init_xz):
    """Invert the pipeline transform (rotate + centre + scale) to recover raw positions.

    processed_global ≈ R @ raw_global * s - t * s
    raw_global       ≈ R^T @ (processed_global / s + t)

    where t = root_pose_init_xz (y == 0).

    Args:
        proc_global      : (F, J, 3) processed global positions
        R                : (3, 3) orientation rotation matrix
        scale_factor     : float
        root_pose_init_xz: (3,) centering shift, produced by move_xz_to_origin
                           (y component is 0)
    Returns:
        (F, J, 3) reconstructed raw global positions
    """
    unscaled   = proc_global / scale_factor           # (F, J, 3)
    uncentered = unscaled + root_pose_init_xz[None, None, :]  # broadcast
    # row-vector multiply: pos_orig = pos_proc @ R  (since forward is pos_proc = pos_orig @ R^T)
    reconstructed = uncentered @ R  # R instead of R^T because row-vector convention
    return reconstructed

def _detect_y_baking(raw_anim, orientation_quat, face_joints, object_type,
                     forward_joint_index, forward_base_joint_index):
    """Return True if _bake_descendant_y_into_translation_root will make changes.

    Baking fires when any joint in the single-child chain below the translation
    root has animated local Y **in the rotated animation** (not raw).
    We must check post-rotation because the rotation may move Y into a different
    axis in the raw frame.
    """
    from data_loaders.truebones.truebones_utils.motion_process import (
        _find_translation_root,
        _find_descendant_transport_chain,
    )
    from data_loaders.truebones.truebones_utils.face_orientation import (
        rotate_to_hml_orientation,
    )
    rotated = rotate_to_hml_orientation(
        raw_anim, object_type, face_joints,
        orientation_quat=orientation_quat,
        forward_joint_index=forward_joint_index,
        forward_base_joint_index=forward_base_joint_index,
    )
    trans_root = _find_translation_root(rotated)
    chain = _find_descendant_transport_chain(rotated.parents, trans_root, max_depth=2)
    bake_joints = [j for j in chain if np.ptp(rotated.positions[:, j, 1]) > 1e-4]
    return len(bake_joints) > 0, trans_root, bake_joints



def _validate_clip(raw_anim, export_anim, R, scale_factor, root_pose_init_xz,
                   orientation_quat, face_joints, object_type,
                   forward_joint_index, forward_base_joint_index):
    """Compare inverse-transformed processed positions against original raw positions.

    Errors are returned in both raw-BVH units and HML-normalised units
    (raw * scale_factor) so they are comparable to a consistent threshold.
    Y-baking is detected on the rotated animation (not raw) to flag intentional
    discrepancies introduced by _bake_descendant_y_into_translation_root.
    """
    is_baking, trans_root_rotated, bake_joints = _detect_y_baking(
        raw_anim, orientation_quat, face_joints, object_type,
        forward_joint_index, forward_base_joint_index,
    )

    raw_global  = positions_global(raw_anim)
    proc_global = positions_global(export_anim)

    reconstructed = _inverse_transform(proc_global, R, scale_factor, root_pose_init_xz)

    n_frames = min(raw_global.shape[0], reconstructed.shape[0])
    raw_global    = raw_global[:n_frames]
    reconstructed = reconstructed[:n_frames]

    diff = reconstructed - raw_global
    per_axis_max_raw = np.abs(diff).max(axis=(0, 1))
    overall_max_raw  = float(np.abs(diff).max())
    overall_rms_raw  = float(np.sqrt((diff ** 2).mean()))

    overall_max_hml  = overall_max_raw * scale_factor
    overall_rms_hml  = overall_rms_raw * scale_factor
    per_axis_max_hml = per_axis_max_raw * scale_factor

    return {
        "overall_max_err_raw": overall_max_raw,
        "overall_rms_err_raw": overall_rms_raw,
        "overall_max_err_hml": overall_max_hml,
        "overall_rms_err_hml": overall_rms_hml,
        "per_axis_max_hml": per_axis_max_hml,
        "n_frames": n_frames,
        "n_joints": raw_global.shape[1],
        "is_y_baking": is_baking,
        "bake_joints": bake_joints,
        "trans_root_rotated": trans_root_rotated,
    }



def _print_validation(clip_name, xz_extent, stats):
    stripped_note = (
        "  [STRIPPED — threshold exceeded!]"
        if xz_extent > ROOT_XZ_STRIP_THRESHOLD
        else f"  (threshold {ROOT_XZ_STRIP_THRESHOLD}, OK)"
    )
    is_baking    = stats["is_y_baking"]
    bake_joints  = stats["bake_joints"]
    trans_root   = stats["trans_root_rotated"]
    # Baking rigs: allow up to 0.5 HML; non-baking: tight 0.05 HML threshold
    hml_threshold = 0.5 if is_baking else 0.05
    ok_recon   = stats["overall_max_err_hml"] < hml_threshold
    recon_note = "OK" if ok_recon else "WARNING: large error"

    print(f"  clip        : {clip_name}")
    print(f"  frames/joints: {stats['n_frames']} / {stats['n_joints']}")
    baking_note = f"  (Y-baking active — joints {bake_joints})" if is_baking else ""
    print(f"  trans_root  : joint {trans_root}{baking_note}")
    print(f"  xz_extent   : {xz_extent:.4f}{stripped_note}")
    hml_str = f"HML max={stats['overall_max_err_hml']:.5f}  rms={stats['overall_rms_err_hml']:.5f}"
    raw_str = f"raw max={stats['overall_max_err_raw']:.4f}  rms={stats['overall_rms_err_raw']:.4f}"
    print(f"  roundtrip   : {hml_str}  [{recon_note}]")
    print(f"              : {raw_str}")
    x_hml, y_hml, z_hml = stats["per_axis_max_hml"]
    print(f"    axis max (HML): x={x_hml:.5f}  y={y_hml:.5f}  z={z_hml:.5f}")
    if is_baking:
        print(f"    NOTE: Y-baking modifies the above joints intentionally — discrepancy is expected.")



# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_animal(animal_dir: Path):
    animal_name = animal_dir.name
    bvh_files = sorted(animal_dir.glob("*.bvh"))
    if not bvh_files:
        print(f"[{animal_name}] No BVH files found, skipping.")
        return

    bvh_file_strs = [str(f) for f in bvh_files]

    print(f"\n{'='*60}")
    print(f"  Animal: {animal_name}  ({len(bvh_file_strs)} BVH file(s))")
    print(f"{'='*60}")

    # --- Resolve T-pose reference ------------------------------------------
    # Pass a copy so find_orientation_reference_path's internal remove() doesn't
    # mutate bvh_file_strs (we need the full list for motion_files below).
    ref_path, ref_source = find_orientation_reference_path(list(bvh_file_strs))
    print(f"  Orientation reference: {os.path.basename(ref_path)}  (source: {ref_source})")

    # Production code (_prepare_object_outputs) only removes a T-pose reference
    # from the motion list (inside find_orientation_reference_path itself).
    # idle / walk / fallback references are also processed as motion clips.
    if ref_source == "tpose":
        motion_files = [f for f in bvh_file_strs if f != ref_path]
    else:
        motion_files = list(bvh_file_strs)

    # --- Extract common features from T-pose BVH ---------------------------
    (
        root_pose_init_xz,
        scale_factor,
        offsets,
        foot_indices,
        tpos_rots,
        names,
        tpos_anim,
        face_joints,
        orientation_quat,
        forward_joint_index,
        forward_base_joint_index,
        contact_joint_source,
    ) = get_common_features_from_T_pose(ref_path, animal_name)

    print(f"  Scale factor  : {scale_factor:.6f}")
    print(f"  XZ centering  : ({root_pose_init_xz[0]:.4f}, {root_pose_init_xz[2]:.4f})")
    print(f"  Joints        : {len(names)}")
    print(f"  Contact joints: {contact_joint_source}")

    R = _rotation_matrix_from_quat(orientation_quat)

    # --- Output directory ---------------------------------------------------
    out_dir = animal_dir / OUTPUT_SUBDIR
    out_dir.mkdir(exist_ok=True)
    motion_out  = out_dir / MOTION_DIR
    bvh_out     = out_dir / BVHS_DIR
    motion_out.mkdir(exist_ok=True)
    bvh_out.mkdir(exist_ok=True)

    # --- Process each motion clip -------------------------------------------
    clip_results = []
    errors = {}
    for bvh_path in motion_files:
        clip_name = os.path.splitext(os.path.basename(bvh_path))[0]
        print(f"\n  --- {clip_name} ---")
        try:
            raw_anim, raw_names, _ = BVH.load(bvh_path)

            new_anim, export_anim, _names = get_hml_aligned_anim(
                bvh_path,
                animal_name,
                root_pose_init_xz,
                scale_factor,
                tpos_rots,
                offsets,
                errors,
                foot_indices,
                face_joints,
                orientation_quat,
                forward_joint_index,
                forward_base_joint_index,
            )
        except Exception as exc:
            print(f"  ERROR during processing: {exc}")
            continue

        # Save processed motion .npy --------------------------------------
        from data_loaders.truebones.truebones_utils.motion_process import (
            get_motion,
        )
        local_errors = {}
        motion, parents, _, _, _ = get_motion(
            bvh_path,
            FOOT_CONTACT_VEL_THRESH,
            animal_name,
            max_joints=23,
            root_pose_init_xz=root_pose_init_xz,
            scale_factor=scale_factor,
            offsets=offsets,
            foot_indices=foot_indices,
            tpos_rots=tpos_rots,
            squared_positions_error=local_errors,
            face_joints=face_joints,
            orientation_quat=orientation_quat,
            forward_joint_index=forward_joint_index,
            forward_base_joint_index=forward_base_joint_index,
        )
        if motion is not None:
            npy_path = motion_out / f"{animal_name}_{clip_name}.npy"
            np.save(str(npy_path), motion)
            print(f"  Saved motion  : {npy_path.relative_to(animal_dir)}")
        else:
            print("  WARNING: get_motion returned None, no .npy saved.")

        # Save processed BVH ----------------------------------------------
        has_animated_nonroot = np.any(
            np.ptp(export_anim.positions[:, 1:, :], axis=0) > 1e-4
        )
        bvh_path_out = bvh_out / f"{animal_name}_{clip_name}.bvh"
        BVH.save(str(bvh_path_out), export_anim, names, positions=bool(has_animated_nonroot))
        print(f"  Saved BVH     : {bvh_path_out.relative_to(animal_dir)}")

        # -----------------------------------------------------------------
        # XZ locomotion check
        # -----------------------------------------------------------------
        trans_root = _find_translation_root(export_anim)
        xz_extent  = _xz_locomotion_extent(export_anim, trans_root)

        # -----------------------------------------------------------------
        # Roundtrip validation
        # -----------------------------------------------------------------
        stats = _validate_clip(
            raw_anim, export_anim, R, scale_factor, root_pose_init_xz,
            orientation_quat, face_joints, animal_name,
            forward_joint_index, forward_base_joint_index,
        )
        _print_validation(clip_name, xz_extent, stats)

        clip_results.append({
            "clip": clip_name,
            "xz_extent": xz_extent,
            "xz_stripped": xz_extent > ROOT_XZ_STRIP_THRESHOLD,
            "baking_rig": stats["is_y_baking"],
            **stats,
        })

    # --- Summary ------------------------------------------------------------
    print(f"\n  {'─'*50}")
    print(f"  SUMMARY for {animal_name}  ({len(clip_results)} clips processed)")
    all_ok_xz      = all(not r["xz_stripped"] for r in clip_results)
    max_err_hml    = max((r["overall_max_err_hml"] for r in clip_results), default=float("nan"))
    # For the summary, non-baking clips have a tight tolerance; baking rigs allow more
    bad_non_baking = [r for r in clip_results if not r["baking_rig"] and r["overall_max_err_hml"] >= 0.05]
    bad_baking     = [r for r in clip_results if r["baking_rig"]     and r["overall_max_err_hml"] >= 0.5]
    print(f"  XZ stripping  : {'none (threshold not exceeded)' if all_ok_xz else 'WARNING: some clips were stripped!'}")
    print(f"  Roundtrip max : {max_err_hml:.5f} HML units")
    if bad_non_baking:
        print(f"  BAD (non-baking): {[r['clip'] for r in bad_non_baking]}")
    if bad_baking:
        print(f"  BAD (Y-baking) : {[r['clip'] for r in bad_baking]}")

    return clip_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print(f"ROOT_XZ_STRIP_THRESHOLD = {ROOT_XZ_STRIP_THRESHOLD}  (locomotion stripping {'effectively disabled' if ROOT_XZ_STRIP_THRESHOLD >= 100 else 'ACTIVE'})")
    print(f"Debug directory: {DEBUG_DIR}")

    animal_dirs = sorted(p for p in DEBUG_DIR.iterdir() if p.is_dir())
    if not animal_dirs:
        print("No animal subdirectories found.  Exiting.")
        return

    all_results = {}
    for animal_dir in animal_dirs:
        results = process_animal(animal_dir)
        if results:
            all_results[animal_dir.name] = results

    # ---- Global summary ----------------------------------------------------
    print(f"\n{'='*60}")
    print("GLOBAL VALIDATION SUMMARY")
    print(f"{'='*60}")
    total_clips   = sum(len(v) for v in all_results.values())
    total_stripped = sum(sum(1 for r in v if r["xz_stripped"]) for v in all_results.values())
    worst_max_hml  = max(
        (r["overall_max_err_hml"] for v in all_results.values() for r in v),
        default=float("nan"),
    )
    worst_y_hml = max(
        (r["per_axis_max_hml"][1] for v in all_results.values() for r in v),
        default=float("nan"),
    )
    bad_non_baking = [
        (animal, r["clip"])
        for animal, clips in all_results.items()
        for r in clips
        if not r["baking_rig"] and r["overall_max_err_hml"] >= 0.05
    ]
    bad_baking = [
        (animal, r["clip"])
        for animal, clips in all_results.items()
        for r in clips
        if r["baking_rig"] and r["overall_max_err_hml"] >= 0.5
    ]

    print(f"Total clips processed : {total_clips}")
    print(f"XZ stripping events   : {total_stripped} / {total_clips}  "
          f"({'NONE — as expected with threshold={}'.format(ROOT_XZ_STRIP_THRESHOLD) if total_stripped == 0 else 'UNEXPECTED'})")
    print(f"Worst roundtrip error : {worst_max_hml:.5f} HML units")
    print(f"Worst Y-axis error    : {worst_y_hml:.5f} HML units  "
          f"(elevated for rigs with trans_root != 0 / Y-baking)")
    print(f"Unexpected failures   : {len(bad_non_baking)} non-baking, {len(bad_baking)} baking")
    if bad_non_baking:
        print(f"  Non-baking failures: {bad_non_baking}")
    if bad_baking:
        print(f"  Baking failures    : {bad_baking}")

    verdict = "PASS" if total_stripped == 0 and not bad_non_baking and not bad_baking else "FAIL"
    print(f"\nOverall verdict: {verdict}")
    if verdict == "FAIL":
        print("  Check per-clip output above for the failing clips.")



if __name__ == "__main__":
    main()
