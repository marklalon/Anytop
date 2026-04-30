"""
restore_bvh_bone_names.py

Restore original BVH bone names, count, and scale from Anytop's processed/generated BVHs.

Anytop's preprocessing:
  1. Canonicalizes bone names (e.g., Bip01_L_Foot → LeftFoot)
  2. BVH.save() collapses leaf joints into unnamed End Sites
  3. Scales positions & offsets so mean bone length = SMPL mean (HML_AVG_BONELEN)

This script reverses all three:

    scale_factor = HML_AVG_BONELEN / mean(raw_bone_lengths)
    restored_positions = processed_positions / scale_factor
    restored_offsets   = processed_offsets   / scale_factor

The scale_factor can come from three sources:
  - cond.npy['scale_factor'] (from future preprocessing runs)
  - --raw_bvh (previeous/preprocessed data where cond.npy lacks it)
  - --scale_factor CLI argument

cond.npy is loaded from the default dataset path automatically;
use --cond_npy to override.

Usage:
    # Basic: cond.npy has scale_factor, no raw BVH needed
    python tools/restore_bvh_bones.py --input_bvh in.bvh --output_bvh out.bvh
    
    # Existing data without scale_factor: provide raw BVH
    python tools/restore_bvh_bones.py --object_type Hound \
        --input_bvh in.bvh --output_bvh out.bvh --raw_bvh raw_hound.bvh
    
    # Explicit scale factor
    python tools/restore_bvh_bones.py --object_type Hound \
        --input_bvh in.bvh --output_bvh out.bvh --scale_factor 0.0185
    
    # Batch restore
    python tools/restore_bvh_bones.py --input_dir bvhs/ --output_dir restored/
"""

import argparse
import numpy as np
import os
import sys

# Add Anytop to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANYTOP_DIR = os.path.join(SCRIPT_DIR, '..')
sys.path.insert(0, ANYTOP_DIR)

from motion_lib.BVH import load as bvh_load, save as bvh_save
from motion_lib.Animation import offset_lengths as animation_offset_lengths
from motion_lib.Quaternions import Quaternions


# Default cond.npy path
_DEFAULT_COND_NPY = os.path.realpath(os.path.join(
    ANYTOP_DIR,
    "dataset/truebones/zoo/truebones_processed/cond.npy"
))

HML_AVG_BONELEN = None  # lazy-loaded


def _get_hml_avg_bonelen():
    """Get HML_AVG_BONELEN constant (lazy-loaded to avoid import order issues)."""
    global HML_AVG_BONELEN
    if HML_AVG_BONELEN is None:
        try:
            from data_loaders.truebones.truebones_utils.param_utils import HML_AVG_BONELEN as _val
            HML_AVG_BONELEN = _val
        except ImportError:
            # For cond.npy that predates scale_factor, we just need the actual
            # scale factor, not HML_AVG_BONELEN directly.
            HML_AVG_BONELEN = None
    return HML_AVG_BONELEN


def _compute_scale_factor_from_raw(raw_bvh_path):
    """Compute the scale factor that was applied during preprocessing,
    by loading the raw BVH and comparing with HML_AVG_BONELEN formula.

    scale_factor = HML_AVG_BONELEN / mean(offset_lengths(raw_offsets))
    Uses the same offset_lengths() formula as motion_process.py for consistency.
    """
    raw_anim, raw_names, _ = bvh_load(raw_bvh_path)
    lengths = animation_offset_lengths(raw_anim)
    mean_len = float(np.mean(lengths))
    hml = _get_hml_avg_bonelen()
    if hml is None:
        print("  ERROR: Cannot import HML_AVG_BONELEN; cannot compute scale factor")
        return None
    sf = hml / mean_len
    print(f"  Computed scale_factor from raw BVH: {sf:.6f} "
          f"(HML_AVG={hml:.4f} / mean_bone_len={mean_len:.4f})")
    return sf


def _compute_scale_factor_from_comparison(raw_bvh_path, cond_offsets):
    """Alternative: compute scale_factor by comparing raw offsets with
    cond.npy's (scaled) offsets directly.

    For each bone: scaled_offset[i] = original_offset[i] * scale_factor
    So scale_factor ≈ mean(scaled_len / raw_len)
    """
    raw_anim, raw_names, _ = bvh_load(raw_bvh_path)
    raw_offsets = raw_anim.offsets
    raw_lens = np.linalg.norm(raw_offsets, axis=1)
    scaled_lens = np.linalg.norm(cond_offsets, axis=1)
    # Avoid division by zero for zero-offset joints (e.g. koshi)
    mask = raw_lens > 1e-8
    ratios = scaled_lens[mask] / raw_lens[mask]
    sf = float(np.mean(ratios))
    print(f"  Computed scale_factor from offset comparison: {sf:.6f} "
          f"(from {mask.sum()}/{len(raw_lens)} non-zero offsets)")
    return sf


def _get_scale_factor(obj_cond, raw_bvh_path=None, scale_factor_arg=None):
    """Resolve the scale factor from available sources (priority order):
    1. --scale_factor CLI argument
    2. cond.npy['scale_factor'] (future preprocessing runs)
    3. --raw_bvh path (existing data, auto-compute)
    Returns (scale_factor, source_description) or (None, reason).
    """
    if scale_factor_arg is not None:
        return float(scale_factor_arg), "CLI argument"

    cond_sf = obj_cond.get('scale_factor')
    if cond_sf is not None:
        return float(cond_sf), "cond.npy['scale_factor']"

    if raw_bvh_path is not None:
        sf = _compute_scale_factor_from_raw(raw_bvh_path)
        if sf is not None:
            return sf, f"computed from raw BVH ({os.path.basename(raw_bvh_path)})"
        # Fallback: direct offset comparison
        cond_offsets = obj_cond.get('offsets')
        if cond_offsets is not None:
            sf = _compute_scale_factor_from_comparison(raw_bvh_path, cond_offsets)
            return sf, f"computed from offset comparison ({os.path.basename(raw_bvh_path)})"

    return None, "not available"


def _auto_detect_object_type_from_filename(bvh_path, cond):
    """Auto-detect the object_type from the BVH filename.
    
    Processed BVH filenames follow the pattern: {ObjectType}___{Action}_{ClipID}.bvh
    e.g. Hound___Attack_392.bvh, Alligator___Bite1_365.bvh
    
    Returns the matched object_type or None if not found.
    """
    basename = os.path.splitext(os.path.basename(bvh_path))[0]
    # Try to extract object_type: everything before the first "___"
    sep = "___"
    if sep in basename:
        obj_type = basename.split(sep)[0]
        if obj_type in cond:
            return obj_type
    # Fallback: try progressively longer prefixes for non-standard naming
    # (handles object types that contain underscores, e.g. "Sea_Lion")
    if "_" in basename:
        parts = basename.split("_")
        for i in range(1, len(parts)):
            candidate = "_".join(parts[:i])
            if candidate in cond:
                return candidate
    return None


def _validate_mapping(bvh_names, canonical_names):
    """Validate that the BVH names are compatible with the canonical name mapping.
    Returns True if names match at non-end-site positions, indicating this is a
    processed/generated BVH from this object type."""
    end_site_count = sum(1 for n in bvh_names if n.endswith('_end_site'))

    if len(bvh_names) != len(canonical_names):
        print(f"  WARNING: BVH has {len(bvh_names)} joints but cond.npy has {len(canonical_names)}")
        return False

    mismatches = 0
    for i, (bvh_n, canon_n) in enumerate(zip(bvh_names, canonical_names)):
        if not bvh_n.endswith('_end_site') and bvh_n != canon_n:
            mismatches += 1
            if mismatches <= 3:
                print(f"  Mismatch at index {i}: BVH='{bvh_n}' vs Canonical='{canon_n}'")

    if mismatches > 0:
        print(f"  WARNING: {mismatches} non-end-site name mismatches found")
        return False

    print(f"  Validation OK: {len(bvh_names)} joints, {end_site_count} end sites")
    return True


def restore_bvh(input_bvh, cond_npy, output_bvh, object_type=None,
                raw_bvh=None, scale_factor=None):
    """
    Restore original bone names and scale in a processed/generated BVH.

    Args:
        input_bvh: Path to the input BVH file (processed or model-generated)
        cond_npy: Path to cond.npy file (str) or already-loaded dict
        output_bvh: Path to write the restored BVH
        object_type: Object type key in cond.npy (e.g., "Hound", "Horse").
                     If None, auto-detect from root joint name.
        raw_bvh: Optional raw BVH path for computing scale_factor
                 (used when cond.npy lacks scale_factor).
        scale_factor: Explicit scale factor override.
    """
    # Load cond.npy
    if isinstance(cond_npy, str):
        cond = np.load(cond_npy, allow_pickle=True).item()
    else:
        cond = cond_npy

    # Load input BVH
    anim, bvh_names, frametime = bvh_load(input_bvh)
    print(f"Loaded BVH: {input_bvh}")
    print(f"  Joints: {len(bvh_names)}, Frames: {anim.shape[0]}, Frametime: {frametime}")

    # Detect object_type if not provided
    if object_type is None:
        object_type = _auto_detect_object_type_from_filename(input_bvh, cond)
        if object_type is None:
            print(f"WARNING: Could not detect object_type from filename '{os.path.basename(input_bvh)}'")
            print(f"  Available types: {list(cond.keys())}")
            print(f"  Use --object_type to specify explicitly.")
            sys.exit(1)
        print(f"  Auto-detected object_type: {object_type}")
    else:
        if object_type not in cond:
            print(f"ERROR: object_type '{object_type}' not found in cond.npy")
            print(f"  Available types: {list(cond.keys())}")
            sys.exit(1)

    # Get cond data for this object type
    obj_cond = cond[object_type]
    canonical_names = list(obj_cond.get('canonical_bvh_joint_names', []))
    original_names = list(obj_cond.get('joints_names', []))

    if len(canonical_names) == 0:
        print(f"ERROR: No canonical_bvh_joint_names found for '{object_type}'")
        sys.exit(1)
    if len(original_names) == 0:
        print(f"ERROR: No joints_names found for '{object_type}'")
        sys.exit(1)

    # Validate the mapping
    if not _validate_mapping(bvh_names, canonical_names):
        print(f"  WARNING: Mapping validation failed — proceeding anyway, but output may be incorrect.")

    # Build the new names list by mapping positions
    new_names = []
    mapped = 0
    for i in range(len(bvh_names)):
        if i < len(original_names):
            new_names.append(original_names[i])
            mapped += 1
        else:
            new_names.append(bvh_names[i])
    print(f"  Restored names: {mapped}/{len(bvh_names)} joints mapped to original names")

    # --- Scale restoration ---
    sf_value, sf_source = _get_scale_factor(obj_cond, raw_bvh_path=raw_bvh,
                                            scale_factor_arg=scale_factor)
    if sf_value is not None:
        print(f"  Scale factor: {sf_value:.6f}")
        # Unscale positions and offsets
        anim.positions = anim.positions / sf_value
        anim.offsets = anim.offsets / sf_value
        print(f"  Unscaled positions and offsets by {sf_value:.6f}")
    else:
        print(f"  WARNING: scale_factor {sf_source}. Scaling NOT restored.")

    # --- Orientation (face direction) restoration ---
    ori_quat = obj_cond.get('orientation_quat')
    ori_restored = False
    if ori_quat is not None:
        ori_q = Quaternions(np.array(ori_quat, dtype=np.float64))
        # Preprocessing did: new_rots[:,0] = qs_rot * rots[:,0]  and  new_pos[:,0] = qs_rot * pos[:,0]
        # Reverse: original = conjugate(qs_rot) * processed
        # Conjugate of (w,x,y,z) is (w,-x,-y,-z) which is -qs_rot in this library
        conj = -ori_q
        # Quaternions.__mul__ uses _broadcast internally, so (1,4) * (F,4) works without explicit repeat
        anim.rotations[:, 0] = conj * anim.rotations[:, 0]
        # anim.positions is numpy (F,J,3); quat * (F,3) vector -> (F,3) vector
        anim.positions[:, 0] = conj * anim.positions[:, 0]
        print(f"  Restored face orientation.")
        ori_restored = True
    else:
        print(f"  WARNING: orientation_quat not found in cond.npy. Orientation NOT restored.")

    # Save the restored BVH
    bvh_save(output_bvh, anim, new_names, frametime=frametime,
             positions=True, all_joints_as_names=True)
    print(f"Saved restored BVH: {output_bvh}")

    # Verification
    check_anim, check_names, check_ft = bvh_load(output_bvh)
    print(f"  Verification: {len(check_names)} joints, root='{check_names[0]}'")
    print(f"  Name match with cond.npy: {check_names == original_names}")
    if ori_restored:
        print(f"  Orientation restored: yes")

    return output_bvh


def batch_restore(cond_npy, input_dir, output_dir, raw_dir=None):
    """Restore all BVH files in a directory."""
    cond = np.load(cond_npy, allow_pickle=True).item()
    os.makedirs(output_dir, exist_ok=True)

    bvh_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.bvh')])
    if not bvh_files:
        print(f"No .bvh files found in {input_dir}")
        return

    for bvh_file in bvh_files:
        input_path = os.path.join(input_dir, bvh_file)
        output_path = os.path.join(output_dir, bvh_file)

        obj_type = _auto_detect_object_type_from_filename(input_path, cond)

        print(f"\n[{bvh_file}]")
        if obj_type is None:
            print(f"  SKIP: Cannot detect object_type from filename")
            continue

        # Find raw BVH for this object if raw_dir provided
        raw_bvh_path = None
        if raw_dir is not None:
            obj_raw_dir = os.path.join(raw_dir, obj_type)
            if os.path.isdir(obj_raw_dir):
                raw_bvhs = [f for f in os.listdir(obj_raw_dir) if f.endswith('.bvh')]
                if raw_bvhs:
                    raw_bvh_path = os.path.join(obj_raw_dir, raw_bvhs[0])

        try:
            restore_bvh(input_path, cond, output_path, object_type=obj_type,
                        raw_bvh=raw_bvh_path)
        except Exception as e:
            print(f"  ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Restore original BVH bone names and scale from Anytop processed/generated BVHs'
    )

    parser.add_argument('--cond_npy', type=str, default=None,
                        help=f'Path to cond.npy file (default: {_DEFAULT_COND_NPY})')

    # Single file mode
    parser.add_argument('--input_bvh', type=str, default=None,
                        help='Input BVH file path')
    parser.add_argument('--output_bvh', type=str, default=None,
                        help='Output BVH file path')
    parser.add_argument('--object_type', type=str, default=None,
                        help='Object type key in cond.npy (e.g., "Hound"). '
                             'If omitted, auto-detected from root joint name.')

    # Batch mode
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Input directory containing BVH files (batch mode)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for restored BVH files (batch mode)')

    # Scale restoration
    parser.add_argument('--raw_bvh', type=str, default=None,
                        help='Raw (unprocessed) BVH file to compute scale_factor. '
                             'Required if cond.npy lacks scale_factor key.')
    parser.add_argument('--raw_dir', type=str, default=None,
                        help='Root directory containing raw BVH folders (batch mode). '
                             'E.g. Anytop/dataset/truebones/zoo/Truebone_Z-OO/')
    parser.add_argument('--scale_factor', type=float, default=None,
                        help='Explicit scale factor override.')

    args = parser.parse_args()
    
    # Apply default cond_npy path
    cond_npy_path = args.cond_npy or _DEFAULT_COND_NPY
    if not os.path.isfile(cond_npy_path):
        parser.error(f"cond.npy not found at {cond_npy_path}. Use --cond_npy to specify a custom path.")

    if args.input_bvh and args.input_dir:
        parser.error("Provide either --input_bvh or --input_dir, not both")
    if not args.input_bvh and not args.input_dir:
        parser.error("Provide either --input_bvh or --input_dir")
    if args.input_bvh and not args.output_bvh:
        parser.error("--output_bvh is required when using --input_bvh")
    if args.input_dir and not args.output_dir:
        parser.error("--output_dir is required when using --input_dir")

    if args.input_bvh:
        restore_bvh(args.input_bvh, cond_npy_path, args.output_bvh,
                     object_type=args.object_type,
                     raw_bvh=args.raw_bvh, scale_factor=args.scale_factor)
    else:
        batch_restore(cond_npy_path, args.input_dir, args.output_dir,
                      raw_dir=args.raw_dir)


if __name__ == '__main__':
    main()
