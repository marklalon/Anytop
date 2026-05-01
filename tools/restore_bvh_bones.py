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

The scale_factor is always computed from the raw BVH (--raw_bvh).

cond.npy is loaded from the default dataset path automatically;
use --cond_npy to override.

Usage:
    python tools/restore_bvh_bones.py --input_bvh in.bvh --output_bvh out.bvh \
        --raw_bvh raw_hound.bvh
"""

import argparse
import numpy as np
import os
import re
import sys

# Add Anytop to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANYTOP_DIR = os.path.join(SCRIPT_DIR, '..')
sys.path.insert(0, ANYTOP_DIR)

from motion_lib.BVH import load as bvh_load, save as bvh_save, channelmap, channelmap_inv
from motion_lib.Animation import offset_lengths as animation_offset_lengths
from motion_lib.AnimationStructure import children_list
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


def _get_scale_factor(raw_bvh_path):
    """Compute the scale factor from the raw BVH.

    scale_factor = HML_AVG_BONELEN / mean(offset_lengths(raw_offsets))
    Uses the same offset_lengths() formula as motion_process.py for consistency.
    Returns scale_factor (float) or None on failure.
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


def _parse_raw_bvh_hierarchy(raw_bvh_path):
    """Extract hierarchy metadata from a raw (unprocessed) BVH file.

    Returns:
        hierarchy_text: The full HIERARCHY block text (before MOTION).
        rotation_order: Euler order string, e.g. 'yxz'.
        raw_joint_names: Ordered list of joint names in the raw hierarchy (no End Sites).
        end_site_parent_names: Set of names of joints that have an End Site child.
    """
    with open(raw_bvh_path, 'r') as f:
        content = f.read()

    parts = content.split('MOTION', 1)
    hierarchy_text = parts[0]

    # Extract rotation order from the first CHANNELS line
    rotation_order = 'xyz'
    for match in re.finditer(
        r'CHANNELS\s+\d+\s+Xposition\s+Yposition\s+Zposition\s+'
        r'(Xrotation|Yrotation|Zrotation)\s+'
        r'(Xrotation|Yrotation|Zrotation)\s+'
        r'(Xrotation|Yrotation|Zrotation)',
        hierarchy_text
    ):
        r1, r2, r3 = match.groups()
        print_order = channelmap[r1] + channelmap[r2] + channelmap[r3]
        rotation_order = print_order[::-1]
        break

    # Parse hierarchy to identify End Site parents and joint order
    raw_joint_names = []
    end_site_parent_names = set()
    stack = []  # names stack for tracking parent during End Site
    current_name = None

    for line in hierarchy_text.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue

        # ROOT or JOINT declaration
        m = re.match(r'(?:ROOT|JOINT)\s+(\S+)', stripped)
        if m:
            name = m.group(1)
            current_name = name
            raw_joint_names.append(name)
            continue

        if '{' in stripped:
            if current_name is not None:
                stack.append(current_name)
            else:
                stack.append(None)
            continue

        if '}' in stripped:
            if stack:
                stack.pop()
            continue

        if 'End Site' in stripped:
            if stack:
                end_site_parent_names.add(stack[-1])

    return hierarchy_text, rotation_order, raw_joint_names, end_site_parent_names


def _save_bvh_with_raw_hierarchy(
    output_path,
    anim,
    joint_names,
    frametime,
    hierarchy_text,
    rotation_order,
    raw_joint_names,
    end_site_parent_names,
):
    """Save a BVH file using a raw BVH hierarchy as template.

    Writes the HIERARCHY block verbatim from the raw BVH, then writes
    MOTION data converted from ``anim`` using the correct rotation order.

    For non-root joints, position channels are filled from anim.offsets
    (which have been unscaled to original size) because the processed BVH
    only has rotation channels for non-root joints (positions are zero),
    but the raw BVH carries the static OFFSET values as per-joint position
    channels.
    """
    # Build ordered list of motion-joint indices (skip End Sites in raw hierarchy)
    motion_joint_order = []
    for raw_name in raw_joint_names:
        if raw_name in joint_names:
            idx = joint_names.index(raw_name)
        else:
            idx = -1
        motion_joint_order.append(idx)

    print_order = rotation_order[::-1]

    # Convert quaternion rotations to Euler degrees in the correct order.
    # euler(order=rotation_order) returns an (F, J, 3) array where the last
    # axis is indexed according to rotation_order (e.g. 'yxz' -> axis 0=y, 1=x, 2=z).
    # Build an axis→index map from the actual euler order so we pick the correct
    # column for each print-order axis.
    rots = np.degrees(anim.rotations.euler(order=rotation_order))
    euler_order_map = {c: i for i, c in enumerate(rotation_order)}
    p0, p1, p2 = euler_order_map[print_order[0]], euler_order_map[print_order[1]], euler_order_map[print_order[2]]
    poss = anim.positions
    offsets = anim.offsets  # unscaled bone offsets

    n_frames = anim.shape[0]

    with open(output_path, 'w') as f:
        f.write(hierarchy_text)
        if not hierarchy_text.rstrip().endswith('MOTION'):
            f.write('MOTION\n')
        f.write(f'Frames: {n_frames}\n')
        f.write(f'Frame Time: {frametime}\n')

        all_vals = np.empty((n_frames, 0), dtype=np.float64)

        for raw_idx, raw_name in enumerate(raw_joint_names):
            proc_idx = motion_joint_order[raw_idx]
            if proc_idx < 0:
                continue

            # Non-root joints in processed BVH only have rotation channels;
            # their positions are zero. The raw BVH carries the static OFFSET
            # values as per-joint position channels, so fill from offsets.
            if raw_idx == 0:
                # Root: use actual position data (includes translation).
                px, py, pz = poss[:, proc_idx, 0], poss[:, proc_idx, 1], poss[:, proc_idx, 2]
            else:
                # Non-root: fill constant offset values.
                px = np.full(n_frames, float(offsets[proc_idx, 0]))
                py = np.full(n_frames, float(offsets[proc_idx, 1]))
                pz = np.full(n_frames, float(offsets[proc_idx, 2]))

            # Write 6 channels: pos + rot in print_order
            cols = np.column_stack([
                px, py, pz,
                rots[:, proc_idx, p0], rots[:, proc_idx, p1], rots[:, proc_idx, p2]
            ])
            all_vals = np.hstack([all_vals, cols])

        np.savetxt(f, all_vals, fmt="%f", delimiter=" ")

    print(f"  Rotation order preserved: {rotation_order} (printed as {print_order})")
    print(f"  End Site blocks preserved: {len(end_site_parent_names)} joints")
    print(f"  Non-root positions filled from offsets (const per joint)")


def restore_bvh(input_bvh, cond_npy, output_bvh, object_type=None,
                raw_bvh=None):
    """
    Restore original bone names and scale in a processed/generated BVH.

    Args:
        input_bvh: Path to the input BVH file (processed or model-generated)
        cond_npy: Path to cond.npy file (str) or already-loaded dict
        output_bvh: Path to write the restored BVH
        object_type: Object type key in cond.npy (e.g., "Hound", "Horse").
                     If None, auto-detect from root joint name.
        raw_bvh: Raw BVH path for computing scale_factor and preserving hierarchy.
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

    # --- Scale restoration (always computed from raw BVH) ---
    sf_value = _get_scale_factor(raw_bvh)
    if sf_value is not None:
        print(f"  Scale factor: {sf_value:.6f}")
        anim.positions = anim.positions / sf_value
        anim.offsets = anim.offsets / sf_value
        print(f"  Unscaled positions and offsets by {sf_value:.6f}")
    else:
        print(f"  WARNING: scale_factor computation failed. Scaling NOT restored.")

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

    # --- Use raw BVH hierarchy as template ---
    hierarchy_text, rotation_order, raw_joint_names, end_site_parent_names = \
        _parse_raw_bvh_hierarchy(raw_bvh)
    print(f"  Extracted raw hierarchy: {len(raw_joint_names)} named joints, "
          f"{len(end_site_parent_names)} End Site parents, order={rotation_order}")

    # Save the restored BVH
    _save_bvh_with_raw_hierarchy(
        output_bvh, anim, new_names, frametime,
        hierarchy_text, rotation_order, raw_joint_names, end_site_parent_names,
    )
    print(f"Saved restored BVH: {output_bvh}")

    # Verification
    check_anim, check_names, check_ft = bvh_load(output_bvh)
    print(f"  Verification: {len(check_names)} joints, root='{check_names[0]}'")
    print(f"  Name match with cond.npy: {check_names == original_names}")
    if ori_restored:
        print(f"  Orientation restored: yes")

    return output_bvh


def main():
    parser = argparse.ArgumentParser(
        description='Restore original BVH bone names and scale from Anytop processed/generated BVHs'
    )

    parser.add_argument('--cond_npy', type=str, default=None,
                        help=f'Path to cond.npy file (default: {_DEFAULT_COND_NPY})')

    parser.add_argument('--input_bvh', type=str, required=True,
                        help='Input BVH file path (processed or model-generated)')
    parser.add_argument('--output_bvh', type=str, required=True,
                        help='Output BVH file path')
    parser.add_argument('--object_type', type=str, default=None,
                        help='Object type key in cond.npy (e.g., "Hound"). '
                             'If omitted, auto-detected from filename.')
    parser.add_argument('--raw_bvh', type=str, required=True,
                        help='Raw (unprocessed) BVH file. Used for computing scale_factor '
                             'and preserving the original hierarchy structure.')

    args = parser.parse_args()

    cond_npy_path = args.cond_npy or _DEFAULT_COND_NPY
    if not os.path.isfile(cond_npy_path):
        parser.error(f"cond.npy not found at {cond_npy_path}. Use --cond_npy to specify a custom path.")
    if not os.path.isfile(args.input_bvh):
        parser.error(f"Input BVH not found: {args.input_bvh}")
    if not os.path.isfile(args.raw_bvh):
        parser.error(f"Raw BVH not found: {args.raw_bvh}")

    restore_bvh(args.input_bvh, cond_npy_path, args.output_bvh,
                object_type=args.object_type, raw_bvh=args.raw_bvh)


if __name__ == '__main__':
    main()
