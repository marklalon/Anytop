"""Compute loop unclosure error for all is_loop motions in the dataset.

Measures the per-joint difference between the first and last frame of each
loop-classified motion clip, sorted from worst to best.  Small residuals
are suppressed so only meaningful discontinuities are flagged.

``mean_pos_err`` is DIRECTLY COMPARABLE to LOOP_DETECTION_POS_THRESHOLD
(= 0.1) in animation_utils.py.  Both measure mean(||pos_last - pos_first||)
on the same root-relative global positions stored in channels 0-2 of each
*.npy feature file.  Use this script to see where the current threshold
sits in the actual data distribution.

Feature layout per joint (13 channels):
  0-2  : root-relative global position (face Z+, translation-root centred)
  3-8  : 6D continuous rotation representation
  9-11 : velocity (per-frame delta, scaled for playspeed)
  12   : binary contact

Usage:
    python tools/compute_loop_unclosure_error.py
    python tools/compute_loop_unclosure_error.py --sort-by mean_pos --top 20
    python tools/compute_loop_unclosure_error.py --object-type Buffalo --mean-threshold 0.05
    python tools/compute_loop_unclosure_error.py --all-motions  # include non-loop motions too
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_ANYTOP_DIR = _SCRIPT_DIR.parent  # Anytop/

if str(_ANYTOP_DIR) not in sys.path:
    sys.path.insert(0, str(_ANYTOP_DIR))


def load_motions(data_root: str, loop_only: bool = True) -> dict[str, dict]:
    """Return {motion_name: metadata} for all motions (or only is_loop)."""
    metadata_path = Path(data_root) / "motion_metadata.json"
    if not metadata_path.exists():
        print(f"ERROR: motion_metadata.json not found at {metadata_path}")
        sys.exit(1)

    with open(metadata_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    motions = payload.get("motions", payload)

    result = {}
    for name, meta in motions.items():
        if not isinstance(meta, dict):
            continue
        if loop_only and not meta.get("is_loop"):
            continue
        result[name] = meta

    return result


def compute_unclosure_error(motion_path: str) -> dict:
    """Compute first-vs-last frame feature-space differences.

    Returns a dict with per-joint and aggregate error metrics.
    """
    motion = np.load(motion_path).astype(np.float64)

    if motion.ndim != 3 or motion.shape[-1] < 13:
        return {"error": True, "message": f"Unexpected shape {motion.shape}"}
    if motion.shape[0] < 2:
        return {"error": True, "message": "Motion has < 2 frames"}

    first = motion[0]
    last = motion[-1]

    # Position error (channels 0-2): Euclidean distance per joint
    pos_diff = np.linalg.norm(first[:, 0:3] - last[:, 0:3], axis=1)

    # Rotation error (channels 3-8): Euclidean distance in 6D space
    rot_diff = np.linalg.norm(first[:, 3:9] - last[:, 3:9], axis=1)

    # Velocity error (channels 9-11): for reference
    vel_diff = np.linalg.norm(first[:, 9:12] - last[:, 9:12], axis=1)

    # Contact mismatch (channel 12): count of joints where binary contact flips
    contact_mismatch = (np.abs(first[:, 12] - last[:, 12]) >= 0.5).astype(np.int32)

    return {
        "n_joints": motion.shape[1],
        "n_frames": motion.shape[0],
        "max_pos_err": float(np.max(pos_diff)),
        "mean_pos_err": float(np.mean(pos_diff)),
        "max_rot_err": float(np.max(rot_diff)),
        "mean_rot_err": float(np.mean(rot_diff)),
        "total_pos_err": float(np.sum(pos_diff)),
        "max_vel_err": float(np.max(vel_diff)),
        "mean_vel_err": float(np.mean(vel_diff)),
        "worst_pos_joint": int(np.argmax(pos_diff)),
        "worst_rot_joint": int(np.argmax(rot_diff)),
        "contact_mismatches": int(np.sum(contact_mismatch)),
        "per_joint_pos_err": pos_diff.astype(np.float32).tolist(),
        "per_joint_rot_err": rot_diff.astype(np.float32).tolist(),
        "per_joint_vel_err": vel_diff.astype(np.float32).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute loop unclosure error for is_loop motions"
    )
    parser.add_argument(
        "--percentile", type=float, default=90,
        help="Only show motions with mean_pos_err >= this percentile (default: 90)"
    )
    parser.add_argument(
        "--top", type=int, default=0,
        help="Show only top N worst motions (0 = all above percentile)"
    )
    parser.add_argument(
        "--all-motions", action="store_true",
        help="Include non-loop motions (useful for calibrating the threshold)"
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Override dataset root directory"
    )
    parser.add_argument(
        "--motion-dir", type=str, default=None,
        help="Override motion directory"
    )
    parser.add_argument(
        "--object-type", type=str, default=None,
        help="Filter to a specific object type (e.g. Buffalo)"
    )
    parser.add_argument(
        "--buckets", type=str, default="0.01,0.02,0.05,0.1,0.15,0.2,0.3,0.5",
        help="Comma-separated bucket edges for mean_pos_err histogram"
    )
    args = parser.parse_args()

    # Determine paths
    from data_loaders.truebones.truebones_utils.get_opt import get_opt
    opt = get_opt(None)
    data_root = args.data_root or opt.data_root
    motion_dir = args.motion_dir or opt.motion_dir

    bucket_edges = [float(x.strip()) for x in args.buckets.split(",")]

    print(f"Data root  : {data_root}")
    print(f"Motion dir : {motion_dir}")
    print(f"Percentile : p{args.percentile}  (mean_pos comparable to LOOP_DETECTION_POS_THRESHOLD=0.1)")
    print(f"All motions: {args.all_motions}")
    print()

    # Load motions
    all_motions = load_motions(data_root, loop_only=not args.all_motions)

    # Filter by object type if specified
    if args.object_type:
        all_motions = {
            name: meta for name, meta in all_motions.items()
            if name.startswith(f"{args.object_type}_")
        }

    label = "motions" if args.all_motions else "loop-classified motions"
    print(f"Found {len(all_motions)} {label}")
    print()

    # Compute errors
    results = []
    for i, (name, meta) in enumerate(sorted(all_motions.items())):
        motion_path = Path(motion_dir) / name
        if not motion_path.exists():
            print(f"  [SKIP] {name}: file not found")
            continue

        err = compute_unclosure_error(str(motion_path))
        if "error" in err:
            print(f"  [SKIP] {name}: {err['message']}")
            continue

        err["name"] = name
        err["object_type"] = meta.get("object_type", "?")
        results.append(err)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(all_motions)}...")

    # ── Summary stats (computed before filtering) ──
    all_mean_pos = [r["mean_pos_err"] for r in results]
    all_mean_rot = [r["mean_rot_err"] for r in results]
    print(f"\n{'='*80}")
    print(f"Summary over {len(results)} {label}:")
    print(f"  mean_pos_err (≡ LOOP_DETECTION_POS_THRESHOLD metric):")
    print(f"    min={min(all_mean_pos):.6e}  p50={np.percentile(all_mean_pos, 50):.6e}")
    print(f"    p90={np.percentile(all_mean_pos, 90):.6e}  p95={np.percentile(all_mean_pos, 95):.6e}")
    print(f"    p99={np.percentile(all_mean_pos, 99):.6e}  max={max(all_mean_pos):.6e}")
    print(f"  mean_rot_err:")
    print(f"    min={min(all_mean_rot):.6e}  p50={np.percentile(all_mean_rot, 50):.6e}")
    print(f"    p90={np.percentile(all_mean_rot, 90):.6e}  p95={np.percentile(all_mean_rot, 95):.6e}")
    print(f"    p99={np.percentile(all_mean_rot, 99):.6e}  max={max(all_mean_rot):.6e}")

    # ── Bucket histogram of mean_pos_err ──
    print(f"\n  mean_pos_err distribution (buckets):")
    bucket_edges = sorted(bucket_edges)
    counts = np.histogram(all_mean_pos, bins=[-np.inf] + bucket_edges + [np.inf])[0]
    prev_label = "  -inf"
    for i, edge in enumerate(bucket_edges):
        print(f"    [{prev_label:>7s}, {edge:<7.3f}): {counts[i]:>5d}")
        prev_label = f"{edge:.3f}"
    print(f"    [{prev_label:>7s},   +inf): {counts[-1]:>5d}")

    # ── Filter to pN and print compact lines ──
    pN_val = np.percentile(all_mean_pos, args.percentile)
    filtered = [r for r in results if r["mean_pos_err"] >= pN_val]
    filtered.sort(key=lambda r: r["mean_pos_err"], reverse=True)

    if args.top > 0:
        filtered = filtered[:args.top]

    print(f"\n{'='*80}")
    print(f"p{args.percentile} threshold = {pN_val:.6e}  →  {len(filtered)} motions above")
    print(f"{'='*80}")

    if not filtered:
        print(f"(none — all within p{args.percentile})")
        return

    print(f"{'#':>4s}  {'name':<55s}  {'mean_pos':>10s}  {'frames':>6s}")
    print(f"{'─'*4}  {'─'*55}  {'─'*10}  {'─'*6}")
    for rank, r in enumerate(filtered):
        print(f"{rank + 1:4d}  {r['name']:<55s}  {r['mean_pos_err']:10.6e}  {r['n_frames']:6d}")


if __name__ == "__main__":
    main()
