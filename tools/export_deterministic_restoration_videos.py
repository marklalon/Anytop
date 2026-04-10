"""
Export BVH files from deterministic restoration outputs.

Description:
    Takes the saved .npy motion arrays from deterministic_restoration_debug.py output
    and exports them as BVH files without re-running the expensive restoration.

Usage:
    python export_deterministic_restoration_videos.py \\
        --eval-dir ./outputs/deterministic_restoration_v4/deterministic_eval \\
        --dataset-dir ./data/processed_anytop \\
        --objects-subset quadropeds_clean \\
        --sample-limit 32

Required Arguments:
    --eval-dir: Path to the deterministic_eval directory from restoration output

Optional Arguments:
    --dataset-dir: Root of processed dataset (auto-detected if empty)
    --objects-subset: Object subset used during restoration (default: 'quadropeds_clean')
    --sample-limit: Number of samples to process (default: 32)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import BVH
from InverseKinematics import animation_from_positions


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loaders.get_data import get_dataset_loader
from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np
from data_loaders.truebones.offline_reference_dataset import resolve_dataset_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export BVH files from deterministic restoration .npy outputs.")
    parser.add_argument("--eval-dir", required=True, help="Path to deterministic_eval directory from restoration output.")
    parser.add_argument("--dataset-dir", default="", help="Root of processed dataset. Auto-detected if empty.")
    parser.add_argument("--objects-subset", default="quadropeds_clean", help="Object subset used during restoration.")
    parser.add_argument("--sample-limit", default=32, type=int, help="Number of samples to process (should match restoration's sample-limit).")
    return parser.parse_args()


def load_sample_list(eval_dir: Path) -> list[dict]:
    """Load list of samples from eval_dir, ordered by sample_index."""
    samples = []
    for sample_dir in sorted(eval_dir.glob("sample_*")):
        metrics_file = sample_dir / "metrics.json"
        if not metrics_file.exists():
            continue
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        samples.append({
            "dir": sample_dir,
            "sample_index": metrics["sample_index"],
            "object_type": metrics["object_type"],
            "n_joints": metrics["n_joints"],
            "length": metrics["length"],
        })
    # Sort by sample_index to match data_loader order
    samples.sort(key=lambda x: x["sample_index"])
    return samples


def export_sample_bvh(
    sample_idx: int,
    sample_info: dict,
    parents: list[int],
    offsets: np.ndarray,
    joints_names: list[str],
) -> dict:
    """Export BVH files for both raw and restored motion for a single sample."""
    sample_dir = sample_info["dir"]
    motion_file_raw = sample_dir / "pred_xstart_raw.npy"
    motion_file_restored = sample_dir / "pred_xstart_restored.npy"

    if not motion_file_raw.exists() or not motion_file_restored.exists():
        return {
            "sample_index": sample_info["sample_index"],
            "status": "missing_files",
            "error": f"Missing .npy files in {sample_dir}",
        }

    try:
        motion_raw = np.load(motion_file_raw).astype(np.float32)
        motion_restored = np.load(motion_file_restored).astype(np.float32)

        n_joints = sample_info["n_joints"]
        length = sample_info["length"]
        motion_raw = motion_raw[:, :n_joints, :length]
        motion_restored = motion_restored[:, :n_joints, :length]

        exported = {}
        for name, motion in [("pred_xstart_raw", motion_raw), ("pred_xstart_restored", motion_restored)]:
            positions = recover_from_bvh_ric_np(motion)
            out_anim, _, _ = animation_from_positions(positions=positions, parents=parents, offsets=offsets, iterations=150)
            if out_anim is not None:
                bvh_path = str(sample_dir / f"{name}.bvh")
                BVH.save(bvh_path, out_anim, joints_names)
                exported[name] = bvh_path

        return {
            "sample_index": sample_info["sample_index"],
            "object_type": sample_info["object_type"],
            "status": "exported",
            **exported,
        }
    except Exception as e:
        return {
            "sample_index": sample_info["sample_index"],
            "status": "error",
            "error": str(e),
        }


def main() -> int:
    args = parse_args()
    eval_dir = Path(args.eval_dir).resolve()

    if not eval_dir.exists():
        print(f"Error: eval_dir not found: {eval_dir}")
        return 1

    samples = load_sample_list(eval_dir)

    if not samples:
        print(f"Error: No samples found in {eval_dir}")
        return 1

    print(f"Found {len(samples)} samples to process")

    dataset_root = resolve_dataset_root(args.dataset_dir or None)

    print(f"Loading skeleton info from dataset...")

    object_type_to_info = {}
    try:
        data_loader = get_dataset_loader(
            batch_size=1,
            num_workers=0,
            objects_subset=args.objects_subset,
            sample_limit=args.sample_limit,
            num_frames=60,
            split="train",
            shuffle=False,
            drop_last=False,
        )

        cond_dict = data_loader.dataset.motion_dataset.cond_dict

        for sample_idx, (motion, cond) in enumerate(data_loader):
            if sample_idx >= len(samples):
                break

            object_type = cond["y"]["object_type"][0]
            if object_type not in object_type_to_info:
                object_type_to_info[object_type] = {
                    "parents": [int(p) for p in cond_dict[object_type]["parents"]],
                    "offsets": cond_dict[object_type]["offsets"],
                    "joints_names": cond_dict[object_type]["joints_names"],
                }
                print(f"  {object_type}: {len(object_type_to_info[object_type]['parents'])} joints")

        if not object_type_to_info:
            print("Error: Could not load skeleton information from dataset")
            return 1

    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1

    manifest = []
    for sample_info in samples:
        object_type = sample_info["object_type"]
        if object_type not in object_type_to_info:
            print(f"Warning: No skeleton info for {object_type}, skipping sample {sample_info['sample_index']}")
            continue

        info = object_type_to_info[object_type]
        result = export_sample_bvh(
            sample_idx=sample_info["sample_index"],
            sample_info=sample_info,
            parents=info["parents"],
            offsets=info["offsets"],
            joints_names=info["joints_names"],
        )
        manifest.append(result)
        if result["status"] == "exported":
            print(f"✓ Sample {result['sample_index']} ({result['object_type']})")
        else:
            print(f"✗ Sample {result['sample_index']}: {result.get('error', result['status'])}")

    manifest.sort(key=lambda x: x["sample_index"])
    manifest_path = eval_dir / "bvh_export_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    successful = sum(1 for m in manifest if m["status"] == "exported")
    print(f"\nExported {successful}/{len(manifest)} samples")
    print(f"Manifest saved to {manifest_path}")

    return 0 if successful == len(manifest) else 1


if __name__ == "__main__":
    raise SystemExit(main())
