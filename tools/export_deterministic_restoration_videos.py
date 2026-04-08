"""
Export MP4 videos from deterministic restoration outputs.

Description:
    Takes the saved .npy motion arrays from deterministic_restoration_debug.py output
    and renders them as MP4 videos without re-running the expensive restoration.

Usage:
    python export_deterministic_restoration_videos.py \\
        --eval-dir ./outputs/deterministic_restoration_v4/deterministic_eval \\
        --dataset-dir ./data/processed_anytop \\
        --objects-subset quadropeds_clean \\
        --sample-limit 32 \\
        --fps 20

Required Arguments:
    --eval-dir: Path to the deterministic_eval directory from restoration output

Optional Arguments:
    --dataset-dir: Root of processed dataset (auto-detected if empty)
    --objects-subset: Object subset used during restoration (default: 'quadropeds_clean')
    --sample-limit: Number of samples to process (default: 32)
    --fps: Output video FPS (default: 20)
    --workers: Parallel render workers (default: 4)
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loaders.get_data import get_dataset_loader
from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np
from data_loaders.truebones.truebones_utils.plot_script import plot_general_skeleton_3d_motion
from data_loaders.truebones.offline_reference_dataset import resolve_dataset_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MP4 videos from deterministic restoration .npy outputs.")
    parser.add_argument("--eval-dir", required=True, help="Path to deterministic_eval directory from restoration output.")
    parser.add_argument("--dataset-dir", default="", help="Root of processed dataset. Auto-detected if empty.")
    parser.add_argument("--objects-subset", default="quadropeds_clean", help="Object subset used during restoration.")
    parser.add_argument("--sample-limit", default=32, type=int, help="Number of samples to process (should match restoration's sample-limit).")
    parser.add_argument("--fps", default=20, type=int, help="Output video FPS.")
    parser.add_argument("--workers", default=4, type=int, help="Parallel render workers.")
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


def get_parallel_video_threads(workers: int) -> int:
    """Reduce ffmpeg threads when using parallel workers."""
    return 1 if workers > 1 else 4


def render_sample_videos(
    sample_idx: int,
    sample_info: dict,
    eval_dir: Path,
    parents: list[int],
    fps: int,
    video_threads: int,
) -> dict:
    """Render both raw and restored motion videos for a single sample."""
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
        
        # Extract joint subset matching the sample
        n_joints = sample_info["n_joints"]
        length = sample_info["length"]
        motion_raw = motion_raw[:, :n_joints, :length]
        motion_restored = motion_restored[:, :n_joints, :length]
        
        # Convert to positions
        positions_raw = recover_from_bvh_ric_np(motion_raw)
        positions_restored = recover_from_bvh_ric_np(motion_restored)
        
        # Render videos
        plot_general_skeleton_3d_motion(
            str(sample_dir / "pred_xstart_raw.mp4"),
            parents,
            positions_raw,
            title="pred_xstart_raw",
            fps=fps,
            video_threads=video_threads,
        )
        plot_general_skeleton_3d_motion(
            str(sample_dir / "pred_xstart_restored.mp4"),
            parents,
            positions_restored,
            title="pred_xstart_restored",
            fps=fps,
            video_threads=video_threads,
        )
        
        return {
            "sample_index": sample_info["sample_index"],
            "object_type": sample_info["object_type"],
            "status": "rendered",
            "raw_video": str(sample_dir / "pred_xstart_raw.mp4"),
            "restored_video": str(sample_dir / "pred_xstart_restored.mp4"),
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
    
    # Load the sample list and get restoration params
    samples = load_sample_list(eval_dir)
    
    if not samples:
        print(f"Error: No samples found in {eval_dir}")
        return 1
    
    print(f"Found {len(samples)} samples to process")
    
    # Load dataset to get skeleton info (parents) for each object_type
    dataset_root = resolve_dataset_root(args.dataset_dir or None)
    
    # Create a minimal dataloader just to extract skeleton info
    print(f"Loading skeleton info from dataset...")
    
    # Load one sample per object_type to get parents
    object_type_to_parents = {}
    try:
        data_loader = get_dataset_loader(
            batch_size=1,
            num_workers=0,
            objects_subset=args.objects_subset,
            sample_limit=args.sample_limit,
            num_frames=120,
            split="train",
            shuffle=False,
            drop_last=False,
        )
        
        cond_dict = data_loader.dataset.motion_dataset.cond_dict
        
        for sample_idx, (motion, cond) in enumerate(data_loader):
            if sample_idx >= len(samples):
                break
            
            object_type = cond["y"]["object_type"][0]
            if object_type not in object_type_to_parents:
                parents = [int(p) for p in cond_dict[object_type]["parents"]]
                object_type_to_parents[object_type] = parents
                print(f"  {object_type}: {len(parents)} joints")
        
        if not object_type_to_parents:
            print("Error: Could not load skeleton information from dataset")
            return 1
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    workers = max(1, int(args.workers))
    video_threads = get_parallel_video_threads(workers)
    
    # Prepare render jobs
    jobs = []
    for sample_info in samples:
        object_type = sample_info["object_type"]
        if object_type not in object_type_to_parents:
            print(f"Warning: No skeleton info for {object_type}, skipping sample {sample_info['sample_index']}")
            continue
        
        parents = object_type_to_parents[object_type]
        jobs.append({
            "sample_info": sample_info,
            "parents": parents,
        })
    
    print(f"Rendering {len(jobs)} samples...")
    
    manifest = []
    if workers == 1:
        for job in jobs:
            result = render_sample_videos(
                sample_idx=job["sample_info"]["sample_index"],
                sample_info=job["sample_info"],
                eval_dir=eval_dir,
                parents=job["parents"],
                fps=args.fps,
                video_threads=video_threads,
            )
            manifest.append(result)
            if result["status"] == "rendered":
                print(f"✓ Sample {result['sample_index']} ({result['object_type']})")
            else:
                print(f"✗ Sample {result['sample_index']}: {result.get('error', result['status'])}")
    else:
        max_workers = min(workers, len(jobs), os.cpu_count() or workers)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    render_sample_videos,
                    sample_idx=job["sample_info"]["sample_index"],
                    sample_info=job["sample_info"],
                    eval_dir=eval_dir,
                    parents=job["parents"],
                    fps=args.fps,
                    video_threads=video_threads,
                ): job
                for job in jobs
            }
            
            for future in as_completed(futures):
                result = future.result()
                manifest.append(result)
                if result["status"] == "rendered":
                    print(f"✓ Sample {result['sample_index']} ({result['object_type']})")
                else:
                    print(f"✗ Sample {result['sample_index']}: {result.get('error', result['status'])}")
    
    # Save manifest
    manifest.sort(key=lambda x: x["sample_index"])
    manifest_path = eval_dir / "render_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    
    successful = sum(1 for m in manifest if m["status"] == "rendered")
    print(f"\nRendered {successful}/{len(manifest)} samples")
    print(f"Manifest saved to {manifest_path}")
    
    return 0 if successful == len(manifest) else 1


if __name__ == "__main__":
    raise SystemExit(main())
