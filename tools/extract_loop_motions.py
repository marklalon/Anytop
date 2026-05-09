"""
Extract Loop-Filling Motion Previews

Description:
    Extracts random N loop-filling motions from the dataset and exports them as BVH files
    for manual verification. Loop-filling occurs when motion length < max_length and
    motion_metadata['is_loop'] == True, causing the dataset to tile the motion.

Usage (from Anytop directory with venv active):
    python tools/extract_loop_motions.py \\
        --num-samples 10 \\
        --output-dir ./outputs/loop_motion_previews \\
        --objects-subset all \\
        --random-seed 42

Arguments:
    --num-samples      : Number of loop motions to extract and export (default: 10).
    --output-dir       : Directory to write BVH files (default: ./outputs/loop_motion_previews).
    --objects-subset   : Object subset to filter ('all' or a named subset, default: 'all').
    --split            : Dataset split ('train', 'val', 'test', 'all', default: 'train').
    --random-seed      : Seed for reproducible sampling (default: 42).
    --max-frames       : Max motion length for dataset (default: 196).
"""

import argparse
import random
import sys
import os
from pathlib import Path
from typing import Optional

import numpy as np

# Ensure imports resolve correctly when run from the Anytop directory
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_lib import Animation, BVH, Quaternions
from data_loaders.truebones.data.dataset import Truebones
from data_loaders.truebones.offline_reference_dataset import resolve_dataset_root
from data_loaders.truebones.truebones_utils.motion_process import (
    needs_bvh_position_channels,
    reorder_animation_to_dfs,
)


def export_animation_bvh(
    save_path: str,
    anim: Animation,
    joints_names: list[str],
) -> bool:
    """Export an Animation object as BVH, preserving required non-root positions."""
    try:
        anim, joints_names = reorder_animation_to_dfs(anim, joints_names)
        BVH.save(save_path, anim, joints_names, positions=needs_bvh_position_channels(anim))
        return True
    except Exception as e:
        print(f"    [WARN] Failed to export BVH: {e}")
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract loop-filling motion previews from dataset."
    )
    parser.add_argument(
        "--num-samples",
        default=10,
        type=int,
        help="Number of loop motions to extract and export.",
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs/loop_motion_previews",
        help="Output directory for BVH files.",
    )
    parser.add_argument(
        "--objects-subset",
        default="all",
        help="Object subset to filter ('all' or a named subset).",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test", "all"],
        help="Dataset split to use.",
    )
    parser.add_argument(
        "--random-seed",
        default=42,
        type=int,
        help="Seed for reproducible sampling.",
    )
    parser.add_argument(
        "--max-frames",
        default=196,
        type=int,
        help="Max motion length for dataset.",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Use balanced sampling across object types.",
    )
    return parser.parse_args()


def collect_loop_motions(
    dataset: Truebones,
    num_samples: int,
    rng: random.Random,
) -> list[dict]:
    """Collect loop motions from dataset."""
    loop_motions = []
    dataset_len = len(dataset)
    motion_dataset = dataset.motion_dataset
    name_list = motion_dataset.name_list
    data_dict = motion_dataset.data_dict

    print(f"Scanning {dataset_len} dataset samples for loop-filling motions...")

    # Collect metadata for all samples to find loop motions
    for idx in range(dataset_len):
        actual_idx = motion_dataset.pointer + idx if not motion_dataset.balanced else idx
        if actual_idx >= len(name_list):
            continue
        name = name_list[actual_idx]

        if name not in data_dict:
            continue

        data = data_dict[name]
        motion_metadata = data.get("motion_metadata")
        if motion_metadata is None:
            continue

        # Check if this motion will be loop-filled in the dataset
        motion_length = data.get("length", 0)
        is_loop = motion_metadata.get("is_loop", False)
        will_loop_fill = is_loop and motion_length < dataset.motion_dataset.max_motion_length

        if will_loop_fill:
            loop_motions.append(
                {
                    "name": name,
                    "motion_length": motion_length,
                    "object_type": data.get("object_type"),
                    "is_loop": is_loop,
                    "motion_path": data.get("motion_path"),
                }
            )

    print(f"Found {len(loop_motions)} loop-filling motions")

    if not loop_motions:
        return []

    # Sample randomly
    if len(loop_motions) > num_samples:
        selected = sorted(rng.sample(loop_motions, num_samples), key=lambda x: x["name"])
    else:
        selected = sorted(loop_motions, key=lambda x: x["name"])

    return selected


def _apply_loop_fill_to_animation(anim: Animation, target_len: int) -> Animation:
    """Tile an Animation to target_len frames, matching dataset loop padding semantics."""
    frame_count = len(anim)
    if frame_count >= target_len:
        return anim[:target_len].copy()

    tiles = (target_len // frame_count) + 1
    rotations = np.concatenate([anim.rotations.qs] * tiles, axis=0)[:target_len]
    positions = np.concatenate([anim.positions] * tiles, axis=0)[:target_len]
    return Animation(
        Quaternions(rotations),
        positions,
        anim.orients.copy(),
        anim.offsets.copy(),
        anim.parents.copy(),
    )


def export_loop_samples(
    loop_motions: list[dict],
    dataset: Truebones,
    output_dir: Path,
) -> None:
    """Export each selected loop motion as two BVH files:
    - <stem>_raw.bvh      : processed dataset BVH with full reference frame count
    - <stem>_loopfill.bvh : loop-tiled version padded to max_motion_length frames
    """
    motion_dataset = dataset.motion_dataset
    max_motion_length = motion_dataset.max_motion_length
    processed_bvh_dir = Path(motion_dataset.opt.data_root) / "bvhs"

    output_dir.mkdir(parents=True, exist_ok=True)
    exported_count = 0

    for i, motion_info in enumerate(loop_motions):
        name = motion_info["name"]
        stem = Path(name).stem
        source_bvh_path = processed_bvh_dir / f"{stem}.bvh"

        # Load processed dataset BVH. This keeps the original reference frame count
        # (e.g. 40 frames for Ostrich___Run_530) instead of the F-1 motion feature length.
        try:
            anim, joints_names, _frame_time = BVH.load(str(source_bvh_path))
        except Exception as e:
            print(f"  [{i+1}/{len(loop_motions)}] {name}: Failed to load processed BVH - {e}")
            continue

        raw_frames = len(anim)

        # --- raw processed reference clip ---
        raw_path = output_dir / f"{stem}_raw.bvh"
        ok_raw = export_animation_bvh(str(raw_path), anim, joints_names)

        # --- loop-filled clip (replicates dataset._prepare_sample padding at BVH level) ---
        anim_filled = _apply_loop_fill_to_animation(anim, max_motion_length)
        filled_path = output_dir / f"{stem}_loopfill.bvh"
        ok_filled = export_animation_bvh(str(filled_path), anim_filled, joints_names)

        tags = []
        if ok_raw:
            tags.append(f"raw={raw_frames}f")
        if ok_filled:
            tags.append(f"loopfill={len(anim_filled)}f")
        if ok_raw or ok_filled:
            print(f"  [{i+1}/{len(loop_motions)}] {name}: {' | '.join(tags)}")
            exported_count += 1
        else:
            print(f"  [{i+1}/{len(loop_motions)}] {name}: Export failed")

    print(f"\nExported {exported_count}/{len(loop_motions)} loop motions (×2 BVH each) to {output_dir}")


def main() -> int:
    args = parse_args()

    # Resolve dataset root
    try:
        dataset_root = resolve_dataset_root(None)
    except Exception as e:
        print(f"[ERROR] Failed to resolve dataset root: {e}")
        return 1

    # Create dataset
    try:
        print(f"Loading dataset (split='{args.split}', objects_subset='{args.objects_subset}')...")
        dataset = Truebones(
            split=args.split,
            temporal_window=31,
            num_frames=args.max_frames,
            balanced=args.balanced,
            objects_subset=args.objects_subset,
        )
        print(f"Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Collect loop motions
    rng = random.Random(args.random_seed)
    loop_motions = collect_loop_motions(dataset, args.num_samples, rng)

    if not loop_motions:
        print("[WARN] No loop-filling motions found in dataset")
        return 1

    # Export loop motions
    output_dir = Path(args.output_dir).resolve()
    print(f"\nExporting {len(loop_motions)} loop motions...")
    export_loop_samples(loop_motions, dataset, output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
