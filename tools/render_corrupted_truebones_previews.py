"""
Corrupted Motion BVH Export and Preview Tool

Description:
    Exports BVH files and confidence heatmaps for stored corrupted reference motions.
    This tool exports both clean target and corrupted reference motions as BVH files
    alongside soft confidence heatmaps.

Features:
    - Loads stored corrupted reference motions and companion data (skeleton, cond dict)
    - Exports clean and corrupted motions as BVH files via inverse kinematics
    - Generates position confidence heatmaps for each motion
    - Supports filtering by object type and configurable sample count
    - Random sampling for quick preview generation

Usage:
    python render_corrupted_truebones_previews.py \\
        --dataset-dir ./data/processed_anytop \\
        --objects-subset quadropeds_clean \\
        --output-dir ./preview_bvh \\
        --sample-limit 6 \\
        --random-seed 1234

    # Use default output directory (input-dir/../corrupted_references_preview):
    python render_corrupted_truebones_previews.py \\
        --dataset-dir ./data/processed_anytop \\
        --objects-subset all

    # Export from custom input directory:
    python render_corrupted_truebones_previews.py \\
        --input-dir ./custom_corrupted_refs \\
        --output-dir ./previews \\
        --sample-limit 10

Optional Arguments:
    --output-dir: Output directory for BVH files and heatmaps (auto-generated if empty)
    --dataset-dir: Processed dataset root (auto-detected if empty)
    --input-dir: Custom directory with .reference.npy files (defaults to dataset's corrupted_references)
    --objects-subset: Object types to preview - 'all' or specific subset (default: 'all')
    --sample-limit: Number of samples to render, 0 renders all (default: 6)
    --random-seed: Seed for deterministic random sampling (default: 1234)

Output:
    - {motion_name}_clean_target.bvh: Clean target motion BVH
    - {motion_name}_corrupted_reference.bvh: Corrupted reference motion BVH
    - {motion_name}_heatmap.png: Position confidence heatmap
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import BVH
from InverseKinematics import animation_from_positions


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loaders.truebones.offline_reference_dataset import (
    infer_object_type,
    get_corrupted_reference_dir,
    list_motion_files,
    load_cond_dict,
    load_corrupted_reference_sample,
    resolve_dataset_root,
)
from data_loaders.truebones.offline_reference_dataset import get_motion_dir
from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export BVH files for stored corrupted-reference motions.")
    parser.add_argument("--output-dir", default="", help="Directory to write BVH files and heatmaps into. Defaults to <input-dir>/../corrupted_references_preview.")
    parser.add_argument("--dataset-dir", default="", help="Processed dataset root. If not specified, uses default path.")
    parser.add_argument("--input-dir", default="", help="Directory containing stored corrupted-reference *.reference.npy files. Defaults to the dataset corrupted_references directory.")
    parser.add_argument("--objects-subset", default="all", help="Object subset to preview.")
    parser.add_argument("--sample-limit", default=6, type=int, help="How many stored samples to preview. 0 renders every matching sample in the input directory.")
    parser.add_argument("--random-seed", default=1234, type=int, help="Random seed used when sampling preview motions.")
    return parser.parse_args()


def select_motion_files(args: argparse.Namespace, dataset_root: Path) -> tuple[Path, list[str]]:
    input_dir = Path(args.input_dir).resolve() if args.input_dir else get_corrupted_reference_dir(dataset_root)
    if not input_dir.exists():
        raise FileNotFoundError(f"Stored corrupted-reference directory was not found: {input_dir}")

    if args.input_dir:
        motion_files = sorted(path.name.replace(".reference.npy", ".npy") for path in input_dir.glob("*.reference.npy"))
    else:
        motion_files = list_motion_files(
            dataset_dir=dataset_root,
            objects_subset=args.objects_subset,
            sample_limit=0,
        )

    if not motion_files:
        raise FileNotFoundError(f"No stored corrupted-reference samples were found under {input_dir}")

    if args.sample_limit > 0 and len(motion_files) > args.sample_limit:
        rng = random.Random(args.random_seed)
        motion_files = sorted(rng.sample(motion_files, args.sample_limit))
    return input_dir, motion_files


def save_confidence_heatmap(confidence: np.ndarray, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    image = ax.imshow(confidence.T, aspect="auto", origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Joint")
    ax.set_title("Soft Confidence")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def load_sample_from_dir(input_dir: Path, motion_file: str, dataset_root: Path) -> dict[str, object]:
    default_dir = get_corrupted_reference_dir(dataset_root).resolve()
    if input_dir.resolve() == default_dir:
        return load_corrupted_reference_sample(motion_file=motion_file, dataset_dir=dataset_root)

    stem = Path(motion_file).stem
    reference_path = input_dir / f"{stem}.reference.npy"
    confidence_path = input_dir / f"{stem}.confidence.npy"
    metadata_path = input_dir / f"{stem}.metadata.json"
    missing = [path for path in [reference_path, confidence_path, metadata_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Stored corrupted-reference sample is incomplete under {input_dir}. Missing: {missing}")

    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return {
        "motion_file": motion_file,
        "reference_motion": np.load(reference_path).astype(np.float32),
        "soft_confidence_mask": np.load(confidence_path).astype(np.float32),
        "metadata": metadata,
    }


def export_motion_bvh(save_path: str, positions: np.ndarray, parents: list[int], offsets: np.ndarray, joints_names: list[str]) -> None:
    out_anim, _, _ = animation_from_positions(positions=positions, parents=parents, offsets=offsets, iterations=150)
    if out_anim is not None:
        BVH.save(save_path, out_anim, joints_names)


def export_preview_sample(
    motion_file: str,
    dataset_root: str,
    input_dir: str,
    output_dir: str,
    parents: list[int],
    offsets: np.ndarray,
    joints_names: list[str],
) -> dict[str, object]:
    dataset_root_path = Path(dataset_root)
    input_dir_path = Path(input_dir)
    output_dir_path = Path(output_dir)
    motion_dir = get_motion_dir(dataset_root_path)

    clean_motion = np.load(motion_dir / motion_file).astype(np.float32)
    stored_sample = load_sample_from_dir(input_dir=input_dir_path, motion_file=motion_file, dataset_root=dataset_root_path)
    corrupted_motion = stored_sample["reference_motion"]
    confidence = stored_sample["soft_confidence_mask"]

    clean_positions = recover_from_bvh_ric_np(clean_motion)
    corrupted_positions = recover_from_bvh_ric_np(corrupted_motion)
    sample_dir = output_dir_path / Path(motion_file).stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    export_motion_bvh(str(sample_dir / "clean_target.bvh"), clean_positions, parents, offsets, joints_names)
    export_motion_bvh(str(sample_dir / "corrupted_reference.bvh"), corrupted_positions, parents, offsets, joints_names)
    save_confidence_heatmap(confidence[..., 0], sample_dir / "soft_confidence_mask.png")
    return {
        "motion_file": motion_file,
        "output_dir": str(sample_dir),
        "input_dir": str(input_dir_path),
    }


def main() -> int:
    args = parse_args()
    dataset_root = resolve_dataset_root(args.dataset_dir or None)
    cond_dict = load_cond_dict(dataset_root)
    input_dir, motion_files = select_motion_files(args, dataset_root)

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = input_dir.parent / "corrupted_references_preview"

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    for motion_file in motion_files:
        object_type = infer_object_type(motion_file, cond_dict.keys())
        rendered = export_preview_sample(
            motion_file=motion_file,
            dataset_root=str(dataset_root),
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            parents=[int(parent) for parent in cond_dict[object_type]["parents"]],
            offsets=cond_dict[object_type]["offsets"],
            joints_names=cond_dict[object_type]["joints_names"],
        )
        manifest.append({**rendered, "object_type": object_type})

    manifest.sort(key=lambda item: item["motion_file"])

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Exported {len(manifest)} BVH preview sets to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
