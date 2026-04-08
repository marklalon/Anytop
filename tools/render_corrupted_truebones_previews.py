"""
Corrupted Motion Preview Rendering Tool

Description:
    Renders MP4 video previews and heatmaps for stored corrupted reference motions.
    This tool visualizes the corruption patterns and motion quality by generating
    skeleton animation videos alongside confidence heatmaps.

Features:
    - Loads stored corrupted reference motions and companion data (skeleton, cond dict)
    - Renders 3D skeletal animations as MP4 videos
    - Generates position confidence heatmaps for each motion
    - Supports filtering by object type and configurable sample count
    - Random sampling for quick preview generation

Usage:
    python render_corrupted_truebones_previews.py \\
        --dataset-dir ./data/processed_anytop \\
        --objects-subset quadropeds_clean \\
        --output-dir ./preview_videos \\
        --sample-limit 6 \\
        --random-seed 1234

    # Use default output directory (input-dir/../preview_animations):
    python render_corrupted_truebones_previews.py \\
        --dataset-dir ./data/processed_anytop \\
        --objects-subset all

    # Render from custom input directory:
    python render_corrupted_truebones_previews.py \\
        --input-dir ./custom_corrupted_refs \\
        --output-dir ./previews \\
        --sample-limit 10

Optional Arguments:
    --output-dir: Output directory for videos and heatmaps (auto-generated if empty)
    --dataset-dir: Processed dataset root (auto-detected if empty)
    --input-dir: Custom directory with .reference.npy files (defaults to dataset's corrupted_references)
    --objects-subset: Object types to preview - 'all' or specific subset (default: 'all')
    --sample-limit: Number of samples to render, 0 renders all (default: 6)
    --random-seed: Seed for deterministic random sampling (default: 1234)

Output:
    - {motion_name}.mp4: Skeletal animation video
    - {motion_name}_heatmap.png: Position confidence heatmap
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
from data_loaders.truebones.truebones_utils.plot_script import plot_general_skeleton_3d_motion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render MP4 previews for stored corrupted-reference motions.")
    parser.add_argument("--output-dir", default="", help="Directory to write preview videos and heatmaps into. Defaults to <input-dir>/../preview_animations.")
    parser.add_argument("--dataset-dir", default="", help="Processed dataset root. Defaults to ANYTOP_DATASET_DIR / built-in dataset path.")
    parser.add_argument("--input-dir", default="", help="Directory containing stored corrupted-reference *.reference.npy files. Defaults to the dataset corrupted_references directory.")
    parser.add_argument("--objects-subset", default="all", help="Object subset to preview.")
    parser.add_argument("--sample-limit", default=6, type=int, help="How many stored samples to preview. 0 renders every matching sample in the input directory.")
    parser.add_argument("--random-seed", default=1234, type=int, help="Random seed used when sampling preview motions.")
    parser.add_argument("--workers", default=8, type=int, help="Preview render parallelism. Values > 1 use that many worker processes and automatically reduce each ffmpeg encode to a single thread.")
    parser.add_argument("--fps", default=20, type=int, help="Output video FPS.")
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


def resolve_video_threads(workers: int) -> int:
    return 1 if workers > 1 else 4


def render_preview_sample(
    motion_file: str,
    dataset_root: str,
    input_dir: str,
    output_dir: str,
    parents: list[int],
    fps: int,
    video_threads: int,
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

    plot_general_skeleton_3d_motion(
        str(sample_dir / "clean_target.mp4"),
        parents,
        clean_positions,
        title="clean_target",
        fps=fps,
        video_threads=video_threads,
    )
    plot_general_skeleton_3d_motion(
        str(sample_dir / "corrupted_reference.mp4"),
        parents,
        corrupted_positions,
        title="corrupted_reference",
        fps=fps,
        video_threads=video_threads,
    )
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
    workers = max(1, int(args.workers))
    video_threads = resolve_video_threads(workers)
    
    # Determine output_dir: use provided value or default to input_dir parent / preview_animations
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = input_dir.parent / "preview_animations"
    
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for motion_file in motion_files:
        object_type = infer_object_type(motion_file, cond_dict.keys())
        jobs.append(
            {
                "motion_file": motion_file,
                "object_type": object_type,
                "parents": [int(parent) for parent in cond_dict[object_type]["parents"]],
            }
        )

    manifest: list[dict[str, object]] = []
    if workers == 1:
        for job in jobs:
            rendered = render_preview_sample(
                motion_file=job["motion_file"],
                dataset_root=str(dataset_root),
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                parents=job["parents"],
                fps=args.fps,
                video_threads=video_threads,
            )
            manifest.append(
                {
                    **rendered,
                    "object_type": job["object_type"],
                }
            )
    else:
        max_workers = min(workers, len(jobs), os.cpu_count() or workers)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_job = {
                executor.submit(
                    render_preview_sample,
                    motion_file=job["motion_file"],
                    dataset_root=str(dataset_root),
                    input_dir=str(input_dir),
                    output_dir=str(output_dir),
                    parents=job["parents"],
                    fps=args.fps,
                    video_threads=video_threads,
                ): job
                for job in jobs
            }
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                manifest.append(
                    {
                        **future.result(),
                        "object_type": job["object_type"],
                    }
                )

    manifest.sort(key=lambda item: item["motion_file"])

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Rendered {len(manifest)} preview sets to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())