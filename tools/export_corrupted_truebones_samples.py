import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loaders.truebones.corruption import MotionCorruptor
from data_loaders.truebones.truebones_utils.get_opt import get_opt
from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np
from data_loaders.truebones.truebones_utils.plot_script import plot_general_skeleton_3d_motion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export offline corrupted-reference samples for restoration QA.")
    parser.add_argument("--output-dir", required=True, help="Directory to write exported samples into.")
    parser.add_argument("--sample-count", default=6, type=int, help="Number of corrupted clips to export.")
    parser.add_argument("--max-frames", default=120, type=int, help="Maximum number of frames to export per clip.")
    parser.add_argument("--curriculum-stage", default=1, choices=[1, 2], type=int, help="Corruption curriculum stage preset.")
    parser.add_argument("--seed", default=1234, type=int, help="Random seed for deterministic exports.")
    return parser.parse_args()


def infer_object_type(file_name: str, object_types: list[str]) -> str:
    for object_type in sorted(object_types, key=len, reverse=True):
        if file_name.startswith(f"{object_type}_"):
            return object_type
    raise KeyError(f"Could not infer object type from motion file '{file_name}'")


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


def main() -> int:
    args = parse_args()
    opt = get_opt(device=None)
    cond_dict = np.load(opt.cond_file, allow_pickle=True).item()
    motion_dir = Path(opt.motion_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    corruptor = MotionCorruptor(
        curriculum_stage=args.curriculum_stage,
        seed=args.seed,
    )
    motion_files = sorted([file_name for file_name in os.listdir(motion_dir) if file_name.endswith(".npy")])
    if not motion_files:
        raise FileNotFoundError(f"No motion .npy files found under {motion_dir}")

    selected_files = list(rng.choice(motion_files, size=min(args.sample_count, len(motion_files)), replace=False))
    manifest: list[dict[str, object]] = []

    for sample_index, file_name in enumerate(selected_files):
        object_type = infer_object_type(file_name, list(cond_dict.keys()))
        object_cond = cond_dict[object_type]
        mean = np.asarray(object_cond["mean"], dtype=np.float32)
        std = np.asarray(object_cond["std"], dtype=np.float32) + 1e-6
        parents = object_cond["parents"]
        kinematic_chains = object_cond.get("kinematic_chains")

        clean_motion = np.load(motion_dir / file_name).astype(np.float32)
        clip_length = min(len(clean_motion), args.max_frames)
        if len(clean_motion) > clip_length:
            start = int(rng.integers(0, len(clean_motion) - clip_length + 1))
            clean_motion = clean_motion[start:start + clip_length]
        clean_motion_norm = np.nan_to_num((clean_motion - mean[None, :]) / std[None, :])
        corrupted_norm, confidence, metadata = corruptor.corrupt(
            clean_motion_norm,
            length=clip_length,
            kinematic_chains=kinematic_chains,
        )

        clean_motion_denorm = clean_motion_norm * std[None, :] + mean[None, :]
        corrupted_denorm = corrupted_norm * std[None, :] + mean[None, :]
        clean_positions = recover_from_bvh_ric_np(clean_motion_denorm)
        corrupted_positions = recover_from_bvh_ric_np(corrupted_denorm)

        sample_dir = output_dir / f"sample_{sample_index:02d}_{file_name.replace('.npy', '')}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        np.save(sample_dir / "clean_target.npy", clean_motion_denorm)
        np.save(sample_dir / "corrupted_reference.npy", corrupted_denorm)
        np.save(sample_dir / "soft_confidence_mask.npy", confidence)
        plot_general_skeleton_3d_motion(str(sample_dir / "clean_target.mp4"), parents, clean_positions, title="clean_target", fps=20)
        plot_general_skeleton_3d_motion(str(sample_dir / "corrupted_reference.mp4"), parents, corrupted_positions, title="corrupted_reference", fps=20)
        save_confidence_heatmap(confidence[..., 0], sample_dir / "soft_confidence_mask.png")

        sample_manifest = {
            "sample_dir": str(sample_dir),
            "motion_file": file_name,
            "object_type": object_type,
            "clip_length": int(clip_length),
            "curriculum_stage": args.curriculum_stage,
            "metadata": metadata,
        }
        with open(sample_dir / "metadata.json", "w", encoding="utf-8") as handle:
            json.dump(sample_manifest, handle, indent=2)
        manifest.append(sample_manifest)

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Exported {len(manifest)} corrupted samples to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())