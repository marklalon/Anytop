from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from data_loaders.get_data import get_dataset_loader
from eval.motion_quality_scorer import MotionQualityScorer
from model.motion_autoencoder import build_motion_valid_mask
from utils.fixseed import fixseed


def build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True, type=str,
                        help="Checkpoint directory or specific model checkpoint for the trained motion scorer.")
    parser.add_argument("--device", default="cuda", type=str,
                        help="Device for scoring. Falls back to CPU if CUDA is unavailable.")
    parser.add_argument("--split", default="train", choices=["train", "all", "val", "test"], type=str,
                        help="Dataset split used to fetch reference training motions.")
    parser.add_argument("--batch_size", default=8, type=int,
                        help="Number of motions to compare in one debug batch.")
    parser.add_argument("--sample_limit", default=32, type=int,
                        help="Optional dataloader sample limit for faster debugging.")
    parser.add_argument("--noise_sigma", default=0.1, type=float,
                        help="Stddev for the lightly corrupted motion variant in normalized motion space.")
    parser.add_argument("--random_sigma", default=1.0, type=float,
                        help="Stddev for the random-noise motion baseline in normalized motion space.")
    parser.add_argument("--seed", default=10, type=int, help="Random seed for reproducible debug output.")
    parser.add_argument("--objects_subset", default="", type=str,
                        help="Override objects_subset. Empty reuses the scorer training args.")
    parser.add_argument("--motion_name_keywords", default="", type=str,
                        help="Override motion_name_keywords. Empty reuses the scorer training args.")
    parser.add_argument("--num_frames", default=0, type=int,
                        help="Override num_frames. 0 reuses the scorer training args.")
    parser.add_argument("--num_workers", default=0, type=int,
                        help="Dataloader workers for the debug sample fetch.")
    parser.add_argument("--output_json", default="", type=str,
                        help="Optional path to save the debug report as JSON. Defaults to checkpoint_dir/debug_score_report.json.")
    parser.add_argument("--fail_on_unexpected_order", action="store_true",
                        help="Exit with code 1 if the average ordering train > noisy > random is violated.")
    return parser


def to_float_list(values: torch.Tensor) -> list[float]:
    return [float(item) for item in values.detach().cpu().tolist()]


def tensor_dict_to_lists(result: dict[str, Any]) -> dict[str, Any]:
    converted = {}
    for key, value in result.items():
        if torch.is_tensor(value):
            converted[key] = to_float_list(value)
        elif isinstance(value, (list, tuple, np.ndarray)):
            converted[key] = value
    return converted


def summarize_scores(result: dict[str, Any]) -> dict[str, float]:
    density_distance_values = result.get("density_distance", result.get("mahal_distance", []))
    return {
        "quality_score_mean": float(np.mean(result["quality_score"])),
        "recon_score_mean": float(np.mean(result["recon_score"])),
        "density_score_mean": float(np.mean(result["density_score"])),
        "recon_error_mean": float(np.mean(result["recon_error"])),
        "density_distance_mean": float(np.mean(density_distance_values)),
        "mahal_distance_mean": float(np.mean(density_distance_values)),
    }


def make_noisy_motion(
    motion: torch.Tensor,
    valid_mask: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    noise = torch.randn_like(motion) * sigma
    return motion + noise * valid_mask


def make_random_motion(
    motion: torch.Tensor,
    valid_mask: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    random_motion = torch.randn_like(motion) * sigma
    return random_motion * valid_mask


def build_dataloader(args, scorer: MotionQualityScorer):
    scorer_args = scorer.args
    num_frames = args.num_frames or int(scorer_args.get("num_frames", 120))
    objects_subset = args.objects_subset or scorer_args.get("objects_subset", "all")
    motion_name_keywords = args.motion_name_keywords or scorer_args.get("motion_name_keywords", "")
    return get_dataset_loader(
        batch_size=args.batch_size,
        num_frames=num_frames,
        split=args.split,
        temporal_window=int(scorer_args.get("temporal_window", 31)),
        t5_name="t5-base",
        balanced=False,
        objects_subset=objects_subset,
        num_workers=args.num_workers,
        prefetch_factor=int(scorer_args.get("prefetch_factor", 2)),
        sample_limit=args.sample_limit,
        shuffle=False,
        drop_last=False,
        use_reference_conditioning=False,
        motion_name_keywords=motion_name_keywords,
    )


def main() -> int:
    args = build_parser().parse_args()
    fixseed(args.seed)

    scorer = MotionQualityScorer(args.checkpoint_dir, device=args.device)
    loader = build_dataloader(args, scorer)
    motion, cond = next(iter(loader))

    n_joints = cond["y"]["n_joints"]
    lengths = cond["y"]["lengths"]
    valid_mask = build_motion_valid_mask(n_joints, lengths, motion.shape[1], motion.shape[-1]).to(motion.dtype)

    train_result = tensor_dict_to_lists(scorer.score(motion, n_joints, lengths))
    noisy_motion = make_noisy_motion(motion, valid_mask, args.noise_sigma)
    noisy_result = tensor_dict_to_lists(scorer.score(noisy_motion, n_joints, lengths))
    random_motion = make_random_motion(motion, valid_mask, args.random_sigma)
    random_result = tensor_dict_to_lists(scorer.score(random_motion, n_joints, lengths))

    object_types = list(cond["y"].get("object_type", [None] * motion.shape[0]))
    motion_names = list(cond["y"].get("motion_name", [None] * motion.shape[0]))

    report = {
        "checkpoint": str(scorer.checkpoint_path),
        "density_mode": scorer.density_mode,
        "settings": {
            "split": args.split,
            "batch_size": int(motion.shape[0]),
            "sample_limit": args.sample_limit,
            "noise_sigma": args.noise_sigma,
            "random_sigma": args.random_sigma,
            "seed": args.seed,
            "objects_subset": args.objects_subset or scorer.args.get("objects_subset", "all"),
            "motion_name_keywords": args.motion_name_keywords or scorer.args.get("motion_name_keywords", ""),
            "num_frames": args.num_frames or scorer.args.get("num_frames", 120),
        },
        "summary": {
            "train": summarize_scores(train_result),
            "noisy": summarize_scores(noisy_result),
            "random": summarize_scores(random_result),
        },
        "ordering": {
            "avg_train_gt_noisy": summarize_scores(train_result)["quality_score_mean"] > summarize_scores(noisy_result)["quality_score_mean"],
            "avg_noisy_gt_random": summarize_scores(noisy_result)["quality_score_mean"] > summarize_scores(random_result)["quality_score_mean"],
            "avg_train_gt_random": summarize_scores(train_result)["quality_score_mean"] > summarize_scores(random_result)["quality_score_mean"],
            "per_sample_train_gt_noisy": int(sum(a > b for a, b in zip(train_result["quality_score"], noisy_result["quality_score"]))),
            "per_sample_noisy_gt_random": int(sum(a > b for a, b in zip(noisy_result["quality_score"], random_result["quality_score"]))),
            "per_sample_train_gt_random": int(sum(a > b for a, b in zip(train_result["quality_score"], random_result["quality_score"]))),
            "num_samples": int(motion.shape[0]),
        },
        "samples": [],
    }

    for index in range(motion.shape[0]):
        report["samples"].append(
            {
                "index": index,
                "object_type": object_types[index],
                "motion_name": motion_names[index],
                "n_joints": int(n_joints[index].item()),
                "length": int(lengths[index].item()),
                "train": {key: value[index] for key, value in train_result.items()},
                "noisy": {key: value[index] for key, value in noisy_result.items()},
                "random": {key: value[index] for key, value in random_result.items()},
            }
        )

    train_mean = report["summary"]["train"]["quality_score_mean"]
    noisy_mean = report["summary"]["noisy"]["quality_score_mean"]
    random_mean = report["summary"]["random"]["quality_score_mean"]

    print(f"checkpoint: {report['checkpoint']}")
    print(f"density_mode: {report['density_mode']}")
    print(f"quality_score_mean: train={train_mean:.4f} noisy={noisy_mean:.4f} random={random_mean:.4f}")
    print(
        "ordering: train>noisy {}/{} | noisy>random {}/{} | train>random {}/{}".format(
            report["ordering"]["per_sample_train_gt_noisy"],
            report["ordering"]["num_samples"],
            report["ordering"]["per_sample_noisy_gt_random"],
            report["ordering"]["num_samples"],
            report["ordering"]["per_sample_train_gt_random"],
            report["ordering"]["num_samples"],
        )
    )

    output_json = args.output_json
    if not output_json:
        output_json = str(Path(scorer.checkpoint_dir) / "debug_score_report.json")
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)
    print(f"saved report: {output_json}")

    ordering_ok = (
        report["ordering"]["avg_train_gt_noisy"]
        and report["ordering"]["avg_noisy_gt_random"]
        and report["ordering"]["avg_train_gt_random"]
    )
    if args.fail_on_unexpected_order and not ordering_ok:
        print("unexpected scorer ordering detected")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
