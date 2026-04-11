"""
Stage 1 Pretrain Sampling Debug Tool

Description:
    Samples motions from a stage1 clean-prior checkpoint on a fixed evaluation
    subset and writes a stochastic debug report with per-sample BVH export.
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import BVH
from data_loaders.get_data import get_dataset_loader
from data_loaders.truebones.truebones_utils.motion_process import recover_animation_from_motion_np, recover_from_bvh_ric_np
from utils.fixseed import fixseed
from utils import dist_util
from utils.model_util import create_model_and_diffusion_general_skeleton, load_model

QUALITY_FLOORS = {
    "mean_joint_speed": 0.01,
    "max_joint_speed": 0.05,
    "root_height_mean": 0.1,
    "root_path_length": 0.05,
    "bbox_diag": 0.5,
    "joint_acceleration_mean": 0.01,
    "joint_jerk_mean": 0.01,
    "root_jerk_mean": 0.01,
    "direction_flip_rate": 0.01,
}
CORE_STAT_KEYS = (
    "root_path_length",
    "mean_joint_speed",
    "max_joint_speed",
    "root_height_mean",
    "bbox_diag",
    "joint_acceleration_mean",
    "joint_jerk_mean",
    "root_jerk_mean",
    "direction_flip_rate",
)
STABILITY_KEYS = (
    "generated_mean_joint_speed",
    "generated_root_path_length",
    "generated_bbox_diag",
    "generated_joint_acceleration_mean",
    "generated_joint_jerk_mean",
    "generated_root_jerk_mean",
    "generated_direction_flip_rate",
    "delta_mean_joint_speed",
    "delta_root_path_length",
    "delta_bbox_diag",
    "delta_joint_acceleration_mean",
    "delta_joint_jerk_mean",
    "delta_root_jerk_mean",
    "delta_direction_flip_rate",
)


def validate_stage1_checkpoint_args(model_args: SimpleNamespace) -> None:
    violations = []
    if not bool(getattr(model_args, "disable_reference_branch", False)):
        violations.append("disable_reference_branch must be true")
    if bool(getattr(model_args, "use_reference_conditioning", False)):
        violations.append("use_reference_conditioning must be false")
    if float(getattr(model_args, "lambda_confidence_recon", 0.0)) != 0.0:
        violations.append("lambda_confidence_recon must be 0")
    if float(getattr(model_args, "lambda_repair_recon", 0.0)) != 0.0:
        violations.append("lambda_repair_recon must be 0")
    if violations:
        details = "; ".join(violations)
        raise ValueError(
            "stage1_pretrain_sampling_debug.py only supports clean-prior stage1 checkpoints. "
            f"Loaded checkpoint args are restoration-oriented: {details}."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage1 pretrain sampling debug tool with stochastic aggregate reports.")
    parser.add_argument("--model-path", required=True, help="Path to a stage1 model checkpoint.")
    parser.add_argument("--output-dir", required=True, help="Directory to write reports and exported samples.")
    parser.add_argument("--device", default=0, type=int, help="CUDA device id. Use -1 for CPU.")
    parser.add_argument("--seed", default=10, type=int, help="Global seed for deterministic setup.")
    parser.add_argument("--objects-subset", default="", help="Override the checkpoint objects_subset when set.")
    parser.add_argument("--motion-name-keywords", default="", help="Override the checkpoint motion_name_keywords when set, e.g. 'walk,run'.")
    parser.add_argument("--num-frames", default=-1, type=int, help="Override num_frames when > 0.")
    parser.add_argument("--eval-split", default="val", choices=["train", "val", "test", "all"], help="Dataset split used to choose the fixed subset.")
    parser.add_argument("--num-eval-samples", default=16, type=int, help="Number of unique samples to evaluate across all trials.")
    parser.add_argument("--eval-batch-size", default=8, type=int, help="Batch size for sampling evaluation.")
    parser.add_argument("--eval-num-workers", default=0, type=int, help="Evaluation DataLoader workers.")
    parser.add_argument("--selection-seed", default=1234, type=int, help="Seed used to select the fixed evaluation subset.")
    parser.add_argument("--num-trials", default=4, type=int, help="How many stochastic trials to run on the same selected subset.")
    parser.add_argument("--base-seed", default=10, type=int, help="Base seed for stochastic trials. Trial k uses base-seed + k.")
    parser.add_argument("--sampling-method", default="ddim", choices=["p", "ddim", "plms"], help="Diffusion sampler to use.")
    parser.add_argument("--sampling-steps", default=0, type=int, help="Respaced diffusion steps. 0 keeps the checkpoint diffusion step count.")
    parser.add_argument("--ddim-eta", default=0.0, type=float, help="DDIM eta parameter.")
    parser.add_argument("--export-samples", default=0, type=int, help="How many selected samples to export per trial. 0 disables exports.")
    parser.add_argument("--no-ema", action="store_true", help="Disable EMA model averaging and use raw model weights instead.")
    return parser.parse_args()


def load_model_args(args: argparse.Namespace) -> SimpleNamespace:
    model_path = Path(args.model_path).resolve()
    args_candidates = [
        model_path.parent / "args.json",
        model_path.parent.parent / "args.json",
    ]
    args_path = next((candidate for candidate in args_candidates if candidate.exists()), None)
    if args_path is None:
        searched = ", ".join(str(candidate) for candidate in args_candidates)
        raise FileNotFoundError(f"Arguments json was not found. Searched: {searched}")

    with open(args_path, "r", encoding="utf-8") as handle:
        model_args = SimpleNamespace(**json.load(handle))

    model_args.model_path = str(model_path)
    model_args.device = args.device
    model_args.batch_size = args.eval_batch_size
    model_args.cond_mask_prob = 0.0
    model_args.disable_reference_branch = bool(model_args.disable_reference_branch)
    model_args.use_reference_conditioning = bool(model_args.use_reference_conditioning)
    model_args.lambda_confidence_recon = float(model_args.lambda_confidence_recon)
    model_args.lambda_repair_recon = float(model_args.lambda_repair_recon)
    if args.objects_subset:
        model_args.objects_subset = args.objects_subset
    if args.motion_name_keywords:
        model_args.motion_name_keywords = args.motion_name_keywords
    if args.num_frames > 0:
        model_args.num_frames = args.num_frames
    model_args.sample_limit = 0
    model_args.num_workers = args.eval_num_workers
    validate_stage1_checkpoint_args(model_args)
    return model_args


def configure_sampling(model_args: SimpleNamespace, args: argparse.Namespace) -> None:
    diffusion_steps = int(model_args.diffusion_steps)
    sampling_steps = int(args.sampling_steps)
    if sampling_steps < 0:
        raise ValueError("--sampling-steps must be >= 0")
    if sampling_steps > diffusion_steps:
        raise ValueError(f"--sampling-steps ({sampling_steps}) cannot exceed diffusion_steps ({diffusion_steps})")
    if sampling_steps == 0:
        model_args.timestep_respacing = ""
    elif args.sampling_method == "ddim":
        model_args.timestep_respacing = f"ddim{sampling_steps}"
    else:
        model_args.timestep_respacing = str(sampling_steps)


def clone_batch_cond(cond: dict) -> dict:
    cloned = {"y": {}}
    for key, value in cond["y"].items():
        if torch.is_tensor(value):
            cloned["y"][key] = value.detach().clone()
        else:
            cloned["y"][key] = copy.deepcopy(value)
    return cloned


def move_cond_to_device(cond: dict, device: torch.device) -> dict:
    moved = {"y": {}}
    for key, value in cond["y"].items():
        moved["y"][key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def combine_batch_samples(batch_samples: list[dict[str, object]]) -> tuple[torch.Tensor, dict]:
    motion = torch.cat([sample["motion"] for sample in batch_samples], dim=0)
    cond = {"y": {}}
    keys = batch_samples[0]["cond"]["y"].keys()
    for key in keys:
        first_value = batch_samples[0]["cond"]["y"][key]
        if torch.is_tensor(first_value):
            cond["y"][key] = torch.cat([sample["cond"]["y"][key] for sample in batch_samples], dim=0)
        elif isinstance(first_value, list):
            merged = []
            for sample in batch_samples:
                merged.extend(sample["cond"]["y"][key])
            cond["y"][key] = merged
        else:
            cond["y"][key] = [sample["cond"]["y"][key] for sample in batch_samples]
    return motion, cond


def sample_motion_batch(
    diffusion,
    model,
    motion_shape: torch.Size,
    cond: dict,
    sampling_method: str,
    ddim_eta: float,
) -> torch.Tensor:
    if sampling_method == "ddim":
        return diffusion.ddim_sample_loop(
            model,
            motion_shape,
            clip_denoised=False,
            model_kwargs=cond,
            progress=False,
            eta=ddim_eta,
        )
    if sampling_method == "plms":
        return diffusion.plms_sample_loop(
            model,
            motion_shape,
            clip_denoised=False,
            model_kwargs=cond,
            progress=False,
        )
    return diffusion.p_sample_loop(
        model,
        motion_shape,
        clip_denoised=False,
        model_kwargs=cond,
        progress=False,
    )


def denormalize_motion(motion_norm: torch.Tensor, n_joints: int, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return motion_norm.permute(2, 0, 1).numpy() * std[None, :n_joints, :] + mean[None, :n_joints, :]


def _mean_vector_norm(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.linalg.norm(values, axis=-1).mean())


def _direction_flip_rate(velocities: np.ndarray, epsilon: float = 1e-6) -> float:
    if velocities.shape[0] < 2:
        return 0.0
    previous_velocity = velocities[:-1]
    next_velocity = velocities[1:]
    previous_norm = np.linalg.norm(previous_velocity, axis=-1)
    next_norm = np.linalg.norm(next_velocity, axis=-1)
    valid = np.logical_and(previous_norm > epsilon, next_norm > epsilon)
    if not np.any(valid):
        return 0.0
    cosine = np.zeros_like(previous_norm, dtype=np.float32)
    cosine[valid] = (
        np.sum(previous_velocity[valid] * next_velocity[valid], axis=-1)
        / (previous_norm[valid] * next_norm[valid])
    )
    return float((cosine[valid] < 0.0).mean())


def compute_motion_statistics(motion_denorm: np.ndarray, positions: np.ndarray) -> dict[str, float]:
    finite_mask = np.isfinite(motion_denorm)
    is_finite = bool(np.all(finite_mask))

    root_positions = positions[:, 0]
    if len(root_positions) > 1:
        root_steps = np.linalg.norm(np.diff(root_positions, axis=0), axis=-1)
        joint_steps = np.linalg.norm(np.diff(positions, axis=0), axis=-1)
    else:
        root_steps = np.zeros((0,), dtype=np.float32)
        joint_steps = np.zeros((0, positions.shape[1]), dtype=np.float32)

    if len(positions) > 2:
        joint_accelerations = np.diff(positions, n=2, axis=0)
    else:
        joint_accelerations = np.zeros((0, positions.shape[1], positions.shape[2]), dtype=np.float32)

    if len(positions) > 3:
        joint_jerks = np.diff(positions, n=3, axis=0)
    else:
        joint_jerks = np.zeros((0, positions.shape[1], positions.shape[2]), dtype=np.float32)

    root_jerks = joint_jerks[:, 0] if joint_jerks.shape[0] > 0 else np.zeros((0, positions.shape[2]), dtype=np.float32)

    pos_min = positions.min(axis=(0, 1))
    pos_max = positions.max(axis=(0, 1))
    pos_extent = pos_max - pos_min

    return {
        "is_finite": float(is_finite),
        "root_path_length": float(root_steps.sum()) if root_steps.size > 0 else 0.0,
        "mean_joint_speed": float(joint_steps.mean()) if joint_steps.size > 0 else 0.0,
        "max_joint_speed": float(joint_steps.max()) if joint_steps.size > 0 else 0.0,
        "root_height_mean": float(root_positions[:, 1].mean()) if len(root_positions) > 0 else 0.0,
        "bbox_diag": float(np.linalg.norm(pos_extent)),
        "joint_acceleration_mean": _mean_vector_norm(joint_accelerations),
        "joint_jerk_mean": _mean_vector_norm(joint_jerks),
        "root_jerk_mean": _mean_vector_norm(root_jerks),
        "direction_flip_rate": _direction_flip_rate(np.diff(positions, axis=0)) if len(positions) > 1 else 0.0,
    }


def compare_motion_statistics(target_stats: dict[str, float], generated_stats: dict[str, float]) -> dict[str, float]:
    deltas = {}
    for key in CORE_STAT_KEYS:
        deltas[f"delta_{key}"] = abs(float(generated_stats[key]) - float(target_stats[key]))
    return deltas


def evaluate_generated_prediction(
    target_norm: torch.Tensor,
    generated_norm: torch.Tensor,
    n_joints: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, object]:
    target_denorm = denormalize_motion(target_norm, n_joints, mean, std).astype(np.float32)
    generated_denorm = denormalize_motion(generated_norm, n_joints, mean, std).astype(np.float32)
    target_positions = recover_from_bvh_ric_np(target_denorm)
    generated_positions = recover_from_bvh_ric_np(generated_denorm)
    target_stats = compute_motion_statistics(target_denorm, target_positions)
    generated_stats = compute_motion_statistics(generated_denorm, generated_positions)
    delta_stats = compare_motion_statistics(target_stats, generated_stats)
    return {
        "target_denorm": target_denorm,
        "generated_denorm": generated_denorm,
        "target_positions": target_positions.astype(np.float32),
        "generated_positions": generated_positions.astype(np.float32),
        "target_stats": target_stats,
        "generated_stats": generated_stats,
        "delta_stats": delta_stats,
    }


def build_sample_metric_record(
    evaluation: dict[str, object],
    sample_index: int,
    sample: dict[str, object],
    trial_index: int,
    trial_seed: int,
) -> dict[str, object]:
    metrics = {
        "sample_index": sample_index,
        "motion_name": str(sample["motion_name"]),
        "object_type": str(sample["object_type"]),
        "length": int(sample["length"]),
        "n_joints": int(sample["n_joints"]),
        "trial_index": trial_index,
        "trial_seed": trial_seed,
    }
    metrics.update({f"target_{key}": value for key, value in evaluation["target_stats"].items()})
    metrics.update({f"generated_{key}": value for key, value in evaluation["generated_stats"].items()})
    metrics.update(evaluation["delta_stats"])
    return metrics


def summarize_sample_metrics(sample_metrics: list[dict[str, object]]) -> dict[str, float]:
    excluded_numeric_keys = {"sample_index", "trial_index", "trial_seed"}
    numeric_keys = [
        key for key, value in sample_metrics[0].items()
        if key not in excluded_numeric_keys and isinstance(value, (int, float, np.integer, np.floating))
    ]
    aggregate = {}
    for key in numeric_keys:
        values = np.array([float(sample[key]) for sample in sample_metrics], dtype=np.float64)
        aggregate[key] = float(values.mean())
    aggregate["evaluated_samples"] = len(sample_metrics)
    aggregate["invalid_samples"] = int(sum(1 for sample in sample_metrics if float(sample["generated_is_finite"]) < 0.5))
    return aggregate


def summarize_metric_dicts(metric_dicts: list[dict[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    mean_metrics = {}
    std_metrics = {}
    keys = metric_dicts[0].keys()
    for key in keys:
        values = np.array([metric[key] for metric in metric_dicts], dtype=np.float64)
        mean_metrics[key] = float(values.mean())
        std_metrics[key] = float(values.std())
    return mean_metrics, std_metrics


def _safe_ratio(value: float, baseline: float, floor: float) -> float:
    return float(value) / max(abs(float(baseline)), float(floor))


def _ratio_to_score(ratio: float) -> float:
    return float(np.exp(-max(0.0, ratio)))


def _quality_band(score: float) -> str:
    if score >= 85.0:
        return "strong"
    if score >= 70.0:
        return "usable"
    if score >= 55.0:
        return "mixed"
    if score >= 40.0:
        return "weak"
    return "poor"


def _top_warnings(core_ratios: dict[str, float], limit: int = 3) -> list[str]:
    ordered = [
        item for item in sorted(core_ratios.items(), key=lambda item: item[1], reverse=True)
        if item[1] > 1e-6
    ]
    return [key for key, _ in ordered[:limit]]


def build_core_report(
    aggregate_mean: dict[str, float],
    aggregate_std: dict[str, float],
    sample_stability: list[dict[str, object]],
) -> dict[str, object]:
    invalid_sample_rate = float(aggregate_mean["invalid_samples"]) / max(float(aggregate_mean["evaluated_samples"]), 1.0)

    core_ratios = {
        "mean_joint_speed": _safe_ratio(
            aggregate_mean["delta_mean_joint_speed"],
            aggregate_mean["target_mean_joint_speed"],
            QUALITY_FLOORS["mean_joint_speed"],
        ),
        "max_joint_speed": _safe_ratio(
            aggregate_mean["delta_max_joint_speed"],
            aggregate_mean["target_max_joint_speed"],
            QUALITY_FLOORS["max_joint_speed"],
        ),
        "root_height_mean": _safe_ratio(
            aggregate_mean["delta_root_height_mean"],
            aggregate_mean["target_root_height_mean"],
            QUALITY_FLOORS["root_height_mean"],
        ),
        "root_path_length": _safe_ratio(
            aggregate_mean["delta_root_path_length"],
            aggregate_mean["target_root_path_length"],
            QUALITY_FLOORS["root_path_length"],
        ),
        "bbox_diag": _safe_ratio(
            aggregate_mean["delta_bbox_diag"],
            aggregate_mean["target_bbox_diag"],
            QUALITY_FLOORS["bbox_diag"],
        ),
        "joint_acceleration_mean": _safe_ratio(
            aggregate_mean["delta_joint_acceleration_mean"],
            aggregate_mean["target_joint_acceleration_mean"],
            QUALITY_FLOORS["joint_acceleration_mean"],
        ),
        "joint_jerk_mean": _safe_ratio(
            aggregate_mean["delta_joint_jerk_mean"],
            aggregate_mean["target_joint_jerk_mean"],
            QUALITY_FLOORS["joint_jerk_mean"],
        ),
        "root_jerk_mean": _safe_ratio(
            aggregate_mean["delta_root_jerk_mean"],
            aggregate_mean["target_root_jerk_mean"],
            QUALITY_FLOORS["root_jerk_mean"],
        ),
        "direction_flip_rate": _safe_ratio(
            aggregate_mean["delta_direction_flip_rate"],
            aggregate_mean["target_direction_flip_rate"],
            QUALITY_FLOORS["direction_flip_rate"],
        ),
    }

    stability_ratios = {
        "mean_joint_speed": _safe_ratio(
            aggregate_std["generated_mean_joint_speed"],
            aggregate_mean["generated_mean_joint_speed"],
            QUALITY_FLOORS["mean_joint_speed"],
        ),
        "root_path_length": _safe_ratio(
            aggregate_std["generated_root_path_length"],
            aggregate_mean["generated_root_path_length"],
            QUALITY_FLOORS["root_path_length"],
        ),
        "bbox_diag": _safe_ratio(
            aggregate_std["generated_bbox_diag"],
            aggregate_mean["generated_bbox_diag"],
            QUALITY_FLOORS["bbox_diag"],
        ),
        "joint_jerk_mean": _safe_ratio(
            aggregate_std["generated_joint_jerk_mean"],
            aggregate_mean["generated_joint_jerk_mean"],
            QUALITY_FLOORS["joint_jerk_mean"],
        ),
        "direction_flip_rate": _safe_ratio(
            aggregate_std["generated_direction_flip_rate"],
            aggregate_mean["generated_direction_flip_rate"],
            QUALITY_FLOORS["direction_flip_rate"],
        ),
    }

    component_scores = {
        "validity": 100.0 * max(0.0, 1.0 - invalid_sample_rate),
        "motion_dynamics": 100.0 * float(np.mean([
            _ratio_to_score(core_ratios["mean_joint_speed"]),
            _ratio_to_score(core_ratios["max_joint_speed"]),
        ])),
        "root_consistency": 100.0 * float(np.mean([
            _ratio_to_score(core_ratios["root_height_mean"]),
            _ratio_to_score(core_ratios["root_path_length"]),
        ])),
        "motion_range": 100.0 * _ratio_to_score(core_ratios["bbox_diag"]),
        "temporal_smoothness": 100.0 * float(np.mean([
            _ratio_to_score(core_ratios["joint_acceleration_mean"]),
            _ratio_to_score(core_ratios["joint_jerk_mean"]),
            _ratio_to_score(core_ratios["root_jerk_mean"]),
            _ratio_to_score(core_ratios["direction_flip_rate"]),
        ])),
        "sampling_stability": 100.0 * float(np.mean([
            _ratio_to_score(stability_ratios["mean_joint_speed"]),
            _ratio_to_score(stability_ratios["root_path_length"]),
            _ratio_to_score(stability_ratios["bbox_diag"]),
            _ratio_to_score(stability_ratios["joint_jerk_mean"]),
            _ratio_to_score(stability_ratios["direction_flip_rate"]),
        ])),
    }

    weighted_score = (        
        + 0.25 * component_scores["motion_dynamics"]
        + 0.20 * component_scores["root_consistency"]
        + 0.10 * component_scores["motion_range"]
        + 0.25 * component_scores["temporal_smoothness"]
        + 0.20 * component_scores["sampling_stability"]
    )

    worst_samples = []
    for sample in sample_stability:
        sample_ratios = {
            "mean_joint_speed": _safe_ratio(
                sample["mean_delta_mean_joint_speed"],
                aggregate_mean["target_mean_joint_speed"],
                QUALITY_FLOORS["mean_joint_speed"],
            ),
            "root_path_length": _safe_ratio(
                sample["mean_delta_root_path_length"],
                aggregate_mean["target_root_path_length"],
                QUALITY_FLOORS["root_path_length"],
            ),
            "bbox_diag": _safe_ratio(
                sample["mean_delta_bbox_diag"],
                aggregate_mean["target_bbox_diag"],
                QUALITY_FLOORS["bbox_diag"],
            ),
            "joint_acceleration_mean": _safe_ratio(
                sample["mean_delta_joint_acceleration_mean"],
                aggregate_mean["target_joint_acceleration_mean"],
                QUALITY_FLOORS["joint_acceleration_mean"],
            ),
            "joint_jerk_mean": _safe_ratio(
                sample["mean_delta_joint_jerk_mean"],
                aggregate_mean["target_joint_jerk_mean"],
                QUALITY_FLOORS["joint_jerk_mean"],
            ),
            "root_jerk_mean": _safe_ratio(
                sample["mean_delta_root_jerk_mean"],
                aggregate_mean["target_root_jerk_mean"],
                QUALITY_FLOORS["root_jerk_mean"],
            ),
            "direction_flip_rate": _safe_ratio(
                sample["mean_delta_direction_flip_rate"],
                aggregate_mean["target_direction_flip_rate"],
                QUALITY_FLOORS["direction_flip_rate"],
            ),
            "stability_mean_joint_speed": _safe_ratio(
                sample["std_generated_mean_joint_speed"],
                sample["mean_generated_mean_joint_speed"],
                QUALITY_FLOORS["mean_joint_speed"],
            ),
            "stability_direction_flip_rate": _safe_ratio(
                sample["std_generated_direction_flip_rate"],
                sample["mean_generated_direction_flip_rate"],
                QUALITY_FLOORS["direction_flip_rate"],
            ),
        }
        sample_score = 100.0 * float(np.mean([
            _ratio_to_score(sample_ratios["mean_joint_speed"]),
            _ratio_to_score(sample_ratios["root_path_length"]),
            _ratio_to_score(sample_ratios["bbox_diag"]),
            _ratio_to_score(sample_ratios["joint_acceleration_mean"]),
            _ratio_to_score(sample_ratios["joint_jerk_mean"]),
            _ratio_to_score(sample_ratios["root_jerk_mean"]),
            _ratio_to_score(sample_ratios["direction_flip_rate"]),
            _ratio_to_score(sample_ratios["stability_mean_joint_speed"]),
            _ratio_to_score(sample_ratios["stability_direction_flip_rate"]),
        ]))
        worst_samples.append(
            {
                "motion_name": sample["motion_name"],
                "object_type": sample["object_type"],
                "sample_quality_score": sample_score,
                "mean_delta_mean_joint_speed": sample["mean_delta_mean_joint_speed"],
                "mean_delta_root_path_length": sample["mean_delta_root_path_length"],
                "mean_delta_bbox_diag": sample["mean_delta_bbox_diag"],
                "mean_delta_joint_jerk_mean": sample["mean_delta_joint_jerk_mean"],
                "mean_delta_root_jerk_mean": sample["mean_delta_root_jerk_mean"],
                "mean_delta_direction_flip_rate": sample["mean_delta_direction_flip_rate"],
                "std_generated_mean_joint_speed": sample["std_generated_mean_joint_speed"],
            }
        )
    worst_samples.sort(key=lambda item: item["sample_quality_score"])

    return {
        "overall_quality_score": weighted_score,
        "quality_band": _quality_band(weighted_score),
        "evaluated_samples": int(aggregate_mean["evaluated_samples"]),
        "invalid_samples": int(aggregate_mean["invalid_samples"]),
        "invalid_sample_rate": invalid_sample_rate,
        "core_metrics": {
            "delta_mean_joint_speed": aggregate_mean["delta_mean_joint_speed"],
            "delta_max_joint_speed": aggregate_mean["delta_max_joint_speed"],
            "delta_root_height_mean": aggregate_mean["delta_root_height_mean"],
            "delta_root_path_length": aggregate_mean["delta_root_path_length"],
            "delta_bbox_diag": aggregate_mean["delta_bbox_diag"],
            "delta_joint_acceleration_mean": aggregate_mean["delta_joint_acceleration_mean"],
            "delta_joint_jerk_mean": aggregate_mean["delta_joint_jerk_mean"],
            "delta_root_jerk_mean": aggregate_mean["delta_root_jerk_mean"],
            "delta_direction_flip_rate": aggregate_mean["delta_direction_flip_rate"],
            "stability_generated_mean_joint_speed": aggregate_std["generated_mean_joint_speed"],
            "stability_generated_root_path_length": aggregate_std["generated_root_path_length"],
            "stability_generated_bbox_diag": aggregate_std["generated_bbox_diag"],
            "stability_generated_joint_jerk_mean": aggregate_std["generated_joint_jerk_mean"],
            "stability_generated_direction_flip_rate": aggregate_std["generated_direction_flip_rate"],
        },
        "relative_error_ratios": core_ratios,
        "stability_ratios": stability_ratios,
        "component_scores": component_scores,
        "top_warnings": _top_warnings(core_ratios),
        "worst_samples": worst_samples[:5],
    }


def build_sample_stability_report(sample_trials: dict[int, list[dict[str, object]]]) -> list[dict[str, object]]:
    sample_stability = []
    for sample_index, trial_metrics in sample_trials.items():
        mean_metrics, std_metrics = summarize_metric_dicts(
            [
                {key: float(metric[key]) for key in STABILITY_KEYS}
                for metric in trial_metrics
            ]
        )
        sample_stability.append(
            {
                "sample_index": sample_index,
                "motion_name": trial_metrics[0]["motion_name"],
                "object_type": trial_metrics[0]["object_type"],
                "trial_count": len(trial_metrics),
                **{f"mean_{key}": value for key, value in mean_metrics.items()},
                **{f"std_{key}": value for key, value in std_metrics.items()},
            }
        )
    return sample_stability


def build_eval_report(
    *,
    args: argparse.Namespace,
    model_args: SimpleNamespace,
    selected_samples: list[dict[str, object]],
    trial_aggregates: list[dict[str, float]],
    sample_trials: dict[int, list[dict[str, object]]],
    num_trials: int,
    sampling_method: str,
    sampling_steps: int,
) -> dict[str, object]:
    aggregate_mean, aggregate_std = summarize_metric_dicts(trial_aggregates)
    sample_stability = build_sample_stability_report(sample_trials)
    core_report = build_core_report(aggregate_mean, aggregate_std, sample_stability)
    return {
        "split": args.eval_split,
        "objects_subset": model_args.objects_subset,
        "selected_sample_count": len(selected_samples),
        "selected_motion_names": [str(sample["motion_name"]) for sample in selected_samples],
        "selected_lengths": [int(sample["length"]) for sample in selected_samples],
        "num_trials": num_trials,
        "sampling_method": sampling_method,
        "sampling_steps": sampling_steps,
        "core_report": core_report,
    }


def build_clean_motion_baseline(
    args: argparse.Namespace,
    model_args: SimpleNamespace,
    selected_samples: list[dict[str, object]],
) -> dict[str, object]:
    baseline_metrics = []
    sample_trials = {sample_index: [] for sample_index in range(len(selected_samples))}

    for sample_index, sample in enumerate(selected_samples):
        n_joints = int(sample["n_joints"])
        length = int(sample["length"])
        target_norm = sample["motion"][0, :n_joints, :, :length]
        evaluation = evaluate_generated_prediction(
            target_norm=target_norm,
            generated_norm=target_norm,
            n_joints=n_joints,
            mean=sample["mean"],
            std=sample["std"],
        )
        metrics = build_sample_metric_record(
            evaluation=evaluation,
            sample_index=sample_index,
            sample=sample,
            trial_index=0,
            trial_seed=args.selection_seed,
        )
        baseline_metrics.append(metrics)
        sample_trials[sample_index].append(metrics)

    trial_aggregate = summarize_sample_metrics(baseline_metrics)
    sampling_steps = int(args.sampling_steps) if args.sampling_steps > 0 else int(model_args.diffusion_steps)
    baseline_report = build_eval_report(
        args=args,
        model_args=model_args,
        selected_samples=selected_samples,
        trial_aggregates=[trial_aggregate],
        sample_trials=sample_trials,
        num_trials=1,
        sampling_method="clean_identity",
        sampling_steps=sampling_steps,
    )
    baseline_report["baseline_type"] = "clean_motion_identity"
    return baseline_report


def build_baseline_comparison(
    sampled_report: dict[str, object],
    clean_report: dict[str, object],
) -> dict[str, object]:
    sampled_core = sampled_report["core_report"]
    clean_core = clean_report["core_report"]
    sampled_score = float(sampled_core["overall_quality_score"])
    clean_score = float(clean_core["overall_quality_score"])
    score_gap = clean_score - sampled_score
    return {
        "sampled_motion_score": sampled_score,
        "clean_motion_score": clean_score,
        "score_gap": score_gap,
        "objective_separation_passed": bool(score_gap >= 10.0),
        "verdict": "separates_clean_from_sampled" if score_gap >= 10.0 else "needs_better_separation",
    }


def export_trial_sample(
    sample_dir: Path,
    parents: list[int],
    offsets: np.ndarray,
    joints_names: list[str],
    target_motion: np.ndarray,
    generated_motion: np.ndarray,
) -> None:
    np.save(sample_dir / "clean_target.npy", target_motion.astype(np.float32))
    np.save(sample_dir / "generated_prediction.npy", generated_motion.astype(np.float32))
    for name, motion in [("clean_target", target_motion), ("generated_prediction", generated_motion)]:
        out_anim, has_animated_pos = recover_animation_from_motion_np(motion.astype(np.float32), parents, offsets)
        if out_anim is not None:
            BVH.save(str(sample_dir / f"{name}.bvh"), out_anim, joints_names, positions=has_animated_pos)


def collect_eval_samples(args: argparse.Namespace, model_args: SimpleNamespace) -> list[dict[str, object]]:
    fixseed(args.selection_seed)
    loader_kwargs = dict(
        batch_size=1,
        num_frames=model_args.num_frames,
        split=args.eval_split,
        temporal_window=model_args.temporal_window,
        t5_name=model_args.t5_name,
        balanced=False,
        objects_subset=model_args.objects_subset,
        num_workers=args.eval_num_workers,
        sample_limit=model_args.sample_limit,
        drop_last=False,
        use_reference_conditioning=False,
        motion_name_keywords=getattr(model_args, "motion_name_keywords", ""),
    )
    if args.eval_num_workers > 0:
        loader_kwargs["prefetch_factor"] = model_args.prefetch_factor
    data = get_dataset_loader(**loader_kwargs)
    motion_dataset = data.dataset.motion_dataset
    dataset_frame_cap = int(getattr(motion_dataset, "max_motion_length", model_args.num_frames))
    selected_frame_cap = int(getattr(motion_dataset, "max_available_length", dataset_frame_cap))
    effective_num_frames = min(int(model_args.num_frames), dataset_frame_cap, selected_frame_cap)
    model_args.dataset_max_motion_length = dataset_frame_cap
    model_args.selected_subset_max_motion_length = selected_frame_cap
    model_args.effective_num_frames = effective_num_frames
    if int(model_args.num_frames) > dataset_frame_cap:
        print(
            f"Warning: requested num_frames={int(model_args.num_frames)} exceeds dataset max_motion_length={dataset_frame_cap}. "
            f"Evaluation will use {effective_num_frames} frames."
        )
    if selected_frame_cap < effective_num_frames:
        print(
            f"Warning: selected evaluation subset only provides up to {selected_frame_cap} frames. "
            f"Evaluation will use {selected_frame_cap} frames."
        )
    motion_dataset.reset_max_len(max(20, effective_num_frames))
    cond_dict = motion_dataset.cond_dict
    samples = []
    for sample_index, (motion, cond) in enumerate(data):
        cond_cpu = clone_batch_cond(cond)
        object_type = cond_cpu["y"]["object_type"][0]
        samples.append(
            {
                "sample_index": sample_index,
                "motion": motion.detach().clone().float(),
                "cond": cond_cpu,
                "motion_name": cond_cpu["y"]["motion_name"][0],
                "object_type": object_type,
                "n_joints": int(cond_cpu["y"]["n_joints"][0].item()),
                "length": int(cond_cpu["y"]["lengths"][0].item()),
                "parents": [int(parent) for parent in cond_dict[object_type]["parents"]],
                "offsets": cond_dict[object_type]["offsets"],
                "joints_names": cond_dict[object_type]["joints_names"],
                "mean": cond_dict[object_type]["mean"].astype(np.float32),
                "std": cond_dict[object_type]["std"].astype(np.float32) + 1e-6,
            }
        )
        if len(samples) >= args.num_eval_samples:
            break

    if not samples:
        raise RuntimeError("No evaluation samples were collected.")
    return samples


def build_eval_model_and_diffusion(
    model_args: SimpleNamespace,
    checkpoint_state: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
):
    eval_model_args = copy.deepcopy(model_args)
    configure_sampling(eval_model_args, args)
    eval_model, eval_diffusion = create_model_and_diffusion_general_skeleton(eval_model_args)
    load_model(eval_model, checkpoint_state)
    eval_model.to(device)
    eval_model.eval()
    return eval_model, eval_diffusion


def stage1_sampling_eval(
    args: argparse.Namespace,
    model_args: SimpleNamespace,
    model: torch.nn.Module,
    diffusion,
    selected_samples: list[dict[str, object]],
    device: torch.device,
    output_dir: Path,
) -> dict[str, object]:
    checkpoint_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    eval_model, eval_diffusion = build_eval_model_and_diffusion(model_args, checkpoint_state, args, device)

    sample_trials = {sample_index: [] for sample_index in range(len(selected_samples))}
    trial_aggregates = []

    print(f"[PROGRESS] Starting sampling evaluation with {args.num_trials} trials and {len(selected_samples)} samples...")

    for trial_index in range(args.num_trials):
        trial_seed = args.base_seed + trial_index
        fixseed(trial_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(trial_seed)

        trial_sample_metrics = []

        total_batches = (len(selected_samples) + args.eval_batch_size - 1) // args.eval_batch_size
        for batch_start in range(0, len(selected_samples), args.eval_batch_size):
            batch_samples = selected_samples[batch_start:batch_start + args.eval_batch_size]
            motion_cpu, cond_cpu = combine_batch_samples(batch_samples)
            motion = motion_cpu.to(device, non_blocking=device.type == "cuda")
            cond = move_cond_to_device(cond_cpu, device)
            print(f"[PROGRESS] Trial {trial_index:02d} - Batch {batch_start//args.eval_batch_size + 1}/{total_batches} ...", end='\r', flush=True)

            with torch.inference_mode():
                generated = sample_motion_batch(
                    eval_diffusion,
                    eval_model,
                    motion.shape,
                    cond,
                    args.sampling_method,
                    args.ddim_eta,
                )

            for item_index, sample in enumerate(batch_samples):
                global_index = batch_start + item_index
                n_joints = int(sample["n_joints"])
                length = int(sample["length"])
                object_type = str(sample["object_type"])

                target_norm = motion_cpu[item_index, :n_joints, :, :length]
                generated_norm = generated[item_index, :n_joints, :, :length].detach().cpu()

                evaluation = evaluate_generated_prediction(
                    target_norm=target_norm,
                    generated_norm=generated_norm,
                    n_joints=n_joints,
                    mean=sample["mean"],
                    std=sample["std"],
                )
                metrics = build_sample_metric_record(evaluation, global_index, sample, trial_index, trial_seed)
                trial_sample_metrics.append(metrics)
                sample_trials[global_index].append(metrics)

                if global_index < args.export_samples:
                    sample_dir = output_dir / "stage1_sampling_eval" / "trials" / f"trial_{trial_index:02d}" / f"sample_{global_index:03d}_{object_type}"
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    export_trial_sample(
                        sample_dir=sample_dir,
                        parents=sample["parents"],
                        offsets=sample["offsets"],
                        joints_names=sample["joints_names"],
                        target_motion=evaluation["target_denorm"].astype(np.float32),
                        generated_motion=evaluation["generated_denorm"].astype(np.float32),
                    )

            print(f"\r[PROGRESS] Trial {trial_index:02d} - Batch {batch_start//args.eval_batch_size + 1}/{total_batches} ... Done")

            with torch.inference_mode():
                generated = sample_motion_batch(
                    eval_diffusion,
                    eval_model,
                    motion.shape,
                    cond,
                    args.sampling_method,
                    args.ddim_eta,
                )

            for item_index, sample in enumerate(batch_samples):
                global_index = batch_start + item_index
                n_joints = int(sample["n_joints"])
                length = int(sample["length"])
                object_type = str(sample["object_type"])

                target_norm = motion_cpu[item_index, :n_joints, :, :length]
                generated_norm = generated[item_index, :n_joints, :, :length].detach().cpu()

                evaluation = evaluate_generated_prediction(
                    target_norm=target_norm,
                    generated_norm=generated_norm,
                    n_joints=n_joints,
                    mean=sample["mean"],
                    std=sample["std"],
                )
                metrics = build_sample_metric_record(evaluation, global_index, sample, trial_index, trial_seed)
                trial_sample_metrics.append(metrics)
                sample_trials[global_index].append(metrics)

                if global_index < args.export_samples:
                    sample_dir = output_dir / "stage1_sampling_eval" / "trials" / f"trial_{trial_index:02d}" / f"sample_{global_index:03d}_{object_type}"
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    export_trial_sample(
                        sample_dir=sample_dir,
                        parents=sample["parents"],
                        offsets=sample["offsets"],
                        joints_names=sample["joints_names"],
                        target_motion=evaluation["target_denorm"].astype(np.float32),
                        generated_motion=evaluation["generated_denorm"].astype(np.float32),
                    )

        trial_aggregate = summarize_sample_metrics(trial_sample_metrics)
        trial_aggregates.append(trial_aggregate)
        print(f"[PROGRESS] Trial {trial_index:02d} complete. Aggregated metrics computed.")

    sampling_steps = int(args.sampling_steps) if args.sampling_steps > 0 else int(model_args.diffusion_steps)
    return build_eval_report(
        args=args,
        model_args=model_args,
        selected_samples=selected_samples,
        trial_aggregates=trial_aggregates,
        sample_trials=sample_trials,
        num_trials=args.num_trials,
        sampling_method=args.sampling_method,
        sampling_steps=sampling_steps,
    )


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    device = dist_util.dev()

    model_args = load_model_args(args)
    with open(output_dir / "args.json", "w", encoding="utf-8") as handle:
        json.dump(vars(model_args), handle, indent=2)

    configure_sampling(model_args, args)
    model, diffusion = create_model_and_diffusion_general_skeleton(model_args)
    state_dict = torch.load(Path(args.model_path).resolve(), map_location="cpu")
    if not args.no_ema and "model_avg" in state_dict:
        state_dict = state_dict["model_avg"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]
    load_model(model, state_dict)
    model.to(device)
    model.eval()

    print(f"[PROGRESS] Collecting {args.num_eval_samples} evaluation samples from dataset...")
    selected_samples = collect_eval_samples(args, model_args)
    print(f"[PROGRESS] Collected {len(selected_samples)} samples. Starting sampling evaluation...")
    
    sampling_report = stage1_sampling_eval(
        args=args,
        model_args=model_args,
        model=model,
        diffusion=diffusion,
        selected_samples=selected_samples,
        device=device,
        output_dir=output_dir,
    )
    print(f"[PROGRESS] Sampling evaluation complete. Building clean motion baseline...")
    clean_motion_report = build_clean_motion_baseline(args, model_args, selected_samples)
    baseline_comparison = build_baseline_comparison(sampling_report, clean_motion_report)

    final_report = {
        "model_path": str(Path(args.model_path).resolve()),
        "output_dir": str(output_dir),
        "objects_subset": model_args.objects_subset,
        "num_frames": model_args.num_frames,
        "dataset_max_motion_length": int(getattr(model_args, "dataset_max_motion_length", model_args.num_frames)),
        "selected_subset_max_motion_length": int(getattr(model_args, "selected_subset_max_motion_length", model_args.num_frames)),
        "effective_num_frames": int(getattr(model_args, "effective_num_frames", model_args.num_frames)),
        "disable_reference_branch": bool(model_args.disable_reference_branch),
        "stage1_checkpoint_validated": True,
        "stage1_semantics": {
            "use_reference_conditioning": bool(model_args.use_reference_conditioning),
            "lambda_confidence_recon": float(model_args.lambda_confidence_recon),
            "lambda_repair_recon": float(model_args.lambda_repair_recon),
            "cond_mask_prob": float(getattr(model_args, "cond_mask_prob", 0.0)),
        },
        "export_samples": int(args.export_samples),
        "stage1_sampling_eval": sampling_report,
        "clean_motion_baseline": clean_motion_report,
        "baseline_comparison": baseline_comparison,
    }
    with open(output_dir / "report.json", "w", encoding="utf-8") as handle:
        json.dump(final_report, handle, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())