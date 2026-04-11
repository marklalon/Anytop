"""
Non-Deterministic Restoration Tool

Description:
    Runs one-shot full-schedule stochastic restoration training and evaluates it
    with the standard sampler (p / ddim / plms).
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
from torch.optim import AdamW


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loaders.get_data import get_dataset_loader
import BVH
from data_loaders.truebones.truebones_utils.motion_process import recover_animation_from_motion_np
from diffusion.resample import LossAwareSampler, create_named_schedule_sampler
from utils.fixseed import fixseed
from utils import dist_util
from utils.model_util import create_model_and_diffusion_general_skeleton, load_model


RELIABLE_POSITION_TOLERANCE = 0.002
RELIABLE_POSITION_RATIO_TOLERANCE = 1.15
POSITION_IMPROVEMENT_EPS = 1e-6
DEFAULT_TEMPORAL_WINDOW = 31
DEFAULT_T5_NAME = "t5-base"
DEFAULT_LATENT_DIM = 64
DEFAULT_LAYERS = 4
DEFAULT_COND_MASK_PROB = 0.1
DEFAULT_NOISE_SCHEDULE = "cosine"
DEFAULT_SIGMA_SMALL = True
DEFAULT_DIFFUSION_STEPS = 100
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 0
DEFAULT_NUM_FRAMES = 60
DEFAULT_SAMPLE_LIMIT = 32
DEFAULT_EVAL_SAMPLES = 32

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Non-deterministic one-shot full-schedule restoration with stochastic sampling evaluation.")
    parser.add_argument("--output-dir", required=True, help="Directory to write checkpoints, reports, and exports into.")
    parser.add_argument("--model-path", default="", help="Optional checkpoint to initialize from; args.json next to it supplies model config.")
    parser.add_argument("--device", default=0, type=int, help="CUDA device id. Use -1 for CPU.")
    parser.add_argument("--seed", default=10, type=int, help="Global seed for training order and initialization.")
    parser.add_argument("--objects-subset", default="all", help="Object subset to load from the preprocessed dataset.")
    parser.add_argument("--sample-limit", default=DEFAULT_SAMPLE_LIMIT, type=int, help="Maximum number of training motions to load. 0 means all available.")
    parser.add_argument("--num-frames", default=DEFAULT_NUM_FRAMES, type=int, help="Maximum frames per motion.")
    parser.add_argument("--batch-size", default=DEFAULT_BATCH_SIZE, type=int, help="Training batch size.")
    parser.add_argument("--num-workers", default=DEFAULT_NUM_WORKERS, type=int, help="Training DataLoader workers.")
    parser.add_argument("--prefetch-factor", default=2, type=int, help="Training DataLoader prefetch factor.")
    parser.add_argument("--num-steps", default=800, type=int, help="Number of stochastic optimization steps.")
    parser.add_argument("--log-interval", default=50, type=int, help="How often to print training metrics.")
    parser.add_argument("--save-interval", default=200, type=int, help="How often to save checkpoints.")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    parser.add_argument("--weight-decay", default=DEFAULT_WEIGHT_DECAY, type=float, help="AdamW weight decay.")
    parser.add_argument("--schedule-sampler", default="uniform", help="Schedule sampler used for full-schedule timestep sampling during training.")
    parser.add_argument("--disable-train-shuffle", action="store_true", help="Disable shuffling for the training loader.")
    parser.add_argument("--lambda-confidence-recon", default=None, type=float, help="Reliable-region preservation weight. Overrides checkpoint args when set.")
    parser.add_argument("--lambda-repair-recon", default=None, type=float, help="Low-confidence repair reconstruction weight. Overrides checkpoint args when set.")
    parser.add_argument("--lambda-root", default=None, type=float, help="Root consistency weight. Overrides checkpoint args when set.")
    parser.add_argument("--lambda-velocity", default=None, type=float, help="Velocity consistency weight. Overrides checkpoint args when set.")
    parser.add_argument("--eval-split", default="train", choices=["train", "val", "test"], help="Dataset split used for post-training stochastic evaluation.")
    parser.add_argument("--num-eval-samples", default=DEFAULT_EVAL_SAMPLES, type=int, help="Number of unique samples to evaluate across all trials.")
    parser.add_argument("--eval-batch-size", default=8, type=int, help="Batch size for stochastic restoration evaluation.")
    parser.add_argument("--eval-num-workers", default=0, type=int, help="Evaluation DataLoader workers for subset collection.")
    parser.add_argument("--eval-shuffle", action="store_true", help="Shuffle the evaluation split before selecting the fixed subset.")
    parser.add_argument("--selection-seed", default=1234, type=int, help="Seed used when selecting the evaluation subset.")
    parser.add_argument("--num-trials", default=8, type=int, help="How many stochastic restoration trials to run on the same selected subset.")
    parser.add_argument("--base-seed", default=10, type=int, help="Base seed for stochastic trials. Trial k uses base-seed + k.")
    parser.add_argument("--sampling-method", default="ddim", choices=["p", "ddim", "plms"], help="Diffusion sampler to use during evaluation.")
    parser.add_argument("--sampling-steps", default=0, type=int, help="Respaced diffusion steps for evaluation. 0 keeps the training diffusion step count.")
    parser.add_argument("--ddim-eta", default=0.0, type=float, help="DDIM eta parameter. Ignored for other samplers.")
    parser.add_argument("--export-samples", default=0, type=int, help="How many selected samples to export per trial. 0 disables exports.")
    return parser.parse_args()


def load_model_args(args: argparse.Namespace) -> SimpleNamespace:
    if args.model_path:
        model_path = Path(args.model_path).resolve()
        args_path = model_path.parent / "args.json"
        if args_path.exists():
            with open(args_path, "r", encoding="utf-8") as handle:
                model_args = SimpleNamespace(**json.load(handle))
            model_args.model_path = str(model_path)
        else:
            model_args = SimpleNamespace(model_path=str(model_path))
    else:
        model_args = SimpleNamespace(
            device=args.device,
            batch_size=args.batch_size,
            num_frames=args.num_frames,
            temporal_window=DEFAULT_TEMPORAL_WINDOW,
            t5_name=DEFAULT_T5_NAME,
            objects_subset=args.objects_subset,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            sample_limit=args.sample_limit,
            latent_dim=DEFAULT_LATENT_DIM,
            layers=DEFAULT_LAYERS,
            lambda_confidence_recon=2.0 if args.lambda_confidence_recon is None else args.lambda_confidence_recon,
            lambda_repair_recon=1.0 if args.lambda_repair_recon is None else args.lambda_repair_recon,
            lambda_root=0.25 if args.lambda_root is None else args.lambda_root,
            lambda_velocity=0.1 if args.lambda_velocity is None else args.lambda_velocity,
            lambda_fs=0.0,
            lambda_geo=0.0,
            noise_schedule=DEFAULT_NOISE_SCHEDULE,
            sigma_small=DEFAULT_SIGMA_SMALL,
            cond_mask_prob=DEFAULT_COND_MASK_PROB,
            skip_t5=False,
            value_emb=False,
            diffusion_steps=DEFAULT_DIFFUSION_STEPS,
            disable_reference_branch=False,
            reference_dropout_threshold=0.05,
            timestep_respacing="",
        )

    model_args.device = args.device
    model_args.batch_size = args.batch_size
    model_args.num_frames = args.num_frames if not hasattr(model_args, "num_frames") else model_args.num_frames
    model_args.temporal_window = getattr(model_args, "temporal_window", DEFAULT_TEMPORAL_WINDOW)
    model_args.t5_name = getattr(model_args, "t5_name", DEFAULT_T5_NAME)
    model_args.objects_subset = args.objects_subset
    model_args.sample_limit = args.sample_limit
    model_args.num_workers = args.num_workers
    model_args.prefetch_factor = args.prefetch_factor if args.num_workers > 0 else getattr(model_args, "prefetch_factor", 2)
    model_args.noise_schedule = getattr(model_args, "noise_schedule", DEFAULT_NOISE_SCHEDULE)
    model_args.sigma_small = getattr(model_args, "sigma_small", DEFAULT_SIGMA_SMALL)
    model_args.lambda_fs = getattr(model_args, "lambda_fs", 0.0)
    model_args.lambda_geo = getattr(model_args, "lambda_geo", 0.0)
    model_args.lambda_confidence_recon = getattr(model_args, "lambda_confidence_recon", 2.0)
    model_args.lambda_repair_recon = getattr(model_args, "lambda_repair_recon", 1.0)
    model_args.lambda_root = getattr(model_args, "lambda_root", 0.25)
    model_args.lambda_velocity = getattr(model_args, "lambda_velocity", 0.1)
    if args.lambda_confidence_recon is not None:
        model_args.lambda_confidence_recon = args.lambda_confidence_recon
    if args.lambda_repair_recon is not None:
        model_args.lambda_repair_recon = args.lambda_repair_recon
    if args.lambda_root is not None:
        model_args.lambda_root = args.lambda_root
    if args.lambda_velocity is not None:
        model_args.lambda_velocity = args.lambda_velocity
    model_args.latent_dim = getattr(model_args, "latent_dim", DEFAULT_LATENT_DIM)
    model_args.layers = getattr(model_args, "layers", DEFAULT_LAYERS)
    model_args.cond_mask_prob = getattr(model_args, "cond_mask_prob", DEFAULT_COND_MASK_PROB)
    model_args.skip_t5 = getattr(model_args, "skip_t5", False)
    model_args.value_emb = getattr(model_args, "value_emb", False)
    model_args.diffusion_steps = getattr(model_args, "diffusion_steps", DEFAULT_DIFFUSION_STEPS)
    model_args.disable_reference_branch = getattr(model_args, "disable_reference_branch", False)
    model_args.reference_dropout_threshold = getattr(model_args, "reference_dropout_threshold", 0.05)
    model_args.timestep_respacing = getattr(model_args, "timestep_respacing", "")
    return model_args


def configure_sampling(model_args: SimpleNamespace, args: argparse.Namespace) -> None:
    diffusion_steps = int(getattr(model_args, "diffusion_steps", DEFAULT_DIFFUSION_STEPS))
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
        elif isinstance(value, list):
            cloned["y"][key] = copy.deepcopy(value)
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


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> float:
    weights = weights.expand_as(pred)
    denom = float(weights.sum().item())
    if denom <= 0.0:
        return 0.0
    return float((((pred - target) ** 2) * weights).sum().item() / denom)


def weighted_position_error(pred_positions: np.ndarray, target_positions: np.ndarray, weights: np.ndarray) -> float:
    denom = float(np.sum(weights))
    if denom <= 0.0:
        return 0.0
    joint_error = np.linalg.norm(pred_positions - target_positions, axis=-1)
    return float(np.sum(joint_error * weights) / denom)


def compute_position_metrics(
    reference_positions: np.ndarray,
    restored_positions: np.ndarray,
    target_positions: np.ndarray,
    confidence: np.ndarray,
) -> dict[str, float]:
    reference_error = np.linalg.norm(reference_positions - target_positions, axis=-1)
    restored_error = np.linalg.norm(restored_positions - target_positions, axis=-1)
    low_conf_weights = np.clip(1.0 - confidence, 0.0, 1.0)
    reliable_weights = np.clip(confidence, 0.0, 1.0)

    reference_frame_error = reference_error.mean(axis=1)
    restored_frame_error = restored_error.mean(axis=1)
    return {
        "baseline_mpjpe": float(reference_error.mean()),
        "restored_mpjpe": float(restored_error.mean()),
        "baseline_low_conf_mpjpe": weighted_position_error(reference_positions, target_positions, low_conf_weights),
        "restored_low_conf_mpjpe": weighted_position_error(restored_positions, target_positions, low_conf_weights),
        "baseline_reliable_mpjpe": weighted_position_error(reference_positions, target_positions, reliable_weights),
        "restored_reliable_mpjpe": weighted_position_error(restored_positions, target_positions, reliable_weights),
        "baseline_p95_joint_error": float(np.percentile(reference_error, 95)),
        "restored_p95_joint_error": float(np.percentile(restored_error, 95)),
        "baseline_max_joint_error": float(reference_error.max()),
        "restored_max_joint_error": float(restored_error.max()),
        "baseline_better_frame_count": int(np.sum(reference_frame_error < restored_frame_error)),
        "restored_better_frame_count": int(np.sum(restored_frame_error < reference_frame_error)),
    }


def classify_restoration(position_metrics: dict[str, float]) -> dict[str, object]:
    low_conf_improved = position_metrics["restored_low_conf_mpjpe"] + POSITION_IMPROVEMENT_EPS < position_metrics["baseline_low_conf_mpjpe"]
    reliable_limit = max(
        position_metrics["baseline_reliable_mpjpe"] + RELIABLE_POSITION_TOLERANCE,
        position_metrics["baseline_reliable_mpjpe"] * RELIABLE_POSITION_RATIO_TOLERANCE,
    )
    reliable_preserved = position_metrics["restored_reliable_mpjpe"] <= reliable_limit
    overall_improved = position_metrics["restored_mpjpe"] + POSITION_IMPROVEMENT_EPS < position_metrics["baseline_mpjpe"]
    success = bool(low_conf_improved and reliable_preserved and overall_improved)
    if success:
        verdict = "successful_repair"
    elif not low_conf_improved:
        verdict = "failed_low_conf_repair"
    elif not reliable_preserved:
        verdict = "failed_reliable_preservation"
    else:
        verdict = "failed_overall_position_error"
    return {
        "low_conf_improved": low_conf_improved,
        "reliable_preserved": reliable_preserved,
        "overall_position_improved": overall_improved,
        "restoration_success": success,
        "restoration_verdict": verdict,
    }


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


def export_trial_sample(
    sample_dir: Path,
    parents: list[int],
    offsets: np.ndarray,
    joints_names: list[str],
    target_motion: np.ndarray,
    reference_motion: np.ndarray,
    restored_motion: np.ndarray,
    confidence: np.ndarray,
) -> None:
    np.save(sample_dir / "clean_target.npy", target_motion.astype(np.float32))
    np.save(sample_dir / "corrupted_reference.npy", reference_motion.astype(np.float32))
    np.save(sample_dir / "restored_prediction.npy", restored_motion.astype(np.float32))
    np.save(sample_dir / "soft_confidence_mask.npy", confidence.astype(np.float32))
    for name, motion in [("clean_target", target_motion), ("corrupted_reference", reference_motion), ("restored_prediction", restored_motion)]:
        out_anim, has_animated_pos = recover_animation_from_motion_np(motion.astype(np.float32), parents, offsets)
        if out_anim is not None:
            BVH.save(str(sample_dir / f"{name}.bvh"), out_anim, joints_names, positions=has_animated_pos)


def compute_feature_metrics(
    target_norm: torch.Tensor,
    reference_norm: torch.Tensor,
    restored_norm: torch.Tensor,
    confidence: torch.Tensor,
) -> dict[str, float]:
    low_conf_weights = (1.0 - confidence).clamp(min=0.0, max=1.0)
    reliable_weights = confidence
    return {
        "feature_baseline_mse": float(torch.mean((reference_norm - target_norm) ** 2).item()),
        "feature_restored_mse": float(torch.mean((restored_norm - target_norm) ** 2).item()),
        "feature_baseline_low_conf_mse": weighted_mse(reference_norm, target_norm, low_conf_weights),
        "feature_restored_low_conf_mse": weighted_mse(restored_norm, target_norm, low_conf_weights),
        "feature_baseline_reliable_mse": weighted_mse(reference_norm, target_norm, reliable_weights),
        "feature_restored_reliable_mse": weighted_mse(restored_norm, target_norm, reliable_weights),
    }


def denormalize_motion(motion_norm: torch.Tensor, n_joints: int, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return motion_norm.permute(2, 0, 1).numpy() * std[None, :n_joints, :] + mean[None, :n_joints, :]


def evaluate_restoration_prediction(
    target_norm: torch.Tensor,
    reference_norm: torch.Tensor,
    restored_norm: torch.Tensor,
    confidence: torch.Tensor,
    n_joints: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, object]:
    feature_metrics = compute_feature_metrics(target_norm, reference_norm, restored_norm, confidence)
    target_denorm = denormalize_motion(target_norm, n_joints, mean, std).astype(np.float32)
    reference_denorm = denormalize_motion(reference_norm, n_joints, mean, std).astype(np.float32)
    restored_denorm = denormalize_motion(restored_norm, n_joints, mean, std).astype(np.float32)
    confidence_np = confidence.permute(2, 0, 1).numpy().astype(np.float32)[..., 0]
    target_positions = recover_from_bvh_ric_np(target_denorm)
    reference_positions = recover_from_bvh_ric_np(reference_denorm)
    restored_positions = recover_from_bvh_ric_np(restored_denorm)
    position_metrics = compute_position_metrics(
        reference_positions=reference_positions,
        restored_positions=restored_positions,
        target_positions=target_positions,
        confidence=confidence_np,
    )
    restoration_flags = classify_restoration(position_metrics)
    return {
        "position_metrics": position_metrics,
        "feature_metrics": feature_metrics,
        "restoration_flags": restoration_flags,
        "target_denorm": target_denorm,
        "reference_denorm": reference_denorm,
        "restored_denorm": restored_denorm,
        "confidence_np": confidence_np,
    }


def build_sample_metric_record(
    base_metrics: dict[str, object],
    sample_index: int,
    sample: dict[str, object],
    trial_index: int | None = None,
    trial_seed: int | None = None,
) -> dict[str, object]:
    position_metrics = base_metrics["position_metrics"]
    feature_metrics = base_metrics["feature_metrics"]
    restoration_flags = base_metrics["restoration_flags"]
    metrics = {
        "sample_index": sample_index,
        "motion_name": str(sample["motion_name"]),
        "object_type": str(sample["object_type"]),
        "length": int(sample["length"]),
        "n_joints": int(sample["n_joints"]),
        **position_metrics,
        **restoration_flags,
        "overall_position_improvement": position_metrics["baseline_mpjpe"] - position_metrics["restored_mpjpe"],
        "low_conf_position_improvement": position_metrics["baseline_low_conf_mpjpe"] - position_metrics["restored_low_conf_mpjpe"],
        "reliable_position_change": position_metrics["baseline_reliable_mpjpe"] - position_metrics["restored_reliable_mpjpe"],
        **feature_metrics,
        "feature_overall_improvement": feature_metrics["feature_baseline_mse"] - feature_metrics["feature_restored_mse"],
        "feature_low_conf_improvement": feature_metrics["feature_baseline_low_conf_mse"] - feature_metrics["feature_restored_low_conf_mse"],
        "feature_reliable_region_change": feature_metrics["feature_baseline_reliable_mse"] - feature_metrics["feature_restored_reliable_mse"],
    }
    if trial_index is not None:
        metrics["trial_index"] = trial_index
    if trial_seed is not None:
        metrics["trial_seed"] = trial_seed
    return metrics


def summarize_sample_metrics(sample_metrics: list[dict[str, object]]) -> dict[str, float]:
    totals = {
        "baseline_mpjpe": [],
        "restored_mpjpe": [],
        "baseline_low_conf_mpjpe": [],
        "restored_low_conf_mpjpe": [],
        "baseline_reliable_mpjpe": [],
        "restored_reliable_mpjpe": [],
        "feature_baseline_mse": [],
        "feature_restored_mse": [],
        "feature_baseline_low_conf_mse": [],
        "feature_restored_low_conf_mse": [],
        "feature_baseline_reliable_mse": [],
        "feature_restored_reliable_mse": [],
        "restoration_success": [],
    }
    for sample in sample_metrics:
        for key in totals:
            totals[key].append(float(sample[key]))
    aggregate = {key: float(np.mean(values)) for key, values in totals.items() if values}
    aggregate["avg_overall_position_improvement"] = aggregate["baseline_mpjpe"] - aggregate["restored_mpjpe"]
    aggregate["avg_low_conf_position_improvement"] = aggregate["baseline_low_conf_mpjpe"] - aggregate["restored_low_conf_mpjpe"]
    aggregate["avg_feature_overall_improvement"] = aggregate["feature_baseline_mse"] - aggregate["feature_restored_mse"]
    aggregate["avg_feature_low_conf_improvement"] = aggregate["feature_baseline_low_conf_mse"] - aggregate["feature_restored_low_conf_mse"]
    aggregate["successful_repairs"] = int(sum(int(sample["restoration_success"]) for sample in sample_metrics))
    aggregate["evaluated_samples"] = len(sample_metrics)
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


def create_train_loader(args: argparse.Namespace, model_args: SimpleNamespace):
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_frames=model_args.num_frames,
        split="train",
        temporal_window=model_args.temporal_window,
        t5_name=model_args.t5_name,
        balanced=False,
        objects_subset=model_args.objects_subset,
        num_workers=args.num_workers,
        sample_limit=args.sample_limit,
        shuffle=not args.disable_train_shuffle,
        drop_last=False,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    return get_dataset_loader(**loader_kwargs)


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
        shuffle=args.eval_shuffle,
        drop_last=False,
    )
    if args.eval_num_workers > 0:
        loader_kwargs["prefetch_factor"] = model_args.prefetch_factor
    data = get_dataset_loader(**loader_kwargs)
    cond_dict = data.dataset.motion_dataset.cond_dict

    samples: list[dict[str, object]] = []
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


def maybe_wrap_checkpoint_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def train_stochastic_debug(
    args: argparse.Namespace,
    model_args: SimpleNamespace,
    model: torch.nn.Module,
    diffusion,
    device: torch.device,
    output_dir: Path,
) -> list[dict[str, float]]:
    model.train()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    history: list[dict[str, float]] = []
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_data = create_train_loader(args, model_args)
    data_iter = iter(train_data)

    for step in range(args.num_steps):
        try:
            motion, cond = next(data_iter)
        except StopIteration:
            data_iter = iter(train_data)
            motion, cond = next(data_iter)

        motion = motion.to(device, non_blocking=device.type == "cuda")
        cond = move_cond_to_device(cond, device)
        timesteps, weights = schedule_sampler.sample(motion.shape[0], device)
        timestep_mode = "full-schedule"

        optimizer.zero_grad(set_to_none=True)
        losses = diffusion.training_losses(model, motion, timesteps, model_kwargs=cond)
        if isinstance(schedule_sampler, LossAwareSampler):
            schedule_sampler.update_with_local_losses(timesteps, losses["loss"].detach())
        loss = (losses["loss"] * weights).mean()
        loss.backward()
        optimizer.step()

        record = {}
        for key, value in losses.items():
            if torch.is_tensor(value):
                record[key] = float((value.detach() * weights).mean().item())
        record["loss"] = float(loss.detach().item())
        record["step"] = float(step)
        record["t_mean"] = float(timesteps.float().mean().item())
        record["t_min"] = float(timesteps.min().item())
        record["t_max"] = float(timesteps.max().item())
        record["timestep_mode"] = timestep_mode
        history.append(record)

        if step % args.log_interval == 0 or step == args.num_steps - 1:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": record.get("loss"),
                        "l_simple": record.get("l_simple"),
                        "confidence_recon_loss": record.get("confidence_recon_loss"),
                        "repair_recon_loss": record.get("repair_recon_loss"),
                        "timestep_mode": record.get("timestep_mode"),
                        "t_mean": record.get("t_mean"),
                        "t_min": record.get("t_min"),
                        "t_max": record.get("t_max"),
                    }
                )
            )

        should_save = (step + 1) % args.save_interval == 0 or step == args.num_steps - 1
        if should_save:
            torch.save(maybe_wrap_checkpoint_state(model), checkpoints_dir / f"stochastic_step_{step + 1:05d}.pt")

    return history


def build_eval_model_and_diffusion(
    train_model: torch.nn.Module,
    model_args: SimpleNamespace,
    args: argparse.Namespace,
    device: torch.device,
):
    if args.sampling_steps <= 0:
        train_model.eval()
        return train_model, None

    eval_model_args = copy.deepcopy(model_args)
    eval_model_args.batch_size = args.eval_batch_size
    configure_sampling(eval_model_args, args)
    eval_model, eval_diffusion = create_model_and_diffusion_general_skeleton(eval_model_args)
    load_model(eval_model, train_model.state_dict())
    eval_model.to(device)
    eval_model.eval()
    return eval_model, eval_diffusion


def stochastic_eval(
    args: argparse.Namespace,
    model_args: SimpleNamespace,
    train_model: torch.nn.Module,
    diffusion,
    selected_samples: list[dict[str, object]],
    device: torch.device,
    output_dir: Path,
) -> dict[str, object]:
    eval_model, eval_diffusion = build_eval_model_and_diffusion(train_model, model_args, args, device)
    effective_diffusion = diffusion if eval_diffusion is None else eval_diffusion

    trial_reports = []
    sample_trials: dict[int, list[dict[str, object]]] = {sample_index: [] for sample_index in range(len(selected_samples))}

    for trial_index in range(args.num_trials):
        trial_seed = args.base_seed + trial_index
        fixseed(trial_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(trial_seed)

        trial_sample_metrics = []

        for batch_start in range(0, len(selected_samples), args.eval_batch_size):
            batch_samples = selected_samples[batch_start:batch_start + args.eval_batch_size]
            motion_cpu, cond_cpu = combine_batch_samples(batch_samples)
            motion = motion_cpu.to(device, non_blocking=device.type == "cuda")
            cond = move_cond_to_device(cond_cpu, device)

            with torch.inference_mode():
                restored = sample_motion_batch(
                    effective_diffusion,
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
                reference_norm = sample["cond"]["y"]["reference_motion"][0, :n_joints, :, :length]
                restored_norm = restored[item_index, :n_joints, :, :length].detach().cpu()
                confidence = sample["cond"]["y"]["soft_confidence_mask"][0, :n_joints, :, :length].detach().cpu().clamp(0.0, 1.0)
                evaluation = evaluate_restoration_prediction(
                    target_norm=target_norm,
                    reference_norm=reference_norm,
                    restored_norm=restored_norm,
                    confidence=confidence,
                    n_joints=n_joints,
                    mean=sample["mean"],
                    std=sample["std"],
                )
                metrics = build_sample_metric_record(evaluation, global_index, sample, trial_index, trial_seed)
                trial_sample_metrics.append(metrics)
                sample_trials[global_index].append(metrics)

                if global_index < args.export_samples:
                    sample_dir = output_dir / "stochastic_eval" / "trials" / f"trial_{trial_index:02d}" / f"sample_{global_index:03d}_{object_type}"
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    export_trial_sample(
                        sample_dir=sample_dir,
                        parents=sample["parents"],
                        offsets=sample["offsets"],
                        joints_names=sample["joints_names"],
                        target_motion=evaluation["target_denorm"].astype(np.float32),
                        reference_motion=evaluation["reference_denorm"].astype(np.float32),
                        restored_motion=evaluation["restored_denorm"].astype(np.float32),
                        confidence=confidence.permute(2, 0, 1).numpy().astype(np.float32),
                    )
                    with open(sample_dir / "metrics.json", "w", encoding="utf-8") as handle:
                        json.dump(metrics, handle, indent=2)

        trial_aggregate = summarize_sample_metrics(trial_sample_metrics)
        trial_report = {
            "trial_index": trial_index,
            "trial_seed": trial_seed,
            "aggregate": trial_aggregate,
            "samples": trial_sample_metrics,
        }
        trial_reports.append(trial_report)
        trial_dir = output_dir / "stochastic_eval" / "trials" / f"trial_{trial_index:02d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        with open(trial_dir / "summary.json", "w", encoding="utf-8") as handle:
            json.dump(trial_report, handle, indent=2)

    aggregate_mean, aggregate_std = summarize_metric_dicts([trial["aggregate"] for trial in trial_reports])
    sample_stability = []
    for sample_index, trial_metrics in sample_trials.items():
        mean_metrics, std_metrics = summarize_metric_dicts(
            [
                {
                    "restoration_success": float(metric["restoration_success"]),
                    "restored_mpjpe": metric["restored_mpjpe"],
                    "restored_low_conf_mpjpe": metric["restored_low_conf_mpjpe"],
                    "restored_reliable_mpjpe": metric["restored_reliable_mpjpe"],
                    "overall_position_improvement": metric["overall_position_improvement"],
                    "low_conf_position_improvement": metric["low_conf_position_improvement"],
                    "reliable_position_change": metric["reliable_position_change"],
                }
                for metric in trial_metrics
            ]
        )
        sample_stability.append(
            {
                "sample_index": sample_index,
                "motion_name": trial_metrics[0]["motion_name"],
                "object_type": trial_metrics[0]["object_type"],
                "success_rate": mean_metrics["restoration_success"],
                "success_count": int(sum(int(metric["restoration_success"]) for metric in trial_metrics)),
                "trial_count": len(trial_metrics),
                "mean_restored_mpjpe": mean_metrics["restored_mpjpe"],
                "std_restored_mpjpe": std_metrics["restored_mpjpe"],
                "mean_restored_low_conf_mpjpe": mean_metrics["restored_low_conf_mpjpe"],
                "std_restored_low_conf_mpjpe": std_metrics["restored_low_conf_mpjpe"],
                "mean_restored_reliable_mpjpe": mean_metrics["restored_reliable_mpjpe"],
                "std_restored_reliable_mpjpe": std_metrics["restored_reliable_mpjpe"],
                "mean_overall_position_improvement": mean_metrics["overall_position_improvement"],
                "mean_low_conf_position_improvement": mean_metrics["low_conf_position_improvement"],
                "mean_reliable_position_change": mean_metrics["reliable_position_change"],
            }
        )

    return {
        "split": args.eval_split,
        "objects_subset": model_args.objects_subset,
        "selected_sample_count": len(selected_samples),
        "selected_motion_names": [sample["motion_name"] for sample in selected_samples],
        "num_trials": args.num_trials,
        "base_seed": args.base_seed,
        "selection_seed": args.selection_seed,
        "shuffle": bool(args.eval_shuffle),
        "sampling_method": args.sampling_method,
        "sampling_steps": int(args.sampling_steps) if args.sampling_steps > 0 else int(getattr(model_args, "diffusion_steps", DEFAULT_DIFFUSION_STEPS)),
        "export_samples": args.export_samples,
        "reliable_position_tolerance": RELIABLE_POSITION_TOLERANCE,
        "reliable_position_ratio_tolerance": RELIABLE_POSITION_RATIO_TOLERANCE,
        "trial_aggregates": [
            {
                "trial_index": trial["trial_index"],
                "trial_seed": trial["trial_seed"],
                "aggregate": trial["aggregate"],
            }
            for trial in trial_reports
        ],
        "aggregate_mean": aggregate_mean,
        "aggregate_std": aggregate_std,
        "best_trial_index": max(trial_reports, key=lambda trial: trial["aggregate"]["successful_repairs"])["trial_index"],
        "worst_trial_index": min(trial_reports, key=lambda trial: trial["aggregate"]["successful_repairs"])["trial_index"],
        "sample_stability": sample_stability,
    }


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

    model, diffusion = create_model_and_diffusion_general_skeleton(model_args)
    if args.model_path:
        state_dict = torch.load(Path(args.model_path).resolve(), map_location="cpu")
        if "model_avg" in state_dict:
            state_dict = state_dict["model_avg"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
        load_model(model, state_dict)
    model.to(device)

    history = train_stochastic_debug(
        args=args,
        model_args=model_args,
        model=model,
        diffusion=diffusion,
        device=device,
        output_dir=output_dir,
    )

    with open(output_dir / "training_history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    selected_samples = collect_eval_samples(args, model_args)
    sampling_report = stochastic_eval(
        args=args,
        model_args=model_args,
        train_model=model,
        diffusion=diffusion,
        selected_samples=selected_samples,
        device=device,
        output_dir=output_dir,
    )

    final_report = {
        "model_path": str(Path(args.model_path).resolve()) if args.model_path else "",
        "output_dir": str(output_dir),
        "num_steps": args.num_steps,
        "batch_size": args.batch_size,
        "num_frames": model_args.num_frames,
        "sample_limit": model_args.sample_limit,
        "schedule_sampler": args.schedule_sampler,
        "train_shuffle": not args.disable_train_shuffle,
        "final_training_loss": history[-1].get("loss") if history else None,
        "lambda_confidence_recon": model_args.lambda_confidence_recon,
        "lambda_repair_recon": model_args.lambda_repair_recon,
        "lambda_root": model_args.lambda_root,
        "lambda_velocity": model_args.lambda_velocity,
        "preservation_confidence_threshold": diffusion.preservation_confidence_threshold,
        "preservation_confidence_power": diffusion.preservation_confidence_power,
        "stochastic_eval": sampling_report,
    }
    with open(output_dir / "report.json", "w", encoding="utf-8") as handle:
        json.dump(final_report, handle, indent=2)

    print(json.dumps(final_report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())