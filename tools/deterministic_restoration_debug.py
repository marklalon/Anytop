"""
Deterministic Restoration Sanity Check Tool

Description:
    Performs deterministic restoration on fixed offline corrupted motions using a trained diffusion model.
    This tool is designed for debugging and validating the restoration pipeline by applying
    deterministic optimization steps to restore motion quality from corrupted inputs.

Features:
    - Loads preprocessed offline corrupted motion data from dataset subsets
    - Applies deterministic diffusion sampling at fixed timesteps
    - Tracks position improvement metrics in reliable vs low-confidence regions
    - Saves checkpoints and restoration results during optimization
    - Supports single-batch processing for deterministic debugging
    - Note: Offline samples are pre-generated during preprocessing; this script loads and uses them

Usage:
    python deterministic_restoration_debug.py \\
        --output-dir ./debug_output \\
        --model-path ./checkpoints/model_00500000.pt \\
        --objects-subset quadropeds_clean \\
        --num-steps 800 \\
        --batch-size 1 \\
        --device 0 \\
        --num-frames 60

    # Minimal example with default values:
    python deterministic_restoration_debug.py \\
        --output-dir ./debug_output

Required Arguments:
    --output-dir: Directory for debug outputs, checkpoints, and restoration reports
    
Note: Offline corrupted samples must be preprocessed and available in the dataset.
      Use export_corrupted_truebones_samples.py to generate them if needed.

Key Optional Arguments:
    --model-path: Path to model checkpoint (auto-loads args.json from same directory)
    --objects-subset: Object types to load from preprocessed dataset (default: 'quadropeds_clean')
    --num-steps: Number of optimization steps (default: 800)
    --num-frames: Maximum frames per motion (default: 60)
    --fixed-timestep: Diffusion timestep to use (default: 10)
    --lr: Learning rate for optimization (default: 1e-4)
"""

import argparse
import copy
import json
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
from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np
from data_loaders.truebones.truebones_utils.plot_script import plot_general_skeleton_3d_motion
from utils.fixseed import fixseed
from utils import dist_util
from utils.model_util import create_model_and_diffusion_general_skeleton, load_model


RELIABLE_POSITION_TOLERANCE = 0.01
POSITION_IMPROVEMENT_EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic restoration sanity check on fixed offline corrupted motions.")
    parser.add_argument("--output-dir", required=True, help="Directory to write checkpoints and reports into.")
    parser.add_argument("--model-path", default="", help="Optional checkpoint to initialize from; args.json next to it supplies model config.")
    parser.add_argument("--device", default=0, type=int, help="CUDA device id. Use -1 for CPU.")
    parser.add_argument("--seed", default=10, type=int, help="Global seed.")
    parser.add_argument("--objects-subset", default="quadropeds_clean", help="Object subset to load from preprocessed dataset.")
    parser.add_argument("--sample-limit", default=0, type=int, help="Maximum number of motions to load from the dataset subset. 0 means all available.")
    parser.add_argument("--num-frames", default=60, type=int, help="Maximum frames per motion.")
    parser.add_argument("--batch-size", default=1, type=int, help="Batch size. Keep 1 for deterministic debugging.")
    parser.add_argument("--num-workers", default=0, type=int, help="DataLoader workers.")
    parser.add_argument("--num-steps", default=800, type=int, help="Number of deterministic optimization steps.")
    parser.add_argument("--log-interval", default=50, type=int, help="How often to print training metrics.")
    parser.add_argument("--save-interval", default=200, type=int, help="How often to save debug checkpoints.")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    parser.add_argument("--weight-decay", default=0.0, type=float, help="Optimizer weight decay.")
    parser.add_argument("--fixed-timestep", default=10, type=int, help="Deterministic diffusion timestep used for every sample.")
    parser.add_argument("--noise-mode", default="zero", choices=["zero", "fixed-random"], help="Use zero noise or one fixed random noise tensor per sample.")
    parser.add_argument("--latent-dim", default=32, type=int, help="Model width if no checkpoint is provided.")
    parser.add_argument("--layers", default=2, type=int, help="Model depth if no checkpoint is provided.")
    parser.add_argument("--lambda-confidence-recon", default=None, type=float, help="Reliable-region preservation weight. Overrides checkpoint args when set.")
    parser.add_argument("--lambda-repair-recon", default=None, type=float, help="Low-confidence repair reconstruction weight. Overrides checkpoint args when set.")
    parser.add_argument("--lambda-root", default=None, type=float, help="Root consistency weight. Overrides checkpoint args when set.")
    parser.add_argument("--lambda-velocity", default=None, type=float, help="Velocity consistency weight. Overrides checkpoint args when set.")
    parser.add_argument("--t5-name", default="t5-base", help="T5 model name when no checkpoint args are available.")
    parser.add_argument("--temporal-window", default=31, type=int, help="Temporal window size when no checkpoint args are available.")
    parser.add_argument("--skip-t5", action="store_true", help="Disable joint-name text conditioning when no checkpoint is provided.")
    parser.add_argument("--value-emb", action="store_true", help="Enable value embeddings when no checkpoint is provided.")
    parser.add_argument("--cond-mask-prob", default=0.1, type=float, help="Condition mask probability when no checkpoint is provided.")
    parser.add_argument("--noise-schedule", default="cosine", choices=["linear", "cosine"], help="Diffusion noise schedule.")
    parser.add_argument("--sigma-small", default=True, type=bool, help="Use smaller diffusion sigma values.")
    parser.add_argument("--diffusion-steps", default=100, type=int, help="Diffusion step count kept for config parity.")
    return parser.parse_args()


def load_model_args(args: argparse.Namespace) -> SimpleNamespace:
    if args.model_path:
        model_path = Path(args.model_path).resolve()
        args_path = model_path.parent / "args.json"
        if not args_path.exists():
            raise FileNotFoundError(f"Arguments json file was not found next to checkpoint: {args_path}")
        with open(args_path, "r", encoding="utf-8") as handle:
            model_args = SimpleNamespace(**json.load(handle))
        model_args.model_path = str(model_path)
    else:
        model_args = SimpleNamespace(
            device=args.device,
            batch_size=args.batch_size,
            num_frames=args.num_frames,
            temporal_window=args.temporal_window,
            t5_name=args.t5_name,
            objects_subset=args.objects_subset,
            num_workers=args.num_workers,
            prefetch_factor=2,
            sample_limit=args.sample_limit,
            latent_dim=args.latent_dim,
            layers=args.layers,
            lambda_confidence_recon=2.0 if args.lambda_confidence_recon is None else args.lambda_confidence_recon,
            lambda_repair_recon=1.0 if args.lambda_repair_recon is None else args.lambda_repair_recon,
            lambda_root=0.25 if args.lambda_root is None else args.lambda_root,
            lambda_velocity=0.1 if args.lambda_velocity is None else args.lambda_velocity,
            lambda_fs=0.0,
            lambda_geo=0.0,
            noise_schedule=args.noise_schedule,
            sigma_small=args.sigma_small,
            cond_mask_prob=args.cond_mask_prob,
            skip_t5=args.skip_t5,
            value_emb=args.value_emb,
            disable_reference_branch=False,
            reference_dropout_threshold=0.05,
        )
    model_args.device = args.device
    model_args.batch_size = args.batch_size
    model_args.num_frames = args.num_frames if not hasattr(model_args, "num_frames") else model_args.num_frames
    model_args.temporal_window = getattr(model_args, "temporal_window", args.temporal_window)
    model_args.t5_name = getattr(model_args, "t5_name", args.t5_name)
    model_args.objects_subset = args.objects_subset
    model_args.sample_limit = args.sample_limit
    model_args.num_workers = args.num_workers
    model_args.prefetch_factor = getattr(model_args, "prefetch_factor", 2)
    model_args.noise_schedule = getattr(model_args, "noise_schedule", args.noise_schedule)
    model_args.sigma_small = getattr(model_args, "sigma_small", args.sigma_small)
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
    model_args.latent_dim = getattr(model_args, "latent_dim", args.latent_dim)
    model_args.layers = getattr(model_args, "layers", args.layers)
    model_args.cond_mask_prob = getattr(model_args, "cond_mask_prob", args.cond_mask_prob)
    model_args.skip_t5 = getattr(model_args, "skip_t5", args.skip_t5)
    model_args.value_emb = getattr(model_args, "value_emb", args.value_emb)
    model_args.disable_reference_branch = getattr(model_args, "disable_reference_branch", False)
    model_args.reference_dropout_threshold = getattr(model_args, "reference_dropout_threshold", 0.05)
    return model_args


def clone_batch_cond(cond: dict) -> dict:
    cloned = {"y": {}}
    for key, value in cond["y"].items():
        if torch.is_tensor(value):
            cloned["y"][key] = value.detach().clone()
        elif isinstance(value, list):
            cloned["y"][key] = copy.deepcopy(value)
        else:
            cloned["y"][key] = value
    return cloned


def move_cond_to_device(cond: dict, device: torch.device) -> dict:
    moved = {"y": {}}
    for key, value in cond["y"].items():
        moved["y"][key] = value.to(device) if torch.is_tensor(value) else value
    return moved


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
    reference_motion: np.ndarray,
    predicted_motion: np.ndarray,
    target_motion: np.ndarray,
    confidence: np.ndarray,
) -> dict[str, float]:
    target_positions = recover_from_bvh_ric_np(target_motion)
    reference_positions = recover_from_bvh_ric_np(reference_motion)
    predicted_positions = recover_from_bvh_ric_np(predicted_motion)

    reference_error = np.linalg.norm(reference_positions - target_positions, axis=-1)
    predicted_error = np.linalg.norm(predicted_positions - target_positions, axis=-1)
    low_conf_weights = np.clip(1.0 - confidence, 0.0, 1.0)
    reliable_weights = np.clip(confidence, 0.0, 1.0)

    reference_frame_error = reference_error.mean(axis=1)
    predicted_frame_error = predicted_error.mean(axis=1)
    return {
        "baseline_mpjpe": float(reference_error.mean()),
        "pred_xstart_mpjpe": float(predicted_error.mean()),
        "baseline_low_conf_mpjpe": weighted_position_error(reference_positions, target_positions, low_conf_weights),
        "pred_xstart_low_conf_mpjpe": weighted_position_error(predicted_positions, target_positions, low_conf_weights),
        "baseline_reliable_mpjpe": weighted_position_error(reference_positions, target_positions, reliable_weights),
        "pred_xstart_reliable_mpjpe": weighted_position_error(predicted_positions, target_positions, reliable_weights),
        "baseline_p95_joint_error": float(np.percentile(reference_error, 95)),
        "pred_xstart_p95_joint_error": float(np.percentile(predicted_error, 95)),
        "baseline_max_joint_error": float(reference_error.max()),
        "pred_xstart_max_joint_error": float(predicted_error.max()),
        "baseline_better_frame_count": int(np.sum(reference_frame_error < predicted_frame_error)),
        "pred_xstart_better_frame_count": int(np.sum(predicted_frame_error < reference_frame_error)),
    }


def classify_restoration(position_metrics: dict[str, float]) -> dict[str, object]:
    low_conf_improved = position_metrics["pred_xstart_low_conf_mpjpe"] + POSITION_IMPROVEMENT_EPS < position_metrics["baseline_low_conf_mpjpe"]
    reliable_preserved = position_metrics["pred_xstart_reliable_mpjpe"] <= position_metrics["baseline_reliable_mpjpe"] + RELIABLE_POSITION_TOLERANCE
    overall_improved = position_metrics["pred_xstart_mpjpe"] + POSITION_IMPROVEMENT_EPS < position_metrics["baseline_mpjpe"]
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


def export_motion(sample_dir: Path, name: str, motion: np.ndarray, parents: list[int]) -> None:
    np.save(sample_dir / f"{name}.npy", motion)
    positions = recover_from_bvh_ric_np(motion)
    plot_general_skeleton_3d_motion(str(sample_dir / f"{name}.mp4"), parents, positions, title=name, fps=20)


def build_fixed_noise(sample_index: int, shape: torch.Size, device: torch.device, mode: str, seed: int) -> torch.Tensor:
    if mode == "zero":
        return torch.zeros(shape, dtype=torch.float32, device=device)
    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(seed + sample_index)
    return torch.randn(shape, generator=generator, device=device, dtype=torch.float32)


def collect_samples(args: argparse.Namespace, model_args: SimpleNamespace) -> tuple[list[dict[str, object]], dict[str, dict[str, np.ndarray]]]:
    data = get_dataset_loader(
        batch_size=1,
        num_frames=model_args.num_frames,
        split="train",
        temporal_window=model_args.temporal_window,
        t5_name=model_args.t5_name,
        balanced=False,
        objects_subset=model_args.objects_subset,
        num_workers=args.num_workers,
        prefetch_factor=model_args.prefetch_factor,
        sample_limit=args.sample_limit,
        shuffle=False,
        drop_last=False,
    )
    cond_dict = data.dataset.motion_dataset.cond_dict
    samples: list[dict[str, object]] = []

    for sample_index, (motion, cond) in enumerate(data):
        cond_cpu = clone_batch_cond(cond)
        motion_cpu = motion.detach().clone().float()
        motion_name = cond_cpu["y"]["motion_name"][0]
        object_type = cond_cpu["y"]["object_type"][0]
        n_joints = int(cond_cpu["y"]["n_joints"][0].item())
        length = int(cond_cpu["y"]["lengths"][0].item())
        mean = cond_dict[object_type]["mean"].astype(np.float32)
        std = (cond_dict[object_type]["std"].astype(np.float32) + 1e-6)

        samples.append(
            {
                "motion": motion_cpu,
                "cond": cond_cpu,
                "motion_name": motion_name,
                "object_type": object_type,
                "length": length,
                "n_joints": n_joints,
                "parents": [int(parent) for parent in cond_dict[object_type]["parents"]],
                "mean": mean,
                "std": std,
            }
        )

    if not samples:
        raise RuntimeError("No samples were collected.")
    return samples, cond_dict


def deterministic_eval(
    model: torch.nn.Module,
    diffusion,
    samples: list[dict[str, object]],
    device: torch.device,
    fixed_timestep: int,
    noise_mode: str,
    seed: int,
    output_dir: Path,
) -> dict[str, object]:
    model.eval()
    summary: list[dict[str, object]] = []
    totals = {
        "baseline_mpjpe": [],
        "pred_xstart_mpjpe": [],
        "baseline_low_conf_mpjpe": [],
        "pred_xstart_low_conf_mpjpe": [],
        "baseline_reliable_mpjpe": [],
        "pred_xstart_reliable_mpjpe": [],
        "feature_baseline_mse": [],
        "feature_pred_xstart_mse": [],
        "feature_baseline_low_conf_mse": [],
        "feature_pred_xstart_low_conf_mse": [],
        "feature_baseline_reliable_mse": [],
        "feature_pred_xstart_reliable_mse": [],
        "restoration_success": [],
    }
    eval_dir = output_dir / "deterministic_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for sample_index, sample in enumerate(samples):
            motion = sample["motion"].to(device)
            cond = move_cond_to_device(sample["cond"], device)
            fixed_t = torch.full((motion.shape[0],), fixed_timestep, device=device, dtype=torch.long)
            fixed_noise = build_fixed_noise(sample_index, motion.shape, device, noise_mode, seed)
            x_t = diffusion.q_sample(motion, fixed_t, noise=fixed_noise)
            pred = diffusion.p_mean_variance(
                model,
                x_t,
                fixed_t,
                clip_denoised=False,
                model_kwargs=cond,
            )["pred_xstart"].detach().cpu()

            n_joints = int(sample["n_joints"])
            length = int(sample["length"])
            target_norm = sample["motion"][0, :n_joints, :, :length]
            reference_norm = sample["cond"]["y"]["reference_motion"][0, :n_joints, :, :length]
            pred_norm = pred[0, :n_joints, :, :length]
            confidence = sample["cond"]["y"]["soft_confidence_mask"][0, :n_joints, :, :length].clamp(0.0, 1.0)
            fused_pred_norm = pred_norm
            low_conf_weights = (1.0 - confidence).clamp(min=0.0, max=1.0)
            reliable_weights = confidence

            feature_baseline_mse = float(torch.mean((reference_norm - target_norm) ** 2).item())
            feature_pred_mse = float(torch.mean((fused_pred_norm - target_norm) ** 2).item())
            feature_baseline_low_conf_mse = weighted_mse(reference_norm, target_norm, low_conf_weights)
            feature_pred_low_conf_mse = weighted_mse(fused_pred_norm, target_norm, low_conf_weights)
            feature_baseline_reliable_mse = weighted_mse(reference_norm, target_norm, reliable_weights)
            feature_pred_reliable_mse = weighted_mse(fused_pred_norm, target_norm, reliable_weights)

            mean = sample["mean"]
            std = sample["std"]
            parents = sample["parents"]

            target_denorm = target_norm.permute(2, 0, 1).numpy() * std[None, :n_joints, :] + mean[None, :n_joints, :]
            reference_denorm = reference_norm.permute(2, 0, 1).numpy() * std[None, :n_joints, :] + mean[None, :n_joints, :]
            pred_denorm = fused_pred_norm.permute(2, 0, 1).numpy() * std[None, :n_joints, :] + mean[None, :n_joints, :]
            confidence_np = confidence.permute(2, 0, 1).numpy().astype(np.float32)[..., 0]
            position_metrics = compute_position_metrics(
                reference_motion=reference_denorm.astype(np.float32),
                predicted_motion=pred_denorm.astype(np.float32),
                target_motion=target_denorm.astype(np.float32),
                confidence=confidence_np,
            )
            restoration_flags = classify_restoration(position_metrics)

            totals["baseline_mpjpe"].append(position_metrics["baseline_mpjpe"])
            totals["pred_xstart_mpjpe"].append(position_metrics["pred_xstart_mpjpe"])
            totals["baseline_low_conf_mpjpe"].append(position_metrics["baseline_low_conf_mpjpe"])
            totals["pred_xstart_low_conf_mpjpe"].append(position_metrics["pred_xstart_low_conf_mpjpe"])
            totals["baseline_reliable_mpjpe"].append(position_metrics["baseline_reliable_mpjpe"])
            totals["pred_xstart_reliable_mpjpe"].append(position_metrics["pred_xstart_reliable_mpjpe"])
            totals["feature_baseline_mse"].append(feature_baseline_mse)
            totals["feature_pred_xstart_mse"].append(feature_pred_mse)
            totals["feature_baseline_low_conf_mse"].append(feature_baseline_low_conf_mse)
            totals["feature_pred_xstart_low_conf_mse"].append(feature_pred_low_conf_mse)
            totals["feature_baseline_reliable_mse"].append(feature_baseline_reliable_mse)
            totals["feature_pred_xstart_reliable_mse"].append(feature_pred_reliable_mse)
            totals["restoration_success"].append(float(restoration_flags["restoration_success"]))

            sample_eval_dir = eval_dir / f"sample_{sample_index:02d}_{sample['object_type']}"
            sample_eval_dir.mkdir(parents=True, exist_ok=True)
            export_motion(sample_eval_dir, "pred_xstart_restored", pred_denorm.astype(np.float32), parents)

            metrics = {
                "sample_index": sample_index,
                "object_type": sample["object_type"],
                "length": length,
                "n_joints": n_joints,
                **position_metrics,
                **restoration_flags,
                "overall_position_improvement": position_metrics["baseline_mpjpe"] - position_metrics["pred_xstart_mpjpe"],
                "low_conf_position_improvement": position_metrics["baseline_low_conf_mpjpe"] - position_metrics["pred_xstart_low_conf_mpjpe"],
                "reliable_position_change": position_metrics["baseline_reliable_mpjpe"] - position_metrics["pred_xstart_reliable_mpjpe"],
                "feature_baseline_mse": feature_baseline_mse,
                "feature_pred_xstart_mse": feature_pred_mse,
                "feature_baseline_low_conf_mse": feature_baseline_low_conf_mse,
                "feature_pred_xstart_low_conf_mse": feature_pred_low_conf_mse,
                "feature_baseline_reliable_mse": feature_baseline_reliable_mse,
                "feature_pred_xstart_reliable_mse": feature_pred_reliable_mse,
                "feature_overall_improvement": feature_baseline_mse - feature_pred_mse,
                "feature_low_conf_improvement": feature_baseline_low_conf_mse - feature_pred_low_conf_mse,
                "feature_reliable_region_change": feature_baseline_reliable_mse - feature_pred_reliable_mse,
            }
            with open(sample_eval_dir / "metrics.json", "w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=2)
            summary.append(metrics)

    aggregate = {key: float(np.mean(values)) for key, values in totals.items() if values}
    aggregate["avg_overall_position_improvement"] = aggregate["baseline_mpjpe"] - aggregate["pred_xstart_mpjpe"]
    aggregate["avg_low_conf_position_improvement"] = aggregate["baseline_low_conf_mpjpe"] - aggregate["pred_xstart_low_conf_mpjpe"]
    aggregate["avg_feature_overall_improvement"] = aggregate["feature_baseline_mse"] - aggregate["feature_pred_xstart_mse"]
    aggregate["avg_feature_low_conf_improvement"] = aggregate["feature_baseline_low_conf_mse"] - aggregate["feature_pred_xstart_low_conf_mse"]
    aggregate["successful_repairs"] = int(sum(int(sample["restoration_success"]) for sample in summary))
    aggregate["evaluated_samples"] = len(summary)
    report = {
        "fixed_timestep": fixed_timestep,
        "noise_mode": noise_mode,
        "aggregate": aggregate,
        "reliable_position_tolerance": RELIABLE_POSITION_TOLERANCE,
        "samples": summary,
    }
    with open(eval_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    device = dist_util.dev()

    model_args = load_model_args(args)
    samples, _ = collect_samples(args, model_args)

    model, diffusion = create_model_and_diffusion_general_skeleton(model_args)
    if args.model_path:
        state_dict = torch.load(Path(args.model_path).resolve(), map_location="cpu")
        if "model_avg" in state_dict:
            state_dict = state_dict["model_avg"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
        load_model(model, state_dict)
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[dict[str, float]] = []
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    for step in range(args.num_steps):
        sample_index = step % len(samples)
        sample = samples[sample_index]
        motion = sample["motion"].to(device)
        cond = move_cond_to_device(sample["cond"], device)
        fixed_t = torch.full((motion.shape[0],), args.fixed_timestep, device=device, dtype=torch.long)
        fixed_noise = build_fixed_noise(sample_index, motion.shape, device, args.noise_mode, args.seed)

        optimizer.zero_grad(set_to_none=True)
        losses = diffusion.training_losses(model, motion, fixed_t, model_kwargs=cond, noise=fixed_noise)
        loss = losses["loss"].mean()
        loss.backward()
        optimizer.step()

        record = {key: float(value.mean().item()) for key, value in losses.items() if torch.is_tensor(value)}
        record["step"] = float(step)
        record["sample_index"] = float(sample_index)
        history.append(record)
        if step % args.log_interval == 0 or step == args.num_steps - 1:
            print(
                json.dumps(
                    {
                        "step": step,
                        "sample_index": sample_index,
                        "loss": record.get("loss"),
                        "l_simple": record.get("l_simple"),
                        "confidence_recon_loss": record.get("confidence_recon_loss"),
                        "repair_recon_loss": record.get("repair_recon_loss"),
                    }
                )
            )
        if (step + 1) % args.save_interval == 0 or step == args.num_steps - 1:
            torch.save(model.state_dict(), checkpoints_dir / f"deterministic_step_{step + 1:05d}.pt")

    with open(output_dir / "training_history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    eval_report = deterministic_eval(
        model=model,
        diffusion=diffusion,
        samples=samples,
        device=device,
        fixed_timestep=args.fixed_timestep,
        noise_mode=args.noise_mode,
        seed=args.seed,
        output_dir=output_dir,
    )
    final_report = {
        "model_path": str(Path(args.model_path).resolve()) if args.model_path else "",
        "output_dir": str(output_dir),
        "num_samples_used": len(samples),
        "num_steps": args.num_steps,
        "fixed_timestep": args.fixed_timestep,
        "noise_mode": args.noise_mode,
        "final_training_loss": history[-1].get("loss") if history else None,
        "deterministic_eval": eval_report,
    }
    with open(output_dir / "report.json", "w", encoding="utf-8") as handle:
        json.dump(final_report, handle, indent=2)
    print(json.dumps(final_report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())