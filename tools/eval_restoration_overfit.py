"""
Restoration Quality Evaluation Tool

Description:
    Evaluates the restoration quality of a trained diffusion model on corrupted motion sequences.
    This tool is specifically designed for assessing tiny-overfit checkpoints and validating
    the model's ability to reconstruct high-quality motions from corrupted inputs.

Features:
    - Loads trained model from checkpoint (with automatic args.json loading)
    - Evaluates on multiple motion samples with controllable batch processing
    - Exports restored motions alongside original and corrupted versions
    - Computes position improvement metrics for reliable and repair regions
    - Supports flexible dataset split selection (train/val/test)

Usage:
    python eval_restoration_overfit.py \\
        --model-path ./checkpoints/model_00050000.pt \\
        --output-dir ./eval_results \\
        --num-eval-samples 10 \\
        --batch-size 1 \\
        --device 0 \\
        --split val

    # Minimal example:
    python eval_restoration_overfit.py \\
        --model-path ./checkpoints/model_00050000.pt \\
        --output-dir ./eval_results

Required Arguments:
    --model-path: Path to model checkpoint (args.json must be in same directory)
    --output-dir: Directory to save restoration outputs and evaluation results

Key Optional Arguments:
    --num-eval-samples: Number of samples to restore (default: 4)
    --split: Dataset split to evaluate on - 'train', 'val', or 'test' (default: 'train')
    --batch-size: Batch size for evaluation (default: 1)
    --device: CUDA device ID or -1 for CPU (default: 0)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import BVH
from InverseKinematics import animation_from_positions


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loaders.get_data import get_dataset_loader
from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np
from utils.fixseed import fixseed
from utils import dist_util
from utils.model_util import create_model_and_diffusion_general_skeleton, load_model


RELIABLE_POSITION_TOLERANCE = 0.002
RELIABLE_POSITION_RATIO_TOLERANCE = 1.15
POSITION_IMPROVEMENT_EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate restoration quality for a tiny-overfit checkpoint.")
    parser.add_argument("--model-path", required=True, help="Path to model####.pt checkpoint.")
    parser.add_argument("--output-dir", required=True, help="Directory to write restoration outputs into.")
    parser.add_argument("--num-eval-samples", default=4, type=int, help="How many samples to restore and export.")
    parser.add_argument("--batch-size", default=1, type=int, help="Batch size for evaluation. Use 1 for easier inspection.")
    parser.add_argument("--device", default=0, type=int, help="CUDA device id. Use -1 for CPU.")
    parser.add_argument("--seed", default=10, type=int, help="Random seed.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Dataset split to sample from.")
    parser.add_argument("--sample-limit", default=-1, type=int, help="Override dataset sample limit. -1 keeps checkpoint args.")
    parser.add_argument("--num-workers", default=0, type=int, help="DataLoader workers for evaluation.")
    parser.add_argument("--sampling-method", default="p", choices=["p", "ddim", "plms"], help="Diffusion sampler to use. DDIM and PLMS can be substantially faster when paired with fewer sampling steps.")
    parser.add_argument("--sampling-steps", default=0, type=int, help="Respaced diffusion steps. 0 keeps the checkpoint step count.")
    parser.add_argument("--ddim-eta", default=0.0, type=float, help="DDIM eta parameter. Ignored for other samplers.")
    return parser.parse_args()


def load_model_args(model_path: Path) -> SimpleNamespace:
    args_path = model_path.parent / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"Arguments json file was not found next to checkpoint: {args_path}")
    with open(args_path, "r", encoding="utf-8") as handle:
        return SimpleNamespace(**json.load(handle))


def move_cond_to_device(cond: dict, device: torch.device) -> dict:
    moved = {"y": {}}
    for key, value in cond["y"].items():
        moved["y"][key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
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


def export_motion_array(sample_dir: Path, name: str, motion: np.ndarray) -> None:
    np.save(sample_dir / f"{name}.npy", motion)


def configure_sampling(model_args: SimpleNamespace, args: argparse.Namespace) -> None:
    diffusion_steps = int(getattr(model_args, "diffusion_steps", 100))
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


def export_motion_bvh(sample_dir: Path, name: str, positions: np.ndarray, parents: list[int], offsets: np.ndarray, joints_names: list[str]) -> None:
    out_anim, _, _ = animation_from_positions(positions=positions.astype(np.float32), parents=parents, offsets=offsets, iterations=150)
    if out_anim is not None:
        BVH.save(str(sample_dir / f"{name}.bvh"), out_anim, joints_names)


def main() -> int:
    args = parse_args()
    model_path = Path(args.model_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_args = load_model_args(model_path)
    configure_sampling(model_args, args)
    fixseed(args.seed)
    dist_util.setup_dist(args.device)

    if args.sample_limit >= 0:
        model_args.sample_limit = args.sample_limit
    model_args.device = args.device
    model_args.batch_size = args.batch_size

    data = get_dataset_loader(
        batch_size=args.batch_size,
        num_frames=model_args.num_frames,
        split=args.split,
        temporal_window=model_args.temporal_window,
        t5_name=model_args.t5_name,
        balanced=False,
        objects_subset=model_args.objects_subset,
        num_workers=args.num_workers,
        prefetch_factor=getattr(model_args, "prefetch_factor", 2),
        sample_limit=model_args.sample_limit,
        shuffle=False,
        drop_last=False,
    )
    cond_dict = data.dataset.motion_dataset.cond_dict

    model, diffusion = create_model_and_diffusion_general_skeleton(model_args)
    state_dict = torch.load(model_path, map_location="cpu")
    if "model_avg" in state_dict:
        state_dict = state_dict["model_avg"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]
    load_model(model, state_dict)
    model.to(dist_util.dev())
    model.eval()

    summary: list[dict[str, object]] = []
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

    exported = 0
    for batch_index, (motion, cond) in enumerate(data):
        motion = motion.to(dist_util.dev(), non_blocking=True)
        cond = move_cond_to_device(cond, dist_util.dev())
        with torch.inference_mode():
            restored = sample_motion_batch(
                diffusion,
                model,
                motion.shape,
                cond,
                args.sampling_method,
                args.ddim_eta,
            )

        batch_size = motion.shape[0]
        for item_index in range(batch_size):
            n_joints = int(cond["y"]["n_joints"][item_index].item())
            length = int(cond["y"]["lengths"][item_index].item())
            object_type = cond["y"]["object_type"][item_index]
            parents = [int(parent) for parent in cond_dict[object_type]["parents"]]

            target_norm = motion[item_index, :n_joints, :, :length].detach().cpu()
            reference_norm = cond["y"]["reference_motion"][item_index, :n_joints, :, :length].detach().cpu()
            restored_norm = restored[item_index, :n_joints, :, :length].detach().cpu()
            confidence = cond["y"]["soft_confidence_mask"][item_index, :n_joints, :, :length].detach().cpu().clamp(0.0, 1.0)
            low_conf_weights = (1.0 - confidence).clamp(min=0.0, max=1.0)
            reliable_weights = confidence

            feature_baseline_mse = float(torch.mean((reference_norm - target_norm) ** 2).item())
            feature_restored_mse = float(torch.mean((restored_norm - target_norm) ** 2).item())
            feature_baseline_low_conf_mse = weighted_mse(reference_norm, target_norm, low_conf_weights)
            feature_restored_low_conf_mse = weighted_mse(restored_norm, target_norm, low_conf_weights)
            feature_baseline_reliable_mse = weighted_mse(reference_norm, target_norm, reliable_weights)
            feature_restored_reliable_mse = weighted_mse(restored_norm, target_norm, reliable_weights)

            mean = cond_dict[object_type]["mean"].astype(np.float32)
            std = (cond_dict[object_type]["std"].astype(np.float32) + 1e-6)

            target_denorm = target_norm.permute(2, 0, 1).numpy() * std[None, :n_joints, :] + mean[None, :n_joints, :]
            reference_denorm = reference_norm.permute(2, 0, 1).numpy() * std[None, :n_joints, :] + mean[None, :n_joints, :]
            restored_denorm = restored_norm.permute(2, 0, 1).numpy() * std[None, :n_joints, :] + mean[None, :n_joints, :]
            confidence_np = confidence.permute(2, 0, 1).numpy().astype(np.float32)[..., 0]

            target_positions = recover_from_bvh_ric_np(target_denorm.astype(np.float32))
            reference_positions = recover_from_bvh_ric_np(reference_denorm.astype(np.float32))
            restored_positions = recover_from_bvh_ric_np(restored_denorm.astype(np.float32))
            position_metrics = compute_position_metrics(
                reference_positions=reference_positions,
                restored_positions=restored_positions,
                target_positions=target_positions,
                confidence=confidence_np,
            )
            restoration_flags = classify_restoration(position_metrics)

            totals["baseline_mpjpe"].append(position_metrics["baseline_mpjpe"])
            totals["restored_mpjpe"].append(position_metrics["restored_mpjpe"])
            totals["baseline_low_conf_mpjpe"].append(position_metrics["baseline_low_conf_mpjpe"])
            totals["restored_low_conf_mpjpe"].append(position_metrics["restored_low_conf_mpjpe"])
            totals["baseline_reliable_mpjpe"].append(position_metrics["baseline_reliable_mpjpe"])
            totals["restored_reliable_mpjpe"].append(position_metrics["restored_reliable_mpjpe"])
            totals["feature_baseline_mse"].append(feature_baseline_mse)
            totals["feature_restored_mse"].append(feature_restored_mse)
            totals["feature_baseline_low_conf_mse"].append(feature_baseline_low_conf_mse)
            totals["feature_restored_low_conf_mse"].append(feature_restored_low_conf_mse)
            totals["feature_baseline_reliable_mse"].append(feature_baseline_reliable_mse)
            totals["feature_restored_reliable_mse"].append(feature_restored_reliable_mse)
            totals["restoration_success"].append(float(restoration_flags["restoration_success"]))

            sample_dir = output_dir / f"sample_{exported:02d}_{object_type}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            export_motion_array(sample_dir, "clean_target", target_denorm.astype(np.float32))
            export_motion_array(sample_dir, "corrupted_reference", reference_denorm.astype(np.float32))
            export_motion_array(sample_dir, "restored_prediction", restored_denorm.astype(np.float32))
            np.save(sample_dir / "soft_confidence_mask.npy", confidence.permute(2, 0, 1).numpy().astype(np.float32))

            offsets = cond_dict[object_type]["offsets"]
            joints_names = cond_dict[object_type]["joints_names"]
            for bvh_name, positions in [
                ("clean_target", target_positions),
                ("corrupted_reference", reference_positions),
                ("restored_prediction", restored_positions),
            ]:
                export_motion_bvh(sample_dir, bvh_name, positions.astype(np.float32), parents, offsets, joints_names)

            metrics = {
                "object_type": object_type,
                "sample_index": exported,
                "source_batch_index": batch_index,
                "length": length,
                "n_joints": n_joints,
                **position_metrics,
                **restoration_flags,
                "overall_position_improvement": position_metrics["baseline_mpjpe"] - position_metrics["restored_mpjpe"],
                "low_conf_position_improvement": position_metrics["baseline_low_conf_mpjpe"] - position_metrics["restored_low_conf_mpjpe"],
                "reliable_position_change": position_metrics["baseline_reliable_mpjpe"] - position_metrics["restored_reliable_mpjpe"],
                "feature_baseline_mse": feature_baseline_mse,
                "feature_restored_mse": feature_restored_mse,
                "feature_baseline_low_conf_mse": feature_baseline_low_conf_mse,
                "feature_restored_low_conf_mse": feature_restored_low_conf_mse,
                "feature_baseline_reliable_mse": feature_baseline_reliable_mse,
                "feature_restored_reliable_mse": feature_restored_reliable_mse,
                "feature_overall_improvement": feature_baseline_mse - feature_restored_mse,
                "feature_low_conf_improvement": feature_baseline_low_conf_mse - feature_restored_low_conf_mse,
                "feature_reliable_region_change": feature_baseline_reliable_mse - feature_restored_reliable_mse,
            }
            with open(sample_dir / "metrics.json", "w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=2)
            summary.append(metrics)
            exported += 1

            if exported >= args.num_eval_samples:
                break
        if exported >= args.num_eval_samples:
            break

    if not summary:
        raise RuntimeError("No evaluation samples were exported.")

    aggregate = {key: float(np.mean(values)) for key, values in totals.items() if values}
    aggregate["avg_overall_position_improvement"] = aggregate["baseline_mpjpe"] - aggregate["restored_mpjpe"]
    aggregate["avg_low_conf_position_improvement"] = aggregate["baseline_low_conf_mpjpe"] - aggregate["restored_low_conf_mpjpe"]
    aggregate["avg_feature_overall_improvement"] = aggregate["feature_baseline_mse"] - aggregate["feature_restored_mse"]
    aggregate["avg_feature_low_conf_improvement"] = aggregate["feature_baseline_low_conf_mse"] - aggregate["feature_restored_low_conf_mse"]
    aggregate["successful_repairs"] = int(sum(int(sample["restoration_success"]) for sample in summary))
    aggregate["evaluated_samples"] = exported

    report = {
        "model_path": str(model_path),
        "output_dir": str(output_dir),
        "split": args.split,
        "sample_limit": int(model_args.sample_limit),
        "sampling_method": args.sampling_method,
        "sampling_steps": int(args.sampling_steps) if args.sampling_steps > 0 else int(getattr(model_args, "diffusion_steps", 100)),
        "reliable_position_tolerance": RELIABLE_POSITION_TOLERANCE,
        "reliable_position_ratio_tolerance": RELIABLE_POSITION_RATIO_TOLERANCE,
        "aggregate": aggregate,
        "samples": summary,
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())