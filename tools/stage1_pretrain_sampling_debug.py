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
from data_loaders.truebones.truebones_utils.motion_process import recover_animation_from_motion_np
from eval.motion_quality_scorer import MotionQualityScorer
from tools.eval_motion_score import DEFAULT_CHECKPOINT_DIR, build_report_summary, score_motion_array
from utils.fixseed import fixseed
from utils import dist_util
from utils.model_util import create_model_and_diffusion_general_skeleton, load_model


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
    parser.add_argument(
        "--motion-score-checkpoint-dir",
        default=DEFAULT_CHECKPOINT_DIR,
        help="Motion scorer checkpoint directory or specific checkpoint path used for evaluation reports.",
    )
    parser.add_argument("--device", default=0, type=int, help="CUDA device id. Use -1 for CPU.")
    parser.add_argument("--seed", default=10, type=int, help="Global seed for deterministic setup.")
    parser.add_argument("--objects-subset", default="", help="Override the checkpoint objects_subset when set.")
    parser.add_argument("--action-tags", default="", help="Override the checkpoint action_tags when set, e.g. 'locomotion,attack'.")
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

    if not getattr(model_args, "action_tags", ""):
        model_args.action_tags = getattr(model_args, "motion_name_keywords", "")

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
    if args.action_tags:
        model_args.action_tags = args.action_tags
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


def evaluate_generated_prediction(
    target_norm: torch.Tensor,
    generated_norm: torch.Tensor,
    n_joints: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, object]:
    target_denorm = denormalize_motion(target_norm, n_joints, mean, std).astype(np.float32)
    generated_denorm = denormalize_motion(generated_norm, n_joints, mean, std).astype(np.float32)
    return {
        "target_denorm": target_denorm,
        "generated_denorm": generated_denorm,
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def cleanup_legacy_json_outputs(output_dir: Path) -> None:
    for file_path in output_dir.glob("*.json"):
        if file_path.is_file():
            file_path.unlink()


def build_motion_score_sample_report(
    *,
    score_result: dict[str, object],
    sample: dict[str, object],
    sample_index: int,
    trial_index: int,
    trial_seed: int,
    path: str,
    source_mode: str,
) -> dict[str, object]:
    sample_report = dict(score_result)
    sample_report.update(
        {
            "path": path,
            "scored_from": path,
            "object_type": str(sample["object_type"]),
            "source_mode": source_mode,
            "motion_name": str(sample["motion_name"]),
            "sample_index": int(sample_index),
            "trial_index": int(trial_index),
            "trial_seed": int(trial_seed),
            "length": int(sample["length"]),
            "n_joints": int(sample["n_joints"]),
        }
    )
    return sample_report


def compact_eval_aggregate(samples: list[dict[str, object]], failures: list[dict[str, str]]) -> dict[str, object]:
    report_summary = build_report_summary(samples, failures)
    metrics = dict(report_summary.get("metrics", {}))
    metrics.pop("mahal_distance", None)
    return {
        "scored_files": int(report_summary["scored_files"]),
        "failed_files": int(report_summary["failed_files"]),
        "metrics": metrics,
    }


def compact_detail_sample(sample: dict[str, object]) -> dict[str, object]:
    compact = {
        "sample_index": int(sample["sample_index"]),
        "trial_index": int(sample["trial_index"]),
        "trial_seed": int(sample["trial_seed"]),
        "motion_name": str(sample["motion_name"]),
        "object_type": str(sample["object_type"]),
        "path": str(sample["path"]),
        "source_mode": str(sample["source_mode"]),
        "frame_count": int(sample["frame_count"]),
        "joint_count": int(sample["joint_count"]),
        "quality_score": float(sample["quality_score"]),
        "recognizability_score": float(sample["recognizability_score"]),
        "density_score": float(sample["density_score"]),
        "plausibility_score": float(sample["plausibility_score"]),
        "physics_score": float(sample["physics_score"]),
        "density_distance": float(sample["density_distance"]),
    }
    if int(sample.get("segment_count", 1)) > 1:
        compact["segment_count"] = int(sample["segment_count"])
        compact["segment_lengths"] = [int(length) for length in sample.get("segment_lengths", [])]
    return compact


def build_eval_result(samples: list[dict[str, object]], failures: list[dict[str, str]]) -> dict[str, object]:
    sorted_samples = sorted(samples, key=lambda sample: str(sample["path"]))
    return {
        "aggregate": compact_eval_aggregate(sorted_samples, failures),
        "samples": [compact_detail_sample(sample) for sample in sorted_samples],
        "failures": failures,
    }


def build_selected_sample_manifest(selected_samples: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "sample_index": int(sample["sample_index"]),
            "motion_name": str(sample["motion_name"]),
            "object_type": str(sample["object_type"]),
            "length": int(sample["length"]),
            "n_joints": int(sample["n_joints"]),
        }
        for sample in selected_samples
    ]


def build_summary_eval_section(eval_result: dict[str, object]) -> dict[str, object]:
    aggregate = eval_result["aggregate"]
    metrics = aggregate.get("metrics", {})
    quality_metrics = metrics.get("quality_score", {})
    recognizability_metrics = metrics.get("recognizability_score", {})
    density_metrics = metrics.get("density_score", {})
    return {
        "overall_quality_score": float(quality_metrics.get("mean", 0.0)),
        "quality_score_median": float(quality_metrics.get("median", 0.0)),
        "quality_score_min": float(quality_metrics.get("min", 0.0)),
        "quality_score_max": float(quality_metrics.get("max", 0.0)),
        "recognizability_score_mean": float(recognizability_metrics.get("mean", 0.0)),
        "density_score_mean": float(density_metrics.get("mean", 0.0)),
        "scored_files": int(aggregate.get("scored_files", 0)),
        "failed_files": int(aggregate.get("failed_files", 0)),
    }


def build_eval_report(
    *,
    args: argparse.Namespace,
    model_args: SimpleNamespace,
    selected_samples: list[dict[str, object]],
    num_trials: int,
    sampling_method: str,
    sampling_steps: int,
    motion_scorer: MotionQualityScorer,
    samples: list[dict[str, object]],
    failures: list[dict[str, str]],
) -> dict[str, object]:
    return {
        "split": args.eval_split,
        "objects_subset": model_args.objects_subset,
        "selected_sample_count": len(selected_samples),
        "num_trials": num_trials,
        "sampling_method": sampling_method,
        "sampling_steps": sampling_steps,
        "motion_score_checkpoint": str(motion_scorer.checkpoint_path),
        "motion_score_device": str(motion_scorer.device),
        "result": build_eval_result(samples, failures),
    }


def build_clean_motion_baseline(
    args: argparse.Namespace,
    model_args: SimpleNamespace,
    selected_samples: list[dict[str, object]],
    motion_scorer: MotionQualityScorer,
    output_dir: Path,
) -> dict[str, object]:
    baseline_samples = []
    failures = []

    for sample_index, sample in enumerate(selected_samples):
        n_joints = int(sample["n_joints"])
        length = int(sample["length"])
        target_norm = sample["motion"][0, :n_joints, :, :length]
        target_denorm = denormalize_motion(target_norm, n_joints, sample["mean"], sample["std"]).astype(np.float32)
        sample_path = (
            output_dir
            / "clean_motion_baseline"
            / f"sample_{sample_index:03d}_{sample['object_type']}"
            / "clean_target.npy"
        )
        try:
            score_result = score_motion_array(
                scorer=motion_scorer,
                motion_np=target_denorm,
                object_type=str(sample["object_type"]),
                object_cond=sample["score_object_cond"],
            )
            baseline_samples.append(
                build_motion_score_sample_report(
                    score_result=score_result,
                    sample=sample,
                    sample_index=sample_index,
                    trial_index=0,
                    trial_seed=args.selection_seed,
                    path=str(sample_path),
                    source_mode="stage1_clean_target_denorm",
                )
            )
        except Exception as exc:
            failures.append({"path": str(sample_path), "error": str(exc)})

    sampling_steps = int(args.sampling_steps) if args.sampling_steps > 0 else int(model_args.diffusion_steps)
    baseline_report = build_eval_report(
        args=args,
        model_args=model_args,
        selected_samples=selected_samples,
        num_trials=1,
        sampling_method="clean_identity",
        sampling_steps=sampling_steps,
        motion_scorer=motion_scorer,
        samples=baseline_samples,
        failures=failures,
    )
    baseline_report["baseline_type"] = "clean_motion_identity"
    return baseline_report


def build_baseline_comparison(
    sampled_report: dict[str, object],
    clean_report: dict[str, object],
) -> dict[str, object]:
    sampled_metrics = sampled_report["result"]["aggregate"].get("metrics", {})
    clean_metrics = clean_report["result"]["aggregate"].get("metrics", {})
    sampled_quality_mean = float(sampled_metrics.get("quality_score", {}).get("mean", 0.0))
    clean_quality_mean = float(clean_metrics.get("quality_score", {}).get("mean", 0.0))
    sampled_recognizability_mean = float(sampled_metrics.get("recognizability_score", {}).get("mean", 0.0))
    clean_recognizability_mean = float(clean_metrics.get("recognizability_score", {}).get("mean", 0.0))
    sampled_density_mean = float(sampled_metrics.get("density_score", {}).get("mean", 0.0))
    clean_density_mean = float(clean_metrics.get("density_score", {}).get("mean", 0.0))
    quality_score_gap = clean_quality_mean - sampled_quality_mean
    recognizability_score_gap = clean_recognizability_mean - sampled_recognizability_mean
    density_score_gap = clean_density_mean - sampled_density_mean
    return {
        "quality_score_gap": quality_score_gap,
        "recognizability_score_gap": recognizability_score_gap,
        "density_score_gap": density_score_gap,
        "objective_separation_passed": bool(quality_score_gap >= 0.10),
        "verdict": "separates_clean_from_sampled" if quality_score_gap >= 0.10 else "needs_better_separation",
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
        action_tags=getattr(model_args, "action_tags", ""),
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
                "score_object_cond": {
                    "mean": cond_dict[object_type]["mean"].astype(np.float32),
                    "std": cond_dict[object_type]["std"].astype(np.float32),
                },
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
    motion_scorer: MotionQualityScorer,
    device: torch.device,
    output_dir: Path,
) -> dict[str, object]:
    checkpoint_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    eval_model, eval_diffusion = build_eval_model_and_diffusion(model_args, checkpoint_state, args, device)

    motion_score_samples = []
    failures = []

    print(f"[PROGRESS] Starting sampling evaluation with {args.num_trials} trials and {len(selected_samples)} samples...")

    for trial_index in range(args.num_trials):
        trial_seed = args.base_seed + trial_index
        fixseed(trial_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(trial_seed)

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

                sample_path = output_dir / "stage1_sampling_eval" / "trials" / f"trial_{trial_index:02d}" / f"sample_{global_index:03d}_{object_type}" / "generated_prediction.npy"
                try:
                    score_result = score_motion_array(
                        scorer=motion_scorer,
                        motion_np=evaluation["generated_denorm"].astype(np.float32),
                        object_type=object_type,
                        object_cond=sample["score_object_cond"],
                    )
                    motion_score_samples.append(
                        build_motion_score_sample_report(
                            score_result=score_result,
                            sample=sample,
                            sample_index=global_index,
                            trial_index=trial_index,
                            trial_seed=trial_seed,
                            path=str(sample_path),
                            source_mode="stage1_generated_denorm",
                        )
                    )
                except Exception as exc:
                    failures.append({"path": str(sample_path), "error": str(exc)})

            print(f"\r[PROGRESS] Trial {trial_index:02d} - Batch {batch_start//args.eval_batch_size + 1}/{total_batches} ... Done")
        print(f"[PROGRESS] Trial {trial_index:02d} complete. Aggregated metrics computed.")

    sampling_steps = int(args.sampling_steps) if args.sampling_steps > 0 else int(model_args.diffusion_steps)
    sampling_report = build_eval_report(
        args=args,
        model_args=model_args,
        selected_samples=selected_samples,
        num_trials=args.num_trials,
        sampling_method=args.sampling_method,
        sampling_steps=sampling_steps,
        motion_scorer=motion_scorer,
        samples=motion_score_samples,
        failures=failures,
    )
    return sampling_report


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_legacy_json_outputs(output_dir)

    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    device = dist_util.dev()

    model_args = load_model_args(args)

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

    motion_scorer = MotionQualityScorer(args.motion_score_checkpoint_dir, device=str(device))

    print(f"[PROGRESS] Collecting {args.num_eval_samples} evaluation samples from dataset...")
    selected_samples = collect_eval_samples(args, model_args)
    print(f"[PROGRESS] Collected {len(selected_samples)} samples. Starting sampling evaluation...")
    
    sampling_report = stage1_sampling_eval(
        args=args,
        model_args=model_args,
        model=model,
        diffusion=diffusion,
        selected_samples=selected_samples,
        motion_scorer=motion_scorer,
        device=device,
        output_dir=output_dir,
    )
    print(f"[PROGRESS] Sampling evaluation complete. Building clean motion baseline...")
    clean_motion_report = build_clean_motion_baseline(args, model_args, selected_samples, motion_scorer, output_dir)
    baseline_comparison = build_baseline_comparison(sampling_report, clean_motion_report)

    selected_sample_manifest = build_selected_sample_manifest(selected_samples)
    run_info = {
        "model_path": str(Path(args.model_path).resolve()),
        "motion_score_checkpoint": str(motion_scorer.checkpoint_path),
        "output_dir": str(output_dir),
        "split": args.eval_split,
        "objects_subset": model_args.objects_subset,
        "num_frames": int(model_args.num_frames),
        "dataset_max_motion_length": int(getattr(model_args, "dataset_max_motion_length", model_args.num_frames)),
        "selected_subset_max_motion_length": int(getattr(model_args, "selected_subset_max_motion_length", model_args.num_frames)),
        "effective_num_frames": int(getattr(model_args, "effective_num_frames", model_args.num_frames)),
        "selected_sample_count": len(selected_sample_manifest),
        "num_trials": int(args.num_trials),
        "sampling_method": args.sampling_method,
        "sampling_steps": int(sampling_report["sampling_steps"]),
        "selection_seed": int(args.selection_seed),
        "base_seed": int(args.base_seed),
        "export_samples": int(args.export_samples),
        "disable_reference_branch": bool(model_args.disable_reference_branch),
        "stage1_checkpoint_validated": True,
        "stage1_semantics": {
            "use_reference_conditioning": bool(model_args.use_reference_conditioning),
            "lambda_confidence_recon": float(model_args.lambda_confidence_recon),
            "lambda_repair_recon": float(model_args.lambda_repair_recon),
            "cond_mask_prob": float(getattr(model_args, "cond_mask_prob", 0.0)),
        },
    }

    summary_report = {
        "run": {
            "model_path": run_info["model_path"],
            "motion_score_checkpoint": run_info["motion_score_checkpoint"],
            "split": run_info["split"],
            "objects_subset": run_info["objects_subset"],
            "selected_sample_count": run_info["selected_sample_count"],
            "num_trials": run_info["num_trials"],
            "sampling_method": run_info["sampling_method"],
            "sampling_steps": run_info["sampling_steps"],
        },
        "sampled_eval": build_summary_eval_section(sampling_report["result"]),
        "clean_baseline": build_summary_eval_section(clean_motion_report["result"]),
        "comparison": baseline_comparison,
    }

    detail_report = {
        "run": run_info,
        "selected_samples": selected_sample_manifest,
        "sampled_eval": sampling_report["result"],
        "clean_baseline": clean_motion_report["result"],
        "comparison": baseline_comparison,
    }

    write_json(output_dir / "summary.json", summary_report)
    write_json(output_dir / "detail.json", detail_report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())