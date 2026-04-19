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

from motion_lib import BVH
from data_loaders.get_data import get_dataset_loader
from data_loaders.tensors import truebones_batch_collate
from data_loaders.truebones.truebones_utils.motion_process import recover_animation_from_motion_np
from utils.fixseed import fixseed
from utils import dist_util
from utils.model_util import create_model_and_diffusion_general_skeleton, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage1 pretrain sampling debug tool with stochastic aggregate reports.")
    parser.add_argument("--model-path", required=True, help="Path to a stage1 model checkpoint.")
    parser.add_argument("--output-dir", required=True, help="Directory to write reports and exported samples.")
    parser.add_argument("--device", default=0, type=int, help="CUDA device id. Use -1 for CPU.")
    parser.add_argument("--seed", default=10, type=int, help="Global seed for deterministic setup.")
    parser.add_argument("--objects-subset", default="", help="Override the checkpoint objects_subset when set.")
    parser.add_argument("--action-tags", default="", help="Override the checkpoint action_tags when set, e.g. 'locomotion,attack'.")
    parser.add_argument("--fixed-motion", default="", help="Override the checkpoint fixed_motion when set. Accepts a processed .npy name or a BVH/.npy path.")
    parser.add_argument("--fixed-window-start", default=None, type=int, help="Override the checkpoint fixed_window_start when set.")
    parser.add_argument("--max-frames", default=-1, type=int, help="Override num_frames when > 0.")
    parser.add_argument(
        "--sample-mode",
        default="eval_subset",
        choices=["eval_subset"],
        help="eval_subset samples different dataset windows.",
    )
    parser.add_argument("--num-eval-samples", default=16, type=int, help="Number of unique samples to evaluate. Each sample runs as an independent batch-1 inference.")
    parser.add_argument("--num-threads", default=4, type=int, help="Number of parallel worker threads used to run sample inferences concurrently.")
    parser.add_argument("--eval-num-workers", default=0, type=int, help="Evaluation DataLoader workers.")
    parser.add_argument("--selection-seed", default=None, type=int, help="Seed used to select the fixed evaluation subset. Defaults to --seed.")
    parser.add_argument("--base-seed", default=None, type=int, help="Base seed for per-sample seeds. Defaults to --seed. Sample k uses base-seed + k.")
    parser.add_argument("--sampling-method", default="ddim", choices=["p", "ddim", "plms"], help="Diffusion sampler to use.")
    parser.add_argument("--sampling-steps", default=0, type=int, help="Respaced diffusion steps. 0 keeps the checkpoint diffusion step count.")
    parser.add_argument("--ddim-eta", default=0.0, type=float, help="DDIM eta parameter.")
    parser.add_argument("--no-ema", action="store_true", help="Disable EMA model averaging and use raw model weights instead.")
    parser.add_argument("--action_category", default="", help="Action category to condition generation on, e.g. 'locomotion'. "
                        "Must be one of: attack, death, emote, fall, jump, locomotion, other, pose, posture, reaction, rise, turn. "
                        "Overrides action_tags in each sample's condition. Only effective when the model was trained with --use_action_cond.")
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

    model_args.action_tags = getattr(model_args, "action_tags", "")
    model_args.action_category = str(getattr(args, "action_category", "") or "").strip().lower()

    model_args.model_path = str(model_path)
    model_args.device = args.device
    model_args.batch_size = 1
    model_args.cond_mask_prob = 0.0
    if args.objects_subset:
        model_args.objects_subset = args.objects_subset
    if args.action_tags:
        model_args.action_tags = args.action_tags
    if args.fixed_motion:
        model_args.fixed_motion = args.fixed_motion
    else:
        model_args.fixed_motion = getattr(model_args, "fixed_motion", "")
    if args.fixed_window_start is not None:
        model_args.fixed_window_start = int(args.fixed_window_start)
    else:
        model_args.fixed_window_start = int(getattr(model_args, "fixed_window_start", 0))
    if args.max_frames > 0:
        model_args.num_frames = args.max_frames
    model_args.sample_limit = -1
    model_args.num_workers = args.eval_num_workers
    return model_args


def resolve_eval_subset_action_tags(model_args: SimpleNamespace) -> str:
    action_category = str(getattr(model_args, "action_category", "") or "").strip().lower()
    if action_category:
        return action_category
    return str(getattr(model_args, "action_tags", "") or "").strip()


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


def cleanup_stage1_sampling_eval_directory(output_dir: Path) -> None:
    import shutil
    stage1_eval_dir = output_dir / "stage1_sampling_eval"
    if stage1_eval_dir.exists():
        shutil.rmtree(stage1_eval_dir)


def build_selected_sample_manifest(selected_samples: list[dict[str, object]]) -> list[dict[str, object]]:
    manifest = []
    for sample in selected_samples:
        target_length = int(sample.get("target_length", sample["length"]))
        record = {
            "sample_index": int(sample["sample_index"]),
            "motion_name": str(sample["motion_name"]),
            "object_type": str(sample["object_type"]),
            "length": target_length,
            "target_length": target_length,
            "n_joints": int(sample["n_joints"]),
        }
        if "source_mode" in sample:
            record["source_mode"] = str(sample["source_mode"])
        if "reference_sample_index" in sample:
            record["reference_sample_index"] = int(sample["reference_sample_index"])
        manifest.append(record)
    return manifest


def build_export_sample_record(
    *,
    sample: dict[str, object],
    sample_index: int,
    sample_seed: int,
    sample_dir: Path,
) -> dict[str, object]:
    target_length = int(sample.get("target_length", sample["length"]))
    return {
        "sample_index": int(sample_index),
        "sample_seed": int(sample_seed),
        "motion_name": str(sample["motion_name"]),
        "object_type": str(sample["object_type"]),
        "sample_dir": str(sample_dir),
        "generated_path": str(sample_dir / "generated_prediction.npy"),
        "target_path": str(sample_dir / "clean_target.npy"),
        "length": target_length,
        "target_length": target_length,
        "n_joints": int(sample["n_joints"]),
    }


def build_export_result(
    *,
    args: argparse.Namespace,
    model_args: SimpleNamespace,
    selected_samples: list[dict[str, object]],
    samples: list[dict[str, object]],
    failures: list[dict[str, str]],
    num_threads: int,
) -> dict[str, object]:
    sampling_steps = int(args.sampling_steps) if args.sampling_steps > 0 else int(model_args.diffusion_steps)
    return {
        "split": "all",
        "objects_subset": model_args.objects_subset,
        "sample_mode": args.sample_mode,
        "selected_sample_count": len(selected_samples),
        "num_threads": int(num_threads),
        "sampling_method": args.sampling_method,
        "sampling_steps": sampling_steps,
        "exported_samples": sorted(samples, key=lambda sample: sample["sample_index"]),
        "failures": failures,
    }


def build_summary_export_section(export_result: dict[str, object]) -> dict[str, object]:
    exported_samples = export_result["exported_samples"]
    return {
        "selected_sample_count": int(export_result["selected_sample_count"]),
        "num_threads": int(export_result["num_threads"]),
        "exported_samples": len(exported_samples),
        "failed_exports": len(export_result["failures"]),
        "sampling_method": str(export_result["sampling_method"]),
        "sampling_steps": int(export_result["sampling_steps"]),
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


def build_virtual_eval_sample(
    motion_dataset,
    cond_dict: dict[str, dict[str, object]],
    motion_name: str,
    window_start: int,
    num_frames: int,
) -> dict[str, object]:
    raw_data = motion_dataset.data_dict[motion_name]
    previous_fixed_motion_name = motion_dataset.fixed_motion_name
    previous_fixed_window_start = motion_dataset.fixed_window_start
    previous_max_motion_length = motion_dataset.max_motion_length
    try:
        motion_dataset.fixed_motion_name = motion_name
        motion_dataset.fixed_window_start = int(window_start)
        motion_dataset.max_motion_length = num_frames
        prepared_sample = motion_dataset._prepare_sample(motion_name, raw_data)
    finally:
        motion_dataset.fixed_motion_name = previous_fixed_motion_name
        motion_dataset.fixed_window_start = previous_fixed_window_start
        motion_dataset.max_motion_length = previous_max_motion_length

    motion_batch, cond_batch = truebones_batch_collate([prepared_sample])
    cond_cpu = clone_batch_cond(cond_batch)
    object_type = cond_cpu["y"]["object_type"][0]
    n_joints = int(cond_cpu["y"]["n_joints"][0].item())
    target_length = int(cond_cpu["y"]["lengths"][0].item())

    return {
        "sample_index": -1,
        "motion": motion_batch.detach().clone().float(),
        "cond": cond_cpu,
        "motion_name": cond_cpu["y"]["motion_name"][0],
        "object_type": object_type,
        "n_joints": n_joints,
        "length": target_length,
        "target_length": target_length,
        "parents": [int(parent) for parent in prepared_sample[2]],
        "offsets": cond_dict[object_type]["offsets"],
        "joints_names": cond_dict[object_type]["joints_names"],
        "mean": prepared_sample[11].astype(np.float32),
        "std": prepared_sample[12].astype(np.float32) + 1e-6,
    }


def clone_eval_sample(sample: dict[str, object]) -> dict[str, object]:
    cloned: dict[str, object] = {}
    for key, value in sample.items():
        if key == "motion":
            cloned[key] = value.detach().clone()
        elif key == "cond":
            cloned[key] = clone_batch_cond(value)
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def collect_eval_samples(args: argparse.Namespace, model_args: SimpleNamespace, verbose: bool = True) -> list[dict[str, object]]:
    """
    Collect evaluation samples with virtual sample expansion.
    Each sample is randomly selected from motions and cropped at a random window.
    """
    import random as py_random

    selection_seed = args.selection_seed if args.selection_seed is not None else args.seed
    fixseed(selection_seed)
    rng = py_random.Random(selection_seed)

    # Load all available motions from the dataset
    eval_subset_action_tags = resolve_eval_subset_action_tags(model_args)
    model_args.eval_subset_action_tags = eval_subset_action_tags

    loader_kwargs = dict(
        batch_size=1,
        num_frames=model_args.num_frames,
        split="all",
        temporal_window=model_args.temporal_window,
        t5_name=model_args.t5_name,
        balanced=False,
        objects_subset=model_args.objects_subset,
        num_workers=0,
        sample_limit=0,  # Load ALL motions
        shuffle=False,
        drop_last=False,
        action_tags=eval_subset_action_tags,
        fixed_motion=getattr(model_args, "fixed_motion", ""),
        fixed_window_start=int(getattr(model_args, "fixed_window_start", 0)),
    )
    data_full = get_dataset_loader(**loader_kwargs)
    motion_dataset = data_full.dataset.motion_dataset

    # Get dataset info
    dataset_frame_cap = int(getattr(motion_dataset, "max_motion_length", model_args.num_frames))
    selected_frame_cap = int(getattr(motion_dataset, "max_available_length", dataset_frame_cap))
    args_num_frames = int(model_args.num_frames)
    model_args.dataset_max_motion_length = dataset_frame_cap
    model_args.selected_subset_max_motion_length = selected_frame_cap
    model_args.effective_num_frames = args_num_frames

    cond_dict = motion_dataset.cond_dict

    eligible_motions = []
    for motion_name, motion_length in zip(motion_dataset.name_list, motion_dataset.length_arr):
        motion_length = int(motion_length)
        eligible_motions.append((motion_name, motion_length))

    if not eligible_motions:
        raise RuntimeError("No suitable motions found for virtual sampling.")

    samples = []
    while len(samples) < args.num_eval_samples:
        motion_name, motion_length = rng.choice(eligible_motions)
        # Per-sample num_frames: use min(args_num_frames, motion_length); window randomly if longer
        if motion_length >= args_num_frames:
            sample_num_frames = args_num_frames
            window_start = rng.randint(0, motion_length - args_num_frames)
        else:
            sample_num_frames = motion_length
            window_start = 0
        sample = build_virtual_eval_sample(
            motion_dataset=motion_dataset,
            cond_dict=cond_dict,
            motion_name=motion_name,
            window_start=window_start,
            num_frames=sample_num_frames,
        )
        sample["sample_index"] = len(samples)
        samples.append(sample)

    if not samples:
        raise RuntimeError("No evaluation samples were collected.")

    if verbose:
        print(f"[PROGRESS] Collected {len(samples)} virtual samples")
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

    exported_samples: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    total = len(selected_samples)
    actual_num_threads = 1

    if int(args.num_threads) != actual_num_threads:
        print(
            "[WARN] stage1_sampling_eval runs serially to keep per-sample RNG isolated. "
            f"Requested num_threads={int(args.num_threads)} will be ignored."
        )

    print(
        f"[PROGRESS] Starting sampling evaluation: {total} samples, "
        f"{actual_num_threads} thread(s) ..."
    )

    def process_sample(sample_index: int) -> None:
        sample = selected_samples[sample_index]
        sample_seed = args.base_seed + sample_index
        fixseed(sample_seed)

        motion_cpu, cond_cpu = combine_batch_samples([sample])
        # Override action condition if --action_category was specified.
        action_category = str(getattr(model_args, "action_category", "") or "").strip().lower()
        if action_category:
            cond_cpu["y"]["action_tags"] = [[action_category]]
            cond_cpu["y"]["action_category"] = [action_category]
        # Override tpos_first_frame with the actual first frame of the clean target motion.
        cond_cpu["y"]["tpos_first_frame"] = motion_cpu[0, :, :, 0].unsqueeze(0).clone()
        motion = motion_cpu.to(device, non_blocking=device.type == "cuda")
        cond = move_cond_to_device(cond_cpu, device)

        with torch.inference_mode():
            generated = sample_motion_batch(
                eval_diffusion,
                eval_model,
                motion.shape,
                cond,
                args.sampling_method,
                args.ddim_eta,
            )

        n_joints = int(sample["n_joints"])
        target_length = int(sample.get("target_length", sample["length"]))
        object_type = str(sample["object_type"])

        target_norm = motion_cpu[0, :n_joints, :, :target_length]
        generated_norm = generated[0, :n_joints, :, :target_length].detach().cpu()

        evaluation = evaluate_generated_prediction(
            target_norm=target_norm,
            generated_norm=generated_norm,
            n_joints=n_joints,
            mean=sample["mean"],
            std=sample["std"],
        )

        sample_dir = output_dir / "stage1_sampling_eval" / f"sample_{sample_index:03d}_{object_type}"
        try:
            sample_dir.mkdir(parents=True, exist_ok=True)
            export_trial_sample(
                sample_dir=sample_dir,
                parents=sample["parents"],
                offsets=sample["offsets"],
                joints_names=sample["joints_names"],
                target_motion=evaluation["target_denorm"].astype(np.float32),
                generated_motion=evaluation["generated_denorm"].astype(np.float32),
            )
            record = build_export_sample_record(
                sample=sample,
                sample_index=sample_index,
                sample_seed=sample_seed,
                sample_dir=sample_dir,
            )
            exported_samples.append(record)
        except Exception as exc:
            failures.append({"path": str(sample_dir), "error": str(exc)})

        print(f"[PROGRESS] Sample {sample_index:03d}/{total - 1:03d} done (seed={sample_seed})")

    for sample_index in range(total):
        process_sample(sample_index)

    return build_export_result(
        args=args,
        model_args=model_args,
        selected_samples=selected_samples,
        samples=exported_samples,
        failures=failures,
        num_threads=actual_num_threads,
    )


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_legacy_json_outputs(output_dir)
    cleanup_stage1_sampling_eval_directory(output_dir)

    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    device = dist_util.dev()

    # 派生种子：如果用户未显式设置，则使用 --seed 的值
    if args.selection_seed is None:
        args.selection_seed = args.seed
    if args.base_seed is None:
        args.base_seed = args.seed

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

    print(f"[PROGRESS] Collecting {args.num_eval_samples} samples using mode={args.sample_mode} ...")
    selected_samples = collect_eval_samples(args, model_args)
    print(f"[PROGRESS] Collected {len(selected_samples)} samples. Starting sampling evaluation...")
    
    export_result = stage1_sampling_eval(
        args=args,
        model_args=model_args,
        model=model,
        diffusion=diffusion,
        selected_samples=selected_samples,
        device=device,
        output_dir=output_dir,
    )

    selected_sample_manifest = build_selected_sample_manifest(selected_samples)
    run_info = {
        "model_path": str(Path(args.model_path).resolve()),
        "output_dir": str(output_dir),
        "split": "all",
        "objects_subset": model_args.objects_subset,
        "sample_mode": args.sample_mode,
        "num_frames": int(model_args.num_frames),
        "dataset_max_motion_length": int(getattr(model_args, "dataset_max_motion_length", model_args.num_frames)),
        "selected_subset_max_motion_length": int(getattr(model_args, "selected_subset_max_motion_length", model_args.num_frames)),
        "effective_num_frames": int(getattr(model_args, "effective_num_frames", model_args.num_frames)),
        "selected_sample_count": len(selected_sample_manifest),
        "num_threads": int(args.num_threads),
        "sampling_method": args.sampling_method,
        "sampling_steps": int(export_result["sampling_steps"]),
        "selection_seed": int(args.selection_seed),
        "base_seed": int(args.base_seed),
        "num_eval_samples": int(args.num_eval_samples),
        "fixed_motion": str(getattr(model_args, "fixed_motion", "")),
        "fixed_window_start": int(getattr(model_args, "fixed_window_start", 0)),
        "dual_length_export": True,
        "eval_subset_action_tags": str(getattr(model_args, "eval_subset_action_tags", getattr(model_args, "action_tags", ""))),
        "action_category": str(getattr(model_args, "action_category", "")),
        "stage1_checkpoint_validated": True,
        "stage1_semantics": {
            "cond_mask_prob": float(getattr(model_args, "cond_mask_prob", 0.0)),
        },
    }

    summary_report = {
        "run": {
            "model_path": run_info["model_path"],
            "split": run_info["split"],
            "objects_subset": run_info["objects_subset"],
            "sample_mode": run_info["sample_mode"],
            "selected_sample_count": run_info["selected_sample_count"],
            "num_threads": run_info["num_threads"],
            "sampling_method": run_info["sampling_method"],
            "sampling_steps": run_info["sampling_steps"],
        },
        "exports": build_summary_export_section(export_result),
    }

    detail_report = {
        "run": run_info,
        "selected_samples": selected_sample_manifest,
        "exports": export_result,
    }

    write_json(output_dir / "summary.json", summary_report)
    write_json(output_dir / "detail.json", detail_report)

    print(
        "[SUMMARY] Exported "
        f"{summary_report['exports']['exported_samples']} samples "
        f"({summary_report['exports']['failed_exports']} failures)."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())