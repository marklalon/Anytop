from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
from collections import OrderedDict
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.get_data import get_dataset_loader
from data_loaders.skeleton_metadata import (
    LabelVocab,
    build_label_vocabs,
)
from diffusion.fp16_util import MixedPrecisionTrainer, format_nonfinite_stats, inspect_optimizer_state, sanitize_optimizer_state
from diffusion.nn import update_ema
from model.motion_autoencoder import MotionScorerNet
from utils import dist_util
from utils.fixseed import fixseed
from utils.ml_platforms import ClearmlPlatform, NoPlatform, TensorboardPlatform, WandBPlatform
from utils.parser_util import add_base_options, add_data_options, add_training_options


def build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_training_options(parser)

    group = parser.add_argument_group("motion_scorer")
    group.add_argument("--feature_dim", default=13, type=int, help="Input feature size per joint.")
    group.add_argument("--d_model", default=128, type=int, help="Hidden size for the motion scorer backbone.")
    group.add_argument("--latent_dim", default=128, type=int, help="Latent size of the scorer bottleneck.")
    group.add_argument("--num_conv_layers", default=3, type=int, help="Number of temporal residual conv blocks.")
    group.add_argument("--kernel_size", default=5, type=int, help="Kernel size for temporal conv blocks.")
    group.add_argument("--max_joints", default=143, type=int, help="Maximum padded joint count.")
    group.add_argument("--train_split", default="train", choices=["train", "all", "val", "test"], type=str,
                       help="Dataset split used for training.")
    group.add_argument("--stats_split", default="", type=str,
                       help="Split used to cache scorer statistics after training. Empty means train_split.")
    group.add_argument("--lr_step_size", default=10000, type=int, help="StepLR step size in optimizer steps.")
    group.add_argument("--lr_gamma", default=0.99, type=float, help="StepLR gamma.")
    group.add_argument("--ema_decay", default=0.999, type=float, help="EMA decay when --use_ema is enabled.")
    group.add_argument("--timing_log_interval", default=1000, type=int,
                       help="Report averaged timing breakdown every N training steps.")
    group.add_argument("--load_optimizer_state", action="store_true",
                       help="Restore optimizer and scaler state when resuming.")
    group.add_argument("--lambda_species", default=1.0, type=float, help="Weight for species classification CE.")
    group.add_argument("--lambda_action", default=1.0, type=float, help="Weight for action classification CE.")
    group.add_argument("--score_alpha", nargs=1, default=[1.0], type=float,
                       help="Geometric-mean weights for recognizability score aggregation.")
    return parser


def _fp32_loss_context(device: torch.device):
    return torch.autocast(device_type=device.type, enabled=False)


def prepare_save_dir(args) -> str:
    save_dir = args.save_dir
    if not save_dir:
        save_root = os.path.join(os.getcwd(), "save")
        os.makedirs(save_root, exist_ok=True)
        prefix = getattr(args, "model_prefix", None) or "MotionScorerV3"
        model_name = f"{prefix}_dataset_truebones_bs_{args.batch_size}_latentdim_{args.latent_dim}"
        save_dir = os.path.join(save_root, model_name)
        args.save_dir = save_dir

    os.makedirs(save_dir, exist_ok=True)

    if args.auto_resume:
        if not args.resume_checkpoint:
            latest_checkpoint = find_latest_checkpoint(save_dir, prefix="model")
            if not latest_checkpoint:
                print(f"[INFO] auto_resume was requested but no checkpoint was found in save_dir [{save_dir}]. Starting fresh training.")
                args.resume_checkpoint = ""
                clear_motion_scorer_artifacts(save_dir)
            else:
                args.resume_checkpoint = latest_checkpoint
                if not getattr(args, "load_optimizer_state", False):
                    args.load_optimizer_state = True
                print(f"[INFO] Auto-resuming motion scorer from {args.resume_checkpoint}")
        else:
            if not getattr(args, "load_optimizer_state", False):
                args.load_optimizer_state = True
            print(f"[INFO] Auto-resuming motion scorer from {args.resume_checkpoint}")
    elif not args.resume_checkpoint:
        args.resume_checkpoint = ""
        clear_motion_scorer_artifacts(save_dir)
    return save_dir


def clear_motion_scorer_artifacts(save_dir: str) -> None:
    if not os.path.isdir(save_dir):
        return
    for file_name in os.listdir(save_dir):
        file_path = os.path.join(save_dir, file_name)
        if re.fullmatch(r"model\d+\.pt", file_name) or re.fullmatch(r"opt\d+\.pt", file_name):
            os.remove(file_path)
            continue
        if file_name in {"args.json", "train_stats.npy", "train_stats_summary.json", "debug_score_report.json", "sanity_checks.json"}:
            os.remove(file_path)
            continue
        if file_name.startswith("model") and file_name.endswith(".pt.samples") and os.path.isdir(file_path):
            shutil.rmtree(file_path)


def create_data_loader(args, split: str, *, shuffle: bool, drop_last: bool, balanced: bool):
    return get_dataset_loader(
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        split=split,
        temporal_window=getattr(args, "temporal_window", 31),
        t5_name="t5-base",
        balanced=balanced,
        objects_subset=args.objects_subset,
        num_workers=args.num_workers,
        prefetch_factor=getattr(args, "prefetch_factor", 2),
        sample_limit=args.sample_limit,
        shuffle=shuffle,
        drop_last=drop_last,
        use_reference_conditioning=False,
        action_tags=getattr(args, "action_tags", ""),
        motion_cache_size=getattr(args, "motion_cache_size", 0),
        main_process_prefetch_batches=getattr(args, "main_process_prefetch_batches", 0),
        fixed_motion=getattr(args, 'fixed_motion', ''),
        fixed_window_start=getattr(args, 'fixed_window_start', 0),
    )


def find_latest_checkpoint(save_dir: str, prefix: str = "model") -> str:
    if not save_dir or not os.path.isdir(save_dir):
        return ""
    candidates = []
    for file_name in os.listdir(save_dir):
        match = re.fullmatch(rf"{re.escape(prefix)}(\d+)\.pt", file_name)
        if match:
            candidates.append((int(match.group(1)), os.path.join(save_dir, file_name)))
    if not candidates:
        return ""
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def parse_checkpoint_number(checkpoint_path: str) -> int:
    match = re.search(r"(\d+)\.pt$", checkpoint_path)
    if match is None:
        raise ValueError(f"Could not parse step number from checkpoint path: {checkpoint_path}")
    return int(match.group(1))


def select_model_state_dict(state_dict: dict, prefer_ema: bool) -> dict:
    if isinstance(state_dict, dict):
        if prefer_ema and "model_avg" in state_dict:
            return state_dict["model_avg"]
        if "model" in state_dict:
            return state_dict["model"]
    return state_dict


def move_cond_to_device(cond, device: torch.device, non_blocking: bool) -> dict:
    return {
        "y": {
            key: value.to(device, non_blocking=non_blocking) if torch.is_tensor(value) else value
            for key, value in cond["y"].items()
        }
    }


def apply_current_optimizer_hparams(opt: AdamW, args) -> None:
    target_lr = float(args.lr)
    target_weight_decay = float(args.weight_decay)
    for param_group in opt.param_groups:
        param_group["lr"] = target_lr
        param_group["initial_lr"] = target_lr
        param_group["weight_decay"] = target_weight_decay


def build_step_lr_scheduler(opt: AdamW, args, completed_steps: int) -> torch.optim.lr_scheduler.StepLR:
    scheduler = torch.optim.lr_scheduler.StepLR(
        opt,
        step_size=max(1, int(args.lr_step_size)),
        gamma=float(args.lr_gamma),
    )
    if completed_steps > 0:
        decay_factor = float(args.lr_gamma) ** (completed_steps // max(1, int(args.lr_step_size)))
        resumed_lr = float(args.lr) * decay_factor
        for param_group in opt.param_groups:
            param_group["lr"] = resumed_lr
        scheduler.last_epoch = completed_steps
        scheduler._last_lr = [param_group["lr"] for param_group in opt.param_groups]
    return scheduler


class MotionScorerTrainer:
    def __init__(
        self,
        args,
        ml_platform,
        data_loader,
        species_vocab: LabelVocab,
        action_vocab: LabelVocab,
    ) -> None:
        self.args = args
        self.ml_platform = ml_platform
        self.data_loader = data_loader
        self.species_vocab = species_vocab
        self.action_vocab = action_vocab
        self.num_species = args.num_species
        self.num_actions = args.num_actions

        dist_util.setup_dist(args.device)
        self.device = dist_util.dev()
        self.non_blocking = self.device.type == "cuda"
        self.amp_dtype = getattr(args, "amp_dtype", "fp32").lower()
        self.amp_enabled = self.amp_dtype in {"fp16", "bf16"}
        if self.amp_enabled and self.device.type != "cuda":
            raise ValueError("AMP requires CUDA. Set --amp_dtype fp32 when training on CPU.")
        self.autocast_dtype = None
        if self.amp_dtype == "fp16":
            self.autocast_dtype = torch.float16
        elif self.amp_dtype == "bf16":
            self.autocast_dtype = torch.bfloat16

        self.model = MotionScorerNet(
            feature_dim=args.feature_dim,
            d_model=args.d_model,
            latent_dim=args.latent_dim,
            num_conv_layers=args.num_conv_layers,
            kernel_size=args.kernel_size,
            max_joints=args.max_joints,
            num_species=self.num_species,
            num_actions=self.num_actions,
        ).to(self.device)
        self.model_avg = copy.deepcopy(self.model) if args.use_ema else None
        self.resume_checkpoint = args.resume_checkpoint.strip() if args.resume_checkpoint else ""
        self.resume_completed_steps = 0

        if self.resume_checkpoint:
            payload = torch.load(self.resume_checkpoint, map_location="cpu", weights_only=False)
            model_state = select_model_state_dict(payload, prefer_ema=False)
            self.model.load_state_dict(model_state, strict=True)
            if self.model_avg is not None:
                avg_state = payload.get("model_avg", model_state)
                self.model_avg.load_state_dict(avg_state, strict=True)
            self.resume_completed_steps = parse_checkpoint_number(self.resume_checkpoint)

        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=False,
            amp_dtype=self.amp_dtype,
            amp_enabled=self.amp_enabled,
            device_type=self.device.type,
            log_norms=False,
        )
        self.opt = AdamW(self.mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)

        if self.resume_checkpoint and args.load_optimizer_state:
            opt_path = os.path.join(os.path.dirname(self.resume_checkpoint), f"opt{self.resume_completed_steps:09d}.pt")
            if os.path.exists(opt_path):
                opt_state = torch.load(opt_path, map_location="cpu", weights_only=False)
                if self.amp_enabled and isinstance(opt_state, dict) and "opt" in opt_state:
                    if "scaler" in opt_state and self.mp_trainer.scaler.is_enabled():
                        self.mp_trainer.scaler.load_state_dict(opt_state["scaler"])
                    opt_state = opt_state["opt"]
                self.opt.load_state_dict(opt_state)
                optimizer_state_stats = sanitize_optimizer_state(self.opt)
                if optimizer_state_stats["found"]:
                    print(
                        "Sanitized non-finite optimizer state after restore "
                        f"({format_nonfinite_stats(optimizer_state_stats)})"
                    )

        apply_current_optimizer_hparams(self.opt, args)
        self.lr_scheduler = build_step_lr_scheduler(self.opt, args, self.resume_completed_steps)
        self.model.train()

    def autocast_context(self):
        if not self.amp_enabled:
            return torch.autocast(device_type=self.device.type, enabled=False)
        return torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)

    def _encode_batch_labels(self, cond: dict) -> tuple[torch.Tensor, torch.Tensor]:
        species_ids = self.species_vocab.encode_many(cond["y"].get("species_label", []), device=self.device)
        action_ids = self.action_vocab.encode_many(cond["y"].get("action_label", []), device=self.device)
        return species_ids, action_ids

    def train_step(self, motion: torch.Tensor, cond: dict, current_step: int) -> dict[str, float]:
        del current_step
        n_joints = cond["y"]["n_joints"]
        lengths = cond["y"]["lengths"]
        species_ids, action_ids = self._encode_batch_labels(cond)

        self.mp_trainer.zero_grad()
        with self.autocast_context():
            clean_outputs = self.model(
                motion,
                n_joints,
                lengths,
            )

        with _fp32_loss_context(self.device):
            species_logits = clean_outputs["species_logits"].float()
            action_logits = clean_outputs["action_logits"].float()
            species_loss = F.cross_entropy(species_logits, species_ids)
            action_loss = F.cross_entropy(action_logits, action_ids)

            loss = (
                float(self.args.lambda_species) * species_loss
                + float(self.args.lambda_action) * action_loss
            )

        self.mp_trainer.backward(loss)
        took_step = self.mp_trainer.optimize(self.opt, self.lr_scheduler)
        if took_step and self.model_avg is not None:
            update_ema(self.model_avg.parameters(), self.model.parameters(), rate=self.args.ema_decay)

        with torch.no_grad():
            species_accuracy = (species_logits.argmax(dim=-1) == species_ids).float().mean()
            action_accuracy = (action_logits.argmax(dim=-1) == action_ids).float().mean()

        return {
            "loss": loss.detach().float(),
            "species_loss": species_loss.detach().float(),
            "action_loss": action_loss.detach().float(),
            "species_accuracy": species_accuracy.detach().float(),
            "action_accuracy": action_accuracy.detach().float(),
        }

    def save(self, completed_step: int) -> None:
        state_dict = self.mp_trainer.master_params_to_state_dict(self.mp_trainer.master_params)
        if self.args.use_ema and self.model_avg is not None:
            state_dict = {"model": state_dict, "model_avg": self.model_avg.state_dict()}
        checkpoint_path = os.path.join(self.args.save_dir, f"model{completed_step:09d}.pt")
        torch.save(state_dict, checkpoint_path)

        opt_state = self.opt.state_dict()
        if self.amp_enabled:
            opt_state = {"opt": opt_state, "scaler": self.mp_trainer.scaler.state_dict()}
        opt_path = os.path.join(self.args.save_dir, f"opt{completed_step:09d}.pt")
        torch.save(opt_state, opt_path)

    def run(self) -> MotionScorerNet:
        completed_steps = self.resume_completed_steps
        running_metrics: dict[str, torch.Tensor] = {}
        running_metric_counts: dict[str, int] = {}
        data_iter = iter(self.data_loader)
        timing_log_interval = max(1, int(getattr(self.args, "timing_log_interval", self.args.log_interval)))
        timing_totals = {
            "data_wait_s": 0.0,
            "host_to_device_s": 0.0,
            "step_s": 0.0,
            "loop_s": 0.0,
        }
        timing_steps = 0

        next_metric_log = min(int(self.args.log_interval), int(self.args.num_steps))
        next_timing_log = min(int(timing_log_interval), int(self.args.num_steps))
        print(
            f"Motion scorer training loop started: next_metrics_step={next_metric_log}, "
            f"next_timing_step={next_timing_log}"
        )

        while completed_steps < self.args.num_steps:
            loop_start = time.perf_counter()
            fetch_start = time.perf_counter()
            if completed_steps == self.resume_completed_steps:
                print("Motion scorer waiting for first batch...")
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.data_loader)
                batch = next(data_iter)
            motion, cond = batch
            data_wait_s = time.perf_counter() - fetch_start

            host_to_device_start = time.perf_counter()
            motion = motion.to(self.device, non_blocking=self.non_blocking)
            cond = move_cond_to_device(cond, self.device, self.non_blocking)
            host_to_device_s = time.perf_counter() - host_to_device_start

            step_start = time.perf_counter()
            step_metrics = self.train_step(motion, cond, completed_steps + 1)
            step_s = time.perf_counter() - step_start
            loop_s = time.perf_counter() - loop_start

            completed_steps += 1
            for metric_name, metric_value in step_metrics.items():
                detached_value = metric_value.detach()
                if metric_name in running_metrics:
                    running_metrics[metric_name] = running_metrics[metric_name] + detached_value
                    running_metric_counts[metric_name] += 1
                else:
                    running_metrics[metric_name] = detached_value.clone()
                    running_metric_counts[metric_name] = 1
            timing_totals["data_wait_s"] += data_wait_s
            timing_totals["host_to_device_s"] += host_to_device_s
            timing_totals["step_s"] += step_s
            timing_totals["loop_s"] += loop_s
            timing_steps += 1

            if completed_steps % self.args.log_interval == 0 or completed_steps == self.args.num_steps:
                self._assert_optimizer_state_finite(completed_steps)
                mean_metrics = {
                    metric_name: float((metric_total / max(running_metric_counts[metric_name], 1)).item())
                    for metric_name, metric_total in running_metrics.items()
                }
                print(
                    "step[{}]: total_loss[{:.6f}] species_ce[{:.6f}] action_ce[{:.6f}] species_acc[{:.4f}] action_acc[{:.4f}]".format(
                        completed_steps,
                        mean_metrics.get("loss", 0.0),
                        mean_metrics.get("species_loss", 0.0),
                        mean_metrics.get("action_loss", 0.0),
                        mean_metrics.get("species_accuracy", 0.0),
                        mean_metrics.get("action_accuracy", 0.0),
                    )
                )
                for metric_name, metric_value in mean_metrics.items():
                    self.ml_platform.report_scalar(metric_name, metric_value, completed_steps, group_name="Train")
                self.ml_platform.report_scalar("lr", self.lr_scheduler.get_last_lr()[0], completed_steps, group_name="Train")
                running_metrics.clear()
                running_metric_counts.clear()

            if completed_steps % timing_log_interval == 0 or completed_steps == self.args.num_steps:
                mean_loop_s = timing_totals["loop_s"] / max(timing_steps, 1)
                mean_data_wait_ms = 1000.0 * timing_totals["data_wait_s"] / max(timing_steps, 1)
                mean_step_ms = 1000.0 * timing_totals["step_s"] / max(timing_steps, 1)
                print(
                    "timing[{}]: data_wait_ms[{:.2f}] step_ms[{:.2f}] total_ms[{:.2f}]".format(
                        completed_steps,
                        mean_data_wait_ms,
                        mean_step_ms,
                        1000.0 * mean_loop_s,
                    )
                )
                self.ml_platform.report_scalar("data_wait_ms", mean_data_wait_ms, completed_steps, group_name="Timing")
                self.ml_platform.report_scalar("step_ms", mean_step_ms, completed_steps, group_name="Timing")
                self.ml_platform.report_scalar("total_ms", 1000.0 * mean_loop_s, completed_steps, group_name="Timing")
                timing_totals = {
                    "data_wait_s": 0.0,
                    "host_to_device_s": 0.0,
                    "step_s": 0.0,
                    "loop_s": 0.0,
                }
                timing_steps = 0

            if completed_steps % self.args.save_interval == 0 or completed_steps == self.args.num_steps:
                self.save(completed_steps)

        return self.model_avg if self.model_avg is not None else self.model

    def _assert_optimizer_state_finite(self, completed_step: int) -> None:
        state_stats = inspect_optimizer_state(self.opt)
        if state_stats["found"]:
            raise RuntimeError(
                "Detected non-finite optimizer state at "
                f"step {completed_step} ({format_nonfinite_stats(state_stats)})"
            )


def compute_and_save_train_stats(args) -> None:
    stats_split = args.stats_split or args.train_split
    explicit_checkpoint = str(getattr(args, "checkpoint_path", "") or "")
    latest_checkpoint = find_latest_checkpoint(args.save_dir, prefix="model")
    stats_checkpoint = explicit_checkpoint or latest_checkpoint
    checkpoint_step = parse_checkpoint_number(stats_checkpoint) if stats_checkpoint else 0
    stats = {
        "score_alpha": np.asarray(args.score_alpha, dtype=np.float32),
        "checkpoint_path": stats_checkpoint,
        "checkpoint_step": checkpoint_step,
        "stats_split": stats_split,
    }
    np.save(os.path.join(args.save_dir, "train_stats.npy"), stats, allow_pickle=True)

    summary = {
        "checkpoint_path": stats_checkpoint,
        "checkpoint_step": checkpoint_step,
        "stats_split": stats_split,
        "score_alpha": [float(value) for value in args.score_alpha],
    }
    with open(os.path.join(args.save_dir, "train_stats_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def prepare_training_assets(args):
    dataset_dir = getattr(args, "data_dir", "") or None
    species_vocab, action_vocab = build_label_vocabs(dataset_dir)
    args.num_species = species_vocab.size
    args.num_actions = action_vocab.size
    args.species_vocab = list(species_vocab.labels)
    args.action_vocab = list(action_vocab.labels)
    return species_vocab, action_vocab


def main() -> None:
    args = build_parser().parse_args()
    startup_start = time.perf_counter()
    fixseed(
        args.seed,
        cudnn_benchmark=getattr(args, "cudnn_benchmark", True),
        allow_tf32=getattr(args, "allow_tf32", True),
    )
    save_dir = prepare_save_dir(args)
    args.checkpoint_step_numbering = "completed_steps"

    species_vocab, action_vocab = prepare_training_assets(args)

    ml_platform_type = eval(args.ml_platform_type)
    ml_platform = ml_platform_type(save_dir=save_dir)
    ml_platform.report_args(args, name="Args")

    with open(os.path.join(save_dir, "args.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=4, sort_keys=True)

    data_loader = create_data_loader(
        args,
        args.train_split,
        shuffle=True,
        drop_last=True,
        balanced=args.balanced,
    )
    print(
        f"Motion scorer DataLoader: num_workers={args.num_workers}, "
        f"prefetch_factor={getattr(args, 'prefetch_factor', 2) if args.num_workers > 0 else 'n/a'}, "
        f"motion_cache_size={getattr(args, 'motion_cache_size', 0)}, "
        f"main_process_prefetch_batches={getattr(args, 'main_process_prefetch_batches', 0)}, "
        f"timing_log_interval={getattr(args, 'timing_log_interval', 1000)}"
    )

    trainer = MotionScorerTrainer(
        args,
        ml_platform,
        data_loader,
        species_vocab=species_vocab,
        action_vocab=action_vocab,
    )

    ml_platform.watch_model(trainer.model)
    trainer.run()
    compute_and_save_train_stats(args)
    ml_platform.close()


if __name__ == "__main__":
    main()
