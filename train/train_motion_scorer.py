from __future__ import annotations

import copy
import json
import os
import re
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.get_data import get_dataset_loader
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.nn import update_ema
from model.motion_autoencoder import MotionAutoencoder, masked_mse_per_sample
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
    group.add_argument("--d_model", default=256, type=int, help="Hidden size for the encoder and decoder.")
    group.add_argument("--latent_dim", default=512, type=int, help="Latent size of the autoencoder bottleneck.")
    group.add_argument("--num_conv_layers", default=4, type=int, help="Number of temporal residual conv blocks.")
    group.add_argument("--kernel_size", default=5, type=int, help="Kernel size for temporal conv blocks.")
    group.add_argument("--max_joints", default=143, type=int, help="Maximum padded joint count.")
    group.add_argument("--train_split", default="train", choices=["train", "all", "val", "test"], type=str,
                       help="Dataset split used for training.")
    group.add_argument("--stats_split", default="", type=str,
                       help="Split used to cache latent statistics after training. Empty means train_split.")
    group.add_argument("--stats_batch_size", default=0, type=int,
                       help="Batch size for the post-training stats pass. 0 reuses batch_size.")
    group.add_argument("--lr_step_size", default=10000, type=int, help="StepLR step size in optimizer steps.")
    group.add_argument("--lr_gamma", default=0.99, type=float, help="StepLR gamma.")
    group.add_argument("--ema_decay", default=0.999, type=float, help="EMA decay when --use_ema is enabled.")
    group.add_argument("--recon_percentile", default=95.0, type=float,
                       help="Percentile used to normalize reconstruction errors into scores.")
    group.add_argument("--density_percentile", default=95.0, type=float,
                       help="Percentile used to normalize Mahalanobis distances into scores.")
    group.add_argument("--score_alpha", default=0.7, type=float,
                       help="Weight assigned to the reconstruction score in the final quality score.")
    group.add_argument("--stats_eps", default=1e-4, type=float,
                       help="Diagonal regularizer added before inverting the latent covariance matrix.")
    group.add_argument("--load_optimizer_state", action="store_true",
                       help="Restore optimizer and scaler state when resuming.")
    return parser


def prepare_save_dir(args) -> str:
    save_dir = args.save_dir
    if not save_dir:
        save_root = os.path.join(os.getcwd(), "save")
        os.makedirs(save_root, exist_ok=True)
        prefix = getattr(args, "model_prefix", None) or "MotionScorer"
        model_name = f"{prefix}_dataset_truebones_bs_{args.batch_size}_latentdim_{args.latent_dim}"
        existing_runs = [name for name in os.listdir(save_root) if name.startswith(model_name)]
        if existing_runs:
            model_name = f"{model_name}_{len(existing_runs)}"
        save_dir = os.path.join(save_root, model_name)
        args.save_dir = save_dir

    existing_checkpoint = find_latest_checkpoint(save_dir, prefix="model") if os.path.isdir(save_dir) else ""
    if os.path.exists(save_dir) and not args.overwrite and not args.resume_checkpoint and not existing_checkpoint:
        raise FileExistsError(f"save_dir [{save_dir}] already exists.")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


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
        motion_name_keywords=getattr(args, "motion_name_keywords", ""),
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


class MotionScorerTrainer:
    def __init__(self, args, ml_platform, data_loader) -> None:
        self.args = args
        self.ml_platform = ml_platform
        self.data_loader = data_loader

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

        self.model = MotionAutoencoder(
            feature_dim=args.feature_dim,
            d_model=args.d_model,
            latent_dim=args.latent_dim,
            num_conv_layers=args.num_conv_layers,
            kernel_size=args.kernel_size,
            max_joints=args.max_joints,
            max_frames=args.num_frames,
        ).to(self.device)
        self.model_avg = copy.deepcopy(self.model) if args.use_ema else None
        self.resume_checkpoint = args.resume_checkpoint.strip() if args.resume_checkpoint else ""
        self.resume_completed_steps = 0

        if self.resume_checkpoint:
            payload = torch.load(self.resume_checkpoint, map_location="cpu")
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
        )
        self.opt = AdamW(self.mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.opt,
            step_size=max(1, int(args.lr_step_size)),
            gamma=float(args.lr_gamma),
        )

        if self.resume_checkpoint and args.load_optimizer_state:
            opt_path = os.path.join(os.path.dirname(self.resume_checkpoint), f"opt{self.resume_completed_steps:09d}.pt")
            if os.path.exists(opt_path):
                opt_state = torch.load(opt_path, map_location="cpu")
                if self.amp_enabled and isinstance(opt_state, dict) and "opt" in opt_state:
                    if "scaler" in opt_state and self.mp_trainer.scaler.is_enabled():
                        self.mp_trainer.scaler.load_state_dict(opt_state["scaler"])
                    opt_state = opt_state["opt"]
                self.opt.load_state_dict(opt_state)

        self.model.train()

    def autocast_context(self):
        if not self.amp_enabled:
            return torch.autocast(device_type=self.device.type, enabled=False)
        return torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)

    def train_step(self, motion: torch.Tensor, cond: dict) -> float:
        n_joints = cond["y"]["n_joints"]
        lengths = cond["y"]["lengths"]

        self.mp_trainer.zero_grad()
        with self.autocast_context():
            reconstruction, _ = self.model(motion, n_joints, lengths)
            loss = masked_mse_per_sample(motion, reconstruction, n_joints, lengths).mean()

        self.mp_trainer.backward(loss)
        took_step = self.mp_trainer.optimize(self.opt, self.lr_scheduler)
        if took_step and self.model_avg is not None:
            update_ema(self.model_avg.parameters(), self.model.parameters(), rate=self.args.ema_decay)
        return float(loss.detach().item())

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

    def run(self) -> MotionAutoencoder:
        completed_steps = self.resume_completed_steps
        running_losses = []
        data_iter = iter(self.data_loader)

        while completed_steps < self.args.num_steps:
            try:
                motion, cond = next(data_iter)
            except StopIteration:
                data_iter = iter(self.data_loader)
                motion, cond = next(data_iter)

            motion = motion.to(self.device, non_blocking=self.non_blocking)
            cond = move_cond_to_device(cond, self.device, self.non_blocking)
            loss_value = self.train_step(motion, cond)

            completed_steps += 1
            running_losses.append(loss_value)

            if completed_steps % self.args.log_interval == 0 or completed_steps == self.args.num_steps:
                mean_loss = float(np.mean(running_losses)) if running_losses else loss_value
                print(f"step[{completed_steps}]: recon_loss[{mean_loss:0.6f}]")
                self.ml_platform.report_scalar("recon_loss", mean_loss, completed_steps, group_name="Train")
                self.ml_platform.report_scalar("lr", self.lr_scheduler.get_last_lr()[0], completed_steps, group_name="Train")
                running_losses.clear()

            if completed_steps % self.args.save_interval == 0 or completed_steps == self.args.num_steps:
                self.save(completed_steps)

        return self.model_avg if self.model_avg is not None else self.model


def mahalanobis_distance_np(latents: np.ndarray, mean: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    diff = latents - mean[None, :]
    distances_sq = np.einsum("bi,ij,bj->b", diff, cov_inv, diff)
    return np.sqrt(np.clip(distances_sq, a_min=0.0, a_max=None))


def compute_and_save_train_stats(args, model: MotionAutoencoder, device: torch.device, autocast_dtype, amp_enabled: bool):
    stats_split = args.stats_split or args.train_split
    stats_batch_size = args.stats_batch_size or args.batch_size
    loader = get_dataset_loader(
        batch_size=stats_batch_size,
        num_frames=args.num_frames,
        split=stats_split,
        temporal_window=getattr(args, "temporal_window", 31),
        t5_name="t5-base",
        balanced=False,
        objects_subset=args.objects_subset,
        num_workers=args.num_workers,
        prefetch_factor=getattr(args, "prefetch_factor", 2),
        sample_limit=args.sample_limit,
        shuffle=False,
        drop_last=False,
        use_reference_conditioning=False,
        motion_name_keywords=getattr(args, "motion_name_keywords", ""),
    )

    model.eval()
    all_latents = []
    all_recon_errors = []
    with torch.no_grad():
        for motion, cond in tqdm(loader, desc="Caching train stats"):
            motion = motion.to(device, non_blocking=device.type == "cuda")
            cond = move_cond_to_device(cond, device, device.type == "cuda")
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled):
                reconstruction, latents = model(motion, cond["y"]["n_joints"], cond["y"]["lengths"])
            recon_error = masked_mse_per_sample(motion, reconstruction, cond["y"]["n_joints"], cond["y"]["lengths"])
            all_latents.append(latents.detach().float().cpu().numpy())
            all_recon_errors.append(recon_error.detach().float().cpu().numpy())

    latents = np.concatenate(all_latents, axis=0).astype(np.float64, copy=False)
    recon_errors = np.concatenate(all_recon_errors, axis=0).astype(np.float64, copy=False)

    mean = latents.mean(axis=0)
    if latents.shape[0] > 1:
        cov = np.cov(latents, rowvar=False)
    else:
        cov = np.eye(latents.shape[1], dtype=np.float64)
    cov = np.atleast_2d(cov)
    cov += np.eye(cov.shape[0], dtype=np.float64) * float(args.stats_eps)
    cov_inv = np.linalg.pinv(cov)
    mahal_distances = mahalanobis_distance_np(latents, mean, cov_inv)

    recon_threshold = float(max(np.percentile(recon_errors, args.recon_percentile), args.stats_eps))
    density_threshold = float(max(np.percentile(mahal_distances, args.density_percentile), args.stats_eps))

    stats = {
        "mean": mean.astype(np.float32),
        "cov": cov.astype(np.float32),
        "cov_inv": cov_inv.astype(np.float32),
        "recon_errors": recon_errors.astype(np.float32),
        "mahal_distances": mahal_distances.astype(np.float32),
        "recon_threshold": recon_threshold,
        "density_threshold": density_threshold,
        "alpha": float(args.score_alpha),
        "recon_percentile": float(args.recon_percentile),
        "density_percentile": float(args.density_percentile),
        "num_samples": int(latents.shape[0]),
        "stats_split": stats_split,
    }
    stats_path = os.path.join(args.save_dir, "train_stats.npy")
    np.save(stats_path, stats, allow_pickle=True)

    summary = {
        "recon_threshold": recon_threshold,
        "density_threshold": density_threshold,
        "alpha": float(args.score_alpha),
        "num_samples": int(latents.shape[0]),
        "stats_split": stats_split,
    }
    with open(os.path.join(args.save_dir, "train_stats_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    model.train()


def main() -> None:
    args = build_parser().parse_args()
    fixseed(
        args.seed,
        cudnn_benchmark=getattr(args, "cudnn_benchmark", True),
        allow_tf32=getattr(args, "allow_tf32", True),
    )
    save_dir = prepare_save_dir(args)
    args.checkpoint_step_numbering = "completed_steps"

    ml_platform_type = eval(args.ml_platform_type)
    ml_platform = ml_platform_type(save_dir=save_dir)
    ml_platform.report_args(args, name="Args")

    with open(os.path.join(save_dir, "args.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=4, sort_keys=True)

    data_loader = create_data_loader(args, args.train_split, shuffle=True, drop_last=True, balanced=args.balanced)
    trainer = MotionScorerTrainer(args, ml_platform, data_loader)
    ml_platform.watch_model(trainer.model)
    final_model = trainer.run()
    compute_and_save_train_stats(args, final_model, trainer.device, trainer.autocast_dtype, trainer.amp_enabled)
    ml_platform.close()


if __name__ == "__main__":
    main()