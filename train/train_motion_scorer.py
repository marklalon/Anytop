from __future__ import annotations

import copy
import json
import os
import re
import shutil
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.get_data import get_dataset_loader
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.nn import update_ema
from model.motion_autoencoder import (
    MotionAutoencoder,
    build_motion_valid_mask,
    masked_l1_per_sample,
    motion_perceptual_recon_error_per_sample,
)
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
    group.add_argument("--d_model", default=128, type=int, help="Hidden size for the encoder and decoder.")
    group.add_argument("--latent_dim", default=64, type=int, help="Latent size of the autoencoder bottleneck.")
    group.add_argument("--num_conv_layers", default=3, type=int, help="Number of temporal residual conv blocks.")
    group.add_argument("--decoder_num_conv_layers", default=0, type=int,
                       help="Number of temporal residual conv blocks used in the decoder. 0 keeps the lighter decoder used by older checkpoints.")
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
    group.add_argument("--timing_log_interval", default=1000, type=int,
                       help="Report averaged timing breakdown every N training steps.")
    group.add_argument("--recon_position_weight", default=0.35, type=float,
                       help="Weight for frame-wise reconstruction in the perceptual recon error.")
    group.add_argument("--recon_velocity_weight", default=0.30, type=float,
                       help="Weight for first-order temporal differences in the perceptual recon error.")
    group.add_argument("--recon_acceleration_weight", default=0.20, type=float,
                       help="Weight for second-order temporal differences in the perceptual recon error.")
    group.add_argument("--recon_blur_weight", default=0.15, type=float,
                       help="Weight for low-frequency temporally blurred reconstruction in the perceptual recon error.")
    group.add_argument("--lambda_quality_invariance", default=0.15, type=float,
                       help="Weight for keeping latent embeddings stable across detail-preserving augmentations.")
    group.add_argument("--lambda_quality_variance", default=0.05, type=float,
                       help="Weight for the variance floor regularizer on normalized latent embeddings.")
    group.add_argument("--lambda_quality_covariance", default=0.01, type=float,
                       help="Weight for decorrelating normalized latent embedding dimensions.")
    group.add_argument("--lambda_quality_margin", default=0.10, type=float,
                       help="Weight for separating clean embeddings from synthetic negative embeddings.")
    group.add_argument("--quality_margin", default=0.20, type=float,
                       help="Cosine-distance margin between clean and negative latent embeddings.")
    group.add_argument("--quality_variance_floor", default=0.5, type=float,
                       help="Minimum per-dimension standard deviation target for normalized latent embeddings.")
    group.add_argument("--lambda_negative_recon", default=0.10, type=float,
                       help="Weight for forcing synthetic negatives to reconstruct worse than clean motions.")
    group.add_argument("--negative_recon_margin", default=0.15, type=float,
                       help="Margin by which synthetic negatives should exceed clean perceptual recon error.")
    group.add_argument("--view_noise_sigma", default=0.03, type=float,
                       help="Gaussian noise strength used for detail-preserving semantic views.")
    group.add_argument("--recon_bootstrap_steps", default=2000, type=int,
                       help="Number of initial steps that optimize only clean reconstruction before auxiliary quality losses are enabled.")
    group.add_argument("--recon_perceptual_warmup_steps", default=4000, type=int,
                       help="Number of steps used to blend from simple frame-wise reconstruction into the full perceptual reconstruction objective.")
    group.add_argument("--aux_warmup_steps", default=4000, type=int,
                       help="Number of steps used to linearly ramp auxiliary quality losses after reconstruction bootstrap.")
    group.add_argument("--negative_noise_sigma_start", default=0.35, type=float,
                       help="Initial synthetic-negative noise strength used when auxiliary losses first turn on.")
    group.add_argument("--negative_noise_sigma", default=1.0, type=float,
                       help="Gaussian noise strength used for synthetic negative motions.")
    return parser


def compute_recon_error(args, target: torch.Tensor, prediction: torch.Tensor, n_joints: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    return motion_perceptual_recon_error_per_sample(
        target,
        prediction,
        n_joints,
        lengths,
        position_weight=float(getattr(args, "recon_position_weight", 0.35)),
        velocity_weight=float(getattr(args, "recon_velocity_weight", 0.30)),
        acceleration_weight=float(getattr(args, "recon_acceleration_weight", 0.20)),
        blur_weight=float(getattr(args, "recon_blur_weight", 0.15)),
    )


def compute_bootstrap_recon_error(target: torch.Tensor, prediction: torch.Tensor, n_joints: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    return masked_l1_per_sample(target, prediction, n_joints, lengths)


def temporal_smooth_motion(motion: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    prev_frames = torch.cat([motion[..., :1], motion[..., :-1]], dim=-1)
    next_frames = torch.cat([motion[..., 1:], motion[..., -1:]], dim=-1)
    smoothed = 0.25 * prev_frames + 0.5 * motion + 0.25 * next_frames
    return smoothed * valid_mask + motion * (1.0 - valid_mask)


def linear_warmup_factor(current_step: int, start_step: int, warmup_steps: int) -> float:
    if current_step <= start_step:
        return 0.0
    if warmup_steps <= 0:
        return 1.0
    progress = (current_step - start_step) / float(warmup_steps)
    return float(min(max(progress, 0.0), 1.0))


def get_auxiliary_loss_factor(args, current_step: int) -> float:
    bootstrap_steps = max(0, int(getattr(args, "recon_bootstrap_steps", 0)))
    recon_warmup_steps = max(0, int(getattr(args, "recon_perceptual_warmup_steps", 0)))
    warmup_steps = max(0, int(getattr(args, "aux_warmup_steps", 0)))
    aux_start_step = bootstrap_steps + recon_warmup_steps
    return linear_warmup_factor(current_step, aux_start_step, warmup_steps)


def get_recon_perceptual_factor(args, current_step: int) -> float:
    bootstrap_steps = max(0, int(getattr(args, "recon_bootstrap_steps", 0)))
    warmup_steps = max(0, int(getattr(args, "recon_perceptual_warmup_steps", 0)))
    return linear_warmup_factor(current_step, bootstrap_steps, warmup_steps)


def get_negative_noise_sigma(args, aux_factor: float) -> float:
    start_sigma = float(getattr(args, "negative_noise_sigma_start", 0.35))
    target_sigma = float(getattr(args, "negative_noise_sigma", 1.0))
    if aux_factor <= 0.0:
        return start_sigma
    return start_sigma + (target_sigma - start_sigma) * float(min(max(aux_factor, 0.0), 1.0))


def make_semantic_view(args, motion: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    batch_size = motion.shape[0]
    smoothed = temporal_smooth_motion(motion, valid_mask)
    blend = torch.empty((batch_size, 1, 1, 1), device=motion.device, dtype=motion.dtype).uniform_(0.15, 0.55)
    scale = 1.0 + 0.02 * torch.randn((batch_size, 1, 1, 1), device=motion.device, dtype=motion.dtype)
    noise = torch.randn_like(motion) * float(getattr(args, "view_noise_sigma", 0.03))
    view = torch.lerp(motion * scale, smoothed, blend) + noise * valid_mask
    return view * valid_mask


def make_temporal_shuffle_negative(motion: torch.Tensor, lengths: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    shuffled = torch.zeros_like(motion)
    for batch_index in range(motion.shape[0]):
        valid_length = int(lengths[batch_index].item())
        if valid_length <= 1:
            shuffled[batch_index] = motion[batch_index]
            continue
        permutation = torch.randperm(valid_length, device=motion.device)
        shuffled[batch_index, :, :, :valid_length] = motion[batch_index, :, :, permutation]
    return shuffled * valid_mask


def make_negative_view(args, motion: torch.Tensor, lengths: torch.Tensor, valid_mask: torch.Tensor, *, noise_sigma: float | None = None) -> torch.Tensor:
    if noise_sigma is None:
        noise_sigma = float(getattr(args, "negative_noise_sigma", 1.0))
    random_negative = torch.randn_like(motion) * float(noise_sigma)
    random_negative = random_negative * valid_mask
    shuffled_negative = make_temporal_shuffle_negative(motion, lengths, valid_mask)
    use_random = torch.rand((motion.shape[0], 1, 1, 1), device=motion.device) < 0.5
    return torch.where(use_random, random_negative, shuffled_negative)


def normalize_density_embeddings(latents: torch.Tensor) -> torch.Tensor:
    return F.normalize(latents.float(), dim=-1)


def _variance_floor_loss(features: torch.Tensor, floor: float) -> torch.Tensor:
    if features.shape[0] <= 1:
        return torch.zeros((), device=features.device, dtype=features.dtype)
    std = torch.sqrt(features.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(float(floor) - std).mean()


def _covariance_loss(features: torch.Tensor) -> torch.Tensor:
    if features.shape[0] <= 1:
        return torch.zeros((), device=features.device, dtype=features.dtype)
    centered = features - features.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / max(features.shape[0] - 1, 1)
    off_diagonal = cov - torch.diag_embed(torch.diagonal(cov))
    return off_diagonal.pow(2).sum() / max(features.shape[1], 1)


def compute_quality_regularization(
    clean_features: torch.Tensor,
    view_features: torch.Tensor,
    negative_features: torch.Tensor,
    *,
    variance_floor: float,
    quality_margin: float,
) -> dict[str, torch.Tensor]:
    invariance = F.mse_loss(clean_features, view_features)
    variance = _variance_floor_loss(clean_features, variance_floor) + _variance_floor_loss(view_features, variance_floor)
    covariance = _covariance_loss(clean_features) + _covariance_loss(view_features)
    positive_distance = 1.0 - (clean_features * view_features).sum(dim=-1)
    negative_distance = 1.0 - (clean_features * negative_features).sum(dim=-1)
    margin = F.relu(positive_distance + float(quality_margin) - negative_distance).mean()
    return {
        "invariance": invariance,
        "variance": variance,
        "covariance": covariance,
        "margin": margin,
    }


def prepare_save_dir(args) -> str:
    save_dir = args.save_dir
    if not save_dir:
        save_root = os.path.join(os.getcwd(), "save")
        os.makedirs(save_root, exist_ok=True)
        prefix = getattr(args, "model_prefix", None) or "MotionScorer"
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
        if file_name in {"args.json", "train_stats.npy", "train_stats_summary.json", "debug_score_report.json"}:
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
            decoder_num_conv_layers=args.decoder_num_conv_layers,
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

        if self.resume_checkpoint and args.load_optimizer_state:
            opt_path = os.path.join(os.path.dirname(self.resume_checkpoint), f"opt{self.resume_completed_steps:09d}.pt")
            if os.path.exists(opt_path):
                opt_state = torch.load(opt_path, map_location="cpu")
                if self.amp_enabled and isinstance(opt_state, dict) and "opt" in opt_state:
                    if "scaler" in opt_state and self.mp_trainer.scaler.is_enabled():
                        self.mp_trainer.scaler.load_state_dict(opt_state["scaler"])
                    opt_state = opt_state["opt"]
                self.opt.load_state_dict(opt_state)

        apply_current_optimizer_hparams(self.opt, args)
        self.lr_scheduler = build_step_lr_scheduler(self.opt, args, self.resume_completed_steps)

        self.model.train()

    def autocast_context(self):
        if not self.amp_enabled:
            return torch.autocast(device_type=self.device.type, enabled=False)
        return torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)

    def train_step(self, motion: torch.Tensor, cond: dict, current_step: int) -> float:
        n_joints = cond["y"]["n_joints"]
        lengths = cond["y"]["lengths"]
        valid_mask = build_motion_valid_mask(n_joints, lengths, motion.shape[1], motion.shape[-1], device=motion.device).to(motion.dtype)
        recon_perceptual_factor = get_recon_perceptual_factor(self.args, current_step)
        aux_factor = get_auxiliary_loss_factor(self.args, current_step)
        negative_noise_sigma = get_negative_noise_sigma(self.args, aux_factor)

        self.mp_trainer.zero_grad()
        with self.autocast_context():
            reconstruction, latents = self.model(motion, n_joints, lengths)
            clean_bootstrap_recon_error = compute_bootstrap_recon_error(motion, reconstruction, n_joints, lengths)
            clean_perceptual_recon_error = compute_recon_error(self.args, motion, reconstruction, n_joints, lengths)
            clean_recon_error = torch.lerp(
                clean_bootstrap_recon_error,
                clean_perceptual_recon_error,
                recon_perceptual_factor,
            )
            recon_loss = clean_recon_error.mean()
            zero = recon_loss.new_zeros(())

            if aux_factor > 0.0:
                semantic_view = make_semantic_view(self.args, motion, valid_mask)
                _, semantic_latents = self.model(semantic_view, n_joints, lengths)

                negative_view = make_negative_view(
                    self.args,
                    motion,
                    lengths,
                    valid_mask,
                    noise_sigma=negative_noise_sigma,
                )
                negative_reconstruction, negative_latents = self.model(negative_view, n_joints, lengths)
                negative_bootstrap_recon_error = compute_bootstrap_recon_error(
                    negative_view,
                    negative_reconstruction,
                    n_joints,
                    lengths,
                )
                negative_perceptual_recon_error = compute_recon_error(
                    self.args,
                    negative_view,
                    negative_reconstruction,
                    n_joints,
                    lengths,
                )
                negative_recon_error = torch.lerp(
                    negative_bootstrap_recon_error,
                    negative_perceptual_recon_error,
                    recon_perceptual_factor,
                )

                clean_features = normalize_density_embeddings(latents)
                semantic_features = normalize_density_embeddings(semantic_latents)
                negative_features = normalize_density_embeddings(negative_latents)
                quality_terms = compute_quality_regularization(
                    clean_features,
                    semantic_features,
                    negative_features,
                    variance_floor=float(getattr(self.args, "quality_variance_floor", 0.5)),
                    quality_margin=float(getattr(self.args, "quality_margin", 0.20)),
                )
                negative_recon_loss = F.relu(
                    clean_perceptual_recon_error.detach() + float(getattr(self.args, "negative_recon_margin", 0.15)) - negative_perceptual_recon_error
                ).mean()
            else:
                quality_terms = {
                    "invariance": zero,
                    "variance": zero,
                    "covariance": zero,
                    "margin": zero,
                }
                negative_recon_loss = zero
                negative_recon_error = zero

            loss = (
                recon_loss
                + aux_factor * float(getattr(self.args, "lambda_quality_invariance", 0.15)) * quality_terms["invariance"]
                + aux_factor * float(getattr(self.args, "lambda_quality_variance", 0.05)) * quality_terms["variance"]
                + aux_factor * float(getattr(self.args, "lambda_quality_covariance", 0.01)) * quality_terms["covariance"]
                + aux_factor * float(getattr(self.args, "lambda_quality_margin", 0.10)) * quality_terms["margin"]
                + aux_factor * float(getattr(self.args, "lambda_negative_recon", 0.10)) * negative_recon_loss
            )

        self.mp_trainer.backward(loss)
        took_step = self.mp_trainer.optimize(self.opt, self.lr_scheduler)
        if took_step and self.model_avg is not None:
            update_ema(self.model_avg.parameters(), self.model.parameters(), rate=self.args.ema_decay)
        return {
            "loss": float(loss.detach().item()),
            "recon_loss": float(recon_loss.detach().item()),
            "bootstrap_recon_loss": float(clean_bootstrap_recon_error.mean().detach().item()),
            "perceptual_recon_loss": float(clean_perceptual_recon_error.mean().detach().item()),
            "quality_invariance": float(quality_terms["invariance"].detach().item()),
            "quality_variance": float(quality_terms["variance"].detach().item()),
            "quality_covariance": float(quality_terms["covariance"].detach().item()),
            "quality_margin": float(quality_terms["margin"].detach().item()),
            "negative_recon_loss": float(negative_recon_loss.detach().item()),
            "clean_recon_error": float(clean_recon_error.mean().detach().item()),
            "negative_recon_error": float(negative_recon_error.mean().detach().item()),
            "recon_perceptual_factor": float(recon_perceptual_factor),
            "aux_factor": float(aux_factor),
            "negative_noise_sigma": float(negative_noise_sigma),
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

    def run(self) -> MotionAutoencoder:
        completed_steps = self.resume_completed_steps
        running_metrics: dict[str, list[float]] = {}
        data_iter = iter(self.data_loader)
        timing_log_interval = max(1, int(getattr(self.args, "timing_log_interval", self.args.log_interval)))
        timing_totals = {
            "data_wait_s": 0.0,
            "host_to_device_s": 0.0,
            "step_s": 0.0,
            "loop_s": 0.0,
        }
        timing_steps = 0
        timing_samples = 0

        while completed_steps < self.args.num_steps:
            loop_start = time.perf_counter()
            fetch_start = time.perf_counter()
            try:
                motion, cond = next(data_iter)
            except StopIteration:
                data_iter = iter(self.data_loader)
                motion, cond = next(data_iter)
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
                running_metrics.setdefault(metric_name, []).append(metric_value)
            timing_totals["data_wait_s"] += data_wait_s
            timing_totals["host_to_device_s"] += host_to_device_s
            timing_totals["step_s"] += step_s
            timing_totals["loop_s"] += loop_s
            timing_steps += 1
            timing_samples += int(motion.shape[0])

            if completed_steps % self.args.log_interval == 0 or completed_steps == self.args.num_steps:
                mean_metrics = {
                    metric_name: float(np.mean(metric_values))
                    for metric_name, metric_values in running_metrics.items()
                    if metric_values
                }
                print(
                    "step[{}]: total_loss[{:.6f}] recon_loss[{:.6f}] bootstrap_recon[{:.6f}] perceptual_recon[{:.6f}] neg_recon_loss[{:.6f}] quality_inv[{:.6f}] quality_margin[{:.6f}] recon_pf[{:.3f}] aux_factor[{:.3f}] neg_sigma[{:.3f}]".format(
                        completed_steps,
                        mean_metrics.get("loss", 0.0),
                        mean_metrics.get("recon_loss", 0.0),
                        mean_metrics.get("bootstrap_recon_loss", 0.0),
                        mean_metrics.get("perceptual_recon_loss", 0.0),
                        mean_metrics.get("negative_recon_loss", 0.0),
                        mean_metrics.get("quality_invariance", 0.0),
                        mean_metrics.get("quality_margin", 0.0),
                        mean_metrics.get("recon_perceptual_factor", 0.0),
                        mean_metrics.get("aux_factor", 0.0),
                        mean_metrics.get("negative_noise_sigma", 0.0),
                    )
                )
                for metric_name, metric_value in mean_metrics.items():
                    self.ml_platform.report_scalar(metric_name, metric_value, completed_steps, group_name="Train")
                self.ml_platform.report_scalar("lr", self.lr_scheduler.get_last_lr()[0], completed_steps, group_name="Train")
                running_metrics.clear()

            if completed_steps % timing_log_interval == 0 or completed_steps == self.args.num_steps:
                mean_loop_s = timing_totals["loop_s"] / max(timing_steps, 1)
                mean_data_wait_ms = 1000.0 * timing_totals["data_wait_s"] / max(timing_steps, 1)
                mean_host_to_device_ms = 1000.0 * timing_totals["host_to_device_s"] / max(timing_steps, 1)
                mean_step_ms = 1000.0 * timing_totals["step_s"] / max(timing_steps, 1)
                data_wait_pct = 100.0 * timing_totals["data_wait_s"] / max(timing_totals["loop_s"], 1e-9)
                step_pct = 100.0 * timing_totals["step_s"] / max(timing_totals["loop_s"], 1e-9)
                samples_per_s = timing_samples / max(timing_totals["loop_s"], 1e-9)
                print(
                    "timing[{}]: data_wait_ms[{:.2f}] host_to_device_ms[{:.2f}] step_ms[{:.2f}] total_ms[{:.2f}] "
                    "data_wait_pct[{:.1f}] step_pct[{:.1f}] samples_per_s[{:.1f}]".format(
                        completed_steps,
                        mean_data_wait_ms,
                        mean_host_to_device_ms,
                        mean_step_ms,
                        1000.0 * mean_loop_s,
                        data_wait_pct,
                        step_pct,
                        samples_per_s,
                    )
                )
                self.ml_platform.report_scalar("data_wait_ms", mean_data_wait_ms, completed_steps, group_name="Timing")
                self.ml_platform.report_scalar("host_to_device_ms", mean_host_to_device_ms, completed_steps, group_name="Timing")
                self.ml_platform.report_scalar("step_ms", mean_step_ms, completed_steps, group_name="Timing")
                self.ml_platform.report_scalar("total_ms", 1000.0 * mean_loop_s, completed_steps, group_name="Timing")
                self.ml_platform.report_scalar("samples_per_s", samples_per_s, completed_steps, group_name="Timing")
                timing_totals = {
                    "data_wait_s": 0.0,
                    "host_to_device_s": 0.0,
                    "step_s": 0.0,
                    "loop_s": 0.0,
                }
                timing_steps = 0
                timing_samples = 0

            if completed_steps % self.args.save_interval == 0 or completed_steps == self.args.num_steps:
                self.save(completed_steps)

        return self.model_avg if self.model_avg is not None else self.model


def mahalanobis_distance_np(latents: np.ndarray, mean: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    diff = latents - mean[None, :]
    distances_sq = np.einsum("bi,ij,bj->b", diff, cov_inv, diff)
    return np.sqrt(np.clip(distances_sq, a_min=0.0, a_max=None))


def diagonal_mahalanobis_distance_np(latents: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    diff = latents - mean[None, :]
    distances_sq = ((diff * diff) / var[None, :]).sum(axis=1)
    return np.sqrt(np.clip(distances_sq, a_min=0.0, a_max=None))


def knn_distance_np(latents: np.ndarray, reference_latents: np.ndarray, k: int, *, exclude_self: bool) -> np.ndarray:
    if latents.ndim != 2 or reference_latents.ndim != 2:
        raise ValueError("Expected 2D latent arrays for kNN density computation.")
    if latents.shape[1] != reference_latents.shape[1]:
        raise ValueError("Latent dimensions must match for kNN density computation.")
    if reference_latents.shape[0] == 0:
        raise ValueError("reference_latents must be non-empty.")

    effective_k = max(1, int(k))
    required_neighbors = effective_k + (1 if exclude_self else 0)
    required_neighbors = min(required_neighbors, reference_latents.shape[0])

    diffs = latents[:, None, :] - reference_latents[None, :, :]
    distances = np.sqrt(np.clip(np.sum(diffs * diffs, axis=2), a_min=0.0, a_max=None))
    partition = np.partition(distances, kth=required_neighbors - 1, axis=1)[:, :required_neighbors]
    partition.sort(axis=1)

    if exclude_self and required_neighbors > 1:
        neighbor_slice = partition[:, 1:required_neighbors]
    else:
        neighbor_slice = partition[:, :required_neighbors]
    return neighbor_slice.mean(axis=1)


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
        action_tags=getattr(args, "action_tags", ""),
        motion_cache_size=getattr(args, "motion_cache_size", 0),
        main_process_prefetch_batches=getattr(args, "main_process_prefetch_batches", 0),
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
            recon_error = compute_recon_error(args, motion, reconstruction, cond["y"]["n_joints"], cond["y"]["lengths"])
            density_embeddings = normalize_density_embeddings(latents)
            all_latents.append(density_embeddings.detach().float().cpu().numpy())
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
    var = latents.var(axis=0) + float(args.stats_eps)
    mahal_distances = diagonal_mahalanobis_distance_np(latents, mean, var)
    density_knn_k = min(5, max(1, latents.shape[0] - 1))
    knn_distances = knn_distance_np(latents, latents, density_knn_k, exclude_self=True)

    recon_threshold = float(max(np.percentile(recon_errors, args.recon_percentile), args.stats_eps))
    density_threshold = float(max(np.percentile(knn_distances, args.density_percentile), args.stats_eps))
    latest_checkpoint = find_latest_checkpoint(args.save_dir, prefix="model")
    checkpoint_step = parse_checkpoint_number(latest_checkpoint) if latest_checkpoint else 0

    stats = {
        "mean": mean.astype(np.float32),
        "cov": cov.astype(np.float32),
        "cov_inv": cov_inv.astype(np.float32),
        "var": var.astype(np.float32),
        "recon_errors": recon_errors.astype(np.float32),
        "mahal_distances": mahal_distances.astype(np.float32),
        "train_latents": latents.astype(np.float32),
        "knn_distances": knn_distances.astype(np.float32),
        "recon_threshold": recon_threshold,
        "density_threshold": density_threshold,
        "alpha": float(args.score_alpha),
        "recon_percentile": float(args.recon_percentile),
        "density_percentile": float(args.density_percentile),
        "num_samples": int(latents.shape[0]),
        "stats_split": stats_split,
        "density_mode": "knn",
        "density_knn_k": int(density_knn_k),
        "density_embedding_kind": "normalized_latent",
        "recon_error_kind": "motion_perceptual_v1",
        "recon_score_calibration": "hybrid_empirical_logistic_v1",
        "recon_empirical_weight": 0.5,
        "legacy_density_mode": "diagonal_mahalanobis",
        "checkpoint_path": latest_checkpoint,
        "checkpoint_step": checkpoint_step,
    }
    stats_path = os.path.join(args.save_dir, "train_stats.npy")
    np.save(stats_path, stats, allow_pickle=True)

    summary = {
        "recon_threshold": recon_threshold,
        "density_threshold": density_threshold,
        "alpha": float(args.score_alpha),
        "num_samples": int(latents.shape[0]),
        "stats_split": stats_split,
        "density_mode": "knn",
        "density_knn_k": int(density_knn_k),
        "density_embedding_kind": "normalized_latent",
        "recon_error_kind": "motion_perceptual_v1",
        "recon_score_calibration": "hybrid_empirical_logistic_v1",
        "recon_empirical_weight": 0.5,
        "legacy_density_mode": "diagonal_mahalanobis",
        "checkpoint_path": latest_checkpoint,
        "checkpoint_step": checkpoint_step,
    }
    with open(os.path.join(args.save_dir, "train_stats_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    model.train()


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

    ml_platform_type = eval(args.ml_platform_type)
    ml_platform = ml_platform_type(save_dir=save_dir)
    ml_platform.report_args(args, name="Args")

    with open(os.path.join(save_dir, "args.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=4, sort_keys=True)

    data_loader_start = time.perf_counter()
    data_loader = create_data_loader(args, args.train_split, shuffle=True, drop_last=True, balanced=args.balanced)
    data_loader_build_s = time.perf_counter() - data_loader_start
    print(
        f"Motion scorer DataLoader: num_workers={args.num_workers}, "
        f"prefetch_factor={getattr(args, 'prefetch_factor', 2) if args.num_workers > 0 else 'n/a'}, "
        f"motion_cache_size={getattr(args, 'motion_cache_size', 0)}, "
        f"main_process_prefetch_batches={getattr(args, 'main_process_prefetch_batches', 0)}, "
        f"timing_log_interval={getattr(args, 'timing_log_interval', 1000)}"
    )
    print(f"Motion scorer startup: data_loader_build_s={data_loader_build_s:.2f}")
    trainer_init_start = time.perf_counter()
    trainer = MotionScorerTrainer(args, ml_platform, data_loader)
    trainer_init_s = time.perf_counter() - trainer_init_start
    print(f"Motion scorer startup: trainer_init_s={trainer_init_s:.2f} total_startup_s={time.perf_counter() - startup_start:.2f}")
    ml_platform.watch_model(trainer.model)
    final_model = trainer.run()
    compute_and_save_train_stats(args, final_model, trainer.device, trainer.autocast_dtype, trainer.amp_enabled)
    ml_platform.close()


if __name__ == "__main__":
    main()