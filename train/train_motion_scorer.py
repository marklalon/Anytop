from __future__ import annotations

import copy
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
from sklearn.mixture import GaussianMixture
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.get_data import get_dataset_loader
from data_loaders.skeleton_metadata import (
    LabelVocab,
    build_label_vocabs,
    build_metadata_feature_lookup,
    load_skeleton_metadata,
    metadata_feature_dim,
)
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.nn import update_ema
from eval.biomechanical_negatives import NEGATIVE_KINDS, generate_biomechanical_negative_batch
from eval.physics_features import extract_physics_features
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
    group.add_argument("--stats_batch_size", default=0, type=int,
                       help="Batch size for the post-training stats pass. 0 reuses batch_size.")
    group.add_argument("--lr_step_size", default=10000, type=int, help="StepLR step size in optimizer steps.")
    group.add_argument("--lr_gamma", default=0.99, type=float, help="StepLR gamma.")
    group.add_argument("--ema_decay", default=0.999, type=float, help="EMA decay when --use_ema is enabled.")
    group.add_argument("--stats_eps", default=1e-4, type=float,
                       help="Diagonal regularizer added before inverting covariance matrices.")
    group.add_argument("--timing_log_interval", default=1000, type=int,
                       help="Report averaged timing breakdown every N training steps.")
    group.add_argument("--load_optimizer_state", action="store_true",
                       help="Restore optimizer and scaler state when resuming.")
    group.add_argument("--metadata_hidden_dim", default=128, type=int,
                       help="Hidden size used to project fixed skeleton metadata features.")
    group.add_argument("--cls_warmup_steps", default=2000, type=int,
                       help="Number of initial steps that optimize only species/action classification.")
    group.add_argument("--full_loss_warmup_steps", default=2000, type=int,
                       help="Number of steps used to linearly ramp discriminator/physics/VICReg losses after classification warmup.")
    group.add_argument("--lambda_species", default=1.0, type=float, help="Weight for species classification CE.")
    group.add_argument("--lambda_action", default=1.0, type=float, help="Weight for action classification CE.")
    group.add_argument("--lambda_disc_p", default=0.5, type=float, help="Weight for positive discriminator BCE.")
    group.add_argument("--lambda_disc_n", default=0.5, type=float, help="Weight for negative discriminator BCE.")
    group.add_argument("--lambda_phys", default=0.2, type=float, help="Weight for physics feature regression.")
    group.add_argument("--lambda_vic_var", default=0.05, type=float, help="Weight for VICReg variance floor.")
    group.add_argument("--lambda_vic_cov", default=0.01, type=float, help="Weight for VICReg covariance decorrelation.")
    group.add_argument("--quality_variance_floor", default=0.5, type=float,
                       help="Minimum per-dimension standard deviation target for normalized latent embeddings.")
    group.add_argument("--gmm_components", default=64, type=int, help="Number of mixture components used for latent density fitting.")
    group.add_argument("--gmm_covariance_type", default="diag", choices=["diag", "full"], type=str,
                       help="Covariance type used for post-training GMM fitting.")
    group.add_argument("--score_alpha", nargs=4, default=[1.0, 1.0, 1.0, 1.0], type=float,
                       help="Geometric-mean weights for recognizability, density, plausibility, and physics scores.")
    return parser


def linear_warmup_factor(current_step: int, start_step: int, warmup_steps: int) -> float:
    if current_step <= start_step:
        return 0.0
    if warmup_steps <= 0:
        return 1.0
    progress = (current_step - start_step) / float(warmup_steps)
    return float(min(max(progress, 0.0), 1.0))


def get_auxiliary_loss_factor(args, current_step: int) -> float:
    cls_warmup_steps = max(0, int(getattr(args, "cls_warmup_steps", 0)))
    warmup_steps = max(0, int(getattr(args, "full_loss_warmup_steps", 0)))
    return linear_warmup_factor(current_step, cls_warmup_steps, warmup_steps)


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


def create_data_loader(args, split: str, *, shuffle: bool, drop_last: bool, balanced: bool, batch_transform=None):
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
        batch_transform=batch_transform,
    )


def move_aux_to_device(aux_batch: dict | None, device: torch.device, non_blocking: bool) -> dict | None:
    if aux_batch is None:
        return None
    moved = {}
    for key, value in aux_batch.items():
        moved[key] = value.to(device, non_blocking=non_blocking) if torch.is_tensor(value) else value
    return moved


class PhysicsTargetLRUCache:
    def __init__(self, max_entries: int) -> None:
        self.max_entries = max(0, int(max_entries))
        self._cache: OrderedDict[tuple[str, int, int, str, int], torch.Tensor] = OrderedDict()
        self._lock = threading.Lock()

    def _sample_key(self, cond: dict, batch_index: int) -> tuple[str, int, int, str, int] | None:
        motion_names = cond["y"].get("motion_name")
        crop_start_indices = cond["y"].get("crop_start_ind")
        lengths = cond["y"].get("lengths")
        object_types = cond["y"].get("object_type")
        n_joints = cond["y"].get("n_joints")
        if motion_names is None or crop_start_indices is None or lengths is None or object_types is None or n_joints is None:
            return None
        return (
            str(motion_names[batch_index]),
            int(crop_start_indices[batch_index]),
            int(lengths[batch_index]),
            str(object_types[batch_index]),
            int(n_joints[batch_index]),
        )

    def _get(self, key: tuple[str, int, int, str, int]) -> torch.Tensor | None:
        with self._lock:
            value = self._cache.get(key)
            if value is None:
                return None
            self._cache.move_to_end(key)
            return value

    def _put(self, key: tuple[str, int, int, str, int], value: torch.Tensor) -> None:
        with self._lock:
            self._cache[key] = value.detach().to(device="cpu", dtype=torch.float32)
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)

    def get_batch(self, motion: torch.Tensor, cond: dict, skeleton_lookup) -> torch.Tensor:
        if self.max_entries <= 0:
            return extract_physics_features(
                motion,
                cond["y"]["n_joints"],
                cond["y"]["lengths"],
                cond["y"].get("object_type"),
                skeleton_lookup,
            ).detach()

        cached_results: list[torch.Tensor | None] = [None] * int(motion.shape[0])
        miss_indices: list[int] = []
        miss_keys: list[tuple[str, int, int, str, int] | None] = []
        for batch_index in range(int(motion.shape[0])):
            key = self._sample_key(cond, batch_index)
            if key is None:
                miss_indices.append(batch_index)
                miss_keys.append(None)
                continue
            cached_value = self._get(key)
            if cached_value is None:
                miss_indices.append(batch_index)
                miss_keys.append(key)
            else:
                cached_results[batch_index] = cached_value

        if miss_indices:
            miss_features = extract_physics_features(
                motion[miss_indices],
                cond["y"]["n_joints"][miss_indices],
                cond["y"]["lengths"][miss_indices],
                [cond["y"]["object_type"][index] for index in miss_indices],
                skeleton_lookup,
            ).detach()
            for miss_offset, batch_index in enumerate(miss_indices):
                feature_cpu = miss_features[miss_offset].to(device="cpu", dtype=torch.float32)
                cached_results[batch_index] = feature_cpu
                key = miss_keys[miss_offset]
                if key is not None:
                    self._put(key, feature_cpu)

        return torch.stack([result for result in cached_results if result is not None], dim=0).to(device=motion.device, dtype=motion.dtype)


class MotionScorerAuxBatchPreprocessor:
    def __init__(self, skeleton_lookup, physics_target_cache: PhysicsTargetLRUCache | None = None) -> None:
        self.skeleton_lookup = skeleton_lookup
        self.physics_target_cache = physics_target_cache
        self.enabled = False
        self.cpu_device = torch.device("cpu")

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def __call__(self, batch):
        motion, cond = batch
        if not self.enabled:
            return motion, cond, None

        object_types = cond["y"].get("object_type")
        if self.physics_target_cache is not None:
            physics_targets = self.physics_target_cache.get_batch(motion, cond, self.skeleton_lookup)
        else:
            physics_targets = extract_physics_features(
                motion,
                cond["y"]["n_joints"],
                cond["y"]["lengths"],
                object_types,
                self.skeleton_lookup,
            )
        negative_batch = generate_biomechanical_negative_batch(
            motion,
            cond["y"]["n_joints"],
            cond["y"]["lengths"],
            object_types,
            self.skeleton_lookup,
            feature_std=cond["y"].get("std"),
            negative_kinds=NEGATIVE_KINDS,
        )
        return motion, cond, {
            "physics_targets": physics_targets,
            "negative_motion": negative_batch["motion"],
        }


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
        *,
        aux_batch_preprocessor=None,
        physics_target_cache: PhysicsTargetLRUCache | None = None,
        skeleton_lookup,
        species_vocab: LabelVocab,
        action_vocab: LabelVocab,
    ) -> None:
        self.args = args
        self.ml_platform = ml_platform
        self.data_loader = data_loader
        self.aux_batch_preprocessor = aux_batch_preprocessor
        self.physics_target_cache = physics_target_cache
        self.skeleton_lookup = skeleton_lookup
        self.species_vocab = species_vocab
        self.action_vocab = action_vocab
        self.num_species = args.num_species
        self.num_actions = args.num_actions
        self.metadata_dim = args.metadata_feature_dim

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
        
        self.metadata_feature_lookup = build_metadata_feature_lookup(
            self.skeleton_lookup,
            max_joints=self.args.max_joints,
            device=self.device,
            dtype=torch.float32,
        )

        self.model = MotionScorerNet(
            feature_dim=args.feature_dim,
            d_model=args.d_model,
            latent_dim=args.latent_dim,
            num_conv_layers=args.num_conv_layers,
            kernel_size=args.kernel_size,
            max_joints=args.max_joints,
            num_species=self.num_species,
            num_actions=self.num_actions,
            metadata_dim=self.metadata_dim,
            metadata_hidden_dim=args.metadata_hidden_dim,
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
            log_norms=False,
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

    def _encode_batch_labels(self, cond: dict) -> tuple[torch.Tensor, torch.Tensor]:
        species_ids = self.species_vocab.encode_many(cond["y"].get("species_label", []), device=self.device)
        action_ids = self.action_vocab.encode_many(cond["y"].get("action_label", []), device=self.device)
        return species_ids, action_ids

    def train_step(self, motion: torch.Tensor, cond: dict, current_step: int, precomputed_aux: dict | None = None) -> dict[str, float]:
        n_joints = cond["y"]["n_joints"]
        lengths = cond["y"]["lengths"]
        object_types = cond["y"].get("object_type")
        species_ids, action_ids = self._encode_batch_labels(cond)
        aux_factor = get_auxiliary_loss_factor(self.args, current_step)
        use_auxiliary_branches = aux_factor > 0.0

        metadata_features = None
        physics_targets = None
        negative_batch = None
        if use_auxiliary_branches:
            metadata_features = torch.stack(
                [self.metadata_feature_lookup[str(object_type)] for object_type in object_types],
                dim=0,
            )
            if metadata_features.dtype != motion.dtype:
                metadata_features = metadata_features.to(dtype=motion.dtype)
            if precomputed_aux is not None:
                physics_targets = precomputed_aux["physics_targets"]
                negative_batch = {
                    "motion": precomputed_aux["negative_motion"],
                }
            else:
                if self.physics_target_cache is not None:
                    physics_targets = self.physics_target_cache.get_batch(motion, cond, self.skeleton_lookup)
                else:
                    physics_targets = extract_physics_features(
                        motion,
                        n_joints,
                        lengths,
                        object_types,
                        self.skeleton_lookup,
                    ).detach()

        self.mp_trainer.zero_grad()
        with self.autocast_context():
            clean_outputs = self.model(
                motion,
                n_joints,
                lengths,
                metadata_features=metadata_features,
                return_disc_logits=use_auxiliary_branches,
                return_phys_features=use_auxiliary_branches,
            )

            species_loss = F.cross_entropy(clean_outputs["species_logits"], species_ids)
            action_loss = F.cross_entropy(clean_outputs["action_logits"], action_ids)
            if use_auxiliary_branches:
                vicreg_features = normalize_density_embeddings(clean_outputs["latents"])
                disc_positive_loss = F.binary_cross_entropy_with_logits(
                    clean_outputs["disc_logits"],
                    torch.ones_like(clean_outputs["disc_logits"]),
                )
                phys_loss = F.mse_loss(clean_outputs["phys_features"], physics_targets)
                vic_variance_loss = _variance_floor_loss(vicreg_features, float(self.args.quality_variance_floor))
                vic_covariance_loss = _covariance_loss(vicreg_features)

                if negative_batch is None:
                    negative_batch = generate_biomechanical_negative_batch(
                        motion,
                        n_joints,
                        lengths,
                        object_types,
                        self.skeleton_lookup,
                        feature_std=cond["y"].get("std"),
                        negative_kinds=NEGATIVE_KINDS,
                    )
                negative_outputs = self.model(
                    negative_batch["motion"],
                    n_joints,
                    lengths,
                    metadata_features=metadata_features,
                    return_species_logits=False,
                    return_action_logits=False,
                    return_phys_features=False,
                )
                disc_negative_loss = F.binary_cross_entropy_with_logits(
                    negative_outputs["disc_logits"],
                    torch.zeros_like(negative_outputs["disc_logits"]),
                )
            else:
                zero = motion.new_zeros(())
                disc_positive_loss = zero
                disc_negative_loss = zero
                phys_loss = zero
                vic_variance_loss = zero
                vic_covariance_loss = zero
                negative_outputs = None

            loss = (
                float(self.args.lambda_species) * species_loss
                + float(self.args.lambda_action) * action_loss
                + aux_factor * float(self.args.lambda_disc_p) * disc_positive_loss
                + aux_factor * float(self.args.lambda_disc_n) * disc_negative_loss
                + aux_factor * float(self.args.lambda_phys) * phys_loss
                + aux_factor * float(self.args.lambda_vic_var) * vic_variance_loss
                + aux_factor * float(self.args.lambda_vic_cov) * vic_covariance_loss
            )

        self.mp_trainer.backward(loss)
        took_step = self.mp_trainer.optimize(self.opt, self.lr_scheduler)
        if took_step and self.model_avg is not None:
            update_ema(self.model_avg.parameters(), self.model.parameters(), rate=self.args.ema_decay)

        with torch.no_grad():
            species_accuracy = (clean_outputs["species_logits"].argmax(dim=-1) == species_ids).float().mean()
            action_accuracy = (clean_outputs["action_logits"].argmax(dim=-1) == action_ids).float().mean()
            if use_auxiliary_branches:
                clean_disc_prob = torch.sigmoid(clean_outputs["disc_logits"]).mean()
                negative_disc_prob = torch.sigmoid(negative_outputs["disc_logits"]).mean()
            else:
                clean_disc_prob = motion.new_zeros(())
                negative_disc_prob = motion.new_zeros(())

        return {
            "loss": float(loss.detach().item()),
            "species_loss": float(species_loss.detach().item()),
            "action_loss": float(action_loss.detach().item()),
            "disc_positive_loss": float(disc_positive_loss.detach().item()),
            "disc_negative_loss": float(disc_negative_loss.detach().item()),
            "phys_loss": float(phys_loss.detach().item()),
            "vic_variance_loss": float(vic_variance_loss.detach().item()),
            "vic_covariance_loss": float(vic_covariance_loss.detach().item()),
            "species_accuracy": float(species_accuracy.detach().item()),
            "action_accuracy": float(action_accuracy.detach().item()),
            "clean_disc_prob": float(clean_disc_prob.detach().item()),
            "negative_disc_prob": float(negative_disc_prob.detach().item()),
            "aux_factor": float(aux_factor),
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
        running_metrics: dict[str, list[float]] = {}
        data_iter = iter(self.data_loader)
        timing_log_interval = max(1, int(getattr(self.args, "timing_log_interval", self.args.log_interval)))
        cls_warmup_steps = max(0, int(getattr(self.args, "cls_warmup_steps", 0)))
        aux_warmup_steps = max(0, int(getattr(self.args, "full_loss_warmup_steps", 0)))
        aux_warmup_start_step = cls_warmup_steps + 1
        aux_full_weight_step = cls_warmup_steps + aux_warmup_steps
        timing_totals = {
            "data_wait_s": 0.0,
            "host_to_device_s": 0.0,
            "step_s": 0.0,
            "loop_s": 0.0,
        }
        timing_steps = 0
        timing_samples = 0

        next_metric_log = min(int(self.args.log_interval), int(self.args.num_steps))
        next_timing_log = min(int(timing_log_interval), int(self.args.num_steps))
        print(
            f"Motion scorer training loop started: next_metrics_step={next_metric_log}, "
            f"next_timing_step={next_timing_log}"
        )
        if cls_warmup_steps > 0 or aux_warmup_steps > 0:
            print("=" * 100)
            if cls_warmup_steps > 0 and aux_warmup_steps > 0:
                print(
                    f"Warmup schedule: classification-only step[1]-step[{cls_warmup_steps}], "
                    f"auxiliary ramp step[{aux_warmup_start_step}]-step[{aux_full_weight_step}]"
                )
            elif cls_warmup_steps > 0:
                print(
                    f"Warmup schedule: classification-only step[1]-step[{cls_warmup_steps}], "
                    f"full auxiliary weights enable at step[{aux_warmup_start_step}]"
                )
            else:
                print(f"Warmup schedule: auxiliary ramp step[1]-step[{aux_full_weight_step}]")
            print("=" * 100)

        while completed_steps < self.args.num_steps:
            loop_start = time.perf_counter()
            fetch_start = time.perf_counter()
            next_step = completed_steps + 1
            if cls_warmup_steps > 0 and next_step == aux_warmup_start_step:
                print("=" * 100)
                print(
                    f"Warmup boundary: classification-only warmup finished at step[{cls_warmup_steps}]; "
                    f"auxiliary loss ramp starts at step[{next_step}]"
                )
                print("=" * 100)
            if aux_warmup_steps > 0 and next_step == aux_full_weight_step:
                print("=" * 100)
                print(
                    f"Warmup boundary: auxiliary loss ramp ends at step[{aux_full_weight_step}]; "
                    "full auxiliary weights are now active"
                )
                print("=" * 100)
            if self.aux_batch_preprocessor is not None:
                self.aux_batch_preprocessor.set_enabled(get_auxiliary_loss_factor(self.args, next_step) > 0.0)
            if completed_steps == self.resume_completed_steps:
                print("Motion scorer waiting for first batch...")
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.data_loader)
                batch = next(data_iter)
            if isinstance(batch, tuple) and len(batch) == 3:
                motion, cond, precomputed_aux = batch
            else:
                motion, cond = batch
                precomputed_aux = None
            data_wait_s = time.perf_counter() - fetch_start

            host_to_device_start = time.perf_counter()
            motion = motion.to(self.device, non_blocking=self.non_blocking)
            cond = move_cond_to_device(cond, self.device, self.non_blocking)
            precomputed_aux = move_aux_to_device(precomputed_aux, self.device, self.non_blocking)
            host_to_device_s = time.perf_counter() - host_to_device_start

            step_start = time.perf_counter()
            step_metrics = self.train_step(motion, cond, completed_steps + 1, precomputed_aux=precomputed_aux)
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
                    "step[{}]: total_loss[{:.6f}] species_ce[{:.6f}] action_ce[{:.6f}] disc_pos[{:.6f}] disc_neg[{:.6f}] phys[{:.6f}] sp_acc[{:.3f}] act_acc[{:.3f}] aux[{:.3f}]".format(
                        completed_steps,
                        mean_metrics.get("loss", 0.0),
                        mean_metrics.get("species_loss", 0.0),
                        mean_metrics.get("action_loss", 0.0),
                        mean_metrics.get("disc_positive_loss", 0.0),
                        mean_metrics.get("disc_negative_loss", 0.0),
                        mean_metrics.get("phys_loss", 0.0),
                        mean_metrics.get("species_accuracy", 0.0),
                        mean_metrics.get("action_accuracy", 0.0),
                        mean_metrics.get("aux_factor", 0.0),
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
                    "timing[{}]: data_wait_ms[{:.2f}] host_to_device_ms[{:.2f}] step_ms[{:.2f}] total_ms[{:.2f}] data_wait_pct[{:.1f}] step_pct[{:.1f}] samples_per_s[{:.1f}]".format(
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


def _mahalanobis_distance_np(values: np.ndarray, mean: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    diff = values - mean[None, :]
    distances_sq = np.einsum("bi,ij,bj->b", diff, cov_inv, diff)
    return np.sqrt(np.clip(distances_sq, a_min=0.0, a_max=None))


def compute_and_save_train_stats(args, model: MotionScorerNet, device: torch.device, autocast_dtype, amp_enabled: bool, *, skeleton_lookup) -> None:
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
    latent_batches = []
    physics_batches = []
    with torch.no_grad():
        for motion, cond in tqdm(loader, desc="Caching scorer stats"):
            motion = motion.to(device, non_blocking=device.type == "cuda")
            cond = move_cond_to_device(cond, device, device.type == "cuda")
            object_types = cond["y"].get("object_type")
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled):
                latents = model.encode(motion, cond["y"]["n_joints"], cond["y"]["lengths"])
            latent_batches.append(latents.detach().float().cpu().numpy())
            physics_batches.append(
                extract_physics_features(
                    motion,
                    cond["y"]["n_joints"],
                    cond["y"]["lengths"],
                    object_types,
                    skeleton_lookup,
                ).detach().float().cpu().numpy()
            )

    latents = np.concatenate(latent_batches, axis=0).astype(np.float64, copy=False)
    physics = np.concatenate(physics_batches, axis=0).astype(np.float64, copy=False)

    component_count = min(max(1, int(args.gmm_components)), latents.shape[0])
    gmm = GaussianMixture(
        n_components=component_count,
        covariance_type=str(args.gmm_covariance_type),
        reg_covar=float(args.stats_eps),
        random_state=int(args.seed),
    )
    gmm.fit(latents)
    density_values = gmm.score_samples(latents)

    mu_phys = physics.mean(axis=0)
    if physics.shape[0] > 1:
        sigma_phys = np.cov(physics, rowvar=False)
    else:
        sigma_phys = np.eye(physics.shape[1], dtype=np.float64)
    sigma_phys = np.atleast_2d(sigma_phys)
    sigma_phys += np.eye(sigma_phys.shape[0], dtype=np.float64) * float(args.stats_eps)
    sigma_phys_inv = np.linalg.pinv(sigma_phys)
    phys_values = -_mahalanobis_distance_np(physics, mu_phys, sigma_phys_inv)

    latest_checkpoint = find_latest_checkpoint(args.save_dir, prefix="model")
    checkpoint_step = parse_checkpoint_number(latest_checkpoint) if latest_checkpoint else 0
    stats = {
        "gmm_covariance_type": str(args.gmm_covariance_type),
        "gmm_means": gmm.means_.astype(np.float32),
        "gmm_covariances": gmm.covariances_.astype(np.float32),
        "gmm_weights": gmm.weights_.astype(np.float32),
        "density_percentiles": np.percentile(density_values, np.arange(101)).astype(np.float32),
        "mu_phys": mu_phys.astype(np.float32),
        "sigma_phys_inv": sigma_phys_inv.astype(np.float32),
        "phys_percentiles": np.percentile(phys_values, np.arange(101)).astype(np.float32),
        "score_alpha": np.asarray(args.score_alpha, dtype=np.float32),
        "checkpoint_path": latest_checkpoint,
        "checkpoint_step": checkpoint_step,
        "stats_split": stats_split,
        "num_samples": int(latents.shape[0]),
    }
    np.save(os.path.join(args.save_dir, "train_stats.npy"), stats, allow_pickle=True)

    summary = {
        "checkpoint_path": latest_checkpoint,
        "checkpoint_step": checkpoint_step,
        "stats_split": stats_split,
        "num_samples": int(latents.shape[0]),
        "gmm_components": int(component_count),
        "gmm_covariance_type": str(args.gmm_covariance_type),
        "score_alpha": [float(value) for value in args.score_alpha],
    }
    with open(os.path.join(args.save_dir, "train_stats_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    model.train()


def prepare_training_assets(args):
    dataset_dir = getattr(args, "data_dir", "") or None
    skeleton_lookup = load_skeleton_metadata(dataset_dir)
    species_vocab, action_vocab = build_label_vocabs(dataset_dir)
    args.num_species = species_vocab.size
    args.num_actions = action_vocab.size
    args.species_vocab = list(species_vocab.labels)
    args.action_vocab = list(action_vocab.labels)
    args.metadata_feature_dim = metadata_feature_dim(args.max_joints)
    return skeleton_lookup, species_vocab, action_vocab


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

    skeleton_lookup, species_vocab, action_vocab = prepare_training_assets(args)
    physics_target_cache = None
    if int(getattr(args, "motion_cache_size", 0)) > 0:
        physics_target_cache = PhysicsTargetLRUCache(getattr(args, "motion_cache_size", 0))
    batch_transform = None
    if int(getattr(args, "num_workers", 0)) == 0 and int(getattr(args, "main_process_prefetch_batches", 0)) > 0:
        batch_transform = MotionScorerAuxBatchPreprocessor(
            skeleton_lookup,
            physics_target_cache=physics_target_cache,
        )

    ml_platform_type = eval(args.ml_platform_type)
    ml_platform = ml_platform_type(save_dir=save_dir)
    ml_platform.report_args(args, name="Args")

    with open(os.path.join(save_dir, "args.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=4, sort_keys=True)

    data_loader_start = time.perf_counter()
    data_loader = create_data_loader(
        args,
        args.train_split,
        shuffle=True,
        drop_last=True,
        balanced=args.balanced,
        batch_transform=batch_transform,
    )
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
    trainer = MotionScorerTrainer(
        args,
        ml_platform,
        data_loader,
        aux_batch_preprocessor=batch_transform,
        physics_target_cache=physics_target_cache,
        skeleton_lookup=skeleton_lookup,
        species_vocab=species_vocab,
        action_vocab=action_vocab,
    )
    trainer_init_s = time.perf_counter() - trainer_init_start
    print(f"Motion scorer startup: trainer_init_s={trainer_init_s:.2f} total_startup_s={time.perf_counter() - startup_start:.2f}")
    ml_platform.watch_model(trainer.model)
    final_model = trainer.run()
    compute_and_save_train_stats(args, final_model, trainer.device, trainer.autocast_dtype, trainer.amp_enabled, skeleton_lookup=skeleton_lookup)
    ml_platform.close()


if __name__ == "__main__":
    main()
