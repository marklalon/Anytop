from __future__ import annotations

import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.truebones.offline_reference_dataset import load_cond_dict
from model.motion_autoencoder import MotionAutoencoder, motion_perceptual_recon_error_per_sample


def _resolve_checkpoint_path(checkpoint_dir: str | os.PathLike[str]) -> Path:
    checkpoint_path = Path(checkpoint_dir)
    if checkpoint_path.is_file():
        return checkpoint_path
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    candidates = []
    for file_path in checkpoint_path.glob("model*.pt"):
        match = re.fullmatch(r"model(\d+)\.pt", file_path.name)
        if match:
            candidates.append((int(match.group(1)), file_path))
    if not candidates:
        raise FileNotFoundError(f"No model checkpoint was found under: {checkpoint_path}")
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _checkpoint_step_from_path(checkpoint_path: Path) -> int:
    match = re.fullmatch(r"model(\d+)\.pt", checkpoint_path.name)
    if match is None:
        return 0
    return int(match.group(1))


def _select_model_state_dict(state_dict: dict[str, Any], prefer_ema: bool) -> dict[str, torch.Tensor]:
    if prefer_ema and "model_avg" in state_dict:
        return state_dict["model_avg"]
    if "model" in state_dict:
        return state_dict["model"]
    return state_dict


def _mahalanobis_distance(
    latents: torch.Tensor,
    mean: torch.Tensor,
    cov_inv: torch.Tensor,
) -> torch.Tensor:
    diff = latents - mean.unsqueeze(0)
    distances_sq = torch.einsum("bi,ij,bj->b", diff, cov_inv, diff)
    return torch.sqrt(torch.clamp(distances_sq, min=0.0))


def _diagonal_mahalanobis_distance(
    latents: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
) -> torch.Tensor:
    diff = latents - mean.unsqueeze(0)
    distances_sq = ((diff * diff) / var.unsqueeze(0)).sum(dim=1)
    return torch.sqrt(torch.clamp(distances_sq, min=0.0))


def _knn_distance(
    latents: torch.Tensor,
    reference_latents: torch.Tensor,
    k: int,
) -> torch.Tensor:
    if reference_latents.ndim != 2 or latents.ndim != 2:
        raise ValueError("Expected [N, D] latent tensors for kNN density scoring.")
    effective_k = max(1, min(int(k), int(reference_latents.shape[0])))
    distances = torch.cdist(latents, reference_latents)
    knn_distances, _ = torch.topk(distances, k=effective_k, largest=False, dim=1)
    return knn_distances.mean(dim=1)


def _empirical_survival_score(values: torch.Tensor, reference_values: torch.Tensor) -> torch.Tensor:
    if reference_values.numel() == 0:
        raise ValueError("reference_values must be non-empty for empirical score calibration.")
    sorted_reference = torch.sort(reference_values.flatten())[0]
    insertion_indices = torch.searchsorted(sorted_reference, values.flatten(), right=False)
    tail_counts = sorted_reference.numel() - insertion_indices
    scores = tail_counts.to(dtype=torch.float32) / float(sorted_reference.numel())
    return scores.view_as(values)


def _smooth_log_reference_score(values: torch.Tensor, reference_values: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    if reference_values.numel() == 0:
        raise ValueError("reference_values must be non-empty for smooth score calibration.")
    safe_reference = reference_values.flatten().float().clamp_min(eps)
    safe_values = values.float().clamp_min(eps)
    log_reference = torch.log(safe_reference)
    q50 = torch.quantile(log_reference, 0.50)
    q95 = torch.quantile(log_reference, 0.95)
    scale = torch.clamp(q95 - q50, min=0.05)
    return torch.sigmoid((q95 - torch.log(safe_values)) / scale)


def _hybrid_reference_score(
    values: torch.Tensor,
    reference_values: torch.Tensor,
    *,
    empirical_weight: float,
) -> torch.Tensor:
    empirical = _empirical_survival_score(values, reference_values)
    smooth = _smooth_log_reference_score(values, reference_values)
    blend = float(min(max(empirical_weight, 0.0), 1.0))
    return blend * empirical + (1.0 - blend) * smooth


def _compute_recon_error(args: dict[str, Any], motion: torch.Tensor, reconstruction: torch.Tensor, n_joints: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    return motion_perceptual_recon_error_per_sample(
        motion,
        reconstruction,
        n_joints,
        lengths,
        position_weight=float(args.get("recon_position_weight", 0.35)),
        velocity_weight=float(args.get("recon_velocity_weight", 0.30)),
        acceleration_weight=float(args.get("recon_acceleration_weight", 0.20)),
        blur_weight=float(args.get("recon_blur_weight", 0.15)),
    )


def _normalize_density_embeddings(latents: torch.Tensor) -> torch.Tensor:
    return F.normalize(latents.float(), dim=-1)


def _to_1d_long_tensor(values, batch_size: int, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.long, device=device)
    if tensor.ndim == 0:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[0] != batch_size:
        raise ValueError(f"Expected batch size {batch_size}, got {tensor.shape[0]}")
    return tensor


class MotionQualityScorer:
    def __init__(
        self,
        checkpoint_dir: str | os.PathLike[str],
        device: str = "cuda",
        prefer_ema: bool = True,
    ) -> None:
        self.checkpoint_path = _resolve_checkpoint_path(checkpoint_dir)
        self.checkpoint_dir = self.checkpoint_path.parent
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            requested_device = torch.device("cpu")
        self.device = requested_device

        args_path = self.checkpoint_dir / "args.json"
        if not args_path.exists():
            raise FileNotFoundError(f"args.json was not found next to the checkpoint: {args_path}")
        with open(args_path, "r", encoding="utf-8") as handle:
            self.args = json.load(handle)

        self.autoencoder = MotionAutoencoder(
            feature_dim=int(self.args.get("feature_dim", 13)),
            d_model=int(self.args.get("d_model", 128)),
            latent_dim=int(self.args.get("latent_dim", 64)),
            num_conv_layers=int(self.args.get("num_conv_layers", 3)),
            decoder_num_conv_layers=int(self.args.get("decoder_num_conv_layers", 0)),
            kernel_size=int(self.args.get("kernel_size", 5)),
            max_joints=int(self.args.get("max_joints", 143)),
            max_frames=int(self.args.get("num_frames", 120)),
        )
        checkpoint_payload = torch.load(self.checkpoint_path, map_location="cpu")
        model_state = _select_model_state_dict(checkpoint_payload, prefer_ema=prefer_ema)
        self.autoencoder.load_state_dict(model_state, strict=True)
        self.autoencoder.to(self.device)
        self.autoencoder.eval()

        stats_path = self.checkpoint_dir / "train_stats.npy"
        if not stats_path.exists():
            raise FileNotFoundError(f"train_stats.npy was not found next to the checkpoint: {stats_path}")
        self.train_stats = np.load(stats_path, allow_pickle=True).item()

        stats_checkpoint_step = int(self.train_stats.get("checkpoint_step", 0) or 0)
        loaded_checkpoint_step = _checkpoint_step_from_path(self.checkpoint_path)
        if stats_checkpoint_step and loaded_checkpoint_step and stats_checkpoint_step != loaded_checkpoint_step:
            raise RuntimeError(
                "train_stats.npy was generated from a different checkpoint step "
                f"(stats={stats_checkpoint_step}, model={loaded_checkpoint_step}). "
                "Recompute scorer stats for the current checkpoint before scoring."
            )
        if not stats_checkpoint_step:
            warnings.warn(
                "train_stats.npy does not record which checkpoint generated it. "
                "If density_score looks wrong, recompute scorer stats for the current checkpoint.",
                stacklevel=2,
            )

        self.mean = torch.as_tensor(self.train_stats["mean"], dtype=torch.float32, device=self.device)
        self.cov_inv = torch.as_tensor(self.train_stats["cov_inv"], dtype=torch.float32, device=self.device)
        self.var = torch.as_tensor(
            self.train_stats.get("var", np.ones_like(self.train_stats["mean"])),
            dtype=torch.float32,
            device=self.device,
        )
        self.reference_recon_errors = torch.as_tensor(
            self.train_stats.get("recon_errors", np.asarray([], dtype=np.float32)),
            dtype=torch.float32,
            device=self.device,
        )
        self.reference_latents = torch.as_tensor(
            self.train_stats.get("train_latents", np.asarray([], dtype=np.float32)),
            dtype=torch.float32,
            device=self.device,
        )
        self.reference_knn_distances = torch.as_tensor(
            self.train_stats.get("knn_distances", np.asarray([], dtype=np.float32)),
            dtype=torch.float32,
            device=self.device,
        )
        self.recon_threshold = float(self.train_stats["recon_threshold"])
        self.density_threshold = float(self.train_stats["density_threshold"])
        self.alpha = float(self.train_stats.get("alpha", 0.7))
        self.density_mode = str(self.train_stats.get("density_mode", "diagonal_mahalanobis"))
        self.density_knn_k = int(self.train_stats.get("density_knn_k", 5))
        self.density_embedding_kind = str(self.train_stats.get("density_embedding_kind", "raw_latent"))
        self.recon_score_calibration = str(self.train_stats.get("recon_score_calibration", "hybrid_empirical_logistic_v1"))
        self.recon_empirical_weight = float(self.train_stats.get("recon_empirical_weight", 0.5))
        self.feature_dim = int(self.args.get("feature_dim", 13))
        self.max_joints = int(self.args.get("max_joints", 143))
        self.max_frames = int(self.args.get("num_frames", 120))

    def score(
        self,
        motion: torch.Tensor | np.ndarray,
        n_joints,
        lengths,
    ) -> dict[str, torch.Tensor]:
        if isinstance(motion, np.ndarray):
            motion = torch.from_numpy(motion)
        motion = motion.to(self.device, dtype=torch.float32)
        if motion.ndim == 3:
            motion = motion.unsqueeze(0)
        if motion.ndim != 4:
            raise ValueError(f"Expected motion to have 4 dims [B, J, F, T], got {tuple(motion.shape)}")
        if motion.shape[1] > self.max_joints:
            raise ValueError(f"Expected at most {self.max_joints} joints, got {motion.shape[1]}")
        if motion.shape[2] != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {motion.shape[2]}")
        if motion.shape[-1] > self.max_frames:
            raise ValueError(f"Expected at most {self.max_frames} frames, got {motion.shape[-1]}")

        batch_size = motion.shape[0]
        n_joints = _to_1d_long_tensor(n_joints, batch_size, self.device)
        lengths = _to_1d_long_tensor(lengths, batch_size, self.device)

        with torch.no_grad():
            reconstruction, latents = self.autoencoder(motion, n_joints, lengths)
            recon_error = _compute_recon_error(self.args, motion, reconstruction, n_joints, lengths)
            density_embeddings = latents.float()
            if self.density_embedding_kind == "normalized_latent":
                density_embeddings = _normalize_density_embeddings(density_embeddings)
            if self.density_mode == "knn" and self.reference_latents.numel() and self.reference_knn_distances.numel():
                density_distance = _knn_distance(density_embeddings, self.reference_latents.float(), self.density_knn_k)
            elif self.density_mode == "diagonal_mahalanobis":
                density_distance = _diagonal_mahalanobis_distance(density_embeddings, self.mean, self.var)
            else:
                density_distance = _mahalanobis_distance(density_embeddings, self.mean, self.cov_inv)

        if self.reference_recon_errors.numel():
            if self.recon_score_calibration == "hybrid_empirical_logistic_v1":
                recon_score = _hybrid_reference_score(
                    recon_error.float(),
                    self.reference_recon_errors.float(),
                    empirical_weight=self.recon_empirical_weight,
                )
            else:
                recon_score = _empirical_survival_score(recon_error.float(), self.reference_recon_errors.float())
        else:
            recon_score = 1.0 - torch.clamp(recon_error / max(self.recon_threshold, 1e-8), min=0.0, max=1.0)

        if self.density_mode == "knn" and self.reference_knn_distances.numel():
            density_score = _empirical_survival_score(density_distance.float(), self.reference_knn_distances.float())
        else:
            density_score = 1.0 - torch.clamp(density_distance / max(self.density_threshold, 1e-8), min=0.0, max=1.0)
        quality_score = self.alpha * recon_score + (1.0 - self.alpha) * density_score

        return {
            "quality_score": quality_score.detach().cpu(),
            "recon_score": recon_score.detach().cpu(),
            "density_score": density_score.detach().cpu(),
            "recon_error": recon_error.detach().cpu(),
            "density_distance": density_distance.detach().cpu(),
            "mahal_distance": density_distance.detach().cpu(),
            "density_mode": self.density_mode,
        }

    def score_npy(
        self,
        npy_path: str | os.PathLike[str],
        object_type: str,
        cond_dict: dict[str, dict[str, np.ndarray]] | None = None,
    ) -> dict[str, torch.Tensor | str]:
        if cond_dict is None:
            cond_dict = load_cond_dict()
        if object_type not in cond_dict:
            available = ", ".join(sorted(cond_dict.keys()))
            raise KeyError(f"Unknown object_type '{object_type}'. Available types: {available}")

        motion_np = np.load(npy_path)
        if motion_np.ndim != 3:
            raise ValueError(f"Expected .npy motion to have shape [T, J, F], got {motion_np.shape}")
        if motion_np.shape[2] != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {motion_np.shape[2]}")
        if motion_np.shape[1] > self.max_joints:
            raise ValueError(f"Expected at most {self.max_joints} joints, got {motion_np.shape[1]}")

        object_cond = cond_dict[object_type]
        mean = np.asarray(object_cond["mean"], dtype=np.float32)
        std = np.asarray(object_cond["std"], dtype=np.float32)
        normalized = (motion_np.astype(np.float32) - mean[None, :, :]) / (std[None, :, :] + 1e-6)
        normalized = np.nan_to_num(normalized)

        motion_tensor = torch.from_numpy(normalized).permute(1, 2, 0).unsqueeze(0)
        result = self.score(
            motion_tensor,
            n_joints=[motion_np.shape[1]],
            lengths=[motion_np.shape[0]],
        )
        result["object_type"] = object_type
        result["path"] = str(npy_path)
        return result

    def score_batch_from_dataloader(self, dataloader) -> list[dict[str, Any]]:
        scored_samples: list[dict[str, Any]] = []
        for motion, cond in dataloader:
            batch_result = self.score(motion, cond["y"]["n_joints"], cond["y"]["lengths"])
            batch_size = motion.shape[0]
            object_types = cond["y"].get("object_type", [None] * batch_size)
            motion_names = cond["y"].get("motion_name", [None] * batch_size)
            for batch_index in range(batch_size):
                scored_samples.append(
                    {
                        "quality_score": float(batch_result["quality_score"][batch_index].item()),
                        "recon_score": float(batch_result["recon_score"][batch_index].item()),
                        "density_score": float(batch_result["density_score"][batch_index].item()),
                        "recon_error": float(batch_result["recon_error"][batch_index].item()),
                        "density_distance": float(batch_result["density_distance"][batch_index].item()),
                        "mahal_distance": float(batch_result["mahal_distance"][batch_index].item()),
                        "object_type": object_types[batch_index],
                        "motion_name": motion_names[batch_index],
                    }
                )
        return scored_samples