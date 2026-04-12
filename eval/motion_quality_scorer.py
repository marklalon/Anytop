from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.truebones.offline_reference_dataset import load_cond_dict
from model.motion_autoencoder import MotionAutoencoder, masked_mse_per_sample


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
            d_model=int(self.args.get("d_model", 256)),
            latent_dim=int(self.args.get("latent_dim", 512)),
            num_conv_layers=int(self.args.get("num_conv_layers", 4)),
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

        self.mean = torch.as_tensor(self.train_stats["mean"], dtype=torch.float32, device=self.device)
        self.cov_inv = torch.as_tensor(self.train_stats["cov_inv"], dtype=torch.float32, device=self.device)
        self.recon_threshold = float(self.train_stats["recon_threshold"])
        self.density_threshold = float(self.train_stats["density_threshold"])
        self.alpha = float(self.train_stats.get("alpha", 0.7))
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
            recon_error = masked_mse_per_sample(motion, reconstruction, n_joints, lengths)
            mahal_distance = _mahalanobis_distance(latents.float(), self.mean, self.cov_inv)

        recon_score = 1.0 - torch.clamp(recon_error / max(self.recon_threshold, 1e-8), min=0.0, max=1.0)
        density_score = 1.0 - torch.clamp(mahal_distance / max(self.density_threshold, 1e-8), min=0.0, max=1.0)
        quality_score = self.alpha * recon_score + (1.0 - self.alpha) * density_score

        return {
            "quality_score": quality_score.detach().cpu(),
            "recon_score": recon_score.detach().cpu(),
            "density_score": density_score.detach().cpu(),
            "recon_error": recon_error.detach().cpu(),
            "mahal_distance": mahal_distance.detach().cpu(),
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
                        "mahal_distance": float(batch_result["mahal_distance"][batch_index].item()),
                        "object_type": object_types[batch_index],
                        "motion_name": motion_names[batch_index],
                    }
                )
        return scored_samples