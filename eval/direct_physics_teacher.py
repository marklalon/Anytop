from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from data_loaders.skeleton_metadata import load_skeleton_metadata
from data_loaders.truebones.offline_reference_dataset import resolve_dataset_root
from eval.physics_features import extract_physics_features


def _resolve_checkpoint_dir(checkpoint_dir: str | os.PathLike[str]) -> Path:
    path = Path(checkpoint_dir)
    if path.is_file():
        path = path.parent
    if not path.is_dir():
        raise FileNotFoundError(f"Physics teacher checkpoint directory does not exist: {path}")
    return path


def _mahalanobis_distance_sq(values: torch.Tensor, mean: torch.Tensor, cov_inv: torch.Tensor) -> torch.Tensor:
    diff = values - mean.unsqueeze(0)
    return torch.einsum("bi,ij,bj->b", diff, cov_inv, diff)


def _percentile_score(values: torch.Tensor, reference_percentiles: torch.Tensor) -> torch.Tensor:
    bins = torch.searchsorted(reference_percentiles, values.float(), right=False).float() / 100.0
    return bins.clamp(0.0, 1.0)


class DirectPhysicsTeacher:

    def __init__(
        self,
        checkpoint_dir: str | os.PathLike[str],
        *,
        device: str | torch.device = "cuda",
        dataset_dir: str | os.PathLike[str] | None = None,
        features_compute_device: str | torch.device | None = None,
    ) -> None:
        checkpoint_path = _resolve_checkpoint_dir(checkpoint_dir)
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            requested_device = torch.device("cpu")
        self.device = requested_device
        self.features_compute_device = (
            torch.device(features_compute_device) if features_compute_device is not None else None
        )

        args_path = checkpoint_path / "args.json"
        if not args_path.exists():
            raise FileNotFoundError(f"args.json was not found next to physics teacher stats: {args_path}")
        with open(args_path, "r", encoding="utf-8") as handle:
            args = json.load(handle)

        stats_path = checkpoint_path / "train_stats.npy"
        if not stats_path.exists():
            raise FileNotFoundError(f"train_stats.npy was not found next to physics teacher stats: {stats_path}")
        train_stats = np.load(stats_path, allow_pickle=True).item()

        required_keys = {"mu_phys", "sigma_phys_inv", "phys_percentiles"}
        missing = sorted(required_keys.difference(train_stats.keys()))
        if missing:
            raise ValueError(
                "Physics teacher stats are incomplete. Missing keys in train_stats.npy: "
                f"{missing}"
            )

        dataset_root = resolve_dataset_root(dataset_dir or args.get("data_dir") or None)
        self.skeleton_lookup = load_skeleton_metadata(dataset_root)
        self.mu_phys = torch.as_tensor(train_stats["mu_phys"], dtype=torch.float32, device=self.device)
        self.sigma_phys_inv = torch.as_tensor(train_stats["sigma_phys_inv"], dtype=torch.float32, device=self.device)
        self.phys_percentiles = torch.as_tensor(train_stats["phys_percentiles"], dtype=torch.float32, device=self.device)
        self.feature_scale = torch.sqrt(torch.diagonal(self.sigma_phys_inv).clamp_min(1e-6))

    def compute_target_features(
        self,
        target_motion: torch.Tensor,
        *,
        n_joints: torch.Tensor,
        lengths: torch.Tensor,
        object_types: Sequence[str],
    ) -> torch.Tensor:
        """Compute physics features for the target motion (no gradient).

        The caller should cache the result and pass it to ``compute_losses``
        via the ``target_features`` argument to avoid redundant computation.
        """
        with torch.no_grad():
            return extract_physics_features(
                target_motion.detach(),
                n_joints.detach(),
                lengths.detach(),
                object_types,
                self.skeleton_lookup,
                compute_device=self.features_compute_device,
            ).float()

    def compute_losses(
        self,
        pred_motion: torch.Tensor,
        target_motion: torch.Tensor,
        *,
        n_joints: torch.Tensor,
        lengths: torch.Tensor,
        object_types: Sequence[str],
        target_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pred_features = extract_physics_features(
            pred_motion,
            n_joints,
            lengths,
            object_types,
            self.skeleton_lookup,
            differentiable=True,
            compute_device=self.features_compute_device,
        ).float()

        if target_features is None:
            target_features = self.compute_target_features(
                target_motion,
                n_joints=n_joints,
                lengths=lengths,
                object_types=object_types,
            )

        scale = self.feature_scale.unsqueeze(0)
        pred_scaled = pred_features * scale
        target_scaled = target_features * scale
        feature_loss = F.smooth_l1_loss(pred_scaled, target_scaled, reduction="none").mean(dim=1)

        pred_distance_sq = _mahalanobis_distance_sq(pred_features, self.mu_phys, self.sigma_phys_inv)
        target_distance_sq = _mahalanobis_distance_sq(target_features, self.mu_phys, self.sigma_phys_inv)
        margin_loss = torch.relu(pred_distance_sq - target_distance_sq)

        pred_distance = torch.sqrt(pred_distance_sq.clamp_min(0.0))
        target_distance = torch.sqrt(target_distance_sq.clamp_min(0.0))
        pred_score = _percentile_score(-pred_distance, self.phys_percentiles)
        target_score = _percentile_score(-target_distance, self.phys_percentiles)

        return {
            "physics_teacher_feature_loss": feature_loss,
            "physics_teacher_margin_loss": margin_loss,
            "physics_teacher_distance": pred_distance,
            "physics_teacher_target_distance": target_distance,
            "physics_teacher_score": pred_score,
            "physics_teacher_target_score": target_score,
        }