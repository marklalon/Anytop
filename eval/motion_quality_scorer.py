from __future__ import annotations

import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.skeleton_metadata import load_skeleton_metadata
from data_loaders.truebones.offline_reference_dataset import load_cond_dict, resolve_dataset_root
from data_loaders.truebones.truebones_utils.motion_labels import infer_species_label
from eval.physics_features import extract_physics_features
from model.motion_autoencoder import MotionScorerNet


EPS = 1e-8
REQUIRED_RUNTIME_STATS_KEYS = {
    "mu_phys",
    "sigma_phys_inv",
    "phys_percentiles",
}


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


def _load_motion_scorer_state_dict(model: MotionScorerNet, state_dict: dict[str, torch.Tensor]) -> None:
    filtered_state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("disc_head.") and not key.startswith("metadata_projection.")
    }
    incompatible = model.load_state_dict(filtered_state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            "Failed to load motion scorer checkpoint. "
            f"Missing keys: {missing}; unexpected keys: {unexpected}"
        )


def _mahalanobis_distance(values: torch.Tensor, mean: torch.Tensor, cov_inv: torch.Tensor) -> torch.Tensor:
    diff = values - mean.unsqueeze(0)
    distances_sq = torch.einsum("bi,ij,bj->b", diff, cov_inv, diff)
    return torch.sqrt(torch.clamp(distances_sq, min=0.0))


def _to_1d_long_tensor(values, batch_size: int, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.long, device=device)
    if tensor.ndim == 0:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[0] != batch_size:
        raise ValueError(f"Expected batch size {batch_size}, got {tensor.shape[0]}")
    return tensor


def _align_cond_feature_tensor(
    values,
    *,
    target_joints: int,
    target_features: int,
    pad_value: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=dtype, device=device)
    if tensor.ndim != 2:
        raise ValueError(f"Expected cond feature tensor with shape [J, F], got {tuple(tensor.shape)}")
    if tensor.shape[1] != target_features:
        raise ValueError(
            f"Expected cond feature tensor feature dim {target_features}, got {tuple(tensor.shape)}"
        )
    aligned = torch.full((target_joints, target_features), float(pad_value), dtype=dtype, device=device)
    joint_count = min(int(tensor.shape[0]), int(target_joints))
    if joint_count > 0:
        aligned[:joint_count] = tensor[:joint_count]
    return aligned


def _percentile_score(values: torch.Tensor, reference_percentiles: torch.Tensor, *, higher_is_better: bool) -> torch.Tensor:
    bins = torch.searchsorted(reference_percentiles, values.float(), right=False).float() / 100.0
    bins = bins.clamp(0.0, 1.0)
    return bins if higher_is_better else 1.0 - bins


def _geometric_mean(scores: Sequence[torch.Tensor], weights: Sequence[float]) -> torch.Tensor:
    weight_tensor = torch.as_tensor(weights, dtype=torch.float32, device=scores[0].device)
    stacked = torch.stack([score.clamp(EPS, 1.0) for score in scores], dim=0)
    weighted_logs = weight_tensor[:, None] * torch.log(stacked)
    return torch.exp(weighted_logs.sum(dim=0) / weight_tensor.sum().clamp_min(EPS))


def _normalize_score_alpha(values: Any) -> list[float]:
    if values is None:
        return [1.0, 1.0]
    alpha = [float(value) for value in values]
    if len(alpha) == 2:
        return alpha
    if len(alpha) >= 4:
        return [alpha[0], alpha[3]]
    return [1.0, 1.0]


class MotionQualityScorer:
    def __init__(
        self,
        checkpoint_dir: str | os.PathLike[str],
        device: str = "cuda",
        prefer_ema: bool = True,
        dataset_dir: str | os.PathLike[str] | None = None,
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

        self.feature_dim = int(self.args.get("feature_dim", 13))
        self.max_joints = int(self.args.get("max_joints", 143))
        self.max_frames = int(self.args.get("num_frames", 120))
        self.dataset_root = resolve_dataset_root(dataset_dir or self.args.get("data_dir") or None)
        self.cond_dict = load_cond_dict(self.dataset_root)

        stats_path = self.checkpoint_dir / "train_stats.npy"
        if not stats_path.exists():
            raise FileNotFoundError(f"train_stats.npy was not found next to the checkpoint: {stats_path}")
        self.train_stats = np.load(stats_path, allow_pickle=True).item()
        self._validate_checkpoint_match()
        self._validate_runtime_stats()
        self._load_model(prefer_ema=prefer_ema)
        self._load_runtime_stats()

    def _validate_checkpoint_match(self) -> None:
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
                "If the loaded scores look wrong, recompute scorer stats for the current checkpoint.",
                stacklevel=2,
            )

    def _validate_runtime_stats(self) -> None:
        missing = sorted(REQUIRED_RUNTIME_STATS_KEYS.difference(self.train_stats.keys()))
        if missing:
            raise ValueError(
                "This scorer requires runtime physics stats next to the checkpoint. "
                f"Missing keys in train_stats.npy: {missing}"
            )

    def _load_model(self, *, prefer_ema: bool) -> None:
        checkpoint_payload = torch.load(self.checkpoint_path, map_location="cpu")
        num_species = int(self.args.get("num_species", len(self.args.get("species_vocab", [])) or 1))
        num_actions = int(self.args.get("num_actions", len(self.args.get("action_vocab", [])) or 1))
        metadata_dim = int(self.args.get("metadata_feature_dim", 0))
        self.model = MotionScorerNet(
            feature_dim=self.feature_dim,
            d_model=int(self.args.get("d_model", 128)),
            latent_dim=int(self.args.get("latent_dim", 128)),
            num_conv_layers=int(self.args.get("num_conv_layers", 3)),
            kernel_size=int(self.args.get("kernel_size", 5)),
            max_joints=self.max_joints,
            num_species=num_species,
            num_actions=num_actions,
            metadata_dim=metadata_dim,
            metadata_hidden_dim=int(self.args.get("metadata_hidden_dim", 128)),
        )
        model_state = _select_model_state_dict(checkpoint_payload, prefer_ema=prefer_ema)
        _load_motion_scorer_state_dict(self.model, model_state)
        self.model.to(self.device)
        self.model.eval()

    def _load_runtime_stats(self) -> None:
        self.skeleton_lookup = load_skeleton_metadata(self.dataset_root)
        self.mu_phys = torch.as_tensor(self.train_stats["mu_phys"], dtype=torch.float32, device=self.device)
        self.sigma_phys_inv = torch.as_tensor(self.train_stats["sigma_phys_inv"], dtype=torch.float32, device=self.device)
        self.phys_percentiles = torch.as_tensor(self.train_stats["phys_percentiles"], dtype=torch.float32, device=self.device)
        self.score_alpha = _normalize_score_alpha(self.train_stats.get("score_alpha"))

    def _prepare_condition_features(
        self,
        motion: torch.Tensor,
        object_types: Sequence[str],
    ) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        batch_size = motion.shape[0]
        if len(object_types) != batch_size:
            raise ValueError(f"Expected {batch_size} object_types, got {len(object_types)}")
        normalized_object_types = [str(object_type) for object_type in object_types]
        target_joints = int(motion.shape[1])
        target_features = int(motion.shape[2])
        feature_mean = torch.stack(
            [
                _align_cond_feature_tensor(
                    self.cond_dict[object_type]["mean"],
                    target_joints=target_joints,
                    target_features=target_features,
                    pad_value=0.0,
                    device=motion.device,
                    dtype=motion.dtype,
                )
                for object_type in normalized_object_types
            ],
            dim=0,
        )
        feature_std = torch.stack(
            [
                _align_cond_feature_tensor(
                    self.cond_dict[object_type]["std"],
                    target_joints=target_joints,
                    target_features=target_features,
                    pad_value=1.0,
                    device=motion.device,
                    dtype=motion.dtype,
                )
                for object_type in normalized_object_types
            ],
            dim=0,
        )
        return normalized_object_types, feature_mean, feature_std

    def score(
        self,
        motion: torch.Tensor | np.ndarray,
        n_joints,
        lengths,
        object_types: Sequence[str],
    ) -> dict[str, torch.Tensor | str]:
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
        object_types, feature_mean, feature_std = self._prepare_condition_features(motion, object_types)

        with torch.no_grad():
            outputs = self.model(
                motion,
                n_joints,
                lengths,
                return_phys_features=False,
            )
            species_probs = torch.softmax(outputs["species_logits"], dim=-1)
            action_probs = torch.softmax(outputs["action_logits"], dim=-1)
            physics_features = extract_physics_features(
                motion,
                n_joints,
                lengths,
                object_types,
                self.skeleton_lookup,
                feature_mean=feature_mean,
                feature_std=feature_std,
            )
            physics_distance = _mahalanobis_distance(physics_features.float(), self.mu_phys, self.sigma_phys_inv)

        species_confidence = species_probs.max(dim=-1).values
        action_confidence = action_probs.max(dim=-1).values
        recognizability_score = (species_confidence * action_confidence).clamp(EPS, 1.0)
        physics_score = _percentile_score(-physics_distance, self.phys_percentiles, higher_is_better=True)
        quality_score = _geometric_mean(
            [recognizability_score, physics_score],
            self.score_alpha,
        )

        return {
            "quality_score": quality_score.detach().cpu(),
            "recognizability_score": recognizability_score.detach().cpu(),
            "physics_score": physics_score.detach().cpu(),
            "species_confidence": species_confidence.detach().cpu(),
            "action_confidence": action_confidence.detach().cpu(),
            "physics_distance": physics_distance.detach().cpu(),
            "mahal_distance": physics_distance.detach().cpu(),
        }

    def score_npy(
        self,
        npy_path: str | os.PathLike[str],
        object_type: str,
        cond_dict: dict[str, dict[str, np.ndarray]] | None = None,
    ) -> dict[str, torch.Tensor | str]:
        if cond_dict is None:
            cond_dict = load_cond_dict(self.dataset_root)
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
            object_types=[object_type],
        )
        result["object_type"] = object_type
        result["path"] = str(npy_path)
        result["species_label"] = infer_species_label(object_type)
        return result

    def score_batch_from_dataloader(self, dataloader) -> list[dict[str, Any]]:
        scored_samples: list[dict[str, Any]] = []
        for motion, cond in dataloader:
            object_types = cond["y"].get("object_type")
            batch_result = self.score(motion, cond["y"]["n_joints"], cond["y"]["lengths"], object_types=object_types)
            batch_size = motion.shape[0]
            motion_names = cond["y"].get("motion_name", [None] * batch_size)
            for batch_index in range(batch_size):
                scored_samples.append(
                    {
                        "quality_score": float(batch_result["quality_score"][batch_index].item()),
                        "recognizability_score": float(batch_result["recognizability_score"][batch_index].item()),
                        "physics_score": float(batch_result["physics_score"][batch_index].item()),
                        "species_confidence": float(batch_result["species_confidence"][batch_index].item()),
                        "action_confidence": float(batch_result["action_confidence"][batch_index].item()),
                        "physics_distance": float(batch_result["physics_distance"][batch_index].item()),
                        "mahal_distance": float(batch_result["mahal_distance"][batch_index].item()),
                        "object_type": object_types[batch_index],
                        "motion_name": motion_names[batch_index],
                    }
                )
        return scored_samples
