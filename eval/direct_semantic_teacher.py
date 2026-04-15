from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from model.motion_autoencoder import MotionScorerNet


def _resolve_checkpoint_dir(checkpoint_dir: str | os.PathLike[str]) -> Path:
    path = Path(checkpoint_dir)
    if path.is_file():
        path = path.parent
    if not path.is_dir():
        raise FileNotFoundError(f"Semantic teacher checkpoint directory does not exist: {path}")
    return path


def _find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    candidates = sorted(checkpoint_dir.glob("model*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No motion scorer checkpoint was found in {checkpoint_dir}")
    return candidates[-1]


def _normalize_label(value: object) -> str:
    text = str(value or "").strip().lower()
    return text or "unknown"


class DirectSemanticTeacher:
    def __init__(
        self,
        checkpoint_dir: str | os.PathLike[str],
        *,
        device: str | torch.device = "cuda",
    ) -> None:
        checkpoint_dir = _resolve_checkpoint_dir(checkpoint_dir)
        checkpoint_path = _find_latest_checkpoint(checkpoint_dir)
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            requested_device = torch.device("cpu")
        self.device = requested_device

        args_path = checkpoint_dir / "args.json"
        if not args_path.exists():
            raise FileNotFoundError(f"args.json was not found next to semantic teacher checkpoint: {args_path}")
        with open(args_path, "r", encoding="utf-8") as handle:
            self.args = json.load(handle)

        species_vocab = tuple(_normalize_label(label) for label in self.args.get("species_vocab", []))
        action_vocab = tuple(_normalize_label(label) for label in self.args.get("action_vocab", []))
        self.species_to_index = {label: index for index, label in enumerate(species_vocab)}
        self.action_to_index = {label: index for index, label in enumerate(action_vocab)}
        self.species_unknown_index = self.species_to_index.get("unknown", 0)
        self.action_unknown_index = self.action_to_index.get("unknown", 0)

        model = MotionScorerNet(
            feature_dim=int(self.args.get("feature_dim", 13)),
            d_model=int(self.args.get("d_model", 128)),
            latent_dim=int(self.args.get("latent_dim", 128)),
            num_conv_layers=int(self.args.get("num_conv_layers", 3)),
            kernel_size=int(self.args.get("kernel_size", 5)),
            max_joints=int(self.args.get("max_joints", 143)),
            num_species=int(self.args.get("num_species", len(species_vocab) or 1)),
            num_actions=int(self.args.get("num_actions", len(action_vocab) or 1)),
            metadata_dim=int(self.args.get("metadata_feature_dim", 0)),
            metadata_hidden_dim=int(self.args.get("metadata_hidden_dim", 128)),
        )
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
        model_state = checkpoint_payload.get("model_avg") or checkpoint_payload.get("model") or checkpoint_payload
        model.load_state_dict(model_state, strict=True)
        model.to(self.device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model

    def _encode_labels(
        self,
        values: Sequence[str],
        *,
        mapping: dict[str, int],
        unknown_index: int,
    ) -> torch.Tensor:
        encoded = [mapping.get(_normalize_label(value), unknown_index) for value in values]
        return torch.as_tensor(encoded, dtype=torch.long, device=self.device)

    def compute_losses(
        self,
        pred_motion: torch.Tensor,
        target_motion: torch.Tensor,
        *,
        n_joints: torch.Tensor,
        lengths: torch.Tensor,
        species_labels: Sequence[str],
        action_labels: Sequence[str],
        temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        temperature = max(float(temperature), 1e-4)
        species_ids = self._encode_labels(
            species_labels,
            mapping=self.species_to_index,
            unknown_index=self.species_unknown_index,
        )
        action_ids = self._encode_labels(
            action_labels,
            mapping=self.action_to_index,
            unknown_index=self.action_unknown_index,
        )

        with torch.autocast(device_type=pred_motion.device.type, enabled=False):
            pred_outputs = self.model(
                pred_motion.float(),
                n_joints,
                lengths,
                return_disc_logits=False,
                return_phys_features=False,
            )
            with torch.no_grad():
                target_outputs = self.model(
                    target_motion.detach().float(),
                    n_joints.detach(),
                    lengths.detach(),
                    return_disc_logits=False,
                    return_phys_features=False,
                )

            pred_species_logits = pred_outputs["species_logits"].float()
            pred_action_logits = pred_outputs["action_logits"].float()
            target_species_logits = target_outputs["species_logits"].float()
            target_action_logits = target_outputs["action_logits"].float()

            species_ce = F.cross_entropy(pred_species_logits, species_ids, reduction="none")
            action_ce = F.cross_entropy(pred_action_logits, action_ids, reduction="none")

            species_log_probs = F.log_softmax(pred_species_logits / temperature, dim=-1)
            action_log_probs = F.log_softmax(pred_action_logits / temperature, dim=-1)
            target_species_probs = F.softmax(target_species_logits / temperature, dim=-1)
            target_action_probs = F.softmax(target_action_logits / temperature, dim=-1)
            species_kl = F.kl_div(species_log_probs, target_species_probs, reduction="none").sum(dim=-1)
            action_kl = F.kl_div(action_log_probs, target_action_probs, reduction="none").sum(dim=-1)
            species_kl = species_kl * (temperature ** 2)
            action_kl = action_kl * (temperature ** 2)

            pred_species_confidence = torch.softmax(pred_species_logits, dim=-1).max(dim=-1).values
            pred_action_confidence = torch.softmax(pred_action_logits, dim=-1).max(dim=-1).values
            target_species_confidence = torch.softmax(target_species_logits, dim=-1).max(dim=-1).values
            target_action_confidence = torch.softmax(target_action_logits, dim=-1).max(dim=-1).values

        return {
            "semantic_teacher_species_ce": species_ce,
            "semantic_teacher_action_ce": action_ce,
            "semantic_teacher_species_kl": species_kl,
            "semantic_teacher_action_kl": action_kl,
            "semantic_teacher_recognizability": pred_species_confidence * pred_action_confidence,
            "semantic_teacher_target_recognizability": target_species_confidence * target_action_confidence,
        }