import torch
import torch.nn as nn


EPS = 1e-8


class MotionQualityProxyHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        # Weights are hardcoded to 1.0 to match motion scorer's score_alpha[0] and score_alpha[3]
        self.recognizability_weight = 1.0
        self.physics_weight = 1.0
        self.backbone = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.recognizability_head = nn.Linear(hidden_dim, 1)
        self.physics_head = nn.Linear(hidden_dim, 1)

    def combine_scores(
        self,
        recognizability_score: torch.Tensor,
        physics_score: torch.Tensor,
    ) -> torch.Tensor:
        # Using equal weights of 1.0, matching motion scorer's geometric mean algorithm
        weight_tensor = torch.as_tensor(
            [1.0, 1.0],
            dtype=recognizability_score.dtype,
            device=recognizability_score.device,
        )
        stacked = torch.stack(
            [recognizability_score.clamp(EPS, 1.0), physics_score.clamp(EPS, 1.0)],
            dim=0,
        )
        weighted_logs = weight_tensor[:, None] * torch.log(stacked)
        return torch.exp(weighted_logs.sum(dim=0) / weight_tensor.sum().clamp_min(EPS))

    def forward(self, pooled_features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.backbone(pooled_features)
        recognizability_score = torch.sigmoid(self.recognizability_head(hidden).squeeze(-1))
        physics_score = torch.sigmoid(self.physics_head(hidden).squeeze(-1))
        quality_score = self.combine_scores(recognizability_score, physics_score)
        return {
            "recognizability_score": recognizability_score,
            "physics_score": physics_score,
            "quality_score": quality_score,
        }