from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


from diffusion.gaussian_diffusion import GaussianDiffusion, LossType, ModelMeanType, ModelVarType  # noqa: E402


class _BFloat16Model(nn.Module):
    def __init__(self, out_channels_multiplier: int = 1):
        super().__init__()
        self.out_channels_multiplier = out_channels_multiplier

    def forward(self, x: torch.Tensor, t: torch.Tensor, **model_kwargs) -> torch.Tensor:
        base = (x + 0.125).to(torch.bfloat16)
        if self.out_channels_multiplier == 1:
            return base
        extra = torch.zeros_like(base)
        return torch.cat([base, extra], dim=1)


class DiffusionLossPrecisionTests(unittest.TestCase):
    def _make_diffusion(self, *, model_var_type: ModelVarType, lambda_geo: float = 0.0, lambda_vel: float = 0.0) -> GaussianDiffusion:
        return GaussianDiffusion(
            betas=np.array([0.001, 0.002, 0.003], dtype=np.float64),
            model_mean_type=ModelMeanType.START_X,
            model_var_type=model_var_type,
            loss_type=LossType.MSE,
            lambda_geo=lambda_geo,
            lambda_vel=lambda_vel,
        )

    def _make_model_kwargs(self, batch_size: int, n_joints: int, n_feats: int, n_frames: int) -> dict:
        return {
            "y": {
                "lengths_mask": torch.ones(batch_size, 1, 1, n_frames, dtype=torch.float32),
                "lengths": torch.full((batch_size,), n_frames, dtype=torch.int64),
                "n_joints": torch.full((batch_size,), n_joints, dtype=torch.int64),
                "joints_padding_mask": torch.ones(batch_size, 1, 1, n_joints + 1, n_joints + 1, dtype=torch.float32),
                "mean": torch.zeros(batch_size, n_joints, n_feats, dtype=torch.float32),
                "std": torch.ones(batch_size, n_joints, n_feats, dtype=torch.float32),
                "norm_mean": torch.zeros(batch_size, n_joints, n_feats, dtype=torch.float32),
                "norm_std": torch.ones(batch_size, n_joints, n_feats, dtype=torch.float32),
            }
        }

    def test_training_losses_force_fp32_for_main_loss_terms(self):
        batch_size, n_joints, n_feats, n_frames = 2, 3, 12, 5
        diffusion = self._make_diffusion(
            model_var_type=ModelVarType.FIXED_LARGE,
            lambda_geo=0.25,
            lambda_vel=0.5,
        )
        model = _BFloat16Model()
        x_start = torch.randn(batch_size, n_joints, n_feats, n_frames, dtype=torch.float32)
        t = torch.tensor([0, 1], dtype=torch.int64)
        model_kwargs = self._make_model_kwargs(batch_size, n_joints, n_feats, n_frames)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            terms = diffusion.training_losses(model, x_start, t, model_kwargs=model_kwargs)

        self.assertEqual(terms["l_simple"].dtype, torch.float32)
        self.assertEqual(terms["geodesic_loss"].dtype, torch.float32)
        self.assertEqual(terms["vel_loss"].dtype, torch.float32)
        self.assertEqual(terms["loss"].dtype, torch.float32)

    def test_training_losses_force_fp32_for_vb_term(self):
        batch_size, n_joints, n_feats, n_frames = 2, 3, 12, 5
        diffusion = self._make_diffusion(model_var_type=ModelVarType.LEARNED_RANGE)
        model = _BFloat16Model(out_channels_multiplier=2)
        x_start = torch.randn(batch_size, n_joints, n_feats, n_frames, dtype=torch.float32)
        t = torch.tensor([0, 1], dtype=torch.int64)
        model_kwargs = self._make_model_kwargs(batch_size, n_joints, n_feats, n_frames)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            terms = diffusion.training_losses(model, x_start, t, model_kwargs=model_kwargs)

        self.assertEqual(terms["vb"].dtype, torch.float32)
        self.assertEqual(terms["l_simple"].dtype, torch.float32)
        self.assertEqual(terms["loss"].dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()