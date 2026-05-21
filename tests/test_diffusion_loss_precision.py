from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


from diffusion.gaussian_diffusion import GaussianDiffusion, LossType, ModelMeanType, ModelVarType  # noqa: E402
from model.anytop import AnyTop  # noqa: E402
from model.joint_mask_utils import sample_subtree_joint_mask  # noqa: E402


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


class _RecordingModel(nn.Module):
    def __init__(self, subtree_mask: torch.Tensor | None = None):
        super().__init__()
        self.subtree_mask = subtree_mask
        self.last_x = None

    def sample_subtree_joint_mask_train(self, y, njoints, device):
        if self.subtree_mask is None:
            return None
        return self.subtree_mask.to(device=device)

    def forward(self, x: torch.Tensor, t: torch.Tensor, **model_kwargs) -> torch.Tensor:
        self.last_x = x.detach().clone()
        return x


class _CaptureDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_kwargs = None

    def forward(self, **kwargs):
        self.last_kwargs = kwargs
        return kwargs["tgt"]


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

    def test_training_losses_renoises_selected_joints_without_touching_others(self):
        batch_size, n_joints, n_feats, n_frames = 1, 3, 12, 4
        diffusion = self._make_diffusion(model_var_type=ModelVarType.FIXED_LARGE)
        subtree_mask = torch.tensor([[False, True, False]], dtype=torch.bool)
        model = _RecordingModel(subtree_mask=subtree_mask)
        x_start = torch.arange(
            batch_size * n_joints * n_feats * n_frames, dtype=torch.float32
        ).reshape(batch_size, n_joints, n_feats, n_frames)
        t = torch.tensor([1], dtype=torch.int64)
        t_random = torch.tensor([2], dtype=torch.int64)
        base_noise = torch.full_like(x_start, 0.5)
        fresh_noise = torch.full_like(x_start, -0.25)
        model_kwargs = self._make_model_kwargs(batch_size, n_joints, n_feats, n_frames)

        expected_x_t = diffusion.q_sample(x_start, t, noise=base_noise)
        expected_masked = diffusion.q_sample(x_start, t_random, noise=fresh_noise)

        with patch("diffusion.gaussian_diffusion.th.randint", return_value=t_random), \
             patch("diffusion.gaussian_diffusion.th.randn_like", return_value=fresh_noise):
            diffusion.training_losses(model, x_start, t, model_kwargs=model_kwargs, noise=base_noise)

        self.assertIsNotNone(model.last_x)
        self.assertTrue(torch.allclose(model.last_x[:, 0], expected_x_t[:, 0]))
        self.assertTrue(torch.allclose(model.last_x[:, 2], expected_x_t[:, 2]))
        self.assertTrue(torch.allclose(model.last_x[:, 1], expected_masked[:, 1]))
        expected_unreliable = subtree_mask[:, None, :].float().expand(-1, n_frames, -1)
        self.assertIn("cross_limb_unreliable_mask", model_kwargs["y"])
        self.assertTrue(torch.equal(model_kwargs["y"]["cross_limb_unreliable_mask"], expected_unreliable))

    def test_training_losses_clears_stale_cross_limb_unreliable_mask_when_no_mask_is_sampled(self):
        batch_size, n_joints, n_feats, n_frames = 1, 3, 12, 4
        diffusion = self._make_diffusion(model_var_type=ModelVarType.FIXED_LARGE)
        model = _RecordingModel(subtree_mask=None)
        x_start = torch.randn(batch_size, n_joints, n_feats, n_frames, dtype=torch.float32)
        t = torch.tensor([1], dtype=torch.int64)
        model_kwargs = self._make_model_kwargs(batch_size, n_joints, n_feats, n_frames)
        model_kwargs["y"]["cross_limb_unreliable_mask"] = torch.ones(batch_size, n_frames, n_joints, dtype=torch.float32)

        diffusion.training_losses(model, x_start, t, model_kwargs=model_kwargs)

        self.assertNotIn("cross_limb_unreliable_mask", model_kwargs["y"])

    def test_anytop_forward_keeps_joint_key_padding_mask_padding_only(self):
        model = AnyTop(
            max_joints=4,
            feature_len=13,
            latent_dim=8,
            ff_size=32,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            skip_t5=True,
            cross_limb=True,
            joint_mask_prob=1.0,
        )
        capture_decoder = _CaptureDecoder()
        model.seqTransDecoder = capture_decoder
        model.train()

        x = torch.randn(1, 4, 13, 3, dtype=torch.float32)
        y = {
            "joints_padding_mask": torch.ones(1, 1, 1, 5, 5, dtype=torch.float32),
            "mask": torch.ones(1, 1, 1, 4, 4, dtype=torch.float32),
            "tpos_first_frame": torch.randn(1, 4, 13, dtype=torch.float32),
            "n_joints": torch.tensor([3], dtype=torch.int64),
            "joints_names_embs": torch.zeros(1, 4, 512, dtype=torch.float32),
            "parents": torch.tensor([[-1, 0, 1, 2]], dtype=torch.int64),
            "joint_mask_candidate_roots": torch.tensor([[False, True, True, True]], dtype=torch.bool),
        }

        model(x, torch.tensor([1], dtype=torch.int64), y=y)

        expected = torch.tensor([[False, False, False, True]], dtype=torch.bool)
        self.assertIsNotNone(capture_decoder.last_kwargs)
        self.assertTrue(torch.equal(capture_decoder.last_kwargs["tgt_key_padding_mask"], expected))
        self.assertTrue(torch.equal(y["joints_key_padding_mask"], expected))

    def test_anytop_forward_reuses_shared_temporal_template_for_masks(self):
        model = AnyTop(
            max_joints=4,
            feature_len=13,
            latent_dim=8,
            ff_size=32,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            skip_t5=True,
            cross_limb=True,
        )
        capture_decoder = _CaptureDecoder()
        model.seqTransDecoder = capture_decoder
        model.eval()

        x = torch.randn(2, 4, 13, 3, dtype=torch.float32)
        temp_mask = torch.ones(2, 1, 1, 4, 4, dtype=torch.float32)
        temp_mask[0, 0, 0, 1, 2] = 0.0
        temp_mask[1, 0, 0, 2, 1] = 0.0
        y = {
            "joints_padding_mask": torch.ones(2, 1, 1, 5, 5, dtype=torch.float32),
            "mask": temp_mask,
            "tpos_first_frame": torch.randn(2, 4, 13, dtype=torch.float32),
            "n_joints": torch.tensor([4, 3], dtype=torch.int64),
            "joints_names_embs": torch.zeros(2, 4, 512, dtype=torch.float32),
        }

        model(x, torch.tensor([1, 2], dtype=torch.int64), y=y)

        self.assertIsNotNone(capture_decoder.last_kwargs)
        spatial_mask = capture_decoder.last_kwargs["spatial_mask"]
        temporal_mask = capture_decoder.last_kwargs["temporal_mask"]
        temporal_template = capture_decoder.last_kwargs["temporal_template"]

        expected_template = (1.0 - temp_mask.reshape(2, -1, 4, 4)[:, :1].float()) * -1e4
        expected_template = expected_template.expand(-1, model.num_heads, -1, -1).reshape(-1, 4, 4)
        expected_mask = expected_template.reshape(2, model.num_heads, 4, 4).unsqueeze(1)
        expected_mask = expected_mask.expand(-1, 4, -1, -1, -1).reshape(-1, 4, 4)

        self.assertEqual(spatial_mask.shape, (2, model.num_heads, 4, 4))
        self.assertTrue(torch.equal(temporal_template, expected_template))
        self.assertTrue(torch.equal(temporal_mask, expected_mask))

    def test_anytop_forward_normalizes_cross_limb_unreliable_mask_without_mutating_input(self):
        model = AnyTop(
            max_joints=4,
            feature_len=13,
            latent_dim=8,
            ff_size=32,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            skip_t5=True,
            cross_limb=True,
        )
        capture_decoder = _CaptureDecoder()
        model.seqTransDecoder = capture_decoder
        model.eval()

        x = torch.randn(1, 4, 13, 3, dtype=torch.float32)
        raw_unreliable = torch.tensor(
            [[[0.0, 1.0, 0.0, 0.0],
              [1.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0]]],
            dtype=torch.float32,
        )
        raw_copy = raw_unreliable.clone()
        y = {
            "joints_padding_mask": torch.ones(1, 1, 1, 5, 5, dtype=torch.float32),
            "mask": torch.ones(1, 1, 1, 4, 4, dtype=torch.float32),
            "tpos_first_frame": torch.randn(1, 4, 13, dtype=torch.float32),
            "n_joints": torch.tensor([4], dtype=torch.int64),
            "joints_names_embs": torch.zeros(1, 4, 512, dtype=torch.float32),
            "cross_limb_unreliable_mask": raw_unreliable,
        }

        model(x, torch.tensor([1], dtype=torch.int64), y=y)

        self.assertIsNotNone(capture_decoder.last_kwargs)
        expected = torch.cat(
            [torch.zeros(1, 1, 4, dtype=torch.float32), raw_unreliable], dim=1
        ).transpose(0, 1).contiguous()
        self.assertTrue(torch.equal(capture_decoder.last_kwargs["cross_limb_unreliable_mask"], expected))
        self.assertTrue(torch.equal(y["cross_limb_unreliable_mask"], raw_copy))

    def test_anytop_sample_subtree_joint_mask_train_matches_sequential_baseline(self):
        model = AnyTop(
            max_joints=9,
            feature_len=13,
            latent_dim=8,
            ff_size=32,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            skip_t5=True,
            cross_limb=False,
            joint_mask_prob=0.5,
        )
        model.train()

        parents = torch.tensor(
            [
                [-1, 0, 1, 2, 3, 0, 5, 0, 7],
                [-1, 0, 1, 2, 0, 0, 0, 0, 0],
            ],
            dtype=torch.int64,
        )
        candidate_root_mask = torch.tensor(
            [
                [False, True, False, False, False, True, False, True, False],
                [False, True, False, True, False, False, False, False, False],
            ],
            dtype=torch.bool,
        )
        y = {
            "n_joints": torch.tensor([9, 5], dtype=torch.int64),
            "parents": parents,
            "joint_mask_candidate_roots": candidate_root_mask,
        }

        np.random.seed(123)
        batch_mask = model.sample_subtree_joint_mask_train(y, njoints=9, device=torch.device("cpu"))

        np.random.seed(123)
        expected = torch.zeros((2, 9), dtype=torch.bool)
        for batch_index, valid_joint_count in enumerate((9, 5)):
            per_sample_mask = sample_subtree_joint_mask(
                parents=parents[batch_index, :valid_joint_count].tolist(),
                candidate_root_mask=candidate_root_mask[batch_index, :valid_joint_count].numpy(),
                joint_mask_prob=0.5,
                rng=np.random,
            )
            if per_sample_mask is not None:
                expected[batch_index, :valid_joint_count] = torch.from_numpy(per_sample_mask)

        self.assertIsNotNone(batch_mask)
        self.assertTrue(torch.equal(batch_mask.cpu(), expected))


if __name__ == "__main__":
    unittest.main()