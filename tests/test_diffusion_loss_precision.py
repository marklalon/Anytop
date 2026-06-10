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


from diffusion.gaussian_diffusion import GaussianDiffusion, LossType, ModelMeanType, ModelVarType, extract_into_tensor  # noqa: E402
from diffusion.respace import SpacedDiffusion, space_timesteps  # noqa: E402
from model.anytop import AnyTop  # noqa: E402
from model.joint_mask_utils import sample_subtree_joint_mask  # noqa: E402
from utils.model_util import create_gaussian_diffusion  # noqa: E402


class _TimestepRecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.timesteps = []

    def forward(self, x: torch.Tensor, t: torch.Tensor, **model_kwargs) -> torch.Tensor:
        self.timesteps.append(t.detach().clone())
        return x


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
    def __init__(
        self,
        subtree_mask: torch.Tensor | None = None,
        temporal_span_mask: torch.Tensor | None = None,
        output_mode: str = "identity",
    ):
        super().__init__()
        self.subtree_mask = subtree_mask
        self.temporal_span_mask = temporal_span_mask
        self.output_mode = output_mode
        self.last_x = None

    def sample_subtree_joint_mask_train(self, y, njoints, device):
        if self.subtree_mask is None:
            return None
        return self.subtree_mask.to(device=device)

    def sample_temporal_span_mask_train(self, y, njoints, nframes, device):
        if self.temporal_span_mask is None:
            return None
        return self.temporal_span_mask.to(device=device)

    def forward(self, x: torch.Tensor, t: torch.Tensor, **model_kwargs) -> torch.Tensor:
        self.last_x = x.detach().clone()
        if self.output_mode == "zero":
            return torch.zeros_like(x)
        return x


class _CaptureDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_kwargs = None

    def forward(self, **kwargs):
        self.last_kwargs = kwargs
        return kwargs["tgt"]


class DiffusionLossPrecisionTests(unittest.TestCase):
    def _make_diffusion(
        self,
        *,
        model_var_type: ModelVarType,
        lambda_geo: float = 0.0,
        lambda_vel: float = 0.0,
        temporal_span_seam_loss_weight: float = 0.0,
        temporal_span_seam_width: int = 0,
    ) -> GaussianDiffusion:
        return GaussianDiffusion(
            betas=np.array([0.001, 0.002, 0.003], dtype=np.float64),
            model_mean_type=ModelMeanType.START_X,
            model_var_type=model_var_type,
            loss_type=LossType.MSE,
            lambda_geo=lambda_geo,
            lambda_vel=lambda_vel,
            temporal_span_seam_loss_weight=temporal_span_seam_loss_weight,
            temporal_span_seam_width=temporal_span_seam_width,
        )

    def _make_spaced_diffusion(self, *, model_var_type: ModelVarType) -> SpacedDiffusion:
        return SpacedDiffusion(
            use_timesteps=space_timesteps(3, [3]),
            betas=np.array([0.001, 0.002, 0.003], dtype=np.float64),
            model_mean_type=ModelMeanType.START_X,
            model_var_type=model_var_type,
            loss_type=LossType.MSE,
            lambda_geo=0.0,
            lambda_vel=0.0,
        )

    def _make_model_kwargs(self, batch_size: int, n_joints: int, n_feats: int, n_frames: int) -> dict:
        return {
            "y": {
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

    def test_extract_into_tensor_keeps_fp32_broadcast_semantics(self):
        arr = np.array([0.25, 0.5, 0.75], dtype=np.float64)
        timesteps = torch.tensor([2, 0], dtype=torch.long)

        result = extract_into_tensor(arr, timesteps, (2, 3, 4))

        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(tuple(result.shape), (2, 3, 4))
        self.assertTrue(torch.allclose(result[0], torch.full((3, 4), 0.75)))
        self.assertTrue(torch.allclose(result[1], torch.full((3, 4), 0.25)))

    def test_wrapped_model_reuses_timestep_map_tensor(self):
        diffusion = self._make_spaced_diffusion(model_var_type=ModelVarType.FIXED_LARGE)
        model = _TimestepRecordingModel()
        wrapped = diffusion._wrap_model(model)
        x = torch.zeros(2, 3, 4)
        ts = torch.tensor([0, 2], dtype=torch.long)

        wrapped(x, ts)
        wrapped(x, ts)

        self.assertEqual(len(wrapped._timestep_map_cache), 1)
        self.assertEqual(len(model.timesteps), 2)
        self.assertTrue(torch.equal(model.timesteps[0], torch.tensor([0, 2])))
        self.assertTrue(torch.equal(model.timesteps[1], torch.tensor([0, 2])))

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

    def test_training_losses_renoises_selected_temporal_spans_without_touching_other_frames(self):
        batch_size, n_joints, n_feats, n_frames = 1, 3, 12, 4
        diffusion = self._make_diffusion(model_var_type=ModelVarType.FIXED_LARGE)
        temporal_span_mask = torch.tensor(
            [[[False, True, True, False],
              [False, True, True, False],
              [False, True, True, False]]],
            dtype=torch.bool,
        )
        model = _RecordingModel(temporal_span_mask=temporal_span_mask)
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
        self.assertTrue(torch.allclose(model.last_x[..., 0], expected_x_t[..., 0]))
        self.assertTrue(torch.allclose(model.last_x[..., 3], expected_x_t[..., 3]))
        self.assertTrue(torch.allclose(model.last_x[..., 1], expected_masked[..., 1]))
        self.assertTrue(torch.allclose(model.last_x[..., 2], expected_masked[..., 2]))
        expected_unreliable = temporal_span_mask.permute(0, 2, 1).float().contiguous()
        self.assertIn("cross_limb_unreliable_mask", model_kwargs["y"])
        self.assertTrue(torch.equal(model_kwargs["y"]["cross_limb_unreliable_mask"], expected_unreliable))

    def test_training_losses_unions_joint_and_temporal_masks_into_time_varying_unreliable(self):
        batch_size, n_joints, n_feats, n_frames = 1, 3, 12, 4
        diffusion = self._make_diffusion(model_var_type=ModelVarType.FIXED_LARGE)
        subtree_mask = torch.tensor([[False, True, False]], dtype=torch.bool)
        temporal_span_mask = torch.tensor(
            [[[False, False, True, True],
              [False, False, True, True],
              [False, False, True, True]]],
            dtype=torch.bool,
        )
        model = _RecordingModel(
            subtree_mask=subtree_mask,
            temporal_span_mask=temporal_span_mask,
        )
        x_start = torch.randn(batch_size, n_joints, n_feats, n_frames, dtype=torch.float32)
        t = torch.tensor([1], dtype=torch.int64)
        model_kwargs = self._make_model_kwargs(batch_size, n_joints, n_feats, n_frames)

        diffusion.training_losses(model, x_start, t, model_kwargs=model_kwargs)

        expected_unreliable = torch.tensor(
            [[[0.0, 1.0, 0.0],
              [0.0, 1.0, 0.0],
              [1.0, 1.0, 1.0],
              [1.0, 1.0, 1.0]]],
            dtype=torch.float32,
        )
        self.assertIn("cross_limb_unreliable_mask", model_kwargs["y"])
        self.assertTrue(torch.equal(model_kwargs["y"]["cross_limb_unreliable_mask"], expected_unreliable))

    def test_training_losses_adds_temporal_span_seam_loss(self):
        # Seam loss is now a target-relative acceleration penalty on the
        # position channel (features 0:3), restricted to the seam band.
        batch_size, n_joints, n_feats, n_frames = 1, 1, 3, 6
        diffusion = self._make_diffusion(
            model_var_type=ModelVarType.FIXED_LARGE,
            temporal_span_seam_loss_weight=0.75,
            temporal_span_seam_width=1,
        )
        temporal_span_mask = torch.tensor(
            [[[False, False, True, True, False, False]]],
            dtype=torch.bool,
        )
        model = _RecordingModel(
            temporal_span_mask=temporal_span_mask,
            output_mode="zero",
        )
        # Position profile per frame (broadcast across the 3 position channels):
        # [0, 0, 1, 1, 0, 0]. With a zero prediction, the residual acceleration
        # at interior frames 1..4 is |target acc| = 1 everywhere, so the
        # seam-band weighted mean squared acceleration error equals 1.0.
        x_start = torch.zeros(batch_size, n_joints, n_feats, n_frames, dtype=torch.float32)
        x_start[:, :, 0:3, 2:4] = 1.0
        t = torch.tensor([1], dtype=torch.int64)
        model_kwargs = self._make_model_kwargs(batch_size, n_joints, n_feats, n_frames)

        terms = diffusion.training_losses(
            model,
            x_start,
            t,
            model_kwargs=model_kwargs,
            noise=torch.zeros_like(x_start),
        )

        self.assertIn("temporal_span_seam_loss", terms)
        self.assertAlmostEqual(float(terms["temporal_span_seam_loss"].item()), 1.0, places=5)
        self.assertAlmostEqual(
            float(terms["loss"].item()),
            float(terms["l_simple"].item()) + 0.75 * 1.0,
            places=5,
        )

    def test_build_temporal_span_seam_weights_peaks_at_boundaries(self):
        diffusion = self._make_diffusion(
            model_var_type=ModelVarType.FIXED_LARGE,
            temporal_span_seam_width=2,
        )
        temporal_span_mask = torch.zeros((1, 1, 50), dtype=torch.bool)
        temporal_span_mask[..., 31:40] = True

        seam_weights = diffusion._build_temporal_span_seam_weights(
            temporal_span_mask,
        )

        self.assertIsNotNone(seam_weights)
        weights = seam_weights[0, 0, 0]
        self.assertAlmostEqual(float(weights[31].item()), 1.0, places=5)
        self.assertAlmostEqual(float(weights[39].item()), 1.0, places=5)
        self.assertAlmostEqual(float(weights[29].item()), float(np.exp(-2.0)), places=5)
        self.assertAlmostEqual(float(weights[30].item()), float(np.exp(-0.5)), places=5)
        self.assertAlmostEqual(float(weights[32].item()), float(np.exp(-0.5)), places=5)
        self.assertAlmostEqual(float(weights[33].item()), float(np.exp(-2.0)), places=5)
        self.assertEqual(float(weights[34].item()), 0.0)
        self.assertEqual(float(weights[35].item()), 0.0)
        self.assertEqual(float(weights[36].item()), 0.0)
        self.assertAlmostEqual(float(weights[37].item()), float(np.exp(-2.0)), places=5)
        self.assertAlmostEqual(float(weights[38].item()), float(np.exp(-0.5)), places=5)
        self.assertAlmostEqual(float(weights[40].item()), float(np.exp(-0.5)), places=5)
        self.assertAlmostEqual(float(weights[41].item()), float(np.exp(-2.0)), places=5)

    def test_create_gaussian_diffusion_keeps_temporal_span_seam_width(self):
        class _Args:
            diffusion_steps = 3
            timestep_respacing = ""
            noise_schedule = "cosine"
            sigma_small = True
            lambda_geo = 0.0
            lambda_vel = 0.0
            temporal_span_seam_loss_weight = 0.75
            temporal_span_seam_width = 2

        diffusion = create_gaussian_diffusion(_Args())

        self.assertEqual(diffusion.temporal_span_seam_loss_weight, 0.75)
        self.assertEqual(diffusion.temporal_span_seam_width, 2)

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

    def test_spaced_diffusion_training_losses_unwraps_model_for_joint_mask_perturbation(self):
        batch_size, n_joints, n_feats, n_frames = 1, 3, 12, 4
        diffusion = self._make_spaced_diffusion(model_var_type=ModelVarType.FIXED_LARGE)
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

    def test_anytop_forward_keeps_joint_key_padding_mask_padding_only(self):
        model = AnyTop(
            max_joints=4,
            feature_len=13,
            latent_dim=8,
            ff_size=32,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            cross_limb=True,
            joint_mask_prob=1.0,
            joint_mask_budget=1.0,
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

    def test_anytop_forward_reuses_shared_temporal_template_for_masks(self):
        model = AnyTop(
            max_joints=4,
            feature_len=13,
            latent_dim=8,
            ff_size=32,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
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

    def test_anytop_forward_accepts_prepared_cross_limb_unreliable_mask_without_mutating_input(self):
        model = AnyTop(
            max_joints=4,
            feature_len=13,
            latent_dim=8,
            ff_size=32,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            cross_limb=True,
        )
        capture_decoder = _CaptureDecoder()
        model.seqTransDecoder = capture_decoder
        model.eval()

        x = torch.randn(1, 4, 13, 3, dtype=torch.float32)
        prepared_unreliable = torch.tensor(
            [[[0.0, 0.0, 0.0, 0.0]],
             [[0.0, 1.0, 0.0, 0.0]],
             [[1.0, 0.0, 0.0, 0.0]],
             [[0.0, 0.0, 1.0, 0.0]]],
            dtype=torch.float32,
        )
        prepared_copy = prepared_unreliable.clone()
        y = {
            "joints_padding_mask": torch.ones(1, 1, 1, 5, 5, dtype=torch.float32),
            "mask": torch.ones(1, 1, 1, 4, 4, dtype=torch.float32),
            "tpos_first_frame": torch.randn(1, 4, 13, dtype=torch.float32),
            "n_joints": torch.tensor([4], dtype=torch.int64),
            "joints_names_embs": torch.zeros(1, 4, 512, dtype=torch.float32),
            "cross_limb_unreliable_mask": prepared_unreliable,
        }

        model(x, torch.tensor([1], dtype=torch.int64), y=y)

        self.assertIsNotNone(capture_decoder.last_kwargs)
        self.assertTrue(torch.equal(capture_decoder.last_kwargs["cross_limb_unreliable_mask"], prepared_unreliable))
        self.assertTrue(torch.equal(y["cross_limb_unreliable_mask"], prepared_copy))

    def test_anytop_sample_subtree_joint_mask_train_matches_sequential_baseline(self):
        model = AnyTop(
            max_joints=9,
            feature_len=13,
            latent_dim=8,
            ff_size=32,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            cross_limb=False,
            joint_mask_prob=1.0,
            joint_mask_budget=0.5,
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
                joint_mask_budget=0.5,
                rng=np.random,
            )
            if per_sample_mask is not None:
                expected[batch_index, :valid_joint_count] = torch.from_numpy(per_sample_mask)

        self.assertIsNotNone(batch_mask)
        self.assertTrue(torch.equal(batch_mask.cpu(), expected))

    def test_anytop_sample_temporal_span_mask_train_marks_contiguous_valid_joint_spans(self):
        model = AnyTop(
            max_joints=4,
            feature_len=13,
            latent_dim=8,
            ff_size=32,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            cross_limb=False,
            temporal_span_mask_prob=1.0,
            temporal_span_mask_min_frames=2,
            temporal_span_mask_max_frames=2,
        )
        model.train()

        y = {
            "n_joints": torch.tensor([4, 2], dtype=torch.int64),
        }

        np.random.seed(123)
        temporal_mask = model.sample_temporal_span_mask_train(
            y,
            njoints=4,
            nframes=5,
            device=torch.device("cpu"),
        )

        self.assertIsNotNone(temporal_mask)
        self.assertEqual(temporal_mask.shape, (2, 4, 5))
        self.assertEqual(temporal_mask.dtype, torch.bool)

        for sample_index, valid_joints in enumerate((4, 2)):
            sample_mask = temporal_mask[sample_index]
            self.assertFalse(sample_mask[valid_joints:, :].any())
            self.assertTrue(torch.equal(sample_mask[:valid_joints], sample_mask[0:1].expand(valid_joints, -1)))
            frame_indices = torch.nonzero(sample_mask[0], as_tuple=False).flatten()
            self.assertEqual(len(frame_indices), 2)
            self.assertEqual(int(frame_indices[-1] - frame_indices[0]), 1)


if __name__ == "__main__":
    unittest.main()
