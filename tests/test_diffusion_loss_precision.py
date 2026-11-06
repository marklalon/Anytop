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
from data_loaders.reference import REST_GEOM_DIM  # noqa: E402
from data_loaders.truebones.truebones_utils.canonical_features import CANONICAL_STAT_LEN, MODEL_CHANNELS, REST_LENGTH_SCALE_FLOOR  # noqa: E402
from data_loaders.truebones.truebones_utils.joint_struct_features import (  # noqa: E402
    JOINT_STRUCT_DIM,
)
from model.motion_diffusion import MotionDiffusion  # noqa: E402
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
        output_mode: str = "identity",
    ):
        super().__init__()
        self.output_mode = output_mode
        self.last_x = None


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


_IDENTITY_6D = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def _rot_z_6d(angle):
    c, s = float(np.cos(angle)), float(np.sin(angle))
    # First two columns of Rz(angle), the 6D layout rotation_6d_to_matrix reads.
    return (c, s, 0.0, -s, c, 0.0)


class BoneVectorLossTests(unittest.TestCase):
    """Tests for bone_vector_consistency_loss.

    The loss normalizes per-frame bone-vector errors by the PHYSICAL rest bone
    length. A rest whose positions are zero (the canonical rest-centered
    residual) made every rest bone length read as 0, masked out every bone, and
    silently zeroed the whole loss regardless of lambda_bone.
    """

    def _make_diffusion(self, betas=(0.001, 0.002, 0.003)) -> GaussianDiffusion:
        return GaussianDiffusion(
            betas=np.array(betas, dtype=np.float64),
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_LARGE,
            loss_type=LossType.MSE,
            lambda_bone=0.2,
        )

    @staticmethod
    def _chain(rest_pos, n_frames=4):
        rest_pos = np.asarray(rest_pos, dtype=np.float32)
        n = rest_pos.shape[0]
        y = {
            "parents": [np.arange(-1, n - 1, dtype=np.int64)],
            "rest_pos_ric_hml": torch.from_numpy(rest_pos).unsqueeze(0),
            "rest_length_scale": torch.tensor([1.0]),
        }
        target = torch.zeros(1, n, 12, n_frames)
        target[:, :, 0:3, :] = torch.from_numpy(rest_pos).view(1, n, 3, 1).expand(1, n, 3, n_frames)
        target[:, :, 3:9, :] = torch.tensor(_IDENTITY_6D).view(1, 1, 6, 1)
        return y, target, torch.ones(1, 1, 1, n)

    def test_direction_error_with_exact_length_is_penalized(self):
        diffusion = self._make_diffusion()
        y, target, spat_mask = self._chain([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        pred = target.clone()
        # Swing bone 1->2 by 90 degrees about z: same length, wrong direction.
        pred[0, 2, 0:3, :] = torch.tensor([1.0, 1.0, 0.0]).view(3, 1)
        loss = diffusion.bone_vector_consistency_loss(pred, target, spat_mask, y)
        # |dv| = sqrt(2), rest = 1 -> linear Huber branch: sqrt(2) - 0.05, over
        # 2 bones and the leaf's virtual bone (rotation unchanged, so 0).
        expected = (np.sqrt(2.0) - 0.05) / 3.0
        self.assertAlmostEqual(loss.item(), expected, places=5)

    def test_rigid_translation_is_free(self):
        # A world translation moves the root's world XZ and every joint's
        # height; the other joints' RIC XZ do not change.
        diffusion = self._make_diffusion()
        y, target, spat_mask = self._chain([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        pred = target.clone()
        pred[:, 0, 0, :] += 0.3
        pred[:, 0, 2, :] += 0.5
        pred[:, :, 1, :] -= 0.2
        loss = diffusion.bone_vector_consistency_loss(pred, target, spat_mask, y)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_short_bone_normalization_is_floored(self):
        diffusion = self._make_diffusion()
        from diffusion.gaussian_diffusion import BONE_REST_FLOOR_FRAC

        # Bone 1->2 is 0.01 long, below the floor BONE_REST_FLOOR_FRAC * L (L = 1).
        y, target, spat_mask = self._chain([[0, 0, 0], [1, 0, 0], [1.01, 0, 0]])
        pred = target.clone()
        pred[0, 2, 0, :] += 0.005
        loss = diffusion.bone_vector_consistency_loss(pred, target, spat_mask, y)
        rel = 0.005 / BONE_REST_FLOOR_FRAC  # floored rest, not 0.005 / 0.01
        assert 0.01 < BONE_REST_FLOOR_FRAC and rel < 0.1  # floored, still in the quadratic regime
        expected = (0.5 * rel * rel / 0.1) / 3.0
        self.assertAlmostEqual(loss.item(), expected, places=5)

    def test_gradient_is_finite_on_exact_and_padded_bones(self):
        diffusion = self._make_diffusion()
        y, target, _ = self._chain([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        # One padded joint row: zero positions on both sides, masked out.
        target = torch.cat([target, torch.zeros(1, 1, 12, target.shape[-1])], dim=1)
        y["rest_pos_ric_hml"] = torch.cat([y["rest_pos_ric_hml"], torch.zeros(1, 1, 3)], dim=1)
        spat_mask = torch.tensor([1.0, 1.0, 1.0, 0.0]).view(1, 1, 1, 4)
        pred = target.clone()
        pred[0, 2, 0, :] += 0.2
        pred.requires_grad_(True)
        loss = diffusion.bone_vector_consistency_loss(pred, target, spat_mask, y, t=torch.tensor([1]))
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(pred.grad.abs().sum().item(), 0.0)
        self.assertEqual(pred.grad[0, 3].abs().sum().item(), 0.0)

    def test_high_noise_timesteps_are_downweighted(self):
        diffusion = self._make_diffusion(betas=(0.01, 0.5, 0.9))
        y, target, spat_mask = self._chain([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        pred = target.clone()
        pred[0, 2, 0, :] += 0.3
        unweighted = diffusion.bone_vector_consistency_loss(pred, target, spat_mask, y)
        low = diffusion.bone_vector_consistency_loss(pred, target, spat_mask, y, t=torch.tensor([0]))
        high = diffusion.bone_vector_consistency_loss(pred, target, spat_mask, y, t=torch.tensor([2]))
        alphas_cumprod = diffusion.alphas_cumprod
        self.assertAlmostEqual(low.item(), unweighted.item() * alphas_cumprod[0], places=6)
        self.assertAlmostEqual(high.item(), unweighted.item() * alphas_cumprod[2], places=6)
        self.assertLess(high.item(), 0.1 * low.item())

    def test_bone_loss_nonzero_when_predicted_lengths_differ(self):
        diffusion = self._make_diffusion()
        # 3-joint chain, rest bone length 1.0 each.
        rest_pos = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32
        )
        parents = np.array([-1, 0, 1], dtype=np.int64)
        y = {
            "parents": [parents],
            "rest_pos_ric_hml": torch.from_numpy(rest_pos).unsqueeze(0),
        }
        n_frames = 4
        target = torch.zeros(1, 3, 12, n_frames)
        target[:, :, 0:3, :] = torch.from_numpy(rest_pos).view(1, 3, 3, 1).expand(1, 3, 3, n_frames)
        pred = target.clone()
        # Stretch bone 0->1 by 10% (joint 1 shifts +x).
        pred[0, 1, 0, :] += 0.1
        spat_mask = torch.ones(1, 1, 1, 3)

        loss = diffusion.bone_vector_consistency_loss(pred, target, spat_mask, y)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

        # Exact match -> zero loss.
        exact = diffusion.bone_vector_consistency_loss(target, target, spat_mask, y)
        self.assertEqual(exact.item(), 0.0)

    def test_leaf_rotation_is_penalized_through_virtual_bone(self):
        diffusion = self._make_diffusion()
        y, target, spat_mask = self._chain([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        pred = target.clone()
        # Rotate only the leaf (joint 2): positions untouched, so no real bone moves.
        angle = np.pi / 2
        pred[0, 2, 3:9, :] = torch.tensor(_rot_z_6d(angle)).view(6, 1)
        loss = diffusion.bone_vector_consistency_loss(pred, target, spat_mask, y)
        # |R - I|_F = 2*sqrt(2)*sin(angle/2) about any axis; /sqrt(3) is the
        # RMS tip error over unit directions. Linear Huber branch, over 3 entries.
        leaf_rel = 2.0 * np.sqrt(2.0) * np.sin(angle / 2) / np.sqrt(3.0)
        self.assertAlmostEqual(loss.item(), (leaf_rel - 0.05) / 3.0, places=5)

    def test_non_leaf_rotation_has_no_virtual_bone(self):
        diffusion = self._make_diffusion()
        y, target, spat_mask = self._chain([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        pred = target.clone()
        # Joint 1 has a child: its rotation reaches this loss only through that
        # child's position, which is left unchanged here.
        pred[0, 1, 3:9, :] = torch.tensor(_rot_z_6d(np.pi / 2)).view(6, 1)
        loss = diffusion.bone_vector_consistency_loss(pred, target, spat_mask, y)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_padded_joint_is_not_a_leaf(self):
        diffusion = self._make_diffusion()
        y, target, _ = self._chain([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        n_frames = target.shape[-1]
        target = torch.cat([target, torch.zeros(1, 1, 12, n_frames)], dim=1)
        y["rest_pos_ric_hml"] = torch.cat([y["rest_pos_ric_hml"], torch.zeros(1, 1, 3)], dim=1)
        spat_mask = torch.tensor([1.0, 1.0, 1.0, 0.0]).view(1, 1, 1, 4)
        pred = target.clone()
        pred[0, 3, 3:9, :] = torch.tensor(_rot_z_6d(1.0)).view(6, 1)
        loss = diffusion.bone_vector_consistency_loss(pred, target, spat_mask, y)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_bone_loss_rejects_a_batch_without_physical_rest(self):
        diffusion = self._make_diffusion()
        parents = np.array([-1, 0, 1], dtype=np.int64)
        # The rest geometry holds unit directions, not lengths.
        y = {
            "parents": [parents],
            "rest_geom": torch.zeros(1, 3, REST_GEOM_DIM),
        }
        target = torch.zeros(1, 3, 12, 4)
        with self.assertRaises(ValueError):
            diffusion.bone_vector_consistency_loss(
                target, target, torch.ones(1, 1, 1, 3), y
            )

    def test_bone_loss_accepts_rest_pose_physical_fallback(self):
        diffusion = self._make_diffusion()
        rest_pos = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32
        )
        parents = np.array([-1, 0, 1], dtype=np.int64)
        rest_physical = np.zeros((3, 12), dtype=np.float32)
        rest_physical[:, 0:3] = rest_pos
        y = {
            "parents": [parents],
            "rest_pose_physical": torch.from_numpy(rest_physical).unsqueeze(0),
        }
        n_frames = 4
        target = torch.zeros(1, 3, 12, n_frames)
        target[:, :, 0:3, :] = torch.from_numpy(rest_pos).view(1, 3, 3, 1).expand(1, 3, 3, n_frames)
        pred = target.clone()
        pred[0, 2, 0, :] += 0.2  # stretch bone 1->2 by 20%
        loss = diffusion.bone_vector_consistency_loss(
            pred, target, torch.ones(1, 1, 1, 3), y
        )
        self.assertGreater(loss.item(), 0.0)


class DiffusionLossPrecisionTests(unittest.TestCase):
    def _make_diffusion(
        self,
        *,
        model_var_type: ModelVarType,
        lambda_geo: float = 0.0,
        lambda_drot: float = 0.0,
        drot_joint_boost: float = 0.0,
        lambda_dpos: float = 0.0,
        lambda_acc: float = 0.0,
        lambda_root_pos: float = 0.0,
        lambda_root_dpos: float = 0.0,
        lambda_root_acc: float = 0.0,
        lambda_bone: float = 0.0,
    ) -> GaussianDiffusion:
        return GaussianDiffusion(
            betas=np.array([0.001, 0.002, 0.003], dtype=np.float64),
            model_mean_type=ModelMeanType.START_X,
            model_var_type=model_var_type,
            loss_type=LossType.MSE,
            lambda_geo=lambda_geo,
            lambda_drot=lambda_drot,
            drot_joint_boost=drot_joint_boost,
            lambda_dpos=lambda_dpos,
            lambda_acc=lambda_acc,
            lambda_root_pos=lambda_root_pos,
            lambda_root_dpos=lambda_root_dpos,
            lambda_root_acc=lambda_root_acc,
            lambda_bone=lambda_bone,
        )

    def _make_spaced_diffusion(self, *, model_var_type: ModelVarType) -> SpacedDiffusion:
        return SpacedDiffusion(
            use_timesteps=space_timesteps(3, [3]),
            betas=np.array([0.001, 0.002, 0.003], dtype=np.float64),
            model_mean_type=ModelMeanType.START_X,
            model_var_type=model_var_type,
            loss_type=LossType.MSE,
            lambda_geo=0.0,
        )

    def _make_model_kwargs(self, batch_size: int, n_joints: int, n_feats: int, n_frames: int) -> dict:
        return {
            "y": {
                "lengths": torch.full((batch_size,), n_frames, dtype=torch.int64),
                "n_joints": torch.full((batch_size,), n_joints, dtype=torch.int64),
                "joints_padding_mask": torch.ones(batch_size, 1, 1, n_joints + 1, n_joints + 1, dtype=torch.float32),
                # Canonical decode (canonical_to_physical_hml) needs rest-pose geometry
                # and the global standardization stats (identity here so the decoded
                # physical space equals the L-normalized space the tests reason about).
                "rest_pos_ric_hml": torch.arange(n_joints * 3, dtype=torch.float32).reshape(n_joints, 3),
                "parents": np.arange(-1, n_joints - 1),
                "canonical_feature_mean": torch.zeros(CANONICAL_STAT_LEN, dtype=torch.float32),
                "canonical_feature_std": torch.ones(CANONICAL_STAT_LEN, dtype=torch.float32),
                "canonical_dpos_std": torch.ones(batch_size, dtype=torch.float32),
            }
        }

    def test_training_uses_uniform_timestep_noise_and_supervises_all_model_channels(self):
        diffusion = self._make_diffusion(model_var_type=ModelVarType.FIXED_LARGE)
        model = _RecordingModel(output_mode="zero")
        x_start = torch.randn(2, 3, 12, 5)
        noise = torch.randn_like(x_start)
        t = torch.tensor([0, 2])
        kwargs = self._make_model_kwargs(2, 3, 12, 5)
        original_keys = set(kwargs["y"])

        terms = diffusion.training_losses(model, x_start, t, model_kwargs=kwargs, noise=noise)

        self.assertTrue(torch.equal(model.last_x, diffusion.q_sample(x_start, t, noise=noise)))
        expected = x_start[:, :, MODEL_CHANNELS, :].square().mean(dim=(1, 2, 3))
        self.assertTrue(torch.allclose(terms["loss"], expected))
        self.assertEqual(set(kwargs["y"]), original_keys)

    def test_training_losses_force_fp32_for_main_loss_terms(self):
        batch_size, n_joints, n_feats, n_frames = 2, 3, 12, 5
        diffusion = self._make_diffusion(
            model_var_type=ModelVarType.FIXED_LARGE,
            lambda_geo=0.25,
            lambda_drot=0.1,
            drot_joint_boost=3.0,
            lambda_dpos=0.5,
            lambda_acc=0.05,
            lambda_root_pos=0.1,
            lambda_root_dpos=0.5,
            lambda_root_acc=0.5,
        )
        model = _BFloat16Model()
        x_start = torch.randn(batch_size, n_joints, n_feats, n_frames, dtype=torch.float32)
        t = torch.tensor([0, 1], dtype=torch.int64)
        model_kwargs = self._make_model_kwargs(batch_size, n_joints, n_feats, n_frames)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            terms = diffusion.training_losses(model, x_start, t, model_kwargs=model_kwargs)

        self.assertEqual(terms["l_simple"].dtype, torch.float32)
        self.assertEqual(terms["drot_loss"].dtype, torch.float32)
        self.assertEqual(terms["geodesic_loss"].dtype, torch.float32)
        self.assertEqual(terms["dpos_loss"].dtype, torch.float32)
        self.assertEqual(terms["acc_loss"].dtype, torch.float32)
        self.assertEqual(terms["root_pos_loss"].dtype, torch.float32)
        self.assertEqual(terms["root_dpos_loss"].dtype, torch.float32)
        self.assertEqual(terms["root_acc_loss"].dtype, torch.float32)
        self.assertEqual(terms["loss"].dtype, torch.float32)

    def test_acceleration_and_root_position_cli_weights_and_legacy_defaults(self):
        from utils.parser_util import train_args
        with patch.object(sys, 'argv', ['train', '--cond_path', 'unused.npy',
                                      '--lambda_acc', '.05', '--lambda_root_pos', '.1']):
            args = train_args()
        diffusion = create_gaussian_diffusion(args)
        self.assertEqual(diffusion.lambda_acc, .05)
        self.assertEqual(diffusion.lambda_root_pos, .1)
        del args.lambda_acc, args.lambda_root_pos
        legacy = create_gaussian_diffusion(args)
        self.assertEqual(legacy.lambda_acc, 0.)
        self.assertEqual(legacy.lambda_root_pos, 0.)

    def test_acceleration_and_root_objectives_match_ae_including_height(self):
        from train.motion_reconstruction_losses import MotionReconstructionLosses
        weights = dict(lambda_acc=.05, lambda_root_pos=.1,
                       lambda_root_dpos=.2, lambda_root_acc=.3)
        diffusion = self._make_diffusion(model_var_type=ModelVarType.FIXED_LARGE, **weights)
        target = torch.zeros(2, 3, 9, 6)
        target[:, :, 3] = 1.
        target[:, :, 7] = 1.
        kwargs = self._make_model_kwargs(2, 3, 9, 6)
        kwargs['y']['is_loop'] = torch.tensor([True, False])
        kwargs['y']['dpos_loss_weight'] = torch.tensor([.5, 2.])
        prediction = target.clone()
        prediction[:, 0, 1, 2] += 1.  # Root height alone must reach both root dynamics terms.
        prediction[:, 1, 0, 3] += .5
        prediction.requires_grad_()
        terms = diffusion.training_losses(
            lambda *args, **kw: prediction, target, torch.tensor([0, 2]),
            model_kwargs=kwargs, noise=torch.zeros_like(target))
        ae = MotionReconstructionLosses(**weights)(prediction, target, kwargs['y'])
        for key in ('acc_loss', 'root_pos_loss', 'root_dpos_loss', 'root_acc_loss'):
            self.assertEqual(terms[key].shape, (2,))
            self.assertTrue(torch.all(terms[key] > 0))
            torch.testing.assert_close(terms[key].mean(), ae[key])
        torch.testing.assert_close(terms['loss'].mean(), ae['loss'])
        terms['loss'].mean().backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(prediction.grad[:, 0, 1, 2].abs().sum().item(), 0.)
        for name in weights:
            setattr(diffusion, name, 0.)
        disabled = diffusion.training_losses(
            lambda *args, **kw: prediction, target, torch.tensor([0, 2]), model_kwargs=kwargs)
        self.assertEqual(set(disabled), {'loss', 'l_simple'})

    def test_rotation_step_weight_is_added_to_total_and_zero_disables_it(self):
        batch_size, joints, features, frames = 2, 3, 12, 5
        diffusion = self._make_diffusion(model_var_type=ModelVarType.FIXED_LARGE,
                                         lambda_drot=.17, drot_joint_boost=3.)
        target = torch.randn(batch_size, joints, features, frames)
        noise = torch.randn_like(target)
        ts = torch.tensor([0, 2])
        kwargs = self._make_model_kwargs(batch_size, joints, features, frames)
        y = kwargs['y']
        y['canonical_feature_mean'] = torch.linspace(-.2, .2, CANONICAL_STAT_LEN)
        y['canonical_feature_std'] = torch.linspace(.5, 2., CANONICAL_STAT_LEN)
        model = _RecordingModel()
        enabled = diffusion.training_losses(model, target, ts, model_kwargs=kwargs, noise=noise)
        # The training path must use global steps after undoing standardization.
        expected_global = diffusion.rotation_step_loss(
            model.last_x, target, torch.ones(batch_size, 1, 1, joints), y['lengths'], y,
            t=ts, rotations=(diffusion._canonical_global_rotation_matrices(target, y),
                             diffusion._canonical_global_rotation_matrices(model.last_x, y)),
        )
        torch.testing.assert_close(enabled['drot_loss'], expected_global)
        torch.testing.assert_close(enabled['loss'], enabled['l_simple'] + .17 * enabled['drot_loss'])
        diffusion.lambda_drot = 0.
        disabled = diffusion.training_losses(model, target, ts, model_kwargs=kwargs, noise=noise)
        self.assertNotIn('drot_loss', disabled)
        torch.testing.assert_close(enabled['l_simple'], disabled['l_simple'])
        torch.testing.assert_close(disabled['loss'], disabled['l_simple'])

    def test_geodesic_and_bone_losses_share_one_rotation_conversion(self):
        batch_size, n_joints, n_feats, n_frames = 2, 3, 12, 5
        diffusion = self._make_diffusion(
            model_var_type=ModelVarType.FIXED_LARGE, lambda_geo=0.25, lambda_bone=0.5,
        )
        x_start = torch.randn(batch_size, n_joints, n_feats, n_frames, dtype=torch.float32)
        t = torch.tensor([0, 1], dtype=torch.int64)
        model_kwargs = self._make_model_kwargs(batch_size, n_joints, n_feats, n_frames)
        model_kwargs["y"]["parents"] = [np.array([-1, 0, 1])] * batch_size
        y_rest = model_kwargs["y"]["rest_pos_ric_hml"]
        model_kwargs["y"]["rest_pos_ric_hml"] = y_rest.unsqueeze(0).expand(batch_size, -1, -1)
        noise = torch.randn_like(x_start)

        convert = GaussianDiffusion._physical_rotation_matrices
        with patch.object(
            GaussianDiffusion, "_physical_rotation_matrices", side_effect=convert,
        ) as counted:
            terms = diffusion.training_losses(
                _RecordingModel(output_mode="zero"), x_start, t, model_kwargs=model_kwargs, noise=noise,
            )
        # One conversion each for the target and the prediction, not one per loss.
        self.assertEqual(counted.call_count, 2)

        # Same values as each loss converting on its own.
        y = model_kwargs["y"]
        from diffusion.gaussian_diffusion import canonical_to_physical_hml
        target_physical = canonical_to_physical_hml(x_start, y)
        pred_physical = canonical_to_physical_hml(torch.zeros_like(x_start), y)
        spat_mask = y["joints_padding_mask"][:, :, :, 1, 1:].float()
        geo = diffusion.geodesic_loss(
            target_physical, pred_physical, spat_mask,
            y["lengths"].float(), y["n_joints"].float(),
        )
        bone = diffusion.bone_vector_consistency_loss(pred_physical, target_physical, spat_mask, y, t=t)
        torch.testing.assert_close(terms["geodesic_loss"], geo)
        torch.testing.assert_close(terms["bone_loss"], bone)

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


    def test_anytop_forward_keeps_joint_key_padding_mask_padding_only(self):
        model = MotionDiffusion(
            max_joints=4,
            feature_len=12,
            latent_dim=8,
            ff_size=32,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            cross_limb=True,
        )
        capture_decoder = _CaptureDecoder()
        model.seqTransDecoder = capture_decoder
        model.train()

        x = torch.randn(1, 4, 12, 3, dtype=torch.float32)
        y = {
            "joints_padding_mask": torch.ones(1, 1, 1, 5, 5, dtype=torch.float32),
            "rest_geom": torch.randn(1, 4, REST_GEOM_DIM, dtype=torch.float32),
            "n_joints": torch.tensor([3], dtype=torch.int64),
            "joints_names_embs": torch.zeros(1, 4, 512, dtype=torch.float32),
            "joint_struct": torch.zeros(1, 4, JOINT_STRUCT_DIM, dtype=torch.float32),
            # Unconditional model input -- every forward reads the output frame.
            "canonical_feature_mean": torch.zeros(12, dtype=torch.float32),
            "canonical_feature_std": torch.ones(12, dtype=torch.float32),
            "parents": torch.tensor([[-1, 0, 1, 2]], dtype=torch.int64),
        }

        model(x, torch.tensor([1], dtype=torch.int64), y=y)

        expected = torch.tensor([[False, False, False, True]], dtype=torch.bool)
        self.assertIsNotNone(capture_decoder.last_kwargs)
        self.assertTrue(torch.equal(capture_decoder.last_kwargs["tgt_key_padding_mask"], expected))

    def test_anytop_forward_leaves_temporal_attention_unmasked(self):
        # Temporal self-attention is full: no window, so nothing but the spatial
        # mask reaches the decoder. The cross-limb block used to receive the same
        # window as a separate template argument -- it must still run without one.
        model = MotionDiffusion(
            max_joints=4,
            feature_len=12,
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

        x = torch.randn(2, 4, 12, 3, dtype=torch.float32)
        y = {
            "joints_padding_mask": torch.ones(2, 1, 1, 5, 5, dtype=torch.float32),
            "rest_geom": torch.randn(2, 4, REST_GEOM_DIM, dtype=torch.float32),
            "n_joints": torch.tensor([4, 3], dtype=torch.int64),
            "joints_names_embs": torch.zeros(2, 4, 512, dtype=torch.float32),
            "joint_struct": torch.zeros(2, 4, JOINT_STRUCT_DIM, dtype=torch.float32),
            # Unconditional model input -- every forward reads the output frame.
            "canonical_feature_mean": torch.zeros(12, dtype=torch.float32),
            "canonical_feature_std": torch.ones(12, dtype=torch.float32),
        }

        model(x, torch.tensor([1, 2], dtype=torch.int64), y=y)

        self.assertIsNotNone(capture_decoder.last_kwargs)
        # The windowed temporal mask and its per-head template are gone: the
        # decoder receives neither, so nothing but the spatial mask constrains
        # temporal attention.
        self.assertNotIn("temporal_mask", capture_decoder.last_kwargs)
        self.assertNotIn("temporal_template", capture_decoder.last_kwargs)
        self.assertEqual(
            capture_decoder.last_kwargs["spatial_mask"].shape, (2, model.num_heads, 4, 4))

    def test_anytop_runs_the_real_cross_limb_decoder_without_a_temporal_mask(self):
        # The capture decoder above never enters the attention blocks; this one
        # runs them, so a cross-limb block that still expected a window template
        # (or an attention that rejected attn_mask=None) would fail here.
        model = MotionDiffusion(
            max_joints=4,
            feature_len=12,
            latent_dim=8,
            ff_size=32,
            num_layers=2,
            num_heads=2,
            dropout=0.0,
            cross_limb=True,
        )
        model.eval()

        x = torch.randn(2, 4, 12, 3, dtype=torch.float32)
        y = {
            "joints_padding_mask": torch.ones(2, 1, 1, 5, 5, dtype=torch.float32),
            "rest_geom": torch.randn(2, 4, REST_GEOM_DIM, dtype=torch.float32),
            "n_joints": torch.tensor([4, 3], dtype=torch.int64),
            "joints_names_embs": torch.zeros(2, 4, 512, dtype=torch.float32),
            "joint_struct": torch.zeros(2, 4, JOINT_STRUCT_DIM, dtype=torch.float32),
            "graph_dist": torch.zeros(2, 4, 4, dtype=torch.int64),
            "joints_relations": torch.zeros(2, 4, 4, dtype=torch.int64),
            "canonical_feature_mean": torch.zeros(12, dtype=torch.float32),
            "canonical_feature_std": torch.ones(12, dtype=torch.float32),
        }

        out = model(x, torch.tensor([1, 2], dtype=torch.int64), y=y)

        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
