"""AE loss integration and compatibility with diffusion's existing reductions."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loaders.truebones.truebones_utils import canonical_features as cf
from diffusion.gaussian_diffusion import GaussianDiffusion, LossType, ModelMeanType, ModelVarType
from model.motion_autoencoder import to_working_motion
from train.motion_reconstruction_losses import LOSS_WEIGHTS, MotionReconstructionLosses


def fixture(frames=8):
    rest = torch.tensor([[0., 0, 0], [1., 0, 0], [-1., 0, 0]])
    y = dict(rest_pos_ric_hml=rest[None], parents=[np.array([-1, 0, 0])],
             rest_length_scale=torch.ones(1), n_joints=torch.tensor([3]),
             lengths=torch.tensor([frames]), is_loop=torch.tensor([True]),
             limb_tip_mask=torch.tensor([[0., 1, 1]]),
             canonical_feature_mean=torch.zeros(1, cf.CANONICAL_STAT_LEN),
             canonical_feature_std=torch.ones(1, cf.CANONICAL_STAT_LEN),
             canonical_dpos_std=torch.tensor([.1]))
    physical = torch.zeros(1, 3, 12, frames)
    t = torch.arange(frames) * (2 * torch.pi / max(1, frames))
    physical[0, :, :3] = rest[:, :, None]
    physical[0, 1, 1] += .2 * t.sin()
    physical[0, 2, 2] += .2 * t.cos()
    angles = t[None] * torch.tensor([[.1], [.2], [.3]])
    physical[0, :, 3] = angles.cos()
    physical[0, :, 4] = angles.sin()
    physical[0, :, 6] = -angles.sin()
    physical[0, :, 7] = angles.cos()
    target = cf.physical_hml_to_canonical(physical, y)[:, :, :9]
    pred = target.clone()
    pred[..., frames // 2] += .05
    return pred, target, y


@pytest.mark.parametrize('frames', [1, 2, 8])
def test_full_ae_losses_match_existing_diffusion_terms_and_have_finite_gradients(frames):
    pred, target, y = fixture(frames)
    pred.requires_grad_()
    weights = {argument: .1 for argument in LOSS_WEIGHTS.values()}
    losses = MotionReconstructionLosses(drot_joint_boost=1, **weights)
    actual = losses(pred, target, y)
    diffusion = GaussianDiffusion(betas=np.array([.01, .2, .8]),
                                  model_mean_type=ModelMeanType.START_X,
                                  model_var_type=ModelVarType.FIXED_LARGE,
                                  loss_type=LossType.MSE, drot_joint_boost=1)
    spat_mask = torch.ones(1, 1, 1, 3)
    gt, global_gt = cf.canonical_to_physical_hml(to_working_motion(target), y, return_canonical_rotations=True)
    decoded, global_pred = cf.canonical_to_physical_hml(to_working_motion(pred), y, return_canonical_rotations=True)
    expected = {
        'dpos_loss': diffusion.position_step_loss(target, pred, spat_mask, y['n_joints'], y),
        'acc_loss': diffusion.position_acceleration_loss(target, pred, spat_mask, y['n_joints'], y),
        'drot_loss': diffusion.rotation_step_loss(decoded, gt, spat_mask, y['lengths'], y,
                                                 rotations=(global_gt, global_pred)),
        'root_dpos_loss': diffusion.root_position_step_loss(target, pred, y),
        'root_acc_loss': diffusion.root_acceleration_loss(target, pred, y),
        'geodesic_loss': diffusion.geodesic_loss(gt, decoded, spat_mask, y['lengths'], y['n_joints']),
        'bone_loss': diffusion.bone_vector_consistency_loss(decoded, gt, spat_mask, y),
        'limb_corr_loss': diffusion.limb_correlation_loss(decoded, gt, y, y['lengths']),
        **diffusion.loop_wrap_loss(decoded, y, y['n_joints']),
    }
    for key, value in expected.items():
        torch.testing.assert_close(actual[key], value.mean(), rtol=0, atol=0)
    torch.testing.assert_close(actual['loss'], actual['pose_mse'] + sum(.1 * v.mean() for v in expected.values()))
    if frames == 8:
        assert actual['limb_corr_loss'] > 0
    actual['loss'].backward()
    assert torch.isfinite(pred.grad).all()


def test_zero_weights_preserve_pose_only_baseline_and_time_padding_is_rejected():
    pred, target, y = fixture()
    losses = MotionReconstructionLosses()
    torch.testing.assert_close(losses(pred, target, y)['loss'], (pred - target).square().mean())
    y['lengths'] -= 1
    with pytest.raises(ValueError, match='without time padding'):
        losses(pred, target, y)


@pytest.mark.parametrize('scale', [0.0, 0.005, 0.5, 1.0])
def test_warmup_only_scales_dpos_drot_contributions_and_gradients(scale):
    pred, target, y = fixture()
    pred.requires_grad_()
    weights = {argument: .1 for argument in LOSS_WEIGHTS.values()}
    full = MotionReconstructionLosses(**weights)(pred, target, y)
    warmed = MotionReconstructionLosses(**weights)(pred, target, y, step_loss_scale=scale)
    for key in full.keys() - {'loss'}:
        torch.testing.assert_close(warmed[key], full[key], rtol=0, atol=0)
    # Other auxiliaries and pose retain their full gradient throughout warm-up.
    without_steps = dict(weights, lambda_dpos=0., lambda_drot=0.)
    base = MotionReconstructionLosses(**without_steps)(pred, target, y)['loss']
    expected = base + scale * .1 * (full['dpos_loss'] + full['drot_loss'])
    torch.testing.assert_close(warmed['loss'], expected)
    actual_grad = torch.autograd.grad(warmed['loss'], pred, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, pred)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize('scale', [-.1, 1.1, float('nan')])
def test_invalid_step_loss_scale_is_rejected(scale):
    pred, target, y = fixture()
    with pytest.raises(ValueError, match='step_loss_scale'):
        MotionReconstructionLosses()(pred, target, y, step_loss_scale=scale)


def acceleration_fixture(frames=8):
    _, target, y = fixture(frames)
    y['canonical_dpos_std'] = torch.ones(1)
    y['is_loop'] = torch.tensor([False])
    return target, y, torch.ones(1, 1, 1, 3, dtype=torch.bool)


def test_acceleration_matches_target_without_penalizing_real_acceleration():
    target, y, mask = acceleration_fixture()
    target[:, 1, 0] = torch.arange(8).float().square()
    criterion = MotionReconstructionLosses()
    # A constant offset also has zero acceleration error despite moving GT.
    pred = (target + .25).requires_grad_()
    loss = criterion.position_acceleration_loss(target, pred, mask, y['n_joints'], y)
    torch.testing.assert_close(loss, torch.zeros(1), atol=1e-10, rtol=0)
    loss.sum().backward()
    assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize('loop', [False, True])
def test_acceleration_penalizes_alternating_jitter_with_known_magnitude(loop):
    target, y, mask = acceleration_fixture()
    y['is_loop'][:] = loop
    pred = target.clone()
    pred[:, 1, 0] += .125 * (-1.) ** torch.arange(8)
    pred.requires_grad_()
    loss = MotionReconstructionLosses().position_acceleration_loss(target, pred, mask, y['n_joints'], y)
    # Second difference amplitude = 4 * .125; mean over 3 joints * 3 axes.
    torch.testing.assert_close(loss, torch.tensor([.5 ** 2 / 9]))
    loss.sum().backward()
    assert pred.grad[:, 1, 0].abs().sum() > 0


@pytest.mark.parametrize('loop', [False, True])
def test_acceleration_padding_matches_unpadded_clip_and_has_zero_gradient(loop):
    target, y, mask = acceleration_fixture()
    y['is_loop'][:] = loop
    pred = target.clone()
    pred[:, 1, 0, 0] += .2  # Exercises the seam for a loop.
    criterion = MotionReconstructionLosses()
    expected = criterion.position_acceleration_loss(target, pred, mask, y['n_joints'], y)
    padded_target = torch.nn.functional.pad(target, (0, 3, 0, 0, 0, 2), value=100.)
    padded_pred = torch.nn.functional.pad(pred, (0, 3, 0, 0, 0, 2), value=-100.).requires_grad_()
    padded_mask = torch.tensor([[[[True, True, True, False, False]]]])
    actual = criterion.position_acceleration_loss(padded_target, padded_pred, padded_mask, y['n_joints'], y)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert padded_pred.grad[..., 8:].count_nonzero() == 0
    assert padded_pred.grad[:, 3:].count_nonzero() == 0


@pytest.mark.parametrize('frames', [1, 2])
def test_acceleration_short_clips_are_zero_with_finite_gradients(frames):
    target, y, mask = acceleration_fixture(frames)
    pred = (target + .2).requires_grad_()
    loss = MotionReconstructionLosses().position_acceleration_loss(target, pred, mask, y['n_joints'], y)
    torch.testing.assert_close(loss, torch.zeros(1))
    loss.sum().backward()
    assert torch.isfinite(pred.grad).all()


def test_acceleration_uses_canonical_scales_and_species_weight():
    target, y, mask = acceleration_fixture()
    pred = target.clone()
    pred[:, 1, 0] += .125 * (-1.) ** torch.arange(8)
    y['canonical_feature_std'][:, 0] = 2
    y['canonical_dpos_std'][:] = .5
    y['dpos_loss_weight'] = torch.tensor([3.])
    actual = MotionReconstructionLosses().position_acceleration_loss(target, pred, mask, y['n_joints'], y)
    torch.testing.assert_close(actual, torch.tensor([(.5 * 4) ** 2 / 9 * 3]))
