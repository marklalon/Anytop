"""Rotation-step supervision retains direction, seams and finite gradients."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffusion.gaussian_diffusion import GaussianDiffusion, LossType, ModelMeanType, ModelVarType
from utils.model_util import create_gaussian_diffusion
from utils.parser_util import train_args


def diffusion(**kwargs):
    return GaussianDiffusion(betas=np.array([.01, .2, .8]),
                             model_mean_type=ModelMeanType.START_X,
                             model_var_type=ModelVarType.FIXED_SMALL,
                             loss_type=LossType.MSE, **kwargs)


def physical(angles):
    # [B,J,T] angles -> physical features with the first two columns of Rz.
    zero = torch.zeros_like(angles)
    rot = torch.stack([angles.cos(), angles.sin(), zero,
                       -angles.sin(), angles.cos(), zero], dim=2)
    padding = torch.zeros(*angles.shape[:2], 3, angles.shape[-1], device=angles.device)
    return torch.cat([padding, rot, padding], dim=2)


def loss(pred, target, *, lengths=None, mask=None, loop=False, **kwargs):
    batch, joints, _, frames = pred.shape
    return diffusion().rotation_step_loss(
        pred, target, torch.ones(batch, 1, 1, joints) if mask is None else mask,
        torch.full((batch,), frames) if lengths is None else lengths,
        {'is_loop': torch.full((batch,), loop)}, **kwargs)


def test_exact_and_tiny_error_have_finite_gradients():
    target = physical(torch.tensor([[[0., .2, -.2, .1]]]))
    pred = target.clone().requires_grad_()
    exact = loss(pred, target)
    torch.testing.assert_close(exact, torch.zeros(1))
    exact.sum().backward()
    assert torch.isfinite(pred.grad).all()
    angle = torch.tensor([[[0., .2 + 1e-5, -.2, .1]]], requires_grad=True)
    tiny = loss(physical(angle), target)
    tiny.sum().backward()
    assert tiny.item() > 0 and torch.isfinite(angle.grad).all()


def test_blur_and_wrong_direction_have_restoring_gradient():
    angles = torch.tensor([[[0., .3, 0., -.3, 0.]]])
    target = physical(angles)
    amplitude = torch.tensor(.5, requires_grad=True)
    attenuated = loss(physical(amplitude * angles), target)
    attenuated.sum().backward()
    assert amplitude.grad < 0
    static = loss(physical(torch.zeros_like(angles)), target)
    reversed_step = loss(physical(-angles), target)
    assert 0 < attenuated.item() < static.item() < reversed_step.item()


def test_constant_pose_offset_is_left_to_pose_losses():
    angles = torch.tensor([[[0., .2, .4]]])
    torch.testing.assert_close(loss(physical(angles + .7), physical(angles)),
                               torch.zeros(1), atol=1e-10, rtol=0)


def test_loop_wrap_uses_last_valid_frame_and_ignores_padding():
    target = physical(torch.tensor([[[0., .2, .4]]]))
    pred = physical(torch.zeros(1, 1, 3))
    assert loss(pred, target, loop=True).item() > loss(pred, target).item()
    padded_target = physical(torch.tensor([[[0., .2, .4, 2., -2.], [0., 0., 0., 1., 2.]]]))
    padded_pred = physical(torch.tensor([[[0., 0., 0., -1., 1.], [2., -2., 1., 0., 0.]]]))
    for loop in (False, True):
        actual = loss(padded_pred, padded_target, lengths=torch.tensor([3]),
                      mask=torch.tensor([[[[1., 0.]]]]), loop=loop)
        torch.testing.assert_close(actual, loss(pred, target, loop=loop))


def test_short_or_fully_padded_samples_have_zero_loss():
    target = physical(torch.tensor([[[0., .2, .4]]]))
    pred = physical(torch.zeros(1, 1, 3))
    for length in (0, 1):
        torch.testing.assert_close(loss(pred, target, lengths=torch.tensor([length]), loop=True), torch.zeros(1))
    torch.testing.assert_close(loss(pred, target, mask=torch.zeros(1, 1, 1, 1)), torch.zeros(1))
    torch.testing.assert_close(loss(pred[..., :1], target[..., :1], loop=True), torch.zeros(1))


def test_signal_fraction_scales_loss_without_renormalizing_it():
    target = physical(torch.tensor([[[0., .2, .4]]]))
    pred = physical(torch.zeros(1, 1, 3))
    for timestep in (0, 2):
        actual = loss(pred, target, t=torch.tensor([timestep]))
        torch.testing.assert_close(actual, loss(pred, target) * float(diffusion().alphas_cumprod[timestep]))


@pytest.mark.parametrize('weight', [-1., float('nan'), float('inf')])
def test_invalid_weights_rejected(weight):
    with pytest.raises(ValueError, match='lambda_drot'):
        diffusion(lambda_drot=weight)


def test_cli_weight_reaches_diffusion_and_defaults_off(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['train', '--cond_path', 'unused.npy', '--lambda_drot', '0.1'])
    args = train_args()
    assert create_gaussian_diffusion(args).lambda_drot == .1
    del args.lambda_drot
    assert create_gaussian_diffusion(args).lambda_drot == 0
    assert create_gaussian_diffusion(args).drot_joint_boost == 0
    args.drot_joint_boost = 3.
    assert create_gaussian_diffusion(args).drot_joint_boost == 3.


def test_global_steps_decode_stats_and_match_fk_with_nonidentity_rest_frames():
    from data_loaders.truebones.truebones_utils.canonical_features import (
        CANONICAL_STAT_LEN, physical_hml_to_canonical)
    # Child stays locally still while inheriting the parent's oscillation.
    target = physical(torch.tensor([[[0., .3, -.3, 0.], [0., 0., 0., 0.]]]))
    pred = physical(torch.zeros(1, 2, 4)).requires_grad_()
    cond = {'parents': np.array([-1, 0]),
            'rest_pos_ric_hml': torch.tensor([[0., 0., 0.], [1., 2., 3.]]),
            'canonical_feature_mean': torch.linspace(-.2, .2, CANONICAL_STAT_LEN),
            'canonical_feature_std': torch.linspace(.5, 2., CANONICAL_STAT_LEN)}
    d = diffusion(drot_joint_boost=3.)
    gt_global = d._canonical_global_rotation_matrices(physical_hml_to_canonical(target, cond), cond)
    pr_global = d._canonical_global_rotation_matrices(physical_hml_to_canonical(pred, cond), cond)
    # Select child only: local loss sees no error; global loss must see motion.
    mask = torch.tensor([[[[0., 1.]]]])
    args = (pred, target, mask, torch.tensor([4]), {'is_loop': True})
    torch.testing.assert_close(d.rotation_step_loss(*args), torch.zeros(1))
    actual = d.rotation_step_loss(*args, rotations=(gt_global, pr_global))
    target_local = d._physical_rotation_matrices(target)
    pred_local = d._physical_rotation_matrices(pred)
    gt_fk = torch.stack([target_local[:, :, 0], target_local[:, :, 0] @ target_local[:, :, 1]], 2)
    pr_fk = torch.stack([pred_local[:, :, 0], pred_local[:, :, 0] @ pred_local[:, :, 1]], 2)
    expected = d.rotation_step_loss(*args, rotations=(gt_fk, pr_fk))
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert actual.item() > 0
    actual.sum().backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad[:, 1, 3:9].abs().sum() > 0


def test_motion_weight_prioritizes_active_joint_and_preserves_scale():
    target = physical(torch.tensor([[[0., .3, 0.], [0., 0., 0.]]]))
    pred = physical(torch.zeros(1, 2, 3))
    mask = torch.ones(1, 1, 1, 2)
    args = (pred, target, mask, torch.tensor([3]), {})
    uniform = diffusion().rotation_step_loss(*args)
    weighted = diffusion(drot_joint_boost=3).rotation_step_loss(*args)
    torch.testing.assert_close(weighted, uniform * 1.6)
    # Equal errors on all joints preserve total scale despite redistribution.
    shifted = physical(torch.tensor([[[0., .4, .2], [0., .1, .2]]]))
    torch.testing.assert_close(diffusion().rotation_step_loss(shifted, target, *args[2:]),
                               diffusion(drot_joint_boost=3).rotation_step_loss(shifted, target, *args[2:]))


def test_weight_ignores_padding_and_static_target_is_finite():
    d = diffusion(drot_joint_boost=3)
    target = physical(torch.tensor([[[0., .2, .4], [0., 0., 0.]]]))
    pred = physical(torch.zeros(1, 2, 3)).requires_grad_()
    mask = torch.ones(1, 1, 1, 2)
    base = d.rotation_step_loss(pred, target, mask, torch.tensor([3]), {'is_loop': True})
    padded = physical(torch.tensor([[[0., .2, .4, 2.], [0., 0., 0., -2.], [0., 2., -2., 1.]]]))
    padded_pred = physical(torch.zeros(1, 3, 4))
    actual = d.rotation_step_loss(padded_pred, padded, torch.tensor([[[[1., 1., 0.]]]]),
                                 torch.tensor([3]), {'is_loop': True})
    torch.testing.assert_close(base, actual)
    static = physical(torch.zeros(1, 2, 3))
    value = d.rotation_step_loss(pred, static, mask, torch.tensor([3]), {})
    value.sum().backward()
    assert torch.isfinite(pred.grad).all()
    torch.testing.assert_close(value, torch.zeros(1))


@pytest.mark.parametrize('boost', [-1., float('nan'), float('inf')])
def test_invalid_joint_boost_rejected(boost):
    with pytest.raises(ValueError, match='drot_joint_boost'):
        diffusion(drot_joint_boost=boost)


def test_zero_6d_prediction_has_restoring_gradient_for_moving_target():
    target = physical(torch.tensor([[[0., .2, .4]]]))
    pred = torch.zeros_like(target, requires_grad=True)
    initial = loss(pred, target)
    initial.sum().backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad[:, :, 3:9].abs().sum() > 0
    # Stay within the fallback neighbourhood and verify the gradient descends.
    step = pred.grad / pred.grad.abs().max() * 1e-5
    assert loss(pred.detach() - step, target).item() < initial.item()
    static = physical(torch.zeros(1, 1, 3))
    torch.testing.assert_close(loss(torch.zeros_like(static), static), torch.zeros(1))


@pytest.mark.parametrize('batched_stats', [False, True])
def test_shared_decode_matches_separate_rotations_and_gradients(batched_stats):
    from data_loaders.truebones.truebones_utils.canonical_features import (
        CANONICAL_STAT_LEN, _apply_global_stats, canonical_to_physical_hml)
    torch.manual_seed(31)
    x = torch.randn(2, 3, 12, 5, requires_grad=True)
    mean = torch.linspace(-.2, .2, CANONICAL_STAT_LEN)
    std = torch.linspace(.5, 2., CANONICAL_STAT_LEN)
    if batched_stats:
        mean = torch.stack((mean, mean + .1))
        std = torch.stack((std, std * 1.2))
    y = {'parents': np.array([-1, 0, 1]),
         'rest_pos_ric_hml': torch.tensor([[0., 0., 0.], [1., 2., 3.], [2., 4., 3.]]),
         'canonical_feature_mean': mean, 'canonical_feature_std': std}
    d = diffusion()
    expected_physical = canonical_to_physical_hml(x, y)
    expected_rot = d._physical_rotation_matrices(_apply_global_stats(x, y, inverse=True))
    sliced_rot = d._canonical_global_rotation_matrices(x, y)
    shared_physical, shared_rot = canonical_to_physical_hml(x, y, return_canonical_rotations=True)
    torch.testing.assert_close(shared_physical, expected_physical)
    torch.testing.assert_close(shared_rot, expected_rot)
    torch.testing.assert_close(sliced_rot, expected_rot)
    weights = torch.randn_like(expected_rot)
    expected_grad = torch.autograd.grad(
        expected_physical.square().sum() + (expected_rot * weights).sum(), x, retain_graph=True)[0]
    shared_grad = torch.autograd.grad(
        shared_physical.square().sum() + (shared_rot * weights).sum(), x, retain_graph=True)[0]
    sliced_grad = torch.autograd.grad(
        expected_physical.square().sum() + (sliced_rot * weights).sum(), x)[0]
    torch.testing.assert_close(shared_grad, expected_grad)
    torch.testing.assert_close(sliced_grad, expected_grad)
