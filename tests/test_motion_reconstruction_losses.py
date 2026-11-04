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


def bone_fixture():
    rest = torch.tensor([[0., 0, 0], [1., 0, 0], [2., 0, 0]])
    physical = torch.zeros(1, 3, 12, 4)
    physical[:, :, :3] = rest[None, :, :, None]
    physical[:, :, 3:9] = torch.tensor([1., 0, 0, 0, 1, 0])[None, None, :, None]
    y = dict(parents=[np.array([-1, 0, 1])], rest_pos_ric_hml=rest[None],
             rest_length_scale=torch.ones(1))
    return physical, y, torch.ones(1, 1, 1, 3)


def test_bone_length_ignores_direction_and_leaf_rotation_but_vector_metric_detects_them():
    target, y, mask = bone_fixture()
    pred = target.clone()
    pred[0, 2, :3] = torch.tensor([1., 1, 0])[:, None]
    pred[0, 2, 3:9] = torch.tensor([0., 1, 0, -1, 0, 0])[:, None]
    criterion = MotionReconstructionLosses()
    assert criterion.bone_length_loss(pred, target, mask, y) == 0
    assert criterion.bone_vector_consistency_loss(pred, target, mask, y) > 0


def test_bone_length_matches_animated_gt_and_has_position_only_finite_gradients():
    target, y, mask = bone_fixture()
    target[:, 2, 0] += torch.tensor([0., .1, .2, .3])
    criterion = MotionReconstructionLosses()
    assert criterion.bone_length_loss(target, target, mask, y) == 0
    pred = target.clone()
    pred[:, 2, 0] += .2
    pred.requires_grad_()
    loss = criterion.bone_length_loss(pred, target, mask, y)
    # Only one of two real bones has relative error .2; no virtual-bone dilution.
    torch.testing.assert_close(loss, torch.tensor(.075))
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad[:, :, :3].abs().sum() > 0
    assert pred.grad[:, :, 3:].count_nonzero() == 0
    assert criterion.bone_length_loss(bone_fixture()[0], target, mask, y) > 0


def test_bone_length_floor_padding_translation_and_collapsed_bone_gradients():
    target, y, mask = bone_fixture()
    target[:, 2, 0] = 1.01
    y['rest_pos_ric_hml'][:, 2, 0] = 1.01
    pred = target.clone()
    pred[:, 2, 0] += .005
    criterion = MotionReconstructionLosses()
    expected = torch.tensor(5 * (.005 / .06) ** 2 / 2)
    torch.testing.assert_close(criterion.bone_length_loss(pred, target, mask, y), expected)
    # World root XZ and shared height translation cannot change RIC bone lengths.
    pred[:, 0, [0, 2]] += 10
    pred[:, :, 1] += 3
    torch.testing.assert_close(criterion.bone_length_loss(pred, target, mask, y), expected)
    pred = torch.nn.functional.pad(pred, (0, 0, 0, 0, 0, 1), value=100.).requires_grad_()
    target = torch.nn.functional.pad(target, (0, 0, 0, 0, 0, 1), value=-100.)
    y['parents'] = [np.array([-1, 0, 1, -1])]
    y['rest_pos_ric_hml'] = torch.nn.functional.pad(y['rest_pos_ric_hml'], (0, 0, 0, 1))
    mask = torch.tensor([[[[1., 1, 1, 0]]]])
    loss = criterion.bone_length_loss(pred, target, mask, y)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert pred.grad[:, 3:].count_nonzero() == 0
    collapsed = target.clone()
    collapsed[:, 2, :3] = collapsed[:, 1, :3]
    collapsed.requires_grad_()
    criterion.bone_length_loss(collapsed, target, mask, y).backward()
    assert torch.isfinite(collapsed.grad).all()


def test_evaluation_reports_both_bone_metrics_independently_of_batch_size():
    from train.train_motion_autoencoder import SKELETON_KEYS, evaluate

    class OffsetAutoencoder:
        def eval(self):
            return self

        def encode(self, x, skeleton, frame, loop):
            return x

        def decode(self, z, skeleton, frame, loop):
            pred = z.clone()
            pred[:, 1, 0] += .1
            return pred

    _, target, y = fixture()
    y.update({key: torch.zeros(1, 3, 1) for key in SKELETON_KEYS if key not in y})
    y = {key: value.repeat(2, *([1] * (value.ndim - 1))) if torch.is_tensor(value)
         else value * 2 for key, value in y.items()}
    y.update(motion_name=['a', 'b'], object_type=['test', 'test'])
    y.update(loop_tile_count=torch.ones(2, dtype=torch.long),
             resample_speed_cond=torch.ones(2))
    target = target.repeat(2, 1, 1, 1)
    first, _ = evaluate(OffsetAutoencoder(), target, y, 1)
    second, _ = evaluate(OffsetAutoencoder(), target, y, 2)
    for key in ('bone_length_error', 'bone_vector_error', 'root_pos_mse',
                'root_dpos_error', 'root_acc_error'):
        assert first['summary'][key] >= 0
        assert first['summary'][key] == pytest.approx(second['summary'][key])
        assert all(key in record for record in first['samples'])


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('loop', [False, True])
def test_ae_root_xyz_dynamics_match_axis_scales_species_weights_and_wrap(axis, loop):
    _, target, y = fixture(frames=5)
    y['is_loop'][:] = loop
    y['canonical_feature_std'][:, cf.ROOT_STAT_SLICE] = torch.tensor([2., 3, 4])
    y['canonical_dpos_std'][:] = .5
    y['dpos_loss_weight'] = torch.tensor([.25])
    pred = target.clone()
    residual = np.array([0., 1., -1., 2., 0.], dtype=np.float32)
    pred[:, 0, axis] += torch.from_numpy(residual)
    pred.requires_grad_()
    terms = MotionReconstructionLosses(lambda_root_dpos=1, lambda_root_acc=1)(pred, target, y)
    scale = (axis + 2) / .5
    step = np.roll(residual, -1) - residual if loop else np.diff(residual)
    acc = np.roll(step, -1) - step if loop else np.diff(step)
    torch.testing.assert_close(terms['root_dpos_loss'], torch.tensor(float(np.mean(step ** 2) * scale ** 2 * .25 / 3)))
    torch.testing.assert_close(terms['root_acc_loss'], torch.tensor(float(np.mean(acc ** 2) * scale ** 2 * .25 / 3)))
    (terms['root_dpos_loss'] + terms['root_acc_loss']).backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad[:, 0, axis].abs().sum() > 0
    assert pred.grad[:, 1:].count_nonzero() == 0
    assert pred.grad[:, 0, [i for i in range(9) if i != axis]].count_nonzero() == 0


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_root_position_detects_constant_offsets_without_joint_or_species_dilution(axis):
    _, target, y = fixture()
    pred = target.clone()
    pred[:, 0, axis] += .3
    pred.requires_grad_()
    y['dpos_loss_weight'] = torch.tensor([.001])
    criterion = MotionReconstructionLosses(lambda_root_pos=.2, lambda_root_dpos=1, lambda_root_acc=1)
    terms = criterion(pred, target, y)
    torch.testing.assert_close(terms['root_pos_loss'], torch.tensor(.3 ** 2 / 3))
    torch.testing.assert_close(terms['root_dpos_loss'], torch.tensor(0.), atol=1e-10, rtol=0)
    torch.testing.assert_close(terms['root_acc_loss'], torch.tensor(0.), atol=1e-10, rtol=0)
    torch.testing.assert_close(terms['loss'], terms['pose_mse'] + .2 * terms['root_pos_loss'])
    terms['root_pos_loss'].backward()
    assert pred.grad[:, 0, axis].abs().sum() > 0
    assert pred.grad[:, 1:].count_nonzero() == 0


def test_root_position_masks_time_padding_and_does_not_require_step_statistics():
    target = torch.zeros(2, 4, 9, 5)
    pred = target.clone()
    pred[:, 0, :3] = 2
    pred[0, 0, :3, 3:] = 100
    pred.requires_grad_()
    y = {'lengths': torch.tensor([3, 5])}
    loss = MotionReconstructionLosses().root_position_loss(target, pred, y)
    torch.testing.assert_close(loss, torch.full((2,), 4.))
    loss.sum().backward()
    assert pred.grad[0, ..., 3:].count_nonzero() == 0
    assert pred.grad[:, 1:].count_nonzero() == 0


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
        'root_dpos_loss': diffusion.root_position_step_loss(target, pred, y, channels=(0, 1, 2)),
        'root_acc_loss': diffusion.root_acceleration_loss(target, pred, y, channels=(0, 1, 2)),
        'root_pos_loss': diffusion.root_position_loss(target, pred, y),
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
def test_warmup_only_scales_dpos_drot_acc_contributions_and_gradients(scale):
    pred, target, y = fixture()
    pred.requires_grad_()
    weights = {argument: .1 for argument in LOSS_WEIGHTS.values()}
    full = MotionReconstructionLosses(**weights)(pred, target, y)
    warmed = MotionReconstructionLosses(**weights)(pred, target, y, step_loss_scale=scale)
    for key in full.keys() - {'loss'}:
        torch.testing.assert_close(warmed[key], full[key], rtol=0, atol=0)
    # Other auxiliaries and pose retain their full gradient throughout warm-up.
    without_steps = dict(weights, lambda_dpos=0., lambda_drot=0., lambda_acc=0.)
    base = MotionReconstructionLosses(**without_steps)(pred, target, y)['loss']
    expected = base + scale * .1 * (full['dpos_loss'] + full['drot_loss'] + full['acc_loss'])
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
