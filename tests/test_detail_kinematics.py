"""Regression cases for supervision, RIC geometry and full training/export paths."""
import copy
from types import SimpleNamespace

import pytest
import torch

from model.motion_detail_refiner import (RefinerConfig, MotionDetailRefiner, compose,
    columns6, exp_so3, fk, ric_body_positions)
from tests.test_motion_detail_refiner import fixture
from train.detail_objectives import detail_features, empirical_target, statistics_loss, geometry_objective
from train.detail_metrics import boundary_acceleration, preservation_metrics
from train.detail_runtime import BatchedDetailRuntime
from train.train_motion_detail import prepare_groups, group_loss, matched_identity_loss, training_context
from sample.detail_refinement import tree_to, transport_export_positions, normalize_backbone_contract
from sample.output_lengths import _resample_window_to_output


def small_config():
    return RefinerConfig(width=16, layers=2, name_dim=8, action_dim=12,
                         max_angle_rad=.3)


def lengths(x, parents):
    p = ric_body_positions(x[..., :3])
    return torch.stack([(p[:, j] - p[:, parent]).norm(dim=-1)
                        for j, parent in enumerate(parents) if parent >= 0], -1)


@pytest.mark.parametrize('parents', [[-1, 0, 1, 0], [-1, 3, 0, 2], [-1, 0, -1, 2]])
def test_transport_arbitrary_geometry_nonunit_rotation_root_path_and_zero(parents):
    _, base, y = fixture()
    x = base[0].permute(2, 0, 1).clone()
    x[..., :3] += torch.randn_like(x[..., :3]) * .1
    x[:, 0, 0] = torch.arange(len(x)) * .03
    x[..., 3:9] += torch.randn_like(x[..., 3:9]) * .1
    rest = y['rest_pos_ric_hml'][0]
    zero = torch.zeros_like(x[..., :3])
    assert torch.equal(compose(x, zero, rest, parents, 'bone_transport'), x)
    delta = (torch.randn_like(zero) * .1).requires_grad_()
    out = compose(x, delta, rest, parents, 'bone_transport')
    torch.testing.assert_close(lengths(out, parents), lengths(x, parents), atol=8e-7, rtol=2e-6)
    for j, parent in enumerate(parents):
        if parent < 0:
            assert torch.equal(out[:, j], x[:, j])
    out.square().mean().backward()
    assert torch.isfinite(delta.grad).all() and delta.grad.abs().sum() > 0


def test_statistics_root_invariance_geometry_independence_and_source_counts():
    _, base, y = fixture(True)
    c = small_config()
    x = base[0].permute(2, 0, 1)
    from utils.rotation_conversions import rotation_6d_to_matrix_safe
    r = rotation_6d_to_matrix_safe(x[..., 3:9])
    rotation = exp_so3(torch.tensor([0., .7, 0.]))
    rotated = base.clone()
    r[:, 0] = rotation @ r[:, 0]
    rotated[0, :, 3:9] = columns6(r).permute(1, 2, 0)
    rotated[:, :, :3] += torch.randn_like(rotated[:, :, :3])  # Redundant geometry is not a target.
    torch.testing.assert_close(detail_features(base, y, c), detail_features(rotated, y, c), atol=1e-6, rtol=1e-5)
    f = torch.tensor([[1., 2.], [3., 4.], [7., 8.]])
    target = empirical_target(f, ['a', 'a', 'b'])
    torch.testing.assert_close(target, torch.tensor([[2., 3.], [7., 8.]]))
    a, b = torch.tensor([[-1.], [1.]]), torch.zeros(1, 1)
    assert statistics_loss(a, b, torch.ones(1)) == 0
    # Static correction and a valid periodic endpoint gap incur no seam anchor.
    assert geometry_objective(torch.tensor(0.)) == 0


@pytest.mark.parametrize('loop,frames', [(True, 37), (False, 23), (False, 1)])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_batched_serial_full_and_crop_gradients(loop, frames, device, full_precision_matmul):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    _, base, y = fixture(loop, frames)
    base[:, 1, 3] += .03
    base = torch.nn.functional.pad(base, (0, 2, 0, 0, 0, 1))
    g = tree_to(dict(y=y, gt=base.clone(), train_base=base.repeat(2, 1, 1, 1),
                     validation_base=base.clone(), clips=['one']), device)
    g['y'] = training_context(g['y'], 1)
    c = small_config()
    scales = prepare_groups([g], c, batched=True)
    net = MotionDetailRefiner(c).to(device)
    with torch.no_grad():
        net.input.weight[:, -102:].normal_(std=.03)
        net.output.weight.normal_(std=.02)  # Exercise active guard and context gradients.
    runtime = BatchedDetailRuntime(net)
    serial, so, sd = group_loss(net, g, 'train_base', 67, scales[0])
    actual, ao, ad = group_loss(net, g, 'train_base', 67, scales[0], runtime)
    torch.testing.assert_close(ao, so, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(ad, sd, atol=3e-6, rtol=3e-5)
    for k in serial:
        torch.testing.assert_close(actual[k], serial[k], atol=4e-6, rtol=1e-4)
    one = matched_identity_loss(net, g, 67, scales[0])[0]
    two = matched_identity_loss(net, g, 67, scales[0], runtime)[0]
    torch.testing.assert_close(one, two, atol=3e-4, rtol=2e-4)
    one.backward()
    grads = [p.grad.clone() for p in net.parameters()]
    net.zero_grad(set_to_none=True)
    two.backward()
    for p, grad in zip(net.parameters(), grads):
        torch.testing.assert_close(p.grad, grad, atol=.003, rtol=.003)


@pytest.mark.parametrize('loop,count', [(True, 20), (False, 20), (False, 85)])
def test_native_export_transport_preserves_resampled_base(loop, count):
    _, base, y = fixture(loop)
    x = base[0].permute(2, 0, 1)
    parents = y['parents'][0]
    refined = compose(x, torch.randn_like(x[..., :3]) * .2,
                      y['rest_pos_ric_hml'][0], parents, 'bone_transport')
    before = _resample_window_to_output(x.numpy(), count, 60, loop)
    after = _resample_window_to_output(refined.numpy(), count, 60, loop)
    args = SimpleNamespace(_detail_export_base=(base, y))
    fixed = torch.from_numpy(transport_export_positions(after, args, 0, count, 60, loop))
    torch.testing.assert_close(lengths(fixed, parents), lengths(torch.from_numpy(before), parents),
                               atol=1e-6, rtol=2e-6)
    assert torch.equal(fixed[:, 0], torch.from_numpy(before)[:, 0])
    assert transport_export_positions(after, SimpleNamespace(), 0, count, 60, loop) is after


def test_known_contract_migration_only():
    assert normalize_backbone_contract({'position_contract': 'base_plus_fk_difference_v1', 'a': 1}) == {'a': 1}
    with pytest.raises(ValueError):
        normalize_backbone_contract({'position_contract': 'unrecognized'})


def test_boundary_metric_does_not_penalize_dc_and_detects_seam_jump():
    signal = torch.sin(torch.arange(60)[:, None] * 2 * torch.pi * 3 / 60)
    _, excess = boundary_acceleration(signal, True)
    _, shifted = boundary_acceleration(signal + 10, True)
    assert float((excess - shifted).abs()) < 3e-6
    assert excess < 1e-5
    broken = signal.clone()
    broken[0] += 2
    assert boundary_acceleration(broken, True)[1] > 1


def test_duplicate_windows_do_not_create_identity_statistical_error():
    _, base, y = fixture()
    changed = base.clone()
    changed[:, 1, 3:9] = changed[:, 1, 3:9].roll(8, -1)
    gt = torch.cat((base, base, changed))
    g = dict(y=y, gt=gt, train_base=base, validation_base=base, clips=['a', 'a', 'b'])
    c = small_config()
    scales = prepare_groups([g], c, batched=True)
    net = MotionDetailRefiner(c)
    with torch.no_grad():
        net.output.weight.zero_()
        net.output.bias.zero_()
    serial = group_loss(net, g, 'matched_base', 5, scales[0])[0]['stat']
    batched = group_loss(net, g, 'matched_base', 5, scales[0], BatchedDetailRuntime(net))[0]['stat']
    assert abs(float(serial.detach())) < 1e-6 and abs(float(batched.detach())) < 1e-6


def test_body_amplitude_diagnostic_excludes_world_root_path():
    c, base, y = fixture(True)
    out = base.clone()
    out[:, 2, 0] *= .5
    expected = preservation_metrics(out, base, y, c)['body_amplitude_relative_change']
    path_base, path_out = base.clone(), out.clone()
    path = torch.arange(base.shape[-1]) * 100.
    path_base[:, 0, 0] = path_out[:, 0, 0] = path
    actual = preservation_metrics(path_out, path_base, y, c)['body_amplitude_relative_change']
    assert expected > .01
    assert actual == pytest.approx(expected, abs=1e-6)


def test_real_training_resume_and_checkpoint_loading(tmp_path):
    from tests.test_train_motion_detail import make_training_cache
    from train.train_motion_detail import main
    from sample.detail_refinement import load_refiner
    cache, _ = make_training_cache(tmp_path)
    save = tmp_path / 'train'
    args = ['--cache-dir', str(cache), '--save-dir', str(save),
            '--width', '16', '--layers', '2', '--max-angle-rad', '.3', '--lr-schedule', 'cosine',
            '--updates-per-group', '1', '--eval-every', '0', '--save-every', '2',
            '--log-every', '2', '--no-progress', '--device', 'cpu']
    main(args)
    net, payload = load_refiner(save / 'refiner_latest.pt', {}, 'cpu')
    assert net.config.calibration_version == 3
    assert payload['step'] == 2
    main(args + ['--updates-per-group', '2'])
    _, resumed = load_refiner(save / 'refiner_latest.pt', {}, 'cpu')
    assert resumed['step'] == 4 and resumed['lr_schedule_total_steps'] == 2
