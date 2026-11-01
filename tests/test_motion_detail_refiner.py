import copy
import math
from types import SimpleNamespace

import pytest
import torch

from model.motion_detail_refiner import (
    MotionDetailRefiner, RefinerConfig, columns6, compose, exp_so3,
    fk, sample_context, spectral_project,
)
from sample.detail_refinement import refine_generated_batch
from tools.cache_motion_detail import repeat_context
from train.detail_objectives import detail_features, feature_scale, preservation_losses, statistics_loss
from utils.rotation_conversions import rotation_6d_to_matrix_safe


def fixture(loop=False, frames=60):
    torch.manual_seed(5)
    c = RefinerConfig(width=16, layers=2, name_dim=8, action_dim=12)
    rest = torch.tensor([[0., 0., 0.], [0., 1., 0.], [0., 2., 0.], [1., 1., 0.]])
    parents = [-1, 0, 1, 0]
    angle = torch.zeros(frames, 4, 3)
    angle[:, 1, 2] = .3 * torch.sin(torch.arange(frames) * 2 * math.pi / frames)
    local = exp_so3(angle)
    _, pos = fk(local, rest, parents)
    base = torch.zeros(1, 4, 12, frames)
    base[0, :, :3] = pos.permute(1, 2, 0)
    base[0, :, 3:9] = columns6(local).permute(1, 2, 0)
    y = {'n_joints': torch.tensor([4]), 'lengths': torch.tensor([frames]),
         'is_loop': torch.tensor([loop]), 'resample_speed_cond': torch.tensor([1.]),
         'parents': [parents], 'rest_pos_ric_hml': rest[None],
         'joint_struct': torch.randn(1, 4, 7), 'joints_names_embs': torch.randn(1, 4, 8),
         'detail_action': torch.randn(1, 12), 'loop_tile_count': torch.ones(1, dtype=torch.long)}
    return c, base, y


@pytest.mark.parametrize('loop', [False, True])
def test_so3_composition_and_exact_zero(loop):
    c, base, y = fixture(loop)
    x = base[0].permute(2, 0, 1)
    rest, parents = y['rest_pos_ric_hml'][0], y['parents'][0]
    assert torch.equal(compose(x, torch.zeros_like(x[..., :3]), rest, parents), x)
    delta = torch.randn_like(x[..., :3]) * .03
    out = compose(x, delta, rest, parents)
    assert torch.equal(out[:, 0], x[:, 0])
    r = rotation_6d_to_matrix_safe(out[..., 3:9])
    assert torch.allclose(r.transpose(-1, -2) @ r, torch.eye(3), atol=1e-6)
    _, before = fk(rotation_6d_to_matrix_safe(x[..., 3:9]), rest, parents)
    _, after = fk(r, rest, parents)
    assert torch.allclose(out[..., :3] - x[..., :3], after - before, atol=1e-6)


@pytest.mark.parametrize('loop', [False, True])
def test_band_projection_and_time_mapping(loop):
    c, _, y = fixture(loop)
    n = 60
    k = torch.arange(n)
    low = torch.sin(2 * math.pi * k / n) if loop else torch.cos(math.pi * (k + .5) / n)
    high = torch.sin(2 * math.pi * 16 * k / n) if loop else torch.cos(math.pi * 32 * (k + .5) / n)
    filtered = spectral_project((low + high)[:, None], 1 / 30, loop, low=4.5)
    assert torch.allclose(filtered[:, 0], high, atol=1e-5)
    # Same physical signal sampled at twice the rate retains its Hz meaning.
    t = torch.arange(120) / 60
    if loop:
        signal = torch.sin(2 * math.pi * 8 * t)
        assert torch.allclose(spectral_project(signal[:, None], 1/60, True, low=4.5)[:, 0], signal, atol=1e-5)
    y['resample_speed_cond'][0] = 2
    assert sample_context(y, 0, c)[3] == pytest.approx((2 if loop else 119/59) / 30)


@pytest.mark.parametrize('loop', [False, True])
def test_seed_isolation_bounds_and_padding(loop):
    c, base, y = fixture(loop)
    net = MotionDetailRefiner(c)
    rng = torch.random.get_rng_state().clone()
    out, delta = net(base, y, seed=99)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.equal(net(base, y, alpha=0)[0], base)
    assert torch.equal(out, net(base, y, seed=99)[0])
    assert not torch.equal(out, net(base, y, seed=100)[0])
    assert torch.equal(out[:, 0], base[:, 0])
    assert torch.linalg.vector_norm(delta, dim=2).max() <= c.max_angle_rad + 1e-6
    low = spectral_project(delta[0].permute(2, 0, 1), 1/30, loop, high=c.cutoff_hz)
    assert low.abs().max() > 1e-6  # All-band residuals include DC/low frequencies.
    padded = torch.randn(1, 7, 12, 75) * 100
    padded[:, :4, :, :60] = base
    yp = copy.deepcopy(y)
    for key in ('rest_pos_ric_hml', 'joint_struct', 'joints_names_embs'):
        yp[key] = torch.cat((yp[key], torch.randn(1, 3, yp[key].shape[-1])), 1)
    op, _ = net(padded, yp, seed=99)
    assert torch.equal(op[:, :4, :, :60], out)
    assert torch.equal(op[:, 4:], padded[:, 4:])
    assert torch.equal(op[..., 60:], padded[..., 60:])


def test_statistics_recover_weak_output_and_never_backprop_to_base():
    c, base, y = fixture(True)
    base.requires_grad_(True)
    net = MotionDetailRefiner(c)
    target_delta = torch.randn(60, 4, 3) * .05
    target_delta = spectral_project(target_delta, 1/30, True, low=c.cutoff_hz)
    target = compose(base[0].detach().permute(2, 0, 1), target_delta,
                     y['rest_pos_ric_hml'][0], y['parents'][0]).permute(1, 2, 0)[None]
    gt = detail_features(target, y, c).detach()
    out, delta = net(base, y, seed=123)
    stat = statistics_loss(detail_features(out, y, c), gt, feature_scale(gt))
    stat.backward()
    assert base.grad is None
    assert torch.isfinite(net.output.weight.grad).all()
    assert net.output.weight.grad.norm() > 1e-5
    net.zero_grad()
    out, delta = net(base, y, seed=123)
    losses = preservation_losses(out, base, delta, y, c)
    sum(losses.values()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
    assert base.grad is None


def test_disabled_adapter_is_exact_and_does_not_load_checkpoint(tmp_path):
    _, base, y = fixture()
    args = SimpleNamespace(refiner_path='does-not-exist.pt', refiner_alpha=0)
    assert refine_generated_batch(base, y, None, args, tmp_path) is base
    args.refiner_path = ''
    args.refiner_alpha = 1
    assert refine_generated_batch(base, y, None, args, tmp_path) is base


def test_empty_high_band_and_single_frame_are_safe():
    c, base, y = fixture(True, frames=1)
    out, delta = MotionDetailRefiner(c)(base, y)
    assert torch.isfinite(out).all() and torch.isfinite(delta).all()
    assert delta.count_nonzero() > 0  # A one-frame mean-pose repair is valid.
    assert spectral_project(torch.randn(20, 3), 1., False, low=4.5).count_nonzero() == 0


def test_checkpoint_contract_rejects_other_backbone(tmp_path):
    from sample.detail_refinement import DETAIL_SCHEMA, load_refiner
    c, _, _ = fixture()
    net = MotionDetailRefiner(c)
    path = tmp_path / 'refiner.pt'
    torch.save({'schema': DETAIL_SCHEMA, 'contract': {'backbone': 'v67'},
                'architecture': net.architecture(), 'refiner': net.state_dict()}, path)
    with pytest.raises(ValueError, match='mismatch'):
        load_refiner(path, {'backbone': 'v68'}, 'cpu')
    loaded, _ = load_refiner(path, {'backbone': 'v67'}, 'cpu')
    assert all(not p.requires_grad for p in loaded.parameters())


def test_output_nyquist_can_refuse_injection():
    c, base, y = fixture(True)
    y['detail_max_hz'] = [0.]
    out, delta = MotionDetailRefiner(c)(base, y)
    assert torch.equal(out, base)
    assert delta.count_nonzero() == 0


def test_physical_evaluation_identity_and_perturbation():
    from eval.eval_motion_detail import preservation_metrics
    c, base, y = fixture(True)
    metrics = preservation_metrics(base, base, y, c)
    assert all(v == 0 for k, v in metrics.items() if 'acceleration' not in k)
    for quantity in ('position', 'rotation'):
        for suffix in ('acceleration', 'acceleration_excess'):
            assert metrics[f'boundary_{quantity}_{suffix}'] == metrics[f'boundary_base_{quantity}_{suffix}']
    changed = base.clone()
    changed[:, 1, 0] += .1
    metrics = preservation_metrics(changed, base, y, c)
    assert metrics['lowfreq_position_rms_relative'] > .01
    assert metrics['bone_length_rms_relative'] > 0


def test_cache_seed_expansion_keeps_topology_and_conditions():
    c, base, y = fixture()
    y3 = repeat_context(y, 3)
    assert y3['detail_action'].shape == (3, 12)
    assert y3['parents'] == [y['parents'][0]] * 3
    net = MotionDetailRefiner(c)
    out, _ = net(base.expand(3, -1, -1, -1), y3, seed=10)
    assert not torch.equal(out[0], out[1])


def test_full_step_does_not_mutate_frozen_backbone_parameters_or_buffers():
    from tools.cache_motion_detail import frozen_state_hash
    c, base, y = fixture()
    backbone = torch.nn.Sequential(torch.nn.Linear(12, 12), torch.nn.BatchNorm1d(12)).eval().requires_grad_(False)
    before = frozen_state_hash(backbone)
    net = MotionDetailRefiner(c)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    with torch.no_grad():
        generated = backbone(base.permute(0, 1, 3, 2).reshape(-1, 12)).reshape(1, 4, 60, 12).permute(0, 1, 3, 2)
    out, delta = net(generated, y)
    (out.square().mean() + delta.square().mean()).backward()
    optimizer.step()
    assert frozen_state_hash(backbone) == before
    assert all(p.grad is None for p in backbone.parameters())
    assert {id(p) for group in optimizer.param_groups for p in group['params']} == {id(p) for p in net.parameters()}


def test_mixed_topologies_lengths_and_boundaries():
    c, base, y = fixture(True)
    mixed = repeat_context(y, 2)
    mixed['parents'] = [y['parents'][0], [-1, 0, 0]]
    mixed['n_joints'][1] = 3
    mixed['lengths'][1] = 37
    mixed['is_loop'][1] = False
    mixed['resample_speed_cond'][1] = .7
    net = MotionDetailRefiner(c)
    out, delta = net(base.expand(2, -1, -1, -1).clone(), mixed)
    assert torch.equal(out[1, 3], base[0, 3])
    assert torch.equal(out[1, :, :, 37:], base[0, :, :, 37:])
    assert torch.equal(out[:, 0], base[:, 0].expand(2, -1, -1))
    assert torch.isfinite(out).all() and torch.isfinite(delta).all()
