"""All-band repair, matched-base restraint and real fast-motion statistics."""
import pytest
import torch

from model.motion_detail_refiner import MotionDetailRefiner, compose, exp_so3, columns6, fk
from sample.detail_refinement import DETAIL_SCHEMA, load_refiner
from tests.test_motion_detail_refiner import fixture
from train.detail_objectives import (detail_features, feature_scale, statistics_loss,
                                     preservation_losses, coordination_pairs)
from train.train_motion_detail import main
from train.detail_objectives import geometry_objective


def composed(base, y, delta):
    return compose(base[0].permute(2, 0, 1), delta, y['rest_pos_ric_hml'][0],
                   y['parents'][0]).permute(1, 2, 0)[None]


def stat(out, target, y, config):
    features = detail_features(target, y, config).detach()
    return statistics_loss(detail_features(out, y, config), features, feature_scale(features))


@pytest.mark.parametrize('frequency', [0., 1., 9.])
def test_missing_dc_low_or_high_motion_provides_repair_gradient(frequency):
    config, base, y = fixture(True)
    wave = torch.ones(60) if frequency == 0 else torch.sin(torch.arange(60) * 2 * torch.pi * frequency / 30)
    unit = torch.zeros(60, 4, 3)
    unit[:, 1, 2] = wave
    target = composed(base, y, .04 * unit)
    strength = torch.tensor(0., requires_grad=True)
    loss = stat(composed(base, y, strength * unit), target, y, config)
    gradient, = torch.autograd.grad(loss, strength)
    assert torch.isfinite(gradient) and gradient < -1e-4
    assert stat(target, target, y, config).abs() < 1e-6


def test_full_band_minimum_change_has_no_low_frequency_exception():
    config, base, y = fixture(True)
    costs = []
    for frequency in [1., 9.]:
        delta = torch.zeros(60, 4, 3)
        delta[:, 3, 0] = .02 * torch.sin(torch.arange(60) * 2 * torch.pi * frequency / 30)
        out = composed(base, y, delta)
        losses = preservation_losses(out, base, delta.permute(1, 2, 0)[None], y, config)
        costs.append(losses['keep'])
    torch.testing.assert_close(costs[0], costs[1], atol=1e-7, rtol=1e-3)


def test_fast_large_wing_is_a_valid_target_and_inherited_motion_is_observed():
    config, base, y = fixture(True)
    angle = torch.zeros(60, 4, 3)
    angle[:, 1, 2] = 1.2 * torch.sin(torch.arange(60) * 2 * torch.pi / 3)
    rotation = exp_so3(angle)
    _, position = fk(rotation, y['rest_pos_ric_hml'][0], y['parents'][0])
    target = base.clone()
    target[0, :, :3] = position.permute(1, 2, 0)
    target[0, :, 3:9] = columns6(rotation).permute(1, 2, 0)
    still = target[..., :1].expand_as(target)
    assert stat(target, target, y, config).abs() < 1e-6
    assert stat(still, target, y, config) > .01
    # A common phase shift is allowed; this is not waveform reconstruction.
    assert stat(target.roll(7, -1), target, y, config).abs() < 1e-5
    # Joint 2 has identity local rotation but inherits joint 1's fast motion.
    # Inspect its local/global RMS channels in the highest analysis bin.
    joint_count = base.shape[1] - 1
    bin_size = 24 * joint_count + len(coordination_pairs(y['parents'][0])[0])
    rms_offset = 21 * joint_count + 2 * bin_size + 21  # non-root joint 2
    values = detail_features(target, y, config)[0]
    assert values[rms_offset:rms_offset + 9].max() < 2e-5
    assert values[rms_offset + 9:rms_offset + 18].max() > .1


def test_sibling_coordination_is_not_reduced_to_individual_energy():
    config, base, y = fixture(True)
    outputs, amplitudes = [], []
    for sign in [1., -1.]:
        angle = torch.zeros(60, 4, 3)
        wave = .7 * torch.sin(torch.arange(60) * 2 * torch.pi * 3 / 30)
        angle[:, 1, 2], angle[:, 3, 2] = wave, sign * wave
        rotation = exp_so3(angle)
        _, position = fk(rotation, y['rest_pos_ric_hml'][0], y['parents'][0])
        out = base.clone()
        out[0, :, :3] = position.permute(1, 2, 0)
        out[0, :, 3:9] = columns6(rotation).permute(1, 2, 0)
        outputs.append(out)
        amplitudes.append((rotation - rotation.mean(0)).square().mean(0).sqrt())
    torch.testing.assert_close(amplitudes[0], amplitudes[1], atol=1e-6, rtol=1e-5)
    assert stat(outputs[1], outputs[0], y, config) > 1e-5


def test_world_root_translation_does_not_leak_into_body_statistics():
    config, base, y = fixture(True)
    wave = torch.sin(torch.arange(60) * 2 * torch.pi / 30)
    base[:, 1, 0] = .1 * wave
    moved = base.clone()
    moved[:, 0, 0] += 10 * wave
    moved[:, 0, 2] -= 5 * wave
    torch.testing.assert_close(detail_features(moved, y, config), detail_features(base, y, config),
                               atol=0, rtol=0)


def test_matched_base_training_reduces_unnecessary_correction():
    config, base, y = fixture(True, frames=30)
    net = MotionDetailRefiner(config)
    opt = torch.optim.Adam(net.parameters(), lr=.001)
    with torch.no_grad():
        _, initial = net(base, y, seed=7)
        before = initial.square().mean()
    target = detail_features(base, y, config).detach()
    scale = feature_scale(target)
    for _ in range(80):
        opt.zero_grad(set_to_none=True)
        out, delta = net(base, y, seed=7)
        losses = preservation_losses(out, base, delta, y, config)
        loss = statistics_loss(detail_features(out, y, config), target, scale) + losses['keep'] + .01 * losses['reg']
        loss.backward()
        opt.step()
    with torch.no_grad():
        out, delta = net(base, y, seed=7)
        assert delta.square().mean() < before * .5
        assert stat(out, base, y, config) < .001
        assert torch.equal(out[:, 0], base[:, 0])


@pytest.mark.parametrize('version', [None, 2])
def test_unsupported_checkpoint_is_not_silently_reinterpreted(tmp_path, version):
    config, _, _ = fixture()
    net = MotionDetailRefiner(config)
    architecture = net.architecture()
    if version is None:
        architecture.pop('calibration_version')
    else:
        architecture['calibration_version'] = version
    path = tmp_path / 'legacy.pt'
    torch.save({'schema': DETAIL_SCHEMA, 'contract': {}, 'architecture': architecture,
                'refiner': net.state_dict()}, path)
    with pytest.raises(ValueError, match='Unsupported refiner'):
        load_refiner(path, {}, 'cpu')
    with pytest.raises(ValueError, match='Unsupported refiner'):
        main(['--cache-dir', str(tmp_path / 'unused'), '--save-dir', str(tmp_path / 'new'),
              '--resume', str(path)])


def test_matched_anchor_gradients_match_between_training_paths():
    from train.train_motion_detail import prepare_groups, matched_identity_loss
    from train.detail_runtime import BatchedDetailRuntime
    config, base, y = fixture(True, frames=30)
    group = {'y': y, 'gt': base, 'train_base': base.repeat(3, 1, 1, 1), 'validation_base': base}
    scale = prepare_groups([group], config, batched=True)[0]
    net = MotionDetailRefiner(config)
    serial = matched_identity_loss(net, group, 123, scale)[0]
    serial.backward()
    gradients = [p.grad.clone() for p in net.parameters()]
    net.zero_grad(set_to_none=True)
    batched = matched_identity_loss(net, group, 123, scale, BatchedDetailRuntime(net))[0]
    batched.backward()
    torch.testing.assert_close(serial, batched, atol=1e-6, rtol=1e-4)
    for parameter, expected in zip(net.parameters(), gradients):
        torch.testing.assert_close(parameter.grad, expected, atol=2e-5, rtol=1e-3)


def test_evaluation_accepts_exact_matched_identity_without_forcing_noise(tmp_path):
    import json
    import shutil
    from pathlib import Path
    from tests.test_train_motion_detail import make_training_cache
    from train.train_motion_detail import load_groups, prepare_groups
    from sample.detail_refinement import file_hash
    from eval.eval_motion_detail import main as evaluate
    cache, config = make_training_cache(tmp_path)
    manifest, groups = load_groups(cache, 'cpu')
    scales = prepare_groups(groups, config)
    net = MotionDetailRefiner(config)
    torch.nn.init.zeros_(net.output.weight)
    torch.nn.init.zeros_(net.output.bias)
    save = tmp_path / 'model'
    save.mkdir()
    checkpoint = save / 'refiner.pt'
    torch.save({'schema': DETAIL_SCHEMA, 'contract': manifest['contract'],
                'architecture': net.architecture(), 'refiner': net.state_dict(),
                'cache_manifest_sha256': file_hash(cache / 'manifest.json'),
                'normalization': scales}, checkpoint)
    thresholds = Path(__file__).resolve().parents[1] / 'docs/motion_calibration_acceptance.json'
    shutil.copyfile(thresholds, save / 'acceptance.json')
    output = tmp_path / 'evaluation.json'
    evaluate(['--cache-dir', str(cache), '--refiner', str(checkpoint),
              '--output', str(output), '--device', 'cpu'])
    report = json.loads(output.read_text())
    assert report['passed']
    assert all(r['matched_base'] and r['checks']['matched_train_inputs_preserved'] for r in report['groups'])
    assert all(r['detail_seed_rotation_rms'] == 0 for r in report['groups'])
