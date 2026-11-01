"""Cached objectives keep gradients; real training emits usable health events."""
import copy
import json
import math
import shutil

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from model.motion_detail_refiner import MotionDetailRefiner
from sample.detail_refinement import DETAIL_SCHEMA, file_hash, tree_to
from tests.test_motion_detail_refiner import fixture
from train.detail_objectives import detail_features, feature_scale
from train.train_motion_detail import group_loss, main, prepare_groups, training_context
from train.detail_runtime import BatchedDetailRuntime
from train.detail_gradients import DetailGradientMonitor
from model.motion_detail_refiner import fk
from train.detail_checkpoint import learning_rate_factor


def make_training_cache(tmp_path):
    config, base, y = fixture()
    cache = tmp_path / 'cache'
    cache.mkdir()
    records = []
    for i in range(2):
        path = cache / f'{i}.pt'
        torch.save({'key': json.dumps({'condition': i}), 'y': y, 'gt': base,
                    'train_base': base, 'validation_base': base, 'clips': [f'clip{i}']}, path)
        records.append({'file': path.name, 'sha256': file_hash(path)})
    (cache / 'manifest.json').write_text(json.dumps({
        'schema': DETAIL_SCHEMA, 'target_split': 'train', 'frozen_state_verified': True,
        'train_seeds': [1], 'validation_seeds': [2], 'groups': records,
        'contract': {}, 'reference_fps': 30.}))
    return cache, config


def test_cosine_schedule_warmup_endpoint_and_short_budget():
    assert learning_rate_factor(1, 6200, 124, 'constant') == 1
    assert learning_rate_factor(1, 6200, 124, 'cosine') == pytest.approx(1/248)
    assert learning_rate_factor(248, 6200, 124, 'cosine') == 1
    assert learning_rate_factor(6200, 6200, 124, 'cosine') == pytest.approx(.01)
    assert learning_rate_factor(1, 1, 1, 'cosine') == pytest.approx(.01)


@pytest.mark.parametrize('loop,frames', [(False, 37), (True, 60), (False, 1)])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_batched_group_matches_serial_output_loss_and_gradients(loop, frames, device):
    config, base, y = fixture(loop, frames)
    # Non-unit base rotations, padding and distinct backbone samples matter here.
    base = torch.nn.functional.pad(base, (0, 3, 0, 0, 0, 2))
    base = base.repeat(3, 1, 1, 1)
    base[1, 1, 3, :frames] += .03
    base[2, 2, 0, :frames] += .04
    group = tree_to({'y': y, 'gt': base[:1], 'train_base': base,
                     'validation_base': base[:1]}, device)
    # Same host conversion performed by the actual cache loader.
    group['y'] = training_context(group['y'], 1)
    scales = prepare_groups([group], config, batched=True)
    net = MotionDetailRefiner(config).to(device)
    expected, output, delta = group_loss(net, group, 'train_base', 99, scales[0])
    sum(expected.values()).backward()
    gradients = [p.grad.clone() for p in net.parameters()]
    net.zero_grad(set_to_none=True)
    actual, batched_output, batched_delta = group_loss(
        net, group, 'train_base', 99, scales[0], runtime=BatchedDetailRuntime(net))
    torch.testing.assert_close(batched_output, output, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(batched_delta, delta, atol=2e-6, rtol=2e-5)
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], atol=2e-6, rtol=2e-5)
    sum(actual.values()).backward()
    for p, grad in zip(net.parameters(), gradients):
        torch.testing.assert_close(p.grad, grad, atol=2e-5, rtol=1e-3)


@pytest.mark.parametrize('loop', [False, True])
def test_prepared_group_preserves_losses_and_parameter_gradients(loop):
    config, base, y = fixture(loop)
    group = {'y': y, 'gt': base.clone(), 'train_base': base.repeat(2, 1, 1, 1),
             'validation_base': base.clone()}
    target = detail_features(base, y, config).detach()
    group['target_features'] = target
    scale = feature_scale(target)
    net = MotionDetailRefiner(config)
    losses, expected, _ = group_loss(net, group, 'train_base', 10, scale)
    sum(losses.values()).backward()
    gradients = [p.grad.clone() for p in net.parameters()]
    expected_losses = {k: v.detach() for k, v in losses.items()}
    prepared = copy.deepcopy(group)
    scales = prepare_groups([prepared], config)
    assert 'gt' not in prepared
    assert all(not v.requires_grad for row in prepared['_references']['train_base'] for v in row.values())
    # Reuse twice: cached tensors must never keep a spent autograd graph.
    for _ in range(2):
        net.zero_grad(set_to_none=True)
        actual, output, _ = group_loss(net, prepared, 'train_base', 10, scales[0])
        torch.testing.assert_close(output, expected)
        for key, value in actual.items():
            torch.testing.assert_close(value, expected_losses[key])
        sum(actual.values()).backward()
        for p, grad in zip(net.parameters(), gradients):
            torch.testing.assert_close(p.grad, grad)


@pytest.mark.parametrize('parents', [[-1, 3, 0, 2], [-1, 3, -1, 2]])
def test_depth_batched_fk_handles_out_of_order_parents_and_multiple_roots(parents):
    config, base, y = fixture(frames=7)
    y['parents'] = [parents]
    group = {'y': y, 'gt': base, 'train_base': base.repeat(2, 1, 1, 1), 'validation_base': base}
    prepare_groups([group], config, batched=True)
    batch = group['_batch']
    local = torch.randn(2, 7, 4, 3, 3, requires_grad=True)
    g, p = batch.fk(local)
    expected_g, expected_p = fk(local.reshape(14, 4, 3, 3), batch.rest, parents)
    torch.testing.assert_close(g, expected_g.reshape_as(g))
    torch.testing.assert_close(p, expected_p.reshape_as(p))
    torch.testing.assert_close(batch.global_rotations(local), g)
    actual_grad, = torch.autograd.grad(g.square().mean() + p.square().mean(), local, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected_g.square().mean() + expected_p.square().mean(), local)
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_control_metadata_stays_on_host():
    _, _, y = fixture()
    y['parents'] = torch.tensor(y['parents'])
    context = training_context(tree_to(y, 'cuda'), 3)
    assert context['n_joints'] == [4] * 3
    assert context['parents'] == [[-1, 0, 1, 0]] * 3
    assert context['detail_action'].is_cuda


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_training_checkpoint_validation_intervals_and_tensorboard(tmp_path, capsys, device):
    cache, config = make_training_cache(tmp_path)
    save = tmp_path / 'train'
    main(['--cache-dir', str(cache), '--save-dir', str(save), '--device', device,
          '--width', str(config.width), '--layers', str(config.layers),
          '--updates-per-group', '3', '--save-every', '2', '--eval-every', '4',
          '--log-every', '2', '--no-progress'])
    assert not (save / 'validation000000002.json').exists()
    assert (save / 'validation000000004.json').exists()
    assert (save / 'validation000000006.json').exists()
    assert file_hash(save / 'refiner_latest.pt') == file_hash(save / 'refiner000000006.pt')
    checkpoint = torch.load(save / 'refiner_latest.pt', weights_only=False, map_location='cpu')
    assert checkpoint['group_updates'] == [3, 3]
    assert len((save / 'training.jsonl').read_text().splitlines()) == 6
    events = EventAccumulator(str(save / 'tensorboard')).Reload()
    expected_tags = {'train/loss', 'validation/loss', 'train/lr', 'train/grad_norm',
                     'train/grad_clip_scale', 'train/epoch_clip_fraction'} | {
        f'train/{k}' for k in DetailGradientMonitor.metric_keys} | {
        f'train/epoch_{k}' for k in ('loss', 'stat', 'identity', 'keep', 'geo', 'reg',
                                   'grad_norm_before_clip')} | {
        f'train/{k}' for k in ('stat', 'identity', 'keep', 'geo', 'reg')}
    assert expected_tags <= set(events.Tags()['scalars'])
    records = [json.loads(line) for line in (save / 'training.jsonl').read_text().splitlines()]
    for record in records:
        assert all(math.isfinite(record[k]) for k in DetailGradientMonitor.metric_keys)
    logged = {r['step']: r for r in records}
    for event in events.Scalars('train/lr'):
        assert event.value == pytest.approx(logged[event.step]['lr'])
    for event in events.Scalars('train/grad_norm'):
        assert event.value == pytest.approx(logged[event.step]['grad_norm_before_clip'])
    for tag in ('train/loss', 'validation/loss'):
        assert events.Scalars(tag), tag
        assert all(math.isfinite(e.value) for e in events.Scalars(tag))
    assert [e.step for e in events.Scalars('validation/loss')] == [4, 6]
    rows = json.loads((save / 'validation000000006.json').read_text())
    assert events.Scalars('validation/matched_stat_max')[-1].value == pytest.approx(
        max(r['matched_stat'] for r in rows))
    assert events.Scalars('validation/bone_length_rms_relative_max')[-1].value == pytest.approx(
        max(r['bone_length_rms_relative'] for r in rows))
    assert all(r['root_max_error'] == 0 for r in rows)
    assert max(r['max_residual_rad'] for r in rows) <= .080001
    output = capsys.readouterr().out
    assert '6/6 (100.0%)' in output and 'ETA=0.0s' in output
    assert 'validation complete' in output


def test_zero_eval_interval_skips_even_final_validation(tmp_path):
    cache, config = make_training_cache(tmp_path)
    save = tmp_path / 'train'
    main(['--cache-dir', str(cache), '--save-dir', str(save), '--device', 'cpu',
          '--width', str(config.width), '--layers', str(config.layers),
          '--updates-per-group', '2', '--eval-every', '0', '--no-progress'])
    assert (save / 'refiner_latest.pt').exists()
    assert not list(save.glob('validation*.json'))
    events = EventAccumulator(str(save / 'tensorboard')).Reload()
    assert 'validation/loss' not in events.Tags()['scalars']
    assert [e.step for e in events.Scalars('train/epoch_loss')] == [2, 4]


def test_disabled_validation_does_not_prepare_held_out_statistics():
    config, base, y = fixture()
    # A validation tensor must not even enter feature/reference computation.
    group = {'y': y, 'gt': base, 'train_base': base, 'validation_base': None}
    prepare_groups([group], config, batched=True, prepare_validation=False)
    assert '_base_stat' not in group
    assert 'validation_base' not in group['_contexts']
    assert 'validation_base' not in group['_references']


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_resume_mid_epoch_matches_uninterrupted_and_can_extend_budget(tmp_path, legacy, device):
    cache, config = make_training_cache(tmp_path)
    full = tmp_path / 'full'
    main(['--cache-dir', str(cache), '--save-dir', str(full), '--device', device,
          '--width', str(config.width), '--layers', str(config.layers), '--updates-per-group', '3',
          '--save-every', '3', '--eval-every', '4', '--log-every', '1', '--no-progress',
          '--lr-schedule', 'cosine'])
    expected = torch.load(full / 'refiner_latest.pt', weights_only=False)
    midpoint = torch.load(full / 'refiner000000003.pt', weights_only=False)
    if legacy:
        midpoint.pop('sampler_state')
        midpoint.pop('rng_state')
        midpoint.pop('lr_schedule_total_steps')
    save = tmp_path / 'resumed'
    save.mkdir()
    torch.save(midpoint, save / 'refiner_latest.pt')
    shutil.copyfile(full / 'acceptance.json', save / 'acceptance.json')
    # Recovery discards uncommitted updates and a torn final JSON record.
    (save / 'training.jsonl').write_text((full / 'training.jsonl').read_text() + '{"step":')
    cli = ['--cache-dir', str(cache), '--save-dir', str(save)]  # Resume is the default when a checkpoint exists.
    main(cli)
    actual = torch.load(save / 'refiner_latest.pt', weights_only=False)
    assert actual['step'] == 6 and actual['group_updates'] == [3, 3]
    # CUDA scatter/reduction kernels may accumulate in a different order after
    # recapture. Keep exact CPU resume checks and tight numerical CUDA checks.
    tolerance = dict(rtol=1e-5, atol=1e-7) if device == 'cuda' else dict(rtol=0, atol=0)
    for key in expected['refiner']:
        torch.testing.assert_close(actual['refiner'][key], expected['refiner'][key], **tolerance)
    for key in expected['optimizer']['state']:
        for field, value in expected['optimizer']['state'][key].items():
            torch.testing.assert_close(actual['optimizer']['state'][key][field], value, **tolerance)
    expected_rows = [json.loads(line) for line in (full / 'training.jsonl').read_text().splitlines()]
    actual_rows = [json.loads(line) for line in (save / 'training.jsonl').read_text().splitlines()]
    assert len(actual_rows) == 6
    assert [(r['group'], r['detail_seed']) for r in actual_rows] == [
        (r['group'], r['detail_seed']) for r in expected_rows]
    checkpoint_hash = file_hash(save / 'refiner_latest.pt')
    main(cli + ['--resume'])  # Explicit resume on a completed budget must not restart or overwrite the checkpoint.
    assert file_hash(save / 'refiner_latest.pt') == checkpoint_hash
    main(cli + ['--updates-per-group', '4'])
    extended = torch.load(save / 'refiner_latest.pt', weights_only=False)
    assert extended['step'] == 8 and extended['group_updates'] == [4, 4]
    assert extended['lr_schedule_total_steps'] == 6
    extended_rows = [json.loads(line) for line in (save / 'training.jsonl').read_text().splitlines()]
    assert all(r['lr'] == pytest.approx(expected_rows[-1]['lr']) for r in extended_rows[-2:])
    with pytest.raises(ValueError, match='budget'):
        main(cli + ['--updates-per-group', '3'])
    with pytest.raises(ValueError, match='architecture mismatch'):
        main(cli + ['--width', '17'])
    with pytest.raises(ValueError, match='empty'):
        main(cli + ['--no-resume'])
    fresh = tmp_path / 'fresh'
    main(cli + ['--no-resume', '--updates-per-group', '3', '--save-dir', str(fresh)])
    assert torch.load(fresh / 'refiner_latest.pt', weights_only=False)['step'] == 6


