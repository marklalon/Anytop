"""Independent skeleton views must keep one time map and clean decoder/loss conditions."""
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from .test_bone_length_aug import make_dataset
from .test_joint_buckets import _item
from .test_motion_autoencoder import LoaderFixture, fixture
from data_loaders.tensors import truebones_batch_collate
from data_loaders.truebones.truebones_utils.canonical_features import canonical_to_physical_hml
from train.motion_reconstruction_losses import MotionReconstructionLosses
from train.train_motion_autoencoder import (
    AutoencoderBatchTransform, create_training_data_loader, dual_view_training_terms,
    evaluate_dual_view_validation, parser, training_batches, training_batch_to_device,
)


@pytest.mark.parametrize('is_loop,high_frequency', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('frames', [12, 45])
def test_independent_geometry_reuses_every_temporal_draw(is_loop, high_frequency, frames, monkeypatch):
    import data_loaders.truebones.data.dataset as module
    dataset, data = make_dataset()
    data['motion_metadata']['is_loop'] = is_loop
    data['high_frequency'] = high_frequency
    dataset.opt.autoencoder_dual_view = True
    dataset.opt.bone_length_aug_max = .2
    dataset.opt.motion_speed_aug = 1.3
    dataset.opt.loop_tile_single_prob = .5
    dataset.opt.loop_cond_prob = .5
    stored = dataset._load_physical_motion(data)[0]
    stored[:, 0, 0] = np.sin(np.linspace(0, 2 * np.pi, len(stored)))
    original = stored.copy()
    calls = []
    for name in ('_sample_motion_speed_target_length', '_sample_loop_offset', '_sample_loop_tile_count'):
        method = getattr(dataset, name)
        def spy(*args, _method=method, _name=name, **kwargs):
            calls.append(_name)
            return _method(*args, **kwargs)
        monkeypatch.setattr(dataset, name, spy)
    closing_calls = []
    closing = module._drop_loop_closing_frame
    def close(motion):
        closing_calls.append(True)
        return closing(motion)
    monkeypatch.setattr(module, '_drop_loop_closing_frame', close)
    random.seed(7)
    first = dataset._prepare_sample('test', data, target_num_frames=frames)
    second = first[12]['autoencoder_view']
    assert second is not None and second[12]['autoencoder_view'] is None
    assert calls.count('_sample_motion_speed_target_length') == 1
    assert all(calls.count(name) <= 1 for name in set(calls))
    assert len(closing_calls) == int(is_loop)
    for key in ('is_loop', 'resample_speed_cond', 'loop_uncond', 'loop_phase_offset',
                'loop_tile_count', 'motion_speed_applied'):
        assert first[10][key] == second[10][key]
    assert not np.allclose(first[4], second[4])
    a = canonical_to_physical_hml(first[0], first[12])
    b = canonical_to_physical_hml(second[0], second[12])
    np.testing.assert_allclose(a[:, 0, [0, 2]], b[:, 0, [0, 2]], atol=1e-6)
    np.testing.assert_allclose(a[..., 3:9], b[..., 3:9], atol=1e-6)
    np.testing.assert_array_equal(stored, original)


def test_independent_topology_has_complete_per_view_canonical_context(monkeypatch):
    import data_loaders.truebones.data.dataset as module
    from data_loaders.truebones.truebones_utils.leaf_drop import drop_joints_from_cond
    dataset, data = make_dataset()
    dataset.opt.autoencoder_dual_view = True
    dataset.opt.topology_aug_prob = 1.
    calls = []
    def augment(motion, cond, root, **kwargs):
        dropped = [3, 6] if not calls else []
        calls.append(dropped)
        if not dropped:
            return motion, cond, dict(applied=False)
        keep, changed = drop_joints_from_cond(cond, dropped)
        return motion[:, keep], changed, dict(applied=True, translation_root_index=0,
                                             new_index=keep)
    monkeypatch.setattr(module, 'augment_topology', augment)
    random.seed(10)
    sample = dataset._prepare_sample('test', data)
    motion, wrapped = truebones_batch_collate([sample], joint_buckets=(5, 7))
    cond = wrapped['y']
    other, other_cond = cond['ae_view_motion'], cond['ae_view_cond']
    assert calls == [[3, 6], []]
    assert motion.shape == other.shape == (1, 7, 12, 30)
    assert cond['n_joints'].item() == 5 and other_cond['n_joints'].item() == 7
    for x, y in ((motion, cond), (other, other_cond)):
        decoded = canonical_to_physical_hml(x, y)
        assert torch.isfinite(decoded).all()
        assert y['joint_struct'].shape[1] == 7
        assert y['rest_geom'][0, :y['n_joints'].item(), -1].eq(1).all()
    assert not torch.equal(cond['rest_pos_ric_hml'], other_cond['rest_pos_ric_hml'])


def view_batch(variational=False):
    model, x, skeleton = fixture(variational=variational, temporal_rope=True,
                                mirror_twin_drop_prob=.5, lateral_drop_prob=.5,
                                decoder_mirror_twin_drop_prob=0., decoder_lateral_drop_prob=0.)
    _, other, other_skeleton = fixture(joints=3)
    def conditions(s):
        return dict(s, lengths=torch.full((2,), 5), is_loop=torch.tensor([False, True]),
                    resample_speed_cond=torch.tensor([.5, 1.5]), loop_tile_count=torch.ones(2))
    clean = conditions(skeleton)
    clean.update(ae_view_motion=other, ae_view_cond=conditions(other_skeleton))
    batch = AutoencoderBatchTransform(1., 1., 1.)((x, {'y': clean}))
    return model, clean, batch


def test_drop_only_corrupts_sources_and_keeps_original_and_loss_conditions():
    _, original, batch = view_batch()
    _, cond = training_batch_to_device(batch, 'cpu')
    for clean, saved in ((cond, original), (cond['ae_view_cond'], original['ae_view_cond'])):
        for key in ('rest_geom', 'joints_names_embs'):
            assert clean['ae_source_skeleton'][key].count_nonzero() == 0
            torch.testing.assert_close(clean[key], saved[key], rtol=0, atol=0)
        assert clean['ae_source_skeleton']['graph_dist'] is clean['graph_dist']
    assert 'ae_source_skeleton' not in original
    assert 'ae_source_skeleton' not in original['ae_view_cond']


@pytest.fixture
def fresh_compiler_cache():
    # Different model/forward signatures must not exhaust the process-wide
    # Dynamo specialization budget for unrelated compile contract tests.
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.mark.parametrize('variational', [False, True])
def test_cross_self_loss_routes_both_targets_and_backpropagates_fullgraph(variational, fresh_compiler_cache):
    model, _, batch = view_batch(variational)
    model.train()
    assert model.encoder.mirror_twin_drop_prob == model.encoder.lateral_drop_prob == .5
    assert model.decoder.mirror_twin_drop_prob == model.decoder.lateral_drop_prob == 0
    compiled = torch.compile(model, backend='eager', fullgraph=True)
    target, cond = training_batch_to_device(batch, 'cpu')
    calls = []
    criterion = MotionReconstructionLosses()
    def loss(prediction, target, y, **kwargs):
        result = criterion(prediction, target, y, **kwargs)
        calls.append((target, y, result['loss']))
        return result
    terms = dual_view_training_terms(compiled, loss, target, cond, self_weight=.5, kl_weight=.01)
    other, other_cond = cond['ae_view_motion'], cond['ae_view_cond']
    for call, expected_target, expected_cond in zip(calls, (other, target, target, other),
                                                  (other_cond, cond, cond, other_cond)):
        assert call[0] is expected_target and call[1] is expected_cond
    expected = (calls[0][2] + calls[2][2] + .5 * (calls[1][2] + calls[3][2])) / 3
    if variational:
        expected = expected + .01 * terms['kl_loss']
    torch.testing.assert_close(terms['loss'], expected)
    terms['loss'].backward()
    for layer in (model.motion_projection, model.to_latent, model.output):
        assert torch.isfinite(layer.weight.grad).all() and layer.weight.grad.abs().sum() > 0
    if variational:
        assert model.to_logvar.weight.grad.abs().sum() > 0


class DualLoaderFixture(LoaderFixture):
    def __getitem__(self, index):
        item = _item(self.counts[index], 100, seed=index)
        item[10].update(is_loop=False, loop_tile_count=1, resample_speed_cond=1.)
        if getattr(self.opt, 'autoencoder_dual_view', False):
            other = _item(max(2, self.counts[index] - 2), 100, seed=index + 100)
            other[10].update(item[10])
            item[12]['autoencoder_view'] = other
        return item


@pytest.mark.parametrize('workers', [0, 2])
def test_dual_view_prefetch_and_spawn_workers(workers, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    args = parser().parse_args(['--save_dir', 'unused', '--dual_view', '--batch_size', '2',
                               '--num_workers', str(workers), '--main_process_prefetch_batches', '2',
                               '--joint_buckets', '8', '16', '100', '--joint_name_drop_prob', '1'])
    loader = create_training_data_loader(args, DualLoaderFixture())
    batches = training_batches(loader)
    for _ in range(3):
        x, cond = training_batch_to_device(next(batches), 'cpu')
        assert x.shape == cond['ae_view_motion'].shape
        assert cond['joints_names_embs'].count_nonzero() > 0
        assert cond['ae_source_skeleton']['joints_names_embs'].count_nonzero() == 0
        assert (cond['n_joints'] > cond['ae_view_cond']['n_joints']).all()


def test_held_out_dual_view_metrics_replay_without_mutating_loader_or_rng():
    from model.motion_autoencoder import MotionAutoencoder
    source, data = make_dataset()
    source.data_dict = {'test': data}
    source.name_list, source.pointer = ['test'], 0
    source.opt.bone_length_aug_max = .2
    loader = SimpleNamespace(dataset=SimpleNamespace(motion_dataset=source))
    args = parser().parse_args(['--save_dir', 'unused', '--dual_view', '--bone_length_aug_prob', '1'])
    model = MotionAutoencoder(t5_out_dim=4, width=16, layers=1, heads=2, slots=2, latent_dim=4)
    random.seed(100)
    expected_rng = random.getstate()
    first = evaluate_dual_view_validation(model, loader, args, 'cpu')
    assert random.getstate() == expected_rng
    again = evaluate_dual_view_validation(model, loader, args, 'cpu')
    assert first == again
    assert first['dual_view_sample_count'] == 1
    assert np.isfinite(first['dual_view_cross_mse']) and first['dual_view_self_mse'] > 0
    assert not getattr(source.opt, 'autoencoder_dual_view', False)
