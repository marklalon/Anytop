"""Seed batching and restart safety for frozen-backbone caches."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from diffusion.gaussian_diffusion import GaussianDiffusion, LossType, ModelMeanType, ModelVarType
from sample.detail_refinement import file_hash
from sample.sampling import _sample_batch
from tools.cache_motion_detail import restore_progress, write_manifest
from train.train_motion_detail import load_groups


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
@pytest.mark.parametrize('method,eta', [('ddpm', 0.), ('ddim', 0.), ('ddim', .7)])
def test_seed_batch_matches_legacy_and_chunking(device, method, eta):
    device = torch.device(device)
    diffusion = GaussianDiffusion(
        betas=np.linspace(.001, .02, 6), model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL, loss_type=LossType.MSE)
    # Depends on every step's input noise and per-row condition, without batch reductions.
    def model(x, t, y):
        return x * .6 + y['offset'][:, None, None, None]

    seeds = [101, 102, 10001]
    offsets = torch.tensor([.1, -.2, .3], device=device)

    def run(indices, independent):
        return _sample_batch(
            diffusion, model, {'y': {'offset': offsets[indices]}}, method,
            (len(indices), 3, 12, 9), eta, seeds[indices[0]], device,
            row_seeds=[seeds[i] for i in indices] if independent else None, progress=False)

    legacy = torch.cat([run([i], False) for i in range(3)])
    batched = run([0, 1, 2], True)
    chunked = torch.cat([run([0, 1], True), run([2], True)])
    torch.testing.assert_close(batched, legacy, rtol=0, atol=0)
    torch.testing.assert_close(batched, chunked, rtol=0, atol=0)


def test_resume_validates_configuration_and_completed_files(tmp_path):
    (tmp_path / 'group.pt').write_bytes(b'complete group')
    expected = {'schema': 1, 'contract': {'steps': 100}, 'source_plan_sha256': 'source',
                'groups': [], 'frozen_state_verified': False}
    saved = copy.deepcopy(expected)
    saved['groups'] = [{'file': 'group.pt', 'sha256': file_hash(tmp_path / 'group.pt')}]
    write_manifest(tmp_path, saved)
    # A cancelled temporary write is never treated as a completed group.
    (tmp_path / 'next.pt.tmp').write_bytes(b'incomplete')
    assert restore_progress(tmp_path, expected) == saved
    changed = copy.deepcopy(expected)
    changed['contract']['steps'] = 10
    with pytest.raises(ValueError, match='mismatch: contract'):
        restore_progress(tmp_path, changed)
    changed = copy.deepcopy(expected)
    changed['source_plan_sha256'] = 'different GT windows'
    with pytest.raises(ValueError, match='mismatch: source_plan'):
        restore_progress(tmp_path, changed)
    (tmp_path / 'group.pt').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='corrupt'):
        restore_progress(tmp_path, expected)


def test_missing_manifest_and_incomplete_training_cache(tmp_path):
    with pytest.raises(ValueError, match='no manifest'):
        restore_progress(tmp_path, {})
    write_manifest(tmp_path, {'schema': 1, 'target_split': 'train', 'groups': [],
                              'frozen_state_verified': False})
    with pytest.raises(ValueError, match='incomplete'):
        load_groups(tmp_path, 'cpu')


def test_cancel_then_resume_skips_completed_group(tmp_path, monkeypatch, capsys):
    from tools import cache_motion_detail as cache

    model = torch.nn.Linear(1, 1)
    model.action_word_embeddings = torch.zeros(1, 2)
    model.action_conditioning_metadata = {'embedding_contract': {}, 'conditioning_contract': {}}
    backbone = tmp_path / 'model.pt'
    (tmp_path / 'args.json').write_text(json.dumps({'num_frames': 4, 'objects_subset': []}))
    args = SimpleNamespace(model_path=str(backbone), reference_motion='', action_label='',
                           action_group='test', ddim_eta=0.)
    monkeypatch.setattr(cache, 'generate_args', lambda cli: args)
    monkeypatch.setattr(cache, 'ActionConditioningBundle', lambda **kwargs: None)
    monkeypatch.setattr(cache, 'prepare_generation_runtime', lambda args: SimpleNamespace(
        model=model, actual_cond_file='cond.npy', species_table=None, opt=SimpleNamespace(fps=30)))

    class Dataset:
        motion_dataset = SimpleNamespace(name_list=['clip0', 'clip1'], pointer=0)

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return [np.zeros((4, 2, 12))] + [index] * 9

    monkeypatch.setattr(cache, 'get_dataset', lambda **kwargs: Dataset())

    def collate(samples):
        index = samples[0][1]
        return torch.zeros(1, 2, 12, 4), {'y': {
            'object_type': ['rig'], 'action_label': [str(index)], 'is_loop': [False],
            'loop_tile_count': [1], 'resample_speed_cond': [1.]}}

    monkeypatch.setattr(cache, 'truebones_batch_collate', collate)
    monkeypatch.setattr(cache, 'backbone_contract', lambda *args: {'test': True})
    monkeypatch.setattr(cache, '_wrap_action_label_cfg', lambda model, *args: model)
    monkeypatch.setattr(cache, 'attach_detail_action', lambda model, y: y)
    monkeypatch.setattr(cache, 'decode_batch', lambda x, y: x)
    calls = []

    def generate(runtime, sampler, y, shape, seeds, *args):
        label = y['action_label'][0]
        calls.append(label)
        if label == '1' and len(calls) == 2:
            raise KeyboardInterrupt
        return torch.zeros(len(seeds), *shape[1:])

    monkeypatch.setattr(cache, 'generate_bases', generate)
    folder = tmp_path / 'cache'
    cli = ['--cache-dir', str(folder)]
    with pytest.raises(KeyboardInterrupt):
        cache.main(cli)
    partial = json.loads((folder / 'manifest.json').read_text())
    assert len(partial['groups']) == 1
    assert partial['frozen_state_verified'] is False
    first_file = folder / partial['groups'][0]['file']
    first_mtime = first_file.stat().st_mtime_ns
    cache.main(cli + ['--resume', '--seed-batch-size', '2'])
    assert calls == ['0', '1', '1']
    assert first_file.stat().st_mtime_ns == first_mtime
    complete = json.loads((folder / 'manifest.json').read_text())
    assert len(complete['groups']) == complete['expected_groups'] == 2
    assert complete['frozen_state_verified'] is True
    output = capsys.readouterr().out
    assert '1/2 (50.0%)' in output
    assert '2/2 (100.0%)' in output
    assert 'elapsed=' in output
    assert 'ETA=estimating...' in output
    assert 'ETA=00:00:00' in output
    cache.main(cli + ['--resume'])
    assert calls == ['0', '1', '1']
    output = capsys.readouterr().out
    assert '2/2 (100.0%)' in output
    assert 'ETA=00:00:00' in output
