"""Validation uses the held-out split and skips empty splits without fallback."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import autoencoder_validation as validation
from train import train_autoencoder as vae
from data_loaders.truebones.data import dataset as dataset_module
from data_loaders.reference import REST_GEOM_DIM
from data_loaders.truebones.truebones_utils.joint_struct_features import JOINT_STRUCT_DIM


def validation_args():
    return SimpleNamespace(cond_path='cond.npy', batch_size=16, num_frames=60,
                           sample_filter='training_only.json', seed=10, motion_cache_size=0,
                           action_label_cond=True, action_conditioning=object(),
                           species_descriptor_table=object())


def test_validation_uses_independent_split_and_no_training_augmentation(monkeypatch):
    captured = {}
    loader = SimpleNamespace(dataset=['val_clip'])

    def build(**kwargs):
        captured.update(kwargs)
        return loader

    monkeypatch.setattr(validation, 'get_dataset_loader', build)
    args = validation_args()
    transform = object()
    assert validation.build_validation_loader(args, transform) is loader
    assert captured['split'] == 'val'
    assert captured['sample_filter'] is None
    assert captured['shuffle'] is captured['drop_last'] is False
    assert captured['joint_buckets'] is None
    assert captured['motion_speed_aug'] == captured['loop_cond_prob'] == 1.0
    assert captured['loop_tile_single_prob'] == 0.0
    for key in ('topology_aug_prob', 'bone_length_aug_prob', 'rest_repose_prob',
                'rest_variant_prob', 'joint_name_drop_prob'):
        assert captured[key] == 0.0
    assert captured['action_conditioning'] is args.action_conditioning
    assert captured['species_table'] is args.species_descriptor_table
    assert captured['batch_transform'] is transform
    assert not captured.get('ref_encoder', False)


@pytest.mark.parametrize('raise_empty', [True, False])
def test_empty_val_disables_validation_without_train_fallback(monkeypatch, raise_empty):
    calls = []

    def build(**kwargs):
        calls.append(kwargs['split'])
        if raise_empty:
            raise validation.EmptySplitError('No validation sources')
        return SimpleNamespace(dataset=[])

    monkeypatch.setattr(validation, 'get_dataset_loader', build)
    assert validation.build_validation_loader(validation_args()) is None
    assert calls == ['val']


def test_invalid_validation_data_is_not_silently_skipped(monkeypatch):
    def build(**kwargs):
        raise ValueError('Malformed skeleton')

    monkeypatch.setattr(validation, 'get_dataset_loader', build)
    with pytest.raises(ValueError, match='Malformed skeleton'):
        validation.build_validation_loader(validation_args())


def test_val_with_only_short_clips_is_an_empty_split(monkeypatch, tmp_path):
    import numpy as np

    name = 'Test_Idle.npy'
    np.save(tmp_path / name, np.zeros((4, 1, 12), dtype=np.float32))
    source = SimpleNamespace(namespace='val', root=str(tmp_path), motion_dir=str(tmp_path), sample_weight=1.0)
    opt = SimpleNamespace(sources=[source], min_length=20, cond_file='unused', motion_cache_size=0)
    monkeypatch.setattr(dataset_module, 'rest_geometry', lambda entry: None)
    monkeypatch.setattr(dataset_module, 'build_joint_struct_features', lambda *a, **kw: None)
    monkeypatch.setattr(dataset_module, '_build_limb_chains', lambda entry: (None, None, None))
    monkeypatch.setattr(dataset_module, 'high_frequency_clips', lambda *args: set())
    monkeypatch.setattr(dataset_module, '_require_motion_metadata_entry', lambda *args: {})
    with pytest.raises(validation.EmptySplitError, match='min_length=20'):
        dataset_module.MotionDataset(
            opt, {'val/Test': {'dataset_namespace': 'val', 'species_name': 'Test'}}, 60,
            motion_metadata_lookup={'val': {name: {}}}, split='val')


@pytest.mark.parametrize('variational', [False, True])
def test_vae_trains_and_resumes_without_val_or_export_and_uses_first_batch(monkeypatch, tmp_path, variational):
    cond_path = tmp_path / 'cond.npy'
    cond_path.write_bytes(b'test skeleton identity')
    save = tmp_path / 'save'
    conditions = dict(
        n_joints=torch.tensor([3, 3]), lengths=torch.tensor([4, 4]),
        joints_names_embs=torch.zeros(2, 3, 6), rest_geom=torch.zeros(2, 3, REST_GEOM_DIM),
        joint_struct=torch.zeros(2, 3, JOINT_STRUCT_DIM), graph_dist=torch.zeros(2, 3, 3, dtype=torch.long),
        joints_relations=torch.zeros(2, 3, 3, dtype=torch.long), is_loop=torch.tensor([False, False]),
        loop_tile_count=torch.ones(2, dtype=torch.long), resample_speed_cond=torch.ones(2))
    first = torch.full((2, 3, 9, 4), .25)
    second = torch.full_like(first, .75)
    conditions = dict(conditions, ae_view_motion=second, ae_view_cond=dict(conditions))
    dataset = SimpleNamespace(motion_dataset=SimpleNamespace(
        name_list=['train/a', 'train/b'], reference_groups={}, opt=SimpleNamespace(max_joints=128)))
    monkeypatch.setattr(vae, 'get_dataset', lambda **kwargs: dataset)
    monkeypatch.setattr(vae, 'create_training_data_loader', lambda *args: [
        vae.AutoencoderBatchTransform()((first, {'y': conditions})),
        vae.AutoencoderBatchTransform()((second, {'y': conditions}))])
    monkeypatch.setattr(vae, 'build_validation_loader', lambda args: None)
    seen = []

    class ReconstructionLoss:
        def __init__(self, **kwargs):
            self.weights = {}

        def __call__(self, prediction, target, cond, **kwargs):
            seen.append(float(target.mean()))
            return {'loss': (prediction - target).square().mean()}

    def unexpected(*args):
        pytest.fail('An empty val split must not evaluate a training batch')

    monkeypatch.setattr(vae, 'MotionReconstructionLosses', ReconstructionLoss)
    monkeypatch.setattr(vae, 'evaluate', unexpected)
    argv = ['train_autoencoder.py', '--cond_path', str(cond_path), '--save_dir', str(save),
            '--device', 'cpu', '--width', '16', '--layers', '1', '--heads', '2',
            '--slots', '2', '--latent_dim', '4', '--num_frames', '4', '--batch_size', '2', '--num_steps', '1']
    if variational:
        argv[1:1] = ['--variational', '--kl_weight', '0.01']
    monkeypatch.setattr(sys, 'argv', argv)
    vae.main()
    checkpoint = torch.load(save / 'autoencoder.pt', weights_only=False)
    assert checkpoint['step'] == 1 and checkpoint['eval_split'] == 'val'
    assert 'fixed_batch' not in checkpoint
    expected_target = .75
    assert seen == [expected_target]
    assert checkpoint['dual_view_objective'] == vae.DUAL_VIEW_OBJECTIVE
    assert checkpoint['model_config']['layerwise_latent'] is True
    assert all(checkpoint['args'][key] is True
               for key in ('dual_view', 'reference_pairs', 'layerwise_latent'))
    assert not (save / 'reconstructions').exists()
    assert not (save / 'reconstruction_metrics.json').exists()
    assert not (save / 'initial_metrics.json').exists()
    # A legacy checkpoint's stored training eval batch must also be ignored.
    checkpoint['fixed_batch'] = ('legacy training data', {})
    torch.save(checkpoint, save / 'autoencoder.pt')
    monkeypatch.setattr(sys, 'argv', argv[:-1] + ['2', '--auto_resume'])
    vae.main()
    resumed = torch.load(save / 'autoencoder.pt', weights_only=False)
    assert resumed['step'] == 2 and 'fixed_batch' not in resumed
    assert seen == [expected_target, expected_target]


@pytest.mark.parametrize('loader', [None, []])
def test_empty_validation_does_not_run_model(monkeypatch, loader):
    def unexpected(*args):
        pytest.fail('Empty validation must not evaluate a training sample')

    monkeypatch.setattr(vae, 'evaluate', unexpected)
    device = torch.device('cpu')
    assert vae.evaluate_validation(None, loader, 16, device) is None



def test_vae_evaluates_all_val_batches_with_sample_weighting(monkeypatch):
    batches = [(torch.full((2, 1, 12, 60), 2.0), {'y': {'tag': ['a', 'b']}}),
               (torch.full((1, 1, 12, 60), 8.0), {'y': {'tag': ['c']}})]

    def evaluate(model, x, y, batch_size):
        assert x.shape[2] == 9
        records = [{'motion_name': tag, 'object_type': 'val_species',
                    'mse': float(x[i].mean()), 'zero_z_mse': 10.0,
                    'shuffled_z_mse': 900.0 if tag == 'c' else 12.0,
                    'angular_step_rms_ratio': None}
                   for i, tag in enumerate(y['tag'])]
        return {'summary': {'shuffle_available': len(x) > 1}, 'samples': records}

    monkeypatch.setattr(vae, 'evaluate', evaluate)
    report = vae.evaluate_validation(None, batches, 16, torch.device('cpu'))
    assert report['split'] == 'val'
    assert report['summary']['sample_count'] == 3
    assert report['summary']['mse'] == 4.0
    assert report['summary']['zero_z_mse_increase'] == 6.0
    assert report['summary']['shuffled_z_mse'] == 12.0
    assert report['summary']['shuffled_z_mse_increase'] == 10.0
    assert [r['motion_name'] for r in report['samples']] == ['a', 'b', 'c']
