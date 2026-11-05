"""Real paired rigs use the AE cross/self path without latent alignment."""
import copy
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from .test_bone_length_aug import make_dataset
from data_loaders.reference import REST_GEOM_COLUMNS, rest_geometry
from data_loaders.tensors import truebones_batch_collate
from data_loaders.truebones.truebones_utils.joint_struct_features import build_joint_struct_features
from data_loaders.truebones.truebones_utils.leaf_drop import drop_joints_from_cond
from model.motion_autoencoder import MotionAutoencoder
from train.motion_reconstruction_losses import MotionReconstructionLosses
from train.train_autoencoder import (
    AutoencoderBatchTransform, configure_autoencoder_views, dual_view_training_terms,
    evaluate_dual_view_validation, parser,
)


def paired_dataset(tmp_path, is_loop=False):
    source, original_data = make_dataset()
    original = source._load_physical_motion(original_data)[0]
    keep, smaller = drop_joints_from_cond(source.cond_dict['test'], [3, 6])
    source.cond_dict['other'] = smaller
    source._joint_struct_cache['other'] = build_joint_struct_features(smaller)
    source._rest_geometry_cache['other'] = rest_geometry(smaller)
    clips = {'test': original, 'other': original[:, keep].copy()}
    # Distinguish the partner's dynamic content, not only its skeleton.
    clips['other'][:, 0, 0] += np.linspace(0, .3, len(original))
    def load(data):
        rig = data['object_type']
        cond = source.cond_dict[rig]
        return (clips[rig], len(original), rig, cond['parents'], cond['joints_graph_dist'],
                cond['joint_relations'], rest_geometry(cond)[:, REST_GEOM_COLUMNS],
                cond['offsets'], cond['joints_names_embs'], [])
    source._load_physical_motion = load
    source.data_dict = {}
    for rig in clips:
        data = copy.deepcopy(original_data)
        data.update(object_type=rig, length=len(original))
        data['motion_metadata']['is_loop'] = is_loop
        source.data_dict[f'fixture/{rig}.npy'] = data
        (tmp_path / f'{rig}.npy').touch()
    (tmp_path / 'reference_pairs.jsonl').write_text(json.dumps(
        dict(motion='other.npy', source='fixture/test.npy')), encoding='utf-8')
    source.opt.sources = [SimpleNamespace(namespace='fixture', root=tmp_path, motion_dir=tmp_path)]
    source.opt.bone_length_aug_prob = 0.
    source.name_list = list(source.data_dict)
    source.pointer, source.use_weighted_sampler = 0, False
    args = parser().parse_args(['--save_dir', 'unused', '--reference_pairs', '--batch_size', '2',
                               '--joint_buckets', '5', '7'])
    configure_autoencoder_views(source, args)
    return source, args


@pytest.mark.parametrize('is_loop', [False, True])
@pytest.mark.parametrize('expected', ['other', 'test'])
def test_uniform_group_including_self_shares_time_map(tmp_path, monkeypatch, is_loop, expected):
    source, _ = paired_dataset(tmp_path, is_loop)
    source.opt.motion_speed_aug = 1.3
    source.opt.loop_tile_single_prob = .5
    source.opt.loop_cond_prob = .5
    choices = []
    def choose(candidates):
        choices.append(tuple(candidates))
        return f'fixture/{expected}.npy'
    monkeypatch.setattr(random, 'choice', choose)
    sample = source.prepare_sample_by_name('fixture/test.npy', target_num_frames=45)
    second = sample[12]['autoencoder_view']
    assert second[11] == f'fixture/{expected}.npy'
    assert len(second[2]) == (5 if expected == 'other' else 7)
    assert second[12]['autoencoder_view'] is None
    assert choices == [source.reference_groups['fixture/test.npy']]
    for key in ('is_loop', 'resample_speed_cond', 'loop_uncond', 'loop_phase_offset',
                'loop_tile_count', 'motion_speed_applied'):
        assert sample[10][key] == second[10][key]
    assert source.opt.autoencoder_dual_view  # --reference_pairs alone enables it.
    np.testing.assert_array_equal(source.sampler_index_joint_counts(), [7, 7])


def test_real_pair_collation_and_single_pass_backward(tmp_path):
    source, _ = paired_dataset(tmp_path)
    source.opt.autoencoder_reference_only_partners = True
    sample = source.prepare_sample_by_name('fixture/test.npy', target_num_frames=12)
    x, wrapped = AutoencoderBatchTransform()(truebones_batch_collate([sample], joint_buckets=(5, 7)))
    y = wrapped['y']
    assert x.shape == y['ae_view_motion'].shape == (1, 7, 9, 12)
    assert y['n_joints'].item() == 7 and y['ae_view_cond']['n_joints'].item() == 5
    model = MotionAutoencoder(t5_out_dim=4, width=16, layers=1, heads=2, slots=2, latent_dim=4)
    terms = dual_view_training_terms(model, MotionReconstructionLosses(), x, y)
    assert set(terms).isdisjoint({'align_loss', 'latent_alignment_loss', 'L_align'})
    terms['loss'].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


@pytest.mark.parametrize('index', [0, 1, 2])
def test_three_member_group_is_one_uniform_draw_including_self(tmp_path, monkeypatch, index):
    source, args = paired_dataset(tmp_path)
    source.data_dict['fixture/third.npy'] = copy.deepcopy(source.data_dict['fixture/test.npy'])
    (tmp_path / 'third.npy').touch()
    with (tmp_path / 'reference_pairs.jsonl').open('a', encoding='utf-8') as stream:
        stream.write('\n' + json.dumps(dict(motion='third.npy', source='fixture/test.npy')))
    configure_autoencoder_views(source, args)
    group = source.reference_groups['fixture/test.npy']
    assert len(group) == 3
    calls = []
    def choose(candidates):
        calls.append(tuple(candidates))
        return candidates[index]
    monkeypatch.setattr(random, 'choice', choose)
    sample = source.prepare_sample_by_name('fixture/test.npy')
    assert calls == [group]
    assert sample[12]['autoencoder_view'][11] == group[index]


def test_filtered_out_partner_falls_back_to_self(tmp_path):
    source, args = paired_dataset(tmp_path)
    del source.data_dict['fixture/other.npy']
    source.name_list = ['fixture/test.npy']
    configure_autoencoder_views(source, args)
    assert source.reference_groups == {}
    sample = source.prepare_sample_by_name('fixture/test.npy')
    assert sample[12]['autoencoder_view'][11] == 'fixture/test.npy'


def test_pair_contract_rejects_mismatched_length(tmp_path):
    source, args = paired_dataset(tmp_path)
    source.data_dict['fixture/other.npy']['length'] += 1
    with pytest.raises(ValueError, match='same motion frame for frame'):
        configure_autoencoder_views(source, args)


def test_validation_reports_real_pairs_and_preserves_training_sampling(tmp_path):
    source, args = paired_dataset(tmp_path)
    loader = SimpleNamespace(dataset=SimpleNamespace(motion_dataset=source))
    model = MotionAutoencoder(t5_out_dim=4, width=16, layers=1, heads=2, slots=2, latent_dim=4)
    first = evaluate_dual_view_validation(model, loader, args, 'cpu')
    assert first == evaluate_dual_view_validation(model, loader, args, 'cpu')
    assert first['reference_pair_sample_count'] == first['dual_view_sample_count'] == 2
    assert np.isfinite(first['reference_pair_cross_mse'])
    assert 'reference_pair_self_mse' not in first
    assert 'dual_view_self_mse' not in first
    assert first['reference_pair_cross_mse'] == first['dual_view_cross_mse']
    assert not source.opt.autoencoder_reference_only_partners
    del source.data_dict['fixture/other.npy']
    source.name_list = ['fixture/test.npy']
    report = evaluate_dual_view_validation(model, loader, args, 'cpu')
    assert report['reference_pair_sample_count'] == 0
    assert 'reference_pair_cross_mse' not in report
