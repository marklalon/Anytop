"""Reconstruction and transfer I/O preserve rig conditions and clip length."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loaders.truebones.truebones_utils import canonical_features as cf
from sample.generate_ae import load_motion, reconstruct, resolve_cond_path
from sample import generate_ae as evaluator


def make_cond():
    rest = np.zeros((3, 12), dtype=np.float32)
    rest[:, :3] = [[0, 1, 0], [1, 1, 0], [-1, 1, 0]]
    rest[:, 3:9] = [1, 0, 0, 0, 1, 0]
    cond = dict(parents=np.array([-1, 0, 0]), rest_pose=rest,
                offsets=np.array([[0, 1, 0], [1, 0, 0], [-1, 0, 0]], dtype=np.float32),
                joints_names=['Root', 'Left', 'Right'], joint_side_labels=['center', 'left', 'right'],
                joints_names_embs=np.zeros((3, 4), dtype=np.float32),
                joint_relations=np.zeros((3, 3), dtype=np.int64),
                joints_graph_dist=np.zeros((3, 3), dtype=np.int64))
    cf.mark_canonical_cond_entry(cond)
    cf.set_canonical_global_stats(cond, np.zeros(cf.CANONICAL_STAT_LEN, dtype=np.float32),
                                  np.ones(cf.CANONICAL_STAT_LEN, dtype=np.float32), .1)
    return cond


class IdentityAutoencoder:
    def encode(self, x, skeleton, frame, loop):
        self.time_metadata = loop
        return x

    def decode(self, z, skeleton, frame, loop):
        assert loop is self.time_metadata
        return z


@pytest.mark.parametrize('is_loop', [False, True])
@pytest.mark.parametrize('length', [1, 2, 5, 8, 12])
def test_identity_reconstruction_preserves_stored_layout_and_duration(length, is_loop):
    cond = make_cond()
    working = np.broadcast_to(cond['rest_pose'], (length, 3, 12)).copy()
    working[:, 0, 0] = np.arange(length) * .2
    working[:, 0, 2] = np.arange(length) * -.1
    stored = cf.working_to_stored(working, is_loop=is_loop)
    model = IdentityAutoencoder()
    actual, mse = reconstruct(model, stored, 'Test', cond, frames=8,
                              is_loop=is_loop, cycles=1, device=torch.device('cpu'), amp_dtype='fp32')
    torch.testing.assert_close(model.time_metadata['resample_speed_cond'], torch.tensor([length / 8]))
    assert mse == 0
    assert actual.shape == stored.shape
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    if length <= 8:
        np.testing.assert_allclose(actual, stored, atol=2e-6)


@pytest.mark.parametrize('shape', [(0, 3, 12), (5, 2, 12), (5, 3, 9), (3, 12)])
def test_reject_incompatible_motion(tmp_path, shape):
    path = tmp_path / 'motion.npy'
    np.save(path, np.zeros(shape))
    with pytest.raises(ValueError, match='expected'):
        load_motion(path, make_cond())


def test_reject_nonfinite_and_raw_animation(tmp_path):
    path = tmp_path / 'motion.npy'
    np.save(path, np.full((5, 3, 12), np.nan))
    with pytest.raises(ValueError, match='finite'):
        load_motion(path, make_cond())
    with pytest.raises(ValueError, match='Only preprocessed'):
        load_motion(tmp_path / 'motion.glb', make_cond())


def test_default_cond_cannot_silently_change_since_training(tmp_path):
    training = tmp_path / 'training.npy'
    training.write_bytes(b'original training cond')
    checkpoint = dict(args=dict(cond_path=str(training)),
                      cond_sha256=hashlib.sha256(training.read_bytes()).hexdigest())
    model_path = tmp_path / 'autoencoder.pt'
    assert resolve_cond_path(checkpoint, model_path) == training
    training.write_bytes(b'new preprocessing')
    with pytest.raises(ValueError, match='differs from the training cond'):
        resolve_cond_path(checkpoint, model_path)
    assert resolve_cond_path(checkpoint, model_path, str(training)) == training


@pytest.mark.parametrize('target_joints', [2, 3])
def test_transfer_uses_source_encoding_target_decoding_and_target_physical_scale(target_joints):
    source = make_cond()
    target_cond = make_cond()
    for key in ('parents', 'rest_pose', 'offsets', 'joints_names_embs'):
        target_cond[key] = target_cond[key][:target_joints].copy()
    for key in ('joints_names', 'joint_side_labels'):
        target_cond[key] = target_cond[key][:target_joints]
    for key in ('joint_relations', 'joints_graph_dist'):
        target_cond[key] = target_cond[key][:target_joints, :target_joints].copy()
    target_cond['rest_pose'][:, :3] *= 2
    target_cond['offsets'] *= 2
    # Rebuild geometry derived by mark_canonical_cond_entry after changing the rig.
    target_cond.pop('rest_pos_ric_hml', None)
    cf.mark_canonical_cond_entry(target_cond)
    length, frames = 5, 8
    working = np.broadcast_to(source['rest_pose'], (length, 3, 12)).copy()
    working[:, 0, 0] = np.arange(length) * .2
    stored = cf.working_to_stored(working, is_loop=False)
    expected_canonical = cf.physical_hml_to_canonical(
        np.broadcast_to(target_cond['rest_pose'], (frames, target_joints, 12)).copy(), target_cond)
    prediction = torch.from_numpy(expected_canonical[..., :9]).permute(1, 2, 0)[None]

    class TransferAutoencoder:
        def encode(self, x, skeleton, frame, loop):
            assert skeleton['n_joints'].item() == 3
            expected = cf.physical_hml_to_canonical(
                evaluator.resample_motion_features(cf.stored_to_working(stored), frames), source)
            torch.testing.assert_close(x, torch.from_numpy(expected[..., :9]).permute(1, 2, 0)[None])
            self.latent = torch.zeros(1, frames, 2, 4)
            self.loop = loop
            return self.latent

        def decode(self, z, skeleton, frame, loop):
            assert z is self.latent
            assert loop is self.loop
            assert skeleton['n_joints'].item() == target_joints
            torch.testing.assert_close(loop['resample_speed_cond'], torch.tensor([length / frames]))
            return prediction

    actual, mse = reconstruct(TransferAutoencoder(), stored, 'Source', source, frames=frames,
                              is_loop=False, cycles=1, device=torch.device('cpu'), amp_dtype='fp32',
                              target_object_type='Target', target_cond=target_cond)
    expected = cf.working_to_stored(
        np.broadcast_to(target_cond['rest_pose'], (length, target_joints, 12)).copy(), is_loop=False)
    assert mse is None  # Even equal joint counts do not define a cross-rig ground truth.
    np.testing.assert_allclose(actual, expected, atol=2e-6)


@pytest.mark.parametrize('target_name', [None, 'Bear', 'Horse'])
@pytest.mark.parametrize('ik_args,expected_options', [
    ([], {'fullbody_ik': False, 'stretch_factor': .1}),
    (['--fullbody_ik', '--stretch_factor', '0'], {'fullbody_ik': True, 'stretch_factor': 0.0}),
    (['--fullbody_ik', '--stretch_factor', '0.2'], {'fullbody_ik': True, 'stretch_factor': .2}),
])
def test_cli_infers_source_and_exports_each_motion_with_its_own_skeleton(
        tmp_path, monkeypatch, target_name, ik_args, expected_options):
    source, target = make_cond(), make_cond()
    target['joints_names'] = ['TargetRoot', 'TargetLeft', 'TargetRight']
    cond_dict = {'Bear': source, 'Horse': target}
    stored = np.broadcast_to(source['rest_pose'], (5, 3, 12)).copy()
    path = tmp_path / 'Bear_WetShake.npy'
    np.save(path, stored)
    checkpoint = dict(kind='motion_autoencoder_7a', canonical_version=cf.CANONICAL_FEATURE_SPACE,
                      decoder_content_version=evaluator.DECODER_CONTENT_VERSION,
                      model_config={}, model={}, args=dict(num_frames=8), step=1)

    class FakeModel:
        config = {'t5_out_dim': 4}
        variational = False

        def load_state_dict(self, *args, **kwargs):
            pass

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(evaluator.torch, 'load', lambda *a, **k: checkpoint)
    monkeypatch.setattr(evaluator, 'MotionAutoencoder', lambda **k: FakeModel())
    monkeypatch.setattr(evaluator, 'resolve_cond_path', lambda *a: tmp_path / 'cond.npy')
    monkeypatch.setattr(evaluator, 'load_cond', lambda *a: cond_dict)
    monkeypatch.setattr(evaluator, 'ensure_joint_name_embeddings', lambda *a, **k: None)
    exports = []
    from sample import export
    bvh_exports = []
    monkeypatch.setattr(export, 'write_feature_bvh', lambda *a, **k: bvh_exports.append((a, k)))

    def export_spy(task):
        exports.append(task)
        return export._export_motion(task)

    monkeypatch.setattr(evaluator, '_export_motion', export_spy)

    def reconstruct_spy(model, motion, object_type, cond, **kwargs):
        assert object_type == 'Bear'
        assert cond is source
        assert kwargs['target_object_type'] == (target_name or 'Bear')
        assert kwargs['target_cond'] is (target if target_name == 'Horse' else source)
        return motion, None if target_name == 'Horse' else 0.0

    monkeypatch.setattr(evaluator, 'reconstruct', reconstruct_spy)
    out = tmp_path / 'out'
    argv = ['--model_path', str(tmp_path / 'autoencoder.pt'), '--npy', str(path),
            '--output_dir', str(out), '--device', 'cpu', '--loop', 'off']
    if target_name:
        argv += ['--object_type', target_name]
    argv += ik_args
    evaluator.main(argv)
    assert exports[0][1] is source
    assert exports[1][1] is (target if target_name == 'Horse' else source)
    assert exports[1][3] == (target if target_name == 'Horse' else source)['joints_names']
    assert len(bvh_exports) == 2
    for task, (bvh_args, options) in zip(exports, bvh_exports):
        assert bvh_args[1] is task[1]
        assert options['fullbody_ik'] == expected_options['fullbody_ik']
        assert options['stretch_factor'] == expected_options['stretch_factor']
        np.testing.assert_array_equal(np.load(out / task[2]), stored)
    record = json.loads((out / 'reconstruction_metrics.json').read_text())['samples'][0]
    assert record['source_object_type'] == 'Bear'
    assert record['target_object_type'] == (target_name or 'Bear')
    assert record['canonical_window_mse'] == (None if target_name == 'Horse' else 0.0)
