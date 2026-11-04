"""Reconstruction I/O must preserve root travel, rotations and clip length."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loaders.truebones.truebones_utils import canonical_features as cf
from sample.eval_autoencoder import load_motion, reconstruct, resolve_cond_path


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
        return x

    def decode(self, z, skeleton, frame, loop):
        return z


@pytest.mark.parametrize('is_loop', [False, True])
@pytest.mark.parametrize('length', [1, 2, 5, 8, 12])
def test_identity_reconstruction_preserves_stored_layout_and_duration(length, is_loop):
    cond = make_cond()
    working = np.broadcast_to(cond['rest_pose'], (length, 3, 12)).copy()
    working[:, 0, 0] = np.arange(length) * .2
    working[:, 0, 2] = np.arange(length) * -.1
    stored = cf.working_to_stored(working, is_loop=is_loop)
    actual, mse = reconstruct(IdentityAutoencoder(), stored, 'Test', cond, frames=8,
                              is_loop=is_loop, cycles=1, device=torch.device('cpu'), amp_dtype='fp32')
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
