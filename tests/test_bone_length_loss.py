"""Unit tests for the target-relative bone-length consistency loss.

Covers the core correctness guarantees:
  * identical pred/target -> zero loss
  * a compressed bone -> positive loss matching a hand computation
  * legitimately-varying GT length is NOT penalized when the prediction
    follows it (the "no false-positive on stretchy joints" guarantee)
  * a rigid prediction against a varying GT IS penalized (the loss tracks the
    target, not a rest prior)
  * padded joints and roots are excluded
  * per-sample parents and masks in a batch
"""
import os
import sys

import numpy as np
import pytest
import torch as th

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion.gaussian_diffusion import GaussianDiffusion


def _loss():
    """Bare instance: the method uses no __init__ state, so skip construction."""
    return object.__new__(GaussianDiffusion)


def _feature_tensor(positions):
    """Embed (B, J, 3, T) RIC positions into a (B, J, F, T) feature tensor.

    Only channels [0:3] are read by the loss; the rest are filled with noise to
    prove they are ignored.
    """
    positions = th.as_tensor(positions, dtype=th.float32)
    B, J, _three, T = positions.shape
    feats = th.randn(B, J, 12, T)
    feats[:, :, 0:3, :] = positions
    return feats


def _spat_mask(B, J, valid_counts):
    mask = th.zeros(B, 1, 1, J)
    for bi, n in enumerate(valid_counts):
        mask[bi, 0, 0, :n] = 1.0
    return mask


def test_identical_pred_target_is_zero():
    # chain: 0 (root) -> 1 -> 2
    parents = [[-1, 0, 1]]
    pos = np.zeros((1, 3, 3, 4), dtype=np.float32)
    pos[0, 1] = np.array([1.0, 0.0, 0.0])[:, None]   # joint 1 at x=1
    pos[0, 2] = np.array([1.0, 2.0, 0.0])[:, None]   # joint 2 at y=2 above joint 1
    feats = _feature_tensor(pos)
    out = _loss().bone_length_consistency_loss(
        feats.clone(), feats.clone(), parents, _spat_mask(1, 3, [3])
    )
    assert out.item() == pytest.approx(0.0, abs=1e-7)


def test_compressed_bone_matches_hand_computation():
    parents = [[-1, 0, 1]]
    T = 5
    tgt = np.zeros((1, 3, 3, T), dtype=np.float32)
    tgt[0, 1] = np.array([1.0, 0.0, 0.0])[:, None]
    tgt[0, 2] = np.array([1.0, 2.0, 0.0])[:, None]   # bone 1->2 length = 2.0

    pred = tgt.copy()
    pred[0, 2] = np.array([1.0, 1.5, 0.0])[:, None]  # compresses bone 1->2 to 1.5

    out = _loss().bone_length_consistency_loss(
        _feature_tensor(pred), _feature_tensor(tgt), parents, _spat_mask(1, 3, [3])
    )
    # Two valid bones (0->1 unchanged, 1->2 off by 0.5). Mean over (bone, frame):
    # bone 0->1 err = 0; bone 1->2 err = (2.0-1.5)^2 = 0.25, over all T frames.
    # mean = (0 + 0.25) / 2 = 0.125
    assert out.item() == pytest.approx(0.125, abs=1e-6)


def test_legitimate_length_variation_not_penalized():
    """GT bone length varies over time; a prediction that follows it -> zero loss.

    This is the key guarantee: target-relative supervision never punishes a
    legitimately stretchy joint as long as the prediction reproduces the true
    (varying) length.
    """
    parents = [[-1, 0]]
    T = 6
    tgt = np.zeros((1, 2, 3, T), dtype=np.float32)
    # joint 1 slides out along y from 1.0 to 2.0 over the clip (length varies).
    tgt[0, 1, 1, :] = np.linspace(1.0, 2.0, T)
    feats = _feature_tensor(tgt)
    out = _loss().bone_length_consistency_loss(
        feats.clone(), feats.clone(), parents, _spat_mask(1, 2, [2])
    )
    assert out.item() == pytest.approx(0.0, abs=1e-7)


def test_rigid_prediction_against_varying_target_is_penalized():
    """Tracks the target, not a rest prior: a constant-length prediction is
    penalized when the GT length actually varies."""
    parents = [[-1, 0]]
    T = 6
    tgt = np.zeros((1, 2, 3, T), dtype=np.float32)
    tgt[0, 1, 1, :] = np.linspace(1.0, 2.0, T)        # varying GT length

    pred = np.zeros((1, 2, 3, T), dtype=np.float32)
    pred[0, 1, 1, :] = 1.0                              # rigid prediction at len 1.0

    out = _loss().bone_length_consistency_loss(
        _feature_tensor(pred), _feature_tensor(tgt), parents, _spat_mask(1, 2, [2])
    )
    expected = float(np.mean((np.linspace(1.0, 2.0, T) - 1.0) ** 2))
    assert out.item() == pytest.approx(expected, abs=1e-6)


def test_padded_joints_excluded():
    """A padded (invalid) joint must not contribute even if its values differ."""
    # J=4 slots, but only 3 valid joints; slot 3 is padding with garbage.
    parents = [[-1, 0, 1]]   # parents only describes the 3 real joints
    T = 3
    base = np.zeros((1, 4, 3, T), dtype=np.float32)
    base[0, 1] = np.array([1.0, 0.0, 0.0])[:, None]
    base[0, 2] = np.array([1.0, 2.0, 0.0])[:, None]
    pred = base.copy()
    pred[0, 3] = 999.0       # garbage in the padded slot
    tgt = base.copy()
    tgt[0, 3] = -999.0

    out = _loss().bone_length_consistency_loss(
        _feature_tensor(pred), _feature_tensor(tgt), parents, _spat_mask(1, 4, [3])
    )
    assert out.item() == pytest.approx(0.0, abs=1e-7)


def test_per_sample_parents_and_masks():
    """Batch with different skeletons / valid counts handled independently."""
    # sample 0: chain 0->1->2 (3 valid); sample 1: chain 0->1 (2 valid, 1 pad)
    parents = [[-1, 0, 1], [-1, 0]]
    T = 2
    pos = np.zeros((2, 3, 3, T), dtype=np.float32)
    # sample 0
    pos[0, 1] = np.array([1.0, 0.0, 0.0])[:, None]
    pos[0, 2] = np.array([1.0, 1.0, 0.0])[:, None]
    # sample 1
    pos[1, 1] = np.array([3.0, 0.0, 0.0])[:, None]

    tgt = _feature_tensor(pos)
    pred_pos = pos.copy()
    pred_pos[1, 1] = np.array([2.0, 0.0, 0.0])[:, None]  # sample1 bone 0->1: 3->2
    pred = _feature_tensor(pred_pos)

    out = _loss().bone_length_consistency_loss(
        pred, tgt, parents, _spat_mask(2, 3, [3, 2])
    )
    # Valid bones: sample0 has 2 (both exact), sample1 has 1 (err (3-2)^2=1 each T).
    # total valid bones across batch = 3; sum of squared err over frames = 1*T.
    # denom = (#valid bones) * T = 3*T; numerator = 1*T -> 1/3.
    assert out.item() == pytest.approx(1.0 / 3.0, abs=1e-6)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
