"""Reference descriptor targets, their collate contract and the descriptor head loss."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loaders.reference import (
    REFERENCE_DESCRIPTOR_CHANNELS, REFERENCE_MIN_HEIGHT_RANGE, collate_reference,
    reference_descriptor_stats, reference_descriptors, reference_sample,
)
from data_loaders.truebones.truebones_utils.canonical_features import (
    REST_LENGTH_SCALE_KEY, _length_scale_from_cond,
)
from data_loaders.truebones.truebones_utils.joint_struct_features import JOINT_STRUCT_DIM
from diffusion.gaussian_diffusion import GaussianDiffusion
from test_bone_length_aug import fixture
from test_reference_motion_encoder import inputs, model

UPRIGHT, ELONGATION = 3, 4


def moving_fixture(seed=0):
    motion, cond, _ = fixture()
    rng = np.random.default_rng(seed)
    motion = motion.copy()
    motion[..., :3] += rng.normal(scale=.1, size=motion[..., :3].shape).astype(np.float32)
    return motion, cond


def yaw(motion, angle):
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)
    out = motion.copy()
    out[..., :3] = motion[..., :3] @ rot.T
    return out


def test_descriptors_ignore_yaw():
    motion, cond = moving_fixture()
    np.testing.assert_allclose(reference_descriptors(yaw(motion, 1.1), cond),
                               reference_descriptors(motion, cond), atol=1e-5)


def test_descriptors_ignore_uniform_scale():
    motion, cond = moving_fixture()
    scaled_cond = copy.deepcopy(cond)
    scaled_cond['rest_pos_ric_hml'] = cond['rest_pos_ric_hml'] * 3
    scaled = motion.copy()
    scaled[..., :3] *= 3
    np.testing.assert_allclose(reference_descriptors(scaled, scaled_cond),
                               reference_descriptors(motion, cond), atol=1e-5)


def test_coincident_helper_joint_weighs_nothing():
    motion, cond = moving_fixture()
    helper_cond = copy.deepcopy(cond)
    helper_cond['parents'] = np.append(cond['parents'], 2)
    helper_cond['rest_pos_ric_hml'] = np.concatenate([cond['rest_pos_ric_hml'],
                                                      cond['rest_pos_ric_hml'][2:3]])
    helper = np.concatenate([motion, motion[:, 2:3]], axis=1)
    # Hold L fixed: an extra joint shifts the rest spread L is measured from,
    # which is L's business, not the descriptors'.
    for c in (cond, helper_cond):
        c[REST_LENGTH_SCALE_KEY] = float(_length_scale_from_cond(cond))
    np.testing.assert_allclose(reference_descriptors(helper, helper_cond),
                               reference_descriptors(motion, cond), atol=1e-5)


def test_uprightness_and_elongation_of_a_stick():
    cond = dict(parents=np.array([-1, 0]), rest_pos_ric_hml=np.array([[0, 0, 0], [0, 1, 0]], np.float32))
    standing = np.zeros((1, 2, 12), np.float32)
    standing[0, 1, 1] = 1
    lying = np.zeros((1, 2, 12), np.float32)
    lying[0, 1, 0] = 1
    up, flat = reference_descriptors(standing, cond)[0], reference_descriptors(lying, cond)[0]
    assert up[UPRIGHT] == pytest.approx(1) and up[ELONGATION] == pytest.approx(0, abs=1e-6)
    assert flat[UPRIGHT] == pytest.approx(0, abs=1e-6) and flat[ELONGATION] == pytest.approx(1)


def test_lowest_joint_height_is_clipped():
    motion, cond = moving_fixture()
    sunk = motion.copy()
    sunk[:, 3, 1] = -100
    assert reference_descriptors(sunk, cond)[:, 1] == pytest.approx(REFERENCE_MIN_HEIGHT_RANGE[0])


def test_stats_cover_every_frame():
    motion, cond = moving_fixture()
    other, _ = moving_fixture(seed=1)
    mean, std = reference_descriptor_stats([(motion, cond), (other[:10], cond)])
    every = np.concatenate([reference_descriptors(motion, cond), reference_descriptors(other[:10], cond)])
    np.testing.assert_allclose(mean, every.mean(0), atol=1e-5)
    np.testing.assert_allclose(std, np.maximum(every.std(0), 1e-3), atol=1e-5)


def test_collate_carries_descriptors_and_zeroes_null_rows():
    motion, cond = moving_fixture()
    sample = reference_sample(motion, cond, cond['joints_names_embs'])
    canvas = torch.zeros(2, 10, 12, 30)
    refs = collate_reference([sample, None], canvas, torch.zeros(2, 10, 4),
                             torch.zeros(2, 10, JOINT_STRUCT_DIM))
    assert refs['ref_desc'].shape == (2, REFERENCE_DESCRIPTOR_CHANNELS, 30)
    np.testing.assert_allclose(refs['ref_desc'][0].numpy(), reference_descriptors(motion, cond).T, atol=1e-6)
    assert not refs['ref_desc'][1].any()
    null = collate_reference([None, None], canvas, torch.zeros(2, 10, 4),
                             torch.zeros(2, 10, JOINT_STRUCT_DIM))
    assert refs.keys() == null.keys()
    sample['desc'] = sample['desc'][:-1]
    with pytest.raises(ValueError, match='descriptors'):
        collate_reference([sample], canvas[:1], torch.zeros(1, 10, 4), torch.zeros(1, 10, JOINT_STRUCT_DIM))


def descriptor_inputs():
    x, y = inputs()
    y['ref_desc'] = torch.randn(2, REFERENCE_DESCRIPTOR_CHANNELS, x.shape[-1])
    return x, y


def test_head_is_built_only_on_request():
    assert model().ref_desc_head is None
    assert not any('ref_desc' in k for k in model().state_dict())
    with pytest.raises(ValueError, match='ref_encoder'):
        from model.anytop import AnyTop
        AnyTop(8, 12, latent_dim=16, ff_size=32, num_layers=2, num_heads=4, t5_out_dim=4,
               ref_desc_head=True)


def forward_then_loss(m, y, x=None):
    x = descriptor_inputs()[0] if x is None else x
    m(x, torch.zeros(x.shape[0], dtype=torch.long), y=y)
    return GaussianDiffusion.reference_descriptor_loss(m, y)


def stats(m):
    m.set_reference_descriptor_stats(np.full(REFERENCE_DESCRIPTOR_CHANNELS, .5),
                                     np.full(REFERENCE_DESCRIPTOR_CHANNELS, 2.))


def test_descriptor_loss_reuses_forward_z_and_zeroes_null_rows():
    torch.manual_seed(0)
    m = model(ref_desc_head=True, ref_cfg_drop_prob=0.0).train()
    stats(m)
    _, y = descriptor_inputs()
    y['ref_active'][1] = False
    calls = []
    m.motion_encoder.register_forward_hook(lambda *_: calls.append(1))
    loss = forward_then_loss(m, y)
    assert len(calls) == 1, 'the descriptor loss must not run a second encoder pass'
    assert loss.shape == (2,) and loss[0] > 0 and loss[1] == 0
    prediction, target, _, rows = m.reference_descriptor_prediction(y)
    torch.testing.assert_close(target, (y['ref_desc'].transpose(1, 2) - .5) / 2.)
    assert prediction.shape == target.shape and rows.tolist() == [0, 1]


def test_descriptor_loss_covers_only_encoded_rows_under_reference_dropout():
    torch.manual_seed(0)
    m = model(ref_desc_head=True, ref_cfg_drop_prob=0.5).train()
    stats(m)
    _, y = descriptor_inputs()
    loss = forward_then_loss(m, y)
    assert y['ref_desc_rows'].numel() == 1 and (loss != 0).sum() == 1
    # one encoded row of two: scaled by B / rows so the batch mean is unbiased
    row = y['ref_desc_rows'].item()
    prediction, target, _, _ = m.reference_descriptor_prediction(y)
    single = torch.nn.functional.smooth_l1_loss(prediction, target, reduction='none').mean((1, 2))[0]
    assert loss[row].item() == pytest.approx(2 * single.item(), rel=1e-5)
    full = model(ref_desc_head=True, ref_cfg_drop_prob=1.0).train()
    stats(full)
    assert not forward_then_loss(full, descriptor_inputs()[1]).any()


def test_descriptor_loss_needs_the_forward_first():
    m = model(ref_desc_head=True).train()
    with pytest.raises(RuntimeError, match='forward'):
        GaussianDiffusion.reference_descriptor_loss(m, descriptor_inputs()[1])


def test_descriptor_loss_trains_only_encoder_and_head():
    m = model(ref_desc_head=True, ref_cfg_drop_prob=0.0).train()
    _, y = descriptor_inputs()
    forward_then_loss(m, y).sum().backward()
    touched = {n.split('.')[0] for n, p in m.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert touched == {'motion_encoder', 'ref_desc_head'}


def test_descriptor_z_handoff_survives_compile():
    torch.manual_seed(0)
    m = model(ref_desc_head=True, ref_cfg_drop_prob=0.5).train()
    stats(m)
    x, y = descriptor_inputs()
    compiled = torch.compile(m, backend='aot_eager', dynamic=False)
    compiled(x, torch.zeros(2, dtype=torch.long), y=y)
    assert 'ref_desc_z' in y and y['ref_desc_z'].requires_grad
    GaussianDiffusion.reference_descriptor_loss(m, y).sum().backward()
    assert m.ref_desc_head.weight.grad is not None
