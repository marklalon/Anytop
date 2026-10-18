"""Reference routing, bottleneck invariance, null semantics and encoder-only augmentation."""
import copy
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model.anytop import AnyTop
from model.cfg_sampler import ClassifierFreeReferenceModel
from data_loaders.reference import collate_reference, prepare_reference_sample
from data_loaders.truebones.truebones_utils.joint_struct_features import JOINT_STRUCT_DIM
from sample.reference_motion import _prepare_encoder_reference_bundle
from test_bone_length_aug import fixture, make_dataset


def model(**kwargs):
    return AnyTop(8, 12, latent_dim=16, ff_size=32, num_layers=2, num_heads=4,
                  t5_out_dim=4, dropout=0, cross_limb=True, ref_encoder=True,
                  ref_width=8, **kwargs)


def inputs(batch=2, joints=4, frames=5, source_joints=6):
    x = torch.randn(batch, joints, 12, frames)
    y = dict(n_joints=torch.full((batch,), joints), rest_pose=torch.randn(batch, joints, 12),
             joints_names_embs=torch.randn(batch, joints, 4),
             joint_struct=torch.randn(batch, joints, JOINT_STRUCT_DIM),
             joints_padding_mask=torch.ones(batch, 1, 1, joints+1, joints+1),
             graph_dist=torch.zeros(batch, joints, joints),
             joints_relations=torch.zeros(batch, joints, joints))
    y.update(ref_feats=torch.randn(batch, source_joints, 12, frames),
             ref_mask=torch.ones(batch, source_joints, dtype=torch.bool),
             ref_active=torch.ones(batch, dtype=torch.bool),
             ref_rest_pose=torch.randn(batch, source_joints, 12),
             ref_joints_names_embs=torch.randn(batch, source_joints, 4),
             ref_joint_struct=torch.randn(batch, source_joints, JOINT_STRUCT_DIM),
             ref_graph_dist=torch.zeros(batch, source_joints, source_joints),
             ref_joints_relations=torch.zeros(batch, source_joints, source_joints))
    return x, y


def open_gates(m):
    with torch.no_grad():
        for layer in m.seqTransDecoder.layers:
            layer.ref_attn_gate.fill_(0.7)
            layer.ref_ff_gate.fill_(0.6)


def test_zero_gates_match_shared_path_and_are_independent():
    torch.manual_seed(5)
    m = model().eval()
    x, y = inputs()
    active = m(x, torch.zeros(2), y)
    null = m(x, torch.zeros(2), dict(y, ref_active=torch.zeros(2, dtype=torch.bool)))
    torch.testing.assert_close(active, null, rtol=0, atol=0)
    a, b = m.seqTransDecoder.layers
    assert a.ref_attn_gate is not a.ref_ff_gate
    assert a.ref_ff[0].weight.data_ptr() != b.ref_ff[0].weight.data_ptr()
    assert a.linear1.out_features == 32 and a.ref_ff[0].out_features == 256


def test_mixed_hard_bypass_includes_bias_and_source_changes_only_active_rows():
    m = model().eval()
    open_gates(m)
    x, y = inputs()
    y['ref_active'][0] = False
    first = m(x, torch.zeros(2), y)
    other = dict(y, ref_feats=torch.randn_like(y['ref_feats']) * 5)
    second = m(x, torch.zeros(2), other)
    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
    assert not torch.allclose(first[1], second[1])
    # Compare against physically removing the two reference branches.
    saved = m.motion_encoder
    m.motion_encoder = None
    plain = m(x, torch.zeros(2), y)
    m.motion_encoder = saved
    torch.testing.assert_close(first[0], plain[0], rtol=0, atol=0)


def test_null_rows_are_finite_and_have_no_reference_gradients():
    m = model(ref_cfg_drop_prob=0).train()
    open_gates(m)
    x, y = inputs()
    y['ref_active'].zero_()
    y['ref_mask'].zero_()
    out = m(x, torch.zeros(2), y)
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    for n, p in m.named_parameters():
        if n.startswith('motion_encoder.') or '.ref_' in n:
            assert p.grad is None or not p.grad.count_nonzero(), n
    assert any(p.grad is not None and p.grad.count_nonzero() for p in m.output_process.parameters())


def test_gates_start_learning_then_encoder_receives_gradient():
    m = model(ref_cfg_drop_prob=0).train()
    x, y = inputs()
    m(x, torch.zeros(2), y).square().mean().backward()
    for layer in m.seqTransDecoder.layers:
        assert layer.ref_attn_gate.grad.abs() > 0
        assert layer.ref_ff_gate.grad.abs() > 0
    m.zero_grad(set_to_none=True)
    open_gates(m)
    m(x, torch.zeros(2), y).square().mean().backward()
    assert m.motion_encoder.queries.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.input_process.parameters())


def test_encoder_permutation_invariance_padding_and_actual_channel_contract():
    m = model().eval()
    _, y = inputs()
    encoder = m.motion_encoder
    z = encoder(y)
    assert z.shape == (2, 5, 2, 8)
    permutation = torch.tensor([0, 4, 1, 5, 2, 3])  # transport root remains zero
    permuted = dict(y)
    for k in ('ref_feats', 'ref_mask', 'ref_rest_pose', 'ref_joints_names_embs', 'ref_joint_struct'):
        permuted[k] = y[k][:, permutation]
    for k in ('ref_graph_dist', 'ref_joints_relations'):
        permuted[k] = y[k][:, permutation][:, :, permutation]
    torch.testing.assert_close(z, encoder(permuted), rtol=1e-5, atol=1e-6)
    changed = copy.deepcopy(y)
    changed['ref_feats'][:, 1:, 9:] = 100
    changed['ref_feats'][:, 0, 10] = 100
    changed['species_emb'] = torch.randn(2, 99)
    torch.testing.assert_close(z, encoder(changed), rtol=0, atol=0)
    padded = dict(y)
    for k in ('ref_feats', 'ref_mask', 'ref_rest_pose', 'ref_joints_names_embs', 'ref_joint_struct'):
        shape = list(y[k].shape)
        shape[1] = 3
        padded[k] = torch.cat([y[k], y[k].new_zeros(shape)], dim=1)
    for k in ('ref_graph_dist', 'ref_joints_relations'):
        padded[k] = torch.nn.functional.pad(y[k], (0, 3, 0, 3))
    torch.testing.assert_close(z, encoder(padded), rtol=1e-5, atol=1e-6)


def test_reference_dropout_extremes_and_frame_mismatch():
    x, y = inputs()
    m = model(ref_cfg_drop_prob=1).train()
    assert not m._reference_condition(y, 2, 5, x.device)[1].any()
    m.eval()
    assert m._reference_condition(y, 2, 5, x.device)[1].all()
    with pytest.raises(ValueError, match='frame aligned'):
        m(x[..., :-1], torch.zeros(2), y)


class Echo(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x, t, y=None):
        ref = y['ref_active'].float()
        text = y.get('action_label_active', torch.ones_like(ref)).float()
        self.calls.append((ref.clone(), text.clone(), y['species_active']))
        return x * 0 + (1 + ref * 2 + text * ref * 3 + y['species_active'].float() * 7)[:, None, None, None]


@pytest.mark.parametrize('use_text,expected,calls', [(False, 12., 2), (True, 21., 3)])
def test_cfg_algebra_sequential_pass_count_and_species_state(use_text, expected, calls):
    inner = Echo()
    wrapper = ClassifierFreeReferenceModel(inner, 2, 3, use_text)
    x = torch.zeros(2, 4, 12, 5)
    species = torch.ones(2, dtype=torch.bool)
    y = dict(ref_active=torch.ones(2, dtype=torch.bool), species_active=species)
    out = wrapper(x, torch.zeros(2), y)
    torch.testing.assert_close(out, torch.full_like(out, expected))
    assert len(inner.calls) == calls
    assert not inner.calls[0][0].any() and not inner.calls[0][1].any()
    assert inner.calls[1][0].all() and not inner.calls[1][1].any()
    assert all(c[2] is species for c in inner.calls)
    assert 'action_label_active' not in y


def test_augmentation_has_independent_cond_and_target_is_untouched():
    motion, cond, metadata = fixture()
    before_motion, before_cond = motion.copy(), copy.deepcopy(cond)
    options = SimpleNamespace(ref_bone_length_aug_prob=1, ref_bone_length_aug=.2,
                              ref_joint_name_drop_prob=1, ref_leaf_drop_prob=0)
    cond['joints_names_embs'][:] = 1
    random.seed(5)
    sample = prepare_reference_sample(motion, cond, metadata, options)
    np.testing.assert_array_equal(motion, before_motion)
    np.testing.assert_array_equal(cond['rest_pose'], before_cond['rest_pose'])
    assert np.all(cond['joints_names_embs'] == 1)
    assert not sample['joints_names_embs'].any()
    assert not np.allclose(sample['rest_pose'][:, :3], before_cond['rest_pose'][:, :3])
    canvas = torch.zeros(2, 10, 12, 30)
    refs = collate_reference([sample, None], canvas, torch.zeros(2, 10, 4),
                             torch.zeros(2, 10, JOINT_STRUCT_DIM))
    assert refs['ref_mask'].sum() == 7
    assert refs['ref_active'].tolist() == [True, False]
    null = collate_reference([None, None], canvas, torch.zeros(2, 10, 4),
                            torch.zeros(2, 10, JOINT_STRUCT_DIM))
    assert refs.keys() == null.keys()
    assert all(torch.is_tensor(v) for v in refs.values())


def test_encode_packs_source_geometry_and_resamples_time():
    motion, cond, _ = fixture()
    packed = _prepare_encoder_reference_bundle(motion, cond, batch_size=2,
                                               n_frames=20, visible_frames=30)
    assert packed['ref_feats'].shape == (2, 7, 12, 20)
    assert packed['ref_active'].all()
    with pytest.raises(ValueError, match='source skeleton'):
        _prepare_encoder_reference_bundle(motion[:, :-1], cond, batch_size=2,
                                          n_frames=20, visible_frames=30)


@pytest.mark.parametrize('kwargs', [dict(ref_dim=0), dict(ref_slots=0), dict(ref_ff_dim=0),
                                   dict(ref_cfg_drop_prob=float('nan'))])
def test_invalid_reference_configuration(kwargs):
    with pytest.raises(ValueError):
        model(**kwargs)


def test_loader_reference_augmentation_keeps_target_and_metadata_identical(monkeypatch):
    from data_loaders.tensors import truebones_batch_collate
    import data_loaders.truebones.truebones_utils.leaf_drop as leaf_drop
    dataset, data = make_dataset()
    dataset.opt.bone_length_aug_prob = 0
    random.seed(2)
    original = dataset._prepare_sample('test', data)
    dataset.opt.ref_encoder = True
    dataset.opt.ref_leaf_drop_prob = 1
    dataset.opt.ref_bone_length_aug_prob = 1
    dataset.opt.ref_bone_length_aug = .2
    dataset.opt.ref_joint_name_drop_prob = 1
    monkeypatch.setattr(leaf_drop, 'sample_leaf_drop', lambda *a: [3])
    random.seed(2)
    augmented = dataset._prepare_sample('test', data)
    for i in (0, 2, 3, 4, 5, 6, 8):
        np.testing.assert_array_equal(original[i], augmented[i])
    assert original[10] == augmented[10]
    reference = augmented[12]['reference']
    assert reference['feats'].shape[1] == 6
    x, kwargs = truebones_batch_collate([augmented, original])
    assert kwargs['y']['ref_mask'].sum() == 6
    assert kwargs['y']['ref_active'].tolist() == [True, False]
    assert x.shape[1] == dataset.opt.max_joints


def test_compile_full_graph_keeps_reference_and_null_tensor_contract():
    m = model().eval()
    open_gates(m)
    x, y = inputs()
    compiled = torch.compile(m, backend='eager', fullgraph=True)
    torch.testing.assert_close(compiled(x, torch.zeros(2), y), m(x, torch.zeros(2), y))
    null_y = dict(y, ref_active=torch.zeros(2, dtype=torch.bool))
    torch.testing.assert_close(compiled(x, torch.zeros(2), null_y), m(x, torch.zeros(2), null_y))


def test_validation_pairs_reference_and_null_with_same_noise_and_target():
    from train.training_loop import TrainLoop
    import contextlib
    calls = []
    class Diffusion:
        def training_losses(self, m, x, t, model_kwargs, noise):
            active = model_kwargs['y']['ref_active']
            calls.append((x, t, noise, active.clone()))
            return {'l_simple': active.float() + 1, 'loss': active.float() + 1}
    stub = TrainLoop.__new__(TrainLoop)
    stub.diffusion = Diffusion()
    stub._autocast_context = contextlib.nullcontext
    stub.step, stub.resume_step = 0, 0
    x, y = inputs()
    t = torch.zeros(2)
    noise = torch.randn_like(x)
    out = stub._validation_losses(model().eval(), x, {'y': y}, t, noise)
    assert out['l_simple'] == 1 and out['reference_l_simple'] == 2
    assert len(calls) == 2
    assert all(c[0] is x and c[1] is t and c[2] is noise for c in calls)
    assert not calls[0][3].any() and calls[1][3].all()
    assert y['ref_active'].all()
