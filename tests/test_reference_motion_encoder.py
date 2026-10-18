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
from data_loaders.reference import (
    bone_directions, collate_reference, prepare_reference_sample, reference_sample,
)
from data_loaders.truebones.data.dataset import resample_motion_features
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
             ref_bone_dir=torch.nn.functional.normalize(
                 torch.randn(batch, source_joints, 3, frames + 1), dim=2),
             ref_mask=torch.ones(batch, source_joints, dtype=torch.bool),
             ref_active=torch.ones(batch, dtype=torch.bool),
             ref_rest_pose=torch.randn(batch, source_joints, 12),
             ref_joints_names_embs=torch.randn(batch, source_joints, 4),
             ref_joint_struct=torch.randn(batch, source_joints, JOINT_STRUCT_DIM))
    return x, y


def open_gates(m):
    with torch.no_grad():
        for layer in m.seqTransDecoder.layers:
            layer.ref_attn_gate.fill_(0.7)


def test_zero_gates_match_shared_path_and_are_independent():
    torch.manual_seed(5)
    m = model().eval()
    x, y = inputs()
    active = m(x, torch.zeros(2), y)
    null = m(x, torch.zeros(2), dict(y, ref_active=torch.zeros(2, dtype=torch.bool)))
    torch.testing.assert_close(active, null, rtol=0, atol=0)
    a, b = m.seqTransDecoder.layers
    assert a.ref_attn_gate is not b.ref_attn_gate
    assert a.ref_key.weight.data_ptr() != b.ref_key.weight.data_ptr()
    assert not any('ref_ff' in n for n, _ in m.named_parameters())


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
    # Compare against physically removing the reference branch.
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
    for k in ('ref_feats', 'ref_bone_dir', 'ref_mask', 'ref_rest_pose', 'ref_joints_names_embs',
              'ref_joint_struct'):
        permuted[k] = y[k][:, permutation]
    torch.testing.assert_close(z, encoder(permuted), rtol=1e-5, atol=1e-6)
    changed = copy.deepcopy(y)
    changed['ref_feats'][:, 1:, 9:] = 100
    changed['ref_feats'][:, 0, 10] = 100
    changed['species_emb'] = torch.randn(2, 99)
    torch.testing.assert_close(z, encoder(changed), rtol=0, atol=0)
    padded = dict(y)
    for k in ('ref_feats', 'ref_bone_dir', 'ref_mask', 'ref_rest_pose', 'ref_joints_names_embs',
              'ref_joint_struct'):
        shape = list(y[k].shape)
        shape[1] = 3
        padded[k] = torch.cat([y[k], y[k].new_zeros(shape)], dim=1)
    torch.testing.assert_close(z, encoder(padded), rtol=1e-5, atol=1e-6)


def test_encoder_reads_bone_directions_including_rest_row():
    m = model().eval()
    _, y = inputs()
    z = m.motion_encoder(y)
    for rows in (slice(0, 1), slice(1, None)):
        changed = copy.deepcopy(y)
        changed['ref_bone_dir'][..., rows] = -changed['ref_bone_dir'][..., rows]
        assert not torch.allclose(z, m.motion_encoder(changed))


def test_bone_directions_are_facing_frame_units_with_rest_row():
    parents = np.array([-1, 0, 1, 1])
    rest = np.array([[0, 1, 0], [0, 1, 0], [0, 1, -1], [0, 1.0005, -1]], np.float32)
    positions = np.repeat(rest[None], 2, axis=0)
    positions[1, 2] = [0, 3, 0]  # second frame: tail raised straight up
    out = bone_directions(positions, parents, rest)
    assert out.shape == (3, 4, 3)
    np.testing.assert_array_equal(out[:, 0], 0)  # root
    np.testing.assert_array_equal(out[:, 1], 0)  # coincident with its parent
    np.testing.assert_allclose(out[0, 2], [0, 0, -1])
    np.testing.assert_allclose(out[1, 2], [0, 0, -1])
    np.testing.assert_allclose(out[2, 2], [0, 1, 0])
    np.testing.assert_allclose(np.linalg.norm(out[:, 3], axis=-1), 1, rtol=1e-6)


def test_bone_directions_sanitize_non_finite_positions():
    parents = np.array([-1, 0])
    rest = np.array([[0, 1, 0], [0, 2, 0]], np.float32)
    positions = np.array([[[0, 1, 0], [np.inf, 2, 0]],
                          [[0, 1, 0], [np.nan, 2, 0]]], np.float32)
    out = bone_directions(positions, parents, rest)
    assert out.shape == (3, 2, 3)
    assert np.isfinite(out).all()


def test_collate_rejects_bone_direction_frame_mismatch():
    motion, cond, _ = fixture()
    sample = prepare_reference_sample(motion, cond, {'translation_root_index': 0},
                                      SimpleNamespace(), augment=False)
    sample['bone_dir'] = sample['bone_dir'][:-1]
    with pytest.raises(ValueError, match='bone directions'):
        collate_reference([sample], torch.zeros(1, 7, 12, 30), torch.zeros(1, 7, 4),
                          torch.zeros(1, 7, JOINT_STRUCT_DIM))


def test_encoder_attention_has_no_graph_relations():
    m = model()
    names = [n for n, _ in m.motion_encoder.named_parameters()]
    assert not any('topology' in n or 'edge_' in n for n in names)
    assert m.seqTransDecoder.topology_key_emb is not None


def test_reference_dropout_keeps_fixed_rows_and_encodes_only_those(monkeypatch):
    m = model(ref_cfg_drop_prob=.5).train()
    x, y = inputs(batch=4)
    seen = []
    forward = m.motion_encoder.forward
    monkeypatch.setattr(m.motion_encoder, 'forward',
                        lambda ref_y: seen.append(ref_y['ref_feats'].shape[0]) or forward(ref_y))
    z, active = m._reference_condition(y, 4, 5, x.device)
    assert seen == [2] and int(active.sum()) == 2
    assert z[~active].abs().sum() == 0 and z[active].abs().sum() > 0
    y['ref_active'][:] = False
    assert not m._reference_condition(y, 4, 5, x.device)[1].any()


def test_reference_rows_bypass_species_film(monkeypatch):
    m = model(ref_cfg_drop_prob=0, species_joint_cond=True, species_cfg_drop_prob=0).eval()
    x, y = inputs()
    y['ref_active'][1] = False
    y['species_active'] = torch.ones(2, dtype=torch.bool)
    y['species_emb'] = torch.randn(2, 4)
    seen = []
    forward = m.input_process.forward
    monkeypatch.setattr(m.input_process, 'forward',
                        lambda *a, **k: seen.append(k['species_active'].clone()) or forward(*a, **k))
    m(x, torch.zeros(2), y)
    assert seen[0].tolist() == [False, True]


def test_reference_rows_use_their_own_label_drop_prob():
    m = model(ref_label_drop_prob=1, action_label_cfg_drop_prob=0).train()
    ref_active = torch.tensor([True, False, True, False])
    active = m._resolve_action_label_active(None, 4, 'cpu', ref_active)
    assert active.tolist() == [False, True, False, True]
    m = model(ref_label_drop_prob=0, action_label_cfg_drop_prob=1).train()
    assert m._resolve_action_label_active(None, 4, 'cpu', ref_active).tolist() == [True, False, True, False]
    assert m.eval()._resolve_action_label_active(None, 4, 'cpu', ref_active).all()


def test_folded_reference_read_matches_explicit_attention():
    m = model(ref_cfg_drop_prob=0).eval()
    layer = m.seqTransDecoder.layers[0]
    torch.manual_seed(3)
    with torch.no_grad():
        for p in (layer.ref_key_bias.weight, layer.ref_out_bias):
            p.normal_()
        layer.ref_attn_gate.fill_(1.)
    frames, batch, joints, width, slots, heads = 6, 2, 3, 16, 2, 4
    x = torch.randn(frames, batch, joints, width)
    z = torch.randn(batch, frames - 1, slots, 8)
    active = torch.tensor([True, False])
    out = layer._read_reference(x, z, active)
    xn = layer.ref_norm(x[1:, 0])  # [T, J, D], active row only
    zt = z[0]  # [T, K, d_z]
    keys = layer.ref_key(zt).view(frames - 1, slots, heads, width)
    values = layer.ref_value(zt).view(frames - 1, slots, heads, width)
    scores = torch.einsum('tjd,tkhd->tjhk', xn, keys) * width ** -.5
    scores = scores + layer.ref_key_bias(zt).permute(0, 2, 1)[:, None]
    read = torch.einsum('tjhk,tkhd->tjd', scores.softmax(-1), values) + layer.ref_out_bias
    torch.testing.assert_close(out[1:, 0], x[1:, 0] + read, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out[0], x[0], rtol=0, atol=0)
    torch.testing.assert_close(out[:, 1], x[:, 1], rtol=0, atol=0)


def test_unaugmented_reference_reuses_shared_target_channels():
    motion, cond, metadata = fixture()
    fresh = prepare_reference_sample(motion, cond, metadata, SimpleNamespace(), augment=False)
    shared = {k: fresh[k] for k in ('feats', 'rest_pose', 'joint_struct')}
    reused = prepare_reference_sample(motion, cond, metadata, SimpleNamespace(), augment=False,
                                      shared=shared)
    assert all(reused[k] is shared[k] for k in shared)
    np.testing.assert_array_equal(reused['bone_dir'], fresh['bone_dir'])
    options = SimpleNamespace(ref_bone_length_aug_prob=1, ref_bone_length_aug=.2)
    random.seed(5)
    augmented = prepare_reference_sample(motion, cond, metadata, options, shared=shared)
    assert all(augmented[k] is not shared[k] for k in shared)


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


@pytest.mark.parametrize('use_text', [False, True])
@pytest.mark.parametrize('ref_scale', [0., 1., 2., 0.5])
@pytest.mark.parametrize('text_scale', [0., 1., 2., 3.])
def test_reference_cfg_skips_zero_weight_branches(use_text, ref_scale, text_scale):
    inner = Echo()
    wrapper = ClassifierFreeReferenceModel(inner, ref_scale, text_scale, use_text)
    x = torch.zeros(2, 1, 1, 1)
    species = torch.tensor([False, True])
    y = dict(ref_active=torch.ones(2, dtype=torch.bool), species_active=species)
    out = wrapper(x, torch.zeros(2), y)
    effective_text_scale = text_scale if use_text else 0.
    base = 1 + species.float() * 7
    expected = base + ref_scale * 2 + effective_text_scale * 3
    torch.testing.assert_close(out, expected[:, None, None, None])
    expected_branches = []
    if ref_scale != 1.:
        expected_branches.append((False, False))
    if ref_scale != effective_text_scale:
        expected_branches.append((True, False))
    if effective_text_scale != 0.:
        expected_branches.append((True, True))
    assert [(bool(ref.all()), bool(text.all())) for ref, text, _ in inner.calls] == expected_branches
    assert all(call[2] is species for call in inner.calls)
    assert 'action_label_active' not in y
    assert y['ref_active'].all()


@pytest.mark.parametrize('use_text', [False, True])
def test_reference_cfg_ignores_species_scale_only_when_disabled(use_text):
    from sample.conditioning import _wrap_reference_cfg
    inner = Echo()
    inner.ref_encoder = True
    inner.ref_cfg_drop_prob = .4
    inner.ref_label_drop_prob = .5
    args = SimpleNamespace(species_active=False, species_cfg_scale=2.,
                           ref_cfg_scale=2., action_label_cfg_scale=3.)
    action = {'action_label': 'walk'} if use_text else None
    wrapper = _wrap_reference_cfg(inner, args, action)
    species = torch.zeros(2, dtype=torch.bool)
    wrapper(torch.zeros(2, 1, 1, 1), torch.zeros(2),
            y=dict(ref_active=torch.ones(2, dtype=torch.bool), species_active=species,
                   action_label_active=torch.ones(2, dtype=torch.bool)))
    assert len(inner.calls) == (3 if use_text else 2)
    assert all(c[2] is species for c in inner.calls)
    args.species_active = True
    with pytest.raises(ValueError, match='species_cfg_scale must be 1'):
        _wrap_reference_cfg(inner, args, action)


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
    # The fixture is static, which cannot show whether inference derives the
    # encoder channels before or after the window resample. Move one joint so
    # the bone directions vary over time.
    moving = motion.copy()
    moving[:, 3, 0] += np.linspace(0.0, 1.0, len(moving), dtype=np.float32)
    packed = _prepare_encoder_reference_bundle(moving, cond, batch_size=2,
                                               n_frames=20, visible_frames=30)
    assert packed['ref_feats'].shape == (2, 7, 12, 20)
    assert packed['ref_bone_dir'].shape == (2, 7, 3, 21)
    # Loader and inference must agree on every row, not just the rest row:
    # both derive them from the same post-resample physical features.
    expected = reference_sample(resample_motion_features(moving, 20), cond,
                                cond['joints_names_embs'])
    assert packed['ref_bone_dir'][0].numpy() == pytest.approx(
        expected['bone_dir'].transpose(1, 2, 0), abs=1e-6)
    assert packed['ref_feats'][0].numpy() == pytest.approx(
        expected['feats'].transpose(1, 2, 0), abs=1e-6)
    assert packed['ref_active'].all()
    with pytest.raises(ValueError, match='source skeleton'):
        _prepare_encoder_reference_bundle(moving[:, :-1], cond, batch_size=2,
                                          n_frames=20, visible_frames=30)


@pytest.mark.parametrize('kwargs', [dict(ref_dim=0), dict(ref_slots=0),
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
