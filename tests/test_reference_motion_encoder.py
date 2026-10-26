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
    REST_GEOM_DIM,
    REFERENCE_POSE_CHANNELS, bone_directions, collate_reference, prepare_reference_sample,
    reference_pose, reference_sample,
)
from data_loaders.truebones.data.dataset import resample_motion_features
from data_loaders.truebones.truebones_utils.canonical_features import canonical_to_physical_hml, rest_pos_from_cond
from data_loaders.truebones.truebones_utils.dataset_tags import species_code_dim
from data_loaders.truebones.truebones_utils.joint_struct_features import JOINT_STRUCT_DIM
from sample.reference_motion import _prepare_encoder_reference_bundle
from test_bone_length_aug import fixture, make_dataset, pin_target_leaf_drop


def model(**kwargs):
    return AnyTop(8, 12, latent_dim=16, ff_size=32, num_layers=2, num_heads=4,
                  t5_out_dim=4, dropout=0, cross_limb=True, ref_encoder=True,
                  ref_width=8, **kwargs)


def inputs(batch=2, joints=4, frames=5, source_joints=6):
    x = torch.randn(batch, joints, 12, frames)
    y = dict(n_joints=torch.full((batch,), joints), rest_geom=torch.randn(batch, joints, REST_GEOM_DIM),
             joints_names_embs=torch.randn(batch, joints, 4),
             joint_struct=torch.randn(batch, joints, JOINT_STRUCT_DIM),
             joints_padding_mask=torch.ones(batch, 1, 1, joints+1, joints+1),
             graph_dist=torch.zeros(batch, joints, joints),
             joints_relations=torch.zeros(batch, joints, joints))
    pose = torch.randn(batch, source_joints, REFERENCE_POSE_CHANNELS, frames)
    pose[:, :, :3] = torch.nn.functional.normalize(pose[:, :, :3], dim=2)
    y.update(ref_feats=torch.randn(batch, source_joints, 12, frames),
             ref_pose=pose,
             ref_rest_pose=pose[..., 0].clone(),
             ref_mask=torch.ones(batch, source_joints, dtype=torch.bool),
             ref_active=torch.ones(batch, dtype=torch.bool),
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
    other = dict(y, ref_pose=torch.randn_like(y['ref_pose']) * 5)
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
    for k in ('ref_feats', 'ref_pose', 'ref_rest_pose', 'ref_mask', 'ref_joints_names_embs',
              'ref_joint_struct'):
        permuted[k] = y[k][:, permutation]
    torch.testing.assert_close(z, encoder(permuted), rtol=1e-5, atol=1e-6)
    # Only the root's world XZ of the canonical window is read: the other
    # joints' positions, every rotation and the root's height never are.
    changed = copy.deepcopy(y)
    changed['ref_feats'][:, 1:] = 100
    changed['ref_feats'][:, 0, 1] = 100
    changed['ref_feats'][:, 0, 3:] = 100
    changed['species_emb'] = torch.randn(2, 99)
    torch.testing.assert_close(z, encoder(changed), rtol=0, atol=0)
    moved = copy.deepcopy(y)
    moved['ref_feats'][:, 0, 0] += 1
    assert not torch.allclose(z, encoder(moved))
    padded = dict(y)
    for k in ('ref_feats', 'ref_pose', 'ref_rest_pose', 'ref_mask', 'ref_joints_names_embs',
              'ref_joint_struct'):
        shape = list(y[k].shape)
        shape[1] = 3
        padded[k] = torch.cat([y[k], y[k].new_zeros(shape)], dim=1)
    torch.testing.assert_close(z, encoder(padded), rtol=1e-5, atol=1e-6)


def test_encoder_reads_bone_directions_and_height():
    m = model().eval()
    _, y = inputs()
    z = m.motion_encoder(y)
    for channels in (slice(0, 3), slice(3, 4)):
        changed = copy.deepcopy(y)
        changed['ref_pose'][:, :, channels] = -changed['ref_pose'][:, :, channels]
        assert not torch.allclose(z, m.motion_encoder(changed))


def test_encoder_reads_rest_pose_and_keeps_padding_inert():
    m = model().eval()
    _, y = inputs()
    z = m.motion_encoder(y)
    changed = copy.deepcopy(y)
    changed['ref_rest_pose'][:, 2] = -changed['ref_rest_pose'][:, 2]
    assert not torch.allclose(z, m.motion_encoder(changed))
    padded = dict(y)
    for k in ('ref_feats', 'ref_pose', 'ref_rest_pose', 'ref_mask', 'ref_joints_names_embs',
              'ref_joint_struct'):
        shape = list(y[k].shape)
        shape[1] = 3
        padded[k] = torch.cat([y[k], y[k].new_zeros(shape)], dim=1)
    torch.testing.assert_close(z, m.motion_encoder(padded), rtol=1e-5, atol=1e-6)


def test_reference_rest_pose_is_the_pose_of_the_rest():
    motion, cond, _ = fixture()
    rest = rest_pos_from_cond(cond)
    sample = reference_sample(motion, cond, cond['joints_names_embs'])
    assert sample['rest_pose'].shape == (len(cond['parents']), REFERENCE_POSE_CHANNELS)
    np.testing.assert_allclose(sample['rest_pose'], reference_pose(rest[None], cond)[0])


def test_bone_directions_are_facing_frame_units():
    parents = np.array([-1, 0, 1, 1])
    rest = np.array([[0, 1, 0], [0, 1, 0], [0, 1, -1], [0, 1.0005, -1]], np.float32)
    positions = np.repeat(rest[None], 2, axis=0)
    positions[1, 2] = [0, 3, 0]  # second frame: tail raised straight up
    out = bone_directions(positions, parents, rest)
    assert out.shape == (2, 4, 3)
    np.testing.assert_array_equal(out[:, 0], 0)  # root
    np.testing.assert_array_equal(out[:, 1], 0)  # coincident with its parent
    np.testing.assert_allclose(out[0, 2], [0, 0, -1])
    np.testing.assert_allclose(out[1, 2], [0, 1, 0])
    np.testing.assert_allclose(np.linalg.norm(out[:, 3], axis=-1), 1, rtol=1e-6)


def test_reference_pose_is_rest_free():
    # Same world pose on two rigs with equal L but different rest geometry:
    # the encoder pose must not tell them apart.
    motion, cond, _ = fixture()
    other = copy.deepcopy(cond)
    rest = np.asarray(cond['rest_pos_ric_hml'], np.float32).copy()
    rest[:, 0] = -rest[:, 0]  # mirrored rest: same spread, different geometry
    other['rest_pos_ric_hml'] = rest
    np.testing.assert_allclose(reference_pose(motion, cond), reference_pose(motion, other), atol=1e-6)


def test_bone_directions_sanitize_non_finite_positions():
    parents = np.array([-1, 0])
    rest = np.array([[0, 1, 0], [0, 2, 0]], np.float32)
    positions = np.array([[[0, 1, 0], [np.inf, 2, 0]],
                          [[0, 1, 0], [np.nan, 2, 0]]], np.float32)
    out = bone_directions(positions, parents, rest)
    assert out.shape == (2, 2, 3)
    assert np.isfinite(out).all()


def test_collate_rejects_bone_direction_frame_mismatch():
    motion, cond, _ = fixture()
    sample = prepare_reference_sample(motion, cond, {'translation_root_index': 0},
                                      SimpleNamespace(), augment=False)
    sample['pose'] = sample['pose'][:-1]
    with pytest.raises(ValueError, match='Reference pose'):
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
    m = model(ref_cfg_drop_prob=0, species_cond=True).eval()
    x, y = inputs()
    y['ref_active'][1] = False
    y['species_code'] = torch.randn(2, species_code_dim(4))
    seen = []
    forward = m.input_process.forward
    monkeypatch.setattr(m.input_process, 'forward',
                        lambda *a, **k: seen.append(k['species_active'].clone()) or forward(*a, **k))
    m(x, torch.zeros(2), y)
    assert seen[0].tolist() == [False, True]


@pytest.mark.parametrize('training', [True, False])
def test_reference_rows_bypass_the_action_label(monkeypatch, training):
    m = model(action_label_cond=True, action_label_cfg_drop_prob=0).train(training)
    ref_active = torch.tensor([True, False, True, False])
    monkeypatch.setattr(m, '_action_slot_inputs', lambda *a: None)
    monkeypatch.setattr(m, '_resolve_action_label_valid', lambda *a: torch.ones(4, dtype=torch.bool))
    y = {'action_label_active': torch.ones(4, dtype=torch.bool)}
    _, active = m._action_condition(y, 4, 'cpu', torch.float32, ref_active)
    assert active.tolist() == [False, True, False, True]


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
    z = torch.randn(batch, frames, slots, 8)
    active = torch.tensor([True, False])
    out = layer._read_reference(x, z, active)
    xn = layer.ref_norm(x[:, 0])  # [T, J, D], active row only
    zt = z[0]  # [T, K, d_z]
    keys = layer.ref_key(zt).view(frames, slots, heads, width)
    values = layer.ref_value(zt).view(frames, slots, heads, width)
    scores = torch.einsum('tjd,tkhd->tjhk', xn, keys) * width ** -.5
    scores = scores + layer.ref_key_bias(zt).permute(0, 2, 1)[:, None]
    read = torch.einsum('tjhk,tkhd->tjd', scores.softmax(-1), values) + layer.ref_out_bias
    torch.testing.assert_close(out[:, 0], x[:, 0] + read, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out[:, 1], x[:, 1], rtol=0, atol=0)


def test_unaugmented_reference_reuses_shared_target_channels():
    motion, cond, metadata = fixture()
    fresh = prepare_reference_sample(motion, cond, metadata, SimpleNamespace(), augment=False)
    shared = {k: fresh[k] for k in ('feats', 'joint_struct')}
    reused = prepare_reference_sample(motion, cond, metadata, SimpleNamespace(), augment=False,
                                      shared=shared)
    assert all(reused[k] is shared[k] for k in shared)
    np.testing.assert_array_equal(reused['pose'], fresh['pose'])
    options = SimpleNamespace(bone_length_aug_prob=1, bone_length_aug_max=.2)
    random.seed(5)
    augmented = prepare_reference_sample(motion, cond, metadata, options, shared=shared)
    # New lengths make new features; the structural channel names the same joints.
    assert augmented['feats'] is not shared['feats']
    assert augmented['joint_struct'] is shared['joint_struct']


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
        self.calls.append((ref.clone(), text.clone()))
        # Like AnyTop, a reference-active row ignores the action label and the
        # species FiLM; every other row applies the species FiLM.
        species = 1 - ref
        return x * 0 + (1 + ref * 2 + text * (1 - ref) * 3 + species * 7)[:, None, None, None]


def test_cfg_algebra_sequential_pass_count():
    inner = Echo()
    wrapper = ClassifierFreeReferenceModel(inner, 2, num_timesteps=100)
    x = torch.zeros(2, 4, 12, 5)
    y = dict(ref_active=torch.ones(2, dtype=torch.bool))
    out = wrapper(x, torch.full((2,), 99), y)
    # base 8 (no label, species on) + 2 * (ref 3 - base 8).
    torch.testing.assert_close(out, torch.full_like(out, -2.))
    assert len(inner.calls) == 2
    assert not inner.calls[0][0].any() and not inner.calls[0][1].any()
    assert inner.calls[1][0].all()
    assert 'action_label_active' not in y


@pytest.mark.parametrize('ref_scale', [0., 1., 2., 0.5])
def test_reference_cfg_skips_zero_weight_branches(ref_scale):
    inner = Echo()
    wrapper = ClassifierFreeReferenceModel(inner, ref_scale, num_timesteps=100)
    x = torch.zeros(2, 1, 1, 1)
    y = dict(ref_active=torch.ones(2, dtype=torch.bool),
             action_label_active=torch.zeros(2, dtype=torch.bool))
    out = wrapper(x, torch.full((2,), 99), y)
    torch.testing.assert_close(out, torch.full_like(out, (1 - ref_scale) * 8 + ref_scale * 3))
    expected_branches = []
    if ref_scale != 1.:
        expected_branches.append(False)
    if ref_scale != 0.:
        expected_branches.append(True)
    assert [bool(ref.all()) for ref, _ in inner.calls] == expected_branches
    assert not y['action_label_active'].any()
    assert y['ref_active'].all()


def test_reference_cfg_schedule_switches_on_noise_level():
    inner = Echo()
    wrapper = ClassifierFreeReferenceModel(inner, 6., num_timesteps=100)
    x = torch.zeros(2, 1, 1, 1)
    y = dict(ref_active=torch.ones(2, dtype=torch.bool))
    # Early (t >= 50): base + 6 * (ref - base) = 8 - 30; two forwards.
    torch.testing.assert_close(wrapper(x, torch.full((2,), 50), y), torch.full_like(x, -22.))
    assert len(inner.calls) == 2
    # Late (t < 50): scale 1 is the reference branch alone; one forward.
    inner.calls.clear()
    torch.testing.assert_close(wrapper(x, torch.full((2,), 49), y), torch.full_like(x, 3.))
    assert len(inner.calls) == 1
    # Rows on both sides of the switch get their own scale.
    out = wrapper(x, torch.tensor([99, 0]), y)
    torch.testing.assert_close(out.flatten(), torch.tensor([-22., 3.]))


def test_reference_cfg_schedule_validation():
    with pytest.raises(ValueError, match='num_timesteps'):
        ClassifierFreeReferenceModel(Echo(), 6., num_timesteps=0)


def test_wrap_reference_cfg_rejects_action_label():
    from sample.conditioning import _wrap_reference_cfg
    inner = Echo()
    inner.ref_encoder = True
    inner.ref_cfg_drop_prob = .4
    args = SimpleNamespace(ref_cfg_scale=2., diffusion_steps=100)
    wrapper = _wrap_reference_cfg(inner, args, None)
    wrapper(torch.zeros(2, 1, 1, 1), torch.full((2,), 99),
            y=dict(ref_active=torch.ones(2, dtype=torch.bool)))
    assert len(inner.calls) == 2
    with pytest.raises(ValueError, match='action_label'):
        _wrap_reference_cfg(inner, args, {'action_label': 'walk'})


def test_augmentation_has_independent_cond_and_target_is_untouched():
    motion, cond, metadata = fixture()
    before_motion, before_cond = motion.copy(), copy.deepcopy(cond)
    options = SimpleNamespace(bone_length_aug_prob=1, bone_length_aug_max=.2,
                              joint_name_drop_prob=1, topology_aug_prob=0)
    cond['joints_names_embs'][:] = 1
    random.seed(5)
    sample = prepare_reference_sample(motion, cond, metadata, options)
    np.testing.assert_array_equal(motion, before_motion)
    np.testing.assert_array_equal(cond['rest_pose'], before_cond['rest_pose'])
    assert np.all(cond['joints_names_embs'] == 1)
    assert not sample['joints_names_embs'].any()
    # This fixture's draw scales its one chain, i.e. the whole rig: invisible
    # to the encoder pose, which is rest-free and normalized by L.
    unaugmented = reference_pose(motion, before_cond)
    np.testing.assert_allclose(sample['pose'], unaugmented, atol=1e-5)
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
    assert packed['ref_pose'].shape == (2, 7, REFERENCE_POSE_CHANNELS, 20)
    # Loader and inference must agree on every row: both derive them from the
    # same post-resample physical features.
    expected = reference_sample(resample_motion_features(moving, 20), cond,
                                cond['joints_names_embs'])
    assert packed['ref_pose'][0].numpy() == pytest.approx(
        expected['pose'].transpose(1, 2, 0), abs=1e-6)
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


def test_target_augmentation_leaves_reference_on_stored_rig(monkeypatch):
    from data_loaders.tensors import truebones_batch_collate
    dataset, data = make_dataset()
    dataset.opt.bone_length_aug_prob = 0
    dataset.opt.ref_encoder = True
    dataset.opt.topology_aug_prob = 0
    dataset.opt.bone_length_aug_prob = 0
    dataset.opt.joint_name_drop_prob = 0
    random.seed(2)
    plain = dataset._prepare_sample('test', data)
    stored_joints = plain[0].shape[1]
    dataset.opt.topology_aug_prob = 1
    pin_target_leaf_drop(monkeypatch, [3])
    sample = dataset._prepare_sample('test', data)
    reference = sample[12]['reference']
    assert sample[0].shape[1] == stored_joints - 1
    assert reference['feats'].shape == (sample[0].shape[0], stored_joints, sample[0].shape[2])
    # The reference is wider than its target: the bucket must hold it.
    _, kwargs = truebones_batch_collate([sample], joint_buckets=(stored_joints - 1, stored_joints))
    assert kwargs['y']['ref_mask'].shape[1] == stored_joints
    assert kwargs['y']['ref_mask'].sum() == stored_joints


def test_reference_skeleton_is_augmented_on_the_whole_stored_clip(monkeypatch):
    import data_loaders.reference as reference_module
    dataset, data = make_dataset()
    dataset.opt.ref_encoder = True
    dataset.opt.topology_aug_prob = 0
    dataset.opt.bone_length_aug_prob = 1
    dataset.opt.joint_name_drop_prob = 0
    stored = dataset._load_physical_motion(data)[0]
    seen = []
    augment = reference_module.augment_reference_skeleton

    def record(motion, *args, **kwargs):
        seen.append(np.array(motion, copy=True))
        return augment(motion, *args, **kwargs)

    monkeypatch.setattr(reference_module, 'augment_reference_skeleton', record)
    random.seed(3)
    sample = dataset._prepare_sample('test', data)
    # Grounded on the clip the target is grounded on, not on its window.
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], stored)
    reference = sample[12]['reference']
    assert reference['feats'].shape[0] == sample[0].shape[0]


def test_target_augmentation_keeps_loop_reference_time_aligned(monkeypatch):
    dataset, data = make_dataset()
    dataset.opt.bone_length_aug_prob = 0
    dataset.opt.ref_encoder = True
    dataset.opt.topology_aug_prob = 0
    dataset.opt.bone_length_aug_prob = 0
    dataset.opt.joint_name_drop_prob = 0
    dataset.opt.topology_aug_prob = 1
    data['motion_metadata']['is_loop'] = True
    pin_target_leaf_drop(monkeypatch, [3])
    random.seed(4)
    sample = dataset._prepare_sample('test', data)
    reference = sample[12]['reference']
    assert reference['feats'].shape[0] == sample[0].shape[0]
    assert reference['feats'].shape[1] == sample[0].shape[1] + 1
    kept = [j for j in range(reference['feats'].shape[1]) if j != 3]
    # Compared as physical positions: dropping the leaf changes the rig's L and
    # its parent's rest bone frame, so the canonical rows differ by design.
    full = canonical_to_physical_hml(reference['feats'], dataset.cond_dict['test'])
    dropped = canonical_to_physical_hml(sample[0], sample[12])
    np.testing.assert_allclose(full[:, kept, 0:3], dropped[..., 0:3], atol=1e-4)


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


def test_hidden_frames_never_reach_any_z():
    m = model().eval()
    _, y = inputs(frames=6)
    encoder = m.motion_encoder
    with torch.no_grad():
        encoder.hidden_frame_embedding.normal_()
    hidden = torch.zeros(2, 6, dtype=torch.bool)
    hidden[0, 2:4] = True
    y['ref_frame_mask'] = hidden
    z = encoder(y)
    # Whatever the hidden frames hold, no frame's z -- hidden ones included --
    # sees it: their inputs are zeroed and temporal attention never reads them.
    changed = copy.deepcopy(y)
    changed['ref_pose'][0, :, :, 2:4] = 50
    changed['ref_feats'][0, 0, :, 2:4] = 50
    torch.testing.assert_close(z, encoder(changed), rtol=0, atol=0)
    # The unmasked row is untouched by another row's mask.
    torch.testing.assert_close(z[1], encoder({**y, 'ref_frame_mask': torch.zeros_like(hidden)})[1],
                               rtol=1e-5, atol=1e-6)
    assert not torch.allclose(z[0], encoder(changed | {'ref_frame_mask': torch.zeros_like(hidden)})[0])
    # A row with every frame hidden keeps its frames.
    all_hidden = torch.ones_like(hidden)
    torch.testing.assert_close(encoder({**y, 'ref_frame_mask': all_hidden}),
                               encoder({**y, 'ref_frame_mask': torch.zeros_like(hidden)}),
                               rtol=1e-5, atol=1e-6)


def test_collate_carries_frame_mask_and_same_rig():
    motion, cond, _ = fixture()
    sample = prepare_reference_sample(motion[:30], cond, {'translation_root_index': 0},
                                      SimpleNamespace(), augment=False)
    joints = len(cond['parents'])
    own = dict(sample, same_rig=True)
    out = collate_reference([own, sample, None], torch.zeros(3, joints, 12, 30),
                            torch.zeros(3, joints, 4), torch.zeros(3, joints, JOINT_STRUCT_DIM))
    assert out['ref_frame_mask'].shape == (3, 30) and not out['ref_frame_mask'].any()
    assert out['ref_same_rig'].tolist() == [True, False, False]


def test_training_hides_corrupted_cells_from_the_reference():
    from diffusion.gaussian_diffusion import GaussianDiffusion
    y = dict(ref_frame_mask=torch.zeros(2, 6, dtype=torch.bool),
             ref_mask=torch.ones(2, 4, dtype=torch.bool),
             ref_same_rig=torch.tensor([True, False]))
    span = torch.zeros(2, 4, 6, dtype=torch.bool)
    span[:, :3, 1:3] = True          # real joints share the span; padding stays False
    subtree = torch.zeros(2, 4, dtype=torch.bool)
    subtree[:, 2] = True
    GaussianDiffusion._hide_corruption_from_reference(y, subtree, span)
    assert y['ref_frame_mask'].tolist() == [[False, True, True, False, False, False]] * 2
    # Joints only where the reference is the target's own rig.
    assert y['ref_mask'].tolist() == [[True, True, False, True], [True, True, True, True]]
    untouched = dict(ref_mask=torch.ones(1, 4, dtype=torch.bool))
    GaussianDiffusion._hide_corruption_from_reference(untouched, subtree[:1], span[:1])
    assert set(untouched) == {'ref_mask'}


def test_generation_hides_the_inpaint_region_from_the_reference():
    from sample.reference_motion import _hide_inpaint_region_from_reference
    reference = dict(ref_frame_mask=torch.zeros(2, 6, dtype=torch.bool),
                     ref_mask=torch.ones(2, 4, dtype=torch.bool))
    mask = torch.zeros(2, 4, 1, 6)
    mask[:, 1:3, 0, 2:5] = 1
    _hide_inpaint_region_from_reference(reference, mask, frames_given=True, free_joints=[1, 2])
    assert reference['ref_frame_mask'][0].tolist() == [False, False, True, True, True, False]
    assert reference['ref_mask'].all()
    joints_only = dict(ref_frame_mask=torch.zeros(2, 6, dtype=torch.bool),
                       ref_mask=torch.ones(2, 4, dtype=torch.bool))
    _hide_inpaint_region_from_reference(joints_only, mask, frames_given=False, free_joints=[1, 2])
    assert not joints_only['ref_frame_mask'].any()
    assert joints_only['ref_mask'][0].tolist() == [True, False, False, True]
