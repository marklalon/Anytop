"""Physical consistency and unrestricted action/skeleton augmentation coverage."""
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_loaders.truebones.truebones_utils.bone_length_aug import (
    augment_bone_lengths, body_groups, validate_bone_length_aug,
)
from data_loaders.truebones.truebones_utils.canonical_features import (
    build_canonical_rest_feature, canonical_to_physical_hml,
)
from data_loaders.truebones.truebones_utils.joint_struct_features import build_joint_struct_features
from data_loaders.truebones.truebones_utils.features import recover_from_bvh_ric_np
from data_loaders.truebones.data.dataset import MotionDataset
from tools.sample_augmented_bvh import PreviewMotionDataset
from utils.retarget_core import batch_forward_kinematics_np


def fixture():
    parents = np.array([-1, 0, 1, 2, 0, 4, 5])
    offsets = np.array([[0, 0, 0], [.2, 0, 0], [0, -.5, 0], [0, -.5, 0],
                        [-.2, 0, 0], [0, -.5, 0], [0, -.5, 0]], dtype=np.float32)
    rest = offsets.copy()
    rest[0, 1] = 1
    for j in range(1, len(parents)):
        rest[j] += rest[parents[j]]
    rest_feature = np.zeros((7, 12), np.float32)
    rest_feature[:, :3] = rest
    rest_feature[:, 3] = rest_feature[:, 7] = 1
    cond = dict(parents=parents, offsets=offsets, rest_pos_ric_hml=rest,
                rest_pose=rest_feature,
                joint_side_labels=['center', 'left', 'left', 'left', 'right', 'right', 'right'],
                canonical_joint_names=['Hips', 'Left Thigh', 'Left Knee', 'Left Foot',
                                       'Right Thigh', 'Right Knee', 'Right Foot'],
                symmetry_partner_indices=[-1, 4, 5, 6, 1, 2, 3], translation_root_index=0,
                canonical_feature_mean=np.zeros(12), canonical_feature_std=np.ones(12),
                joint_mask_candidate_roots=np.zeros(7), joints_names_embs=np.zeros((7, 4)),
                joints_graph_dist=np.zeros((7, 7)), joint_relations=np.zeros((7, 7)),
                kinematic_chains=[])
    motion = np.broadcast_to(rest_feature, (30, 7, 12)).copy()
    metadata = dict(action_label='walk', translation_root_index=0, is_loop=False)
    return motion, cond, metadata


@pytest.mark.parametrize('prob, magnitude', [(-.1, .1), (1.1, .1), (np.nan, .1),
                                            (.2, np.inf), (.2, -.1), (.2, 1)])
def test_invalid_configuration(prob, magnitude):
    with pytest.raises(ValueError):
        validate_bone_length_aug(prob, magnitude)


def test_symmetric_lengths_grounding_and_unmutated_source():
    motion, cond, metadata = fixture()
    saved_motion, saved_rest = motion.copy(), cond['rest_pose'].copy()
    out, new_cond, info = augment_bone_lengths(motion, cond, metadata, rng=random.Random(3))
    assert info['applied'], info
    factor = info['scales']['chain_1']
    assert .9 <= factor <= 1.1 and factor != 1
    # Topology groups: each leg is one branch-free chain, the mirror pair one group.
    assert body_groups(cond) == {'chain_1': [1, 2, 3, 4, 5, 6]}
    np.testing.assert_allclose(new_cond['offsets'][1:], cond['offsets'][1:] * factor)
    np.testing.assert_array_equal(new_cond['offsets'][0], cond['offsets'][0])
    # The clip's lowest point (the feet) keeps its height.
    np.testing.assert_allclose(out[:, [3, 6], 1], 0, atol=1e-6)
    np.testing.assert_array_equal(motion, saved_motion)
    np.testing.assert_array_equal(cond['rest_pose'], saved_rest)
    assert new_cond['parents'] is cond['parents']
    # Scaling retains the exact rigid bone lengths and source rotations.
    np.testing.assert_array_equal(out[..., 3:9], motion[..., 3:9])
    parents = cond['parents'][1:]
    lengths = np.linalg.norm(out[:, 1:, :3] - out[:, parents, :3], axis=-1)
    np.testing.assert_allclose(lengths, np.broadcast_to(
        np.linalg.norm(new_cond['offsets'][1:], axis=-1), lengths.shape), atol=1e-6)


@pytest.mark.parametrize('label', ['attack', 'idle, eat', 'walk, carry', 'die', 'jump', '', 'unknown'])
def test_all_actions_and_missing_labels_are_augmented(label):
    motion, cond, metadata = fixture()
    metadata['action_label'] = label
    out, new_cond, info = augment_bone_lengths(motion, cond, metadata)
    assert info['applied'] and not info['skip_reason']
    assert out is not motion and new_cond is not cond
    assert not np.allclose(new_cond['offsets'], cond['offsets'])


def test_velocity_and_authored_deformation_with_rotating_parent():
    motion, cond, metadata = fixture()
    cond['canonical_joint_names'] = ['Hips', 'Tail 1', 'Tail 2', 'Tail 3', '', '', '']
    cond['joint_side_labels'] = ['center'] * 7
    cond.pop('symmetry_partner_indices')
    # Zero-length bones are never stretched: the tail chain is the only group.
    cond['offsets'][4:] = 0
    rotations = np.zeros((30, 7, 4))
    rotations[..., 0] = 1
    angle = np.linspace(0, .5, 30)
    rotations[:, 1, 0] = np.cos(angle / 2)
    rotations[:, 1, 3] = np.sin(angle / 2)
    positions = np.broadcast_to(cond['offsets'], (30, 7, 3)).copy()
    positions[:, 0, 1] = 1
    positions[:, 0, 0] = np.arange(30) * .01
    positions[:, 2, 1] *= np.linspace(1, 1.05, 30)
    world, _ = batch_forward_kinematics_np(rotations, positions, cond['parents'])
    from motion_lib.Quaternions import Quaternions
    motion[..., :3] = world
    motion[..., 0] -= world[:, :1, 0]
    motion[..., 3:9] = Quaternions(rotations).rotation_matrix(cont6d=True)
    motion[:-1, :, 9:12] = np.diff(world, axis=0)
    motion[-1, :, 9:12] = motion[-2, :, 9:12]
    out, new_cond, info = augment_bone_lengths(motion, cond, metadata, rng=random.Random(5))
    assert info['applied']
    actual = recover_from_bvh_ric_np(out, translation_root_index=0)
    np.testing.assert_allclose(np.diff(actual, axis=0), out[:-1, :, 9:12], atol=2e-7)
    factor = info['scales']['chain_1']
    scaled_positions = positions.copy()
    scaled_positions[:, [1, 2, 3]] *= factor
    expected, _ = batch_forward_kinematics_np(rotations, scaled_positions, cond['parents'])
    expected[..., 1] += world[..., 1].min() - expected[..., 1].min()
    np.testing.assert_allclose(actual, expected, atol=2e-7)
    for j in [1, 2, 3]:
        p = cond['parents'][j]
        np.testing.assert_allclose(np.linalg.norm(actual[:, j]-actual[:, p], axis=-1),
                                   np.linalg.norm(world[:, j]-world[:, p], axis=-1)*factor, atol=2e-7)
    np.testing.assert_allclose(out[..., 3:9], motion[..., 3:9], atol=2e-7)


@pytest.mark.parametrize('periodic', [False, True])
def test_nonroot_translation_carrier_and_separate_roots(periodic):
    motion, cond, metadata = fixture()
    cond['parents'][4] = -1
    cond['canonical_joint_names'] = ['Hips', 'Tail 1', 'Tail 2', 'Tail 3', '', '', '']
    cond['joint_side_labels'] = ['center'] * 7
    cond.pop('symmetry_partner_indices')
    cond['translation_root_index'] = metadata['translation_root_index'] = 2
    metadata['is_loop'] = periodic
    rotations = np.zeros((30, 7, 4))
    rotations[..., 0] = 1
    angle = np.linspace(0, .5, 30)
    rotations[:, 0, 0] = np.cos(angle / 2)
    rotations[:, 0, 3] = np.sin(angle / 2)
    local = np.broadcast_to(cond['offsets'], (30, 7, 3)).copy()
    local[:, 0, 0] = np.arange(30) * .01
    local[:, 0, 1] = 1
    world, _ = batch_forward_kinematics_np(rotations, local, cond['parents'])
    from motion_lib.Quaternions import Quaternions
    motion[..., :3] = world
    motion[..., 0] -= world[:, 2:3, 0]
    motion[..., 2] -= world[:, 2:3, 2]
    motion[..., 3:9] = Quaternions(rotations).rotation_matrix(cont6d=True)
    motion[:-1, :, 9:12] = np.diff(world, axis=0)
    motion[-1, :, 9:12] = world[0] - world[-1] if periodic else motion[-2, :, 9:12]
    out, _, info = augment_bone_lengths(motion, cond, metadata, rng=random.Random(5))
    scaled_local = local.copy()
    groups = body_groups(cond)
    assert groups['chain_1'] == [1, 3]
    for group, factor in info['scales'].items():
        scaled_local[:, groups[group]] *= factor
    expected, _ = batch_forward_kinematics_np(rotations, scaled_local, cond['parents'])
    expected[..., 1] += world[..., 1].min() - expected[..., 1].min()
    # World recovery chooses the first carrier's XZ as its origin.
    expected[..., 0] -= expected[0, 2, 0]
    expected[..., 2] -= expected[0, 2, 2]
    actual = recover_from_bvh_ric_np(out, translation_root_index=2)
    np.testing.assert_allclose(actual, expected, atol=2e-7)
    np.testing.assert_allclose(out[:-1, :, 9:12], np.diff(expected, axis=0), atol=2e-7)
    terminal = expected[0] - expected[-1] if periodic else out[-2, :, 9:12]
    np.testing.assert_allclose(out[-1, :, 9:12], terminal, atol=2e-7)


def test_contact_quality_does_not_reject_sample():
    motion, cond, metadata = fixture()
    # Deliberately floating/penetrating contacts remain eligible.
    motion[:, 3, 1] += 1
    motion[:, 6, 1] -= 1
    out, new_cond, info = augment_bone_lengths(motion, cond, metadata, rng=random.Random(3))
    assert info['applied'] and not info['skip_reason']
    assert out[:, 3, 1].min() > .5 and out[:, 6, 1].max() < -.5
    assert out is not motion and new_cond is not cond


def test_contact_ik_is_never_called_and_rotations_are_unchanged(monkeypatch):
    motion, cond, metadata = fixture()
    # A bent source previously triggered contact IK after length changes.
    motion[:, 3, 0] += .1
    from motion_edit.ik import LimbSolver
    def failed_solve(*args, **kwargs):
        pytest.fail('Bone-length augmentation must not call contact IK')
    monkeypatch.setattr(LimbSolver, 'solve', failed_solve)
    out, new_cond, info = augment_bone_lengths(motion, cond, metadata, rng=random.Random(3))
    assert info['applied'] and np.isfinite(out).all()
    assert info['contact_ik_applied'] is False
    np.testing.assert_array_equal(out[..., 3:9], motion[..., 3:9])
    assert not np.allclose(new_cond['offsets'], cond['offsets'])


def test_unnamed_skeleton_without_contact_limbs_uses_mirror_chains():
    motion, cond, metadata = fixture()
    cond['canonical_joint_names'] = [''] * 7
    cond['joint_side_labels'] = ['center'] * 7
    cond['symmetry_partner_indices'] = [-1, 4, 5, 6, 1, 2, 3]
    groups = body_groups(cond)
    assert groups == {'chain_1': [1, 2, 3, 4, 5, 6]}
    out, new_cond, info = augment_bone_lengths(motion, cond, metadata, rng=random.Random(3))
    assert info['applied'] and info['contact_ik_applied'] is False
    ratios = np.linalg.norm(new_cond['offsets'][1:], axis=-1) / np.linalg.norm(cond['offsets'][1:], axis=-1)
    np.testing.assert_allclose(ratios, np.full(6, ratios[0]))


def test_zero_length_skeleton_is_exportable_with_explicit_note():
    motion, cond, metadata = fixture()
    cond['offsets'][:] = 0
    out, new_cond, info = augment_bone_lengths(motion, cond, metadata)
    assert info['applied'] and info['changed_bone_count'] == 0
    assert 'unchanged' in info['notes'][0]
    np.testing.assert_array_equal(out, motion)


def test_tiny_named_helpers_do_not_hide_real_fallback_bones():
    motion, cond, metadata = fixture()
    cond['canonical_joint_names'] = ['Hips', 'Spine', '', '', '', '', '']
    cond['joint_side_labels'] = ['center'] * 7
    cond['offsets'][1] = [1e-9, 0, 0]
    groups = body_groups(cond)
    assert 'trunk' not in groups
    assert {j for joints in groups.values() for j in joints} == {2, 3, 4, 5, 6}
    # A wholly microscopic rig is still eligible, not rejected.
    cond['offsets'][1:] *= 1e-9
    assert body_groups(cond)


def test_non_parent_first_joint_order_is_supported():
    motion, cond, metadata = fixture()
    order = np.array([0, 3, 6, 2, 5, 1, 4])
    inverse = np.argsort(order)
    parents = cond['parents'][order]
    cond['parents'] = np.array([inverse[p] if p >= 0 else -1 for p in parents])
    for key in ['offsets', 'rest_pos_ric_hml', 'rest_pose']:
        cond[key] = cond[key][order]
    for key in ['canonical_joint_names', 'joint_side_labels']:
        cond[key] = [cond[key][j] for j in order]
    cond['symmetry_partner_indices'] = [inverse[cond['symmetry_partner_indices'][j]] if j else -1
                                        for j in order]
    out, new_cond, info = augment_bone_lengths(motion[:, order], cond, metadata, rng=random.Random(3))
    assert info['applied'] and np.isfinite(out).all()
    np.testing.assert_allclose(out[:, np.flatnonzero(np.isin(order, [3, 6])), 1], 0, atol=1e-6)


def make_dataset(cls=MotionDataset):
    motion, cond, metadata = fixture()
    dataset = cls.__new__(cls)
    dataset.opt = SimpleNamespace(bone_length_aug_prob=1., bone_length_aug=.1,
                                  motion_speed_aug=1., preview_mode='bone-length')
    dataset.max_motion_length = 30
    dataset.min_length = 2
    dataset.opt.max_joints = 7
    dataset.action_conditioning = None
    dataset.cond_dict = {'test': cond}
    dataset._joint_struct_cache = {'test': build_joint_struct_features(cond)}
    dataset._load_physical_motion = lambda data: (
        motion, len(motion), 'test', cond['parents'], cond['joints_graph_dist'],
        cond['joint_relations'], build_canonical_rest_feature(cond), cond['offsets'],
        cond['joints_names_embs'], [])
    return dataset, dict(motion_metadata=metadata)


def test_loader_rebuilds_geometry_and_canonical_roundtrip():
    dataset, data = make_dataset()
    random.seed(3)
    sample = dataset._prepare_sample('test', data, return_aug_info=True)
    assert sample[13]['bone_length_aug']['applied']
    new_cond = sample[13]['object_cond']
    np.testing.assert_allclose(sample[12]['joint_struct'], build_joint_struct_features(new_cond))
    np.testing.assert_allclose(sample[3], build_canonical_rest_feature(new_cond))
    decoded = canonical_to_physical_hml(sample[0], sample[12])
    np.testing.assert_allclose(decoded[:, [3, 6], 1], 0, atol=1e-6)
    assert not np.allclose(sample[4], dataset.cond_dict['test']['offsets'])


def test_preview_retains_source_loop_for_physical_augmentation(monkeypatch):
    dataset, data = make_dataset(PreviewMotionDataset)
    data['motion_metadata']['is_loop'] = True
    captured = {}
    import data_loaders.truebones.data.dataset as module
    real_aug = module.augment_bone_lengths
    def spy(motion, cond, metadata, magnitude):
        captured.update(metadata)
        return real_aug(motion, cond, metadata, magnitude)
    monkeypatch.setattr(module, 'augment_bone_lengths', spy)
    sample = dataset._prepare_sample('test', data, return_aug_info=True)
    assert captured['is_loop']
    assert sample[13]['loop_applied'] is False
    assert data['motion_metadata']['is_loop'] is True


def test_disabled_loader_does_not_draw_rng(monkeypatch):
    dataset, data = make_dataset()
    dataset.opt.bone_length_aug_prob = 0
    monkeypatch.setattr(random, 'random', lambda: pytest.fail('Disabled augmentation drew RNG'))
    sample = dataset._prepare_sample('test', data, return_aug_info=True)
    assert not sample[13]['bone_length_aug']['applied']


def test_loop_closing_pose_and_terminal_velocity_survive():
    motion, cond, metadata = fixture()
    metadata['is_loop'] = True
    # A travelling clip with a repeated closing pose and nonzero root speed.
    motion[:, :, 9] = .01
    out, new_cond, info = augment_bone_lengths(motion, cond, metadata, rng=random.Random(2))
    assert info['applied'], info
    np.testing.assert_allclose(out[0, :, :9], out[-1, :, :9], atol=1e-6)
    np.testing.assert_allclose(out[-1, :, 9:12], motion[-1, :, 9:12], atol=1e-6)


def test_bvh_export_uses_augmented_offsets(tmp_path):
    motion, cond, metadata = fixture()
    out, new_cond, info = augment_bone_lengths(motion, cond, metadata, rng=random.Random(2))
    assert info['applied']
    new_cond.update(joints_names=cond['canonical_joint_names'], scale_factor=1.,
                    orientation_quat=np.array([1., 0., 0., 0.]))
    from utils.npy_restore import write_feature_bvh
    from motion_lib import BVH
    path = tmp_path / 'augmented.bvh'
    write_feature_bvh(out, new_cond, str(path), fps=30.)
    anim, names, frame_time = BVH.load(str(path), collapse_root=False)
    # BVH traversal changes joint ordering: compare using anatomical names.
    for j in range(1, len(names)):
        index = [name.replace(' ', '_') for name in new_cond['joints_names']].index(names[j])
        np.testing.assert_allclose(anim.offsets[j], new_cond['offsets'][index], atol=1e-6)
    assert len(anim.positions) == len(out)
    assert frame_time == pytest.approx(1 / 30, abs=1e-6)


@pytest.mark.parametrize('draw, expected', [(.19, True), (.2, False), (.21, False)])
def test_twenty_percent_probability_gate(monkeypatch, draw, expected):
    dataset, data = make_dataset()
    dataset.opt.bone_length_aug_prob = .2
    import data_loaders.truebones.data.dataset as module
    called = []
    def spy(motion, cond, metadata, magnitude):
        called.append(magnitude)
        return motion, cond, dict(applied=False, scales={}, skip_reason='test')
    monkeypatch.setattr(module, 'augment_bone_lengths', spy)
    monkeypatch.setattr(random, 'random', lambda: draw)
    dataset._prepare_sample('test', data)
    assert bool(called) == expected
    if expected:
        assert called == [.1]


@pytest.mark.parametrize('split', ['val'])
def test_eval_rejects_random_bone_shapes(split):
    from data_loaders.get_data import get_dataset
    cond_path = Path(__file__).resolve().parents[1] / 'dataset/merged/cond.npy'
    if not cond_path.exists():
        pytest.skip('Merged dataset absent')
    with pytest.raises(ValueError, match='training splits only'):
        get_dataset(num_frames=60, split=split, objects_subset='all',
                    bone_length_aug_prob=.2, bone_length_aug=.1, cond_path=str(cond_path))


def test_real_training_combines_bone_leaf_and_temporal_augmentation():
    from data_loaders.get_data import get_dataset
    from data_loaders.tensors import truebones_batch_collate
    cond_path = Path(__file__).resolve().parents[1] / 'dataset/merged/cond.npy'
    if not cond_path.exists():
        pytest.skip('Merged dataset absent')
    dataset = get_dataset(num_frames=60, split='train', objects_subset='all',
                          bone_length_aug_prob=1., bone_length_aug=.1, leaf_drop_prob=1.,
                          motion_speed_aug=1.3, cond_path=str(cond_path)).motion_dataset
    candidates = list(dataset.name_list)
    assert candidates
    rng = random.Random(4)
    samples = []
    leaf_seen = bone_seen = False
    for name in rng.sample(candidates, min(20, len(candidates))):
        random.seed(rng.randrange(10000))
        sample = dataset._prepare_sample(name, dataset.data_dict[name], return_aug_info=True)
        cond = sample[13]['object_cond']
        np.testing.assert_allclose(sample[12]['joint_struct'], build_joint_struct_features(cond))
        assert np.isfinite(sample[0]).all()
        leaf_seen |= sample[13]['leaf_drop_count'] > 0
        bone_seen |= sample[13]['bone_length_aug']['applied']
        samples.append(sample[:13])
    assert leaf_seen and bone_seen
    batch, context = truebones_batch_collate(samples)
    assert batch.isfinite().all()
    assert context['y']['rest_length_scale'].isfinite().all()
    assert context['y']['bone_length_aug_applied'].any()
