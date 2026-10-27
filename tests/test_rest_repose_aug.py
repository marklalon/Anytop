"""Rest re-pose: the world motion is unchanged, only its rest parameterization moves."""
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_loaders.truebones.truebones_utils.rest_repose_aug import (
    _matrix_to_rotation_6d, augment_rest_repose, validate_rest_repose_aug,
)
from data_loaders.truebones.truebones_utils.canonical_features import canonical_to_physical_hml
from data_loaders.truebones.truebones_utils.features import (
    recover_from_bvh_rot_np, recover_processed_animation_from_feature_animation,
)
from data_loaders.truebones.truebones_utils.joint_struct_features import build_joint_struct_features
from data_loaders.reference import (
    REST_GEOM_COLUMNS, REST_POSE_COLUMNS, prepare_reference_sample, rest_geometry,
)
from data_loaders.truebones.data.dataset import MotionDataset
from motion_lib.Animation import rotations_global
from motion_lib.Quaternions import Quaternions

# Hips; left leg 1-3; right leg 4-6; a midline spine/tail run 7-9.
PARENTS = np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8])
OFFSETS = np.array([[0, 0, 0], [.2, 0, 0], [0, -.5, .05], [0, -.5, 0],
                    [-.2, 0, 0], [0, -.5, .05], [0, -.5, 0],
                    [0, .1, -.2], [0, 0, -.3], [0, -.1, -.3]], dtype=np.float32)
PARTNERS = [-1, 4, 5, 6, 1, 2, 3, -1, -1, -1]
SIDES = ['center'] + ['left'] * 3 + ['right'] * 3 + ['center'] * 3


def random_rotations(rng, shape):
    q = rng.normal(size=shape + (4,))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return Quaternions(q)


def fixture(frames=12, seed=0):
    rng = np.random.default_rng(seed)
    rest = OFFSETS.astype(np.float64).copy()
    rest[0, 1] = 1
    for j in range(1, len(PARENTS)):
        rest[j] += rest[PARENTS[j]]
    rest_feature = np.zeros((len(PARENTS), 12), np.float32)
    rest_feature[:, :3] = rest
    rest_feature[:, 3] = rest_feature[:, 7] = 1
    cond = dict(parents=PARENTS, offsets=OFFSETS.copy(), rest_pos_ric_hml=rest.astype(np.float32),
                rest_pose=rest_feature, symmetry_partner_indices=PARTNERS, translation_root_index=0,
                joint_side_labels=list(SIDES), tpose_rest_rotations=random_rotations(rng, (len(PARENTS),)).qs.astype(np.float32),
                canonical_feature_mean=np.zeros(15), canonical_feature_std=np.ones(15),
                joint_mask_candidate_roots=np.zeros(len(PARENTS)),
                joints_names_embs=np.zeros((len(PARENTS), 4)),
                joints_graph_dist=np.zeros((len(PARENTS),) * 2),
                joint_relations=np.zeros((len(PARENTS),) * 2), kinematic_chains=[])
    motion = np.broadcast_to(rest_feature, (frames, len(PARENTS), 12)).copy()
    motion[..., 3:9] = _matrix_to_rotation_6d(random_rotations(rng, (frames, len(PARENTS))).transforms())
    motion[..., :3] += rng.normal(scale=.05, size=motion[..., :3].shape)
    motion[:, :, 9:12] = rng.normal(scale=.01, size=motion[..., 9:12].shape)
    return motion, cond, dict(translation_root_index=0, is_loop=False)


def world(motion, cond):
    return recover_from_bvh_rot_np(motion, cond['parents'], np.asarray(cond['offsets'], np.float64),
                                   translation_root_index=0)


@pytest.mark.parametrize('prob, max_deg', [(-.1, 10), (1.1, 10), (np.nan, 10), (.2, -1), (.2, 91), (.2, np.inf)])
def test_invalid_configuration(prob, max_deg):
    with pytest.raises(ValueError):
        validate_rest_repose_aug(prob, max_deg)


@pytest.mark.parametrize('seed', range(8))
def test_world_motion_and_absolute_channels_are_unchanged(seed):
    motion, cond, metadata = fixture(seed=seed)
    out, new_cond, info = augment_rest_repose(motion, cond, metadata, 40., rng=random.Random(seed))
    assert info['applied'] and info['max_rest_angle_deg'] > 0
    np.testing.assert_allclose(world(out, new_cond)[0], world(motion, cond)[0], atol=1e-5)
    np.testing.assert_array_equal(out[..., :3], motion[..., :3])
    np.testing.assert_array_equal(out[..., 9:12], motion[..., 9:12])
    assert not np.allclose(out[..., 3:9], motion[..., 3:9])


def test_rest_keeps_lengths_symmetry_midline_and_grounding():
    motion, cond, metadata = fixture()
    _, new_cond, _ = augment_rest_repose(motion, cond, metadata, 40., rng=random.Random(5))
    np.testing.assert_allclose(np.linalg.norm(new_cond['offsets'], axis=-1),
                               np.linalg.norm(cond['offsets'], axis=-1), atol=1e-6)
    rest, old = new_cond['rest_pos_ric_hml'], cond['rest_pos_ric_hml']
    assert not np.allclose(rest, old, atol=1e-3)
    mirror = np.array([-1., 1., 1.])
    np.testing.assert_allclose(rest[1:4], rest[4:7] * mirror, atol=1e-6)
    np.testing.assert_allclose(rest[7:, 0], 0, atol=1e-7)
    assert rest[:, 1].min() == pytest.approx(old[:, 1].min(), abs=1e-6)
    np.testing.assert_allclose(new_cond['rest_pose'][:, :3], rest)
    assert 'rest_length_scale' not in new_cond


def test_source_cond_and_motion_are_not_written():
    motion, cond, metadata = fixture()
    motion_copy, offsets, rest = motion.copy(), cond['offsets'].copy(), cond['rest_pos_ric_hml'].copy()
    augment_rest_repose(motion, cond, metadata, 30., rng=random.Random(1))
    np.testing.assert_array_equal(motion, motion_copy)
    np.testing.assert_array_equal(cond['offsets'], offsets)
    np.testing.assert_array_equal(cond['rest_pos_ric_hml'], rest)


def test_zero_magnitude_is_a_skip():
    motion, cond, metadata = fixture()
    out, new_cond, info = augment_rest_repose(motion, cond, metadata, 0., rng=random.Random(1))
    assert not info['applied'] and out is motion and new_cond is cond


def test_export_bind_keeps_processed_animation():
    motion, cond, metadata = fixture()
    out, new_cond, _ = augment_rest_repose(motion, cond, metadata, 40., rng=random.Random(3))

    def processed(m, c):
        anim = world(m, c)[1]
        return rotations_global(recover_processed_animation_from_feature_animation(
            anim, c['tpose_rest_rotations'])).transforms()

    np.testing.assert_allclose(processed(out, new_cond), processed(motion, cond), atol=1e-5)


def make_dataset():
    motion, cond, metadata = fixture(frames=30)
    dataset = MotionDataset.__new__(MotionDataset)
    dataset.opt = SimpleNamespace(rest_repose_prob=1., rest_repose_max_deg=30., motion_speed_aug=1.)
    dataset.max_motion_length = 30
    dataset.min_length = 2
    dataset.opt.max_joints = len(PARENTS)
    dataset.action_conditioning = None
    dataset.cond_dict = {'test': cond}
    dataset._joint_struct_cache = {'test': build_joint_struct_features(cond)}
    # MotionDataset.__init__ sets it; the fixture rig has no foot chains.
    cond['limb_chain_id'] = np.full(len(cond['parents']), -1)
    cond['limb_wing'] = np.zeros(len(cond['parents']), dtype=bool)
    cond['limb_reach'] = np.zeros(len(cond['parents']))
    dataset._load_physical_motion = lambda data: (
        motion, len(motion), 'test', cond['parents'], cond['joints_graph_dist'],
        cond['joint_relations'], rest_geometry(cond)[:, REST_GEOM_COLUMNS], cond['offsets'],
        cond['joints_names_embs'], [])
    return dataset, dict(motion_metadata=dict(metadata, action_label='walk'))


def test_loader_rebuilds_rest_geometry_and_keeps_structural_channel():
    dataset, data = make_dataset()
    random.seed(3)
    sample = dataset._prepare_sample('test', data, return_aug_info=True)
    assert sample[13]['rest_repose_aug']['applied']
    new_cond = sample[13]['object_cond']
    np.testing.assert_allclose(sample[3], rest_geometry(new_cond)[:, REST_GEOM_COLUMNS])
    np.testing.assert_array_equal(sample[12]['joint_struct'], dataset._joint_struct_cache['test'])
    np.testing.assert_allclose(sample[4], new_cond['offsets'])
    decoded = canonical_to_physical_hml(sample[0], sample[12])
    assert np.isfinite(decoded).all()


@pytest.mark.parametrize('split', ['val'])
def test_eval_rejects_random_rests(split):
    from data_loaders.get_data import get_dataset
    cond_path = Path(__file__).resolve().parents[1] / 'dataset/cond_all.npy'
    if not cond_path.exists():
        pytest.skip('Merged dataset absent')
    with pytest.raises(ValueError, match='training splits only'):
        get_dataset(num_frames=60, split=split, objects_subset='all',
                    rest_repose_prob=.2, cond_path=str(cond_path))


def test_reference_keeps_the_stored_structural_channel():
    motion, cond, metadata = fixture()
    stored = build_joint_struct_features(cond)
    opt = SimpleNamespace(rest_repose_prob=1., rest_repose_max_deg=40.)
    random.seed(2)
    sample = prepare_reference_sample(motion, cond, metadata, opt, shared=dict(joint_struct=stored))
    np.testing.assert_array_equal(sample['joint_struct'], stored)
    assert not np.allclose(sample['rest_pose'], rest_geometry(cond)[:, REST_POSE_COLUMNS])


def test_canonical_features_are_rest_invariant():
    from data_loaders.truebones.truebones_utils.canonical_features import physical_hml_to_canonical
    motion, cond, metadata = fixture()
    for seed in range(20):
        random.seed(seed)
        new_motion, new_cond, info = augment_rest_repose(motion, cond, metadata, 30.0)
        if info['applied']:
            break
    assert info['applied']
    np.testing.assert_allclose(
        physical_hml_to_canonical(new_motion, new_cond)[..., :9],
        physical_hml_to_canonical(motion, cond)[..., :9], atol=1e-4)


def frame_edge_fixture(frames=12, seed=0):
    """Rest frames ``_bone_frames`` takes from a joint's own bone: a spine (7)
    whose two clavicles cancel sideways, and a mirror pair whose right arm
    (11) is a leaf while the left one (10) carries a hand."""
    parents = np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 7, 8, 9, 10])
    offsets = np.array([[0, 0, 0], [.2, 0, 0], [0, -.5, .05], [0, -.5, 0],
                        [-.2, 0, 0], [0, -.5, .05], [0, -.5, 0],
                        [0, .3, 0], [.2, 0, 0], [-.2, 0, 0], [.3, 0, 0], [-.3, 0, 0],
                        [.2, 0, 0]], dtype=np.float32)
    partners = [-1, 4, 5, 6, 1, 2, 3, -1, 9, 8, 11, 10, -1]
    rng = np.random.default_rng(seed)
    rest = offsets.astype(np.float64).copy()
    rest[0, 1] = 1
    for j in range(1, len(parents)):
        rest[j] += rest[parents[j]]
    rest_feature = np.zeros((len(parents), 12), np.float32)
    rest_feature[:, :3] = rest
    rest_feature[:, 3] = rest_feature[:, 7] = 1
    cond = dict(parents=parents, offsets=offsets, rest_pos_ric_hml=rest.astype(np.float32),
                rest_pose=rest_feature, symmetry_partner_indices=partners, translation_root_index=0,
                canonical_feature_mean=np.zeros(15), canonical_feature_std=np.ones(15))
    motion = np.broadcast_to(rest_feature, (frames, len(parents), 12)).copy()
    motion[..., 3:9] = _matrix_to_rotation_6d(random_rotations(rng, (frames, len(parents))).transforms())
    return motion, cond, dict(translation_root_index=0, is_loop=False)


def test_canonical_rotation_is_exact_where_frames_follow_the_parent():
    from data_loaders.truebones.truebones_utils.canonical_features import physical_hml_to_canonical
    motion, cond, metadata = frame_edge_fixture()
    base = physical_hml_to_canonical(motion, cond)[..., 3:9]
    applied = 0
    for seed in range(40):
        new_motion, new_cond, info = augment_rest_repose(
            motion, cond, metadata, 40.0, rng=random.Random(seed))
        applied += info['applied']
        np.testing.assert_allclose(
            physical_hml_to_canonical(new_motion, new_cond)[..., 3:9], base, atol=1e-4, err_msg=str(seed))
    assert applied


def test_reference_keeps_the_structural_channel_across_bone_lengths():
    motion, cond, metadata = fixture()
    stored = build_joint_struct_features(cond)
    opt = SimpleNamespace(bone_length_aug_prob=1., bone_length_aug_max=.5, bone_length_aug_min=.4)
    random.seed(0)
    sample = prepare_reference_sample(motion, cond, metadata, opt, shared=dict(joint_struct=stored))
    np.testing.assert_array_equal(sample['joint_struct'], stored)
    assert not np.allclose(sample['rest_pose'], rest_geometry(cond)[:, REST_POSE_COLUMNS])


def test_bent_limbs_stay_on_their_side():
    motion, cond, metadata = fixture()
    side = cond['rest_pos_ric_hml'][:, 0] - cond['rest_pos_ric_hml'][0, 0]
    sided = np.abs(side) > 1e-4
    applied = 0
    for seed in range(200):
        random.seed(seed)
        _, new_cond, info = augment_rest_repose(motion, cond, metadata, 90.0)
        applied += info['applied']
        new_side = new_cond['rest_pos_ric_hml'][:, 0] - new_cond['rest_pos_ric_hml'][0, 0]
        assert np.all(new_side[sided] * side[sided] > 0), seed
    assert applied
