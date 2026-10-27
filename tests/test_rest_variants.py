"""Rest variants: a species' hand-made rest poses, re-expressing one clip exactly."""
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loaders.truebones.truebones_utils.fbx_filename_rules import (
    find_tpose_reference_path, tpose_variant_paths, tpose_variant_tag,
)
from data_loaders.truebones.truebones_utils.rest_variants import (
    REST_VARIANTS_KEY, apply_rest_variant, augment_rest_variant, empty_rest_variants,
    rest_variant_count, stored_rest_struct_cond, validate_rest_variant_prob,
)
from data_loaders.truebones.truebones_utils.features import recover_processed_animation_from_feature_animation
from data_loaders.truebones.truebones_utils.joint_struct_features import build_joint_struct_features
from data_loaders.reference import REST_GEOM_COLUMNS, REST_POSE_COLUMNS, prepare_reference_sample, rest_geometry
from motion_lib.Animation import rotations_global
from test_rest_repose_aug import PARENTS, fixture, make_dataset, world


def axis_angle(axis, deg):
    axis = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    a = np.radians(deg)
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k


def with_variants(cond):
    """Two variants: legs swung forward and spread (A), tail curled up (B)."""
    a = np.repeat(np.eye(3)[None], len(PARENTS), axis=0)
    a[1], a[4] = axis_angle([1, .3, 0], 50), axis_angle([1, -.3, 0], 50)
    a[2], a[5] = axis_angle([1, 0, .2], -35), axis_angle([1, 0, -.2], -35)
    b = np.repeat(np.eye(3)[None], len(PARENTS), axis=0)
    b[7], b[8] = axis_angle([1, 0, 0], 40), axis_angle([0, 1, 1], 70)
    block = empty_rest_variants('primary')
    block.update(tags=['A', 'B'], rotations=np.stack([a, b]).astype(np.float32), sha1=['a', 'b'])
    return dict(cond, **{REST_VARIANTS_KEY: block})


@pytest.mark.parametrize('name, tag', [
    ('TPose.glb', None), ('Bear-TPOSE.glb', None), ('CrabALL-TPOSE.glb', None), ('Walk_01.glb', None),
    ('TPose_A.glb', 'A'), ('TPose_1.glb', '1'), ('Bear-Tpose_crouch.glb', 'crouch'), ('T-Pose_2.fbx', '2'),
    ('TPose_A_KIHumanRetarget.glb', None),
])
def test_variant_tag(name, tag):
    assert tpose_variant_tag(name) == tag


def test_variants_never_become_primary_or_clips():
    files = ['s/TPose_A.glb', 's/Idle.glb', 's/TPose.glb', 's/TPose_1.glb', 's/Walk.glb']
    assert tpose_variant_paths(files) == [('1', 's/TPose_1.glb'), ('A', 's/TPose_A.glb')]
    remaining = list(files)
    assert find_tpose_reference_path(remaining) == 's/TPose.glb'
    assert remaining == ['s/Idle.glb', 's/Walk.glb']
    only_variants = ['s/TPose_A.glb', 's/Idle.glb']
    assert find_tpose_reference_path(only_variants) == 's/Idle.glb'


def test_duplicate_variant_tags_are_rejected():
    with pytest.raises(ValueError, match='more than one file'):
        tpose_variant_paths(['s/TPose_A.glb', 's/TPose_a.fbx'])


@pytest.mark.parametrize('prob', [-.1, 1.1, np.nan])
def test_invalid_probability(prob):
    with pytest.raises(ValueError):
        validate_rest_variant_prob(prob)


@pytest.mark.parametrize('index', [0, 1])
def test_world_motion_export_and_bones_are_unchanged(index):
    motion, cond, _ = fixture(seed=index)
    cond = with_variants(cond)
    out, new_cond, info = apply_rest_variant(motion, cond, index)
    assert info['applied'] and info['tag'] == 'AB'[index]
    np.testing.assert_allclose(world(out, new_cond)[0], world(motion, cond)[0], atol=1e-5)
    np.testing.assert_array_equal(out[..., :3], motion[..., :3])
    assert not np.allclose(out[..., 3:9], motion[..., 3:9])
    np.testing.assert_allclose(np.linalg.norm(new_cond['offsets'], axis=-1),
                               np.linalg.norm(cond['offsets'], axis=-1), atol=1e-6)
    assert not np.allclose(new_cond['rest_pos_ric_hml'], cond['rest_pos_ric_hml'], atol=1e-3)
    assert REST_VARIANTS_KEY not in new_cond

    def processed(m, c):
        return rotations_global(recover_processed_animation_from_feature_animation(
            world(m, c)[1], c['tpose_rest_rotations'])).transforms()

    np.testing.assert_allclose(processed(out, new_cond), processed(motion, cond), atol=1e-5)


def test_species_without_variants_is_a_skip():
    motion, cond, _ = fixture()
    assert rest_variant_count(cond) == 0
    out, new_cond, info = augment_rest_variant(motion, cond)
    assert not info['applied'] and out is motion and new_cond is cond
    assert rest_variant_count(dict(cond, **{REST_VARIANTS_KEY: empty_rest_variants()})) == 0


def test_wrong_joint_count_is_rejected():
    motion, cond, _ = fixture()
    cond = with_variants(cond)
    cond[REST_VARIANTS_KEY]['rotations'] = cond[REST_VARIANTS_KEY]['rotations'][:, :-1]
    with pytest.raises(ValueError, match='re-run preprocessing'):
        apply_rest_variant(motion, cond, 0)


def test_struct_after_topology_edit_reads_the_stored_rest():
    _, cond, _ = fixture()
    cond = with_variants(cond)
    _, variant, _ = apply_rest_variant(fixture()[0], cond, 0)
    # Drop the two leg tips: kept joints renumbered in order.
    new_index = np.array([0, 1, 2, -1, 3, 4, -1, 5, 6, 7])
    kept = np.flatnonzero(new_index >= 0)
    edited = dict(variant, rest_pos_ric_hml=variant['rest_pos_ric_hml'][kept])
    struct_cond = stored_rest_struct_cond(edited, cond, new_index)
    np.testing.assert_array_equal(struct_cond['rest_pos_ric_hml'], cond['rest_pos_ric_hml'][kept])
    np.testing.assert_array_equal(edited['rest_pos_ric_hml'], variant['rest_pos_ric_hml'][kept])


def test_loader_draws_a_variant_and_keeps_structural_channel():
    dataset, data = make_dataset()
    cond = with_variants(dataset.cond_dict['test'])
    dataset.cond_dict['test'] = cond
    dataset.opt.rest_repose_prob = 0.
    dataset.opt.rest_variant_prob = 1.
    random.seed(4)
    sample = dataset._prepare_sample('test', data, return_aug_info=True)
    info = sample[13]['rest_variant']
    assert info['applied'] and info['tag'] in ('A', 'B')
    new_cond = sample[13]['object_cond']
    np.testing.assert_allclose(sample[3], rest_geometry(new_cond)[:, REST_GEOM_COLUMNS])
    assert not np.allclose(sample[3], rest_geometry(cond)[:, REST_GEOM_COLUMNS])
    np.testing.assert_array_equal(sample[12]['joint_struct'], dataset._joint_struct_cache['test'])
    np.testing.assert_allclose(sample[4], new_cond['offsets'])


def test_reference_draws_its_own_variant_and_keeps_structural_channel():
    motion, cond, metadata = fixture()
    cond = with_variants(cond)
    stored = build_joint_struct_features(cond)
    opt = SimpleNamespace(rest_variant_prob=1.)
    random.seed(2)
    sample = prepare_reference_sample(motion, cond, metadata, opt, shared=dict(joint_struct=stored))
    np.testing.assert_array_equal(sample['joint_struct'], stored)
    assert not np.allclose(sample['rest_pose'], rest_geometry(cond)[:, REST_POSE_COLUMNS])
