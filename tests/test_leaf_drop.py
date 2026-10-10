"""Leaf-drop topology augmentation (data_loaders/truebones/truebones_utils/leaf_drop.py).

The synthetic tests pin the selection rules and the cond remapping; the real-data
tests check that selected standalone body-part leaves survive on real rigs, that evaluation splits
refuse the augmentation, and the property it rests on -- every kept joint's
decoded physical row is the stored one, up to float32 rounding.
"""

from __future__ import annotations

import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.truebones.truebones_utils.leaf_drop import (
    LEAF_STATIC_MAX_DEG,
    classify_leaves,
    drop_joints_from_cond,
    sample_leaf_drop,
    static_joint_mask,
)
from data_loaders.truebones.truebones_utils.topology_relations import (
    create_topology_edge_relations,
)

_NAMES = [
    'Hips', 'Spine', 'Head', 'Head Nub',                       # 0-3
    'Left Finger 1', 'Left Finger 2',                          # 4-5
    'Right Finger 1', 'Right Finger 2',                        # 6-7
    'Left Foot', 'Tail 1', 'Tail 2', 'Jaw',                    # 8-11
    '', '',                                                    # 12-13 nameless
    'Left Leg 1', 'Left Leg 2', 'Right Leg 1', 'Right Leg 2',  # 14-17
]
_PARENTS = [-1, 0, 1, 2, 1, 4, 1, 6, 0, 0, 9, 2, 2, 2, 0, 14, 0, 16]
_HEAD_NUB, _L_FINGER_TIP, _R_FINGER_TIP, _FOOT, _TAIL_TIP, _JAW = 3, 5, 7, 8, 10, 11
_NAMELESS_STATIC, _NAMELESS_LIVE, _L_LEG_TIP, _R_LEG_TIP = 12, 13, 15, 17
_ALL_LEAVES = [3, 5, 7, 8, 10, 11, 12, 13, 15, 17]


def _entry():
    joint_count = len(_PARENTS)
    partners = [-1] * joint_count
    for left, right in ((4, 6), (5, 7), (14, 16), (15, 17)):
        partners[left], partners[right] = right, left
    rest = np.arange(joint_count * 3, dtype=np.float32).reshape(joint_count, 3)
    embs = np.eye(joint_count, dtype=np.float32)
    embs[[_NAMELESS_STATIC, _NAMELESS_LIVE]] = 0.0
    entry = {
        'parents': np.asarray(_PARENTS, dtype=np.int64),
        'offsets': rest.copy(),
        'rest_pose': np.concatenate([rest, np.zeros((joint_count, 9), np.float32)], axis=1),
        'rest_pos_ric_hml': rest.copy(),
        'joints_names_embs': embs,
        'joint_mask_candidate_roots': np.ones(joint_count, dtype=np.bool_),
        'joints_names': [f'j{j}' for j in range(joint_count)],
        'canonical_joint_names': list(_NAMES),
        'symmetry_partner_indices': partners,
        'symmetric_joint_pairs': [[4, 6], [5, 7], [14, 16], [15, 17]],
        'symmetric_joint_pair_names': [['j4', 'j6'], ['j5', 'j7'], ['j14', 'j16'], ['j15', 'j17']],
        'joint_parts': np.arange(joint_count, dtype=np.int16),
        'joint_contact': np.isin(np.arange(joint_count), [_FOOT]),
        'face_joints': [4, 6],
        'kinematic_chains': [[0, 1, 2, 3], [1, 4, 5], [1, 6, 7], [0, 8], [0, 9, 10]],
        'translation_root_index': 0,
        'forward_joint_index': 2,
        'forward_base_joint_index': None,
    }
    edge, dist = create_topology_edge_relations(entry['parents'], symmetry_partner_indices=partners)
    entry['joint_relations'], entry['joints_graph_dist'] = edge, dist
    return entry


def _motion(live_joints, frames=8):
    """(T, J, 12) with identity rotations, except ``live_joints`` swinging
    about z well past the static threshold."""
    motion = np.zeros((frames, len(_PARENTS), 12), dtype=np.float32)
    motion[:, :, 3] = 1.0  # x column
    motion[:, :, 7] = 1.0  # y column
    angle = np.radians(np.linspace(0.0, 4.0 * LEAF_STATIC_MAX_DEG, frames))
    for joint in live_joints:
        motion[:, joint, 3:6] = np.stack([np.cos(angle), np.sin(angle), 0 * angle], axis=1)
        motion[:, joint, 6:9] = np.stack([-np.sin(angle), np.cos(angle), 0 * angle], axis=1)
    return motion


def test_static_joint_mask_threshold():
    motion = _motion(live_joints=[_TAIL_TIP])
    assert static_joint_mask(motion, [_HEAD_NUB, _TAIL_TIP]).tolist() == [True, False]


def test_classify_leaves():
    # The named terminator counts even when it moves; the nameless one only when static.
    motion = _motion(live_joints=[_HEAD_NUB, _NAMELESS_LIVE, _L_FINGER_TIP, _R_FINGER_TIP, _TAIL_TIP])
    assert classify_leaves(_entry(), motion, _ALL_LEAVES) == {
        _HEAD_NUB: 'terminator',
        _NAMELESS_STATIC: 'terminator',
        _L_FINGER_TIP: 'segment',
        _R_FINGER_TIP: 'segment',
        _TAIL_TIP: 'segment',
    }


def test_drop_remaps_every_index():
    keep, entry = drop_joints_from_cond(_entry(), [_HEAD_NUB, _L_FINGER_TIP, _R_FINGER_TIP])
    assert keep.tolist() == [0, 1, 2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    assert entry['parents'].tolist() == [-1, 0, 1, 1, 1, 0, 0, 6, 2, 2, 2, 0, 11, 0, 13]
    assert entry['symmetry_partner_indices'] == [-1, -1, -1, 4, 3, -1, -1, -1, -1, -1, -1, 13, 14, 11, 12]
    assert entry['symmetric_joint_pairs'] == [[3, 4], [11, 13], [12, 14]]
    assert entry['symmetric_joint_pair_names'] == [['j4', 'j6'], ['j14', 'j16'], ['j15', 'j17']]
    assert np.flatnonzero(entry['joint_contact']).tolist() == [5]
    np.testing.assert_array_equal(entry['joint_parts'], _entry()['joint_parts'][keep])
    assert entry['face_joints'] == [3, 4]
    assert entry['kinematic_chains'] == [[0, 1, 2], [1, 3], [1, 4], [0, 5], [0, 6, 7]]
    assert entry['forward_joint_index'] == 2 and entry['forward_base_joint_index'] is None
    assert entry['canonical_joint_names'] == [_NAMES[j] for j in keep]
    np.testing.assert_array_equal(entry['joints_names_embs'], _entry()['joints_names_embs'][keep])
    np.testing.assert_array_equal(entry['rest_pos_ric_hml'], _entry()['rest_pos_ric_hml'][keep])
    edge, dist = create_topology_edge_relations(
        entry['parents'], symmetry_partner_indices=entry['symmetry_partner_indices'])
    np.testing.assert_array_equal(entry['joint_relations'], edge)
    np.testing.assert_array_equal(entry['joints_graph_dist'], dist)


def test_drop_rejects_a_non_leaf():
    with pytest.raises(ValueError):
        drop_joints_from_cond(_entry(), [2])


def test_selection_rules():
    entry = _entry()
    motion = _motion(live_joints=[_NAMELESS_LIVE, _L_FINGER_TIP, _R_FINGER_TIP, _TAIL_TIP])
    rng = random.Random(0)
    seen_segments = set()
    for _ in range(200):
        dropped = set(sample_leaf_drop(entry, motion, translation_root_index=0, rng=rng))
        assert {_HEAD_NUB, _NAMELESS_STATIC} <= dropped    # the terminator group always goes
        # body parts, a leg's last joint, a moving nameless joint, roots, face, contact: never
        assert not dropped & {0, 4, 6, _FOOT, _JAW, _NAMELESS_LIVE, _L_LEG_TIP, _R_LEG_TIP}
        assert (_L_FINGER_TIP in dropped) == (_R_FINGER_TIP in dropped)  # mirror twins together
        seen_segments.update(dropped - {_HEAD_NUB, _NAMELESS_STATIC})
    assert seen_segments == {_L_FINGER_TIP, _R_FINGER_TIP, _TAIL_TIP}


def test_without_a_terminator_a_segment_still_goes():
    entry = _entry()
    entry['canonical_joint_names'][_HEAD_NUB] = 'Head Top'
    motion = _motion(live_joints=[_NAMELESS_STATIC, _NAMELESS_LIVE])
    rng = random.Random(1)
    for _ in range(50):
        dropped = sample_leaf_drop(entry, motion, translation_root_index=0, rng=rng)
        assert dropped and set(dropped) <= {_L_FINGER_TIP, _R_FINGER_TIP, _TAIL_TIP}


def test_a_twin_of_a_protected_joint_is_kept():
    entry = _entry()
    entry['joint_contact'] = np.isin(np.arange(len(_PARENTS)), [_L_FINGER_TIP])
    motion = _motion(live_joints=[])
    rng = random.Random(2)
    for _ in range(50):
        dropped = set(sample_leaf_drop(entry, motion, translation_root_index=0, rng=rng))
        assert not dropped & {_L_FINGER_TIP, _R_FINGER_TIP}


_MERGED_COND = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'dataset', 'merged', 'cond.npy')
_needs_merged = pytest.mark.skipif(not os.path.isfile(_MERGED_COND), reason='merged dataset not present')


@_needs_merged
@pytest.mark.parametrize('species, clip, body_parts', [
    # Rear hooves, jaw and ears are body parts; the wings hang off the spine.
    ('truebones/zoo/Deer', 'Deer_Gallop.npy',
     ['Right Rear Hoof', 'Left Rear Hoof', 'Jaw', 'Right Ear', 'Left Ear']),
    ('truebones/zoo/Roach', 'Roach_Scurry.npy', ['Spine 1 Left Wing', 'Spine 1 Right Wing']),
])
def test_standalone_body_part_leaves_are_not_dropped(species, clip, body_parts):
    from data_loaders.truebones.truebones_utils.cond_schema import load_cond
    from data_loaders.truebones.truebones_utils.dataset_sources import resolve_anytop_path

    entry = load_cond(_MERGED_COND)[species]
    motion = np.load(os.path.join(str(resolve_anytop_path(entry['dataset_root'])), 'motions', clip))
    names = list(entry['canonical_joint_names'])
    guarded = {names.index(name) for name in body_parts}
    rng = random.Random(3)
    for _ in range(100):
        dropped = set(sample_leaf_drop(entry, motion, entry['translation_root_index'], rng=rng))
        assert not dropped & guarded, [names[j] for j in dropped & guarded]


@_needs_merged
@pytest.mark.parametrize('split', ['val', 'test'])
def test_evaluation_splits_refuse_the_augmentation(split):
    from data_loaders.get_data import get_dataset

    with pytest.raises(ValueError, match='training splits only'):
        get_dataset(num_frames=60, split=split, objects_subset='all', action_group='all',
                    leaf_drop_prob=0.5, cond_path=_MERGED_COND)


@_needs_merged
def test_kept_rows_decode_to_the_stored_motion():
    import data_loaders.truebones.data.dataset as dataset_module
    from data_loaders.get_data import get_dataset
    from data_loaders.truebones.truebones_utils.canonical_features import canonical_to_physical_hml
    from data_loaders.truebones.truebones_utils.param_utils import MAX_SOURCE_FRAMES_MULT

    motion_dataset = get_dataset(
        num_frames=60, split='train', objects_subset='all', action_group='all',
        leaf_drop_prob=1.0, cond_path=_MERGED_COND,
    ).motion_dataset
    captured = {}
    real_drop = dataset_module.drop_joints_from_cond

    def spy(entry, dropped):
        keep, new_entry = real_drop(entry, dropped)
        captured['keep'] = keep
        return keep, new_entry

    # Only clips whose every temporal stage is deterministic: a loop's roll/tile
    # and an over-budget clip's crop draw from the same RNG the leaf drop
    # advances, so the two paths would see different windows.
    budget = 60 * MAX_SOURCE_FRAMES_MULT
    names = [
        name for name in motion_dataset.name_list[motion_dataset.pointer:]
        if not motion_dataset.data_dict[name]['motion_metadata'].get('is_loop')
        and motion_dataset.data_dict[name]['length'] <= budget
    ]
    checked = 0
    try:
        dataset_module.drop_joints_from_cond = spy
        for name in random.Random(0).sample(names, min(60, len(names))):
            data = motion_dataset.data_dict[name]
            motion_dataset.opt.leaf_drop_prob = 0.0
            random.seed(7)
            plain = motion_dataset._prepare_sample(name, data)
            captured.clear()
            motion_dataset.opt.leaf_drop_prob = 1.0
            random.seed(7)
            augmented = motion_dataset._prepare_sample(name, data)
            if 'keep' not in captured or augmented[1] != plain[1]:
                continue
            keep = captured['keep']
            expected = canonical_to_physical_hml(plain[0], plain[12])[:, keep]
            actual = canonical_to_physical_hml(augmented[0], augmented[12])
            np.testing.assert_allclose(actual, expected, atol=1e-5)
            tri = augmented[10]['translation_root_index']
            assert keep[tri] == plain[10]['translation_root_index']
            assert augmented[12]['joint_struct'].shape[0] == keep.shape[0]
            checked += 1
    finally:
        dataset_module.drop_joints_from_cond = real_drop
    assert checked > 0
