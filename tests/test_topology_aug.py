"""Bone split / chain merge and the three-way topology draw
(data_loaders/truebones/truebones_utils/topology_aug.py).

The synthetic tests pin the cond remapping and the motion identities each edit
rests on: a split keeps every forward-kinematics position and adds the
midpoint, a merge keeps every kept joint's position and global rotation. The
real-data test checks the same through the loader's canonical encode.
"""

from __future__ import annotations

import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.truebones.truebones_utils import topology_aug
from data_loaders.truebones.truebones_utils.topology_aug import (
    CHAIN_MERGE_MAX_RESIDUAL,
    TOPOLOGY_AUG_KINDS,
    _MergeState,
    _forward_kinematics,
    augment_topology,
    merge_candidates,
    merge_joints,
    split_bones_in_cond,
    split_bones_in_motion,
    split_candidates,
)
from data_loaders.truebones.truebones_utils.topology_relations import create_topology_edge_relations
from utils.rotation_conversions import rotation_6d_to_matrix_np

_NAMES = ['Root', 'Spine 1', 'Spine 2', 'Neck', 'Head',
          'Left Upper Arm', 'Left Forearm', 'Left Hand',
          'Right Upper Arm', 'Right Forearm', 'Right Hand',
          'Tail 1', 'Tail 2']
_PARENTS = [-1, 0, 1, 2, 3, 2, 5, 6, 2, 8, 9, 0, 11]
_OFFSETS = [[0, 1, 0], [0, .3, 0], [0, .3, 0], [0, .2, .1], [0, .1, .2],
            [.3, 0, 0], [.3, 0, 0], [.1, 0, 0],
            [-.3, 0, 0], [-.3, 0, 0], [-.1, 0, 0],
            [0, 0, -.4], [0, 0, -.4]]
_PAIRS = ((5, 8), (6, 9), (7, 10))
_SPINE_1, _SPINE_2, _L_FOREARM, _R_FOREARM, _TAIL_1 = 1, 2, 6, 9, 11


def _entry():
    joint_count = len(_PARENTS)
    parents = np.asarray(_PARENTS, dtype=np.int64)
    offsets = np.asarray(_OFFSETS, dtype=np.float32)
    rest = offsets.copy()
    for joint in range(1, joint_count):
        rest[joint] += rest[parents[joint]]
    partners = [-1] * joint_count
    sides = ['center'] * joint_count
    for left, right in _PAIRS:
        partners[left], partners[right] = right, left
        sides[left], sides[right] = 'left', 'right'
    entry = {
        'parents': parents,
        'offsets': offsets,
        'rest_pose': np.concatenate([rest, np.zeros((joint_count, 9), np.float32)], axis=1),
        'rest_pos_ric_hml': rest.copy(),
        'tpose_rest_rotations': np.tile(np.asarray([1, 0, 0, 0], np.float32), (joint_count, 1)),
        'joints_names_embs': np.eye(joint_count, dtype=np.float32),
        'joints_names': [name.replace(' ', '_') for name in _NAMES],
        'canonical_joint_names': list(_NAMES),
        'symmetry_partner_indices': partners,
        'joint_side_labels': sides,
        'symmetric_joint_pairs': [list(pair) for pair in _PAIRS],
        'symmetric_joint_pair_names': [[_NAMES[a], _NAMES[b]] for a, b in _PAIRS],
        'face_joints': [5, 8],
        'kinematic_chains': [[0, 1, 2, 3, 4], [2, 5, 6, 7], [2, 8, 9, 10], [0, 11, 12]],
        'translation_root_index': 0,
        'forward_joint_index': 4,
        'forward_base_joint_index': None,
    }
    entry['joint_relations'], entry['joints_graph_dist'] = create_topology_edge_relations(
        parents, symmetry_partner_indices=partners, joint_side_labels=sides)
    return entry


def _axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def _motion(entry, bends, frames=6):
    """Working-layout clip whose positions are the FK of its rotations;
    ``bends`` maps a joint to its peak angle (radians) about one axis."""
    joint_count = len(entry['parents'])
    local = np.repeat(np.repeat(np.eye(3)[None, None], frames, 0), joint_count, 1)
    for joint, (axis, angle) in bends.items():
        for t in range(frames):
            local[t, joint] = _axis_angle(axis, angle * t / (frames - 1))
    motion = np.zeros((frames, joint_count, 12), np.float32)
    motion[..., 3:9] = np.concatenate([local[..., :, 0], local[..., :, 1]], axis=-1)
    _globals, positions = _forward_kinematics(motion, entry)
    motion[..., :3] = positions + np.asarray(entry['rest_pos_ric_hml'][0])
    motion[:, 0, [0, 2]] = np.linspace(0, 1, frames)[:, None]  # root world XZ
    return motion


def _ric(motion):
    out = np.array(motion[..., :3], dtype=np.float64)
    out[:, 0, [0, 2]] = 0.0
    return out


def test_split_candidates_are_long_bones_with_twins():
    units = split_candidates(_entry())
    # Mean bone length is ~0.27: spine, arms' upper/forearm, tail, root->spine.
    assert (_L_FOREARM, _R_FOREARM) in units and (_TAIL_1,) in units
    assert all(len(unit) == 1 or tuple(sorted(unit)) in {(5, 8), (6, 9)} for unit in units)
    assert not any(7 in unit or 10 in unit for unit in units)  # hands are short


def test_split_remaps_cond_and_keeps_motion():
    entry = _entry()
    children = [_L_FOREARM, _R_FOREARM, _TAIL_1]
    source, new_index, mid_index, new = split_bones_in_cond(entry, children)
    assert len(new['parents']) == len(_PARENTS) + 3
    for child in children:
        mid = mid_index[child]
        assert new['parents'][new_index[child]] == mid
        assert new['parents'][mid] == new_index[_PARENTS[child]]
        np.testing.assert_allclose(new['offsets'][mid], np.asarray(_OFFSETS[child]) / 2)
        np.testing.assert_allclose(new['offsets'][new_index[child]], np.asarray(_OFFSETS[child]) / 2)
        np.testing.assert_allclose(
            new['rest_pos_ric_hml'][mid],
            (entry['rest_pos_ric_hml'][child] + entry['rest_pos_ric_hml'][_PARENTS[child]]) / 2)
        assert new['joint_side_labels'][mid] == entry['joint_side_labels'][child]
        np.testing.assert_array_equal(new['tpose_rest_rotations'][mid], [1, 0, 0, 0])
    # Every new parent precedes its child.
    assert all(p < j for j, p in enumerate(new['parents']) if p >= 0)
    # Named after the bone it splits, its parent on the same side.
    assert new['canonical_joint_names'][mid_index[_L_FOREARM]] == 'Left Upper Arm Mid'
    assert len(set(new['joints_names'])) == len(new['joints_names'])
    left, right = mid_index[_L_FOREARM], mid_index[_R_FOREARM]
    assert new['symmetry_partner_indices'][left] == right
    assert [left, right] in new['symmetric_joint_pairs']
    assert new['face_joints'] == [int(new_index[5]), int(new_index[8])]
    assert new['forward_joint_index'] == int(new_index[4])
    assert [int(new_index[2]), int(new_index[5]), left, int(new_index[6]), int(new_index[7])] \
        in new['kinematic_chains']
    edge, dist = create_topology_edge_relations(
        new['parents'], symmetry_partner_indices=new['symmetry_partner_indices'],
        joint_side_labels=new['joint_side_labels'])
    np.testing.assert_array_equal(new['joint_relations'], edge)
    np.testing.assert_array_equal(new['joints_graph_dist'], dist)

    motion = _motion(entry, {5: ((0, 0, 1), .8), 6: ((0, 1, 0), .6), 11: ((1, 0, 0), .5)})
    out = split_bones_in_motion(motion, entry, mid_index, source)
    np.testing.assert_array_equal(out[:, new_index], motion)
    # The split rig's FK lands on the stored positions, the midpoints included.
    _globals, positions = _forward_kinematics(out, new)
    np.testing.assert_allclose(positions - positions[:, :1], _ric(out) - _ric(out)[:, :1], atol=1e-5)


def test_merge_keeps_positions_and_global_rotations():
    entry = _entry()
    # A twist about the bone below it moves nothing: Spine 2's child bone is +Y.
    motion = _motion(entry, {_SPINE_1: ((0, 1, 0), .9), 5: ((0, 0, 1), .7)})
    global_rot, positions = _forward_kinematics(motion, entry)
    state = _MergeState(entry, global_rot, positions)
    assert state.merge(_SPINE_1)
    keep, out, new = merge_joints(motion, entry, state)
    assert _SPINE_1 not in keep
    np.testing.assert_array_equal(out[..., :3], motion[:, keep, :3])
    new_globals, new_positions = _forward_kinematics(out, new)
    np.testing.assert_allclose(new_globals, global_rot[:, keep], atol=1e-6)
    np.testing.assert_allclose(new_positions, positions[:, keep], atol=1e-6)
    np.testing.assert_allclose(new['offsets'][list(keep).index(_SPINE_2)], [0, .6, 0], atol=1e-6)


def test_merge_of_a_run_composes_rotations():
    entry = _entry()
    # Twists about the arm's own axis: Upper Arm and Forearm both merge away.
    motion = _motion(entry, {5: ((1, 0, 0), .9), _L_FOREARM: ((1, 0, 0), -.4)})
    state = _MergeState(entry, *_forward_kinematics(motion, entry))
    assert state.merge(_L_FOREARM) and state.merge(5)
    keep, out, new = merge_joints(motion, entry, state)
    hand = list(keep).index(7)
    assert new['parents'][hand] == list(keep).index(_SPINE_2)
    np.testing.assert_allclose(new['offsets'][hand], [.7, 0, 0], atol=1e-6)
    global_rot, positions = _forward_kinematics(motion, entry)
    new_globals, new_positions = _forward_kinematics(out, new)
    np.testing.assert_allclose(new_globals, global_rot[:, keep], atol=1e-6)
    np.testing.assert_allclose(new_positions, positions[:, keep], atol=1e-6)


def test_a_bent_joint_is_not_merged():
    entry = _entry()
    motion = _motion(entry, {_L_FOREARM: ((0, 0, 1), 1.2)})
    state = _MergeState(entry, *_forward_kinematics(motion, entry))
    assert not state.merge(_L_FOREARM)
    assert not state.removed
    # The same joint at rest merges.
    still = _motion(entry, {})
    assert _MergeState(entry, *_forward_kinematics(still, entry)).merge(_L_FOREARM)
    assert CHAIN_MERGE_MAX_RESIDUAL < 1.0


def test_merge_protects_indexed_joints_and_pairs_twins():
    units = merge_candidates(_entry(), 0)
    flat = {joint for unit in units for joint in unit}
    assert not flat & {0, 4, 5, 8}          # root, forward, face joints
    assert (_L_FOREARM, _R_FOREARM) in units
    assert 2 not in flat and 3 in flat      # Spine 2 branches; Neck has one child


def test_draw_covers_every_kind_and_respects_limits():
    entry = _entry()
    entry['canonical_joint_names'][12] = 'Tail Nub'
    motion = _motion(entry, {})
    seen = set()
    rng = random.Random(0)
    for _ in range(60):
        out, new, info = augment_topology(motion, entry, 0, max_joints=15, rng=rng)
        assert info['applied']
        seen.add(info['kind'])
        assert out.shape[1] == len(new['parents']) <= 15
        assert len(new['parents']) >= topology_aug.LEAF_DROP_MIN_KEPT_JOINTS
        assert info['translation_root_index'] == 0
    assert seen == set(TOPOLOGY_AUG_KINDS)


def test_draw_falls_back_when_a_kind_has_nothing():
    entry = _entry()
    motion = _motion(entry, {})
    rng = random.Random(1)
    for _ in range(20):
        # No room to split: the draw lands on a merge (no leaf is droppable).
        _out, _new, info = augment_topology(motion, entry, 0, max_joints=len(_PARENTS),
                                            kinds=('split', 'merge'), rng=rng)
        assert info['kind'] == 'merge'


_MERGED_COND = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'dataset', 'cond_all.npy')
_needs_merged = pytest.mark.skipif(not os.path.isfile(_MERGED_COND), reason='merged dataset not present')


@_needs_merged
@pytest.mark.parametrize('kind', ['split', 'merge'])
def test_kept_rows_decode_to_the_stored_motion(kind):
    import data_loaders.truebones.data.dataset as dataset_module
    from data_loaders.get_data import get_dataset
    from data_loaders.truebones.truebones_utils.canonical_features import canonical_to_physical_hml
    from data_loaders.truebones.truebones_utils.param_utils import MAX_SOURCE_FRAMES_MULT

    motion_dataset = get_dataset(
        num_frames=60, split='train', objects_subset='all', action_group='all',
        topology_aug_prob=1.0, cond_path=_MERGED_COND,
    ).motion_dataset
    motion_dataset.opt.topology_aug_kinds = (kind,)
    captured = {}
    real_augment = dataset_module.augment_topology

    def spy(*args, **kwargs):
        result = real_augment(*args, **kwargs)
        captured['info'] = result[2]
        return result

    budget = 60 * MAX_SOURCE_FRAMES_MULT
    names = [
        name for name in motion_dataset.name_list[motion_dataset.pointer:]
        if not motion_dataset.data_dict[name]['motion_metadata'].get('is_loop')
        and motion_dataset.data_dict[name]['length'] <= (
            60 if motion_dataset.data_dict[name].get('high_frequency', False) else budget
        )
    ]
    checked = 0
    try:
        dataset_module.augment_topology = spy
        for name in random.Random(0).sample(names, min(40, len(names))):
            data = motion_dataset.data_dict[name]
            motion_dataset.opt.topology_aug_prob = 0.0
            random.seed(7)
            plain = motion_dataset._prepare_sample(name, data)
            captured.clear()
            motion_dataset.opt.topology_aug_prob = 1.0
            random.seed(7)
            augmented = motion_dataset._prepare_sample(name, data)
            info = captured.get('info')
            if not info or not info['applied'] or augmented[1] != plain[1]:
                continue
            new_index = info['new_index']
            kept = np.flatnonzero(new_index >= 0)
            expected = canonical_to_physical_hml(plain[0], plain[12])[:, kept]
            actual = canonical_to_physical_hml(augmented[0], augmented[12])[:, new_index[kept]]
            channels = slice(0, 9) if kind == 'split' else slice(0, 3)
            np.testing.assert_allclose(actual[..., channels], expected[..., channels], atol=1e-4)
            assert augmented[12]['joint_struct'].shape[0] == augmented[0].shape[1]
            checked += 1
    finally:
        dataset_module.augment_topology = real_augment
    assert checked > 0
