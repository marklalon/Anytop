"""Chain merge and the topology draw
(data_loaders/truebones/truebones_utils/topology_aug.py).

The synthetic tests pin the motion identity a merge rests on: every kept
joint keeps its position and global rotation. The real-data test checks the
same through the loader's canonical encode.
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
)
from data_loaders.truebones.truebones_utils.topology_relations import create_topology_edge_relations

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
_SPINE_1, _SPINE_2, _L_FOREARM, _R_FOREARM = 1, 2, 6, 9


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
        out, new, info = augment_topology(motion, entry, 0, rng=rng)
        assert info['applied']
        seen.add(info['kind'])
        assert out.shape[1] == len(new['parents']) < len(_PARENTS)
        assert len(new['parents']) >= topology_aug.LEAF_DROP_MIN_KEPT_JOINTS
        assert info['translation_root_index'] == 0
    assert seen == set(TOPOLOGY_AUG_KINDS)


def test_draw_falls_back_when_a_kind_has_nothing(monkeypatch):
    entry = _entry()
    motion = _motion(entry, {})
    rng = random.Random(1)
    monkeypatch.setattr(topology_aug, 'sample_leaf_drop', lambda *args, **kwargs: [])
    for _ in range(20):
        # No leaf is droppable: the draw lands on a merge.
        _out, _new, info = augment_topology(motion, entry, 0, rng=rng)
        assert info['kind'] == 'merge'


_MERGED_COND = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'dataset', 'cond_all.npy')
_needs_merged = pytest.mark.skipif(not os.path.isfile(_MERGED_COND), reason='merged dataset not present')


@_needs_merged
def test_kept_rows_decode_to_the_stored_motion():
    import data_loaders.truebones.data.dataset as dataset_module
    from data_loaders.get_data import get_dataset
    from data_loaders.truebones.truebones_utils.canonical_features import canonical_to_physical_hml
    from data_loaders.truebones.truebones_utils.param_utils import MAX_SOURCE_FRAMES_MULT

    motion_dataset = get_dataset(
        num_frames=60, split='train', objects_subset='all', action_group='all',
        topology_aug_prob=1.0, cond_path=_MERGED_COND,
    ).motion_dataset
    motion_dataset.opt.topology_aug_kinds = ('merge',)
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
            np.testing.assert_allclose(actual[..., :3], expected[..., :3], atol=1e-4)
            assert augmented[12]['joint_struct'].shape[0] == augmented[0].shape[1]
            checked += 1
    finally:
        dataset_module.augment_topology = real_augment
    assert checked > 0
