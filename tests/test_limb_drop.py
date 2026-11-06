"""Whole-chain removal, independent eligibility rules and source semantics."""

import copy
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loaders.truebones.truebones_utils import joint_parts, topology_aug
from data_loaders.truebones.truebones_utils.leaf_drop import _guarded_chain_roots, drop_joints_from_cond
from data_loaders.truebones.truebones_utils.limb_drop import limb_drop_candidates
from data_loaders.truebones.truebones_utils.rest_repose_aug import apply_rest_rotations
from data_loaders.truebones.truebones_utils.rest_variants import apply_rest_variant
from tests.test_topology_aug import _entry, _motion


def rig(legs=0, arms=0, wings=0, extras=(), tail_support=False):
    # Ten body joints let every eligible subtree/pair pass the minimum.
    parts = ['trunk'] * 10
    if tail_support:
        parts[9] = 'arm'
    parents = [-1] + list(range(9))
    sides = ['center'] * 10
    partners = [-1] * 10
    chains = []
    for kind, number in [('leg', legs), ('arm', arms), ('wing', wings), *extras]:
        current = []
        for i in range(number):
            root = len(parts)
            distal = {'leg': 'foot', 'arm': 'hand'}.get(kind, kind)
            parts.extend([kind, distal, distal])
            parents.extend([0, root, root + 1])
            sides.extend(['left' if i % 2 == 0 else 'right'] * 3)
            partners.extend([-1] * 3)
            current.append(tuple(range(root, root + 3)))
            if i % 2:
                for a, b in zip(current[-2], current[-1]):
                    partners[a], partners[b] = b, a
        chains.extend(current)
    count = len(parts)
    return dict(parents=np.asarray(parents), offsets=np.ones((count, 3), np.float32),
                joint_parts=tuple(parts), joints_names=[f'joint{i}' for i in range(count)],
                joint_side_labels=sides, symmetry_partner_indices=partners,
                translation_root_index=0), chains


def candidates(entry):
    return limb_drop_candidates(entry, topology_aug._indexed_joints(entry, 0))


@pytest.mark.parametrize('kind', ['tail', 'soft', 'helper'])
@pytest.mark.parametrize('number', [1, 2, 4])
def test_peripheral_chains_are_whole_single_or_mirror_units(kind, number):
    entry, chains = rig(extras=[(kind, number)], tail_support=kind == 'tail')
    expected = [tuple(j for chain in chains[i:i + 2] for j in chain)
                for i in range(0, number, 2)]
    assert candidates(entry) == expected


@pytest.mark.parametrize('support', ['trunk', 'fin', 'wing', 'hand', 'foot', 'arm', 'leg'])
def test_tail_requires_an_arm_or_leg_part(support):
    entry, chains = rig(extras=[('tail', 2)])
    labels = list(entry['joint_parts'])
    labels[9] = support
    entry['joint_parts'] = tuple(labels)
    expected = [chains[0] + chains[1]] if support in ('arm', 'leg') else []
    assert candidates(entry) == expected
    motion = np.zeros((2, len(labels), 12), np.float32)
    out, new, info = topology_aug.augment_topology(motion, entry, 0, kinds=('limb_drop',))
    assert info['applied'] == bool(expected)
    if not expected:
        assert out is motion and new is entry


@pytest.mark.parametrize('legs', [2, 4, 5, 6, 8])
def test_leg_threshold_counts_individual_chains(legs):
    entry, chains = rig(legs=legs)
    units = candidates(entry)
    assert len(units) == (legs // 2 if legs >= 6 else 0)
    for unit in units:
        assert len(unit) == 6
        assert all(entry['joint_parts'][j] in ('leg', 'foot') for j in unit)


@pytest.mark.parametrize('legs,arms,wings,allowed', [
    (2, 2, 2, {'arm', 'wing'}), (6, 2, 2, {'leg', 'arm', 'wing'}),
    (6, 0, 2, {'leg'}), (2, 0, 2, set()), (0, 2, 2, set()), (2, 2, 0, set()),
])
def test_coexistence_rule_is_independent_of_leg_threshold(legs, arms, wings, allowed):
    entry, _ = rig(legs=legs, arms=arms, wings=wings)
    assert {entry['joint_parts'][unit[0]] for unit in candidates(entry)} == allowed
    for seed in range(20):
        motion = np.zeros((2, len(entry['parents']), 12), np.float32)
        _, _, info = topology_aug.augment_topology(
            motion, entry, 0, kinds=('limb_drop',), rng=random.Random(seed))
        if allowed:
            assert info['applied'] and len(info['joints']) == 6
            families = {entry['joint_parts'][j] for j in info['joints']}
            assert len(families & {'leg', 'arm', 'wing'}) == 1
        else:
            assert not info['applied']


def test_branches_and_attachments_go_with_limb():
    entry, chains = rig(legs=6)
    # Two feet branch below each leg; soft decorations belong to that limb.
    for chain in chains:
        entry['parents'][chain[2]] = chain[0]
    labels = list(entry['joint_parts'])
    labels[chains[0][2]] = labels[chains[1][2]] = 'soft'
    entry['joint_parts'] = tuple(labels)
    assert tuple(chains[0] + chains[1]) in candidates(entry)
    assert len([unit for unit in candidates(entry) if entry['joint_parts'][unit[0]] == 'leg']) == 3


@pytest.mark.parametrize('part', ['leg', 'trunk', 'head', 'wing', 'fin'])
def test_auxiliary_subtree_cannot_swallow_another_body_part(part):
    entry, chains = rig(extras=[('helper', 1)])
    labels = list(entry['joint_parts'])
    labels[chains[0][-1]] = part
    entry['joint_parts'] = tuple(labels)
    assert not candidates(entry)


@pytest.mark.parametrize('key', ['translation_root_index', 'forward_joint_index',
                                 'forward_base_joint_index', 'face_joints'])
def test_protected_joint_disqualifies_entire_pair(key):
    entry, chains = rig(extras=[('tail', 2)], tail_support=True)
    joint = chains[0][-1]
    entry[key] = [joint] if key == 'face_joints' else joint
    assert not candidates(entry)


def test_minimum_filters_units_before_sampling(monkeypatch):
    entry, _ = rig(extras=[('tail', 2), ('soft', 1)], tail_support=True)
    # Pair deletion leaves 13 joints, single deletion leaves 16.
    assert len(candidates(entry)) == 2
    import data_loaders.truebones.truebones_utils.limb_drop as module
    monkeypatch.setattr(module, 'LEAF_DROP_MIN_KEPT_JOINTS', 14)
    units = candidates(entry)
    assert len(units) == 1 and entry['joint_parts'][units[0][0]] == 'soft'
    monkeypatch.setattr(module, 'LEAF_DROP_MIN_KEPT_JOINTS', 17)
    assert not candidates(entry)


def test_partial_mirror_annotation_pairs_by_distal_joint():
    entry, chains = rig(extras=[('tail', 2)], tail_support=True)
    for chain in chains:
        entry['symmetry_partner_indices'][chain[0]] = -1
    assert candidates(entry) == [chains[0] + chains[1]]


def test_one_way_mirror_annotation_never_drops_one_side_alone():
    entry, chains = rig(extras=[('tail', 2)], tail_support=True)
    for j in chains[0]:
        entry['symmetry_partner_indices'][j] = -1
    assert candidates(entry) == [chains[0] + chains[1]]


def test_out_of_range_mirror_annotation_skips_pair():
    entry, chains = rig(extras=[('tail', 2)], tail_support=True)
    entry['symmetry_partner_indices'][chains[0][0]] = 999
    assert not candidates(entry)


def test_conflicting_mirrors_skip_instead_of_removing_three_chains():
    entry, chains = rig(extras=[('tail', 4)], tail_support=True)
    a, b = chains[0][-1], chains[2][-1]
    entry['symmetry_partner_indices'][a] = b
    entry['symmetry_partner_indices'][b] = a
    assert not candidates(entry)


def test_shared_midline_wing_connector_is_kept():
    entry, chains = rig(legs=2, arms=2, wings=2)
    labels = list(entry['joint_parts'])
    labels[9] = 'wing'
    entry['joint_parts'] = tuple(labels)
    for chain in chains[-2:]:
        entry['parents'][chain[0]] = 9
    wing = [unit for unit in candidates(entry) if entry['joint_parts'][unit[0]] == 'wing']
    assert wing == [chains[-2] + chains[-1]]


def test_deletion_preserves_motion_remaps_metadata_and_does_not_mutate_source():
    entry = _entry()
    entry['joint_parts'] = joint_parts.joint_part_labels(entry)
    entry['rest_length_scale'] = 99.
    source = copy.deepcopy(entry)
    motion = _motion(entry, {})
    out, new, info = topology_aug.augment_topology(motion, entry, 0, kinds=('limb_drop',))
    assert info['applied'] and info['joints'] == [11, 12]
    keep = np.flatnonzero(info['new_index'] >= 0)
    np.testing.assert_array_equal(out, motion[:, keep])
    assert new['joint_parts'] == tuple(entry['joint_parts'][j] for j in keep)
    assert 'rest_length_scale' not in new
    np.testing.assert_array_equal(new['parents'], entry['parents'][keep])
    assert new['joint_relations'].shape == (len(keep), len(keep))
    assert new['joints_graph_dist'].shape == (len(keep), len(keep))
    assert new['symmetry_partner_indices'] == entry['symmetry_partner_indices'][:11]
    for key in source:
        if isinstance(source[key], np.ndarray):
            np.testing.assert_array_equal(entry[key], source[key])
        else:
            assert entry[key] == source[key]


def test_source_labels_survive_rest_variants_and_repose_without_reclassification(monkeypatch):
    entry = _entry()
    labels = joint_parts.joint_part_labels(entry)
    entry['joint_parts'] = labels
    motion = _motion(entry, {})
    roots = _guarded_chain_roots(entry)
    units = candidates(entry)
    rotations = np.broadcast_to(np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]]),
                                (len(labels), 3, 3)).copy()
    entry['rest_variants'] = {'tags': ['other'], 'rotations': rotations[None]}

    def unexpected(*args, **kwargs):
        pytest.fail('Stored source labels must prevent reclassification')

    monkeypatch.setattr(joint_parts, 'prefill_joint_parts', unexpected)
    _, variant, _ = apply_rest_variant(motion, entry, 0)
    _, reposed = apply_rest_rotations(motion, entry, rotations)
    for changed in (variant, reposed):
        assert changed['joint_parts'] is labels
        assert joint_parts.joint_part_labels(changed) is labels
        assert _guarded_chain_roots(changed) == roots
        assert candidates(changed) == units
        joint_parts.prefill_limb_end_chains(changed)
        keep, dropped = drop_joints_from_cond(changed, [11, 12])
        assert joint_parts.joint_part_labels(dropped) == tuple(labels[j] for j in keep)


def test_limb_failure_falls_back_in_random_order():
    entry = _entry()
    entry['joint_parts'] = ('trunk',) * len(entry['parents'])
    motion = _motion(entry, {})

    class FirstLimb(random.Random):
        def shuffle(self, order):
            super().shuffle(order)
            if 'limb_drop' in order:
                order.remove('limb_drop')
                order.insert(0, 'limb_drop')

    seen = set()
    for seed in range(30):
        _, _, info = topology_aug.augment_topology(motion, entry, 0, rng=FirstLimb(seed))
        assert info['applied']
        seen.add(info['kind'])
    assert seen == {'leaf_drop', 'merge'}
    _, _, info = topology_aug.augment_topology(motion, entry, 0, kinds=('limb_drop',))
    assert not info['applied']
