"""Joint-part prefill: the rules, and the foot chains grounding reads off them.

The rules are pinned on small synthetic rigs, one per decision the prefill
makes that a name alone cannot (fore/hind, wings spelled as arms, multiped legs
spelled as arms, tentacles).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_loaders.truebones.truebones_utils.joint_name_canonical import (  # noqa: E402
    refresh_joint_metadata_in_object_cond,
)
from data_loaders.truebones.truebones_utils.joint_parts import (  # noqa: E402
    foot_chains,
    ground_floors,
    prefill_foot_chains,
    prefill_joint_parts,
    skeleton_entry,
)
from data_loaders.truebones.truebones_utils.physics_joint_annotation import (  # noqa: E402
    rest_positions_from_offsets,
)


def _entry(joints, species_tags, species='Synthetic'):
    """A cond entry from ``[(name, parent name or None, (x, y, z))]`` world positions."""
    names = [name for name, _, _ in joints]
    index = {name: i for i, name in enumerate(names)}
    parents = np.array([index[parent] if parent else -1 for _, parent, _ in joints], dtype=np.int64)
    positions = np.array([position for _, _, position in joints], dtype=np.float64)
    offsets = positions - np.where(parents[:, None] >= 0, positions[np.maximum(parents, 0)], 0.0)
    offsets[parents < 0] = positions[parents < 0]
    entry = {
        'joints_names': names,
        'parents': parents,
        'offsets': offsets,
        'species_name': species,
        'object_type': f'test/{species}',
        'species_tags': species_tags,
        'translation_root_index': 0,
    }
    refresh_joint_metadata_in_object_cond(entry)
    return entry


def _limb(side, x, top, z, names, y_top):
    """A straight limb hanging from ``top`` down to the ground at (x, ·, z)."""
    joints = []
    parent = top
    for k, name in enumerate(names):
        y = y_top * (1.0 - (k + 1) / len(names))
        joints.append((f'{side}_{name}', parent, (x, y, z)))
        parent = f'{side}_{name}'
    return joints


def _quadruped(fore_names, hind_names, tags=('Quadruped', 'Galloping')):
    joints = [
        ('Hips', None, (0.0, 0.5, -0.4)),
        ('Spine', 'Hips', (0.0, 0.5, 0.0)),
        ('Chest', 'Spine', (0.0, 0.5, 0.4)),
        ('Neck', 'Chest', (0.0, 0.7, 0.6)),
        ('Head', 'Neck', (0.0, 0.8, 0.8)),
        ('Tail1', 'Hips', (0.0, 0.5, -0.7)),
    ]
    for side, x in (('L', 0.15), ('R', -0.15)):
        joints += _limb(side, x, 'Chest', 0.4, fore_names, 0.5)
        joints += _limb(side, x, 'Hips', -0.4, hind_names, 0.5)
    return _entry(joints, tags)


def _parts(entry):
    proposal = prefill_joint_parts(entry)
    return {name: proposal[name]['part'] for name in entry['joints_names']}, proposal


def test_quadruped_forelegs_are_arms_by_front_word():
    entry = _quadruped(['FrontLeg1', 'FrontLeg2', 'FrontFoot'], ['BackLeg1', 'BackLeg2', 'BackFoot'])
    parts, _ = _parts(entry)
    assert parts['L_FrontLeg1'] == 'arm'
    assert parts['L_FrontFoot'] == 'hand'
    assert parts['L_BackLeg1'] == 'leg'
    assert parts['L_BackFoot'] == 'foot'
    assert parts['Head'] == 'head' and parts['Tail1'] == 'tail' and parts['Spine'] == 'trunk'


def test_quadruped_plain_legs_split_fore_hind_by_geometry():
    # Both pairs spelled "Leg": the pair attached ahead along the trunk is fore.
    joints = [
        ('Hips', None, (0.0, 0.5, -0.4)),
        ('Spine', 'Hips', (0.0, 0.5, 0.0)),
        ('Chest', 'Spine', (0.0, 0.5, 0.4)),
        ('Neck', 'Chest', (0.0, 0.7, 0.6)),
        ('Head', 'Neck', (0.0, 0.8, 0.8)),
    ]
    for side, x in (('L', 0.15), ('R', -0.15)):
        joints += _limb(side, x, 'Chest', 0.4, ['Leg1', 'Leg2', 'Foot1'], 0.5)
        joints += _limb(side, x, 'Hips', -0.4, ['Leg3', 'Leg4', 'Foot2'], 0.5)
    parts, _ = _parts(_entry(joints, ('Quadruped', 'Trotting')))
    assert parts['L_Leg1'] == 'arm' and parts['L_Foot1'] == 'hand'
    assert parts['L_Leg3'] == 'leg' and parts['L_Foot2'] == 'foot'


def test_multiped_walking_arm_is_a_leg_but_pincers_stay():
    joints = [
        ('Hips', None, (0.0, 0.3, 0.0)),
        ('Head', 'Hips', (0.0, 0.3, 0.3)),
    ]
    for side, x in (('L', 0.1), ('R', -0.1)):
        joints += _limb(side, x, 'Hips', 0.0, ['UpperArm', 'Forearm', 'Finger'], 0.3)
        joints += _limb(side, x, 'Hips', -0.2, ['Thigh', 'Calf', 'Toe'], 0.3)
        joints += [(f'{side}_Pincers1', 'Head', (x, 0.3, 0.45)), (f'{side}_Pincers2', f'{side}_Pincers1', (x, 0.3, 0.55))]
    entry = _entry(joints, ('Multiped', 'Scuttling'))
    parts, proposal = _parts(entry)
    assert proposal['L_Finger']['contact'] == 1
    assert parts['L_UpperArm'] == 'leg' and parts['L_Finger'] == 'foot'
    assert parts['L_Pincers1'] == 'hand'


def test_winged_species_arm_chain_is_the_wing():
    joints = [
        ('Hips', None, (0.0, 0.3, 0.0)),
        ('Spine', 'Hips', (0.0, 0.35, 0.1)),
        ('Head', 'Spine', (0.0, 0.4, 0.25)),
        ('L_UpperArm', 'Spine', (0.2, 0.35, 0.1)),
        ('L_Forearm', 'L_UpperArm', (0.4, 0.35, 0.1)),
        ('L_Finger', 'L_Forearm', (0.6, 0.35, 0.1)),
        ('R_UpperArm', 'Spine', (-0.2, 0.35, 0.1)),
        ('R_Forearm', 'R_UpperArm', (-0.4, 0.35, 0.1)),
        ('R_Finger', 'R_Forearm', (-0.6, 0.35, 0.1)),
    ]
    joints += _limb('L', 0.05, 'Hips', 0.0, ['Thigh', 'Foot'], 0.3)
    joints += _limb('R', -0.05, 'Hips', 0.0, ['Thigh', 'Foot'], 0.3)
    parts, _ = _parts(_entry(joints, ('Winged', 'Flapping')))
    assert {parts[name] for name in ('L_UpperArm', 'L_Forearm', 'L_Finger')} == {'wing'}
    assert parts['L_Thigh'] == 'leg' and parts['L_Foot'] == 'foot'


@pytest.mark.parametrize('tags, body_tentacle', [
    (('Aquatic', 'Undulating'), 'arm'),
    (('Drifting', 'Hovering'), 'soft'),
])
def test_tentacles_follow_where_they_hang(tags, body_tentacle):
    joints = [
        ('Body', None, (0.0, 0.5, 0.0)),
        ('Head', 'Body', (0.0, 0.8, 0.0)),
        ('Tentacle1', 'Head', (0.0, 0.9, 0.1)),
        ('Tentacle2', 'Body', (0.2, 0.2, 0.0)),
        ('Tentacle3', 'Tentacle2', (0.3, 0.0, 0.0)),
    ]
    parts, _ = _parts(_entry(joints, tags))
    assert parts['Tentacle1'] == 'head'
    assert parts['Tentacle2'] == body_tentacle and parts['Tentacle3'] == body_tentacle


def test_root_is_trunk_and_blank_child_inherits():
    joints = [
        ('Armature', None, (0.0, 0.0, 0.0)),
        ('Hips', 'Armature', (0.0, 0.5, 0.0)),
        ('Spine', 'Hips', (0.0, 0.6, 0.0)),
        ('Head', 'Spine', (0.0, 0.9, 0.0)),
        ('Bone02', 'Head', (0.0, 1.0, 0.0)),
    ]
    parts, proposal = _parts(_entry(joints, ('Biped', 'Striding')))
    assert parts['Armature'] == 'trunk' and proposal['Armature']['contact'] == 0
    assert parts['Bone02'] == 'head' and proposal['Bone02']['src'] == 'inherit'


def test_horselink_is_a_leg_segment_even_spelled_ankle():
    # A Biped bird leg: Thigh -> Calf -> HorseLink -> Foot -> Toe, toe on the ground.
    joints = [
        ('Bip01', None, (0.0, 0.0, 0.0)),
        ('Bip01_Pelvis', 'Bip01', (0.0, 0.6, 0.0)),
        ('Bip01_Spine', 'Bip01_Pelvis', (0.0, 0.7, 0.1)),
        ('Bip01_Head', 'Bip01_Spine', (0.0, 0.9, 0.3)),
    ]
    for side, x in (('L', 0.1), ('R', -0.1)):
        joints += [
            (f'Bip01_{side}_Thigh', 'Bip01_Pelvis', (x, 0.55, 0.0)),
            (f'Bip01_{side}_Calf', f'Bip01_{side}_Thigh', (x, 0.4, 0.05)),
            (f'Bip01_{side}_HorseLink', f'Bip01_{side}_Calf', (x, 0.2, -0.05)),
            (f'Bip01_{side}_Foot', f'Bip01_{side}_HorseLink', (x, 0.03, 0.0)),
            (f'Bip01_{side}_Toe0', f'Bip01_{side}_Foot', (x, 0.0, 0.08)),
        ]
    parts, proposal = _parts(_entry(joints, ('Biped', 'Walking')))
    assert parts['Bip01_L_HorseLink'] == 'leg'
    assert proposal['Bip01_L_HorseLink']['src'] == 'name'
    assert parts['Bip01_L_Calf'] == 'leg'
    assert parts['Bip01_L_Foot'] == 'foot' and parts['Bip01_L_Toe0'] == 'foot'


def test_feelers_are_soft_not_head():
    joints = [
        ('Body', None, (0.0, 0.3, 0.0)),
        ('Head', 'Body', (0.0, 0.35, 0.3)),
        ('L_Feeler1', 'Head', (0.05, 0.45, 0.4)),
        ('L_Feeler2', 'L_Feeler1', (0.1, 0.55, 0.5)),
        ('Jaw', 'Head', (0.0, 0.3, 0.38)),
        ('R_Shall1', 'Head', (-0.05, 0.4, 0.4)),
    ]
    parts, _ = _parts(_entry(joints, ('Multiped', 'Crawling')))
    assert parts['L_Feeler1'] == 'soft' and parts['L_Feeler2'] == 'soft'
    assert parts['R_Shall1'] == 'soft'
    assert parts['Head'] == 'head' and parts['Jaw'] == 'head'


def test_a_foot_joint_above_its_toes_is_not_a_contact():
    joints = [('Hips', None, (0.0, 0.5, 0.0))]
    for side, x in (('L', 0.1), ('R', -0.1)):
        joints += [
            (f'{side}_Thigh', 'Hips', (x, 0.45, 0.0)),
            (f'{side}_Calf', f'{side}_Thigh', (x, 0.25, 0.0)),
            (f'{side}_Foot', f'{side}_Calf', (x, 0.02, 0.0)),
            (f'{side}_Toe', f'{side}_Foot', (x, 0.0, 0.08)),
            (f'{side}_ToeEnd', f'{side}_Toe', (x, 0.0, 0.12)),
        ]
    joints += [('Spine', 'Hips', (0.0, 0.7, 0.0)), ('Head', 'Spine', (0.0, 0.9, 0.0))]
    _, proposal = _parts(_entry(joints, ('Biped', 'Walking')))
    assert proposal['L_Foot']['part'] == 'foot' and proposal['L_Foot']['contact'] == 0
    assert proposal['L_Toe']['contact'] == 1 and proposal['L_ToeEnd']['contact'] == 1


# ---------------------------------------------------------------------------
# Foot chains (grounding)
# ---------------------------------------------------------------------------

def test_foot_chains_are_connected_runs_of_foot_joints():
    parents = [-1, 0, 1, 2, 0, 4, 5, 5]
    parts = ['trunk', 'leg', 'foot', 'foot', 'leg', 'foot', 'foot', 'foot']
    assert foot_chains(parents, parts) == [[2, 3], [5, 6, 7]]
    assert foot_chains(parents, ['trunk'] * 8) == []


def test_hand_chains_ground_only_with_a_contact_joint():
    parents = [-1, 0, 1, 2, 0, 4, 5]
    parts = ['trunk', 'arm', 'hand', 'hand', 'leg', 'foot', 'foot']
    assert foot_chains(parents, parts) == [[5, 6]]
    assert foot_chains(parents, parts, [0] * 7) == [[5, 6]]
    assert foot_chains(parents, parts, [0, 0, 0, 1, 0, 0, 1]) == [[2, 3], [5, 6]]


def test_quadruped_grounds_on_fore_and_hind_hooves():
    entry = _quadruped(['FrontLeg1', 'FrontFoot'], ['BackLeg1', 'BackFoot'])
    chains = prefill_foot_chains(entry)
    names = entry['joints_names']
    assert sorted(names[j] for chain in chains for j in chain) == [
        'L_BackFoot', 'L_FrontFoot', 'R_BackFoot', 'R_FrontFoot']


def test_external_rig_entry_matches_the_cond_prefill():
    entry = _quadruped(['FrontLeg1', 'FrontFoot'], ['BackLeg1', 'BackFoot'])
    rest = rest_positions_from_offsets(entry['offsets'], entry['parents'])
    external = skeleton_entry(entry['joints_names'], entry['parents'], rest)
    # Without species tags there is no fore/hind split: every foot grounds.
    assert len(prefill_foot_chains(external)) == 4
    external['species_tags'] = entry['species_tags']
    assert prefill_foot_chains(external) == prefill_foot_chains(entry)


def test_ground_floors_take_each_foot_lowest_joint_else_the_lowest_joint():
    lowest = np.array([1.0, 0.5, 0.2, 0.3, 0.6, 0.1])
    np.testing.assert_allclose(ground_floors(lowest, [[2, 3], [4, 5]]), [0.2, 0.1])
    np.testing.assert_allclose(ground_floors(lowest, []), [0.1])
