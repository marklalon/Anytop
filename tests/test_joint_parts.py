"""Joint-part annotation: the prefill rules, and the sidecar contract readers rely on.

The rules are pinned on small synthetic rigs, one per decision the prefill
makes that a name alone cannot (fore/hind, wings spelled as arms, multiped legs
spelled as arms, tentacles); the sidecar tests pin what a reader may assume --
binding by name, refusing stale or partial rows, and a prefill never
overwriting a person's edit.
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
    HELPER_PART_ID,
    JOINT_PARTS_FILE,
    PART_IDS,
    JointPartsError,
    bind_joint_parts,
    load_joint_parts,
    merge_prefill,
    prefill_joint_parts,
    read_joint_parts_sidecar,
    skeleton_signature,
    write_joint_parts_sidecar,
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


def test_unnamed_wrapper_root_is_helper_and_blank_child_inherits():
    joints = [
        ('Armature', None, (0.0, 0.0, 0.0)),
        ('Hips', 'Armature', (0.0, 0.5, 0.0)),
        ('Spine', 'Hips', (0.0, 0.6, 0.0)),
        ('Head', 'Spine', (0.0, 0.9, 0.0)),
        ('Bone02', 'Head', (0.0, 1.0, 0.0)),
    ]
    parts, proposal = _parts(_entry(joints, ('Biped', 'Striding')))
    assert parts['Armature'] == 'helper' and proposal['Armature']['contact'] == 0
    assert parts['Bone02'] == 'head' and proposal['Bone02']['src'] == 'inherit'


# ---------------------------------------------------------------------------
# Sidecar contract
# ---------------------------------------------------------------------------

def _row(entry, reviewed=False):
    proposal = prefill_joint_parts(entry)
    return {
        'species': entry['species_name'],
        'skeleton_sig': skeleton_signature(entry['joints_names'], entry['parents']),
        'reviewed': reviewed,
        'joints': proposal,
    }


def _small_entry():
    return _quadruped(['FrontLeg1', 'FrontFoot'], ['BackLeg1', 'BackFoot'])


def test_sidecar_round_trip_and_bind(tmp_path):
    entry = _small_entry()
    row = _row(entry)
    write_joint_parts_sidecar(tmp_path / JOINT_PARTS_FILE, [row])
    assert read_joint_parts_sidecar(tmp_path / JOINT_PARTS_FILE) == {row['species']: row}
    bound = load_joint_parts(tmp_path, row['species'], entry)
    names = entry['joints_names']
    assert bound.part_ids.tolist() == [PART_IDS[row['joints'][name]['part']] for name in names]
    assert bound.contact_joints == [i for i, name in enumerate(names) if row['joints'][name]['contact']]
    assert bound.contact_joints, 'the feet stand on the ground'


def test_bind_follows_names_not_row_order():
    entry = _small_entry()
    row = _row(entry)
    row['joints'] = dict(reversed(list(row['joints'].items())))
    bound = bind_joint_parts(row, entry['joints_names'], entry['parents'])
    assert bound.part_ids[entry['joints_names'].index('Head')] == PART_IDS['head']


def test_bind_refuses_stale_and_partial_rows():
    entry = _small_entry()
    row = _row(entry)
    with pytest.raises(JointPartsError, match='another skeleton'):
        bind_joint_parts(row, entry['joints_names'][:-1], entry['parents'][:-1])
    partial = dict(row, joints={k: v for k, v in row['joints'].items() if k != 'Head'})
    with pytest.raises(JointPartsError, match='no entry'):
        bind_joint_parts(partial, entry['joints_names'], entry['parents'])


def test_sidecar_rejects_contact_on_helper(tmp_path):
    entry = _small_entry()
    row = _row(entry)
    row['joints']['Head'] = {'part': 'helper', 'contact': 1, 'src': 'manual', 'why': ''}
    with pytest.raises(JointPartsError, match='helper'):
        write_joint_parts_sidecar(tmp_path / JOINT_PARTS_FILE, [row])


def test_helper_binds_to_helper_id():
    entry = _small_entry()
    row = _row(entry)
    row['joints']['Tail1'] = {'part': 'helper', 'contact': 0, 'src': 'manual', 'why': ''}
    bound = bind_joint_parts(row, entry['joints_names'], entry['parents'])
    assert bound.part_ids[entry['joints_names'].index('Tail1')] == HELPER_PART_ID


def test_merge_keeps_manual_joints_and_reviewed_rows():
    entry = _small_entry()
    sig = skeleton_signature(entry['joints_names'], entry['parents'])
    proposal = prefill_joint_parts(entry)
    old = _row(entry)
    old['joints']['Tail1'] = {'part': 'soft', 'contact': 0, 'src': 'manual', 'why': 'person'}
    old['joints']['Head'] = {'part': 'neck', 'contact': 0, 'src': 'name', 'why': 'stale proposal'}

    row, changed = merge_prefill(old, proposal, old['species'], sig)
    assert row['joints']['Tail1']['src'] == 'manual'
    assert row['joints']['Head']['part'] == 'head'
    assert changed == ['Head']

    reviewed = dict(old, reviewed=True)
    row, changed = merge_prefill(reviewed, proposal, old['species'], sig)
    assert row['reviewed'] and row['joints']['Head']['part'] == 'neck' and not changed


def test_merge_of_stale_row_drops_reviewed_and_keeps_surviving_manual():
    entry = _small_entry()
    old = dict(_row(entry), reviewed=True, skeleton_sig='0' * 16)
    old['joints']['Tail1'] = {'part': 'soft', 'contact': 0, 'src': 'manual', 'why': 'person'}
    old['joints']['Removed'] = {'part': 'soft', 'contact': 0, 'src': 'manual', 'why': 'gone'}
    sig = skeleton_signature(entry['joints_names'], entry['parents'])
    row, changed = merge_prefill(old, prefill_joint_parts(entry), old['species'], sig)
    assert not row['reviewed'] and row['skeleton_sig'] == sig
    assert row['joints']['Tail1']['part'] == 'soft'
    assert 'Removed' not in row['joints'] and 'Removed' in changed


# ---------------------------------------------------------------------------
# Real datasets
# ---------------------------------------------------------------------------

def test_prefill_contacts_match_cond_on_real_datasets():
    """The contact prefill reproduces cond's contact_joints, rig for rig."""
    manifest = REPO_ROOT / 'dataset' / 'datasets.jsonl'
    if not manifest.is_file():
        pytest.skip('dataset manifest not present')
    from data_loaders.truebones.truebones_utils.cond_schema import load_cond
    from data_loaders.truebones.truebones_utils.dataset_sources import load_datasets_manifest

    checked = 0
    for source in load_datasets_manifest(manifest):
        if not Path(source.cond_path).is_file():
            continue
        for key, entry in load_cond(source.cond_path).items():
            if 'contact_joints' not in entry:
                continue
            proposal = prefill_joint_parts(entry)
            names = entry['joints_names']
            assert all(proposal[name]['part'] for name in names), key
            proposed = [i for i, name in enumerate(names) if proposal[name]['contact']]
            assert proposed == sorted(int(i) for i in entry['contact_joints']), key
            checked += 1
    if not checked:
        pytest.skip('no dataset cond with contact_joints present')
