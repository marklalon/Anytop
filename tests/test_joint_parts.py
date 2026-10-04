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
    """The contact prefill is cond's contact_joints minus Foot/Hand joints with children, rig for rig."""
    manifest = REPO_ROOT / 'dataset' / 'datasets.jsonl'
    if not manifest.is_file():
        pytest.skip('dataset manifest not present')
    from data_loaders.truebones.truebones_utils.cond_schema import load_cond
    from data_loaders.truebones.truebones_utils.dataset_sources import load_datasets_manifest
    from data_loaders.truebones.truebones_utils.joint_embedding_text import build_joint_embedding_texts

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
            parents = [int(parent) for parent in entry['parents']]

            texts = build_joint_embedding_texts(entry)

            def keeps_contact(i):
                if not {'foot', 'hand'} & set(texts[i].lower().split()):
                    return True
                return not any(
                    parent == i and proposal[names[child]]['part'] != 'helper'
                    for child, parent in enumerate(parents)
                )

            expected = [int(i) for i in sorted(entry['contact_joints']) if keeps_contact(int(i))]
            assert proposed == expected, key
            checked += 1
    if not checked:
        pytest.skip('no dataset cond with contact_joints present')


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
# Readers: dataset binding, generation rows, loss statistics
# ---------------------------------------------------------------------------

def test_bake_writes_the_row_into_the_entry():
    from data_loaders.truebones.truebones_utils.joint_parts import (
        JOINT_CONTACT_KEY,
        JOINT_PARTS_KEY,
        JOINT_PARTS_REVIEWED_KEY,
        JOINT_PARTS_SIG_KEY,
        bake_joint_parts,
        cond_contact_joints,
        has_joint_parts,
        joint_parts_row_signature,
        strip_joint_parts,
    )

    entry = _small_entry()
    row = _row(entry)
    assert not has_joint_parts(entry)
    with pytest.raises(JointPartsError, match='joint-parts-only'):
        cond_contact_joints(entry)
    bake_joint_parts(entry, row)
    names = entry['joints_names']
    assert entry[JOINT_PARTS_KEY].tolist() == [PART_IDS[row['joints'][n]['part']] for n in names]
    assert cond_contact_joints(entry) == np.flatnonzero(entry[JOINT_CONTACT_KEY]).tolist()
    assert cond_contact_joints(entry)
    assert entry[JOINT_PARTS_REVIEWED_KEY] is row['reviewed']
    # The signature follows what the bake takes, not provenance notes.
    sig = entry[JOINT_PARTS_SIG_KEY]
    noted = {**row, 'joints': {n: dict(j, why='edited note') for n, j in row['joints'].items()}}
    assert joint_parts_row_signature(noted, names) == sig
    flipped = {**row, 'joints': dict(row['joints'])}
    flipped['joints'][names[0]] = dict(row['joints'][names[0]], contact=1 - row['joints'][names[0]]['contact'])
    assert joint_parts_row_signature(flipped, names) != sig
    strip_joint_parts(entry)
    assert not has_joint_parts(entry) and JOINT_PARTS_SIG_KEY not in entry


def test_bake_refuses_a_stale_row():
    from data_loaders.truebones.truebones_utils.joint_parts import bake_joint_parts

    entry = _small_entry()
    row = _row(entry)
    renamed = dict(entry, joints_names=[f'{name}_v2' for name in entry['joints_names']])
    with pytest.raises(JointPartsError, match='another skeleton'):
        bake_joint_parts(renamed, row)


def test_output_rows_keep_a_matching_skeleton_and_replace_a_changed_one(tmp_path):
    from data_loaders.truebones.truebones_utils.joint_parts import (
        joint_parts_row,
        write_output_joint_parts,
    )

    entry = _small_entry()
    names, parents = entry['joints_names'], entry['parents']
    count = len(names)
    first = joint_parts_row('Synthetic', names, parents, part_ids=[PART_IDS['trunk']] * count,
                            contact=[0] * count, source='model', src='model',
                            part_prob=np.full((count, 11), 1 / 11), contact_prob=np.zeros(count))
    assert write_output_joint_parts(tmp_path, [first]) == []
    again = joint_parts_row('Synthetic', names, parents, part_ids=[PART_IDS['leg']] * count,
                            contact=[1] * count, source='model', src='model')
    # Same skeleton: the directory keeps the labelling its earlier motions saw.
    assert write_output_joint_parts(tmp_path, [again]) == []
    bound = load_joint_parts(tmp_path, 'Synthetic', entry)
    assert set(bound.part_ids.tolist()) == {PART_IDS['trunk']} and bound.source == 'model'
    assert 'part_prob' in read_joint_parts_sidecar(tmp_path / JOINT_PARTS_FILE)['Synthetic']['joints'][names[0]]

    renamed = [f'{name}_v2' for name in names]
    changed = joint_parts_row('Synthetic', renamed, parents, part_ids=[PART_IDS['leg']] * count,
                              contact=[1] * count, source='model', src='model')
    warnings = write_output_joint_parts(tmp_path, [changed])
    assert len(warnings) == 1 and 'another skeleton' in warnings[0]
    rebound = load_joint_parts(tmp_path, 'Synthetic', dict(entry, joints_names=renamed))
    assert set(rebound.part_ids.tolist()) == {PART_IDS['leg']}


def test_part_class_weights_are_inverse_sqrt_frequency_with_unit_mean():
    from data_loaders.truebones.truebones_utils.joint_parts import JOINT_PARTS, part_class_weights

    counts = np.zeros(len(JOINT_PARTS))
    counts[PART_IDS['trunk']], counts[PART_IDS['soft']] = 900, 100
    weights = np.asarray(part_class_weights(counts))
    assert weights[PART_IDS['soft']] == pytest.approx(3.0 * weights[PART_IDS['trunk']])
    assert float((counts / counts.sum() * weights).sum()) == pytest.approx(1.0)
    assert weights[PART_IDS['wing']] == 0.0


def test_retarget_contacts_fall_back_to_the_prefill_outside_every_dataset():
    from data_loaders.truebones.truebones_utils.joint_parts import (
        bake_joint_parts,
        prefill_contacts,
        retarget_target_contacts,
    )
    from data_loaders.truebones.truebones_utils.physics_joint_annotation import (
        rest_positions_from_offsets,
    )

    entry = _small_entry()
    rest = rest_positions_from_offsets(entry['offsets'], entry['parents'])
    heuristic = prefill_contacts(entry['joints_names'], entry['parents'], rest)
    # No baked annotation: a process_new_skeleton rig.
    assert retarget_target_contacts(entry) == heuristic
    row = _row(entry)
    first_foot = next(name for name in entry['joints_names'] if row['joints'][name]['contact'])
    row['joints'][first_foot] = dict(row['joints'][first_foot], contact=0, src='manual')
    bake_joint_parts(entry, row)
    annotated = retarget_target_contacts(entry)
    assert entry['joints_names'].index(first_foot) not in annotated
