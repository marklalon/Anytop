"""The preflight ``joint_name_collision_report`` cannot do.

That scan compares a rig against *itself*, so a joint name that is unique, legal
and canonicalizes cleanly passes it while landing nowhere near the distribution
the checkpoint was trained on. Since the joint-name embedding is the only
per-joint identity signal the model has, such a token silently borrows another
body part's motion prior -- which is how a Mixamo "LeftLeg" hinged a knee
backwards while the byte-identical "LeftShin" rig walked correctly.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.truebones.truebones_utils.joint_name_support import (
    _shares_part_word,
    collect_joint_name_support_rows,
    write_joint_name_support_report,
)

_DIM = 16


def _direction(seed):
    """A deterministic near-orthogonal unit vector, so cosines are readable."""
    generator = np.random.default_rng(seed)
    vector = generator.standard_normal(_DIM).astype(np.float32)
    return vector / np.linalg.norm(vector)


_CALF = _direction(1)
_LEG = _direction(2)
_ARM = _direction(3)
_HIPS = _direction(4)


def _entry(texts, vectors, raw_names=None):
    return {
        'joints_names': list(raw_names or texts),
        'joints_names_embs': np.stack(vectors).astype(np.float32),
        'joints_names_embs_meta': {'embedding_texts': list(texts)},
    }


def _reference_cond():
    """A corpus where "Calf" is densely carried and "Leg" is nearly unheard of."""
    cond = {}
    for index in range(12):
        cond[f'zoo/Mammal{index}'] = _entry(['Hips', 'Left Calf'], [_HIPS, _CALF])
    for index in range(2):
        # The Hen and the Rat of the real corpus: the only two carriers.
        cond[f'zoo/Bird{index}'] = _entry(['Hips', 'Left Leg'], [_HIPS, _LEG])
    for index in range(4):
        cond[f'zoo/Arthropod{index}'] = _entry(['Hips', 'Left Arm'], [_HIPS, _ARM])
    return cond


def test_a_densely_carried_token_is_low_risk():
    rows = collect_joint_name_support_rows(
        {'new/Rig': _entry(['Left Calf'], [_CALF])}, _reference_cond()
    )
    assert len(rows) == 1
    assert rows[0]['risk'] == 'low', rows[0]
    assert rows[0]['support_species_count'] == 12


def test_a_token_two_species_carry_is_flagged_high_with_its_nearest_other():
    rows = collect_joint_name_support_rows(
        {'new/Rig': _entry(['Left Leg'], [_LEG])}, _reference_cond()
    )
    assert len(rows) == 1
    row = rows[0]
    assert row['risk'] == 'high', row
    assert row['support_species_count'] == 2
    # The report has to name what the model will fall back on -- "2 species" on
    # its own does not tell a reader that the fallback is a different body part.
    assert row['nearest_other'], row
    assert row['nearest_other_cos'] < 0.85, row


def test_blanked_prop_joints_are_not_scanned():
    rows = collect_joint_name_support_rows(
        {'new/Rig': _entry(['Left Calf', ''], [_CALF, _HIPS], raw_names=['LeftCalf', 'Sword'])},
        _reference_cond(),
    )
    assert [row['embedding_text'] for row in rows] == ['Left Calf']


def test_report_puts_the_worst_supported_joint_first(tmp_path):
    cond = {'new/Rig': _entry(
        ['Hips', 'Left Calf', 'Left Leg'],
        [_HIPS, _CALF, _LEG],
        raw_names=['Hips', 'LeftShin', 'LeftLeg'],
    )}
    rows = write_joint_name_support_report(cond, str(tmp_path), _reference_cond())

    assert rows[0]['embedding_text'] == 'Left Leg', rows
    assert rows[0]['risk'] == 'high'

    report = json.loads((tmp_path / 'joint_name_support_report.json').read_text(encoding='utf-8'))
    assert report['num_joints_scanned'] == 3
    assert report['num_high_risk'] == 1
    assert report['joints'][0]['raw_name'] == 'LeftLeg'


def test_an_empty_reference_costs_the_report_not_the_run(tmp_path):
    rows = write_joint_name_support_report(
        {'new/Rig': _entry(['Left Calf'], [_CALF])}, str(tmp_path), {}
    )
    assert rows == []
    assert not (tmp_path / 'joint_name_support_report.json').exists()


def _unka_rig(index):
    """A rig whose distinctive tokens no other species in the corpus carries."""
    texts = ['Hips', 'Left Calf', 'Left Horn', 'Left Wing Finger'] + [
        f'Spine {position}' for position in range(index, index + 6)
    ]
    # Seeded by position, not by ``hash``: string hashing is salted per process.
    return _entry(texts, [_direction(100 + offset) for offset in range(len(texts))])


def test_a_rig_the_corpus_already_carries_is_not_warned_about():
    """The re-import case: kinship, not species count, is the right question.

    A creature already in the training set comes back through preprocessing with
    tokens only it carries. Counting *other* species that carry them measures a
    cross-species transfer nobody is asking for -- the prior for those joints was
    fitted on this very geometry.
    """
    reference = _reference_cond()
    reference['unitybundles/MB_Unka'] = _unka_rig(0)
    reference['unitybundles/MB_Unka']['loop_period_by_action'] = {f'a{i}': 1.0 for i in range(9)}

    rows = collect_joint_name_support_rows({'new/Dragon': _unka_rig(0)}, reference)

    horn = next(row for row in rows if row['embedding_text'] == 'Left Horn')
    assert horn['support_species_count'] == 1, horn
    assert horn['risk'] == 'low', horn
    assert horn['reason'] == 'sibling_rig', horn
    assert horn['sibling_support_species'] == ['unitybundles/MB_Unka'], horn


def test_kinship_is_decided_per_token_not_per_rig():
    """A near-twin rig only vouches for the tokens it actually carries."""
    reference = _reference_cond()
    reference['unitybundles/MB_Unka'] = _unka_rig(0)
    reference['unitybundles/MB_Unka']['loop_period_by_action'] = {f'a{i}': 1.0 for i in range(9)}

    query = _unka_rig(0)
    query['joints_names_embs_meta']['embedding_texts'][2] = 'Left Leg'
    query['joints_names_embs'][2] = _LEG
    query['joints_names'] = list(query['joints_names_embs_meta']['embedding_texts'])

    rows = collect_joint_name_support_rows({'new/Dragon': query}, reference)

    leg = next(row for row in rows if row['embedding_text'] == 'Left Leg')
    assert leg['sibling_support_species'] == [], leg
    assert leg['risk'] == 'high', leg
    assert leg['reason'] == 'narrow', leg


def test_a_rig_the_corpus_barely_animates_vouches_for_nothing():
    """Matching a species with no clips behind it is not evidence of training."""
    reference = _reference_cond()
    reference['unitybundles/MB_Unka'] = _unka_rig(0)
    reference['unitybundles/MB_Unka']['loop_period_by_action'] = {'idle': 1.0}

    rows = collect_joint_name_support_rows({'new/Dragon': _unka_rig(0)}, reference)

    horn = next(row for row in rows if row['embedding_text'] == 'Left Horn')
    assert horn['sibling_support_species'] == [], horn
    assert horn['risk'] == 'high', horn


def test_an_unseen_token_is_judged_by_the_body_part_it_lands_on():
    """The docstring's own distinction, made real.

    An unseen finger landing on another finger is fine; a leg landing on an arm
    is the failure this module exists to catch. Side words and the canonicaliser's
    modifier vocabulary carry no part identity, so they cannot make two tokens kin.
    """
    assert _shares_part_word('Left Wing Finger', 'Left Arm Finger')
    assert _shares_part_word('Chest Extra', 'Chest')
    assert not _shares_part_word('Left Leg', 'Left Arm')
    assert not _shares_part_word('Left Arm Twist', 'Left Leg Twist')
    assert not _shares_part_word('Left Leg Back', 'Left Arm Back')
