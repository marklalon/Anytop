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
