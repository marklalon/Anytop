"""A new skeleton's joint text the checkpoint never saw gets the blank name.

Training reads an all-zero name row as "unknown" (``--joint_name_drop_prob``) and
the blank text encodes to exactly that row, so blanking keeps an unseen name
inside the training distribution instead of encoding it to a T5 vector the
model has no prior for.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


from data_loaders.truebones.truebones_utils.joint_embedding_text import (  # noqa: E402
    _blank_unseen_joint_texts,
    _reference_joint_texts,
)
from data_loaders.truebones.truebones_utils.joint_embedding_text import (  # noqa: E402
    JOINT_NAME_EMBEDDING_SCHEMA_VERSION,
)


def _entry(texts, t5_name='t5-base', schema_version=JOINT_NAME_EMBEDDING_SCHEMA_VERSION):
    return {
        'joints_names_embs': np.zeros((len(texts), 4), dtype=np.float32),
        'joints_names_embs_meta': {
            't5_name': t5_name,
            'schema_version': schema_version,
            'embedding_texts': list(texts),
        },
    }


def test_reference_texts_pool_every_species():
    reference = {'a/Horse': _entry(['Spine', 'Left Calf']), 'b/Bird': _entry(['Left Wing', ''])}
    assert _reference_joint_texts(reference, 't5-base') == {'Spine', 'Left Calf', 'Left Wing', ''}


def test_reference_under_another_schema_is_refused():
    """Texts built by another text builder would call every name unseen."""
    reference = {'a/Horse': _entry(['Spine'], schema_version=JOINT_NAME_EMBEDDING_SCHEMA_VERSION - 1)}
    with pytest.raises(ValueError, match='schema'):
        _reference_joint_texts(reference, 't5-base')


def test_reference_without_matching_t5_texts_is_refused():
    with pytest.raises(ValueError):
        _reference_joint_texts({'a/Horse': _entry(['Spine'], t5_name='t5-large')}, 't5-base')
    with pytest.raises(ValueError):
        _reference_joint_texts(None, 't5-base')


def test_only_unseen_named_joints_are_blanked():
    texts = {'x/New': ['Spine', 'Left Tentacle', '', 'Left Calf', 'Left Tentacle']}
    blanked = _blank_unseen_joint_texts(texts, {'Spine', 'Left Calf', ''})
    assert texts['x/New'] == ['Spine', '', '', 'Left Calf', '']
    # The already-blank prop row is not reported as a blanked name.
    assert blanked == {'x/New': {1: 'Left Tentacle', 4: 'Left Tentacle'}}


def test_fully_covered_rig_is_untouched():
    texts = {'x/Horse': ['Spine', 'Left Calf']}
    assert _blank_unseen_joint_texts(texts, {'Spine', 'Left Calf'}) == {'x/Horse': {}}
    assert texts['x/Horse'] == ['Spine', 'Left Calf']
