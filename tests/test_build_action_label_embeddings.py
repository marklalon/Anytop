"""The word-table builder's decision to skip or re-encode.

The encoder itself is not exercised here -- that needs a local T5 -- but the
question this file answers does not need one: given a table already on disk,
does the builder correctly decide whether it is the table the caller asked for?
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / 'tools') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'tools'))

import tools.build_action_label_embeddings as builder  # noqa: E402
from data_loaders.truebones.truebones_utils.action_label_conditioning_contract import (  # noqa: E402
    action_word_embedding_payload,
    load_action_conditioning_bundle,
)
from tests.action_label_test_utils import make_test_bundle  # noqa: E402


def _write_table(tmp_path, bundle):
    path = tmp_path / 'action_word_embeddings.npy'
    np.save(
        path,
        action_word_embedding_payload(bundle.word_embeddings, bundle.embedding_contract),
        allow_pickle=True,
    )
    return path


@pytest.fixture
def no_encoding(monkeypatch):
    """Fail loudly if a test that should skip reaches the encoder."""
    def refuse(*_args, **_kwargs):
        raise AssertionError('re-encoded when the table was already current')

    monkeypatch.setattr(builder, '_encode_vocabulary', refuse)


def _stub_material(monkeypatch, t5_hash):
    monkeypatch.setattr(
        builder, '_resolve_t5_material',
        lambda t5_model, t5_path: (Path(t5_path or '.models/t5-base'), t5_hash),
    )


def test_a_table_from_the_same_encoder_is_kept(tmp_path, monkeypatch, no_encoding):
    bundle = make_test_bundle()
    path = _write_table(tmp_path, bundle)
    _stub_material(monkeypatch, bundle.embedding_contract['t5_artifact_sha256'])

    builder.build_word_table(path, bundle.embedding_contract['t5_name'], None, force=False)
    assert load_action_conditioning_bundle(path).embedding_fingerprint == (
        bundle.embedding_fingerprint
    )


def test_a_table_from_other_weights_is_re_encoded(tmp_path, monkeypatch):
    """Same --t5-model name, different bytes on disk: the name is not the identity."""
    stale = make_test_bundle(seed=1)
    fresh = make_test_bundle(seed=2)
    path = _write_table(tmp_path, stale)
    _stub_material(monkeypatch, fresh.embedding_contract['t5_artifact_sha256'])
    monkeypatch.setattr(
        builder, '_encode_vocabulary',
        lambda *_args, **_kwargs: (fresh.word_embeddings, fresh.embedding_contract),
    )

    # The two contracts differ only in which weights they name, so matching on
    # t5_name alone kept the stale table here.
    assert stale.embedding_contract['t5_name'] == fresh.embedding_contract['t5_name']
    builder.build_word_table(path, fresh.embedding_contract['t5_name'], None, force=False)
    assert load_action_conditioning_bundle(path).embedding_fingerprint == (
        fresh.embedding_fingerprint
    )


def test_a_missing_t5_directory_is_reported_even_on_the_skip_path(tmp_path):
    """The encoder is resolved before anything decides to skip."""
    path = _write_table(tmp_path, make_test_bundle())
    with pytest.raises(FileNotFoundError, match='local T5 directory not found'):
        builder.build_word_table(
            path, 't5-base', str(tmp_path / 'no-such-t5'), force=False
        )
