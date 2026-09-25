"""The process_new_skeleton embedding path never loads T5 and bakes no species vector."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import model.conditioners as conditioners  # noqa: E402
from data_loaders.truebones.truebones_utils import joint_embedding_text as jet  # noqa: E402


@pytest.fixture
def stub_text_builders(monkeypatch):
    texts = {"New": ["Hips", "Left Tentacle", ""]}
    monkeypatch.setattr(jet, "refresh_joint_metadata_in_object_cond", lambda entry: None)
    monkeypatch.setattr(jet, "build_joint_embedding_texts", lambda entry: list(texts[entry["object_type"]]))
    monkeypatch.setattr(jet, "build_joint_name_inspection_rows", lambda entry, texts: [])
    monkeypatch.setattr(jet, "assert_species_tags_cover", lambda keys: None)

    def no_t5(*_args, **_kwargs):
        raise AssertionError("T5 must not be loaded")

    monkeypatch.setattr(conditioners, "T5Conditioner", no_t5)


def test_blank_unseen_path_uses_cache_and_zero_blank(tmp_path, stub_text_builders):
    hips = np.arange(4, dtype=np.float32)
    reference = {
        "ref/Horse": {
            "joints_names_embs": hips[None],
            "joints_names_embs_meta": {
                "t5_name": "t5-base",
                "schema_version": jet.JOINT_NAME_EMBEDDING_SCHEMA_VERSION,
                "embedding_texts": ["Hips"],
            },
        }
    }
    cond = {"New": {"object_type": "New", "species_emb": np.ones(4), "species_emb_meta": {}}}

    jet.attach_t5_embeddings_to_cond(
        cond, str(tmp_path), embedding_cache_cond=reference,
        blank_unseen_joint_names=True, write_collision_report=False,
    )

    embs = cond["New"]["joints_names_embs"]
    np.testing.assert_array_equal(embs[0], hips)
    np.testing.assert_array_equal(embs[1:], np.zeros((2, 4), dtype=np.float32))
    assert cond["New"]["joints_names_embs_meta"]["blanked_unseen_texts"] == {1: "Left Tentacle"}
    assert "species_emb" not in cond["New"]
    assert "species_emb_meta" not in cond["New"]
