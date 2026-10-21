"""The precomputed species descriptor table and its closed vocabulary."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loaders.truebones.truebones_utils import dataset_tags as dt  # noqa: E402
from data_loaders.truebones.truebones_utils import species_descriptor_table as sdt  # noqa: E402
from sample import conditioning  # noqa: E402

DIM = 4


class _FakeT5:
    """Deterministic stand-in: a text's vector depends on the text alone."""

    def __init__(self):
        self.encoded = []

    def tokenize_entries(self, texts):
        self.encoded.extend(texts)
        return list(texts)

    def __call__(self, texts):
        rows = [
            np.random.default_rng(abs(hash(text)) % (2 ** 32)).standard_normal(DIM)
            for text in texts
        ]
        return torch.tensor(np.stack(rows), dtype=torch.float32)


def _table(descriptors=None):
    descriptors = tuple(descriptors or dt.species_descriptor_vocabulary())
    return sdt.SpeciesDescriptorTable(
        t5_name="t5-base",
        descriptors=descriptors,
        embeddings=np.arange(len(descriptors) * DIM, dtype=np.float32).reshape(-1, DIM),
        source="<test>",
    )


def _cond_entry(tags, t5_name="t5-base"):
    return {
        "species_tags": tuple(tags),
        "joints_names_embs_meta": {"t5_name": t5_name},
    }


def test_check_species_tags_rejects_words_outside_the_vocabulary():
    dt.check_species_tags(("Quadruped",), "ok")
    with pytest.raises(ValueError, match="body plan 'Robot'"):
        dt.check_species_tags(("Robot",), "x")
    dt.check_species_tags(("Biped", "Clanking"), "ok")
    with pytest.raises(ValueError, match="variant 'Striding'"):
        dt.check_species_tags(("Biped", "Striding"), "x")
    with pytest.raises(ValueError, match="body plan 'Clanking'"):
        dt.check_species_tags(("Clanking",), "x")
    with pytest.raises(ValueError, match=r"\[body-plan\] or \[body-plan, variant\]"):
        dt.check_species_tags(("Biped", "Clanking", "Waddling"), "x")


def test_vocabulary_is_every_body_plan_bare_and_with_each_variant():
    vocabulary = dt.species_descriptor_vocabulary()
    plans, variants = dt.SPECIES_BODY_PLANS, dt.SPECIES_VARIANTS
    assert len(vocabulary) == len(plans) * (1 + len(variants))
    assert vocabulary[:len(plans)] == tuple((body_plan,) for body_plan in plans)
    assert ("Aquatic",) in vocabulary
    assert ("Biped", "Clanking") in vocabulary


def test_build_encodes_every_descriptor_then_skips(tmp_path):
    fake = _FakeT5()
    out = tmp_path / "table.npy"
    table = sdt.build_species_descriptor_table(out, "t5-base", t5_conditioner=fake)
    assert set(table.descriptors) == set(dt.species_descriptor_vocabulary())
    assert len(fake.encoded) == len(table.descriptors)

    again = _FakeT5()
    reloaded = sdt.build_species_descriptor_table(out, "t5-base", t5_conditioner=again)
    assert again.encoded == []
    np.testing.assert_array_equal(reloaded.embeddings, table.embeddings)


def test_grown_vocabulary_only_encodes_new_rows(tmp_path, monkeypatch):
    out = tmp_path / "table.npy"
    table = sdt.build_species_descriptor_table(out, "t5-base", t5_conditioner=_FakeT5())
    monkeypatch.setattr(dt, "SPECIES_BODY_PLANS", dt.SPECIES_BODY_PLANS + ("Rolling",))
    fake = _FakeT5()
    grown = sdt.build_species_descriptor_table(out, "t5-base", t5_conditioner=fake)
    assert fake.encoded == ["Rolling"] + [f"Rolling {v}" for v in dt.SPECIES_VARIANTS]
    for tags in table.descriptors:
        np.testing.assert_array_equal(grown.lookup(tags, "t"), table.lookup(tags, "t"))


def test_lookup_refuses_a_missing_descriptor():
    table = _table([("Biped",)])
    np.testing.assert_array_equal(table.lookup("biped", "t"), np.arange(DIM))
    with pytest.raises(sdt.SpeciesDescriptorError, match="not in the precomputed"):
        table.lookup(("Quadruped",), "t")


def test_bind_sets_rows_and_refuses_unknown_descriptors():
    table = _table([("Biped",), ("Quadruped",)])
    cond = {"a/Horse": _cond_entry(("Quadruped",))}
    sdt.bind_cond_species_embs(cond, table, "t")
    np.testing.assert_array_equal(cond["a/Horse"]["species_emb"], table.lookup(("Quadruped",), "t"))
    assert cond["a/Horse"]["species_emb_meta"]["embedding_text"] == "Quadruped"
    # Binding the same dict again (a reused runtime cond) is fine.
    sdt.bind_cond_species_embs(cond, table, "t")

    cond["a/Snake"] = _cond_entry(("Serpentine",))
    with pytest.raises(sdt.SpeciesDescriptorError, match="a/Snake"):
        sdt.bind_cond_species_embs(cond, table, "t")


def test_bind_refuses_a_cond_with_a_baked_species_emb():
    table = _table([("Biped",)])
    old_entry = _cond_entry(("Biped",))
    old_entry.update({"species_emb": np.zeros(DIM),
                      "species_emb_meta": {"t5_name": "t5-base",
                                           "embedding_text": "Biped"}})
    old = {"a/Human": old_entry}
    with pytest.raises(sdt.SpeciesDescriptorError, match="regenerate"):
        sdt.bind_cond_species_embs(old, table, "t")


def test_bind_refuses_a_table_encoded_with_another_t5_before_mutating_cond():
    table = _table([("Biped",)])
    cond = {"a/Human": _cond_entry(("Biped",), t5_name="t5-large")}

    with pytest.raises(sdt.SpeciesDescriptorError, match="t5-base.*t5-large"):
        sdt.bind_cond_species_embs(cond, table, "checkpoint cond")

    assert "species_emb" not in cond["a/Human"]
    assert "species_emb_meta" not in cond["a/Human"]


def test_cond_t5_name_reads_the_joint_name_encoder():
    cond = {"a": {"joints_names_embs_meta": {"t5_name": "t5-base"}},
            "b": {"joints_names_embs_meta": {"t5_name": "t5-base"}}}
    assert sdt.cond_t5_name(cond) == "t5-base"
    cond["b"]["joints_names_embs_meta"]["t5_name"] = "t5-large"
    with pytest.raises(sdt.SpeciesDescriptorError):
        sdt.cond_t5_name(cond)


def test_species_tags_override_reads_the_table(monkeypatch):
    table = _table([("Biped",), ("Winged",)])
    model = SimpleNamespace(species_joint_cond=True)
    monkeypatch.setattr(conditioning, "unwrap_anytop_model", lambda m: m)
    cond = {"a/Dragon": {"species_emb_meta": {"embedding_text": "Winged"}}}

    args = SimpleNamespace(species_tags="biped")
    emb = conditioning._resolve_species_emb_override(args, model, cond, "a/Dragon", table)
    np.testing.assert_array_equal(emb, table.lookup(("Biped",), "t"))

    args = SimpleNamespace(species_tags="Quadruped")
    with pytest.raises(SystemExit, match="not in the precomputed"):
        conditioning._resolve_species_emb_override(args, model, cond, "a/Dragon", table)
