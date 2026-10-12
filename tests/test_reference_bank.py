from __future__ import annotations

import os
import sys

import numpy as np
import pytest


_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ANYTOP_ROOT = os.path.dirname(_TESTS_DIR)
_REPO_ROOT = os.path.dirname(_ANYTOP_ROOT)

for _path in [_REPO_ROOT, _ANYTOP_ROOT]:
    if _path not in sys.path:
        sys.path.insert(0, _path)


import eval.motion_quality.reference_bank as reference_bank_mod
from eval.motion_quality.reference_bank import ReferenceCorpus


def _make_cond_entry(tags: tuple[str, str], joint_count: int = 2) -> dict:
    return {
        "parents": np.array([-1, 0], dtype=np.int32)[:joint_count],
        "offsets": np.zeros((joint_count, 3), dtype=np.float64),
        "joints_names": ["root", "RightThigh"][:joint_count],
        "species_tags": tuple(tags),
    }


def _make_corpus(cond_lookup: dict, clips_by_species: dict | None = None) -> ReferenceCorpus:
    """A corpus over an in-memory cond and clip index (no dataset on disk)."""
    corpus = object.__new__(ReferenceCorpus)
    corpus.sources = ()
    corpus.cond_lookup = dict(cond_lookup)
    corpus._clips_by_species = {
        species: {head: list(paths) for head, paths in by_head.items()}
        for species, by_head in (clips_by_species or {}).items()
    }
    corpus._profiles = {}
    corpus._motions = {}
    corpus._banks = {}
    return corpus


def test_reference_prior_words_are_the_head_words_of_the_action_label() -> None:
    derive = reference_bank_mod.reference_prior_words

    # Direction and secondary words do not select reference clips: a
    # direction is shared by every travelling action.
    assert derive("fly, forward") == ("fly",)
    assert derive("attack, bite, left") == ("attack",)
    assert derive("walk, run") == ("run", "walk")
    # Head order is spelling, not a different prior: the words come back in
    # vocabulary order (HEAD_VOCAB is alphabetical), whichever way they were written.
    assert derive("land, fly") == derive("fly, land") == ("fly", "land")
    assert derive("") == ()
    assert derive(None) == ()


def test_reference_prior_words_reject_what_generate_rejects() -> None:
    # A typo must fail, not silently narrow "wlak, run" to the run references.
    with pytest.raises(ValueError, match="controlled vocabulary"):
        reference_bank_mod.reference_prior_words("wlak, run")
    with pytest.raises(ValueError, match="no head word"):
        reference_bank_mod.reference_prior_words("forward")


def test_reference_bank_refuses_an_empty_label() -> None:
    corpus = _make_corpus({"horse": _make_cond_entry(("Quadruped", "Galloping"))})
    with pytest.raises(ValueError, match="names no head word"):
        corpus.build_bank("horse", "")


def test_clip_paths_match_on_head_words_and_list_a_transition_clip_once() -> None:
    corpus = _make_corpus(
        {"horse": _make_cond_entry(("Quadruped", "Galloping"))},
        {"horse": {"walk": ["walk.npy", "walk_run.npy"], "run": ["walk_run.npy", "run.npy"]}},
    )
    assert corpus.clip_paths(("walk",)) == {"horse": ["walk.npy", "walk_run.npy"]}
    assert corpus.clip_paths(("walk", "run")) == {"horse": ["walk.npy", "walk_run.npy", "run.npy"]}
    assert corpus.clip_paths(("idle",)) == {}


def test_reference_species_selection_accepts_external_query_cond() -> None:
    baseline_cond = {
        "horse": _make_cond_entry(("Quadruped", "Galloping")),
        "snake": _make_cond_entry(("Serpentine", "Slithering")),
    }
    query_cond = _make_cond_entry(("Serpentine", "Slithering"))
    corpus = _make_corpus(baseline_cond)

    selected = corpus.select_species(
        query_object_type="dragon",
        prior_words=("walk", "run"),
        paths_by_species={"horse": ["horse.npy"], "snake": ["snake.npy"]},
        top_k_species=1,
        min_reference_clips=1,
        query_cond=query_cond,
    )

    assert len(selected) == 1
    assert selected[0].name == "snake"
    assert selected[0].tag_distance == pytest.approx(0.0)
    assert selected[0].weight == pytest.approx(1.0)


def test_reference_species_selection_widens_to_the_clip_floor() -> None:
    cond = {
        "horse": _make_cond_entry(("Quadruped", "Galloping")),
        "deer": _make_cond_entry(("Quadruped", "Trotting")),
        "bear": _make_cond_entry(("Quadruped", "Lumbering")),
        "crow": _make_cond_entry(("Winged", "Flapping")),
    }
    corpus = _make_corpus(cond)
    paths = {"horse": ["h1"], "deer": ["d1", "d2"], "bear": ["b1", "b2", "b3"], "crow": ["c1"] * 10}

    # top_k=1 is the horse alone; the floor of 6 clips pulls in bear and deer
    # (nearest first; equally far, so by name) and stops before the crow.
    selected = corpus.select_species("horse", ("walk",), paths, top_k_species=1, min_reference_clips=6)
    assert [entry.name for entry in selected] == ["horse", "bear", "deer"]
    assert sum(entry.weight for entry in selected) == pytest.approx(1.0)
    assert selected[0].weight > selected[-1].weight

    # A floor already met by the top-k adds nothing.
    selected = corpus.select_species("horse", ("walk",), paths, top_k_species=2, min_reference_clips=1)
    assert [entry.name for entry in selected] == ["horse", "bear"]
