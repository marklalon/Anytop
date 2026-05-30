from __future__ import annotations

import copy
import os
import sys

import numpy as np

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ANYTOP_ROOT = os.path.dirname(_TESTS_DIR)
_REPO_ROOT = os.path.dirname(_ANYTOP_ROOT)

for _path in [_REPO_ROOT, _ANYTOP_ROOT]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from utils.skeleton_similarity import (  # noqa: E402
    DEFAULT_WEIGHTS,
    lineage_tags,
    rank_species,
)


# ── lineage_tags helper ──────────────────────────────────────────────────────
def test_lineage_tags_are_case_insensitive() -> None:
    assert lineage_tags("Cat") == frozenset({"Mammal", "Felid"})
    assert lineage_tags("cat") == lineage_tags("CAT") == lineage_tags("Cat")


def test_lineage_overlap_is_graded() -> None:
    cat = lineage_tags("Cat")
    # Same family -> 2 shared tags; same clade only -> 1; unrelated -> 0.
    assert len(cat & lineage_tags("Lion")) == 2
    assert len(cat & lineage_tags("Horse")) == 1
    assert len(cat & lineage_tags("Eagle")) == 0


def test_newly_registered_species_reuse_existing_tags() -> None:
    assert lineage_tags("Monkey") == frozenset({"Mammal", "Biped"})
    assert lineage_tags("Skunk") == frozenset({"Mammal", "Canid"})
    assert lineage_tags("Pirrana") == frozenset({"Fish", "Snake"})


def test_sandmouse_is_a_rodent_not_a_felid() -> None:
    assert lineage_tags("SandMouse") == frozenset({"Mammal", "Rodent"})


def test_unregistered_species_has_no_tags() -> None:
    assert lineage_tags("Wombat") == frozenset()


# ── graded discount inside rank_species ──────────────────────────────────────
def _cond(parents, joints_names) -> dict:
    return {
        "parents": np.asarray(parents, dtype=np.int32),
        "joints_names": list(joints_names),
    }


def test_graded_lineage_discount_orders_and_scales_distance() -> None:
    # Query differs morphologically from the candidates (so the pre-discount
    # combined distance is > 0), while the three candidates are morphologically
    # identical to one another -> only their lineage relationship to the query
    # can differentiate them.
    query = _cond([-1, 0, 1], ["Hip", "RightThigh", "RightCalf"])
    template = _cond([-1, 0, 1, 2], ["Spine", "Neck", "Head", "Beak"])
    candidate_conds = {
        "Lion": copy.deepcopy(template),   # (Mammal, Felid)  -> overlap 2 with Cat
        "Horse": copy.deepcopy(template),  # (Mammal, Megafauna) -> overlap 1
        "Eagle": copy.deepcopy(template),  # (Flying, Bird)   -> overlap 0
    }

    ranked = rank_species(query, candidate_conds, query_hint="Cat", top_k=None)
    by_name = {r.name: r for r in ranked}

    # Closer lineage -> larger discount -> smaller distance -> earlier in order.
    assert [r.name for r in ranked] == ["Lion", "Horse", "Eagle"]

    # All three share the same pre-discount base, so the ratios of the final
    # combined distances must equal the ratios of the graded group factors.
    bonus = DEFAULT_WEIGHTS.group_bonus
    base = by_name["Eagle"].combined_distance          # factor 1.0
    assert base > 0
    assert np.isclose(by_name["Lion"].combined_distance, base * (1.0 - bonus))
    assert np.isclose(by_name["Horse"].combined_distance, base * (1.0 - bonus / 2.0))

    # same_group now means "same family" (full overlap).
    assert by_name["Lion"].same_group is True
    assert by_name["Horse"].same_group is False
    assert by_name["Eagle"].same_group is False


def test_unregistered_query_gets_no_discount() -> None:
    query = _cond([-1, 0, 1], ["Hip", "RightThigh", "RightCalf"])
    template = _cond([-1, 0, 1, 2], ["Spine", "Neck", "Head", "Beak"])
    candidate_conds = {
        "Lion": copy.deepcopy(template),
        "Eagle": copy.deepcopy(template),
    }

    ranked = rank_species(query, candidate_conds, query_hint="Wombat", top_k=None)
    # No lineage tags for the query -> overlap 0 everywhere -> no discount, so
    # the candidates tie on distance and fall back to name ordering.
    assert all(r.same_group is False for r in ranked)
    assert np.isclose(ranked[0].combined_distance, ranked[1].combined_distance)
