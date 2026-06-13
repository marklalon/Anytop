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
    group_tags,
    rank_species,
)


# ── group_tags helper ────────────────────────────────────────────────────────
def test_group_tags_are_case_insensitive() -> None:
    assert group_tags("Cat") == frozenset({"Quadruped", "Small", "Stalking"})
    assert group_tags("cat") == group_tags("CAT") == group_tags("Cat")


def test_group_overlap_is_graded() -> None:
    jaguar = group_tags("Jaguar")
    # Identical mover (Jaguar/Lynx share all 3 tags) -> 3;
    # same body-plan + one trait (Cat/Lion: Quadruped + Stalking) -> 2;
    # unrelated (Cat/Eagle) -> 0.
    assert len(jaguar & group_tags("Lynx")) == 3
    assert len(group_tags("Cat") & group_tags("Lion")) == 2  # {Quadruped, Stalking}
    assert len(group_tags("Cat") & group_tags("Eagle")) == 0


def test_newly_registered_species_reuse_existing_tags() -> None:
    assert group_tags("Monkey") == frozenset({"Quadruped", "Medium", "Climbing"})
    assert group_tags("Skunk") == frozenset({"Quadruped", "Small", "Scurrying"})
    assert group_tags("Pirrana") == frozenset({"Aquatic", "Small", "Swimming"})


def test_sandmouse_is_a_small_scurrier() -> None:
    assert group_tags("SandMouse") == frozenset({"Quadruped", "Small", "Scurrying"})


def test_unregistered_species_has_no_tags() -> None:
    assert group_tags("Wombat") == frozenset()


# ── graded discount inside rank_species ──────────────────────────────────────
def _cond(parents, joints_names) -> dict:
    return {
        "parents": np.asarray(parents, dtype=np.int32),
        "joints_names": list(joints_names),
    }


def test_graded_group_discount_orders_and_scales_distance() -> None:
    # Query differs morphologically from the candidates (so the pre-discount
    # combined distance is > 0), while the three candidates are morphologically
    # identical to one another -> only their motion-tag relationship to the
    # query can differentiate them.
    query = _cond([-1, 0, 1], ["Hip", "RightThigh", "RightCalf"])
    template = _cond([-1, 0, 1, 2], ["Spine", "Neck", "Head", "Beak"])
    candidate_conds = {
        # Cat -> (Quadruped, Small, Stalking)
        "Lion": copy.deepcopy(template),   # (Quadruped, Medium, Stalking)  -> overlap 2
        "Horse": copy.deepcopy(template),  # (Quadruped, Large, Galloping)  -> overlap 1
        "Eagle": copy.deepcopy(template),  # (Winged, Medium, Soaring)      -> overlap 0
    }

    ranked = rank_species(query, candidate_conds, query_hint="Cat", top_k=None)
    by_name = {r.name: r for r in ranked}

    # Closer motion group -> larger discount -> smaller distance -> earlier in order.
    assert [r.name for r in ranked] == ["Lion", "Horse", "Eagle"]

    # All three share the same pre-discount base, so the ratios of the final
    # combined distances must equal the ratios of the graded group factors
    # (Cat vs candidates: Lion overlap 2/3, Horse overlap 1/3, Eagle overlap 0/3).
    bonus = DEFAULT_WEIGHTS.group_bonus
    base = by_name["Eagle"].combined_distance          # factor 1.0
    assert base > 0
    assert np.isclose(by_name["Lion"].combined_distance, base * (1.0 - bonus * 2.0 / 3.0))
    assert np.isclose(by_name["Horse"].combined_distance, base * (1.0 - bonus * 1.0 / 3.0))

    # same_group means "identical motion descriptor" (full overlap).
    # Lion shares 2/3 tags with Cat (Medium vs Small mismatch) — not full overlap.
    assert by_name["Lion"].same_group is False
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
    # No motion tags for the query -> overlap 0 everywhere -> no discount, so
    # the candidates tie on distance and fall back to name ordering.
    assert all(r.same_group is False for r in ranked)
    assert np.isclose(ranked[0].combined_distance, ranked[1].combined_distance)
