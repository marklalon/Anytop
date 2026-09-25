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

from utils.skeleton_similarity import (  # noqa: E402
    assign_softmax_weights,
    part_token_set,
    rank_species,
    species_tags_of,
    tag_distance,
)


def _cond(parents, joints_names, tags=None, slim=None) -> dict:
    cond = {
        "parents": np.asarray(parents, dtype=np.int32),
        "joints_names": list(joints_names),
    }
    if tags is not None:
        cond["species_tags"] = tuple(tags)
    if slim is not None:
        cond["joints_names_embs_meta"] = {"embedding_texts": list(slim)}
    return cond


# ── motion tags ──────────────────────────────────────────────────────────────
def test_species_tags_read_the_baked_pair_only() -> None:
    baked = _cond([-1], ["Root"], tags=("Winged", "Flapping"))
    assert species_tags_of(baked, "anything") == ("winged", "flapping")
    with pytest.raises(ValueError, match="regenerate"):
        species_tags_of(_cond([-1], ["Root"]), "cat")
    with pytest.raises(ValueError, match="regenerate"):
        species_tags_of(_cond([-1], ["Root"], tags=("Winged", "Heavy", "Flapping")), "x")


def test_tag_distance_weights_slots() -> None:
    horse = ("quadruped", "galloping")
    assert tag_distance(horse, horse) == pytest.approx(0.0)
    # Same body plan, different gait.
    assert tag_distance(horse, ("quadruped", "trotting")) == pytest.approx(0.4)
    # Different body plan is the larger single penalty.
    assert tag_distance(horse, ("winged", "galloping")) == pytest.approx(0.6)


# ── body parts ───────────────────────────────────────────────────────────────
def test_part_tokens_drop_qualifiers_and_prefer_slim_texts() -> None:
    cond = _cond(
        [-1, 0, 1, 1],
        ["Hips", "Tail 01", "Right Front Upper Leg", "Left Back Upper Leg"],
        slim=["Hips", "Tail", "Right Front Thigh", "Left Back Thigh"],
    )
    assert part_token_set(cond, "x") == {"hips", "tail", "thigh"}
    # Raw fallback strips segment numbers, sides and helper suffixes.
    raw = _cond([-1, 0, 1, 1], ["Hips", "Tail 01", "Right Thigh", "Left Toe 0 Nub"])
    assert part_token_set(raw, "x") == {"hips", "tail", "thigh", "toe"}


# ── ranking ──────────────────────────────────────────────────────────────────
def test_rank_species_puts_the_same_mover_first_regardless_of_joint_names() -> None:
    # A quadruped galloper named one way; candidates: the same mover named the
    # other way, a quadruped with the same names but a different gait, and a
    # bird with similar joint count.
    query = _cond(
        [-1, 0, 1, 1, 0],
        ["Hips", "Spine", "Right Front Upper Leg", "Left Front Upper Leg", "Tail 01"],
        tags=("Quadruped", "Galloping"),
        slim=["Hips", "Spine", "Right Front Thigh", "Left Front Thigh", "Tail"],
    )
    candidates = {
        "Deer": _cond(
            [-1, 0, 1, 1, 0],
            ["Pelvis", "Spine1", "RightThigh", "LeftThigh", "Tail1"],
            tags=("Quadruped", "Galloping"),
            slim=["Pelvis", "Spine", "Right Thigh", "Left Thigh", "Tail"],
        ),
        "Bear": _cond(
            [-1, 0, 1, 1, 0],
            ["Hips", "Spine", "Right Front Upper Leg", "Left Front Upper Leg", "Tail 01"],
            tags=("Quadruped", "Lumbering"),
            slim=["Hips", "Spine", "Right Front Thigh", "Left Front Thigh", "Tail"],
        ),
        "Crow": _cond(
            [-1, 0, 1, 1, 0],
            ["Hips", "Spine", "Right Wing", "Left Wing", "Tail 01"],
            tags=("Winged", "Flapping"),
            slim=["Hips", "Spine", "Right Wing", "Left Wing", "Tail"],
        ),
    }
    ranked = rank_species(query, candidates, query_hint="Horse", top_k=None)
    # Deer: tags match, 3/5 parts (0.3 * 0.4 = 0.12); Bear: parts match, gait differs (0.5 * 0.4 = 0.20).
    assert [r.name for r in ranked] == ["Deer", "Bear", "Crow"]
    by_name = {r.name: r for r in ranked}
    # Bear shares every body part; Deer shares the tags. Both beat the bird by a margin.
    assert by_name["Deer"].same_tags is True and by_name["Deer"].tag_distance == pytest.approx(0.0)
    assert by_name["Bear"].jaccard == pytest.approx(1.0)
    assert by_name["Crow"].combined_distance > 2 * max(by_name["Deer"].combined_distance, by_name["Bear"].combined_distance)
    assert sum(r.weight for r in ranked) == pytest.approx(1.0)
    assert by_name["Crow"].weight < min(by_name["Deer"].weight, by_name["Bear"].weight)


def test_rank_species_top_k_and_softmax_over_the_selection() -> None:
    query = _cond([-1, 0], ["Hips", "Tail"], tags=("Serpentine", "Slithering"))
    candidates = {
        "Snake": _cond([-1, 0], ["Hips", "Tail"], tags=("Serpentine", "Slithering")),
        "Worm": _cond([-1, 0], ["Hips", "Tail"], tags=("Serpentine", "Undulating")),
        "Eel": _cond([-1, 0], ["Hips", "Tail"], tags=("Aquatic", "Swimming")),
    }
    ranked = rank_species(query, candidates, query_hint="Snake", top_k=2)
    assert [r.name for r in ranked] == ["Snake", "Worm"]
    assert sum(r.weight for r in ranked) == pytest.approx(1.0)
    # Widening the selection re-weights over the new set.
    widened = rank_species(query, candidates, query_hint="Snake", top_k=None)
    assign_softmax_weights(widened)
    assert [r.name for r in widened] == ["Snake", "Worm", "Eel"]
    assert sum(r.weight for r in widened) == pytest.approx(1.0)
    assert widened[0].weight > widened[1].weight > widened[2].weight


def test_rank_species_fills_the_profile_memo() -> None:
    query = _cond([-1, 0], ["Hips", "Tail"], tags=("Serpentine", "Slithering"))
    candidates = {"Snake": _cond([-1, 0], ["Hips", "Tail"], tags=("Serpentine", "Slithering"))}
    profiles: dict = {}
    rank_species(query, candidates, query_hint="Snake", profiles=profiles)
    assert set(profiles) == {"Snake"}
    assert profiles["Snake"].tags == ("serpentine", "slithering")


