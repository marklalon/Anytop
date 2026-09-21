"""Head-word sampling: ``--rare_head_word_floor`` and ``--head_word_weights``.

Draws are uniform over clips; the two multipliers are the only deviations and
both are meant for the corpus tail (stop, sheathe, crawl: 1 clip each). These
tests pin:

* the floor: a word with fewer clips than the floor is weighted as if it had
  the floor's count (per clip floor/count), capped at the max boost; words at
  or above the floor and the unlabelled key are untouched;
* explicit weights: weight w = drawn w times as often as an unlisted clip,
  nothing else moves; they stack multiplicatively on the floor;
* the key is the label's FIRST head word, multi-head labels included;
* species are not balanced;
* the CLI spellings are validated (head words only, > 0, no repeats; floor
  >= 0, max boost >= 1);
* a listed word with no clip in the subset is an error, not a no-op;
* with neither the loader stays on the plain (uniform) sampler.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loaders.truebones.data.dataset import (  # noqa: E402
    RARE_HEAD_WORD_FLOOR_OFF,
    UNLABELED_ACTION_GROUP,
    TruebonesSampler,
    clip_action_head_word,
    parse_head_word_weights,
    parse_rare_head_word_floor,
    rare_head_word_boost,
)


class _FakeMotionDataset:
    """The attributes TruebonesSampler reads, and nothing else."""

    def __init__(self, labels, object_types=None, head_word_weights=None, pointer=0,
                 rare_head_word_floor=0, rare_head_word_max_boost=4.0):
        object_types = list(object_types or ["A"] * len(labels))
        self.name_list = [f"clip{index}" for index in range(len(labels))]
        self.data_dict = {
            name: {
                "object_type": object_type,
                "motion_name": name,
                "motion_metadata": {"action_label": label},
            }
            for name, label, object_type in zip(self.name_list, labels, object_types)
        }
        self.cond_dict = {object_type: {} for object_type in dict.fromkeys(object_types)}
        self.head_word_weights = parse_head_word_weights(head_word_weights)
        self.rare_head_word_floor, self.rare_head_word_max_boost = parse_rare_head_word_floor(
            rare_head_word_floor, rare_head_word_max_boost
        )
        self.pointer = pointer

    def __len__(self):
        return len(self.name_list) - self.pointer


class _FakeDataSource:
    def __init__(self, motion_dataset):
        self.motion_dataset = motion_dataset

    def __len__(self):
        return len(self.motion_dataset)


def _weights(labels, **kwargs):
    dataset = _FakeMotionDataset(labels, **kwargs)
    sampler = TruebonesSampler(_FakeDataSource(dataset))
    return np.asarray(sampler.weights, dtype=np.float64)


# --------------------------------------------------------------------------
# The key is the first head word
# --------------------------------------------------------------------------
def test_head_word_of_a_label_is_its_first_head_word():
    assert clip_action_head_word({"action_label": "attack, jump, charge"}) == "attack"
    assert clip_action_head_word({"action_label": "land, jump"}) == "land"
    assert clip_action_head_word({"action_label": "walk, forward, slow"}) == "walk"


def test_head_word_of_an_unlabelled_clip_is_the_unlabelled_key():
    assert clip_action_head_word({}) == UNLABELED_ACTION_GROUP
    assert clip_action_head_word({"action_label": ""}) == UNLABELED_ACTION_GROUP
    assert clip_action_head_word(None) == UNLABELED_ACTION_GROUP


def test_an_unparsable_label_names_the_clip_it_came_from():
    with pytest.raises(RuntimeError, match="clip7"):
        clip_action_head_word({"action_label": "attack, notaword"}, "clip7")


def test_a_multi_head_label_is_keyed_under_its_first_head_word():
    # "attack, jump" is an attack: the jump multiplier does not reach it.
    labels = ["attack", "attack, jump", "jump"]
    weights = _weights(labels, head_word_weights="jump=3")
    assert weights[0] == pytest.approx(weights[1])
    assert weights[2] / weights[0] == pytest.approx(3.0)


# --------------------------------------------------------------------------
# The multiplier rule
# --------------------------------------------------------------------------
def test_a_weighted_clip_is_drawn_weight_times_as_often():
    labels = ["attack"] * 100 + ["stop"] * 5 + ["sheathe"]
    weights = _weights(labels, head_word_weights="stop=3,sheathe=4")
    assert weights.sum() == pytest.approx(1.0)
    assert np.allclose(weights[:100], weights[0])
    assert weights[100] / weights[0] == pytest.approx(3.0)
    assert weights[105] / weights[0] == pytest.approx(4.0)
    # Closed form: total mass 100 + 15 + 4 = 119.
    assert weights[0] == pytest.approx(1 / 119)


def test_unlisted_words_keep_the_corpus_proportions_between_them():
    labels = ["attack"] * 891 + ["idle"] * 732 + ["stop"] * 5
    weights = _weights(labels, head_word_weights="stop=3")
    attack = weights[:891].sum()
    idle = weights[891:891 + 732].sum()
    assert attack / idle == pytest.approx(891 / 732)
    # The tail took its extra mass from everyone alike: 10 clip-equivalents
    # on a pool of 1628, so attack's share falls by that factor and no more.
    assert attack == pytest.approx(891 / (891 + 732 + 15))


def test_weight_one_is_inert():
    labels = ["attack"] * 3 + ["stop"]
    assert np.allclose(_weights(labels, head_word_weights="stop=1"), 1 / 4)


def test_species_are_not_balanced():
    """9 clips of one species against 1 of another under the same head word
    stay equally likely, weighted or not."""
    weights = _weights(["attack"] * 10, object_types=["A"] * 9 + ["B"], head_word_weights="attack=2")
    assert np.allclose(weights, weights[0])


def test_entries_below_the_pointer_carry_no_weight():
    labels = ["attack"] * 4 + ["stop"] * 2
    weights = _weights(labels, head_word_weights="stop=2", pointer=2)
    assert np.allclose(weights[:2], 0.0)
    assert weights[2:].sum() == pytest.approx(1.0)
    assert weights[4] / weights[2] == pytest.approx(2.0)


def test_a_weighted_word_absent_from_the_subset_is_an_error():
    with pytest.raises(RuntimeError, match="sheathe"):
        _weights(["attack"] * 3, head_word_weights="sheathe=4")


def test_unlabelled_clips_cannot_be_weighted_but_still_count_as_one():
    labels = ["attack"] * 4 + ["", ""]
    weights = _weights(labels, head_word_weights="attack=1")
    assert np.allclose(weights, 1 / 6)


# --------------------------------------------------------------------------
# The CLI spelling
# --------------------------------------------------------------------------
def test_parse_accepts_the_cli_string_a_mapping_and_nothing():
    assert parse_head_word_weights("stop=3, sheathe=4.5") == {"stop": 3.0, "sheathe": 4.5}
    assert parse_head_word_weights({"stop": 3}) == {"stop": 3.0}
    assert parse_head_word_weights("") == {}
    assert parse_head_word_weights(None) == {}
    assert parse_head_word_weights(" , ") == {}


@pytest.mark.parametrize("spec, message", [
    ("attack=2,notaword=3", "not a head word"),
    ("swat=2", "not a head word"),          # a modifier, not a head
    ("stop=0", "> 0"),
    ("stop=-1", "> 0"),
    ("stop=nan", "> 0"),
    ("stop=three", "not a number"),
    ("stop", "word=weight"),
    ("stop=2,stop=3", "twice"),
])
def test_parse_rejects_bad_spellings(spec, message):
    with pytest.raises(ValueError, match=message):
        parse_head_word_weights(spec)


# --------------------------------------------------------------------------
# The rare-word floor
# --------------------------------------------------------------------------
def test_floor_lifts_every_word_below_it_to_the_floor_count():
    """floor 20, cap 4: 1 clip x4 (capped), 5 clips x4 (=20/5), 10 clips x2,
    16 clips x1.25, 20 and above x1."""
    assert rare_head_word_boost(1, 20, 4.0) == pytest.approx(4.0)
    assert rare_head_word_boost(5, 20, 4.0) == pytest.approx(4.0)
    assert rare_head_word_boost(6, 20, 4.0) == pytest.approx(20 / 6)
    assert rare_head_word_boost(10, 20, 4.0) == pytest.approx(2.0)
    assert rare_head_word_boost(16, 20, 4.0) == pytest.approx(1.25)
    assert rare_head_word_boost(20, 20, 4.0) == pytest.approx(1.0)
    assert rare_head_word_boost(891, 20, 4.0) == pytest.approx(1.0)


def test_floor_gives_a_rare_word_the_mass_of_a_floor_sized_word():
    labels = ["attack"] * 100 + ["land"] * 10 + ["putdown"] * 5
    weights = _weights(labels, rare_head_word_floor=10, rare_head_word_max_boost=10.0)
    assert weights.sum() == pytest.approx(1.0)
    assert np.allclose(weights[:100], weights[0])
    # land is at the floor: untouched. putdown (5) is lifted to 10 clips'
    # worth: x2 per clip, so its total mass equals land's.
    assert weights[100] == pytest.approx(weights[0])
    assert weights[110] / weights[0] == pytest.approx(2.0)
    assert weights[110:].sum() == pytest.approx(weights[100:110].sum())
    # Closed form: total mass 100 + 10 + 10 = 120.
    assert weights[0] == pytest.approx(1 / 120)


def test_floor_boost_is_capped():
    labels = ["attack"] * 50 + ["sheathe"]
    weights = _weights(labels, rare_head_word_floor=20, rare_head_word_max_boost=4.0)
    assert weights[50] / weights[0] == pytest.approx(4.0)
    assert weights[0] == pytest.approx(1 / 54)


def test_floor_uses_the_cli_max_boost_default_when_cap_is_omitted():
    labels = ["attack"] * 50 + ["sheathe"]
    weights = _weights(labels, rare_head_word_floor=20)
    assert weights[50] / weights[0] == pytest.approx(4.0)


def test_floor_does_not_touch_the_unlabelled_key():
    labels = ["attack"] * 50 + ["", ""]
    weights = _weights(labels, rare_head_word_floor=20, rare_head_word_max_boost=10.0)
    assert np.allclose(weights, 1 / 52)


def test_floor_and_explicit_weights_stack_multiplicatively():
    labels = ["attack"] * 50 + ["stop"] * 2 + ["land"] * 10
    weights = _weights(
        labels, rare_head_word_floor=10, rare_head_word_max_boost=4.0, head_word_weights="stop=2,land=3"
    )
    # stop: floor 10/2 = 5 capped to 4, times explicit 2 = 8.
    assert weights[50] / weights[0] == pytest.approx(8.0)
    # land: at the floor (x1), times explicit 3.
    assert weights[52] / weights[0] == pytest.approx(3.0)


def test_floor_of_zero_or_one_is_off():
    labels = ["attack"] * 3 + ["stop"]
    assert np.allclose(_weights(labels, rare_head_word_floor=0, rare_head_word_max_boost=4.0), 1 / 4)
    assert np.allclose(_weights(labels, rare_head_word_floor=1, rare_head_word_max_boost=4.0), 1 / 4)
    assert RARE_HEAD_WORD_FLOOR_OFF == 1


def test_floor_alone_turns_on_the_weighted_sampler_flag():
    """MotionDataset decides the sampler from the floor as well as the list;
    mirror its rule here so a change to it is caught."""
    floor, _ = parse_rare_head_word_floor(20, 4.0)
    assert floor > RARE_HEAD_WORD_FLOOR_OFF
    floor, _ = parse_rare_head_word_floor(None, None)
    assert floor <= RARE_HEAD_WORD_FLOOR_OFF


@pytest.mark.parametrize("floor, max_boost, message", [
    (-1, 4.0, ">= 0"),
    (20, 0.5, ">= 1"),
    (20, 0.0, ">= 1"),
    (20, float("nan"), ">= 1"),
    (20, float("inf"), ">= 1"),
])
def test_floor_rejects_bad_values(floor, max_boost, message):
    with pytest.raises(ValueError, match=message):
        parse_rare_head_word_floor(floor, max_boost)
