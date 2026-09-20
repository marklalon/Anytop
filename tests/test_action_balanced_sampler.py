"""``--balanced``: sampling mass per action head word.

The corpus is lopsided on the action axis in a way it is not on any other:
891 ``attack`` clips and 732 ``idle`` ones against 5 ``stop`` and 1 ``sheathe``
(outputs/_action_firstword_counts.txt). Drawing clips uniformly hands that ratio
to the model, so the head word it was asked for stops deciding much. These tests
pin the properties of the fix:

* mass goes to GROUPS by sqrt of their clip count, not to clips;
* the group is the label's FIRST head word, multi-head labels included;
* species are deliberately NOT balanced (AnyTop's original sampler balanced
  them and nothing else -- that mode is gone);
* the aux-pool budget and the unbalanced (uniform) path are untouched.
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
    UNLABELED_ACTION_GROUP,
    TruebonesSampler,
    clip_action_head_word,
)


class _FakeMotionDataset:
    """The attributes TruebonesSampler reads, and nothing else."""

    def __init__(
        self,
        labels,
        object_types=None,
        balanced=True,
        aux_flags=None,
        aux_group_mass=0.0,
    ):
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
        self.aux_mask = None if aux_flags is None else np.asarray(aux_flags, dtype=bool)
        self.aux_group_mass = aux_group_mass
        self.balanced = balanced
        self.pointer = 0

    def __len__(self):
        return len(self.name_list)


class _FakeDataSource:
    def __init__(self, motion_dataset):
        self.motion_dataset = motion_dataset

    def __len__(self):
        return len(self.motion_dataset)


def _weights(labels, **kwargs):
    dataset = _FakeMotionDataset(labels, **kwargs)
    sampler = TruebonesSampler(_FakeDataSource(dataset))
    return np.asarray(sampler.weights, dtype=np.float64)


def _mass(weights, indices):
    return float(np.asarray(weights)[list(indices)].sum())


# --------------------------------------------------------------------------
# The group is the first head word
# --------------------------------------------------------------------------
def test_head_word_of_a_label_is_its_first_head_word():
    assert clip_action_head_word({"action_label": "attack, jump, charge"}) == "attack"
    assert clip_action_head_word({"action_label": "land, jump"}) == "land"
    assert clip_action_head_word({"action_label": "walk, forward, slow"}) == "walk"


def test_head_word_of_an_unlabelled_clip_is_the_unlabelled_group():
    assert clip_action_head_word({}) == UNLABELED_ACTION_GROUP
    assert clip_action_head_word({"action_label": ""}) == UNLABELED_ACTION_GROUP
    assert clip_action_head_word(None) == UNLABELED_ACTION_GROUP


def test_an_unparsable_label_names_the_clip_it_came_from():
    with pytest.raises(RuntimeError, match="clip7"):
        clip_action_head_word({"action_label": "attack, notaword"}, "clip7")


def test_a_multi_head_label_is_counted_under_its_first_head_word():
    # "attack, jump" belongs with the attacks, not with the jumps: three clips
    # in the attack group, one in the jump group.
    labels = ["attack", "attack, jump", "attack, spin", "jump"]
    weights = _weights(labels)
    assert weights[0] == pytest.approx(weights[1]) == pytest.approx(weights[2])
    assert _mass(weights, [0, 1, 2]) == pytest.approx(np.sqrt(3) / (np.sqrt(3) + 1))
    assert _mass(weights, [3]) == pytest.approx(1 / (np.sqrt(3) + 1))


# --------------------------------------------------------------------------
# The mass rule
# --------------------------------------------------------------------------
def test_group_mass_is_sqrt_of_clip_count():
    labels = ["attack"] * 9 + ["stop"]
    weights = _weights(labels)
    assert _mass(weights, range(9)) == pytest.approx(0.75)  # sqrt(9) / (sqrt(9) + 1)
    assert _mass(weights, [9]) == pytest.approx(0.25)
    assert weights.sum() == pytest.approx(1.0)


def test_a_rare_actions_clip_outweighs_a_common_ones_clip():
    """The per-clip ratio is sqrt(n_common / n_rare), not 1 and not n."""
    labels = ["attack"] * 100 + ["sheathe"]
    weights = _weights(labels)
    assert weights[-1] / weights[0] == pytest.approx(10.0)


def test_mass_inside_a_group_is_split_evenly_over_its_clips():
    # Uneven species inside one head word: --balanced deliberately does NOT
    # re-balance them, so all four walk clips weigh the same.
    labels = ["walk"] * 4 + ["run"]
    weights = _weights(labels, object_types=["A", "A", "A", "B", "A"])
    assert np.allclose(weights[:4], weights[0])


def test_weights_stay_a_distribution_over_many_groups():
    labels = ["attack"] * 891 + ["idle"] * 732 + ["stop"] * 5 + ["sheathe"]
    weights = _weights(labels)
    assert weights.sum() == pytest.approx(1.0)
    assert (weights > 0).all()
    # The whole point, measured against what uniform per-clip draws would give:
    # attack gives share back and the single sheathe clip rises off the floor.
    uniform = 1.0 / len(labels)
    attack_mass = _mass(weights, range(891))
    sheathe_mass = _mass(weights, [len(labels) - 1])
    assert attack_mass < 891 * uniform
    assert sheathe_mass > 20 * uniform


# --------------------------------------------------------------------------
# What must not change
# --------------------------------------------------------------------------
def test_species_are_not_balanced():
    """The species axis is left alone: 9 clips of one species against 1 of
    another, all under the same head word, stay equally likely."""
    weights = _weights(["attack"] * 10, object_types=["A"] * 9 + ["B"])
    assert np.allclose(weights, weights[0])


def test_unbalanced_runs_stay_uniform_per_clip():
    weights = _weights(
        ["attack"] * 3 + ["stop"],
        balanced=False,
        aux_flags=[False] * 4,
        aux_group_mass=0.0,
    )
    assert np.allclose(weights, weights[0])


def test_unbalanced_runs_never_read_the_label():
    """An unbalanced run must not start failing on an unparsable label."""
    weights = _weights(["attack", "notaword"], balanced=False)
    assert weights.sum() == pytest.approx(1.0)


def test_unlabelled_clips_form_their_own_group():
    labels = ["attack"] * 4 + ["", ""]
    weights = _weights(labels)
    assert _mass(weights, [4, 5]) == pytest.approx(np.sqrt(2) / (2 + np.sqrt(2)))
    assert weights[4] == pytest.approx(weights[5])


def test_aux_budget_holds_under_action_balancing():
    labels = ["attack"] * 2 + ["stop"] * 20
    aux_flags = [False] * 2 + [True] * 20
    weights = _weights(labels, aux_flags=aux_flags, aux_group_mass=0.25)
    aux_mask = np.asarray(aux_flags, dtype=bool)
    assert _mass(weights, np.flatnonzero(~aux_mask)) == pytest.approx(0.75)
    assert _mass(weights, np.flatnonzero(aux_mask)) == pytest.approx(0.25)
