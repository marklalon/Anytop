"""Auxiliary group membership: split safety, leak safety, and the mass budget.

An aux clip is borrowed from another action group. The properties that make
that safe are all about what it must NOT disturb:

* the species split, which is computed from primary membership alone -- one
  extra species would re-deal every later species into a different split and
  silently void the comparison with every earlier run;
* val and test, which stay exactly the group's own clips;
* a species this group holds out for evaluation, which may not walk back into
  train through its aux clips.

Plus the one thing it must do: hold the borrowed clips to a fixed share of the
sampling mass however many of them there are.

See docs/aux_group_and_head_word_augmentation.md §4.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loaders.truebones.data.dataset import (  # noqa: E402
    SUPPORTED_SPLITS,
    aux_action_groups_of,
    filter_motion_names_by_action_group,
    filter_motion_names_by_aux_action_group,
    load_aux_motion_names_for_train,
    load_motion_names_for_split_with_action_group,
)
from data_loaders.truebones.truebones_utils.motion_labels import (  # noqa: E402
    AUX_ACTION_GROUPS_KEY,
    aux_key_present_in,
    load_action_labels,
)
from data_loaders.truebones.truebones_utils.param_utils import (  # noqa: E402
    ACTION_LABELS_FILE,
)


# --------------------------------------------------------------------------
# A synthetic corpus
# --------------------------------------------------------------------------
# NOTE on the shipped split: DEFAULT_SPLIT_RATIOS is {train 1.0, val 0, test 0},
# so from 4 species up every species lands in train and val/test are EMPTY.
# Only 2-3 species hold anything out. The leak test below therefore uses 3
# species on purpose -- it is the only size at which a held-out species exists
# to leak, and the guard it covers is what keeps the feature correct if those
# ratios are ever changed back.
SPECIES = [f"Sp{index:02d}" for index in range(12)]
HOLDOUT_SPECIES = [f"Sp{index:02d}" for index in range(3)]


def _write_corpus(tmp_path: Path, aux_for=None, with_aux_key=True,
                  species=None) -> tuple[Path, Path, dict]:
    """One clip per species per group, plus the sidecar rows and motion files."""
    dataset_dir = tmp_path / "ds"
    motion_dir = dataset_dir / "motions"
    motion_dir.mkdir(parents=True)

    rows = []
    metadata = {}
    for species in (species or SPECIES):
        for group in ("locomotion", "stationary", "transition"):
            clip = f"{species}_{group}"
            (motion_dir / f"{clip}.npy").write_bytes(b"")
            row = {
                "clip": clip,
                "action_group": group,
                "action_label": {"locomotion": "walk, forward",
                                 "stationary": "idle",
                                 "transition": "die"}[group],
            }
            if with_aux_key:
                row[AUX_ACTION_GROUPS_KEY] = list(
                    (aux_for or (lambda c, g: ()))(clip, group)
                )
            row["is_loop"] = False
            rows.append(row)
            metadata[f"{clip}.npy"] = {
                "object_type": species,
                "action_group": group,
                "action_label": row["action_label"],
                "is_loop": False,
                "translation_root_index": 0,
            }
            if with_aux_key:
                metadata[f"{clip}.npy"][AUX_ACTION_GROUPS_KEY] = tuple(
                    row[AUX_ACTION_GROUPS_KEY]
                )
    (dataset_dir / ACTION_LABELS_FILE).write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return dataset_dir, motion_dir, metadata


def _splits(dataset_dir, motion_dir, metadata, group):
    """Resolve every split, skipping the ones this corpus size leaves empty.

    An empty split is a hard error in the loader (training on nothing is never
    what the caller meant), so a test that just wants to see the manifests has
    to step around it.
    """
    resolved = {}
    for split in SUPPORTED_SPLITS:
        try:
            resolved[split] = load_motion_names_for_split_with_action_group(
                split, str(dataset_dir), str(motion_dir), group, metadata
            )
        except RuntimeError:
            resolved[split] = set()
    return resolved


# --------------------------------------------------------------------------
# The split is computed from primary membership alone
# --------------------------------------------------------------------------
def test_manifests_are_byte_identical_with_and_without_aux(tmp_path):
    """The invariant the whole design rests on.

    Aux clips are added after the split, so turning them on cannot change which
    species land in val/test -- and therefore cannot make an earlier run
    incomparable.
    """
    plain_dir, plain_motions, plain_meta = _write_corpus(tmp_path / "plain", aux_for=None)
    _splits(plain_dir, plain_motions, plain_meta, "transition")
    plain_manifests = {
        split: (plain_dir / f"{split}.txt").read_bytes()
        for split in SUPPORTED_SPLITS
        if (plain_dir / f"{split}.txt").exists()
    }

    # Every locomotion and stationary clip now also trains transition.
    aux_dir, aux_motions, aux_meta = _write_corpus(
        tmp_path / "aux",
        aux_for=lambda clip, group: () if group == "transition" else ("transition",),
    )
    _splits(aux_dir, aux_motions, aux_meta, "transition")
    aux_manifests = {
        split: (aux_dir / f"{split}.txt").read_bytes()
        for split in SUPPORTED_SPLITS
        if (aux_dir / f"{split}.txt").exists()
    }

    assert plain_manifests.keys() == aux_manifests.keys()
    for split in plain_manifests:
        assert plain_manifests[split] == aux_manifests[split], split


def test_primary_split_is_unchanged_by_aux(tmp_path):
    plain_dir, plain_motions, plain_meta = _write_corpus(tmp_path / "plain", aux_for=None)
    aux_dir, aux_motions, aux_meta = _write_corpus(
        tmp_path / "aux",
        aux_for=lambda clip, group: () if group == "transition" else ("transition",),
    )
    plain = _splits(plain_dir, plain_motions, plain_meta, "transition")
    borrowed = _splits(aux_dir, aux_motions, aux_meta, "transition")
    assert plain == borrowed


# --------------------------------------------------------------------------
# Aux clips reach train only, and never a held-out species
# --------------------------------------------------------------------------
def test_aux_clips_are_train_only(tmp_path):
    dataset_dir, motion_dir, metadata = _write_corpus(
        tmp_path,
        aux_for=lambda clip, group: () if group == "transition" else ("transition",),
    )
    for split in ("val", "test"):
        assert load_aux_motion_names_for_train(
            split, str(motion_dir), "transition", metadata
        ) == set()
    assert load_aux_motion_names_for_train(
        "train", str(motion_dir), "transition", metadata
    )


def test_a_held_out_species_cannot_return_through_its_aux_clips(tmp_path):
    dataset_dir, motion_dir, metadata = _write_corpus(
        tmp_path,
        aux_for=lambda clip, group: () if group == "transition" else ("transition",),
        species=HOLDOUT_SPECIES,
    )
    splits = _splits(dataset_dir, motion_dir, metadata, "transition")
    held_out = {
        str(metadata[name]["object_type"])
        for split in ("val", "test")
        for name in splits[split]
    }
    assert held_out, "the fixture must hold some species out for this test to mean anything"

    aux_train = load_aux_motion_names_for_train(
        "train", str(motion_dir), "transition", metadata
    )
    leaked = {
        name for name in aux_train if str(metadata[name]["object_type"]) in held_out
    }
    assert leaked == set()


def test_a_species_absent_from_the_group_is_not_treated_as_held_out(tmp_path):
    """The case the whole feature exists for.

    A species with no clip in this group has no split verdict here, so it is
    not held out and its borrowed clips must be kept -- that is exactly the
    species whose prior the aux clips are meant to fix.
    """
    dataset_dir = tmp_path / "ds"
    motion_dir = dataset_dir / "motions"
    motion_dir.mkdir(parents=True)
    rows, metadata = [], {}

    def add(clip, species, group, aux=()):
        (motion_dir / f"{clip}.npy").write_bytes(b"")
        rows.append({
            "clip": clip, "action_group": group, "action_label": "idle",
            AUX_ACTION_GROUPS_KEY: list(aux), "is_loop": False,
        })
        metadata[f"{clip}.npy"] = {
            "object_type": species, "action_group": group, "action_label": "idle",
            AUX_ACTION_GROUPS_KEY: tuple(aux), "is_loop": False,
            "translation_root_index": 0,
        }

    for species in SPECIES:
        add(f"{species}_transition", species, "transition")
    # Loner has nothing in transition at all, only a borrowed stationary clip.
    add("Loner_stationary", "Loner", "stationary", aux=("transition",))
    (dataset_dir / ACTION_LABELS_FILE).write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    aux_train = load_aux_motion_names_for_train(
        "train", str(motion_dir), "transition", metadata
    )
    assert "Loner_stationary.npy" in aux_train


def test_primary_and_aux_pools_are_disjoint(tmp_path):
    dataset_dir, motion_dir, metadata = _write_corpus(
        tmp_path,
        aux_for=lambda clip, group: () if group == "transition" else ("transition",),
    )
    names = {f"{clip}" for clip in metadata}
    primary = filter_motion_names_by_action_group(names, "transition", metadata)
    aux = filter_motion_names_by_aux_action_group(names, "transition", metadata)
    assert primary & aux == set()


# --------------------------------------------------------------------------
# The two "empty"s (§4.2)
# --------------------------------------------------------------------------
def test_aux_key_presence_distinguishes_a_stale_sidecar_from_an_empty_pool(tmp_path):
    migrated_dir, _, migrated_meta = _write_corpus(tmp_path / "migrated", aux_for=None)
    stale_dir, _, stale_meta = _write_corpus(tmp_path / "stale", with_aux_key=False)
    # Migrated but matching nothing: the key is there, so the sidecar is current.
    assert aux_key_present_in(migrated_meta) is True
    # Never migrated: no row has the key at all.
    assert aux_key_present_in(stale_meta) is False


def test_a_row_may_not_list_its_own_group(tmp_path):
    dataset_dir = tmp_path / "ds"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / ACTION_LABELS_FILE).write_text(
        json.dumps({
            "clip": "Sp_x", "action_group": "transition", "action_label": "die",
            AUX_ACTION_GROUPS_KEY: ["transition"], "is_loop": False,
        }) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        load_action_labels(dataset_dir)


def test_aux_groups_round_trip_through_the_sidecar(tmp_path):
    dataset_dir = tmp_path / "ds"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / ACTION_LABELS_FILE).write_text(
        json.dumps({
            "clip": "Sp_x", "action_group": "stationary",
            "action_label": "attack, jump, charge",
            AUX_ACTION_GROUPS_KEY: ["transition"], "is_loop": False,
        }) + "\n",
        encoding="utf-8",
    )
    labels = load_action_labels(dataset_dir)
    assert labels["Sp_x"][AUX_ACTION_GROUPS_KEY] == ("transition",)
    assert aux_action_groups_of(labels["Sp_x"]) == ("transition",)


# --------------------------------------------------------------------------
# The mass budget
# --------------------------------------------------------------------------
class _FakeMotionDataset:
    """The three attributes TruebonesSampler reads, and nothing else."""

    def __init__(self, object_types, aux_flags, aux_group_mass, balanced=False):
        self.name_list = [f"clip{index}" for index in range(len(object_types))]
        self.data_dict = {
            name: {"object_type": object_type}
            for name, object_type in zip(self.name_list, object_types)
        }
        self.cond_dict = {object_type: {} for object_type in dict.fromkeys(object_types)}
        self.aux_mask = np.asarray(aux_flags, dtype=bool)
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


def _weights(object_types, aux_flags, aux_group_mass, balanced=False):
    from data_loaders.truebones.data.dataset import TruebonesSampler

    dataset = _FakeMotionDataset(object_types, aux_flags, aux_group_mass, balanced)
    sampler = TruebonesSampler(_FakeDataSource(dataset))
    return np.asarray(sampler.weights, dtype=np.float64), dataset.aux_mask


def test_aux_pool_gets_exactly_its_budget_however_many_clips_it_holds():
    # 2 own clips, 20 borrowed ones: the budget, not the count, decides.
    object_types = ["A"] * 2 + ["B"] * 20
    aux_flags = [False] * 2 + [True] * 20
    weights, aux_mask = _weights(object_types, aux_flags, 0.25)
    assert weights[~aux_mask].sum() == pytest.approx(0.75)
    assert weights[aux_mask].sum() == pytest.approx(0.25)


def test_mass_budget_is_independent_of_aux_pool_size():
    small, small_mask = _weights(["A", "A", "B"], [False, False, True], 0.25)
    large, large_mask = _weights(["A", "A"] + ["B"] * 50,
                                 [False, False] + [True] * 50, 0.25)
    assert small[~small_mask].sum() == pytest.approx(large[~large_mask].sum())
    assert small[small_mask].sum() == pytest.approx(large[large_mask].sum())


def test_zero_budget_leaves_the_own_pool_with_everything():
    weights, aux_mask = _weights(["A", "A", "B"], [False, False, True], 0.0)
    assert weights[~aux_mask].sum() == pytest.approx(1.0)
    assert weights[aux_mask].sum() == pytest.approx(0.0)


def test_unbalanced_runs_stay_uniform_per_clip_within_a_pool():
    """Turning the weighted sampler on for the aux budget must not start
    balancing species behind --balanced's back."""
    weights, aux_mask = _weights(["A", "A", "A", "B"], [False] * 4, 0.0, balanced=False)
    assert np.allclose(weights, weights[0])
