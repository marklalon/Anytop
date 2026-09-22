"""Species names that contain underscores must never be truncated.

The UnityBundles dataset names every clip ``<Pack>_<Species>_<Action>_<id>``
(``FEP_MagmaDemon_Attack01_1.npy``), so any "everything before the first
underscore" shortcut resolves a whole asset pack to a single pseudo-species.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ANYTOP_ROOT = Path(__file__).resolve().parents[1]
if str(ANYTOP_ROOT) not in sys.path:
    sys.path.insert(0, str(ANYTOP_ROOT))

import data_loaders.truebones.data.dataset as dataset_module  # noqa: E402
from data_loaders.truebones.data.dataset import (  # noqa: E402
    load_motion_names_for_split_with_action_group,
    resolve_motion_object_type,
)
from preprocess_and_validate import _species_of_motion_name  # noqa: E402
from utils.misc import infer_object_type_from_filename  # noqa: E402

SPECIES = ("FEP_MagmaDemon", "FEP_IceDemon", "MU06_Death", "MU06_DeathMage")
LOOKUP = {name: f"unitybundles/{name}" for name in SPECIES}


def test_registry_match_keeps_the_full_species_name():
    assert (
        infer_object_type_from_filename("FEP_MagmaDemon_Attack01_1.npy", valid_types=LOOKUP)
        == "unitybundles/FEP_MagmaDemon"
    )
    # A species whose name is *not* a token prefix of a longer one still wins.
    assert (
        infer_object_type_from_filename("MU06_Death_Idle_1.npy", valid_types=LOOKUP)
        == "unitybundles/MU06_Death"
    )
    assert (
        infer_object_type_from_filename("MU06_DeathMage_Idle_1.npy", valid_types=LOOKUP)
        == "unitybundles/MU06_DeathMage"
    )


def test_species_of_motion_name_uses_the_registry():
    assert _species_of_motion_name("FEP_MagmaDemon_Attack01_1.npy", LOOKUP) == "FEP_MagmaDemon"
    # No registry at all (a dataset without cond.npy) is the only blind case.
    assert _species_of_motion_name("FEP_MagmaDemon_Attack01_1.npy", {}) == "FEP"


def test_resolve_motion_object_type_prefers_metadata_and_never_guesses(tmp_path):
    metadata = {"FEP_MagmaDemon_Attack01_1.npy": {"object_type": "FEP_MagmaDemon"}}
    assert (
        resolve_motion_object_type("FEP_MagmaDemon_Attack01_1.npy", str(tmp_path), metadata)
        == "FEP_MagmaDemon"
    )
    assert (
        resolve_motion_object_type("FEP_MagmaDemon_Attack01_1.npy", str(tmp_path), {}, LOOKUP)
        == "FEP_MagmaDemon"
    )
    try:
        resolve_motion_object_type("Unregistered_Walk_1.npy", str(tmp_path), {}, LOOKUP)
    except RuntimeError as exc:
        assert "Unregistered_Walk_1.npy" in str(exc)
    else:
        raise AssertionError("expected a RuntimeError instead of a blind guess")


def _write_species_motions(tmp_path):
    motions_dir = tmp_path / "motions"
    motions_dir.mkdir()
    metadata = {}
    for species in SPECIES:
        for action in ("Idle", "Move"):
            name = f"{species}_{action}_1.npy"
            np.save(motions_dir / name, np.zeros((4, 3, 12), dtype=np.float32))
            metadata[name] = {"object_type": species}
    return motions_dir, metadata


def _clips_per_split(split_paths):
    return {
        split: [line for line in path.read_text(encoding="utf-8").split() if line]
        for split, path in split_paths.items()
    }


def _manifests(tmp_path, motions_dir, metadata, split="val"):
    load_motion_names_for_split_with_action_group(split, str(tmp_path), str(motions_dir), "", metadata)
    return _clips_per_split({s: tmp_path / f"{s}.txt" for s in ("train", "val", "test")})


def test_split_manifests_deal_clips_not_species(tmp_path, monkeypatch):
    # A 50/50 split of 8 clips from 4 species, dealt clip by clip: species end
    # up straddling the splits instead of being held out whole.
    monkeypatch.setattr(
        dataset_module, "DEFAULT_SPLIT_RATIOS", {"train": 0.5, "val": 0.5, "test": 0.0}
    )
    motions_dir, metadata = _write_species_motions(tmp_path)

    clips_per_split = _manifests(tmp_path, motions_dir, metadata)

    assert len(clips_per_split["train"]) + len(clips_per_split["val"]) == 8
    assert clips_per_split["test"] == []
    species_of = lambda name: "_".join(name.split("_")[:2])  # noqa: E731
    straddling = {species_of(n) for n in clips_per_split["train"]} & {species_of(n) for n in clips_per_split["val"]}
    assert straddling, "a per-clip deal must split at least one species across train and val"


def test_default_split_ratios_reserve_a_small_val_split(tmp_path):
    # 0.98 / 0.02 / 0: the hash alone would give 8 clips no val clip most of
    # the time, but a non-zero ratio always keeps at least one; test (ratio 0)
    # gets none.
    motions_dir, metadata = _write_species_motions(tmp_path)

    clips_per_split = _manifests(tmp_path, motions_dir, metadata)

    assert len(clips_per_split["val"]) >= 1
    assert len(clips_per_split["train"]) + len(clips_per_split["val"]) == 8
    assert clips_per_split["test"] == []
    # Deterministic: a second call regenerates identical manifests.
    assert _manifests(tmp_path, motions_dir, metadata) == clips_per_split


def test_val_gate_keeps_rare_label_buckets_whole_in_train():
    assign = dataset_module.assign_clips_to_splits
    names = [f"Species{i % 37}_Action{i % 11}_{i}.npy" for i in range(3000)]
    eligible = {name for name in names if int(name.rsplit("_", 1)[1][:-4]) % 2 == 0}
    dealt = assign(names, val_eligible=eligible)
    assert set(dealt["val"]) <= eligible
    assert set(dealt["val"]) == set(assign(names)["val"]) & eligible, (
        "the gate only holds ineligible clips back; it must not move eligible ones"
    )
    # No eligible clip at all: val stays empty rather than pulling a rare clip.
    assert assign(names[:10], val_eligible=set())["val"] == []
    # Eligible clips exist but the hash gave val none: the fallback picks one of them.
    few = assign(names[:10], val_eligible={names[3]})
    assert few["val"] == [names[3]]


def test_val_eligibility_counts_label_buckets_across_sources(monkeypatch):
    monkeypatch.setattr(dataset_module, "VAL_BUCKET_MIN_CLIPS", 3)
    # Bucket = head words only. ("attack", "jump") has 4 clips over two sources
    # (> 3, eligible) whatever modifier / direction words ride along;
    # ("attack",) alone is a different bucket with 2 clips; ("walk",) 1 clip;
    # unlabeled never eligible.
    meta = {
        "a": {
            "x1.npy": {"action_label": "attack, jump"},
            "x2.npy": {"action_label": "attack, jump, spin, right"},
            "x3.npy": {"action_label": "attack"},
            "x4.npy": {"action_label": "walk, fast"},
            "x5.npy": {},
        },
        "b": {
            "y1.npy": {"action_label": "attack, jump, left"},
            "y2.npy": {"action_label": "attack, jump, charge"},
            "y3.npy": {"action_label": "attack, bite"},
        },
    }
    names = {ns: set(entries) for ns, entries in meta.items()}
    eligible = dataset_module.val_eligible_motion_names(names, meta)
    assert eligible == {"a": {"x1.npy", "x2.npy"}, "b": {"y1.npy", "y2.npy"}}
    # Per source alone, ("attack", "jump") has only 2 clips: not eligible.
    assert dataset_module.val_eligible_motion_names({"a": names["a"]}, meta) == {"a": set()}
    bucket = dataset_module.action_label_split_bucket
    assert bucket({"action_label": "attack, jump, spin, right"}) == ("attack", "jump")
    assert bucket({"action_label": "walk, fast, left"}) == ("walk",)
    assert bucket({}) is None


def test_empty_val_across_all_sources_is_its_own_error(tmp_path):
    motions_dir, metadata = _write_species_motions(tmp_path)  # unlabeled -> nothing eligible

    class Source:
        namespace, root, motion_dir = "s", str(tmp_path), str(motions_dir)

    with pytest.raises(dataset_module.EmptySplitError):
        dataset_module.load_allowed_motion_names_per_source("val", [Source()], "", {"s": metadata})
    # train still resolves, with every clip.
    allowed = dataset_module.load_allowed_motion_names_per_source("train", [Source()], "", {"s": metadata})
    assert len(allowed["s"]) == 8


def test_clip_split_is_stable_when_other_clips_come_and_go():
    assign = dataset_module.assign_clips_to_splits
    names = [f"Species{i % 37}_Action{i % 11}_{i}.npy" for i in range(3000)]
    before = assign(names)
    # A regen that drops a third of the clips and adds new ones.
    after = assign(names[::3] + [f"NewSpecies_Walk_{i}.npy" for i in range(500)])
    for split in ("train", "val"):
        kept = set(before[split]) & set(names[::3])
        assert kept <= set(after[split])
    # ... and the val share is still the ratio, not a count re-dealt from scratch.
    frac = len(before["val"]) / len(names)
    assert 0.01 <= frac <= 0.03, frac


def test_tiny_sets_never_leave_a_non_zero_split_empty():
    assign = dataset_module.assign_clips_to_splits
    assert assign(["A_Walk_1.npy"]) == {"train": ["A_Walk_1.npy"], "val": [], "test": []}
    two = assign(["A_Walk_1.npy", "B_Walk_1.npy"])
    assert len(two["train"]) == 1 and len(two["val"]) == 1 and two["test"] == []
    assert assign([]) == {"train": [], "val": [], "test": []}


def test_exact_case_wins_over_a_folded_match():
    # The zoo carries both "Rhino" and "rhino"; folding first would hand every
    # rhino_*.npy to whichever the cond lists first.
    lookup = {"Rhino": "truebones/zoo/Rhino", "rhino": "truebones/zoo_upgrade/rhino"}
    assert (
        infer_object_type_from_filename("rhino_Walk_1.npy", valid_types=lookup)
        == "truebones/zoo_upgrade/rhino"
    )
    assert (
        infer_object_type_from_filename("Rhino_Walk_1.npy", valid_types=lookup)
        == "truebones/zoo/Rhino"
    )
    # A casing that matches neither exactly still resolves case-insensitively.
    assert infer_object_type_from_filename("RHINO_Walk_1.npy", valid_types=lookup) in set(
        lookup.values()
    )


def test_a_file_named_after_the_species_alone_resolves():
    lookup = {"Deer": "truebones/zoo/Deer", "Deer_Buck": "truebones/zoo_upgrade/Deer_Buck"}
    assert (
        infer_object_type_from_filename("Deer_Buck.glb", valid_types=lookup)
        == "truebones/zoo_upgrade/Deer_Buck"
    )
    assert (
        infer_object_type_from_filename("Deer_Buck_Walk_1.npy", valid_types=lookup)
        == "truebones/zoo_upgrade/Deer_Buck"
    )
    assert (
        infer_object_type_from_filename("Deer_Walk_1.npy", valid_types=lookup)
        == "truebones/zoo/Deer"
    )
    assert (
        infer_object_type_from_filename("Horse.npy", valid_types={"Horse": "truebones/zoo/Horse"})
        == "truebones/zoo/Horse"
    )


def test_grouping_key_is_namespace_free_from_either_branch(tmp_path):
    # A metadata entry that ever carried a namespaced key must group with the
    # registry branch, not beside it.
    metadata = {"Horse_Run_1.npy": {"object_type": "truebones/zoo/Horse"}}
    assert resolve_motion_object_type("Horse_Run_1.npy", str(tmp_path), metadata) == "Horse"
    assert (
        resolve_motion_object_type(
            "Horse_Run_1.npy", str(tmp_path), {}, {"Horse": "truebones/zoo/Horse"}
        )
        == "Horse"
    )
