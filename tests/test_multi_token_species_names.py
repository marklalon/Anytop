"""Species names that contain underscores must never be truncated.

The UnityBundles dataset names every clip ``<Pack>_<Species>_<Action>_<id>``
(``FEP_MagmaDemon_Attack01_1.npy``), so any "everything before the first
underscore" shortcut resolves a whole asset pack to a single pseudo-species.
"""

import sys
from pathlib import Path

import numpy as np

ANYTOP_ROOT = Path(__file__).resolve().parents[1]
if str(ANYTOP_ROOT) not in sys.path:
    sys.path.insert(0, str(ANYTOP_ROOT))

import data_loaders.truebones.data.dataset as dataset_module  # noqa: E402
from data_loaders.truebones.data.dataset import (  # noqa: E402
    ensure_split_manifests,
    resolve_motion_object_type,
)
from data_loaders.truebones.truebones_utils.motion_labels import (  # noqa: E402
    infer_action_tags_from_clip_name,
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


def test_action_tags_ignore_the_species_name():
    # 'Sting' / 'DeathMage' are species names, not actions.
    assert infer_action_tags_from_clip_name("MU01_Sting_Idle_1.npy", "MU01_Sting") == ["idle"]
    assert infer_action_tags_from_clip_name("MU06_DeathMage_Idle_1.npy", "MU06_DeathMage") == ["idle"]
    assert infer_action_tags_from_clip_name("MU06_Death_TurnLeft_1.npy", "MU06_Death") == ["turn"]
    # Single-token species keep working, with or without the hint.
    assert infer_action_tags_from_clip_name("Horse_Run_3.npy", "Horse") == ["locomotion"]
    assert infer_action_tags_from_clip_name("Horse_Run_3.npy") == ["locomotion"]


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
            np.save(motions_dir / name, np.zeros((4, 3, 13), dtype=np.float32))
            metadata[name] = {"object_type": species}
    return motions_dir, metadata


def _clips_per_split(split_paths):
    return {
        split: [line for line in path.read_text(encoding="utf-8").split() if line]
        for split, path in split_paths.items()
    }


def test_split_manifests_hold_out_whole_species_not_whole_packs(tmp_path, monkeypatch):
    # The default ratios put everything in train (val/test 0.0), which holds out
    # nothing; restore a val split to exercise the per-species grouping.
    monkeypatch.setattr(
        dataset_module, "DEFAULT_SPLIT_RATIOS", {"train": 0.95, "val": 0.05, "test": 0.0}
    )
    motions_dir, metadata = _write_species_motions(tmp_path)

    split_paths = ensure_split_manifests(str(tmp_path), str(motions_dir), metadata)
    clips_per_split = _clips_per_split(split_paths)

    # 4 species -> 3 train / 1 val. Grouping by pack prefix would give 2 groups
    # and put a whole pack (4 clips) in val.
    assert len(clips_per_split["val"]) == 2
    assert len(clips_per_split["train"]) == 6
    for clips in clips_per_split.values():
        species_in_split = {"_".join(name.split("_")[:2]) for name in clips}
        for species in species_in_split:
            assert sum(1 for name in clips if name.startswith(f"{species}_")) == 2


def test_default_split_ratios_put_everything_in_train(tmp_path):
    # val/test ratios are 0.0 on purpose: the eval split is empty and validation
    # is skipped gracefully. The manifests still exist, just empty.
    motions_dir, metadata = _write_species_motions(tmp_path)

    split_paths = ensure_split_manifests(str(tmp_path), str(motions_dir), metadata)
    clips_per_split = _clips_per_split(split_paths)

    assert len(clips_per_split["train"]) == 8
    assert clips_per_split["val"] == []
    assert clips_per_split["test"] == []


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


def test_species_prefix_is_stripped_case_insensitively():
    # The filename -> species inference folds case, so the prefix strip must too:
    # "deer_buck_Idle_1.npy" resolves to "Deer_Buck", and a case-sensitive test
    # would leave 'buck' in the action tokens (tagging the clip "emote").
    for clip in ("Deer_Buck_Idle_1.npy", "deer_buck_Idle_1.npy", "DEER_BUCK_Idle_1.npy"):
        assert infer_action_tags_from_clip_name(clip, "Deer_Buck") == ["idle"]


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
