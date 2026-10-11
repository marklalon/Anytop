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


def test_dedicated_dataset_split(tmp_path):
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "validation"
    train_dir.mkdir()
    val_dir.mkdir()
    train_motions, train_metadata = _write_species_motions(train_dir)
    val_motions, val_metadata = _write_species_motions(val_dir)

    class Source:
        def __init__(self, namespace, root, motion_dir, split):
            self.namespace, self.root, self.motion_dir, self.split = namespace, str(root), str(motion_dir), split

    sources = [Source("first", train_dir, train_motions, "train"), Source("second", val_dir, val_motions, "val")]
    metadata = {"first": train_metadata, "second": val_metadata}
    train = dataset_module.load_allowed_motion_names_per_source("train", sources, "", metadata)
    val = dataset_module.load_allowed_motion_names_per_source("val", sources, "", metadata)
    all_clips = dataset_module.load_allowed_motion_names_per_source("all", sources, "", metadata)
    assert len(train["first"]) == len(val["second"]) == 8
    assert train["second"] == val["first"] == set()
    assert all_clips == {"first": train["first"], "second": val["second"]}
    assert not list(tmp_path.rglob("*.txt"))


def test_validation_placeholder_disables_eval(tmp_path):
    motions_dir, metadata = _write_species_motions(tmp_path)

    class Source:
        namespace, root, motion_dir, split = "training", str(tmp_path), str(motions_dir), "train"

    with pytest.raises(dataset_module.EmptySplitError):
        dataset_module.load_allowed_motion_names_per_source("val", [Source()], "", {"training": metadata})
    allowed = dataset_module.load_allowed_motion_names_per_source("train", [Source()], "", {"training": metadata})
    assert len(allowed["training"]) == 8


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


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("Pet_Kiki_A.glb", "Pet_Kiki_A"),
        ("Pet_Kiki_A_Tpose.glb", "Pet_Kiki_A"),
        ("FEP_MagmaDemon_Tpose.glb", "FEP_MagmaDemon"),
        ("Horse_Tpose.fbx", "Horse"),
        ("Wyvern-T-Pose.fbx", "Wyvern"),
        ("Elephant.rig.glb", "Elephant"),
        ("dragon.fbx", "dragon"),
    ],
)
def test_new_skeleton_species_is_the_whole_stem(filename, expected):
    # A new skeleton has no registry to match prefixes against, so the stem is
    # the species (as the raw directory name is in training), minus pose tokens.
    from tools.process_new_skeleton import species_from_tpose_stem

    assert species_from_tpose_stem(filename) == expected
