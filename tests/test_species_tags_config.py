from pathlib import Path
import sys
import numpy as np


ANYTOP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ANYTOP_ROOT.parent
for path in (REPO_ROOT, ANYTOP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from data_loaders.truebones.truebones_utils.param_utils import (  # noqa: E402
    CHAIN_FORWARD_JOINTS,
    OBJECT_SUBSETS_DICT,
    SPECIES_TAGS,
    configure_chain_forward_joints,
    configure_species_tags,
    load_chain_forward_joints,
)
from data_loaders.truebones.truebones_utils.face_orientation import _get_facing_candidates  # noqa: E402


def test_configure_species_tags_loads_custom_sidecar(tmp_path):
    tags_path = tmp_path / "species_tags.jsonl"
    tags_path.write_text(
        '{"species": "Kappa_gorilla", "species_tags": '
        '["Biped", "Medium", "Striding"]}\n'
        '{"species": "Pikefish", "species_tags": '
        '["Aquatic", "Small", "Swimming"]}\n',
        encoding="utf-8",
    )

    try:
        configure_species_tags(species_tags_file=tags_path)

        assert SPECIES_TAGS == {
            "Kappa_gorilla": ("Biped", "Medium", "Striding"),
            "Pikefish": ("Aquatic", "Small", "Swimming"),
        }
        assert OBJECT_SUBSETS_DICT["biped"] == ["Kappa_gorilla"]
        assert OBJECT_SUBSETS_DICT["aquatic"] == ["Pikefish"]
    finally:
        # Restore the repository default for tests that run after this one.
        configure_species_tags()


def test_configure_chain_forward_joints_loads_custom_sidecar(tmp_path):
    joints_path = tmp_path / "chain_forward_joints.jsonl"
    joints_path.write_text(
        '{"species": "Crow", "chain_forward_joints": [28, 31]}\n',
        encoding="utf-8",
    )

    try:
        configure_chain_forward_joints(chain_forward_joints_file=joints_path)

        assert CHAIN_FORWARD_JOINTS == {"Crow": (28, 31)}
        joints = np.zeros((1, 32, 3), dtype=float)
        joints[:, 28] = [0.0, 0.0, 0.0]
        joints[:, 31] = [0.0, 0.0, -1.0]
        candidates = _get_facing_candidates(joints, "Crow")
        assert set(candidates) == {"chain"}
        assert candidates["chain"][0, 2] == -1.0
    finally:
        # Restore the repository default for tests that run after this one.
        configure_chain_forward_joints()


def test_missing_chain_forward_joints_sidecar_is_optional(tmp_path):
    missing_path = tmp_path / "missing_chain_forward_joints.jsonl"

    assert load_chain_forward_joints(chain_forward_joints_file=missing_path) == {}

    try:
        configure_chain_forward_joints(chain_forward_joints_file=missing_path)
        assert CHAIN_FORWARD_JOINTS == {}
    finally:
        configure_chain_forward_joints()
