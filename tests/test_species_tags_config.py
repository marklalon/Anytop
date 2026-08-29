from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest


ANYTOP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ANYTOP_ROOT.parent
for path in (REPO_ROOT, ANYTOP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from data_loaders.truebones.truebones_utils import dataset_tags as dt  # noqa: E402
from data_loaders.truebones.truebones_utils.face_orientation import (  # noqa: E402
    _get_facing_candidates,
)


@pytest.fixture(autouse=True)
def restore_default_configuration():
    """Every test configures a sidecar; put the repository default back after."""
    yield
    dt.configure()


def _write_species_tags(tmp_path):
    path = tmp_path / "species_tags.jsonl"
    path.write_text(
        '{"species": "Kappa_gorilla", "species_tags": ["Biped", "Medium", "Striding"]}\n'
        '{"species": "Pikefish", "species_tags": ["Aquatic", "Small", "Swimming"]}\n',
        encoding="utf-8",
    )
    return path


def test_configure_loads_custom_species_tags(tmp_path):
    dt.configure(species_tags_file=_write_species_tags(tmp_path))

    tags = dt.dataset_tags()
    assert tags.species_tags == {
        "Kappa_gorilla": ("Biped", "Medium", "Striding"),
        "Pikefish": ("Aquatic", "Small", "Swimming"),
    }
    assert tags.object_subsets["biped"] == ["Kappa_gorilla"]
    assert tags.object_subsets["aquatic"] == ["Pikefish"]
    # Canonical subsets exist even with no member species.
    assert tags.object_subsets["serpentine"] == []
    # Derived views come off the same snapshot, so they cannot drift.
    assert tags.subset_members["aquatic"] == frozenset({"Pikefish"})
    assert tags.object_subset_for("kappa_GORILLA") == "biped"
    assert tags.tags_for("Pikefish") == ("Aquatic", "Small", "Swimming")
    assert tags.object_subset_for("Nonesuch") is None
    # A selector is a subset key or a bare species name.
    assert tags.species_for("biped") == ["Kappa_gorilla"]
    assert tags.species_for("Pikefish") == ["Pikefish"]


def test_configure_does_no_io_until_first_use(tmp_path):
    """configure() only records sources, so it is safe before consumers exist."""
    missing = tmp_path / "absent_species_tags.jsonl"

    paths = dt.configure(species_tags_file=missing)
    assert paths.species_tags == missing

    with pytest.raises(FileNotFoundError):
        dt.dataset_tags()


def test_snapshot_is_cached_until_reconfigured(tmp_path):
    dt.configure(species_tags_file=_write_species_tags(tmp_path))
    first = dt.dataset_tags()
    assert dt.dataset_tags() is first

    dt.configure(species_tags_file=_write_species_tags(tmp_path))
    assert dt.dataset_tags() is not first


def test_configure_loads_custom_chain_forward_joints(tmp_path):
    joints_path = tmp_path / "chain_forward_joints.jsonl"
    joints_path.write_text(
        '{"species": "Crow", "chain_forward_joints": [28, 31]}\n', encoding="utf-8"
    )

    dt.configure(chain_forward_joints_file=joints_path)

    assert dt.dataset_tags().chain_forward_joints == {"Crow": (28, 31)}
    joints = np.zeros((1, 32, 3), dtype=float)
    joints[:, 28] = [0.0, 0.0, 0.0]
    joints[:, 31] = [0.0, 0.0, -1.0]
    candidates = _get_facing_candidates(joints, "Crow")
    assert set(candidates) == {"chain"}
    assert candidates["chain"][0, 2] == -1.0


def test_absent_default_chain_forward_sidecar_is_optional(tmp_path):
    """A dataset with no overrides falls through to semantic detection."""
    (tmp_path / "species_tags.jsonl").write_text(
        '{"species": "Crow", "species_tags": ["Winged", "Small", "Flying"]}\n',
        encoding="utf-8",
    )

    dt.configure(dataset_dir=tmp_path)

    assert dt.dataset_tags().chain_forward_joints == {}


def test_explicitly_requested_chain_forward_sidecar_must_exist(tmp_path):
    """A typo'd --chain-forward-joints-file must fail, not silently degrade."""
    dt.configure(chain_forward_joints_file=tmp_path / "typo.jsonl")

    with pytest.raises(FileNotFoundError):
        dt.dataset_tags()


def test_register_species_tags_rebuilds_derived_views():
    dt.configure()
    dt.register_species_tags("Kappa_gorilla", ("Biped", "Medium", "Striding"))

    tags = dt.dataset_tags()
    assert tags.species_tags["Kappa_gorilla"] == ("Biped", "Medium", "Striding")
    assert "Kappa_gorilla" in tags.object_subsets["biped"]
    assert tags.object_subset_for("Kappa_gorilla") == "biped"
    dt.assert_species_tags_cover(["Kappa_gorilla"])


def test_assert_species_tags_cover_reports_unregistered_species():
    dt.configure()
    with pytest.raises(SystemExit) as excinfo:
        dt.assert_species_tags_cover(["Nonesuch_beast"])
    assert "Nonesuch_beast" in str(excinfo.value)


def test_tag_modules_are_never_imported_twice():
    """The whole configure()-sync problem came from duplicate module copies.

    Short-name imports of a package module (``param_utils`` next to
    ``data_loaders.truebones.truebones_utils.param_utils``) give two module
    objects with two sets of globals, so a configured snapshot in one is
    invisible to consumers bound to the other. Entry points must import
    package-qualified only.
    """
    for stem in ("param_utils", "dataset_tags", "face_orientation"):
        loaded = sorted(
            name
            for name, module in sys.modules.items()
            if module is not None and name.split(".")[-1] == stem
            and getattr(module, "__file__", "")
            and Path(module.__file__).parent.name == "truebones_utils"
        )
        assert len(loaded) <= 1, f"{stem} was imported under several names: {loaded}"


def test_entry_point_imports_a_single_module_copy():
    """``preprocess_and_validate`` must not resurrect the short-name imports.

    Runs the real entry-point import in a fresh interpreter (its sys.path setup
    is what used to create the duplicates) and checks the loaded module names.
    """
    probe = (
        "import runpy, sys, importlib.util;"
        "spec = importlib.util.spec_from_file_location("
        "'preprocess_and_validate', r'{path}');"
        "module = importlib.util.module_from_spec(spec);"
        "spec.loader.exec_module(module);"
        "print(sorted(n for n in sys.modules if n.endswith('param_utils')))"
    ).format(path=ANYTOP_ROOT / "preprocess_and_validate.py")
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=str(ANYTOP_ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith(
        "['data_loaders.truebones.truebones_utils.param_utils']"
    ), result.stdout
