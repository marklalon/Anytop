"""The dataset validator must read the sidecars of the dataset it was handed.

``dataset_tags`` is process-wide and falls back to ``DEFAULT_DATASET_DIR`` when
nothing configured it. Validating any other dataset under that fallback silently
borrows the default dataset's ``species_tags.jsonl`` and, worse, its
``chain_forward_joints.jsonl`` -- whose joint indices address one dataset's
collapsed joint order and mean something else entirely in another.
"""

from pathlib import Path
import sys

import pytest


ANYTOP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ANYTOP_ROOT.parent
for path in (REPO_ROOT, ANYTOP_ROOT, ANYTOP_ROOT / "utils"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from data_loaders.truebones.truebones_utils import dataset_tags as dt  # noqa: E402
import validate_anytop_dataset as validator  # noqa: E402


@pytest.fixture(autouse=True)
def restore_default_configuration():
    yield
    dt.configure()


def _make_dataset_dir(root: Path, chain_forward: str | None = None) -> Path:
    root.mkdir(parents=True)
    (root / dt.SPECIES_TAGS_FILE).write_text(
        '{"species": "Crow", "species_tags": ["Winged", "Small", "Flapping"]}\n',
        encoding="utf-8",
    )
    if chain_forward is not None:
        (root / dt.CHAIN_FORWARD_JOINTS_FILE).write_text(chain_forward, encoding="utf-8")
    return root


class _StopAfterConfigure(Exception):
    """Cuts the run short once the only thing under test has happened."""


class _Args:
    objects_subset = "all"
    sample_count = 0
    root_motion_threshold = 1.0
    motion_orientation_threshold = 45.0
    orientation_threshold_deg = 5.0


def test_validate_one_dataset_reads_that_dataset_s_sidecars(tmp_path, monkeypatch):
    # The dataset the process happens to be pointed at: it defines a forward
    # chain for a species name the validated dataset also uses.
    configured = _make_dataset_dir(
        tmp_path / "configured",
        '{"species": "Crow", "chain_forward_joints": [8, 22]}\n',
    )
    # The dataset actually being validated: same bare species name, its own
    # (differently ordered) skeleton, and deliberately no forward-chain override.
    target = _make_dataset_dir(tmp_path / "target")

    dt.configure(dataset_dir=str(configured))
    assert dt.dataset_tags().chain_forward_for("Crow") == (8, 22)

    observed = {}

    def _capture(*args, **kwargs):
        observed["chain_forward"] = dt.dataset_tags().chain_forward_for("Crow")
        observed["species_tags_path"] = dt.dataset_tags().species_tags
        raise _StopAfterConfigure

    monkeypatch.setattr(validator, "prepare_dataset_for_validation", _capture)

    with pytest.raises(_StopAfterConfigure):
        validator._validate_one_dataset(target, _Args())

    # Not (8, 22): those indices belong to the other dataset's joint order, and
    # applying them here is what turned the recovered-facing check into a false
    # positive on the zoo / zoo_upgrade Crow pair.
    assert observed["chain_forward"] is None
    assert set(observed["species_tags_path"]) == {"Crow"}
