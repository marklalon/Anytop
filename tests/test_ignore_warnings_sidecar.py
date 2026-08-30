from pathlib import Path
import sys

import numpy as np
import pytest


ANYTOP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ANYTOP_ROOT.parent
for path in (REPO_ROOT, ANYTOP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from motion_lib.Animation import Animation  # noqa: E402
from motion_lib.Quaternions import Quaternions  # noqa: E402

from data_loaders.truebones.truebones_utils import dataset_tags as dt  # noqa: E402
from data_loaders.truebones.truebones_utils import features as features_module  # noqa: E402
from data_loaders.truebones.truebones_utils import ignore_warnings as iw  # noqa: E402
from data_loaders.truebones.truebones_utils import face_orientation as fo  # noqa: E402


@pytest.fixture(autouse=True)
def restore_default_configuration():
    """Every test points the process at a temporary dataset; restore the default."""
    yield
    dt.configure()
    fo._EMITTED_DEGENERATE_FACING_WARNINGS.clear()


def _make_dataset_dir(tmp_path, sidecar_text=None, *, sidecar_in_parent=False):
    """A minimal dataset directory, optionally carrying an ignore_warnings.txt."""
    dataset_dir = tmp_path / "processed"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / dt.SPECIES_TAGS_FILE).write_text(
        '{"species": "TestCreature", "species_tags": ["Quadruped", "Medium", "Striding"]}\n',
        encoding="utf-8",
    )
    if sidecar_text is not None:
        root = tmp_path if sidecar_in_parent else dataset_dir
        (root / iw.FILENAME).write_text(sidecar_text, encoding="utf-8")
    return dataset_dir


def test_parse_splits_stems_patterns_and_directives():
    sidecar = iw.parse(
        "\n"
        "Horse_Run_1\n"
        "  Horse_Walk_2.npy  \n"
        "# Bat: fell back to the across-vector heuristic\n"
        "!skip_orientation_detection\n"
        "!no-such-switch\n"
    )

    assert sidecar.stems == {"Horse_Run_1", "Horse_Walk_2"}
    assert sidecar.patterns == ("bat: fell back to the across-vector heuristic",)
    # Separator and case are free, so the '_' spelling is the same switch.
    assert sidecar.has(iw.SKIP_ORIENTATION_DETECTION)
    assert sidecar.unknown_directives == ("no-such-switch",)


def test_directive_line_is_not_read_as_a_motion_stem_or_pattern():
    sidecar = iw.parse("!skip-orientation-detection\n")

    assert sidecar.stems == frozenset()
    assert sidecar.patterns == ()


def test_sidecar_falls_back_to_the_parent_directory(tmp_path):
    dataset_dir = _make_dataset_dir(
        tmp_path, "!skip-orientation-detection\n", sidecar_in_parent=True
    )

    sidecar = iw.load(dataset_dir)

    assert sidecar.path == tmp_path / iw.FILENAME
    assert sidecar.has(iw.SKIP_ORIENTATION_DETECTION)


def test_dataset_own_sidecar_wins_over_the_parents(tmp_path):
    dataset_dir = _make_dataset_dir(tmp_path, "Horse_Run_1\n")
    (tmp_path / iw.FILENAME).write_text("!skip-orientation-detection\n", encoding="utf-8")

    sidecar = iw.load(dataset_dir)

    assert sidecar.path == dataset_dir / iw.FILENAME
    assert not sidecar.has(iw.SKIP_ORIENTATION_DETECTION)


def test_a_rewritten_sidecar_is_re_read(tmp_path):
    dataset_dir = _make_dataset_dir(tmp_path, "Horse_Run_1\n")
    assert not iw.load(dataset_dir).has(iw.SKIP_ORIENTATION_DETECTION)

    (dataset_dir / iw.FILENAME).write_text("!skip-orientation-detection\n", encoding="utf-8")

    assert iw.load(dataset_dir).has(iw.SKIP_ORIENTATION_DETECTION)


def test_skip_orientation_detection_follows_the_configured_dataset(tmp_path):
    switched = _make_dataset_dir(tmp_path / "switched", "!skip-orientation-detection\n")
    plain = _make_dataset_dir(tmp_path / "plain")

    dt.configure(dataset_dir=switched)
    assert iw.skip_orientation_detection()

    dt.configure(dataset_dir=plain)
    assert not iw.skip_orientation_detection()

    # Inference configures from a cond dict: no dataset directory, no switch.
    dt.configure_from_cond({"TestCreature": {"species_tags": ["Quadruped"]}})
    assert not iw.skip_orientation_detection()


def test_skip_orientation_detection_accepts_an_explicit_dataset_dir(tmp_path):
    """Validation asks about the directory it is validating, not the configured one.

    ``validate_anytop_dataset`` does now point the process at each dataset it
    validates, but the explicit form must still decide on its own: it is what
    keeps a caller that forgot to configure -- or one looping over several
    datasets -- from reading the wrong dataset's switch.
    """
    switched = _make_dataset_dir(tmp_path / "switched", "!skip-orientation-detection\n")
    plain = _make_dataset_dir(tmp_path / "plain")

    dt.configure(dataset_dir=plain)
    assert iw.skip_orientation_detection(switched)

    dt.configure(dataset_dir=switched)
    assert not iw.skip_orientation_detection(plain)


def test_facing_warnings_are_silenced_under_the_switch(tmp_path, capsys):
    dt.configure(dataset_dir=_make_dataset_dir(tmp_path / "switched", "!skip-orientation-detection\n"))
    fo._facing_warning("TestCreature", "across_selected", "TestCreature: across fallback")
    assert capsys.readouterr().out == ""

    dt.configure(dataset_dir=_make_dataset_dir(tmp_path / "plain"))
    fo._facing_warning("TestCreature", "across_selected", "TestCreature: across fallback")
    assert "TestCreature: across fallback" in capsys.readouterr().out


def _fake_sideways_rest_pose(monkeypatch):
    """A rest pose whose head points along +X, i.e. a quarter turn off canonical."""
    names = ["Root", "LeftLeg", "RightLeg", "Neck", "Head"]
    parents = np.array([-1, 0, 0, 0, 3], dtype=np.int32)
    offsets = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    loaded_anim = Animation(
        Quaternions.id((1, len(names))),
        offsets[None].copy(),
        Quaternions.id(len(names)),
        offsets,
        parents,
    )
    monkeypatch.setattr(
        features_module.FBX,
        "load",
        lambda _path: (loaded_anim, names, 1.0 / 30.0),
    )


def test_rest_pose_orientation_is_identity_under_the_switch(tmp_path, monkeypatch):
    _fake_sideways_rest_pose(monkeypatch)
    identity = np.array([1.0, 0.0, 0.0, 0.0])

    dt.configure(dataset_dir=_make_dataset_dir(tmp_path / "plain"))
    detected = features_module.get_common_features_from_rest_pose("fake.fbx", "TestCreature")
    # Guard the fixture: without the switch this rest pose is genuinely corrected.
    assert not np.allclose(np.asarray(detected.orientation_quat.qs).reshape(4), identity)

    dt.configure(dataset_dir=_make_dataset_dir(tmp_path / "switched", "!skip-orientation-detection\n"))
    skipped = features_module.get_common_features_from_rest_pose("fake.fbx", "TestCreature")

    np.testing.assert_allclose(
        np.asarray(skipped.orientation_quat.qs).reshape(4), identity, atol=1e-12
    )
    # The face/forward joints are still resolved: motion validation compares each
    # clip's recovered facing against the rest pose through them.
    assert skipped.forward_joint_index == detected.forward_joint_index
