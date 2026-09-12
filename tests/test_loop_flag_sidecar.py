"""``is_loop`` is an annotation in action_labels.jsonl, not a metadata field.

Preprocessing PROPOSES it (the detector's verdict, written into a row that has
none) and a person VERIFIES it in the review UI. Once a row carries the flag,
it is the truth everywhere: the terminal velocity row extraction writes, the
loop-period table regeneration bakes, the training loader's loop path. Nothing
downstream re-derives it, and motion_metadata.json no longer stores a copy.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion_lib.Animation import Animation
from motion_lib.Quaternions import Quaternions

from data_loaders.truebones.truebones_utils import dataset_pipeline
from data_loaders.truebones.truebones_utils.animation_utils import (
    detect_loop_from_features,
)
from data_loaders.truebones.truebones_utils.features import (
    extract_motion_features_from_aligned_anims,
)
from data_loaders.truebones.truebones_utils.loop_verdict import (
    apply_loop_verdict,
    rewrite_terminal_row,
    terminal_velocity_for_verdict,
)
from data_loaders.truebones.truebones_utils.motion_labels import (
    LOOP_FLAG_KEY,
    MOTION_METADATA_SCHEMA_VERSION,
    fill_missing_loop_flags,
    load_action_labels,
    load_motion_metadata,
    write_motion_metadata,
)


# ── fixtures ──────────────────────────────────────────────────────────────

def _swing_anim(angles: np.ndarray) -> Animation:
    """A root-spine-tip rig whose spine sweeps ``angles`` (see test_boundary_frame_trim)."""
    n_frames = len(angles)
    parents = np.array([-1, 0, 1], dtype=np.int64)
    offsets = np.array(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    rotations = Quaternions(
        np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_frames, len(parents), 1))
    )
    rotations[:, 1] = Quaternions.from_angle_axis(angles, np.array([1.0, 0.0, 0.0]))
    positions = np.tile(offsets, (n_frames, 1, 1))
    return Animation(rotations, positions, Quaternions.id(len(parents)), offsets, parents)


def _cycle_anim(n_frames: int = 24) -> Animation:
    """One closed swing cycle: the detector reads it as a loop."""
    phase = 2.0 * np.pi * np.arange(n_frames) / n_frames
    return _swing_anim(np.radians(10.0) * np.sin(phase))


def _open_anim(n_frames: int = 24) -> Animation:
    """A one-way sweep: the detector reads it as a one-shot clip."""
    return _swing_anim(np.linspace(0.0, np.radians(60.0), n_frames))


def _extract(anim: Animation, **kwargs):
    features, _max_joints, _anim, _export, is_loop, _flattened = (
        extract_motion_features_from_aligned_anims(
            anim,
            anim,
            object_type='TestSkeleton',
            max_joints=8,
            orientation_quat=Quaternions.id(1).qs[0],
            translation_root_index=0,
            **kwargs,
        )
    )
    return features, is_loop


def _write_labels(path: Path, rows, newline="\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline=newline) as handle:
        handle.write("\n".join(json.dumps(row) for row in rows) + "\n")


def _read_labels(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ── the sidecar ───────────────────────────────────────────────────────────

def test_the_sidecar_carries_the_flag_only_when_a_row_has_it(tmp_path):
    # The sidecar is keyed by the extension-less clip name.
    _write_labels(tmp_path / "action_labels.jsonl", [
        {"clip": "a", "action_group": "stationary", "action_label": "idle", "is_loop": True},
        {"clip": "b", "action_group": "transition", "action_label": "die", "is_loop": False},
        {"clip": "c", "action_group": "locomotion", "action_label": "walk"},
    ])
    labels = load_action_labels(tmp_path)
    assert labels["a"][LOOP_FLAG_KEY] is True
    assert labels["b"][LOOP_FLAG_KEY] is False
    # Absent means "not judged yet" -- distinguishable from either verdict.
    assert LOOP_FLAG_KEY not in labels["c"]


def test_a_legacy_row_spelling_the_npy_name_is_normalized_to_the_stem_key(tmp_path):
    _write_labels(tmp_path / "action_labels.jsonl", [
        {"clip": "a.npy", "action_group": "stationary", "action_label": "idle", "is_loop": True},
    ])
    labels = load_action_labels(tmp_path)
    assert set(labels) == {"a"}
    assert labels["a"][LOOP_FLAG_KEY] is True


@pytest.mark.parametrize("bad", ["true", 1, None, "yes"])
def test_a_flag_that_is_not_a_json_bool_is_refused(tmp_path, bad):
    _write_labels(tmp_path / "action_labels.jsonl", [
        {"clip": "a.npy", "action_group": "stationary", "action_label": "idle", "is_loop": bad},
    ])
    with pytest.raises(SystemExit):
        load_action_labels(tmp_path)


def test_the_join_takes_the_flag_from_the_sidecar_and_drops_a_stale_copy(tmp_path):
    (tmp_path / "motion_metadata.json").write_text(json.dumps({
        "schema_version": 6, "total_clips": 1,
        # A copy an older build baked in, contradicting the sidecar.
        "motions": {"a.npy": {"object_type": "A", "is_loop": False, "translation_root_index": 0}},
    }), encoding="utf-8")
    _write_labels(tmp_path / "action_labels.jsonl", [
        {"clip": "a.npy", "action_group": "stationary", "action_label": "idle", "is_loop": True},
    ])
    entry = load_motion_metadata(tmp_path)["a.npy"]
    assert entry[LOOP_FLAG_KEY] is True


def test_the_join_refuses_a_clip_on_disk_with_no_verdict(tmp_path):
    (tmp_path / "motion_metadata.json").write_text(json.dumps({
        "schema_version": 7, "total_clips": 1,
        "motions": {"a.npy": {"object_type": "A", "translation_root_index": 0}},
    }), encoding="utf-8")
    _write_labels(tmp_path / "action_labels.jsonl", [
        {"clip": "a.npy", "action_group": "stationary", "action_label": "idle"},
    ])
    # A default here would train an unannotated loop as a one-shot clip.
    with pytest.raises(SystemExit):
        load_motion_metadata(tmp_path)
    # The pre-fill bookkeeping read is allowed to see the row without one.
    lenient = load_motion_metadata(tmp_path, require_loop_flag=False)
    assert LOOP_FLAG_KEY not in lenient["a.npy"]


def test_the_metadata_writer_strips_the_flag(tmp_path):
    write_motion_metadata(tmp_path, {"a.npy": {"object_type": "A", "is_loop": True}}, 1)
    payload = json.loads((tmp_path / "motion_metadata.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == MOTION_METADATA_SCHEMA_VERSION
    assert LOOP_FLAG_KEY not in payload["motions"]["a.npy"]


def test_fill_writes_only_the_rows_without_a_verdict_and_keeps_everything_else(tmp_path):
    labels = tmp_path / "action_labels.jsonl"
    _write_labels(labels, [
        {"clip": "a.npy", "action_group": "stationary", "action_label": "idle", "reviewed": True},
        {"clip": "b.npy", "action_group": "transition", "action_label": "die", "is_loop": True},
        {"clip": "c.npy", "action_group": "locomotion", "action_label": "walk", "pending_delete": True},
        {"clip": "d.npy", "action_group": "locomotion", "action_label": "run"},
    ], newline="\r\n")
    before = labels.read_bytes()

    filled = fill_missing_loop_flags(tmp_path, {"a.npy": True, "b.npy": False, "c.npy": False})

    assert filled == 2
    rows = _read_labels(labels)
    # Row order and the other keys survive; the flag sits right after the label.
    assert [row["clip"] for row in rows] == ["a.npy", "b.npy", "c.npy", "d.npy"]
    assert list(rows[0]) == ["clip", "action_group", "action_label", "is_loop", "reviewed"]
    assert rows[0][LOOP_FLAG_KEY] is True and rows[0]["reviewed"] is True
    # An existing verdict is an annotation: never overwritten.
    assert rows[1][LOOP_FLAG_KEY] is True
    assert rows[2][LOOP_FLAG_KEY] is False and rows[2]["pending_delete"] is True
    # No verdict offered -> still unjudged.
    assert LOOP_FLAG_KEY not in rows[3]
    # The file's newline style is kept.
    assert b"\r\n" in labels.read_bytes()
    # A second pass with the same verdicts is a no-op.
    assert fill_missing_loop_flags(tmp_path, {"a.npy": False}) == 0
    assert _read_labels(labels)[0][LOOP_FLAG_KEY] is True
    assert labels.read_bytes() != before


# ── extraction honours the verdict ────────────────────────────────────────

def test_extraction_defers_to_the_sidecar_verdict_and_writes_its_terminal_row():
    cycle = _cycle_anim()
    detected, detected_loop = _extract(cycle)
    assert detected_loop is True
    # Told it is not a loop, the clip ships a one-shot terminal row: the previous
    # frame's velocity, not the wrap delta.
    forced, forced_loop = _extract(cycle, is_loop=False)
    assert forced_loop is False
    assert np.array_equal(forced[-1, :, 9:12], forced[-2, :, 9:12])
    assert not np.array_equal(forced[-1, :, 9:12], detected[-1, :, 9:12])
    # Everything before the terminal row is the same clip.
    assert np.array_equal(forced[:-1], detected[:-1])

    sweep = _open_anim()
    _open, open_loop = _extract(sweep)
    assert open_loop is False
    as_loop, as_loop_flag = _extract(sweep, is_loop=True)
    assert as_loop_flag is True
    # The terminal row is now the wrap delta pos[0] - pos[-1] (no root travel here).
    np.testing.assert_allclose(as_loop[-1, :, 9:12], as_loop[0, :, 0:3] - as_loop[-1, :, 0:3], atol=1e-6)


def test_the_stored_tensor_detector_agrees_with_extraction():
    for anim in (_cycle_anim(), _open_anim()):
        features, is_loop = _extract(anim)
        assert detect_loop_from_features(features, translation_root_index=0) is is_loop


# ── the terminal row recomputed from the tensor ───────────────────────────

@pytest.mark.parametrize("anim_factory", [_cycle_anim, _open_anim])
@pytest.mark.parametrize("is_loop", [True, False])
def test_the_closed_form_terminal_row_matches_what_extraction_wrote(anim_factory, is_loop):
    features, _ = _extract(anim_factory(), is_loop=is_loop)
    recomputed = terminal_velocity_for_verdict(features, is_loop, translation_root_index=0)
    np.testing.assert_allclose(recomputed, features[-1, :, 9:12], atol=1e-6)


def test_the_closed_form_row_accounts_for_root_travel():
    # A root that walks +X while the spine swings: the wrap delta is the RIC
    # delta minus the travel every joint's RIC position hides.
    anim = _cycle_anim()
    travel = np.linspace(0.0, 0.5, anim.rotations.shape[0])
    anim.positions[:, 0, 0] += travel
    features, _ = _extract(anim, is_loop=True)
    recomputed = terminal_velocity_for_verdict(features, True, translation_root_index=0)
    np.testing.assert_allclose(recomputed, features[-1, :, 9:12], atol=1e-6)


def test_applying_a_verdict_flips_only_the_terminal_row_and_round_trips(tmp_path):
    features, _ = _extract(_cycle_anim(), is_loop=True)
    as_one_shot = apply_loop_verdict(features, False, 0)
    assert as_one_shot.dtype == features.dtype
    assert np.array_equal(as_one_shot[:-1], features[:-1])
    assert np.array_equal(as_one_shot[-1, :, :9], features[-1, :, :9])
    assert not np.array_equal(as_one_shot[-1, :, 9:12], features[-1, :, 9:12])
    np.testing.assert_allclose(apply_loop_verdict(as_one_shot, True, 0), features, atol=1e-6)

    path = tmp_path / "clip.npy"
    np.save(path, features)
    # The pipeline-written loop row matches to rounding, not bitwise: still a no-op.
    assert rewrite_terminal_row(path, True, 0) is False
    assert np.array_equal(np.load(path), features)
    assert rewrite_terminal_row(path, False, 0) is True
    assert np.array_equal(np.load(path), as_one_shot)
    assert rewrite_terminal_row(path, False, 0) is False
    assert not list(tmp_path.glob("*.tmp*"))


# ── backfilling clips already on disk ─────────────────────────────────────

def _stored_dataset(tmp_path, clips):
    """A processed dir with stored tensors, metadata and an unjudged sidecar."""
    motions = tmp_path / "motions"
    motions.mkdir(parents=True)
    metadata, rows = {}, []
    for clip, (object_type, anim) in clips.items():
        features, _ = _extract(anim)
        np.save(motions / clip, features)
        metadata[clip] = {"object_type": object_type, "translation_root_index": 0}
        rows.append({"clip": clip, "action_group": "stationary", "action_label": "idle"})
    write_motion_metadata(tmp_path, metadata, len(metadata))
    _write_labels(tmp_path / "action_labels.jsonl", rows)


def test_backfill_judges_stored_tensors_and_skips_species_about_to_be_rebuilt(tmp_path):
    _stored_dataset(tmp_path, {
        "Cat_Cycle.npy": ("Cat", _cycle_anim()),
        "Cat_Sweep.npy": ("Cat", _open_anim()),
        "Dog_Cycle.npy": ("Dog", _cycle_anim()),
    })
    # A row for a clip that is not on disk yet: left for the build that writes it.
    labels = tmp_path / "action_labels.jsonl"
    rows = _read_labels(labels)
    rows.append({"clip": "Cat_New.npy", "action_group": "stationary", "action_label": "idle"})
    _write_labels(labels, rows)

    filled = dataset_pipeline.backfill_loop_flags_from_stored_clips(tmp_path, exclude_object_types=("Dog",))

    assert filled == 2
    by_clip = {row["clip"]: row for row in _read_labels(labels)}
    assert by_clip["Cat_Cycle.npy"][LOOP_FLAG_KEY] is True
    assert by_clip["Cat_Sweep.npy"][LOOP_FLAG_KEY] is False
    assert LOOP_FLAG_KEY not in by_clip["Dog_Cycle.npy"]
    assert LOOP_FLAG_KEY not in by_clip["Cat_New.npy"]
    # Now the strict join is satisfied for what is on disk (Dog aside).
    assert dataset_pipeline.backfill_loop_flags_from_stored_clips(tmp_path) == 1
    assert {name: entry[LOOP_FLAG_KEY] for name, entry in load_motion_metadata(tmp_path).items()} == {
        "Cat_Cycle.npy": True, "Cat_Sweep.npy": False, "Dog_Cycle.npy": True,
    }


def test_load_loop_verdicts_returns_only_judged_rows(tmp_path):
    _write_labels(tmp_path / "action_labels.jsonl", [
        # Legacy .npy spelling: the verdicts are keyed by the extension-less name.
        {"clip": "a.npy", "action_group": "stationary", "action_label": "idle", "is_loop": True},
        {"clip": "b.npy", "action_group": "stationary", "action_label": "idle"},
    ])
    assert dataset_pipeline.load_loop_verdicts(tmp_path) == {"a": True}


# ── the review server ─────────────────────────────────────────────────────

def _load_serve():
    serve_path = Path(__file__).parents[1] / "dataset" / "review" / "serve.py"
    spec = importlib.util.spec_from_file_location("action_review_serve_loop", serve_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_the_review_store_writes_the_flag_and_never_pops_it(tmp_path):
    review = _load_serve()
    labels = tmp_path / "action_labels.jsonl"
    _write_labels(labels, [
        {"clip": "a.npy", "action_group": "stationary", "action_label": "idle"},
    ])
    store = review.LabelStore(labels)
    assert store.update("a.npy", is_loop=True)[LOOP_FLAG_KEY] is True
    assert store.update("a.npy", is_loop=False)[LOOP_FLAG_KEY] is False
    assert _read_labels(labels)[0][LOOP_FLAG_KEY] is False
    # Other patches leave the verdict alone.
    assert store.update("a.npy", reviewed=True)[LOOP_FLAG_KEY] is False


def test_the_review_server_keeps_the_tensor_in_step_with_a_flipped_flag(tmp_path):
    review = _load_serve()
    features, _ = _extract(_cycle_anim(), is_loop=True)
    processed = tmp_path / "processed"
    (processed / "motions").mkdir(parents=True)
    np.save(processed / "motions" / "Cat_Cycle.npy", features)
    write_motion_metadata(processed, {"Cat_Cycle.npy": {"object_type": "Cat", "translation_root_index": 0}}, 1)
    ds = {"processed": str(processed), "metadata": processed / "motion_metadata.json"}

    assert review.Handler._sync_terminal_row(ds, "Cat_Cycle.npy", False) == ""
    assert np.array_equal(np.load(processed / "motions" / "Cat_Cycle.npy"), apply_loop_verdict(features, False, 0))
    assert review.Handler._sync_terminal_row(ds, "Cat_Cycle.npy", True) == ""
    np.testing.assert_allclose(np.load(processed / "motions" / "Cat_Cycle.npy"), features, atol=1e-6)
    # No tensor on disk: the flag still stands, and the caller is told.
    assert "NPY" in review.Handler._sync_terminal_row(ds, "Cat_Missing.npy", True)


def test_the_review_metadata_writer_strips_the_flag(tmp_path):
    review = _load_serve()
    path = tmp_path / "motion_metadata.json"
    review._write_metadata(path, {"motions": {"a.npy": {"object_type": "A", "is_loop": True}}})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == MOTION_METADATA_SCHEMA_VERSION
    assert LOOP_FLAG_KEY not in payload["motions"]["a.npy"]
