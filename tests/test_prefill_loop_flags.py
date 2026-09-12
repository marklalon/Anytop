"""``tools/prefill_loop_flags.py`` proposes ``is_loop`` ahead of preprocessing.

It selects the species whose sidecar rows still lack a verdict, runs each one
through preprocessing's own preparation with no verdict handed in (so the
detector judges every clip on the aligned animation), and writes the verdicts
into the rows that had none. A build then reads them and never writes the
sidecar itself.
"""

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
from data_loaders.truebones.truebones_utils import dataset_tags
from data_loaders.truebones.truebones_utils.features import (
    extract_motion_features_from_aligned_anims,
)
from data_loaders.truebones.truebones_utils.motion_labels import LOOP_FLAG_KEY
from tools import prefill_loop_flags


# ── fixtures ──────────────────────────────────────────────────────────────

def _swing_anim(angles: np.ndarray) -> Animation:
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


def _encoded(anim: Animation) -> dict:
    """What _encode_prepared_motion_file produces for a clip with no verdict."""
    features, _max_joints, _anim, _export, is_loop, _flattened = (
        extract_motion_features_from_aligned_anims(
            anim,
            anim,
            object_type="TestSkeleton",
            max_joints=8,
            orientation_quat=Quaternions.id(1).qs[0],
            translation_root_index=0,
            is_loop=None,
        )
    )
    return {"motion": features, "is_loop": is_loop, "translation_root_index": 0}


def _write_rows(path: Path, rows, newline="\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline=newline) as handle:
        handle.write("\n".join(json.dumps(row) for row in rows) + "\n")


def _read_rows(path: Path):
    return {
        row["clip"]: row
        for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    }


def _write_species_tags(dataset_dir: Path, species):
    _write_rows(
        dataset_dir / "species_tags.jsonl",
        [{"species": name, "species_tags": ["Quadruped", "Medium", "Striding"]} for name in species],
    )


def _raw_species(raw_root: Path, object_type: str, stems):
    folder = raw_root / object_type
    folder.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        (folder / f"{stem}.glb").write_bytes(b"")


@pytest.fixture
def isolated_dataset_tags(monkeypatch):
    """main() points the process at the temp dataset's sidecars; put it back after."""
    monkeypatch.setattr(dataset_tags, "_sources", dataset_tags._sources)
    monkeypatch.setattr(dataset_tags, "_snapshot", dataset_tags._snapshot)


def _fake_prepare(clips_by_object, calls):
    """A stand-in for _prepare_object_outputs: records the call, encodes canned anims.

    ``clips_by_object`` maps object -> {action: Animation}; the payload carries
    one result per action, encoded the way a verdict-less build would encode it.
    """

    def fake(object_type, max_joints, face_joints=None, fbxs_dir=None, t_pos_path=None,
             max_files=None, raw_data_dir=None, filter_min_length=10, resample_min_length=20,
             skip_source_paths=None, frozen_translation_root_index=None,
             frozen_promote_root_depth=None, locomotion_clips=frozenset(), loop_verdicts=None):
        calls.append({
            "object_type": object_type,
            "skip_source_paths": set(skip_source_paths or ()),
            "frozen_translation_root_index": frozen_translation_root_index,
            "frozen_promote_root_depth": frozen_promote_root_depth,
            "locomotion_clips": set(locomotion_clips),
            "loop_verdicts": dict(loop_verdicts or {}),
        })
        results = []
        for action, anim in clips_by_object[object_type].items():
            result = _encoded(anim)
            result["action"] = action
            results.append(result)
        return {
            "object_type": object_type,
            "object_cond": {"object_type": object_type, "translation_root_index": 0},
            "tpose_reference_path": None,
            "errors": {},
            "max_joints": 3,
            "results": results,
            "files_counter": len(results),
            "frames_counter": 0,
            "face_joints": face_joints,
            "motion_errors": [],
        }

    return fake


# ── selection ─────────────────────────────────────────────────────────────

def test_plans_only_the_species_with_pending_rows_under_the_frozen_root(tmp_path):
    raw = tmp_path / "raw"
    _raw_species(raw, "Cat", ["Walk", "Run", "Idle"])
    _raw_species(raw, "Dog", ["Idle"])
    _raw_species(raw, "Bird", ["Fly"])
    dataset_dir = tmp_path / "dataset"
    _write_rows(dataset_dir / "action_labels.jsonl", [
        {"clip": "Cat_Walk", "action_group": "locomotion", "action_label": "walk", "is_loop": True},
        {"clip": "Cat_Run", "action_group": "locomotion", "action_label": "run"},
        # Cat_Idle has no row: nothing to fill, reported for the labelling step.
        {"clip": "Dog_Idle", "action_group": "stationary", "action_label": "idle", "is_loop": True},
        {"clip": "Bird_Fly", "action_group": "locomotion", "action_label": "fly"},
    ])
    # Cat is already in cond with a frozen root; Bird is new.
    np.save(dataset_dir / "cond.npy", {
        "Cat": {"object_type": "Cat", "joints_names": ["Root", "Tail"],
                "parents": np.array([-1, 0]), "offsets": np.zeros((2, 3)),
                "translation_root_index": 1, "root_promote_depth": 0},
    })
    labels = dataset_pipeline.load_action_labels(dataset_dir)

    plans, warnings, known = prefill_loop_flags.plan_objects(
        dataset_dir, str(raw), ("Bird", "Cat", "Dog"), labels,
    )

    assert [plan.object_type for plan in plans] == ["Bird", "Cat"]
    bird, cat = plans
    assert bird.pending == {"Bird_Fly"}
    assert bird.frozen_translation_root_index is None and bird.skip_source_paths is None
    assert cat.pending == {"Cat_Run"}
    assert cat.frozen_translation_root_index == 1 and cat.frozen_promote_root_depth == 0
    # Under a frozen root only the pending sources are loaded.
    assert cat.skip_source_paths == {
        os.path.realpath(raw / "Cat" / "Walk.glb"),
        os.path.realpath(raw / "Cat" / "Idle.glb"),
    }
    assert known == {"Bird_Fly", "Cat_Walk", "Cat_Run", "Cat_Idle", "Dog_Idle"}
    assert len(warnings) == 1 and warnings[0].startswith("Cat_Idle: no row")


def test_rejudge_plans_every_clip_except_the_reviewed_ones(tmp_path):
    raw = tmp_path / "raw"
    _raw_species(raw, "Cat", ["Walk", "Run"])
    dataset_dir = tmp_path / "dataset"
    _write_rows(dataset_dir / "action_labels.jsonl", [
        {"clip": "Cat_Walk", "action_group": "locomotion", "action_label": "walk", "is_loop": True, "reviewed": True},
        {"clip": "Cat_Run", "action_group": "locomotion", "action_label": "run", "is_loop": True},
    ])
    labels = dataset_pipeline.load_action_labels(dataset_dir)

    plans, _warnings, _known = prefill_loop_flags.plan_objects(
        dataset_dir, str(raw), ("Cat",), labels, rejudge=True,
    )

    assert len(plans) == 1 and plans[0].pending == {"Cat_Run"}


# ── judging ───────────────────────────────────────────────────────────────

def test_judge_object_asks_the_detector_for_every_clip_and_reads_its_verdict(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        dataset_pipeline, "_prepare_object_outputs",
        _fake_prepare({"Cat": {"Cycle": _cycle_anim(), "Sweep": _open_anim()}}, calls),
    )
    plan = prefill_loop_flags.ObjectPlan(
        object_type="Cat",
        clips={"Cat_Cycle": "x/Cycle.glb", "Cat_Sweep": "x/Sweep.glb"},
        pending={"Cat_Cycle", "Cat_Sweep"},
        frozen_translation_root_index=0,
        frozen_promote_root_depth=0,
        skip_source_paths=None,
    )

    judgement = prefill_loop_flags.judge_object(
        plan,
        raw_data_dir=str(tmp_path),
        locomotion_clips=frozenset({"Cat_Sweep"}),
        filter_min_length=10,
        resample_min_length=20,
    )

    # The build's own preparation, with NO verdict handed in for any clip.
    assert calls == [{
        "object_type": "Cat",
        "skip_source_paths": set(),
        "frozen_translation_root_index": 0,
        "frozen_promote_root_depth": 0,
        "locomotion_clips": {"Cat_Sweep"},
        "loop_verdicts": {},
    }]
    assert judgement.verdicts == {"Cat_Cycle": True, "Cat_Sweep": False}
    # The report diagnostics are the detector's own numbers on the same tensor.
    assert judgement.diagnostics["Cat_Cycle"]["is_loop"] is True
    assert judgement.diagnostics["Cat_Sweep"]["is_loop"] is False
    assert judgement.diagnostics["Cat_Cycle"]["frames"] == 24


# ── the whole run ─────────────────────────────────────────────────────────

def test_main_fills_only_the_pending_rows_and_leaves_the_rest_byte_for_byte(monkeypatch, tmp_path, isolated_dataset_tags):
    raw = tmp_path / "raw"
    _raw_species(raw, "Cat", ["Cycle", "Sweep", "Walk"])
    _raw_species(raw, "Dog", ["Idle"])
    dataset_dir = tmp_path / "dataset"
    _write_rows(dataset_dir / "action_labels.jsonl", [
        {"clip": "Cat_Walk", "action_group": "locomotion", "action_label": "walk", "is_loop": False, "reviewed": True},
        {"clip": "Cat_Cycle", "action_group": "stationary", "action_label": "idle"},
        {"clip": "Cat_Sweep", "action_group": "transition", "action_label": "die", "pending_delete": True},
        {"clip": "Dog_Idle", "action_group": "stationary", "action_label": "idle", "is_loop": True},
    ], newline="\r\n")
    _write_species_tags(dataset_dir, ("Cat", "Dog"))
    before = (dataset_dir / "action_labels.jsonl").read_bytes()
    calls = []
    # Cat_Walk's canned anim would read as a loop -- but its row already has a
    # verdict, so the detector's reading of it must not land anywhere.
    monkeypatch.setattr(
        dataset_pipeline, "_prepare_object_outputs",
        _fake_prepare({"Cat": {"Cycle": _cycle_anim(), "Sweep": _open_anim(), "Walk": _cycle_anim()}}, calls),
    )
    report = tmp_path / "report.jsonl"

    code = prefill_loop_flags.main([
        "--dataset-dir", str(dataset_dir), "--raw-data-dir", str(raw),
        "--object-workers", "1", "--report", str(report),
    ])

    assert code == 0
    # Dog is fully annotated: never loaded.
    assert [call["object_type"] for call in calls] == ["Cat"]
    # Cat is not in cond yet, so every source is scanned for the root contract.
    assert calls[0]["skip_source_paths"] == set() and calls[0]["frozen_translation_root_index"] is None
    assert calls[0]["locomotion_clips"] == {"Cat_Walk"}
    rows = _read_rows(dataset_dir / "action_labels.jsonl")
    assert rows["Cat_Cycle"][LOOP_FLAG_KEY] is True
    assert rows["Cat_Sweep"][LOOP_FLAG_KEY] is False and rows["Cat_Sweep"]["pending_delete"] is True
    assert list(rows["Cat_Sweep"]) == ["clip", "action_group", "action_label", "is_loop", "pending_delete"]
    assert rows["Cat_Walk"][LOOP_FLAG_KEY] is False
    assert rows["Dog_Idle"][LOOP_FLAG_KEY] is True
    # Untouched lines are byte-identical and the newline style is kept.
    after_lines = (dataset_dir / "action_labels.jsonl").read_bytes().split(b"\r\n")
    before_lines = before.split(b"\r\n")
    assert after_lines[0] == before_lines[0] and after_lines[3] == before_lines[3]
    # The report covers every judged clip, borderline first, and says which rows it filled.
    report_rows = [json.loads(line) for line in report.read_text(encoding="utf-8").splitlines()]
    assert {row["clip"] for row in report_rows} == {"Cat_Cycle", "Cat_Sweep", "Cat_Walk"}
    margins = [abs(row["loop_margin"] - 1.0) for row in report_rows]
    assert margins == sorted(margins)
    assert {row["clip"]: row["pending"] for row in report_rows} == {
        "Cat_Cycle": True, "Cat_Sweep": True, "Cat_Walk": False,
    }

    # Everything is annotated now: a second run loads nothing and changes nothing.
    settled = (dataset_dir / "action_labels.jsonl").read_bytes()
    calls.clear()
    assert prefill_loop_flags.main([
        "--dataset-dir", str(dataset_dir), "--raw-data-dir", str(raw), "--object-workers", "1",
    ]) == 0
    assert calls == []
    assert (dataset_dir / "action_labels.jsonl").read_bytes() == settled


def test_dry_run_judges_but_writes_nothing(monkeypatch, tmp_path, isolated_dataset_tags):
    raw = tmp_path / "raw"
    _raw_species(raw, "Cat", ["Cycle"])
    dataset_dir = tmp_path / "dataset"
    _write_rows(dataset_dir / "action_labels.jsonl", [
        {"clip": "Cat_Cycle", "action_group": "stationary", "action_label": "idle"},
    ])
    _write_species_tags(dataset_dir, ("Cat",))
    before = (dataset_dir / "action_labels.jsonl").read_bytes()
    calls = []
    monkeypatch.setattr(
        dataset_pipeline, "_prepare_object_outputs",
        _fake_prepare({"Cat": {"Cycle": _cycle_anim()}}, calls),
    )

    code = prefill_loop_flags.main([
        "--dataset-dir", str(dataset_dir), "--raw-data-dir", str(raw),
        "--object-workers", "1", "--dry-run",
    ])

    assert code == 0
    assert len(calls) == 1
    assert (dataset_dir / "action_labels.jsonl").read_bytes() == before


def test_rejudge_flips_unreviewed_verdicts_and_keeps_reviewed_ones(monkeypatch, tmp_path, isolated_dataset_tags):
    raw = tmp_path / "raw"
    _raw_species(raw, "Cat", ["Cycle", "Sweep"])
    dataset_dir = tmp_path / "dataset"
    # Both rows carry the WRONG verdict for their canned anim; one is signed off.
    _write_rows(dataset_dir / "action_labels.jsonl", [
        {"clip": "Cat_Cycle", "action_group": "stationary", "action_label": "idle", "is_loop": False, "reviewed": True},
        {"clip": "Cat_Sweep", "action_group": "transition", "action_label": "die", "is_loop": True},
    ])
    _write_species_tags(dataset_dir, ("Cat",))
    calls = []
    monkeypatch.setattr(
        dataset_pipeline, "_prepare_object_outputs",
        _fake_prepare({"Cat": {"Cycle": _cycle_anim(), "Sweep": _open_anim()}}, calls),
    )

    # Without --rejudge there is nothing pending: the species is not even loaded.
    assert prefill_loop_flags.main([
        "--dataset-dir", str(dataset_dir), "--raw-data-dir", str(raw), "--object-workers", "1",
    ]) == 0
    assert calls == []

    code = prefill_loop_flags.main([
        "--dataset-dir", str(dataset_dir), "--raw-data-dir", str(raw),
        "--object-workers", "1", "--rejudge",
    ])

    assert code == 0
    rows = _read_rows(dataset_dir / "action_labels.jsonl")
    assert rows["Cat_Cycle"][LOOP_FLAG_KEY] is False and rows["Cat_Cycle"]["reviewed"] is True
    assert rows["Cat_Sweep"][LOOP_FLAG_KEY] is False


def test_a_pending_clip_the_pipeline_drops_is_reported_not_written(monkeypatch, tmp_path, isolated_dataset_tags, capsys):
    raw = tmp_path / "raw"
    _raw_species(raw, "Cat", ["Cycle", "Blink"])
    dataset_dir = tmp_path / "dataset"
    _write_rows(dataset_dir / "action_labels.jsonl", [
        {"clip": "Cat_Cycle", "action_group": "stationary", "action_label": "idle"},
        {"clip": "Cat_Blink", "action_group": "stationary", "action_label": "idle"},
    ])
    _write_species_tags(dataset_dir, ("Cat",))
    calls = []
    # The fake never produces Cat_Blink: too short for the length filter, say.
    monkeypatch.setattr(
        dataset_pipeline, "_prepare_object_outputs",
        _fake_prepare({"Cat": {"Cycle": _cycle_anim()}}, calls),
    )

    code = prefill_loop_flags.main([
        "--dataset-dir", str(dataset_dir), "--raw-data-dir", str(raw), "--object-workers", "1",
    ])

    assert code == 0
    rows = _read_rows(dataset_dir / "action_labels.jsonl")
    assert rows["Cat_Cycle"][LOOP_FLAG_KEY] is True
    assert LOOP_FLAG_KEY not in rows["Cat_Blink"]
    assert "Cat_Blink: no clip came out of the pipeline" in capsys.readouterr().out
