from __future__ import annotations

import os
import sys

import json

import numpy as np
import pytest


_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ANYTOP_ROOT = os.path.dirname(_TESTS_DIR)
_REPO_ROOT = os.path.dirname(_ANYTOP_ROOT)

for _path in [_REPO_ROOT, _ANYTOP_ROOT]:
    if _path not in sys.path:
        sys.path.insert(0, _path)


import eval.evaluate_motion_quality as eval_mod
from eval.motion_quality.quality import MotionQualityReport


def _make_cond_entry(embedding: np.ndarray) -> dict:
    return {
        "parents": np.array([-1, 0], dtype=np.int32),
        "offsets": np.zeros((2, 3), dtype=np.float64),
        "joints_names": ["root", "RightThigh"],
        "joints_names_embs": np.tile(np.asarray(embedding, dtype=np.float64), (2, 1)),
    }


def _fake_scorer_class(captured: dict[str, object]):
    class FakeScorer:
        def __init__(self, dataset_root=None):
            captured["dataset_root"] = dataset_root

        def register_cond(self, cond):
            captured["cond"] = cond

        def species_lookup(self):
            return {}

        def evaluate(self, motions, object_type, action_label, names=None):
            captured["object_type"] = object_type
            captured["action_label"] = action_label
            captured["n_motions"] = len(motions)
            captured["names"] = list(names or [])
            return MotionQualityReport(
                object_type=object_type,
                action_label=action_label,
                subset="quadruped",
                score=0.5,
                dimensions={"physics": 0.25, "prior": 1.0},
                terms={"physics": {"rigidity": 0.25}, "prior": {"coordination": 1.0}},
                clips=[
                    {"file": name, "score": 0.5, "physics": 0.25, "prior": 1.0}
                    for name in (names or [str(i) for i in range(len(motions))])
                ],
            )

    return FakeScorer


def test_main_registers_cond_path_for_novel_query_species(tmp_path, monkeypatch) -> None:
    motion_path = tmp_path / "dragon_0.npy"
    cond_path = tmp_path / "cond.npy"

    np.save(motion_path, np.zeros((8, 2, 12), dtype=np.float32))
    cond_dict = {"dragon": _make_cond_entry(np.array([0.0, 1.0], dtype=np.float64))}
    np.save(cond_path, cond_dict, allow_pickle=True)

    captured: dict[str, object] = {}
    monkeypatch.setattr(eval_mod, "MotionQualityScorer", _fake_scorer_class(captured))

    exit_code = eval_mod.main([
        "--motions", str(motion_path),
        "--object-type", "dragon",
        "--action-label", "fly, forward",
        "--cond-path", str(cond_path),
        "--no_color",
    ])

    assert exit_code == 0
    loaded_cond = captured["cond"]
    assert set(loaded_cond.keys()) == {"dragon"}
    np.testing.assert_array_equal(loaded_cond["dragon"]["parents"], cond_dict["dragon"]["parents"])
    np.testing.assert_allclose(loaded_cond["dragon"]["offsets"], cond_dict["dragon"]["offsets"])
    assert loaded_cond["dragon"]["joints_names"] == cond_dict["dragon"]["joints_names"]
    np.testing.assert_allclose(
        loaded_cond["dragon"]["joints_names_embs"],
        cond_dict["dragon"]["joints_names_embs"],
    )
    assert captured["object_type"] == "dragon"
    assert captured["action_label"] == "fly, forward"
    assert captured["n_motions"] == 1
    # The prior is pooled over every dataset by default.
    assert str(captured["dataset_root"]).replace("\\", "/").endswith("dataset/datasets_all.jsonl")


def test_main_without_action_label_is_rejected(tmp_path, monkeypatch, capsys) -> None:
    motion_path = tmp_path / "Buffalo_0.npy"
    np.save(motion_path, np.zeros((8, 2, 12), dtype=np.float32))

    captured: dict[str, object] = {}
    monkeypatch.setattr(eval_mod, "MotionQualityScorer", _fake_scorer_class(captured))

    # argparse exits before anything is scored: there is no default prior.
    with pytest.raises(SystemExit) as exc_info:
        eval_mod.main([
            "--motions", str(motion_path),
            "--object-type", "Buffalo",
            "--no_color",
        ])

    assert exc_info.value.code == 2
    assert "--action_label" in capsys.readouterr().err
    assert "action_label" not in captured


def test_eval_checkpoint_scores_each_task_with_its_own_action_label() -> None:
    from eval import eval_checkpoint

    label = eval_checkpoint._task_score_label
    assert label(["--object_type", "Buffalo", "--action_label", "run", "--loop"], None) == "run"
    assert label(["--cond_path", "cond.npy", "--action_label", "fly, forward"], None) == "fly, forward"
    # A reference-only task names the action of its reference clip.
    assert label(["--reference_motion", "Buffalo_RunLoop.npy", "--loop"], "run") == "run"
    assert label(["--reference_motion", "Buffalo_RunLoop.npy"], "  run ") == "run"


def test_eval_checkpoint_rejects_a_task_without_a_scoring_label() -> None:
    from eval import eval_checkpoint

    label = eval_checkpoint._task_score_label
    for args, eval_label in [
        (["--object_type", "Buffalo"], None),
        (["--action_label", ""], None),
        (["--action_label"], None),
        (["--reference_motion", "x.npy"], ""),
    ]:
        with pytest.raises(ValueError, match="no --action_label and no eval_label"):
            label(args, eval_label)
    # Both at once is a contradiction; a typo in either fails the contract.
    with pytest.raises(ValueError, match="both"):
        label(["--action_label", "run"], "run")
    with pytest.raises(ValueError, match="controlled vocabulary"):
        label(["--reference_motion", "x.npy"], "wlak")
    with pytest.raises(ValueError, match="no head word"):
        label(["--action_label", "forward"], None)


def test_eval_checkpoint_config_load_fails_fast_on_a_label_less_task(tmp_path) -> None:
    from eval import eval_checkpoint

    config = {
        "checkpoint": {"RUN_NAME": "run"},
        "tasks": [
            {"category": "Basic", "args": ["--object_type", "Buffalo", "--action_label", "run"]},
            {"category": "Inpaint", "args": ["--object_type", "Buffalo", "--inpaint_frames", "1-2"]},
        ],
    }
    config_path = tmp_path / "tasks.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match=r"Task #1 \(Inpaint\).*no --action_label and no eval_label"):
        eval_checkpoint._load_task_config(config_path)

    config["tasks"][1]["eval_label"] = "run"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    _checkpoint, tasks = eval_checkpoint._load_task_config(config_path)
    assert [(category, score_label) for category, _args, score_label in tasks] == [
        ("Basic", "run"),
        ("Inpaint", "run"),
    ]


def test_shipped_task_configs_all_carry_a_scoring_label() -> None:
    from eval import eval_checkpoint

    for name in ("eval_tasks.json",):
        _checkpoint, tasks = eval_checkpoint._load_task_config(eval_checkpoint._SCRIPT_DIR / name)
        assert all(score_label for _category, _args, score_label in tasks), name


def test_eval_checkpoint_output_dir_overrides_the_run_folder(tmp_path) -> None:
    from eval import eval_checkpoint

    resolve = eval_checkpoint._resolve_output_root
    eval_root = eval_checkpoint._ANYTOP_DIR / "outputs" / "eval_checkpoint"

    # Default: the run name, with one subdir per checkpoint file.
    assert resolve({}, "merged_all_v22", "model000400000") == (
        eval_root / "merged_all_v22" / "model000400000"
    ).resolve()
    # A bare name is a sibling run folder, so one checkpoint's batteries keep
    # their own task dirs (Basic / NewSkeleton) and their own report.
    assert resolve({"OUTPUT_DIR": "merged_all_v22_transition"}, "merged_all_v22", "m") == (
        eval_root / "merged_all_v22_transition" / "m"
    ).resolve()
    # A value with a separator is a path relative to the Anytop dir, so it is
    # not nested under outputs/eval_checkpoint a second time.
    assert resolve({"OUTPUT_DIR": "outputs/eval_checkpoint/foo"}, "run", "m") == (
        eval_root / "foo" / "m"
    ).resolve()
    # An absolute value is taken as-is.
    assert resolve({"OUTPUT_DIR": str(tmp_path / "abs")}, "run", "m") == (
        tmp_path / "abs" / "m"
    ).resolve()


def test_eval_checkpoint_config_rejects_an_empty_output_dir(tmp_path) -> None:
    from eval import eval_checkpoint

    config = {
        "checkpoint": {"RUN_NAME": "run", "OUTPUT_DIR": "   "},
        "tasks": [{"category": "Basic", "args": ["--object_type", "Buffalo", "--action_label", "run"]}],
    }
    config_path = tmp_path / "tasks.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint.OUTPUT_DIR must be a non-empty string"):
        eval_checkpoint._load_task_config(config_path)


def test_main_scores_the_files_of_one_species_as_one_set(tmp_path, monkeypatch) -> None:
    for i in range(3):
        np.save(tmp_path / f"Buffalo_{i}.npy", np.zeros((8, 2, 12), dtype=np.float32))
    output_json = tmp_path / "out.json"

    captured: dict[str, object] = {}
    monkeypatch.setattr(eval_mod, "MotionQualityScorer", _fake_scorer_class(captured))

    exit_code = eval_mod.main([
        "--motions", str(tmp_path / "Buffalo_*.npy"),
        "--object-type", "Buffalo",
        "--action-label", "run, forward",
        "--output_json", str(output_json),
        "--no_color",
    ])

    assert exit_code == 0
    assert captured["n_motions"] == 3
    assert captured["names"] == ["Buffalo_0.npy", "Buffalo_1.npy", "Buffalo_2.npy"]
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert len(payload["sets"]) == 1
    assert payload["sets"][0]["files"] == ["Buffalo_0.npy", "Buffalo_1.npy", "Buffalo_2.npy"]
    assert payload["sets"][0]["score"]["score"] == 0.5
    assert payload["sets"][0]["score"]["physics"] == 0.25
