from __future__ import annotations

import os
import sys

import numpy as np


_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ANYTOP_ROOT = os.path.dirname(_TESTS_DIR)
_REPO_ROOT = os.path.dirname(_ANYTOP_ROOT)

for _path in [_REPO_ROOT, _ANYTOP_ROOT]:
    if _path not in sys.path:
        sys.path.insert(0, _path)


import eval.evaluate_motion_quality as eval_mod
from eval.motion_quality.reference_bank import DEFAULT_SCORE_ACTION_LABEL
from eval.motion_quality.scorer import DistributionEvalReport


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

        def evaluate(self, motions, object_type, action_label, top_k_species):
            captured["object_type"] = object_type
            captured["action_label"] = action_label
            captured["top_k_species"] = top_k_species
            captured["n_motions"] = len(motions)
            return DistributionEvalReport(
                object_type=object_type,
                action_label=action_label,
                n_input=1,
                n_reference=1,
                input_total_frames=int(motions[0].shape[0]),
                reference_total_frames=8,
                scoring_mode="test",
                top_k_species=top_k_species,
                reference_species=[],
                overall_score=0.5,
                spectral_flatness_score=0.5,
                jerk_score=0.5,
                snap_score=0.5,
                bone_length_score=0.5,
                raw={},
            )

    return FakeScorer


def test_main_registers_cond_path_for_novel_query_species(tmp_path, monkeypatch) -> None:
    motion_path = tmp_path / "dragon_0.npy"
    cond_path = tmp_path / "cond.npy"

    np.save(motion_path, np.zeros((8, 2, 12), dtype=np.float32))
    cond_dict = {"dragon": _make_cond_entry(np.array([0.0, 1.0], dtype=np.float64))}
    np.save(cond_path, cond_dict, allow_pickle=True)

    captured: dict[str, object] = {}
    monkeypatch.setattr(eval_mod, "DistributionMotionQualityScorer", _fake_scorer_class(captured))

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
    assert captured["top_k_species"] == 3
    assert captured["n_motions"] == 1


def test_main_without_action_label_scores_against_the_default_prior(tmp_path, monkeypatch) -> None:
    motion_path = tmp_path / "Buffalo_0.npy"
    np.save(motion_path, np.zeros((8, 2, 12), dtype=np.float32))

    captured: dict[str, object] = {}
    monkeypatch.setattr(eval_mod, "DistributionMotionQualityScorer", _fake_scorer_class(captured))

    exit_code = eval_mod.main([
        "--motions", str(motion_path),
        "--object-type", "Buffalo",
        "--no_color",
    ])

    assert exit_code == 0
    assert captured["action_label"] == DEFAULT_SCORE_ACTION_LABEL == "walk, run"


def test_eval_checkpoint_scores_each_task_with_its_own_action_label() -> None:
    from eval import eval_checkpoint

    extract = eval_checkpoint._extract_action_label
    assert extract(["--object_type", "Buffalo", "--action_label", "run", "--loop"]) == "run"
    assert extract(["--cond_path", "cond.npy", "--action_label", "fly, forward"]) == "fly, forward"
    # No label (unconditional / reference-only tasks) keeps the walk/run prior.
    assert extract(["--object_type", "Buffalo"]) == DEFAULT_SCORE_ACTION_LABEL
    assert extract(["--action_label", ""]) == DEFAULT_SCORE_ACTION_LABEL
    assert extract(["--action_label"]) == DEFAULT_SCORE_ACTION_LABEL
