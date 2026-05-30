from __future__ import annotations

import os
import sys

import numpy as np
import pytest


_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ANYTOP_ROOT = os.path.dirname(_TESTS_DIR)
_REPO_ROOT = os.path.dirname(_ANYTOP_ROOT)

for _path in [_REPO_ROOT, _ANYTOP_ROOT]:
    if _path not in sys.path:
        sys.path.insert(0, _path)


import eval.motion_quality.scorer as scorer_mod
import eval.motion_quality.reference_bank as reference_bank_mod
from eval.motion_quality.reference_bank import ReferenceClip, ReferenceSpeciesSummary, WeightedReferenceBank


def _make_random_motion(seed: int, t_len: int, joint_count: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((t_len, joint_count, 13), dtype=np.float32)


def _make_cond_entry(embedding: np.ndarray, joint_count: int = 2) -> dict:
    return {
        "parents": np.array([-1, 0], dtype=np.int32)[:joint_count],
        "offsets": np.zeros((joint_count, 3), dtype=np.float64),
        "joints_names": ["root", "RightThigh"][:joint_count],
        "joints_names_embs": np.tile(np.asarray(embedding, dtype=np.float64), (joint_count, 1)),
    }


def test_bone_length_score_keeps_valid_zero_drift_clips(monkeypatch: pytest.MonkeyPatch) -> None:
    motions = [np.zeros((2, 1, 13), dtype=np.float32) for _ in range(2)]
    drifts = [
        np.array([[0.0], [0.0]], dtype=np.float64),
        np.array([[0.0], [0.0]], dtype=np.float64),
    ]

    def fake_compute_drift(_motion: np.ndarray, _parents: np.ndarray, _offsets: np.ndarray) -> np.ndarray:
        return drifts.pop(0)

    monkeypatch.setattr(scorer_mod, "_compute_bone_length_drift_from_motion", fake_compute_drift)

    result = scorer_mod._score_bone_length_from_drift(
        motions,
        np.array([0.4, 0.6], dtype=np.float64),
        np.array([-1], dtype=np.int32),
        np.zeros((1, 3), dtype=np.float64),
    )

    assert result["median_abs_drift_pct"] == pytest.approx(0.0)
    assert result["mean_abs_drift_pct"] == pytest.approx(0.0)
    assert result["max_abs_drift_pct"] == pytest.approx(0.0)
    assert result["score"] == pytest.approx(scorer_mod._sigmoid_score(0.0))


def test_bone_length_score_uses_surviving_clip_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    motions = [np.zeros((2, 1, 13), dtype=np.float32) for _ in range(4)]
    drifts = [
        np.full((2, 0), np.nan, dtype=np.float64),
        np.array([[0.0], [0.2]], dtype=np.float64),
        np.full((2, 0), np.nan, dtype=np.float64),
        np.array([[0.0], [1.0]], dtype=np.float64),
    ]

    def fake_compute_drift(_motion: np.ndarray, _parents: np.ndarray, _offsets: np.ndarray) -> np.ndarray:
        return drifts.pop(0)

    monkeypatch.setattr(scorer_mod, "_compute_bone_length_drift_from_motion", fake_compute_drift)

    result = scorer_mod._score_bone_length_from_drift(
        motions,
        np.array([0.1, 0.7, 0.1, 0.1], dtype=np.float64),
        np.array([-1], dtype=np.int32),
        np.zeros((1, 3), dtype=np.float64),
    )

    assert result["median_abs_drift_pct"] == pytest.approx(15.0)
    assert result["mean_abs_drift_pct"] == pytest.approx(15.0)
    assert result["max_abs_drift_pct"] == pytest.approx(30.0)


def test_compute_features_batch_matches_single_for_same_shape() -> None:
    motions = [
        _make_random_motion(0, 12, 5),
        _make_random_motion(1, 12, 5),
        _make_random_motion(2, 12, 5),
    ]

    batch_results = scorer_mod._compute_features_batch(motions, nperseg=8)

    for motion, batch_features in zip(motions, batch_results):
        single_features = scorer_mod._compute_features(motion, nperseg=8)
        for key in ["spectral_flatness", "jerk_norm", "snap_norm"]:
            np.testing.assert_allclose(batch_features[key], single_features[key], rtol=1e-5, atol=1e-7)


def test_compute_features_batch_matches_single_for_mixed_shapes() -> None:
    motions = [
        _make_random_motion(10, 4, 2),
        _make_random_motion(11, 5, 2),
        _make_random_motion(12, 8, 3),
        _make_random_motion(13, 8, 3),
    ]

    batch_results = scorer_mod._compute_features_batch(motions, nperseg=8)

    for motion, batch_features in zip(motions, batch_results):
        single_features = scorer_mod._compute_features(motion, nperseg=8)
        for key in ["spectral_flatness", "jerk_norm", "snap_norm"]:
            np.testing.assert_allclose(batch_features[key], single_features[key], rtol=1e-5, atol=1e-7)


def test_reference_species_selection_accepts_external_query_cond() -> None:
    baseline_cond = {
        "horse": _make_cond_entry(np.array([1.0, 0.0], dtype=np.float64)),
        "snake": _make_cond_entry(np.array([0.0, 1.0], dtype=np.float64)),
    }
    query_cond = _make_cond_entry(np.array([0.0, 1.0], dtype=np.float64))

    selected = reference_bank_mod._select_species_weights(
        query_object_type="dragon",
        action_tags="locomotion",
        action_paths_by_species={"horse": ["horse.npy"], "snake": ["snake.npy"]},
        cond_lookup=baseline_cond,
        top_k_species=1,
        query_cond=query_cond,
    )

    assert selected == [("snake", pytest.approx(0.0), 1.0)]


def test_registered_cond_is_query_only_reference_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_cond = {"horse": _make_cond_entry(np.array([1.0, 0.0], dtype=np.float64))}
    custom_cond = {"dragon": _make_cond_entry(np.array([0.0, 1.0], dtype=np.float64))}

    scorer = object.__new__(scorer_mod.DistributionMotionQualityScorer)
    scorer.dataset_root = None
    scorer._cond_lookup = dict(baseline_cond)
    scorer._query_cond_lookup = dict(baseline_cond)
    scorer._custom_cond_keys = set()
    scorer._joint_group_cache = {}
    scorer.register_cond(custom_cond)

    captured: dict[str, object] = {}

    def fake_build_weighted_reference_bank(**kwargs) -> WeightedReferenceBank:
        captured.update(kwargs)
        return WeightedReferenceBank(
            dataset_root="test",
            object_type=str(kwargs["object_type"]),
            action_tags=str(kwargs["action_tags"]),
            top_k_species=int(kwargs["top_k_species"]),
            clips=[
                ReferenceClip(
                    path="horse.npy",
                    object_type="horse",
                    motion_name="horse",
                    n_frames=8,
                    weight=1.0,
                    motion=np.zeros((8, 2, 13), dtype=np.float32),
                )
            ],
            species=[
                ReferenceSpeciesSummary(
                    object_type="horse",
                    cosine_distance=0.0,
                    species_weight=1.0,
                    clip_count=1,
                    total_frames=8,
                )
            ],
        )

    def fake_compute_low_shot(*_args, **_kwargs) -> dict:
        return {
            "score": 1.0,
            "component_scores": {
                "spectral_flatness": 1.0,
                "jerk_norm": 1.0,
                "snap_norm": 1.0,
                "bone_length": 1.0,
            },
            "raw": {},
        }

    monkeypatch.setattr(scorer_mod, "build_weighted_reference_bank", fake_build_weighted_reference_bank)
    monkeypatch.setattr(scorer_mod.DistributionMotionQualityScorer, "_compute_low_shot", fake_compute_low_shot)

    report = scorer.evaluate(
        motions=[np.zeros((8, 2, 13), dtype=np.float32)],
        object_type="dragon",
        action_tags="locomotion",
        top_k_species=1,
    )

    assert report.object_type == "dragon"
    assert "dragon" not in scorer._cond_lookup
    assert captured["cond_lookup"] is scorer._cond_lookup
    assert "dragon" not in captured["cond_lookup"]
    assert captured["query_cond"] is scorer._query_cond_lookup["dragon"]


def test_low_shot_bone_length_is_global_but_contributes_to_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = object.__new__(scorer_mod.DistributionMotionQualityScorer)
    scorer._cond_lookup = {
        "wolf": {
            "parents": np.array([-1, 0], dtype=np.int32),
            "offsets": np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            ),
        }
    }
    scorer._query_cond_lookup = dict(scorer._cond_lookup)
    scorer._resolve_joint_groups = lambda _object_type, _n_joints: (
        {
            "root": np.array([0], dtype=np.int64),
            "limbs": np.array([1], dtype=np.int64),
        },
        "test",
    )

    monkeypatch.setattr(
        scorer_mod,
        "_compute_features_batch",
        lambda motions, _nperseg: [
            {
                "spectral_flatness": np.zeros(motions[0].shape[1], dtype=np.float64),
                "jerk_norm": np.zeros(motions[0].shape[1], dtype=np.float64),
                "snap_norm": np.zeros(motions[0].shape[1], dtype=np.float64),
            }
            for _ in motions
        ],
    )
    monkeypatch.setattr(
        scorer_mod,
        "_score_query_against_reference",
        lambda *_args, **_kwargs: {
            "reference_median": 0.0,
            "scale": 1.0,
            "normalized_deviation": 0.0,
            "score": 0.0,
        },
    )
    monkeypatch.setattr(
        scorer_mod,
        "_score_bone_length_from_drift",
        lambda *_args, **_kwargs: {
            "score": 0.5,
            "median_abs_drift_pct": 0.0,
            "mean_abs_drift_pct": 0.0,
            "max_abs_drift_pct": 0.0,
            "score_median_abs": 0.5,
            "score_mean_abs": 0.5,
            "score_max_abs": 0.5,
        },
    )

    query_joint_groups = {
        "root": np.array([0], dtype=np.int64),
        "limbs": np.array([1], dtype=np.int64),
    }
    query_motions = [np.zeros((8, 2, 13), dtype=np.float32)]
    reference_clips = [
        ReferenceClip(
            path="ref.npy",
            object_type="wolf",
            motion_name="ref",
            n_frames=8,
            weight=1.0,
            motion=np.zeros((8, 2, 13), dtype=np.float32),
        )
    ]

    result = scorer._compute_low_shot(
        query_motions,
        np.array([1.0], dtype=np.float64),
        reference_clips,
        np.array([1.0], dtype=np.float64),
        8,
        query_joint_groups,
        "wolf",
    )

    assert result["component_scores"]["bone_length"] == pytest.approx(0.5)
    assert result["score"] == pytest.approx(0.228 * 0.5)
    assert result["raw"]["joint_group_scores"]["root"] == pytest.approx(0.0)
    assert result["raw"]["joint_group_scores"]["limbs"] == pytest.approx(0.0)