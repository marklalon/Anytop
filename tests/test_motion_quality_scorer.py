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
from eval.motion_quality.reference_bank import ReferenceClip
from data_loaders.truebones.truebones_utils.dataset_pipeline import get_mean_std, POS_RESIDUAL_STD_FLOOR


def _make_random_motion(seed: int, t_len: int, joint_count: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((t_len, joint_count, 13), dtype=np.float32)


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


def test_get_mean_std_preserves_translation_root_position_stats() -> None:
    data = np.zeros((5, 3, 13), dtype=np.float32)
    data[:, 0, :3] = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 2.0, 0.5],
            [2.0, 3.0, 1.0],
            [3.0, 4.0, 1.5],
            [4.0, 5.0, 2.0],
        ],
        dtype=np.float32,
    )
    data[:, 1, :3] = np.array(
        [
            [10.0, 2.0, 0.0],
            [11.0, 3.0, 1.0],
            [12.0, 4.0, 2.0],
            [13.0, 5.0, 3.0],
            [14.0, 6.0, 4.0],
        ],
        dtype=np.float32,
    )
    data[:, 2, :3] = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1e-4, 0.0, -1e-4],
            [0.0, 0.0, 0.0],
            [-1e-4, 0.0, 1e-4],
        ],
        dtype=np.float32,
    )

    mean, std = get_mean_std(data, preserve_position_rows=[1])

    np.testing.assert_allclose(mean[1, :3], data[:, 1, :3].mean(axis=0), atol=1e-7)
    assert np.all(std[1, :3] > POS_RESIDUAL_STD_FLOOR)
    np.testing.assert_allclose(mean[2, :3], np.zeros(3, dtype=np.float32), atol=1e-7)
    np.testing.assert_allclose(std[2, :3], np.full(3, POS_RESIDUAL_STD_FLOOR, dtype=np.float32), atol=1e-7)


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
        "_recover_analysis_positions",
        lambda motion, *_args, **_kwargs: np.zeros((motion.shape[0], motion.shape[1], 3), dtype=np.float32),
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


def test_low_shot_reconstructs_spatial_series_before_local_metrics(
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
            "translation_root_index": 0,
        }
    }
    scorer._resolve_joint_groups = lambda _object_type, _n_joints: (
        {
            "root": np.array([0], dtype=np.int64),
            "limbs": np.array([1], dtype=np.int64),
        },
        "test",
    )

    monkeypatch.setattr(
        scorer_mod,
        "_recover_analysis_positions",
        lambda motion, *_args, **_kwargs: np.zeros((motion.shape[0], motion.shape[1], 3), dtype=np.float32),
    )

    observed_dims = []

    def fake_compute_features_batch(motions, _nperseg):
        observed_dims.append([motion.shape[-1] for motion in motions])
        return [
            {
                "spectral_flatness": np.zeros(motions[0].shape[1], dtype=np.float64),
                "jerk_norm": np.zeros(motions[0].shape[1], dtype=np.float64),
                "snap_norm": np.zeros(motions[0].shape[1], dtype=np.float64),
            }
            for _ in motions
        ]

    monkeypatch.setattr(scorer_mod, "_compute_features_batch", fake_compute_features_batch)
    monkeypatch.setattr(
        scorer_mod,
        "_score_query_against_reference",
        lambda *_args, **_kwargs: {
            "reference_median": 0.0,
            "scale": 1.0,
            "normalized_deviation": 0.0,
            "score": 1.0,
        },
    )
    monkeypatch.setattr(
        scorer_mod,
        "_score_bone_length_from_drift",
        lambda *_args, **_kwargs: {
            "score": 1.0,
            "median_abs_drift_pct": 0.0,
            "mean_abs_drift_pct": 0.0,
            "max_abs_drift_pct": 0.0,
            "score_median_abs": 1.0,
            "score_mean_abs": 1.0,
            "score_max_abs": 1.0,
        },
    )

    query_motions = [np.zeros((8, 2, 13), dtype=np.float32)]
    reference_clips = [
        ReferenceClip(
            path="ref.npy",
            object_type="wolf",
            motion_name="ref.npy",
            n_frames=8,
            weight=1.0,
            motion=np.zeros((8, 2, 13), dtype=np.float32),
        )
    ]

    scorer._compute_low_shot(
        query_motions,
        np.array([1.0], dtype=np.float64),
        reference_clips,
        np.array([1.0], dtype=np.float64),
        8,
        {"root": np.array([0], dtype=np.int64), "limbs": np.array([1], dtype=np.int64)},
        "wolf",
    )

    assert observed_dims == [[3], [3]]