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


def test_bone_rotation_excess_uses_filtered_reference_weights() -> None:
    result = scorer_mod._score_bone_rotation_excess(
        query_values=np.array([2.0], dtype=np.float64),
        query_weights=np.array([1.0], dtype=np.float64),
        reference_series_with_weights=[
            (np.array([1.0], dtype=np.float64), 0.8),
            (np.array([10.0], dtype=np.float64), 0.1),
        ],
    )

    assert result["max_ref"] == pytest.approx(1.0)
    assert result["score"] == pytest.approx(1.0 / 6.0)
    assert result["normalized_excess"] == pytest.approx(1.0)


def test_local_low_shot_bone_length_is_global_but_contributes_to_score(
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
    scorer._resolve_macro_joint_groups = lambda _object_type, _n_joints: (
        {
            "root": np.array([0], dtype=np.int64),
            "limbs": np.array([1], dtype=np.int64),
        },
        "test",
    )

    monkeypatch.setattr(
        scorer_mod,
        "_compute_local_features",
        lambda _motion, _nperseg: {
            "spectral_flatness": np.zeros(2, dtype=np.float64),
            "jerk_norm": np.zeros(2, dtype=np.float64),
            "snap_norm": np.zeros(2, dtype=np.float64),
        },
    )
    monkeypatch.setattr(
        scorer_mod,
        "_compute_bone_rotation_angle",
        lambda motion, _parents: np.zeros((motion.shape[0], 1), dtype=np.float64),
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
        "_score_bone_rotation_excess",
        lambda *_args, **_kwargs: {
            "max_ref": 0.0,
            "normalized_excess": 0.0,
            "penalty": 1.0,
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

    result = scorer._compute_local_low_shot(
        query_motions,
        np.array([1.0], dtype=np.float64),
        reference_clips,
        np.array([1.0], dtype=np.float64),
        8,
        query_joint_groups,
        "wolf",
    )

    assert result["component_scores"]["bone_length"] == pytest.approx(0.5)
    assert result["score"] == pytest.approx(0.168 * 0.5)
    assert result["raw"]["local_joint_group_scores"]["root"] == pytest.approx(0.0)
    assert result["raw"]["local_joint_group_scores"]["limbs"] == pytest.approx(0.0)