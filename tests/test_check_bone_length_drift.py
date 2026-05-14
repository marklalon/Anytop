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


from tools.check_bone_length_drift import MotionWorldData, ReferenceSkeleton, _compute_drift_report


def test_drift_report_uses_animation_first_frame_as_main_baseline() -> None:
    reference = ReferenceSkeleton(
        object_type="Unit",
        bone_names=["Root", "Child"],
        parents=np.array([-1, 0], dtype=np.int32),
        offsets=np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )
    motion = MotionWorldData(
        file_path="synthetic.glb",
        file_format="glb",
        bone_names=["Root", "Child"],
        parents=np.array([-1, 0], dtype=np.int32),
        world_positions=np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
            ],
            dtype=np.float64,
        ),
        sample_frames=[0.0, 1.0],
    )

    report = _compute_drift_report(reference, motion)
    drift = report["drift"]

    assert drift["reference_basis"] == "animation_first_frame"
    assert drift["baseline_frame_index"] == 0
    assert drift["max_abs_drift_pct"] == pytest.approx(50.0)
    assert drift["mean_abs_drift_pct"] == pytest.approx(25.0)
    assert drift["worst_bone"] == "Child"
    assert drift["worst_frame_index"] == 1
    assert report["first_frame_reference"]["mean_bone_length"] == pytest.approx(1.0)
    np.testing.assert_allclose(drift["per_bone"]["baseline_length"], [1.0])
    np.testing.assert_allclose(drift["per_bone"]["mean_length"], [1.25])


def test_drift_report_includes_first_frame_reference_stats() -> None:
    reference = ReferenceSkeleton(
        object_type="Unit",
        bone_names=["Root", "BoneA", "BoneB"],
        parents=np.array([-1, 0, 0], dtype=np.int32),
        offsets=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )
    motion = MotionWorldData(
        file_path="synthetic.glb",
        file_format="glb",
        bone_names=["Root", "BoneA", "BoneB"],
        parents=np.array([-1, 0, 0], dtype=np.int32),
        world_positions=np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ],
            dtype=np.float64,
        ),
        sample_frames=[0.0, 1.0],
    )

    report = _compute_drift_report(reference, motion)
    first_frame_reference = report["first_frame_reference"]

    assert first_frame_reference["frame_index"] == 0
    assert first_frame_reference["frame_value"] == pytest.approx(0.0)
    # BoneA length = 1.0, BoneB length = 1.0 (world distance from Root at frame 0)
    assert first_frame_reference["mean_bone_length"] == pytest.approx(1.0)