"""Key times a hair apart are one sample, not a frozen duplicate frame.

A glTF clip can key its channels through several time accessors holding the
same instants up to float32 noise. Scaled from seconds to scene frames, those
land ~1e-4 frames apart; sampling both repeats every such frame.
"""

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

from motion_lib.FBX import infer_sample_fps, merge_sample_times


class _Render:
    fps = 24
    fps_base = 1.0


class _Scene:
    render = _Render()


def test_float32_twin_accessors_merge_to_source_keys():
    seconds = np.arange(37, dtype=np.float64) / 30.0
    a = (seconds.astype(np.float32) * np.float32(24)).tolist()
    b = ((seconds + 3e-6).astype(np.float32) * np.float32(24)).tolist()

    times = merge_sample_times(a + b)

    assert len(times) == 37
    assert abs(infer_sample_fps(_Scene(), times) - 30.0) < 1e-3


def test_dense_subframe_keys_are_kept():
    times = [i * 0.1 for i in range(50)]  # 240 fps source in a 24 fps scene

    assert merge_sample_times(reversed(times)) == times


def test_keys_just_outside_the_tolerance_are_kept():
    assert merge_sample_times([1.0, 1.0 + 0.005, 1.0 + 0.0001]) == [1.0, 1.005]
