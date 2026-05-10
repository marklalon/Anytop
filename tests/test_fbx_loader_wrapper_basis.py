from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("bpy")


_TESTS_DIR = Path(__file__).resolve().parent
_ANYTOP_ROOT = _TESTS_DIR.parent
_REPO_ROOT = _ANYTOP_ROOT.parent

for _path in (_REPO_ROOT, _ANYTOP_ROOT):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


from motion_lib import BVH  # noqa: E402
from motion_lib import FBX  # noqa: E402
from motion_lib.Animation import positions_global  # noqa: E402
from data_loaders.truebones.offline_reference_dataset import load_cond_dict  # noqa: E402
from data_loaders.truebones.truebones_utils.motion_process import (  # noqa: E402
    get_common_features_from_T_pose,
    TPoseFeatures,
    process_anim,
)


_HORSE_DIR = _ANYTOP_ROOT / "dataset" / "truebones" / "zoo" / "Truebone_Z-OO" / "Horse"
_TPOSE_FBX = _HORSE_DIR / "HorseALL-TPOSE.fbx"
_IDLE_BVH = _HORSE_DIR / "__Idle.bvh"


def _processed_stats(clip_name: str) -> tuple[tuple[float, float, float], float, float]:
    tp: TPoseFeatures = get_common_features_from_T_pose(str(_TPOSE_FBX), "Horse")
    _cond_entry = load_cond_dict().get("Horse")
    if _cond_entry is None or "scale_factor" not in _cond_entry:
        _scale_factor = float(tp.scale_factor)
    else:
        _scale_factor = float(_cond_entry["scale_factor"])

    raw_anim, _raw_names, _frame_time = FBX.load(str(_HORSE_DIR / clip_name))
    processed_anim, _root_xz_center, _scale_factor = process_anim(
        raw_anim,
        "Horse",
        tp.orientation_quat,
        scale_factor=_scale_factor,
    )
    processed_global = positions_global(processed_anim)
    foot_y = processed_global[:, tp.foot_indices, 1]
    frame0 = processed_global[0]
    spans = tuple(float(np.ptp(frame0[:, axis])) for axis in range(3))
    return spans, float(foot_y.min()), float(foot_y[0].min())


def test_fbx_load_matches_source_bvh_position_channels() -> None:
    fbx_path = _HORSE_DIR / "HorseALL-Idle.fbx"
    animation, names, _frame_time = FBX.load(str(fbx_path))
    source_anim, source_names, _source_frame_time = BVH.load(str(_IDLE_BVH))

    common_names = [name for name in names if name in source_names]
    fbx_indices = [names.index(name) for name in common_names]
    source_indices = [source_names.index(name) for name in common_names]
    frame0_error = np.linalg.norm(
        animation.positions[0, fbx_indices] - source_anim.positions[0, source_indices],
        axis=-1,
    )

    assert float(frame0_error.mean()) < 1e-3
    assert float(frame0_error.max()) < 5e-3
    assert np.allclose(animation.positions[0, names.index("C_ctrl")], [0.0, 0.0, 0.0], atol=1e-5)


def test_horse_idle_clips_export_upright_and_near_floor() -> None:
    idle_spans, idle_min_y, idle_frame0_min_y = _processed_stats("HorseALL-Idle.fbx")
    idle_ears_spans, idle_ears_min_y, idle_ears_frame0_min_y = _processed_stats("HorseALL-IdleEars.fbx")

    for spans in (idle_spans, idle_ears_spans):
        x_span, y_span, z_span = spans
        assert y_span > x_span * 2.0, (
            "processed Horse clips should stay upright instead of lying on their side: "
            f"x_span={x_span:.6f}, y_span={y_span:.6f}, z_span={z_span:.6f}"
        )

    for value in (idle_min_y, idle_ears_min_y, idle_frame0_min_y, idle_ears_frame0_min_y):
        assert -0.05 <= value < 0.1, f"processed foot height should stay close to the floor, got {value:.6f}"

    assert abs(idle_min_y - idle_ears_min_y) < 0.15, (
        "similar Horse idle clips should not diverge widely in processed foot height: "
        f"Idle={idle_min_y:.6f}, IdleEars={idle_ears_min_y:.6f}"
    )
    assert abs(idle_frame0_min_y - idle_ears_frame0_min_y) < 0.05, (
        "similar Horse idle clips should start at a comparable floor height: "
        f"Idle={idle_frame0_min_y:.6f}, IdleEars={idle_ears_frame0_min_y:.6f}"
    )