"""
Native-space GLB -> GLB retarget test.

Guards the property the native path exists for: a self-retarget (same file as
source and target) must round-trip, root translation included. The feature-space
path cannot -- ``get_motion`` strips the root XZ trajectory once a clip travels,
and hands it back on a channel the feature retarget never receives -- so this
test asserts both that the pose survives and that the XZ trajectory does.

Usage:
    python tests/test_retarget_glb.py
    python tests/test_retarget_glb.py --glb <path/to/animated.glb>
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np
import pytest


_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ANYTOP_ROOT = os.path.dirname(_TESTS_DIR)
_REPO_ROOT = os.path.dirname(_ANYTOP_ROOT)

for _path in [_REPO_ROOT, _ANYTOP_ROOT]:
    if _path not in sys.path:
        sys.path.insert(0, _path)


# A clip with real locomotion: its root travels ~7.6 units in Z, far past
# ROOT_XZ_STRIP_THRESHOLD, so the feature path would strip it entirely.
_DEFAULT_GLB = os.path.join(
    _ANYTOP_ROOT, "dataset", "truebones", "zoo", "Truebone_Z-OO",
    "Buffalo", "Buffalo-WalkLoop.glb",
)

# Float32 GLB channels plus an inverse-FK round trip: the observed self-retarget
# error is ~0.0014% of character size and ~0.004 deg, so these leave an order of
# magnitude of headroom while still catching any real regression.
_POS_TOLERANCE_PCT_CHAR = 0.05
_ROT_TOLERANCE_DEG = 0.05
_ROOT_XZ_TOLERANCE = 1e-4


def _root_world_xz_extent(glb_path: str) -> np.ndarray:
    """Return the (2,) peak-to-peak world XZ extent of the hierarchy root."""
    from motion_lib import FBX

    anim, _names, _frametime = FBX.load(glb_path, collapse_root=False)
    # The hierarchy root has no parent, so its local position is already its
    # world position -- no FK needed, and no rest-rotation convention to get
    # wrong.
    root_positions = np.asarray(anim.positions[:, 0], dtype=np.float64)
    return np.ptp(root_positions[:, [0, 2]], axis=0)


def _run_self_retarget(source_glb: str, output_dir: str) -> str:
    from utils.auto_retarget import retarget_glb_to_glb

    output_glb = os.path.join(output_dir, "self_retarget.glb")
    return retarget_glb_to_glb(
        source_glb, source_glb, output_glb, verbose=False,
    )


def _assert_matches_source(source_glb: str, exported_glb: str) -> None:
    from tools.compare_motions import (
        load_motion,
        _validate_compatible,
        detect_and_align,
        compare_motions,
        print_summary,
    )

    motion_a = load_motion(source_glb)
    motion_b = load_motion(exported_glb)
    _validate_compatible(motion_a, motion_b)

    motion_b_aligned, alignment = detect_and_align(motion_a, motion_b)
    result = compare_motions(motion_a, motion_b_aligned, alignment)
    print_summary(motion_a, motion_b, alignment, result)

    errors: list[str] = []
    if alignment.rotation_label != "identity":
        errors.append(
            f"self-retarget needed a rigid coordinate alignment "
            f"({alignment.rotation_label}); it must be identity"
        )
    position_pct = float(result["position"]["max_error_pct_char"])
    if position_pct > _POS_TOLERANCE_PCT_CHAR:
        errors.append(
            f"position error {position_pct:.4f}% exceeds {_POS_TOLERANCE_PCT_CHAR}%"
        )
    rotation_deg = float(result["rotation"]["max_error_deg"])
    if rotation_deg > _ROT_TOLERANCE_DEG:
        errors.append(
            f"rotation error {rotation_deg:.4f} deg exceeds {_ROT_TOLERANCE_DEG} deg"
        )
    assert not errors, "; ".join(errors)


def _assert_root_trajectory_preserved(source_glb: str, exported_glb: str) -> None:
    source_extent = _root_world_xz_extent(source_glb)
    exported_extent = _root_world_xz_extent(exported_glb)

    assert float(np.max(source_extent)) > 1.0, (
        f"test fixture {os.path.basename(source_glb)} has no root XZ travel "
        f"(extent {source_extent}), so it cannot detect XZ stripping"
    )
    delta = np.abs(exported_extent - source_extent)
    assert float(np.max(delta)) <= _ROOT_XZ_TOLERANCE, (
        f"root XZ trajectory changed: source extent {source_extent} vs "
        f"exported {exported_extent}"
    )


def run_test_retarget_glb_self_roundtrip(source_glb: str, output_dir: str | None = None) -> None:
    if not os.path.isfile(source_glb):
        message = f"source GLB not found: {source_glb}"
        if "PYTEST_CURRENT_TEST" in os.environ:
            pytest.skip(message)
        raise AssertionError(message)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        exported_glb = _run_self_retarget(source_glb, output_dir)
        _assert_matches_source(source_glb, exported_glb)
        _assert_root_trajectory_preserved(source_glb, exported_glb)
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        exported_glb = _run_self_retarget(source_glb, temp_dir)
        _assert_matches_source(source_glb, exported_glb)
        _assert_root_trajectory_preserved(source_glb, exported_glb)


def test_retarget_glb_self_roundtrip() -> None:
    pytest.importorskip("bpy", reason="native GLB retarget requires bpy")
    run_test_retarget_glb_self_roundtrip(_DEFAULT_GLB)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glb", default=_DEFAULT_GLB, help="Animated source GLB.")
    parser.add_argument(
        "--output-dir", default=None,
        help="Keep the exported GLB here instead of a temporary directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    _args = parse_args()
    run_test_retarget_glb_self_roundtrip(_args.glb, _args.output_dir)
    print("PASS: native self-retarget round-trips, root XZ trajectory preserved")
