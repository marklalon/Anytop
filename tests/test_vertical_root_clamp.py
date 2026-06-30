import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = type("Tensor", (), {})
    sys.modules["torch"] = torch_stub

from motion_lib.Animation import Animation, positions_global
from motion_lib.Quaternions import Quaternions

from data_loaders.truebones.truebones_utils.animation_utils import clamp_vertical_trajectory
from data_loaders.truebones.truebones_utils.param_utils import ROOT_Y_MIN_HEIGHT


def _animated_root_y(values):
    frames = len(values)
    parents = np.array([-1, 0], dtype=np.int32)
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    positions = np.repeat(offsets[None, :, :], frames, axis=0)
    positions[:, 0, 1] = np.asarray(values, dtype=np.float64)
    return Animation(
        Quaternions.id((frames, len(parents))),
        positions,
        Quaternions.id(len(parents)),
        offsets,
        parents,
    )


def _animated_root_y_with_body_length(values, body_length):
    frames = len(values)
    parents = np.array([-1, 0], dtype=np.int32)
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, float(body_length), 0.0],
        ],
        dtype=np.float64,
    )
    positions = np.repeat(offsets[None, :, :], frames, axis=0)
    positions[:, 0, 1] = np.asarray(values, dtype=np.float64)
    return Animation(
        Quaternions.id((frames, len(parents))),
        positions,
        Quaternions.id(len(parents)),
        offsets,
        parents,
    )


def _animated_descendant_root_y(values):
    frames = len(values)
    parents = np.array([-1, 0, 1], dtype=np.int32)
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    positions = np.repeat(offsets[None, :, :], frames, axis=0)
    positions[:, 1, 1] = np.asarray(values, dtype=np.float64)
    return Animation(
        Quaternions.id((frames, len(parents))),
        positions,
        Quaternions.id(len(parents)),
        offsets,
        parents,
    )


def test_non_aquatic_root_y_is_floored_to_min_height():
    anim = _animated_root_y([0.2, -0.25, -0.75, -1.2])

    clamped = clamp_vertical_trajectory(anim, "Pteranodon")
    root_y = positions_global(clamped)[:, 0, 1]

    assert root_y.min() == pytest.approx(ROOT_Y_MIN_HEIGHT)
    np.testing.assert_allclose(root_y[:2], [0.2, -0.25], atol=1e-8)
    np.testing.assert_allclose(root_y[2:], [ROOT_Y_MIN_HEIGHT, ROOT_Y_MIN_HEIGHT], atol=1e-8)


def test_non_aquatic_descendant_translation_root_y_is_floored_to_min_height():
    anim = _animated_descendant_root_y([0.2, -0.25, -0.75, -1.2])

    clamped = clamp_vertical_trajectory(anim, "Pteranodon")
    translation_root_y = positions_global(clamped)[:, 1, 1]

    assert translation_root_y.min() == pytest.approx(ROOT_Y_MIN_HEIGHT)
    np.testing.assert_allclose(translation_root_y[:2], [0.2, -0.25], atol=1e-8)
    np.testing.assert_allclose(
        translation_root_y[2:],
        [ROOT_Y_MIN_HEIGHT, ROOT_Y_MIN_HEIGHT],
        atol=1e-8,
    )


def test_aquatic_root_y_is_also_floored_to_min_height():
    anim = _animated_root_y([0.2, -0.25, -0.75])

    clamped = clamp_vertical_trajectory(anim, "Pirrana")
    root_y = positions_global(clamped)[:, 0, 1]

    assert root_y.min() == pytest.approx(ROOT_Y_MIN_HEIGHT)


def test_aquatic_vertical_ratios_apply_as_negative_swim_depth_limit():
    anim = _animated_root_y_with_body_length([0.0, -0.1, -0.25, -0.4], body_length=0.4)

    clamped = clamp_vertical_trajectory(anim, "Pirrana")
    root_y = positions_global(clamped)[:, 0, 1]

    assert root_y.min() == pytest.approx(-0.2)
    np.testing.assert_allclose(root_y[:2], [0.0, -0.1], atol=1e-8)


def test_aquatic_vertical_ratios_still_apply_positive_jump_height_limit():
    anim = _animated_root_y_with_body_length([0.0, 0.1, 0.25, 0.4], body_length=0.4)

    clamped = clamp_vertical_trajectory(anim, "Pirrana")
    root_y = positions_global(clamped)[:, 0, 1]

    assert root_y.max() == pytest.approx(0.2)
    np.testing.assert_allclose(root_y[:2], [0.0, 0.1], atol=1e-8)
