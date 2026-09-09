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

from data_loaders.truebones.truebones_utils.animation_utils import (
    clamp_vertical_trajectory,
    find_translation_root,
)
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


def _wrapper_animated_chain(values, lift=0.5):
    """``Wrapper(0) -> Root(1) -> Tip(2)``, with the WRAPPER carrying the motion.

    The joint a species may be frozen on sits ``lift`` above it, so reading the
    height off one instead of the other shifts the whole curve by a bone.
    """
    frames = len(values)
    parents = np.array([-1, 0, 1], dtype=np.int32)
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, float(lift), 0.0],
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


def test_the_vertical_clamp_floors_the_species_root_it_is_given():
    """Height and trajectory have to be read off the SAME joint.

    A species whose clips author transport on different joints is frozen on one
    of them (MB_TigerDrago on Pelvis, Bear on Pelvis), while per-clip detection
    still answers with whichever joint that clip happens to animate. The two are
    a bone apart, so the floor lands in a different place.
    """
    anim = _wrapper_animated_chain([0.2, -0.25, -0.75, -1.2], lift=0.5)
    assert find_translation_root(anim) == 0

    clamped = clamp_vertical_trajectory(anim, "Pteranodon", translation_root_index=1)
    species_root_y = positions_global(clamped)[:, 1, 1]

    assert species_root_y.min() == pytest.approx(ROOT_Y_MIN_HEIGHT)
    np.testing.assert_allclose(species_root_y[:2], [0.7, 0.25], atol=1e-8)


def test_left_to_itself_the_clamp_floors_whatever_the_clip_animates():
    """The fallback, and why it is not good enough inside preprocessing."""
    anim = _wrapper_animated_chain([0.2, -0.25, -0.75, -1.2], lift=0.5)

    clamped = clamp_vertical_trajectory(anim, "Pteranodon")
    global_pos = positions_global(clamped)

    assert global_pos[:, 0, 1].min() == pytest.approx(ROOT_Y_MIN_HEIGHT)
    # Joint 1 -- what the species is actually rooted on -- stops half a bone
    # short of the floor, because a different curve was the one clamped.
    assert global_pos[:, 1, 1].min() == pytest.approx(ROOT_Y_MIN_HEIGHT + 0.5)


def test_process_anim_hands_the_frozen_root_to_the_vertical_clamp():
    """``get_hml_aligned_anim`` -> ``process_anim`` is the only path preprocessing
    has to say which joint a species is rooted on."""
    from data_loaders.truebones.truebones_utils.features import process_anim

    anim = _wrapper_animated_chain([0.2, -0.25, -0.75, -1.2], lift=0.5)

    frozen, _center, _scale = process_anim(
        anim,
        "Pteranodon",
        Quaternions.id(1),
        scale_factor=1.0,
        translation_root_index=1,
    )
    detected, _center2, _scale2 = process_anim(
        anim,
        "Pteranodon",
        Quaternions.id(1),
        scale_factor=1.0,
    )

    assert positions_global(frozen)[:, 1, 1].min() == pytest.approx(ROOT_Y_MIN_HEIGHT)
    assert positions_global(detected)[:, 1, 1].min() == pytest.approx(ROOT_Y_MIN_HEIGHT + 0.5)


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
