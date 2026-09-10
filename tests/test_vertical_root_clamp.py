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
from data_loaders.truebones.truebones_utils.param_utils import (
    ROOT_Y_MIN_HEIGHT,
    ROOT_Y_SOFT_CLAMP_KNEE,
    VERTICAL_CLAMP_MAX_RATIO,
    VERTICAL_CLAMP_MIN_RATIO,
)


def _expected_band_scale(extent, min_h, max_h):
    """The one factor an excursion reaching ``extent`` is scaled by.

    The band's hyperbola with the excess divided back out, written independently:
    ``(g(extent) - min_h) / (extent - min_h)`` reduces to ``w / (e + w)``.
    """
    width = max_h - min_h
    return width / ((extent - min_h) + width)


def _expected_soft_floor(values):
    """The bound the root-Y descent is held to, written out independently.

    ``ROOT_Y_MIN_HEIGHT`` is an asymptote, not a reachable height: above the knee
    this is the identity, below it the excess depth is compressed by the same
    hyperbola the root-XZ extent uses.
    """
    knee = abs(ROOT_Y_SOFT_CLAMP_KNEE)
    limit = abs(ROOT_Y_MIN_HEIGHT)
    width = limit - knee
    depth = -np.asarray(values, dtype=np.float64)
    compressed = limit - width * width / (depth - knee + width)
    return np.where(depth > knee, -compressed, np.asarray(values, dtype=np.float64))


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

    # The frozen joint rides half a bone above the wrapper, so it is ITS curve
    # -- [0.7, 0.25, -0.25, -0.7] -- that reaches past the knee.
    assert species_root_y.min() == pytest.approx(_expected_soft_floor(-0.7))
    np.testing.assert_allclose(species_root_y[:2], [0.7, 0.25], atol=1e-8)


def test_left_to_itself_the_clamp_floors_whatever_the_clip_animates():
    """The fallback, and why it is not good enough inside preprocessing."""
    anim = _wrapper_animated_chain([0.2, -0.25, -0.75, -1.2], lift=0.5)

    clamped = clamp_vertical_trajectory(anim, "Pteranodon")
    global_pos = positions_global(clamped)

    assert global_pos[:, 0, 1].min() == pytest.approx(_expected_soft_floor(-1.2))
    # Joint 1 -- what the species is actually rooted on -- lands half a bone
    # above the bound, because a different curve was the one clamped.
    assert global_pos[:, 1, 1].min() == pytest.approx(_expected_soft_floor(-1.2) + 0.5)


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

    assert positions_global(frozen)[:, 1, 1].min() == pytest.approx(_expected_soft_floor(-0.7))
    assert positions_global(detected)[:, 1, 1].min() == pytest.approx(_expected_soft_floor(-1.2) + 0.5)


def test_non_aquatic_root_y_descent_is_compressed_not_floored():
    anim = _animated_root_y([0.2, -0.25, -0.75, -1.2])

    clamped = clamp_vertical_trajectory(anim, "Pteranodon")
    root_y = positions_global(clamped)[:, 0, 1]

    # Above the knee, bit-for-bit; below it, two frames that a hard floor would
    # have collapsed onto one height stay apart and stay ordered.
    np.testing.assert_allclose(root_y[:2], [0.2, -0.25], atol=1e-8)
    np.testing.assert_allclose(root_y[2:], _expected_soft_floor([-0.75, -1.2]), atol=1e-12)
    assert root_y[3] < root_y[2]
    assert root_y.min() > ROOT_Y_MIN_HEIGHT


def test_non_aquatic_descendant_translation_root_y_is_bounded_the_same_way():
    anim = _animated_descendant_root_y([0.2, -0.25, -0.75, -1.2])

    clamped = clamp_vertical_trajectory(anim, "Pteranodon")
    translation_root_y = positions_global(clamped)[:, 1, 1]

    assert translation_root_y.min() > ROOT_Y_MIN_HEIGHT
    np.testing.assert_allclose(translation_root_y[:2], [0.2, -0.25], atol=1e-8)
    np.testing.assert_allclose(
        translation_root_y[2:],
        _expected_soft_floor([-0.75, -1.2]),
        atol=1e-12,
    )


def test_aquatic_root_y_is_also_held_to_the_same_bound():
    anim = _animated_root_y([0.2, -0.25, -0.75])

    clamped = clamp_vertical_trajectory(anim, "Pirrana")
    root_y = positions_global(clamped)[:, 0, 1]

    assert ROOT_Y_MIN_HEIGHT < root_y.min() < ROOT_Y_SOFT_CLAMP_KNEE


def test_aquatic_vertical_ratios_apply_as_negative_swim_depth_limit():
    anim = _animated_root_y_with_body_length([0.0, -0.1, -0.25, -0.4], body_length=0.4)

    clamped = clamp_vertical_trajectory(anim, "Pirrana")
    root_y = positions_global(clamped)[:, 0, 1]

    # min_h = 0.12, max_h = 0.2. The deepest frame no longer lands ON the band
    # edge -- it approaches it -- and the frame above the knee is left alone.
    min_h, max_h = 0.4 * VERTICAL_CLAMP_MIN_RATIO, 0.4 * VERTICAL_CLAMP_MAX_RATIO
    scale = _expected_band_scale(0.4, min_h, max_h)
    assert root_y.min() == pytest.approx(-(min_h + (0.4 - min_h) * scale))
    assert -max_h < root_y.min() < -min_h
    np.testing.assert_allclose(root_y[:2], [0.0, -0.1], atol=1e-8)


def test_aquatic_vertical_ratios_still_apply_positive_jump_height_limit():
    anim = _animated_root_y_with_body_length([0.0, 0.1, 0.25, 0.4], body_length=0.4)

    clamped = clamp_vertical_trajectory(anim, "Pirrana")
    root_y = positions_global(clamped)[:, 0, 1]

    min_h, max_h = 0.4 * VERTICAL_CLAMP_MIN_RATIO, 0.4 * VERTICAL_CLAMP_MAX_RATIO
    scale = _expected_band_scale(0.4, min_h, max_h)
    assert root_y.max() == pytest.approx(min_h + (0.4 - min_h) * scale)
    assert min_h < root_y.max() < max_h
    np.testing.assert_allclose(root_y[:2], [0.0, 0.1], atol=1e-8)


def test_a_clip_spent_entirely_below_the_bound_keeps_its_vertical_motion():
    """The regression the soft clamp exists for.

    ``np.maximum(y, -0.5)`` returned a constant for any clip that never came back
    above the floor, which is what a fish at depth, a burrow or a long fall all
    look like: Pirrana_MidSwim shipped with all 97 of its frames pinned at exactly
    -0.5, its entire vertical channel dead. 86 clips carried a plateau like that.
    """
    swim = -0.8 + 0.15 * np.sin(np.linspace(0.0, 4.0 * np.pi, 48))
    anim = _animated_root_y(swim)

    clamped = clamp_vertical_trajectory(anim, "Pteranodon")
    root_y = positions_global(clamped)[:, 0, 1]

    assert np.ptp(root_y) > 0.01
    assert len(np.unique(np.round(root_y, 9))) > 40
    # Ordering survives the compression, so the peaks and troughs of the swim are
    # still in the same places and still the same way round.
    assert np.argmin(root_y) == np.argmin(swim)
    assert np.argmax(root_y) == np.argmax(swim)
    order = np.argsort(swim, kind="stable")
    assert np.all(np.diff(root_y[order]) >= 0.0)


def test_the_bound_is_an_asymptote_and_the_knee_is_seamless():
    depths = np.array([-0.3, -0.5, -1.0, -5.0, -500.0, -1e6])
    anim = _animated_root_y(depths)

    clamped = clamp_vertical_trajectory(anim, "Pteranodon")
    root_y = positions_global(clamped)[:, 0, 1]

    assert np.all(root_y > ROOT_Y_MIN_HEIGHT)
    assert np.all(np.diff(root_y) < 0.0)
    # Value and slope are continuous at the knee: a frame a hair below it moves by
    # a hair, so nothing in the dataset carries a velocity step at -0.3 for the
    # model to read as a feature.
    eps = 1e-7
    just_under = clamp_vertical_trajectory(
        _animated_root_y([ROOT_Y_SOFT_CLAMP_KNEE - eps, 0.0]), "Pteranodon"
    )
    assert positions_global(just_under)[0, 0, 1] == pytest.approx(
        ROOT_Y_SOFT_CLAMP_KNEE - eps, abs=1e-12
    )


def test_a_clip_that_stays_above_the_knee_is_untouched():
    anim = _animated_root_y([0.4, 0.1, -0.2, -0.3, 0.0])

    clamped = clamp_vertical_trajectory(anim, "Pteranodon")

    assert clamped is anim


def test_the_aquatic_band_survives_the_bound_instead_of_being_sheared_flat():
    """The two used to contradict each other.

    The aquatic branch compresses swim depth into [-maxH, -minH], which for a
    body span near the HML reference reaches past -0.5; the hard floor then cut
    the deeper half of that band off onto one height. The soft clamp bounds the
    band's own output instead, so the dive keeps its shape.
    """
    depths = [0.0, -0.3, -0.6, -0.9, -1.2]
    anim = _animated_root_y_with_body_length(depths, body_length=1.389)

    clamped = clamp_vertical_trajectory(anim, "Pirrana")
    root_y = positions_global(clamped)[:, 0, 1]

    assert np.all(np.diff(root_y) < 0.0)
    assert root_y.min() > ROOT_Y_MIN_HEIGHT
    assert len(np.unique(np.round(root_y[2:], 9))) == 3


def test_two_climbs_of_different_size_no_longer_report_the_same_height():
    """What the hard band target cost: every clip that reached it read ``max_h``.

    304 of the 582 shipped winged clips peaked at exactly 0.5 body lengths, so a
    hop and a full climb were the same number in the features. The asymptote keeps
    them apart and keeps them in order.
    """
    min_h, max_h = 0.4 * VERTICAL_CLAMP_MIN_RATIO, 0.4 * VERTICAL_CLAMP_MAX_RATIO
    peaks = [0.25, 0.4, 1.2, 6.0]
    tops = [
        positions_global(
            clamp_vertical_trajectory(
                _animated_root_y_with_body_length([0.0, peak], body_length=0.4),
                "Pteranodon",
            )
        )[:, 0, 1].max()
        for peak in peaks
    ]

    assert all(np.diff(tops) > 0.0)
    assert all(min_h < top < max_h for top in tops)


def test_the_band_scales_the_excursion_by_one_factor():
    """A climb keeps its shape; only its size changes.

    The reason this is ``scale_root_xz_extent`` and not ``soft_clamp_root_xz``: a
    per-frame map would compress the top of the arc harder than its base.
    """
    min_h = 0.4 * VERTICAL_CLAMP_MIN_RATIO
    heights = np.array([0.0, min_h, 0.3, 0.6, 1.5])
    anim = _animated_root_y_with_body_length(heights, body_length=0.4)

    root_y = positions_global(clamp_vertical_trajectory(anim, "Pteranodon"))[:, 0, 1]

    np.testing.assert_allclose(root_y[:2], heights[:2], atol=1e-12)
    factors = (root_y[2:] - min_h) / (heights[2:] - min_h)
    np.testing.assert_allclose(factors, factors[0], rtol=1e-12)


def test_a_clip_that_never_clears_the_knee_is_untouched():
    min_h = 0.4 * VERTICAL_CLAMP_MIN_RATIO
    anim = _animated_root_y_with_body_length([0.0, 0.05, min_h], body_length=0.4)

    assert clamp_vertical_trajectory(anim, "Pteranodon") is anim


def test_clearing_the_knee_by_a_hair_barely_moves_the_clip():
    """Value and slope are continuous at the knee, so there is no step at 0.3L."""
    min_h = 0.4 * VERTICAL_CLAMP_MIN_RATIO
    anim = _animated_root_y_with_body_length([0.0, min_h + 1e-9], body_length=0.4)

    root_y = positions_global(clamp_vertical_trajectory(anim, "Pteranodon"))[:, 0, 1]

    assert root_y.max() == pytest.approx(min_h + 1e-9, abs=1e-14)

