"""No clip may carry a redundant edge frame.

A held edge (0 == 1, N-1 == N) is dead time in any clip, and so is a closing key
that copies frame 0 -- the pose is still there at index 0. In a loop the closing
key is worse than dead: playback wraps N -> 0, so it stalls the motion for one
frame every cycle (the hitch the eye reads as a stutter) and zeroes the terminal
velocity, which a loop writes as the wrap delta ``pos[0] - pos[-1]``.

Only a repeated key is dropped; a frame that moved even a little is authored
motion and stays, because nothing downstream can put it back. Preprocessing drops
those frames and nothing else, so a recovered or round-tripped clip still
re-extracts frame for frame.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion_lib.Animation import Animation
from motion_lib.Quaternions import Quaternions

from data_loaders.truebones.truebones_utils.animation_utils import (
    TRIM_MIN_FRAMES,
    find_redundant_boundary_frames,
)
from data_loaders.truebones.truebones_utils.features import (
    extract_motion_features_from_aligned_anims,
)


def _swing_anim(angles: np.ndarray) -> Animation:
    """A root-spine-tip rig whose spine sweeps ``angles`` and carries the tip.

    The rotation is on the SPINE so the tip actually travels -- a leaf's own
    rotation moves no joint, and a stack of identical global positions is a
    motionless clip, which the trim leaves alone by design. The root stays put,
    so nothing the root XZ machinery does is in play here.
    """
    n_frames = len(angles)
    parents = np.array([-1, 0, 1], dtype=np.int64)
    offsets = np.array(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    rotations = Quaternions(
        np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_frames, len(parents), 1))
    )
    rotations[:, 1] = Quaternions.from_angle_axis(angles, np.array([1.0, 0.0, 0.0]))
    positions = np.tile(offsets, (n_frames, 1, 1))
    return Animation(rotations, positions, Quaternions.id(len(parents)), offsets, parents)


# Amplitude and frame count together keep the per-frame step inside the loop
# detector's own tolerance, so these fixtures are read as the loops they are.
CYCLE_AMPLITUDE_DEG = 10.0


def _cycle_anim(n_frames: int) -> Animation:
    """One full closed swing cycle: frame N is one step short of frame 0."""
    phase = 2.0 * np.pi * np.arange(n_frames) / n_frames
    return _swing_anim(np.radians(CYCLE_AMPLITUDE_DEG) * np.sin(phase))


def _repeat_frame(anim: Animation, source: int, at: int) -> Animation:
    """Insert a copy of frame ``source`` at index ``at`` -- an exporter's duplicate key."""
    order = np.arange(anim.rotations.shape[0])
    order = np.insert(order, at, source)
    return anim[order]


def _extract(anim: Animation, trim: bool = True):
    features, _max_joints, motion_anim, export_anim, is_loop, _flattened = (
        extract_motion_features_from_aligned_anims(
            anim,
            anim,
            object_type='TestSkeleton',
            max_joints=8,
            orientation_quat=Quaternions.id(1).qs[0],
            translation_root_index=0,
            trim_redundant_frames=trim,
        )
    )
    return features, is_loop, motion_anim, export_anim


# ── the detector ──────────────────────────────────────────────────────────

def test_a_clean_cycle_keeps_every_frame():
    anim = _cycle_anim(40)
    assert _extract(anim)[1] is True
    assert find_redundant_boundary_frames(_positions(anim)) == (False, False)


@pytest.mark.parametrize(
    'source, at, expected',
    [
        (0, 0, (True, False)),    # frame 0 held: 0 == 1
        (39, 40, (False, True)),  # frame N held: N-1 == N
        (0, 40, (False, True)),   # duplicated closing key: N == 0
    ],
)
def test_a_duplicated_edge_frame_is_found(source, at, expected):
    anim = _repeat_frame(_cycle_anim(40), source, at)
    assert find_redundant_boundary_frames(_positions(anim)) == expected


def test_both_ends_are_trimmed_at_once():
    anim = _repeat_frame(_repeat_frame(_cycle_anim(40), 0, 0), 40, 41)
    assert find_redundant_boundary_frames(_positions(anim)) == (True, True)


def test_only_one_frame_per_end_so_an_authored_hold_keeps_its_timing():
    anim = _cycle_anim(40)
    for _ in range(4):
        anim = _repeat_frame(anim, 0, 0)
    assert find_redundant_boundary_frames(_positions(anim)) == (True, False)


@pytest.mark.parametrize('fraction_of_a_step', [0.5, 0.05, 0.01])
def test_a_near_copy_is_left_alone(fraction_of_a_step):
    """The line is drawn at a repeated key, not at what looks like a stall.

    Unity's clip splitter interpolates the closing key, so a run cycle can
    arrive with a wrap step a few percent of a stride. That frame still carries
    motion, and this rule would rather leave its hitch alone than delete it.
    """
    phase = 2.0 * np.pi * np.arange(40) / 40
    angles = np.radians(CYCLE_AMPLITUDE_DEG) * np.sin(phase)
    step = float(np.median(np.abs(np.diff(angles))))
    near_copy = np.concatenate([angles, [angles[0] + fraction_of_a_step * step]])

    assert find_redundant_boundary_frames(_positions(_swing_anim(near_copy))) == (False, False)


def test_a_slow_edge_is_not_a_duplicate():
    """Ease-in is real motion, not a repeat: it must survive."""
    anim = _cycle_anim(40)
    assert find_redundant_boundary_frames(_positions(anim)) == (False, False)
    # ...even where the sine is flattest, which is where the cycle turns around.
    rolled = anim[np.roll(np.arange(40), 10)]
    assert find_redundant_boundary_frames(_positions(rolled)) == (False, False)


def test_a_motionless_clip_is_left_alone():
    """Every frame repeats every other; the frame count IS the held duration."""
    anim = _cycle_anim(40)
    anim.rotations[:] = anim.rotations[0:1]
    assert find_redundant_boundary_frames(_positions(anim)) == (False, False)


def test_a_clip_with_no_frames_to_spare_is_left_alone():
    """The floor is the loop math's, not a clip-length policy -- preprocessing
    never gets near it (the shortest shipped clip is 20 frames), but at two
    frames the head pair and the wrap pair are the same pair."""
    # Three frames: the pairs still overlap enough that there is nothing to give.
    assert find_redundant_boundary_frames(
        _positions(_repeat_frame(_cycle_anim(TRIM_MIN_FRAMES), 0, 0))[:3]
    ) == (False, False)
    # Four, with a duplicated closing key: one frame can go, and exactly one.
    assert find_redundant_boundary_frames(
        _positions(_repeat_frame(_cycle_anim(4), 0, 4))[:5]
    ) == (False, True)


# ── the pipeline ──────────────────────────────────────────────────────────

def test_extraction_drops_the_duplicated_closing_key():
    duplicated = _repeat_frame(_cycle_anim(40), 0, 40)
    features, is_loop, motion_anim, export_anim = _extract(duplicated)

    assert is_loop is True
    assert features.shape[0] == 40
    # The exported anims follow the tensor, or the BVH drifts away from it.
    assert motion_anim.rotations.shape[0] == 40
    assert export_anim.rotations.shape[0] == 40
    # ...and the clip that comes out is the one that was authored.
    np.testing.assert_allclose(features, _extract(_cycle_anim(40))[0], atol=1e-9)


def test_the_terminal_velocity_stays_a_real_step():
    """A loop's last velocity row is the wrap delta; a duplicate zeroed it."""
    duplicated = _repeat_frame(_cycle_anim(40), 0, 40)
    trimmed, _is_loop, _anim, _export = _extract(duplicated)
    untrimmed, _is_loop, _anim, _export = _extract(duplicated, trim=False)

    assert np.abs(untrimmed[-1, :, 9:12]).max() == pytest.approx(0.0, abs=1e-12)
    assert np.abs(trimmed[-1, :, 9:12]).max() > 1e-3


def test_a_held_edge_is_trimmed_out_of_a_non_loop_too():
    """A repeated key carries nothing whether or not the clip wraps."""
    # A one-way sweep, so the ends are nowhere near each other, with frame 0 held.
    open_anim = _repeat_frame(
        _swing_anim(np.radians(np.linspace(0.0, 4.0 * CYCLE_AMPLITUDE_DEG, 40))), 0, 0
    )
    features, is_loop, motion_anim, _export = _extract(open_anim)

    assert is_loop is False
    assert features.shape[0] == 40
    assert motion_anim.rotations.shape[0] == 40


def test_an_end_frame_copying_the_start_goes_whether_or_not_it_wraps():
    """The wrap pair is NOT loop-gated: a copy is a copy.

    The clip ends exactly where it began -- a swing out and back. In a loop that
    closing key stalls the wrap; anywhere else it is a frame whose pose the clip
    still holds at index 0, so dropping it costs one frame of duration and no
    motion. Exactness is what makes that true, and it is the whole rule.
    """
    out_and_back = np.concatenate([
        np.radians(np.linspace(0.0, 4.0 * CYCLE_AMPLITUDE_DEG, 20)),
        np.radians(np.linspace(4.0 * CYCLE_AMPLITUDE_DEG, 0.0, 20)),
    ])
    positions = _positions(_swing_anim(out_and_back))
    assert find_redundant_boundary_frames(positions) == (False, True)

    # ...and the frame that survives holds the pose the dropped one repeated.
    np.testing.assert_array_equal(positions[0], positions[-1])


def test_a_duplicated_closing_key_cannot_make_an_open_sweep_a_loop():
    """The verdict describes the tensor that ships, not the frame removed from it.

    This clip closes ONLY because its last frame copies the first: underneath, it
    is an open sweep that ends nowhere near where it began. Reading closure off
    the untrimmed stack reads the exporter's habit of duplicating the closing key
    rather than the motion, and reads it off a frame this pass then deletes --
    so the verdict is taken after the trim, on the 40 frames that remain.
    """
    open_anim = _swing_anim(np.radians(np.linspace(0.0, 4.0 * CYCLE_AMPLITUDE_DEG, 40)))
    closed_by_copy = _repeat_frame(open_anim, 0, 40)

    features, is_loop, _anim, _export = _extract(closed_by_copy)

    assert is_loop is False
    assert features.shape[0] == 40


def test_a_real_cycle_survives_losing_its_duplicated_closing_key():
    """The other half of the rule: the trim must not cost a genuine loop.

    A cycle that ships the redundant closing key most loop takes arrive with
    still closes once that key is gone, because the frame before it already
    steps cleanly into frame 0.
    """
    duplicated = _repeat_frame(_cycle_anim(40), 0, 40)

    features, is_loop, _anim, _export = _extract(duplicated)

    assert is_loop is True
    assert features.shape[0] == 40


def test_trimming_is_opt_in_so_recovery_reproduces_a_stored_clip():
    duplicated = _repeat_frame(_cycle_anim(40), 0, 40)
    assert _extract(duplicated, trim=False)[0].shape[0] == 41


def _positions(anim: Animation) -> np.ndarray:
    from motion_lib.Animation import positions_global

    return positions_global(anim)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
