"""A death's direction is where the BODY goes, on all four planar axes.

Routed to the SIDE measurement -- which is where every non-locomotion row went
before -- a death could only ever come out ``left`` or ``right``, and 30 of the
41 hand-labelled ones say ``forward`` or ``backward``.  So the topple
measurement reads the horizontal travel of the joint CENTROID and names its
dominant axis with one word.

The centroid and not the root is the point of it: a quadruped that keels over
where it stands moves its root by nothing.

The other half of the rule is when a death has NO direction, which is not a
question of how far it travelled: MLS_BattleOwl_Die drifts 2.21 body lengths
out of a hover -- more than the median hand-labelled death -- and topples
toward nothing.  A death only has a direction when the body toppled ONTO THE
GROUND, so a death in the air, a death that never descends and a death in
place are all ``keep``.
"""

import numpy as np

from tools.prefill_direction_words import (
    TOPPLE_MIN_DROP,
    TOPPLE_MIN_TRAVEL,
    TOPPLE_START_CLEARANCE_MAX,
    dominant_word,
    topple_measure,
    topple_verdict,
)


class _Args:
    topple_min_travel = TOPPLE_MIN_TRAVEL
    topple_min_drop = TOPPLE_MIN_DROP
    topple_start_clearance_max = TOPPLE_START_CLEARANCE_MAX


def _row(label="die"):
    return {"clip": "Wolf_Die.npy", "species": "zoo/Wolf", "group": "transition",
            "label": label, "gif_path": "", "motion_path": ""}


class _Clip:
    """A decoded clip whose joints move where *travel* says, over 30 frames.

    The body starts one unit up and ends on the floor, so it topples unless a
    test says otherwise: *drop* replaces that descent, *start_clearance* lifts
    the whole body off the floor it eventually reaches.
    """

    L = 1.0
    root_index = 0

    def __init__(self, travel, root_travel=None, drop=1.0, start_clearance=0.0):
        frames = 30
        self.frames = frames
        self.world = np.zeros((frames, 3, 3))
        ramp = np.linspace(0.0, 1.0, frames)
        # Every joint carries the body travel ...
        self.world[:, :, 0] = (ramp * travel[0])[:, None]
        self.world[:, :, 2] = (ramp * travel[1])[:, None]
        # ... descends by *drop* ...
        self.world[:, :, 1] = (drop * (1.0 - ramp))[:, None]
        # ... and the lowest joint it ever reaches sits *start_clearance* below
        # where the body starts, which is what "off the ground" is measured on.
        self.world[-1, 0, 1] = -start_clearance
        # ... and the root may then be pinned where it started.
        if root_travel is not None:
            self.world[:, self.root_index, 0] = ramp * root_travel[0]
            self.world[:, self.root_index, 2] = ramp * root_travel[1]


def test_the_four_axes_read_off_the_centroid():
    # +Z is the character's forward, +X its left (PLANAR_AXES).
    for travel, word in (((0.0, 2.0), "forward"), ((0.0, -2.0), "backward"),
                         ((2.0, 0.0), "left"), ((-2.0, 0.0), "right")):
        status, words, _reason = topple_verdict(topple_measure(_Clip(travel)), _row(), _Args())
        assert (status, words) == ("write", [word]), travel


def test_a_death_is_spelled_with_one_word_even_when_it_is_diagonal():
    """A person picked the dominant axis at an off-axis share of 0.81."""
    measure = topple_measure(_Clip((0.8, -1.0)))
    assert round(measure["tie_share"], 2) == 0.80
    assert topple_verdict(measure, _row(), _Args())[:2] == ("write", ["backward"])


def test_a_near_perfect_diagonal_is_listed_rather_than_guessed():
    status, words, reason = topple_verdict(topple_measure(_Clip((1.0, -0.99))), _row(), _Args())
    assert (status, words) == ("review", []) and "no dominant axis" in reason


def test_a_collapse_in_place_keeps_its_empty_slot():
    """Correctly empty, not a proposal: nothing about it points anywhere."""
    status, words, reason = topple_verdict(topple_measure(_Clip((0.05, 0.05))), _row(), _Args())
    assert (status, words) == ("keep", []) and "collapse in place" in reason


def test_the_root_may_stay_put_while_the_body_falls():
    """BrownBear_Twitching travels 0.00 at the root and 0.60 at the centroid."""
    clip = _Clip((-1.5, 0.0), root_travel=(0.0, 0.0))
    assert clip.world[-1, clip.root_index, 0] == 0.0
    assert topple_verdict(topple_measure(clip), _row(), _Args())[:2] == ("write", ["right"])


def test_the_travel_is_measured_in_body_lengths():
    """A rig twice the size has to fall twice as far for the same verdict."""
    clip = _Clip((0.0, 0.5))
    clip.L = 2.0
    assert topple_verdict(topple_measure(clip), _row(), _Args())[0] == "keep"
    clip.L = 1.0
    assert topple_verdict(topple_measure(clip), _row(), _Args())[:2] == ("write", ["forward"])


def test_dominant_word_breaks_an_exact_tie_without_raising():
    assert dominant_word(np.array([1.0, 1.0])) in {"left", "forward"}


# ── the four ways a death has no direction ───────────────────────────────────

def test_a_death_in_the_air_keeps_its_empty_slot_however_far_it_drifts():
    """MLS_BattleOwl_Die drifts 2.21 body lengths out of a hover.

    That is more than the median hand-labelled death travels, so no threshold
    on the travel can catch it -- only the label saying it died in the air.
    """
    measure = topple_measure(_Clip((0.0, -2.21)))
    assert topple_verdict(measure, _row("die"), _Args())[0] == "write"
    for label in ("die, hover", "die, fall", "die, fly, backward"):
        status, words, reason = topple_verdict(measure, _row(label), _Args())
        assert (status, words) == ("keep", []), label
        assert "in the air" in reason


def test_a_body_that_starts_off_the_ground_fell_rather_than_toppled():
    clip = _Clip((0.0, -2.0), start_clearance=TOPPLE_START_CLEARANCE_MAX + 0.05)
    status, words, reason = topple_verdict(topple_measure(clip), _row(), _Args())
    assert (status, words) == ("keep", []) and "off the ground" in reason
    # ... and one that starts on it still topples.
    on_ground = _Clip((0.0, -2.0), start_clearance=0.0)
    assert topple_verdict(topple_measure(on_ground), _row(), _Args())[:2] == (
        "write", ["backward"])


def test_a_body_that_never_descends_did_not_topple():
    """Cobra_Death travels 0.80 body lengths and drops 0.21: it slid."""
    clip = _Clip((0.80, 0.0), drop=0.21)
    status, words, reason = topple_verdict(topple_measure(clip), _row(), _Args())
    assert (status, words) == ("keep", []) and "nothing toppled" in reason
