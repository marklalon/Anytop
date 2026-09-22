"""An unset ``--num_frames``: the clip-length prior baked into cond.npy.

A requested length M reaches the model as ``resample_speed_cond = M / n``, the
same number a training clip of M source frames was given, so the auto default
answers with the median length the corpus holds for the requested action label.
These tests pin the contract that default rests on: what the bake records, the
order the lookup widens in, and the priorities generate.py applies around it
(a --reference_motion outranks the label; no label at all keeps the
checkpoint's native window).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.clip_length_prior import (
    COND_KEY,
    LOOP_BUCKET,
    ONESHOT_BUCKET,
    PRIOR_SCHEMA_VERSION,
    auto_num_frames,
    build_clip_length_prior,
    has_clip_length_prior,
    label_key,
    merge_prior_pool,
    source_clip_length,
)


def _entry(records, **extra):
    entry = {COND_KEY: build_clip_length_prior(records)}
    entry.update(extra)
    return entry


def _ask(cond, target, label, *, group="locomotion", loop=False,
         min_frames=20, max_frames=120):
    return auto_num_frames(
        cond,
        target,
        action_group=group,
        action_label=label,
        loop=loop,
        min_frames=min_frames,
        max_frames=max_frames,
    )


# -- Baking ------------------------------------------------------------------
def test_build_splits_loop_and_oneshot_per_label():
    table = build_clip_length_prior([
        ("locomotion", "walk, forward", True, 41),
        ("locomotion", "walk, forward", False, 130),
        ("locomotion", "walk, forward", True, 45),
        ("stationary", "idle", True, 121),
    ])
    assert table["schema"] == PRIOR_SCHEMA_VERSION
    walk = table["by_label"][label_key("locomotion", "walk, forward")]
    assert walk[LOOP_BUCKET] == (41, 45)
    assert walk[ONESHOT_BUCKET] == (130,)
    # The group is part of the key: the same head word in another group is a
    # different motion with a different duration.
    assert label_key("stationary", "idle") in table["by_label"]


def test_build_drops_nonpositive_lengths():
    table = build_clip_length_prior([("locomotion", "walk", True, 0)])
    assert table["by_label"] == {}


def test_source_clip_length_drops_a_loops_closing_key():
    """A loop's recorded length is its PERIOD, so --loop asks for one cycle."""
    motion = np.zeros((10, 3, 12), dtype=np.float32)

    def drop_closing(array):
        return array[:-1]

    assert source_clip_length(motion, True, drop_closing) == 9
    # A one-shot clip has no closing key to drop, and the rule is never run.
    assert source_clip_length(motion, False, drop_closing) == 10


# -- Lookup ------------------------------------------------------------------
def test_own_clips_win_over_neighbours():
    """The target's own clip IS the length the model fitted for that species."""
    cond = {
        "ns/Horse": _entry([("locomotion", "walk, forward", True, 84)]),
        "ns/Cat": _entry([("locomotion", "walk, forward", True, 30)] * 6),
    }
    frames, why = _ask(cond, "ns/Horse", "walk, forward", loop=True)
    assert frames == 84
    assert "ns/Horse" in why


def test_head_words_are_the_first_fallback():
    """'walk, left' with no such clip falls back to the species' walks."""
    cond = {"ns/Horse": _entry([
        ("locomotion", "walk, forward", True, 84),
        ("locomotion", "run, forward", True, 30),
    ])}
    frames, why = _ask(cond, "ns/Horse", "walk, left", loop=True)
    assert frames == 84
    assert "walk" in why


def test_head_word_fallback_ignores_action_modifiers():
    """A modifier must not hide clips that have the requested head set."""
    cond = {"ns/Archer": _entry([
        ("transition", "draw, bow", False, 42),
        ("transition", "draw, gun", False, 46),
    ])}
    frames, why = _ask(
        cond, "ns/Archer", "draw", group="transition", loop=False
    )
    assert frames == 44
    assert "draw" in why


def test_own_head_word_clips_beat_a_neighbours_exact_label():
    """Duration is set more by the body than by the modifier, so every matcher is
    tried on the target species before any neighbour is consulted."""
    cond = {
        "ns/Horse": _entry([("locomotion", "walk, left", True, 84)]),
        "ns/Pigeon": _entry([("locomotion", "walk, forward", True, 20)] * 6),
    }
    frames, why = _ask(cond, "ns/Horse", "walk, forward", loop=True)
    assert frames == 84
    assert "ns/Horse" in why


def test_loop_falls_through_to_the_default_rather_than_borrowing_one_shots():
    """A label the corpus never annotated as a loop ('walk, turn, left') has no
    period to answer with, at any tier -- so the caller keeps its own default."""
    cond = {
        "ns/Horse": _entry([("locomotion", "walk, turn, left", False, 28)]),
        "ns/Cat": _entry([("locomotion", "walk, turn, left", False, 30)] * 4),
    }
    assert _ask(cond, "ns/Horse", "walk, turn, left", loop=True) is None
    # The same ask without --loop is answered from those very clips.
    assert _ask(cond, "ns/Horse", "walk, turn, left", loop=False)[0] == 28


def test_head_word_fallback_stays_inside_the_group():
    cond = {"ns/Horse": _entry([("stationary", "walk, backward", True, 84)])}
    assert _ask(cond, "ns/Horse", "walk, forward", loop=True) is None


def test_neighbours_are_pooled_when_the_species_has_no_such_clip():
    cond = {
        "ns/NewRig": _entry([]),
        "ns/A": _entry([("locomotion", "walk, forward", True, 40)] * 3),
        "ns/B": _entry([("locomotion", "walk, forward", True, 60)] * 3),
    }
    frames, why = _ask(cond, "ns/NewRig", "walk, forward", loop=True)
    assert frames == 50  # A is not deep enough on its own, so B joins the pool
    assert "similar species" in why


def test_neighbour_pooling_stops_once_the_pool_is_deep_enough():
    """Each further species is less like the target than the last, so widening
    stops at the first one that makes the median trustworthy."""
    cond = {
        "ns/NewRig": _entry([]),
        "ns/A": _entry([("locomotion", "walk, forward", True, 40)] * 5),
        "ns/B": _entry([("locomotion", "walk, forward", True, 60)] * 5),
    }
    frames, why = _ask(cond, "ns/NewRig", "walk, forward", loop=True)
    assert frames == 40
    assert "1 similar species" in why


def test_loop_request_never_borrows_a_one_shot_length():
    """A one-shot clip's length is not a period; it would mis-time --loop."""
    cond = {"ns/Horse": _entry([("locomotion", "walk, forward", False, 130)])}
    assert _ask(cond, "ns/Horse", "walk, forward", loop=True) is None
    frames, _ = _ask(cond, "ns/Horse", "walk, forward", loop=False)
    assert frames == 120  # clamped to the source-frame budget


def test_each_clip_is_clamped_before_the_median():
    """Mirrors the loader: an over-long clip is cropped to the budget, so what
    it contributed to training is the budget -- not its own length."""
    cond = {"ns/Horse": _entry([
        ("locomotion", "walk, forward", True, 50),
        ("locomotion", "walk, forward", True, 400),
    ])}
    frames, _ = _ask(cond, "ns/Horse", "walk, forward", loop=True, max_frames=120)
    assert frames == 85  # median(50, 120), not median(50, 400) clamped to 120


def test_result_is_clamped_into_the_legal_range():
    cond = {"ns/Horse": _entry([("locomotion", "run, forward", True, 21)])}
    frames, _ = _ask(cond, "ns/Horse", "run, forward", loop=True, min_frames=30)
    assert frames == 30


def test_all_species_mode_pools_the_whole_corpus():
    """With no target to rank against, stopping early would answer from
    whichever species happened to sort first."""
    cond = {
        f"ns/S{i}": _entry([("locomotion", "walk, forward", True, length)])
        for i, length in enumerate([30, 30, 30, 90, 90, 90, 90])
    }
    frames, why = _ask(cond, None, "walk, forward", loop=True)
    assert frames == 90
    assert "all species" in why


def test_empty_label_and_missing_table_infer_nothing():
    cond = {"ns/Horse": _entry([("locomotion", "walk, forward", True, 84)])}
    assert _ask(cond, "ns/Horse", "") is None
    assert _ask({"ns/Horse": {}}, "ns/Horse", "walk, forward") is None


def test_a_future_schema_is_ignored_rather_than_guessed_at():
    cond = {"ns/Horse": _entry([("locomotion", "walk, forward", True, 84)])}
    cond["ns/Horse"][COND_KEY] = dict(
        cond["ns/Horse"][COND_KEY], schema=PRIOR_SCHEMA_VERSION + 1
    )
    assert not has_clip_length_prior(cond)
    assert _ask(cond, "ns/Horse", "walk, forward", loop=True) is None


def test_has_clip_length_prior_separates_no_table_from_no_match():
    assert has_clip_length_prior({"ns/Horse": _entry([("locomotion", "walk", True, 40)])})
    assert not has_clip_length_prior({"ns/Horse": {}})
    # A species with a table but no clips still counts as baked: the bake ran, so
    # "this label is not in the corpus" is the answer, not "re-bake the cond".
    assert has_clip_length_prior({"ns/Horse": _entry([])}) is True


def test_merge_prior_pool_keeps_the_active_skeleton_but_borrows_the_table():
    """A narrow --cond_path cut out of a full cond keeps its own skeleton and
    inherits the prior it was cut away from."""
    primary = {"ns/Horse": {"parents": "narrow"}}
    fallback = {
        "ns/Horse": _entry([("locomotion", "walk, forward", True, 84)], parents="full"),
        "ns/Cat": _entry([("locomotion", "walk, forward", True, 30)]),
    }
    pool = merge_prior_pool(primary, fallback)
    assert pool["ns/Horse"]["parents"] == "narrow"
    assert set(pool) == {"ns/Horse", "ns/Cat"}
    frames, why = _ask(pool, "ns/Horse", "walk, forward", loop=True)
    assert frames == 84
    assert "ns/Horse" in why


def test_merge_prior_pool_never_overwrites_an_existing_table():
    primary = {"ns/Horse": _entry([("locomotion", "walk, forward", True, 84)])}
    fallback = {"ns/Horse": _entry([("locomotion", "walk, forward", True, 20)])}
    pool = merge_prior_pool(primary, fallback)
    assert _ask(pool, "ns/Horse", "walk, forward", loop=True)[0] == 84


# -- generate.py wiring ------------------------------------------------------
def test_generate_num_frames_defaults_to_none_meaning_auto():
    """``None`` is the whole 'auto' spelling: there is no literal token, so a
    caller that only passes --num_frames when it has a number (the HTTP server)
    gets the auto path for free."""
    import argparse

    from utils.parser_util import add_generate_options

    parser = argparse.ArgumentParser()
    add_generate_options(parser)
    assert parser.parse_args([]).num_frames is None
    assert parser.parse_args(["--num_frames", "75"]).num_frames == 75


def test_auto_output_lengths_falls_back_to_the_native_window(capsys):
    from sample.output_lengths import _resolve_auto_output_lengths

    cond = {"ns/Horse": _entry([("locomotion", "walk, forward", True, 84)])}
    # No action label -> nothing to key a length on.
    requested, target, speed = _resolve_auto_output_lengths(
        cond, "ns/Horse", None,
        min_length=20, internal_num_frames=60, default_frames=60, loop=False,
    )
    assert (requested, target, speed) == (60, 60, 1.0)
    assert "no --action_label" in capsys.readouterr().out

    # A label the corpus knows -> its median, and the matching speed condition.
    requested, target, speed = _resolve_auto_output_lengths(
        cond, "ns/Horse",
        {"action_group": "locomotion", "action_label": "walk, forward"},
        min_length=20, internal_num_frames=60, default_frames=60, loop=True,
    )
    assert (requested, target) == (84, 84)
    assert speed == pytest.approx(84 / 60)


def test_auto_output_lengths_names_an_unbaked_cond(capsys):
    from sample.output_lengths import _resolve_auto_output_lengths

    frames, _, _ = _resolve_auto_output_lengths(
        {"ns/Horse": {}}, "ns/Horse",
        {"action_group": "locomotion", "action_label": "walk, forward"},
        min_length=20, internal_num_frames=60, default_frames=60, loop=False,
    )
    assert frames == 60
    assert "no clip-length prior" in capsys.readouterr().out


def test_auto_output_lengths_tells_a_baked_cond_apart_from_an_unbaked_one(capsys):
    """A baked cond whose clips carry no label is not an unbaked one: re-running
    the bake would change nothing, so the user must not be sent there."""
    from sample.output_lengths import _resolve_auto_output_lengths

    frames, _, _ = _resolve_auto_output_lengths(
        {"ns/Horse": _entry([])}, "ns/Horse",
        {"action_group": "locomotion", "action_label": "walk, forward"},
        min_length=20, internal_num_frames=60, default_frames=60, loop=False,
    )
    assert frames == 60
    out = capsys.readouterr().out
    assert "no training clip matches" in out
    assert "no clip-length prior" not in out


def test_auto_output_lengths_falls_back_to_the_checkpoint_cond(capsys):
    """A one-species --cond_path has no neighbours; the checkpoint's own cond is
    the pool the weights were trained on."""
    from sample.output_lengths import _resolve_auto_output_lengths

    calls = []

    def loader():
        calls.append(1)
        return {"ns/Horse": _entry([("locomotion", "walk, forward", True, 84)])}

    frames, _, _ = _resolve_auto_output_lengths(
        {"ns/Horse": {}}, "ns/Horse",
        {"action_group": "locomotion", "action_label": "walk, forward"},
        min_length=20, internal_num_frames=60, default_frames=60, loop=True,
        fallback_cond_loader=loader,
    )
    assert frames == 84
    assert "checkpoint's cond" in capsys.readouterr().out
    assert len(calls) == 1

    # The active cond answering on its own never pays for the second load.
    calls.clear()
    _resolve_auto_output_lengths(
        {"ns/Horse": _entry([("locomotion", "walk, forward", True, 40)])}, "ns/Horse",
        {"action_group": "locomotion", "action_label": "walk, forward"},
        min_length=20, internal_num_frames=60, default_frames=60, loop=True,
        fallback_cond_loader=loader,
    )
    assert calls == []


# -- --loop auto ---------------------------------------------------------------
def _ask_loop(cond, target, label, *, group="transition"):
    from utils.clip_length_prior import auto_loop

    return auto_loop(cond, target, action_group=group, action_label=label)


def test_auto_loop_follows_the_majority_of_the_labels_clips():
    """The verdict is the majority of the label's clips, so an unflagged
    request never pairs a label the corpus authors as loops with is_loop False."""
    cond = {"ns/Horse": _entry(
        [("transition", "jump, up", True, 20)] * 4
        + [("transition", "jump, up", False, 40)]
    )}
    is_loop, why = _ask_loop(cond, "ns/Horse", "jump, up")
    assert is_loop is True
    assert why.startswith("4 of 5 ns/Horse clip(s) matching exact label")

    cond = {"ns/Horse": _entry([("transition", "die", False, 40)] * 3)}
    assert _ask_loop(cond, "ns/Horse", "die") == (
        False, "0 of 3 ns/Horse clip(s) matching exact label are loops",
    )


def test_auto_loop_tie_keeps_the_open_window():
    cond = {"ns/Horse": _entry([
        ("transition", "jump, up", True, 20), ("transition", "jump, up", False, 40),
    ])}
    assert _ask_loop(cond, "ns/Horse", "jump, up")[0] is False


def test_auto_loop_walks_the_same_ladder_as_the_length():
    """Own species before neighbours, exact label before head words, and the
    verdict is read off the same clips the auto length is."""
    cond = {
        "ns/Horse": _entry([("transition", "jump, forward", False, 40)]),
        "ns/Deer": _entry([("transition", "jump, up", True, 20)] * 3),
    }
    # Own head-word clip outranks a neighbour's exact label.
    is_loop, why = _ask_loop(cond, "ns/Horse", "jump, up")
    assert is_loop is False and "ns/Horse" in why and "head word(s) 'jump'" in why
    # A species with no jump at all borrows from the neighbour.
    is_loop, why = _ask_loop(cond, "ns/Cow", "jump, up")
    assert is_loop is True and "ns/Deer" in why
    # Nothing in the corpus -> caller's default.
    assert _ask_loop(cond, "ns/Horse", "swim") is None
    assert _ask_loop(cond, "ns/Horse", "") is None


def test_resolve_loop_condition_modes(capsys):
    from sample.output_lengths import resolve_loop_condition

    cond = {"ns/Horse": _entry([("transition", "jump, up", True, 20)] * 3)}
    label = {"action_group": "transition", "action_label": "jump, up"}
    common = {}
    # Explicit modes never consult the corpus.
    assert resolve_loop_condition("on", {}, "ns/Horse", None, **common) is True
    assert resolve_loop_condition("off", cond, "ns/Horse", label, **common) is False
    assert resolve_loop_condition(True, {}, "ns/Horse", None, **common) is True
    # auto follows the label's clips...
    assert resolve_loop_condition("auto", cond, "ns/Horse", label, **common) is True
    assert "loop (auto) -> on (3 of 3" in capsys.readouterr().out
    # ...follows a reference over the label...
    open_clip = _clip_with_verdict(False)
    assert resolve_loop_condition(
        "auto", cond, "ns/Horse", label, reference_features=open_clip, translation_root_index=0,
    ) is False
    assert "stored loop verdict" in capsys.readouterr().out
    # ...and is off with nothing to key on.
    assert resolve_loop_condition("auto", cond, "ns/Horse", None, **common) is False
    assert "no --action_label" in capsys.readouterr().out
    assert resolve_loop_condition(
        "auto", cond, "ns/Horse", {"action_group": "locomotion", "action_label": "swim"}, **common,
    ) is False
    assert "no training clip matches 'swim'" in capsys.readouterr().out
    with pytest.raises(ValueError):
        resolve_loop_condition("maybe", cond, "ns/Horse", label, **common)


def test_resolve_loop_condition_borrows_the_checkpoint_cond(capsys):
    from sample.output_lengths import resolve_loop_condition

    narrow = {"ns/NewRig": {}}
    full = {"ns/Horse": _entry([("transition", "jump, up", True, 20)] * 2)}
    label = {"action_group": "transition", "action_label": "jump, up"}
    assert resolve_loop_condition(
        "auto", narrow, "ns/NewRig", label, fallback_cond_loader=lambda: full,
    ) is True
    assert "from the checkpoint's cond" in capsys.readouterr().out


def _clip_with_verdict(is_loop, frames=40, joints=4, seed=0):
    """A random (T, J, 12) clip whose terminal row was written under ``is_loop``,
    the way preprocessing (or a review-UI flip) leaves a stored clip."""
    from data_loaders.truebones.truebones_utils.loop_verdict import apply_loop_verdict

    rng = np.random.default_rng(seed)
    clip = rng.normal(size=(frames, joints, 12)).astype(np.float32) * 0.1
    return apply_loop_verdict(clip, is_loop, 0)


def test_reference_loop_verdict_reads_the_stored_row_first():
    """A stored clip's terminal row IS its verdict (hand-verified for a dataset
    clip), so an open-looking loop annotated as one still reads as a loop."""
    from data_loaders.truebones.truebones_utils.loop_verdict import stored_loop_verdict
    from sample.output_lengths import reference_loop_verdict

    for verdict in (True, False):
        clip = _clip_with_verdict(verdict)
        assert stored_loop_verdict(clip, 0) is verdict
        is_loop, why = reference_loop_verdict(clip, 0)
        assert is_loop is verdict and "stored loop verdict" in why


def test_reference_loop_verdict_falls_back_to_the_detector():
    """A tensor nobody wrote a verdict into (a generated sample) is judged on
    its geometry: a closed clip is a loop, an open one is not."""
    from data_loaders.truebones.truebones_utils.loop_verdict import stored_loop_verdict
    from sample.output_lengths import reference_loop_verdict

    frames, joints = 40, 4
    phase = np.linspace(0.0, 2.0 * np.pi, frames, endpoint=False)
    closed = np.zeros((frames, joints, 12), dtype=np.float32)
    # A small circular orbit: its per-frame steps sit at the detector's physical scale.
    closed[..., 0] = 0.2 * np.sin(phase)[:, None]
    closed[..., 1] = 0.2 * np.cos(phase)[:, None] + np.arange(joints)[None, :]
    # The last velocity row is model output, not a verdict: scribble on it.
    closed[-1, :, 9:12] = 0.37
    assert stored_loop_verdict(closed, 0) is None
    is_loop, why = reference_loop_verdict(closed, 0)
    assert is_loop is True and "no stored verdict" in why

    open_clip = closed.copy()
    open_clip[..., 1] += np.linspace(0.0, 1.0, frames)[:, None]  # drifts away
    assert stored_loop_verdict(open_clip, 0) is None
    is_loop, why = reference_loop_verdict(open_clip, 0)
    assert is_loop is False and "do not close" in why


def test_loop_flag_spellings():
    """A bare --loop still means on; omitted is auto; on/off are explicit."""
    import argparse

    from utils.parser_util import add_sampling_options

    parser = argparse.ArgumentParser()
    add_sampling_options(parser)
    base = ["--model_path", "x"]
    assert parser.parse_args(base).loop == "auto"
    assert parser.parse_args(base + ["--loop"]).loop == "on"
    assert parser.parse_args(base + ["--loop", "--fullbody_ik"]).loop == "on"
    assert parser.parse_args(base + ["--loop", "off"]).loop == "off"
    assert parser.parse_args(base + ["--loop", "auto"]).loop == "auto"


def test_all_species_lengths_resolve_loop_per_species(capsys):
    from sample.output_lengths import _all_species_output_lengths

    cond = {
        "ns/Horse": _entry([("transition", "jump, up", True, 20)] * 2),
        "ns/Frog": _entry([("transition", "jump, up", False, 40)] * 2),
    }
    lengths = _all_species_output_lengths(
        cond, {"action_group": "transition", "action_label": "jump, up"},
        explicit=None, min_length=10, internal_num_frames=60, default_frames=60,
        loop_mode="auto",
    )
    assert lengths["ns/Horse"] == (20, pytest.approx(20 / 60), True)
    assert lengths["ns/Frog"] == (40, pytest.approx(40 / 60), False)
    assert "loop (auto): 1 of 2 species closed" in capsys.readouterr().out
    # An explicit length still gets its own loop verdict per species.
    lengths = _all_species_output_lengths(
        cond, {"action_group": "transition", "action_label": "jump, up"},
        explicit=(30, 0.5), min_length=10, internal_num_frames=60, default_frames=60,
        loop_mode="on",
    )
    assert lengths == {"ns/Horse": (30, 0.5, True), "ns/Frog": (30, 0.5, True)}
