"""Actions that take no PLANAR direction word (2026-09-19).

``left`` / ``right`` / ``forward`` / ``backward`` say which way an action is
aimed or travels. A hit reaction, a body state, a draw and a head strike are
aimed nowhere, so a side word on them is read off an incidental lean and spends
part of a condition every species shares. The rule therefore runs three ways,
and all three are checked here:

* no sidecar row spells one (the corpus invariant, checked against the real
  ``action_labels.jsonl`` files -- a relabel that reintroduces one fails here);
* ``prefill_direction_words`` never proposes one, and never calibrates on one;
* the audit rule that DEMANDS a direction (R4's heading) does not demand it of
  these actions.  R3 demands none of anybody any more -- it never reads a
  direction off a clip name -- which is checked here too.

``hover`` is deliberately NOT on this list: a hovering strike IS aimed
somewhere ("attack, hover, left, swat"). It only exempts R4's heading.

The VERTICAL axis is untouched throughout: ``idle, up, aim, bow`` aims upward
and keeps its word.
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.truebones.truebones_utils.motion_labels import (
    ACTION_VOCAB,
    parse_action_label,
)
from tools.audit_action_labels import (
    NO_PLANAR_DIRECTION_WORDS,
    PLANAR_DIRECTIONS,
    check_r3,
    check_r4,
    takes_planar_direction,
)
from tools.prefill_direction_words import jump_verdict, kind_of

ANYTOP_DIR = Path(__file__).resolve().parent.parent
SIDECARS = [
    ANYTOP_DIR / "dataset" / "truebones" / "zoo" / "truebones_processed",
    ANYTOP_DIR / "dataset" / "truebones" / "zoo_upgrade" / "clean_processed",
    ANYTOP_DIR / "dataset" / "unitybundles" / "processed",
]


def _clip(name, label, group="stationary", species="zoo/Trex"):
    return {"clip": name, "species": species, "group": group, "label": label,
            "gif_path": "", "motion_path": ""}


# ── the rule itself ──────────────────────────────────────────────────────────

def test_every_exempt_word_is_a_real_action_word():
    """A word nothing can spell would exempt nothing (the module raises on it)."""
    assert set(NO_PLANAR_DIRECTION_WORDS) <= set(ACTION_VOCAB)


def test_the_exemption_matches_anywhere_in_the_label_not_just_the_head():
    assert not takes_planar_direction(["idle", "look"])
    assert not takes_planar_direction(["attack", "bite"])       # modifier
    assert not takes_planar_direction(["attack", "hover", "headbutt"])
    assert takes_planar_direction(["attack", "swat"])           # an aimed strike
    assert takes_planar_direction(["walk"])


# ── the corpus ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("root", SIDECARS, ids=lambda p: p.parent.name)
def test_no_sidecar_row_spells_a_planar_direction_on_these_actions(root):
    path = root / "action_labels.jsonl"
    if not path.exists():
        pytest.skip(f"{path} is not in this checkout")
    offenders = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        words = list(parse_action_label(row.get("action_label", "")))
        if not takes_planar_direction(words) and set(words) & set(PLANAR_DIRECTIONS):
            offenders.append((row.get("clip"), row.get("action_label")))
    assert offenders == []


def test_the_vertical_axis_survives_on_the_same_actions():
    """'idle, up, aim, bow' aims upward: the exemption is planar only."""
    path = SIDECARS[-1] / "action_labels.jsonl"
    if not path.exists():
        pytest.skip(f"{path} is not in this checkout")
    labels = {json.loads(line)["action_label"]
              for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    assert any("up" in parse_action_label(label) and not takes_planar_direction(parse_action_label(label))
               for label in labels)


# ── the prefill tool ─────────────────────────────────────────────────────────

def test_an_exempt_row_is_out_of_the_prefill_scope():
    assert kind_of(_clip("Trex_LookLeft", "idle, look")) is None
    assert kind_of(_clip("Dog_HitLeft", "hurt")) is None
    assert kind_of(_clip("KI_Archer_Sheathe01", "sheathe", "transition")) is None
    assert kind_of(_clip("Imp_KnockedBack", "hurt, fall", "locomotion")) is None


def test_a_row_that_is_aimed_somewhere_stays_in_scope():
    assert kind_of(_clip("Wolf_AtkL", "attack, swat")) == "side"
    assert kind_of(_clip("Camel_Walk", "walk", "locomotion")) == "heading"
    assert kind_of(_clip("Frog_Jump", "jump", "transition")) == "jump"


def test_a_death_goes_to_the_topple_measurement_not_the_side_one():
    """A death is not aimed with a limb, and 30 of 41 say forward / backward."""
    assert kind_of(_clip("Hippopotamus_Die", "die", "transition")) == "topple"
    assert kind_of(_clip("MB_Unka_DeathLeft", "die, hover", "transition")) == "topple"


def test_an_exempt_jump_may_still_be_written_up_but_never_a_planar_word():
    """The vertical half of the jump verdict is not exempt; the planar half is."""

    class _Args:
        jump_up_max = 0.05
        jump_planar_min = 0.15

    up = {"rise": 0.8, "air_frames": 9, "air_disp": [0.0, 0.0], "net": [0.0, 0.0]}
    planar = {"rise": 0.8, "air_frames": 9, "air_disp": [0.0, 0.9], "net": [0.0, 0.9]}
    exempt = _clip("Frog_HopUp", "jump, idle", "transition")
    exempt["is_loop"] = False
    plain = _clip("Frog_Hop", "jump", "transition")
    plain["is_loop"] = False

    assert jump_verdict(_np(up), exempt, _Args())[:2] == ("write", ["up"])
    assert jump_verdict(_np(planar), plain, _Args())[:2] == ("write", ["forward"])
    status, words, reason = jump_verdict(_np(planar), exempt, _Args())
    assert (status, words) == ("review", []) and "no planar direction" in reason


def _np(measure):
    import numpy as np
    return {**measure, "air_disp": np.asarray(measure["air_disp"], dtype=float),
            "net": np.asarray(measure["net"], dtype=float)}


# ── the audit rules ──────────────────────────────────────────────────────────

def test_r3_passes_an_exempt_mirror_pair_spelled_the_same():
    """Both halves spelled the same IS the mirror for an action aimed nowhere."""
    findings, stats = check_r3([_clip("Trex_BiteLeft", "attack, bite"),
                                _clip("Trex_BiteRight", "attack, bite")])
    assert findings == []
    assert (stats["same_label"], stats["same_label_aimed"]) == (1, 0)


def test_r3_counts_an_aimed_pair_spelled_the_same_but_does_not_report_it():
    """Whether those two are mirror TAKES is a fact about the motion.

    ``prefill_direction_words`` measures it there (mirror detection on the
    positions, then side energy). R3 only has the names, and the names cannot
    say either that the clips are mirror takes or which side each one strikes.
    """
    findings, stats = check_r3([_clip("Wolf_SwatLeft", "attack, swat"),
                                _clip("Wolf_SwatRight", "attack, swat")])
    assert findings == []
    assert (stats["same_label"], stats["same_label_aimed"]) == (1, 1)


def test_r3_passes_a_pair_whose_side_words_look_crossed_against_the_names():
    """MB_Unka_DeathLeft really does fall to the character's right.

    Such a pair is a valid mirror of itself, and only the clip NAME says it is
    backwards -- so R3 says nothing about it. This is the check that produced
    four findings against four correct labels before it was removed.
    """
    findings, stats = check_r3([_clip("MB_Unka_DeathLeft", "die, right", "transition"),
                                _clip("MB_Unka_DeathRight", "die, left", "transition")])
    assert findings == [] and stats["same_label"] == 0


def test_r3_still_reports_an_exempt_pair_that_is_not_a_mirror():
    """The exemption lifts the side-word demand, not the mirror check itself."""
    findings, _stats = check_r3([_clip("Dog_HitLeft", "hurt"),
                                 _clip("Dog_HitRight", "hurt, fall")])
    assert [code for item in findings for code in item["problem_codes"]] == ["not_mirror"]


def test_r4_demands_no_heading_of_an_exempt_locomotion_row():
    findings, stats = check_r4([_clip("Horse_RunToStop", "stop", "locomotion")])
    assert findings == [] and stats["no_heading_clips"] == 1


def test_r4_still_demands_a_heading_of_an_aimed_locomotion_row():
    findings, _stats = check_r4([_clip("Camel_Walk", "walk", "locomotion")])
    assert [item["clip"] for item in findings] == ["Camel_Walk"]
