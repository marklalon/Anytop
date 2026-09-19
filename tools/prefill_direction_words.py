#!/usr/bin/env python3
"""Propose the missing DIRECTION word of an ``action_labels.jsonl`` row from its motion.

The direction slot has a dropout in training (``--direction_slot_drop_prob``),
so an empty slot means "the marginal over directions" and nothing else. That
makes every row whose content HAS one dominant heading or side and spells none
a mislabel: the model learns that ``attack, swat`` is a right swat, and a bare
prompt gets a right swat instead of either. This tool finds those rows and
fills them, under four rules that never bend:

* Only an EMPTY slot is filled. A direction word already on a row is a
  hand-verified truth; the tool calibrates its own sign and thresholds against
  those rows and never rewrites one. A row filled here is marked
  ``"reviewed": false`` and flagged ``"autofill": true``, so the review UI
  lists it for a person.
* A ``reviewed: true`` row is left alone. A person watched that GIF and
  settled the label, so its empty slot says "this clip is aimed nowhere" and
  is not a gap to fill. The row keeps every other job it had -- it calibrates,
  it is a mirror partner, it is a labelled reference -- and only stops being a
  target. ``--include-reviewed`` judges those rows too, which is what to pass
  when a rule change (a new exemption, a re-measured axis) makes the old
  sign-offs stale.
* The verdict comes from the MOTION only. The clip name is the weakest
  evidence there is and is not consulted; it is not even written as evidence.
* Nothing is written until the measurement has been checked against the rows
  a person already labelled, and a measurement below its gate only produces
  the review list (``--dry-run`` is the default; ``--apply`` writes).

Out of scope, whatever the motion says: the actions that have no planar
direction to name (``audit_action_labels.NO_PLANAR_DIRECTION_WORDS`` -- hurt,
getup, idle, rest, stop, draw, sheathe, headbutt, bite). A hit reaction or an
idle is aimed nowhere, a headbutt is delivered with a CENTER joint, and the
side of a draw is the scabbard's, so the side energy of such a clip reads its
incidental lean and would spell it into a condition every species shares. Those
rows are neither judged nor used to calibrate. Their VERTICAL word is
untouched ("idle, up, aim, bow" keeps it), and a transition jump is still
judged for ``up``.

Four measurements, one per kind of row:

  SIDE  (stationary and transition rows, ``left`` / ``right``)
      The clip's positions are split into their mirror-symmetric and
      mirror-antisymmetric halves (partner joints swapped, X negated). A still
      clip or one whose antisymmetric half is small is symmetric and stays
      empty. An asymmetric clip is then judged by the velocity energy of its
      left joints against its right joints: a clear majority names the side; a
      balanced one (alternating strikes, a rotation, a centred motion) is
      listed. Rigs with no mirror pairs (``cond.is_symmetric`` false) and
      joints without a partner (a one-armed rig's arm) are left out of both
      measurements. A ``turn`` row is out of scope: its side word is the yaw
      direction, not a limb, and so is a ``die`` row: a death is delivered
      with no limb at all and its word names where the BODY goes (TOPPLE
      below). Stationary rows only ever receive a side word -- a stationary
      clip has no travel to spell ``forward`` for.

  HEADING  (locomotion rows, ``forward`` / ``backward`` / ``left`` / ``right``)
      Preprocessing detrends every locomotion clip's root XZ, so the net travel
      is gone from the tensor. What survives is the support foot: over the
      frames where a contact joint is on the ground, its velocity relative to
      the root is minus the travel direction. That is the only usable cue and
      it only exists for a gait with ground contact (walk / run / crawl);
      swim / fly / roll rows are listed, never filled. A diagonal (foot velocity
      split across two axes) spells two words, as the labelled strafes do.

  JUMP  (transition rows with ``jump``, ``up`` or a planar word)
      With the direction slot dropped out at random, a bare ``jump`` cannot
      also mean "vertical jump", so a vertical jump is spelled ``jump, up``
      like ``fly, up``. The root's XZ displacement over its airborne frames
      (highest root height) decides: none = ``up``, a clear one = its planar
      word(s). A non-loop transition still carries its net root travel, which
      is used only as a cross-check (a contradiction lists the row). ``land``
      rows keep their spelling -- ``land`` carries the vertical itself.

  TOPPLE  (``die`` rows, one of ``forward`` / ``backward`` / ``left`` / ``right``)
      A death's direction is the way the body goes down. It is not a side --
      the hand-labelled deaths spell ``forward`` and ``backward`` as often as
      ``left`` and ``right``, which the side measurement cannot even produce
      -- and it is not a limb: the side measurement would read it off the
      twitch of a corpse's arms. It is the horizontal travel of the joint
      CENTROID, the clip's first tenth against its last, in body lengths. The
      centroid rather than the root, because a quadruped that keels over where
      it stands moves its root by nothing (BrownBear_Twitching: root 0.00,
      centroid 0.60). Its dominant axis names the fall with ONE word, the way
      every labelled death is spelled, and reproduces all 41 of them.

      A death only HAS that direction when the body topples onto the ground,
      and four things say it did not. None of them is a proposal: the empty
      slot is correct, so they are ``keep``.
        * the label already says the death happens in the air -- ``hover``,
          ``fall``, ``fly`` (``AIRBORNE_DEATH_WORDS``). A creature that dies in
          flight falls, and what its body does in the ground plane on the way
          down is the last of its flight momentum plus the arc of the fall:
          ``MLS_BattleOwl_Die`` drifts 2.21 body lengths and topples toward
          nothing. This is ``NO_HEADING_WORDS``' own argument about ``hover``,
          and the first version of this measurement overrode it, wrongly.
        * the body starts off the ground although the label does not say so
          (``--topple-start-clearance-max``): the same fall, unspelled.
        * the body never descends (``--topple-min-drop``): nothing toppled, it
          slid or settled where it stood. ``Cobra_Death`` travels 0.80 body
          lengths and drops 0.21.
        * the body barely moves in the plane (``--topple-min-travel``): it
          collapsed in place, which is aimed nowhere.
      Every threshold sits outside the whole hand-labelled set, so not one of
      the 41 falls foul of one.

Calibration (printed first, always): the side measurement's sign and margin on
the rows that already have ``left`` / ``right``, the heading measurement on the
labelled ground gaits, the jump measurement on the labelled jumps, the topple
measurement on the labelled deaths. Each has a gate (``--side-gate`` 0.90,
``--heading-gate`` 0.95, ``--jump-gate`` 0.90, ``--topple-gate`` 0.90); a
measurement under its gate never writes.

Output: the console summary and ``--report`` (CSV, one line per judged row).
``--apply`` writes the ``write`` rows into their sidecars, skipping any row a
person filled or reviewed since the run started; a written row comes back as
``reviewed:false`` and is reviewed in ``dataset/review/serve.py`` with the rest.

Usage:
    python tools/prefill_direction_words.py                       # dry run, all groups
    python tools/prefill_direction_words.py --action-group stationary
    python tools/prefill_direction_words.py --apply
    python tools/prefill_direction_words.py --include-reviewed  # reviewed rows too
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ANYTOP_DIR = Path(__file__).resolve().parent.parent
for _candidate in (str(ANYTOP_DIR), str(ANYTOP_DIR.parent)):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from data_loaders.truebones.truebones_utils.motion_labels import (  # noqa: E402
    ACTION_VOCAB,
    DIRECTION_VOCAB,
)
from tools.audit_action_labels import (  # noqa: E402
    NO_HEADING_WORDS,
    NO_PLANAR_DIRECTION_WORDS,
    PLANAR_DIRECTIONS,
    VERTICAL_WORDS,
    label_words,
    takes_planar_direction,
)
from tools.prefill_common import (  # noqa: E402
    DecodedClip,
    Proposal,
    add_reviewed_flag,
    apply_proposals,
    check_head_order,
    drop_reviewed,
    head_of,
    load_corpus,
    spell_with,
    write_csv,
)

# Frames per second of every stored clip; speeds below read in body lengths / s.
FPS = 30.0

# ── side (stationary / transition) ──
# Antisymmetric amplitude under which a clip is "still" (body lengths, RMS).
SIDE_AMP_FLOOR = 0.05
# Antisymmetric share of the motion (0 symmetric .. 1 alternating) under which a
# clip is symmetric and stays empty without being listed ...
SIDE_SYM_MAX = 0.15
# ... and from which it is asymmetric enough to be judged by side energy.
SIDE_ASYM_MIN = 0.30
# |left share of side-joint velocity energy - 0.5| from which a side is named.
SIDE_RATIO_MARGIN = 0.20
# Mirror distance under which two clips of one species are mirror takes.
MIRROR_PAIR_MAX = 0.15
MIRROR_PAIR_FRAMES = 32

# ── heading (locomotion) ──
GROUND_GAIT_HEADS = ("walk", "run", "crawl")
# Contact frames: a contact joint within this share of its height range from its lowest.
CONTACT_HEIGHT_SHARE = 0.20
# Support-foot travel speed (body lengths / s) under which no heading is read.
HEADING_MIN_SPEED = 1.0
# Second planar axis at least this share of the first -> two words (a strafe).
HEADING_DIAG_SHARE = 0.5

# ── jump (transition) ──
JUMP_MIN_RISE = 0.30          # root Y range (body lengths) to count as a jump
JUMP_AIR_SHARE = 0.60         # airborne = root above this share of its height range
JUMP_UP_MAX = 0.05            # airborne XZ displacement (body lengths) at most -> up
JUMP_PLANAR_MIN = 0.15        # ... at least -> planar word(s)
JUMP_NET_MIN = 0.30           # non-loop cross-check: net root travel that contradicts "up"

# ── topple (a death's own travel) ──
# The words whose direction names where the BODY ends up rather than which limb
# delivers the action.  A death is the only one in the corpus, and it is the
# reason this measurement exists: routed to SIDE (which is where every
# non-locomotion row went before), a death could only ever come out ``left`` or
# ``right``, while 30 of the 41 hand-labelled ones say ``forward`` or
# ``backward``.  Matched anywhere in the label, like NO_PLANAR_DIRECTION_WORDS.
BODY_TRAVEL_WORDS = ("die",)
# Ends of the clip averaged for the travel: first frames // this against last.
TOPPLE_END_SHARE = 10
# Centroid travel (body lengths) under which the body collapsed in place and
# has no planar direction to name.  The smallest labelled death travels 0.46.
TOPPLE_MIN_TRAVEL = 0.40
# Off-axis share (weaker axis / stronger) over which the fall is a diagonal
# with no dominant axis.  Deliberately loose: a person picked the dominant axis
# at 0.81 (KI_Warrior_Death01A, "left"), so only a near-tie is listed.  A death
# is spelled with ONE word -- none of the 41 labelled ones carries two.
TOPPLE_TIE_SHARE = 0.85
# Words that say the death happens IN THE AIR, so there is no topple to name a
# direction of: the horizontal component is the fall's, not the body's aim.
# Matched anywhere in the label.  No hand-labelled death carries one of these,
# so nothing a person spelled is rejected by it.
AIRBORNE_DEATH_WORDS = ("hover", "fall", "fly")
# ... and the same fall with nothing in the label to say so: the lowest joint's
# clearance above the clip's floor, over the first tenth, in body lengths.  The
# gate sits just above the highest a hand-labelled death has -- LH_Hero_FlyDie
# at 1.00, a flying death a person DID spell -- so it rejects only what starts
# higher still (MU01_Bird_Die, 1.05).  The margin is thin on purpose: the
# clearance axis does not separate those two clips, a person's judgement did,
# and this errs on the side of the person.
TOPPLE_START_CLEARANCE_MAX = 1.00
# Centroid drop under which nothing toppled -- the body slid or settled where
# it stood (Cobra_Death: travels 0.80, drops 0.21).  The smallest hand-labelled
# death drops 0.36.
TOPPLE_MIN_DROP = 0.35

_unknown_body_travel = set(BODY_TRAVEL_WORDS) - set(ACTION_VOCAB)
if _unknown_body_travel:
    raise RuntimeError(
        "BODY_TRAVEL_WORDS names %s, which ACTION_VOCAB does not have: a word "
        "nothing can spell routes nothing" % ", ".join(sorted(_unknown_body_travel))
    )
# kind_of() tests the exemption FIRST, so a word on both lists would be dead
# here and the topple measurement would silently never run for it.
_both_lists = set(BODY_TRAVEL_WORDS) & set(NO_PLANAR_DIRECTION_WORDS)
if _both_lists:
    raise RuntimeError(
        "%s is in BODY_TRAVEL_WORDS and NO_PLANAR_DIRECTION_WORDS at once; an "
        "action either has a planar direction or it does not"
        % ", ".join(sorted(_both_lists))
    )

PLANAR_AXES = {
    "forward": np.array([0.0, 1.0]),
    "backward": np.array([0.0, -1.0]),
    "left": np.array([1.0, 0.0]),     # facing +Z with +Y up, +X is the character's left
    "right": np.array([-1.0, 0.0]),
}
SIDE_WORDS = ("left", "right")


# ── measurements ─────────────────────────────────────────────────────────────

class Symmetry:
    """A rig's mirror map on the joints that take part in the side measurement."""

    def __init__(self, entry):
        partner = np.asarray(entry["symmetry_partner_indices"], dtype=int)
        side = np.asarray(entry["joint_side_labels"])
        paired = partner >= 0
        centre = side == "center"
        # Center joints mirror onto themselves (X negated); a side joint with
        # no partner has no mirror and would read as asymmetric whatever it
        # does, so it takes no part.
        self.used = np.where(paired | centre)[0]
        self.paired_used = paired[self.used]
        local = {int(j): k for k, j in enumerate(self.used)}
        self.mirror_index = np.array(
            [local[int(partner[j])] if partner[j] >= 0 else local[int(j)] for j in self.used]
        )
        self.side = side[self.used]
        self.ok = bool(entry.get("is_symmetric")) and bool(paired.any())

    def mirror(self, positions):
        mirrored = positions[:, self.mirror_index, :].copy()
        mirrored[..., 0] *= -1.0
        return mirrored


def side_measure(decoded: DecodedClip, symmetry: Symmetry) -> dict:
    """Antisymmetric amplitude / share and the left share of side energy."""
    X = decoded.ric[:, symmetry.used, :]
    X = X - X.mean(axis=0, keepdims=True)
    M = symmetry.mirror(X)
    anti = (X - M) / 2.0
    sym = (X + M) / 2.0
    amp_anti = float(np.sqrt((anti ** 2).sum(-1).mean())) / decoded.L
    amp_sym = float(np.sqrt((sym ** 2).sum(-1).mean())) / decoded.L
    share = amp_anti ** 2 / max(amp_anti ** 2 + amp_sym ** 2, 1e-12)
    velocity = np.diff(X, axis=0)
    energy = (velocity ** 2).sum(axis=(0, 2))
    left = float(energy[(symmetry.side == "left") & symmetry.paired_used].sum())
    right = float(energy[(symmetry.side == "right") & symmetry.paired_used].sum())
    side_energy = left + right
    # Center joints can make the clip asymmetric even when neither side limb
    # moves. With no side energy, the ratio has no direction to report.
    left_share = left / side_energy if side_energy > 1e-12 else 0.5
    return {"amp_anti": amp_anti, "amp_sym": amp_sym, "asym_share": share,
            "left_share": left_share}


def side_word(left_share: float, margin: float):
    if abs(left_share - 0.5) < margin:
        return None
    return "left" if left_share > 0.5 else "right"


def planar_words(vector, diag_share: float) -> list[str]:
    dots = {word: float(axis @ vector) for word, axis in PLANAR_AXES.items()}
    best = max(dots, key=dots.get)
    words = [best]
    for word, dot in dots.items():
        if word != best and dot >= diag_share * dots[best]:
            words.append(word)
    return sorted(words, key=DIRECTION_VOCAB.index)


def heading_measure(decoded: DecodedClip, entry) -> np.ndarray | None:
    """Support-foot travel direction (body lengths / s) in the XZ plane, or None."""
    contacts = [int(j) for j in entry.get("contact_joints", [])]
    if not contacts or decoded.frames < 3:
        return None
    world = decoded.world
    velocity = np.diff(world, axis=0)
    root_velocity = velocity[:, decoded.root_index, :]
    total = np.zeros(2)
    count = 0
    for joint in contacts:
        height = world[:, joint, 1]
        low, high = float(height.min()), float(height.max())
        threshold = low + CONTACT_HEIGHT_SHARE * max(high - low, 1e-9)
        grounded = (height[:-1] <= threshold) & (height[1:] <= threshold)
        if not grounded.any():
            continue
        relative = velocity[grounded][:, joint, :] - root_velocity[grounded]
        total += -relative[:, [0, 2]].sum(axis=0)
        count += int(grounded.sum())
    if count == 0:
        return None
    return total / count / decoded.L * FPS


def jump_measure(decoded: DecodedClip) -> dict:
    root = decoded.world[:, decoded.root_index, :]
    height = root[:, 1]
    low, high = float(height.min()), float(height.max())
    rise = (high - low) / decoded.L
    airborne = height >= low + JUMP_AIR_SHARE * max(high - low, 1e-9)
    velocity = np.diff(root[:, [0, 2]], axis=0)
    in_air = airborne[:-1] & airborne[1:]
    air_disp = velocity[in_air].sum(axis=0) / decoded.L if in_air.any() else np.zeros(2)
    net = (root[-1, [0, 2]] - root[0, [0, 2]]) / decoded.L
    return {"rise": rise, "air_frames": int(in_air.sum()), "air_disp": air_disp, "net": net}


def topple_measure(decoded: DecodedClip) -> dict:
    """Where the body goes as it falls: travel, drop and start clearance.

    The centroid and not the root: a death drops the body onto the ground and a
    rig whose root is a hip barely moves it (BrownBear_Twitching travels 0.00 at
    the root and 0.60 at the centroid). The clip's first tenth against its last
    rather than the endpoints, so one jittering frame cannot name the fall.

    ``drop`` and ``start_clearance`` are what say the fall was a topple at all:
    a body that never descends did not topple, and one that starts in the air
    fell rather than toppled. The floor is the lowest any joint gets in this
    clip, which is what the clip itself offers -- there is no ground plane in
    the tensor.
    """
    n = max(1, decoded.frames // TOPPLE_END_SHARE)
    world = decoded.world
    centroid = world.mean(axis=1)
    travel = ((centroid[-n:][:, [0, 2]].mean(axis=0)
               - centroid[:n][:, [0, 2]].mean(axis=0)) / decoded.L)
    strong, weak = float(np.abs(travel).max()), float(np.abs(travel).min())
    floor = float(world[..., 1].min())
    return {"travel": travel, "distance": float(np.linalg.norm(travel)),
            "tie_share": weak / strong if strong > 1e-9 else 1.0,
            "drop": float(centroid[:n, 1].mean() - centroid[-n:, 1].mean()) / decoded.L,
            "start_clearance": float(world[:n, :, 1].min() - floor) / decoded.L}


def dominant_word(vector) -> str:
    """The single planar word of *vector*'s dominant axis."""
    dots = {word: float(axis @ vector) for word, axis in PLANAR_AXES.items()}
    return max(dots, key=dots.get)


def topple_verdict(measure, clip, args):
    """``(status, words, reason)`` for a death.

    The four ``keep`` cases come first: a death that did not topple onto the
    ground has no planar direction to name, whatever its horizontal travel
    reads (MLS_BattleOwl_Die drifts 2.21 body lengths out of a hover, which is
    more than the median hand-labelled death, and topples toward nothing).
    """
    travel = np.round(measure["travel"], 2).tolist()
    airborne = [word for word in label_words(clip["label"])
                if word in AIRBORNE_DEATH_WORDS]
    if airborne:
        return ("keep", [], f"'{', '.join(airborne)}' on the label: a death in the air "
                            f"falls, and its {measure['distance']:.2f} body lengths in "
                            f"the ground plane are the fall's, not a direction")
    if measure["start_clearance"] > args.topple_start_clearance_max:
        return ("keep", [], f"the body starts {measure['start_clearance']:.2f} body "
                            f"lengths off the ground: this death falls, it does not topple")
    if measure["drop"] < args.topple_min_drop:
        return ("keep", [], f"the body descends {measure['drop']:.2f} body lengths: "
                            f"nothing toppled, it slid or settled where it stood")
    if measure["distance"] < args.topple_min_travel:
        return ("keep", [], f"the body travels {measure['distance']:.2f} body lengths: "
                            f"a collapse in place, aimed nowhere")
    if measure["tie_share"] > TOPPLE_TIE_SHARE:
        return ("review", [], f"body travel {travel} is a diagonal with no dominant "
                              f"axis (off-axis share {measure['tie_share']:.2f})")
    word = dominant_word(measure["travel"])
    return ("write", [word], f"{word}: the body topples {travel} body lengths, "
                             f"descending {measure['drop']:.2f}")


def mirror_pairs(decoded_by_key: dict, symmetry: Symmetry, keys: list[str]) -> dict[str, tuple[str, float]]:
    """``key -> (partner key, distance)`` for clips that are mirror takes of each other."""
    if len(keys) < 2:
        return {}
    signatures = {}
    for key in keys:
        decoded = decoded_by_key[key]
        X = decoded.ric[:, symmetry.used, :]
        X = X - X.mean(axis=0, keepdims=True)
        index = np.linspace(0, decoded.frames - 1, MIRROR_PAIR_FRAMES)
        lo = np.floor(index).astype(int)
        hi = np.minimum(lo + 1, decoded.frames - 1)
        weight = (index - lo)[:, None, None]
        resampled = X[lo] * (1.0 - weight) + X[hi] * weight
        signatures[key] = (resampled / decoded.L, decoded.frames)
    pairs: dict[str, tuple[str, float]] = {}
    for i, a in enumerate(keys):
        Xa, fa = signatures[a]
        norm_a = float(np.linalg.norm(Xa))
        if norm_a < 1e-9:
            continue
        for b in keys[i + 1:]:
            Xb, fb = signatures[b]
            if abs(fa - fb) > 0.1 * max(fa, fb):
                continue
            norm_b = float(np.linalg.norm(Xb))
            if norm_b < 1e-9:
                continue
            distance = float(np.linalg.norm(Xa - symmetry.mirror(Xb))) / np.sqrt(norm_a * norm_b)
            if distance < MIRROR_PAIR_MAX:
                for key, other in ((a, b), (b, a)):
                    if key not in pairs or pairs[key][1] > distance:
                        pairs[key] = (other, distance)
    return pairs


# ── scope ────────────────────────────────────────────────────────────────────

def direction_words_of(label: str) -> list[str]:
    return [word for word in label_words(label) if word in DIRECTION_VOCAB]


def kind_of(clip) -> str | None:
    """Which measurement a row falls under, or None when it is out of scope."""
    words = label_words(clip["label"])
    if not words:
        return None
    group = clip["group"]
    # A turn's side word is the yaw direction, which neither a limb's energy
    # nor the support foot can read; every turn row already carries it.
    if "turn" in words:
        return None
    if group == "transition" and "jump" in words and "land" not in words:
        # A jump is judged for its VERTICAL word too, which no exemption
        # touches, so an exempt jump stays in scope (jump_verdict drops the
        # planar half of the verdict for it).
        return "jump"
    # hurt / getup / idle / rest / stop / draw / sheathe / headbutt / bite are
    # aimed nowhere: whatever side energy or support foot reads off them is
    # incidental, so neither measurement runs and neither calibrates on them.
    if not takes_planar_direction(words):
        return None
    # A death is not delivered with a limb, so its word is not a side: it names
    # where the body ends up, on all four planar axes. Ahead of the group
    # dispatch, which would otherwise hand it to the side measurement.
    if set(words) & set(BODY_TRAVEL_WORDS):
        return "topple"
    if group == "locomotion":
        return "heading"
    return "side"


def heading_slot_empty(label: str) -> bool:
    words = set(label_words(label))
    if words & set(PLANAR_DIRECTIONS) or words & set(VERTICAL_WORDS):
        return False
    return not (words & set(NO_HEADING_WORDS))


# ── the run ──────────────────────────────────────────────────────────────────

def run(args) -> int:
    cond, sources, clips = load_corpus(args.cond_path, None if args.action_group == "all" else args.action_group)
    print(f"[OK] {len(clips)} clip(s) from {args.cond_path} (pending_delete rows dropped)")

    symmetry_by_species: dict[str, Symmetry] = {}
    decoded: dict[tuple[str, str], DecodedClip] = {}

    def decode(clip) -> DecodedClip:
        key = (clip["species"], clip["key"])
        if key not in decoded:
            decoded[key] = DecodedClip(clip["motion_path"], cond[clip["species"]])
        return decoded[key]

    def symmetry(clip) -> Symmetry:
        species = clip["species"]
        if species not in symmetry_by_species:
            symmetry_by_species[species] = Symmetry(cond[species])
        return symmetry_by_species[species]

    # ── calibration ──
    side_cal = {"decided": 0, "agree": 0, "undecided": 0, "disagree": []}
    heading_cal = {"decided": 0, "agree": 0, "undecided": 0, "disagree": []}
    jump_cal = {"decided": 0, "agree": 0, "undecided": 0, "disagree": []}
    topple_cal = {"decided": 0, "agree": 0, "undecided": 0, "disagree": []}
    side_scope, heading_scope, jump_scope, topple_scope = [], [], [], []
    for clip in clips:
        kind = kind_of(clip)
        if kind is None:
            continue
        words = label_words(clip["label"])
        present = direction_words_of(clip["label"])
        if kind == "side":
            if not symmetry(clip).ok:
                if not present:
                    side_scope.append((clip, "keep", "no mirror pairs on this rig"))
                continue
            sides = [w for w in present if w in SIDE_WORDS]
            if present and not sides:
                continue          # a vertical word only: slot spelled, not our case
            measure = side_measure(decode(clip), symmetry(clip))
            if sides:
                if measure["amp_anti"] < args.side_amp_floor:
                    side_cal["undecided"] += 1
                    continue
                verdict = side_word(measure["left_share"], args.side_ratio_margin)
                if verdict is None:
                    side_cal["undecided"] += 1
                    continue
                side_cal["decided"] += 1
                if verdict == sides[0]:
                    side_cal["agree"] += 1
                else:
                    side_cal["disagree"].append((clip, verdict, measure))
            else:
                side_scope.append((clip, measure, None))
        elif kind == "heading":
            if head_of(clip["label"]) not in GROUND_GAIT_HEADS:
                if heading_slot_empty(clip["label"]):
                    heading_scope.append((clip, None))
                continue
            heading = heading_measure(decode(clip), cond[clip["species"]])
            planar = [w for w in present if w in PLANAR_DIRECTIONS]
            if planar:
                if heading is None or float(np.linalg.norm(heading)) < args.heading_min_speed:
                    heading_cal["undecided"] += 1
                    continue
                verdict = planar_words(heading, HEADING_DIAG_SHARE)
                heading_cal["decided"] += 1
                if set(verdict) == set(planar):
                    heading_cal["agree"] += 1
                else:
                    heading_cal["disagree"].append((clip, verdict, heading))
            elif heading_slot_empty(clip["label"]):
                heading_scope.append((clip, heading))
        elif kind == "topple":
            measure = topple_measure(decode(clip))
            verdict = topple_verdict(measure, clip, args)
            planar = [w for w in present if w in PLANAR_DIRECTIONS]
            if planar:
                if verdict[0] != "write":
                    topple_cal["undecided"] += 1
                    continue
                topple_cal["decided"] += 1
                if set(verdict[1]) == set(planar):
                    topple_cal["agree"] += 1
                else:
                    topple_cal["disagree"].append((clip, verdict[1], measure))
            elif not present:
                topple_scope.append((clip, measure, verdict))
        else:  # jump
            measure = jump_measure(decode(clip))
            verdict = jump_verdict(measure, clip, args)
            if present:
                truth = [w for w in present if w in PLANAR_DIRECTIONS or w == "up"]
                if not truth:
                    continue
                if verdict[0] != "write":
                    jump_cal["undecided"] += 1
                    continue
                jump_cal["decided"] += 1
                if set(verdict[1]) == set(truth):
                    jump_cal["agree"] += 1
                else:
                    jump_cal["disagree"].append((clip, verdict[1], measure))
            else:
                jump_scope.append((clip, measure, verdict))

    def rate(cal):
        return cal["agree"] / cal["decided"] if cal["decided"] else 0.0

    side_open = rate(side_cal) >= args.side_gate and side_cal["decided"] > 0
    heading_open = rate(heading_cal) >= args.heading_gate and heading_cal["decided"] > 0
    jump_open = rate(jump_cal) >= args.jump_gate and jump_cal["decided"] > 0
    topple_open = rate(topple_cal) >= args.topple_gate and topple_cal["decided"] > 0

    print()
    print("=" * 72)
    print("  CALIBRATION on rows that already carry a direction word")
    print("=" * 72)
    for name, cal, gate, is_open in (("side (left/right energy share)", side_cal, args.side_gate, side_open),
                                     ("heading (support foot)", heading_cal, args.heading_gate, heading_open),
                                     ("jump (airborne root travel)", jump_cal, args.jump_gate, jump_open),
                                     ("topple (centroid travel)", topple_cal, args.topple_gate, topple_open)):
        print(f"  {name}: {cal['agree']}/{cal['decided']} agree ({rate(cal):.1%}), "
              f"{cal['undecided']} undecided; gate {gate:.0%} -> {'OPEN' if is_open else 'CLOSED (list only)'}")
        for clip, verdict, measure in cal["disagree"]:
            detail = ""
            if isinstance(measure, dict) and "left_share" in measure:
                detail = f"left_share={measure['left_share']:.2f}"
            elif isinstance(measure, dict) and "air_disp" in measure:
                detail = f"air_disp={np.round(measure['air_disp'], 2).tolist()} net={np.round(measure['net'], 2).tolist()}"
            elif isinstance(measure, dict) and "travel" in measure:
                detail = f"travel={np.round(measure['travel'], 2).tolist()}"
            elif measure is not None:
                detail = f"h={np.round(measure, 2).tolist()}"
            print(f"      disagree: {clip['clip']:48s} label={clip['label']!r} measured={verdict} {detail}")

    # ── proposals ──
    proposals: list[Proposal] = []

    # side: mirror pairs first, per (species, group)
    side_by_bucket: dict[tuple[str, str], list] = defaultdict(list)
    for clip, measure, _ in side_scope:
        side_by_bucket[(clip["species"], clip["group"])].append((clip, measure))
    labelled_side = {}
    for clip in clips:
        if kind_of(clip) == "side":
            sides = [w for w in direction_words_of(clip["label"]) if w in SIDE_WORDS]
            if sides and symmetry(clip).ok:
                labelled_side[(clip["species"], clip["group"], clip["key"])] = (clip, sides[0])
    for (species, group), members in side_by_bucket.items():
        sym = symmetry_by_species.get(species)
        empty_keys = [clip["key"] for clip, measure in members if measure != "keep"]
        pool = {clip["key"]: clip for clip, measure in members if measure != "keep"}
        for (s, g, key), (clip, _side) in labelled_side.items():
            if s == species and g == group:
                pool[key] = clip
        pairs = {}
        if sym is not None and sym.ok and len(pool) >= 2:
            by_key = {key: decode(clip) for key, clip in pool.items()}
            pairs = mirror_pairs(by_key, sym, sorted(pool))
        for clip, measure in members:
            if measure == "keep":
                proposals.append(Proposal(clip, "keep", "no mirror pairs on this rig"))
                continue
            metrics = {k: round(v, 3) for k, v in measure.items()}
            partner = pairs.get(clip["key"])
            partner_clip = partner_side = None
            if partner is not None:
                partner_key, distance = partner
                partner_clip = pool[partner_key]["clip"]
                metrics["mirror_distance"] = round(distance, 3)
                labelled = labelled_side.get((species, group, partner_key))
                if labelled is not None:
                    partner_side = labelled[1]
            evidence = {"measure": "side_energy", **metrics}
            if partner_clip:
                evidence["mirror_of"] = partner_clip
            if measure["amp_anti"] < args.side_amp_floor:
                proposals.append(Proposal(clip, "keep", "still (antisymmetric amplitude under floor)", metrics=metrics))
                continue
            if measure["asym_share"] < args.side_sym_max:
                proposals.append(Proposal(clip, "keep", "symmetric", metrics=metrics))
                continue
            verdict = side_word(measure["left_share"], args.side_ratio_margin)
            if partner_side is not None:
                opposite = "right" if partner_side == "left" else "left"
                if verdict is None:
                    verdict = opposite
                    evidence["decided_by"] = "mirror of a labelled clip"
                elif verdict != opposite:
                    proposals.append(Proposal(
                        clip, "review",
                        f"side energy says {verdict} but the clip mirrors {partner_clip} ({partner_side})",
                        metrics=metrics, partner=partner_clip))
                    continue
            if measure["asym_share"] < args.side_asym_min and "decided_by" not in evidence:
                proposals.append(Proposal(clip, "review", "weakly asymmetric", metrics=metrics, partner=partner_clip))
                continue
            if verdict is None:
                proposals.append(Proposal(
                    clip, "review",
                    "asymmetric but no dominant side (alternating / rotating / centred)",
                    metrics=metrics, partner=partner_clip))
                continue
            proposed = spell_with(clip["label"], [verdict])
            status = "write" if side_open else "review"
            reason = (f"{verdict}: left share of side-joint energy {measure['left_share']:.2f}"
                      if "decided_by" not in evidence else
                      f"{verdict}: mirror take of {partner_clip} ({partner_side})")
            proposals.append(Proposal(clip, status, reason, proposed, evidence, metrics, partner_clip))
        # two empty mirror takes must come out on opposite sides
        by_key = {p.clip["key"]: p for p in proposals if p.clip["species"] == species and p.clip["group"] == group}
        for key, (partner_key, _distance) in pairs.items():
            a, b = by_key.get(key), by_key.get(partner_key)
            if a is None or b is None or a.status != "write" or b.status != "write":
                continue
            if direction_words_of(a.proposed) == direction_words_of(b.proposed):
                for p in (a, b):
                    p.status = "review"
                    p.reason = f"mirror takes {a.clip['clip']} / {b.clip['clip']} both read {direction_words_of(a.proposed)}"

    # heading
    for clip, heading in heading_scope:
        if head_of(clip["label"]) not in GROUND_GAIT_HEADS:
            proposals.append(Proposal(clip, "review", "no ground contact to read a heading from (swim / fly / roll)"))
            continue
        if heading is None:
            proposals.append(Proposal(clip, "review", "no contact frames found"))
            continue
        speed = float(np.linalg.norm(heading))
        metrics = {"heading_x": round(float(heading[0]), 3), "heading_z": round(float(heading[1]), 3),
                   "speed": round(speed, 3)}
        if speed < args.heading_min_speed:
            proposals.append(Proposal(clip, "review", f"support-foot speed {speed:.2f} body lengths/s under {args.heading_min_speed}", metrics=metrics))
            continue
        words = planar_words(heading, HEADING_DIAG_SHARE)
        proposals.append(Proposal(
            clip, "write" if heading_open else "review",
            f"{', '.join(words)}: support-foot travel {np.round(heading, 2).tolist()} body lengths/s",
            spell_with(clip["label"], words), {"measure": "support_foot", **metrics}, metrics))

    # jump
    for clip, measure, verdict in jump_scope:
        metrics = {"rise": round(measure["rise"], 3), "air_frames": measure["air_frames"],
                   "air_x": round(float(measure["air_disp"][0]), 3), "air_z": round(float(measure["air_disp"][1]), 3),
                   "net_x": round(float(measure["net"][0]), 3), "net_z": round(float(measure["net"][1]), 3),
                   "is_loop": str(clip.get("is_loop"))}
        status, words, reason = verdict
        if status == "write":
            proposals.append(Proposal(
                clip, "write" if jump_open else "review", reason,
                spell_with(clip["label"], words), {"measure": "airborne_root", **metrics}, metrics))
        else:
            proposals.append(Proposal(clip, "review", reason, metrics=metrics))

    # topple
    for clip, measure, verdict in topple_scope:
        metrics = {"travel_x": round(float(measure["travel"][0]), 3),
                   "travel_z": round(float(measure["travel"][1]), 3),
                   "distance": round(measure["distance"], 3),
                   "tie_share": round(measure["tie_share"], 3),
                   "drop": round(measure["drop"], 3),
                   "start_clearance": round(measure["start_clearance"], 3)}
        status, words, reason = verdict
        if status == "write":
            proposals.append(Proposal(
                clip, "write" if topple_open else "review", reason,
                spell_with(clip["label"], words), {"measure": "body_travel", **metrics}, metrics))
        else:
            proposals.append(Proposal(clip, status, reason, metrics=metrics))

    # ── report ──
    # Every cross-check above (mirror pairing, the two-takes-one-side rule) has
    # already run with the reviewed rows in place; they drop out here, as the
    # last step before anything is counted, printed or written.
    proposals = drop_reviewed(proposals, args.include_reviewed)

    print()
    print("=" * 72)
    print("  PROPOSALS")
    print("=" * 72)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for p in proposals:
        counts[(p.clip["group"], p.status)] += 1
    for key in sorted(counts):
        print(f"  {key[0]:11s} {key[1]:7s} {counts[key]}")
    if args.verbose:
        for p in sorted(proposals, key=lambda p: (p.status, p.clip["species"], p.clip["clip"])):
            if p.status == "keep":
                continue
            print(f"    {p.status:6s} {p.clip['clip']:48s} {p.label!r} -> {p.proposed!r}  {p.reason}")

    if args.report:
        count = write_csv(args.report, [p for p in proposals if p.status != "keep" or args.verbose])
        print(f"\n[OK] wrote {count} row(s) to {args.report}")
    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps([p.as_row() for p in proposals], ensure_ascii=False, indent=1), encoding="utf-8")

    writes = [p for p in proposals if p.status == "write"]
    if not args.apply:
        print(f"\n[OK] dry run: {len(writes)} row(s) would be written; pass --apply to write them")
        return 0
    if not writes:
        print("\n[OK] nothing to write")
        return 0
    check_head_order(sources, writes)
    written = apply_proposals(sources, writes, slot="direction", slot_vocab=set(DIRECTION_VOCAB),
                              include_reviewed=args.include_reviewed)
    for root, count in written.items():
        print(f"[OK] {root}: {count} row(s) written (reviewed:false + autofill)")
    return 0


def jump_verdict(measure, clip, args):
    """``(status, words, reason)`` for a transition jump."""
    if measure["rise"] < JUMP_MIN_RISE:
        return ("review", [], f"root rises only {measure['rise']:.2f} body lengths: not a jump?")
    disp = measure["air_disp"]
    magnitude = float(np.linalg.norm(disp))
    net = float(np.linalg.norm(measure["net"]))
    is_loop = clip.get("is_loop")
    if magnitude <= args.jump_up_max:
        if is_loop is False and net >= JUMP_NET_MIN:
            return ("review", [], f"airborne root travel {magnitude:.2f} says up, but net root travel {net:.2f} says otherwise")
        return ("write", ["up"], f"up: airborne root travel {magnitude:.2f} body lengths, rise {measure['rise']:.2f}")
    if magnitude >= args.jump_planar_min:
        words = planar_words(disp, HEADING_DIAG_SHARE)
        if not takes_planar_direction(label_words(clip["label"])):
            return ("review", [], f"airborne root travel {np.round(disp, 2).tolist()} reads "
                                  f"{', '.join(words)}, but this action takes no planar direction")
        return ("write", words, f"{', '.join(words)}: airborne root travel {np.round(disp, 2).tolist()} body lengths")
    return ("review", [], f"airborne root travel {magnitude:.2f} between up ({args.jump_up_max}) and planar ({args.jump_planar_min})")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Propose missing direction words from the motions (dry run by default).",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--cond-path", "--cond_path", dest="cond_path", default=None,
                        help="cond.npy naming the species and their dataset roots (default: the standard dataset cond).")
    parser.add_argument("--action-group", "--action_group", dest="action_group", default="all",
                        choices=["all", "locomotion", "stationary", "transition"])
    parser.add_argument("--apply", action="store_true", help="Write the 'write' rows (default: dry run).")
    parser.add_argument("--dry-run", dest="apply", action="store_false", help="Judge and report only (the default).")
    add_reviewed_flag(parser)
    parser.add_argument("--report", default="", help="CSV of every judged row.")
    parser.add_argument("--json", dest="json_path", default="", help="Also dump the proposals as JSON.")
    parser.add_argument("--verbose", action="store_true", help="Print every proposal and include 'keep' rows in the CSV.")
    parser.add_argument("--side-gate", type=float, default=0.90)
    parser.add_argument("--heading-gate", type=float, default=0.95)
    parser.add_argument("--jump-gate", type=float, default=0.90)
    parser.add_argument("--topple-gate", type=float, default=0.90)
    parser.add_argument("--side-amp-floor", type=float, default=SIDE_AMP_FLOOR)
    parser.add_argument("--side-sym-max", type=float, default=SIDE_SYM_MAX)
    parser.add_argument("--side-asym-min", type=float, default=SIDE_ASYM_MIN)
    parser.add_argument("--side-ratio-margin", type=float, default=SIDE_RATIO_MARGIN)
    parser.add_argument("--heading-min-speed", type=float, default=HEADING_MIN_SPEED)
    parser.add_argument("--jump-up-max", type=float, default=JUMP_UP_MAX)
    parser.add_argument("--jump-planar-min", type=float, default=JUMP_PLANAR_MIN)
    parser.add_argument("--topple-min-travel", type=float, default=TOPPLE_MIN_TRAVEL)
    parser.add_argument("--topple-min-drop", type=float, default=TOPPLE_MIN_DROP)
    parser.add_argument("--topple-start-clearance-max", type=float,
                        default=TOPPLE_START_CLEARANCE_MAX)
    args = parser.parse_args(argv)
    if args.cond_path is None:
        from data_loaders.truebones.truebones_utils.get_opt import DEFAULT_COND_PATH
        args.cond_path = DEFAULT_COND_PATH
    return args


if __name__ == "__main__":
    sys.exit(run(parse_args()))
