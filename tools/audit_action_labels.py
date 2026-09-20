#!/usr/bin/env python3
"""
Cross-clip audit of ``action_labels.jsonl``.

``_validate_action_label_entry`` checks one row at a time (vocabulary, canonical
order; repeats are already stripped by normalize_action_label). What it cannot
see is a label that is wrong *relative to another row* -- a mirror pair whose two
halves disagree -- or wrong relative to its own action group. Those are what this
tool reports.

Three rules, each independently selectable with ``--rules``. All three are
DETERMINISTIC TEXT RULES: each reads the label spelling (and, for R3, the clip
name) and fires only where the spelling is provably wrong. None of them measures
the motion, guesses at a threshold, or reports a case that "looks unusual" --
a finding here is a defect, not a candidate.

  R3  mirror consistency.  Clips whose names differ only by Left/Right (or a
      trailing L/R) must carry labels that are mirrors of each other.  That is
      the whole rule, and it is deliberately the only thing a name can settle:
      the verdict is SYMMETRIC -- it says the pair disagrees with itself, never
      which half is wrong and never which side word belongs on which name.
      NOTHING HERE READS A DIRECTION OFF A CLIP NAME.  Two checks that used to
      are gone, because both were wrong in this corpus:
        * "the side words are crossed" (the clip named Left says 'right').
          ``MB_Unka_DeathLeft`` really does fall to the character's right, so
          its 'die, right' is correct and the NAME is what lies.  A
          crossed-looking pair is a valid mirror of itself; this rule passes it.
        * "each side must carry its own side word".  Whether two identically
          spelled clips are mirror takes at all -- and which way each one goes
          -- is a fact about the MOTION.  ``tools/prefill_direction_words.py``
          settles it there (mirror detection on the positions, then side
          energy); the console here counts the identically spelled pairs and
          points at that tool instead of calling them violations.

  R4  direction spelling.  Every ``locomotion`` label must name a heading:
      a planar one, or a vertical one for a clip that travels out of the ground
      plane ("fly, up", "jump, fall").  A label carrying a word with no planar
      travel to name -- ``hover`` (held in place) -- is exempt, because there
      is no heading to spell.  ``jump`` is not exempt: the direction slot is
      dropped out at random in training, so a bare ``jump`` is the marginal over
      jumps and a vertical one is spelled ``jump, up``; a locomotion
      ``run, jump`` names where the run goes like any run.  The actions with no
      planar direction at all (``NO_PLANAR_DIRECTION_WORDS``) are exempt here
      too.  Corpus-wide a plain forward walk is spelled "walk" 87 times and
      "walk, forward" 30 times, which makes "forward" noise in a condition
      shared across every species.

  R5  gait-word conflicts.  ``die`` + ``fall``, ``die`` + ``idle``, ``walk`` +
      ``run`` (should be ``walk, trot``), ``glide`` + ``flap``, and ``slow`` +
      ``fast`` -- each names one axis twice.  The last three are gated on
      their replacement word being in ACTION_VOCAB; all five are active today.

Every finding carries the paths of the review GIFs it concerns, and none of them
prescribes a fix.  A clip NAME is the weakest evidence there is about what a clip
does -- it is good enough to pair two clips up and to notice that the pair
disagrees with itself, and not good enough to decide which half is wrong, which
is why no rule here proposes a corrected label.  What a clip actually does is
settled from the render (``dataset/review/relabel_actions_llm.py`` feeds those
same GIFs to the vision model) or measured off the motion
(``tools/prefill_direction_words.py``).

Output is an HTML review page (``tools/audit_report_html.py``), because acting
on a finding means watching a GIF and the console cannot show one.  Every
finding becomes a panel holding its clips' review GIFs, their current labels and
a field for the corrected one; the page validates what is typed against the same
contract ``_validate_action_label_entry`` enforces, and "复制修复指令" puts the
whole verdict on the clipboard as an instruction naming the sidecar, the clip,
the old label and the new one -- paste that into an LLM and it performs the
edit.  Nothing here writes to ``action_labels.jsonl``.  The console keeps the
per-rule counts (that is what ``--strict`` gates on) and prints each finding in
full only under ``--text`` or ``--no-html``.

Open the page from disk: it reaches its GIFs through absolute ``file://`` URIs,
which an ``http://`` page served by ``dataset/review/serve.py`` may not load.

Usage:
    python tools/audit_action_labels.py
    python tools/audit_action_labels.py --cond-path dataset/merged/cond.npy
    python tools/audit_action_labels.py --action-group all --open
    python tools/audit_action_labels.py --action-group locomotion --strict
    python tools/audit_action_labels.py --rules R3,R4 --no-html --json audit.json

Exit code is 1 under ``--strict`` when any violation is found, so this can gate
a preprocessing run or CI.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ANYTOP_DIR = Path(__file__).resolve().parent.parent
_PARENT_DIR = ANYTOP_DIR.parent
sys.path.insert(0, str(_PARENT_DIR))
sys.path.insert(0, str(ANYTOP_DIR))

from data_loaders.truebones.truebones_utils.cond_schema import load_cond  # noqa: E402
from data_loaders.truebones.truebones_utils.dataset_sources import (  # noqa: E402
    sources_from_cond,
)
from data_loaders.truebones.truebones_utils.param_utils import (  # noqa: E402
    ACTION_LABELS_FILE,
)
from data_loaders.truebones.truebones_utils.motion_labels import (  # noqa: E402
    ACTION_GROUPS,
    ACTION_VOCAB,
    DIRECTION_VOCAB,
    clip_key,
    load_action_labels,
)
from tools.action_label_sidecar import read_action_label_rows  # noqa: E402
from tools.audit_report_html import (  # noqa: E402
    DEFAULT_REPORT_PATH,
    write_html_report,
)

ALL_RULES = ("R3", "R4", "R5")

# Direction words that name travel in the ground plane.
PLANAR_DIRECTIONS = ("forward", "backward", "left", "right")

# ... and words that name travel out of it. A dive, a jump-fall or a vertical
# take-off has no planar heading to spell, so R4 must not demand one: doing so
# turned ~40 correct vertical labels into violations and drowned the handful of
# real ones. ``fall`` and ``dive`` are ACTION_VOCAB words, not directions, but
# they name the same vertical axis and settle the same question.
VERTICAL_WORDS = ("up", "down", "dive", "fall")

# Words that name an action with no planar heading to spell: ``hover`` holds
# the body in place, so R4 must not demand a direction of it the way it does
# of a walk or a run.  Distinct from VERTICAL_WORDS (which name travel OUT of
# the plane): this names the absence of a planar component.  Matched anywhere
# in the label, not just as the head.  NOT the same list as
# NO_PLANAR_DIRECTION_WORDS below, and R4-only: a hovering strike IS aimed
# somewhere ("attack, hover, left, swat"), it just has no heading of travel to
# spell.  ``jump`` used to be here; since the direction slot has a training
# dropout an empty slot is the marginal only, so a vertical jump spells
# ``jump, up`` and a running jump spells its heading
# (tools/prefill_direction_words.py fills both from the motion).
NO_HEADING_WORDS = ("hover",)

# Actions with NO PLANAR DIRECTION TO NAME, whatever their group.  The planar
# words (left / right / forward / backward) say which way the action is aimed or
# travels; these actions are not aimed anywhere:
#   hurt / getup / stop / rest / idle  -- a reaction or a body state, not a
#       strike: whichever way the body happens to lean is incidental, and a
#       measurement that reads a side off it spells noise into a condition
#       shared by every species;
#   draw / sheathe  -- the side is the scabbard's, not the action's;
#   headbutt / bite  -- delivered with the head, which is a CENTER joint; a
#       side word on it is read off the body's turn, not off the strike.
# So no rule may demand a planar word of them (R3's "each side carries its own
# side word", R4's heading) and prefill_direction_words.py never proposes one.
# Unlike NO_HEADING_WORDS this is not an R4 knob: it is a corpus-wide rule about
# what these actions can spell at all, and tests/test_direction_exemption.py
# checks the sidecars against it.
# The VERTICAL axis is untouched: "idle, up, aim, bow" aims upward and keeps
# its word -- this names the absence of a PLANAR component only, exactly like
# NO_HEADING_WORDS above.  Matched anywhere in the label, not just as the head
# ("idle, right, look" -> "idle, look").  The LLM annotator's "reset" (settle
# into a neutral start pose, dataset/review/relabel_actions_llm.py) is the same
# case and reaches the sidecar as "rest".
NO_PLANAR_DIRECTION_WORDS = (
    "hurt", "getup", "idle", "rest", "stop", "draw", "sheathe", "headbutt", "bite",
)

_unknown_directionless = set(NO_PLANAR_DIRECTION_WORDS) - set(ACTION_VOCAB)
if _unknown_directionless:
    raise RuntimeError(
        "NO_PLANAR_DIRECTION_WORDS names %s, which ACTION_VOCAB does not have: a "
        "word nothing can spell exempts nothing" % ", ".join(sorted(_unknown_directionless))
    )


def takes_planar_direction(words) -> bool:
    """False when a label's action has no planar direction to name (see above)."""
    return not (set(words) & set(NO_PLANAR_DIRECTION_WORDS))


# A renamed direction word would otherwise turn R4 into "every locomotion clip
# is a violation" without anything saying why.
_missing_directions = set(PLANAR_DIRECTIONS) - set(DIRECTION_VOCAB)
if _missing_directions:
    raise RuntimeError(
        "DIRECTION_VOCAB no longer contains %s; update PLANAR_DIRECTIONS in this "
        "tool to match motion_labels.py" % ", ".join(sorted(_missing_directions))
    )

# Trailing side markers on a clip stem, longest first so "Left" wins over "L".
_SIDE_SUFFIXES = (
    ("left", "L"),
    ("right", "R"),
)


# ─────────────────────────────── label helpers ──────────────────────────────
def label_words(label: str) -> list[str]:
    """The label's words, in stored order. Empty label -> empty list."""
    return [word.strip() for word in str(label or "").split(",") if word.strip()]


def mirror_label(label: str) -> str:
    """*label* with left and right swapped, everything else untouched."""
    swap = {"left": "right", "right": "left"}
    return ", ".join(swap.get(word, word) for word in label_words(label))


def clip_stem(clip_name: str) -> str:
    """``MB_Unka_FlyTurnLeft.npy`` -> ``MB_Unka_FlyTurnLeft``.

    Nothing else is stripped. Preprocessing is 1:1 with the source files, so a
    trailing ``_<i>`` is part of the clip's own name (a numbered variant take),
    not a slice index -- removing it would collapse two different clips onto one
    pairing key in R3.
    """
    return clip_name.rsplit(".", 1)[0]


def clip_side(clip_name: str):
    """``'L'``, ``'R'`` or ``None`` for the side this clip's name names.

    Only a *trailing* marker counts. ``FlyLeftWing`` is a body part, and
    ``LeftFoot`` at the front is not a heading either; both would be false
    positives for a substring test.
    """
    stem = clip_stem(clip_name)
    for word, side in _SIDE_SUFFIXES:
        if re.search(r"(?i)%s$" % word, stem):
            return side
    if re.search(r"(?<=[a-z])L$", stem):
        return "L"
    if re.search(r"(?<=[a-z])R$", stem):
        return "R"
    return None


def clip_side_base(clip_name: str) -> str:
    """The clip stem with its trailing side marker removed."""
    stem = clip_stem(clip_name)
    for word, _side in _SIDE_SUFFIXES:
        stripped = re.sub(r"(?i)%s$" % word, "", stem)
        if stripped != stem:
            return stripped
    return re.sub(r"(?<=[a-z])[LR]$", "", stem)


# ─────────────────────────────── clip loading ───────────────────────────────
def collect_clips(cond_dict, sources, action_group=None, verbose=False):
    """``[{clip, species, group, label, motion_path}, ...]`` for the whole corpus.

    Species membership follows the training loader exactly (``MotionDataset``):
    a clip belongs to the species whose ``species_name`` prefixes it followed by
    an underscore.  Matching that rule matters -- an audit that grouped clips
    differently from training would report on labels the model never sees.

    Also what the two prefill tools walk the corpus with
    (``tools/prefill_common.load_corpus``), so a clip this audit does not judge
    is a clip they do not write to either.
    """
    species_by_namespace = defaultdict(list)
    for object_key, entry in cond_dict.items():
        species_by_namespace[entry["dataset_namespace"]].append(
            (object_key, str(entry["species_name"]))
        )

    clips = []
    for source in sources:
        source_species = species_by_namespace.get(source.namespace)
        if not source_species:
            continue
        labels = load_action_labels(source.root)
        # A row marked pending_delete is a clip on its way out (the review UI
        # retires it on the next cleanup); load_action_labels keeps it -- the
        # trainer still sees it until then -- but an audit that reported it
        # would be asking for a fix to a clip nobody is keeping.
        retiring = {
            clip_key(row.get("clip", ""))
            for row in read_action_label_rows(source.root)
            if row.get("pending_delete")
        }
        if retiring:
            labels = {clip: entry for clip, entry in labels.items()
                      if clip not in retiring}
        motion_dir = Path(source.motion_dir)
        if not motion_dir.is_dir():
            raise FileNotFoundError(
                f"motions directory not found for namespace '{source.namespace}': "
                f"{motion_dir}"
            )
        available = {path.name for path in motion_dir.glob("*.npy")}
        # The review render is the only thing that can settle what a clip really
        # does: a clip's NAME is the weakest hint there is (see check_r3), so a
        # finding has to hand the operator the picture, not just the name.
        gif_dir = Path(source.root) / "review" / "gif"
        # The HTML report has to name the file a fix is written back to, and
        # offer the .bvh the way the review front-end does; both are per source.
        labels_path = Path(source.root) / ACTION_LABELS_FILE
        bvh_dir = Path(source.root) / "bvhs"

        # Longest species name first so 'Dog-2_Walk' is not claimed by 'Dog'.
        ordered = sorted(source_species, key=lambda pair: -len(pair[1]))
        claimed = set()
        for object_key, species_name in ordered:
            prefix = f"{species_name}_"
            for name in sorted(available):
                if name in claimed or not name.startswith(prefix):
                    continue
                # the sidecar is keyed by the extension-less clip name
                entry = labels.get(name[:-4])
                if entry is None:
                    if verbose:
                        print(f"  [skip] {name}: no action_labels.jsonl row")
                    continue
                if action_group and entry["action_group"] != action_group:
                    continue
                claimed.add(name)
                gif_path = gif_dir / (name[:-4] + ".gif")
                bvh_path = bvh_dir / (name[:-4] + ".bvh")
                clips.append({
                    "clip": name,
                    "species": object_key,
                    "group": entry["action_group"],
                    "label": entry["action_label"],
                    "motion_path": str(motion_dir / name),
                    "gif_path": str(gif_path) if gif_path.is_file() else "",
                    "labels_path": str(labels_path),
                    # Same scheme as dataset/review/serve.py, so a clip name in
                    # the report opens in the BVH viewer on click.
                    "bvhview": (f"bvhview://open?--reuse&url={bvh_path.as_uri()}"
                                if bvh_path.is_file() else ""),
                })
    return clips


# ───────────────────────────────── the rules ────────────────────────────────
def check_r3(clips):
    """Mirror pairs whose labels are not mirrors of each other.

    Name-paired, but the verdict never reads a direction off a name: it is
    symmetric under swapping the two halves, so it can say the pair disagrees
    and nothing about which half, or which side word belongs where. See the
    module docstring for the two checks that broke that and were removed.
    """
    findings = []
    stats = {"pairs_checked": 0, "same_label": 0, "same_label_aimed": 0}
    groups = defaultdict(dict)
    for clip in clips:
        side = clip_side(clip["clip"])
        if side is None:
            continue
        key = (clip["species"], clip["group"], clip_side_base(clip["clip"]))
        # Two clips on the same side under one base name (e.g. WalkLeft and
        # WalkTurnLeft collapsing) would make the pairing ambiguous; keep the
        # first rather than report a pair the names did not actually settle.
        groups[key].setdefault(side, clip)

    for (species, group, base), sides in sorted(groups.items()):
        if "L" not in sides or "R" not in sides:
            continue
        stats["pairs_checked"] += 1
        left, right = sides["L"], sides["R"]
        left_label, right_label = left["label"], right["label"]
        left_words = set(label_words(left_label))
        problems = []
        # A stable code beside each sentence: the HTML report words these in
        # Chinese, and keying that off the English prose would silently fall
        # back to the prose the day someone rewords it.
        codes = []

        # Two halves spelled the SAME are mirrors of each other, so the rule
        # passes them. For an action with no planar direction to name (hurt,
        # idle, bite ...) that is the correct spelling and the end of it; for an
        # aimed one it may instead be a lost L/R axis -- but only the motion can
        # say whether the two clips are mirror takes at all, so this is counted
        # for the console rather than reported as a violation.
        if left_label == right_label:
            stats["same_label"] += 1
            if takes_planar_direction(left_words):
                stats["same_label_aimed"] += 1

        # The one thing the names can settle: the pair disagrees with itself.
        # Which half is wrong is not decided here -- and neither is the case
        # where the two side words look swapped relative to the names, because
        # a name is not evidence about the motion (MB_Unka_DeathLeft falls to
        # the character's right, and its label says so).
        if mirror_label(left_label) != right_label:
            problems.append("labels are not mirrors of each other")
            codes.append("not_mirror")

        if not problems:
            continue
        findings.append({
            "rule": "R3",
            "species": species,
            "action_group": group,
            "base": base,
            "problems": problems,
            "problem_codes": codes,
            "left": {"clip": left["clip"], "label": left_label,
                     "gif": left.get("gif_path", "")},
            "right": {"clip": right["clip"], "label": right_label,
                      "gif": right.get("gif_path", "")},
        })
    return findings, stats


def check_r4(clips):
    """``locomotion`` labels that name neither a planar nor a vertical heading."""
    # Two exemptions, both constants: a word with no heading of travel to spell
    # (hover), and an action with no planar direction at all (a locomotion row
    # carries none of the latter today; this keeps R4 from inventing one if a
    # "stop, ..." ever lands here).
    no_heading = set(NO_HEADING_WORDS) | set(NO_PLANAR_DIRECTION_WORDS)
    findings = []
    stats = {"locomotion_clips": 0, "vertical_clips": 0, "no_heading_clips": 0}
    planar = set(PLANAR_DIRECTIONS)
    vertical = set(VERTICAL_WORDS)
    for clip in sorted(clips, key=lambda item: (item["species"], item["clip"])):
        if clip["group"] != "locomotion":
            continue
        stats["locomotion_clips"] += 1
        words = set(label_words(clip["label"]))
        if planar & words:
            continue
        if vertical & words:
            # 'fly, up', 'swim, down', 'jump, fall', 'fly, dive, down': the
            # heading is named, it just is not in the ground plane.
            stats["vertical_clips"] += 1
            continue
        if no_heading & words:
            # 'hover', 'hover, slow', 'stop': the action has no planar travel
            # to name, so a direction is not a defect.
            stats["no_heading_clips"] += 1
            continue
        findings.append({
            "rule": "R4",
            "species": clip["species"],
            "clip": clip["clip"],
            "label": clip["label"],
            "gif": clip.get("gif_path", ""),
            "problem": "locomotion label names no direction, planar or vertical",
        })
    return findings, stats


def check_r5(clips):
    """Word pairs that contradict each other, or say the same thing twice."""
    vocab = set(ACTION_VOCAB)
    # A pair is a violation when both words are present. Two kinds live here:
    # an axis named twice (walk+run, glide+flap, slow+fast) and a word that
    # adds nothing beside its partner (fall beside die held on 93% of deaths,
    # so it separated nothing while spending part of a mean-pooled budget).
    conflicts = [
        (("die", "fall"), "a death already falls; write 'die' alone"),
        (("die", "idle"), "'idle' is a live motionless stance and contradicts a death"),
    ]
    # The rest are checked only once their replacement word exists, so they stay
    # silent until the vocabulary is extended.
    if "trot" in vocab:
        conflicts.append((("walk", "run"), "use 'walk, trot' for the gait between them"))
    if "glide" in vocab:
        conflicts.append((("glide", "flap"), "powered and unpowered flight are exclusive"))
    if "slow" in vocab:
        conflicts.append((("slow", "fast"), "the speed axis takes one direction, not both"))
    findings = []
    stats = {"checks_active": len(conflicts)}
    if not conflicts:
        return findings, stats
    for clip in sorted(clips, key=lambda item: (item["species"], item["clip"])):
        words = set(label_words(clip["label"]))
        for pair, advice in conflicts:
            if set(pair) <= words:
                findings.append({
                    "rule": "R5",
                    "species": clip["species"],
                    "clip": clip["clip"],
                    "label": clip["label"],
                    "gif": clip.get("gif_path", ""),
                    "words": list(pair),
                    "advice": advice,
                    "problem": "'%s' and '%s' together -- %s" % (pair[0], pair[1], advice),
                })
    return findings, stats


# ──────────────────────────────── reporting ─────────────────────────────────
def report_r3(findings, stats, detail=True):
    print("\n== R3  mirror consistency ==")
    print(f"   {stats['pairs_checked']} left/right pair(s) found")
    if stats.get("same_label"):
        print(f"   {stats['same_label']} pair(s) spelled identically, which IS a mirror "
              f"({stats.get('same_label_aimed', 0)} of them on an aimed action)")
        print("     whether those lost an L/R axis is a question about the MOTION, not "
              "about the names:")
        print("     tools/prefill_direction_words.py measures it")
    if not findings:
        print("   OK -- every mirror pair carries mirrored labels")
        return
    print(f"   {len(findings)} pair(s) inconsistent")
    if not detail:
        return
    print()
    for item in findings:
        print(f"   {item['species']}  {item['base']}")
        for problem in item["problems"]:
            print(f"     ! {problem}")
        print(f"     L: {item['left']['label']!r}   ({item['left']['clip']})")
        print(f"     R: {item['right']['label']!r}   ({item['right']['clip']})")
        gifs = [item[side]["gif"] for side in ("left", "right") if item[side]["gif"]]
        for gif in gifs:
            print(f"     watch: {gif}")


def report_r4(findings, stats, detail=True):
    print("\n== R4  direction spelling ==")
    total = stats["locomotion_clips"]
    vertical = stats["vertical_clips"]
    no_heading = stats.get("no_heading_clips", 0)
    parts = []
    if vertical:
        parts.append(f"{vertical} vertical")
    if no_heading:
        parts.append(f"{no_heading} no-heading")
    exempt = f" ({', '.join(parts)}, exempt)" if parts else ""
    if not findings:
        print(f"   OK -- all {total} locomotion label(s) name a heading{exempt}")
        return
    share = len(findings) / total if total else 0.0
    print(f"   {len(findings)}/{total} ({share:.0%}) locomotion label(s) name no "
          f"heading at all{exempt}")
    if not detail:
        return
    print()
    by_species = defaultdict(list)
    for item in findings:
        by_species[item["species"]].append(item)
    for species, items in sorted(by_species.items(), key=lambda kv: -len(kv[1])):
        labels = sorted({item["label"] for item in items})
        preview = ", ".join(repr(label) for label in labels[:4])
        if len(labels) > 4:
            preview += f", +{len(labels) - 4} more"
        print(f"   {len(items):4d}  {species:38s} {preview}")


def report_r5(findings, stats, detail=True):
    print("\n== R5  gait-word conflicts ==")
    if not stats["checks_active"]:
        print("   skipped -- no conflict pair is active in the current ACTION_VOCAB")
        return
    if not findings:
        print("   OK -- no contradictory gait words")
        return
    print(f"   {len(findings)} label(s) with contradictory gait words")
    if not detail:
        return
    print()
    for item in findings:
        print(f"   {item['species']:38s} {item['clip']}")
        print(f"     {item['label']!r} -- {item['problem']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-clip audit of action_labels.jsonl (R3/R4/R5).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--cond-path", "--cond_path", dest="cond_path", default=None,
        help="cond.npy naming the species and their dataset roots "
             "(default: the standard dataset cond).",
    )
    parser.add_argument(
        "--action-group", "--action_group", dest="action_group", default="locomotion",
        choices=list(ACTION_GROUPS) + ["all"],
        help="Restrict the audit to one action group (default: locomotion). "
             "'all' audits every group.",
    )
    parser.add_argument(
        "--rules", default=",".join(ALL_RULES),
        help="Comma-separated subset of %s (default: all)." % ",".join(ALL_RULES),
    )
    parser.add_argument(
        "--html", dest="html_path", default=None, metavar="HTML",
        help="Where to write the review page (default: %s)."
             % DEFAULT_REPORT_PATH.relative_to(ANYTOP_DIR).as_posix(),
    )
    parser.add_argument(
        "--no-html", "--no_html", dest="write_html", action="store_false",
        help="Skip the HTML page and print the full findings as text instead.",
    )
    parser.add_argument(
        "--text", dest="text", action="store_true",
        help="Print every finding as text as well (the HTML page carries them "
             "otherwise; implied by --no-html).",
    )
    parser.add_argument(
        "--open", dest="open_report", action="store_true",
        help="Open the HTML page in the default browser when it is written.",
    )
    parser.add_argument("--json", dest="json_path", default=None,
                        help="Also write the findings to this JSON file.")
    parser.add_argument("--verbose", action="store_true",
                        help="Report clips skipped for want of a sidecar row.")
    parser.add_argument("--strict", action="store_true",
                        help="Exit 1 when any violation is found (CI gate).")
    args = parser.parse_args()

    requested = [rule.strip().upper() for rule in args.rules.split(",") if rule.strip()]
    unknown = [rule for rule in requested if rule not in ALL_RULES]
    if unknown:
        parser.error("unknown rule(s) %s; valid rules are %s"
                     % (", ".join(unknown), ", ".join(ALL_RULES)))

    cond_path = args.cond_path
    if cond_path is None:
        from data_loaders.truebones.truebones_utils.get_opt import DEFAULT_COND_PATH
        cond_path = DEFAULT_COND_PATH
    cond_dict = load_cond(cond_path)
    sources = sources_from_cond(cond_dict, cond_path)

    group_filter = None if args.action_group == "all" else args.action_group
    clips = collect_clips(cond_dict, sources, group_filter, verbose=args.verbose)
    if not clips:
        print("No clips matched -- nothing to audit.")
        return 0

    print(f"cond      : {cond_path}")
    print(f"sources   : {', '.join(source.namespace for source in sources)}")
    print(f"scope     : {args.action_group}"
          + ("" if group_filter is None else
             "   <- the other groups are NOT audited; pass --action-group all"))
    print(f"clips     : {len(clips)} over "
          f"{len({clip['species'] for clip in clips})} species")
    print(f"rules     : {', '.join(requested)}")
    print(f"exempt    : R4 skips a locomotion label carrying "
          f"{', '.join(NO_HEADING_WORDS)} (no heading of travel to name)")
    print(f"exempt    : R4 asks no planar direction of "
          f"{', '.join(NO_PLANAR_DIRECTION_WORDS)} (aimed nowhere)")
    print("note      : no rule reads a direction off a clip name -- R3's verdict is "
          "symmetric")

    # The HTML page shows every finding beside its GIF, which is what the text
    # dump was standing in for; keep the text when there is no page to read.
    detail = args.text or not args.write_html

    findings: list[dict] = []
    for rule, check, report in (("R3", check_r3, report_r3),
                                ("R4", check_r4, report_r4),
                                ("R5", check_r5, report_r5)):
        if rule not in requested:
            continue
        found, stats = check(clips)
        report(found, stats, detail)
        findings += found

    counts = {rule: sum(1 for item in findings if item["rule"] == rule)
              for rule in requested}
    print("\n== summary ==")
    for rule in requested:
        print(f"   {rule}: {counts[rule]} violation(s)")
    print(f"   total: {len(findings)}")

    if args.json_path:
        payload = {
            "cond_path": str(cond_path),
            "action_group": args.action_group,
            "rules": requested,
            "counts": counts,
            "findings": findings,
        }
        Path(args.json_path).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"   wrote {args.json_path}")

    if args.write_html:
        html_path = Path(args.html_path) if args.html_path else DEFAULT_REPORT_PATH
        html_path = write_html_report(
            html_path,
            findings=findings,
            clips=clips,
            meta={
                "cond_path": str(cond_path),
                "action_group": args.action_group,
                "rules": requested,
                "counts": counts,
                "clip_count": len(clips),
                "species_count": len({clip["species"] for clip in clips}),
                "command": " ".join(["python"] + sys.argv),
            },
        )
        # Printed as a URL because the page reaches its GIFs through absolute
        # file:// URIs -- it is opened from disk, not through review/serve.py
        # (an http:// page may not load a file:// image).
        print(f"\n   review page: {html_path.resolve().as_uri()}")
        print("   看 GIF -> 改标签 -> 「复制修复指令」-> 粘给 LLM 写回 "
              f"{ACTION_LABELS_FILE}")
        if args.open_report:
            import webbrowser

            webbrowser.open(html_path.resolve().as_uri())

    if args.strict and findings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
