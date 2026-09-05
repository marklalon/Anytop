#!/usr/bin/env python3
"""
Cross-clip audit of ``action_labels.jsonl``.

``_validate_action_label_entry`` checks one row at a time (vocabulary, canonical
order; repeats are already stripped by normalize_action_label). The defects that
actually hurt training are *between* rows --
two clips of one species sharing a label that cannot describe both of them -- and
nothing checked those until this tool. See
``E:\\Dataset\\UnityBundles\\build\\anytop_action_label_plan.md`` for the full
derivation; the short version is that inside one species the action label is the
only thing that picks a mode, so a label covering two incompatible motions makes
``p(x | species, label)`` bimodal and the x0 prediction lands between the modes.

Four rules, each independently selectable with ``--rules``:

  R1  label-bucket spread.  Inside one (species, action_group, action_label)
      bucket, no two clips may be further apart than the species' own median
      pairwise distance.  The denominator is per-species on purpose: one
      absolute threshold necessarily mis-judges either the tight species or the
      diverse one.  A shared label is NOT a defect by itself -- Dragon's four
      clips all say "fly, flap" and it is the best-behaved species in the set.

      R1 is deliberately conservative so its findings read as real problems,
      not a wall of near-misses.  Three things narrow it down: a bucket must
      clear ``--ratio-threshold`` (default 1.5 -- the worst pair more than half
      again as far apart as the species' own median spread) before it is flagged
      at all; a bucket whose label starts with an ``--r1-exempt-labels`` head
      word (default die/hurt/getup) is skipped, because those categories are
      diverse by nature -- a species' deaths land forward, backward, collapse
      and twitch, but all of them are "die"; and a bucket a human already
      cleared in a previous run is loaded from an ignore file (``--ignore`` /
      ``dataset/review/action_label_audit_ignore.jsonl``) and never re-flagged.
      The review page's "无需修改" verdict on an R1 case appends exactly the
      line this tool reads back, so a confirmed-fine decision survives the run.

  R3  mirror consistency.  Clips whose names differ only by Left/Right (or a
      trailing L/R) must carry mirrored labels, each side must carry its own
      side word, and the two side words must not be crossed -- a crossed pair is
      still a valid mirror of itself, so only an explicit check catches it.

Every finding carries the paths of the review GIFs it concerns, and none of them
prescribes a fix.  A clip NAME is the weakest evidence there is about what a clip
does -- it is good enough to pair two clips up and to notice that the pair
disagrees with itself, and not good enough to decide which half is wrong.  That
is settled from the render: ``dataset/review/relabel_actions_llm.py`` feeds those
same GIFs to the vision model for exactly this purpose.

  R4  direction spelling.  Every ``locomotion`` label must name a heading:
      a planar one, or a vertical one for a clip that travels out of the ground
      plane ("fly, up", "jump, fall").  Corpus-wide a plain forward walk is
      spelled "walk" 87 times and "walk, forward" 30 times, which makes
      "forward" noise in a condition shared across every species.

  R5  gait-word conflicts.  ``die`` + ``fall``, ``die`` + ``idle``, ``walk`` +
      ``run`` (should be ``walk, trot``), ``glide`` + ``flap``, and ``slow`` +
      ``fast`` -- each names one axis twice.  The last three are gated on
      their replacement word being in ACTION_VOCAB; all five are active today.

Distance metric (R1): a clip is resampled to ``--frames`` frames and reduced to
its per-joint position AND velocity channels.  Both are needed: the canonical
position channel is a rest-centered residual that holds the pose, and every bit
of the travel is in the velocity channels, so position alone cannot tell a
forward walk from a backward one.  Each block is divided by its (species,
action_group) partition's median RMS, so the two weigh equally and the
comparison is about shape rather than skeleton scale.  Distance is the RMS of
the elementwise difference.

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
    python tools/audit_action_labels.py --ignore extras/cleared.jsonl
    python tools/audit_action_labels.py --r1-exempt-labels die,hurt --ratio-threshold 1.2

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

import numpy as np

ANYTOP_DIR = Path(__file__).resolve().parent.parent
_PARENT_DIR = ANYTOP_DIR.parent
sys.path.insert(0, str(_PARENT_DIR))
sys.path.insert(0, str(ANYTOP_DIR))

from data_loaders.truebones.truebones_utils.canonical_features import (  # noqa: E402
    CANONICAL_FEATURE_SPACE,
)
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
    _validate_action_label_entry,
    load_action_labels,
    normalize_action_group,
    normalize_action_label,
    reset_action_label_warning_state,
)
from tools.audit_report_html import (  # noqa: E402
    DEFAULT_REPORT_PATH,
    write_html_report,
)

ALL_RULES = ("R1", "R3", "R4", "R5")

# R1 defaults, kept as constants so the console and the review page describe one
# set of thresholds.
DEFAULT_RATIO_THRESHOLD = 1.5
# Label head words whose R1 buckets are skipped outright: these categories are
# diverse by nature (a species' deaths land forward/backward/collapse/twitch;
# hurt and getup are the same), so a shared label there is normal, not the
# bimodal defect R1 hunts for. Override with --r1-exempt-labels ('' disables).
DEFAULT_R1_EXEMPT_LABELS = ("die", "hurt", "getup")

# Sidecar where a human's "confirmed fine" verdicts live, rule-agnostic: one
# JSONL entry per line, either {"species", "action_group", "action_label"} (an
# R1 bucket -- every member clip fine) or {"species", "clip"} (an R3/R4/R5 clip
# fine; R3 needs both halves of a pair). Loaded automatically when the file is
# present (disable with --no-default-ignore); the review page emits the lines to
# append here for any case marked "无需修改".
DEFAULT_IGNORE_PATH = ANYTOP_DIR / "dataset" / "review" / "action_label_audit_ignore.jsonl"

# Channel offsets into a canonical_motion_v3 frame, per joint (n_feats == 13):
# 0:3 position (rest-centered residual), 3:9 rotation 6d, 9:12 local velocity,
# 12 foot contact. See data_loaders/truebones/truebones_utils/canonical_features.py
# -- R1 reads the first and third, and main() checks the cond declares this space.
POSITION_CHANNELS = (0, 3)
VELOCITY_CHANNELS = (9, 12)

# Direction words that name travel in the ground plane.
PLANAR_DIRECTIONS = ("forward", "backward", "left", "right")

# ... and words that name travel out of it. A dive, a jump-fall or a vertical
# take-off has no planar heading to spell, so R4 must not demand one: doing so
# turned ~40 correct vertical labels into violations and drowned the handful of
# real ones. ``fall`` and ``dive`` are ACTION_VOCAB words, not directions, but
# they name the same vertical axis and settle the same question.
VERTICAL_WORDS = ("up", "down", "dive", "fall")

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
def load_label_overrides(paths):
    """``{clip: {action_group, action_label}}`` from extra labels files.

    Lets a fresh annotation pass be audited BEFORE it is copied into the
    dataset.  The rules here are cross-clip, so the only way to know whether a
    pass actually fixed anything is to measure it against the motions -- and
    doing that by writing the pass into the dataset first means a bad pass has
    already replaced the good labels by the time the audit says so.

    Rows are validated exactly as the sidecar's are: an override that is not
    canonical would be measured in a spelling the trainer will never see.
    """
    overrides = {}
    # Vocabulary warnings are deduplicated per word for the whole process, so a
    # fresh pass must start with a clean slate or the corpus load would have
    # already silenced the very words this file is being audited for.
    reset_action_label_warning_state()
    for path in paths or ():
        path = Path(path)
        if not path.is_file():
            raise SystemExit(f"labels override not found: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                clip = str(entry["clip"])
                group = normalize_action_group(entry.get("action_group"))
                label = normalize_action_label(entry.get("action_label"))
                _validate_action_label_entry(group, label, clip, line_number)
                overrides[clip] = {"action_group": group, "action_label": label}
    return overrides


def load_ignored_keys(paths):
    """Confirmed-fine entries from an ignore JSONL -- rule-agnostic.

    ONE file feeds every rule; nothing in a line names a rule, so the same
    "confirmed fine" verdict works whether R1, R3, R4 or R5 reported it. A line
    is one of two shapes, decided here rather than tagged in the file:

      * a shared-label BUCKET -- ``{species, action_group, action_label}``.
        What an R1 finding concerns (the label a whole bucket of clips shares);
        marking it fine means every member clip is fine.
      * a single CLIP -- ``{species, clip}``. What an R3/R4/R5 finding concerns
        (a mirror pair, or one locomotion / gait clip); R3 is suppressed only
        when BOTH halves of the pair carry a clip line.

    A line that does not parse, or names neither a bucket nor a clip, stops the
    run -- a typo would otherwise silently re-flag something the operator
    already decided on.

    Returns ``{"buckets": {(species, action_group, action_label)},
               "clips": {(species, clip)}}``.
    """
    buckets, clips = set(), set()
    for path in paths or ():
        path = Path(path)
        if not path.is_file():
            raise SystemExit(f"ignore file not found: {path}")
        for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: bad ignore line: {exc}") \
                    from exc
            species = str(entry.get("species", "")).strip()
            label = str(entry.get("action_label", "")).strip()
            if "clip" in entry or not label:
                # A clip entry: one clip a human confirmed fine.
                clip = str(entry.get("clip", "")).strip()
                if not (species and clip):
                    raise SystemExit(
                        f"{path}:{line_number}: clip ignore line needs 'species' "
                        "and 'clip'")
                clips.add((species, clip))
            else:
                # A bucket entry: a shared-label bucket a human cleared.
                group = str(entry.get("action_group", "")).strip()
                if not (species and group and label):
                    raise SystemExit(
                        f"{path}:{line_number}: bucket ignore line needs "
                        "'species', 'action_group' and 'action_label'")
                buckets.add((species, group, label))
    return {"buckets": buckets, "clips": clips}


def collect_clips(cond_dict, sources, action_group=None, verbose=False,
                  overrides=None):
    """``[{clip, species, group, label, motion_path}, ...]`` for the whole corpus.

    Species membership follows the training loader exactly (``MotionDataset``):
    a clip belongs to the species whose ``species_name`` prefixes it followed by
    an underscore.  Matching that rule matters -- an audit that grouped clips
    differently from training would report on buckets the model never sees.
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
        if overrides:
            labels = dict(labels)
            labels.update({clip: entry for clip, entry in overrides.items()
                           if clip in labels})
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
                entry = labels.get(name)
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


def load_trajectory(motion_path, n_joints, frames):
    """One clip resampled to *frames*, as a ``(position, velocity)`` pair.

    Both blocks are needed. In ``canonical_motion_v3`` the position channel is a
    rest-centered residual -- the root joint's is constant to the last digit --
    so ALL of the travel lives in the velocity channels: a walk and an idle of
    one species differ by 40x there and barely at all in position. A metric
    built on position alone therefore measures pose and phase only, and is
    blind to the very axis (forward vs backward, fast vs slow travel) that a
    locomotion label is supposed to pin down.

    Rotation is still excluded: it is redundant with position for this purpose
    and would double the vector for no new axis.

    Scaling is left to the caller, which normalises per (species, action_group)
    partition -- see :func:`scale_blocks`.
    """
    motion = np.load(motion_path, mmap_mode="r")
    total = int(motion.shape[0])
    if total == 0:
        return None
    index = np.linspace(0, total - 1, frames)
    low = np.floor(index).astype(int)
    high = np.minimum(low + 1, total - 1)
    weight = (index - low)[:, None, None]

    def resample(channels):
        start, stop = channels
        block = np.asarray(motion[low, :n_joints, start:stop], dtype=np.float64)
        block = block * (1.0 - weight)
        block += np.asarray(motion[high, :n_joints, start:stop],
                            dtype=np.float64) * weight
        return np.nan_to_num(block)

    position = resample(POSITION_CHANNELS)
    if float(np.sqrt((position ** 2).mean())) <= 1e-8:
        return None
    velocity = (resample(VELOCITY_CHANNELS)
                if int(motion.shape[2]) >= VELOCITY_CHANNELS[1]
                else np.zeros_like(position))
    return position, velocity


def scale_blocks(blocks):
    """Flat vectors whose position and velocity halves carry equal weight.

    Each block is divided by the *partition's* median RMS over that block, so
    the two carry comparable energy despite their different units and the
    comparison stays about shape rather than skeleton scale.

    The scale is per partition and NOT per clip on purpose: dividing every clip
    by its own velocity RMS would normalise away exactly what the rule is
    looking for -- a still clip and a fast locomotion clip would both come out at RMS 1 and
    land on top of each other. A block whose partition median is degenerate
    (every clip still) is dropped for that partition rather than amplified into
    pure noise.
    """
    scales = []
    for position in range(len(blocks[0])):
        magnitudes = [float(np.sqrt((item[position] ** 2).mean())) for item in blocks]
        median = float(np.median(magnitudes))
        scales.append(median if median > 1e-8 else None)

    vectors = []
    for item in blocks:
        parts = [block.reshape(-1) / scale
                 for block, scale in zip(item, scales) if scale is not None]
        vectors.append(np.concatenate(parts))
    return vectors


def pairwise_distances(vectors):
    """Symmetric matrix of RMS elementwise differences between flat *vectors*."""
    stacked = np.stack(vectors)
    gram = stacked @ stacked.T
    square = np.diag(gram)
    squared = np.maximum(square[:, None] + square[None, :] - 2.0 * gram, 0.0)
    return np.sqrt(squared / stacked.shape[1])


# ───────────────────────────────── the rules ────────────────────────────────
def check_r1(clips, cond_dict, frames, ratio_threshold, min_distance,
             ignored=None, exempt_heads=(), verbose=False):
    """Label buckets whose internal spread exceeds their species' own spread.

    Two skips narrow this down to buckets worth a human's time. ``exempt_heads``
    names label HEAD words whose buckets are skipped outright -- categories that
    are diverse by nature (die/hurt/getup). ``ignored`` is a set of
    (species, action_group, action_label) bucket keys a human already cleared in
    a previous run; they are skipped and reported as ``suppressed`` so they do
    not re-surface every run.
    """
    findings = []
    # One partition is a (species, action_group) pair: the spread denominator is
    # per-species *within a group*, since a species' idle clips would otherwise
    # inflate the baseline its locomotion buckets are measured against.
    stats = {"partitions_checked": 0, "buckets_checked": 0, "clips_unreadable": 0,
             "suppressed": 0, "exempted": 0}
    ignored = set(ignored or ())
    exempt_heads = set(exempt_heads or ())
    by_species_group = defaultdict(list)
    for clip in clips:
        by_species_group[(clip["species"], clip["group"])].append(clip)

    for (species, group), members in sorted(by_species_group.items()):
        if len(members) < 2:
            continue
        n_joints = len(cond_dict[species]["parents"])
        blocks, kept = [], []
        for clip in members:
            loaded = load_trajectory(clip["motion_path"], n_joints, frames)
            if loaded is None:
                stats["clips_unreadable"] += 1
                if verbose:
                    print(f"  [skip] {clip['clip']}: empty or degenerate motion")
                continue
            blocks.append(loaded)
            kept.append(clip)
        if len(kept) < 2:
            continue
        vectors = scale_blocks(blocks)

        stats["partitions_checked"] += 1
        distances = pairwise_distances(vectors)
        upper = distances[np.triu_indices(len(kept), k=1)]
        spread = float(np.median(upper))
        if spread <= 1e-8:
            continue

        buckets = defaultdict(list)
        for index, clip in enumerate(kept):
            buckets[clip["label"]].append(index)
        for label, indices in sorted(buckets.items()):
            if len(indices) < 2:
                continue
            stats["buckets_checked"] += 1
            words = label_words(label)
            head_word = words[0] if words else ""
            if (species, group, label) in ignored:
                stats["suppressed"] += 1
                continue
            if head_word in exempt_heads:
                stats["exempted"] += 1
                continue
            block = distances[np.ix_(indices, indices)]
            local = np.unravel_index(int(np.argmax(block)), block.shape)
            worst = float(block[local])
            ratio = worst / spread
            if ratio <= ratio_threshold or worst <= min_distance:
                continue
            findings.append({
                "rule": "R1",
                "species": species,
                "action_group": group,
                "label": label,
                "ratio": round(ratio, 3),
                "max_distance": round(worst, 3),
                "species_spread": round(spread, 3),
                "clips": [kept[i]["clip"] for i in indices],
                "worst_pair": [kept[indices[local[0]]]["clip"],
                               kept[indices[local[1]]]["clip"]],
                # The distance says the two are far apart; only the render says
                # which of them the shared label actually describes.
                "worst_pair_gifs": [kept[indices[local[0]]].get("gif_path", ""),
                                    kept[indices[local[1]]].get("gif_path", "")],
            })
    findings.sort(key=lambda item: -item["ratio"])
    return findings, stats


def check_r3(clips, ignored_clips=None):
    """Mirror pairs whose labels are not mirrors of each other.

    A pair is skipped when BOTH halves carry a ``{species, clip}`` ignore line
    -- confirming one side fine while the other stays flagged would hide a real
    mismatch, and it is the pair, not one clip, that R3 reports.
    """
    ignored = set(ignored_clips or ())
    findings = []
    stats = {"pairs_checked": 0, "suppressed": 0}
    groups = defaultdict(dict)
    for clip in clips:
        side = clip_side(clip["clip"])
        if side is None:
            continue
        key = (clip["species"], clip["group"], clip_side_base(clip["clip"]))
        # Two clips on the same side under one base name (e.g. WalkLeft and
        # WalkTurnLeft collapsing) would make the pairing ambiguous; keep the
        # first and let R1 speak for the rest.
        groups[key].setdefault(side, clip)

    for (species, group, base), sides in sorted(groups.items()):
        if "L" not in sides or "R" not in sides:
            continue
        stats["pairs_checked"] += 1
        left, right = sides["L"], sides["R"]
        if ((species, left["clip"]) in ignored
                and (species, right["clip"]) in ignored):
            stats["suppressed"] += 1
            continue
        left_label, right_label = left["label"], right["label"]
        left_words, right_words = set(label_words(left_label)), set(label_words(right_label))
        problems = []
        # A stable code beside each sentence: the HTML report words these in
        # Chinese, and keying that off the English prose would silently fall
        # back to the prose the day someone rewords it.
        codes = []

        # Labels that ARE mirrors of each other but mirrored the wrong way round
        # pass the mirror test, so nothing but naming the swap explains the pair.
        # This is a real failure mode -- Scorpion-2_StrafeLeft carrying 'right'
        # and StrafeRight carrying 'left' is self-consistent and still backwards.
        swapped = ("right" in left_words and "left" not in left_words
                   and "left" in right_words and "right" not in right_words)
        if swapped:
            problems.append("the side words are crossed: the clip named Left says "
                            "'right' and the clip named Right says 'left'")
            codes.append("crossed")
            candidate = {"left": mirror_label(left_label),
                         "right": mirror_label(right_label)}
        else:
            if mirror_label(left_label) != right_label:
                problems.append("labels are not mirrors of each other")
                codes.append("not_mirror")
            if "left" not in left_words or "right" not in right_words:
                problems.append("a side carries no side word, so the L/R axis is lost")
                codes.append("no_side_word")
            candidate = {"left": left_label, "right": mirror_label(left_label)}

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
            # NAME-DERIVED and unconfirmed: this rule pairs clips by their name
            # and can therefore say the pair disagrees, but a clip name is the
            # weakest evidence there is about what the clip does -- a clip called
            # StrafeLeft may well strafe right. Which side to rewrite is decided
            # from the review GIF, not from here.
            "candidate_left": candidate["left"],
            "candidate_right": candidate["right"],
            "candidate_basis": "clip name (unconfirmed -- watch the GIFs)",
        })
    return findings, stats


def check_r4(clips, ignored_clips=None):
    """``locomotion`` labels that name neither a planar nor a vertical heading."""
    ignored = set(ignored_clips or ())
    findings = []
    stats = {"locomotion_clips": 0, "vertical_clips": 0, "suppressed": 0}
    planar = set(PLANAR_DIRECTIONS)
    vertical = set(VERTICAL_WORDS)
    for clip in sorted(clips, key=lambda item: (item["species"], item["clip"])):
        if clip["group"] != "locomotion":
            continue
        stats["locomotion_clips"] += 1
        if (clip["species"], clip["clip"]) in ignored:
            stats["suppressed"] += 1
            continue
        words = set(label_words(clip["label"]))
        if planar & words:
            continue
        if vertical & words:
            # 'fly, up', 'swim, down', 'jump, fall', 'fly, dive, down': the
            # heading is named, it just is not in the ground plane.
            stats["vertical_clips"] += 1
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


def check_r5(clips, ignored_clips=None):
    """Word pairs that contradict each other, or say the same thing twice."""
    ignored = set(ignored_clips or ())
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
    stats = {"checks_active": len(conflicts), "suppressed": 0}
    if not conflicts:
        return findings, stats
    for clip in sorted(clips, key=lambda item: (item["species"], item["clip"])):
        if (clip["species"], clip["clip"]) in ignored:
            stats["suppressed"] += 1
            continue
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
def report_r1(findings, stats, detail=True):
    print("\n== R1  label-bucket spread ==")
    print(f"   {stats['buckets_checked']} shared-label bucket(s) over "
          f"{stats['partitions_checked']} species x action_group partition(s)"
          + (f" ({stats['clips_unreadable']} clips unreadable)"
             if stats["clips_unreadable"] else ""))
    pre = []
    if stats["suppressed"]:
        pre.append(f"{stats['suppressed']} skipped (already confirmed fine)")
    if stats["exempted"]:
        pre.append(f"{stats['exempted']} skipped (exempt head words)")
    if pre:
        print("   " + " · ".join(pre))
    if not findings:
        print("   OK -- every shared label covers clips no further apart than "
              "their species' own median spread")
        return
    species = sorted({item["species"] for item in findings})
    print(f"   {len(findings)} bucket(s) over threshold, {len(species)} species")
    if not detail:
        return
    print(f"\n   {'ratio':>6s} {'maxd':>5s} {'spread':>6s}  species / label")
    for item in findings:
        print(f"   {item['ratio']:6.2f} {item['max_distance']:5.2f} "
              f"{item['species_spread']:6.2f}  {item['species']}  '{item['label']}'")
        print(f"   {'':21s}  {item['worst_pair'][0]}")
        print(f"   {'':21s}  {item['worst_pair'][1]}")
        others = [name for name in item["clips"] if name not in item["worst_pair"]]
        if others:
            print(f"   {'':21s}  (+{len(others)} more in this bucket)")


def report_r3(findings, stats, detail=True):
    print("\n== R3  mirror consistency ==")
    print(f"   {stats['pairs_checked']} left/right pair(s) found")
    if stats.get("suppressed"):
        print(f"   {stats['suppressed']} pair(s) already confirmed fine")
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
        # A candidate, not a verdict: it is read off the clip NAME, and the name
        # is the weakest evidence about the motion. Only the sides that would
        # actually change are shown -- echoing a side's current value back at it
        # reads as a tool bug rather than a finding.
        changes = [(side, item[f"candidate_{side}"])
                   for side in ("left", "right")
                   if item[f"candidate_{side}"] != item[side]["label"]]
        if changes:
            print("     candidate (from the clip name, NOT confirmed):")
            for side, value in changes:
                print(f"       {side[0].upper()}: {value!r}")
        gifs = [item[side]["gif"] for side in ("left", "right") if item[side]["gif"]]
        for gif in gifs:
            print(f"     watch: {gif}")


def report_r4(findings, stats, detail=True):
    print("\n== R4  direction spelling ==")
    total = stats["locomotion_clips"]
    vertical = stats["vertical_clips"]
    exempt = f" ({vertical} vertical, exempt)" if vertical else ""
    if stats.get("suppressed"):
        print(f"   {stats['suppressed']} clip(s) already confirmed fine")
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
    if stats.get("suppressed"):
        print(f"   {stats['suppressed']} clip(s) already confirmed fine")
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
        description="Cross-clip audit of action_labels.jsonl (R1/R3/R4/R5).",
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
        "--labels", action="append", default=None, metavar="JSONL",
        help="Extra action_labels.jsonl whose rows OVERRIDE the dataset "
             "sidecar for the clips they name (repeatable). Audits a fresh "
             "annotation pass before it is copied into the dataset.",
    )
    parser.add_argument(
        "--rules", default=",".join(ALL_RULES),
        help="Comma-separated subset of %s (default: all)." % ",".join(ALL_RULES),
    )
    parser.add_argument(
        "--ratio-threshold", "--ratio_threshold", dest="ratio_threshold",
        type=float, default=DEFAULT_RATIO_THRESHOLD,
        help="R1: flag a bucket whose worst pair exceeds this multiple of the "
             "species' median pairwise distance (default: %(default)s).",
    )
    parser.add_argument(
        "--min-distance", "--min_distance", dest="min_distance",
        type=float, default=0.5,
        help="R1: absolute floor on the worst pair, so a very tight species "
             "cannot trip the ratio on noise (default: %(default)s).",
    )
    parser.add_argument(
        "--r1-exempt-labels", "--r1_exempt_labels", dest="r1_exempt_labels",
        default=",".join(DEFAULT_R1_EXEMPT_LABELS),
        help="R1: comma-separated label HEAD words whose buckets are skipped -- "
             "categories diverse by nature (default: %(default)s). Pass '' to "
             "disable the exemption.",
    )
    parser.add_argument(
        "--ignore", action="append", default=None, metavar="JSONL",
        help="Extra ignore file of already-confirmed-fine entries "
             "{species, action_group, action_label} (an R1 bucket) or "
             "{species, clip} (an R3/R4/R5 clip), one JSON per line "
             "(repeatable). Loaded on top of the default "
             "dataset/review/action_label_audit_ignore.jsonl.",
    )
    parser.add_argument(
        "--no-default-ignore", dest="use_default_ignore", action="store_false",
        help="Do not load the default ignore file.",
    )
    parser.add_argument(
        "--frames", type=int, default=32,
        help="R1: frames each clip is resampled to before comparison (default: 32).",
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
                        help="Report skipped clips.")
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

    # POSITION_CHANNELS / VELOCITY_CHANNELS are offsets into one specific
    # feature space. A cond that declares another one would be measured with the
    # wrong channels and report confident nonsense, so say so rather than guess.
    spaces = {str(entry.get("feature_space", CANONICAL_FEATURE_SPACE))
              for entry in cond_dict.values()}
    unexpected = sorted(spaces - {CANONICAL_FEATURE_SPACE})
    if unexpected:
        print(f"[WARN] cond declares feature space(s) {unexpected}; R1 reads "
              f"channels {POSITION_CHANNELS} and {VELOCITY_CHANNELS} of "
              f"{CANONICAL_FEATURE_SPACE} and its distances may be meaningless.",
              file=sys.stderr)

    group_filter = None if args.action_group == "all" else args.action_group
    overrides = load_label_overrides(args.labels)
    clips = collect_clips(cond_dict, sources, group_filter, verbose=args.verbose,
                          overrides=overrides)
    if overrides:
        applied = sum(1 for clip in clips if clip["clip"] in overrides)
        print(f"overrides : {len(overrides)} row(s) from "
              f"{', '.join(args.labels)}; {applied} applied in scope")
    if not clips:
        print("No clips matched -- nothing to audit.")
        return 0

    ignore_paths = list(args.ignore or ())
    if args.use_default_ignore and DEFAULT_IGNORE_PATH.is_file():
        ignore_paths.append(str(DEFAULT_IGNORE_PATH))
    _ignore = load_ignored_keys(ignore_paths)
    ignored_buckets, ignored_clips = _ignore["buckets"], _ignore["clips"]
    exempt_heads = {word.strip().lower()
                    for word in (args.r1_exempt_labels or "").split(",")
                    if word.strip()}
    n_ignore = len(ignored_buckets) + len(ignored_clips)
    if n_ignore:
        print(f"ignore    : {n_ignore} confirmed-fine item(s) "
              f"({len(ignored_buckets)} bucket, {len(ignored_clips)} clip) from "
              f"{', '.join(ignore_paths)}")
    if exempt_heads:
        print(f"exempt    : R1 skips label head word(s) "
              f"{', '.join(sorted(exempt_heads))}")

    print(f"cond      : {cond_path}")
    print(f"sources   : {', '.join(source.namespace for source in sources)}")
    print(f"scope     : {args.action_group}"
          + ("" if group_filter is None else
             "   <- the other groups are NOT audited; pass --action-group all"))
    print(f"clips     : {len(clips)} over "
          f"{len({clip['species'] for clip in clips})} species")
    print(f"rules     : {', '.join(requested)}")

    # The HTML page shows every finding beside its GIF, which is what the text
    # dump was standing in for; keep the text when there is no page to read.
    detail = args.text or not args.write_html

    findings: list[dict] = []
    if "R1" in requested:
        found, stats = check_r1(cond_dict=cond_dict, clips=clips, frames=args.frames,
                                ratio_threshold=args.ratio_threshold,
                                min_distance=args.min_distance,
                                ignored=ignored_buckets, exempt_heads=exempt_heads,
                                verbose=args.verbose)
        report_r1(found, stats, detail)
        findings += found
    if "R3" in requested:
        found, stats = check_r3(clips, ignored_clips)
        report_r3(found, stats, detail)
        findings += found
    if "R4" in requested:
        found, stats = check_r4(clips, ignored_clips)
        report_r4(found, stats, detail)
        findings += found
    if "R5" in requested:
        found, stats = check_r5(clips, ignored_clips)
        report_r5(found, stats, detail)
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
            "thresholds": {
                "ratio": args.ratio_threshold,
                "min_distance": args.min_distance,
                "frames": args.frames,
            },
            "r1_exempt": sorted(exempt_heads),
            "ignored_buckets": len(ignored_buckets),
            "ignored_clips": len(ignored_clips),
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
                "thresholds": {
                    "ratio": args.ratio_threshold,
                    "min_distance": args.min_distance,
                    "frames": args.frames,
                },
                "r1_exempt": sorted(exempt_heads),
                "ignore_path": str(DEFAULT_IGNORE_PATH),
                "ignored_buckets": len(ignored_buckets),
                "ignored_clips": len(ignored_clips),
                "counts": counts,
                "clip_count": len(clips),
                "species_count": len({clip["species"] for clip in clips}),
                "command": " ".join(["python"] + sys.argv),
                "labels_overrides": [str(path) for path in (args.labels or ())],
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
