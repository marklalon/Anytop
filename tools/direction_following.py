#!/usr/bin/env python3
"""
Direction instruction-following evaluation.

The question is narrow: when the prompt names a heading, does the generated
motion go that way?

    sweep   run the prompt grid through sample/generate.py at several
            --action_label_cfg_scale values, into one output tree + a manifest
    score   report the metrics, split forward/backward vs left/right and per CFG
            scale. Two ways to get the answers:
              --auto     measure each clip's heading geometrically (below). It
                         calibrates on the real corpus and refuses any species
                         or prompt it cannot measure.
              (default)  join a human annotator's blinded answers
    sheet   build the blinded CSV the human path needs: shuffled, prompt
            withheld, one row per clip
    phase   a secondary metric: how consistent the left/right contact phase
            offset is (see contact_phase_offset)

Measuring a heading when the root does not move
-----------------------------------------------
Root XZ is stripped from every locomotion clip (features.py, has_locomotion), so
all four headings of KI_Human_Walk01 have byte-identical, exactly-zero root
translation. There is no travel to read off the root -- which is also why the
training labels could not have their directions filled in geometrically.

But stripping the root is what makes the heading measurable somewhere else: it
turns every clip into a treadmill. The support foot is planted on ground that
does not move, so once the body is pinned in place that foot has to slide at
minus the travel velocity. Average the horizontal velocity of the planted feet,
negate it, and that is the body-frame heading.

Find the support foot by HEIGHT, not by the contact channel: channel 12 is
thresholded on low velocity, so it selects away precisely the sliding this
measures.

Checked against every singly-directioned locomotion clip in the corpus. On the
species this evaluation actually prompts -- the ones carrying a full four-heading
gait set, KI_Human and its siblings -- it is exact: 68/68, 100% on both axes. It
is NOT universally valid, and the failure is not noise but a different meaning of
the word: for a quadruped or a dragon a "left" clip is a curving TURN rather than
a side-step, and its body-frame travel genuinely is forward (MB_TigerDrago,
MB_Unka and Trex all read 100% FB / 0% LR for that reason). So --auto calibrates
per species on that species' own labeled clips and refuses what it cannot verify.

When --auto covers your species, prefer it: reproducible, free, and no
annotator in the loop. Keep the human path for species --auto refuses, and as a
spot-check.

Read the heading ERROR, not top-1
---------------------------------
Top-1 snaps the measured heading to the nearest cardinal, so everything inside
+-45 degrees of the prompt counts as a hit. That is far too coarse to be the
metric: a run tracking 30 degrees off "forward" is plainly wrong on screen and
top-1 scores it correct, and a systematic offset -- every "left" sample landing
on the forward-left diagonal -- is invisible in it, because the answer is never
the wrong cardinal, only a bad angle.

So --auto reports the signed angle off the prompted heading, per direction, and
prints the same measurement over that species' REAL clips as the floor. That
floor is what makes the number readable, and it is tight: on KI_Human the corpus
clips sit within ~1 degree of the heading they are labeled with, so a generated
mean error in the double digits is not natural variation, it is the model
missing the instruction by an amount top-1 cannot see. Single-digit degrees is
roughly where a viewer stops noticing.

Top-1 is still printed, as the coarse read and the only thing the human path can
produce -- but the angle table is the one to judge a round on.

Report every direction separately, always, and never pool left with right: they
are known to fail differently (T5 places "left" and "right" closer together than
forward-vs-left) and, in R0, they failed asymmetrically -- "right" landed within
a few degrees while "left" sat ~30 degrees short of the axis at every CFG scale.
Pooling hides exactly the failure the evaluation exists to find. "mixed" is
counted on its own too -- it is the direct read on mode collapse, a different
failure from picking the wrong heading.

The SHAPE of the error against CFG scale is itself the decision point: falling
monotonically means the additive token is fine and only needed more gain; an
error that stops falling (or grows) while artifacts appear is the expressivity
signature, the only condition under which changing the injection style (the FiLM
fallback, not built) is worth trying.

Run the R0 baseline (the pre-refactor checkpoint) before anything else. It needs
no training and without it there is no threshold to judge R1 against.

This is a throwaway verification tool for the action-label keyword refactor
(docs/action_label_keyword_refactor.md), not part of the permanent eval suite.

Usage (from the Anytop/ directory):
    python tools/direction_following.py sweep --model_path save/.../model.pt \\
        --species KI_Human --output_dir eval_out/R1

    # automatic scoring (preferred)
    python tools/direction_following.py score eval_out/R1 --auto
        --cond_path dataset/merged/cond.npy
        --reference dataset/unitybundles/processed

    # human path, for a species --auto refuses to score
    python tools/direction_following.py sheet eval_out/R1
    #   ... a human fills in the 'answer' column of eval_out/R1/annotate.csv ...
    python tools/direction_following.py score eval_out/R1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ANYTOP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ANYTOP_DIR.parent))
sys.path.insert(0, str(ANYTOP_DIR))

from data_loaders.truebones.truebones_utils.motion_labels import (  # noqa: E402
    DIRECTION_VOCAB,
    canonical_action_label,
)

# The four planar headings, plus the two answers that are not a heading. "mixed"
# means the clip visibly blends headings (the mode-collapse read); "unclear"
# means the annotator could not tell, which is not the same thing and must not be
# folded into it.
ANSWERS: tuple[str, ...] = DIRECTION_VOCAB + ("mixed", "unclear")

# Which metric group each heading belongs to. Reported separately -- see module
# docstring.
_AXIS = {"forward": "forward/backward", "backward": "forward/backward",
         "left": "left/right", "right": "left/right"}

# Where each heading points in the canonical frame, in the same degrees
# classify_heading measures: atan2(+X left, +Z forward).
_TARGET_ANGLE = {"forward": 0.0, "left": 90.0, "right": -90.0, "backward": 180.0}

MANIFEST_NAME = "manifest.jsonl"
SHEET_NAME = "annotate.csv"

# Feature channel carrying binary foot contact, and the per-joint position block.
CONTACT_CHANNEL = 12
POSITION_CHANNELS = slice(0, 3)

# Canonical frame: process_anim rotates every skeleton to face +Z and the feature
# frame uses r_rot = identity, so these axes hold across species. +X is the
# character's left. A rig that disagrees is caught by calibration rather than
# assumed away -- truebones/zoo/Scorpion-2 is one, and reads left/right swapped.
_FORWARD_AXIS, _LEFT_AXIS = 2, 0

# How close to its own floor a foot must sit to count as support. Deliberately
# NOT the contact channel: that is thresholded on low velocity and would select
# away the very sliding this measures.
DEFAULT_HEIGHT_BAND = 0.05

# Minimum |travel| per frame, as a fraction of body height, for a clip to have a
# heading at all. Below it the clip is an in-place gait and the measured angle is
# noise, so the scorer abstains instead of guessing. At this floor the corpus
# check reads 100% on forward/backward.
DEFAULT_SPEED_FLOOR = 0.005

# How much of a species' own labeled corpus --auto must get right before it will
# score generated samples for that species.
DEFAULT_CALIBRATION_THRESHOLD = 0.9

# Words that make a prompt unmeasurable this way whatever the species: 'turn'
# names a change of facing, which disagrees with a travel heading by
# construction, and swimming / flying have no support foot to read.
UNMEASURABLE_WITH = ("turn", "swim", "fly")


# ---------------------------------------------------------------------------
# the geometric heading estimator (used by `score --auto`)
# ---------------------------------------------------------------------------

def support_foot_heading(motion, contact_joints, height_band=DEFAULT_HEIGHT_BAND):
    """``(x, z, speed)`` body-frame heading of one clip, or ``None``.

    ``speed`` is |travel| per frame divided by body height, so it is comparable
    across rigs; ``DEFAULT_SPEED_FLOOR`` is expressed in the same unit. See the
    module docstring for why the planted foot carries the heading and why the
    contact channel must not be used to find it.
    """
    positions = motion[:, list(contact_joints), POSITION_CHANNELS]
    if positions.shape[0] < 3 or positions.shape[1] == 0:
        return None
    # Per-joint floor, not a global one: a rig can carry contact joints at
    # genuinely different heights (a toe and an ankle).
    floor = np.percentile(positions[:, :, 1], 5.0, axis=0, keepdims=True)
    planted = np.abs(positions[:, :, 1] - floor) <= height_band
    # Both endpoints of the step have to be planted, or the lift-off frame
    # contributes a swing velocity to what is supposed to be stance.
    support = planted[1:] & planted[:-1]
    if not support.any():
        return None
    velocity = positions[1:] - positions[:-1]
    weight = support[..., None]
    travel = -(velocity * weight).sum(axis=(0, 1)) / weight.sum()
    body_height = float(
        np.percentile(motion[:, :, 1], 95) - np.percentile(motion[:, :, 1], 5)
    )
    if not np.isfinite(body_height) or body_height <= 0:
        body_height = 1.0
    x, z = float(travel[_LEFT_AXIS]), float(travel[_FORWARD_AXIS])
    return x, z, math.hypot(x, z) / body_height


def classify_heading(x, z):
    """Nearest cardinal to the measured heading. +Z is forward, +X is left."""
    angle = math.degrees(math.atan2(x, z))
    if -45.0 <= angle < 45.0:
        return "forward"
    if 45.0 <= angle < 135.0:
        return "left"
    if -135.0 <= angle < -45.0:
        return "right"
    return "backward"


def heading_error(angle, prompted):
    """Signed degrees from *angle* to the *prompted* heading, on (-180, 180].

    Positive is toward the character's left. Signed rather than absolute because
    the sign is where the interesting failure lives: samples scattered either way
    around the axis are noise, while a whole prompt leaning one way is a
    systematic offset -- the model reading the word and then under-rotating.
    """
    return (angle - _TARGET_ANGLE[prompted] + 180.0) % 360.0 - 180.0


def measure_clip(motion, contact_joints, speed_floor=DEFAULT_SPEED_FLOOR,
                 height_band=DEFAULT_HEIGHT_BAND):
    """``(direction, speed, angle)``, or ``(None, speed, None)`` for no heading.

    ``angle`` is the measured heading in degrees before it is snapped to a
    cardinal -- the continuous quantity the report is actually built on. See
    "Read the heading ERROR, not top-1" in the module docstring.
    """
    result = support_foot_heading(motion, contact_joints, height_band)
    if result is None:
        return None, 0.0, None
    x, z, speed = result
    if speed < speed_floor:
        return None, speed, None
    return classify_heading(x, z), speed, math.degrees(math.atan2(x, z))


def prompt_is_measurable(action_label):
    """The words in *action_label* that put a prompt out of reach of the estimator."""
    words = set(action_label.split(", ")) if action_label else set()
    return sorted(words & set(UNMEASURABLE_WITH))


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------

def _prompt_grid(species, actions, directions):
    for one_species in species:
        for action in actions:
            for direction in directions:
                words = [w.strip() for w in action.split(",") if w.strip()]
                yield one_species, action, direction, canonical_action_label(words + [direction])


def run_sweep(args) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    species = [s.strip() for s in args.species.split(",") if s.strip()]
    actions = [a.strip() for a in args.actions.split(";") if a.strip()]
    directions = [d.strip() for d in args.directions.split(",") if d.strip()]
    scales = [float(s) for s in args.cfg_scales.split(",") if s.strip()]
    unknown = [d for d in directions if d not in DIRECTION_VOCAB]
    if unknown:
        raise SystemExit(f"ERROR: {unknown} are not direction words; valid: {list(DIRECTION_VOCAB)}")

    rows = []
    for one_species, action, direction, label in _prompt_grid(species, actions, directions):
        for scale in scales:
            slug = f"{one_species}__{action.replace(', ', '-').replace(',', '-')}__{direction}__cfg{scale:g}"
            prompt_dir = output_dir / "samples" / slug
            existing = sorted(prompt_dir.glob("*.npy")) if prompt_dir.exists() else []
            if existing and not args.force:
                print(f"[skip] {slug} ({len(existing)} clip(s) already there)")
            else:
                prompt_dir.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable, "-m", "sample.generate",
                    "--model_path", args.model_path,
                    "--object_type", one_species,
                    "--action_label", label,
                    "--action_label_cfg_scale", f"{scale:g}",
                    "--batch_size", str(args.batch_size),
                    "--seed", str(args.seed),
                    "--output_dir", str(prompt_dir),
                ]
                if args.num_frames:
                    command += ["--num_frames", str(args.num_frames)]
                if args.cond_path:
                    command += ["--cond_path", args.cond_path]
                command += args.generate_args
                print(f"[run ] {slug}\n       {' '.join(command)}")
                result = subprocess.run(command, cwd=str(ANYTOP_DIR))
                if result.returncode != 0:
                    raise SystemExit(f"ERROR: generation failed for {slug} (exit {result.returncode})")
                existing = sorted(prompt_dir.glob("*.npy"))
            for index, npy_path in enumerate(existing):
                rows.append({
                    "sample_id": f"{slug}#{index:02d}",
                    "species": one_species,
                    "action": action,
                    "direction": direction,
                    "action_label": label,
                    "cfg_scale": scale,
                    "clip": str(npy_path.relative_to(output_dir).as_posix()),
                })

    manifest = output_dir / MANIFEST_NAME
    with open(manifest, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n[OK] {len(rows)} clip(s) -> {manifest}")
    print(f"     next: python tools/direction_following.py sheet {output_dir}")
    return 0


# ---------------------------------------------------------------------------
# sheet
# ---------------------------------------------------------------------------

def _load_manifest(output_dir: Path) -> list[dict]:
    manifest = output_dir / MANIFEST_NAME
    if not manifest.exists():
        raise SystemExit(f"ERROR: no {MANIFEST_NAME} in {output_dir}. Run `sweep` first.")
    with open(manifest, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_sheet(args) -> int:
    output_dir = Path(args.output_dir)
    rows = _load_manifest(output_dir)
    # Shuffled so the annotator cannot infer the prompt from row order, and
    # seeded so re-running does not invalidate answers already filled in.
    shuffled = list(rows)
    random.Random(args.shuffle_seed).shuffle(shuffled)

    sheet_path = output_dir / SHEET_NAME
    if sheet_path.exists() and not args.force:
        raise SystemExit(f"ERROR: {sheet_path} exists. Pass --force to overwrite "
                         f"(this DISCARDS any answers already filled in).")
    with open(sheet_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "clip", "answer"])
        for row in shuffled:
            # The prompt is deliberately absent from every column: sample_id is
            # opaque only if the annotator never sees the manifest, so keep the
            # two files apart while annotating.
            writer.writerow([row["sample_id"], row["clip"], ""])
    print(f"[OK] {len(shuffled)} row(s) -> {sheet_path}")
    print(f"     Render each 'clip' (the .bvh next to it) and fill 'answer' with "
          f"one of: {', '.join(ANSWERS)}")
    print(f"     Do NOT show the annotator {output_dir / MANIFEST_NAME}.")
    return 0


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

def _report(graded, source: str, errors=None, floor=None) -> None:
    """Print the metric table shared by the human and the automatic path.

    *graded* is a list of ``(manifest row, answer)``. Answers outside
    DIRECTION_VOCAB ("mixed" from a human, "unclear" from either) are counted in
    their own columns rather than as wrong, because they are a different failure:
    a wrong heading means the prompt was misread, a mixed one means it was not
    resolved at all.

    *errors* (--auto only) maps sample_id to the signed heading error in degrees.
    When given it drives the table that actually decides the round; top-1 above
    is kept as the coarse read and as the only thing a human annotator can
    produce. *floor* is the same statistic over the corpus, per direction, which
    is what makes a generated error readable as good or bad.
    """
    buckets: dict = defaultdict(lambda: defaultdict(Counter))
    confusion: Counter = Counter()
    for row, answer in graded:
        axis = _AXIS[row["direction"]]
        outcome = ("correct" if answer == row["direction"]
                   else "mixed" if answer == "mixed"
                   else "unclear" if answer == "unclear"
                   else "wrong")
        buckets[row["cfg_scale"]][axis][outcome] += 1
        confusion[(row["direction"], answer)] += 1

    print(f"\nDirection instruction-following  ({len(graded)} clip(s), {source})")
    print(f"{'cfg':>5}  {'axis':<16} {'n':>4}  {'top-1':>7}  {'mixed':>7}  {'unclear':>7}")
    for scale in sorted(buckets):
        for axis in ("forward/backward", "left/right"):
            counts = buckets[scale][axis]
            total = sum(counts.values())
            if not total:
                continue
            print(f"{scale:>5g}  {axis:<16} {total:>4}  "
                  f"{counts['correct'] / total:>6.1%}  "
                  f"{counts['mixed'] / total:>6.1%}  "
                  f"{counts['unclear'] / total:>6.1%}")

    print("\nconfusion (prompted -> answered):")
    for prompted in DIRECTION_VOCAB:
        row_total = sum(v for (p, _), v in confusion.items() if p == prompted)
        if not row_total:
            continue
        cells = "  ".join(
            f"{answer}={confusion[(prompted, answer)]}"
            for answer in ANSWERS if confusion[(prompted, answer)]
        )
        print(f"   {prompted:<9} (n={row_total:>3})  {cells}")

    if errors:
        _report_heading_error(graded, errors, floor)

    print("\nRead DOWN the cfg scales, per direction:")
    print("   error falling steadily   -> the additive token is fine, it only needed gain")
    print("   error stops falling / grows + artifacts -> expressivity-bound; the only")
    print("                               case in which changing the injection style")
    print("                               (FiLM, R2) is worth it")
    print("   one heading stuck short of its axis while the others are clean -> T5 cannot")
    print("                               place that word; the fix there is the 2-bit L/R")
    print("                               hard input, not the injection style")


def _report_heading_error(graded, errors, floor=None) -> None:
    """The continuous metric: degrees off the prompted heading, per direction.

    Split by direction and never pooled into an axis: R0 read "right" within a
    few degrees while "left" sat ~30 degrees short of its axis at every scale,
    and an axis average would have reported that pair as healthy.

    Both a magnitude and a signed mean are printed because they answer different
    questions. mean|e| is how wrong the samples are; bias is whether they are
    wrong in one direction -- when |bias| approaches mean|e| the whole prompt is
    leaning, which is a systematic offset rather than scatter, and no amount of
    sampling averages it away.
    """
    grouped: dict = defaultdict(list)
    for row, _ in graded:
        error = errors.get(row["sample_id"])
        if error is not None:
            grouped[(row["cfg_scale"], row["direction"])].append(error)
    if not grouped:
        return

    print("\nHeading error, degrees off the prompted heading (+ = toward the "
          "character's left).")
    print("This is the metric to judge the round on -- top-1 above snaps to the "
          "nearest")
    print("cardinal, so it scores a 30-degree miss and a dead-on sample the same.")
    print(f"{'cfg':>5}  {'direction':<9} {'n':>4}  {'mean|e|':>8}  {'median':>7}  "
          f"{'p90':>7}  {'bias':>7}")
    for scale in sorted({key[0] for key in grouped}):
        for direction in DIRECTION_VOCAB:
            values = grouped.get((scale, direction))
            if not values:
                continue
            magnitude = np.abs(values)
            print(f"{scale:>5g}  {direction:<9} {len(values):>4}  "
                  f"{magnitude.mean():>6.1f}deg  {np.median(magnitude):>5.1f}deg  "
                  f"{np.percentile(magnitude, 90):>5.1f}deg  "
                  f"{np.mean(values):>+5.1f}deg")

    if floor:
        print("\ncorpus floor -- the same measurement on this species' REAL clips, "
              "against")
        print("the heading they are labeled with. Generated error is only "
              "meaningful against")
        print("it: a rig whose own clips read a few degrees off cannot be asked "
              "for better.")
        for species in sorted(floor, key=str):
            cells = "  ".join(
                f"{direction}={value:.1f}deg (n={count})"
                for direction, (value, count) in sorted(
                    floor[species].items(), key=lambda kv: DIRECTION_VOCAB.index(kv[0]))
            )
            print(f"   {str(species):<34} {cells}")


def _contact_joints_by_species(cond):
    """A ``clip name -> (canonical species key, contact joints)`` resolver."""
    from data_loaders.truebones.truebones_utils.dataset_sources import resolve_species_key

    cache: dict = {}

    def lookup(name):
        if name not in cache:
            key = resolve_species_key(cond, name)
            entry = cond.get(key) if key else None
            cache[name] = (
                (key, [int(index) for index in (entry.get("contact_joints") or [])])
                if entry is not None else (None, [])
            )
        return cache[name]

    def for_clip(clip_name, namespace=None):
        # Corpus clips are "<species>_<motion>_<n>.npy" and the species part can
        # itself hold underscores, so try the longest prefix that resolves.
        # When *namespace* is given (multi-dataset calibration), index the merged
        # cond by "<namespace>/<species>" instead of a bare name: the same species
        # name can exist in two namespaces with different joint counts, and a bare
        # lookup would resolve the first one -- wrong joints, or an out-of-bounds
        # index when the counts differ.
        parts = Path(clip_name).stem.split("_")
        for cut in range(len(parts) - 1, 0, -1):
            prefix = "_".join(parts[:cut])
            if namespace is not None:
                entry = cond.get(f"{namespace}/{prefix}")
                if entry is not None and entry.get("contact_joints"):
                    return f"{namespace}/{prefix}", [
                        int(index) for index in entry["contact_joints"]
                    ]
                continue
            key, joints = lookup(prefix)
            if joints:
                return key, joints
        return None, []

    return lookup, for_clip


def calibrate(dataset_dirs, cond, speed_floor, height_band, species_filter=None):
    """Per-species accuracy of the estimator on that species' own labeled clips.

    This is what makes --auto safe to trust. The estimator reports body-frame
    TRAVEL, and for most non-bipeds a "left" clip is a curving turn whose travel
    is genuinely forward -- so a species is only scoreable if its own corpus
    clips come back with the labels they carry. Species with too few usable
    reference clips are reported as unknown, not as passing.
    """
    from data_loaders.truebones.truebones_utils.motion_labels import load_action_labels
    from data_loaders.truebones.truebones_utils.dataset_sources import (
        infer_namespace_from_root,
    )

    _, for_clip = _contact_joints_by_species(cond)
    hits: dict = defaultdict(Counter)
    for dataset_dir in dataset_dirs:
        dataset_dir = Path(dataset_dir)
        motions_dir = dataset_dir / "motions"
        if not motions_dir.is_dir():
            continue
        # Clips here belong to this dataset, so resolve their species within its
        # namespace -- see for_clip for why a bare name is unsafe across datasets.
        namespace = infer_namespace_from_root(dataset_dir)
        try:
            labels = load_action_labels(dataset_dir)
        except FileNotFoundError:
            print(f"[warn] no action_labels.jsonl in {dataset_dir} -- skipping")
            continue
        for clip_name, entry in labels.items():
            label = entry["action_label"]
            if entry["action_group"] != "locomotion" or not label:
                continue
            words = label.split(", ")
            truth = [word for word in words if word in DIRECTION_VOCAB]
            if len(truth) != 1 or prompt_is_measurable(label):
                continue
            motion_path = motions_dir / clip_name
            if not motion_path.exists():
                continue
            species, contact_joints = for_clip(clip_name, namespace=namespace)
            if not contact_joints:
                continue
            if species_filter is not None and species not in species_filter:
                continue
            predicted, _, angle = measure_clip(
                np.load(motion_path), contact_joints, speed_floor, height_band)
            if predicted is None:
                continue
            hits[species]["n"] += 1
            hits[species]["correct"] += int(predicted == truth[0])
            axis = _AXIS[truth[0]]
            hits[species][axis] += 1
            hits[species][axis + "_correct"] += int(predicted == truth[0])
            # The floor for the generated angle error: how far a REAL clip of
            # this species sits from the heading it is labeled with. Without it
            # a generated "12 degrees off" cannot be told from how the corpus
            # itself moves. Per direction, because a rig can be clean on one
            # heading and not another.
            hits[species]["abs_error_sum"] += abs(heading_error(angle, truth[0]))
            hits[species][truth[0] + "_n"] += 1
            hits[species][truth[0] + "_abs_error_sum"] += abs(
                heading_error(angle, truth[0]))
    return hits


def run_score_auto(args) -> int:
    from data_loaders.truebones.truebones_utils.cond_schema import load_cond

    output_dir = Path(args.output_dir)
    rows = _load_manifest(output_dir)
    if not args.cond_path:
        raise SystemExit("ERROR: --auto needs --cond_path (it reads each species' "
                         "contact joints out of cond).")
    cond = load_cond(args.cond_path)
    lookup, _ = _contact_joints_by_species(cond)

    reference_dirs = [d.strip() for d in args.reference.split(",") if d.strip()]
    # Only the species this run actually prompts: calibrating the whole corpus
    # would read thousands of clips to print a table nobody needs.
    wanted = {lookup(row["species"])[0] for row in rows}
    wanted.discard(None)
    # Trust is per axis, not per species: the estimator reports body-frame
    # travel, and a species is only scoreable on an axis if that axis's own
    # reference clips reproduce their labels. A species with only forward/
    # backward clips proves the estimator on that axis and nothing else --
    # scoring its left/right prompts with an unverified estimator would read a
    # curving turn as forward.
    trusted = {"forward/backward": set(), "left/right": set()}
    calibration = {}
    if reference_dirs:
        calibration = calibrate(reference_dirs, cond, args.speed_floor,
                                args.height_band, species_filter=wanted)
        print("calibration on the real corpus (the estimator must reproduce the "
              "labels a species already carries):")
        print(f"   {'species':<34} {'n':>4} {'top-1':>7}  {'FB':>7} {'LR':>7}   "
              f"{'floor':>9}   verdict")
        for species in sorted(wanted, key=str):
            counts = calibration.get(species, Counter())
            n = counts["n"]
            accuracy = counts["correct"] / n if n else 0.0
            fb = counts["forward/backward"]
            lr = counts["left/right"]
            fb_text = f"{counts['forward/backward_correct']/fb:6.1%}" if fb else "     -"
            lr_text = f"{counts['left/right_correct']/lr:6.1%}" if lr else "     -"
            if not n:
                print(f"   {str(species):<34} {0:>4} {'-':>6}  {'-':>6} {'-':>6}   "
                      f"{'-':>9}   no usable reference clip")
                continue
            trusted_axes = []
            for axis, short in (("forward/backward", "FB"), ("left/right", "LR")):
                ax_n = counts[axis]
                if ax_n >= args.min_calibration_clips and \
                        counts[axis + "_correct"] / ax_n >= args.calibration_threshold:
                    trusted[axis].add(species)
                    trusted_axes.append(short)
            if not trusted_axes:
                if fb >= args.min_calibration_clips or lr >= args.min_calibration_clips:
                    verdict = "REFUSED -- estimator can't reproduce this species' labels"
                else:
                    verdict = f"too few clips (<{args.min_calibration_clips} per axis)"
            elif len(trusted_axes) == 2:
                verdict = "OK (both axes)"
            else:
                other = "LR" if trusted_axes[0] == "FB" else "FB"
                verdict = f"OK ({trusted_axes[0]} only) -- {other} unverified"
            print(f"   {str(species):<34} {n:>4} {accuracy:>6.1%}  {fb_text} {lr_text}   "
                  f"{counts['abs_error_sum'] / n:5.1f}deg off   {verdict}")
    else:
        print("[warn] no --reference given, so nothing was calibrated. The estimator "
              "is only known to be exact on four-heading humanoid rigs; on a "
              "quadruped or a dragon 'left' means a curving turn and this will "
              "score it as forward. Pass --reference unless you have checked.")

    graded, errors, skipped = [], {}, Counter()
    for row in rows:
        blockers = prompt_is_measurable(row.get("action_label", ""))
        if blockers:
            skipped[f"prompt names {blockers} -- not a travel heading"] += 1
            continue
        species, contact_joints = lookup(row["species"])
        if not contact_joints:
            skipped[f"{row['species']}: no contact joints in cond"] += 1
            continue
        axis = _AXIS[row["direction"]]
        if reference_dirs and species not in trusted[axis]:
            skipped[f"{species}: not calibrated for {axis} -- score it by hand"] += 1
            continue
        motion_path = output_dir / row["clip"]
        if not motion_path.exists():
            skipped["missing clip file"] += 1
            continue
        predicted, _, angle = measure_clip(
            np.load(motion_path), contact_joints, args.speed_floor, args.height_band)
        # Below the speed floor the sample barely travels, so it has no heading to
        # read. That is a real outcome of the generation, not a measurement gap:
        # it is counted as 'unclear' rather than dropped.
        graded.append((row, predicted if predicted is not None else "unclear"))
        if angle is not None:
            # Kept even when the cardinal is wrong: a sample that missed by 100
            # degrees belongs in the error distribution, and dropping it would
            # flatter exactly the runs that need flattering least.
            errors[row["sample_id"]] = heading_error(angle, row["direction"])

    for reason, count in skipped.items():
        print(f"[skip] {count} clip(s): {reason}")
    if not graded:
        raise SystemExit("ERROR: nothing measurable. Score this run by hand "
                         "(`sheet`, then `score` without --auto).")
    floor = {
        species: {
            direction: (counts[direction + "_abs_error_sum"] / counts[direction + "_n"],
                        counts[direction + "_n"])
            for direction in DIRECTION_VOCAB if counts[direction + "_n"]
        }
        for species, counts in calibration.items()
    }
    _report(graded, source="geometric, --auto", errors=errors, floor=floor)
    print("\nNote: 'mixed' is always 0% here -- a per-clip measurement cannot see a")
    print("blend, only a heading. Read the spread instead: `phase` reports how tightly")
    print("the samples of one prompt agree, and a human pass can still see the rest.")
    return 0


def run_score(args) -> int:
    if getattr(args, "auto", False):
        return run_score_auto(args)
    output_dir = Path(args.output_dir)
    rows = {row["sample_id"]: row for row in _load_manifest(output_dir)}
    answers_path = Path(args.answers) if args.answers else output_dir / SHEET_NAME
    if not answers_path.exists():
        raise SystemExit(f"ERROR: {answers_path} not found. Run `sheet` first.")

    graded = []
    unknown_ids, blank, bad_answer = [], 0, Counter()
    with open(answers_path, "r", encoding="utf-8", newline="") as handle:
        for entry in csv.DictReader(handle):
            answer = (entry.get("answer") or "").strip().lower()
            if not answer:
                blank += 1
                continue
            row = rows.get(entry.get("sample_id", ""))
            if row is None:
                unknown_ids.append(entry.get("sample_id", ""))
                continue
            if answer not in ANSWERS:
                bad_answer[answer] += 1
                continue
            graded.append((row, answer))

    if blank:
        print(f"[warn] {blank} row(s) not annotated yet -- excluded")
    if unknown_ids:
        print(f"[warn] {len(unknown_ids)} answer row(s) match no manifest entry: "
              f"{unknown_ids[:5]}")
    if bad_answer:
        print(f"[warn] unrecognized answers ignored: {dict(bad_answer)}")
    if not graded:
        raise SystemExit("ERROR: nothing annotated yet.")
    _report(graded, source="human blind read")
    return 0


# ---------------------------------------------------------------------------
# phase -- the automatable secondary metric
# ---------------------------------------------------------------------------

def contact_phase_offset(motion: np.ndarray, left_joints, right_joints):
    """Left-vs-right contact phase offset, in cycles on [-0.5, 0.5), or ``None``.

    Do NOT read the absolute value as "antiphase or not". The dominant bin can
    land on the stride frequency or on the step frequency depending on how the
    contact detector chopped that clip, so real walks in this corpus come out
    bimodal around 0 and around +-0.5. The signed value is not comparable across
    rigs either -- the left/right assignment comes from the cond symmetry pair
    order, which flips the sign but nothing else.

    What IS comparable, and what this metric is for, is AGREEMENT within one
    prompt: the same species, action and heading generated N times should land on
    one offset (in the corpus, identical clips agree to a circular variance of
    1e-5); samples that mix gait modes do not.

    Read off the dominant frequency bin of the two contact signals rather than by
    counting onsets, because a generated contact channel can flicker and an
    onset counter turns one flicker into a spurious half cycle.

    ``None`` when a foot never leaves (or never touches) the ground, which is the
    honest answer: there is no phase in a constant signal.
    """
    left = motion[:, list(left_joints), CONTACT_CHANNEL].mean(axis=1)
    right = motion[:, list(right_joints), CONTACT_CHANNEL].mean(axis=1)
    left = left - left.mean()
    right = right - right.mean()
    if left.std() <= 1e-6 or right.std() <= 1e-6 or left.size < 4:
        return None
    spectrum_left = np.fft.rfft(left)
    spectrum_right = np.fft.rfft(right)
    # Bin 0 is the (removed) mean; the dominant shared bin is the gait frequency.
    bin_index = int(np.argmax(np.abs(spectrum_left[1:]) + np.abs(spectrum_right[1:]))) + 1
    delta = np.angle(spectrum_right[bin_index]) - np.angle(spectrum_left[bin_index])
    return ((delta / (2.0 * math.pi)) + 0.5) % 1.0 - 0.5


def circular_spread(offsets) -> dict:
    """Circular mean / variance of a set of phase offsets (period = 1 cycle).

    Circular, not linear: -0.49 and +0.49 cycles are almost the same gait, and a
    linear variance would call them maximally different.
    """
    angles = np.asarray(offsets, dtype=np.float64) * 2.0 * math.pi
    resultant = np.exp(1j * angles).mean()
    length = float(abs(resultant))
    return {
        "n": int(len(offsets)),
        "mean": float(np.angle(resultant) / (2.0 * math.pi)),
        # 1 - R: 0 = every clip agrees, 1 = no preferred offset at all.
        "circular_variance": 1.0 - length,
    }


def _contact_side_indices(cond_entry):
    """(first-side contact joints, other-side contact joints), or ``None``.

    Pairs the cond entry's contact joints through its symmetry pairs, so this
    works on any rig that has a mirror axis -- the pair list is the same one the
    dataset uses, not a name heuristic. Which side lands first is whatever order
    that list happens to use, which only flips the sign of the offset.
    """
    contact = [int(index) for index in (cond_entry.get("contact_joints") or [])]
    if not contact:
        return None
    contact_set = set(contact)
    left, right = [], []
    for pair in (cond_entry.get("symmetric_joint_pairs") or []):
        first, second = int(pair[0]), int(pair[1])
        if first in contact_set and second in contact_set:
            left.append(first)
            right.append(second)
    if not left:
        return None
    return left, right


def _reference_offsets(dataset_dirs, cond, sides_for_clip):
    """Per (canonical species, frozenset of label words) contact offsets from the corpus.

    A generated circular variance means nothing on its own -- the scale is set by
    the contact detector and the clip length. The comparison that carries
    information is against the REAL clips for the same prompt, so this collects
    them once and ``run_phase`` prints the two side by side.
    """
    from data_loaders.truebones.truebones_utils.motion_labels import load_action_labels
    from data_loaders.truebones.truebones_utils.dataset_sources import (
        infer_namespace_from_root,
    )

    reference: dict = defaultdict(list)
    for dataset_dir in dataset_dirs:
        dataset_dir = Path(dataset_dir)
        motions_dir = dataset_dir / "motions"
        if not motions_dir.is_dir():
            continue
        # Clips here belong to this dataset, so resolve their species within its
        # namespace -- see sides_for_clip for why a bare name is unsafe across
        # datasets.
        namespace = infer_namespace_from_root(dataset_dir)
        try:
            labels = load_action_labels(dataset_dir)
        except FileNotFoundError:
            print(f"[warn] no action_labels.jsonl in {dataset_dir} -- skipping")
            continue
        for clip_name, entry in labels.items():
            label = entry["action_label"]
            motion_path = motions_dir / clip_name
            if not label or not motion_path.exists():
                continue
            species_key, sides = sides_for_clip(clip_name, namespace=namespace)
            if sides is None:
                continue
            offset = contact_phase_offset(np.load(motion_path), sides[0], sides[1])
            if offset is None:
                continue
            reference[(species_key, frozenset(label.split(", ")))].append(offset)
    return reference


def _reference_for_prompt(reference, species_key, label):
    """Corpus offsets for every clip of *species_key* whose label covers *label*.

    Superset, not equality: the corpus writes "walk, retreat, backward" where the
    prompt says "walk, backward", and those are the same request with one extra
    word of description.
    """
    wanted = frozenset(label.split(", "))
    pooled = []
    for (key, words), offsets in reference.items():
        if key == species_key and wanted <= words:
            pooled.extend(offsets)
    return pooled


def run_phase(args) -> int:
    from data_loaders.truebones.truebones_utils.cond_schema import load_cond
    from data_loaders.truebones.truebones_utils.dataset_sources import resolve_species_key

    target = Path(args.output_dir)
    manifest_path = target / MANIFEST_NAME
    if manifest_path.exists():
        rows = _load_manifest(target)
        clips = [(row, target / row["clip"]) for row in rows]
    else:
        clips = [({"species": args.species or "", "action": "", "direction": "",
                   "cfg_scale": float("nan")}, path)
                 for path in sorted(target.rglob("*.npy"))]
        if not clips:
            raise SystemExit(f"ERROR: no .npy under {target} and no {MANIFEST_NAME}.")
        if not args.species:
            raise SystemExit("ERROR: --species is required when there is no manifest.")

    cond = load_cond(args.cond_path)
    side_cache: dict = {}

    def sides_for(species):
        if species not in side_cache:
            key = resolve_species_key(cond, species) or species
            entry = cond.get(key)
            side_cache[species] = (key, _contact_side_indices(entry) if entry else None)
        return side_cache[species]

    def sides_for_clip(clip_name, namespace=None):
        # Corpus clips are named "<species>_<motion>_<n>.npy" and the species part
        # can itself hold underscores, so try the longest prefix that resolves.
        # When *namespace* is given (multi-dataset reference), index the merged
        # cond by "<namespace>/<species>" instead of a bare name: the same species
        # name can exist in two namespaces with different joint counts, and a bare
        # lookup would resolve the first one -- wrong side pairs, or an
        # out-of-bounds index when the counts differ.
        parts = Path(clip_name).stem.split("_")
        for cut in range(len(parts) - 1, 0, -1):
            prefix = "_".join(parts[:cut])
            if namespace is not None:
                entry = cond.get(f"{namespace}/{prefix}")
                if entry is not None:
                    sides = _contact_side_indices(entry)
                    if sides is not None:
                        return f"{namespace}/{prefix}", sides
                continue
            key, sides = sides_for(prefix)
            if sides is not None:
                return key, sides
        return None, None

    groups: dict = defaultdict(list)
    labels_by_group: dict = {}
    skipped = Counter()
    for row, path in clips:
        species = row["species"]
        species_key, sides = sides_for(species)
        if sides is None:
            skipped["no left/right contact pair in cond"] += 1
            continue
        offset = contact_phase_offset(np.load(path), sides[0], sides[1])
        if offset is None:
            skipped["contact channel is constant"] += 1
            continue
        group_key = (species, row["action"], row["direction"], row["cfg_scale"])
        groups[group_key].append(offset)
        labels_by_group[group_key] = (species_key, row.get("action_label", ""))

    if skipped:
        for reason, count in skipped.items():
            print(f"[warn] {count} clip(s) skipped: {reason}")
    if not groups:
        raise SystemExit("ERROR: nothing measurable.")

    reference = {}
    if args.reference:
        reference = _reference_offsets(
            [d.strip() for d in args.reference.split(",") if d.strip()],
            cond, sides_for_clip)
        print(f"[ref ] {sum(len(v) for v in reference.values())} corpus clip(s) in "
              f"{len(reference)} (species, label) bucket(s)")

    print("\nLeft/right contact phase offset, in cycles. circvar is the metric:")
    print("0 = every sample lands on one gait, 1 = no preferred phase at all.")
    header = (f"{'species':<16} {'action':<10} {'dir':<9} {'cfg':>5} {'n':>4} "
              f"{'mean':>7} {'circvar':>8}")
    if reference:
        header += f"  | {'ref n':>6} {'ref circvar':>12}"
    print(header)
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2], k[3])):
        species, action, direction, scale = key
        stats = circular_spread(groups[key])
        scale_text = "-" if scale != scale else f"{scale:g}"
        line = (f"{species:<16} {action:<10} {direction:<9} {scale_text:>5} "
                f"{stats['n']:>4} {stats['mean']:>7.3f} {stats['circular_variance']:>8.3f}")
        if reference:
            species_key, label = labels_by_group[key]
            pooled = _reference_for_prompt(reference, species_key, label) if label else []
            if pooled:
                ref_stats = circular_spread(pooled)
                line += f"  | {ref_stats['n']:>6} {ref_stats['circular_variance']:>12.3f}"
            else:
                line += f"  | {'-':>6} {'-':>12}"
        print(line)
    print("\nRead it as generated-vs-corpus, not as an absolute: the scale is set by")
    print("the contact detector and the clip length, so only the comparison carries")
    print("information. Rising circvar under a fixed prompt is the measurable face of")
    print("mode mixing, and it needs no root translation -- so unlike travel direction")
    print("it stays measurable on the in-place clips that make up much of this corpus.")
    return 0


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sweep = sub.add_parser("sweep", help="generate the prompt grid at each CFG scale")
    sweep.add_argument("--model_path", required=True)
    sweep.add_argument("--output_dir", required=True)
    sweep.add_argument("--species", default="KI_Human",
                       help="Comma-separated species. Pick ones whose corpus covers all "
                            "four headings -- a species that never trained on a heading "
                            "cannot be asked to follow it. Default: KI_Human.")
    sweep.add_argument("--actions", default="walk;run",
                       help="Semicolon-separated action-word groups (each may itself be "
                            "comma-separated, e.g. 'run, sprint'). Default: 'walk;run'.")
    sweep.add_argument("--directions", default=",".join(DIRECTION_VOCAB))
    sweep.add_argument("--cfg_scales", default="1,2,3,4,6",
                       help="--action_label_cfg_scale values to sweep. The SHAPE of the "
                            "resulting curve is the R2/FiLM decision. Default: 1,2,3,4,6.")
    sweep.add_argument("--batch_size", type=int, default=16,
                       help="Clips per prompt (16-32 is the protocol). Default: 16.")
    sweep.add_argument("--num_frames", type=int, default=None)
    sweep.add_argument("--seed", type=int, default=10,
                       help="Fixed across the whole grid, so two prompts differ by the "
                            "prompt and nothing else. Default: 10.")
    sweep.add_argument("--cond_path", default="")
    sweep.add_argument("--force", action="store_true",
                       help="Re-generate prompts whose output dir is already populated.")
    sweep.add_argument("generate_args", nargs="*",
                       help="Extra args passed straight through to sample.generate.")
    sweep.set_defaults(func=run_sweep)

    sheet = sub.add_parser("sheet", help="build the blinded annotation CSV")
    sheet.add_argument("output_dir")
    sheet.add_argument("--shuffle_seed", type=int, default=0)
    sheet.add_argument("--force", action="store_true")
    sheet.set_defaults(func=run_sheet)

    score = sub.add_parser("score", help="score a sweep, automatically or from a CSV")
    score.add_argument("output_dir")
    score.add_argument("--answers", default="",
                       help=f"Human path: the filled-in CSV "
                            f"(default: <output_dir>/{SHEET_NAME}).")
    score.add_argument("--auto", action="store_true",
                       help="Measure each clip's heading geometrically instead of "
                            "reading annotator answers. Exact on the four-heading "
                            "humanoid rigs this evaluation prompts; pass --reference "
                            "so it can prove that on your species before trusting it.")
    score.add_argument("--cond_path", default="",
                       help="--auto: cond.npy, for each species' contact joints.")
    score.add_argument("--reference", default="",
                       help="--auto: comma-separated processed dataset dirs to "
                            "calibrate on. A species whose own labeled clips the "
                            "estimator cannot reproduce is refused rather than "
                            "scored wrong.")
    score.add_argument("--speed_floor", type=float, default=DEFAULT_SPEED_FLOOR,
                       help="--auto: |travel| per frame over body height, below which "
                            "a clip counts as having no heading (scored 'unclear'). "
                            f"Default {DEFAULT_SPEED_FLOOR}.")
    score.add_argument("--height_band", type=float, default=DEFAULT_HEIGHT_BAND,
                       help="--auto: how close to its floor a foot must sit to count "
                            f"as support. Default {DEFAULT_HEIGHT_BAND}.")
    score.add_argument("--calibration_threshold", type=float,
                       default=DEFAULT_CALIBRATION_THRESHOLD,
                       help="--auto: accuracy a species must reach on its own corpus "
                            f"clips to be scored. Default {DEFAULT_CALIBRATION_THRESHOLD}.")
    score.add_argument("--min_calibration_clips", type=int, default=4,
                       help="--auto: reference clips a species needs on an axis "
                            "before that axis is trusted. Default 4.")
    score.set_defaults(func=run_score)

    phase = sub.add_parser("phase", help="left/right contact phase spread (automatable)")
    phase.add_argument("output_dir")
    phase.add_argument("--cond_path", required=True)
    phase.add_argument("--species", default="",
                       help="Required only when the directory has no manifest.")
    phase.add_argument("--reference", default="",
                       help="Comma-separated processed dataset dirs. Prints the same "
                            "statistic over the REAL clips of each prompt next to the "
                            "generated one, which is the only way the number reads as "
                            "good or bad.")
    phase.set_defaults(func=run_phase)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
