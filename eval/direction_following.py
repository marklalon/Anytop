#!/usr/bin/env python3
"""
Direction instruction-following evaluation.

The question is narrow: when the prompt names a heading, does the generated
motion go that way? There is no automatic answer to it -- much of this corpus is
animated in place, so a large share of clips have no root translation to measure
a heading from (which is also why training labels could not have their directions
filled in geometrically). So the primary metric is a HUMAN BLIND read, and this
module is the harness around it:

    sweep   run the prompt grid through sample/generate.py at several
            --action_label_cfg_scale values, into one output tree + a manifest
    sheet   turn that manifest into a blinded annotation CSV: shuffled, prompt
            withheld, one row per clip
    score   join the annotator's answers back to the manifest and report the
            metrics, split forward/backward vs left/right and per CFG scale
    phase   the automatable SECONDARY metric: how consistent the left/right
            contact phase offset is (see ContactPhase below)

Report FB and LR separately, always. They are known to fail differently: T5
places "left" and "right" closer together than forward-vs-left, so left/right is
the one pair the text encoder genuinely struggles to separate, and a pooled
number would hide it behind the easy axis. "mixed" is counted on its own too --
it is the direct read on mode collapse, a different failure from picking the
wrong heading.

The SHAPE of accuracy against CFG scale is itself the decision point: rising
monotonically means the additive token is fine and only needed more gain;
saturating early while artifacts grow means the bottleneck is expressivity, the
only condition under which changing the injection style (the FiLM fallback, not
built) is worth trying.

Run the R0 baseline (the pre-refactor checkpoint) before anything else. It needs
no training and without it there is no threshold to judge R1 against.

Usage:
    python -m eval.direction_following sweep --model_path save/.../model.pt \\
        --species KI_Human --output_dir eval_out/R1
    python -m eval.direction_following sheet eval_out/R1
    #   ... a human fills in the 'answer' column of eval_out/R1/annotate.csv ...
    python -m eval.direction_following score eval_out/R1
    python -m eval.direction_following phase eval_out/R1
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

MANIFEST_NAME = "manifest.jsonl"
SHEET_NAME = "annotate.csv"

# Feature channel carrying binary foot contact. Unaffected by root-XZ stripping,
# so it is measurable on in-place clips too -- which is the whole reason this is
# the automatable metric and travel direction is not.
CONTACT_CHANNEL = 12


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
    print(f"     next: python -m eval.direction_following sheet {output_dir}")
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

def run_score(args) -> int:
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

    # scale -> axis -> Counter of outcomes
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

    print(f"\nDirection instruction-following  ({len(graded)} annotated clip(s))")
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

    print("\nRead the top-1 column DOWN the cfg scales, per axis:")
    print("   rising steadily          -> the additive token is fine, it only needed gain")
    print("   flat/saturating + artifacts -> expressivity-bound; the only case in which")
    print("                               changing the injection style (FiLM, R2) is worth it")
    print("   left/right alone lagging -> T5 cannot separate the pair; the fix there is")
    print("                               the 2-bit L/R hard input, not the injection style")
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

    reference: dict = defaultdict(list)
    for dataset_dir in dataset_dirs:
        dataset_dir = Path(dataset_dir)
        motions_dir = dataset_dir / "motions"
        if not motions_dir.is_dir():
            continue
        for clip_name, entry in load_action_labels(dataset_dir).items():
            label = entry["action_label"]
            motion_path = motions_dir / clip_name
            if not label or not motion_path.exists():
                continue
            species_key, sides = sides_for_clip(clip_name)
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

    def sides_for_clip(clip_name):
        # Corpus clips are named "<species>_<motion>_<n>.npy" and the species part
        # can itself hold underscores, so try the longest prefix that resolves.
        parts = Path(clip_name).stem.split("_")
        for cut in range(len(parts) - 1, 0, -1):
            key, sides = sides_for("_".join(parts[:cut]))
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

    score = sub.add_parser("score", help="score the filled-in annotation CSV")
    score.add_argument("output_dir")
    score.add_argument("--answers", default="",
                       help=f"Filled-in CSV (default: <output_dir>/{SHEET_NAME}).")
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
