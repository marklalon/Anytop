#!/usr/bin/env python3
"""Prefill ``is_loop`` in action_labels.jsonl from the source animations.

The loop verdict is a per-clip annotation in ``action_labels.jsonl``, next to
the action group and label, and preprocessing REQUIRES it: it shapes the clip's
tensor (the terminal velocity row is the wrap delta for a loop) and it is the
model's loop condition. This tool proposes it, ahead of preprocessing, so the
dataset workflow reads:

    1. drop the source animations (GLB / FBX) into <raw-data-dir>/<Species>/
    2. label every clip in action_labels.jsonl (action_group, action_label)
    3. python tools/prefill_loop_flags.py --dataset-dir D --raw-data-dir R   <- this
    4. verify the proposals in dataset/review (serve.py + index.html)
    5. python preprocess_and_validate.py  (reads the sidecar, never writes it)

The verdict here is preprocessing's own: each species goes through the same
phase-1/phase-2 preparation a build runs -- rest-pose cond, HML alignment,
translation-root contract, locomotion detrend and clamp, resample of short
clips -- and the detector reads the aligned clip exactly where extraction would
(``extract_motion_features_from_aligned_anims`` with ``is_loop=None``). Nothing
is written to motions/, bvhs/ or cond.npy; only the sidecar rows change.

Rules:
    * Only a row WITHOUT ``is_loop`` is filled. A value already there -- an
      earlier proposal or a hand correction -- is an annotation and stays;
      delete the key from a row to have it judged again.
    * ``--rejudge`` re-proposes for every clip of the selected species, except
      rows marked ``"reviewed": true`` (a person signed those off).
    * A species none of whose rows needs a verdict is not loaded at all, so a
      run over a fully annotated dataset is a no-op.
    * A species already in cond.npy keeps its frozen translation root (as an
      incremental build does) and only its pending sources are loaded; a
      species not in cond yet is scanned in full, because the root contract
      is fixed from every clip's transport carrier.
    * A source file with no row in action_labels.jsonl is reported and left
      alone: label it first (step 2). A clip shorter than --filter-min-length
      is reported too -- preprocessing drops it, so its row should go.

Usage:
    python tools/prefill_loop_flags.py --dataset-dir D --raw-data-dir R [--filter GLOB]
                                       [--object-workers N] [--dry-run] [--rejudge]
                                       [--report PATH]

``--report`` writes one JSON line per judged clip with the detector's
diagnostics (wrap gap, tolerance, margin, root XZ closure), borderline clips
first, so the review pass knows where to look.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ANYTOP_DIR = Path(__file__).resolve().parent.parent
if str(ANYTOP_DIR) not in sys.path:
    sys.path.insert(0, str(ANYTOP_DIR))
if str(ANYTOP_DIR.parent) not in sys.path:
    sys.path.insert(0, str(ANYTOP_DIR.parent))

# Package-qualified imports only: a short-name import (via a sys.path entry
# inside the package) would create a SECOND copy of these modules in this
# process, each with its own module globals.
from data_loaders.truebones.truebones_utils import dataset_pipeline  # noqa: E402
from data_loaders.truebones.truebones_utils import dataset_tags  # noqa: E402
from data_loaders.truebones.truebones_utils import ignore_warnings  # noqa: E402
from data_loaders.truebones.truebones_utils.animation_utils import (  # noqa: E402
    compute_motion_loop_diagnostics,
)
from data_loaders.truebones.truebones_utils.cond_schema import load_cond  # noqa: E402
from data_loaders.truebones.truebones_utils.motion_labels import (  # noqa: E402
    LOOP_FLAG_KEY,
    clip_key,
    fill_missing_loop_flags,
    load_action_labels,
)
from data_loaders.truebones.truebones_utils.param_utils import (  # noqa: E402
    ACTION_LABELS_FILE,
    get_dataset_dir,
    get_raw_data_dir,
)


@dataclass
class ObjectPlan:
    """One species to load: which of its clips get a verdict and under which root."""
    object_type: str
    # Every clip a build of this species would write: clip -> source path.
    clips: dict[str, str]
    # The clips whose sidecar row receives this run's verdict.
    pending: set[str]
    frozen_translation_root_index: int | None = None
    frozen_promote_root_depth: int | None = None
    # Sources not to load. Only set under a frozen root: without one the
    # species root is fixed from every clip, so every clip has to be seen.
    skip_source_paths: set[str] | None = None


@dataclass
class ObjectJudgement:
    """What the pipeline said about one species' clips."""
    object_type: str
    verdicts: dict[str, bool] = field(default_factory=dict)
    diagnostics: dict[str, dict] = field(default_factory=dict)
    warn_messages: list[str] = field(default_factory=list)
    motion_errors: list[str] = field(default_factory=list)


# ── selection ──────────────────────────────────────────────────────────────

def _parse_filter_patterns(object_filter: str) -> list[str]:
    if not object_filter:
        return []
    return [p.strip() for p in object_filter.replace(";", ",").split(",") if p.strip()]


def discover_objects(raw_data_dir: str | None, object_filter: str = "") -> tuple[str, ...]:
    """The species directories under the raw data dir, narrowed by ``--filter``."""
    raw_root = Path(get_raw_data_dir(raw_data_dir or None))
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw data directory not found: {raw_root}")
    objects = tuple(sorted(p.name for p in raw_root.iterdir() if p.is_dir()))
    patterns = _parse_filter_patterns(object_filter)
    if not patterns:
        return objects
    return tuple(
        obj for obj in objects
        if any(fnmatch.fnmatch(obj.lower(), pattern.lower()) for pattern in patterns)
    )


def _reviewed_clips(dataset_dir: Path) -> set[str]:
    """Clips whose row carries ``"reviewed": true`` (load_action_labels drops the mark)."""
    reviewed = set()
    for line in (dataset_dir / ACTION_LABELS_FILE).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if isinstance(row, dict) and row.get("reviewed") is True:
            reviewed.add(clip_key(row.get("clip", "")))
    return reviewed


def plan_objects(
    dataset_dir: Path,
    raw_data_dir: str | None,
    objects: tuple[str, ...],
    labels: dict[str, dict],
    *,
    rejudge: bool = False,
) -> tuple[list[ObjectPlan], list[str], set[str]]:
    """Decide which species to load and which rows they fill; name-only.

    Returns the plans (species with at least one pending clip), the warnings
    the scan produced -- a source file with no sidecar row can never be filled,
    and is the labelling step's job -- and every clip name the selected
    species' sources would produce.
    """
    cond_path = dataset_dir / "cond.npy"
    existing_cond = load_cond(cond_path) if cond_path.exists() else {}
    reviewed = _reviewed_clips(dataset_dir) if rejudge else set()

    plans: list[ObjectPlan] = []
    warnings: list[str] = []
    all_clips: set[str] = set()
    for object_type in objects:
        clips = dataset_pipeline.enumerate_object_clips(object_type, raw_data_dir)
        if not clips:
            continue
        all_clips.update(clips)
        pending = set()
        for clip in sorted(clips):
            row = labels.get(clip)
            if row is None:
                warnings.append(
                    f"{clip}: no row in {ACTION_LABELS_FILE} "
                    f"({os.path.basename(clips[clip])}); label it before it can be judged"
                )
                continue
            if rejudge:
                if clip in reviewed:
                    continue
                pending.add(clip)
            elif LOOP_FLAG_KEY not in row:
                pending.add(clip)
        if not pending:
            continue
        frozen_root, promote_depth = dataset_pipeline.frozen_root_contract(existing_cond, object_type)
        skip = None
        if frozen_root is not None:
            skip = {
                os.path.realpath(path) for clip, path in clips.items() if clip not in pending
            } or None
        plans.append(ObjectPlan(
            object_type=object_type,
            clips=clips,
            pending=pending,
            frozen_translation_root_index=frozen_root,
            frozen_promote_root_depth=promote_depth,
            skip_source_paths=skip,
        ))
    return plans, warnings, all_clips


# ── judging ────────────────────────────────────────────────────────────────

def judge_object(
    plan: ObjectPlan,
    *,
    raw_data_dir: str | None,
    locomotion_clips: frozenset,
    filter_min_length: int,
    resample_min_length: int,
) -> ObjectJudgement:
    """Run preprocessing's preparation for one species and read the verdicts off it.

    ``loop_verdicts={}`` hands the encoder no annotation for any clip, so the
    detector judges every one of them on the aligned clip -- the same call, at
    the same point, as a build that had no verdict would have made. Nothing is
    written; the payload is read and dropped.
    """
    payload = dataset_pipeline._prepare_object_outputs_worker(
        plan.object_type,
        None,
        raw_data_dir,
        filter_min_length,
        resample_min_length,
        plan.skip_source_paths,
        plan.frozen_translation_root_index,
        plan.frozen_promote_root_depth,
        locomotion_clips,
        {},
    )
    judgement = ObjectJudgement(object_type=plan.object_type)
    if payload is None:
        return judgement
    judgement.warn_messages = list(payload.get("_warn_messages", []))
    judgement.motion_errors = list(payload.get("motion_errors", []))
    for result in payload["results"]:
        clip = f"{plan.object_type}_{result['action']}"
        judgement.verdicts[clip] = bool(result["is_loop"])
        motion = np.asarray(result["motion"])
        # The detector's own numbers on the tensor a build would have stored:
        # it drops the terminal velocity row itself, so the verdict just made
        # cannot vote on its own diagnostics.
        judgement.diagnostics[clip] = compute_motion_loop_diagnostics(
            motion[..., 0:3],
            root_xz_velocity=motion[..., 9:12],
            translation_root_index=int(result["translation_root_index"]),
        )
        judgement.diagnostics[clip]["frames"] = int(motion.shape[0])
    return judgement


def judge_objects(
    plans: list[ObjectPlan],
    *,
    raw_data_dir: str | None,
    locomotion_clips: frozenset,
    filter_min_length: int,
    resample_min_length: int,
    object_workers: int,
) -> list[ObjectJudgement]:
    """One judgement per plan, in plan order; species run in parallel processes."""
    workers = min(max(1, len(plans)), max(1, int(object_workers)))
    kwargs = dict(
        raw_data_dir=raw_data_dir,
        locomotion_clips=locomotion_clips,
        filter_min_length=filter_min_length,
        resample_min_length=resample_min_length,
    )
    if workers <= 1:
        return [judge_object(plan, **kwargs) for plan in plans]
    judgements: list[ObjectJudgement | None] = [None] * len(plans)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=dataset_tags.configure,
        initargs=dataset_tags.worker_initargs(),
    ) as executor:
        future_to_idx = {
            executor.submit(judge_object, plan, **kwargs): idx
            for idx, plan in enumerate(plans)
        }
        for future in as_completed(future_to_idx):
            judgements[future_to_idx[future]] = future.result()
    return [j for j in judgements if j is not None]


# ── writing ────────────────────────────────────────────────────────────────

def write_report(path: Path, plans: list[ObjectPlan], judgements: list[ObjectJudgement]) -> int:
    """One JSON line per judged clip, borderline verdicts first."""
    plan_by_object = {plan.object_type: plan for plan in plans}
    rows = []
    for judgement in judgements:
        plan = plan_by_object[judgement.object_type]
        for clip, verdict in judgement.verdicts.items():
            diag = judgement.diagnostics[clip]
            rows.append({
                "clip": clip,
                "object_type": judgement.object_type,
                "source": plan.clips.get(clip),
                "pending": clip in plan.pending,
                LOOP_FLAG_KEY: verdict,
                "frames": diag["frames"],
                "wrap_gap": diag["wrap_gap"],
                "effective_tolerance": diag["effective_tolerance"],
                "loop_margin": diag["loop_margin"],
                "is_closed": diag["is_closed"],
                "root_xz_total_disp": diag["root_xz_total_disp"],
                "root_xz_is_closed": diag["root_xz_is_closed"],
            })
    # A margin of 1 is the position-wrap threshold; the closer a clip sits to
    # it, the more a human's eye is worth there.
    rows.sort(key=lambda r: (abs(float(r["loop_margin"]) - 1.0), r["clip"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def collect_verdicts(
    plans: list[ObjectPlan], judgements: list[ObjectJudgement]
) -> tuple[dict[str, bool], list[str]]:
    """The verdicts to write (pending clips only) and the pending clips no result reached."""
    plan_by_object = {plan.object_type: plan for plan in plans}
    verdicts: dict[str, bool] = {}
    unjudged: list[str] = []
    for judgement in judgements:
        plan = plan_by_object[judgement.object_type]
        for clip in sorted(plan.pending):
            if clip in judgement.verdicts:
                verdicts[clip] = judgement.verdicts[clip]
            else:
                unjudged.append(clip)
    return verdicts, unjudged


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Propose is_loop for action_labels.jsonl rows from the source animations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset-dir", default="", help="Processed dataset dir holding action_labels.jsonl (default: the default dataset).")
    parser.add_argument("--raw-data-dir", default="", help="Raw source root, one species per subdirectory (default: the default raw dir).")
    parser.add_argument("--filter", dest="object_filter", default="", help="Comma/semicolon-separated case-insensitive glob(s) of species to consider.")
    parser.add_argument("--object-workers", default=16, type=int, help="Species judged concurrently (default: 16).")
    parser.add_argument("--filter-min-length", default=10, type=int, help="Same as preprocessing: clips shorter than this are dropped (default: 10).")
    parser.add_argument("--resample-min-length", default=20, type=int, help="Same as preprocessing: clips shorter than this are resampled to it (default: 20).")
    parser.add_argument("--species-tags-file", default="", help="Species tag sidecar (default: <dataset-dir>/species_tags.jsonl).")
    parser.add_argument("--chain-forward-joints-file", default="", help="Forward-chain sidecar (default: <dataset-dir>/chain_forward_joints.jsonl).")
    parser.add_argument("--rejudge", action="store_true", help="Re-propose for every clip of the selected species; rows marked reviewed are kept.")
    parser.add_argument("--dry-run", action="store_true", help="Judge and report, write nothing.")
    parser.add_argument("--report", default="", help="Write per-clip detector diagnostics (JSONL) to this path.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.filter_min_length < 0 or args.resample_min_length < 0:
        print("ERROR: --filter-min-length and --resample-min-length must be >= 0")
        return 1
    if args.resample_min_length > 0 and args.resample_min_length <= args.filter_min_length:
        print("ERROR: --resample-min-length must be > --filter-min-length")
        return 1

    dataset_dir = Path(get_dataset_dir(args.dataset_dir or None))
    raw_data_dir = args.raw_data_dir or None

    # The same sidecar configuration a build runs under: the rest-pose cond and
    # the alignment read species tags and the forward-chain sidecar, and the
    # pool initializer replays this into every worker.
    paths = dataset_tags.configure(
        dataset_dir=str(dataset_dir),
        species_tags_file=args.species_tags_file,
        chain_forward_joints_file=args.chain_forward_joints_file,
    )
    print(f"[OK] using species tags: {paths.species_tags}")
    sidecar = ignore_warnings.load(str(dataset_dir))
    if sidecar.path is not None:
        directives = " ".join(f"!{name}" for name in sorted(sidecar.directives))
        print(f"[OK] using ignore_warnings: {sidecar.path}{f' ({directives})' if directives else ''}")

    # Prerequisites, as for a build: both sidecars must exist and be valid.
    dataset_tags.dataset_tags()
    labels = load_action_labels(dataset_dir)
    print(f"[OK] {ACTION_LABELS_FILE}: {len(labels)} row(s), "
          f"{sum(1 for row in labels.values() if LOOP_FLAG_KEY not in row)} without {LOOP_FLAG_KEY}")

    objects = discover_objects(raw_data_dir, args.object_filter)
    if not objects:
        print(f"[INFO] --filter '{args.object_filter}' matched no species under {get_raw_data_dir(raw_data_dir)}")
        return 0

    try:
        plans, warnings, known_clips = plan_objects(
            dataset_dir, raw_data_dir, objects, labels, rejudge=args.rejudge,
        )
    except dataset_pipeline.DatasetPreprocessingError as err:
        for message in err.motion_errors:
            print(f"[ERROR] {message}", file=sys.stderr)
        return 1
    for message in warnings:
        print(f"[WARN] {message}")
    if not args.object_filter:
        # Only a full scan can tell a stale row from one whose species was
        # simply not selected.
        stale = sorted(
            clip for clip, row in labels.items()
            if clip not in known_clips and LOOP_FLAG_KEY not in row
        )
        for clip in stale:
            print(f"[WARN] {clip}: row has no source animation under {get_raw_data_dir(raw_data_dir)}; "
                  f"nothing can judge it")
    if not plans:
        if args.rejudge:
            print("[OK] nothing to do: every clip of the selected species is on a reviewed row "
                  "(un-review it in dataset/review to have it judged again)")
        else:
            print(f"[OK] nothing to do: every clip of the selected species already carries {LOOP_FLAG_KEY}")
        return 0

    pending_total = sum(len(plan.pending) for plan in plans)
    print(f"\nJudging {pending_total} clip(s) across {len(plans)} species: "
          + ", ".join(plan.object_type for plan in plans))
    for plan in plans:
        if plan.frozen_translation_root_index is None:
            print(f"  {plan.object_type}: {len(plan.pending)}/{len(plan.clips)} clip(s) pending; "
                  f"not in cond.npy, scanning every source to fix the species root")
        else:
            print(f"  {plan.object_type}: {len(plan.pending)}/{len(plan.clips)} clip(s) pending; "
                  f"frozen root {plan.frozen_translation_root_index}, loading only the pending sources")
    print()

    locomotion_clips = dataset_pipeline.load_locomotion_clip_names(str(dataset_dir))
    judgements = judge_objects(
        plans,
        raw_data_dir=raw_data_dir,
        locomotion_clips=locomotion_clips,
        filter_min_length=args.filter_min_length,
        resample_min_length=args.resample_min_length,
        object_workers=args.object_workers,
    )

    seen_warnings: set[str] = set()
    for judgement in judgements:
        for message in judgement.warn_messages:
            key = message.strip().lower()
            if key not in seen_warnings:
                seen_warnings.add(key)
                print(f"[WARN] {message}")
    motion_errors = [err for judgement in judgements for err in judgement.motion_errors]
    for err in motion_errors:
        print(err)

    verdicts, unjudged = collect_verdicts(plans, judgements)
    for clip in unjudged:
        print(f"[WARN] {clip}: no clip came out of the pipeline (shorter than "
              f"--filter-min-length {args.filter_min_length}, or its source failed); "
              f"preprocessing will not build it either -- remove the row or set {LOOP_FLAG_KEY} by hand")

    print()
    print("=" * 70)
    print(f"  {LOOP_FLAG_KEY} PROPOSALS ({len(verdicts)} clip(s))")
    print("=" * 70)
    for judgement in judgements:
        judged_here = [clip for clip in sorted(verdicts) if clip in judgement.verdicts]
        if not judged_here:
            continue
        loops = sum(1 for clip in judged_here if verdicts[clip])
        print(f"  {judgement.object_type}: {len(judged_here)} judged, {loops} loop / {len(judged_here) - loops} one-shot")
        for clip in judged_here:
            previous = labels[clip].get(LOOP_FLAG_KEY)
            margin = judgement.diagnostics[clip]["loop_margin"]
            change = ""
            if args.rejudge and previous is not None and previous != verdicts[clip]:
                change = f"   (was {previous})"
            print(f"    {'loop    ' if verdicts[clip] else 'one-shot'}  margin {margin:6.2f}  {clip}{change}")

    if args.report:
        count = write_report(Path(args.report), plans, judgements)
        print(f"\n[OK] wrote {count} diagnostic row(s) to {args.report}")

    if args.dry_run:
        print(f"\n[OK] --dry-run: {ACTION_LABELS_FILE} left untouched")
    else:
        written = fill_missing_loop_flags(dataset_dir, verdicts, overwrite=args.rejudge)
        print(f"\n[OK] {LOOP_FLAG_KEY} written on {written} row(s) of {dataset_dir / ACTION_LABELS_FILE}"
              + (" (rejudge: unchanged and reviewed rows kept)" if args.rejudge else "")
              + " -- verify them in dataset/review (serve.py)")

    return 1 if motion_errors else 0


if __name__ == "__main__":
    sys.exit(main())
