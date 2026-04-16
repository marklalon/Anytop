#!/usr/bin/env python
"""
Lightweight Motion Quality Evaluator
=====================================

Evaluates one or more motion .npy files (shape T×J×13) and prints a
quality report for each, with an optional JSON/CSV summary.

Usage examples
--------------
# Single file
python eval/evaluate_motion_quality.py path/to/motion.npy

# All pred motions in a trial (glob)
python eval/evaluate_motion_quality.py \
    "outputs/stage1_*/trials/trial_00/*/generated_prediction.npy"

# Compare clean vs pred for every sample in trial_00
python eval/evaluate_motion_quality.py \
    "outputs/stage1_tiny_overfit_all_move_clean_s100000/stage1_sampling_eval/trials/trial_00/*/clean_target.npy" \
    "outputs/stage1_tiny_overfit_all_move_clean_s100000/stage1_sampling_eval/trials/trial_00/*/generated_prediction.npy"

# With reference statistics and JSON output
python eval/evaluate_motion_quality.py \
    "outputs/stage1_*/trials/*/sample_*/*.npy" \
    --dataset-dir dataset/truebones/zoo/truebones_processed \
    --cond-file   dataset/truebones/zoo/truebones_processed/cond.npy \
    --output-json eval_report.json \
    --output-csv  eval_report.csv

# Quiet mode (summary table only, no per-file reports)
python eval/evaluate_motion_quality.py "path/**/*.npy" --quiet
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path when running from any directory
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent          # eval/
_ANYTOP_DIR = _SCRIPT_DIR.parent                       # Anytop/
if str(_ANYTOP_DIR) not in sys.path:
    sys.path.insert(0, str(_ANYTOP_DIR))

from eval.motion_quality.scorer import LightweightMotionQualityScorer, MotionQualityReport
from eval.motion_quality.reference_stats import get_reference_stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_object_type(npy_path: str, cond: Optional[dict]) -> Optional[str]:
    """
    Try to infer the skeleton/object type from the file path or parent dirs.

    Strategy:
      1. If cond is provided, check if any key appears as a path component.
      2. Parse "sample_NNN_<ObjectType>" directory name convention.
      3. Parse "<ObjectType>___<AnimName>_<id>.npy" filename convention.
    """
    parts = Path(npy_path).parts

    if cond:
        # Longest-match first so "BrownBear" wins over "Bear"
        candidates = sorted(cond.keys(), key=len, reverse=True)
        for part in parts:
            for name in candidates:
                if part == name or part.startswith(name + "_") or part.endswith("_" + name):
                    return name
        # Also check substring match in parent directory names
        for part in parts:
            for name in candidates:
                if name in part:
                    return name

    # Fallback: parse filename "<ObjectType>___<rest>"
    stem = Path(npy_path).stem
    m = re.match(r"^([A-Za-z][A-Za-z0-9]*)___", stem)
    if m:
        return m.group(1)

    return None


def _expand_patterns(patterns: List[str]) -> List[str]:
    """Expand glob patterns into sorted unique file paths."""
    paths: list[str] = []
    for pat in patterns:
        expanded = sorted(glob.glob(pat, recursive=True))
        if not expanded:
            print(f"[warn] No files matched: {pat}", file=sys.stderr)
        paths.extend(expanded)
    # Deduplicate preserving order
    seen: set[str] = set()
    result: list[str] = []
    for p in paths:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            result.append(ap)
    return result


def _load_motion(path: str) -> Optional[np.ndarray]:
    try:
        m = np.load(path)
        if m.ndim == 3 and m.shape[-1] == 13:
            return m.astype(np.float32)
        print(
            f"[warn] Skipping {path}: expected shape (T,J,13), got {m.shape}",
            file=sys.stderr,
        )
        return None
    except Exception as e:
        print(f"[warn] Failed to load {path}: {e}", file=sys.stderr)
        return None


def _build_scorer(
    object_type: Optional[str],
    dataset_dir: Optional[str],
    cache_path: Optional[str],
) -> LightweightMotionQualityScorer:
    ref = None
    if object_type and dataset_dir:
        ref = get_reference_stats(
            object_type,
            dataset_dir,
            cache_path=cache_path,
        )
    return LightweightMotionQualityScorer(ref_stats=ref)


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _bar(score: float, width: int = 20) -> str:
    filled = round(score * width)
    return "#" * filled + "." * (width - filled)


def _color(score: float, text: str) -> str:
    """ANSI colour: red < 0.4, yellow < 0.7, green >= 0.7."""
    if score < 0.4:
        code = "\033[91m"
    elif score < 0.7:
        code = "\033[93m"
    else:
        code = "\033[92m"
    return f"{code}{text}\033[0m"


def _print_report(path: str, report: MotionQualityReport, use_color: bool = True) -> None:
    s = report
    clr = _color if use_color else (lambda v, t: t)

    print(f"\n{'-'*70}")
    print(f"  {path}")
    print(f"{'-'*70}")
    print(f"  Object: {s.object_type or 'unknown'}  |  "
          f"Frames x Joints: {s.n_frames} x {s.n_joints}  |  "
          f"Reference: {'yes' if s.has_reference else 'no'}")
    print()

    rows = [
        ("Rotation 6D consistency",   s.rotation_6d_consistency,   "w=0.45 [primary]"),
        ("Jerk smoothness",            s.jerk_smoothness,           "w=0.275"),
        ("Temporal variance",          s.temporal_variance,         "w=0.275"),
    ]

    for label, score, note in rows:
        bar   = _bar(score)
        score_str = clr(score, f"{score:.3f}")
        print(f"  {label:<32s} {score_str}  {bar}  {note}")

    print()
    total_str = clr(s.total_score, f"{s.total_score:.3f}")
    total_bar = _bar(s.total_score)
    print(f"  {'TOTAL QUALITY SCORE':<32s} {total_str}  {total_bar}")
    print()

    print("  Raw diagnostics:")
    for k, v in s.raw.items():
        print(f"    {k:<44s} {v:.6f}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(records: List[Tuple[str, MotionQualityReport]], use_color: bool = True) -> None:
    clr = _color if use_color else (lambda v, t: t)

    print(f"\n{'='*90}")
    print("  SUMMARY")
    print(f"{'='*90}")
    hdr = (
        f"  {'File':<45s}  {'Type':<14s}  "
        f"{'rot6d':>5s}  {'jerk':>5s}  {'var':>5s}  {'TOTAL':>6s}"
    )
    print(hdr)
    print(f"  {'-'*45}  {'-'*14}  " + "  ".join(["-----"] * 3) + "  ------")

    for path, r in records:
        short = path[-44:] if len(path) > 45 else path
        otype = (r.object_type or "?")[:14]
        total_str = clr(r.total_score, f"{r.total_score:.3f}")
        print(
            f"  {short:<45s}  {otype:<14s}  "
            f"{r.rotation_6d_consistency:5.3f}  "
            f"{r.jerk_smoothness:5.3f}  "
            f"{r.temporal_variance:5.3f}  "
            f"{total_str:>6s}"
        )

    scores = [r.total_score for _, r in records]
    if scores:
        print(f"\n  Files evaluated : {len(scores)}")
        print(f"  Score  mean     : {np.mean(scores):.3f}")
        print(f"  Score  std      : {np.std(scores):.3f}")
        print(f"  Score  min/max  : {np.min(scores):.3f} / {np.max(scores):.3f}")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _write_json(records: List[Tuple[str, MotionQualityReport]], out_path: str) -> None:
    data = [
        {"file": path, **report.as_dict()}
        for path, report in records
    ]
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    print(f"\n[info] JSON report written -> {out_path}")


def _write_csv(records: List[Tuple[str, MotionQualityReport]], out_path: str) -> None:
    if not records:
        return
    fieldnames = [
        "file", "object_type", "n_frames", "n_joints", "has_reference",
        "total_score",
        "rotation_6d_consistency",
        "jerk_smoothness", "temporal_variance",
    ] + list(records[0][1].raw.keys())

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for path, r in records:
            row = {
                "file":                    path,
                "object_type":             r.object_type or "",
                "n_frames":                r.n_frames,
                "n_joints":                r.n_joints,
                "has_reference":           int(r.has_reference),
                "total_score":             r.total_score,
                "rotation_6d_consistency": r.rotation_6d_consistency,
                "jerk_smoothness":         r.jerk_smoothness,
                "temporal_variance":       r.temporal_variance,
            }
            row.update(r.raw)
            writer.writerow(row)
    print(f"[info] CSV  report written -> {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evaluate_motion_quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Lightweight Motion Quality Evaluator
            ─────────────────────────────────────
            Scores .npy motion files (T×J×13) without any trained model.

            Supports glob patterns with wildcards.  Quote patterns so your
            shell does not expand them before Python can.
        """),
        epilog=textwrap.dedent("""\
            Examples:
              # Single file
              python eval/evaluate_motion_quality.py motion.npy

              # All clean targets in trial_00
              python eval/evaluate_motion_quality.py \\
                "outputs/**/trials/trial_00/*/clean_target.npy"

              # Full comparison with reference stats + CSV output
              python eval/evaluate_motion_quality.py \\
                "outputs/**/trial_00/**/*.npy" \\
                --dataset-dir dataset/truebones/zoo/truebones_processed \\
                --cond-file   dataset/truebones/zoo/truebones_processed/cond.npy \\
                --output-csv  report.csv --quiet
        """),
    )
    p.add_argument(
        "patterns",
        nargs="+",
        metavar="PATTERN",
        help=(
            "One or more file paths or glob patterns (e.g. "
            "'outputs/**/*/clean_target.npy').  Patterns are expanded "
            "recursively when ** is present."
        ),
    )
    p.add_argument(
        "--dataset-dir",
        default=None,
        metavar="DIR",
        help=(
            "Path to the truebones_processed directory "
            "(contains motions/ sub-dir with reference .npy files). "
            "Used to compute per-skeleton reference statistics that "
            "calibrate the supporting sub-scores."
        ),
    )
    p.add_argument(
        "--cond-file",
        default=None,
        metavar="FILE",
        help=(
            "Path to cond.npy (maps skeleton names → metadata / mean / std). "
            "Used for object-type inference when the filename alone is ambiguous."
        ),
    )
    p.add_argument(
        "--ref-cache",
        default=None,
        metavar="FILE",
        help=(
            "Pickle file to cache computed reference statistics.  "
            "Subsequent runs load from this file instead of re-computing.  "
            "Defaults to <dataset-dir>/cache/motion_quality_ref_stats.pkl "
            "when --dataset-dir is given."
        ),
    )
    p.add_argument(
        "--output-json",
        default=None,
        metavar="FILE",
        help="Write full JSON report to this file.",
    )
    p.add_argument(
        "--output-csv",
        default=None,
        metavar="FILE",
        help="Write CSV summary to this file.",
    )
    p.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-file verbose reports; print summary table only.",
    )
    p.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colour in terminal output.",
    )
    p.add_argument(
        "--sort-by",
        choices=["total", "rot", "file"],
        default="file",
        help="Sort summary table by this column (default: file).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    use_color = not args.no_color and sys.stdout.isatty()

    # Load cond.npy for type inference
    cond: Optional[dict] = None
    if args.cond_file:
        if not os.path.isfile(args.cond_file):
            print(f"[warn] --cond-file not found: {args.cond_file}", file=sys.stderr)
        else:
            cond = np.load(args.cond_file, allow_pickle=True).item()

    # Default ref cache path
    cache_path = args.ref_cache
    if cache_path is None and args.dataset_dir:
        cache_path = os.path.join(
            args.dataset_dir, "cache", "motion_quality_ref_stats.pkl"
        )

    # Expand glob patterns
    paths = _expand_patterns(args.patterns)
    if not paths:
        print("[error] No .npy files found matching the given patterns.", file=sys.stderr)
        return 1

    print(f"Evaluating {len(paths)} motion file(s)...\n")

    # Scorer cache (one per object_type)
    scorer_cache: Dict[Optional[str], LightweightMotionQualityScorer] = {}
    records: List[Tuple[str, MotionQualityReport]] = []

    for path in paths:
        motion = _load_motion(path)
        if motion is None:
            continue

        object_type = _infer_object_type(path, cond)

        if object_type not in scorer_cache:
            scorer_cache[object_type] = _build_scorer(
                object_type, args.dataset_dir, cache_path
            )
        scorer = scorer_cache[object_type]

        report = scorer.score(motion, object_type=object_type)
        records.append((path, report))

        if not args.quiet:
            _print_report(path, report, use_color=use_color)

    if not records:
        print("[error] No valid motion files were evaluated.", file=sys.stderr)
        return 1

    # Sort
    sort_key = {
        "total": lambda x: -x[1].total_score,
        "rot":   lambda x: -x[1].rotation_6d_consistency,
        "file":  lambda x: x[0],
    }[args.sort_by]
    records.sort(key=sort_key)

    _print_summary(records, use_color=use_color)

    if args.output_json:
        _write_json(records, args.output_json)
    if args.output_csv:
        _write_csv(records, args.output_csv)

    return 0


if __name__ == "__main__":
    sys.exit(main())
