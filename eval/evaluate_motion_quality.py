#!/usr/bin/env python
"""
Low-Shot Weighted-Reference Motion Quality Evaluator
====================================================

Evaluates one or more motion clips by comparing them against a weighted
reference prior built from dataset motions that share the requested semantic
action category.

Usage
-----
python eval/evaluate_motion_quality.py \
    --motions "outputs/trial_00/*.npy" \
    --object-type Buffalo \
    --action-tags locomotion,attack
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import List, Optional

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_ANYTOP_DIR = _SCRIPT_DIR.parent
if str(_ANYTOP_DIR) not in sys.path:
    sys.path.insert(0, str(_ANYTOP_DIR))

from eval.motion_quality.scorer import DistributionEvalReport, DistributionMotionQualityScorer


def _expand_patterns(patterns: List[str]) -> List[str]:
    seen: set[str] = set()
    result: list[str] = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern, recursive=True)):
            absolute_path = os.path.abspath(path)
            if absolute_path not in seen:
                seen.add(absolute_path)
                result.append(absolute_path)
    return result


def _load_motions(paths: List[str], label: str) -> List[np.ndarray]:
    motions = []
    skipped = 0
    for path in paths:
        try:
            motion = np.load(path)
        except Exception as exc:
            print(f"[warn] {label}: failed to load {path}: {exc}", file=sys.stderr)
            skipped += 1
            continue
        if motion.ndim != 3 or motion.shape[-1] != 13:
            print(
                f"[warn] {label}: expected (T,J,13), got {motion.shape} - skipping {path}",
                file=sys.stderr,
            )
            skipped += 1
            continue
        motions.append(motion.astype(np.float32))
    if skipped:
        print(f"[info] {label}: skipped {skipped} invalid file(s)", file=sys.stderr)
    return motions


def _color(score: float, text: str, use_color: bool) -> str:
    if not use_color:
        return text
    if score < 0.4:
        code = "\033[91m"
    elif score < 0.7:
        code = "\033[93m"
    else:
        code = "\033[92m"
    return f"{code}{text}\033[0m"


def _bar(score: float, width: int = 20) -> str:
    filled = round(score * width)
    return "#" * filled + "." * (width - filled)


def _print_report(report: DistributionEvalReport, use_color: bool) -> None:
    clr = lambda value, text: _color(value, text, use_color)

    print()
    print(f"{'=' * 74}")
    print("  Low-Shot Weighted-Reference Motion Quality Report")
    print(f"{'=' * 74}")
    print(
        f"  Object : {report.object_type or 'unknown'}  |  "
        f"Action : {report.action_tags or 'unknown'}"
    )
    print(
        f"  Query  : {report.n_input} clip(s) / {report.input_total_frames} frames  |  "
        f"Reference : {report.n_reference} clip(s) / {report.reference_total_frames} frames"
    )
    print()

    rows = [
        ("Macro distribution fidelity", report.macro_fidelity_score, "w=0.60"),
        ("Local joint naturalness", report.local_naturalness_score, "w=0.40"),
    ]
    for label, score, note in rows:
        print(f"  {label:<32s} {clr(score, f'{score:.3f}')}  {_bar(score)}  {note}")

    print()
    print(f"  {'OVERALL SCORE':<32s} {clr(report.overall_score, f'{report.overall_score:.3f}')}  {_bar(report.overall_score)}")
    print()
    print("  Reference species:")
    for species in report.reference_species:
        print(
            "    "
            f"{species['object_type']:<18} weight={species['species_weight']:.4f} "
            f"distance={species['cosine_distance']:.4f} clips={species['clip_count']} frames={species['total_frames']}"
        )
    print()
    print(f"  Macro feature groups : {json.dumps(report.macro_feature_group_scores, sort_keys=True)}")
    print(f"  Macro joint groups   : {json.dumps(report.macro_joint_group_scores, sort_keys=True)}")
    local_component_scores = report.raw.get("local_component_scores")
    if local_component_scores:
        print(f"  Local metric scores  : {json.dumps(local_component_scores, sort_keys=True)}")
    local_joint_group_scores = report.raw.get("local_joint_group_scores")
    if local_joint_group_scores:
        print(f"  Local joint groups   : {json.dumps(local_joint_group_scores, sort_keys=True)}")
    print()


def _write_json(report: DistributionEvalReport, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report.as_dict(), handle, indent=2)
    print(f"[info] JSON report written -> {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate_motion_quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
            Low-Shot Weighted-Reference Motion Quality Evaluator
            ─────────────────────────────────────────────────────
            Compares one or more query motions against a weighted reference prior
            assembled from dataset motions with the same semantic action tags.

            Reference construction:
              • semantic Top-K species neighbors in cond.npy joint-name embedding space
              • dataset motions filtered by action_tags (supports comma/semicolon separation)
              • species weights distributed across reference clips by frame count

            Scores two dimensions:
              • Macro distribution fidelity  — robust deviation of kinematic clip features
              • Local joint naturalness      — robust deviation of spectral/smoothness summaries

            Score range: 0.0 (worst) → 1.0 (best match to the weighted reference prior).
            """
        ),
    )
    parser.add_argument(
        "--motions",
        "-m",
        nargs="+",
        required=True,
        metavar="PATTERN",
        help="Glob pattern(s) for query motion .npy files.",
    )
    parser.add_argument(
        "--object_type",
        required=True,
        metavar="TYPE",
        help="Object type key present in cond.npy, e.g. Buffalo or Horse.",
    )
    parser.add_argument(
        "--action_tags",
        required=True,
        metavar="TAGS",
        help="Semantic action tags (comma/semicolon-separated), e.g. 'locomotion' or 'attack,jump'.",
    )
    parser.add_argument(
        "--dataset_root",
        default=None,
        metavar="DIR",
        help="Dataset root containing cond.npy, motion_metadata.json, and motions/. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--top_k_species",
        type=int,
        default=5,
        metavar="N",
        help="Number of semantic neighbor species to use for the weighted reference prior (default: 5).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        metavar="N",
        help="Frame rate of the motion data (default: 30).",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        metavar="FILE",
        help="Write the report as JSON.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress the verbose report and print only the score line.",
    )
    parser.add_argument(
        "--no_color",
        action="store_true",
        help="Disable ANSI colour in terminal output.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    use_color = not args.no_color and sys.stdout.isatty()

    motion_paths = _expand_patterns(args.motions)
    if not motion_paths:
        print("[error] No query motion files found.", file=sys.stderr)
        return 1

    print(f"Loading {len(motion_paths)} query motion(s) ...")
    motions = _load_motions(motion_paths, "query")
    if not motions:
        print("[error] No valid query motions were loaded.", file=sys.stderr)
        return 1

    scorer = DistributionMotionQualityScorer(fps=args.fps, dataset_root=args.dataset_root)
    try:
        report = scorer.evaluate(
            motions=motions,
            object_type=args.object_type,
            action_tags=args.action_tags,
            top_k_species=args.top_k_species,
        )
    except (ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    if args.quiet:
        print(
            f"overall={_color(report.overall_score, f'{report.overall_score:.3f}', use_color)}  "
            f"macro={report.macro_fidelity_score:.3f}  "
            f"local={report.local_naturalness_score:.3f}"
        )
    else:
        _print_report(report, use_color)

    if args.output_json:
        _write_json(report, args.output_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())