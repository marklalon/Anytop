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
from utils.misc import infer_object_type_from_filename


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


def _validate_motion(path: str) -> Optional[np.ndarray]:
    """Load and validate a single motion file. Returns None if invalid."""
    try:
        motion = np.load(path)
    except Exception as exc:
        print(f"[warn] failed to load {path}: {exc}", file=sys.stderr)
        return None
    if motion.ndim != 3 or motion.shape[-1] != 13:
        print(
            f"[warn] expected (T,J,13), got {motion.shape} - skipping {path}",
            file=sys.stderr,
        )
        return None
    return motion.astype(np.float32)


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

    def _round4_dict(values: dict) -> dict:
        return {
            key: (round(float(value), 4) if isinstance(value, (int, float, np.floating)) else value)
            for key, value in values.items()
        }

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
        ("Joint naturalness", report.overall_score, ""),
    ]
    for label, score, note in rows:
        print(f"  {label:<32s} {clr(score, f'{score:.4f}')}  {_bar(score)}  {note}")

    print()
    print("  Reference species:")
    for species in report.reference_species:
        print(
            "    "
            f"{species['object_type']:<18} weight={species['species_weight']:.4f} "
            f"distance={species['cosine_distance']:.4f} clips={species['clip_count']} frames={species['total_frames']}"
        )
    print()
    component_scores = report.raw.get("component_scores")
    if component_scores:
        print(f"  Metric scores  : {json.dumps(_round4_dict(component_scores))}")
    joint_group_scores = report.raw.get("joint_group_scores")
    if joint_group_scores:
        print(f"  Joint groups   : {json.dumps(_round4_dict(joint_group_scores), sort_keys=True)}")
    print()


def _write_json(report: DistributionEvalReport, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report.as_dict(), handle, indent=2)
    print(f"[info] JSON report written -> {path}")


def _print_per_file_summary(
    results: list[tuple[str, DistributionEvalReport]],
    use_color: bool,
    quiet: bool,
) -> None:
    """Print per-file scores and average."""
    if quiet:
        for path, report in results:
            name = os.path.basename(path)
            print(
                f"{name:<50s} "
                f"score={_color(report.overall_score, f'{report.overall_score:.4f}', use_color)}"
            )
        if len(results) > 1:
            avg_score = np.mean([r.overall_score for _, r in results])
            print(
                f"{'AVERAGE':<50s} "
                f"score={_color(avg_score, f'{avg_score:.4f}', use_color)}"
            )
        return

    # Verbose mode
    clr = lambda value, text: _color(value, text, use_color)

    def _round4_dict(values: dict) -> dict:
        return {
            key: (round(float(value), 4) if isinstance(value, (int, float, np.floating)) else value)
            for key, value in values.items()
        }

    def _avg_dict(dicts: list[dict]) -> dict:
        """Average a list of dicts with the same keys."""
        if not dicts:
            return {}
        keys = list(dicts[0].keys())
        return {k: round(float(np.mean([d[k] for d in dicts])), 4) for k in keys}

    is_multi = len(results) > 1

    print()
    print(f"{'=' * 74}")
    if is_multi:
        print(f"  Evaluating {len(results)} motion(s) — Per-File Scores")
    else:
        print("  Motion Quality Report")
    print(f"{'=' * 74}")
    print()

    for path, report in results:
        name = os.path.basename(path)
        if is_multi:
            # Multi-file: only score line per file
            print(
                f"  {name:<50s} "
                f"score={clr(report.overall_score, f'{report.overall_score:.4f}')}",
            )
        else:
            # Single file: full detailed report
            print(f"  {name}")
            print(
                f"    {'Overall Score':<32s} "
                f"{clr(report.overall_score, f'{report.overall_score:.4f}')}  "
                f"{_bar(report.overall_score)}"
            )
            component_scores = report.raw.get("component_scores")
            if component_scores:
                print(f"    Metric scores  : {json.dumps(_round4_dict(component_scores))}")
            joint_group_scores = report.raw.get("joint_group_scores")
            if joint_group_scores:
                print(f"    Joint groups   : {json.dumps(_round4_dict(joint_group_scores), sort_keys=True)}")
            bone_length_drift_stats = report.raw.get("bone_length_drift_stats")
            if bone_length_drift_stats:
                print(f"    Bone length drift    : max_abs={bone_length_drift_stats['max_abs_drift_pct']:.2f}%  mean_abs={bone_length_drift_stats['mean_abs_drift_pct']:.2f}%  median_abs={bone_length_drift_stats['median_abs_drift_pct']:.2f}%")
            print()

    if is_multi:
        # ── Average summary with full detail ──────────────────────────────
        avg_score = np.mean([r.overall_score for _, r in results])

        raw_list = [r.raw for _, r in results]
        comp = raw_list[0].get("component_scores") if raw_list else None
        jg = raw_list[0].get("joint_group_scores") if raw_list else None
        avg_comp = _avg_dict([r.raw.get("component_scores", {}) for _, r in results]) if comp else None
        avg_jg = _avg_dict([r.raw.get("joint_group_scores", {}) for _, r in results]) if jg else None

        print()
        print(f"{'─' * 74}")
        print(f"  AVERAGE ({len(results)} files)")
        print()
        print(
            f"    {'Overall Score':<32s} "
            f"{clr(avg_score, f'{avg_score:.4f}')}  "
            f"{_bar(avg_score)}"
        )
        if avg_comp:
            print(f"    Metric scores  : {json.dumps(avg_comp)}")
        if avg_jg:
            print(f"    Joint groups   : {json.dumps(avg_jg, sort_keys=True)}")

        # Average bone length drift across files
        drift_stats_list = [r.raw.get("bone_length_drift_stats") for _, r in results
                            if r.raw.get("bone_length_drift_stats")]
        if drift_stats_list:
            avg_max = np.mean([s["max_abs_drift_pct"] for s in drift_stats_list])
            avg_mean = np.mean([s["mean_abs_drift_pct"] for s in drift_stats_list])
            avg_median = np.mean([s["median_abs_drift_pct"] for s in drift_stats_list])
            print(
                f"    Bone length drift    : max_abs={avg_max:.2f}%  "
                f"mean_abs={avg_mean:.2f}%  "
                f"median_abs={avg_median:.2f}%"
            )
        print()

    # Print reference species from the first report
    if results:
        report = results[0][1]
        print("  Reference species:")
        for species in report.reference_species:
            print(
                "    "
                f"{species['object_type']:<18} weight={species['species_weight']:.4f} "
                f"distance={species['cosine_distance']:.4f} clips={species['clip_count']} frames={species['total_frames']}"
            )
        print()


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

            Scores:
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
        "--object_type", "--object-type",
        default=None,
        metavar="TYPE",
        help="Object type key present in cond.npy, e.g. Buffalo or Horse. "
             "Auto-inferred from motion filenames when omitted.",
    )
    parser.add_argument(
        "--action_tags", "--action-tags",
        required=True,
        metavar="TAGS",
        help="Semantic action tags (comma/semicolon-separated), e.g. 'locomotion' or 'attack,jump'.",
    )
    parser.add_argument(
        "--dataset_root", "--dataset-root",
        default=None,
        metavar="DIR",
        help="Dataset root containing cond.npy, motion_metadata.json, and motions/. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--top_k_species", "--top-k-species",
        type=int,
        default=3,
        metavar="N",
        help="Number of semantic neighbor species to use for the weighted reference prior (default: 3).",
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

    # ── Auto-infer object_type from filenames ──────────────────────────────
    if args.object_type is None:
        inferred = infer_object_type_from_filename(motion_paths[0])
        if inferred is not None:
            args.object_type = inferred
            print(f"Auto-detected object_type: {inferred}")
        else:
            print(
                f"[error] Cannot auto-detect object_type from '{os.path.basename(motion_paths[0])}'.\n"
                f"  Pass --object_type explicitly.",
                file=sys.stderr,
            )
            return 1

    # ── Evaluate each motion file independently ────────────────────────────
    scorer = DistributionMotionQualityScorer(fps=args.fps, dataset_root=args.dataset_root)
    results: list[tuple[str, DistributionEvalReport]] = []
    skipped = 0

    print(f"Evaluating {len(motion_paths)} motion(s) ...")
    for path in motion_paths:
        motion = _validate_motion(path)
        if motion is None:
            skipped += 1
            continue
        try:
            report = scorer.evaluate(
                motions=[motion],
                object_type=args.object_type,
                action_tags=args.action_tags,
                top_k_species=args.top_k_species,
            )
        except (ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
            print(f"[warn] evaluation failed for {path}: {exc}", file=sys.stderr)
            skipped += 1
            continue
        results.append((path, report))

    if skipped:
        print(f"[info] skipped {skipped} invalid file(s)", file=sys.stderr)
    if not results:
        print("[error] No valid motion evaluations were produced.", file=sys.stderr)
        return 1

    # ── Print results ──────────────────────────────────────────────────────
    _print_per_file_summary(results, use_color, args.quiet)

    if args.output_json:
        # Write combined JSON with per-file scores
        combined = {
            "per_file": [
                {"file": os.path.basename(path), "score": report.as_dict()}
                for path, report in results
            ],
            "average": {
                "overall_score": round(float(np.mean([r.overall_score for _, r in results])), 4),
            },
        }
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(combined, handle, indent=2)
        print(f"[info] JSON report written -> {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())