#!/usr/bin/env python
"""
Distribution-Based Motion Quality Evaluator
============================================

Evaluates motion quality by comparing the statistical distributions of
generated vs clean (GT) motion samples.  Requires ≥32 clips in each set.

Usage
-----
# Basic comparison
python eval/evaluate_motion_quality.py \\
    --generated "outputs/trial_00/*/generated_prediction.npy" \\
    --clean     "outputs/trial_00/*/clean_target.npy"

# With JSON output (object-type is inferred automatically from clean filenames)
python eval/evaluate_motion_quality.py \\
    --generated "outputs/trial_00/*/generated_prediction.npy" \\
    --clean     "dataset/.../motions/Human_*.npy" \\
    --output-json report.json

# Sanity check: clean vs clean should score ~1.0
python eval/evaluate_motion_quality.py \\
    --generated "dataset/truebones/zoo/truebones_processed/motions/Human_*.npy" \\
    --clean     "dataset/truebones/zoo/truebones_processed/motions/Human_*.npy"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent      # eval/
_ANYTOP_DIR = _SCRIPT_DIR.parent                   # Anytop/
if str(_ANYTOP_DIR) not in sys.path:
    sys.path.insert(0, str(_ANYTOP_DIR))

from eval.motion_quality.scorer import DistributionMotionQualityScorer, DistributionEvalReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expand_patterns(patterns: List[str]) -> List[str]:
    """Expand glob patterns to a sorted, deduplicated list of absolute paths."""
    seen: set[str] = set()
    result: list[str] = []
    for pat in patterns:
        for p in sorted(glob.glob(pat, recursive=True)):
            ap = os.path.abspath(p)
            if ap not in seen:
                seen.add(ap)
                result.append(ap)
    return result


def _load_motions(paths: List[str], label: str) -> List[np.ndarray]:
    """Load .npy motion files; skip and warn on shape mismatches."""
    motions = []
    skipped = 0
    for path in paths:
        try:
            m = np.load(path)
        except Exception as e:
            print(f"[warn] {label}: failed to load {path}: {e}", file=sys.stderr)
            skipped += 1
            continue
        if m.ndim != 3 or m.shape[-1] != 13:
            print(
                f"[warn] {label}: expected (T,J,13), got {m.shape} — skipping {path}",
                file=sys.stderr,
            )
            skipped += 1
            continue
        motions.append(m.astype(np.float32))
    if skipped:
        print(f"[info] {label}: skipped {skipped} file(s) with invalid format", file=sys.stderr)
    return motions


def _infer_object_type(paths: List[str]) -> Optional[str]:
    """Infer object type from clean file paths.

    Checks (in order):
    1. Filename stem: matches 'Horse_001.npy' -> 'Horse'
    2. Parent directory name: matches 'sample_000_Horse' -> 'Horse'

    Only accepts uppercase-initial prefixes to avoid matching generic names
    like 'clean_target' or 'generated_prediction'.
    """
    for path in paths:
        p = Path(path)
        # Try filename stem first
        m = re.match(r'^([A-Z][A-Za-z0-9]*)_', p.stem)
        if m:
            return m.group(1)
        # Try parent directory name (e.g. sample_000_Horse -> Horse)
        for part in reversed(p.parent.parts):
            m2 = re.search(r'_([A-Z][A-Za-z0-9]+)$', part)
            if m2:
                return m2.group(1)
    return None


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
    clr = lambda v, t: _color(v, t, use_color)

    print()
    print(f"{'=' * 70}")
    print(f"  Distribution Motion Quality Report")
    print(f"{'=' * 70}")
    print(
        f"  Object : {report.object_type or 'unknown'}  |  "
        f"{report.n_generated} generated  /  {report.n_clean} clean"
    )
    print()

    rows = [
        ("Macro distribution fidelity", report.macro_fidelity_score, "w=0.60"),
        ("Local joint naturalness",     report.local_naturalness_score, "w=0.40"),
    ]
    for label, score, note in rows:
        bar       = _bar(score)
        score_str = clr(score, f"{score:.3f}")
        print(f"  {label:<32s} {score_str}  {bar}  {note}")

    print()
    total_str = clr(report.overall_score, f"{report.overall_score:.3f}")
    print(f"  {'OVERALL SCORE':<32s} {total_str}  {_bar(report.overall_score)}")

    print()
    print("  Macro detail:")
    print(f"    Feature groups            {json.dumps(report.macro_feature_group_scores, sort_keys=True)}")
    print(f"    Joint groups              {json.dumps(report.macro_joint_group_scores, sort_keys=True)}")
    print(f"    Joint group sizes         {json.dumps(report.macro_joint_group_sizes, sort_keys=True)}")
    print()

    print("  Local detail:")
    local_component_scores = report.raw.get("local_component_scores")
    if local_component_scores:
        print(f"    Metric scores            {json.dumps(local_component_scores, sort_keys=True)}")
    local_joint_group_scores = report.raw.get("local_joint_group_scores")
    if local_joint_group_scores:
        print(f"    Joint groups             {json.dumps(local_joint_group_scores, sort_keys=True)}")
    local_psd_jsd_by_group = report.raw.get("local_psd_jsd_by_group")
    if local_psd_jsd_by_group:
        print(f"    PSD JSD groups           {json.dumps(local_psd_jsd_by_group, sort_keys=True)}")
    print()


def _write_json(report: DistributionEvalReport, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report.as_dict(), fh, indent=2)
    print(f"[info] JSON report written -> {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evaluate_motion_quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Distribution-Based Motion Quality Evaluator
            ─────────────────────────────────────────────
            Compares the statistical distributions of generated vs clean (GT)
            motion clips.  Requires ≥32 clips in each set.

            Scores two dimensions (50/50 weight each):
                            • Macro distribution fidelity  — robust per-feature Wasserstein
                                aggregation on kinematic feature vectors with root/axial/limb
                                joint grouping (temporal statistics + frequency content)
              • Local joint naturalness      — spectral flatness, PSD shape,
                autocorrelation, jerk and snap distributions

            Score range: 0.0 (worst) → 1.0 (perfect match with clean).
            A clean-vs-clean split should always score ≈ 1.0 (sanity check).
        """),
        epilog=textwrap.dedent("""\
            Examples:
              python eval/evaluate_motion_quality.py \\
                  --generated "outputs/trial_00/*/generated_prediction.npy" \\
                  --clean     "outputs/trial_00/*/clean_target.npy"

              python eval/evaluate_motion_quality.py \\
                  --generated "outputs/trial_00/*/generated.npy" \\
                  --clean     "dataset/.../motions/Human_*.npy" \\
                  --output-json report.json
        """),
    )
    p.add_argument(
        "--generated", "-g",
        nargs="+",
        required=True,
        metavar="PATTERN",
        help="Glob pattern(s) for generated motion .npy files.",
    )
    p.add_argument(
        "--clean", "-c",
        nargs="+",
        required=True,
        metavar="PATTERN",
        help="Glob pattern(s) for clean (GT) motion .npy files.",
    )
    p.add_argument(
        "--object-type",
        default=None,
        metavar="TYPE",
        help="Skeleton type name stored as metadata (e.g. Human, Cat). "
             "Inferred automatically from clean filenames when not specified.",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=30,
        metavar="N",
        help="Frame rate of the motion data (default: 30).",
    )
    p.add_argument(
        "--min-samples",
        type=int,
        default=32,
        metavar="N",
        help="Minimum number of clips required in each set (default: 32).",
    )
    p.add_argument(
        "--output-json",
        default=None,
        metavar="FILE",
        help="Write JSON report to this file.",
    )
    p.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose report; print only the score line.",
    )
    p.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colour in terminal output.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    use_color = not args.no_color and sys.stdout.isatty()

    gen_paths = _expand_patterns(args.generated)
    cln_paths = _expand_patterns(args.clean)

    if args.object_type is None:
        args.object_type = _infer_object_type(cln_paths)

    if not gen_paths:
        print("[error] No generated files found.", file=sys.stderr)
        return 1
    if not cln_paths:
        print("[error] No clean files found.", file=sys.stderr)
        return 1

    print(f"Loading {len(gen_paths)} generated and {len(cln_paths)} clean motions ...")
    gen_motions = _load_motions(gen_paths, "generated")
    cln_motions = _load_motions(cln_paths, "clean")

    if len(gen_motions) < args.min_samples:
        print(
            f"[error] Only {len(gen_motions)} valid generated clips; "
            f"need ≥{args.min_samples}.",
            file=sys.stderr,
        )
        return 1
    if len(cln_motions) < args.min_samples:
        print(
            f"[error] Only {len(cln_motions)} valid clean clips; "
            f"need ≥{args.min_samples}.",
            file=sys.stderr,
        )
        return 1

    print(f"Evaluating {len(gen_motions)} generated vs {len(cln_motions)} clean clips ...\n")

    scorer = DistributionMotionQualityScorer(fps=args.fps, min_batch_size=args.min_samples)
    try:
        report = scorer.evaluate(gen_motions, cln_motions, object_type=args.object_type)
    except ValueError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1

    if args.quiet:
        clr = lambda v, t: _color(v, t, use_color)
        print(
            f"overall={clr(report.overall_score, f'{report.overall_score:.3f}')}  "
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
