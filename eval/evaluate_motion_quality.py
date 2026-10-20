#!/usr/bin/env python
"""
Motion Quality Evaluator
========================

Scores motion clips on two dimensions (see ``eval/motion_quality/quality.py``):
physical plausibility -- rigid bones, smooth parts, no frame jumps -- and
data-prior conformity -- limb symmetry, coordination and tempo against real
clips whose action_label shares a head word with the label the clips were
generated with. A clip's quality is the geometric mean of the two; the clips of
one species are reported together (median over clips). The reference is pooled
over every dataset in ``dataset/datasets_all.jsonl`` unless ``--dataset_root`` says
otherwise.

Usage
-----
python eval/evaluate_motion_quality.py \
    --motions "outputs/trial_00/*.npy" \
    --object-type Buffalo \
    --action-label run

python eval/evaluate_motion_quality.py \
    --motions "outputs/new_skeleton/*.npy" \
    --object-type dragon \
    --action-label "fly, forward" \
    --cond-path outputs/new_skeleton/cond.npy
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_ANYTOP_DIR = _SCRIPT_DIR.parent
if str(_ANYTOP_DIR) not in sys.path:
    sys.path.insert(0, str(_ANYTOP_DIR))

from data_loaders.truebones.truebones_utils.param_utils import FEATS_LEN
from eval.motion_quality.quality import DIMENSIONS, MotionQualityReport, MotionQualityScorer
from utils.misc import infer_object_type_from_filename

# The datasets the reference is pooled over by default: every processed
# dataset the training cond was merged from.
_DEFAULT_DATASET_ROOT = _ANYTOP_DIR / "dataset" / "datasets_all.jsonl"


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
    if motion.ndim != 3 or motion.shape[-1] != FEATS_LEN:
        print(
            f"[warn] expected (T,J,{FEATS_LEN}), got {motion.shape} - skipping {path}",
            file=sys.stderr,
        )
        return None
    return motion.astype(np.float32)


def _register_cond_path(scorer: MotionQualityScorer, cond_path: str) -> Path:
    """Register an optional custom cond.npy for novel query species."""
    resolved = Path(cond_path).expanduser()
    if not resolved.is_absolute():
        resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"cond.npy not found: {resolved}")

    cond_dict = np.load(resolved, allow_pickle=True).item()
    if not isinstance(cond_dict, dict):
        raise ValueError(f"Expected cond.npy to contain a dict, got {type(cond_dict).__name__}")

    scorer.register_cond(cond_dict)
    return resolved


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


def _print_report(report: MotionQualityReport, use_color: bool, quiet: bool) -> None:
    score_text = _color(report.score, f"{report.score:.4f}", use_color)
    if quiet:
        dims = "  ".join(f"{dim}={report.dimensions[dim]:.4f}" for dim in DIMENSIONS)
        print(f"{report.object_type:<40s} clips={report.n_clips:<3d} quality={score_text}  {dims}")
        return
    print()
    print(f"{'=' * 74}")
    print(f"  {report.object_type} ({report.subset})  |  {report.action_label}  |  {report.n_clips} clip(s)")
    print(f"{'=' * 74}")
    print(f"    {'Quality':<20s} {score_text}  {_bar(report.score)}")
    for dim in DIMENSIONS:
        value = report.dimensions[dim]
        terms = "  ".join(f"{k}={v:.3f}" for k, v in report.terms[dim].items())
        print(f"    {dim.capitalize():<20s} {_color(value, f'{value:.4f}', use_color)}  {terms}")
    if report.worst:
        print(f"    {'Weakest':<20s} " + "  ".join(f"{k}={v:.3f}" for k, v in report.worst))
    print()
    for clip in report.clips:
        print(f"    {clip['file']:<40s} quality={clip['score']:.4f}  physics={clip['physics']:.4f}  prior={clip['prior']:.4f}")
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate_motion_quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
            Motion Quality Evaluator
            ────────────────────────
            Scores clips on two dimensions, each in [0, 1]:

              • physics: bone rigidity, per-part smoothness and frame continuity,
                against real clips of the query's body plan (label-free)
              • prior: left/right symmetry, limb coordination and part tempo,
                against real clips sharing a head word with --action_label (never
                the query's own species), calibrated so a typical real clip scores 1

            A clip's quality is the geometric mean of the two.
            """
        ),
    )
    parser.add_argument(
        "--motions",
        "-m",
        nargs="+",
        required=True,
        metavar="PATTERN",
        help="Glob pattern(s) for query motion .npy files; the files of one species are reported together.",
    )
    parser.add_argument(
        "--object_type", "--object-type",
        default=None,
        metavar="TYPE",
        help="Object type key present in the default cond.npy or in --cond-path, "
             "e.g. Buffalo or dragon. Auto-inferred from motion filenames when omitted.",
    )
    parser.add_argument(
        "--cond_path", "--cond-path",
        default=None,
        metavar="FILE",
        help="Optional custom cond.npy for novel query species. This is used only for query-side skeleton metadata; "
             "the reference distribution still comes from the dataset corpus.",
    )
    parser.add_argument(
        "--action_label", "--action-label",
        required=True,
        metavar="LABEL",
        help="Required. The action_label the query motions were generated with, spelled like "
             "generate.py's --action_label (e.g. 'run' or 'fly, forward'). There is no default: "
             "for clips generated without a label, name the action of their reference clip.",
    )
    parser.add_argument(
        "--dataset_root", "--dataset-root",
        default=str(_DEFAULT_DATASET_ROOT),
        metavar="DIR_OR_MANIFEST",
        help="Dataset root containing cond.npy, motion_metadata.json, and motions/. "
             "A path ending in .jsonl is read as a dataset manifest instead, pooling the "
             "reference distribution over every dataset it lists. Default: dataset/datasets_all.jsonl.",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        metavar="FILE",
        help="Write the reports as JSON.",
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

    scorer = MotionQualityScorer(dataset_root=args.dataset_root)
    if args.cond_path:
        try:
            _register_cond_path(scorer, args.cond_path)
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"[error] failed to load --cond-path: {exc}", file=sys.stderr)
            return 1

    # An explicit --object_type applies to every file. Otherwise the type is
    # inferred per file from its name (validated against the scorer's species, so
    # a multi-token name is not cut at its first underscore) and each species'
    # files are scored as their own set.
    species_lookup = scorer.species_lookup() if args.object_type is None else None
    groups: Dict[str, List[tuple[str, np.ndarray]]] = {}
    skipped = 0
    for path in motion_paths:
        object_type = args.object_type or infer_object_type_from_filename(path, valid_types=species_lookup)
        if object_type is None:
            print(
                f"[warn] cannot auto-detect object_type from "
                f"'{os.path.basename(path)}' - skipping (pass --object_type to override)",
                file=sys.stderr,
            )
            skipped += 1
            continue
        motion = _validate_motion(path)
        if motion is None:
            skipped += 1
            continue
        groups.setdefault(object_type, []).append((path, motion))

    print(f"Evaluating {sum(len(g) for g in groups.values())} motion(s) in {len(groups)} set(s) ...")
    results: list[tuple[List[str], MotionQualityReport]] = []
    for object_type, members in groups.items():
        try:
            report = scorer.evaluate(
                [motion for _, motion in members], object_type, args.action_label,
                names=[os.path.basename(path) for path, _ in members],
            )
        except (ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
            print(f"[warn] evaluation failed for {object_type}: {exc}", file=sys.stderr)
            skipped += len(members)
            continue
        paths = [path for path, _ in members]
        results.append((paths, report))
        _print_report(report, use_color, args.quiet)

    if skipped:
        print(f"[info] skipped {skipped} file(s)", file=sys.stderr)
    if not results:
        print("[error] No valid motion evaluations were produced.", file=sys.stderr)
        return 1

    if args.output_json:
        combined = {
            "sets": [
                {"files": [os.path.basename(path) for path in paths], "score": report.as_dict()}
                for paths, report in results
            ],
        }
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(combined, handle, indent=2)
        print(f"[info] JSON report written -> {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
