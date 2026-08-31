#!/usr/bin/env python3
"""Report each species' rest bone lengths against the ones its clips actually use.

Read-only. The re-seat itself is a step of ``regenerate_dataset_artifacts``
(which ``preprocess_and_validate`` invokes for you), so a dataset build already
carries it and there is nothing to re-apply by hand -- see
``data_loaders/truebones/truebones_utils/rest_geometry.py``. This tool answers
"what did that step decide, and what did it leave alone", over the same
predicates.

Two numbers per species, both RMS over bones and frames:

  vs_rest   |clip length / rest length - 1|      -- rest disagrees with the clips
  self      |clip length / that clip's own mean length - 1|
                                                 -- the clip itself is non-rigid

They separate the two populations cleanly. A species with a large ``vs_rest`` and
a near-zero ``self`` has *rigid clips seated on the wrong rest* -- fixable rest
geometry, typically nub / locator / expression-control bones the animation
collapses onto their parent while the rest pose gives them an offset. A species
where both are large is simply animated non-rigidly; its rest is fine and nothing
here can help it.

Run it against a freshly regenerated cond and the re-seatable count should be 0:
the pipeline already moved everything it is willing to move, and what is left is
what the guards reject (an interior joint, a genuinely animated bone, a prop
socket). See docs/canonical_frame_and_label_transfer.md section 4.3.

Usage:
    python tools/audit_rest_vs_clip_geometry.py
    python tools/audit_rest_vs_clip_geometry.py --joints --threshold 5
    python tools/audit_rest_vs_clip_geometry.py --cond dataset/truebones/zoo/.../cond.npy
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ANYTOP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ANYTOP_DIR.parent))
sys.path.insert(0, str(ANYTOP_DIR))

from data_loaders.truebones.truebones_utils.cond_schema import load_cond  # noqa: E402
from data_loaders.truebones.truebones_utils.dataset_sources import (  # noqa: E402
    bare_species_name,
    load_datasets_manifest,
)
from data_loaders.truebones.truebones_utils.rest_geometry import (  # noqa: E402
    RESEAT_MISMATCH,
    RESEAT_SELF_RIGID,
    accumulate_rest_vs_clip,
    finalize_rest_vs_clip,
    reseat_candidates,
)


def _match_species(clip_name, species_by_name):
    """Longest-prefix match of ``{species}_...npy`` inside one source."""
    best_key, best_len = None, -1
    for species, key in species_by_name.items():
        if clip_name.startswith(species + "_") and len(species) > best_len:
            best_key, best_len = key, len(species)
    return best_key


def index_clips(cond, manifest_path):
    """``{species_key: [clip path, ...]}`` over the manifest's sources."""
    clips = defaultdict(list)
    for source in load_datasets_manifest(manifest_path):
        motion_dir = Path(source.motion_dir)
        if not motion_dir.is_dir():
            continue
        species_by_name = {
            bare_species_name(key): key for key in cond
            if cond[key].get("dataset_namespace") in (source.namespace, None)
        }
        for motion_path in sorted(motion_dir.glob("*.npy")):
            key = _match_species(motion_path.name, species_by_name)
            if key is not None:
                clips[key].append(motion_path)
    return clips


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cond", default="dataset/merged/cond.npy",
                        help="cond.npy to audit (default: the merged training cond).")
    parser.add_argument("--datasets", default="dataset/datasets.jsonl")
    parser.add_argument("--threshold", type=float, default=10.0,
                        help="Report species whose vs_rest RMS exceeds this %% (default 10).")
    parser.add_argument("--max-clips", type=int, default=0,
                        help="Clips per species (0 = all).")
    parser.add_argument("--joints", action="store_true",
                        help="List the re-seatable joints of every reported species.")
    args = parser.parse_args()

    cond = load_cond(args.cond)
    clips_by_species = index_clips(cond, args.datasets)
    max_clips = args.max_clips or None

    rows = []
    for key, entry in cond.items():
        paths = clips_by_species.get(key)
        if not paths:
            continue
        acc = None
        for motion_path in (paths if max_clips is None else paths[:max_clips]):
            acc = accumulate_rest_vs_clip(entry, np.load(motion_path), acc=acc)
        report = finalize_rest_vs_clip(entry, acc)
        if report is not None:
            rows.append((key, entry, report))

    flagged = [row for row in rows if row[2]["vs_rest_pct"] > args.threshold]
    flagged.sort(key=lambda row: -row[2]["vs_rest_pct"])

    print(f"cond: {args.cond}   species measured: {len(rows)}")
    print(f"re-seat guards: mismatch > {RESEAT_MISMATCH:.0%}, clip spread <= "
          f"{RESEAT_SELF_RIGID:.0%}, leaf joint, not a prop socket")
    print(f"over {args.threshold:.0f}% rest disagreement: {len(flagged)}\n")
    print(f"{'species':<44}{'clips':>6}{'vs_rest%':>10}{'self%':>9}   verdict")

    total_candidates = 0
    for key, entry, report in flagged:
        candidates = reseat_candidates(entry, report)
        total_candidates += len(candidates)
        verdict = ("clip is non-rigid (rest is fine)"
                   if report["self_pct"] > 0.5 * report["vs_rest_pct"]
                   else "rest disagrees with rigid clips")
        print(f"{key:<44}{report['n_clips']:>6}{report['vs_rest_pct']:>10.1f}"
              f"{report['self_pct']:>9.1f}   {verdict}; "
              f"{len(candidates)} re-seatable")
        if args.joints:
            for candidate in sorted(
                candidates, key=lambda c: abs(np.log(max(c["ratio"], 1e-9))), reverse=True
            )[:12]:
                print(f"      {candidate['name']:<38}rest {candidate['rest_len']:.4f} "
                      f"-> clip {candidate['clip_len']:.4f}  (x{candidate['ratio']:.2f})")

    print(f"\nre-seatable joints across flagged species: {total_candidates}")
    if total_candidates:
        print("This tool is read-only. Regenerate the dataset artifacts to apply them --\n"
              "the re-seat is a step there, so preprocessing already does it:\n"
              "  python tools/regenerate_dataset_artifacts.py --dataset-dir <dataset>\n"
              "  python tools/merge_dataset_cond.py --datasets dataset/datasets.jsonl "
              "--out dataset/merged/cond.npy")
    else:
        print("Nothing left to re-seat: every remaining disagreement is one the guards\n"
              "reject (an interior joint, a genuinely animated bone, or a prop socket).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
