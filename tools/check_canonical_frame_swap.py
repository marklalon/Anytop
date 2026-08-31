#!/usr/bin/env python3
"""Frame-swap self-check for the canonical standardization table.

Decode a clip through the WRONG object_subset's canonical statistics and measure
what it does to the skeleton's bone lengths. This is the hard, model-free
criterion for the shared-position-gain change (docs/canonical_frame_and_label_transfer.md
step 2): the off-diagonal of the matrix must collapse onto the diagonal.

Why it is the right criterion. The decode is ``x * std + mean`` -> ``* L`` ->
``+ rest``. ``mean`` is per-channel and identical at every joint and frame, so a
mean mismatch translates the whole skeleton rigidly and every bone length is
exact. Only ``std`` -- the gain -- can deform, and bone lengths are decided by the
position channel alone. So once ``collapse_stat_blocks`` makes the position gain
one globally shared constant, a subset mismatch is structurally incapable of
deforming anything, and every row of the matrix flattens to its own diagonal
(the clip's intrinsic non-rigidity against its own rest pose, which this change
does not touch -- that is section 4.3's separate item).

Second table: the amplitude each subset's real clips actually land at in
canonical space. Sharing the position gain trades an exactly-unit-variance
position channel for a spread across subsets; this is where that cost is read
off (rot / vel must be unchanged, and every mean must stay ~0).

Usage:
    python tools/check_canonical_frame_swap.py --cond dataset/merged/cond.npy
    python tools/check_canonical_frame_swap.py --cond save/run/cond.npy --clips 12
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

from data_loaders.truebones.truebones_utils.canonical_features import (  # noqa: E402
    canonical_to_physical_hml,
    get_canonical_global_stats,
    physical_hml_to_canonical,
)
from data_loaders.truebones.truebones_utils.cond_schema import load_cond  # noqa: E402
from data_loaders.truebones.truebones_utils.dataset_sources import (  # noqa: E402
    bare_species_name,
    load_datasets_manifest,
)

_POS = slice(0, 3)
_ROT = slice(3, 9)
_VEL = slice(9, 12)


def _match_species(clip_name, species_by_name):
    """Longest-prefix match of ``{species}_...npy`` inside one source."""
    best_key, best_len = None, -1
    for species, key in species_by_name.items():
        if clip_name.startswith(species + "_") and len(species) > best_len:
            best_key, best_len = key, len(species)
    return best_key


def index_clips(cond, manifest_path):
    """``{species_key: [clip path, ...]}`` over every source in the manifest."""
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


def subset_of_entry(entry):
    tags = entry.get("species_tags") or ()
    return str(tags[0]).strip().lower() if len(tags) else None


def sample_clips(cond, clips_by_species, per_subset):
    """Pick up to ``per_subset`` clips per object_subset, spread over its species.

    Round-robin over the subset's species (sorted, one clip each per pass) so a
    single clip-rich species cannot supply the whole row.
    """
    by_subset = defaultdict(list)
    for key, entry in cond.items():
        subset = subset_of_entry(entry)
        if subset and clips_by_species.get(key):
            by_subset[subset].append(key)

    sampled = {}
    for subset, species in sorted(by_subset.items()):
        species = sorted(species)
        picked, depth = [], 0
        while len(picked) < per_subset:
            added = False
            for key in species:
                paths = clips_by_species[key]
                if depth < len(paths):
                    picked.append((key, paths[depth]))
                    added = True
                    if len(picked) >= per_subset:
                        break
            if not added:
                break
            depth += 1
        if picked:
            sampled[subset] = picked
    return sampled


def bone_length_error_pct(positions, rest_pos, parents):
    """RMS deviation (%) of every bone's length from its own rest length.

    ``positions``: ``[T, J, 3]`` decoded RIC positions. Zero-length rest bones
    carry no ratio and are skipped.
    """
    parents = np.asarray(parents, dtype=np.int64)
    children = np.flatnonzero(parents >= 0)
    if children.size == 0:
        return float("nan")
    rest_len = np.linalg.norm(rest_pos[children] - rest_pos[parents[children]], axis=-1)
    keep = rest_len > 1e-6
    if not keep.any():
        return float("nan")
    children, rest_len = children[keep], rest_len[keep]
    lengths = np.linalg.norm(
        positions[:, children] - positions[:, parents[children]], axis=-1
    )
    relative = lengths / rest_len[None, :] - 1.0
    relative = relative[np.isfinite(relative)]
    return float(100.0 * np.sqrt(np.mean(relative ** 2)))


def swap_matrix(cond, sampled, subsets, stats_by_subset):
    """``{clip subset: {decode subset: median bone-length error %}}``.

    The clip is encoded once with its OWN statistics, then the canonical tensor
    is held fixed and decoded through each subset's statistics in turn -- exactly
    the mismatch a species whose subset the model misreads would suffer.
    """
    matrix = {}
    for clip_subset in subsets:
        row = {}
        per_decode = defaultdict(list)
        for species_key, motion_path in sampled.get(clip_subset, []):
            entry = cond[species_key]
            motion = np.load(motion_path).astype(np.float32, copy=False)
            if motion.ndim != 3 or motion.shape[-1] < 13:
                continue
            canonical = physical_hml_to_canonical(motion, entry)
            rest_pos = np.asarray(entry["rest_pos_ric_hml"], dtype=np.float32)
            parents = np.asarray(entry["parents"], dtype=np.int64)
            for decode_subset in subsets:
                mean, std = stats_by_subset[decode_subset]
                swapped = dict(entry)
                swapped["canonical_feature_mean"] = mean
                swapped["canonical_feature_std"] = std
                physical = canonical_to_physical_hml(canonical, swapped)
                per_decode[decode_subset].append(
                    bone_length_error_pct(physical[..., 0:3], rest_pos, parents)
                )
        for decode_subset in subsets:
            values = [v for v in per_decode[decode_subset] if np.isfinite(v)]
            row[decode_subset] = float(np.median(values)) if values else float("nan")
        matrix[clip_subset] = row
    return matrix


def amplitude_table(cond, sampled, subsets):
    """Per-subset achieved amplitude of real clips in canonical space.

    Reports the mean per-channel std inside each block (1.0 == the calibration
    the standardization aims at) and the largest per-channel |mean|.
    """
    table = {}
    for subset in subsets:
        acc = []
        for species_key, motion_path in sampled.get(subset, []):
            motion = np.load(motion_path).astype(np.float32, copy=False)
            if motion.ndim != 3 or motion.shape[-1] < 13:
                continue
            canonical = np.asarray(
                physical_hml_to_canonical(motion, cond[species_key]), dtype=np.float64
            )
            flat = canonical.reshape(-1, canonical.shape[-1])
            acc.append(flat[np.isfinite(flat).all(axis=1)])
        if not acc:
            continue
        pooled = np.concatenate(acc, axis=0)
        std = pooled.std(axis=0)
        mean = pooled.mean(axis=0)
        table[subset] = {
            "pos": float(std[_POS].mean()),
            "rot": float(std[_ROT].mean()),
            "vel": float(std[_VEL].mean()),
            "max_abs_mean": float(np.abs(mean[:12]).max()),
            "frames_joints": int(pooled.shape[0]),
        }
    return table


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cond", default="dataset/merged/cond.npy",
                        help="cond.npy holding the canonical statistics to check.")
    parser.add_argument("--datasets", default="dataset/datasets.jsonl",
                        help="Manifest naming where the clips live.")
    parser.add_argument("--clips", type=int, default=12,
                        help="Clips sampled per object_subset (default 12).")
    args = parser.parse_args()

    cond = load_cond(args.cond)
    stats_by_subset = {}
    for entry in cond.values():
        subset = subset_of_entry(entry)
        stats = get_canonical_global_stats(entry)
        if subset and stats is not None and subset not in stats_by_subset:
            stats_by_subset[subset] = (
                np.asarray(stats[0], dtype=np.float32),
                np.asarray(stats[1], dtype=np.float32),
            )
    if not stats_by_subset:
        raise SystemExit(f"{args.cond} carries no canonical statistics.")

    clips_by_species = index_clips(cond, args.datasets)
    if not clips_by_species:
        raise SystemExit("No clips found for any species in the cond; check --datasets.")
    sampled = sample_clips(cond, clips_by_species, args.clips)
    subsets = [s for s in sorted(stats_by_subset) if s in sampled]

    print(f"cond: {args.cond}")
    print("position gain per subset (std[0:3]):")
    for subset in sorted(stats_by_subset):
        std = stats_by_subset[subset][1]
        print(f"  {subset:<12} {std[0]:.4f} {std[1]:.4f} {std[2]:.4f}")
    pos_gains = {tuple(np.round(stats[1][0:3], 6)) for stats in stats_by_subset.values()}
    isotropic = len(pos_gains) == 1 and len(set(next(iter(pos_gains)))) == 1
    print("  -> position gain is "
          + ("SHARED and isotropic" if isotropic else "per-subset / anisotropic"))

    matrix = swap_matrix(cond, sampled, subsets, stats_by_subset)
    width = max(len(s) for s in subsets) + 2
    print(f"\nbone-length RMS error (%) -- rows: clips of, cols: decoded as "
          f"({args.clips} clips/cell, median)")
    print(" " * 14 + "".join(f"{s[:width - 1]:>{width}}" for s in subsets))
    off_diagonal_excess = []
    for clip_subset in subsets:
        cells = []
        for decode_subset in subsets:
            value = matrix[clip_subset][decode_subset]
            marker = "*" if decode_subset == clip_subset else " "
            cells.append(f"{value:>{width - 1}.1f}{marker}")
            if decode_subset != clip_subset and np.isfinite(value):
                off_diagonal_excess.append(value - matrix[clip_subset][clip_subset])
        print(f"{clip_subset:<14}" + "".join(cells))
    print("  * = decoded with its own statistics (the clip's intrinsic "
          "non-rigidity floor)")
    if off_diagonal_excess:
        excess = np.asarray(off_diagonal_excess)
        print(f"\noff-diagonal EXCESS over each row's own diagonal: "
              f"median {np.median(excess):+.2f}  p90 {np.percentile(excess, 90):+.2f}  "
              f"max {excess.max():+.2f} (percentage points)")
        print("PASS: every row is flat -- a subset mismatch cannot deform a skeleton."
              if excess.max() < 0.5 else
              "FAIL: rows are not flat -- the position gain is still subset-dependent.")

    print("\namplitude of real clips in canonical space (1.0 = the calibration target)")
    print("(sampled clips only -- pooled over a subset's WHOLE corpus the mean is 0 by "
          "construction,\n so max|mean| here is sampling noise divided by the gain, not a DC "
          "offset in the table)")
    print(f"{'subset':<14}{'pos':>8}{'rot':>8}{'vel':>8}{'max|mean|':>12}{'samples':>12}")
    for subset, row in sorted(amplitude_table(cond, sampled, subsets).items()):
        print(f"{subset:<14}{row['pos']:>8.2f}{row['rot']:>8.2f}{row['vel']:>8.2f}"
              f"{row['max_abs_mean']:>12.3f}{row['frames_joints']:>12}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
