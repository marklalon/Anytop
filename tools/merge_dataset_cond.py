#!/usr/bin/env python3
"""Merge several datasets' ``cond.npy`` into one training cond.

Because schema v4 keys every entry by ``<namespace>/<species>`` and records the
dataset it came from, merging is a plain union -- **no motion file is copied**.
The output is a single ``cond.npy``; training then differs from a single-dataset
run only in which ``--cond_path`` it is given.

The one thing that genuinely has to be recomputed is the per-object_subset
canonical standardization statistics.  Each dataset computes them over its own
clips, and they differ noticeably (quadruped mean[0] 0.0098 vs 0.0573), so a
naive union would leave one subset bucket holding two different normalization
spaces -- exactly the cross-species sharing AnyTop's generalization rests on.
This tool is the only place that can see every source's clips at once, so it
recomputes them here and writes the result onto every merged entry.  The source
datasets' own ``cond.npy`` files are never modified.

Usage:
    python tools/merge_dataset_cond.py \
        --datasets dataset/datasets.jsonl \
        --out dataset/merged/truebones_all/cond.npy \
        [--no-recompute-stats] [--dry-run]
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ANYTOP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ANYTOP_DIR.parent))
sys.path.insert(0, str(ANYTOP_DIR))

from data_loaders.truebones.truebones_utils.canonical_features import (  # noqa: E402
    accumulate_lnorm_stats,
    finalize_lnorm_stats,
    set_canonical_global_stats,
)
from data_loaders.truebones.truebones_utils.cond_schema import (  # noqa: E402
    load_cond,
    save_cond,
    upgrade_cond_dict,
)
from data_loaders.truebones.truebones_utils.dataset_sources import (  # noqa: E402
    COND_FILE,
    bare_species_name,
    build_species_file_tokens,
    load_datasets_manifest,
)
from data_loaders.truebones.truebones_utils.dataset_tags import (  # noqa: E402
    SPECIES_TAGS_FILE,
    build_object_subsets,
    load_species_tags,
)
from data_loaders.truebones.truebones_utils.param_utils import MAX_JOINTS  # noqa: E402

_COLOR_RESET = "\033[0m"
_COLOR_YELLOW = "\033[93m"


# ---------------------------------------------------------------------------
# Cross-source consistency
# ---------------------------------------------------------------------------
def _feature_space_signature(entry) -> tuple:
    """The parts of a cond entry that must agree for two datasets to be mergeable.

    Feature-space and embedding identity only -- anything per-species (joint
    counts, offsets, statistics) is expected to differ.
    """
    joint_meta = entry.get("joints_names_embs_meta") or {}
    species_meta = entry.get("species_emb_meta") or {}
    embs = np.asarray(entry.get("joints_names_embs"))
    return (
        str(entry.get("feature_space")),
        str(entry.get("physical_feature_space")),
        int(joint_meta.get("schema_version", -1)),
        str(joint_meta.get("t5_name")),
        int(embs.shape[1]) if embs.ndim == 2 else -1,
        str(species_meta.get("t5_name")),
    )


def _describe_signature(signature) -> str:
    labels = (
        "feature_space",
        "physical_feature_space",
        "joints_names_embs_meta.schema_version",
        "joints_names_embs_meta.t5_name",
        "embedding_dim",
        "species_emb_meta.t5_name",
    )
    return ", ".join(f"{label}={value!r}" for label, value in zip(labels, signature))


def _check_source_consistency(per_source_entries, reference=None):
    """Fast-fail on any mismatch that would make the merged cond incoherent."""
    for namespace, entries in per_source_entries.items():
        for key, entry in entries.items():
            signature = _feature_space_signature(entry)
            if reference is None:
                reference = (namespace, key, signature)
            elif signature != reference[2]:
                raise SystemExit(
                    "Datasets are not mergeable: feature-space / embedding metadata differs.\n"
                    f"  {reference[0]} / {reference[1]}: {_describe_signature(reference[2])}\n"
                    f"  {namespace} / {key}: {_describe_signature(signature)}"
                )
            joint_count = len(np.asarray(entry["parents"]))
            if joint_count > MAX_JOINTS:
                raise SystemExit(
                    f"{namespace} / {key} has {joint_count} joints, over the MAX_JOINTS={MAX_JOINTS} "
                    "cap the model's padding is built for. Re-preprocess it with skeleton cropping."
                )
    return reference


# ---------------------------------------------------------------------------
# Statistics recomputation
# ---------------------------------------------------------------------------
def _recompute_canonical_stats(merged_cond, sources) -> dict[str, tuple]:
    """Recompute per-object_subset standardization stats over every source's clips.

    Same accumulation as ``regenerate_dataset_artifacts._compute_canonical_stats_per_object_subset``,
    but bucketed across datasets: a merged run trains one shared normalization
    space per body plan, not one per (dataset, body plan).
    """
    subset_of: dict[str, str] = {}
    for namespace_tags, source in ((load_species_tags(Path(s.root) / SPECIES_TAGS_FILE), s) for s in sources):
        for species, tags in namespace_tags.items():
            subset_of[source.key_for(species)] = tags[0].strip().lower()

    unsubsetted = sorted(key for key in merged_cond if key not in subset_of)
    if unsubsetted:
        raise SystemExit(
            f"These species have no {SPECIES_TAGS_FILE} entry in their own dataset, so they "
            "cannot be bucketed:\n  " + "\n  ".join(unsubsetted)
        )

    subset_accs: dict[str, dict] = {}
    used = 0
    skipped = 0
    for source in sources:
        motion_dir = Path(source.motion_dir)
        if not motion_dir.is_dir():
            raise SystemExit(
                f"Cannot recompute statistics: {motion_dir} does not exist. Pass "
                "--no-recompute-stats only to validate the pipeline shape."
            )
        # Bare filename prefixes are unique inside one source, which is where
        # they are matched -- the collisions only appear across sources.
        species_by_name = {
            bare_species_name(key): key for key in merged_cond
            if merged_cond[key]["dataset_namespace"] == source.namespace
        }
        for motion_path in sorted(motion_dir.glob("*.npy")):
            object_key = _match_species(motion_path.name, species_by_name)
            if object_key is None:
                skipped += 1
                continue
            motion = np.load(motion_path).astype(np.float32, copy=False)
            if motion.ndim != 3 or motion.shape[-1] < 13:
                skipped += 1
                continue
            subset = subset_of[object_key]
            try:
                subset_accs[subset] = accumulate_lnorm_stats(
                    motion, merged_cond[object_key], acc=subset_accs.get(subset)
                )
            except (KeyError, ValueError):
                skipped += 1
                continue
            used += 1

    usable = {subset: acc for subset, acc in subset_accs.items() if acc and acc["count"] > 0}
    subset_stats = {subset: finalize_lnorm_stats(acc) for subset, acc in usable.items()}

    # No global-pooled fallback: the model only ever sees per-object_subset
    # normalized features, so a species standardized with pooled stats would be
    # out of distribution at inference.
    unresolved = sorted(
        f"{key} (object_subset={subset_of.get(key)!r})"
        for key in merged_cond
        if subset_of.get(key) not in subset_stats
    )
    if unresolved:
        raise SystemExit(
            "Cannot compute per-object_subset canonical stats: no usable clips for the "
            "object_subset(s) of these species:\n  " + "\n  ".join(unresolved)
        )

    for key, entry in merged_cond.items():
        mean, std = subset_stats[subset_of[key]]
        set_canonical_global_stats(entry, mean, std)

    print(f"[OK] recomputed canonical stats over {used} clip(s) ({skipped} skipped)")
    return subset_stats


def _match_species(motion_name: str, species_by_name: dict[str, str]) -> str | None:
    """Longest-prefix match of ``{species}_...npy`` inside one source."""
    best_key = None
    best_len = -1
    for species, key in species_by_name.items():
        if motion_name.startswith(f"{species}_") and len(species) > best_len:
            best_key, best_len = key, len(species)
    return best_key


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _count_clips(source, merged_cond) -> Counter:
    counts: Counter = Counter()
    motion_dir = Path(source.motion_dir)
    if not motion_dir.is_dir():
        return counts
    species_by_name = {
        bare_species_name(key): key for key in merged_cond
        if merged_cond[key]["dataset_namespace"] == source.namespace
    }
    for motion_path in motion_dir.glob("*.npy"):
        key = _match_species(motion_path.name, species_by_name)
        if key is not None:
            counts[key] += 1
    return counts


def _report(sources, merged_cond, before_stats, after_stats):
    print("\n" + "=" * 70)
    print("Merged cond report")
    print("=" * 70)

    for source in sources:
        keys = [k for k, v in merged_cond.items() if v["dataset_namespace"] == source.namespace]
        clip_counts = _count_clips(source, merged_cond)
        print(
            f"  {source.namespace:<28} {len(keys):>3} species  "
            f"{sum(clip_counts.values()):>5} clips  ({source.portable_root})"
        )

    by_bare: dict[str, list[str]] = defaultdict(list)
    for key in merged_cond:
        by_bare[bare_species_name(key)].append(key)
    collisions = {name: keys for name, keys in by_bare.items() if len(keys) > 1}
    tokens = build_species_file_tokens(merged_cond)
    if collisions:
        print(f"\n  {len(collisions)} bare-name collision(s); output filenames use the qualified token:")
        for name in sorted(collisions):
            for key in collisions[name]:
                print(f"    {key:<40} -> {tokens[key]}")
    else:
        print("\n  No bare-name collisions; every output filename stays the plain species name.")

    if after_stats:
        print("\n  Per-object_subset canonical stats (channel 0), before -> after:")
        for subset in sorted(after_stats):
            after_mean, after_std = after_stats[subset]
            befores = before_stats.get(subset, [])
            before_text = ", ".join(
                f"{namespace}: mean={mean:+.4f} std={std:.4f}" for namespace, mean, std in befores
            )
            print(f"    [{subset}]")
            if before_text:
                print(f"      before  {before_text}")
            print(f"      after   mean={float(after_mean[0]):+.4f} std={float(after_std[0]):.4f}")
    print()


def _collect_before_stats(per_source_entries) -> dict[str, list]:
    """Channel-0 mean/std each source currently carries, per object_subset."""
    before: dict[str, set] = defaultdict(set)
    for namespace, entries in per_source_entries.items():
        for entry in entries.values():
            mean = entry.get("canonical_feature_mean")
            std = entry.get("canonical_feature_std")
            if mean is None or std is None:
                continue
            tags = entry.get("species_tags") or ()
            subset = tags[0].strip().lower() if tags else "?"
            before[subset].add((namespace, round(float(np.asarray(mean)[0]), 6), round(float(np.asarray(std)[0]), 6)))
    return {subset: sorted(values) for subset, values in before.items()}


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------
def merge_dataset_cond(manifest_path, out_path, recompute_stats=True, dry_run=False) -> Path:
    sources = load_datasets_manifest(manifest_path)
    print(f"Merging {len(sources)} dataset(s) from {manifest_path}")

    per_source_entries: dict[str, dict[str, dict]] = {}
    for source in sources:
        cond_path = Path(source.root) / COND_FILE
        if not cond_path.is_file():
            raise SystemExit(f"{source.namespace}: cond.npy not found at {cond_path}")
        species_tags = load_species_tags(Path(source.root) / SPECIES_TAGS_FILE)
        raw = load_cond(cond_path, namespace=source.namespace)
        # Re-stamp: the source cond stores dataset_root=None ("my own directory"),
        # while a merged cond lives elsewhere and must point back at each source.
        entries = upgrade_cond_dict(
            {key: dict(entry, cond_schema_version=0) for key, entry in raw.items()},
            namespace=source.namespace,
            dataset_root=source.root,
            species_tags=species_tags,
            store_root=True,
        )
        entries = {
            key: entry for key, entry in entries.items()
            if source.accepts(entry["species_name"])
        }
        missing_tags = sorted(key for key, entry in entries.items() if not entry["species_tags"])
        if missing_tags:
            raise SystemExit(
                f"{source.namespace}: {SPECIES_TAGS_FILE} is missing entries for "
                + ", ".join(missing_tags)
            )
        per_source_entries[source.namespace] = entries
        print(f"  [OK] {source.namespace}: {len(entries)} species from {cond_path}")

    _check_source_consistency(per_source_entries)

    merged_cond: dict[str, dict] = {}
    for namespace, entries in per_source_entries.items():
        for key, entry in entries.items():
            if key in merged_cond:
                raise SystemExit(f"Duplicate canonical species key across sources: {key}")
            merged_cond[key] = copy.deepcopy(entry)

    before_stats = _collect_before_stats(per_source_entries)
    after_stats: dict[str, tuple] = {}
    if recompute_stats:
        t0 = time.time()
        after_stats = _recompute_canonical_stats(merged_cond, sources)
        print(f"[OK] statistics recomputed in {time.time() - t0:.1f}s")
    else:
        print(
            f"{_COLOR_YELLOW}[WARN] --no-recompute-stats: each species keeps its own dataset's "
            f"normalization statistics, so one object_subset bucket spans two standardization "
            f"spaces. For pipeline validation only, never for a real training run.{_COLOR_RESET}"
        )

    _report(sources, merged_cond, before_stats, after_stats)

    subsets = build_object_subsets({key: entry["species_tags"] for key, entry in merged_cond.items()})
    print(f"  object subsets: " + ", ".join(
        f"{name}={len(members)}" for name, members in sorted(subsets.items()) if members
    ))

    if dry_run:
        print("[DRY RUN] nothing written.")
        return Path(out_path)

    written = save_cond(out_path, merged_cond)
    print(f"[PASS] wrote {len(merged_cond)} species to {written}")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge several datasets' cond.npy into one training cond.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--datasets",
        default="dataset/datasets.jsonl",
        help="Dataset manifest (JSONL). Line order sets bare-name resolution priority.",
    )
    parser.add_argument("--out", required=True, help="Output cond.npy path.")
    parser.add_argument(
        "--no-recompute-stats",
        action="store_true",
        help="Keep each source's own per-object_subset statistics. Escape hatch for quickly "
             "validating the pipeline; not valid for a real training run.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    args = parser.parse_args()

    try:
        merge_dataset_cond(
            args.datasets,
            args.out,
            recompute_stats=not args.no_recompute_stats,
            dry_run=args.dry_run,
        )
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {exc}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
