#!/usr/bin/env python3
"""
Quick script to regenerate dataset sidecar artifacts without re-preprocessing motions.

This is a lightweight alternative to: python preprocess_and_validate.py --re-encode-joint-names-only

Usage:
    python tools/regenerate_dataset_artifacts.py [--dataset-dir PATH] [--t5-model NAME]

Options:
    --dataset-dir PATH      Path to dataset directory (uses default if not specified)
    --t5-model NAME         T5 model name to use (default: t5-base)

Examples:
    # Regenerate sidecar artifacts with default settings
    python tools/regenerate_dataset_artifacts.py

    # Re-encode with custom dataset directory
    python tools/regenerate_dataset_artifacts.py --dataset-dir /path/to/dataset

    # Re-encode with different T5 model
    python tools/regenerate_dataset_artifacts.py --t5-model t5-large
"""

import argparse
import copy
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ANYTOP_DIR = Path(__file__).resolve().parent.parent
_PARENT_DIR = ANYTOP_DIR.parent
sys.path.insert(0, str(_PARENT_DIR))
sys.path.insert(0, str(ANYTOP_DIR))

# Package-qualified imports only: a short-name import (via a sys.path entry
# inside the package) would create a SECOND copy of these modules in this
# process, each with its own module globals.
from data_loaders.truebones.truebones_utils.motion_labels import (  # noqa: E402
    ACTION_GROUPS,
    build_motion_labels,
    load_motion_metadata,
    write_motion_metadata,
)
from data_loaders.truebones.truebones_utils.motion_process import (  # noqa: E402
    attach_t5_embeddings_to_cond,
    write_joint_name_collision_report,
)
from data_loaders.truebones.truebones_utils.canonical_features import (  # noqa: E402
    mark_canonical_cond_entry,
    accumulate_lnorm_stats,
    finalize_lnorm_stats,
    set_canonical_global_stats,
)
from data_loaders.truebones.truebones_utils.cond_schema import (  # noqa: E402
    load_cond,
    save_cond,
    stamp_dataset_cond,
)
from data_loaders.truebones.truebones_utils.dataset_sources import (  # noqa: E402
    species_lookup_map,
)
from data_loaders.truebones.truebones_utils.param_utils import (  # noqa: E402
    MOTION_DIR,
    MOTION_METADATA_FILE,
    ACTION_LABELS_FILE,
    get_dataset_dir,
)
from data_loaders.truebones.truebones_utils import dataset_tags  # noqa: E402
from data_loaders.truebones.truebones_utils.physics_joint_annotation import (  # noqa: E402
    build_semantic_metadata,
)


from utils.misc import (
    infer_object_type_from_filename,
    normalize_identifier as _normalize_identifier,
)


def _resolve_dataset_dir_path(dataset_dir: str | Path | None) -> Path:
    raw_value = str(dataset_dir) if dataset_dir else None
    return Path(get_dataset_dir(raw_value)).resolve()


def _write_metadata_summary(
    dataset_dir_path: Path,
    object_counts: Counter[str],
    max_joints: int,
    total_clips: int,
    total_frames: int,
) -> None:
    metadata_path = dataset_dir_path / "metadata.txt"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        handle.write(f"max joints: {max_joints}\n")
        handle.write(f"total frames: {total_frames}\n")
        handle.write(f"duration: {total_frames / 12.5 / 60}\n")
        handle.write(f"~~~~ objects_counts - Total: {total_clips} ~~~~\n")
        for object_type in sorted(object_counts):
            handle.write(f"{object_type}: {object_counts[object_type]}\n")


def _rewrite_positions_error_file(dataset_dir_path: Path, motion_entries: dict[str, dict[str, object]]) -> None:
    positions_error_path = dataset_dir_path / "positions_error_rate.txt"
    existing_lines: list[str] = []
    if positions_error_path.exists():
        existing_lines = positions_error_path.read_text(encoding="utf-8").splitlines()

    header = "Position squared error per source clip:"
    existing_entries: list[str] = []
    if existing_lines:
        first_line = existing_lines[0].strip()
        if first_line.startswith(header):
            trailing_entry = first_line[len(header):].strip()
            if trailing_entry:
                existing_entries.append(trailing_entry)
        existing_entries.extend(line.strip() for line in existing_lines[1:] if line.strip())

    motion_signatures = [
        _normalize_identifier(str(entry.get("object_type", "")))
        for entry in motion_entries.values()
    ]
    filtered_lines: list[str] = []
    for line in existing_entries:
        normalized_line = _normalize_identifier(line)
        if any(obj and obj in normalized_line for obj in motion_signatures):
            filtered_lines.append(line)

    output_lines = ["Position squared error per source clip:__artifact_regenerated__: 0.000000"]
    output_lines.extend(filtered_lines)
    if len(output_lines) == 1:
        output_lines.append("__artifact_regenerated_placeholder__: 0.000000")
    positions_error_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def _infer_object_type_from_motion_name(
    motion_name: str,
    cond_lookup,
) -> str:
    """Canonical cond key for a clip filename.

    *cond_lookup* is the ``{filename token: canonical key}`` map, so a species
    whose bare name is unique keeps its plain filename while a collision resolves
    through its qualified token.
    """
    resolved = infer_object_type_from_filename(motion_name, valid_types=cond_lookup)
    if resolved is None:
        resolved = Path(motion_name).stem.split("_", 1)[0]
    return str(resolved)


def _mark_object_feature_spaces(
    rebuilt_cond: dict[str, dict],
) -> None:
    """Ensure cond entries carry canonical feature-space metadata."""

    for object_type, object_cond in rebuilt_cond.items():
        mark_canonical_cond_entry(object_cond)
    if rebuilt_cond:
        print(f"[OK] marked canonical feature space for {len(rebuilt_cond)} species")


def _compute_canonical_stats_per_object_subset(
    rebuilt_cond: dict[str, dict],
    motion_files: list[Path],
) -> None:
    """Compute per-object_subset per-channel standardization statistics and store
    each object_subset's 13-vectors on its member cond entries.

    Each physical clip is encoded into the L-normalized space (rest-centered
    position + per-skeleton size division) and accumulated into the bucket of the
    species' object_subset (the first motion tag in species_tags.jsonl: quadruped
    / biped / multiped / serpentine / aquatic / winged / drifting). Pooling within an
    object_subset (across its species, joints, frames, and clips) keeps the
    resulting mean/std a cross-species constant *per object_subset* (no per-species
    motion prior), so they generalize to held-out species of the same
    object_subset while giving each object_subset its own zero-mean / unit-variance
    calibration -- closer to the behavior the diffusion noise schedule expects than
    a single global constant that averages flapping wings against quadruped gaits.

    There is NO global-pooled fallback: the model is trained exclusively on
    per-object_subset normalized features, so standardizing any species with stats
    pooled across all subsets would place its features in a space the model never
    saw (out-of-distribution at inference). A species that cannot resolve to an
    object_subset with usable clips is therefore a hard error -- the build
    fast-fails listing the offending species. Requires rest geometry (set by
    mark_canonical_cond_entry) to already be present on each cond entry."""

    cond_lookup = species_lookup_map(rebuilt_cond)
    tags = dataset_tags.dataset_tags()
    subset_of = {ot: tags.object_subset_for(ot) for ot in rebuilt_cond}

    subset_accs: dict[str, dict] = {}
    used = 0
    for motion_path in motion_files:
        object_type = _infer_object_type_from_motion_name(motion_path.name, cond_lookup)
        object_cond = rebuilt_cond.get(object_type)
        if object_cond is None:
            continue
        subset = subset_of.get(object_type)
        if not subset:
            # Untagged species cannot be bucketed; it will be caught by the
            # unresolved-species fast-fail below (no global fallback).
            continue
        motion = np.load(motion_path).astype(np.float32, copy=False)
        if motion.ndim != 3 or motion.shape[-1] < 13:
            continue
        try:
            subset_accs[subset] = accumulate_lnorm_stats(motion, object_cond, acc=subset_accs.get(subset))
        except (KeyError, ValueError):
            # cond entry lacks rest geometry (e.g. minimal synthetic fixtures);
            # such clips cannot be encoded, so skip them.
            continue
        used += 1

    usable_accs = {s: a for s, a in subset_accs.items() if a is not None and a["count"] > 0}
    if not usable_accs:
        print("[WARN] no usable motion clips with rest geometry; canonical stats not written")
        return

    subset_stats = {subset: finalize_lnorm_stats(acc) for subset, acc in usable_accs.items()}

    # No global-pooled fallback (would be OOD at inference -- see docstring). Every
    # species MUST resolve to an object_subset that has usable clips.
    unresolved = sorted(
        f"{ot} (object_subset={subset_of.get(ot)!r})"
        for ot in rebuilt_cond
        if subset_of.get(ot) is None or subset_of.get(ot) not in subset_stats
    )
    if unresolved:
        raise ValueError(
            "Cannot compute per-object_subset canonical stats: no usable clips for "
            "the object_subset(s) of these species:\n  " + "\n  ".join(unresolved)
            + "\nEvery species must belong to an object_subset that has at least one "
            "usable motion clip (the model is trained per-object_subset; there is no "
            "global fallback). Add clips for the subset, or register the species in "
            "species_tags.jsonl."
        )

    for object_type, object_cond in rebuilt_cond.items():
        mean, std = subset_stats[subset_of[object_type]]
        set_canonical_global_stats(object_cond, mean, std)

    with np.printoptions(precision=3, suppress=True, linewidth=160):
        print(f"[OK] canonical stats over {used} clip(s) bucketed by object_subset:")
        for subset, (mean, std) in sorted(subset_stats.items()):
            print(
                f"     [{subset}] {usable_accs[subset]['count']} frames-joints"
            )


def _recompute_contact_joints(rebuilt_cond: dict[str, dict]) -> None:
    """Re-infer contact joints for every object using current skeleton info.

    Contact joints depend only on skeleton topology (names, parents, offsets),
    so they can be safely recomputed from cond.npy without re-loading source FBX."""

    for object_type, object_cond in sorted(rebuilt_cond.items()):
        parents = np.asarray(object_cond["parents"], dtype=np.int64)
        offsets = np.asarray(object_cond["offsets"], dtype=np.float64)
        joint_names = list(object_cond["joints_names"])

        semantic_metadata = build_semantic_metadata(
            joint_names,
            parents,
            offsets,
            species_name=object_cond.get("species_name") or object_cond.get("object_type") or object_type,
        )

        old_contact = list(object_cond.get("contact_joints", []))
        new_contact = semantic_metadata["contact_joints"]
        old_source = object_cond.get("contact_joint_source", "")
        new_source = semantic_metadata["contact_joint_source"]

        object_cond["contact_joints"] = new_contact
        object_cond["contact_joint_names"] = semantic_metadata["contact_joint_names"]
        object_cond["contact_joint_source"] = new_source
        object_cond["end_effector_joints"] = semantic_metadata["end_effector_joints"]
        object_cond["end_effector_names"] = semantic_metadata["end_effector_names"]

        if old_contact != new_contact or old_source != new_source:
            print(
                f"[OK] {object_type}: contact_joints {old_contact} -> {new_contact} "
                f"(source: {old_source} -> {new_source})"
            )


def _normalize_object_translation_roots(
    rebuilt_cond: dict[str, dict],
    motion_files: list[Path],
    motion_metadata: dict[str, dict],
    cond_lookup: dict[str, str],
) -> dict[str, int]:
    """Collapse per-motion translation_root_index to one canonical root per object.

    Each motion's metadata already contains a `translation_root_index` from
    preprocessing.  We just aggregate the existing values per object and pick
    the most common one.  Ties fall back to the smaller index for determinism.
    """
    motion_names = {p.name for p in motion_files}
    object_root_counts: dict[str, Counter[int]] = {}

    for motion_name, entry in motion_metadata.items():
        if motion_name not in motion_names:
            continue
        # motion_metadata stores the bare species name; cond is canonically keyed.
        object_type = cond_lookup.get(str(entry.get("object_type", "")))
        if object_type is None or object_type not in rebuilt_cond:
            continue
        if "translation_root_index" not in entry:
            raise RuntimeError(
                f"Motion metadata for '{motion_name}' (object '{object_type}') is missing "
                f"'translation_root_index'. The metadata may be stale or from an older "
                f"preprocessing version. Re-run preprocess_and_validate.py to regenerate it."
            )
        root_index = int(entry["translation_root_index"])
        object_root_counts.setdefault(object_type, Counter())[root_index] += 1

    canonical_roots: dict[str, int] = {}
    for object_type, root_counts in sorted(object_root_counts.items()):
        canonical_root_index = min(
            root_index
            for root_index, count in root_counts.items()
            if count == max(root_counts.values())
        )
        unique_roots = sorted(int(root_index) for root_index in root_counts)
        rebuilt_cond[object_type]["translation_root_index"] = canonical_root_index
        canonical_roots[object_type] = canonical_root_index
        if len(unique_roots) > 1:
            print(
                f"[OK] normalized {object_type} translation_root_index "
                f"from {dict(sorted(root_counts.items()))} to {canonical_root_index}"
            )

    # Ensure every object in rebuilt_cond has a canonical root, even if no
    # motion metadata existed (e.g., freshly created dataset or test fixtures).
    for object_type in sorted(rebuilt_cond.keys()):
        if object_type not in canonical_roots:
            canonical_roots[object_type] = int(
                rebuilt_cond[object_type].get("translation_root_index", 0)
            )

    return canonical_roots


def regenerate_dataset_artifacts(
    dataset_dir: str | Path | None = None,
    t5_model: str = "t5-base",
) -> Path:
    dataset_dir_path = _resolve_dataset_dir_path(dataset_dir)
    # Read this dataset's tag sidecars for the duration of the rebuild. A caller
    # that already configured the same dataset (with, say, an explicit
    # --species-tags-file) keeps its configuration.
    with dataset_tags.using_dataset_dir(dataset_dir_path):
        return _regenerate_dataset_artifacts(dataset_dir_path, t5_model=t5_model)


def _regenerate_dataset_artifacts(dataset_dir_path: Path, t5_model: str = "t5-base") -> Path:
    motions_dir = dataset_dir_path / MOTION_DIR
    cond_path = dataset_dir_path / "cond.npy"

    if not motions_dir.exists():
        raise RuntimeError(f"motions directory not found at {motions_dir}")
    if not cond_path.exists():
        raise RuntimeError(f"cond.npy not found at {cond_path}")

    motion_files = sorted(motions_dir.glob("*.npy"))
    if not motion_files:
        raise RuntimeError(f"no motion files found under {motions_dir}")

    # Fast-fail: motion_metadata.json must exist.  Without it, load_motion_metadata
    # returns {} and the rebuilt metadata will be missing is_loop, source_file,
    # translation_root_index, and other per-clip fields.
    metadata_path = dataset_dir_path / MOTION_METADATA_FILE
    if not metadata_path.exists():
        raise RuntimeError(
            f"{MOTION_METADATA_FILE} not found at {metadata_path}.\n"
            f"This script requires an existing motion_metadata.json to preserve "
            f"is_loop, source_file, translation_root_index, and other per-clip metadata.\n"
            f"If you've deleted it, re-run preprocess_and_validate.py to regenerate "
            f"the full dataset, or restore it from a backup."
        )

    # Fast-fail: action_labels.jsonl must exist (same contract as species_tags.jsonl
    # -- single source of truth, no inference fallback, no auto-creation).
    # load_motion_metadata below also hard-exits when a clip has no entry, but a
    # missing file is reported up front with the fix spelled out.
    action_labels_path = dataset_dir_path / ACTION_LABELS_FILE
    if not action_labels_path.exists():
        raise RuntimeError(
            f"{ACTION_LABELS_FILE} not found at {action_labels_path}.\n"
            f"It is the single source of truth for per-clip action group/label.\n"
            f"Add one entry per clip in motions/ (one "
            f'{{"clip": "<name>.npy", "action_group": "{ACTION_GROUPS[0]}", '
            f'"action_label": "run, gallops with head lowered"}} object per line), '
            f"then re-run."
        )

    existing_cond = load_cond(cond_path)
    existing_motion_metadata = load_motion_metadata(dataset_dir_path)
    existing_lookup = species_lookup_map(existing_cond)
    active_object_types = sorted(
        {
            _infer_object_type_from_motion_name(motion_path.name, existing_lookup)
            for motion_path in motion_files
        }
    )
    missing_object_types = [object_type for object_type in active_object_types if object_type not in existing_cond]
    if missing_object_types:
        raise RuntimeError(
            f"cond.npy is missing object entries required by current motions: {missing_object_types}"
        )

    active_cond = {
        object_type: existing_cond[object_type]
        for object_type in active_object_types
    }
    stale_object_types = sorted(object_type for object_type in existing_cond if object_type not in active_cond)
    if stale_object_types:
        print(
            "[OK] pruned stale cond.npy object entries with no matching motions: "
            + ", ".join(stale_object_types)
        )

    rebuilt_cond = {
        object_type: copy.deepcopy(object_cond)
        for object_type, object_cond in active_cond.items()
    }
    _mark_object_feature_spaces(rebuilt_cond)

    t0 = time.time()
    _recompute_contact_joints(rebuilt_cond)
    print(f"[OK] contact joints recomputed in {time.time() - t0:.1f}s")

    t0 = time.time()
    canonical_translation_roots = _normalize_object_translation_roots(
        rebuilt_cond,
        motion_files,
        existing_motion_metadata,
        species_lookup_map(rebuilt_cond),
    )
    print(f"[OK] translation roots normalized in {time.time() - t0:.1f}s")

    t0 = time.time()

    inspection_dir = dataset_dir_path / "joint_name_inspection"
    if inspection_dir.exists():
        shutil.rmtree(inspection_dir)
    collision_report_path = dataset_dir_path / "joint_name_collision_report.json"
    if collision_report_path.exists():
        collision_report_path.unlink()

    attach_t5_embeddings_to_cond(
        rebuilt_cond,
        str(dataset_dir_path),
        t5_name=t5_model,
        write_collision_report=False,
    )
    print(f"[OK] T5 embeddings attached in {time.time() - t0:.1f}s")
    write_joint_name_collision_report(rebuilt_cond, str(dataset_dir_path))

    t0 = time.time()
    _compute_canonical_stats_per_object_subset(rebuilt_cond, motion_files)
    print(f"[OK] per-object_subset canonical stats computed in {time.time() - t0:.1f}s")

    # Re-stamp on write: this is the dataset's own cond, so entries stay keyed
    # <namespace>/<species> with dataset_root=None ("wherever this file lives").
    save_cond(cond_path, stamp_dataset_cond(rebuilt_cond, dataset_dir_path))

    rebuilt_lookup = species_lookup_map(rebuilt_cond)
    rebuilt_motion_metadata: dict[str, dict[str, object]] = {}
    object_counts: Counter[str] = Counter()
    total_frames = 0
    max_joints = 0
    for motion_path in motion_files:
        motion = np.load(motion_path, mmap_mode="r")
        total_frames += int(motion.shape[0])
        max_joints = max(max_joints, int(motion.shape[1]))

        motion_entry = dict(existing_motion_metadata.get(motion_path.name, {}))
        object_key = _infer_object_type_from_motion_name(motion_path.name, rebuilt_lookup)
        # motion_metadata.json is a per-dataset sidecar and stays keyed by the
        # BARE species name, which is what joins it back to cond.species_name.
        species_name = str(rebuilt_cond[object_key]["species_name"]) if object_key in rebuilt_cond else object_key
        motion_entry.update(build_motion_labels(species_name, motion_name=motion_path.name))
        motion_entry["translation_root_index"] = int(canonical_translation_roots[object_key])
        rebuilt_motion_metadata[motion_path.name] = motion_entry
        object_counts[object_key] += 1

    _write_metadata_summary(
        dataset_dir_path,
        object_counts,
        max_joints,
        len(motion_files),
        total_frames,
    )
    write_motion_metadata(dataset_dir_path, rebuilt_motion_metadata, len(motion_files))
    _rewrite_positions_error_file(dataset_dir_path, rebuilt_motion_metadata)

    return dataset_dir_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate dataset sidecar artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset-dir",
        default="",
        type=str,
        help="Path to dataset directory. If not specified, uses default path.",
    )
    parser.add_argument(
        "--t5-model",
        default="t5-base",
        type=str,
        help="T5 model to use for joint name embeddings (default: t5-base).",
    )
    parser.add_argument(
        "--species-tags-file",
        default="",
        type=str,
        help="Path to the species_tags.jsonl sidecar used for this run.",
    )
    parser.add_argument(
        "--chain-forward-joints-file",
        default="",
        type=str,
        help="Path to the chain_forward_joints.jsonl sidecar used for this run.",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("Regenerating dataset sidecar artifacts")
    print("=" * 70 + "\n")

    try:
        # The entry point owns configuration; the library call only scopes it.
        dataset_tags.configure(
            dataset_dir=args.dataset_dir,
            species_tags_file=args.species_tags_file,
            chain_forward_joints_file=args.chain_forward_joints_file,
        )
        dataset_dir_path = regenerate_dataset_artifacts(
            args.dataset_dir,
            t5_model=args.t5_model,
        )
        cond_path = dataset_dir_path / "cond.npy"
        cond = dict(np.load(cond_path, allow_pickle=True).item())

        print(f"Regenerated dataset artifacts under {dataset_dir_path}")
        sorted_keys = sorted(cond.keys())
        max_shown = 3
        shown = ", ".join(sorted_keys[:max_shown]) + ("..." if len(sorted_keys) > max_shown else "")
        print(f"Saved cond.npy with {len(cond)} objects: {shown}")
        print("[PASS] Dataset sidecar regeneration completed successfully")
        return 0
    except Exception as e:
        print(f"ERROR: Failed to regenerate dataset artifacts: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
