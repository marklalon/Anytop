#!/usr/bin/env python3
"""
Unified Preprocessing + Validation Workflow
============================================
Automatically chains AnyTop dataset creation with validation:
    1. Preprocessing: Incremental by default - keyed on source anim files, so only newly
       added animations are processed while clips already on disk are kept (mean/std are
       recomputed over the merged set). --overwrite forces a full (re)build of the target set.
    2. Validation: Validates the preprocessed dataset

Usage:
    python preprocess_and_validate.py [OPTIONS]

Options:
    --validate-only                      Skip preprocessing, only validate existing dataset
    --re-encode-joint-names-only         Skip preprocessing and validation, only re-encode joint names into cond.npy
    --skip-validate                      Skip validation step (faster for CI)
    --skip-orientation-check             Skip T-pose face-orientation validation during dataset checks
    --overwrite                          Reprocess every targeted object, deleting existing outputs first (a full wipe when no --filter is set). Without it, already-processed objects are skipped.
    --filter PATTERN                     Comma/semicolon-separated case-insensitive glob(s) restricting which object names are considered for processing
    --object-workers N                   Concurrent characters to preprocess (default: 16)
    --sample-count N                     Limit file validation to first N motions (0=all, default: 0)
    --orientation-threshold-deg DEG      Maximum allowed T-pose face-orientation delta from the nearest cardinal XZ axis (+x/-x/+z/-z) before warning (default: 15.0)
    --motion-orientation-threshold DEG   Maximum allowed first/last-frame recovered-facing delta from T-pose facing before warning (default: 45.0)

Examples:
    # Default workflow: process only newly added source animations -> validate
    python preprocess_and_validate.py

    # Force a full rebuild of every object
    python preprocess_and_validate.py --overwrite

    # Validate only (assumes preprocessing already done)
    python preprocess_and_validate.py --validate-only

    # Validate only, skipping orientation check
    python preprocess_and_validate.py --validate-only --skip-orientation-check

    # Preprocess without validation
    python preprocess_and_validate.py --skip-validate

    # Re-encode joint names only (fast, no motion re-export)
    python preprocess_and_validate.py --re-encode-joint-names-only

    # Incrementally add new animations for matching objects (e.g. new Horse clips)
    python preprocess_and_validate.py --filter "Horse"

    # Force-rebuild only the objects matching a wildcard, preserving the rest
    python preprocess_and_validate.py --overwrite --filter "Raptor*,*Bear*" --object-workers 4

    # Remove motions matching a wildcard pattern from the dataset
    python preprocess_and_validate.py --rm "Horse_Run*"

    # Remove motions within a specific species only, and purge species if emptied
    python preprocess_and_validate.py --rm "Horse_Run*;Dog_Jump*" --filter "Horse,Dog"

    # Fast CI workflow (skip validation after preprocessing)
    python preprocess_and_validate.py --skip-validate
"""

import argparse
import fnmatch
import json
import os
import sys
import subprocess
import shutil
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

ANYTOP_DIR = Path(__file__).resolve().parent

# Make the bundled truebones helpers importable directly (param_utils, truebones_utils.motion_labels).
_TRUEBONES_DIR = ANYTOP_DIR / "data_loaders" / "truebones"
for _path in (_TRUEBONES_DIR, _TRUEBONES_DIR / "truebones_utils"):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from param_utils import BVHS_DIR, MOTION_DIR, MOTION_TAGS_FILE, SPECIES_TAGS_FILE, get_dataset_dir, get_raw_data_dir  # noqa: E402
from truebones_utils.motion_labels import load_motion_metadata, write_motion_metadata  # noqa: E402


def _discover_all_objects(raw_data_dir: str = "") -> tuple[str, ...]:
    """Full object universe, discovered by scanning the raw source data directory.

    Each top-level subdirectory of the raw Truebones folder is one object type.
    This is the same enumeration ``create_data_samples`` uses, so the workflow
    operates over exactly the objects present on disk; ``--filter`` narrows it.
    """
    resolved_raw_data_dir = Path(get_raw_data_dir(raw_data_dir or None))
    return tuple(sorted(p.name for p in resolved_raw_data_dir.iterdir() if p.is_dir()))


@dataclass
class PreservedSideArtifacts:
    cond: dict[str, dict[str, object]] = field(default_factory=dict)
    motion_metadata: dict[str, dict[str, object]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Warning collector — captures [WARN] / [warn] lines during preprocessing
# and prints a summary before STEP 1 finishes.
# ---------------------------------------------------------------------------
class _WarnCollector:
    """Collect [WARN] messages by patching known warning emitters.

    Usage::

        collector = _WarnCollector()
        collector.install()
        try:
            ...  # preprocessing code that may emit [WARN]
        finally:
            collector.summarize()
            collector.uninstall()
    """

    def __init__(self):
        self.messages: list[str] = []
        self._patches: list[tuple[object, str, object]] = []  # (module, attr, original)
        self._summarizing = False  # gate to prevent recursive capture during summarize()

    def install(self):
        """Patch known warning emitters to silently collect (no immediate print)."""
        import sys as _sys

        # 1) animation_utils._warn  →  suppress & collect
        try:
            from data_loaders.truebones.truebones_utils import animation_utils as _au
            self._patches.append((_au, '_warn', _au._warn))
            _au._warn = lambda msg: self.messages.append(msg)
        except Exception:
            pass

        # 2) face_orientation._emit_degenerate_facing_warning  →  suppress & collect
        try:
            from data_loaders.truebones.truebones_utils import face_orientation as _fo
            self._patches.append((_fo, '_emit_degenerate_facing_warning', _fo._emit_degenerate_facing_warning))
            _fo._emit_degenerate_facing_warning = lambda ot, wk, msg: self.messages.append(msg)
        except Exception:
            pass

        # 3) Anything else printed to stdout containing "[WARN]"  →  swallow & collect.
        self._stdout_patch = _StdoutWarnCapture(self)
        self._stdout_patch.install()

    def uninstall(self):
        """Restore all original emitters."""
        for mod, attr, original in self._patches:
            try:
                setattr(mod, attr, original)
            except Exception:
                pass
        self._patches.clear()
        if hasattr(self, '_stdout_patch'):
            self._stdout_patch.uninstall()

    def summarize(self):
        """Print a summary block of every collected warning at the end of STEP 1.

        Temporarily disables the stdout wrapper so the summary itself is not
        re-captured recursively.
        """
        if not self.messages:
            return
        # Prevent recursive capture of our own [WARN] output
        self._summarizing = True
        if hasattr(self, '_stdout_patch'):
            self._stdout_patch.uninstall()
        try:
            self._print_summary()
        finally:
            if hasattr(self, '_stdout_patch'):
                self._stdout_patch.install()
            self._summarizing = False

    def _print_summary(self):
        seen: set[str] = set()
        print()
        print('=' * 70)
        print(f"  PREPROCESSING WARNINGS ({len(self.messages)} total)")
        print('=' * 70)
        for msg in self.messages:
            key = msg.strip().lower()
            if key not in seen:
                print(f"  \x1b[33m[WARN]\x1b[0m {msg}")
                seen.add(key)


class _StdoutWarnCapture:
    """Wraps sys.stdout.  Lines containing '[WARN]' are swallowed (not shown)
    and collected; all other output passes through transparently."""

    def __init__(self, collector: _WarnCollector):
        self._collector = collector
        self._original = None

    def install(self):
        import sys as _sys
        self._original = _sys.stdout
        _sys.stdout = self

    def uninstall(self):
        import sys as _sys
        if self._original is not None and _sys.stdout is self:
            _sys.stdout = self._original
        self._original = None

    def write(self, text: str):
        if self._collector._summarizing:
            # Pass through unchanged during summary printing
            if self._original is not None:
                self._original.write(text)
            return
        if '[WARN]' in text.upper():
            # Suppress & collect — do NOT forward to original stdout.
            # Strip ANSI escapes first, then the [WARN] annotation, so
            # summarize() can add a clean uniform prefix.
            import re as _re
            clean = _re.sub(r'\x1b\[[0-9;]*m', '', text)          # strip ANSI
            clean = _re.sub(r'(?i)\[WARN\]\s*', '', clean).strip()  # strip [WARN]
            if clean:
                self._collector.messages.append(clean)
        else:
            if self._original is not None:
                self._original.write(text)

    def flush(self):
        if self._original is not None:
            self._original.flush()


def _resolve_dataset_paths(dataset_dir: str = "") -> tuple[Path, Path, Path, Path]:
    dataset_dir_path = Path(get_dataset_dir(dataset_dir or None))
    return (
        dataset_dir_path,
        dataset_dir_path / MOTION_DIR,
        dataset_dir_path / BVHS_DIR,
        dataset_dir_path / "joint_name_inspection",
    )


def _parse_filter_patterns(object_filter: str) -> list[str]:
    """Split a comma/semicolon-separated wildcard filter into individual glob patterns."""
    if not object_filter:
        return []
    return [p.strip() for p in object_filter.replace(";", ",").split(",") if p.strip()]


def _matches_any(name: str, patterns: list[str]) -> bool:
    """Case-insensitive fnmatch against any of the supplied glob patterns."""
    name_lower = name.lower()
    return any(fnmatch.fnmatch(name_lower, pattern.lower()) for pattern in patterns)


def _resolve_target_object_types(object_filter: str = "", raw_data_dir: str = "") -> tuple[str, ...]:
    all_objects = _discover_all_objects(raw_data_dir)
    patterns = _parse_filter_patterns(object_filter)
    if not patterns:
        return all_objects
    return tuple(obj for obj in all_objects if _matches_any(obj, patterns))


def _path_targets_object_type(path: Path, target_object_types: tuple[str, ...]) -> bool:
    stem = path.stem
    return any(stem == t or stem.startswith(f"{t}_") for t in target_object_types)


def _collect_targeted_files(directory: Path | None, target_object_types: tuple[str, ...]) -> list[Path]:
    if directory is None or not directory.exists():
        return []
    return [
        p for p in sorted(directory.iterdir())
        if p.is_file() and _path_targets_object_type(p, target_object_types)
    ]


def _collect_nonempty_directories(*directories: Path | None) -> list[Path]:
    return [d for d in directories if d is not None and d.exists() and any(d.iterdir())]


def _capture_preserved_side_artifacts(
    dataset_dir_path: Path,
    target_object_types: tuple[str, ...],
) -> PreservedSideArtifacts:
    preserved = PreservedSideArtifacts()

    cond_path = dataset_dir_path / "cond.npy"
    if cond_path.exists():
        current_cond = dict(np.load(cond_path, allow_pickle=True).item())
        preserved.cond = {
            str(obj): obj_cond
            for obj, obj_cond in current_cond.items()
            if str(obj) not in target_object_types
        }

    motions_dir = dataset_dir_path / MOTION_DIR
    # Tolerate clips that still lack hand-labeled action_tags: this is a preserve-only
    # read, the strict tag check belongs to artifact regeneration / training, not here.
    for motion_name, entry in load_motion_metadata(dataset_dir_path, require_action_tags=False).items():
        if not (motions_dir / motion_name).exists():
            continue
        object_type = str(entry.get("object_type") or Path(motion_name).stem.split("_", 1)[0])
        if object_type in target_object_types:
            continue
        preserved.motion_metadata[motion_name] = dict(entry)

    return preserved


def _merge_preserved_side_artifacts(dataset_dir_path: Path, preserved: PreservedSideArtifacts) -> None:
    if not preserved.cond and not preserved.motion_metadata:
        return

    cond_path = dataset_dir_path / "cond.npy"
    current_cond: dict[str, dict[str, object]] = {}
    if cond_path.exists():
        current_cond = dict(np.load(cond_path, allow_pickle=True).item())
    for obj, obj_cond in preserved.cond.items():
        current_cond.setdefault(obj, obj_cond)
    if current_cond:
        np.save(cond_path, current_cond)

    motions_dir = dataset_dir_path / MOTION_DIR
    # Carry-forward read only; freshly preprocessed clips may not be hand-labeled yet.
    # Artifact regeneration backfills inferred tags and re-applies the strict check.
    current_metadata = load_motion_metadata(dataset_dir_path, require_action_tags=False)
    for motion_name, entry in preserved.motion_metadata.items():
        if motion_name in current_metadata:
            continue
        if not (motions_dir / motion_name).exists():
            continue
        current_metadata[motion_name] = dict(entry)
    if current_metadata:
        total_clips = sum(1 for _ in motions_dir.glob("*.npy"))
        write_motion_metadata(dataset_dir_path, current_metadata, total_clips)


def _confirm_yes_no(prompt: str) -> bool:
    while True:
        response = input(prompt).strip().lower()
        if response in ("yes", "y"):
            return True
        if response in ("no", "n"):
            return False
        print("Invalid response. Please enter 'yes', 'y', 'no', or 'n'.")


def _delete_paths(paths: list[Path]) -> bool:
    try:
        for path in paths:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
            print(f"  [OK] Deleted {path}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to delete old data: {e}")
        return False


def check_and_clean_old_data(
    dataset_dir: str = "",
    object_filter: str = "",
    raw_data_dir: str = "",
    overwrite: bool = False,
) -> tuple[bool, PreservedSideArtifacts, tuple[str, ...]]:
    """
    Resolve which object types to (re)process and clean up stale outputs.

    Default (incremental): processing is keyed on source anim files - only objects with
    at least one not-yet-processed source file need work, and create_data_samples skips
    the already-processed source files within them while self-merging the existing
    dataset (so nothing is deleted and no side artifacts are preserved here). ``--overwrite``
    instead (re)builds the whole target set, deleting matching outputs first (a full wipe
    when no ``--filter`` is set). A non-empty ``object_filter`` narrows the target universe
    in both modes.

    Returns (should_proceed, preserved_side_artifacts, objects_to_process).
    """
    dataset_dir_path, motions_dir, bvhs_dir, joint_name_inspection_dir = _resolve_dataset_paths(dataset_dir)
    target_object_types = _resolve_target_object_types(object_filter, raw_data_dir)

    if not overwrite:
        # Incremental: a cheap (no-geometry) scan finds objects with new source files.
        if str(ANYTOP_DIR.parent) not in sys.path:
            sys.path.insert(0, str(ANYTOP_DIR.parent))
        from data_loaders.truebones.truebones_utils.motion_process import find_new_source_files

        new_sources = find_new_source_files(
            target_object_types, str(dataset_dir_path), raw_data_dir or None
        )
        objects_to_process = tuple(obj for obj in target_object_types if obj in new_sources)
        up_to_date = [obj for obj in target_object_types if obj not in new_sources]
        if up_to_date:
            print(f"\nUp to date (no new source files): {len(up_to_date)} object(s) skipped.")
        if objects_to_process:
            total_new = sum(len(new_sources[obj]) for obj in objects_to_process)
            print(
                f"Incremental: {total_new} new source file(s) across "
                f"{len(objects_to_process)} object(s): {', '.join(objects_to_process)}\n"
            )
        # create_data_samples self-merges the existing dataset, so no preservation needed.
        return True, PreservedSideArtifacts(), objects_to_process

    # Overwrite: a filter narrows the run to matching objects; without one we do a full wipe.
    is_full_refresh = not _parse_filter_patterns(object_filter)
    objects_to_process = target_object_types

    if is_full_refresh:
        paths_to_delete = _collect_nonempty_directories(motions_dir, bvhs_dir, joint_name_inspection_dir)
        preserved = PreservedSideArtifacts()
        title = "WARNING: Old preprocessed data detected"
        summary = [
            f"Dataset directory: {dataset_dir_path}",
            "--overwrite with no --filter selected, using full dataset rebuild",
            *[f"  - {p} contains existing data" for p in paths_to_delete],
        ]
    else:
        preserved = _capture_preserved_side_artifacts(dataset_dir_path, target_object_types)
        targeted = [
            ("motion file(s)", motions_dir, _collect_targeted_files(motions_dir, target_object_types)),
            ("BVH file(s)", bvhs_dir, _collect_targeted_files(bvhs_dir, target_object_types)),
            ("inspection file(s)", joint_name_inspection_dir, _collect_targeted_files(joint_name_inspection_dir, target_object_types)),
        ]
        paths_to_delete = [p for _, _, files in targeted for p in files]
        title = "WARNING: Existing preprocessed files detected for selected object types"
        summary = [
            f"Dataset directory: {dataset_dir_path}",
            f"Object types to rebuild ({len(target_object_types)}): {', '.join(target_object_types)}",
            *[f"  - {dir_path}: {len(files)} matching {label}" for label, dir_path, files in targeted if files],
        ]

    if not paths_to_delete:
        return True, preserved, objects_to_process

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    for line in summary:
        print(line)
    print("\nDo you want to delete the matching files and proceed with preprocessing?")

    if not _confirm_yes_no("Enter 'yes' to delete and continue, or 'no' to abort: "):
        print("\nPreprocessing aborted.")
        return False, preserved, objects_to_process

    print("\nDeleting...")
    if not _delete_paths(paths_to_delete):
        print("Aborting preprocessing.")
        return False, preserved, objects_to_process
    print("Done.\n")
    return True, preserved, objects_to_process


def run_preprocessing(
    objects: list[str],
    object_workers: int,
    raw_data_dir: str = "",
    dataset_dir: str = "",
    filter_min_length: int = 10,
    resample_min_length: int = 20,
    incremental: bool = False,
) -> int:
    """Run the AnyTop dataset preprocessing in-process over the given object list."""
    print("\n" + "=" * 70)
    print("STEP 1: PREPROCESSING - Creating AnyTop dataset")
    print("=" * 70 + "\n")

    objects = list(objects)
    print(f"Preprocessing {len(objects)} object(s): {', '.join(objects) or '(none)'}\n")

    if str(ANYTOP_DIR.parent) not in sys.path:
        sys.path.insert(0, str(ANYTOP_DIR.parent))

    from data_loaders.truebones.truebones_utils.motion_process import (
        DatasetPreprocessingError,
        create_data_samples,
    )

    collector = _WarnCollector()
    collector.install()
    try:
        create_data_samples(
            objects=objects,
            dataset_dir=dataset_dir or None,
            raw_data_dir=raw_data_dir or None,
            filter_min_length=filter_min_length,
            resample_min_length=resample_min_length,
            object_workers=object_workers,
            incremental=incremental,
        )
        collector.summarize()
        return 0
    except DatasetPreprocessingError:
        collector.summarize()
        return 1
    except Exception as e:
        print(f"ERROR: Failed to preprocess dataset: {e}")
        import traceback
        traceback.print_exc()
        collector.summarize()
        return 1
    finally:
        collector.uninstall()


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    """Load a JSONL file as a list of dicts. Returns empty list if missing."""
    if not path.exists():
        return []
    entries: list[dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _write_jsonl(path: Path, entries: list[dict[str, object]]) -> None:
    """Write a list of dicts as a JSONL file."""
    with open(path, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _collect_species_from_motions(motions_dir: Path) -> dict[str, set[str]]:
    """Return {species: {motion_filename, ...}} for all .npy files in motions_dir."""
    species_map: dict[str, set[str]] = {}
    if not motions_dir.exists():
        return species_map
    for p in motions_dir.glob("*.npy"):
        stem = p.stem
        species = stem.split("_", 1)[0]
        species_map.setdefault(species, set()).add(p.name)
    return species_map


def run_remove_motions(
    dataset_dir: str = "",
    object_filter: str = "",
    rm_pattern: str = "",
    raw_data_dir: str = "",
) -> int:
    """Remove motions matching *rm_pattern* from the preprocessed dataset.

    Supports fnmatch wildcards against motion filenames (e.g. ``Horse_Run*``,
    ``Dog_*``, ``*Dead*``). Combine with ``--filter`` to scope to specific species.
    When **all** motions of a species are deleted the species entry is also purged
    from ``cond.npy``, ``species_tags.jsonl``, ``joint_name_inspection/``, and the
    dataset ``cache/`` files.
    """

    dataset_dir_path, motions_dir, bvhs_dir, joint_name_inspection_dir = _resolve_dataset_paths(dataset_dir)
    target_species = _resolve_target_object_types(object_filter, raw_data_dir)

    patterns = _parse_filter_patterns(rm_pattern)
    if not patterns:
        print("ERROR: --rm requires a non-empty pattern (e.g. --rm 'Horse_Run*')")
        return 1

    # Gather all motions currently on disk, scoped by --filter if provided.
    all_species_motions = _collect_species_from_motions(motions_dir)
    if object_filter:
        all_species_motions = {s: m for s, m in all_species_motions.items() if s in target_species}

    # Match motions against rm_pattern
    to_delete: list[str] = []
    for species, motions in sorted(all_species_motions.items()):
        for mname in sorted(motions):
            if _matches_any(mname, patterns):
                to_delete.append(mname)

    if not to_delete:
        print(f"No motions matched pattern '{rm_pattern}'" + (f" within species: {', '.join(target_species)}" if object_filter else ""))
        return 0

    # --- Summary ---
    affected_species = sorted({m.split("_", 1)[0] for m in to_delete})
    species_remaining: dict[str, int] = {}
    empty_species: set[str] = set()
    for species in affected_species:
        total = len(all_species_motions.get(species, set()))
        deleted = sum(1 for m in to_delete if m.startswith(f"{species}_"))
        remaining = total - deleted
        species_remaining[species] = remaining
        if remaining <= 0:
            empty_species.add(species)

    print("\n" + "=" * 70)
    print("MOTIONS TO REMOVE")
    print("=" * 70)
    print(f"Dataset directory : {dataset_dir_path}")
    print(f"Pattern           : {rm_pattern}")
    if object_filter:
        print(f"Species filter    : {object_filter}  ->  {', '.join(target_species)}")
    print(f"Motions to delete : {len(to_delete)}")
    print(f"Affected species  : {', '.join(affected_species)}")
    if empty_species:
        print(f"\n⚠  Species that will be EMPTIED (ALL motions removed): {', '.join(sorted(empty_species))}")
        print("   Their cond.npy entries, species_tags.jsonl, inspection files,")
        print("   and cache files will also be purged.")
    print()

    # List a preview
    preview_n = min(len(to_delete), 20)
    for m in to_delete[:preview_n]:
        print(f"  - {m}")
    if len(to_delete) > preview_n:
        print(f"  ... and {len(to_delete) - preview_n} more")
    print()

    if not _confirm_yes_no("Enter 'yes' to delete these motions, or 'no' to abort: "):
        print("\nRemoval aborted.")
        return 0

    # --- Delete motion .npy files ---
    print("\nDeleting motion files...")
    deleted_count = 0
    for mname in to_delete:
        mp = motions_dir / mname
        if mp.exists():
            mp.unlink()
            deleted_count += 1
    print(f"  [OK] Deleted {deleted_count} motion file(s)")

    # --- Delete corresponding .bvh files ---
    if bvhs_dir.exists():
        bvh_deleted = 0
        for mname in to_delete:
            bvh_name = Path(mname).stem + ".bvh"
            bp = bvhs_dir / bvh_name
            if bp.exists():
                bp.unlink()
                bvh_deleted += 1
        if bvh_deleted:
            print(f"  [OK] Deleted {bvh_deleted} BVH file(s)")

    # --- Delete corresponding inspection files ---
    if joint_name_inspection_dir.exists():
        insp_deleted = 0
        for mname in to_delete:
            insp_name = Path(mname).stem + ".png"
            ip = joint_name_inspection_dir / insp_name
            if ip.exists():
                ip.unlink()
                insp_deleted += 1
        if insp_deleted:
            print(f"  [OK] Deleted {insp_deleted} inspection file(s)")

    # --- Update motion_metadata.json ---
    metadata = load_motion_metadata(dataset_dir_path, require_action_tags=False)
    if metadata:
        removed_meta = 0
        for mname in to_delete:
            if mname in metadata:
                del metadata[mname]
                removed_meta += 1
        total_clips = sum(1 for _ in motions_dir.glob("*.npy")) if motions_dir.exists() else 0
        write_motion_metadata(dataset_dir_path, metadata, total_clips)
        if removed_meta:
            print(f"  [OK] Removed {removed_meta} entries from motion_metadata.json (total: {total_clips})")

    # --- Update motion_tags.jsonl ---
    tags_path = dataset_dir_path / MOTION_TAGS_FILE
    if tags_path.exists():
        entries = _load_jsonl(tags_path)
        delete_set = set(to_delete)
        new_entries = [e for e in entries if e.get("clip", "") not in delete_set]
        if len(new_entries) != len(entries):
            _write_jsonl(tags_path, new_entries)
            print(f"  [OK] Removed {len(entries) - len(new_entries)} entries from motion_tags.jsonl")

    # --- Handle species that became empty ---
    if empty_species:
        print(f"\nCleaning up emptied species: {', '.join(sorted(empty_species))}")

        # cond.npy
        cond_path = dataset_dir_path / "cond.npy"
        if cond_path.exists():
            cond = dict(np.load(cond_path, allow_pickle=True).item())
            cond_removed = 0
            for species in empty_species:
                if species in cond:
                    del cond[species]
                    cond_removed += 1
            if cond_removed:
                if cond:
                    np.save(cond_path, cond)
                else:
                    cond_path.unlink()
                print(f"  [OK] Removed {cond_removed} species from cond.npy" + (" (deleted — no species remaining)" if not cond else ""))

        # species_tags.jsonl
        st_path = dataset_dir_path / SPECIES_TAGS_FILE
        if st_path.exists():
            st_entries = _load_jsonl(st_path)
            new_st = [e for e in st_entries if e.get("species", "") not in empty_species]
            if len(new_st) != len(st_entries):
                _write_jsonl(st_path, new_st)
                print(f"  [OK] Removed {len(st_entries) - len(new_st)} species from species_tags.jsonl")

        # joint_name_inspection/ per-species .json files
        if joint_name_inspection_dir.exists():
            for species in empty_species:
                sp = joint_name_inspection_dir / f"{species}.json"
                if sp.exists():
                    sp.unlink()
                    print(f"  [OK] Deleted joint_name_inspection/{species}.json")

        # Cache: delete action_tags_cache.json (it will be regenerated)
        cache_dir = dataset_dir_path / "cache"
        if cache_dir.exists():
            at_cache = cache_dir / "action_tags_cache.json"
            if at_cache.exists():
                at_cache.unlink()
                print(f"  [OK] Deleted cache/action_tags_cache.json (will be regenerated)")
            ml_cache = cache_dir / "motion_lengths.npy"
            if ml_cache.exists():
                ml_cache.unlink()
                print(f"  [OK] Deleted cache/motion_lengths.npy (will be regenerated)")

    # --- Regenerate side artifacts to recompute mean/std after deletion ---
    print("\nRegenerating dataset side artifacts...")
    ret = run_regenerate_side_artifacts(
        dataset_dir,
        recompute_stats=bool(deleted_count),
        recompute_stats_objects=tuple(affected_species) if deleted_count else (),
    )
    if ret != 0:
        print("\n[WARN] Side artifact regeneration returned non-zero exit code.")

    print("\n[OK] Removal complete.")
    return 0


def run_regenerate_side_artifacts(
    dataset_dir: str = "",
    preserved_side_artifacts: PreservedSideArtifacts | None = None,
    t5_model: str = "t5-base",
    recompute_stats: bool = False,
    recompute_stats_objects: tuple[str, ...] = (),
) -> int:
    """Regenerate non-motion dataset artifacts without re-preprocessing motions.

    ``recompute_stats`` rebuilds per-object mean/std over every clip on disk; required
    after incremental preprocessing, where create_data_samples only saw the newly added
    clips and therefore wrote provisional stats. ``recompute_stats_objects`` scopes that
    recompute to the touched objects so an incremental run does not re-read every other
    species' clips (their carried-forward stats are unchanged)."""

    try:
        dataset_dir_path = Path(get_dataset_dir(dataset_dir or None))
        if preserved_side_artifacts:
            _merge_preserved_side_artifacts(
                dataset_dir_path,
                preserved_side_artifacts,
            )

        cmd = [
            sys.executable,
            str(ANYTOP_DIR / "tools" / "regenerate_dataset_artifacts.py"),
            "--dataset-dir",
            str(dataset_dir_path),
            "--t5-model",
            t5_model,
        ]
        if recompute_stats:
            cmd.append("--recompute-stats")
            if recompute_stats_objects:
                cmd += ["--recompute-stats-objects", ",".join(recompute_stats_objects)]

        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(ANYTOP_DIR.parent) + os.pathsep + existing_pythonpath

        result = subprocess.run(cmd, cwd=str(ANYTOP_DIR), capture_output=False, env=env)
        if result.returncode != 0:
            return result.returncode

        return 0
    except Exception as e:
        print(f"ERROR: Failed to regenerate dataset artifacts: {e}")
        import traceback
        traceback.print_exc()
        return 1


def run_validation(
    skip_orientation_check: bool,
    orientation_threshold_deg: float,
    sample_count: int,
    dataset_dir: str = "",
    motion_orientation_threshold: float = 45.0,
) -> int:
    """Run dataset validation."""
    print("\n" + "=" * 70)
    print("STEP 2: VALIDATION - Checking preprocessed dataset")
    print("=" * 70 + "\n")

    # The workflow always operates over every object, so validate the whole dataset.
    objects_subset = "all"

    # Ensure parent of Anytop/ is on sys.path so `from Anytop.utils...` imports work
    if str(ANYTOP_DIR.parent) not in sys.path:
        sys.path.insert(0, str(ANYTOP_DIR.parent))

    # Import and call validate_anytop_dataset.py main() directly instead of subprocess
    sys.path.insert(0, str(ANYTOP_DIR / "utils"))
    from validate_anytop_dataset import (
        resolve_dataset_dir,
        print_ok,
        print_warn,
        ValidationError,
    )
    from data_loaders.truebones.truebones_utils.motion_process import ROOT_XZ_STRIP_THRESHOLD

    # Resolve dataset directory
    dataset_dir = resolve_dataset_dir(dataset_dir or None)

    print_ok(f"dataset_dir: {dataset_dir}")
    print_ok(f"objects_subset: {objects_subset}")
    print_ok(f"file_validation_scope: {'all files' if sample_count == 0 else f'first {sample_count} files'}")

    from validate_anytop_dataset import (
        prepare_dataset_for_validation,
        read_required_artifacts,
        validate_metadata,
        validate_cond_file,
        validate_motion_files,
        validate_motion_metadata,
        validate_positions_error_file,
    )

    try:
        prepare_dataset_for_validation(
            dataset_dir,
            objects_subset,
            sample_count,
        )

        motions_dir, bvhs_dir, cond_path, metadata_path, positions_error_path = read_required_artifacts(dataset_dir)
        cond = validate_cond_file(cond_path, objects_subset)
        motion_files = sorted(motions_dir.glob("*.npy"))
        validate_metadata(metadata_path, motion_files, cond)
        validate_motion_metadata(dataset_dir, motion_files, cond)
        validate_motion_files(
            motions_dir,
            bvhs_dir,
            cond,
            sample_count,
            ROOT_XZ_STRIP_THRESHOLD,
            motion_orientation_threshold=motion_orientation_threshold,
        )

        if skip_orientation_check:
            print_warn("skipping T-pose face-orientation validation by request")
        else:
            from validate_anytop_dataset import validate_tpose_orientation
            validate_tpose_orientation(cond, orientation_threshold_deg)

        validate_positions_error_file(positions_error_path)

        print(f"[OK] total motions: {len(motion_files)}")
        print("[PASS] dataset validation completed successfully")
        return 0
    except ValidationError as e:
        print(f"[WARN] dataset validation warning: {e}")
        return 1
    finally:
        # Force-delete split manifests so they are regenerated on next training run.
        # This ensures train/val/test.txt always reflect the current motion files.
        # Execute this after validation (even if validation failed).
        try:
            for split_name in ("train", "val", "test"):
                split_path = dataset_dir / f"{split_name}.txt"
                if split_path.exists():
                    split_path.unlink()
                    print(f"[OK] deleted {split_path.name} (will be regenerated on next training)")
        except Exception as e:
            print(f"[WARN] failed to delete split manifests: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chain preprocessing and validation into a single workflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Skip preprocessing and only validate the existing dataset.",
    )
    parser.add_argument(
        "--re-encode-joint-names-only",
        action="store_true",
        help="Skip preprocessing and validation, only re-encode joint names into cond.npy.",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip validation (faster, useful for CI).",
    )
    parser.add_argument(
        "--skip-orientation-check",
        action="store_true",
        help="Skip T-pose face-orientation validation during dataset checks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Reprocess every targeted object, deleting existing outputs first (a full "
            "wipe when no --filter is set). Without it, preprocessing is incremental: only "
            "newly added source animations are processed; clips already on disk are kept."
        ),
    )
    parser.add_argument(
        "--motion-orientation-threshold",
        default=45.0,
        type=float,
        help="Maximum allowed first/last-frame recovered-facing delta from T-pose facing before warning. Defaults to 45.0.",
    )
    parser.add_argument(
        "--object-workers",
        default=16,
        type=int,
        help="Concurrent characters to preprocess. Defaults to 16.",
    )
    parser.add_argument(
        "--filter",
        dest="object_filter",
        default="",
        type=str,
        help=(
            "Comma/semicolon-separated case-insensitive glob pattern(s) restricting which object "
            "names are considered for processing (e.g. 'Horse', 'Raptor*', '*Bear*,Cat'). Without "
            "a filter every object is considered. Non-matching objects' artifacts are always "
            "preserved. Combine with --overwrite to force-rebuild only the matching objects."
        ),
    )
    parser.add_argument(
        "--orientation-threshold-deg",
        default=15.0,
        type=float,
        help="Maximum allowed T-pose face-orientation delta from the nearest cardinal XZ axis (+x/-x/+z/-z) before warning. Defaults to 15.0.",
    )
    parser.add_argument(
        "--sample-count",
        default=0,
        type=int,
        help="Limit file validation to the first N motions. Use 0 to validate all files.",
    )
    parser.add_argument(
        "--raw-data-dir",
        default="",
        type=str,
        help="Path to raw Truebones FBX folders. If not specified, uses default path.",
    )
    parser.add_argument(
        "--dataset-dir",
        default="",
        type=str,
        help="Output directory for processed dataset. If not specified, uses default path.",
    )
    parser.add_argument(
        "--filter-min-length",
        default=10,
        type=int,
        help="Minimum number of frames a motion clip must have; shorter clips are filtered out. Defaults to 10.",
    )
    parser.add_argument(
        "--resample-min-length",
        default=20,
        type=int,
        help="When a motion has >= filter-min-length but < resample-min-length frames, resample it to resample-min-length frames. 0 disables. Defaults to 20.",
    )
    parser.add_argument(
        "--rm",
        dest="rm_pattern",
        default="",
        type=str,
        help=(
            "Remove motions matching the wildcard pattern from the preprocessed dataset. "
            "Supports fnmatch globs against motion filenames (e.g. 'Horse_Run*', 'Dog_*', "
            "'*Dead*'). Combine with --filter to scope to specific species. When all "
            "motions of a species are deleted, the species is also removed from cond.npy, "
            "species_tags.jsonl, inspection files, and cache files."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_count < 0:
        print("ERROR: --sample-count must be >= 0")
        return 1
    if args.orientation_threshold_deg < 0:
        print("ERROR: --orientation-threshold-deg must be >= 0")
        return 1
    if args.filter_min_length < 0:
        print("ERROR: --filter-min-length must be >= 0")
        return 1
    if args.resample_min_length < 0:
        print("ERROR: --resample-min-length must be >= 0")
        return 1
    if args.resample_min_length > 0 and args.resample_min_length <= args.filter_min_length:
        print("ERROR: --resample-min-length must be > --filter-min-length")
        return 1
    if args.motion_orientation_threshold < 0:
        print("ERROR: --motion-orientation-threshold must be >= 0")
        return 1
    if args.object_filter and not args.validate_only and not args.re_encode_joint_names_only:
        matched = _resolve_target_object_types(args.object_filter, args.raw_data_dir)
        if not matched:
            print(
                f"ERROR: --filter '{args.object_filter}' matched no objects.\n"
                f"Available objects: {', '.join(_discover_all_objects(args.raw_data_dir))}"
            )
            return 1

    # Handle re-encode joint names only mode
    if args.re_encode_joint_names_only:
        return run_regenerate_side_artifacts(
            args.dataset_dir,
        )

    # Handle remove motions mode
    if args.rm_pattern:
        return run_remove_motions(
            args.dataset_dir,
            args.object_filter,
            args.rm_pattern,
            args.raw_data_dir,
        )

    steps_completed = []
    preserved_side_artifacts = PreservedSideArtifacts()
    objects_to_process: tuple[str, ...] = ()

    # Check and clean old data before preprocessing
    if not args.validate_only:
        should_proceed, preserved_side_artifacts, objects_to_process = check_and_clean_old_data(
            args.dataset_dir, args.object_filter, args.raw_data_dir, overwrite=args.overwrite
        )
        if not should_proceed:
            print("\n" + "=" * 70)
            print("Preprocessing skipped due to user abort")
            print("=" * 70)
            return 1

    # Preprocess if not validate-only and there is something new to process.
    if not args.validate_only:
        if not objects_to_process:
            if args.overwrite:
                print("\nNo objects to process (filter matched no objects).")
            else:
                print("\nNo objects to process: every targeted object is up to date "
                      "(no new source files). Use --overwrite to force a rebuild.")
        else:
            incremental = not args.overwrite
            ret = run_preprocessing(
                list(objects_to_process),
                args.object_workers,
                args.raw_data_dir,
                args.dataset_dir,
                filter_min_length=args.filter_min_length,
                resample_min_length=args.resample_min_length,
                incremental=incremental,
            )
            if ret != 0:
                print("\n[FAIL] Preprocessing failed, aborting workflow.")
                return ret

            # Incremental preprocessing wrote provisional mean/std (new clips only), so
            # recompute them during regeneration - but only for the touched objects, since
            # untouched species' clips are unchanged and their stats carry forward intact.
            ret = run_regenerate_side_artifacts(
                args.dataset_dir,
                preserved_side_artifacts=preserved_side_artifacts,
                recompute_stats=incremental,
                recompute_stats_objects=objects_to_process if incremental else (),
            )
            if ret != 0:
                print("\n[FAIL] Side artifact regeneration failed, aborting workflow.")
                return ret

            steps_completed.append("Preprocess")

    # Validate
    if not args.skip_validate:
        ret = run_validation(
            args.skip_orientation_check,
            args.orientation_threshold_deg,
            args.sample_count,
            args.dataset_dir,
            motion_orientation_threshold=args.motion_orientation_threshold,
        )
        # Don't return on validation failure - continue to next step
        steps_completed.append("Validate")

    # Success
    print("\n" + "=" * 70)
    workflow_desc = " ->".join(steps_completed) if steps_completed else "No steps executed"
    print(f"[OK] WORKFLOW COMPLETE: {workflow_desc} succeeded")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())

