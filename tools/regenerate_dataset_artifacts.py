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
import json
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
sys.path.insert(0, str(ANYTOP_DIR / "data_loaders" / "truebones"))

from truebones_utils.motion_labels import (  # noqa: E402
    build_motion_labels,
    infer_action_tags_from_clip_name,
    load_motion_metadata,
    load_action_tags,
    write_motion_metadata,
)
from truebones_utils.motion_process import (  # noqa: E402
    attach_t5_embeddings_to_cond,
    write_joint_name_collision_report,
)
from truebones_utils.canonical_features import (  # noqa: E402
    mark_canonical_cond_entry,
    accumulate_lnorm_stats,
    finalize_lnorm_stats,
    set_canonical_global_stats,
)
from truebones_utils.param_utils import (  # noqa: E402
    MOTION_DIR,
    MOTION_METADATA_FILE,
    ACTION_TAGS_FILE,
    get_dataset_dir,
)
from truebones_utils.physics_joint_annotation import (  # noqa: E402
    build_semantic_metadata,
)


from Anytop.utils.misc import (
    infer_object_type_from_filename,
    normalize_identifier as _normalize_identifier,
)


# ---------------------------------------------------------------------------
# Action-tag fallback backfill (I/O + reporting)
# ---------------------------------------------------------------------------
# Action tags are normally hand-maintained in action_tags.jsonl, and
# load_motion_metadata() / load_action_tags() hard-exit when any clip on disk is
# missing an entry. When clips are added incrementally, hand-labeling lags behind,
# so we backfill missing entries with a best-effort tag inferred from the clip
# name (the inference itself lives in motion_labels.infer_action_tags_from_clip_name).
# These guesses are HEURISTIC and must be reviewed by hand — the run reports every
# fallback in yellow.

_COLOR_RESET = "\033[0m"
_COLOR_YELLOW = "\033[93m"


def _ensure_action_tags_fallback(
    dataset_dir_path: Path,
    motion_files: list[Path],
) -> list[tuple[str, list[str]]]:
    """Backfill clips absent from action_tags.jsonl with clip-name-inferred tags.

    Returns the list of (clip, inferred_tags) that were appended (empty if every
    clip already had an entry). Only clips with *no* entry are backfilled — an
    explicitly empty action_tags list is left untouched (load_action_tags treats
    absence, not emptiness, as the fatal case). The file is created if missing.
    """
    tags_path = dataset_dir_path / ACTION_TAGS_FILE
    existing_clips: set[str] = set()
    if tags_path.exists():
        existing_clips = set(load_action_tags(dataset_dir_path).keys())

    fallbacks = [
        (motion_path.name, infer_action_tags_from_clip_name(motion_path.name))
        for motion_path in motion_files
        if motion_path.name not in existing_clips
    ]
    if not fallbacks:
        return []

    with open(tags_path, "a", encoding="utf-8") as handle:
        for clip, inferred in sorted(fallbacks):
            handle.write(json.dumps({"clip": clip, "action_tags": inferred}) + "\n")
    print(
        f"[OK] backfilled {len(fallbacks)} missing action_tags entr"
        f"{'y' if len(fallbacks) == 1 else 'ies'} into {ACTION_TAGS_FILE}"
    )
    return fallbacks


def _print_action_tag_fallback_report(fallbacks: list[tuple[str, list[str]]]) -> None:
    """Print every fallback-tagged clip in yellow as a manual-review reminder."""
    print(
        f"\n{_COLOR_YELLOW}{'=' * 70}\n"
        f"[REVIEW] {len(fallbacks)} clip(s) had no hand-labeled action_tags; tags below "
        f"were auto-inferred from the clip name and written to {ACTION_TAGS_FILE}.\n"
        f"Please verify them by hand (especially any 'unknown'):{_COLOR_RESET}"
    )
    for clip, inferred in sorted(fallbacks):
        print(f"  {_COLOR_YELLOW}{clip:<48s} -> {inferred}{_COLOR_RESET}")
    print(f"{_COLOR_YELLOW}{'=' * 70}{_COLOR_RESET}")


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
    object_types: tuple[str, ...],
) -> str:
    resolved = infer_object_type_from_filename(
        motion_name,
        valid_types=set(object_types),
    )
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


def _compute_global_canonical_stats(
    rebuilt_cond: dict[str, dict],
    motion_files: list[Path],
) -> None:
    """Compute the GLOBAL per-channel standardization statistics over every motion
    clip and store the same 13-vectors on every cond entry.

    Each physical clip is encoded into the L-normalized space (rest-centered
    position + per-skeleton size division) and pooled across all joints, frames,
    clips, and species. The resulting mean/std are a single cross-species
    constant (no per-species motion prior), so they generalize to held-out
    species while restoring the zero-mean / unit-variance behavior the diffusion
    noise schedule expects. Requires rest geometry (set by mark_canonical_cond_entry)
    to already be present on each cond entry."""

    known_object_types = tuple(rebuilt_cond.keys())
    acc = None
    used = 0
    for motion_path in motion_files:
        object_type = _infer_object_type_from_motion_name(motion_path.name, known_object_types)
        object_cond = rebuilt_cond.get(object_type)
        if object_cond is None:
            continue
        motion = np.load(motion_path).astype(np.float32, copy=False)
        if motion.ndim != 3 or motion.shape[-1] < 13:
            continue
        try:
            acc = accumulate_lnorm_stats(motion, object_cond, acc=acc)
        except (KeyError, ValueError):
            # cond entry lacks rest geometry (e.g. minimal synthetic fixtures);
            # such clips cannot be encoded, so skip them.
            continue
        used += 1

    if acc is None or acc["count"] <= 0:
        print("[WARN] no usable motion clips with rest geometry; global canonical stats not written")
        return

    mean, std = finalize_lnorm_stats(acc)
    for object_cond in rebuilt_cond.values():
        set_canonical_global_stats(object_cond, mean, std)
    with np.printoptions(precision=3, suppress=True, linewidth=160):
        print(
            f"[OK] global canonical stats over {used} clip(s) / {acc['count']} frames-joints\n"
            f"     mean={mean}\n     std ={std}"
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
            joint_names, parents, offsets
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
        object_type = str(entry.get("object_type", ""))
        if object_type not in rebuilt_cond:
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
    # translation_root_index, and other per-clip fields.  (action_tags are sourced
    # from action_tags.jsonl at load time and stripped on write; any clip missing
    # an entry there is backfilled by _ensure_action_tags_fallback below before
    # load_motion_metadata runs, so the load no longer hard-exits on new clips.)
    metadata_path = dataset_dir_path / MOTION_METADATA_FILE
    if not metadata_path.exists():
        raise RuntimeError(
            f"{MOTION_METADATA_FILE} not found at {metadata_path}.\n"
            f"This script requires an existing motion_metadata.json to preserve "
            f"is_loop, source_file, translation_root_index, and other per-clip metadata.\n"
            f"If you've deleted it, re-run preprocess_and_validate.py to regenerate "
            f"the full dataset, or restore it from a backup."
        )

    # Backfill any clips missing from action_tags.jsonl BEFORE load_motion_metadata,
    # which otherwise hard-exits when a clip on disk has no action_tags entry.
    action_tag_fallbacks = _ensure_action_tags_fallback(dataset_dir_path, motion_files)

    existing_cond = dict(np.load(cond_path, allow_pickle=True).item())
    existing_motion_metadata = load_motion_metadata(dataset_dir_path)
    known_object_types = tuple(existing_cond.keys())
    active_object_types = sorted(
        {
            _infer_object_type_from_motion_name(
                motion_path.name,
                known_object_types,
            )
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
    _compute_global_canonical_stats(rebuilt_cond, motion_files)
    print(f"[OK] global canonical stats computed in {time.time() - t0:.1f}s")

    np.save(str(cond_path), rebuilt_cond)

    rebuilt_motion_metadata: dict[str, dict[str, object]] = {}
    object_counts: Counter[str] = Counter()
    total_frames = 0
    max_joints = 0
    for motion_path in motion_files:
        motion = np.load(motion_path, mmap_mode="r")
        total_frames += int(motion.shape[0])
        max_joints = max(max_joints, int(motion.shape[1]))

        motion_entry = dict(existing_motion_metadata.get(motion_path.name, {}))
        object_type = _infer_object_type_from_motion_name(
            motion_path.name, tuple(rebuilt_cond.keys())
        )
        motion_entry.update(build_motion_labels(object_type, motion_name=motion_path.name))
        motion_entry["translation_root_index"] = int(
            canonical_translation_roots[str(motion_entry["object_type"])]
        )
        rebuilt_motion_metadata[motion_path.name] = motion_entry
        object_counts[str(motion_entry["object_type"])] += 1

    _write_metadata_summary(
        dataset_dir_path,
        object_counts,
        max_joints,
        len(motion_files),
        total_frames,
    )
    write_motion_metadata(dataset_dir_path, rebuilt_motion_metadata, len(motion_files))
    _rewrite_positions_error_file(dataset_dir_path, rebuilt_motion_metadata)

    if action_tag_fallbacks:
        _print_action_tag_fallback_report(action_tag_fallbacks)

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
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("Regenerating dataset sidecar artifacts")
    print("=" * 70 + "\n")

    try:
        dataset_dir_path = regenerate_dataset_artifacts(
            args.dataset_dir,
            t5_model=args.t5_model,
        )
        cond_path = dataset_dir_path / "cond.npy"
        cond = dict(np.load(cond_path, allow_pickle=True).item())

        print(f"Regenerated dataset artifacts under {dataset_dir_path}")
        print(f"Saved cond.npy with {len(cond)} objects: {', '.join(sorted(cond.keys()))}")        
        print("[PASS] Dataset sidecar regeneration completed successfully")
        return 0
    except Exception as e:
        print(f"ERROR: Failed to regenerate dataset artifacts: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
