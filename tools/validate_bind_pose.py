#!/usr/bin/env python3
"""Validate that source FBX/GLB bind poses match cond.npy for a dataset.

Loads each source motion FBX/GLB (deduplicated by path) and compares its
skeleton structure (joint count, parent topology, scaled bone lengths)
against the reference stored in ``cond.npy``.  A mismatch indicates the
motion was animated on a different skeleton than the T-pose used to build
the conditioning.

Usage::

    python tools/validate_bind_pose.py
    python tools/validate_bind_pose.py --dataset-dir /path/to/dataset
    python tools/validate_bind_pose.py --sample-count 50
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Add Anytop root to path for absolute imports
_anytop_root = Path(__file__).resolve().parent.parent
if str(_anytop_root) not in sys.path:
    sys.path.insert(0, str(_anytop_root))
if str(_anytop_root.parent) not in sys.path:
    sys.path.insert(0, str(_anytop_root.parent))

from data_loaders.truebones.truebones_utils.param_utils import (  # noqa: E402
    MOTION_DIR,
    MOTION_METADATA_FILE,
    get_dataset_dir,
)
from data_loaders.truebones.truebones_utils.motion_labels import (  # noqa: E402
    load_motion_metadata,
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _print_warn(message: str) -> None:
    print(f"\033[33m[WARN] {message}\033[0m")


def _print_ok(message: str) -> None:
    print(f"[OK] {message}")


def _select_files(files: list[Path], sample_limit: int) -> list[Path]:
    if sample_limit <= 0:
        return files
    return files[: min(sample_limit, len(files))]


# ── multiprocessing worker (module-level for pickling) ────────────────────────


def _check_one_source(cond_subset: dict, source_path: str) -> str | None:
    """Load one FBX/GLB and compare its bind pose against *cond_subset*.

    Returns a warning string on mismatch, or *None* on pass.
    ``cond_subset`` must contain ``parents``, ``offsets``,
    ``original_joint_count``, and ``scale_factor``.
    """
    from motion_lib import FBX

    source_path_obj = Path(source_path)
    if not source_path_obj.exists():
        return (
            f"source bind pose check skipped — file not found: {source_path}"
        )

    try:
        source_anim, source_names, _frametime = FBX.load(str(source_path_obj))
    except Exception as exc:
        return (
            f"failed to load source FBX for bind pose check: "
            f"{source_path}: {exc}"
        )

    raw_offsets = np.asarray(source_anim.offsets, dtype=np.float64)
    raw_parents = np.asarray(source_anim.parents, dtype=np.int64)
    raw_joint_count = len(raw_parents)

    cond_parents = np.asarray(cond_subset["parents"], dtype=np.int64)
    cond_offsets = np.asarray(cond_subset["offsets"], dtype=np.float64)
    original_joint_count = int(
        cond_subset.get("original_joint_count", raw_joint_count)
    )
    scale_factor = float(cond_subset.get("scale_factor", 1.0))

    # --- 1. Compare joint count (vs original without helper joints) ---
    if raw_joint_count != original_joint_count:
        return (
            f"{source_path_obj.name} — joint count {raw_joint_count} differs "
            f"from cond.npy reference ({original_joint_count})"
        )

    # --- 2. Compare parent topology (child count distribution) ---
    source_child_counts = sorted(
        len([c for c, p in enumerate(raw_parents) if p == j])
        for j in range(raw_joint_count)
    )
    cond_child_counts = sorted(
        len([c for c, p in enumerate(cond_parents[:original_joint_count]) if p == j])
        for j in range(original_joint_count)
    )
    if source_child_counts != cond_child_counts:
        return (
            f"{source_path_obj.name} — parent hierarchy differs from "
            f"cond.npy reference"
        )

    # --- 3. Compare bone lengths (sorted for ordering-invariance) ---
    raw_lengths = np.sort(np.linalg.norm(raw_offsets, axis=1)[1:])  # exclude root
    cond_lengths = np.sort(
        np.linalg.norm(cond_offsets[:original_joint_count], axis=1)[1:]
    )

    if len(raw_lengths) != len(cond_lengths):
        return (
            f"{source_path_obj.name} — non-root joint count mismatch after "
            f"excluding helpers"
        )

    scaled_raw = raw_lengths * scale_factor
    ratio = np.where(
        cond_lengths > 1e-8,
        scaled_raw / cond_lengths,
        np.ones_like(scaled_raw),
    )
    max_deviation = float(np.max(np.abs(ratio - 1.0)))

    if max_deviation > 0.05:  # 5% tolerance
        max_idx = int(np.argmax(np.abs(ratio - 1.0)))
        return (
            f"{source_path_obj.name} — bone length deviation "
            f"{max_deviation * 100:.2f}% at sorted position {max_idx} "
            f"(cond={cond_lengths[max_idx]:.4f}, "
            f"source*scale={scaled_raw[max_idx]:.4f})"
        )

    return None


# ── core validation ───────────────────────────────────────────────────────────


def _collect_unique_sources(
    motion_files: list[Path],
    motion_metadata_lookup: dict,
    cond: dict,
    sample_limit: int,
) -> list[tuple[str, str, dict]]:
    """Collect deduplicated (object_type, source_path, cond_subset) tuples."""
    checked_keys: set[tuple[str, str]] = set()
    tasks: list[tuple[str, str, dict]] = []

    files_to_check = _select_files(motion_files, sample_limit)
    for motion_path in files_to_check:
        meta = motion_metadata_lookup.get(motion_path.name)
        if not isinstance(meta, dict):
            continue
        object_type = meta.get("object_type")
        source_path = meta.get("source_fbx_path")
        if not object_type or not source_path:
            continue
        key = (str(object_type), str(source_path))
        if key in checked_keys:
            continue
        checked_keys.add(key)

        object_cond = cond.get(object_type)
        if object_cond is None:
            continue
        # Pass only the fields the worker actually needs (avoids pickling
        # large arrays like joints_names_embs, kinematic_chains, etc.)
        slim_cond = {
            "parents": object_cond["parents"],
            "offsets": object_cond["offsets"],
            "original_joint_count": object_cond.get("original_joint_count"),
            "scale_factor": object_cond.get("scale_factor", 1.0),
            "orientation_reference_fbx_path": object_cond.get(
                "orientation_reference_fbx_path", ""
            ),
        }
        tasks.append((str(object_type), str(source_path), slim_cond))

    return tasks


def validate_source_bind_pose(
    motions_dir: Path,
    cond: dict,
    motion_files: list[Path],
    motion_metadata_lookup: dict,
    sample_limit: int,
    bind_pose_workers: int = 16,
) -> tuple[int, int]:
    """Check that each source FBX/GLB motion file's bind pose matches cond.npy.

    Loads the source FBX/GLB for each motion (deduplicated by source path) and
    compares its skeleton structure (joint count, bone lengths scaled by
    ``scale_factor``, parent topology) against ``cond.npy`` for that object
    type.  A mismatch indicates the motion was animated on a different skeleton
    than the reference T-pose used to build the condition.

    Args:
        motions_dir: Dataset motions directory.
        cond: Loaded ``cond.npy`` dict.
        motion_files: List of ``.npy`` motion file paths.
        motion_metadata_lookup: Per-motion metadata dict.
        sample_limit: Only check first N files (0 = all).
        bind_pose_workers: Parallel workers (1 = sequential, >1 = multiprocessing).

    Returns
    -------
    (warn_count, total_sources) : tuple[int, int]
        Number of mismatched source files, and total unique source files checked.
    """
    tasks = _collect_unique_sources(
        motion_files, motion_metadata_lookup, cond, sample_limit,
    )
    if not tasks:
        return 0, 0

    total = len(tasks)
    results: list[str | None] = [None] * total

    def _report_progress(done: int, source_name: str, status: str) -> None:
        print(
            f"\r  [{done}/{total}] {source_name} ... {status}",
            end="", flush=True,
        )

    if bind_pose_workers <= 1:
        for i, (object_type, source_path, cond_subset) in enumerate(tasks):
            source_name = Path(source_path).name
            results[i] = _check_one_source(cond_subset, source_path)
            _report_progress(
                i + 1, source_name,
                "OK   " if results[i] is None else "MISMATCH",
            )
    else:
        with ProcessPoolExecutor(max_workers=bind_pose_workers) as executor:
            future_to_idx = {
                executor.submit(
                    _check_one_source, cond_subset, source_path
                ): (idx, object_type, source_path)
                for idx, (object_type, source_path, cond_subset)
                in enumerate(tasks)
            }
            completed = 0
            for future in as_completed(future_to_idx):
                idx, object_type, source_path = future_to_idx[future]
                completed += 1
                source_name = Path(source_path).name
                try:
                    results[idx] = future.result()
                    status = "OK   " if results[idx] is None else "MISMATCH"
                except Exception as exc:
                    results[idx] = f"worker exception: {exc}"
                    status = "ERROR "
                _report_progress(completed, source_name, status)

    print()  # final newline after progress line

    warn_count = 0
    for idx, (object_type, source_path, cond_subset) in enumerate(tasks):
        warn_msg = results[idx]
        if warn_msg is not None:
            source_name = Path(source_path).name
            _print_warn(
                f"source bind pose mismatch: {source_name} "
                f"({object_type}) — {warn_msg}"
            )
            warn_count += 1

    # T-pose filename check — per species (deduplicated), only for mismatched
    tpose_warn_count = 0
    checked_object_types: set[str] = set()
    for idx, (object_type, source_path, cond_subset) in enumerate(tasks):
        warn_msg = results[idx]
        if warn_msg is None:
            continue
        if object_type in checked_object_types:
            continue
        checked_object_types.add(object_type)
        tpose_path = cond_subset.get("orientation_reference_fbx_path", "")
        if tpose_path:
            tpose_name = Path(tpose_path).name
            name_upper = tpose_name.upper()
            if "TPOSE" not in name_upper and "TPOS" not in name_upper:
                _print_warn(
                    f"{tpose_name} ({object_type}) — T-pose source "
                    f"filename does not contain 'TPOSE'/'TPOS', may not "
                    f"be a proper T-pose"
                )
                tpose_warn_count += 1

    if len(checked_object_types) > 0 and tpose_warn_count == 0:
        _print_ok(
            "T-pose filename check passed for all mismatched species"
        )
    elif tpose_warn_count > 0:
        _print_warn(
            f"{tpose_warn_count} mismatched species T-pose source filename(s) "
            f"do not contain 'TPOSE'/'TPOS'"
        )

    return warn_count, total


# ── standalone entry point ────────────────────────────────────────────────────


def resolve_dataset_dir(raw_value: str | None) -> Path:
    if raw_value:
        path = Path(raw_value)
    else:
        path = Path(get_dataset_dir(None))
    if not path.is_absolute():
        path = _anytop_root / path
    return path.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate source FBX/GLB bind poses against cond.npy.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Dataset directory.  Uses default path when omitted.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=0,
        help="Only check the first N motion files (0 = all, default=0).",
    )
    parser.add_argument(
        "--bind-pose-workers",
        type=int,
        default=16,
        help="Parallel workers for loading source FBX/GLB files "
             "(default=16, 1=sequential).  Each worker uses its own Blender "
             "process, so higher values are safe.",
    )
    args = parser.parse_args()

    dataset_dir = resolve_dataset_dir(args.dataset_dir)
    motions_dir = dataset_dir / MOTION_DIR
    cond_path = dataset_dir / "cond.npy"

    # Validate prerequisites
    if not motions_dir.exists():
        _print_warn(f"motions directory not found: {motions_dir}")
        return 1
    if not cond_path.exists():
        _print_warn(f"cond.npy not found: {cond_path}")
        return 1

    # Load cond
    _print_ok(f"loading cond.npy from {cond_path}")
    cond = np.load(cond_path, allow_pickle=True).item()

    # Load motion metadata
    try:
        motion_metadata_lookup = load_motion_metadata(dataset_dir)
    except Exception as exc:
        _print_warn(f"failed to load {MOTION_METADATA_FILE}: {exc}")
        return 1

    motion_files = sorted(motions_dir.glob("*.npy"))
    if not motion_files:
        _print_warn("motion directory is empty")
        return 1

    sample_label = (
        str(args.sample_count) if args.sample_count > 0 else "all"
    )
    _print_ok(
        f"checking {len(motion_files)} motion file(s) ({sample_label}) "
        f"with {args.bind_pose_workers} worker(s)"
    )

    warn_count, total_sources = validate_source_bind_pose(
        motions_dir,
        cond,
        motion_files,
        motion_metadata_lookup,
        args.sample_count,
        bind_pose_workers=args.bind_pose_workers,
    )

    passed = total_sources - warn_count

    print()
    _print_ok(
        f"bind pose validation: {passed}/{total_sources} source skeleton(s) "
        f"match cond.npy"
    )
    if warn_count:
        _print_warn(f"{warn_count} source skeleton(s) had mismatches")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
