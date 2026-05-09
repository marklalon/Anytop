#!/usr/bin/env python3
"""
Unified Preprocessing + Validation Workflow
============================================
Automatically chains AnyTop dataset creation with validation:
    1. Preprocessing: Incrementally refreshes the selected objects-subset; `all` falls back to a full dataset refresh
    2. Validation: Validates the preprocessed dataset

Usage:
    python preprocess_and_validate.py [OPTIONS]

Options:
    --validate-only                      Skip preprocessing, only validate existing dataset
    --re-encode-joint-names-only         Skip preprocessing and validation, only re-encode joint names into cond.npy
    --skip-validate                      Skip validation step (faster for CI)
    --skip-orientation-check             Skip T-pose face-orientation validation during dataset checks
    --objects-subset SUBSET              Object subset to process incrementally; `all` uses full refresh (default: all)
    --object-workers N                   Concurrent characters to preprocess (default: 16)
    --sample-count N                     Limit file validation to first N motions (0=all, default: 0)
    --orientation-threshold-deg DEG      Maximum allowed T-pose face-orientation delta from the nearest cardinal XZ axis (+x/-x/+z/-z) before warning (default: 15.0)

Examples:
    # Full workflow: full refresh for objects-subset=all -> validate
    python preprocess_and_validate.py

    # Validate only (assumes preprocessing already done)
    python preprocess_and_validate.py --validate-only

    # Validate only, skipping orientation check
    python preprocess_and_validate.py --validate-only --skip-orientation-check

    # Preprocess without validation
    python preprocess_and_validate.py --skip-validate

    # Re-encode joint names only (fast, no motion re-export)
    python preprocess_and_validate.py --re-encode-joint-names-only

    # Incrementally refresh a specific object subset with custom settings
    python preprocess_and_validate.py --objects-subset "bipeds" --object-workers 4

    # Validate only a specific object subset
    python preprocess_and_validate.py --validate-only --objects-subset "flying"

    # Fast CI workflow (skip validation after subset/full refresh)
    python preprocess_and_validate.py --skip-validate
"""

import argparse
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

from param_utils import BVHS_DIR, MOTION_DIR, OBJECT_SUBSETS_DICT, get_dataset_dir  # noqa: E402
from truebones_utils.motion_labels import load_motion_metadata, write_motion_metadata  # noqa: E402


@dataclass
class PreservedSideArtifacts:
    cond: dict[str, dict[str, object]] = field(default_factory=dict)
    motion_metadata: dict[str, dict[str, object]] = field(default_factory=dict)


def _resolve_dataset_paths(dataset_dir: str = "") -> tuple[Path, Path, Path, Path | None, Path]:
    dataset_dir_path = Path(get_dataset_dir(dataset_dir or None))
    return (
        dataset_dir_path,
        dataset_dir_path / MOTION_DIR,
        dataset_dir_path / BVHS_DIR,
        dataset_dir_path / "glb" if BVHS_DIR != "glb" else None,
        dataset_dir_path / "joint_name_inspection",
    )


def _resolve_target_object_types(objects_subset: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(obj) for obj in OBJECT_SUBSETS_DICT[objects_subset]))


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
    for motion_name, entry in load_motion_metadata(dataset_dir_path).items():
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
    current_metadata = load_motion_metadata(dataset_dir_path)
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


def check_and_clean_old_data(objects_subset: str, dataset_dir: str = "") -> tuple[bool, PreservedSideArtifacts]:
    """
    Check for existing preprocessed data targeting the selected objects_subset.
    For incremental subsets, capture artifacts for non-target objects so they can be merged back.

    Returns (should_proceed, preserved_side_artifacts).
    """
    dataset_dir_path, motions_dir, bvhs_dir, legacy_glb_dir, joint_name_inspection_dir = _resolve_dataset_paths(dataset_dir)
    is_full_refresh = objects_subset.strip().lower() == "all"

    if is_full_refresh:
        paths_to_delete = _collect_nonempty_directories(motions_dir, bvhs_dir, legacy_glb_dir, joint_name_inspection_dir)
        preserved = PreservedSideArtifacts()
        title = "WARNING: Old preprocessed data detected"
        summary = [
            f"Dataset directory: {dataset_dir_path}",
            "objects-subset=all selected, using full dataset refresh",
            *[f"  - {p} contains existing data" for p in paths_to_delete],
        ]
    else:
        target_object_types = _resolve_target_object_types(objects_subset)
        preserved = _capture_preserved_side_artifacts(dataset_dir_path, target_object_types)
        targeted = [
            ("motion file(s)", motions_dir, _collect_targeted_files(motions_dir, target_object_types)),
            ("BVH file(s)", bvhs_dir, _collect_targeted_files(bvhs_dir, target_object_types)),
            ("legacy preview file(s)", legacy_glb_dir, _collect_targeted_files(legacy_glb_dir, target_object_types)),
            ("inspection file(s)", joint_name_inspection_dir, _collect_targeted_files(joint_name_inspection_dir, target_object_types)),
        ]
        paths_to_delete = [p for _, _, files in targeted for p in files]
        title = "WARNING: Existing preprocessed files detected for selected object types"
        summary = [
            f"Dataset directory: {dataset_dir_path}",
            f"Object types to refresh ({len(target_object_types)}): {', '.join(target_object_types)}",
            *[f"  - {dir_path}: {len(files)} matching {label}" for label, dir_path, files in targeted if files],
        ]

    if not paths_to_delete:
        return True, preserved

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    for line in summary:
        print(line)
    print("\nDo you want to delete the matching files and proceed with preprocessing?")

    if not _confirm_yes_no("Enter 'yes' to delete and continue, or 'no' to abort: "):
        print("\nPreprocessing aborted.")
        return False, preserved

    print("\nDeleting...")
    if not _delete_paths(paths_to_delete):
        print("Aborting preprocessing.")
        return False, preserved
    print("Done.\n")
    return True, preserved


def run_preprocessing(
    objects_subset: str,
    object_workers: int,
    raw_data_dir: str = "",
    dataset_dir: str = "",
) -> int:
    """Run the AnyTop dataset preprocessing."""
    print("\n" + "=" * 70)
    print("STEP 1: PREPROCESSING - Creating AnyTop dataset")
    print("=" * 70 + "\n")

    cmd = [
        sys.executable, "-m", "utils.create_dataset",
        "--objects-subset", objects_subset,
        "--object-workers", str(object_workers),
    ]

    if raw_data_dir:
        cmd.extend(["--raw-data-dir", raw_data_dir])
    if dataset_dir:
        cmd.extend(["--dataset-dir", dataset_dir])

    # Add parent of Anytop/ to PYTHONPATH so `from Anytop.utils...` imports work
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ANYTOP_DIR.parent) + os.pathsep + existing_pythonpath

    result = subprocess.run(cmd, cwd=str(ANYTOP_DIR), capture_output=False, env=env)
    return result.returncode


def run_re_encode_joint_names_only(
    dataset_dir: str = "",
    preserved_side_artifacts: PreservedSideArtifacts | None = None,
    t5_model: str = "t5-base",
) -> int:
    """Regenerate non-motion dataset artifacts without re-preprocessing motions."""

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
    objects_subset: str,
    skip_orientation_check: bool,
    orientation_threshold_deg: float,
    sample_count: int,
    dataset_dir: str = "",
) -> int:
    """Run dataset validation."""
    print("\n" + "=" * 70)
    print("STEP 2: VALIDATION - Checking preprocessed dataset")
    print("=" * 70 + "\n")

    # Ensure parent of Anytop/ is on sys.path so `from Anytop.utils...` imports work
    if str(ANYTOP_DIR.parent) not in sys.path:
        sys.path.insert(0, str(ANYTOP_DIR.parent))

    # Import and call validate_anytop_dataset.py main() directly instead of subprocess
    sys.path.insert(0, str(ANYTOP_DIR / "utils"))
    from validate_anytop_dataset import (
        _resolve_dataset_dir,
        _print_ok,
        _print_warn,
        _require,
        ValidationError,
    )
    from data_loaders.truebones.truebones_utils.motion_process import ROOT_XZ_STRIP_THRESHOLD

    # Resolve dataset directory
    dataset_dir = _resolve_dataset_dir(dataset_dir or None)

    _print_ok(f"dataset_dir: {dataset_dir}")
    _print_ok(f"objects_subset: {objects_subset}")
    _print_ok(f"file_validation_scope: {'all files' if sample_count == 0 else f'first {sample_count} files'}")

    from validate_anytop_dataset import (
        _prepare_dataset_for_validation,
        _read_required_artifacts,
        _validate_metadata,
        _validate_cond_file,
        _validate_motion_files,
        _validate_motion_metadata,
        _validate_positions_error_file,
    )

    try:
        _prepare_dataset_for_validation(
            dataset_dir,
            objects_subset,
            sample_count,
        )

        motions_dir, bvhs_dir, cond_path, metadata_path, positions_error_path = _read_required_artifacts(dataset_dir)
        cond = _validate_cond_file(cond_path, objects_subset)
        motion_files = sorted(motions_dir.glob("*.npy"))
        _validate_metadata(metadata_path, motion_files, cond)
        _validate_motion_metadata(dataset_dir, motion_files, cond)
        _validate_motion_files(motions_dir, bvhs_dir, cond, sample_count, ROOT_XZ_STRIP_THRESHOLD)

        if skip_orientation_check:
            _print_warn("skipping T-pose face-orientation validation by request")
        else:
            from validate_anytop_dataset import _validate_tpose_orientation
            _validate_tpose_orientation(cond, orientation_threshold_deg)

        _validate_positions_error_file(positions_error_path)

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
        "--objects-subset",
        default="all",
        choices=sorted(OBJECT_SUBSETS_DICT.keys()),
        help="Object subset to preprocess incrementally; `all` falls back to full refresh.",
    )
    parser.add_argument(
        "--object-workers",
        default=16,
        type=int,
        help="Concurrent characters to preprocess. Defaults to 16.",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_count < 0:
        print("ERROR: --sample-count must be >= 0")
        return 1
    if args.orientation_threshold_deg < 0:
        print("ERROR: --orientation-threshold-deg must be >= 0")
        return 1

    # Handle re-encode joint names only mode
    if args.re_encode_joint_names_only:
        return run_re_encode_joint_names_only(
            args.dataset_dir,
        )

    steps_completed = []
    preserved_side_artifacts = PreservedSideArtifacts()

    # Check and clean old data before preprocessing
    if not args.validate_only:
        should_proceed, preserved_side_artifacts = check_and_clean_old_data(args.objects_subset, args.dataset_dir)
        if not should_proceed:
            print("\n" + "=" * 70)
            print("Preprocessing skipped due to user abort")
            print("=" * 70)
            return 1

    # Preprocess if not validate-only
    if not args.validate_only:
        ret = run_preprocessing(
            args.objects_subset,
            args.object_workers,
            args.raw_data_dir,
            args.dataset_dir,
        )
        if ret != 0:
            print("\n[FAIL] Preprocessing failed, aborting workflow.")
            return ret

        ret = run_re_encode_joint_names_only(
            args.dataset_dir,
            preserved_side_artifacts=preserved_side_artifacts,
        )
        if ret != 0:
            print("\n[FAIL] Side artifact regeneration failed, aborting workflow.")
            return ret

        steps_completed.append("Preprocess")

    # Validate
    if not args.skip_validate:
        ret = run_validation(
            args.objects_subset,
            args.skip_orientation_check,
            args.orientation_threshold_deg,
            args.sample_count,
            args.dataset_dir,
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
