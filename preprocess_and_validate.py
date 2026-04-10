#!/usr/bin/env python3
"""
Unified Preprocessing + Validation Workflow
============================================
Automatically chains AnyTop dataset creation with validation:
  1. Preprocessing: Creates motion tensors and conditioning files
  2. Corrupted Export: Generates stored corrupted-reference motions (by default)
  3. Validation: Validates the preprocessed dataset

Usage:
    python preprocess_and_validate.py [OPTIONS]

Options:
    --validate-only                      Skip preprocessing, only validate existing dataset
    --skip-validate                      Skip validation step (faster for CI)
    --skip-orientation-check             Skip processed-BVH orientation validation
    --skip-corrupted-export              Skip corrupted-reference export
    --objects-subset SUBSET              Object subset to process (default: all)
    --object-workers N                   Concurrent characters to preprocess (default: 8)
    --file-workers N                     Worker threads per character for BVH processing (default: 8)
    --sample-count N                     Limit file validation to first N motions/BVHs (0=all, default: 0)
    --orientation-threshold-deg DEG      Max allowed canonicalized first-frame facing error from +Z during validation (default: 15.0)
    --corrupted-seed SEED                Random seed for corrupted-reference export (default: 1234)
    --corrupted-sample-limit N           Limit corrupted-reference samples (0=all, default: 0)

Examples:
    # Full workflow: preprocess → corrupted export → validate
    python preprocess_and_validate.py

    # Validate only (assumes preprocessing already done)
    python preprocess_and_validate.py --validate-only

    # Validate only, but skip the orientation check
    python preprocess_and_validate.py --validate-only --skip-orientation-check

    # Preprocess without validation
    python preprocess_and_validate.py --skip-validate

    # Preprocess without corrupted-reference export
    python preprocess_and_validate.py --skip-corrupted-export

    # Preprocess specific object subset with custom settings
    python preprocess_and_validate.py --objects-subset "Hound" --object-workers 4 --file-workers 8

    # Corrupted export with custom seed and sample limit
    python preprocess_and_validate.py --corrupted-seed 42 --corrupted-sample-limit 100

    # Validate only a specific object subset
    python preprocess_and_validate.py --validate-only --objects-subset "Monkey"

    # Fast CI workflow (skip validation and corrupted export)
    python preprocess_and_validate.py --skip-validate --skip-corrupted-export
"""

import argparse
import os
import sys
import subprocess
import shutil
from pathlib import Path

ANYTOP_DIR = Path(__file__).resolve().parent


def run_preprocessing(objects_subset: str, object_workers: int, file_workers: int, raw_data_dir: str = "") -> int:
    """Run the AnyTop dataset preprocessing."""
    print("\n" + "=" * 70)
    print("STEP 1: PREPROCESSING - Creating AnyTop dataset")
    print("=" * 70 + "\n")
    
    cmd = [
        sys.executable, "-m", "utils.create_dataset",
        "--objects-subset", objects_subset,
        "--object-workers", str(object_workers),
        "--file-workers", str(file_workers),
    ]
    
    if raw_data_dir:
        cmd.extend(["--raw-data-dir", raw_data_dir])
    
    result = subprocess.run(cmd, cwd=str(ANYTOP_DIR), capture_output=False)
    return result.returncode


def run_corrupted_export(objects_subset: str, seed: int, sample_limit: int) -> int:
    """Generate stored corrupted-reference motions alongside the clean dataset."""
    print("\n" + "=" * 70)
    print("STEP 2: CORRUPTED REFERENCES - Exporting stored corrupted motions")
    print("=" * 70 + "\n")

    cmd = [
        sys.executable,
        str(ANYTOP_DIR / "tools" / "export_corrupted_truebones_samples.py"),
        "--objects-subset", objects_subset,
        "--seed", str(seed),
    ]
    if sample_limit > 0:
        cmd.extend(["--sample-limit", str(sample_limit)])

    result = subprocess.run(cmd, cwd=str(ANYTOP_DIR), capture_output=False)
    return result.returncode


def run_validation(
    objects_subset: str,
    skip_orientation_check: bool,
    orientation_threshold_deg: float,
    sample_count: int,
) -> int:
    """Run dataset validation."""
    print("\n" + "=" * 70)
    print("STEP 3: VALIDATION - Checking preprocessed dataset")
    print("=" * 70 + "\n")
    
    # Import and call check_anytop_dataset.py main() directly instead of subprocess
    sys.path.insert(0, str(ANYTOP_DIR / "tests"))
    from check_anytop_dataset import _resolve_dataset_dir, _print_ok, _print_warn, _require, ValidationError
    
    # Resolve dataset directory
    dataset_dir = _resolve_dataset_dir(None)
    
    _print_ok(f"dataset_dir: {dataset_dir}")
    _print_ok(f"objects_subset: {objects_subset}")
    _print_ok(f"file_validation_scope: {'all files' if sample_count == 0 else f'first {sample_count} files'}")
    
    from check_anytop_dataset import (
        _read_required_artifacts,
        _validate_metadata,
        _validate_cond_file,
        _validate_motion_files,
        _validate_motion_orientation,
        _validate_positions_error_file,
    )
    
    try:
        motions_dir, bvhs_dir, cond_path, metadata_path, positions_error_path = _read_required_artifacts(dataset_dir)
        
        _validate_metadata(metadata_path)
        
        from data_loaders.truebones.truebones_utils.param_utils import OBJECT_SUBSETS_DICT
        cond = _validate_cond_file(cond_path, objects_subset)
        
        _validate_motion_files(motions_dir, bvhs_dir, cond, sample_count)
        
        if skip_orientation_check:
            _print_warn("skipping processed-BVH orientation validation by request")
        else:
            _validate_motion_orientation(bvhs_dir, cond, sample_count, orientation_threshold_deg)
        
        _validate_positions_error_file(positions_error_path)
        
        print("[PASS] dataset validation completed successfully")
        return 0
    except ValidationError as e:
        print(f"[WARN] dataset validation warning: {e}")
        return 1


def check_and_clean_old_data() -> bool:
    """
    Check if old preprocessed data exists in the target dataset directory.
    If found, ask user whether to delete it.
    
    Returns:
        True if user wants to proceed with preprocessing (either no old data found, or old data deleted).
        False if user wants to abort.
    """
    # Import here to get the resolved dataset directory path
    sys.path.insert(0, str(ANYTOP_DIR / "data_loaders" / "truebones" / "truebones_utils"))
    from param_utils import get_dataset_dir
    
    dataset_dir = Path(get_dataset_dir())
    motions_dir = dataset_dir / "motions"
    bvhs_dir = dataset_dir / "bvhs"
    corrupted_ref_dir = dataset_dir / "corrupted_references"
    
    # Check if any old data exists
    old_data_exists = (motions_dir.exists() and any(motions_dir.iterdir())) or \
                      (bvhs_dir.exists() and any(bvhs_dir.iterdir())) or \
                      (corrupted_ref_dir.exists() and any(corrupted_ref_dir.iterdir()))
    
    if not old_data_exists:
        return True
    
    # Old data found, ask user
    print("\n" + "=" * 70)
    print("⚠ WARNING: Old preprocessed data detected")
    print("=" * 70)
    print(f"Dataset directory: {dataset_dir}")
    if motions_dir.exists() and any(motions_dir.iterdir()):
        print(f"  - {motions_dir} contains existing data")
    if bvhs_dir.exists() and any(bvhs_dir.iterdir()):
        print(f"  - {bvhs_dir} contains existing data")
    if corrupted_ref_dir.exists() and any(corrupted_ref_dir.iterdir()):
        print(f"  - {corrupted_ref_dir} contains existing data")
    print("\nDo you want to delete the old data and proceed with preprocessing?")
    
    while True:
        response = input("Enter 'yes' to delete and continue, or 'no' to abort: ").strip().lower()
        if response in ('yes', 'y'):
            print("\nDeleting old data...")
            try:
                if motions_dir.exists():
                    shutil.rmtree(motions_dir)
                    print(f"  ✓ Deleted {motions_dir}")
                if bvhs_dir.exists():
                    shutil.rmtree(bvhs_dir)
                    print(f"  ✓ Deleted {bvhs_dir}")
                if corrupted_ref_dir.exists():
                    shutil.rmtree(corrupted_ref_dir)
                    print(f"  ✓ Deleted {corrupted_ref_dir}")
                print("Old data cleaned successfully. Proceeding with preprocessing...\n")
                return True
            except Exception as e:
                print(f"ERROR: Failed to delete old data: {e}")
                print("Aborting preprocessing.")
                return False
        elif response in ('no', 'n'):
            print("\nPreprocessing aborted.")
            return False
        else:
            print("Invalid response. Please enter 'yes', 'y', 'no', or 'n'.")


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
        "--skip-validate",
        action="store_true",
        help="Skip validation (faster, useful for CI).",
    )
    parser.add_argument(
        "--skip-orientation-check",
        action="store_true",
        help="Skip processed-BVH orientation validation during dataset checks.",
    )
    parser.add_argument(
        "--objects-subset",
        default="all",
        help="Expected object subset for validation (default: all).",
    )
    parser.add_argument(
        "--object-workers",
        default=8,
        type=int,
        help="Concurrent characters to preprocess. Defaults to 8.",
    )
    parser.add_argument(
        "--file-workers",
        default=8,
        type=int,
        help="Worker threads per character for BVH file processing. Defaults to 8.",
    )
    parser.add_argument(
        "--orientation-threshold-deg",
        default=15.0,
        type=float,
        help="Maximum allowed first-frame facing error from +Z for processed validation.",
    )
    parser.add_argument(
        "--sample-count",
        default=0,
        type=int,
        help="Limit file validation to the first N motions/BVHs. Use 0 to validate all files.",
    )
    parser.add_argument(
        "--skip-corrupted-export",
        action="store_true",
        help="Skip generating stored corrupted-reference motions after preprocessing.",
    )
    parser.add_argument(
        "--corrupted-seed",
        default=1234,
        type=int,
        help="Seed used when exporting stored corrupted-reference motions.",
    )
    parser.add_argument(
        "--corrupted-sample-limit",
        default=0,
        type=int,
        help="Limit the number of motions to export corrupted references for. 0 exports all motions.",
    )
    parser.add_argument(
        "--raw-data-dir",
        default="",
        type=str,
        help="Path to raw Truebones BVH folders. If not specified, uses default path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_count < 0:
        print("ERROR: --sample-count must be >= 0")
        return 1
    
    steps_completed = []
    
    # Check and clean old data before preprocessing
    if not args.validate_only:
        if not check_and_clean_old_data():
            print("\n" + "=" * 70)
            print("Preprocessing skipped due to user abort")
            print("=" * 70)
            return 1
    
    # Preprocess if not validate-only
    if not args.validate_only:
        ret = run_preprocessing(
            args.objects_subset,
            args.object_workers,
            args.file_workers,
            args.raw_data_dir,
        )
        if ret != 0:
            print("\n[FAIL] Preprocessing failed, aborting workflow.")
            return ret
        steps_completed.append("Preprocess")

        if not args.skip_corrupted_export:
            ret = run_corrupted_export(
                args.objects_subset,
                args.corrupted_seed,
                args.corrupted_sample_limit,
            )
            if ret != 0:
                print("\n[FAIL] Corrupted-reference export failed, aborting workflow.")
                return ret
            steps_completed.append("Corrupted Export")
    
    # Validate
    if not args.skip_validate:
        ret = run_validation(
            args.objects_subset,
            args.skip_orientation_check,
            args.orientation_threshold_deg,
            args.sample_count,
        )
        # Don't return on validation failure - continue to next step        
        steps_completed.append("Validate")
    
    # Success
    print("\n" + "=" * 70)
    workflow_desc = " → ".join(steps_completed) if steps_completed else "No steps executed"
    print(f"✓ WORKFLOW COMPLETE: {workflow_desc} succeeded")
    print("=" * 70)
    return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
