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
    --skip-corrupted-export              Skip corrupted-reference export
    --objects-subset SUBSET              Object subset to process (default: all)
    --object-workers N                   Concurrent characters to preprocess (default: 8)
    --file-workers N                     Worker threads per character for BVH processing (default: 8)
    --corrupted-seed SEED                Random seed for corrupted-reference export (default: 1234)
    --corrupted-sample-limit N           Limit corrupted-reference samples (0=all, default: 0)

Examples:
    # Full workflow: preprocess → corrupted export → validate
    python preprocess_and_validate.py

    # Validate only (assumes preprocessing already done)
    python preprocess_and_validate.py --validate-only

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
from pathlib import Path

ANYTOP_DIR = Path(__file__).resolve().parent


def run_preprocessing(objects_subset: str, object_workers: int, file_workers: int) -> int:
    """Run the AnyTop dataset preprocessing."""
    print("\n" + "=" * 70)
    print("STEP 1: PREPROCESSING - Creating AnyTop dataset")
    print("=" * 70 + "\n")
    
    # Set default raw data directory if not already set
    if "ANYTOP_RAW_DATA_DIR" not in os.environ:
        os.environ["ANYTOP_RAW_DATA_DIR"] = r"E:\Dataset\Truebone_Z-OO"
    
    cmd = [
        sys.executable, "-m", "utils.create_dataset",
        "--objects-subset", objects_subset,
        "--object-workers", str(object_workers),
        "--file-workers", str(file_workers),
    ]
    
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


def run_validation(objects_subset: str) -> int:
    """Run dataset validation."""
    print("\n" + "=" * 70)
    print("STEP 3: VALIDATION - Checking preprocessed dataset")
    print("=" * 70 + "\n")
    
    cmd = [
        sys.executable,
        str(ANYTOP_DIR / "tests" / "check_anytop_dataset.py"),
        "--objects-subset", objects_subset,
    ]
    
    result = subprocess.run(cmd, cwd=str(ANYTOP_DIR), capture_output=False)
    return result.returncode


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
        "--objects-subset",
        default="all",
        help="Expected object subset for validation (default: all).",
    )
    parser.add_argument(
        "--object-workers",
        default=6,
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    steps_completed = []
    
    # Preprocess if not validate-only
    if not args.validate_only:
        ret = run_preprocessing(
            args.objects_subset,
            args.object_workers,
            args.file_workers,
        )
        if ret != 0:
            print("\n[FAIL] Preprocessing failed. Aborting workflow.")
            return ret
        steps_completed.append("Preprocess")
        
        if not args.skip_corrupted_export:
            ret = run_corrupted_export(
                args.objects_subset,
                args.corrupted_seed,
                args.corrupted_sample_limit,
            )
            if ret != 0:
                print("\n[FAIL] Corrupted-reference export failed. Aborting workflow.")
                return ret
            steps_completed.append("Corrupted Export")
    
    # Validate
    if not args.skip_validate:
        ret = run_validation(args.objects_subset)
        if ret != 0:
            print("\n" + "=" * 70)
            print("✗ WORKFLOW FAILED during validation step")
            print("=" * 70)
            return ret
        steps_completed.append("Validate")
    
    # Success
    print("\n" + "=" * 70)
    workflow_desc = " → ".join(steps_completed) if steps_completed else "No steps executed"
    print(f"✓ WORKFLOW COMPLETE: {workflow_desc} succeeded")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
