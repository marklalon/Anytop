#!/usr/bin/env python3
"""
Unified Preprocessing + Validation Workflow
============================================
Automatically chains AnyTop dataset creation with validation:
  1. Preprocessing: Creates motion tensors and conditioning files
  2. Validation: Validates the preprocessed dataset

Usage:
    python preprocess_and_validate.py [--validate-only] [--skip-validate] [--objects-subset all|...]

Examples:
    # Full workflow (preprocess + validate)
    python preprocess_and_validate.py

    # Validate only (skip preprocessing)
    python preprocess_and_validate.py --validate-only

    # Skip validation (for faster checks or CI)
    python preprocess_and_validate.py --skip-validate
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

ANYTOP_DIR = Path(__file__).resolve().parent


def run_preprocessing(objects_subset: str, num_workers: int | None) -> int:
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
    ]
    if num_workers is not None:
        cmd.extend(["--num-workers", str(num_workers)])
    
    result = subprocess.run(cmd, cwd=str(ANYTOP_DIR), capture_output=False)
    return result.returncode


def run_validation(objects_subset: str, skip_loader: bool) -> int:
    """Run dataset validation."""
    print("\n" + "=" * 70)
    print("STEP 2: VALIDATION - Checking preprocessed dataset")
    print("=" * 70 + "\n")
    
    cmd = [
        sys.executable,
        str(ANYTOP_DIR / "tests" / "check_anytop_dataset.py"),
        "--objects-subset", objects_subset,
    ]
    if skip_loader:
        cmd.append("--skip-validate")
    
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
        "--num-workers",
        default=None,
        type=int,
        help="Number of worker threads used to process BVH files within each object. Defaults to min(8, cpu_count).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    # Preprocess if not validate-only
    if not args.validate_only:
        ret = run_preprocessing(args.objects_subset, args.num_workers)
        if ret != 0:
            print("\n[FAIL] Preprocessing failed. Aborting validation.")
            return ret
    
    # Validate
    ret = run_validation(args.objects_subset, args.skip_validate)
    
    if ret == 0:
        print("\n" + "=" * 70)
        print("✓ WORKFLOW COMPLETE: Preprocess + Validate succeeded")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("✗ WORKFLOW FAILED at validation step")
        print("=" * 70)
    
    return ret


if __name__ == "__main__":
    sys.exit(main())
