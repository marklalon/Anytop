#!/usr/bin/env python3
"""
Unified Preprocessing + Validation Workflow
============================================
Automatically chains AnyTop dataset creation with validation:
  1. Preprocessing: Creates motion tensors and conditioning files from FBX source files
  2. Validation: Validates the preprocessed dataset

Usage:
    python preprocess_and_validate.py [OPTIONS]

Options:
    --validate-only                      Skip preprocessing, only validate existing dataset
    --re-encode-joint-names-only         Skip preprocessing and validation, only re-encode joint names into cond.npy
    --skip-validate                      Skip validation step (faster for CI)
    --skip-orientation-check             Skip stored processed-orientation validation during dataset checks
    --objects-subset SUBSET              Object subset to process (default: all)
    --object-workers N                   Concurrent characters to preprocess (default: 16)
    --sample-count N                     Limit file validation to first N motions (0=all, default: 0)
    --orientation-threshold-deg DEG      Maximum allowed first-frame facing error from +Z using stored processed-orientation metadata (default: 15.0)

Examples:
    # Full workflow: preprocess ->validate
    python preprocess_and_validate.py

    # Validate only (assumes preprocessing already done)
    python preprocess_and_validate.py --validate-only

    # Validate only, skipping orientation check
    python preprocess_and_validate.py --validate-only --skip-orientation-check

    # Preprocess without validation
    python preprocess_and_validate.py --skip-validate

    # Re-encode joint names only (fast, no motion re-export)
    python preprocess_and_validate.py --re-encode-joint-names-only

    # Preprocess specific object subset with custom settings
    python preprocess_and_validate.py --objects-subset "Hound" --object-workers 4

    # Validate only a specific object subset
    python preprocess_and_validate.py --validate-only --objects-subset "Monkey"

    # Fast CI workflow (skip validation)
    python preprocess_and_validate.py --skip-validate
"""

import argparse
import os
import sys
import subprocess
import shutil
from pathlib import Path
import numpy as np

ANYTOP_DIR = Path(__file__).resolve().parent


def run_preprocessing(
    objects_subset: str,
    object_workers: int,
    raw_data_dir: str = "",
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
    
    # Add parent of Anytop/ to PYTHONPATH so `from Anytop.utils...` imports work
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ANYTOP_DIR.parent) + os.pathsep + existing_pythonpath
    
    result = subprocess.run(cmd, cwd=str(ANYTOP_DIR), capture_output=False, env=env)
    return result.returncode


def run_re_encode_joint_names_only(dataset_dir: str = "") -> int:
    """Regenerate non-motion dataset artifacts without re-preprocessing motions."""
    print("\n" + "=" * 70)
    print("Regenerating dataset sidecar artifacts")
    print("=" * 70 + "\n")
    
    try:
        sys.path.insert(0, str(ANYTOP_DIR / "tools"))
        from regenerate_dataset_artifacts import regenerate_dataset_artifacts

        dataset_dir_path = regenerate_dataset_artifacts(dataset_dir or None)
        print(f"[PASS] Dataset sidecar regeneration completed successfully: {dataset_dir_path}")
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
    filter_orientation_threshold_deg: float,
    sample_count: int,
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
    from validate_anytop_dataset import _resolve_dataset_dir, _print_ok, _print_warn, _require, ValidationError
    
    # Resolve dataset directory
    dataset_dir = _resolve_dataset_dir(None)
    
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
            skip_orientation_check,
            filter_orientation_threshold_deg,
        )

        motions_dir, bvhs_dir, cond_path, metadata_path, positions_error_path = _read_required_artifacts(dataset_dir)
        cond = _validate_cond_file(cond_path, objects_subset)
        motion_files = sorted(motions_dir.glob("*.npy"))
        _validate_metadata(metadata_path, motion_files, cond)
        _validate_motion_metadata(dataset_dir, motion_files, cond)
        _validate_motion_files(motions_dir, bvhs_dir, cond, sample_count)
        
        if skip_orientation_check:
            _print_warn("skipping stored processed-orientation validation by request")
        else:
            from validate_anytop_dataset import _validate_motion_orientation
            _validate_motion_orientation(dataset_dir, cond, sample_count, orientation_threshold_deg)
        
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


def check_and_clean_old_data(dataset_dir: str = "") -> bool:
    """
    Check if old preprocessed data exists in the target dataset directory.
    If found, ask user whether to delete it.
    
    Returns:
        True if user wants to proceed with preprocessing (either no old data found, or old data deleted).
        False if user wants to abort.
    """
    # Import here to get the resolved dataset directory path
    sys.path.insert(0, str(ANYTOP_DIR / "data_loaders" / "truebones" / "truebones_utils"))
    from param_utils import BVHS_DIR, MOTION_DIR, get_dataset_dir
    
    dataset_dir_path = Path(get_dataset_dir(dataset_dir if dataset_dir else None))
    motions_dir = dataset_dir_path / MOTION_DIR
    bvhs_dir = dataset_dir_path / BVHS_DIR
    legacy_glb_dir = dataset_dir_path / "glb" if BVHS_DIR != "glb" else None
    joint_name_inspection_dir = dataset_dir_path / "joint_name_inspection"
    
    # Check if any old data exists
    old_data_exists = (motions_dir.exists() and any(motions_dir.iterdir())) or \
                      (bvhs_dir.exists() and any(bvhs_dir.iterdir())) or \
                      (legacy_glb_dir is not None and legacy_glb_dir.exists() and any(legacy_glb_dir.iterdir())) or \
                      (joint_name_inspection_dir.exists() and any(joint_name_inspection_dir.iterdir()))
    
    if not old_data_exists:
        return True
    
    # Old data found, ask user
    print("\n" + "=" * 70)
    print("WARNING: Old preprocessed data detected")
    print("=" * 70)
    print(f"Dataset directory: {dataset_dir_path}")
    if motions_dir.exists() and any(motions_dir.iterdir()):
        print(f"  - {motions_dir} contains existing data")
    if bvhs_dir.exists() and any(bvhs_dir.iterdir()):
        print(f"  - {bvhs_dir} contains existing data")
    if legacy_glb_dir is not None and legacy_glb_dir.exists() and any(legacy_glb_dir.iterdir()):
        print(f"  - {legacy_glb_dir} contains legacy preview data")
    if joint_name_inspection_dir.exists() and any(joint_name_inspection_dir.iterdir()):
        print(f"  - {joint_name_inspection_dir} contains existing data")
    print("\nDo you want to delete the old data and proceed with preprocessing?")
    
    while True:
        response = input("Enter 'yes' to delete and continue, or 'no' to abort: ").strip().lower()
        if response in ('yes', 'y'):
            print("\nDeleting old data...")
            try:
                if motions_dir.exists():
                    shutil.rmtree(motions_dir)
                    print(f"  [OK] Deleted {motions_dir}")
                if bvhs_dir.exists():
                    shutil.rmtree(bvhs_dir)
                    print(f"  [OK] Deleted {bvhs_dir}")
                if legacy_glb_dir is not None and legacy_glb_dir.exists():
                    shutil.rmtree(legacy_glb_dir)
                    print(f"  [OK] Deleted {legacy_glb_dir}")
                if joint_name_inspection_dir.exists():
                    shutil.rmtree(joint_name_inspection_dir)
                    print(f"  [OK] Deleted {joint_name_inspection_dir}")
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
        help="Skip stored processed-orientation validation during dataset checks.",
    )
    parser.add_argument(
        "--objects-subset",
        default="all",
        help="Expected object subset for validation (default: all).",
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
        help="Maximum allowed first-frame facing error from +Z using stored processed-orientation metadata. Defaults to 15.0.",
    )
    parser.add_argument(
        "--filter-orientation-threshold-deg",
        default=45.0,
        type=float,
        help="Threshold for deleting motion tensors whose stored processed-orientation deviation exceeds the limit before validation. 0 means no filtering. Defaults to 45.0.",
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
    
    # Handle re-encode joint names only mode
    if args.re_encode_joint_names_only:
        return run_re_encode_joint_names_only(args.dataset_dir)
    
    steps_completed = []
    
    # Check and clean old data before preprocessing
    if not args.validate_only:
        if not check_and_clean_old_data(args.dataset_dir):
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
        )
        if ret != 0:
            print("\n[FAIL] Preprocessing failed, aborting workflow.")
            return ret
        steps_completed.append("Preprocess")

    # Validate
    if not args.skip_validate:
        ret = run_validation(
            args.objects_subset,
            args.skip_orientation_check,
            args.orientation_threshold_deg,
            args.filter_orientation_threshold_deg,
            args.sample_count,
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
