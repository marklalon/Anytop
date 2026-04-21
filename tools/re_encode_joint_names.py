#!/usr/bin/env python3
"""
Quick script to re-encode joint names into cond.npy without re-preprocessing motions.

This is a lightweight alternative to: python preprocess_and_validate.py --re-encode-joint-names-only

Usage:
    python tools/re_encode_joint_names.py [--dataset-dir PATH] [--t5-model NAME]

Options:
    --dataset-dir PATH      Path to dataset directory (uses default if not specified)
    --t5-model NAME         T5 model name to use (default: t5-base)

Examples:
    # Re-encode with default settings
    python tools/re_encode_joint_names.py

    # Re-encode with custom dataset directory
    python tools/re_encode_joint_names.py --dataset-dir /path/to/dataset

    # Re-encode with different T5 model
    python tools/re_encode_joint_names.py --t5-model t5-large
"""

import argparse
import sys
from pathlib import Path
import numpy as np

ANYTOP_DIR = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-encode joint names into cond.npy",
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
    print("Re-encoding joint names into cond.npy")
    print("=" * 70 + "\n")
    
    # Import utilities
    sys.path.insert(0, str(ANYTOP_DIR / "data_loaders" / "truebones" / "truebones_utils"))
    from param_utils import get_dataset_dir
    from motion_process import _attach_joint_name_embeddings_to_cond
    
    # Resolve dataset directory
    dataset_dir_path = get_dataset_dir(args.dataset_dir if args.dataset_dir else None)
    cond_path = Path(dataset_dir_path) / "cond.npy"
    
    if not cond_path.exists():
        print(f"ERROR: cond.npy not found at {cond_path}")
        print("Please run full preprocessing first with: python preprocess_and_validate.py")
        return 1
    
    try:
        print(f"Loading cond.npy from {cond_path}")
        cond = dict(np.load(cond_path, allow_pickle=True).item())
        
        print(f"Found {len(cond)} objects in cond.npy: {', '.join(sorted(cond.keys()))}")
        
        # Re-encode joint names
        print(f"Re-encoding joint names with {args.t5_model}...")
        _attach_joint_name_embeddings_to_cond(cond, str(Path(dataset_dir_path)), t5_name=args.t5_model)
        
        # Save back
        print(f"Saving updated cond.npy")
        np.save(str(cond_path), cond)
        
        print("\n" + "=" * 70)
        print("[PASS] Joint name re-encoding completed successfully")
        print("=" * 70)
        return 0
    except Exception as e:
        print(f"ERROR: Failed to re-encode joint names: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
