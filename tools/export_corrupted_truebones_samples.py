"""
Corrupted Reference Dataset Export Tool

Description:
    Generates and stores corrupted reference motions alongside the processed AnyTop dataset.
    This tool creates a set of synthetically corrupted motion sequences that serve as
    corruption targets for training the restoration diffusion model.

Features:
    - Processes motion data from specified object subsets
    - Generates corrupted versions of clean motions with controlled corruption patterns
    - Stores corrupted references as numpy files for efficient loading during training
    - Supports filtering by dataset split and sample count limits
    - Uses deterministic seeding for reproducible corruption generation

Usage:
    python export_corrupted_truebones_samples.py \\
        --dataset-dir ./data/processed_anytop \\
        --objects-subset quadropeds_clean \\
        --sample-limit 100 \\
        --seed 1234

    # Export all samples with default dataset directory:
    python export_corrupted_truebones_samples.py \\
        --objects-subset all

    # Minimal example (uses built-in dataset path):
    python export_corrupted_truebones_samples.py

Optional Arguments:
    --dataset-dir: Processed dataset root directory (auto-detected if empty)
    --objects-subset: Object types to process - 'all' or specific subset (default: 'all')
    --sample-limit: Maximum number of motions to export, 0 means all (default: 0)
    --seed: Random seed for deterministic corruption (default: 1234)

Output:
    Creates .reference.npy files in the dataset's corrupted_references directory
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loaders.truebones.offline_reference_dataset import export_corrupted_reference_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate stored corrupted-reference motions alongside the processed AnyTop dataset.")
    parser.add_argument("--dataset-dir", default="", help="Processed dataset root. If not specified, uses default path.")
    parser.add_argument("--objects-subset", default="all", help="Object subset to export corrupted references for.")
    parser.add_argument("--sample-limit", default=0, type=int, help="Limit number of motions to export. 0 exports all matching motions.")
    parser.add_argument("--seed", default=1234, type=int, help="Random seed for deterministic exports.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    export_corrupted_reference_dataset(
        dataset_dir=args.dataset_dir or None,
        objects_subset=args.objects_subset,
        seed=args.seed,
        sample_limit=args.sample_limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())