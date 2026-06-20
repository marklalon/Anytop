#!/usr/bin/env python3
"""Analyze skeleton joint count distribution from the training data."""

import numpy as np
import json
from pathlib import Path
from collections import defaultdict, Counter
import sys

# Add Anytop root to path for absolute imports
_anytop_root = Path(__file__).resolve().parent.parent
if str(_anytop_root) not in sys.path:
    sys.path.insert(0, str(_anytop_root))

from data_loaders.truebones.truebones_utils.param_utils import (
    get_dataset_dir, MOTION_DIR, BVHS_DIR, MAX_JOINTS
)

def load_motion_files():
    """Load all motion files and analyze joint distributions."""
    dataset_dir = Path(get_dataset_dir())
    motions_dir = dataset_dir / MOTION_DIR
    
    joint_counts = Counter()
    species_joints = defaultdict(list)
    
    print(f"Dataset directory: {dataset_dir}")
    print(f"Motions directory: {motions_dir}")
    print(f"MAX_JOINTS constant: {MAX_JOINTS}\n")
    
    motion_files = sorted(motions_dir.glob("*.npy"))
    print(f"Total motion files: {len(motion_files)}\n")
    
    for motion_file in motion_files:
        motion = np.load(motion_file)
        n_joints = motion.shape[1]
        joint_counts[n_joints] += 1
        
        # Extract species from filename (e.g., "Horse_Walk_001.npy" -> "Horse")
        species = motion_file.stem.split("_")[0]
        species_joints[species].append(n_joints)
    
    return joint_counts, species_joints

def analyze_and_print(joint_counts, species_joints):
    """Print detailed analysis of joint distribution."""
    
    total_files = sum(joint_counts.values())
    
    print("=" * 70)
    print("OVERALL JOINT COUNT DISTRIBUTION")
    print("=" * 70)
    
    sorted_counts = sorted(joint_counts.items())
    
    print(f"\n{'Joint Count':<15} {'Count':<10} {'Percentage':<12} {'Cumulative %'}")
    print("-" * 55)
    
    cumulative = 0
    for n_joints, count in sorted_counts:
        percent = (count / total_files) * 100
        cumulative += percent
        print(f"{n_joints:<15} {count:<10} {percent:>10.2f}% {cumulative:>10.2f}%")
    
    # Statistics
    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)
    
    joint_values = []
    for n_joints, count in joint_counts.items():
        joint_values.extend([n_joints] * count)
    
    joint_values = np.array(joint_values)
    
    print(f"Total motion files:  {total_files}")
    print(f"Min joints:          {joint_values.min()}")
    print(f"Max joints:          {joint_values.max()}")
    print(f"Mean joints:         {joint_values.mean():.2f}")
    print(f"Median joints:       {np.median(joint_values):.0f}")
    print(f"Std deviation:       {joint_values.std():.2f}")
    print(f"MAX_JOINTS constant: {MAX_JOINTS}")
    print(f"Padding ratio:       {MAX_JOINTS / joint_values.mean():.2f}x")
    print(f"Max wasted capacity: {(MAX_JOINTS - joint_values.min()) / MAX_JOINTS * 100:.1f}%")
    
    # Percentiles
    percentiles = [25, 50, 75, 90, 95, 99]
    print(f"\nPercentiles:")
    for p in percentiles:
        val = np.percentile(joint_values, p)
        print(f"  {p}th: {val:.0f}")
    
    # Per-species analysis
    print("\n" + "=" * 70)
    print("PER-SPECIES ANALYSIS")
    print("=" * 70)
    print(f"\n{'Species':<20} {'Count':<10} {'Min':<8} {'Max':<8} {'Mean':<10} {'Padding Ratio'}")
    print("-" * 75)
    
    for species in sorted(species_joints.keys()):
        counts = species_joints[species]
        count = len(counts)
        min_j = min(counts)
        max_j = max(counts)
        mean_j = np.mean(counts)
        padding_ratio = MAX_JOINTS / mean_j
        
        print(f"{species:<20} {count:<10} {min_j:<8} {max_j:<8} {mean_j:<10.1f} {padding_ratio:.2f}x")
    
    # Species count
    print(f"\nTotal species: {len(species_joints)}")
    
    # Recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    p99 = np.percentile(joint_values, 99)
    p95 = np.percentile(joint_values, 95)
    
    print(f"\nCurrent MAX_JOINTS: {MAX_JOINTS}")
    print(f"99th percentile:    {p99:.0f}")
    print(f"95th percentile:    {p95:.0f}")
    print(f"99th percentile + buffer (×1.1): {p99 * 1.1:.0f}")
    
    if MAX_JOINTS > p99 * 1.2:
        print(f"\n⚠️  MAX_JOINTS={MAX_JOINTS} is significantly higher than 99th percentile ({p99:.0f})")
        print(f"   Consider reducing to ~{int(p99 * 1.1)} to reduce padding overhead")
    elif MAX_JOINTS < joint_values.max():
        print(f"\n⚠️  MAX_JOINTS={MAX_JOINTS} is less than max actual joints ({joint_values.max()})")
        print(f"   Some motion data may be truncated!")
    else:
        print(f"\n✓ MAX_JOINTS={MAX_JOINTS} covers all data with reasonable padding")

if __name__ == "__main__":
    print("Analyzing skeleton joint count distribution...\n")
    
    try:
        joint_counts, species_joints = load_motion_files()
        analyze_and_print(joint_counts, species_joints)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
