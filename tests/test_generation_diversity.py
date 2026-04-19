"""Measure generated-vs-training diversity ratio for motion samples.

Compares mean pairwise L2 distance of generated .npy samples against the
training clips they map to, per action category.

Usage:
    python tests/test_generation_diversity.py [--gen_dir <path>] [--action_tags locomotion,pose]

A diversity ratio (gen/train) below 0.3 indicates collapse; above 0.7 is healthy.
"""
import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MOTIONS_DIR   = "dataset/truebones/zoo/truebones_processed/motions"
METADATA_PATH = "dataset/truebones/zoo/truebones_processed/motion_metadata.json"


def mean_feature(arr: np.ndarray) -> np.ndarray:
    return np.nanmean(arr.reshape(arr.shape[0], -1), axis=0)


def pairwise_diversity(feats) -> float:
    feats = np.stack(feats)
    if len(feats) < 2:
        return 0.0
    dists = [
        np.linalg.norm(feats[i] - feats[j])
        for i in range(len(feats))
        for j in range(i + 1, len(feats))
    ]
    return float(np.mean(dists))


def load_training_clips(metadata_path, motions_dir, object_type, action_tags):
    with open(metadata_path) as f:
        md = json.load(f)["motions"]
    clips = {}
    for name, info in md.items():
        if info.get("object_type") != object_type:
            continue
        if info.get("action_category") not in action_tags:
            continue
        fpath = os.path.join(motions_dir, name)
        if not os.path.exists(fpath):
            continue
        arr = np.load(fpath)
        clips[name] = {"feat": mean_feature(arr), "action": info["action_category"]}
    return clips


def load_generated_samples(gen_dir):
    samples = {}
    for fname in sorted(os.listdir(gen_dir)):
        if not fname.endswith(".npy"):
            continue
        arr = np.load(os.path.join(gen_dir, fname))
        samples[fname] = mean_feature(arr)
    return samples


def nn_action(gen_feat, train_clips):
    best, best_d = None, float("inf")
    for name, info in train_clips.items():
        d = float(np.linalg.norm(gen_feat - info["feat"]))
        if d < best_d:
            best_d, best = d, name
    return train_clips[best]["action"] if best else None


def report_diversity(object_type, action_tags, train_clips, gen_samples):
    """Generate diversity report for a single object_type."""
    if not gen_samples:
        return

    # bucket generated samples by nearest-neighbour action category
    gen_by_action = defaultdict(list)
    for feat in gen_samples.values():
        action = nn_action(feat, train_clips)
        if action:
            gen_by_action[action].append(feat)

    train_by_action = defaultdict(list)
    for info in train_clips.values():
        train_by_action[info["action"]].append(info["feat"])

    print(f"\n{'='*75}")
    print(f"Object Type: {object_type}")
    print(f"{'='*75}")
    print(f"{'Action':<14} {'Train clips':>11} {'Train div':>10} {'Gen samples':>12} {'Gen div':>10} {'Ratio':>8}  Status")
    print("-" * 75)
    for tag in action_tags:
        t_feats = train_by_action.get(tag, [])
        g_feats = gen_by_action.get(tag, [])
        t_div = pairwise_diversity(t_feats) if len(t_feats) > 1 else 0.0
        g_div = pairwise_diversity(g_feats) if len(g_feats) > 1 else 0.0
        ratio = g_div / t_div if t_div > 1e-8 else float("nan")
        if np.isnan(ratio):
            status = "N/A"
        elif ratio < 0.3:
            status = "<<< COLLAPSED"
        elif ratio < 0.7:
            status = "moderate"
        else:
            status = "OK"
        print(f"{tag:<14} {len(t_feats):>11} {t_div:>10.4f} {len(g_feats):>12} {g_div:>10.4f} {ratio:>8.3f}  {status}")

    all_t = [info["feat"] for info in train_clips.values()]
    all_g = list(gen_samples.values())
    overall_ratio = pairwise_diversity(all_g) / max(pairwise_diversity(all_t), 1e-8)
    print(f"\n{'Overall':<14} {len(all_t):>11} {pairwise_diversity(all_t):>10.4f} {len(all_g):>12} {pairwise_diversity(all_g):>10.4f} {overall_ratio:>8.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", required=True)
    parser.add_argument("--object_type", default="Horse", type=str,
                        help="Object type(s) to analyze. Can be comma or space separated for multiple types.")
    parser.add_argument("--action_tags", default="locomotion,pose")
    args = parser.parse_args()

    action_tags = [t.strip() for t in args.action_tags.split(",")]
    
    # Parse multiple object_types (space or comma separated)
    object_types_str = args.object_type.replace(',', ' ')
    object_types = [t.strip() for t in object_types_str.split() if t.strip()]
    
    gen_samples = load_generated_samples(args.gen_dir)
    if not gen_samples:
        print("No generated .npy files found.")
        return

    # Process each object_type
    for obj_type in object_types:
        train_clips = load_training_clips(METADATA_PATH, MOTIONS_DIR, obj_type, action_tags)
        # Filter gen_samples for this object_type (by filename prefix)
        filtered_gen_samples = {k: v for k, v in gen_samples.items() if k.startswith(obj_type)}
        if filtered_gen_samples:
            report_diversity(obj_type, action_tags, train_clips, filtered_gen_samples)
        else:
            print(f"\nNo generated samples found for object_type: {obj_type}")


if __name__ == "__main__":
    main()
