"""
Mirror Augmentation Verification Tool

Description:
    Exports BVH pairs (clean + mirrored) for manual verification of aug_mirror_prob.
    For each sampled motion, writes:
      - {stem}_clean.bvh    : original motion
      - {stem}_mirror.bvh   : mirror-augmented motion (joint swap + YZ-plane flip)
    Skips objects that have no bilateral symmetry metadata.

Usage:
    python export_mirror_verification.py \\
        --dataset-dir ./data/processed_anytop \\
        --output-dir ./mirror_verification \\
        --sample-count 4

    # Filter to a specific object subset:
    python export_mirror_verification.py \\
        --dataset-dir ./data/processed_anytop \\
        --objects-subset quadropeds_clean \\
        --sample-count 6 \\
        --random-seed 42

Arguments:
    --dataset-dir   : Processed dataset root (auto-detected if omitted).
    --output-dir    : Where to write BVH files (default: ./mirror_verification).
    --objects-subset: "all" or a named subset from OBJECT_SUBSETS_DICT.
    --sample-count  : Number of motions to export per symmetric object type (default: 2).
    --random-seed   : Seed for reproducible sampling (default: 0).
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_lib import BVH
from data_loaders.truebones.truebones_utils.motion_process import (
    mirror_features_with_safeguards,
    recover_bvh_export_animation_from_motion_np,
)
from utils.misc import infer_object_type_from_filename

from data_loaders.truebones.offline_reference_dataset import (
    list_motion_files,
    load_cond_dict,
    get_motion_dir,
    resolve_dataset_root,
)


def export_bvh(
    save_path: Path,
    motion: np.ndarray,
    parents: list[int],
    offsets: np.ndarray,
    joints_names: list[str],
) -> bool:
    anim, joints_names, has_animated_pos = recover_bvh_export_animation_from_motion_np(
        motion,
        parents,
        offsets,
        joints_names,
    )
    if anim is None:
        return False
    BVH.save(str(save_path), anim, joints_names, positions=has_animated_pos)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export clean + mirrored BVH pairs for aug_mirror_prob verification.")
    parser.add_argument("--dataset-dir", default="", help="Processed dataset root directory.")
    parser.add_argument("--output-dir", default="outputs/mirror_verification", help="Output directory for BVH files.")
    parser.add_argument("--objects-subset", default="all", help="Object subset to sample from.")
    parser.add_argument("--sample-count", default=8, type=int, help="Motions to export per symmetric object type.")
    parser.add_argument("--random-seed", default=0, type=int, help="RNG seed for reproducible sampling.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = resolve_dataset_root(args.dataset_dir or None)
    motion_dir = get_motion_dir(dataset_root)
    cond_dict = load_cond_dict(dataset_root)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.random_seed)

    all_motion_files = list_motion_files(
        dataset_dir=dataset_root,
        objects_subset=args.objects_subset,
        sample_limit=0,
    )

    # Group files by object type, keeping only symmetric objects.
    by_object: dict[str, list[str]] = {}
    skipped_asymmetric: set[str] = set()
    for motion_file in all_motion_files:
        obj = infer_object_type_from_filename(motion_file, valid_types=cond_dict.keys())
        if obj is None:
            continue
        spi = cond_dict[obj].get('symmetry_partner_indices')
        if spi is None or all(int(p) == -1 for p in spi):
            skipped_asymmetric.add(obj)
            continue
        by_object.setdefault(obj, []).append(motion_file)

    if skipped_asymmetric:
        print(f"Skipped (no bilateral symmetry): {sorted(skipped_asymmetric)}")

    if not by_object:
        print("No symmetric object types found. Nothing to export.")
        return 1

    exported_total = 0
    remaining = args.sample_count
    for obj, files in sorted(by_object.items()):
        n = min(remaining, len(files))
        selected = sorted(rng.sample(files, n)) if n > 0 else []
        remaining -= n
        object_cond = cond_dict[obj]
        parents = [int(p) for p in object_cond['parents']]
        offsets = object_cond['offsets']
        joints_names = list(
            object_cond.get('canonical_bvh_joint_names', object_cond['joints_names'])
        )
        spi = np.asarray(object_cond['symmetry_partner_indices'])

        obj_dir = output_dir / obj
        obj_dir.mkdir(parents=True, exist_ok=True)

        symmetric_pairs = [(i, int(spi[i])) for i in range(len(spi)) if spi[i] != -1]
        print(f"\n[{obj}] {len(symmetric_pairs)} symmetric joint pairs:")
        for left, right in symmetric_pairs:
            print(f"  {joints_names[left]} <-> {joints_names[right]}")

        for motion_file in selected:
            stem = Path(motion_file).stem
            motion = np.load(motion_dir / motion_file).astype(np.float32)
            mirrored, mirrored_offsets = mirror_features_with_safeguards(motion, object_cond)

            clean_path = obj_dir / f"{stem}_clean.bvh"
            mirror_path = obj_dir / f"{stem}_mirror.bvh"

            ok_clean = export_bvh(clean_path, motion, parents, offsets, joints_names)
            ok_mirror = export_bvh(mirror_path, mirrored, parents, mirrored_offsets, joints_names)

            status = []
            if ok_clean:
                status.append(f"clean -> {clean_path.name}")
            else:
                status.append("clean FAILED")
            if ok_mirror:
                status.append(f"mirror -> {mirror_path.name}")
            else:
                status.append("mirror FAILED")
            print(f"  {stem}: {' | '.join(status)}")
            if ok_clean and ok_mirror:
                exported_total += 1

    print(f"\nDone. Exported {exported_total} clean+mirror pairs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
