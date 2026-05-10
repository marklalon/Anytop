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
import os
from pathlib import Path

import numpy as np

# ANSI colors (safe on Windows 10+)
os.system("")  # enables VT100 on Windows

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_lib import BVH
from data_loaders.truebones.truebones_utils.motion_labels import load_motion_metadata
from data_loaders.truebones.truebones_utils.motion_process import (
    mirror_features_with_safeguards,
    recover_bvh_export_animation_from_motion_np,
    resolve_mirrored_export_skeleton_metadata,
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
    motion_metadata: dict[str, object],
    *,
    object_cond: dict[str, object] | None = None,
    mirror_export_compat: bool = False,
) -> bool:
    export_parents = list(parents)
    export_offsets = np.asarray(offsets)
    export_joint_names = list(joints_names)
    if mirror_export_compat and object_cond is not None:
        export_parents, export_offsets, export_joint_names = resolve_mirrored_export_skeleton_metadata(
            object_cond,
            export_parents,
            export_offsets,
            export_joint_names,
        )

    anim, joints_names, has_animated_pos = recover_bvh_export_animation_from_motion_np(
        motion,
        export_parents,
        export_offsets,
        export_joint_names,
        motion_metadata=motion_metadata,
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
    parser.add_argument("--sample-count", default=1, type=int, help="Motions to export per symmetric object type.")
    parser.add_argument("--random-seed", default=1234, type=int, help="RNG seed for reproducible sampling.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = resolve_dataset_root(args.dataset_dir or None)
    motion_dir = get_motion_dir(dataset_root)
    cond_dict = load_cond_dict(dataset_root)
    motion_metadata_lookup = load_motion_metadata(dataset_root)
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
    for obj, files in sorted(by_object.items()):
        n = min(args.sample_count, len(files))
        selected = sorted(rng.sample(files, n)) if n > 0 else []
        object_cond = cond_dict[obj]
        parents = [int(p) for p in object_cond['parents']]
        offsets = object_cond['offsets']
        joints_names = list(
            object_cond.get('canonical_bvh_joint_names', object_cond['joints_names'])
        )
        spi = np.asarray(object_cond['symmetry_partner_indices'])

        symmetric_pairs = [(i, int(spi[i])) for i in range(len(spi)) if spi[i] != -1]
        print(f"[{obj}] {len(symmetric_pairs)} symmetric pairs, exporting {len(selected)} motions…")

        for motion_file in selected:
            stem = Path(motion_file).stem
            motion = np.load(motion_dir / motion_file).astype(np.float32)
            motion_metadata = motion_metadata_lookup.get(motion_file)
            if not isinstance(motion_metadata, dict):
                raise KeyError(f"Motion '{motion_file}' is missing explicit motion metadata.")
            mirrored, mirrored_offsets = mirror_features_with_safeguards(
                motion,
                object_cond,
                motion_metadata=motion_metadata,
            )

            clean_path = output_dir / f"{obj}_{stem}_clean.bvh"
            mirror_path = output_dir / f"{obj}_{stem}_mirror.bvh"

            ok_clean = export_bvh(clean_path, motion, parents, offsets, joints_names, motion_metadata)
            ok_mirror = export_bvh(
                mirror_path,
                mirrored,
                parents,
                mirrored_offsets,
                joints_names,
                motion_metadata,
                object_cond=object_cond,
                mirror_export_compat=True,
            )

            if not ok_clean or not ok_mirror:
                fail = "\033[91mFAILED\033[0m"
                status = []
                status.append(f"clean: {'OK' if ok_clean else fail}")
                status.append(f"mirror: {'OK' if ok_mirror else fail}")
                print(f"  {stem}: {' | '.join(status)}")
            if ok_clean and ok_mirror:
                exported_total += 1

    print(f"\nDone. Exported {exported_total} clean+mirror pairs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
