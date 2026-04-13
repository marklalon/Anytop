from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import BVH
from data_loaders.get_data import get_dataset_loader
from data_loaders.skeleton_metadata import load_skeleton_metadata
from data_loaders.truebones.offline_reference_dataset import load_cond_dict, resolve_dataset_root
from data_loaders.truebones.truebones_utils.motion_process import recover_animation_from_motion_np
from eval.biomechanical_negatives import NEGATIVE_KINDS, generate_biomechanical_negative_batch
from utils.fixseed import fixseed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export clean/negative motion windows as BVH files for manual review.")
    parser.add_argument("--output-dir", required=True, help="Directory to write BVH exports and manifest.json into.")
    parser.add_argument("--dataset-dir", default="", help="Processed dataset root. Empty uses the repo default dataset path.")
    parser.add_argument("--objects-subset", default="all", help="Object subset to sample from.")
    parser.add_argument("--action-tags", default="", help="Optional action-tag filter, e.g. 'locomotion,attack'.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "all"], help="Dataset split to sample from.")
    parser.add_argument("--num-frames", default=60, type=int, help="Sampled window length in frames.")
    parser.add_argument("--batch-size", default=16, type=int, help="Loader batch size used while searching for examples.")
    parser.add_argument("--samples-per-kind", default=10, type=int, help="How many examples to export per negative kind.")
    parser.add_argument("--sample-limit", default=0, type=int, help="Optional dataset sample cap for debug runs. 0 keeps the full split.")
    parser.add_argument("--num-workers", default=0, type=int, help="DataLoader worker count.")
    parser.add_argument("--main-process-prefetch-batches", default=0, type=int, help="Background-prefetch batches when num_workers=0.")
    parser.add_argument("--seed", default=1234, type=int, help="Random seed used for dataset sampling and negative generation.")
    parser.add_argument(
        "--negative-kinds",
        nargs="*",
        default=list(NEGATIVE_KINDS),
        help="Subset of negative kinds to export. Defaults to all registered kinds.",
    )
    return parser.parse_args()


def _normalize_requested_negative_kinds(values: list[str]) -> tuple[str, ...]:
    requested = tuple(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
    if not requested:
        raise ValueError("At least one negative kind must be requested.")
    invalid = [value for value in requested if value not in NEGATIVE_KINDS]
    if invalid:
        raise ValueError(f"Unknown negative kinds: {invalid}. Valid values: {list(NEGATIVE_KINDS)}")
    return requested


def _denormalize_motion(sample_motion: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, n_joints: int, length: int) -> np.ndarray:
    motion_tjf = sample_motion[:n_joints, :, :length].permute(2, 0, 1).detach().cpu().numpy().astype(np.float32, copy=False)
    mean_jf = mean[:n_joints].detach().cpu().numpy().astype(np.float32, copy=False)
    std_jf = std[:n_joints].detach().cpu().numpy().astype(np.float32, copy=False)
    return np.nan_to_num(motion_tjf * std_jf[None, :, :] + mean_jf[None, :, :]).astype(np.float32, copy=False)


def _export_motion_bvh(save_path: Path, motion_tjf: np.ndarray, parents: list[int], offsets: np.ndarray, joint_names: list[str]) -> None:
    animation, has_animated_positions = recover_animation_from_motion_np(motion_tjf, parents, offsets)
    if animation is None:
        raise RuntimeError(f"Failed to recover animation for {save_path}")
    BVH.save(str(save_path), animation, joint_names, positions=has_animated_positions)


def main() -> int:
    args = parse_args()
    requested_negative_kinds = _normalize_requested_negative_kinds(args.negative_kinds)
    if args.samples_per_kind <= 0:
        raise ValueError("samples-per-kind must be > 0")

    fixseed(int(args.seed))
    dataset_root = resolve_dataset_root(args.dataset_dir or None)
    cond_dict = load_cond_dict(dataset_root)
    skeleton_lookup = load_skeleton_metadata(cond_dict=cond_dict)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = get_dataset_loader(
        batch_size=int(args.batch_size),
        num_frames=int(args.num_frames),
        split=str(args.split),
        temporal_window=31,
        t5_name="t5-base",
        balanced=False,
        objects_subset=str(args.objects_subset),
        num_workers=int(args.num_workers),
        prefetch_factor=2,
        sample_limit=int(args.sample_limit),
        shuffle=True,
        drop_last=False,
        use_reference_conditioning=False,
        action_tags=str(args.action_tags),
        motion_cache_size=0,
        main_process_prefetch_batches=int(args.main_process_prefetch_batches),
    )

    exported_counts = {kind: 0 for kind in requested_negative_kinds}
    manifest: list[dict[str, object]] = []

    for motion, cond in loader:
        object_types = [str(value) for value in cond["y"].get("object_type", [])]
        negative_batch = generate_biomechanical_negative_batch(
            motion,
            cond["y"]["n_joints"],
            cond["y"]["lengths"],
            object_types,
            skeleton_lookup,
            feature_std=cond["y"].get("std"),
            negative_kinds=requested_negative_kinds,
        )

        for batch_index, negative_kind in enumerate(negative_batch["negative_kinds"]):
            if exported_counts[negative_kind] >= int(args.samples_per_kind):
                continue

            object_type = object_types[batch_index]
            motion_name = str(cond["y"].get("motion_name", [f"sample_{batch_index:04d}"] * len(object_types))[batch_index])
            length = int(cond["y"]["lengths"][batch_index].item())
            n_joints = int(cond["y"]["n_joints"][batch_index].item())
            sample_index = exported_counts[negative_kind] + 1
            sample_dir = output_dir / negative_kind / f"{sample_index:02d}_{Path(motion_name).stem}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            clean_motion = _denormalize_motion(motion[batch_index], cond["y"]["mean"][batch_index], cond["y"]["std"][batch_index], n_joints, length)
            negative_motion = _denormalize_motion(negative_batch["motion"][batch_index], cond["y"]["mean"][batch_index], cond["y"]["std"][batch_index], n_joints, length)
            motion_abs_delta = np.abs(negative_motion - clean_motion)

            object_cond = cond_dict[object_type]
            parents = [int(parent) for parent in object_cond["parents"]]
            offsets = np.asarray(object_cond["offsets"], dtype=np.float32)
            joint_names = list(object_cond["joints_names"])

            _export_motion_bvh(sample_dir / "clean.bvh", clean_motion, parents, offsets, joint_names)
            _export_motion_bvh(sample_dir / "negative.bvh", negative_motion, parents, offsets, joint_names)

            record = {
                "negative_kind": negative_kind,
                "motion_name": motion_name,
                "object_type": object_type,
                "length": length,
                "n_joints": n_joints,
                "crop_start_ind": int(cond["y"]["crop_start_ind"][batch_index].item()),
                "species_label": str(cond["y"].get("species_label", [""] * len(object_types))[batch_index]),
                "action_label": str(cond["y"].get("action_label", [""] * len(object_types))[batch_index]),
                "feature_max_abs_delta": float(motion_abs_delta.max()),
                "feature_mean_abs_delta": float(motion_abs_delta.mean()),
                "sample_dir": str(sample_dir),
                "clean_bvh": str(sample_dir / "clean.bvh"),
                "negative_bvh": str(sample_dir / "negative.bvh"),
            }
            with open(sample_dir / "metadata.json", "w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2)
            manifest.append(record)
            exported_counts[negative_kind] = sample_index

        if all(count >= int(args.samples_per_kind) for count in exported_counts.values()):
            break

    summary = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "requested_negative_kinds": list(requested_negative_kinds),
        "samples_per_kind": int(args.samples_per_kind),
        "exported_counts": exported_counts,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    missing = {kind: int(args.samples_per_kind) - count for kind, count in exported_counts.items() if count < int(args.samples_per_kind)}
    if missing:
        print(f"Export incomplete. Missing counts: {missing}")
        return 1

    print(f"Exported {len(manifest)} negative preview pairs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())