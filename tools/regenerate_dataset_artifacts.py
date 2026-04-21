#!/usr/bin/env python3
"""
Quick script to regenerate dataset sidecar artifacts without re-preprocessing motions.

This is a lightweight alternative to: python preprocess_and_validate.py --re-encode-joint-names-only

Usage:
    python tools/regenerate_dataset_artifacts.py [--dataset-dir PATH] [--t5-model NAME]

Options:
    --dataset-dir PATH      Path to dataset directory (uses default if not specified)
    --t5-model NAME         T5 model name to use (default: t5-base)

Examples:
    # Regenerate sidecar artifacts with default settings
    python tools/regenerate_dataset_artifacts.py

    # Re-encode with custom dataset directory
    python tools/regenerate_dataset_artifacts.py --dataset-dir /path/to/dataset

    # Re-encode with different T5 model
    python tools/regenerate_dataset_artifacts.py --t5-model t5-large
"""

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ANYTOP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ANYTOP_DIR))
sys.path.insert(0, str(ANYTOP_DIR / "data_loaders" / "truebones"))

from truebones_utils.motion_labels import (  # noqa: E402
    infer_motion_labels_from_motion_name,
    load_motion_metadata,
    write_motion_metadata,
)
from truebones_utils.motion_process import _attach_joint_name_embeddings_to_cond  # noqa: E402
from truebones_utils.param_utils import MOTION_DIR, get_dataset_dir  # noqa: E402


def _normalize_identifier(value: str) -> str:
    return "".join(char for char in value.lower() if char.isalnum())


def _resolve_dataset_dir_path(dataset_dir: str | Path | None) -> Path:
    raw_value = str(dataset_dir) if dataset_dir else None
    return Path(get_dataset_dir(raw_value)).resolve()


def _write_metadata_summary(
    dataset_dir_path: Path,
    object_counts: Counter[str],
    max_joints: int,
    total_clips: int,
    total_frames: int,
) -> None:
    metadata_path = dataset_dir_path / "metadata.txt"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        handle.write(f"max joints: {max_joints}\n")
        handle.write(f"total frames: {total_frames}\n")
        handle.write(f"duration: {total_frames / 12.5 / 60}\n")
        handle.write(f"~~~~ objects_counts - Total: {total_clips} ~~~~\n")
        for object_type in sorted(object_counts):
            handle.write(f"{object_type}: {object_counts[object_type]}\n")


def _rewrite_positions_error_file(dataset_dir_path: Path, motion_entries: dict[str, dict[str, object]]) -> None:
    positions_error_path = dataset_dir_path / "positions_error_rate.txt"
    existing_lines: list[str] = []
    if positions_error_path.exists():
        existing_lines = positions_error_path.read_text(encoding="utf-8").splitlines()

    motion_signatures = [
        (
            _normalize_identifier(str(entry.get("object_type", ""))),
            _normalize_identifier(str(entry.get("action_label", ""))),
        )
        for entry in motion_entries.values()
    ]
    filtered_lines: list[str] = []
    for line in existing_lines[1:]:
        normalized_line = _normalize_identifier(line)
        if any(obj and obj in normalized_line and (not action or action in normalized_line) for obj, action in motion_signatures):
            filtered_lines.append(line)

    output_lines = ["Position squared error per bvh file:__artifact_regenerated__: 0.000000"]
    output_lines.extend(filtered_lines)
    if len(output_lines) == 1:
        output_lines.append("__artifact_regenerated_placeholder__: 0.000000")
    positions_error_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def regenerate_dataset_artifacts(dataset_dir: str | Path | None = None, t5_model: str = "t5-base") -> Path:
    dataset_dir_path = _resolve_dataset_dir_path(dataset_dir)
    motions_dir = dataset_dir_path / MOTION_DIR
    cond_path = dataset_dir_path / "cond.npy"

    if not motions_dir.exists():
        raise RuntimeError(f"motions directory not found at {motions_dir}")
    if not cond_path.exists():
        raise RuntimeError(f"cond.npy not found at {cond_path}")

    motion_files = sorted(motions_dir.glob("*.npy"))
    if not motion_files:
        raise RuntimeError(f"no motion files found under {motions_dir}")

    existing_cond = dict(np.load(cond_path, allow_pickle=True).item())
    known_object_types = tuple(existing_cond.keys())
    active_object_types = sorted(
        {
            str(infer_motion_labels_from_motion_name(motion_path.name, object_types=known_object_types).get("object_type"))
            for motion_path in motion_files
        }
    )
    missing_object_types = [object_type for object_type in active_object_types if object_type not in existing_cond]
    if missing_object_types:
        raise RuntimeError(
            f"cond.npy is missing object entries required by current motions: {missing_object_types}"
        )

    rebuilt_cond = {object_type: existing_cond[object_type] for object_type in active_object_types}

    inspection_dir = dataset_dir_path / "joint_name_inspection"
    if inspection_dir.exists():
        shutil.rmtree(inspection_dir)
    collision_report_path = dataset_dir_path / "joint_name_collision_report.json"
    if collision_report_path.exists():
        collision_report_path.unlink()

    _attach_joint_name_embeddings_to_cond(rebuilt_cond, str(dataset_dir_path), t5_name=t5_model)
    np.save(str(cond_path), rebuilt_cond)

    existing_motion_metadata = load_motion_metadata(dataset_dir_path)
    rebuilt_motion_metadata: dict[str, dict[str, object]] = {}
    object_counts: Counter[str] = Counter()
    total_frames = 0
    max_joints = 0
    for motion_path in motion_files:
        motion = np.load(motion_path, mmap_mode="r")
        total_frames += int(motion.shape[0])
        max_joints = max(max_joints, int(motion.shape[1]))

        motion_entry = dict(existing_motion_metadata.get(motion_path.name, {}))
        motion_entry.update(
            infer_motion_labels_from_motion_name(motion_path.name, object_types=tuple(rebuilt_cond.keys()))
        )
        motion_entry["motion_name"] = motion_path.name
        motion_entry.setdefault("is_loop", False)
        rebuilt_motion_metadata[motion_path.name] = motion_entry
        object_counts[str(motion_entry["object_type"])] += 1

    _write_metadata_summary(
        dataset_dir_path,
        object_counts,
        max_joints,
        len(motion_files),
        total_frames,
    )
    write_motion_metadata(dataset_dir_path, rebuilt_motion_metadata, len(motion_files))
    _rewrite_positions_error_file(dataset_dir_path, rebuilt_motion_metadata)

    return dataset_dir_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate dataset sidecar artifacts",
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
    print("Regenerating dataset sidecar artifacts")
    print("=" * 70 + "\n")
    
    try:
        dataset_dir_path = regenerate_dataset_artifacts(args.dataset_dir, t5_model=args.t5_model)
        cond_path = dataset_dir_path / "cond.npy"
        cond = dict(np.load(cond_path, allow_pickle=True).item())

        print(f"Regenerated dataset artifacts under {dataset_dir_path}")
        print(f"Saved cond.npy with {len(cond)} objects: {', '.join(sorted(cond.keys()))}")
        
        print("\n" + "=" * 70)
        print("[PASS] Dataset sidecar regeneration completed successfully")
        print("=" * 70)
        return 0
    except Exception as e:
        print(f"ERROR: Failed to regenerate dataset artifacts: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
