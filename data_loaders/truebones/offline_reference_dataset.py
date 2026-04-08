from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from data_loaders.truebones.corruption import MotionCorruptor
from data_loaders.truebones.truebones_utils.param_utils import (
    CORRUPTED_REFERENCE_DIR,
    MOTION_DIR,
    OBJECT_SUBSETS_DICT,
    get_dataset_dir,
)


def resolve_dataset_root(dataset_dir: str | Path | None = None) -> Path:
    return Path(dataset_dir or get_dataset_dir()).resolve()


def get_motion_dir(dataset_dir: str | Path | None = None) -> Path:
    return resolve_dataset_root(dataset_dir) / MOTION_DIR


def get_corrupted_reference_dir(
    dataset_dir: str | Path | None = None,
) -> Path:
    return resolve_dataset_root(dataset_dir) / CORRUPTED_REFERENCE_DIR


def get_legacy_corrupted_reference_dir(
    dataset_dir: str | Path | None = None,
) -> Path:
    return resolve_dataset_root(dataset_dir) / CORRUPTED_REFERENCE_DIR / "stage_2"


def infer_object_type(file_name: str, object_types: Iterable[str]) -> str:
    for object_type in sorted(object_types, key=len, reverse=True):
        if file_name.startswith(f"{object_type}_"):
            return object_type
    raise KeyError(f"Could not infer object type from motion file '{file_name}'")


def _matches_object_subset(file_name: str, object_types: set[str]) -> bool:
    return any(file_name.startswith(f"{object_type}_") for object_type in object_types)


def load_cond_dict(dataset_dir: str | Path | None = None) -> dict[str, dict[str, np.ndarray]]:
    dataset_root = resolve_dataset_root(dataset_dir)
    cond_path = dataset_root / "cond.npy"
    if not cond_path.exists():
        raise FileNotFoundError(f"Condition file was not found: {cond_path}")
    return np.load(cond_path, allow_pickle=True).item()


def list_motion_files(
    dataset_dir: str | Path | None = None,
    objects_subset: str = "all",
    motion_names: Iterable[str] | None = None,
    sample_limit: int = 0,
) -> list[str]:
    motion_dir = get_motion_dir(dataset_dir)
    if not motion_dir.exists():
        raise FileNotFoundError(f"Motion directory was not found: {motion_dir}")
    all_motion_files = sorted(path.name for path in motion_dir.glob("*.npy"))
    if motion_names is not None:
        requested = {name for name in motion_names}
        selected = [name for name in all_motion_files if name in requested]
    else:
        allowed_objects = set(OBJECT_SUBSETS_DICT.get(objects_subset, OBJECT_SUBSETS_DICT["all"]))
        selected = [name for name in all_motion_files if _matches_object_subset(name, allowed_objects)]
    if sample_limit > 0:
        selected = selected[:sample_limit]
    return selected


def get_corrupted_sample_paths(
    motion_file: str,
    dataset_dir: str | Path | None = None,
) -> dict[str, Path]:
    sample_stem = Path(motion_file).stem
    reference_dir = get_corrupted_reference_dir(dataset_dir=dataset_dir)
    return {
        "reference_dir": reference_dir,
        "reference": reference_dir / f"{sample_stem}.reference.npy",
        "confidence": reference_dir / f"{sample_stem}.confidence.npy",
        "metadata": reference_dir / f"{sample_stem}.metadata.json",
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def export_corrupted_reference_dataset(
    dataset_dir: str | Path | None = None,
    objects_subset: str = "all",
    seed: int = 1234,
    motion_names: Iterable[str] | None = None,
    sample_limit: int = 0,
) -> dict[str, object]:
    dataset_root = resolve_dataset_root(dataset_dir)
    motion_dir = get_motion_dir(dataset_root)
    cond_dict = load_cond_dict(dataset_root)
    reference_dir = get_corrupted_reference_dir(dataset_root)
    reference_dir.mkdir(parents=True, exist_ok=True)

    motion_files = list_motion_files(
        dataset_dir=dataset_root,
        objects_subset=objects_subset,
        motion_names=motion_names,
        sample_limit=sample_limit,
    )
    if not motion_files:
        raise FileNotFoundError(f"No motion files matched under {motion_dir}")

    corruptor = MotionCorruptor(seed=seed)
    manifest: list[dict[str, object]] = []
    
    # Collect object types in order of appearance
    object_types_in_order = []
    for motion_file in motion_files:
        object_type = infer_object_type(motion_file, cond_dict.keys())
        if not object_types_in_order or object_types_in_order[-1] != object_type:
            object_types_in_order.append(object_type)
    
    total_object_types = len(object_types_in_order)
    object_index = {obj_type: idx for idx, obj_type in enumerate(object_types_in_order, start=1)}
    
    current_object_type = None
    object_count = {}

    for motion_file in motion_files:
        object_type = infer_object_type(motion_file, cond_dict.keys())
        
        # Print when switching to a new object type
        if current_object_type != object_type:
            if current_object_type is not None:
                print(f" ✓ Completed {current_object_type}: {object_count[current_object_type]} samples")
            current_object_type = object_type
            object_count[object_type] = 0
            idx = object_index[object_type]
            print(f"[{idx}/{total_object_types}] Processing {object_type}...", end="")
        
        object_cond = cond_dict[object_type]
        mean = np.asarray(object_cond["mean"], dtype=np.float32)
        std = np.asarray(object_cond["std"], dtype=np.float32) + 1e-6
        clean_motion = np.load(motion_dir / motion_file).astype(np.float32)
        clean_motion_norm = np.nan_to_num((clean_motion - mean[None, :]) / std[None, :]).astype(np.float32)
        reference_motion_norm, confidence, corruption_metadata = corruptor.corrupt(
            clean_motion_norm,
            length=len(clean_motion_norm),
            kinematic_chains=object_cond.get("kinematic_chains"),
        )
        reference_motion = reference_motion_norm * std[None, :] + mean[None, :]

        sample_paths = get_corrupted_sample_paths(
            motion_file=motion_file,
            dataset_dir=dataset_root,
        )
        np.save(sample_paths["reference"], reference_motion.astype(np.float32))
        np.save(sample_paths["confidence"], confidence.astype(np.float32))
        sample_metadata = {
            "motion_file": motion_file,
            "object_type": object_type,
            "clip_length": int(len(clean_motion_norm)),
            "seed": int(seed),
            "corruption_metadata": _json_safe(corruption_metadata),
        }
        with open(sample_paths["metadata"], "w", encoding="utf-8") as handle:
            json.dump(sample_metadata, handle, indent=2)
        manifest.append(sample_metadata)
        object_count[object_type] += 1
    
    # Print the last object type
    if current_object_type is not None:
        print(f" ✓ Completed {current_object_type}: {object_count[current_object_type]} samples")

    manifest_path = reference_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    
    print(f"\n✓ Exported {len(manifest)} corrupted-reference samples to {reference_dir}")
    
    return {
        "dataset_dir": str(dataset_root),
        "reference_dir": str(reference_dir),
        "manifest_path": str(manifest_path),
        "generated_samples": len(manifest),
        "seed": int(seed),
    }


def load_corrupted_reference_sample(
    motion_file: str,
    dataset_dir: str | Path | None = None,
) -> dict[str, object]:
    sample_paths = get_corrupted_sample_paths(
        motion_file=motion_file,
        dataset_dir=dataset_dir,
    )
    missing = [path for key, path in sample_paths.items() if key != "reference_dir" and not path.exists()]
    if missing:
        legacy_dir = get_legacy_corrupted_reference_dir(dataset_dir)
        legacy_paths = {
            "reference_dir": legacy_dir,
            "reference": legacy_dir / sample_paths["reference"].name,
            "confidence": legacy_dir / sample_paths["confidence"].name,
            "metadata": legacy_dir / sample_paths["metadata"].name,
        }
        legacy_missing = [path for key, path in legacy_paths.items() if key != "reference_dir" and not path.exists()]
        if not legacy_missing:
            sample_paths = legacy_paths
            missing = []
    if missing:
        raise FileNotFoundError(
            "Stored corrupted-reference sample is incomplete. "
            f"Run export_corrupted_truebones_samples.py first. Missing: {missing}"
        )
    with open(sample_paths["metadata"], "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return {
        "motion_file": motion_file,
        "reference_motion": np.load(sample_paths["reference"]).astype(np.float32),
        "soft_confidence_mask": np.load(sample_paths["confidence"]).astype(np.float32),
        "metadata": metadata,
    }