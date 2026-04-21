from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from data_loaders.truebones.truebones_utils.motion_labels import load_motion_metadata
from data_loaders.truebones.truebones_utils.param_utils import (
    MOTION_DIR,
    OBJECT_SUBSETS_DICT,
    get_dataset_dir,
)


def resolve_dataset_root(dataset_dir: str | Path | None = None) -> Path:
    return Path(dataset_dir or get_dataset_dir()).resolve()


def get_motion_dir(dataset_dir: str | Path | None = None) -> Path:
    return resolve_dataset_root(dataset_dir) / MOTION_DIR


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
        if objects_subset in OBJECT_SUBSETS_DICT:
            allowed_objects = set(OBJECT_SUBSETS_DICT[objects_subset])
        else:
            # Treat as a single object type name (e.g. "Horse")
            allowed_objects = {objects_subset}
        selected = [name for name in all_motion_files if _matches_object_subset(name, allowed_objects)]
    if sample_limit > 0:
        selected = selected[:sample_limit]
    return selected
