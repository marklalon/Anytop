from __future__ import annotations

from pathlib import Path
from typing import Iterable

from data_loaders.truebones.truebones_utils.cond_schema import load_cond
from data_loaders.truebones.truebones_utils.dataset_sources import (
    COND_FILE,
    DatasetSource,
    infer_namespace_from_root,
    load_datasets_manifest,
    make_source,
)
from data_loaders.truebones.truebones_utils.motion_labels import load_motion_metadata
from data_loaders.truebones.truebones_utils.param_utils import (
    MOTION_DIR,
    get_dataset_dir,
)
from data_loaders.truebones.truebones_utils.dataset_tags import dataset_tags

MANIFEST_SUFFIX = ".jsonl"


def resolve_dataset_root(dataset_dir: str | Path | None = None) -> Path:
    return Path(dataset_dir or get_dataset_dir()).resolve()


def get_motion_dir(dataset_dir: str | Path | None = None) -> Path:
    return resolve_dataset_root(dataset_dir) / MOTION_DIR


def resolve_sources(dataset_root=None) -> tuple[DatasetSource, ...]:
    """Normalize the many things ``dataset_root`` may be into a source list.

    Accepted forms, so ``--dataset_root`` keeps working exactly as before while
    gaining multi-dataset support:

    * ``None``            -- the default processed dataset directory
    * a directory path    -- one source (current behaviour)
    * a ``.jsonl`` path   -- a dataset manifest naming several sources
    * a ``DatasetSource`` or a sequence of them -- already resolved (this is what
      the training loop hands the scorer)
    """
    if isinstance(dataset_root, DatasetSource):
        return (dataset_root,)
    if isinstance(dataset_root, (list, tuple)) and dataset_root:
        if all(isinstance(item, DatasetSource) for item in dataset_root):
            return tuple(dataset_root)
        raise TypeError(f"Unsupported dataset_root sequence: {dataset_root!r}")

    if dataset_root is not None and str(dataset_root).lower().endswith(MANIFEST_SUFFIX):
        return tuple(load_datasets_manifest(dataset_root))

    root = resolve_dataset_root(dataset_root)
    return (make_source(infer_namespace_from_root(root), root),)


def load_cond_dict(dataset_dir: str | Path | None = None) -> dict[str, dict]:
    """Canonically-keyed cond entries for every source, unioned in source order.

    Reference scoring only reads skeleton metadata, so the sources' conds are
    merged as-is; the per-object_subset normalization statistics are *not*
    recomputed here (that belongs to ``tools/merge_dataset_cond.py``, which is
    the only place that owns a merged training cond).
    """
    merged: dict[str, dict] = {}
    for source in resolve_sources(dataset_dir):
        cond_path = Path(source.root) / COND_FILE
        if not cond_path.exists():
            raise FileNotFoundError(f"Condition file was not found: {cond_path}")
        for key, entry in load_cond(cond_path, namespace=source.namespace).items():
            if not source.accepts(entry["species_name"]):
                continue
            merged[key] = entry
    if not merged:
        raise RuntimeError("No cond entries found for the requested dataset source(s).")
    return merged


def list_motion_files(
    dataset_dir: str | Path | None = None,
    objects_subset: str = "all",
    motion_names: Iterable[str] | None = None,
    sample_limit: int = 0,
) -> list[str]:
    """Absolute paths of the reference clips selected from every source.

    Membership comes from ``motion_metadata.json`` rather than a filename
    prefix: after merging, ``Horse_Idle_1.npy`` exists in two datasets and a
    prefix test would claim it for both.
    """
    sources = resolve_sources(dataset_dir)
    requested = None if motion_names is None else {str(name) for name in motion_names}
    # The tag snapshot is keyed by bare names under a single-dataset
    # configuration and by canonical keys under a multi-source one, so accept
    # either form here.
    allowed_objects = (
        None if requested is not None else set(dataset_tags().species_for(objects_subset))
    )

    selected: list[str] = []
    for source in sources:
        motion_dir = Path(source.motion_dir)
        if not motion_dir.exists():
            raise FileNotFoundError(f"Motion directory was not found: {motion_dir}")
        metadata = load_motion_metadata(source.root)
        for path in sorted(motion_dir.glob("*.npy")):
            name = path.name
            if requested is not None:
                if name not in requested and str(path) not in requested:
                    continue
            else:
                entry = metadata.get(name)
                species = str((entry or {}).get("object_type") or "").strip()
                if not species:
                    continue
                if species not in allowed_objects and source.key_for(species) not in allowed_objects:
                    continue
            selected.append(str(path))
    if sample_limit > 0:
        selected = selected[:sample_limit]
    return selected
