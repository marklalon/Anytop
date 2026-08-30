"""Reading, writing, and upgrading ``cond.npy`` (schema v4).

Schema v4 makes a cond dict self-describing so that *merging two datasets is a
plain dict union*: every entry is keyed by ``<namespace>/<species>`` and carries

``cond_schema_version``  int    -- always 4; validated on load
``dataset_namespace``    str    -- ``"truebones/zoo"``
``dataset_root``         str|None -- Anytop-root-relative POSIX path, or ``None``
                                  meaning "the directory holding this cond.npy"
                                  (what a single-dataset cond stores, so it stays
                                  portable)
``species_name``         str    -- the bare name, used to join
                                  ``motion_metadata.json`` / ``species_tags.jsonl``
                                  and to prefix ``motions/{species_name}_*.npy``
``species_tags``         tuple  -- baked copy for the inference contract; training
                                  still reads each source's ``species_tags.jsonl``

``object_type`` becomes the canonical key (matching the dict key); the bare name
lives on in ``species_name``.  No non-species top-level keys are introduced --
a great deal of code iterates ``for key in cond_dict`` assuming every key is a
species.

Legacy (pre-v4) files are upgraded **in memory** on load, so an existing dataset
keeps working before it is re-preprocessed; re-running the preprocessing pipeline
persists the same upgrade (``stamp_dataset_cond``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np

from data_loaders.truebones.truebones_utils.dataset_sources import (
    COND_FILE,
    COND_SCHEMA_VERSION,
    canonical_key,
    infer_namespace_from_root,
    normalize_namespace,
    resolve_anytop_path,
    species_lookup_map,
    split_canonical_key,
    to_anytop_relative,
)

SPECIES_TAGS_FILE = "species_tags.jsonl"


def cond_schema_version(cond_dict: Mapping[str, Mapping]) -> int:
    """Lowest schema version across entries (0 when any entry predates v4)."""
    if not cond_dict:
        return COND_SCHEMA_VERSION
    return min(int(entry.get("cond_schema_version", 0) or 0) for entry in cond_dict.values())


def is_schema_v4(cond_dict: Mapping[str, Mapping]) -> bool:
    return cond_schema_version(cond_dict) >= COND_SCHEMA_VERSION


def _read_species_tags_sidecar(dataset_root) -> dict[str, tuple[str, ...]]:
    """Best-effort read of ``species_tags.jsonl`` next to a cond file.

    Used only to bake the inference-side copy; a missing sidecar is not fatal
    here because ``dataset_tags`` still fails loudly wherever tags are required.
    """
    path = Path(dataset_root) / SPECIES_TAGS_FILE
    if not path.is_file():
        return {}
    tags: dict[str, tuple[str, ...]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        record = json.loads(text)
        species = str(record["species"]).strip()
        tags[species] = tuple(str(tag).strip() for tag in record["species_tags"])
    return tags


def upgrade_cond_dict(
    cond_dict: Mapping[str, Mapping],
    *,
    namespace: str,
    dataset_root=None,
    species_tags: Mapping[str, tuple[str, ...]] | None = None,
    store_root: bool = False,
) -> dict[str, dict]:
    """Return *cond_dict* re-keyed to canonical keys with the v4 fields filled in.

    ``store_root`` writes an explicit Anytop-relative ``dataset_root``; leave it
    off for a dataset's own cond.npy so the entry stays portable (``None`` =
    "wherever this cond.npy lives").  The merge tool sets it, since a merged cond
    sits in its own directory and must point back at each source.

    Entries already at v4 are passed through untouched, which makes this callable
    on a mixed dict and idempotent on a merged one.
    """
    namespace = normalize_namespace(namespace)
    if species_tags is None:
        species_tags = _read_species_tags_sidecar(dataset_root) if dataset_root else {}
    tags_by_lower = {name.lower(): tags for name, tags in species_tags.items()}
    stored_root = to_anytop_relative(dataset_root) if (store_root and dataset_root) else None

    upgraded: dict[str, dict] = {}
    for key, entry in cond_dict.items():
        entry = dict(entry)
        if int(entry.get("cond_schema_version", 0) or 0) >= COND_SCHEMA_VERSION:
            upgraded[str(key)] = entry
            continue
        # A pre-v4 key is the bare species name; a partially-upgraded one may
        # already be canonical.
        _, species_name = split_canonical_key(key)
        species_name = str(entry.get("species_name") or species_name).strip()
        entry["cond_schema_version"] = COND_SCHEMA_VERSION
        entry["dataset_namespace"] = namespace
        entry["dataset_root"] = stored_root
        entry["species_name"] = species_name
        entry["species_tags"] = tuple(
            entry.get("species_tags") or tags_by_lower.get(species_name.lower(), ())
        )
        new_key = canonical_key(namespace, species_name)
        entry["object_type"] = new_key
        if new_key in upgraded:
            raise ValueError(
                f"cond upgrade produced a duplicate key {new_key!r}; two entries share "
                f"the species name {species_name!r} within namespace {namespace!r}."
            )
        upgraded[new_key] = entry
    return upgraded


def normalize_cond_dict(cond_dict: Mapping[str, Mapping], cond_path=None, namespace=None) -> dict[str, dict]:
    """Bring a loaded cond dict to schema v4, upgrading legacy files in memory.

    The namespace of a legacy file is inferred from the directory holding it
    (``dataset/truebones/zoo/truebones_processed`` -> ``truebones/zoo``), which is
    exactly what the manifest states for the existing datasets.  Pass *namespace*
    explicitly wherever the caller knows better.
    """
    if is_schema_v4(cond_dict):
        return dict(cond_dict)
    root = Path(cond_path).resolve().parent if cond_path else None
    if namespace is None:
        if root is None:
            raise ValueError(
                "Cannot upgrade a pre-v4 cond dict without either a cond.npy path or an "
                "explicit namespace."
            )
        namespace = infer_namespace_from_root(root)
    return upgrade_cond_dict(cond_dict, namespace=namespace, dataset_root=root)


def load_cond(cond_path, namespace=None) -> dict[str, dict]:
    """Load ``cond.npy`` and normalize it to schema v4."""
    path = Path(resolve_anytop_path(cond_path))
    if not path.is_file():
        raise FileNotFoundError(f"Condition file was not found: {path}")
    raw = np.load(str(path), allow_pickle=True).item()
    return normalize_cond_dict(raw, cond_path=path, namespace=namespace)


def species_lookup_map_for_dataset_dir(dataset_dir) -> dict[str, str]:
    """``{filename token: canonical key}`` read from ``<dataset_dir>/cond.npy``.

    The registry every filename->object_type inference must be validated against:
    without it, ``infer_object_type_from_filename`` falls back to "everything up
    to the first underscore", which truncates a multi-token species name
    (``FEP_MagmaDemon_Attack01_1.npy`` -> ``FEP``). Returns ``{}`` only when the
    dataset has no cond yet, which is the one case where a caller has nothing to
    validate against.
    """
    cond_path = Path(dataset_dir) / COND_FILE
    if not cond_path.is_file():
        return {}
    return species_lookup_map(load_cond(cond_path))


def save_cond(cond_path, cond_dict: Mapping[str, Mapping]) -> Path:
    """Atomically write a cond dict (temp file + replace)."""
    path = Path(cond_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    # Write through an open handle: np.save() would otherwise append ".npy" to a
    # temp name that does not already end in it.
    with open(temp_path, "wb") as handle:
        np.save(handle, dict(cond_dict), allow_pickle=True)
    os.replace(str(temp_path), str(path))
    return path


def stamp_dataset_cond(cond_dict: Mapping[str, Mapping], dataset_dir) -> dict[str, dict]:
    """Bring a dataset's own cond to v4 just before it is written to disk.

    The preprocessing chain stays single-dataset and keeps building entries under
    bare species names; this is the single point where they are re-keyed and the
    v4 fields are filled in.  ``dataset_root`` is deliberately left ``None`` --
    "the directory holding this cond.npy" -- so a dataset directory stays movable.

    Every entry is re-stamped, including already-v4 ones, so an edit to
    ``species_tags.jsonl`` reaches the baked inference-side copy on the next
    write instead of going stale.
    """
    restamped = {
        key: dict(entry, cond_schema_version=0, species_tags=())
        for key, entry in cond_dict.items()
    }
    return upgrade_cond_dict(
        restamped,
        namespace=infer_namespace_from_root(dataset_dir),
        dataset_root=dataset_dir,
        species_tags=_read_species_tags_sidecar(dataset_dir),
        store_root=False,
    )


def species_tags_from_cond(cond_dict: Mapping[str, Mapping]) -> dict[str, tuple[str, ...]]:
    """``{canonical key: species_tags}`` from the baked inference-side copies."""
    return {
        str(key): tuple(entry.get("species_tags") or ())
        for key, entry in cond_dict.items()
    }
