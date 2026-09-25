"""The precomputed T5 table of every legal species descriptor.

A species descriptor is ``[body-plan, locomotion]`` drawn from two closed
vocabularies (``dataset_tags.SPECIES_BODY_PLANS`` x ``SPECIES_LOCOMOTIONS``), so
the set of descriptor texts is finite. The whole product is encoded into one
table, and ``species_emb`` never lives in a dataset's cond.npy -- a cond only
carries each species' baked ``species_tags``, and the vector is bound from a
table in memory:

* training builds (or extends) the repo-global table
  ``dataset/species_descriptor_embs.npy`` at startup -- the only place T5 runs
  for a species -- binds the cond it trains on from it, and copies it into
  ``save_dir`` next to the checkpoint and its ``cond.npy``;
* generation and ``--species_tags`` (the CLI and the AnyTop service alike)
  read only the checkpoint's copy, and refuse a combination it lacks rather
  than encoding one the weights never saw.

The rows are ``T5Conditioner``'s masked token mean of the descriptor text, the
same encoding a cond entry's ``species_emb`` always had.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from data_loaders.truebones.truebones_utils.dataset_tags import (
    parse_species_tags,
    species_descriptor_text,
    species_descriptor_vocabulary,
)
from data_loaders.truebones.truebones_utils.param_utils import (
    DEFAULT_SPECIES_DESCRIPTOR_TABLE,
    SPECIES_DESCRIPTOR_TABLE_FILE,
    _resolve_project_path,
)

# Bump when the on-disk payload layout changes.
SPECIES_DESCRIPTOR_TABLE_SCHEMA_VERSION = 1

# Descriptors per T5 forward pass when encoding the table.
_ENCODE_BATCH = 64


class SpeciesDescriptorError(ValueError):
    """A descriptor or table that cannot condition the model."""


@dataclass(frozen=True)
class SpeciesDescriptorTable:
    """``{descriptor: T5 vector}`` over a closed set of descriptors."""

    t5_name: str
    descriptors: tuple[tuple[str, ...], ...]
    embeddings: np.ndarray
    source: str = ""
    _rows: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        embeddings = np.asarray(self.embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(self.descriptors):
            raise SpeciesDescriptorError(
                f"species descriptor table {self.source or '<memory>'}: expected "
                f"({len(self.descriptors)}, D) embeddings, got {tuple(embeddings.shape)}"
            )
        rows = {}
        for index, tags in enumerate(self.descriptors):
            text = species_descriptor_text(tags)
            if text in rows:
                raise SpeciesDescriptorError(
                    f"species descriptor table {self.source or '<memory>'} holds "
                    f"{text!r} twice"
                )
            rows[text] = index
        object.__setattr__(self, "embeddings", embeddings)
        object.__setattr__(self, "_rows", rows)

    @property
    def embedding_dim(self) -> int:
        return int(self.embeddings.shape[1])

    def __contains__(self, tags) -> bool:
        return species_descriptor_text(parse_species_tags(tags)) in self._rows

    def missing(self, descriptors: Iterable) -> list[tuple[str, ...]]:
        """The descriptors in *descriptors* this table has no row for."""
        return [tuple(tags) for tags in descriptors if tags not in self]

    def lookup(self, tags, where: str) -> np.ndarray:
        """The vector for *tags* (a copy), or a ``SpeciesDescriptorError``."""
        parsed = parse_species_tags(tags)
        row = self._rows.get(species_descriptor_text(parsed))
        if row is None:
            raise SpeciesDescriptorError(
                f"{where}: species descriptor {list(parsed)!r} is not in the "
                f"precomputed species descriptor table ({self.source or '<memory>'}); "
                "only the combinations encoded before training can condition the model."
            )
        return self.embeddings[row].copy()

    def extended(self, descriptors, embeddings) -> "SpeciesDescriptorTable":
        """This table plus new rows, existing rows untouched."""
        return SpeciesDescriptorTable(
            t5_name=self.t5_name,
            descriptors=self.descriptors + tuple(tuple(tags) for tags in descriptors),
            embeddings=np.concatenate(
                [self.embeddings, np.asarray(embeddings, dtype=np.float32)], axis=0
            ),
            source=self.source,
        )

    def payload(self) -> dict:
        return {
            "schema_version": SPECIES_DESCRIPTOR_TABLE_SCHEMA_VERSION,
            "t5_name": self.t5_name,
            "descriptors": [list(tags) for tags in self.descriptors],
            "embeddings": self.embeddings,
        }


# ── Locations ────────────────────────────────────────────────────────────────
def default_species_descriptor_table_path(path=None) -> Path:
    """The repo-global table the datasets are baked from (or an explicit override)."""
    return _resolve_project_path(path or DEFAULT_SPECIES_DESCRIPTOR_TABLE)


def sibling_species_descriptor_table_path(file_path) -> Path:
    """The table next to a checkpoint or its ``cond.npy``."""
    return Path(os.path.dirname(os.path.abspath(str(file_path)))) / SPECIES_DESCRIPTOR_TABLE_FILE


# ── Disk I/O ─────────────────────────────────────────────────────────────────
def load_species_descriptor_table(path) -> SpeciesDescriptorTable:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"species descriptor table not found: {path}")
    raw = np.load(str(path), allow_pickle=True).item()
    if not isinstance(raw, dict):
        raise SpeciesDescriptorError(f"{path}: not a species descriptor table")
    version = raw.get("schema_version")
    if version != SPECIES_DESCRIPTOR_TABLE_SCHEMA_VERSION:
        raise SpeciesDescriptorError(
            f"{path}: schema {version}, this code reads "
            f"{SPECIES_DESCRIPTOR_TABLE_SCHEMA_VERSION}"
        )
    return SpeciesDescriptorTable(
        t5_name=str(raw["t5_name"]),
        descriptors=tuple(tuple(str(tag) for tag in tags) for tags in raw["descriptors"]),
        embeddings=raw["embeddings"],
        source=str(path),
    )


def save_species_descriptor_table(path, table: SpeciesDescriptorTable) -> Path:
    """Atomically write *table* (temp file + replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    with open(temp_path, "wb") as handle:
        np.save(handle, table.payload(), allow_pickle=True)
    os.replace(str(temp_path), str(path))
    return path


def load_checkpoint_species_descriptor_table(file_path, what: str) -> SpeciesDescriptorTable:
    """The table next to *file_path* (a checkpoint or its cond.npy), required."""
    path = sibling_species_descriptor_table_path(file_path)
    if not path.is_file():
        raise SpeciesDescriptorError(
            f"no {SPECIES_DESCRIPTOR_TABLE_FILE} next to {what} {file_path}; training "
            "copies it into save_dir, so this checkpoint is incomplete"
        )
    return load_species_descriptor_table(path)


# ── Encoding ─────────────────────────────────────────────────────────────────
def cond_t5_name(cond_dict) -> str:
    """The T5 model a cond's joint names were encoded with (the table must match)."""
    names = {
        str((entry.get("joints_names_embs_meta") or {}).get("t5_name") or "")
        for entry in cond_dict.values()
    }
    names.discard("")
    if len(names) != 1:
        raise SpeciesDescriptorError(
            f"cond joint names were encoded with {sorted(names) or 'no'} T5 model(s); "
            "expected exactly one"
        )
    return names.pop()


def load_t5_conditioner(t5_name: str):
    import torch

    from model.conditioners import T5Conditioner

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading T5 model {t5_name} on {device.upper()} ...")
    return T5Conditioner(
        name=t5_name,
        finetune=False,
        word_dropout=0.0,
        normalize_text=False,
        device=device,
        autocast_dtype=None,
        local_files_only=True,
    )


def encode_species_descriptors(descriptors, t5_conditioner) -> np.ndarray:
    import torch

    texts = [species_descriptor_text(tags) for tags in descriptors]
    chunks = []
    with torch.no_grad():
        for start in range(0, len(texts), _ENCODE_BATCH):
            tokens = t5_conditioner.tokenize_entries(texts[start:start + _ENCODE_BATCH])
            chunks.append(
                t5_conditioner(tokens).detach().float().cpu().numpy().astype(np.float32, copy=False)
            )
    return np.concatenate(chunks, axis=0)


def build_species_descriptor_table(
    out_path=None,
    t5_name: str = "t5-base",
    *,
    t5_conditioner=None,
) -> SpeciesDescriptorTable:
    """Make the table at *out_path* cover the whole descriptor vocabulary.

    A current table is returned as-is. A table missing some combinations (the
    vocabulary grew) is extended with just those rows, so every run sees the
    same vector for the same descriptor. A table encoded with another T5, or one
    this code cannot read, is re-encoded from scratch.
    """
    out_path = default_species_descriptor_table_path(out_path)
    vocabulary = species_descriptor_vocabulary()
    existing = None
    if out_path.is_file():
        try:
            existing = load_species_descriptor_table(out_path)
        except SpeciesDescriptorError as exc:
            print(f"[rebuild] {exc}")
        if existing is not None and existing.t5_name != t5_name:
            print(f"[rebuild] {out_path}: encoded with '{existing.t5_name}', asked for '{t5_name}'")
            existing = None

    wanted = vocabulary if existing is None else existing.missing(vocabulary)
    if existing is not None and not wanted:
        print(f"[skip] {out_path} already holds all {len(vocabulary)} species descriptors")
        return existing

    if t5_conditioner is None:
        t5_conditioner = load_t5_conditioner(t5_name)
    print(f"Encoding {len(wanted)} species descriptor(s) via T5 '{t5_name}' ...")
    embeddings = encode_species_descriptors(wanted, t5_conditioner)
    if existing is None:
        table = SpeciesDescriptorTable(
            t5_name=t5_name, descriptors=tuple(wanted), embeddings=embeddings,
            source=str(out_path),
        )
    else:
        table = existing.extended(wanted, embeddings)
    save_species_descriptor_table(out_path, table)
    print(
        f"[OK] wrote {out_path} ({len(table.descriptors)} descriptors x "
        f"{table.embedding_dim}d)"
    )
    return table


# ── Cond binding ─────────────────────────────────────────────────────────────
def _entry_species_tags(object_type, entry) -> tuple[str, ...]:
    tags = tuple(str(tag) for tag in (entry.get("species_tags") or ()))
    if not tags:
        raise SpeciesDescriptorError(
            f"cond entry '{object_type}' carries no baked species_tags; regenerate its cond.npy"
        )
    return tags


def bind_cond_species_embs(cond_dict, table: SpeciesDescriptorTable, where: str) -> None:
    """Set every entry's ``species_emb`` to its table row, by its baked ``species_tags``.

    Training binds from the table it ships with the checkpoint, generation from
    that shipped copy; an entry whose descriptor the table lacks is refused.
    A ``species_emb`` already on an entry that no table bound is a cond baked
    before species vectors left cond.npy, and is refused: regenerate it.
    """
    cond_name = cond_t5_name(cond_dict)
    if table.t5_name != cond_name:
        raise SpeciesDescriptorError(
            f"{where}: species descriptor table {table.source or '<memory>'} was encoded "
            f"with T5 {table.t5_name!r}, but the cond joint names were encoded with "
            f"{cond_name!r}"
        )
    stale, missing = [], []
    for object_type, entry in cond_dict.items():
        if "species_emb" in entry and "table" not in (entry.get("species_emb_meta") or {}):
            stale.append(str(object_type))
            continue
        tags = _entry_species_tags(object_type, entry)
        if tags not in table:
            missing.append(f"{object_type}: {list(tags)}")
            continue
        entry["species_emb"] = table.lookup(tags, where)
        entry["species_emb_meta"] = {
            "table": table.source,
            "t5_name": table.t5_name,
            "embedding_dim": table.embedding_dim,
            "embedding_text": species_descriptor_text(tags),
        }
    if stale:
        raise SpeciesDescriptorError(
            f"{where}: cond entries carry a baked species_emb, so the cond predates "
            f"the species descriptor table; regenerate it ({', '.join(stale[:5])}"
            f"{' ...' if len(stale) > 5 else ''})"
        )
    if missing:
        raise SpeciesDescriptorError(
            f"{where}: species descriptors not in the species descriptor table "
            f"{table.source}:\n  " + "\n  ".join(missing)
        )
