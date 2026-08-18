"""Dataset-driven species tags and forward-chain overrides.

Single owner of the sidecar-backed dataset metadata (``species_tags.jsonl`` and
``chain_forward_joints.jsonl``).  Three rules keep this simple:

1. **Nothing is read at import time.**  ``configure()`` only records where the
   sidecars live; the first ``dataset_tags()`` call loads them.  A run may
   therefore point at its own dataset before any consumer touches the data,
   regardless of module import order.
2. **The loaded state is one immutable snapshot.**  Every derived view
   (case-insensitive lookup, object subsets, winged/aquatic sets) is computed
   on that snapshot, so invalidation is a single assignment and no consumer
   needs its own cache.
3. **Consumers call ``dataset_tags()`` at use time**, never
   ``from ... import SOME_DICT`` at import time -- a value copied at import
   cannot see a later ``configure()``.

Sub-processes start with the default sources, so pass ``configure`` as the
``ProcessPoolExecutor`` initializer (see ``dataset_pipeline.create_data_samples``).
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from data_loaders.truebones.truebones_utils.param_utils import (
    _resolve_project_path,
    get_dataset_dir,
)

# Per-species motion descriptor (body-plan, size/build, locomotion). One object
# per line:  {"species": "Cat", "species_tags": ["Quadruped", "Small", "Stalking"]}
# Single source of truth for the species condition and the object subsets; the
# species->tags mapping is never duplicated in code.
SPECIES_TAGS_FILE = "species_tags.jsonl"

# Dataset-specific forward-chain overrides, for creatures without usable limb
# pairs (snakes, fish).  One object per line:
#   {"species": "Pirrana", "chain_forward_joints": [10, 3, 4]}
# A 2-tuple (neck, head) means ``head - neck``; a 3-tuple (base, neck, head)
# means ``(head - neck) + (neck - base)``.  The indices are tied to a particular
# dataset's collapsed-skeleton ordering, so they must not live in source code.
CHAIN_FORWARD_JOINTS_FILE = "chain_forward_joints.jsonl"

# Canonical object_subset keys (the lower-cased first species tag). Always
# present in ``object_subsets`` even when a dataset has no member species, so
# callers and ``--object_subsets`` choices stay stable across datasets.
CANONICAL_OBJECT_SUBSETS = (
    "quadruped",
    "biped",
    "multiped",
    "serpentine",
    "aquatic",
    "winged",
)

# Every subset key ``build_object_subsets`` is guaranteed to produce, for CLI
# ``choices=`` lists that must be built before a dataset directory is known.
OBJECT_SUBSET_CHOICES = ("all",) + CANONICAL_OBJECT_SUBSETS + ("podata",)


@dataclass(frozen=True)
class SidecarPaths:
    """Resolved sidecar locations for the configured run."""

    species_tags: Path
    chain_forward_joints: Path


@dataclass(frozen=True)
class DatasetTags:
    """Immutable snapshot of one dataset's tag sidecars and all derived views."""

    species_tags: Mapping[str, tuple[str, ...]]
    chain_forward_joints: Mapping[str, tuple[int, ...]]
    object_subsets: Mapping[str, list[str]]
    subset_members: Mapping[str, frozenset]
    species_tags_lower: Mapping[str, tuple[str, ...]]

    def tags_for(self, object_type) -> tuple[str, ...]:
        """Motion tags for a species; empty when it carries no entry (case-insensitive)."""
        if object_type is None:
            return ()
        return self.species_tags_lower.get(str(object_type).strip().lower(), ())

    def object_subset_for(self, object_type) -> str | None:
        """``object_subset`` key for a species -- its lower-cased first motion tag.

        The per-object_subset canonical standardization statistics are bucketed on
        this value, so held-out species inherit the stats of their object_subset.
        Returns ``None`` when the species carries no tags entry.
        """
        tags = self.tags_for(object_type)
        return tags[0].strip().lower() if tags else None

    def species_for(self, selector) -> list[str]:
        """Species named by an ``--object_subsets`` selector.

        A selector is either a subset key ("all", "quadruped", "podata", ...) or
        a single species name ("Horse").
        """
        return list(self.object_subsets.get(selector, [selector]))


# ── Sidecar loading ──────────────────────────────────────────────────────────
def load_species_tags(path: Path) -> dict[str, tuple[str, ...]]:
    """Parse ``species_tags.jsonl`` into an insertion-ordered ``{species: tags}``.

    There is no in-code fallback, so a missing or malformed file fails loudly
    rather than silently degrading the species condition.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Species motion tags file not found at: {path}\n"
            f"It is the single source of truth for species tags and object subsets."
        )
    species_tags: dict[str, tuple[str, ...]] = {}
    for line_no, record in _iter_jsonl(path):
        species = str(record["species"]).strip()
        tags = tuple(str(tag).strip() for tag in record["species_tags"])
        if not species or not tags:
            raise ValueError(
                f"{path.name}:{line_no} has an empty species or species_tags."
            )
        if species in species_tags:
            raise ValueError(f"{path.name}:{line_no} duplicates species '{species}'.")
        species_tags[species] = tags
    return species_tags


def load_chain_forward_joints(path: Path) -> dict[str, tuple[int, ...]]:
    """Parse ``chain_forward_joints.jsonl`` into ``{species: joint indices}``."""
    chain_forward_joints: dict[str, tuple[int, ...]] = {}
    for line_no, record in _iter_jsonl(path):
        species = str(record["species"]).strip()
        raw_indices = record["chain_forward_joints"]
        if not species:
            raise ValueError(f"{path.name}:{line_no} has an empty species.")
        if not isinstance(raw_indices, (list, tuple)) or len(raw_indices) not in (2, 3):
            raise ValueError(
                f"{path.name}:{line_no} chain_forward_joints must contain 2 or 3 joint indices."
            )
        try:
            indices = tuple(int(index) for index in raw_indices)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{path.name}:{line_no} contains non-integer chain_forward_joints."
            ) from exc
        if any(index < 0 for index in indices):
            raise ValueError(
                f"{path.name}:{line_no} contains a negative chain_forward_joints index."
            )
        if species in chain_forward_joints:
            raise ValueError(f"{path.name}:{line_no} duplicates species '{species}'.")
        chain_forward_joints[species] = indices
    return chain_forward_joints


def _iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if line:
                yield line_no, json.loads(line)


def build_object_subsets(species_tags: Mapping[str, tuple[str, ...]]) -> dict[str, list[str]]:
    """Group species by object_subset (the lower-cased first motion tag).

    Keys are ``"all"``, the six canonical body plans (always present, possibly
    empty), any extra subset a dataset introduces, and the ``"podata"``
    composite -- all footed creatures (有足动物), i.e. everything but serpentine
    and aquatic.  This is the only place subsets are assembled.
    """
    subsets: dict[str, list[str]] = {"all": list(species_tags.keys())}
    for object_subset in CANONICAL_OBJECT_SUBSETS:
        subsets[object_subset] = []
    for species, tags in species_tags.items():
        subsets.setdefault(tags[0].strip().lower(), []).append(species)
    subsets["podata"] = (
        subsets["quadruped"] + subsets["biped"] + subsets["multiped"] + subsets["winged"]
    )
    return subsets


# ── Process-wide configuration ───────────────────────────────────────────────
@dataclass(frozen=True)
class _Sources:
    dataset_dir: str | None = None
    species_tags_file: str | None = None
    chain_forward_joints_file: str | None = None


_sources = _Sources()
_snapshot: DatasetTags | None = None


def _clean(value) -> str | None:
    if value is None or not str(value).strip():
        return None
    return str(value)


def _resolve(sources: _Sources) -> SidecarPaths:
    dataset_dir = Path(get_dataset_dir(sources.dataset_dir))
    return SidecarPaths(
        species_tags=(
            _resolve_project_path(sources.species_tags_file)
            if sources.species_tags_file
            else dataset_dir / SPECIES_TAGS_FILE
        ),
        chain_forward_joints=(
            _resolve_project_path(sources.chain_forward_joints_file)
            if sources.chain_forward_joints_file
            else dataset_dir / CHAIN_FORWARD_JOINTS_FILE
        ),
    )


def configure(dataset_dir=None, species_tags_file=None, chain_forward_joints_file=None) -> SidecarPaths:
    """Point the loader at a run's sidecars and drop the cached snapshot.

    Does no I/O -- the sidecars are read on the next ``dataset_tags()`` call, so
    this is safe to call before the consuming modules are imported and cheap
    enough to use as a ``ProcessPoolExecutor`` initializer.  Returns the
    resolved sidecar paths for logging.
    """
    global _sources, _snapshot
    _sources = _Sources(
        dataset_dir=_clean(dataset_dir),
        species_tags_file=_clean(species_tags_file),
        chain_forward_joints_file=_clean(chain_forward_joints_file),
    )
    _snapshot = None
    return _resolve(_sources)


@contextmanager
def using_dataset_dir(dataset_dir):
    """Scope the process configuration to ``dataset_dir`` for the duration.

    Library functions (``create_data_samples`` and friends) wrap their work in
    this so a direct API caller working on a non-default dataset gets that
    dataset's sidecars -- without leaving the process reconfigured afterwards.
    A caller already pointed at the same dataset (a CLI that ran
    ``configure()``, possibly with explicit file overrides) keeps its
    configuration untouched.
    """
    global _sources, _snapshot
    previous_sources, previous_snapshot = _sources, _snapshot
    reconfigured = get_dataset_dir(_clean(dataset_dir)) != get_dataset_dir(_sources.dataset_dir)
    if reconfigured:
        configure(dataset_dir=dataset_dir)
    try:
        yield
    finally:
        if reconfigured:
            _sources, _snapshot = previous_sources, previous_snapshot


def worker_initargs() -> tuple:
    """Args for ``ProcessPoolExecutor(initializer=configure, initargs=...)``.

    A sub-process starts with the default sources, so the pool must replay the
    parent's configuration before any worker touches the tag snapshot.
    """
    return (
        _sources.dataset_dir,
        _sources.species_tags_file,
        _sources.chain_forward_joints_file,
    )


def dataset_tags() -> DatasetTags:
    """The configured dataset's tag snapshot, loading the sidecars on first use."""
    global _snapshot
    if _snapshot is None:
        _snapshot = _load(_sources)
    return _snapshot


def _load(sources: _Sources) -> DatasetTags:
    """Read both sidecars. This is the one place that decides what may be absent."""
    paths = _resolve(sources)

    # Species tags have no in-code fallback: a missing file fails loudly rather
    # than silently degrading the species condition.
    if not paths.species_tags.is_file():
        raise FileNotFoundError(
            f"Species motion tags file not found at: {paths.species_tags}\n"
            "It is the single source of truth for species tags and object subsets."
        )
    # Most datasets need no index-based forward override and fall through to
    # semantic head/limb detection -- but a file named explicitly on the command
    # line and not found is a typo, not a dataset without overrides.
    if not paths.chain_forward_joints.is_file() and sources.chain_forward_joints_file:
        raise FileNotFoundError(
            f"Chain forward-joints file not found at: {paths.chain_forward_joints}"
        )

    return _snapshot_from(
        load_species_tags(paths.species_tags),
        load_chain_forward_joints(paths.chain_forward_joints)
        if paths.chain_forward_joints.is_file()
        else {},
    )


def _snapshot_from(species_tags, chain_forward_joints) -> DatasetTags:
    object_subsets = build_object_subsets(species_tags)
    return DatasetTags(
        species_tags=species_tags,
        chain_forward_joints=chain_forward_joints,
        object_subsets=object_subsets,
        subset_members={key: frozenset(names) for key, names in object_subsets.items()},
        species_tags_lower={key.lower(): tags for key, tags in species_tags.items()},
    )


def register_species_tags(species: str, tags) -> DatasetTags:
    """Add an unregistered species' tags for this process only.

    For entry points that process a brand-new skeleton before its tags are
    written to the sidecar (``tools/process_new_skeleton.py --species-tags``).
    Rebuilds the snapshot so every derived view stays consistent.
    """
    current = dataset_tags()
    species_tags = dict(current.species_tags)
    species_tags[str(species).strip()] = tuple(str(tag).strip() for tag in tags)
    global _snapshot
    _snapshot = _snapshot_from(species_tags, current.chain_forward_joints)
    return _snapshot


def assert_species_tags_cover(object_types) -> None:
    """Fast-fail unless every ``object_type`` has an entry in ``species_tags.jsonl``.

    The sidecar is the single source of truth for the per-species descriptor
    (``build_species_embedding_text``) and the retarget group discount, and it
    carries no fallback. Call this at preprocessing and before training so a
    newly added species without motion tags surfaces immediately rather than
    silently degrading the species condition.
    """
    tags_lower = dataset_tags().species_tags_lower
    missing = sorted(
        {
            str(object_type)
            for object_type in object_types
            if str(object_type).strip()
            and str(object_type).strip().lower() not in tags_lower
        }
    )
    if missing:
        raise SystemExit(
            f"\033[91m{SPECIES_TAGS_FILE} is missing tags for object_type(s): "
            f"{', '.join(missing)}. Add them in the {SPECIES_TAGS_FILE} sidecar "
            "before preprocessing or training.\033[0m"
        )
