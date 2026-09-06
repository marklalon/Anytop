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

A run may be pointed at three different things, in rising order of
independence from the filesystem:

* one dataset directory (``configure(dataset_dir=...)``) -- the preprocessing
  case; the snapshot is keyed by **bare** species names, exactly as the sidecars
  are written;
* several dataset directories (``configure(sources=[...])``) -- multi-dataset
  training; the snapshot is keyed by **canonical** ``<namespace>/<species>``
  keys so two datasets may both contain a ``Horse``;
* a cond dict alone (``configure_from_cond(cond_dict)``) -- inference, where no
  dataset directory need exist; the tags come from each entry's baked
  ``species_tags``.

Lookups accept either key form in all three modes, so preprocessing code that
passes a bare species name keeps working under a multi-source configuration.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from data_loaders.truebones.truebones_utils.dataset_sources import (
    DatasetSource,
    bare_species_name,
    canonical_key,
    make_source,
)
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
# pairs (serpentine or aquatic animals).  One object per line:
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
    # No leg chains and no propulsive body wave: the root carries the whole body
    # along, whether it hovers (ghost, elemental, hover robot) or slides over the
    # ground (slime, shell, a robed figure whose hem drags).
    "drifting",
)

# Every subset key ``build_object_subsets`` is guaranteed to produce, for CLI
# ``choices=`` lists that must be built before a dataset directory is known.
OBJECT_SUBSET_CHOICES = ("all",) + CANONICAL_OBJECT_SUBSETS


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
    # Bare-species-name index over the same snapshot, so a caller holding only
    # "Horse" resolves under a multi-source (canonical-key) configuration. First
    # source wins, matching ``dataset_sources.resolve_species_key`` rule 3.
    species_tags_bare_lower: Mapping[str, tuple[str, ...]] = ()

    def tags_for(self, object_type) -> tuple[str, ...]:
        """Motion tags for a species; empty when it carries no entry (case-insensitive).

        Accepts a canonical ``<namespace>/<species>`` key or a bare species name.
        """
        if object_type is None:
            return ()
        lowered = str(object_type).strip().lower()
        tags = self.species_tags_lower.get(lowered)
        if tags is not None:
            return tags
        return (self.species_tags_bare_lower or {}).get(bare_species_name(lowered), ())

    def chain_forward_for(self, object_type) -> tuple[int, ...] | None:
        """Forward-chain joint indices for a species, or ``None``.

        These indices are tied to one dataset's collapsed-skeleton ordering, so
        they are stored bare in the ``chain_forward_joints.jsonl`` sidecar and
        never shared across datasets. ``object_type`` may be a canonical
        ``<namespace>/<species>`` key or a bare species name. In a single-source
        setup both resolve to the same bare-name sidecar entry; in a
        multi-source setup a canonical key resolves to that dataset's own entry
        while a bare name resolves to nothing when more than one dataset defines
        the same bare name, rather than to the wrong skeleton's indices.
        """
        if object_type is None:
            return None
        name = str(object_type).strip()
        # The sidecar keys are bare species names. Accept either a canonical
        # <namespace>/<species> key or a bare name by resolving to the bare name.
        bare = bare_species_name(name)
        exact = self.chain_forward_joints.get(name)
        if exact is None:
            exact = self.chain_forward_joints.get(bare)
        if exact is not None:
            return exact
        matches = [
            indices for key, indices in self.chain_forward_joints.items()
            if bare_species_name(key) == bare
        ]
        return matches[0] if len(matches) == 1 else None

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

        A selector is either a subset key ("all", "quadruped", ...) or
        a single species name ("Horse"), which under a multi-source configuration
        is resolved to its canonical key.
        """
        members = self.object_subsets.get(selector)
        if members is not None:
            return list(members)
        lowered = str(selector).strip().lower()
        for key in self.species_tags:
            if key.lower() == lowered or bare_species_name(key).lower() == lowered:
                return [key]
        return [selector]


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

    Keys are ``"all"``, the seven canonical body plans (always present, possibly
    empty), and any extra subset a dataset introduces. This is the only place
    subsets are assembled.
    """
    subsets: dict[str, list[str]] = {"all": list(species_tags.keys())}
    for object_subset in CANONICAL_OBJECT_SUBSETS:
        subsets[object_subset] = []
    for species, tags in species_tags.items():
        subsets.setdefault(tags[0].strip().lower(), []).append(species)
    return subsets


# ── Process-wide configuration ───────────────────────────────────────────────
@dataclass(frozen=True)
class _Sources:
    dataset_dir: str | None = None
    species_tags_file: str | None = None
    chain_forward_joints_file: str | None = None
    # Multi-dataset training: ``((namespace, root), ...)``. Kept as plain tuples
    # so ``worker_initargs`` stays picklable for a ProcessPoolExecutor.
    dataset_sources: tuple[tuple[str, str], ...] | None = None
    # Inference: tags come from a cond dict, not from any dataset directory.
    # ``((canonical key, (tag, ...)), ...)``.
    cond_species_tags: tuple[tuple[str, tuple[str, ...]], ...] | None = None

    @property
    def is_multi_source(self) -> bool:
        return bool(self.dataset_sources)

    @property
    def is_cond_backed(self) -> bool:
        return self.cond_species_tags is not None


_sources = _Sources()
_snapshot: DatasetTags | None = None


def _clean(value) -> str | None:
    if value is None or not str(value).strip():
        return None
    return str(value)


def _resolve(sources: _Sources) -> SidecarPaths:
    if sources.is_multi_source:
        # Reported for logging only; each source is read from its own root.
        first_root = Path(sources.dataset_sources[0][1])
        return SidecarPaths(
            species_tags=first_root / SPECIES_TAGS_FILE,
            chain_forward_joints=first_root / CHAIN_FORWARD_JOINTS_FILE,
        )
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


def configure(
    dataset_dir=None,
    species_tags_file=None,
    chain_forward_joints_file=None,
    sources=None,
) -> SidecarPaths:
    """Point the loader at a run's sidecars and drop the cached snapshot.

    Does no I/O -- the sidecars are read on the next ``dataset_tags()`` call, so
    this is safe to call before the consuming modules are imported and cheap
    enough to use as a ``ProcessPoolExecutor`` initializer.  Returns the
    resolved sidecar paths for logging (the first source's, when several are
    given).

    *sources* is a sequence of ``DatasetSource`` (or ``(namespace, root)``
    pairs).  With it the snapshot is keyed by canonical ``<namespace>/<species>``
    keys and the per-file overrides do not apply -- each source brings its own
    sidecars.
    """
    global _sources, _snapshot
    _sources = _Sources(
        dataset_dir=_clean(dataset_dir),
        species_tags_file=_clean(species_tags_file),
        chain_forward_joints_file=_clean(chain_forward_joints_file),
        dataset_sources=_normalize_source_pairs(sources),
    )
    _snapshot = None
    return _resolve(_sources)


def configure_from_cond(cond_dict) -> DatasetTags:
    """Take the tag snapshot from a cond dict's baked ``species_tags``.

    The inference contract is ``cond.npy`` alone: no dataset directory need
    exist.  ``chain_forward_joints`` stays empty because its indices are tied to
    a dataset's collapsed-skeleton ordering and are only consumed by
    preprocessing, which always runs against a real dataset directory.
    """
    global _sources, _snapshot
    baked = tuple(
        (str(key), tuple(str(tag).strip() for tag in (entry.get("species_tags") or ())))
        for key, entry in cond_dict.items()
    )
    _sources = _Sources(cond_species_tags=baked)
    _snapshot = _snapshot_from({key: tags for key, tags in baked if tags}, {})
    return _snapshot


def _normalize_source_pairs(sources) -> tuple[tuple[str, str], ...] | None:
    if not sources:
        return None
    pairs: list[tuple[str, str]] = []
    for source in sources:
        if isinstance(source, DatasetSource):
            pairs.append((source.namespace, source.root))
        else:
            namespace, root = source
            resolved = make_source(namespace, root)
            pairs.append((resolved.namespace, resolved.root))
    return tuple(pairs)


@contextmanager
def using_dataset_dir(dataset_dir):
    """Scope the process configuration to ``dataset_dir`` for the duration.

    Library functions (``create_data_samples`` and friends) wrap their work in
    this so a direct API caller working on a non-default dataset gets that
    dataset's sidecars -- without leaving the process reconfigured afterwards.
    A caller already pointed at the same single dataset (a CLI that ran
    ``configure()``, possibly with explicit file overrides) keeps its
    configuration untouched.  A multi-source or cond-backed configuration is
    always replaced for the duration: it names no single dataset directory, so
    "already pointed here" cannot hold.
    """
    global _sources, _snapshot
    previous_sources, previous_snapshot = _sources, _snapshot
    reconfigured = (
        _sources.is_multi_source
        or _sources.is_cond_backed
        or get_dataset_dir(_clean(dataset_dir)) != get_dataset_dir(_sources.dataset_dir)
    )
    if reconfigured:
        configure(dataset_dir=dataset_dir)
    try:
        yield
    finally:
        if reconfigured:
            _sources, _snapshot = previous_sources, previous_snapshot


def configured_dataset_dir() -> str | None:
    """The single dataset directory this process is pointed at, if there is one.

    ``None`` under a multi-source or cond-backed configuration, neither of which
    names one directory.  The other dataset-directory-scoped sidecars
    (``ignore_warnings.txt``) resolve through this, so the pool initializer that
    replays ``configure()`` into a worker carries them along with the tags.
    """
    if _sources.is_multi_source or _sources.is_cond_backed:
        return None
    return get_dataset_dir(_sources.dataset_dir)


def worker_initargs() -> tuple:
    """Args for ``ProcessPoolExecutor(initializer=configure, initargs=...)``.

    A sub-process starts with the default sources, so the pool must replay the
    parent's configuration before any worker touches the tag snapshot.
    """
    return (
        _sources.dataset_dir,
        _sources.species_tags_file,
        _sources.chain_forward_joints_file,
        _sources.dataset_sources,
    )


def dataset_tags() -> DatasetTags:
    """The configured dataset's tag snapshot, loading the sidecars on first use."""
    global _snapshot
    if _snapshot is None:
        _snapshot = _load(_sources)
    return _snapshot


def _load(sources: _Sources) -> DatasetTags:
    """Read both sidecars. This is the one place that decides what may be absent."""
    if sources.is_cond_backed:
        return _snapshot_from(
            {key: tags for key, tags in sources.cond_species_tags if tags}, {}
        )
    if sources.is_multi_source:
        return _load_multi_source(sources.dataset_sources)

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


def _load_multi_source(source_pairs: tuple[tuple[str, str], ...]) -> DatasetTags:
    """Merge every source's sidecars into one canonically-keyed snapshot.

    Duplicate species names across sources are expected (both datasets have a
    ``Horse``) and become distinct canonical keys; a duplicate *within* one
    source is still an error, raised by ``load_species_tags``.  Forward-chain
    joint indices are tied to their dataset's collapsed-skeleton ordering, so
    they are namespaced too and never shared between sources.
    """
    species_tags: dict[str, tuple[str, ...]] = {}
    chain_forward_joints: dict[str, tuple[int, ...]] = {}
    for namespace, root in source_pairs:
        root_path = Path(root)
        species_path = root_path / SPECIES_TAGS_FILE
        if not species_path.is_file():
            raise FileNotFoundError(
                f"Species motion tags file not found at: {species_path}\n"
                f"Every dataset source must carry its own {SPECIES_TAGS_FILE}."
            )
        for species, tags in load_species_tags(species_path).items():
            species_tags[canonical_key(namespace, species)] = tags
        chain_path = root_path / CHAIN_FORWARD_JOINTS_FILE
        if chain_path.is_file():
            for species, indices in load_chain_forward_joints(chain_path).items():
                chain_forward_joints[canonical_key(namespace, species)] = indices
    return _snapshot_from(species_tags, chain_forward_joints)


def _snapshot_from(species_tags, chain_forward_joints) -> DatasetTags:
    object_subsets = build_object_subsets(species_tags)
    bare_lower: dict[str, tuple[str, ...]] = {}
    for key, tags in species_tags.items():
        bare_lower.setdefault(bare_species_name(key).lower(), tags)
    return DatasetTags(
        species_tags=species_tags,
        chain_forward_joints=chain_forward_joints,
        object_subsets=object_subsets,
        subset_members={key: frozenset(names) for key, names in object_subsets.items()},
        species_tags_lower={key.lower(): tags for key, tags in species_tags.items()},
        species_tags_bare_lower=bare_lower,
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


def restore(snapshot: DatasetTags) -> None:
    """Reinstate a snapshot captured earlier from ``dataset_tags()``.

    For persistent workers that hand control to code which may call
    ``register_species_tags``: capture before, restore after, so one request's
    tags do not leak into the next.
    """
    global _snapshot
    _snapshot = snapshot


@contextmanager
def registered_species_tags(species: str, tags):
    """Add a species' tags for the duration of one request, then restore.

    For long-lived processes (the inference service, its persistent pool
    workers) that accept per-request ``--species-tags`` for a skeleton absent
    from the sidecar: the tags must not leak into the next request.
    """
    previous = dataset_tags()
    register_species_tags(species, tags)
    try:
        yield
    finally:
        restore(previous)


def assert_species_tags_cover(object_types) -> None:
    """Fast-fail unless every ``object_type`` has an entry in ``species_tags.jsonl``.

    The sidecar is the single source of truth for the per-species descriptor
    (``build_species_embedding_text``) and the retarget group discount, and it
    carries no fallback. Call this at preprocessing and before training so a
    newly added species without motion tags surfaces immediately rather than
    silently degrading the species condition.
    """
    tags = dataset_tags()
    missing = sorted(
        {
            str(object_type)
            for object_type in object_types
            if str(object_type).strip() and not tags.tags_for(object_type)
        }
    )
    if missing:
        raise SystemExit(
            f"\033[91m{SPECIES_TAGS_FILE} is missing tags for object_type(s): "
            f"{', '.join(missing)}. Add them in the {SPECIES_TAGS_FILE} sidecar "
            "before preprocessing or training.\033[0m"
        )
