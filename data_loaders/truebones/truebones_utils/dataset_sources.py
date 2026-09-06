"""Dataset namespaces, canonical species keys, and the multi-dataset source list.

Two contracts run through this module:

* **Training contract** -- a ``cond.npy`` plus the dataset directories its
  entries point at (``motions/``, ``motion_metadata.json``, the tag sidecars,
  the split manifests).
* **Inference contract** -- the ``cond.npy`` alone, self-sufficient because
  every entry carries its own baked ``species_tags``.

``cond.npy`` is the single entry point for both; a *dataset manifest*
(``datasets.jsonl``) is read only by the offline merge tool and by ``eval``.

Everything here is pure path/key bookkeeping with no dataset I/O beyond the
manifest, so ``dataset_tags`` can import it without a cycle.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from data_loaders.truebones.truebones_utils.param_utils import (
    MOTION_DIR,
    _ANYTOP_ROOT,
    _resolve_project_path,
)

# cond.npy schema version. v4 re-keys every entry to ``<namespace>/<species>``
# and adds dataset_namespace / dataset_root / species_name / species_tags, so a
# merged cond is a plain union of single-dataset conds.
COND_SCHEMA_VERSION = 4

COND_FILE = "cond.npy"

# A namespace is one or more ``[A-Za-z0-9_-]`` segments joined by ``/``.
_NAMESPACE_RE = re.compile(r"^[A-Za-z0-9_-]+(/[A-Za-z0-9_-]+)*$")
_NAMESPACE_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_-]+")

# Separator between a bare species name and its namespace in a file token.
# ``/`` cannot appear in a filename, so ``Horse@truebones_zoo_upgrade`` is the
# disambiguated form (see ``species_file_token``).
FILE_TOKEN_SEPARATOR = "@"


# -- Paths -------------------------------------------------------------------
def anytop_root() -> Path:
    """Absolute path of the ``Anytop`` package root."""
    return _ANYTOP_ROOT


def resolve_anytop_path(path_value) -> Path:
    """Resolve an Anytop-root-relative (or absolute) path to an absolute one."""
    return _resolve_project_path(path_value)


def to_anytop_relative(path_value) -> str:
    """Portable storage form of *path_value*: Anytop-root-relative POSIX.

    Paths outside the Anytop tree (or on another Windows drive) stay absolute --
    they cannot be made relative, and silently rewriting them would point at the
    wrong place on the next machine.
    """
    absolute = Path(path_value).resolve()
    try:
        relative = absolute.relative_to(anytop_root())
    except ValueError:
        return absolute.as_posix()
    return relative.as_posix()


# -- Namespaces and canonical keys -------------------------------------------
def normalize_namespace(namespace) -> str:
    """Validate a namespace and return it stripped of surrounding whitespace."""
    text = str(namespace or "").strip().strip("/")
    if not _NAMESPACE_RE.match(text):
        raise ValueError(
            f"Invalid dataset namespace {namespace!r}: expected one or more "
            "'[A-Za-z0-9_-]' segments joined by '/', e.g. 'truebones/zoo'."
        )
    return text


def sanitize_namespace_segment(segment) -> str:
    cleaned = _NAMESPACE_SANITIZE_RE.sub("_", str(segment).strip()).strip("_")
    return cleaned or "dataset"


def infer_namespace_from_root(dataset_root) -> str:
    """Best-effort namespace for a dataset directory that carries none.

    ``dataset/truebones/zoo/truebones_processed`` -> ``truebones/zoo``: the path
    under ``Anytop/dataset`` minus the processed-output leaf.  Only a fallback
    for legacy (pre-v4) cond files -- a manifest always states the namespace
    explicitly.
    """
    root = Path(dataset_root).resolve()
    try:
        relative_parts = list(root.relative_to(anytop_root() / "dataset").parts)
    except ValueError:
        relative_parts = []
    if len(relative_parts) > 1:
        relative_parts = relative_parts[:-1]
    if not relative_parts:
        relative_parts = [root.name]
    return "/".join(sanitize_namespace_segment(part) for part in relative_parts)


def canonical_key(namespace, species_name) -> str:
    """``<namespace>/<species>`` -- the one key form used by cond dicts."""
    return f"{normalize_namespace(namespace)}/{str(species_name).strip()}"


def split_canonical_key(key) -> tuple[str, str]:
    """``('truebones/zoo', 'Horse')``; a bare key yields an empty namespace."""
    text = str(key)
    namespace, separator, species = text.rpartition("/")
    if not separator:
        return "", text
    return namespace, species


def bare_species_name(key) -> str:
    return split_canonical_key(key)[1]


def assert_namespaces_disjoint(namespaces: Iterable[str]) -> None:
    """Fast-fail when one namespace is a path prefix of another (decision D5).

    Suffix resolution (``zoo/Horse``) is only unambiguous while no namespace
    nests inside another, so ``truebones`` and ``truebones/zoo`` cannot coexist.
    """
    seen: list[str] = []
    for namespace in namespaces:
        normalized = normalize_namespace(namespace)
        if normalized in seen:
            raise ValueError(f"Duplicate dataset namespace: {normalized!r}")
        for other in seen:
            shorter, longer = sorted((normalized, other), key=len)
            if longer.startswith(shorter + "/"):
                raise ValueError(
                    f"Dataset namespace {shorter!r} is a path prefix of {longer!r}; "
                    "suffix matching would be ambiguous. Rename one of them."
                )
        seen.append(normalized)


# -- Dataset sources ---------------------------------------------------------
@dataclass(frozen=True)
class DatasetSource:
    """One dataset directory participating in a training/eval run."""

    namespace: str
    root: str
    species_include: tuple[str, ...] = field(default=())
    species_exclude: tuple[str, ...] = field(default=())

    @property
    def root_path(self) -> Path:
        return Path(self.root)

    @property
    def motion_dir(self) -> str:
        return str(self.root_path / MOTION_DIR)

    @property
    def cond_path(self) -> str:
        return str(self.root_path / COND_FILE)

    @property
    def portable_root(self) -> str:
        return to_anytop_relative(self.root)

    def accepts(self, species_name) -> bool:
        name = str(species_name).strip()
        if self.species_include and name not in self.species_include:
            return False
        if name in self.species_exclude:
            return False
        return True

    def key_for(self, species_name) -> str:
        return canonical_key(self.namespace, species_name)


def make_source(namespace, root, species_include=(), species_exclude=()) -> DatasetSource:
    return DatasetSource(
        namespace=normalize_namespace(namespace),
        root=str(resolve_anytop_path(root)),
        species_include=tuple(str(name).strip() for name in species_include or ()),
        species_exclude=tuple(str(name).strip() for name in species_exclude or ()),
    )


def load_datasets_manifest(manifest_path) -> list[DatasetSource]:
    """Read ``datasets.jsonl`` into an ordered ``DatasetSource`` list.

    Line order is the bare-name resolution priority (see ``resolve_species_key``),
    so it is preserved verbatim.  Disabled lines are dropped after validation, so
    a typo in a disabled row still surfaces.
    """
    path = Path(resolve_anytop_path(manifest_path))
    if not path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {path}")

    sources: list[DatasetSource] = []
    enabled_flags: list[bool] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            record = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{line_no} is not valid JSON: {exc}") from exc
        if "namespace" not in record or "path" not in record:
            raise ValueError(f"{path.name}:{line_no} must define both 'namespace' and 'path'.")
        try:
            source = make_source(
                record["namespace"],
                record["path"],
                species_include=record.get("species_include") or (),
                species_exclude=record.get("species_exclude") or (),
            )
        except ValueError as exc:
            raise ValueError(f"{path.name}:{line_no}: {exc}") from exc
        sources.append(source)
        enabled_flags.append(bool(record.get("enabled", True)))

    assert_namespaces_disjoint(source.namespace for source in sources)

    enabled = [source for source, is_enabled in zip(sources, enabled_flags) if is_enabled]
    if not enabled:
        raise ValueError(f"{path} lists no enabled datasets.")
    return enabled


def sources_from_cond(cond_dict: Mapping[str, Mapping], cond_path=None) -> tuple[DatasetSource, ...]:
    """Derive the training sources from a v4 cond dict.

    Entries are grouped by ``dataset_namespace``; a ``dataset_root`` of ``None``
    means "the directory holding this cond.npy", which is what a single-dataset
    cond stores so it stays portable.
    """
    fallback_root = str(Path(cond_path).resolve().parent) if cond_path else None
    ordered: dict[str, DatasetSource] = {}
    for entry in cond_dict.values():
        namespace = normalize_namespace(entry["dataset_namespace"])
        stored_root = entry.get("dataset_root")
        root = str(resolve_anytop_path(stored_root)) if stored_root else fallback_root
        if root is None:
            raise ValueError(
                f"cond entry for namespace {namespace!r} has dataset_root=None and no "
                "cond.npy path was supplied to resolve it against."
            )
        existing = ordered.get(namespace)
        if existing is None:
            ordered[namespace] = DatasetSource(namespace=namespace, root=root)
        elif existing.root != root:
            raise ValueError(
                f"cond namespace {namespace!r} maps to two dataset roots: "
                f"{existing.root!r} and {root!r}."
            )
    if not ordered:
        raise ValueError("cond dict is empty; cannot derive dataset sources.")
    assert_namespaces_disjoint(ordered)
    return tuple(ordered.values())


# -- Species key resolution (CLI input -> canonical key) ---------------------
def resolve_species_key(cond_dict: Mapping[str, object], user_input) -> str | None:
    """Resolve user-facing species text to a canonical cond key.

    Tried in order, case-sensitively first and then case-insensitively (the
    dataset carries real ``Rhino``/``rhino`` and ``Scorpion``/``scorpion``
    pairs, so an exact hit must win before any folded one):

    1. the canonical key itself -- ``truebones/zoo/Horse``
    2. a namespace suffix -- ``zoo/Horse``; accepted only when unique
    3. the bare species name -- ``Horse``, taking the first cond entry in
       insertion order (i.e. the first dataset in the manifest)

    A filename token (``Horse@truebones_zoo_upgrade``) is translated back to its
    canonical form first, so anything read off a generated filename resolves the
    same way user input does.

    Returns ``None`` when nothing matches; raises when a suffix is ambiguous.
    """
    if user_input is None:
        return None
    needle = str(user_input).strip().strip("/")
    if not needle:
        return None
    keys = list(cond_dict.keys())
    if FILE_TOKEN_SEPARATOR in needle:
        # A namespace may itself contain '_', so the token is matched by
        # re-deriving each key's token rather than by substituting separators.
        species, _, namespace_token = needle.partition(FILE_TOKEN_SEPARATOR)
        for key in keys:
            namespace, key_species = split_canonical_key(key)
            if key_species == species and namespace.replace("/", "_") == namespace_token:
                return str(key)
        return None
    for fold in (False, True):
        match = _resolve_species_key_round(keys, needle, fold=fold)
        if match is not None:
            return match
    return None


def _resolve_species_key_round(keys: Sequence[str], needle: str, *, fold: bool) -> str | None:
    def norm(text: str) -> str:
        return text.lower() if fold else text

    target = norm(needle)

    for key in keys:
        if norm(str(key)) == target:
            return str(key)

    # Suffix matching applies only to a namespace-qualified selector
    # ('zoo/Horse'). A bare name deliberately falls through to the next round,
    # where the first dataset wins rather than the request being rejected.
    if "/" in needle:
        suffix_hits = [key for key in keys if norm(str(key)).endswith("/" + target)]
        if len(suffix_hits) == 1:
            return str(suffix_hits[0])
        if len(suffix_hits) > 1:
            raise ValueError(
                f"Species selector {needle!r} is ambiguous; it matches "
                f"{', '.join(sorted(str(hit) for hit in suffix_hits))}. "
                "Qualify it with the full namespace."
            )

    for key in keys:
        if norm(bare_species_name(key)) == target:
            return str(key)
    return None


def require_species_key(cond_dict: Mapping[str, object], user_input) -> str:
    key = resolve_species_key(cond_dict, user_input)
    if key is None:
        available = ", ".join(sorted(str(k) for k in cond_dict))
        raise KeyError(f"Species {user_input!r} not found in cond. Available: {available}")
    return key


# -- Filename tokens (canonical key -> filename-safe token and back) ---------
def build_species_file_tokens(cond_dict: Mapping[str, object]) -> dict[str, str]:
    """``{canonical key: filename token}``.

    A species whose bare name is unique across the whole cond keeps that bare
    name, so a single-dataset run produces exactly the filenames it always has.
    Collisions get the namespace appended: ``Horse@truebones_zoo_upgrade``.
    """
    bare_counts: dict[str, int] = {}
    for key in cond_dict:
        species = bare_species_name(key)
        bare_counts[species] = bare_counts.get(species, 0) + 1

    tokens: dict[str, str] = {}
    for key in cond_dict:
        namespace, species = split_canonical_key(key)
        if bare_counts[species] == 1 or not namespace:
            tokens[str(key)] = species
        else:
            tokens[str(key)] = f"{species}{FILE_TOKEN_SEPARATOR}{namespace.replace('/', '_')}"
    return tokens


def species_file_token(cond_dict: Mapping[str, object], key) -> str:
    canonical = str(key)
    tokens = build_species_file_tokens(cond_dict)
    if canonical not in tokens:
        raise KeyError(f"Species key {canonical!r} is not present in the cond dict.")
    return tokens[canonical]


def species_lookup_map(cond_dict: Mapping[str, object]) -> dict[str, str]:
    """``{filename token or bare name: canonical key}`` for filename inference.

    Both the disambiguated token and the bare name are registered; the bare name
    resolves to the first cond entry that carries it, matching rule 3 of
    ``resolve_species_key`` so CLI input and filename inference agree.
    """
    lookup: dict[str, str] = {}
    for key, token in build_species_file_tokens(cond_dict).items():
        lookup.setdefault(token, key)
    for key in cond_dict:
        lookup.setdefault(bare_species_name(key), str(key))
        lookup.setdefault(str(key), str(key))
    return lookup
