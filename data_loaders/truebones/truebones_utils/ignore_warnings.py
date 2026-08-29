"""The per-dataset ``ignore_warnings.txt`` sidecar and its one parser.

Blank lines are skipped; every other line is one entry of three kinds:

* ``!directive`` -- a switch that applies to the whole dataset.  The only one
  today is ``!skip-orientation-detection`` (see below).  Separators are free:
  ``!skip_orientation_detection`` is the same switch.
* ``#pattern``  -- a case-insensitive substring.  Any warning whose text
  contains it is dropped from a run's warning summary.
* ``name``      -- a motion stem (``Horse_Run_1``, with or without ``.npy``)
  whose per-clip validation warnings are ignored.

``!skip-orientation-detection`` declares that the dataset's rest poses already
face the canonical +Z.  Preprocessing then leaves ``orientation_quat`` at
identity instead of estimating each character's facing from limb pairs and
head/neck references, and the estimator's fallback warnings are silenced --
they describe a computation nothing consumes any more.

The sidecar is read from the dataset directory or, when that has none, from its
parent: a processed dataset nested inside a per-dataset folder
(``dataset/unitybundles/processed``) keeps its switches next to the dataset
family (``dataset/unitybundles/ignore_warnings.txt``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from data_loaders.truebones.truebones_utils.param_utils import get_dataset_dir

FILENAME = "ignore_warnings.txt"

# Skip the automatic T-pose face-orientation detection and its correction: the
# dataset's rest poses are already canonically oriented.
SKIP_ORIENTATION_DETECTION = "skip-orientation-detection"

KNOWN_DIRECTIVES = frozenset({SKIP_ORIENTATION_DETECTION})


@dataclass(frozen=True)
class IgnoreWarnings:
    """One parsed sidecar. Every field is empty when no file was found."""

    path: Path | None = None
    stems: frozenset[str] = frozenset()
    patterns: tuple[str, ...] = ()
    directives: frozenset[str] = frozenset()
    # Directive lines that name no known switch. Reported by the callers that
    # print the sidecar's state, so a typo does not silently do nothing.
    unknown_directives: tuple[str, ...] = ()

    def has(self, directive: str) -> bool:
        return _normalize_directive(directive) in self.directives


_EMPTY = IgnoreWarnings()

# Parsed sidecars, keyed by path and invalidated by the file's own stat, so a
# rewritten file is picked up while the hot per-character/per-clip lookups stay
# free. Every worker process builds its own.
_cache: dict[str, tuple[tuple[int, int], IgnoreWarnings]] = {}


def _normalize_directive(token: str) -> str:
    return str(token).strip().lower().replace("_", "-")


def resolve_path(dataset_dir=None) -> Path | None:
    """The sidecar governing *dataset_dir*: its own, else its parent's, else None."""
    root = Path(get_dataset_dir(str(dataset_dir) if dataset_dir else None))
    for candidate in (root / FILENAME, root.parent / FILENAME):
        if candidate.is_file():
            return candidate
    return None


def parse(text: str, path: Path | None = None) -> IgnoreWarnings:
    stems: set[str] = set()
    patterns: list[str] = []
    directives: set[str] = set()
    unknown: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("!"):
            directive = _normalize_directive(stripped[1:])
            if not directive:
                continue
            if directive in KNOWN_DIRECTIVES:
                directives.add(directive)
            else:
                unknown.append(directive)
            continue
        if stripped.startswith("#"):
            pattern = stripped[1:].strip()
            if pattern:
                patterns.append(pattern.lower())
            continue
        # Accept both "name.npy" and bare "name" forms.
        stems.add(stripped.replace(".npy", ""))
    return IgnoreWarnings(
        path=path,
        stems=frozenset(stems),
        patterns=tuple(patterns),
        directives=frozenset(directives),
        unknown_directives=tuple(unknown),
    )


def load(dataset_dir=None) -> IgnoreWarnings:
    """The sidecar governing *dataset_dir*, empty when the dataset has none."""
    path = resolve_path(dataset_dir)
    if path is None:
        return _EMPTY
    key = str(path)
    stat = path.stat()
    stamp = (stat.st_mtime_ns, stat.st_size)
    cached = _cache.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    parsed = parse(path.read_text("utf-8"), path=path)
    _cache[key] = (stamp, parsed)
    return parsed


def for_configured_dataset() -> IgnoreWarnings:
    """The sidecar of the dataset this process is pointed at.

    The dataset directory comes from ``dataset_tags``, which the preprocessing
    worker pool already replays into every sub-process, so a switch reaches the
    workers without a second initializer.  Empty under a multi-source or
    cond-backed configuration -- neither names one dataset directory.
    """
    from data_loaders.truebones.truebones_utils.dataset_tags import configured_dataset_dir

    dataset_dir = configured_dataset_dir()
    if dataset_dir is None:
        return _EMPTY
    return load(dataset_dir)


def skip_orientation_detection(dataset_dir=None) -> bool:
    """True when the dataset declares its rest poses already canonically oriented."""
    sidecar = load(dataset_dir) if dataset_dir else for_configured_dataset()
    return sidecar.has(SKIP_ORIENTATION_DETECTION)
