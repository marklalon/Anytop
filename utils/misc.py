import torch


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array".format(
            type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(
            type(ndarray)))
    return ndarray


def cleanexit():
    import sys
    import os
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)


def freeze_joints(x, joints_to_freeze):
    # Freezes selected joint *rotations* as they appear in the first frame
    # x [bs, [root+n_joints], joint_dim(6), seqlen]
    frozen = x.detach().clone()
    frozen[:, joints_to_freeze, :, :] = frozen[:, joints_to_freeze, :, :1]
    return frozen


# ── Object type inference from filenames ───────────────────────────────────

import glob as _glob
import os as _os
from collections.abc import Container as _Container, Mapping as _Mapping


def _strip_common_suffixes(candidate: str) -> str:
    """Strip common Truebones filename suffixes like 'All' from a candidate object type."""
    for suffix in ("All", "Tpose", "T-Pose", "TPose"):
        if candidate.endswith(suffix):
            stripped = candidate[: -len(suffix)]
            if stripped:
                return stripped
    return candidate


def infer_object_type_from_filename(
    filename: str,
    valid_types: _Container[str] | None = None,
) -> str | None:
    """Infer an object type key from a motion/fbx filename.

    Handles these patterns (in priority order):

        ``{Type}___{Action}_{ID}.ext``     — triple underscore (e.g. ``Horse___RunToStop_29.npy``)
        ``{Type}_{Action}_{ID}.ext``        — single underscore (e.g. ``Sea_Lion_Swim_42.npy``)
        ``{Type}-{Action}.ext``             — hyphen (e.g. ``Wyvern-Tpose.fbx``)
        ``{Type}.{suffix}.ext``             — dot (e.g. ``Elephant.rig.glb``)

    When *valid_types* is provided the extracted candidate(s) are validated
    against that container.  Multi-word types (e.g. ``Sea_Lion``) are handled
    via progressive prefix matching. Validation is **case-insensitive** and the
    canonical key from *valid_types* is returned, so a lowercase filename like
    ``dragon_tpose.glb`` resolves to a registered ``Dragon`` key (matching the
    case-insensitive registry check downstream) instead of silently missing.

    *valid_types* may also be a **mapping** of filename token → canonical key,
    which is how a multi-dataset cond is passed in (see
    ``dataset_sources.species_lookup_map``): filenames carry the token
    ``Horse@truebones_zoo_upgrade`` while the cond key is
    ``truebones/zoo_upgrade/Horse``, and the mapping is what turns one into the
    other.  With a plain container the key is its own token, so single-dataset
    behaviour is unchanged.

    Common Truebones suffixes (e.g. ``All`` in ``LionAll-Walk.fbx``) are
    stripped from candidates so that the inferred type matches the
    preprocessing convention.

    Args:
        filename:   A file path or plain filename.
        valid_types: Optional set/container of known object types, or a mapping
                     of filename token → canonical key, for validation.

    Returns:
        The inferred object type, or ``None`` if inference fails.
    """
    stem = _os.path.splitext(_os.path.basename(filename))[0]
    if not stem:
        return None

    # Lookup from candidate → canonical valid_types key (built once). ``_match``
    # returns the canonical key when validating, the raw candidate when
    # validation is off, or ``None`` on a validated miss. An exact hit is tried
    # before a case-folded one, matching ``resolve_species_key``: the dataset
    # carries real ``Rhino``/``rhino`` and ``Scorpion``/``scorpion`` pairs, and
    # folding first would hand every ``rhino_*.npy`` to whichever of the two the
    # cond happens to list first.
    _canon = None
    _canon_folded = None
    if valid_types is not None:
        _canon = {}
        _canon_folded = {}
        _pairs = (
            valid_types.items()
            if isinstance(valid_types, _Mapping)
            else ((known, known) for known in valid_types)
        )
        for token, canonical in _pairs:
            _canon.setdefault(token, canonical)
            _canon_folded.setdefault(token.lower(), canonical)

    def _match(candidate: str) -> str | None:
        if valid_types is None:
            return candidate
        exact = _canon.get(candidate)
        if exact is not None:
            return exact
        return _canon_folded.get(candidate.lower())

    # 1. Triple-underscore separator  (highest priority)
    sep_triple = "___"
    if sep_triple in stem:
        matched = _match(_strip_common_suffixes(stem.split(sep_triple, 1)[0]))
        if matched is not None:
            return matched

    # 2. Progressive single-underscore prefix matching
    #    (handles multi-word types like "Sea_Lion"). The range includes the whole
    #    stem so a file named after the species alone ("Deer_Buck.glb") matches
    #    "Deer_Buck" instead of stopping at the shorter "Deer".
    if valid_types is not None and "_" in stem:
        parts = stem.split("_")
        best: str | None = None
        for i in range(1, len(parts) + 1):
            matched = _match(_strip_common_suffixes("_".join(parts[:i])))
            if matched is not None:
                best = matched  # keep going for a longer match
        if best is not None:
            return best

    # 3. Single underscore — first token (blind, when no valid_types)
    if "_" in stem:
        first_token = _strip_common_suffixes(stem.split("_", 1)[0])
        if first_token:
            matched = _match(first_token)
            if matched is not None:
                return matched

    # 4. Hyphen separator (for FBX stems like "Wyvern-Tpose")
    if "-" in stem:
        first_token = _strip_common_suffixes(stem.split("-", 1)[0])
        if first_token:
            matched = _match(first_token)
            if matched is not None:
                return matched

    # 4b. Dot separator (for stems like "Elephant.rig" from "Elephant.rig.glb")
    if "." in stem:
        first_token = _strip_common_suffixes(stem.split(".", 1)[0])
        if first_token:
            matched = _match(first_token)
            if matched is not None:
                return matched

    # 5. Fallback: bare stem (e.g. "dragon.fbx" → "dragon"). Validated too, so a
    # file named after the species alone ("Sea_Lion.npy", "Deer_Buck.npy") still
    # resolves once a registry is supplied -- rule 2 only ever tries *proper*
    # prefixes, and rule 3 would stop at "Sea".
    stripped = _strip_common_suffixes(stem)
    if not stripped:
        return None
    if valid_types is None:
        return stripped
    return _match(stripped)


# ── Dataset asset path portability ─────────────────────────────────────────
# Dataset sidecars store asset paths such as the ``tpose_reference_path`` in the
# tpose_reference_paths.jsonl sidecar. Historically these were absolute paths, which
# break when the repo/dataset is moved to another machine or mounted into a
# container at a different prefix (e.g. a Windows ``D:\...`` path inside a Linux
# container). They are now stored as an *AnyTop-root-relative POSIX* path via
# ``to_portable_dataset_path`` — the same anchor every other relative path in the
# module uses (``param_utils._ANYTOP_ROOT``), so ``Anytop`` can be relocated or
# vendored without rewriting its sidecars. All forms (AnyTop-relative, legacy
# repo-root-relative with an ``Anytop/`` prefix, and foreign-absolute paths) are
# resolved back to a local path by ``resolve_dataset_path``.

def anytop_root_dir() -> str:
    """Absolute path of the ``Anytop`` module root (the parent of ``utils``)."""
    return _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))


def repo_root_dir() -> str:
    """Absolute path of the repository root (the parent of the ``Anytop`` dir).

    Kept only as a *fallback* resolution root for legacy sidecars whose paths
    were written repo-root-relative (i.e. with an ``Anytop/`` prefix).
    """
    return _os.path.dirname(anytop_root_dir())


def to_portable_dataset_path(path: str | None) -> str | None:
    """Return a portable form of *path* for storage in a dataset sidecar.

    Paths inside the ``Anytop`` module tree become AnyTop-root-relative POSIX
    paths; paths outside it (or on a different Windows drive) are kept as a
    normalised absolute path. ``None``/empty input returns ``None``.
    """
    if not path:
        return None
    abs_path = _os.path.abspath(path)
    root = anytop_root_dir()
    try:
        rel = _os.path.relpath(abs_path, root)
    except ValueError:
        return abs_path  # different drive on Windows — cannot be made relative
    if rel.startswith(_os.pardir):
        return abs_path  # outside the Anytop tree
    return rel.replace(_os.sep, "/")


def resolve_dataset_path(stored, *, extra_roots=None) -> str | None:
    """Resolve a stored dataset asset path to a local path.

    Resolution order:
      1. ``None``/empty → ``None``.
      2. An absolute path that exists as-is → returned unchanged.
      3. A relative path → joined against the AnyTop root, then the repo root
         (legacy ``Anytop/...`` entries), then any *extra_roots*.

    Raises ``FileNotFoundError`` if no candidate exists.
    """
    if not stored:
        return None
    raw = str(stored)
    if _os.path.isabs(raw) and _os.path.isfile(raw):
        return raw

    roots = [anytop_root_dir(), repo_root_dir()]
    if extra_roots:
        roots.extend(r for r in extra_roots if r)

    if not _os.path.isabs(raw):
        rel = raw.replace("/", _os.sep)
        for root in roots:
            candidate = _os.path.join(root, rel)
            if _os.path.isfile(candidate):
                return candidate

    raise FileNotFoundError(
        f"Dataset path not found: {stored!r} (searched roots: {roots})"
    )


# ── tpose_reference_paths.jsonl sidecar I/O ────────────────────────────────
# A sidecar written into a scratch/cache dir (e.g. the server's per-skeleton
# cache under ``outputs/server/cache/skeletons/<hash>/``) usually points at a
# mesh copied into that very dir. Such assets live outside the Anytop tree, so
# ``to_portable_dataset_path`` can only keep them absolute — which pins the cache
# dir to one machine. The sidecar I/O therefore stores paths *inside the sidecar's
# own directory* relative to it, leaving the cache dir self-contained and movable;
# loading resolves those back against the sidecar dir.

def _relative_to_sidecar_dir(path: str | None, sidecar_dir: str) -> str | None:
    """Return *path* relative to *sidecar_dir* when it lives inside it, else *path*."""
    if not path or not _os.path.isabs(path):
        return path
    try:
        rel = _os.path.relpath(path, sidecar_dir)
    except ValueError:
        return path  # different drive on Windows
    if rel.startswith(_os.pardir):
        return path  # outside the sidecar dir
    return rel.replace(_os.sep, "/")


def load_tpose_reference_sidecar(path: str) -> dict[str, str]:
    """Load the JSONL sidecar (``object_type`` → mesh path).

    Returns a dict keyed by ``object_type``; entries with a ``None`` path are
    omitted.  Returns an empty dict if the file does not exist.  Entries stored
    relative to the sidecar's own directory are returned as absolute local paths;
    every other form is returned verbatim for ``resolve_dataset_path``.
    """
    import json
    refs: dict[str, str] = {}
    if not _os.path.isfile(path):
        return refs
    sidecar_dir = _os.path.dirname(_os.path.abspath(path))
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            path_val = entry.get("path")
            if path_val is None:
                continue
            if not _os.path.isabs(path_val):
                beside = _os.path.join(sidecar_dir, path_val.replace("/", _os.sep))
                if _os.path.isfile(beside):
                    path_val = beside
            refs[entry["object_type"]] = path_val
    return refs


def save_tpose_reference_sidecar(path: str, refs: dict[str, str | None]) -> None:
    """Write the JSONL sidecar (one ``{"object_type": ..., "path": ...}`` per line).

    Paths are stored in the most portable form available: relative to the
    sidecar's own directory when the asset lives inside it (so scratch/cache
    dirs stay self-contained), otherwise AnyTop-root-relative, otherwise
    absolute. Entries with a ``None`` path are written as ``null`` so the
    consumer can distinguish an explicitly cleared entry from a missing one.
    """
    import json
    sidecar_dir = _os.path.dirname(_os.path.abspath(path))
    with open(path, "w", encoding="utf-8") as f:
        for ot, p in refs.items():
            stored = _relative_to_sidecar_dir(p, sidecar_dir)
            if stored and _os.path.isabs(stored):
                stored = to_portable_dataset_path(stored)
            f.write(json.dumps(
                {"object_type": ot, "path": stored},
                ensure_ascii=False,
            ) + "\n")


# ── String normalisation helpers (shared across tools) ───────────────────

def normalize_bone_key(name: str) -> str:
    """Normalise a bone/joint name for dictionary-key or comparison use."""
    return name.replace(" ", "_").lower()


def normalize_identifier(value: str) -> str:
    """Strip all non-alphanumeric characters and lower-case."""
    import re
    return re.sub(r"[^a-z0-9]+", "", value.lower())
