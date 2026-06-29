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
from collections.abc import Container as _Container


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

    When *valid_types* is provided the extracted candidate(s) are validated
    against that container.  Multi-word types (e.g. ``Sea_Lion``) are handled
    via progressive prefix matching.

    Common Truebones suffixes (e.g. ``All`` in ``LionAll-Walk.fbx``) are
    stripped from candidates so that the inferred type matches the
    preprocessing convention.

    Args:
        filename:   A file path or plain filename.
        valid_types: Optional set/container of known object types for validation.

    Returns:
        The inferred object type, or ``None`` if inference fails.
    """
    stem = _os.path.splitext(_os.path.basename(filename))[0]
    if not stem:
        return None

    # 1. Triple-underscore separator  (highest priority)
    sep_triple = "___"
    if sep_triple in stem:
        candidate = _strip_common_suffixes(stem.split(sep_triple, 1)[0])
        if valid_types is None or candidate in valid_types:
            return candidate

    # 2. Progressive single-underscore prefix matching
    #    (handles multi-word types like "Sea_Lion")
    if valid_types is not None and "_" in stem:
        parts = stem.split("_")
        best: str | None = None
        for i in range(1, len(parts)):
            candidate = _strip_common_suffixes("_".join(parts[:i]))
            if candidate in valid_types:
                best = candidate  # keep going for a longer match
        if best is not None:
            return best

    # 3. Single underscore — first token (blind, when no valid_types)
    if "_" in stem:
        first_token = _strip_common_suffixes(stem.split("_", 1)[0])
        if first_token:
            if valid_types is None or first_token in valid_types:
                return first_token

    # 4. Hyphen separator (for FBX stems like "Wyvern-Tpose")
    if "-" in stem:
        first_token = _strip_common_suffixes(stem.split("-", 1)[0])
        if first_token:
            if valid_types is None or first_token in valid_types:
                return first_token

    # 5. Fallback: bare stem (e.g. "dragon.fbx" → "dragon")
    if valid_types is None:
        stripped = _strip_common_suffixes(stem)
        return stripped if stripped else None

    return None


# ── Dataset asset path portability ─────────────────────────────────────────
# Dataset sidecars store asset paths such as the ``tpose_reference_path`` in the
# tpose_reference_paths sidecar. Historically these were absolute paths, which
# break when the repo/dataset is moved to another machine or mounted into a
# container at a different prefix (e.g. a Windows ``D:\...`` path inside a Linux
# container). They are now stored as a *repo-root-relative POSIX* path via
# ``to_portable_dataset_path``; both forms
# (and legacy foreign-absolute paths) are resolved back to a local path by
# ``resolve_dataset_path``.

def repo_root_dir() -> str:
    """Absolute path of the repository root (the parent of the ``Anytop`` dir)."""
    return _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


def to_portable_dataset_path(path: str | None) -> str | None:
    """Return a portable form of *path* for storage in cond.npy.

    Paths inside the repo become repo-root-relative POSIX paths; paths outside
    the repo (or on a different Windows drive) are kept as a normalised absolute
    path. ``None``/empty input returns ``None``.
    """
    if not path:
        return None
    abs_path = _os.path.abspath(path)
    root = repo_root_dir()
    try:
        rel = _os.path.relpath(abs_path, root)
    except ValueError:
        return abs_path  # different drive on Windows — cannot be made relative
    if rel.startswith(_os.pardir):
        return abs_path  # outside the repo tree
    return rel.replace(_os.sep, "/")


def resolve_dataset_path(stored, *, extra_roots=None) -> str | None:
    """Resolve a cond.npy asset path (repo-root-relative or absolute) to a
    local path.

    Resolution order:
      1. ``None``/empty → ``None``.
      2. An absolute path that exists as-is → returned unchanged.
      3. A relative path → joined against the repo root (and any *extra_roots*).

    Raises ``FileNotFoundError`` if no candidate exists.
    """
    if not stored:
        return None
    raw = str(stored)
    if _os.path.isabs(raw) and _os.path.isfile(raw):
        return raw

    roots = [repo_root_dir()]
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


# ── String normalisation helpers (shared across tools) ───────────────────

def normalize_bone_key(name: str) -> str:
    """Normalise a bone/joint name for dictionary-key or comparison use."""
    return name.replace(" ", "_").lower()


def normalize_identifier(value: str) -> str:
    """Strip all non-alphanumeric characters and lower-case."""
    import re
    return re.sub(r"[^a-z0-9]+", "", value.lower())
