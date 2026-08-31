"""
Persistent cache for LLM-based retarget joint mappings.

Provides both an in-memory dict (fast, process-local) and a disk-backed
JSON cache (survives process restarts).  Cache entries are keyed by a
SHA-256 hash of the prompt+messages, with a version prefix for easy
invalidation.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Bump this to invalidate all existing cache entries.
# v2: bone_len precision increased from .2f to .4f, top_p=1.0 for determinism.
CACHE_VERSION = "v2"

# Cache directory — resolved relative to this file's location: <Anytop>/cache/retarget/.
# The cache key is a SHA-256 of the prompt and messages (which already carry the
# bone names and lengths), so entries are species- and dataset-independent and the
# cache does not belong inside any one dataset directory.
_ANYTOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE_DIR = os.path.join(_ANYTOP_DIR, "cache", "retarget")


# ---------------------------------------------------------------------------
# In-memory cache (process-local, keyed by same hash as disk cache)
# ---------------------------------------------------------------------------

_in_memory_cache: dict[str, dict[str, Optional[str]]] = {}


def get_from_memory(system_msg: str, user_msg: str) -> Optional[dict[str, Optional[str]]]:
    """Return cached mapping from in-memory cache, or None on miss.

    Uses the same hash key as the disk cache so both layers stay consistent.
    """
    key = _get_cache_path(system_msg, user_msg)
    return _in_memory_cache.get(key)


def set_in_memory(system_msg: str, user_msg: str, result: dict[str, Optional[str]]):
    """Store a mapping in the in-memory cache."""
    key = _get_cache_path(system_msg, user_msg)
    _in_memory_cache[key] = result


# ---------------------------------------------------------------------------
# Disk cache (persistent, keyed by SHA-256 of prompt content)
# ---------------------------------------------------------------------------


def _cache_key(system_msg: str, user_msg: str) -> str:
    raw = f"{CACHE_VERSION}|{system_msg}|{user_msg}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _get_cache_path(system_msg: str, user_msg: str) -> str:
    """Compute the file path for a cache entry based on prompt+message content."""
    return os.path.join(_CACHE_DIR, f"{_cache_key(system_msg, user_msg)}.json")


def load_from_disk(system_msg: str, user_msg: str) -> Optional[dict[str, Optional[str]]]:
    """Load a cached LLM joint mapping from disk, or return None if not found."""
    path = _get_cache_path(system_msg, user_msg)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Validate structure and version
        if isinstance(data, dict) and "_version" in data and "_mapping" in data:
            if data["_version"] == CACHE_VERSION:
                return data["_mapping"]
            # Version mismatch — treat as cache miss
            return None
    except (json.JSONDecodeError, OSError):
        return None
    return None


def save_to_disk(system_msg: str, user_msg: str, result: dict[str, Optional[str]]):
    """Save an LLM joint mapping to disk.

    Silently ignores I/O errors — the in-memory cache still works.
    """
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = _get_cache_path(system_msg, user_msg)
    data = {
        "_version": CACHE_VERSION,
        "_mapping": result,
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError:
        pass  # Non-critical — in-memory cache still works
