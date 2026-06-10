from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from data_loaders.truebones.truebones_utils.param_utils import (
    MOTION_METADATA_FILE,
)


MOTION_METADATA_SCHEMA_VERSION = 4

_TOKEN_PATTERN = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+")

# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def _split_identifier_tokens(value: str) -> list[str]:
    raw_parts = re.split(r"[^A-Za-z0-9]+", value)
    tokens: list[str] = []
    for part in raw_parts:
        if not part:
            continue
        matches = _TOKEN_PATTERN.findall(part)
        if matches:
            tokens.extend(matches)
        else:
            tokens.append(part)
    return [token.lower() for token in tokens if token]


def _strip_species_variant(object_type: str) -> str:
    base = re.sub(r"[-_\s]*\d+$", "", object_type).strip("-_")
    if len(base) > 1 and base[-1].isupper() and base[-2].islower():
        return base[:-1]
    return base


# ---------------------------------------------------------------------------
# LLM integration — keyword fallback has been removed.
# All classification MUST go through the LLM; prefetch before querying.
# ---------------------------------------------------------------------------

def _get_llm_module():
    """Import the LLM classifier module; raises ImportError if unavailable."""
    from data_loaders.truebones.truebones_utils import motion_labels_llm
    return motion_labels_llm


def _llm_classify_batch(action_names: list[str]) -> dict[str, list[str]]:
    if not action_names:
        return {}
    return _get_llm_module().classify_action_tags_batch(action_names)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def infer_species_label(object_type: str) -> str:
    base = _strip_species_variant(object_type)
    tokens = _split_identifier_tokens(base)
    return " ".join(tokens) if tokens else object_type.lower()


def normalize_action_label(action_name: str) -> str:
    tokens = _split_identifier_tokens(action_name)
    return " ".join(tokens) if tokens else action_name.strip().lower()


def normalize_action_tags(raw_action_tags) -> list[str]:
    if raw_action_tags is None:
        return []
    if isinstance(raw_action_tags, str):
        values = [raw_action_tags]
    elif isinstance(raw_action_tags, (list, tuple, set)):
        values = raw_action_tags
    else:
        values = [raw_action_tags]

    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        tag = str(value).strip().lower()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        normalized.append(tag)
    return normalized


def infer_action_tags(action_name: str) -> list[str]:
    """Return the resolved action tags for an action name (LLM-based).

    First checks the in-memory / disk cache.  On a cache miss, issues a
    single-name LLM call and caches the result.  Call
    ``prefetch_action_tags`` upfront to avoid per-name LLM latency.

    Returns ``["unknown"]`` if the LLM cannot classify the name.
    """
    results = _llm_classify_batch([action_name])
    return normalize_action_tags(results.get(action_name, ["unknown"]))


def prefetch_action_tags(action_names: list[str]) -> None:
    """Pre-classify a batch of action names, filling the in-memory + disk cache.

    Optional optimisation: call this once with all known action names before
    looping over individual files.  ``infer_action_tags`` works correctly
    without this — the cache ensures only the first cold query per name ever
    reaches the LLM.

    Raises ImportError if the LLM module is unavailable.
    """
    if not action_names:
        return
    _llm_classify_batch(action_names)


# ---------------------------------------------------------------------------
# Metadata builders
# ---------------------------------------------------------------------------

def build_object_labels(object_type: str) -> dict[str, str]:
    return {"species_label": infer_species_label(object_type)}


def build_motion_labels(
    object_type: str,
    action_name: str,
    motion_name: str | None = None,
    source_file: str | None = None,
) -> dict[str, object]:
    action_label = normalize_action_label(action_name)
    payload: dict[str, object] = {
        "object_type": object_type,
        "action_label": action_label,
        "action_tags": infer_action_tags(action_name),
    }
    payload.update(build_object_labels(object_type))
    if motion_name is not None:
        payload["motion_name"] = motion_name
    return payload


def infer_motion_labels_from_motion_name(
    motion_name: str,
    object_type: str | None = None,
    object_types: Iterable[str] | None = None,
) -> dict[str, object]:
    try:
        from utils.misc import infer_object_type_from_filename
    except ImportError:
        from Anytop.utils.misc import infer_object_type_from_filename

    stem = Path(motion_name).stem
    resolved_object_type = object_type
    if resolved_object_type is None:
        resolved_object_type = infer_object_type_from_filename(
            motion_name, valid_types=set(object_types) if object_types is not None else None
        )
        if resolved_object_type is None:
            resolved_object_type = stem.split("_", 1)[0]

    action_stem = stem
    prefix = f"{resolved_object_type}_"
    if action_stem.startswith(prefix):
        action_stem = action_stem[len(prefix):]
    action_stem = re.sub(r"_\d+$", "", action_stem).strip("_")
    if not action_stem:
        action_stem = stem
    return build_motion_labels(
        resolved_object_type,
        action_stem,
        motion_name=motion_name,
    )


def _normalize_motion_metadata_entry(metadata: dict[str, object]) -> dict[str, object]:
    normalized = dict(metadata)
    legacy_action_category = normalized.pop("action_category", None)
    raw_action_tags = normalized.get("action_tags")
    if raw_action_tags is None and legacy_action_category is not None:
        raw_action_tags = [legacy_action_category]
    normalized["action_tags"] = normalize_action_tags(raw_action_tags)
    return normalized


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_motion_metadata(dataset_dir: str | Path) -> dict[str, dict[str, object]]:
    metadata_path = Path(dataset_dir) / MOTION_METADATA_FILE
    if not metadata_path.exists():
        return {}

    with open(metadata_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    motions = payload.get("motions", payload)
    if not isinstance(motions, dict):
        return {}

    normalized: dict[str, dict[str, object]] = {}
    for motion_name, metadata in motions.items():
        if isinstance(metadata, dict):
            normalized[motion_name] = _normalize_motion_metadata_entry(metadata)
    return normalized


def write_motion_metadata(
    save_dir: str | Path,
    motion_entries: dict[str, dict[str, object]],
    total_clips: int,
) -> Path:
    output_path = Path(save_dir) / MOTION_METADATA_FILE
    sanitized_entries = {
        motion_name: _normalize_motion_metadata_entry(metadata)
        for motion_name, metadata in motion_entries.items()
        if isinstance(metadata, dict)
    }
    payload = {
        "schema_version": MOTION_METADATA_SCHEMA_VERSION,
        "total_clips": int(total_clips),
        "motions": dict(sorted(sanitized_entries.items())),
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return output_path
