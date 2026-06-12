from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from data_loaders.truebones.truebones_utils.param_utils import (
    MOTION_METADATA_FILE,
    MOTION_TAGS_FILE,
)


MOTION_METADATA_SCHEMA_VERSION = 5

_TOKEN_PATTERN = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+")

# ---------------------------------------------------------------------------
# Canonical action-tag vocabulary
# ---------------------------------------------------------------------------
# The full set of valid action tags. Order is significant: it defines the
# multi-hot index layout the model conditions on, so trained checkpoints depend
# on it staying stable. Tags themselves are maintained by hand in
# ``motion_tags.jsonl`` (see ``load_motion_tags``); this module never generates
# them automatically.

ACTION_TAGS: tuple[str, ...] = (
    "idle",
    "locomotion",
    "getup",
    "swim",
    "fly",
    "jump",
    "turn",
    "attack",
    "gethurt",
    "rest",
    "emote",
    "interact",
    "death",
    "fall",
    "unknown",
)

# Label fields that used to be baked into motion_metadata.json but are now
# either sourced from motion_tags.jsonl (action_tags) or redundant
# (action_label, motion_name == entry key). They are stripped on both load and
# write so the metadata file stays free of them.
_LEGACY_LABEL_FIELDS: tuple[str, ...] = (
    "action_tags",
    "action_label",
    "action_category",
    "motion_name",
)

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
# Public API
# ---------------------------------------------------------------------------

def infer_species_label(object_type: str) -> str:
    base = _strip_species_variant(object_type)
    tokens = _split_identifier_tokens(base)
    return " ".join(tokens) if tokens else object_type.lower()


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


# ---------------------------------------------------------------------------
# Metadata builders
# ---------------------------------------------------------------------------

def build_object_labels(object_type: str) -> dict[str, str]:
    return {"species_label": infer_species_label(object_type)}


def build_motion_labels(
    object_type: str,
    motion_name: str | None = None,
    source_file: str | None = None,
) -> dict[str, object]:
    """Build the (non-action) label fields for a motion clip.

    Action tags are no longer produced here — they are maintained by hand in
    ``motion_tags.jsonl`` and merged in by :func:`load_motion_metadata`.
    """
    payload: dict[str, object] = {"object_type": object_type}
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

    return build_motion_labels(resolved_object_type, motion_name=motion_name)


def _strip_legacy_label_fields(metadata: dict[str, object]) -> dict[str, object]:
    """Return a copy of *metadata* with the deprecated label fields removed."""
    return {
        key: value
        for key, value in metadata.items()
        if key not in _LEGACY_LABEL_FIELDS
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_motion_tags(dataset_dir: str | Path) -> dict[str, list[str]]:
    """Load the hand-maintained ``motion_tags.jsonl`` sidecar.

    Each line is a JSON object ``{"clip": "<name>.npy", "action_tags": [...]}``.
    Returns a mapping ``clip -> [tag, ...]``. Raises ``FileNotFoundError`` if the
    file is absent so callers fail fast rather than silently training without
    action conditioning.
    """
    tags_path = Path(dataset_dir) / MOTION_TAGS_FILE
    if not tags_path.exists():
        raise FileNotFoundError(
            f"{MOTION_TAGS_FILE} not found at {tags_path}. Action tags are now "
            f"maintained by hand in this file (one "
            f'{{"clip": "<name>.npy", "action_tags": [...]}} object per line).'
        )

    motion_tags: dict[str, list[str]] = {}
    with open(tags_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{MOTION_TAGS_FILE}:{line_number} is not valid JSON: {exc}"
                ) from exc
            if not isinstance(entry, dict):
                raise ValueError(
                    f"{MOTION_TAGS_FILE}:{line_number} must be a JSON object, "
                    f"got {type(entry).__name__}"
                )
            clip = entry.get("clip")
            if not clip:
                raise ValueError(
                    f"{MOTION_TAGS_FILE}:{line_number} is missing the 'clip' field"
                )
            motion_tags[str(clip)] = normalize_action_tags(entry.get("action_tags"))
    return motion_tags


def load_motion_metadata(dataset_dir: str | Path) -> dict[str, dict[str, object]]:
    metadata_path = Path(dataset_dir) / MOTION_METADATA_FILE
    if not metadata_path.exists():
        return {}

    with open(metadata_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    motions = payload.get("motions", payload)
    if not isinstance(motions, dict):
        return {}

    motion_tags = load_motion_tags(dataset_dir)

    normalized: dict[str, dict[str, object]] = {}
    missing_tags: list[str] = []
    for motion_name, metadata in motions.items():
        if not isinstance(metadata, dict):
            continue
        tags = motion_tags.get(motion_name)
        if tags is None:
            missing_tags.append(motion_name)
            continue
        entry = _strip_legacy_label_fields(metadata)
        entry["action_tags"] = list(tags)
        normalized[motion_name] = entry

    if missing_tags:
        preview = ", ".join(sorted(missing_tags)[:10])
        more = "" if len(missing_tags) <= 10 else f" (+{len(missing_tags) - 10} more)"
        raise KeyError(
            f"{MOTION_TAGS_FILE} is missing action_tags for {len(missing_tags)} "
            f"clip(s): {preview}{more}. Add an entry for each clip."
        )
    return normalized


def apply_attack_loop_override(
    motion_metadata_lookup: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Force ``is_loop=False`` for attack actions, in memory only.

    Attack actions are inherently non-cyclic, so the loop condition must never
    activate for them at training or inference time. We deliberately do *not*
    persist this to the metadata file (the on-disk ``is_loop`` value is kept
    untouched); instead callers apply this override after loading the metadata.
    Mutates the entries in place and returns the same lookup for convenience.
    """
    for entry in motion_metadata_lookup.values():
        if not isinstance(entry, dict):
            continue
        action_tags = entry.get("action_tags") or []
        if "attack" in action_tags:
            entry["is_loop"] = False
    return motion_metadata_lookup


def write_motion_metadata(
    save_dir: str | Path,
    motion_entries: dict[str, dict[str, object]],
    total_clips: int,
) -> Path:
    output_path = Path(save_dir) / MOTION_METADATA_FILE
    sanitized_entries = {
        motion_name: _strip_legacy_label_fields(metadata)
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
