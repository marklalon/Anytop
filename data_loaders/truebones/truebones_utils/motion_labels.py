from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from data_loaders.truebones.truebones_utils.param_utils import (
    BIPEDS,
    FISH,
    FLYING,
    MILLIPEDS,
    MOTION_METADATA_FILE,
    QUADROPEDS,
    SNAKES,
)


MOTION_METADATA_SCHEMA_VERSION = 1

_TOKEN_PATTERN = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+")
_ACTION_CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("fall", ("fall", "falling", "fallen", "drop", "dropping")),
    ("rise", ("rise", "rising", "recover", "recovery")),
    ("death", ("death", "dead", "die", "dying", "collapse", "knockout")),
    ("attack", ("attack", "bite", "sting", "slash", "strike", "kick", "punch", "peck", "gore", "hit", "whip", "swat")),
    ("reaction", ("react", "recoil", "flinch", "hurt", "stagger", "impact")),
    ("jump", ("jump", "hop", "leap", "vault", "pounce", "spring", "bound", "land", "landing")),
    ("locomotion", ("walk", "run", "jog", "sprint", "trot", "gallop", "crawl", "creep", "move", "forward", "backward", "strafe", "turnwalk", "swim", "fly", "glide", "slither")),
    ("turn", ("turn", "spin", "pivot", "rotate", "twist")),
    ("pose", ("idle", "rest", "pose", "stand", "wait", "breath", "breathe", "tpose", "bindpose")),
    ("posture", ("sit", "sleep", "lie", "crouch", "kneel")),
    ("emote", ("dance", "roar", "taunt", "celebrate", "wave", "look")),
)


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


def _match_action_rule(token: str, keywords: Iterable[str]) -> bool:
    if len(token) <= 1:
        return False
    return any(token == keyword or keyword in token for keyword in keywords)


def _collect_action_category_matches(tokens: list[str]) -> list[tuple[str, int, int, int]]:
    matches: list[tuple[str, int, int, int]] = []
    for rule_index, (category, keywords) in enumerate(_ACTION_CATEGORY_RULES):
        matched_indices = [token_index for token_index, token in enumerate(tokens) if _match_action_rule(token, keywords)]
        if matched_indices:
            matches.append((category, matched_indices[0], matched_indices[-1], rule_index))
    return matches


def infer_species_group(object_type: str) -> str:
    if object_type in BIPEDS:
        return "biped"
    if object_type in QUADROPEDS:
        return "quadruped"
    if object_type in MILLIPEDS:
        return "millipede"
    if object_type in SNAKES:
        return "snake"
    if object_type in FISH:
        return "fish"
    if object_type in FLYING:
        return "flying"
    return "other"


def infer_species_label(object_type: str) -> str:
    base = _strip_species_variant(object_type)
    tokens = _split_identifier_tokens(base)
    return " ".join(tokens) if tokens else object_type.lower()


def infer_species_key(object_type: str) -> str:
    return infer_species_label(object_type).replace(" ", "_")


def normalize_action_label(action_name: str) -> str:
    tokens = _split_identifier_tokens(action_name)
    return " ".join(tokens) if tokens else action_name.strip().lower()


def infer_action_tags(action_name: str) -> list[str]:
    tokens = _split_identifier_tokens(action_name)
    if not tokens:
        return ["other"]

    matches = _collect_action_category_matches(tokens)
    if not matches:
        return ["other"]
    ordered_matches = sorted(matches, key=lambda item: (item[1], item[2], item[3]))
    return [category for category, _first_index, _last_index, _rule_index in ordered_matches]


def infer_action_category(action_name: str) -> str:
    tokens = _split_identifier_tokens(action_name)
    if not tokens:
        return "other"

    matches = _collect_action_category_matches(tokens)
    if not matches:
        return "other"

    primary_match = max(matches, key=lambda item: (item[2], item[1], -item[3]))
    return primary_match[0]


def build_object_labels(object_type: str) -> dict[str, str]:
    return {
        "species_label": infer_species_label(object_type),
        "species_group": infer_species_group(object_type),
    }


def build_motion_labels(
    object_type: str,
    action_name: str,
    motion_name: str | None = None,
    source_file: str | None = None,
) -> dict[str, object]:
    action_label = normalize_action_label(action_name)
    action_tags = infer_action_tags(action_name)
    payload: dict[str, object] = {
        "object_type": object_type,
        "action_label": action_label,
        "action_category": infer_action_category(action_name),
        "action_tags": action_tags,
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
    stem = Path(motion_name).stem
    resolved_object_type = object_type
    if resolved_object_type is None:
        if object_types is None:
            resolved_object_type = stem.split("_", 1)[0]
        else:
            matches = [candidate for candidate in object_types if stem.startswith(f"{candidate}_")]
            if not matches:
                resolved_object_type = stem.split("_", 1)[0]
            else:
                resolved_object_type = max(matches, key=len)

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
            normalized[motion_name] = metadata
    return normalized


def write_motion_metadata(
    save_dir: str | Path,
    motion_entries: dict[str, dict[str, object]],
    total_clips: int,
) -> Path:
    output_path = Path(save_dir) / MOTION_METADATA_FILE
    payload = {
        "schema_version": MOTION_METADATA_SCHEMA_VERSION,
        "total_clips": int(total_clips),
        "motions": dict(sorted(motion_entries.items())),
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return output_path