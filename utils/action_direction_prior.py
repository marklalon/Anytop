"""Checkpoint-local direction defaults, counted once per selected training clip."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from data_loaders.truebones.truebones_utils.motion_labels import (
    DIRECTION_VOCAB, canonical_action_label, head_words_in, parse_action_label,
)

PRIOR_FILE = 'action_direction_prior.json'
SCHEMA = 1
COVERAGE = 0.8
NEAR_TOP = 0.1


def build_direction_prior(records):
    """records are (stable clip id, label), after training eligibility filters.

    Head order is preserved: it changes head-slot weighting. Modifiers are
    pooled, but additional heads are never mixed into a single-head count.
    """
    records = sorted((str(key), str(label or '')) for key, label in records)
    if len({key for key, _ in records}) != len(records):
        raise ValueError('Direction prior requires unique clip ids')
    counts = defaultdict(Counter)
    for _, label in records:
        words = parse_action_label(label)
        heads = tuple(head_words_in(words))
        if heads:
            directions = [word for word in DIRECTION_VOCAB if word in words]
            counts[', '.join(heads)][', '.join(directions)] += 1
    return {
        'schema': SCHEMA,
        'coverage_threshold': COVERAGE,
        'near_top_gap': NEAR_TOP,
        'source_fingerprint': hashlib.sha256(
            json.dumps(records, ensure_ascii=False).encode('utf-8')
        ).hexdigest(),
        'by_heads': {key: dict(sorted(value.items())) for key, value in sorted(counts.items())},
    }


def prior_from_dataset(dataset):
    motions = dataset.motion_dataset
    return build_direction_prior(
        (key, motions.data_dict[key]['motion_metadata'].get('action_label', ''))
        for key in motions.name_list
    )


def save_direction_prior(prior, directory):
    path = Path(directory) / PRIOR_FILE
    path.write_text(json.dumps(prior, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return path


def load_direction_prior(model_path):
    """Older weights need no conversion. A saved selection can supply counts."""
    if not model_path:
        return None
    directory = Path(model_path).resolve().parent
    path = directory / PRIOR_FILE
    if path.is_file():
        prior = json.loads(path.read_text(encoding='utf-8'))
        if prior.get('schema') != SCHEMA:
            raise ValueError(f'Unsupported direction prior schema in {path}')
        return prior
    selection = directory / 'sample_selection.json'
    if selection.is_file():
        snapshot = json.loads(selection.read_text(encoding='utf-8'))
        return build_direction_prior(
            (row['clip_id'], row['action_label']) for row in snapshot['clips']
        )
    return None


def complete_direction(label, prior, seed=0):
    """Return (canonical label, decision details), using a private seeded RNG."""
    words = parse_action_label(label)
    if any(word in DIRECTION_VOCAB for word in words) or prior is None:
        return label, None
    heads = ', '.join(head_words_in(words))
    counts = prior['by_heads'].get(heads, {})
    total = sum(counts.values())
    directed = total - counts.get('', 0)
    threshold = prior.get('coverage_threshold', COVERAGE)
    if not total or directed / total < threshold:
        return label, None
    top = max(count for direction, count in counts.items() if direction)
    gap = prior.get('near_top_gap', NEAR_TOP)
    candidates = sorted(
        (direction, count) for direction, count in counts.items()
        if direction and (top - count) / directed <= gap + 1e-12
    )
    weights = np.asarray([count for _, count in candidates], dtype=np.float64)
    weights /= weights.sum()
    index = int(np.random.default_rng(seed).choice(len(candidates), p=weights))
    direction = candidates[index][0]
    resolved = canonical_action_label(words + direction.split(', '))
    # Use the parser's validation for the completed label as well.
    parse_action_label(resolved)
    return resolved, {
        'heads': heads, 'coverage': directed / total,
        'direction': direction, 'candidates': dict(zip(
            (candidate for candidate, _ in candidates), weights.tolist())),
    }
