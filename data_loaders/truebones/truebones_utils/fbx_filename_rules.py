"""Filename-based animation preprocessing rules.

This module isolates the rules that infer meaning from animation filenames
(FBX/GLB/GLTF):
- selecting a reference clip by filename
- stripping duplicated object prefixes from action names
- normalizing action names for exported dataset filenames
- filtering non-motion or aggregate files before preprocessing
"""

import os
import re

from .physics_joint_annotation import normalize_joint_name


def _reference_stem_tokens(file_path):
    stem = os.path.splitext(os.path.basename(file_path))[0]
    normalized = normalize_joint_name(stem)
    return normalized.split(), normalized.replace(' ', '')


_IDLE_REFERENCE_TAIL_PATTERN = re.compile(
    r'^idle(?:\d+)?(?:loop|cyc|cycle|repeat|repeating)?$'
)
_WALK_REFERENCE_TAIL_PATTERN = re.compile(
    r'^walk(?:ing)?(?:\d+)?(?:loop|cyc|cycle|repeat|repeating|forward|forwards)?$'
)
_RUN_REFERENCE_TAIL_PATTERN = re.compile(
    r'^run(?:ning)?(?:\d+)?(?:loop|cyc|cycle|repeat|repeating|forward|forwards)?$'
)
_FLY_REFERENCE_TAIL_PATTERN = re.compile(
    r'^fly(?:ing)?(?:\d+)?(?:loop|cyc|cycle|repeat|repeating|forward|forwards)?$'
)


def _reference_tail_candidates(file_path):
    stem = os.path.splitext(os.path.basename(file_path))[0].lower()
    segments = [segment for segment in re.split(r'[^a-z0-9]+', stem) if segment]
    return [''.join(segments[index:]) for index in range(len(segments))]


def _matches_reference_tail(file_path, tail_pattern):
    return any(tail_pattern.fullmatch(candidate) for candidate in _reference_tail_candidates(file_path))


def _is_tpose_reference_path(file_path):
    tokens, compact = _reference_stem_tokens(file_path)
    return (
        'tpose' in compact
        or 'tpos' in compact
        or 'bindpose' in compact
        or 'restpose' in compact
        or 'nosaddle' in compact
        or ('pose' in tokens and 't' in tokens)
    )


def _is_idle_reference_path(file_path):
    return _matches_reference_tail(file_path, _IDLE_REFERENCE_TAIL_PATTERN)


def _is_walk_reference_path(file_path):
    return _matches_reference_tail(file_path, _WALK_REFERENCE_TAIL_PATTERN)


def _is_run_reference_path(file_path):
    return _matches_reference_tail(file_path, _RUN_REFERENCE_TAIL_PATTERN)


def _is_fly_reference_path(file_path):
    return _matches_reference_tail(file_path, _FLY_REFERENCE_TAIL_PATTERN)


def find_tpose_reference_path(anim_files):
    """Find a character-level orientation reference clip with priority T-pose > idle > walk > run > fly."""
    for file_path in anim_files:
        if _is_tpose_reference_path(file_path):
            anim_files.remove(file_path)
            return file_path

    for matcher, _source_name in (
        (_is_idle_reference_path, 'idle'),
        (_is_walk_reference_path, 'walk'),
        (_is_run_reference_path, 'run'),
        (_is_fly_reference_path, 'fly'),
    ):
        for file_path in anim_files:
            if matcher(file_path):
                return file_path

    return anim_files[0]


def _compact_normalized_text(value: str) -> str:
    normalized = normalize_joint_name(str(value or ''))
    return normalized.replace(' ', '')


def _object_type_tokens(object_type: str) -> list[str]:
    object_text = str(object_type or '')
    if not object_text:
        return []
    object_text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', object_text)
    object_text = re.sub(r'(?<=[A-Za-z])(?=[0-9])', ' ', object_text)
    object_text = re.sub(r'(?<=[0-9])(?=[A-Za-z])', ' ', object_text)
    object_text = re.sub(r'[^0-9A-Za-z]+', ' ', object_text)
    normalized = normalize_joint_name(object_text)
    return [token for token in normalized.split() if token]


def _object_prefix_aliases(object_type: str) -> set[str]:
    tokens = _object_type_tokens(object_type)
    aliases = set()

    if not tokens:
        compact = _compact_normalized_text(object_type)
        return {compact} if compact else set()

    full_alias = ''.join(tokens)
    if full_alias:
        aliases.add(full_alias)
        aliases.add(re.sub(r'\d+$', '', full_alias))

    for token in tokens:
        aliases.add(token)
        aliases.add(re.sub(r'\d+$', '', token))

    if len(tokens) > 1:
        aliases.update(''.join(tokens[index:]) for index in range(len(tokens)))
        aliases.update(''.join(tokens[:index + 1]) for index in range(len(tokens)))

    return {alias for alias in aliases if alias and alias != 'all'}


def _bounded_levenshtein_distance(left: str, right: str, limit: int) -> int:
    if left == right:
        return 0
    if abs(len(left) - len(right)) > limit:
        return limit + 1

    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        row_min = current[0]
        for right_index, right_char in enumerate(right, start=1):
            insertion = current[right_index - 1] + 1
            deletion = previous[right_index] + 1
            substitution = previous[right_index - 1] + (left_char != right_char)
            value = min(insertion, deletion, substitution)
            current.append(value)
            row_min = min(row_min, value)
        if row_min > limit:
            return limit + 1
        previous = current
    return previous[-1]


def _matches_object_alias(candidate_compact: str, object_type: str) -> bool:
    candidate = str(candidate_compact or '').lower()
    if not candidate:
        return False

    aliases = _object_prefix_aliases(object_type)
    if candidate in aliases:
        return True

    if len(candidate) < 5:
        return False

    for alias in aliases:
        if len(alias) < 5:
            continue
        max_distance = 1 if max(len(candidate), len(alias)) <= 6 else 2
        if _bounded_levenshtein_distance(candidate, alias, max_distance) <= max_distance:
            return True
    return False


def _strip_leading_object_prefix(object_type: str, raw_action: str) -> str:
    action_text = str(raw_action or '')
    if not action_text:
        return action_text

    for separator_match in re.finditer(r'[-_\s]+', action_text):
        prefix_candidate = action_text[:separator_match.start()]
        prefix_compact = _compact_normalized_text(prefix_candidate)
        if not prefix_compact:
            continue

        prefix_base = prefix_compact[:-3] if prefix_compact.endswith('all') else prefix_compact
        if _matches_object_alias(prefix_base, object_type):
            return action_text[separator_match.end():]

    return action_text


def _is_all_bundle_stem(stem: str, object_type: str) -> bool:
    segments = [segment for segment in re.split(r'[-_\s]+', str(stem or '')) if segment]
    if not segments:
        return False

    if len(segments) == 1:
        # CamelCase "All" or uppercase "ALL" suffix → unambiguous bundle marker.
        if stem.endswith('ALL') or stem.endswith('All'):
            return True
        compact_segment = _compact_normalized_text(segments[0])
        return compact_segment.endswith('all') and _matches_object_alias(compact_segment[:-3], object_type)

    if _compact_normalized_text(segments[-1]) != 'all':
        return False

    # Multi-segment with trailing "all" → always a bundle, regardless of prefix.
    return True


def _camel_case_action_name(raw_action: str) -> str:
    parts = [part for part in re.split(r'[^0-9A-Za-z]+', str(raw_action or '').strip()) if part]
    if not parts:
        return ''

    normalized_parts = []
    for part in parts:
        if part.isupper():
            part = part.lower()
        normalized_parts.append(part[0].upper() + part[1:])
    return ''.join(normalized_parts)


def normalize_action_name(object_type: str, raw_action: str) -> str:
    """Normalize an action name extracted from an FBX filename."""
    if not raw_action:
        return raw_action

    raw_action = _strip_leading_object_prefix(object_type, raw_action).strip()
    if not raw_action:
        return raw_action

    return _camel_case_action_name(raw_action)


def should_skip_anim(file_path: str, object_type: str) -> bool:
    """Check whether an animation file should be skipped during preprocessing."""
    stem = os.path.splitext(os.path.basename(file_path))[0]
    compact_stem = _compact_normalized_text(stem)

    if _is_all_bundle_stem(stem, object_type):
        print(
            f'  [SKIP] {os.path.basename(file_path)}: '
            f'all-in-one file (contains all animations)'
        )
        return True

    if compact_stem.endswith('nosaddle') and _matches_object_alias(compact_stem[:-8], object_type):
        print(
            f'  [SKIP] {os.path.basename(file_path)}: '
            f'NoSaddle T-pose file (no animation)'
        )
        return True

    if _matches_object_alias(compact_stem, object_type):
        print(
            f'  [SKIP] {os.path.basename(file_path)}: '
            f'no inferable action name (species name only)'
        )
        return True

    stripped_action = _strip_leading_object_prefix(object_type, stem).strip()
    compact_action = _compact_normalized_text(stripped_action)
    if compact_action in {'', 'all'}:
        print(
            f'  [SKIP] {os.path.basename(file_path)}: '
            f'no inferable action name after removing object prefix'
        )
        return True

    if compact_action in {'still', 'static'}:
        print(
            f'  [SKIP] {os.path.basename(file_path)}: '
            f'static/no-animation clip'
        )
        return True

    variant1 = re.compile(r'^[a-z]+[a-z]_\w+$', re.IGNORECASE)
    variant2 = re.compile(r'^[a-z]+_[a-z]\d+$', re.IGNORECASE)
    if variant1.match(stripped_action) or variant2.match(stripped_action):
        print(
            f'  [SKIP] {os.path.basename(file_path)}: '
            f'variant codename, no inferable action name'
        )
        return True

    stripped_compact = compact_stem
    for alias in sorted(_object_prefix_aliases(object_type), key=len, reverse=True):
        if stripped_compact.startswith(alias):
            stripped_compact = stripped_compact[len(alias):]
            break
    if re.fullmatch(r'[a-z]\d{2,}', stripped_compact, re.IGNORECASE):
        print(
            f'  [SKIP] {os.path.basename(file_path)}: '
            f'variant codename, no inferable action name'
        )
        return True

    return False


__all__ = [
    'find_tpose_reference_path',
    '_compact_normalized_text',
    '_is_all_bundle_stem',
    '_matches_object_alias',
    'normalize_action_name',
    '_should_skip_fbx',
    '_strip_leading_object_prefix',
]