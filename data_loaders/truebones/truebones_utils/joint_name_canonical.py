"""Joint-name canonicalization.

Rig prefix/suffix stripping, the Japanese romaji table and glued-compound
splitting that turn a raw bone name into its canonical form; canonical-name
assignment with collision disambiguation; the joint metadata refresh of cond
dicts. The joint texts and their T5 embeddings live in ``joint_embedding_text``.
Re-exported by ``animation_utils``.
"""

from collections import Counter, defaultdict
import json
import numpy as np
from os.path import join as pjoin
import re


################## Joint Name Canonicalization #####################

# Joint name canonicalization
_CANONICAL_NAME_PREFIXES = (
    'BN_Bip01',
    'Bip001',
    'Bip01',
    'Sabrecat',
    'NPC',
    'Rig',
    'BN',
    'jt',
    'Elk',
)
# Trailing rig suffixes stripped from joint names during canonicalization.
# Matched case-insensitively against the raw (pre-lowercased) name.
_CANONICAL_NAME_SUFFIXES = (
    'SHJnt',
)
JAPANESE_NAME_REPLACEMENTS = {
    'momo': 'Thigh',
    'sippo': 'Tail',
    'shippo': 'Tail',
    'mune': 'Chest',
    'hiza': 'Knee',
    'hara': 'Stomach',
    'ashi': 'Leg',
    'hiji': 'Elbow',
    'koshi': 'Hips',
    'kubi': 'Neck',
    'atama': 'Head',
    'ago': 'Jaw',
    'kata': 'Shoulder',
    'munabire': 'Pectoral Fin',
    'sebire': 'Dorsal Fin',
    'harabire': 'Pelvic Fin',
    'shiribire': 'Anal Fin',
    'shirihire': 'Anal Fin',
    'obire': 'Caudal Fin',
    'tai': 'Tail',
}

# Romaji tokens that mark a rig as Japanese-named. Short or ambiguous tokens
# ("te", "o") are left out so one coincidental match cannot enable the gated
# replacements below.
_JAPANESE_EVIDENCE_TOKENS = frozenset({
    'momo', 'sippo', 'shippo', 'mune', 'hiza', 'hara', 'ashi', 'hiji',
    'koshi', 'kubi', 'atama', 'ago', 'kata',
    'munabire', 'sebire', 'harabire', 'shiribire', 'shirihire', 'obire',
})
_JAPANESE_EVIDENCE_MIN_DISTINCT = 3

# Replacements applied only to a rig confirmed Japanese-named: some are too short
# to map globally ("o" = tail).
JAPANESE_GATED_REPLACEMENTS = {
    'kao': 'Head',
    'kosi': 'Hips',
    'o': 'Tail',
    'te': 'Hand',
    'era': 'Gill',
}

# Misspellings and abbreviations folded while canonicalizing, so the physics
# annotation (contacts, end effectors) reads the body part as well as the text
# does ("Lag" -> "Leg", "Eyeild" -> "Eyelid", "Dn" -> "Down").
CANONICAL_SPELLING_REPLACEMENTS = {
    'lag': 'Leg',
    'feller': 'Feeler',
    'eyeild': 'Eyelid',
    'eyei': 'Eye',
    'dn': 'Down',
}

EMBED_TEXT_HEAD_FEATURE_TOKENS = {
    'beard',
    'ear',
    'eye',
    'tongue',
}

# Species words glued onto joint names ("GorillaJaw", "MooseNeck"), stripped from
# canonical names and embedding text. Species is conditioned through
# ``species_emb``; left in, the word makes T5 cluster joints by species instead
# of body part. Joints that collide after stripping get a Variant suffix. Words
# that are also anatomy or rig vocabulary stay out ("ant" = antenna, "horse" in
# "HorseLink").
EMBED_TEXT_CREATURE_TOKENS = {
    'antilope', 'bat', 'bear', 'bee', 'boar', 'buffalo', 'buzzard', 'camel',
    'cat', 'centipede', 'chicken', 'cobra', 'coyote', 'crab', 'cricket', 'crocodile',
    'crow', 'deer', 'dinosaur', 'dog', 'donkey', 'dragon', 'eagle', 'elephant',
    'elk', 'flamingo', 'fox', 'gazelle', 'goat', 'gorilla', 'hamster', 'hen',
    'hippopotamus', 'hound', 'hyena', 'jaguar', 'kappa', 'khitan', 'leapord',
    'leopard', 'lion', 'lynx', 'mammoth', 'monkey', 'moose', 'mouse', 'ostrich',
    'parrot', 'pigeon', 'puppy', 'quilin', 'rabbit', 'raptor', 'rat', 'rhino',
    'roach', 'sabrecat', 'scorpion', 'seagull', 'serpent', 'skunk', 'spider',
    'stego', 'tarantula', 'tiger', 'trex', 'tricera', 'tukan', 'turtle', 'tyranno',
    'wyvern',
}


def joint_name_token_is_species(token):
    """Whether one normalized joint-name token is a known species label."""
    clean_token = re.sub(r'[^a-z0-9]+', '', str(token or '').casefold())
    return clean_token in EMBED_TEXT_CREATURE_TOKENS

# Vocabulary for splitting an all-lowercase glued name ("smallfrontarm") into
# words during canonicalization.
_COMPOUND_MODIFIER_TOKENS = frozenset({
    'back', 'big', 'bottom', 'down', 'first', 'fore', 'front', 'hind', 'inner',
    'large', 'left', 'long', 'low', 'lower', 'mid', 'middle', 'outer', 'outter',
    'rear', 'right', 'second', 'short', 'small', 'third', 'top', 'upper',
})
_COMPOUND_ANATOMY_TOKENS = frozenset({
    'ankle', 'arm', 'belly', 'body', 'calf', 'chest', 'claw', 'elbow', 'fat',
    'fin', 'finger', 'foot', 'forearm', 'hand', 'head', 'hoof', 'horn', 'jaw',
    'knee', 'leg', 'lip', 'neck', 'nose', 'palm', 'paw', 'spine', 'tail', 'thigh',
    'thumb', 'toe', 'tongue', 'tooth', 'wing', 'wrist',
})
_COMPOUND_SPLIT_VOCABULARY = _COMPOUND_MODIFIER_TOKENS | _COMPOUND_ANATOMY_TOKENS
# Real words that decompose into vocabulary entries and must stay whole
# ("ponytail", "eyebrow").
_COMPOUND_SPLIT_PROTECTED_TOKENS = frozenset({
    'backbone', 'collarbone', 'eyeball', 'eyebrow', 'eyelid', 'fingertip',
    'foreleg', 'headtop', 'ponytail', 'ribcage', 'toenail', 'topknot',
})
_COMPOUND_SPLIT_MIN_LENGTH = 6
_COMPOUND_SPLIT_MIN_PART_LENGTH = 3


def _split_glued_compound_token(token):
    """Segment an all-lowercase glued joint token into vocabulary words.

    Returns the parts (>= 2) when the *whole* token is covered by
    ``_COMPOUND_SPLIT_VOCABULARY``, otherwise None -- an all-or-nothing rule so
    an unknown word is never half-split into noise. Prefers the fewest parts,
    breaking ties toward the longest leading word.
    """
    if len(token) < _COMPOUND_SPLIT_MIN_LENGTH:
        return None
    if token in _COMPOUND_SPLIT_PROTECTED_TOKENS or token in _COMPOUND_SPLIT_VOCABULARY:
        return None

    best_by_start = [None] * (len(token) + 1)
    best_by_start[len(token)] = []
    for start in range(len(token) - _COMPOUND_SPLIT_MIN_PART_LENGTH, -1, -1):
        for end in range(len(token), start + _COMPOUND_SPLIT_MIN_PART_LENGTH - 1, -1):
            word = token[start:end]
            if word not in _COMPOUND_SPLIT_VOCABULARY:
                continue
            tail = best_by_start[end]
            if tail is None:
                continue
            candidate = [word] + tail
            if best_by_start[start] is None or len(candidate) < len(best_by_start[start]):
                best_by_start[start] = candidate

    parts = best_by_start[0]
    return parts if parts is not None and len(parts) >= 2 else None


def normalize_joint_name(name):
    # Split on lowercase→UPPER (e.g. "ElkRFemur" → "Elk RFemur")
    split_name = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', name)
    # Also split on UPPER→UPPER+lower (e.g. "RFemur" → "R Femur")
    split_name = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', split_name)
    split_name = re.sub(r'([A-Za-z])([0-9])', r'\1 \2', split_name)
    split_name = re.sub(r'([0-9])([A-Za-z])', r'\1 \2', split_name)
    return re.sub(r'[^a-z0-9]+', ' ', split_name.lower()).strip()


def _has_joint_name_prefix(name, prefix, *, case_sensitive=True):
    """Return whether *prefix* is one complete leading identifier token."""
    name = str(name or '')
    prefix = str(prefix or '')
    leading = name[:len(prefix)]
    prefix_matches = leading == prefix if case_sensitive else leading.casefold() == prefix.casefold()
    if not prefix or not prefix_matches:
        return False

    prefix_end = len(prefix)
    return (
        prefix_end == len(name)
        or not name[prefix_end].isalnum()
        or name[prefix_end].isupper()
        or name[prefix_end].isdigit()
    )


def infer_species_joint_name_prefixes(joint_names, species_name=None):
    """Infer a character/species prefix shared by the whole skeleton.

    Dataset identifiers commonly include a pack code (``IAC_Caveman``), while
    their bones use only the species suffix (``Caveman Pelvis``).  Generate all
    separator-preserving suffix forms and accept only the longest form that is
    a complete leading token on *every* joint.  The all-joints gate is what keeps
    an anatomical name such as ``HorseLink`` intact on an ordinary Horse rig.
    Failing that, a ``Bone_<code>`` stamp is looked for
    (``_infer_bone_code_joint_name_prefixes``).
    """
    names = [] if joint_names is None else [str(name or '') for name in joint_names]
    if not names:
        return ()
    if not species_name:
        return _infer_bone_code_joint_name_prefixes(names)

    bare_species = str(species_name).replace('\\', '/').rsplit('/', 1)[-1]
    parts = [part for part in re.split(r'[^0-9A-Za-z]+', bare_species) if part]
    candidates = set()
    for start in range(len(parts)):
        suffix = parts[start:]
        if not any(any(character.isalpha() for character in part) for part in suffix):
            continue
        candidates.update({
            ''.join(suffix),
            ' '.join(suffix),
            '_'.join(suffix),
            '-'.join(suffix),
        })

    for candidate in sorted(candidates, key=lambda value: (len(value), value), reverse=True):
        if all(
            len(name) > len(candidate)
            and _has_joint_name_prefix(name, candidate, case_sensitive=False)
            for name in names
        ):
            return (candidate,)
    return _infer_bone_code_joint_name_prefixes(names)


# Leading word some packs stamp ahead of a species or pack code on every bone
# ("Bone_Ant_L_Leg01", "Bone_KSB_Head").
_BONE_CODE_PREFIX_WORD = 'bone'


def _infer_bone_code_joint_name_prefixes(names):
    """``Bone_<code>`` shared by every joint but the rig root.

    The code is a species name or a pack abbreviation that need not spell the
    species ("Bone_SBM_Head" on a StagBeetle). A word that sits on the head, the
    legs and the tail alike tells no joint apart, so it is stripped as a prefix.
    Two guards keep a real body part: the code must not be an anatomy word
    ("Bone_Head_LM01" keeps the head its mouth codes are read against), and the
    names must still differ after it by more than side and index
    ("Bone_Eye_L_01" / "Bone_Eye_R_01" keeps its eye).
    """
    body_names = [
        name for name in names
        if not all(token == 'root' or token.isdigit() for token in normalize_joint_name(name).split())
    ]
    if len(body_names) < 2:
        return ()
    token_lists = [normalize_joint_name(name).split() for name in body_names]
    if any(len(tokens) < 3 or tokens[0] != _BONE_CODE_PREFIX_WORD for tokens in token_lists):
        return ()
    code = token_lists[0][1]
    if code.isdigit() or any(tokens[1] != code for tokens in token_lists):
        return ()
    if code in _COMPOUND_ANATOMY_TOKENS or code in EMBED_TEXT_HEAD_FEATURE_TOKENS:
        return ()
    # Side words and single letters are dropped the same way canonicalization
    # reads them, so "L"/"R" alone does not count as a second word.
    residual_words = {
        tuple(
            token for token in tokens[2:]
            if not token.isdigit() and len(token) > 1 and token not in ('left', 'right')
        )
        for tokens in token_lists
    }
    if len(residual_words) < 2:
        return ()

    prefix_pattern = re.compile(rf'{_BONE_CODE_PREFIX_WORD}[^0-9A-Za-z]*{re.escape(code)}', re.IGNORECASE)
    prefixes = set()
    for name in body_names:
        match = prefix_pattern.match(name)
        if match is None or not _has_joint_name_prefix(name, match.group(0), case_sensitive=False):
            return ()
        prefixes.add(match.group(0))
    return tuple(sorted(prefixes, key=len, reverse=True))


def strip_joint_name_prefix(name, additional_prefixes=()):
    stripped = name
    prefixes = (
        *((prefix, False) for prefix in tuple(additional_prefixes or ())),
        *((prefix, True) for prefix in _CANONICAL_NAME_PREFIXES),
    )
    for prefix, case_sensitive in sorted(prefixes, key=lambda item: len(item[0]), reverse=True):
        # Prefixes are complete rig/character tokens, not arbitrary character
        # sequences.  A following separator, digit, or CamelCase boundary is
        # valid ("Rig_Head", "Rig01", "RigHead"); a lowercase continuation is
        # not ("RightArm", "RigidBody", "Belly").  This boundary check is
        # especially important for the short Unity prefix "Rig".
        if _has_joint_name_prefix(stripped, prefix, case_sensitive=case_sensitive):
            stripped = stripped[len(prefix):]
            break
    # Strip known rig suffixes (case-insensitive), but never reduce the name
    # to an empty string (e.g. a joint literally named "SHJnt").
    for suffix in sorted(_CANONICAL_NAME_SUFFIXES, key=len, reverse=True):
        suffix_len = len(suffix)
        if len(stripped) > suffix_len and stripped[-suffix_len:].lower() == suffix.lower():
            stripped = stripped[:-suffix_len]
            break
    return stripped


def is_japanese_style_naming(joint_names):
    """True when the joint name set shows clear Japanese romaji rig naming.

    Requires at least ``_JAPANESE_EVIDENCE_MIN_DISTINCT`` distinct unambiguous
    romaji tokens so that a single coincidental match cannot trigger the gated
    Japanese-only replacements.
    """
    if not joint_names:
        return False
    seen = set()
    for name in joint_names:
        for token in normalize_joint_name(name).split():
            if token in _JAPANESE_EVIDENCE_TOKENS:
                seen.add(token)
                if len(seen) >= _JAPANESE_EVIDENCE_MIN_DISTINCT:
                    return True
    return False


def effective_canonical_replacements(joint_names):
    """Base canonical replacements, plus Japanese-only entries when warranted.

    Falls back to the shared ``JAPANESE_NAME_REPLACEMENTS`` object (no copy)
    unless the skeleton is confirmed Japanese-style, in which case the gated
    ``kao``/``kosi``/``o`` mappings are merged in.
    """
    if is_japanese_style_naming(joint_names):
        return {**JAPANESE_NAME_REPLACEMENTS, **JAPANESE_GATED_REPLACEMENTS}
    return JAPANESE_NAME_REPLACEMENTS


def _collapse_repeated_name_parts(canonical_parts):
    """Drop words a rig name repeats verbatim.

    Rigs that encode the parent path *and* the joint's own name emit the same
    words twice ("Sabrecat_HeadLeftEar_LEar_" -> "Head Left Ear Left Ear"). Two
    exact-match rules, so they can only ever remove a verbatim echo: collapse an
    adjacent duplicate, then drop a trailing block that repeats the block right
    before it.

    Numeric parts are exempt: repeated digits are two index fields that happen to
    hold the same value, not an echo. Boar's "LEFT_Ear_01_01SHJnt" is chain 01
    segment 01 -- its siblings "..._01_02" and "..._01_03" prove it -- so
    collapsing it would desync one member of a chain from the rest.

    A trailing *abbreviation* of an earlier word ("LeftThigh_LThi_") is
    deliberately left alone too: that would take a prefix heuristic, and the
    three joints it covers do not justify the risk of eating a real short word.
    """
    collapsed = []
    for part in canonical_parts:
        if collapsed and collapsed[-1] == part and not part.isdigit():
            continue
        collapsed.append(part)

    for block_length in range(2, len(collapsed) // 2 + 1):
        block = collapsed[-block_length:]
        if any(part.isdigit() for part in block):
            continue
        if block == collapsed[-2 * block_length:-block_length]:
            return collapsed[:-block_length]
    return collapsed


def _drop_species_name_parts(canonical_parts):
    return [part for part in canonical_parts if not joint_name_token_is_species(part)]


def canonicalize_joint_name(name, replacements=None, additional_prefixes=()):
    replacements = JAPANESE_NAME_REPLACEMENTS if replacements is None else replacements
    split_name = normalize_joint_name(strip_joint_name_prefix(name, additional_prefixes))
    canonical_parts = []
    for part in split_name.split():
        clean_part = re.sub(r'[^a-z0-9]+', '', part)
        if not clean_part:
            continue
        if clean_part in ('l', 'left'):
            canonical_parts.append('Left')
        elif clean_part in ('r', 'right'):
            canonical_parts.append('Right')
        elif clean_part in replacements:
            canonical_parts.append(replacements[clean_part])
        elif clean_part in CANONICAL_SPELLING_REPLACEMENTS:
            canonical_parts.append(CANONICAL_SPELLING_REPLACEMENTS[clean_part])
        elif len(clean_part) == 1:
            # Skip single letters (except digits which are preserved for disambiguation)
            if not clean_part.isdigit():
                continue
            canonical_parts.append(clean_part)
        else:
            compound_parts = _split_glued_compound_token(clean_part)
            if compound_parts is None:
                canonical_parts.append(clean_part.capitalize())
            else:
                canonical_parts.extend(part.capitalize() for part in compound_parts)

    canonical_parts = _collapse_repeated_name_parts(canonical_parts)
    canonical_parts = _drop_species_name_parts(canonical_parts)
    return ' '.join(canonical_parts) if canonical_parts else name.strip()


def collapse_solitary_head_feature_indices(canonical_joint_names):
    normalized_tokens = [normalize_joint_name(name).split() for name in canonical_joint_names]
    base_counts = Counter(
        tuple(tokens[:-1])
        for tokens in normalized_tokens
        if len(tokens) >= 2
        and tokens[-1].isdigit()
        and any(token in EMBED_TEXT_HEAD_FEATURE_TOKENS for token in tokens[:-1])
    )

    collapsed_names = []
    for name, tokens in zip(canonical_joint_names, normalized_tokens):
        if (
            len(tokens) >= 2
            and tokens[-1].isdigit()
            and any(token in EMBED_TEXT_HEAD_FEATURE_TOKENS for token in tokens[:-1])
            and base_counts[tuple(tokens[:-1])] == 1
        ):
            collapsed_names.append(' '.join(token.capitalize() for token in tokens[:-1]))
            continue
        collapsed_names.append(name)
    return collapsed_names


def canonical_name_for_bvh(name, fallback_name):
    compact_name = re.sub(r'[^0-9A-Za-z_]+', '', str(name or ''))
    if compact_name:
        return compact_name
    fallback_compact = re.sub(r'[^0-9A-Za-z_]+', '', str(fallback_name or ''))
    return fallback_compact or 'Joint'


def build_joint_name_inspection_rows(object_cond, embedding_texts):
    raw_names = list(object_cond.get('joints_names') or [])
    canonical_names = list(object_cond.get('canonical_joint_names') or raw_names)
    canonical_bvh_names = list(object_cond.get('canonical_bvh_joint_names') or canonical_names)
    side_labels = list(object_cond.get('joint_side_labels') or ['center'] * len(raw_names))
    contact_joints = {int(joint_index) for joint_index in list(object_cond.get('contact_joints') or [])}
    end_effector_joints = {int(joint_index) for joint_index in list(object_cond.get('end_effector_joints') or [])}

    inspection_rows = []
    for joint_index, raw_name in enumerate(raw_names):
        canonical_name = canonical_names[joint_index] if joint_index < len(canonical_names) else raw_name
        embedding_text = embedding_texts[joint_index] if joint_index < len(embedding_texts) else ''
        inspection_rows.append({
            'index': int(joint_index),
            'raw_name': str(raw_name),
            'canonical_name': str(canonical_name),
            'canonical_bvh_name': str(canonical_bvh_names[joint_index] if joint_index < len(canonical_bvh_names) else canonical_name),
            'embedding_text': str(embedding_text),
            'is_anatomical': bool(str(embedding_text).strip()),
            'side': str(side_labels[joint_index] if joint_index < len(side_labels) else 'center'),
            'is_contact': bool(joint_index in contact_joints),
            'is_end_effector': bool(joint_index in end_effector_joints),
        })
    return inspection_rows


def _remove_token_counts(tokens, counts_to_remove):
    remaining_counts = Counter(counts_to_remove)
    remaining_tokens = []
    for token in tokens:
        if remaining_counts.get(token, 0) > 0:
            remaining_counts[token] -= 1
            continue
        remaining_tokens.append(token)
    return remaining_tokens


def _joint_disambiguation_tokens(raw_name, canonical_name, additional_prefixes=(), translated_tokens=()):
    raw_value = str(raw_name or '')
    stripped_raw = strip_joint_name_prefix(raw_value, additional_prefixes)
    raw_tokens = normalize_joint_name(stripped_raw).split()
    canonical_tokens = normalize_joint_name(canonical_name).split()
    residual_tokens = _remove_token_counts(raw_tokens, Counter(canonical_tokens))
    # A word the canonicalizer translated is already in the canonical name under
    # its English spelling; appended again it gets translated a second time
    # (Pirrana "shiribire" -> "Anal Fin Shiribire" -> text "Anal Fin Anal Fin").
    residual_tokens = [
        token for token in residual_tokens
        if not joint_name_token_is_species(token) and token not in translated_tokens
    ]
    if raw_value.lower().startswith('jt'):
        residual_tokens.append('joint')
    return residual_tokens


def _display_disambiguation_tokens(raw_tokens):
    token_map = {
        'c': 'Center',
        'joint': 'Joint',
        'l': 'Left',
        'left': 'Left',
        'r': 'Right',
        'right': 'Right',
        'x': 'Copy',
    }
    token_priority = {
        'Copy': 0,
        'Joint': 1,
        'Left': 2,
        'Right': 3,
        'Center': 4,
    }

    display_tokens = []
    seen = set()
    for token in raw_tokens:
        display_token = token_map.get(token, token.capitalize())
        if display_token in seen:
            continue
        seen.add(display_token)
        display_tokens.append(display_token)

    display_tokens.sort(key=lambda token: (token_priority.get(token, 99), token))
    if len(display_tokens) > 1 and 'Center' in display_tokens:
        display_tokens = [token for token in display_tokens if token != 'Center']
    return display_tokens


def _disambiguate_duplicate_canonical_names(
    raw_names,
    canonical_names,
    additional_prefixes=(),
):
    updated_names = list(canonical_names)
    translated_tokens = frozenset(effective_canonical_replacements(raw_names)) | frozenset(CANONICAL_SPELLING_REPLACEMENTS)
    grouped_indices = defaultdict(list)
    for joint_index, canonical_name in enumerate(canonical_names):
        grouped_indices[str(canonical_name)].append(joint_index)

    for canonical_name, indices in grouped_indices.items():
        raw_name_set = {str(raw_names[index]) for index in indices}
        if len(indices) <= 1 or len(raw_name_set) <= 1:
            continue

        residual_token_lists = [
            _joint_disambiguation_tokens(
                raw_names[index],
                canonical_name,
                additional_prefixes,
                translated_tokens,
            )
            for index in indices
        ]
        common_counts = Counter(residual_token_lists[0])
        for tokens in residual_token_lists[1:]:
            common_counts &= Counter(tokens)

        candidate_suffixes = []
        for tokens in residual_token_lists:
            unique_tokens = _remove_token_counts(tokens, common_counts)
            candidate_suffixes.append(_display_disambiguation_tokens(unique_tokens))

        for local_index, joint_index in enumerate(indices):
            suffix_tokens = candidate_suffixes[local_index]
            if suffix_tokens:
                updated_names[joint_index] = ' '.join([str(canonical_name), *suffix_tokens])

        seen_names = set()
        duplicate_positions = []
        for local_index, joint_index in enumerate(indices):
            resolved_name = updated_names[joint_index]
            if resolved_name in seen_names:
                duplicate_positions.append(local_index)
            else:
                seen_names.add(resolved_name)

        if duplicate_positions:
            occurrence_counts = Counter()
            for local_index, joint_index in enumerate(indices):
                occurrence_counts[updated_names[joint_index]] += 1
                if occurrence_counts[updated_names[joint_index]] > 1:
                    updated_names[joint_index] = f"{updated_names[joint_index]} Variant{occurrence_counts[updated_names[joint_index]]}"

    return updated_names


def assign_canonical_joint_names(object_cond, joint_names, canonical_names):
    """Store the disambiguated canonical names plus their BVH-safe spellings.

    Both keys must derive from the *disambiguated* list. BVH bone names have to
    be unique, and canonical_name_for_bvh only strips punctuation, so deriving
    them from the raw canonicalizer output emits duplicate bones (e.g. Camel's
    front and back legs both exported as "LeftLeg01"). Single entry point so the
    preprocessing pipeline and the on-load refresh cannot drift apart.
    """
    species_prefixes = infer_species_joint_name_prefixes(
        joint_names,
        object_cond.get('species_name') or object_cond.get('object_type'),
    )
    disambiguated_names = _disambiguate_duplicate_canonical_names(
        joint_names,
        canonical_names,
        additional_prefixes=species_prefixes,
    )
    object_cond['canonical_joint_names'] = disambiguated_names
    object_cond['canonical_bvh_joint_names'] = [
        canonical_name_for_bvh(canonical_name, raw_name)
        for canonical_name, raw_name in zip(disambiguated_names, joint_names)
    ]


def collect_joint_name_collision_groups(cond):
    collision_groups = []
    for object_type in sorted(cond):
        object_cond = cond[object_type]
        raw_names = list(object_cond.get('joints_names') or [])
        canonical_names = list(object_cond.get('canonical_joint_names') or raw_names)
        canonical_bvh_names = list(object_cond.get('canonical_bvh_joint_names') or canonical_names)
        grouped_rows = defaultdict(list)

        for joint_index, raw_name in enumerate(raw_names):
            canonical_name = canonical_names[joint_index] if joint_index < len(canonical_names) else str(raw_name)
            grouped_rows[str(canonical_name)].append({
                'index': int(joint_index),
                'raw_name': str(raw_name),
                'canonical_bvh_name': str(canonical_bvh_names[joint_index] if joint_index < len(canonical_bvh_names) else canonical_name),
            })

        for canonical_name, items in grouped_rows.items():
            if len({item['raw_name'] for item in items}) <= 1:
                continue
            collision_groups.append({
                'object_type': str(object_type),
                'canonical_name': str(canonical_name),
                'rows': items,
            })
    return collision_groups


def write_joint_name_collision_report(cond, save_dir):
    collision_groups = collect_joint_name_collision_groups(cond)
    report = {
        'num_objects': int(len(cond)),
        'num_collision_groups': int(len(collision_groups)),
        'collision_groups': collision_groups,
    }
    report_path = pjoin(save_dir, 'joint_name_collision_report.json')
    with open(report_path, 'w', encoding='utf-8') as report_file:
        json.dump(report, report_file, indent=2)

    if collision_groups:
        # Looked up at call time: the warning collectors monkey-patch
        # ``animation_utils._warn``.
        from . import animation_utils as _animation_utils
        _animation_utils._warn(f'canonical joint-name collision scan found {len(collision_groups)} group(s); report: {report_path}')
        for group in collision_groups[:20]:
            raw_names = ' | '.join(row['raw_name'] for row in group['rows'])
            print(f"  - {group['object_type']}: {group['canonical_name']} <- {raw_names}")
        if len(collision_groups) > 20:
            print(f'  ... {len(collision_groups) - 20} additional group(s) omitted from console output')
    else:
        print(f'[OK] canonical joint-name collision scan found no duplicate canonical names')

    return collision_groups


def refresh_joint_metadata_in_object_cond(object_cond):
    # Imported here: physics_joint_annotation reads this module's name rules.
    from .physics_joint_annotation import build_semantic_metadata

    joint_names = list(object_cond.get('joints_names') or [])
    if not joint_names:
        return

    parents = np.asarray(object_cond.get('parents'), dtype=np.int64)
    offsets = np.asarray(object_cond.get('offsets'), dtype=np.float64)
    semantic_metadata = build_semantic_metadata(
        joint_names,
        parents,
        offsets,
        species_name=object_cond.get('species_name') or object_cond.get('object_type'),
    )
    assign_canonical_joint_names(object_cond, joint_names, semantic_metadata['canonical_joint_names'])
    object_cond['end_effector_joints'] = semantic_metadata['end_effector_joints']
    object_cond['end_effector_names'] = semantic_metadata['end_effector_names']
    object_cond['contact_joints'] = semantic_metadata['contact_joints']
    object_cond['contact_joint_names'] = semantic_metadata['contact_joint_names']
    object_cond['contact_joint_source'] = semantic_metadata['contact_joint_source']
    object_cond['joint_side_labels'] = semantic_metadata['joint_side_labels']
    object_cond['symmetry_partner_indices'] = semantic_metadata['symmetry_partner_indices']
    object_cond['symmetric_joint_pairs'] = semantic_metadata['symmetric_joint_pairs']
    object_cond['symmetric_joint_pair_names'] = semantic_metadata['symmetric_joint_pair_names']
    object_cond['is_symmetric'] = semantic_metadata['is_symmetric']


def refresh_joint_metadata_in_cond_dict(cond_dict):
    if not isinstance(cond_dict, dict):
        return cond_dict

    for object_cond in cond_dict.values():
        if isinstance(object_cond, dict):
            refresh_joint_metadata_in_object_cond(object_cond)
    return cond_dict
