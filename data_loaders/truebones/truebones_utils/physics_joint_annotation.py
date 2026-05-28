"""End effector detection and symmetry analysis utilities."""

from collections import Counter

import numpy as np
import re


# End effector joint detection tokens
_END_EFFECTOR_DISTAL_TOKENS = (
    'toe',
    'foot',
    'hoof',
    'paw',
    'phalanx',
    'claw',
    'finger',
    'thumb',
    'hand',
    'leg',
)
_END_EFFECTOR_TAIL_TOKENS = (
    'tail',
    'sippo',
    'tai',
)
_END_EFFECTOR_HEAD_TOKENS = (
    'head',
    'jaw',
    'mouth',
    'nose',
    'snout',
    'muzzle',
    'beak',
    'tongue',
    'mandible',
    'fang',
    'chin',
)
_END_EFFECTOR_APPENDAGE_TOKENS = (
    'wing',
    'forearm',
    'clip',
    'pincer',
    'plier',
    'feeler',
    'antenna',
    'horn',
    'spike',
)
_END_EFFECTOR_EXCLUDE_TOKENS = (
    'jiggle',
    'twist',
    'hair',
    'fur',
    'beard',
    'eyebrow',
    'eyelid',
    'eyeball',
    'eye',
    'ear',
    'lip',
    'saddle',
    'halter',
    'reins',
    'handle',
    'trajectory',
    'projectile',
    'magic',
    'mesh',
    'ik',
    'chain',
    'xtra',
    'extra',
    'ponytail',
    'body',
    'spine',
    'shell',
    'center',
    'mascara',
)

# Contact joint detection tokens
_CONTACT_JOINT_KEYWORDS = (
    'toe',
    'foot',
    'hoof',
    'phalanx',
    'ashi',
    'ankle',
    'heel',
    'paw',
)
_CONTACT_JOINT_CONTEXT_KEYWORDS = _CONTACT_JOINT_KEYWORDS + (
    'leg',
)
_CONTACT_JOINT_UPPER_LIMB_TOKENS = (
    'hand',
    'finger',
    'thumb',
    'arm',
    'wrist',
    'elbow',
    'forearm',
    'shoulder',
    'wing',
)
_CONTACT_JOINT_WEAK_KEYWORDS = (
    'leg',
)
_CONTACT_GEOMETRY_DISTAL_TOKENS = (
    'toe',
    'foot',
    'hoof',
    'paw',
    'phalanx',
    'claw',
    'finger',
    'thumb',
    'hand',
    'leg',
)
_CONTACT_CHAIN_STOP_TOKENS = (
    'hip',
    'hips',
    'pelvis',
    'root',
    'cog',
    'spine',
    'chest',
    'thigh',
    'knee',
    'upperleg',
    'upleg',
    'neck',
    'head',
    'tail',
    'jaw',
    'body',
)
_CONTACT_CHAIN_INCLUDE_TOKENS = (
    'toe',
    'foot',
    'hoof',
    'paw',
    'phalanx',
    'claw',
    'finger',
    'thumb',
    'hand',
    'palm',
    'ball',
    'ankle',
    'wrist',
)
_CONTACT_PARENT_OFFSET_RATIO = 0.22
_CONTACT_PARENT_OFFSET_MIN = 0.10
_CONTACT_PARENT_OFFSET_CAP = 0.20
_CONTACT_CUMULATIVE_OFFSET_RATIO = 0.44
_CONTACT_CUMULATIVE_OFFSET_MIN = 0.15
_CONTACT_CUMULATIVE_OFFSET_CAP = 0.34

# Joint name canonicalization
_CANONICAL_NAME_PREFIXES = (
    'BN_Bip01',
    'Bip01',
    'Sabrecat',
    'NPC',
    'BN',
    'jt',
    'Elk',
)
_CANONICAL_NAME_REPLACEMENTS = {
    'momo': 'Thigh',
    'sippo': 'Tail',
    'mune': 'Chest',
    'hiza': 'Knee',
    'hara': 'Stomach',
    'ashi': 'Leg',
    'hiji': 'Elbow',
    'koshi': 'Hips',
    'te': 'Hand',
    'kubi': 'Neck',
    'atama': 'Head',
    'ago': 'Jaw',
    'kata': 'Shoulder',
    'tai': 'Tail',
}
_EMBED_TEXT_SKIP_TOKENS = {
    'mid',
    'rear',
    'front',
    'back',
    'base',
    'tip',
    'nub',
    'end',
    'site',
}
_EMBED_TEXT_NON_ANATOMICAL_TOKENS = {
    'brain',
    'center',
    'copy',
    'cog',
    'dummy',
    'fur',
    'ik',
    'joint',
    'locator',
    'mesh',
    'node',
    'ponytail',
    'projectile',
    'trajectory',
}
_EMBED_TEXT_HEAD_FEATURE_TOKENS = {
    'beard',
    'ear',
    'eye',
    'tongue',
}

JOINT_NAME_EMBEDDING_SCHEMA_VERSION = 7

_CHAIN_INDEX_ORDINAL_TOKENS = {
    1: 'First',
    2: 'Second',
    3: 'Third',
    4: 'Fourth',
    5: 'Fifth',
    6: 'Sixth',
    7: 'Seventh',
    8: 'Eighth',
    9: 'Ninth',
    10: 'Tenth',
}
_SPECIES_LINEAGE_TAGS = {
    'Alligator': ('Reptile', 'Crocodilian'),
    'Anaconda': ('Reptile', 'Snake'),
    'Ant': ('Arthropod', 'Insect'),
    'Bat': ('Flying', 'Mammal'),
    'Bear': ('Mammal', 'Ursid'),
    'Bird': ('Flying', 'Bird'),
    'BrownBear': ('Mammal', 'Ursid'),
    'Buffalo': ('Mammal', 'Bovid'),
    'Buzzard': ('Flying', 'Bird'),
    'Camel': ('Mammal', 'Megafauna'),
    'Cat': ('Mammal', 'Felid'),
    'Centipede': ('Arthropod', 'Myriapod'),
    'Chicken': ('Bird', 'Biped'),
    'Comodoa': ('Reptile', 'Lizard'),
    'Coyote': ('Mammal', 'Canid'),
    'Crab': ('Arthropod', 'Crustacean'),
    'Cricket': ('Arthropod', 'Insect'),
    'Crocodile': ('Reptile', 'Crocodilian'),
    'Deer': ('Mammal', 'Cervid'),
    'Dragon': ('Flying', 'Reptile'),
    'Eagle': ('Flying', 'Bird'),
    'Elephant': ('Mammal', 'Proboscidean'),
    'FireAnt': ('Arthropod', 'Insect'),
    'Flamingo': ('Bird', 'Biped'),
    'Fox': ('Mammal', 'Canid'),
    'Gazelle': ('Mammal', 'Bovid'),
    'Giantbee': ('Flying', 'Insect'),
    'Goat': ('Mammal', 'Bovid'),
    'Hamster': ('Mammal', 'Rodent'),
    'HermitCrab': ('Arthropod', 'Crustacean'),
    'Hippopotamus': ('Mammal', 'Megafauna'),
    'Horse': ('Mammal', 'Megafauna'),
    'Hound': ('Mammal', 'Canid'),
    'Isopetra': ('Arthropod', 'Myriapod'),
    'Jaguar': ('Mammal', 'Felid'),
    'KingCobra': ('Reptile', 'Snake'),
    'Leapord': ('Mammal', 'Felid'),
    'Lion': ('Mammal', 'Felid'),
    'Lynx': ('Mammal', 'Felid'),
    'Mammoth': ('Mammal', 'Proboscidean'),
    'Ostrich': ('Bird', 'Biped'),
    'Parrot': ('Flying', 'Bird'),
    'Parrot2': ('Flying', 'Bird'),
    'Pigeon': ('Flying', 'Bird'),
    'PolarBear': ('Mammal', 'Ursid'),
    'PolarBearB': ('Mammal', 'Ursid'),
    'Pteranodon': ('Reptile', 'Pterosaur'),
    'Puppy': ('Mammal', 'Canid'),
    'Raindeer': ('Mammal', 'Cervid'),
    'Raptor': ('Reptile', 'Dinosaur'),
    'Raptor2': ('Reptile', 'Dinosaur'),
    'Raptor3': ('Reptile', 'Dinosaur'),
    'Rat': ('Mammal', 'Rodent'),
    'Rhino': ('Mammal', 'Megafauna'),
    'Roach': ('Arthropod', 'Insect'),
    'SabreToothTiger': ('Mammal', 'Felid'),
    'SandMouse': ('Mammal', 'Felid'),
    'Scorpion': ('Arthropod', 'Arachnid'),
    'Scorpion-2': ('Arthropod', 'Arachnid'),
    'Spider': ('Arthropod', 'Arachnid'),
    'SpiderG': ('Arthropod', 'Arachnid'),
    'Stego': ('Reptile', 'Dinosaur'),
    'Trex': ('Reptile', 'Dinosaur'),
    'Tricera': ('Reptile', 'Dinosaur'),
    'Tukan': ('Flying', 'Bird'),
    'Turtle': ('Reptile', 'Chelonian'),
    'Tyranno': ('Reptile', 'Dinosaur'),
}


def normalize_joint_name(name):
    # Split on lowercase→UPPER (e.g. "ElkRFemur" → "Elk RFemur")
    split_name = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', name)
    # Also split on UPPER→UPPER+lower (e.g. "RFemur" → "R Femur")
    split_name = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', split_name)
    split_name = re.sub(r'([A-Za-z])([0-9])', r'\1 \2', split_name)
    split_name = re.sub(r'([0-9])([A-Za-z])', r'\1 \2', split_name)
    return re.sub(r'[^a-z0-9]+', ' ', split_name.lower()).strip()


def strip_joint_name_prefix(name):
    stripped = name
    for prefix in sorted(_CANONICAL_NAME_PREFIXES, key=len, reverse=True):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    return stripped


def _canonicalize_joint_name(name):
    split_name = normalize_joint_name(strip_joint_name_prefix(name))
    canonical_parts = []
    for part in split_name.split():
        clean_part = re.sub(r'[^a-z0-9]+', '', part)
        if not clean_part:
            continue
        if clean_part in ('l', 'left'):
            canonical_parts.append('Left')
        elif clean_part in ('r', 'right'):
            canonical_parts.append('Right')
        elif clean_part in _CANONICAL_NAME_REPLACEMENTS:
            canonical_parts.append(_CANONICAL_NAME_REPLACEMENTS[clean_part])
        elif len(clean_part) == 1:
            # Skip single letters (except digits which are preserved for disambiguation)
            if not clean_part.isdigit():
                continue
            canonical_parts.append(clean_part)
        else:
            canonical_parts.append(clean_part.capitalize())
    return ' '.join(canonical_parts) if canonical_parts else name.strip()


def _titlecase_identifier_tokens(value):
    normalized = normalize_joint_name(str(value))
    if not normalized:
        return []
    return [token.capitalize() for token in normalized.split() if token]


def _collapse_solitary_head_feature_indices(canonical_joint_names):
    normalized_tokens = [normalize_joint_name(name).split() for name in canonical_joint_names]
    base_counts = Counter(
        tuple(tokens[:-1])
        for tokens in normalized_tokens
        if len(tokens) >= 2
        and tokens[-1].isdigit()
        and any(token in _EMBED_TEXT_HEAD_FEATURE_TOKENS for token in tokens[:-1])
    )

    collapsed_names = []
    for name, tokens in zip(canonical_joint_names, normalized_tokens):
        if (
            len(tokens) >= 2
            and tokens[-1].isdigit()
            and any(token in _EMBED_TEXT_HEAD_FEATURE_TOKENS for token in tokens[:-1])
            and base_counts[tuple(tokens[:-1])] == 1
        ):
            collapsed_names.append(' '.join(token.capitalize() for token in tokens[:-1]))
            continue
        collapsed_names.append(name)
    return collapsed_names


def _species_lineage_tokens(object_cond):
    object_type = str(object_cond.get('object_type') or '').strip()
    return list(_SPECIES_LINEAGE_TAGS.get(object_type, ()))


def _refine_joint_embedding_name(name):
    canonical_name = _canonicalize_joint_name(name)
    refined_tokens = []
    for token in canonical_name.split():
        clean_token = re.sub(r'[^a-z0-9]+', '', token.lower())
        clean_token = re.sub(r'\d+$', '', clean_token)
        if not clean_token or clean_token.isdigit() or clean_token in _EMBED_TEXT_SKIP_TOKENS:
            continue
        if clean_token in _EMBED_TEXT_NON_ANATOMICAL_TOKENS:
            continue
        if clean_token in ('sippo', 'tai') or clean_token.startswith('tail'):
            refined_tokens.append('Tail')
        elif clean_token.startswith('toe'):
            refined_tokens.append('Toe')
        elif clean_token.startswith('finger'):
            refined_tokens.append('Finger')
        elif clean_token == 'arm':
            refined_tokens.append('UpperArm')
        elif clean_token in ('fore', 'forearm'):
            refined_tokens.append('Forearm')
        elif clean_token == 'upleg':
            refined_tokens.append('UpperLeg')
        elif clean_token == 'clip':
            refined_tokens.append('Appendage')
        elif clean_token in _EMBED_TEXT_HEAD_FEATURE_TOKENS:
            refined_tokens.append('HeadFeature')
        else:
            refined_tokens.append(clean_token.capitalize())

    merged_tokens = []
    index = 0
    while index < len(refined_tokens):
        pair = tuple(token.lower() for token in refined_tokens[index:index + 2])
        if pair in (('upper', 'leg'), ('up', 'leg')):
            merged_tokens.append('Thigh')
            index += 2
            continue
        if pair == ('fore', 'arm'):
            merged_tokens.append('Forearm')
            index += 2
            continue
        if pair == ('upper', 'arm'):
            merged_tokens.append('UpperArm')
            index += 2
            continue
        merged_tokens.append(refined_tokens[index])
        index += 1

    return merged_tokens or canonical_name.split()


def _chain_index_token(index):
    index = int(index)
    return _CHAIN_INDEX_ORDINAL_TOKENS.get(index, f'Index{index}')


def _chain_role_token(chain_index, chain_length):
    chain_index = int(chain_index)
    chain_length = int(chain_length)
    if chain_length <= 1:
        return None
    if chain_index <= 1:
        return 'ChainStart'
    if chain_index >= chain_length:
        return 'ChainEnd'
    relative_position = float(chain_index - 1) / float(max(chain_length - 1, 1))
    if relative_position <= 0.34:
        return 'ChainEarly'
    if relative_position >= 0.67:
        return 'ChainLate'
    return 'ChainMiddle'


def _build_chain_relative_joint_tokens(refined_tokens_per_joint, parents):
    joint_count = len(refined_tokens_per_joint)
    if parents is None or len(parents) != joint_count:
        return [[] for _ in range(joint_count)]

    parents = np.asarray(parents, dtype=np.int64)
    children = _child_map(parents)
    signatures = [tuple(tokens) for tokens in refined_tokens_per_joint]
    upward_steps = np.zeros(joint_count, dtype=np.int32)
    downward_steps = np.zeros(joint_count, dtype=np.int32)

    for joint_index in range(joint_count):
        parent_index = int(parents[joint_index])
        if parent_index >= 0 and signatures[parent_index] and signatures[parent_index] == signatures[joint_index]:
            upward_steps[joint_index] = upward_steps[parent_index] + 1

    for joint_index in range(joint_count - 1, -1, -1):
        matching_children = [
            child_index
            for child_index in children[joint_index]
            if signatures[joint_index] and signatures[child_index] == signatures[joint_index]
        ]
        if matching_children:
            downward_steps[joint_index] = 1 + max(downward_steps[child_index] for child_index in matching_children)

    chain_lengths = upward_steps + downward_steps + 1
    chain_tokens = []
    for joint_index in range(joint_count):
        signature = signatures[joint_index]
        chain_length = int(chain_lengths[joint_index])
        if not signature or chain_length <= 1:
            chain_tokens.append([])
            continue

        chain_index = int(upward_steps[joint_index]) + 1
        joint_tokens = ['Segment', _chain_index_token(chain_index), 'Of', str(chain_length)]
        role_token = _chain_role_token(chain_index, chain_length)
        if role_token is not None:
            joint_tokens.append(role_token)
        chain_tokens.append(joint_tokens)

    return chain_tokens


def build_joint_embedding_texts(object_cond):
    base_joint_names = object_cond.get('canonical_joint_names') or object_cond.get('joints_names') or []
    if not base_joint_names:
        return []

    lineage_tokens = _species_lineage_tokens(object_cond)
    joint_side_labels = list(object_cond.get('joint_side_labels') or ['center'] * len(base_joint_names))
    contact_joints = {int(joint_index) for joint_index in list(object_cond.get('contact_joints') or [])}
    end_effector_joints = {int(joint_index) for joint_index in list(object_cond.get('end_effector_joints') or [])}
    refined_tokens_per_joint = [_refine_joint_embedding_name(joint_name) for joint_name in base_joint_names]
    chain_relative_tokens = _build_chain_relative_joint_tokens(refined_tokens_per_joint, object_cond.get('parents'))

    texts = []
    for joint_index, joint_name in enumerate(base_joint_names):
        refined_tokens = refined_tokens_per_joint[joint_index]
        lowered_tokens = {token.lower() for token in refined_tokens}
        if lowered_tokens & _EMBED_TEXT_NON_ANATOMICAL_TOKENS:
            texts.append('')
            continue

        semantic_tokens = list()
        semantic_tokens.extend(lineage_tokens)
        semantic_tokens.extend(refined_tokens)
        semantic_tokens.extend(chain_relative_tokens[joint_index])

        side = joint_side_labels[joint_index] if joint_index < len(joint_side_labels) else 'center'
        if side in ('left', 'right'):
            semantic_tokens.append(side.capitalize())
        if joint_index in contact_joints:
            semantic_tokens.append('Contact')
        if joint_index in end_effector_joints:
            semantic_tokens.append('EndEffector')
        texts.append(' '.join(semantic_tokens))

    return texts


def _joint_signature(name):
    signature_tokens = [
        token for token in _canonicalize_joint_name(name).lower().split()
        if token not in ('left', 'right', 'lf', 'rf')
    ]
    if signature_tokens:
        return ' '.join(signature_tokens)

    fallback_tokens = [
        token for token in normalize_joint_name(name).split()
        if token not in ('left', 'right', 'l', 'r', 'lf', 'rf')
    ]
    return ' '.join(fallback_tokens)


def _fallback_child_signature(name):
    return ' '.join(
        token for token in _joint_signature(name).split()
        if not token.isdigit()
    )


def _joint_semantic_text(name):
    normalized = normalize_joint_name(name)
    canonical = _canonicalize_joint_name(name).lower()
    return f'{normalized} {canonical}'.strip()


def _text_matches_keywords(text, keywords):
    return any(keyword in text for keyword in keywords)


def _joint_family_semantic_text(joint_index, joint_names, parents, max_depth=3):
    semantic_chunks = []
    current_index = int(joint_index)
    depth = 0
    while current_index >= 0 and depth <= max_depth:
        semantic_chunks.append(_joint_semantic_text(joint_names[current_index]))
        current_index = int(parents[current_index])
        depth += 1
    return ' '.join(chunk for chunk in semantic_chunks if chunk)


def _is_informative_joint_name(name):
    normalized = normalize_joint_name(name)
    if not normalized:
        return False
    tokens = [token for token in normalized.split() if token]
    return any(len(token) > 1 for token in tokens)


def _child_map(parents):
    children = [[] for _ in range(len(parents))]
    for joint_index, parent_index in enumerate(parents):
        if parent_index >= 0:
            children[parent_index].append(joint_index)
    return children


def _select_representative_joint(indices, rest_positions, axis, prefer_max=True):
    if not indices:
        return None
    if rest_positions is None or len(rest_positions) <= max(indices):
        return indices[0]

    direction = 1.0 if prefer_max else -1.0
    return max(
        indices,
        key=lambda joint_index: (
            direction * float(rest_positions[joint_index, axis]),
            float(np.linalg.norm(rest_positions[joint_index])),
            -joint_index,
        ),
    )


def _filter_grounded_joint_indices(candidate_indices, rest_positions, margin_ratio=0.18):
    if len(candidate_indices) == 0 or len(rest_positions) == 0:
        return []

    unique_candidates = sorted({int(joint_index) for joint_index in candidate_indices})
    body_height = max(float(np.ptp(rest_positions[:, 1])), 1e-6)
    ground_margin = max(body_height * margin_ratio, 1e-3)
    ground_level = float(np.min(rest_positions[unique_candidates, 1]))
    return [
        joint_index
        for joint_index in unique_candidates
        if rest_positions[joint_index, 1] <= ground_level + ground_margin
    ]


def _expand_grounded_contact_chain(candidate_indices, grounded_indices, parents, rest_positions, margin_ratio=0.2):
    if not grounded_indices:
        return []

    candidate_set = {int(joint_index) for joint_index in candidate_indices}
    expanded = set(int(joint_index) for joint_index in grounded_indices)
    body_height = max(float(np.ptp(rest_positions[:, 1])), 1e-6)
    parent_margin = max(body_height * margin_ratio, 1e-3)
    frontier = list(expanded)

    while frontier:
        joint_index = frontier.pop()
        parent_index = int(parents[joint_index])
        if parent_index < 0 or parent_index not in candidate_set or parent_index in expanded:
            continue
        if abs(float(rest_positions[parent_index, 1] - rest_positions[joint_index, 1])) > parent_margin:
            continue
        expanded.add(parent_index)
        frontier.append(parent_index)

    return sorted(expanded)


def _select_grounded_contact_end_effectors(candidate_indices, joint_names, parents, rest_positions):
    if len(candidate_indices) == 0:
        return []

    candidate_indices = sorted({int(joint_index) for joint_index in candidate_indices})
    body_height = max(float(np.ptp(rest_positions[:, 1])), 1e-6)
    pair_height_margin = max(body_height * 0.24, 1e-3)
    single_height_margin = max(body_height * 0.18, 1e-3)

    _, symmetry_partner_indices, _ = _infer_symmetry_metadata(joint_names, parents, rest_positions)
    paired_groups = []
    paired_joint_indices = set()

    for joint_index in candidate_indices:
        partner_index = int(symmetry_partner_indices[joint_index])
        if partner_index < 0 or partner_index not in candidate_indices or joint_index >= partner_index:
            continue
        paired_groups.append((
            float((rest_positions[joint_index, 1] + rest_positions[partner_index, 1]) / 2.0),
            joint_index,
            partner_index,
        ))
        paired_joint_indices.add(joint_index)
        paired_joint_indices.add(partner_index)

    selected = set()
    if paired_groups:
        min_pair_height = min(group[0] for group in paired_groups)
        for pair_height, left_index, right_index in paired_groups:
            if pair_height <= min_pair_height + pair_height_margin:
                selected.add(left_index)
                selected.add(right_index)

    if not selected:
        min_height = float(np.min(rest_positions[candidate_indices, 1]))
        for joint_index in candidate_indices:
            if rest_positions[joint_index, 1] <= min_height + single_height_margin:
                selected.add(joint_index)

    for joint_index in candidate_indices:
        if joint_index in paired_joint_indices:
            continue
        if rest_positions[joint_index, 1] <= min(float(rest_positions[index, 1]) for index in selected) + single_height_margin:
            selected.add(joint_index)

    return sorted(selected)


def _expand_contact_chain_from_leaves(leaf_indices, joint_names, parents, rest_positions, max_depth=4):
    if not leaf_indices:
        return []

    body_height = max(float(np.ptp(rest_positions[:, 1])), 1e-6)
    chain_margin = max(body_height * 0.2, 1e-3)
    # Cap support-joint backfilling when the parent-child bone itself is too long.
    # This keeps obvious mid-limb transport bones such as Calf/HorseLink from being
    # mislabeled as direct contact points, while still allowing short foot/hand/palm
    # support bones to remain in the contact chain.
    max_parent_contact_offset = min(
        max(body_height * _CONTACT_PARENT_OFFSET_RATIO, _CONTACT_PARENT_OFFSET_MIN),
        _CONTACT_PARENT_OFFSET_CAP,
    )
    # Also cap the cumulative distance from the terminal contact leaf. Even when
    # every individual bone is short, a long multi-bone chain should not turn a
    # clearly upstream support joint into a direct contact point.
    max_cumulative_contact_offset = min(
        max(body_height * _CONTACT_CUMULATIVE_OFFSET_RATIO, _CONTACT_CUMULATIVE_OFFSET_MIN),
        _CONTACT_CUMULATIVE_OFFSET_CAP,
    )
    expanded = set(int(joint_index) for joint_index in leaf_indices)

    for joint_index in leaf_indices:
        current_index = int(joint_index)
        cumulative_contact_offset = 0.0
        for _ in range(max_depth):
            parent_index = int(parents[current_index])
            if parent_index < 0:
                break
            parent_text = _joint_semantic_text(joint_names[parent_index])
            if _text_matches_keywords(parent_text, _CONTACT_CHAIN_STOP_TOKENS):
                break
            if not _text_matches_keywords(parent_text, _CONTACT_CHAIN_INCLUDE_TOKENS):
                break
            parent_contact_offset = float(np.linalg.norm(rest_positions[parent_index] - rest_positions[current_index]))
            if parent_contact_offset > max_parent_contact_offset:
                break
            cumulative_contact_offset += parent_contact_offset
            if cumulative_contact_offset > max_cumulative_contact_offset:
                break
            if abs(float(rest_positions[parent_index, 1] - rest_positions[current_index, 1])) > chain_margin:
                break
            expanded.add(parent_index)
            current_index = parent_index

    return sorted(expanded)


def _infer_contact_leaf_candidates(parents, joint_names, rest_positions):
    end_effectors = _infer_end_effector_joints(parents, joint_names=joint_names, rest_positions=rest_positions)
    return [
        joint_index
        for joint_index in end_effectors
        if _text_matches_keywords(_joint_semantic_text(joint_names[joint_index]), _CONTACT_GEOMETRY_DISTAL_TOKENS)
    ]


def rest_positions_from_offsets(offsets, parents):
    offsets = np.asarray(offsets, dtype=np.float64)
    rest_positions = np.zeros_like(offsets, dtype=np.float64)
    for joint_index, parent_index in enumerate(parents):
        if parent_index >= 0:
            rest_positions[joint_index] = rest_positions[parent_index] + offsets[joint_index]
    return rest_positions


def _infer_end_effector_joints(parents, joint_names=None, rest_positions=None):
    children = _child_map(parents)
    leaf_joints = [joint_index for joint_index, child_indices in enumerate(children) if not child_indices]
    if joint_names is None:
        return leaf_joints

    distal_joints = []
    tail_joints = []
    head_joints = []
    appendage_joints = []
    filtered_leaf_joints = []

    for joint_index in leaf_joints:
        semantic_text = _joint_semantic_text(joint_names[joint_index])
        if not _is_informative_joint_name(joint_names[joint_index]):
            continue
        if _text_matches_keywords(semantic_text, _END_EFFECTOR_EXCLUDE_TOKENS):
            continue

        filtered_leaf_joints.append(joint_index)
        if _text_matches_keywords(semantic_text, _END_EFFECTOR_DISTAL_TOKENS):
            distal_joints.append(joint_index)
        elif _text_matches_keywords(semantic_text, _END_EFFECTOR_TAIL_TOKENS):
            tail_joints.append(joint_index)
        elif _text_matches_keywords(semantic_text, _END_EFFECTOR_HEAD_TOKENS):
            head_joints.append(joint_index)
        elif _text_matches_keywords(semantic_text, _END_EFFECTOR_APPENDAGE_TOKENS):
            appendage_joints.append(joint_index)

    semantic_end_effectors = set(distal_joints)
    semantic_end_effectors.update(appendage_joints)

    tail_joint = _select_representative_joint(tail_joints, rest_positions, axis=2, prefer_max=False)
    if tail_joint is not None:
        semantic_end_effectors.add(tail_joint)

    head_joint = _select_representative_joint(head_joints, rest_positions, axis=2, prefer_max=True)
    if head_joint is not None:
        semantic_end_effectors.add(head_joint)

    if semantic_end_effectors:
        return sorted(semantic_end_effectors)
    if filtered_leaf_joints:
        return sorted(filtered_leaf_joints)
    return leaf_joints


def _infer_contact_joints_from_names(joint_names, parents, rest_positions):
    strong_candidates = []
    weak_candidates = []
    children = _child_map(parents)

    for joint_index, joint_name in enumerate(joint_names):
        semantic_text = _joint_semantic_text(joint_name)
        family_text = _joint_family_semantic_text(joint_index, joint_names, parents, max_depth=3)
        has_upper_limb_context = _text_matches_keywords(family_text, _CONTACT_JOINT_UPPER_LIMB_TOKENS)
        has_lower_limb_context = _text_matches_keywords(family_text, _CONTACT_JOINT_CONTEXT_KEYWORDS)

        is_strong_contact = _text_matches_keywords(semantic_text, _CONTACT_JOINT_KEYWORDS)
        is_ball_contact = 'ball' in semantic_text and has_lower_limb_context and not has_upper_limb_context
        is_claw_contact = 'claw' in semantic_text and has_lower_limb_context and not has_upper_limb_context
        is_end_site_contact = (
            ('nub' in semantic_text or 'end site' in semantic_text)
            and has_lower_limb_context
            and not has_upper_limb_context
        )

        if is_strong_contact or is_ball_contact or is_claw_contact or is_end_site_contact:
            strong_candidates.append(joint_index)
            continue

        if not children[joint_index] and not has_upper_limb_context and _text_matches_keywords(semantic_text, _CONTACT_JOINT_WEAK_KEYWORDS):
            weak_candidates.append(joint_index)

    grounded_candidates = _filter_grounded_joint_indices(strong_candidates, rest_positions, margin_ratio=0.24)
    if grounded_candidates:
        return _expand_grounded_contact_chain(strong_candidates, grounded_candidates, parents, rest_positions)

    grounded_weak_candidates = _filter_grounded_joint_indices(weak_candidates, rest_positions, margin_ratio=0.24)
    if grounded_weak_candidates:
        return grounded_weak_candidates

    return []


def _infer_contact_joints_from_geometry(joint_names, rest_positions, parents):
    if len(rest_positions) == 0:
        return []

    candidates = _infer_contact_leaf_candidates(parents, joint_names, rest_positions)
    if not candidates:
        return []

    grounded_leaves = _select_grounded_contact_end_effectors(candidates, joint_names, parents, rest_positions)
    if not grounded_leaves:
        return []

    return _expand_contact_chain_from_leaves(grounded_leaves, joint_names, parents, rest_positions)


def infer_contact_joints(joint_names, parents, rest_positions):
    contact_joints = _infer_contact_joints_from_geometry(joint_names, rest_positions, parents)
    if contact_joints:
        return contact_joints, 'geometry'

    contact_joints = _infer_contact_joints_from_names(joint_names, parents, rest_positions)
    if contact_joints:
        return contact_joints, 'names'

    return _infer_end_effector_joints(parents, joint_names=joint_names, rest_positions=rest_positions), 'end_effectors'


def _joint_depths(parents):
    depths = [0] * len(parents)
    for joint_index in range(1, len(parents)):
        parent_index = parents[joint_index]
        if parent_index >= 0:
            depths[joint_index] = depths[parent_index] + 1
    return depths


def detect_joint_side(name):
    normalized = normalize_joint_name(name)
    compact = normalized.replace(' ', '')
    tokens = normalized.split()
    right_markers = (
        ' right ',
        ' npc r',
        ' bip01 r',
        ' bn r',
        ' r ',
        ' r_',
        ' rleg',
        ' rarm',
        ' rthigh',
        ' rclavicle',
        ' rupperarm',
        ' r momo',
        ' r kata',
        ' r hiji',
    )
    left_markers = (
        ' left ',
        ' npc l',
        ' bip01 l',
        ' bn l',
        ' l ',
        ' l_',
        ' lleg',
        ' larm',
        ' lthigh',
        ' lclavicle',
        ' lupperarm',
        ' l momo',
        ' l kata',
        ' l hiji',
    )
    padded = f' {normalized} '
    if any(marker in padded for marker in right_markers) or compact.startswith(('r_', 'rleg', 'rarm', 'rthigh', 'rmomo', 'rkata', 'rhiji')):
        return 'right'
    if any(marker in padded for marker in left_markers) or compact.startswith(('l_', 'lleg', 'larm', 'lthigh', 'lmomo', 'lkata', 'lhiji')):
        return 'left'

    has_rf = 'rf' in tokens
    has_lf = 'lf' in tokens
    if has_rf and not has_lf:
        return 'right'
    if has_lf and not has_rf:
        return 'left'
    return None


def _symmetry_pair_score(left_index, right_index, rest_positions, depths, parents, joint_names):
    mirror_error = abs(float(rest_positions[left_index, 0] + rest_positions[right_index, 0]))
    yz_error = float(np.linalg.norm(rest_positions[left_index, 1:] - rest_positions[right_index, 1:]))
    depth_error = abs(depths[left_index] - depths[right_index])

    left_parent = parents[left_index]
    right_parent = parents[right_index]
    left_parent_sig = _joint_signature(joint_names[left_parent]) if left_parent >= 0 else ''
    right_parent_sig = _joint_signature(joint_names[right_parent]) if right_parent >= 0 else ''
    parent_penalty = 0 if left_parent_sig == right_parent_sig else 1
    return parent_penalty, depth_error, mirror_error + yz_error, left_index, right_index


def _local_mirror_error(left_index, right_index, left_parent, right_parent, rest_positions):
    left_anchor = rest_positions[left_parent] if left_parent >= 0 else np.zeros(3, dtype=np.float64)
    right_anchor = rest_positions[right_parent] if right_parent >= 0 else np.zeros(3, dtype=np.float64)
    left_delta = rest_positions[left_index] - left_anchor
    right_delta = rest_positions[right_index] - right_anchor
    mirror_error = abs(float(left_delta[0] + right_delta[0]))
    yz_error = float(np.linalg.norm(left_delta[1:] - right_delta[1:]))
    local_scale = max(float(np.linalg.norm(left_delta)), float(np.linalg.norm(right_delta)), 1e-6)
    return mirror_error, yz_error, local_scale


def _passes_conservative_child_mirror_check(left_index, right_index, left_parent, right_parent, rest_positions):
    mirror_error, yz_error, local_scale = _local_mirror_error(
        left_index,
        right_index,
        left_parent,
        right_parent,
        rest_positions,
    )
    tolerance = max(1e-3, local_scale * 0.6)
    return mirror_error <= tolerance and yz_error <= tolerance


def _infer_symmetry_metadata(joint_names, parents, rest_positions, return_details=False):
    depths = _joint_depths(parents)
    joint_side_labels = []
    grouped_indices = {}

    for joint_index, joint_name in enumerate(joint_names):
        side = detect_joint_side(joint_name)
        if side is None:
            side = detect_joint_side(_canonicalize_joint_name(joint_name))
        side = side if side in ('left', 'right') else 'center'
        joint_side_labels.append(side)

        if side == 'center':
            continue

        signature = _joint_signature(joint_name)
        if not signature:
            continue
        if signature not in grouped_indices:
            grouped_indices[signature] = {'left': [], 'right': []}
        grouped_indices[signature][side].append(joint_index)

    symmetry_partner_indices = [-1] * len(joint_names)
    symmetric_joint_pairs = []

    for signature in sorted(grouped_indices):
        left_indices = sorted(grouped_indices[signature]['left'], key=lambda index: (depths[index], index))
        remaining_right_indices = set(grouped_indices[signature]['right'])
        for left_index in left_indices:
            if not remaining_right_indices:
                break
            best_right = min(
                remaining_right_indices,
                key=lambda right_index: _symmetry_pair_score(
                    left_index,
                    right_index,
                    rest_positions,
                    depths,
                    parents,
                    joint_names,
                ),
            )
            remaining_right_indices.remove(best_right)
            symmetry_partner_indices[left_index] = best_right
            symmetry_partner_indices[best_right] = left_index
            symmetric_joint_pairs.append([left_index, best_right])

    children = [[] for _ in range(len(parents))]
    for joint_index, parent_index in enumerate(parents):
        if parent_index >= 0:
            children[parent_index].append(joint_index)

    changed = True
    while changed:
        changed = False
        for left_parent, right_parent in list(symmetric_joint_pairs):
            left_unpaired = [joint_index for joint_index in children[left_parent] if symmetry_partner_indices[joint_index] < 0]
            right_unpaired = [joint_index for joint_index in children[right_parent] if symmetry_partner_indices[joint_index] < 0]
            if len(left_unpaired) != 1 or len(right_unpaired) != 1:
                continue

            left_index = left_unpaired[0]
            right_index = right_unpaired[0]
            if not _passes_conservative_child_mirror_check(
                left_index,
                right_index,
                left_parent,
                right_parent,
                rest_positions,
            ):
                continue

            symmetry_partner_indices[left_index] = right_index
            symmetry_partner_indices[right_index] = left_index
            joint_side_labels[left_index] = 'left'
            joint_side_labels[right_index] = 'right'
            symmetric_joint_pairs.append([left_index, right_index])
            changed = True

    if return_details:
        return {
            'joint_side_labels': joint_side_labels,
            'symmetry_partner_indices': symmetry_partner_indices,
            'symmetric_joint_pairs': symmetric_joint_pairs,
        }

    return joint_side_labels, symmetry_partner_indices, symmetric_joint_pairs


def _infer_is_symmetric(symmetric_joint_pairs, joint_side_labels):
    """Determine if skeleton has bilateral symmetry based on paired joints and side labels.
    
    Returns True if:
    - At least 2 symmetric pairs were found, OR
    - At least 30% of joints are labeled as left or right (not center)
    """
    num_pairs = len(symmetric_joint_pairs)
    if num_pairs >= 2:
        return True
    
    if joint_side_labels:
        sided_count = sum(1 for label in joint_side_labels if label in ('left', 'right'))
        sided_ratio = sided_count / len(joint_side_labels)
        if sided_ratio >= 0.3:
            return True
    
    return False


def build_semantic_metadata(joint_names, parents, offsets, rest_positions=None):
    parents = np.asarray(parents, dtype=np.int64)
    rest_positions = rest_positions_from_offsets(offsets, parents) if rest_positions is None else np.asarray(rest_positions, dtype=np.float64)
    canonical_joint_names = [_canonicalize_joint_name(name) for name in joint_names]
    canonical_joint_names = _collapse_solitary_head_feature_indices(canonical_joint_names)
    contact_joints, contact_joint_source = infer_contact_joints(
        joint_names,
        parents,
        rest_positions,
    )
    leaf_contact_joints = {
        int(joint_index)
        for joint_index in contact_joints
        if not np.any(np.asarray(parents) == int(joint_index))
    }
    end_effector_joints = sorted(
        set(_infer_end_effector_joints(parents, joint_names=joint_names, rest_positions=rest_positions))
        | leaf_contact_joints
    )
    symmetry_metadata = _infer_symmetry_metadata(joint_names, parents, rest_positions, return_details=True)
    joint_side_labels = symmetry_metadata['joint_side_labels']
    symmetry_partner_indices = symmetry_metadata['symmetry_partner_indices']
    symmetric_joint_pairs = symmetry_metadata['symmetric_joint_pairs']
    is_symmetric = _infer_is_symmetric(symmetric_joint_pairs, joint_side_labels)
    return {
        'canonical_joint_names': canonical_joint_names,
        'end_effector_joints': end_effector_joints,
        'end_effector_names': [joint_names[index] for index in end_effector_joints],
        'contact_joints': list(contact_joints),
        'contact_joint_names': [joint_names[index] for index in contact_joints],
        'contact_joint_source': contact_joint_source,
        'joint_side_labels': joint_side_labels,
        'symmetry_partner_indices': symmetry_partner_indices,
        'symmetric_joint_pairs': symmetric_joint_pairs,
        'symmetric_joint_pair_names': [[joint_names[left], joint_names[right]] for left, right in symmetric_joint_pairs],
        'is_symmetric': bool(is_symmetric),
    }
