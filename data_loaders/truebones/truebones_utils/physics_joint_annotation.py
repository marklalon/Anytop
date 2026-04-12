"""End effector detection and symmetry analysis utilities."""

import numpy as np
import re
from .param_utils import SNAKES


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


def _normalize_joint_name(name):
    # Split on lowercase→UPPER (e.g. "ElkRFemur" → "Elk RFemur")
    split_name = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', name)
    # Also split on UPPER→UPPER+lower (e.g. "RFemur" → "R Femur")
    split_name = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', split_name)
    return re.sub(r'[^a-z0-9]+', ' ', split_name.lower()).strip()


def _strip_joint_name_prefix(name):
    stripped = name
    for prefix in sorted(_CANONICAL_NAME_PREFIXES, key=len, reverse=True):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    return stripped


def _canonicalize_joint_name(name):
    split_name = _normalize_joint_name(_strip_joint_name_prefix(name))
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
            continue
        else:
            canonical_parts.append(clean_part.capitalize())
    return ' '.join(canonical_parts) if canonical_parts else name.strip()


def _joint_signature(name):
    signature_tokens = [
        token for token in _canonicalize_joint_name(name).lower().split()
        if token not in ('left', 'right')
    ]
    if signature_tokens:
        return ' '.join(signature_tokens)

    fallback_tokens = [
        token for token in _normalize_joint_name(name).split()
        if token not in ('left', 'right', 'l', 'r')
    ]
    return ' '.join(fallback_tokens)


def _joint_semantic_text(name):
    normalized = _normalize_joint_name(name)
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
    normalized = _normalize_joint_name(name)
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


def _rest_positions_from_offsets(offsets, parents):
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


def _infer_contact_joints(object_type, joint_names, parents, rest_positions):
    if object_type in SNAKES:
        return list(range(len(joint_names))), 'full_body'

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


def _detect_joint_side(name):
    normalized = _normalize_joint_name(name)
    compact = normalized.replace(' ', '')
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


def _infer_symmetry_metadata(joint_names, parents, rest_positions):
    depths = _joint_depths(parents)
    joint_side_labels = []
    grouped_indices = {}

    for joint_index, joint_name in enumerate(joint_names):
        side = _detect_joint_side(joint_name)
        if side is None:
            side = _detect_joint_side(_canonicalize_joint_name(joint_name))
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


def _build_semantic_metadata(object_type, joint_names, parents, offsets, rest_positions=None):
    parents = np.asarray(parents, dtype=np.int64)
    rest_positions = _rest_positions_from_offsets(offsets, parents) if rest_positions is None else np.asarray(rest_positions, dtype=np.float64)
    canonical_joint_names = [_canonicalize_joint_name(name) for name in joint_names]
    contact_joints, contact_joint_source = _infer_contact_joints(
        object_type,
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
    joint_side_labels, symmetry_partner_indices, symmetric_joint_pairs = _infer_symmetry_metadata(joint_names, parents, rest_positions)
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
