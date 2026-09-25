"""End effector detection and symmetry analysis utilities."""

import numpy as np

from .joint_name_canonical import (
    canonicalize_joint_name,
    collapse_solitary_head_feature_indices,
    effective_canonical_replacements,
    infer_species_joint_name_prefixes,
    normalize_joint_name,
)
from .joint_embedding_text import (
    EMBED_TEXT_FACE_QUADRANT_CONTEXT_TOKENS,
    EMBED_TEXT_HEAD_CODE_CONTEXT_TOKENS,
    EMBED_TEXT_QUADRANT_LIMB_CONTEXT_TOKENS,
)
from .joint_struct_features import child_lists


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


# Limb and face corner codes in the symmetry signature: the side half is dropped
# and the fore/hind/middle (or top/bottom) half kept, so LfLeg01 pairs only with
# RfLeg01 and MouthTL only with MouthTR.
_LIMB_CODE_SIGNATURE_TOKENS = {
    'lf': 'f', 'rf': 'f', 'lb': 'b', 'rb': 'b',
    'fl': 'f', 'fr': 'f', 'bl': 'b', 'br': 'b',
    'lm': 'm', 'rm': 'm', 'ml': 'm', 'mr': 'm',
    'tl': 't', 'tr': 't',
}


# Spelling repairs for the symmetry signature, so both halves of a pair share one
# key ("UpperReg" -> "UpperLeg", "Lwing" -> "wing"). Spelling only: folding the
# synonym table in would regroup existing rigs.
_SIGNATURE_SPELLING_TOKENS = {
    'rower': 'lower',
    'reg': 'leg',
    'piers': 'pliers',
    'lwing': 'wing',
    'rwing': 'wing',
}


def _signature_tokens(tokens, side_tokens):
    signature_tokens = []
    for token in tokens:
        if token in side_tokens:
            continue
        token = _SIGNATURE_SPELLING_TOKENS.get(token, token)
        signature_tokens.append(_LIMB_CODE_SIGNATURE_TOKENS.get(token, token))
    return signature_tokens


def joint_signature(name):
    signature_tokens = _signature_tokens(
        canonicalize_joint_name(name).lower().split(), ('left', 'right'),
    )
    if signature_tokens:
        return ' '.join(signature_tokens)

    fallback_tokens = _signature_tokens(
        normalize_joint_name(name).split(), ('left', 'right', 'l', 'r'),
    )
    return ' '.join(fallback_tokens)


def _fallback_child_signature(name):
    return ' '.join(
        token for token in joint_signature(name).split()
        if not token.isdigit()
    )


def _joint_semantic_text(name):
    normalized = normalize_joint_name(name)
    canonical = canonicalize_joint_name(name).lower()
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

    _, symmetry_partner_indices, _ = infer_symmetry_metadata(joint_names, parents, rest_positions)
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
    children = child_lists(parents)
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
    children = child_lists(parents)

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

    return [], 'none'


def joint_depths(parents):
    depths = [0] * len(parents)
    for joint_index in range(1, len(parents)):
        parent_index = parents[joint_index]
        if parent_index >= 0:
            depths[joint_index] = depths[parent_index] + 1
    return depths


def detect_joint_side(name):
    normalized = normalize_joint_name(name)
    compact = normalized.replace(' ', '')
    tokens = set(normalized.split())
    right_markers = (
        ' right ',
        ' r ',
        ' r_',
        ' rleg',
        ' rarm',
        ' rwing',
        ' rthigh',
        ' rclavicle',
        ' rupperarm',
        ' r momo',
        ' r kata',
        ' r hiji',
    )
    left_markers = (
        ' left ',
        ' l ',
        ' l_',
        ' lleg',
        ' larm',
        ' lwing',
        ' lthigh',
        ' lclavicle',
        ' lupperarm',
        ' l momo',
        ' l kata',
        ' l hiji',
    )
    # Markers are matched on whole words at their start. A rig prefix followed by
    # a bare side letter ("NPC_R_Thigh", "BN_L_Arm") is already ' r ' / ' l ';
    # spelled as a prefix marker (' npc r') it also caught the first letter of the
    # next word, siding Bear's "NPC_Ribcage" right and "NPC_LowerFrontLip" left.
    padded = f' {normalized} '
    if any(marker in padded for marker in right_markers) or compact.startswith(('r_', 'rleg', 'rarm', 'rwing', 'rthigh', 'rmomo', 'rkata', 'rhiji')):
        return 'right'
    if any(marker in padded for marker in left_markers) or compact.startswith(('l_', 'lleg', 'larm', 'lwing', 'lthigh', 'lmomo', 'lkata', 'lhiji')):
        return 'left'

    # Quadruped limb codes. Both halves of the rig use them -- Lf/Rf for the fore
    # limbs, Lb/Rb for the hind -- but only the fore pair was ever read, so every
    # Lb*/Rb* joint in Bear, Dinosaur, Tiger, antilope and rhino (52 joints) came
    # back 'center' and lost its side, taking its symmetry pairing with it.
    # Fires only on an unambiguous single side, same as the explicit markers above.
    right_codes = tokens & {'rf', 'rb'}
    left_codes = tokens & {'lf', 'lb'}
    if right_codes and not left_codes:
        return 'right'
    if left_codes and not right_codes:
        return 'left'

    # The same code with the halves swapped (Fl/Fr, Bl/Br) plus the hexapod's
    # middle pair (Lm/Rm, Ml/Mr). Read only next to a limb word, for the same
    # reason EMBED_TEXT_QUADRANT_LIMB_CODE_TOKENS is gated: a mouth corner named
    # "MouthBL" is a bottom-left corner, not a back-left leg. Without this a
    # whole quadruped ("FlLeg1".."BrLegFoot2") came back 'center' and formed no
    # symmetry pairs at all.
    if tokens & EMBED_TEXT_QUADRANT_LIMB_CONTEXT_TOKENS:
        return _single_side(tokens & {'fr', 'br', 'rm', 'mr'}, tokens & {'fl', 'bl', 'lm', 'ml'})
    side = None
    # Lm/Rm on a head: the mouth corners (EMBED_TEXT_HEAD_MOUTH_CODE_TOKENS).
    if tokens & EMBED_TEXT_HEAD_CODE_CONTEXT_TOKENS:
        side = _single_side(tokens & {'rm'}, tokens & {'lm'})
    # Tl/Tr/Bl/Br on a face part: its top/bottom corners
    # (EMBED_TEXT_FACE_QUADRANT_CODE_TOKENS).
    if side is None and tokens & EMBED_TEXT_FACE_QUADRANT_CONTEXT_TOKENS:
        side = _single_side(tokens & {'tr', 'br'}, tokens & {'tl', 'bl'})
    return side


def _single_side(right_codes, left_codes):
    if right_codes and not left_codes:
        return 'right'
    if left_codes and not right_codes:
        return 'left'
    return None


def _symmetry_pair_score(left_index, right_index, rest_positions, depths, parents, joint_names):
    mirror_error = abs(float(rest_positions[left_index, 0] + rest_positions[right_index, 0]))
    yz_error = float(np.linalg.norm(rest_positions[left_index, 1:] - rest_positions[right_index, 1:]))
    depth_error = abs(depths[left_index] - depths[right_index])

    left_parent = parents[left_index]
    right_parent = parents[right_index]
    left_parent_sig = joint_signature(joint_names[left_parent]) if left_parent >= 0 else ''
    right_parent_sig = joint_signature(joint_names[right_parent]) if right_parent >= 0 else ''
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


# Two candidate errors closer than this count as a tie. The errors are
# normalized by bone length, so this is a fraction of the joint's own bone.
_NEAREST_TIE_TOLERANCE = 1e-6


def _unambiguous_mutual_nearest(errors):
    """Keys of ``errors`` whose two joints are each other's unique closest match.

    ``errors`` maps (joint, joint) to a mirror error. A joint whose closest
    match is tied between two candidates -- coincident helper bones, say --
    has no closest match: picking either would pair by index order, and a wrong
    parent pair also blocks every child pair beneath it. Returned sorted.
    """
    best = {}
    for (first, second), error in errors.items():
        for index, other in ((first, second), (second, first)):
            if index not in best or error < best[index][0] - _NEAREST_TIE_TOLERANCE:
                best[index] = (error, other)
            elif error <= best[index][0] + _NEAREST_TIE_TOLERANCE:
                best[index] = (min(error, best[index][0]), None)
    return [
        (first, second) for first, second in sorted(errors)
        if best[first][1] == second and best[second][1] == first
    ]


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


# How unanimous the rig's own named joints must be about which X half is its
# left before a name that contradicts it is overruled, and how many off-midline
# named joints that vote needs.
_WRONG_SIDE_NAME_MIN_AGREEMENT = 0.9
_WRONG_SIDE_NAME_MIN_VOTES = 4
# Fraction of a joint's offset from its parent below which it counts as on the
# midline and casts no vote / cannot be on the wrong side.
_WRONG_SIDE_MIDLINE_RATIO = 0.1


def _subtree_mean_x(parents, rest_positions):
    """Mean X over each joint's subtree, the joint included."""
    subtree_x_sum = rest_positions[:, 0].copy()
    subtree_size = np.ones(len(parents), dtype=np.float64)
    depths = joint_depths(parents)
    for index in sorted(range(len(parents)), key=lambda i: -depths[i]):
        if parents[index] >= 0:
            subtree_x_sum[parents[index]] += subtree_x_sum[index]
            subtree_size[parents[index]] += subtree_size[index]
    return subtree_x_sum / subtree_size


def _midline_tolerance(index, parents, rest_positions):
    parent = parents[index]
    anchor = rest_positions[parent] if parent >= 0 else np.zeros(3, dtype=np.float64)
    return _WRONG_SIDE_MIDLINE_RATIO * max(float(np.linalg.norm(rest_positions[index] - anchor)), 1e-6)


def _correct_wrong_side_mirror_siblings(joint_names, joint_side_labels, parents, rest_positions):
    """Relabel named-side joints that sit on the other half of the rig from
    their mirror twin. Mutates *joint_side_labels*; returns the indices.

    Two slips in the source rigs, one rule. Leopard names both mane joints
    right ("BN_Mane_R_01" at x=-0.08, "BN_Mane_R_02" at x=+0.08), so with no
    left joint the pair can never form. Spider swaps its jaws as a pair
    ("R_Jaw_" at +X, "L_Jaw_" at -X) while every other joint of the rig keeps
    its left at -X. Which X half is left is read off the rig's own names, never
    assumed, and only a near-unanimous vote is trusted. A joint on the wrong
    half is overruled only when a sibling with the same name apart from its
    side and indices ("Mane 01" / "Mane 02") is its mirror image and sits on the
    half this joint's label claims -- whatever that sibling is labelled. The
    same-part test matters: the mirror check alone is loose enough to match a
    neighbouring part (Spider's "R_Jaw_" against "FangR_00_").

    A joint's half is where its subtree lies, not its own point: Biped and jt_
    rigs seat the clavicle and hip joints across the midline ("Bip01_R_Clavicle"
    at x=+0.08 in Lion) while the limb below them sits on the named side.
    """
    rest_positions = np.asarray(rest_positions, dtype=np.float64)
    side_x = _subtree_mean_x(parents, rest_positions)

    def midline_tolerance(index):
        return _midline_tolerance(index, parents, rest_positions)

    left_positive = 0
    left_negative = 0
    for index, side in enumerate(joint_side_labels):
        x = float(side_x[index])
        if side not in ('left', 'right') or abs(x) <= midline_tolerance(index):
            continue
        if (side == 'left') == (x > 0):
            left_positive += 1
        else:
            left_negative += 1
    votes = left_positive + left_negative
    if votes < _WRONG_SIDE_NAME_MIN_VOTES:
        return []
    if left_positive >= _WRONG_SIDE_NAME_MIN_AGREEMENT * votes:
        left_sign = 1.0
    elif left_negative >= _WRONG_SIDE_NAME_MIN_AGREEMENT * votes:
        left_sign = -1.0
    else:
        return []

    def claimed_sign(index):
        return left_sign if joint_side_labels[index] == 'left' else -left_sign

    def on_claimed_half(index):
        x = float(side_x[index])
        if abs(x) <= midline_tolerance(index):
            return None
        return x * claimed_sign(index) > 0

    corrected = []
    for index, side in enumerate(joint_side_labels):
        if side not in ('left', 'right') or on_claimed_half(index) is not False:
            continue
        parent = parents[index]
        has_mirror_twin = any(
            sibling != index
            and parents[sibling] == parent
            and joint_side_labels[sibling] in ('left', 'right')
            and _fallback_child_signature(joint_names[sibling]) == _fallback_child_signature(joint_names[index])
            and abs(float(side_x[sibling])) > midline_tolerance(sibling)
            and float(side_x[sibling]) * claimed_sign(index) > 0
            and _passes_conservative_child_mirror_check(sibling, index, parent, parent, rest_positions)
            for sibling in range(len(joint_side_labels))
        )
        if has_mirror_twin:
            corrected.append(index)
    for index in corrected:
        joint_side_labels[index] = 'right' if joint_side_labels[index] == 'left' else 'left'
    return corrected


def _pair_sided_joints_by_geometry(joint_side_labels, symmetry_partner_indices, parents,
                                   rest_positions, depths):
    """Pair named-side joints whose signatures never met, by mirror geometry.

    The name decides the side but the signature can still differ between the
    halves: FireAnt's 3ds Max Biped extras keep their Ponytail index in the name
    ("Bip01_Ponytail2_R_Antenna1" against "Bip01_Ponytail1_L_Antenna"), so the
    two antennae and mandibles land in different signature groups. A left and a
    right joint pair here only when they hang off the same parent or off an
    already-mirrored parent pair, sit at the same depth, pass the child mirror
    check, and are each other's closest match. Side labels are never created
    here -- a 'center' joint stays out.
    """
    left_indices = [
        index for index, side in enumerate(joint_side_labels)
        if side == 'left' and symmetry_partner_indices[index] < 0
    ]
    right_indices = [
        index for index, side in enumerate(joint_side_labels)
        if side == 'right' and symmetry_partner_indices[index] < 0
    ]
    errors = {}
    for left_index in left_indices:
        left_parent = parents[left_index]
        for right_index in right_indices:
            right_parent = parents[right_index]
            if depths[left_index] != depths[right_index]:
                continue
            if left_parent != right_parent and (
                left_parent < 0 or symmetry_partner_indices[left_parent] != right_parent
            ):
                continue
            if not _passes_conservative_child_mirror_check(
                left_index, right_index, left_parent, right_parent, rest_positions,
            ):
                continue
            mirror_error, yz_error, local_scale = _local_mirror_error(
                left_index, right_index, left_parent, right_parent, rest_positions,
            )
            errors[left_index, right_index] = (mirror_error + yz_error) / local_scale

    return _unambiguous_mutual_nearest(errors)


# Smallest ratio of the two subtree sizes an unnamed twin may have: one half
# can carry an extra prop joint (a blade in one hand).
_UNSIDED_TWIN_MIN_SUBTREE_RATIO = 0.8


def _subtree_sizes(parents):
    """Per joint, its subtree size and its number of direct children."""
    sizes = np.ones(len(parents), dtype=np.int64)
    child_counts = np.zeros(len(parents), dtype=np.int64)
    depths = joint_depths(parents)
    for index in sorted(range(len(parents)), key=lambda i: -depths[i]):
        if parents[index] >= 0:
            sizes[parents[index]] += sizes[index]
            child_counts[parents[index]] += 1
    return sizes, child_counts


def _pair_unsided_twins_by_name(joint_side_labels, symmetry_partner_indices, parents,
                                rest_positions, depths, signatures, subtree_x, subtree_sizes):
    """Pair a named-side joint with a 'center' joint that is its unnamed twin.

    Some rigs spell the side on one half only: serpent_man has "R_Arm_Shoulder"
    and a plain "Arm_Shoulder" on the other half. With the side word stripped
    the two names are the same, so the 'center' joint is a candidate when it
    hangs off the same parent (or the mirror of the sided joint's parent), has
    as many children and a subtree of about the same size, lies off the
    midline on the other half -- judged by its subtree, as Biped clavicles
    cross the midline -- and passes the child mirror check. Each candidate pair
    must be the other's closest match. The 'center' joint takes the opposite
    side; the pairs are returned as (left, right).
    """
    sizes, child_counts = subtree_sizes
    def off_midline(index):
        return abs(float(subtree_x[index])) > _midline_tolerance(index, parents, rest_positions)

    sided = [
        index for index, side in enumerate(joint_side_labels)
        if side in ('left', 'right') and symmetry_partner_indices[index] < 0
        and signatures[index] and off_midline(index)
    ]
    unsided = [
        index for index, side in enumerate(joint_side_labels)
        if side == 'center' and symmetry_partner_indices[index] < 0
        and signatures[index] and off_midline(index)
    ]
    errors = {}
    for sided_index in sided:
        sided_parent = parents[sided_index]
        for unsided_index in unsided:
            unsided_parent = parents[unsided_index]
            if signatures[sided_index] != signatures[unsided_index]:
                continue
            if depths[sided_index] != depths[unsided_index]:
                continue
            if child_counts[sided_index] != child_counts[unsided_index]:
                continue
            if (min(sizes[sided_index], sizes[unsided_index])
                    < _UNSIDED_TWIN_MIN_SUBTREE_RATIO * max(sizes[sided_index], sizes[unsided_index])):
                continue
            if float(subtree_x[sided_index]) * float(subtree_x[unsided_index]) >= 0:
                continue
            if sided_parent != unsided_parent and (
                sided_parent < 0 or symmetry_partner_indices[sided_parent] != unsided_parent
            ):
                continue
            if not _passes_conservative_child_mirror_check(
                sided_index, unsided_index, sided_parent, unsided_parent, rest_positions,
            ):
                continue
            mirror_error, yz_error, local_scale = _local_mirror_error(
                sided_index, unsided_index, sided_parent, unsided_parent, rest_positions,
            )
            errors[sided_index, unsided_index] = (mirror_error + yz_error) / local_scale

    pairs = []
    for sided_index, unsided_index in _unambiguous_mutual_nearest(errors):
        if joint_side_labels[sided_index] == 'left':
            joint_side_labels[unsided_index] = 'right'
            pairs.append((sided_index, unsided_index))
        else:
            joint_side_labels[unsided_index] = 'left'
            pairs.append((unsided_index, sided_index))
    return pairs


# Largest mirror error, as a fraction of the joint's own bone length, an unnamed
# pair may have. Nothing in the names vouches for such a pair, so only a near
# exact mirror image counts.
_UNNAMED_TWIN_MIRROR_TOLERANCE = 0.05


def _rig_left_sign(joint_side_labels, subtree_x):
    """+1 when the rig's named left half is +X, -1 when it is -X.

    Read off the joints already labelled, by where their subtrees lie. A rig
    with no side names at all gets +X, the convention every named rig follows.
    """
    vote = 0.0
    for index, side in enumerate(joint_side_labels):
        if side in ('left', 'right'):
            vote += np.sign(float(subtree_x[index])) * (1.0 if side == 'left' else -1.0)
    return -1.0 if vote < 0 else 1.0


def _pair_unnamed_twins_by_geometry(joint_side_labels, symmetry_partner_indices, parents,
                                    rest_positions, depths, subtree_x, subtree_sizes):
    """Pair any two unpaired joints that are exact mirror images of each other.

    The name rules all need something in the names to agree. Some rigs name
    neither half -- Dog's ears are "Bip01_Ponytail1" and "Bip01_Ponytail2",
    Crow's tail feathers "Tail02" and "Tail03" -- and a joint can carry a side
    word while its twin carries none and shares no signature with it. Here the
    geometry carries the whole decision, whatever the labels, so the gate is
    strict: both joints off the midline on opposite halves, the same parent or
    an already-mirrored parent pair, the same depth, child count and subtree
    size, and both the joint's offset from its parent and its absolute position
    mirrored to within ``_UNNAMED_TWIN_MIRROR_TOLERANCE`` of its bone length. A
    joint that already names a side must lie on that side's half. Each must be
    the other's closest match. Both joints take the side of the half they lie
    on; the pairs are returned as (left, right).
    """
    rest_positions = np.asarray(rest_positions, dtype=np.float64)
    sizes, child_counts = subtree_sizes
    mirror = np.array([-1.0, 1.0, 1.0])
    left_sign = _rig_left_sign(joint_side_labels, subtree_x)

    def on_named_half(index):
        side = joint_side_labels[index]
        if side == 'center':
            return True
        return float(subtree_x[index]) * left_sign * (1.0 if side == 'left' else -1.0) > 0

    candidates = [
        index for index in range(len(joint_side_labels))
        if symmetry_partner_indices[index] < 0 and parents[index] >= 0
        and abs(float(subtree_x[index])) > _midline_tolerance(index, parents, rest_positions)
        and on_named_half(index)
    ]
    errors = {}
    for position, first in enumerate(candidates):
        for second in candidates[position + 1:]:
            if float(subtree_x[first]) * float(subtree_x[second]) >= 0:
                continue
            if (depths[first] != depths[second]
                    or child_counts[first] != child_counts[second]
                    or sizes[first] != sizes[second]):
                continue
            first_parent, second_parent = parents[first], parents[second]
            if first_parent != second_parent and symmetry_partner_indices[first_parent] != second_parent:
                continue
            mirror_error, yz_error, local_scale = _local_mirror_error(
                first, second, first_parent, second_parent, rest_positions,
            )
            absolute_error = float(np.linalg.norm(
                rest_positions[first] - mirror * rest_positions[second]
            ))
            limit = _UNNAMED_TWIN_MIRROR_TOLERANCE * local_scale
            if mirror_error + yz_error > limit or absolute_error > limit:
                continue
            errors[first, second] = (mirror_error + yz_error + absolute_error) / local_scale

    pairs = []
    for first, second in _unambiguous_mutual_nearest(errors):
        left, right = (first, second) if float(subtree_x[first]) * left_sign > 0 else (second, first)
        joint_side_labels[left] = 'left'
        joint_side_labels[right] = 'right'
        pairs.append((left, right))
    return pairs


def infer_symmetry_metadata(joint_names, parents, rest_positions, return_details=False):
    depths = joint_depths(parents)
    joint_side_labels = []
    grouped_indices = {}

    for joint_name in joint_names:
        side = detect_joint_side(joint_name)
        if side is None:
            side = detect_joint_side(canonicalize_joint_name(joint_name))
        joint_side_labels.append(side if side in ('left', 'right') else 'center')
    _correct_wrong_side_mirror_siblings(joint_names, joint_side_labels, parents, rest_positions)

    for joint_index, joint_name in enumerate(joint_names):
        side = joint_side_labels[joint_index]
        if side == 'center':
            continue

        signature = joint_signature(joint_name)
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

    signatures = [joint_signature(joint_name) for joint_name in joint_names]
    subtree_x = _subtree_mean_x(parents, np.asarray(rest_positions, dtype=np.float64))
    subtree_sizes = _subtree_sizes(parents)

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

        for left_index, right_index in _pair_sided_joints_by_geometry(
            joint_side_labels, symmetry_partner_indices, parents, rest_positions, depths,
        ):
            symmetry_partner_indices[left_index] = right_index
            symmetry_partner_indices[right_index] = left_index
            symmetric_joint_pairs.append([left_index, right_index])
            changed = True

        for left_index, right_index in _pair_unsided_twins_by_name(
            joint_side_labels, symmetry_partner_indices, parents, rest_positions, depths,
            signatures, subtree_x, subtree_sizes,
        ):
            symmetry_partner_indices[left_index] = right_index
            symmetry_partner_indices[right_index] = left_index
            symmetric_joint_pairs.append([left_index, right_index])
            changed = True

        # Last: every name-backed rule has had its chance at these joints.
        for left_index, right_index in _pair_unnamed_twins_by_geometry(
            joint_side_labels, symmetry_partner_indices, parents, rest_positions, depths,
            subtree_x, subtree_sizes,
        ):
            symmetry_partner_indices[left_index] = right_index
            symmetry_partner_indices[right_index] = left_index
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


def build_semantic_metadata(joint_names, parents, offsets, rest_positions=None, species_name=None):
    parents = np.asarray(parents, dtype=np.int64)
    rest_positions = rest_positions_from_offsets(offsets, parents) if rest_positions is None else np.asarray(rest_positions, dtype=np.float64)
    replacements = effective_canonical_replacements(joint_names)
    species_prefixes = infer_species_joint_name_prefixes(joint_names, species_name)
    canonical_joint_names = [
        canonicalize_joint_name(name, replacements, species_prefixes)
        for name in joint_names
    ]
    canonical_joint_names = collapse_solitary_head_feature_indices(canonical_joint_names)
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
    symmetry_metadata = infer_symmetry_metadata(joint_names, parents, rest_positions, return_details=True)
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
