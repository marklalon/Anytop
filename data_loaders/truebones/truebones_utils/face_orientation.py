"""Face orientation and forward direction detection utilities."""

import numpy as np
import re
from Quaternions import Quaternions
from Animation import Animation
from .param_utils import CHAIN_FORWARD_JOINTS


# Face joint detection tokens
_FACE_JOINT_EXCLUDE_TOKENS = (
    'jiggle',
    'toe',
    'foot',
    'ankle',
    'ball',
    'nub',
    'finger',
    'thumb',
    'jaw',
    'lip',
    'nose',
    'eye',
    'ear',
)
_FACE_JOINT_NEAR_ROOT_EXCLUDE_TOKENS = (
    'head',
    'neck',
    'spine',
)
_FACE_JOINT_HIP_PRIORITIES = (
    ('thigh', 'leg1', 'upperleg', 'upleg', 'momo', 'femur'),
    ('leg',),
)
_FACE_JOINT_UPPER_PRIORITIES = (
    ('collarbone', 'clavicle', 'shoulder', 'upperarm', 'arm1', 'kata', 'wing', 'scapula', 'humerus'),
    ('arm', 'hiji', 'elbow', 'forearm'),
    ('te', 'hand'),
)
_FORWARD_REFERENCE_PRIORITIES = (
    ('nose', 'snout', 'muzzle', 'beak'),
    ('head',),
    ('neck',),
)

# Shared utilities from end_effector_symmetry
def _normalize_joint_name(name):
    split_name = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', name)
    split_name = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', split_name)
    return re.sub(r'[^a-z0-9]+', ' ', split_name.lower()).strip()


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


def _canonicalize_joint_name(name):
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
    
    # Strip prefix
    stripped = name
    for prefix in sorted(_CANONICAL_NAME_PREFIXES, key=len, reverse=True):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    
    # Normalize and canonicalize
    split_name = _normalize_joint_name(stripped)
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


def _face_joint_name_allowed(name):
    normalized = _normalize_joint_name(name)
    if any(token in normalized for token in _FACE_JOINT_EXCLUDE_TOKENS):
        return False
    return True


def _find_semantic_joint_pair(joint_names, parents, priorities, *, exclude_near_root=True):
    depths = _joint_depths(parents)
    candidates = {'right': [], 'left': []}

    for joint_index, joint_name in enumerate(joint_names):
        if not _face_joint_name_allowed(joint_name):
            continue
        normalized = _normalize_joint_name(joint_name)
        if exclude_near_root and any(token in normalized for token in _FACE_JOINT_NEAR_ROOT_EXCLUDE_TOKENS):
            continue

        side = _detect_joint_side(joint_name)
        if side is None:
            continue

        priority_index = None
        for current_priority, keyword_group in enumerate(priorities):
            if any(keyword in normalized for keyword in keyword_group):
                priority_index = current_priority
                break
        if priority_index is None:
            continue

        candidates[side].append((priority_index, depths[joint_index], joint_index))

    if not candidates['right'] or not candidates['left']:
        return None

    right_index = min(candidates['right'])[2]
    left_index = min(candidates['left'])[2]
    return right_index, left_index


def _find_forward_reference_joint(joint_names, parents):
    depths = _joint_depths(parents)
    candidates = []

    for joint_index, joint_name in enumerate(joint_names):
        normalized = _normalize_joint_name(joint_name)
        priority_index = None
        for current_priority, keyword_group in enumerate(_FORWARD_REFERENCE_PRIORITIES):
            if any(keyword in normalized for keyword in keyword_group):
                priority_index = current_priority
                break
        if priority_index is None:
            continue
        candidates.append((priority_index, -depths[joint_index], joint_index))

    if not candidates:
        return None

    return min(candidates)[2]


def _find_neck_reference_joint(joint_names, parents):
    depths = _joint_depths(parents)
    candidates = []

    for joint_index, joint_name in enumerate(joint_names):
        normalized = _normalize_joint_name(joint_name)
        if 'neck' not in normalized:
            continue
        candidates.append((-depths[joint_index], joint_index))

    if not candidates:
        return None

    return min(candidates)[1]


def _normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return vectors / norms


def _vector_angle_deg(vector_a, vector_b):
    a = np.asarray(vector_a, dtype=np.float64).reshape(-1)
    b = np.asarray(vector_b, dtype=np.float64).reshape(-1)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm <= 1e-8 or b_norm <= 1e-8:
        return 180.0
    cosine = float(np.dot(a / a_norm, b / b_norm))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _get_chain_forward(joints, object_type):
    chain = CHAIN_FORWARD_JOINTS.get(object_type)
    if chain is None:
        return None

    if len(chain) == 2:
        neck, head = chain
        forward = joints[:, head] - joints[:, neck]
    else:
        body_base, neck, head = chain
        forward = (joints[:, head] - joints[:, neck]) + (joints[:, neck] - joints[:, body_base])
    forward = forward * np.array([[1.0, 0.0, 1.0]])
    return _normalize_vectors(forward)


def _get_head_forward(joints, face_joint_indx, forward_joint_index, forward_base_joint_index=None):
    if forward_joint_index is None or not face_joint_indx:
        return None

    if forward_base_joint_index is not None and forward_joint_index == forward_base_joint_index:
        return None

    if forward_base_joint_index is not None:
        forward = joints[:, forward_joint_index] - joints[:, forward_base_joint_index]
    else:
        r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
        hip_center = 0.5 * (joints[:, r_hip] + joints[:, l_hip])
        shoulder_center = 0.5 * (joints[:, sdr_r] + joints[:, sdr_l])
        torso_center = 0.5 * (hip_center + shoulder_center)
        forward = joints[:, forward_joint_index] - torso_center
    forward = forward * np.array([[1.0, 0.0, 1.0]])
    if not np.isfinite(forward).all():
        return None
    if np.all(np.linalg.norm(forward, axis=-1) < 1e-8):
        return None
    return _normalize_vectors(forward)


def _get_facing_candidates(
    joints,
    object_type,
    face_joint_indx=None,
    forward_joint_index=None,
    forward_base_joint_index=None,
):
    if object_type in CHAIN_FORWARD_JOINTS:
        chain_forward = _get_chain_forward(joints, object_type)
        if chain_forward is None:
            return {}
        return {'chain': chain_forward}

    candidates = {}
    torso_head = _get_head_forward(
        joints,
        face_joint_indx,
        forward_joint_index,
        forward_base_joint_index=None,
    )
    if torso_head is not None:
        candidates['torso_head'] = torso_head

    neck_head = _get_head_forward(
        joints,
        face_joint_indx,
        forward_joint_index,
        forward_base_joint_index=forward_base_joint_index,
    )
    if neck_head is not None:
        candidates['neck_head'] = neck_head

    if face_joint_indx:
        r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
        across1 = joints[:, r_hip] - joints[:, l_hip]
        across2 = joints[:, sdr_r] - joints[:, sdr_l]
        across = across1 + across2
        across_norm = np.linalg.norm(across, axis=-1, keepdims=True)
        if np.isfinite(across).all() and not np.all(across_norm < 1e-8):
            across = across / np.where(across_norm < 1e-8, 1.0, across_norm)
            forward = np.cross(np.array([[0.0, 1.0, 0.0]]), across, axis=-1)
            forward_norm = np.linalg.norm(forward, axis=-1, keepdims=True)
            if np.isfinite(forward).all() and not np.all(forward_norm < 1e-8):
                candidates['across'] = forward / np.where(forward_norm < 1e-8, 1.0, forward_norm)

    return candidates


def _score_facing_candidates(candidates):
    if not candidates:
        return None, None, None, None

    target = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    best_name = None
    best_forward = None
    best_score = None
    best_tiebreak = None

    for name, forward in candidates.items():
        root_quat = Quaternions.between(forward, target)
        angles = []
        for other_forward in candidates.values():
            aligned = root_quat * other_forward
            angles.append(_vector_angle_deg(aligned[0], target[0]))
        score = float(np.median(angles))
        tiebreak = float(np.mean(angles))
        if best_score is None or score < best_score - 1e-6 or (abs(score - best_score) <= 1e-6 and tiebreak < best_tiebreak):
            best_name = name
            best_forward = forward
            best_score = score
            best_tiebreak = tiebreak

    return best_name, best_forward, best_score, best_tiebreak


def _choose_facing_forward(candidates):
    best_name, best_forward, _best_score, _best_tiebreak = _score_facing_candidates(candidates)
    return best_name, best_forward


def _get_facing_forward(
    joints,
    object_type,
    face_joint_indx=None,
    forward_joint_index=None,
    forward_base_joint_index=None,
):
    _, forward = _choose_facing_forward(
        _get_facing_candidates(
            joints,
            object_type,
            face_joint_indx=face_joint_indx,
            forward_joint_index=forward_joint_index,
            forward_base_joint_index=forward_base_joint_index,
        )
    )
    return forward


def resolve_face_joints(object_type, joint_names=None, parents=None, face_joints=None):
    if face_joints:
        if joint_names is not None and isinstance(face_joints[0], str):
            return [joint_names.index(name) for name in face_joints]
        return list(face_joints)

    # Snakes use CHAIN_FORWARD_JOINTS for direction; _get_facing_candidates
    # returns early for them and never unpacks face_joint_indx.
    if object_type in CHAIN_FORWARD_JOINTS:
        return []

    if joint_names is not None and parents is not None:
        hip_pair = _find_semantic_joint_pair(joint_names, parents, _FACE_JOINT_HIP_PRIORITIES)
        upper_pair = _find_semantic_joint_pair(joint_names, parents, _FACE_JOINT_UPPER_PRIORITIES)
        if hip_pair is not None and upper_pair is not None:
            return [hip_pair[0], hip_pair[1], upper_pair[0], upper_pair[1]]
        # Armless animals (e.g. Raptor in NO_HANDS) have no shoulder joints.
        # Reuse the hip pair as the upper pair so the across-vector and torso
        # direction still work (forward = nose → hip_center remains valid).
        if hip_pair is not None:
            return [hip_pair[0], hip_pair[1], hip_pair[0], hip_pair[1]]

    raise ValueError(
        f"Could not resolve face joints for '{object_type}'. Provide --face_joints_names explicitly or add naming rules."
    )


def get_root_quat(joints, object_type, face_joint_indx=None, forward_joint_index=None, forward_base_joint_index=None):
    if face_joint_indx is None:
        face_joint_indx = resolve_face_joints(object_type)
    forward = _get_facing_forward(
        joints,
        object_type,
        face_joint_indx=face_joint_indx,
        forward_joint_index=forward_joint_index,
        forward_base_joint_index=forward_base_joint_index,
    )
    if forward is None:
        forward = np.array([[0.0, 0.0, 1.0]]).repeat(len(joints), axis=0)
    target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
    root_quat = Quaternions.between(forward, target)
    return root_quat


def _get_hml_orientation_quat(anim, object_type, face_joints=None, orientation_quat=None, forward_joint_index=None, forward_base_joint_index=None):
    return orientation_quat


def rotate_to_hml_orientation(anim, object_type, face_joints=None, orientation_quat=None, forward_joint_index=None, forward_base_joint_index=None):
    qs_rot = _get_hml_orientation_quat(
        anim,
        object_type,
        face_joints=face_joints,
        orientation_quat=orientation_quat,
        forward_joint_index=forward_joint_index,
        forward_base_joint_index=forward_base_joint_index,
    )
    new_rots = anim.rotations.copy()
    new_rots[:, 0] = qs_rot.repeat(new_rots.shape[0], axis=0) * new_rots[:, 0]
    new_pos = anim.positions.copy()
    new_pos[:, 0] = qs_rot.repeat(new_rots.shape[0], axis=0) * new_pos[:, 0]
    new_anim = Animation(new_rots, new_pos, anim.orients.copy(), anim.offsets.copy(), anim.parents.copy())
    return new_anim
