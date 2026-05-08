"""Face orientation and forward direction detection utilities."""

import numpy as np
import re
from motion_lib.Quaternions import Quaternions
from motion_lib.Animation import Animation
from .param_utils import CHAIN_FORWARD_JOINTS


_EMITTED_DEGENERATE_FACING_WARNINGS = set()
_FACING_NEAR_Y_AXIS_ANGLE_DEG = 15.0


def _emit_degenerate_facing_warning(object_type, warning_kind, message):
    warning_key = (str(object_type or ''), str(warning_kind))
    if warning_key in _EMITTED_DEGENERATE_FACING_WARNINGS:
        return
    _EMITTED_DEGENERATE_FACING_WARNINGS.add(warning_key)
    print(f"[WARN] {message}")


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
_BODY_AXIS_FORWARD_PRIORITIES = (
    ('spine',),
    ('chest', 'thorax', 'sternum'),
    ('pelvis', 'hips', 'hip'),
)
_BODY_AXIS_BASE_PRIORITIES = (
    ('tail',),
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
        if 'nub' in normalized:
            continue
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


def _find_centerline_reference_joint(joint_names, parents, priorities, *, prefer_deepest):
    depths = _joint_depths(parents)
    candidates = []

    for joint_index, joint_name in enumerate(joint_names):
        if _detect_joint_side(joint_name) is not None:
            continue

        normalized = _normalize_joint_name(_canonicalize_joint_name(joint_name))
        normalized_tokens = set(normalized.split())
        priority_index = None
        for current_priority, keyword_group in enumerate(priorities):
            if any(keyword in normalized_tokens for keyword in keyword_group):
                priority_index = current_priority
                break
        if priority_index is None:
            continue

        depth_rank = -depths[joint_index] if prefer_deepest else depths[joint_index]
        candidates.append((priority_index, depth_rank, joint_index))

    if not candidates:
        return None

    return min(candidates)[2]


def _find_body_axis_forward_joint(joint_names, parents):
    return _find_centerline_reference_joint(
        joint_names,
        parents,
        _BODY_AXIS_FORWARD_PRIORITIES,
        prefer_deepest=True,
    )


def _find_body_axis_base_joint(joint_names, parents):
    return _find_centerline_reference_joint(
        joint_names,
        parents,
        _BODY_AXIS_BASE_PRIORITIES,
        prefer_deepest=False,
    )


def resolve_forward_reference_joints(joint_names, parents, object_type=None):
    forward_joint_index = _find_forward_reference_joint(joint_names, parents)

    if forward_joint_index is not None:
        return forward_joint_index, None

    body_axis_forward_joint = _find_body_axis_forward_joint(joint_names, parents)
    body_axis_base_joint = _find_body_axis_base_joint(joint_names, parents)
    if body_axis_forward_joint is None or body_axis_base_joint is None or body_axis_forward_joint == body_axis_base_joint:
        return None, None

    prefix = f'{object_type}: ' if object_type else ''
    _emit_degenerate_facing_warning(
        object_type,
        'tail_spine_fallback',
        f"{prefix}no head/neck forward reference was found; falling back to tail->spine body-axis orientation.",
    )

    return body_axis_forward_joint, body_axis_base_joint


def _normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return vectors / norms


def _project_forward_to_xz(vectors):
    projected = np.asarray(vectors, dtype=np.float64).copy()
    projected[..., 1] = 0.0
    norms = np.linalg.norm(projected, axis=-1, keepdims=True)
    if np.all(norms < 1e-8):
        return None
    return projected / np.where(norms < 1e-8, 1.0, norms)


def _is_forward_near_y_axis(vectors, angle_threshold_deg=_FACING_NEAR_Y_AXIS_ANGLE_DEG):
    raw_vectors = np.asarray(vectors, dtype=np.float64)
    if raw_vectors.ndim == 1:
        raw_vectors = raw_vectors[None, :]
    if not np.isfinite(raw_vectors).all():
        return True

    norms = np.linalg.norm(raw_vectors, axis=-1)
    valid = norms > 1e-8
    if not np.any(valid):
        return True

    cos_threshold = float(np.cos(np.deg2rad(angle_threshold_deg)))
    cos_to_y = np.abs(raw_vectors[valid, 1] / norms[valid])
    return bool(np.any(cos_to_y >= cos_threshold))


def _build_forward_candidate(vectors):
    if vectors is None:
        return None, True
    if not np.isfinite(vectors).all():
        return None, True

    projected = _project_forward_to_xz(vectors)
    if projected is None:
        return None, True
    return projected, _is_forward_near_y_axis(vectors)


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
        return None, True

    if len(chain) == 2:
        neck, head = chain
        forward = joints[:, head] - joints[:, neck]
    else:
        body_base, neck, head = chain
        forward = (joints[:, head] - joints[:, neck]) + (joints[:, neck] - joints[:, body_base])
    return _build_forward_candidate(forward)


def _get_head_forward(joints, face_joint_indx, forward_joint_index, forward_base_joint_index=None):
    if forward_joint_index is None:
        return None, True

    if forward_base_joint_index is not None and forward_joint_index == forward_base_joint_index:
        return None, True

    if forward_base_joint_index is not None:
        forward = joints[:, forward_joint_index] - joints[:, forward_base_joint_index]
    else:
        if not face_joint_indx:
            return None, True
        r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
        hip_center = 0.5 * (joints[:, r_hip] + joints[:, l_hip])
        shoulder_center = 0.5 * (joints[:, sdr_r] + joints[:, sdr_l])
        torso_center = 0.5 * (hip_center + shoulder_center)
        forward = joints[:, forward_joint_index] - torso_center
    return _build_forward_candidate(forward)


def _get_across_forward(joints, face_joint_indx):
    if not face_joint_indx:
        return None

    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = joints[:, r_hip] - joints[:, l_hip]
    across2 = joints[:, sdr_r] - joints[:, sdr_l]
    across = across1 + across2
    across_norm = np.linalg.norm(across, axis=-1, keepdims=True)
    if not np.isfinite(across).all() or np.all(across_norm < 1e-8):
        return None

    across = across / np.where(across_norm < 1e-8, 1.0, across_norm)
    forward = np.cross(np.array([[0.0, 1.0, 0.0]]), across, axis=-1)
    if not np.isfinite(forward).all():
        return None
    return _project_forward_to_xz(forward)


def _get_facing_candidates_with_diagnostics(
    joints,
    object_type,
    face_joint_indx=None,
    forward_joint_index=None,
    forward_base_joint_index=None,
    emit_warnings=True,
):
    candidates = {}
    near_y_candidates = {}

    if object_type in CHAIN_FORWARD_JOINTS:
        chain_forward, chain_near_y = _get_chain_forward(joints, object_type)
        if chain_forward is None:
            return {}, {}
        return {'chain': chain_forward}, {'chain': chain_near_y}

    torso_head, torso_head_near_y = _get_head_forward(
        joints,
        face_joint_indx,
        forward_joint_index,
        forward_base_joint_index=None,
    )
    if torso_head is not None:
        candidates['torso_head'] = torso_head
        near_y_candidates['torso_head'] = torso_head_near_y

    tail_spine, tail_spine_near_y = _get_head_forward(
        joints,
        face_joint_indx,
        forward_joint_index,
        forward_base_joint_index=forward_base_joint_index,
    )
    if tail_spine is not None:
        candidates['tail_spine'] = tail_spine
        near_y_candidates['tail_spine'] = tail_spine_near_y

    across_forward = _get_across_forward(joints, face_joint_indx)
    if across_forward is not None:
        candidates['across'] = across_forward
        near_y_candidates['across'] = False

    if not candidates and emit_warnings:
        _emit_degenerate_facing_warning(
            object_type,
            'no_candidates',
            f"{object_type}: orientation calculation produced no valid facing candidates; falling back to +Z.",
        )

    return candidates, near_y_candidates


def _get_facing_candidates(
    joints,
    object_type,
    face_joint_indx=None,
    forward_joint_index=None,
    forward_base_joint_index=None,
    emit_warnings=True,
):
    candidates, _near_y_candidates = _get_facing_candidates_with_diagnostics(
        joints,
        object_type,
        face_joint_indx=face_joint_indx,
        forward_joint_index=forward_joint_index,
        forward_base_joint_index=forward_base_joint_index,
        emit_warnings=emit_warnings,
    )
    return candidates


_PRIMARY_FACING_CANDIDATE_PRIORITY = (
    'chain',
    'torso_head',
    'tail_spine',
)


def _choose_facing_forward(candidates, object_type=None, near_y_candidates=None, emit_warnings=True):
    near_y_candidates = dict(near_y_candidates or {})

    selected_name = None
    selected_forward = None
    for candidate_name in _PRIMARY_FACING_CANDIDATE_PRIORITY:
        forward = candidates.get(candidate_name)
        if forward is None:
            continue
        selected_name = candidate_name
        selected_forward = forward
        break

    if selected_name is not None and not near_y_candidates.get(selected_name, False):
        return selected_name, selected_forward

    across_forward = candidates.get('across')
    if across_forward is not None:
        if emit_warnings:
            _emit_degenerate_facing_warning(
                object_type,
                'across_selected',
                f"{object_type}: orientation calculation fell back to the across-vector heuristic because higher-priority forward references were unavailable or near-parallel to the Y axis.",
            )
        return 'across', across_forward

    return selected_name, selected_forward


def _get_facing_forward(
    joints,
    object_type,
    face_joint_indx=None,
    forward_joint_index=None,
    forward_base_joint_index=None,
    emit_warnings=True,
):
    candidates, near_y_candidates = _get_facing_candidates_with_diagnostics(
        joints,
        object_type,
        face_joint_indx=face_joint_indx,
        forward_joint_index=forward_joint_index,
        forward_base_joint_index=forward_base_joint_index,
        emit_warnings=emit_warnings,
    )
    _, forward = _choose_facing_forward(
        candidates,
        object_type=object_type,
        near_y_candidates=near_y_candidates,
        emit_warnings=emit_warnings,
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
        f"Could not resolve face joints for '{object_type}'. Provide --face-joints-names explicitly or add naming rules."
    )


def calculate_root_quat(joints, object_type, face_joint_indx=None, forward_joint_index=None, forward_base_joint_index=None, emit_warnings=True):
    if face_joint_indx is None:
        face_joint_indx = resolve_face_joints(object_type)
    forward = _get_facing_forward(
        joints,
        object_type,
        face_joint_indx=face_joint_indx,
        forward_joint_index=forward_joint_index,
        forward_base_joint_index=forward_base_joint_index,
        emit_warnings=emit_warnings,
    )
    if forward is None:
        forward = np.array([[0.0, 0.0, 1.0]]).repeat(len(joints), axis=0)
    target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
    root_quat = Quaternions.between(forward, target)
    return root_quat


def rotate_to_hml_orientation(anim, orientation_quat):
    qs_rot = orientation_quat
    new_rots = anim.rotations.copy()
    new_rots[:, 0] = qs_rot.repeat(new_rots.shape[0], axis=0) * new_rots[:, 0]
    new_pos = anim.positions.copy()
    new_pos[:, 0] = qs_rot.repeat(new_rots.shape[0], axis=0) * new_pos[:, 0]
    new_anim = Animation(new_rots, new_pos, anim.orients.copy(), anim.offsets.copy(), anim.parents.copy())
    return new_anim
