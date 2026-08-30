"""Face orientation and forward direction detection utilities."""

import numpy as np
import re
from motion_lib.Quaternions import Quaternions
from motion_lib.Animation import Animation
from .dataset_tags import dataset_tags
from .ignore_warnings import skip_orientation_detection
from .physics_joint_annotation import (
    normalize_joint_name,
    detect_joint_side,
    _joint_depths,
    strip_joint_name_prefix,
    effective_canonical_replacements,
)


_EMITTED_DEGENERATE_FACING_WARNINGS = set()
_FACING_NEAR_Y_AXIS_ANGLE_DEG = 15.0
# A forward-reference joint whose bone (distance to its parent) is shorter than
# this fraction of the skeleton's overall extent carries no directional
# information and is treated as geometrically degenerate.
_DEGENERATE_BONE_LENGTH_FRACTION = 1e-3


def _emit_degenerate_facing_warning(object_type, warning_kind, message):
    """Print a facing-estimate warning once per (object, kind).

    The preprocessing warning collectors replace this name to redirect the text
    into their end-of-run summary, so anything that decides whether a warning is
    wanted at all belongs in ``_facing_warning`` above it, not here.
    """
    warning_key = (str(object_type or ''), str(warning_kind))
    if warning_key in _EMITTED_DEGENERATE_FACING_WARNINGS:
        return
    _EMITTED_DEGENERATE_FACING_WARNINGS.add(warning_key)
    print(f"[WARN] {message}")


def _facing_warning(object_type, warning_kind, message):
    """Emit a facing warning unless the dataset opted out of facing detection.

    A dataset carrying ``!skip-orientation-detection`` keeps ``orientation_quat``
    at identity, so every fallback these warnings describe feeds a computation
    nothing consumes -- reporting them would be noise over the whole dataset.
    """
    if skip_orientation_detection():
        return
    _emit_degenerate_facing_warning(object_type, warning_kind, message)


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

def _canonicalize_joint_name(name, replacements=None):
    """Canonicalize a joint name using shared constants from physics_joint_annotation.

    Note: drops single-character tokens (including isolated digits), unlike the
    sibling implementation in physics_joint_annotation which preserves digits.
    This preserves existing face-orientation matching behavior.

    ``replacements`` lets callers pass the skeleton-aware replacement table from
    ``effective_canonical_replacements`` so Japanese-only mappings (kao/kosi/o)
    apply only when the rig is confirmed Japanese-style; defaults to the base
    table otherwise.
    """
    from .physics_joint_annotation import _JAPANESE_NAME_REPLACEMENTS

    if replacements is None:
        replacements = _JAPANESE_NAME_REPLACEMENTS

    # Strip prefix using the shared utility
    stripped = strip_joint_name_prefix(name)

    # Normalize and canonicalize
    split_name = normalize_joint_name(stripped)
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
        elif len(clean_part) == 1:
            continue
        else:
            canonical_parts.append(clean_part.capitalize())
    return ' '.join(canonical_parts) if canonical_parts else name.strip()


# Side half of a quadruped limb code, dropped so a left and a right limb share a
# signature; the fore/hind half is kept as a bare 'f'/'b' so LfLeg01 can only
# pair with RfLeg01. Mirrors _LIMB_CODE_SIGNATURE_TOKENS in
# physics_joint_annotation -- the hind codes were missing here too, so a rig that
# names its limbs Lb/Rb had no hind pair to derive the lateral axis from, and the
# swapped spellings (Fl/Fr, Bl/Br) plus the hexapod middle pair (Lm/Rm) are kept
# in sync with it for the same reason.
_LIMB_CODE_SIGNATURE_TOKENS = {
    'lf': 'f', 'rf': 'f', 'lb': 'b', 'rb': 'b',
    'fl': 'f', 'fr': 'f', 'bl': 'b', 'br': 'b',
    'lm': 'm', 'rm': 'm',
}


def _joint_signature(name, replacements=None):
    signature_tokens = [
        _LIMB_CODE_SIGNATURE_TOKENS.get(token, token)
        for token in _canonicalize_joint_name(name, replacements).lower().split()
        if token not in ('left', 'right')
    ]
    if signature_tokens:
        return ' '.join(signature_tokens)

    fallback_tokens = [
        token for token in normalize_joint_name(name).split()
        if token not in ('left', 'right', 'l', 'r', 'lf', 'rf')
    ]
    return ' '.join(fallback_tokens)


def _face_joint_name_allowed(name):
    normalized = normalize_joint_name(name)
    if any(token in normalized for token in _FACE_JOINT_EXCLUDE_TOKENS):
        return False
    return True


def _find_semantic_joint_pair(joint_names, parents, priorities, *, exclude_near_root=True):
    depths = _joint_depths(parents)
    replacements = effective_canonical_replacements(joint_names)
    candidates = {'right': [], 'left': []}
    paired_candidates = {}

    for joint_index, joint_name in enumerate(joint_names):
        if not _face_joint_name_allowed(joint_name):
            continue
        normalized = normalize_joint_name(_canonicalize_joint_name(joint_name, replacements))
        if exclude_near_root and any(token in normalized for token in _FACE_JOINT_NEAR_ROOT_EXCLUDE_TOKENS):
            continue

        side = detect_joint_side(joint_name)
        if side is None:
            continue

        priority_index = None
        for current_priority, keyword_group in enumerate(priorities):
            if any(keyword in normalized for keyword in keyword_group):
                priority_index = current_priority
                break
        if priority_index is None:
            continue

        candidate = (priority_index, depths[joint_index], joint_index)
        candidates[side].append(candidate)

        signature = _joint_signature(joint_name, replacements)
        if signature:
            if signature not in paired_candidates:
                paired_candidates[signature] = {'right': [], 'left': []}
            paired_candidates[signature][side].append(candidate)

    if not candidates['right'] or not candidates['left']:
        return None

    signature_pairs = []
    for signature, signature_candidates in paired_candidates.items():
        if not signature_candidates['right'] or not signature_candidates['left']:
            continue

        right_candidate = min(signature_candidates['right'])
        left_candidate = min(signature_candidates['left'])
        signature_pairs.append(
            (
                max(right_candidate[0], left_candidate[0]),
                max(right_candidate[1], left_candidate[1]),
                min(right_candidate[2], left_candidate[2]),
                max(right_candidate[2], left_candidate[2]),
                signature,
                right_candidate[2],
                left_candidate[2],
            )
        )

    if signature_pairs:
        _priority, _depth, _min_index, _max_index, _signature, right_index, left_index = min(signature_pairs)
        return right_index, left_index

    right_index = min(candidates['right'])[2]
    left_index = min(candidates['left'])[2]
    return right_index, left_index


def _find_forward_reference_joint(joint_names, parents, rest_positions=None):
    depths = _joint_depths(parents)
    replacements = effective_canonical_replacements(joint_names)

    # When rest positions are available, measure a skeleton scale so bone lengths
    # can be judged degenerate relative to the body's overall size (unit-agnostic).
    pos = _rest_positions_2d(rest_positions)
    scale = None
    if pos is not None:
        centered = pos - pos.mean(axis=0, keepdims=True)
        scale = float(np.linalg.norm(centered, axis=1).max())
        if scale < 1e-8:
            pos = None  # degenerate skeleton overall: cannot judge bone lengths

    candidates = []

    for joint_index, joint_name in enumerate(joint_names):
        normalized = normalize_joint_name(_canonicalize_joint_name(joint_name, replacements))
        if 'nub' in normalized:
            continue
        priority_index = None
        for current_priority, keyword_group in enumerate(_FORWARD_REFERENCE_PRIORITIES):
            if any(keyword in normalized for keyword in keyword_group):
                priority_index = current_priority
                break
        if priority_index is None:
            continue
        # Skip geometrically degenerate references: a joint coincident with its
        # parent (zero-length bone, e.g. a placeholder 'Neck' sitting on the hips)
        # carries no directional information. Picking it yields a meaningless
        # forward vector that can silently reverse the facing; skipping it lets the
        # search fall through to a deeper head/neck joint or, failing that, the
        # tail->spine body-axis fallback.
        if pos is not None:
            parent_index = int(parents[joint_index])
            if parent_index >= 0:
                bone_length = float(np.linalg.norm(pos[joint_index] - pos[parent_index]))
                if bone_length < _DEGENERATE_BONE_LENGTH_FRACTION * scale:
                    continue
        candidates.append((priority_index, -depths[joint_index], joint_index))

    if not candidates:
        return None

    return min(candidates)[2]


def _find_centerline_reference_joint(joint_names, parents, priorities, *, prefer_deepest):
    depths = _joint_depths(parents)
    replacements = effective_canonical_replacements(joint_names)
    candidates = []

    for joint_index, joint_name in enumerate(joint_names):
        if detect_joint_side(joint_name) is not None:
            continue

        normalized = normalize_joint_name(_canonicalize_joint_name(joint_name, replacements))
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


def resolve_forward_reference_joints(joint_names, parents, object_type=None, rest_positions=None):
    forward_joint_index = _find_forward_reference_joint(joint_names, parents, rest_positions=rest_positions)

    if forward_joint_index is not None:
        return forward_joint_index, None

    body_axis_forward_joint = _find_body_axis_forward_joint(joint_names, parents)
    body_axis_base_joint = _find_body_axis_base_joint(joint_names, parents)
    if body_axis_forward_joint is None or body_axis_base_joint is None or body_axis_forward_joint == body_axis_base_joint:
        return None, None

    prefix = f'{object_type}: ' if object_type else ''
    _facing_warning(
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
    chain = dataset_tags().chain_forward_for(object_type)
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

    if dataset_tags().chain_forward_for(object_type) is not None:
        chain_forward, chain_near_y = _get_chain_forward(joints, object_type)
        if chain_forward is None:
            return {}, {}
        return {'chain': chain_forward}, {'chain': chain_near_y}

    if forward_base_joint_index is None:
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
            _facing_warning(
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


def _collect_homologous_pairs(joint_names, parents, *, allow_noisy):
    """All left/right homologous joint pairs, grouped by shared signature.

    Returns ``[(pair_depth, right_index, left_index), ...]`` sorted shallowest
    first (closest to the root). Any joint with a detectable side and a non-empty
    signature qualifies, so calves, hands, wings, etc. count even when they miss
    the narrow hip/shoulder keyword lists. ``allow_noisy=False`` skips the distal
    / unreliable joints (feet, toes, ears, ...) filtered by
    ``_face_joint_name_allowed``; ``True`` keeps them as a last resort, since even
    a foot pair still pins the left-right axis.
    """
    depths = _joint_depths(parents)
    replacements = effective_canonical_replacements(joint_names)
    by_signature = {}

    for joint_index, joint_name in enumerate(joint_names):
        if not allow_noisy and not _face_joint_name_allowed(joint_name):
            continue
        side = detect_joint_side(joint_name)
        if side is None:
            continue
        signature = _joint_signature(joint_name, replacements)
        if not signature:
            continue
        bucket = by_signature.setdefault(signature, {'right': [], 'left': []})
        bucket[side].append((depths[joint_index], joint_index))

    pairs = []
    for sides in by_signature.values():
        if not sides['right'] or not sides['left']:
            continue
        right_depth, right_index = min(sides['right'])
        left_depth, left_index = min(sides['left'])
        pairs.append((max(right_depth, left_depth), right_index, left_index))

    pairs.sort()
    return pairs


def _rest_positions_2d(rest_positions):
    """Coerce rest positions to a finite ``(J, 3)`` array, or None."""
    if rest_positions is None:
        return None
    pos = np.asarray(rest_positions, dtype=np.float64)
    if pos.ndim == 3:
        pos = pos[0]
    if pos.ndim != 2 or pos.shape[0] < 2 or pos.shape[1] != 3 or not np.isfinite(pos).all():
        return None
    return pos


def _find_generic_lateral_pairs(joint_names, parents, rest_positions=None):
    """Two homologous L/R pairs (lower, upper) when the semantic search misses.

    Falls back from name-keyword matching to *any* symmetric pair. When rest
    positions are available the pairs are ranked by their right-left separation
    (a wider gap is a stronger, less noise-sensitive lateral signal); otherwise
    by hierarchy depth. Returns ``(lower_pair, upper_pair)`` where each pair is
    ``(right_index, left_index)``, or ``(None, None)`` when nothing is found.
    """
    pairs = _collect_homologous_pairs(joint_names, parents, allow_noisy=False)
    if not pairs:
        pairs = _collect_homologous_pairs(joint_names, parents, allow_noisy=True)
    if not pairs:
        return None, None

    pos = _rest_positions_2d(rest_positions)
    if pos is not None:
        pairs = sorted(
            pairs,
            key=lambda pair: float(np.linalg.norm(pos[pair[1]] - pos[pair[2]])),
            reverse=True,
        )

    primary = pairs[0]
    upper = (primary[1], primary[2])

    # A second, distinct pair stabilizes the averaged across-vector; reuse the
    # primary pair when the skeleton only exposes one symmetric pair.
    lower = upper
    for candidate in pairs[1:]:
        if candidate[1] != primary[1] and candidate[2] != primary[2]:
            lower = (candidate[1], candidate[2])
            break
    return lower, upper


def _find_mirror_symmetry_pair(rest_positions):
    """Estimate the lateral axis from bilateral symmetry of the rest pose.

    Name-independent last resort: searches candidate sagittal-plane normals in
    the XZ (ground) plane and keeps the one under which joints best mirror onto
    one another, returning ``(right_index, left_index)`` of the widest reliable
    mirror pair (or None when no usable symmetry exists). Right/left is assigned
    by the sign of the lateral coordinate — arbitrary but internally consistent.
    The head/tail forward references, not this sign, fix the final facing; the
    across-vector derived from this pair is only consulted as a last resort, so a
    flipped guess cannot silently reverse a skeleton that has a head reference.
    """
    pos = _rest_positions_2d(rest_positions)
    if pos is None:
        return None

    centered = pos - pos.mean(axis=0, keepdims=True)
    scale = float(np.linalg.norm(centered, axis=1).max())
    if scale < 1e-8:
        return None
    n_joints = centered.shape[0]

    best_score = None
    best_pair = None
    # The plane normal and its negation are equivalent, so [0, pi) covers every
    # orientation; 1-degree steps are ample for a 90-degree-snapped result.
    for theta in np.linspace(0.0, np.pi, 180, endpoint=False):
        axis = np.array([np.cos(theta), 0.0, np.sin(theta)])
        lateral = centered @ axis
        reflected = centered - 2.0 * lateral[:, None] * axis[None, :]

        total_err = 0.0
        matched = 0
        widest_sep = -1.0
        widest_pair = None
        for i in range(n_joints):
            if abs(lateral[i]) < 0.05 * scale:
                continue  # on the midline: no lateral information
            distances = np.linalg.norm(centered - reflected[i][None, :], axis=1)
            distances[i] = np.inf
            j = int(np.argmin(distances))
            if lateral[i] * lateral[j] >= 0:
                continue  # mirror partner must sit on the opposite side
            err = float(distances[j]) / scale
            if err > 0.15:
                continue
            total_err += err
            matched += 1
            separation = abs(lateral[i]) + abs(lateral[j])
            if separation > widest_sep:
                widest_sep = separation
                widest_pair = (i, j) if lateral[i] >= lateral[j] else (j, i)

        if widest_pair is None:
            continue
        # Prefer planes that explain more joints at lower average error.
        score = (total_err / matched, -matched)
        if best_score is None or score < best_score:
            best_score = score
            best_pair = widest_pair

    return best_pair


def resolve_face_joints(object_type, joint_names=None, parents=None, face_joints=None, rest_positions=None):
    if face_joints:
        if joint_names is not None and isinstance(face_joints[0], str):
            return [joint_names.index(name) for name in face_joints]
        return list(face_joints)

    # Species with a chain_forward_joints entry take their direction from that
    # chain; _get_facing_candidates returns early for them and never unpacks
    # face_joint_indx.
    if dataset_tags().chain_forward_for(object_type) is not None:
        return []

    if joint_names is not None and parents is not None:
        hip_pair = _find_semantic_joint_pair(joint_names, parents, _FACE_JOINT_HIP_PRIORITIES)
        upper_pair = _find_semantic_joint_pair(joint_names, parents, _FACE_JOINT_UPPER_PRIORITIES)

        # Either girdle alone already pins a valid lateral axis. When one (or
        # both) keyword-based pairs is missing, fill the empty slot from any
        # homologous left/right pair (calves, hands, wings, ...) so a partially
        # or oddly named skeleton still yields an orientation instead of the
        # blind +Z fallback. Armless animals (e.g. Raptor in NO_HANDS) naturally
        # resolve here by reusing their leg pair for the upper slot.
        if hip_pair is None or upper_pair is None:
            generic_lower, generic_upper = _find_generic_lateral_pairs(
                joint_names, parents, rest_positions=rest_positions
            )
            if hip_pair is None:
                hip_pair = generic_lower or generic_upper
            if upper_pair is None:
                # When a semantic hip pair is already found, reuse it for the
                # upper (shoulder) slot rather than taking the widest generic
                # homologous pair, which may be a distal limb tip (leg 末端)
                # whose global position shifts dramatically during motion and
                # corrupts the torso_head forward calculation.
                upper_pair = hip_pair if hip_pair is not None else (generic_upper or generic_lower)

        if hip_pair is not None and upper_pair is not None:
            return [hip_pair[0], hip_pair[1], upper_pair[0], upper_pair[1]]
        if hip_pair is not None:
            return [hip_pair[0], hip_pair[1], hip_pair[0], hip_pair[1]]
        if upper_pair is not None:
            return [upper_pair[0], upper_pair[1], upper_pair[0], upper_pair[1]]

        # No named sides anywhere: recover the lateral axis from the rest pose's
        # bilateral symmetry directly. Strictly better than a blind +Z guess —
        # the axis is correct even though right/left is arbitrary.
        mirror_pair = _find_mirror_symmetry_pair(rest_positions)
        if mirror_pair is not None:
            _facing_warning(
                object_type,
                'geometric_mirror',
                f"{object_type}: no named left-right joint pairs found; estimated the "
                "lateral axis from rest-pose mirror symmetry. Pass --face-joints-names "
                "if the facing looks wrong.",
            )
            return [mirror_pair[0], mirror_pair[1], mirror_pair[0], mirror_pair[1]]

    # Asymmetric or incomplete skeletons (e.g., legs-only procedural test
    # rigs) have no left-right pairs at all.  Fall back to an empty list so
    # that _get_facing_candidates skips across/torso_head heuristics and
    # calculate_root_quat uses the default +Z forward direction.
    _facing_warning(
        object_type,
        'no_pairs',
        f"{object_type}: no left-right joint pairs found; using default +Z orientation. "
        "Provide --face-joints-names explicitly if a different orientation is needed.",
    )
    return []


_Y_AXIS = np.array([0.0, 1.0, 0.0])


def _forward_to_y_angle(forward):
    """Angle (rad) about +Y that rotates +Z onto the given XZ-plane forward."""
    forward = np.asarray(forward, dtype=np.float64).reshape(-1, 3)
    return np.arctan2(forward[:, 0], forward[:, 2])


def _snap_angle_to_quarter_turn(angles):
    """Round each angle (rad) to the nearest multiple of 90 degrees."""
    quarter = np.pi / 2.0
    return np.round(np.asarray(angles, dtype=np.float64) / quarter) * quarter


def _y_rotation_quat(angles):
    """Pure +Y rotation quaternion for each angle (rad)."""
    angles = np.atleast_1d(np.asarray(angles, dtype=np.float64))
    axis = np.broadcast_to(_Y_AXIS, (angles.shape[0], 3))
    return Quaternions.from_angle_axis(angles, axis)


def snap_forward_alignment_quat(source_forward, target_forward):
    """Quaternion aligning ``source``'s dominant axis onto ``target``'s.

    Both forwards must already be projected onto the XZ plane.  Each forward is
    snapped to its nearest cardinal axis (+X/-X/+Z/-Z) before the rotation is
    measured, so the result is restricted to a multiple of 90 degrees about +Y.
    Using snapped angles (instead of ``Quaternions.between`` on the snapped
    vectors) also avoids the degenerate zero-quaternion that ``between`` returns
    for antiparallel vectors, e.g. the 180-degree case.
    """
    source = _snap_angle_to_quarter_turn(_forward_to_y_angle(source_forward))
    target = _snap_angle_to_quarter_turn(_forward_to_y_angle(target_forward))
    return _y_rotation_quat(target - source)


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
    # Align the body's dominant axis to +Z using only a 90-degree-multiple turn,
    # collapsing the saved orientation_quat to one of four possible values.
    snapped = _snap_angle_to_quarter_turn(_forward_to_y_angle(forward))
    return _y_rotation_quat(-snapped)


def rotate_to_hml_orientation(anim, orientation_quat):
    qs_rot = orientation_quat
    new_rots = anim.rotations.copy()
    new_rots[:, 0] = qs_rot.repeat(new_rots.shape[0], axis=0) * new_rots[:, 0]
    new_pos = anim.positions.copy()
    new_pos[:, 0] = qs_rot.repeat(new_rots.shape[0], axis=0) * new_pos[:, 0]
    new_anim = Animation(new_rots, new_pos, anim.orients.copy(), anim.offsets.copy(), anim.parents.copy())
    return new_anim
