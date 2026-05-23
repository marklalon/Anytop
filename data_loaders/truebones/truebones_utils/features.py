"""Feature extraction & motion recovery.

Middle layer of the motion-processing pipeline. Extracts feature tensors from
animations, recovers animations from features, analyses T-poses, and infers
translation root indices.

Depends on: animation_utils.py
"""

from dataclasses import dataclass

from motion_lib import FBX, Animation, Quaternions
from motion_lib.Animation import positions_global, rotations_global, offsets_from_positions
import numpy as np
import os
from os.path import join as pjoin
import torch
from data_loaders.truebones.truebones_utils.param_utils import (
    FOOT_CONTACT_HEIGHT_THRESH,
    MAX_JOINTS,
    FOOT_CONTACT_VEL_THRESH,
)
from Anytop.utils.rotation_conversions import rotation_6d_to_matrix_np
from .physics_joint_annotation import (
    infer_contact_joints,
    detect_joint_side,
)
from .face_orientation import (
    resolve_face_joints,
    calculate_root_quat,
    rotate_to_hml_orientation,
    resolve_forward_reference_joints,
)

from .animation_utils import (
    ROOT_XZ_STRIP_THRESHOLD,
    detect_motion_loop,
    find_translation_root,
    bake_descendant_y_into_translation_root,
    clamp_vertical_trajectory,
    move_xz_to_origin,
    xz_locomotion_extent,
    strip_translation_root_xz,
    resolve_detected_translation_root_index,
    needs_bvh_position_channels,
    reorder_animation_to_dfs,
    get_average_axial_bone_length,
    get_rest_body_max_span,
    compute_scale_factor,
    scale_anim,
    build_leaf_rotation_helper_metadata,
    append_leaf_rotation_helpers_to_animation,
    compute_rots_from_tpos,
    solve_local_positions_for_target_global,
    warn_mirror_disabled_subtrees,
    neutralize_animation_subtrees,
)


################## Contact & Feature Building #####################

"""Compute framewise binary contact states for the provided contact-capable joints."""
def get_contact_state(positions, contact_joint_indices, vel_thresh):
    frames_num, joints_num = positions.shape[:2]
    contact_joint_indices = np.asarray(contact_joint_indices, dtype=np.int64)
    if contact_joint_indices.size == 0:
        return np.zeros((frames_num - 1, joints_num))

    foot_vel_x = (positions[1:, contact_joint_indices, 0] - positions[:-1, contact_joint_indices, 0]) ** 2
    foot_vel_y = (positions[1:, contact_joint_indices, 1] - positions[:-1, contact_joint_indices, 1]) ** 2
    foot_vel_z = (positions[1:, contact_joint_indices, 2] - positions[:-1, contact_joint_indices, 2]) ** 2
    total_vel = foot_vel_x + foot_vel_y + foot_vel_z
    foot_floor = np.percentile(positions[:, contact_joint_indices, 1], 5.0, axis=0, keepdims=True)
    relative_height = positions[1:, contact_joint_indices, 1] - foot_floor
    foot_contact_vel_map = np.where(
        np.logical_and(total_vel <= vel_thresh, np.abs(relative_height) <= FOOT_CONTACT_HEIGHT_THRESH),
        1,
        0,
    )
    foot_cont = np.zeros((frames_num-1, joints_num))
    foot_cont[:, contact_joint_indices] = foot_contact_vel_map.astype(int)

    return foot_cont


def get_terminal_contact_state(positions, contact_joint_indices, vel_thresh, is_loop):
    """Compute the terminal-frame contact row for feature export.

    For looping clips we evaluate the wrap-around transition from the last frame
    back to the first frame. For non-looping clips we emit zeros, matching the
    user's requested terminal-velocity semantics.
    """
    joints_num = positions.shape[1]
    terminal_contact = np.zeros((joints_num,), dtype=np.float32)
    contact_joint_indices = np.asarray(contact_joint_indices, dtype=np.int64)
    if not is_loop or positions.shape[0] < 2 or contact_joint_indices.size == 0:
        return terminal_contact

    foot_delta = positions[0, contact_joint_indices] - positions[-1, contact_joint_indices]
    total_vel = np.sum(foot_delta ** 2, axis=-1)
    foot_floor = np.percentile(positions[:, contact_joint_indices, 1], 5.0, axis=0)
    relative_height = positions[0, contact_joint_indices, 1] - foot_floor
    terminal_contact[contact_joint_indices] = np.where(
        np.logical_and(total_vel <= vel_thresh, np.abs(relative_height) <= FOOT_CONTACT_HEIGHT_THRESH),
        1.0,
        0.0,
    )
    return terminal_contact


def get_foot_contact(positions, foot_joints_indices, vel_thresh):
    return get_contact_state(positions, foot_joints_indices, vel_thresh)


""" get 6d rotations continuous representation"""
def get_6d_rep(qs):
    qs_ = qs.copy()
    return qs_.rotation_matrix(cont6d=True)


def _compute_terminal_local_velocity(global_positions, root_rot, is_loop, prev_frame_velocity=None):
    """Return the final per-joint velocity row for exported features.

    Looping clips use the wrap-around delta last->first, expressed in the first
    frame's root coordinate system. Non-looping clips use the velocity from the
    previous frame if provided, otherwise emit zeros.
    """
    terminal_velocity = np.zeros((global_positions.shape[1], 3), dtype=global_positions.dtype)
    if global_positions.shape[0] < 2:
        return terminal_velocity
    
    if is_loop:
        wrap_delta = global_positions[0] - global_positions[-1]
        terminal_velocity = np.repeat(root_rot[0:1], global_positions.shape[1], axis=0) * wrap_delta
    elif prev_frame_velocity is not None:
        terminal_velocity = prev_frame_velocity
    
    return terminal_velocity


'''return positions in root coords system. Meaning, each frame faces Z+, and the root is at [0, root_height, 0]'''
def get_rifke(global_positions, root_rot, translation_root_index=0):
    positions = global_positions.copy()
    '''Local pose'''
    positions[..., 0] -= positions[:, translation_root_index:translation_root_index + 1, 0]
    positions[..., 2] -= positions[:, translation_root_index:translation_root_index + 1, 2]
    '''All pose face Z+'''
    positions = np.repeat(root_rot[:, None], positions.shape[1], axis=1) * positions
    return positions


def get_motion_features(ric_positions, rotations, foot_contact, velocity, terminal_velocity, terminal_contact, max_joints):
    # F = Frames# , J = joints# 
    # parents (J,1)
    # positions (F, J, 3)
    # rotations (F, J, 6)
    # foot_contact (F - 1, J, 1) + one terminal row
    # velocity (F - 1, J, 3) + one terminal row
    # offsets (J, 3)
    
    # feature len = 13 (pos, rot, vel, foot)

    frames, joints = ric_positions.shape[0:2]
    if joints > max_joints:
        max_joints = joints
    pos = ric_positions  ## (Frames, joints, 3)
    rot = rotations ## (Frames, joints, 6)
    vel = np.concatenate([velocity, terminal_velocity[None, ...]], axis=0) ## (Frames, joints, 3)
    foot = np.concatenate([
        foot_contact.reshape(frames - 1, joints, 1),
        terminal_contact.reshape(1, joints, 1),
    ], axis=0) ## (Frames, joints, 1)
    features= np.concatenate([pos, rot, vel, foot], axis=-1) 
    return features, max_joints


""" returns cont6d params, including joints rotations, root rotation and rotational velocity,
linear velocity and positions. Unlike BVH (and accordingly, Animation object) in which the parent holds the rotagtion of the child joint,
in our data structure each joints holds it's own rotation (similar to humanML3D data structure and FK model)"""
def get_bvh_cont6d_params(anim, object_type, orientation_quat, translation_root_index=0):
    positions = positions_global(anim)
    quat_params = anim.rotations
    # ``anim`` is ALREADY canonicalized: process_anim/rotate_to_hml_orientation
    # rotated it by ``orientation_quat`` so the skeleton faces the canonical
    # +Z direction (this single application is yaw-invariant w.r.t. the source
    # FBX authoring). The root-facing used here for RIC de-rotation / the root
    # rotation channel / velocity frame must therefore be IDENTITY. Re-using
    # ``orientation_quat`` a second time applied q twice (canonical = q²·native),
    # which is only self-consistent when every skeleton shares the same q
    # (true for the Truebones family, q≈-90°, but NOT for arbitrary skeletons
    # such as a +Z-authored dragon, q≈identity) and made the stored feature
    # frame skeleton-orientation-dependent instead of normalized. ``orientation_quat``
    # is retained as a parameter for call-site/signature compatibility and is
    # still stored separately in cond for metadata/retarget consumers.
    r_rot = Quaternions.id(positions.shape[0])
    '''Quaternion to continuous 6D'''
    cont_6d_params = get_6d_rep(quat_params)
    cont_6d_params_reordered = np.zeros_like(cont_6d_params)
    for j, p in enumerate(anim.parents[1:], 1):
        cont_6d_params_reordered[:, j] = cont_6d_params[:, p]
    cont_6d_params_reordered[:, 0] = get_6d_rep(r_rot)
    # (seq_len, 4)
    '''Root Linear Velocity'''
    # (seq_len - 1, 3)
    velocity = (positions[1:, translation_root_index] - positions[:-1, translation_root_index]).copy()
    velocity = r_rot[1:] * velocity
    '''Root Angular Velocity'''
    # (seq_len - 1, 4)
    r_velocity = r_rot[1:] * -r_rot[:-1]
    # (seq_len, joints_num, 4)
    return cont_6d_params_reordered, r_velocity, velocity, r_rot, positions


"""" process anim object """
def process_anim(anim, object_type, orientation_quat, root_xz_center=None, *, scale_factor):
    rotated = rotate_to_hml_orientation(anim, orientation_quat)
    baked = bake_descendant_y_into_translation_root(rotated)
    centered, root_xz_center_ = move_xz_to_origin(baked, root_xz_center)
    scaled = scale_anim(centered, scale_factor)
    return scaled, root_xz_center_, scale_factor


################## Translation Root Resolution #####################

def _coerce_translation_root_index(translation_root_index, joint_count=None, context='motion'):
    if translation_root_index is None:
        raise ValueError(f"{context} requires a stored translation_root_index")
    try:
        index = int(translation_root_index)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} has invalid translation_root_index: {translation_root_index}") from exc
    if joint_count is not None and (index < 0 or index >= int(joint_count)):
        raise ValueError(
            f"{context} translation_root_index out of range: {index} for {joint_count} joints"
        )
    return index


def _translation_root_index_from_motion_metadata(motion_metadata, joint_count=None, context='motion metadata'):
    if not isinstance(motion_metadata, dict) or 'translation_root_index' not in motion_metadata:
        return None
    return _coerce_translation_root_index(
        motion_metadata.get('translation_root_index'),
        joint_count=joint_count,
        context=context,
    )


def _require_translation_root_index_from_motion_metadata(motion_metadata, joint_count=None, context='motion metadata'):
    translation_root_index = _translation_root_index_from_motion_metadata(
        motion_metadata,
        joint_count=joint_count,
        context=context,
    )
    if translation_root_index is None:
        raise ValueError(f"{context} requires motion_metadata['translation_root_index']")
    return int(translation_root_index)


def resolve_feature_translation_root_index(
    data,
    *,
    parents=None,
    offsets=None,
    translation_root_index=None,
    motion_metadata=None,
    allow_infer=False,
    anim_pos_threshold=0.01,
    context='motion feature tensor',
):
    motion = np.asarray(data)
    if motion.ndim == 2:
        if translation_root_index is not None:
            return _coerce_translation_root_index(translation_root_index, context=context)
        return _require_translation_root_index_from_motion_metadata(
            motion_metadata,
            context=f'{context} metadata',
        )

    if motion.ndim != 3:
        raise ValueError(f"Expected feature tensor with shape (F, C) or (F, J, C), got {motion.shape}.")

    joint_count = int(motion.shape[1])
    if translation_root_index is not None:
        return _coerce_translation_root_index(
            translation_root_index,
            joint_count=joint_count,
            context=context,
        )

    if not allow_infer:
        return _require_translation_root_index_from_motion_metadata(
            motion_metadata,
            joint_count=joint_count,
            context=f'{context} metadata',
        )

    meta_index = _translation_root_index_from_motion_metadata(
        motion_metadata,
        joint_count=joint_count,
        context=f'{context} metadata',
    )
    if meta_index is not None:
        return int(meta_index)

    if parents is None or offsets is None:
        raise ValueError(
            f'{context} requires translation_root_index, motion_metadata, or parents/offsets to infer it'
        )

    return infer_translation_root_index_from_features(
        motion,
        parents,
        offsets,
        anim_pos_threshold=anim_pos_threshold,
    )


def infer_translation_root_index_from_features(data, parents, offsets, anim_pos_threshold=0.01):
    motion = np.asarray(data)
    if motion.ndim != 3 or motion.shape[1] == 0:
        return 0

    xz_norm = np.linalg.norm(np.asarray(motion[:, :, [0, 2]], dtype=np.float64), axis=-1)
    candidate_order = np.argsort(np.mean(xz_norm, axis=0), kind='stable')
    best_candidate = 0
    best_score = None

    for candidate in candidate_order.tolist():
        try:
            anim, _has_animated_pos = recover_animation_from_motion_np(
                motion,
                parents,
                offsets,
                translation_root_index=int(candidate),
                anim_pos_threshold=anim_pos_threshold,
            )
            detected = find_translation_root(anim)
        except Exception:
            continue

        score = (
            0 if detected == int(candidate) else 1,
            0 if detected >= 0 else 1,
            float(np.mean(xz_norm[:, candidate])),
            int(candidate),
        )
        if best_score is None or score < best_score:
            best_score = score
            best_candidate = int(candidate)
            if detected == int(candidate):
                break

    return int(best_candidate)


################## Mirror #####################

def _neutralize_mirror_disabled_subtrees(
    features,
    object_cond,
    mirrored_offsets,
    *,
    translation_root_index=None,
    motion_metadata=None,
    allow_infer=False,
    anim_pos_threshold=0.01,
):
    disabled_joint_indices = sorted({
        int(index)
        for index in object_cond['mirror_disabled_joint_indices']
        if int(index) > 0
    })
    if not disabled_joint_indices:
        return np.asarray(features).copy()

    motion = np.asarray(features, dtype=np.float32)
    squeeze_frame = False
    if motion.ndim == 2:
        motion = motion[None, ...]
        squeeze_frame = True
    elif motion.ndim != 3:
        raise ValueError(f"Expected motion features with shape (F, J, C) or (J, C), got {motion.shape}.")

    parents = np.asarray(object_cond['parents'], dtype=np.int64)
    offsets = np.asarray(mirrored_offsets, dtype=np.float64)
    resolved_translation_root_index = resolve_feature_translation_root_index(
        motion,
        parents=parents,
        offsets=offsets,
        translation_root_index=translation_root_index,
        motion_metadata=motion_metadata,
        allow_infer=allow_infer,
        anim_pos_threshold=anim_pos_threshold,
        context=f"{object_cond['object_type']} mirrored motion",
    )
    anim, _has_animated_pos = recover_animation_from_motion_np(
        motion,
        parents,
        offsets,
        translation_root_index=resolved_translation_root_index,
        anim_pos_threshold=anim_pos_threshold,
    )
    if anim is None:
        neutralized = motion.copy()
        return neutralized[0] if squeeze_frame else neutralized

    neutral_anim = neutralize_animation_subtrees(anim, disabled_joint_indices)
    contact_joint_indices = list(object_cond['contact_joints'])
    cont_6d_params, _r_velocity, _velocity, r_rot, global_positions = get_bvh_cont6d_params(
        neutral_anim,
        str(object_cond['object_type']),
        object_cond['orientation_quat'],
        translation_root_index=resolved_translation_root_index,
    )
    positions = get_rifke(global_positions, r_rot, translation_root_index=resolved_translation_root_index)
    is_loop = detect_motion_loop(positions)
    local_vel = np.repeat(r_rot[1:, None], global_positions.shape[1], axis=1) * (global_positions[1:] - global_positions[:-1])
    prev_velocity = local_vel[-1] if local_vel.shape[0] > 0 else None
    terminal_local_vel = _compute_terminal_local_velocity(global_positions, r_rot, is_loop, prev_frame_velocity=prev_velocity)
    foot_contact = get_contact_state(global_positions, contact_joint_indices, FOOT_CONTACT_VEL_THRESH)
    terminal_contact = get_terminal_contact_state(global_positions, contact_joint_indices, FOOT_CONTACT_VEL_THRESH, is_loop)
    neutralized, _max_joints = get_motion_features(
        positions,
        cont_6d_params,
        foot_contact,
        local_vel,
        terminal_local_vel,
        terminal_contact,
        motion.shape[1],
    )
    neutralized = neutralized.astype(motion.dtype, copy=False)
    return neutralized[0] if squeeze_frame else neutralized


def mirror_features_with_safeguards(
    features,
    object_cond,
    *,
    translation_root_index=None,
    motion_metadata=None,
    allow_infer=False,
    anim_pos_threshold=0.01,
):
    spi = np.asarray(object_cond['symmetry_partner_indices'], dtype=np.int64)
    perm = np.arange(len(spi), dtype=np.int64)
    valid = spi >= 0
    perm[valid] = spi[valid]

    mirrored = np.asarray(features)[perm].copy() if np.asarray(features).ndim == 2 else np.asarray(features)[:, perm, :].copy()
    mirrored[..., [0, 4, 5, 6, 9]] *= -1

    mirrored_offsets = np.asarray(object_cond['offsets'], dtype=np.float32)[perm].copy()
    mirrored_offsets[:, 0] *= -1

    if object_cond['mirror_disabled_joint_indices']:
        warn_mirror_disabled_subtrees(object_cond)
        # Unpaired joints have no mirror partner so they can't be reflected meaningfully.
        # Restore their original x offsets so that when they are neutralized to rest pose,
        # they retain their original rest orientation rather than the x-flipped version
        # produced by the global mirror pass above.
        disabled_indices = [int(i) for i in object_cond['mirror_disabled_joint_indices'] if int(i) > 0]
        if disabled_indices:
            orig_offsets = np.asarray(object_cond['offsets'], dtype=np.float32)
            mirrored_offsets[disabled_indices, 0] = orig_offsets[disabled_indices, 0]
        mirrored = _neutralize_mirror_disabled_subtrees(
            mirrored,
            object_cond,
            mirrored_offsets,
            translation_root_index=translation_root_index,
            motion_metadata=motion_metadata,
            allow_infer=allow_infer,
            anim_pos_threshold=anim_pos_threshold,
        )

    return mirrored, mirrored_offsets


################## T-Pose & Motion Extraction #####################

""" get object_type common characteristics, extracted from T-pose FBX"""
def get_common_features_from_T_pose(
    t_pose_path,
    object_type,
    face_joints=None,
    *,
    augment_leaf_rotation_helpers=False,
    max_joints=MAX_JOINTS,
):
    t_pose_anim, t_pos_names, _t_pose_frame_time = FBX.load(t_pose_path)
    reference_anim = t_pose_anim[:1] if len(t_pose_anim) > 1 else t_pose_anim
    face_joints = resolve_face_joints(object_type, t_pos_names, reference_anim.parents, face_joints=face_joints)
    forward_joint_index, forward_base_joint_index = resolve_forward_reference_joints(
        t_pos_names,
        reference_anim.parents,
        object_type=object_type,
    )

    reference_positions = positions_global(reference_anim)
    t_pose_orientation_quat = calculate_root_quat(reference_positions, object_type, face_joint_indx=face_joints, forward_joint_index=forward_joint_index, forward_base_joint_index=forward_base_joint_index)[0]

    # Pre-compute the per-character scale factor once from the raw T-pose
    # offsets and reuse it for every motion clip of the same character.
    _tpose_side_labels = []
    for name in t_pos_names:
        detected = detect_joint_side(name)
        _tpose_side_labels.append(detected if detected in ('left', 'right') else 'center')
    axial_avg_len = get_average_axial_bone_length(reference_anim.offsets, reference_anim.parents, _tpose_side_labels)
    reference_body_max_span = get_rest_body_max_span(reference_anim.offsets, reference_anim.parents)
    scale_factor = compute_scale_factor(axial_avg_len, body_max_span=reference_body_max_span)

    scaled, _root_xz_center, scale_factor = process_anim(
        reference_anim,
        object_type,
        t_pose_orientation_quat,
        scale_factor=scale_factor,
    )
    scaled_rest_offsets = offsets_from_positions(positions_global(scaled)[0], scaled.parents)
    helper_metadata = build_leaf_rotation_helper_metadata(
        t_pos_names,
        scaled.parents,
        offsets=scaled_rest_offsets,
        max_joints=max_joints if augment_leaf_rotation_helpers else len(scaled.parents),
    )
    if augment_leaf_rotation_helpers and helper_metadata['helper_joint_count'] > 0:
        scaled, t_pos_names = append_leaf_rotation_helpers_to_animation(
            scaled,
            t_pos_names,
            helper_metadata,
        )
    scaled_positions = positions_global(scaled)
    scaled_rest_positions = scaled_positions[0]
    offsets = offsets_from_positions(scaled_rest_positions, scaled.parents)
    suspected_foot_indices, contact_joint_source = infer_contact_joints(
        t_pos_names,
        scaled.parents,
        scaled_rest_positions,
    )
    return TPoseFeatures(
        scale_factor=scale_factor,
        offsets=offsets,
        foot_indices=suspected_foot_indices,
        tpos_rots=scaled.rotations,
        names=t_pos_names,
        tpos_anim=scaled,
        face_joints=face_joints,
        orientation_quat=t_pose_orientation_quat,
        forward_joint_index=forward_joint_index,
        forward_base_joint_index=forward_base_joint_index,
        contact_joint_source=contact_joint_source,
        axial_avg_len=axial_avg_len,
        helper_metadata=helper_metadata,
    )


@dataclass
class TPoseFeatures:
    """Packaged return from get_common_features_from_T_pose."""
    scale_factor: float
    offsets: np.ndarray
    foot_indices: list
    tpos_rots: np.ndarray
    names: list
    tpos_anim: Animation
    face_joints: list
    orientation_quat: np.ndarray
    forward_joint_index: int
    forward_base_joint_index: int
    contact_joint_source: str
    axial_avg_len: float
    helper_metadata: dict[str, object]


def _extract_motion_features_from_aligned_anims(
    new_anim,
    export_anim,
    foot_contact_vel_thresh,
    object_type,
    max_joints,
    foot_indices,
    orientation_quat,
    translation_root_index,
):
    feature_translation_root_index = int(translation_root_index)
    has_locomotion = False
    motion_anim = new_anim
    motion_export_anim = export_anim
    xz_extent = xz_locomotion_extent(export_anim, feature_translation_root_index)
    has_locomotion = xz_extent > ROOT_XZ_STRIP_THRESHOLD
    if has_locomotion:
        motion_anim = strip_translation_root_xz(new_anim, feature_translation_root_index)
        motion_export_anim = strip_translation_root_xz(export_anim, feature_translation_root_index)

    cont_6d_params, r_velocity, velocity, r_rot, global_positions = get_bvh_cont6d_params(
        motion_anim,
        object_type,
        orientation_quat,
        translation_root_index=feature_translation_root_index,
    )
    foot_contact = get_contact_state(global_positions, foot_indices, foot_contact_vel_thresh)
    positions = get_rifke(global_positions, r_rot, translation_root_index=feature_translation_root_index)
    is_loop = detect_motion_loop(positions)
    local_vel = np.repeat(r_rot[1:, None], global_positions.shape[1], axis=1) * (global_positions[1:] - global_positions[:-1])
    prev_velocity = local_vel[-1] if local_vel.shape[0] > 0 else None
    terminal_local_vel = _compute_terminal_local_velocity(global_positions, r_rot, is_loop, prev_frame_velocity=prev_velocity)
    if has_locomotion:
        local_vel[:, feature_translation_root_index, [0, 2]] = 0.0
        terminal_local_vel[feature_translation_root_index, [0, 2]] = 0.0
    terminal_contact = get_terminal_contact_state(
        global_positions,
        foot_indices,
        foot_contact_vel_thresh,
        is_loop,
    )
    features, max_joints = get_motion_features(
        positions,
        cont_6d_params,
        foot_contact,
        local_vel,
        terminal_local_vel,
        terminal_contact,
        max_joints,
    )
    return features, max_joints, motion_anim, motion_export_anim, is_loop


""" processes animation, and returns a new animation that aligns with humanML3D in terms of orientation and scale"""
def get_hml_aligned_anim(fbx_path_or_anim, object_type, tpos_rots, offsets, squared_positions_error, *, scale_factor, foot_indices=None, orientation_quat, slice_inds=None, preloaded=None, helper_metadata=None, animation_input_is_tpose_aligned=True):
    if not isinstance(fbx_path_or_anim, Animation):
        if preloaded is not None:
            raw_anim, names = preloaded
        else:
            raw_anim, names, frame_time = FBX.load(fbx_path_or_anim)
        if slice_inds:
            raw_anim = raw_anim[slice_inds[0]:slice_inds[1]]
        #print('frame time', frame_time )
        frames_num, joints_num = raw_anim.positions.shape[:2]

        ## process animation: rotate to correct orientation, center, and scale
        processed_anim, root_translation_xz, _sf = process_anim(
            raw_anim,
            object_type,
            orientation_quat,
            scale_factor=scale_factor,
        )
        ## clamp vertical trajectory for flying/fish creatures (after scale, in HML units)
        processed_anim = clamp_vertical_trajectory(processed_anim, object_type)
    else:
        names = list()
        processed_anim = fbx_path_or_anim
        frames_num = len(processed_anim)
        root_translation_xz = None

    if processed_anim.positions.shape[1] != offsets.shape[0]:
        if helper_metadata is None:
            raise ValueError(
                f'Processed animation joint count {processed_anim.positions.shape[1]} does not match '
                f'offset count {offsets.shape[0]} without helper metadata'
            )
        if not names:
            raise ValueError('Cannot append helper joints to an Animation input without joint names')
        processed_anim, names = append_leaf_rotation_helpers_to_animation(
            processed_anim,
            names,
            helper_metadata,
        )
        frames_num = len(processed_anim)

    ## create new animation object in which the rotations are w.r.t the actual Tpos
    tpos_rots_correct_shape  = tpos_rots[None, 0].repeat(frames_num, axis = 0)
    if isinstance(fbx_path_or_anim, Animation) and animation_input_is_tpose_aligned:
        # Recovered / retargeted feature animations are already expressed in the
        # T-pose-relative local frame. Re-applying the T-pose transform would
        # double-transform them.
        rots = processed_anim.rotations.copy()
    else:
        # FBX input and raw T-pose Animation inputs still carry FBX-local rest
        # rotations and must be reparameterized against the character T-pose.
        rots = compute_rots_from_tpos(tpos_rots_correct_shape, processed_anim.rotations, processed_anim.parents)
    anim_positions = offsets.copy()[None, :].repeat(frames_num, axis = 0)
    anim_positions[:, 0] = processed_anim.positions[:, 0]
    processed_global_pos = positions_global(processed_anim)
    anim_positions = solve_local_positions_for_target_global(
        rots,
        processed_global_pos,
        offsets,
        processed_anim.parents,
        processed_anim.orients,
        initial_positions=anim_positions,
    )
    # create animation object which is defined over correct tpos
    new_anim = Animation(rots, anim_positions, processed_anim.orients, offsets, processed_anim.parents)

    new_global_pos = positions_global(new_anim)
    squared_error = np.mean((processed_global_pos - new_global_pos) ** 2)
    error_key = fbx_path_or_anim if isinstance(fbx_path_or_anim, str) else '__animation__'
    if slice_inds is not None and not isinstance(fbx_path_or_anim, Animation):
        error_key = f'{fbx_path_or_anim}[{slice_inds[0]}:{slice_inds[1]}]'
    squared_positions_error[error_key] = float(squared_error)

    return new_anim, processed_anim, names, root_translation_xz


""" get motion feature representation"""
def get_motion(fbx_path_or_anim, foot_contact_vel_thresh, object_type, max_joints, offsets, foot_indices, tpos_rots, squared_positions_error, *, scale_factor, orientation_quat, slice_inds=None, preloaded=None, helper_metadata=None, animation_input_is_tpose_aligned=True):
    try:
        new_anim, export_anim, names, root_translation_xz = get_hml_aligned_anim(
            fbx_path_or_anim,
            object_type,
            tpos_rots,
            offsets,
            squared_positions_error,
            scale_factor=scale_factor,
            foot_indices=foot_indices,
            orientation_quat=orientation_quat,
            slice_inds=slice_inds,
            preloaded=preloaded,
            helper_metadata=helper_metadata,
            animation_input_is_tpose_aligned=animation_input_is_tpose_aligned,
        )
        translation_root_index = resolve_detected_translation_root_index(
            find_translation_root(new_anim),
            find_translation_root(export_anim),
            object_type,
        )
        features, max_joints, motion_anim, motion_export_anim, is_loop = _extract_motion_features_from_aligned_anims(
            new_anim,
            export_anim,
            foot_contact_vel_thresh,
            object_type,
            max_joints,
            foot_indices,
            orientation_quat,
            translation_root_index=translation_root_index,
        )
        return features, motion_anim.parents, max_joints, motion_anim, motion_export_anim, is_loop, translation_root_index, root_translation_xz
    except Exception as err:
        print(err)
        return None, None, max_joints, None, None, False, None, None


################## Motion Recovery #####################

def recover_processed_animation_from_feature_animation(
    feature_anim,
    tpose_rest_rotations,
    position_match_threshold=1e-5,
    max_passes=2,
):
    from motion_lib.Quaternions import Quaternions

    frames_num = len(feature_anim)
    parents = feature_anim.parents.copy()
    offsets = feature_anim.offsets.copy()
    tpose_rest_rotations = np.asarray(tpose_rest_rotations, dtype=np.float64)
    tpose_quats = Quaternions(np.repeat(tpose_rest_rotations[None, :, :], frames_num, axis=0))

    feature_rots = feature_anim.rotations.copy()
    processed_rots = feature_rots.copy()
    processed_rots[:, 0] = feature_rots[:, 0] * tpose_quats[:, 0]

    cumulative_tpose = tpose_quats.copy()
    for joint_idx, parent_idx in enumerate(parents[1:], start=1):
        cumulative_tpose[:, joint_idx] = cumulative_tpose[:, parent_idx] * tpose_quats[:, joint_idx]
        processed_rots[:, joint_idx] = (
            -cumulative_tpose[:, parent_idx]
        ) * feature_rots[:, joint_idx] * cumulative_tpose[:, parent_idx] * tpose_quats[:, joint_idx]

    initial_positions = offsets.copy()[None, :].repeat(frames_num, axis=0)
    initial_positions[:, 0] = feature_anim.positions[:, 0]
    target_global_positions = positions_global(feature_anim)
    processed_positions = solve_local_positions_for_target_global(
        processed_rots,
        target_global_positions,
        offsets,
        parents,
        feature_anim.orients.copy(),
        initial_positions=initial_positions,
        position_match_threshold=position_match_threshold,
        max_passes=max_passes,
    )

    return Animation(
        processed_rots,
        processed_positions,
        feature_anim.orients.copy(),
        offsets,
        parents,
    )


def recover_root_quat_and_pos_np(
    data,
    translation_root_index=None,
    parents=None,
    offsets=None,
    anim_pos_threshold=0.01,
    motion_metadata=None,
    allow_infer=False,
    loop_close=False,
):
    motion = np.asarray(data)
    if motion.ndim == 2:
        root_features = motion
        translation_features = motion
    elif motion.ndim == 3:
        translation_root_index = resolve_feature_translation_root_index(
            motion,
            parents=parents,
            offsets=offsets,
            translation_root_index=translation_root_index,
            motion_metadata=motion_metadata,
            allow_infer=allow_infer,
            anim_pos_threshold=anim_pos_threshold,
            context='motion feature tensor',
        )
        root_features = motion[:, 0, :]
        translation_features = motion[:, translation_root_index, :]
    else:
        raise ValueError(f"Expected feature tensor with shape (F, C) or (F, J, C), got {motion.shape}.")

    # joint row 0 stores the root-facing rotation used by the representation.
    r_rot_quat = Quaternions.from_transforms(rotation_6d_to_matrix_np(root_features[:, 3:9]))

    # Normalize sign: ensure w >= 0 for each frame.
    # SciPy Rotation.from_matrix may return q or -q arbitrarily;
    # a consistent sign is required for the downstream ``-r_rot_quat * ...``
    # adjustment in ``recover_from_bvh_rot_np``.
    mask = r_rot_quat.qs[..., 0:1] < 0
    r_rot_quat.qs = np.where(mask, -r_rot_quat.qs, r_rot_quat.qs)

    r_pos = np.zeros(root_features.shape[:-1] + (3,))
    r_pos[..., 1:, [0, 2]] = translation_features[..., :-1, [9, 11]]
    '''Add Y-axis rotation to root position'''
    r_pos = -r_rot_quat * r_pos

    r_pos = np.cumsum(r_pos, axis = -2)
    if loop_close and r_pos.shape[-2] >= 2:
        # Stationary-loop de-drift: subtract a linear XZ ramp so frame L-1
        # lands exactly on frame 0 (which is the origin after cumsum).
        # Equivalent to subtracting mean XZ world-velocity from each step.
        L = r_pos.shape[-2]
        drift_x = r_pos[..., -1:, 0:1].copy()
        drift_z = r_pos[..., -1:, 2:3].copy()
        ramp = (np.arange(L, dtype=r_pos.dtype) / (L - 1)).reshape((1,) * (r_pos.ndim - 2) + (L, 1))
        r_pos[..., 0:1] -= drift_x * ramp
        r_pos[..., 2:3] -= drift_z * ramp
    r_pos[...,1] = translation_features[..., 1]
    return r_rot_quat, r_pos


""" recover quaternions and positions from features for numpy only"""
def recover_root_quat_and_pos(data):
    # root_feature_vector.shape = (frames, angular_vel || linear_xz_vel || root_height || zero pad)
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)
    r_rot_quat = Quaternions(r_rot_quat)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = -r_rot_quat * r_pos

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


""" recover xyz positions from ric (root relative positions) torch """
def recover_from_bvh_ric_np(
    data,
    translation_root_index=None,
    parents=None,
    offsets=None,
    anim_pos_threshold=0.01,
    motion_metadata=None,
    allow_infer=False,
    loop_close=False,
):
    motion = np.asarray(data)
    translation_root_index = resolve_feature_translation_root_index(
        motion,
        parents=parents,
        offsets=offsets,
        translation_root_index=translation_root_index,
        motion_metadata=motion_metadata,
        allow_infer=allow_infer,
        anim_pos_threshold=anim_pos_threshold,
        context='motion feature tensor',
    )
    r_rot_quat, r_pos = recover_root_quat_and_pos_np(
        data,
        translation_root_index=translation_root_index,
        parents=parents,
        offsets=offsets,
        anim_pos_threshold=anim_pos_threshold,
        motion_metadata=motion_metadata,
        allow_infer=allow_infer,
        loop_close=loop_close,
    )
    positions = np.asarray(data[..., :3], dtype=np.float32).copy()
    positions = np.repeat(-r_rot_quat[..., None, :], positions.shape[-2], axis=-2) * positions
    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    return positions


""" recover xyz positions from rot (root relative positions) torch """
def _normalize_quaternion_signs(qs, parents):
    """Normalize quaternion signs for temporal consistency.

    ``Quaternions.from_transforms`` (via SciPy ``Rotation.from_matrix``) has no
    guarantee on the sign of the recovered quaternions.  Both ``q`` and ``-q``
    represent the same rotation, but downstream operations like
    ``compute_rots_from_tpos`` are sensitive to sign flips.

    Strategy:
      1. For each joint, ensure the first frame has ``w >= 0``.
      2. For each subsequent frame, flip sign if the dot product with the
         previous frame is negative (temporal consistency).

    Args:
        qs: (F, J, 4) quaternion array (WXYZ).
        parents: (J,) parent indices, -1 for root.

    Returns:
        (F, J, 4) sign-normalized quaternion array.
    """
    qs = np.asarray(qs, dtype=np.float64)
    F, J = qs.shape[:2]

    # Step 1: ensure first frame has w >= 0 for each joint
    for j in range(J):
        if qs[0, j, 0] < 0:
            qs[0, j] = -qs[0, j]

    # Step 2: temporal consistency — flip if dot product with previous frame < 0
    for f in range(1, F):
        dots = np.sum(qs[f] * qs[f - 1], axis=1)  # (J,)
        flip = dots < 0
        qs[f, flip] = -qs[f, flip]

    return qs


def recover_from_bvh_rot_np(
    data,
    parents,
    offsets,
    translation_root_index=None,
    anim_pos_threshold=0.01,
    motion_metadata=None,
    allow_infer=False,
    loop_close=False,
):
    translation_root_index = resolve_feature_translation_root_index(
        data,
        parents=parents,
        offsets=offsets,
        translation_root_index=translation_root_index,
        motion_metadata=motion_metadata,
        allow_infer=allow_infer,
        anim_pos_threshold=anim_pos_threshold,
        context='motion feature tensor',
    )
    r_rot_quat, r_pos = recover_root_quat_and_pos_np(
        data,
        translation_root_index=translation_root_index,
        parents=parents,
        offsets=offsets,
        anim_pos_threshold=anim_pos_threshold,
        motion_metadata=motion_metadata,
        allow_infer=allow_infer,
        loop_close=loop_close,
    )
    r_rot_cont6d = get_6d_rep(r_rot_quat)
    start_indx = 3
    end_indx = 9
    cont6d_params = data[..., 1:, start_indx:end_indx]
    cont6d_params = np.concatenate([r_rot_cont6d[:, None, :], cont6d_params], axis=-2)
    cont6d_params_hml_order = rotation_6d_to_matrix_np(cont6d_params)
    cont6d_params = np.eye(3)[None, None].repeat(cont6d_params.shape[0], axis=0).repeat(cont6d_params.shape[1], axis=1)
    for j, p in enumerate(parents[1:], 1):
        cont6d_params[:, p] = cont6d_params_hml_order[:, j]
    rotations = Quaternions.from_transforms(cont6d_params)

    # Normalize quaternion signs for roundtrip stability.
    # Without this, SciPy's Rotation.from_matrix may return q or -q
    # arbitrarily, causing 6D rotation features to diverge after
    # a features → Animation → features roundtrip.
    rotations.qs = _normalize_quaternion_signs(rotations.qs, parents)
    positions = offsets[None].repeat(data.shape[0], axis=0)
    root_global = (-r_rot_quat) * np.asarray(data[:, 0, :3], dtype=np.float32)
    root_global[:, 0] += r_pos[:, 0]
    root_global[:, 2] += r_pos[:, 2]
    positions[:, 0] = root_global
    anim = Animation(rotations=rotations, positions=positions, parents=parents, offsets=offsets, orients=Quaternions.id(0))

    if translation_root_index != 0 and parents[translation_root_index] >= 0:
        global_rots = rotations_global(anim)
        global_pos = positions_global(anim)
        parent_index = parents[translation_root_index]
        positions[:, translation_root_index] = (-global_rots[:, parent_index]) * (r_pos - global_pos[:, parent_index])
        anim = Animation(rotations=rotations, positions=positions, parents=parents, offsets=offsets, orients=Quaternions.id(0))

    return positions_global(anim), anim


""" Reconstruct a BVH-ready Animation from the per-joint feature tensor.

Combines the rotation path (recover_from_bvh_rot_np) with the RIC position
path (recover_from_bvh_ric_np) to correctly handle skeletons that carry
animated positions on non-root joints (e.g. Horse Bip01, Bear NPC_Pelvis).

Unlike using animation_from_positions (pure IK), this preserves the
per-joint position channels that the training features explicitly encode,
reducing max global-position error from ~0.3 to ~0.02 units.

Returns:
    anim            : Animation with corrected local positions
    has_animated_pos: bool — True when any non-root joint needed position fix
                      (caller should pass this as BVH.save(..., positions=...))
"""
def recover_animation_from_motion_np(
    data,
    parents,
    offsets,
    translation_root_index=None,
    anim_pos_threshold=0.01,
    motion_metadata=None,
    allow_infer=False,
    loop_close=False,
):
    translation_root_index = resolve_feature_translation_root_index(
        data,
        parents=parents,
        offsets=offsets,
        translation_root_index=translation_root_index,
        motion_metadata=motion_metadata,
        allow_infer=allow_infer,
        anim_pos_threshold=anim_pos_threshold,
        context='motion feature tensor',
    )
    target_global        = recover_from_bvh_ric_np(
        data,
        translation_root_index=translation_root_index,
        parents=parents,
        offsets=offsets,
        anim_pos_threshold=anim_pos_threshold,
        motion_metadata=motion_metadata,
        allow_infer=allow_infer,
        loop_close=loop_close,
    )              # (F, J, 3)
    _, anim_rot          = recover_from_bvh_rot_np(
        data,
        parents,
        offsets,
        translation_root_index=translation_root_index,
        anim_pos_threshold=anim_pos_threshold,
        motion_metadata=motion_metadata,
        allow_infer=allow_infer,
        loop_close=loop_close,
    )
    glob_rot             = positions_global(anim_rot)                  # (F, J, 3)

    # joints whose FK-predicted global position drifts from the RIC truth
    per_joint_err = np.abs(target_global - glob_rot).max(axis=(0, 2)) # (J,)
    animated_joints = sorted(
        j for j in range(len(parents)) if per_joint_err[j] > anim_pos_threshold
    )

    if not animated_joints:
        return anim_rot, needs_bvh_position_channels(anim_rot)

    new_pos = solve_local_positions_for_target_global(
        anim_rot.rotations,
        target_global,
        anim_rot.offsets,
        anim_rot.parents,
        anim_rot.orients,
        initial_positions=anim_rot.positions.copy(),
        position_match_threshold=1e-5,
        max_passes=2,
    )

    anim_fixed = Animation(anim_rot.rotations, new_pos, anim_rot.orients,
                           anim_rot.offsets, anim_rot.parents)
    return anim_fixed, needs_bvh_position_channels(anim_fixed)


def recover_bvh_export_animation_from_motion_np(
    data,
    parents,
    offsets,
    joint_names,
    translation_root_index=None,
    anim_pos_threshold=0.01,
    motion_metadata=None,
    allow_infer=False,
    tpose_rest_rotations=None,
    loop_close=False,
):
    """Recover a motion tensor and remap it into BVH-safe DFS order.

    ``recover_animation_from_motion_np`` intentionally preserves the input joint
    indexing because non-export callers still address joints by the original cond
    metadata indices. BVH export has the additional requirement that joint arrays
    must match hierarchy DFS order, so this helper layers the DFS remap on top of
    recovery without changing the base function's semantics.

    When *tpose_rest_rotations* is provided (``(J, 4)`` quaternion array in
    ``[w, x, y, z]`` order), the recovered T-pose-relative rotations are baked
    back into total local rotations (rest ⊗ pose) so the BVH displays correctly
    for skeletons with non-identity rest rotations (e.g. GLB-derived skeletons).
    """
    anim, has_animated_pos = recover_animation_from_motion_np(
        data,
        parents,
        offsets,
        translation_root_index=translation_root_index,
        anim_pos_threshold=anim_pos_threshold,
        motion_metadata=motion_metadata,
        allow_infer=allow_infer,
        loop_close=loop_close,
    )
    if anim is None:
        return None, list(joint_names), has_animated_pos

    if tpose_rest_rotations is not None:
        anim = recover_processed_animation_from_feature_animation(
            anim, tpose_rest_rotations,
        )

    anim, joint_names = reorder_animation_to_dfs(anim, joint_names)
    return anim, joint_names, has_animated_pos

