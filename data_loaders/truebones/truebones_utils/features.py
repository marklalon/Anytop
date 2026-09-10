"""Feature extraction & motion recovery.

Middle layer of the motion-processing pipeline. Extracts feature tensors from
animations, recovers animations from features, analyses rest poses, and infers
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
    DROP_PROP_SOCKET_JOINTS,
    DROP_END_SITE_JOINTS,
    MAX_JOINTS,
)
from utils.rotation_conversions import rotation_6d_to_matrix_np
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
from .ignore_warnings import skip_orientation_detection

from .animation_utils import (
    ROOT_XZ_DRIFT_THRESHOLD,
    detect_motion_loop,
    find_translation_root,
    clamp_vertical_trajectory,
    collapse_translation_root_chain,
    promote_translation_root_to_hierarchy_root,
    move_xz_to_origin,
    root_xz_trajectory,
    root_xz_heading,
    flatten_root_xz_drift,
    soft_clamp_root_xz,
    scale_root_xz_extent,
    set_translation_root_xz,
    resolve_detected_translation_root_index,
    needs_bvh_position_channels,
    reorder_animation_to_dfs,
    crop_animation_to_max_joints,
    drop_prop_socket_joints,
    drop_end_site_joints,
    get_average_axial_bone_length,
    get_scale_reference_extent,
    rest_pose_animation,
    compute_scale_factor,
    scale_anim,
    compute_rots_from_tpos,
    solve_local_positions_for_target_global,
)


################## Feature Building #####################


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


def get_motion_features(ric_positions, rotations, velocity, terminal_velocity, max_joints):
    # F = Frames# , J = joints# 
    # parents (J,1)
    # positions (F, J, 3)
    # rotations (F, J, 6)
    # velocity (F - 1, J, 3) + one terminal row
    # offsets (J, 3)
    
    # feature len = 12 (pos, rot, vel)

    joints = ric_positions.shape[1]
    if joints > max_joints:
        max_joints = joints
    pos = ric_positions  ## (Frames, joints, 3)
    rot = rotations ## (Frames, joints, 6)
    vel = np.concatenate([velocity, terminal_velocity[None, ...]], axis=0) ## (Frames, joints, 3)
    features= np.concatenate([pos, rot, vel], axis=-1) 
    return features, max_joints


""" returns cont6d params, including joints rotations, root rotation and rotational velocity,
linear velocity and positions. Each joint stores its own local rotation directly
(unlike BVH where the parent holds the rotation of the child joint)."""
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
    '''Quaternion to continuous 6D — each joint stores its own local rotation'''
    cont_6d_params = get_6d_rep(quat_params)
    # (seq_len, 4)
    '''Root Linear Velocity'''
    # (seq_len - 1, 3)
    velocity = (positions[1:, translation_root_index] - positions[:-1, translation_root_index]).copy()
    velocity = r_rot[1:] * velocity
    '''Root Angular Velocity'''
    # (seq_len - 1, 4)
    r_velocity = r_rot[1:] * -r_rot[:-1]
    # (seq_len, joints_num, 4)
    return cont_6d_params, r_velocity, velocity, r_rot, positions


"""" process anim object """
def process_anim(
    anim,
    object_type,
    orientation_quat,
    root_xz_center=None,
    *,
    scale_factor,
    translation_root_index=None,
):
    rotated = rotate_to_hml_orientation(anim, orientation_quat)
    centered, root_xz_center_ = move_xz_to_origin(
        rotated,
        root_xz_center,
        translation_root_index=translation_root_index,
    )
    scaled = scale_anim(centered, scale_factor)
    # Keep rest-pose conditioning and motion clips on the same normalized
    # geometry.  Both paths pass through process_anim, while only raw motion
    # files continue through the loading branch in get_hml_aligned_anim.
    processed = clamp_vertical_trajectory(
        scaled,
        object_type,
        translation_root_index=translation_root_index,
    )
    return processed, root_xz_center_, scale_factor


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


################## Rest Pose & Motion Extraction #####################

def _remap_joint_indices(joint_indices, kept_joint_indices):
    """Move caller-supplied joint indices onto a filtered skeleton.

    Indices that fell inside a removed subtree are dropped. ``None`` for either
    argument means there is nothing to remap (no explicit indices were given, or
    the filter kept every joint).
    """
    if kept_joint_indices is None or joint_indices is None:
        return joint_indices
    index_remap = {old: new for new, old in enumerate(kept_joint_indices)}
    return [index_remap[j] for j in joint_indices if j in index_remap]


def _rest_pose_animation_from_loaded_anim(anim):
    """Return a one-frame bind/rest-pose Animation from a loaded FBX animation."""
    return rest_pose_animation(anim)


""" get object_type common characteristics, extracted from an FBX/GLB bind/rest pose"""
def get_common_features_from_rest_pose(
    rest_pose_path,
    object_type,
    face_joints=None,
    *,
    max_joints=None,
    drop_prop_sockets=None,
    drop_end_sites=None,
    promote_root_depth=0,
    raw_load_cache=None,
):
    if drop_prop_sockets is None:
        drop_prop_sockets = DROP_PROP_SOCKET_JOINTS
    if drop_end_sites is None:
        drop_end_sites = DROP_END_SITE_JOINTS
    # Phase 1 re-derives the rest pose once per fold depth from the same file, so
    # it shares the caller's realpath-keyed import cache with the motion sources.
    # Every step below rebuilds rather than writes -- rest_pose_animation copies,
    # and so do the drops, the fold and the crop -- so passes can share the load.
    _rest_pose_key = os.path.realpath(str(rest_pose_path))
    if raw_load_cache is not None and _rest_pose_key in raw_load_cache:
        loaded_anim, _cached_names, _rest_pose_frame_time = raw_load_cache[_rest_pose_key]
        rest_pose_names = list(_cached_names)
    else:
        loaded_anim, rest_pose_names, _rest_pose_frame_time = FBX.load(rest_pose_path)
        if raw_load_cache is not None:
            raw_load_cache[_rest_pose_key] = (
                loaded_anim, list(rest_pose_names), _rest_pose_frame_time,
            )
    max_joints = int(max_joints) if max_joints is not None else max(len(rest_pose_names), 1)
    reference_anim = _rest_pose_animation_from_loaded_anim(loaded_anim)
    rest_pose_context = f"{object_type} rest pose '{os.path.basename(str(rest_pose_path))}'"
    # Parked weapon/prop sockets go first, so nothing below -- face joints,
    # contact joints, offsets, scale -- is inferred on rig furniture, and the
    # MAX_JOINTS budget in the crop is spent on real bones. The names are carried
    # out on TPoseFeatures so every motion clip of this character drops the same
    # joints instead of re-detecting them on its own file.
    prop_socket_names = ()
    if drop_prop_sockets:
        pre_drop_names = rest_pose_names
        reference_anim, rest_pose_names, _kept_after_drop = drop_prop_socket_joints(
            reference_anim,
            rest_pose_names,
            context=rest_pose_context,
        )
        if _kept_after_drop is not None:
            _kept = set(_kept_after_drop)
            prop_socket_names = tuple(
                name for index, name in enumerate(pre_drop_names) if index not in _kept
            )
            face_joints = _remap_joint_indices(face_joints, _kept_after_drop)
    # BVH End-Site terminators go next, before the crop and independently of it.
    # A tpose that round-tripped through BVH carries one extra leaf per chain end;
    # left in, each steals the EndEffector/ChainEnd marker from the real joint it
    # hangs off and shifts every 'Segment N Of M' count up that chain, so the same
    # character conditions differently depending on which file it was built from.
    # Like the prop sockets, the names travel out on TPoseFeatures so every clip
    # of this character drops exactly the same joints.
    end_site_names = ()
    if drop_end_sites:
        pre_end_site_names = rest_pose_names
        reference_anim, rest_pose_names, _kept_after_end_sites = drop_end_site_joints(
            reference_anim,
            rest_pose_names,
            context=rest_pose_context,
        )
        if _kept_after_end_sites is not None:
            _kept_es = set(_kept_after_end_sites)
            end_site_names = tuple(
                name for index, name in enumerate(pre_end_site_names) if index not in _kept_es
            )
            face_joints = _remap_joint_indices(face_joints, _kept_after_end_sites)
    # The wrapper joints above the real root go next, after the two drops and
    # BEFORE the crop. Two constraints pin it to exactly this slot:
    #
    # * After the drops, because the fold refuses a root with more than one
    #   child, and a raw rig routinely has one -- Tukan's ``Hips`` carries the
    #   body chain AND a ``MESH`` branch, Crow's root carries ``ecr1``. Folding
    #   first would hard-fail those species on rig furniture that is about to be
    #   dropped anyway.
    # * Before the crop, so the MAX_JOINTS budget is never spent on control
    #   nodes that are about to be folded away. Cropping first costs one real
    #   bone per wrapper joint on any skeleton at the cap (Horse loses 2, Camel
    #   2, Bear 1 when the cap bites).
    #
    # The depth itself is measured by phase 1 on the fully normalized skeleton,
    # so it is applied here one crop earlier than it was measured. That is safe
    # by construction, not by luck: select_cropped_joint_indices only ever
    # removes current leaves and never the root, while every joint on the root
    # chain has exactly one child, so no crop can shorten the chain the depth
    # counts along. Everything below here (face joints, contact joints, offsets,
    # scale) is then inferred on the skeleton the model will see, whose joint 0
    # IS the translation root.
    if promote_root_depth:
        reference_anim, rest_pose_names, _kept_after_promote = (
            promote_translation_root_to_hierarchy_root(
                reference_anim,
                rest_pose_names,
                promote_root_depth,
                context=rest_pose_context,
            )
        )
        face_joints = _remap_joint_indices(face_joints, _kept_after_promote)
    # Crop oversized skeletons down to max_joints BEFORE any face/contact/offset
    # inference, so every downstream rest-pose artifact is built on the cropped
    # skeleton. Leaves are removed deepest-first, same-depth ties prefer shorter
    # bones, and longer-than-average bones are preserved whenever possible. Each
    # motion clip crops independently from the same skeleton definition
    # (same topology and offsets), yielding the identical joint set (validated
    # by the offset-count guard in get_hml_aligned_anim).
    reference_anim, rest_pose_names, _kept_joint_indices = crop_animation_to_max_joints(
        reference_anim,
        rest_pose_names,
        max_joints=max_joints,
        context=rest_pose_context,
    )
    face_joints = _remap_joint_indices(face_joints, _kept_joint_indices)
    reference_positions = positions_global(reference_anim)
    face_joints = resolve_face_joints(
        object_type,
        rest_pose_names,
        reference_anim.parents,
        face_joints=face_joints,
        rest_positions=reference_positions,
    )
    forward_joint_index, forward_base_joint_index = resolve_forward_reference_joints(
        rest_pose_names,
        reference_anim.parents,
        object_type=object_type,
        rest_positions=reference_positions,
    )
    if skip_orientation_detection():
        # The dataset declares its rest poses already facing the canonical +Z
        # (``!skip-orientation-detection`` in ignore_warnings.txt), so no facing
        # is estimated and the correction stays identity. The face/forward joints
        # resolved above are still recorded on the object cond -- validation
        # compares each clip's recovered facing against the rest pose with them.
        rest_pose_orientation_quat = Quaternions.id(len(reference_positions))[0]
    else:
        rest_pose_orientation_quat = calculate_root_quat(reference_positions, object_type, face_joint_indx=face_joints, forward_joint_index=forward_joint_index, forward_base_joint_index=forward_base_joint_index)[0]

    # Pre-compute the per-character scale factor once from the raw rest-pose
    # offsets and reuse it for every motion clip of the same character.
    _rest_pose_side_labels = []
    for name in rest_pose_names:
        detected = detect_joint_side(name)
        _rest_pose_side_labels.append(detected if detected in ('left', 'right') else 'center')
    axial_avg_len = get_average_axial_bone_length(
        reference_anim.offsets, reference_anim.parents, _rest_pose_side_labels, rest_pose_names
    )
    # Extent, not joint span: a rig whose root is seated far above the origin
    # (hovering/drifting creatures) must be normalized against that elevation
    # too, or bone-driven scaling leaves its root height an outlier. Measured on
    # the FK'd rest pose above: reference_anim.offsets are parent-bone-local and
    # sum to a straightened skeleton.
    reference_body_max_span = get_scale_reference_extent(
        reference_positions[0], reference_anim.parents, rest_pose_names
    )
    scale_factor = compute_scale_factor(axial_avg_len, body_max_span=reference_body_max_span)

    scaled, _root_xz_center, scale_factor = process_anim(
        reference_anim,
        object_type,
        rest_pose_orientation_quat,
        scale_factor=scale_factor,
    )
    # Under own-rotation encoding, no leaf rotation helpers are needed.
    scaled_positions = positions_global(scaled)
    scaled_rest_positions = scaled_positions[0]
    offsets = offsets_from_positions(scaled_rest_positions, scaled.parents)
    suspected_foot_indices, contact_joint_source = infer_contact_joints(
        rest_pose_names,
        scaled.parents,
        scaled_rest_positions,
    )
    return TPoseFeatures(
        scale_factor=scale_factor,
        offsets=offsets,
        foot_indices=suspected_foot_indices,
        tpos_rots=scaled.rotations,
        names=rest_pose_names,
        tpos_anim=scaled,
        face_joints=face_joints,
        orientation_quat=rest_pose_orientation_quat,
        forward_joint_index=forward_joint_index,
        forward_base_joint_index=forward_base_joint_index,
        contact_joint_source=contact_joint_source,
        axial_avg_len=axial_avg_len,
        prop_socket_names=prop_socket_names,
        end_site_names=end_site_names,
        promote_root_depth=int(promote_root_depth),
    )


def get_common_features_from_T_pose(*args, **kwargs):
    """Backward-compatible name; the returned base now comes from bind/rest pose."""
    return get_common_features_from_rest_pose(*args, **kwargs)


def tpose_features_from_cond(cond_entry, object_type=None):
    """Reconstruct TPoseFeatures from a prebuilt cond.npy entry — no mesh access.

    Every field the retarget / reference-preprocessing paths consume is already
    baked into cond at dataset-build time, so the original bind/rest pose
    FBX/GLB does not need to be re-loaded at inference time:

      offsets, parents, joint names, bind-pose local rotations
      (``tpose_rest_rotations``), orientation_quat, forward joint indices, face
      joints, contact (foot) joints, per-character scale factor and axial bone
      length.

    Note ``tpose_rest_rotations`` (the scaled/oriented skeleton's per-joint bind
    LOCAL rotations) is a distinct quantity from ``rest_pose[:, 3:9]`` (the
    feature-space rest rotations) and is what the retargeter needs; it is baked
    separately into cond because it is not derivable from rest_pose or offsets.

    The result is duck-typed to :class:`TPoseFeatures` (identical attributes).
    ``tpos_anim`` is a minimal one-frame rest-pose Animation carrying topology
    (parents/offsets) and rest rotations; the only attribute downstream
    consumers read from it is ``parents``.
    """
    from .physics_joint_annotation import rest_positions_from_offsets

    parents = np.asarray(cond_entry['parents'], dtype=np.int32)
    offsets = np.asarray(cond_entry['offsets'], dtype=np.float32)
    names = list(cond_entry['joints_names'])
    joint_count = len(parents)

    rest_rots = cond_entry.get('tpose_rest_rotations')
    if rest_rots is None:
        raise KeyError(
            f"cond entry for '{object_type or cond_entry.get('object_type')}' is "
            "missing 'tpose_rest_rotations'; regenerate cond to bake the bind-pose "
            "local rotations (they are not derivable from rest_pose/offsets)."
        )
    rest_quats = np.asarray(rest_rots, dtype=np.float64).reshape(joint_count, 4)
    tpos_rots = Quaternions(rest_quats[None, :, :])  # (1, J, 4)

    orientation_qs = np.asarray(cond_entry['orientation_quat'], dtype=np.float64).reshape(4)
    orientation_quat = Quaternions(orientation_qs[None, :]).normalized()  # (1, 4)

    rest_positions = rest_positions_from_offsets(offsets, parents)
    tpos_anim = Animation(
        tpos_rots.copy(),
        rest_positions[None, :, :].astype(np.float64),
        Quaternions.id(joint_count),
        offsets.astype(np.float64),
        parents,
    )

    return TPoseFeatures(
        scale_factor=float(cond_entry['scale_factor']),
        offsets=offsets,
        foot_indices=list(cond_entry.get('contact_joints') or []),
        tpos_rots=tpos_rots,
        names=names,
        tpos_anim=tpos_anim,
        face_joints=list(cond_entry.get('face_joints') or []),
        orientation_quat=orientation_quat,
        forward_joint_index=cond_entry.get('forward_joint_index'),
        forward_base_joint_index=cond_entry.get('forward_base_joint_index'),
        contact_joint_source=cond_entry.get('contact_joint_source', 'cond'),
        axial_avg_len=float(cond_entry.get('axial_avg_len', 0.0)),
    )


@dataclass
class TPoseFeatures:
    """Packaged return from get_common_features_from_rest_pose.

    Field names keep the legacy ``tpos_*`` spelling for internal pipeline use;
    all values are computed from the file bind/rest pose.
    """
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
    # Rest-pose names of the prop-socket joints removed from this skeleton, so
    # every motion clip of the character drops exactly the same joints. Empty for
    # a cond-reconstructed rest pose: cond was already built on the filtered
    # skeleton, so there is nothing left to drop.
    prop_socket_names: tuple = ()
    # Rest-pose names of the BVH end-site terminators removed from this skeleton,
    # carried for the same reason as prop_socket_names: every motion clip of the
    # character must drop exactly the same joints. Empty for a cond-reconstructed
    # rest pose, which was already built on the filtered skeleton.
    end_site_names: tuple = ()
    # How many wrapper joints above the real root this skeleton already had
    # folded away, so every motion clip of the character drops the same ones.
    promote_root_depth: int = 0


def extract_motion_features_from_aligned_anims(
    new_anim,
    export_anim,
    object_type,
    max_joints,
    orientation_quat,
    translation_root_index,
    *,
    flatten_root_travel=False,
    clamp_root_xz_extent=False,
):
    feature_translation_root_index = int(translation_root_index)

    # Seat the inert control nodes above the effective root onto it, before
    # anything reads a position off either anim. Nothing below the root moves --
    # this only stops the wrapper's RIC channel from carrying an arbitrary
    # per-clip constant and the negated root trajectory.
    new_anim = collapse_translation_root_chain(new_anim, feature_translation_root_index)
    export_anim = collapse_translation_root_chain(export_anim, feature_translation_root_index)

    # The root trajectory the edits below measure and rewrite is the one the
    # source was AUTHORED with, so it is read once here, ahead of any of them.
    source_global_positions = positions_global(new_anim)

    # The root XZ trajectory is decided here in two steps and applied once, so
    # the skeleton is put through FK a single time: remove a gait's travel, then
    # bound whatever excursion is left. Both steps are opt-in and either can be
    # a no-op; the anims are only rebuilt if the target ended up different.
    source_root_xz = np.asarray(
        source_global_positions[:, feature_translation_root_index][:, [0, 2]],
        dtype=np.float64,
    )
    target_root_xz = source_root_xz

    # Step one takes two conditions, and both are needed.
    #
    # ``flatten_root_travel`` is the caller's verdict that this clip is a gait --
    # in practice its action group. No measurement can stand in for it: a gait
    # take and a lunging attack are both one closed cycle that ends displaced,
    # and on the shipped datasets a pure drift test would have flattened 393
    # death and knockdown clips (Monkey_Die drifts 0.91, TNR_Archer_DeathA 0.82)
    # whose displacement IS the action.
    #
    # The measurement then says whether this particular gait take actually
    # travels, so a clip already authored in place is left untouched rather than
    # passed through an operator that would only add float noise. It is the only
    # thing the flatten is gated on: what the detrend leaves behind is BOUNDED
    # below, not exempted here.
    root_xz_flattened = False
    if flatten_root_travel:
        flattened_root_xz, root_xz_drift = flatten_root_xz_drift(
            source_root_xz,
            root_xz_heading(new_anim, feature_translation_root_index),
        )
        root_xz_flattened = bool(root_xz_drift > ROOT_XZ_DRIFT_THRESHOLD)
        if root_xz_flattened:
            # Nothing is zeroed: the within-cycle surge and sway survive, only
            # the travel underneath them is removed.
            target_root_xz = flattened_root_xz

    if clamp_root_xz_extent:
        # Second and last, on whatever the first step left. Both bounds live here
        # rather than inside the detrend, because the validator calls that
        # operator to MEASURE a stored clip and must not reshape what it reads.
        #
        # This is an OPT-IN because it is not idempotent: re-extracting features
        # from an already-bounded clip (recovery, retarget, the resample branch's
        # second pass) would compress the excursion a second time. Only a fresh
        # pass over source animation asks for it.
        if flatten_root_travel:
            # Locomotion's own bound, on EVERY locomotion clip and not only the
            # ones that travelled, so the group's limit holds by construction and
            # the validator can read it straight off the tensor. One factor for the
            # whole clip: what a detrend leaves behind IS the gait cycle, and a
            # per-frame map would reshape the surge instead of only sizing it.
            target_root_xz = scale_root_xz_extent(target_root_xz)
        # The dataset-wide ceiling, on every clip. A lunge, a death slide or an
        # attack reaches this one; a locomotion clip is already far inside it.
        target_root_xz = soft_clamp_root_xz(target_root_xz)

    motion_anim = new_anim
    motion_export_anim = export_anim
    if not np.array_equal(target_root_xz, source_root_xz):
        # Both anims take the SAME correction so the exported BVH cannot drift
        # away from the features.
        correction = source_root_xz - target_root_xz
        motion_anim = set_translation_root_xz(
            new_anim, feature_translation_root_index, target_root_xz,
        )
        export_root_xz = root_xz_trajectory(export_anim, feature_translation_root_index)
        motion_export_anim = set_translation_root_xz(
            export_anim, feature_translation_root_index, export_root_xz - correction,
        )

    cont_6d_params, r_velocity, velocity, r_rot, global_positions = get_bvh_cont6d_params(
        motion_anim,
        object_type,
        orientation_quat,
        translation_root_index=feature_translation_root_index,
    )
    positions = get_rifke(global_positions, r_rot, translation_root_index=feature_translation_root_index)
    local_vel = np.repeat(r_rot[1:, None], global_positions.shape[1], axis=1) * (global_positions[1:] - global_positions[:-1])
    is_loop = detect_motion_loop(
        positions,
        root_xz_velocity=local_vel,
        translation_root_index=feature_translation_root_index,
    )
    prev_velocity = local_vel[-1] if local_vel.shape[0] > 0 else None
    terminal_local_vel = _compute_terminal_local_velocity(global_positions, r_rot, is_loop, prev_frame_velocity=prev_velocity)
    features, max_joints = get_motion_features(
        positions,
        cont_6d_params,
        local_vel,
        terminal_local_vel,
        max_joints,
    )
    return features, max_joints, motion_anim, motion_export_anim, is_loop, root_xz_flattened


""" processes animation, and returns a new animation that aligns with humanML3D in terms of orientation and scale"""
def get_hml_aligned_anim(fbx_path_or_anim, object_type, tpos_rots, offsets, squared_positions_error, *, scale_factor, orientation_quat, slice_inds=None, preloaded=None, animation_input_is_tpose_aligned=True, translation_root_index=None):
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
            translation_root_index=translation_root_index,
        )
    else:
        names = list()
        processed_anim = fbx_path_or_anim
        frames_num = len(processed_anim)
        root_translation_xz = None

    if processed_anim.positions.shape[1] != offsets.shape[0]:
        raise ValueError(
            f'Processed animation joint count {processed_anim.positions.shape[1]} does not match '
            f'offset count {offsets.shape[0]}'
        )

    ## create new animation object in which the rotations are w.r.t the rest pose
    tpos_rots_correct_shape  = tpos_rots[None, 0].repeat(frames_num, axis = 0)
    if isinstance(fbx_path_or_anim, Animation) and animation_input_is_tpose_aligned:
        # Recovered / retargeted feature animations are already expressed in the
        # rest-pose-relative local frame. Re-applying the rest-pose transform would
        # double-transform them.
        rots = processed_anim.rotations.copy()
    else:
        # FBX input and raw rest-pose Animation inputs still carry FBX-local rest
        # rotations and must be reparameterized against the character rest pose.
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
    # create animation object which is defined over the correct rest-pose base
    new_anim = Animation(rots, anim_positions, processed_anim.orients, offsets, processed_anim.parents)

    new_global_pos = positions_global(new_anim)
    squared_error = np.mean((processed_global_pos - new_global_pos) ** 2)
    error_key = fbx_path_or_anim if isinstance(fbx_path_or_anim, str) else '__animation__'
    if slice_inds is not None and not isinstance(fbx_path_or_anim, Animation):
        error_key = f'{fbx_path_or_anim}[{slice_inds[0]}:{slice_inds[1]}]'
    squared_positions_error[error_key] = float(squared_error)

    return new_anim, processed_anim, names, root_translation_xz


""" get motion feature representation"""
def get_motion(fbx_path_or_anim, object_type, max_joints, offsets, tpos_rots, squared_positions_error, *, scale_factor, orientation_quat, slice_inds=None, preloaded=None, animation_input_is_tpose_aligned=True, translation_root_index=None, flatten_root_travel=False, clamp_root_xz_extent=False):
    try:
        new_anim, export_anim, names, root_translation_xz = get_hml_aligned_anim(
            fbx_path_or_anim,
            object_type,
            tpos_rots,
            offsets,
            squared_positions_error,
            scale_factor=scale_factor,
            orientation_quat=orientation_quat,
            slice_inds=slice_inds,
            preloaded=preloaded,
            animation_input_is_tpose_aligned=animation_input_is_tpose_aligned,
            translation_root_index=translation_root_index,
        )
        if translation_root_index is None:
            translation_root_index = resolve_detected_translation_root_index(
                find_translation_root(new_anim),
                find_translation_root(export_anim),
                object_type,
            )
        else:
            translation_root_index = _coerce_translation_root_index(
                translation_root_index,
                joint_count=new_anim.positions.shape[1],
                context=f"{object_type} motion",
            )
        features, max_joints, motion_anim, motion_export_anim, is_loop, root_xz_flattened = extract_motion_features_from_aligned_anims(
            new_anim,
            export_anim,
            object_type,
            max_joints,
            orientation_quat,
            translation_root_index=translation_root_index,
            flatten_root_travel=flatten_root_travel,
            clamp_root_xz_extent=clamp_root_xz_extent,
        )
        return features, motion_anim.parents, max_joints, motion_anim, motion_export_anim, is_loop, translation_root_index, root_translation_xz, root_xz_flattened
    except Exception as err:
        print(err)
        return None, None, max_joints, None, None, False, None, None, False


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

    # Under the own-rotation encoding, slot 0 stores the root's own local
    # rotation (used for FK).  The RIC representation was encoded with
    # identity-facing (r_rot = identity after canonicalization), so
    # de-rotation / velocity recovery must also use identity here.
    r_rot_quat = Quaternions.id(motion.shape[0])

    r_pos = np.zeros(root_features.shape[:-1] + (3,))
    r_pos[..., 1:, [0, 2]] = translation_features[..., :-1, [9, 11]]
    '''Add Y-axis rotation to root position'''
    r_pos = -r_rot_quat * r_pos

    r_pos = np.cumsum(r_pos, axis = -2)
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
    )
    # Under the own-rotation encoding, each feature slot j stores joint j's
    # own local rotation as 6D continuous parameters (channels 3:9).
    cont6d_params = rotation_6d_to_matrix_np(np.asarray(data[:, :, 3:9], dtype=np.float64))
    rotations = Quaternions.from_transforms(cont6d_params)
    # NOTE: slot 0 now stores the root joint's own local rotation, so it is
    # recovered directly alongside every other joint — no scatter required.

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
    rigid_bone=False,
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
    _, anim_rot          = recover_from_bvh_rot_np(
        data,
        parents,
        offsets,
        translation_root_index=translation_root_index,
        anim_pos_threshold=anim_pos_threshold,
        motion_metadata=motion_metadata,
        allow_infer=allow_infer,
    )
    # Rigid-bone export: trust the rotation channel + fixed rest offsets only
    # (pure FK), skipping the RIC position-channel override below. Bones stay
    # perfectly rigid; genuinely animated non-root joint translation is dropped.
    if rigid_bone:
        return anim_rot, needs_bvh_position_channels(anim_rot)

    target_global        = recover_from_bvh_ric_np(
        data,
        translation_root_index=translation_root_index,
        parents=parents,
        offsets=offsets,
        anim_pos_threshold=anim_pos_threshold,
        motion_metadata=motion_metadata,
        allow_infer=allow_infer,
    )              # (F, J, 3)
    glob_rot             = positions_global(anim_rot)                  # (F, J, 3)

    # Zero-offset leaf joints have no bone length, so the model's small per-joint
    # position error during generation must NOT be solved into a phantom local
    # translation on them. Pin them to their (zero) rest offset instead.
    parents_arr = np.asarray(parents)
    offsets_arr = np.asarray(offsets)
    joint_count = len(parents_arr)
    is_leaf = np.ones(joint_count, dtype=bool)
    is_leaf[parents_arr[parents_arr >= 0]] = False
    zero_offset_leaf = is_leaf & (np.linalg.norm(offsets_arr, axis=-1) <= 1e-6)
    zero_offset_leaf &= parents_arr >= 0
    if translation_root_index is not None and 0 <= translation_root_index < joint_count:
        zero_offset_leaf[translation_root_index] = False

    # joints whose FK-predicted global position drifts from the RIC truth
    per_joint_err = np.abs(target_global - glob_rot).max(axis=(0, 2)) # (J,)
    animated_joints = sorted(
        j for j in range(joint_count)
        if per_joint_err[j] > anim_pos_threshold and not zero_offset_leaf[j]
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
    # The direct solver above rewrites every joint, so re-pin the zero-offset
    # leaf joints back onto their rest offset.
    if zero_offset_leaf.any():
        new_pos[:, zero_offset_leaf] = offsets_arr[zero_offset_leaf]

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
    rigid_bone=False,
):
    """Recover a motion tensor and remap it into BVH-safe DFS order.

    ``recover_animation_from_motion_np`` intentionally preserves the input joint
    indexing because non-export callers still address joints by the original cond
    metadata indices. BVH export has the additional requirement that joint arrays
    must match hierarchy DFS order, so this helper layers the DFS remap on top of
    recovery without changing the base function's semantics.

    When *tpose_rest_rotations* is provided (``(J, 4)`` quaternion array in
    ``[w, x, y, z]`` order), the recovered rest-pose-relative rotations are baked
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
        rigid_bone=rigid_bone,
    )
    if anim is None:
        return None, list(joint_names), has_animated_pos

    if tpose_rest_rotations is not None:
        anim = recover_processed_animation_from_feature_animation(
            anim, tpose_rest_rotations,
        )
        # Baking the rest rotations into the local rotations leaves the offsets in
        # the feature (rest-removed) basis, so the solved local positions deviate
        # from the rest offsets even when the pre-bake feature animation was pure
        # rotation. ``has_animated_pos`` from the feature animation reflects a
        # different basis; recompute it on the baked animation so BVH export writes
        # the position channels the reconstructed pose actually needs. Without this,
        # rotation-only BVH (positions=False) reconstructs a garbled pose for
        # skeletons with non-identity rest rotations (e.g. GLB-derived references).
        has_animated_pos = needs_bvh_position_channels(anim)

    anim, joint_names = reorder_animation_to_dfs(anim, joint_names)
    return anim, joint_names, has_animated_pos
