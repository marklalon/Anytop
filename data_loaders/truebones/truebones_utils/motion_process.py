import BVH
from Animation import *
from InverseKinematics import animation_from_positions
import numpy as np 
import os 
from os.path import join as pjoin
from Quaternions import Quaternions
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
import random
import math
import statistics
import traceback
import torch
import bisect
import re 
from data_loaders.truebones.truebones_utils.param_utils import HML_AVG_BONELEN, FOOT_CONTACT_HEIGHT_THRESH, DATASET_DIR, MAX_PATH_LEN, MOTION_DIR, FOOT_CONTACT_VEL_THRESH, BVHS_DIR, OBJECT_SUBSETS_DICT, get_raw_data_dir, SNAKES, CHAIN_FORWARD_JOINTS, FLYING, FISH, VERTICAL_CLAMP_H
from utils.rotation_conversions import rotation_6d_to_matrix_np


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

def _normalize_joint_name(name):
    # Split on lowercase→UPPER (e.g. "ElkRFemur" → "Elk RFemur")
    split_name = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', name)
    # Also split on UPPER→UPPER+lower (e.g. "RFemur" → "R Femur")
    split_name = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', split_name)
    return re.sub(r'[^a-z0-9]+', ' ', split_name.lower()).strip()


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

################## Data Generation #####################
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
    target = np.array([[0,0,1]]).repeat(len(forward), axis=0)
    root_quat = Quaternions.between(forward, target)
    return root_quat


def _find_translation_root(anim):
    """Return the index of the first joint (from root) with significant position animation.

    Most skeletons carry root motion on joint 0, but some Truebones rigs
    (Horse, Bear, Camel, Trex, etc.) use intermediate bones like Bip01 or
    jt_Cog_C.  Detecting this allows the rest of the pipeline to centre,
    strip trajectory, and export BVH correctly.
    """
    for j in range(anim.positions.shape[1]):
        ptp = np.ptp(anim.positions[:, j], axis=0)
        if np.any(ptp > 1e-3):
            return j
    return 0

def _clamp_vertical_trajectory(processed_anim, object_type, H=VERTICAL_CLAMP_H):
    """Scale the translation root's Y animation so it stays within the vertical bound H.

    For FLYING creatures the world-Y of the translation root must not exceed +H.
    For FISH creatures it must not go below -H (deeper than H below the water surface).

    After HML orientation correction the character faces Z+ and Y is the vertical
    axis, so the local Y of the translation root closely tracks world-space height.
    We scale that channel by the ratio needed to bring the extreme value to ±H,
    preserving relative dynamics (a swoop that was 2× higher than a hover stays 2×
    higher, just compressed into the allowed range).
    """
    if object_type not in FLYING and object_type not in FISH:
        return processed_anim

    trans_root = _find_translation_root(processed_anim)
    # Use global positions so non-root translation joints (e.g. Bip01) are handled correctly.
    global_pos = positions_global(processed_anim)
    world_y = global_pos[:, trans_root, 1]

    if object_type in FLYING:
        y_max = world_y.max()
        if y_max <= H:
            return processed_anim
        scale_y = H / y_max
    else:  # FISH
        y_min = world_y.min()
        if y_min >= -H:
            return processed_anim
        scale_y = H / abs(y_min)

    new_positions = processed_anim.positions.copy()
    new_positions[:, trans_root, 1] *= scale_y
    return Animation(
        processed_anim.rotations.copy(),
        new_positions,
        processed_anim.orients.copy(),
        processed_anim.offsets.copy(),
        processed_anim.parents.copy(),
    )


""" move motion s.t the effective translation root's XZ is at the origin on the first frame.

For most skeletons joint 0 carries the root motion, but some rigs store it
on an intermediate bone (e.g. Bip01 for Horse).  We detect the effective
root via its global position and apply the shift to joint 0 (whose local
position equals its global position), so the entire skeleton moves via FK.
"""
def move_xz_to_origin(anim, root_pose_init_xz=None):
    if root_pose_init_xz is None:
        global_pos = positions_global(anim)
        trans_root = _find_translation_root(anim)
        root_pose_init_xz = global_pos[0, trans_root] * np.array([1, 0, 1])
    new_positions = anim.positions.copy()
    new_positions[:, 0] -= root_pose_init_xz
    new_offsets = anim.offsets.copy()
    new_offsets[0] -= root_pose_init_xz
    new_anim = Animation(anim.rotations.copy(), new_positions, anim.orients.copy(), new_offsets, anim.parents.copy())
    return new_anim, root_pose_init_xz


def strip_translation_root_xz(anim, translation_root_index):
    """Return an in-place version of the animation with the effective root XZ removed.

    For rigs whose locomotion lives on an intermediate joint such as Bip01, we must
    modify that joint's own local translation channel rather than pushing the motion
    up to joint 0. Otherwise the exported BVH changes skeleton dynamics and makes
    Hips appear to translate incorrectly.
    """
    global_pos = positions_global(anim)
    root_xz = global_pos[:, translation_root_index, [0, 2]]
    if np.max(np.abs(root_xz)) <= 1e-8:
        return anim

    new_positions = anim.positions.copy()
    if translation_root_index == 0 or anim.parents[translation_root_index] < 0:
        new_positions[:, translation_root_index, 0] -= root_xz[:, 0]
        new_positions[:, translation_root_index, 2] -= root_xz[:, 1]
    else:
        global_rots = rotations_global(anim)
        parent_index = anim.parents[translation_root_index]
        parent_global_pos = global_pos[:, parent_index]
        parent_global_rots = global_rots[:, parent_index]
        desired_global = global_pos[:, translation_root_index].copy()
        desired_global[:, 0] = 0.0
        desired_global[:, 2] = 0.0
        new_positions[:, translation_root_index] = (-parent_global_rots) * (desired_global - parent_global_pos)

    return Animation(
        anim.rotations.copy(),
        new_positions,
        anim.orients.copy(),
        anim.offsets.copy(),
        anim.parents.copy(),
    )

def _get_hml_orientation_quat(anim, object_type, face_joints=None, orientation_quat=None, forward_joint_index=None, forward_base_joint_index=None):
    return orientation_quat

"""" rotate the motion to initially face z+, ground at xz axis (negative y is below ground)"""
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

""" scale skeleton s.t longest armature is of length HML_AVG_BONELEN """
def scale(anim, scale_factor=None):
    if scale_factor is None:
        lengths = offset_lengths(anim)
        mean_len = statistics.mean(lengths)
        scale_factor = HML_AVG_BONELEN/mean_len
    new_anim = Animation(anim.rotations.copy(), anim.positions * scale_factor ,anim.orients.copy(), anim.offsets * scale_factor,
                         anim.parents.copy())
    return new_anim, scale_factor

""" get foot contact """
def get_foot_contact(positions, foot_joints_indices, vel_thresh):
    frames_num, joints_num = positions.shape[:2]
    foot_vel_x = (positions[1:,foot_joints_indices ,0] - positions[:-1,foot_joints_indices ,0]) ** 2
    foot_vel_y = (positions[1:, foot_joints_indices, 1] - positions[:-1, foot_joints_indices, 1]) **2
    foot_vel_z = (positions[1:, foot_joints_indices, 2] - positions[:-1, foot_joints_indices, 2]) **2
    total_vel = foot_vel_x + foot_vel_y + foot_vel_z
    foot_floor = np.percentile(positions[:, foot_joints_indices, 1], 5.0, axis=0, keepdims=True)
    relative_height = positions[1:, foot_joints_indices, 1] - foot_floor
    foot_contact_vel_map = np.where(
        np.logical_and(total_vel <= vel_thresh, np.abs(relative_height) <= FOOT_CONTACT_HEIGHT_THRESH),
        1,
        0,
    )
    foot_cont = np.zeros((frames_num-1, joints_num))
    foot_cont[:, foot_joints_indices] = foot_contact_vel_map.astype(int)

    return foot_cont

""" get 6d rotations continuous representation"""
def get_6d_rep(qs):
    qs_ = qs.copy()
    return qs_.rotation_matrix(cont6d=True)

"""" process anim object """
def process_anim(anim, object_type, root_pose_init_xz=None, scale_factor=None, face_joints=None, orientation_quat=None, forward_joint_index=None, forward_base_joint_index=None):
    rotated = rotate_to_hml_orientation(anim, object_type, face_joints, orientation_quat=orientation_quat, forward_joint_index=forward_joint_index, forward_base_joint_index=forward_base_joint_index)
    centered, root_pose_init_xz_ = move_xz_to_origin(rotated, root_pose_init_xz)
    scaled, scale_factor_ = scale(centered, scale_factor)
    return scaled, root_pose_init_xz_, scale_factor_

""" get object_type common characteristics, extracted from Tsode bvh"""
def get_common_features_from_T_pose(t_pose_bvh, object_type, face_joints=None):
    t_pose_anim, t_pos_names, t_pose_frame_time = BVH.load(t_pose_bvh)
    face_joints = resolve_face_joints(object_type, t_pos_names, t_pose_anim.parents, face_joints=face_joints)
    forward_joint_index = _find_forward_reference_joint(t_pos_names, t_pose_anim.parents)
    forward_base_joint_index = _find_neck_reference_joint(t_pos_names, t_pose_anim.parents)
    if object_type in SNAKES:  # limbless animals have no distinct foot joints
        suspected_foot_indices = [i for i in range(len(t_pos_names))]
    else:
        suspected_foot_indices = [i for i in range(len(t_pos_names)) if 'toe' in t_pos_names[i].lower() or 'foot' in t_pos_names[i].lower() or 
                                  'phalanx' in t_pos_names[i].lower() or 'hoof' in t_pos_names[i].lower() or 'ashi' in t_pos_names[i].lower()]
                # edge cases
        for si in suspected_foot_indices:
            if si in t_pose_anim.parents:
                #check if all childeren also in suspected_foot_indices, otherwise add them 
                children = [i for i in range(len(t_pos_names)) if t_pose_anim.parents[i] == si]
                for c in children:
                    if c not in suspected_foot_indices:
                        suspected_foot_indices.append(c)
    # first recover global positions, and then create a brand new non-damaged animation, with position consistent to the offsets 
    t_pose_positions = positions_global(t_pose_anim)
    with open(os.devnull, 'w') as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
        t_pose_anim, _1, _2 = animation_from_positions(positions=t_pose_positions, parents=t_pose_anim.parents, offsets=t_pose_anim.offsets, iterations=150, silent=True)
    t_pose_orientation_quat = get_root_quat(positions_global(t_pose_anim), object_type, face_joint_indx=face_joints, forward_joint_index=forward_joint_index, forward_base_joint_index=forward_base_joint_index)[0]
    scaled, root_pose_init_xz, scale_factor = process_anim(
        t_pose_anim,
        object_type,
        face_joints=face_joints,
        orientation_quat=t_pose_orientation_quat,
        forward_joint_index=forward_joint_index,
        forward_base_joint_index=forward_base_joint_index,
    )
    offsets = offsets_from_positions(positions_global(scaled), scaled.parents)[0]
    return root_pose_init_xz, scale_factor, offsets, suspected_foot_indices, scaled.rotations, t_pos_names, scaled, face_joints, t_pose_orientation_quat, forward_joint_index, forward_base_joint_index

def get_motion_features(ric_positions, rotations, foot_contact, velocity, max_joints):
    # F = Frames# , J = joints# 
    # parents (J,1)
    # positions (F, J, 3)
    # rotations (F, J, 6)
    # foot_contact (F - 1, J, 1)
    # velocity (F - 1, J, 3)
    # offsets (J, 3)
    
    # feature len = 13 (pos, rot, vel, foot)

    frames, joints = ric_positions.shape[0:2]
    if joints > max_joints:
        max_joints = joints
    pos = ric_positions[:-1]  ## (Frames-1, joints, 3)
    rot = rotations[:-1] ## (frames -1, joints, 6)
    vel = velocity ## (Frames - 1, joints , 3)
    foot = foot_contact.reshape(frames - 1, joints , 1) ## (Frames - 1, 1)
    features= np.concatenate([pos, rot, vel, foot], axis=-1) 
    return features, max_joints

'''return positions in root coords system. Meaning, each frame faces Z+, and the root is at [0, root_height, 0]'''
def get_rifke(global_positions, root_rot, translation_root_index=0):
    positions = global_positions.copy()
    '''Local pose'''
    positions[..., 0] -= positions[:, translation_root_index:translation_root_index + 1, 0]
    positions[..., 2] -= positions[:, translation_root_index:translation_root_index + 1, 2]
    '''All pose face Z+'''
    positions = np.repeat(root_rot[:, None], positions.shape[1], axis=1) * positions
    return positions

""" compute new rotations for anim that are relative to a natural tpose """
def compute_rots_from_tpos(tpos_quats, dest_quats, parents):
    new_rots = dest_quats.copy()
    new_rots[:, 0] = new_rots[:, 0] * -tpos_quats[:, 0]
    cum_rots = tpos_quats.copy()
    for j, p in enumerate(parents[1:], start=1):
        cum_rots[:, j] = cum_rots[:, p] * tpos_quats[:, j]
        new_rots[:, j] = cum_rots[:, p] * dest_quats[:, j] * -tpos_quats[:, j] * -cum_rots[:, p]
    return new_rots

""" returns policy for extracting kinematic chains from parent array, 
in attempt to divide the skeleton to meaningful kinchains. h_first mean the head joints are at the 
beggining of the parent array"""
def object_policy(obj):
    if obj in ["Mousey_m", "MouseyNoFingers", "Scorpion", "Raptor2"]:
        return "l_first"
    else:
        return "h_first"

""" returns cont6d params, including joints rotations, root rotation and rotational velocity, 
linear velocity and positions. Unlike BVH (and accordingly, Animation object) in which the parent holds the rotagtion of the child joint, 
in our data structure each joints holds it's own rotation (similar to humanML3D data structure and FK model)"""
def get_bvh_cont6d_params(anim, object_type, face_joints=None, joint_names=None, forward_joint_index=None, forward_base_joint_index=None, translation_root_index=0):
    positions = positions_global(anim)
    if face_joints is None:
        face_joints = resolve_face_joints(object_type, joint_names=joint_names, parents=anim.parents)
    quat_params = anim.rotations
    r_rot = get_root_quat(positions, object_type, face_joints, forward_joint_index=forward_joint_index, forward_base_joint_index=forward_base_joint_index)
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

""" processes animation, and returns a new animation that aligns with humanML3D in terms of orientation and scale"""
def get_hml_aligned_anim(bvh_path, object_type, root_pose_init_xz, scale_factor, tpos_rots, offsets, squared_positions_error, foot_indices=None, face_joints=None, orientation_quat=None, forward_joint_index=None, forward_base_joint_index=None, slice_inds=None, preloaded=None):
    if not isinstance(bvh_path, Animation):
        if preloaded is not None:
            raw_anim, names = preloaded
        else:
            raw_anim, names, frame_time = BVH.load(bvh_path)
        if slice_inds:
            raw_anim = raw_anim[slice_inds[0]:slice_inds[1]]
        #print('frame time', frame_time )
        frames_num, joints_num = raw_anim.positions.shape[:2]

        ## process animation: rotate to correct orientation, center, and scale
        processed_anim, _xz, _sf = process_anim(
            raw_anim,
            object_type,
            root_pose_init_xz,
            scale_factor,
            face_joints=face_joints,
            orientation_quat=orientation_quat,
            forward_joint_index=forward_joint_index,
            forward_base_joint_index=forward_base_joint_index,
        )
        ## clamp vertical trajectory for flying/fish creatures (after scale, in HML units)
        processed_anim = _clamp_vertical_trajectory(processed_anim, object_type)
    else:
        names = list()
        processed_anim = bvh_path
        frames_num = len(processed_anim)

    ## create new animation object in which the rotations are w.r.t the actual Tpos
    tpos_rots_correct_shape  = tpos_rots[None, 0].repeat(frames_num, axis = 0)
    rots = compute_rots_from_tpos(tpos_rots_correct_shape, processed_anim.rotations, processed_anim.parents)
    anim_positions = offsets.copy()[None, :].repeat(frames_num, axis = 0)
    anim_positions[:, 0] = processed_anim.positions[:, 0]
    # Preserve position animation on intermediate bones that carry root motion
    # (e.g. Bip01 for Horse, jt_Cog_C for Trex, NPC_Pelvis for Bear).
    # The T-pose rotation reparameterization changes the parent rotation chain,
    # so we cannot simply copy local positions from processed_anim — doing so
    # produces wildly wrong global positions (e.g. Horse Bip01 loses 90% of its
    # jump height).  Instead we solve for the local position that reproduces
    # processed_anim's global position under the new parent rotations.
    # Joints are processed in index order (parent-before-child) so that each
    # solve sees the already-corrected ancestors.
    animated_pos_joints = sorted(
        j for j in range(1, processed_anim.positions.shape[1])
        if np.any(np.ptp(processed_anim.positions[:, j], axis=0) > 1e-4)
    )
    if animated_pos_joints:
        processed_global_pos = positions_global(processed_anim)
        for j in animated_pos_joints:
            temp_anim = Animation(rots, anim_positions, processed_anim.orients, offsets, processed_anim.parents)
            temp_global_rots = rotations_global(temp_anim)
            temp_global_pos = positions_global(temp_anim)
            p = processed_anim.parents[j]
            anim_positions[:, j] = (-temp_global_rots[:, p]) * (processed_global_pos[:, j] - temp_global_pos[:, p])
    # create animation object which is defined over correct tpos
    new_anim = Animation(rots, anim_positions  , processed_anim.orients, offsets, processed_anim.parents)

    processed_global_pos = positions_global(processed_anim)
    new_global_pos = positions_global(new_anim)
    squared_error = np.mean((processed_global_pos - new_global_pos) ** 2)
    error_key = bvh_path if isinstance(bvh_path, str) else '__animation__'
    if slice_inds is not None and not isinstance(bvh_path, Animation):
        error_key = f'{bvh_path}[{slice_inds[0]}:{slice_inds[1]}]'
    squared_positions_error[error_key] = float(squared_error)

    return new_anim, names  
    
""" get motion feature representation"""
def get_motion(bvh_path, foot_contact_vel_thresh, object_type, max_joints, root_pose_init_xz, scale_factor, offsets, foot_indices, tpos_rots, squared_positions_error, face_joints=None, orientation_quat=None, forward_joint_index=None, forward_base_joint_index=None, slice_inds=None, preloaded=None):
    try:
        new_anim, names = get_hml_aligned_anim(
            bvh_path,
            object_type,
            root_pose_init_xz,
            scale_factor,
            tpos_rots,
            offsets,
            squared_positions_error,
            foot_indices,
            face_joints,
            orientation_quat,
            forward_joint_index,
            forward_base_joint_index,
            slice_inds,
            preloaded=preloaded,
        )
        translation_root_index = _find_translation_root(new_anim)
        new_anim = strip_translation_root_xz(new_anim, translation_root_index)
        ## extract features
        # cont_6d_params, r_velocity, velocity, r_rot, global_positions = get_bvh_cont6d_params(new_anim, object_type)
        cont_6d_params, r_velocity, velocity, r_rot, global_positions = get_bvh_cont6d_params(
            new_anim,
            object_type,
            face_joints=face_joints,
            joint_names=names,
            forward_joint_index=forward_joint_index,
            forward_base_joint_index=forward_base_joint_index,
            translation_root_index=translation_root_index,
        )
        foot_contact = get_foot_contact(global_positions, foot_indices, foot_contact_vel_thresh) 
        '''Get Joint Rotation Invariant Position Represention'''
        # local velocity wrt root coords system as described in get_rifke definition 
        positions = get_rifke(global_positions, r_rot, translation_root_index=translation_root_index)
        # root_y = positions[:, 0, 1:2]
        # r_velocity = np.arcsin(r_velocity[:, 2:3])
        # l_velocity = velocity[:, [0, 2]]
        local_vel = np.repeat(r_rot[1:, None], global_positions.shape[1], axis=1) * (global_positions[1:] - global_positions[:-1])
        # Strip root XZ plane velocity so the representation is fully root-relative and
        # consistent with RIFKE (which already zeros root XZ in position space).
        # This removes trajectory variation across creature types, letting the model focus
        # on body articulation patterns regardless of locomotion speed or direction.
        local_vel[:, translation_root_index, [0, 2]] = 0.0
        # root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)
        features, max_joints = get_motion_features(positions, cont_6d_params, foot_contact, local_vel, max_joints)
        return features, new_anim.parents, max_joints, new_anim
    except Exception as err:
        print(err)
        return None, None, max_joints, None

""" computes mean and std for a list of motions """
def get_mean_std(data):
    if len(data) > 0:
        Mean = data.mean(axis=0) # (Joints, 25)
        Std = data.std(axis=0) # # (Joints, 25)
        Std[0, :3] = Std[0, :3].mean() / 1.0 # all joints except root ric pos
        Std[0, 3:9] = Std[0, 3:9].mean() / 1.0 # all joints except root rotation
        Std[0, 9:12] = Std[0, 9:12].mean() / 1.0 # all joints except root local velocity

        Std[1:, :3] = Std[1:, :3].mean() / 1.0 # all joints except root ric pos
        Std[1:, 3:9] = Std[1:, 3:9].mean() / 1.0 # all joints except root rotation
        Std[1:, 9:12] = Std[1:, 9:12].mean() / 1.0 # all joints except root local velocity
        if len(Std[:, 12][Std[:, 12]!=0]) > 0:
            Std[:, 12][Std[:, 12]!=0] = Std[:, 12][Std[:, 12]!=0].mean() / 1.0 
        Std[:, 12][Std[:, 12]==0] = 1.0 # replace zeros with ones
        
        return Mean, Std
  
""" compures Relations and Distance marices"""
def create_topology_edge_relations(parents, max_path_len = 5): # joint j+1 contains len(j, j+1)
    edge_types = {'self':0, 'parent':1, 'child':2, 'sibling':3, 'no_relation':4, 'end_effector':5, 'ts_token_conn': 6}
    n = len(parents)
    topo_rel = np.zeros((n, n))
    edge_rel = np.ones((n, n)) * edge_types['no_relation'] 
    for i in range(n):
        parent = parents[i]
        ee = True
        for j in range(n):
            parent_j = parents[j]
            """Update edge type"""
            edge_type = edge_types['no_relation']
            if i == j: #self
                edge_type = edge_types['self'] 
            elif parent_j == i: #child
                ee=False
                edge_type = edge_types['child']
            elif j == parent: #parent
                edge_type = edge_types['parent'] 
            elif parent_j == parent: #sibling
                edge_type = edge_types['sibling']
            edge_rel[i, j] = edge_type

            """Update path length type"""
            
            if i == j:
                topo_rel[i, j] = 0      
            elif j < i:
                topo_rel[i, j] = topo_rel[j, i]
            elif parent_j == i: # parent-child relation
                topo_rel[i, j] = 1
            else: #any other 
                topo_rel[i, j] = topo_rel[i, parent_j] + 1
        if ee:
            edge_rel[i, i] = edge_types['end_effector']
            
    topo_rel[topo_rel > max_path_len] = max_path_len
    return edge_rel, topo_rel

def _reference_stem_tokens(file_path):
    stem = os.path.splitext(os.path.basename(file_path))[0]
    normalized = _normalize_joint_name(stem)
    return normalized.split(), normalized.replace(' ', '')


def _is_tpose_reference_path(file_path):
    tokens, compact = _reference_stem_tokens(file_path)
    return (
        'tpose' in compact
        or 'tpos' in compact
        or 'bindpose' in compact
        or 'restpose' in compact
        or ('pose' in tokens and 't' in tokens)
    )


def _is_idle_reference_path(file_path):
    tokens, compact = _reference_stem_tokens(file_path)
    return 'idle' in compact or any(token.startswith('idle') for token in tokens)


def _is_walk_reference_path(file_path):
    tokens, compact = _reference_stem_tokens(file_path)
    return 'walk' in compact or any(token.startswith('walk') for token in tokens)


""" find a character-level orientation reference clip with priority T Pose > Idle > Walk """
def find_orientation_reference_path(bvh_files):
    for file_path in bvh_files:
        if _is_tpose_reference_path(file_path):
            bvh_files.remove(file_path)
            return file_path, 'tpose'

    for matcher, source_name in (
        (_is_idle_reference_path, 'idle'),
        (_is_walk_reference_path, 'walk'),
    ):
        for file_path in bvh_files:
            if matcher(file_path):
                return file_path, source_name

    return bvh_files[0], 'fallback'


def _process_bvh_file(file_path, object_type, max_joints, root_pose_init_xz, scale_factor,
                      offsets, foot_indices, tpos_rots, face_joints, orientation_quat, forward_joint_index, forward_base_joint_index):
    local_errors = dict()
    # Load the BVH file once; pass it as `preloaded` to every get_motion call so that
    raw_anim, names, frame_time = BVH.load(file_path)
    anim_len = len(raw_anim)
    begin = 0
    file_max_joints = max_joints
    file_results = []

    while begin < anim_len:
        if anim_len - begin > 240:
            slice_ind = begin + 200
        else:
            slice_ind = anim_len

        motion, parents, file_max_joints, new_anim = get_motion(
            file_path,
            FOOT_CONTACT_VEL_THRESH,
            object_type,
            file_max_joints,
            root_pose_init_xz,
            scale_factor,
            offsets,
            foot_indices,
            tpos_rots,
            local_errors,
            face_joints=face_joints,
            orientation_quat=orientation_quat,
            forward_joint_index=forward_joint_index,
            forward_base_joint_index=forward_base_joint_index,
            slice_inds=[begin, slice_ind],
            preloaded=(raw_anim, names),
        )
        current_begin = begin
        begin = slice_ind

        if motion is None:
            print(f'failed to process file: {file_path}, slice {current_begin}:{slice_ind}')
            continue

        _, file_name = os.path.split(file_path)
        file_results.append({
            'action': file_name.split('.')[0],
            'motion': motion,
            'parents': parents,
            'new_anim': new_anim,
            'names': names,
        })

    return {
        'errors': local_errors,
        'max_joints': file_max_joints,
        'results': file_results,
    }
     
"""Prepare processed tensors for all the files of a given object without writing them to disk yet."""
def _prepare_object_outputs(object_type, max_joints, face_joints=None, bvhs_dir=None, t_pos_path=None, max_files=None, num_workers=1):
    object_cond = dict()
    if bvhs_dir is None:
        bvhs_dir = pjoin(get_raw_data_dir(), object_type)
    if not os.path.isdir(bvhs_dir):
        print(f'skipping {object_type}: raw BVH directory not found at {bvhs_dir}')
        return None
    bvh_files = sorted([pjoin(bvhs_dir, f) for f in os.listdir(bvhs_dir) if f.lower().endswith('.bvh')])
    if len(bvh_files) == 0:
        print(f'skipping {object_type}: no BVH files found in {bvhs_dir}')
        return None
    ## get a character-level orientation reference clip
    if t_pos_path is None or t_pos_path == '':
        t_pos_path, orientation_reference_source = find_orientation_reference_path(bvh_files)
    else: 
        orientation_reference_source = 'explicit'
        # removes tpos bvh fron bvh_files, as it represents a static motion and should be used only for
        # extracting common characteristics. If this is not the case, disable this part
        bvh_files.remove(t_pos_path)
    if max_files is not None:
        bvh_files = bvh_files[:max_files]

    squared_positions_error = dict()
    root_pose_init_xz, scale_factor, offsets, foot_indices, tpos_rots, names, tpos_anim, face_joints, orientation_quat, forward_joint_index, forward_base_joint_index = get_common_features_from_T_pose(t_pos_path, object_type, face_joints=face_joints)
    t_pos_motion, parents, max_joints, new_anim = get_motion(tpos_anim, FOOT_CONTACT_VEL_THRESH, object_type, max_joints, root_pose_init_xz, scale_factor, offsets, foot_indices, tpos_rots, squared_positions_error, face_joints=face_joints, orientation_quat=orientation_quat, forward_joint_index=forward_joint_index, forward_base_joint_index=forward_base_joint_index)
    object_cond['tpos_first_frame'] = t_pos_motion[0]
    # create topology conditions
    joint_relations, joints_graph_dist = create_topology_edge_relations(tpos_anim.parents, max_path_len = MAX_PATH_LEN)
    object_cond['joint_relations'] = joint_relations
    object_cond['joints_graph_dist'] = joints_graph_dist
    object_cond['object_type'] = object_type
    object_cond['parents'] = parents
    object_cond['offsets'] = offsets
    object_cond['joints_names'] = names
    object_cond['face_joints'] = list(face_joints)
    object_cond['face_joint_names'] = [names[index] for index in face_joints]
    object_cond['orientation_reference_source'] = orientation_reference_source
    object_cond['orientation_reference_file'] = os.path.basename(t_pos_path)
    kinematic_chains = parents2kinchains(parents, object_policy(object_type))
    object_cond['kinematic_chains'] = kinematic_chains
    all_tensors = list()

    num_workers = min(len(bvh_files), max(1, int(num_workers)))
    if num_workers > 1:
        print(f'processing {len(bvh_files)} BVH files for {object_type} with {num_workers} worker threads', flush=True)

    def process_file(file_path):
        print("processing file: " + file_path, flush=True)
        return _process_bvh_file(
            file_path,
            object_type,
            max_joints,
            root_pose_init_xz,
            scale_factor,
            offsets,
            foot_indices,
            tpos_rots,
            face_joints,
            orientation_quat,
            forward_joint_index,
            forward_base_joint_index,
        )

    if num_workers == 1:
        file_outputs = [process_file(file_path) for file_path in bvh_files]
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            file_outputs = list(executor.map(process_file, bvh_files))

    files_counter = 0
    frames_counter = 0
    prepared_results = []
    for file_output in file_outputs:
        squared_positions_error.update(file_output['errors'])
        max_joints = max(max_joints, file_output['max_joints'])
        for result in file_output['results']:
            motion = result['motion']
            all_tensors.append(motion)
            files_counter += 1
            frames_counter += motion.shape[0]
            prepared_results.append(result)

    if len(all_tensors) == 0:
        print(f'skipping {object_type}: no valid motion tensors were produced')
        return None
    all_tensors = np.concatenate(all_tensors, axis=0)
    mean, std = get_mean_std(all_tensors)
    object_cond["mean"] = mean
    object_cond["std"] = std

    return {
        'object_type': object_type,
        'object_cond': object_cond,
        'errors': squared_positions_error,
        'max_joints': max_joints,
        'results': prepared_results,
        'files_counter': files_counter,
        'frames_counter': frames_counter,
        'face_joints': face_joints,
    }

"""Write a prepared object payload to disk with stable sequential clip naming."""
def _write_object_outputs(save_dir, object_payload, files_counter):
    object_type = object_payload['object_type']
    face_joints = object_payload['face_joints']
    frames_counter = 0

    for result in object_payload['results']:
        motion = result['motion']
        parents = result['parents']
        files_counter += 1
        frames_counter += motion.shape[0]
        name = object_type + "_" + result['action'] + "_" + str(files_counter)
        np.save(pjoin(save_dir, MOTION_DIR, name + '.npy'), motion)
        # Use positions=True whenever any non-root joint carries animated positions
        # (e.g. Horse Bip01).  Without this, BVH.save silently drops those
        # channels because non-root joints get only 3-channel (rotation) entries.
        anim_obj = result['new_anim']
        has_animated_nonroot_pos = np.any(
            np.ptp(anim_obj.positions[:, 1:, :], axis=0) > 1e-4
        )
        BVH.save(pjoin(save_dir, BVHS_DIR, name+".bvh"), anim_obj, result['names'],
                 positions=bool(has_animated_nonroot_pos))

    return files_counter, frames_counter

def _resolve_preprocessing_workers(objects, object_workers=8, file_workers=8):
    object_count = max(1, len(objects))
    object_workers = min(object_count, max(1, int(object_workers)))
    file_workers = max(1, int(file_workers))
    total_workers = object_workers * file_workers
    return total_workers, object_workers, file_workers


def _prepare_object_outputs_worker(object_type, max_files, file_workers):
    return _prepare_object_outputs(
        object_type,
        max_joints=23,
        max_files=max_files,
        num_workers=file_workers,
    )

""" creates processed tensors for all the files of a given object. Returens statistics and the object condition,
which includes tpos, relation/distances matrices, offsets, parents, joints names, kinematic chains, mean and std"""    
def process_object(object_type, files_counter, frames_counter, max_joints, squared_positions_error, save_dir = DATASET_DIR, face_joints=None, bvhs_dir=None, t_pos_path=None, max_files=None, num_workers=1):
    object_payload = _prepare_object_outputs(
        object_type,
        max_joints,
        face_joints=face_joints,
        bvhs_dir=bvhs_dir,
        t_pos_path=t_pos_path,
        max_files=max_files,
        num_workers=num_workers,
    )
    if object_payload is None:
        return files_counter, frames_counter, max_joints, None

    squared_positions_error.update(object_payload['errors'])
    max_joints = max(max_joints, object_payload['max_joints'])
    files_counter, object_frames_counter = _write_object_outputs(
        save_dir,
        object_payload,
        files_counter,
    )
    frames_counter += object_frames_counter

    return files_counter, frames_counter, max_joints, object_payload['object_cond']

""" create dataset """
def create_data_samples(objects=None, max_files_per_object=None, dataset_dir=None, object_workers=8, file_workers=8):
    ## prepare
    target_dataset_dir = dataset_dir or DATASET_DIR
    os.makedirs(pjoin(target_dataset_dir, MOTION_DIR), exist_ok=True)
    os.makedirs(pjoin(target_dataset_dir, BVHS_DIR), exist_ok=True)
    
    ## process
    if objects is None:
        raw_data_dir = get_raw_data_dir()
        objects = sorted(
            obj for obj in os.listdir(raw_data_dir)
            if os.path.isdir(pjoin(raw_data_dir, obj))
        )

    total_workers, obj_workers, fw = _resolve_preprocessing_workers(
        objects,
        object_workers=object_workers,
        file_workers=file_workers,
    )
    print(f'Preprocessing {len(objects)} characters: '
          f'{obj_workers} object workers x {fw} file workers '
          f'(up to {total_workers} concurrent preprocess workers)')

    payloads = [None] * len(objects)
    if obj_workers <= 1:
        for idx, object_type in enumerate(objects):
            payloads[idx] = _prepare_object_outputs(
                object_type,
                max_joints=23,
                max_files=max_files_per_object,
                num_workers=fw,
            )
    else:
        with ProcessPoolExecutor(max_workers=obj_workers) as executor:
            future_to_idx = {
                executor.submit(
                    _prepare_object_outputs_worker,
                    object_type,
                    max_files_per_object,
                    fw,
                ): idx
                for idx, object_type in enumerate(objects)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                payloads[idx] = future.result()  # propagates exception to abort all processing

    files_counter = 0
    frames_counter = 0
    max_joints = 23
    objects_counter = dict()
    squared_positions_error = dict()
    cond = dict()

    for idx, object_type in enumerate(objects):
        payload = payloads[idx]
        if payload is None:
            continue
        squared_positions_error.update(payload['errors'])
        max_joints = max(max_joints, payload['max_joints'])
        cur_counter = files_counter
        files_counter, object_frames = _write_object_outputs(
            target_dataset_dir,
            payload,
            files_counter,
        )
        frames_counter += object_frames
        cond[object_type] = payload['object_cond']
        objects_counter[object_type] = files_counter - cur_counter

    print('Total clips: %d, Frames: %d, Duration: %fm' %(files_counter, frames_counter, frames_counter / 12.5 / 60))
    print('max joints: %d' %(max_joints))
    text_file = open(pjoin(target_dataset_dir, 'metadata.txt'), "w")
    n = text_file.write('max joints: %d\n' %(max_joints))
    n = text_file.write('total frames: %d\n' %(frames_counter))
    n = text_file.write('duration: %d\n' %(frames_counter / 12.5 / 60))
    n = text_file.write('~~~~ objects_counts - Total: %d ~~~~\n' %(files_counter) )
    for obj in objects_counter:
        text_file.write('%s: %d\n' %(obj, objects_counter[obj]))
    text_file.close()

    error_file = open(pjoin(target_dataset_dir, 'positions_error_rate.txt'), "w")
    n = error_file.write('Position squared error per bvh file:')
    for f in squared_positions_error.keys():
        error_file.write('%s: %f\n' %(f, squared_positions_error[f]))
    error_file.close()
    
    np.save(pjoin(target_dataset_dir, "cond.npy"), cond)
##################################################################

############ Recover animation from motion features ##############
def recover_root_quat_and_pos_np(data):
    # root_feature_vector.shape = (frames, angular_vel || linear_xz_vel || root_height || zero pad)
    r_rot_quat = Quaternions.from_transforms(rotation_6d_to_matrix_np(data[:, 3:9]))

    r_pos = np.zeros(data.shape[:-1] + (3,))
    r_pos[..., 1:, [0, 2]] = data[..., :-1, [9, 11]]
    '''Add Y-axis rotation to root position'''
    r_pos = -r_rot_quat * r_pos

    r_pos = np.cumsum(r_pos, axis = -2)
    r_pos[...,1] = data[..., 1]
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
def recover_from_bvh_ric_np(data):
    r_rot_quat, r_pos = recover_root_quat_and_pos_np(data[..., 0, :])
    positions = data[..., 1:, :3]
    positions = np.repeat(-r_rot_quat[..., None, :], positions.shape[-2], axis=-2) * positions
    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    '''Concate root and joints'''
    positions = np.concatenate([r_pos[..., np.newaxis, :], positions], axis=-2)
    return positions

""" recover xyz positions from rot (root relative positions) torch """
def recover_from_bvh_rot_np(data, parents, offsets):
    r_rot_quat, r_pos = recover_root_quat_and_pos_np(data[:,0])
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
    rotations[:, 0] = -r_rot_quat * rotations[:, 0]
    positions = offsets[None].repeat(data.shape[0], axis=0)
    positions[:, 0] = r_pos
    anim = Animation(rotations=rotations, positions=positions, parents=parents, offsets=offsets, orients=Quaternions.id(0))
    
    return positions_global(anim), anim

################################################################

################ Parents to kinematic chains ###################
def reverse_insort(a, x, lo=0, hi=None):
    """Insert item x in list a, and keep it reverse-sorted assuming a
    is reverse-sorted.

    If x is already in a, insert it to the right of the rightmost x.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if x > a[mid]: hi = mid
        else: lo = mid+1
    a.insert(lo, x)

def parents2kinchains(parents, policy = 'h_first'):
    chains = list()
    children_dict = {i:[] for i in range(len(parents))}
    for j,p in enumerate(parents[1: ], start=1):
        if policy == 'h_first':
            reverse_insort(children_dict[p], j)
        else:
            bisect.insort(children_dict[p], j)
    recursion_kinchains([], 0, children_dict, chains, policy)
    return chains

def recursion_kinchains(chain, j, children_dict, chains, policy):
    children = children_dict[j]
    if len(children) == 0: #ee
        chain.append(j)
        chains.append(chain) 
    elif len(children) == 1:
        chain.append(j)
        recursion_kinchains(chain, children[0], children_dict, chains, policy)
    else:
        chain.append(j)
        if policy == 'h_first':
            main_child = max(children)
        else:
            main_child = min(children)
        for child in children:
            if child == main_child:
                recursion_kinchains(chain, child, children_dict, chains, policy)
            else:
                recursion_kinchains([j], child, children_dict, chains, policy)  
      
################################################################

####################### Augmentations ##########################
def remove_joints_augmentation(data, removal_rate, mean, std):
    motion, m_length, object_type, parents, joints_graph_dist, joints_relations, tpos_first_frame, offsets, joints_names_embs, kinematic_chains = data['motion'], data['length'], data['object_type'], data['parents'], data['joints_graph_dist'], data['joints_relations'], data['tpos_first_frame'], data['offsets'], data['joints_names_embs'], data['kinematic_chains']
    ee = [chain[-1] for chain in kinematic_chains]
    possible_feet = np.unique(np.where(motion[..., -1] > 0)[1])
    if object_type in SNAKES:
        possible_feet=[]
    removal_options = [j for j in ee if j not in possible_feet]
    # removal_rate = min(1.0, (removal_rate*len(parents)) / len(removal_options))
    remove_joints = sorted(random.sample(removal_options, math.floor(len(removal_options) * removal_rate)), reverse=True)
    motion = np.delete(motion, remove_joints, axis=1)
    new_ee = [parents[j] for j in remove_joints if np.count_nonzero(parents == parents[j]) == 1]
    for el in new_ee:
        joints_relations[el, el] = 5    
    parents = np.delete(parents, remove_joints, axis=0)
    joints_relations = np.delete(np.delete(joints_relations, remove_joints, axis=0), remove_joints, axis=1)
        
    for rj in remove_joints:
        parents[parents > rj] -= 1
    joints_graph_dist = np.delete(np.delete(joints_graph_dist, remove_joints, axis=0), remove_joints, axis=1)
    tpos_first_frame = np.delete(tpos_first_frame, remove_joints, axis=0)
    offsets = np.delete(offsets, remove_joints, axis=0)
    joints_names_embs = np.delete(joints_names_embs, remove_joints, axis=0)
    mean = np.delete(mean, remove_joints, axis=0)
    std = np.delete(std, remove_joints, axis=0)
    object_type = f'{object_type}__remove{remove_joints}'
    return motion, m_length, object_type, parents, joints_graph_dist, joints_relations, tpos_first_frame, offsets, joints_names_embs, kinematic_chains, mean, std

def add_joint_augmentation(data, mean, std):
    motion, m_length, object_type, parents, joints_graph_dist, joints_relations, tpos_first_frame, offsets, joints_names_embs, kinematic_chains = data['motion'], data['length'], data['object_type'], data['parents'], data['joints_graph_dist'], data['joints_relations'], data['tpos_first_frame'], data['offsets'], data['joints_names_embs'], data['kinematic_chains']
    n_joints = motion.shape[1]
    n_frames = motion.shape[0]
    # added joint mut follow:
    # j has exactly 1 child 
    # j parent is not the root joint
    # j is not the root joint
    possible_joints_to_add = [j for j in range(1, n_joints) if np.count_nonzero(joints_relations[j] == 2) == 1 and joints_relations[j,0] != 1]
    if len(possible_joints_to_add) == 0:
        return motion, m_length, object_type, parents, joints_graph_dist, joints_relations, tpos_first_frame, offsets, joints_names_embs, kinematic_chains, mean, std
    add_j = random.choice(possible_joints_to_add)
    # motion features
    j_feats = motion[:, add_j].copy()
    p_feats = motion[:, parents[add_j]]
    new_feats = ((j_feats + p_feats)/2).copy()
    new_feats[..., 3:9] = j_feats[..., 3:9].copy() # rotations
    new_feats[..., 12] = j_feats[..., 12].copy() # feet 
    j_feats[..., 3:9] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])[None].repeat(n_frames, axis=0)
    
    # tpos features
    tpos_j_feats = tpos_first_frame[add_j].copy()
    tpos_p_feats = tpos_first_frame[parents[add_j]]
    tpos_new_feats = ((tpos_j_feats + tpos_p_feats)/2)
    tpos_new_feats[3:9] = tpos_j_feats[3:9].copy() # rotations
    tpos_new_feats[12] = tpos_j_feats[12] # feet 
    tpos_j_feats[3:9] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    
    # mean features
    mean_j_feats = mean[add_j].copy()
    mean_p_feats = mean[parents[add_j]]
    mean_new_feats = ((mean_j_feats + mean_p_feats)/2).copy()
    mean_new_feats[3:9] = mean_j_feats[3:9].copy() # rotations
    mean_new_feats[12] = mean_j_feats[12] # feet 
    mean_j_feats[3:9] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    
    # std features
    std_new_feats = std[add_j].copy()
    
    # joints names embs features 
    emb_j_feats = joints_names_embs[add_j]
    emb_p_feats = joints_names_embs[parents[add_j]]
    emb_new_feats = (emb_j_feats + emb_p_feats)/2
    
    # apply augmentation
    #motion
    augmented = np.concatenate([motion[:, :add_j], new_feats[:, None], j_feats[:, None], motion[:, add_j+1:]], axis=1).copy()
    #tpos_first_frame
    tpos_first_frame_augmented = np.vstack([tpos_first_frame[:add_j], tpos_new_feats[None], tpos_j_feats[None], tpos_first_frame[add_j+1:]]).copy()
    #mean TODO: AUGMENT LIKE MOTION AND TPOS 
    mean_augmented = np.vstack([mean[:add_j], mean_new_feats[None], mean_j_feats[None], mean[add_j+1:]]).copy()
    #std TODO: AUGMENT LIKE MOTION AND TPOS 
    std_augmented = np.vstack([std[:add_j], std_new_feats[None], std[add_j:]]).copy()
    #joints_names_embs
    joints_names_embs_augmented = np.vstack([joints_names_embs[:add_j], emb_new_feats[None], joints_names_embs[add_j:]]).copy()
    # parents 
    augmented_parents = parents.copy()
    augmented_parents[augmented_parents >= add_j] += 1
    augmented_parents = augmented_parents.tolist()
    augmented_parents = np.array(augmented_parents[:add_j] + [add_j] + augmented_parents[add_j:])

    # topology conditions 
    relations, graph_dist = create_topology_edge_relations(augmented_parents.tolist(), max_path_len = MAX_PATH_LEN)
    
    # all others 
    offsets = np.vstack([offsets[:add_j], offsets[add_j]/2, offsets[add_j]/2, offsets[add_j+1:]])
    object_type = f'{object_type}__add{add_j}'
    return augmented, m_length, object_type, augmented_parents, graph_dist, relations, tpos_first_frame_augmented, offsets, joints_names_embs_augmented, kinematic_chains, mean_augmented, std_augmented
################################################################

########################### Tests ##############################
def process_single_object_type(object_type, save_dir, file_workers=8):
    ## prepare
    os.makedirs(pjoin(save_dir, MOTION_DIR), exist_ok=True)
    os.makedirs(pjoin(save_dir, BVHS_DIR), exist_ok=True)
    
    ## process
    files_counter = 0
    frames_counter = 0
    max_joints = 23
    objects_counter = dict()
    squared_positions_error = dict()
    cond = dict()
    cur_counter = files_counter
    files_counter, frames_counter, max_joints, object_cond = process_object(
        object_type,
        files_counter,
        frames_counter,
        max_joints,
        squared_positions_error,
        save_dir=save_dir,
        num_workers=file_workers,
    )
    cond[object_type] = object_cond
    objects_counter[object_type] = files_counter - cur_counter 

    print('Total clips: %d, Frames: %d, Duration: %fm' %(files_counter, frames_counter, frames_counter / 12.5 / 60))
    print('max joints: %d' %(max_joints))
    text_file = open(pjoin(save_dir, 'metadata.txt'), "w")
    n = text_file.write('max joints: %d\n' %(max_joints))
    n = text_file.write('total frames: %d\n' %(frames_counter))
    n = text_file.write('duration: %d\n' %(frames_counter / 12.5 / 60))
    n = text_file.write('~~~~ objects_counts - Total: %d ~~~~\n' %(files_counter) )
    for obj in objects_counter:
        text_file.write('%s: %d\n' %(obj, objects_counter[obj]))
    text_file.close()

    error_file = open(pjoin(save_dir, 'positions_error_rate.txt'), "w")
    n = error_file.write('Position squared error per bvh file:')
    for f in squared_positions_error.keys():
        error_file.write('%s: %f\n' %(f, squared_positions_error[f]))
    error_file.close()
    
    np.save(pjoin(save_dir, "cond.npy"), cond)
    
    
def process_skeleton(object_name, bvh_dir, face_joints, save_dir, tpos_bvh=None):
    ## prepare
    os.makedirs(pjoin(save_dir, MOTION_DIR), exist_ok=True)
    os.makedirs(pjoin(save_dir, BVHS_DIR), exist_ok=True)
    
    ## process
    files_counter = 0
    frames_counter = 0
    max_joints = 23
    objects_counter = dict()
    squared_positions_error = dict()
    cond = dict()
    cur_counter = files_counter
    files_counter, frames_counter, max_joints, object_cond = process_object(object_name, files_counter, frames_counter, max_joints, squared_positions_error, save_dir=save_dir, bvhs_dir=bvh_dir, face_joints=face_joints, t_pos_path=tpos_bvh)
    # BUG4 (intentional): MP4 generation is omitted here to skip expensive video
    # generation during process_skeleton. Generating video previews is not
    # Note: MP4 generation has been removed - no save_animations parameter needed.
    if object_cond is None:
        print(f"No valid BVH data found for '{object_name}', aborting.")
        return
    cond[object_name] = object_cond
    objects_counter[object_name] = files_counter - cur_counter 

    print('Total clips: %d, Frames: %d, Duration: %fm' %(files_counter, frames_counter, frames_counter / 12.5 / 60))
    print('max joints: %d' %(max_joints))
    text_file = open(pjoin(save_dir, 'metadata.txt'), "w")
    n = text_file.write('max joints: %d\n' %(max_joints))
    n = text_file.write('total frames: %d\n' %(frames_counter))
    n = text_file.write('duration: %d\n' %(frames_counter / 12.5 / 60))
    n = text_file.write('~~~~ objects_counts - Total: %d ~~~~\n' %(files_counter) )
    for obj in objects_counter:
        text_file.write('%s: %d\n' %(obj, objects_counter[obj]))
    text_file.close()

    error_file = open(pjoin(save_dir, 'positions_error_rate.txt'), "w")
    n = error_file.write('Position squared error per bvh file:')
    for f in squared_positions_error.keys():
        error_file.write('%s: %f\n' %(f, squared_positions_error[f]))
    error_file.close()
    
    np.save(pjoin(save_dir, "cond.npy"), cond)
################################################################