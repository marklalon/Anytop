from motion_lib import BVH, FBX, Animation, Quaternions
from motion_lib.Animation import positions_global, rotations_global, offsets_from_positions, offsets_global, offset_lengths
from motion_lib import animation_from_positions
from collections import Counter, defaultdict
import json
import numpy as np 
import os 
from os.path import join as pjoin
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
import time 
from data_loaders.truebones.truebones_utils.param_utils import HML_AVG_BONELEN, FOOT_CONTACT_HEIGHT_THRESH, DEFAULT_DATASET_DIR, MAX_PATH_LEN, MOTION_DIR, FOOT_CONTACT_VEL_THRESH, BVHS_DIR, OBJECT_SUBSETS_DICT, get_raw_data_dir, SNAKES, CHAIN_FORWARD_JOINTS, FLYING, FISH, VERTICAL_CLAMP_MIN_RATIO, VERTICAL_CLAMP_MAX_RATIO
from Anytop.utils.rotation_conversions import rotation_6d_to_matrix_np
from Anytop.utils.roundtrip_common import _load_fbx_skeleton_metadata
from .motion_labels import build_motion_labels, build_object_labels, write_motion_metadata
from .physics_joint_annotation import (
    _infer_end_effector_joints,
    _infer_contact_joints,
    _build_semantic_metadata,
    _rest_positions_from_offsets,
    _normalize_joint_name,
    _strip_joint_name_prefix,
    build_joint_embedding_texts,
    JOINT_NAME_EMBEDDING_SCHEMA_VERSION,
)
from .face_orientation import (
    resolve_face_joints,
    get_root_quat,
    rotate_to_hml_orientation,
    _get_facing_candidates,
    _find_forward_reference_joint,
    _find_neck_reference_joint,
    _vector_angle_deg,
)


################## Data Generation #####################

# Maximum XZ displacement (in HML-normalised units) a clip's root may travel
# before we consider it a locomotion clip and forcibly zero the root XZ.
# In-place actions (attacks, idles, jumps) drift < 1 units; walkers/runners
# travel several units per second at typical Truebones frame-rates.
ROOT_XZ_STRIP_THRESHOLD = 1

# Mean L2 distance (per joint, in HML-normalised units) between first and last
# frame poses below which a clip is classified as looping.
LOOP_DETECTION_POS_THRESHOLD = 0.10


_EMITTED_MIRROR_SAFEGUARD_WARNINGS = set()


def _canonical_name_for_bvh(name, fallback_name):
    compact_name = re.sub(r'[^0-9A-Za-z_]+', '', str(name or ''))
    if compact_name:
        return compact_name
    fallback_compact = re.sub(r'[^0-9A-Za-z_]+', '', str(fallback_name or ''))
    return fallback_compact or 'Joint'


def _build_joint_name_inspection_rows(object_cond, embedding_texts):
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


def _joint_disambiguation_tokens(raw_name, canonical_name):
    raw_value = str(raw_name or '')
    stripped_raw = _strip_joint_name_prefix(raw_value)
    raw_tokens = _normalize_joint_name(stripped_raw).split()
    canonical_tokens = _normalize_joint_name(canonical_name).split()
    residual_tokens = _remove_token_counts(raw_tokens, Counter(canonical_tokens))
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


def _disambiguate_duplicate_canonical_names(raw_names, canonical_names):
    updated_names = list(canonical_names)
    grouped_indices = defaultdict(list)
    for joint_index, canonical_name in enumerate(canonical_names):
        grouped_indices[str(canonical_name)].append(joint_index)

    for canonical_name, indices in grouped_indices.items():
        raw_name_set = {str(raw_names[index]) for index in indices}
        if len(indices) <= 1 or len(raw_name_set) <= 1:
            continue

        residual_token_lists = [
            _joint_disambiguation_tokens(raw_names[index], canonical_name)
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


def _collect_joint_name_collision_groups(cond):
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


def _write_joint_name_collision_report(cond, save_dir):
    collision_groups = _collect_joint_name_collision_groups(cond)
    report = {
        'num_objects': int(len(cond)),
        'num_collision_groups': int(len(collision_groups)),
        'collision_groups': collision_groups,
    }
    report_path = pjoin(save_dir, 'joint_name_collision_report.json')
    with open(report_path, 'w', encoding='utf-8') as report_file:
        json.dump(report, report_file, indent=2)

    if collision_groups:
        print(f'[WARN] canonical joint-name collision scan found {len(collision_groups)} group(s); report: {report_path}')
        for group in collision_groups[:20]:
            raw_names = ' | '.join(row['raw_name'] for row in group['rows'])
            print(f"  - {group['object_type']}: {group['canonical_name']} <- {raw_names}")
        if len(collision_groups) > 20:
            print(f'  ... {len(collision_groups) - 20} additional group(s) omitted from console output')
    else:
        print(f'[PASS] canonical joint-name collision scan found no duplicate canonical names; report: {report_path}')

    return collision_groups


def _refresh_joint_metadata_in_object_cond(object_cond):
    joint_names = list(object_cond.get('joints_names') or [])
    if not joint_names:
        return

    semantic_metadata = _build_semantic_metadata(
        joint_names,
        np.asarray(object_cond.get('parents'), dtype=np.int64),
        np.asarray(object_cond.get('offsets'), dtype=np.float64),
    )
    object_cond['canonical_joint_names'] = _disambiguate_duplicate_canonical_names(
        joint_names,
        semantic_metadata['canonical_joint_names'],
    )
    object_cond['canonical_bvh_joint_names'] = [
        _canonical_name_for_bvh(canonical_name, raw_name)
        for canonical_name, raw_name in zip(semantic_metadata['canonical_joint_names'], joint_names)
    ]
    object_cond['canonical_bvh_joint_names'] = [
        _canonical_name_for_bvh(canonical_name, raw_name)
        for canonical_name, raw_name in zip(object_cond['canonical_joint_names'], joint_names)
    ]
    object_cond['end_effector_joints'] = semantic_metadata['end_effector_joints']
    object_cond['end_effector_names'] = semantic_metadata['end_effector_names']
    object_cond['contact_joints'] = semantic_metadata['contact_joints']
    object_cond['contact_joint_names'] = semantic_metadata['contact_joint_names']
    object_cond['contact_joint_source'] = semantic_metadata['contact_joint_source']
    object_cond['joint_side_labels'] = semantic_metadata['joint_side_labels']
    object_cond['symmetry_partner_indices'] = semantic_metadata['symmetry_partner_indices']
    object_cond['symmetric_joint_pairs'] = semantic_metadata['symmetric_joint_pairs']
    object_cond['symmetric_joint_pair_names'] = semantic_metadata['symmetric_joint_pair_names']
    object_cond['mirror_disabled_joint_indices'] = semantic_metadata['mirror_disabled_joint_indices']
    object_cond['mirror_disabled_joint_names'] = semantic_metadata['mirror_disabled_joint_names']
    object_cond['mirror_disabled_warnings'] = semantic_metadata['mirror_disabled_warnings']
    object_cond['is_symmetric'] = semantic_metadata['is_symmetric']


def _attach_joint_name_embeddings_to_cond(cond, save_dir, t5_name='t5-base'):
    from model.conditioners import T5Conditioner

    if not cond:
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inspection_dir = pjoin(save_dir, 'joint_name_inspection')
    os.makedirs(inspection_dir, exist_ok=True)

    print(f'Encoding joint names into cond.npy with {t5_name} on {device.upper()}')
    t5_conditioner = T5Conditioner(
        name=t5_name,
        finetune=False,
        word_dropout=0.0,
        normalize_text=False,
        device=device,
        autocast_dtype=None,
        local_files_only=True,
    )

    with torch.no_grad():
        for object_type in sorted(cond):
            object_cond = cond[object_type]
            _refresh_joint_metadata_in_object_cond(object_cond)
            embedding_texts = build_joint_embedding_texts(object_cond)
            names_tokens = t5_conditioner.tokenize_entries(embedding_texts)
            embs = t5_conditioner(names_tokens).detach().cpu().numpy().astype(np.float32, copy=False)
            object_cond['joints_names_embs'] = embs
            object_cond['joints_names_embs_meta'] = {
                't5_name': t5_name,
                'schema_version': JOINT_NAME_EMBEDDING_SCHEMA_VERSION,
                'embedding_dim': int(embs.shape[1]) if embs.ndim == 2 else 0,
                'embedding_texts': list(embedding_texts),
            }

            inspection_path = pjoin(inspection_dir, f'{object_type}.json')
            with open(inspection_path, 'w', encoding='utf-8') as inspection_file:
                json.dump(_build_joint_name_inspection_rows(object_cond, embedding_texts), inspection_file, indent=2)

    _write_joint_name_collision_report(cond, save_dir)


def _detect_motion_loop(positions):
    """Return True if the last frame's root-relative pose is close to the first frame's."""
    if positions.shape[0] < 2:
        return False
    per_joint_dist = np.linalg.norm(positions[-1] - positions[0], axis=-1)
    return bool(np.mean(per_joint_dist) < LOOP_DETECTION_POS_THRESHOLD)


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


def _find_descendant_transport_chain(parents, trans_root, max_depth=2):
    chain = []
    current = trans_root
    for _depth in range(max_depth):
        children = [joint_index for joint_index, parent_index in enumerate(parents) if parent_index == current]
        if len(children) != 1:
            break
        current = children[0]
        chain.append(current)
    return chain


def _bake_descendant_y_into_translation_root(anim, max_depth=2):
    """Bake near-root locator-style Y transport back onto the translation root.

    The heuristic is intentionally narrow: follow a single chain at most two levels
    below the effective translation root, allow one dummy node in the middle, and
    bake only joints in that chain that carry animated local Y.
    """
    trans_root = _find_translation_root(anim)
    chain = _find_descendant_transport_chain(anim.parents, trans_root, max_depth=max_depth)
    bake_joints = [joint_index for joint_index in chain if np.ptp(anim.positions[:, joint_index, 1]) > 1e-4]
    if not bake_joints:
        return anim

    frozen_positions = anim.positions.copy()
    for joint_index in bake_joints:
        frozen_positions[:, joint_index, 1] = frozen_positions[0, joint_index, 1]

    frozen_anim = Animation(
        anim.rotations.copy(),
        frozen_positions,
        anim.orients.copy(),
        anim.offsets.copy(),
        anim.parents.copy(),
    )
    global_pos = positions_global(anim)
    frozen_global_pos = positions_global(frozen_anim)
    anchor_joint = chain[-1]
    delta_y = global_pos[:, anchor_joint, 1] - frozen_global_pos[:, anchor_joint, 1]
    if np.max(np.abs(delta_y)) <= 1e-8:
        return anim

    new_positions = frozen_positions.copy()
    if trans_root == 0 or anim.parents[trans_root] < 0:
        new_positions[:, trans_root, 1] += delta_y
    else:
        global_rots = rotations_global(frozen_anim)
        parent_index = anim.parents[trans_root]
        parent_global_pos = frozen_global_pos[:, parent_index]
        parent_global_rots = global_rots[:, parent_index]
        desired_global = frozen_global_pos[:, trans_root].copy()
        desired_global[:, 1] += delta_y
        new_positions[:, trans_root] = (-parent_global_rots) * (desired_global - parent_global_pos)

    return Animation(
        anim.rotations.copy(),
        new_positions,
        anim.orients.copy(),
        anim.offsets.copy(),
        anim.parents.copy(),
    )

def _get_reference_body_length(anim):
    """Estimate a character size from the processed rest pose joint span."""
    rest_positions = np.zeros_like(anim.offsets)
    for j, parent in enumerate(anim.parents):
        if parent >= 0:
            rest_positions[j] = rest_positions[parent] + anim.offsets[j]
    joint_deltas = rest_positions[:, None, :] - rest_positions[None, :, :]
    max_span = np.linalg.norm(joint_deltas, axis=-1).max()
    return max(float(max_span), 1e-8)


def _compress_positive_excursion(values, min_h, max_h):
    peak = float(values.max())
    if peak <= max_h or max_h <= min_h:
        return values, False

    scale = (max_h - min_h) / max(peak - min_h, 1e-8)
    compressed = values.copy()
    mask = compressed > min_h
    compressed[mask] = min_h + (compressed[mask] - min_h) * scale
    return compressed, True


def _compress_negative_excursion(values, min_h, max_h):
    trough = float(values.min())
    if trough >= -max_h or max_h <= min_h:
        return values, False

    scale = (max_h - min_h) / max(abs(trough) - min_h, 1e-8)
    compressed = values.copy()
    mask = compressed < -min_h
    compressed[mask] = -min_h - ((-compressed[mask]) - min_h) * scale
    return compressed, True


def _clamp_vertical_trajectory(
    processed_anim,
    object_type,
    min_ratio=VERTICAL_CLAMP_MIN_RATIO,
    max_ratio=VERTICAL_CLAMP_MAX_RATIO,
):
    """Compress only the vertical excursion that exceeds the allowed size-relative band.

    The allowed band is derived from the processed skeleton's reference body length.
    Motion within ±minH is left untouched. Only the excess beyond minH is compressed
    so that the final excursion stays within [minH, maxH].
    """
    if object_type not in FLYING and object_type not in FISH:
        return processed_anim

    trans_root = _find_translation_root(processed_anim)
    global_pos = positions_global(processed_anim)
    world_y = global_pos[:, trans_root, 1]
    body_length = _get_reference_body_length(processed_anim)
    min_h = body_length * min_ratio
    max_h = body_length * max_ratio

    if object_type in FLYING:
        clamped_world_y, changed = _compress_positive_excursion(world_y, min_h, max_h)
    else:
        clamped_world_y = world_y.copy()
        clamped_world_y, changed_pos = _compress_positive_excursion(clamped_world_y, min_h, max_h)
        clamped_world_y, changed_neg = _compress_negative_excursion(clamped_world_y, min_h, max_h)
        changed = changed_pos or changed_neg

    if not changed:
        return processed_anim

    new_positions = processed_anim.positions.copy()
    if trans_root == 0 or processed_anim.parents[trans_root] < 0:
        new_positions[:, trans_root, 1] += clamped_world_y - world_y
    else:
        global_rots = rotations_global(processed_anim)
        parent_index = processed_anim.parents[trans_root]
        parent_global_pos = global_pos[:, parent_index]
        parent_global_rots = global_rots[:, parent_index]
        desired_global = global_pos[:, trans_root].copy()
        desired_global[:, 1] = clamped_world_y
        new_positions[:, trans_root] = (-parent_global_rots) * (desired_global - parent_global_pos)

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


def _xz_locomotion_extent(anim, translation_root_index):
    """Return the clip-wide XZ span of the translation root.

    Using only the first/last-frame delta misses clips that lunge far away and
    then return near their starting point by the end of the slice. Those clips
    still produce visible root drift in exported processed BVHs, so we classify
    locomotion from the full XZ coverage of the trajectory instead.
    """
    global_pos = positions_global(anim)
    root_xz = global_pos[:, translation_root_index, [0, 2]]
    xz_span = np.ptp(root_xz, axis=0)
    return float(np.linalg.norm(xz_span))


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


def needs_bvh_position_channels(anim, tol=1e-4):
    """Return True when BVH export must write non-root position channels.

    This repo's internal FK uses ``anim.positions`` directly as each joint's local
    translation. Exporting with ``positions=False`` makes BVH viewers reconstruct
    every non-root joint from the static rest ``offsets`` instead, so any non-root
    local position that differs from those offsets must be written explicitly.
    """
    if anim.positions.shape[1] <= 1:
        return False

    nonroot_positions = np.asarray(anim.positions[:, 1:, :], dtype=np.float64)
    rest_offsets = np.asarray(anim.offsets[1:], dtype=np.float64)[None, :, :]
    return bool(np.any(np.abs(nonroot_positions - rest_offsets) > tol))

def scale(anim, scale_factor=None):
    if scale_factor is None:
        lengths = offset_lengths(anim)
        mean_len = statistics.mean(lengths)
        scale_factor = HML_AVG_BONELEN / mean_len
    new_anim = Animation(
        anim.rotations.copy(),
        anim.positions * scale_factor,
        anim.orients.copy(),
        anim.offsets * scale_factor,
        anim.parents.copy(),
    )
    return new_anim, scale_factor

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

"""" process anim object """
def process_anim(anim, object_type, root_pose_init_xz=None, scale_factor=None, face_joints=None, orientation_quat=None, forward_joint_index=None, forward_base_joint_index=None):
    rotated = rotate_to_hml_orientation(anim, object_type, face_joints, orientation_quat=orientation_quat, forward_joint_index=forward_joint_index, forward_base_joint_index=forward_base_joint_index)
    baked = _bake_descendant_y_into_translation_root(rotated)
    centered, root_pose_init_xz_ = move_xz_to_origin(baked, root_pose_init_xz)
    scaled, scale_factor_ = scale(centered, scale_factor)
    return scaled, root_pose_init_xz_, scale_factor_


def _reference_clip_needs_local_position_rebuild(anim, tol=1e-4):
    """Return True when the reference clip's first frame is not already rest-offset-aligned."""
    if len(anim) == 0 or anim.positions.shape[1] <= 1:
        return False

    root_candidates = np.where(np.asarray(anim.parents) < 0)[0]
    if root_candidates.size == 0:
        return False

    nonroot_indices = np.delete(np.arange(anim.positions.shape[1]), int(root_candidates[0]))
    if nonroot_indices.size == 0:
        return False

    local_positions = np.asarray(anim.positions[0, nonroot_indices], dtype=np.float64)
    rest_offsets = np.asarray(anim.offsets[nonroot_indices], dtype=np.float64)
    return bool(np.max(np.abs(local_positions - rest_offsets)) > tol)

""" get object_type common characteristics, extracted from T-pose FBX"""
def get_common_features_from_T_pose(t_pose_fbx_path, object_type, face_joints=None):
    _t0 = time.time()
    t_pose_anim, t_pos_names, _t_pose_frame_time = FBX.load(t_pose_fbx_path)
    reference_anim = t_pose_anim[:1] if len(t_pose_anim) > 1 else t_pose_anim
    face_joints = resolve_face_joints(object_type, t_pos_names, reference_anim.parents, face_joints=face_joints)
    forward_joint_index = _find_forward_reference_joint(t_pos_names, reference_anim.parents)
    forward_base_joint_index = _find_neck_reference_joint(t_pos_names, reference_anim.parents)

    # This function only consumes reference-pose metadata from frame 0, so avoid
    # repairing every frame of long T-pose clips unless the first frame is malformed.
    if _reference_clip_needs_local_position_rebuild(reference_anim):
        reference_positions = positions_global(reference_anim)
        with open(os.devnull, 'w') as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
            reference_anim, _1, _2 = animation_from_positions(
                positions=reference_positions,
                parents=reference_anim.parents,
                offsets=reference_anim.offsets,
                iterations=100,
                silent=True,
            )

    reference_positions = positions_global(reference_anim)
    t_pose_orientation_quat = get_root_quat(reference_positions, object_type, face_joint_indx=face_joints, forward_joint_index=forward_joint_index, forward_base_joint_index=forward_base_joint_index)[0]
    scaled, root_pose_init_xz, scale_factor = process_anim(
        reference_anim,
        object_type,
        face_joints=face_joints,
        orientation_quat=t_pose_orientation_quat,
        forward_joint_index=forward_joint_index,
        forward_base_joint_index=forward_base_joint_index,
    )
    scaled_positions = positions_global(scaled)
    scaled_rest_positions = scaled_positions[0]
    offsets = offsets_from_positions(scaled_rest_positions, scaled.parents)
    suspected_foot_indices, contact_joint_source = _infer_contact_joints(
        t_pos_names,
        scaled.parents,
        scaled_rest_positions,
    )
    print(f'[TIME] {object_type}: get_common_features_from_T_pose = {time.time() - _t0:.2f}s')
    return root_pose_init_xz, scale_factor, offsets, suspected_foot_indices, scaled.rotations, t_pos_names, scaled, face_joints, t_pose_orientation_quat, forward_joint_index, forward_base_joint_index, contact_joint_source

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


def infer_translation_root_from_features(data, tol=1e-5):
    """Infer which joint row carries the effective translation-root trajectory.

    The feature tensor stores root-facing rotation on joint row 0, but the XZ
    trajectory itself lives on the effective translation root row. For rigs like
    Horse, this is an intermediate joint (e.g. Bip01), whose RIFKE XZ is exactly
    zero because all joints are expressed relative to it.
    """
    motion = np.asarray(data, dtype=np.float64)
    if motion.ndim != 3:
        raise ValueError(f"Expected motion features with shape (F, J, C), got {motion.shape}.")

    xz_abs_max = np.max(np.abs(motion[:, :, [0, 2]]), axis=(0, 2))
    zero_xz_candidates = np.flatnonzero(xz_abs_max <= tol)
    if zero_xz_candidates.size > 0:
        return int(zero_xz_candidates[0])

    return int(np.argmin(xz_abs_max))


def _warn_mirror_disabled_subtrees(object_cond):
    disabled_joint_indices = tuple(int(index) for index in object_cond['mirror_disabled_joint_indices'])
    if not disabled_joint_indices:
        return

    object_type = str(object_cond['object_type'])
    warning_key = (object_type, disabled_joint_indices)
    if warning_key in _EMITTED_MIRROR_SAFEGUARD_WARNINGS:
        return

    _EMITTED_MIRROR_SAFEGUARD_WARNINGS.add(warning_key)
    warning_messages = list(object_cond['mirror_disabled_warnings'])
    if not warning_messages:
        names = ', '.join(str(name) for name in object_cond['mirror_disabled_joint_names'])
        warning_messages = [f'unpaired mirrored joints [{names}] will be neutralized during mirror augmentation.']

    for message in warning_messages:
        print(f'[WARN] {object_type}: {message}')


def _neutralize_mirror_disabled_subtrees(features, object_cond, mirrored_offsets):
    disabled_joint_indices = sorted({
        int(index)
        for index in object_cond['mirror_disabled_joint_indices']
        if int(index) > 0
    })
    if not disabled_joint_indices:
        return np.asarray(features).copy()

    motion = np.asarray(features, dtype=np.float32)
    parents = np.asarray(object_cond['parents'], dtype=np.int64)
    offsets = np.asarray(mirrored_offsets, dtype=np.float64)
    anim, _has_animated_pos = recover_animation_from_motion_np(motion, parents, offsets)
    if anim is None:
        return motion.copy()

    neutral_positions = anim.positions.copy()
    neutral_rotations = anim.rotations.copy()
    neutral_positions[:, disabled_joint_indices] = offsets[disabled_joint_indices][None, :, :]
    neutral_rotations[:, disabled_joint_indices] = Quaternions.id((motion.shape[0], len(disabled_joint_indices)))

    neutral_anim = Animation(
        neutral_rotations,
        neutral_positions,
        anim.orients.copy(),
        offsets.copy(),
        parents.copy(),
    )
    translation_root_index = _find_translation_root(neutral_anim)
    contact_joint_indices = list(object_cond['contact_joints'])
    face_joints = list(object_cond['face_joints']) or None

    cont_6d_params, _r_velocity, _velocity, r_rot, global_positions = get_bvh_cont6d_params(
        neutral_anim,
        str(object_cond['object_type']),
        face_joints=face_joints,
        joint_names=list(object_cond['joints_names']),
        translation_root_index=translation_root_index,
    )
    positions = get_rifke(global_positions, r_rot, translation_root_index=translation_root_index)
    is_loop = _detect_motion_loop(positions)
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
    return neutralized.astype(motion.dtype, copy=False)


def mirror_features_with_safeguards(features, object_cond):
    spi = np.asarray(object_cond['symmetry_partner_indices'], dtype=np.int64)
    perm = np.arange(len(spi), dtype=np.int64)
    valid = spi >= 0
    perm[valid] = spi[valid]

    mirrored = np.asarray(features)[perm].copy() if np.asarray(features).ndim == 2 else np.asarray(features)[:, perm, :].copy()
    mirrored[..., [0, 4, 5, 6, 9]] *= -1

    mirrored_offsets = np.asarray(object_cond['offsets'], dtype=np.float32)[perm].copy()
    mirrored_offsets[:, 0] *= -1

    if object_cond['mirror_disabled_joint_indices']:
        _warn_mirror_disabled_subtrees(object_cond)
        mirrored = _neutralize_mirror_disabled_subtrees(mirrored, object_cond, mirrored_offsets)

    return mirrored, mirrored_offsets


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

""" compute new rotations for anim that are relative to a natural tpose """
def compute_rots_from_tpos(tpos_quats, dest_quats, parents):
    new_rots = dest_quats.copy()
    new_rots[:, 0] = new_rots[:, 0] * -tpos_quats[:, 0]
    cum_rots = tpos_quats.copy()
    for j, p in enumerate(parents[1:], start=1):
        cum_rots[:, j] = cum_rots[:, p] * tpos_quats[:, j]
        new_rots[:, j] = cum_rots[:, p] * dest_quats[:, j] * -tpos_quats[:, j] * -cum_rots[:, p]
    return new_rots


def _solve_local_positions_for_target_global(
    rotations,
    target_global_positions,
    offsets,
    parents,
    orients,
    initial_positions=None,
    position_match_threshold=1e-5,
    max_passes=2,
):
    frames_num = target_global_positions.shape[0]
    if initial_positions is None:
        local_positions = offsets.copy()[None, :].repeat(frames_num, axis=0)
    else:
        local_positions = initial_positions.copy()

    for _ in range(max_passes):
        temp_anim = Animation(rotations, local_positions, orients, offsets, parents)
        temp_global_pos = positions_global(temp_anim)
        per_joint_err = np.max(np.abs(target_global_positions - temp_global_pos), axis=(0, 2))
        joints_to_fix = [
            joint_idx
            for joint_idx in range(len(parents))
            if per_joint_err[joint_idx] > position_match_threshold
        ]
        if not joints_to_fix:
            break

        for joint_idx in joints_to_fix:
            if joint_idx == 0 or parents[joint_idx] < 0:
                local_positions[:, joint_idx] = target_global_positions[:, joint_idx]
                continue

            temp_anim = Animation(rotations, local_positions, orients, offsets, parents)
            temp_global_rots = rotations_global(temp_anim)
            temp_global_pos = positions_global(temp_anim)
            parent_idx = parents[joint_idx]
            local_positions[:, joint_idx] = (
                -temp_global_rots[:, parent_idx]
            ) * (target_global_positions[:, joint_idx] - temp_global_pos[:, parent_idx])

    return local_positions


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
    processed_positions = _solve_local_positions_for_target_global(
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
            raw_anim, names, frame_time = FBX.load(bvh_path)
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
    processed_global_pos = positions_global(processed_anim)
    anim_positions = _solve_local_positions_for_target_global(
        rots,
        processed_global_pos,
        offsets,
        processed_anim.parents,
        processed_anim.orients,
        initial_positions=anim_positions,
    )
    # create animation object which is defined over correct tpos
    new_anim = Animation(rots, anim_positions  , processed_anim.orients, offsets, processed_anim.parents)

    processed_global_pos = positions_global(processed_anim)
    new_global_pos = positions_global(new_anim)
    squared_error = np.mean((processed_global_pos - new_global_pos) ** 2)
    error_key = bvh_path if isinstance(bvh_path, str) else '__animation__'
    if slice_inds is not None and not isinstance(bvh_path, Animation):
        error_key = f'{bvh_path}[{slice_inds[0]}:{slice_inds[1]}]'
    squared_positions_error[error_key] = float(squared_error)

    return new_anim, processed_anim, names  
    
""" get motion feature representation"""
def get_motion(bvh_path, foot_contact_vel_thresh, object_type, max_joints, root_pose_init_xz, scale_factor, offsets, foot_indices, tpos_rots, squared_positions_error, face_joints=None, orientation_quat=None, forward_joint_index=None, forward_base_joint_index=None, slice_inds=None, preloaded=None):
    try:
        new_anim, export_anim, names = get_hml_aligned_anim(
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
        export_translation_root_index = _find_translation_root(export_anim)
        xz_extent = _xz_locomotion_extent(export_anim, export_translation_root_index)
        has_locomotion = xz_extent > ROOT_XZ_STRIP_THRESHOLD
        if has_locomotion:
            new_anim = strip_translation_root_xz(new_anim, translation_root_index)
            export_anim = strip_translation_root_xz(export_anim, export_translation_root_index)
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
        foot_contact = get_contact_state(global_positions, foot_indices, foot_contact_vel_thresh)
        '''Get Joint Rotation Invariant Position Represention'''
        # local velocity wrt root coords system as described in get_rifke definition 
        positions = get_rifke(global_positions, r_rot, translation_root_index=translation_root_index)
        is_loop = _detect_motion_loop(positions)
        # root_y = positions[:, 0, 1:2]
        # r_velocity = np.arcsin(r_velocity[:, 2:3])
        # l_velocity = velocity[:, [0, 2]]
        local_vel = np.repeat(r_rot[1:, None], global_positions.shape[1], axis=1) * (global_positions[1:] - global_positions[:-1])
        prev_velocity = local_vel[-1] if local_vel.shape[0] > 0 else None
        terminal_local_vel = _compute_terminal_local_velocity(global_positions, r_rot, is_loop, prev_frame_velocity=prev_velocity)
        # For locomotion clips the root XZ position has already been zeroed by
        # strip_translation_root_xz, so the velocity must be zeroed too to stay
        # consistent with RIFKE.  For in-place clips we keep the XZ velocity so
        # the representation faithfully captures small positional shifts.
        if has_locomotion:
            local_vel[:, translation_root_index, [0, 2]] = 0.0
            terminal_local_vel[translation_root_index, [0, 2]] = 0.0
        terminal_contact = get_terminal_contact_state(
            global_positions,
            foot_indices,
            foot_contact_vel_thresh,
            is_loop,
        )
        # root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)
        features, max_joints = get_motion_features(
            positions,
            cont_6d_params,
            foot_contact,
            local_vel,
            terminal_local_vel,
            terminal_contact,
            max_joints,
        )
        return features, new_anim.parents, max_joints, new_anim, export_anim, is_loop
    except Exception as err:
        print(err)
        return None, None, max_joints, None, None, False

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


_IDLE_REFERENCE_TAIL_PATTERN = re.compile(
    r'^idle(?:\d+)?(?:loop|cyc|cycle|repeat|repeating)?$'
)
_WALK_REFERENCE_TAIL_PATTERN = re.compile(
    r'^walk(?:ing)?(?:\d+)?(?:loop|cyc|cycle|repeat|repeating|forward|forwards)?$'
)


def _reference_tail_candidates(file_path):
    stem = os.path.splitext(os.path.basename(file_path))[0].lower()
    segments = [segment for segment in re.split(r'[^a-z0-9]+', stem) if segment]
    return [''.join(segments[index:]) for index in range(len(segments))]


def _matches_reference_tail(file_path, tail_pattern):
    return any(tail_pattern.fullmatch(candidate) for candidate in _reference_tail_candidates(file_path))


def _is_tpose_reference_path(file_path):
    tokens, compact = _reference_stem_tokens(file_path)
    return (
        'tpose' in compact
        or 'tpos' in compact
        or 'bindpose' in compact
        or 'restpose' in compact
        or 'nosaddle' in compact
        or ('pose' in tokens and 't' in tokens)
    )


def _is_idle_reference_path(file_path):
    return _matches_reference_tail(file_path, _IDLE_REFERENCE_TAIL_PATTERN)


def _is_walk_reference_path(file_path):
    return _matches_reference_tail(file_path, _WALK_REFERENCE_TAIL_PATTERN)


""" find a character-level orientation reference clip with priority T Pose > Idle > Walk """
def find_tpose_reference_path(fbx_files):
    for file_path in fbx_files:
        if _is_tpose_reference_path(file_path):
            fbx_files.remove(file_path)
            return file_path

    for matcher, _source_name in (
        (_is_idle_reference_path, 'idle'),
        (_is_walk_reference_path, 'walk'),
    ):
        for file_path in fbx_files:
            if matcher(file_path):
                return file_path

    return fbx_files[0]


def _normalize_action_name(object_type: str, raw_action: str) -> str:
    """Normalize an action name extracted from an FBX filename.

    Steps:
    1. Strip ``{object_type}ALL`` / ``{Species}All`` **prefixed across a
       separator** so that ``HorseALL-RunToStop`` → ``RunToStop``.
    2. Strip ``{object_type}`` **prefixed across a separator** so that
       ``Hound-Attack`` → ``Attack``, ``Crab-Walk`` → ``Walk``.
    3. Convert all-lowercase or space-separated action names to CamelCase
       so that ``atk 1`` → ``Atk1``,  ``down loop`` → ``DownLoop``.
    """
    obj_lower = object_type.lower()
    if not raw_action:
        return raw_action

    # Step 1 — strip {species}ALL{sep} (e.g. HorseALL-RunToStop → RunToStop)
    all_prefix = re.compile(
        rf'^{re.escape(obj_lower)}all[-_\s]', re.IGNORECASE
    )
    raw_action = all_prefix.sub('', raw_action)

    if not raw_action:
        return raw_action

    # Step 2 — strip {species}{sep} (e.g. Hound-Attack → Attack)
    species_prefix = re.compile(
        rf'^{re.escape(obj_lower)}[-_\s]', re.IGNORECASE
    )
    raw_action = species_prefix.sub('', raw_action)

    if not raw_action:
        return raw_action

    # Step 3 — CamelCase for all-lowercase or space-separated action names
    # that start with a lowercase letter (indicating a raw action description).
    # Well-formed names like ``Back Away`` (starts with uppercase) are
    # left untouched.
    has_spaces = ' ' in raw_action
    is_all_lowercase = raw_action.islower()
    starts_with_lower = raw_action[0].islower() if raw_action else False

    if (has_spaces and starts_with_lower) or is_all_lowercase:
        parts = re.split(r'[^a-zA-Z0-9]+', raw_action)
        parts = [p for p in parts if p]
        if not parts:
            return raw_action
        return ''.join(p[0].upper() + p[1:] for p in parts)

    return raw_action


def _should_skip_fbx(file_path: str, object_type: str) -> bool:
    """Check whether an FBX file should be skipped during preprocessing.

    Skips:
    1. **All-in-one files** that bundle every animation clip into a single file
       (e.g. ``CrabAll.fbx``, ``HorseALL.fbx``, ``Camel_ALL.fbx``, ``Cat-ALL.fbx``).
    2. **Files without an inferable action name**:
       - Standalone species-name files (``Fox.fbx``, ``Monkey.fbx``, …)
       - Variant-code files (``FoxA_A02.fbx``, ``Monkey_B01.fbx``, …)
    """
    stem = os.path.splitext(os.path.basename(file_path))[0]
    stem_lower = stem.lower()
    obj_lower = object_type.lower()

    # ── 1. All-in-one files ──────────────────────────────────────────────
    for sep in ('', '_', '-'):
        pattern = re.compile(
            rf'^{re.escape(obj_lower)}{re.escape(sep)}all$', re.IGNORECASE
        )
        if pattern.match(stem):
            print(
                f'  [SKIP] {os.path.basename(file_path)}: '
                f'all-in-one file (contains all animations)'
            )
            return True

    # ── 2. NoSaddle variants (T-pose with no animation) ─────────────────
    nosaddle_pattern = re.compile(
        rf'^{re.escape(obj_lower)}[-_]\s*nosaddle$', re.IGNORECASE
    )
    if nosaddle_pattern.match(stem_lower):
        print(
            f'  [SKIP] {os.path.basename(file_path)}: '
            f'NoSaddle T-pose file (no animation)'
        )
        return True

    # ── 3. Standalone species name (no action component) ─────────────────
    if stem_lower == obj_lower:
        print(
            f'  [SKIP] {os.path.basename(file_path)}: '
            f'no inferable action name (species name only)'
        )
        return True

    # ── 3. Variant codenames ─────────────────────────────────────────────
    #   FoxA_A02, FoxA_A03     → {species}{letter}_{code}
    #   Monkey_B01, Monkey_B02 → {species}_{letter}{digits}
    variant1 = re.compile(
        rf'^{re.escape(obj_lower)}[a-z]_\w+$', re.IGNORECASE
    )
    variant2 = re.compile(
        rf'^{re.escape(obj_lower)}_[a-z]\d+$', re.IGNORECASE
    )
    if variant1.match(stem) or variant2.match(stem):
        print(
            f'  [SKIP] {os.path.basename(file_path)}: '
            f'variant codename, no inferable action name'
        )
        return True

    return False


def _process_motion_file(file_path, object_type, max_joints, root_pose_init_xz, scale_factor,
                         offsets, foot_indices, tpos_rots, face_joints, orientation_quat, forward_joint_index, forward_base_joint_index):
    local_errors = dict()
    # Load the FBX file once; pass it as `preloaded` to every get_motion call so that
    raw_anim, names, frame_time = FBX.load(file_path)
    anim_len = len(raw_anim)
    begin = 0
    file_max_joints = max_joints
    file_results = []

    while begin < anim_len:
        if anim_len - begin > 240:
            slice_ind = begin + 200
        else:
            slice_ind = anim_len

        motion, parents, file_max_joints, new_anim, export_anim, is_loop = get_motion(
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

        orientation_summary = None
        try:
            orientation_summary = _summarize_processed_orientation(
                export_anim,
                object_type,
                face_joints,
                forward_joint_index,
                forward_base_joint_index,
            )
        except Exception as exc:
            print(f'  [WARN] failed to summarize processed orientation for {file_path} [{current_begin}:{slice_ind}]: {exc}')

        _, file_name = os.path.split(file_path)
        raw_action = file_name.split('.')[0]
        raw_action = _normalize_action_name(object_type, raw_action)
        file_results.append({
            'action': raw_action,
            'motion': motion,
            'parents': parents,
            'new_anim': new_anim,
            'export_anim': export_anim,
            'names': names,
            'frame_time': frame_time,
            'is_loop': is_loop,
            'source_fbx_path': file_path,
            'slice_range': (current_begin, slice_ind),
            'orientation_summary': orientation_summary,
            'motion_labels': build_motion_labels(object_type, raw_action),
        })

    return {
        'errors': local_errors,
        'max_joints': file_max_joints,
        'results': file_results,
    }


def _attach_orientation_reference_metadata(
    object_cond,
    orientation_quat,
    forward_joint_index,
    forward_base_joint_index,
    orientation_reference_fbx_path,
):
    orientation_qs = getattr(orientation_quat, 'qs', orientation_quat)
    orientation_qs = np.asarray(orientation_qs, dtype=np.float64)
    if orientation_qs.ndim > 1:
        orientation_qs = orientation_qs[0]
    object_cond['orientation_quat'] = orientation_qs.reshape(4)
    object_cond['forward_joint_index'] = int(forward_joint_index) if forward_joint_index is not None else None
    object_cond['forward_base_joint_index'] = int(forward_base_joint_index) if forward_base_joint_index is not None else None
    object_cond['orientation_reference_fbx_path'] = (
        os.path.abspath(orientation_reference_fbx_path)
        if orientation_reference_fbx_path
        else None
    )


def _summarize_frame_orientation(frame_positions, object_type, face_joints, forward_joint_index, forward_base_joint_index):
    raw_candidates = _get_facing_candidates(
        frame_positions,
        object_type,
        face_joint_indx=face_joints,
        forward_joint_index=forward_joint_index,
        forward_base_joint_index=forward_base_joint_index,
    )
    target_forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    candidate_angles = {
        name: _vector_angle_deg(forward[0], target_forward)
        for name, forward in raw_candidates.items()
        if forward is not None and np.isfinite(forward).all()
    }
    if not candidate_angles:
        raise ValueError(f'{object_type} produced no valid orientation candidates')
    best_candidate, best_angle_deg = min(candidate_angles.items(), key=lambda item: item[1])
    return best_candidate, float(best_angle_deg)


def _summarize_processed_orientation(export_anim, object_type, face_joints, forward_joint_index, forward_base_joint_index):
    global_positions = positions_global(export_anim)
    if global_positions.shape[0] <= 0:
        raise ValueError(f'{object_type} processed animation has no frames for orientation summary')

    first_candidate, first_angle_deg = _summarize_frame_orientation(
        global_positions[0:1],
        object_type,
        face_joints,
        forward_joint_index,
        forward_base_joint_index,
    )
    if global_positions.shape[0] == 1:
        last_candidate, last_angle_deg = first_candidate, first_angle_deg
    else:
        last_candidate, last_angle_deg = _summarize_frame_orientation(
            global_positions[-1:],
            object_type,
            face_joints,
            forward_joint_index,
            forward_base_joint_index,
        )

    return {
        'orientation_first_frame_best_candidate': first_candidate,
        'orientation_first_frame_best_angle_deg': float(first_angle_deg),
        'orientation_last_frame_best_candidate': last_candidate,
        'orientation_last_frame_best_angle_deg': float(last_angle_deg),
    }


def _build_motion_metadata_entry(result, motion_file_name):
    motion_labels = dict(result['motion_labels'])
    motion_labels['motion_name'] = motion_file_name
    motion_labels['is_loop'] = result.get('is_loop', False)

    source_fbx_path = result.get('source_fbx_path')
    if source_fbx_path:
        motion_labels['source_fbx_path'] = os.path.abspath(source_fbx_path)

    source_frame_range = result.get('slice_range')
    if source_frame_range is not None:
        motion_labels['source_frame_range'] = [
            int(source_frame_range[0]),
            int(source_frame_range[1]),
        ]

    orientation_summary = dict(result.get('orientation_summary') or {})
    for key in (
        'orientation_first_frame_best_candidate',
        'orientation_first_frame_best_angle_deg',
        'orientation_last_frame_best_candidate',
        'orientation_last_frame_best_angle_deg',
    ):
        if key in orientation_summary:
            motion_labels[key] = orientation_summary[key]

    return motion_labels
     
"""Prepare processed tensors for all the files of a given object without writing them to disk yet."""
def _prepare_object_outputs(object_type, max_joints, face_joints=None, fbxs_dir=None, t_pos_path=None, max_files=None, num_workers=1, raw_data_dir=None):
    object_cond = dict()
    if fbxs_dir is None:
        fbxs_dir = pjoin(get_raw_data_dir(raw_data_dir), object_type)
    if not os.path.isdir(fbxs_dir):
        print(f'skipping {object_type}: raw FBX directory not found at {fbxs_dir}')
        return None
    fbx_files = sorted([pjoin(fbxs_dir, f) for f in os.listdir(fbxs_dir) if f.lower().endswith('.fbx')])
    if len(fbx_files) == 0:
        print(f'skipping {object_type}: no FBX files found in {fbxs_dir}')
        return None
    ## get a character-level orientation reference clip
    if t_pos_path is None or t_pos_path == '':
        t_pos_path = find_tpose_reference_path(fbx_files)
    else:
        # removes T-pose FBX from fbx_files, as it represents a static pose and should be used only for
        # extracting common characteristics. If this is not the case, disable this part
        fbx_files.remove(t_pos_path)
    if max_files is not None:
        fbx_files = fbx_files[:max_files]

    # Filter out files with no inferable action name or all-in-one animation bundles
    fbx_files = [f for f in fbx_files if not _should_skip_fbx(f, object_type)]
    if len(fbx_files) == 0:
        print(f'skipping {object_type}: no valid FBX files after filtering')
        return None

    squared_positions_error = dict()
    root_pose_init_xz, scale_factor, offsets, foot_indices, tpos_rots, names, tpos_anim, face_joints, orientation_quat, forward_joint_index, forward_base_joint_index, contact_joint_source = get_common_features_from_T_pose(t_pos_path, object_type, face_joints=face_joints)
    t_pos_motion, parents, max_joints, new_anim, _export_anim, _tpos_is_loop = get_motion(tpos_anim, FOOT_CONTACT_VEL_THRESH, object_type, max_joints, root_pose_init_xz, scale_factor, offsets, foot_indices, tpos_rots, squared_positions_error, face_joints=face_joints, orientation_quat=orientation_quat, forward_joint_index=forward_joint_index, forward_base_joint_index=forward_base_joint_index)
    rest_positions = _rest_positions_from_offsets(offsets, parents)
    semantic_metadata = _build_semantic_metadata(names, parents, offsets, rest_positions=rest_positions)
    object_cond['tpos_first_frame'] = t_pos_motion[0]
    # create topology conditions
    joint_relations, joints_graph_dist = create_topology_edge_relations(tpos_anim.parents, max_path_len = MAX_PATH_LEN)
    object_cond['joint_relations'] = joint_relations
    object_cond['joints_graph_dist'] = joints_graph_dist
    object_cond['object_type'] = object_type
    object_cond['parents'] = parents
    object_cond['offsets'] = offsets
    object_cond['joints_names'] = names
    object_cond['canonical_joint_names'] = semantic_metadata['canonical_joint_names']
    object_cond['canonical_bvh_joint_names'] = [
        _canonical_name_for_bvh(canonical_name, raw_name)
        for canonical_name, raw_name in zip(semantic_metadata['canonical_joint_names'], names)
    ]
    object_cond['face_joints'] = list(face_joints)
    object_cond['face_joint_names'] = [names[index] for index in face_joints]
    _attach_orientation_reference_metadata(
        object_cond,
        orientation_quat,
        forward_joint_index,
        forward_base_joint_index,
        t_pos_path,
    )
    object_cond['end_effector_joints'] = semantic_metadata['end_effector_joints']
    object_cond['end_effector_names'] = semantic_metadata['end_effector_names']
    object_cond['contact_joints'] = semantic_metadata['contact_joints']
    object_cond['contact_joint_names'] = semantic_metadata['contact_joint_names']
    object_cond['contact_joint_source'] = semantic_metadata['contact_joint_source']
    object_cond['joint_side_labels'] = semantic_metadata['joint_side_labels']
    object_cond['symmetry_partner_indices'] = semantic_metadata['symmetry_partner_indices']
    object_cond['symmetric_joint_pairs'] = semantic_metadata['symmetric_joint_pairs']
    object_cond['symmetric_joint_pair_names'] = semantic_metadata['symmetric_joint_pair_names']
    object_cond['mirror_disabled_joint_indices'] = semantic_metadata['mirror_disabled_joint_indices']
    object_cond['mirror_disabled_joint_names'] = semantic_metadata['mirror_disabled_joint_names']
    object_cond['mirror_disabled_warnings'] = semantic_metadata['mirror_disabled_warnings']
    object_cond['is_symmetric'] = semantic_metadata['is_symmetric']
    object_cond['root_pose_init_xz'] = np.array(root_pose_init_xz, dtype=np.float64)
    object_cond['scale_factor'] = float(scale_factor)
    object_cond['kinematic_chains'] = parents2kinchains(parents, object_policy(object_type))
    object_cond.update(build_object_labels(object_type))
    all_tensors = list()

    # FBX loading via bpy is single-threaded (clear_scene is a global side effect),
    # so file-level parallelism is disabled regardless of the num_workers setting.
    print(f'processing {len(fbx_files)} FBX files for {object_type} (serial — bpy is single-threaded)', flush=True)

    def process_file(file_path):
        print("processing file: " + file_path, flush=True)
        return _process_motion_file(
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

    file_outputs = [process_file(file_path) for file_path in fbx_files]

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
            result['canonical_names'] = list(object_cond['canonical_bvh_joint_names'])
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
    import time as _time_mod
    object_type = object_payload['object_type']
    frames_counter = 0
    motion_metadata = {}
    _t0 = _time_mod.time()

    for result in object_payload['results']:
        motion = result['motion']
        files_counter += 1
        frames_counter += motion.shape[0]
        name = object_type + "_" + result['action'] + "_" + str(files_counter)
        motion_file_name = name + '.npy'
        np.save(pjoin(save_dir, MOTION_DIR, motion_file_name), motion)
        # Export the visually faithful processed animation rather than the
        # T-pose-reparameterized training animation. The latter preserves global
        # positions under this repo's FK but can look distorted in external BVH
        # viewers because its local position/offset decomposition is training-oriented.
        anim_obj = result['export_anim']
        BVH.save(
            pjoin(save_dir, BVHS_DIR, name + '.bvh'),
            anim_obj,
            result.get('canonical_names', result['names']),
            frametime=result.get('frame_time', 1.0 / 24.0),
            positions=needs_bvh_position_channels(anim_obj),
            all_joints_as_names=True,
        )

        motion_labels = _build_motion_metadata_entry(result, motion_file_name)
        motion_metadata[motion_file_name] = motion_labels

    print(f'[TIME] {object_type}: _write_object_outputs = {_time_mod.time() - _t0:.2f}s ({files_counter} clips)')
    return files_counter, frames_counter, motion_metadata


def _write_dataset_artifacts(save_dir, cond, motion_metadata, objects_counter, max_joints, files_counter, frames_counter, squared_positions_error):
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
    n = error_file.write('Position squared error per source clip:')
    for f in squared_positions_error.keys():
        error_file.write('%s: %f\n' %(f, squared_positions_error[f]))
    error_file.close()

    _attach_joint_name_embeddings_to_cond(cond, save_dir)
    np.save(pjoin(save_dir, "cond.npy"), cond)
    write_motion_metadata(save_dir, motion_metadata, files_counter)

def _resolve_preprocessing_workers(objects, object_workers=8, file_workers=8):
    object_count = max(1, len(objects))
    object_workers = min(object_count, max(1, int(object_workers)))
    file_workers = max(1, int(file_workers))
    total_workers = object_workers * file_workers
    return total_workers, object_workers, file_workers


def _prepare_object_outputs_worker(object_type, max_files, file_workers, raw_data_dir=None):
    return _prepare_object_outputs(
        object_type,
        max_joints=23,
        max_files=max_files,
        num_workers=file_workers,
        raw_data_dir=raw_data_dir,
    )

""" creates processed tensors for all the files of a given object. Returens statistics and the object condition,
which includes tpos, relation/distances matrices, offsets, parents, joints names, kinematic chains, mean and std"""    
def process_object(object_type, files_counter, frames_counter, max_joints, squared_positions_error, save_dir = DEFAULT_DATASET_DIR, face_joints=None, fbxs_dir=None, t_pos_path=None, max_files=None, num_workers=1, raw_data_dir=None, bvhs_dir=None):
    object_payload = _prepare_object_outputs(
        object_type,
        max_joints,
        face_joints=face_joints,
        fbxs_dir=fbxs_dir or bvhs_dir,  # bvhs_dir kept for backward compatibility
        t_pos_path=t_pos_path,
        max_files=max_files,
        num_workers=num_workers,
        raw_data_dir=raw_data_dir,
    )
    if object_payload is None:
        return files_counter, frames_counter, max_joints, None, {}

    squared_positions_error.update(object_payload['errors'])
    max_joints = max(max_joints, object_payload['max_joints'])
    files_counter, object_frames_counter, object_motion_metadata = _write_object_outputs(
        save_dir,
        object_payload,
        files_counter,
    )
    frames_counter += object_frames_counter

    return files_counter, frames_counter, max_joints, object_payload['object_cond'], object_motion_metadata

""" create dataset """
def create_data_samples(objects=None, max_files_per_object=None, dataset_dir=None, raw_data_dir=None, object_workers=8, file_workers=8):
    ## prepare
    target_dataset_dir = dataset_dir or DEFAULT_DATASET_DIR
    os.makedirs(pjoin(target_dataset_dir, MOTION_DIR), exist_ok=True)
    os.makedirs(pjoin(target_dataset_dir, BVHS_DIR), exist_ok=True)
    
    ## process
    if objects is None:
        resolved_raw_data_dir = get_raw_data_dir(raw_data_dir)
        objects = sorted(
            obj for obj in os.listdir(resolved_raw_data_dir)
            if os.path.isdir(pjoin(resolved_raw_data_dir, obj))
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
                raw_data_dir=raw_data_dir,
            )
    else:
        with ProcessPoolExecutor(max_workers=obj_workers) as executor:
            future_to_idx = {
                executor.submit(
                    _prepare_object_outputs_worker,
                    object_type,
                    max_files_per_object,
                    fw,
                    raw_data_dir,
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
    motion_metadata = {}

    for idx, object_type in enumerate(objects):
        payload = payloads[idx]
        if payload is None:
            continue
        squared_positions_error.update(payload['errors'])
        max_joints = max(max_joints, payload['max_joints'])
        cur_counter = files_counter
        files_counter, object_frames, object_motion_metadata = _write_object_outputs(
            target_dataset_dir,
            payload,
            files_counter,
        )
        frames_counter += object_frames
        cond[object_type] = payload['object_cond']
        objects_counter[object_type] = files_counter - cur_counter
        motion_metadata.update(object_motion_metadata)

    _write_dataset_artifacts(
        target_dataset_dir,
        cond,
        motion_metadata,
        objects_counter,
        max_joints,
        files_counter,
        frames_counter,
        squared_positions_error,
    )
##################################################################

############ Recover animation from motion features ##############
def recover_root_quat_and_pos_np(data, translation_root_index=None):
    motion = np.asarray(data)
    if motion.ndim == 2:
        root_features = motion
        translation_features = motion
    elif motion.ndim == 3:
        if translation_root_index is None:
            translation_root_index = infer_translation_root_from_features(motion)
        root_features = motion[:, 0, :]
        translation_features = motion[:, translation_root_index, :]
    else:
        raise ValueError(f"Expected feature tensor with shape (F, C) or (F, J, C), got {motion.shape}.")

    # joint row 0 stores the root-facing rotation used by the representation.
    r_rot_quat = Quaternions.from_transforms(rotation_6d_to_matrix_np(root_features[:, 3:9]))

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
def recover_from_bvh_ric_np(data, translation_root_index=None):
    if translation_root_index is None:
        translation_root_index = infer_translation_root_from_features(data)

    r_rot_quat, r_pos = recover_root_quat_and_pos_np(data, translation_root_index=translation_root_index)
    positions = np.asarray(data[..., :3], dtype=np.float32).copy()
    positions = np.repeat(-r_rot_quat[..., None, :], positions.shape[-2], axis=-2) * positions
    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    return positions

""" recover xyz positions from rot (root relative positions) torch """
def recover_from_bvh_rot_np(data, parents, offsets, translation_root_index=None):
    if translation_root_index is None:
        translation_root_index = infer_translation_root_from_features(data)

    r_rot_quat, r_pos = recover_root_quat_and_pos_np(data, translation_root_index=translation_root_index)
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
def recover_animation_from_motion_np(data, parents, offsets, anim_pos_threshold=0.01):
    translation_root_index = infer_translation_root_from_features(data)
    target_global        = recover_from_bvh_ric_np(data, translation_root_index=translation_root_index)              # (F, J, 3)
    _, anim_rot          = recover_from_bvh_rot_np(data, parents, offsets, translation_root_index=translation_root_index)
    glob_rot             = positions_global(anim_rot)                  # (F, J, 3)

    # joints whose FK-predicted global position drifts from the RIC truth
    per_joint_err = np.abs(target_global - glob_rot).max(axis=(0, 2)) # (J,)
    animated_joints = sorted(
        j for j in range(len(parents)) if per_joint_err[j] > anim_pos_threshold
    )

    if not animated_joints:
        return anim_rot, needs_bvh_position_channels(anim_rot)

    # solve for the local position that reproduces target_global under the new rots
    new_pos = anim_rot.positions.copy()
    for j in animated_joints:
        if j == 0 or parents[j] < 0:
            new_pos[:, j] = target_global[:, j]
            continue
        temp      = Animation(anim_rot.rotations, new_pos, anim_rot.orients,
                              anim_rot.offsets, anim_rot.parents)
        tg_rots   = rotations_global(temp)
        tg_pos    = positions_global(temp)
        p         = parents[j]
        new_pos[:, j] = (-tg_rots[:, p]) * (target_global[:, j] - tg_pos[:, p])

    anim_fixed = Animation(anim_rot.rotations, new_pos, anim_rot.orients,
                           anim_rot.offsets, anim_rot.parents)
    return anim_fixed, needs_bvh_position_channels(anim_fixed)

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
    motion_metadata = {}
    cur_counter = files_counter
    files_counter, frames_counter, max_joints, object_cond, object_motion_metadata = process_object(
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
    motion_metadata.update(object_motion_metadata)

    _write_dataset_artifacts(
        save_dir,
        cond,
        motion_metadata,
        objects_counter,
        max_joints,
        files_counter,
        frames_counter,
        squared_positions_error,
    )
    
    
def process_skeleton(object_name, bvh_dir, face_joints, save_dir, tpos_bvh=None, fbx_dir=None):
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
    motion_metadata = {}
    cur_counter = files_counter
    files_counter, frames_counter, max_joints, object_cond, object_motion_metadata = process_object(object_name, files_counter, frames_counter, max_joints, squared_positions_error, save_dir=save_dir, fbxs_dir=fbx_dir or bvh_dir, face_joints=face_joints, t_pos_path=tpos_bvh)
    # BUG4 (intentional): MP4 generation is omitted here to skip expensive video
    # generation during process_skeleton. Generating video previews is not
    # Note: MP4 generation has been removed - no save_animations parameter needed.
    if object_cond is None:
        print(f"No valid FBX data found for '{object_name}', aborting.")
        return
    cond[object_name] = object_cond
    objects_counter[object_name] = files_counter - cur_counter 
    motion_metadata.update(object_motion_metadata)

    _write_dataset_artifacts(
        save_dir,
        cond,
        motion_metadata,
        objects_counter,
        max_joints,
        files_counter,
        frames_counter,
        squared_positions_error,
    )
################################################################