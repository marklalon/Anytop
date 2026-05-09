from dataclasses import dataclass

from motion_lib import BVH, FBX, Animation, Quaternions
from motion_lib.Animation import positions_global, rotations_global, offsets_from_positions, offsets_global
from motion_lib import animation_from_positions
from collections import Counter, defaultdict
import json
import numpy as np 
import os 
from os.path import join as pjoin
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
import random
import math
import traceback
import torch
import bisect
import re 
import time 
from data_loaders.truebones.truebones_utils.param_utils import HML_REF_AXIAL_BONE_LENGTH, FOOT_CONTACT_HEIGHT_THRESH, DEFAULT_DATASET_DIR, MAX_JOINTS, MAX_PATH_LEN, MOTION_DIR, FOOT_CONTACT_VEL_THRESH, BVHS_DIR, OBJECT_SUBSETS_DICT, get_raw_data_dir, SNAKES, CHAIN_FORWARD_JOINTS, FLYING, FISH, VERTICAL_CLAMP_MIN_RATIO, VERTICAL_CLAMP_MAX_RATIO
from utils.rotation_conversions import rotation_6d_to_matrix_np
from utils.roundtrip_common import _load_fbx_skeleton_metadata
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
    _detect_joint_side,
)
from .face_orientation import (
    resolve_face_joints,
    calculate_root_quat,
    rotate_to_hml_orientation,
    resolve_forward_reference_joints,
)
from .fbx_filename_rules import (
    find_tpose_reference_path,
    _compact_normalized_text,
    _is_all_bundle_stem,
    _matches_object_alias,
    _normalize_action_name,
    _should_skip_fbx,
    _strip_leading_object_prefix,
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

LEAF_ROTATION_HELPER_SUFFIX = '__leafrot_helper'


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
        print(f'[PASS] canonical joint-name collision scan found no duplicate canonical names')

    return collision_groups


def _refresh_joint_metadata_in_object_cond(object_cond):
    joint_names = list(object_cond.get('joints_names') or [])
    if not joint_names:
        return

    parents = np.asarray(object_cond.get('parents'), dtype=np.int64)
    offsets = np.asarray(object_cond.get('offsets'), dtype=np.float64)
    original_joint_count = object_cond.get('original_joint_count')
    helper_joint_indices = [
        int(joint_index)
        for joint_index in list(object_cond.get('helper_joint_indices') or [])
    ]
    helper_source_leaf_indices = [
        int(joint_index)
        for joint_index in list(object_cond.get('helper_source_leaf_indices') or [])
    ]

    if (
        original_joint_count is not None
        and helper_joint_indices
        and len(helper_joint_indices) == len(helper_source_leaf_indices)
    ):
        original_joint_count = int(original_joint_count)
        if 0 < original_joint_count <= len(joint_names) and original_joint_count <= len(parents):
            base_semantic_metadata = _build_semantic_metadata(
                joint_names[:original_joint_count],
                parents[:original_joint_count],
                offsets[:original_joint_count],
            )
            semantic_metadata = _extend_semantic_metadata_with_leaf_helpers(
                base_semantic_metadata,
                joint_names,
                {
                    'helper_joint_indices': helper_joint_indices,
                    'helper_source_leaf_indices': helper_source_leaf_indices,
                },
            )
        else:
            semantic_metadata = _build_semantic_metadata(
                joint_names,
                parents,
                offsets,
            )
    else:
        semantic_metadata = _build_semantic_metadata(
            joint_names,
            parents,
            offsets,
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


def refresh_joint_metadata_in_cond_dict(cond_dict):
    if not isinstance(cond_dict, dict):
        return cond_dict

    for object_cond in cond_dict.values():
        if isinstance(object_cond, dict):
            _refresh_joint_metadata_in_object_cond(object_cond)
    return cond_dict


def _joint_name_embeddings_are_current(object_cond, embedding_texts, t5_name):
    joint_names = list(object_cond.get('joints_names') or [])
    embs = object_cond.get('joints_names_embs')
    meta = object_cond.get('joints_names_embs_meta')

    if embs is None or not isinstance(meta, dict):
        return False

    embs = np.asarray(embs)
    if embs.ndim != 2:
        return False
    if len(joint_names) != len(embedding_texts) or embs.shape[0] != len(joint_names):
        return False

    try:
        schema_version = int(meta.get('schema_version'))
        embedding_dim = int(meta.get('embedding_dim'))
    except (TypeError, ValueError):
        return False

    if meta.get('t5_name') != t5_name:
        return False
    if schema_version != JOINT_NAME_EMBEDDING_SCHEMA_VERSION:
        return False
    if embedding_dim != int(embs.shape[1]):
        return False
    if list(meta.get('embedding_texts') or []) != list(embedding_texts):
        return False

    return True


def _attach_joint_name_embeddings_to_cond(cond, save_dir, t5_name='t5-base', write_collision_report=True, force_reencode=True):

    if not cond:
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inspection_dir = pjoin(save_dir, 'joint_name_inspection')
    os.makedirs(inspection_dir, exist_ok=True)

    embedding_texts_by_object = {}
    object_types_to_encode = []
    for object_type in sorted(cond):
        object_cond = cond[object_type]
        _refresh_joint_metadata_in_object_cond(object_cond)
        embedding_texts = build_joint_embedding_texts(object_cond)
        embedding_texts_by_object[object_type] = embedding_texts
        if force_reencode or not _joint_name_embeddings_are_current(object_cond, embedding_texts, t5_name):
            object_types_to_encode.append(object_type)

    if object_types_to_encode:
        from model.conditioners import T5Conditioner

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
            for object_type in object_types_to_encode:
                object_cond = cond[object_type]
                embedding_texts = embedding_texts_by_object[object_type]
                names_tokens = t5_conditioner.tokenize_entries(embedding_texts)
                embs = t5_conditioner(names_tokens).detach().cpu().numpy().astype(np.float32, copy=False)
                object_cond['joints_names_embs'] = embs
                object_cond['joints_names_embs_meta'] = {
                    't5_name': t5_name,
                    'schema_version': JOINT_NAME_EMBEDDING_SCHEMA_VERSION,
                    'embedding_dim': int(embs.shape[1]) if embs.ndim == 2 else 0,
                    'embedding_texts': list(embedding_texts),
                }
    else:
        print(f'Reusing cached joint-name embeddings from cond.npy for {len(cond)} object types ({t5_name})')

    for object_type in sorted(cond):
        object_cond = cond[object_type]
        embedding_texts = embedding_texts_by_object[object_type]
        inspection_path = pjoin(inspection_dir, f'{object_type}.json')
        with open(inspection_path, 'w', encoding='utf-8') as inspection_file:
            json.dump(_build_joint_name_inspection_rows(object_cond, embedding_texts), inspection_file, indent=2)

    if write_collision_report:
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


def reorder_animation_to_dfs(anim, names):
    """Reindex an Animation into true DFS order for BVH export.

    Helper joints are appended at the tail of the joint arrays during
    preprocessing, which keeps parent-before-child ordering but no longer matches
    the BVH hierarchy traversal order. ``motion_lib.BVH.save`` writes its
    HIERARCHY recursively in DFS order while the MOTION block is emitted in array
    index order, so helper-augmented animations must be remapped so both orders
    agree.
    """
    parents = np.asarray(anim.parents, dtype=np.int32)
    joint_count = int(parents.shape[0])
    names = list(names)
    if len(names) != joint_count:
        raise ValueError(
            f"Expected {joint_count} joint names for BVH export, got {len(names)}"
        )
    if joint_count <= 1:
        return anim, names

    children = [[] for _ in range(joint_count)]
    for joint_index, parent_index in enumerate(parents):
        if parent_index < 0:
            continue
        if parent_index >= joint_count:
            raise ValueError(
                f"Joint {joint_index} has invalid parent {parent_index} for {joint_count} joints"
            )
        children[int(parent_index)].append(joint_index)

    roots = np.flatnonzero(parents < 0).tolist()
    if not roots:
        raise ValueError("BVH export requires at least one root joint")

    dfs_order = []
    stack = list(reversed(roots))
    while stack:
        joint_index = stack.pop()
        dfs_order.append(joint_index)
        stack.extend(reversed(children[joint_index]))

    if len(dfs_order) != joint_count:
        raise ValueError(
            f"DFS traversal covered {len(dfs_order)} of {joint_count} joints during BVH export"
        )

    if dfs_order == list(range(joint_count)):
        return anim, names

    old_to_new = np.empty((joint_count,), dtype=np.int32)
    for new_index, old_index in enumerate(dfs_order):
        old_to_new[old_index] = new_index

    reordered_parents = np.array([
        old_to_new[parent_index] if parent_index >= 0 else -1
        for parent_index in parents[dfs_order]
    ], dtype=np.int32)

    orient_count = len(anim.orients)
    if orient_count not in (0, joint_count):
        raise ValueError(
            f"Expected 0 or {joint_count} joint orients for BVH export, got {orient_count}"
        )
    reordered_orients = anim.orients.copy() if orient_count == 0 else anim.orients[dfs_order].copy()

    reordered_anim = Animation(
        anim.rotations[:, dfs_order].copy(),
        anim.positions[:, dfs_order].copy(),
        reordered_orients,
        anim.offsets[dfs_order].copy(),
        reordered_parents,
    )
    reordered_names = [names[joint_index] for joint_index in dfs_order]
    return reordered_anim, reordered_names

def _get_average_axial_bone_length(offsets, parents, joint_side_labels):
    """Compute the mean bone length of axial (center-labeled) bones, excluding root.

    Falls back to the mean bone length across *all* non-root bones when no
    center-labeled bones exist, so the return value is always a positive float.
    """
    total_length = 0.0
    axial_count = 0
    for joint_index in range(1, len(parents)):  # skip root (no parent bone)
        if joint_index < len(joint_side_labels) and joint_side_labels[joint_index] == 'center':
            bone_length = float(np.linalg.norm(offsets[joint_index]))
            total_length += bone_length
            axial_count += 1
    if axial_count >= 10:
        return total_length / axial_count
    # Fallback: average bone length across all non-root bones.
    all_lengths = [float(np.linalg.norm(offsets[j])) for j in range(1, len(parents))]
    if all_lengths:
        return sum(all_lengths) / len(all_lengths)
    return 0.1  # ultimate fallback for single-bone skeletons


def compute_scale_factor(axial_avg_len):
    if axial_avg_len <= 0:
        raise ValueError(f"Expected positive axial_avg_len, got {axial_avg_len}.")
    return float(HML_REF_AXIAL_BONE_LENGTH / axial_avg_len)


def scale(anim, scale_factor):
    if scale_factor is None:
        raise ValueError("scale_factor must be precomputed once per character and passed explicitly.")
    new_anim = Animation(
        anim.rotations.copy(),
        anim.positions * scale_factor,
        anim.orients.copy(),
        anim.offsets * scale_factor,
        anim.parents.copy(),
    )
    return new_anim

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
def process_anim(anim, object_type, orientation_quat, root_pose_init_xz=None, *, scale_factor):
    rotated = rotate_to_hml_orientation(anim, orientation_quat)
    baked = _bake_descendant_y_into_translation_root(rotated)
    centered, root_pose_init_xz_ = move_xz_to_origin(baked, root_pose_init_xz)
    scaled = scale(centered, scale_factor)
    return scaled, root_pose_init_xz_, scale_factor


def _reference_clip_needs_local_position_rebuild(anim, tol=1e-4):
    """Return the max absolute error between first-frame local positions and rest offsets.

    Returns 0.0 when the clip is already aligned (error <= tol) or when there are
    insufficient joints to compare. Callers can truthily check the return value
    to decide whether repair is needed.
    """
    if len(anim) == 0 or anim.positions.shape[1] <= 1:
        return 0.0

    root_candidates = np.where(np.asarray(anim.parents) < 0)[0]
    if root_candidates.size == 0:
        return 0.0

    nonroot_indices = np.delete(np.arange(anim.positions.shape[1]), int(root_candidates[0]))
    if nonroot_indices.size == 0:
        return 0.0

    local_positions = np.asarray(anim.positions[0, nonroot_indices], dtype=np.float64)
    rest_offsets = np.asarray(anim.offsets[nonroot_indices], dtype=np.float64)
    error = float(np.max(np.abs(local_positions - rest_offsets)))
    return error if error > tol else 0.0


def _leaf_rotation_helper_name(source_name, source_index):
    return f'{source_name}{LEAF_ROTATION_HELPER_SUFFIX}_{int(source_index)}'


def _dfs_leaf_joint_indices(parents):
    parents = np.asarray(parents, dtype=np.int32)
    if parents.size == 0:
        return []

    child_counts = np.bincount(parents[parents >= 0], minlength=len(parents))
    return [
        int(joint_index)
        for joint_index in range(len(parents))
        if parents[joint_index] >= 0 and child_counts[joint_index] == 0
    ]


def _build_leaf_rotation_helper_metadata(joint_names, parents, *, max_joints=MAX_JOINTS):
    parents = np.asarray(parents, dtype=np.int32)
    original_joint_count = int(len(parents))
    original_leaf_joint_indices = _dfs_leaf_joint_indices(parents)
    helper_budget = max(int(max_joints) - original_joint_count, 0)
    helper_source_leaf_indices = original_leaf_joint_indices[:helper_budget]
    helper_joint_indices = list(
        range(original_joint_count, original_joint_count + len(helper_source_leaf_indices))
    )
    helper_joint_names = [
        _leaf_rotation_helper_name(joint_names[source_index], source_index)
        for source_index in helper_source_leaf_indices
    ]
    return {
        'original_joint_count': original_joint_count,
        'original_leaf_joint_indices': list(original_leaf_joint_indices),
        'helper_joint_indices': list(helper_joint_indices),
        'helper_joint_names': list(helper_joint_names),
        'helper_joint_count': int(len(helper_joint_indices)),
        'helper_source_leaf_indices': list(helper_source_leaf_indices),
        'unaugmented_leaf_indices': list(original_leaf_joint_indices[len(helper_source_leaf_indices):]),
        'leaf_rotation_helper_suffix': LEAF_ROTATION_HELPER_SUFFIX,
        'max_joints_budget': int(max_joints),
    }


def _append_leaf_rotation_helpers_to_animation(anim, joint_names, helper_metadata):
    helper_joint_indices = list(helper_metadata.get('helper_joint_indices') or [])
    if len(helper_joint_indices) == 0:
        return anim.copy(), list(joint_names)

    original_joint_count = int(helper_metadata.get('original_joint_count', anim.shape[1]))
    expected_joint_count = original_joint_count + len(helper_joint_indices)
    if anim.shape[1] == expected_joint_count:
        return anim.copy(), list(joint_names)
    if anim.shape[1] != original_joint_count:
        raise ValueError(
            f'Animation joint count {anim.shape[1]} does not match helper source skeleton '
            f'count {original_joint_count}'
        )
    if len(joint_names) != original_joint_count:
        raise ValueError(
            f'Joint-name count {len(joint_names)} does not match helper source skeleton '
            f'count {original_joint_count}'
        )

    helper_source_leaf_indices = np.asarray(
        helper_metadata.get('helper_source_leaf_indices') or [],
        dtype=np.int32,
    )
    helper_joint_names = list(helper_metadata.get('helper_joint_names') or [])
    if helper_source_leaf_indices.shape != (len(helper_joint_indices),):
        raise ValueError('helper_source_leaf_indices must align with helper_joint_indices')
    if len(helper_joint_names) != len(helper_joint_indices):
        raise ValueError('helper_joint_names must align with helper_joint_indices')

    frame_count = anim.shape[0]
    helper_count = len(helper_joint_indices)
    helper_rotations = Quaternions.id((frame_count, helper_count)).qs.astype(anim.rotations.qs.dtype, copy=False)
    helper_positions = np.zeros((frame_count, helper_count, 3), dtype=anim.positions.dtype)
    helper_orients = Quaternions.id(helper_count).qs.astype(anim.orients.qs.dtype, copy=False)
    helper_offsets = np.zeros((helper_count, 3), dtype=anim.offsets.dtype)

    return Animation(
        Quaternions(np.concatenate([anim.rotations.qs.copy(), helper_rotations], axis=1)),
        np.concatenate([anim.positions.copy(), helper_positions], axis=1),
        Quaternions(np.concatenate([anim.orients.qs.copy(), helper_orients], axis=0)),
        np.concatenate([anim.offsets.copy(), helper_offsets], axis=0),
        np.concatenate([anim.parents.copy(), helper_source_leaf_indices], axis=0),
    ), list(joint_names) + helper_joint_names


def _extend_semantic_metadata_with_leaf_helpers(base_semantic_metadata, joint_names, helper_metadata):
    helper_joint_indices = list(helper_metadata.get('helper_joint_indices') or [])
    helper_source_leaf_indices = list(helper_metadata.get('helper_source_leaf_indices') or [])
    if len(helper_joint_indices) == 0:
        return dict(base_semantic_metadata)

    helper_joint_index_by_source = {
        int(source_index): int(helper_index)
        for source_index, helper_index in zip(helper_source_leaf_indices, helper_joint_indices)
    }

    canonical_joint_names = list(base_semantic_metadata['canonical_joint_names'])
    joint_side_labels = list(base_semantic_metadata['joint_side_labels'])
    symmetry_partner_indices = list(base_semantic_metadata['symmetry_partner_indices'])
    symmetric_joint_pairs = [list(pair) for pair in base_semantic_metadata['symmetric_joint_pairs']]
    symmetric_joint_pair_names = [list(pair) for pair in base_semantic_metadata['symmetric_joint_pair_names']]
    base_symmetry_partner_indices = list(base_semantic_metadata['symmetry_partner_indices'])

    for source_index in helper_source_leaf_indices:
        source_canonical_name = canonical_joint_names[int(source_index)]
        canonical_joint_names.append(f'{source_canonical_name} Helper')
        joint_side_labels.append(joint_side_labels[int(source_index)])
        symmetry_partner_indices.append(-1)

    for source_index, helper_index in zip(helper_source_leaf_indices, helper_joint_indices):
        partner_index = int(base_symmetry_partner_indices[int(source_index)])
        partner_helper_index = helper_joint_index_by_source.get(partner_index)
        if partner_helper_index is None:
            continue
        symmetry_partner_indices[int(helper_index)] = int(partner_helper_index)
        if int(helper_index) < int(partner_helper_index):
            symmetric_joint_pairs.append([int(helper_index), int(partner_helper_index)])
            symmetric_joint_pair_names.append([
                joint_names[int(helper_index)],
                joint_names[int(partner_helper_index)],
            ])

    return {
        'canonical_joint_names': canonical_joint_names,
        'end_effector_joints': list(base_semantic_metadata['end_effector_joints']),
        'end_effector_names': list(base_semantic_metadata['end_effector_names']),
        'contact_joints': list(base_semantic_metadata['contact_joints']),
        'contact_joint_names': list(base_semantic_metadata['contact_joint_names']),
        'contact_joint_source': base_semantic_metadata['contact_joint_source'],
        'joint_side_labels': joint_side_labels,
        'symmetry_partner_indices': symmetry_partner_indices,
        'symmetric_joint_pairs': symmetric_joint_pairs,
        'symmetric_joint_pair_names': symmetric_joint_pair_names,
        'mirror_disabled_joint_indices': list(base_semantic_metadata['mirror_disabled_joint_indices']),
        'mirror_disabled_joint_names': list(base_semantic_metadata['mirror_disabled_joint_names']),
        'mirror_disabled_warnings': list(base_semantic_metadata['mirror_disabled_warnings']),
        'is_symmetric': bool(base_semantic_metadata['is_symmetric']),
    }

""" get object_type common characteristics, extracted from T-pose FBX"""
def get_common_features_from_T_pose(
    t_pose_fbx_path,
    object_type,
    face_joints=None,
    *,
    augment_leaf_rotation_helpers=False,
    max_joints=MAX_JOINTS,
):
    t_pose_anim, t_pos_names, _t_pose_frame_time = FBX.load(t_pose_fbx_path)
    reference_anim = t_pose_anim[:1] if len(t_pose_anim) > 1 else t_pose_anim
    face_joints = resolve_face_joints(object_type, t_pos_names, reference_anim.parents, face_joints=face_joints)
    forward_joint_index, forward_base_joint_index = resolve_forward_reference_joints(
        t_pos_names,
        reference_anim.parents,
        object_type=object_type,
    )

    # This function only consumes reference-pose metadata from frame 0, so avoid
    # repairing every frame of long T-pose clips unless the first frame is malformed.
    # NOTE: skipped — pipeline uses positions_global + offsets_from_positions which
    # already compensates for local-position deviations; the IK repair is expensive
    # and the warning was misleading (orientation is unaffected in practice).
    # actual_error = _reference_clip_needs_local_position_rebuild(reference_anim)
    # if actual_error:
    #     import sys
    #     print(
    #         f"\x1b[33m[WARN] T-pose FBX local positions don't match rest offsets "
    #         f"(max error: {actual_error:.6f}); "
    #         "skipping expensive IK repair (animation_from_positions). "
    #         "This may cause orientation error for this character.\x1b[0m",
    #         file=sys.stderr,
    #         flush=True,
    #     )

    reference_positions = positions_global(reference_anim)
    t_pose_orientation_quat = calculate_root_quat(reference_positions, object_type, face_joint_indx=face_joints, forward_joint_index=forward_joint_index, forward_base_joint_index=forward_base_joint_index)[0]

    # Pre-compute the per-character scale factor once from the raw T-pose
    # offsets and reuse it for every motion clip of the same character.
    _tpose_side_labels = []
    for name in t_pos_names:
        detected = _detect_joint_side(name)
        _tpose_side_labels.append(detected if detected in ('left', 'right') else 'center')
    axial_avg_len = _get_average_axial_bone_length(reference_anim.offsets, reference_anim.parents, _tpose_side_labels)
    scale_factor = compute_scale_factor(axial_avg_len)

    scaled, root_pose_init_xz, scale_factor = process_anim(
        reference_anim,
        object_type,
        t_pose_orientation_quat,
        scale_factor=scale_factor,
    )
    helper_metadata = _build_leaf_rotation_helper_metadata(
        t_pos_names,
        scaled.parents,
        max_joints=max_joints if augment_leaf_rotation_helpers else len(scaled.parents),
    )
    if augment_leaf_rotation_helpers and helper_metadata['helper_joint_count'] > 0:
        scaled, t_pos_names = _append_leaf_rotation_helpers_to_animation(
            scaled,
            t_pos_names,
            helper_metadata,
        )
    scaled_positions = positions_global(scaled)
    scaled_rest_positions = scaled_positions[0]
    offsets = offsets_from_positions(scaled_rest_positions, scaled.parents)
    suspected_foot_indices, contact_joint_source = _infer_contact_joints(
        t_pos_names,
        scaled.parents,
        scaled_rest_positions,
    )
    return TPoseFeatures(
        root_pose_init_xz=root_pose_init_xz,
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
    root_pose_init_xz: tuple
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


def _coerce_single_orientation_quat(orientation_quat):
    if orientation_quat is None:
        raise ValueError(
            "orientation_quat must be precomputed from the reference T-pose and provided to downstream motion processing"
        )

    orientation_qs = getattr(orientation_quat, 'qs', orientation_quat)
    orientation_qs = np.asarray(orientation_qs, dtype=np.float64)
    if orientation_qs.ndim > 1:
        orientation_qs = orientation_qs[0]
    return Quaternions(orientation_qs.reshape(1, 4)).normalized()


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


def neutralize_animation_subtrees(anim, joint_indices):
    disabled_joint_indices = sorted({
        int(index)
        for index in joint_indices
        if int(index) > 0
    })
    if not disabled_joint_indices:
        return Animation(
            anim.rotations.copy(),
            anim.positions.copy(),
            anim.orients.copy(),
            anim.offsets.copy(),
            anim.parents.copy(),
        )

    neutral_positions = anim.positions.copy()
    neutral_rotations = anim.rotations.copy()
    neutral_positions[:, disabled_joint_indices] = np.asarray(anim.offsets, dtype=np.float64)[disabled_joint_indices][None, :, :]
    neutral_rotations[:, disabled_joint_indices] = Quaternions.id((len(anim), len(disabled_joint_indices)))

    return Animation(
        neutral_rotations,
        neutral_positions,
        anim.orients.copy(),
        anim.offsets.copy(),
        anim.parents.copy(),
    )


def _neutralize_mirror_disabled_subtrees(features, object_cond, mirrored_offsets):
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
    anim, _has_animated_pos = recover_animation_from_motion_np(motion, parents, offsets)
    if anim is None:
        neutralized = motion.copy()
        return neutralized[0] if squeeze_frame else neutralized

    neutral_anim = neutralize_animation_subtrees(anim, disabled_joint_indices)
    translation_root_index = _find_translation_root(neutral_anim)
    contact_joint_indices = list(object_cond['contact_joints'])
    face_joints = list(object_cond['face_joints']) or None

    cont_6d_params, _r_velocity, _velocity, r_rot, global_positions = get_bvh_cont6d_params(
        neutral_anim,
        str(object_cond['object_type']),
        object_cond['orientation_quat'],
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
    neutralized = neutralized.astype(motion.dtype, copy=False)
    return neutralized[0] if squeeze_frame else neutralized


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
        # Unpaired joints have no mirror partner so they can't be reflected meaningfully.
        # Restore their original x offsets so that when they are neutralized to rest pose,
        # they retain their original rest orientation rather than the x-flipped version
        # produced by the global mirror pass above.
        disabled_indices = [int(i) for i in object_cond['mirror_disabled_joint_indices'] if int(i) > 0]
        if disabled_indices:
            orig_offsets = np.asarray(object_cond['offsets'], dtype=np.float32)
            mirrored_offsets[disabled_indices, 0] = orig_offsets[disabled_indices, 0]
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

    # Local translations can be recovered directly from the target parent/child
    # global positions because global joint rotations depend only on `rotations`
    # and the hierarchy, not on the local translations we are solving for.
    global_rots = rotations_global(Animation(rotations, local_positions, orients, offsets, parents))
    for joint_idx, parent_idx in enumerate(parents):
        if parent_idx < 0:
            local_positions[:, joint_idx] = target_global_positions[:, joint_idx]
            continue

        local_positions[:, joint_idx] = (
            -global_rots[:, parent_idx]
        ) * (target_global_positions[:, joint_idx] - target_global_positions[:, parent_idx])

    direct_anim = Animation(rotations, local_positions, orients, offsets, parents)
    direct_global_pos = positions_global(direct_anim)
    direct_error = np.max(np.abs(target_global_positions - direct_global_pos))
    if direct_error <= position_match_threshold:
        return local_positions

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
            if parents[joint_idx] < 0:
                local_positions[:, joint_idx] = target_global_positions[:, joint_idx]
                continue

            parent_idx = parents[joint_idx]
            local_positions[:, joint_idx] = (
                -global_rots[:, parent_idx]
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
def get_bvh_cont6d_params(anim, object_type, orientation_quat, translation_root_index=0):
    positions = positions_global(anim)
    quat_params = anim.rotations
    r_rot = _coerce_single_orientation_quat(orientation_quat).repeat(positions.shape[0], axis=0)
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
def get_hml_aligned_anim(fbx_path_or_anim, object_type, root_pose_init_xz, tpos_rots, offsets, squared_positions_error, *, scale_factor, foot_indices=None, orientation_quat, slice_inds=None, preloaded=None, helper_metadata=None):
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
        processed_anim, _xz, _sf = process_anim(
            raw_anim,
            object_type,
            orientation_quat,
            root_pose_init_xz=root_pose_init_xz,
            scale_factor=scale_factor,
        )
        ## clamp vertical trajectory for flying/fish creatures (after scale, in HML units)
        processed_anim = _clamp_vertical_trajectory(processed_anim, object_type)
    else:
        names = list()
        processed_anim = fbx_path_or_anim
        frames_num = len(processed_anim)

    if processed_anim.positions.shape[1] != offsets.shape[0]:
        if helper_metadata is None:
            raise ValueError(
                f'Processed animation joint count {processed_anim.positions.shape[1]} does not match '
                f'offset count {offsets.shape[0]} without helper metadata'
            )
        if not names:
            raise ValueError('Cannot append helper joints to an Animation input without joint names')
        processed_anim, names = _append_leaf_rotation_helpers_to_animation(
            processed_anim,
            names,
            helper_metadata,
        )
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
    error_key = fbx_path_or_anim if isinstance(fbx_path_or_anim, str) else '__animation__'
    if slice_inds is not None and not isinstance(fbx_path_or_anim, Animation):
        error_key = f'{fbx_path_or_anim}[{slice_inds[0]}:{slice_inds[1]}]'
    squared_positions_error[error_key] = float(squared_error)

    return new_anim, processed_anim, names  
    
""" get motion feature representation"""
def get_motion(fbx_path_or_anim, foot_contact_vel_thresh, object_type, max_joints, root_pose_init_xz, offsets, foot_indices, tpos_rots, squared_positions_error, *, scale_factor, orientation_quat, slice_inds=None, preloaded=None, helper_metadata=None):
    try:
        new_anim, export_anim, names = get_hml_aligned_anim(
            fbx_path_or_anim,
            object_type,
            root_pose_init_xz,
            tpos_rots,
            offsets,
            squared_positions_error,
            scale_factor=scale_factor,
            foot_indices=foot_indices,
            orientation_quat=orientation_quat,
            slice_inds=slice_inds,
            preloaded=preloaded,
            helper_metadata=helper_metadata,
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
            orientation_quat,
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

def _process_motion_file(file_path, object_type, max_joints, root_pose_init_xz,
                         offsets, foot_indices, tpos_rots, scale_factor,
                         helper_metadata, orientation_quat):
    local_errors = dict()
    # Load the FBX file once; pass it as `preloaded` to every get_motion call so that
    raw_anim, names, frame_time = FBX.load(file_path)
    anim_len = len(raw_anim)
    begin = 0
    file_max_joints = max_joints
    file_results = []
    file_motion_errors = []

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
            offsets,
            foot_indices,
            tpos_rots,
            local_errors,
            scale_factor=scale_factor,
            orientation_quat=orientation_quat,
            slice_inds=[begin, slice_ind],
            preloaded=(raw_anim, names),
            helper_metadata=helper_metadata,
        )
        current_begin = begin
        begin = slice_ind

        if motion is None:
            err_msg = f"[FAIL] Object '{object_type}', file: {file_path}, slice {current_begin}:{slice_ind}"
            file_motion_errors.append(err_msg)
            continue

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
            'motion_labels': build_motion_labels(object_type, raw_action),
        })

    return {
        'errors': local_errors,
        'max_joints': file_max_joints,
        'results': file_results,
        'motion_errors': file_motion_errors,
    }


def _attach_orientation_reference_metadata(
    object_cond,
    orientation_quat,
    forward_joint_index,
    forward_base_joint_index,
    orientation_reference_fbx_path,
):
    orientation_qs = _coerce_single_orientation_quat(orientation_quat).qs[0]
    object_cond['orientation_quat'] = orientation_qs.reshape(4)
    object_cond['forward_joint_index'] = int(forward_joint_index) if forward_joint_index is not None else None
    object_cond['forward_base_joint_index'] = int(forward_base_joint_index) if forward_base_joint_index is not None else None
    object_cond['orientation_reference_fbx_path'] = (
        os.path.abspath(orientation_reference_fbx_path)
        if orientation_reference_fbx_path
        else None
    )


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

    return motion_labels
     
"""Prepare processed tensors for all the files of a given object without writing them to disk yet."""
def _prepare_object_outputs(object_type, max_joints, face_joints=None, fbxs_dir=None, t_pos_path=None, max_files=None, raw_data_dir=None, allow_skeleton_only=False):
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
        if allow_skeleton_only:
            print(f'processing {object_type} in skeleton-only mode using T-pose reference {os.path.basename(t_pos_path)}')
        else:
            print(f'skipping {object_type}: no valid FBX files after filtering')
            return None

    squared_positions_error = dict()
    tp = get_common_features_from_T_pose(
        t_pos_path,
        object_type,
        face_joints=face_joints,
        augment_leaf_rotation_helpers=True,
        max_joints=MAX_JOINTS,
    )
    character_scale_factor = float(tp.scale_factor)
    t_pos_motion, parents, max_joints, new_anim, _export_anim, _tpos_is_loop = get_motion(
        tp.tpos_anim,
        FOOT_CONTACT_VEL_THRESH,
        object_type,
        max_joints,
        tp.root_pose_init_xz,
        tp.offsets,
        tp.foot_indices,
        tp.tpos_rots,
        squared_positions_error,
        scale_factor=character_scale_factor,
        orientation_quat=tp.orientation_quat,
        helper_metadata=tp.helper_metadata,
    )
    rest_positions = _rest_positions_from_offsets(tp.offsets, parents)
    original_joint_count = int(tp.helper_metadata['original_joint_count'])
    base_semantic_metadata = _build_semantic_metadata(
        tp.names[:original_joint_count],
        parents[:original_joint_count],
        tp.offsets[:original_joint_count],
        rest_positions=rest_positions[:original_joint_count],
    )
    semantic_metadata = _extend_semantic_metadata_with_leaf_helpers(
        base_semantic_metadata,
        tp.names,
        tp.helper_metadata,
    )
    object_cond['tpos_first_frame'] = t_pos_motion[0]
    # create topology conditions
    joint_relations, joints_graph_dist = create_topology_edge_relations(tp.tpos_anim.parents, max_path_len = MAX_PATH_LEN)
    object_cond['joint_relations'] = joint_relations
    object_cond['joints_graph_dist'] = joints_graph_dist
    object_cond['object_type'] = object_type
    object_cond['parents'] = parents
    object_cond['offsets'] = tp.offsets
    object_cond['joints_names'] = tp.names
    object_cond['canonical_joint_names'] = semantic_metadata['canonical_joint_names']
    object_cond['canonical_bvh_joint_names'] = [
        _canonical_name_for_bvh(canonical_name, raw_name)
        for canonical_name, raw_name in zip(semantic_metadata['canonical_joint_names'], tp.names)
    ]
    object_cond['face_joints'] = list(tp.face_joints)
    object_cond['face_joint_names'] = [tp.names[index] for index in tp.face_joints]
    _attach_orientation_reference_metadata(
        object_cond,
        tp.orientation_quat,
        tp.forward_joint_index,
        tp.forward_base_joint_index,
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
    object_cond['original_joint_count'] = int(tp.helper_metadata['original_joint_count'])
    object_cond['original_leaf_joint_indices'] = list(tp.helper_metadata['original_leaf_joint_indices'])
    object_cond['helper_joint_indices'] = list(tp.helper_metadata['helper_joint_indices'])
    object_cond['helper_joint_names'] = list(tp.helper_metadata['helper_joint_names'])
    object_cond['helper_joint_count'] = int(tp.helper_metadata['helper_joint_count'])
    object_cond['helper_source_leaf_indices'] = list(tp.helper_metadata['helper_source_leaf_indices'])
    object_cond['unaugmented_leaf_indices'] = list(tp.helper_metadata['unaugmented_leaf_indices'])
    object_cond['leaf_rotation_helper_suffix'] = tp.helper_metadata['leaf_rotation_helper_suffix']
    object_cond['root_pose_init_xz'] = np.array(tp.root_pose_init_xz, dtype=np.float64)
    object_cond['scale_factor'] = character_scale_factor
    object_cond['axial_avg_len'] = float(tp.axial_avg_len)
    object_cond['kinematic_chains'] = parents2kinchains(parents, object_policy(object_type))
    object_cond.update(build_object_labels(object_type))
    all_tensors = list()

    # FBX loading via bpy is single-threaded inside a process because clear_scene
    # mutates global Blender state, so file-level parallelism is intentionally removed.
    print(f'processing {len(fbx_files)} FBX files for {object_type} (serial — bpy is single-threaded)', flush=True)

    def process_file(file_path):
        print("processing file: " + file_path, flush=True)
        return _process_motion_file(
            file_path,
            object_type,
            max_joints,
            tp.root_pose_init_xz,
            tp.offsets,
            tp.foot_indices,
            tp.tpos_rots,
            character_scale_factor,
            tp.helper_metadata,
            orientation_quat=tp.orientation_quat,
        )

    file_outputs = [process_file(file_path) for file_path in fbx_files]

    files_counter = 0
    frames_counter = 0
    prepared_results = []
    all_motion_errors = []
    for file_output in file_outputs:
        squared_positions_error.update(file_output['errors'])
        max_joints = max(max_joints, file_output['max_joints'])
        all_motion_errors.extend(file_output.get('motion_errors', []))
        for result in file_output['results']:
            motion = result['motion']
            all_tensors.append(motion)
            files_counter += 1
            frames_counter += motion.shape[0]
            result['canonical_names'] = list(object_cond['canonical_bvh_joint_names'])
            prepared_results.append(result)

    if len(all_tensors) == 0:
        if not allow_skeleton_only:
            print(f'skipping {object_type}: no valid motion tensors were produced')
            return None
        print(f'no motion clips were produced for {object_type}; using the T-pose features to populate cond statistics only')
        stats_tensors = np.asarray(t_pos_motion, dtype=np.float32)
        if stats_tensors.ndim == 2:
            stats_tensors = stats_tensors[None, ...]
    else:
        stats_tensors = np.concatenate(all_tensors, axis=0)

    mean, std = get_mean_std(stats_tensors)
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
        'motion_errors': all_motion_errors,
    }


"""Write a prepared object payload to disk with stable sequential clip naming."""
def _write_object_outputs(save_dir, object_payload, files_counter):
    object_type = object_payload['object_type']
    frames_counter = 0
    motion_metadata = {}

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
        bvh_names = list(result.get('canonical_names', result['names']))
        anim_obj, bvh_names = reorder_animation_to_dfs(anim_obj, bvh_names)
        BVH.save(
            pjoin(save_dir, BVHS_DIR, name + '.bvh'),
            anim_obj,
            bvh_names,
            frametime=result.get('frame_time', 1.0 / 24.0),
            positions=needs_bvh_position_channels(anim_obj),
        )

        motion_labels = _build_motion_metadata_entry(result, motion_file_name)
        motion_metadata[motion_file_name] = motion_labels

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

def _resolve_preprocessing_workers(objects, object_workers=8):
    object_count = max(1, len(objects))
    return min(object_count, max(1, int(object_workers)))


def _prepare_object_outputs_worker(object_type, max_files, raw_data_dir=None):
    return _prepare_object_outputs(
        object_type,
        max_joints=23,
        max_files=max_files,
        raw_data_dir=raw_data_dir,
    )

""" creates processed tensors for all the files of a given object. Returens statistics and the object condition,
which includes tpos, relation/distances matrices, offsets, parents, joints names, kinematic chains, mean and std"""    
def process_object(object_type, files_counter, frames_counter, max_joints, squared_positions_error, save_dir = DEFAULT_DATASET_DIR, face_joints=None, fbxs_dir=None, t_pos_path=None, max_files=None, raw_data_dir=None, bvhs_dir=None, allow_skeleton_only=False):
    object_payload = _prepare_object_outputs(
        object_type,
        max_joints,
        face_joints=face_joints,
        fbxs_dir=fbxs_dir or bvhs_dir,  # bvhs_dir kept for backward compatibility
        t_pos_path=t_pos_path,
        max_files=max_files,
        raw_data_dir=raw_data_dir,
        allow_skeleton_only=allow_skeleton_only,
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
def create_data_samples(objects=None, max_files_per_object=None, dataset_dir=None, raw_data_dir=None, object_workers=8):
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

    obj_workers = _resolve_preprocessing_workers(
        objects,
        object_workers=object_workers,
    )
    print(f'Preprocessing {len(objects)} characters with {obj_workers} object workers')

    payloads = [None] * len(objects)
    if obj_workers <= 1:
        for idx, object_type in enumerate(objects):
            payloads[idx] = _prepare_object_outputs(
                object_type,
                max_joints=23,
                max_files=max_files_per_object,
                raw_data_dir=raw_data_dir,
            )
    else:
        with ProcessPoolExecutor(max_workers=obj_workers) as executor:
            future_to_idx = {
                executor.submit(
                    _prepare_object_outputs_worker,
                    object_type,
                    max_files_per_object,
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

    all_motion_errors = []
    for idx, object_type in enumerate(objects):
        payload = payloads[idx]
        if payload is None:
            continue
        squared_positions_error.update(payload['errors'])
        max_joints = max(max_joints, payload['max_joints'])
        all_motion_errors.extend(payload.get('motion_errors', []))
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

    if all_motion_errors:
        print(f"\n{'=' * 70}")
        print(f"MOTION PROCESSING ERRORS ({len(all_motion_errors)} total)")
        print('=' * 70)
        for err in all_motion_errors:
            print(err)
        print(f"{'=' * 70}\n")

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

    new_pos = _solve_local_positions_for_target_global(
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
    anim_pos_threshold=0.01,
):
    """Recover a motion tensor and remap it into BVH-safe DFS order.

    ``recover_animation_from_motion_np`` intentionally preserves the input joint
    indexing because non-export callers still address joints by the original cond
    metadata indices. BVH export has the additional requirement that joint arrays
    must match hierarchy DFS order, so this helper layers the DFS remap on top of
    recovery without changing the base function's semantics.
    """
    anim, has_animated_pos = recover_animation_from_motion_np(
        data,
        parents,
        offsets,
        anim_pos_threshold=anim_pos_threshold,
    )
    if anim is None:
        return None, list(joint_names), has_animated_pos

    anim, joint_names = reorder_animation_to_dfs(anim, joint_names)
    return anim, joint_names, has_animated_pos

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
def process_single_object_type(object_type, save_dir):
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
    files_counter, frames_counter, max_joints, object_cond, object_motion_metadata = process_object(object_name, files_counter, frames_counter, max_joints, squared_positions_error, save_dir=save_dir, fbxs_dir=fbx_dir or bvh_dir, face_joints=face_joints, t_pos_path=tpos_bvh, allow_skeleton_only=True)
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