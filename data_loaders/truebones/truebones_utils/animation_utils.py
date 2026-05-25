"""Animation processing & joint metadata utilities.

Lowest layer of the motion-processing pipeline. Handles FK transforms,
coordinate normalization, scaling, BVH export preparation, joint-name
canonicalization, leaf rotation helpers, and mirror augmentation.
"""

from motion_lib import Animation, Quaternions
from motion_lib.Animation import positions_global, rotations_global
from collections import Counter, defaultdict
import json
import numpy as np
import os
from os.path import join as pjoin
import re
import torch
from data_loaders.truebones.truebones_utils.param_utils import HML_REF_AXIAL_BONE_LENGTH, HML_REF_MAX_SPAN, MAX_JOINTS, FLYING, FISH, SCALE_BODY_SPAN_BLEND_WEIGHT, VERTICAL_CLAMP_MIN_RATIO, VERTICAL_CLAMP_MAX_RATIO
from .physics_joint_annotation import (
    build_semantic_metadata,
    normalize_joint_name,
    strip_joint_name_prefix,
    build_joint_embedding_texts,
    JOINT_NAME_EMBEDDING_SCHEMA_VERSION,
)


################## Constants #####################

ANSI_YELLOW = '\033[93m'
ANSI_RESET = '\033[0m'


def _warn(msg: str):
    """Print a warning message in yellow."""
    print(f'{ANSI_YELLOW}[WARN] {msg}{ANSI_RESET}')


# Maximum translation-root XZ distance from the centred origin (in
# HML-normalised units) before we consider a clip locomotion and forcibly zero
# the root XZ. Clips are centred on the effective translation root's initial XZ
# position before this threshold is evaluated.
ROOT_XZ_STRIP_THRESHOLD = 1

# Mean L2 distance (per joint, in HML-normalised units) between first and last
# frame poses below which a clip is classified as looping.
LOOP_DETECTION_POS_THRESHOLD = 0.05

LEAF_ROTATION_HELPER_SUFFIX = '__rot_helper'


_EMITTED_MIRROR_SAFEGUARD_WARNINGS = set()


################## Joint Name Canonicalization #####################

def canonical_name_for_bvh(name, fallback_name):
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
    stripped_raw = strip_joint_name_prefix(raw_value)
    raw_tokens = normalize_joint_name(stripped_raw).split()
    canonical_tokens = normalize_joint_name(canonical_name).split()
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


def collect_joint_name_collision_groups(cond):
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


def write_joint_name_collision_report(cond, save_dir):
    collision_groups = collect_joint_name_collision_groups(cond)
    report = {
        'num_objects': int(len(cond)),
        'num_collision_groups': int(len(collision_groups)),
        'collision_groups': collision_groups,
    }
    report_path = pjoin(save_dir, 'joint_name_collision_report.json')
    with open(report_path, 'w', encoding='utf-8') as report_file:
        json.dump(report, report_file, indent=2)

    if collision_groups:
        _warn(f'canonical joint-name collision scan found {len(collision_groups)} group(s); report: {report_path}')
        for group in collision_groups[:20]:
            raw_names = ' | '.join(row['raw_name'] for row in group['rows'])
            print(f"  - {group['object_type']}: {group['canonical_name']} <- {raw_names}")
        if len(collision_groups) > 20:
            print(f'  ... {len(collision_groups) - 20} additional group(s) omitted from console output')
    else:
        print(f'[OK] canonical joint-name collision scan found no duplicate canonical names')

    return collision_groups


def refresh_joint_metadata_in_object_cond(object_cond):
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
            base_semantic_metadata = build_semantic_metadata(
                joint_names[:original_joint_count],
                parents[:original_joint_count],
                offsets[:original_joint_count],
            )
            semantic_metadata = extend_semantic_metadata_with_leaf_helpers(
                base_semantic_metadata,
                joint_names,
                {
                    'helper_joint_indices': helper_joint_indices,
                    'helper_source_leaf_indices': helper_source_leaf_indices,
                },
            )
        else:
            semantic_metadata = build_semantic_metadata(
                joint_names,
                parents,
                offsets,
            )
    else:
        semantic_metadata = build_semantic_metadata(
            joint_names,
            parents,
            offsets,
        )
    object_cond['canonical_joint_names'] = _disambiguate_duplicate_canonical_names(
        joint_names,
        semantic_metadata['canonical_joint_names'],
    )
    object_cond['canonical_bvh_joint_names'] = [
        canonical_name_for_bvh(canonical_name, raw_name)
        for canonical_name, raw_name in zip(semantic_metadata['canonical_joint_names'], joint_names)
    ]
    object_cond['canonical_bvh_joint_names'] = [
        canonical_name_for_bvh(canonical_name, raw_name)
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
            refresh_joint_metadata_in_object_cond(object_cond)
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


def attach_joint_name_embeddings_to_cond(cond, save_dir, t5_name='t5-base', write_collision_report=True, force_reencode=True):

    if not cond:
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inspection_dir = pjoin(save_dir, 'joint_name_inspection')
    os.makedirs(inspection_dir, exist_ok=True)

    embedding_texts_by_object = {}
    object_types_to_encode = []
    for object_type in sorted(cond):
        object_cond = cond[object_type]
        refresh_joint_metadata_in_object_cond(object_cond)
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
        #print(f'Reusing cached joint-name embeddings from cond.npy for {len(cond)} object types ({t5_name})')
        pass

    for object_type in sorted(cond):
        object_cond = cond[object_type]
        embedding_texts = embedding_texts_by_object[object_type]
        inspection_path = pjoin(inspection_dir, f'{object_type}.json')
        with open(inspection_path, 'w', encoding='utf-8') as inspection_file:
            json.dump(_build_joint_name_inspection_rows(object_cond, embedding_texts), inspection_file, indent=2)

    if write_collision_report:
        write_joint_name_collision_report(cond, save_dir)


################## Animation Transform Utilities #####################

def detect_motion_loop(positions):
    """Return True if the last frame's root-relative pose is close to the first frame's."""
    if positions.shape[0] < 2:
        return False
    per_joint_dist = np.linalg.norm(positions[-1] - positions[0], axis=-1)
    return bool(np.mean(per_joint_dist) < LOOP_DETECTION_POS_THRESHOLD)


def _translation_root_candidate_chain(parents, max_depth=5):
    parents = np.asarray(parents, dtype=np.int32).reshape(-1)
    if parents.size == 0:
        return [0]

    root_candidates = np.flatnonzero(parents < 0)
    if root_candidates.size == 0:
        return [0]

    current = int(root_candidates[0])
    chain = [current]
    for _depth in range(max(int(max_depth), 0)):
        children = np.flatnonzero(parents == current)
        if children.size != 1:
            break
        current = int(children[0])
        chain.append(current)

    return chain


def find_translation_root(anim, max_depth=5):
    """Return the first near-root joint with significant local position animation.

    Most skeletons carry root motion on joint 0, but some Truebones rigs
    (Horse, Bear, Camel, Trex, etc.) use intermediate bones like Bip01 or
    jt_Cog_C. Detecting this allows the rest of the pipeline to centre,
    strip trajectory, and export BVH correctly.

    Detection is intentionally limited to the unbranched chain directly below
    the hierarchy root. Once the chain branches, deeper descendants are treated
    as limb / local motion rather than transport candidates. Search also stops
    after ``max_depth`` descendants to avoid latching onto deep noisy joints.

    Returns the hierarchy root (normally joint 0) when no candidate carries
    significant local position animation.
    """
    frame_count = int(anim.positions.shape[0]) if anim.positions.ndim >= 3 else 0
    min_active_frames = max(3, int(np.ceil(frame_count * 0.2))) if frame_count > 0 else 0
    candidate_chain = _translation_root_candidate_chain(anim.parents, max_depth=max_depth)
    fallback_joint = int(candidate_chain[0]) if candidate_chain else 0

    for j in candidate_chain:
        joint_positions = np.asarray(anim.positions[:, j], dtype=np.float64)
        ptp = np.ptp(joint_positions, axis=0)
        if not np.any(ptp > 5e-3):
            continue

        delta_from_first = np.linalg.norm(joint_positions - joint_positions[0:1], axis=-1)
        active_frames = int(np.count_nonzero(delta_from_first > 5e-3))

        if active_frames >= min_active_frames:
            return j

        # Only consider as fallback if there are genuinely active frames,
        # not just a single-frame PTP spike from numerical noise.
        if fallback_joint == candidate_chain[0] and j != candidate_chain[0] and active_frames >= 2:
            fallback_joint = j

    return int(fallback_joint)


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


def bake_descendant_y_into_translation_root(anim, max_depth=2):
    """Bake near-root locator-style Y transport back onto the translation root.

    The heuristic is intentionally narrow: follow a single chain at most two levels
    below the effective translation root, allow one dummy node in the middle, and
    bake only joints in that chain that carry animated local Y.
    """
    trans_root = find_translation_root(anim)
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
    return get_rest_body_max_span(anim.offsets, anim.parents)


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


def clamp_vertical_trajectory(
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

    trans_root = find_translation_root(processed_anim)
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


def _coerce_root_xz_center(root_xz_center):
    root_xz_center = np.asarray(root_xz_center, dtype=np.float64).reshape(-1)
    if root_xz_center.size == 3:
        return root_xz_center
    if root_xz_center.size == 2:
        return np.array([root_xz_center[0], 0.0, root_xz_center[1]], dtype=np.float64)
    raise ValueError(f"root_xz_center must have shape (2,) or (3,), got {root_xz_center.shape}")


def _get_translation_root_initial_xz(anim, translation_root_index=None):
    """Return the effective translation root's initial XZ position in global space."""
    if translation_root_index is None:
        translation_root_index = find_translation_root(anim)

    global_pos = positions_global(anim)
    root_xz = np.asarray(global_pos[0, translation_root_index, [0, 2]], dtype=np.float64)
    return np.array([root_xz[0], 0.0, root_xz[1]], dtype=np.float64)


""" move motion s.t the effective translation root's initial XZ is centred at the origin.

For most skeletons joint 0 carries the root motion, but some rigs store it
on an intermediate bone (e.g. Bip01 for Horse).  We detect the effective
root via its global position and apply the shift to joint 0 (whose local
position equals its global position), so the entire skeleton moves via FK.
"""
def move_xz_to_origin(anim, root_xz_center=None):
    if root_xz_center is None:
        root_xz_center = _get_translation_root_initial_xz(anim)
    else:
        root_xz_center = _coerce_root_xz_center(root_xz_center)
    new_positions = anim.positions.copy()
    new_positions[:, 0] -= root_xz_center
    new_offsets = anim.offsets.copy()
    new_offsets[0] -= root_xz_center
    new_anim = Animation(anim.rotations.copy(), new_positions, anim.orients.copy(), new_offsets, anim.parents.copy())
    return new_anim, root_xz_center


def xz_locomotion_extent(anim, translation_root_index):
    """Return the maximum translation-root XZ distance from the current origin."""
    global_pos = positions_global(anim)
    root_xz = global_pos[:, translation_root_index, [0, 2]]
    return float(np.linalg.norm(root_xz, axis=1).max())


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


def resolve_detected_translation_root_index(aligned_index, export_index, object_type):
    valid_indices = sorted({int(index) for index in (aligned_index, export_index) if int(index) >= 0})
    if len(valid_indices) > 1:
        raise ValueError(
            f"{object_type}: inconsistent translation_root_index between aligned and export animations: "
            f"{aligned_index} vs {export_index}"
        )
    if valid_indices:
        return valid_indices[0]
    return -1


################## BVH Export Utilities #####################

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


################## Scaling Utilities #####################

def get_average_axial_bone_length(offsets, parents, joint_side_labels):
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


def get_rest_body_max_span(offsets, parents):
    rest_positions = np.zeros_like(np.asarray(offsets, dtype=np.float64))
    for joint_index, parent_index in enumerate(parents):
        if parent_index >= 0:
            rest_positions[joint_index] = rest_positions[parent_index] + offsets[joint_index]
        else:
            rest_positions[joint_index] = offsets[joint_index]
    joint_deltas = rest_positions[:, None, :] - rest_positions[None, :, :]
    max_span = np.linalg.norm(joint_deltas, axis=-1).max()
    return max(float(max_span), 1e-8)


def compute_scale_factor(axial_avg_len, body_max_span=None, *, span_blend_weight=SCALE_BODY_SPAN_BLEND_WEIGHT):
    """Blend axial and whole-body span scaling to reduce size outliers symmetrically.

    Axial mean bone length remains the primary normalization signal because it
    tracks body thickness better than raw max span. A secondary max-span term is
    blended in log-space so compact skeletons scale up and wide/long skeletons
    scale down without letting tails or wings dominate as aggressively as pure
    max-span scaling.
    """
    if axial_avg_len <= 0:
        raise ValueError(f"Expected positive axial_avg_len, got {axial_avg_len}.")
    if not 0.0 <= span_blend_weight <= 1.0:
        raise ValueError(f"Expected span_blend_weight in [0, 1], got {span_blend_weight}.")

    axial_scale_factor = HML_REF_AXIAL_BONE_LENGTH / axial_avg_len
    if body_max_span is None or span_blend_weight == 0.0:
        return float(axial_scale_factor)
    if body_max_span <= 0:
        raise ValueError(f"Expected positive body_max_span, got {body_max_span}.")

    span_scale_factor = HML_REF_MAX_SPAN / body_max_span
    return float(
        (axial_scale_factor ** (1.0 - span_blend_weight))
        * (span_scale_factor ** span_blend_weight)
    )


def scale_anim(anim, scale_factor):
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


################## Leaf Rotation Helpers #####################

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


def _is_terminal_leaf_name(name):
    """Return True if the joint name already looks like a terminal / non-rotating leaf.

    These typically do not need an extra rotation helper because they are
    already semantic end-effectors (end sites, helpers, nubs, etc.).
    """
    upper = name.upper()
    for keyword in ('END', 'HELPER'):
        if keyword in upper:
            return True
    return False


def _select_leaf_rotation_helper_source_indices(
    joint_names,
    parents,
    original_leaf_joint_indices,
    helper_budget,
    *,
    offsets=None,
):
    original_leaf_joint_indices = [int(joint_index) for joint_index in original_leaf_joint_indices]
    if helper_budget <= 0 or not original_leaf_joint_indices:
        return []
    if offsets is None or helper_budget >= len(original_leaf_joint_indices):
        return original_leaf_joint_indices[:helper_budget]

    offsets = np.asarray(offsets, dtype=np.float64)
    if offsets.shape[0] != len(parents):
        return original_leaf_joint_indices[:helper_budget]

    semantic_metadata = build_semantic_metadata(joint_names, parents, offsets)
    symmetry_partner_indices = [
        int(partner_index)
        for partner_index in list(semantic_metadata.get('symmetry_partner_indices') or [])
    ]
    joint_side_labels = list(semantic_metadata.get('joint_side_labels') or [])
    leaf_order = {
        int(joint_index): order_index
        for order_index, joint_index in enumerate(original_leaf_joint_indices)
    }
    leaf_set = set(original_leaf_joint_indices)

    paired_groups = []
    singleton_groups = []
    seen_leaf_indices = set()
    for leaf_index in original_leaf_joint_indices:
        if leaf_index in seen_leaf_indices:
            continue

        partner_index = symmetry_partner_indices[leaf_index] if leaf_index < len(symmetry_partner_indices) else -1
        side_label = joint_side_labels[leaf_index] if leaf_index < len(joint_side_labels) else 'center'
        if (
            side_label in ('left', 'right')
            and partner_index in leaf_set
            and partner_index != leaf_index
        ):
            group = sorted(
                [int(leaf_index), int(partner_index)],
                key=lambda joint_index: leaf_order[joint_index],
            )
            paired_groups.append(group)
            seen_leaf_indices.update(group)
            continue

        singleton_groups.append([int(leaf_index)])
        seen_leaf_indices.add(int(leaf_index))

    selected_leaf_indices = []
    for groups in (paired_groups, singleton_groups):
        for group in groups:
            if len(selected_leaf_indices) + len(group) > helper_budget:
                continue
            selected_leaf_indices.extend(group)

    return sorted(selected_leaf_indices, key=lambda joint_index: leaf_order[joint_index])


def build_leaf_rotation_helper_metadata(joint_names, parents, *, max_joints=MAX_JOINTS, offsets=None):
    parents = np.asarray(parents, dtype=np.int32)
    original_joint_count = int(len(parents))
    original_leaf_joint_indices = _dfs_leaf_joint_indices(parents)
    # Skip leaves that already have terminal names (End, Helper, Nub, etc.) —
    # they don't need an extra rotation helper.
    candidate_leaf_indices = [
        idx for idx in original_leaf_joint_indices
        if not _is_terminal_leaf_name(joint_names[idx])
    ]
    helper_budget = max(int(max_joints) - original_joint_count, 0)
    # Under a tight joint budget, prefer complete left/right helper pairs so a
    # mirrored export never carries a single-sided helper across the body.
    helper_source_leaf_indices = _select_leaf_rotation_helper_source_indices(
        joint_names,
        parents,
        candidate_leaf_indices,
        helper_budget,
        offsets=offsets,
    )
    helper_joint_indices = list(
        range(original_joint_count, original_joint_count + len(helper_source_leaf_indices))
    )
    helper_joint_names = [
        _leaf_rotation_helper_name(joint_names[source_index], source_index)
        for source_index in helper_source_leaf_indices
    ]
    selected_helper_sources = {int(source_index) for source_index in helper_source_leaf_indices}
    return {
        'original_joint_count': original_joint_count,
        'original_leaf_joint_indices': list(original_leaf_joint_indices),
        'helper_joint_indices': list(helper_joint_indices),
        'helper_joint_names': list(helper_joint_names),
        'helper_joint_count': int(len(helper_joint_indices)),
        'helper_source_leaf_indices': list(helper_source_leaf_indices),
        'unaugmented_leaf_indices': [
            int(joint_index)
            for joint_index in original_leaf_joint_indices
            if int(joint_index) not in selected_helper_sources
        ],
        'leaf_rotation_helper_suffix': LEAF_ROTATION_HELPER_SUFFIX,
        'max_joints_budget': int(max_joints),
    }


def append_leaf_rotation_helpers_to_animation(anim, joint_names, helper_metadata):
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


def extend_semantic_metadata_with_leaf_helpers(base_semantic_metadata, joint_names, helper_metadata):
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
    mirror_disabled_joint_indices = list(base_semantic_metadata['mirror_disabled_joint_indices'])
    mirror_disabled_joint_names = list(base_semantic_metadata['mirror_disabled_joint_names'])
    mirror_disabled_warnings = list(base_semantic_metadata['mirror_disabled_warnings'])
    mirror_disabled_set = {int(joint_index) for joint_index in mirror_disabled_joint_indices}

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

    truncated_helper_names = []
    for source_index, helper_index in zip(helper_source_leaf_indices, helper_joint_indices):
        partner_index = int(base_symmetry_partner_indices[int(source_index)])
        if partner_index < 0:
            continue
        if helper_joint_index_by_source.get(partner_index) is not None:
            continue
        if int(helper_index) in mirror_disabled_set:
            continue

        mirror_disabled_set.add(int(helper_index))
        mirror_disabled_joint_indices.append(int(helper_index))
        mirror_disabled_joint_names.append(joint_names[int(helper_index)])
        truncated_helper_names.append(joint_names[int(helper_index)])

    if truncated_helper_names:
        helper_names = ', '.join(str(name) for name in truncated_helper_names)
        mirror_disabled_warnings.append(
            'leaf rotation helper budget omitted mirrored helper partners for '
            f'[{helper_names}]. Mirror augmentation will neutralize these helpers.'
        )

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
        'mirror_disabled_joint_indices': mirror_disabled_joint_indices,
        'mirror_disabled_joint_names': mirror_disabled_joint_names,
        'mirror_disabled_warnings': mirror_disabled_warnings,
        'is_symmetric': bool(base_semantic_metadata['is_symmetric']),
    }


def resolve_mirrored_export_skeleton_metadata(object_cond, parents, offsets, joint_names):
    """Return export metadata that keeps old single-sided helpers attached to the mirrored leaf.

    Older processed datasets can contain a helper bone for only one side of a
    mirrored pair. After feature-space mirroring, the helper's recovered global
    position lands on the opposite side, but the original hierarchy still keeps
    it parented to the source-side leaf. For BVH export we can repair this by
    reparenting the helper to the mirrored source leaf and zeroing its offset.
    Helpers already marked mirror-disabled are left untouched because the mirror
    pipeline neutralizes them instead of reflecting them.
    """
    mirrored_parents = np.asarray(parents, dtype=np.int32).copy()
    mirrored_offsets = np.asarray(offsets).copy()
    mirrored_joint_names = list(joint_names)

    helper_joint_indices = [
        int(joint_index)
        for joint_index in list(object_cond.get('helper_joint_indices') or [])
    ]
    helper_source_leaf_indices = [
        int(joint_index)
        for joint_index in list(object_cond.get('helper_source_leaf_indices') or [])
    ]
    symmetry_partner_indices = np.asarray(
        object_cond.get('symmetry_partner_indices') or [],
        dtype=np.int32,
    )
    mirror_disabled_joint_indices = {
        int(joint_index)
        for joint_index in list(object_cond.get('mirror_disabled_joint_indices') or [])
    }
    if (
        len(helper_joint_indices) == 0
        or len(helper_joint_indices) != len(helper_source_leaf_indices)
        or symmetry_partner_indices.size == 0
    ):
        return mirrored_parents, mirrored_offsets, mirrored_joint_names

    for helper_index, source_index in zip(helper_joint_indices, helper_source_leaf_indices):
        if helper_index < 0 or helper_index >= len(mirrored_parents):
            continue
        if helper_index in mirror_disabled_joint_indices:
            continue
        if source_index < 0 or source_index >= len(symmetry_partner_indices):
            continue
        if helper_index < len(symmetry_partner_indices) and int(symmetry_partner_indices[helper_index]) >= 0:
            continue

        mirrored_source_index = int(symmetry_partner_indices[source_index])
        if mirrored_source_index < 0 or mirrored_source_index >= len(mirrored_parents):
            continue

        mirrored_parents[helper_index] = mirrored_source_index
        if helper_index < len(mirrored_offsets):
            mirrored_offsets[helper_index] = np.zeros_like(mirrored_offsets[helper_index])

        if helper_index < len(mirrored_joint_names) and mirrored_source_index < len(mirrored_joint_names):
            mirrored_source_name = str(mirrored_joint_names[mirrored_source_index] or '')
            if mirrored_source_name:
                mirrored_joint_names[helper_index] = canonical_name_for_bvh(
                    f'{mirrored_source_name}Helper',
                    mirrored_joint_names[helper_index],
                )

    return mirrored_parents, mirrored_offsets, mirrored_joint_names


################## Mirror & Neutralization #####################

def warn_mirror_disabled_subtrees(object_cond):
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
        _warn(f'{object_type}: {message}')


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


################## FK Helpers #####################

def coerce_single_orientation_quat(orientation_quat):
    if orientation_quat is None:
        raise ValueError(
            "orientation_quat must be precomputed from the reference T-pose and provided to downstream motion processing"
        )

    orientation_qs = getattr(orientation_quat, 'qs', orientation_quat)
    orientation_qs = np.asarray(orientation_qs, dtype=np.float64)
    if orientation_qs.ndim > 1:
        orientation_qs = orientation_qs[0]
    return Quaternions(orientation_qs.reshape(1, 4)).normalized()


def compute_rots_from_tpos(tpos_quats, dest_quats, parents):
    new_rots = dest_quats.copy()
    new_rots[:, 0] = new_rots[:, 0] * -tpos_quats[:, 0]
    cum_rots = tpos_quats.copy()
    for j, p in enumerate(parents[1:], start=1):
        cum_rots[:, j] = cum_rots[:, p] * tpos_quats[:, j]
        new_rots[:, j] = cum_rots[:, p] * dest_quats[:, j] * -tpos_quats[:, j] * -cum_rots[:, p]
    return new_rots


def solve_local_positions_for_target_global(
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

