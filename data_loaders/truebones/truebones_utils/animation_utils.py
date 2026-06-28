"""Animation processing & joint metadata utilities.

Lowest layer of the motion-processing pipeline. Handles FK transforms,
coordinate normalization, scaling, BVH export preparation, joint-name
canonicalization and leaf rotation helpers.
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
from data_loaders.truebones.truebones_utils.param_utils import HML_REF_AXIAL_BONE_LENGTH, HML_REF_MAX_SPAN, MAX_JOINTS, OBJECT_SUBSETS_DICT, SCALE_BODY_SPAN_BLEND_WEIGHT, VERTICAL_CLAMP_MIN_RATIO, VERTICAL_CLAMP_MAX_RATIO
from data_loaders.truebones.truebones_utils.skeleton_cropping import (
    select_cropped_joint_indices,
)

# Body-plan groups used by the vertical clamp; sourced from the species motion
# tags (winged == old FLYING, aquatic == old FISH) so they never drift.
FLYING = frozenset(OBJECT_SUBSETS_DICT['winged'])
FISH = frozenset(OBJECT_SUBSETS_DICT['aquatic'])
from .physics_joint_annotation import (
    build_semantic_metadata,
    normalize_joint_name,
    strip_joint_name_prefix,
    build_joint_embedding_texts,
    build_species_embedding_text,
    assert_species_tags_cover,
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
ROOT_XZ_STRIP_THRESHOLD = 0.6

# Loop detection judges the wrap-around gap (last frame -> first frame) against
# the clip's own frame-to-frame motion distribution. A high percentile gives a
# compact robust envelope without tying tolerance to skeleton size.
#
# A clip loops only when the endpoint gap fits inside that transition envelope
# and the translation root's accumulated XZ displacement returns to the start.
LOOP_DETECTION_GAP_RATIO = 2.2
LOOP_DETECTION_STEP_MIN = 0.02
LOOP_DETECTION_STEP_MAX = 0.08
LOOP_DETECTION_ROOT_XZ_TOLERANCE = 0.08


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
    semantic_metadata = build_semantic_metadata(joint_names, parents, offsets)
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
    object_cond['is_symmetric'] = semantic_metadata['is_symmetric']


def refresh_joint_metadata_in_cond_dict(cond_dict):
    if not isinstance(cond_dict, dict):
        return cond_dict

    for object_cond in cond_dict.values():
        if isinstance(object_cond, dict):
            refresh_joint_metadata_in_object_cond(object_cond)
    return cond_dict


def attach_t5_embeddings_to_cond(cond, save_dir, t5_name='t5-base', write_collision_report=True,
                                  t5_conditioner=None):

    if not cond:
        return

    inspection_dir = pjoin(save_dir, 'joint_name_inspection')
    os.makedirs(inspection_dir, exist_ok=True)

    embedding_texts_by_object = {}
    for object_type in sorted(cond):
        object_cond = cond[object_type]
        refresh_joint_metadata_in_object_cond(object_cond)
        embedding_texts = build_joint_embedding_texts(object_cond)
        embedding_texts_by_object[object_type] = embedding_texts

    object_types_to_encode = sorted(cond)
    joint_count = len(object_types_to_encode)

    if t5_conditioner is None:
        # Fast-fail before any encoding: the per-species descriptor has no fallback,
        # so a species missing from _SPECIES_TAGS must surface here.
        assert_species_tags_cover(cond.keys())
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Loading T5 model {t5_name} on {device.upper()} ...')
        from model.conditioners import T5Conditioner
        t5_conditioner = T5Conditioner(
            name=t5_name,
            finetune=False,
            word_dropout=0.0,
            normalize_text=False,
            device=device,
            autocast_dtype=None,
            local_files_only=True,
        )
    else:
        print(f'Using pre-loaded T5 conditioner ({t5_name}) ...')

    print(f'Encoding joint-name embeddings for {joint_count} object types ...')

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

    print(f'Encoding species embeddings for {joint_count} object types ...')
    with torch.no_grad():
        for object_type in object_types_to_encode:
            object_cond = cond[object_type]
            species_text = build_species_embedding_text(object_cond)
            species_tokens = t5_conditioner.tokenize_entries([species_text])
            species_emb = t5_conditioner(species_tokens).detach().cpu().numpy().astype(np.float32, copy=False)
            object_cond['species_emb'] = species_emb[0]
            object_cond['species_emb_meta'] = {
                't5_name': t5_name,
                'schema_version': JOINT_NAME_EMBEDDING_SCHEMA_VERSION,
                'embedding_dim': int(species_emb.shape[1]) if species_emb.ndim == 2 else 0,
                'embedding_text': species_text,
            }

    for object_type in sorted(cond):
        object_cond = cond[object_type]
        embedding_texts = embedding_texts_by_object[object_type]
        inspection_path = pjoin(inspection_dir, f'{object_type}.json')
        with open(inspection_path, 'w', encoding='utf-8') as inspection_file:
            json.dump(_build_joint_name_inspection_rows(object_cond, embedding_texts), inspection_file, indent=2)

    if write_collision_report:
        write_joint_name_collision_report(cond, save_dir)


################## Animation Transform Utilities #####################

def compute_motion_loop_diagnostics(positions, root_xz_velocity=None, translation_root_index=0):
    """Return loop diagnostics on the exact runtime boundary used by detect_motion_loop.

    The endpoint gap is compared against a robust upper envelope of the clip's
    own per-frame motion. When ``root_xz_velocity`` is provided, the translation
    root's accumulated XZ displacement must also close for the clip to count as
    a loop.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape[0] < 3:
        return {
            'wrap_gap': 0.0,
            'transition_envelope': 0.0,
            'effective_tolerance': 0.0,
            'root_xz_total_disp': 0.0,
            'root_xz_is_closed': True,
            'is_loop': False,
        }

    # wrap_gap: p75 of per-joint endpoint distance — robust against a single
    # outlier joint while still capturing the bulk of the discontinuity.
    wrap_gap = float(np.percentile(np.linalg.norm(positions[-1] - positions[0], axis=-1), 75))

    # Use only frames near the clip boundaries to estimate the "normal transition"
    # envelope. The wrap_gap measures the jump from last frame -> first frame, so
    # it should be compared against the typical motion amplitude at the clip edges
    # rather than the peak motion in the middle (e.g., a fast swing or stride).
    frame_steps = np.linalg.norm(np.diff(positions, axis=0), axis=-1)  # (T-1, J)
    boundary_ratio = 0.15  # consider the outer 15% at each end
    boundary_count = max(1, int(np.ceil((frame_steps.shape[0] - 1) * boundary_ratio)))
    boundary_steps = np.concatenate([
        frame_steps[:boundary_count],
        frame_steps[-boundary_count:],
    ], axis=0)
    transition_envelope = float(np.percentile(boundary_steps, 65.0))  # p65 — tighter than p75 wrap_gap
    effective_tolerance = min(
        max(
            LOOP_DETECTION_GAP_RATIO * transition_envelope,
            LOOP_DETECTION_STEP_MIN,
        ),
        LOOP_DETECTION_STEP_MAX,
    )

    root_xz_total_disp = 0.0
    root_xz_is_closed = True

    if root_xz_velocity is not None:
        velocity = np.asarray(root_xz_velocity, dtype=np.float64)
        if velocity.ndim != 3 or velocity.shape[2] < 3:
            raise ValueError(
                f"root_xz_velocity must have shape (T, J, 3), got {velocity.shape}"
            )

        translation_root_index = int(translation_root_index)
        if not 0 <= translation_root_index < velocity.shape[1]:
            raise ValueError(
                f"translation_root_index {translation_root_index} is out of bounds for {velocity.shape[1]} joints"
            )

        root_velocity = velocity[:, translation_root_index, :]
        if velocity.shape[0] == positions.shape[0]:
            root_velocity = root_velocity[:-1]
        elif velocity.shape[0] != positions.shape[0] - 1:
            raise ValueError(
                f"root_xz_velocity frame count must be T or T-1 relative to positions, "
                f"got {velocity.shape[0]} vs positions T={positions.shape[0]}"
            )

        root_xz_steps = np.linalg.norm(root_velocity[:, [0, 2]], axis=-1)
        root_xz_total_disp = float(np.linalg.norm(np.sum(root_velocity[:, [0, 2]], axis=0)))
        root_xz_is_closed = bool(root_xz_total_disp <= LOOP_DETECTION_ROOT_XZ_TOLERANCE)

    return {
        'wrap_gap': wrap_gap,
        'transition_envelope': float(transition_envelope),
        'effective_tolerance': float(effective_tolerance),
        'root_xz_total_disp': float(root_xz_total_disp),
        'root_xz_is_closed': bool(root_xz_is_closed),
        'is_loop': bool(wrap_gap <= effective_tolerance and root_xz_is_closed),
    }


def detect_motion_loop(positions, root_xz_velocity=None, translation_root_index=0):
    return compute_motion_loop_diagnostics(
        positions,
        root_xz_velocity=root_xz_velocity,
        translation_root_index=translation_root_index,
    )['is_loop']


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


def crop_animation_to_max_joints(anim, names, max_joints=MAX_JOINTS, *, context=None):
    """Crop ``anim``/``names`` to at most ``max_joints`` joints.

    Cropping always removes current leaves first. Deeper leaves are removed
    before shallower ones; at the same depth, shorter bones are trimmed before
    longer ones. Non-root joints whose rest-offset length exceeds the mean
    non-root bone length are preserved whenever possible and are only dropped as
    a last resort when the cap still cannot be met.

    Returns ``(anim, names, keep_indices)``. When no cropping is needed the
    inputs are returned unchanged with ``keep_indices=None``. When cropping
    happens a yellow [WARN] log line names the dropped joints so the affected
    clip/skeleton is easy to spot. The keep-set is a deterministic function of
    the parent topology and rest-pose offsets, so the rest-pose skeleton and
    every motion clip of the same character crop to the identical joint set.
    """
    parents = np.asarray(anim.parents, dtype=np.int64)
    n = int(parents.shape[0])
    names = list(names)
    if len(names) != n:
        raise ValueError(
            f"Expected {n} joint names to crop skeleton, got {len(names)}"
        )

    selection = select_cropped_joint_indices(parents, max_joints, offsets=anim.offsets)
    if selection is None:
        return anim, names, None
    keep_indices, removed_order = selection

    old_to_new = -np.ones((n,), dtype=np.int64)
    for new_index, old_index in enumerate(keep_indices):
        old_to_new[old_index] = new_index
    parent_dtype = getattr(anim.parents, 'dtype', np.int32)
    new_parents = np.array(
        [
            int(old_to_new[parents[old_index]]) if parents[old_index] >= 0 else -1
            for old_index in keep_indices
        ],
        dtype=parent_dtype,
    )

    keep_arr = np.asarray(keep_indices, dtype=np.int64)
    orient_count = len(anim.orients)
    if orient_count not in (0, n):
        raise ValueError(
            f"Expected 0 or {n} joint orients to crop skeleton, got {orient_count}"
        )
    new_orients = anim.orients.copy() if orient_count == 0 else anim.orients[keep_arr].copy()

    cropped = Animation(
        anim.rotations[:, keep_arr].copy(),
        anim.positions[:, keep_arr].copy(),
        new_orients,
        anim.offsets[keep_arr].copy(),
        new_parents,
    )
    new_names = [names[old_index] for old_index in keep_indices]

    removed_names = [names[old_index] for old_index in removed_order]
    label = f' for {context}' if context else ''
    _warn(f'skeleton exceeds MAX_JOINTS ({n} > {max_joints}){label}')
    return cropped, new_names, keep_indices


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


################## FK Helpers #####################

def coerce_single_orientation_quat(orientation_quat):
    if orientation_quat is None:
        raise ValueError(
            "orientation_quat must be precomputed from the reference rest pose and provided to downstream motion processing"
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

