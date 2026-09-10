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
from data_loaders.truebones.truebones_utils.param_utils import (
    DEGENERATE_BONE_LENGTH_RATIO,
    HML_REF_AXIAL_BONE_LENGTH,
    HML_REF_MAX_SPAN,
    MAX_JOINTS,
    PROP_SOCKET_BONE_LENGTH_RATIO,
    PROP_SOCKET_MAX_SUBTREE_JOINTS,
    ROOT_Y_MIN_HEIGHT,
    ROOT_Y_SOFT_CLAMP_KNEE,
    SCALE_BODY_SPAN_BLEND_WEIGHT,
    VERTICAL_CLAMP_MIN_RATIO,
    VERTICAL_CLAMP_MAX_RATIO,
)
from data_loaders.truebones.truebones_utils.skeleton_cropping import (
    select_cropped_joint_indices,
)
from data_loaders.truebones.truebones_utils.dataset_tags import (
    assert_species_tags_cover,
    dataset_tags,
)
from .physics_joint_annotation import (
    build_semantic_metadata,
    joint_name_is_non_anatomical,
    infer_species_joint_name_prefixes,
    joint_name_token_is_species,
    normalize_joint_name,
    strip_joint_name_prefix,
    build_joint_embedding_texts,
    build_species_embedding_text,
    JOINT_NAME_EMBEDDING_SCHEMA_VERSION,
    JOINT_NAME_EMBEDDING_SLIM,
)


################## Constants #####################

ANSI_YELLOW = '\033[93m'
ANSI_RESET = '\033[0m'


def _warn(msg: str):
    """Print a warning message in yellow."""
    print(f'{ANSI_YELLOW}[WARN] {msg}{ANSI_RESET}')


# Sustained one-directional translation-root XZ displacement (in HML-normalised
# units, where a body span is 1.389) above which a clip is re-seated in place.
#
# Measured on the TRANSPORT the clip carries in its own heading frame, never on
# its raw extent. An out-and-back excursion has no net transport however far it
# reaches, so a lunge, a dodge or a swing keeps its root motion verbatim; a gait
# that travels is nothing but transport and gets flattened however small each
# step is. Clips are centred on the effective translation root's initial XZ
# position before this is evaluated.
ROOT_XZ_DRIFT_THRESHOLD = 0.08

# Root XZ soft clamp, in HML-normalised units (a body span is 1.389).
#
# Travel is bounded, not gated: inside the knee nothing is touched at all, past
# it the excess is compressed smoothly, and the ceiling is an asymptote the path
# approaches but never reaches. This is what keeps a lunge, a dodge or a death
# slide recognisable at a magnitude the representation can carry -- the old
# extent gate answered the same question by zeroing the whole trajectory.
ROOT_XZ_SOFT_CLAMP_KNEE = 0.6
ROOT_XZ_SOFT_CLAMP_LIMIT = 0.8

# Locomotion's own extent bound, tighter than the soft clamp above and applied to
# EVERY locomotion clip, not only the ones that travelled -- that is what makes it
# an invariant of the group rather than of whether a clip happened to move. The
# knee sits inside a normal gait's range (post-detrend extent runs p50 0.013,
# p90 0.110, p95 0.159), so unlike the 0.6 ceiling it is touched routinely and must
# leave the cycle's shape alone: it scales the whole clip by ONE factor, not each
# frame's radius (see ``scale_root_xz_extent``).
ROOT_XZ_LOCOMOTION_KNEE = 0.1
ROOT_XZ_LOCOMOTION_LIMIT = 0.2


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


def _joint_disambiguation_tokens(raw_name, canonical_name, additional_prefixes=()):
    raw_value = str(raw_name or '')
    stripped_raw = strip_joint_name_prefix(raw_value, additional_prefixes)
    raw_tokens = normalize_joint_name(stripped_raw).split()
    canonical_tokens = normalize_joint_name(canonical_name).split()
    residual_tokens = _remove_token_counts(raw_tokens, Counter(canonical_tokens))
    residual_tokens = [token for token in residual_tokens if not joint_name_token_is_species(token)]
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


def _disambiguate_duplicate_canonical_names(
    raw_names,
    canonical_names,
    additional_prefixes=(),
):
    updated_names = list(canonical_names)
    grouped_indices = defaultdict(list)
    for joint_index, canonical_name in enumerate(canonical_names):
        grouped_indices[str(canonical_name)].append(joint_index)

    for canonical_name, indices in grouped_indices.items():
        raw_name_set = {str(raw_names[index]) for index in indices}
        if len(indices) <= 1 or len(raw_name_set) <= 1:
            continue

        residual_token_lists = [
            _joint_disambiguation_tokens(
                raw_names[index],
                canonical_name,
                additional_prefixes,
            )
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


def assign_canonical_joint_names(object_cond, joint_names, canonical_names):
    """Store the disambiguated canonical names plus their BVH-safe spellings.

    Both keys must derive from the *disambiguated* list. BVH bone names have to
    be unique, and canonical_name_for_bvh only strips punctuation, so deriving
    them from the raw canonicalizer output emits duplicate bones (e.g. Camel's
    front and back legs both exported as "LeftLeg01"). Single entry point so the
    preprocessing pipeline and the on-load refresh cannot drift apart.
    """
    species_prefixes = infer_species_joint_name_prefixes(
        joint_names,
        object_cond.get('species_name') or object_cond.get('object_type'),
    )
    disambiguated_names = _disambiguate_duplicate_canonical_names(
        joint_names,
        canonical_names,
        additional_prefixes=species_prefixes,
    )
    object_cond['canonical_joint_names'] = disambiguated_names
    object_cond['canonical_bvh_joint_names'] = [
        canonical_name_for_bvh(canonical_name, raw_name)
        for canonical_name, raw_name in zip(disambiguated_names, joint_names)
    ]


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
    semantic_metadata = build_semantic_metadata(
        joint_names,
        parents,
        offsets,
        species_name=object_cond.get('species_name') or object_cond.get('object_type'),
    )
    assign_canonical_joint_names(object_cond, joint_names, semantic_metadata['canonical_joint_names'])
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
        # so a species missing from species_tags.jsonl must surface here.
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
                'slim': bool(JOINT_NAME_EMBEDDING_SLIM),
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

    # cond keys are '<namespace>/<species>', which cannot go into a filename;
    # the file token degrades to the plain species name whenever it is unique.
    from .dataset_sources import build_species_file_tokens
    file_tokens = build_species_file_tokens(cond)
    for object_type in sorted(cond):
        object_cond = cond[object_type]
        embedding_texts = embedding_texts_by_object[object_type]
        inspection_path = pjoin(inspection_dir, f'{file_tokens[object_type]}.json')
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
    significant local position animation. This answers "where is THIS clip's
    root motion", which is not the same question as "which joint is this
    species' transport control" -- see :func:`select_transport_carrier`.
    """
    frame_count = int(anim.positions.shape[0]) if anim.positions.ndim >= 3 else 0
    min_active_frames = max(3, int(np.ceil(frame_count * 0.2))) if frame_count > 0 else 0
    candidate_chain = _translation_root_candidate_chain(anim.parents, max_depth=max_depth)
    first_candidate = int(candidate_chain[0]) if candidate_chain else 0
    fallback_joint = first_candidate

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
        if fallback_joint == first_candidate and j != first_candidate and active_frames >= 2:
            fallback_joint = j

    return int(fallback_joint)


# A chain joint counts as carrying a clip's transport when its own horizontal
# travel is at least this share of the largest travel anywhere on that chain.
# Rigs stack several near-root control joints and animators do not always use the
# same one, so the test has to be "is this where the motion is", not "does this
# move at all": Crow's Spine drifts 0.026 while its Pelvis flies 0.809, and a
# rule that counted any motion would read the whole species' trajectory off a
# body joint.
ROOT_TRANSPORT_CARRIER_SHARE = 0.5

# ...and at least this much travel outright, in HML-normalised units. The same
# magnitude as the flatten threshold, for the same reason: below it nothing
# downstream reacts to the travel at all, so it cannot be worth re-rooting a
# species over. Idle sway and vertical bob on a spine joint live down here.
ROOT_TRANSPORT_MIN_TRAVEL = 0.08


def chain_xz_travel(anim, max_depth=5):
    """Return ``{joint: horizontal travel}`` for the translation-root candidates.

    Travel is measured the way :func:`find_translation_root` measures activity --
    local position, distance from the first frame -- but in XZ only, because the
    root trajectory, the gait flattener and the soft clamp are all horizontal.
    A joint that only bobs vertically carries no transport.
    """
    chain = _translation_root_candidate_chain(anim.parents, max_depth=max_depth)
    positions = np.asarray(anim.positions, dtype=np.float64)
    travel = {}
    for joint in chain:
        xz = positions[:, joint][:, [0, 2]]
        travel[int(joint)] = (
            float(np.linalg.norm(xz - xz[0:1], axis=1).max()) if xz.shape[0] else 0.0
        )
    return travel


def select_transport_carrier(chain_travel,
                             share=ROOT_TRANSPORT_CARRIER_SHARE,
                             min_travel=ROOT_TRANSPORT_MIN_TRAVEL):
    """Return the deepest chain joint that carries the horizontal travel, or None.

    ``None`` means this clip never goes anywhere and has no opinion about where
    its species keeps transport.

    DEEPEST, not most active: a joint's global position already contains every
    ancestor's translation, so a root at or below the carrier sees all of them,
    while a root above it sees none of that clip's travel -- it lands in a
    descendant's RIC channel instead, where the flattener, the clamp and both
    validators are blind to it. That is why the choice is forced rather than a
    vote, and why the two filters above it have to do the work of deciding what
    counts as travel in the first place.
    """
    if not chain_travel:
        return None
    peak = max(chain_travel.values())
    if peak < min_travel:
        return None
    threshold = max(share * peak, min_travel)
    carriers = [joint for joint, travel in chain_travel.items() if travel >= threshold]
    return max(carriers) if carriers else None


def rest_pose_animation(anim):
    """The one-frame rest pose of ``anim``: its orients seated on its offsets.

    ``positions_global`` reads only the per-frame ``rotations``/``positions``, so
    FK'ing a multi-frame clip never yields its rest pose -- reduce it here first.
    """
    rest_rotations = np.asarray(anim.orients.qs, dtype=np.float64)
    offsets = np.asarray(anim.offsets, dtype=np.float64)
    if rest_rotations.shape[0] != offsets.shape[0]:
        raise ValueError(
            f"Animation has {offsets.shape[0]} offsets but "
            f"{rest_rotations.shape[0]} rest rotations"
        )
    return Animation(
        Quaternions(rest_rotations[None].copy()),
        offsets[None].copy(),
        anim.orients.copy(),
        offsets.copy(),
        np.asarray(anim.parents, dtype=np.int32).copy(),
    )


def _get_reference_body_length(anim):
    """Character size from the rest-pose joint span.

    ``anim`` is a multi-frame clip; reduce it to its rest pose first so the band
    this feeds is a species property, not tied to the pose the clip opens with.
    """
    return max_joint_span(positions_global(rest_pose_animation(anim))[0])


def _excursion_scale(extent, min_h, max_h):
    """The factor an excursion reaching ``extent`` is scaled by.

    ``min_h`` is the knee and ``max_h`` the asymptote of the shared hyperbola, so
    the peak lands on ``soft_clamp_extent`` and everything below the knee is left
    where it is. ``max_h`` used to be the target rather than the asymptote, which
    made it the reported height of EVERY clip that reached it -- 304 of the 582
    winged clips, so a bird's hop and a dragon's climb came out at the same
    number. Ordering across clips is what that cost, and what this returns.

    The excess drops out of the ratio as ``w / (e + w)``, so a clip that barely
    clears the knee is scaled by ~1 and the map is continuous there. Cancellation
    in the numerator only bites once ``e`` is down around 1e-13, and the shift it
    could cause is itself bounded by ``e`` -- there is nothing left to protect.
    """
    excess = extent - min_h
    return (float(soft_clamp_extent(extent, min_h, max_h)) - min_h) / excess


def _compress_positive_excursion(values, min_h, max_h):
    """Scale the part of ``values`` above ``min_h`` so its peak approaches ``max_h``.

    ONE factor for the whole clip, taken from the clip's own peak, exactly as
    ``scale_root_xz_extent`` does and for the same reason: past the knee the
    excursion IS the flight, so a per-frame map would bend the top of every arc
    harder than its base and change the shape of the climb rather than its size.
    The knee is reached routinely here -- 71.3% of winged clips peak above it.
    """
    peak = float(values.max())
    if peak <= min_h or max_h <= min_h:
        return values, False

    scale = _excursion_scale(peak, min_h, max_h)
    compressed = values.copy()
    mask = compressed > min_h
    compressed[mask] = min_h + (compressed[mask] - min_h) * scale
    return compressed, True


def _compress_negative_excursion(values, min_h, max_h):
    """The mirror of :func:`_compress_positive_excursion` on downward depth."""
    depth = -float(values.min())
    if depth <= min_h or max_h <= min_h:
        return values, False

    scale = _excursion_scale(depth, min_h, max_h)
    compressed = values.copy()
    mask = compressed < -min_h
    compressed[mask] = -min_h - ((-compressed[mask]) - min_h) * scale
    return compressed, True


def _compress_below_negative_band(values, min_h, max_h):
    """Compress values below negative thresholds derived from positive ratios."""
    return _compress_negative_excursion(values, abs(min_h), abs(max_h))


def _soft_clamp_min_height(values, knee, min_height):
    """Bound the root's downward excursion by ``min_height`` without a hard floor.

    The root-XZ hyperbola (:func:`soft_clamp_extent`) mirrored onto depth: heights
    above ``knee`` come back bit-for-bit, the descent past it is compressed with a
    continuous value and slope, order is preserved, and ``min_height`` is an
    asymptote rather than a value.

    A hard ``np.maximum`` floor has none of the last three. It mapped every frame
    past -0.5 onto exactly -0.5, so what a dive, a fall or a burrow left in the
    data was a pinned plateau, not a descent -- 86 shipped clips carried one, 19 of
    them for 8+ consecutive frames, and Pirrana_MidSwim was a dead constant for all
    97 of its frames. It also fought the aquatic band directly: that band puts
    swim depth in [-minH, -maxH], which at the HML reference span is
    [-0.417, -0.694], and the floor then sheared the deeper half of it onto one
    height.
    """
    knee_depth = abs(float(knee))
    if float(values.min()) >= -knee_depth:
        return values, False
    # soft_clamp_extent is identity at and below its knee, negative radii included,
    # so every frame shallower than the knee (and every positive height) survives
    # the round trip through the negation exactly.
    return -soft_clamp_extent(-values, knee_depth, abs(float(min_height))), True


def clamp_vertical_trajectory(
    processed_anim,
    object_type,
    min_ratio=VERTICAL_CLAMP_MIN_RATIO,
    max_ratio=VERTICAL_CLAMP_MAX_RATIO,
    root_y_min_height=ROOT_Y_MIN_HEIGHT,
    root_y_soft_clamp_knee=ROOT_Y_SOFT_CLAMP_KNEE,
    translation_root_index=None,
):
    """Constrain the processed translation-root vertical trajectory.

    What this compresses is the root's ABSOLUTE height, not its excursion: once
    the peak passes ``min_ratio`` of the body span, everything above ``min_ratio``
    is scaled by one factor that sends the peak toward ``max_ratio`` without ever
    reaching it. The band is calibrated on flight, where an unbounded climb is the
    thing worth bounding.

    ``drifting`` deliberately does NOT take this branch, though it too leaves the
    ground. A drifting species holds a near-constant hover height that is a trait
    of the species, and that height routinely sits above ``max_ratio`` (a hover
    robot's root rides at 2-3 body spans while a walking biped's sits at 0.1-0.4),
    so the flight band would crush the hover back down to walking height -- and by
    a factor that depends on each clip's own peak, which would make one species'
    hover height differ from clip to clip. Its vertical range in level travel is
    already as small as level flight's, so there is nothing here worth clamping;
    it keeps only the root-Y lower bound, the same as a ground species.

    Aquatic species use the same positive height clamp and also apply the same
    ratios with a negative sign so their downward swim depth is compressed into
    [-maxH, -minH]. Every species then gets the root-Y lower bound, which is a
    soft clamp, not a floor: it leaves everything above its knee alone and
    compresses the descent below it into ``(root_y_min_height, knee]`` (see
    :func:`_soft_clamp_min_height`). It runs last, so an aquatic dive whose band
    reaches past the bound comes out as a monotone descent -- compressed further
    than the band left it, but never the flat plateau a hard floor cut.

    ``translation_root_index`` is the species' frozen root. Without it this falls
    back to per-clip detection, which is right for a bare rest pose or a raw
    retarget source but wrong inside preprocessing: a species whose clips author
    transport on different joints (see :func:`select_transport_carrier`) would
    get its height read off one joint and its trajectory off another, and the
    two differ by the bone between them plus whatever that bone animates.
    """
    if translation_root_index is None:
        trans_root = find_translation_root(processed_anim)
    else:
        trans_root = int(translation_root_index)
    global_pos = positions_global(processed_anim)
    world_y = global_pos[:, trans_root, 1]

    clamped_world_y = world_y.copy()
    changed = False
    # object_subset_for accepts a bare species name or a canonical
    # '<namespace>/<species>' key, unlike a raw subset_members membership test.
    object_subset = dataset_tags().object_subset_for(object_type)
    if object_subset == 'winged':
        body_length = _get_reference_body_length(processed_anim)
        min_h = body_length * min_ratio
        max_h = body_length * max_ratio
        clamped_world_y, changed = _compress_positive_excursion(clamped_world_y, min_h, max_h)
    elif object_subset == 'aquatic':
        body_length = _get_reference_body_length(processed_anim)
        positive_min_h = body_length * min_ratio
        positive_max_h = body_length * max_ratio
        clamped_world_y, changed_pos = _compress_positive_excursion(
            clamped_world_y,
            positive_min_h,
            positive_max_h,
        )
        negative_min_h = -positive_min_h
        negative_max_h = -positive_max_h
        clamped_world_y, changed_neg = _compress_below_negative_band(
            clamped_world_y,
            negative_min_h,
            negative_max_h,
        )
        changed = changed_pos or changed_neg
    else:
        clamped_world_y = world_y.copy()

    clamped_world_y, changed_floor = _soft_clamp_min_height(
        clamped_world_y,
        root_y_soft_clamp_knee,
        root_y_min_height,
    )
    changed = changed or changed_floor

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
def move_xz_to_origin(anim, root_xz_center=None, translation_root_index=None):
    if root_xz_center is None:
        root_xz_center = _get_translation_root_initial_xz(
            anim,
            translation_root_index=translation_root_index,
        )
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


def root_xz_trajectory(anim, translation_root_index):
    """Return the effective translation root's ``(T, 2)`` world XZ path."""
    global_pos = positions_global(anim)
    return np.asarray(global_pos[:, translation_root_index][:, [0, 2]], dtype=np.float64)


def root_xz_heading(anim, translation_root_index):
    """Return the translation root's per-frame world heading, unwrapped, in radians.

    The angle of the root's own forward axis in the XZ plane. ``anim`` is
    canonicalised to face +Z by ``rotate_to_hml_orientation`` before it ever gets
    here, so the forward axis is +Z and no per-species orientation enters.

    Unwrapped, so a clip that turns past +/-pi reads as one continuous ramp
    rather than a jump -- the ramp is the whole point, it is what a straight-line
    fit can follow and a raw angle cannot.

    ``anim`` is expected to have been through ``collapse_translation_root_chain``
    already, which seats the inert wrapper nodes onto the root; the root's world
    rotation then carries the wrapper's rotation too, which is where some rigs
    keep the turn.
    """
    rotations = rotations_global(anim)[:, int(translation_root_index)]
    forward = rotations * np.array([0.0, 0.0, 1.0])
    return np.unwrap(np.arctan2(forward[:, 0], forward[:, 2]))


def _xz_rotation(angle):
    """Return ``(T, 2, 2)`` rotations by ``angle`` about +Y, acting on ``(x, z)``."""
    angle = np.asarray(angle, dtype=np.float64)
    cos, sin = np.cos(angle), np.sin(angle)
    return np.stack([np.stack([cos, sin], -1), np.stack([-sin, cos], -1)], -2)


def _detrend_frame(heading):
    """Return the per-frame frame angle to detrend in.

    A straight ramp between the heading's endpoints, so the frame turns by the
    clip's NET turn and nothing else: a yaw that wanders out and comes back lands
    where it started and cannot bend it, however that wander was shaped, while a
    clip that really turns bends it in full. A least-squares line through the
    whole heading cannot say that -- it weights every frame, so an asymmetric
    wobble tilts it (MB_TigerDrago_RunJump travels dead straight under 2.68 rad of
    pelvis yaw for a net of 0.21, and the fitted line invented 0.42 of lateral
    travel).

    The heading is the right signal and the path is not: a 180-degree turning gait
    and an out-and-back lunge trace nearly the same path, and only the body's
    orientation says which is which. Fitting to the path's velocity direction
    invents 0.76 of transport on a lunge whose net displacement is zero.

    The frame reads the heading alone, never the path, which is what lets the
    dataset validator re-derive it from the stored features: a constant offset
    between the two headings shifts both endpoints alike and cancels. See
    docs/root_xz_motion_refactor.md §2.1 for the measured comparison against the
    constant-frame and least-squares alternatives.
    """
    heading = np.asarray(heading, dtype=np.float64).reshape(-1)
    n_frames = heading.shape[0]
    if n_frames < 2:
        return heading.copy()
    steps = np.arange(n_frames, dtype=np.float64) / (n_frames - 1)
    return heading[0] + (heading[-1] - heading[0]) * steps


def _frame_correction(traj, frame):
    """Return the accumulated transport of ``traj`` measured in ``frame``."""
    local_step = np.einsum('tij,tj->ti', _xz_rotation(-frame), np.diff(traj, axis=0))
    transport = np.einsum('tij,j->ti', _xz_rotation(frame), local_step.mean(axis=0))
    return np.concatenate([np.zeros((1, 2)), np.cumsum(transport, axis=0)], axis=0)


def root_xz_drift_correction(traj, heading):
    """Return ``(correction, drift)`` for a ``(T, 2)`` root XZ path.

    The travel is removed IN THE ROOT'S OWN HEADING FRAME, and that frame is the
    whole of it. In world space a turning gait's velocity direction rotates with
    the character, so there is no straight trend to subtract and no low-order
    curve that fits one either: an end-to-end line fit leaves the arc's sagitta
    behind, which on the shipped data reached 0.371 for MB_Unka_GlideLeft and
    0.732 for IAC_Cavewoman_RunTurnRight -- 27% and 53% of a body span of
    residual travel, turning as it went.

    Rotated into the heading frame the same motion is a near-constant forward
    speed with the within-cycle surge and sway riding on it, and removing a
    constant is exactly what the old world-space line fit was already doing --
    just in the wrong frame. So the arithmetic is: rotate the frame-to-frame
    displacement by the negated frame angle, subtract its mean, rotate back,
    re-integrate from frame 0.

    The root's heading is only a proxy for the travel direction, so the frame is
    read from its NET turn alone (``_detrend_frame``): a pelvis that yaws through
    the stride and comes back cannot bend it.

    ``correction`` is the accumulated transport, so ``traj - correction`` keeps
    frame 0 where the pipeline centred it and keeps everything the transport
    could not explain -- the surge and sway that make a gait read as a gait
    rather than as a rig welded to the floor.

    ``drift`` is where that transport ENDS UP, not how far it ever reached. Only
    the endpoint separates "went and stayed" from "went and came back", and both
    of those spend most of the clip away from the origin, so no statistic over
    time can tell them apart. An out-and-back lunge reverses in the heading frame
    too and nets out to nothing, so it is kept whatever its amplitude.
    """
    traj = np.asarray(traj, dtype=np.float64)
    if traj.shape[0] < 2:
        return np.zeros_like(traj), 0.0
    frame = _detrend_frame(np.asarray(heading, dtype=np.float64)[:-1])
    correction = _frame_correction(traj, frame)
    return correction, float(np.linalg.norm(correction[-1]))


def flatten_root_xz_drift(traj, heading):
    """Return ``(flattened_traj, drift)`` for a root XZ path.

    The single entry point the pipeline and the dataset validator share, so both
    answer "does this clip travel" with the same arithmetic. ``heading`` is the
    translation root's per-frame world yaw, from ``root_xz_heading``; the frame
    it is detrended in turns by that heading's net turn (see
    ``root_xz_drift_correction``).

    It removes the travel and nothing else. Bounding what is left is a separate
    step the caller owns -- ``scale_root_xz_extent`` for a locomotion clip, the
    dataset-wide ``soft_clamp_root_xz`` for every clip -- and it stays out of here
    because the validator calls this to MEASURE a stored clip.
    """
    correction, drift = root_xz_drift_correction(traj, heading)
    return np.asarray(traj, dtype=np.float64) - correction, drift


def soft_clamp_extent(radius, knee=ROOT_XZ_SOFT_CLAMP_KNEE,
                      limit=ROOT_XZ_SOFT_CLAMP_LIMIT):
    """Return ``radius`` compressed into ``[0, limit)`` past ``knee``.

    ``g(r) = r`` up to the knee, then ``limit - w**2 / (r - knee + w)`` with
    ``w = limit - knee``. Four things are needed at once and this hyperbola is
    the simplest form that has all of them:

    * identity below the knee, so a clip that already stays near the origin is
      bit-for-bit untouched;
    * continuous VALUE AND SLOPE at the knee (``g'(knee) = 1``), so nothing in
      the dataset has a velocity step at 0.6 for the model to learn as a feature;
    * strictly increasing, so ordering survives -- a bigger lunge still reads as
      a bigger lunge, which a hard clamp destroys by mapping every excursion past
      the ceiling onto the same value;
    * ``limit`` as an asymptote rather than a value, so the bound is strict and
      the far tail decelerates smoothly instead of hitting a wall.

    An exponential knee has the same four properties on paper and loses the third
    one in float64: past about 34 knee-widths ``limit - w * exp(...)`` rounds to
    ``limit`` exactly, and the deepest excursions in the shipped data (up to 4.79)
    would land inside a 1e-3 band of each other. The hyperbola decays as ``1/r``
    and keeps them apart.

    With the shipped knee/limit: ``g(0.8) = 0.700``, ``g(1.2) = 0.750``,
    ``g(3.0) = 0.785``, ``g(10.0) = 0.796``.
    """
    radius = np.asarray(radius, dtype=np.float64)
    width = float(limit) - float(knee)
    if width <= 0.0:
        return np.minimum(radius, float(limit))
    excess = np.maximum(radius - float(knee), 0.0) + width
    return np.where(radius > knee, float(limit) - width * width / excess, radius)


def soft_clamp_root_xz(traj, knee=ROOT_XZ_SOFT_CLAMP_KNEE,
                       limit=ROOT_XZ_SOFT_CLAMP_LIMIT):
    """Return a ``(T, 2)`` root XZ path with its distance from the origin bounded.

    Applied per frame on the radius, so the map is a fixed function of position:
    the parts of the clip that stay within the knee are preserved exactly, and
    only the frames that reach past it are compressed. Scaling the whole
    trajectory by one factor instead would shrink a clip's near-origin footwork
    in proportion to an excursion elsewhere in the clip, and would let a single
    outlier frame resize everything around it.

    Direction is preserved frame by frame, so a closed path stays closed and
    frame 0 -- which the pipeline centred on the origin -- stays there.
    """
    traj = np.asarray(traj, dtype=np.float64)
    if traj.shape[0] == 0:
        return traj.copy()
    radius = np.linalg.norm(traj, axis=1)
    scale = np.ones_like(radius)
    over = radius > float(knee)
    if np.any(over):
        scale[over] = soft_clamp_extent(radius[over], knee, limit) / radius[over]
    return traj * scale[:, None]


def scale_root_xz_extent(traj, knee=ROOT_XZ_LOCOMOTION_KNEE,
                         limit=ROOT_XZ_LOCOMOTION_LIMIT):
    """Return a ``(T, 2)`` root XZ path scaled by ONE factor into ``[0, limit)``.

    The bound every locomotion clip is held to, applied after the detrend. It
    reuses the hyperbola of ``soft_clamp_extent`` -- identity below the knee, a
    strict asymptote at the limit, order-preserving in between -- but evaluates it
    once on the clip's own extent and scales the whole path by that ratio.

    The single factor is the deliberate difference from ``soft_clamp_root_xz``.
    What a detrend leaves behind IS the gait cycle, spread over the whole clip, so
    a per-frame radial map would compress the far half of every stride harder than
    the near half and change the SHAPE of the surge -- and this knee, unlike the
    0.6 ceiling, is reached routinely. One factor changes only its size.

    Frame 0 sits at the origin, so scaling keeps it there and keeps a closed path
    closed.
    """
    traj = np.asarray(traj, dtype=np.float64)
    if traj.shape[0] == 0:
        return traj.copy()
    extent = float(np.linalg.norm(traj, axis=1).max())
    if extent <= float(knee):
        return traj.copy()
    return traj * (float(soft_clamp_extent(extent, knee, limit)) / extent)


# A joint counts as carrying transport when its world XZ moves at all. In
# HML-normalised units (body span 1.389) this is 0.36% of a span -- authoring
# noise on a joint that is meant to sit still stays well under it.
TRANSPORT_CARRIER_EPS = 5e-3


def translation_root_ancestor_chain(parents, translation_root_index):
    """Return the joints from the hierarchy root down to the translation root."""
    parents = np.asarray(parents, dtype=np.int64).reshape(-1)
    chain = []
    joint = int(translation_root_index)
    seen = set()
    while 0 <= joint < parents.shape[0] and joint not in seen:
        seen.add(joint)
        chain.append(joint)
        joint = int(parents[joint])
    chain.reverse()
    return chain


def collapse_translation_root_chain(anim, translation_root_index):
    """Seat the joints above the translation root rigidly onto it.

    The joints between the hierarchy root and the effective root are inert
    control nodes -- ``Cg``, ``Ctrl``, ``All``, a bare ``Root`` -- that exist to
    hold the character, not to move relative to it. Nothing hangs off them but
    the chain itself, and the transport was measured to live at or below the
    effective root, so their offset from it carries no motion of its own.

    Left alone it carries two artifacts instead, and both land in RIC channels
    the model has to account for:

    * an arbitrary per-clip CONSTANT. Centring subtracts the effective root's
      initial XZ from joint 0, so a wrapper ends up at minus wherever the
      animator happened to park the character: ``MB_TigerDrago_DogdeLeftG``
      seats its ``Cg`` 1.546 from the origin, ``Run`` 0.193, ``Idle`` 0.102,
      ``GetHitR`` 0.000 -- same rig, same species, four different answers.
    * the NEGATED root trajectory, whenever the wrapper stands still while the
      root walks away from it (``Horse_Attack`` 0.718, ``Pirrana_Jump2`` 0.783).

    This rewrites the chain so every joint on it sits on its rest offset and the
    hierarchy root carries the whole trajectory. The effective root and its
    entire subtree keep their world positions EXACTLY -- only the inert joints
    move -- so pose, foot contacts and the root trajectory are untouched, and
    what is left in the wrapper's RIC is the rest geometry rather than an
    accident of authoring. Idempotent: a second pass finds the chain already
    seated and reproduces the same positions.
    """
    chain = translation_root_ancestor_chain(anim.parents, translation_root_index)
    if len(chain) <= 1:
        return anim

    parents = np.asarray(anim.parents, dtype=np.int64).reshape(-1)
    # Only safe while nothing but the chain hangs off these joints. Every root
    # this pipeline picks comes off the unbranched candidate chain, so this is a
    # guard against a hand-set or stale index, not a case that occurs.
    for joint in chain[:-1]:
        if int(np.count_nonzero(parents == joint)) != 1:
            return anim

    target = np.asarray(
        positions_global(anim)[:, translation_root_index], dtype=np.float64
    )

    seated_positions = anim.positions.copy()
    for joint in chain[1:]:
        seated_positions[:, joint] = anim.offsets[joint]
    seated_positions[:, chain[0]] = 0.0

    # Joint 0's local position IS its world position and adds straight through FK
    # to every descendant, so one probe gives the constant to solve against.
    probe = Animation(
        anim.rotations, seated_positions, anim.orients, anim.offsets, anim.parents
    )
    base = np.asarray(
        positions_global(probe)[:, translation_root_index], dtype=np.float64
    )
    seated_positions[:, chain[0]] = target - base

    return Animation(
        anim.rotations.copy(),
        seated_positions,
        anim.orients.copy(),
        anim.offsets.copy(),
        anim.parents.copy(),
    )


def _transport_carrier_index(anim, translation_root_index, global_pos=None,
                             eps=TRANSPORT_CARRIER_EPS):
    """Return the joint a root XZ correction has to be applied to.

    The highest joint on the hierarchy-root-to-translation-root chain that is not
    static in world XZ. Everything below it -- the translation root included --
    then moves rigidly with it, so the correction lands exactly on the
    translation root without inventing relative motion between joints that
    travelled together in the source.

    Both shapes occur in the data and they need opposite answers:

    * Horse, Jaws, Bear, Crow, Pirrana: a wrapper joint sits still at the origin
      while the effective root walks away from it. Shifting the wrapper would
      make a static joint travel, so the correction belongs on the root itself.
    * Dog, Dog-2: joint 0 (Hips) carries the travel and the effective root
      Spine0 rides along rigidly. Re-seating only Spine0 leaves Hips travelling
      and tears the two apart -- measured at 5.27 and 8.68 of fabricated
      divergence on Dog_Running and Dog-2_RunFast, all of it landing in Hips'
      RIC channel, which is what "the body is in place but Hips still slides"
      looks like from the outside.
    """
    chain = translation_root_ancestor_chain(anim.parents, translation_root_index)
    if len(chain) <= 1:
        return int(translation_root_index)
    if global_pos is None:
        global_pos = positions_global(anim)
    for joint in chain:
        xz = np.asarray(global_pos[:, joint][:, [0, 2]], dtype=np.float64)
        if float(np.ptp(xz, axis=0).max()) > eps:
            return int(joint)
    return int(translation_root_index)


def set_translation_root_xz(anim, translation_root_index, target_xz):
    """Return the animation with the effective root's world XZ set to ``target_xz``.

    The edit is applied to whichever joint actually carries the transport (see
    :func:`_transport_carrier_index`), never unconditionally to joint 0 and never
    unconditionally to the translation root. Pushing every rig's correction up to
    joint 0 would make a static wrapper slide backwards; keeping every rig's on
    the translation root tears a travelling ancestor away from it.
    """
    target_xz = np.asarray(target_xz, dtype=np.float64)
    global_pos = positions_global(anim)
    root_xz = np.asarray(global_pos[:, translation_root_index][:, [0, 2]], dtype=np.float64)
    if target_xz.shape != root_xz.shape:
        raise ValueError(
            f"target_xz must have shape {root_xz.shape}, got {target_xz.shape}"
        )
    delta = target_xz - root_xz
    if np.max(np.abs(delta)) <= 1e-8:
        return anim

    carrier = _transport_carrier_index(anim, translation_root_index, global_pos=global_pos)

    new_positions = anim.positions.copy()
    if anim.parents[carrier] < 0:
        new_positions[:, carrier, 0] += delta[:, 0]
        new_positions[:, carrier, 2] += delta[:, 1]
    else:
        global_rots = rotations_global(anim)
        parent_index = anim.parents[carrier]
        parent_global_pos = global_pos[:, parent_index]
        parent_global_rots = global_rots[:, parent_index]
        # The carrier takes the same delta the translation root needs, so the
        # root lands on the target exactly and the two stay rigidly linked.
        desired_global = global_pos[:, carrier].copy()
        desired_global[:, 0] += delta[:, 0]
        desired_global[:, 2] += delta[:, 1]
        new_positions[:, carrier] = (-parent_global_rots) * (desired_global - parent_global_pos)

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


def reindex_animation_to_kept_joints(anim, names, keep_indices):
    """Rebuild ``anim``/``names`` over ``keep_indices``, remapping the parents.

    ``keep_indices`` must be ascending and closed under parenthood -- a kept
    joint's parent is kept too -- which is what makes the remapped hierarchy a
    valid tree. Both callers guarantee it: cropping only ever peels current
    leaves, and the prop-socket filter drops whole subtrees.
    """
    parents = np.asarray(anim.parents, dtype=np.int64)
    n = int(parents.shape[0])
    keep_set = set(int(index) for index in keep_indices)
    for old_index in keep_indices:
        parent_index = int(parents[old_index])
        # A dropped parent would remap to -1 and silently promote its child to a
        # second root, so reject the caller's keep-set instead of the tree.
        if parent_index >= 0 and parent_index not in keep_set:
            raise ValueError(
                f"joint {old_index} is kept but its parent {parent_index} was dropped"
            )

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
            f"Expected 0 or {n} joint orients to reindex skeleton, got {orient_count}"
        )
    new_orients = anim.orients.copy() if orient_count == 0 else anim.orients[keep_arr].copy()

    reindexed = Animation(
        anim.rotations[:, keep_arr].copy(),
        anim.positions[:, keep_arr].copy(),
        new_orients,
        anim.offsets[keep_arr].copy(),
        new_parents,
    )
    return reindexed, [names[old_index] for old_index in keep_indices]


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

    cropped, new_names = reindex_animation_to_kept_joints(anim, names, keep_indices)

    removed_names = [names[old_index] for old_index in removed_order]
    label = f' for {context}' if context else ''
    _warn(f'skeleton exceeds MAX_JOINTS ({n} > {max_joints}){label}')
    return cropped, new_names, keep_indices


def promote_translation_root_to_hierarchy_root(anim, names, depth, *, context=None):
    """Drop ``depth`` wrapper joints above the effective root, folding them into it.

    The joints between the hierarchy root and the measured transport carrier are
    inert control nodes -- ``Cg``, ``Ctrl``, ``All``, and rigs whose wrapper is
    merely NAMED ``Hips`` / ``Root`` / ``Body``. Their offset from the real root
    carries no motion (that is what the carrier measurement establishes), so all
    they contribute to the model is a joint token whose whole 12-dim feature
    vector is a per-species constant, plus a second meaning for "joint 0".

    Each step folds the dropped joint's rotation, offset and animated translation
    into its child (``promote_root_once``), so every remaining joint keeps its
    world transform exactly and the character's yaw -- which 9 of the 34 wrapper
    species author on the wrapper itself, ``SabreToothTiger_180RIght`` and
    ``KI_Human_RunTurn02Right`` among them -- survives on the real root.

    Deliberately NOT done in the FBX/BVH loader. That layer is a conservative
    fallback that has to guess from names and offsets, and it cannot know where
    the transport is: the species root is only decided after phase 1 has measured
    every clip. Returns ``(anim, names, keep_indices)``, inputs unchanged and
    ``keep_indices=None`` when ``depth`` is 0.
    """
    depth = int(depth)
    if depth <= 0:
        return anim, list(names), None

    names = list(names)
    parents = np.asarray(anim.parents, dtype=np.int64)
    joint_count = int(parents.shape[0])
    if len(names) != joint_count:
        raise ValueError(
            f"Expected {joint_count} joint names to promote the translation root, "
            f"got {len(names)}"
        )
    if depth >= joint_count:
        raise ValueError(
            f"Cannot promote past the whole skeleton: depth {depth} of {joint_count} joints"
        )

    from motion_lib.root_collapse import promote_root_once

    working_names = names
    working_parents = np.asarray(anim.parents, dtype=np.int32)
    working_offsets = np.asarray(anim.offsets, dtype=np.float64)
    working_rotations = np.asarray(anim.rotations.qs, dtype=np.float64)
    working_positions = np.asarray(anim.positions, dtype=np.float64)
    working_orients = anim.orients

    for step in range(depth):
        if int(np.count_nonzero(working_parents == 0)) != 1:
            where = f" for {context}" if context else ""
            raise ValueError(
                f"Cannot promote the translation root{where}: joint "
                f"'{working_names[0]}' has more than one child, so dropping it "
                f"would move something other than the root chain"
            )
        (
            working_names,
            working_parents,
            working_offsets,
            working_rotations,
            working_positions,
            working_orients,
        ) = promote_root_once(
            working_names,
            working_parents,
            working_offsets,
            working_rotations,
            working_positions,
            working_orients,
        )

    promoted = Animation(
        Quaternions(working_rotations),
        working_positions,
        working_orients,
        working_offsets,
        np.asarray(working_parents, dtype=anim.parents.dtype),
    )
    return promoted, working_names, list(range(depth, joint_count))


def drop_prop_socket_joints(anim, names, *, drop_names=None, context=None):
    """Drop parked prop/weapon socket subtrees from a loaded skeleton.

    A held weapon parked away from the body (MLH_Archer's ``Bow``/``Arrow``,
    RMW_Orc's ``Weapon01``) is rig furniture, not anatomy: it carries no body
    motion to learn and spends a joint slot in every window that samples it.
    ``find_prop_socket_joints`` picks the sockets; this removes them and their
    subtrees, then remaps the hierarchy.

    Run this BEFORE ``crop_animation_to_max_joints`` so the MAX_JOINTS budget is
    spent on anatomy, and so every downstream rest-pose artifact (face joints,
    contact joints, offsets) is inferred on the filtered skeleton.

    ``drop_names`` makes the removal follow an explicit name list instead of
    re-detecting. The rest pose and each motion clip are *different files*, so
    detecting independently on both risks the two disagreeing; the rest pose
    decides once and every clip of that character follows the same names. A name
    the clip's rig does not carry is an error, not a silent skip. Detection logs
    a [WARN] naming the dropped joints (once per character); following an
    explicit list stays quiet, since it only mirrors a decision already logged.

    Returns ``(anim, names, keep_indices)``, with ``keep_indices=None`` and the
    inputs unchanged when nothing is dropped.
    """
    parents = np.asarray(anim.parents, dtype=np.int64)
    joint_count = int(parents.shape[0])
    names = list(names)
    if len(names) != joint_count:
        raise ValueError(
            f"Expected {joint_count} joint names to drop prop sockets, got {len(names)}"
        )
    label = f' for {context}' if context else ''

    if drop_names is None:
        prop_joints = find_prop_socket_joints(anim.offsets, parents, names)
    else:
        wanted = set(drop_names)
        missing = wanted.difference(names)
        if missing:
            raise ValueError(
                f"prop-socket joints {sorted(missing)} named by the rest pose are "
                f"missing from this skeleton{label}"
            )
        prop_joints = {
            joint_index
            for joint_index in range(joint_count)
            if names[joint_index] in wanted
        }
    if 0 in prop_joints:
        raise ValueError(f"prop-socket filter would drop the root joint{label}")
    # Sweep the sockets' descendants in, so the removed set is always whole
    # subtrees. Parents always precede their children, so one ascending pass
    # carries a socket down its whole chain.
    for joint_index in range(1, joint_count):
        if int(parents[joint_index]) in prop_joints:
            prop_joints.add(joint_index)
    if not prop_joints:
        return anim, names, None

    keep_indices = [
        joint_index for joint_index in range(joint_count) if joint_index not in prop_joints
    ]
    filtered, new_names = reindex_animation_to_kept_joints(anim, names, keep_indices)
    if drop_names is None:
        dropped = [names[joint_index] for joint_index in sorted(prop_joints)]
        _warn(f'dropping {len(dropped)} prop-socket joint(s){label}: {dropped}')
    return filtered, new_names, keep_indices


# A BVH "End Site" re-imported as a real bone: a leaf whose name is its parent's
# plus an ``end``/``end site`` terminator ("Tail10_end", "L Toe01_end_site",
# "Hand End"). The separator is required so real anatomy never matches -- "Bend"
# and "Legend" are names, not terminators.
_END_SITE_NAME_RE = re.compile(r'[_\s.-](end|end[_\s.-]?site)$', re.IGNORECASE)


def find_end_site_joints(parents, names):
    """Indices of the End-Site terminator leaves in a skeleton.

    A joint qualifies only when it is a leaf AND its name carries an End-Site
    suffix, so a real bone that happens to sit at the end of a chain is never
    touched. Terminators that stack ("Foo_end" carrying a "Foo_end_site") are
    peeled repeatedly, since removing the outer one turns the inner one into a
    leaf.
    """
    parents = np.asarray(parents, dtype=np.int64)
    joint_count = int(parents.shape[0])
    dropped = set()
    while True:
        has_child = {
            int(parents[joint_index])
            for joint_index in range(joint_count)
            if joint_index not in dropped and int(parents[joint_index]) >= 0
        }
        newly = {
            joint_index
            for joint_index in range(1, joint_count)
            if joint_index not in dropped
            and joint_index not in has_child
            and _END_SITE_NAME_RE.search(str(names[joint_index]))
        }
        if not newly:
            return dropped
        dropped |= newly


def drop_end_site_joints(anim, names, *, drop_names=None, context=None):
    """Drop BVH End-Site terminator leaves from a loaded skeleton.

    A tpose that round-tripped through BVH comes back with every End Site
    materialised as a real bone. They carry no motion, and each one steals the
    ``EndEffector``/``ChainEnd`` marker from the joint it hangs off -- which
    changes that joint's canonical name and so its T5 conditioning vector, and
    lengthens every ``Segment N Of M`` count up its chain. The same character
    loaded from FBX and from BVH would then condition differently, so the
    terminators are removed rather than cropped: this is not a joint-budget
    decision and it runs whether or not ``crop_animation_to_max_joints`` does.

    Run it AFTER ``drop_prop_socket_joints`` (a parked weapon's terminator goes
    out with the weapon) and BEFORE ``crop_animation_to_max_joints``, so the crop
    spends its budget on anatomy instead of punctuation.

    ``drop_names`` mirrors the prop-socket contract: the rest pose detects once
    and every motion clip of that character follows the same explicit list, so
    the two can never disagree about the joint set. A name the clip's rig does
    not carry is an error, not a silent skip.

    Returns ``(anim, names, keep_indices)``, with ``keep_indices=None`` and the
    inputs unchanged when nothing is dropped.
    """
    parents = np.asarray(anim.parents, dtype=np.int64)
    joint_count = int(parents.shape[0])
    names = list(names)
    if len(names) != joint_count:
        raise ValueError(
            f"Expected {joint_count} joint names to drop end sites, got {len(names)}"
        )
    label = f' for {context}' if context else ''

    if drop_names is None:
        end_site_joints = find_end_site_joints(parents, names)
    else:
        wanted = set(drop_names)
        missing = wanted.difference(names)
        if missing:
            raise ValueError(
                f"end-site joints {sorted(missing)} named by the rest pose are "
                f"missing from this skeleton{label}"
            )
        end_site_joints = {
            joint_index
            for joint_index in range(joint_count)
            if names[joint_index] in wanted
        }
    if 0 in end_site_joints:
        raise ValueError(f"end-site filter would drop the root joint{label}")
    if not end_site_joints:
        return anim, names, None

    keep_indices = [
        joint_index for joint_index in range(joint_count) if joint_index not in end_site_joints
    ]
    # reindex_animation_to_kept_joints rejects a keep-set that orphans a child,
    # which is the guard that an explicit drop_names list really did name leaves.
    filtered, new_names = reindex_animation_to_kept_joints(anim, names, keep_indices)
    if drop_names is None:
        dropped = [names[joint_index] for joint_index in sorted(end_site_joints)]
        _warn(f'dropping {len(dropped)} BVH end-site joint(s){label}: {dropped}')
    return filtered, new_names, keep_indices


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

def find_prop_socket_joints(
    offsets,
    parents,
    joint_names,
    length_ratio=PROP_SOCKET_BONE_LENGTH_RATIO,
    max_subtree_joints=PROP_SOCKET_MAX_SUBTREE_JOINTS,
):
    """Joints held by a detached prop/weapon socket bone, as an index set.

    Unity character packs park a held weapon in its own bone far from the body
    (MLH_Archer's ``Bow``/``Arrow`` at (+/-3, 1, 0) on a 1.03-tall archer), which
    dominates the rest-pose size statistics and shrinks the character ~3x. A
    socket is a joint satisfying all three of:

    * its name carries no body part (``joint_name_is_non_anatomical``),
    * it carries at most ``max_subtree_joints`` joints below it,
    * its bone is ``length_ratio`` times the skeleton's 90th-percentile bone
      (median as floor for near-degenerate rigs).

    The socket and its whole subtree are returned. Name and geometry fail in
    opposite directions and are exact only together: 194 dataset joints are
    non-anatomically named but sit on the body (armor, fur, saddles), while
    geometry alone cannot tell a parked staff from a single-bone tail.
    """
    offsets = np.asarray(offsets, dtype=np.float64)
    parents = np.asarray(parents)
    joint_count = len(parents)
    if joint_count < 2:
        return set()
    if len(joint_names) != joint_count:
        raise ValueError(
            f"Expected {joint_count} joint names for the socket-name guard, got {len(joint_names)}"
        )

    # The root's own offset is rig placement, not a bone: keep it out of both
    # the reference and the scan, the same way every other bone statistic does.
    lengths = np.linalg.norm(offsets, axis=1)
    reference = max(float(np.percentile(lengths[1:], 90)), float(np.median(lengths[1:])))
    if reference <= 0.0:
        return set()

    subtree_size = np.ones(joint_count, dtype=np.int64)
    for joint_index in range(joint_count - 1, 0, -1):
        parent_index = int(parents[joint_index])
        if parent_index >= 0:
            subtree_size[parent_index] += subtree_size[joint_index]

    prop_joints = {
        joint_index
        for joint_index in range(1, joint_count)
        if subtree_size[joint_index] <= max_subtree_joints
        and lengths[joint_index] > length_ratio * reference
        and joint_name_is_non_anatomical(joint_names[joint_index])
    }
    # Sweep the sockets' descendants in. Parents always precede their children,
    # so one ascending pass carries a socket down its whole chain.
    for joint_index in range(1, joint_count):
        if int(parents[joint_index]) in prop_joints:
            prop_joints.add(joint_index)
    return prop_joints


def get_average_axial_bone_length(offsets, parents, joint_side_labels, joint_names):
    """Compute the mean bone length of axial (center-labeled) bones, excluding root.

    Falls back to the mean bone length across *all* non-root bones when no
    center-labeled bones exist, so the return value is always a positive float.

    Degenerate bones -- rig helpers that sit exactly on their parent, e.g. the
    zero-length ``RigSpine`` of MU04_Pollen -- are dropped from the mean. They
    carry no size information but drag the average down, which inflates the
    character's scale factor (Pollen: 2.95 instead of 4.43, a 1.22x oversize).
    Prop-socket bones (see ``find_prop_socket_joints``) are dropped for the
    mirror-image reason: MLH_Archer's 3.05-long ``Bow``/``Arrow`` lift the 0.31
    body mean to 0.55 and shrink the archer 1.8x. The branch is still chosen by
    the *unfiltered* center-bone count, so filtering never flips skeletons
    between branches.
    """
    prop_joints = find_prop_socket_joints(offsets, parents, joint_names)
    all_lengths = [
        float(np.linalg.norm(offsets[j]))
        for j in range(1, len(parents))
        if j not in prop_joints
    ]
    if not all_lengths:
        return 0.1  # ultimate fallback for single-bone skeletons
    min_bone_length = DEGENERATE_BONE_LENGTH_RATIO * (sum(all_lengths) / len(all_lengths))

    center_count = 0
    axial_lengths = []
    for joint_index in range(1, len(parents)):  # skip root (no parent bone)
        if joint_index < len(joint_side_labels) and joint_side_labels[joint_index] == 'center':
            center_count += 1
            if joint_index not in prop_joints:
                axial_lengths.append(float(np.linalg.norm(offsets[joint_index])))
    if center_count >= 10 and axial_lengths:
        return _mean_excluding_degenerate(axial_lengths, min_bone_length)
    # Fallback: average bone length across all non-root bones.
    return _mean_excluding_degenerate(all_lengths, min_bone_length)


def _mean_excluding_degenerate(lengths, min_bone_length):
    """Mean of ``lengths`` ignoring degenerate ones; unfiltered mean if all are."""
    kept = [length for length in lengths if length >= min_bone_length]
    if not kept:
        return sum(lengths) / len(lengths)
    return sum(kept) / len(kept)


def max_joint_span(positions, exclude_joints=()):
    """Largest joint-to-joint distance among ``positions``.

    ``exclude_joints`` drops joints from the measurement only -- the positions
    are already resolved, so excluding one never moves another. The unfiltered
    span is returned when the exclusions would leave fewer than two joints.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if exclude_joints:
        kept = [j for j in range(len(positions)) if j not in exclude_joints]
        if len(kept) >= 2:
            positions = positions[kept]
    joint_deltas = positions[:, None, :] - positions[None, :, :]
    max_span = np.linalg.norm(joint_deltas, axis=-1).max()
    return max(float(max_span), 1e-8)


def rest_positions_from_offsets(offsets, parents):
    """Accumulate parent-to-child ``offsets`` into rest-pose joint positions.

    The offsets must already be rest-pose deltas (``rest_positions[j] -
    rest_positions[parent[j]]``). Bone-local ``Animation.offsets`` are stated in
    the parent bone's frame and sum to a straightened-out skeleton, so FK a
    loaded animation instead of passing ``anim.offsets`` here.
    """
    offsets = np.asarray(offsets, dtype=np.float64)
    rest_positions = np.zeros_like(offsets)
    for joint_index, parent_index in enumerate(parents):
        if parent_index >= 0:
            rest_positions[joint_index] = rest_positions[parent_index] + offsets[joint_index]
        else:
            rest_positions[joint_index] = offsets[joint_index]
    return rest_positions


def get_rest_body_max_span(offsets, parents, exclude_joints=()):
    """``max_joint_span`` of the rest pose accumulated from ``offsets``."""
    return max_joint_span(rest_positions_from_offsets(offsets, parents), exclude_joints)


def get_scale_reference_extent(rest_positions, parents, joint_names):
    """Character extent used for scale normalization.

    This is the rest-pose joint span widened to the root's own elevation above
    the rig origin. A handful of rigs (hovering/drifting creatures, effect rigs)
    seat their root far above the origin while their bones stay tiny; that
    elevation is part of the character's size but never shows up in a
    joint-to-joint span, so a purely bone-driven scale blows the normalized root
    height up into an outlier -- MU04_Pollen ends up with its root at 14.7 joint
    spans (8.6 units against a dataset median of 0.45). Folding the elevation in
    re-anchors those characters and is a no-op for every rig whose root already
    sits inside its own joint span, which is all 104 truebones/zoo species.

    Only the vertical component counts: the root offset's XZ is authoring
    placement rather than size, and is exactly 0.0 for all 260 dataset species.

    Prop-socket joints (see ``find_prop_socket_joints``) are left out of the
    span for the same reason: a parked weapon sets the span from where the rig
    authored it, not from the body. ``rest_positions`` must be the FK'd rest
    pose, not hand-accumulated offsets: see ``rest_positions_from_offsets``.
    """
    rest_positions = np.asarray(rest_positions, dtype=np.float64)
    parents = np.asarray(parents)
    # Parent-to-child deltas. Their lengths are the bone lengths the socket
    # filter needs, and are identical to the bone-local offsets' lengths.
    bone_deltas = rest_positions.copy()
    has_parent = parents >= 0
    bone_deltas[has_parent] -= rest_positions[parents[has_parent]]
    prop_joints = find_prop_socket_joints(bone_deltas, parents, joint_names)
    joint_span = max_joint_span(rest_positions, exclude_joints=prop_joints)
    root_elevation = abs(float(rest_positions[0][1]))
    return max(joint_span, root_elevation)


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
