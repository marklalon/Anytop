"""Joint-name canonicalization and per-joint text embeddings.

Assigns canonical joint names (disambiguating collisions), refreshes the joint
metadata stored in cond dicts, and attaches the cached T5 embeddings of the
joint / species texts. Re-exported by ``animation_utils``.
"""

from collections import Counter, defaultdict
import json
import numpy as np
import os
from os.path import join as pjoin
import re
import torch

from data_loaders.truebones.truebones_utils.dataset_tags import (
    assert_species_tags_cover,
)
from .physics_joint_annotation import (
    build_semantic_metadata,
    effective_canonical_replacements,
    infer_species_joint_name_prefixes,
    joint_name_token_is_species,
    normalize_joint_name,
    strip_joint_name_prefix,
    build_joint_embedding_texts,
    build_species_embedding_text,
    JOINT_NAME_EMBEDDING_SCHEMA_VERSION,
)


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


def _joint_disambiguation_tokens(raw_name, canonical_name, additional_prefixes=(), translated_tokens=()):
    raw_value = str(raw_name or '')
    stripped_raw = strip_joint_name_prefix(raw_value, additional_prefixes)
    raw_tokens = normalize_joint_name(stripped_raw).split()
    canonical_tokens = normalize_joint_name(canonical_name).split()
    residual_tokens = _remove_token_counts(raw_tokens, Counter(canonical_tokens))
    # A word the canonicalizer translated is already in the canonical name under
    # its English spelling; appended again it gets translated a second time
    # (Pirrana "shiribire" -> "Anal Fin Shiribire" -> text "Anal Fin Anal Fin").
    residual_tokens = [
        token for token in residual_tokens
        if not joint_name_token_is_species(token) and token not in translated_tokens
    ]
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
    translated_tokens = frozenset(effective_canonical_replacements(raw_names))
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
                translated_tokens,
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
        # Looked up at call time: the warning collectors monkey-patch
        # ``animation_utils._warn``.
        from . import animation_utils as _animation_utils
        _animation_utils._warn(f'canonical joint-name collision scan found {len(collision_groups)} group(s); report: {report_path}')
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


# Texts per T5 forward pass; bounds padding memory on a full-dataset encode.
_T5_ENCODE_BATCH = 256


def _build_t5_text_cache(cache_cond, t5_name):
    """Map every embedding text already baked into *cache_cond* to its T5 vector.

    The encoder is a masked mean over one text's own tokens, so a text encoded
    under the same T5 model yields the same vector whichever cond it was baked
    into. Joint-name rows and species descriptors share one table because they
    go through the same encoder.
    """
    cache = {}
    if not isinstance(cache_cond, dict):
        return cache
    for entry in cache_cond.values():
        if not isinstance(entry, dict):
            continue
        joint_meta = entry.get('joints_names_embs_meta')
        joint_embs = entry.get('joints_names_embs')
        if (isinstance(joint_meta, dict) and joint_embs is not None
                and str(joint_meta.get('t5_name') or '') == t5_name):
            texts = list(joint_meta.get('embedding_texts') or ())
            joint_embs = np.asarray(joint_embs, dtype=np.float32)
            if joint_embs.ndim == 2 and joint_embs.shape[0] == len(texts):
                for text, emb in zip(texts, joint_embs):
                    cache.setdefault(str(text), emb)
        species_meta = entry.get('species_emb_meta')
        species_emb = entry.get('species_emb')
        if (isinstance(species_meta, dict) and species_emb is not None
                and str(species_meta.get('t5_name') or '') == t5_name
                and species_meta.get('embedding_text')):
            cache.setdefault(str(species_meta['embedding_text']),
                             np.asarray(species_emb, dtype=np.float32))
    return cache


def _reference_joint_texts(reference_cond, t5_name):
    """Every joint-name text the reference cond encodes under *t5_name*.

    Membership is only meaningful when the reference was built by the same text
    builder, so an entry under another joint-name schema is a hard error rather
    than a source of false "unseen" verdicts.
    """
    texts = set()
    if not isinstance(reference_cond, dict):
        raise ValueError('blanking unseen joint names needs the reference cond.npy')
    for object_type, entry in reference_cond.items():
        meta = entry.get('joints_names_embs_meta') if isinstance(entry, dict) else None
        if not isinstance(meta, dict) or str(meta.get('t5_name') or '') != t5_name:
            continue
        schema_version = meta.get('schema_version')
        if schema_version is None or int(schema_version) != JOINT_NAME_EMBEDDING_SCHEMA_VERSION:
            raise ValueError(
                f"reference cond entry '{object_type}' was encoded under joint-name schema "
                f"{schema_version}, this code is at {JOINT_NAME_EMBEDDING_SCHEMA_VERSION}; "
                f"which names it covers cannot be judged across schemas"
            )
        texts.update(str(text) for text in meta.get('embedding_texts') or ())
    if not texts:
        raise ValueError(f'the reference cond holds no joint-name texts encoded with {t5_name}')
    return texts


def _blank_unseen_joint_texts(embedding_texts_by_object, known_texts):
    """Swap every joint text the reference never encodes for the blank text.

    The model is trained to read an all-zero name row as "name unknown"
    (``--joint_name_drop_prob``), and the blank text encodes to exactly that row.
    A text the checkpoint was never trained on would instead be encoded into a
    point of T5 space the model has no prior for. Returns
    ``{object_type: {joint_index: original_text}}``.
    """
    blanked_by_object = {}
    for object_type, texts in embedding_texts_by_object.items():
        blanked = {
            index: text for index, text in enumerate(texts)
            if str(text).strip() and text not in known_texts
        }
        embedding_texts_by_object[object_type] = [
            '' if index in blanked else text for index, text in enumerate(texts)
        ]
        blanked_by_object[object_type] = blanked
        if blanked:
            named = sum(1 for text in texts if str(text).strip())
            print(f'[{object_type}] blanked {len(blanked)}/{named} named joint(s) whose '
                  f'text the reference cond never encodes:')
            for index, text in sorted(blanked.items()):
                print(f'  - joint {index}: {text!r}')
    return blanked_by_object


def attach_t5_embeddings_to_cond(cond, save_dir, t5_name='t5-base', write_collision_report=True,
                                  t5_conditioner=None, embedding_cache_cond=None,
                                  blank_unseen_joint_names=False):
    """Bake joint-name and species T5 embeddings into every entry of *cond*.

    ``embedding_cache_cond`` is an already-encoded cond (e.g. the checkpoint's
    reference cond.npy): any text it holds under the same T5 model is reused
    verbatim, and T5 is loaded only when some text is missing from it.

    ``blank_unseen_joint_names`` gives every joint whose text that reference
    never encodes the blank (all-zero) name instead of a fresh T5 vector; the
    originals are kept in ``joints_names_embs_meta['blanked_unseen_texts']``.
    """
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

    blanked_by_object = {}
    if blank_unseen_joint_names:
        blanked_by_object = _blank_unseen_joint_texts(
            embedding_texts_by_object, _reference_joint_texts(embedding_cache_cond, t5_name)
        )

    object_types_to_encode = sorted(cond)
    joint_count = len(object_types_to_encode)

    if t5_conditioner is None:
        # Fast-fail before any encoding: the per-species descriptor has no fallback,
        # so a species missing from species_tags.jsonl must surface here.
        assert_species_tags_cover(cond.keys())

    species_texts_by_object = {
        object_type: build_species_embedding_text(cond[object_type])
        for object_type in object_types_to_encode
    }

    # Every text either comes from the cache or is encoded once, in one batch.
    text_cache = _build_t5_text_cache(embedding_cache_cond, t5_name)
    wanted = []
    for object_type in object_types_to_encode:
        wanted.extend(embedding_texts_by_object[object_type])
        wanted.append(species_texts_by_object[object_type])
    missing = list(dict.fromkeys(text for text in wanted if text not in text_cache))
    if text_cache:
        print(f'Reusing cached T5 embeddings for {len(set(wanted)) - len(missing)}/'
              f'{len(set(wanted))} texts.')

    if missing:
        if t5_conditioner is None:
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
        if text_cache:
            print(f'Texts not in the cache: {missing}')
        print(f'Encoding {len(missing)} texts via T5 ...')
        with torch.no_grad():
            for start in range(0, len(missing), _T5_ENCODE_BATCH):
                chunk = missing[start:start + _T5_ENCODE_BATCH]
                tokens = t5_conditioner.tokenize_entries(chunk)
                embs = t5_conditioner(tokens).detach().cpu().numpy().astype(np.float32, copy=False)
                text_cache.update(zip(chunk, embs))
    else:
        print('All embedding texts cached; skipped T5.')

    print(f'Attaching T5 embeddings for {joint_count} object types ...')
    for object_type in object_types_to_encode:
        object_cond = cond[object_type]
        embedding_texts = embedding_texts_by_object[object_type]
        embs = np.stack([text_cache[text] for text in embedding_texts]).astype(np.float32, copy=False)
        object_cond['joints_names_embs'] = embs
        object_cond['joints_names_embs_meta'] = {
            't5_name': t5_name,
            'schema_version': JOINT_NAME_EMBEDDING_SCHEMA_VERSION,
            'embedding_dim': int(embs.shape[1]) if embs.ndim == 2 else 0,
            'embedding_texts': list(embedding_texts),
            'blanked_unseen_texts': dict(blanked_by_object.get(object_type, {})),
        }

        species_text = species_texts_by_object[object_type]
        species_emb = np.array(text_cache[species_text], dtype=np.float32)
        object_cond['species_emb'] = species_emb
        object_cond['species_emb_meta'] = {
            't5_name': t5_name,
            'schema_version': JOINT_NAME_EMBEDDING_SCHEMA_VERSION,
            'embedding_dim': int(species_emb.shape[-1]),
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
