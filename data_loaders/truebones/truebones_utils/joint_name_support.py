"""Support diagnostics for a new skeleton's joint-name tokens.

A joint token is only as good as what the trained model has behind it. Two
numbers say that: how many training species carry the identical text, and --
since a novel-but-well-placed token is fine -- how close the best *different*
token is. A bare "Left Leg" fails both (a handful of species, and the nearest
other token is "Left Arm"); a finger the corpus has never seen fails only the
first, and lands on another finger. Thresholds are read off the
leave-one-species-out distribution over the merged reference cond, where they
flag roughly a tenth of training joints at each level.

This is a diagnostic bolted onto the end of preprocessing -- it never gates a
run. See ``collect_joint_name_support_rows`` / ``write_joint_name_support_report``.
"""

import json
from os.path import join as pjoin

import numpy as np


def _warn(msg: str):
    """Print a warning message in yellow (mirrors animation_utils._warn)."""
    print(f'\033[93m[WARN] {msg}\033[0m')


_JOINT_SUPPORT_MIN_SPECIES = 10
_JOINT_SUPPORT_HIGH_SPECIES = 2
_JOINT_SUPPORT_HIGH_FALLBACK_COS = 0.85
_JOINT_SUPPORT_MEDIUM_FALLBACK_COS = 0.90
_JOINT_SUPPORT_TOP_K = 5


def _joint_name_reference_bank(reference_cond):
    """Mean-centred unit vectors for every joint token in a reference cond.

    Mean-centring is not cosmetic: raw T5 sentence vectors share a large common
    component, so *every* pair scores high and the ranking is dominated by length
    and EOS effects rather than by meaning.
    """
    vectors = []
    texts = []
    species = []
    for object_type in sorted(reference_cond):
        object_cond = reference_cond[object_type]
        embeddings = object_cond.get('joints_names_embs')
        meta = object_cond.get('joints_names_embs_meta') or {}
        entry_texts = list(meta.get('embedding_texts') or [])
        if embeddings is None or not entry_texts:
            continue
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.shape[0] != len(entry_texts):
            continue
        for joint_index, text in enumerate(entry_texts):
            vectors.append(embeddings[joint_index])
            texts.append(str(text))
            species.append(str(object_type))
    if not vectors:
        return None
    matrix = np.stack(vectors)
    mean = matrix.mean(axis=0)
    centred = matrix - mean
    norms = np.linalg.norm(centred, axis=1, keepdims=True)
    centred /= np.maximum(norms, 1e-12)
    return {
        'mean': mean,
        'unit': centred,
        'texts': np.asarray(texts),
        'species': np.asarray(species),
    }


def collect_joint_name_support_rows(cond, reference_cond, top_k=_JOINT_SUPPORT_TOP_K):
    """Per-joint support of a new skeleton's name tokens in the training corpus.

    The blind spot ``joint_name_collision_report`` leaves: it scans a rig against
    *itself*, so a name that is unique, legal and canonicalizes cleanly still
    passes while landing nowhere near the training distribution the checkpoint
    was fitted on. Since the joint-name embedding is the only per-joint identity
    signal the model has (topology and the rest pose are shared by every spelling
    of the same skeleton), a token with no support silently borrows another
    species' motion prior.
    """
    bank = _joint_name_reference_bank(reference_cond)
    if bank is None:
        return []

    rows = []
    for object_type in sorted(cond):
        object_cond = cond[object_type]
        embeddings = object_cond.get('joints_names_embs')
        meta = object_cond.get('joints_names_embs_meta') or {}
        texts = list(meta.get('embedding_texts') or [])
        raw_names = list(object_cond.get('joints_names') or [])
        if embeddings is None or not texts:
            continue
        embeddings = np.asarray(embeddings, dtype=np.float32)

        for joint_index, text in enumerate(texts):
            if not str(text).strip():
                # A blanked prop/control joint carries no body token by design.
                continue
            query = embeddings[joint_index] - bank['mean']
            query = query / max(float(np.linalg.norm(query)), 1e-12)
            similarity = bank['unit'] @ query

            is_exact = bank['texts'] == str(text)
            support_species = sorted(set(bank['species'][is_exact].tolist()))
            other = ~is_exact
            if other.any():
                other_scores = np.where(other, similarity, -np.inf)
                order = np.argsort(-other_scores)[:max(int(top_k), 1)]
                neighbours = [
                    {
                        'text': str(bank['texts'][j]),
                        'species': str(bank['species'][j]),
                        'cos': round(float(similarity[j]), 4),
                    }
                    for j in order
                ]
                fallback_cos = float(similarity[order[0]])
            else:
                neighbours = []
                fallback_cos = -1.0

            support_count = len(support_species)
            if support_count <= _JOINT_SUPPORT_HIGH_SPECIES and fallback_cos < _JOINT_SUPPORT_HIGH_FALLBACK_COS:
                risk = 'high'
            elif support_count < _JOINT_SUPPORT_MIN_SPECIES and fallback_cos < _JOINT_SUPPORT_MEDIUM_FALLBACK_COS:
                risk = 'medium'
            else:
                risk = 'low'

            rows.append({
                'object_type': str(object_type),
                'index': int(joint_index),
                'raw_name': str(raw_names[joint_index]) if joint_index < len(raw_names) else '',
                'embedding_text': str(text),
                'support_species_count': support_count,
                'support_species': support_species[:10],
                'nearest_other_cos': round(fallback_cos, 4),
                'nearest_other': neighbours,
                'risk': risk,
            })
    # Loneliest first: the fallback cosine is what says how far the model has to
    # reach, and it is comparable across joints in a way the support count is not.
    rows.sort(key=lambda row: (row['risk'] != 'high', row['risk'] != 'medium', row['nearest_other_cos']))
    return rows


def write_joint_name_support_report(cond, save_dir, reference_cond, top_k=_JOINT_SUPPORT_TOP_K):
    """Write ``joint_name_support_report.json`` and warn about weak tokens.

    A diagnostic, never a gate: an unusual creature legitimately carries tokens
    the corpus has never seen, and only a reader can say whether a given one
    matters. It fails soft -- a reference cond that cannot be read costs the
    report, not the preprocessing run.
    """
    rows = collect_joint_name_support_rows(cond, reference_cond, top_k=top_k)
    if not rows:
        return []

    high = [row for row in rows if row['risk'] == 'high']
    medium = [row for row in rows if row['risk'] == 'medium']
    report = {
        'num_joints_scanned': int(len(rows)),
        'num_high_risk': int(len(high)),
        'num_medium_risk': int(len(medium)),
        'thresholds': {
            'high': f'support_species <= {_JOINT_SUPPORT_HIGH_SPECIES} and '
                    f'nearest_other_cos < {_JOINT_SUPPORT_HIGH_FALLBACK_COS}',
            'medium': f'support_species < {_JOINT_SUPPORT_MIN_SPECIES} and '
                      f'nearest_other_cos < {_JOINT_SUPPORT_MEDIUM_FALLBACK_COS}',
        },
        'joints': rows,
    }
    report_path = pjoin(save_dir, 'joint_name_support_report.json')
    with open(report_path, 'w', encoding='utf-8') as report_file:
        json.dump(report, report_file, indent=2)

    if high:
        _warn(
            f'{len(high)} joint name(s) are barely represented in the reference cond; '
            f'the model will condition them on another body part. Report: {report_path}'
        )
        for row in high[:10]:
            nearest = row['nearest_other'][0] if row['nearest_other'] else None
            nearest_text = (
                f"{nearest['text']!r} ({nearest['species']}) cos={nearest['cos']}"
                if nearest else 'nothing comparable'
            )
            print(
                f"  - [{row['index']}] {row['raw_name']!r} -> {row['embedding_text']!r}: "
                f"{row['support_species_count']} species; nearest other token {nearest_text}"
            )
        if len(high) > 10:
            print(f'  ... {len(high) - 10} additional joint(s) omitted from console output')
    else:
        print(f'[OK] joint-name support scan found no weakly-supported joint names')
    if medium:
        print(f'  ({len(medium)} additional joint(s) flagged medium; see {report_path})')

    return rows
