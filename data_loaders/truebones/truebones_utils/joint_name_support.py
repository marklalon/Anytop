"""Support diagnostics for a new skeleton's joint-name tokens.

A joint token is only as good as what the trained model has behind it. Three
numbers say that: whether the query rig *is* one the corpus already carries, how
many training species carry the identical text, and -- since a novel-but-well-
placed token is fine -- how close the best *different* token is. A bare
"Left Leg" on a rig the corpus spells "Calf" fails all three (a handful of
species, no kin, and the nearest other token is "Left Arm"); a finger the corpus
has never seen fails only the second, and lands on another finger. Thresholds
are read off the leave-one-species-out distribution over the merged reference
cond, where they flag roughly a tenth of training joints at each level.

The species count answers "how many *different* creatures teach this token",
which is the right question for a genuinely new skeleton and the wrong one for a
re-import: when the query rig is token-for-token a training creature, a token
only that creature carries was fitted on this very geometry and there is no
transfer for the model to get wrong. ``_sibling_reference_species`` finds that
kinship and exempts the tokens it covers, which is what keeps a re-imported
training rig quiet instead of burying a reader in warnings that do not apply.

This is a diagnostic bolted onto the end of preprocessing -- it never gates a
run. See ``collect_joint_name_support_rows`` / ``write_joint_name_support_report``.
"""

import json
from collections import Counter
from os.path import join as pjoin

import numpy as np

from data_loaders.truebones.truebones_utils.physics_joint_annotation import (
    _COMPOUND_MODIFIER_TOKENS,
    _EMBED_TEXT_SIDE_TOKENS,
)


def _warn(msg: str):
    """Print a warning message in yellow (mirrors animation_utils._warn)."""
    print(f'\033[93m[WARN] {msg}\033[0m')


_JOINT_SUPPORT_MIN_SPECIES = 10
_JOINT_SUPPORT_HIGH_SPECIES = 2
_JOINT_SUPPORT_HIGH_FALLBACK_COS = 0.85
_JOINT_SUPPORT_MEDIUM_FALLBACK_COS = 0.90
_JOINT_SUPPORT_TOP_K = 5

# A reference species counts as the same rig as the query when four fifths of
# each side's tokens are shared. Coverage is required in *both* directions so a
# stub rig cannot claim kinship with everything by being a subset of it. On the
# merged corpus nothing sits near the line: a re-imported training creature
# scores ~0.97 against its own entry and ~0.53 against the next species down.
_RIG_SIBLING_OVERLAP = 0.80
_RIG_SIBLING_MIN_TOKENS = 8
# Matching a rig the corpus barely animates is not evidence that its tokens were
# trained on anything. Eight of the 260 reference species sit below this; the
# median species carries 8 actions.
_RIG_SIBLING_MIN_ACTIONS = 3

# Words that never carry part identity, so they cannot make two tokens name the
# same body part. The modifier and side vocabularies are the canonicaliser's
# own; the rest are canonical suffixes that attach to any part ("Arm Twist" and
# "Leg Twist" are not kin) plus the generic head-feature category tag.
_NON_PART_WORDS = frozenset(
    {word.lower() for word in _COMPOUND_MODIFIER_TOKENS}
    | {word.lower() for word in _EMBED_TEXT_SIDE_TOKENS}
    | {'base', 'end', 'extra', 'headfeature', 'root', 'tip', 'twist'}
)


def _token_counts(texts):
    """Multiset of a rig's non-blank joint tokens (blanked props carry none)."""
    return Counter(str(text) for text in texts if str(text).strip())


def _joint_name_reference_bank(reference_cond):
    """Mean-centred unit vectors for every joint token in a reference cond.

    Mean-centring is not cosmetic: raw T5 sentence vectors share a large common
    component, so *every* pair scores high and the ranking is dominated by length
    and EOS effects rather than by meaning.
    """
    vectors = []
    texts = []
    species = []
    tokens_by_species = {}
    actions_by_species = {}
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
        tokens_by_species[str(object_type)] = _token_counts(entry_texts)
        actions_by_species[str(object_type)] = len(object_cond.get('loop_period_by_action') or {})
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
        'tokens_by_species': tokens_by_species,
        'actions_by_species': actions_by_species,
    }


def _sibling_reference_species(query_texts, bank):
    """Reference species whose rig is, token for token, the query's own rig.

    Returned as ``{object_type: overlap}``. A rig too small to be characteristic
    claims no kin, and neither does one the corpus barely animates.
    """
    query = _token_counts(query_texts)
    query_total = sum(query.values())
    if query_total < _RIG_SIBLING_MIN_TOKENS:
        return {}
    siblings = {}
    for object_type, tokens in bank['tokens_by_species'].items():
        reference_total = sum(tokens.values())
        if reference_total < _RIG_SIBLING_MIN_TOKENS:
            continue
        if bank['actions_by_species'].get(object_type, 0) < _RIG_SIBLING_MIN_ACTIONS:
            continue
        shared = sum((query & tokens).values())
        overlap = min(shared / query_total, shared / reference_total)
        if overlap >= _RIG_SIBLING_OVERLAP:
            siblings[object_type] = round(float(overlap), 4)
    return siblings


def _part_words(text):
    return {
        word for word in str(text).lower().split()
        if word and word not in _NON_PART_WORDS
    }


def _shares_part_word(left, right):
    """Do two tokens name the same body part, ignoring side and modifiers?

    An unseen ``Left Wing Finger`` landing on ``Left Arm Finger`` is the case
    this module's docstring calls acceptable. ``Left Leg`` landing on
    ``Left Arm`` is not. Only the words carrying part identity separate the two.
    """
    return bool(_part_words(left) & _part_words(right))


def _collect_rows_and_siblings(cond, reference_cond, top_k=_JOINT_SUPPORT_TOP_K):
    """``collect_joint_name_support_rows`` plus the per-rig sibling map it used."""
    bank = _joint_name_reference_bank(reference_cond)
    if bank is None:
        return [], {}

    rows = []
    siblings_by_object_type = {}
    for object_type in sorted(cond):
        object_cond = cond[object_type]
        embeddings = object_cond.get('joints_names_embs')
        meta = object_cond.get('joints_names_embs_meta') or {}
        texts = list(meta.get('embedding_texts') or [])
        raw_names = list(object_cond.get('joints_names') or [])
        if embeddings is None or not texts:
            continue
        embeddings = np.asarray(embeddings, dtype=np.float32)
        siblings = _sibling_reference_species(texts, bank)
        if siblings:
            siblings_by_object_type[str(object_type)] = siblings

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
            sibling_support = [name for name in support_species if name in siblings]
            if sibling_support:
                # The exact token was fitted on this very rig, so the species
                # count is measuring a transfer nobody is asking the model for.
                risk, reason = 'low', 'sibling_rig'
            elif support_count <= _JOINT_SUPPORT_HIGH_SPECIES and fallback_cos < _JOINT_SUPPORT_HIGH_FALLBACK_COS:
                if support_count == 0 and neighbours and _shares_part_word(text, neighbours[0]['text']):
                    # Never seen, but it lands on the same part -- the docstring's
                    # unseen finger. Worth reading, not worth shouting about.
                    risk, reason = 'medium', 'unseen_same_part'
                elif support_count == 0:
                    risk, reason = 'high', 'unseen'
                else:
                    risk, reason = 'high', 'narrow'
            elif support_count < _JOINT_SUPPORT_MIN_SPECIES and fallback_cos < _JOINT_SUPPORT_MEDIUM_FALLBACK_COS:
                risk, reason = 'medium', 'thin'
            else:
                risk, reason = 'low', 'covered'

            rows.append({
                'object_type': str(object_type),
                'index': int(joint_index),
                'raw_name': str(raw_names[joint_index]) if joint_index < len(raw_names) else '',
                'embedding_text': str(text),
                'support_species_count': support_count,
                'support_species': support_species[:10],
                'sibling_support_species': sibling_support[:10],
                'nearest_other_cos': round(fallback_cos, 4),
                'nearest_other': neighbours,
                'risk': risk,
                'reason': reason,
            })
    # Loneliest first: the fallback cosine is what says how far the model has to
    # reach, and it is comparable across joints in a way the support count is not.
    rows.sort(key=lambda row: (row['risk'] != 'high', row['risk'] != 'medium', row['nearest_other_cos']))
    return rows, siblings_by_object_type


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
    return _collect_rows_and_siblings(cond, reference_cond, top_k=top_k)[0]


def _describe(row):
    """One line saying what the model actually has behind a flagged token."""
    nearest = row['nearest_other'][0] if row['nearest_other'] else None
    nearest_text = (
        f"{nearest['text']!r} ({nearest['species']}) cos={nearest['cos']}"
        if nearest else 'nothing comparable'
    )
    if row['reason'] == 'unseen':
        return f'no species carries it; it falls back on {nearest_text}'
    return (
        f"{row['support_species_count']} species carry it, none kin to this rig; "
        f'nearest other token {nearest_text}'
    )


def _best_sibling(siblings_by_object_type):
    """The closest rig match found, as ``(overlap, object_type)`` or ``None``."""
    best = None
    for found in siblings_by_object_type.values():
        for object_type, overlap in found.items():
            if best is None or overlap > best[0]:
                best = (overlap, object_type)
    return best


def write_joint_name_support_report(cond, save_dir, reference_cond, top_k=_JOINT_SUPPORT_TOP_K):
    """Write ``joint_name_support_report.json`` and warn about weak tokens.

    A diagnostic, never a gate: an unusual creature legitimately carries tokens
    the corpus has never seen, and only a reader can say whether a given one
    matters. It fails soft -- a reference cond that cannot be read costs the
    report, not the preprocessing run.
    """
    rows, siblings = _collect_rows_and_siblings(cond, reference_cond, top_k=top_k)
    if not rows:
        return []

    high = [row for row in rows if row['risk'] == 'high']
    medium = [row for row in rows if row['risk'] == 'medium']
    exempt = [row for row in rows if row['reason'] == 'sibling_rig']
    report = {
        'num_joints_scanned': int(len(rows)),
        'num_high_risk': int(len(high)),
        'num_medium_risk': int(len(medium)),
        'num_sibling_exempt': int(len(exempt)),
        'rig_siblings': siblings,
        'thresholds': {
            'sibling_rig': f'a reference species sharing >= {_RIG_SIBLING_OVERLAP} of the '
                           'tokens in both directions; the tokens it carries are exempt',
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

    best = _best_sibling(siblings)
    if best is not None:
        overlap, object_type = best
        print(
            f'[OK] this rig matches {object_type} at {overlap:.2f} token coverage; '
            f'the {len(exempt)} joint(s) it already carries are exempt from this scan'
        )

    if high:
        # One line per distinct token: a rig with twelve wing fingers spelled the
        # same way has one problem, not twelve, and a console listing them
        # separately pushes the other problems off the screen.
        by_text = {}
        for row in high:
            by_text.setdefault(row['embedding_text'], []).append(row)
        _warn(
            f'{len(by_text)} joint name(s) covering {len(high)} joint(s) are barely '
            f'represented in the reference cond; the model has no prior fitted on this '
            f'body part. Report: {report_path}'
        )
        for text, group in list(by_text.items())[:10]:
            first = group[0]
            where = f"[{first['index']}]" if len(group) == 1 else f"[{len(group)} joints]"
            print(f"  - {where} {first['raw_name']!r} -> {text!r}: {_describe(first)}")
        if len(by_text) > 10:
            print(f'  ... {len(by_text) - 10} additional token(s) omitted from console output')
    else:
        print(f'[OK] joint-name support scan found no weakly-supported joint names')
    if medium:
        print(f'  ({len(medium)} additional joint(s) flagged medium; see {report_path})')

    return rows
