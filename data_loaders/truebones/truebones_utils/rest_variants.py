"""Rest variants: hand-made rest poses of one species, drawn per sample.

A species may ship more than one rest pose. ``TPose.glb`` is the primary rest:
every clip is encoded on it, and it alone decides the species' scale, facing,
root and joint set. Each ``TPose_<tag>.glb`` beside it is the same skeleton held
in another rest (arms lowered, knees bent). Preprocessing stores a variant as
the local rest rotations that turn the primary rest into it
(``REST_VARIANTS_KEY``); the loader re-expresses a clip on one of them with
``apply_rest_rotations``, exactly as the rest re-pose augmentation does with
drawn rotations, so the model sees one motion under several real rests.

The structural channel names the stored rig's joints and keeps reading the
primary rest (see ``rest_repose_aug``); ``stored_rest_struct_cond`` carries that
rest through a topology edit made on a variant.
"""
from __future__ import annotations

import random

import numpy as np

from data_loaders.truebones.truebones_utils.canonical_features import rest_pos_from_cond
from data_loaders.truebones.truebones_utils.rest_repose_aug import apply_rest_rotations

# Cond entry of a species' rest variants: ``tags`` [K], ``rotations``
# ``[K, J, 3, 3]`` local rest rotations on the primary rest, and the
# ``primary_sha1`` / ``sha1`` [K] of the source files they were measured from.
REST_VARIANTS_KEY = 'rest_variants'


def validate_rest_variant_prob(probability):
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError('rest_variant_prob must be finite and in [0, 1].')


def empty_rest_variants(primary_sha1=None):
    return {'tags': [], 'rotations': np.zeros((0, 0, 3, 3), dtype=np.float32),
            'primary_sha1': primary_sha1, 'sha1': []}


def rest_variant_count(cond):
    variants = cond.get(REST_VARIANTS_KEY)
    return 0 if not variants else len(variants['tags'])


def apply_rest_variant(motion, cond, index):
    """Return (physical motion, private cond, diagnostics) of ``motion`` on the
    species' rest variant ``index``."""
    variants = cond[REST_VARIANTS_KEY]
    rotations = np.asarray(variants['rotations'][index], dtype=np.float64)
    if rotations.shape != (len(cond['parents']), 3, 3):
        raise ValueError(
            f"rest variant '{variants['tags'][index]}' has rotations {rotations.shape} "
            f"for {len(cond['parents'])} joints; re-run preprocessing.")
    out, new_cond = apply_rest_rotations(motion, cond, rotations)
    new_cond.pop(REST_VARIANTS_KEY, None)
    return out, new_cond, {'applied': True, 'tag': str(variants['tags'][index]),
                           'index': int(index), 'skip_reason': ''}


def augment_rest_variant(motion, cond, *, rng=random):
    """``apply_rest_variant`` on a uniformly drawn variant; a species without
    any is a no-op."""
    count = rest_variant_count(cond)
    if count == 0:
        return motion, cond, {'applied': False, 'tag': None, 'index': -1,
                              'skip_reason': 'species has no rest variant'}
    return apply_rest_variant(motion, cond, rng.randrange(count))


def stored_rest_struct_cond(cond, stored_cond, new_index):
    """``cond`` (a topology edit of a rest variant of ``stored_cond``) with the
    stored rest positions of its joints, for the structural channel.

    A topology edit only drops or merges joints, so every kept joint keeps its
    rest position; ``new_index`` maps each stored joint to its edited index
    (-1 when gone).
    """
    new_index = np.asarray(new_index, dtype=np.int64)
    kept = np.flatnonzero(new_index >= 0)
    old_of_new = np.empty(kept.size, dtype=np.int64)
    old_of_new[new_index[kept]] = kept
    stored_rest = rest_pos_from_cond(stored_cond)
    return dict(cond, rest_pos_ric_hml=stored_rest[old_of_new])
