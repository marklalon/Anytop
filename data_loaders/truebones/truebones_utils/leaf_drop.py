"""Leaf-joint removal: a topology augmentation that is exact on the features.

Every joint's row holds its own position and its own local rotation, and a
leaf's rotation moves nothing but the leaf. Deleting
a leaf's row therefore leaves every other row a valid description of the same
motion on a skeleton that never had that leaf -- no FK re-run, no approximation.
The same holds for a chain peeled leaf by leaf: each step removes a leaf of the
rig left by the previous one.

What changes is the skeleton: the cond entry is sliced to the kept joints and
every index it carries is remapped, so the sample is indistinguishable from a
rig authored without those joints. The length scale ``L`` is re-derived from the
kept rest positions downstream, exactly as it would be for such a rig.
"""

from __future__ import annotations

import random
import re
from typing import Optional

import numpy as np

from data_loaders.truebones.truebones_utils.joint_parts import prefill_joint_parts
from data_loaders.truebones.truebones_utils.topology_relations import (
    refresh_topology_relations_in_object_cond,
)
from utils.rotation_conversions import rotation_6d_to_matrix_np

# Name tokens that mark a leaf as a terminator or attachment point rather than
# a body part: a Biped "Nub", an end site, a prop container.
TERMINATOR_TOKENS = frozenset({'nub', 'end', 'site', 'container'})
# A leaf without name tokens or with a zero name embedding counts as a terminator
# only when its local rotation never leaves this angle (degrees) from its first
# frame over the clip. Moving nameless fin rays and tentacle tips are retained.
LEAF_STATIC_MAX_DEG = 2.0
# Side words ignored when comparing a leaf's name stem with its parent's.
_SIDE_TOKENS = frozenset({'left', 'right', 'l', 'r'})
# A repeated-name leaf is excluded when its stem ends in one of these words.
# Other repeated chains, including ears and eyes, remain eligible.
SUPPORT_LIMB_TOKENS = frozenset({'leg', 'arm', 'foot', 'hoof', 'paw', 'hand'})
# Range of chain segments peeled on top of the terminator group, one leaf at a
# time; a mirror pair counts as one.
LEAF_DROP_MIN_SEGMENT_UNITS = 1
LEAF_DROP_MAX_SEGMENT_UNITS = 4
# A rig is never cut below this many joints.
LEAF_DROP_MIN_KEPT_JOINTS = 10
# Body-part groups (joint_parts labels) whose chain top joint is never dropped,
# so a rig keeps every limb and wing it has. Forelegs are labelled arm/hand.
# Tails and other chains may go whole.
LEAF_DROP_GUARDED_PART_GROUPS = (
    frozenset({'arm', 'hand', 'leg', 'foot'}),
    frozenset({'wing'}),
)

# Per-joint arrays/lists sliced along their first axis.
_ROW_KEYS = (
    'offsets',
    'rest_pose',
    'rest_pose_physical',
    'rest_pos_ric_hml',
    'tpose_rest_rotations',
    'joints_names_embs',
    'limb_chain_id',
    'limb_wing',
    'limb_reach',
    'joints_names',
    'canonical_joint_names',
    'canonical_bvh_joint_names',
    'joint_side_labels',
)
# Joint-index lists, each optionally paired with a parallel list of names.
_INDEX_LIST_KEYS = (
    ('face_joints', 'face_joint_names'),
    ('helper_joint_indices', None),
)
# Single joint indices; ``None`` / negative means "unset".
_INDEX_SCALAR_KEYS = (
    'translation_root_index',
    'forward_joint_index',
    'forward_base_joint_index',
)


def _int_or_none(value) -> Optional[int]:
    if value is None:
        return None
    value = int(value)
    return value if value >= 0 else None


def _protected_joints(entry, translation_root_index) -> set[int]:
    """Joints the augmentation must keep: the roots and every joint the
    preprocessing, the facing frame or the losses address by index."""
    protected = {0}
    for key in ('forward_joint_index', 'forward_base_joint_index'):
        index = _int_or_none(entry.get(key))
        if index is not None:
            protected.add(index)
    for index in (translation_root_index, entry.get('translation_root_index')):
        index = _int_or_none(index)
        if index is not None:
            protected.add(index)
    protected.update(int(index) for index in (entry.get('face_joints') or []))
    protected.update(_guarded_chain_roots(entry))
    return protected


_GUARDED_ROOTS_CACHE: dict[tuple, frozenset[int]] = {}


def _guarded_chain_roots(entry) -> frozenset[int]:
    """Top joint of every chain of a ``LEAF_DROP_GUARDED_PART_GROUPS`` group,
    from the prefilled body parts; cached per skeleton."""
    parents = np.asarray(entry['parents'], dtype=np.int64)
    names = [str(name) for name in entry['joints_names']]
    key = (
        tuple(names),
        parents.tobytes(),
        np.asarray(entry['offsets'], dtype=np.float64).tobytes(),
        tuple(entry.get('species_tags') or ()),
        str(entry.get('species_name') or entry.get('object_type') or ''),
        tuple(str(name) for name in entry.get('canonical_joint_names') or ()),
        tuple(str(side) for side in entry.get('joint_side_labels') or ()),
        _int_or_none(entry.get('translation_root_index')),
    )
    roots = _GUARDED_ROOTS_CACHE.get(key)
    if roots is None:
        prefill = prefill_joint_parts(entry)
        parts = [prefill[name]['part'] for name in names]
        roots = frozenset(
            joint for joint in range(parents.shape[0]) for group in LEAF_DROP_GUARDED_PART_GROUPS
            if parts[joint] in group and (parents[joint] < 0 or parts[parents[joint]] not in group)
        )
        _GUARDED_ROOTS_CACHE[key] = roots
    return roots


def static_joint_mask(motion, joint_indices) -> np.ndarray:
    """``True`` per entry of ``joint_indices`` whose local rotation stays within
    ``LEAF_STATIC_MAX_DEG`` of its first frame over the whole clip."""
    joint_indices = np.asarray(joint_indices, dtype=np.int64)
    if joint_indices.size == 0:
        return np.zeros((0,), dtype=np.bool_)
    rot = rotation_6d_to_matrix_np(
        np.asarray(motion[:, joint_indices, 3:9], dtype=np.float64)
    )  # [T, K, 3, 3]
    relative = np.einsum('kab,tkac->tkbc', rot[0], rot)
    cos = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    spread_deg = np.degrees(np.arccos(cos)).max(axis=0)
    return spread_deg < LEAF_STATIC_MAX_DEG


def _name_tokens(name) -> list[str]:
    """Lower-cased word tokens with trailing digits stripped, as the joint-mask
    helper filter reads names."""
    tokens = []
    for token in str(name or '').split():
        token = re.sub(r'[^a-z0-9]+', '', token.lower())
        token = re.sub(r'\d+$', '', token)
        if token and not token.isdigit():
            tokens.append(token)
    return tokens


def _name_stem(name) -> tuple[str, ...]:
    return tuple(token for token in _name_tokens(name) if token not in _SIDE_TOKENS)


def classify_leaves(entry, motion, leaves) -> dict[int, str]:
    """``{leaf: 'terminator' | 'segment'}`` for the leaves that may be dropped.

    * terminator -- named as one (``TERMINATOR_TOKENS``), or static over the clip
      with either no name tokens or a zero name embedding.
    * segment -- the trailing segment of a repeated chain: same name stem as its
      parent ("Tail 3" under "Tail 2", "Index 3 Left" under "Index 2 Left"),
      with a stem whose last word is not a support-limb token. This can include
      an animated ear or eye tip as well as a tail or finger tip.

    Leaves without a terminator marker or a repeated parent-name stem are not
    offered. The protected index set is applied by ``sample_leaf_drop``.
    """
    names = list(entry.get('canonical_joint_names') or entry.get('joints_names') or [])
    embs = entry.get('joints_names_embs')
    parents = np.asarray(entry['parents'], dtype=np.int64)
    leaves = [int(leaf) for leaf in leaves]
    nameless = [
        leaf for leaf in leaves
        if (embs is not None and not np.any(np.asarray(embs[leaf])))
        or not _name_tokens(names[leaf] if leaf < len(names) else '')
    ]
    static = dict(zip(nameless, static_joint_mask(motion, nameless).tolist())) if nameless else {}
    kinds = {}
    for leaf in leaves:
        name = names[leaf] if leaf < len(names) else ''
        if TERMINATOR_TOKENS & set(_name_tokens(name)) or static.get(leaf, False):
            kinds[leaf] = 'terminator'
            continue
        if leaf in static:
            continue
        stem = _name_stem(name)
        parent_name = names[parents[leaf]] if parents[leaf] < len(names) else ''
        if stem and stem == _name_stem(parent_name) and stem[-1] not in SUPPORT_LIMB_TOKENS:
            kinds[leaf] = 'segment'
    return kinds


def sample_leaf_drop(entry, motion, translation_root_index, rng=random) -> list[int]:
    """Pick the joints to remove from one training sample, or ``[]``.

    Candidates are unprotected leaves that ``classify_leaves`` offers; the
    protected set includes the top joint of every limb and wing chain. A leaf
    whose mirror twin is a candidate of the same kind forms a unit with it (both
    go or neither does); one whose twin is not is kept, so a symmetric rig stays
    symmetric. Every terminator unit is dropped, then a uniform number of
    segment units in ``[LEAF_DROP_MIN_SEGMENT_UNITS, LEAF_DROP_MAX_SEGMENT_UNITS]``
    is peeled one at a time. A joint whose children are all gone becomes a
    segment candidate when its name stem matches its parent's or the dropped
    child's ("Tail 1" once "Tail 2" is gone), so a chain can be peeled whole.
    Peeling stops early when the rig runs out of segments or the next unit would
    leave fewer than ``LEAF_DROP_MIN_KEPT_JOINTS`` joints; a terminator group
    that alone would do so cancels the cut.
    """
    parents = np.asarray(entry['parents'], dtype=np.int64)
    joint_count = int(parents.shape[0])
    if joint_count <= LEAF_DROP_MIN_KEPT_JOINTS:
        return []
    child_count = np.bincount(parents[parents >= 0], minlength=joint_count)
    protected = _protected_joints(entry, translation_root_index)
    leaves = [
        joint for joint in range(joint_count)
        if parents[joint] >= 0 and child_count[joint] == 0 and joint not in protected
    ]
    kinds = classify_leaves(entry, motion, leaves)
    partners = list(entry.get('symmetry_partner_indices') or [-1] * joint_count)
    terminator_units = _mirror_units({j for j, kind in kinds.items() if kind == 'terminator'}, partners)
    segments = {j for j, kind in kinds.items() if kind == 'segment'}
    if not terminator_units and not segments:
        return []

    dropped = {joint for unit in terminator_units for joint in unit}
    if joint_count - len(dropped) < LEAF_DROP_MIN_KEPT_JOINTS:
        return []
    names = list(entry.get('canonical_joint_names') or entry.get('joints_names') or [])
    stems = [_name_stem(names[j]) if j < len(names) else () for j in range(joint_count)]
    remaining = child_count.copy()

    def drop(joint):
        """Drop ``joint``; offer its parent once that is a leaf continuing the chain."""
        dropped.add(joint)
        segments.discard(joint)
        parent = int(parents[joint])
        remaining[parent] -= 1
        if remaining[parent] or parent in protected or parents[parent] < 0:
            return
        stem = stems[parent]
        if stem and stem[-1] not in SUPPORT_LIMB_TOKENS and stem in (stems[joint], stems[parents[parent]]):
            segments.add(parent)

    for joint in sorted(dropped):
        drop(joint)
    for _ in range(rng.randint(LEAF_DROP_MIN_SEGMENT_UNITS, LEAF_DROP_MAX_SEGMENT_UNITS)):
        units = _mirror_units(segments, partners)
        if not units:
            break
        unit = rng.choice(units)
        if joint_count - len(dropped) - len(unit) < LEAF_DROP_MIN_KEPT_JOINTS:
            break
        for joint in unit:
            drop(joint)
    return sorted(dropped)


def _mirror_units(candidates, partners) -> list[tuple[int, ...]]:
    """``candidates`` grouped with their mirror twins: a twin pair when both are
    candidates, a single when the joint has no twin; a joint whose twin is not
    a candidate is left out."""
    units = []
    for joint in sorted(candidates):
        partner = int(partners[joint]) if joint < len(partners) else -1
        if partner < 0 or partner == joint:
            units.append((joint,))
        elif partner in candidates and joint < partner:
            units.append((joint, partner))
    return units


def drop_joints_from_cond(entry, dropped):
    """Return ``(keep, new_entry)``: ``entry`` restricted to the joints not in
    ``dropped``, with every per-joint field sliced and every joint index
    remapped. ``dropped`` must hold leaves only. ``entry`` is not modified."""
    parents = np.asarray(entry['parents'], dtype=np.int64)
    joint_count = int(parents.shape[0])
    dropped_set = {int(joint) for joint in dropped}
    keep = np.asarray([j for j in range(joint_count) if j not in dropped_set], dtype=np.int64)
    new_index = np.full((joint_count,), -1, dtype=np.int64)
    new_index[keep] = np.arange(keep.shape[0])
    if np.any(np.isin(parents[keep], sorted(dropped_set))):
        raise ValueError("drop_joints_from_cond removes leaves only; a kept joint's parent was dropped.")

    def remap(index):
        index = _int_or_none(index)
        return None if index is None else int(new_index[index])

    new_entry = dict(entry)
    new_entry['parents'] = np.where(parents[keep] >= 0, new_index[np.maximum(parents[keep], 0)], -1)

    for key in _ROW_KEYS:
        value = entry.get(key)
        if value is None:
            continue
        if isinstance(value, np.ndarray):
            new_entry[key] = value[keep]
        else:
            new_entry[key] = [value[j] for j in keep]

    for index_key, names_key in _INDEX_LIST_KEYS:
        indices = entry.get(index_key)
        if indices is None:
            continue
        names = entry.get(names_key) if names_key else None
        kept = [position for position, joint in enumerate(indices) if int(joint) not in dropped_set]
        new_entry[index_key] = [int(new_index[int(indices[position])]) for position in kept]
        if names is not None and len(names) == len(indices):
            new_entry[names_key] = [names[position] for position in kept]

    for key in _INDEX_SCALAR_KEYS:
        if key in entry:
            new_entry[key] = remap(entry[key])

    partners = entry.get('symmetry_partner_indices')
    if partners is not None:
        new_entry['symmetry_partner_indices'] = [
            -1 if int(partners[j]) < 0 or int(partners[j]) in dropped_set else int(new_index[int(partners[j])])
            for j in keep
        ]
    pairs = entry.get('symmetric_joint_pairs')
    if pairs is not None:
        pair_names = entry.get('symmetric_joint_pair_names')
        kept = [i for i, pair in enumerate(pairs) if not dropped_set & {int(j) for j in pair}]
        new_entry['symmetric_joint_pairs'] = [[int(new_index[int(j)]) for j in pairs[i]] for i in kept]
        if pair_names is not None and len(pair_names) == len(pairs):
            new_entry['symmetric_joint_pair_names'] = [pair_names[i] for i in kept]
    chains = entry.get('kinematic_chains')
    if chains is not None:
        new_chains = [[int(new_index[int(j)]) for j in chain if int(j) not in dropped_set] for chain in chains]
        new_entry['kinematic_chains'] = [chain for chain in new_chains if chain]

    refresh_topology_relations_in_object_cond(new_entry)
    return keep, new_entry
