"""Topology augmentation: leaf drop, bone split or chain merge, one per draw.

Leaf drop lives in ``leaf_drop.py``; this module adds the other two and the
draw between them (``augment_topology``).

Split. A joint inserted at the midpoint of a bone with an identity local
rotation moves nothing: its parent's global rotation carries both halves, so
the bone's child keeps its own local rotation and every stored row stays a
valid description of the motion. The new row is the midpoint of the two
positions and an identity rotation -- exact, like a leaf drop.

Merge. Removing a joint that has one child joins its two bones into one. The
position channels are absolute, so every kept row keeps its position; the
child's local rotation becomes the removed joint's composed with its own, so
every kept joint keeps its global rotation. What a merge cannot keep is the
bend at the removed joint: forward kinematics of the merged rig puts the child
where the straight bone points. A merge is drawn only where that residual stays
within ``CHAIN_MERGE_MAX_RESIDUAL`` of the merged bone on every frame -- a twist
joint, a helper riding its parent, a chain that barely bends.

Either way the cond entry becomes that of a rig authored with (without) the
joint: every per-joint field is rebuilt and every index remapped, and ``L`` is
re-derived from the new rest downstream.
"""

from __future__ import annotations

import random

import numpy as np

from data_loaders.truebones.truebones_utils.bone_length_aug import _parent_order
from data_loaders.truebones.truebones_utils.canonical_features import root_relative_positions
from data_loaders.truebones.truebones_utils.leaf_drop import (
    _INDEX_LIST_KEYS,
    _INDEX_SCALAR_KEYS,
    _ROW_KEYS,
    LEAF_DROP_MIN_KEPT_JOINTS,
    _int_or_none,
    _mirror_units,
    drop_joints_from_cond,
    sample_leaf_drop,
)
from data_loaders.truebones.truebones_utils.param_utils import MAX_JOINTS
from data_loaders.truebones.truebones_utils.topology_relations import (
    refresh_topology_relations_in_object_cond,
)
from utils.rotation_conversions import rotation_6d_to_matrix_np

# Range of split or merge units per applied draw; a mirror pair counts as one.
TOPOLOGY_AUG_MIN_UNITS = 1
TOPOLOGY_AUG_MAX_UNITS = 3
# Largest forward-kinematics residual of a merged bone, relative to its rest
# length, on any frame.
CHAIN_MERGE_MAX_RESIDUAL = 0.1
# Per-joint rest-position keys; a split joint sits at its bone's midpoint.
_REST_POSITION_KEYS = ('rest_pos_ric_hml', 'rest_pose', 'rest_pose_physical')
_IDENTITY_6D = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
_IDENTITY_QUAT = np.asarray([1.0, 0.0, 0.0, 0.0])
# Suffix of a split joint's name, per name list.
_SPLIT_NAME_SUFFIX = {
    'joints_names': '_Mid',
    'canonical_joint_names': ' Mid',
    'canonical_bvh_joint_names': 'Mid',
}


def validate_topology_aug(probability):
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f'topology_aug_prob must be finite and in [0, 1], got {probability}.')


def _partners(entry, joint_count) -> list[int]:
    partners = entry.get('symmetry_partner_indices')
    if partners is None or len(partners) != joint_count:
        return [-1] * joint_count
    return [int(partner) for partner in partners]


def _matrix_to_rotation_6d(mat):
    return np.concatenate([mat[..., :, 0], mat[..., :, 1]], axis=-1)


def _protected_joints(entry, translation_root_index) -> set[int]:
    """Joints a merge keeps: the roots and every joint addressed by index."""
    protected = {0}
    for index in (translation_root_index, entry.get('translation_root_index'),
                  entry.get('forward_joint_index'), entry.get('forward_base_joint_index')):
        index = _int_or_none(index)
        if index is not None:
            protected.add(index)
    protected.update(int(index) for index in (entry.get('face_joints') or []))
    return protected


# ── Split ──

def split_candidates(entry) -> list[tuple[int, ...]]:
    """Units of bones to split, each named by its child joint.

    A bone qualifies when it is at least the rig's mean bone length: a short
    distal bone (a toe, a nub) gains nothing from a midpoint. A lateral bone
    goes with its mirror twin, the bone between the two joints' partners.
    """
    parents = np.asarray(entry['parents'], dtype=np.int64)
    joint_count = int(parents.shape[0])
    lengths = np.linalg.norm(np.asarray(entry['offsets'], dtype=np.float64), axis=-1)
    has_parent = parents >= 0
    live = has_parent & (lengths > max(1e-6, float(lengths[has_parent].max(initial=0.0)) * 1e-4))
    if not live.any():
        return []
    mean = float(lengths[live].mean())
    partners = _partners(entry, joint_count)
    candidates = set()
    for child in np.flatnonzero(live & (lengths >= mean * (1.0 - 1e-6))):
        child = int(child)
        twin = partners[child]
        if twin >= 0 and twin != child:
            parent, twin_parent = int(parents[child]), int(parents[twin])
            mirror_parent = partners[parent] if partners[parent] >= 0 else parent
            if twin_parent != mirror_parent:
                continue
        candidates.add(child)
    return _mirror_units(candidates, partners)


def _unique_name(name, taken):
    candidate, suffix = name, 2
    while candidate in taken:
        candidate, suffix = f'{name}{suffix}', suffix + 1
    taken.add(candidate)
    return candidate


def split_bones_in_cond(entry, children):
    """Return ``(source, new_index, mid_index, new_entry)`` with a joint inserted
    at the midpoint of the bone above each joint of ``children``.

    ``source[k]`` is the old joint whose per-joint row new joint ``k`` copies,
    ``new_index[j]`` old joint ``j``'s new index and ``mid_index[c]`` the new
    index of the joint inserted above ``c``. A split joint goes right before
    its child, so a parents-first order stays one. It carries its bone's name
    and name embedding (its parent's, or its child's when the bone leaves the
    midline, so name and side agree), the child's side label, an identity bind
    rotation and half the bone; the child keeps the other half. ``entry`` is
    not modified.
    """
    parents = np.asarray(entry['parents'], dtype=np.int64)
    joint_count = int(parents.shape[0])
    split = sorted({int(child) for child in children})
    if any(parents[child] < 0 for child in split):
        raise ValueError('split_bones_in_cond splits bones; a root has none.')
    split_set = set(split)
    sides = entry.get('joint_side_labels')
    source, new_index, mid_index = [], np.empty(joint_count, dtype=np.int64), {}
    for joint in range(joint_count):
        if joint in split_set:
            parent = int(parents[joint])
            same_side = sides is None or sides[parent] == sides[joint]
            mid_index[joint] = len(source)
            source.append(parent if same_side else joint)
        new_index[joint] = len(source)
        source.append(joint)
    source = np.asarray(source, dtype=np.int64)
    mids = np.asarray([mid_index[child] for child in split], dtype=np.int64)
    split_children = np.asarray(split, dtype=np.int64)
    split_parents = parents[split_children]

    new_parents = np.full(source.shape[0], -1, dtype=np.int64)
    new_parents[new_index] = np.where(parents >= 0, new_index[np.maximum(parents, 0)], -1)
    new_parents[new_index[split_children]] = mids
    new_parents[mids] = new_index[split_parents]

    new_entry = dict(entry)
    new_entry['parents'] = new_parents.astype(np.asarray(entry['parents']).dtype)
    for key in _ROW_KEYS:
        value = entry.get(key)
        if value is None:
            continue
        if isinstance(value, np.ndarray):
            new_entry[key] = value[source]
        else:
            new_entry[key] = [value[j] for j in source]

    offsets = np.asarray(entry['offsets'])
    half = offsets[split_children] * 0.5
    new_offsets = np.array(new_entry['offsets'], copy=True)
    new_offsets[mids] = half
    new_offsets[new_index[split_children]] = half
    new_entry['offsets'] = new_offsets
    for key in _REST_POSITION_KEYS:
        if entry.get(key) is None:
            continue
        rest = np.asarray(entry[key])
        new_rest = np.array(new_entry[key], copy=True)
        new_rest[mids, :3] = (rest[split_parents, :3] + rest[split_children, :3]) * 0.5
        new_entry[key] = new_rest
    if entry.get('tpose_rest_rotations') is not None:
        tpose = np.array(new_entry['tpose_rest_rotations'], copy=True)
        tpose[mids] = _IDENTITY_QUAT.astype(tpose.dtype)
        new_entry['tpose_rest_rotations'] = tpose
    if sides is not None:
        for mid, child in zip(mids, split_children):
            new_entry['joint_side_labels'][mid] = sides[child]
    for key, suffix in _SPLIT_NAME_SUFFIX.items():
        names = new_entry.get(key)
        if names is None:
            continue
        taken = {str(name) for name in names}
        for mid in mids:
            names[mid] = _unique_name(f'{names[mid]}{suffix}', taken)

    for index_key, _names_key in _INDEX_LIST_KEYS:
        if entry.get(index_key) is not None:
            new_entry[index_key] = [int(new_index[int(joint)]) for joint in entry[index_key]]
    for key in _INDEX_SCALAR_KEYS:
        if key in entry:
            index = _int_or_none(entry[key])
            new_entry[key] = None if index is None else int(new_index[index])

    partners = _partners(entry, joint_count)
    new_partners = [-1] * source.shape[0]
    for joint in range(joint_count):
        if partners[joint] >= 0:
            new_partners[new_index[joint]] = int(new_index[partners[joint]])
    for child in split:
        twin = partners[child]
        if twin in mid_index and twin != child:
            new_partners[mid_index[child]] = mid_index[twin]
    if entry.get('symmetry_partner_indices') is not None:
        new_entry['symmetry_partner_indices'] = new_partners
    pairs = entry.get('symmetric_joint_pairs')
    if pairs is not None:
        new_pairs = [[int(new_index[int(j)]) for j in pair] for pair in pairs]
        pair_names = entry.get('symmetric_joint_pair_names')
        new_pair_names = list(pair_names) if pair_names is not None and len(pair_names) == len(pairs) else None
        names = new_entry.get('joints_names')
        for child in split:
            twin = partners[child]
            if twin in mid_index and child < twin:
                new_pairs.append([mid_index[child], mid_index[twin]])
                if new_pair_names is not None and names is not None:
                    new_pair_names.append([names[mid_index[child]], names[mid_index[twin]]])
        new_entry['symmetric_joint_pairs'] = new_pairs
        if new_pair_names is not None:
            new_entry['symmetric_joint_pair_names'] = new_pair_names
    chains = entry.get('kinematic_chains')
    if chains is not None:
        new_entry['kinematic_chains'] = [
            [index for j in chain for index in
             ((mid_index[int(j)], int(new_index[int(j)])) if int(j) in mid_index else (int(new_index[int(j)]),))]
            for chain in chains
        ]
    # Never reuse an L explicitly cached on an input cond.
    new_entry.pop('rest_length_scale', None)
    refresh_topology_relations_in_object_cond(new_entry)
    return source, new_index, mid_index, new_entry


def split_bones_in_motion(motion, entry, mid_index, source):
    """``motion`` on the split rig: each inserted row at its bone's midpoint
    with an identity rotation; every other row as stored."""
    parents = np.asarray(entry['parents'], dtype=np.int64)
    out = np.array(motion[:, source], dtype=np.float32, copy=True)
    positions = np.asarray(root_relative_positions(motion[..., :3]), dtype=np.float64)
    for child, mid in mid_index.items():
        parent = int(parents[child])
        out[:, mid] = 0.0
        out[:, mid, :3] = (positions[:, parent] + positions[:, child]) * 0.5
        out[:, mid, 3:9] = _IDENTITY_6D
    return out


# ── Merge ──

def _forward_kinematics(motion, entry):
    """Global rotations ``[T, J, 3, 3]`` and rotation-only FK positions
    ``[T, J, 3]`` (root at the origin) of a physical clip."""
    parents = np.asarray(entry['parents'], dtype=np.int64)
    offsets = np.asarray(entry['offsets'], dtype=np.float64)
    order, _children = _parent_order(parents)
    local = rotation_6d_to_matrix_np(np.asarray(motion[..., 3:9], dtype=np.float64))
    global_rot = local.copy()
    positions = np.zeros(local.shape[:2] + (3,), dtype=np.float64)
    for joint in order:
        parent = parents[joint]
        if parent >= 0:
            global_rot[:, joint] = global_rot[:, parent] @ local[:, joint]
            positions[:, joint] = positions[:, parent] + global_rot[:, parent] @ offsets[joint]
    return global_rot, positions


class _MergeState:
    """Working parents and offsets while merges are drawn; the globals and FK
    positions of the source clip stay those of every kept joint."""

    def __init__(self, entry, global_rot, positions):
        self.parents = np.array(entry['parents'], dtype=np.int64, copy=True)
        self.offsets = np.array(entry['offsets'], dtype=np.float64, copy=True)
        self.removed: list[int] = []
        self.global_rot = global_rot
        self.positions = positions

    def copy(self):
        other = object.__new__(_MergeState)
        other.parents = self.parents.copy()
        other.offsets = self.offsets.copy()
        other.removed = list(self.removed)
        other.global_rot, other.positions = self.global_rot, self.positions
        return other

    def merge(self, joint) -> bool:
        """Remove ``joint`` (one child); ``False`` when the merged bone's FK
        residual exceeds ``CHAIN_MERGE_MAX_RESIDUAL``."""
        (child,) = np.flatnonzero(self.parents == joint)
        parent = int(self.parents[joint])
        offset = self.offsets[joint] + self.offsets[child]
        length = float(np.linalg.norm(offset))
        if length <= 1e-9:
            return False
        actual = self.positions[:, child] - self.positions[:, parent]
        predicted = self.global_rot[:, parent] @ offset
        residual = np.linalg.norm(actual - predicted, axis=-1).max() / length
        if not residual <= CHAIN_MERGE_MAX_RESIDUAL:
            return False
        self.parents[child] = parent
        self.offsets[child] = offset
        self.parents[joint] = -2  # detached; dropped as a leaf
        self.removed.append(int(joint))
        return True


def merge_candidates(entry, translation_root_index) -> list[tuple[int, ...]]:
    """Units of joints a merge may remove: unprotected joints with one child;
    a joint whose mirror twin is not a candidate is left out."""
    parents = np.asarray(entry['parents'], dtype=np.int64)
    joint_count = int(parents.shape[0])
    child_count = np.bincount(parents[parents >= 0], minlength=joint_count)
    protected = _protected_joints(entry, translation_root_index)
    candidates = {
        joint for joint in range(joint_count)
        if parents[joint] >= 0 and child_count[joint] == 1 and joint not in protected
    }
    return _mirror_units(candidates, _partners(entry, joint_count))


def merge_joints(motion, entry, state):
    """``(keep, motion, new_entry)`` with ``state.removed`` merged away: each
    removed joint's child takes its bone, its local rotation and its bind."""
    order, _children = _parent_order(np.asarray(entry['parents'], dtype=np.int64))
    removed_set = set(state.removed)
    removed = [int(joint) for joint in order if joint in removed_set]
    local = rotation_6d_to_matrix_np(np.asarray(motion[..., 3:9], dtype=np.float64))
    tpose = entry.get('tpose_rest_rotations')
    bind = None
    if tpose is not None:
        from motion_lib.Quaternions import Quaternions
        bind = Quaternions(np.asarray(tpose, dtype=np.float64)).transforms()
    source_parents = np.asarray(entry['parents'], dtype=np.int64)
    # Each removed joint hands its rotation to the kept joint below it; a run
    # of removed joints composes onto that joint bottom-up, R_a R_b R_c.
    heir = {}
    for joint in removed:
        (child,) = np.flatnonzero(source_parents == joint)
        heir[joint] = int(child)
    for joint in reversed(removed):
        if heir[joint] in heir:
            heir[joint] = heir[heir[joint]]
    for joint in reversed(removed):
        child = heir[joint]
        local[:, child] = local[:, joint] @ local[:, child]
        if bind is not None:
            bind[child] = bind[joint] @ bind[child]

    work = dict(entry)
    work['parents'] = np.where(state.parents == -2, 0, state.parents).astype(
        np.asarray(entry['parents']).dtype)
    work['offsets'] = state.offsets.astype(np.asarray(entry['offsets']).dtype)
    if bind is not None:
        from motion_lib.Quaternions import Quaternions
        quats = Quaternions.from_transforms(bind).qs
        flip = np.sum(quats * np.asarray(tpose, dtype=np.float64), axis=-1) < 0
        quats[flip] *= -1
        work['tpose_rest_rotations'] = quats.astype(np.asarray(tpose).dtype)
    keep, new_entry = drop_joints_from_cond(work, state.removed)
    new_entry.pop('rest_length_scale', None)
    out = np.array(motion, dtype=np.float32, copy=True)
    out[..., 3:9] = _matrix_to_rotation_6d(local)
    return keep, out[:, keep], new_entry


# ── Draw ──

def _apply_leaf_drop(motion, cond, translation_root_index, max_joints, target, rng):
    dropped = sample_leaf_drop(cond, motion, translation_root_index, rng=rng)
    if not dropped:
        return None
    keep, new_cond = drop_joints_from_cond(cond, dropped)
    return motion[:, keep], new_cond, dropped, _kept_index(keep, cond)


def _apply_split(motion, cond, translation_root_index, max_joints, target, rng):
    room = int(max_joints) - int(np.asarray(cond['parents']).shape[0])
    units = [unit for unit in split_candidates(cond) if len(unit) <= room]
    if not units:
        return None
    rng.shuffle(units)
    chosen = []
    for unit in units:
        if len(chosen) == target:
            break
        if sum(map(len, chosen)) + len(unit) <= room:
            chosen.append(unit)
    children = sorted(joint for unit in chosen for joint in unit)
    source, new_index, mid_index, new_cond = split_bones_in_cond(cond, children)
    return split_bones_in_motion(motion, cond, mid_index, source), new_cond, children, new_index


def _apply_merge(motion, cond, translation_root_index, max_joints, target, rng):
    joint_count = int(np.asarray(cond['parents']).shape[0])
    if joint_count <= LEAF_DROP_MIN_KEPT_JOINTS:
        return None
    units = merge_candidates(cond, translation_root_index)
    if not units:
        return None
    rng.shuffle(units)
    state = _MergeState(cond, *_forward_kinematics(motion, cond))
    accepted = 0
    for unit in units:
        if accepted == target:
            break
        if joint_count - len(state.removed) - len(unit) < LEAF_DROP_MIN_KEPT_JOINTS:
            continue
        trial = state.copy()
        if all(trial.merge(joint) for joint in unit):
            state, accepted = trial, accepted + 1
    if not accepted:
        return None
    keep, out, new_cond = merge_joints(motion, cond, state)
    return out, new_cond, sorted(state.removed), _kept_index(keep, cond)


def _kept_index(keep, cond):
    new_index = np.full(int(np.asarray(cond['parents']).shape[0]), -1, dtype=np.int64)
    new_index[keep] = np.arange(keep.shape[0])
    return new_index


_TOPOLOGY_AUGS = {
    'leaf_drop': _apply_leaf_drop,
    'split': _apply_split,
    'merge': _apply_merge,
}
TOPOLOGY_AUG_KINDS = tuple(_TOPOLOGY_AUGS)


def augment_topology(motion, cond, translation_root_index, max_joints=MAX_JOINTS, *,
                     kinds=TOPOLOGY_AUG_KINDS, rng=random):
    """Return ``(physical motion, private cond, diagnostics)``; a skip is a no-op.

    One of ``kinds`` -- leaf drop (``leaf_drop.py``), bone split or chain
    merge -- is drawn uniformly; when the rig offers it nothing, the others
    follow in random order. Split and merge apply a uniform number of units in
    ``[TOPOLOGY_AUG_MIN_UNITS, TOPOLOGY_AUG_MAX_UNITS]``. A split never takes
    the rig past ``max_joints``; a merge never below
    ``LEAF_DROP_MIN_KEPT_JOINTS`` and skips any unit whose merged bones fail
    the residual test, so fewer units may land. ``info['joints']`` are the
    dropped or merged joints (old indices) or the split bones' children, and
    ``info['new_index']`` maps each old joint to its new index (-1 when gone).
    """
    info = {'applied': False, 'kind': None, 'joints': [], 'skip_reason': '',
            'translation_root_index': int(translation_root_index)}
    order = list(kinds)
    rng.shuffle(order)
    target = rng.randint(TOPOLOGY_AUG_MIN_UNITS, TOPOLOGY_AUG_MAX_UNITS)
    for kind in order:
        result = _TOPOLOGY_AUGS[kind](motion, cond, translation_root_index, max_joints, target, rng)
        if result is None:
            continue
        out, new_cond, joints, new_index = result
        info.update(applied=True, kind=kind, joints=[int(j) for j in joints], new_index=new_index,
                    translation_root_index=int(new_index[int(translation_root_index)]))
        return out, new_cond, info
    info['skip_reason'] = 'no leaf to drop, bone to split or joint to merge'
    return motion, cond, info
