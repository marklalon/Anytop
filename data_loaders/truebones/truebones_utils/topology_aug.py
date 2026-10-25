"""Topology augmentation: leaf drop or chain merge, one per draw.

Leaf drop lives in ``leaf_drop.py``; this module adds chain merge and the
draw between them (``augment_topology``).

Merge. Removing a joint that has one child joins its two bones into one. The
position channels are absolute, so every kept row keeps its position; the
child's local rotation becomes the removed joint's composed with its own, so
every kept joint keeps its global rotation. What a merge cannot keep is the
bend at the removed joint: forward kinematics of the merged rig puts the child
where the straight bone points. A merge is drawn only where that residual stays
within ``CHAIN_MERGE_MAX_RESIDUAL`` of the merged bone on every frame -- a twist
joint, a helper riding its parent, a chain that barely bends.

The cond entry becomes that of a rig authored without the joint: every
per-joint field is rebuilt and every index remapped, and ``L`` is re-derived
from the new rest downstream.
"""

from __future__ import annotations

import random

import numpy as np

from data_loaders.truebones.truebones_utils.bone_length_aug import _parent_order
from data_loaders.truebones.truebones_utils.leaf_drop import (
    LEAF_DROP_MIN_KEPT_JOINTS,
    _int_or_none,
    _mirror_units,
    drop_joints_from_cond,
    sample_leaf_drop,
)
from utils.rotation_conversions import rotation_6d_to_matrix_np

# Range of merge units per applied draw; a mirror pair counts as one.
TOPOLOGY_AUG_MIN_UNITS = 1
TOPOLOGY_AUG_MAX_UNITS = 3
# Largest forward-kinematics residual of a merged bone, relative to its rest
# length, on any frame.
CHAIN_MERGE_MAX_RESIDUAL = 0.1


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

def _apply_leaf_drop(motion, cond, translation_root_index, target, rng):
    dropped = sample_leaf_drop(cond, motion, translation_root_index, rng=rng)
    if not dropped:
        return None
    keep, new_cond = drop_joints_from_cond(cond, dropped)
    return motion[:, keep], new_cond, dropped, _kept_index(keep, cond)


def _apply_merge(motion, cond, translation_root_index, target, rng):
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
    'merge': _apply_merge,
}
TOPOLOGY_AUG_KINDS = tuple(_TOPOLOGY_AUGS)


def augment_topology(motion, cond, translation_root_index, *,
                     kinds=TOPOLOGY_AUG_KINDS, rng=random):
    """Return ``(physical motion, private cond, diagnostics)``; a skip is a no-op.

    One of ``kinds`` -- leaf drop (``leaf_drop.py``) or chain merge -- is
    drawn uniformly; when the rig offers it nothing, the other follows. A
    merge applies a uniform number of units in
    ``[TOPOLOGY_AUG_MIN_UNITS, TOPOLOGY_AUG_MAX_UNITS]``, never takes the rig
    below ``LEAF_DROP_MIN_KEPT_JOINTS`` and skips any unit whose merged bones
    fail the residual test, so fewer units may land. ``info['joints']`` are
    the dropped or merged joints (old indices), and ``info['new_index']`` maps each old joint to its new index (-1 when gone).
    """
    info = {'applied': False, 'kind': None, 'joints': [], 'skip_reason': '',
            'translation_root_index': int(translation_root_index)}
    order = list(kinds)
    rng.shuffle(order)
    target = rng.randint(TOPOLOGY_AUG_MIN_UNITS, TOPOLOGY_AUG_MAX_UNITS)
    for kind in order:
        result = _TOPOLOGY_AUGS[kind](motion, cond, translation_root_index, target, rng)
        if result is None:
            continue
        out, new_cond, joints, new_index = result
        info.update(applied=True, kind=kind, joints=[int(j) for j in joints], new_index=new_index,
                    translation_root_index=int(new_index[int(translation_root_index)]))
        return out, new_cond, info
    info['skip_reason'] = 'no leaf to drop or joint to merge'
    return motion, cond, info
