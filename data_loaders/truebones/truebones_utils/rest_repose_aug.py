"""Rest re-pose augmentation: one clip, re-expressed against a different rest pose.

Each joint with children gets a local rest rotation ``Q_j``; the new rest is
the forward kinematics of those rotations on the stored offsets, so every bone
keeps its length and the rig its topology. The world motion is unchanged: the
position channels are absolute and are kept as they are, and the
local rotations are re-expressed so that forward kinematics on the new rest
lands on the same world pose.

With ``G`` the new rest's global rotations (``G_j = G_parent(j) Q_j``), the
stored rest of identity rotations becomes one of identity rotations again on
the offsets ``o'_j = G_parent(j) o_j``, and a frame's local rotation becomes
``R'_j = G_parent(j) R_j Q_j^T G_parent(j)^T``.

The rest-pose geometry is what a rig is recognized by (bone directions and
heights); the augmentation gives every rig a spread of rests, so the motion
cannot be keyed to one exact rest. A draw keeps the mirror symmetry (a partner
gets the mirrored rotation, a midline joint only bends in the sagittal plane)
and only turns a joint about an axis that leaves the canonical rotation
channel exactly unchanged (see ``_subtree_valid``).

The structural channel names the stored rig's joints (which side, which
sibling) and is not rest geometry: callers keep the one built before this
augmentation instead of rebuilding it from the new rest, where a bent limb
could cross the midline or reorder siblings.
"""
from __future__ import annotations

import random

import numpy as np

from data_loaders.truebones.truebones_utils.bone_length_aug import _parent_order
from data_loaders.truebones.truebones_utils.canonical_features import (
    _TWIST_REFERENCE_AXES, _twist_references, rest_bone_frames)

# Largest rest rotation of a joint whose subtree is a vanishing part of the rig.
DEFAULT_REST_REPOSE_MAX_DEG = 20.0
# Largest deviation of a turned twist reference from the one the new rest derives.
_TWIST_TOLERANCE = 1e-6
# A rest joint closer to the sagittal plane than this is on the midline.
_SIDE_EPS = 1e-4
# Reflection across the sagittal plane: the canonical rest faces +Z with X lateral.
_MIRROR = np.diag([-1.0, 1.0, 1.0])


def validate_rest_repose_aug(probability, max_deg):
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError('rest_repose_prob must be finite and in [0, 1].')
    if not np.isfinite(max_deg) or not 0.0 <= max_deg <= 90.0:
        raise ValueError('rest_repose_max_deg must be finite and in [0, 90].')


def _axis_angle_matrix(axis, angle):
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + s * k + (1.0 - c) * (k @ k)


def _rotation_6d_to_matrix(d6):
    from utils.rotation_conversions import rotation_6d_to_matrix_np
    return rotation_6d_to_matrix_np(np.asarray(d6, dtype=np.float64))


def _matrix_to_rotation_6d(mat):
    # Columns 0 and 1, the layout of Quaternions.rotation_matrix(cont6d=True).
    return np.concatenate([mat[..., :, 0], mat[..., :, 1]], axis=-1)


def _subtree_sizes(order, parents):
    sizes = np.ones(len(parents), dtype=np.int64)
    for j in order[::-1]:
        if parents[j] >= 0:
            sizes[parents[j]] += sizes[j]
    return sizes


def _partners(cond, joint_count):
    partners = cond.get('symmetry_partner_indices')
    if partners is None or len(partners) != joint_count:
        return np.full(joint_count, -1, dtype=np.int64)
    partners = np.asarray(partners, dtype=np.int64)
    valid = (partners >= 0) & (partners < joint_count) & (partners != np.arange(joint_count))
    return np.where(valid, partners, -1)


def _subtree_layout(children, joint):
    """Joints under ``joint`` in breadth-first order, with what batches their FK.

    Returns ``(nodes, levels)``: each level is ``(slice, parent_pos)`` -- the
    slice of ``nodes`` at one depth and the position in ``nodes`` of each
    one's parent (the first level is the joint itself).
    """
    nodes = [joint]
    levels = [(slice(0, 1), None)]
    start = 0
    while start < len(nodes):
        stop = len(nodes)
        parent_pos = []
        for k in range(start, stop):
            for c in children[nodes[k]]:
                nodes.append(int(c))
                parent_pos.append(k)
        if len(nodes) > stop:
            levels.append((slice(stop, len(nodes)), np.asarray(parent_pos)))
        start = stop
    return np.asarray(nodes), levels


def _subtree_valid(rotations, parents, offsets, rest, frames, tri, layout):
    """True when the draws under a joint are an exact, side-keeping re-pose.

    Exact: the canonical rotation ``G A`` stays unchanged iff the turned twist
    reference ``G p`` is the one the new rest derives from the turned bone
    direction ``G d``; a twist reference axis fixed in the world only
    guarantees that for turns about it (or in the plane of it and the bone).
    Side-keeping: no joint crosses the root's sagittal plane, so a bent limb
    never reaches over to its partner's side (an X-shaped pose).
    """
    nodes, levels = layout
    joint = int(nodes[0])
    path = [joint]
    while parents[path[-1]] >= 0:
        path.append(int(parents[path[-1]]))
    path.reverse()
    position, frame = rest[path[0]], np.eye(3)
    for a, b in zip(path[:-1], path[1:]):
        frame = frame @ rotations[a]
        position = position + frame @ offsets[b]
    # ``frame`` is the global rotation of the joint's parent here.
    count = len(nodes)
    globals_ = np.empty((count, 3, 3))
    positions = np.empty((count, 3))
    globals_[0] = frame @ rotations[joint]
    positions[0] = position
    for rows, parent_pos in levels[1:]:
        parent_global = globals_[parent_pos]
        globals_[rows] = parent_global @ rotations[nodes[rows]]
        positions[rows] = positions[parent_pos] + np.einsum(
            'kij,kj->ki', parent_global, offsets[nodes[rows]])
    midline = rest[tri, 0]
    side = rest[nodes, 0] - midline
    sided = np.abs(side) > _SIDE_EPS
    if np.any((positions[sided, 0] - midline) * side[sided] <= 0.0):
        return False
    d = np.einsum('kij,kj->ki', globals_, frames[nodes, :, 0])
    norm = np.linalg.norm(d, axis=-1)
    live = norm > 1e-9
    if not live.any():
        return True
    expected = _twist_references(d[live] / norm[live, None])
    twist = np.einsum('kij,kj->ki', globals_[live], frames[nodes[live], :, 1])
    return bool(np.all(np.linalg.norm(twist - expected, axis=-1) <= _TWIST_TOLERANCE))


def _draw_rest_rotations(cond, order, children, tri, max_rad, rng):
    """``[J, 3, 3]`` local rest rotations, identity where nothing is drawn.

    A joint's largest angle shrinks with the share of the rig it moves, so a
    hub carrying most of the body barely turns and a root never does. A paired
    joint turns about one of the twist reference axes and hands the mirrored
    rotation to its partner; an unpaired joint turns about the lateral axis
    only. A draw is kept only when every frame under the joint (and its
    partner) keeps its canonical rotation exactly and stays on its side of
    the midline (``_subtree_valid``).
    """
    parents = np.asarray(cond['parents'], dtype=np.int64)
    offsets = np.asarray(cond['offsets'], dtype=np.float64)
    joint_count = len(parents)
    lengths = np.linalg.norm(offsets, axis=-1)
    threshold = max(1e-6, float(lengths[parents >= 0].max(initial=0.0)) * 1e-4)
    sizes = _subtree_sizes(order, parents)
    partners = _partners(cond, joint_count)
    rest = np.asarray(cond['rest_pos_ric_hml'], dtype=np.float64)
    frames = rest_bone_frames(rest, parents)
    rotations = np.repeat(np.eye(3)[None], joint_count, axis=0)
    drawn = np.zeros(joint_count, dtype=bool)
    twist_axes = np.asarray(_TWIST_REFERENCE_AXES, dtype=np.float64)
    layouts = {}
    for j in order:
        j = int(j)
        if parents[j] < 0 or j == tri or drawn[j]:
            continue
        if not any(lengths[c] > threshold for c in children[j]):
            continue
        limit = max_rad * (1.0 - sizes[j] / float(joint_count))
        angle = rng.uniform(-limit, limit)
        partner = int(partners[j])
        if partner < 0:
            axes = [np.array([1.0, 0.0, 0.0])]
        else:
            axes = [twist_axes[i] for i in rng.sample(range(len(twist_axes)), len(twist_axes))]
        for k in (j, partner) if partner >= 0 else (j,):
            if k not in layouts:
                layouts[k] = _subtree_layout(children, k)
        for axis in axes:
            rotations[j] = _axis_angle_matrix(axis, angle)
            if partner >= 0:
                rotations[partner] = _MIRROR @ rotations[j] @ _MIRROR
            ok = all(
                _subtree_valid(rotations, parents, offsets, rest, frames, tri, layouts[k])
                for k in ((j, partner) if partner >= 0 else (j,)))
            if ok:
                drawn[j] = True
                if partner >= 0:
                    drawn[partner] = True
                break
            rotations[j] = np.eye(3)
            if partner >= 0:
                rotations[partner] = np.eye(3)
    return rotations, drawn


def _global_rotations(local, parents, order):
    out = local.copy()
    for j in order:
        p = parents[j]
        if p >= 0:
            out[j] = out[p] @ local[j]
    return out


def _reposed_tpose_rotations(tpose_quats, rest_local, parents, order):
    """Bind rotations that keep the exported (FBX-local) animation unchanged.

    The export composes a joint's global bind ``C_j`` with its feature
    rotations; the new features need ``C'_j = G_j C_j``, so each local bind
    becomes ``T'_j = C_p^T Q_j C_p T_j``.
    """
    from motion_lib.Quaternions import Quaternions
    tpose = Quaternions(np.asarray(tpose_quats, dtype=np.float64)).transforms()
    cumulative = _global_rotations(tpose, parents, order)
    new = tpose.copy()
    for j in order:
        p = parents[j]
        if p >= 0:
            new[j] = cumulative[p].T @ rest_local[j] @ cumulative[p] @ tpose[j]
        else:
            new[j] = rest_local[j] @ tpose[j]
    quats = Quaternions.from_transforms(new).qs
    # The same rotation, with the stored quaternion's sign.
    flip = np.sum(quats * np.asarray(tpose_quats, dtype=np.float64), axis=-1) < 0
    quats[flip] *= -1
    return quats.astype(np.asarray(tpose_quats).dtype)


def augment_rest_repose(motion, cond, metadata, max_deg=DEFAULT_REST_REPOSE_MAX_DEG, *, rng=random):
    """Return (physical motion, private cond, diagnostics); a skip is a no-op.

    The physical position channels are returned unchanged; only
    the rotation channel and the rest-derived cond entries change.
    """
    validate_rest_repose_aug(1.0, max_deg)
    info = {'applied': False, 'skip_reason': '', 'rotated_joints': 0, 'max_rest_angle_deg': 0.0}

    def skip(reason):
        info['skip_reason'] = reason
        return motion, cond, info

    if max_deg == 0.0:
        return skip('disabled magnitude')
    parents = np.asarray(cond['parents'], dtype=np.int64)
    order, children = _parent_order(parents)
    tri = int(metadata.get('translation_root_index', cond.get('translation_root_index', 0)))
    if not 0 <= tri < len(parents):
        raise ValueError('Invalid rest re-pose translation root index.')
    local, drawn = _draw_rest_rotations(cond, order, children, tri, np.radians(max_deg), rng)
    if not drawn.any():
        return skip('no joint with a child bone to bend')
    rest_global = _global_rotations(local, parents, order)
    offsets = np.asarray(cond['offsets'], dtype=np.float64)
    old_rest = np.asarray(cond['rest_pos_ric_hml'], dtype=np.float64)
    new_offsets = offsets.copy()
    rest = old_rest.copy()
    for j in order:
        p = parents[j]
        if p >= 0:
            new_offsets[j] = rest_global[p] @ offsets[j]
            rest[j] = rest[p] + new_offsets[j]
    # The rest's lowest joint keeps its height (its grounding convention).
    rest[:, 1] += float(old_rest[:, 1].min() - rest[:, 1].min())
    new_cond = dict(cond)
    new_cond['offsets'] = new_offsets.astype(np.asarray(cond['offsets']).dtype)
    new_cond['rest_pos_ric_hml'] = rest.astype(np.float32)
    for key in ('rest_pose', 'rest_pose_physical'):
        if key in cond:
            new_cond[key] = np.array(cond[key], copy=True)
            new_cond[key][..., :3] = rest
    # Never reuse an L explicitly cached on an input cond.
    new_cond.pop('rest_length_scale', None)
    if cond.get('tpose_rest_rotations') is not None:
        new_cond['tpose_rest_rotations'] = _reposed_tpose_rotations(
            cond['tpose_rest_rotations'], local, parents, order)

    # R'_j = G_p R_j Q_j^T G_p^T; a joint whose parent frame and own rest are
    # both unrotated keeps its stored channel bit for bit.
    parent_global = np.repeat(np.eye(3)[None], len(parents), axis=0)
    has_parent = parents >= 0
    parent_global[has_parent] = rest_global[parents[has_parent]]
    changed = drawn | np.any(parent_global != np.eye(3)[None], axis=(1, 2))
    rot = _rotation_6d_to_matrix(motion[..., changed, 3:9])
    g = parent_global[changed]
    rot = g @ rot @ np.swapaxes(local[changed], -1, -2) @ np.swapaxes(g, -1, -2)
    out = np.array(motion, dtype=np.float32, copy=True)
    out[..., changed, 3:9] = _matrix_to_rotation_6d(rot)
    if not np.isfinite(out).all():
        raise ValueError('Non-finite rest re-pose motion; invalid numerical input.')
    angles = np.degrees(np.arccos(np.clip((np.trace(local, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)))
    info.update(applied=True, rotated_joints=int(drawn.sum()), max_rest_angle_deg=float(angles.max()))
    return out, new_cond, info
