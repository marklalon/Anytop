"""Symmetric body-proportion augmentation in physical feature space.

One or two topology groups -- branch-free bone chains, mirror chains merged --
get one constant multiplier per clip. Only long groups are drawn (mean bone
length at least the skeleton's mean bone length), and each multiplier changes
the length by a relative amount in [min, max], growing or shrinking. Topology,
names and subset statistics stay fixed. Local translations (including authored
deformation) are scaled, not discarded. All action labels are accepted. Contact IK is disabled: local
rotations are preserved, and the lowest point of the rest pose and of the clip
keeps its height.
"""
from __future__ import annotations

import random

import numpy as np

from data_loaders.truebones.truebones_utils.canonical_features import root_relative_positions

# Largest relative length change of a group when a caller does not choose one.
DEFAULT_BONE_LENGTH_AUG = 0.2
# Smallest relative length change, so a draw is always visible.
DEFAULT_BONE_LENGTH_AUG_MIN = 0.1


def _parent_order(parents):
    """Accept arbitrary joint order; malformed graphs are input errors."""
    count = len(parents)
    children = [[] for _ in parents]
    roots = []
    for j, p in enumerate(parents):
        if p < 0:
            roots.append(j)
        elif p >= count or p == j:
            raise ValueError('Invalid bone-length augmentation parent index.')
        else:
            children[p].append(j)
    order = list(roots)
    cursor = 0
    while cursor < len(order):
        order.extend(children[order[cursor]])
        cursor += 1
    if not count or len(order) != count:
        raise ValueError('Bone-length augmentation requires a nonempty acyclic skeleton.')
    if all(p < j for j, p in enumerate(parents)):
        order = list(range(count))
    return np.asarray(order), children


def _scale_positions(source, parents, order, scales):
    """Scale bone vectors directly; fixed rotations make local-space FK redundant."""
    target = source.copy()
    for j in order:
        p = parents[j]
        if p >= 0:
            target[..., j, :] = target[..., p, :] + (source[..., j, :] - source[..., p, :]) * scales[j]
    return target


def _topology_groups(cond, order, children, tri):
    """Branch-free bone chains, with annotated mirror chains grouped together.
    Zero offsets and the translation carrier are never anatomy to stretch."""
    parents = np.asarray(cond['parents'])
    lengths = np.linalg.norm(np.asarray(cond['offsets']), axis=-1)
    threshold = max(1e-6, float(lengths[parents >= 0].max(initial=0.0)) * 1e-4)
    eligible = {int(j) for j in order if parents[j] >= 0 and j != tri and lengths[j] > threshold}
    if not eligible:
        eligible = {int(j) for j in order if parents[j] >= 0 and j != tri and lengths[j] > 0.0}
    chains = []
    seen = set()
    for j in order:
        if j not in eligible or j in seen:
            continue
        chain = [int(j)]
        cursor = int(j)
        while len(children[cursor]) == 1:
            cursor = children[cursor][0]
            if cursor in eligible:
                chain.append(cursor)
        seen.update(chain)
        chains.append(chain)
    owners = {j: i for i, chain in enumerate(chains) for j in chain}
    union = list(range(len(chains)))
    def find(i):
        while union[i] != i:
            union[i] = union[union[i]]
            i = union[i]
        return i
    partners = cond.get('symmetry_partner_indices', [])
    if len(partners) == len(parents):
        for j, i in owners.items():
            other = owners.get(int(partners[j]))
            if other is not None:
                union[find(other)] = find(i)
    merged = {}
    for i, chain in enumerate(chains):
        merged.setdefault(find(i), []).extend(chain)
    return {f'chain_{min(joints)}': sorted(joints) for joints in merged.values()}


def validate_bone_length_aug(probability, magnitude, min_magnitude=0.0):
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError('bone_length_aug_prob must be finite and in [0, 1].')
    if not np.isfinite(magnitude) or not 0.0 <= magnitude < 1.0:
        raise ValueError('bone_length_aug must be finite and in [0, 1).')
    if not np.isfinite(min_magnitude) or not 0.0 <= min_magnitude <= magnitude:
        raise ValueError('bone_length_aug_min must be finite and in [0, bone_length_aug].')


def long_groups(cond, groups):
    """Groups whose mean bone length reaches the mean over all grouped bones.
    Short distal chains (fingers, toes) barely show a length change."""
    lengths = np.linalg.norm(np.asarray(cond['offsets'], dtype=np.float64), axis=-1)
    means = {name: float(lengths[joints].mean()) for name, joints in groups.items()}
    overall = float(lengths[sorted({j for joints in groups.values() for j in joints})].mean())
    # Relative tolerance: a uniform-length rig keeps every group.
    return {name: groups[name] for name in sorted(groups) if means[name] >= overall * (1.0 - 1e-6)}


def body_groups(cond):
    """The topology groups of ``cond``'s skeleton (``_topology_groups``)."""
    parents = np.asarray(cond['parents'], dtype=np.int64)
    order, children = _parent_order(parents)
    return _topology_groups(cond, order, children, int(cond.get('translation_root_index', 0)))


def augment_bone_lengths(motion, cond, metadata, magnitude=DEFAULT_BONE_LENGTH_AUG,
                         min_magnitude=DEFAULT_BONE_LENGTH_AUG_MIN, *, rng=random):
    """Return (physical motion, private cond, diagnostics); skip is a no-op.

    ``motion`` is in the working layout (canonical_features, "Root"), whose
    translation root is joint 0. Positions use the existing identity-facing HML
    basis. No temporal operations run here. Closed clips retain their closing
    key.
    """
    validate_bone_length_aug(1.0, magnitude, min_magnitude)
    info = {'applied': False, 'scales': {}, 'skip_reason': '', 'notes': [],
            'contact_ik_applied': False}

    def skip(reason):
        info['skip_reason'] = reason
        return motion, cond, info

    if magnitude == 0.0:
        return skip('disabled magnitude')
    parents = np.asarray(cond['parents'], dtype=np.int64)
    order, children = _parent_order(parents)
    tri = int(metadata.get('translation_root_index', cond.get('translation_root_index', 0)))
    if tri != 0:
        raise ValueError('Bone-length augmentation needs joint 0 as the translation root (the working layout).')
    groups = _topology_groups(cond, order, children, tri)
    if not groups:
        # There is no length to scale on a zero-offset/one-joint rig. Export it
        # too, but report that it is unchanged instead of inventing a bone.
        info['applied'] = True
        info['changed_bone_count'] = 0
        info['notes'].append('no nonzero bone offsets; skeleton unchanged')
        return np.array(motion, copy=True), dict(cond), info
    candidates = long_groups(cond, groups)
    selected = rng.sample(sorted(candidates), rng.randint(1, min(2, len(candidates))))
    scales = np.ones(len(parents), dtype=np.float64)
    for group in selected:
        change = rng.uniform(min_magnitude, magnitude)
        factor = 1.0 + change if rng.random() < 0.5 else 1.0 - change
        scales[groups[group]] = factor
        info['scales'][group] = factor
    info['changed_bone_count'] = int(np.count_nonzero(scales != 1.0))
    old_offsets = np.asarray(cond['offsets'], dtype=np.float64)
    offsets = old_offsets * scales[:, None]
    old_rest = np.asarray(cond['rest_pos_ric_hml'], dtype=np.float64)
    rest = _scale_positions(old_rest, parents, order, scales)
    # The rest's lowest joint keeps its height (its grounding convention).
    rest[:, 1] += float(old_rest[:, 1].min() - rest[:, 1].min())
    new_cond = dict(cond)
    new_cond['offsets'] = offsets.astype(np.float32)
    new_cond['rest_pos_ric_hml'] = rest.astype(np.float32)
    for key in ('rest_pose', 'rest_pose_physical'):
        if key in cond:
            new_cond[key] = np.array(cond[key], copy=True)
            new_cond[key][..., :3] = rest
    # Never reuse an L explicitly cached on an input cond.
    new_cond.pop('rest_length_scale', None)

    # Root XZ travel cancels in each bone vector. Work in RIC space directly
    # (the all-RIC view of the working layout), without world recovery,
    # 6D->quaternion conversion, or a second FK pass.
    source = np.asarray(root_relative_positions(motion[..., :3]), dtype=np.float64)
    target = _scale_positions(source, parents, order, scales)
    # So does the clip's lowest point: a foot that reached the floor still does.
    target[..., 1] += float(source[..., 1].min() - target[..., 1].min())

    out = np.array(motion, dtype=np.float32, copy=True)
    out[..., :3] = target
    out[..., 0] -= target[:, tri:tri+1, 0]
    out[..., 2] -= target[:, tri:tri+1, 2]
    # The root keeps its world XZ: bone lengths do not move the trajectory.
    out[:, tri, 0] = motion[:, tri, 0]
    out[:, tri, 2] = motion[:, tri, 2]
    # No contact IK: retain the source local-rotation channels exactly.
    # The copy above already carries motion[..., 3:9].
    if not np.isfinite(out).all():
        raise ValueError('Non-finite bone-length reconstructed motion; invalid numerical input.')
    info['applied'] = True
    return out, new_cond, info
