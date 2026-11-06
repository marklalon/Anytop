"""Remove one complete peripheral chain, together with its mirror if present."""

from __future__ import annotations

import random

import numpy as np

from data_loaders.truebones.truebones_utils.bone_length_aug import _parent_order
from data_loaders.truebones.truebones_utils.joint_parts import joint_part_labels
from data_loaders.truebones.truebones_utils.leaf_drop import LEAF_DROP_MIN_KEPT_JOINTS


_FAMILY = {'arm': 'arm', 'hand': 'arm', 'leg': 'leg', 'foot': 'leg',
           'wing': 'wing', 'tail': 'tail', 'soft': 'soft', 'helper': 'helper'}
_ATTACHMENTS = {'soft', 'helper'}


def limb_drop_candidates(entry, protected) -> list[tuple[int, ...]]:
    """Legal deletion units, each one whole subtree and at most its mirror.

    Legs are counted individually before mirror grouping. Six or more legs
    permit leg removal independently of the wing/arm/leg coexistence rule.
    A tail is eligible only when the rig has an arm or leg part.
    With side labels, a shared midline connector does not join left and right
    limbs into one chain; without them the family chain is all that separates
    the sides, and a wing pair hanging off one body joint shares that joint.
    """
    parents = np.asarray(entry['parents'], dtype=np.int64)
    count = len(parents)
    if count <= LEAF_DROP_MIN_KEPT_JOINTS:
        return []
    parts = joint_part_labels(entry)
    protected = set(protected)
    family = [_FAMILY.get(part) for part in parts]
    order, children = _parent_order(parents)
    sides = entry.get('joint_side_labels')
    # A centre-labelled connector carrying sided limbs belongs to the body.
    sided = [False] * count
    for j in reversed(order):
        sided[j] = (sides is not None and sides[j] in ('left', 'right')) or any(
            sided[c] for c in children[j] if family[c] == family[j])
    connector = {
        j for j in order if sides is not None and sides[j] == 'center'
        and family[j] in ('arm', 'leg', 'wing') and sided[j]
    }
    root_of = {}
    chains = {}
    for j in order:
        if family[j] is None or j in connector:
            continue
        p = int(parents[j])
        same = p in root_of and family[p] == family[j]
        if sides is not None and p >= 0 and {sides[p], sides[j]} == {'left', 'right'}:
            same = False
        root = root_of[p] if same else j
        root_of[j] = root
        chains.setdefault(root, []).append(j)

    leg_count = sum(family[root] == 'leg' for root in chains)
    present = {family[root] for root in chains}
    allowed = {'soft', 'helper'}
    if {'arm', 'leg'} & set(parts):
        allowed.add('tail')
    if leg_count >= 6:
        allowed.add('leg')
    if {'wing', 'arm', 'leg'} <= present:
        allowed.update(('wing', 'arm'))

    subtrees = {}
    for root in chains:
        if family[root] not in allowed or parents[root] < 0:
            continue
        members = [root]
        for j in members:
            members.extend(children[j])
        removed = set(members)
        # No candidate may swallow another anatomical limb or body region.
        if removed & protected or any(
            family[j] != family[root] and parts[j] not in _ATTACHMENTS for j in removed
        ) or removed & connector:
            continue
        subtrees[root] = removed

    partners = entry.get('symmetry_partner_indices')
    if partners is None or len(partners) != count:
        partners = [-1] * count
    mirrors = [set() for _ in range(count)]
    for j, partner in enumerate(partners):
        partner = int(partner)
        if 0 <= partner < count:
            mirrors[j].add(partner)
            mirrors[partner].add(j)
    units = set()
    for root, removed in subtrees.items():
        # Use every core joint: even a chain root without a mirror annotation
        # can be paired by an annotated distal joint.
        mirror_roots = set()
        valid = True
        for j in chains[root]:
            if int(partners[j]) >= count:
                valid = False
                break
            for partner in mirrors[j] - removed:
                other = root_of.get(partner)
                if other not in subtrees or family[other] != family[root]:
                    valid = False
                    break
                mirror_roots.add(other)
        if not valid or len(mirror_roots) > 1:
            continue
        unit = set(removed)
        for other in mirror_roots:
            unit.update(subtrees[other])
        # Attachment mirrors must also stay inside the one-chain/pair unit.
        if any(int(partners[j]) >= count or mirrors[j] - unit for j in unit):
            continue
        if count - len(unit) < LEAF_DROP_MIN_KEPT_JOINTS:
            continue
        units.add(tuple(sorted(unit)))
    return sorted(units)


def sample_limb_drop(entry, protected, rng=random) -> list[int]:
    """Choose one legal chain/mirror unit; no partial cuts or repeated draws."""
    units = limb_drop_candidates(entry, protected)
    return list(rng.choice(units)) if units else []
