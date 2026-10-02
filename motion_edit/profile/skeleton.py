"""Static skeleton structure: contacts, limbs, lengths, roles."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CENTER = "center"
# A sided limb shorter than this fraction of the leg (ears, mouth corners,
# whiskers) carries no locomotion or gesture meaning: role "other".
SMALL_LIMB_RATIO = 0.25


def children_of(parents) -> list[list[int]]:
    children = [[] for _ in parents]
    for j, p in enumerate(parents):
        if p >= 0:
            children[int(p)].append(j)
    return children


def rest_positions(parents, offsets) -> np.ndarray:
    """Rest-pose global positions (identity rest rotations, the cond convention)."""
    offsets = np.asarray(offsets, dtype=np.float64)
    out = np.zeros_like(offsets)
    for j, p in enumerate(parents):
        out[j] = offsets[j] + (out[p] if p >= 0 else 0.0)
    return out


@dataclass
class ContactSet:
    cond: list[int]
    added: list[int] = field(default_factory=list)
    removed: list[int] = field(default_factory=list)
    unknown_names: list[str] = field(default_factory=list)

    @property
    def used(self) -> list[int]:
        return sorted((set(self.cond) | set(self.added)) - set(self.removed))


def resolve_contacts(cond_entry: dict, override_entries: dict | None) -> ContactSet:
    """cond's ``contact_joints`` with a ``{"add": [...], "remove": [...]}`` name override."""
    names = [str(n) for n in cond_entry["joints_names"]]
    index_of = {n: i for i, n in enumerate(names)}
    contacts = ContactSet(cond=sorted(int(j) for j in cond_entry.get("contact_joints", [])))
    for key, target in (("add", contacts.added), ("remove", contacts.removed)):
        for name in (override_entries or {}).get(key, []):
            if name in index_of:
                target.append(index_of[name])
            else:
                contacts.unknown_names.append(str(name))
    return contacts


class SkeletonStructure:
    """Limbs and lengths of one rig, from cond alone."""

    def __init__(self, cond_entry: dict, contacts: list[int]):
        self.names = [str(n) for n in cond_entry["joints_names"]]
        self.parents = np.asarray(cond_entry["parents"], dtype=np.int64)
        self.offsets = np.asarray(cond_entry["offsets"], dtype=np.float64)
        self.joint_count = len(self.parents)
        self.children = children_of(self.parents)
        sides = cond_entry.get("joint_side_labels")
        self.sides = [str(s) for s in sides] if sides is not None else [CENTER] * self.joint_count
        self.root = int(cond_entry.get("translation_root_index", 0) or 0)
        self.contacts = list(contacts)
        self.rest = rest_positions(self.parents, self.offsets)
        self.bone_length = np.linalg.norm(self.offsets, axis=-1)
        self._axial_fallback = float(cond_entry.get("axial_avg_len", 0.0)) * float(cond_entry.get("scale_factor", 1.0))

    # ── limbs ────────────────────────────────────────────────────────────
    def limb_root(self, j: int) -> int:
        """Topmost ancestor of ``j`` on the same side; a center joint is its own limb root."""
        side = self.sides[j]
        if side == CENTER:
            return j
        while True:
            p = int(self.parents[j])
            if p < 0 or self.sides[p] != side:
                return j
            j = p

    def path_to(self, j: int, ancestor: int) -> list[int]:
        """``j`` up to and including ``ancestor``."""
        path = [j]
        while j != ancestor:
            j = int(self.parents[j])
            if j < 0:
                raise ValueError(f"{ancestor} is not an ancestor")
            path.append(j)
        return path

    def subtree(self, j: int) -> list[int]:
        out, stack = [], [j]
        while stack:
            k = stack.pop()
            out.append(k)
            stack.extend(self.children[k])
        return out

    def support_joints(self) -> set[int]:
        support = set()
        for c in self.contacts:
            support.update(self.path_to(c, self.limb_root(c)))
        return support

    def leaf_chains(self) -> list[list[int]]:
        """Each leaf's chain up to (not including) its nearest branching ancestor, root first."""
        chains = []
        for leaf in range(self.joint_count):
            if self.children[leaf]:
                continue
            chain = [leaf]
            j = leaf
            while True:
                p = int(self.parents[j])
                if p < 0 or len(self.children[p]) != 1 or p == self.root:
                    break
                chain.append(p)
                j = p
            chains.append(chain[::-1])
        return chains

    # ── lengths ──────────────────────────────────────────────────────────
    def contact_limbs(self) -> dict[int, list[int]]:
        """Contact joints grouped by limb root (hip / shoulder), each list lowest-at-rest first.

        cond's contact set holds whole foot chains (ankle, toe, toe tip), so a
        limb is the unit a leg length, a plant or a gait phase is measured on.
        """
        limbs: dict[int, list[int]] = {}
        for c in self.contacts:
            limbs.setdefault(self.limb_root(c), []).append(c)
        return {root: sorted(joints, key=lambda j: (self.rest[j, 1], j))
                for root, joints in sorted(limbs.items())}

    def limb_foot(self, limb_root: int) -> int:
        """The limb's lowest contact joint at rest."""
        return self.contact_limbs()[limb_root][0]

    def leg_lengths(self) -> dict[int, float]:
        """Per contact limb root: rest length from the limb root down to its lowest contact."""
        out = {}
        for root, joints in self.contact_limbs().items():
            path = self.path_to(joints[0], root)[:-1]   # bones below the hip
            out[root] = float(self.bone_length[path].sum())
        return out

    @property
    def leg_length(self) -> float:
        lengths = [v for v in self.leg_lengths().values() if v > 1e-6]
        if lengths:
            return float(np.mean(lengths))
        return self._axial_fallback

    @property
    def has_legs(self) -> bool:
        return any(v > 1e-6 for v in self.leg_lengths().values())

    @property
    def hip_height(self) -> float | None:
        hips = [root for root, joints in self.contact_limbs().items() if root not in joints]
        if not hips:
            return None
        ground = min(self.rest[c, 1] for c in self.contacts)
        return float(np.mean([self.rest[h, 1] for h in hips]) - ground)

    @property
    def axial_length(self) -> float:
        center = [j for j in range(self.joint_count)
                  if self.sides[j] == CENTER and self.parents[j] >= 0]
        return float(self.bone_length[center].sum())

    def limb_length(self, limb_root: int) -> float:
        best = 0.0
        for j in self.subtree(limb_root):
            if not self.children[j]:
                path = self.path_to(j, limb_root)[:-1]
                best = max(best, float(self.bone_length[path].sum()))
        return best

    # ── roles ────────────────────────────────────────────────────────────
    def base_roles(self) -> list[str]:
        """Roles before the passive fit: root / support / axial / swing / other."""
        support = self.support_joints()
        leg = self.leg_length
        roles = []
        for j in range(self.joint_count):
            if j == self.root:
                roles.append("root")
            elif j in support:
                roles.append("support")
            elif self.sides[j] == CENTER:
                roles.append("axial")
            elif self.limb_length(self.limb_root(j)) < SMALL_LIMB_RATIO * leg:
                roles.append("other")
            else:
                roles.append("swing")
        return roles

    def passive_chains(self) -> list[tuple[list[int], list[int]]]:
        """``(leaf chain, its secondary-motion candidates)`` per leaf chain.

        Candidates are the chain's joints off the support chains that have a
        bone to swing; chains without one are left out.
        """
        support = self.support_joints()
        out = []
        for chain in self.leaf_chains():
            candidates = [j for j in chain
                          if j not in support and j != self.root and self.children[j]
                          and self.bone_length[self.children[j]].max() > 1e-6]
            if candidates:
                out.append((chain, candidates))
        return out

    def passive_candidates(self) -> list[int]:
        return sorted(j for _, candidates in self.passive_chains() for j in candidates)

    def swing_length(self, chain: list[int], j: int) -> float:
        """Rest length of the chain hanging below ``j`` (``j``'s bone to the tip)."""
        below = chain[chain.index(j) + 1:]
        return float(self.bone_length[below].sum())
