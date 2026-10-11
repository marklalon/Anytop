"""Limb IK for the edit runtime (section 5.2 step 4).

A limb is the chain from its limb root (hip / shoulder) down to its *foot*:
the deepest joint whose subtree holds every contact joint of the limb.  The
foot's subtree moves rigidly; the chain joints above it are solved so the
foot lands on its target, by damped least squares started from the frame's
own pose.  Starting from the pose keeps each knee and elbow bent the way the
frame bends it; a chain at full extension that has to shorten is first bent a
little in its profile flex direction, the only frames where the pose itself
cannot tell.

Joint freedoms come from the profile, as preferences rather than walls: a
hinge turns freely about its first principal axis, a planar joint about the
first two, and every other direction (all three for a fixed joint) is open at
``OFF_AXIS_WEIGHT``.  A hard hinge would leave a chain of near-parallel hinges
unable to move a foot sideways at all.  Joints without a confident profile
are balls.  All axes are in the parent frame, where the profile measured them
(``q = exp(v) * mean``).  Bones keep the lengths of the frame's local
translations: IK only rotates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from motion_edit.rotations import quat_from_rotvec, quat_inv, quat_mul, quat_rotate

# Profiles below this confidence constrain nothing: the joint is solved as a ball.
MIN_DOF_CONFIDENCE = 0.3
# Weight of the directions a joint's profile does not move it in.
OFF_AXIS_WEIGHT = 0.1
FIXED_WEIGHT = 0.05
ITERATIONS = 100
DAMPING = 0.002           # x limb length
MAX_STEP = 0.2            # rad per joint and iteration
TOLERANCE = 1e-7          # x limb length; stop when every frame is this close
STRAIGHT_RATIO = 0.995    # reach / chain length above which a chain counts as straight
PRE_BEND = 0.1            # rad, per hinge, for a straight chain that must shorten


@dataclass
class Limb:
    root: int                 # limb root (first chain joint)
    foot: int                 # end effector; its subtree moves rigidly
    chain: list[int]          # limb root .. parent(foot), root first
    contacts: list[int]       # the limb's contact joints (package contact indices are columns)
    columns: list[int]        # their columns in the package contact arrays
    depth: dict               # contact joint -> depth below the foot


def _ancestors(j: int, parents) -> list[int]:
    out = [j]
    while parents[j] >= 0:
        j = int(parents[j])
        out.append(j)
    return out


def build_limbs(parents, sides, contact_joints, translation_root: int,
                parts=None) -> tuple[list[Limb], list[str]]:
    """Contact joints grouped into solvable limbs, and notes on the ones that are not.

    A sided contact joint's limb starts at the top of its run of same-side joints;
    with ``parts`` (one per joint) the run also stops where the limb's kind does
    (``parts.LIMB_KIND``: a foot belongs to its leg, not to a sided hip pad above it).
    """
    from motion_edit.profile.parts import LIMB_KIND

    parents = np.asarray(parents)
    notes: list[str] = []
    groups: dict[int, list[int]] = {}

    def same_limb(j: int, parent: int) -> bool:
        if sides[parent] != sides[j]:
            return False
        return parts is None or (LIMB_KIND.get(parts[j]) is not None
                                 and LIMB_KIND.get(parts[parent]) == LIMB_KIND.get(parts[j]))

    for column, joint in enumerate(contact_joints):
        root = int(joint)
        if sides[root] != "center":
            while parents[root] >= 0 and same_limb(root, int(parents[root])):
                root = int(parents[root])
        groups.setdefault(root, []).append(column)
    limbs = []
    for root, columns in sorted(groups.items()):
        joints = [int(contact_joints[c]) for c in columns]
        paths = [_ancestors(j, parents)[::-1] for j in joints]   # from the skeleton root down
        common = paths[0]
        for path in paths[1:]:
            n = 0
            while n < min(len(common), len(path)) and common[n] == path[n]:
                n += 1
            common = common[:n]
        foot = common[-1]
        if root not in common or foot == root:
            notes.append(f"contact joint(s) {joints}: no chain between limb root and foot; not solved by IK")
            continue
        chain = common[common.index(root):-1]
        if translation_root in chain:
            notes.append(f"contact joint(s) {joints}: chain passes the translation root; not solved by IK")
            continue
        depth = {j: len(p) - len(common) for j, p in zip(joints, paths)}
        limbs.append(Limb(root=root, foot=foot, chain=chain, contacts=joints, columns=list(columns), depth=depth))
    return limbs, notes


def joint_axes(j: int, dof: np.ndarray | None, axes: np.ndarray | None,
               confidence: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Parent-frame rotation axes (3, 3) of joint ``j`` and the weight of each."""
    if dof is None or confidence is None or confidence[j] < MIN_DOF_CONFIDENCE:
        return np.eye(3), np.ones(3)
    basis = np.asarray(axes[j], dtype=np.float64)
    kind = str(dof[j])
    if kind == "hinge":
        return basis, np.array([1.0, OFF_AXIS_WEIGHT, OFF_AXIS_WEIGHT])
    if kind == "planar":
        return basis, np.array([1.0, 1.0, OFF_AXIS_WEIGHT])
    if kind == "fixed":
        return basis, np.full(3, FIXED_WEIGHT)
    return np.eye(3), np.ones(3)


class LimbSolver:
    """Solves one limb over all frames at once."""

    def __init__(self, limb: Limb, parents, dof=None, axes=None, confidence=None, flex_sign=None):
        self.limb = limb
        self.parents = np.asarray(parents)
        freedoms = [joint_axes(j, dof, axes, confidence) for j in limb.chain]
        self.axes = [a for a, _ in freedoms]
        self.weights = [w for _, w in freedoms]
        self.flex = []
        for j in limb.chain:
            sign = int(flex_sign[j]) if flex_sign is not None else 0
            hinge = dof is not None and str(dof[j]) == "hinge" and confidence[j] >= MIN_DOF_CONFIDENCE
            self.flex.append(np.asarray(axes[j][0]) * sign if (hinge and sign != 0) else None)

    def _fk(self, parent_rot, parent_pos, local_rot, local_pos):
        """Chain FK: global rotations / positions of the chain joints and the foot."""
        n = len(self.limb.chain)
        rots, poss = [], []
        g, p = parent_rot, parent_pos
        for i in range(n):
            p = p + quat_rotate(g, local_pos[:, i])
            g = quat_mul(g, local_rot[:, i])
            rots.append(g)
            poss.append(p)
        foot = p + quat_rotate(g, local_pos[:, n])
        return rots, poss, foot

    def solve(self, parent_rot, parent_pos, local_rot, local_pos, target, scale):
        """``local_rot`` (F, n, 4) chain rotations, ``local_pos`` (F, n + 1, 3) chain + foot
        translations; returns solved rotations and the final foot error (F,) from the
        given target (a target out of reach is solved as its nearest reachable point)."""
        local_rot = np.array(local_rot, dtype=np.float64, copy=True)
        given = target
        n = len(self.limb.chain)
        frames = local_rot.shape[0]
        lam2 = (DAMPING * scale) ** 2
        tol = TOLERANCE * scale

        lengths = np.linalg.norm(local_pos[:, 1:], axis=-1).sum(axis=1)
        _, poss, foot = self._fk(parent_rot, parent_pos, local_rot, local_pos)
        # a target out of reach is moved onto the reach sphere, along the line from the
        # limb root: DLS aimed past full extension never settles, it swings the chain
        # about, and the iteration it stops on decides the foot
        offset = target - poss[0]
        distance = np.linalg.norm(offset, axis=-1)
        cap = STRAIGHT_RATIO * lengths
        far = distance > cap
        target = np.where(far[:, None], poss[0] + offset * (cap / np.maximum(distance, 1e-12))[:, None], target)

        # straight chain that must shorten: pre-bend its hinges in the flex direction
        reach = np.linalg.norm(foot - poss[0], axis=-1)
        need = np.linalg.norm(target - poss[0], axis=-1)
        bend = (reach > STRAIGHT_RATIO * lengths) & (need < reach - tol)
        if bend.any():
            for i, axis in enumerate(self.flex):
                if axis is not None:
                    turn = quat_from_rotvec(np.where(bend[:, None], axis * PRE_BEND, 0.0))
                    local_rot[:, i] = quat_mul(turn, local_rot[:, i])

        active = np.ones(frames, dtype=bool)
        for _ in range(ITERATIONS):
            rots, poss, foot = self._fk(parent_rot, parent_pos, local_rot, local_pos)
            err = target - foot
            active &= np.linalg.norm(err, axis=-1) > tol
            if not active.any():
                break
            columns = []
            for i in range(n):
                g_parent = parent_rot if i == 0 else rots[i - 1]
                for axis, weight in zip(self.axes[i], self.weights[i]):
                    world = quat_rotate(g_parent, np.broadcast_to(axis, (frames, 3)))
                    columns.append(weight * np.cross(world, foot - poss[i]))
            jac = np.stack(columns, axis=-1)                       # (F, 3, 3n), weighted
            jjt = jac @ np.swapaxes(jac, 1, 2) + lam2 * np.eye(3)
            step = np.swapaxes(jac, 1, 2) @ np.linalg.solve(jjt, err[..., None])
            step = step[..., 0] * active[:, None]                  # (F, 3n)
            for i in range(n):
                rotvec = (step[:, 3 * i:3 * i + 3] * self.weights[i]) @ self.axes[i]
                angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
                rotvec = rotvec * np.minimum(1.0, MAX_STEP / np.maximum(angle, 1e-12))
                local_rot[:, i] = quat_mul(quat_from_rotvec(rotvec), local_rot[:, i])
        _, _, foot = self._fk(parent_rot, parent_pos, local_rot, local_pos)
        return local_rot, np.linalg.norm(given - foot, axis=-1)


def world_to_local(global_rot_parent: np.ndarray, global_rot: np.ndarray) -> np.ndarray:
    return quat_mul(quat_inv(global_rot_parent), global_rot)
