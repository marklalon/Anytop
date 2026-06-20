"""Pure skeleton-cropping helpers shared by preprocessing and validation."""

from __future__ import annotations

import numpy as np


def _joint_depths_from_parents(parents: np.ndarray) -> np.ndarray:
    """Return per-joint depth (root = 0) for a parent array."""
    parents = np.asarray(parents, dtype=np.int64)
    depths = np.zeros(parents.shape[0], dtype=np.int64)
    for joint_index in range(parents.shape[0]):
        depth = 0
        cursor = joint_index
        while int(parents[cursor]) >= 0:
            cursor = int(parents[cursor])
            depth += 1
        depths[joint_index] = depth
    return depths


def select_cropped_joint_indices(parents, max_joints, offsets=None):
    """Pick the joints to keep so a skeleton fits within ``max_joints``.

    Joints are removed one at a time, always taking the deepest current leaf.
    Same-depth ties prefer shorter bones, while longer-than-average bones are
    preserved whenever possible. Roots are never removed.

    Returns ``(keep_indices, removed_order)`` or ``None`` when no cropping is
    needed.
    """
    parents = np.asarray(parents, dtype=np.int64)
    n = int(parents.shape[0])
    if n <= int(max_joints):
        return None

    if offsets is not None:
        offsets = np.asarray(offsets, dtype=np.float64)
        if offsets.shape[0] != n:
            raise ValueError(
                f"Expected {n} joint offsets to crop skeleton, "
                f"got {offsets.shape[0]}"
            )
        bone_lengths = np.linalg.norm(offsets, axis=-1)
        nonroot_lengths = bone_lengths[parents >= 0]
        mean_bone_length = (
            float(nonroot_lengths.mean()) if nonroot_lengths.size else 0.0
        )
        protected = (parents >= 0) & (bone_lengths > mean_bone_length)
    else:
        bone_lengths = np.zeros((n,), dtype=np.float64)
        protected = np.zeros((n,), dtype=bool)

    depths = _joint_depths_from_parents(parents)
    kept = [True] * n
    child_count = [0] * n
    for joint_index in range(n):
        parent_index = int(parents[joint_index])
        if parent_index >= 0:
            child_count[parent_index] += 1

    num_kept = n
    removed_order = []
    while num_kept > int(max_joints):
        best = -1
        best_depth = -1
        best_length = float("inf")
        protected_best = -1
        protected_best_depth = -1
        protected_best_length = float("inf")
        for joint_index in range(n):
            if not kept[joint_index] or child_count[joint_index] != 0:
                continue
            if int(parents[joint_index]) < 0:
                continue
            depth = int(depths[joint_index])
            bone_length = float(bone_lengths[joint_index])
            is_better = (
                depth > best_depth
                or (
                    depth == best_depth
                    and (
                        bone_length < best_length
                        or (
                            np.isclose(bone_length, best_length)
                            and joint_index > best
                        )
                    )
                )
            )
            if protected[joint_index]:
                protected_is_better = (
                    depth > protected_best_depth
                    or (
                        depth == protected_best_depth
                        and (
                            bone_length < protected_best_length
                            or (
                                np.isclose(bone_length, protected_best_length)
                                and joint_index > protected_best
                            )
                        )
                    )
                )
                if protected_is_better:
                    protected_best_depth = depth
                    protected_best_length = bone_length
                    protected_best = joint_index
                continue
            if is_better:
                best_depth = depth
                best_length = bone_length
                best = joint_index
        if best < 0:
            best = protected_best
        if best < 0:
            break
        kept[best] = False
        removed_order.append(best)
        num_kept -= 1
        parent_index = int(parents[best])
        if parent_index >= 0:
            child_count[parent_index] -= 1

    keep_indices = [joint_index for joint_index in range(n) if kept[joint_index]]
    return keep_indices, removed_order
