"""Shared utilities for subtree joint mask sampling.

Core logic used by both:
  - ``model/anytop.py`` (PyTorch, batched)
  - ``tools/sample_augmented_bvh.py`` (NumPy, single-sample)
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Subtree collection
# ---------------------------------------------------------------------------

def collect_subtree_indices(root_index: int, children: list[list[int]]) -> list[int]:
    """DFS gather of all descendants of *root_index* (inclusive)."""
    stack = [int(root_index)]
    subtree = []
    while stack:
        idx = stack.pop()
        subtree.append(idx)
        stack.extend(children[idx])
    return subtree


# ---------------------------------------------------------------------------
# Single-skeleton subtree mask sampler (NumPy)
# ---------------------------------------------------------------------------

def sample_subtree_joint_mask(
    parents: list[int],
    candidate_root_mask: np.ndarray,
    joint_mask_prob: float,
    rng: Any,
) -> Optional[np.ndarray]:
    """Replicate AnyTop._sample_subtree_joint_mask for a single skeleton.

    Parameters
    ----------
    parents : list[int]
        Parent indices, length = J.
    candidate_root_mask : np.ndarray
        Boolean array of shape ``(J,)`` — ``True`` where a joint is a valid
        subtree root.
    joint_mask_prob : float
        Fraction of non-root joints to mask (budget).  Must be in ``[0, 1]``.
    rng : object
        NumPy-compatible RNG object exposing ``choice`` and ``permutation``.
        This can be a ``np.random.Generator`` or the global ``np.random``
        module so training can participate in checkpointed NumPy RNG state.

    Returns
    -------
    np.ndarray or None
        Boolean mask of shape ``(J,)`` — ``True`` = joint is masked, or
        ``None`` if no masking occurred.
    """
    n_joints = len(parents)
    non_root_count = max(n_joints - 1, 0)
    budget = min(int(joint_mask_prob * non_root_count), non_root_count)
    if budget <= 0 or n_joints <= 1:
        return None

    # Build children lookup
    children = [[] for _ in range(n_joints)]
    for child_idx in range(1, n_joints):
        p = int(parents[child_idx])
        if 0 <= p < n_joints:
            children[p].append(child_idx)

    # Root (index 0) is never a candidate
    root_mask = candidate_root_mask.copy()
    root_mask[0] = False
    candidate_root_indices = np.flatnonzero(root_mask)

    # Collect all candidate subtrees that fit within the budget
    candidate_subtrees = []
    for root_idx in candidate_root_indices:
        subtree = collect_subtree_indices(int(root_idx), children)
        if 0 < len(subtree) <= budget:
            candidate_subtrees.append(subtree)

    if not candidate_subtrees:
        return None

    # Prefer larger connected regions so subtree masking more often removes
    # a coherent limb/body part instead of accumulating many tiny subtrees.
    mask = np.zeros(n_joints, dtype=bool)
    remaining = budget
    subtree_sizes = np.asarray([len(subtree) for subtree in candidate_subtrees], dtype=np.float64)
    available = np.ones(len(candidate_subtrees), dtype=bool)
    while remaining > 0:
        compatible_positions = []
        compatible_weights = []
        for pos, subtree in enumerate(candidate_subtrees):
            if not available[pos]:
                continue
            sz = int(subtree_sizes[pos])
            if sz > remaining:
                continue
            if np.any(mask[subtree]):
                continue
            compatible_positions.append(pos)
            compatible_weights.append(float(sz * sz))

        if not compatible_positions:
            break

        weights = np.asarray(compatible_weights, dtype=np.float64)
        weights /= weights.sum()
        chosen_pos = int(rng.choice(np.asarray(compatible_positions, dtype=np.int64), p=weights))
        subtree = candidate_subtrees[chosen_pos]
        sz = len(subtree)
        mask[subtree] = True
        remaining -= sz
        available[chosen_pos] = False
        if remaining == 0:
            break

    if not np.any(mask):
        return None
    return mask
