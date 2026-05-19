"""Shared utilities for subtree joint mask sampling.

Core logic used by both:
  - ``model/anytop.py`` (PyTorch, batched)
  - ``tools/sample_augmented_bvh.py`` (NumPy, single-sample)
"""

from __future__ import annotations

from typing import Optional

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
    rng: np.random.Generator,
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
    rng : np.random.Generator
        NumPy RNG for reproducible random selection.

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

    # Greedy random selection of non-overlapping subtrees
    mask = np.zeros(n_joints, dtype=bool)
    remaining = budget
    order = rng.permutation(len(candidate_subtrees))
    for pos in order:
        subtree = candidate_subtrees[pos]
        sz = len(subtree)
        if sz > remaining:
            continue
        # Check overlap
        if np.any(mask[subtree]):
            continue
        mask[subtree] = True
        remaining -= sz
        if remaining == 0:
            break

    if not np.any(mask):
        return None
    return mask
