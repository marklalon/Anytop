"""Shared utilities for subtree joint mask sampling.

Core logic used by ``model/anytop.py`` (``AnyTop._sample_subtree_joint_mask``).

A masked draw takes one of two forms:
  * whole limbs: one or two of the rig's limbs (``limb_root_mask``), each with
    every joint below its root. With the limb's own proximal joints gone too,
    its phase can only be recovered from the other limbs -- this is what
    trains inter-limb coordination;
  * one subtree: the subtree of a single non-root, non-leaf joint covering at
    most half the rig -- a tail, a head, a hand -- the shapes
    ``--inpaint_joints`` regenerates at inference.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


# Share of masked draws that take whole limbs, on rigs with at least two limbs;
# the rest take one subtree.
LIMB_MASK_SHARE = 0.6
# Within a whole-limb draw, the share that takes two limbs instead of one.
LIMB_PAIR_SHARE = 0.5


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
    limb_root_mask: np.ndarray,
    rng: Any,
) -> Optional[np.ndarray]:
    """Draw one joint mask for a single skeleton.

    Parameters
    ----------
    parents : list[int]
        Parent indices, length = J.
    limb_root_mask : np.ndarray
        Boolean array of shape ``(J,)`` -- ``True`` at the top joint of each
        limb (``_limb_root_mask`` in the dataset).
    rng : object
        NumPy-compatible RNG object exposing ``random`` and ``choice``.
        This can be a ``np.random.Generator`` or the global ``np.random``
        module so training can participate in checkpointed NumPy RNG state.

    Returns
    -------
    np.ndarray or None
        Boolean mask of shape ``(J,)`` -- ``True`` = joint is masked, or
        ``None`` if the rig has nothing to mask.
    """
    n_joints = len(parents)
    if n_joints <= 1:
        return None

    children = [[] for _ in range(n_joints)]
    for child_idx in range(1, n_joints):
        p = int(parents[child_idx])
        if 0 <= p < n_joints:
            children[p].append(child_idx)

    mask = np.zeros(n_joints, dtype=bool)
    limb_roots = np.flatnonzero(np.asarray(limb_root_mask, dtype=bool)[:n_joints])
    limb_roots = limb_roots[limb_roots != 0]
    if limb_roots.size >= 2 and float(rng.random()) < LIMB_MASK_SHARE:
        count = 2 if float(rng.random()) < LIMB_PAIR_SHARE else 1
        for root_idx in rng.choice(limb_roots, size=count, replace=False):
            mask[collect_subtree_indices(int(root_idx), children)] = True
        return mask

    subtrees = [
        subtree
        for subtree in (
            collect_subtree_indices(idx, children)
            for idx in range(1, n_joints) if children[idx]
        )
        if len(subtree) <= n_joints // 2
    ]
    if not subtrees:
        return None
    mask[subtrees[int(rng.choice(len(subtrees)))]] = True
    return mask


def sample_subtree_joint_mask_batch(
    parents_batch: Any,
    limb_root_mask_batch: Optional[np.ndarray],
    n_joints: np.ndarray,
    max_joints: int,
    joint_mask_prob: float,
    rng: Any,
) -> Optional[np.ndarray]:
    """Batch wrapper around ``sample_subtree_joint_mask``.

    ``joint_mask_prob`` gates whether each sample is perturbed at all. Mask
    assembly stays on the CPU in one NumPy array and is copied back to torch
    only once.
    """
    if joint_mask_prob <= 0.0:
        return None

    batch_size = int(np.asarray(n_joints).shape[0])
    subtree_joint_mask = np.zeros((batch_size, max_joints), dtype=bool)
    any_masked = False

    for batch_index in range(batch_size):
        valid_joint_count = int(n_joints[batch_index])
        if valid_joint_count <= 1:
            continue
        if joint_mask_prob < 1.0 and float(rng.random()) >= joint_mask_prob:
            continue

        parents = np.asarray(parents_batch[batch_index], dtype=np.int64)[:valid_joint_count].tolist()
        if limb_root_mask_batch is None:
            limb_root_mask = np.zeros(valid_joint_count, dtype=bool)
        else:
            limb_root_mask = np.asarray(
                limb_root_mask_batch[batch_index, :valid_joint_count], dtype=bool,
            )

        per_sample_mask = sample_subtree_joint_mask(
            parents=parents,
            limb_root_mask=limb_root_mask,
            rng=rng,
        )
        if per_sample_mask is not None:
            subtree_joint_mask[batch_index, :valid_joint_count] = per_sample_mask
            any_masked = True

    if not any_masked:
        return None
    return subtree_joint_mask
