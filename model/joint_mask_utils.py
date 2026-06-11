"""Shared utilities for subtree joint mask sampling.

Core logic used by both:
  - ``model/anytop.py`` (PyTorch, batched)
  - ``tools/sample_augmented_bvh.py`` (NumPy, single-sample)
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch


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
    joint_mask_budget: float,
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
    joint_mask_budget : float
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
    budget = min(int(joint_mask_budget * non_root_count), non_root_count)
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

    # Single-subtree selection: pick one candidate weighted by size**2 so larger
    # limbs dominate, matching the behaviour of the torch graph-capturable
    # sampler (sample_subtree_joint_mask_batch_torch).
    subtree_sizes = np.asarray([len(subtree) for subtree in candidate_subtrees], dtype=np.float64)
    weights = subtree_sizes ** 2
    weights /= weights.sum()
    chosen_pos = int(rng.choice(len(candidate_subtrees), p=weights))
    subtree = candidate_subtrees[chosen_pos]

    mask = np.zeros(n_joints, dtype=bool)
    mask[subtree] = True
    return mask


def sample_subtree_joint_mask_batch(
    parents_batch: Any,
    candidate_root_mask_batch: Optional[np.ndarray],
    n_joints: np.ndarray,
    max_joints: int,
    joint_mask_prob: float,
    joint_mask_budget: float,
    rng: Any,
) -> Optional[np.ndarray]:
    """Batch wrapper around ``sample_subtree_joint_mask``.

    ``joint_mask_prob`` gates whether each sample is perturbed at all;
    ``joint_mask_budget`` caps the size of the selected subtrees. Mask
    assembly stays on the CPU in one NumPy array and is copied back to torch
    only once.
    """
    if joint_mask_prob <= 0.0 or joint_mask_budget <= 0.0:
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
        if candidate_root_mask_batch is None:
            candidate_root_mask = np.ones(valid_joint_count, dtype=bool)
        else:
            candidate_root_mask = np.asarray(
                candidate_root_mask_batch[batch_index, :valid_joint_count],
                dtype=bool,
            ).copy()
        candidate_root_mask[0] = False

        per_sample_mask = sample_subtree_joint_mask(
            parents=parents,
            candidate_root_mask=candidate_root_mask,
            joint_mask_budget=joint_mask_budget,
            rng=rng,
        )
        if per_sample_mask is not None:
            subtree_joint_mask[batch_index, :valid_joint_count] = per_sample_mask
            any_masked = True

    if not any_masked:
        return None
    return subtree_joint_mask


# ---------------------------------------------------------------------------
# Batched subtree mask sampler (pure torch, on-device)
# ---------------------------------------------------------------------------

def sample_subtree_joint_mask_batch_torch(
    parents_padded: "torch.Tensor",
    valid_mask: "torch.Tensor",
    candidate_root_mask: Optional["torch.Tensor"],
    n_joints: "torch.Tensor",
    joint_mask_prob: float,
    joint_mask_budget: float,
    generator: Optional["torch.Generator"] = None,
) -> Optional["torch.Tensor"]:
    """On-device, CUDA-graph-capturable subtree joint-mask sampler.

    This is a *simplified* sampler (it intentionally does not reproduce the
    NumPy greedy multi-subtree fill): each sample masks at most **one** coherent
    subtree, chosen among the budget-fitting candidate roots with probability
    proportional to ``size**2`` (so larger limbs dominate). That keeps the whole
    routine free of the two things that break CUDA graph capture:

      * **no host<->device sync** -- no ``.item()`` / ``bool(tensor)`` / ``.cpu()``;
        the only control flow is the fixed-trip ancestor climb (a constant
        number of kernels, captured once and replayed as a single graph launch);
      * **no ``torch.multinomial``** -- the single weighted pick uses the
        Gumbel-max trick (``argmax(log w + Gumbel)``), built from ``torch.rand``,
        which is graph-capturable.

    It always returns a ``[B, J]`` tensor (never ``None`` on the hot path); an
    all-``False`` row -- sample gated off, or no budget-fitting candidate -- is a
    numeric no-op downstream (the reliability bias multiplies it to zero), which
    keeps the control flow constant across steps for capture.

    Parameters
    ----------
    parents_padded : torch.Tensor
        ``[B, J]`` long. Parent index per joint; root and padded slots use a
        negative sentinel (``-1``). Padded/invalid columns are ignored.
    valid_mask : torch.Tensor
        ``[B, J]`` bool, ``True`` for real joints (``j < n_joints``).
    candidate_root_mask : torch.Tensor or None
        ``[B, J]`` bool — ``True`` where a joint may root a masked subtree.
        ``None`` means every valid non-root joint is a candidate.
    n_joints : torch.Tensor
        ``[B]`` long, real joint count per sample.
    joint_mask_prob, joint_mask_budget : float
        Per-sample gate probability and budget fraction (of non-root joints).

    Returns
    -------
    torch.Tensor or None
        ``[B, J]`` bool mask (``True`` = joint selected). ``None`` only for the
        config-time short-circuits (probability/budget disabled, empty batch),
        which are constant across steps.
    """
    if joint_mask_prob <= 0.0 or joint_mask_budget <= 0.0:
        return None

    device = parents_padded.device
    batch_size, n_joints_pad = parents_padded.shape
    if batch_size == 0 or n_joints_pad == 0:
        return None

    n_joints = n_joints.to(device=device, dtype=torch.long).reshape(-1)
    joint_index = torch.arange(n_joints_pad, device=device)

    # --- subtree membership via bounded ancestor climb -------------------
    # member[b, r, j] == True  <=>  r is an ancestor-or-self of j, i.e. the
    # subtree rooted at r contains joint j (matches collect_subtree_indices).
    # The loop trip count is the (static) joint dim, so it captures cleanly.
    member = torch.zeros(batch_size, n_joints_pad, n_joints_pad, dtype=torch.bool, device=device)
    current = joint_index[None, :].expand(batch_size, n_joints_pad).clone()  # each j starts at itself
    ones_src = torch.ones(batch_size, 1, n_joints_pad, dtype=torch.bool, device=device)
    for _ in range(n_joints_pad):  # depth is bounded by the joint count
        write_index = current.clamp(0, n_joints_pad - 1).unsqueeze(1)        # [B, 1, J]
        member.scatter_(1, write_index, ones_src)                            # member[b, current, j] = True
        parent = parents_padded.gather(1, current.clamp(min=0))              # [B, J]
        current = torch.where(parent >= 0, parent, current)                  # stop at root/padding

    member = member & valid_mask[:, None, :] & valid_mask[:, :, None]
    subtree_size = member.sum(-1)  # [B, J] — counts only valid joints

    non_root = (n_joints - 1).clamp(min=0)
    budget = torch.minimum(
        (joint_mask_budget * non_root.to(torch.float64)).floor().to(torch.long),
        non_root,
    )  # [B] — int() truncation in float64 to match the NumPy budget

    if candidate_root_mask is None:
        candidate = valid_mask.clone()
    else:
        candidate = candidate_root_mask.to(device=device, dtype=torch.bool) & valid_mask
    candidate = candidate & (joint_index[None, :] >= 1)  # root (index 0) is never a candidate

    is_candidate = candidate & (subtree_size > 0) & (subtree_size <= budget[:, None])  # [B, J]

    # --- one weighted pick per sample via Gumbel-max (weight = size**2) ---
    gate = torch.rand(batch_size, device=device, generator=generator)
    uniform = torch.rand(batch_size, n_joints_pad, device=device, generator=generator).clamp_(1e-9, 1.0 - 1e-9)
    gumbel = -torch.log(-torch.log(uniform))
    log_weight = 2.0 * torch.log(subtree_size.clamp(min=1).to(torch.float32))
    neg_inf = torch.full_like(log_weight, float('-inf'))
    keys = torch.where(is_candidate, log_weight + gumbel, neg_inf)  # [B, J]
    chosen = keys.argmax(dim=-1)  # [B]

    has_candidate = is_candidate.any(dim=-1)  # [B] (tensor reduction, not a host sync)
    active = (gate < joint_mask_prob) & (n_joints > 1) & (budget > 0) & has_candidate

    batch_arange = torch.arange(batch_size, device=device)
    chosen_subtree = member[batch_arange, chosen]  # [B, J]
    return chosen_subtree & active[:, None]
