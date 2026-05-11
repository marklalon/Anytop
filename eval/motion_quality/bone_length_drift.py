"""
Shared bone-length drift utilities.

Used by both eval/motion_quality/scorer.py and tools/check_bone_length_drift.py.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Comparison-edge resolution
# ---------------------------------------------------------------------------
def resolve_comparison_edges(
    parents: np.ndarray,
    offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build parent-child edge indices from skeleton data.

    Filters out edges whose rest-pose length is too small or non-finite.

    Returns:
        (edge_parent_idx, edge_child_idx) — arrays of edge indices.
    """
    parents = np.asarray(parents, dtype=np.int32)
    offsets = np.asarray(offsets, dtype=np.float64)

    positive_lengths = np.linalg.norm(offsets[parents >= 0], axis=-1)
    positive_lengths = positive_lengths[np.isfinite(positive_lengths)]
    if positive_lengths.size > 0:
        mean_ref_length = float(np.mean(positive_lengths))
        min_ref_length = max(mean_ref_length * 0.1, 1e-8)
    else:
        min_ref_length = 1e-8

    edge_parent_idx: list[int] = []
    edge_child_idx: list[int] = []

    for child_idx in range(len(parents)):
        parent_idx = int(parents[child_idx])
        if parent_idx < 0:
            continue
        ref_length = float(np.linalg.norm(offsets[child_idx]))
        if not np.isfinite(ref_length) or ref_length <= min_ref_length:
            continue
        edge_parent_idx.append(parent_idx)
        edge_child_idx.append(child_idx)

    return (
        np.asarray(edge_parent_idx, dtype=np.int32),
        np.asarray(edge_child_idx, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# Drift computation
# ---------------------------------------------------------------------------
def compute_bone_length_drift(
    world_pos: np.ndarray,       # [T, J, 3]  world FK positions
    edge_parent_idx: np.ndarray, # [E]
    edge_child_idx: np.ndarray,  # [E]
) -> np.ndarray:
    """Compute per-edge bone-length drift relative to frame 0.

    drift[t] = length[t] / length[0] - 1

    Returns (T, E) array. Edges with zero/non-finite frame-0 length are NaN.
    """
    edge_lengths = np.linalg.norm(
        world_pos[:, edge_child_idx, :] - world_pos[:, edge_parent_idx, :],
        axis=-1,
    )  # (T, E)

    drift = np.full(edge_lengths.shape, np.nan, dtype=np.float64)
    first_frame_lengths = np.asarray(edge_lengths[0], dtype=np.float64)
    valid_mask = np.isfinite(first_frame_lengths) & (first_frame_lengths > 1e-8)
    if not np.any(valid_mask):
        return drift

    drift[:, valid_mask] = edge_lengths[:, valid_mask] / first_frame_lengths[valid_mask][np.newaxis, :] - 1.0
    return drift



