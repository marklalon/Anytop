from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    from utils.rotation_numpy import quat_multiply_wxyz_np, quat_rotate_wxyz_np
except ImportError:
    from Anytop.utils.rotation_numpy import quat_multiply_wxyz_np, quat_rotate_wxyz_np


# Common root joint names: when the root already carries a semantic name like
# these, it is a real skeleton root and should not be collapsed as a wrapper.
COMMON_ROOT_NAMES = frozenset(
    n.lower()
    for n in (
        "hips", "hip", "pelvis", "root", "cog",
        "spine", "spine1", "body",
        "bip", "bip01",
        "koshi",
    )
)


def collapse_root_skeleton(
    joint_names: list[str],
    parents: np.ndarray,
    offsets: np.ndarray,
    local_rotations: np.ndarray,
    local_positions: np.ndarray,
    orients: Any | None = None,
    *,
    warn_path: str | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any | None]:
    """Collapse redundant/wrapper roots using the shared FBX/BVH loader rules.

    The same structural rules apply to both sampled animation channels and a
    one-frame rest pose represented as local rotations/positions.
    """
    collapsed_names = list(joint_names)
    collapsed_parents = np.asarray(parents, dtype=np.int32).copy()
    collapsed_offsets = np.asarray(offsets).copy()
    collapsed_local_rotations = np.asarray(local_rotations).copy()
    collapsed_local_positions = np.asarray(local_positions).copy()
    collapsed_orients = orients

    def _root_is_semantic() -> bool:
        return bool(collapsed_names) and collapsed_names[0].lower() in COMMON_ROOT_NAMES

    def _drop_redundant_joint_one() -> None:
        nonlocal collapsed_names, collapsed_parents, collapsed_offsets
        nonlocal collapsed_local_rotations, collapsed_local_positions, collapsed_orients

        collapsed_offsets[1] = collapsed_offsets[0]
        collapsed_offsets = collapsed_offsets[1:]
        collapsed_local_rotations[:, 1] = collapsed_local_rotations[:, 0]
        collapsed_local_rotations = collapsed_local_rotations[:, 1:]
        collapsed_local_positions[:, 1] = collapsed_local_positions[:, 0]
        collapsed_local_positions = collapsed_local_positions[:, 1:]
        if collapsed_orients is not None:
            collapsed_orients = collapsed_orients[1:]
        collapsed_parents = collapsed_parents[1:] - 1
        collapsed_parents[1:][collapsed_parents[1:] < 0] = 0
        collapsed_names[1] = collapsed_names[0]
        collapsed_names = collapsed_names[1:]

    def _promote_child_root(*, emit_warning: bool) -> None:
        nonlocal collapsed_names, collapsed_parents, collapsed_offsets
        nonlocal collapsed_local_rotations, collapsed_local_positions, collapsed_orients

        if emit_warning and warn_path is not None:
            print(
                f"\033[33m[WARN] {Path(warn_path).name}: collapsing root joint "
                f"'{collapsed_names[0]}' (all-zero offset, single child) "
                f"to child '{collapsed_names[1]}'\033[0m"
            )

        parent_rots = collapsed_local_rotations[:, 0]
        collapsed_offsets[1] = collapsed_offsets[0] + quat_rotate_wxyz_np(
            parent_rots[0:1],
            collapsed_offsets[1:2],
        )[0]
        collapsed_offsets = collapsed_offsets[1:]
        collapsed_local_rotations[:, 1] = quat_multiply_wxyz_np(
            collapsed_local_rotations[:, 0],
            collapsed_local_rotations[:, 1],
        )
        collapsed_local_rotations = collapsed_local_rotations[:, 1:]
        collapsed_local_positions[:, 1] = collapsed_local_positions[:, 0] + quat_rotate_wxyz_np(
            parent_rots,
            collapsed_local_positions[:, 1],
        )
        collapsed_local_positions = collapsed_local_positions[:, 1:]
        if collapsed_orients is not None:
            collapsed_orients = collapsed_orients[1:]
        collapsed_parents = collapsed_parents[1:] - 1
        collapsed_names = collapsed_names[1:]

    if len(collapsed_names) > 1 and np.isclose(collapsed_offsets[1], 0).all():
        if len(collapsed_parents[collapsed_parents == 1]) == 0:
            _drop_redundant_joint_one()
        elif len(collapsed_parents[collapsed_parents == 0]) == 1 and not _root_is_semantic():
            _promote_child_root(emit_warning=False)

    while (
        len(collapsed_names) > 1
        and np.isclose(collapsed_offsets[0], 0).all()
        and len(collapsed_parents[collapsed_parents == 0]) == 1
        and not _root_is_semantic()
    ):
        _promote_child_root(emit_warning=False)

    return (
        collapsed_names,
        collapsed_parents,
        collapsed_offsets,
        collapsed_local_rotations,
        collapsed_local_positions,
        collapsed_orients,
    )
