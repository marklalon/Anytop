from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    from utils.rotation_numpy import quat_multiply_wxyz_np, quat_rotate_wxyz_np
except ImportError:
    from utils.rotation_numpy import quat_multiply_wxyz_np, quat_rotate_wxyz_np


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


def promote_root_once(
    joint_names: list[str],
    parents: np.ndarray,
    offsets: np.ndarray,
    local_rotations: np.ndarray,
    local_positions: np.ndarray,
    orients: Any | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any | None]:
    """Drop the hierarchy root and fold its whole transform into its single child.

    The child's world transform is unchanged: it inherits the dropped joint's
    rotation composed on the outside (``root ⊗ child``), its offset rotated into
    the parent frame and added, and its animated translation likewise. The rest
    ``orients`` take the same composition, or the bind pose ends up rotated
    relative to the animation -- the 90° roll once seen on Alligator, Scorpion
    and Deer.

    Structure only: the caller decides WHICH root deserves this. The loader
    decides conservatively by name and offset; preprocessing decides by measuring
    where the transport actually is (:func:`select_transport_carrier`), which is
    the only one of the two that can tell a wrapper called ``Hips`` from a pelvis
    called ``Hips``.
    """
    names = list(joint_names)
    parents = np.asarray(parents, dtype=np.int32).copy()
    offsets = np.asarray(offsets).copy()
    local_rotations = np.asarray(local_rotations).copy()
    local_positions = np.asarray(local_positions).copy()

    if len(names) < 2:
        raise ValueError("promote_root_once needs at least two joints")
    if int(np.count_nonzero(parents == 0)) != 1:
        raise ValueError(
            "promote_root_once needs the hierarchy root to have exactly one child, "
            f"found {int(np.count_nonzero(parents == 0))}"
        )

    parent_rots = local_rotations[:, 0]
    offsets[1] = offsets[0] + quat_rotate_wxyz_np(parent_rots[0:1], offsets[1:2])[0]
    offsets = offsets[1:]
    local_rotations[:, 1] = quat_multiply_wxyz_np(
        local_rotations[:, 0], local_rotations[:, 1]
    )
    local_rotations = local_rotations[:, 1:]
    local_positions[:, 1] = local_positions[:, 0] + quat_rotate_wxyz_np(
        parent_rots, local_positions[:, 1]
    )
    local_positions = local_positions[:, 1:]
    if orients is not None:
        # Copy first: the loader used to fold in place and clobber its caller's
        # array, which was invisible only because the caller threw it away.
        orients = orients.copy()
        _oq = orients.qs if hasattr(orients, "qs") else orients
        _oq[1] = quat_multiply_wxyz_np(_oq[0:1], _oq[1:2])[0]
        orients = orients[1:]
    parents = parents[1:] - 1
    names = names[1:]

    return names, parents, offsets, local_rotations, local_positions, orients


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
            # Mirror the per-frame rotation handling above: the child inherits the
            # dropped root's rest orientation. Without this the bind/rest pose loses
            # the root's orientation while the animation keeps it, leaving the
            # rest-pose-derived cond t-pose rotated relative to the motion.
            _oq = collapsed_orients.qs if hasattr(collapsed_orients, "qs") else collapsed_orients
            _oq[1] = _oq[0]
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

        (
            collapsed_names,
            collapsed_parents,
            collapsed_offsets,
            collapsed_local_rotations,
            collapsed_local_positions,
            collapsed_orients,
        ) = promote_root_once(
            collapsed_names,
            collapsed_parents,
            collapsed_offsets,
            collapsed_local_rotations,
            collapsed_local_positions,
            collapsed_orients,
        )

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
