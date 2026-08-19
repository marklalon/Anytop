"""
Full-body Inverse Kinematics utilities for the AnyTop pipeline.

The core solver is a ``BasicInverseKinematics``-style iterative IK that
rotates each joint so that the vectors to its children match the target
directions.  It supports:

* Frozen rotation indices — joints whose rotation is preserved as-is.
* Bone stretch correction — elastic per-edge scaling to reduce IK residual.
* Constrained IK targets — projecting noisy world-space targets back onto a
  plausible rigid skeleton before solving.
* Seed position construction — resetting non-root joints to rest offsets
  while preserving trusted position channels.

Usage::

    from utils.fullbody_ik import rebuild_fullbody_animation_with_ik

    rebuilt_anim, mean_err, max_err = rebuild_fullbody_animation_with_ik(
        target_anim,
        rigid_offsets=export_offsets,
        rigid_parents=export_parents,
        iterations=2,
        stretch_factor=0.1,
    )
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from motion_lib.Animation import Animation, positions_global
from motion_lib.Quaternions import Quaternions


# ── Defaults ──────────────────────────────────────────────────────────────────

FULLBODY_IK_ITERATIONS: int = 2
"""Default number of IK iterations for full-body solve."""

DEFAULT_IK_STRETCH_FACTOR: float = 0.1
"""Default ±10 % bone-length elasticity during IK."""


# ── Public helpers ────────────────────────────────────────────────────────────

def normalize_joint_index_selection(
    indices: np.ndarray | list[int] | None,
    joint_count: int,
    *,
    label: str = "indices",
) -> np.ndarray:
    """Validate and deduplicate a set of joint indices.

    Returns a 1-D int32 array (possibly empty).  Raises ``ValueError`` if
    any index is out of bounds.
    """
    if indices is None:
        return np.zeros((0,), dtype=np.int32)

    normalized = np.asarray(indices, dtype=np.int32).reshape(-1)
    if normalized.size == 0:
        return normalized
    if np.any((normalized < 0) | (normalized >= joint_count)):
        raise ValueError(f"{label} must be within [0, {joint_count - 1}]")
    return np.unique(normalized)


def resolve_ik_rebuild_inputs(
    target_anim: Animation,
    *,
    rigid_offsets: np.ndarray | None = None,
    rigid_parents: np.ndarray | None = None,
    preserved_position_indices: np.ndarray | list[int] | None = None,
    preserved_rotation_indices: np.ndarray | list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Validate and normalise IK rebuild parameters.

    Returns
    -------
    parents : (J,) int32 ndarray
    rest_offsets : (J, 3) float64 ndarray
    preserved_position_indices : (P,) int32 ndarray
    preserved_rotation_indices : (R,) int32 ndarray
    root_index : int
    """
    parents = np.asarray(
        target_anim.parents if rigid_parents is None else rigid_parents,
        dtype=np.int32,
    )
    rest_offsets = np.asarray(
        target_anim.offsets if rigid_offsets is None else rigid_offsets,
        dtype=np.float64,
    )
    if target_anim.shape[1] != len(parents):
        raise ValueError(
            f"rigid skeleton joint count {len(parents)} does not match "
            f"animation joint count {target_anim.shape[1]}"
        )
    if rest_offsets.shape != (len(parents), 3):
        raise ValueError(
            f"rest_offsets must have shape ({len(parents)}, 3), "
            f"got {rest_offsets.shape}"
        )

    preserved_position_indices = normalize_joint_index_selection(
        preserved_position_indices,
        len(parents),
        label="preserved_position_indices",
    )
    preserved_rotation_indices = normalize_joint_index_selection(
        preserved_rotation_indices,
        len(parents),
        label="preserved_rotation_indices",
    )

    root_indices = np.flatnonzero(parents < 0)
    if root_indices.size != 1:
        raise ValueError(f"Expected exactly one root joint, got {root_indices.size}")

    return (
        parents,
        rest_offsets,
        preserved_position_indices,
        preserved_rotation_indices,
        int(root_indices[0]),
    )


def build_fullbody_ik_seed_positions(
    target_anim: Animation,
    *,
    rest_offsets: np.ndarray,
    root_index: int,
    preserved_position_indices: np.ndarray,
) -> np.ndarray:
    """Build IK seed positions.

    All joints are reset to rest offsets, then the root position and any
    preserved-position joints are restored from the target animation.
    """
    target_positions = np.asarray(target_anim.positions, dtype=np.float64)

    # Start from rest offsets (rigid skeleton)
    seed_positions = np.broadcast_to(
        rest_offsets[None, :, :], target_positions.shape
    ).copy()

    # Restore root position
    seed_positions[:, root_index, :] = target_positions[:, root_index, :]

    # Restore preserved position joints
    if preserved_position_indices.size > 0:
        seed_positions[:, preserved_position_indices, :] = (
            target_positions[:, preserved_position_indices, :]
        )

    return seed_positions


def _safe_normalize_vectors(
    vectors: np.ndarray,
    *,
    fallback: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(vectors, lengths, out=fallback.copy(), where=lengths > eps)


def _orthogonal_unit_vectors(directions: np.ndarray) -> np.ndarray:
    axes = np.zeros_like(directions)
    abs_dirs = np.abs(directions)
    use_x = (abs_dirs[..., 0] <= abs_dirs[..., 1]) & (
        abs_dirs[..., 0] <= abs_dirs[..., 2]
    )
    use_y = (~use_x) & (abs_dirs[..., 1] <= abs_dirs[..., 2])
    axes[use_x, 0] = 1.0
    axes[use_y, 1] = 1.0
    axes[~use_x & ~use_y, 2] = 1.0
    orthogonal = np.cross(directions, axes)
    return _safe_normalize_vectors(
        orthogonal,
        fallback=np.broadcast_to(np.array([1.0, 0.0, 0.0]), directions.shape),
    )


def _limit_direction_deviation(
    reference_dirs: np.ndarray,
    target_dirs: np.ndarray,
    *,
    max_degrees: float,
) -> np.ndarray:
    """Clamp target directions to a cone around reference directions."""
    if max_degrees >= 180.0:
        return target_dirs
    if max_degrees <= 0.0:
        return reference_dirs

    max_radians = np.deg2rad(float(max_degrees))
    dots = np.sum(reference_dirs * target_dirs, axis=-1, keepdims=True).clip(-1.0, 1.0)
    angles = np.arccos(dots)
    limited = target_dirs.copy()
    mask = (angles[..., 0] > max_radians) & (angles[..., 0] > 1e-8)
    if not np.any(mask):
        return limited

    ref = reference_dirs[mask]
    tgt = target_dirs[mask]
    theta = angles[mask]
    sin_theta = np.sin(theta)
    limited_values = np.empty_like(ref)

    opposite = np.abs(sin_theta[..., 0]) <= 1e-6
    if np.any(opposite):
        ref_opposite = ref[opposite]
        ortho = _orthogonal_unit_vectors(ref_opposite)
        limited_values[opposite] = (
            np.cos(max_radians) * ref_opposite
            + np.sin(max_radians) * ortho
        )

    regular = ~opposite
    if np.any(regular):
        t = max_radians / theta[regular]
        ref_regular = ref[regular]
        tgt_regular = tgt[regular]
        theta_regular = theta[regular]
        sin_regular = sin_theta[regular]
        values = (
            np.sin((1.0 - t) * theta_regular) / sin_regular * ref_regular
            + np.sin(t * theta_regular) / sin_regular * tgt_regular
        )
        limited_values[regular] = _safe_normalize_vectors(
            values,
            fallback=ref_regular,
        )

    limited[mask] = limited_values
    return limited


def constrain_fullbody_ik_targets(
    target_anim: Animation,
    reference_anim: Animation,
    *,
    rest_offsets: np.ndarray,
    parents: np.ndarray,
    root_index: int,
    preserved_position_indices: np.ndarray,
    stretch_factor: float,
) -> np.ndarray:
    """Project noisy IK targets onto plausible per-edge directions and lengths.

    Generated motions can contain non-root local translations that make sibling
    edges under a branching joint disagree strongly.  Basic IK then tries to
    explain those translation residuals as rotations, producing folded joints.
    This projection keeps the useful world-space pose signal, but clamps every
    non-preserved edge to the same length limits the solver may actually output
    and to a cone around the rigid-offset pose implied by the current rotations.
    """
    rest_offsets = np.asarray(rest_offsets, dtype=np.float64)
    parents = np.asarray(parents, dtype=np.int32)
    raw_positions = positions_global(target_anim).astype(np.float64, copy=False)
    reference_positions = positions_global(reference_anim).astype(
        np.float64, copy=False
    )
    if raw_positions.shape != reference_positions.shape:
        raise ValueError(
            "target/reference global positions shape mismatch: "
            f"{raw_positions.shape} vs {reference_positions.shape}"
        )

    projected = np.empty_like(raw_positions)
    projected[:, root_index, :] = raw_positions[:, root_index, :]
    preserved_lookup = {int(index) for index in preserved_position_indices.tolist()}

    children: list[list[int]] = [[] for _ in range(len(parents))]
    for joint_index, parent_index in enumerate(parents):
        if parent_index >= 0:
            children[int(parent_index)].append(int(joint_index))

    stretch = min(1.0, max(0.0, float(stretch_factor)))

    def visit(parent_index: int) -> None:
        for joint_index in children[parent_index]:
            if joint_index in preserved_lookup:
                projected[:, joint_index, :] = raw_positions[:, joint_index, :]
                visit(joint_index)
                continue

            raw_edge = raw_positions[:, joint_index] - raw_positions[:, parent_index]
            raw_length = np.linalg.norm(raw_edge, axis=-1, keepdims=True)
            reference_edge = (
                reference_positions[:, joint_index]
                - reference_positions[:, parent_index]
            )
            reference_dirs = _safe_normalize_vectors(
                reference_edge,
                fallback=np.zeros_like(reference_edge),
            )
            raw_dirs = _safe_normalize_vectors(raw_edge, fallback=reference_dirs)
            target_dirs = _limit_direction_deviation(
                reference_dirs,
                raw_dirs,
                max_degrees=45.0,
            )

            rest_length = float(np.linalg.norm(rest_offsets[joint_index]))
            if rest_length > 1e-8:
                target_length = np.clip(
                    raw_length,
                    rest_length * (1.0 - stretch),
                    rest_length * (1.0 + stretch),
                )
            else:
                target_length = raw_length

            projected[:, joint_index, :] = (
                projected[:, parent_index, :] + target_dirs * target_length
            )
            visit(joint_index)

    visit(root_index)
    return projected


def apply_bone_stretch_correction(
    animation: Animation,
    target_global_positions: np.ndarray,
    parents: np.ndarray,
    stretch_factor: float,
) -> None:
    """Apply per-edge local-translation stretch/compression after IK rotation.

    For each parent→child edge, compute the ratio of target global distance to
    current global distance, clamp to ``[1-stretch_factor, 1+stretch_factor]``,
    and scale the child's local translation by the clamped ratio.

    This allows the skeleton to elastically reach targets that are slightly
    out of reach of the rigid skeleton, reducing IK residual error.
    """
    if abs(stretch_factor) < 1e-9:
        return

    current_positions = positions_global(animation)
    F, J, _ = current_positions.shape

    # Skip root (parent=-1)
    edge_mask = parents >= 0

    for j in range(J):
        if not edge_mask[j]:
            continue
        p = int(parents[j])

        # Current global edge length
        edge_vec = current_positions[:, j] - current_positions[:, p]  # (F, 3)
        edge_len = np.linalg.norm(edge_vec, axis=-1, keepdims=True) + 1e-20

        # Target global edge length (magnitude only)
        target_edge_vec = (
            target_global_positions[:, j] - target_global_positions[:, p]
        )
        target_len = np.linalg.norm(target_edge_vec, axis=-1, keepdims=True) + 1e-20

        # Ratio clamped to [1-stretch, 1+stretch]
        ratio = target_len / edge_len
        lo = 1.0 - stretch_factor
        hi = 1.0 + stretch_factor
        ratio = np.clip(ratio, lo, hi)

        # Scale local translation by the ratio
        animation.positions[:, j] *= ratio


def run_basic_inverse_kinematics_with_constraints(
    animation: Animation,
    target_global_positions: np.ndarray,
    *,
    frozen_rotation_indices: np.ndarray | list[int] | None = None,
    iterations: int = FULLBODY_IK_ITERATIONS,
    stretch_factor: float = 0.0,
) -> Animation:
    """Run full-body IK with optional frozen-rotation constraints.

    Parameters
    ----------
    animation : Animation
        Seed animation (positions will be overwritten by the solver).
    target_global_positions : (F, J, 3) ndarray
        Target world-space joint positions.
    frozen_rotation_indices : array-like, optional
        Joint indices whose rotations are preserved (not touched by IK).
    iterations : int
        Number of IK passes.  Default is ``FULLBODY_IK_ITERATIONS``.
    stretch_factor : float
        Allowed bone-length elasticity.  ``0.0`` = rigid.
    """
    animation_module = importlib.import_module("motion_lib.Animation")
    animation_structure = importlib.import_module("motion_lib.AnimationStructure")

    frozen_rotation_indices = normalize_joint_index_selection(
        frozen_rotation_indices,
        animation.shape[1],
        label="frozen_rotation_indices",
    )

    frozen_rotation_lookup = {int(index) for index in frozen_rotation_indices.tolist()}
    children = animation_structure.children_list(animation.parents)

    for _iteration in range(iterations):
        for joint_index in animation_structure.joints(animation.parents):
            if joint_index in frozen_rotation_lookup:
                continue

            child_indices = np.asarray(children[joint_index], dtype=np.int32)
            if child_indices.size == 0:
                continue

            anim_transforms = animation_module.transforms_global(animation)
            anim_positions = anim_transforms[:, :, :3, 3]
            anim_rotations = Quaternions.from_transforms(
                anim_transforms[:, :, :3, :3]
            )

            joint_dirs = (
                anim_positions[:, child_indices]
                - anim_positions[:, np.newaxis, joint_index]
            )
            target_dirs = (
                target_global_positions[:, child_indices]
                - target_global_positions[:, np.newaxis, joint_index]
            )

            if (
                child_indices.size > 1
                and (joint_dirs == 0).all()
                and (target_dirs == 0).all()
            ):
                continue

            joint_lengths = np.sqrt(np.sum(joint_dirs ** 2.0, axis=-1)) + 1e-20
            target_lengths = np.sqrt(np.sum(target_dirs ** 2.0, axis=-1)) + 1e-20

            joint_dirs = joint_dirs / joint_lengths[:, :, np.newaxis]
            target_dirs = target_dirs / target_lengths[:, :, np.newaxis]

            angles = np.arccos(
                np.sum(joint_dirs * target_dirs, axis=2).clip(-1, 1)
            )
            axes = np.cross(joint_dirs, target_dirs)
            axes = -anim_rotations[:, joint_index, np.newaxis] * axes

            valid_directions = (joint_lengths > 1e-4)[0]
            if not np.any(valid_directions):
                continue

            rotations = Quaternions.from_angle_axis(angles, axes)
            if rotations.shape[1] == 1:
                averaged_rotation = rotations[:, 0]
            else:
                averaged_rotation = Quaternions.exp(
                    rotations[:, valid_directions].log().mean(axis=-2)
                )

            animation.rotations[:, joint_index] = (
                animation.rotations[:, joint_index] * averaged_rotation
            )

    # Post-rotation: apply bone stretch correction
    if abs(stretch_factor) > 1e-9:
        apply_bone_stretch_correction(
            animation, target_global_positions, animation.parents, stretch_factor
        )

    return animation


def rebuild_fullbody_animation_with_ik(
    target_anim: Animation,
    *,
    rigid_offsets: np.ndarray | None = None,
    rigid_parents: np.ndarray | None = None,
    preserved_position_indices: np.ndarray | list[int] | None = None,
    preserved_rotation_indices: np.ndarray | list[int] | None = None,
    iterations: int = FULLBODY_IK_ITERATIONS,
    stretch_factor: float = DEFAULT_IK_STRETCH_FACTOR,
) -> tuple[Animation, float, float]:
    """Force a full-body IK rebuild against the current world-space motion.

    Non-preserved local translations are reset to the requested rigid skeleton
    offsets before IK.  The world-space IK target is first projected onto
    plausible per-edge directions and the same bone-length stretch limits
    the rebuilt animation may output.

    Parameters
    ----------
    target_anim : Animation
        Source animation whose world-space positions are the IK target.
    rigid_offsets : (J, 3) ndarray, optional
        Rigid-skeleton rest offsets.  Defaults to ``target_anim.offsets``.
    rigid_parents : (J,) ndarray, optional
        Rigid-skeleton parent indices.  Defaults to ``target_anim.parents``.
    preserved_position_indices : array-like, optional
        Joint indices whose local positions are kept from the source.
    preserved_rotation_indices : array-like, optional
        Joint indices whose rotations are kept from the source (frozen).
    iterations : int
        Number of IK passes.  Default is ``FULLBODY_IK_ITERATIONS``.
    stretch_factor : float
        Allowed bone-length elasticity (e.g. 0.1 = ±10 %).
        Default is ``DEFAULT_IK_STRETCH_FACTOR``.

    Returns
    -------
    rebuilt_anim : Animation
        IK-reconstructed animation.
    ik_mean_error : float
        Mean per-joint position error (mm).
    ik_max_error : float
        Max per-joint position error (mm).
    """
    parents, rest_offsets, preserved_position_indices, preserved_rotation_indices, root_index = (
        resolve_ik_rebuild_inputs(
            target_anim,
            rigid_offsets=rigid_offsets,
            rigid_parents=rigid_parents,
            preserved_position_indices=preserved_position_indices,
            preserved_rotation_indices=preserved_rotation_indices,
        )
    )

    seed_positions = build_fullbody_ik_seed_positions(
        target_anim,
        rest_offsets=rest_offsets,
        root_index=root_index,
        preserved_position_indices=preserved_position_indices,
    )

    ik_seed = Animation(
        target_anim.rotations.copy(),
        seed_positions,
        target_anim.orients.copy(),
        rest_offsets.copy(),
        parents.copy(),
    )

    target_global_positions = constrain_fullbody_ik_targets(
        target_anim,
        ik_seed,
        rest_offsets=rest_offsets,
        parents=parents,
        root_index=root_index,
        preserved_position_indices=preserved_position_indices,
        stretch_factor=stretch_factor,
    )
    rebuilt_anim = run_basic_inverse_kinematics_with_constraints(
        ik_seed,
        target_global_positions,
        frozen_rotation_indices=preserved_rotation_indices,
        iterations=iterations,
        stretch_factor=stretch_factor,
    )
    rebuilt_global_positions = positions_global(rebuilt_anim).astype(
        np.float64, copy=False
    )
    per_joint_error = np.linalg.norm(
        rebuilt_global_positions - target_global_positions, axis=-1
    )
    return rebuilt_anim, float(per_joint_error.mean()), float(per_joint_error.max())
