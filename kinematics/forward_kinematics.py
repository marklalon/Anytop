"""
Differentiable Forward Kinematics.

  θ (joint rotations [F,J,4]) + root pose → world transforms [F,J,4,4]

Uses depth-batched computation: all joints at the same tree depth are
processed in a single batched matmul, reducing ~J sequential Python ops
to ~max_depth batched ops.
"""
from __future__ import annotations
import torch
from torch import Tensor
from .skeleton import Skeleton
from Anytop.utils.quaternion import (
    matrix_to_quat,
    quat_conjugate,
    quat_multiply,
    quat_normalize,
    quat_to_matrix,
)
from Anytop.utils.rotation_conversions import rotation_6d_to_matrix_safe


_PARENTS_DEPTH_CACHE: dict[tuple[int, ...], list[tuple[tuple[int, ...], tuple[int, ...]]]] = {}


def _parents_depth_levels(parents) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    parents_key = tuple(int(parent) for parent in parents)
    cached = _PARENTS_DEPTH_CACHE.get(parents_key)
    if cached is not None:
        return cached

    joint_count = len(parents_key)
    depths = [0] * joint_count
    for joint_index in range(1, joint_count):
        parent_index = parents_key[joint_index]
        depths[joint_index] = depths[parent_index] + 1 if parent_index >= 0 else 0

    levels = []
    max_depth = max(depths) if depths else 0
    for depth in range(1, max_depth + 1):
        joint_ids = tuple(joint_index for joint_index, joint_depth in enumerate(depths) if joint_depth == depth)
        if not joint_ids:
            continue
        parent_ids = tuple(parents_key[joint_index] for joint_index in joint_ids)
        levels.append((joint_ids, parent_ids))

    _PARENTS_DEPTH_CACHE[parents_key] = levels
    return levels


def _expand_batch_param(param: Tensor, batch_size: int) -> Tensor:
    if param.dim() == 2:
        return param.unsqueeze(0).expand(batch_size, -1, -1)
    return param


def _quat_apply(q: Tensor, v: Tensor) -> Tensor:
    return torch.matmul(quat_to_matrix(q), v.unsqueeze(-1)).squeeze(-1)


def _cumulative_rest_rotations(rest_q: Tensor, parents) -> list[Tensor]:
    joint_count = rest_q.shape[1]
    cumulative_by_joint = [None] * joint_count
    cumulative_by_joint[0] = rest_q[:, 0, :]

    for joint_ids_raw, parent_ids_raw in _parents_depth_levels(parents):
        parent_cumulative = torch.stack([cumulative_by_joint[parent_id] for parent_id in parent_ids_raw], dim=1)
        joint_rest = rest_q[:, joint_ids_raw, :]
        cumulative_batch = quat_multiply(parent_cumulative, joint_rest)
        for batch_offset, joint_id in enumerate(joint_ids_raw):
            cumulative_by_joint[joint_id] = cumulative_batch[:, batch_offset, :]

    return cumulative_by_joint


def _decode_root_translation_from_features(root_rotation: Tensor, root_features: Tensor) -> Tensor:
    local_delta = torch.zeros(root_features.shape[:-1] + (3,), dtype=root_features.dtype, device=root_features.device)
    local_delta[:, 1:, 0] = root_features[:, :-1, 9]
    local_delta[:, 1:, 2] = root_features[:, :-1, 11]
    world_delta = _quat_apply(quat_conjugate(root_rotation), local_delta)
    root_path = torch.cumsum(world_delta, dim=1)
    root_path[..., 1] = root_features[..., 1]

    root_offset_world = _quat_apply(quat_conjugate(root_rotation), root_features[..., :3])
    root_translation = root_offset_world.clone()
    root_translation[..., 0] += root_path[..., 0]
    root_translation[..., 2] += root_path[..., 2]
    return root_translation


def batched_fk(
    rot6d: Tensor,
    root_translation: Tensor,
    offsets: Tensor,
    stretch: Tensor,
    rest_q: Tensor,
    canon_joint_rot: Tensor,
    parents,
    padding_mask: Tensor | None = None,
) -> Tensor:
    """Decode v4 motion channels to world-space joint positions.

    Shapes:
      rot6d           [B, F, J, 6]
      root_translation[B, F, 3]
      offsets         [B, J, 3] or [J, 3]
      stretch         [B, F, J]
      rest_q          [B, J, 4] or [J, 4]
      canon_joint_rot [B, J, 4] or [J, 4]
    """
    batch_size, frame_count, joint_count = rot6d.shape[:3]
    dtype = rot6d.dtype
    device = rot6d.device

    offsets = _expand_batch_param(offsets.to(device=device, dtype=dtype), batch_size)
    rest_q = quat_normalize(_expand_batch_param(rest_q.to(device=device, dtype=dtype), batch_size))
    canon_joint_rot = quat_normalize(_expand_batch_param(canon_joint_rot.to(device=device, dtype=dtype), batch_size))
    stretch = stretch.to(device=device, dtype=dtype)

    rot_q = matrix_to_quat(rotation_6d_to_matrix_safe(rot6d))
    rot_q = quat_normalize(rot_q)
    cumulative_rest_by_joint = _cumulative_rest_rotations(rest_q, parents)
    root_rest = rest_q[:, None, :1, :].expand(batch_size, frame_count, 1, 4)

    if joint_count > 1:
        canon_nonroot = canon_joint_rot[:, None, 1:, :].expand(batch_size, frame_count, joint_count - 1, 4)
        canon_nonroot_inv = quat_conjugate(canon_nonroot)
        delta_nonroot = quat_multiply(quat_multiply(canon_nonroot_inv, rot_q[:, :, 1:]), canon_nonroot)
        parent_cumulative_rest = torch.stack([cumulative_rest_by_joint[parent_id] for parent_id in parents[1:]], dim=1)
        parent_cumulative_rest = parent_cumulative_rest[:, None, :, :].expand(batch_size, frame_count, joint_count - 1, 4)
        rest_nonroot = rest_q[:, None, 1:, :].expand(batch_size, frame_count, joint_count - 1, 4)
        local_nonroot = quat_multiply(
            quat_multiply(
                quat_multiply(quat_conjugate(parent_cumulative_rest), delta_nonroot),
                parent_cumulative_rest,
            ),
            rest_nonroot,
        )
        local_q = torch.cat((quat_multiply(rot_q[:, :, :1], root_rest), local_nonroot), dim=2)
    else:
        local_q = quat_multiply(rot_q[:, :, :1], root_rest)

    root_local_pos = offsets[:, None, :1, :].expand(batch_size, frame_count, 1, 3)
    if joint_count > 1:
        nonroot_local_pos = offsets[:, None, 1:, :].expand(batch_size, frame_count, joint_count - 1, 3)
        local_pos = torch.cat((root_local_pos, nonroot_local_pos * stretch[:, :, 1:, None]), dim=2)
    else:
        local_pos = root_local_pos

    world_q_by_joint = [None] * joint_count
    world_pos_by_joint = [None] * joint_count
    world_q_by_joint[0] = local_q[:, :, 0]
    world_pos_by_joint[0] = root_translation

    for joint_ids_raw, parent_ids_raw in _parents_depth_levels(parents):
        parent_q = torch.stack([world_q_by_joint[parent_id] for parent_id in parent_ids_raw], dim=2)
        joint_q = local_q[:, :, joint_ids_raw, :]
        world_q_batch = quat_multiply(parent_q, joint_q)
        parent_pos = torch.stack([world_pos_by_joint[parent_id] for parent_id in parent_ids_raw], dim=2)
        joint_local_pos = local_pos[:, :, joint_ids_raw, :]
        world_pos_batch = parent_pos + _quat_apply(parent_q, joint_local_pos)

        for batch_offset, joint_id in enumerate(joint_ids_raw):
            world_q_by_joint[joint_id] = world_q_batch[:, :, batch_offset, :]
            world_pos_by_joint[joint_id] = world_pos_batch[:, :, batch_offset, :]

    world_pos = torch.stack(world_pos_by_joint, dim=2)

    if padding_mask is not None:
        world_pos = world_pos * padding_mask.to(dtype=dtype, device=device)[:, None, :, None]
    return world_pos


def batched_fk_from_features(
    motion_features: Tensor,
    offsets: Tensor,
    rest_q: Tensor,
    canon_joint_rot: Tensor,
    parents,
    padding_mask: Tensor | None = None,
    stretch_limit: float = 0.30,
) -> Tensor:
    root_rot6d = motion_features[:, :, 0, 3:9]
    root_rotation = matrix_to_quat(rotation_6d_to_matrix_safe(root_rot6d))
    root_rotation = quat_normalize(root_rotation)
    root_translation = _decode_root_translation_from_features(root_rotation, motion_features[:, :, 0, :])

    if motion_features.shape[2] > 1:
        stretch = torch.cat(
            (
                torch.ones(motion_features.shape[:2] + (1,), dtype=motion_features.dtype, device=motion_features.device),
                motion_features[:, :, 1:, 9].clamp(1.0 - stretch_limit, 1.0 + stretch_limit),
            ),
            dim=2,
        )
    else:
        stretch = torch.ones(motion_features.shape[:3], dtype=motion_features.dtype, device=motion_features.device)

    return batched_fk(
        motion_features[:, :, :, 3:9],
        root_translation,
        offsets,
        stretch,
        rest_q,
        canon_joint_rot,
        parents,
        padding_mask=padding_mask,
    )


def forward_kinematics(
    joint_rotations: Tensor,    # [F, J, 4]  local joint quaternions
    root_translation: Tensor,   # [F, 3]     root world position
    root_rotation: Tensor,      # [F, 4]     root world orientation
    skeleton: Skeleton,
) -> tuple[Tensor, Tensor]:
    """Compute world-space transforms for every joint via depth-batched FK.

    All joints at the same tree depth are computed in one batched matmul,
    turning O(J) sequential Python ops into O(max_depth) batched ops.

    Returns:
        world_transforms: [F, J, 4, 4]  world matrices for each bone
        joint_positions:  [F, J, 3]     world position of each joint origin
    """
    F_frames = joint_rotations.shape[0]
    J = skeleton.num_joints
    device = joint_rotations.device

    rest_q = torch.stack([b.rest_rotation for b in skeleton.bones], dim=0).to(device=device, dtype=joint_rotations.dtype)
    local_q = quat_multiply(
        rest_q.unsqueeze(0).expand(F_frames, -1, -1),
        joint_rotations,
    )

    # Convert all local joint quaternions to 3×3 rotation matrices in one batch
    all_R = quat_to_matrix(local_q)  # [F, J, 3, 3]

    # Build all local transforms at once: [F, J, 4, 4]
    local_T = torch.zeros(F_frames, J, 4, 4, dtype=joint_rotations.dtype, device=device)
    local_T[:, :, :3, :3] = all_R
    local_T[:, :, :3, 3] = skeleton.rest_offsets.unsqueeze(0).expand(F_frames, -1, -1)
    local_T[:, :, 3, 3] = 1.0

    # Root world transform: [F, 4, 4]
    root_R = quat_to_matrix(root_rotation)  # [F, 3, 3]
    root_T = torch.zeros(F_frames, 4, 4, dtype=joint_rotations.dtype, device=device)
    root_T[:, :3, :3] = root_R
    root_T[:, :3, 3] = root_translation
    root_T[:, 3, 3] = 1.0

    # Output buffer
    world_T = torch.zeros(F_frames, J, 4, 4, dtype=joint_rotations.dtype, device=device)

    for bone_ids, parent_ids in skeleton.depth_levels:
        # local_T for this batch: [F, batch_size, 4, 4]
        local_batch = local_T[:, bone_ids]

        if parent_ids[0].item() == -1:
            # Root level: parent is the root world transform
            parent_batch = root_T.unsqueeze(1).expand(-1, bone_ids.shape[0], -1, -1)
        else:
            parent_batch = world_T[:, parent_ids]  # [F, batch_size, 4, 4]

        # Batched matmul: [F, batch_size, 4, 4] @ [F, batch_size, 4, 4]
        # Under AMP autocast, matmul may return fp16 while world_T is fp32.
        # Cast explicitly to keep assignment dtype-safe.
        world_T[:, bone_ids] = (parent_batch @ local_batch).to(world_T.dtype)

    joint_positions = world_T[:, :, :3, 3]  # [F, J, 3]
    return world_T, joint_positions
