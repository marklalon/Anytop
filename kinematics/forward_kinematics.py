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
from utils.quaternion import quat_multiply, quat_to_matrix


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
