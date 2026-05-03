"""
Smoke test for BVH export/import consistency.

Creates a tiny procedural skeleton animation, exports it to BVH, imports it
back through Blender, and verifies the imported pose matches FK from the
original animation.

Run:
    .venv\Scripts\python.exe tests/test_bvh_roundtrip.py
"""
from __future__ import annotations

import math
import os
import shutil
import sys
import tempfile

import torch

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


def _build_minimal_skeleton(device: torch.device):
    from Anytop.kinematics import forward_kinematics
    from Anytop.kinematics.skeleton import Bone, Skeleton

    bones = [
        Bone(
            id=0,
            name="root",
            parent_id=None,
            rest_offset=torch.tensor([0.0, 0.0, 0.0], device=device),
            rest_rotation=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device),
        ),
        Bone(
            id=1,
            name="mid",
            parent_id=0,
            rest_offset=torch.tensor([0.0, 1.0, 0.0], device=device),
            rest_rotation=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device),
        ),
        Bone(
            id=2,
            name="tip",
            parent_id=1,
            rest_offset=torch.tensor([0.0, 1.0, 0.0], device=device),
            rest_rotation=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device),
        ),
    ]
    skeleton = Skeleton(bones)

    num_joints = len(bones)
    q_identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    joint_rotations = q_identity.view(1, 1, 4).expand(1, num_joints, 4).clone()
    root_translation = torch.zeros(1, 3, device=device)
    root_rotation = q_identity.view(1, 4).clone()

    world_transforms, _ = forward_kinematics(
        joint_rotations,
        root_translation,
        root_rotation,
        skeleton,
    )
    skeleton.bind_matrices = torch.inverse(world_transforms[0])
    return skeleton


def _axis_angle_to_quat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / axis.norm()
    half_angle = angle * 0.5
    sin_half = torch.sin(half_angle)
    return torch.stack([
        torch.cos(half_angle),
        axis[0] * sin_half,
        axis[1] * sin_half,
        axis[2] * sin_half,
    ], dim=-1)


def _make_animation(num_frames: int, num_joints: int, device: torch.device):
    q_identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

    joint_rotations = q_identity.view(1, 1, 4).expand(num_frames, num_joints, 4).clone()

    root_translation = torch.zeros(num_frames, 3, device=device)
    root_translation[:, 1] = torch.linspace(0.0, 1.5, num_frames, device=device)

    root_angles = torch.linspace(0.0, math.pi * 0.5, num_frames, device=device)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device)
    root_rotation = torch.stack([
        _axis_angle_to_quat(z_axis, angle) for angle in root_angles
    ], dim=0)

    return joint_rotations, root_translation, root_rotation


def _expected_tip_positions(skeleton, joint_rotations, root_translation, root_rotation):
    from Anytop.kinematics import forward_kinematics

    world_transforms, joint_positions = forward_kinematics(
        joint_rotations,
        root_translation,
        root_rotation,
        skeleton,
    )

    tip_head = joint_positions[:, 2]

    tip_tail_local = torch.tensor([0.0, 0.1, 0.0, 1.0], dtype=world_transforms.dtype)
    tip_tail = (world_transforms[:, 2] @ tip_tail_local.to(world_transforms.device)).cpu()[:, :3]
    return tip_head.cpu(), tip_tail


def _parse_motion_width(bvh_path: str) -> tuple[int, list[int]]:
    with open(bvh_path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    motion_idx = lines.index("MOTION")
    frame_lines = lines[motion_idx + 3 :]
    widths = [len(line.split()) for line in frame_lines]
    return len(frame_lines), widths


def main() -> None:
    import bpy

    from Anytop.utils.exporter import AnimationExporter

    device = torch.device("cpu")
    skeleton = _build_minimal_skeleton(device)
    joint_rotations, root_translation, root_rotation = _make_animation(
        num_frames=6,
        num_joints=skeleton.num_joints,
        device=device,
    )
    expected_head, expected_tail = _expected_tip_positions(
        skeleton,
        joint_rotations,
        root_translation,
        root_rotation,
    )

    exporter = AnimationExporter(skeleton, fps=30.0)

    tmp_dir = tempfile.mkdtemp(prefix="bvh_smoke_")
    try:
        output_path = os.path.join(tmp_dir, "smoke_roundtrip.bvh")
        exporter.export(
            joint_rotations,
            root_translation,
            root_rotation,
            output_path,
        )

        num_frames, widths = _parse_motion_width(output_path)
        expected_width = 6 + 3 * (skeleton.num_joints - 1)
        assert num_frames == joint_rotations.shape[0], (
            f"Expected {joint_rotations.shape[0]} frames, got {num_frames}"
        )
        assert all(width == expected_width for width in widths), (
            f"Expected motion width {expected_width}, got {widths}"
        )

        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_anim.bvh(filepath=output_path)

        armature = next((obj for obj in bpy.data.objects if obj.type == "ARMATURE"), None)
        assert armature is not None, "No armature found after BVH import"

        scene = bpy.context.scene
        tip_bone = armature.pose.bones.get("tip")
        assert tip_bone is not None, "Imported armature is missing pose bone 'tip'"

        # Blender's BVH importer swaps Y and Z (BVH is traditionally Z-up,
        # Blender is Y-up).  Our internal FK uses Y-up, so we must swap
        # Y↔Z on the expected positions before comparing.
        def _swap_yz(p):
            return (p[0], p[2], p[1])

        max_head_error = 0.0
        max_tail_error = 0.0
        for frame_idx in range(num_frames):
            scene.frame_set(frame_idx + 1)

            imported_head = (armature.matrix_world @ tip_bone.head).to_tuple()
            imported_tail = (armature.matrix_world @ tip_bone.tail).to_tuple()

            head_error = math.dist(imported_head, _swap_yz(expected_head[frame_idx].tolist()))
            tail_error = math.dist(imported_tail, _swap_yz(expected_tail[frame_idx].tolist()))
            max_head_error = max(max_head_error, head_error)
            max_tail_error = max(max_tail_error, tail_error)

        print(f"[BVH] max tip-head error: {max_head_error:.6f}")
        print(f"[BVH] max tip-tail error: {max_tail_error:.6f}")

        tolerance = 1e-3
        assert max_head_error < tolerance, (
            f"Tip head mismatch after BVH round-trip: {max_head_error:.6f}"
        )
        assert max_tail_error < tolerance, (
            f"Tip tail mismatch after BVH round-trip: {max_tail_error:.6f}"
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("[BVH] round-trip smoke test passed")


if __name__ == "__main__":
    main()