"""
Smoke test for BVH export/import consistency.

Creates a tiny procedural skeleton animation, exports it to BVH, loads it
back through ``motion_lib.BVH.load``, and verifies the loaded pose matches
FK from the original animation.

Run:
    .venv\Scripts\python.exe tests/test_bvh_roundtrip.py

--------------------------------------------------------------------------------
踩坑记录 / 注意事项
--------------------------------------------------------------------------------

1. motion_lib.BVH.load vs Blender BVH importer
   The old version of this test used Blender (bpy) to re-import the BVH, which
   required a Blender Python environment and involved complex Z-up→Y-up
   coordinate transforms plus bone-Y rotation-delta compensation.
   The current version uses motion_lib.BVH.load(), which reads the Euler angles
   from the BVH file and converts them back to quaternions in the same
   coordinate space as the exporter wrote them. No additional transforms needed.

2. BVH Euler-quaternion 往返精度
   BVH 以 Euler 角编码旋转，exporter 写 order='xyz'。Euler→四元数→Euler 转换
   数值稳定，round-trip 误差在 1e-6 量级，1e-3 容差绰绰有余。

3. 旋转误差用"关节转角"度量而非矩阵 Frobenius 范数
   angular_error = arccos( (trace(R_imp^T @ R_exp) - 1) / 2 )
   这给出的是两个旋转之间的最小角度，单位 rad，物理含义明确，容差 1e-3 rad ≈ 0.057°。
--------------------------------------------------------------------------------
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

    # Animate the mid joint with a non-trivial rotation around X so the
    # rotation roundtrip check has something meaningful to verify.
    mid_angles = torch.linspace(0.0, math.pi * 0.4, num_frames, device=device)
    x_axis = torch.tensor([1.0, 0.0, 0.0], device=device)
    joint_rotations[:, 1] = torch.stack([
        _axis_angle_to_quat(x_axis, angle) for angle in mid_angles
    ], dim=0)

    root_translation = torch.zeros(num_frames, 3, device=device)
    root_translation[:, 1] = torch.linspace(0.0, 1.5, num_frames, device=device)

    root_angles = torch.linspace(0.0, math.pi * 0.5, num_frames, device=device)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device)
    root_rotation = torch.stack([
        _axis_angle_to_quat(z_axis, angle) for angle in root_angles
    ], dim=0)

    return joint_rotations, root_translation, root_rotation


def _expected_fk(skeleton, joint_rotations, root_translation, root_rotation):
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

    world_rotations = world_transforms[:, :, :3, :3].cpu()
    return tip_head.cpu(), tip_tail, world_rotations


def _parse_motion_width(bvh_path: str) -> tuple[int, list[int]]:
    with open(bvh_path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    motion_idx = lines.index("MOTION")
    frame_lines = lines[motion_idx + 3 :]
    widths = [len(line.split()) for line in frame_lines]
    return len(frame_lines), widths


def main() -> None:
    import numpy as np

    from Anytop.utils.exporter import AnimationExporter, _batch_forward_kinematics_np
    from Anytop.utils.rotation_numpy import quat_rotate_wxyz_np

    device = torch.device("cpu")
    skeleton = _build_minimal_skeleton(device)
    joint_rotations, root_translation, root_rotation = _make_animation(
        num_frames=6,
        num_joints=skeleton.num_joints,
        device=device,
    )
    expected_head, expected_tail, expected_world_rot = _expected_fk(
        skeleton,
        joint_rotations,
        root_translation,
        root_rotation,
    )

    exporter = AnimationExporter(skeleton, fps=30.0)

    tmp_dir = tempfile.mkdtemp(prefix="bvh_smoke_")
    try:
        output_path = os.path.join(tmp_dir, "smoke_roundtrip.bvh")
        exporter.export_bvh(
            joint_rotations,
            root_translation,
            root_rotation,
            output_path,
        )

        # ── Parse BVH header to validate frame count/width ──────────
        num_frames, widths = _parse_motion_width(output_path)
        # _export_bvh writes explicit local position channels for every joint,
        # so each joint contributes 6 channels (3 position + 3 rotation).
        expected_width = 6 * skeleton.num_joints
        assert num_frames == joint_rotations.shape[0], (
            f"Expected {joint_rotations.shape[0]} frames, got {num_frames}"
        )
        assert all(width == expected_width for width in widths), (
            f"Expected motion width {expected_width}, got {widths}"
        )

        # ── Load BVH via motion_lib ─────────────────────────────────
        # motion_lib sits inside Anytop/, so add Anytop/ to sys.path.
        anytop_root = os.path.dirname(os.path.dirname(__file__))
        if anytop_root not in sys.path:
            sys.path.insert(0, anytop_root)

        from motion_lib.BVH import load as bvh_load

        anim, loaded_names, frametime = bvh_load(output_path)
        # The exporter always writes all joints as named JOINTs — no End Site.
        # Keep the _end_site filter as a safety net for future exporter changes.
        loaded_names_filtered = [n for n in loaded_names if not n.endswith("_end_site")]
        assert loaded_names_filtered == ["root", "mid", "tip"], (
            f"Expected bone names ['root','mid','tip'], got {loaded_names_filtered}"
        )

        # ── Convert loaded animation to world-space for comparison ──
        # Only use non-end-site joints.
        J_loaded = len(loaded_names_filtered)
        loaded_rotations = anim.rotations.qs[:, :J_loaded]   # (F, J, 4) wxyz
        loaded_positions = anim.positions[:, :J_loaded]       # (F, J, 3)
        loaded_parents   = anim.parents[:J_loaded]             # (J,)

        loaded_wpos, loaded_wrot = _batch_forward_kinematics_np(
            loaded_rotations,
            loaded_positions,
            loaded_parents,
            rest_rotations=None,  # loaded quats already incorporate rest rot
        )

        # ── Compare with expected FK values ─────────────────────────
        expected_head_np  = expected_head.numpy()       # (F, 3)
        expected_tail_np  = expected_tail.numpy()       # (F, 3)

        # tip tail = head + rotation * [0, 0.1, 0]
        loaded_tail = loaded_wpos[:, 2] + quat_rotate_wxyz_np(
            loaded_wrot[:, 2],
            np.tile([0.0, 0.1, 0.0], (num_frames, 1)),
        )

        max_head_error = float(np.max(
            np.linalg.norm(loaded_wpos[:, 2] - expected_head_np, axis=-1)
        ))
        max_tail_error = float(np.max(
            np.linalg.norm(loaded_tail - expected_tail_np, axis=-1)
        ))

        # Convert loaded quaternions → rotation matrices for comparison
        expected_wrot_np  = expected_world_rot.numpy()  # (F, J, 3, 3)

        def _quat_to_mat3_wxyz(q_arr):
            w, x, y, z = (q_arr[..., i] for i in range(4))
            m = np.zeros((*q_arr.shape[:-1], 3, 3), dtype=q_arr.dtype)
            m[..., 0, 0] = 1 - 2*y*y - 2*z*z
            m[..., 0, 1] = 2*x*y - 2*z*w
            m[..., 0, 2] = 2*x*z + 2*y*w
            m[..., 1, 0] = 2*x*y + 2*z*w
            m[..., 1, 1] = 1 - 2*x*x - 2*z*z
            m[..., 1, 2] = 2*y*z - 2*x*w
            m[..., 2, 0] = 2*x*z - 2*y*w
            m[..., 2, 1] = 2*y*z + 2*x*w
            m[..., 2, 2] = 1 - 2*x*x - 2*y*y
            return m

        loaded_wrot_mats = _quat_to_mat3_wxyz(loaded_wrot)  # (F, J, 3, 3)

        max_rot_error = 0.0
        for frame_idx in range(num_frames):
            for joint_idx in range(skeleton.num_joints):
                R_imp = loaded_wrot_mats[frame_idx, joint_idx]
                R_exp = expected_wrot_np[frame_idx, joint_idx]
                rel = R_imp.T @ R_exp
                cos_theta = (np.trace(rel) - 1.0) * 0.5
                cos_theta = max(-1.0, min(1.0, cos_theta))
                angle_error = math.acos(cos_theta)
                max_rot_error = max(max_rot_error, angle_error)

        print(f"[BVH] max tip-head error: {max_head_error:.6f}")
        print(f"[BVH] max tip-tail error: {max_tail_error:.6f}")
        print(f"[BVH] max joint rotation error (rad): {max_rot_error:.6f}")

        tolerance = 1e-3
        assert max_head_error < tolerance, (
            f"Tip head mismatch after BVH round-trip: {max_head_error:.6f}"
        )
        assert max_tail_error < tolerance, (
            f"Tip tail mismatch after BVH round-trip: {max_tail_error:.6f}"
        )
        rot_tolerance = 1e-3
        assert max_rot_error < rot_tolerance, (
            f"Joint rotation mismatch after BVH round-trip: {max_rot_error:.6f} rad"
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("[BVH] round-trip smoke test passed")


# ── pytest entry point ───────────────────────────────────────────────────────

def test_bvh_roundtrip():
    """Pytest-compatible entry point for the BVH round-trip smoke test."""
    main()


if __name__ == "__main__":
    main()