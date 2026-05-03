"""
Smoke test for BVH export/import consistency.

Creates a tiny procedural skeleton animation, exports it to BVH, imports it
back through Blender, and verifies the imported pose matches FK from the
original animation.

Run:
    .venv\Scripts\python.exe tests/test_bvh_roundtrip.py

--------------------------------------------------------------------------------
踩坑记录 / 注意事项
--------------------------------------------------------------------------------

1. Blender BVH 导入是 R_x(+90°) 而非 Y/Z 简单互换
   BVH 文件约定 Z-up，Blender world 是 Y-up。Blender BVH importer 做的变换是：
       (x, y, z)_bvh → (x, -z, y)_blender_world
   对应 3×3 矩阵 S = [[1,0,0],[0,0,-1],[0,1,0]]，det=+1（正交旋转，保持手性）。
   如果只做 Y/Z 互换 (x,y,z)→(x,z,y) 会引入反射（det=-1），坐标轴 Z 分量会
   符号出错。原测试因为 Y-up Z 分量恒为 0（只做 Y 方向平移 + Z 轴旋转）而蒙混
   过关；一旦加入 mid 关节的 X 轴旋转，Z 分量非零，head/tail 误差立刻飙到 ~2.0。
   正确做法：读取 Blender 坐标后用 S^T 反变换回 Y-up，再与 FK 直接比较。

2. BVH Euler-quaternion 往返精度
   BVH 以 Euler 角编码旋转，exporter 写 order='xyz'（Blender 默认解读为 Z·Y·X
   外旋顺序）。Euler→四元数→Euler 转换数值稳定，round-trip 误差在 1e-6 量级，
   1e-3 容差绰绰有余。

3. rotation 比较必须用 rest-pose-relative delta
   Blender bone 的 local Y 轴沿 head→tail 方向（bone-Y 约定），导致 rest-pose
   下 pose_bone.matrix 不是单位矩阵（存在 bone roll）。若直接比较绝对 world
   rotation 矩阵，rest-pose 的 roll 差异会产生系统偏差。
   解决：先在 frame 1（rest pose）采集 blender_rest_rot，然后用 delta = R @ R0^T
   比较"相对 rest-pose 的变化量"——delta 只反映实际运动，bone-Y roll 在两侧同时
   出现后消掉。

4. Blender 帧号从 1 开始，FK 帧索引从 0 开始
   BVH 导入后 Blender frame_start=1，frame 1 对应 BVH 第一帧（即 FK frame 0）。
   循环中务必用 scene.frame_set(frame_idx + 1)，否则会比较错帧。

5. 动画帧 0 必须是 rest-pose，rotation delta 基线才有效
   本测试的 linspace 从 0 开始（root 平移/旋转、mid 旋转均为 0），保证 FK frame 0
   == rest pose。如果修改动画使 frame 0 不是 rest pose，需要同步修改 rest-pose
   基线采集逻辑。

6. tip 骨骼为零长度，Blender BVH 导入会打印 "zero length node found: tip"
   这是 BVH END Site 节点（子节点 offset 不为 0 但没有旋转通道）的警告，不影响
   导入正确性；测试里比较的是 tip bone 的 head（位置）和 tail（通过 world_transforms
   计算出来的虚拟延伸点），两者都正常。

7. 旋转误差用"关节转角"度量而非矩阵 Frobenius 范数
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
    import bpy

    from Anytop.utils.exporter import AnimationExporter

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

        # Blender's BVH importer treats the file as Z-up and rotates the
        # whole armature into Blender's Y-up world via R_x(+90°):
        #   (x, y, z)_bvh → (x, -z, y)_blender
        # Our internal FK uses Y-up directly, so we transform Blender's
        # imported coordinates back into FK's frame with S_inv = S^T and
        # compare against FK output directly.
        S = torch.tensor([
            [1.0, 0.0,  0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0,  0.0],
        ], dtype=expected_world_rot.dtype)
        S_inv = S.T

        def _to_fk_frame(p):
            v = torch.tensor([p[0], p[1], p[2]], dtype=expected_world_rot.dtype)
            return (S_inv @ v).tolist()

        # Bone names in topological (joint) order.
        bone_names = [b.name for b in skeleton.bones]
        pose_bones = [armature.pose.bones.get(name) for name in bone_names]
        assert all(pb is not None for pb in pose_bones), (
            f"Imported armature missing pose bones: {bone_names}"
        )

        # Capture rest-pose world rotations (Blender frame 1 = rest pose
        # since FK frame 0 is identity); used as the baseline so Blender's
        # bone-Y axis convention cancels out of the rotation comparison.
        scene.frame_set(1)

        def _blender_world_rot(pb):
            m3 = (armature.matrix_world @ pb.matrix).to_3x3()
            return torch.tensor(
                [[m3[i][j] for j in range(3)] for i in range(3)],
                dtype=expected_world_rot.dtype,
            )

        blender_rest_rot = [_blender_world_rot(pb) for pb in pose_bones]
        fk_rest_rot = expected_world_rot[0]  # [J, 3, 3]

        max_head_error = 0.0
        max_tail_error = 0.0
        max_rot_error = 0.0
        for frame_idx in range(num_frames):
            scene.frame_set(frame_idx + 1)

            imported_head = (armature.matrix_world @ tip_bone.head).to_tuple()
            imported_tail = (armature.matrix_world @ tip_bone.tail).to_tuple()

            head_error = math.dist(_to_fk_frame(imported_head), expected_head[frame_idx].tolist())
            tail_error = math.dist(_to_fk_frame(imported_tail), expected_tail[frame_idx].tolist())
            max_head_error = max(max_head_error, head_error)
            max_tail_error = max(max_tail_error, tail_error)

            for joint_idx, pb in enumerate(pose_bones):
                R_imp = _blender_world_rot(pb)
                # Express both deltas in the same world frame (Blender Y-up).
                # Rest-relative delta absorbs Blender's bone-Y convention.
                delta_imp = R_imp @ blender_rest_rot[joint_idx].T
                delta_fk_yup = expected_world_rot[frame_idx, joint_idx] @ fk_rest_rot[joint_idx].T
                delta_fk_blender = S @ delta_fk_yup @ S_inv

                rel = delta_imp.T @ delta_fk_blender
                cos_theta = (rel.diagonal().sum() - 1.0) * 0.5
                cos_theta = max(-1.0, min(1.0, float(cos_theta)))
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


if __name__ == "__main__":
    main()