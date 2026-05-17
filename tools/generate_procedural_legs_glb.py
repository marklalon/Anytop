"""Generate a procedural two-leg skeleton and export it as a skeleton-only GLB.

The generated hierarchy is intentionally minimal and follows the repo's
quadruped naming style:

    Hips
      -> Pelvis
        -> Spine
                    -> LeftThigh / RightThigh
                        -> LeftCalf / RightCalf
                            -> LeftHorseLink / RightHorseLink
                                -> LeftFoot / RightFoot
                                    -> LeftToe1 / LeftToe2 / LeftToe3
                                    -> RightToe1 / RightToe2 / RightToe3

Usage:
    .venv\Scripts\python.exe Anytop\tools\generate_procedural_legs_glb.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch


ANYTOP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(ANYTOP_ROOT)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if ANYTOP_ROOT not in sys.path:
    sys.path.insert(1, ANYTOP_ROOT)

from Anytop.kinematics.skeleton import Bone, Skeleton
from Anytop.utils.exporter import AnimationExporter
from Anytop.utils.rotation_numpy import matrix_to_quat_wxyz_np


TPOSE_FPS = 30.0
TPOSE_FRAME_COUNT = 1
BONE_ORDER = [
    "Hips",
    "Pelvis",
    "Spine",
    "LeftThigh",
    "LeftCalf",
    "LeftHorseLink",
    "LeftFoot",
    "LeftToe2",
    "LeftToe1",
    "LeftToe3",
    "RightThigh",
    "RightCalf",
    "RightHorseLink",
    "RightFoot",
    "RightToe2",
    "RightToe1",
    "RightToe3",
]
NAME_TO_ID = {bone_name: bone_id for bone_id, bone_name in enumerate(BONE_ORDER)}
PARENT_NAMES = {
    "Hips": None,
    "Pelvis": "Hips",
    "Spine": "Pelvis",
    "LeftThigh": "Spine",
    "LeftCalf": "LeftThigh",
    "LeftHorseLink": "LeftCalf",
    "LeftFoot": "LeftHorseLink",
    "LeftToe2": "LeftFoot",
    "LeftToe1": "LeftFoot",
    "LeftToe3": "LeftFoot",
    "RightThigh": "Spine",
    "RightCalf": "RightThigh",
    "RightHorseLink": "RightCalf",
    "RightFoot": "RightHorseLink",
    "RightToe2": "RightFoot",
    "RightToe1": "RightFoot",
    "RightToe3": "RightFoot",
}
WORLD_HEADS = {
    "Hips": (0.00, 0.76, 0.00),
    "Pelvis": (0.00, 0.68, 0.00),
    "Spine": (0.00, 0.66, 0.10),
    "LeftThigh": (-0.08, 0.54, 0.10),
    "LeftCalf": (-0.08, 0.30, 0.12),
    "LeftHorseLink": (-0.08, 0.16, 0.17),
    "LeftFoot": (-0.08, 0.04, 0.24),
    "LeftToe2": (-0.08, 0.00, 0.33),
    "LeftToe1": (-0.11, 0.00, 0.32),
    "LeftToe3": (-0.05, 0.00, 0.32),
    "RightThigh": (0.08, 0.54, 0.10),
    "RightCalf": (0.08, 0.30, 0.12),
    "RightHorseLink": (0.08, 0.16, 0.17),
    "RightFoot": (0.08, 0.04, 0.24),
    "RightToe2": (0.08, 0.00, 0.33),
    "RightToe1": (0.11, 0.00, 0.32),
    "RightToe3": (0.05, 0.00, 0.32),
}


def _quat_identity() -> torch.Tensor:
    return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)


def _normalize(vector: np.ndarray, fallback: tuple[float, float, float] | np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm > 1e-8:
        return vector / norm

    fallback_vector = np.asarray(fallback, dtype=np.float64)
    fallback_norm = float(np.linalg.norm(fallback_vector))
    if fallback_norm <= 1e-8:
        raise ValueError("Fallback vector must be non-zero")
    return fallback_vector / fallback_norm


def _build_world_rotation(aim_direction: np.ndarray) -> np.ndarray:
    """Build a right-handed world rotation whose local +Y follows *aim_direction*."""
    y_axis = _normalize(aim_direction, fallback=(0.0, -1.0, 0.0))

    z_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    z_axis = z_hint - y_axis * float(np.dot(z_hint, y_axis))
    if np.linalg.norm(z_axis) <= 1e-8:
        x_hint = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        z_axis = x_hint - y_axis * float(np.dot(x_hint, y_axis))
    z_axis = _normalize(z_axis, fallback=(0.0, 0.0, 1.0))

    x_axis = _normalize(np.cross(y_axis, z_axis), fallback=(1.0, 0.0, 0.0))
    z_axis = _normalize(np.cross(x_axis, y_axis), fallback=(0.0, 0.0, 1.0))
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def _build_children_map() -> dict[str, list[str]]:
    children_map = {name: [] for name in BONE_ORDER}
    for bone_name, parent_name in PARENT_NAMES.items():
        if parent_name is not None:
            children_map[parent_name].append(bone_name)
    return children_map


def _scaled_world_heads(scale: float) -> dict[str, np.ndarray]:
    return {
        bone_name: np.asarray(head_position, dtype=np.float64) * scale
        for bone_name, head_position in WORLD_HEADS.items()
    }


def _solve_local_rest_pose(
    world_heads: dict[str, np.ndarray],
) -> tuple[dict[str, tuple[float, float, float]], dict[str, torch.Tensor]]:
    children_map = _build_children_map()
    world_rotations: dict[str, np.ndarray] = {}
    local_offsets: dict[str, tuple[float, float, float]] = {}
    local_rotations: dict[str, torch.Tensor] = {}

    for bone_name in BONE_ORDER:
        parent_name = PARENT_NAMES[bone_name]
        child_names = children_map[bone_name]

        if child_names:
            child_directions = [
                _normalize(world_heads[child_name] - world_heads[bone_name], fallback=(0.0, 1.0, 0.0))
                for child_name in child_names
            ]
            aim_direction = np.mean(child_directions, axis=0)
        elif parent_name is not None:
            # Leaves continue along their incoming segment.
            aim_direction = world_heads[bone_name] - world_heads[parent_name]
        else:
            aim_direction = np.array([0.0, -1.0, 0.0], dtype=np.float64)

        world_rotation = _build_world_rotation(aim_direction)

        if parent_name is None:
            local_offset = world_heads[bone_name]
            local_rotation_matrix = world_rotation
        else:
            parent_world_rotation = world_rotations[parent_name]
            local_offset = parent_world_rotation.T @ (world_heads[bone_name] - world_heads[parent_name])
            local_rotation_matrix = parent_world_rotation.T @ world_rotation

        local_offsets[bone_name] = tuple(float(value) for value in local_offset)
        local_rotations[bone_name] = torch.tensor(
            matrix_to_quat_wxyz_np(local_rotation_matrix),
            dtype=torch.float32,
        )
        world_rotations[bone_name] = world_rotation

    return local_offsets, local_rotations


def _bone(
    bone_id: int,
    name: str,
    parent_id: int | None,
    offset: tuple[float, float, float],
    rotation: torch.Tensor | None = None,
) -> Bone:
    return Bone(
        id=bone_id,
        name=name,
        parent_id=parent_id,
        rest_offset=torch.tensor(offset, dtype=torch.float32),
        rest_rotation=_quat_identity() if rotation is None else rotation,
    )


def build_two_leg_skeleton(scale: float) -> Skeleton:
    """Create a static quadruped-style two-leg hierarchy from a world-space layout."""
    local_offsets, local_rotations = _solve_local_rest_pose(_scaled_world_heads(scale))
    bones = [
        _bone(
            bone_id,
            bone_name,
            None if PARENT_NAMES[bone_name] is None else NAME_TO_ID[PARENT_NAMES[bone_name]],
            local_offsets[bone_name],
            rotation=local_rotations[bone_name],
        )
        for bone_id, bone_name in enumerate(BONE_ORDER)
    ]
    return Skeleton(bones)


def build_single_leg_skeleton(scale: float) -> Skeleton:
    """Backward-compatible alias for callers still using the old name."""
    return build_two_leg_skeleton(scale)


def build_tpose_tensors(num_joints: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    joint_rotations = torch.zeros((TPOSE_FRAME_COUNT, num_joints, 4), dtype=torch.float32)
    joint_rotations[..., 0] = 1.0

    root_translation = torch.zeros((TPOSE_FRAME_COUNT, 3), dtype=torch.float32)
    root_rotation = torch.zeros((TPOSE_FRAME_COUNT, 4), dtype=torch.float32)
    root_rotation[..., 0] = 1.0
    return joint_rotations, root_translation, root_rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a procedural one-leg skeleton GLB.",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(
            REPO_ROOT,
            "Anytop",
            "outputs",
            "procedural_bones",
            "procedural_legs.glb",
        ),
        help="Output GLB path.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Uniform scale multiplier applied to all rest offsets.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.scale <= 0.0:
        raise ValueError("--scale must be positive")

    output_path = os.path.abspath(args.out)
    skeleton = build_two_leg_skeleton(scale=args.scale)
    joint_rotations, root_translation, root_rotation = build_tpose_tensors(skeleton.num_joints)

    exporter = AnimationExporter(skeleton=skeleton, fps=TPOSE_FPS)
    exporter.export_glb(
        joint_rotations=joint_rotations,
        root_translation=root_translation,
        root_rotation=root_rotation,
        output_path=output_path,
        mesh_path=None,
    )

    print(f"Exported procedural leg skeleton to: {output_path}")
    print("Bones:")
    for bone in skeleton.bones:
        parent_name = "<root>" if bone.parent_id is None else skeleton.bones[bone.parent_id].name
        print(f"  {bone.name:<14} parent={parent_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())