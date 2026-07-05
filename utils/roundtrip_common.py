"""
Shared helpers for FBX/GLB/NPY/BVH roundtrip tests.

Extracted from test_fbx_glb_npy_roundtrip.py to avoid duplication
between NPY and BVH roundtrip tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


# ── Skeleton helpers for exporter ────────────────────────────────────────────

@dataclass
class _SimpleBone:
    id: int
    name: str
    parent_id: Optional[int]
    rest_offset: torch.Tensor
    rest_rotation: torch.Tensor


class _SimpleSkeleton:
    """Minimal skeleton that matches the API expected by AnimationExporter."""

    def __init__(self, bones: list[_SimpleBone]):
        self.bones = bones
        self.rest_offsets = torch.stack([bone.rest_offset for bone in bones], dim=0)
        self._build_depth_levels()

    @property
    def num_joints(self) -> int:
        return len(self.bones)

    def _build_depth_levels(self):
        joint_count = len(self.bones)
        parents = torch.tensor(
            [bone.parent_id if bone.parent_id is not None else -1 for bone in self.bones],
            dtype=torch.long,
        )
        depths = torch.zeros(joint_count, dtype=torch.long)
        for joint_idx in range(1, joint_count):
            parent_idx = parents[joint_idx].item()
            if parent_idx >= 0:
                depths[joint_idx] = depths[parent_idx] + 1

        max_depth = depths.max().item()
        device = self.rest_offsets.device
        self.depth_levels: list[tuple] = []
        self.root_bone_ids: list[int] = []

        for depth in range(max_depth + 1):
            ids = [bone.id for bone in self.bones if depths[bone.id].item() == depth]
            if depth == 0:
                self.root_bone_ids = ids
                parent_ids = [-1] * len(ids)
            else:
                parent_ids = [self.bones[bone_id].parent_id for bone_id in ids]
            self.depth_levels.append((
                torch.tensor(ids, dtype=torch.long, device=device),
                torch.tensor(parent_ids, dtype=torch.long, device=device),
            ))


def build_skeleton(
    bone_names, offsets, parents, rest_rotations=None, device=None,
) -> _SimpleSkeleton:
    if device is None:
        device = torch.device("cpu")

    bones = []
    for joint_idx, bone_name in enumerate(bone_names):
        parent_idx = parents[joint_idx]
        if rest_rotations is None:
            rest_rotation = [1.0, 0.0, 0.0, 0.0]
        else:
            rest_rotation = rest_rotations[joint_idx]
        bones.append(_SimpleBone(
            id=joint_idx,
            name=bone_name,
            parent_id=None if parent_idx < 0 else int(parent_idx),
            rest_offset=torch.tensor(offsets[joint_idx], dtype=torch.float32, device=device),
            rest_rotation=torch.tensor(rest_rotation, dtype=torch.float32, device=device),
        ))
    return _SimpleSkeleton(bones)


# ── FBX loading helpers ──────────────────────────────────────────────────────

def load_fbx_skeleton_metadata(
    fbx_path: str,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    from ..motion_lib.FBX import load_fbx_scene, extract_armature_skeleton_data
    armature = load_fbx_scene(fbx_path)
    return extract_armature_skeleton_data(armature)


def load_fbx_armature_object_scale(fbx_path: str) -> float:
    """Return the uniform world object-scale of the T-pose armature.

    ``extract_armature_skeleton_data`` reads armature-LOCAL bone data and
    deliberately drops the armature object scale (the ``0.01`` Truebones
    centimetre wrapper); the dataset pipeline re-introduces that factor via the
    per-character ``scale_factor``. A skeleton-only restore, however, rebuilds a
    fresh armature at object scale ``1.0`` directly from those armature-local
    offsets, so without this factor the exported skeleton comes out
    ``1 / object_scale`` (e.g. 100×) larger than the source mesh. Callers use
    this to rescale the skeleton-only export back into the mesh's world scale.

    Returns ``1.0`` outside a live Blender session or when the armature has no
    object-level scale.
    """
    from ..motion_lib.FBX import load_fbx_scene
    armature = load_fbx_scene(fbx_path)
    if not hasattr(armature, "matrix_world"):
        return 1.0
    sx, sy, sz = armature.matrix_world.to_scale()
    return float((abs(sx) + abs(sy) + abs(sz)) / 3.0)


# ── Identity rest rotations ────────────────────────────────────────────────

def identity_rest_rotations(joint_count: int) -> np.ndarray:
    """Return (J, 4) identity quaternion array."""
    rest_rotations = np.zeros((joint_count, 4), dtype=np.float32)
    rest_rotations[:, 0] = 1.0
    return rest_rotations
