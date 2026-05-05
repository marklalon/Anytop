"""
Shared helpers for FBX/GLB/NPY/BVH roundtrip tests.

Extracted from test_fbx_glb_npy_roundtrip.py to avoid duplication
between NPY and BVH roundtrip tests.
"""
from __future__ import annotations

import contextlib
import io
import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

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


def _build_skeleton(
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

def _load_fbx_skeleton_metadata(
    fbx_path: str,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    from ..motion_lib.FBX import _load_fbx_scene, _extract_armature_skeleton_data
    armature = _load_fbx_scene(fbx_path)
    return _extract_armature_skeleton_data(armature)


# ── GLB loading helpers ──────────────────────────────────────────────────────

def _load_glb_scene(glb_path: str):
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        bpy.ops.import_scene.gltf(filepath=glb_path)

    for obj in list(bpy.data.objects):
        if obj.type in {"LIGHT", "CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)

    armature = next((obj for obj in bpy.data.objects if obj.type == "ARMATURE"), None)
    if armature is None:
        raise RuntimeError(f"No armature found in {glb_path}")
    return armature


def _load_glb_skeleton_metadata(
    glb_path: str,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Load a GLB file and extract bone names, parents, offsets, and rest rotations."""
    from ..motion_lib.FBX import _extract_armature_skeleton_data
    armature = _load_glb_scene(glb_path)
    return _extract_armature_skeleton_data(armature)


# ── Animation → GLB export ──────────────────────────────────────────────────

def _export_animation_to_glb(
    animation,
    skeleton,
    output_glb: str,
    mesh_path: str,
    fps: float,
) -> None:
    from .exporter import AnimationExporter

    exporter = AnimationExporter(skeleton, fps=fps)
    joint_rotations = torch.from_numpy(animation.rotations.qs.astype(np.float32))
    root_translation = torch.from_numpy(animation.positions[:, 0, :].astype(np.float32))
    root_rotation = torch.zeros((animation.shape[0], 4), dtype=torch.float32)
    root_rotation[:, 0] = 1.0
    bone_translations_np = animation.positions.astype(np.float32).copy()
    bone_translations_np[:, 0, :] = 0.0
    bone_translations = torch.from_numpy(bone_translations_np)
    exporter.export(
        joint_rotations,
        root_translation,
        root_rotation,
        output_glb,
        mesh_path=mesh_path,
        bone_translations=bone_translations,
    )



