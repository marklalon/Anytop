"""The HML skinned rig and cond-only rig must play the same recovered pose."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

bpy = pytest.importorskip("bpy")

from data_loaders.truebones.truebones_utils.cond_schema import load_cond
from tools.restore_glb_from_npy import restore_glb


ANYTOP = Path(__file__).resolve().parents[1]
PROCESSED = ANYTOP / "dataset/truebones/zoo/truebones_processed"
COND_NPY = PROCESSED / "cond.npy"
MOTION_NPY = PROCESSED / "motions/Horse_Attack.npy"
TPOSE_MESH = ANYTOP / "dataset/truebones/zoo/Truebone_Z-OO/Horse/HorseALL-TPOSE.glb"


def _glb_joint_world_poses(
    path: Path, names: list[str], frames: int, *, expect_skin: bool
) -> tuple[np.ndarray, np.ndarray]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(path))
    armature = next(obj for obj in bpy.data.objects if obj.type == "ARMATURE")
    assert all(name in armature.pose.bones for name in names)
    skinned_meshes = [
        obj for obj in bpy.data.objects
        if obj.type == "MESH" and any(
            mod.type == "ARMATURE" and mod.object == armature for mod in obj.modifiers
        )
    ]
    assert bool(skinned_meshes) == expect_skin

    # Blender Z-up -> glTF/HML Y-up, matching export_yup=True.
    axis = np.array(((1, 0, 0), (0, 0, 1), (0, -1, 0)), dtype=np.float64)
    positions = np.empty((frames, len(names), 3), dtype=np.float64)
    rotations = np.empty((frames, len(names), 3, 3), dtype=np.float64)
    for frame in range(frames):
        bpy.context.scene.frame_set(frame)
        for joint, name in enumerate(names):
            transform = armature.matrix_world @ armature.pose.bones[name].matrix
            positions[frame, joint] = axis @ np.asarray(transform.translation)
            rest_basis = np.asarray(transform.to_3x3().normalized())
            rotations[frame, joint] = axis @ rest_basis @ axis.T
    return positions, rotations


@pytest.mark.parametrize("fullbody_ik", [False, True])
def test_hml_skinned_matches_skeleton_only(tmp_path: Path, fullbody_ik: bool) -> None:
    if not all(path.is_file() for path in (COND_NPY, MOTION_NPY, TPOSE_MESH)):
        pytest.skip("Horse test motion, cond, or T-pose mesh is unavailable")

    cond = load_cond(str(COND_NPY))["truebones/zoo/Horse"]
    assert not np.allclose(cond["orientation_quat"], (1, 0, 0, 0))
    assert abs(float(cond["scale_factor"]) - 1.0) > 0.1
    raw_names = list(cond["joints_names"])
    canonical_names = list(cond["canonical_bvh_joint_names"])

    features = np.load(MOTION_NPY)[:3]
    npy_path = tmp_path / "Horse_Attack.npy"
    np.save(npy_path, features)
    # tmp_path stands in for a generation directory, which carries the sidecar.
    shutil.copy2(COND_NPY.parent / "joint_parts.jsonl", tmp_path / "joint_parts.jsonl")
    skeleton_path = tmp_path / "skeleton.glb"
    tpose_skeleton_path = tmp_path / "tpose_skeleton.glb"
    skinned_path = tmp_path / "skinned.glb"
    common = dict(
        npy_path=str(npy_path),
        cond_npy=str(COND_NPY),
        object_type="Horse",
        restore_space="hml",
        fullbody_ik=fullbody_ik,
    )
    restore_glb(output_glb=str(skeleton_path), skeleton_only=True, **common)
    restore_glb(
        output_glb=str(tpose_skeleton_path),
        tpose_mesh=str(TPOSE_MESH),
        skeleton_only=True,
        **common,
    )
    restore_glb(output_glb=str(skinned_path), tpose_mesh=str(TPOSE_MESH), **common)

    skinned_pos, skinned_rot = _glb_joint_world_poses(
        skinned_path, canonical_names, len(features), expect_skin=True
    )
    for path in (skeleton_path, tpose_skeleton_path):
        skeleton_pos, skeleton_rot = _glb_joint_world_poses(
            path, raw_names, len(features), expect_skin=False
        )
        assert np.max(np.linalg.norm(skeleton_pos - skinned_pos, axis=-1)) < 1e-4

        # The imported mesh has different static bone-roll axes. Compare
        # changes in world rotation from frame 0, which describe the motion.
        skeleton_delta = skeleton_rot @ np.swapaxes(skeleton_rot[0:1], -1, -2)
        skinned_delta = skinned_rot @ np.swapaxes(skinned_rot[0:1], -1, -2)
        relative = skeleton_delta @ np.swapaxes(skinned_delta, -1, -2)
        angles = np.degrees(
            np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1))
        )
        assert np.max(angles) < 0.2
