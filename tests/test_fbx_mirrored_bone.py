"""A mirrored bone must keep its chain where Blender puts it, in rest and in pose.

glTF rigs exported from 3ds Max carry a negative node scale on mirrored bones
(the left legs of the Taobao insects). Blender edit bones cannot store it, so the
importer leaves the sign in every frame's pose basis. Reading the rest from the
edit bones and the pose through ``to_quaternion`` then bends the whole mirrored
chain the wrong way.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

bpy = pytest.importorskip("bpy")

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ANYTOP_ROOT = os.path.dirname(_TESTS_DIR)
_REPO_ROOT = os.path.dirname(_ANYTOP_ROOT)

for _path in [_REPO_ROOT, _ANYTOP_ROOT]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from motion_lib import FBX
from motion_lib.Animation import Animation, positions_global
from motion_lib.Quaternions import Quaternions


def _build_mirrored_chain():
    """Root -> Mirror -> Tip, with an off-axis tip and a mirrored middle bone."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    armature_data = bpy.data.armatures.new("TestArmature")
    armature = bpy.data.objects.new("TestArmature", armature_data)
    bpy.context.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature

    bpy.ops.object.mode_set(mode="EDIT")
    root = armature_data.edit_bones.new("Root")
    root.head = (0.0, 0.0, 0.0)
    root.tail = (0.0, 0.0, 1.0)
    mirror = armature_data.edit_bones.new("Mirror")
    mirror.head = (0.0, 0.0, 1.0)
    mirror.tail = (1.0, 0.0, 1.0)
    mirror.parent = root
    tip = armature_data.edit_bones.new("Tip")
    tip.head = (2.0, 0.5, 1.3)
    tip.tail = (3.0, 0.5, 1.3)
    tip.parent = mirror
    bpy.ops.object.mode_set(mode="OBJECT")

    armature.rotation_mode = "XYZ"
    armature.rotation_euler = (np.pi / 2.0, 0.0, 0.0)
    armature.pose.bones["Mirror"].scale = (-1.0, -1.0, -1.0)
    bpy.context.view_layer.update()
    return armature


def test_mirrored_chain_matches_blender_in_rest_and_pose(monkeypatch):
    armature = _build_mirrored_chain()
    monkeypatch.setattr(FBX, "_load_scene", lambda _path: armature)

    anim, names, _fps = FBX._scene_to_animation("unused.glb", collapse_root=False)

    # The armature object's Rx(90) is the Z-up -> Y-up turn itself, so the
    # loader applies no correction and the heads compare in armature space.
    assert FBX._armature_yup_correction(armature) is None
    heads = np.array([list(armature.pose.bones[name].head) for name in names])
    np.testing.assert_allclose(positions_global(anim)[0], heads, atol=1e-5)

    # The mirror is the rest state of this rig, so the rest pose sits exactly
    # where the (never animated) pose does.
    rest = Animation(
        Quaternions(anim.orients.qs[None].copy()),
        anim.offsets[None].copy(),
        anim.orients,
        anim.offsets,
        anim.parents,
    )
    np.testing.assert_allclose(positions_global(rest)[0], heads, atol=1e-5)
    assert np.all(np.abs(np.linalg.norm(anim.rotations.qs, axis=-1) - 1.0) < 1e-6)


def _animate_tip(armature):
    """Key a swing on the tip below the mirror, and the mirror's own scale."""
    tip = armature.pose.bones["Tip"]
    tip.rotation_mode = "QUATERNION"
    mirror = armature.pose.bones["Mirror"]
    for frame, angle in [(1, 0.0), (10, 0.7)]:
        tip.rotation_quaternion = (np.cos(angle / 2.0), 0.0, np.sin(angle / 2.0), 0.0)
        tip.keyframe_insert("rotation_quaternion", frame=frame)
        mirror.keyframe_insert("scale", frame=frame)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, 10


def test_export_onto_mirrored_rig_round_trips(tmp_path):
    from utils.exporter import AnimationExporter, animation_to_exporter_inputs
    from utils.roundtrip_common import build_skeleton, load_fbx_skeleton_metadata

    source = str(tmp_path / "mirrored.glb")
    _animate_tip(_build_mirrored_chain())
    bpy.ops.export_scene.gltf(filepath=source, export_format="GLB")

    anim, names, fps = FBX._scene_to_animation(source, collapse_root=False)
    meta_names, parents, offsets, rest_rotations = load_fbx_skeleton_metadata(source)
    assert meta_names == names
    # The rest a separate reader gets is the rest the loaded motion is built on.
    np.testing.assert_allclose(offsets, anim.offsets, atol=1e-6)
    np.testing.assert_allclose(
        np.abs(np.sum(rest_rotations * anim.orients.qs, axis=-1)), 1.0, atol=1e-6
    )

    skeleton = build_skeleton(names, offsets, parents, rest_rotations)
    exported = str(tmp_path / "exported.glb")
    joint_rotations, root_translation, root_rotation, bone_translations = (
        animation_to_exporter_inputs(anim, skeleton)
    )
    AnimationExporter(skeleton, fps=fps).export_glb(
        joint_rotations,
        root_translation,
        root_rotation,
        exported,
        mesh_path=source,
        bone_translations=bone_translations,
    )

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=exported)
    armature = next(obj for obj in bpy.data.objects if obj.type == "ARMATURE")
    assert armature.pose.bones["Mirror"].scale[0] < 0.0

    reloaded, reloaded_names, _ = FBX._scene_to_animation(exported, collapse_root=False)
    assert reloaded_names == names
    np.testing.assert_allclose(
        positions_global(reloaded), positions_global(anim), atol=1e-4
    )
