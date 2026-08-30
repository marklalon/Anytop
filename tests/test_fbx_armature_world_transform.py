"""The armature OBJECT transform must reach the root in armature-local units.

Unity-authored GLB rigs park the character's standing height on the armature
object (``IAC_Caveman`` sits at ``(0, 0, 0.572)`` with a ``0.01`` object scale)
and keep every bone matrix at the origin.  Both ``bone.matrix_local`` and
``pose_bone.matrix`` are armature-OBJECT space, so that height reaches neither
of them on its own, and it is expressed in world units while they are not.
Getting either half wrong sinks the whole character below the floor.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

bpy = pytest.importorskip("bpy")
from mathutils import Matrix

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ANYTOP_ROOT = os.path.dirname(_TESTS_DIR)
_REPO_ROOT = os.path.dirname(_ANYTOP_ROOT)

for _path in [_REPO_ROOT, _ANYTOP_ROOT]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from motion_lib import FBX


class _FakeArmature:
    """Just enough of a Blender object for ``_armature_yup_correction``."""

    def __init__(self, matrix_world: Matrix) -> None:
        self.matrix_world = matrix_world.copy()


def _unity_style_world(height: float, scale: float) -> Matrix:
    """``T @ R @ S`` the way Blender's glTF importer parks a Unity rig."""
    return (
        Matrix.Translation((0.0, 0.0, height))
        @ Matrix.Rotation(math.radians(-90.0), 4, "Z")
        @ Matrix.Diagonal((scale, scale, scale, 1.0))
    )


def test_world_translation_is_converted_into_armature_local_units():
    # 0.572 m of standing height on an object scaled to 0.01 is 57.2 units in
    # the bone matrices' own frame; the preprocessing scale_factor scales the
    # whole skeleton back to world proportions afterwards.
    armature = _FakeArmature(_unity_style_world(0.572, 0.01))

    correction = FBX._armature_yup_correction(armature)

    assert correction is not None
    # Rx(-90) maps the Blender Z-up height onto AnyTop's Y-up frame.
    assert correction.translation.x == pytest.approx(0.0, abs=1e-5)
    assert correction.translation.y == pytest.approx(57.2, rel=1e-6)
    assert correction.translation.z == pytest.approx(0.0, abs=1e-5)


def test_unit_scaled_world_translation_passes_through_unchanged():
    armature = _FakeArmature(_unity_style_world(0.501, 1.0))

    correction = FBX._armature_yup_correction(armature)

    assert correction is not None
    assert correction.translation.y == pytest.approx(0.501, rel=1e-6)


def test_zup_parked_world_transform_needs_no_correction():
    # The Truebones path: the importer parks the Z-up -> Y-up Rx(+90) on the
    # armature object and leaves it at the origin, so the correction cancels.
    zup_world = Matrix.Rotation(math.radians(90.0), 4, "X")
    assert FBX._armature_yup_correction(_FakeArmature(zup_world)) is None


def _build_armature(height: float, scale: float):
    """A two-bone armature whose bones sit at the origin of armature space."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    armature_data = bpy.data.armatures.new("TestArmature")
    armature = bpy.data.objects.new("TestArmature", armature_data)
    bpy.context.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature

    bpy.ops.object.mode_set(mode="EDIT")
    root = armature_data.edit_bones.new("Root")
    root.head = (0.0, 0.0, 0.0)
    root.tail = (0.0, 0.0, 10.0)
    child = armature_data.edit_bones.new("Child")
    child.head = (0.0, 0.0, 10.0)
    child.tail = (0.0, 0.0, 20.0)
    child.parent = root
    bpy.ops.object.mode_set(mode="OBJECT")

    armature.location = (0.0, 0.0, height)
    armature.scale = (scale, scale, scale)
    bpy.context.view_layer.update()
    return armature


def test_rest_offsets_and_animated_frames_share_the_object_height(monkeypatch):
    height, scale = 0.572, 0.01
    armature = _build_armature(height, scale)
    monkeypatch.setattr(FBX, "_load_scene", lambda _path: armature)

    anim, names, _fps = FBX._scene_to_animation("unused.glb", collapse_root=False)

    assert names[0] == "Root"
    expected_root = np.array([0.0, height / scale, 0.0])
    # The rest offset carries the object height ...
    np.testing.assert_allclose(anim.offsets[0], expected_root, atol=1e-4)
    # ... and so does every sampled frame. When only one of the two did, the
    # skeleton stood a leg length below the floor.
    np.testing.assert_allclose(anim.positions[:, 0], expected_root[None, :], atol=1e-4)
