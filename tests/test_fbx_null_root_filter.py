from __future__ import annotations

import os
import sys

import numpy as np


_ANYTOP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_ANYTOP_ROOT)
for _p in [_REPO_ROOT, _ANYTOP_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


from Anytop.motion_lib import FBX


class _FakeTranslation:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


class _FakeQuaternion:
    def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.w = w
        self.x = x
        self.y = y
        self.z = z


class _FakeMatrix:
    def __init__(self, x: float, y: float, z: float):
        self.translation = _FakeTranslation(x, y, z)

    def copy(self):
        t = self.translation
        return _FakeMatrix(t.x, t.y, t.z)

    def inverted_safe(self):
        t = self.translation
        return _FakeMatrix(-t.x, -t.y, -t.z)

    def __matmul__(self, other):
        t1 = self.translation
        t2 = other.translation
        return _FakeMatrix(t1.x + t2.x, t1.y + t2.y, t1.z + t2.z)

    def to_quaternion(self):
        return _FakeQuaternion()


class _FakeBone:
    def __init__(self, name: str, *, matrix_local: _FakeMatrix):
        self.name = name
        self.parent = None
        self.children = []
        self.matrix_local = matrix_local

    def add_child(self, child: "_FakeBone") -> None:
        child.parent = self
        self.children.append(child)


class _FakeArmatureData:
    def __init__(self, bones):
        self.bones = bones


class _FakeArmature:
    def __init__(self, bones):
        self.data = _FakeArmatureData(bones)


def test_extract_armature_skeleton_data_promotes_null_root_children_and_keeps_largest_subtree():
    null_root = _FakeBone("NuLl", matrix_local=_FakeMatrix(10.0, 0.0, 0.0))

    small_root = _FakeBone("SmallRoot", matrix_local=_FakeMatrix(11.0, 0.0, 0.0))
    small_leaf = _FakeBone("SmallLeaf", matrix_local=_FakeMatrix(12.0, 0.0, 0.0))
    small_root.add_child(small_leaf)

    large_root = _FakeBone("LargeRoot", matrix_local=_FakeMatrix(20.0, 0.0, 0.0))
    large_mid = _FakeBone("LargeMid", matrix_local=_FakeMatrix(21.0, 0.0, 0.0))
    large_leaf = _FakeBone("LargeLeaf", matrix_local=_FakeMatrix(22.0, 0.0, 0.0))
    large_root.add_child(large_mid)
    large_mid.add_child(large_leaf)

    other_root = _FakeBone("OtherRoot", matrix_local=_FakeMatrix(30.0, 0.0, 0.0))
    other_leaf = _FakeBone("OtherLeaf", matrix_local=_FakeMatrix(31.0, 0.0, 0.0))
    other_root.add_child(other_leaf)

    null_root.add_child(small_root)
    null_root.add_child(large_root)

    armature = _FakeArmature(
        [
            null_root,
            small_root,
            small_leaf,
            large_root,
            large_mid,
            large_leaf,
            other_root,
            other_leaf,
        ]
    )

    bone_names, parents, offsets, rest_rotations = FBX._extract_armature_skeleton_data(armature)

    assert bone_names == ["LargeRoot", "LargeMid", "LargeLeaf"]
    np.testing.assert_array_equal(parents, np.asarray([-1, 0, 1], dtype=np.int32))
    np.testing.assert_allclose(offsets[0], [20.0, 0.0, 0.0])
    np.testing.assert_allclose(offsets[1:], [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(rest_rotations, [[1.0, 0.0, 0.0, 0.0]] * 3)