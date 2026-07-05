from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest
import torch

bpy = pytest.importorskip("bpy")
from mathutils import Matrix, Quaternion


_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ANYTOP_ROOT = os.path.dirname(_TESTS_DIR)
_REPO_ROOT = os.path.dirname(_ANYTOP_ROOT)

for _path in [_REPO_ROOT, _ANYTOP_ROOT]:
    if _path not in sys.path:
        sys.path.insert(0, _path)


from Anytop.utils.roundtrip_common import build_skeleton
import Anytop.utils.exporter as exporter_mod


class _AbortRetarget(RuntimeError):
    pass


class _FakeModifier:
    def __init__(self, armature) -> None:
        self.type = "ARMATURE"
        self.object = armature


class _FakeObject:
    def __init__(
        self,
        *,
        name: str,
        object_type: str,
        matrix_world: Matrix,
        parent=None,
        modifiers=None,
    ) -> None:
        self.name = name
        self.type = object_type
        self.matrix_world = matrix_world.copy()
        self.parent = parent
        self.modifiers = list(modifiers or [])
        self.matrix_parent_inverse = Matrix.Identity(4)
        self.animation_data = None
        self.data = types.SimpleNamespace(bones=[], edit_bones=[])

    def animation_data_clear(self) -> None:
        self.animation_data = None

    def animation_data_create(self) -> None:
        self.animation_data = types.SimpleNamespace(action=None)


class _FakeBpy:
    def __init__(self) -> None:
        self.data = types.SimpleNamespace(objects=[], actions=[])
        self.context = types.SimpleNamespace(
            scene=types.SimpleNamespace(render=types.SimpleNamespace(fps=30, fps_base=1.0)),
            view_layer=types.SimpleNamespace(objects=types.SimpleNamespace(active=None)),
        )
        self.ops = types.SimpleNamespace(
            wm=types.SimpleNamespace(read_factory_settings=self._read_factory_settings),
            import_scene=types.SimpleNamespace(gltf=self._import_gltf),
        )
        self._import_callback = None

    def _read_factory_settings(self, use_empty=True) -> None:
        del use_empty
        self.data.objects = []
        self.data.actions = []

    def _import_gltf(self, filepath: str) -> None:
        if self._import_callback is None:
            raise AssertionError(f"Unexpected glTF import: {filepath}")
        self._import_callback(filepath)


def _matrix_to_np(matrix: Matrix) -> np.ndarray:
    return np.asarray([list(row) for row in matrix], dtype=np.float64)


def _make_transform(
    *,
    translation=(0.0, 0.0, 0.0),
    rotation_axis=(0.0, 0.0, 1.0),
    rotation_rad=0.0,
    scale=(1.0, 1.0, 1.0),
) -> Matrix:
    rotation = Quaternion(rotation_axis, rotation_rad).to_matrix().to_4x4()
    scale_matrix = Matrix.Diagonal((float(scale[0]), float(scale[1]), float(scale[2]), 1.0))
    return Matrix.Translation(tuple(float(v) for v in translation)) @ rotation @ scale_matrix


def _make_test_skeleton():
    joint_names = ["Hips", "Spine"]
    parents = np.array([-1, 0], dtype=np.int32)
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    rest_rotations = np.zeros((2, 4), dtype=np.float32)
    rest_rotations[:, 0] = 1.0
    skeleton = build_skeleton(joint_names, offsets, parents, rest_rotations)
    joint_rotations = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    root_translation = torch.zeros((1, 3), dtype=torch.float32)
    root_rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    return skeleton, joint_rotations, root_translation, root_rotation


def test_normalize_imported_armature_and_meshes_drops_object_translation_scale_and_preserves_bind() -> None:
    armature_world = _make_transform(
        translation=(2.0, -3.0, 5.0),
        rotation_axis=(0.0, 1.0, 0.0),
        rotation_rad=np.pi / 3.0,
        scale=(1.7, 1.7, 1.7),
    )
    armature = _FakeObject(name="Armature", object_type="ARMATURE", matrix_world=armature_world)

    mesh_parented = _FakeObject(
        name="MeshParented",
        object_type="MESH",
        matrix_world=_make_transform(
            translation=(1.5, 0.25, -0.75),
            rotation_axis=(1.0, 0.0, 0.0),
            rotation_rad=np.pi / 5.0,
            scale=(0.25, 0.25, 0.25),
        ),
        parent=armature,
    )
    mesh_modifier_only = _FakeObject(
        name="MeshModifierOnly",
        object_type="MESH",
        matrix_world=_make_transform(
            translation=(-1.0, 2.0, 3.5),
            rotation_axis=(0.0, 0.0, 1.0),
            rotation_rad=np.pi / 7.0,
            scale=(0.5, 0.75, 1.25),
        ),
        modifiers=[_FakeModifier(armature)],
    )
    unrelated_mesh = _FakeObject(
        name="UnrelatedMesh",
        object_type="MESH",
        matrix_world=_make_transform(
            translation=(9.0, 8.0, 7.0),
            rotation_axis=(1.0, 0.0, 0.0),
            rotation_rad=np.pi / 9.0,
            scale=(1.0, 2.0, 3.0),
        ),
    )

    fake_bpy = types.SimpleNamespace(
        data=types.SimpleNamespace(objects=[armature, mesh_parented, mesh_modifier_only, unrelated_mesh])
    )

    old_arm_world = armature.matrix_world.copy()
    old_mesh_parented_world = mesh_parented.matrix_world.copy()
    old_mesh_modifier_world = mesh_modifier_only.matrix_world.copy()
    old_unrelated_world = unrelated_mesh.matrix_world.copy()
    old_parented_bind = old_arm_world.inverted() @ old_mesh_parented_world
    old_modifier_bind = old_arm_world.inverted() @ old_mesh_modifier_world

    exporter_mod._normalize_imported_armature_and_meshes(fake_bpy, armature)

    arm_translation = armature.matrix_world.to_translation()
    arm_scale = armature.matrix_world.to_scale()
    assert np.allclose([arm_translation.x, arm_translation.y, arm_translation.z], [0.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose([arm_scale.x, arm_scale.y, arm_scale.z], [1.0, 1.0, 1.0], atol=1e-6)
    assert np.allclose(
        _matrix_to_np(armature.matrix_world.to_quaternion().to_matrix().to_4x4()),
        _matrix_to_np(old_arm_world.to_quaternion().to_matrix().to_4x4()),
        atol=1e-6,
    )

    new_parented_bind = armature.matrix_world.inverted() @ mesh_parented.matrix_world
    new_modifier_bind = armature.matrix_world.inverted() @ mesh_modifier_only.matrix_world
    assert np.allclose(_matrix_to_np(new_parented_bind), _matrix_to_np(old_parented_bind), atol=1e-6)
    assert np.allclose(_matrix_to_np(new_modifier_bind), _matrix_to_np(old_modifier_bind), atol=1e-6)
    assert np.allclose(_matrix_to_np(unrelated_mesh.matrix_world), _matrix_to_np(old_unrelated_world), atol=1e-6)
    assert np.allclose(_matrix_to_np(mesh_parented.matrix_parent_inverse), _matrix_to_np(Matrix.Identity(4)), atol=1e-6)
    assert np.allclose(_matrix_to_np(mesh_modifier_only.matrix_parent_inverse), _matrix_to_np(Matrix.Identity(4)), atol=1e-6)


def test_remove_mesh_objects_for_skeleton_only_export_keeps_armature() -> None:
    class _FakeObjectCollection(list):
        def remove(self, obj, do_unlink=False) -> None:
            del do_unlink
            super().remove(obj)

    armature = _FakeObject(
        name="Armature",
        object_type="ARMATURE",
        matrix_world=Matrix.Identity(4),
    )
    mesh_a = _FakeObject(
        name="Body",
        object_type="MESH",
        matrix_world=Matrix.Identity(4),
    )
    mesh_b = _FakeObject(
        name="Eyes",
        object_type="MESH",
        matrix_world=Matrix.Identity(4),
    )
    fake_bpy = types.SimpleNamespace(
        data=types.SimpleNamespace(
            objects=_FakeObjectCollection([armature, mesh_a, mesh_b])
        )
    )

    removed = exporter_mod._remove_mesh_objects_for_skeleton_only_export(fake_bpy)

    assert removed == 2
    assert list(fake_bpy.data.objects) == [armature]


def test_clear_imported_animation_data_removes_source_actions() -> None:
    armature = _FakeObject(
        name="Armature",
        object_type="ARMATURE",
        matrix_world=Matrix.Identity(4),
    )
    mesh = _FakeObject(
        name="Body",
        object_type="MESH",
        matrix_world=Matrix.Identity(4),
    )
    old_arm_action = types.SimpleNamespace(name="SourceArmatureAction")
    old_mesh_action = types.SimpleNamespace(name="SourceMeshAction")
    armature.animation_data = types.SimpleNamespace(action=old_arm_action)
    mesh.animation_data = types.SimpleNamespace(action=old_mesh_action)

    fake_bpy = types.SimpleNamespace(
        data=types.SimpleNamespace(
            objects=[armature, mesh],
            actions=[old_arm_action, old_mesh_action],
        )
    )

    cleared = exporter_mod._clear_imported_animation_data(fake_bpy)

    assert cleared == 2
    assert armature.animation_data is None
    assert mesh.animation_data is None
    assert fake_bpy.data.actions == []


def _run_export_branch_case(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    *,
    mesh_path: str,
    global_similarity,
) -> tuple[dict[str, object], list[str]]:
    skeleton, joint_rotations, root_translation, root_rotation = _make_test_skeleton()
    exporter = exporter_mod.AnimationExporter(skeleton, fps=30.0)
    fake_bpy = _FakeBpy()
    fake_armature = _FakeObject(
        name="Armature",
        object_type="ARMATURE",
        matrix_world=_make_transform(
            translation=(0.5, -0.25, 1.0),
            rotation_axis=(1.0, 0.0, 0.0),
            rotation_rad=np.pi / 4.0,
            scale=(2.0, 2.0, 2.0),
        ),
    )
    fake_mesh = _FakeObject(
        name="Mesh",
        object_type="MESH",
        matrix_world=_make_transform(
            translation=(2.0, 1.0, -1.0),
            rotation_axis=(0.0, 0.0, 1.0),
            rotation_rad=np.pi / 6.0,
            scale=(0.25, 0.5, 0.75),
        ),
        parent=fake_armature,
        modifiers=[_FakeModifier(fake_armature)],
    )
    fake_armature._skeleton_data = (
        ["Hips", "Spine"],
        np.array([-1, 0], dtype=np.int32),
        np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
    )

    def _populate_scene(_filepath: str) -> None:
        fake_bpy.data.objects = [fake_armature, fake_mesh]

    fake_bpy._import_callback = _populate_scene
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    monkeypatch.setattr(exporter_mod, "remove_lights_and_cameras", lambda: None)
    monkeypatch.setattr(exporter_mod, "import_fbx", lambda filepath, use_image_search=False: _populate_scene(filepath))
    monkeypatch.setattr(exporter_mod, "extract_armature_skeleton_data", lambda armature: armature._skeleton_data)
    monkeypatch.setattr(exporter_mod, "_build_canonical_match_names", lambda names, parents, offsets, log_hint: list(names))
    monkeypatch.setattr(
        exporter_mod,
        "_build_canonical_name_variants",
        lambda names, parents, offsets, log_hint: (list(names), list(names)),
    )

    calls: list[str] = []
    monkeypatch.setattr(
        exporter_mod,
        "_normalize_imported_armature_and_meshes",
        lambda bpy, armature: calls.append("normalize"),
    )
    monkeypatch.setattr(
        exporter_mod,
        "_apply_gltf_output_space_similarity",
        lambda bpy, armature, scale_factor, orientation_quat_wxyz: calls.append("similarity"),
    )

    captured: dict[str, object] = {}

    def _fake_retarget_world_space_np(**kwargs):
        captured["coordinate_search"] = bool(kwargs["coordinate_search"])
        captured["src_root_translation"] = np.asarray(kwargs["src_root_translation"], dtype=np.float64)
        raise _AbortRetarget("stop after branch capture")

    monkeypatch.setattr(exporter_mod, "retarget_world_space_np", _fake_retarget_world_space_np)

    with pytest.raises(_AbortRetarget, match="branch capture"):
        exporter.export_glb(
            joint_rotations,
            root_translation,
            root_rotation,
            str(tmp_path / "out.glb"),
            mesh_path=mesh_path,
            global_similarity=global_similarity,
        )

    return captured, calls


def test_export_glb_plain_gltf_mesh_keeps_existing_basis(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    captured, calls = _run_export_branch_case(
        monkeypatch,
        tmp_path,
        mesh_path="character_source.glb",
        global_similarity=None,
    )

    assert captured["coordinate_search"] is False
    assert calls == []


def test_export_glb_hml_reverse_aligned_gltf_reenables_coordinate_search(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    captured, calls = _run_export_branch_case(
        monkeypatch,
        tmp_path,
        mesh_path="character_source.glb",
        global_similarity=(1.4354808536266768, np.array([0.70710678, 0.0, -0.70710678, 0.0], dtype=np.float64)),
    )

    assert captured["coordinate_search"] is True
    assert calls == ["normalize", "similarity"]


def test_export_glb_fbx_mesh_still_normalizes_and_searches_coordinates(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    captured, calls = _run_export_branch_case(
        monkeypatch,
        tmp_path,
        mesh_path="character_source.fbx",
        global_similarity=None,
    )

    assert captured["coordinate_search"] is True
    assert calls == ["normalize"]
