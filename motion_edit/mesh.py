"""Skinned mesh of a package: the T-pose mesh it was decomposed with.

A package decomposed with ``decompose_clip --tpose_mesh`` keeps, next to its
manifest and arrays, a ``mesh/`` directory:

    mesh/source.json       {"tpose_mesh": <absolute path of the T-pose FBX / GLB>}
    mesh/preview.glb       the mesh skinned on the package skeleton's rest pose
    mesh/calibration.json  per preview bone, the joint that drives it and a constant
                           matrix (:func:`calibrate`); the tuning UI skins the
                           edited skeleton with it, no decode in the browser

The directory is not part of the arrays the runtime composes from, so
re-decomposing (same skeleton, same rest pose) or editing contacts keeps it.
Skinned exports re-import the T-pose mesh itself: it has to stay where it was
when the package was made.  They are written in the mesh's native space (the
default of ``tools/restore_glb_from_npy.py``), not the HML space of the
skeleton-only export.

Writing the preview and the skinned exports runs bpy, which only works on a
process's main thread.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Optional

import numpy as np

from motion_edit.package import EditPackage, decode_json

MESH_DIR = "mesh"
SOURCE_FILE = "source.json"
PREVIEW_FILE = "preview.glb"
CALIBRATION_FILE = "calibration.json"


def mesh_source(package_dir: str) -> Optional[str]:
    """The package's T-pose mesh path, ``None`` without one."""
    path = os.path.join(package_dir, MESH_DIR, SOURCE_FILE)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle).get("tpose_mesh") or None


def preview_path(package_dir: str) -> Optional[str]:
    path = os.path.join(package_dir, MESH_DIR, PREVIEW_FILE)
    return path if os.path.isfile(path) else None


def mesh_context(package: EditPackage, tpose_mesh: str):
    from utils.npy_restore import build_mesh_restore_context

    cond = decode_json(package["source_cond"])
    if "root_promote_depth" not in cond:
        # without it the T-pose armature keeps its wrapper roots and every bone's rest is off
        raise ValueError("package's cond subset has no root_promote_depth: decompose it again "
                         "with motion_edit.decompose_clip")
    return build_mesh_restore_context(cond, tpose_mesh, package.manifest["object_type"],
                                      feature_joint_count=package.joint_count)


def hml_similarity(package: EditPackage) -> tuple[float, np.ndarray]:
    """``(s, q)`` with ``hml = s · rotate(q, native)``: the preprocess similarity."""
    cond = decode_json(package["source_cond"])
    return (float(cond["scale_factor"]),
            np.asarray(cond["orientation_quat"], dtype=np.float64).reshape(-1, 4)[0])


def export_skinned_glb(package: EditPackage, animation, fps: float, path: str, tpose_mesh: str) -> str:
    """``animation`` (the package's HML feature basis) on the T-pose mesh, in the mesh's
    native space: the default skinned path of ``tools/restore_glb_from_npy.py``, run
    on an edited animation."""
    from data_loaders.truebones.truebones_utils.features import recover_processed_animation_from_feature_animation
    from utils.exporter import AnimationExporter, animation_to_exporter_inputs
    from utils.npy_restore import invert_preprocess_transform
    from utils.roundtrip_common import build_skeleton

    ctx = mesh_context(package, tpose_mesh)
    baked = recover_processed_animation_from_feature_animation(animation, ctx.tpose_rest_rotations)
    native = invert_preprocess_transform(baked, scale_factor=ctx.scale_factor, root_translation_xz=None,
                                         orientation_quat=ctx.orientation_quat)
    skeleton = build_skeleton(ctx.export_joint_names, ctx.export_offsets, ctx.export_parents,
                              ctx.export_rest_rotations)
    joint_rotations, root_translation, root_rotation, bone_translations = (
        animation_to_exporter_inputs(native, skeleton))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    AnimationExporter(skeleton, fps=fps).export_glb(
        joint_rotations, root_translation, root_rotation, path,
        mesh_path=tpose_mesh, bone_translations=bone_translations, export_mesh=True,
    )
    return os.path.abspath(path)


def rest_animation(package: EditPackage, frames: int = 2):
    """The package skeleton standing in its rest pose (identity local rotations)."""
    from motion_lib.Animation import Animation
    from motion_lib.Quaternions import Quaternions

    joints = package.joint_count
    rotations = np.zeros((frames, joints, 4))
    rotations[..., 0] = 1.0
    offsets = np.asarray(package["anim_offsets"], dtype=np.float64)
    return Animation(Quaternions(rotations), np.repeat(offsets[None], frames, axis=0),
                     Quaternions(np.array(package["orients"], copy=True)), offsets.copy(),
                     np.array(package["parents"], copy=True))


def _quat_matrix(xyzw) -> np.ndarray:
    x, y, z, w = xyzw
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _glb_skin_worlds(path: str) -> dict[str, np.ndarray]:
    """World matrix of every skin joint of a GLB at its first animation key, by node name."""
    import pygltflib

    gltf = pygltflib.GLTF2().load(path)
    blob = gltf.binary_blob()
    trs = [[np.array(n.translation or (0, 0, 0), float), np.array(n.rotation or (0, 0, 0, 1), float),
            np.array(n.scale or (1, 1, 1), float)] for n in gltf.nodes]
    for animation in gltf.animations[:1]:
        for channel in animation.channels:
            slot = {"translation": 0, "rotation": 1, "scale": 2}.get(channel.target.path)
            accessor = gltf.accessors[animation.samplers[channel.sampler].output]
            if slot is None or accessor.componentType != pygltflib.FLOAT:
                continue
            width = 4 if slot == 1 else 3
            view = gltf.bufferViews[accessor.bufferView]
            trs[channel.target.node][slot] = np.frombuffer(
                blob, np.float32, width, (view.byteOffset or 0) + (accessor.byteOffset or 0)).astype(float)
    parent = {c: i for i, n in enumerate(gltf.nodes) for c in (n.children or [])}
    worlds: dict[int, np.ndarray] = {}

    def world(i: int) -> np.ndarray:
        if i not in worlds:
            t, r, s = trs[i]
            m = np.eye(4)
            m[:3, :3] = _quat_matrix(r) * s
            m[:3, 3] = t
            worlds[i] = world(parent[i]) @ m if i in parent else m
        return worlds[i]

    return {gltf.nodes[j].name: world(j) for skin in gltf.skins for j in skin.joints}


def _similarity(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """4x4 similarity taking ``source`` points onto ``target`` (Umeyama)."""
    mu_s, mu_t = source.mean(0), target.mean(0)
    a, b = source - mu_s, target - mu_t
    u, d, vt = np.linalg.svd(b.T @ a / len(source))
    e = np.eye(3)
    e[2, 2] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ e @ vt
    scale = (d * np.diag(e)).sum() / (a ** 2).sum() * len(source)
    m = np.eye(4)
    m[:3, :3] = scale * rotation
    m[:3, 3] = mu_t - scale * rotation @ mu_s
    return m


def calibrate(package: EditPackage, preview: str) -> dict:
    """How the tuning UI drives the preview's bones from the package joints.

    The preview stands in the package's rest pose; a similarity S takes its world
    space onto the package's (the import / export unit and axis conventions).  A
    bone then sits at ``W_a(t) · C`` for every frame t, where ``W_a`` is the world
    transform of the bone's own joint (or its nearest ancestor joint) and
    ``C = W_a(rest)⁻¹ · S · M(rest)``; a bone above every joint stays at ``S · M``.
    Matrices are column-major, for three.js.
    """
    from motion_edit.runtime import forward_kinematics

    bones = _glb_skin_worlds(preview)
    names = [str(n) for n in package["names"]]
    rest = rest_animation(package, frames=1)
    _, positions = forward_kinematics(package["parents"], np.asarray(rest.rotations.qs, float),
                                      np.asarray(rest.positions, float))
    joint_world = []
    for p in positions[0]:
        m = np.eye(4)
        m[:3, 3] = p
        joint_world.append(m)
    matched = [j for j, n in enumerate(names) if n in bones]
    if len(matched) < 3:
        raise ValueError(f"{preview}: only {len(matched)} bone(s) named like the package joints")
    similarity = _similarity(np.array([bones[names[j]][:3, 3] for j in matched]),
                             np.array([positions[0, j] for j in matched]))
    residual = max(np.linalg.norm((similarity @ bones[names[j]])[:3, 3] - positions[0, j]) for j in matched)
    import pygltflib

    gltf = pygltflib.GLTF2().load(preview)
    parent = {gltf.nodes[c].name: n.name for n in gltf.nodes for c in (n.children or [])}
    index = {n: j for j, n in enumerate(names)}
    out = {}
    for name, world in bones.items():
        anchor = name
        while anchor is not None and anchor not in index:
            anchor = parent.get(anchor)
        if anchor is None:
            out[name] = {"joint": -1, "matrix": (similarity @ world).T.ravel().tolist()}
        else:
            j = index[anchor]
            matrix = np.linalg.inv(joint_world[j]) @ similarity @ world
            out[name] = {"joint": j, "matrix": matrix.T.ravel().tolist()}
    return {"bones": out, "rest_residual": float(residual),
            "unskinned_joints": [n for n in names if n not in bones]}


def mesh_calibration(package_dir: str) -> Optional[dict]:
    path = os.path.join(package_dir, MESH_DIR, CALIBRATION_FILE)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def attach_mesh(package_dir: str, package: EditPackage, tpose_mesh: Optional[str]) -> Optional[dict]:
    """Give the package at ``package_dir`` the T-pose mesh (``None`` removes it):
    write its preview and calibration; returns the calibration."""
    if tpose_mesh:
        tpose_mesh = os.path.abspath(tpose_mesh)
        if not os.path.isfile(tpose_mesh):
            raise FileNotFoundError(f"T-pose mesh not found: {tpose_mesh}")
    directory = os.path.join(package_dir, MESH_DIR)
    if os.path.isdir(directory):
        shutil.rmtree(directory)
    if not tpose_mesh:
        return None
    preview = export_skinned_glb(package, rest_animation(package), package.fps,
                                 os.path.join(directory, PREVIEW_FILE), tpose_mesh)
    calibration = calibrate(package, preview)
    for file_name, payload in ((CALIBRATION_FILE, calibration), (SOURCE_FILE, {"tpose_mesh": tpose_mesh})):
        with open(os.path.join(directory, file_name), "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False)
            handle.write("\n")
    return calibration
