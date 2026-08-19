"""
restore_glb_from_npy.py

Restore a preprocessed Anytop NPY motion file back to a skinned GLB,
using a T-pose FBX as the mesh/rig source.

Pipeline:
    NPY features
        → recover_from_features(...)                    — feature-space Animation
        → recover_processed_animation_from_feature_animation(...)  — undo T-pose reparameterization
        → invert preprocess transform back to raw FBX rig space
        → animation_to_exporter_inputs(...)
        → AnimationExporter + T-pose FBX → skinned GLB

Metadata is resolved from cond.npy (dataset-wide metadata indexed by
object_type).  The T-pose mesh is loaded to obtain collapsed skeleton
info for the exporter and T-pose rest rotations; all structural fields
(joints_names, parents, offsets, scale_factor, orientation_quat) must
be present in cond.npy — missing fields will cause an error.

Note: locomotion XZ stripped during preprocessing cannot be recovered from a
plain feature tensor alone. Non-locomotion clips also stay in their centred
preprocessed space unless an explicit root-translation XZ override is passed
during restore.

Usage:
    # Skinned GLB in native (mesh) space
    python tools/restore_glb_from_npy.py \\
        --npy "F:/npy/Horse___RunToStop_29.npy" \\
        --tpose-mesh "D:/Models/HorseALL-TPOSE.fbx"

    # Skinned GLB in HML preprocessed space
    python tools/restore_glb_from_npy.py \\
        --npy "F:/npy/Horse___RunToStop_29.npy" \\
        --tpose-mesh "D:/Models/HorseALL-TPOSE.fbx" \\
        --restore-space hml

    # Skeleton-only GLB from cond.npy (HML space)
    python tools/restore_glb_from_npy.py \\
        --npy "F:/npy/Horse___RunToStop_29.npy" \\
        --skeleton-only

    # Skeleton-only GLB using T-pose armature for rest rotations (native space)
    python tools/restore_glb_from_npy.py \\
        --npy "F:/npy/Horse___RunToStop_29.npy" \\
        --tpose-mesh "D:/Models/HorseALL-TPOSE.fbx" \\
        --skeleton-only

``--restore-space`` modes:
    native (default)  Align the animation to the mesh's original orientation / scale.
    hml               Keep the NPY's preprocessed orientation / scale / placement.

"""

import argparse
import importlib.util
import math
import os
import subprocess
import sys

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANYTOP_DIR = os.path.realpath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.dirname(ANYTOP_DIR)

for _p in [REPO_ROOT, ANYTOP_DIR, os.path.join(ANYTOP_DIR, "tests")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_utils_module(module_name: str) -> None:
    module_path = os.path.join(ANYTOP_DIR, "utils", f"{module_name.rsplit('.', 1)[-1]}.py")
    if not os.path.isfile(module_path) or module_name in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


_load_utils_module("utils.rotation_conversions")
_load_utils_module("utils.npy_roundtrip_utils")
_load_utils_module("utils.misc")

from utils.misc import infer_object_type_from_filename
from utils.npy_roundtrip_utils import recover_from_features
from utils.roundtrip_common import (
    load_fbx_skeleton_metadata,
)
from motion_lib.FBX import collapse_root_skeleton

# ── Default cond.npy path ─────────────────────────────────────────────────────

_DEFAULT_COND_NPY = os.path.realpath(
    os.path.join(ANYTOP_DIR, "dataset", "truebones", "zoo", "truebones_processed", "cond.npy")
)

from utils.fullbody_ik import (
    DEFAULT_IK_STRETCH_FACTOR,
    rebuild_fullbody_animation_with_ik,
)

def _load_tpose_restore_metadata(
    tpose_mesh: str,
    object_type: str,
    *,
    expected_joint_count: int | None = None,
) -> dict[str, object]:
    from data_loaders.truebones.truebones_utils.motion_process import get_common_features_from_T_pose, TPoseFeatures

    tpose_lower = tpose_mesh.lower()
    if not tpose_lower.endswith(('.fbx', '.glb', '.gltf')):
        raise ValueError(f"Unsupported T-pose mesh format: {tpose_mesh} - expected .fbx, .glb, or .gltf")

    raw_joint_names, raw_parents, raw_offsets, raw_rest_rotations = load_fbx_skeleton_metadata(tpose_mesh)
    uncropped_joint_cap = max(
        len(raw_joint_names),
        int(expected_joint_count or 0),
        1,
    )
    # Restore/inference should consume the full T-pose skeleton. The default
    # get_common_features_from_T_pose(max_joints=MAX_JOINTS) cap is a training
    # concern and would silently drop tail joints here.
    tp: TPoseFeatures = get_common_features_from_T_pose(
        tpose_mesh,
        object_type,
        max_joints=uncropped_joint_cap,
    )
    raw_parents = np.asarray(raw_parents, dtype=np.int32)
    raw_offsets = np.asarray(raw_offsets, dtype=np.float32)
    raw_rest_rotations = np.asarray(raw_rest_rotations, dtype=np.float32)
    collapsed_joint_names, collapsed_parents, collapsed_offsets, collapsed_rest_rotations = (
        collapse_root_skeleton(
            raw_joint_names,
            raw_parents,
            raw_offsets,
            raw_rest_rotations[None, ...],
            raw_offsets[None, ...],
        )
    )[:4]
    return {
        "joint_names": list(tp.names),
        "parents": np.asarray(tp.tpos_anim.parents, dtype=np.int32),
        "offsets": np.asarray(tp.offsets, dtype=np.float32),
        "tpose_rest_rotations": np.asarray(tp.tpos_rots[0], dtype=np.float32),
        "orientation_quat": np.asarray(tp.orientation_quat, dtype=np.float64),
        "scale_factor": float(tp.scale_factor),
        "raw_joint_names": list(raw_joint_names),
        "raw_parents": raw_parents,
        "raw_offsets": raw_offsets,
        "raw_rest_rotations": raw_rest_rotations,
        "collapsed_joint_names": list(collapsed_joint_names),
        "collapsed_parents": np.asarray(collapsed_parents, dtype=np.int32),
        "collapsed_offsets": np.asarray(collapsed_offsets, dtype=np.float32),
        "collapsed_rest_rotations": np.asarray(collapsed_rest_rotations[0], dtype=np.float32),
    }


def _warn_on_missing_mesh_joints(
    joint_names: list[str],
    tpose_mesh: str,
    mesh_bone_names: list[str] | None = None,
) -> None:
    """Warn about recovered joints missing from the T-pose armature.

    When *mesh_bone_names* is provided (from a previous FBX load), skip
    re-loading the FBX.
    """
    if mesh_bone_names is None:
        mesh_bone_names, _mesh_parents, _mesh_offsets, _mesh_rest_rots = load_fbx_skeleton_metadata(
            tpose_mesh
        )

    mesh_name_set = set(mesh_bone_names)
    missing = [joint_name for joint_name in joint_names if joint_name not in mesh_name_set]
    if missing:
        preview = missing[:10]
        suffix = "..." if len(missing) > 10 else ""
        print(
            f"WARNING: {len(missing)} recovered joints not found in the T-pose armature:\n"
            f"  {preview}{suffix}\n"
            f"These bones stay at rest pose in the exported mesh."
        )
        return

    print(f"All {len(joint_names)} recovered joints found in the T-pose armature.")


def _remap_skeleton_metadata(
    source_names: list[str],
    source_parents: np.ndarray,
    source_offsets: np.ndarray,
    source_rest_rotations: np.ndarray,
    target_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if list(source_names) == list(target_names):
        return (
            np.asarray(source_parents, dtype=np.int32),
            np.asarray(source_offsets),
            np.asarray(source_rest_rotations),
        )

    source_index = {name: index for index, name in enumerate(source_names)}
    target_index = {name: index for index, name in enumerate(target_names)}
    missing = [name for name in target_names if name not in source_index]
    if missing:
        preview = missing[:10]
        suffix = "..." if len(missing) > 10 else ""
        raise ValueError(f"T-pose mesh is missing export skeleton joints: {preview}{suffix}")

    parents = np.full((len(target_names),), -1, dtype=np.int32)
    offsets = np.zeros((len(target_names), 3), dtype=np.float32)
    rest_rotations = np.zeros((len(target_names), 4), dtype=np.float32)
    for target_joint_idx, joint_name in enumerate(target_names):
        source_joint_idx = source_index[joint_name]
        offsets[target_joint_idx] = source_offsets[source_joint_idx]
        rest_rotations[target_joint_idx] = source_rest_rotations[source_joint_idx]
        parent_idx = int(source_parents[source_joint_idx])
        if parent_idx >= 0:
            parent_name = source_names[parent_idx]
            if parent_name not in target_index:
                raise ValueError(f"T-pose mesh parent '{parent_name}' for joint '{joint_name}' is missing")
            parents[target_joint_idx] = target_index[parent_name]

    return parents, offsets, rest_rotations


def _build_restore_context(
    raw_npy,
    object_type: str,
    tpose_mesh: str,
    cond_entry: dict,
) -> dict[str, object]:
    features = np.asarray(raw_npy)
    feature_joint_count = int(features.shape[1]) if features.ndim >= 2 else 0

    # ── Require all fields from cond.npy ────────────────────────────────
    for key in ("joints_names", "parents", "offsets", "scale_factor", "orientation_quat"):
        if key not in cond_entry:
            raise ValueError(
                f"cond.npy entry for '{object_type}' is missing required field '{key}'."
            )

    joint_names = list(cond_entry["joints_names"])
    parents = np.asarray(cond_entry["parents"], dtype=np.int32)
    offsets = np.asarray(cond_entry["offsets"], dtype=np.float32)
    tpose_meta = _load_tpose_restore_metadata(
        tpose_mesh,
        object_type,
        expected_joint_count=len(joint_names),
    )

    if feature_joint_count not in (0, len(joint_names)):
        raise ValueError(
            f"NPY has J={feature_joint_count} joints but cond.npy has "
            f"{len(joint_names)} joints for '{object_type}'."
        )

    # ── T-pose rest rotations (from T-pose mesh) ────────────────────────
    tpose_joint_names = list(tpose_meta["joint_names"])
    tpose_rest_src = np.asarray(tpose_meta["tpose_rest_rotations"], dtype=np.float32)
    tpose_name_index = {name: idx for idx, name in enumerate(tpose_joint_names)}
    tpose_rest_rotations = np.zeros((len(joint_names), 4), dtype=np.float32)
    tpose_rest_rotations[:, 0] = 1.0
    for j, name in enumerate(joint_names):
        if name in tpose_name_index:
            tpose_rest_rotations[j] = tpose_rest_src[tpose_name_index[name]]

    scale_factor = float(cond_entry["scale_factor"])
    orientation_quat = np.asarray(cond_entry["orientation_quat"], dtype=np.float64)

    export_joint_names = list(joint_names)

    export_parents, export_offsets, export_rest_rotations = _remap_skeleton_metadata(
        list(tpose_meta["collapsed_joint_names"]),
        np.asarray(tpose_meta["collapsed_parents"], dtype=np.int32),
        np.asarray(tpose_meta["collapsed_offsets"], dtype=np.float32),
        np.asarray(tpose_meta["collapsed_rest_rotations"], dtype=np.float32),
        export_joint_names,
    )

    return {
        "features": features,
        "joint_names": joint_names,
        "export_joint_names": export_joint_names,
        "parents": parents,
        "offsets": offsets,
        "tpose_rest_rotations": tpose_rest_rotations,
        "orientation_quat": orientation_quat,
        "scale_factor": scale_factor,
        "mesh_bone_names": list(tpose_meta["raw_joint_names"]),
        "export_parents": export_parents,
        "export_offsets": export_offsets,
        "export_rest_rotations": export_rest_rotations,
    }


def _build_skeleton_only_context(
    raw_npy,
    object_type: str,
    cond_entry: dict,
) -> dict[str, object]:
    """Build restore context from cond.npy only — no T-pose mesh required.

    All skeleton metadata (joint_names, parents, offsets, scale_factor,
    orientation_quat) comes from *cond_entry*.  Rest rotations are identity
    (no T-pose mesh to supply real ones), and export metadata is a direct
    copy of the cond.npy data with no remapping.

    The exported GLB lives in HML preprocessed space (centred, oriented,
    scaled) — ``_invert_preprocess_transform`` must be skipped.
    """
    features = np.asarray(raw_npy)
    feature_joint_count = int(features.shape[1]) if features.ndim >= 2 else 0

    # ── Require all fields from cond.npy ────────────────────────────────
    for key in ("joints_names", "parents", "offsets", "scale_factor", "orientation_quat"):
        if key not in cond_entry:
            raise ValueError(
                f"cond.npy entry for '{object_type}' is missing required field '{key}'."
            )

    joint_names = list(cond_entry["joints_names"])
    parents = np.asarray(cond_entry["parents"], dtype=np.int32)
    offsets = np.asarray(cond_entry["offsets"], dtype=np.float32)
    scale_factor = float(cond_entry["scale_factor"])
    orientation_quat = np.asarray(cond_entry["orientation_quat"], dtype=np.float64)

    if feature_joint_count not in (0, len(joint_names)):
        raise ValueError(
            f"NPY has J={feature_joint_count} joints but cond.npy has "
            f"{len(joint_names)} joints for '{object_type}'."
        )

    # Identity rest rotations — no T-pose mesh available
    identity_rest = np.zeros((len(joint_names), 4), dtype=np.float32)
    identity_rest[:, 0] = 1.0

    return {
        "features": features,
        "joint_names": joint_names,
        "export_joint_names": list(joint_names),
        "parents": parents,
        "offsets": offsets,
        "tpose_rest_rotations": identity_rest,
        "orientation_quat": orientation_quat,
        "scale_factor": scale_factor,
        "mesh_bone_names": list(joint_names),
        "export_parents": parents.copy(),
        "export_offsets": offsets.copy(),
        "export_rest_rotations": identity_rest.copy(),
    }


def _coerce_root_translation_xz(root_translation_xz: np.ndarray) -> np.ndarray:
    root_translation_xz = np.asarray(root_translation_xz, dtype=np.float64).reshape(-1)
    if root_translation_xz.size == 3:
        return root_translation_xz
    if root_translation_xz.size == 2:
        return np.array([root_translation_xz[0], 0.0, root_translation_xz[1]], dtype=np.float64)
    raise ValueError(
        f"root_translation_xz must have shape (2,) or (3,), got {root_translation_xz.shape}"
    )


# (fullbody IK functions extracted to Anytop/utils/fullbody_ik.py)


def _invert_preprocess_transform(
    processed_anim,
    *,
    scale_factor: float,
    root_translation_xz: np.ndarray | None,
    orientation_quat: np.ndarray,
):
    from motion_lib.Animation import Animation
    from motion_lib.Quaternions import Quaternions

    positions = processed_anim.positions.copy().astype(np.float64, copy=False)
    offsets = processed_anim.offsets.copy().astype(np.float64, copy=False)
    rotations = processed_anim.rotations.copy()

    scale_factor = float(scale_factor)
    if scale_factor <= 0.0:
        raise ValueError(f"scale_factor must be positive, got {scale_factor}")
    if abs(scale_factor - 1.0) > 1e-8:
        inv_scale = 1.0 / scale_factor
        positions *= inv_scale
        offsets *= inv_scale

    if root_translation_xz is not None:
        root_offset = _coerce_root_translation_xz(root_translation_xz)
        positions[:, 0] += root_offset
        offsets[0] += root_offset

    orientation_quat = np.asarray(orientation_quat, dtype=np.float64)
    if orientation_quat.ndim > 1:
        orientation_quat = orientation_quat[0]
    if orientation_quat.shape != (4,):
        raise ValueError(f"orientation_quat must have shape (4,), got {orientation_quat.shape}")
    inverse_orientation = -Quaternions(orientation_quat[None, :])
    inverse_orientation = inverse_orientation.repeat(processed_anim.shape[0], axis=0)
    rotations[:, 0] = inverse_orientation * rotations[:, 0]
    positions[:, 0] = inverse_orientation * positions[:, 0]

    return Animation(
        rotations,
        positions,
        processed_anim.orients.copy(),
        offsets,
        processed_anim.parents.copy(),
    )


def _resample_frame_indices(
    frame_count: int,
    src_fps: float,
    tgt_fps: float,
    min_length: int | None = None,
) -> list[float]:
    """Fractional source-frame indices that resample ``frame_count`` to ``tgt_fps``.

    The indices span ``[0, frame_count - 1]`` spaced ``src_fps / tgt_fps`` frames
    apart, so a clip sampled at ``src_fps`` plays back at ``tgt_fps`` over the same
    time span (e.g. 136 frames at 30fps → 68 indices at 15fps).  When ``min_length``
    is given and the resampled clip is shorter, the whole clip is instead
    *time-stretched* to exactly ``min_length`` frames — ``min_length`` indices
    spread evenly across the source range — so a short motion is interpolated
    (slowed down) to fill the minimum length with no looping/seam jump.  Returns
    plain ``range(frame_count)`` when resampling is impossible/unnecessary.
    """
    if frame_count < 1:
        return []
    if src_fps and tgt_fps and src_fps > 0 and tgt_fps > 0 and frame_count >= 2:
        step = src_fps / tgt_fps
        n = int(math.floor((frame_count - 1) / step + 1e-6)) + 1 if step > 0 else frame_count
        times = [i * step for i in range(max(n, 1))]
    else:
        times = [float(i) for i in range(frame_count)]
    if min_length and len(times) < min_length:
        times = np.linspace(0.0, float(frame_count - 1), min_length).tolist()
    return times


def _resample_animation(animation, frame_times):
    """Resample an Animation in time at fractional ``frame_times`` (source-frame units).

    Positions are linearly interpolated and rotations are slerped between the two
    bracketing source frames; the rest pose (orients/offsets/parents) is unchanged.
    Integer-ratio downsampling (e.g. 30→15fps) lands exactly on source frames, so
    it is plain decimation with no interpolation error.
    """
    from motion_lib.Animation import Animation
    from motion_lib.Quaternions import Quaternions

    frame_count = animation.shape[0]
    times = np.clip(np.asarray(frame_times, dtype=np.float64), 0.0, frame_count - 1)
    lo = np.floor(times).astype(np.int64)
    hi = np.minimum(lo + 1, frame_count - 1)
    alpha = (times - lo)[:, None]   # (T, 1) — broadcasts over joints in slerp/lerp

    positions = np.asarray(animation.positions, dtype=np.float64)
    new_positions = (
        positions[lo] * (1.0 - alpha[..., None]) + positions[hi] * alpha[..., None]
    )
    new_rotations = Quaternions.slerp(
        animation.rotations[lo], animation.rotations[hi], alpha
    )
    return Animation(
        new_rotations,
        new_positions,
        animation.orients.copy(),
        animation.offsets.copy(),
        animation.parents.copy(),
    )


# ── Main restore function ─────────────────────────────────────────────────────

def restore_glb(
    npy_path: str,
    output_glb: str,
    tpose_mesh: str | None = None,
    cond_npy: str | None = None,
    object_type: str | None = None,
    fps: float | None = None,
    root_translation_xz: np.ndarray | None = None,
    fullbody_ik: bool = False,
    stretch_factor: float = DEFAULT_IK_STRETCH_FACTOR,
    restore_space: str = "native",
    use_image_search: bool = False,
    resample_fps: float | None = None,
    resample_min_length: int | None = None,
    skeleton_only: bool = False,
) -> str:
    """Restore a preprocessed NPY motion file to a GLB.

    A skinned GLB is produced **only** when the caller explicitly supplies
    *tpose_mesh* (the user-provided skinned mesh). When no explicit mesh is given
    and *skeleton_only* is left ``False``, restore falls back to a skeleton-only
    GLB — no FBX/GLB asset is ever read. cond.npy carries no T-pose mesh path,
    so there is no implicit dataset-mesh resolution.

    When *skeleton_only* is ``True``, the output is a skeleton-only GLB (no mesh,
    no skinning).  Without a T-pose mesh, ``restore_space`` is forced to ``"hml"``
    and all metadata comes from ``cond.npy``.  With a T-pose mesh, ``restore_space``
    is honoured — the mesh supplies proper rest rotations for the skeleton.

    Args:
        npy_path:            Path to the preprocessed .npy motion file.
        output_glb:          Path for the output .glb file.
        tpose_mesh:          Path to the T-pose FBX (provides skin + armature).
                             When *skeleton_only* is also True, the mesh's
                             armature still provides rest rotations and enables
                             FBX-space export, but no skin is bound in the output.
        cond_npy:            Path to cond.npy; defaults to the dataset default.
        object_type:         Character type key (e.g. "Horse").  Auto-detected
                             from the NPY filename if None.
        fps:                 Animation frame rate.  Defaults to 30 if not
                             specified.
        root_translation_xz: Optional explicit XZ translation to add back after
                     inverse scale and before inverse orientation. When
                     omitted, restore keeps the clip in centred
                     preprocessed space.  Ignored when *skeleton_only* is True
                     and no ``--tpose-mesh`` is given (always stays in centred
                     HML space).
        fullbody_ik:          If True, perform a full-body IK reconstruction
                             on the raw export skeleton after recovering the
                             animation.  Default is False (skip IK, use
                             recovered pose directly).
        stretch_factor:       Allowed bone-length elasticity ratio for IK.
                             Each edge may stretch/compress by ±stretch_factor
                             (e.g. 0.1 = ±10 %).  Default is {DEFAULT_IK_STRETCH_FACTOR}.
                             Only effective when fullbody_ik is True.
        restore_space:        Output coordinate space:
                             ``"native"`` (default) aligns the animation to the
                             T-pose mesh's native orientation/scale/translation.
                             ``"hml"`` reverse-aligns the T-pose mesh onto the
                             NPY so the GLB keeps the NPY's orientation, scale,
                             and centered placement (like the corresponding
                             processed BVH).  For skeleton-only exports without
                             a T-pose mesh, this is forced to ``"hml"``.
        use_image_search:     If True, resolve textures for the skinned mesh:
                             the FBX importer first searches directories near
                             the source mesh, then a fallback resolver wires a
                             matching diffuse/alpha texture from the mesh's
                             ``tex/`` folder onto any main character mesh still
                             lacking one. Default False (no texture resolution).
        resample_fps:         If set (and > 0), resample the recovered motion in
                             time from its native rate (``fps``) to this rate
                             before export, and write the GLB at this rate
                             (positions lerped, rotations slerped; integer ratios
                             are exact decimation).  Default None (no resample —
                             the GLB keeps the NPY's native frame count at ``fps``).
        resample_min_length:  When resampling, if the resampled clip is shorter
                             than this, time-stretch the whole clip to exactly this
                             many frames (even interpolation, no looping).  Only
                             effective when ``resample_fps`` is set.  Default None.
        skeleton_only:        If True, export a skeleton-only GLB (no mesh, no
                             skinning).  Without a T-pose mesh the output stays
                             in HML preprocessed space; with a T-pose mesh the
                             restore space is honoured.  Default False.

    Returns:
        The absolute path of the written GLB file.
    """
    from utils.exporter import AnimationExporter, animation_to_exporter_inputs
    from utils.roundtrip_common import build_skeleton
    from data_loaders.truebones.truebones_utils.motion_process import (
        find_translation_root,
        recover_processed_animation_from_feature_animation,
    )

    output_glb = os.path.abspath(output_glb)
    if stretch_factor < 0 or stretch_factor > 1.0:
        raise ValueError(f"stretch_factor must be in [0, 1], got {stretch_factor}")

    # No implicit dataset-mesh resolution: a skinned GLB requires an explicit
    # user-provided mesh.  Absent one, fall back to a skeleton-only export.
    if not skeleton_only and tpose_mesh is None:
        print(
            "No T-pose mesh provided: falling back to skeleton-only export."
        )
        skeleton_only = True

    if skeleton_only and tpose_mesh is None:
        restore_space = "hml"
        print("Skeleton-only mode (no T-pose mesh): restore_space forced to 'hml'")
    elif restore_space not in ("native", "hml"):
        raise ValueError(f"restore_space must be 'native' or 'hml', got {restore_space!r}")

    # ── Load cond.npy ─────────────────────────────────────────────────────────
    cond_npy_path = cond_npy or _DEFAULT_COND_NPY
    if not os.path.isfile(cond_npy_path):
        raise FileNotFoundError(f"cond.npy not found: {cond_npy_path}")
    cond = np.load(cond_npy_path, allow_pickle=True).item()

    # ── Detect object_type ────────────────────────────────────────────────────
    if object_type is None:
        object_type = infer_object_type_from_filename(npy_path, valid_types=cond.keys())
        if object_type is None:
            raise ValueError(
                f"Cannot auto-detect object_type from '{os.path.basename(npy_path)}'.\n"
                f"  Available: {list(cond.keys())}\n"
                f"  Pass --object-type explicitly."
            )
        print(f"Auto-detected object_type: {object_type}")
    elif object_type not in cond:
        raise ValueError(
            f"object_type '{object_type}' not found in cond.npy.\n"
            f"  Available: {list(cond.keys())}"
        )

    # ── Resolve T-pose mesh / Build context ──────────────────────────────────
    cond_entry = cond[object_type]
    if skeleton_only and tpose_mesh is None:
        raw = np.load(npy_path)
        ctx = _build_skeleton_only_context(raw, object_type, cond_entry)
        features = ctx["features"]
        feature_joint_names: list[str] = ctx["joint_names"]
        export_joint_names: list[str] = ctx["export_joint_names"]
        parents = ctx["parents"]
        offsets_hml = ctx["offsets"]
        tpose_rest_rotations = ctx["tpose_rest_rotations"]
        export_parents = np.asarray(ctx["export_parents"], dtype=np.int32)
        export_offsets = np.asarray(ctx["export_offsets"], dtype=np.float32)
        export_rest_rotations = np.asarray(ctx["export_rest_rotations"], dtype=np.float32)
        scale_factor_val = ctx["scale_factor"]
        orientation_quat_val = ctx["orientation_quat"]
        tpose_mesh_resolved = None
    else:
        raw = np.load(npy_path)
        restore_ctx = _build_restore_context(
            raw,
            object_type,
            tpose_mesh,
            cond_entry=cond_entry,
        )

        features = restore_ctx["features"]
        feature_joint_names: list[str] = restore_ctx["joint_names"]
        export_joint_names: list[str] = restore_ctx["export_joint_names"]
        parents = restore_ctx["parents"]
        offsets_hml = restore_ctx["offsets"]
        tpose_rest_rotations = restore_ctx["tpose_rest_rotations"]
        export_parents = np.asarray(restore_ctx["export_parents"], dtype=np.int32)
        export_offsets = np.asarray(restore_ctx["export_offsets"], dtype=np.float32)
        export_rest_rotations = np.asarray(restore_ctx["export_rest_rotations"], dtype=np.float32)
        scale_factor_val = float(restore_ctx["scale_factor"])
        orientation_quat_val = np.asarray(restore_ctx["orientation_quat"], dtype=np.float64)
        # skeleton-only native with tpose_mesh: import the source armature and
        # drop only its meshes at export time. This preserves the same
        # armature-object scale / local-offset decomposition as a skinned GLB.
        # HML skeleton-only remains mesh-free/canonical, as before.
        tpose_mesh_resolved = (
            tpose_mesh
            if (not skeleton_only or restore_space == "native")
            else None
        )

    translation_root_index = None

    # ── Resolve FPS ─────────────────────────────────────────────────────
    if fps is None:
        fps = 30.0

    print(f"Skeleton: {len(feature_joint_names)} joints, root='{feature_joint_names[0]}'")

    F, J, C = features.shape
    if J != len(feature_joint_names):
        raise ValueError(
            f"NPY has J={J} joints but cond.npy has {len(feature_joint_names)} joints for '{object_type}'."
        )
    if C != 13:
        raise ValueError(f"Expected 13 channels per joint, got {C}.")

    print(f"NPY: {F} frames, {J} joints, {C} channels")

    print(f"T-pose preprocessing scale_factor: {scale_factor_val:.6f}")
    if root_translation_xz is None:
        print("Root translation XZ: keeping centred preprocessed placement")
    else:
        coerced_root_translation_xz = _coerce_root_translation_xz(root_translation_xz)
        print(
            "Root translation XZ override: "
            f"[{coerced_root_translation_xz[0]:.6f}, {coerced_root_translation_xz[2]:.6f}]"
        )
        root_translation_xz = coerced_root_translation_xz

    if not skeleton_only:
        _warn_on_missing_mesh_joints(
            export_joint_names,
            tpose_mesh_resolved,
            mesh_bone_names=restore_ctx["mesh_bone_names"],
        )

    # ── Recover Animation (in HML feature space) ──────────────────────────────
    print("Recovering feature-space animation from NPY...")
    recovered_feature_anim, has_animated_pos = recover_from_features(
        raw,
        parents,
        offsets_hml,
        translation_root_index=translation_root_index,
    )
    print(f"Recovered: {recovered_feature_anim.shape[0]} frames")
    translation_root_index = find_translation_root(recovered_feature_anim)

    print("Recovering processed animation channels for export...")
    export_anim = recover_processed_animation_from_feature_animation(
        recovered_feature_anim,
        tpose_rest_rotations,
    )

    if restore_space == "hml":
        print("HML space: staying in HML space (skipping inverse preprocess transform)")
    else:
        export_anim = _invert_preprocess_transform(
            export_anim,
            scale_factor=scale_factor_val,
            root_translation_xz=root_translation_xz,
            orientation_quat=orientation_quat_val,
        )

    if fullbody_ik:
        print(f"Force full-body IK reconstruction on export skeleton (stretch_factor={stretch_factor:.2f})...")
        export_anim, ik_mean_error, ik_max_error = rebuild_fullbody_animation_with_ik(
            export_anim,
            rigid_offsets=export_offsets,
            rigid_parents=export_parents,
            preserved_position_indices=[translation_root_index],
            preserved_rotation_indices=[translation_root_index],
            stretch_factor=stretch_factor,
        )
        print(
            "Full-body IK residual joint error: "
            f"mean={ik_mean_error:.6f}, max={ik_max_error:.6f}"
        )
        print(
            f"Preserving translation-root local pose during IK: "
            f"{export_joint_names[translation_root_index]} (index {translation_root_index})"
        )
    else:
        print("Skipping IK (use --fullbody-ik to enable).")

    # ── Resample in time (optional) ─────────────────────────────────────────
    # Off by default: the GLB keeps the NPY's native frame count at ``fps``.
    # When resample_fps is set, retime the recovered motion from ``fps`` (native)
    # to resample_fps and write the GLB at resample_fps, so downstream consumers
    # (e.g. render_images_from_glb.py) can render every keyframe as-is instead of
    # resampling at render time.
    output_fps = fps
    if resample_fps is not None and resample_fps > 0:
        if abs(resample_fps - fps) < 1e-6:
            print(
                f"resample_fps ({resample_fps}) equals fps ({fps}), "
                "skipping resample."
            )
        else:
            src_frames = export_anim.shape[0]
            frame_times = _resample_frame_indices(
                src_frames, fps, resample_fps, min_length=resample_min_length
            )
            export_anim = _resample_animation(export_anim, frame_times)
            output_fps = resample_fps
            print(
                f"Resampled motion {src_frames} -> {export_anim.shape[0]} frames "
                f"({fps:g}fps -> {resample_fps:g}fps"
                + (f", min_length={resample_min_length}" if resample_min_length else "")
                + ")"
            )

    # ── Reconcile skeleton-only export scale ────────────────────────────────
    # With a T-pose mesh in native mode, skeleton-only uses the imported source
    # armature as the export rig and removes only mesh objects before writing
    # the GLB. That keeps the same object-scale/local-offset decomposition as a
    # skinned export, which matters for downstream tools that copy local TRS by
    # node name. In HML mode we still create a canonical mesh-free armature, so
    # bring its rest offsets into normalized HML scale to match the motion.
    skeleton_only_from_tpose = skeleton_only and tpose_mesh is not None
    if skeleton_only_from_tpose and restore_space == "native":
        print(
            "Skeleton-only (native): using the T-pose armature as export rig "
            "and omitting meshes, preserving source node scale/local offsets"
        )
    elif skeleton_only_from_tpose and restore_space == "hml" and abs(scale_factor_val - 1.0) > 1e-12:
        export_offsets = (export_offsets * scale_factor_val).astype(np.float32)
        print(
            "Skeleton-only (hml): rescaled skeleton offsets by scale_factor "
            f"{scale_factor_val:.6f} into normalized HML space to match the motion"
        )

    # ── Build skeleton for exporter ─────────────────────────────────────────
    skeleton = build_skeleton(
        export_joint_names,
        export_offsets,
        export_parents,
        export_rest_rotations,
    )

    joint_rotations, root_translation, root_rotation, bone_translations = (
        animation_to_exporter_inputs(export_anim, skeleton)
    )

    os.makedirs(os.path.dirname(output_glb) or ".", exist_ok=True)

    # ── HML reverse-alignment (restore_space="hml") ─────────────────────────
    # In "native" mode the recovered animation is exported in the T-pose mesh's
    # native space. In "hml" mode we instead reverse-align the rig onto the NPY
    # by re-applying the forward preprocessing similarity (scale + orientation)
    # to the imported mesh/armature, so the GLB lands in the same space as the
    # NPY / corresponding processed BVH.
    # When there is no mesh (tpose_mesh_resolved is None), reverse-alignment
    # is unnecessary — the skeleton is already in the correct space.
    global_similarity = None
    if restore_space == "hml" and tpose_mesh_resolved is not None:
        hml_scale = scale_factor_val
        hml_orientation = np.asarray(orientation_quat_val, dtype=np.float64).reshape(-1)
        print(
            "Reverse-aligning rig into HML/npy space "
            f"(scale={float(hml_scale):.6f}, orientation_quat set)"
        )
        global_similarity = (hml_scale, hml_orientation)

    # ── Export GLB ──────────────────────────────────────────────────────────
    exporter = AnimationExporter(skeleton, fps=output_fps)
    if skeleton_only:
        print(f"Exporting skeleton-only GLB → {output_glb}")
    else:
        print(f"Exporting skinned GLB → {output_glb}")
    exporter.export_glb(
        joint_rotations,
        root_translation,
        root_rotation,
        output_glb,
        mesh_path=tpose_mesh_resolved,
        bone_translations=bone_translations,
        global_similarity=global_similarity,
        use_image_search=use_image_search,
        export_mesh=not skeleton_only,
        rename_bones_to_canonical=(restore_space == "hml"),
        prune_unmapped_bones=(restore_space == "hml"),
    )

    return os.path.abspath(output_glb)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Restore a preprocessed Anytop NPY motion to a GLB.\n"
            "  Default: skeleton-only GLB in HML space, no mesh access."
            "\n"
            "  Pass --tpose-mesh <file> for a skinned GLB using that mesh as the"
            "\n"
            "  rig/skin source."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--npy", required=True,
        help="Path to the preprocessed .npy motion file.",
    )
    parser.add_argument(
        "--tpose-mesh",
        default=None,
        help=(
            "Path to the T-pose FBX/GLB/GLTF that provides skin weights + armature "
            "for a skinned GLB.  If omitted, restore stays skeleton-only."
        ),
    )
    parser.add_argument(
        "--output-glb",
        default=None,
        help=(
            "Output GLB path.  Defaults to outputs/restore_glb_from_npy/<stem>.glb "
            "relative to the Anytop directory."
        ),
    )
    parser.add_argument(
        "--cond-npy",
        default=None,
        help=f"Path to cond.npy.  Default: {_DEFAULT_COND_NPY}",
    )
    parser.add_argument(
        "--object-type",
        default=None,
        help=(
            "Character type key in cond.npy (e.g. 'Horse').  "
            "Auto-detected from the NPY filename if not specified."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Animation frame rate.  Defaults to 30 if not specified.",
    )
    parser.add_argument(
        "--root-translation-xz",
        type=float,
        nargs=2,
        metavar=("X", "Z"),
        default=None,
        help=(
            "Explicit XZ translation to add back during restore. When omitted, "
            "the restored clip stays in centred preprocessed space."
        ),
    )
    parser.add_argument(
        "--fullbody-ik",
        action="store_true",
        default=False,
        help=(
            "Perform full-body IK reconstruction on the raw export skeleton "
            "after recovering the animation.  Disabled by default."
        ),
    )
    parser.add_argument(
        "--stretch-factor",
        type=float,
        default=DEFAULT_IK_STRETCH_FACTOR,
        help=(
            "Allowed bone-length elasticity ratio for IK.  Each edge may "
            f"stretch/compress by ±stretch_factor (default: {DEFAULT_IK_STRETCH_FACTOR}, "
            "i.e. ±10 %).  Only effective when --fullbody-ik is enabled."
        ),
    )
    parser.add_argument(
        "--restore-space",
        choices=("native", "hml"),
        default="native",
        help=(
            "Output coordinate space. 'native' (default) aligns the animation to the "
            "T-pose mesh's original orientation/scale/translation. 'hml' reverse-aligns "
            "the T-pose mesh onto the NPY so the GLB keeps the NPY's orientation, "
            "scale, and centered placement (like the corresponding processed BVH)."
        ),
    )
    parser.add_argument(
        "--use-image-search",
        action="store_true",
        default=False,
        help=(
            "Resolve textures for the skinned mesh: the FBX importer searches "
            "directories near the source mesh, then a fallback wires a matching "
            "diffuse/alpha texture from the mesh's tex/ folder onto any main "
            "character mesh still missing one. Disabled by default."
        ),
    )
    parser.add_argument(
        "--resample-fps",
        type=float,
        default=None,
        help=(
            "Resample the recovered motion in time from its native rate (--fps) to "
            "this rate before export, and write the GLB at this rate. Disabled by "
            "default (the GLB keeps the NPY's native frame count)."
        ),
    )
    parser.add_argument(
        "--resample-min-length",
        type=int,
        default=None,
        help=(
            "When --resample-fps is set, time-stretch the resampled clip so it has "
            "at least this many frames (interpolated, no looping). Default: no minimum."
        ),
    )
    parser.add_argument(
        "--skeleton-only",
        action="store_true",
        default=False,
        help=(
            "Export a skeleton-only GLB (no mesh, no skinning).  "
            "Without --tpose-mesh, uses cond.npy metadata and forces "
            "--restore-space to hml (the default automatic fallback).  "
            "With --tpose-mesh, the mesh armature supplies rest rotations "
            "and --restore-space is honoured."
        ),
    )
    parser.add_argument(
        "--check-bone-length",
        action="store_true",
        default=False,
        help=(
            "Run check_bone_length_drift on the restored GLB after export. "
            "Disabled by default."
        ),
    )

    args = parser.parse_args()

    if not os.path.isfile(args.npy):
        parser.error(f"NPY file not found: {args.npy}")
    if not args.npy.lower().endswith('.npy'):
        parser.error(
            f"Expected a .npy file, got: {args.npy}\n"
            f"  This tool restores preprocessed NPY motion features, not raw BVH/FBX files."
        )
    if args.tpose_mesh is not None and not os.path.isfile(args.tpose_mesh):
        parser.error(f"T-pose mesh not found: {args.tpose_mesh}")

    if args.output_glb is None:
        stem = os.path.splitext(os.path.basename(args.npy))[0]
        args.output_glb = os.path.join(
            ANYTOP_DIR, "outputs", "restore_glb_from_npy", f"{stem}.glb"
        )

    cond_npy_path = args.cond_npy or _DEFAULT_COND_NPY
    if not os.path.isfile(cond_npy_path):
        parser.error(
            f"cond.npy not found: {cond_npy_path}\n"
            "Use --cond-npy to specify a custom path."
        )

    print(f"NPY           : {args.npy}")
    print(f"T-pose mesh   : {args.tpose_mesh}")
    print(f"Output GLB    : {args.output_glb}")
    print(f"cond.npy      : {cond_npy_path}")
    print(f"FPS           : {args.fps or '(auto)'}")
    print(f"Root XZ       : {args.root_translation_xz or '(centered default)'}")
    print(f"Stretch factor: {args.stretch_factor}")
    print(f"Restore space : {args.restore_space}")
    print(f"Skeleton-only : {args.skeleton_only}")
    print()

    restore_glb(
        npy_path=args.npy,
        output_glb=args.output_glb,
        tpose_mesh=args.tpose_mesh,
        cond_npy=cond_npy_path,
        object_type=args.object_type,
        fps=args.fps,
        root_translation_xz=args.root_translation_xz,
        fullbody_ik=args.fullbody_ik,
        stretch_factor=args.stretch_factor,
        restore_space=args.restore_space,
        use_image_search=args.use_image_search,
        resample_fps=args.resample_fps,
        resample_min_length=args.resample_min_length,
        skeleton_only=args.skeleton_only,
    )

    if args.check_bone_length and not args.skeleton_only:
        _run_bone_length_check(args.output_glb, cond_npy_path, args.object_type)


def _run_bone_length_check(glb_path: str, cond_npy: str, object_type: str | None) -> None:
    """Run check_bone_length_drift.py on the restored GLB."""
    check_script = os.path.join(os.path.dirname(__file__), "check_bone_length_drift.py")
    if not os.path.isfile(check_script):
        print(f"\n[check-bone-length] Script not found: {check_script}")
        return

    print(f"\n{'='*60}")
    print(f"[check-bone-length] Running bone length drift check on: {glb_path}")
    print(f"{'='*60}\n")

    # Execute the check script in the current Python environment
    python_exe = sys.executable
    cmd = [python_exe, check_script, "--input", glb_path, "--cond-npy", cond_npy]
    if object_type is not None:
        cmd.extend(["--object-type", object_type])
    result = subprocess.run(
        cmd,
        cwd=os.path.dirname(check_script),
    )
    if result.returncode != 0:
        print(f"\n[check-bone-length] check_bone_length_drift exited with code {result.returncode}")


if __name__ == "__main__":
    main()
