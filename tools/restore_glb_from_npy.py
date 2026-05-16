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

Metadata is resolved from two sources in descending priority:

    1. cond.npy entry  — dataset-wide metadata indexed by object_type
                         (e.g. "Horse"), stores joints_names, parents,
                         offsets, rest_rotations, etc.
    2. T-pose FBX fallback — computed on demand via
                              get_common_features_from_T_pose(); most
                              expensive, only loaded when cond.npy lacks
                              the needed fields.

Note: locomotion XZ stripped during preprocessing cannot be recovered from a
plain feature tensor alone. Non-locomotion clips also stay in their centred
preprocessed space unless an explicit root-translation XZ override is passed
during restore.

Usage:
        # Using FBX T-pose (explicit)
        python tools/restore_glb_from_npy.py \
                --npy "D:/AI/.../Horse___RunToStop_29.npy" \
                --tpose-mesh "D:/AI/.../HorseALL-TPOSE.fbx" \
                --output-glb "outputs/Horse___RunToStop_29.glb"

        # Using T-pose path saved in cond.npy (no --tpose-mesh needed)
        python tools/restore_glb_from_npy.py \
                --npy "D:/AI/.../Horse___RunToStop_29.npy" \
                --output-glb "outputs/Horse___RunToStop_29.glb"

"""

import argparse
import importlib.util
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

# ── Fix utils namespace conflict (same workaround as test_fbx_npy_glb_roundtrip.py) ──

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
from Anytop.utils.roundtrip_common import _load_fbx_skeleton_metadata
from Anytop.motion_lib.FBX import _collapse_root_skeleton

# ── Default cond.npy path ─────────────────────────────────────────────────────

_DEFAULT_COND_NPY = os.path.realpath(
    os.path.join(ANYTOP_DIR, "dataset", "truebones", "zoo", "truebones_processed", "cond.npy")
)

_FULLBODY_IK_ITERATIONS = 2

def _load_tpose_restore_metadata(
    tpose_mesh: str,
    object_type: str,
    cond_entry: dict | None = None,
) -> dict[str, object]:
    from data_loaders.truebones.truebones_utils.motion_process import get_common_features_from_T_pose, TPoseFeatures

    tpose_lower = tpose_mesh.lower()
    if not tpose_lower.endswith(('.fbx', '.glb', '.gltf')):
        raise ValueError(f"Unsupported T-pose mesh format: {tpose_mesh} - expected .fbx, .glb, or .gltf")

    helper_joint_indices = []
    if cond_entry is not None:
        helper_joint_indices = list(cond_entry.get("helper_joint_indices") or [])
    tp_kwargs = {}
    if helper_joint_indices:
        tp_kwargs["augment_leaf_rotation_helpers"] = True
        cond_parents = cond_entry.get("parents") if cond_entry is not None else None
        tp_kwargs["max_joints"] = len(cond_parents) if cond_parents is not None else 0

    tp: TPoseFeatures = get_common_features_from_T_pose(tpose_mesh, object_type, **tp_kwargs)
    raw_joint_names, raw_parents, raw_offsets, raw_rest_rotations = _load_fbx_skeleton_metadata(tpose_mesh)
    raw_parents = np.asarray(raw_parents, dtype=np.int32)
    raw_offsets = np.asarray(raw_offsets, dtype=np.float32)
    raw_rest_rotations = np.asarray(raw_rest_rotations, dtype=np.float32)
    collapsed_joint_names, collapsed_parents, collapsed_offsets, collapsed_rest_rotations = (
        _collapse_root_skeleton(
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
        "helper_metadata": dict(tp.helper_metadata),
    }


def _remap_joint_array(
    source_names: list[str],
    target_names: list[str],
    values: np.ndarray,
    label: str,
) -> np.ndarray:
    if list(source_names) == list(target_names):
        return np.asarray(values)

    source_index = {name: index for index, name in enumerate(source_names)}
    missing = [name for name in target_names if name not in source_index]
    if missing:
        preview = missing[:10]
        suffix = "..." if len(missing) > 10 else ""
        raise ValueError(f"T-pose mesh is missing {label} joints: {preview}{suffix}")

    reordered = [values[source_index[name]] for name in target_names]
    return np.asarray(reordered)


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
        mesh_bone_names, _mesh_parents, _mesh_offsets, _mesh_rest_rots = _load_fbx_skeleton_metadata(
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
    cond_entry: dict | None = None,
) -> dict[str, object]:
    features = np.asarray(raw_npy)
    feature_joint_count = int(features.shape[1]) if features.ndim >= 2 else 0

    # ── Check availability across two tiers ─────────────────────────────
    #   Tier 1: cond.npy entry (dataset-wide metadata)
    #   Tier 2: T-pose FBX (fallback, most expensive)
    cond_has_skeleton = cond_entry is not None and all(
        key in cond_entry for key in ("joints_names", "parents", "offsets")
    )
    cond_has_scale = cond_entry is not None and "scale_factor" in cond_entry

    # Determine the effective joint count from the NPY and align all
    # metadata (cond entry, T-pose) to that count.  Helpers are always
    # appended at the end of the joint array.
    tpose_meta = _load_tpose_restore_metadata(tpose_mesh, object_type, cond_entry=cond_entry)

    # ── Skeleton info (joint_names / parents / offsets) ─────────────────
    if cond_has_skeleton:
        joint_names = list(cond_entry["joints_names"])
        parents = np.asarray(cond_entry["parents"], dtype=np.int32)
        offsets = np.asarray(cond_entry["offsets"], dtype=np.float32)
    else:
        joint_names = list(tpose_meta["joint_names"])
        parents = np.asarray(tpose_meta["parents"], dtype=np.int32)
        offsets = np.asarray(tpose_meta["offsets"], dtype=np.float32)

    original_joint_count = len(joint_names)
    helper_joint_indices = []
    helper_source_leaf_indices = []
    if cond_entry is not None:
        original_joint_count = int(cond_entry.get("original_joint_count", len(joint_names)))
        helper_joint_indices = [int(index) for index in list(cond_entry.get("helper_joint_indices") or [])]
        helper_source_leaf_indices = [
            int(index) for index in list(cond_entry.get("helper_source_leaf_indices") or [])
        ]
    if original_joint_count <= 0 or original_joint_count > len(joint_names):
        raise ValueError(
            f"original_joint_count must be within [1, {len(joint_names)}], got {original_joint_count}"
        )

    # Align feature joint count with available metadata.
    # Helpers are appended at the end, so we can safely trim.
    if feature_joint_count == original_joint_count and len(joint_names) != original_joint_count:
        # NPY has no helpers — trim everything to original skeleton.
        joint_names = list(joint_names[:original_joint_count])
        parents = np.asarray(parents[:original_joint_count], dtype=np.int32)
        offsets = np.asarray(offsets[:original_joint_count], dtype=np.float32)
        helper_joint_indices = []
        helper_source_leaf_indices = []
    elif original_joint_count < feature_joint_count <= len(joint_names):
        # NPY has some helpers (possibly fewer than cond.npy).
        # Trim cond metadata to match the NPY's joint count.
        joint_names = list(joint_names[:feature_joint_count])
        parents = np.asarray(parents[:feature_joint_count], dtype=np.int32)
        offsets = np.asarray(offsets[:feature_joint_count], dtype=np.float32)
        helper_joint_indices = [
            i for i in helper_joint_indices if i < feature_joint_count
        ]
        helper_source_leaf_indices = [
            i for i in helper_source_leaf_indices if i < feature_joint_count
        ]
    elif feature_joint_count not in (0, len(joint_names)):
        raise ValueError(
            f"NPY has J={feature_joint_count} joints but restore metadata resolves to {len(joint_names)} "
            f"joints for '{object_type}' (original_joint_count={original_joint_count})."
        )

    # ── T-pose rest rotations ───────────────────────────────────────────
    # The T-pose may have a different helper budget than the NPY (e.g., the
    # cond.npy was built with a different _is_terminal_leaf_name filter or
    # max_joints budget).  Build rest rotations by matching what the T-pose
    # provides and filling identity quaternions for any helper joints the
    # T-pose doesn't know about (helpers are zero-offset, identity-rotation dummies).
    tpose_joint_names = list(tpose_meta["joint_names"])
    tpose_rest_src = np.asarray(tpose_meta["tpose_rest_rotations"], dtype=np.float32)
    tpose_name_index = {name: idx for idx, name in enumerate(tpose_joint_names)}
    tpose_rest_rotations = np.zeros((len(joint_names), 4), dtype=np.float32)
    tpose_rest_rotations[:, 0] = 1.0  # default identity quaternion
    for j, name in enumerate(joint_names):
        if name in tpose_name_index:
            tpose_rest_rotations[j] = tpose_rest_src[tpose_name_index[name]]

    # ── Scale factor ────────────────────────────────────────────────────
    scale_factor = None
    if cond_has_scale:
        scale_factor = float(cond_entry["scale_factor"])
    elif tpose_meta is not None and tpose_meta.get("scale_factor") is not None:
        scale_factor = float(tpose_meta["scale_factor"])

    orientation_quat = None
    if cond_entry is not None and cond_entry.get("orientation_quat") is not None:
        orientation_quat = np.asarray(cond_entry["orientation_quat"], dtype=np.float64)
    elif tpose_meta.get("orientation_quat") is not None:
        orientation_quat = np.asarray(tpose_meta["orientation_quat"], dtype=np.float64)

    feature_rotation_channel_mask = _bare_feature_rotation_channel_mask(parents)
    export_joint_names = list(joint_names[:original_joint_count])
    export_rotation_channel_mask = np.asarray(
        feature_rotation_channel_mask[:original_joint_count],
        dtype=bool,
    )

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
        "original_joint_count": original_joint_count,
        "helper_joint_indices": helper_joint_indices,
        "helper_source_leaf_indices": helper_source_leaf_indices,
        "rotation_channel_mask": export_rotation_channel_mask,
        "mesh_bone_names": list(tpose_meta["raw_joint_names"]),
        "export_parents": export_parents,
        "export_offsets": export_offsets,
        "export_rest_rotations": export_rest_rotations,
    }


def _strip_appended_helper_joints(
    animation,
    *,
    original_joint_count: int,
):
    from motion_lib.Animation import Animation

    if original_joint_count <= 0:
        raise ValueError(f"original_joint_count must be positive, got {original_joint_count}")
    if animation.shape[1] < original_joint_count:
        raise ValueError(
            f"Animation joint count {animation.shape[1]} is smaller than original_joint_count {original_joint_count}"
        )
    if animation.shape[1] == original_joint_count:
        return animation

    return Animation(
        animation.rotations[:, :original_joint_count].copy(),
        animation.positions[:, :original_joint_count, :].copy(),
        animation.orients[:original_joint_count].copy(),
        animation.offsets[:original_joint_count].copy(),
        animation.parents[:original_joint_count].copy(),
    )


def _bare_feature_rotation_channel_mask(parents: np.ndarray) -> np.ndarray:
    parents = np.asarray(parents, dtype=np.int32)
    joint_count = len(parents)
    if joint_count == 0:
        return np.zeros((0,), dtype=bool)

    child_counts = np.bincount(parents[parents >= 0], minlength=joint_count)
    rotation_channel_mask = child_counts > 0
    rotation_channel_mask[0] = True
    return rotation_channel_mask


def _clamp_unobservable_joint_positions_to_rest(
    animation,
    *,
    rest_offsets: np.ndarray,
    rotation_channel_mask: np.ndarray,
):
    from motion_lib.Animation import Animation

    rest_offsets = np.asarray(rest_offsets, dtype=np.float64)
    rotation_channel_mask = np.asarray(rotation_channel_mask, dtype=bool)

    if rest_offsets.shape != (animation.shape[1], 3):
        raise ValueError(
            f"rest_offsets must have shape ({animation.shape[1]}, 3), got {rest_offsets.shape}"
        )
    if rotation_channel_mask.shape != (animation.shape[1],):
        raise ValueError(
            f"rotation_channel_mask must have shape ({animation.shape[1]},), got {rotation_channel_mask.shape}"
        )
    if np.all(rotation_channel_mask):
        return animation

    clamped_positions = np.asarray(animation.positions, dtype=np.float64).copy()
    clamped_positions[:, ~rotation_channel_mask, :] = rest_offsets[~rotation_channel_mask][None, :, :]
    return Animation(
        animation.rotations.copy(),
        clamped_positions,
        animation.orients.copy(),
        animation.offsets.copy(),
        animation.parents.copy(),
    )


def _coerce_root_translation_xz(root_translation_xz: np.ndarray) -> np.ndarray:
    root_translation_xz = np.asarray(root_translation_xz, dtype=np.float64).reshape(-1)
    if root_translation_xz.size == 3:
        return root_translation_xz
    if root_translation_xz.size == 2:
        return np.array([root_translation_xz[0], 0.0, root_translation_xz[1]], dtype=np.float64)
    raise ValueError(
        f"root_translation_xz must have shape (2,) or (3,), got {root_translation_xz.shape}"
    )


def _normalize_joint_index_selection(
    indices: np.ndarray | list[int] | None,
    joint_count: int,
    *,
    label: str,
) -> np.ndarray:
    if indices is None:
        return np.zeros((0,), dtype=np.int32)

    normalized = np.asarray(indices, dtype=np.int32).reshape(-1)
    if normalized.size == 0:
        return normalized
    if np.any((normalized < 0) | (normalized >= joint_count)):
        raise ValueError(f"{label} must be within [0, {joint_count - 1}]")
    return np.unique(normalized)


def _validate_ik_mode_selection(fullbody_ik: bool) -> None:
    pass


def _resolve_ik_rebuild_inputs(
    target_anim,
    *,
    rigid_offsets: np.ndarray | None,
    rigid_parents: np.ndarray | None,
    preserved_position_indices: np.ndarray | list[int] | None,
    preserved_rotation_indices: np.ndarray | list[int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    parents = np.asarray(
        target_anim.parents if rigid_parents is None else rigid_parents,
        dtype=np.int32,
    )
    rest_offsets = np.asarray(
        target_anim.offsets if rigid_offsets is None else rigid_offsets,
        dtype=np.float64,
    )
    if target_anim.shape[1] != len(parents):
        raise ValueError(
            f"rigid skeleton joint count {len(parents)} does not match animation joint count {target_anim.shape[1]}"
        )
    if rest_offsets.shape != (len(parents), 3):
        raise ValueError(
            f"rest_offsets must have shape ({len(parents)}, 3), got {rest_offsets.shape}"
        )

    preserved_position_indices = _normalize_joint_index_selection(
        preserved_position_indices,
        len(parents),
        label="preserved_position_indices",
    )
    preserved_rotation_indices = _normalize_joint_index_selection(
        preserved_rotation_indices,
        len(parents),
        label="preserved_rotation_indices",
    )

    root_indices = np.flatnonzero(parents < 0)
    if root_indices.size != 1:
        raise ValueError(f"Expected exactly one root joint, got {root_indices.size}")

    return (
        parents,
        rest_offsets,
        preserved_position_indices,
        preserved_rotation_indices,
        int(root_indices[0]),
    )


def _build_selective_ik_seed_positions(
    target_anim,
    *,
    rest_offsets: np.ndarray,
    root_index: int,
    rigidize_joint_mask: np.ndarray,
    preserved_position_indices: np.ndarray,
) -> np.ndarray:
    target_positions = np.asarray(target_anim.positions, dtype=np.float64)
    rigidize_joint_mask = np.asarray(rigidize_joint_mask, dtype=bool)
    if rigidize_joint_mask.shape != target_positions.shape[:2]:
        raise ValueError(
            f"rigidize_joint_mask must have shape {target_positions.shape[:2]}, got {rigidize_joint_mask.shape}"
        )

    seed_positions = target_positions.copy()
    if np.any(rigidize_joint_mask):
        rigid_template = np.broadcast_to(rest_offsets[None, :, :], target_positions.shape)
        seed_positions[rigidize_joint_mask] = rigid_template[rigidize_joint_mask]

    seed_positions[:, root_index, :] = target_positions[:, root_index, :]
    if preserved_position_indices.size > 0:
        seed_positions[:, preserved_position_indices, :] = target_positions[:, preserved_position_indices, :]
    return seed_positions


def _run_basic_inverse_kinematics_with_constraints(
    animation,
    target_global_positions: np.ndarray,
    *,
    frozen_rotation_indices: np.ndarray | list[int] | None = None,
    active_child_mask: np.ndarray | None = None,
    iterations: int = _FULLBODY_IK_ITERATIONS,
):
    import importlib

    from motion_lib.Quaternions import Quaternions

    animation_module = importlib.import_module("motion_lib.Animation")
    animation_structure = importlib.import_module("motion_lib.AnimationStructure")

    frozen_rotation_indices = _normalize_joint_index_selection(
        frozen_rotation_indices,
        animation.shape[1],
        label="frozen_rotation_indices",
    )
    if active_child_mask is not None:
        active_child_mask = np.asarray(active_child_mask, dtype=bool)
        if active_child_mask.shape != (animation.shape[0], animation.shape[1]):
            raise ValueError(
                f"active_child_mask must have shape {(animation.shape[0], animation.shape[1])}, got {active_child_mask.shape}"
            )

    frozen_rotation_lookup = {int(index) for index in frozen_rotation_indices.tolist()}
    children = animation_structure.children_list(animation.parents)

    for _iteration in range(iterations):
        for joint_index in animation_structure.joints(animation.parents):
            if joint_index in frozen_rotation_lookup:
                continue

            child_indices = np.asarray(children[joint_index], dtype=np.int32)
            if child_indices.size == 0:
                continue

            joint_active_child_mask = None
            if active_child_mask is not None:
                joint_active_child_mask = active_child_mask[:, child_indices]
                if not np.any(joint_active_child_mask):
                    continue

            anim_transforms = animation_module.transforms_global(animation)
            anim_positions = anim_transforms[:, :, :3, 3]
            anim_rotations = Quaternions.from_transforms(anim_transforms[:, :, :3, :3])

            joint_dirs = anim_positions[:, child_indices] - anim_positions[:, np.newaxis, joint_index]
            target_dirs = (
                target_global_positions[:, child_indices]
                - target_global_positions[:, np.newaxis, joint_index]
            )

            if child_indices.size > 1 and (joint_dirs == 0).all() and (target_dirs == 0).all():
                continue

            joint_lengths = np.sqrt(np.sum(joint_dirs ** 2.0, axis=-1)) + 1e-20
            target_lengths = np.sqrt(np.sum(target_dirs ** 2.0, axis=-1)) + 1e-20

            joint_dirs = joint_dirs / joint_lengths[:, :, np.newaxis]
            target_dirs = target_dirs / target_lengths[:, :, np.newaxis]

            angles = np.arccos(np.sum(joint_dirs * target_dirs, axis=2).clip(-1, 1))
            axes = np.cross(joint_dirs, target_dirs)
            axes = -anim_rotations[:, joint_index, np.newaxis] * axes

            valid_directions = (joint_lengths > 1e-4)[0]
            if not np.any(valid_directions):
                continue

            rotations = Quaternions.from_angle_axis(angles, axes)
            if joint_active_child_mask is None:
                if rotations.shape[1] == 1:
                    averaged_rotation = rotations[:, 0]
                else:
                    averaged_rotation = Quaternions.exp(
                        rotations[:, valid_directions].log().mean(axis=-2)
                    )

                animation.rotations[:, joint_index] = (
                    animation.rotations[:, joint_index] * averaged_rotation
                )
                continue

            active_frames = np.flatnonzero(np.any(joint_active_child_mask, axis=1))
            for frame_index in active_frames.tolist():
                frame_child_mask = (
                    joint_active_child_mask[frame_index]
                    & (joint_lengths[frame_index] > 1e-4)
                    & (target_lengths[frame_index] > 1e-4)
                )
                if not np.any(frame_child_mask):
                    continue

                frame_slice = slice(frame_index, frame_index + 1)
                if np.count_nonzero(frame_child_mask) == 1:
                    child_slot = int(np.flatnonzero(frame_child_mask)[0])
                    averaged_rotation = rotations[frame_slice, child_slot]
                else:
                    averaged_rotation = Quaternions.exp(
                        rotations[frame_index, frame_child_mask].log().mean(axis=0, keepdims=True)
                    )

                animation.rotations[frame_slice, joint_index] = (
                    animation.rotations[frame_slice, joint_index] * averaged_rotation
                )

    return animation


def _rebuild_animation_with_ik(
    target_anim,
    *,
    rigid_offsets: np.ndarray | None = None,
    rigid_parents: np.ndarray | None = None,
    preserved_position_indices: np.ndarray | list[int] | None = None,
    preserved_rotation_indices: np.ndarray | list[int] | None = None,
    rigidize_joint_mask: np.ndarray | None = None,
    active_child_mask: np.ndarray | None = None,
    iterations: int = _FULLBODY_IK_ITERATIONS,
) -> tuple[object, float, float]:
    from motion_lib.Animation import Animation, positions_global

    parents, rest_offsets, preserved_position_indices, preserved_rotation_indices, root_index = (
        _resolve_ik_rebuild_inputs(
            target_anim,
            rigid_offsets=rigid_offsets,
            rigid_parents=rigid_parents,
            preserved_position_indices=preserved_position_indices,
            preserved_rotation_indices=preserved_rotation_indices,
        )
    )
    if rigidize_joint_mask is None:
        rigidize_joint_mask = np.zeros(target_anim.shape[:2], dtype=bool)

    rigid_positions = _build_selective_ik_seed_positions(
        target_anim,
        rest_offsets=rest_offsets,
        root_index=root_index,
        rigidize_joint_mask=rigidize_joint_mask,
        preserved_position_indices=preserved_position_indices,
    )

    ik_seed = Animation(
        target_anim.rotations.copy(),
        rigid_positions,
        target_anim.orients.copy(),
        rest_offsets.copy(),
        parents.copy(),
    )

    target_global_positions = positions_global(target_anim).astype(np.float64, copy=False)
    rebuilt_anim = _run_basic_inverse_kinematics_with_constraints(
        ik_seed,
        target_global_positions,
        frozen_rotation_indices=preserved_rotation_indices,
        active_child_mask=active_child_mask,
        iterations=iterations,
    )
    rebuilt_global_positions = positions_global(rebuilt_anim).astype(np.float64, copy=False)
    per_joint_error = np.linalg.norm(rebuilt_global_positions - target_global_positions, axis=-1)
    return rebuilt_anim, float(per_joint_error.mean()), float(per_joint_error.max())


def _rebuild_fullbody_animation_with_ik(
    target_anim,
    *,
    rigid_offsets: np.ndarray | None = None,
    rigid_parents: np.ndarray | None = None,
    preserved_position_indices: np.ndarray | list[int] | None = None,
    preserved_rotation_indices: np.ndarray | list[int] | None = None,
    iterations: int = _FULLBODY_IK_ITERATIONS,
) -> tuple[object, float, float]:
    """Force a full-body IK rebuild against the current world-space motion.

    The restore path intentionally discards all non-root local translations and
    reconstructs the entire pose as a rigid skeleton animation on the requested
    skeleton definition. This lets restore target the raw export skeleton
    directly instead of rigidizing the intermediate processed skeleton, while
    optionally preserving trusted local pose channels on selected joints.
    """
    fullbody_rigidize_mask = np.ones(target_anim.shape[:2], dtype=bool)
    return _rebuild_animation_with_ik(
        target_anim,
        rigid_offsets=rigid_offsets,
        rigid_parents=rigid_parents,
        preserved_position_indices=preserved_position_indices,
        preserved_rotation_indices=preserved_rotation_indices,
        rigidize_joint_mask=fullbody_rigidize_mask,
        iterations=iterations,
    )


def _invert_preprocess_transform(
    processed_anim,
    *,
    scale_factor: float | None,
    root_translation_xz: np.ndarray | None,
    orientation_quat: np.ndarray | None,
):
    from motion_lib.Animation import Animation
    from motion_lib.Quaternions import Quaternions

    positions = processed_anim.positions.copy().astype(np.float64, copy=False)
    offsets = processed_anim.offsets.copy().astype(np.float64, copy=False)
    rotations = processed_anim.rotations.copy()

    if scale_factor is not None:
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

    if orientation_quat is not None:
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
) -> str:
    """Restore a preprocessed NPY motion file to a skinned GLB.

    Args:
        npy_path:            Path to the preprocessed .npy motion file.
        output_glb:          Path for the output .glb file.
        tpose_mesh:          Path to the T-pose FBX (provides skin + armature).
        cond_npy:            Path to cond.npy; defaults to the dataset default.
        object_type:         Character type key (e.g. "Horse").  Auto-detected
                             from the NPY filename if None.
        fps:                 Animation frame rate.  Defaults to 30 if not
                             specified.
        root_translation_xz: Optional explicit XZ translation to add back after
                     inverse scale and before inverse orientation. When
                     omitted, restore keeps the clip in centred
                     preprocessed space.
        fullbody_ik:          If True, perform a full-body IK reconstruction
                             on the raw export skeleton after recovering the
                             animation.  Default is False (skip IK, use
                             recovered pose directly).

    Returns:
        The absolute path of the written GLB file.
    """
    from Anytop.utils.exporter import AnimationExporter, animation_to_exporter_inputs
    from Anytop.utils.roundtrip_common import _build_skeleton
    from data_loaders.truebones.truebones_utils.motion_process import (
        _find_translation_root,
        recover_processed_animation_from_feature_animation,
    )

    output_glb = os.path.abspath(output_glb)
    _validate_ik_mode_selection(fullbody_ik)

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

    # ── Resolve T-pose mesh ───────────────────────────────────────────────────
    cond_entry = cond.get(object_type)
    if tpose_mesh is None:
        if cond_entry is not None and isinstance(cond_entry.get('orientation_reference_fbx_path'), str):
            tpose_mesh = cond_entry['orientation_reference_fbx_path']
            print(f"Resolved T-pose mesh from cond.npy: {tpose_mesh}")
        else:
            raise ValueError(
                f"No --tpose-mesh provided and cond.npy entry for '{object_type}' "
                f"does not contain 'orientation_reference_fbx_path'."
            )

    # ── Load NPY features ─────────────────────────────────────────────────────
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
    rotation_channel_mask = np.asarray(restore_ctx["rotation_channel_mask"], dtype=bool)
    original_joint_count = int(restore_ctx["original_joint_count"])
    translation_root_index = None

    # ── Resolve FPS ─────────────────────────────────────────────────────
    if fps is None:
        fps = 30.0

    helper_joint_count = len(feature_joint_names) - len(export_joint_names)
    if helper_joint_count > 0:
        print(
            f"Skeleton: {len(feature_joint_names)} feature joints "
            f"({helper_joint_count} helper joints), export root='{export_joint_names[0]}'"
        )
    else:
        print(f"Skeleton: {len(feature_joint_names)} joints, root='{feature_joint_names[0]}'")

    F, J, C = features.shape
    if J != len(feature_joint_names):
        raise ValueError(
            f"NPY has J={J} joints but cond.npy has {len(feature_joint_names)} joints for '{object_type}'."
        )
    if C != 13:
        raise ValueError(f"Expected 13 channels per joint, got {C}.")

    print(f"NPY: {F} frames, {J} joints, {C} channels")

    if restore_ctx["scale_factor"] is not None:
        print(f"T-pose preprocessing scale_factor: {restore_ctx['scale_factor']:.6f}")
    if root_translation_xz is None:
        print("Root translation XZ: keeping centred preprocessed placement")
    else:
        coerced_root_translation_xz = _coerce_root_translation_xz(root_translation_xz)
        print(
            "Root translation XZ override: "
            f"[{coerced_root_translation_xz[0]:.6f}, {coerced_root_translation_xz[2]:.6f}]"
        )
        root_translation_xz = coerced_root_translation_xz

    _warn_on_missing_mesh_joints(
        export_joint_names, tpose_mesh, mesh_bone_names=restore_ctx.get("mesh_bone_names")
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
    translation_root_index = _find_translation_root(recovered_feature_anim)

    print("Recovering processed animation channels for export...")
    export_anim = recover_processed_animation_from_feature_animation(
        recovered_feature_anim,
        tpose_rest_rotations,
    )

    export_anim = _invert_preprocess_transform(
        export_anim,
        scale_factor=restore_ctx.get("scale_factor"),
        root_translation_xz=root_translation_xz,
        orientation_quat=restore_ctx.get("orientation_quat"),
    )
    export_anim = _strip_appended_helper_joints(
        export_anim,
        original_joint_count=original_joint_count,
    )

    if fullbody_ik:
        print("Force full-body IK reconstruction on export skeleton...")
        export_anim, ik_mean_error, ik_max_error = _rebuild_fullbody_animation_with_ik(
            export_anim,
            rigid_offsets=export_offsets,
            rigid_parents=export_parents,
            preserved_position_indices=[translation_root_index],
            preserved_rotation_indices=[translation_root_index],
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

    if rotation_channel_mask is not None:
        frozen_joint_indices = np.flatnonzero(~rotation_channel_mask).tolist()
        if frozen_joint_indices:
            preview = ", ".join(export_joint_names[index] for index in frozen_joint_indices[:10])
            suffix = "..." if len(frozen_joint_indices) > 10 else ""
            print(
                f"\x1b[33m[WARN] Production features do not encode local rotations for {len(frozen_joint_indices)} "
                f"leaf joints; exporting them at rest: {preview}{suffix}\x1b[0m"
            )
            export_anim = _clamp_unobservable_joint_positions_to_rest(
                export_anim,
                rest_offsets=export_offsets,
                rotation_channel_mask=rotation_channel_mask,
            )
            print(
                f"Clamped local translations to rest offsets for the same {len(frozen_joint_indices)} "
                "unobservable leaf joints before export."
            )

    # ── Build skeleton for exporter ─────────────────────────────────────────
    skeleton = _build_skeleton(
        export_joint_names,
        export_offsets,
        export_parents,
        export_rest_rotations,
    )

    joint_rotations, root_translation, root_rotation, bone_translations = (
        animation_to_exporter_inputs(export_anim, skeleton)
    )

    os.makedirs(os.path.dirname(output_glb) or ".", exist_ok=True)

    # ── Export skinned GLB + BVH ────────────────────────────────────────────
    exporter = AnimationExporter(skeleton, fps=fps)
    print(f"Exporting skinned GLB → {output_glb}")
    exporter.export_glb(
        joint_rotations,
        root_translation,
        root_rotation,
        output_glb,
        mesh_path=tpose_mesh,
        bone_translations=bone_translations,
        rotation_channel_mask=rotation_channel_mask,
    )

    return os.path.abspath(output_glb)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Restore a preprocessed Anytop NPY motion to a skinned GLB "
            "using a T-pose FBX as the rig/skin source."
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
            "Path to the T-pose FBX/GLB/GLTF that provides skin weights + armature. "
            "If not specified, the path is read from cond.npy "
            "(orientation_reference_fbx_path)."
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
    ik_mode_group = parser.add_mutually_exclusive_group()
    ik_mode_group.add_argument(
        "--fullbody-ik",
        action="store_true",
        default=False,
        help=(
            "Perform full-body IK reconstruction on the raw export skeleton "
            "after recovering the animation.  Disabled by default."
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
    )

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
