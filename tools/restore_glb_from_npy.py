"""
restore_glb_from_npy.py

Restore a preprocessed Anytop NPY motion file back to a skinned GLB,
using a T-pose FBX as the mesh/rig source.

Pipeline:
    NPY features
        → recover_from_features(...)                    — feature-space Animation
        → recover_processed_animation_from_feature_animation(...)  — undo T-pose reparameterization
        → _freeze_unencoded_bare_joint_rotations(...)   — freeze leaf/helper joint rotations
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
plain feature tensor alone.

Usage:
        # Using FBX T-pose
        python tools/restore_glb_from_npy.py \
                --npy "D:/AI/.../Horse___RunToStop_29.npy" \
                --tpose_mesh "D:/AI/.../HorseALL-TPOSE.fbx" \
                --output_glb "outputs/Horse___RunToStop_29.glb"

"""

import argparse
import importlib.util
import os
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

from utils.npy_roundtrip_utils import recover_from_features
from Anytop.utils.roundtrip_common import identity_rest_rotations, _load_fbx_skeleton_metadata

# ── Default cond.npy path ─────────────────────────────────────────────────────

_DEFAULT_COND_NPY = os.path.realpath(
    os.path.join(ANYTOP_DIR, "dataset", "truebones", "zoo", "truebones_processed", "cond.npy")
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _auto_detect_object_type_from_filename(npy_path: str, cond: dict) -> str | None:
    """Auto-detect object_type from NPY filename.

    Filenames follow the pattern: {ObjectType}___{Action}_{ClipID}.npy
    e.g. Horse___RunToStop_29.npy, Sea_Lion___Swim_42.npy
    """
    basename = os.path.splitext(os.path.basename(npy_path))[0]
    sep = "___"
    if sep in basename:
        candidate = basename.split(sep)[0]
        if candidate in cond:
            return candidate
    # Fallback: progressively longer prefixes (handles "Sea_Lion" etc.)
    if "_" in basename:
        parts = basename.split("_")
        for i in range(1, len(parts)):
            candidate = "_".join(parts[:i])
            if candidate in cond:
                return candidate
    return None


def _load_tpose_restore_metadata(tpose_mesh: str, object_type: str) -> dict[str, object]:
    from data_loaders.truebones.truebones_utils.motion_process import get_common_features_from_T_pose

    tpose_lower = tpose_mesh.lower()
    if not tpose_lower.endswith(".fbx"):
        raise ValueError(f"Unsupported T-pose mesh format: {tpose_mesh} - expected .fbx")

    (
        _root_pose_init_xz,
        scale_factor,
        offsets_hml,
        _foot_indices,
        tpose_rotations,
        tpose_bone_names,
        tpose_anim,
        _face_joints,
        _orientation_quat,
        _forward_joint_index,
        _forward_base_joint_index,
        _contact_joint_source,
    ) = get_common_features_from_T_pose(tpose_mesh, object_type)
    return {
        "joint_names": list(tpose_bone_names),
        "parents": np.asarray(tpose_anim.parents, dtype=np.int32),
        "offsets": np.asarray(offsets_hml, dtype=np.float32),
        "tpose_rest_rotations": np.asarray(tpose_rotations[0], dtype=np.float32),
        "scale_factor": float(scale_factor),
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


def _warn_on_missing_mesh_joints(joint_names: list[str], tpose_mesh: str) -> None:
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


def _build_restore_context(
    raw_npy,
    object_type: str,
    tpose_mesh: str,
    cond_entry: dict | None = None,
) -> dict[str, object]:
    features = np.asarray(raw_npy)

    # ── Check availability across two tiers ─────────────────────────────
    #   Tier 1: cond.npy entry (dataset-wide metadata)
    #   Tier 2: T-pose FBX (fallback, most expensive)
    cond_has_skeleton = cond_entry is not None and all(
        key in cond_entry for key in ("joints_names", "parents", "offsets")
    )
    cond_has_scale = cond_entry is not None and "scale_factor" in cond_entry

    # T-pose FBX is always loaded: it provides `tpose_rest_rotations` (which
    # cond.npy does not store), and may supply scale_factor when cond.npy lacks it.
    tpose_meta = _load_tpose_restore_metadata(tpose_mesh, object_type)

    # ── Skeleton info (joint_names / parents / offsets) ─────────────────
    if cond_has_skeleton:
        joint_names = list(cond_entry["joints_names"])
        parents = np.asarray(cond_entry["parents"], dtype=np.int32)
        offsets = np.asarray(cond_entry["offsets"], dtype=np.float32)
    else:
        joint_names = list(tpose_meta["joint_names"])
        parents = np.asarray(tpose_meta["parents"], dtype=np.int32)
        offsets = np.asarray(tpose_meta["offsets"], dtype=np.float32)

    # ── T-pose rest rotations (always from T-pose FBX) ─────────────────
    tpose_rest_rotations = _remap_joint_array(
        tpose_meta["joint_names"],
        joint_names,
        np.asarray(tpose_meta["tpose_rest_rotations"], dtype=np.float32),
        "feature-space",
    )

    # ── Scale factor ────────────────────────────────────────────────────
    scale_factor = None
    if cond_has_scale:
        scale_factor = float(cond_entry["scale_factor"])
    elif tpose_meta is not None and tpose_meta.get("scale_factor") is not None:
        scale_factor = float(tpose_meta["scale_factor"])

    return {
        "features": features,
        "joint_names": joint_names,
        "parents": parents,
        "offsets": offsets,
        "tpose_rest_rotations": tpose_rest_rotations,
        "scale_factor": scale_factor,
    }


def _bare_feature_rotation_channel_mask(parents: np.ndarray) -> np.ndarray:
    parents = np.asarray(parents, dtype=np.int32)
    joint_count = len(parents)
    if joint_count == 0:
        return np.zeros((0,), dtype=bool)

    child_counts = np.bincount(parents[parents >= 0], minlength=joint_count)
    rotation_channel_mask = child_counts > 0
    rotation_channel_mask[0] = True
    return rotation_channel_mask


def _freeze_unencoded_bare_joint_rotations(animation, rotation_channel_mask: np.ndarray):
    from motion_lib.Animation import Animation
    from motion_lib.Quaternions import Quaternions

    frozen_joint_indices = np.flatnonzero(~np.asarray(rotation_channel_mask, dtype=bool))
    if frozen_joint_indices.size == 0:
        return animation, []

    rotations = animation.rotations.copy()
    rotations[:, frozen_joint_indices.tolist()] = Quaternions.id(
        (animation.shape[0], int(frozen_joint_indices.size))
    )
    frozen_animation = Animation(
        rotations,
        animation.positions.copy(),
        animation.orients.copy(),
        animation.offsets.copy(),
        animation.parents.copy(),
    )
    return frozen_animation, frozen_joint_indices.tolist()


# ── Main restore function ─────────────────────────────────────────────────────

def restore_glb(
    npy_path: str,
    tpose_mesh: str,
    output_glb: str,
    cond_npy: str | None = None,
    object_type: str | None = None,
    fps: float | None = None,
) -> str:
    """Restore a preprocessed NPY motion file to a skinned GLB.

    Args:
        npy_path:            Path to the preprocessed .npy motion file.
        tpose_mesh:          Path to the T-pose FBX (provides skin + armature).
        output_glb:          Path for the output .glb file.
        cond_npy:            Path to cond.npy; defaults to the dataset default.
        object_type:         Character type key (e.g. "Horse").  Auto-detected
                             from the NPY filename if None.
        fps:                 Animation frame rate.  Defaults to 30 if not
                             specified.

    Returns:
        The absolute path of the written GLB file.
    """
    from Anytop.utils.exporter import AnimationExporter, animation_to_exporter_inputs
    from Anytop.utils.roundtrip_common import _build_skeleton
    from data_loaders.truebones.truebones_utils.motion_process import (
        recover_processed_animation_from_feature_animation,
    )

    output_glb = os.path.abspath(output_glb)

    # ── Load cond.npy ─────────────────────────────────────────────────────────
    cond_npy_path = cond_npy or _DEFAULT_COND_NPY
    if not os.path.isfile(cond_npy_path):
        raise FileNotFoundError(f"cond.npy not found: {cond_npy_path}")
    cond = np.load(cond_npy_path, allow_pickle=True).item()

    # ── Detect object_type ────────────────────────────────────────────────────
    if object_type is None:
        object_type = _auto_detect_object_type_from_filename(npy_path, cond)
        if object_type is None:
            raise ValueError(
                f"Cannot auto-detect object_type from '{os.path.basename(npy_path)}'.\n"
                f"  Available: {list(cond.keys())}\n"
                f"  Pass --object_type explicitly."
            )
        print(f"Auto-detected object_type: {object_type}")
    elif object_type not in cond:
        raise ValueError(
            f"object_type '{object_type}' not found in cond.npy.\n"
            f"  Available: {list(cond.keys())}"
        )

    # ── Load NPY features ─────────────────────────────────────────────────────
    raw = np.load(npy_path)
    cond_entry = cond.get(object_type)
    restore_ctx = _build_restore_context(
        raw,
        object_type,
        tpose_mesh,
        cond_entry=cond_entry,
    )

    features = restore_ctx["features"]
    joints_names: list[str] = restore_ctx["joint_names"]
    parents = restore_ctx["parents"]
    offsets_hml = restore_ctx["offsets"]
    tpose_rest_rotations = restore_ctx["tpose_rest_rotations"]
    rotation_channel_mask = _bare_feature_rotation_channel_mask(parents)

    # ── Resolve FPS ─────────────────────────────────────────────────────
    if fps is None:
        fps = 30.0

    print(f"Skeleton: {len(joints_names)} joints, root='{joints_names[0]}'")

    F, J, C = features.shape
    if J != len(joints_names):
        raise ValueError(
            f"NPY has J={J} joints but cond.npy has {len(joints_names)} joints for '{object_type}'."
        )
    if C != 13:
        raise ValueError(f"Expected 13 channels per joint, got {C}.")

    print(f"NPY: {F} frames, {J} joints, {C} channels")

    if restore_ctx["scale_factor"] is not None:
        print(f"T-pose preprocessing scale_factor: {restore_ctx['scale_factor']:.6f}")

    _warn_on_missing_mesh_joints(joints_names, tpose_mesh)

    # ── Recover Animation (in HML feature space) ──────────────────────────────
    print("Recovering feature-space animation from NPY...")
    recovered_feature_anim, has_animated_pos = recover_from_features(raw, parents, offsets_hml)
    print(f"Recovered: {recovered_feature_anim.shape[0]} frames")
    if has_animated_pos:
        print("Note: non-root bone translations detected (unusual for BVH-sourced clips).")

    print("Recovering processed animation channels for export...")
    export_anim = recover_processed_animation_from_feature_animation(
        recovered_feature_anim,
        tpose_rest_rotations,
    )

    if rotation_channel_mask is not None:
        export_anim, frozen_joint_indices = _freeze_unencoded_bare_joint_rotations(
            export_anim,
            rotation_channel_mask,
        )
        if frozen_joint_indices:
            preview = ", ".join(joints_names[index] for index in frozen_joint_indices[:10])
            suffix = "..." if len(frozen_joint_indices) > 10 else ""
            print(
                f"Production features do not encode local rotations for {len(frozen_joint_indices)} "
                f"leaf/helper joints; keeping rest rotation on export: {preview}{suffix}"
            )

    # ── Build skeleton for exporter ─────────────────────────────────────────
    skeleton = _build_skeleton(
        joints_names,
        offsets_hml,
        parents,
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
        "--tpose_mesh", required=True,
        help="Path to the T-pose FBX that provides skin weights + armature.",
    )
    parser.add_argument(
        "--output_glb",
        default=None,
        help=(
            "Output GLB path.  Defaults to outputs/restore_glb_from_npy/<stem>.glb "
            "relative to the Anytop directory."
        ),
    )
    parser.add_argument(
        "--cond_npy",
        default=None,
        help=f"Path to cond.npy.  Default: {_DEFAULT_COND_NPY}",
    )
    parser.add_argument(
        "--object_type",
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

    args = parser.parse_args()

    if not os.path.isfile(args.npy):
        parser.error(f"NPY file not found: {args.npy}")
    if not os.path.isfile(args.tpose_mesh):
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
            "Use --cond_npy to specify a custom path."
        )

    print(f"NPY           : {args.npy}")
    print(f"T-pose mesh   : {args.tpose_mesh}")
    print(f"Output GLB    : {args.output_glb}")
    print(f"cond.npy      : {cond_npy_path}")
    print(f"FPS           : {args.fps or '(auto)'}")
    print()

    restore_glb(
        npy_path=args.npy,
        tpose_mesh=args.tpose_mesh,
        output_glb=args.output_glb,
        cond_npy=cond_npy_path,
        object_type=args.object_type,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
