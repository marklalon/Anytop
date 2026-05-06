"""
restore_glb_from_npy.py

Restore a preprocessed Anytop NPY motion file back to a skinned GLB,
using a T-pose FBX as the mesh/rig source.

Pipeline:
    NPY payload/features
        → recover_from_features(...)
        → recover_processed_animation_from_feature_animation(...)
        → animation_to_exporter_inputs(...)
        → AnimationExporter + T-pose FBX → skinned GLB

If the NPY file contains a self-contained metadata payload with embedded
skeleton metadata (joint_names, parents, offsets, etc.), that metadata is
used directly. For bare (F, J, 13) tensors, the script derives the same
T-pose preprocessing metadata from the supplied T-pose FBX.

Metadata is resolved from three sources in descending priority:

    1. NPY embedded payload  — self-contained dict with joint_names, parents,
                               offsets, tpose_rest_rotations, fps, etc.
    2. cond.npy entry        — dataset-wide metadata indexed by object_type
                               (e.g. "Horse"), stores joints_names, parents,
                               offsets, rest_rotations, orientation_quat, etc.
    3. T-pose FBX fallback   — computed on demand via
                               get_common_features_from_T_pose(); most
                               expensive, only loaded when tiers 1–2 lack
                               the needed fields.

Within each tier, specific fields may come from different sources. For
example, orientation_quat is available from cond.npy or T-pose FBX but
is never embedded in the NPY payload.

Note: locomotion XZ stripped during preprocessing cannot be recovered from a
plain feature tensor alone. Metadata payloads preserve the initial translation
root offset and therefore round-trip more faithfully.

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

from utils.npy_roundtrip_utils import coerce_feature_payload, recover_from_features
from Anytop.utils.roundtrip_common import identity_rest_rotations

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
        orientation_quat,
        _forward_joint_index,
        _forward_base_joint_index,
        _contact_joint_source,
    ) = get_common_features_from_T_pose(tpose_mesh, object_type)
    return {
        "joint_names": list(tpose_bone_names),
        "parents": np.asarray(tpose_anim.parents, dtype=np.int32),
        "offsets": np.asarray(offsets_hml, dtype=np.float32),
        "tpose_rest_rotations": np.asarray(tpose_rotations[0], dtype=np.float32),
        "orientation_quat": np.asarray(orientation_quat, dtype=np.float64),
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
    from Anytop.utils.roundtrip_common import _load_fbx_skeleton_metadata

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
    *,
    need_orientation_quat: bool,
) -> dict[str, object]:
    features, payload = coerce_feature_payload(raw_npy)

    # ── Check availability across all three tiers ───────────────────────
    #   Tier 1: npy payload (embedded metadata)
    #   Tier 2: cond.npy entry (dataset-wide metadata)
    #   Tier 3: T-pose FBX (fallback, most expensive)
    has_payload_skeleton = payload is not None and all(
        key in payload for key in ("joint_names", "parents", "offsets")
    )
    has_payload_tpose = payload is not None and "tpose_rest_rotations" in payload

    cond_has_skeleton = cond_entry is not None and all(
        key in cond_entry for key in ("joints_names", "parents", "offsets")
    )
    cond_has_tpose = cond_entry is not None and "tpose_rest_rotations" in cond_entry
    cond_has_orientation = cond_entry is not None and "orientation_quat" in cond_entry
    cond_has_scale = cond_entry is not None and "scale_factor" in cond_entry

    # Only load the T-pose FBX when embedded metadata is missing the pieces we
    # need. cond.npy stores original bind-frame rest rotations under
    # `rest_rotations`; those are not the same as the feature-space
    # `tpose_rest_rotations` used to invert T-pose reparameterization.
    need_tpose_skeleton = not has_payload_skeleton and not cond_has_skeleton
    need_tpose_tpose = not has_payload_tpose and not cond_has_tpose
    need_tpose_orientation = need_orientation_quat and not cond_has_orientation
    need_tpose_scale = not cond_has_scale

    tpose_meta = None
    if need_tpose_skeleton or need_tpose_tpose or need_tpose_orientation or need_tpose_scale:
        tpose_meta = _load_tpose_restore_metadata(tpose_mesh, object_type)

    # ── Skeleton info (joint_names / parents / offsets) ─────────────────
    if has_payload_skeleton:
        joint_names = list(payload["joint_names"])
        parents = np.asarray(payload["parents"], dtype=np.int32)
        offsets = np.asarray(payload["offsets"], dtype=np.float32)
        skeleton_rest_rotations = np.asarray(
            payload.get("rest_rotations", identity_rest_rotations(len(joint_names))),
            dtype=np.float32,
        )
    elif cond_has_skeleton:
        joint_names = list(cond_entry["joints_names"])
        parents = np.asarray(cond_entry["parents"], dtype=np.int32)
        offsets = np.asarray(cond_entry["offsets"], dtype=np.float32)
        skeleton_rest_rotations = np.asarray(
            cond_entry.get("rest_rotations", identity_rest_rotations(len(joint_names))),
            dtype=np.float32,
        )
    else:
        joint_names = list(tpose_meta["joint_names"])
        parents = np.asarray(tpose_meta["parents"], dtype=np.int32)
        offsets = np.asarray(tpose_meta["offsets"], dtype=np.float32)
        skeleton_rest_rotations = identity_rest_rotations(len(joint_names))

    # ── T-pose rest rotations ───────────────────────────────────────────
    if has_payload_tpose:
        tpose_rest_rotations = np.asarray(payload["tpose_rest_rotations"], dtype=np.float32)
    elif cond_has_tpose:
        tpose_rest_rotations = _remap_joint_array(
            list(cond_entry["joints_names"]),
            joint_names,
            np.asarray(cond_entry["tpose_rest_rotations"], dtype=np.float32),
            "feature-space",
        )
    else:
        tpose_rest_rotations = _remap_joint_array(
            tpose_meta["joint_names"],
            joint_names,
            np.asarray(tpose_meta["tpose_rest_rotations"], dtype=np.float32),
            "feature-space",
        )

    # ── Orientation quat ────────────────────────────────────────────────
    orientation_quat = None
    if need_orientation_quat:
        if cond_has_orientation:
            orientation_quat = np.asarray(cond_entry["orientation_quat"], dtype=np.float64)
        elif tpose_meta is not None and tpose_meta.get("orientation_quat") is not None:
            orientation_quat = np.asarray(tpose_meta["orientation_quat"], dtype=np.float64)

    # ── Scale factor ────────────────────────────────────────────────────
    scale_factor = None
    if cond_has_scale:
        scale_factor = float(cond_entry["scale_factor"])
    elif tpose_meta is not None and tpose_meta.get("scale_factor") is not None:
        scale_factor = float(tpose_meta["scale_factor"])

    return {
        "features": features,
        "payload": payload,
        "joint_names": joint_names,
        "parents": parents,
        "offsets": offsets,
        "skeleton_rest_rotations": skeleton_rest_rotations,
        "tpose_rest_rotations": tpose_rest_rotations,
        "orientation_quat": orientation_quat,
        "scale_factor": scale_factor,
    }


def _restore_root_from_canonical_facing(animation, orientation_quat):
    from motion_lib.Animation import Animation
    from motion_lib.Quaternions import Quaternions

    orientation = Quaternions(np.asarray(orientation_quat, dtype=np.float64))
    inverse_orientation = -orientation
    rotations = animation.rotations.copy()
    positions = animation.positions.copy()
    rotations[:, 0] = inverse_orientation * rotations[:, 0]
    positions[:, 0] = inverse_orientation * positions[:, 0]
    return Animation(
        rotations,
        positions,
        animation.orients.copy(),
        animation.offsets.copy(),
        animation.parents.copy(),
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
    restore_orientation: bool = True,
) -> str:
    """Restore a preprocessed NPY motion file to a skinned GLB.

    Args:
        npy_path:            Path to the preprocessed .npy motion file.
        tpose_mesh:          Path to the T-pose FBX (provides skin + armature).
        output_glb:          Path for the output .glb file.
        cond_npy:            Path to cond.npy; defaults to the dataset default.
        object_type:         Character type key (e.g. "Horse").  Auto-detected
                             from the NPY filename if None.
        fps:                 Animation frame rate.  Auto-detected from the
                             NPY metadata payload if None; falls back to 30.
        restore_orientation: When True (default), reverse the +Z face-direction
                             transform applied during preprocessing.

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
    raw = np.load(npy_path, allow_pickle=True)
    if getattr(raw, "dtype", None) == object:
        raw = raw.item()
    cond_entry = cond.get(object_type)
    restore_ctx = _build_restore_context(
        raw,
        object_type,
        tpose_mesh,
        cond_entry=cond_entry,
        need_orientation_quat=restore_orientation,
    )

    features = restore_ctx["features"]
    payload = restore_ctx["payload"]
    joints_names: list[str] = restore_ctx["joint_names"]
    parents = restore_ctx["parents"]
    offsets_hml = restore_ctx["offsets"]
    skeleton_rest_rotations = restore_ctx["skeleton_rest_rotations"]
    tpose_rest_rotations = restore_ctx["tpose_rest_rotations"]
    is_bare_feature_tensor = payload is None
    rotation_channel_mask = None

    if is_bare_feature_tensor:
        # Production bare tensors are exported against the feature skeleton with
        # identity rest rotations. Using cond/T-pose rest rotations here changes
        # the armature basis and introduces a spurious global wrapper rotation.
        skeleton_rest_rotations = identity_rest_rotations(len(joints_names))
        rotation_channel_mask = _bare_feature_rotation_channel_mask(parents)

    # ── Resolve FPS ─────────────────────────────────────────────────────
    if fps is None:
        if payload is not None and "fps" in payload:
            fps = float(payload["fps"])
        else:
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

    if restore_ctx["payload"] is None:
        print("Input NPY has no metadata payload; deriving restore metadata from the T-pose mesh.")
    else:
        print("Using embedded NPY metadata for recovery.")

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

    if is_bare_feature_tensor and rotation_channel_mask is not None:
        export_anim, frozen_joint_indices = _freeze_unencoded_bare_joint_rotations(
            export_anim,
            rotation_channel_mask,
        )
        if frozen_joint_indices:
            preview = ", ".join(joints_names[index] for index in frozen_joint_indices[:10])
            suffix = "..." if len(frozen_joint_indices) > 10 else ""
            print(
                f"Bare production features do not encode local rotations for {len(frozen_joint_indices)} "
                f"leaf/helper joints; keeping rest rotation on export: {preview}{suffix}"
            )

    if restore_orientation and not is_bare_feature_tensor:
        orientation_quat = restore_ctx["orientation_quat"]
        if orientation_quat is None:
            print("WARNING: orientation metadata unavailable - output keeps canonical +Z facing.")
        else:
            export_anim = _restore_root_from_canonical_facing(export_anim, orientation_quat)
            print("Restored original facing from preprocessing orientation metadata.")
    elif restore_orientation and is_bare_feature_tensor:
        print(
            "Bare production features already decode in source-rig facing; "
            "skipping explicit orientation restore."
        )
    else:
        print("Orientation restore skipped; keeping canonical +Z facing.")

    # ── Build skeleton for exporter ─────────────────────────────────────────
    skeleton = _build_skeleton(
        joints_names,
        offsets_hml,
        parents,
        rest_rotations=skeleton_rest_rotations,
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
        help=(
            "Animation frame rate.  Auto-detected from the NPY metadata payload "
            "if not specified; falls back to 30."
        ),
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
