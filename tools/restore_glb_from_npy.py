"""
restore_glb_from_npy.py

Restore a preprocessed Anytop NPY motion file back to a skinned GLB,
using a T-pose FBX/GLB/GLTF as the mesh/rig source.

Anytop's preprocessing applies three transforms to every motion:
  1. Bone names canonicalized  (Bip01_L_Foot → LeftFoot, etc.)
  2. Skeleton scaled           (mean bone length → HML_AVG_BONELEN)
  3. Root rotated to face +Z   (orientation_quat applied)

This script reverses all three and exports a skinned GLB whose armature
matches the original T-pose FBX bone names, scale, and facing direction.

Pipeline:
  NPY features (F, J, 13)
    → recover_from_features (HML space, +Z facing)
    → unscale positions ÷ scale_factor
    → reverse orientation (conjugate of orientation_quat)
    → AnimationExporter + T-pose FBX → skinned GLB

A BVH file is also auto-exported alongside the GLB (same directory, same stem,
.bvh extension) using the same AnimationExporter.

Note: locomotion XZ is stripped during preprocessing and cannot be
recovered from the NPY alone.  The output GLB plays the motion in-place.

Usage:
    # Using FBX T-pose
    python tools/restore_glb_from_npy.py \\
        --npy   "D:/AI/.../Horse___RunToStop_29.npy" \\
        --tpose_fbx "D:/AI/.../HorseALL-TPOSE.fbx"  \\
        --output_glb "outputs/Horse___RunToStop_29.glb"

    # Using GLB T-pose
    python tools/restore_glb_from_npy.py \\
        --npy   "D:/AI/.../Horse___RunToStop_29.npy" \\
        --tpose_fbx "D:/AI/.../HorseALL-RunToStop.glb"  \\
        --output_glb "outputs/Horse___RunToStop_29.glb"

    # Override auto-detected values
    python tools/restore_glb_from_npy.py \\
        --npy   my_motion.npy \\
        --tpose_fbx my_tpose.fbx \\
        --output_glb my_motion.glb \\
        --object_type Horse \\
        --fps 30 \\
        --cond_npy /custom/path/cond.npy
"""

import argparse
import importlib.machinery
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

_rotconv_path = os.path.join(ANYTOP_DIR, "utils", "rotation_conversions.py")
if os.path.isfile(_rotconv_path) and "utils.rotation_conversions" not in sys.modules:
    _loader = importlib.machinery.SourceFileLoader("utils.rotation_conversions", _rotconv_path)
    _spec = importlib.util.spec_from_loader("utils.rotation_conversions", _loader, origin=_rotconv_path)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["utils.rotation_conversions"] = _mod
    _spec.loader.exec_module(_mod)

_npy_rt_path = os.path.join(ANYTOP_DIR, "utils", "npy_roundtrip_utils.py")
if os.path.isfile(_npy_rt_path) and "utils.npy_roundtrip_utils" not in sys.modules:
    _loader = importlib.machinery.SourceFileLoader("utils.npy_roundtrip_utils", _npy_rt_path)
    _spec = importlib.util.spec_from_loader("utils.npy_roundtrip_utils", _loader, origin=_npy_rt_path)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["utils.npy_roundtrip_utils"] = _mod
    _spec.loader.exec_module(_mod)

from utils.npy_roundtrip_utils import coerce_feature_payload, recover_from_features
from motion_lib.Quaternions import Quaternions

# ── Default cond.npy path ─────────────────────────────────────────────────────

_DEFAULT_COND_NPY = os.path.realpath(
    os.path.join(ANYTOP_DIR, "dataset", "truebones", "zoo", "truebones_processed", "cond.npy")
)

# ── HML_AVG_BONELEN (lazy-loaded) ─────────────────────────────────────────────

_HML_AVG_BONELEN = None


def _get_hml_avg_bonelen() -> float:
    global _HML_AVG_BONELEN
    if _HML_AVG_BONELEN is None:
        from data_loaders.truebones.truebones_utils.param_utils import HML_AVG_BONELEN as _v
        _HML_AVG_BONELEN = float(_v)
    return _HML_AVG_BONELEN


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


def _compute_scale_factor(
    joints_names: list[str],
    fbx_bone_names: list[str],
    fbx_offsets: np.ndarray,
) -> float | None:
    """Compute HML_AVG_BONELEN / mean(FBX_bone_lengths) for named non-root bones.

    Matches cond bones to FBX bones by name.  Excludes the root (index 0)
    because the root bone typically has a zero offset.
    """
    fbx_off_map = {n: o for n, o in zip(fbx_bone_names, fbx_offsets)}
    matched_offsets = [
        fbx_off_map[n]
        for n in joints_names[1:]   # skip root
        if n in fbx_off_map
    ]
    if not matched_offsets:
        return None
    lengths = np.linalg.norm(np.array(matched_offsets, dtype=np.float64), axis=-1)
    non_zero = lengths[lengths > 1e-8]
    if non_zero.size == 0:
        return None
    mean_len = float(np.mean(non_zero))
    return _get_hml_avg_bonelen() / mean_len


# ── Main restore function ─────────────────────────────────────────────────────

def restore_glb(
    npy_path: str,
    tpose_fbx: str,
    output_glb: str,
    cond_npy: str | None = None,
    object_type: str | None = None,
    fps: float = 30.0,
    restore_orientation: bool = True,
) -> str:
    """Restore a preprocessed NPY motion file to a skinned GLB.

    Args:
        npy_path:            Path to the preprocessed .npy motion file.
        tpose_fbx:           Path to the T-pose FBX (provides skin + armature).
        output_glb:          Path for the output .glb file.
        cond_npy:            Path to cond.npy; defaults to the dataset default.
        object_type:         Character type key (e.g. "Horse").  Auto-detected
                             from the NPY filename if None.
        fps:                 Animation frame rate for the output GLB.
        restore_orientation: When True (default), reverse the +Z face-direction
                             transform applied during preprocessing.

    Returns:
        The absolute path of the written GLB file.
        A BVH file is also auto-exported alongside the GLB at the same path
        (same stem, .bvh extension).
    """
    import torch
    from Anytop.utils.exporter import AnimationExporter

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

    obj_cond = cond[object_type]
    joints_names: list[str] = list(obj_cond["joints_names"])
    parents = np.array(obj_cond["parents"], dtype=np.int32)
    offsets_hml = np.array(obj_cond["offsets"], dtype=np.float64)  # HML scale

    print(f"Skeleton: {len(joints_names)} joints, root='{joints_names[0]}'")

    # ── Load NPY features ─────────────────────────────────────────────────────
    raw = np.load(npy_path, allow_pickle=True)
    if raw.dtype == object:
        raw = raw.item()
    features, _ = coerce_feature_payload(raw)

    F, J, C = features.shape
    if J != len(joints_names):
        raise ValueError(
            f"NPY has J={J} joints but cond.npy has {len(joints_names)} joints for '{object_type}'."
        )
    if C != 13:
        raise ValueError(f"Expected 13 channels per joint, got {C}.")

    print(f"NPY: {F} frames, {J} joints, {C} channels")

    # ── Load T-pose mesh (FBX or GLB) for bone-name verification and scale computation ──
    print("Loading T-pose mesh for scale computation...")
    from Anytop.utils._roundtrip_common import _load_fbx_skeleton_metadata, _load_glb_skeleton_metadata, _build_skeleton

    tpose_lower = tpose_fbx.lower()
    if tpose_lower.endswith(".fbx"):
        fbx_bone_names, _fbx_parents, fbx_offsets, _fbx_rest_rots = _load_fbx_skeleton_metadata(
            tpose_fbx
        )
        print(f"FBX armature: {len(fbx_bone_names)} bones, root='{fbx_bone_names[0]}'")
    elif tpose_lower.endswith((".glb", ".gltf")):
        fbx_bone_names, _fbx_parents, fbx_offsets, _fbx_rest_rots = _load_glb_skeleton_metadata(
            tpose_fbx
        )
        print(f"GLB armature: {len(fbx_bone_names)} bones, root='{fbx_bone_names[0]}'")
    else:
        raise ValueError(
            f"Unsupported T-pose mesh format: {tpose_fbx} — expected .fbx, .glb, or .gltf"
        )

    # Verify all cond joint names exist in FBX armature
    fbx_name_set = set(fbx_bone_names)
    missing = [n for n in joints_names if n not in fbx_name_set]
    if missing:
        print(
            f"WARNING: {len(missing)} cond.npy joints not found in FBX armature:\n"
            f"  {missing[:10]}{'...' if len(missing) > 10 else ''}\n"
            f"These bones will be skipped during export (kept at FBX rest pose)."
        )
    else:
        print(f"All {J} cond joints found in FBX armature.")

    # ── Compute scale factor ──────────────────────────────────────────────────
    scale_factor = _compute_scale_factor(joints_names, fbx_bone_names, fbx_offsets)
    if scale_factor is None:
        print("WARNING: Could not compute scale_factor from FBX offsets. Using 1.0.")
        scale_factor = 1.0
    else:
        print(
            f"Scale factor: {scale_factor:.6f}  "
            f"(HML_AVG={_get_hml_avg_bonelen():.4f} / mean_fbx_bonelen)"
        )

    # ── Recover Animation (in HML feature space) ──────────────────────────────
    print("Recovering animation from features...")
    recovered_anim, has_animated_pos = recover_from_features(features, parents, offsets_hml)
    print(f"Recovered: {recovered_anim.shape[0]} frames")
    if has_animated_pos:
        print("Note: non-root bone translations detected (unusual for BVH-sourced clips).")

    # ── Unscale positions to original mesh scale ───────────────────────────────
    # Rotations are scale-invariant; only translations need unscaling.
    recovered_anim.positions = recovered_anim.positions / scale_factor
    recovered_anim.offsets = offsets_hml / scale_factor

    # ── Reverse orientation (preprocessing rotated root to face +Z) ────────────
    if restore_orientation:
        ori_quat = obj_cond.get("orientation_quat")
        if ori_quat is not None:
            ori_q = Quaternions(np.array(ori_quat, dtype=np.float64))
            # Preprocessing: new_rots[:,0] = ori_q * old_rots[:,0]
            # Reverse:       old_rots[:,0] = conjugate(ori_q) * new_rots[:,0]
            conj = -ori_q
            recovered_anim.rotations[:, 0] = conj * recovered_anim.rotations[:, 0]
            recovered_anim.positions[:, 0] = conj * recovered_anim.positions[:, 0]
            print("Orientation restored.")
        else:
            print("WARNING: orientation_quat not found in cond.npy — orientation not restored.")

    # ── Build skeleton (identity rest, HML offsets scaled to FBX size) ─────
    scaled_offsets = offsets_hml / scale_factor
    skeleton = _build_skeleton(
        joints_names,
        scaled_offsets,
        parents,
        rest_rotations=None,
    )

    # ── Common animation tensors ───────────────────────────────────────────
    joint_rotations = torch.from_numpy(recovered_anim.rotations.qs.astype(np.float32))
    root_translation = torch.from_numpy(recovered_anim.positions[:, 0, :].astype(np.float32))
    # Exporter semantics are format-independent: root_rotation is only an extra
    # world-space wrapper transform applied before the hierarchy. The recovered
    # root joint animation already lives in joint_rotations[:, 0], so use an
    # identity wrapper for both GLB and BVH.
    root_rotation = torch.zeros((F, 4), dtype=torch.float32)
    root_rotation[:, 0] = 1.0

    # Exporter bone_translations use Blender-style pose-bone.location channels
    # for non-root joints. The root entry is ignored because root_translation
    # carries the world-space root motion.
    if has_animated_pos:
        pose_translations = recovered_anim.positions.astype(np.float32).copy()
        pose_translations[:, 0, :] = 0.0
        pose_translations[:, 1:, :] -= recovered_anim.offsets[None, 1:, :].astype(np.float32)
        bone_translations = torch.from_numpy(pose_translations)
    else:
        bone_translations = None

    os.makedirs(os.path.dirname(os.path.abspath(output_glb)) or ".", exist_ok=True)

    # ── Export skinned GLB via mesh_path (T-pose FBX) ──────────────────────
    # The exporter handles retargeting: world-space alignment from HML skeleton
    # to FBX armature, followed by FBX-local conversion.
    exporter = AnimationExporter(skeleton, fps=fps)
    print(f"Exporting skinned GLB → {output_glb}")
    exporter.export(
        joint_rotations,
        root_translation,
        root_rotation,
        output_glb,
        mesh_path=tpose_fbx,
        bone_translations=bone_translations,
    )

    # ── Auto-export BVH alongside GLB ──────────────────────────────────────
    output_bvh = os.path.splitext(output_glb)[0] + ".bvh"
    exporter.export(
        joint_rotations,
        root_translation,
        root_rotation,
        output_bvh,
        bone_translations=bone_translations,
    )

    return os.path.abspath(output_glb)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Restore a preprocessed Anytop NPY motion to a skinned GLB "
            "using a T-pose FBX/GLB/GLTF as the rig/skin source."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--npy", required=True,
        help="Path to the preprocessed .npy motion file.",
    )
    parser.add_argument(
        "--tpose_fbx", required=True,
        help="Path to the T-pose mesh (.fbx, .glb, or .gltf) that provides skin weights + armature.",
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
        default=30.0,
        help="Animation frame rate for the output GLB (default: 30).",
    )
    parser.add_argument(
        "--no_orientation_restore",
        action="store_true",
        default=False,
        help=(
            "Skip orientation reversal.  The output GLB will face +Z "
            "(canonical direction) rather than the original FBX facing."
        ),
    )
    parser.add_argument(
        "--skip_compare",
        action="store_true",
        default=False,
        help=(
            "Skip automatic BVH-vs-GLB comparison after export via compare_motions.py."
        ),
    )

    args = parser.parse_args()

    if not os.path.isfile(args.npy):
        parser.error(f"NPY file not found: {args.npy}")
    if not os.path.isfile(args.tpose_fbx):
        parser.error(f"T-pose mesh not found: {args.tpose_fbx}")

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

    output_bvh = os.path.splitext(args.output_glb)[0] + ".bvh"

    print(f"NPY           : {args.npy}")
    print(f"T-pose mesh   : {args.tpose_fbx}")
    print(f"Output GLB    : {args.output_glb}")
    print(f"Output BVH    : {output_bvh}")
    print(f"cond.npy      : {cond_npy_path}")
    print(f"FPS           : {args.fps}")
    print(f"Restore ori   : {not args.no_orientation_restore}")
    print(f"Skip compare  : {args.skip_compare}")
    print()

    out = restore_glb(
        npy_path=args.npy,
        tpose_fbx=args.tpose_fbx,
        output_glb=args.output_glb,
        cond_npy=cond_npy_path,
        object_type=args.object_type,
        fps=args.fps,
        restore_orientation=not args.no_orientation_restore,
    )

    # ── Auto-compare: BVH vs GLB via compare_motions.py ────────────────────
    compare_script = os.path.join(SCRIPT_DIR, "compare_motions.py")
    if not args.skip_compare and os.path.isfile(compare_script):
        print()
        print("─" * 60)
        print("  Running compare_motions.py (BVH vs GLB) ...")
        print("─" * 60)
        print()
        try:
            subprocess.run(
                [
                    sys.executable, compare_script,
                    "--motion_a", output_bvh,
                    "--motion_b", args.output_glb,
                ],
                cwd=ANYTOP_DIR,
                check=True,
            )
        except FileNotFoundError:
            print("  [WARN] bpy not available in this environment — comparison skipped.")
        except subprocess.CalledProcessError as e:
            print(f"  [WARN] compare_motions.py exited with code {e.returncode}.")
        except Exception as e:
            print(f"  [WARN] compare_motions.py failed: {e}")


if __name__ == "__main__":
    main()
