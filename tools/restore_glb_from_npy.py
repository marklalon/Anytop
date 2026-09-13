"""
restore_glb_from_npy.py

Restore a preprocessed Anytop NPY motion file back to a skinned GLB,
using a T-pose FBX as the mesh/rig source.

Pipeline (the decode lives in ``utils.npy_restore`` and is shared with the
BVH preview ``sample/generate.py`` writes next to every generated NPY):
    NPY features
        → restore_animation_from_features(...)  — recover, bake rest, invert
                                                  preprocess transform, optional
                                                  full-body IK, resample
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
from data_loaders.truebones.truebones_utils.cond_schema import load_cond
from data_loaders.truebones.truebones_utils.dataset_sources import (
    resolve_species_key,
    species_lookup_map,
)
from utils.roundtrip_common import (
    load_fbx_skeleton_metadata,
)

# ── Default cond.npy path ─────────────────────────────────────────────────────

_DEFAULT_COND_NPY = os.path.realpath(
    os.path.join(ANYTOP_DIR, "dataset", "truebones", "zoo", "truebones_processed", "cond.npy")
)

from utils.fullbody_ik import DEFAULT_IK_STRETCH_FACTOR
from utils.npy_restore import (
    build_mesh_restore_context,
    build_skeleton_only_context,
    coerce_root_translation_xz,
    restore_animation_from_features,
)


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
    cond = load_cond(cond_npy_path)

    # ── Detect object_type ────────────────────────────────────────────────────
    if object_type is None:
        object_type = infer_object_type_from_filename(npy_path, valid_types=species_lookup_map(cond))
        if object_type is None:
            raise ValueError(
                f"Cannot auto-detect object_type from '{os.path.basename(npy_path)}'.\n"
                f"  Available: {list(cond.keys())}\n"
                f"  Pass --object-type explicitly."
            )
        print(f"Auto-detected object_type: {object_type}")
    else:
        # Bare name, namespace suffix, canonical key, or filename token.
        resolved = resolve_species_key(cond, object_type)
        if resolved is None:
            raise ValueError(
                f"object_type '{object_type}' not found in cond.npy.\n"
                f"  Available: {list(cond.keys())}"
            )
        object_type = resolved

    # ── Build the skeleton context ────────────────────────────────────────────
    cond_entry = cond[object_type]
    features = np.load(npy_path)
    feature_joint_count = int(features.shape[1]) if features.ndim == 3 else None
    if skeleton_only and tpose_mesh is None:
        ctx = build_skeleton_only_context(
            cond_entry,
            object_type=object_type,
            feature_joint_count=feature_joint_count,
        )
        tpose_mesh_resolved = None
    else:
        ctx = build_mesh_restore_context(
            cond_entry,
            tpose_mesh,
            object_type,
            feature_joint_count=feature_joint_count,
        )
        # skeleton-only native with tpose_mesh: import the source armature and
        # drop only its meshes at export time. This preserves the same
        # armature-object scale / local-offset decomposition as a skinned GLB.
        # HML skeleton-only remains mesh-free/canonical, as before.
        tpose_mesh_resolved = (
            tpose_mesh
            if (not skeleton_only or restore_space == "native")
            else None
        )

    # ── Resolve FPS ─────────────────────────────────────────────────────
    if fps is None:
        fps = 30.0

    print(f"Skeleton: {ctx.joint_count} joints, root='{ctx.joint_names[0]}'")
    print(f"NPY: {features.shape[0]} frames, {features.shape[1]} joints, {features.shape[2]} channels")
    print(f"T-pose preprocessing scale_factor: {ctx.scale_factor:.6f}")
    if root_translation_xz is None:
        print("Root translation XZ: keeping centred preprocessed placement")
    else:
        root_translation_xz = coerce_root_translation_xz(root_translation_xz)
        print(
            "Root translation XZ override: "
            f"[{root_translation_xz[0]:.6f}, {root_translation_xz[2]:.6f}]"
        )

    if not skeleton_only:
        _warn_on_missing_mesh_joints(
            ctx.export_joint_names,
            tpose_mesh_resolved,
            mesh_bone_names=ctx.mesh_bone_names,
        )

    # ── Decode (shared with the generate-time BVH preview) ────────────────────
    restored = restore_animation_from_features(
        features,
        ctx,
        restore_space=restore_space,
        fullbody_ik=fullbody_ik,
        stretch_factor=stretch_factor,
        root_translation_xz=root_translation_xz,
        fps=fps,
        resample_fps=resample_fps,
        resample_min_length=resample_min_length,
        log=print,
    )
    if not fullbody_ik:
        print("Skipping IK (use --fullbody-ik to enable).")
    export_anim = restored.animation
    skeleton = restored.skeleton
    output_fps = restored.fps

    # ── Export skeleton units ───────────────────────────────────────────────
    # The core builds the export skeleton in the animation's units, so in HML
    # mode a mesh rig comes back scaled by scale_factor (IK and skeleton agree).
    # That is what the mesh-free skeleton-only HML armature needs. The skinned
    # HML export instead keeps the rig in native units: the exporter retargets
    # world-space onto the reverse-aligned mesh armature, and the pose
    # translations carry the scale as they always have.
    skeleton_only_from_tpose = skeleton_only and tpose_mesh is not None
    if skeleton_only_from_tpose and restore_space == "native":
        print(
            "Skeleton-only (native): using the T-pose armature as export rig "
            "and omitting meshes, preserving source node scale/local offsets"
        )
    elif skeleton_only_from_tpose and restore_space == "hml" and abs(ctx.scale_factor - 1.0) > 1e-12:
        print(
            "Skeleton-only (hml): skeleton offsets rescaled by scale_factor "
            f"{ctx.scale_factor:.6f} into normalized HML space to match the motion"
        )
    elif not skeleton_only and restore_space == "hml":
        from utils.roundtrip_common import build_skeleton

        skeleton = build_skeleton(
            ctx.export_joint_names,
            ctx.export_offsets,
            ctx.export_parents,
            ctx.export_rest_rotations,
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
        hml_scale = ctx.scale_factor
        hml_orientation = np.asarray(ctx.orientation_quat, dtype=np.float64).reshape(-1)
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
