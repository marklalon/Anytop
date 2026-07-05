"""
align_tpose_orientation.py

Force every action FBX/GLB in a directory to face the same orientation as the
character's T-pose, then export each one as a same-named skinned GLB.

The orientation logic mirrors the dataset preprocessing pipeline
(``data_loaders/truebones/.../dataset_pipeline.py``):

    1. List the FBX/GLB/GLTF action files in the input directory.
    2. Load the T-pose bind pose from ``--bind-pose`` to derive orientation.
    3. Filter with ``should_skip_anim`` to keep only valid action clips.
    4. Compute the T-pose ``orientation_quat`` and apply to every action via
       ``rotate_to_hml_orientation``.
    5. Re-export each rotated action onto its own rig/skin as a GLB.

The bind pose is used only to derive the orientation and is not exported.

Requires bpy (Blender as a Python module) — run with the project's .venv:

    .venv/Scripts/python.exe Anytop/tools/dataset_cleanup/align_tpose_orientation.py --dir <folder> --bind-pose <path>
"""

from __future__ import annotations

import argparse
import os
import sys

# Put the Anytop package root on sys.path so the data_loaders/motion_lib/utils
# imports resolve to the Anytop copies (NOT the top-level pcvg `utils`).
_ANYTOP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ANYTOP_DIR not in sys.path:
    sys.path.insert(0, _ANYTOP_DIR)


def _list_mesh_files(directory: str) -> list[str]:
    """List all FBX/GLB/GLTF files in *directory*, sorted."""
    _extensions = (".fbx", ".glb", ".gltf")
    return sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.lower().endswith(_extensions)
    )


def align_directory(
    directory: str,
    object_type: str | None = None,
    output_dir: str | None = None,
    overwrite: str = "skip",
    bind_pose_path: str | None = None,
) -> list[str]:
    """Align all action FBX/GLB files in *directory* to the T-pose orientation and export GLBs.

    Args:
        directory:      Folder holding the character's action files
                        (FBX, GLB, or GLTF).
        object_type:    Species/type key used to strip filename prefixes during
                        action filtering. Defaults to the directory's basename
                        (matches the Truebones raw_data_dir/<object_type> layout).
        output_dir:     Where to write the GLBs. Defaults to *directory* (same-named).
        overwrite:      "force" (overwrite existing), or "skip" (default, skip existing).
        bind_pose_path: Path to the T-pose bind pose file (FBX/GLB/GLTF).
                        Used to derive the orientation quat; not exported.

    Returns:
        Absolute paths of the GLB files written.
    """
    import numpy as np

    from motion_lib import FBX
    from data_loaders.truebones.truebones_utils.fbx_filename_rules import (
        should_skip_anim,
    )
    from data_loaders.truebones.truebones_utils.features import (
        get_common_features_from_T_pose,
        _rest_pose_animation_from_loaded_anim,
    )
    from data_loaders.truebones.truebones_utils.face_orientation import (
        rotate_to_hml_orientation,
        resolve_face_joints,
        resolve_forward_reference_joints,
        snap_forward_alignment_quat,
        _get_facing_forward,
    )
    from motion_lib.Animation import positions_global
    from motion_lib.FBX import (
        collapse_root_skeleton,
        extract_armature_skeleton_data,
        load_fbx_scene,
    )
    from utils.exporter import AnimationExporter, animation_to_exporter_inputs
    from utils.roundtrip_common import build_skeleton

    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Input directory not found: {directory}")

    if bind_pose_path is None:
        raise ValueError("--bind-pose is required; specify the T-pose bind pose file path.")
    bind_pose_path = os.path.abspath(bind_pose_path)
    if not os.path.isfile(bind_pose_path):
        raise FileNotFoundError(f"Bind pose file not found: {bind_pose_path}")

    if object_type is None:
        object_type = os.path.basename(os.path.normpath(directory))
    if output_dir is None:
        output_dir = directory
    os.makedirs(output_dir, exist_ok=True)

    anim_files = _list_mesh_files(directory)
    if not anim_files:
        raise FileNotFoundError(f"No .fbx/.glb/.gltf files found in {directory}")

    print(f"Object type        : {object_type}")
    print(f"Bind pose          : {os.path.basename(bind_pose_path)}")

    action_files = [f for f in anim_files if not should_skip_anim(f, object_type)]
    if not action_files:
        raise RuntimeError(f"No valid action FBX files after filtering in {directory}")
    print(f"Action clips        : {len(action_files)}")

    bind_anim, bind_names, _bind_frame_time = FBX.load(bind_pose_path)
    uncropped_joint_cap = max(len(bind_names), 1)
    tp = get_common_features_from_T_pose(
        bind_pose_path,
        object_type,
        max_joints=uncropped_joint_cap,
    )

    # T-pose rest/bind-pose forward, computed with the same face/forward joints
    # that produced tp.orientation_quat so native-matching clips map to ~identity.
    # Use the rest pose (bind pose), NOT frame 0 of the animation, because some
    # bind-pose files carry non-identity root rotation at frame 0 (e.g. Crab
    # has +X rest pose but -Z at frame 0).
    ref_rest_anim = _rest_pose_animation_from_loaded_anim(bind_anim)
    tpose_forward = _get_facing_forward(
        positions_global(ref_rest_anim),
        object_type,
        face_joint_indx=tp.face_joints,
        forward_joint_index=tp.forward_joint_index,
        forward_base_joint_index=tp.forward_base_joint_index,
        emit_warnings=False,
    )

    def _clip_alignment_quat(anim, names):
        """Rotation mapping this clip's frame-0 facing onto the T-pose rest-pose facing."""
        if tpose_forward is None:
            return tp.orientation_quat
        clip_positions = positions_global(anim[:1])
        fj = resolve_face_joints(object_type, names, anim.parents, rest_positions=clip_positions)
        fwd_j, fwd_b = resolve_forward_reference_joints(
            names, anim.parents, object_type=object_type, rest_positions=clip_positions
        )
        clip_forward = _get_facing_forward(
            clip_positions,
            object_type,
            face_joint_indx=fj,
            forward_joint_index=fwd_j,
            forward_base_joint_index=fwd_b,
            emit_warnings=False,
        )
        if clip_forward is None:
            return tp.orientation_quat
        # Align the clip's dominant axis onto the T-pose's dominant axis using
        # only a 90-degree-multiple turn, matching the canonical orientation_quat.
        return snap_forward_alignment_quat(clip_forward, tpose_forward)[0]

    written: list[str] = []
    for fbx_path in action_files:
        stem = os.path.splitext(os.path.basename(fbx_path))[0]
        output_glb = os.path.abspath(os.path.join(output_dir, f"{stem}.glb"))

        if os.path.exists(output_glb):
            if overwrite == "skip":
                print(f"[skip] {stem}.glb exists")
                continue

        print(f"[align] {os.path.basename(fbx_path)} -> {stem}.glb")
        anim, names, frametime = FBX.load(fbx_path)
        aligned = rotate_to_hml_orientation(anim, _clip_alignment_quat(anim, names))

        # Rest rotations matching FBX.load's collapsed skeleton, so the exporter's
        # source rest geometry matches the FBX bind pose and the world-space
        # retarget onto this same rig stays near-identity (no spurious basis flip).
        raw_names, raw_parents, raw_offsets, raw_rest_rotations = extract_armature_skeleton_data(
            load_fbx_scene(fbx_path)
        )
        collapsed_rest_rotations = collapse_root_skeleton(
            list(raw_names),
            np.asarray(raw_parents, dtype=np.int32),
            np.asarray(raw_offsets, dtype=np.float64),
            np.asarray(raw_rest_rotations, dtype=np.float64)[None, ...],
            np.asarray(raw_offsets, dtype=np.float64)[None, ...],
        )[3][0]

        skeleton = build_skeleton(
            names,
            aligned.offsets,
            aligned.parents,
            collapsed_rest_rotations,
        )
        joint_rotations, root_translation, root_rotation, bone_translations = (
            animation_to_exporter_inputs(aligned, skeleton)
        )

        fps = 1.0 / frametime if frametime and frametime > 0 else 30.0
        exporter = AnimationExporter(skeleton, fps=fps)
        exporter.export_glb(
            joint_rotations,
            root_translation,
            root_rotation,
            output_glb,
            mesh_path=fbx_path,
            bone_translations=bone_translations,
        )
        written.append(output_glb)

    print(f"\nDone. Wrote {len(written)} GLB file(s) to {os.path.abspath(output_dir)}")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Align every action FBX/GLB in a directory to the character's T-pose "
            "orientation and export same-named skinned GLB files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dir", required=True,
        help="Directory containing the action FBX/GLB/GLTF files.",
    )
    parser.add_argument(
        "--bind-pose", required=True,
        help="Path to the T-pose bind pose file (FBX/GLB/GLTF). Used for orientation; not exported.",
    )
    parser.add_argument(
        "--object-type", default=None,
        help="Species/type key for filename filtering. Defaults to the directory basename.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Where to write GLBs. Defaults to the input directory (same-named output).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing GLB output files.",
    )
    args = parser.parse_args()

    overwrite = "force" if args.overwrite else "skip"

    align_directory(
        directory=args.dir,
        object_type=args.object_type,
        output_dir=args.output_dir,
        overwrite=overwrite,
        bind_pose_path=args.bind_pose,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
