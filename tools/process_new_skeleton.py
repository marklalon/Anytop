"""
Inference-Time Skeleton Preprocessing
=======================================
Prepares a skeleton outside the Truebones dataset for AnyTop *inference* (generation).
The output cond.npy is designed to be passed via ``--cond-path`` to ``generate.py``.

.. warning::
   The generated data is **not suitable for training**. Skeleton cropping
   (MAX_JOINTS=100) is disabled by default (``--crop-enabled`` to enable), and the
   mean/std statistics are computed from retargeted donor motions or a limited set of
   input animations, which do not reflect the training distribution. Use
   ``preprocess_and_validate.py`` for training dataset creation.

While designed to be as generic as possible, some skeleton-specific adjustments may be
needed since it was originally tailored for Truebones (joint-name-based foot classification,
velocity/height thresholds for foot contact detection). Tested on FBX from Mixamo and other
sources.

Input Arguments:
object_type       - Species/type name (e.g. "Dragon"). Inferred from filenames when omitted.
anim-dir          - Directory with animation files (FBX/GLB/GLTF) of the skeleton.
                    More files improve mean/std accuracy for motion denormalization.
                    Mutually exclusive with --retarget-top-k.
tpos-path         - An FBX/GLB/GLTF file whose bind/rest pose defines the NPY encoding base.
                    When omitted (and --anim-dir given), auto-selects a reference carrier
                    from anim files (T-pose/rest/bind > idle > walk > first).
face-joints-names - Manual override for the four orientation joints
                    ([right hip, left hip, right shoulder, left shoulder] or equivalent).
save-dir          - Output directory (required).
retarget-top-k    - Auto-select the top-k most similar training skeletons as motion donors,
                    retarget their motions to the new skeleton, and use those coarse motions
                    to compute mean/std. Mutually exclusive with --anim-dir.
training-cond-path - Path to the training dataset's cond.npy for donor selection when
                     --retarget-top-k is set. (default: dataset/truebones/.../cond.npy)
donor-skeletons   - Comma-separated donor names to use instead of auto-selection,
                    e.g. 'Bison,Cow,Horse'. Only effective with --retarget-top-k.
crop-enabled      - Enable skeleton cropping to MAX_JOINTS=100.
                    Off by default (inference has no joint cap).
update            - Incremental mode: merge new clips into the existing dataset instead of
                    wiping --save-dir. Requires an existing dataset at --save-dir.
                    Supports both --anim-dir and --retarget-top-k.

Output (under save_dir/):
  motions/    - .npy files of processed motion features for each input clip.
  bvhs/       - BVH previews exported from the processed animation representation.
  cond.npy    - Skeleton representation (joint name embeddings, graph conditions, mean/std)
                consumed by AnyTop inference via ``--cond-path``.
"""
import sys, os, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.truebones.truebones_utils.motion_process import (
    process_skeleton,
    validate_anim_dir_update_state,
)
from data_loaders.truebones.truebones_utils.fbx_filename_rules import find_tpose_reference_path
from utils.misc import infer_object_type_from_filename
from utils.parser_util import process_new_skeleton_args


def main():
    args = process_new_skeleton_args()
    save_dir = args.save_dir

    # --update: incremental mode. Keeps the existing dataset and adds/replaces
    # motions instead of wiping --save-dir. Works for both the --anim-dir path
    # and the --retarget-top-k path; side artifacts are rebuilt afterwards.
    update_mode = bool(getattr(args, 'update', False))
    if update_mode and not os.path.exists(os.path.join(save_dir, 'cond.npy')):
        print(f"[process_new_skeleton] --update requested but no existing dataset "
              f"found in '{save_dir}'; performing a full build instead.")
        update_mode = False

    if update_mode:
        os.makedirs(save_dir, exist_ok=True)
    elif os.path.exists(save_dir):
        # Clear known data subdirectories (motions/, bvhs/, etc.) but preserve
        # top-level files (cond.npy, metadata.txt, motion_metadata.json, etc.).
        # Aligns with preprocess_and_validate.py behavior.
        known_subdirs = ['motions', 'bvhs', 'joint_name_inspection']
        existing_subdirs = [
            s for s in known_subdirs
            if os.path.isdir(os.path.join(save_dir, s)) and os.listdir(os.path.join(save_dir, s))
        ]

        if existing_subdirs:
            print("\n" + "=" * 70)
            print("WARNING: Existing preprocessed data detected")
            print("=" * 70)
            print(f"Dataset directory: {save_dir}")
            print(f"Subdirectories to clear ({len(existing_subdirs)}): {', '.join(existing_subdirs)}")
            print("\nDo you want to delete the matching subdirectories and proceed?")
            reply = input("Enter 'yes' to delete and continue, or 'no' to abort: ")
            if reply.strip().lower() not in ('y', 'yes'):
                print("\nAborted by user.")
                sys.exit(0)

            print("\nDeleting...")
            cleared = []
            for subdir in existing_subdirs:
                subdir_path = os.path.join(save_dir, subdir)
                shutil.rmtree(subdir_path)
                cleared.append(subdir)
            print(f"Done. Cleared: {', '.join(cleared)}\n")
        else:
            print(f"No existing data subdirectories found in {save_dir}")
    else:
        os.makedirs(save_dir, exist_ok=True)

    # Resolve tpose_path: auto-select a rest-pose reference carrier from anim_dir when not provided
    tpose_path = args.tpos_path
    if tpose_path is None or tpose_path == '':
        if args.anim_dir is None:
            raise FileNotFoundError(
                "Either --tpos-path or --anim-dir must be provided. "
                "--tpos-path is required for rest-pose-only mode, and --anim-dir "
                "is required to auto-select a rest-pose reference."
            )
        anim_files = sorted([
            os.path.join(args.anim_dir, f)
            for f in os.listdir(args.anim_dir)
            if f.lower().endswith(('.fbx', '.glb', '.gltf'))
        ])
        if len(anim_files) == 0:
            raise FileNotFoundError(
                f"No animation files (.fbx/.glb/.gltf) found in --anim-dir '{args.anim_dir}'."
            )
        tpose_path = find_tpose_reference_path(anim_files)
        print(f"Auto-selected rest-pose reference carrier: {tpose_path}")

    object_type = args.object_type
    if object_type is None:
        object_type = infer_object_type_from_filename(tpose_path)
        if object_type is None:
            raise FileNotFoundError(
                f"Cannot infer object-type from reference file '{tpose_path}'."
            )
        print(f"Auto-detected object_type: {object_type}")

    # If --tpos-path is given without --retarget-top-k, default to retarget-top-k 1
    # (prefer retarget over rest-pose-only mode with no motions)
    # If --donor-skeletons is given, also default to 1 but suppress the log
    # since the user is explicitly configuring retarget.
    # Skeleton cropping: off by default (inference has no joint cap).
    # Use --crop-enabled to enable MAX_JOINTS=100 cropping.
    crop_enabled = args.crop_enabled

    retarget_top_k = args.retarget_top_k
    if retarget_top_k == 0 and (args.donor_skeletons or args.anim_dir):
        raise SystemExit(
            "Error: --retarget-top-k 0 (rest-pose-only) is mutually exclusive with "
            "--donor-skeletons and --anim-dir (it builds graph metadata from the "
            "--tpos-path skeleton alone, with no motions)."
        )

    if retarget_top_k == 0:
        # Rest-pose-only: no donor retargeting, no motions/. Graph metadata is a
        # pure function of the skeleton topology; the donors only ever fed the
        # position mean/std, which Video2Pose no longer consumes.
        print("[process_new_skeleton] --retarget-top-k 0: rest-pose-only build "
              "(graph metadata from --tpos-path, no donor motions)")
    elif retarget_top_k is None and args.anim_dir is None:
        retarget_top_k = 1
        if args.donor_skeletons is None or args.donor_skeletons.strip() == '':
            print(f"[process_new_skeleton] No --retarget-top-k specified, defaulting to 1")

    if update_mode and args.anim_dir:
        try:
            validate_anim_dir_update_state(object_type, save_dir)
        except RuntimeError as exc:
            raise SystemExit(f"Error: {exc}") from exc

    if update_mode:
        print(f"[process_new_skeleton] --update: incrementally updating {save_dir} "
              f"(existing clips preserved)")

    if retarget_top_k:
        if args.anim_dir:
            raise SystemExit(
                "Error: --retarget-top-k and --anim-dir are mutually exclusive. "
                "--retarget-top-k auto-generates motion data from retargeted donors."
            )
        from utils.auto_retarget import auto_retarget_pipeline
        result = auto_retarget_pipeline(
            target_object_type=object_type,
            target_tpose_path=tpose_path,
            save_dir=args.save_dir,
            top_k=retarget_top_k,
            training_cond_path=args.training_cond_path,
            face_joints_names=args.face_joints_names,
            donor_skeletons_override=(
                [s.strip() for s in args.donor_skeletons.split(',')]
                if args.donor_skeletons else None
            ),
            crop_enabled=crop_enabled,
        )
        process_skeleton(
            object_type,
            args.face_joints_names,
            args.save_dir,
            tpose_path,
            motions_from_npys=result['retargeted_npys'],
            target_cond_partial=result['target_cond'],
            update=update_mode,
            crop_enabled=crop_enabled,
        )
    else:
        process_skeleton(
            object_type,
            args.face_joints_names,
            args.save_dir,
            tpose_path,
            args.anim_dir,
            update=update_mode,
            crop_enabled=crop_enabled,
        )

    # In --update mode process_skeleton only writes motions plus a provisional
    # cond.npy / motion_metadata. Rebuild the side artifacts over the merged
    # clip set, recomputing mean/std so normalization reflects the added motions.
    if update_mode:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from regenerate_dataset_artifacts import regenerate_dataset_artifacts
        print("[process_new_skeleton] --update: rebuilding side artifacts "
              "(embeddings, mean/std, metadata)")
        # Only this object's clips changed, so scope the mean/std recompute to it.
        regenerate_dataset_artifacts(
            args.save_dir, recompute_stats=True, recompute_stats_objects={object_type}
        )

if __name__ == '__main__':
    try:
        main()
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception as exc:
        print(f"\n[process_new_skeleton] Error: {exc}", file=sys.stderr)
        sys.exit(1)
    