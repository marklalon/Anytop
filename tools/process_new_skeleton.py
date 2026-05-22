""" 
We provide a preprocessing code for skeletons outside the Truebones dataset. 
While designed to be as generic as possible, some skeleton-specific adjustments may be needed since it 
was originally tailored for Truebones. For example, it relies on joint names for foot classification 
and specific velocity/height thresholds for foot contact detection. However, we have tested it on FBX 
files from Mixamo and other sources to ensure its generalizability.

Input Arguments:
object_type - A character's species/type name (e.g., "Dog"). Optional — inferred from filenames when omitted.
anim-dir - Directory containing animation files (FBX/GLB/GLTF) of the skeleton. More files improve statistical accuracy for motion denormalization.
face-joints-names - Optional manual override for four joints defining skeleton orientation ([right hip, left hip, right shoulder, left shoulder] or equivalent). 
            When omitted, preprocessing tries to infer them from semantic joint names. If inference is ambiguous,
            pass the four joint names explicitly. 
save-dir - Output directory.
tpos-path - An FBX/GLB/GLTF file of the character's natural rest pose for meaningful rotation learning.
        If missing, the code selects a pose from the provided files in --anim-dir.
update - Incremental update flag. Without it, --save-dir is wiped and rebuilt from scratch.
        With it, the existing dataset is kept and motions are added/replaced:
    --anim-dir merges the newly processed clips into the existing dataset,
    replacing only prior clips from the same source files (untouched clips kept);
        --retarget-top-k adds the newly generated donor clips alongside existing ones.
        Side artifacts (cond.npy embeddings + mean/std, motion_metadata.json, metadata.txt)
    are then rebuilt over the merged clip set. Requires an existing --save-dir/cond.npy;
    for --anim-dir, existing target clips must also be tracked in motion_metadata.json
    with explicit source metadata, otherwise the script exits instead of mixing old and
    new target motions. Falls back to a full build only when no dataset exists yet.

Output:
The code will create the following under save_dir:
save_dir/
        |_motions
        |_bvhs
        cond.npy
1. In motions directory, you will find npy files, which are the processed motion features of each input clip. 
This is useful in case you would like to use this data for training. 
2. In bvhs directory you can find BVH previews exported from the processed animation representation.
3. cond.npy contains the skeletons representation, including joints names embeddings and graph conditions,
which is given as input to AnyTop during inference. Please follow sampling instructions in README. 
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
        # Clear old files in the target directory before processing
        for entry in os.listdir(save_dir):
            entry_path = os.path.join(save_dir, entry)
            if os.path.isdir(entry_path):
                shutil.rmtree(entry_path)
            else:
                os.remove(entry_path)
        print(f"Cleared existing files in {save_dir}")
    else:
        os.makedirs(save_dir, exist_ok=True)

    # Resolve tpose_path: auto-select from anim_dir when not provided
    tpose_path = args.tpos_path
    if tpose_path is None or tpose_path == '':
        if args.anim_dir is None:
            raise FileNotFoundError(
                "Either --tpos-path or --anim-dir must be provided. "
                "--tpos-path is required for T-pose-only mode, and --anim-dir "
                "is required to auto-select a T-pose reference."
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
        print(f"Auto-selected T-pose reference: {tpose_path}")

    object_type = args.object_type
    if object_type is None:
        object_type = infer_object_type_from_filename(tpose_path)
        if object_type is None:
            raise FileNotFoundError(
                f"Cannot infer object-type from T-pose file '{tpose_path}'."
            )
        print(f"Auto-detected object_type: {object_type}")

    # If --tpos-path is given without --retarget-top-k, default to retarget-top-k 1
    # (prefer retarget over T-pose-only mode with no motions)
    # If --donor-skeletons is given, also default to 1 but suppress the log
    # since the user is explicitly configuring retarget.
    retarget_top_k = args.retarget_top_k
    if retarget_top_k is None and args.anim_dir is None:
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
        )
        process_skeleton(
            object_type,
            args.face_joints_names,
            args.save_dir,
            tpose_path,
            motions_from_npys=result['retargeted_npys'],
            target_cond_partial=result['target_cond'],
            update=update_mode,
        )
    else:
        process_skeleton(
            object_type,
            args.face_joints_names,
            args.save_dir,
            tpose_path,
            args.anim_dir,
            update=update_mode,
        )

    # In --update mode process_skeleton only writes motions plus a provisional
    # cond.npy / motion_metadata. Rebuild the side artifacts over the merged
    # clip set, recomputing mean/std so normalization reflects the added motions.
    if update_mode:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from regenerate_dataset_artifacts import regenerate_dataset_artifacts
        print("[process_new_skeleton] --update: rebuilding side artifacts "
              "(embeddings, mean/std, metadata)")
        regenerate_dataset_artifacts(args.save_dir, recompute_stats=True)

if __name__ == '__main__':
    try:
        main()
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception as exc:
        print(f"\n[process_new_skeleton] Error: {exc}", file=sys.stderr)
        sys.exit(1)
    