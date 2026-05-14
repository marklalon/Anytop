""" 
We provide a preprocessing code for skeletons outside the Truebones dataset. 
While designed to be as generic as possible, some skeleton-specific adjustments may be needed since it 
was originally tailored for Truebones. For example, it relies on joint names for foot classification 
and specific velocity/height thresholds for foot contact detection. However, we have tested it on FBX 
files from Mixamo and other sources to ensure its generalizability.

Input Arguments:
object_type - A character's species/type name (e.g., "Dog"). Optional — inferred from FBX filenames when omitted.
fbx-dir - Directory containing FBX files of the skeleton. More files improve statistical accuracy for motion denormalization.
face-joints-names - Optional manual override for four joints defining skeleton orientation ([right hip, left hip, right shoulder, left shoulder] or equivalent). 
            When omitted, preprocessing tries to infer them from semantic joint names in the FBX. If inference is ambiguous,
            pass the four joint names explicitly. 
save-dir - Output directory.
tpos-fbx - An FBX file of the character's natural rest pose for meaningful rotation learning. 
        If missing, the code selects a pose from the provided FBX files.
        
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

from data_loaders.truebones.truebones_utils.motion_process import process_skeleton
from data_loaders.truebones.truebones_utils.fbx_filename_rules import find_tpose_reference_path
from utils.misc import infer_object_type_from_filename
from utils.parser_util import process_new_skeleton_args


def main():
    args = process_new_skeleton_args()

    # Clear old files in the target directory before processing
    save_dir = args.save_dir
    if os.path.exists(save_dir):
        for entry in os.listdir(save_dir):
            entry_path = os.path.join(save_dir, entry)
            if os.path.isdir(entry_path):
                shutil.rmtree(entry_path)
            else:
                os.remove(entry_path)
        print(f"Cleared existing files in {save_dir}")
    else:
        os.makedirs(save_dir, exist_ok=True)

    # Resolve tpos_fbx: auto-select from fbx_dir when not provided
    tpos_fbx = args.tpos_fbx
    if tpos_fbx is None or tpos_fbx == '':
        if args.fbx_dir is None:
            raise FileNotFoundError(
                "Either --tpos-fbx or --fbx-dir must be provided. "
                "--tpos-fbx is required for T-pose-only mode, and --fbx-dir "
                "is required to auto-select a T-pose reference."
            )
        fbx_files = sorted([
            os.path.join(args.fbx_dir, f)
            for f in os.listdir(args.fbx_dir)
            if f.lower().endswith('.fbx')
        ])
        if len(fbx_files) == 0:
            raise FileNotFoundError(
                f"No FBX files found in --fbx-dir '{args.fbx_dir}'."
            )
        tpos_fbx = find_tpose_reference_path(fbx_files)
        print(f"Auto-selected T-pose reference: {tpos_fbx}")

    object_type = args.object_type
    if object_type is None:
        object_type = infer_object_type_from_filename(tpos_fbx)
        if object_type is None:
            raise FileNotFoundError(
                f"Cannot infer object-type from T-pose FBX '{tpos_fbx}'."
            )
        print(f"Auto-detected object_type: {object_type}")

    # If --tpos-fbx is given without --retarget-top-k, default to retarget-top-k 1
    # (prefer retarget over T-pose-only mode with no motions)
    retarget_top_k = args.retarget_top_k
    auto_defaulted_retarget = False
    if retarget_top_k is None and args.fbx_dir is None:
        retarget_top_k = 1
        auto_defaulted_retarget = True
        print(f"[process_new_skeleton] No --retarget-top-k specified, defaulting to 1")

    if retarget_top_k:
        if args.fbx_dir:
            raise SystemExit(
                "Error: --retarget-top-k and --fbx-dir are mutually exclusive. "
                "--retarget-top-k auto-generates motion data from retargeted donors."
            )
        from utils.auto_retarget import auto_retarget_pipeline
        result = auto_retarget_pipeline(
            target_object_type=object_type,
            target_tpose_fbx=tpos_fbx,
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
            tpos_fbx,
            motions_from_npys=result['retargeted_npys'],
            target_cond_partial=result['target_cond'],
        )
    else:
        process_skeleton(
            object_type,
            args.face_joints_names,
            args.save_dir,
            tpos_fbx,
            args.fbx_dir,
        )

if __name__ == '__main__':
        main()
    