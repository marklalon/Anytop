""" 
We provide a preprocessing code for skeletons outside the Truebones dataset. 
While designed to be as generic as possible, some skeleton-specific adjustments may be needed since it 
was originally tailored for Truebones. For example, it relies on joint names for foot classification 
and specific velocity/height thresholds for foot contact detection. However, we have tested it on FBX 
files from Mixamo and other sources to ensure its generalizability.

Input Arguments:
object_type - A character's species/type name (e.g., "Dog").
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
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.truebones.truebones_utils.motion_process import process_skeleton
from utils.parser_util import process_new_skeleton_args

def main():
    args = process_new_skeleton_args()
    process_skeleton(args.object_type, args.fbx_dir, args.face_joints_names, args.save_dir, tpos_bvh=args.tpos_fbx)
    
if __name__ == '__main__':
        main()
    