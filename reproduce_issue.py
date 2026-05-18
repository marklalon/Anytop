import os
import numpy as np
import torch
from dataset_fbx import get_motion
from skeleton import Skeleton

class MockArgs:
    def __init__(self):
        self.augment_leaf_rotation_helpers = True
        self.object_type = 'Buffalo'
        self.window_size = 64
        self.fps = 30
        self.no_translation = False
        self.ignore_joint_names = None

args = MockArgs()
tp = Skeleton(args.object_type, args.augment_leaf_rotation_helpers)

# get_motion expects a .npy file with shape (frames, joints, 3) or similar
# Buffalo joints count:
num_joints = len(tp.names)
motion = np.zeros((100, num_joints, 3))
dummy_path = 'dummy_buffalo.npy'
np.save(dummy_path, motion)

try:
    res = get_motion(dummy_path, tp, args)
    motion_out, cond, translation_root_index = res
    
    print(f"1) translation_root_index: {translation_root_index}")
    print(f"2) joint name: {tp.names[translation_root_index]}")
    
    # Check cond structure
    if isinstance(cond, np.ndarray):
        print(f"3) cond entry: {cond[translation_root_index] if translation_root_index < len(cond) else 'OOB'}")
    else:
        print(f"3) cond entry: {cond}")

    print(f"4) equals 0: {translation_root_index == 0}")

finally:
    if os.path.exists(dummy_path):
        os.remove(dummy_path)
