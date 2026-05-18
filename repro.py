import os
import numpy as np
from dataset_fbx import get_motion
from skeleton import Skeleton

class MockArgs:
    augment_leaf_rotation_helpers = True
    object_type = 'Buffalo'
    window_size = 64
    fps = 30
    no_translation = False
    ignore_joint_names = None

args = MockArgs()
tp = Skeleton(args.object_type, args.augment_leaf_rotation_helpers)
motion = np.zeros((100, len(tp.names), 3))
np.save('d.npy', motion)
try:
    res = get_motion('d.npy', tp, args)
    print(f"1) {res[2]}")
    print(f"2) {tp.names[res[2]]}")
    cond = res[1]
    if isinstance(cond, np.ndarray):
        print(f"3) {cond[res[2]]}")
    elif isinstance(cond, dict) and res[2] in cond:
        print(f"3) {cond[res[2]]}")
    else:
        print(f"3) Not found in cond")
    print(f"4) {res[2] == 0}")
except Exception as e:
    import traceback
    traceback.print_exc()
finally:
    if os.path.exists('d.npy'):
        os.remove('d.npy')
