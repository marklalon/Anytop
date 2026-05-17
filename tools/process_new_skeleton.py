"""
Preprocess a new skeleton in T-pose-only mode.

This writes a structural-only cond package for inference/training on skeletons that are not part of the
processed Truebones dataset. Motion statistics are not estimated from target clips here; instead the script
reuses the shared structural prior bank stored next to the default training cond.npy.
"""
import sys, os, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.truebones.truebones_utils.motion_process import process_skeleton
from utils.misc import infer_object_type_from_filename
from utils.parser_util import process_new_skeleton_args


STRUCTURAL_NORM_PRIORS_FILE = "structural_norm_priors.npy"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRAINING_COND_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "truebones",
    "zoo",
    "truebones_processed",
    "cond.npy",
)


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

    tpose_path = args.tpos_path

    object_type = args.object_type
    if object_type is None:
        object_type = infer_object_type_from_filename(tpose_path)
        if object_type is None:
            raise FileNotFoundError(
                f"Cannot infer object-type from T-pose file '{tpose_path}'."
            )
        print(f"Auto-detected object_type: {object_type}")

    structural_prior_bank_path = os.path.join(
        os.path.dirname(DEFAULT_TRAINING_COND_PATH),
        STRUCTURAL_NORM_PRIORS_FILE,
    )

    process_skeleton(
        object_type,
        args.face_joints_names,
        args.save_dir,
        tpose_path,
        structural_prior_bank_path=structural_prior_bank_path,
    )

if __name__ == '__main__':
        main()
    