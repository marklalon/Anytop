import argparse
import os

from data_loaders.truebones.truebones_utils.param_utils import FACE_JOINTS, OBJECT_SUBSETS_DICT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-dir", default="", type=str,
                        help="Path to raw Truebones BVH folders. Can also be set via ANYTOP_RAW_DATA_DIR.")
    parser.add_argument("--dataset-dir", default="", type=str,
                        help="Output directory for processed dataset. Can also be set via ANYTOP_DATASET_DIR.")
    parser.add_argument("--objects-subset", default="all", choices=sorted(OBJECT_SUBSETS_DICT.keys()), type=str,
                        help="Preprocess only a named object subset.")
    parser.add_argument("--object-types", nargs='+', default=None,
                        help="Preprocess only the specified object types.")
    parser.add_argument("--max-files-per-object", default=None, type=int,
                        help="Limit the number of BVH files processed per object for smoke tests.")
    # MP4 export removed - no --save-animations argument needed
    parser.add_argument("--object-workers", default=8, type=int,
                        help="Concurrent characters to preprocess. Defaults to 8.")
    parser.add_argument("--file-workers", default=8, type=int,
                        help="Worker threads per character for BVH file processing. Defaults to 8.")
    args = parser.parse_args()

    if args.raw_data_dir:
        os.environ["ANYTOP_RAW_DATA_DIR"] = args.raw_data_dir
    if args.dataset_dir:
        os.environ["ANYTOP_DATASET_DIR"] = args.dataset_dir

    objects = args.object_types
    if objects is None:
        objects = [obj for obj in OBJECT_SUBSETS_DICT[args.objects_subset] if obj in FACE_JOINTS]

    from data_loaders.truebones.truebones_utils.motion_process import create_data_samples

    create_data_samples(
        objects=objects,
        max_files_per_object=args.max_files_per_object,
        object_workers=args.object_workers,
        file_workers=args.file_workers,
        dataset_dir=args.dataset_dir or None,
    )


if __name__=="__main__":
    main()