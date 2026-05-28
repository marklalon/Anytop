import argparse
import sys

from data_loaders.truebones.truebones_utils.param_utils import OBJECT_SUBSETS_DICT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-dir", default="", type=str,
                        help="Path to raw Truebones FBX folders. If not specified, uses default path.")
    parser.add_argument("--dataset-dir", default="", type=str,
                        help="Output directory for processed dataset. If not specified, uses default path.")
    parser.add_argument("--objects-subset", default="all", choices=sorted(OBJECT_SUBSETS_DICT.keys()), type=str,
                        help="Preprocess only a named object subset.")
    parser.add_argument("--object-types", nargs='+', default=None,
                        help="Preprocess only the specified object types.")
    parser.add_argument("--max-files-per-object", default=None, type=int,
                        help="Limit the number of FBX files processed per object for smoke tests.")
    # MP4 export removed - no --save-animations argument needed
    parser.add_argument("--object-workers", default=8, type=int,
                        help="Concurrent characters to preprocess. Defaults to 8.")
    parser.add_argument("--filter-min-length", default=10, type=int,
                        help="Minimum number of frames a motion clip must have; shorter clips are filtered out. Defaults to 10.")
    parser.add_argument("--resample-min-length", default=20, type=int,
                        help="When a motion has >= filter-min-length but < resample-min-length frames, resample it to resample-min-length frames. 0 disables. Defaults to 20.")
    args = parser.parse_args()

    objects = args.object_types
    if objects is None:
        objects = list(OBJECT_SUBSETS_DICT[args.objects_subset])

    from data_loaders.truebones.truebones_utils.motion_process import (
        DatasetPreprocessingError,
        create_data_samples,
    )

    try:
        create_data_samples(
            objects=objects,
            max_files_per_object=args.max_files_per_object,
            dataset_dir=args.dataset_dir or None,
            raw_data_dir=args.raw_data_dir or None,
            object_workers=args.object_workers,
            filter_min_length=args.filter_min_length,
            resample_min_length=args.resample_min_length,
        )
    except DatasetPreprocessingError:
        return 1

    # Preprocessing writes the seed artifacts regenerate needs (cond.npy,
    # motion_metadata.json, positions_error_rate.txt). Regeneration then
    # rebuilds the full side-artifact set from that seed state.
    from tools.regenerate_dataset_artifacts import regenerate_dataset_artifacts
    regenerate_dataset_artifacts(dataset_dir=args.dataset_dir or None)

    return 0


if __name__=="__main__":
    sys.exit(main())