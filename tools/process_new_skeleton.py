"""
Inference-Time Skeleton Preprocessing
=======================================
Prepares a skeleton outside the Truebones dataset for AnyTop *inference* (generation).
The output cond.npy is designed to be passed via ``--cond-path`` to ``generate.py``.

.. warning::
   The generated data is **not suitable for training**. Skeleton cropping
   (MAX_JOINTS=100) is disabled by default (``--crop-enabled`` to enable). Use
   ``preprocess_and_validate.py`` for training dataset creation.

While designed to be as generic as possible, some skeleton-specific adjustments may be
needed since it was originally tailored for Truebones (joint-name-based foot classification,
velocity/height thresholds for foot contact detection). Tested on FBX from Mixamo and other
sources.

Input Arguments:
tpos-path         - An FBX/GLB/GLTF file whose bind/rest pose defines the NPY encoding base (required).
save-dir          - Output directory (required).
object_type       - Species/type name (e.g. "Dragon"). Inferred from tpos-path filename when omitted.
species-tags      - Comma-separated explicit species tags for --object-type,
                    e.g. 'Quadruped,Large,Lumbering'. When specified, takes
                    precedence over species_tags.jsonl.
crop-enabled      - Enable skeleton cropping to MAX_JOINTS=100.
                    Off by default (inference has no joint cap).

Output (under save_dir/):
  cond.npy    - Skeleton representation (joint name embeddings, graph conditions,
                canonical feature-space metadata)
                consumed by AnyTop inference via ``--cond-path``.
"""
import sys, os, shutil
from typing import Any
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.truebones.truebones_utils.motion_process import process_skeleton
from utils.misc import infer_object_type_from_filename
from utils.parser_util import process_new_skeleton_args

def process_new_skeleton(
    *,
    save_dir: str,
    tpos_path: str,
    object_type: str | None = None,
    crop_enabled: bool = False,
    species_tags: str | None = None,
    skip_t5_embeddings: bool = False,
    yes: bool = False,
) -> dict[str, Any]:
    """Process a new skeleton for AnyTop inference and return resolved metadata.

    This is the programmatic equivalent of the CLI entry point and keeps the
    CLI behaviour unchanged while allowing server-side worker reuse.
    """
    args = type("ProcessNewSkeletonArgs", (), {
        "save_dir": save_dir,
        "object_type": object_type,
        "tpos_path": tpos_path,
        "crop_enabled": crop_enabled,
        "species_tags": species_tags,
        "skip_t5_embeddings": skip_t5_embeddings,
        "yes": yes,
    })()
    return _process_new_skeleton_from_args(args)


def _process_new_skeleton_from_args(args) -> dict[str, Any]:
    save_dir = args.save_dir

    if os.path.exists(save_dir):
        # Clear known data subdirectories (motions/, bvhs/, etc.) but preserve
        # top-level files.
        # Aligns with preprocess_and_validate.py behavior.
        known_subdirs = ['motions', 'bvhs', 'joint_name_inspection']
        existing_subdirs = [
            s for s in known_subdirs
            if os.path.isdir(os.path.join(save_dir, s)) and os.listdir(os.path.join(save_dir, s))
        ]

        if existing_subdirs:
            if not getattr(args, 'yes', False):
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
            else:
                print(f"[process_new_skeleton] clearing {len(existing_subdirs)} subdirectories...")

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

    tpose_path = args.tpos_path
    if not tpose_path:
        raise FileNotFoundError("--tpos-path is required. Provide a FBX/GLB/GLTF file whose "
                                "bind/rest pose defines the skeleton.")

    object_type = args.object_type
    if object_type is None:
        object_type = infer_object_type_from_filename(tpose_path)
        if object_type is None:
            raise FileNotFoundError(
                f"Cannot infer object-type from reference file '{tpose_path}'."
            )
        print(f"Auto-detected object_type: {object_type}")

    # Skeleton cropping: off by default (inference has no joint cap).
    # Use --crop-enabled to enable MAX_JOINTS=100 cropping.
    crop_enabled = args.crop_enabled

    # ── Species tags ─────────────────────────────────────────────────────
    import data_loaders.truebones.truebones_utils.physics_joint_annotation as _pja
    if args.species_tags is not None:
        parsed_tags = tuple(
            t.strip() for t in args.species_tags.split(',') if t.strip()
        )
        if parsed_tags:
            _pja._SPECIES_TAGS[object_type] = parsed_tags
            _pja._SPECIES_TAGS_LOWER = None  # invalidate lazy cache
            print(
                f"[process_new_skeleton] Using explicit --species-tags for "
                f"'{object_type}': {parsed_tags}"
            )
    # ──────────────────────────────────────────────────────────────────────

    process_skeleton(
        object_type,
        None,
        args.save_dir,
        tpose_path,
        crop_enabled=crop_enabled,
        skip_t5=args.skip_t5_embeddings,
    )

    return {
        "save_dir": save_dir,
        "object_type": object_type,
        "tpose_path": tpose_path,
        "cond_npy": os.path.join(save_dir, "cond.npy"),
    }


def main():
    args = process_new_skeleton_args()
    _process_new_skeleton_from_args(args)

if __name__ == '__main__':
    try:
        main()
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception as exc:
        print(f"\n[process_new_skeleton] Error: {exc}", file=sys.stderr)
        sys.exit(1)
    
