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
species-tags      - Comma-separated species tags (motion descriptor) for --object-type,
                    e.g. 'Quadruped,Large,Lumbering'. REQUIRED: it defines the
                    descriptor baked into cond.npy. There is no fallback to the
                    default dataset's species_tags.jsonl.
crop-enabled      - Enable skeleton cropping to MAX_JOINTS=100.
                    Off by default (inference has no joint cap).
reference-cond-path - REQUIRED. cond.npy to inherit the per-object_subset
                    standardization statistics from. Those statistics belong to a
                    trained checkpoint, so pass the checkpoint's own cond.npy
                    snapshot. There is no fallback to the processed dataset dir.

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

# Trailing stem tokens that name the *pose*, not the species, so dropping them is
# expected and needs no warning ("Horse_Tpose.fbx" -> "Horse").
_POSE_STEM_SUFFIXES = frozenset({
    "tpose", "apose", "pose", "rest", "restpose", "bind", "bindpose",
    "rig", "skeleton", "ref", "reference", "all",
})

def _upsert_species_tags_sidecar(save_dir: str, species: str, tags) -> str:
    """Write/update the ``species_tags.jsonl`` sidecar for one species.

    The sidecar is the single source of truth the cond bakes its ``species_tags``
    field from, so a new skeleton must have its entry here. Upserts: replaces an
    existing line for *species*, appends if absent, and leaves other species'
    lines untouched. Returns the sidecar path.
    """
    import json
    path = os.path.join(save_dir, "species_tags.jsonl")
    kept = []
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    kept.append(line)
                    continue
                if str(record.get("species", "")).strip() == species:
                    continue  # drop the stale line; the fresh one is rewritten below
                kept.append(line)
    kept.append(json.dumps(
        {"species": species, "species_tags": list(tags)}, ensure_ascii=False
    ))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(kept) + "\n")
    return path

def process_new_skeleton(
    *,
    save_dir: str,
    tpos_path: str,
    reference_cond_path: str,
    object_type: str | None = None,
    crop_enabled: bool = False,
    species_tags: str | None = None,
    skip_t5_embeddings: bool = False,
    yes: bool = False,
) -> dict[str, Any]:
    """Process a new skeleton for AnyTop inference and return resolved metadata.

    This is the programmatic equivalent of the CLI entry point and keeps the
    CLI behaviour unchanged while allowing server-side worker reuse.

    ``reference_cond_path`` is required: the per-object_subset standardization
    statistics are inherited from a trained checkpoint's cond.npy snapshot (there
    is no fallback to the processed dataset directory).
    """
    args = type("ProcessNewSkeletonArgs", (), {
        "save_dir": save_dir,
        "object_type": object_type,
        "tpos_path": tpos_path,
        "crop_enabled": crop_enabled,
        "species_tags": species_tags,
        "skip_t5_embeddings": skip_t5_embeddings,
        "reference_cond_path": reference_cond_path,
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
        # This is the one inference with no cond.npy to validate against -- a new
        # skeleton is by definition not registered anywhere yet -- so the filename
        # is split blindly at the first underscore. Say so out loud when the rest
        # of the stem is not just a pose suffix, because a multi-token species
        # ("FEP_MagmaDemon_Tpose.glb") would silently register as "FEP".
        _stem = os.path.splitext(os.path.basename(tpose_path))[0]
        _remainder = _stem[len(object_type):].strip("_-. ")
        if _remainder and _remainder.lower().replace("-", "") not in _POSE_STEM_SUFFIXES:
            print(
                f"[WARN] '{_stem}' carries more than the species name; only "
                f"'{object_type}' was taken and '{_remainder}' dropped. A new "
                f"skeleton has no cond.npy to validate the name against -- pass "
                f"--object-type explicitly if that is wrong."
            )
        print(f"Auto-detected object_type: {object_type}")

    # Skeleton cropping: off by default (inference has no joint cap).
    # Use --crop-enabled to enable MAX_JOINTS=100 cropping.
    crop_enabled = args.crop_enabled

    # ── Species tags (required) ────────────────────────────────────────────
    # A new skeleton must carry its own motion descriptor. There is no fallback
    # to the default dataset's species_tags.jsonl -- that would silently borrow a
    # same-named species' tags. Register the tags into the process snapshot (so
    # the species_emb is encoded from them) and write the sidecar (the single
    # source of truth the cond bakes its species_tags field from).
    from data_loaders.truebones.truebones_utils import dataset_tags
    raw_tags = str(getattr(args, 'species_tags', '') or '').strip()
    if not raw_tags:
        raise ValueError(
            "--species-tags is required for a new skeleton. It defines the motion "
            "descriptor (body-plan, size, locomotion) baked into cond.npy. There is "
            "no fallback to the default dataset's tags."
        )
    parsed_tags = tuple(t.strip() for t in raw_tags.split(',') if t.strip())
    if not parsed_tags:
        raise ValueError("--species-tags must contain at least one non-empty tag.")
    os.makedirs(save_dir, exist_ok=True)
    dataset_tags.register_species_tags(object_type, parsed_tags)
    _upsert_species_tags_sidecar(save_dir, object_type, parsed_tags)
    print(f"[process_new_skeleton] Using --species-tags for '{object_type}': {parsed_tags}")
    # ──────────────────────────────────────────────────────────────────────

    process_skeleton(
        object_type,
        None,
        args.save_dir,
        tpose_path,
        crop_enabled=crop_enabled,
        skip_t5=args.skip_t5_embeddings,
        reference_cond_path=getattr(args, 'reference_cond_path', None) or None,
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
    
