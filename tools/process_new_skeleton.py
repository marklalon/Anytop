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
needed since it was originally tailored for Truebones (joint-name-based foot
classification). Tested on FBX from Mixamo and other sources.

Input Arguments:
tpos-path         - An FBX/GLB/GLTF file whose bind/rest pose defines the NPY encoding base (required).
save-dir          - Output directory (required).
object_type       - Species/type name (e.g. "Dragon"). Inferred from the tpos-path file stem
                    when omitted: the whole stem minus trailing pose tokens
                    ("Pet_Kiki_A_Tpose.glb" -> "Pet_Kiki_A").
species-tags      - Comma-separated species tags (motion descriptor) for --object-type,
                    e.g. 'Quadruped,Lumbering'. REQUIRED: it defines the
                    descriptor baked into cond.npy. There is no fallback to the
                    default dataset's species_tags.jsonl. Only the tags are
                    baked; generation looks their vector up in the checkpoint's
                    species descriptor table (no T5 runs here).
crop-enabled      - Enable skeleton cropping to MAX_JOINTS=100.
                    Off by default (inference has no joint cap).
reference-cond-path - REQUIRED. cond.npy to inherit the per-object_subset
                    standardization statistics from. Those statistics belong to a
                    trained checkpoint, so pass the checkpoint's own cond.npy
                    snapshot. There is no fallback to the processed dataset dir.
                    A joint whose name text it never encodes gets the blank
                    (all-zero) name the model is trained to read as unknown.
export-tpose-bvh  - Also write a single-frame t-pose BVH preview of the processed
                    skeleton (same as tools/sample_tpose_bvh.py).

Output (under save_dir/):
  cond.npy    - Skeleton representation (joint name embeddings, graph conditions,
                canonical feature-space metadata)
                consumed by AnyTop inference via ``--cond-path``.
  bvh_tpose/<object_type>.bvh - Optional (``--export-tpose-bvh``) t-pose preview
                built from the cond.npy rest offsets, in the canonical frame/units.
"""
import re
import sys, os, shutil
from typing import Any
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.truebones.truebones_utils.motion_process import process_skeleton
from utils.parser_util import process_new_skeleton_args

# Trailing stem tokens that name the *pose*, not the species ("Horse_Tpose.fbx"
# -> "Horse"). Matched case-insensitively, with an optional hyphen before "pose"
# ("Wyvern-T-Pose").
_POSE_STEM_SUFFIXES = frozenset({
    "tpose", "apose", "pose", "rest", "restpose", "bind", "bindpose",
    "rig", "skeleton", "ref", "reference", "all",
})
_POSE_STEM_SUFFIX_RE = re.compile(
    r"[_\-.](?:"
    + "|".join(
        re.escape(s[:-4]) + "-?pose" if s.endswith("pose") and len(s) > 4 else re.escape(s)
        for s in sorted(_POSE_STEM_SUFFIXES, key=len, reverse=True)
    )
    + r")$",
    re.IGNORECASE,
)


def species_from_tpose_stem(tpose_path: str) -> str | None:
    """The species name a new skeleton's rest-pose file stands for.

    Training takes the species from the raw directory name, never from a
    filename, so a multi-token species ("MLH_Worker", "Pet_Kiki_A") survives
    whole. A new skeleton has no directory convention and no cond.npy to match
    prefixes against, so the file stem *is* the species name here, minus any
    trailing pose tokens ("Pet_Kiki_A_Tpose.glb" -> "Pet_Kiki_A").
    """
    name = os.path.splitext(os.path.basename(tpose_path))[0]
    while True:
        stripped = _POSE_STEM_SUFFIX_RE.sub("", name)
        if stripped == name or not stripped:
            break
        name = stripped
    name = name.strip("_-. ")
    return name or None

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
    yes: bool = False,
    export_tpose_bvh: bool = False,
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
        "reference_cond_path": reference_cond_path,
        "yes": yes,
        "export_tpose_bvh": export_tpose_bvh,
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
        object_type = species_from_tpose_stem(tpose_path)
        if object_type is None:
            raise FileNotFoundError(
                f"Cannot infer object-type from reference file '{tpose_path}'."
            )
        print(
            f"Auto-detected object_type: {object_type} (the whole file stem minus "
            f"pose tokens; pass --object-type if the stem carries more than the "
            f"species name)"
        )

    # Skeleton cropping: off by default (inference has no joint cap).
    # Use --crop-enabled to enable MAX_JOINTS=100 cropping.
    crop_enabled = args.crop_enabled

    # ── Species tags (required) ────────────────────────────────────────────
    # A new skeleton must carry its own motion descriptor. There is no fallback
    # to the default dataset's species_tags.jsonl -- that would silently borrow a
    # same-named species' tags. Register the tags into the process snapshot (so
    # the cond bakes them) and write the sidecar (the single
    # source of truth the cond bakes its species_tags field from).
    from data_loaders.truebones.truebones_utils import dataset_tags
    raw_tags = str(getattr(args, 'species_tags', '') or '').strip()
    if not raw_tags:
        raise ValueError(
            "--species-tags is required for a new skeleton. It defines the motion "
            "descriptor (body-plan, locomotion) baked into cond.npy. There is "
            "no fallback to the default dataset's tags."
        )
    parsed_tags = dataset_tags.parse_species_tags(raw_tags)
    dataset_tags.check_species_tags(parsed_tags, "--species-tags")
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
        args.reference_cond_path,
        crop_enabled=crop_enabled,
    )

    tpose_bvh = None
    if getattr(args, 'export_tpose_bvh', False):
        from tools.sample_tpose_bvh import sample_tpose_bvh
        (tpose_bvh,) = map(str, sample_tpose_bvh(save_dir, only_objects={object_type}))

    return {
        "save_dir": save_dir,
        "object_type": object_type,
        "tpose_path": tpose_path,
        "cond_npy": os.path.join(save_dir, "cond.npy"),
        "tpose_bvh": tpose_bvh,
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
    
