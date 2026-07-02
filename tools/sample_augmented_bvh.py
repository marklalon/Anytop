"""
Augmented Training Motion Sampler — BVH Preview Export

Randomly samples N motions from the training dataset and exports them as BVH files
for manual verification. Every augmentation that the model sees during training is
applied faithfully in the same order as dataset.py:

  1. random crop       — random start offset when clip > num_frames
  2. loop simulation   — loop motions are repeated/resampled to num_frames,
      with random phase offset and multi-cycle phase metadata

Exported filenames encode the applied augmentations, e.g.:
  Horse___Gallop_123_loop7.bvh

Usage
-----
    # From inside the Anytop/ directory:
    python tools/sample_augmented_bvh.py \\
        --n 10 \\
        --num-frames 60 \\
        --objects-subset quadropeds_test \\
        --output-dir ./augmented_bvh_samples

Arguments
---------
  --n                 Number of samples to export  (default: 10)
  --num-frames        Window length in frames, must match --num_frames in training (default: 60)
  --loop-only         Sample only motions marked as loop clips
  --loop-cond-prob    Probability that a loop clip follows the loop-conditioned path (default: 0.0)
  --objects-subset    Subset name or single species name (default: "all")
  --action-tags       Comma-separated action tags to filter (default: "")
  --split             train / test / all (default: "train")
  --seed              RNG seed for reproducibility (default: 0)
  --dataset-dir       Dataset root (auto-detected if omitted)
  --output-dir        Where to write BVH files (default: ./augmented_bvh_samples)
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from os.path import join as pjoin
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Repo root setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_lib import BVH
from data_loaders.truebones.truebones_utils.motion_process import (
    recover_bvh_export_animation_from_motion_np,
    refresh_joint_metadata_in_cond_dict,
)
from data_loaders.truebones.truebones_utils.get_opt import get_opt
from data_loaders.truebones.truebones_utils.param_utils import OBJECT_SUBSETS_DICT
from data_loaders.truebones.truebones_utils.motion_labels import (
    load_motion_metadata,
)
from data_loaders.truebones.data.dataset import (
    MotionDataset,
    load_motion_names_for_split_with_action_tags,
    ALL_SPLIT_NAME,
    SUPPORTED_SPLITS,
    _build_joint_mask_candidate_roots,
)
from data_loaders.truebones.truebones_utils.canonical_features import (
    canonical_to_physical_hml,
    mark_canonical_cond_entry,
)
from model.joint_mask_utils import sample_subtree_joint_mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_cond_dict(opt, objects_subset: str) -> dict:
    """Load cond.npy and prepare static canonical metadata.
    Joint-name T5 embeddings are stubbed with zeros so we can avoid loading a
    large language model just for BVH export.
    """
    cond_dict_raw: dict = np.load(opt.cond_file, allow_pickle=True).item()
    cond_dict_raw = refresh_joint_metadata_in_cond_dict(cond_dict_raw)

    if objects_subset in OBJECT_SUBSETS_DICT:
        species_list = OBJECT_SUBSETS_DICT[objects_subset]
    else:
        species_list = [objects_subset]  # single species name

    cond_dict = {k: cond_dict_raw[k] for k in species_list if k in cond_dict_raw}
    if not cond_dict:
        raise RuntimeError(
            f"No species found for subset '{objects_subset}'. "
            f"Available species: {sorted(cond_dict_raw.keys())}"
        )

    for object_type, cond in cond_dict.items():
        mark_canonical_cond_entry(cond)
        # Stub T5 embeddings — only used by the model, not needed for BVH export.
        n_joints = np.asarray(cond["parents"]).shape[0]
        if "joints_names_embs" not in cond:
            cond["joints_names_embs"] = np.zeros((n_joints, 768), dtype=np.float32)

        # Precompute joint_mask_candidate_roots for subtree masking export
        cond["joint_mask_candidate_roots"] = _build_joint_mask_candidate_roots(cond)

    return cond_dict


def _export_bvh(
    save_path: Path,
    motion_raw: np.ndarray,
    parents: list[int],
    offsets: np.ndarray,
    joints_names: list[str],
    motion_metadata: dict[str, object],
    *,
    object_cond: dict[str, object] | None = None,
) -> bool:
    """Denormalized (F, J, 13) → BVH file.  Returns True on success."""
    export_parents = list(parents)
    export_offsets = np.asarray(offsets)
    export_joint_names = list(joints_names)

    anim, joints_names, has_animated_pos = recover_bvh_export_animation_from_motion_np(
        motion_raw,
        export_parents,
        export_offsets,
        export_joint_names,
        motion_metadata=motion_metadata,
    )
    if anim is None:
        return False
    BVH.save(str(save_path), anim, joints_names, positions=has_animated_pos)
    return True


def _format_float_tag(value: object) -> str:
    number = float(value)
    if np.isclose(number, round(number)):
        return str(int(round(number)))
    return f"{number:.2f}".rstrip("0").rstrip(".").replace(".", "p")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export augmented training motions as BVH files for manual verification."
    )
    p.add_argument("--n", type=int, default=10, help="Number of samples to export.")
    p.add_argument("--num-frames", type=int, default=60,
                   help="Temporal window length in frames (must match --num_frames in training).")
    p.add_argument("--loop-only", action="store_true",
                   help="Only sample/export motions whose metadata marks them as loop clips.")
    p.add_argument("--loop-cond-prob", type=float, default=1.0,
                   help="Probability that a loop clip follows the loop-conditioned path. Match --loop_cond_prob.")
    p.add_argument("--objects-subset", default="all",
                   help="Predefined subset name or single species (e.g. 'quadropeds_test', 'Horse').")
    p.add_argument("--action-tags", default="",
                   help="Comma-separated action tags to filter (e.g. 'locomotion,jump').")
    p.add_argument("--split", default="train",
                   help="Dataset split: train / test / all.")
    p.add_argument("--seed", type=int, default=1234,
                   help="RNG seed for reproducible sampling.")
    p.add_argument("--dataset-dir", default="",
                   help="Processed dataset root (auto-detected if omitted).")
    p.add_argument("--output-dir", default="outputs/augmented_bvh_samples",
                   help="Directory to write BVH files.")
    p.add_argument("--joint-mask-prob", type=float, default=1.0,
                   help="Probability of exporting a masked counterpart for each sampled clip (0 = disabled). "
                        "Matches --joint_mask_prob in training. (default: 1.0)")
    p.add_argument("--joint-mask-budget", type=float, default=0.15,
                   help="Maximum fraction of non-root joints in each masked export. "
                        "Matches --joint_mask_budget in training. (default: 0.15)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # -----------------------------------------------------------------------
    # Validate split
    # -----------------------------------------------------------------------
    if args.split not in SUPPORTED_SPLITS and args.split != ALL_SPLIT_NAME:
        print(f"[ERROR] Unknown split '{args.split}'. Choose from: {SUPPORTED_SPLITS + (ALL_SPLIT_NAME,)}")
        return 1

    # -----------------------------------------------------------------------
    # Setup — seed ALL random sources for reproducibility
    # -----------------------------------------------------------------------
    random.seed(args.seed)      # global random module (used by dataset augmentations)
    np.random.seed(args.seed)   # numpy RNG
    rng_py = random.Random(args.seed)  # independent RNG for name sampling

    device = None
    opt = get_opt(device)
    if args.dataset_dir:
        opt.data_root = str(Path(args.dataset_dir).resolve())
        opt.cond_file = pjoin(opt.data_root, "cond.npy")
        opt.motion_dir = pjoin(opt.data_root, "motions")
    opt.motion_dir = pjoin(".", opt.motion_dir)
    opt.data_root = pjoin(".", opt.data_root)

    # Augmentation settings
    opt.loop_cond_prob = args.loop_cond_prob
    opt.motion_cache_size = 0  # no cache needed for sampling

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading condition data from: {opt.cond_file}")
    cond_dict = _build_cond_dict(opt, args.objects_subset)
    print(f"[INFO] Species loaded: {sorted(cond_dict.keys())}")

    # -----------------------------------------------------------------------
    # Build MotionDataset (applies augmentations via augment() + _prepare_sample())
    # -----------------------------------------------------------------------
    motion_metadata_lookup = load_motion_metadata(opt.data_root)
    allowed_motion_names = load_motion_names_for_split_with_action_tags(
        args.split,
        opt.data_root,
        opt.motion_dir,
        args.action_tags,
        motion_metadata_lookup,
    )
    print(f"[INFO] Eligible motions in split '{args.split}': {len(allowed_motion_names)}")

    dataset = MotionDataset(
        opt=opt,
        cond_dict=cond_dict,
        temporal_window=31,       # standard window — only affects temporal_mask, not augmentation
        balanced=False,
        num_frames=args.num_frames,
        sample_limit=0,
        allowed_motion_names=allowed_motion_names,
        motion_metadata_lookup=motion_metadata_lookup,
    )
    print(f"[INFO] Dataset size (after min-length filter): {len(dataset)} motions")

    if len(dataset) == 0:
        print("[ERROR] No motions available after filtering. Check subset / split / action-tags.")
        return 1

    candidate_names = list(dataset.name_list)
    if args.loop_only:
        candidate_names = [
            name for name in candidate_names
            if bool(dataset.data_dict[name].get("motion_metadata", {}).get("is_loop", False))
        ]
        print(f"[INFO] Loop-only candidates: {len(candidate_names)}")
        if not candidate_names:
            print("[ERROR] No loop motions available after filtering. Check subset / split / action-tags.")
            return 1

    # -----------------------------------------------------------------------
    # Sample N names (with replacement if N > dataset size)
    # -----------------------------------------------------------------------
    n = min(args.n, len(candidate_names))
    if args.n > len(candidate_names):
        print(
            f"[WARN] Requested {args.n} samples but only {len(candidate_names)} motions available. "
            f"Sampling all of them."
        )
    sampled_names = rng_py.sample(candidate_names, n)

    # -----------------------------------------------------------------------
    # Export loop
    # -----------------------------------------------------------------------
    exported = 0
    failed = 0

    for idx, name in enumerate(sampled_names):
        print(f"[{idx+1}/{n}] Processing: {name} ...", end=" ")

        try:
            # _prepare_sample applies augmentations and returns canonical motion.
            (
                motion_canonical,  # (num_frames, J, 13) canonical model space
                m_length,     # actual frames (before padding)
                parents,
                rest_pose,
                offsets,
                _temporal_mask,
                _joints_graph_dist,
                _joints_relations,
                object_type,
                _joints_names_embs,
                _crop_start,
                _max_joints,
                motion_metadata,
                _name,
                _candidate_roots_info,
                aug_info,     # dict: crop_start, loop_applied, playspeed_cond, loop_uncond
            ) = dataset._prepare_sample(name, dataset.data_dict[name], return_aug_info=True)

            # ----------------------------------------------------------------
            # Decode canonical model-space features back to physical HML-like features.
            # ----------------------------------------------------------------
            motion_raw = canonical_to_physical_hml(
                motion_canonical[:m_length],
                cond_dict[object_type],
            ).astype(np.float32)

            # ----------------------------------------------------------------
            # Retrieve joint names from cond_dict for BVH hierarchy
            # Use canonical_bvh_joint_names (anatomical names) so the exported
            # BVH matches the naming convention used by sample/generate.py and
            # the preprocessing pipeline.
            # ----------------------------------------------------------------
            joints_names = list(
                cond_dict[object_type].get(
                    "canonical_bvh_joint_names",
                    cond_dict[object_type].get("joints_names", []),
                )
            )
            n_joints = np.asarray(parents).shape[0]
            if not joints_names:
                joints_names = [f"joint_{j}" for j in range(n_joints)]

            # ----------------------------------------------------------------
            # Build a descriptive filename that encodes what augmentations fired
            # ----------------------------------------------------------------
            stem = Path(name).stem
            tags: list[str] = []
            source_metadata = dataset.data_dict[name].get("motion_metadata", {})
            source_length = int(dataset.data_dict[name].get("length", motion_canonical.shape[0]))
            is_source_loop = bool(source_metadata.get("is_loop", False))
            loop_phase_length = float(motion_metadata.get("loop_phase_length", m_length))

            # aug_info contains actual augmentation results (not just parameters)
            if aug_info.get("loop_applied"):
                tags.append(f"phase{_format_float_tag(loop_phase_length)}")
            elif is_source_loop:
                tags.append("loopuncond")
            actual_crop = aug_info.get("crop_start", 0)
            if actual_crop > 0:
                tags.append(f"crop{actual_crop}")

            fname = f"{stem}__{'+'.join(tags)}.bvh"
            save_path = output_dir / fname

            ok = _export_bvh(
                save_path,
                motion_raw,
                list(parents),
                np.asarray(offsets, dtype=np.float32),
                joints_names,
                motion_metadata,
                object_cond=cond_dict[object_type],
            )
            if ok:
                loop_note = ""
                if is_source_loop:
                    loop_note = (
                        f", loop_applied={bool(aug_info.get('loop_applied'))}"
                        f", phase_len={loop_phase_length:.3g}"
                        f", source={source_length}f"
                    )
                print(f"OK  → {save_path.name}  [{m_length}f, {object_type}{loop_note}]")
                exported += 1
            else:
                print(f"FAIL (recover_animation returned None)")
                failed += 1

            # ----------------------------------------------------------------
            # Subtree joint mask export (if enabled)
            # ----------------------------------------------------------------
            if args.joint_mask_prob > 0.0 and args.joint_mask_budget > 0.0:
                mask_rng = np.random.default_rng(args.seed + idx)
                if args.joint_mask_prob >= 1.0 or float(mask_rng.random()) < args.joint_mask_prob:
                    mask = sample_subtree_joint_mask(
                        parents=list(parents),
                        candidate_root_mask=cond_dict[object_type]["joint_mask_candidate_roots"][: len(parents)],
                        joint_mask_budget=args.joint_mask_budget,
                        rng=mask_rng,
                    )
                else:
                    mask = None
                if mask is not None:
                    # Apply mask in canonical model space, then decode for BVH export.
                    motion_masked_canonical = motion_canonical.copy()
                    motion_masked_canonical[:, mask, :] = 0.0
                    motion_masked_raw = canonical_to_physical_hml(
                        motion_masked_canonical[:m_length],
                        cond_dict[object_type],
                    ).astype(np.float32)

                    masked_tags = list(tags) + [f"mask{int(round(args.joint_mask_budget * 100))}"]
                    masked_fname = f"{stem}__{'+'.join(masked_tags)}_masked.bvh"
                    masked_path = output_dir / masked_fname

                    ok2 = _export_bvh(
                        masked_path,
                        motion_masked_raw,
                        list(parents),
                        np.asarray(offsets, dtype=np.float32),
                        joints_names,
                        motion_metadata,
                        object_cond=cond_dict[object_type],
                    )
                    if ok2:
                        print(f"     └─ masked  → {masked_path.name}")
                        exported += 1
                    else:
                        print(f"     └─ masked export FAILED")
                        failed += 1
                else:
                    pass

        except Exception as exc:
            print(f"ERROR: {exc}")
            failed += 1

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"[DONE] Exported {exported} files ({n} samples) to: {output_dir}")
    if failed:
        print(f"       {failed} file(s) failed — check error messages above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
