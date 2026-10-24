"""
Augmented Training Motion Sampler — BVH Preview Export

Randomly samples N motions from the training dataset and exports them as BVH files
for manual verification. Required --mode enables exactly one augmentation with
probability 100%; other augmentations are disabled:

  loop          — loop closing-key removal, phase roll and tiling, using only
                  loop clips and respecting training's phase-anchored labels
  motion-speed  — time scaling with range --motion-speed-aug (default: 1.3,
                  matching train_all.bat; 1.0 disables random speed changes)
  leaf-drop     — remove eligible leaf joints and export the reduced skeleton
  bone-length   — symmetric body proportions, paired original/augmented BVHs
                  with actual scales in filenames and contact residuals in logs
  rest-repose   — a randomly bent rest pose; the motion features are unchanged
                  by it, so a one-frame rest-pose BVH pair is exported

Every mode also writes <stem>__original.bvh beside each augmented file: the same
clip with augmentations off as a control (rest-repose: the stored rig's rest).
Bone-length and loop replay the same temporal draws; motion-speed keeps the
deterministic fit instead of the sampled tempo.

All modes keep training's automatic speed fitting for phase-anchored actions,
capped by MAX_FIT_SPEEDUP. Clips still above the source-length budget are randomly
cropped, followed by the normal model-window resampling.

Exported filenames encode the applied augmentations, e.g.:
  Horse_Gallop__loop+loop7x+roll12.bvh
  Horse_Gallop__motion-speed+spd1.130.bvh

By default every mode exports at the real 30 fps tempo by stretching the window
back to resample_speed_cond * num_frames frames, as sample/generate.py does.
Pass --no-real-time to export the model window itself at its compressed tempo.
Real-time export makes speed changes visible: the window content of a time-scaled
clip is the same as the original's; its resample_speed differs.

Usage
-----
    # From inside the Anytop/ directory:
    python tools/sample_augmented_bvh.py \\
        --mode loop \\
        --n 10 \\
        --num-frames 60 \\
        --filter 'Horse*' \\
        --output-dir ./augmented_bvh_samples

Arguments
---------
  --mode              Required: loop / motion-speed / leaf-drop / bone-length / rest-repose (one only)
  --bone-length-aug-max  Max relative group-length change (default: 0.2 = +/-20%)
  --bone-length-aug-min  Min relative group-length change (default: 0.1 = +/-10%)
  --rest-repose-max-deg  Largest rest rotation per joint in degrees (default: 20)
  --n                 Number of samples to export  (default: 10)
  --num-frames        Window length in frames, must match --num_frames in training (default: 60)
  --motion-speed-aug  Speed range R >= 1, log-uniform in [1/R, R] (default: 1.3)
  --loop-tile-single-prob  Floor on P(loop tile count == 1) (default: 0.5; 0.0 = uniform)
  --no-real-time      Export the model window instead of the default real 30 fps tempo
  --filter            One or more species-name wildcards, e.g. 'Insect_*' 'Horse*' (case-insensitive)
  --split             train / val / all (default: "all")
  --seed              RNG seed for reproducibility (default: 1234)
  --dataset-dir       Dataset root (auto-detected if omitted)
  --output-dir        Where to write BVH files (default: ./augmented_bvh_samples)
  --purge             Delete existing *.bvh files in the output directory first
"""
from __future__ import annotations

import argparse
import fnmatch
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

from data_loaders.truebones.truebones_utils.motion_process import (
    refresh_joint_metadata_in_cond_dict,
)
from utils.npy_restore import write_feature_bvh
from data_loaders.truebones.truebones_utils.get_opt import get_opt
from data_loaders.truebones.truebones_utils.cond_schema import load_cond
from data_loaders.truebones.truebones_utils.motion_labels import (
    load_motion_metadata,
)
from data_loaders.truebones.data.dataset import (
    MotionDataset,
    load_allowed_motion_names_per_source,
    resample_motion_features,
    ALL_SPLIT_NAME,
    SUPPORTED_SPLITS,
    _build_joint_mask_candidate_roots,
)
from data_loaders.truebones.truebones_utils.canonical_features import (
    canonical_to_physical_hml,
    mark_canonical_cond_entry,
    working_to_stored,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class PreviewMotionDataset(MotionDataset):
    """Isolate preview augmentations without changing training behavior."""

    def _prepare_sample(self, name, data, **kwargs):
        if self.opt.preview_mode != "loop":
            # loop_cond_prob=0 only hides the condition; it still rolls/tiles.
            data = dict(data)
            data["motion_metadata"] = dict(data["motion_metadata"], is_loop=False)
        return super()._prepare_sample(name, data, **kwargs)

    def _sample_motion_speed_target_length(
        self, length, is_loop, max_source_length, *, fit_budget=False, slow_only=False,
    ):
        if self.opt.preview_mode != "motion-speed" or self.opt.motion_speed_aug == 1.0:
            # The training sampler keeps deterministic fitting even when
            # random speed augmentation is disabled (R=1 or probability=0).
            return super()._sample_motion_speed_target_length(
                length, is_loop, max_source_length, fit_budget=fit_budget, slow_only=slow_only,
            )
        for _ in range(32):
            target_length = super()._sample_motion_speed_target_length(
                length, is_loop, max_source_length, fit_budget=fit_budget, slow_only=slow_only,
            )
            if target_length != length:
                return target_length
        raise ValueError("Speed range cannot produce a changed frame count for this clip.")


def _build_cond_dict(opt) -> dict:
    """Load cond.npy and prepare static canonical metadata.
    Joint-name T5 embeddings are stubbed with zeros so we can avoid loading a
    large language model just for BVH export.
    """
    cond_dict_raw: dict = load_cond(opt.cond_file)
    cond_dict_raw = refresh_joint_metadata_in_cond_dict(cond_dict_raw)

    cond_dict = dict(cond_dict_raw)
    if not cond_dict:
        raise RuntimeError(f"No species found in {opt.cond_file}.")

    for object_type, cond in cond_dict.items():
        mark_canonical_cond_entry(cond)
        # Stub T5 embeddings — only used by the model, not needed for BVH export.
        n_joints = np.asarray(cond["parents"]).shape[0]
        if "joints_names_embs" not in cond:
            cond["joints_names_embs"] = np.zeros((n_joints, 768), dtype=np.float32)
        # Required sample metadata even though this tool does not mask joints.
        cond["joint_mask_candidate_roots"] = _build_joint_mask_candidate_roots(cond)

    return cond_dict


def _export_bvh(
    save_path: Path,
    motion_raw: np.ndarray,
    joints_names: list[str],
    object_cond: dict[str, object],
    *,
    fps: float = 30.0,
) -> bool:
    """Denormalized (F, J, 12) → BVH file through the shared NPY decode.  Returns True on success."""
    try:
        write_feature_bvh(
            motion_raw, object_cond, str(save_path), fps=fps, joint_names=list(joints_names),
        )
    except Exception as exc:
        print(f"[export] {save_path.name}: {exc}")
        return False
    return True


def _bvh_joint_names(object_cond: dict[str, object], n_joints: int) -> list[str]:
    """Anatomical BVH joint names from cond, with placeholders as fallback."""
    joints_names = list(
        object_cond.get(
            "canonical_bvh_joint_names",
            object_cond.get("joints_names", []),
        )
    )
    if not joints_names:
        joints_names = [f"joint_{j}" for j in range(n_joints)]
    return joints_names


def _rest_pose_bvh_features(object_cond: dict[str, object]) -> np.ndarray:
    """The cond's rest as a one-frame physical (1, J, 12) feature tensor."""
    return np.asarray(object_cond["rest_pose"], dtype=np.float32)[None]


def _decode_for_export(
    motion_canonical: np.ndarray,
    m_length: int,
    object_cond: dict[str, object],
    aug_info: dict,
    num_frames: int,
    real_time: bool,
) -> np.ndarray:
    """Canonical (F, J, 12) window → stored-layout physical features at the export tempo."""
    motion_raw = canonical_to_physical_hml(
        motion_canonical[:m_length],
        object_cond,
    ).astype(np.float32)
    if real_time:
        export_frames = max(2, int(round(float(aug_info["resample_speed_cond"]) * num_frames)))
        if export_frames != motion_raw.shape[0]:
            # Invert the loader's window resample: periodic for a
            # loop-conditioned window.
            motion_raw = resample_motion_features(
                motion_raw, export_frames, periodic=bool(aug_info.get("loop_applied")),
            )
    return working_to_stored(motion_raw, is_loop=bool(aug_info.get("loop_applied")))


def _replay_original(
    dataset: PreviewMotionDataset,
    name: str,
    aug_info: dict,
    num_frames: int,
    real_time: bool,
) -> tuple[np.ndarray, dict[str, object], list[str]]:
    """Replay the augmented sample's temporal draws on the un-augmented clip.

    ``bone_length_rng_state`` is captured after the leaf-drop and bone-length
    draws, right before the temporal stages, so restoring it keeps the same
    crop and loop-tile draws. Outside loop mode the "baseline" preview mode
    treats the clip as non-loop, so no roll or tile is applied to the control;
    in loop mode the clip keeps its loop flag and replays the same roll and tile.
    """
    opt = dataset.opt
    saved = (opt.preview_mode, opt.leaf_drop_prob, opt.bone_length_aug_prob, opt.rest_repose_prob,
             opt.motion_speed_aug_prob)
    post_state = random.getstate()
    try:
        if saved[0] != "loop":
            opt.preview_mode = "baseline"
        opt.leaf_drop_prob = 0.0
        opt.bone_length_aug_prob = 0.0
        opt.rest_repose_prob = 0.0
        opt.motion_speed_aug_prob = 0.0
        random.setstate(aug_info["bone_length_rng_state"])
        baseline = dataset._prepare_sample(name, dataset.data_dict[name], return_aug_info=True)
    finally:
        (opt.preview_mode, opt.leaf_drop_prob, opt.bone_length_aug_prob, opt.rest_repose_prob,
         opt.motion_speed_aug_prob) = saved
        random.setstate(post_state)
    baseline_aug = baseline[13]
    baseline_cond = baseline_aug["object_cond"]
    motion_raw = _decode_for_export(
        baseline[0], baseline[1], baseline_cond, baseline_aug, num_frames, real_time,
    )
    joints_names = _bvh_joint_names(baseline_cond, np.asarray(baseline[2]).shape[0])
    return motion_raw, baseline_cond, joints_names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

class _SingleMode(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest, None) is not None:
            raise argparse.ArgumentError(self, "--mode may only be specified once")
        setattr(namespace, self.dest, values)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export augmented training motions as BVH files for manual verification."
    )
    p.add_argument("--mode", required=True, action=_SingleMode,
                   choices=("loop", "motion-speed", "leaf-drop", "bone-length", "rest-repose"),
                   help="Preview exactly one augmentation, enabled with probability 100%%.")
    p.add_argument("--n", type=int, default=10, help="Number of samples to export.")
    p.add_argument('--bone-length-aug-max', type=float, default=0.2,
                   help='Max relative body-proportion change; 0.2 = +/-20%% (bone-length mode only).')
    p.add_argument('--bone-length-aug-min', type=float, default=0.1,
                   help='Min relative body-proportion change; 0.1 = +/-10%% (bone-length mode only).')
    p.add_argument('--rest-repose-max-deg', type=float, default=20.0,
                   help='Largest rest rotation per joint in degrees (rest-repose mode only).')
    p.add_argument("--num-frames", type=int, default=60,
                   help="Temporal window length in frames (must match --num_frames in training).")
    p.add_argument("--loop-tile-single-prob", type=float, default=0.5,
                   help="Floor on the probability that a loop window holds one cycle. Match --loop_tile_single_prob.")
    p.add_argument("--motion-speed-aug", type=float, default=1.3,
                   help="Speed range R >= 1, log-uniform in [1/R, R]; default: 1.3, matching "
                        "train_all.bat (1.0 disables random speed changes). Used only in motion-speed mode.")
    p.add_argument("--no-real-time", dest="real_time", action="store_false", default=True,
                   help="Export the model window at its compressed tempo. By default all modes export "
                        "at the real 30 fps tempo (resample_speed_cond * num-frames frames).")
    p.add_argument("--filter", nargs="+", default=[], metavar="PATTERN",
                   help="Wildcard patterns (case-insensitive, e.g. 'Insect_*') matched against "
                        "species names; only matching species' clips are sampled.")
    p.add_argument("--split", default="all",
                   help="Dataset split: train / val / all.")
    p.add_argument("--seed", type=int, default=1234,
                   help="RNG seed for reproducible sampling.")
    p.add_argument("--cond-path", dest="cond_path", default="",
                   help="cond.npy defining the run (single dataset or merged). "
                        "Takes precedence over --dataset-dir.")
    p.add_argument("--dataset-dir", default="",
                   help="Processed dataset root (auto-detected if omitted).")
    p.add_argument("--output-dir", default="outputs/augmented_bvh_samples",
                   help="Directory to write BVH files.")
    p.add_argument("--purge", action="store_true",
                   help="Delete existing *.bvh files in the output directory before exporting.")
    args = p.parse_args()
    if args.mode == "motion-speed" and (
        not np.isfinite(args.motion_speed_aug) or args.motion_speed_aug < 1.0
    ):
        p.error("--motion-speed-aug must be finite and >= 1 in motion-speed mode")
    if not 0.0 <= args.loop_tile_single_prob <= 1.0:
        p.error("--loop-tile-single-prob must be in [0, 1]")
    if args.mode == 'bone-length' and (
        not np.isfinite(args.bone_length_aug_max) or not 0.0 < args.bone_length_aug_max < 1.0
    ):
        p.error('--bone-length-aug-max must be finite and in (0, 1) in bone-length mode')
    if args.mode == 'bone-length' and not 0.0 <= args.bone_length_aug_min <= args.bone_length_aug_max:
        p.error('--bone-length-aug-min must be in [0, --bone-length-aug-max] in bone-length mode')
    if args.mode == 'rest-repose' and not (np.isfinite(args.rest_repose_max_deg)
                                           and 0.0 < args.rest_repose_max_deg <= 90.0):
        p.error('--rest-repose-max-deg must be finite and in (0, 90] in rest-repose mode')
    if args.n <= 0 or args.num_frames <= 0:
        p.error("--n and --num-frames must be positive")
    return args


class _AugmentationNotApplicable(Exception):
    """The requested augmentation could not be applied to a clip.

    This is a property of the clip (e.g. no droppable leaf joints, or a speed
    draw that rounded back to the original length), not a tool error, so it is
    reported as a skip rather than a failure.
    """


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
    # One cond.npy defines the run; --dataset-dir is shorthand for that
    # directory's own cond.npy.
    cond_path = args.cond_path
    if not cond_path and args.dataset_dir:
        cond_path = str(Path(args.dataset_dir).resolve() / "cond.npy")
    opt = get_opt(device, cond_path)

    # Augmentation settings
    opt.preview_mode = args.mode
    opt.loop_cond_prob = 1.0
    opt.motion_speed_aug = args.motion_speed_aug if args.mode == "motion-speed" else 1.0
    opt.motion_speed_aug_prob = 1.0 if args.mode == "motion-speed" else 0.0
    opt.leaf_drop_prob = 1.0 if args.mode == "leaf-drop" else 0.0
    opt.bone_length_aug_prob = 1.0 if args.mode == 'bone-length' else 0.0
    opt.bone_length_aug_max = args.bone_length_aug_max
    opt.bone_length_aug_min = args.bone_length_aug_min
    opt.rest_repose_prob = 1.0 if args.mode == 'rest-repose' else 0.0
    opt.rest_repose_max_deg = args.rest_repose_max_deg
    opt.loop_tile_single_prob = args.loop_tile_single_prob
    opt.motion_cache_size = 0  # no cache needed for sampling

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.purge:
        stale = sorted(output_dir.glob("*.bvh"))
        for path in stale:
            path.unlink()
        print(f"[INFO] Purged {len(stale)} old BVH file(s) from: {output_dir}")

    print(f"[INFO] Loading condition data from: {opt.cond_file}")
    cond_dict = _build_cond_dict(opt)
    print(f"[INFO] Species loaded: {len(cond_dict)}")

    # -----------------------------------------------------------------------
    # Build MotionDataset (applies augmentations via augment() + _prepare_sample())
    # -----------------------------------------------------------------------
    motion_metadata_lookup = {
        source.namespace: load_motion_metadata(source.root) for source in opt.sources
    }
    allowed_motion_names = load_allowed_motion_names_per_source(
        args.split,
        opt.sources,
        "",
        motion_metadata_lookup,
    )
    eligible = sum(len(names) for names in allowed_motion_names.values())
    print(f"[INFO] Eligible motions in split '{args.split}': {eligible} "
          f"across {len(allowed_motion_names)} dataset source(s)")

    dataset = PreviewMotionDataset(
        opt=opt,
        cond_dict=cond_dict,
        num_frames=args.num_frames,
        sample_limit=0,
        allowed_motion_names=allowed_motion_names,
        motion_metadata_lookup=motion_metadata_lookup,
    )
    print(f"[INFO] Dataset size (after min-length filter): {len(dataset)} motions")

    if len(dataset) == 0:
        print("[ERROR] No motions available after filtering. Check subset / split.")
        return 1

    candidate_names = list(dataset.name_list)
    if args.filter:
        patterns = [p.lower() for p in args.filter]

        def _labels(name: str) -> list[str]:
            # Bare name too: the stored key may carry a dataset namespace prefix.
            object_type = str(dataset.data_dict[name]['object_type']).lower()
            return [object_type, object_type.rpartition('/')[2]]

        candidate_names = [
            name for name in candidate_names
            if any(fnmatch.fnmatchcase(label, p) for label in _labels(name) for p in patterns)
        ]
        print(f"[INFO] Species filter {args.filter}: {len(candidate_names)} candidates")
        if not candidate_names:
            print("[ERROR] No motions match --filter. Check the species pattern.")
            return 1
    if args.mode == 'bone-length':
        print(f'[INFO] Bone-length candidates (all actions and skeletons): {len(candidate_names)}')
    if args.mode == "loop":
        candidate_names = [
            name for name in candidate_names
            if bool(dataset.data_dict[name].get("motion_metadata", {}).get("is_loop", False))
        ]
        print(f"[INFO] Loop-only candidates: {len(candidate_names)}")
        if not candidate_names:
            print("[ERROR] No loop motions available after filtering. Check subset / split.")
            return 1

    # -----------------------------------------------------------------------
    # Sample distinct names, capped at the candidate count.
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
    skipped = 0
    multi_source = len(opt.sources) > 1

    for idx, name in enumerate(sampled_names):
        print(f"[{idx+1}/{n}] Processing: {name} ...", end=" ")

        try:
            # _prepare_sample applies augmentations and returns canonical motion.
            (
                motion_canonical,  # (num_frames, J, 12) canonical model space
                m_length,     # prepared-motion frame count; _prepare_sample resamples to num_frames, so the [:m_length] slice is the whole array
                parents,
                _rest_geom,
                offsets,
                _joints_graph_dist,
                _joints_relations,
                object_type,
                _joints_names_embs,
                _max_joints,
                motion_metadata,
                _name,
                _candidate_roots_info,
                aug_info,     # dict: loop_applied, resample_speed_cond, ...
            ) = dataset._prepare_sample(name, dataset.data_dict[name], return_aug_info=True)

            object_cond = aug_info["object_cond"]
            bone_info = aug_info.get('bone_length_aug', {})
            if args.mode == 'bone-length':
                if not bone_info.get('applied'):
                    raise _AugmentationNotApplicable(bone_info.get('skip_reason', 'No eligible body groups.'))
            repose_info = aug_info.get('rest_repose_aug', {})
            if args.mode == 'rest-repose' and not repose_info.get('applied'):
                raise _AugmentationNotApplicable(repose_info.get('skip_reason', 'No joint to bend.'))
            leaf_drop_count = int(aug_info.get("leaf_drop_count", 0))
            if args.mode == "leaf-drop" and leaf_drop_count == 0:
                raise _AugmentationNotApplicable("No eligible leaf joints to drop.")
            if (args.mode == "motion-speed" and args.motion_speed_aug > 1.0
                    and float(aug_info["motion_speed_applied"]) == 1.0):
                raise _AugmentationNotApplicable("Speed draw rounded to the original length.")

            # ----------------------------------------------------------------
            # Decode canonical model-space features back to physical HML-like features.
            # ----------------------------------------------------------------
            skeleton_mode = args.mode == "rest-repose"
            if skeleton_mode:
                motion_raw = _rest_pose_bvh_features(object_cond)
            else:
                motion_raw = _decode_for_export(
                    motion_canonical, m_length, object_cond, aug_info, args.num_frames, args.real_time,
                )
            export_frames = int(motion_raw.shape[0])

            # ----------------------------------------------------------------
            # Retrieve joint names from cond_dict for BVH hierarchy
            # Use canonical_bvh_joint_names (anatomical names) so the exported
            # BVH matches the naming convention used by sample/generate.py and
            # the preprocessing pipeline.
            # ----------------------------------------------------------------
            n_joints = np.asarray(parents).shape[0]
            joints_names = _bvh_joint_names(object_cond, n_joints)

            # ----------------------------------------------------------------
            # Build a descriptive filename that encodes what augmentations fired
            # ----------------------------------------------------------------
            # name is the composite clip id '<namespace>/<file>.npy'. With one
            # source the namespace is redundant and dropped, so filenames match
            # what a single-dataset run has always produced; with several it is
            # flattened in, since the bare filename repeats across datasets.
            clip_label = name if multi_source else name.rpartition("/")[2]
            stem = Path(clip_label.replace("/", "_")).stem
            tags: list[str] = [args.mode]
            source_metadata = dataset.data_dict[name].get("motion_metadata", {})
            source_length = int(dataset.data_dict[name].get("length", motion_canonical.shape[0]))
            is_source_loop = bool(source_metadata.get("is_loop", False))
            loop_tile_count = int(aug_info.get("loop_tile_count", 1))

            # aug_info contains actual augmentation results (not just parameters)
            if aug_info.get("loop_applied"):
                tags.append(f"loop{loop_tile_count}x")
            if args.mode == "loop":
                tags.append(f"roll{int(aug_info.get('loop_phase_offset', 0))}")
            motion_speed_applied = float(aug_info.get("motion_speed_applied", 1.0))
            if args.mode == "motion-speed":
                tags.append(f"spd{motion_speed_applied:.3f}")
            if leaf_drop_count:
                tags.append(f"drop{leaf_drop_count}j")
            if args.mode == 'bone-length':
                tags.extend(f'{group}{factor:.3f}' for group, factor in bone_info['scales'].items())
            if args.mode == 'rest-repose':
                tags.append(f"repose{repose_info['max_rest_angle_deg']:.0f}deg")

            fname = f"{stem}__{'+'.join(tags)}.bvh"
            save_path = output_dir / fname

            # Control: the stored rig's rest, or the same clip with augmentations
            # off on the same temporal draws.
            if skeleton_mode:
                original_cond = dataset.cond_dict[str(object_type)]
                original_raw = _rest_pose_bvh_features(original_cond)
                original_names = _bvh_joint_names(original_cond, original_raw.shape[1])
            else:
                original_raw, original_cond, original_names = _replay_original(
                    dataset, name, aug_info, args.num_frames, args.real_time,
                )
            original_path = output_dir / f'{stem}__original.bvh'
            if not _export_bvh(original_path, original_raw, original_names, original_cond):
                raise RuntimeError('Original control BVH export failed.')

            ok = _export_bvh(
                save_path,
                motion_raw,
                joints_names,
                object_cond,
            )
            if ok:
                loop_note = ""
                if is_source_loop and not skeleton_mode:
                    loop_note = (
                        f", loop_applied={bool(aug_info.get('loop_applied'))}"
                        f", loop_uncond={bool(aug_info.get('loop_uncond'))}"
                        f", tiles={loop_tile_count}"
                        f", source={source_length}f"
                    )
                speed_note = ""
                if args.mode == "motion-speed":
                    speed_note = f", speed={motion_speed_applied:.3f}"
                if leaf_drop_count:
                    speed_note += f", dropped={leaf_drop_count}j, kept={n_joints}j"
                if args.mode == 'bone-length':
                    speed_note += f", scales={bone_info['scales']}, contact_ik=off"
                    if bone_info['notes']:
                        speed_note += f", notes={'; '.join(bone_info['notes'])}"
                if args.mode == 'rest-repose':
                    speed_note += (f", rotated={repose_info['rotated_joints']}j"
                                   f", max_angle={repose_info['max_rest_angle_deg']:.1f}deg")
                print(f"OK  → {save_path.name}  [{export_frames}f, {object_type}{loop_note}{speed_note}]")
                exported += 1
            else:
                print("FAIL (BVH export failed)")
                failed += 1

        except _AugmentationNotApplicable as exc:
            print(f"SKIP  -> {exc}")
            skipped += 1
        except Exception as exc:
            print(f"ERROR: {exc}")
            failed += 1

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"[DONE] Exported {exported} files ({n} samples) to: {output_dir}")
    if skipped:
        print(f"       {skipped} clip(s) skipped — requested augmentation not applicable.")
    if failed:
        print(f"       {failed} file(s) failed — check error messages above.")
    return 1 if failed or (args.mode in ('bone-length', 'rest-repose') and exported == 0) else 0


if __name__ == "__main__":
    sys.exit(main())
