from argparse import ArgumentParser
import argparse
import os
import json
import copy

# The three action groups, duplicated from
# data_loaders.truebones.truebones_utils.motion_labels.ACTION_GROUPS so this
# module stays import-light (that one reaches numpy through param_utils).
# tests/test_action_group_checkpoint_binding.py pins the two together.
ACTION_GROUPS = ('locomotion', 'stationary', 'transition')


def parse_loop_cycles(raw):
    """argparse type of ``--loop_cycles``: a positive int."""
    try:
        cycles = int(str(raw).strip())
    except ValueError:
        cycles = 0
    if cycles < 1:
        raise argparse.ArgumentTypeError(f"--loop_cycles must be a positive integer, got {raw!r}")
    return cycles
# The training value for one model over the whole corpus
# (docs/action_group_and_label.md). Recorded in args.json in a group's
# place; generation then reads the group off the label instead of the weights.
ACTION_GROUP_ALL = 'all'

# Checkpoint compatibility stamp, written into every save_dir's args.json by
# train_motion_diffusion and required back by extract_args. Bump it for ANY change that
# alters training or inference semantics, including ones that leave the
# state_dict layout untouched -- those are exactly the changes that would
# otherwise load cleanly and generate wrong motion, reading as a quality
# regression rather than an incompatibility.
CKPT_VERSION = 60

# Data-side contracts stamped alongside the checkpoint version. Unlike a flag,
# these version the *content* of an input the args.json cannot otherwise
# describe: a cond regenerated under a different joint-name schema, or a
# structural descriptor whose channels were reordered, hands the same shapes to
# the same weights with a different meaning behind them.
def joint_condition_schema_versions():
    # Imported lazily: this module is deliberately import-light (the training and
    # generation entry points both parse args long before numpy is wanted).
    from data_loaders.truebones.truebones_utils.joint_struct_features import (
        JOINT_STRUCT_FEATURE_SCHEMA_VERSION,
    )
    from data_loaders.truebones.truebones_utils.joint_embedding_text import (
        JOINT_NAME_EMBEDDING_SCHEMA_VERSION,
    )
    return {
        'joint_struct_schema_version': int(JOINT_STRUCT_FEATURE_SCHEMA_VERSION),
        'joint_name_embedding_schema_version': int(JOINT_NAME_EMBEDDING_SCHEMA_VERSION),
    }


def assert_joint_condition_schema(model_args, args_path, action='sample'):
    """Refuse a checkpoint whose joint-condition contracts differ from this code.

    Separate from the ``version`` stamp because these two can move without any
    training-semantics change of their own -- and because the failure they guard
    is the quiet kind: the tensors line up, only their meaning has shifted.
    """
    expected = joint_condition_schema_versions()
    for key, wanted in expected.items():
        recorded = model_args.get(key)
        if recorded is None:
            raise SystemExit(
                f"ERROR: {args_path} records no {key}, so it predates the joint-condition "
                f"restructure and cannot be used to {action}. Retrain."
            )
        if int(recorded) != wanted:
            raise SystemExit(
                f"ERROR: {args_path} records {key}={recorded}, this code is at {wanted}. "
                f"The joint conditions would load with the shapes intact and a different "
                f"meaning behind them; retrain (and regenerate cond.npy if the joint-name "
                f"schema is what moved)."
            )

def parse_and_load_from_model(parser, argv=None, preserve_cli_args=None):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_model_options(parser)
    args = parser.parse_args(argv)
    args_to_overwrite = []
    for group_name in ['dataset', 'model', 'diffusion']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    # Remove args that should NOT be overwritten by stored training defaults.
    if preserve_cli_args:
        args_to_overwrite = [a for a in args_to_overwrite if a not in preserve_cli_args]

    if isinstance(args.model_path, list) and len(args.model_path) == 1:
        args.model_path = args.model_path[0]

    assert not isinstance(args.model_path, list), "model_path should not be a list at this point"
    args = extract_args(copy.deepcopy(args), args_to_overwrite, args.model_path)

    return args

def assert_checkpoint_version(model_args, args_path):
    """Refuse a checkpoint written by code with different training semantics.

    ``parse_and_load_from_model`` only restores keys the *current* parser still
    defines, so a removed flag in a stored args.json is silently dropped: the
    weights load and then run under semantics they were never trained with. That
    failure looks like a quality regression, not an incompatibility, which is why
    it has to be caught here rather than left to the state_dict loader -- these
    changes need not touch the state_dict at all.

    A checkpoint predating versioning records no ``version`` and is rejected too;
    it was trained with the windowed temporal attention this code no longer has.
    """
    recorded = model_args.get('version')
    if recorded == CKPT_VERSION:
        return
    if recorded is None:
        detail = (
            "it records no 'version' field, so it predates checkpoint versioning: "
            "it was trained with windowed temporal attention (--temporal_window), "
            "which has been replaced by full temporal attention"
        )
    else:
        detail = (
            f"it records version {recorded!r} but this code is version {CKPT_VERSION}"
        )
    raise SystemExit(
        f"ERROR: {args_path} is not compatible with this code -- {detail}. Its "
        "weights may well still load, which is the problem: they would run under "
        "training semantics they were never fitted for and quietly generate wrong "
        "motion. Such a checkpoint can only be read as a historical result, not "
        "re-run; retrain to use it."
    )


def extract_args(args, args_to_overwrite, model_path):
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    assert_checkpoint_version(model_args, args_path)
    assert_joint_condition_schema(model_args, args_path)

    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])

    for a, default in (('min_length', 20),):
        if a in model_args:
            setattr(args, a, model_args[a])
        elif not hasattr(args, a):
            setattr(args, a, default)
    # num_frames is intentionally NOT overridden here — it is a per-generation
    # parameter (output frame count) and must preserve the generate parser's
    # default of None so the caller can distinguish "user explicitly requested
    # N frames" from "use reference native length / checkpoint default".
    return args

def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError('group_name was not found.')

def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument('model_path')
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        raise ValueError('model_path argument must be specified.')

def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=16, type=int, help="Batch size during training.")
    group = parser.add_argument_group('diffusion')
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                       help="Noise schedule type")
    group.add_argument("--noise_schedule_shift", default=1.0, type=float,
                       help="Shift every noise level of --noise_schedule toward noise: the SNR of "
                            "each step is divided by shift^2. The step count and the uniform "
                            "training draw over steps stay as they are, so shift > 1 spends more "
                            "steps (and training samples) at high noise, where global structure "
                            "such as a clip's tempo is decided, and fewer on low-noise detail. "
                            "1 = the plain schedule. A training-time choice: generation reads it "
                            "from the checkpoint, and the same --skip_timesteps then starts from "
                            "more noise.")
    group.add_argument("--noise_schedule_shift_min", default=None, type=float,
                       help="Shift of the lowest-noise step; over the steps the plain schedule "
                            "has at SNR > 1 the shift grows (linear in log sigma) up to "
                            "--noise_schedule_shift, which every noisier step keeps. "
                            "Must be <= --noise_schedule_shift. Below 1 it lowers the smallest "
                            "noise level, so small motion detail has training steps where it "
                            "stands above the noise. Unset = --noise_schedule_shift on every step. "
                            "A training-time choice, read from the checkpoint at generation.")
    group.add_argument("--diffusion_steps", default=100, type=int,
                       help="Number of diffusion steps (denoted T in the paper)")
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")

def add_model_options(parser):
    group = parser.add_argument_group('model')
    group.add_argument("--layers", default=4, type=int,
                       help="Number of layers.")
    group.add_argument("--latent_dim", default=128, type=int,
                       help="Transformer/GRU width.")
    group.add_argument("--ff_size", default=1024, type=int,
                       help="Feed-forward hidden dimension in each decoder layer. "
                            "Controls the bottleneck size of the two-layer FFN "
                            "inside each GraphMotionDecoderLayer.")
    group.add_argument("--last_layer_ff", default=0, type=int,
                       help="Feed-forward hidden width of the LAST decoder layer only "
                            "(0 = same as --ff_size). That layer feeds a single linear "
                            "readout, and in a full run ~3/4 of its FFN units went dead; "
                            "512 recovers those parameters for the trunk.")
    group.add_argument("--lambda_geo", default=0.0, type=float, help="Geodesic rotation loss weight (SO(3) distance between predicted and target rotations).")
    group.add_argument("--lambda_drot", default=0.0, type=float,
                       help="Weight of target-relative global rotation-step loss: R[t]^T R[t+1], "
                            "squared chordal error in 0.1-radian units. Includes valid loop wrap steps "
                            "and is weighted by the diffusion signal fraction alpha_bar[t]. 0 disables it.")
    group.add_argument("--drot_joint_boost", default=0.0, type=float,
                       help="Rotation-step GT motion joint weighting: 1 + boost * RMS/max_RMS, "
                            "renormalized per sample. 0 keeps uniform supervision; try 3.")
    group.add_argument("--lambda_dpos", default=0.04, type=float,
                       help="Weight of the frame-to-frame position-step MSE against the target, "
                            "in units of the corpus position-step RMS; a loop sample also counts its wrap "
                            "step pos[0]-pos[-1]. The root's step covers its XYZ position channels, so this also "
                            "supervises the root trajectory and a loop's closure. Scaled by the species' "
                            "dpos_loss_weight.")
    group.add_argument("--lambda_acc", default=0.0, type=float,
                       help="Weight of target-relative position acceleration MSE for all joints, "
                            "in corpus position-step RMS units with species weights; includes loop wrap. "
                            "0 disables it.")
    group.add_argument("--lambda_root_pos", default=0.0, type=float,
                       help="Weight of root XYZ position MSE in canonical standardized units, "
                            "without joint-count dilution or species weights. 0 disables it.")
    group.add_argument("--lambda_root_dpos", default=0.0, type=float,
                       help="Weight of root XYZ position-step MSE in corpus position-step RMS units, "
                            "with wrap steps and species weights, matching the AE. 0 disables it.")
    group.add_argument("--lambda_root_acc", default=0.0, type=float,
                       help="Weight of root XYZ acceleration MSE in --lambda_root_dpos's units "
                            "and species weights; includes circular second differences on loops, "
                            "matching the AE. 0 disables it.")
    group.add_argument("--lambda_loop_wrap", default=0.0, type=float,
                       help="Weight for the loop-only rotation seam continuity loss on denormalized "
                            "outputs: the wrap step's rotation angle matches its neighbouring steps. A loop "
                            "window's last frame is one step before frame 0, never a copy of it; the "
                            "position seam is --lambda_dpos's wrap step.")
    group.add_argument("--lambda_bone", default=0.0, type=float,
                       help="Weight for the target-relative, rest-length-normalized bone-vector loss (0.0=off). "
                            "Penalizes |v_pred - v_gt| for every parent->child bone vector of the position "
                            "channel, i.e. both the bone's length and its direction, divided by the rest bone "
                            "length (floored at a fraction of the skeleton length scale) so short/distal bones, "
                            "which l_simple leaves nearly unconstrained, get proportionally larger gradient. "
                            "Anchoring on GT (not rest) preserves genuinely animated bone-length deformation. "
                            "Every leaf joint adds a virtual unit child bone in its local frame (isotropic: "
                            "|R_pred - R_gt|_F / sqrt(3)), so a leaf's rotation is held like a joint with a child. "
                            "Weighted by alphas_cumprod[t] so it fades out at high noise. Computed on "
                            "denormalized outputs; recommended range ~0.1-0.3.")
    group.add_argument("--lambda_limb_corr", default=0.0, type=float,
                       help="Weight of the inter-limb coordination loss: the lagged, normalized correlations "
                            "between limb-tip steps (one tip per hand, foot and wing chain) against the target's, weighted by "
                            "the target's limb motion so a still limb drops out. Invariant to time shift and "
                            "amplitude, so it supervises relative timing between limbs only. 0 disables it.")
    group.add_argument("--motion_speed_aug", default=1.0, type=float,
                       help="Motion-speed augmentation range R (1.0 = off). Each training clip is first "
                            "time-scaled by a log-uniform ratio in [1/R, R] -- played faster (fewer frames) "
                            "or slower (more frames) at the same fps -- and then treated as an ordinary "
                            "source clip: loop roll/tile, crop, the window resample and resample_speed_cond "
                            "all see the scaled length, and the model is told nothing. Loader-only: no cond "
                            "regen, no model change, no CKPT_VERSION bump. Spreads the clustered clip lengths "
                            "(the corpus piles up on a few exact frame counts) so an inference num_frames "
                            "between the clusters is in distribution. The range is narrowed per clip so the "
                            "scaled clip stays >= min_length, a clip that fits the source budget still fits "
                            "(never cropped by slowing down), and a clip over the budget is only sped up.")
    group.add_argument("--motion_speed_aug_prob", default=1.0, type=float,
                       help="Per-sample probability of applying --motion_speed_aug (default 1.0 = every clip). "
                            "The recorded tempo is one point of the continuum, so leaving a mass at exactly "
                            "1.0 only keeps part of the length spike; lower this only to compare against it.")
    group.add_argument("--loop_tile_single_prob", default=0.5, type=float,
                       help="Floor on the probability that a loop training window holds ONE cycle "
                            "(loop tile count 1); the rest of the mass stays uniform over 2..max tiles. "
                            "0.0 is the plain uniform draw over 1..max, which for a short loop makes the "
                            "single-cycle window -- the regime --loop with the auto length generates in -- "
                            "a small minority of draws, while every other draw is k bit-identical copies "
                            "of the cycle. 1.0 turns loop tiling off. Loader-only, like --motion_speed_aug: no regen, no bump.")
    group.add_argument("--topology_aug_prob", default=0.0, type=float,
                       help="Per-sample probability of the topology augmentation (0.0 = off). A firing "
                            "draw picks one of three edits uniformly, trying the others when the "
                            "rig offers it nothing (truebones_utils/topology_aug.py). Leaf drop: every "
                            "eligible terminator leaf (named Nub / end / site / Container, or locally "
                            "static and nameless) plus trailing segments of repeated-name chains, mirror "
                            "twins together. Limb drop: one whole tail (only with an arm or leg part), "
                            "soft/helper chain, leg chain "
                            "(at least six legs), or arm/wing chain (wings, arms and legs coexist), "
                            "with its mirror if present; at least 10 joints remain. "
                            "Chain merge: 1..3 single-child "
                            "joints removed, their child taking the joined bone, only where the merged "
                            "bone's FK residual stays small on every frame. Leaf drop is exact "
                            "on the kept features; joint_struct, the relation matrices and L are rebuilt "
                            "for the edited rig. Root, translation root, facing and face joints are never "
                            "removed. Training split only (the dataset refuses it for val). "
                            "Loader-only: no regen, no model change, no CKPT_VERSION bump.")
    group.add_argument("--bone_length_aug_prob", default=0.0, type=float,
                       help="Probability of symmetric body-proportion augmentation; training only (0 = off).")
    group.add_argument("--bone_length_aug_max", default=0.2, type=float,
                       help="Max relative bone-length change; with --bone_length_aug_min 0.1 a group "
                            "scale is drawn from [0.8, 0.9] or [1.1, 1.2]. Only groups whose mean bone "
                            "length reaches the skeleton's mean bone length are scaled. "
                            "All actions/rigs; anatomical or fallback topology groups. "
                            "Contact IK is disabled; source local rotations are preserved, with no quality rejection. "
                            "Loader-only: no dataset regeneration or checkpoint schema change.")
    group.add_argument("--bone_length_aug_min", default=0.1, type=float,
                       help="Min relative bone-length change, so every draw is visible; "
                            "must be in [0, --bone_length_aug_max].")
    group.add_argument("--rest_repose_prob", default=0.0, type=float,
                       help="Probability of re-expressing a sample against a randomly bent rest pose "
                            "(target and reference input draw independently); training only (0 = off). "
                            "Exact: the world motion is unchanged, the local rotations are re-expressed, "
                            "and rest_geom / offsets / L follow the new rest, so a rig's rest geometry "
                            "stops being a fixed fingerprint. The structural channel stays the stored rig's. "
                            "Loader-only: no dataset regeneration or checkpoint schema change.")
    group.add_argument("--rest_repose_max_deg", default=20.0, type=float,
                       help="Largest rest rotation per joint in degrees, scaled down by the share of the "
                            "rig the joint moves (a hub carrying most of the body barely turns). Paired "
                            "joints turn about a twist reference axis and mirror to the partner; "
                            "midline joints turn about the lateral axis. A joint whose turn would change "
                            "the canonical rotation channel stays unbent.")
    group.add_argument("--rest_variant_prob", default=0.0, type=float,
                       help="Probability of re-expressing a sample on one of its species' hand-made rest "
                            "variants (TPose_<tag>.glb beside the primary TPose, drawn uniformly; target "
                            "and reference input draw independently); training only (0 = off). Exact like "
                            "--rest_repose_prob, but with real, large rest differences (T- vs A-pose). "
                            "Species without variants are untouched. The structural channel stays the "
                            "primary rest's. Loader-only: no checkpoint schema change.")
    group.add_argument("--loop_cond_prob", default=1.0, type=float,
                       help="Probability that a loop training clip is TOLD it is a loop (is_loop=1: "
                            "circular time table, periodic window resample, wrap losses). The rest of "
                            "the loop clips keep every loop augmentation (closing-key drop, circular "
                            "roll, tiling) but are resampled as open clips and handed over with "
                            "is_loop=0. Deliberate label noise: the flag is confounded with content in "
                            "the corpus (most loops are stationary idles; within a species-tag cluster "
                            "the loop-authored gaits differ from the one-shot ones), and a flag the "
                            "model cannot trust as a content key is one it has to read as time "
                            "topology only. 1.0 = off. Loader-only, like --motion_speed_aug: no regen, "
                            "no bump. The eval loader always uses 1.0.")
    group.add_argument("--loop_cond_prob_free", default=None, type=float,
                       help="Override --loop_cond_prob for phase-free loop samples. Defaults to --loop_cond_prob.")
    group.add_argument("--loop_cond_prob_anchored", default=None, type=float,
                       help="Override --loop_cond_prob for phase-anchored loop samples. Defaults to --loop_cond_prob.")
    group.add_argument("--t5_out_dim", default=0, type=int, help=argparse.SUPPRESS)
    group.add_argument("--value_emb", action='store_true',
                       help="If passed, graph multihead attention learns GRPE value embeddings")
    group.add_argument("--cross_limb", default=1, type=int, choices=[0, 1],
                       help="Enable cross-limb temporal blocks (1 = on, 0 = off).")
    group.add_argument("--cross_limb_latents", default=8, type=int,
                       help="Number of latent tokens K in the cross-limb temporal block.")
    group.add_argument("--cross_limb_dim", default=64, type=int,
                       help="Bottleneck width of each cross-limb temporal block. "
                            "Clamped to <= latent_dim and a multiple of num_heads.")
    group.add_argument("--cross_limb_last_n", default=0, type=int,
                       help="Apply the cross-limb block only on the last N decoder "
                            "layers (0 = all layers). Each active layer gets its "
                            "own independent block, so this also controls the "
                            "cross-limb parameter count.")
    group.add_argument("--dropout_prob", default=0.1, type=float,
                       help="Dropout probability for AnyTop model layers. Set to 0 to disable dropout.")
    group.add_argument("--joint_name_drop_prob", default=0.0, type=float,
                       help="Per-joint probability of zeroing a joint's ENTIRE name embedding during "
                            "training, encouraging fallback to rest geometry/graph_dist/joints_relations "
                            "when a name is unknown. Default 0.0 (off).")
    group.add_argument("--rest_geom_joint_drop_prob", default=0.1, type=float,
                       help="Per-joint probability in training of hiding one joint's rest geometry, "
                            "drawn independently of the name drop. "
                            "Inference never drops.")
    group.add_argument("--species_cond", action='store_true',
                       help="FiLM-modulate joint-name embeddings using [joint_name_emb || species_code], "
                            "species_code = [T5(body plan) || variant one-hot]. Zero-initialized to "
                            "identity. Applied on every row; it has no CFG drop.")
    group.add_argument("--action_label_cond", action='store_true',
                       help="Enable action-label conditioning: the clip's action_label ('run, forward, "
                            "left, glide' -- controlled keywords, not prose) is split into words, pooled "
                            "into one frozen-T5 channel per role slot (head / direction / "
                            "modifier), "
                            "projected and added to the timestep token. Requires the word table "
                            "dataset/action_word_embeddings.npy "
                            "(tools/build_action_label_embeddings.py).")
    group.add_argument("--action_word_embeddings", default="", type=str,
                       help="Override the frozen action-word table path (default: "
                            "dataset/action_word_embeddings.npy). Its embedding_fingerprint is written "
                            "into the checkpoint, so a resume against a different table is refused.")
    group.add_argument("--action_label_cfg_drop_prob", default=0.2, type=float,
                       help="Per-sample probability of hard-dropping the action condition during "
                            "training (replaced by a learned null embedding), enabling classifier-free "
                            "guidance at sampling time via --action_label_cfg_scale. Default 0.2.")


def add_data_options(parser, training=False):
    """Training selects all clips by default; optional JSON rules narrow the pool."""
    group = parser.add_argument_group('dataset')
    group.add_argument("--train_split", default='train', choices=['train', 'val', 'all'], type=str,
                       dest='train_split',
                       help="Dataset selection: 'train'=all non-validation datasets, 'val'=validation dataset, 'all'=all datasets.")
    if not training:
        group.add_argument("--objects_subset", default='all', type=str,
                           help="Legacy checkpoint object subset (read from checkpoint settings).")
    group.add_argument("--joint_buckets", default=None, type=int, nargs='+',
                       help="Training batch padding widths (joint-count ceilings); MAX_JOINTS is always appended. "
                            "Default: param_utils.JOINT_BUCKETS. A small subset whose rigs span several buckets can "
                            "pass one ceiling (e.g. '--joint_buckets 80') so its clips share batches.")
    if training:
        group.add_argument("--sample_filter", default=None, type=str,
                           help="Optional JSON include/exclude clip rules. Default: all clips in the training split.")
        group.add_argument("--val_sample_filter", default=None, type=str,
                           help="Independent JSON filter for the validation split (never inherits the training filter).")

def add_training_options(parser):
    group = parser.add_argument_group('training')
    group.add_argument("--cond_path", required=True, type=str,
                       help="Path to the cond.npy that defines this run. It names the species and, "
                            "through each entry's dataset namespace/root, the dataset directories "
                            "holding their clips -- so single-dataset and merged multi-dataset "
                            "training differ only in which cond.npy is passed. Build a merged one "
                            "with tools/merge_dataset_cond.py. A copy is written to save_dir so the "
                            "checkpoint carries its own inference contract.")
    group.add_argument("--save_dir", type=str,
                       help="Path to save checkpoints and results.")
    group.add_argument("--model_prefix", type=str,
                       help="Unique string at the beggining of the model name.")
    group.add_argument("--auto_resume", action='store_true',
                       help="If passed, automatically resume from the latest checkpoint in save_dir. Without this flag, training starts fresh and existing checkpoints in save_dir are overwritten.")
    group.add_argument("--ml_platform_type", default='NoPlatform', choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform', 'WandBPlatform'], type=str,
                       help="Choose platform to log results. NoPlatform means no logging.")
    group.add_argument("--amp_dtype", default='fp32', choices=['fp32', 'fp16', 'bf16'], type=str,
                       help="Autocast precision for training. fp32 disables AMP; fp16 uses selective autocast with GradScaler; bf16 uses selective autocast on linear and attention modules only.")
    group.add_argument("--compile", default='None',
                       choices=['None', 'default', 'reduce-overhead', 'max-autotune-no-cudagraphs'],
                       help="Wrap the model forward with torch.compile to fuse the many tiny "
                            "per-layer kernels (training is kernel-launch-bound). "
                            "Mode: 'None' (no compile), 'default' (torch.compile default), "
                            "'reduce-overhead' (CUDA Graphs for eligible model forward/backward regions), "
                            "'max-autotune-no-cudagraphs'. Sampling, optimizer and logging stay outside "
                            "the compiled region. On Windows, requires Triton's bundled "
                            "TinyCC; MSVC is not used and C++ codegen is disabled. "
                            "First step pays a one-time compile "
                            "cost; shapes are static so no recompile thrashing afterward "
                            "(one graph per JOINT_BUCKETS joint bucket).")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--lr_decay_start", default=None, type=int,
                       help="Optimizer step at which the LR starts a cosine decay from --lr "
                            "to --lr_final, reaching --lr_final at --num_steps. Omitted: the "
                            "LR stays constant for the whole run.")
    group.add_argument("--lr_final", default=1e-5, type=float,
                       help="LR at the end of the cosine decay (only with --lr_decay_start).")

    group.add_argument("--weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
    group.add_argument("--eval_interval", default=0, type=int,
                       help="Compute loss over val every N training steps (and at the last step), "
                            "logged under Val/. 0 disables validation.")
    group.add_argument("--val_t_strata", default=4, type=int,
                       help="Timesteps scored per val clip, one per equal-width stratum of "
                            "[0, T) (1 = a single uniform draw). More strata cut the variance "
                            "of Val/ losses; the mean stays the uniform-t objective.")
    group.add_argument("--log_interval", default=100, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=10_000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default=600_000, type=int,
                       help="Training will stop after the specified number of steps.")
    group.add_argument("--num_frames", default=60, type=int,
                       help="The model's window T in frames. Each clip's L source frames (cropped to at most "
                            "MAX_SOURCE_FRAMES_MULT * T) are resampled into it, and L / T is its "
                            "resample_speed_cond.")
    group.add_argument("--min_length", default=20, type=int,
                       help="Shortest source clip, in frames: shorter clips are dropped, and generation "
                            "rejects a shorter --num_frames.")
    group.add_argument("--sample_limit", default=0, type=int,
                       help="Limit the number of motion clips loaded for tiny overfit/debug runs. 0 keeps the full dataset.")
    group.add_argument("--motion_cache_size", default=0, type=int,
                       help="Number of entries to keep in in-memory LRU cache for raw motion clips per dataset instance. 0 disables the cache.")
    group.add_argument("--main_process_prefetch_batches", default=4, type=int,
                       help="Number of training batches kept ready ahead of the GPU (spread across --num_workers when it is > 0). 0 selects the default of 4.")
    group.add_argument("--num_workers", default=0, type=int,
                       help="Training data worker processes. 0 prepares batches on a background thread of the training "
                            "process, sharing its GIL; N > 0 runs N persistent worker processes. Validation always uses the thread.")
    group.add_argument("--detect_anomaly", action='store_true',
                       help="Enable PyTorch autograd anomaly detection. Useful for debugging, but significantly slows training.")
    # Spike-capture probe is always on (hardcoded in TrainLoop) and always
    # serializes the offending batch (.pt) + top per-parameter grad norms next to
    # the JSON summary. It dumps to <save_dir>/spikes whenever a step's pre-clip
    # grad_norm exceeds the threshold below. Only the threshold / dump cap tune.
    group.add_argument("--spike_grad_threshold", default=50.0, type=float,
                       help="Pre-clip grad_norm above this value triggers a spike dump.")
    group.add_argument("--spike_start_step", default=1000, type=int,
                       help="Skip spike checks for the first N steps (warmup), where the optimizer has not "
                           "settled and early grad spikes are routine noise. 0 = check from step 1.")
    group.add_argument("--spike_max_dumps", default=10, type=int,
                       help="Stop writing spike dumps after this many, to bound disk usage. 0 = unlimited.")
    group.add_argument("--sample_loss_limit", default=8.0, type=float,
                       help="Cap each sample's l_simple gradient at that of a sample whose l_simple is this many "
                            "times the running geometric mean at its diffusion timestep (a per-sample Huber on the "
                            "RMS error; see train/sample_loss_limit.py). Stops one outlier clip from owning the "
                            "clipped batch gradient, touching only the far tail of the samples. 0 disables it.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")
    group.add_argument("--use_ema", action='store_true',
                       help="If True, will use EMA model averaging.")
    group.add_argument("--ema_rate", default=0.99, type=float,
                       help="EMA decay rate (closer to 1 = slower updates). Default 0.99.")
    group.add_argument("--rare_head_word_floor", default=0, type=int,
                       help="Minimum sampling mass for RARE head words: a head word (the "
                            "action_label's first head word) with fewer than this many clips in "
                            "the training subset is weighted as if it had this many, so each of "
                            "its clips is drawn floor/count times as often as a clip of a common "
                            "word (capped by --rare_head_word_max_boost). Words at or above the "
                            "floor are untouched; draws are otherwise uniform over clips and "
                            "species are not balanced. One rule for the whole tail, so a new rare "
                            "word needs no entry in --head_word_weights. 0 or 1 disables it. "
                            "Changes the training distribution: start a new run rather than "
                            "resuming an old one with a different floor. Default 0 (off).")
    group.add_argument("--rare_head_word_max_boost", default=4.0, type=float,
                       help="Cap on the per-clip multiplier --rare_head_word_floor may give a "
                            "word, so a 1-clip word is not drawn floor times as often (the tail "
                            "mostly needs data, not mass; past ~4x the model just memorises the "
                            "clip). With floor 20 and cap 4: 1-5 clips x4, 6 x3.3, 10 x2, 16 "
                            "x1.25, >=20 x1. Must be >= 1; 1 turns the floor off. Default 4.")
    group.add_argument("--head_word_weights", default="", type=str,
                       help="Explicit per-clip sampling multipliers keyed on the action_label's "
                            "first head word, as 'word=weight,word=weight' (e.g. 'stop=3,sheathe=4'), "
                            "applied ON TOP of --rare_head_word_floor. A clip of weight w is drawn "
                            "w times as often as an unlisted clip (weight 1). For the case where "
                            "one word needs a hand-set weight; the floor covers the tail in general, "
                            "so this is normally left empty. A multiplier on attack/idle only takes "
                            "exposure from everything else. Words must be HEAD_VOCAB entries present "
                            "in the training subset. Changes the training distribution: start a "
                            "new run rather than resuming an old one with a different list. "
                            "Default: none.")


def add_sampling_options(parser):
    group = parser.add_argument_group('sampling')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--output_dir", default='', type=str,
                       help="Path to results dir (auto created by the script). "
                            "If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--cond_path", default='', type=str,
                       help="provide cond.py path in case you wish to generate motion for skeleton not included in Truebones dataset.")
    group.add_argument("--amp_dtype", default='fp32', choices=['fp32', 'bf16'], type=str,
                       help="Autocast precision for inference. fp32 = full precision with TF32 matmuls on CUDA "
                            "(default; slightly slower than bf16). "
                            "bf16 = selective autocast on linear / attention / conv modules; "
                            "softmax stays fp32. Requires a CUDA device with bf16 support (Ampere+). "
                            "bf16 rounding adds frame-to-frame noise that inflates jerk/snap scores "
                            "(docs/bf16_precision_issues.md).")
    group.add_argument("--no_ema", action='store_true',
                       help="Load the online weights (payload 'model') instead of the EMA copy, for "
                            "checkpoints whose EMA still lags the online model (short runs).")
    group.add_argument("--loop", nargs='?', const='on', default='auto', choices=['auto', 'on', 'off'],
                       help="Whether to generate a closed window (loop conditioning + loop-aware temporal masks). "
                            "'on' (also a bare --loop): the window is periodic -- its last frame is one step "
                            "before frame 0, whatever number of cycles it holds. A --reference_motion is placed "
                            "into it the same way and the sampled window is rescaled to --num_frames the same "
                            "way, so the output is a loop whatever the reference is. "
                            "'off': an open window. 'auto' (default): with a --reference_motion, follow its own "
                            "loop verdict; else, with --action_label and no reference, 'on' when the label "
                            "is phase-free (heads walk, run, swim, fly, idle ... with only gait / posture / "
                            "held-object modifiers -- the loader's roll/tile whitelist) and 'off' for an "
                            "anchored event (heads jump, attack ...; modifiers look, eat, shake ...); "
                            "else 'off'.")
    group.add_argument("--loop_cycles", default=None, type=parse_loop_cycles,
                       help="How many cycles a loop window holds (k: the circular table's period is "
                            "N / k and the loop tempo L / k). "
                            "Unset: with neither this nor --num_frames, 1 cycle at the label's loop period "
                            "P (the median loop clip length the auto length uses); with --num_frames N, the "
                            "integer nearest N / P in log space (rounded down for a high-frequency label, "
                            "which training only slows down); 1 when no period is known. Given alone, "
                            "--num_frames defaults to k * P. With a --reference_motion the window is read "
                            "as one cycle (the reference is taken as unseen, its cycles are its own), and "
                            "the flag is refused. A phase-anchored loop label accepts only 1. "
                            "Warns when the cycle length N / k or the source budget is outside what training drew.")
    group.add_argument("--fullbody_ik", action='store_true',
                       help="Decode the BVH preview with the same full-body IK as restore_glb_from_npy --fullbody-ik: "
                            "rotations are re-solved on the rigid cond skeleton so the position channels are honoured "
                            "through rotations instead of per-joint local translations. "
                            "Only affects BVH export; .npy features are unchanged.")
    group.add_argument("--stretch_factor", default=0.1, type=float,
                       help="Allowed bone-length elasticity for --fullbody_ik (0.1 = +/-10%%; 0 = perfectly rigid, "
                            "which also makes the BVH rotation-only). Same meaning as restore_glb_from_npy --stretch-factor.")


def add_generate_options(parser):
    group = parser.add_argument_group('generate')
    group.add_argument("--num_frames", default=None, type=int,
                       help="The number of frames in the sampled motion. Omit it to let "
                            "generation pick one, in this priority order: --reference_motion's "
                            "native length (R frames) when a reference is given; else, when "
                            "--action_label is given, the median length of the training clips "
                            "carrying that label -- the target species' own clips if it has any, "
                            "otherwise the most similar species' -- so the resample_speed "
                            "condition the model is handed stays inside its training distribution "
                            "(needs a cond.npy baked by tools/regenerate_dataset_artifacts.py); "
                            "else the checkpoint's native window (60). "
                            "When R != M the reference is resampled to M frames (tempo "
                            "change). "
                            "Valid range: [min_length, MAX_SOURCE_FRAMES_MULT*num_frames] "
                            "of the checkpoint (param_utils.MAX_SOURCE_FRAMES_MULT); an "
                            "auto-picked length is clamped into it.")
    group.add_argument("--object_type", default=None, type=str,
                       help="Target object type. Optional if --reference_motion is provided "
                            "(inferred from filename), or if --cond_path points at a cond file "
                            "containing a single species (that species is used). Otherwise "
                            "required for pure-random generation. When both flags are provided "
                            "the reference motion must use the same species and skeleton.")
    group.add_argument("--sampling_method", default="ddpm", choices=["p", "ddpm", "ddim"],
                       help="Diffusion sampler to use. 'p'/'ddpm' = DDPM (default). 'ddim' = DDIM. "
                            "Both cost the same at the same --sampling_steps. DDIM is deterministic "
                            "at ddim_eta=0, and that determinism is what makes it worse at holding a "
                            "steady stride rate -- an early, locally inconsistent phase estimate "
                            "can never be re-mixed away. Default: ddpm.")
    group.add_argument("--sampling_steps", default=100, type=int,
                       help="Number of respaced diffusion steps. 0 = use checkpoint's full diffusion_steps. Default: 100.")
    group.add_argument("--ddim_eta", default=0.0, type=float,
                       help="DDIM eta parameter. 0.0 = deterministic. Default: 0.0.")
    group.add_argument("--reference_motion", default=None, type=str,
                       help="Same-rig motion .npy/.fbx/.glb/.gltf used for img2img. Requires --skip_timesteps. "
                            "Cross-skeleton motion transfer is handled by the motion autoencoder.")
    group.add_argument("--skip_timesteps", default=None, type=int,
                       help="Timesteps to skip for same-rig img2img. Higher = more faithful to reference. "
                            "Range: 0~sampling_steps. Required with --reference_motion.")
    group.add_argument("--action_label", default="", type=str,
                       help="Text-to-motion prompt for this generation. Controlled-vocabulary "
                            "tokens ONLY, comma-separated, in the canonical order the labels use: "
                            "action word(s) first, then the direction bound to the last one, then "
                            "any other modifiers in vocabulary order ('walk, forward', "
                            "'fly, left, glide', 'attack, bite'). Free text is NOT accepted -- an "
                            "unknown token is a hard error listing the valid ones, because the "
                            "vectors live in the checkpoint and no T5 runs at generation. A "
                            "recognizable prompt written out of canonical order is rewritten to it "
                            "(with a printed note); head-word order is kept as given, since it "
                            "affects the head-slot weighting. An omitted direction is filled from "
                            "checkpoint training statistics when direction coverage is >= 80%; "
                            "use --auto_direction off to keep it empty. A label names "
                            "the MOTION only and says nothing about what the character holds, so "
                            "'attack, slash' covers the armed and the unarmed strike alike; name an "
                            "implement as a modifier ('attack, slash, shield') when the implement "
                            "IS the motion. Empty = "
                            "unconditional (the learned null embedding). Requires a checkpoint "
                            "trained with --action_label_cond.")
    group.add_argument("--auto_direction", default='auto', choices=['auto', 'off'],
                       help="Fill omitted directions using checkpoint-local statistics. Near-top "
                            "direction combinations (within 10 percentage points) are sampled "
                            "in proportion to training counts, reproducibly using --seed.")
    group.add_argument("--action_label_cfg_scale", default=1.0, type=float,
                       help="Classifier-free guidance scale over --action_label. 1.0 (default) = "
                            "off: one forward per diffusion step, the conditional prediction as-is. "
                            ">1 amplifies the prompt by extrapolating away from the model's own "
                            "unconditional prediction (x0_uncond + s*(x0_cond - x0_uncond)) on the "
                            "high-noise half of the steps, where global posture is decided; the "
                            "detail steps use 1 (plain conditional, one forward). Requires "
                            "--action_label and a checkpoint trained with a non-zero "
                            "--action_label_cfg_drop_prob (without it there is no unconditional "
                            "mode to guide away from).")
    group.add_argument("--species_tags", default="", type=str,
                       help="Override the target species' descriptor for this generation: a body "
                            "plan, optionally followed by a variant, e.g. 'Quadruped' or "
                            "'Biped,Clanking'. Comma/semicolon-separated. It must be in the checkpoint's "
                            "precomputed species descriptor table (the closed body-plan x variant "
                            "vocabulary; no T5 runs at generation); its row replaces the species "
                            "descriptor baked into cond.npy (default from species_tags.jsonl), letting "
                            "you restyle the generated motion (e.g. make a Winged Dragon move as a "
                            "Quadruped, or a human walk like a machine). "
                            "Requires a checkpoint trained with --species_cond. "
                            "Incompatible with --object_type all. "
                            "Editing a raw .glb/.fbx of a species the checkpoint does not know "
                            "still reads the tags, as that skeleton's own descriptor.")

def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser, training=True)
    add_model_options(parser)
    add_training_options(parser)
    return parser.parse_args()


def reference_requires_edit_options(reference_motion, skip_timesteps=None):
    """True when a reference is given without an explicit img2img strength."""
    return bool(reference_motion) and skip_timesteps is None


def generate_args(argv=None):
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_data_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    # These CLI args are generation-time overrides and must NOT be
    # overwritten by the training args.json (which stores their defaults).
    # Checkpoint action_group is training metadata and is not restored for generation.
    preserve_cli_args = {'action_label', 'species_tags', 'auto_direction'}
    args = parse_and_load_from_model(
        parser, argv=argv,
        preserve_cli_args=preserve_cli_args,
    )
    if reference_requires_edit_options(args.reference_motion, args.skip_timesteps):
        parser.error('--reference_motion requires --skip_timesteps; '
                     'use the motion autoencoder for cross-skeleton transfer')
    return args

def process_new_skeleton_args():
    parser = ArgumentParser()
    group = parser.add_argument_group('process_new_skeleton')
    group.add_argument("--tpos-path", default=None, type=str,
                       help="An FBX/GLB/GLTF file whose bind/rest pose defines the NPY encoding base. "
                            "Optional with --motion-path: the first motion file's own bind pose is used.")
    group.add_argument("--motion-path", nargs='+', default=None, type=str,
                       help="Animation FBX/GLB/GLTF file(s), or directories of them, of this skeleton. "
                            "Each is encoded into motions/<object_type>_<action>.npy alongside cond.npy, on "
                            "the --tpos-path rest pose (default: the first file's own bind pose), so a "
                            "self-contained animation yields its cond and its feature clip in one run.")
    group.add_argument("--save-dir", required=True, type=str,
                       help="Output directory.")
    group.add_argument("--object-type", default=None, type=str,
                       help="A character's species/type name (e.g. \"Dragon\"). "
                            "When omitted, the tpos-path file stem minus trailing pose "
                            "tokens (e.g. 'Pet_Kiki_A_Tpose.glb' -> 'Pet_Kiki_A').")
    group.add_argument("--crop-enabled", action='store_true', default=False,
                       help="Enable automatic skeleton cropping to MAX_JOINTS=100. "
                            "Off by default because inference has no joint cap; "
                            "enable for training-compatible preprocessing.")
    group.add_argument("--species-tags", required=True, type=str,
                       help="Comma-separated species tags (motion descriptor) for --object-type, "
                            "e.g. 'Quadruped' or 'Biped,Clanking'. REQUIRED for a new skeleton: it "
                            "defines the descriptor baked into cond.npy (generation looks its vector up "
                            "in the checkpoint's species descriptor table). There is no fallback to "
                            "the default dataset's species_tags.jsonl, so it must be supplied "
                            "explicitly.")
    group.add_argument("--action-group", default=None, choices=ACTION_GROUPS,
                       help="Action group of every --motion-path clip. Only 'locomotion' changes the "
                            "encoding: those clips take the root-travel detrend and extent bound, as in "
                            "dataset preprocessing. Default: decided per clip from its file name "
                            "(e.g. 'Run01Forwards' is locomotion, 'FlyAttack' is not).")
    group.add_argument("--reference-cond-path", required=True, type=str,
                       help="REQUIRED. cond.npy to inherit the global canonical "
                            "standardization statistics from. Those statistics belong to a "
                            "trained checkpoint, so this must be the checkpoint's own "
                            "cond.npy snapshot (there is no fallback to the processed "
                            "dataset directory). A joint whose name text it never encodes "
                            "gets the blank (all-zero) name the model reads as unknown.")
    group.add_argument("--yes", action='store_true', default=False,
                       help="Skip all interactive confirmation prompts (e.g. existing data "
                            "overwrite prompt). Useful for headless / automated calls.")
    group.add_argument("--export-tpose-bvh", action='store_true', default=False,
                       help="Also write a single-frame t-pose BVH preview to "
                            "<save-dir>/bvh_tpose/<object_type>.bvh (rest offsets from "
                            "cond.npy, canonical frame and units).")
    args = parser.parse_args()
    return args
