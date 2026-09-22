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
# The training value for one model over the whole corpus
# (docs/unified_action_group_training.md). Recorded in args.json in a group's
# place; generation then reads the group off the label instead of the weights.
ACTION_GROUP_ALL = 'all'

# Checkpoint compatibility stamp, written into every save_dir's args.json by
# train_anytop and required back by extract_args. Bump it for ANY change that
# alters training or inference semantics, including ones that leave the
# state_dict layout untouched -- those are exactly the changes that would
# otherwise load cleanly and generate wrong motion, reading as a quality
# regression rather than an incompatibility.
CKPT_VERSION = 20

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
    from data_loaders.truebones.truebones_utils.physics_joint_annotation import (
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


def apply_checkpoint_action_group(args, model_args, args_path):
    """Set this generation's action group from the checkpoint being sampled.

    A single-group checkpoint was trained on that group's corpus alone, so the
    group is a property of the weights and generation has no ``--action_group``
    flag to contradict it: the value comes out of args.json and nowhere else.

    An ``all`` checkpoint was trained on the whole corpus and carries no group
    condition. It leaves the group EMPTY, which every consumer reads as "any
    group": the label's first head word determines the group by itself, so
    nothing downstream needs the value except the clip-length prior, which
    matches across groups when it is empty (utils/clip_length_prior).

    A run predating the mandatory flag also records no group.
    """
    recorded_group = str(model_args.get('action_group', '') or '').strip().lower()
    if recorded_group == ACTION_GROUP_ALL:
        recorded_group = ''
    elif recorded_group and recorded_group not in ACTION_GROUPS:
        print(
            f"[parser_util] WARNING: {args_path} records action_group "
            f"'{recorded_group}', which is not one of {', '.join(ACTION_GROUPS)} "
            f"or '{ACTION_GROUP_ALL}'. Treating this checkpoint as group-less: "
            f"the clip-length prior will match a label across every group."
        )
        recorded_group = ''
    args.action_group = recorded_group


def extract_args(args, args_to_overwrite, model_path):
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    assert_checkpoint_version(model_args, args_path)
    assert_joint_condition_schema(model_args, args_path)
    apply_checkpoint_action_group(args, model_args, args_path)

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
    group.add_argument("--action_adaln_bottleneck", default=0, type=int,
                       help="Hidden width of the --action_label_adaln head (0 = latent_dim). "
                            "The action label set has slot-source rank ~137, so ~192 "
                            "loses nothing and halves the model's largest matrix.")
    group.add_argument("--species_film_bottleneck", default=0, type=int,
                       help="Hidden width of the --species_cond FiLM head (0 = latent_dim). "
                            "The species descriptors span rank 94; 128 is enough.")
    group.add_argument("--lambda_geo", default=0.0, type=float, help="Geodesic rotation loss weight (SO(3) distance between predicted and target rotations).")
    group.add_argument("--lambda_vel", default=0.0, type=float,
                       help="Weight for velocity-position consistency loss (0.0=off)."
                            " Penalizes |pos[t+1]-pos[t] - (vel[t] - vel_root[t])|^2 on denormalized outputs,"
                            " the root's velocity subtracted on X/Z only (RIC positions are root-XZ-relative,"
                            " velocities are world deltas), so the residual is exactly zero on real data."
                            " Couples position and velocity feature groups to prevent independent memorization.")
    group.add_argument("--lambda_loop_wrap", default=0.0, type=float,
                       help="Weight for the loop-only seam continuity loss on denormalized outputs: the "
                            "terminal velocity row equals the wrap delta pos[0]-pos[-1], and the wrap "
                            "step's rotation angle matches its neighbouring steps. A loop window's last "
                            "frame is one step before frame 0, never a copy of it.")
    group.add_argument("--lambda_loop_root_closure", default=0.0, type=float,
                       help="Weight for the loop-only full-cycle closure of the translation root's XZ "
                            "velocity (0.0=off): ||sum_t vel_xz[t] * step||^2 over ALL rows, the terminal "
                            "wrap row included, on denormalized outputs. The root's world XZ path lives "
                            "only in ch9/ch11 and every loop target sums to exactly zero there; l_simple "
                            "cannot see the DC bias that integrates into a seam pop, and loop_wrap masks "
                            "the root's XZ. Linear in the output, so the weight carries no bias cost, but "
                            "it is one scalar per sample pushing T*2 elements: on a converged model a "
                            "weight well below 1 already matches l_simple's gradient norm at low t, and "
                            "1.0 is an order of magnitude above it. loop_root_xz_drift (the per-cycle "
                            "seam pop in physical units) is logged whenever this or --lambda_loop_wrap is on.")
    group.add_argument("--lambda_bone", default=0.0, type=float,
                       help="Weight for the target-relative, rest-length-normalized bone-length loss (0.0=off). "
                            "Penalizes each predicted bone length's deviation from the GROUND-TRUTH bone length "
                            "at the same frame, normalized by the rest bone length so short/distal bones (which "
                            "l_simple under-weights and which stretch most on novel skeletons) get proportionally "
                            "larger gradient. Anchoring on GT (not rest) preserves genuinely animated bone-length "
                            "deformation. Computed on denormalized outputs; recommended range ~0.1-0.3.")
    group.add_argument("--motion_speed_aug", default=1.0, type=float,
                       help="Motion-speed augmentation range R (1.0 = off). Each training clip is first "
                            "time-scaled by a log-uniform ratio in [1/R, R] -- played faster (fewer frames) "
                            "or slower (more frames) at the same fps -- and then treated as an ordinary "
                            "source clip: loop roll/tile, crop, the window resample and resample_speed_cond "
                            "all see the scaled length, and the model is told nothing. Loader-only: no cond "
                            "regen, no model change, no CKPT_VERSION bump. Spreads the clustered clip lengths "
                            "(the corpus piles up on a few exact frame counts) so an inference num_frames "
                            "between the clusters is in distribution. The range is narrowed per clip so the "
                            "scaled clip stays >= min_length and a loop that fits the source budget still "
                            "fits (never downgraded to non-loop by slowing down). 1.2 is the intended value.")
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
                            "of the cycle. Loader-only, like --motion_speed_aug: no regen, no bump.")
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
    group.add_argument("--t5_out_dim", default=0, type=int, help=argparse.SUPPRESS)
    group.add_argument("--value_emb", action='store_true',
                       help="If passed, graph multihead attention learns GRPE value embeddings")
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
                       help="Per-joint probability of replacing a joint's ENTIRE name embedding with a "
                            "learned 'unknown joint' vector during training. --dropout_prob thins that "
                            "vector elementwise but never hides the joint's identity, so the model is "
                            "never trained to fall back on rest_pose/graph_dist/joints_relations and a "
                            "single rare name token can flip a limb's motion prior. Default 0.0 (off).")
    group.add_argument("--species_cond", action='store_true',
                       help="Enable per-species FiLM conditioning: the T5-derived species descriptor "
                            "modulates the timestep token multiplicatively (gamma=1+res, beta; zero-init "
                            "so it starts at identity), which every decoder layer re-injects. CFG-droppable "
                            "(see --species_cfg_drop_prob). Requires cond.npy regenerated with 'species_emb'.")
    group.add_argument("--species_cfg_drop_prob", default=0.15, type=float,
                       help="Per-sample probability of hard-dropping the species FiLM condition during "
                            "training (replaced by identity modulation gamma=1, beta=0), enabling "
                            "classifier-free guidance over the species descriptor. Default 0.15.")
    group.add_argument("--species_joint_cond", action='store_true',
                       help="Also fuse the species descriptor into the per-joint name embedding: a FiLM "
                            "head reads [joint_name_emb || species_emb] and emits (gamma, beta) per joint, "
                            "shifting and rescaling joint semantics toward the body-plan context of the "
                            "species. Zero-initialized, so it starts at identity. Never CFG-dropped -- it "
                            "is the always-on species channel. Per-joint structural conditioning, "
                            "orthogonal to (and combinable with) the --species_cond FiLM on the timestep "
                            "token. Requires 'species_emb'.")
    group.add_argument("--action_label_cond", action='store_true',
                       help="Enable action-label conditioning: the clip's action_label ('run, forward, "
                            "left, fast' -- controlled keywords, not prose) is split into words, pooled "
                            "into one frozen-T5 channel per role slot (head / direction / modifier / "
                            "hands), "
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
    group.add_argument("--action_label_adaln", action='store_true',
                       help="Give the action label a MULTIPLICATIVE pathway: a zero-initialised head "
                            "turns the action token into a per-channel scale and shift on the "
                            "temporal and feed-forward BRANCH INPUTS of every decoder layer (the "
                            "residual stream is left unmodulated, and the spatial branch already "
                            "carries the additive timestep offset). The additive action token stays; "
                            "this adds the gain the additive path cannot express, which is what two "
                            "labels sharing a slot (whose vectors are near-parallel by construction, "
                            "e.g. 'turn, left' and 'run, turn, left') need to produce different motion "
                            "without leaning on --action_label_cfg_scale and paying its quality cost. "
                            "Requires --action_label_cond. Costs d^2 + 4*layers*d^2 parameters. "
                            "See docs/conditional_modulation_upgrade.md section 3.")
    group.add_argument("--direction_slot_drop_prob", default=0.15, type=float,
                       help="Per-sample probability of blanking the label's DIRECTION words during "
                            "training while the rest of the label stays. Trains the empty direction "
                            "slot as 'any direction', so a prompt that names none draws one side "
                            "or heading instead of a blend. Stacks with --action_label_cfg_drop_prob, "
                            "which drops the WHOLE label: a row contributes explicit direction "
                            "supervision with probability (1 - cfg_drop) * (1 - this). Lower it if "
                            "direction control comes out weak. Never applied at inference. Default 0.15.")
    group.add_argument("--modifier_slot_drop_prob", default=0.15, type=float,
                       help="Per-sample probability of blanking the label's MODIFIER words during "
                            "training while the head, direction and hands words stay. Trains the "
                            "empty modifier slot as the marginal over modifiers, so a bare 'attack' "
                            "at inference draws some attack (bite, cast, swat ...) instead of the "
                            "few clips annotated with no modifier. Independent of "
                            "--direction_slot_drop_prob per row; the hands slot is never dropped "
                            "(empty there means empty hands). Never applied at inference. Default 0.15.")

def add_data_options(parser, training=False):
    """Dataset selection. ``training=True`` adds the training-only options.

    ``--action_group`` is one of those: it is mandatory when training (it splits
    the corpus, and each group trains its own model) and absent when generating,
    where the group is read back out of the checkpoint instead -- see
    :func:`apply_checkpoint_action_group`.
    """
    group = parser.add_argument_group('dataset')
    group.add_argument("--train_split", default='train', choices=['train', 'val', 'test', 'all'], type=str,
                       dest='train_split',
                       help="Data split to use for training. 'train'=training set, 'val'=validation set, 'test'=test set, 'all'=use all data.")
    group.add_argument("--objects_subset", default='all', type=str,
                       help="Object subset. Can be a predefined category (e.g. 'all', 'quadruped', 'winged', 'biped', 'multiped', etc.) or a single species name (e.g. 'Horse', 'Dragon').")
    if training:
        group.add_argument("--action_group", required=True, type=str,
                           choices=list(ACTION_GROUPS) + [ACTION_GROUP_ALL],
                           help="REQUIRED. The corpus to train on: one group -- 'locomotion' "
                                "(sustained displacement), 'stationary' (in-place / interactive), "
                                "'transition' (pose changes) -- or 'all' for one model over the "
                                "whole corpus. No lists: each clip belongs to exactly one group. "
                                "'all' carries NO group condition; the action label's first head "
                                "word already determines the group (a property of the corpus, "
                                "verified by tools/audit_action_labels.py), so a group token would "
                                "add no information the label does not already carry. Draws are "
                                "uniform over clips, so the group mix is the corpus's own "
                                "(stationary ~53%); see --rare_head_word_floor for the tail. "
                                "Recorded in the checkpoint's args.json, where generation reads it "
                                "back -- a single-group checkpoint can only ever be sampled as that "
                                "group. See docs/unified_action_group_training.md.")

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
                       choices=['None', 'default', 'max-autotune-no-cudagraphs'],
                       help="Wrap the transformer decoder with torch.compile to fuse the many tiny "
                            "per-layer kernels (training is kernel-launch-bound). "
                            "Mode: 'None' (no compile), 'default' (torch.compile default), "
                            "'max-autotune-no-cudagraphs'. Requires the MSVC build env "
                            "(start_torch_compile_env.ps1). First step pays a one-time compile "
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
    group.add_argument("--eval_interval", default=1_000, type=int,
                       help="Compute the loss over the val split every N training steps (and at the "
                            "last step), logged under Val/. 0 disables validation.")
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
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--min_length", default=20, type=int,
                       help="Variable-length pipeline lower bound in frames. Inference rejects shorter requested lengths.")
    group.add_argument("--sample_limit", default=0, type=int,
                       help="Limit the number of motion clips loaded for tiny overfit/debug runs. 0 keeps the full dataset.")
    group.add_argument("--motion_cache_size", default=0, type=int,
                       help="Number of entries to keep in in-memory LRU cache for raw motion clips per dataset instance. 0 disables the cache.")
    group.add_argument("--main_process_prefetch_batches", default=4, type=int,
                       help="Prefetch this many batches on background thread for main-thread data loading to overlap with GPU compute. 0 disables it.")
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
    group.add_argument("--joint_mask_prob", default=0.5, type=float,
                       help="Per-sample probability of applying a training-time subtree joint perturbation. "
                           "Selected joints keep their supervision loss and remain visible to attention, but their x_t "
                           "features are re-noised at an independent timestep to mimic mixed reliability during inpainting.")
    group.add_argument("--joint_mask_budget", default=0.15, type=float,
                       help="Maximum fraction of non-root joints to include in each sampled subtree perturbation.")
    group.add_argument("--unreliable_mask_drop_prob", default=0.0, type=float,
                       help="Per-sample probability of hiding the training-time unreliable mask from the model. "
                            "The selected joints / frames are still re-noised, the model just is not told which ones, "
                            "so it learns to localize and repair the damage on its own -- the regime at inference when "
                            "no mask is supplied. 0 disables it (the model always sees the mask).")
    group.add_argument("--temporal_span_mask_prob", default=0.0, type=float,
                       help="Probability of sampling a contiguous training-time temporal span perturbation. "
                            "Selected frames are re-noised for all real joints to teach native frame inpainting.")
    group.add_argument("--temporal_span_mask_min_frames", default=4, type=int,
                       help="Minimum length of each sampled training-time temporal span perturbation.")
    group.add_argument("--temporal_span_mask_max_frames", default=12, type=int,
                       help="Maximum length of each sampled training-time temporal span perturbation. "
                            "Must be >= temporal_span_mask_min_frames.")
    group.add_argument("--temporal_span_seam_loss_weight", default=0.0, type=float,
                       help="Weight of a target-relative acceleration (2nd temporal difference) penalty on the position channel, applied in a Gaussian seam band around sampled temporal-span boundaries. Suppresses inpainting-seam acceleration spikes that l_simple and vel_loss do not catch. 0 disables it.")
    group.add_argument("--temporal_span_seam_width", default=2, type=int,
                       help="Radius of the Gaussian seam band on each side of a sampled temporal-span boundary frame.")
    group.add_argument("--renoise_same_level_prob", default=1.0, type=float,
                       help="Per-sample probability that a flagged (joint-mask / temporal-span) region is re-noised "
                            "at the SAME timestep as the rest of the sample (fresh noise, t_random = t) instead of "
                            "the hard branch uniform on [t, T). The same-level branch decouples the unreliable flag "
                            "from 'much noisier than the surroundings', which is the regime inpainting presents at "
                            "every step; the hard branch keeps training the repair of severe local damage. "
                            "0 restores the pure [t, T) draw, 1 (default) makes every flagged region same-level.")
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
    group.add_argument("--loop", nargs='?', const='on', default='auto', choices=['auto', 'on', 'off'],
                       help="Whether to generate a closed window (loop conditioning + loop-aware temporal masks). "
                            "'on' (also a bare --loop): the window is periodic -- its last frame is one step "
                            "before frame 0, whatever number of cycles it holds. A --reference_motion is placed "
                            "into it the same way and the sampled window is rescaled to --num_frames the same "
                            "way, so the output is a loop whatever the reference is. "
                            "'off': an open window. 'auto' (default): with a --reference_motion, follow its own "
                            "loop verdict; else, with --action_label and no reference, 'on' when most of that "
                            "label's training clips are loops (the model only saw each label paired with "
                            "is_loop the way its clips were authored); else 'off'.")
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
                            "When specified with --reference_motion: if R < M the tail "
                            "is auto-outpainted, if R > M the reference is cropped to M. "
                            "Valid range: [min_length, MAX_SOURCE_FRAMES_MULT*num_frames] "
                            "of the checkpoint (param_utils.MAX_SOURCE_FRAMES_MULT); an "
                            "auto-picked length is clamped into it.")
    group.add_argument("--object_type", default=None, type=str,
                       help="Target object type. Optional if --reference_motion is provided "
                            "(inferred from filename), or if --cond_path points at a cond file "
                            "containing a single species (that species is used). Otherwise "
                            "required for pure-random generation. When both flags are provided "
                            "and the inferred type differs from this value, the reference motion "
                            "is auto-retargeted to this skeleton.")
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
                       help="Path to a reference motion .npy/.fbx/.glb/.gltf file. Non-NPY inputs are "
                           "preprocessed into the same 12-channel feature-space NPY used by training, then "
                           "noised to an intermediate timestep (img2img-style). If --object_type is not given, "
                           "it is inferred from the reference filename. If --object_type is given and differs "
                           "from the reference's inferred type, the reference is auto-retargeted to the requested "
                           "skeleton before noising. When filename inference is invalid, the target object_type "
                           "is used as a fallback. "
                            "Omit for pure random generation.")
    group.add_argument("--skip_timesteps", default=None, type=int,
                       help="Number of timesteps to skip when using --reference_motion (img2img). Higher = more faithful to reference. "
                           "Range: 0~sampling_steps. Default: required when using --reference_motion without --inpaint_*; "
                           "default: 0 when combined with --inpaint_* (disabled; use --skip_timesteps N to enable). "
                           "When combined with --inpaint_*, the skip is applied "
                           "only inside the masked region by starting that region from an img2img-noised reference, while "
                          "the unmasked region stays clamped to the original reference throughout denoising.")
    group.add_argument("--inpaint_joints", default="", type=str,
                       help="Motion inpainting (mask painting): comma-separated joint names whose motion is "
                            "REGENERATED while the rest is held to --reference_motion. Names accept any of the "
                            "raw / canonical / canonical_bvh aliases (use the names you see in the exported "
                            "GLB/BVH). Empty = all real joints (pure temporal inpainting). Requires "
                            "--reference_motion. NOTE: do NOT list the root joint for limb inpainting "
                            "(it is the global anchor).")
    group.add_argument("--inpaint_include_subtree", action="store_true", default=True,
                       help="When resolving --inpaint_joints, also regenerate all descendants of each named "
                            "joint (so naming a hip regenerates the whole leg). Enabled by default; disable "
                            "with --no_inpaint_include_subtree.")
    group.add_argument("--no_inpaint_include_subtree", dest="inpaint_include_subtree",
                       action="store_false",
                       help="Disable subtree expansion for --inpaint_joints (regenerate only the named joints).")
    group.add_argument("--inpaint_frames", default="", type=str,
                       help="Motion inpainting frame ranges to REGENERATE, e.g. '40-90' or '0-20,150-180' "
                            "(inclusive, clipped to the reference length). Empty = all frames. Combined with "
                            "--inpaint_joints, the regenerated region is selected-joints x selected-frames; "
                            "everything else is clamped to --reference_motion. Requires --reference_motion.")
    group.add_argument("--action_label", default="", type=str,
                       help="Text-to-motion prompt for this generation. Controlled-vocabulary "
                            "tokens ONLY, comma-separated, in the canonical order the labels use: "
                            "action word(s) first, then the direction bound to the last one, then "
                            "any other modifiers in vocabulary order ('walk, forward', "
                            "'run, left, fast', 'attack, bite'). Free text is NOT accepted -- an "
                            "unknown token is a hard error listing the valid ones, because the "
                            "vectors live in the checkpoint and no T5 runs at generation. A "
                            "recognizable prompt written out of canonical order is rewritten to it "
                            "(with a printed note); head-word order is kept as given, since it "
                            "carries no meaning to the model. Naming no direction is legal and means "
                            "'any' (the model answers with the marginal over directions, which "
                            "training drops direction words at random to teach). The hands axis is "
                            "the opposite: nothing there means EMPTY HANDS, so write 'hand1' / "
                            "'hand2' for one / both hands holding something ('idle' is an unarmed "
                            "idle; 'idle, hand2' a two-handed armed one). Empty = "
                            "unconditional (the learned null embedding). Requires a checkpoint "
                            "trained with --action_label_cond.")
    group.add_argument("--action_label_cfg_scale", default=1.0, type=float,
                       help="Classifier-free guidance scale over --action_label. 1.0 (default) = "
                            "off: one forward per diffusion step, the conditional prediction as-is. "
                            ">1 amplifies the prompt by extrapolating away from the model's own "
                            "unconditional prediction (x0_uncond + s*(x0_cond - x0_uncond)), at 2x "
                            "sampling cost since every step then runs a conditional AND an "
                            "unconditional forward. Requires "
                            "--action_label and a checkpoint trained with a non-zero "
                            "--action_label_cfg_drop_prob (without it there is no unconditional "
                            "mode to guide away from).")
    group.add_argument("--species_tags", default="", type=str,
                       help="Override the target species' motion style tags for this generation, e.g. "
                            "'Quadruped,Heavy,Lumbering'. Comma/semicolon-separated. The tags are re-encoded "
                            "through the same T5 conditioner used at preprocessing and replace the species "
                            "descriptor baked into cond.npy (default from species_tags.jsonl), letting you "
                            "restyle the generated motion (e.g. make a Winged Dragon walk on the ground). "
                            "Requires a checkpoint trained with --species_cond and/or --species_joint_cond. "
                            "Incompatible with --object_type all.")


def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser, training=True)
    add_model_options(parser)
    add_training_options(parser)
    return parser.parse_args()


def generate_args(argv=None):
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_data_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    # These CLI args are generation-time overrides and must NOT be
    # overwritten by the training args.json (which stores their defaults).
    # There is deliberately no --action_group here: the group belongs to the
    # weights, so apply_checkpoint_action_group() sets args.action_group from the
    # checkpoint's own args.json.
    preserve_cli_args = {'action_label', 'species_tags'}
    args = parse_and_load_from_model(
        parser, argv=argv,
        preserve_cli_args=preserve_cli_args,
    )
    return args

def process_new_skeleton_args():
    parser = ArgumentParser()
    group = parser.add_argument_group('process_new_skeleton')
    group.add_argument("--tpos-path", required=True, type=str,
                       help="An FBX/GLB/GLTF file whose bind/rest pose defines the NPY encoding base.")
    group.add_argument("--save-dir", required=True, type=str,
                       help="Output directory.")
    group.add_argument("--object-type", default=None, type=str,
                       help="A character's species/type name (e.g. \"Dragon\"). "
                            "When omitted, inferred from the tpos-path filename.")
    group.add_argument("--crop-enabled", action='store_true', default=False,
                       help="Enable automatic skeleton cropping to MAX_JOINTS=100. "
                            "Off by default because inference has no joint cap; "
                            "enable for training-compatible preprocessing.")
    group.add_argument("--species-tags", required=True, type=str,
                       help="Comma-separated species tags (motion descriptor) for --object-type, "
                            "e.g. 'Quadruped,Large,Lumbering'. REQUIRED for a new skeleton: it "
                            "defines the descriptor baked into cond.npy. There is no fallback to "
                            "the default dataset's species_tags.jsonl, so it must be supplied "
                            "explicitly.")
    group.add_argument("--reference-cond-path", required=True, type=str,
                       help="REQUIRED. cond.npy to inherit the per-object_subset "
                            "standardization statistics from. Those statistics belong to a "
                            "trained checkpoint, so this must be the checkpoint's own "
                            "cond.npy snapshot (there is no fallback to the processed "
                            "dataset directory).")
    group.add_argument("--skip-t5-embeddings", action='store_true', default=False,
                       help="Skip T5 embedding computation (caller will inject via "
                            "attach_t5_embeddings_to_cond with a pre-loaded conditioner).")
    group.add_argument("--yes", action='store_true', default=False,
                       help="Skip all interactive confirmation prompts (e.g. existing data "
                            "overwrite prompt). Useful for headless / automated calls.")
    args = parser.parse_args()
    return args

