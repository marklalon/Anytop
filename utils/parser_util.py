from argparse import ArgumentParser
import argparse
import os
import json
import copy
import sys

def parse_and_load_from_model(parser, argv=None):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_model_options(parser)
    args = parser.parse_args(argv)
    args_to_overwrite = []
    for group_name in ['dataset', 'model', 'diffusion']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    if isinstance(args.model_path, list) and len(args.model_path) == 1:
        args.model_path = args.model_path[0]
    
    # load args from model
    assert not isinstance(args, list) and not isinstance(args.model_path, list), 'Deprecated feature..'
    args = extract_args(copy.deepcopy(args), args_to_overwrite, args.model_path)

    return args

def extract_args(args, args_to_overwrite, model_path):
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])

    for a, default in (('num_frames', 60), ('min_length', 20)):
        if a in model_args:
            setattr(args, a, model_args[a])
        elif not hasattr(args, a):
            setattr(args, a, default)

    # backward compatibility
    if isinstance(args.emb_trans_dec, bool):
        if args.emb_trans_dec:
            args.emb_trans_dec = 'cls_tcond_cross_tcond'
        else: 
            args.emb_trans_dec = 'cls_none_cross_tcond'
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
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
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
    group.add_argument("--arch", default='trans_enc',
                       choices=['trans_enc', 'trans_dec', 'gru'], type=str,
                       help="Architecture types as reported in the paper.")
    group.add_argument("--emb_trans_dec", default=False, type=bool,
                       help="For trans_dec architecture only, if true, will inject condition as a class token"
                            " (in addition to cross-attention).")
    group.add_argument("--layers", default=4, type=int,
                       help="Number of layers.")
    group.add_argument("--latent_dim", default=128, type=int,
                       help="Transformer/GRU width.")
    group.add_argument("--lambda_geo", default=0.0, type=float, help="Geodesic rotation loss weight (SO(3) distance between predicted and target rotations).")
    group.add_argument("--lambda_vel", default=0.0, type=float,
                       help="Weight for velocity-position consistency loss (0.0=off)."
                            " Penalizes |pos[t+1]-pos[t] - vel[t]|^2 on denormalized outputs."
                            " Couples position and velocity feature groups to prevent independent memorization.")
    group.add_argument("--lambda_loop_wrap", default=0.0, type=float,
                       help="Weight for loop-only wrap loss on denormalized pose/rotation/terminal_vel channels.")
    group.add_argument("--lambda_loop_root_xz", default=0.0, type=float,
                       help="Weight for loop-only translation-root XZ closure loss."
                           " Penalizes net X/Z velocity drift over the visible loop while preserving in-loop root motion.")
    group.add_argument("--loop_cond_prob", default=0.0, type=float,
                       help="Probability that a loop training clip stays loop-conditioned "
                            "(periodic resampling, circular phase, and loop-condition embedding)."
                            " 0.0 = all loop clips treated as non-loop; 1.0 = always keep loop path."
                            " Controls both the model loop-condition projection and dataset loop processing.")
    group.add_argument("--t5_out_dim", default=0, type=int, help=argparse.SUPPRESS)
    group.add_argument("--temporal_window", default=31, type=int,
                       help="temporal window size")
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
    group.add_argument("--global_energy_cond", action='store_true',
                       help="Enable an always-on clip-level global energy condition derived from the training motion's global energy mean/std.")
    group.add_argument("--global_energy_cfg_drop_prob", default=0.1, type=float,
                       help="CFG drop probability for global energy during training. Randomly replaces "
                           "the energy condition with the running mean, forcing the model to learn "
                           "to actually respond to it. Default 0.1.")

def add_data_options(parser):
    group = parser.add_argument_group('dataset')
    group.add_argument("--data_dir", default="", type=str,
                       help="If empty, will use defaults according to the specified dataset.")
    group.add_argument("--train_split", default='train', choices=['train', 'val', 'test', 'all'], type=str,
                       dest='train_split',
                       help="Data split to use for training. 'train'=training set, 'val'=validation set, 'test'=test set, 'all'=use all data.")
    group.add_argument("--objects_subset", default='all', type=str,
                       help="Object subset. Can be a predefined category (e.g. 'all', 'quadropeds', 'flying', 'bipeds', 'millipeds', etc.) or a single species name (e.g. 'Horse', 'Dragon').")
    group.add_argument("--action_tags", default='', type=str,
                       help="Comma-separated action tags, e.g. 'locomotion,attack'. During training, filters "
                            "kept motions whose metadata tags match. During generation with --score, filters "
                            "the scorer's reference prior selection.")

def add_training_options(parser):
    group = parser.add_argument_group('training')
    group.add_argument("--save_dir", type=str,
                       help="Path to save checkpoints and results.")
    group.add_argument("--model_prefix", type=str,
                       help="Unique string at the beggining of the model name.")
    group.add_argument("--auto_resume", action='store_true',
                       help="If passed, automatically resume from the latest checkpoint in save_dir. Without this flag, training starts fresh and existing checkpoints in save_dir are overwritten.")
    group.add_argument("--ml_platform_type", default='NoPlatform', choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform', 'WandBPlatform'], type=str,
                       help="Choose platform to log results. NoPlatform means no logging.")
    group.add_argument("--amp_dtype", default='fp32', choices=['fp32', 'bf16'], type=str,
                       help="Autocast precision for training. fp32 disables AMP; bf16 uses selective autocast on linear and attention modules only.")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--lr_scheduler_step_size", default=10000, type=int,
                       help="StepLR step size: decay LR every N optimizer steps.")
    group.add_argument("--lr_scheduler_gamma", default=0.99, type=float,
                       help="StepLR gamma: multiplicative factor for LR decay.")

    group.add_argument("--weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")
    group.add_argument("--eval_batch_size", default=16, type=int,
                       help="Batch size during evaluation loop. Do not change this unless you know what you are doing. "
                            "T2m precision calculation is based on fixed batch size 16.")
    group.add_argument("--eval_split", default='val', choices=['val', 'test'], type=str,
                       help="Which held-out split to evaluate on during training.")
    group.add_argument("--eval_during_training", action='store_true',
                       help="If True, will run evaluation during training.")
    group.add_argument("--eval_interval", default=1_000, type=int,
                       help="Run validation loss every N training steps when eval_during_training is enabled.")
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
    group.add_argument("--joint_mask_prob", default=0.5, type=float,
                       help="Per-sample probability of applying a training-time subtree joint perturbation. "
                           "Selected joints keep their supervision loss and remain visible to attention, but their x_t "
                           "features are re-noised at an independent timestep to mimic mixed reliability during inpainting.")
    group.add_argument("--joint_mask_budget", default=0.15, type=float,
                       help="Maximum fraction of non-root joints to include in each sampled subtree perturbation.")
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
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")
    group.add_argument("--use_ema", action='store_true',
                       help="If True, will use EMA model averaging.")
    group.add_argument("--ema_rate", default=0.99, type=float,
                       help="EMA decay rate (closer to 1 = slower updates). Default 0.99.")
    group.add_argument("--balanced", action='store_true',
                       help="Use balancing sampler for fairness between topologies")
    group.add_argument("--drop_last", action='store_true', default=False,
                       help="Drop the last incomplete batch at the end of each epoch. "
                            "False means the last batch may be smaller than the others.")


def add_sampling_options(parser):
    group = parser.add_argument_group('sampling')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--output_dir", default='', type=str,
                       help="Path to results dir (auto created by the script). "
                            "If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--num_samples", default=10, type=int,
                       help="Maximal number of prompts to sample, "
                            "if loading dataset from file, this field will be ignored.")
    group.add_argument("--num_repetitions", default=3, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--cond_path", default='', type=str,
                       help="provide cond.py path in case you wish to generate motion for skeleton not included in Truebones dataset.")
    group.add_argument("--amp_dtype", default='fp32', choices=['fp32', 'bf16'], type=str,
                       help="Autocast precision for inference. fp32 = full precision (default). "
                            "bf16 = selective autocast on linear / attention / conv modules; "
                            "softmax stays fp32. Requires a CUDA device with bf16 support (Ampere+).")
    group.add_argument("--loop", action='store_true',
                       help="Generate with loop conditioning and loop-aware temporal masks when supported by the checkpoint.")


def add_generate_options(parser):
    group = parser.add_argument_group('generate')
    group.add_argument("--motion_frames", default=None, type=int,
                       help="The number of frames in the sampled motion. "
                            "If omitted with --reference_motion, defaults to the "
                            "reference's native length (R frames); otherwise defaults to 60. "
                            "When specified with --reference_motion: if R < M the tail "
                            "is auto-outpainted, if R > M the reference is cropped to M. "
                            "Valid range: [min_length, 2*num_frames] of the checkpoint.")
    group.add_argument("--object_type", default=None, type=str,
                       help="Target object type. Optional if --reference_motion is provided "
                            "(inferred from filename). Required for pure-random generation. "
                            "When both flags are provided and the inferred type differs from "
                            "this value, the reference motion is auto-retargeted to this skeleton.")
    group.add_argument("--sampling_method", default="ddim", choices=["p", "ddpm", "ddim"],
                       help="Diffusion sampler to use. 'p'/'ddpm' = DDPM (slow, high quality). 'ddim' = DDIM (fast, recommended). Default: ddim.")
    group.add_argument("--sampling_steps", default=100, type=int,
                       help="Number of respaced diffusion steps. 0 = use checkpoint's full diffusion_steps. Default: 100.")
    group.add_argument("--ddim_eta", default=0.0, type=float,
                       help="DDIM eta parameter. 0.0 = deterministic. Default: 0.0.")
    group.add_argument("--reference_motion", default=None, type=str,
                       help="Path to a reference motion .npy/.fbx/.glb/.gltf file. Non-NPY inputs are "
                           "preprocessed into the same 13-channel feature-space NPY used by training, then "
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
    group.add_argument("--global_energy", default=None, type=float,
                       help="Override the clip-level global energy in normalized (Z-score) space. "
                           "0.0 = training average intensity, 1.0 = 1 std above average, -0.5 = half std below average. "
                           "If omitted, generation defaults to the checkpoint's running mean (equivalent to 0.0).")
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

    group.add_argument("--score", action='store_true',
                       help="After generation, automatically run the motion quality scorer on the output "
                            "and print a quality report. Uses --action_tags (already supported for training) "
                            "to filter the scorer's reference prior, e.g. 'locomotion,attack'.")

def add_render_options(parser):
    group = parser.add_argument_group('render')
    group.add_argument('--bvh_path', type=str, default='assets/Truebones_Chicken', help='path of animation bvh file')
    group.add_argument('--save_dir', type=str, default='save/render_out', help='')
    group.add_argument('--scale', type=float, default=0.7, help='')
    group.add_argument('--cylinder_radius', type=float, default=0.42, help='')
    group.add_argument('--sphere_radius', type=float, default=0.56, help='')
    group.add_argument("--subset", default='bipeds', choices=['quadropeds' , 'flying', 'bipeds', 'millipeds_snakes'], type=str, help="Object subset.")


def add_evaluation_options(parser):
    group = parser.add_argument_group('eval')
    group.add_argument("--eval_mode", default='npy_loc',type=str, choices=['npy_rot', 'npy_loc'], help="Path to gt dir.")
    group.add_argument("--benchmark_path", default='eval/benchmarks/benchmark_all.txt', type=str,  help="Path to benchmark character names. If empty, will use all excluding the characters_to_exclude")
    group.add_argument("--eval_gt_dir", default='dataset/truebones/zoo/truebones_processed/motions', type=str, help="Path to gt dir.")
    group.add_argument("--eval_gen_dir", required=True, type=str, help="Path to gen dir.")
    group.add_argument("--characters_to_exclude", default='MouseyNoFingers,Mousey_m,Trex,SabreToothTiger,Raptor2', type=str, help="Comma separated list of characters to exclude. The default is character with more than 40 motions.")
    group.add_argument("--unique_str", default='', type=str, help="A string to be added to the file name to identify a specific change. Should start with '_'.")

def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
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
    args = parse_and_load_from_model(parser, argv=argv)
    return args

def process_new_skeleton_args():
    parser = ArgumentParser()
    group = parser.add_argument_group('process_new_skeleton')
    group.add_argument("--object-type", default=None, type=str,
                       help="A character's species/type name (e.g. \"Dragon\"). "
                            "When omitted, inferred from the first filename in --anim-dir.")
    group.add_argument("--anim-dir", default=None, type=str,
                       help="Path to a directory containing animation files (FBX/GLB/GLTF) of the skeleton. "
                            "More files improve statistical accuracy for motion denormalization. "
                            "If omitted, defaults to the parent directory of --tpos-path.")
    group.add_argument("--save-dir", required=True, type=str,
                       help="Output directory.")
    group.add_argument("--face-joints-names", default=None, type=str, nargs=4,
                       help="Optional manual override for the four orientation joints ([right hip, left hip, right shoulder, left shoulder] or equivalent). \
                           When omitted, preprocessing tries to infer them from semantic joint names.")
    group.add_argument("--tpos-path", default=None, type=str,
                       help="An FBX/GLB/GLTF file whose bind/rest pose defines the NPY encoding base. "
                           "If omitted, the code will auto-select one from --anim-dir using filename heuristics "
                           "(T-pose/rest/bind > idle > walk > first file).")
    group.add_argument("--retarget-top-k", default=None, type=int,
                       help="Auto-select the top-k most similar training skeletons as motion donors, "
                            "retarget all their motions to the new skeleton, and use those coarse motions "
                            "to compute proper mean/std for cond.npy. Mutually exclusive with --anim-dir.")
    group.add_argument("--training-cond-path",
                       default="dataset/truebones/zoo/truebones_processed/cond.npy",
                       type=str,
                       help="Path to the training dataset's cond.npy, used for donor selection when "
                            "--retarget-top-k is set. Default: dataset/truebones/zoo/truebones_processed/cond.npy")
    group.add_argument("--donor-skeletons", default=None, type=str,
                       help="Comma-separated donor skeleton names to use instead of auto-selection, "
                            "e.g. 'Bison,Cow,Horse'. Only effective with --retarget-top-k.")
    group.add_argument("--update", action='store_true',
                       help="Incremental update mode. Instead of clearing --save-dir and rebuilding "
                           "from scratch, merge the current --anim-dir batch into the existing "
                           "dataset, replacing any older clips produced from the same source files "
                           "while keeping untouched sources, then rebuild the side "
                            "artifacts (cond.npy mean/std, motion_metadata.json, metadata.txt, "
                            "positions_error_rate.txt) over the merged clip set. Clips whose source "
                           "file is present in the update batch are force-replaced. Supports both "
                           "--anim-dir updates and --retarget-top-k updates, and requires an existing "
                           "--save-dir when updating in place.")
    args = parser.parse_args()
    return args

def evaluation_parser():
    parser = ArgumentParser()
    add_base_options(parser)
    add_evaluation_options(parser)
    return parser.parse_args()

def render_parser():
    parser = ArgumentParser()
    add_render_options(parser)
    if "--" not in sys.argv:
        argv = []
    else:
        argv = sys.argv[sys.argv.index("--") + 1:]
    return parser.parse_args(argv)

