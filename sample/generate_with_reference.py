"""generate_with_reference.py
============================
Generate motion for a target skeleton guided by a reference motion sequence
(which may come from a *different* skeleton and may contain noise or missing data).

Three guidance modes
--------------------
soft    Gradient guidance via cond_fn (robust to noisy / incomplete references).
        Use --guidance_scale to interpolate between free generation (0) and
        tighter reference following (> 5).  No model changes required.

hard    Inpainting-style hard constraint on the mapped joints (all frames set to True
        in inpainting_mask).  Works best for same-skeleton, clean references.
        Cannot handle noisy inputs gracefully.

hybrid  Root joint is hard-constrained; all other mapped joints use soft guidance.
        Gives stable root trajectory while keeping other joints flexible.

Cross-skeleton joint alignment
-------------------------------
For each target joint we find the reference joint whose T5-encoded joint-name
embedding has the highest cosine similarity.  Only pairs above
--confidence_threshold are included.  The mapped reference features are
re-normalised from reference-skeleton statistics into target-skeleton statistics
before any comparison, so the guidance loss is in a meaningful space.

By default only the velocity channels (9, 10, 11) are used for the guidance
loss.  These are the most transferable features across different skeletons.
Add position channels (0-2) for same-skeleton guidance or when skeletons are
very similar.  Rotation channels (3-8) are skeleton-specific and should only
be added for exact same-topology cases.

NOTE  The reference .npy file must be a *processed* Truebones motion file
      (shape [T, J, 13], denormalised) – the same format written by generate.py
      and stored in the assets/ directory.

Example usage
-------------
# Cross-skeleton: generate Bear motion guided by a Hound reference
python -m sample.generate_with_reference ^
    --model_path save/.../model000024999.pt ^
    --object_type Bear ^
    --reference_path assets/Hound___Attack_470.npy ^
    --reference_object_type Hound ^
    --guidance_mode soft ^
    --guidance_scale 5.0 ^
    --motion_length 3

# Same-skeleton: tighter control with position + velocity channels
python -m sample.generate_with_reference ^
    --model_path save/.../model000024999.pt ^
    --object_type Hound ^
    --reference_path assets/Hound___Attack_470.npy ^
    --reference_object_type Hound ^
    --guidance_mode hybrid ^
    --guidance_scale 8.0 ^
    --guidance_channels 0 1 2 9 10 11
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from os.path import join as pjoin

from utils.fixseed import fixseed
from utils import dist_util
from utils.model_util import create_model_and_diffusion_general_skeleton, load_model
from data_loaders.tensors import truebones_batch_collate
from data_loaders.truebones.data.dataset import create_temporal_mask_for_window
from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np
from data_loaders.truebones.truebones_utils.get_opt import get_opt
from data_loaders.truebones.truebones_utils.plot_script import plot_general_skeleton_3d_motion
from model.conditioners import T5Conditioner
import BVH
from InverseKinematics import animation_from_positions


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = ArgumentParser(description=__doc__)

    # ── Paths ──────────────────────────────────────────────────────────────
    p.add_argument("--model_path", required=True,
                   help="Path to model checkpoint (.pt file).")
    p.add_argument("--cond_path", default="",
                   help="Path to cond.npy (auto-detected from opt.cond_file if empty).")
    p.add_argument("--output_dir", default="",
                   help="Output directory (auto-named under model dir if empty).")

    # ── Run control ────────────────────────────────────────────────────────
    p.add_argument("--seed", default=10, type=int)
    p.add_argument("--device", default=0, type=int)
    p.add_argument("--num_repetitions", default=1, type=int)
    p.add_argument("--motion_length", default=6.0, type=float,
                   help="Duration of generated motion in seconds.")

    # ── Target skeleton ────────────────────────────────────────────────────
    p.add_argument("--object_type", required=True,
                   help="Target skeleton to generate (must be a key in cond.npy).")

    # ── Reference motion ───────────────────────────────────────────────────
    p.add_argument("--reference_path", required=True,
                   help="Path to processed .npy reference motion (shape [T, J, 13]).")
    p.add_argument("--reference_object_type", required=True,
                   help="Skeleton type of the reference (must be a key in cond.npy).")

    # ── Guidance ───────────────────────────────────────────────────────────
    p.add_argument("--guidance_mode", default="soft",
                   choices=["soft", "hard", "hybrid"],
                   help="soft: gradient guidance (robust to noise/missing data).\n"
                        "hard: inpainting constraint (same-skeleton, clean reference).\n"
                        "hybrid: hard root + soft remaining joints.")
    p.add_argument("--guidance_scale", default=5.0, type=float,
                   help="Gradient guidance strength for soft/hybrid mode. "
                        "0 = no guidance.  Typical range 1-20.")
    p.add_argument("--confidence_threshold", default=0.3, type=float,
                   help="Minimum T5 cosine similarity to include a joint pair in the mapping.")
    p.add_argument("--guidance_channels", default=[9, 10, 11], type=int, nargs="+",
                   help="Feature-vector channel indices used in the soft guidance loss.  "
                        "9-11 are root-relative velocity (safe for cross-skeleton).  "
                        "0-2 are positions, 3-8 are rotations (same-skeleton only).")

    # ── Model hyperparams (auto-loaded from args.json, kept as fallback) ──
    p.add_argument("--latent_dim", default=128, type=int)
    p.add_argument("--layers", default=4, type=int)
    p.add_argument("--t5_name", default="t5-base")
    p.add_argument("--temporal_window", default=31, type=int)
    p.add_argument("--noise_schedule", default="cosine")
    p.add_argument("--sigma_small", default=True, type=bool)
    p.add_argument("--lambda_fs", default=0.0, type=float)
    p.add_argument("--lambda_geo", default=0.0, type=float)
    p.add_argument("--cond_mask_prob", default=0.1, type=float)
    p.add_argument("--skip_t5", action="store_true")
    p.add_argument("--value_emb", action="store_true")

    args = p.parse_args()

    # Override model hyperparams from the checkpoint's saved args.json
    saved_path = os.path.join(os.path.dirname(args.model_path), "args.json")
    if os.path.exists(saved_path):
        with open(saved_path) as f:
            saved = json.load(f)
        for key in ["latent_dim", "layers", "t5_name", "temporal_window",
                    "noise_schedule", "sigma_small", "lambda_fs", "lambda_geo",
                    "cond_mask_prob", "skip_t5", "value_emb"]:
            if key in saved:
                setattr(args, key, saved[key])

    args.batch_size = 1
    return args


# ─────────────────────────────────────────────────────────────────────────────
# Joint mapping via T5 name embeddings
# ─────────────────────────────────────────────────────────────────────────────

def compute_joint_mapping(ref_cond, tgt_cond, t5, threshold):
    """For each target joint find the best-matching reference joint by cosine
    similarity of T5-encoded joint names.

    Returns
    -------
    mapping : dict {tgt_j: (ref_j, confidence)}
              Only entries with confidence >= threshold are included.
    """
    ref_embs = F.normalize(
        t5(t5.tokenize(ref_cond["joints_names"])).detach(), dim=-1)   # [J_ref, D]
    tgt_embs = F.normalize(
        t5(t5.tokenize(tgt_cond["joints_names"])).detach(), dim=-1)   # [J_tgt, D]

    cos = tgt_embs @ ref_embs.T   # [J_tgt, J_ref]

    mapping = {}
    for tgt_j in range(cos.shape[0]):
        ref_j = int(cos[tgt_j].argmax().item())
        conf  = float(cos[tgt_j, ref_j].item())
        if conf >= threshold:
            mapping[tgt_j] = (ref_j, conf)

    same = (ref_cond["joints_names"] == tgt_cond["joints_names"])
    mode_str = "same-skeleton" if same else "cross-skeleton"
    print(f"\n[joint mapping] {mode_str} | "
          f"{len(mapping)}/{len(tgt_cond['joints_names'])} target joints mapped "
          f"(threshold={threshold})")
    for tgt_j, (ref_j, conf) in sorted(mapping.items()):
        tgt_n = tgt_cond["joints_names"][tgt_j]
        ref_n = ref_cond["joints_names"][ref_j]
        print(f"  tgt[{tgt_j:2d}] {tgt_n:<30s} <- ref[{ref_j:2d}] {ref_n:<30s}  "
              f"conf={conf:.3f}")
    print()
    return mapping


# ─────────────────────────────────────────────────────────────────────────────
# Reference motion preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def resample_temporal(motion, target_len):
    """Linear-resample motion [T_src, J, C] to [target_len, J, C]."""
    if motion.shape[0] == target_len:
        return motion
    t = torch.from_numpy(motion).float()          # [T, J, C]
    t = t.permute(1, 2, 0).unsqueeze(0)           # [1, J, C, T]
    _B, J, C, T = t.shape
    resampled = F.interpolate(
        t.reshape(1, J * C, T),
        size=target_len, mode="linear", align_corners=False,
    ).reshape(1, J, C, target_len).squeeze(0)     # [J, C, target_len]
    return resampled.permute(2, 0, 1).numpy()      # [target_len, J, C]


def build_reference_tensor(ref_path, ref_cond, tgt_cond, joint_mapping,
                           max_joints, n_frames, device):
    """Load and preprocess the reference motion into target-skeleton feature space.

    The reference .npy is expected to be *denormalised* (raw features, shape
    [T, J_ref, 13]) – the same format produced by generate.py and stored in
    assets/.

    For each mapped joint pair (ref_j -> tgt_j) the raw features are
    re-normalised into target-skeleton statistics so that the values are
    directly comparable with the model's normalised predictions.

    Returns
    -------
    ref_tensor : torch.Tensor  [1, max_joints, 13, n_frames] on `device`
                 Only the mapped target-joint slots are filled; the rest are 0.
    """
    motion = np.load(ref_path, allow_pickle=True)

    # Guard: some outputs (e.g. from edit.py) are saved as dicts
    if isinstance(motion, np.ndarray) and motion.ndim == 0:
        motion = motion.item()
    if isinstance(motion, dict):
        motion = motion["motion"]  # edit.py saves {"motion": ..., ...}

    motion = np.asarray(motion, dtype=np.float32)  # [T_ref, J_ref, 13]

    if motion.ndim != 3 or motion.shape[-1] != 13:
        raise ValueError(
            f"Expected reference motion shape [T, J, 13], got {motion.shape}.  "
            f"Provide a processed Truebones .npy file (output of generate.py).")

    motion = resample_temporal(motion, n_frames)   # [n_frames, J_ref, 13]

    mean_ref = ref_cond["mean"]   # [J_ref, 13]
    std_ref  = ref_cond["std"]    # [J_ref, 13]
    mean_tgt = tgt_cond["mean"]   # [J_tgt, 13]
    std_tgt  = tgt_cond["std"]    # [J_tgt, 13]

    ref_tensor = torch.zeros(1, max_joints, 13, n_frames, device=device)

    for tgt_j, (ref_j, _conf) in joint_mapping.items():
        feat_raw = motion[:, ref_j, :]                                      # [T, 13] – already raw
        feat_tgt = (feat_raw - mean_tgt[tgt_j]) / (std_tgt[tgt_j] + 1e-6)  # norm into tgt space
        feat_tgt = np.nan_to_num(feat_tgt).astype(np.float32)
        ref_tensor[0, tgt_j, :, :] = (
            torch.from_numpy(feat_tgt).T.to(device)   # [13, T]
        )

    return ref_tensor.detach()


# ─────────────────────────────────────────────────────────────────────────────
# Soft gradient guidance
# ─────────────────────────────────────────────────────────────────────────────

def make_soft_guidance_fn(ref_tensor, joint_mapping, guidance_channels, guidance_scale):
    """Return a cond_fn compatible with gaussian_diffusion.condition_mean_with_grad.

    At each diffusion step:
      1. Reads pred_xstart (the model's clean-sample prediction) from p_mean_var.
      2. Computes a confidence-weighted MSE between pred_xstart and ref_tensor
         at the mapped joints over the selected feature channels.
      3. Returns  -guidance_scale * grad_{x_t}(loss)  so that condition_mean_with_grad
         shifts the diffusion mean toward the reference.

    The gradient flows through the model because pred_xstart depends on x_t.
    Torch's enable_grad() inside p_sample_with_grad ensures this works even
    when the outer loop runs under no_grad().
    """
    ch = guidance_channels

    def cond_fn(x, t, p_mean_var, **kwargs):
        pred = p_mean_var["pred_xstart"]   # [B, max_joints, 13, T]

        loss = torch.tensor(0.0, device=pred.device)
        total_conf = 0.0

        for tgt_j, (_ref_j, conf) in joint_mapping.items():
            pred_f = pred[:, tgt_j, :, :][:, ch, :]       # [B, C, T]
            ref_f  = ref_tensor[:, tgt_j, :, :][:, ch, :]  # [B, C, T]
            loss   = loss + conf * F.mse_loss(pred_f, ref_f)
            total_conf += conf

        if total_conf > 0:
            loss = loss / total_conf

        grad = torch.autograd.grad(loss, x, allow_unused=True)[0]
        if grad is None:
            return torch.zeros_like(x)
        return -guidance_scale * grad   # negative: ascent on log p(y|x)

    return cond_fn


# ─────────────────────────────────────────────────────────────────────────────
# Hard inpainting helpers
# ─────────────────────────────────────────────────────────────────────────────

def apply_hard_inpainting(model_kwargs, ref_tensor, joint_mapping, sample_shape, device):
    """Set inpainted_motion and inpainting_mask in model_kwargs.

    All frames of each mapped target joint are hard-constrained to the
    corresponding reference features.  The diffusion model fills everything
    else freely.
    """
    inpainted = torch.zeros(sample_shape, device=device)
    mask      = torch.zeros(sample_shape, dtype=torch.bool, device=device)

    for tgt_j in joint_mapping:
        inpainted[:, tgt_j, :, :] = ref_tensor[:, tgt_j, :, :]
        mask[:, tgt_j, :, :]      = True

    model_kwargs["y"]["inpainted_motion"] = inpainted
    model_kwargs["y"]["inpainting_mask"]  = mask


# ─────────────────────────────────────────────────────────────────────────────
# Skeleton condition construction (metadata only, no source motion)
# ─────────────────────────────────────────────────────────────────────────────

def _encode_joints_names(names, t5):
    return t5(t5.tokenize(names)).detach().cpu().numpy()


def build_skeleton_condition(object_type, cond_dict, n_frames, temporal_window,
                             t5, max_joints, feature_len):
    """Build the cond['y'] dict for unconditional generation of one skeleton.

    Mirrors the logic in sample/generate.py::create_condition().
    """
    cond  = cond_dict[object_type]
    n_j   = len(cond["parents"])
    mean  = cond["mean"]
    std   = cond["std"]
    tpos  = np.nan_to_num((cond["tpos_first_frame"] - mean) / (std + 1e-6))
    embs  = _encode_joints_names(cond["joints_names"], t5)

    batch = [
        np.zeros((n_frames, n_j, feature_len)),          # placeholder motion
        n_frames,                                         # m_length
        cond["parents"],
        tpos,                                             # tpos_first_frame
        cond["offsets"],
        create_temporal_mask_for_window(temporal_window, n_frames),
        cond["joints_graph_dist"],
        cond["joint_relations"],
        object_type,
        embs,                                             # joints_names_embs
        0,                                                # crop_start_ind
        mean,
        std,
        max_joints,
    ]
    return truebones_batch_collate([batch])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    fixseed(args.seed)

    opt        = get_opt(args.device)
    cond_path  = args.cond_path or opt.cond_file
    cond_dict  = np.load(cond_path, allow_pickle=True).item()

    for ot in [args.object_type, args.reference_object_type]:
        if ot not in cond_dict:
            available = ", ".join(sorted(cond_dict.keys()))
            raise KeyError(f"Unknown object_type '{ot}'.  Available: {available}")

    fps         = opt.fps
    n_frames    = int(args.motion_length * fps)
    max_joints  = opt.max_joints
    feature_len = opt.feature_len

    dist_util.setup_dist(args.device)
    dev = dist_util.dev()

    out_path = args.output_dir or os.path.join(
        os.path.dirname(args.model_path),
        f"ref_guided_{args.object_type}_from_{args.reference_object_type}"
        f"_{args.guidance_mode}_scale{args.guidance_scale}_seed{args.seed}",
    )
    os.makedirs(out_path, exist_ok=True)

    # ── Model & T5 ───────────────────────────────────────────────────────────
    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion_general_skeleton(args)
    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location="cpu")
    if "model_avg" in state_dict:
        print("EMA checkpoint detected, loading model_avg weights.")
        state_dict = state_dict["model_avg"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]
    load_model(model, state_dict)

    print("Loading T5 conditioner...")
    t5 = T5Conditioner(name=args.t5_name, finetune=False, word_dropout=0.0,
                       normalize_text=False, device="cuda")
    model.to(dev)
    model.eval()

    # ── Joint mapping (T5 cosine similarity) ─────────────────────────────────
    ref_cond      = cond_dict[args.reference_object_type]
    tgt_cond      = cond_dict[args.object_type]
    joint_mapping = compute_joint_mapping(
        ref_cond, tgt_cond, t5, args.confidence_threshold)

    if not joint_mapping:
        print("[WARNING] No joints mapped above confidence threshold.  "
              "Generation will be unconditional.  "
              "Try lowering --confidence_threshold.\n")

    # ── Reference tensor (renorm'd into target-skeleton feature space) ────────
    ref_tensor = build_reference_tensor(
        args.reference_path, ref_cond, tgt_cond,
        joint_mapping, max_joints, n_frames, dev,
    )

    # ── Skeleton condition dict ───────────────────────────────────────────────
    _, model_kwargs = build_skeleton_condition(
        args.object_type, cond_dict, n_frames,
        args.temporal_window, t5, max_joints, feature_len,
    )
    model_kwargs["y"] = {
        k: v.to(dev) if torch.is_tensor(v) else v
        for k, v in model_kwargs["y"].items()
    }

    sample_shape = (args.batch_size, max_joints, feature_len, n_frames)

    # ── Guidance-mode-specific setup ──────────────────────────────────────────
    cond_fn           = None
    cond_fn_with_grad = False

    if args.guidance_mode == "soft" and joint_mapping:
        cond_fn = make_soft_guidance_fn(
            ref_tensor, joint_mapping,
            args.guidance_channels, args.guidance_scale,
        )
        cond_fn_with_grad = True

    elif args.guidance_mode == "hard" and joint_mapping:
        apply_hard_inpainting(
            model_kwargs, ref_tensor, joint_mapping, sample_shape, dev)

    elif args.guidance_mode == "hybrid" and joint_mapping:
        # Root joint (index 0) → hard constraint
        root_map = {j: v for j, v in joint_mapping.items() if j == 0}
        rest_map = {j: v for j, v in joint_mapping.items() if j != 0}

        if root_map:
            apply_hard_inpainting(
                model_kwargs, ref_tensor, root_map, sample_shape, dev)
        if rest_map:
            cond_fn = make_soft_guidance_fn(
                ref_tensor, rest_map,
                args.guidance_channels, args.guidance_scale,
            )
            cond_fn_with_grad = True

    print(f"[guidance] mode={args.guidance_mode}  scale={args.guidance_scale}"
          f"  channels={args.guidance_channels}"
          f"  cond_fn={'yes' if cond_fn else 'no'}"
          f"  cond_fn_with_grad={cond_fn_with_grad}\n")

    # ── Sampling ──────────────────────────────────────────────────────────────
    for rep_i in range(args.num_repetitions):
        print(f"### Sampling repetition {rep_i}")

        sample = diffusion.p_sample_loop(
            model,
            sample_shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
            cond_fn=cond_fn,
            cond_fn_with_grad=cond_fn_with_grad,
        )

        # ── Decode & save ─────────────────────────────────────────────────────
        n_joints = model_kwargs["y"]["n_joints"][0].item()
        motion   = sample[0, :n_joints]                          # [J, 13, T]
        mean     = cond_dict[args.object_type]["mean"][None, :]  # [1, J, 13]
        std      = cond_dict[args.object_type]["std"][None, :]

        motion_np = motion.cpu().permute(2, 0, 1).numpy() * std + mean  # [T, J, 13]

        parents = cond_dict[args.object_type]["parents"]
        offsets = cond_dict[args.object_type]["offsets"]

        global_positions = recover_from_bvh_ric_np(motion_np)
        out_anim, _, _   = animation_from_positions(
            positions=global_positions, parents=parents,
            offsets=offsets, iterations=150,
        )

        stem = (
            f"{args.object_type}_from_{args.reference_object_type}"
            f"_{args.guidance_mode}_scale{args.guidance_scale:.1f}_rep{rep_i}"
        )
        mp4_out = pjoin(out_path, stem + ".mp4")
        npy_out = pjoin(out_path, stem + ".npy")
        bvh_out = pjoin(out_path, stem + ".bvh")

        plot_general_skeleton_3d_motion(
            mp4_out, parents, global_positions, title=stem, fps=fps)
        np.save(npy_out, motion_np)
        if out_anim is not None:
            BVH.save(bvh_out, out_anim, cond_dict[args.object_type]["joints_names"])

        print(f"Saved: {mp4_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
