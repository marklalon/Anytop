from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from model.anytop import AnyTop
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps


def infer_t5_out_dim_from_cond(cond_source: str | Path | dict) -> int:
    if isinstance(cond_source, dict):
        cond_dict = cond_source
        source_label = 'cond dictionary'
    else:
        source_label = str(cond_source)
        cond_dict = np.load(source_label, allow_pickle=True).item()

    if not isinstance(cond_dict, dict) or not cond_dict:
        raise RuntimeError(f"Unable to infer t5_out_dim from {source_label}: cond is empty or invalid.")

    for object_type in sorted(cond_dict):
        object_cond = cond_dict[object_type]
        joints_names_embs = object_cond.get('joints_names_embs')
        if joints_names_embs is None:
            continue
        joints_names_embs = np.asarray(joints_names_embs)
        if joints_names_embs.ndim == 2 and joints_names_embs.shape[1] > 0:
            return int(joints_names_embs.shape[1])

    raise RuntimeError(
        f"Unable to infer t5_out_dim from {source_label}: no object contains a valid joints_names_embs matrix."
    )


def resolve_t5_out_dim(args, cond_source: str | Path | dict | None = None) -> int:
    configured_dim = int(getattr(args, 't5_out_dim', 0) or 0)
    if configured_dim > 0:
        return configured_dim

    if cond_source is not None:
        resolved_dim = infer_t5_out_dim_from_cond(cond_source)
        setattr(args, 't5_out_dim', resolved_dim)
        return resolved_dim

    raise RuntimeError(
        't5_out_dim is not set and could not be inferred. '
        'Provide a cond.npy with precomputed joints_names_embs or load a checkpoint that stores t5_out_dim.'
    )


def unwrap_anytop_model(model):
    unwrapped_model = model
    while hasattr(unwrapped_model, 'model') and not isinstance(unwrapped_model, AnyTop):
        next_model = getattr(unwrapped_model, 'model')
        if next_model is None or next_model is unwrapped_model:
            break
        unwrapped_model = next_model
    return unwrapped_model


def model_supports_global_energy_conditioning(model) -> bool:
    unwrapped_model = unwrap_anytop_model(model)
    return bool(
        getattr(unwrapped_model, 'global_energy_cond', False)
        and getattr(unwrapped_model, 'global_energy_projection', None) is not None
    )


def load_model(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0, f"Unexpected keys in checkpoint: {unexpected_keys}"
    # QK-norm params (added to bound attention logits) are absent from older
    # checkpoints. They are freshly constructed at their identity init (weight=1),
    # which is exactly what we want when resuming a pre-QK-norm model, so tolerate
    # them as missing. Any other missing key is still a hard error.
    tolerated_suffixes = ('.q_norm.weight', '.k_norm.weight')
    unresolved = [k for k in missing_keys if not k.endswith(tolerated_suffixes)]
    assert len(unresolved) == 0, f"Missing keys in checkpoint: {unresolved}"

def create_model_and_diffusion_general_skeleton(args):
    model = AnyTop(**get_gmdm_args(args))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion

def get_gmdm_args(args):
    t5_out_dim = resolve_t5_out_dim(args)
    njoints = 23
    nfeats = 1
    max_joints=143 #irrelevant
    feature_len=13 #irrelevant
    cond_mode = 'object_type'
    feature_len=13

    return {'njoints': njoints, 'nfeats': nfeats, 't5_out_dim': t5_out_dim,
            'latent_dim': args.latent_dim, 'ff_size': getattr(args, 'ff_size', 1024), 'num_layers': args.layers, 'num_heads': 4,
            'dropout': getattr(args, 'dropout_prob', 0.1), 'activation': "gelu", 'cond_mode': cond_mode,
            'max_joints': max_joints, 
            'feature_len':feature_len,  'value_emb': args.value_emb,
            'cross_limb': True, 'cross_limb_latents': args.cross_limb_latents,
            'cross_limb_dim': getattr(args, 'cross_limb_dim', 64),
            'cross_limb_last_n': getattr(args, 'cross_limb_last_n', 0),
            'joint_mask_prob': getattr(args, 'joint_mask_prob', 0.5),
            'joint_mask_budget': getattr(args, 'joint_mask_budget', 0.15),
            'temporal_span_mask_prob': getattr(args, 'temporal_span_mask_prob', 0.0),
            'temporal_span_mask_min_frames': getattr(args, 'temporal_span_mask_min_frames', 4),
            'temporal_span_mask_max_frames': getattr(args, 'temporal_span_mask_max_frames', 12),
            'global_energy_cond': getattr(args, 'global_energy_cond', False),
            'global_energy_cfg_drop_prob': getattr(args, 'global_energy_cfg_drop_prob', 0.1),
            'species_cond': getattr(args, 'species_cond', False),
            'action_tag_cond': getattr(args, 'action_tag_cond', False),
            'action_tag_cfg_drop_prob': getattr(args, 'action_tag_cfg_drop_prob', 0.3),
            'loop_cond_prob': getattr(args, 'loop_cond_prob', 1.0),
            'root_input_feats': 13}

def create_gaussian_diffusion(args):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = int(getattr(args, 'diffusion_steps', 100))
    scale_beta = 1.  # no scaling
    timestep_respacing = getattr(args, 'timestep_respacing', '')
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_geo=args.lambda_geo,
        lambda_vel=getattr(args, 'lambda_vel', 0.0),
        lambda_loop_wrap=getattr(args, 'lambda_loop_wrap', 0.0),
        temporal_span_seam_loss_weight=getattr(args, 'temporal_span_seam_loss_weight', 0.0),
        temporal_span_seam_width=getattr(args, 'temporal_span_seam_width', 2),
    )