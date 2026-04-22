from __future__ import annotations

from pathlib import Path

import numpy as np

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

def load_model(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    unexpected_keys = [key for key in unexpected_keys if not key.startswith('quality_proxy.')]
    assert len(unexpected_keys) == 0, f"Unexpected keys in checkpoint: {unexpected_keys}"
    assert all([
        k.startswith('clip_model.') or k.startswith('action_conditioner.')
        for k in missing_keys
    ]), f"Unexpected missing keys: {[k for k in missing_keys if not k.startswith('clip_model.') and not k.startswith('action_conditioner.')]}"

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
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': getattr(args, 'dropout_prob', 0.1), 'activation': "gelu", 'cond_mode': cond_mode,
            'action_cond_mask_prob': args.action_cond_mask_prob, 'max_joints': max_joints, 
            'feature_len':feature_len,  'skip_t5': args.skip_t5, 'value_emb': args.value_emb, 'root_input_feats': 13,
            'use_action_cond': getattr(args, 'use_action_cond', False)}

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
        joint_mask_prob=getattr(args, 'joint_mask_prob', 0.0),
        joint_mask_max_frac=getattr(args, 'joint_mask_max_frac', 0.3),
        lambda_vel=getattr(args, 'lambda_vel', 0.0),
    )