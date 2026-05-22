from __future__ import annotations

from pathlib import Path

import numpy as np
import torch.nn as nn

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


def model_supports_reference_conditioning(model) -> bool:
    unwrapped_model = unwrap_anytop_model(model)
    return bool(
        getattr(unwrapped_model, 'reference_cond', False)
        and getattr(unwrapped_model, 'reference_encoder', None) is not None
    )


class ClassifierFreeReferenceModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.num_layers = getattr(model, 'num_layers', 0)
        self.reference_cond = getattr(model, 'reference_cond', False)

    @staticmethod
    def _copy_y(y, reference_motion):
        if y is None:
            return None
        routed_y = dict(y)
        routed_y.pop('reference_cond_mask', None)
        if reference_motion is None:
            for key in list(routed_y.keys()):
                if key.startswith('reference_'):
                    routed_y.pop(key, None)
        else:
            routed_y['reference_motion'] = reference_motion
        return routed_y

    def forward(self, x, timesteps, get_layer_activation=-1, y=None, train_step=None, **unused_kwargs):
        if y is None or y.get('reference_motion') is None:
            return self.model(
                x,
                timesteps,
                get_layer_activation=get_layer_activation,
                y=y,
                train_step=train_step,
                **unused_kwargs,
            )

        scale = float(y.get('reference_scale', 1.0))
        if scale == 0.0:
            return self.model(
                x,
                timesteps,
                get_layer_activation=get_layer_activation,
                y=self._copy_y(y, None),
                train_step=train_step,
                **unused_kwargs,
            )

        cond_y = self._copy_y(y, y.get('reference_motion'))
        if scale == 1.0:
            return self.model(
                x,
                timesteps,
                get_layer_activation=get_layer_activation,
                y=cond_y,
                train_step=train_step,
                **unused_kwargs,
            )

        uncond_y = self._copy_y(y, None)
        cond_output = self.model(
            x,
            timesteps,
            get_layer_activation=get_layer_activation,
            y=cond_y,
            train_step=train_step,
            **unused_kwargs,
        )
        uncond_output = self.model(
            x,
            timesteps,
            get_layer_activation=get_layer_activation,
            y=uncond_y,
            train_step=train_step,
            **unused_kwargs,
        )

        if isinstance(cond_output, tuple):
            cond_tensor, cond_activations = cond_output
            if isinstance(uncond_output, tuple):
                uncond_tensor, uncond_activations = uncond_output
            else:
                uncond_tensor = uncond_output
                uncond_activations = None
            guided = uncond_tensor + scale * (cond_tensor - uncond_tensor)
            if uncond_activations is None:
                return guided, cond_activations
            guided_activations = {
                layer: uncond_activations[layer] + scale * (cond_activations[layer] - uncond_activations[layer])
                for layer in cond_activations
            }
            return guided, guided_activations

        return uncond_output + scale * (cond_output - uncond_output)

def load_model(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0, f"Unexpected keys in checkpoint: {unexpected_keys}"
    assert len(missing_keys) == 0, f"Missing keys in checkpoint: {missing_keys}"

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
            'max_joints': max_joints, 
            'feature_len':feature_len,  'skip_t5': args.skip_t5, 'value_emb': args.value_emb,
            'cross_limb': True, 'cross_limb_latents': args.cross_limb_latents,
            'cross_limb_dim': getattr(args, 'cross_limb_dim', 64),
            'cross_limb_last_n': getattr(args, 'cross_limb_last_n', 0),
            'joint_mask_prob': getattr(args, 'joint_mask_prob', 0.0),
            'reference_cond': getattr(args, 'reference_cond', False),
            'reference_encoder_layers': getattr(args, 'reference_encoder_layers', 1),
            'reference_cond_prob': getattr(args, 'reference_cond_prob', 0.2),
            'reference_residual_gate': getattr(args, 'reference_residual_gate', 1.0),
            'reference_token_dropout_prob': getattr(args, 'reference_token_dropout_prob', 0.0),
            'reference_token_noise_std': getattr(args, 'reference_token_noise_std', 0.0),
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
    )