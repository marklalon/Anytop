from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from model.anytop import AnyTop
from data_loaders.truebones.truebones_utils.action_label_conditioning_contract import (
    ACTION_CHECKPOINT_VERSION,
    ActionConditioningError,
)
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


def build_checkpoint_payload(model_state_dict, model_avg_state_dict, model):
    """Wrap the weights with the metadata that says what contract they are under.

    Top-level metadata rather than ``get_extra_state``: the extra-state hook runs
    inside ``state_dict()``/``load_state_dict()``, which the EMA copy also drives,
    and a per-module blob there would have to be kept in sync with the EMA buffer
    sync instead of simply being written once per save.
    """
    action_conditioning = getattr(
        unwrap_anytop_model(model), 'action_conditioning_metadata', None
    )
    return {
        'model': model_state_dict,
        'model_avg': model_avg_state_dict,
        'metadata': {
            'checkpoint_version': ACTION_CHECKPOINT_VERSION,
            'action_conditioning': action_conditioning,
        },
    }


def read_checkpoint_metadata(payload, source: str):
    """The metadata block of a loaded checkpoint, or a hard error.

    A payload with no metadata is a pre-v2 checkpoint: its action condition was
    one vector for the whole label string, so its ``action_label_projection`` is
    a different shape and its weights mean something else. Refused here with the
    reason rather than left to a shape mismatch deep in ``load_state_dict``.
    """
    metadata = payload.get('metadata') if isinstance(payload, dict) else None
    if not isinstance(metadata, dict):
        raise ActionConditioningError(
            f"{source} carries no checkpoint metadata, so it predates checkpoint "
            f"version {ACTION_CHECKPOINT_VERSION}: it was trained on the whole-label "
            "action condition this code has replaced with per-slot word channels. "
            "Its weights cannot be read under the current contract; retrain."
        )
    recorded = metadata.get('checkpoint_version')
    if int(recorded or 0) != ACTION_CHECKPOINT_VERSION:
        raise ActionConditioningError(
            f"{source} records checkpoint_version {recorded!r}, this code writes "
            f"{ACTION_CHECKPOINT_VERSION}."
        )
    return metadata


def load_checkpoint_weights(payload, source: str, prefer_ema: bool):
    """``(model_state, model_avg_state, metadata)`` from a validated payload."""
    metadata = read_checkpoint_metadata(payload, source)
    model_state = payload.get('model')
    model_avg_state = payload.get('model_avg')
    if model_state is None and model_avg_state is None:
        raise ActionConditioningError(f"{source} carries no weights")
    if prefer_ema and model_avg_state is not None:
        return model_avg_state, model_avg_state, metadata
    return (model_state if model_state is not None else model_avg_state,
            model_avg_state, metadata)


def bind_checkpoint_action_conditioning(model, metadata, source: str):
    """Certify the loaded weights against the contract they were trained under."""
    unwrapped = unwrap_anytop_model(model)
    if not getattr(unwrapped, 'action_label_cond', False):
        return None
    action_conditioning = (metadata or {}).get('action_conditioning')
    if action_conditioning is None:
        raise ActionConditioningError(
            f"{source} was saved by a model with no action conditioning, but this run "
            "builds one with --action_label_cond."
        )
    return unwrapped.validate_loaded_action_conditioning(action_conditioning, source)

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
            'species_cond': getattr(args, 'species_cond', False),
            'species_cfg_drop_prob': getattr(args, 'species_cfg_drop_prob', 0.15),
            'species_joint_cond': getattr(args, 'species_joint_cond', False),
            'action_label_cond': getattr(args, 'action_label_cond', False),
            'action_label_cfg_drop_prob': getattr(args, 'action_label_cfg_drop_prob', 0.2),
            # The training entry point builds one bundle and hands the same
            # object to the loader and to the model. Absent at inference: there
            # the checkpoint's own buffers are the word table.
            'action_conditioning': getattr(args, 'action_conditioning', None),
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
        lambda_bone=getattr(args, 'lambda_bone', 0.0),
        temporal_span_seam_loss_weight=getattr(args, 'temporal_span_seam_loss_weight', 0.0),
        temporal_span_seam_width=getattr(args, 'temporal_span_seam_width', 2),
    )