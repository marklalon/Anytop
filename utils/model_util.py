from model.anytop import AnyTop
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps

def load_model(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    unexpected_keys = [key for key in unexpected_keys if not key.startswith('quality_proxy.')]
    assert len(unexpected_keys) == 0
    assert all([
        k.startswith('clip_model.')
        for k in missing_keys
    ])

def create_model_and_diffusion_general_skeleton(args):
    model = AnyTop(**get_gmdm_args(args))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion

def get_gmdm_args(args):
    t5_model_dim = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
    }
    # default args
    t5_out_dim = t5_model_dim[args.t5_name]
    njoints = 23
    nfeats = 1
    max_joints=143 #irrelevant
    feature_len=13 #irrelevant
    cond_mode = 'object_type'
    feature_len=13

    return {'njoints': njoints, 'nfeats': nfeats, 't5_out_dim': t5_out_dim,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': getattr(args, 'dropout_prob', 0.1), 'activation': "gelu", 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'max_joints': max_joints, 
            'feature_len':feature_len,  'skip_t5': args.skip_t5, 'value_emb': args.value_emb, 'root_input_feats': 13,
            'disable_reference_branch': args.disable_reference_branch, 'reference_dropout_threshold': args.reference_dropout_threshold}

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
        lambda_fs=args.lambda_fs,
        lambda_geo=args.lambda_geo,
        lambda_confidence_recon=args.lambda_confidence_recon,
        lambda_repair_recon=args.lambda_repair_recon,
        physics_teacher_weight=getattr(args, 'physics_teacher_weight', 0.0),
        physics_teacher_feature_weight=getattr(args, 'physics_teacher_feature_weight', 1.0),
        physics_teacher_margin_weight=getattr(args, 'physics_teacher_margin_weight', 0.25),
        physics_teacher_start_step=getattr(args, 'physics_teacher_start_step', 0),
        physics_teacher_ramp_steps=getattr(args, 'physics_teacher_ramp_steps', 0),
        physics_teacher_max_t=getattr(args, 'physics_teacher_max_t', 30),
        semantic_teacher_weight=getattr(args, 'semantic_teacher_weight', 0.05),
        semantic_teacher_species_weight=getattr(args, 'semantic_teacher_species_weight', 1.0),
        semantic_teacher_action_weight=getattr(args, 'semantic_teacher_action_weight', 1.0),
        semantic_teacher_kl_weight=getattr(args, 'semantic_teacher_kl_weight', 0.25),
        semantic_teacher_start_step=getattr(args, 'semantic_teacher_start_step', 0),
        semantic_teacher_ramp_steps=getattr(args, 'semantic_teacher_ramp_steps', 0),
        semantic_teacher_max_t=getattr(args, 'semantic_teacher_max_t', 30),
        semantic_teacher_temperature=getattr(args, 'semantic_teacher_temperature', 1.0),
    )