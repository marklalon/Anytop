from model.anytop import AnyTop
from diffusion.flow_matching import FlowMatching

def load_model(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])

def create_model_and_diffusion_general_skeleton(args):
    model = AnyTop(**get_gmdm_args(args))
    diffusion = create_flow_matching(args)
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
            'dropout': 0.1, 'activation': "gelu", 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'max_joints': max_joints, 
            'feature_len':feature_len,  'skip_t5': args.skip_t5, 'value_emb': args.value_emb, 'root_input_feats': 13,
            'disable_reference_branch': args.disable_reference_branch, 'reference_dropout_threshold': args.reference_dropout_threshold}

def create_flow_matching(args):
    return FlowMatching(
        num_timesteps=int(getattr(args, 'diffusion_steps', 100)),
        sigma_min=float(getattr(args, 'fm_sigma_min', 1e-4)),
        solver=getattr(args, 'fm_solver', 'euler'),
        timestep_scale=float(getattr(args, 'fm_timestep_scale', 1000.0)),
        sampling_steps=int(getattr(args, 'fm_num_steps', getattr(args, 'diffusion_steps', 100))),
        lambda_fs=args.lambda_fs,
        lambda_geo=args.lambda_geo,
        lambda_confidence_recon=args.lambda_confidence_recon,
        lambda_repair_recon=args.lambda_repair_recon,
        lambda_root=args.lambda_root,
        lambda_velocity=args.lambda_velocity,
    )