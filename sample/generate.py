# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import os
import sys

# Ensure parent directory is in path for local imports when running as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.fixseed import fixseed
import numpy as np
import torch
import torch.nn as nn
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion_general_skeleton, load_model
from utils import dist_util
from data_loaders.tensors import truebones_batch_collate
from data_loaders.truebones.truebones_utils.motion_process import recover_animation_from_motion_np
from data_loaders.truebones.data.dataset import create_temporal_mask_for_window, attach_joint_name_embeddings
from os.path import join as pjoin
from motion_lib import BVH
from data_loaders.truebones.truebones_utils.get_opt import get_opt

class _CFGWrapper(nn.Module):
    """Wraps a model for classifier-free guidance at inference time.

    Performs two forward passes per denoising step (conditioned and
    unconditioned) and returns the linearly extrapolated prediction:
        out = out_uncond + guidance_scale * (out_cond - out_uncond)

    The unconditioned pass is obtained by zeroing the action-tag condition
    (empty action_tags list → zero embedding from ActionTagConditioner).
    All structural conditions (skeleton topology, T-pose, joint names) are
    kept unchanged in both passes.
    """

    def __init__(self, model: nn.Module, guidance_scale: float) -> None:
        super().__init__()
        self.model = model
        self.guidance_scale = guidance_scale

    @property
    def feature_len(self) -> int:
        return self.model.feature_len

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, **model_kwargs) -> torch.Tensor:
        out_cond = self.model(x, timesteps, **model_kwargs)
        if self.guidance_scale == 1.0:
            return out_cond
        bs = x.shape[0]
        y = model_kwargs.get("y", {})
        y_uncond = {**y, "action_tags": [[] for _ in range(bs)], "action_category": [None] * bs}
        out_uncond = self.model(x, timesteps, **{**model_kwargs, "y": y_uncond})
        return out_uncond + self.guidance_scale * (out_cond - out_uncond)


def main(args = None, cond_dict = None):
    if args is None:
        # args is None unless this method is called from another function (e.g. during training)
        args = generate_args()
    fixseed(args.seed)
    opt = get_opt(args.device)
    if cond_dict is None:
        if args.cond_path:
            cond_dict=np.load(args.cond_path, allow_pickle=True).item()
            actual_cond_file = args.cond_path
        else:
            cond_dict = np.load(opt.cond_file, allow_pickle=True).item()
            actual_cond_file = opt.cond_file
    else:
        actual_cond_file = opt.cond_file

    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))        
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    fps = opt.fps
    n_frames = int(args.motion_length * fps)
    max_joints = opt.max_joints
    dist_util.setup_dist(args.device)
    object_types = args.object_type
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
    # mkdir outpath
    os.makedirs(out_path, exist_ok=True)
    args.batch_size = len(object_types)  # Sampling a single batch from the testset, with exactly args.num_samples
    # args.num_repetitions = 1

    print("Creating model and diffusion...")
    # Configure respaced sampling steps before creating diffusion
    sampling_steps = int(getattr(args, 'sampling_steps', 100))
    sampling_method = str(getattr(args, 'sampling_method', 'ddim')).lower()
    if sampling_steps > 0:
        if sampling_method == 'ddim':
            args.timestep_respacing = f'ddim{sampling_steps}'
        elif sampling_method == 'plms':
            args.timestep_respacing = f'ddim{sampling_steps}'  # plms uses same respacing format
        else:
            args.timestep_respacing = str(sampling_steps)
    else:
        args.timestep_respacing = ''
    model, diffusion = create_model_and_diffusion_general_skeleton(args)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    if 'model_avg' in state_dict:
        print("EMA checkpoint detected, loading model_avg weights.")
        state_dict = state_dict['model_avg']
    elif 'model' in state_dict:
        state_dict = state_dict['model']
    load_model(model, state_dict)

    print("Building/loading joint-name T5 embedding cache...")
    attach_joint_name_embeddings(cond_dict, actual_cond_file, opt.data_root, args.t5_name)
    model.to(dist_util.dev())
    model.eval()  # disable random masking
    action_guidance_scale = float(getattr(args, 'action_guidance_scale', 1.0))
    if action_guidance_scale == 0.0:
        # Scale=0 means unconditional output; skip CFG wrapper to avoid double forward pass.
        # The model will be called with empty action_tags (no action_category set below).
        print(f"Action CFG disabled (scale=0): unconditional generation")
    elif action_guidance_scale != 1.0:
        print(f"Action CFG enabled: action_guidance_scale={action_guidance_scale}")
        model = _CFGWrapper(model, action_guidance_scale)
    action_category = getattr(args, 'action_category', None) or None
    _, model_kwargs = create_condition(object_types, cond_dict, n_frames, args.temporal_window, max_joints=opt.max_joints, feature_len=opt.feature_len, action_category=action_category)


    ddim_eta = float(getattr(args, 'ddim_eta', 0.0))
    for rep_i in range(args.num_repetitions):
        fixseed(args.seed + rep_i)
        print(f'### Sampling [repetitions #{rep_i}] method={sampling_method} steps={sampling_steps or "full"}')
        if sampling_method == 'ddim':
            sample = diffusion.ddim_sample_loop(
                model,
                (args.batch_size, max_joints, model.feature_len, n_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True,
                eta=ddim_eta,
            )
        elif sampling_method == 'plms':
            sample = diffusion.plms_sample_loop(
                model,
                (args.batch_size, max_joints, model.feature_len, n_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True,
            )
        elif sampling_method in ('p', 'ddpm'):
            sample = diffusion.p_sample_loop(
                model,
                (args.batch_size, max_joints, model.feature_len, n_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
        else:
            raise ValueError(f"Unknown sampling_method: {sampling_method}")

        # Recover XYZ *positions* from matrix representation
        bs, max_joints, n_feats, n_frames = sample.shape
        for i, motion in enumerate(sample):
            n_joints = model_kwargs['y']["n_joints"][i].item()
            motion = motion[:n_joints]
            object_type = model_kwargs['y']["object_type"][i]
            parents = model_kwargs['y']["parents"][i]
            mean = cond_dict[object_type]['mean'][None, :]
            std = cond_dict[object_type]['std'][None, :]
            motion = motion.cpu().permute(2, 0, 1).numpy() * std + mean
            offsets = cond_dict[object_type]['offsets']
            out_anim, has_animated_pos = recover_animation_from_motion_np(motion, parents, offsets)
            name_pref = '%s_rep_%d'%(object_type, rep_i)
            existing_npy_files = [filename for filename in os.listdir(out_path) if filename.startswith(name_pref) and filename.endswith('.npy')]
            npy_name = name_pref+'_#%d.npy'%(len(existing_npy_files))
            bvh_name = name_pref+'_#%d.bvh'%(len(existing_npy_files))
            np.save(pjoin(out_path, npy_name), motion)
            if out_anim is not None:
                BVH.save(pjoin(out_path, bvh_name), out_anim, cond_dict[object_type]['joints_names'],
                         positions=has_animated_pos)
            print("repetition #" + str(rep_i) + " ,created motion: "+ npy_name)

def create_condition(object_types, cond_dict, n_frames, temporal_window, max_joints, feature_len, action_category=None):
    """Build model_kwargs for a batch of object_types.

    Args:
        action_category (str | None): Optional action category to condition
            generation on (e.g. ``'locomotion'``, ``'attack'``). Must be one
            of the 12 known tags: attack, death, emote, fall, jump,
            locomotion, other, pose, posture, reaction, rise, turn.
            When provided, the action embedding is injected into each sample
            in the batch. Only effective if the model was trained with
            ``--use_action_cond``.
    """
    batches = list()
    for object_type in object_types:
        if object_type not in cond_dict:
            available = ', '.join(sorted(cond_dict.keys()))
            raise KeyError(
                f"Unknown object_type '{object_type}'. Available object types in cond file: {available}"
            )
        batch=list()
         # motion, m_length, parents, joints_perm, inv_joints_perm, tpos_first_frame, offsets, self.temporal_mask_template, joints_graph_dist, joints_relations, object_type, joints_names
        parents = cond_dict[object_type]['parents']
        n_joints = len(parents)
        mean = cond_dict[object_type]['mean']
        std = cond_dict[object_type]['std']
        tpos_first_frame = cond_dict[object_type]['tpos_first_frame']
        tpos_first_frame =  (tpos_first_frame - mean) / (std + 1e-6)
        tpos_first_frame = np.nan_to_num(tpos_first_frame)
        joint_relations = cond_dict[object_type]['joint_relations']
        joints_graph_dist = cond_dict[object_type]['joints_graph_dist']
        offsets = cond_dict[object_type]['offsets']
        joints_names_embs = cond_dict[object_type]['joints_names_embs']
        batch.append(np.zeros((n_frames, n_joints, feature_len)))
        batch.append(n_frames)
        batch.append(parents)
        batch.append(tpos_first_frame)
        batch.append(offsets)
        batch.append(create_temporal_mask_for_window(temporal_window, n_frames))
        batch.append(joints_graph_dist)
        batch.append(joint_relations)
        batch.append(object_type)
        batch.append(joints_names_embs)
        batch.append(0)
        batch.append(mean)
        batch.append(std)
        batch.append(max_joints)
        # Inject action condition metadata so truebones_batch_collate picks it up.
        # b[-2] must be a dict with 'action_category' or 'species_label' (len >= 16).
        tag = str(action_category).strip().lower() if action_category else None
        motion_metadata = {
            'action_category': tag,
            'action_tags': [tag] if tag else [],
        }
        batch.append(motion_metadata)   # index 14: detected as b[-2] when len==16
        batch.append(object_type)       # index 15: used as motion_name string
        batches.append(batch)

    return truebones_batch_collate(batches)


if __name__ == "__main__":
    main()
