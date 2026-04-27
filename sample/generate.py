# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import concurrent.futures
import os
import sys

# Ensure parent directory is in path for local imports when running as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data_loaders.get_data import get_dataset, get_dataset_loader
from data_loaders.tensors import truebones_batch_collate
from data_loaders.truebones.data.dataset import (
    create_temporal_mask_for_window,
    ensure_joint_name_embeddings,
)
from data_loaders.truebones.truebones_utils.get_opt import get_opt
from data_loaders.truebones.truebones_utils.motion_process import recover_animation_from_motion_np
from diffusion.resample import create_named_schedule_sampler
from motion_lib import BVH
from os.path import join as pjoin
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import (
    create_model_and_diffusion_general_skeleton,
    load_model,
    resolve_t5_out_dim,
)
from utils.parser_util import generate_args
from eval.motion_quality import DistributionMotionQualityScorer


def _move_batch_to_device(batch, device):
    return batch.to(device, non_blocking=True)


def _export_motion(task):
    motion_np, parents_np, offsets, npy_name, joint_names, out_path = task
    out_anim, has_animated_pos = recover_animation_from_motion_np(
        motion_np, parents_np, offsets
    )
    np.save(pjoin(out_path, npy_name), motion_np)
    if out_anim is not None:
        BVH.save(
            pjoin(out_path, npy_name.replace('.npy', '.bvh')),
            out_anim,
            joint_names,
            positions=has_animated_pos,
        )
    return npy_name


def _sample_object_type_batch(
    diffusion,
    model,
    model_kwargs,
    sampling_method,
    sample_shape,
    ddim_eta,
    seed,
    device,
):
    fixseed(seed)
    init_image = None
    noise = torch.randn(sample_shape, device=device)
    common_kwargs = dict(
        model=model,
        shape=sample_shape,
        noise=noise,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        device=device,
        init_image=init_image,
    )

    if sampling_method == 'ddim':
        return diffusion.ddim_sample_loop(
            progress=True,
            eta=ddim_eta,
            **common_kwargs,
        )
    if sampling_method == 'plms':
        return diffusion.plms_sample_loop(
            progress=True,
            **common_kwargs,
        )
    if sampling_method in ('p', 'ddpm'):
        return diffusion.p_sample_loop(
            progress=True,
            skip_timesteps=0,
            dump_steps=None,
            const_noise=False,
            **common_kwargs,
        )
    raise ValueError(f'Unknown sampling_method: {sampling_method}')


def main(args=None, cond_dict=None):
    if args is None:
        args = generate_args()

    fixseed(args.seed)
    opt = get_opt(args.device)
    if cond_dict is None:
        if args.cond_path:
            cond_dict = np.load(args.cond_path, allow_pickle=True).item()
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
        out_path = os.path.join(
            os.path.dirname(args.model_path),
            'samples_{}_{}_seed{}'.format(name, niter, args.seed),
        )
    os.makedirs(out_path, exist_ok=True)

    print('Creating model and diffusion...')
    resolve_t5_out_dim(args, cond_source=actual_cond_file)
    sampling_steps = int(getattr(args, 'sampling_steps', 100))
    sampling_method = str(getattr(args, 'sampling_method', 'ddim')).lower()
    if sampling_steps > 0:
        if sampling_method == 'ddim':
            args.timestep_respacing = f'ddim{sampling_steps}'
        elif sampling_method == 'plms':
            args.timestep_respacing = f'ddim{sampling_steps}'
        else:
            args.timestep_respacing = str(sampling_steps)
    else:
        args.timestep_respacing = ''
    model, diffusion = create_model_and_diffusion_general_skeleton(args)

    print(f'Loading checkpoints from [{args.model_path}]...')
    state_dict = torch.load(args.model_path, map_location='cpu')
    if 'model_avg' in state_dict:
        print('EMA checkpoint detected, loading model_avg weights.')
        state_dict = state_dict['model_avg']
    elif 'model' in state_dict:
        state_dict = state_dict['model']
    load_model(model, state_dict)

    print('Validating precomputed joint-name embeddings from cond.npy...')
    ensure_joint_name_embeddings(
        cond_dict,
        expected_embedding_dim=args.t5_out_dim,
        cond_source=actual_cond_file,
    )
    model.to(dist_util.dev())
    model.eval()
    action_tags = getattr(args, 'action_tags', None) or None

    ddim_eta = float(getattr(args, 'ddim_eta', 0.0))

    # Create thread pool ONCE to minimize overhead across all object_types.
    # Threads still benefit from GIL release inside np.save / np.savetxt (C code),
    # and recover_animation_from_motion_np uses numpy ops that often release the GIL.
    num_workers = 8
    export_pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
    try:
        # Generate samples per object_type sequentially, with one batch per object_type.
        for object_idx, object_type in enumerate(object_types):
            print(f'\n### Sampling object_type: {object_type}')
            print(f'    method={sampling_method} steps={sampling_steps or "full"} batch_size={args.batch_size}')

            # Create condition for batch_size samples of the same object_type
            obj_batch = [object_type] * args.batch_size
            _, model_kwargs = create_condition(
                obj_batch,
                cond_dict,
                n_frames,
                args.temporal_window,
                max_joints=opt.max_joints,
                feature_len=opt.feature_len
            )
            sample = _sample_object_type_batch(
                diffusion=diffusion,
                model=model,
                model_kwargs=model_kwargs,
                sampling_method=sampling_method,
                sample_shape=(args.batch_size, max_joints, model.feature_len, n_frames),
                ddim_eta=ddim_eta,
                seed=args.seed + object_idx,
                device=dist_util.dev(),
            )

            # Pre-compute filenames with a single directory scan
            existing_npy_files = [
                f for f in os.listdir(out_path)
                if f.startswith(object_type) and f.endswith('.npy')
            ]
            base_index = len(existing_npy_files)

            # Collect export tasks (in-process, no pickling needed)
            joint_names = cond_dict[object_type].get(
                'canonical_bvh_joint_names',
                cond_dict[object_type]['joints_names'],
            )
            export_tasks = []
            for sample_idx, motion in enumerate(sample):
                n_joints = model_kwargs['y']['n_joints'][sample_idx].item()
                motion = motion[:n_joints]
                parents = model_kwargs['y']['parents'][sample_idx]
                mean = cond_dict[object_type]['mean'][None, :]
                std = cond_dict[object_type]['std'][None, :]
                motion_np = motion.cpu().permute(2, 0, 1).numpy() * std + mean
                offsets = cond_dict[object_type]['offsets']

                npy_name = f'{object_type}_#{base_index + sample_idx}.npy'
                export_tasks.append((
                    motion_np,
                    parents,  # already np.ndarray, shared in-process
                    offsets,
                    npy_name,
                    joint_names,
                    out_path,
                ))

            # Parallel export using ThreadPoolExecutor.
            # np.save / np.savetxt release the GIL (C-level I/O), so threads
            # can overlap I/O with each other and with the host sampler loop.
            results = list(
                tqdm(export_pool.map(_export_motion, export_tasks),
                      total=len(export_tasks),
                      desc=f'{object_type} export')
            )
            for npy_name in results:
                print(f'    Created motion: {npy_name}')
    finally:
        export_pool.shutdown(wait=True)

    # Evaluate generated motions using DistributionMotionQualityScorer
    if action_tags:
        print('\n### Evaluating motion quality with DistributionMotionQualityScorer...')
        try:
            scorer = DistributionMotionQualityScorer(fps=fps)

            # Group generated motions by object_type
            for object_type in object_types:
                # Find all .npy files for this object type
                npy_files = [
                    f for f in os.listdir(out_path)
                    if f.startswith(f'{object_type}_#') and f.endswith('.npy')
                ]

                if not npy_files:
                    print(f'  {object_type}: no generated motions found')
                    continue

                # Load motions
                motions = []
                for npy_file in npy_files:
                    motion = np.load(pjoin(out_path, npy_file))
                    motions.append(motion)

                # Evaluate
                try:
                    report = scorer.evaluate(
                        motions=motions,
                        object_type=object_type,
                        action_tags=action_tags,
                    )
                    print(f'  {object_type}: {report.overall_score:.3f}')
                except Exception as e:
                    print(f'  {object_type}: evaluation failed - {e}')
        except Exception as e:
            print(f'  Quality evaluation skipped: {e}')
    else:
        print('\n### Skipping motion quality evaluation: no action_tags were provided.')

    return out_path


def create_condition(object_types, cond_dict, n_frames, temporal_window, max_joints, feature_len):
    """Build model_kwargs for a batch of object_types.
    """
    batches = list()
    for object_type in object_types:
        if object_type not in cond_dict:
            available = ', '.join(sorted(cond_dict.keys()))
            raise KeyError(
                f"Unknown object_type '{object_type}'. Available object types in cond file: {available}"
            )
        batch = list()
        parents = cond_dict[object_type]['parents']
        n_joints = len(parents)
        mean = cond_dict[object_type]['mean']
        std = cond_dict[object_type]['std']
        tpos_first_frame = cond_dict[object_type]['tpos_first_frame']
        tpos_first_frame = (tpos_first_frame - mean) / (std + 1e-6)
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
        batch.append(object_type)
        batches.append(batch)

    return truebones_batch_collate(batches)


if __name__ == '__main__':
    main()
