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
from tqdm import tqdm

from data_loaders.tensors import truebones_batch_collate
from data_loaders.truebones.data.dataset import (
    create_temporal_mask_for_window,
    ensure_joint_name_embeddings,
)
from data_loaders.truebones.truebones_utils.get_opt import get_opt
from data_loaders.truebones.truebones_utils.motion_process import (
    recover_bvh_export_animation_from_motion_np,
)
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
from utils.misc import infer_object_type_from_filename


def _export_motion(task):
    motion_np, parents_np, offsets, npy_name, joint_names, out_path, fps = task
    out_anim, joint_names, has_animated_pos = recover_bvh_export_animation_from_motion_np(
        motion_np,
        parents_np,
        offsets,
        joint_names,
        allow_infer=True,
    )
    np.save(pjoin(out_path, npy_name), motion_np)
    if out_anim is not None:
        BVH.save(
            pjoin(out_path, npy_name.replace('.npy', '.bvh')),
            out_anim,
            joint_names,
            frametime=1.0 / fps,
            positions=has_animated_pos,
        )
    return npy_name


def _sample_batch(
    diffusion,
    model,
    model_kwargs,
    sampling_method,
    sample_shape,
    ddim_eta,
    seed,
    device,
    reference_motion=None,
    skip_timesteps=0,
):
    fixseed(seed)

    # If reference motion is provided, use it as init_image with skip_timesteps.
    # skip_timesteps: how many of the noisiest timesteps to skip.
    # Higher = start from less noisy state = more faithful to reference.
    # e.g., skip_timesteps=80 with 100 total steps means:
    #   - reference is noised to t=19 (light noise)
    #   - only 20 denoising steps performed
    if reference_motion is not None and skip_timesteps > 0:
        init_image = reference_motion.to(device, non_blocking=True)
        noise = torch.randn(sample_shape, device=device)
    else:
        init_image = None
        noise = torch.randn(sample_shape, device=device)
        skip_timesteps = 0  # No reference => full denoising from pure noise

    common_kwargs = dict(
        model=model,
        shape=sample_shape,
        noise=noise,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        device=device,
        init_image=init_image,
        skip_timesteps=skip_timesteps,
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
            skip_timesteps=skip_timesteps,
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
    object_type = args.object_type
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

    ddim_eta = float(getattr(args, 'ddim_eta', 0.0))
    reference_motion_path = getattr(args, 'reference_motion', None)
    skip_timesteps = int(getattr(args, 'skip_timesteps', 80))

    # If reference motion is provided, infer object_type from filename
    if reference_motion_path:
        inferred_type = infer_object_type_from_filename(
            reference_motion_path, valid_types=cond_dict.keys()
        )

        if inferred_type is None:
            available = ', '.join(sorted(cond_dict.keys()))
            print(f"ERROR: Cannot infer object_type from reference motion filename: {reference_motion_path}")
            print(f"Available object types: {available}")
            print("Please rename the file to follow the naming convention (e.g., 'ObjectType___action_id.npy') "
                  "or specify --object_type explicitly.")
            sys.exit(1)

        # Validate against explicitly specified object_type
        if object_type != inferred_type:
            print(f"ERROR: Reference motion infers object_type '{inferred_type}' "
                  f"but --object_type specifies '{object_type}'.")
            print("Cross-species reference is not supported. "
                  "Use a reference motion matching the target skeleton.")
            sys.exit(1)

        print(f"Reference motion inferred object_type: {inferred_type}")

    # Create thread pool for export.
    # Threads still benefit from GIL release inside np.save / np.savetxt (C code),
    # and recover_animation_from_motion_np uses numpy ops that often release the GIL.
    num_workers = 8
    export_pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
    try:
        print(f'\n### Sampling object_type: {object_type}')
        print(f'    method={sampling_method} steps={sampling_steps or "full"} batch_size={args.batch_size}')

        # Prepare reference motion (normalize + reshape)
        ref_motion = None
        effective_n_frames = n_frames  # May be overridden by reference motion

        if reference_motion_path:
            ref_raw = np.load(reference_motion_path).astype(np.float32)
            # ref_raw shape: [frames, joints, features]
            ref_frames = ref_raw.shape[0]
            ref_joints = ref_raw.shape[1]

            # Use reference motion's frame count (truncated to model's max frames)
            effective_n_frames = min(ref_frames, n_frames)
            if effective_n_frames != n_frames:
                print(f'  Reference motion overrides frame count: {n_frames} -> {effective_n_frames}')

            # Truncate reference if longer than max output
            if ref_frames > effective_n_frames:
                ref_raw = ref_raw[:effective_n_frames]

            # Normalize using the same object_type's stats (same-skeleton only)
            obj_mean = cond_dict[object_type]['mean']
            obj_std = np.asarray(cond_dict[object_type]['std'], dtype=np.float32) + 1e-6

            # Normalize using same stats as training data
            ref_norm = np.nan_to_num((ref_raw - obj_mean[None, :]) / obj_std[None, :], copy=True).astype(np.float32)

            # Pad joints to max_joints if needed
            if ref_joints < max_joints:
                pad = np.zeros((effective_n_frames, max_joints - ref_joints, ref_norm.shape[2]), dtype=np.float32)
                ref_norm = np.concatenate([ref_norm, pad], axis=1)

            # Convert to model input shape: [batch, joints, features, frames]
            ref_tensor = torch.from_numpy(ref_norm).permute(1, 2, 0)  # [joints, features, frames]
            # Ensure feature dim matches model expectation
            ref_feat = ref_tensor.shape[1]
            target_feat = model.feature_len
            if ref_feat < target_feat:
                pad = torch.zeros((max_joints, target_feat - ref_feat, effective_n_frames), dtype=torch.float32)
                ref_tensor = torch.cat([ref_tensor, pad], dim=1)
            elif ref_feat > target_feat:
                ref_tensor = ref_tensor[:, :target_feat, :]
            ref_tensor = ref_tensor.unsqueeze(0).expand(args.batch_size, -1, -1, -1)
            ref_motion = ref_tensor
            print(f'  Reference motion loaded: {reference_motion_path}')
            print(f'    Original: [{ref_frames} frames, {ref_joints} joints] -> Target: [{effective_n_frames} frames, {max_joints} joints]')
            print(f'    skip_timesteps: {skip_timesteps} (higher = more faithful to reference)')

        # Create condition with effective frame count
        obj_batch = [object_type] * args.batch_size
        _, model_kwargs = create_condition(
            obj_batch,
            cond_dict,
            effective_n_frames,
            args.temporal_window,
            max_joints=opt.max_joints,
            feature_len=opt.feature_len
        )
        sample = _sample_batch(
            diffusion=diffusion,
            model=model,
            model_kwargs=model_kwargs,
            sampling_method=sampling_method,
            sample_shape=(args.batch_size, max_joints, model.feature_len, effective_n_frames),
            ddim_eta=ddim_eta,
            seed=args.seed,
            device=dist_util.dev(),
            reference_motion=ref_motion,
            skip_timesteps=skip_timesteps,
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
                fps,
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
