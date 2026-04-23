# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import os
import sys

# Ensure parent directory is in path for local imports when running as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

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


def _move_batch_to_device(batch, device):
    return batch.to(device, non_blocking=True)


def _move_cond_to_device(cond, device):
    return {
        'y': {
            key: val.to(device, non_blocking=True) if torch.is_tensor(val) else val
            for key, val in cond['y'].items()
        }
    }


def _slice_cond_batch(cond, count):
    sliced = {'y': {}}
    for key, val in cond['y'].items():
        if torch.is_tensor(val):
            sliced['y'][key] = val[:count]
        elif isinstance(val, list):
            sliced['y'][key] = val[:count]
        else:
            sliced['y'][key] = val
    return sliced


def _with_train_step(cond, train_step):
    updated = {'y': dict(cond['y'])}
    updated['train_step'] = int(train_step)
    return updated


def _compute_eval_losses(model, diffusion, batch, cond, device):
    schedule_sampler = create_named_schedule_sampler('uniform', diffusion)
    t, weights = schedule_sampler.sample(batch.shape[0], device)
    with torch.no_grad():
        losses = diffusion.training_losses(
            model,
            batch,
            t,
            model_kwargs=_with_train_step(cond, 0),
        )

    reduced = {}
    for key, value in losses.items():
        if not torch.is_tensor(value):
            continue
        reduced[key] = float((value.detach() * weights).mean().item())
    return reduced


def build_reference_init_batch(args, n_frames, object_types, action_category, device, repetition_index):
    eval_split = str(getattr(args, 'eval_split', 'val'))
    action_tags = str(action_category or '').strip().lower()
    motion_cache_size = getattr(args, 'motion_cache_size', 0)
    dataset_cache = {}
    object_occurrences = {}
    selected_motion_names = []
    batch_motions = []

    for object_type in object_types:
        dataset = dataset_cache.get(object_type)
        if dataset is None:
            dataset = get_dataset(
                num_frames=n_frames,
                split=eval_split,
                temporal_window=args.temporal_window,
                balanced=False,
                objects_subset=object_type,
                sample_limit=0,
                action_tags=action_tags,
                motion_cache_size=motion_cache_size,
            )
            dataset_cache[object_type] = dataset

        motion_dataset = dataset.motion_dataset

        if len(motion_dataset.name_list) == 0:
            raise RuntimeError(
                f"No reference motions found in split='{eval_split}' for "
                f"object_type='{object_type}' action_category='{action_tags or 'any'}'."
            )

        occurrence_index = object_occurrences.get(object_type, 0)
        requested_index = repetition_index + occurrence_index
        motion_name = motion_dataset.name_list[requested_index % len(motion_dataset.name_list)]
        sample = motion_dataset.prepare_sample_by_name(
            motion_name,
            target_num_frames=n_frames,
            crop_start=0,
            loop_offset=0,
        )
        motion, _ = truebones_batch_collate([sample])
        batch_motions.append(motion[0])
        selected_motion_names.append(motion_name)
        object_occurrences[object_type] = occurrence_index + 1

    init_image = torch.stack(batch_motions, dim=0).to(device=device, non_blocking=True)
    return init_image, selected_motion_names


def compute_reference_eval_losses(model, diffusion, args, n_frames, object_types, action_category, device):
    eval_split = str(getattr(args, 'eval_split', 'val'))
    eval_batch_size = max(1, int(getattr(args, 'eval_batch_size', 32)))
    action_tags = str(action_category or '').strip().lower()
    per_object_target = max(1, int(getattr(args, 'num_repetitions', 1)))
    totals = {}
    seen_samples = 0
    used_motion_names = {}
    per_object_totals = {}
    per_object_seen = {}

    for object_type in list(dict.fromkeys(object_types)):
        try:
            loader = get_dataset_loader(
                batch_size=min(eval_batch_size, per_object_target),
                num_frames=n_frames,
                split=eval_split,
                temporal_window=args.temporal_window,
                balanced=False,
                objects_subset=object_type,
                sample_limit=0,
                shuffle=False,
                drop_last=False,
                action_tags=action_tags,
                motion_cache_size=getattr(args, 'motion_cache_size', 0),
                main_process_prefetch_batches=getattr(args, 'main_process_prefetch_batches', 0),
            )
        except Exception as exc:
            print(f"[WARN] Reference eval skipped for {object_type}: {exc}")
            continue

        collected = 0
        used_motion_names[object_type] = []
        per_object_totals[object_type] = {}
        per_object_seen[object_type] = 0
        for motion, cond in loader:
            remaining = per_object_target - collected
            if remaining <= 0:
                break
            if motion.shape[0] > remaining:
                motion = motion[:remaining]
                cond = _slice_cond_batch(cond, remaining)

            used_motion_names[object_type].extend(cond['y'].get('motion_name', []))
            motion = _move_batch_to_device(motion, device)
            cond = _move_cond_to_device(cond, device)
            batch_losses = _compute_eval_losses(model, diffusion, motion, cond, device)
            batch_size = motion.shape[0]
            for key, value in batch_losses.items():
                totals[key] = totals.get(key, 0.0) + (value * batch_size)
                per_object_totals[object_type][key] = per_object_totals[object_type].get(key, 0.0) + (value * batch_size)
            collected += batch_size
            seen_samples += batch_size
            per_object_seen[object_type] += batch_size

        if collected == 0:
            print(
                f"[WARN] No reference motions found in split='{eval_split}' for "
                f"object_type='{object_type}' action_category='{action_tags or 'any'}'."
            )

    if seen_samples == 0:
        return None, None, used_motion_names, seen_samples

    averaged = {key: value / seen_samples for key, value in totals.items()}
    per_object_averaged = {}
    for object_type, obj_totals in per_object_totals.items():
        obj_seen = per_object_seen.get(object_type, 0)
        if obj_seen <= 0:
            continue
        per_object_averaged[object_type] = {
            key: value / obj_seen for key, value in obj_totals.items()
        }
    return averaged, per_object_averaged, used_motion_names, seen_samples


class _CFGWrapper(nn.Module):
    """Wraps a model for classifier-free guidance at inference time.

    Performs two forward passes per denoising step (conditioned and
    unconditioned) and returns the linearly extrapolated prediction:
        out = out_uncond + guidance_scale * (out_cond - out_uncond)

    The unconditioned pass is obtained by zeroing the action-tag condition
    (empty action_tags list -> zero embedding from ActionTagConditioner).
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
        y = model_kwargs.get('y', {})
        y_uncond = {**y, 'action_tags': [[] for _ in range(bs)], 'action_category': [None] * bs}
        out_uncond = self.model(x, timesteps, **{**model_kwargs, 'y': y_uncond})
        return out_uncond + self.guidance_scale * (out_cond - out_uncond)


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
    args.batch_size = len(object_types)

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
    action_guidance_scale = float(getattr(args, 'action_guidance_scale', 1.0))
    if action_guidance_scale == 0.0:
        print('Action CFG disabled (scale=0): unconditional generation')
    elif action_guidance_scale != 1.0:
        print(f'Action CFG enabled: action_guidance_scale={action_guidance_scale}')
        model = _CFGWrapper(model, action_guidance_scale)
    action_category = getattr(args, 'action_category', None) or None
    _, model_kwargs = create_condition(
        object_types,
        cond_dict,
        n_frames,
        args.temporal_window,
        max_joints=opt.max_joints,
        feature_len=opt.feature_len,
        action_category=action_category,
    )

    ddim_eta = float(getattr(args, 'ddim_eta', 0.0))
    use_reference_noise = bool(getattr(args, 'use_reference_noise', False))

    for rep_i in range(args.num_repetitions):
        fixseed(args.seed + rep_i)
        init_image = None
        if use_reference_noise:
            init_image, reference_motion_names = build_reference_init_batch(
                args,
                n_frames,
                object_types,
                action_category,
                dist_util.dev(),
                rep_i,
            )
            print('### Using reference motions for init noise:')
            for object_type, motion_name in zip(object_types, reference_motion_names):
                print(f'  {object_type}: {motion_name}')
        print(f'### Sampling [repetitions #{rep_i}] method={sampling_method} steps={sampling_steps or "full"}')
        if sampling_method == 'ddim':
            sample = diffusion.ddim_sample_loop(
                model,
                (args.batch_size, max_joints, model.feature_len, n_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True,
                eta=ddim_eta,
                init_image=init_image,
            )
        elif sampling_method == 'plms':
            sample = diffusion.plms_sample_loop(
                model,
                (args.batch_size, max_joints, model.feature_len, n_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True,
                init_image=init_image,
            )
        elif sampling_method in ('p', 'ddpm'):
            sample = diffusion.p_sample_loop(
                model,
                (args.batch_size, max_joints, model.feature_len, n_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,
                init_image=init_image,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
        else:
            raise ValueError(f'Unknown sampling_method: {sampling_method}')

        bs, max_joints, n_feats, n_frames = sample.shape
        for i, motion in enumerate(sample):
            n_joints = model_kwargs['y']['n_joints'][i].item()
            motion = motion[:n_joints]
            object_type = model_kwargs['y']['object_type'][i]
            parents = model_kwargs['y']['parents'][i]
            mean = cond_dict[object_type]['mean'][None, :]
            std = cond_dict[object_type]['std'][None, :]
            motion = motion.cpu().permute(2, 0, 1).numpy() * std + mean
            offsets = cond_dict[object_type]['offsets']
            out_anim, has_animated_pos = recover_animation_from_motion_np(motion, parents, offsets)
            name_pref = '%s_rep_%d' % (object_type, rep_i)
            existing_npy_files = [
                filename for filename in os.listdir(out_path)
                if filename.startswith(name_pref) and filename.endswith('.npy')
            ]
            npy_name = name_pref + '_#%d.npy' % (len(existing_npy_files))
            bvh_name = name_pref + '_#%d.bvh' % (len(existing_npy_files))
            np.save(pjoin(out_path, npy_name), motion)
            if out_anim is not None:
                BVH.save(
                    pjoin(out_path, bvh_name),
                    out_anim,
                    cond_dict[object_type].get(
                        'canonical_bvh_joint_names',
                        cond_dict[object_type]['joints_names'],
                    ),
                    positions=has_animated_pos,
                )
            print('repetition #' + str(rep_i) + ' ,created motion: ' + npy_name)

    if use_reference_noise:
        print('\n### Computing reference validation losses from original motions...')
        eval_model = model.model if isinstance(model, _CFGWrapper) else model
        losses, per_object_losses, used_motion_names, seen_samples = compute_reference_eval_losses(
            eval_model,
            diffusion,
            args,
            n_frames,
            object_types,
            action_category,
            dist_util.dev(),
        )
        if losses is None:
            print('### Reference Validation Losses: skipped (no matching original motions found)')
        else:
            print(f'### Reference Validation Losses ({seen_samples} original motions):')
            for object_type, motion_names in used_motion_names.items():
                if motion_names:
                    print(f"  {object_type}: {', '.join(motion_names)}")
            if per_object_losses:
                print('### Reference Validation Losses By Object Type:')
                for object_type, metrics in per_object_losses.items():
                    metrics_text = ', '.join(f"{k}={v:.5f}" for k, v in metrics.items())
                    print(f"  {object_type}: {metrics_text}")

    return out_path


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
        tag = str(action_category).strip().lower() if action_category else None
        motion_metadata = {
            'action_category': tag,
            'action_tags': [tag] if tag else [],
        }
        batch.append(motion_metadata)
        batch.append(object_type)
        batches.append(batch)

    return truebones_batch_collate(batches)


if __name__ == '__main__':
    main()
