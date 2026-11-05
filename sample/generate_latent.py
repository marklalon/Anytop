"""Generate motion through latent diffusion and decode once with its bound frozen AE."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from data_loaders.truebones.data.dataset import ensure_joint_name_embeddings
from data_loaders.truebones.truebones_utils.cond_schema import load_cond
from data_loaders.truebones.truebones_utils.dataset_sources import resolve_species_key, species_lookup_map
from data_loaders.truebones.truebones_utils.canonical_features import (
    canonical_to_physical_hml, working_to_stored, stored_to_working, mark_canonical_cond_entry)
from data_loaders.truebones.truebones_utils.species_descriptor_table import (
    SPECIES_DESCRIPTOR_TABLE_FILE, load_species_descriptor_table, bind_cond_species_embs)
from data_loaders.truebones.truebones_utils.param_utils import FPS, MAX_SOURCE_FRAMES_MULT
from diffusion.latent_diffusion import create_latent_diffusion
from model.motion_autoencoder import to_working_motion
from sample.conditioning import (
    create_condition, _resolve_action_condition, _resolve_species_code_override,
    _wrap_action_label_cfg, _wrap_reference_cfg)
from sample.output_lengths import (
    resolve_loop_condition, loop_period, resolve_loop_cycles, reference_loop_cycles,
    _resolve_auto_output_lengths, _finalize_output_lengths, _resample_window_to_output,
    resample_reference_to_output)
from sample.reference_motion import _prepare_encoder_reference_bundle, _load_raw_encoder_reference
from sample.generation_runtime import _configure_sampling_args, _resolve_generation_cond_path
from sample.export import _bvh_preview_options, _export_motion
from utils.fixseed import fixseed
from utils.fullbody_ik import DEFAULT_IK_STRETCH_FACTOR
from utils.misc import infer_object_type_from_filename
from utils.latent_util import load_latent_checkpoint, file_sha256, ae_inputs


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model_path', required=True)
    p.add_argument('--ae_checkpoint', help='Optional relocated copy; exact AE hash must match')
    p.add_argument('--cond_path', help='Custom skeleton condition file; a single species is auto-selected')
    p.add_argument('--object_type', help='Target species; optional for single-species --cond_path or a named reference')
    p.add_argument('--action_label', default='')
    p.add_argument('--species_tags', default='')
    p.add_argument('--reference_motion')
    p.add_argument('--reference_object_type')
    p.add_argument('--num_frames', type=int)
    p.add_argument('--batch_size', '--num_samples', dest='batch_size', type=int, default=16,
                   help='Number of generated motions; --num_samples is a legacy alias')
    p.add_argument('--loop', nargs='?', const='on', choices=('auto', 'on', 'off'), default='auto')
    p.add_argument('--loop_cycles', type=int)
    p.add_argument('--action_label_cfg_scale', type=float, default=1.0)
    p.add_argument('--ref_cfg_scale', type=float, default=1.0)
    p.add_argument('--timestep_respacing', default='')
    p.add_argument('--ddim', action='store_true')
    p.add_argument('--sampling_method', choices=('p', 'ddpm', 'ddim'), default='ddpm')
    p.add_argument('--sampling_steps', type=int, default=100,
                   help='Respaced diffusion steps; 0 uses the full checkpoint schedule')
    p.add_argument('--amp_dtype', choices=('fp32',), default='fp32',
                   help='Latent generation currently runs in fp32')
    p.add_argument('--seed', type=int, default=10)
    p.add_argument('--device', default='0', help='GPU index; -1 for CPU. Also accepts cpu/cuda:N')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--fullbody_ik', action='store_true', help='Apply full-body IK to BVH previews')
    p.add_argument('--stretch_factor', type=float, default=DEFAULT_IK_STRETCH_FACTOR,
                   help='Allowed bone-length elasticity for full-body IK (0.1 = +/-10%%; 0 = rigid)')
    return p


def _resolve_target_type(args, cond):
    """Match motion-space generation's target selection rules."""
    if args.object_type:
        return resolve_species_key(cond, args.object_type)
    if args.cond_path and len(cond) == 1:
        target = next(iter(cond))
        print(f'[generate] --object_type omitted; the cond file {args.cond_path} '
              f'contains a single species, using it: {target}')
        return target
    if args.reference_motion:
        target = infer_object_type_from_filename(args.reference_motion, valid_types=species_lookup_map(cond))
        if target is not None:
            return target
        raise ValueError('Cannot infer object_type from reference motion filename; pass --object_type')
    raise ValueError('Pass --object_type, --reference_motion, or a single-species --cond_path')


def _resolve_cond_path(args):
    """Use exactly the same condition path policy as motion-space generation."""
    return Path(_resolve_generation_cond_path(args))


@torch.no_grad()
def main(argv=None, runtime=None):
    args = argv if isinstance(argv, argparse.Namespace) else parser().parse_args(argv)
    if args.batch_size < 1:
        raise ValueError('batch_size must be positive')
    if args.reference_motion and args.action_label:
        raise ValueError('Encoded reference and action label are mutually exclusive')
    if not args.reference_motion and args.ref_cfg_scale != 1:
        raise ValueError('ref_cfg_scale requires reference_motion')
    if args.reference_motion and args.action_label_cfg_scale != 1:
        raise ValueError('Action and reference CFG cannot be combined')
    fixseed(args.seed)
    torch.set_num_threads(4)
    device_arg = str(args.device)
    if device_arg.lstrip('-').isdigit():
        index = int(device_arg)
        device_arg = f'cuda:{index}' if index >= 0 and torch.cuda.is_available() else 'cpu'
    device = torch.device(device_arg)
    model_path = Path(args.model_path).resolve()
    # A battery may reuse the frozen weights across tasks, including tasks with
    # different conditions or sampling schedules. Standalone generation is unchanged.
    key = (str(model_path), str(device), args.ae_checkpoint)
    if runtime is None:
        runtime = {}
    if key not in runtime:
        runtime[key] = load_latent_checkpoint(model_path, device, args.ae_checkpoint)
    model, ae, norm, checkpoint = runtime[key]
    cond_path = _resolve_cond_path(args)
    # Explicit cond overrides supply a new inference skeleton, just as in the
    # motion-space generator. Only the implicit training cond is hash-bound.
    if not args.cond_path and file_sha256(cond_path) != checkpoint['cond_sha256']:
        raise ValueError('Skeleton condition data hash differs from training')
    cond = load_cond(str(cond_path))
    for entry in cond.values():
        mark_canonical_cond_entry(entry)
    ensure_joint_name_embeddings(cond, expected_embedding_dim=model.t5_out_dim, cond_source=str(cond_path))
    table = None
    if model.species_cond:
        table_path = model_path.parent / SPECIES_DESCRIPTOR_TABLE_FILE
        if file_sha256(table_path) != checkpoint['species_sha256']:
            raise ValueError('Checkpoint species descriptor table hash mismatch')
        table = load_species_descriptor_table(table_path)
        bind_cond_species_embs(cond, table, str(model_path))
    object_type = _resolve_target_type(args, cond)
    action = _resolve_action_condition(args, model)
    override = _resolve_species_code_override(args, model, cond, object_type, table)
    features = source_cond = None
    if args.reference_motion:
        if not model.ref_encoder:
            raise ValueError('Checkpoint has no reference encoder')
        reference_path = Path(args.reference_motion)
        if reference_path.suffix.lower() in ('.fbx', '.glb', '.gltf'):
            features, source_cond = _load_raw_encoder_reference(str(reference_path), str(cond_path), cond)
        elif reference_path.suffix.lower() == '.npy':
            source = (resolve_species_key(cond, args.reference_object_type) if args.reference_object_type else
                      infer_object_type_from_filename(str(reference_path), valid_types=species_lookup_map(cond)))
            if source is None:
                raise ValueError('Cannot infer reference skeleton; pass --reference_object_type')
            source_cond = cond[source]
            features = np.load(reference_path)
        else:
            raise ValueError('Reference must be preprocessed .npy or raw .fbx/.glb/.gltf')
    # Read the stored terminal-velocity verdict before working-layout conversion
    # clears those channels. Raw preprocessing returns stored features too.
    loop = resolve_loop_condition(
        args.loop, action, reference_features=features,
        translation_root_index=int(source_cond.get('translation_root_index', 0)) if source_cond else 0)
    if features is not None:
        features = stored_to_working(features)
    period = loop_period(cond, object_type, action, min_length=20, internal_num_frames=60)
    if features is not None:
        cycles = reference_loop_cycles(args.loop_cycles, loop=loop)
        frames = args.num_frames
        if frames is None:
            frames = int(np.clip(len(features), 20, MAX_SOURCE_FRAMES_MULT * 60))
            if frames != len(features):
                print(f'  Reference R={len(features)} frames clamped to {frames} '
                      f'(variable-length window [20, {MAX_SOURCE_FRAMES_MULT * 60}])')
        lengths = _finalize_output_lengths(frames, 20, 60)
    else:
        cycles, frames = resolve_loop_cycles(args.loop_cycles, args.num_frames, loop=loop,
                                             action_label=action['action_label'] if action else None,
                                             period=period)
        lengths = (_finalize_output_lengths(frames, 20, 60) if frames is not None else
                   _resolve_auto_output_lengths(cond, object_type, action, min_length=20,
                                                internal_num_frames=60, default_frames=60, loop=loop))
    output_frames, _, speed = lengths
    _, kwargs = create_condition([object_type] * args.batch_size, cond, 60,
                                 max_joints=len(cond[object_type]['parents']), feature_len=12,
                                 loop=loop, loop_cycles=cycles, action_condition=action,
                                 species_code_override=override)
    y = dict(kwargs['y'])
    y.pop('ref_frame_mask', None)
    if features is not None:
        features, _, _ = resample_reference_to_output(features, output_frames)
        reference = _prepare_encoder_reference_bundle(
            features, source_cond, batch_size=args.batch_size, n_frames=60, visible_frames=60)
        reference.pop('ref_frame_mask', None)
        y.update(reference)
    y = {k: v.to(device) if torch.is_tensor(v) else v for k, v in y.items()}
    y['resample_speed_cond'] = torch.full((args.batch_size,), speed, device=device)
    # These true Decoder conditions remain unchanged by either CFG branch.
    skeleton, time = ae_inputs(y)
    schedule = checkpoint['diffusion_config']
    args.diffusion_steps = schedule['steps']
    args.noise_schedule = schedule['noise_schedule']
    args.noise_schedule_shift = schedule['shift']
    args.action_label_cfg_drop_prob = model.action_label_cfg_drop_prob
    guided = _wrap_reference_cfg(model, args, action) if features is not None else _wrap_action_label_cfg(model, args, action)
    if args.ddim:
        args.sampling_method = 'ddim'
    sampling_method, _ = _configure_sampling_args(args)
    diffusion = create_latent_diffusion(**schedule, respacing=args.timestep_respacing)
    latent = diffusion.sample(guided, (args.batch_size, 60, ae.slots, ae.latent_dim), y, device,
                              ddim=sampling_method == 'ddim', progress=True)
    decoded = ae.decode(norm.denormalize(latent), skeleton, loop_metadata=time)
    if not torch.isfinite(decoded).all():
        raise FloatingPointError('Nonfinite decoded motion')
    physical = canonical_to_physical_hml(to_working_motion(decoded), cond[object_type])
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = list(cond[object_type].get('canonical_bvh_joint_names', cond[object_type]['joints_names']))
    file_prefix = object_type.replace('/', '__').replace('\\', '__')
    for i, motion in enumerate(physical):
        value = motion.permute(2, 0, 1).cpu().numpy()
        value = _resample_window_to_output(value, output_frames, 60, loop)
        _export_motion((working_to_stored(value, is_loop=loop), cond[object_type],
                        f'{file_prefix}_latent_{i:03d}.npy', names, str(out), FPS,
                        _bvh_preview_options(args)))
    (out / 'generation.json').write_text(json.dumps(dict(
        model=str(model_path), step=checkpoint['step'], seed=args.seed,
        object_type=object_type, action_label=args.action_label, reference_motion=args.reference_motion,
        cond_path=str(cond_path.resolve()),
        frames=output_frames, loop=loop, cycles=cycles, speed=speed), indent=2), encoding='utf-8')
    print(f'Generated {args.batch_size} motion(s): {out.resolve()}')


if __name__ == '__main__':
    main()
