"""Fixed-window self-reconstruction and latent bypass checks (plan section 7.A)."""
import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from tqdm import tqdm

from data_loaders.get_data import build_dataset_loader, get_dataset
from data_loaders.joint_buckets import resolve_joint_buckets
from data_loaders.tensors import truebones_batch_collate
from data_loaders.truebones.truebones_utils.canonical_features import (
    CANONICAL_FEATURE_SPACE, canonical_to_physical_hml, working_to_stored,
)
from data_loaders.truebones.truebones_utils.param_utils import FPS, JOINT_BUCKETS
from diffusion.losses import geodesic_distance
from diffusion.logger import HumanOutputFormat
from model.motion_autoencoder import (
    DECODER_CONTENT_VERSION, MotionAutoencoder,
    diagonal_gaussian_kl, motion_masks, to_working_motion,
)
from train.motion_reconstruction_losses import LOSS_WEIGHTS, MotionGeometryLosses, MotionReconstructionLosses
from utils.fixseed import fixseed
from utils.npy_restore import write_feature_bvh
from torch.utils.tensorboard import SummaryWriter


SKELETON_KEYS = ('n_joints', 'joints_names_embs', 'rest_geom', 'joint_struct',
                 'graph_dist', 'joints_relations')
TIME_KEYS = ('is_loop', 'loop_tile_count', 'resample_speed_cond')


AUGMENTATION_DEFAULTS = dict(motion_speed_aug=1.0, loop_tile_single_prob=1.0,
                             loop_cond_prob=1.0, loop_cond_prob_free=None,
                             loop_cond_prob_anchored=None, joint_name_drop_prob=0.0,
                             topology_aug_prob=0.0, bone_length_aug_prob=0.0,
                             rest_repose_prob=0.0, rest_variant_prob=0.0,
                             rest_geom_drop_prob=0.0, rest_geom_joint_drop_prob=0.0)


MODEL_REGULARIZATION_DEFAULTS = dict(dropout_prob=0.0, mirror_twin_drop_prob=0.0, lateral_drop_prob=0.0)

class AutoencoderBatchTransform:
    """Picklable CPU transform, run inside the shared worker/prefetch pipeline."""

    def __init__(self, joint_name_drop_prob=0.0, rest_geom_drop_prob=0.0,
                 rest_geom_joint_drop_prob=0.0):
        self.joint_name_drop_prob = joint_name_drop_prob
        self.rest_geom_drop_prob = rest_geom_drop_prob
        self.rest_geom_joint_drop_prob = rest_geom_joint_drop_prob

    def __call__(self, batch):
        motion, conditions = batch
        y = {k: v for k, v in conditions['y'].items() if not k.startswith('ref_')}
        # Draw once on CPU with the loader RNG; E and D share these names.
        # Do not mutate cached names or the fixed clean evaluation bundle.
        if self.joint_name_drop_prob > 0:
            names_emb = y['joints_names_embs']
            drop = torch.from_numpy(np.random.random(names_emb.shape[:2]) < self.joint_name_drop_prob)
            y['joints_names_embs'] = names_emb.masked_fill(drop[..., None], 0)
        # Hide all geometry channels, including the present flag. Keep loss
        # conversion context intact; E and D share the same corrupted condition.
        if self.rest_geom_drop_prob > 0 or self.rest_geom_joint_drop_prob > 0:
            geometry = y['rest_geom']
            sample_drop = np.random.random((geometry.shape[0], 1)) < self.rest_geom_drop_prob
            joint_drop = np.random.random(geometry.shape[:2]) < self.rest_geom_joint_drop_prob
            drop = torch.from_numpy(sample_drop | joint_drop)
            y['rest_geom'] = geometry.masked_fill(drop[..., None], 0)
        return motion[:, :, :9].contiguous(), {'y': y}


def create_training_data_loader(args, dataset):
    for key in AUGMENTATION_DEFAULTS:
        setattr(dataset.motion_dataset.opt, key, getattr(args, key))
    return build_dataset_loader(
        dataset, batch_size=args.batch_size, drop_last=True,
        num_workers=args.num_workers,
        main_process_prefetch_batches=args.main_process_prefetch_batches,
        joint_buckets=args.joint_buckets or JOINT_BUCKETS,
        batch_transform=AutoencoderBatchTransform(
            args.joint_name_drop_prob, args.rest_geom_drop_prob, args.rest_geom_joint_drop_prob),
    )


def training_batches(loader):
    # A small bucket may produce no full batch in one pass. The shared sampler
    # carries its remainder forward; do not treat such a pass as an empty dataset.
    while True:
        yield from loader


def training_batch_to_device(batch, device):
    target, conditions = batch
    return target.to(device, non_blocking=True), {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in conditions['y'].items()
    }


def select_batch(x, y, indices):
    device_indices = indices.to(x.device)
    selected = {}
    for key, value in y.items():
        if torch.is_tensor(value):
            selected[key] = value.index_select(0, device_indices)
        elif isinstance(value, list):
            selected[key] = [value[i] for i in indices.tolist()]
    return x.index_select(0, device_indices), selected


def model_inputs(x, y):
    skeleton = {k: y[k] for k in SKELETON_KEYS}
    frame = torch.arange(x.shape[-1], device=x.device)[None] < y['lengths'][:, None]
    return skeleton, frame, {k: y[k] for k in TIME_KEYS}


def reconstruction_mse(prediction, target, skeleton, frame):
    joint, frame = motion_masks(target, skeleton, frame)
    valid = joint[:, :, None, None] & frame[:, None, None, :]
    error = (prediction.float() - target.float()).square().masked_fill(~valid, 0)
    return error.sum((1, 2, 3)) / (valid.sum((1, 2, 3)) * target.shape[2])


def scaled_optimizer_step(loss, model, optimizer, scaler):
    """Unscale before clipping; report whether FP16 overflow skipped the update."""
    previous_scale = scaler.get_scale()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0, error_if_nonfinite=not scaler.is_enabled())
    scaler.step(optimizer)
    scaler.update()
    return grad_norm, scaler.get_scale() >= previous_scale


@torch.no_grad()
def evaluate(model, x, y, batch_size):
    model.eval()
    geometry = MotionGeometryLosses()
    # Encode separately, then swap whole motion latents while keeping each target skeleton fixed.
    latents = []
    for indices in torch.arange(len(x)).split(batch_size):
        target, cond = select_batch(x, y, indices)
        skeleton, frame, loop = model_inputs(target, cond)
        latents.append(model.encode(target, skeleton, frame, loop))
    z = torch.cat(latents)
    shuffled = z.roll(1, dims=0)
    records, predictions = [], []
    for indices in torch.arange(len(x)).split(batch_size):
        target, cond = select_batch(x, y, indices)
        skeleton, frame, loop = model_inputs(target, cond)
        latent = z[indices.to(z.device)]
        pred = model.decode(latent, skeleton, frame, loop)
        zero = model.decode(torch.zeros_like(latent), skeleton, frame, loop)
        swapped = model.decode(shuffled[indices.to(z.device)], skeleton, frame, loop)
        errors = {name: reconstruction_mse(value, target, skeleton, frame)
                  for name, value in (('mse', pred), ('zero_z_mse', zero), ('shuffled_z_mse', swapped))}
        errors['position_mse'] = reconstruction_mse(pred[:, :, :3], target[:, :, :3], skeleton, frame)
        errors['rotation6d_mse'] = reconstruction_mse(pred[:, :, 3:9], target[:, :, 3:9], skeleton, frame)
        errors['root_pos_mse'] = geometry.root_position_loss(target, pred, cond)
        errors['root_dpos_error'] = geometry.root_position_step_loss(target, pred, cond, channels=(0, 1, 2))
        errors['root_acc_error'] = geometry.root_acceleration_loss(target, pred, cond, channels=(0, 1, 2))
        physical_gt, rot_gt = canonical_to_physical_hml(to_working_motion(target), cond,
                                                       return_canonical_rotations=True)
        physical_pred, rot_pred = canonical_to_physical_hml(to_working_motion(pred), cond,
                                                           return_canonical_rotations=True)
        joint, _ = motion_masks(target, skeleton, frame)
        spat_mask = joint[:, None, None, :]
        # Per-window reductions keep these metrics independent of evaluation batch size.
        for metric, method in (('bone_length_error', geometry.bone_length_loss),
                               ('bone_vector_error', geometry.bone_vector_consistency_loss)):
            errors[metric] = torch.stack([
                method(physical_pred[b:b + 1], physical_gt[b:b + 1], spat_mask[b:b + 1],
                       select_batch(target, cond, torch.tensor([b]))[1])
                for b in range(len(target))
            ])
        errors['dpos_error'] = geometry.position_step_loss(target, pred, spat_mask, cond['n_joints'], cond)
        errors['acc_error'] = geometry.position_acceleration_loss(target, pred, spat_mask, cond['n_joints'], cond)
        errors['drot_error'] = geometry.rotation_step_loss(
            physical_pred, physical_gt, spat_mask, cond['lengths'], cond,
            rotations=(rot_gt, rot_pred))
        live = frame[:, :, None] & joint[:, None]
        angle = geodesic_distance(rot_pred, rot_gt).squeeze(-1)
        errors['rotation_error_deg'] = (angle * live).sum((1, 2)) / live.sum((1, 2)) * (180 / torch.pi)
        # RMS angular step is a simple dynamic diagnostic, not an acceptance threshold.
        step_live = live[:, 1:] & live[:, :-1]
        rms = []
        for rotations in (rot_gt, rot_pred):
            angle = geodesic_distance(rotations[:, 1:], rotations[:, :-1]).squeeze(-1)
            rms.append(((angle.square() * step_live).sum((1, 2))
                        / step_live.sum((1, 2)).clamp_min(1)).sqrt())
        for row, index in enumerate(indices.tolist()):
            record = {'motion_name': cond['motion_name'][row], 'object_type': cond['object_type'][row],
                      **{key: float(value[row]) for key, value in errors.items()}}
            record['angular_step_rms_gt'] = float(rms[0][row])
            record['angular_step_rms_reconstruction'] = float(rms[1][row])
            record['angular_step_rms_ratio'] = float(rms[1][row] / rms[0][row]) if rms[0][row] > 1e-4 else None
            records.append(record)
            joints = int(cond['n_joints'][row])
            predictions.append((physical_gt[row, :joints].permute(2, 0, 1).cpu().numpy(),
                                physical_pred[row, :joints].permute(2, 0, 1).cpu().numpy()))
    summary = {key: float(np.mean([r[key] for r in records])) for key in errors}
    summary['sample_count'] = len(records)
    summary['shuffle_available'] = len(x) > 1
    summary['zero_z_mse_increase'] = summary['zero_z_mse'] - summary['mse']
    summary['shuffled_z_mse_increase'] = summary['shuffled_z_mse'] - summary['mse']
    return {'summary': summary, 'samples': records}, predictions


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cond_path', default='dataset/cond_all.npy')
    p.add_argument('--sample_filter', default='configs/sample_filters/tiny.json')
    p.add_argument('--save_dir', required=True)
    p.add_argument('--seed', type=int, default=10)
    p.add_argument('--num_frames', type=int, default=60)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--num_workers', type=int, default=0,
                   help='Shared loader: 0 uses a background thread; N uses persistent workers')
    p.add_argument('--main_process_prefetch_batches', type=int, default=4,
                   help='Training batches prepared ahead of the GPU; 0 selects 4')
    p.add_argument('--motion_cache_size', type=int, default=0)
    p.add_argument('--joint_buckets', type=int, nargs='+', default=None,
                   help='Same joint-count ceilings as train_anytop; defaults to JOINT_BUCKETS')
    p.add_argument('--width', type=int, default=128)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--encoder_layers', type=int, default=None,
                   help='Encoder depth; defaults to --layers when unspecified')
    p.add_argument('--decoder_layers', type=int, default=None,
                   help='Decoder depth; defaults to --layers when unspecified')
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--slots', type=int, default=8)
    p.add_argument('--latent_dim', type=int, default=32)
    p.add_argument('--ff_size', type=int, default=0)
    p.add_argument('--last_layer_ff', type=int, default=0)
    p.add_argument('--cross_limb_dim', type=int, default=0)
    p.add_argument('--cross_limb_last_n', type=int, default=0)
    p.add_argument('--temporal_rope', action='store_true')
    p.add_argument('--layerwise_latent', action='store_true')
    p.add_argument('--decoder_temporal_kernel', type=int, default=0,
                   help='Optional decoder output temporal residual convolution; zero disables')
    p.add_argument('--compile', choices=('None', 'default', 'reduce-overhead'), default='None')
    p.add_argument('--variational', action='store_true')
    p.add_argument('--kl_weight', type=float, default=0.0,
                   help='KL weight; KL is averaged per valid latent scalar')
    p.add_argument('--amp_dtype', choices=('fp32', 'fp16'), default='fp32')
    p.add_argument('--num_steps', type=int, default=5000)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--log_interval', type=int, default=100)
    p.add_argument('--save_interval', type=int, default=1000)
    p.add_argument('--eval_interval', type=int, default=1000)
    p.add_argument('--export_samples', type=int, default=4, help='BVH pairs; all samples also receive .npz exports')
    p.add_argument('--export_motion_names', nargs='+',
                   help='Export only these clip stems, in the requested order')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--auto_resume', action='store_true')
    for argument in LOSS_WEIGHTS.values():
        p.add_argument('--' + argument, type=float, default=0.0)
    p.add_argument('--drot_joint_boost', type=float, default=0.0)
    p.add_argument('--motion_step_warmup_steps', type=int, default=0,
                   help='Linearly ramp dpos/drot/acc and KL weights from zero to their configured values')
    for key, default in (AUGMENTATION_DEFAULTS | MODEL_REGULARIZATION_DEFAULTS).items():
        p.add_argument('--' + key, type=float, default=default)
    return p


def main():
    args = parser().parse_args()
    if min(args.num_frames, args.batch_size, args.num_steps, args.log_interval,
           args.save_interval, args.eval_interval) < 1:
        raise ValueError('Lengths, batch size, steps and intervals must be positive')
    if args.motion_step_warmup_steps < 0:
        raise ValueError('motion_step_warmup_steps must be nonnegative')
    if min(args.num_workers, args.main_process_prefetch_batches, args.motion_cache_size) < 0:
        raise ValueError('Worker, prefetch and cache counts must be nonnegative')
    for key, default in (AUGMENTATION_DEFAULTS | MODEL_REGULARIZATION_DEFAULTS).items():
        value = getattr(args, key)
        if value is None and default is None:
            continue
        if not np.isfinite(value) or (value < 1 if key == 'motion_speed_aug' else not 0 <= value <= 1):
            raise ValueError(f'Invalid {key}: {value}')
    if not np.isfinite(args.kl_weight) or args.kl_weight < 0 or args.variational != (args.kl_weight > 0):
        raise ValueError('VAE requires --variational and a positive --kl_weight; AE uses kl_weight=0')
    torch.set_num_threads(4)
    fixseed(args.seed)
    device = torch.device(args.device)
    if args.amp_dtype == 'fp16' and device.type != 'cuda':
        raise ValueError('AMP FP16 training requires CUDA')
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
    save = Path(args.save_dir)
    save.mkdir(parents=True, exist_ok=True)
    checkpoint = save / 'autoencoder.pt'
    if checkpoint.exists() and not args.auto_resume:
        raise FileExistsError(f'{checkpoint} exists; use --auto_resume or a new --save_dir')
    resume = torch.load(checkpoint, map_location='cpu', weights_only=False) if checkpoint.exists() else None
    if resume is not None:
        if resume['decoder_content_version'] != DECODER_CONTENT_VERSION:
            raise ValueError('Checkpoint decoder content version does not match this code')
    cond_hash = hashlib.sha256(Path(args.cond_path).read_bytes()).hexdigest()
    dataset = get_dataset(num_frames=args.num_frames, split='train', cond_path=args.cond_path,
                          sample_filter=args.sample_filter, sample_filter_seed=args.seed,
                          motion_cache_size=args.motion_cache_size,
                          loop_tile_single_prob=1.0, motion_speed_aug=1.0,
                          topology_aug_prob=0.0, bone_length_aug_prob=0.0,
                          rest_repose_prob=0.0, rest_variant_prob=0.0, ref_encoder=False)
    source = dataset.motion_dataset
    names = list(source.name_list)
    if resume is not None:
        if resume['args']['amp_dtype'] != args.amp_dtype:
            raise ValueError('Cannot resume with changed amp_dtype; use a new run')
        if (resume['kind'] != 'motion_autoencoder_7a' or resume['motion_names'] != names
                or resume['canonical_version'] != CANONICAL_FEATURE_SPACE
                or resume['cond_sha256'] != cond_hash):
            raise ValueError('Checkpoint type, canonical contract, skeleton data or selected training clips changed')
        for key in ('seed', 'num_frames', 'width', 'layers', 'heads', 'slots', 'latent_dim', 'batch_size', 'lr', 'weight_decay'):
            if resume['args'][key] != getattr(args, key):
                raise ValueError(f'Cannot resume with changed {key}')
        for key in ('encoder_layers', 'decoder_layers'):
            previous = resume['args'][key]
            previous = resume['args']['layers'] if previous is None else previous
            current = getattr(args, key)
            current = args.layers if current is None else current
            if previous != current:
                raise ValueError(f'Cannot resume with changed {key}; use a new run')
        for key in (*LOSS_WEIGHTS.values(), 'drot_joint_boost', 'motion_step_warmup_steps'):
            if resume['args'][key] != getattr(args, key):
                raise ValueError(f'Cannot resume with changed {key}; use a new run for a new objective')
        for key in ('ff_size', 'last_layer_ff', 'cross_limb_dim', 'cross_limb_last_n',
                    'temporal_rope', 'layerwise_latent', 'decoder_temporal_kernel'):
            if resume['args'][key] != getattr(args, key):
                raise ValueError(f'Cannot resume with changed {key}; use a new run')
        for key in AUGMENTATION_DEFAULTS | MODEL_REGULARIZATION_DEFAULTS:
            if resume['args'][key] != getattr(args, key):
                raise ValueError(f'Cannot resume with changed {key}; use a new run')
        for key in ('variational', 'kl_weight'):
            if resume['args'][key] != getattr(args, key):
                raise ValueError(f'Cannot resume with changed {key}; use a new run')
        fixed = resume['fixed_batch']
    else:
        samples = []
        for index, name in enumerate(names):
            random.seed(args.seed + index)
            np.random.seed(args.seed + index)
            samples.append(source.prepare_sample_by_name(name, loop_offset=0))
        width = max(sample[0].shape[1] for sample in samples)
        motion, conditions = truebones_batch_collate(samples, joint_buckets=(width,))
        # Keep conversion context for metrics/export, but discard the unused reference bundle.
        y = {k: v for k, v in conditions['y'].items() if not k.startswith('ref_')}
        fixed = (motion[:, :, :9].contiguous(), y)
    # Evaluation windows stay unaugmented; training draws fresh data from the same source.
    augmentation = {key: getattr(args, key) for key in AUGMENTATION_DEFAULTS}
    train_loader = create_training_data_loader(args, dataset)
    buckets = resolve_joint_buckets(args.joint_buckets or JOINT_BUCKETS, source.opt.max_joints)
    x_cpu, y_cpu = fixed
    x = x_cpu.to(device)
    y = {k: v.to(device) if torch.is_tensor(v) else v for k, v in y_cpu.items()}
    model = MotionAutoencoder(t5_out_dim=y['joints_names_embs'].shape[-1], width=args.width,
                              layers=args.layers, heads=args.heads, slots=args.slots,
                              encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers,
                              latent_dim=args.latent_dim, ff_size=args.ff_size,
                              last_layer_ff=args.last_layer_ff, cross_limb_dim=args.cross_limb_dim,
                              cross_limb_last_n=args.cross_limb_last_n, temporal_rope=args.temporal_rope,
                              layerwise_latent=args.layerwise_latent,
                              decoder_temporal_kernel=args.decoder_temporal_kernel,
                              variational=args.variational,
                              **{key: getattr(args, key) for key in MODEL_REGULARIZATION_DEFAULTS}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp_dtype == 'fp16')
    criterion = MotionReconstructionLosses(
        drot_joint_boost=args.drot_joint_boost,
        **{key: getattr(args, key) for key in LOSS_WEIGHTS.values()})
    start = 0
    if resume is not None:
        model.load_state_dict(resume['model'])
        optimizer.load_state_dict(resume['optimizer'])
        if scaler.is_enabled():
            scaler.load_state_dict(resume['grad_scaler'])
        start = resume['step']
        # Network/graph dropout consumes torch RNG even for deterministic AE.
        torch.set_rng_state(resume['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(resume['cuda_rng'], device)
    (save / 'args.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    console = HumanOutputFormat(sys.stdout)
    tqdm.write(f'[INFO] save_dir: {save}')
    tqdm.write(f'[INFO] Train split: {len(x):,} motions, {x.shape[-1]} frames/window, '
               f'joints={int(y["n_joints"].min())}..{int(y["n_joints"].max())} '
               f'(fixed evaluation padded to {x.shape[1]}), train batch_size={args.batch_size}')
    tqdm.write(f'[INFO] shared loader: workers={args.num_workers}, '
               f'prefetch={args.main_process_prefetch_batches or 4}, joint_buckets={buckets}; '
               'full training batches, one static shape per bucket')
    tqdm.write(f'[model] {"VAE" if args.variational else "AE"}, '
               f'parameters={sum(p.numel() for p in model.parameters()):,}, '
               f'width={args.width}, encoder_layers={len(model.encoder.layers)}, '
               f'decoder_layers={len(model.decoder.layers)}, slots={args.slots}, latent_dim={args.latent_dim}')
    reader_layers = [i + 1 for i, layer in enumerate(model.decoder.layers)
                     if hasattr(layer, 'motion_latent_attn')]
    tqdm.write(f'[model] latent reads: initial + decoder layers {reader_layers}')
    tqdm.write(f'[INFO] device={device}, train={args.amp_dtype}, losses/evaluation=fp32, '
               f'lr={args.lr:.3e}, beta={args.kl_weight:g}, warmup_steps={args.motion_step_warmup_steps}')
    active_augmentation = {key: value for key, value in augmentation.items()
                           if value != AUGMENTATION_DEFAULTS[key]}
    tqdm.write(f'[INFO] online augmentation: {active_augmentation or "disabled"}')
    if resume is not None:
        tqdm.write(f'[INFO] Resuming {checkpoint} at completed step {start:,}/{args.num_steps:,}')
        tqdm.write('[INFO] Shared prefetch loader restarts its data stream on resume; '
                   'worker RNGs and queued batches are not checkpointed (same as train_anytop).')
    forward_model = model
    if args.compile != 'None':
        from utils.compile_util import configure_triton_c_compiler
        compiler = configure_triton_c_compiler()
        if args.compile == 'reduce-overhead':
            os.environ['TORCHINDUCTOR_CUDAGRAPHS'] = '1'
        forward_model = torch.compile(model, mode=args.compile, dynamic=False, fullgraph=True)
        tqdm.write(f'[INFO] compile={args.compile}, fullgraph=True, CUDA Graphs='
                   f'{args.compile == "reduce-overhead"}, launcher={compiler}; '
                   'the first batch of each joint bucket includes compilation')
    tb_dir = save / datetime.now().strftime('run_%Y%m%d_%H%M%S')
    writer = SummaryWriter(str(tb_dir), purge_step=start + 1 if resume else None)
    tqdm.write(f'[INFO] TensorBoard log_dir: {tb_dir}')
    writer.add_text('config', json.dumps(vars(args), indent=2), start)

    def run_eval(step):
        report, predictions = evaluate(model, x, y, args.batch_size)
        report['step'] = step
        (save / 'reconstruction_metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        summary = report['summary']
        tqdm.write(f'eval_step[{step}]: mse[{summary["mse"]:.5f}] '
                   f'rotation_error[{summary["rotation_error_deg"]:.3f} deg] '
                   f'({summary["sample_count"]} fixed training windows)')
        # A shuffled batch of one is unchanged, so it cannot diagnose latent usage.
        console.writekvs({key: value for key, value in summary.items()
                          if key not in ('sample_count', 'shuffle_available')
                          and (summary['shuffle_available'] or not key.startswith('shuffled_z_'))})
        if not summary['shuffle_available']:
            tqdm.write('[eval] shuffled-z diagnostic unavailable: requires at least two windows')
        for key, value in report['summary'].items():
            if isinstance(value, (int, float)):
                writer.add_scalar('eval/' + key, value, step)
        ratios = [r['angular_step_rms_ratio'] for r in report['samples']
                  if r['angular_step_rms_ratio'] is not None]
        if ratios:
            writer.add_scalar('eval/angular_step_rms_ratio_median', float(np.median(ratios)), step)
        writer.flush()
        return report, predictions

    initial, _ = run_eval(start)
    if start == 0:
        (save / 'initial_metrics.json').write_text(json.dumps(initial, indent=2), encoding='utf-8')
    tqdm.write(f'train steps: {args.num_steps} (log={args.log_interval}, '
               f'eval={args.eval_interval}, save={args.save_interval})')
    batches = training_batches(train_loader)
    with (save / 'training.jsonl').open('a', encoding='utf-8') as log, tqdm(
            range(start + 1, args.num_steps + 1), initial=start, total=max(start, args.num_steps),
            desc='AE training', unit='step', dynamic_ncols=True) as progress:
        for step in progress:
            model.train()
            target, cond = training_batch_to_device(next(batches), device)
            skeleton, frame, loop = model_inputs(target, cond)
            step_scale = (min(step / args.motion_step_warmup_steps, 1.0)
                          if args.motion_step_warmup_steps else 1.0)
            # Retry the same batch after overflow: num_steps counts actual optimizer updates.
            for overflow_retries in range(17):
                optimizer.zero_grad(set_to_none=True)
                if args.compile != 'None':
                    torch.compiler.cudagraph_mark_step_begin()
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=args.amp_dtype == 'fp16'):
                    if args.variational:
                        prediction, mu, logvar = forward_model(target, skeleton, frame, loop,
                                                               return_posterior=True)
                    else:
                        prediction = forward_model(target, skeleton, frame, loop)
                terms = criterion(prediction, target, cond, step_loss_scale=step_scale)
                if args.variational:
                    terms['kl_loss'] = diagonal_gaussian_kl(mu, logvar, frame)
                    terms['loss'] = terms['loss'] + args.kl_weight * step_scale * terms['kl_loss']
                loss = terms['loss']
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite reconstruction loss at step {step}')
                grad_norm, updated = scaled_optimizer_step(loss, model, optimizer, scaler)
                if updated:
                    break
                tqdm.write(f'[AMP] step[{step}]: overflow on attempt {overflow_retries + 1}/17; '
                           'optimizer update skipped, '
                           f'scale={scaler.get_scale():g}')
            else:
                raise FloatingPointError(f'Repeated FP16 gradient overflow at step {step}')
            if overflow_retries:
                writer.add_scalar('train/amp_overflow_retries', overflow_retries, step)
            if step % args.log_interval == 0 or step == start + 1 or step == args.num_steps:
                record = {'step': step, **{key: float(value.detach()) for key, value in terms.items()}}
                effective_weights = {
                    key: weight * (step_scale if key in ('dpos_loss', 'drot_loss', 'acc_loss') else 1.0)
                    for key, weight in criterion.weights.items() if weight}
                record.update({key + '_weighted': weight * float(terms[key].detach())
                               for key, weight in effective_weights.items()})
                if args.variational:
                    record['kl_loss_weighted'] = args.kl_weight * step_scale * float(terms['kl_loss'].detach())
                record['motion_step_warmup_scale'] = step_scale
                record.update(grad_norm=float(grad_norm), amp_scale=scaler.get_scale())
                record['lr'] = optimizer.param_groups[0]['lr']
                console.writekvs(record)
                tqdm.write(f'step[{step}]: loss[{record["loss"]:.5f}]')
                log.write(json.dumps(record) + '\n')
                log.flush()
                for key, value in record.items():
                    if key != 'step' and not key.endswith('_weighted'):
                        writer.add_scalar('train/' + key, value, step)
                writer.flush()
            if step % args.save_interval == 0 or step == args.num_steps:
                payload = dict(kind='motion_autoencoder_7a', step=step, args=vars(args),
                               decoder_content_version=DECODER_CONTENT_VERSION,
                               model_config=model.config, model=model.state_dict(),
                               optimizer=optimizer.state_dict(), data_pipeline='shared_loader_v1',
                               grad_scaler=scaler.state_dict(),
                               fixed_batch=fixed, motion_names=names, canonical_version=CANONICAL_FEATURE_SPACE,
                               cond_sha256=cond_hash,
                               latent_layout='B,T,K,d',
                               beta=args.kl_weight,
                               kl_warmup_steps=args.motion_step_warmup_steps if args.variational else 0,
                               torch_rng=torch.get_rng_state(),
                               cuda_rng=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)
                temporary = save / 'autoencoder.tmp.pt'
                torch.save(payload, temporary)
                temporary.replace(checkpoint)
                tqdm.write(f'[INFO] step[{step}]: saved checkpoint to {checkpoint}')
            if step % args.eval_interval == 0 and step != args.num_steps:
                run_eval(step)
    report, predictions = run_eval(max(start, args.num_steps))
    export = save / 'reconstructions'
    export.mkdir(exist_ok=True)
    export_indices = list(range(len(report['samples'])))
    if args.export_motion_names:
        export_indices = []
        for name in args.export_motion_names:
            matches = [i for i, record in enumerate(report['samples'])
                       if Path(record['motion_name']).stem == name]
            if len(matches) != 1:
                raise ValueError(f'Export clip must match exactly once: {name}')
            export_indices.append(matches[0])
    for index in export_indices:
        record, (gt, pred) = report['samples'][index], predictions[index]
        stem = Path(record['motion_name']).stem if args.export_motion_names else f'{index:03d}'
        np.savez_compressed(export / f'{stem}.npz', original=gt, reconstruction=pred,
                            motion_name=record['motion_name'], layout='physical working T,J,12', fps=FPS)
        if args.export_motion_names or index < args.export_samples:
            cond = source.cond_dict[record['object_type']]
            is_loop = bool(y_cpu['is_loop'][index])
            for label, motion in (('original', gt), ('reconstruction', pred)):
                stored = working_to_stored(motion, is_loop=is_loop)
                np.save(export / f'{stem}_{label}.npy', stored)
                write_feature_bvh(stored, cond, str(export / f'{stem}_{label}.bvh'), fps=FPS)
    tqdm.write(f'[INFO] Saved checkpoint, metrics and reconstructions to {save}')
    writer.close()


if __name__ == '__main__':
    main()
