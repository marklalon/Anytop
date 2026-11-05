"""Train a motion autoencoder with held-out validation and latent bypass checks."""
import argparse
import copy
import hashlib
import json
import os
import sys
from datetime import datetime
from itertools import chain
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from tqdm import tqdm

from data_loaders.get_data import build_dataset_loader, get_dataset
from data_loaders.joint_buckets import resolve_joint_buckets
from data_loaders.tensors import truebones_batch_collate
from data_loaders.truebones.truebones_utils.canonical_features import (
    CANONICAL_FEATURE_SPACE, canonical_to_physical_hml,
)
from data_loaders.truebones.truebones_utils.param_utils import JOINT_BUCKETS
from diffusion.losses import geodesic_distance
from diffusion.nn import update_ema
from diffusion.logger import HumanOutputFormat
from model.motion_autoencoder import (
    DECODER_CONTENT_VERSION, MotionAutoencoder,
    diagonal_gaussian_kl, motion_masks, to_working_motion,
)
from train.motion_reconstruction_losses import LOSS_WEIGHTS, MotionGeometryLosses, MotionReconstructionLosses
from train.latent_validation import build_validation_loader
from train.training_loop import _fixed_validation_rng
from utils.fixseed import fixseed
from utils.checkpoint_cond import snapshot_training_cond
from utils.lr_schedule import cosine_decay_lr_lambda
from torch.utils.tensorboard import SummaryWriter


SKELETON_KEYS = ('n_joints', 'joints_names_embs', 'rest_geom', 'joint_struct',
                 'graph_dist', 'joints_relations')
TIME_KEYS = ('is_loop', 'loop_tile_count', 'resample_speed_cond')


AUGMENTATION_DEFAULTS = dict(motion_speed_aug=1.0, loop_tile_single_prob=1.0,
                             loop_cond_prob=1.0, loop_cond_prob_free=None,
                             loop_cond_prob_anchored=None, joint_name_drop_prob=0.0,
                             topology_aug_prob=0.0, bone_length_aug_prob=0.0,
                             rest_repose_prob=0.0, rest_variant_prob=0.0,
                             rest_geom_joint_drop_prob=0.0)


MODEL_REGULARIZATION_DEFAULTS = dict(dropout_prob=0.0)

class AutoencoderBatchTransform:
    """Picklable CPU transform, run inside the shared worker/prefetch pipeline."""

    def __init__(self, joint_name_drop_prob=0.0, rest_geom_joint_drop_prob=0.0):
        self.joint_name_drop_prob = joint_name_drop_prob
        self.rest_geom_joint_drop_prob = rest_geom_joint_drop_prob

    def _skeleton(self, y, *, source):
        skeleton = {k: y[k] for k in SKELETON_KEYS if k in y}
        if self.joint_name_drop_prob > 0:
            names = skeleton['joints_names_embs']
            drop = torch.from_numpy(np.random.random(names.shape[:2]) < self.joint_name_drop_prob)
            skeleton['joints_names_embs'] = names.masked_fill(drop[..., None], 0)
        # Decoder geometry always stays complete. Loss conversion contexts are
        # never corrupted, including the rest/bind frames and canonical stats.
        if source and self.rest_geom_joint_drop_prob > 0:
            geometry = skeleton['rest_geom']
            drop = torch.from_numpy(np.random.random(geometry.shape[:2]) < self.rest_geom_joint_drop_prob)
            skeleton['rest_geom'] = geometry.masked_fill(drop[..., None], 0)
        return skeleton

    def __call__(self, batch):
        motion, conditions = batch
        y = conditions['y']
        clean = {k: v for k, v in y.items() if not k.startswith(('ref_', 'ae_view_'))}
        # Independent name masks for encoder and decoder; never mutate cached
        # tensors or reuse the encoder's name mask as the decoder's mask.
        clean['ae_source_skeleton'] = self._skeleton(clean, source=True)
        clean['ae_target_skeleton'] = self._skeleton(clean, source=False)
        if 'ae_view_cond' in y:
            other_motion, other = self((y['ae_view_motion'], {'y': y['ae_view_cond']}))
            clean.update(ae_view_motion=other_motion, ae_view_cond=other['y'])
        return motion[:, :, :9].contiguous(), {'y': clean}


def create_training_data_loader(args, dataset):
    for key in AUGMENTATION_DEFAULTS:
        setattr(dataset.motion_dataset.opt, key, getattr(args, key))
    dataset.motion_dataset.opt.autoencoder_dual_view = args.dual_view
    return build_dataset_loader(
        dataset, batch_size=args.batch_size, drop_last=True,
        num_workers=args.num_workers,
        main_process_prefetch_batches=args.main_process_prefetch_batches,
        joint_buckets=args.joint_buckets or JOINT_BUCKETS,
        batch_transform=AutoencoderBatchTransform(
            args.joint_name_drop_prob, args.rest_geom_joint_drop_prob),
    )


def training_batches(loader):
    # A small bucket may produce no full batch in one pass. The shared sampler
    # carries its remainder forward; do not treat such a pass as an empty dataset.
    while True:
        yield from loader


def training_batch_to_device(batch, device):
    target, conditions = batch
    def move(value):
        if torch.is_tensor(value):
            return value.to(device, non_blocking=True)
        if isinstance(value, dict):
            return {k: move(v) for k, v in value.items()}
        return value
    return target.to(device, non_blocking=True), move(conditions['y'])


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


def dual_view_training_terms(model, criterion, target, cond, *, step_scale=1.0,
                             kl_weight=0.0, self_weight=1.0):
    """Average A->B/B->A and retain equally weighted self reconstruction by default."""
    other, other_cond = cond['ae_view_motion'], cond['ae_view_cond']
    _, frame, loop = model_inputs(target, cond)
    clean_a, clean_b = cond['ae_target_skeleton'], other_cond['ae_target_skeleton']
    cross_terms, self_terms, kls = [], [], []
    for source, source_cond, destination, destination_cond, clean_source, clean_target in (
            (target, cond, other, other_cond, clean_a, clean_b),
            (other, other_cond, target, cond, clean_b, clean_a)):
        prediction, reconstruction, mu, logvar = model(
            source, source_cond['ae_source_skeleton'], frame, loop,
            target_skeleton=clean_target, self_skeleton=clean_source)
        cross_terms.append(criterion(prediction, destination, destination_cond, step_loss_scale=step_scale))
        self_terms.append(criterion(reconstruction, source, source_cond, step_loss_scale=step_scale))
        if logvar is not None:
            kls.append(diagonal_gaussian_kl(mu, logvar, frame))
    terms = {key: (sum(t[key] for t in cross_terms)
                   + self_weight * sum(t[key] for t in self_terms)) / (2 * (1 + self_weight))
             for key in cross_terms[0]}
    terms['cross_reconstruction_loss'] = sum(t['loss'] for t in cross_terms) / 2
    terms['self_reconstruction_loss'] = sum(t['loss'] for t in self_terms) / 2
    if kls:
        terms['kl_loss'] = sum(kls) / 2
        terms['loss'] = terms['loss'] + kl_weight * step_scale * terms['kl_loss']
    return terms


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
def update_autoencoder_ema(model_avg, model, rate):
    update_ema(model_avg.parameters(), model.parameters(), rate=rate)
    # Copy persistent buffers only; attention caches may have dynamic shapes.
    parameters = set(dict(model_avg.named_parameters()))
    for name in model_avg.state_dict().keys() - parameters:
        model_avg.get_buffer(name).copy_(model.get_buffer(name))


def create_lr_scheduler(optimizer, args, completed_steps=0):
    for group in optimizer.param_groups:
        group['initial_lr'] = args.lr
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, cosine_decay_lr_lambda(
            args.lr, args.lr_final, args.lr_decay_start, args.num_steps),
        last_epoch=completed_steps - 1)


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
    records = []
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
        for row in range(len(target)):
            record = {'motion_name': cond['motion_name'][row], 'object_type': cond['object_type'][row],
                      **{key: float(value[row]) for key, value in errors.items()}}
            record['angular_step_rms_gt'] = float(rms[0][row])
            record['angular_step_rms_reconstruction'] = float(rms[1][row])
            record['angular_step_rms_ratio'] = float(rms[1][row] / rms[0][row]) if rms[0][row] > 1e-4 else None
            records.append(record)
    summary = {key: float(np.mean([r[key] for r in records])) for key in errors}
    summary['sample_count'] = len(records)
    summary['shuffle_available'] = len(x) > 1
    summary['zero_z_mse_increase'] = summary['zero_z_mse'] - summary['mse']
    summary['shuffled_z_mse_increase'] = summary['shuffled_z_mse'] - summary['mse']
    return {'summary': summary, 'samples': records}


def evaluate_validation(model, loader, batch_size, device):
    if loader is None:
        return None
    records = []
    shuffled_records = []
    with _fixed_validation_rng(device):
        for batch in loader:
            x, y = training_batch_to_device(batch, device)
            report = evaluate(model, x[:, :, :9], y, batch_size)
            records.extend(report['samples'])
            if report['summary']['shuffle_available']:
                shuffled_records.extend(report['samples'])
    if not records:
        return None
    keys = [key for key in records[0] if key not in (
        'motion_name', 'object_type', 'angular_step_rms_ratio')]
    summary = {key: float(np.mean([r[key] for r in records])) for key in keys}
    # A singleton tail batch cannot swap latents. Exclude it from both sides
    # of the shuffled comparison rather than diluting the diagnostic.
    if shuffled_records:
        summary['shuffled_z_mse'] = float(np.mean([r['shuffled_z_mse'] for r in shuffled_records]))
    summary.update(sample_count=len(records), shuffle_available=bool(shuffled_records),
                   zero_z_mse_increase=summary['zero_z_mse'] - summary['mse'],
                   shuffled_z_mse_increase=float(np.mean([
                       r['shuffled_z_mse'] - r['mse'] for r in shuffled_records
                   ])) if shuffled_records else 0.0)
    return {'summary': summary, 'samples': records, 'split': 'val'}


@torch.no_grad()
def evaluate_dual_view_validation(model, loader, args, device):
    """Fixed RNG, held-out motions, independent skeleton views, no condition drops."""
    if loader is None:
        return {}
    device = torch.device(device)
    # Do not modify the ordinary validation loader (which may be prefetching).
    source = copy.copy(loader.dataset.motion_dataset)
    source.opt = copy.copy(source.opt)
    for key in ('topology_aug_prob', 'bone_length_aug_prob', 'rest_repose_prob', 'rest_variant_prob'):
        setattr(source.opt, key, getattr(args, key))
    source.opt.autoencoder_dual_view = True
    model.eval()
    totals = dict(cross_reconstruction_loss=0., self_reconstruction_loss=0.)
    count = 0
    transform = AutoencoderBatchTransform()
    criterion = MotionReconstructionLosses()
    names = source.name_list[source.pointer:]
    with _fixed_validation_rng(device):
        for start in range(0, len(names), args.batch_size):
            selected = names[start:start + args.batch_size]
            samples = [source.prepare_sample_by_name(name) for name in selected]
            batch = transform(truebones_batch_collate(samples, joint_buckets=args.joint_buckets or JOINT_BUCKETS))
            x, y = training_batch_to_device(batch, device)
            terms = dual_view_training_terms(model, criterion, x, y)
            for key in totals:
                totals[key] += float(terms[key]) * len(selected)
            count += len(selected)
    if not count:
        return {}
    return {'dual_view_cross_mse': totals['cross_reconstruction_loss'] / count,
            'dual_view_self_mse': totals['self_reconstruction_loss'] / count,
            'dual_view_sample_count': count}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cond_path', default='dataset/cond_all.npy')
    p.add_argument('--sample_filter', default=None, help='Optional clip filter JSON; omitted selects the full dataset')
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
    p.add_argument('--encoder_width', type=int, default=None,
                   help='Encoder width; defaults to --width when unspecified')
    p.add_argument('--decoder_width', type=int, default=None,
                   help='Decoder width; defaults to --width when unspecified')
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--encoder_layers', type=int, default=None,
                   help='Encoder depth; defaults to --layers when unspecified')
    p.add_argument('--decoder_layers', type=int, default=None,
                   help='Decoder depth; defaults to --layers when unspecified')
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--slots', type=int, default=8)
    p.add_argument('--latent_dim', type=int, default=32)
    p.add_argument('--ff_size', type=int, default=0)
    p.add_argument('--encoder_ff_size', type=int, default=None,
                   help='Encoder FF size; defaults to --ff_size; zero uses 4 * encoder width')
    p.add_argument('--decoder_ff_size', type=int, default=None,
                   help='Decoder FF size; defaults to --ff_size; zero uses 4 * decoder width')
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
    p.add_argument('--lr_final', type=float, default=1e-5)
    p.add_argument('--lr_decay_start', type=int, default=None,
                   help='Start cosine decay to lr_final; omitted keeps LR constant')
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--use_ema', action='store_true', help='Evaluate EMA weights')
    p.add_argument('--ema_rate', type=float, default=0.99)
    p.add_argument('--log_interval', type=int, default=100)
    p.add_argument('--save_interval', type=int, default=1000)
    p.add_argument('--eval_interval', type=int, default=1000)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--auto_resume', action='store_true')
    p.add_argument('--dual_view', action='store_true',
                   help='Independent skeleton views, shared time map; bidirectional cross reconstruction with independent encoder/decoder name drop')
    p.add_argument('--dual_view_self_weight', type=float, default=1.0,
                   help='Self reconstruction relative to cross reconstruction in dual-view mode (default: equal weight)')
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
    if not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('lr must be finite and positive')
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError('weight_decay must be finite and nonnegative')
    if not np.isfinite(args.ema_rate) or not 0 <= args.ema_rate <= 1:
        raise ValueError('ema_rate must lie in [0, 1]')
    cosine_decay_lr_lambda(args.lr, args.lr_final, args.lr_decay_start, args.num_steps)
    if min(args.num_frames, args.batch_size, args.num_steps, args.log_interval,
           args.save_interval, args.eval_interval) < 1:
        raise ValueError('Lengths, batch size, steps and intervals must be positive')
    if args.motion_step_warmup_steps < 0:
        raise ValueError('motion_step_warmup_steps must be nonnegative')
    if not np.isfinite(args.dual_view_self_weight) or args.dual_view_self_weight < 0:
        raise ValueError('dual_view_self_weight must be finite and nonnegative')
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
        for key in ('seed', 'num_frames', 'layers', 'heads', 'slots', 'latent_dim', 'batch_size', 'lr', 'weight_decay'):
            if resume['args'][key] != getattr(args, key):
                raise ValueError(f'Cannot resume with changed {key}')
        for key, default in (('lr_decay_start', None), ('lr_final', 1e-5),
                             ('use_ema', False), ('ema_rate', 0.99)):
            if resume['args'].get(key, default) != getattr(args, key):
                raise ValueError(f'Cannot resume with changed {key}; use a new run')
        for key, fallback in (('encoder_layers', 'layers'), ('decoder_layers', 'layers'),
                              ('encoder_width', 'width'), ('decoder_width', 'width'),
                              ('encoder_ff_size', 'ff_size'), ('decoder_ff_size', 'ff_size')):
            previous = resume['args'][key]
            previous = resume['args'][fallback] if previous is None else previous
            current = getattr(args, key)
            current = getattr(args, fallback) if current is None else current
            if previous != current:
                raise ValueError(f'Cannot resume with changed {key}; use a new run')
        for key in (*LOSS_WEIGHTS.values(), 'drot_joint_boost', 'motion_step_warmup_steps'):
            if resume['args'][key] != getattr(args, key):
                raise ValueError(f'Cannot resume with changed {key}; use a new run for a new objective')
        for key in ('last_layer_ff', 'cross_limb_dim', 'cross_limb_last_n',
                    'temporal_rope', 'layerwise_latent', 'decoder_temporal_kernel'):
            if resume['args'][key] != getattr(args, key):
                raise ValueError(f'Cannot resume with changed {key}; use a new run')
        for key in AUGMENTATION_DEFAULTS | MODEL_REGULARIZATION_DEFAULTS:
            if resume['args'][key] != getattr(args, key):
                raise ValueError(f'Cannot resume with changed {key}; use a new run')
        for key in ('variational', 'kl_weight'):
            if resume['args'][key] != getattr(args, key):
                raise ValueError(f'Cannot resume with changed {key}; use a new run')
        for key, default in (('dual_view', False), ('dual_view_self_weight', 1.0)):
            if resume['args'].get(key, default) != getattr(args, key):
                raise ValueError(f'Cannot resume with changed {key}; use a new run for a new objective')
    # Preserve the accepted training contract for inference, including resumes.
    snapshot_training_cond(args.cond_path, save, cond_hash)
    augmentation = {key: getattr(args, key) for key in AUGMENTATION_DEFAULTS}
    train_loader = create_training_data_loader(args, dataset)
    batches = training_batches(train_loader)
    # The training example supplies model dimensions only; evaluation uses val.
    example = next(batches)
    batches = chain((example,), batches)
    val_loader = build_validation_loader(args)
    buckets = resolve_joint_buckets(args.joint_buckets or JOINT_BUCKETS, source.opt.max_joints)
    x, y = training_batch_to_device(example, device)
    model = MotionAutoencoder(t5_out_dim=y['joints_names_embs'].shape[-1], width=args.width,
                              layers=args.layers, heads=args.heads, slots=args.slots,
                              encoder_width=args.encoder_width, decoder_width=args.decoder_width,
                              encoder_ff_size=args.encoder_ff_size, decoder_ff_size=args.decoder_ff_size,
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
    model_avg = copy.deepcopy(model).eval().requires_grad_(False) if args.use_ema else None
    if resume is not None and model_avg is not None:
        model_avg.load_state_dict(resume['model_ema'])
    lr_scheduler = create_lr_scheduler(optimizer, args, start)
    (save / 'args.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    console = HumanOutputFormat(sys.stdout)
    tqdm.write(f'[INFO] save_dir: {save}')
    tqdm.write(f'[INFO] Train split: {len(names):,} motions, {x.shape[-1]} frames/window, '
               f'joints={int(y["n_joints"].min())}..{int(y["n_joints"].max())} '
               f'(first training batch padded to {x.shape[1]}), train batch_size={args.batch_size}')
    tqdm.write(f'[INFO] shared loader: workers={args.num_workers}, '
               f'prefetch={args.main_process_prefetch_batches or 4}, joint_buckets={buckets}; '
               'full training batches, one static shape per bucket')
    tqdm.write(f'[model] {"VAE" if args.variational else "AE"}, '
               f'parameters={sum(p.numel() for p in model.parameters()):,}, '
               f'encoder_width={model.encoder_width}, decoder_width={model.decoder_width}, '
               f'encoder_ff_size={model.encoder.layers[0].linear1.out_features}, '
               f'decoder_ff_size={model.decoder.layers[0].linear1.out_features}, '
               f'encoder_layers={len(model.encoder.layers)}, '
               f'decoder_layers={len(model.decoder.layers)}, slots={args.slots}, latent_dim={args.latent_dim}')
    reader_layers = [i + 1 for i, layer in enumerate(model.decoder.layers)
                     if hasattr(layer, 'motion_latent_attn')]
    tqdm.write(f'[model] latent reads: initial + decoder layers {reader_layers}')
    tqdm.write(f'[INFO] device={device}, train={args.amp_dtype}, losses/evaluation=fp32, '
               f'lr={args.lr:.3e}, beta={args.kl_weight:g}, warmup_steps={args.motion_step_warmup_steps}')
    tqdm.write(f'[INFO] lr_decay_start={args.lr_decay_start}, lr_final={args.lr_final:.3e}, '
               f'weight_decay={args.weight_decay:g}, EMA={args.use_ema}, ema_rate={args.ema_rate:g}')
    active_augmentation = {key: value for key, value in augmentation.items()
                           if value != AUGMENTATION_DEFAULTS[key]}
    tqdm.write(f'[INFO] online augmentation: {active_augmentation or "disabled"}')
    if args.dual_view:
        tqdm.write('[INFO] dual views: independent skeleton augmentation, shared temporal map; '
                   'independent encoder/decoder name drop, source-only joint geometry drop; '
                   f'bidirectional cross/self relative weights=1/{args.dual_view_self_weight:g}')
    del example, x, y
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
        evaluation_model = model_avg if model_avg is not None else model
        report = evaluate_validation(
            evaluation_model, val_loader, args.batch_size, device)
        if report is None:
            return None
        if args.dual_view:
            report['summary'].update(evaluate_dual_view_validation(
                evaluation_model, val_loader, args, device))
        report['step'] = step
        (save / 'reconstruction_metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        summary = report['summary']
        tqdm.write(f'eval_step[{step}]: mse[{summary["mse"]:.5f}] '
                   f'rotation_error[{summary["rotation_error_deg"]:.3f} deg] '
                   f'({summary["sample_count"]} validation windows)')
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
        return report

    initial = run_eval(start)
    if start == 0 and initial is not None:
        (save / 'initial_metrics.json').write_text(json.dumps(initial, indent=2), encoding='utf-8')
    tqdm.write(f'train steps: {args.num_steps} (log={args.log_interval}, '
               f'eval={args.eval_interval}, save={args.save_interval})')
    with (save / 'training.jsonl').open('a', encoding='utf-8') as log, tqdm(
            range(start + 1, args.num_steps + 1), initial=start, total=max(start, args.num_steps),
            desc='AE training', unit='step', dynamic_ncols=True) as progress:
        for step in progress:
            model.train()
            step_lr = optimizer.param_groups[0]['lr']
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
                    if args.dual_view:
                        terms = dual_view_training_terms(
                            forward_model, criterion, target, cond, step_scale=step_scale,
                            kl_weight=args.kl_weight, self_weight=args.dual_view_self_weight)
                    elif args.variational:
                        prediction, mu, logvar = forward_model(
                            target, cond['ae_source_skeleton'], frame, loop,
                            target_skeleton=cond['ae_target_skeleton'], return_posterior=True)
                    else:
                        prediction = forward_model(target, cond['ae_source_skeleton'], frame, loop,
                                                   target_skeleton=cond['ae_target_skeleton'])
                if not args.dual_view:
                    terms = criterion(prediction, target, cond, step_loss_scale=step_scale)
                if args.variational and not args.dual_view:
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
            if model_avg is not None:
                update_autoencoder_ema(model_avg, model, args.ema_rate)
            lr_scheduler.step()
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
                record['lr'] = step_lr
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
                               model_ema=model_avg.state_dict() if model_avg is not None else None,
                               optimizer=optimizer.state_dict(), data_pipeline='shared_loader_v1',
                               grad_scaler=scaler.state_dict(),
                               eval_split='val',
                               motion_names=names, canonical_version=CANONICAL_FEATURE_SPACE,
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
    run_eval(max(start, args.num_steps))
    tqdm.write(f'[INFO] Saved checkpoint and available validation metrics to {save}')
    writer.close()


if __name__ == '__main__':
    main()
