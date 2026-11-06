"""Train fixed-60 latent diffusion on online-augmented motions with a frozen AE."""
import argparse
import copy
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_loaders.get_data import get_dataset, build_dataset_loader
from data_loaders.truebones.truebones_utils.param_utils import JOINT_BUCKETS
from diffusion.latent_diffusion import create_latent_diffusion
from model.latent_motion_denoiser import LatentMotionDenoiser
from train.train_anytop import bootstrap_action_conditioning, bootstrap_species_descriptor_table
from train.latent_cuda_graph import EagerCudaGraphSteps
from train.latent_validation import build_validation_loader
from train.training_loop import _fixed_validation_rng
from train.train_autoencoder import (
    AUGMENTATION_DEFAULTS, training_batches, training_batch_to_device,
    scaled_optimizer_step, update_autoencoder_ema, reconstruction_mse)
from model.motion_autoencoder import motion_masks
from utils.fixseed import fixseed
from utils.checkpoint_cond import snapshot_training_cond
from utils.latent_util import (
    LATENT_CHECKPOINT_VERSION, TIME_CONTRACT, LatentNormalizer, ae_inputs,
    load_frozen_autoencoder, encode_target, file_sha256)
from utils.latent_analysis import load_or_analyze, validate_sensitivity_loss
from utils.lr_schedule import cosine_decay_lr_lambda


class LatentBatchTransform:
    def __init__(self, joint_name_drop_prob=0.0, rest_geom_joint_drop_prob=0.0):
        self.joint_name_drop_prob = float(joint_name_drop_prob)
        self.rest_geom_joint_drop_prob = float(rest_geom_joint_drop_prob)
        for name in ('joint_name_drop_prob', 'rest_geom_joint_drop_prob'):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f'{name} must be in [0, 1], got {value}')

    def __call__(self, batch):
        motion, conditions = batch
        # The shared collator always adds reference-motion placeholders. They
        # are not conditions for latent diffusion, so keep them out of every
        # downstream AE and denoiser path.
        y = {key: value for key, value in conditions['y'].items()
             if not key.startswith('ref_')}
        # Mask the batch condition once so the frozen AE target encoder and the
        # denoiser receive exactly the same skeleton information.
        if self.joint_name_drop_prob:
            names = y['joints_names_embs']
            drop = torch.rand(names.shape[:2], device=names.device) < self.joint_name_drop_prob
            y['joints_names_embs'] = names.masked_fill(drop[..., None], 0)
        if self.rest_geom_joint_drop_prob:
            geometry = y['rest_geom']
            drop = torch.rand(geometry.shape[:2], device=geometry.device) < self.rest_geom_joint_drop_prob
            y['rest_geom'] = geometry.masked_fill(drop[..., None], 0)
        return motion[:, :, :9].contiguous(), {'y': y}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ae_checkpoint', default='save/tiny_autoencoder_shake_v12_root_xyz')
    p.add_argument('--cond_path', default='dataset/cond_all.npy')
    p.add_argument('--sample_filter', default=None, help='Optional clip filter JSON; omitted selects the full dataset')
    p.add_argument('--save_dir', required=True)
    p.add_argument('--auto_resume', action='store_true')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--amp_dtype', choices=('fp32', 'fp16', 'bf16'), default='fp32')
    p.add_argument('--compile', choices=('None', 'default', 'reduce-overhead', 'cuda-graph'), default='None',
                   help='Training optimization: None, torch.compile mode, or cuda-graph '
                        '(eager denoiser forward/backward capture without torch.compile; '
                        'cache up to two input signatures, run additional signatures eagerly)')
    p.add_argument('--latent_mode', choices=('mean', 'sample'), default='mean',
                   help='AE posterior distribution used by both statistics and training')
    p.add_argument('--action_word_embeddings', default=None)
    for key in ('action_label_cond', 'species_cond', 'temporal_rope', 'use_ema'):
        p.add_argument('--' + key, action=argparse.BooleanOptionalAction, default=True)
    for key, value in dict(seed=10, num_frames=60, batch_size=16, num_workers=2,
                           main_process_prefetch_batches=64, motion_cache_size=16384,
                           width=256, layers=8, heads=4, ff_size=1024, skeleton_layers=2,
                           stats_batches=64, num_steps=10000, diffusion_steps=100,
                           sensitivity_probes=8,
                           log_interval=100, save_interval=1000, eval_interval=1000).items():
        p.add_argument('--' + key, type=int, default=value)
    p.add_argument('--sensitivity_samples', type=int, default=64,
                   help='Total samples in the uniform global decoder-sensitivity reservoir')
    for key, value in dict(lr=1e-4, lr_final=5e-5, weight_decay=0.0, ema_rate=0.995,
                           dropout=0.05, action_label_cfg_drop_prob=0.2,
                           direction_slot_drop_prob=0.0, modifier_slot_drop_prob=0.0,
                           noise_schedule_shift=1.0,
                           joint_name_drop_prob=0.15,
                           rest_geom_joint_drop_prob=0.1).items():
        p.add_argument('--' + key, type=float, default=value)
    p.add_argument('--lr_decay_start', type=int, default=0)
    p.add_argument('--lr_warmup_steps', type=int, default=0,
                   help='Linearly ramp LR over the first N optimizer steps, then follow the original cosine schedule')
    p.add_argument('--noise_schedule', default='cosine', choices=('cosine', 'linear'))
    p.add_argument('--joint_buckets', type=int, nargs='+')
    # These augment the paired AE target; joint name and rest geometry dropout
    # are shared by the AE target encoder and denoiser through LatentBatchTransform.
    for key, value in AUGMENTATION_DEFAULTS.items():
        if key not in ('joint_name_drop_prob', 'rest_geom_joint_drop_prob'):
            p.add_argument('--' + key, type=float, default=value)
    return p


def recorded_args(args):
    return {key: value for key, value in vars(args).items()
            if key not in ('action_conditioning', 'species_descriptor_table')}


def create_lr_scheduler(optimizer, args, completed_steps=0):
    warmup = args.lr_warmup_steps
    if not 0 <= warmup <= args.num_steps:
        raise ValueError('lr_warmup_steps must be in [0, num_steps]')
    decay = cosine_decay_lr_lambda(args.lr, args.lr_final, args.lr_decay_start, args.num_steps)

    def multiplier(step):
        # LambdaLR index 0 supplies the LR for optimizer step 1.
        ramp = min((step + 1) / warmup, 1.0) if warmup else 1.0
        return decay(step) * ramp

    for group in optimizer.param_groups:
        group['initial_lr'] = args.lr
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, multiplier, last_epoch=completed_steps - 1)


def training_forwards(model, ae, norm, compile_mode):
    """Keep original modules for optimizer/EMA/checkpoints and eager evaluation."""
    def target_encoder(x, y):
        return encode_target(ae, norm, x, y)

    if compile_mode in ('None', 'cuda-graph'):
        return model, target_encoder
    from utils.compile_util import configure_triton_c_compiler
    compiler = configure_triton_c_compiler()
    forward_model = torch.compile(model, mode=compile_mode, dynamic=False, fullgraph=True)
    compiled_encoder = torch.compile(target_encoder, mode=compile_mode, dynamic=False, fullgraph=True)

    def encode_batch(x, y):
        # Ignore clip-name lists and denoiser-only conditions: they are not AE inputs
        # and would otherwise produce unnecessary guards/recompilations.
        skeleton, time = ae_inputs(y, x.shape[-1])
        target = compiled_encoder(x, {**skeleton, **time})
        # Own the storage across denoiser backward/retries and CUDA Graph replays.
        return target.clone()

    print(f'compile={compile_mode}, fullgraph=True, launcher={compiler}; '
          'frozen AE encoding and denoiser training compile on first use of each joint bucket; '
          'statistics and evaluation remain eager')
    return forward_model, encode_batch


def build_loader(args):
    kwargs = {key: getattr(args, key) for key in AUGMENTATION_DEFAULTS
              if hasattr(args, key) and key not in (
                  'joint_name_drop_prob', 'rest_geom_joint_drop_prob')}
    dataset = get_dataset(
        num_frames=60, split='train', cond_path=args.cond_path,
        sample_filter=args.sample_filter, sample_filter_seed=args.seed,
        motion_cache_size=args.motion_cache_size, action_label_cond=args.action_label_cond,
        action_conditioning=args.action_conditioning, species_table=args.species_descriptor_table,
        **kwargs)
    loader = build_dataset_loader(
        dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers,
        main_process_prefetch_batches=args.main_process_prefetch_batches,
        joint_buckets=args.joint_buckets or JOINT_BUCKETS,
        batch_transform=LatentBatchTransform(
            args.joint_name_drop_prob, args.rest_geom_joint_drop_prob))
    return dataset, loader


class AnalysisReferenceDataset:
    """Read clips in order with a fixed loop phase instead of random rolling."""
    def __init__(self, dataset):
        self.motion = dataset.motion_dataset

    def __len__(self):
        return len(self.motion)

    def __getitem__(self, index):
        name = self.motion.name_list[self.motion.pointer + index]
        return self.motion.prepare_sample_by_name(name, loop_offset=0)


def build_analysis_loader(args):
    """Synchronous sequential reference pass: no augmentation, weighting or dropout."""
    from torch.utils.data import DataLoader
    from data_loaders.tensors import truebones_batch_collate
    kwargs = {key: 0.0 for key in AUGMENTATION_DEFAULTS
              if key not in ('joint_name_drop_prob', 'rest_geom_joint_drop_prob')}
    kwargs.update(motion_speed_aug=1.0, loop_tile_single_prob=1.0, loop_cond_prob=1.0,
                  loop_cond_prob_free=None, loop_cond_prob_anchored=None)
    dataset = get_dataset(
        num_frames=60, split='train', cond_path=args.cond_path,
        sample_filter=args.sample_filter, sample_filter_seed=args.seed,
        motion_cache_size=args.motion_cache_size, action_label_cond=args.action_label_cond,
        action_conditioning=args.action_conditioning, species_table=args.species_descriptor_table,
        **kwargs)
    dataset.motion_dataset.use_weighted_sampler = False
    loader = DataLoader(AnalysisReferenceDataset(dataset), batch_size=args.batch_size, shuffle=False,
                        drop_last=False, num_workers=0, collate_fn=truebones_batch_collate)
    transform = LatentBatchTransform()
    return lambda: (transform(batch) for batch in loader)


def analysis_reference_identity(dataset, cond_hash):
    """Invalidate on selected clips, their content/metadata or skeletons."""
    motion = dataset.motion_dataset
    records = []
    for name in motion.name_list:
        entry = getattr(motion, 'data_dict', {}).get(name, {})
        path = entry.get('motion_path')
        records.append(dict(name=name, sha256=file_sha256(path) if path else None,
                            metadata=entry.get('motion_metadata'),
                            high_frequency=bool(entry.get('high_frequency', False))))
    import hashlib
    digest = hashlib.sha256(json.dumps(records, sort_keys=True, default=str).encode()).hexdigest()
    return dict(cond_sha256=cond_hash, clips_sha256=digest, clip_count=len(records),
                reference='sequential_clean_complete_passes_global_v3')


@torch.no_grad()
def evaluate(model, ae, norm, diffusion, batch, device):
    model.eval()
    x, y = training_batch_to_device(batch, device)
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
        torch.manual_seed(1234)
        target = encode_target(ae, norm, x, y)
        t = torch.full((len(x),), diffusion.num_timesteps // 2, device=device, dtype=torch.long)
        noisy = diffusion.q_sample(target, t)
        prediction = diffusion._wrap_model(model)(noisy, t, y=y)
        skeleton, time = ae_inputs(y)
        decoded = ae.decode(norm.denormalize(prediction), skeleton, loop_metadata=time)
        baseline = ae.decode(norm.denormalize(target), skeleton, loop_metadata=time)
        _, frame = motion_masks(x, skeleton)
        metrics = {key: float(value.mean()) for key, value in
                   diffusion.latent_error_terms(prediction, target).items()}
        return dict(metrics, decoded_mse=float(reconstruction_mse(decoded, x, skeleton, frame).mean()),
                    ae_mse=float(reconstruction_mse(baseline, x, skeleton, frame).mean()))


def evaluate_validation(model, ae, norm, diffusion, loader, device):
    if loader is None:
        return None
    sums, count = {}, 0
    with _fixed_validation_rng(device):
        for batch in loader:
            size = len(batch[0])
            metrics = evaluate(model, ae, norm, diffusion, batch, device)
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + value * size
            count += size
    if count == 0:
        return None
    return {**{key: value / count for key, value in sums.items()}, 'sample_count': count}


def main(argv=None):
    args = parser().parse_args(argv)
    if args.num_frames != 60:
        raise ValueError('Only fixed 60-frame training is supported')
    if not 0 <= args.lr_warmup_steps <= args.num_steps:
        raise ValueError('lr_warmup_steps must be in [0, num_steps]')
    if min(args.batch_size, args.stats_batches, args.sensitivity_samples, args.sensitivity_probes,
           args.num_steps, args.log_interval,
           args.save_interval) < 1 or args.eval_interval < 0:
        raise ValueError('Batch/step/log/save counts must be positive and eval_interval nonnegative')
    if not np.isfinite(args.lr) or args.lr <= 0 or not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError('Invalid optimizer settings')
    if not 0 <= args.ema_rate <= 1:
        raise ValueError('ema_rate must be in [0,1]')
    fixseed(args.seed)
    torch.set_num_threads(4)
    # Allow TF32 tensor cores for FP32 AE encoding and compiled matmuls.
    torch.set_float32_matmul_precision('high')
    device = torch.device(args.device)
    if args.amp_dtype == 'fp16' and device.type != 'cuda':
        raise ValueError('fp16 requires CUDA')
    if args.compile == 'cuda-graph' and device.type != 'cuda':
        raise ValueError('--compile cuda-graph requires a CUDA device')
    save = Path(args.save_dir)
    save.mkdir(parents=True, exist_ok=True)
    path = save / 'latent_diffusion.pt'
    if path.exists() and not args.auto_resume:
        raise FileExistsError(f'{path} exists; use --auto_resume or another save_dir')
    resume = torch.load(path, map_location='cpu', weights_only=False) if path.exists() else None
    ae, binding = load_frozen_autoencoder(args.ae_checkpoint, device)
    if resume:
        # Changing normalization or enabling this objective requires a fresh
        # optimizer run, rather than silently changing a saved model's target.
        LatentNormalizer.from_metadata(resume['normalization'])
        validate_sensitivity_loss(resume.get('latent_loss'), (ae.slots, ae.latent_dim))
    cond_hash = file_sha256(args.cond_path)
    current_args = recorded_args(args)
    if resume:
        if (resume.get('kind') != LATENT_CHECKPOINT_VERSION or resume.get('ae_binding') != binding
                or resume.get('time_contract') != TIME_CONTRACT or resume.get('cond_sha256') != cond_hash):
            raise ValueError('Checkpoint/AE/skeleton/time contract mismatch')
        mutable = {'auto_resume', 'num_steps', 'log_interval', 'eval_interval', 'save_interval',
                   'num_workers', 'main_process_prefetch_batches', 'motion_cache_size', 'device', 'compile'}
        for key, value in current_args.items():
            if key not in mutable and value != resume['args'][key]:
                raise ValueError(f'Cannot resume with changed {key}')
        if args.num_steps != resume['args']['num_steps']:
            raise ValueError('num_steps binds the LR schedule; use a new run to change it')
    # Like motion-space training, refresh cond.npy on every accepted launch,
    # including resumes. Keep args.cond_path unchanged for the dataset/resume contract.
    snapshot_training_cond(args.cond_path, save, cond_hash)
    args.resume_checkpoint = str(path) if resume else ''
    bundle = bootstrap_action_conditioning(args)
    if resume and (bundle.checkpoint_metadata() if bundle else None) != resume['action_conditioning']:
        raise ValueError('Action vocabulary/role-slot contract changed')
    table = bootstrap_species_descriptor_table(args, args.cond_path, str(save))
    species_hash = file_sha256(table.source) if table else None
    if resume and resume['species_sha256'] != species_hash:
        raise ValueError('Species descriptor table changed')
    dataset, loader = build_loader(args)
    motion_names = list(dataset.motion_dataset.name_list)
    if resume and resume['motion_names'] != motion_names:
        raise ValueError('Selected training clips changed')
    batches = training_batches(loader)
    val_loader = (build_validation_loader(args, LatentBatchTransform())
                  if args.eval_interval > 0 else None)
    if resume:
        norm = LatentNormalizer.from_metadata(resume['normalization']).to(device)
        analysis = resume.get('latent_analysis')
        latent_loss = resume['latent_loss']
    else:
        # Offline work uses its own synchronous stream before training prefetch starts.
        identity = analysis_reference_identity(dataset, cond_hash)
        identity['batch_size'] = args.batch_size
        batches_per_pass = max(1, math.ceil(len(motion_names) / args.batch_size))
        # Repeat only complete passes: a partial final pass would overweight
        # the shortest clips, because the dataset is sorted by source length.
        count = math.ceil(args.stats_batches / batches_per_pass) * batches_per_pass
        reference_factory = None

        def reference_batches():
            nonlocal reference_factory
            if reference_factory is None:
                reference_factory = build_analysis_loader(args)
            return reference_factory()

        norm, latent_loss, analysis = load_or_analyze(
            ae, binding, identity, reference_batches, count, device,
            args.latent_mode, args.seed, args.sensitivity_samples, args.sensitivity_probes)
    config = dict(t5_out_dim=ae.config['t5_out_dim'], slots=ae.slots, latent_channels=ae.latent_dim)
    for key in ('width', 'layers', 'heads', 'ff_size', 'skeleton_layers', 'dropout',
                'temporal_rope', 'action_label_cond', 'species_cond',
                'action_label_cfg_drop_prob', 'direction_slot_drop_prob', 'modifier_slot_drop_prob'):
        config[key] = getattr(args, key)
    # The shared batch transform applies these masks before both target encoding
    # and denoiser conditioning, so the model must not sample a second mask.
    config['joint_name_drop_prob'] = 0.0
    config['rest_geom_joint_drop_prob'] = 0.0
    model = LatentMotionDenoiser(**config, action_conditioning=args.action_conditioning).to(device)
    print('Skeleton read gates: enabled (initial value 0.1)')
    if resume:
        model.load_state_dict(resume['model'], strict=True)
        if args.action_label_cond:
            model.validate_loaded_action_conditioning(resume['action_conditioning'], str(path))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp_dtype == 'fp16')
    start = resume['step'] if resume else 0
    if resume:
        optimizer.load_state_dict(resume['optimizer'])
        scaler.load_state_dict(resume['grad_scaler'])
    average = copy.deepcopy(model).eval().requires_grad_(False) if args.use_ema else None
    if resume and average is not None:
        average.load_state_dict(resume['model_ema'])
    scheduler = create_lr_scheduler(optimizer, args, start)
    if resume:
        torch.set_rng_state(resume['torch_rng'])
        random.setstate(resume['python_rng']); np.random.set_state(resume['numpy_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(resume['cuda_rng'], device)
    diffusion_config = dict(steps=args.diffusion_steps, noise_schedule=args.noise_schedule,
                            shift=args.noise_schedule_shift)
    loss_weights = validate_sensitivity_loss(latent_loss, norm.mean.shape).to(device)
    diffusion = create_latent_diffusion(**diffusion_config, loss_weights=loss_weights)
    forward_model, target_encoder = training_forwards(model, ae, norm, args.compile)
    graph_steps = (EagerCudaGraphSteps(
        model, diffusion, optimizer,
        {'fp32': None, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.amp_dtype], device, scaler)
        if args.compile == 'cuda-graph' else None)
    (save / 'args.json').write_text(json.dumps(current_args, indent=2), encoding='utf-8')
    print(f'Frozen AE: {binding["path"]} ({binding["weights"]}); latent={ae.slots}x{ae.latent_dim}; '
          f'{args.latent_mode} statistics: {norm.count} frame tokens; resume step={start}')
    print('Online encoding; fixed 60 valid frames; no motion geometry training losses or editing.')
    print(f'LR warmup: {args.lr_warmup_steps} optimizer steps; current LR={optimizer.param_groups[0]["lr"]:.6g}')
    print(f'Global latent normalization; decoder-sensitivity weighted x0 MSE; '
          f'weights mean={float(loss_weights.mean()):.4f}, '
          f'range=[{float(loss_weights.min()):.4f}, {float(loss_weights.max()):.4f}]')
    if resume:
        print('Resume restores optimizer/model RNG; the shared prefetch data stream restarts.')
    tb_dir = save / datetime.now().strftime('run_%Y%m%d_%H%M%S')
    writer = SummaryWriter(str(tb_dir), purge_step=start + 1 if resume else None)
    tqdm.write(f'[INFO] TensorBoard log_dir: {tb_dir}')
    dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16
    try:
        with (save / 'training.jsonl').open('a', encoding='utf-8') as log:
            for step in tqdm(range(start + 1, args.num_steps + 1),
                             desc='Latent diffusion', unit='step', smoothing=0.05):
                model.train()
                if args.compile in ('default', 'reduce-overhead'):
                    torch.compiler.cudagraph_mark_step_begin()
                x, y = training_batch_to_device(next(batches), device)
                target = target_encoder(x, y)
                t = torch.randint(diffusion.num_timesteps, (len(x),), device=device)
                noise = torch.randn_like(target)
                for retry in range(17):
                    if graph_steps is not None:
                        loss = graph_steps.step(target, t, noise, y)
                    else:
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device.type, dtype=dtype, enabled=args.amp_dtype != 'fp32'):
                            terms = diffusion.training_losses(forward_model, target, t, {'y': y}, noise)
                            loss = terms['loss'].mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Nonfinite latent loss at step {step}')
                    if graph_steps is not None:
                        grad_norm, updated = graph_steps.optimizer_step()
                    else:
                        grad_norm, updated = scaled_optimizer_step(loss, model, optimizer, scaler)
                    if updated:
                        break
                else:
                    raise FloatingPointError('Repeated fp16 gradient overflow')
                if average is not None:
                    update_autoencoder_ema(average, model, args.ema_rate)
                scheduler.step()
                if step % args.log_interval == 0 or step == start + 1:
                    metrics = (graph_steps.last_metrics if graph_steps is not None else
                               {key: value.detach().mean() for key, value in terms.items()})
                    record = dict(step=step, **{key: float(value) for key, value in metrics.items()},
                                  grad_norm=float(grad_norm),
                                  lr=optimizer.param_groups[0]['lr'], amp_overflow_retries=retry)
                    log.write(json.dumps(record) + '\n'); log.flush()
                    tqdm.write(str(record))
                    for key, value in record.items():
                        if key != 'step':
                            writer.add_scalar('train/' + key, value, step)
                if val_loader is not None and (step % args.eval_interval == 0 or step == args.num_steps):
                    report = evaluate_validation(
                        average if average is not None else model, ae, norm, diffusion, val_loader, device)
                    if report is not None:
                        (save / 'validation.json').write_text(json.dumps(dict(step=step, split='val', **report), indent=2))
                        for key, value in report.items():
                            writer.add_scalar('validation/' + key, value, step)
                        tqdm.write(f'validation: {report}')
                if step % args.save_interval == 0 or step == args.num_steps:
                    payload = dict(kind=LATENT_CHECKPOINT_VERSION, step=step, args=current_args,
                                   model_config=model.config, model=model.state_dict(),
                                   model_ema=average.state_dict() if average is not None else None,
                                   optimizer=optimizer.state_dict(), grad_scaler=scaler.state_dict(),
                                   ae_binding=binding, normalization=norm.metadata(), cond_sha256=cond_hash,
                                   latent_analysis=analysis, latent_loss=latent_loss,
                                   species_sha256=species_hash, time_contract=TIME_CONTRACT,
                                   action_conditioning=model.action_conditioning_metadata,
                                   diffusion_config=diffusion_config, prediction='standardized_z0',
                                   capabilities=['generate'] + (['action'] if model.action_label_cond else [])
                                   + (['species'] if model.species_cond else []),
                                   motion_names=motion_names,
                                   eval_split='val', torch_rng=torch.get_rng_state(),
                                   python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                                   cuda_rng=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)
                    temporary = save / 'latent_diffusion.tmp.pt'
                    torch.save(payload, temporary); temporary.replace(path)
    finally:
        writer.close()


if __name__ == '__main__':
    main()
