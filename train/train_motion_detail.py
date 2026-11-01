"""Independent all-band calibration; the backbone is never loaded or optimized."""
import argparse
import copy
import json
import random
import shutil
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from model.motion_detail_refiner import MotionDetailRefiner, RefinerConfig
from sample.detail_refinement import DETAIL_SCHEMA, file_hash, tree_to, normalize_backbone_contract
from tools.cache_motion_detail import repeat_context
from train.detail_objectives import (detail_features, feature_scale, preservation_losses,
                                     preservation_reference, statistics_loss, empirical_target)
from train.detail_runtime import BatchedDetailRuntime, PreparedDetailBatch
from train.detail_cuda_graph import CudaGraphDetailTrainer
from train.detail_gradients import DetailGradientMonitor
from train.detail_metrics import preservation_metrics
from train.detail_checkpoint import (GroupSchedule, restore_rng, rng_state, validate_resume,
                                     learning_rate_factor)


# These fields drive Python control flow. Keeping them on CUDA causes a device
# synchronization for every scalar/joint read in every forward/loss call.
HOST_CONTEXT = {'n_joints', 'lengths', 'is_loop', 'resample_speed_cond', 'parents',
                'loop_tile_count', 'detail_max_hz'}


def training_context(y, count):
    context = repeat_context(y, count)
    for key in HOST_CONTEXT & context.keys():
        if torch.is_tensor(context[key]):
            context[key] = context[key].detach().cpu().tolist()
        elif isinstance(context[key], (list, tuple)):
            context[key] = [v.detach().cpu().tolist() if torch.is_tensor(v) else v
                            for v in context[key]]
    return context


def load_groups(folder, device, progress=False):
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != DETAIL_SCHEMA or manifest.get('target_split') != 'train':
        raise ValueError('Expected a versioned train-split cache')
    if not manifest.get('frozen_state_verified') or len(manifest['groups']) != manifest.get('expected_groups', len(manifest['groups'])):
        raise ValueError('Cache is incomplete; finish generation with --resume before training/evaluation')
    if set(manifest['train_seeds']) & set(manifest['validation_seeds']):
        raise ValueError('Backbone seed leakage')
    groups = []
    for record in tqdm(manifest['groups'], desc='detail cache: verify/load', disable=not progress):
        path = folder / record['file']
        if path.parent.resolve() != folder.resolve() or file_hash(path) != record['sha256']:
            raise ValueError('Invalid or modified cache group')
        group = torch.load(path, map_location='cpu', weights_only=False)
        for key in ('gt', 'train_base', 'validation_base'):
            if group[key].shape[0] < 1 or not torch.isfinite(group[key]).all():
                raise ValueError(f'Invalid cache tensor: {key}')
        host = {k: v for k, v in group['y'].items() if k in HOST_CONTEXT}
        group = tree_to(group, device)
        group['y'].update(host)
        groups.append(group)
    if not groups:
        raise ValueError('Cache is empty')
    return manifest, groups


@torch.no_grad()
def prepare_groups(groups, config, progress=False, batched=False, prepare_validation=True):
    scales = []
    for group in tqdm(groups, desc='detail cache: prepare features', disable=not progress):
        gt = group['gt']
        group['target_features'] = empirical_target(
            detail_features(gt, training_context(group['y'], len(gt)), config),
            group.get('clips'))
        # Known matched inputs train the shared network's identity behaviour.
        # These are train-split inputs, never validation examples or waveforms
        # paired against independently generated backbone samples.
        group['matched_base'] = gt.detach()
        scale = feature_scale(group['target_features'])
        scales.append(scale)
        group['_contexts'], group['_references'] = {}, {}
        base_keys = ('train_base', 'matched_base', 'validation_base') if prepare_validation else (
                    'train_base', 'matched_base')
        for key in base_keys:
            base = group[key]
            y = training_context(group['y'], len(base))
            group['_contexts'][key] = y
            group['_references'][key] = preservation_reference(base, y, config)
        if prepare_validation:
            group['_base_stat'] = statistics_loss(
                detail_features(group['validation_base'], group['_contexts']['validation_base'], config),
                group['target_features'], scale)
        if batched:
            group['_batch'] = PreparedDetailBatch(group['train_base'], group['_contexts']['train_base'],
                                                  config, group['_references']['train_base'])
            # The batch owns stacked references; do not retain per-row duplicates.
            del group['_references']['train_base']
            group['_matched_batch'] = PreparedDetailBatch(group['matched_base'], group['_contexts']['matched_base'],
                                                  config, group['_references']['matched_base'], matched=True,
                                                  clips=group.get('clips'))
            del group['_references']['matched_base']
        group['_identity_crops'] = []
        y = group['_contexts']['matched_base']
        from model.motion_detail_refiner import sample_context
        _, t, _, dt, _ = sample_context(y, 0, config)
        lengths = sorted({max(1, int(t * ratio)) for ratio in (2/3, 1/2)}, reverse=True)
        for length in lengths:
            if length == t:
                continue
            start = (t - length) // 2
            crop = gt[..., start:start + length].contiguous()
            yc = copy.deepcopy(y)
            yc.update(lengths=[length] * len(gt), is_loop=[False] * len(gt),
                resample_speed_cond=[(dt * config.reference_fps * (length - 1) + 1) / length] * len(gt),
                loop_tile_count=[1] * len(gt))
            if torch.is_tensor(yc.get('mask')):
                yc['mask'] = yc['mask'][..., start:start + length]
            target = empirical_target(detail_features(crop, yc, config),
                                      group.get('clips'))
            norm = feature_scale(target)
            ref = preservation_reference(crop, yc, config)
            batch = PreparedDetailBatch(crop, yc, config, ref, matched=True,
                                         clips=group.get('clips')) if batched else None
            group['_identity_crops'].append(dict(base=crop, y=yc, target=target,
                                                 scale=norm, reference=ref, batch=batch,
                                                 clips=group.get('clips')))
        # Keep GT only as the explicit matched-input anchor, not a paired
        # reconstruction target for independently generated backbone samples.
        del group['gt']
    return scales


def group_loss(net, group, base_key, detail_seed, scale, runtime=None):
    if runtime is not None and base_key in ('train_base', 'matched_base'):
        batch_key = '_batch' if base_key == 'train_base' else '_matched_batch'
        return runtime.loss(group[batch_key], group['target_features'], scale, detail_seed)
    base = group[base_key].detach()
    y = group.get('_contexts', {}).get(base_key)
    if y is None:
        y = training_context(group['y'], len(base))
    out, delta = net(base, y, seed=detail_seed)
    target = group['target_features']
    losses = preservation_losses(out, base, delta, y, net.config,
                                 reference=group.get('_references', {}).get(base_key), matched=base_key == 'matched_base')
    features = detail_features(out, y, net.config)
    if base_key == 'matched_base':
        features = empirical_target(features, group.get('clips'))
    losses['stat'] = statistics_loss(features, target, scale)
    return losses, out, delta


def matched_identity_loss(net, group, seed, scale, runtime=None):
    losses, out, delta = group_loss(net, group, 'matched_base', seed, scale, runtime)
    # Same statistical target plus explicit full-band minimal-change pressure.
    value = losses['stat'] + losses['keep'] + losses['geo'] + .01 * losses['reg']
    # Always retain the full-window anchor as well as open-window augmentations.
    for i, crop in enumerate(group.get('_identity_crops', ())):
        crop_seed = seed + 1000 + i * 100
        if runtime is not None:
            extra = crop['batch'].loss(runtime, crop['target'], crop['scale'], crop_seed,
                                        materialize=False)
        else:
            co, cd = net(crop['base'], crop['y'], seed=crop_seed)
            extra = preservation_losses(co, crop['base'], cd, crop['y'], net.config,
                                         crop['reference'], matched=True)
            extra['stat'] = statistics_loss(empirical_target(detail_features(co, crop['y'], net.config),
                crop['clips']),
                crop['target'], crop['scale'])
        value = value + .5 * (extra['stat'] + extra['keep'] + extra['geo'] + .01 * extra['reg'])
    return value, losses, out, delta


@torch.no_grad()
def evaluate(net, groups, scales, seed, progress=False):
    net.eval()
    rows = []
    for i, group in enumerate(tqdm(groups, desc='detail validation', disable=not progress, leave=False)):
        losses, out, delta = group_loss(net, group, 'validation_base', seed, scales[i])
        y = group.get('_contexts', {}).get('validation_base')
        if y is None:
            y = training_context(group['y'], len(out))
        other, _ = net(group['validation_base'], y, seed=seed + 100000)
        identity, matched_losses, matched_out, matched_delta = matched_identity_loss(net, group, seed, scales[i])
        physical = preservation_metrics(out, group['validation_base'], y, net.config)
        matched_y = group.get('_contexts', {}).get('matched_base')
        if matched_y is None:
            matched_y = training_context(group['y'], len(matched_out))
        matched_physical = preservation_metrics(matched_out, group['matched_base'], matched_y, net.config)
        base_stat = group.get('_base_stat')
        if base_stat is None:
            base_stat = statistics_loss(detail_features(group['validation_base'], y, net.config),
                                        group['target_features'], scales[i])
        rows.append({'condition': json.loads(group['key']), **{k: float(v) for k, v in losses.items()},
            **physical, **{f'matched_{k}': v for k, v in matched_physical.items()},
            'identity': float(identity), 'matched_stat': float(matched_losses['stat']),
            'matched_delta_rms': float(matched_delta.square().mean().sqrt()),
            'base_stat': float(base_stat),
            'detail_seed_change_rms': float((out - other).square().mean().sqrt()),
            'max_residual_rad': float(torch.linalg.vector_norm(delta, dim=2).max()),
            'root_max_error': float((out[:, 0] - group['validation_base'][:, 0]).abs().max()),
            'independent_gt_clips': len(set(group['clips']))})
    net.train()
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir', required=True)
    p.add_argument('--save-dir', required=True)
    p.add_argument('--resume', nargs='?', const='latest', default='auto',
                   help='Default: continue save-dir/refiner_latest.pt when it exists, else train from scratch; '
                        'pass a checkpoint path to resume elsewhere; budget is the total per group')
    p.add_argument('--no-resume', dest='resume', action='store_const', const=None,
                   help='Ignore an existing checkpoint and train from scratch (requires an empty save directory)')
    p.add_argument('--updates-per-group', type=int, default=200)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--lr-schedule', choices=['constant', 'cosine'], default='cosine')
    p.add_argument('--warmup-updates-per-group', type=int, default=2)
    p.add_argument('--min-lr-ratio', type=float, default=.01)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--cutoff-hz', type=float, default=4.5,
                   help='Statistics band boundary in Hz; does not restrict residual frequencies')
    p.add_argument('--max-angle-rad', type=float, default=.3)
    p.add_argument('--output-max-hz', type=float, default=None)
    p.add_argument('--lambda-stat', type=float, default=1.)
    p.add_argument('--lambda-keep', type=float, default=1.,
                   help='Uniform full-band correction cost relative to the frozen base')
    p.add_argument('--lambda-geo', type=float, default=1.)
    p.add_argument('--lambda-reg', type=float, default=.01)
    p.add_argument('--lambda-identity', type=float, default=1.,
                   help='Train-only matched-input statistics and minimum-correction anchor')
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--save-every', type=int, default=5000)
    p.add_argument('--eval-every', type=int, default=0,
                   help='Full held-out seed validation interval; 0 disables validation including final step')
    p.add_argument('--log-every', type=int, default=10,
                   help='TensorBoard/console interval and JSONL flush interval (JSONL keeps every step)')
    p.add_argument('--tb-dir', default=None, help='Default: save-dir/tensorboard')
    p.add_argument('--no-progress', action='store_true', help='Disable progress bars for redirected logs')
    p.add_argument('--serial-groups', action='store_true',
                   help='Use the original per-seed reference path instead of batched group training')
    p.add_argument('--compile', nargs='?', const='default', default=None,
                   choices=['default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'],
                   help='Compile batched neural/statistics regions with Inductor; first shapes require warmup')
    p.add_argument('--no-compile', dest='compile', action='store_const', const=None,
                   help='Disable compilation, including a compile setting inherited by --resume')
    p.add_argument('--compile-dynamic', action=argparse.BooleanOptionalAction, default=True,
                   help='Use symbolic shapes to limit recompilation across different rigs/windows')
    p.add_argument('--cuda-graphs', action=argparse.BooleanOptionalAction, default=True,
                   help='Capture complete batched CUDA forward/backward steps (default on CUDA); '
                        'takes precedence over partial --compile regions')
    p.add_argument('--cuda-graph-max-groups', type=int, default=256,
                   help='Maximum cached group graphs with a shared activation pool; overflow uses eager')
    p.add_argument('--acceptance', default=None,
                   help='Predeclared thresholds; saved before the first update')
    a = p.parse_args(argv)
    if a.resume == 'auto':
        a.resume = 'latest' if (Path(a.save_dir) / 'refiner_latest.pt').is_file() else None
    requested_resume = a.resume
    resume = None
    resume_path = None
    if requested_resume:
        resume_path = Path(a.save_dir) / 'refiner_latest.pt' if requested_resume == 'latest' else Path(requested_resume)
        resume = torch.load(resume_path, map_location='cpu', weights_only=False)
        if resume.get('architecture', {}).get('calibration_version') != 3:
            raise ValueError('Unsupported refiner checkpoint; only calibration_version=3 can resume')
        # Retain the checkpoint configuration unless a CLI option explicitly overrides it.
        inherited = {k: v for k, v in resume.get('training_args', {}).items()
                     if k in {action.dest for action in p._actions}
                     and k not in ('cache_dir', 'save_dir', 'resume', 'tb_dir', 'acceptance')}
        inherited.setdefault('lr_schedule', 'constant')
        p.set_defaults(**inherited)
        a = p.parse_args(argv)
        a.resume = requested_resume
    if min(a.updates_per_group, a.save_every, a.log_every) < 1 or a.eval_every < 0 or a.lr <= 0 or min(a.lambda_stat, a.lambda_keep, a.lambda_geo, a.lambda_reg, a.lambda_identity) < 0:
        p.error('Invalid budget/loss/optimizer options')
    if a.compile and a.serial_groups:
        p.error('--compile requires batched group training; remove --serial-groups')
    if a.cuda_graph_max_groups < 1:
        p.error('--cuda-graph-max-groups must be positive')
    if a.warmup_updates_per_group < 0 or not 0 <= a.min_lr_ratio <= 1:
        p.error('Invalid learning rate warmup/minimum ratio')
    save = Path(a.save_dir)
    save.mkdir(parents=True, exist_ok=True)
    if resume is None and any(save.iterdir()):
        raise ValueError('No refiner_latest.pt to resume; use an empty save directory (or remove the incomplete run)')
    if resume is not None and resume_path.parent.resolve() != save.resolve() and any(save.iterdir()):
        raise ValueError('Resuming into another save directory requires an empty destination')
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    started = time.perf_counter()
    print(f'[detail train] loading and verifying cache on {a.device}', flush=True)
    manifest, groups = load_groups(a.cache_dir, a.device, progress=not a.no_progress)
    first = groups[0]['y']
    config = RefinerConfig(
        width=a.width, layers=a.layers, name_dim=first['joints_names_embs'].shape[-1],
        action_dim=first['detail_action'].shape[-1], cutoff_hz=a.cutoff_hz,
        max_angle_rad=a.max_angle_rad, reference_fps=manifest['reference_fps'],
        output_max_hz=a.output_max_hz)
    net = MotionDetailRefiner(config).to(a.device)
    scales = prepare_groups(groups, config, progress=not a.no_progress, batched=not a.serial_groups,
                            prepare_validation=bool(a.eval_every))
    use_cuda_graphs = a.cuda_graphs and not a.serial_groups and next(net.parameters()).is_cuda
    runtime = None if a.serial_groups else BatchedDetailRuntime(
        net, None if use_cuda_graphs else a.compile, a.compile_dynamic)
    warm_shapes = set()
    if use_cuda_graphs:
        print('[detail train] CUDA Graphs: full batched forward/backward, shared activation pool; '
              'optimizer/checks outside capture; partial --compile regions superseded', flush=True)
    elif a.compile:
        print(f'[detail train] torch.compile enabled: mode={a.compile}, '
              f'dynamic={a.compile_dynamic}, fullgraph=True (tensor regions); '
              'first forward/backward for a new shape can take minutes', flush=True)
    optimizer = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=.01)
    counts = [0] * len(groups)
    step = 0
    weights = {'stat': a.lambda_stat, 'keep': a.lambda_keep, 'geo': a.lambda_geo,
               'reg': a.lambda_reg, 'identity': a.lambda_identity}
    manifest_hash = file_hash(Path(a.cache_dir) / 'manifest.json')
    if resume is not None:
        validate_resume(resume, manifest, manifest_hash, net.architecture(), weights,
                        [g['key'] for g in groups], a.updates_per_group)
        if a.seed != resume['training_args']['seed']:
            raise ValueError('Cannot resume: seed mismatch')
        net.load_state_dict(resume['refiner'])
        optimizer.load_state_dict(resume['optimizer'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = a.lr
        scales = [s.to(a.device) for s in resume['normalization']]
        counts, step = list(resume['group_updates']), resume['step']
        restore_rng(resume.get('rng_state'))
    schedule = GroupSchedule(len(groups), a.updates_per_group, a.seed,
                             state=resume.get('sampler_state') if resume else None,
                             legacy_step=step if resume and 'sampler_state' not in resume else 0)
    if resume and schedule.counts() != counts:
        raise ValueError('Cannot resume: sampler progress differs from group update counts')
    graph_trainer = CudaGraphDetailTrainer(net, weights, a.cuda_graph_max_groups) if use_cuda_graphs else None
    mixer = None if use_cuda_graphs else DetailGradientMonitor(net)
    print('[detail train] weighted objective gradients with matched-input identity', flush=True)
    (save / 'args.json').write_text(json.dumps(vars(a), indent=2), encoding='utf-8')
    default_acceptance = Path(__file__).resolve().parents[1] / 'docs' / 'motion_calibration_acceptance.json'
    acceptance_path = resume_path.with_name('acceptance.json') if resume_path else Path(a.acceptance or default_acceptance)
    acceptance = json.loads(acceptance_path.read_text(encoding='utf-8'))
    if acceptance.get('calibration_version') != config.calibration_version:
        raise ValueError('Acceptance thresholds must match calibration_version')
    (save / 'acceptance.json').write_text(json.dumps(acceptance, indent=2), encoding='utf-8')
    total = a.updates_per_group * len(groups)
    lr_total = total
    if resume and a.lr_schedule == 'cosine' and resume['training_args'].get('lr_schedule') == 'cosine':
        # Extending training must not silently reheat a completed cosine run.
        lr_total = resume.get('lr_schedule_total_steps',
                              resume['training_args']['updates_per_group'] * len(groups))
    print(f'[detail train] ready: {len(groups)} groups, {total} updates; '
          f'prepare={time.perf_counter() - started:.1f}s; '
          f'save every {a.save_every}, validate every {a.eval_every}; '
          f'TensorBoard={a.tb_dir or save / "tensorboard"}', flush=True)
    train_started = time.perf_counter()
    initial_step = step
    interval_seconds = 0.
    interval_steps = 0
    metric_epoch, epoch_records = None, []
    log_path = save / 'training.jsonl'
    if resume is not None:
        print(f'[detail train] resumed {resume_path}: step={step}/{total}, '
              f'group updates={min(counts)}..{max(counts)}', flush=True)
        if log_path.exists():
            temporary = log_path.with_suffix('.jsonl.tmp')
            with log_path.open(encoding='utf-8') as source, temporary.open('w', encoding='utf-8') as target:
                for line in source:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        break  # A killed process can leave an incomplete trailing record.
                    if row['step'] <= step:
                        target.write(line)
            temporary.replace(log_path)
    with SummaryWriter(log_dir=a.tb_dir or str(save / 'tensorboard'),
                       purge_step=step + 1 if resume else None) as writer, \
            log_path.open('a' if resume else 'w', encoding='utf-8') as log, \
            tqdm(total=total, initial=step, desc='detail train', unit='update', disable=a.no_progress) as progress:
        for epoch, i, seed in schedule:
            current_lr = a.lr * learning_rate_factor(step + 1, lr_total, len(groups),
                a.lr_schedule, a.warmup_updates_per_group, a.min_lr_ratio)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            if a.compile and not use_cuda_graphs:
                torch.compiler.cudagraph_mark_step_begin()
            signature = (tuple(groups[i]['train_base'].shape), groups[i]['_batch'].loop) if a.compile and not use_cuda_graphs else None
            warming = a.compile and not use_cuda_graphs and signature not in warm_shapes
            if warming:
                tqdm.write(f'[detail compile] first forward/backward: shape={signature}; compiling/warming...')
            step_started = time.perf_counter()
            if graph_trainer is not None:
                metrics = graph_trainer.backward(groups[i], scales[i], seed)
                loss = metrics[0]
                losses = dict(zip(graph_trainer.keys, metrics[1:1 + len(graph_trainer.keys)]))
                diagnostics = dict(zip(graph_trainer.gradient_keys, metrics[1 + len(graph_trainer.keys):]))
            else:
                # Training seeds [0,2**31); validation seeds use a disjoint range.
                losses, _, _ = group_loss(net, groups[i], 'train_base', seed, scales[i], runtime=runtime)
                losses['identity'] = (matched_identity_loss(net, groups[i], seed, scales[i], runtime)[0]
                                      if weights['identity'] else losses['stat'].new_zeros(()))
                loss = sum(weights[k] * v for k, v in losses.items())
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite refiner loss')
            if graph_trainer is None:
                main_loss = sum(weights[k] * v for k, v in losses.items() if k != 'identity')
                identity_loss = weights['identity'] * losses['identity'] if weights['identity'] else None
                diagnostics = mixer.backward(main_loss, identity_loss)
            norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            counts[i] += 1
            step += 1
            # One host transfer for the scalar metrics rather than one sync per loss.
            logged = {**losses, **diagnostics}
            keys = list(logged)
            values = torch.stack([loss.detach(), norm.detach()] +
                                 [logged[k].detach() for k in keys]).cpu().tolist()
            record = {'step': step, 'group': i, 'updates': counts[i], 'detail_seed': seed,
                      'lr': current_lr,
                      'loss': values[0], 'grad_norm_before_clip': values[1],
                      **dict(zip(keys, values[2:]))}
            if metric_epoch != epoch:
                metric_epoch, epoch_records = epoch, []
            epoch_records.append(record)
            # Compare complete condition sweeps, not unrelated instantaneous losses.
            # A resumed partial sweep is intentionally omitted.
            if len(epoch_records) == len(groups):
                for key in ('loss', 'stat', 'identity', 'keep', 'geo', 'reg',
                            'grad_norm_before_clip'):
                    average = sum(r[key] for r in epoch_records) / len(groups)
                    writer.add_scalar(f'train/epoch_{key}', average, step)
                    record[f'epoch_mean_{key}'] = average
                writer.add_scalar('train/epoch_clip_fraction', sum(
                    r['grad_norm_before_clip'] > 1 for r in epoch_records) / len(groups), step)
            step_seconds = time.perf_counter() - step_started
            if warming:
                warm_shapes.add(signature)
                tqdm.write(f'[detail compile] first shape step finished in {step_seconds:.2f}s')
            record['step_seconds'] = step_seconds
            interval_seconds += step_seconds
            interval_steps += 1
            log.write(json.dumps(record) + '\n')
            progress.update(1)
            if step % a.log_every == 0 or step == total:
                log.flush()
                writer.add_scalar('train/loss', record['loss'], step)
                writer.add_scalar('train/lr', current_lr, step)
                writer.add_scalar('train/grad_norm', record['grad_norm_before_clip'], step)
                writer.add_scalar('train/grad_clip_scale', min(
                    1., 1. / (record['grad_norm_before_clip'] + 1e-6)), step)
                for key in weights:
                    writer.add_scalar(f'train/{key}', record[key], step)
                for key in DetailGradientMonitor.metric_keys:
                    writer.add_scalar(f'train/{key}', record[key], step)
                status = {'epoch': f'{epoch + 1}/{a.updates_per_group}', 'group': i,
                          'loss': f'{record["loss"]:.5g}', 'grad': f'{values[1]:.4g}',
                          'it/s': f'{interval_steps / max(interval_seconds, 1e-9):.1f}'}
                progress.set_postfix(status)
                if a.no_progress:
                    elapsed = time.perf_counter() - train_started
                    print(f'[detail train] {step}/{total} ({step / total:.1%}) '
                          f'elapsed={elapsed:.1f}s ETA={elapsed / (step-initial_step) * (total-step):.1f}s '
                          f'{status}', flush=True)
                interval_seconds, interval_steps = 0., 0
            if step % a.save_every == 0 or step == total:
                tqdm.write(f'[detail train] step={step}: saving checkpoint...')
                save_started = time.perf_counter()
                payload = {'schema': DETAIL_SCHEMA, 'contract': normalize_backbone_contract(manifest['contract']),
                    'composition_contract': config.composition_contract,
                    'architecture': net.architecture(), 'refiner': net.state_dict(),
                    'optimizer': optimizer.state_dict(), 'step': step, 'group_updates': counts,
                    'normalization': [s.cpu() for s in scales], 'conditions': [g['key'] for g in groups],
                    'cache_manifest_sha256': manifest_hash,
                    'loss_weights': weights, 'training_args': vars(a),
                    'lr_schedule_total_steps': lr_total,
                    'sampler_state': schedule.state_dict(), 'rng_state': rng_state()}
                checkpoint = save / f'refiner{step:09d}.pt'
                temporary = checkpoint.with_suffix('.pt.tmp')
                torch.save(payload, temporary)
                temporary.replace(checkpoint)
                latest = save / 'refiner_latest.pt.tmp'
                shutil.copyfile(checkpoint, latest)
                latest.replace(save / 'refiner_latest.pt')
                tqdm.write(f'[detail train] checkpoint saved ({time.perf_counter() - save_started:.2f}s)')
            if a.eval_every and (step % a.eval_every == 0 or step == total):
                tqdm.write(f'[detail train] step={step}: validating {len(groups)} groups...')
                eval_started = time.perf_counter()
                rows = evaluate(net, groups, scales, 2**40, progress=not a.no_progress)
                (save / f'validation{step:09d}.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
                writer.add_scalar('validation/loss', sum(
                    sum(weights[k] * r[k] for k in weights) for r in rows) / len(rows), step)
                for key in rows[0]:
                    if key in ('condition', 'independent_gt_clips'):
                        continue
                    values = [r[key] for r in rows]
                    writer.add_scalar(f'validation/{key}', sum(values) / len(rows), step)
                    writer.add_scalar(f'validation/{key}_max', max(values), step)
                tqdm.write(f'[detail train] step={step}: validation complete '
                           f'({time.perf_counter() - eval_started:.1f}s)')
            if step % a.log_every == 0 or (a.eval_every and step % a.eval_every == 0) or step == total:
                writer.flush()
    print(f'[detail train] complete: {step}/{total} updates; '
          f'elapsed={time.perf_counter() - train_started:.1f}s', flush=True)


if __name__ == '__main__':
    main()
