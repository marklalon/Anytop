"""Cache real frozen-backbone samples and train-split conditional GT windows.

Run from Anytop: python -m tools.cache_motion_detail --help
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from data_loaders.get_data import get_dataset
from data_loaders.tensors import truebones_batch_collate
from data_loaders.truebones.truebones_utils.action_label_conditioning_contract import ActionConditioningBundle
from sample.detail_refinement import (attach_detail_action, backbone_contract, decode_batch,
                                      tree_to, file_hash, DETAIL_SCHEMA, normalize_backbone_contract)
from sample.generation_runtime import prepare_generation_runtime
from sample.conditioning import _wrap_action_label_cfg
from sample.sampling import _sample_batch
from utils.fixseed import fixseed
from utils.parser_util import generate_args


DEFAULT_BACKBONE = 'save/zoo_locomotion_tiny_v67/model000010000.pt'


def format_duration(seconds):
    if seconds is None:
        return 'estimating...'
    hours, remainder = divmod(max(0, int(seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def progress_status(completed, total, elapsed, generated):
    remaining = total - completed
    # Resumed groups did not consume this session's generation time.
    eta = 0 if remaining == 0 else elapsed / generated * remaining if generated else None
    return (f'{completed}/{total} ({100 * completed / total:.1f}%) '
            f'elapsed={format_duration(elapsed)} ETA={format_duration(eta)}')


def frozen_state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def repeat_context(y, count):
    result = {}
    for key, value in y.items():
        if torch.is_tensor(value) and value.ndim and value.shape[0] == 1:
            result[key] = value.expand(count, *value.shape[1:]).clone()
        elif isinstance(value, list) and len(value) == 1:
            result[key] = value * count
        else:
            result[key] = value
    return result


def write_manifest(folder, manifest):
    # Replace only after the complete JSON is durable enough to be read back.
    temporary = folder / 'manifest.json.tmp'
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(folder / 'manifest.json')


def source_plan_hash(groups):
    digest = hashlib.sha256()
    for key, group in groups.items():
        digest.update(json.dumps([key, group['clips']], ensure_ascii=False).encode())
        for gt in group['gt']:
            digest.update(str((tuple(gt.shape), gt.dtype)).encode())
            digest.update(gt.contiguous().numpy().tobytes())
    return digest.hexdigest()


def restore_progress(folder, expected):
    path = folder / 'manifest.json'
    if not path.is_file():
        raise ValueError('Cannot resume: no manifest.json; use a new empty cache directory')
    saved = json.loads(path.read_text(encoding='utf-8'))
    for key, value in expected.items():
        actual = saved.get(key)
        if key == 'contract' and isinstance(actual, dict):
            actual, value = normalize_backbone_contract(actual), normalize_backbone_contract(value)
        if key not in ('groups', 'frozen_state_verified') and actual != value:
            raise ValueError(f'Cannot resume: cache configuration/source mismatch: {key}')
    seen = set()
    for entry in saved['groups']:
        name = entry['file']
        if Path(name).name != name or name in seen:
            raise ValueError('Cannot resume: invalid or duplicate cache filename')
        seen.add(name)
        if not (folder / name).is_file() or file_hash(folder / name) != entry['sha256']:
            raise ValueError(f'Cannot resume: missing or corrupt completed cache file: {name}')
    return saved


def generate_bases(runtime, sampler, y, shape, seeds, batch_size, args, device):
    bases = []
    for start in range(0, len(seeds), batch_size):
        batch_seeds = seeds[start:start + batch_size]
        context = repeat_context(y, len(batch_seeds))
        generated = _sample_batch(
            runtime.diffusion, sampler, {'y': context}, runtime.sampling_method,
            (len(batch_seeds), *shape[1:]), args.ddim_eta, batch_seeds[0], device,
            autocast_dtype=torch.bfloat16 if runtime.amp_dtype == 'bf16' else None,
            row_seeds=batch_seeds, progress=True)
        bases.append(decode_batch(generated, context).cpu())
    return torch.cat(bases)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--resume', action='store_true', help='Validate and skip completed groups in an existing cache')
    parser.add_argument('--seed-batch-size', type=int, default=8, help='Seeds sampled together per condition; 1 uses minimum memory (default: 8)')
    parser.add_argument('--gt-batch-size', type=int, default=32, help='Maximum GT windows decoded together (default: 32)')
    parser.add_argument('--backbone', default=DEFAULT_BACKBONE)
    parser.add_argument('--max-clips', type=int, default=0, help='0: all eligible clips; no topology/time augmentation')
    parser.add_argument('--clip-filter', default='', help='substring of native namespaced clip ID')
    parser.add_argument('--train-seeds', type=int, nargs='+', default=[101, 102, 103, 104])
    parser.add_argument('--validation-seeds', type=int, nargs='+', default=[10001])
    parser.add_argument('--window-seed', type=int, default=17)
    options, generation_cli = parser.parse_known_args(argv)
    if options.seed_batch_size < 1 or options.gt_batch_size < 1:
        parser.error('Batch sizes must be positive')
    if set(options.train_seeds) & set(options.validation_seeds):
        parser.error('Train and held-out backbone seeds must be disjoint')
    if options.max_clips < 0 or len(set(options.train_seeds)) != len(options.train_seeds) or len(set(options.validation_seeds)) != len(options.validation_seeds):
        parser.error('Invalid clip budget or duplicate seeds')
    folder = Path(options.cache_dir)
    folder.mkdir(parents=True, exist_ok=True)
    if not options.resume and any(folder.iterdir()):
        raise ValueError('Use an empty cache directory, or --resume to validate and continue an existing cache')
    args = generate_args(['--model_path', options.backbone, *generation_cli])
    if args.reference_motion or args.action_label or getattr(args, 'refiner_path', ''):
        raise ValueError('Cache conditions come from the train loader; no reference/action/refiner override')
    runtime = prepare_generation_runtime(args)
    model = runtime.model.eval().requires_grad_(False)
    frozen_hash = frozen_state_hash(model)
    metadata = model.action_conditioning_metadata
    bundle = ActionConditioningBundle(word_embeddings=model.action_word_embeddings.cpu().numpy(),
        embedding_contract=metadata['embedding_contract'], conditioning_contract=metadata['conditioning_contract'], source=args.model_path)
    recorded = json.loads(Path(args.model_path).with_name('args.json').read_text())
    dataset = get_dataset(num_frames=recorded['num_frames'], split='train',
        objects_subset=recorded['objects_subset'], dataset_namespaces=recorded.get('dataset_namespaces'),
        action_group=args.action_group, action_label_cond=True, action_conditioning=bundle,
        cond_path=runtime.actual_cond_file, species_table=runtime.species_table,
        motion_speed_aug=1., loop_cond_prob=1., loop_tile_single_prob=1., min_length=recorded.get('min_length', 20))
    groups = {}
    clip_count = 0
    prepare_started = time.perf_counter()
    for index in range(len(dataset)):
        raw_name = dataset.motion_dataset.name_list[dataset.motion_dataset.pointer + index]
        if options.clip_filter and options.clip_filter.lower() not in raw_name.lower():
            continue
        fixseed(options.window_seed + index)
        sample = dataset[index]
        # Use the actual rig width, not the loader's corpus-wide padding budget.
        sample = list(sample)
        sample[9] = sample[0].shape[1]
        gt, cond = truebones_batch_collate([sample])
        y = cond['y']
        key = json.dumps({'species': y['object_type'][0], 'label': y['action_label'][0],
                          'loop': bool(y['is_loop'][0]), 'cycles': int(y['loop_tile_count'][0]),
                          'speed': float(y['resample_speed_cond'][0]), 'frames': gt.shape[-1]}, sort_keys=True)
        group = groups.setdefault(key, {'gt': [], 'clips': [], 'y': y})
        group['gt'].append(gt)
        group['clips'].append(raw_name)
        clip_count += 1
        if options.max_clips and clip_count >= options.max_clips:
            break
    if not groups:
        raise ValueError('No eligible training clips')
    contract = backbone_contract(args, runtime.actual_cond_file)
    manifest = {'schema': DETAIL_SCHEMA, 'contract': contract, 'backbone_path': str(Path(args.model_path).resolve()),
                'target_split': 'train', 'train_seeds': options.train_seeds,
                'validation_seeds': options.validation_seeds, 'window_seed': options.window_seed,
                'groups': [], 'time_rule': 'loop: speed/fps; open: (speed*T-1)/(T-1)/fps',
                'reference_fps': float(runtime.opt.fps)}
    manifest['frozen_state_sha256'] = frozen_hash
    manifest['source_plan_sha256'] = source_plan_hash(groups)
    manifest['expected_groups'] = len(groups)
    manifest['frozen_state_verified'] = False
    if options.resume:
        manifest = restore_progress(folder, manifest)
    completed = {entry['file'] for entry in manifest['groups']}
    print(f'[detail cache] {clip_count} clips, {len(groups)} conditions; '
          f'{len(completed)} completed; seed batch={options.seed_batch_size}; '
          f'prepare={time.perf_counter() - prepare_started:.1f}s', flush=True)
    manifest['frozen_state_verified'] = False
    write_manifest(folder, manifest)
    sampler = _wrap_action_label_cfg(model, args, {'action_label': '(per-row native labels)'})
    device = next(model.parameters()).device
    generation_started = time.perf_counter()
    generated_groups = 0
    print(f'[detail cache] {progress_status(len(completed), len(groups), 0, 0)}', flush=True)
    with torch.no_grad():
        for key, group in groups.items():
            name = hashlib.sha256(key.encode()).hexdigest()[:20]
            if f'{name}.pt' in completed:
                print(f'[detail cache] {name}: verified, skipped', flush=True)
                continue
            started = time.perf_counter()
            print(f'[detail cache] generating {name}: '
                  f'{progress_status(len(manifest["groups"]), len(groups), started - generation_started, generated_groups)}',
                  flush=True)
            y = attach_detail_action(model, tree_to(group['y'], device))
            bases = generate_bases(runtime, sampler, y, group['gt'][0].shape,
                options.train_seeds + options.validation_seeds, options.seed_batch_size, args, device)
            sampled = time.perf_counter()
            targets = []
            for start in range(0, len(group['gt']), options.gt_batch_size):
                chunk = group['gt'][start:start + options.gt_batch_size]
                gt_context = repeat_context(y, len(chunk))
                targets.append(decode_batch(torch.cat(chunk).to(device), gt_context).cpu())
            target = torch.cat(targets)
            payload = {'schema': DETAIL_SCHEMA, 'key': key, 'y': tree_to(y, 'cpu'),
                       'gt': target, 'clips': group['clips'],
                       'train_base': bases[:len(options.train_seeds)],
                       'validation_base': bases[len(options.train_seeds):]}
            path = folder / f'{name}.pt'
            temporary = path.with_suffix('.pt.tmp')
            torch.save(payload, temporary)
            temporary.replace(path)
            manifest['groups'].append({'file': path.name, 'sha256': file_hash(path),
                                       'condition': json.loads(key), 'independent_clips': len(set(group['clips']))})
            write_manifest(folder, manifest)
            elapsed = time.perf_counter() - started
            generated_groups += 1
            status = progress_status(len(manifest['groups']), len(groups),
                                     time.perf_counter() - generation_started, generated_groups)
            print(f'[detail cache] {status} {name}: '
                  f'{len(group["clips"])} GT clips, {len(bases)} generated seeds; '
                  f'sample={sampled - started:.2f}s total={elapsed:.2f}s '
                  f'({len(bases) / max(sampled - started, 1e-9):.2f} samples/s)', flush=True)
    if frozen_state_hash(model) != frozen_hash or any(p.grad is not None or p.requires_grad for p in model.parameters()):
        raise RuntimeError('Backbone isolation violation during caching')
    manifest['frozen_state_verified'] = True
    write_manifest(folder, manifest)
    status = progress_status(len(manifest['groups']), len(groups),
                             time.perf_counter() - generation_started, generated_groups)
    print(f'[detail cache] complete: {status}; backbone state verified', flush=True)


if __name__ == '__main__':
    main()
