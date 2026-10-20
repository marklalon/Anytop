"""Paired checkpoint denoising probe and within-model cross-limb ablations."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_loaders.get_data import get_dataset
from data_loaders.tensors import truebones_batch_collate
from data_loaders.truebones.data.dataset import EmptySplitError, load_action_conditioning
from data_loaders.truebones.truebones_utils.species_descriptor_table import load_checkpoint_species_descriptor_table
from utils.model_util import (bind_checkpoint_action_conditioning,
                              create_model_and_diffusion_general_skeleton,
                              load_checkpoint_weights, load_model)


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move(v, device) for v in value]
    return value


def load_checkpoint(path, device):
    args = argparse.Namespace(**json.loads((path.parent / 'args.json').read_text()))
    model, diffusion = create_model_and_diffusion_general_skeleton(args)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    state, _, metadata = load_checkpoint_weights(payload, str(path), prefer_ema=True)
    load_model(model, state)
    bind_checkpoint_action_conditioning(model, metadata, str(path))
    model.to(device)
    model.eval()
    return model, diffusion, args, 'EMA' if payload.get('model_avg') is not None else 'online'


@contextlib.contextmanager
def ablate(model, mode):
    restores = []
    for block in model.seqTransDecoder.cross_limb_blocks or []:
        if mode in ('bypass', 'norm_only'):
            original = block.forward
            block.forward = (lambda x, *a, **kw: x) if mode == 'bypass' else (
                lambda x, *a, _block=block, **kw: _block.norm_cl(x))
            restores.append(lambda b=block, f=original: setattr(b, 'forward', f))
        elif mode == 'no_temporal':
            original = block.temporal_attn.forward
            block.temporal_attn.forward = lambda q, *a, **kw: (torch.zeros_like(q), None)
            restores.append(lambda b=block, f=original: setattr(b.temporal_attn, 'forward', f))
        elif mode == 'no_cross_k':
            original = block.cross_k_scale.detach().clone()
            block.cross_k_scale.zero_()
            restores.append(lambda b=block, v=original: b.cross_k_scale.copy_(v))
    try:
        yield
    finally:
        for restore in reversed(restores):
            restore()


def paired_stats(base, other):
    # Resample clips, keeping their timesteps together; descriptive bootstrap CI.
    delta = np.asarray(other) - np.asarray(base)
    rng = np.random.default_rng(20261018)
    boot = np.mean(delta[rng.integers(0, len(delta), size=(4000, len(delta)))], axis=1)
    return dict(delta=float(delta.mean()), percent=float(100 * delta.mean() / np.mean(base)),
                ci95=np.quantile(boot, [.025, .975]).tolist(),
                clips_worse_percent=float(100 * np.mean(delta > 0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with_model', type=Path, default=Path('save/zoo_locomotion_tiny_v6/model000050000.pt'))
    parser.add_argument('--without_model', type=Path, default=Path('save/zoo_locomotion_tiny_v7/model000050000.pt'))
    parser.add_argument('--output', type=Path, default=Path('outputs/cross_limb_probe_v6_v7'))
    parser.add_argument('--samples', type=int, default=64, help='0 = all clips')
    parser.add_argument('--batch_size', type=int, default=4)
    cli = parser.parse_args()
    cli.output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(20261018)
    np.random.seed(20261018)
    random.seed(20261018)
    m6, diffusion, a6, weights6 = load_checkpoint(cli.with_model, device)
    m7, _, a7, weights7 = load_checkpoint(cli.without_model, device)
    assert m6.seqTransDecoder.cross_limb_blocks is not None
    assert m7.seqTransDecoder.cross_limb_blocks is None
    bundle = load_action_conditioning()
    for model in (m6, m7):
        np.testing.assert_array_equal(bundle.word_embeddings, model.action_word_embeddings.cpu().numpy())
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        cli.with_model.parent / 'cond.npy', cli.without_model.parent / 'cond.npy',
        cli.with_model.parent / 'species_descriptor_embs.npy', cli.without_model.parent / 'species_descriptor_embs.npy')}
    assert hashes[str(cli.with_model.parent / 'cond.npy')] == hashes[str(cli.without_model.parent / 'cond.npy')]
    assert hashes[str(cli.with_model.parent / 'species_descriptor_embs.npy')] == hashes[str(cli.without_model.parent / 'species_descriptor_embs.npy')]
    dataset_cond = Path(a6.cond_path)
    assert hashlib.sha256(dataset_cond.read_bytes()).hexdigest() == hashes[str(cli.with_model.parent / 'cond.npy')], 'Dataset cond differs from checkpoint snapshot'
    dataset_kwargs = dict(num_frames=a6.num_frames, cond_path=str(dataset_cond),
                          objects_subset=a6.objects_subset, action_group=a6.action_group,
                          action_label_cond=True, action_conditioning=bundle,
                          species_table=load_checkpoint_species_descriptor_table(cli.with_model, 'probe checkpoint'),
                          min_length=a6.min_length, loop_cond_prob=1., loop_tile_single_prob=0., motion_speed_aug=1.)
    split = 'val'
    try:
        dataset = get_dataset(split=split, **dataset_kwargs)
    except EmptySplitError:
        split = 'train'
        dataset = get_dataset(split=split, **dataset_kwargs)
    count = min(cli.samples or len(dataset), len(dataset))
    indices = sorted(np.random.default_rng(20261018).choice(len(dataset), count, replace=False).tolist())
    # Materialize once, before forwards, to give every model exactly the same crops.
    batches = [truebones_batch_collate([dataset[i] for i in indices[start:start + cli.batch_size]])
               for start in range(0, count, cli.batch_size)]
    print(f'PROBE device={device}, split={split}, sampled={count}/{len(dataset)}, weights={weights6}/{weights7}', flush=True)
    variants = ['v6_full', 'v6_bypass', 'v6_norm_only', 'v6_no_temporal', 'v6_no_cross_k', 'v7_full']
    timesteps = [5, 25, 50, 75, 95]
    results = {v: [] for v in variants}
    block_stats = {str(i): [] for i, _ in enumerate(m6.seqTransDecoder.cross_limb_blocks)}
    recording = [False]
    handles = []
    for i, block in enumerate(m6.seqTransDecoder.cross_limb_blocks):
        def hook(module, inputs, kwargs, output, index=i):
            if recording[0]:
                x = inputs[0]
                valid = ~(inputs[1] if len(inputs) > 1 else kwargs['joints_key_padding_mask'])
                mask = valid[None, :, :, None].expand_as(x)
                # Input/output difference includes the block's LayerNorm.
                block_stats[str(index)].append(float((output - x)[mask].square().mean().sqrt() / x[mask].square().mean().sqrt()))
        handles.append(block.register_forward_hook(hook, with_kwargs=True))
    with torch.inference_mode():
        for batch_index, (motion, cond) in enumerate(batches):
            x, cond = move(motion, device), move(cond, device)
            y = cond['y']
            batch_records = {v: [] for v in variants}
            for timestep in timesteps:
                t = torch.full((len(x),), timestep, dtype=torch.long, device=device)
                noise = torch.randn_like(x)
                outputs = {}
                for variant in variants:
                    model = m7 if variant.startswith('v7') else m6
                    mode = variant.removeprefix('v6_')
                    captured = []
                    handle = model.register_forward_hook(lambda mod, args, out: captured.append(out.detach()))
                    recording[0] = variant == 'v6_full'
                    with ablate(model, mode):
                        losses = diffusion.training_losses(model, x, t, model_kwargs={'y': dict(y)}, noise=noise)
                    recording[0] = False
                    handle.remove()
                    outputs[variant] = captured[0]
                    record = {k: v.cpu().tolist() for k, v in losses.items() if torch.is_tensor(v) and v.ndim == 1}
                    mask = y['joints_padding_mask'][:, :, :, 1, 1:]
                    diff = outputs[variant][:, :, :9] - outputs['v6_full'][:, :, :9]
                    diff_mse = diffusion.spatial_masked_l2(torch.zeros_like(diff), diff, mask, y['lengths'], y['n_joints'])
                    pred = outputs['v6_full'][:, :, :9]
                    pred_mse = diffusion.spatial_masked_l2(torch.zeros_like(pred), pred, mask, y['lengths'], y['n_joints'])
                    record['prediction_change_relative_rms'] = (diff_mse / pred_mse.clamp_min(1e-12)).sqrt().cpu().tolist()
                    batch_records[variant].append(dict(t=timestep, metrics=record))
            for variant in variants:
                for i in range(len(x)):
                    results[variant].append(dict(dataset_index=indices[batch_index * cli.batch_size + i],
                        motion_name=y.get('motion_name', ['?'] * len(x))[i],
                        draws=[dict(t=r['t'], **{k: v[i] for k, v in r['metrics'].items()}) for r in batch_records[variant]]))
            print(f'batch {batch_index + 1}/{len(batches)} complete', flush=True)
    for handle in handles:
        handle.remove()
    clip_means = {v: {k: [float(np.mean([d[k] for d in clip['draws']])) for clip in clips]
                       for k in clips[0]['draws'][0] if k != 't'} for v, clips in results.items()}
    summary = {v: {k: float(np.mean(values)) for k, values in metrics.items()} for v, metrics in clip_means.items()}
    paired = {v: {k: paired_stats(clip_means['v6_full'][k], clip_means[v][k]) for k in ('l_simple', 'loss')}
              for v in variants if v != 'v6_full'}
    gates = {name: value.item() for name, value in m6.named_parameters()
             if 'cross_limb_blocks' in name and value.numel() == 1}
    parameter_counts = {v: sum(p.numel() for p in m.parameters()) for v, m in [('v6', m6), ('v7', m7)]}
    parameter_counts['v6_cross_limb'] = sum(p.numel() for n, p in m6.named_parameters() if 'cross_limb_blocks' in n)
    report = dict(with_model=str(cli.with_model), without_model=str(cli.without_model),
                  weights=[weights6, weights7], device=str(device), precision='fp32, TF32 disabled', seed=20261018,
                  split=split, dataset_size=len(dataset), sampled_clips=count, timesteps=timesteps,
                  config_differences={k: [vars(a6).get(k, '<absent>'), vars(a7).get(k, '<absent>')]
                                      for k in sorted(vars(a6).keys() | vars(a7).keys()) if vars(a6).get(k) != vars(a7).get(k)},
                  hashes=hashes, parameter_counts=parameter_counts, gates=gates,
                  block_relative_change={k: float(np.mean(v)) for k, v in block_stats.items()},
                  summary=summary, paired_vs_v6_full=paired,
                  by_timestep={v: {str(t): {k: float(np.mean([clip['draws'][j][k] for clip in clips]))
                                           for k in clip_means[v]} for j, t in enumerate(timesteps)} for v, clips in results.items()},
                  samples=results,
                  limitations=['Denoising probe, not a full generative-quality evaluation.',
                               'Ablating a trained model changes its activation distribution.',
                               'Independent training runs can differ beyond the module causal effect.',
                               'Five fixed timesteps are a noise-level sweep, not uniform-t validation loss.'])
    (cli.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(summary=summary, paired=paired, gates=gates, parameter_counts=parameter_counts), indent=2), flush=True)
    print(f'SAVED {cli.output / "report.json"}', flush=True)


if __name__ == '__main__':
    main()
