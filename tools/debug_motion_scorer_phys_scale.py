from __future__ import annotations

import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from data_loaders.get_data import get_dataset_loader
from data_loaders.skeleton_metadata import load_skeleton_metadata
from data_loaders.truebones.offline_reference_dataset import load_cond_dict, resolve_dataset_root
from eval.physics_features import PHYSICS_FEATURE_DIM, extract_physics_features
from tools.validate_motion_scorer_inputs import PHYSICS_FEATURE_NAMES
from utils.fixseed import fixseed


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Inspect motion scorer physics-target scale on sampled training windows.')
    parser.add_argument('--dataset_dir', default='', type=str)
    parser.add_argument('--split', default='train', type=str)
    parser.add_argument('--objects_subset', default='all', type=str)
    parser.add_argument('--action_tags', default='', type=str)
    parser.add_argument('--num_frames', default=60, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_batches', default=8, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--motion_cache_size', default=256, type=int)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--inspect_motion_name', default='', type=str)
    return parser


def _inspect_motion(
    motion: torch.Tensor,
    cond: dict,
    batch_index: int,
    cond_dict: dict,
) -> None:
    object_type = str(cond['y']['object_type'][batch_index])
    motion_name = str(cond['y'].get('motion_name', [''])[batch_index])
    n_joints = int(cond['y']['n_joints'][batch_index].item())
    length = int(cond['y']['lengths'][batch_index].item())
    mean = cond['y']['mean'][batch_index, :n_joints].float()
    std = cond['y']['std'][batch_index, :n_joints].float().clamp_min(1e-6)
    denorm_motion = motion[batch_index, :n_joints, :, :length].float() * std.unsqueeze(-1) + mean.unsqueeze(-1)
    positions = denorm_motion[:, :3, :].permute(2, 0, 1).contiguous()

    parents = [int(value) for value in cond_dict[object_type]['parents'][:n_joints]]
    joint_names = [str(value) for value in cond_dict[object_type]['joints_names'][:n_joints]]
    edge_rows: list[tuple[float, float, float, int, int]] = []
    for child_index, parent_index in enumerate(parents):
        if parent_index < 0:
            continue
        lengths_series = torch.linalg.norm(positions[:, child_index] - positions[:, parent_index], dim=-1)
        baseline = float(lengths_series.median().item())
        baseline_clamped = max(baseline, 1e-6)
        deviation = (lengths_series - baseline).abs() / baseline_clamped
        edge_rows.append(
            (
                float(deviation.max().item()),
                float(deviation.mean().item()),
                baseline,
                child_index,
                parent_index,
            )
        )

    edge_rows.sort(reverse=True)
    print(f'inspect_motion: motion={motion_name} object={object_type} n_joints={n_joints} length={length}')
    for max_dev, mean_dev, baseline, child_index, parent_index in edge_rows[:10]:
        print(
            f'  child={child_index}:{joint_names[child_index]} parent={parent_index}:{joint_names[parent_index]} '
            f'baseline={baseline:.9f} mean_dev={mean_dev:.6f} max_dev={max_dev:.6f}'
        )


def main() -> int:
    args = build_parser().parse_args()
    fixseed(int(args.seed))

    dataset_root = resolve_dataset_root(args.dataset_dir or None)
    cond_dict = load_cond_dict(dataset_root)
    skeleton_lookup = load_skeleton_metadata(dataset_root, cond_dict=cond_dict)
    loader = get_dataset_loader(
        batch_size=int(args.batch_size),
        num_frames=int(args.num_frames),
        split=args.split,
        temporal_window=31,
        t5_name='t5-base',
        balanced=False,
        objects_subset=args.objects_subset,
        num_workers=int(args.num_workers),
        prefetch_factor=2,
        sample_limit=0,
        shuffle=True,
        drop_last=False,
        use_reference_conditioning=False,
        action_tags=args.action_tags,
        motion_cache_size=int(args.motion_cache_size),
    )

    feature_batches: list[torch.Tensor] = []
    sample_descriptors: list[tuple[str, str]] = []
    sampled = 0
    for batch_index, (motion, cond) in enumerate(loader):
        object_types = [str(value) for value in cond['y']['object_type']]
        motion_names = [str(value) for value in cond['y'].get('motion_name', [''] * len(object_types))]
        if args.inspect_motion_name:
            for sample_index, motion_name in enumerate(motion_names):
                if motion_name == args.inspect_motion_name:
                    _inspect_motion(motion, cond, sample_index, cond_dict)
        features = extract_physics_features(
            motion.float(),
            cond['y']['n_joints'],
            cond['y']['lengths'],
            object_types,
            skeleton_lookup,
            feature_mean=cond['y']['mean'].float(),
            feature_std=cond['y']['std'].float(),
        ).detach().cpu().float()
        if int(features.shape[-1]) != PHYSICS_FEATURE_DIM:
            raise ValueError(f'Expected phys dim {PHYSICS_FEATURE_DIM}, got {tuple(features.shape)}')
        feature_batches.append(features)
        sample_descriptors.extend(zip(motion_names, object_types))
        sampled += int(features.shape[0])
        if batch_index + 1 >= int(args.num_batches):
            break

    all_features = torch.cat(feature_batches, dim=0)
    abs_features = all_features.abs()
    mse_from_zero = all_features.pow(2).mean(dim=0)
    quantiles = torch.quantile(all_features, torch.tensor([0.5, 0.9, 0.99], dtype=torch.float32), dim=0)

    print(f'dataset_root: {dataset_root}')
    print(f'sampled_windows: {sampled}')
    print(f'overall_mean_sq: {all_features.pow(2).mean().item():.6f}')
    print(f'overall_max_abs: {abs_features.max().item():.6f}')
    top_dims = torch.topk(mse_from_zero, k=min(10, mse_from_zero.numel())).indices.tolist()
    print('top_dims_by_mean_sq:')
    for index in top_dims:
        print(
            f'  {index:02d} {PHYSICS_FEATURE_NAMES[index]} '
            f'mean={all_features[:, index].mean().item():.6f} '
            f'std={all_features[:, index].std(unbiased=False).item():.6f} '
            f'mean_sq={mse_from_zero[index].item():.6f} '
            f'max_abs={abs_features[:, index].max().item():.6f} '
            f'q50={quantiles[0, index].item():.6f} '
            f'q90={quantiles[1, index].item():.6f} '
            f'q99={quantiles[2, index].item():.6f}'
        )

    per_sample_mean_sq = all_features.pow(2).mean(dim=1).numpy()
    print(
        'per_sample_mean_sq: '
        f'mean={float(np.mean(per_sample_mean_sq)):.6f} '
        f'p90={float(np.quantile(per_sample_mean_sq, 0.9)):.6f} '
        f'p99={float(np.quantile(per_sample_mean_sq, 0.99)):.6f} '
        f'max={float(np.max(per_sample_mean_sq)):.6f}'
    )
    top_sample_indices = np.argsort(per_sample_mean_sq)[-10:][::-1].tolist()
    print('top_samples_by_mean_sq:')
    for index in top_sample_indices:
        motion_name, object_type = sample_descriptors[index]
        sample_features = all_features[index]
        dominant_feature_index = int(torch.argmax(sample_features.abs()).item())
        print(
            f'  sample={index:03d} motion={motion_name} object={object_type} '
            f'mean_sq={per_sample_mean_sq[index]:.6f} '
            f'max_abs={sample_features.abs().max().item():.6f} '
            f'dominant={PHYSICS_FEATURE_NAMES[dominant_feature_index]} '
            f'dominant_value={sample_features[dominant_feature_index].item():.6f}'
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())