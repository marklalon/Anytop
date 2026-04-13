from __future__ import annotations

import json
import math
import os
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from data_loaders.get_data import get_dataset_loader
from data_loaders.skeleton_metadata import SkeletonMetadata, load_skeleton_metadata
from data_loaders.truebones.offline_reference_dataset import load_cond_dict, resolve_dataset_root
from data_loaders.truebones.truebones_utils.motion_labels import load_motion_metadata
from eval.physics_features import PHYSICS_FEATURE_DIM, extract_physics_features
from utils.fixseed import fixseed


REQUIRED_COND_KEYS = {
    'tpos_first_frame',
    'joint_relations',
    'joints_graph_dist',
    'object_type',
    'parents',
    'offsets',
    'joints_names',
    'kinematic_chains',
    'mean',
    'std',
}

PHYSICS_GROUP_SLICES: dict[str, slice] = {
    'rigid': slice(0, 5),
    'dynamics': slice(5, 13),
    'contact': slice(13, 19),
    'symmetry': slice(19, 23),
    'global': slice(23, 30),
}

LOW_CONTACT_WARN_THRESHOLD = 0.01
CONTACT_SLIDE_MIN_ACTIVE_RATIO = 0.03

PHYSICS_FEATURE_NAMES = [
    'rigid_mean_bone_deviation',
    'rigid_max_bone_deviation',
    'rigid_bone_dev_q95',
    'rigid_bone_dev_q99',
    'rigid_bone_dev_gt_5pct_ratio',
    'dyn_mean_joint_speed',
    'dyn_std_joint_speed',
    'dyn_joint_speed_kurtosis',
    'dyn_max_joint_speed',
    'dyn_mean_jerk',
    'dyn_jerk_q95',
    'dyn_high_freq_ratio',
    'dyn_max_accel',
    'contact_mean_active_speed',
    'contact_active_speed_q95',
    'contact_autocorr_peak',
    'contact_on_ratio',
    'contact_landing_cluster',
    'contact_support_offset',
    'sym_corr_mean',
    'sym_amp_ratio_mean',
    'sym_phase_offset_mean',
    'sym_energy_ratio_mean',
    'global_pose_center_smoothness',
    'global_kinetic_energy_std',
    'global_log_bbox_volume_std',
    'global_bbox_height_std',
    'global_pose_spread_mean',
    'global_root_height_std',
    'global_root_rot6d_delta_mean',
]


@dataclass
class Finding:
    severity: str
    scope: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            'severity': self.severity,
            'scope': self.scope,
            'message': self.message,
        }


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Validate motion scorer inputs, metadata, and physics-feature behavior.')
    parser.add_argument('--dataset_dir', default='', type=str,
                        help='Dataset root. Empty uses the AnyTop default dataset path.')
    parser.add_argument('--split', default='train', choices=['train', 'val', 'test', 'all'], type=str,
                        help='Dataset split to sample through the same loader path used by training.')
    parser.add_argument('--objects_subset', default='all', type=str,
                        help='Object subset passed to the Truebones loader.')
    parser.add_argument('--action_tags', default='', type=str,
                        help='Optional action tag filter, same semantics as train_motion_scorer.py.')
    parser.add_argument('--num_frames', default=60, type=int,
                        help='Temporal window passed to the loader, matching scorer training by default.')
    parser.add_argument('--temporal_window', default=31, type=int,
                        help='Temporal attention window passed to the loader.')
    parser.add_argument('--num_samples', default=12, type=int,
                        help='Number of sampled training windows to inspect in detail.')
    parser.add_argument('--sample_pool_size', default=48, type=int,
                        help='How many candidate windows to let the dataset randomly draw before sampling batches.')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size used while collecting sampled windows.')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='DataLoader workers. 0 is the safest default for debugging.')
    parser.add_argument('--motion_cache_size', default=128, type=int,
                        help='Dataset motion cache size to reduce repeated disk reads during validation.')
    parser.add_argument('--seed', default=123, type=int,
                        help='Random seed for reproducible sampling.')
    parser.add_argument('--output_dir', default='', type=str,
                        help='Directory for the generated JSON and Markdown reports.')
    parser.add_argument('--strict', action='store_true',
                        help='Exit with code 1 if any error-level finding is produced.')
    return parser


def _add_finding(findings: list[Finding], severity: str, scope: str, message: str) -> None:
    findings.append(Finding(severity=severity, scope=scope, message=message))


def _safe_max_abs(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    return float(tensor.detach().abs().max().item())


def _tensor_is_finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def _np_all_finite(array: np.ndarray) -> bool:
    return bool(np.isfinite(array).all())


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _format_float(value: float, digits: int = 4) -> str:
    return f'{float(value):.{digits}f}'


def _escape_md(text: object) -> str:
    return str(text).replace('|', '\\|').replace('\n', ' ')


def _format_md_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    if not rows:
        return '_No rows._'
    header_line = '| ' + ' | '.join(_escape_md(header) for header in headers) + ' |'
    separator_line = '| ' + ' | '.join('---' for _ in headers) + ' |'
    body_lines = [
        '| ' + ' | '.join(_escape_md(cell) for cell in row) + ' |'
        for row in rows
    ]
    return '\n'.join([header_line, separator_line, *body_lines])


def _resolve_output_dir(args) -> Path:
    if args.output_dir:
        return Path(args.output_dir).resolve()
    return (Path(__file__).resolve().parents[1] / 'tmp' / 'motion_scorer_input_validation').resolve()


def _selected_object_types(cond_dict: Mapping[str, Mapping[str, object]], objects_subset: str) -> list[str]:
    if objects_subset == 'all':
        return sorted(cond_dict.keys())
    prefix = f'{objects_subset}_'
    exact = [object_type for object_type in cond_dict if object_type == objects_subset]
    prefix_matches = [object_type for object_type in cond_dict if object_type.startswith(prefix)]
    selected = sorted(set(exact + prefix_matches))
    if selected:
        return selected
    return sorted(cond_dict.keys())


def validate_cond_and_metadata(
    dataset_root: Path,
    cond_dict: Mapping[str, Mapping[str, object]],
    skeleton_lookup: Mapping[str, SkeletonMetadata],
    motion_metadata_lookup: Mapping[str, Mapping[str, object]],
    selected_objects: Sequence[str],
) -> tuple[list[Finding], list[dict[str, object]], dict[str, int]]:
    findings: list[Finding] = []
    object_rows: list[dict[str, object]] = []
    missing_motion_metadata_entries = 0

    for object_type in selected_objects:
        object_cond = cond_dict.get(object_type)
        if object_cond is None:
            _add_finding(findings, 'error', object_type, 'object type missing from cond.npy')
            continue

        missing_keys = sorted(REQUIRED_COND_KEYS - set(object_cond.keys()))
        if missing_keys:
            _add_finding(findings, 'error', object_type, f'missing cond keys: {missing_keys}')
            continue

        parents = np.asarray(object_cond['parents'])
        offsets = np.asarray(object_cond['offsets'])
        tpos_first_frame = np.asarray(object_cond['tpos_first_frame'])
        mean = np.asarray(object_cond['mean'])
        std = np.asarray(object_cond['std'])
        joint_relations = np.asarray(object_cond['joint_relations'])
        joints_graph_dist = np.asarray(object_cond['joints_graph_dist'])
        joints_names = list(object_cond['joints_names'])
        n_joints = int(len(parents))

        if n_joints <= 0:
            _add_finding(findings, 'error', object_type, 'parent array is empty')
            continue
        if offsets.shape != (n_joints, 3):
            _add_finding(findings, 'error', object_type, f'offsets shape mismatch: {offsets.shape}')
        if tpos_first_frame.shape != (n_joints, 13):
            _add_finding(findings, 'error', object_type, f'tpos_first_frame shape mismatch: {tpos_first_frame.shape}')
        if mean.shape != (n_joints, 13):
            _add_finding(findings, 'error', object_type, f'mean shape mismatch: {mean.shape}')
        if std.shape != (n_joints, 13):
            _add_finding(findings, 'error', object_type, f'std shape mismatch: {std.shape}')
        if joint_relations.shape != (n_joints, n_joints):
            _add_finding(findings, 'error', object_type, f'joint_relations shape mismatch: {joint_relations.shape}')
        if joints_graph_dist.shape != (n_joints, n_joints):
            _add_finding(findings, 'error', object_type, f'joints_graph_dist shape mismatch: {joints_graph_dist.shape}')
        if len(joints_names) != n_joints:
            _add_finding(findings, 'error', object_type, f'joints_names length mismatch: {len(joints_names)} vs {n_joints}')
        if not _np_all_finite(offsets):
            _add_finding(findings, 'error', object_type, 'offsets contain NaN or Inf')
        if not _np_all_finite(tpos_first_frame):
            _add_finding(findings, 'error', object_type, 'tpos_first_frame contains NaN or Inf')
        if not _np_all_finite(mean):
            _add_finding(findings, 'error', object_type, 'mean contains NaN or Inf')
        if not _np_all_finite(std):
            _add_finding(findings, 'error', object_type, 'std contains NaN or Inf')
        if not bool((std > 0).any()):
            _add_finding(findings, 'error', object_type, 'std is entirely non-positive')
        if not np.allclose(np.diag(joints_graph_dist), 0.0, atol=1e-6):
            _add_finding(findings, 'warn', object_type, 'joints_graph_dist diagonal is not zero')

        metadata = skeleton_lookup.get(object_type)
        if metadata is None:
            _add_finding(findings, 'error', object_type, 'failed to derive SkeletonMetadata for object type')
            continue
        if metadata.n_joints != n_joints:
            _add_finding(findings, 'error', object_type, f'SkeletonMetadata n_joints mismatch: {metadata.n_joints} vs {n_joints}')
        invalid_contacts = [index for index in metadata.contact_joints if index < 0 or index >= n_joints]
        if invalid_contacts:
            _add_finding(findings, 'error', object_type, f'contact_joints out of range: {invalid_contacts[:8]}')
        invalid_edges = [
            (child, parent)
            for child, parent in zip(metadata.edge_child_indices, metadata.edge_parent_indices)
            if child < 0 or child >= n_joints or parent < 0 or parent >= n_joints
        ]
        if invalid_edges:
            _add_finding(findings, 'error', object_type, f'edge pairs out of range: {invalid_edges[:8]}')
        if len(metadata.symmetry_left_indices) != len(metadata.symmetry_right_indices):
            _add_finding(findings, 'error', object_type, 'symmetry_left_indices and symmetry_right_indices length mismatch')
        if metadata.is_symmetric and not metadata.symmetric_joint_pairs:
            _add_finding(findings, 'warn', object_type, 'is_symmetric is true but symmetric_joint_pairs is empty')

        object_rows.append(
            {
                'object_type': object_type,
                'n_joints': n_joints,
                'contact_joints': len(metadata.contact_joints),
                'symmetry_pairs': len(metadata.symmetric_joint_pairs),
                'edge_count': len(metadata.edge_child_indices),
                'max_depth': int(metadata.max_joint_depth),
                'has_motion_metadata': int(any(meta.get('object_type') == object_type for meta in motion_metadata_lookup.values())),
            }
        )

    motions_dir = dataset_root / 'motions'
    motion_files = sorted(path.name for path in motions_dir.glob('*.npy'))
    if not motion_files:
        _add_finding(findings, 'error', 'dataset', f'no motion files found under {motions_dir}')
    else:
        for motion_name in motion_files[: min(len(motion_files), 64)]:
            metadata = motion_metadata_lookup.get(motion_name)
            if metadata is None:
                missing_motion_metadata_entries += 1
                continue
            if not metadata.get('object_type'):
                _add_finding(findings, 'warn', motion_name, 'motion_metadata.json entry is missing object_type')
            if not metadata.get('species_label'):
                _add_finding(findings, 'warn', motion_name, 'motion_metadata.json entry is missing species_label')
            if not metadata.get('action_label'):
                _add_finding(findings, 'warn', motion_name, 'motion_metadata.json entry is missing action_label')
            if not metadata.get('action_category'):
                _add_finding(findings, 'warn', motion_name, 'motion_metadata.json entry is missing action_category')

    overview = {
        'motion_files_total': len(motion_files),
        'motion_metadata_total': len(motion_metadata_lookup),
        'sampled_motion_metadata_missing_in_first_64_files': int(missing_motion_metadata_entries),
    }
    return findings, object_rows, overview


def collect_sample_records(args, skeleton_lookup: Mapping[str, SkeletonMetadata]) -> list[dict[str, object]]:
    loader = get_dataset_loader(
        batch_size=max(1, min(int(args.batch_size), int(args.num_samples))),
        num_frames=int(args.num_frames),
        split=args.split,
        temporal_window=int(args.temporal_window),
        t5_name='t5-base',
        balanced=False,
        objects_subset=args.objects_subset,
        num_workers=int(args.num_workers),
        prefetch_factor=2,
        sample_limit=max(int(args.sample_pool_size), int(args.num_samples)),
        shuffle=True,
        drop_last=False,
        use_reference_conditioning=False,
        action_tags=args.action_tags,
        motion_cache_size=int(args.motion_cache_size),
    )

    samples: list[dict[str, object]] = []
    for motion, cond in loader:
        batch_size = int(motion.shape[0])
        for batch_index in range(batch_size):
            if len(samples) >= int(args.num_samples):
                return samples
            object_type = str(cond['y']['object_type'][batch_index])
            metadata = cond['y'].get('motion_metadata', [None] * batch_size)[batch_index] or {}
            samples.append(
                {
                    'motion': motion[batch_index:batch_index + 1].detach().cpu(),
                    'n_joints': int(cond['y']['n_joints'][batch_index].item()),
                    'length': int(cond['y']['lengths'][batch_index].item()),
                    'object_type': object_type,
                    'motion_name': str(cond['y'].get('motion_name', [''])[batch_index]),
                    'species_label': str(cond['y'].get('species_label', [''])[batch_index]),
                    'action_label': str(cond['y'].get('action_label', [''])[batch_index]),
                    'action_category': str(cond['y'].get('action_category', [''])[batch_index]),
                    'action_tags': cond['y'].get('action_tags', [''])[batch_index],
                    'crop_start_ind': int(cond['y'].get('crop_start_ind', torch.zeros(batch_size))[batch_index].item()),
                    'motion_metadata': dict(metadata),
                    'skeleton_metadata': skeleton_lookup.get(object_type),
                    'mean': cond['y']['mean'][batch_index].detach().cpu(),
                    'std': cond['y']['std'][batch_index].detach().cpu(),
                }
            )
    return samples


def _make_contact_binary_metrics(contact_values: torch.Tensor) -> tuple[float, float]:
    if contact_values.numel() == 0:
        return 0.0, 0.0
    non_binary = ((contact_values - 0.0).abs() > 1e-5) & ((contact_values - 1.0).abs() > 1e-5)
    return float(non_binary.float().mean().item()), float((contact_values > 0.5).float().mean().item())


def _action_text(sample: Mapping[str, object]) -> str:
    parts: list[str] = [
        str(sample.get('action_label') or ''),
        str(sample.get('action_category') or ''),
    ]
    action_tags = sample.get('action_tags')
    if isinstance(action_tags, (list, tuple, set)):
        parts.extend(str(tag) for tag in action_tags)
    else:
        parts.append(str(action_tags or ''))
    return ' '.join(parts).lower()


def _expects_sparse_contact(sample: Mapping[str, object]) -> bool:
    action_text = _action_text(sample)
    return any(keyword in action_text for keyword in ('jump', 'land', 'landing', 'takeoff', 'airborne', 'fall', 'leap', 'hop'))


def _contact_review_values(
    valid_motion: torch.Tensor,
    sample_mean: torch.Tensor,
    sample_std: torch.Tensor,
    metadata: SkeletonMetadata,
    n_joints: int,
    length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    contact_indices = [index for index in metadata.contact_joints if index < n_joints]
    if not contact_indices:
        contact_indices = list(range(n_joints))
    contact_index_tensor = torch.as_tensor(contact_indices, dtype=torch.long)
    normalized_contact = valid_motion[contact_index_tensor, 12, :length]
    mean_contact = sample_mean[contact_index_tensor, 12].unsqueeze(-1)
    std_contact = sample_std[contact_index_tensor, 12].unsqueeze(-1)
    raw_contact = normalized_contact * std_contact + mean_contact
    return normalized_contact, raw_contact


def _feature_group_stats(features: torch.Tensor) -> dict[str, float]:
    stats: dict[str, float] = {}
    for group_name, feature_slice in PHYSICS_GROUP_SLICES.items():
        group = features[feature_slice]
        stats[f'{group_name}_mean'] = _round(group.mean().item())
        stats[f'{group_name}_abs_mean'] = _round(group.abs().mean().item())
        stats[f'{group_name}_max_abs'] = _round(group.abs().max().item())
    return stats


def _top_feature_values(features: torch.Tensor, limit: int = 5) -> list[dict[str, float | str]]:
    top_indices = torch.topk(features.abs(), k=min(limit, features.numel())).indices.tolist()
    return [
        {
            'name': PHYSICS_FEATURE_NAMES[index],
            'value': _round(features[index].item()),
            'abs_value': _round(abs(features[index].item())),
        }
        for index in top_indices
    ]


def analyze_sample_records(
    samples: Sequence[dict[str, object]],
    skeleton_lookup: Mapping[str, SkeletonMetadata],
    motion_metadata_lookup: Mapping[str, Mapping[str, object]],
) -> tuple[list[Finding], list[dict[str, object]]]:
    findings: list[Finding] = []
    sample_rows: list[dict[str, object]] = []

    for sample in samples:
        motion = sample['motion']
        n_joints = int(sample['n_joints'])
        length = int(sample['length'])
        object_type = str(sample['object_type'])
        motion_name = str(sample['motion_name'])
        metadata = skeleton_lookup.get(object_type)
        scope = motion_name or object_type
        sample_mean = sample['mean']
        sample_std = sample['std']

        if int(motion.shape[2]) != 13:
            _add_finding(findings, 'error', scope, f'feature dim is {motion.shape[2]}, expected 13')
            continue
        if length <= 0:
            _add_finding(findings, 'error', scope, f'invalid sampled length: {length}')
            continue
        if n_joints <= 0:
            _add_finding(findings, 'error', scope, f'invalid sampled n_joints: {n_joints}')
            continue
        if metadata is None:
            _add_finding(findings, 'error', scope, f'no SkeletonMetadata for object_type={object_type}')
            continue

        valid_motion = motion[0, :n_joints, :, :length]
        if not _tensor_is_finite(valid_motion):
            _add_finding(findings, 'error', scope, 'valid motion tensor contains NaN or Inf')

        joint_padding_max = _safe_max_abs(motion[0, n_joints:, :, :])
        if joint_padding_max > 1e-6:
            _add_finding(findings, 'warn', scope, f'joint padding is not zeroed, max_abs={joint_padding_max:.6f}')
        time_padding_max = _safe_max_abs(motion[0, :n_joints, :, length:])
        if time_padding_max > 1e-6:
            _add_finding(findings, 'warn', scope, f'time padding is not zeroed, max_abs={time_padding_max:.6f}')

        normalized_contact_values, raw_contact_values = _contact_review_values(
            valid_motion,
            sample_mean,
            sample_std,
            metadata,
            n_joints,
            length,
        )
        raw_contact_non_binary_ratio, raw_contact_on_ratio = _make_contact_binary_metrics(raw_contact_values)
        contact_mask_mismatch_ratio = float(
            ((normalized_contact_values > 0.5) != (raw_contact_values > 0.5)).float().mean().item()
        ) if raw_contact_values.numel() else 0.0
        if raw_contact_non_binary_ratio > 0.0:
            _add_finding(findings, 'warn', scope, f'raw contact values are not strictly binary, non_binary_ratio={raw_contact_non_binary_ratio:.4f}')

        metadata_entry = motion_metadata_lookup.get(motion_name, {})
        sample_metadata = dict(sample.get('motion_metadata') or {})
        expected_object_type = metadata_entry.get('object_type')
        if expected_object_type and expected_object_type != object_type:
            _add_finding(findings, 'warn', scope, f'motion_metadata object_type mismatch: {expected_object_type} vs sampled {object_type}')
        if sample_metadata.get('species_label') and str(sample_metadata.get('species_label')) != str(sample.get('species_label')):
            _add_finding(findings, 'warn', scope, 'species_label in cond does not match motion_metadata payload')
        if sample_metadata.get('action_label') and str(sample_metadata.get('action_label')) != str(sample.get('action_label')):
            _add_finding(findings, 'warn', scope, 'action_label in cond does not match motion_metadata payload')
        if not str(sample.get('species_label') or '').strip():
            _add_finding(findings, 'warn', scope, 'sampled cond is missing species_label')
        if not str(sample.get('action_label') or '').strip():
            _add_finding(findings, 'warn', scope, 'sampled cond is missing action_label')

        features = extract_physics_features(
            motion.float(),
            torch.as_tensor([n_joints], dtype=torch.long),
            torch.as_tensor([length], dtype=torch.long),
            [object_type],
            skeleton_lookup,
            feature_mean=sample_mean.unsqueeze(0).float(),
            feature_std=sample_std.unsqueeze(0).float(),
        )[0].detach().cpu().float()
        if int(features.shape[0]) != PHYSICS_FEATURE_DIM:
            _add_finding(findings, 'error', scope, f'physics feature dim is {features.shape[0]}, expected {PHYSICS_FEATURE_DIM}')
        if not _tensor_is_finite(features):
            _add_finding(findings, 'error', scope, 'physics features contain NaN or Inf')

        if metadata.symmetric_joint_pairs and float(features[PHYSICS_GROUP_SLICES['symmetry']].abs().max().item()) == 0.0:
            _add_finding(findings, 'warn', scope, 'symmetry feature group is exactly zero despite symmetric_joint_pairs being present')
        action_text = _action_text(sample)
        locomotion_like = 'locomotion' in action_text
        sparse_contact_expected = _expects_sparse_contact(sample)
        if locomotion_like and not sparse_contact_expected and len(metadata.contact_joints) > 0 and raw_contact_on_ratio < LOW_CONTACT_WARN_THRESHOLD:
            _add_finding(findings, 'warn', scope, f'raw contact_on_ratio is very low for a locomotion-like clip: {raw_contact_on_ratio:.4f}')

        sample_rows.append(
            {
                'motion_name': motion_name,
                'object_type': object_type,
                'species_label': str(sample.get('species_label') or ''),
                'action_label': str(sample.get('action_label') or ''),
                'action_category': str(sample.get('action_category') or ''),
                'action_tags': str(sample.get('action_tags') or ''),
                'length': length,
                'n_joints': n_joints,
                'crop_start_ind': int(sample.get('crop_start_ind') or 0),
                'raw_contact_non_binary_ratio': _round(raw_contact_non_binary_ratio),
                'raw_contact_on_ratio': _round(raw_contact_on_ratio),
                'contact_mask_mismatch_ratio': _round(contact_mask_mismatch_ratio),
                'low_contact_expected': bool(sparse_contact_expected),
                'joint_padding_max_abs': _round(joint_padding_max),
                'time_padding_max_abs': _round(time_padding_max),
                'physics_features': [_round(value) for value in features.tolist()],
                'physics_group_stats': _feature_group_stats(features),
                'top_abs_features': _top_feature_values(features),
                'motion_metadata_available': bool(motion_metadata_lookup.get(motion_name)),
                'manual_focus': [],
            }
        )

    return findings, sample_rows


def _recompute_velocity_channels(motion: torch.Tensor, n_joints: int, length: int) -> None:
    if length <= 1:
        return
    positions = motion[0, :n_joints, 0:3, :length]
    velocities = motion[0, :n_joints, 9:12, :length]
    velocities[:, :, :-1] = positions[:, :, 1:] - positions[:, :, :-1]
    velocities[:, :, -1] = velocities[:, :, -2]


def _apply_jitter(motion: torch.Tensor, n_joints: int, length: int, _metadata: SkeletonMetadata) -> torch.Tensor | None:
    if length <= 3:
        return None
    mutated = motion.clone()
    pattern = ((torch.arange(length, dtype=mutated.dtype) % 2) * 2.0 - 1.0).view(1, 1, length)
    mutated[0, :n_joints, 0:3, :length] += 0.15 * pattern
    mutated[0, :n_joints, 9:12, :length] += 0.25 * pattern
    mutated[0, 0, 3:9, :length] += 0.05 * pattern.repeat(1, 6, 1).squeeze(0)
    return mutated


def _apply_bone_length_drift(motion: torch.Tensor, n_joints: int, length: int, metadata: SkeletonMetadata) -> torch.Tensor | None:
    valid_edges = [
        (child, parent)
        for child, parent in zip(metadata.edge_child_indices, metadata.edge_parent_indices)
        if child < n_joints and parent < n_joints
    ]
    if not valid_edges or length <= 1:
        return None
    mutated = motion.clone()
    scale = (1.0 + torch.linspace(0.0, 0.35, length, dtype=mutated.dtype)).view(1, length)
    for child_index, parent_index in valid_edges:
        parent = mutated[0, parent_index, 0:3, :length]
        child = mutated[0, child_index, 0:3, :length]
        offset = child - parent
        mutated[0, child_index, 0:3, :length] = parent + offset * scale
    _recompute_velocity_channels(mutated, n_joints, length)
    return mutated


def _apply_contact_slide(motion: torch.Tensor, n_joints: int, length: int, metadata: SkeletonMetadata) -> torch.Tensor | None:
    valid_contacts = [index for index in metadata.contact_joints if index < n_joints]
    if not valid_contacts:
        return None
    mutated = motion.clone()
    contact_mask = mutated[0, valid_contacts, 12, :length] > 0.5
    if not bool(contact_mask.any().item()):
        return None
    drift = torch.linspace(0.0, 0.6, length, dtype=mutated.dtype)
    for offset, joint_index in enumerate(valid_contacts):
        active = contact_mask[offset]
        if not bool(active.any().item()):
            continue
        mutated[0, joint_index, 0, :length][active] += drift[active]
        mutated[0, joint_index, 9, :length][active] += 0.45
    return mutated


def _apply_contact_slide_with_raw_mask(
    motion: torch.Tensor,
    n_joints: int,
    length: int,
    metadata: SkeletonMetadata,
    sample_mean: torch.Tensor,
    sample_std: torch.Tensor,
) -> torch.Tensor | None:
    valid_contacts = [index for index in metadata.contact_joints if index < n_joints]
    if not valid_contacts:
        return None
    mutated = motion.clone()
    _, raw_contact_values = _contact_review_values(
        mutated[0, :n_joints, :, :length],
        sample_mean,
        sample_std,
        metadata,
        n_joints,
        length,
    )
    contact_mask = raw_contact_values > 0.5
    if not bool(contact_mask.any().item()):
        return None
    drift = torch.linspace(0.0, 0.6, length, dtype=mutated.dtype)
    for offset, joint_index in enumerate(valid_contacts):
        active = contact_mask[offset]
        if not bool(active.any().item()):
            continue
        mutated[0, joint_index, 0, :length][active] += drift[active]
        mutated[0, joint_index, 9, :length][active] += 0.45
    return mutated


def _apply_symmetry_break(motion: torch.Tensor, n_joints: int, length: int, metadata: SkeletonMetadata) -> torch.Tensor | None:
    left_indices = [index for index in metadata.symmetry_left_indices if index < n_joints]
    right_indices = [index for index in metadata.symmetry_right_indices if index < n_joints]
    if not left_indices or len(left_indices) != len(right_indices):
        return None
    mutated = motion.clone()
    time_wave = torch.sin(torch.linspace(0.0, math.pi * 2.0, length, dtype=mutated.dtype)).view(1, 1, length)
    mutated[0, left_indices, 0:3, :length] += 0.18 * time_wave
    mutated[0, left_indices, 9:12, :length] += 0.22 * time_wave
    return mutated


PERTURBATIONS: tuple[tuple[str, Any, tuple[str, ...], float], ...] = (
    ('bone_length_drift', _apply_bone_length_drift, ('rigid',), 0.01),
    ('jitter', _apply_jitter, ('dynamics', 'global'), 0.01),
    ('contact_slide', _apply_contact_slide, ('contact',), 0.001),
    ('symmetry_break', _apply_symmetry_break, ('symmetry',), 0.005),
)


def run_perturbation_checks(
    samples: Sequence[dict[str, object]],
    skeleton_lookup: Mapping[str, SkeletonMetadata],
) -> tuple[list[Finding], list[dict[str, object]]]:
    findings: list[Finding] = []
    rows: list[dict[str, object]] = []

    for sample in samples:
        motion = sample['motion'].float()
        n_joints = int(sample['n_joints'])
        length = int(sample['length'])
        object_type = str(sample['object_type'])
        motion_name = str(sample['motion_name'])
        metadata = skeleton_lookup.get(object_type)
        sample_mean = sample['mean'].float()
        sample_std = sample['std'].float()
        if metadata is None:
            continue

        base_features = extract_physics_features(
            motion,
            torch.as_tensor([n_joints], dtype=torch.long),
            torch.as_tensor([length], dtype=torch.long),
            [object_type],
            skeleton_lookup,
            feature_mean=sample_mean.unsqueeze(0),
            feature_std=sample_std.unsqueeze(0),
        )[0].detach().cpu().float()

        for perturbation_name, perturb_fn, target_groups, threshold in PERTURBATIONS:
            if perturbation_name == 'contact_slide':
                _, raw_contact_values = _contact_review_values(
                    motion[0, :n_joints, :, :length],
                    sample_mean,
                    sample_std,
                    metadata,
                    n_joints,
                    length,
                )
                _, raw_contact_on_ratio = _make_contact_binary_metrics(raw_contact_values)
                if raw_contact_on_ratio < CONTACT_SLIDE_MIN_ACTIVE_RATIO:
                    rows.append(
                        {
                            'motion_name': motion_name,
                            'object_type': object_type,
                            'perturbation': perturbation_name,
                            'status': 'skipped',
                            'reason': f'insufficient active raw contact in sampled window (raw_contact_on_ratio={raw_contact_on_ratio:.4f})',
                            'target_groups': list(target_groups),
                            'group_abs_delta': {},
                        }
                    )
                    continue
                mutated_motion = _apply_contact_slide_with_raw_mask(motion, n_joints, length, metadata, sample_mean, sample_std)
            else:
                mutated_motion = perturb_fn(motion, n_joints, length, metadata)
            scope = f'{motion_name}:{perturbation_name}'
            if mutated_motion is None:
                rows.append(
                    {
                        'motion_name': motion_name,
                        'object_type': object_type,
                        'perturbation': perturbation_name,
                        'status': 'skipped',
                        'reason': 'preconditions not met for this sample',
                        'target_groups': list(target_groups),
                        'group_abs_delta': {},
                    }
                )
                continue

            mutated_features = extract_physics_features(
                mutated_motion,
                torch.as_tensor([n_joints], dtype=torch.long),
                torch.as_tensor([length], dtype=torch.long),
                [object_type],
                skeleton_lookup,
                feature_mean=sample_mean.unsqueeze(0),
                feature_std=sample_std.unsqueeze(0),
            )[0].detach().cpu().float()
            delta = mutated_features - base_features
            group_abs_delta = {
                group_name: _round(delta[group_slice].abs().mean().item())
                for group_name, group_slice in PHYSICS_GROUP_SLICES.items()
            }
            target_effect = max(group_abs_delta[group_name] for group_name in target_groups)
            status = 'pass' if target_effect > threshold else 'warn'
            if status == 'warn':
                _add_finding(
                    findings,
                    'warn',
                    scope,
                    f'perturbation changed target groups too weakly: target_effect={target_effect:.4f}, threshold={threshold:.4f}',
                )
            rows.append(
                {
                    'motion_name': motion_name,
                    'object_type': object_type,
                    'perturbation': perturbation_name,
                    'status': status,
                    'reason': '',
                    'target_groups': list(target_groups),
                    'group_abs_delta': group_abs_delta,
                }
            )

    return findings, rows


def attach_manual_focus(sample_rows: list[dict[str, object]], perturbation_rows: Sequence[dict[str, object]]) -> None:
    perturbation_by_motion: dict[str, list[dict[str, object]]] = {}
    for row in perturbation_rows:
        perturbation_by_motion.setdefault(str(row['motion_name']), []).append(row)

    for sample in sample_rows:
        focus: list[str] = []
        if float(sample['raw_contact_non_binary_ratio']) > 0.0:
            focus.append('Check whether the raw contact channel should really be binary for this clip.')
        if float(sample['raw_contact_on_ratio']) < LOW_CONTACT_WARN_THRESHOLD:
            if bool(sample.get('low_contact_expected')):
                focus.append('Low contact looks consistent with a jump or landing transition in this sampled window.')
            elif 'locomotion' in _action_text(sample):
                focus.append('This locomotion-like clip has almost no active contact. Verify contact labels and foot joints.')
        if not sample['motion_metadata_available']:
            focus.append('motion_metadata.json has no entry for this motion. Verify labels manually from the filename and animation.')
        for perturbation in perturbation_by_motion.get(str(sample['motion_name']), []):
            if perturbation['status'] == 'warn':
                focus.append(
                    f"{perturbation['perturbation']} barely moved its target physics group. Inspect the related metadata and channels."
                )
        if not focus:
            focus.append('Spot-check labels, contact timing, and whether the motion visually matches the object_type and action_label.')
        sample['manual_focus'] = focus


def build_markdown_report(
    args,
    dataset_root: Path,
    overview: Mapping[str, int],
    object_rows: Sequence[Mapping[str, object]],
    sample_rows: Sequence[Mapping[str, object]],
    perturbation_rows: Sequence[Mapping[str, object]],
    findings: Sequence[Finding],
) -> str:
    error_count = sum(1 for finding in findings if finding.severity == 'error')
    warn_count = sum(1 for finding in findings if finding.severity == 'warn')

    object_table_rows = [
        [
            row['object_type'],
            row['n_joints'],
            row['contact_joints'],
            row['symmetry_pairs'],
            row['edge_count'],
            row['max_depth'],
            row['has_motion_metadata'],
        ]
        for row in object_rows
    ]

    sample_table_rows = [
        [
            row['motion_name'],
            row['object_type'],
            row['species_label'],
            row['action_label'],
            row['length'],
            row['n_joints'],
            _format_float(row['raw_contact_on_ratio']),
            _format_float(row['raw_contact_non_binary_ratio']),
            _format_float(row['contact_mask_mismatch_ratio']),
            _format_float(row['joint_padding_max_abs']),
            _format_float(row['time_padding_max_abs']),
        ]
        for row in sample_rows
    ]

    perturbation_table_rows = [
        [
            row['motion_name'],
            row['perturbation'],
            row['status'],
            ', '.join(row['target_groups']),
            _format_float(max(row['group_abs_delta'].values()) if row['group_abs_delta'] else 0.0),
            row['reason'],
        ]
        for row in perturbation_rows
    ]

    high_priority_findings = [finding for finding in findings if finding.severity in {'error', 'warn'}][:20]
    finding_lines = '\n'.join(
        f'- [{finding.severity.upper()}] {finding.scope}: {finding.message}'
        for finding in high_priority_findings
    ) or '- None.'

    sample_sections: list[str] = []
    for sample in sample_rows:
        group_stats = sample['physics_group_stats']
        group_rows = [
            [group_name, _format_float(group_stats[f'{group_name}_mean']), _format_float(group_stats[f'{group_name}_abs_mean']), _format_float(group_stats[f'{group_name}_max_abs'])]
            for group_name in PHYSICS_GROUP_SLICES
        ]
        top_feature_rows = [
            [feature['name'], _format_float(feature['value']), _format_float(feature['abs_value'])]
            for feature in sample['top_abs_features']
        ]
        manual_focus_lines = '\n'.join(f'- {item}' for item in sample['manual_focus'])
        sample_sections.append(
            '\n'.join(
                [
                    f"### {sample['motion_name']}",
                    '',
                    f"- object_type: {sample['object_type']}",
                    f"- labels: species={sample['species_label']} | action={sample['action_label']} | category={sample['action_category']}",
                    f"- action_tags: {sample['action_tags']}",
                    f"- length={sample['length']} | n_joints={sample['n_joints']} | crop_start_ind={sample['crop_start_ind']}",
                    f"- raw_contact_on_ratio={_format_float(sample['raw_contact_on_ratio'])} | raw_contact_non_binary_ratio={_format_float(sample['raw_contact_non_binary_ratio'])}",
                    f"- contact_mask_mismatch_ratio={_format_float(sample['contact_mask_mismatch_ratio'])}",
                    '',
                    _format_md_table(
                        ['Group', 'Mean', 'Abs Mean', 'Max Abs'],
                        group_rows,
                    ),
                    '',
                    _format_md_table(
                        ['Top Feature', 'Value', 'Abs Value'],
                        top_feature_rows,
                    ),
                    '',
                    'Manual checks:',
                    manual_focus_lines,
                ]
            )
        )

    return '\n'.join(
        [
            '# Motion Scorer Input Validation Report',
            '',
            f'- Generated: {datetime.now().isoformat(timespec="seconds")}',
            f'- Dataset root: {dataset_root}',
            f'- Split: {args.split}',
            f'- Objects subset: {args.objects_subset}',
            f'- Action tags: {args.action_tags or "<all>"}',
            f'- num_frames: {args.num_frames}',
            f'- Sampled windows: {len(sample_rows)}',
            '',
            '## Summary',
            '',
            f'- Errors: {error_count}',
            f'- Warnings: {warn_count}',
            f"- Dataset overview: motion_files_total={overview.get('motion_files_total', 0)}, motion_metadata_total={overview.get('motion_metadata_total', 0)}, sampled_motion_metadata_missing_in_first_64_files={overview.get('sampled_motion_metadata_missing_in_first_64_files', 0)}",
            '',
            '## What To Look At First',
            '',
            '- If an error mentions cond shapes or out-of-range indices, fix metadata before trusting any scorer training.',
            '- If raw_contact_on_ratio is near 0 for a steady locomotion clip, inspect contact joints and contact labels first; jump or landing windows can legitimately stay near zero.',
            '- If contact_mask_mismatch_ratio is non-zero, thresholding the normalized contact channel directly would disagree with raw contact semantics. Physics extraction now compensates for this by denormalizing first.',
            '- contact_slide is skipped when the sampled window has too little active raw contact to produce a meaningful perturbation response.',
            '- If symmetry_break barely changes the symmetry feature group, inspect symmetric_joint_pairs and left/right joint mapping.',
            '- If bone_length_drift barely changes the rigid group, inspect parents, offsets, and edge pairs.',
            '',
            '## Automatic Findings',
            '',
            finding_lines,
            '',
            '## Object Metadata Overview',
            '',
            _format_md_table(
                ['Object', 'n_joints', 'contact_joints', 'sym_pairs', 'edges', 'max_depth', 'has_motion_metadata'],
                object_table_rows,
            ),
            '',
            '## Sample Overview',
            '',
            _format_md_table(
                ['Motion', 'Object', 'Species', 'Action', 'Length', 'Joints', 'raw_contact_on', 'raw_contact_non_binary', 'contact_mask_mismatch', 'joint_pad_max', 'time_pad_max'],
                sample_table_rows,
            ),
            '',
            '## Perturbation Response',
            '',
            _format_md_table(
                ['Motion', 'Perturbation', 'Status', 'Target Groups', 'Max Group Delta', 'Reason'],
                perturbation_table_rows,
            ),
            '',
            '## Per-Sample Review',
            '',
            '\n\n'.join(sample_sections) if sample_sections else '_No sampled windows._',
            '',
        ]
    )


def main() -> int:
    args = build_parser().parse_args()
    fixseed(int(args.seed))

    dataset_root = resolve_dataset_root(args.dataset_dir or None)
    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    cond_dict = load_cond_dict(dataset_root)
    skeleton_lookup = load_skeleton_metadata(dataset_root, cond_dict=cond_dict)
    motion_metadata_lookup = load_motion_metadata(dataset_root)
    selected_objects = _selected_object_types(cond_dict, args.objects_subset)

    cond_findings, object_rows, overview = validate_cond_and_metadata(
        dataset_root,
        cond_dict,
        skeleton_lookup,
        motion_metadata_lookup,
        selected_objects,
    )
    sample_records = collect_sample_records(args, skeleton_lookup)
    sample_findings, sample_rows = analyze_sample_records(sample_records, skeleton_lookup, motion_metadata_lookup)
    perturbation_findings, perturbation_rows = run_perturbation_checks(sample_records, skeleton_lookup)
    attach_manual_focus(sample_rows, perturbation_rows)

    findings = [*cond_findings, *sample_findings, *perturbation_findings]
    report = {
        'settings': {
            'dataset_root': str(dataset_root),
            'split': args.split,
            'objects_subset': args.objects_subset,
            'action_tags': args.action_tags,
            'num_frames': int(args.num_frames),
            'num_samples': int(args.num_samples),
            'sample_pool_size': int(args.sample_pool_size),
            'seed': int(args.seed),
        },
        'overview': dict(overview),
        'findings': [finding.to_dict() for finding in findings],
        'object_rows': list(object_rows),
        'sample_rows': list(sample_rows),
        'perturbation_rows': list(perturbation_rows),
    }

    report_json_path = output_dir / 'motion_scorer_input_validation.json'
    report_md_path = output_dir / 'motion_scorer_input_validation.md'
    report_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding='utf-8')
    report_md_path.write_text(
        build_markdown_report(args, dataset_root, overview, object_rows, sample_rows, perturbation_rows, findings),
        encoding='utf-8',
    )

    error_count = sum(1 for finding in findings if finding.severity == 'error')
    warn_count = sum(1 for finding in findings if finding.severity == 'warn')
    print(f'dataset_root: {dataset_root}')
    print(f'sampled_windows: {len(sample_rows)}')
    print(f'findings: errors={error_count} warnings={warn_count}')
    print(f'json_report: {report_json_path}')
    print(f'markdown_report: {report_md_path}')

    if args.strict and error_count > 0:
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())