from __future__ import annotations

from typing import Mapping, Sequence

import torch

from data_loaders.skeleton_metadata import SkeletonMetadata


PHYSICS_FEATURE_DIM = 30
_INDEX_TENSOR_CACHE: dict[tuple[tuple[int, ...], str], torch.Tensor] = {}


def _cached_index_tensor(indices: Sequence[int], device: torch.device) -> torch.Tensor:
    key = (tuple(int(index) for index in indices), str(device))
    tensor = _INDEX_TENSOR_CACHE.get(key)
    if tensor is None:
        tensor = torch.as_tensor(key[0], dtype=torch.long, device=device)
        _INDEX_TENSOR_CACHE[key] = tensor
    return tensor


def _safe_quantile(values: torch.Tensor, q: float) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros(())
    return torch.quantile(values.float(), q)


def _safe_mean(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros(())
    return values.float().mean()


def _safe_std(values: torch.Tensor) -> torch.Tensor:
    if values.numel() <= 1:
        return values.new_zeros(())
    return values.float().std(unbiased=False)


def _safe_max(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros(())
    return values.float().max()


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return numerator / denominator.clamp_min(1e-6)


def _kurtosis(values: torch.Tensor) -> torch.Tensor:
    if values.numel() <= 3:
        return values.new_zeros(())
    centered = values.float() - values.float().mean()
    variance = centered.pow(2).mean().clamp_min(1e-6)
    return centered.pow(4).mean() / variance.pow(2)


def _autocorr_peak(signal: torch.Tensor, max_lag: int = 12) -> torch.Tensor:
    if signal.numel() <= 2:
        return signal.new_zeros(())
    centered = signal.float() - signal.float().mean()
    denom = centered.pow(2).sum().clamp_min(1e-6)
    peak = centered.new_zeros(())
    for lag in range(1, min(max_lag, int(signal.numel()) - 1) + 1):
        corr = (centered[:-lag] * centered[lag:]).sum() / denom
        peak = torch.maximum(peak, corr)
    return peak


def _phase_offset(left: torch.Tensor, right: torch.Tensor, max_lag: int = 8) -> torch.Tensor:
    if left.numel() <= 2 or right.numel() <= 2:
        return left.new_zeros(())
    best_lag = 0
    best_score = None
    for lag in range(-min(max_lag, int(left.numel()) - 1), min(max_lag, int(left.numel()) - 1) + 1):
        if lag < 0:
            lhs = left[-lag:]
            rhs = right[: lag + right.shape[0]]
        elif lag > 0:
            lhs = left[: left.shape[0] - lag]
            rhs = right[lag:]
        else:
            lhs = left
            rhs = right
        if lhs.numel() <= 1 or rhs.numel() <= 1:
            continue
        score = (lhs.float() * rhs.float()).mean()
        if best_score is None or score > best_score:
            best_score = score
            best_lag = lag
    return left.new_tensor(float(best_lag) / max(float(left.numel()), 1.0))


def _high_frequency_ratio(signal: torch.Tensor) -> torch.Tensor:
    if signal.numel() <= 4:
        return signal.new_zeros(())
    spectrum = torch.fft.rfft(signal.float(), dim=0)
    power = spectrum.abs().pow(2)
    if power.shape[0] <= 2:
        return signal.new_zeros(())
    split_index = max(1, int(power.shape[0] * 0.6))
    high = power[split_index:].sum()
    total = power[1:].sum().clamp_min(1e-6)
    return high / total


def _extract_single_sample_features(
    motion: torch.Tensor,
    length: int,
    metadata: SkeletonMetadata,
) -> torch.Tensor:
    joint_count = min(int(metadata.n_joints), int(motion.shape[0]))
    motion = motion[:joint_count, :, :length]
    positions = motion[:, :3, :].permute(2, 0, 1).contiguous()
    velocities = motion[:, 9:12, :].permute(2, 0, 1).contiguous()
    contact_channel = motion[:, 12, :].permute(1, 0).contiguous()
    root_rot6d = motion[0, 3:9, :].permute(1, 0).contiguous()
    if metadata.edge_child_indices:
        edge_child_indices = [index for index in metadata.edge_child_indices if index < joint_count]
        edge_parent_indices = metadata.edge_parent_indices[: len(edge_child_indices)]
        child_index_tensor = _cached_index_tensor(edge_child_indices, positions.device)
        parent_index_tensor = _cached_index_tensor(edge_parent_indices, positions.device)
        bone_lengths = torch.linalg.norm(positions[:, child_index_tensor] - positions[:, parent_index_tensor], dim=-1)
        baseline = bone_lengths.median(dim=0).values.clamp_min(1e-6)
        bone_deviation = (bone_lengths - baseline.unsqueeze(0)).abs() / baseline.unsqueeze(0)
    else:
        bone_deviation = positions.new_zeros((max(length, 1), 1))

    joint_speed = torch.linalg.norm(velocities, dim=-1)
    accel = velocities[1:] - velocities[:-1] if length > 1 else velocities.new_zeros((0, joint_count, 3))
    accel_norm = torch.linalg.norm(accel, dim=-1) if accel.numel() else velocities.new_zeros((0, joint_count))
    jerk = accel[1:] - accel[:-1] if accel.shape[0] > 1 else velocities.new_zeros((0, joint_count, 3))
    jerk_norm = torch.linalg.norm(jerk, dim=-1) if jerk.numel() else velocities.new_zeros((0, joint_count))

    contact_indices = metadata.contact_joints[:]
    if contact_indices:
        valid_contact_indices = [index for index in contact_indices if index < joint_count]
    else:
        valid_contact_indices = []
    if valid_contact_indices:
        contact_index_tensor = _cached_index_tensor(valid_contact_indices, positions.device)
        contact_mask = contact_channel[:, contact_index_tensor] > 0.5
        contact_speed = torch.linalg.norm(velocities[:, contact_index_tensor][:, :, [0, 2]], dim=-1)
        active_contact_speed = contact_speed[contact_mask]
        contact_binary = contact_mask.any(dim=1).float()
        contact_positions = positions[:, contact_index_tensor]
        active_positions = contact_positions[contact_mask]
        if active_positions.numel():
            landing_cluster = torch.linalg.norm(active_positions[:, [0, 2]] - active_positions[:, [0, 2]].mean(dim=0), dim=-1)
            support_centroid = contact_positions[:, :, [0, 2]].mean(dim=1)
            pose_center = positions.mean(dim=1)[:, [0, 2]]
            support_offset = torch.linalg.norm(pose_center - support_centroid, dim=-1)
        else:
            landing_cluster = positions.new_zeros((0,))
            support_offset = positions.new_zeros((0,))
    else:
        active_contact_speed = positions.new_zeros((0,))
        contact_binary = positions.new_zeros((length,))
        landing_cluster = positions.new_zeros((0,))
        support_offset = positions.new_zeros((0,))

    symmetry_left_indices = [index for index in metadata.symmetry_left_indices if index < joint_count]
    symmetry_right_indices = [index for index in metadata.symmetry_right_indices if index < joint_count]
    if symmetry_left_indices and len(symmetry_left_indices) == len(symmetry_right_indices):
        left_index_tensor = _cached_index_tensor(symmetry_left_indices, positions.device)
        right_index_tensor = _cached_index_tensor(symmetry_right_indices, positions.device)
        left_speeds = joint_speed[:, left_index_tensor]
        right_speeds = joint_speed[:, right_index_tensor]
        left_centered = left_speeds - left_speeds.mean(dim=0, keepdim=True)
        right_centered = right_speeds - right_speeds.mean(dim=0, keepdim=True)
        denom = (left_centered.pow(2).sum(dim=0) * right_centered.pow(2).sum(dim=0)).sqrt().clamp_min(1e-6)
        corr = (left_centered * right_centered).sum(dim=0) / denom
        amp_ratio = torch.minimum(left_speeds.mean(dim=0), right_speeds.mean(dim=0)) / torch.maximum(left_speeds.mean(dim=0), right_speeds.mean(dim=0)).clamp_min(1e-6)
        energy_ratio = torch.minimum(left_speeds.pow(2).mean(dim=0), right_speeds.pow(2).mean(dim=0)) / torch.maximum(left_speeds.pow(2).mean(dim=0), right_speeds.pow(2).mean(dim=0)).clamp_min(1e-6)
        phase_values = torch.stack(
            [_phase_offset(left_speeds[:, index], right_speeds[:, index]) for index in range(left_speeds.shape[1])],
            dim=0,
        )
        symmetry_metrics_tensor = torch.stack(
            [corr.mean(), amp_ratio.mean(), phase_values.mean(), energy_ratio.mean()],
            dim=0,
        )
    else:
        symmetry_metrics_tensor = positions.new_zeros((4,))

    pose_center = positions.mean(dim=1)
    pose_center_smoothness = _safe_mean(torch.linalg.norm(pose_center[2:] - 2.0 * pose_center[1:-1] + pose_center[:-2], dim=-1))
    kinetic_energy_series = joint_speed.pow(2).mean(dim=1)
    bbox_min = positions.min(dim=1).values
    bbox_max = positions.max(dim=1).values
    bbox_extent = (bbox_max - bbox_min).clamp_min(1e-6)
    bbox_volume = bbox_extent.prod(dim=-1)
    bbox_height = bbox_extent[:, 1]
    pose_spread = torch.linalg.norm(bbox_extent, dim=-1)

    rigid_features = torch.stack(
        [
            _safe_mean(bone_deviation),
            _safe_max(bone_deviation),
            _safe_quantile(bone_deviation.flatten(), 0.95),
            _safe_quantile(bone_deviation.flatten(), 0.99),
            _safe_mean((bone_deviation > 0.05).float()),
        ]
    )
    dynamics_features = torch.stack(
        [
            _safe_mean(joint_speed),
            _safe_std(joint_speed),
            _kurtosis(joint_speed.flatten()),
            _safe_max(joint_speed),
            _safe_mean(jerk_norm),
            _safe_quantile(jerk_norm.flatten(), 0.95),
            _high_frequency_ratio(joint_speed.mean(dim=1)),
            _safe_max(accel_norm),
        ]
    )
    contact_features = torch.stack(
        [
            _safe_mean(active_contact_speed),
            _safe_quantile(active_contact_speed.flatten(), 0.95),
            _autocorr_peak(contact_binary),
            _safe_mean(contact_binary),
            _safe_mean(landing_cluster),
            _safe_mean(support_offset),
        ]
    )
    global_features = torch.stack(
        [
            pose_center_smoothness,
            _safe_std(kinetic_energy_series),
            _safe_std(torch.log1p(bbox_volume)),
            _safe_std(bbox_height),
            _safe_mean(pose_spread),
            _safe_std(positions[:, 0, 1]),
            _safe_mean(torch.linalg.norm(root_rot6d[1:] - root_rot6d[:-1], dim=-1)),
        ]
    )
    return torch.cat([rigid_features, dynamics_features, contact_features, symmetry_metrics_tensor, global_features], dim=0)


def extract_physics_features(
    motion: torch.Tensor,
    n_joints: torch.Tensor,
    lengths: torch.Tensor,
    object_types: Sequence[str],
    metadata_lookup: Mapping[str, SkeletonMetadata],
) -> torch.Tensor:
    if motion.ndim != 4:
        raise ValueError(f"Expected [B, J, F, T] motion tensor, got {tuple(motion.shape)}")
    batch_features = []
    for batch_index, object_type in enumerate(object_types):
        length = max(1, int(lengths[batch_index].item()))
        joint_count = max(1, int(n_joints[batch_index].item()))
        sample_motion = motion[batch_index, :joint_count, :, :length]
        batch_features.append(_extract_single_sample_features(sample_motion, length, metadata_lookup[str(object_type)]))
    return torch.stack(batch_features, dim=0).to(device=motion.device, dtype=motion.dtype)
