from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
from data_loaders.skeleton_metadata import SkeletonMetadata


PHYSICS_FEATURE_DIM = 30
_INDEX_TENSOR_CACHE: dict[tuple[tuple[int, ...], str], torch.Tensor] = {}
_METADATA_FEATURE_CACHE: dict[tuple[str, str], tuple] = {}
_MIN_ABSOLUTE_BONE_BASELINE = 1e-4
_MIN_RELATIVE_BONE_BASELINE_RATIO = 0.01


def _cached_index_tensor(indices: Sequence[int], device: torch.device) -> torch.Tensor:
    key = (tuple(int(index) for index in indices), str(device))
    tensor = _INDEX_TENSOR_CACHE.get(key)
    if tensor is not None and tensor.is_inference():
        with torch.inference_mode(False):
            tensor = tensor.clone()
        _INDEX_TENSOR_CACHE[key] = tensor
    if tensor is None:
        with torch.inference_mode(False):
            tensor = torch.as_tensor(key[0], dtype=torch.long, device=device)
        _INDEX_TENSOR_CACHE[key] = tensor
    return tensor


def _get_cached_metadata_indices(metadata: SkeletonMetadata, joint_count: int, device: torch.device):
    """Cache and return per-object-type index tensors derived from metadata.
    
    This avoids recomputing filtered index lists and tensors for the same
    object_type on every training step.
    """
    cache_key = (metadata.object_type, str(device))
    cached = _METADATA_FEATURE_CACHE.get(cache_key)
    if cached is not None and cached[0] == joint_count:
        return cached[1]
    
    # Compute all index tensors once
    result = {}
    
    # Edge indices filtered by joint_count
    if metadata.edge_child_indices:
        edge_child_indices = [index for index in metadata.edge_child_indices if index < joint_count]
        edge_parent_indices = metadata.edge_parent_indices[: len(edge_child_indices)]
    else:
        edge_child_indices = []
        edge_parent_indices = []
    result['edge_child_indices'] = edge_child_indices
    result['edge_parent_indices'] = edge_parent_indices
    if edge_child_indices:
        result['child_index_tensor'] = _cached_index_tensor(edge_child_indices, device)
        result['parent_index_tensor'] = _cached_index_tensor(edge_parent_indices, device)
    
    # Contact indices
    contact_indices = metadata.contact_joints[:]
    if contact_indices:
        valid_contact_indices = [index for index in contact_indices if index < joint_count]
    else:
        valid_contact_indices = []
    result['valid_contact_indices'] = valid_contact_indices
    if valid_contact_indices:
        result['contact_index_tensor'] = _cached_index_tensor(valid_contact_indices, device)
    
    # Symmetry indices
    symmetry_left_indices = [index for index in metadata.symmetry_left_indices if index < joint_count]
    symmetry_right_indices = [index for index in metadata.symmetry_right_indices if index < joint_count]
    result['symmetry_left_indices'] = symmetry_left_indices
    result['symmetry_right_indices'] = symmetry_right_indices
    result['has_symmetry'] = bool(symmetry_left_indices and len(symmetry_left_indices) == len(symmetry_right_indices))
    if result['has_symmetry']:
        result['left_index_tensor'] = _cached_index_tensor(symmetry_left_indices, device)
        result['right_index_tensor'] = _cached_index_tensor(symmetry_right_indices, device)
    
    # Cache and return
    _METADATA_FEATURE_CACHE[cache_key] = (joint_count, result)
    return result


def _as_float_tensor(values: torch.Tensor) -> torch.Tensor:
    return values if values.dtype in (torch.float32, torch.float64) else values.float()


def _ensure_autograd_compatible(values: torch.Tensor | None) -> torch.Tensor | None:
    if values is None or not values.is_inference():
        return values
    with torch.inference_mode(False):
        return values.clone()


def _safe_quantile(values: torch.Tensor, q: float) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros(())
    flat = _as_float_tensor(values).flatten()
    if flat.numel() == 1:
        return flat[0]
    rank = float(q) * float(flat.numel() - 1)
    lower_index = int(math.floor(rank))
    upper_index = int(math.ceil(rank))
    lower_value = flat.kthvalue(lower_index + 1).values
    if upper_index == lower_index:
        return lower_value
    upper_value = flat.kthvalue(upper_index + 1).values
    return torch.lerp(lower_value, upper_value, flat.new_tensor(rank - lower_index))


def _safe_mean(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros(())
    return _as_float_tensor(values).mean()


def _safe_std(values: torch.Tensor) -> torch.Tensor:
    if values.numel() <= 1:
        return values.new_zeros(())
    return _as_float_tensor(values).std(unbiased=False)


def _safe_max(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros(())
    return _as_float_tensor(values).max()


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return numerator / denominator.clamp_min(1e-6)


def _kurtosis(values: torch.Tensor) -> torch.Tensor:
    if values.numel() <= 3:
        return values.new_zeros(())
    float_values = _as_float_tensor(values)
    centered = float_values - float_values.mean()
    variance = centered.pow(2).mean().clamp_min(1e-6)
    return centered.pow(4).mean() / variance.pow(2)


def _autocorr_peak(signal: torch.Tensor, max_lag: int = 12) -> torch.Tensor:
    if signal.numel() <= 2:
        return signal.new_zeros(())
    centered_signal = _as_float_tensor(signal)
    centered = centered_signal - centered_signal.mean()
    denom = centered.pow(2).sum().clamp_min(1e-6)
    T = centered.numel()
    max_lag = min(max_lag, T - 1)
    peak = centered.new_zeros(())
    for lag in range(1, max_lag + 1):
        corr = (centered[:T - lag] * centered[lag:]).sum() / denom
        peak = torch.maximum(peak, corr)
    return peak


def _phase_offset(left: torch.Tensor, right: torch.Tensor, max_lag: int = 8) -> torch.Tensor:
    if left.numel() <= 2 or right.numel() <= 2:
        return left.new_zeros(())
    left_float = _as_float_tensor(left)
    right_float = _as_float_tensor(right)
    T = left_float.numel()
    max_lag = min(max_lag, T - 1)
    best_score = None
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            lhs = left_float[-lag:]
            rhs = right_float[:lag + T]
        elif lag > 0:
            lhs = left_float[:T - lag]
            rhs = right_float[lag:]
        else:
            lhs = left_float
            rhs = right_float
        if lhs.numel() <= 1 or rhs.numel() <= 1:
            continue
        score = (lhs * rhs).mean()
        if best_score is None or score > best_score:
            best_score = score
            best_lag = lag
    return left_float.new_tensor(float(best_lag) / max(float(T), 1.0))


def _high_frequency_ratio(signal: torch.Tensor) -> torch.Tensor:
    if signal.numel() <= 4:
        return signal.new_zeros(())
    spectrum = torch.fft.rfft(_as_float_tensor(signal), dim=0)
    power = spectrum.abs().pow(2)
    if power.shape[0] <= 2:
        return signal.new_zeros(())
    split_index = max(1, int(power.shape[0] * 0.6))
    high = power[split_index:].sum()
    total = power[1:].sum().clamp_min(1e-6)
    return high / total


def _maybe_denormalize_motion(
    motion: torch.Tensor,
    feature_mean: torch.Tensor | None,
    feature_std: torch.Tensor | None,
) -> torch.Tensor:
    if feature_mean is None or feature_std is None:
        return motion
    mean = feature_mean.to(device=motion.device, dtype=motion.dtype)
    std = feature_std.to(device=motion.device, dtype=motion.dtype).clamp_min(1e-6)
    if mean.ndim != 2 or std.ndim != 2:
        raise ValueError(
            f"Expected feature_mean/feature_std to have shape [J, F], got {tuple(mean.shape)} and {tuple(std.shape)}"
        )
    if mean.shape != motion.shape[:2] or std.shape != motion.shape[:2]:
        raise ValueError(
            f"feature_mean/feature_std shape mismatch for motion slice: motion={tuple(motion.shape[:2])}, "
            f"mean={tuple(mean.shape)}, std={tuple(std.shape)}"
        )
    return motion * std.unsqueeze(-1) + mean.unsqueeze(-1)


def _extract_single_sample_features(
    motion: torch.Tensor,
    length: int,
    metadata: SkeletonMetadata,
) -> torch.Tensor:
    # Dataset-provided lengths may refer to the source clip before temporal
    # cropping; physics features must only read the frames present in `motion`.
    length = max(1, min(int(length), int(motion.shape[-1])))
    joint_count = min(int(metadata.n_joints), int(motion.shape[0]))
    motion = motion[:joint_count, :, :length]
    positions = motion[:, :3, :].permute(2, 0, 1)
    velocities = motion[:, 9:12, :].permute(2, 0, 1)
    contact_channel = motion[:, 12, :].permute(1, 0)
    root_rot6d = motion[0, 3:9, :].permute(1, 0)

    # Use cached metadata indices to avoid recomputing per-sample
    cached_md = _get_cached_metadata_indices(metadata, joint_count, positions.device)

    if cached_md.get('edge_child_indices'):
        child_index_tensor = cached_md['child_index_tensor']
        parent_index_tensor = cached_md['parent_index_tensor']
        bone_lengths = torch.linalg.norm(positions[:, child_index_tensor] - positions[:, parent_index_tensor], dim=-1)
        baseline = bone_lengths.median(dim=0).values
        positive_baseline = baseline[baseline > 1e-6]
        if positive_baseline.numel():
            min_valid_baseline = torch.maximum(
                baseline.new_tensor(_MIN_ABSOLUTE_BONE_BASELINE),
                positive_baseline.median() * _MIN_RELATIVE_BONE_BASELINE_RATIO,
            )
            valid_edge_mask = baseline >= min_valid_baseline
        else:
            valid_edge_mask = torch.zeros_like(baseline, dtype=torch.bool)
        if bool(valid_edge_mask.any().item()):
            valid_baseline = baseline[valid_edge_mask].clamp_min(1e-6)
            valid_bone_lengths = bone_lengths[:, valid_edge_mask]
            bone_deviation = (valid_bone_lengths - valid_baseline.unsqueeze(0)).abs() / valid_baseline.unsqueeze(0)
        else:
            bone_deviation = positions.new_zeros((max(length, 1), 1))
    else:
        bone_deviation = positions.new_zeros((max(length, 1), 1))

    joint_speed = torch.linalg.norm(velocities, dim=-1)
    accel = velocities[1:] - velocities[:-1] if length > 1 else velocities.new_zeros((0, joint_count, 3))
    accel_norm = torch.linalg.norm(accel, dim=-1) if accel.numel() else velocities.new_zeros((0, joint_count))
    jerk = accel[1:] - accel[:-1] if accel.shape[0] > 1 else velocities.new_zeros((0, joint_count, 3))
    jerk_norm = torch.linalg.norm(jerk, dim=-1) if jerk.numel() else velocities.new_zeros((0, joint_count))
    pose_center = positions.mean(dim=1)
    pose_center_xz = pose_center[:, [0, 2]]

    if cached_md.get('valid_contact_indices'):
        contact_index_tensor = cached_md['contact_index_tensor']
        contact_mask = contact_channel[:, contact_index_tensor] > 0.5
        contact_speed = torch.linalg.norm(velocities[:, contact_index_tensor][:, :, [0, 2]], dim=-1)
        active_contact_speed = contact_speed[contact_mask]
        contact_binary = contact_mask.any(dim=1).float()
        contact_positions = positions[:, contact_index_tensor]
        active_positions = contact_positions[contact_mask]
        if active_positions.numel():
            landing_cluster = torch.linalg.norm(active_positions[:, [0, 2]] - active_positions[:, [0, 2]].mean(dim=0), dim=-1)
            support_centroid = contact_positions[:, :, [0, 2]].mean(dim=1)
            support_offset = torch.linalg.norm(pose_center_xz - support_centroid, dim=-1)
        else:
            landing_cluster = positions.new_zeros((0,))
            support_offset = positions.new_zeros((0,))
    else:
        active_contact_speed = positions.new_zeros((0,))
        contact_binary = positions.new_zeros((length,))
        landing_cluster = positions.new_zeros((0,))
        support_offset = positions.new_zeros((0,))

    if cached_md.get('has_symmetry'):
        left_index_tensor = cached_md['left_index_tensor']
        right_index_tensor = cached_md['right_index_tensor']
        left_speeds = joint_speed[:, left_index_tensor]
        right_speeds = joint_speed[:, right_index_tensor]
        left_centered = left_speeds - left_speeds.mean(dim=0, keepdim=True)
        right_centered = right_speeds - right_speeds.mean(dim=0, keepdim=True)
        denom = (left_centered.pow(2).sum(dim=0) * right_centered.pow(2).sum(dim=0)).sqrt().clamp_min(1e-6)
        corr = (left_centered * right_centered).sum(dim=0) / denom
        left_mean = left_speeds.mean(dim=0)
        right_mean = right_speeds.mean(dim=0)
        amp_ratio = torch.minimum(left_mean, right_mean) / torch.maximum(left_mean, right_mean).clamp_min(1e-6)
        left_energy = left_speeds.pow(2).mean(dim=0)
        right_energy = right_speeds.pow(2).mean(dim=0)
        energy_ratio = torch.minimum(left_energy, right_energy) / torch.maximum(left_energy, right_energy).clamp_min(1e-6)
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

    pose_center_smoothness = _safe_mean(torch.linalg.norm(pose_center[2:] - 2.0 * pose_center[1:-1] + pose_center[:-2], dim=-1))
    kinetic_energy_series = joint_speed.pow(2).mean(dim=1)
    bbox_min = positions.min(dim=1).values
    bbox_max = positions.max(dim=1).values
    bbox_extent = (bbox_max - bbox_min).clamp_min(1e-6)
    bbox_volume = bbox_extent.prod(dim=-1)
    bbox_height = bbox_extent[:, 1]
    pose_spread = torch.linalg.norm(bbox_extent, dim=-1)

    bone_deviation_flat = bone_deviation.flatten()
    rigid_features = torch.stack(
        [
            _safe_mean(bone_deviation),
            _safe_max(bone_deviation),
            _safe_quantile(bone_deviation_flat, 0.95),
            _safe_quantile(bone_deviation_flat, 0.99),
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
    *,
    feature_mean: torch.Tensor | None = None,
    feature_std: torch.Tensor | None = None,
    differentiable: bool = False,
) -> torch.Tensor:
    if motion.ndim != 4:
        raise ValueError(f"Expected [B, J, F, T] motion tensor, got {tuple(motion.shape)}")
    if (feature_mean is None) != (feature_std is None):
        raise ValueError('feature_mean and feature_std must either both be provided or both be None')
    if feature_mean is not None:
        if feature_mean.ndim != 3 or feature_std is None or feature_std.ndim != 3:
            raise ValueError(
                f"Expected feature_mean/feature_std to have shape [B, J, F], got "
                f"{tuple(feature_mean.shape)} and {tuple(feature_std.shape) if feature_std is not None else None}"
            )
        if tuple(feature_mean.shape) != tuple(motion.shape[:3]) or tuple(feature_std.shape) != tuple(motion.shape[:3]):
            raise ValueError(
                f"feature_mean/feature_std shape mismatch for motion batch: motion={tuple(motion.shape)}, "
                f"mean={tuple(feature_mean.shape)}, std={tuple(feature_std.shape)}"
            )

    B = motion.shape[0]
    lengths_cpu = lengths.detach().cpu().tolist()
    joint_counts_cpu = n_joints.detach().cpu().tolist()

    all_features = [None] * B
    for i in range(B):
        length = max(1, min(int(lengths_cpu[i]), int(motion.shape[-1])))
        joint_count = max(1, int(joint_counts_cpu[i]))
        metadata = metadata_lookup[str(object_types[i])]
        sample_motion = motion[i, :joint_count, :, :length]
        sample_mean = None if feature_mean is None else feature_mean[i, :joint_count, :]
        sample_std = None if feature_std is None else feature_std[i, :joint_count, :]

        if differentiable:
            with torch.inference_mode(False):
                sample_motion = _ensure_autograd_compatible(sample_motion)
                sample_mean = _ensure_autograd_compatible(sample_mean)
                sample_std = _ensure_autograd_compatible(sample_std)
                denormalized_motion = _maybe_denormalize_motion(sample_motion, sample_mean, sample_std)
                all_features[i] = _extract_single_sample_features(denormalized_motion, length, metadata)
        else:
            with torch.inference_mode():
                denormalized_motion = _maybe_denormalize_motion(sample_motion, sample_mean, sample_std)
                all_features[i] = _extract_single_sample_features(denormalized_motion, length, metadata)

    if differentiable:
        with torch.inference_mode(False):
            return torch.stack(all_features, dim=0).to(device=motion.device, dtype=motion.dtype)
    return torch.stack(all_features, dim=0).to(device=motion.device, dtype=motion.dtype)
