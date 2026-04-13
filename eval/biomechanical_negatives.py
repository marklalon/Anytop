from __future__ import annotations

import random
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from data_loaders.skeleton_metadata import SkeletonMetadata


NEGATIVE_KINDS = (
    "foot_slip",
    "bone_length_drift",
    "symmetry_break",
    "joint_jitter",
    "acceleration_burst",
    "interpenetration",
    "time_warp",
)

NEGATIVE_STRENGTH_SCALES = (1.0, 1.5, 2.0)

# Salience thresholds are evaluated in **denormalized meter space** (delta is
# multiplied by feature_std before comparison, which for non-root joints equals
# the physical XYZ std in meters). Tuned so that post-BVH the perturbation is
# unmistakable to the eye rather than a hairline drift on a single joint.
NEGATIVE_MIN_MAX_POSITION_DELTA = 0.25      # at least one joint moves >=25 cm
NEGATIVE_MIN_MEAN_POSITION_DELTA = 0.015    # average displacement across joints/frames
NEGATIVE_CHANGED_JOINT_DELTA = 0.08         # joint counts as "changed" if it moves >=8 cm somewhere
NEGATIVE_MIN_CHANGED_JOINTS = 2


def _rand_range(sample: torch.Tensor, low: float, high: float) -> float:
    # CPU-side RNG — every caller of this used to wrap the result in `float(...)`,
    # which forced a CUDA sync. `fixseed` seeds python's `random`, so this keeps
    # determinism without paying a device round-trip per draw.
    del sample
    return random.uniform(low, high)


def _choose_joint(sample: torch.Tensor, metadata: SkeletonMetadata, *, include_root: bool = False) -> int:
    del sample
    candidates = tuple(range(metadata.n_joints)) if include_root else metadata.non_root_joints
    if not candidates:
        return 0
    return int(candidates[random.randrange(len(candidates))])


def _subtree_indices(metadata: SkeletonMetadata, joint_index: int) -> list[int]:
    if 0 <= joint_index < len(metadata.subtree_indices):
        return list(metadata.subtree_indices[joint_index])
    return [joint_index]


def _non_root_subtree(metadata: SkeletonMetadata, joint_index: int) -> list[int]:
    return [index for index in _subtree_indices(metadata, joint_index) if index > 0]


def _kinematic_chain_indices(metadata: SkeletonMetadata, joint_index: int, *, max_depth: int = 3) -> list[int]:
    if joint_index < 0:
        return []
    indices = [joint_index]
    current_index = joint_index
    while len(indices) < max_depth:
        parent_index = metadata.parents[current_index] if current_index < len(metadata.parents) else -1
        if parent_index < 0:
            break
        indices.append(parent_index)
        current_index = parent_index
    return indices


def _refresh_derived_channels(sample: torch.Tensor, metadata: SkeletonMetadata, length: int) -> None:
    positions = sample[:, :3, :length]
    if length > 1:
        velocity = torch.zeros_like(positions)
        velocity[:, :, :-1] = positions[:, :, 1:] - positions[:, :, :-1]
        velocity[:, :, -1] = velocity[:, :, max(length - 2, 0)]
        sample[:, 9:12, :length] = velocity

    contact_channel = torch.zeros((sample.shape[0], length), dtype=sample.dtype, device=sample.device)
    contact_indices = [index for index in metadata.contact_joints if index < sample.shape[0]]
    if contact_indices:
        contact_positions = positions[contact_indices]
        ground = contact_positions[:, 1, :].amin(dim=1, keepdim=True)
        horiz_speed = torch.linalg.norm(sample[contact_indices, 9:12, :length][:, [0, 2], :], dim=1)
        close_to_ground = (contact_positions[:, 1, :] - ground) <= 0.04
        contact_channel[contact_indices] = (close_to_ground & (horiz_speed <= 0.04)).to(sample.dtype)
    sample[:, 12, :length] = contact_channel


def _motion_delta_stats(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    length: int,
    *,
    sample_std: torch.Tensor | None,
) -> tuple[float, float, int, float]:
    # Evaluate saliency only on non-root joints — root channels 0-2 encode yaw
    # angular velocity and linear xz velocity, not positions, so std-weighted
    # deltas there have different units and would pollute the metric.
    position_delta = (candidate[1:, :3, :length] - reference[1:, :3, :length]).abs()
    if sample_std is not None:
        std_positions = sample_std[1:, :3].to(device=candidate.device, dtype=candidate.dtype).unsqueeze(-1)
        position_delta = position_delta * std_positions
    if position_delta.numel() == 0:
        return 0.0, 0.0, 0, 0.0
    per_joint_max = position_delta.amax(dim=(1, 2))
    # Stack into a single 3-element tensor so we pay one device->host sync
    # per saliency check instead of three. The comparison and change count are
    # done on GPU before the single `tolist()` transfer.
    stats = torch.stack(
        (
            position_delta.max(),
            position_delta.mean(),
            (per_joint_max >= NEGATIVE_CHANGED_JOINT_DELTA).sum().to(position_delta.dtype),
        )
    ).tolist()
    return float(stats[0]), float(stats[1]), int(stats[2]), 0.0


def _negative_is_salient_enough(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    length: int,
    *,
    sample_std: torch.Tensor | None,
) -> bool:
    max_delta, mean_delta, changed_joint_count, _ = _motion_delta_stats(
        reference,
        candidate,
        length,
        sample_std=sample_std,
    )
    if max_delta < NEGATIVE_MIN_MAX_POSITION_DELTA:
        return False
    if mean_delta < NEGATIVE_MIN_MEAN_POSITION_DELTA and changed_joint_count < NEGATIVE_MIN_CHANGED_JOINTS:
        return False
    return True


def _apply_position_shift_meters(
    sample: torch.Tensor,
    joint_indices: Sequence[int],
    start_index: int,
    end_index: int,
    shift_meters: torch.Tensor,
    *,
    sample_std: torch.Tensor | None,
) -> None:
    """Add a meter-space translation to channels 0-2 of each given joint over the
    [start, end) frame window. The root joint is always skipped because its
    channels 0-2 are yaw angular velocity / linear xz velocity / (unused), not
    positions. `shift_meters` is either shape (3,) for a constant translation or
    (3, T) for a per-frame shift."""
    if end_index <= start_index:
        return
    valid_joints = [int(j) for j in joint_indices if 0 < int(j) < sample.shape[0]]
    if not valid_joints:
        return
    frame_count = end_index - start_index
    if shift_meters.ndim == 1:
        shift = shift_meters.view(1, 3, 1).expand(len(valid_joints), 3, frame_count)
    else:
        shift = shift_meters
        if shift.shape[-1] != frame_count:
            shift = shift[..., :frame_count]
        if shift.ndim == 2:
            shift = shift.unsqueeze(0).expand(len(valid_joints), 3, frame_count)

    idx = torch.as_tensor(valid_joints, device=sample.device, dtype=torch.long)
    if sample_std is None:
        scaled_shift = shift
    else:
        std_xyz = sample_std[idx, :3].to(device=sample.device, dtype=sample.dtype).clamp_min(1e-4)
        inv_std = (1.0 / std_xyz).unsqueeze(-1)
        scaled_shift = shift * inv_std
    sample[idx, :3, start_index:end_index] = (
        sample[idx, :3, start_index:end_index] + scaled_shift
    )


def _joint_meter_positions(
    sample: torch.Tensor,
    joint_index: int,
    length: int,
    sample_std: torch.Tensor | None,
) -> torch.Tensor:
    if sample_std is None:
        return sample[joint_index, :3, :length].clone()
    std_xyz = sample_std[joint_index, :3].to(device=sample.device, dtype=sample.dtype).clamp_min(1e-4)
    return sample[joint_index, :3, :length] * std_xyz.view(3, 1)


def _apply_foot_slip(
    sample: torch.Tensor,
    metadata: SkeletonMetadata,
    length: int,
    *,
    strength_scale: float = 1.0,
    sample_std: torch.Tensor | None = None,
) -> None:
    contact_indices = [index for index in metadata.contact_joints if 0 < index < sample.shape[0]]
    if not contact_indices:
        contact_indices = [index for index in metadata.end_effector_joints if 0 < index < sample.shape[0]]
    if not contact_indices:
        return
    joint_index = contact_indices[random.randrange(len(contact_indices))]
    contact_signal = sample[joint_index, 12, :length] > 0.5
    active_frames = torch.nonzero(contact_signal, as_tuple=False).flatten()
    min_window = max(10, length // 3)
    if active_frames.numel() >= min_window:
        # One host transfer for the two endpoints instead of two.
        endpoints = active_frames[[0, -1]].tolist()
        start_index = int(endpoints[0])
        end_index = int(endpoints[1]) + 1
    else:
        start_index = max(0, length // 4)
        end_index = min(length, start_index + max(min_window, length // 2))
    frame_count = end_index - start_index
    if frame_count <= 0:
        return

    drift_x_total = float(_rand_range(sample, 0.35, 0.75) * strength_scale)
    drift_z_total = float(_rand_range(sample, -0.60, 0.60) * strength_scale)
    drift_y_total = float(_rand_range(sample, 0.08, 0.22) * strength_scale)
    ramp = torch.linspace(0.0, 1.0, frame_count, device=sample.device, dtype=sample.dtype)
    shift = torch.stack(
        [ramp * drift_x_total, ramp * drift_y_total, ramp * drift_z_total],
        dim=0,
    )
    _apply_position_shift_meters(sample, [joint_index], start_index, end_index, shift, sample_std=sample_std)


def _apply_bone_length_drift(
    sample: torch.Tensor,
    metadata: SkeletonMetadata,
    length: int,
    *,
    strength_scale: float = 1.0,
    sample_std: torch.Tensor | None = None,
) -> None:
    non_root = [index for index in metadata.non_root_joints if 0 < index < sample.shape[0]]
    if not non_root:
        return
    joint_index = non_root[random.randrange(len(non_root))]
    parent_index = metadata.parents[joint_index] if joint_index < len(metadata.parents) else -1
    subtree = _non_root_subtree(metadata, joint_index)
    if not subtree:
        return

    scale_values = torch.linspace(
        1.0,
        1.0 + float(_rand_range(sample, 0.1, 0.6) * strength_scale),
        length,
        device=sample.device,
        dtype=sample.dtype,
    )

    child_pos_m = _joint_meter_positions(sample, joint_index, length, sample_std)
    if parent_index > 0:
        # Regular bone: stretch along the (child - parent) vector.
        parent_pos_m = _joint_meter_positions(sample, parent_index, length, sample_std)
        bone_vector_m = child_pos_m - parent_pos_m
    else:
        # Bone is rooted at the skeleton root (joint 0). Root channels 0-2 are
        # yaw/linear velocity, *not* positions, so we must NOT subtract them.
        # In the RIC yaw-stabilized frame the root sits at the origin, so the
        # bone vector equals the child's RIC position directly.
        bone_vector_m = child_pos_m

    shift_m = bone_vector_m * (scale_values.view(1, -1) - 1.0)
    _apply_position_shift_meters(sample, subtree, 0, length, shift_m, sample_std=sample_std)


def _apply_symmetry_break(
    sample: torch.Tensor,
    metadata: SkeletonMetadata,
    length: int,
    *,
    strength_scale: float = 1.0,
    sample_std: torch.Tensor | None = None,
) -> None:
    valid_pairs = [
        pair
        for pair in metadata.symmetric_joint_pairs
        if 0 < int(pair[0]) < sample.shape[0] and 0 < int(pair[1]) < sample.shape[0]
    ]
    if not valid_pairs:
        return
    pair = valid_pairs[random.randrange(len(valid_pairs))]
    dominant_joint, passive_joint = int(pair[0]), int(pair[1])
    dominant_scale = 1.0 + float(_rand_range(sample, 0.45, 0.85) * strength_scale)
    passive_scale = 1.0 / max(dominant_scale, 1e-6)

    for side_joint, side_scale in ((dominant_joint, dominant_scale), (passive_joint, passive_scale)):
        pos_m = _joint_meter_positions(sample, side_joint, length, sample_std)
        shift_m = pos_m * (side_scale - 1.0)
        subtree = _non_root_subtree(metadata, side_joint)
        if not subtree:
            subtree = [side_joint]
        _apply_position_shift_meters(sample, subtree, 0, length, shift_m, sample_std=sample_std)


def _apply_joint_jitter(
    sample: torch.Tensor,
    metadata: SkeletonMetadata,
    length: int,
    *,
    strength_scale: float = 1.0,
    sample_std: torch.Tensor | None = None,
) -> None:
    non_root = [index for index in metadata.non_root_joints if 0 < index < sample.shape[0]]
    if not non_root:
        return
    joint_index = non_root[random.randrange(len(non_root))]
    freq = random.randrange(8, 18)
    amplitude = float(_rand_range(sample, 0.05, 0.1) * strength_scale)
    t = torch.linspace(0.0, 1.0, length, device=sample.device, dtype=sample.dtype)
    phase = torch.rand((3,), device=sample.device, dtype=sample.dtype) * (2.0 * torch.pi)
    angles = (2.0 * torch.pi * float(freq)) * t.unsqueeze(0) + phase.unsqueeze(-1)
    jitter = amplitude * torch.sin(angles)
    _apply_position_shift_meters(sample, [joint_index], 0, length, jitter, sample_std=sample_std)


def _apply_acceleration_burst(
    sample: torch.Tensor,
    metadata: SkeletonMetadata,
    length: int,
    *,
    strength_scale: float = 1.0,
    sample_std: torch.Tensor | None = None,
) -> None:
    if length < 3:
        return
    non_root = [index for index in metadata.non_root_joints if 0 < index < sample.shape[0]]
    if not non_root:
        return
    joint_index = non_root[random.randrange(len(non_root))]
    center = random.randrange(1, max(length - 1, 2))
    radius = max(4, length // 5)
    start_index = max(0, center - radius)
    end_index = min(length, center + radius + 1)
    frame_count = end_index - start_index
    if frame_count <= 0:
        return
    pulse_t = torch.linspace(-1.0, 1.0, frame_count, device=sample.device, dtype=sample.dtype)
    gaussian = torch.exp(-pulse_t.pow(2) / 0.25)
    direction = F.normalize(torch.randn((3,), device=sample.device, dtype=sample.dtype), dim=0)
    magnitude = float(_rand_range(sample, 0.25, 0.50) * strength_scale)
    shift = direction.view(3, 1) * (magnitude * gaussian.unsqueeze(0))
    _apply_position_shift_meters(sample, [joint_index], start_index, end_index, shift, sample_std=sample_std)


def _apply_interpenetration(
    sample: torch.Tensor,
    metadata: SkeletonMetadata,
    length: int,
    *,
    strength_scale: float = 1.0,
    sample_std: torch.Tensor | None = None,
) -> None:
    end_effectors = [index for index in metadata.end_effector_joints if 0 < index < sample.shape[0]]
    if not end_effectors:
        return
    joint_index = end_effectors[random.randrange(len(end_effectors))]
    affected_joints = [
        j
        for j in _kinematic_chain_indices(metadata, joint_index, max_depth=5)
        if 0 < j < sample.shape[0]
    ]
    if not affected_joints:
        return

    non_root_range = list(range(1, metadata.n_joints))
    if not non_root_range:
        return
    # Build pose_center in meters, averaging only over non-root joints so that
    # root yaw/linear velocity channels do not pollute the target.
    if sample_std is None:
        pose_center = sample[non_root_range, :3, :length].mean(dim=0)
    else:
        std_positions = sample_std[:, :3].to(device=sample.device, dtype=sample.dtype).clamp_min(1e-4)
        pose_positions_m = sample[non_root_range, :3, :length] * std_positions[non_root_range].unsqueeze(-1)
        pose_center = pose_positions_m.mean(dim=0)

    mix = min(0.98, 0.60 + _rand_range(sample, 0.25, 0.35) * strength_scale)
    # Compute per-depth mix weights and the combined shift for all affected
    # joints in one shot, then hand the stacked shift to the (vectorized)
    # position-shift helper. This collapses what was an O(depth) kernel loop
    # into a single broadcasted operation.
    joint_mixes = [max(0.55, min(0.98, mix * (1.0 - 0.08 * d))) for d in range(len(affected_joints))]
    joint_idx_tensor = torch.as_tensor(affected_joints, device=sample.device, dtype=torch.long)
    if sample_std is None:
        current_m = sample[joint_idx_tensor, :3, :length]
    else:
        std_xyz = sample_std[joint_idx_tensor, :3].to(
            device=sample.device, dtype=sample.dtype
        ).clamp_min(1e-4)
        current_m = sample[joint_idx_tensor, :3, :length] * std_xyz.unsqueeze(-1)
    mix_tensor = torch.tensor(joint_mixes, device=sample.device, dtype=sample.dtype).view(-1, 1, 1)
    shift_m = (pose_center.unsqueeze(0) - current_m) * mix_tensor
    _apply_position_shift_meters(
        sample, affected_joints, 0, length, shift_m, sample_std=sample_std
    )


def _warp_time_axis(values: torch.Tensor, *, strength_scale: float = 1.0) -> torch.Tensor:
    length = values.shape[-1]
    if length <= 4:
        return values
    split = max(2, length // 2)
    compressed_length = max(2, int(round(split / (1.0 + 0.60 * strength_scale))))
    first_half = torch.linspace(0.0, max(split - 1, 1), compressed_length, device=values.device, dtype=values.dtype)
    second_half = torch.linspace(split, length - 1, max(2, length - compressed_length), device=values.device, dtype=values.dtype)
    grid = torch.cat([first_half, second_half], dim=0)[:length].clamp(0, length - 1)
    floor = grid.floor().long()
    ceil = grid.ceil().long().clamp_max(length - 1)
    alpha = (grid - floor.to(values.dtype)).view(1, 1, -1)
    flat = values.view(-1, 1, length)
    warped = torch.lerp(flat[..., floor], flat[..., ceil], alpha)
    return warped.view_as(values)


def _apply_time_warp(sample: torch.Tensor, length: int, *, strength_scale: float = 1.0) -> None:
    sample[:, :, :length] = _warp_time_axis(sample[:, :, :length], strength_scale=strength_scale)


def _apply_negative_kind_with_std(
    sample: torch.Tensor,
    metadata: SkeletonMetadata,
    length: int,
    kind: str,
    *,
    strength_scale: float,
    sample_std: torch.Tensor | None,
) -> None:
    if kind == "foot_slip":
        _apply_foot_slip(sample, metadata, length, strength_scale=strength_scale, sample_std=sample_std)
    elif kind == "bone_length_drift":
        _apply_bone_length_drift(sample, metadata, length, strength_scale=strength_scale, sample_std=sample_std)
    elif kind == "symmetry_break":
        _apply_symmetry_break(sample, metadata, length, strength_scale=strength_scale, sample_std=sample_std)
    elif kind == "joint_jitter":
        _apply_joint_jitter(sample, metadata, length, strength_scale=strength_scale, sample_std=sample_std)
    elif kind == "acceleration_burst":
        _apply_acceleration_burst(sample, metadata, length, strength_scale=strength_scale, sample_std=sample_std)
    elif kind == "interpenetration":
        _apply_interpenetration(sample, metadata, length, strength_scale=strength_scale, sample_std=sample_std)
    elif kind == "time_warp":
        _apply_time_warp(sample, length, strength_scale=strength_scale)


def _apply_negative_kind(sample: torch.Tensor, metadata: SkeletonMetadata, length: int, kind: str, *, strength_scale: float) -> None:
    return _apply_negative_kind_with_std(sample, metadata, length, kind, strength_scale=strength_scale, sample_std=None)


def generate_biomechanical_negative_batch(
    motion: torch.Tensor,
    n_joints: torch.Tensor,
    lengths: torch.Tensor,
    object_types: Sequence[str],
    metadata_lookup: Mapping[str, SkeletonMetadata],
    feature_std: torch.Tensor | None = None,
    negative_kinds: Sequence[str] | None = None,
) -> dict[str, object]:
    sampled_negative_kinds = tuple(negative_kinds) if negative_kinds is not None else NEGATIVE_KINDS
    if not sampled_negative_kinds:
        raise ValueError("negative_kinds must contain at least one entry")

    negative_motion = motion.clone()
    negative_kinds: list[str] = []
    # Lift the length readback into one host transfer for the whole batch
    # instead of calling .item() per sample inside the hot loop.
    lengths_cpu = lengths.detach().to("cpu", non_blocking=True).tolist()

    for batch_index, object_type in enumerate(object_types):
        metadata = metadata_lookup[str(object_type)]
        length = max(1, int(lengths_cpu[batch_index]))
        sample = negative_motion[batch_index, : metadata.n_joints, :, :length]
        reference_sample = sample.clone()
        sample_std = None if feature_std is None else feature_std[batch_index, : metadata.n_joints, :]
        kind = sampled_negative_kinds[random.randrange(len(sampled_negative_kinds))]
        negative_kinds.append(kind)

        applied = False
        for attempt_index, strength_scale in enumerate(NEGATIVE_STRENGTH_SCALES):
            # On the very first attempt `sample` already equals `reference_sample`
            # (we just cloned it), so skip the redundant device-to-device copy.
            if attempt_index > 0:
                sample.copy_(reference_sample)
            _apply_negative_kind_with_std(sample, metadata, length, kind, strength_scale=strength_scale, sample_std=sample_std)
            _refresh_derived_channels(sample, metadata, length)
            if _negative_is_salient_enough(reference_sample, sample, length, sample_std=sample_std):
                applied = True
                break

        if not applied:
            # Hard fallback: apply the requested kind at boosted strength, then
            # stack interpenetration on top so the sample is definitely salient.
            sample.copy_(reference_sample)
            _apply_negative_kind_with_std(
                sample,
                metadata,
                length,
                kind,
                strength_scale=NEGATIVE_STRENGTH_SCALES[-1] * 1.6,
                sample_std=sample_std,
            )
            if kind != "interpenetration":
                _apply_interpenetration(sample, metadata, length, strength_scale=1.8, sample_std=sample_std)
            _refresh_derived_channels(sample, metadata, length)

    return {
        "motion": negative_motion,
        "negative_kinds": negative_kinds,
    }
