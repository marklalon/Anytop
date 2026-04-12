from __future__ import annotations

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
    "cross_species_mismatch",
)


def _rand_range(sample: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return torch.empty((), device=sample.device, dtype=sample.dtype).uniform_(low, high)


def _choose_joint(sample: torch.Tensor, metadata: SkeletonMetadata, *, include_root: bool = False) -> int:
    candidates = tuple(range(metadata.n_joints)) if include_root else metadata.non_root_joints
    if not candidates:
        return 0
    random_index = int(torch.randint(0, len(candidates), (1,), device=sample.device).item())
    return int(candidates[random_index])


def _subtree_indices(metadata: SkeletonMetadata, joint_index: int) -> list[int]:
    if 0 <= joint_index < len(metadata.subtree_indices):
        return list(metadata.subtree_indices[joint_index])
    return [joint_index]


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


def _apply_foot_slip(sample: torch.Tensor, metadata: SkeletonMetadata, length: int) -> None:
    contact_indices = [index for index in metadata.contact_joints if index < sample.shape[0]]
    if not contact_indices:
        return
    joint_index = contact_indices[int(torch.randint(0, len(contact_indices), (1,), device=sample.device).item())]
    contact_signal = sample[joint_index, 12, :length] > 0.5
    active_frames = torch.nonzero(contact_signal, as_tuple=False).flatten()
    if active_frames.numel() < 3:
        active_frames = torch.arange(max(2, length // 3), min(length, max(3, 2 * length // 3)), device=sample.device)
    start_index = int(active_frames[0].item())
    end_index = int(active_frames[-1].item()) + 1
    drift = torch.zeros((2, end_index - start_index), dtype=sample.dtype, device=sample.device)
    drift[0] = torch.linspace(0.0, float(_rand_range(sample, 0.08, 0.20)), end_index - start_index, device=sample.device, dtype=sample.dtype)
    drift[1] = torch.linspace(0.0, float(_rand_range(sample, -0.18, 0.18)), end_index - start_index, device=sample.device, dtype=sample.dtype)
    sample[joint_index, 0, start_index:end_index] += drift[0]
    sample[joint_index, 2, start_index:end_index] += drift[1]


def _apply_bone_length_drift(sample: torch.Tensor, metadata: SkeletonMetadata, length: int) -> None:
    joint_index = _choose_joint(sample, metadata)
    parent_index = metadata.parents[joint_index] if joint_index < len(metadata.parents) else -1
    if parent_index < 0:
        return
    subtree = _subtree_indices(metadata, joint_index)
    scale = torch.linspace(1.0, float(_rand_range(sample, 1.05, 1.15)), length, device=sample.device, dtype=sample.dtype)
    parent_positions = sample[parent_index, :3, :length]
    child_positions = sample[joint_index, :3, :length]
    delta = (child_positions - parent_positions) * scale.unsqueeze(0) - (child_positions - parent_positions)
    for subtree_joint in subtree:
        sample[subtree_joint, :3, :length] += delta


def _apply_symmetry_break(sample: torch.Tensor, metadata: SkeletonMetadata, length: int) -> None:
    if not metadata.symmetric_joint_pairs:
        return
    pair = metadata.symmetric_joint_pairs[int(torch.randint(0, len(metadata.symmetric_joint_pairs), (1,), device=sample.device).item())]
    dominant_joint, passive_joint = pair
    scale = _rand_range(sample, 1.5, 2.0)
    sample[dominant_joint, :3, :length] *= scale
    sample[passive_joint, :3, :length] *= 1.0 / scale.clamp_min(1e-6)


def _apply_joint_jitter(sample: torch.Tensor, metadata: SkeletonMetadata, length: int) -> None:
    joint_index = _choose_joint(sample, metadata, include_root=True)
    freq = int(torch.randint(10, 21, (1,), device=sample.device).item())
    amplitude = _rand_range(sample, 0.01, 0.03)
    t = torch.linspace(0.0, 1.0, length, device=sample.device, dtype=sample.dtype)
    jitter = torch.sin(2.0 * torch.pi * float(freq) * t).unsqueeze(0)
    sample[joint_index, :3, :length] += amplitude * jitter


def _apply_acceleration_burst(sample: torch.Tensor, metadata: SkeletonMetadata, length: int) -> None:
    if length < 3:
        return
    joint_index = _choose_joint(sample, metadata, include_root=True)
    center = int(torch.randint(1, max(length - 1, 2), (1,), device=sample.device).item())
    radius = min(4, max(1, length // 8))
    start_index = max(0, center - radius)
    end_index = min(length, center + radius + 1)
    pulse_t = torch.linspace(-1.0, 1.0, end_index - start_index, device=sample.device, dtype=sample.dtype)
    gaussian = torch.exp(-pulse_t.pow(2) / 0.15)
    direction = F.normalize(torch.randn((3,), device=sample.device, dtype=sample.dtype), dim=0)
    magnitude = _rand_range(sample, 0.03, 0.08)
    sample[joint_index, :3, start_index:end_index] += direction[:, None] * magnitude * gaussian.unsqueeze(0)


def _apply_interpenetration(sample: torch.Tensor, metadata: SkeletonMetadata, length: int) -> None:
    candidates = [index for index in metadata.end_effector_joints if index < sample.shape[0]]
    if not candidates:
        return
    joint_index = candidates[int(torch.randint(0, len(candidates), (1,), device=sample.device).item())]
    pose_center = sample[: metadata.n_joints, :3, :length].mean(dim=0)
    current = sample[joint_index, :3, :length]
    sample[joint_index, :3, :length] = torch.lerp(current, pose_center, float(_rand_range(sample, 0.35, 0.6)))


def _warp_time_axis(values: torch.Tensor) -> torch.Tensor:
    length = values.shape[-1]
    if length <= 4:
        return values
    split = max(2, length // 2)
    first_half = torch.linspace(0.0, max(split - 1, 1), split * 2, device=values.device, dtype=values.dtype)
    second_half = torch.linspace(split, length - 1, max(2, length - split // 2), device=values.device, dtype=values.dtype)
    grid = torch.cat([first_half[:split], second_half[: length - split]], dim=0).clamp(0, length - 1)
    floor = grid.floor().long()
    ceil = grid.ceil().long().clamp_max(length - 1)
    alpha = (grid - floor.to(values.dtype)).view(1, 1, -1)
    flat = values.view(-1, 1, length)
    warped = torch.lerp(flat[..., floor], flat[..., ceil], alpha)
    return warped.view_as(values)


def _apply_time_warp(sample: torch.Tensor, length: int) -> None:
    sample[:, :, :length] = _warp_time_axis(sample[:, :, :length])


def generate_biomechanical_negative_batch(
    motion: torch.Tensor,
    n_joints: torch.Tensor,
    lengths: torch.Tensor,
    object_types: Sequence[str],
    metadata_lookup: Mapping[str, SkeletonMetadata],
    species_indices: torch.Tensor,
    action_indices: torch.Tensor,
    *,
    num_species: int,
) -> dict[str, object]:
    negative_motion = motion.clone()
    negative_species_indices = species_indices.clone()
    negative_action_indices = action_indices.clone()
    negative_kinds: list[str] = []

    for batch_index, object_type in enumerate(object_types):
        metadata = metadata_lookup[str(object_type)]
        length = max(1, int(lengths[batch_index].item()))
        sample = negative_motion[batch_index, : metadata.n_joints, :, :length]
        kind = NEGATIVE_KINDS[int(torch.randint(0, len(NEGATIVE_KINDS), (1,), device=motion.device).item())]
        negative_kinds.append(kind)

        if kind == "foot_slip":
            _apply_foot_slip(sample, metadata, length)
        elif kind == "bone_length_drift":
            _apply_bone_length_drift(sample, metadata, length)
        elif kind == "symmetry_break":
            _apply_symmetry_break(sample, metadata, length)
        elif kind == "joint_jitter":
            _apply_joint_jitter(sample, metadata, length)
        elif kind == "acceleration_burst":
            _apply_acceleration_burst(sample, metadata, length)
        elif kind == "interpenetration":
            _apply_interpenetration(sample, metadata, length)
        elif kind == "time_warp":
            _apply_time_warp(sample, length)
        elif kind == "cross_species_mismatch" and num_species > 1:
            offset = int(torch.randint(1, num_species, (1,), device=motion.device).item())
            negative_species_indices[batch_index] = (negative_species_indices[batch_index] + offset) % num_species

        if kind != "cross_species_mismatch":
            _refresh_derived_channels(sample, metadata, length)

    return {
        "motion": negative_motion,
        "species_indices": negative_species_indices,
        "action_indices": negative_action_indices,
        "negative_kinds": negative_kinds,
    }