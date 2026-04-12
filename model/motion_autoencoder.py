from __future__ import annotations

import torch
import torch.nn as nn


def build_motion_valid_mask(
    n_joints: torch.Tensor,
    lengths: torch.Tensor,
    max_joints: int,
    max_frames: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    if device is None:
        device = n_joints.device
    joint_mask = torch.arange(max_joints, device=device).unsqueeze(0) < n_joints.unsqueeze(1)
    time_mask = torch.arange(max_frames, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    return joint_mask[:, :, None, None] & time_mask[:, None, None, :]


def masked_mse_per_sample(
    target: torch.Tensor,
    prediction: torch.Tensor,
    n_joints: torch.Tensor,
    lengths: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    mask = build_motion_valid_mask(
        n_joints,
        lengths,
        max_joints=target.shape[1],
        max_frames=target.shape[-1],
        device=target.device,
    ).to(dtype=target.dtype)
    squared_error = (target - prediction).pow(2) * mask
    denom = (mask.sum(dim=(1, 2, 3)) * target.shape[2]).clamp_min(eps)
    return squared_error.sum(dim=(1, 2, 3)) / denom


def masked_l1_per_sample(
    target: torch.Tensor,
    prediction: torch.Tensor,
    n_joints: torch.Tensor,
    lengths: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    mask = build_motion_valid_mask(
        n_joints,
        lengths,
        max_joints=target.shape[1],
        max_frames=target.shape[-1],
        device=target.device,
    ).to(dtype=target.dtype)
    abs_error = (target - prediction).abs() * mask
    denom = (mask.sum(dim=(1, 2, 3)) * target.shape[2]).clamp_min(eps)
    return abs_error.sum(dim=(1, 2, 3)) / denom


def masked_temporal_difference_l1_per_sample(
    target: torch.Tensor,
    prediction: torch.Tensor,
    n_joints: torch.Tensor,
    lengths: torch.Tensor,
    *,
    order: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    if order < 1:
        raise ValueError("order must be >= 1")

    target_delta = target
    prediction_delta = prediction
    for _ in range(order):
        if target_delta.shape[-1] <= 1:
            return torch.zeros(target.shape[0], device=target.device, dtype=target.dtype)
        target_delta = target_delta[..., 1:] - target_delta[..., :-1]
        prediction_delta = prediction_delta[..., 1:] - prediction_delta[..., :-1]

    delta_lengths = torch.clamp(lengths - order, min=0)
    return masked_l1_per_sample(target_delta, prediction_delta, n_joints, delta_lengths, eps=eps)


def masked_temporal_blur_l1_per_sample(
    target: torch.Tensor,
    prediction: torch.Tensor,
    n_joints: torch.Tensor,
    lengths: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    if target.shape[-1] < 3:
        return torch.zeros(target.shape[0], device=target.device, dtype=target.dtype)

    target_blur = 0.25 * target[..., :-2] + 0.5 * target[..., 1:-1] + 0.25 * target[..., 2:]
    prediction_blur = 0.25 * prediction[..., :-2] + 0.5 * prediction[..., 1:-1] + 0.25 * prediction[..., 2:]
    blur_lengths = torch.clamp(lengths - 2, min=0)
    return masked_l1_per_sample(target_blur, prediction_blur, n_joints, blur_lengths, eps=eps)


def motion_perceptual_recon_error_per_sample(
    target: torch.Tensor,
    prediction: torch.Tensor,
    n_joints: torch.Tensor,
    lengths: torch.Tensor,
    *,
    position_weight: float = 0.35,
    velocity_weight: float = 0.30,
    acceleration_weight: float = 0.20,
    blur_weight: float = 0.15,
    eps: float = 1e-8,
) -> torch.Tensor:
    total_weight = float(position_weight + velocity_weight + acceleration_weight + blur_weight)
    if total_weight <= 0.0:
        raise ValueError("At least one perceptual reconstruction weight must be positive.")

    position_error = masked_l1_per_sample(target, prediction, n_joints, lengths, eps=eps)
    velocity_error = masked_temporal_difference_l1_per_sample(
        target,
        prediction,
        n_joints,
        lengths,
        order=1,
        eps=eps,
    )
    acceleration_error = masked_temporal_difference_l1_per_sample(
        target,
        prediction,
        n_joints,
        lengths,
        order=2,
        eps=eps,
    )
    blur_error = masked_temporal_blur_l1_per_sample(target, prediction, n_joints, lengths, eps=eps)

    weighted_error = (
        position_weight * position_error
        + velocity_weight * velocity_error
        + acceleration_weight * acceleration_error
        + blur_weight * blur_error
    )
    return weighted_error / total_weight


def _group_norm_groups(channels: int, max_groups: int = 8) -> int:
    for group_count in range(min(max_groups, channels), 0, -1):
        if channels % group_count == 0:
            return group_count
    return 1


class MotionEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int = 13,
        d_model: int = 128,
        latent_dim: int = 64,
        num_conv_layers: int = 3,
        kernel_size: int = 5,
        max_joints: int = 143,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.max_joints = max_joints

        self.input_projection = nn.Linear(feature_dim, d_model)
        self.joint_position_embedding = nn.Embedding(max_joints, d_model)
        self.conv_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2),
                    nn.GroupNorm(_group_norm_groups(d_model), d_model),
                    nn.GELU(),
                )
                for _ in range(num_conv_layers)
            ]
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, latent_dim),
        )

    def forward(
        self,
        motion: torch.Tensor,
        n_joints: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, joint_count, feature_dim, frame_count = motion.shape
        if feature_dim != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {feature_dim}")
        if joint_count > self.max_joints:
            raise ValueError(f"Expected at most {self.max_joints} joints, got {joint_count}")

        x = motion.permute(0, 1, 3, 2)
        x = self.input_projection(x)
        x = x.permute(0, 1, 3, 2)

        joint_indices = torch.arange(joint_count, device=motion.device)
        joint_embeddings = self.joint_position_embedding(joint_indices)
        x = x + joint_embeddings.unsqueeze(0).unsqueeze(-1)

        x = x.reshape(batch_size * joint_count, self.d_model, frame_count)
        for conv_block in self.conv_blocks:
            x = x + conv_block(x)
        x = x.view(batch_size, joint_count, self.d_model, frame_count)

        time_mask = (torch.arange(frame_count, device=motion.device).unsqueeze(0) < lengths.unsqueeze(1)).to(x.dtype)
        x = (x * time_mask[:, None, None, :]).sum(dim=-1) / lengths.clamp_min(1).to(x.dtype)[:, None, None]

        joint_mask = (torch.arange(joint_count, device=motion.device).unsqueeze(0) < n_joints.unsqueeze(1)).to(x.dtype)
        x = (x * joint_mask[:, :, None]).sum(dim=1) / n_joints.clamp_min(1).to(x.dtype)[:, None]
        return self.output_projection(x)


class MotionDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 64,
        d_model: int = 128,
        num_conv_layers: int = 0,
        kernel_size: int = 5,
        feature_dim: int = 13,
        max_joints: int = 143,
        max_frames: int = 120,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.num_conv_layers = num_conv_layers
        self.kernel_size = kernel_size
        self.feature_dim = feature_dim
        self.max_joints = max_joints
        self.max_frames = max_frames

        self.latent_projection = nn.Linear(latent_dim, d_model)
        self.latent_to_scale_shift = nn.Linear(latent_dim, d_model * 2)
        self.joint_position_embedding = nn.Embedding(max_joints, d_model)
        self.frame_position_embedding = nn.Embedding(max_frames, d_model)
        self.position_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.conv_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2),
                    nn.GroupNorm(_group_norm_groups(d_model), d_model),
                    nn.GELU(),
                )
                for _ in range(num_conv_layers)
            ]
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, feature_dim),
        )

    def forward(
        self,
        z: torch.Tensor,
        n_joints: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        max_frames: int | None = None,
    ) -> torch.Tensor:
        del n_joints, lengths
        if max_frames is None:
            raise ValueError("max_frames must be provided to MotionDecoder.forward")
        if max_frames > self.max_frames:
            raise ValueError(f"Expected at most {self.max_frames} frames, got {max_frames}")

        batch_size = z.shape[0]
        base = self.latent_projection(z)
        scale, shift = self.latent_to_scale_shift(z).chunk(2, dim=-1)
        joint_embeddings = self.joint_position_embedding(torch.arange(self.max_joints, device=z.device))
        frame_embeddings = self.frame_position_embedding(torch.arange(max_frames, device=z.device))

        hidden = joint_embeddings[None, :, None, :] + frame_embeddings[None, None, :, :]
        hidden = self.position_mlp(hidden)
        hidden = hidden * (1.0 + scale[:, None, None, :]) + shift[:, None, None, :] + base[:, None, None, :]
        hidden = hidden.permute(0, 1, 3, 2).reshape(batch_size * self.max_joints, self.d_model, max_frames)
        for conv_block in self.conv_blocks:
            hidden = hidden + conv_block(hidden)
        hidden = hidden.view(batch_size, self.max_joints, self.d_model, max_frames).permute(0, 1, 3, 2).contiguous()
        output = self.output_projection(hidden)
        return output.permute(0, 1, 3, 2).contiguous()


class MotionAutoencoder(nn.Module):
    def __init__(
        self,
        feature_dim: int = 13,
        d_model: int = 128,
        latent_dim: int = 64,
        num_conv_layers: int = 3,
        decoder_num_conv_layers: int = 0,
        kernel_size: int = 5,
        max_joints: int = 143,
        max_frames: int = 120,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.num_conv_layers = num_conv_layers
        self.decoder_num_conv_layers = decoder_num_conv_layers
        self.kernel_size = kernel_size
        self.max_joints = max_joints
        self.max_frames = max_frames

        self.encoder = MotionEncoder(
            feature_dim=feature_dim,
            d_model=d_model,
            latent_dim=latent_dim,
            num_conv_layers=num_conv_layers,
            kernel_size=kernel_size,
            max_joints=max_joints,
        )
        self.decoder = MotionDecoder(
            latent_dim=latent_dim,
            d_model=d_model,
            num_conv_layers=decoder_num_conv_layers,
            kernel_size=kernel_size,
            feature_dim=feature_dim,
            max_joints=max_joints,
            max_frames=max_frames,
        )

    def forward(
        self,
        motion: torch.Tensor,
        n_joints: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if motion.shape[-1] > self.max_frames:
            raise ValueError(f"Expected at most {self.max_frames} frames, got {motion.shape[-1]}")
        z = self.encoder(motion, n_joints, lengths)
        reconstruction = self.decoder(z, n_joints, lengths, motion.shape[-1])
        reconstruction = reconstruction[:, : motion.shape[1], :, :]
        return reconstruction, z

    def encode(
        self,
        motion: torch.Tensor,
        n_joints: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(motion, n_joints, lengths)