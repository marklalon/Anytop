"""Tempo head: how many source frames the sampled window holds.

The model draws a fixed ``T``-frame window. The motion in it spans some number
``L`` of frames at the dataset fps -- the loader resampled ``L`` source frames
into the window -- and ``L`` is the output length: the window is exported at
one output frame per source frame. ``L`` is NOT a model input, so it can never
pull the motion away from the action label; the model draws whatever tempo the
label and species call for, and this head reads ``L`` back off the denoised
window and its conditions.

``L`` is a distribution over integer-frame bins (``tempo_bin_frames``), trained
with a two-hot target so a length between two bins is still exact in
expectation, and read at inference as the expected frame count.
"""
import numpy as np
import torch
import torch.nn as nn

from data_loaders.truebones.truebones_utils.param_utils import MAX_SOURCE_FRAMES_MULT

# Spacing of the bin centres, in source frames.
TEMPO_BIN_STEP_FRAMES = 5


def tempo_bin_frames(window_frames, min_length):
    """Bin centres in source frames: ``min_length`` up to the source budget
    ``MAX_SOURCE_FRAMES_MULT * window_frames`` every ``TEMPO_BIN_STEP_FRAMES``,
    the budget itself always the last bin (the loader never puts more source
    frames in a window, nor fewer than ``min_length``)."""
    low = int(min_length)
    high = int(MAX_SOURCE_FRAMES_MULT) * int(window_frames)
    if low < 2 or high <= low:
        raise ValueError(
            f"tempo bins need 2 <= min_length < source budget, got min_length={low}, "
            f"budget={high} (window {window_frames})"
        )
    bins = list(range(low, high + 1, TEMPO_BIN_STEP_FRAMES))
    if bins[-1] != high:
        bins.append(high)
    return np.asarray(bins, dtype=np.float32)


def two_hot_targets(frames, bin_frames):
    """``[B, K]`` soft target: each frame count split linearly between its two
    neighbouring bins, so ``targets @ bin_frames == frames`` (clamped into the
    bin range)."""
    bins = bin_frames.to(dtype=torch.float32)
    frames = frames.to(device=bins.device, dtype=torch.float32).reshape(-1)
    frames = frames.clamp(min=float(bins[0]), max=float(bins[-1]))
    count = bins.shape[0]
    lower = (torch.searchsorted(bins, frames, right=True) - 1).clamp(min=0, max=count - 2)
    lo = bins[lower]
    hi = bins[lower + 1]
    upper_weight = (frames - lo) / (hi - lo)
    targets = frames.new_zeros(frames.shape[0], count)
    targets.scatter_(1, lower.unsqueeze(1), (1.0 - upper_weight).unsqueeze(1))
    targets.scatter_add_(1, (lower + 1).unsqueeze(1), upper_weight.unsqueeze(1))
    return targets


def expected_frames(logits, bin_frames):
    """``[B]`` expected source frame count of ``[B, K]`` tempo logits."""
    probs = torch.softmax(logits.float(), dim=-1)
    return probs @ bin_frames.to(device=probs.device, dtype=torch.float32)


class TempoHead(nn.Module):
    """Pool the decoder's motion tokens over frames and valid joints, then
    classify the window's source frame count into ``bin_frames``."""

    def __init__(self, latent_dim, bin_frames):
        super().__init__()
        # Derived from the run's window and min_length (args.json), so it is
        # rebuilt rather than stored with the weights.
        self.register_buffer(
            'bin_frames', torch.as_tensor(np.asarray(bin_frames, dtype=np.float32)), persistent=False
        )
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, int(self.bin_frames.shape[0])),
        )

    def forward(self, hidden, joint_valid):
        """``hidden``: decoder output ``[frames + 1, B, J, D]`` whose frame 0 is
        the rest-pose token; ``joint_valid``: ``[B, J]``. Returns ``[B, K]``
        logits in fp32."""
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            motion = hidden[1:].float()
            weight = joint_valid.to(device=motion.device, dtype=motion.dtype)
            pooled = (motion * weight[None, :, :, None]).sum(dim=(0, 2))
            pooled = pooled / (weight.sum(dim=1, keepdim=True).clamp_min(1.0) * motion.shape[0])
            return self.net(pooled)
