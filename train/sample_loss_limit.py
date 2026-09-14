"""Bound how much one sample can pull on a batch gradient.

A handful of clips carry targets tens of standard deviations out (a spawn
rising from underground, a fall on a rig with almost no joint spread). Their
``l_simple`` at a high timestep runs 100-1000x the typical sample, and after the
global ``clip_grad_norm_(1.0)`` that one sample *is* the step. Clipping the total
norm bounds the step size, not whose direction it takes.

The limiter gives every sample the gradient of a Huber loss on its RMS error:
unchanged below a cap, and above it scaled by ``sqrt(cap / l)``, so the sample's
gradient magnitude stops growing at what a sample sitting on the cap would
produce. The sample still trains, in its own direction. The cap is ``ratio``
times a running geometric mean of ``l_simple`` at the same diffusion timestep --
``l_simple`` spans two orders of magnitude from t=1 to t=T, so a single cap
would either never engage at low t or clamp every high-t sample.

Everything stays on the device: no host sync is added to the training step.
"""

from __future__ import annotations

import math

import torch


class SampleLossLimiter:
    # Per-sample EMA rate of the reference once a timestep is warm. At batch 16
    # over 100 timesteps a bucket sees ~0.16 samples/step, i.e. a time constant
    # of ~300 steps -- slow next to the batch, fast next to the loss curve.
    EMA_RATE = 0.02
    # Samples a timestep must have seen before its cap engages. Until then the
    # reference is a plain running mean and nothing is limited.
    WARMUP_SAMPLES = 20

    def __init__(self, num_timesteps: int, ratio: float, device):
        if ratio <= 1.0:
            raise ValueError(f"sample loss limit ratio must be > 1, got {ratio}.")
        self.num_timesteps = int(num_timesteps)
        self.ratio = float(ratio)
        self._log_ratio = math.log(self.ratio)
        self.log_ref = torch.zeros(self.num_timesteps, device=device)
        self.count = torch.zeros(self.num_timesteps, device=device)

    @torch.no_grad()
    def weights(self, t: torch.Tensor, l_simple: torch.Tensor) -> torch.Tensor:
        """Detached per-sample gradient weights in ``(0, 1]``; updates the reference.

        Apply as ``loss + (w - 1) * l_simple`` so only ``l_simple``'s gradient is
        rescaled and every other term keeps its own.
        """
        log_l = torch.log(l_simple.detach().float().clamp_min(1e-12))
        cap = self.log_ref[t] + self._log_ratio
        warm = self.count[t] >= self.WARMUP_SAMPLES
        # w = sqrt(min(1, cap / l)), in log space.
        w = torch.where(warm, torch.exp(0.5 * (cap - log_l).clamp(max=0.0)), torch.ones_like(log_l))

        # Winsorized update: a capped sample enters the reference at the cap, so
        # the outliers being limited cannot drag their own cap upward.
        sample_log = torch.where(warm, torch.minimum(log_l, cap), log_l)
        n = torch.zeros_like(self.count).index_add_(0, t, torch.ones_like(sample_log))
        total = torch.zeros_like(self.log_ref).index_add_(0, t, sample_log)
        batch_mean = total / n.clamp_min(1.0)
        # Running mean while a bucket is cold, EMA once it is warm (whichever
        # moves faster), and no move at all for buckets absent from this batch.
        rate = torch.maximum(
            n / (self.count + n).clamp_min(1.0),
            1.0 - (1.0 - self.EMA_RATE) ** n,
        )
        self.log_ref += rate * (batch_mean - self.log_ref)
        self.count += n
        return w

    def state_dict(self) -> dict:
        return {
            'ratio': self.ratio,
            'log_ref': self.log_ref.detach().cpu(),
            'count': self.count.detach().cpu(),
        }

    def load_state_dict(self, state: dict) -> bool:
        """Restore the running reference; ``False`` (and a cold start) when the
        saved one was built for a different timestep count."""
        log_ref = torch.as_tensor(state['log_ref'])
        if log_ref.shape != self.log_ref.shape:
            return False
        self.log_ref.copy_(log_ref.to(self.log_ref.device))
        self.count.copy_(torch.as_tensor(state['count']).to(self.count.device))
        return True
