"""Learning-rate schedules shared by diffusion and autoencoder training."""
import math


def cosine_decay_lr_lambda(lr, lr_final, decay_start, num_steps):
    """Keep LR constant until decay_start, then decay to lr_final at num_steps."""
    if decay_start is None:
        return lambda step: 1.0
    if not 0 <= decay_start < num_steps:
        raise ValueError(f"--lr_decay_start {decay_start} must lie in [0, num_steps={num_steps})")
    if not 0 <= lr_final <= lr:
        raise ValueError(f"--lr_final {lr_final} must lie in [0, lr={lr}]")
    floor = lr_final / lr
    span = num_steps - decay_start

    def multiplier(step):
        if step < decay_start:
            return 1.0
        t = min((step - decay_start) / span, 1.0)
        return floor + 0.5 * (1.0 - floor) * (1.0 + math.cos(math.pi * t))
    return multiplier
