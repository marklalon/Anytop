import torch


class FlowMatchingUniformSampler:
    def __init__(self, sigma_min=1e-4):
        self.sigma_min = float(sigma_min)

    def sample(self, batch_size, device):
        t = torch.rand(batch_size, device=device) * (1.0 - self.sigma_min) + self.sigma_min
        weights = torch.ones_like(t)
        return t, weights


def create_named_schedule_sampler(name, diffusion):
    if name != "uniform":
        raise NotImplementedError(f"unknown flow matching schedule sampler: {name}")
    sigma_min = getattr(diffusion, "sigma_min", 1e-4)
    return FlowMatchingUniformSampler(sigma_min=sigma_min)
