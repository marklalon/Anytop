"""Weighted-objective backward with diagnostic task gradients; no surgery."""
import torch


class DetailGradientMonitor:
    metric_keys = ('grad_main_norm', 'grad_identity_norm', 'grad_cosine',
                   'grad_conflict', 'grad_merged_main_dot', 'grad_merged_identity_dot')

    def __init__(self, net):
        self.parameters = tuple(p for p in net.parameters() if p.requires_grad)
        # Stable addresses are required across CUDA graph captures/replays.
        self.main = tuple(torch.zeros_like(p) for p in self.parameters)
        for p in self.parameters:
            if p.grad is None:
                p.grad = torch.zeros_like(p)

    def zero(self):
        for p in self.parameters:
            p.grad.zero_()

    def backward(self, main, identity=None):
        self.zero()
        main.backward()
        with torch.no_grad():
            for saved, p in zip(self.main, self.parameters):
                saved.copy_(p.grad)
        if identity is not None:
            self.zero()
            identity.backward()
        with torch.no_grad():
            # Reduce over the whole parameter vector, not independently per layer.
            aa = torch.stack([g.square().sum() for g in self.main]).sum()
            if identity is None:
                zero = aa.new_zeros(())
                return dict(zip(self.metric_keys, (aa.sqrt(), zero, zero, zero, aa, zero)))
            bb = torch.stack([p.grad.square().sum() for p in self.parameters]).sum()
            ab = torch.stack([(g * p.grad).sum() for g, p in zip(self.main, self.parameters)]).sum()
            eps = torch.finfo(aa.dtype).tiny
            for g, p in zip(self.main, self.parameters):
                p.grad.add_(g)
            cosine = ab / (aa.sqrt() * bb.sqrt()).clamp_min(eps)
            return dict(zip(self.metric_keys, (aa.sqrt(), bb.sqrt(), cosine,
                (ab < 0).to(aa.dtype), aa + ab, ab + bb)))
