"""Training-time Gaussian noise on joint-name embeddings, inside the names' own subspace.

The noise lives in the top principal directions of the distinct training name
vectors, scaled by each direction's spread, so it makes nearby names (two
segments of one limb) interchangeable while distant body parts stay apart. The
name keeps coarse semantics; exact joint identity is left to the structural
channel and the rest geometry.

An all-zero row means "name unknown" (padding, a whole-joint drop, or a name
``process_new_skeleton`` blanked) and stays exactly zero.
"""
import numpy as np
import torch
from torch import nn


def joint_name_noise_basis(cond_dict, rank):
    """``(basis [rank, D], std [rank])`` from the cond entries' distinct non-zero names.

    Rows of ``basis`` are the leading principal directions of the centered name
    vectors; ``std`` is the standard deviation along each. Deterministic for a
    given cond, so a resume rebuilds the same basis.
    """
    if rank < 1:
        raise ValueError(f"name noise rank must be >= 1, got {rank}")
    names = np.concatenate([
        np.asarray(entry['joints_names_embs'], dtype=np.float64) for entry in cond_dict.values()
    ])
    names = np.unique(names[np.any(names != 0, axis=1)], axis=0)
    if len(names) <= rank:
        raise ValueError(
            f"name noise rank {rank} needs more than {rank} distinct names, got {len(names)}"
        )
    centered = names - names.mean(axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    std = singular[:rank] / np.sqrt(len(names) - 1)
    return (torch.from_numpy(vt[:rank].astype(np.float32)),
            torch.from_numpy(std.astype(np.float32)))


def add_joint_name_noise(names, noise_std, basis, std):
    """``names`` plus ``noise_std * (z * std) @ basis`` on every non-zero row.

    ``names`` is ``[..., D]``; ``z`` is standard normal per (row, direction).
    Branch-free on tensor values, so it does not break torch.compile.
    """
    z = torch.randn(*names.shape[:-1], basis.shape[0], device=names.device, dtype=basis.dtype)
    noise = (z * (noise_std * std)) @ basis
    named = names.ne(0).any(dim=-1, keepdim=True)
    return names + noise.to(names.dtype) * named.to(names.dtype)


class JointNameNoise(nn.Module):
    """Applies ``add_joint_name_noise`` in training; identity in eval.

    The basis rides along as non-persistent buffers: training attaches it with
    ``set_basis`` after the model is built, the state_dict keys do not change,
    and inference never needs it.
    """

    def __init__(self, noise_std):
        super().__init__()
        if not noise_std >= 0.0:
            raise ValueError(f"name noise std must be >= 0, got {noise_std}")
        self.noise_std = float(noise_std)
        self.register_buffer('basis', None, persistent=False)
        self.register_buffer('std', None, persistent=False)

    def set_basis(self, basis, std, device=None):
        self.basis = basis.to(device=device, dtype=torch.float32)
        self.std = std.to(device=device, dtype=torch.float32)

    def forward(self, names):
        if not self.training or self.noise_std <= 0.0:
            return names
        if self.basis is None:
            raise ValueError(
                "name noise is on but no basis is attached; training calls "
                "AnyTop.set_joint_name_noise_basis(joint_name_noise_basis(cond_dict, rank))."
            )
        return add_joint_name_noise(names, self.noise_std, self.basis, self.std)
