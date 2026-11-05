"""Frozen AE binding and reversible, fixed per-slot/channel normalization."""
import hashlib
from pathlib import Path

import torch
from torch import nn

from utils.model_config import without_removed_skeleton_drops
from model.motion_autoencoder import MotionAutoencoder, DECODER_CONTENT_VERSION
from data_loaders.truebones.truebones_utils.canonical_features import CANONICAL_FEATURE_SPACE
from diffusion.latent_diffusion import validate_latent_conditions

LATENT_CHECKPOINT_VERSION = 'latent_diffusion_v2'
TIME_CONTRACT = 'fixed60_v1'
SKELETON_KEYS = ('n_joints', 'joints_names_embs', 'rest_geom', 'joint_struct',
                 'graph_dist', 'joints_relations')
TIME_KEYS = ('is_loop', 'loop_tile_count', 'resample_speed_cond')


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_autoencoder(path, device='cpu'):
    path = Path(path)
    if path.is_dir():
        path = path / 'autoencoder.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if (checkpoint.get('kind') != 'motion_autoencoder_7a'
            or checkpoint.get('canonical_version') != CANONICAL_FEATURE_SPACE
            or checkpoint.get('decoder_content_version') != DECODER_CONTENT_VERSION
            or checkpoint['args']['num_frames'] != 60):
        raise ValueError('AE checkpoint type, canonical/content version or 60-frame contract mismatch')
    model = MotionAutoencoder(**without_removed_skeleton_drops(checkpoint['model_config']))
    ema = checkpoint['args'].get('use_ema', False)
    model.load_state_dict(checkpoint['model_ema'] if ema else checkpoint['model'], strict=True)
    model.to(device).eval().requires_grad_(False)
    binding = dict(sha256=file_sha256(path), weights='ema' if ema else 'model',
                   model_config=checkpoint['model_config'], canonical_version=CANONICAL_FEATURE_SPACE,
                   decoder_content_version=DECODER_CONTENT_VERSION, frames=60,
                   layout='B,T,K,d', path=str(path.resolve()))
    return model, binding


def ae_inputs(y, frames=60):
    # Select the shared (possibly masked) skeleton and time conditions used to
    # encode the target latent.
    validate_latent_conditions(y, frames)
    return {key: y[key] for key in SKELETON_KEYS}, {key: y[key] for key in TIME_KEYS}


class LatentNormalizer(nn.Module):
    def __init__(self, mean, std, mode='mean', count=0):
        super().__init__()
        mean, std = torch.as_tensor(mean).float(), torch.as_tensor(std).float()
        if mean.ndim != 2 or mean.shape != std.shape or mode not in ('mean', 'sample'):
            raise ValueError('Invalid per-slot/channel latent normalization')
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
            raise ValueError('Nonfinite/degenerate normalization')
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.mode, self.count = mode, count

    def normalize(self, z):
        return (z.float() - self.mean) / self.std

    def denormalize(self, z):
        return z.float() * self.std + self.mean

    def metadata(self):
        return dict(version=1, axes='B,T', mode=self.mode, count=self.count,
                    std_floor=1e-4, mean=self.mean.cpu(), std=self.std.cpu())

    @classmethod
    def from_metadata(cls, metadata):
        if metadata.get('version') != 1 or metadata.get('axes') != 'B,T' or metadata.get('std_floor') != 1e-4:
            raise ValueError('Unsupported latent statistics contract')
        return cls(metadata['mean'], metadata['std'], metadata['mode'], metadata['count'])


@torch.no_grad()
def estimate_normalizer(ae, batches, count, device, mode='mean'):
    if count < 1:
        raise ValueError('At least one statistics batch is required')
    total = second = None
    tokens = 0
    for _ in range(count):
        x, conditions = next(batches)
        x = x.to(device)
        y = {k: v.to(device) if torch.is_tensor(v) else v for k, v in conditions['y'].items()}
        skeleton, time = ae_inputs(y, x.shape[-1])
        mu, logvar = ae.encode_posterior(x[:, :, :9], skeleton, loop_metadata=time)
        values = mu.double()
        # Parallel Welford merge; sampled VAE statistics include E[posterior variance].
        size = values.shape[0] * values.shape[1]
        mean = values.mean((0, 1))
        m2 = (values - mean).square().sum((0, 1))
        if mode == 'sample' and logvar is not None:
            m2 = m2 + logvar.double().exp().sum((0, 1))
        if total is None:
            total, second = mean, m2
        else:
            delta = mean - total
            second = second + m2 + delta.square() * (tokens * size / (tokens + size))
            total = total + delta * (size / (tokens + size))
        tokens += size
    return LatentNormalizer(total.float(), (second / tokens).sqrt().float().clamp_min(1e-4), mode, tokens).to(device)


@torch.no_grad()
def encode_target(ae, normalizer, x, y):
    skeleton, time = ae_inputs(y, x.shape[-1])
    return normalizer.normalize(ae.encode(x[:, :, :9], skeleton, loop_metadata=time,
                                          sample=normalizer.mode == 'sample'))


def load_latent_checkpoint(path, device='cpu', ae_checkpoint=None):
    from model.latent_motion_denoiser import LatentMotionDenoiser
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if (payload.get('kind') != LATENT_CHECKPOINT_VERSION
            or payload.get('time_contract') != TIME_CONTRACT
            or payload.get('prediction') != 'standardized_z0'):
        raise ValueError('Unsupported latent checkpoint/condition contract')
    ae, binding = load_frozen_autoencoder(ae_checkpoint or payload['ae_binding']['path'], device)
    if {k: v for k, v in binding.items() if k != 'path'} != {
            k: v for k, v in payload['ae_binding'].items() if k != 'path'}:
        raise ValueError('Frozen AE hash/configuration does not match latent checkpoint')
    config = without_removed_skeleton_drops(payload['model_config'])
    # Older checkpoints may record tempo dropout; tempo is now always kept.
    config.pop('tempo_cfg_drop_prob', None)
    model = LatentMotionDenoiser(**config)
    model.load_state_dict(payload['model_ema'] if payload['model_ema'] is not None else payload['model'])
    if model.action_label_cond:
        model.validate_loaded_action_conditioning(payload['action_conditioning'], str(path))
    norm = LatentNormalizer.from_metadata(payload['normalization']).to(device)
    if tuple(norm.mean.shape) != (ae.slots, ae.latent_dim):
        raise ValueError('Normalizer and AE latent layout mismatch')
    if (model.slots, model.latent_channels) != (ae.slots, ae.latent_dim):
        raise ValueError('Denoiser and AE latent layout mismatch')
    return model.to(device).eval(), ae, norm, payload
