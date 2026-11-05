"""Frozen AE binding and reversible, fixed per-slot/channel normalization."""
import hashlib
from pathlib import Path

import torch
from torch import nn
from data_loaders.truebones.truebones_utils.dataset_tags import dataset_tags

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
    def __init__(self, mean, std, mode='mean', count=0, subsets=None):
        super().__init__()
        mean, std = torch.as_tensor(mean).float(), torch.as_tensor(std).float()
        if mean.ndim != 2 or mean.shape != std.shape or mode not in ('mean', 'sample'):
            raise ValueError('Invalid per-slot/channel latent normalization')
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
            raise ValueError('Nonfinite/degenerate normalization')
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.mode, self.count = mode, count
        self.subset_names = tuple(sorted(subsets or {}))
        self.subset_counts = {key: int(subsets[key]['count']) for key in self.subset_names}
        self.register_buffer('subset_mean', torch.stack(
            [mean] + [torch.as_tensor(subsets[k]['mean']).float() for k in self.subset_names]))
        self.register_buffer('subset_std', torch.stack(
            [std] + [torch.as_tensor(subsets[k]['std']).float() for k in self.subset_names]))
        if (self.subset_mean.shape[1:] != mean.shape or
                not torch.isfinite(self.subset_mean).all() or
                not torch.isfinite(self.subset_std).all() or not (self.subset_std > 0).all()):
            raise ValueError('Invalid subset latent statistics')

    def prepare_conditions(self, y):
        """Resolve host metadata before compiled encoding; unknown subsets use the global bank."""
        if not self.subset_names or 'latent_subset_id' in y:
            return y
        labels = latent_subset_labels(y, len(y['n_joints']))
        lookup = {name: i + 1 for i, name in enumerate(self.subset_names)}
        ids = torch.tensor([lookup.get(name, 0) for name in labels],
                           device=self.mean.device, dtype=torch.long)
        return dict(y, latent_subset_id=ids)

    def _statistics(self, y):
        if not self.subset_names:
            return self.mean, self.std
        if y is None:
            raise ValueError('Subset normalization requires sample conditions')
        ids = self.prepare_conditions(y)['latent_subset_id']
        return self.subset_mean[ids][:, None], self.subset_std[ids][:, None]

    def normalize(self, z, y=None):
        mean, std = self._statistics(y)
        return (z.float() - mean) / std

    def denormalize(self, z, y=None):
        mean, std = self._statistics(y)
        return z.float() * std + mean

    def metadata(self):
        result = dict(version=2 if self.subset_names else 1, axes='B,T', mode=self.mode, count=self.count,
                      std_floor=1e-4, mean=self.mean.cpu(), std=self.std.cpu())
        if self.subset_names:
            result['subsets'] = {name: dict(mean=self.subset_mean[i + 1].cpu(),
                                           std=self.subset_std[i + 1].cpu(), count=self.subset_counts[name])
                                 for i, name in enumerate(self.subset_names)}
            result['unseen_subset'] = 'global'
        return result

    @classmethod
    def from_metadata(cls, metadata):
        if metadata.get('version') not in (1, 2) or metadata.get('axes') != 'B,T' or metadata.get('std_floor') != 1e-4:
            raise ValueError('Unsupported latent statistics contract')
        if metadata['version'] == 2 and (not metadata.get('subsets') or metadata.get('unseen_subset') != 'global'):
            raise ValueError('Invalid subset normalization contract')
        return cls(metadata['mean'], metadata['std'], metadata['mode'], metadata['count'],
                   metadata.get('subsets'))


def latent_subset_labels(y, batch):
    explicit = y.get('object_subset')
    if explicit is not None:
        labels = [explicit] * batch if isinstance(explicit, str) else list(explicit)
        if len(labels) != batch:
            raise ValueError('object_subset must contain one label per sample')
        return [str(label).lower() if label else '__global__' for label in labels]
    species = y.get('object_type')
    if species is None:
        return ['__global__'] * batch
    species = [species] * batch if isinstance(species, str) else species
    if len(species) != batch:
        raise ValueError('object_type must contain one species per sample')
    tags = dataset_tags()
    return [tags.object_subset_for(name) or '__global__' for name in species]


@torch.no_grad()
def estimate_normalizer(ae, batches, count, device, mode='mean', on_encoded=None):
    """Pool subset moments; optionally retain reference posteriors without re-encoding."""
    if count < 1:
        raise ValueError('At least one statistics batch is required')
    moments = {}

    def merge(key, values, variance):
        size = values.shape[0] * values.shape[1]
        mean = values.mean((0, 1))
        m2 = (values - mean).square().sum((0, 1))
        if variance is not None:
            m2 = m2 + variance.sum((0, 1))
        if key in moments:
            old, old_m2, tokens = moments[key]
            delta = mean - old
            m2 = old_m2 + m2 + delta.square() * (tokens * size / (tokens + size))
            mean = old + delta * (size / (tokens + size))
            size += tokens
        moments[key] = mean, m2, size
    for _ in range(count):
        x, conditions = next(batches)
        x = x.to(device)
        y = {k: v.to(device) if torch.is_tensor(v) else v for k, v in conditions['y'].items()}
        skeleton, time = ae_inputs(y, x.shape[-1])
        mu, logvar = ae.encode_posterior(x[:, :, :9], skeleton, loop_metadata=time)
        if on_encoded is not None:
            on_encoded(mu, logvar)
        values = mu.double()
        # Parallel Welford merge; sampled VAE statistics include E[posterior variance].
        variance = logvar.double().exp() if mode == 'sample' and logvar is not None else None
        merge('__global__', values, variance)
        labels = latent_subset_labels(y, len(x))
        for label in sorted(set(labels) - {'__global__'}):
            indices = [i for i, item in enumerate(labels) if item == label]
            merge(label, values[indices], variance[indices] if variance is not None else None)
    stats = {key: dict(mean=mean.float(), std=(m2 / size).sqrt().float().clamp_min(1e-4), count=size)
             for key, (mean, m2, size) in moments.items()}
    global_stats = stats.pop('__global__')
    return LatentNormalizer(global_stats['mean'], global_stats['std'], mode, global_stats['count'], stats).to(device)


@torch.no_grad()
def encode_target(ae, normalizer, x, y):
    skeleton, time = ae_inputs(y, x.shape[-1])
    return normalizer.normalize(ae.encode(x[:, :, :9], skeleton, loop_metadata=time,
                                          sample=normalizer.mode == 'sample'), y)


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
