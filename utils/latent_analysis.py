"""Reusable, offline decoder sensitivity in standardized latent coordinates.

This measures the diagonal pullback metric for canonical position, rotation
and one-frame position differences in one global standardized coordinate system.
"""
import hashlib
import json
import os
import random
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from utils.latent_util import LatentNormalizer, ae_inputs, estimate_normalizer


ANALYSIS_VERSION = 3
FEATURE_CONTRACT = 'canonical_pos_rot_dpos_per_sample_mean_v1'
SENSITIVITY_LOSS_VERSION = 'global_decoder_diagonal_mse_v1'
WEIGHT_CONTRACT = 'half_uniform_clipped_relative_0.25_4_v1'


@contextmanager
def analysis_rng(seed, device):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    device = torch.device(device)
    with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def sensitivity_features(decoded, n_joints):
    """Equal total weight per sample and feature family; exclude padded joints.

    Values are in the AE's canonical units, not physical/FK/contact metrics.
    Velocity here is a one-internal-frame difference (not metres/second).
    """
    joints = decoded.shape[1]
    valid = torch.arange(joints, device=decoded.device)[None] < n_joints[:, None]
    features = []
    for value in (decoded[:, :, :3], decoded[:, :, 3:9], decoded[:, :, :3].diff(dim=-1)):
        mask = valid[:, :, None, None]
        scale = (n_joints * value.shape[2] * value.shape[3]).float().sqrt()[:, None, None, None]
        features.append((value.masked_fill(~mask, 0) / scale).flatten(1))
    return torch.cat(features, dim=1)


def diagonal_sensitivity(ae, norm, x, y, probes, latent=None):
    """E[(J^T r)^2] = diag(J^T J), with independent Rademacher output probes."""
    skeleton, time = ae_inputs(y)
    with torch.no_grad():
        z = (ae.encode(x[:, :, :9], skeleton, loop_metadata=time,
                       sample=norm.mode == 'sample') if latent is None else latent)
        u = norm.normalize(z)
    u = u.detach().requires_grad_(True)
    # SDPA's flash forward supports first-order gradients; no higher derivatives
    # or explicit Jacobian are needed. Decoder parameters remain frozen.
    with torch.enable_grad():
        features = sensitivity_features(ae.decode(norm.denormalize(u), skeleton,
                                                 loop_metadata=time), y['n_joints'])
        diagonal = torch.zeros_like(u)
        for index in range(probes):
            signs = torch.empty_like(features).bernoulli_(0.5).mul_(2).sub_(1)
            gradient, = torch.autograd.grad(features, u, grad_outputs=signs,
                                            retain_graph=index + 1 < probes)
            diagonal.add_(gradient.detach().square())
    result = diagonal.mean(1) / probes
    if not torch.isfinite(result).all():
        raise FloatingPointError('Nonfinite offline latent sensitivity')
    return result.cpu()


def sensitivity_loss_weights(sensitivity):
    sensitivity = torch.as_tensor(sensitivity).detach().float()
    mean = sensitivity.double().mean()
    if (not torch.isfinite(sensitivity).all() or (sensitivity < 0).any()
            or mean <= 0):
        raise ValueError('Decoder sensitivity must be finite, nonnegative and nonzero')
    relative = (sensitivity.double() / mean).float()
    # Keep a uniform component and bound metric anisotropy before mean reduction.
    weights = 0.5 + 0.5 * relative.clamp(0.25, 4.0)
    return weights / weights.mean()


def validate_sensitivity_loss(metadata, shape):
    """Reject an objective change on resume; weights travel inside the checkpoint."""
    if (not isinstance(metadata, dict) or metadata.get('version') != SENSITIVITY_LOSS_VERSION
            or metadata.get('coordinates') != 'global_standardized_z0'
            or metadata.get('weighting') != WEIGHT_CONTRACT):
        raise ValueError('Decoder sensitivity training objective changed; use a new save_dir')
    sensitivity = torch.as_tensor(metadata['sensitivity']).float()
    weights = torch.as_tensor(metadata['weights']).float()
    if tuple(weights.shape) != tuple(shape) or sensitivity.shape != weights.shape:
        raise ValueError('Decoder sensitivity and AE latent layout mismatch')
    expected = sensitivity_loss_weights(sensitivity)
    if not torch.allclose(weights, expected, atol=1e-6, rtol=1e-5):
        raise ValueError('Saved decoder sensitivity weights do not match the loss contract')
    return weights.detach()


def load_or_analyze(ae, binding, identity, batch_factory, batch_count, device,
                    mode='mean', seed=10, sample_count=64, probes=8):
    """Cache normalization and sensitivity together beside the bound AE.

    batch_factory must provide a fresh, synchronous, unaugmented reference
    stream. Collect moments over all reference batches and one uniform global
    reservoir for sensitivity. Cache hits never touch that stream.
    """
    if min(batch_count, sample_count, probes) < 1:
        raise ValueError('Offline batch, sample and probe counts must be positive')
    contract = dict(version=ANALYSIS_VERSION, features=FEATURE_CONTRACT,
                    normalization='global_v1', training_loss=SENSITIVITY_LOSS_VERSION,
                    weighting=WEIGHT_CONTRACT,
                    ae={k: v for k, v in binding.items() if k != 'path'},
                    reference=identity, batches=batch_count, mode=mode, seed=seed,
                    sampling='uniform_global_reservoir_v1', sample_count=sample_count, probes=probes)
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    directory = Path(binding['path']).parent / 'latent_analysis'
    path = directory / f'{fingerprint}.pt'
    if path.exists():
        result = torch.load(path, map_location='cpu', weights_only=False)
        if result.get('contract') != contract or result.get('fingerprint') != fingerprint:
            raise ValueError(f'Latent analysis cache contract mismatch: {path}')
        norm = LatentNormalizer.from_metadata(result['normalization']).to(device)
        tqdm.write(f'[latent analysis] cache hit: {path}')
        validate_sensitivity_loss(result['training_loss'], norm.mean.shape)
        return norm, result['training_loss'], dict(path=str(path.resolve()), fingerprint=fingerprint)

    from train.train_autoencoder import select_batch, training_batch_to_device
    examples = []
    seen = 0
    current_batch = None

    def reference_batches():
        nonlocal current_batch
        iterator = iter(batch_factory())
        for _ in range(batch_count):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(batch_factory())
                try:
                    batch = next(iterator)
                except StopIteration:
                    raise ValueError('Offline latent reference dataset is empty') from None
            current_batch = batch
            yield batch

    def retain_posteriors(mu, logvar):
        nonlocal seen
        # The source conditions/motion stay on CPU. Retain only the bounded
        # reservoir's latent posteriors, not every encoded reference batch.
        x, conditions = current_batch
        y = conditions['y']
        for i in range(len(x)):
            seen += 1
            position = len(examples) if len(examples) < sample_count else random.randrange(seen)
            if position < sample_count:
                sample, condition = select_batch(x, y, torch.tensor([i]))
                value = dict(batch=(sample, {'y': condition}), mu=mu[i:i + 1].detach().cpu().clone(),
                             logvar=logvar[i:i + 1].detach().cpu().clone()
                             if mode == 'sample' and logvar is not None else None)
                if position == len(examples):
                    examples.append(value)
                else:
                    examples[position] = value

    tqdm.write(f'[latent analysis] estimating global statistics ({batch_count} batches)')
    with analysis_rng(seed, device):
        norm = estimate_normalizer(ae, reference_batches(), batch_count, device, mode,
                                   on_encoded=retain_posteriors)
        current_batch = None
        total = None
        for row in tqdm(examples, desc='latent sensitivity: global'):
            x, y = training_batch_to_device(row['batch'], device)
            latent = row['mu'].to(device)
            if row['logvar'] is not None:
                latent = latent.float() + torch.exp(0.5 * row['logvar'].to(device).float()) * torch.randn_like(latent, dtype=torch.float32)
            value = diagonal_sensitivity(ae, norm, x, y, probes, latent=latent).sum(0)
            total = value if total is None else total + value
        global_sensitivity = total / len(examples)
    training_loss = dict(version=SENSITIVITY_LOSS_VERSION, coordinates='global_standardized_z0',
                         weighting=WEIGHT_CONTRACT, sensitivity=global_sensitivity,
                         weights=sensitivity_loss_weights(global_sensitivity))
    result = dict(contract=contract, fingerprint=fingerprint, normalization=norm.metadata(),
                  sensitivity=global_sensitivity, sample_count=len(examples), reference_count=seen,
                  training_loss=training_loss, applied_to_training_loss=True)
    summary = dict(fingerprint=fingerprint, contract=contract, sample_count=len(examples),
                   normalization_frame_count=norm.count, reference_count=seen,
                   applied_to_training_loss=True,
                   sensitivity=global_sensitivity.tolist(),
                   training_loss={key: value.tolist() if torch.is_tensor(value) else value
                                  for key, value in training_loss.items()})
    directory.mkdir(parents=True, exist_ok=True)
    # Unique temporary names allow concurrent runs without corrupting a cache.
    handle, temporary = tempfile.mkstemp(dir=directory, suffix='.tmp')
    os.close(handle)
    try:
        torch.save(result, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    handle, temporary = tempfile.mkstemp(dir=directory, suffix='.json.tmp')
    os.close(handle)
    try:
        Path(temporary).write_text(json.dumps(summary, indent=2), encoding='utf-8')
        os.replace(temporary, path.with_suffix('.json'))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    tqdm.write(f'[latent analysis] saved: {path}; global frames: {norm.count}')
    return norm, training_loss, dict(path=str(path.resolve()), fingerprint=fingerprint)
