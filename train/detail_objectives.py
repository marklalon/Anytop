"""Conditional, phase-independent statistics of FINAL output; no GT reconstruction."""
import torch
from model.motion_detail_refiner import fk, sample_context, spectral_project, ric_body_positions
from utils.rotation_conversions import rotation_6d_to_matrix_safe


def rms(x, dim):
    # Smooth near zero without a singular sqrt backward.
    return (x.square().mean(dim) + 1e-10).sqrt()


def frequency_bands(config):
    """Analysis bins only; none is designated as body/detail or blocked."""
    return ((0., config.cutoff_hz), (config.cutoff_hz, 1.5 * config.cutoff_hz),
            (1.5 * config.cutoff_hz, float('inf')))


def coordination_pairs(parents):
    # Parent-child and sibling pairs include cross-wing/limb coordination,
    # without an all-joint quadratic feature count on feather-heavy rigs.
    pairs = {(min(j, p), max(j, p)) for j, p in enumerate(parents) if p >= 0}
    for j in range(1, len(parents)):
        for k in range(j + 1, len(parents)):
            if parents[j] >= 0 and parents[j] == parents[k]:
                pairs.add((j, k))
    pairs = {pair for pair in pairs if all(parents[j] >= 0 for j in pair)}
    return tuple(zip(*sorted(pairs))) if pairs else ((), ())


def amplitude_units(product):
    # Covariance and cross-band products have squared-amplitude units. A
    # smooth signed square-root puts them on the RMS scale, finite at zero.
    return product / (product.abs() + 1e-6).sqrt()


def calibration_features(local, global_rotation, positions, scale, project, quantiles, config, pairs,
                         active=None):
    """Shared [B,T,J,*] statistics for the serial and batched paths.

    Local articulation plus body-frame rotations/positions capture inherited
    wing motion. RIC non-root XZ already excludes root translation; only root
    height is subtracted before rotating into the root's frame.
    """
    root_frame = global_rotation[:, :, :1].transpose(-1, -2)
    body_rotation = root_frame @ global_rotation
    offset = torch.zeros_like(positions[:, :, :1])
    offset[..., 1] = positions[:, :, :1, 1]
    relative_position = positions - offset
    # Root XZ is a world path, unlike every other joint's RIC XZ. It must
    # never leak into parent-child covariance as if it were body motion.
    relative_position[:, :, 0] = 0
    body_position = (root_frame @ relative_position[..., None]).squeeze(-1) / scale
    signal = torch.cat((local.flatten(-2), body_rotation.flatten(-2), body_position), -1)
    mean = signal.mean(1, keepdim=True)
    # Mean posture is a calibratable DC component, not an unobserved nullspace.
    active = slice(1, None) if active is None else active
    row = [mean[:, 0, active].flatten(1)]
    centered = signal - mean
    envelopes = []
    left, right = pairs
    for lo, hi in frequency_bands(config):
        band = project(centered, lo, hi)
        envelope = rms(band, -1)
        envelopes.append(envelope)
        covariance = (band[:, :, left] * band[:, :, right]).mean((1, 3))
        row.extend((rms(band, 1)[:, active].flatten(1),
                    quantiles(envelope)[:, :, active].flatten(1),
                    amplitude_units(covariance)))
    # Envelope relationships across every pair of bands, without a privileged
    # low-frequency driver or frame-by-frame alignment to a random GT clip.
    for i in range(len(envelopes)):
        for j in range(i + 1, len(envelopes)):
            row.append(amplitude_units((envelopes[i] * envelopes[j]).mean(1))[:, active])
    return torch.cat(row, -1)


def detail_features(motion, y, config):
    features = []
    for b in range(len(motion)):
        n, t, loop, dt, parents = sample_context(y, b, config)
        x = motion[b, :n, :, :t].permute(2, 0, 1)
        r = rotation_6d_to_matrix_safe(x[..., 3:9])
        rest = y['rest_pos_ric_hml'][b, :n].to(x)
        g, virtual = fk(r, rest, parents)
        positions = virtual - virtual[:, :1]
        scale = (rest - rest[:1]).square().sum(-1).mean().sqrt().clamp_min(1e-3)
        def project(v, lo, hi):
            return spectral_project(v[0], dt, loop, lo, hi)[None]
        def quantiles(v):
            return torch.quantile(v, v.new_tensor([.25, .5, .9]), dim=1).permute(1, 0, 2)
        features.append(calibration_features(r[None], g[None], positions[None],
                        scale, project, quantiles, config,
                        coordination_pairs(parents),
                        [j for j, p in enumerate(parents) if p >= 0]
                        )[0])
    return torch.stack(features)


def feature_scale(target):
    # A fixed amplitude tolerance prevents nearly-still GT channels from
    # dominating all other joints through an arbitrarily tiny denominator.
    return rms(target, 0).clamp(0.02, 1.0).detach()


def empirical_target(features, clips):
    """One observation per independent clip, not per overlapping window."""
    ids = list(clips) if clips is not None else [None] * len(features)
    if len(ids) != len(features):
        raise ValueError('Clip identifiers must match target observations')
    unique = list(dict.fromkeys(ids))
    return torch.stack([features[[i for i, clip in enumerate(ids) if clip == key]].mean(0)
                        for key in unique])


def statistics_loss(pred, target, scale):
    a, b = pred / scale, target.detach() / scale
    def kernel(x, z):
        distance = (x[:, None] - z[None]).square().mean(-1)
        return sum(torch.exp(-distance / bandwidth) for bandwidth in (.25, 1., 4.)) / 3
    mmd = kernel(a, a).mean() + kernel(b, b).mean() - 2 * kernel(a, b).mean()
    difference = a.mean(0) - b.mean(0)
    # Smooth robust moment matching: quadratic near agreement, bounded slope
    # on large mismatches. Finite cache outliers cannot set the whole gradient.
    moment = (torch.sqrt(1 + difference.square()) - 1).mean()
    # One clip cannot identify a distribution or justify collapsing seed variance.
    return moment if len(target) == 1 else mmd.clamp_min(0) + moment


def geometry_objective(bone_mse):
    """Monitor bone-length change; endpoint displacement is not a seam target."""
    return bone_mse.mean()


@torch.no_grad()
def preservation_reference(base, y, config):
    """Fixed-base work shared by all updates; never retain an autograd graph."""
    rows = []
    for b in range(len(base)):
        n, t, loop, dt, parents = sample_context(y, b, config)
        ref = base[b, :n, :, :t].detach().permute(2, 0, 1)
        rest = y['rest_pos_ric_hml'][b, :n].to(ref)
        rb = rotation_6d_to_matrix_safe(ref[..., 3:9])
        gb, _ = fk(rb, rest, parents)
        children = [j for j, p in enumerate(parents) if p >= 0]
        ancestors = [parents[j] for j in children]
        positions = ric_body_positions(ref[..., :3])
        rows.append({'rotation': rb, 'global_rotation': gb,
                     'bones': torch.linalg.vector_norm(positions[:, children] - positions[:, ancestors], dim=-1)})
    return rows


def preservation_losses(out, base, delta, y, config, reference=None, matched=False):
    if reference is None:
        reference = preservation_reference(base, y, config)
    keep, geo = [], []
    for b in range(len(out)):
        n, t, loop, dt, parents = sample_context(y, b, config)
        x = out[b, :n, :, :t].permute(2, 0, 1)
        ref = base[b, :n, :, :t].detach().permute(2, 0, 1)
        rest = y['rest_pos_ric_hml'][b, :n].to(x)
        scale = (rest - rest[:1]).square().sum(-1).mean().sqrt().clamp_min(1e-3)
        r = rotation_6d_to_matrix_safe(x[..., 3:9])
        rb, gb = reference[b]['rotation'], reference[b]['global_rotation']
        g, _ = fk(r, rest, parents)
        # Full-band, uniform minimum-change cost. Low-frequency repairs are
        # allowed and pay exactly the same amplitude cost as high-frequency ones.
        kp = ((x[..., :3] - ref[..., :3]) / scale).square().mean()
        kr = (g - gb).square().mean() + (r - rb).square().mean()
        identity = x.new_zeros(())
        if matched:
            identity = identity_guard(kp, (r - rb).square().mean(), config)
        keep.append(kp + kr + identity)
        children = [j for j, p in enumerate(parents) if p >= 0]
        ancestors = [parents[j] for j in children]
        positions = ric_body_positions(x[..., :3])
        bones = torch.linalg.vector_norm(positions[:, children] - positions[:, ancestors], dim=-1)
        bone_loss = ((bones - reference[b]['bones']) / scale).square().mean()
        geo.append(geometry_objective(bone_loss))
    return {'keep': torch.stack(keep).mean(), 'geo': torch.stack(geo).mean(),
            'reg': delta.square().sum() / sum(int(y['lengths'][b]) * max(1, int(y['n_joints'][b]) - 1) * 3 for b in range(len(out)))}


def identity_guard(position_mse, rotation_mse, config):
    def excess(mse, tolerance):
        return ((mse + 1e-12).sqrt() / tolerance - 1).clamp_min(0).square()
    return (excess(position_mse, config.identity_position_tolerance)
            + excess(rotation_mse, config.identity_rotation_tolerance)).mean()
