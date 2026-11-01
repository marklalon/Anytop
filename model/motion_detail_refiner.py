"""Frozen-backbone all-band motion calibration in physical local SO(3).

Public motion layout is [B,J,F,T]; all filtering is on unpadded [T,J,C].
Positions follow transported current bone vectors, preserving input bone lengths.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from utils.rotation_conversions import rotation_6d_to_matrix_safe
from data_loaders.truebones.truebones_utils.joint_struct_features import JOINT_STRUCT_DIM


@dataclass
class RefinerConfig:
    calibration_version: int = 3
    width: int = 64
    layers: int = 3
    name_dim: int = 768
    action_dim: int = 2304
    noise_dim: int = 4
    cutoff_hz: float = 4.5
    reference_fps: float = 30.0
    max_angle_rad: float = 0.3
    # Fixed checkpoint contract metadata; these are not training variants.
    window_context: bool = False
    composition: str | None = None
    motion_context: str | None = None
    output_max_hz: float | None = None
    identity_position_tolerance: float = .004
    identity_rotation_tolerance: float = .004
    geometry_loss: str | None = None
    geometry_bone_tolerance: float = .008
    geometry_boundary_tolerance: float = .016

    def __post_init__(self):
        # Fixed contract fields remain serialized so the selected checkpoint loads exactly.
        if self.calibration_version != 3 or self.window_context:
            raise ValueError('Only the current refiner architecture (calibration_version=3) is supported')
        fixed = {'composition': 'bone_transport', 'motion_context': 'kinematic_spectral',
                 'geometry_loss': 'kinematic'}
        for key, value in fixed.items():
            if getattr(self, key) not in (None, value):
                raise ValueError(f'Unsupported refiner {key}; expected {value}')
            setattr(self, key, value)
        if self.output_max_hz is None:
            self.output_max_hz = self.reference_fps / 2
        if min(self.width, self.layers, self.name_dim, self.noise_dim) < 1:
            raise ValueError('Refiner dimensions must be positive')
        if self.action_dim < 1 or self.cutoff_hz <= 0 or self.reference_fps <= 0 or self.max_angle_rad <= 0:
            raise ValueError('Invalid refiner time/amplitude configuration')
        if self.output_max_hz <= 0 or math.isnan(self.output_max_hz):
            raise ValueError('Invalid output frequency limit')
        if any(not math.isfinite(v) or v <= 0 for v in (
                self.identity_position_tolerance, self.identity_rotation_tolerance)):
            raise ValueError('Identity tolerances must be positive')

    @property
    def composition_contract(self):
        return 'current_bone_transport_v2'


def columns6(matrix):
    # The project's safe decoder uses COLUMNS (matrix_to_rotation_6d uses rows).
    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), -1)


def calibration_context(local, global_rotation, observed, virtual, scale, dt, loop, project, cutoff):
    """Observable per-frame kinematics and per-joint whole-window spectral context.

    Contains no GT or condition-specific reference. Static spectral channels
    expose the same controllable quantities optimized by the statistical loss.
    """
    root = global_rotation[:, :, :1].transpose(-1, -2)
    body_rotation = root @ global_rotation
    offset = torch.zeros_like(observed[:, :, :1])
    offset[..., 1] = observed[:, :, :1, 1]
    relative = observed - offset
    relative[:, :, 0] = 0
    body_position = (root @ relative[..., None]).squeeze(-1) / scale
    pose = torch.cat((columns6(body_rotation), body_position), -1)
    previous = pose.roll(1, 1) if loop else torch.cat((pose[:, :1], pose[:, :-1]), 1)
    virtual = (root @ (virtual - virtual[:, :, :1])[..., None]).squeeze(-1) / scale
    signal = torch.cat((local.flatten(-2), body_rotation.flatten(-2), virtual), -1)
    mean = signal.mean(1, keepdim=True)
    parts = [mean]
    for low, high in ((0., cutoff), (cutoff, 1.5 * cutoff), (1.5 * cutoff, float('inf'))):
        band = project(signal - mean, low, high)
        parts.append(torch.log1p((band.square().mean(1, keepdim=True) + 1e-10).sqrt() / .02))
    spectral = torch.cat(parts, -1).expand(*signal.shape[:-1], 84)
    return torch.cat((pose, (pose - previous) / dt, spectral), -1)


def exp_so3(v):
    """Rodrigues with analytic, finite gradients at exactly zero."""
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    k = torch.stack((o, -z, y, z, o, -x, -y, x, o), -1).reshape(*v.shape[:-1], 3, 3)
    theta = torch.linalg.vector_norm(v, dim=-1)[..., None, None]
    eye = torch.eye(3, device=v.device, dtype=v.dtype)
    return eye + torch.sinc(theta / math.pi) * k + 0.5 * torch.sinc(theta / (2 * math.pi)).square() * (k @ k)


def spectral_project(x, dt, loop, low=0., high=float('inf')):
    """Orthogonal Fourier / open-window DCT-II band projection, frequencies in Hz.

    DCT is implemented through an even extension; open ends never wrap together.
    An empty band deliberately returns zero, including beyond output Nyquist.
    """
    if not math.isfinite(float(dt)) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    n = x.shape[0]
    if n < 2:
        return x if low <= 0 < high else x * 0
    extended = x if loop else torch.cat((x, x.flip(0)), 0)
    freq = torch.fft.rfftfreq(len(extended), d=float(dt), device=x.device)
    mask = ((freq >= low) & (freq < high)).to(x.dtype)
    shape = (-1,) + (1,) * (x.ndim - 1)
    result = torch.fft.irfft(torch.fft.rfft(extended, dim=0) * mask.reshape(shape), n=len(extended), dim=0)
    return result[:n]


def fk(local, rest, parents):
    """Topological FK with world-aligned rest offsets, as physical HML expects."""
    globals_, positions = {}, {}

    def visit(j):
        if j in globals_:
            return
        p = int(parents[j])
        if p < 0:
            globals_[j] = local[:, j]
            positions[j] = rest[j].expand(local.shape[0], -1)
        else:
            visit(p)
            globals_[j] = globals_[p] @ local[:, j]
            positions[j] = positions[p] + (globals_[p] @ (rest[j] - rest[p])[..., None]).squeeze(-1)
    for j in range(len(parents)):
        visit(j)
    return torch.stack([globals_[j] for j in range(len(parents))], 1), torch.stack([positions[j] for j in range(len(parents))], 1)


def transport_positions(positions, before_global, after_global, parents):
    """Propagate rotation changes through CURRENT bone vectors, preserving lengths.

    RIC non-root XZ is relative to the translation root, whose own XZ is world.
    Local per-frame edge vectors are inferred from the base pose, not rest FK.
    Differences of global rotations make a zero update exactly zero.
    """
    relative = positions.clone()
    relative[:, 0, 0] = relative[:, 0, 2] = 0
    displacements = {}

    def visit(j):
        if j in displacements:
            return
        p = int(parents[j])
        if p < 0:
            displacements[j] = torch.zeros_like(positions[:, j])
        else:
            visit(p)
            edge = relative[:, j] - relative[:, p]
            local_edge = (before_global[:, p].transpose(-1, -2) @ edge[..., None])
            change = ((after_global[:, p] - before_global[:, p]) @ local_edge).squeeze(-1)
            displacements[j] = displacements[p] + change
    for j in range(len(parents)):
        visit(j)
    return positions + torch.stack([displacements[j] for j in range(len(parents))], 1)


def ric_body_positions(positions):
    """Root XZ stores a world path; all other joints already use root-relative XZ."""
    relative = positions.clone()
    relative[..., 0, 0] = relative[..., 0, 2] = 0
    return relative


def compose(base, delta, rest, parents, composition='bone_transport'):
    """[T,J,F] physical motion + [T,J,3] radians, with exact root preservation."""
    if composition != 'bone_transport':
        raise ValueError('Only current-bone transport is supported')
    rotation_input = base[..., 3:9].contiguous()
    local = rotation_6d_to_matrix_safe(rotation_input)
    nonroot = torch.as_tensor([int(p) >= 0 for p in parents], device=base.device)
    delta = delta * nonroot[None, :, None]
    refined = local @ exp_so3(delta)
    out = base.clone()
    # Add the 6D difference: zero residual returns even a non-unit base 6D exactly.
    out[..., 3:9] = base[..., 3:9] + (columns6(refined) - columns6(local))
    actual = rotation_6d_to_matrix_safe(out[..., 3:9].contiguous())
    gb, _ = fk(local, rest, parents)
    ga, _ = fk(actual, rest, parents)
    out[..., :3] = transport_positions(base[..., :3], gb, ga, parents)
    return torch.where(nonroot[None, :, None], out, base)


def sample_context(y, row, config):
    n = int(y['n_joints'][row])
    t = int(y['lengths'][row])
    loop = bool(y['is_loop'][row])
    speed = float(y['resample_speed_cond'][row])
    dt = (speed if loop or t <= 1 else (speed * t - 1) / (t - 1)) / config.reference_fps
    if dt <= 0 or not math.isfinite(dt):
        raise ValueError('Invalid resample_speed_cond')
    parents = y['parents'][row][:n]
    parents = [int(p) for p in parents]
    if (n < 2 or not parents or parents[0] >= 0 or not any(p >= 0 for p in parents)
            or any(p < -1 or p >= n or p == j for j, p in enumerate(parents))):
        raise ValueError('Invalid skeleton parents')
    # Fail cycles explicitly instead of recursive FK hanging.
    for j in range(n):
        seen = set()
        while j >= 0:
            if j in seen:
                raise ValueError('Cyclic skeleton')
            seen.add(j)
            j = parents[j]
    return n, t, loop, dt, parents


class DetailBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.temporal = nn.Conv1d(width, width, 5)
        self.graph = nn.Linear(width, width)

    def forward(self, h, adjacency, loop):
        v = self.norm(h)
        temporal = v.permute(1, 2, 0)
        if loop:
            temporal = temporal.index_select(-1, torch.arange(-2, h.shape[0] + 2, device=h.device) % h.shape[0])
        else:
            temporal = F.pad(temporal, (2, 2), mode='replicate')
        temporal = self.temporal(temporal).permute(2, 0, 1)
        return h + F.silu(temporal + self.graph(torch.einsum('jk,tkc->tjc', adjacency, v)))


class MotionDetailRefiner(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or RefinerConfig()
        c = self.config
        self.names = nn.Linear(c.name_dim, 16)
        self.action = nn.Linear(c.action_dim, 16)
        self.input = nn.Linear(6 + 6 + 3 + 3 + JOINT_STRUCT_DIM + 16 + 16 + 4 + c.noise_dim, c.width)
        self.blocks = nn.ModuleList([DetailBlock(c.width) for _ in range(c.layers)])
        self.output = nn.Linear(c.width, 3)
        nn.init.normal_(self.output.weight, std=0.001)
        nn.init.zeros_(self.output.bias)
        context_width = 102
        if context_width:
            # Initialize added kinematic context columns to zero.
            original = self.input
            self.input = nn.Linear(original.in_features + context_width, c.width)
            with torch.no_grad():
                self.input.weight[:, :original.in_features].copy_(original.weight)
                self.input.weight[:, original.in_features:].zero_()
                self.input.bias.copy_(original.bias)

    def forward(self, base, y, seed=0, alpha=1.):
        if not math.isfinite(float(alpha)) or not 0 <= alpha <= 1:
            raise ValueError('alpha must be in [0,1]')
        if alpha == 0:
            return base, torch.zeros_like(base[:, :, :3])
        base = base.detach()
        out = base.clone()
        residual = torch.zeros_like(base[:, :, :3])
        c = self.config
        for b in range(base.shape[0]):
            n, t, loop, dt, parents = sample_context(y, b, c)
            if not 0 < t <= base.shape[-1] or not 0 < n <= base.shape[1]:
                raise ValueError('Invalid valid motion extent')
            x = base[b, :n, :, :t].permute(2, 0, 1)
            rest = y['rest_pos_ric_hml'][b, :n].to(x)
            rest = rest - rest[0]
            scale = rest.square().sum(-1).mean().sqrt().clamp_min(1e-3)
            offsets = torch.stack([rest[j] - rest[p] if p >= 0 else rest[j] * 0 for j, p in enumerate(parents)]) / scale
            rot = x[..., 3:9]
            change = rot - (rot.roll(1, 0) if loop else torch.cat((rot[:1], rot[:-1]), 0))
            root = x[:, 0, :3]
            root_change = root - (root.roll(1, 0) if loop else torch.cat((root[:1], root[:-1]), 0))
            gen = torch.Generator(device=x.device).manual_seed(int(seed) + b)
            z = torch.randn((t, n, c.noise_dim), device=x.device, dtype=x.dtype, generator=gen)
            # Correlated stochastic context, using the same boundary as the residual.
            z = spectral_project(z, dt, loop, high=0.8 * (0.5 / dt))
            static = torch.cat((offsets, y['joint_struct'][b, :n].to(x),
                                self.names(y['joints_names_embs'][b, :n].to(x))), -1)
            action = self.action(y['detail_action'][b].to(x))
            timing = x.new_tensor([dt, t * dt, float(loop), float(y.get('loop_tile_count', [1] * len(base))[b])])
            features = torch.cat((rot, change / dt, (root_change / scale / dt)[:, None].expand(-1, n, -1),
                                  static[None].expand(t, -1, -1), action[None, None].expand(t, n, -1),
                                  timing[None, None].expand(t, n, -1), z), -1)
            local = rotation_6d_to_matrix_safe(rot.contiguous())
            global_rotation, virtual = fk(local, y['rest_pos_ric_hml'][b, :n].to(x), parents)
            def project(v, low, high):
                return spectral_project(v[0], dt, loop, low, high)[None]
            context = calibration_context(local[None], global_rotation[None], x[None, ..., :3],
                virtual[None], scale, dt, loop, project, c.cutoff_hz)[0]
            features = torch.cat((features, context), -1)
            adjacency = torch.eye(n, device=x.device, dtype=x.dtype)
            for j, p in enumerate(parents):
                if p >= 0:
                    adjacency[j, p] = adjacency[p, j] = 1
            adjacency = adjacency / adjacency.sum(-1, keepdim=True)
            h = F.silu(self.input(features))
            for block in self.blocks:
                h = block(h, adjacency, loop)
            upper_hz = min(c.output_max_hz, float(y.get('detail_max_hz', [float('inf')] * len(base))[b]))
            # All bands, including DC: cutoff_hz only partitions statistics.
            delta = spectral_project(self.output(h), dt, loop, high=upper_hz)
            # One scale per joint/window preserves the chosen output bandwidth.
            peak = torch.linalg.vector_norm(delta, dim=-1).amax(0, keepdim=True)[..., None]
            delta = delta * (c.max_angle_rad / peak.clamp_min(c.max_angle_rad))
            delta = delta * x.new_tensor([p >= 0 for p in parents])[None, :, None] * alpha
            out[b, :n, :, :t] = compose(x, delta, y['rest_pos_ric_hml'][b, :n].to(x), parents,
                                       c.composition).permute(1, 2, 0)
            residual[b, :n, :, :t] = delta.permute(1, 2, 0)
        return out, residual

    def architecture(self):
        architecture = asdict(self.config)
        # The selected checkpoint omits the disabled experimental context field.
        architecture.pop('window_context')
        return architecture
