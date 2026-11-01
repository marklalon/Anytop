"""Homogeneous cache-group training, batched across frozen-backbone seeds.

The public refiner remains the mixed-rig inference/reference implementation.
Only reusable tensor regions are compiled: RNG, FFT, topology, quantiles,
and logging stay outside Dynamo so changing seeds/rig identities cannot cause
graph breaks or Python-value recompilation of the neural network.
"""
import torch
import torch.nn.functional as F

from model.motion_detail_refiner import (columns6, exp_so3, sample_context,
                                         calibration_context, ric_body_positions)
from train.detail_objectives import (calibration_features, coordination_pairs, statistics_loss,
                                     geometry_objective, identity_guard)
from utils.rotation_conversions import rotation_6d_to_matrix_safe


def neural_forward(net, motion, static, names, action, timing, noise, adjacency, loop, context=None):
    """Shared parameters, [B,T,J,C] activations; no per-seed neural loop."""
    b, t, n, _ = motion.shape
    joint = torch.cat((static, net.names(names)), -1)
    label = net.action(action)
    features = torch.cat((motion, joint[:, None].expand(-1, t, -1, -1),
                          label[:, None, None].expand(-1, t, n, -1),
                          timing[:, None, None].expand(-1, t, n, -1), noise), -1)
    features = torch.cat((features, context), -1)
    h = F.silu(net.input(features))
    for block in net.blocks:
        v = block.norm(h)
        temporal = v.permute(0, 2, 3, 1).reshape(b * n, v.shape[-1], t)
        if loop:
            temporal = temporal.index_select(-1, torch.arange(-2, t + 2, device=h.device) % t)
        else:
            temporal = F.pad(temporal, (2, 2), mode='replicate')
        temporal = block.temporal(temporal).reshape(b, n, -1, t).permute(0, 3, 1, 2)
        h = h + F.silu(temporal + block.graph(torch.matmul(adjacency, v)))
    return net.output(h)


def refine_rotation(local, delta):
    return local @ exp_so3(delta)


class BatchedDetailRuntime:
    def __init__(self, net, compile_mode=None, dynamic=True, backend='inductor'):
        self.net = net
        self.neural = neural_forward
        self.statistics = statistics_loss
        if compile_mode:
            if backend == 'inductor':
                from utils.compile_util import configure_triton_c_compiler
                configure_triton_c_compiler()
            def compile_region(fn, mode='default'):
                return torch.compile(fn, backend=backend, mode=mode,
                                     fullgraph=True, dynamic=dynamic)
            # Keep CUDA graph capture on the neural region only. Interleaving
            # multiple captured forward/backward regions across changing rigs
            # can violate CUDAGraph Trees tensor-lifetime invariants.
            self.neural = compile_region(neural_forward, compile_mode)
            self.statistics = compile_region(statistics_loss)
        self.compile_mode = compile_mode

    def loss(self, batch, target, scale, seed):
        return batch.loss(self, target, scale, seed)


class PreparedDetailBatch:
    """Immutable fixed-input work; valid only for one homogeneous cache group."""
    @torch.no_grad()
    def __init__(self, base, y, config, references, matched=False, clips=None):
        self.config = config
        self.matched = matched
        self.base = base.detach()
        contexts = [sample_context(y, i, config) for i in range(len(base))]
        if any(context != contexts[0] for context in contexts[1:]):
            raise ValueError('Batched training requires identical topology and time settings within a group')
        self.n, self.t, self.loop, self.dt, self.parents = contexts[0]
        n, t = self.n, self.t
        if not 0 < t <= base.shape[-1] or not 0 < n <= base.shape[1]:
            raise ValueError('Invalid valid motion extent')
        self.x = base[:, :n, :, :t].permute(0, 3, 1, 2)
        self.b = len(base)
        self.observation_groups = None
        if matched:
            ids = list(clips) if clips is not None else [None] * self.b
            if len(ids) != self.b:
                raise ValueError('Clip identifiers must match matched observations')
            self.observation_groups = [torch.tensor([i for i, clip in enumerate(ids) if clip == key],
                device=base.device) for key in dict.fromkeys(ids)]
        self.rest = y['rest_pos_ric_hml'][0, :n].to(base)
        if not torch.equal(y['rest_pos_ric_hml'][:, :n], self.rest[None].expand(len(base), -1, -1)):
            raise ValueError('Batched training requires the same rest skeleton within a group')
        relative = self.rest - self.rest[:1]
        self.scale = relative.square().sum(-1).mean().sqrt().clamp_min(1e-3)
        offsets = torch.stack([relative[j] - relative[p] if p >= 0 else relative[j] * 0
                               for j, p in enumerate(self.parents)]) / self.scale
        self.static = torch.cat((offsets[None].expand(self.b, -1, -1),
                                 y['joint_struct'][:, :n].to(base)), -1)
        self.names = y['joints_names_embs'][:, :n].to(base)
        self.action = y['detail_action'].to(base)
        self.timing = base.new_tensor([[self.dt, t * self.dt, float(self.loop),
                                       float(y.get('loop_tile_count', [1] * self.b)[i])]
                                      for i in range(self.b)])
        rot = self.x[..., 3:9]
        change = rot - self.previous(rot)
        root = self.x[:, :, 0, :3]
        root_change = root - self.previous(root)
        self.motion = torch.cat((rot, change / self.dt,
                                 (root_change / self.scale / self.dt)[:, :, None].expand(-1, -1, n, -1)), -1)
        children = [j for j, p in enumerate(self.parents) if p >= 0]
        self.children = torch.tensor(children, device=base.device, dtype=torch.long)
        self.ancestors = torch.tensor([self.parents[j] for j in children],
                                      device=base.device, dtype=torch.long)
        self.coordination_pairs = tuple(torch.tensor(indices, device=base.device, dtype=torch.long)
                                       for indices in coordination_pairs(self.parents))
        self.nonroot = base.new_tensor([p >= 0 for p in self.parents])[None, None, :, None]
        self.nonroot_bool = self.nonroot.bool()
        self.adjacency = torch.eye(n, device=base.device, dtype=base.dtype)
        for j, p in enumerate(self.parents):
            if p >= 0:
                self.adjacency[j, p] = self.adjacency[p, j] = 1
        self.adjacency /= self.adjacency.sum(-1, keepdim=True)
        self.reference = {k: torch.stack([row[k] for row in references]) for k in references[0]}
        self.local = self.reference['rotation']
        self.local = rotation_6d_to_matrix_safe(self.x[..., 3:9].contiguous())
        self.reference['rotation'] = self.local
        self.base_columns = columns6(self.local)
        input_cap = float(y.get('detail_max_hz', [float('inf')])[0])
        if any(float(v) != input_cap for v in y.get('detail_max_hz', [input_cap] * self.b)):
            raise ValueError('Batched training requires the same detail frequency cap within a group')
        self.upper_hz = min(config.output_max_hz, input_cap)
        self.levels = []
        depths = {}

        def depth(j):
            if j not in depths:
                depths[j] = 0 if self.parents[j] < 0 else depth(self.parents[j]) + 1
            return depths[j]
        ordered = []
        previous = []
        for d in range(max(depth(j) for j in range(n)) + 1):
            nodes = [j for j in range(n) if depths[j] == d]
            slots = [previous.index(self.parents[j]) for j in nodes] if d else []
            indices = torch.tensor(nodes, device=base.device)
            parent_slots = torch.tensor(slots, device=base.device, dtype=torch.long)
            offsets = torch.stack([self.rest[j] - self.rest[self.parents[j]] for j in nodes]) if d else self.rest[indices]
            self.levels.append((indices, parent_slots, offsets))
            ordered.extend(nodes)
            previous = nodes
        self.joint_order = torch.tensor([ordered.index(j) for j in range(n)], device=base.device)
        self.masks = {}
        frequencies = torch.fft.rfftfreq(t if self.loop else 2 * t, d=self.dt, device=base.device)
        for lo, hi in ((0., config.cutoff_hz), (0., .8 * (.5 / self.dt)),
                       (0., self.upper_hz),
                       (config.cutoff_hz, config.cutoff_hz * 1.5), (config.cutoff_hz * 1.5, float('inf'))):
            self.masks[(lo, hi)] = ((frequencies >= lo) & (frequencies < hi)).to(base.dtype)
        ranks = torch.tensor([.25, .5, .9], dtype=base.dtype) * (t - 1)
        self.quantile_lower = ranks.floor().long().to(base.device)
        self.quantile_upper = ranks.ceil().long().to(base.device)
        self.quantile_weight = (ranks - ranks.floor()).to(base.device)[None, :, None]
        self.generators = [torch.Generator(device=base.device) for _ in range(self.b)]
        self.transport_levels = None
        self.base_global = self.global_rotations(self.local)
        self.reference['global_rotation'] = self.base_global
        position = self.x[..., :3].clone()
        position[:, :, 0, 0] = position[:, :, 0, 2] = 0
        self.transport_levels = []
        for indices, slots, _ in self.levels:
            parents = torch.tensor([self.parents[j] for j in indices.cpu().tolist()], device=base.device)
            parent_global = edge = None
            if len(slots):
                parent_global = self.base_global.index_select(2, parents)
                vector = position.index_select(2, indices) - position.index_select(2, parents)
                edge = (parent_global.transpose(-1, -2) @ vector[..., None]).squeeze(-1)
            self.transport_levels.append((indices, slots, parents, parent_global, edge))
        self.context = None
        global_rotation, virtual = self.fk(self.local)
        self.context = calibration_context(self.local, global_rotation, self.x[..., :3],
            virtual, self.scale, self.dt, self.loop, self.project, config.cutoff_hz)

    def previous(self, x):
        return x.roll(1, 1) if self.loop else torch.cat((x[:, :1], x[:, :-1]), 1)

    def project(self, x, low=0., high=float('inf')):
        if self.t < 2:
            return x if low <= 0 < high else x * 0
        key = (low, high)
        if key not in self.masks:
            freq = torch.fft.rfftfreq(self.t if self.loop else 2 * self.t, d=self.dt, device=x.device)
            self.masks[key] = ((freq >= low) & (freq < high)).to(x.dtype)
        extended = x if self.loop else torch.cat((x, x.flip(1)), 1)
        mask = self.masks[key].reshape(1, -1, *([1] * (x.ndim - 2)))
        return torch.fft.irfft(torch.fft.rfft(extended, dim=1) * mask,
                               n=extended.shape[1], dim=1)[:, :self.t]

    def fk(self, local):
        globals_, positions = [], []
        for indices, parent_slots, offsets in self.levels:
            own = local.index_select(2, indices)
            if not globals_:
                globals_.append(own)
                positions.append(offsets[None, None].expand(self.b, self.t, -1, -1))
            else:
                parent = globals_[-1].index_select(2, parent_slots)
                globals_.append(parent @ own)
                positions.append(positions[-1].index_select(2, parent_slots) +
                                 (parent @ offsets[..., None]).squeeze(-1))
        return (torch.cat(globals_, 2).index_select(2, self.joint_order),
                torch.cat(positions, 2).index_select(2, self.joint_order))

    def features(self, r, g, positions):
        _, positions = self.fk(r)
        positions = positions - positions[:, :, :1]
        return calibration_features(r, g, positions, self.scale, self.project,
                                    self.quantiles, self.config, self.coordination_pairs,
                                    self.children)

    def global_rotations(self, local):
        # Preservation needs rotations only; skip all FK position work.
        globals_ = []
        for indices, parent_slots, _ in self.levels:
            own = local.index_select(2, indices)
            globals_.append(own if not globals_ else globals_[-1].index_select(2, parent_slots) @ own)
        return torch.cat(globals_, 2).index_select(2, self.joint_order)

    def transported_positions(self, global_rotation):
        displacements = []
        for indices, slots, parents, before, edge in self.transport_levels:
            if before is None:
                change = torch.zeros_like(self.x[..., :3].index_select(2, indices))
            else:
                delta = ((global_rotation.index_select(2, parents) - before) @ edge[..., None]).squeeze(-1)
                change = displacements[-1].index_select(2, slots) + delta
            displacements.append(change)
        return self.x[..., :3] + torch.cat(displacements, 2).index_select(2, self.joint_order)

    def quantiles(self, envelope):
        # torch.quantile validates q via a CUDA scalar read on each call. These
        # fixed ranks are validated/prepared on CPU once, preserving linear interpolation.
        ordered = envelope.sort(dim=1).values
        return torch.lerp(ordered.index_select(1, self.quantile_lower),
                          ordered.index_select(1, self.quantile_upper), self.quantile_weight)

    def sample_noise(self, seed, out=None):
        # Preserve independent row seed streams exactly; RNG never enters a compiled graph.
        if out is not None:
            for b, generator in enumerate(self.generators):
                torch.randn(out[b].shape, device=out.device, dtype=out.dtype,
                            generator=generator.manual_seed(int(seed) + b), out=out[b])
            return out
        return torch.stack([torch.randn((self.t, self.n, self.config.noise_dim),
                                         device=self.base.device, dtype=self.base.dtype,
                                         generator=self.generators[b].manual_seed(int(seed) + b))
                             for b in range(self.b)])

    def loss(self, runtime, target, scale, seed=None, *, noise=None, materialize=True):
        if noise is None:
            noise = self.sample_noise(seed)
        noise = self.project(noise, high=.8 * (.5 / self.dt))
        raw = runtime.neural(runtime.net, self.motion, self.static, self.names, self.action, self.timing,
                             noise, self.adjacency, self.loop, self.context)
        delta = self.project(raw, high=self.upper_hz)
        peak = torch.linalg.vector_norm(delta, dim=-1).amax(1, keepdim=True)[..., None]
        delta = delta * (self.config.max_angle_rad / peak.clamp_min(self.config.max_angle_rad)) * self.nonroot
        refined = refine_rotation(self.local, delta)
        rotation = torch.where(self.nonroot_bool,
            self.x[..., 3:9] + (columns6(refined) - self.base_columns), self.x[..., 3:9])
        r = rotation_6d_to_matrix_safe(rotation)
        g = self.global_rotations(r)
        x = torch.where(self.nonroot_bool, torch.cat((self.transported_positions(g), rotation,
                                                     self.x[..., 9:]), -1), self.x)
        kp = ((x[..., :3] - self.x[..., :3]) / self.scale).square().mean()
        kr = (g - self.reference['global_rotation']).square().mean() + (r - self.local).square().mean()
        positions = ric_body_positions(x[..., :3])
        bones = torch.linalg.vector_norm(positions[:, :, self.children] - positions[:, :, self.ancestors], dim=-1)
        bone_mse = ((bones - self.reference['bones']) / self.scale).square().mean((1, 2))
        difference = (x[..., :3] - self.x[..., :3]) / self.scale
        guard = kp.new_zeros(())
        if self.matched:
            guard = identity_guard(difference.square().mean((1, 2, 3)),
                (r - self.local).square().mean((1, 2, 3, 4)), self.config)
        features = self.features(r, g, x[..., :3])
        if self.observation_groups is not None:
            features = torch.stack([features.index_select(0, indices).mean(0)
                                    for indices in self.observation_groups])
        losses = {'keep': kp + kr + guard, 'geo': geometry_objective(bone_mse),
                  'reg': delta.square().sum() / (self.b * self.t * max(1, self.n - 1) * 3),
                  'stat': runtime.statistics(features, target, scale)}
        if not materialize:
            return losses
        out = self.base.clone()
        residual = torch.zeros_like(self.base[:, :, :3])
        out[:, :self.n, :, :self.t] = x.permute(0, 2, 3, 1)
        residual[:, :self.n, :, :self.t] = delta.permute(0, 2, 3, 1)
        return losses, out, residual
