"""Capture fixed-cache forward/backward work; keep AdamW and checks on the host.

Every graph writes the same persistent parameter gradients. Temporary graph
allocations share a pool: no activation/output from a replay survives the next
replay. Only noise and detached scalar metrics have group-specific storage.
"""
import torch

from train.detail_runtime import BatchedDetailRuntime
from train.detail_gradients import DetailGradientMonitor


class CudaGraphDetailTrainer:
    def __init__(self, net, weights, max_groups=256):
        self.net = net
        self.weights = dict(weights)
        self.keys = ('keep', 'geo', 'reg', 'stat', 'identity')
        self.gradient_keys = DetailGradientMonitor.metric_keys
        self.mixer = DetailGradientMonitor(net)
        self.runtime = BatchedDetailRuntime(net)
        self.max_groups = max_groups
        self.graphs = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.stream = torch.cuda.Stream(device=next(net.parameters()).device)
        # The mixer owns stable gradient/scratch addresses across captures.
        # Never use zero_grad(set_to_none=True) on this path.

    def _backward(self, group, scale, noise, metrics):
        losses = group['_batch'].loss(self.runtime, group['target_features'], scale,
                                      noise=noise[0], materialize=False)
        if self.weights['identity']:
            matched = group['_matched_batch'].loss(
                self.runtime, group['target_features'], scale,
                noise=noise[1], materialize=False)
            losses['identity'] = (matched['stat'] + matched['keep'] + matched['geo']
                                  + .01 * matched['reg'])
            for crop, buffer in zip(group.get('_identity_crops', ()), noise[2:]):
                extra = crop['batch'].loss(self.runtime, crop['target'], crop['scale'],
                                           noise=buffer, materialize=False)
                losses['identity'] = losses['identity'] + .5 * (
                    extra['stat'] + extra['keep'] + extra['geo'] + .01 * extra['reg'])
        else:
            losses['identity'] = losses['stat'].new_zeros(())
        loss = sum(self.weights[k] * losses[k] for k in self.keys)
        main = sum(self.weights[k] * losses[k] for k in self.keys if k != 'identity')
        identity = self.weights['identity'] * losses['identity'] if self.weights['identity'] else None
        diagnostics = self.mixer.backward(main, identity)
        metrics.copy_(torch.stack([loss.detach()] + [losses[k].detach() for k in self.keys]
                                 + [diagnostics[k] for k in self.gradient_keys]))

    def backward(self, group, scale, seed):
        """Return detached loss metrics, with fresh (unclipped) gradients ready.

        The caller must validate loss/gradients before calling optimizer.step().
        Capture warmup never steps the optimizer or changes network parameters.
        Cache overflow uses eager execution rather than repeated recapture.
        """
        key = (id(group['_batch']), scale.data_ptr())
        entry = self.graphs.get(key)
        batches = [group['_batch']]
        if self.weights['identity']:
            batches.append(group['_matched_batch'])
            batches.extend(crop['batch'] for crop in group.get('_identity_crops', ()))
        seeds = [seed, seed] + [seed + 1000 + i * 100 for i in range(max(0, len(batches) - 2))]
        if entry is None:
            noise = [batch.sample_noise(s) for batch, s in zip(batches, seeds)]
            metrics = noise[0].new_empty(1 + len(self.keys) + len(self.gradient_keys))
            if len(self.graphs) >= self.max_groups:
                self._backward(group, scale, noise, metrics)
                return metrics
            current = torch.cuda.current_stream(noise[0].device)
            self.stream.wait_stream(current)
            with torch.cuda.stream(self.stream):
                # Initialize library workspaces/plans outside stream capture.
                for _ in range(2):
                    self._backward(group, scale, noise, metrics)
            current.wait_stream(self.stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool, stream=self.stream):
                self._backward(group, scale, noise, metrics)
            entry = (graph, noise, metrics)
            self.graphs[key] = entry
        graph, noise, metrics = entry
        for batch, buffer, s in zip(batches, noise, seeds):
            batch.sample_noise(s, out=buffer)
        graph.replay()
        return metrics
