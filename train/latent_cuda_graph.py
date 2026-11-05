"""Eager CUDA Graph training steps for latent diffusion (without torch.compile)."""
import warnings

import numpy as np
import torch

from diffusion.latent_diffusion import validate_latent_conditions
from model.latent_motion_denoiser import LATENT_CONDITION_KEYS


class EagerCudaGraphSteps:
    """Cache bounded denoiser captures; run excess input signatures eagerly."""
    def __init__(self, model, diffusion, optimizer, amp_dtype, device, scaler, max_graphs=2):
        if max_graphs < 1:
            raise ValueError('max_graphs must be positive')
        self.model, self.diffusion, self.optimizer = model, diffusion, optimizer
        self.amp_dtype, self.device = amp_dtype, device
        self.scaler = scaler
        self.graphs = {}
        self.max_graphs = max_graphs
        self._warned_cache_limit = False

    def _conditions(self, y):
        # Tensor values change through copy_(), not through new cache buckets.
        # Convert host conditions before capture so they cannot become constants.
        return {key: (value.to(self.device) if torch.is_tensor(value)
                      else torch.as_tensor(value, device=self.device))
                for key, value in y.items()
                if key in LATENT_CONDITION_KEYS and value is not None}

    def _eager_step(self, target, t, noise, y):
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=self.amp_dtype,
                            enabled=self.amp_dtype is not None):
            loss = self.diffusion.training_losses(
                self.model, target, t, {'y': y}, noise)['loss'].mean()
        self.scaler.scale(loss).backward()
        return loss.detach()

    @staticmethod
    def _clone_tree(value):
        if torch.is_tensor(value):
            return value.detach().clone()
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, dict):
            return {key: EagerCudaGraphSteps._clone_tree(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(EagerCudaGraphSteps._clone_tree(item) for item in value)
        if isinstance(value, list):
            return [EagerCudaGraphSteps._clone_tree(item) for item in value]
        return value

    @staticmethod
    def _copy_tree(destination, source):
        if torch.is_tensor(destination):
            destination.copy_(source)
        elif isinstance(destination, dict):
            for key in destination:
                EagerCudaGraphSteps._copy_tree(destination[key], source[key])
        elif isinstance(destination, (tuple, list)):
            for dst, src in zip(destination, source):
                EagerCudaGraphSteps._copy_tree(dst, src)

    @staticmethod
    def _signature(value):
        if torch.is_tensor(value):
            return ('tensor', tuple(value.shape), value.dtype, value.device)
        if isinstance(value, np.ndarray):
            # Host arrays are constants during capture, unlike copied tensor
            # inputs. Their contents must distinguish graphs, not just shape.
            return ('ndarray', value.shape, value.dtype, value.tobytes())
        if isinstance(value, dict):
            return tuple((key, EagerCudaGraphSteps._signature(value[key])) for key in sorted(value))
        if isinstance(value, (tuple, list)):
            return (type(value), tuple(EagerCudaGraphSteps._signature(item) for item in value))
        return (type(value), value)

    def _key(self, target, t, noise, y):
        return self._signature((target, t, noise, y))

    def _build(self, target, t, noise, y):
        static_target = target.detach().clone()
        static_t = t.detach().clone()
        static_noise = noise.detach().clone()
        static_y = self._clone_tree(y)
        # Each graph owns its gradient storage. Reattach it when switching
        # buckets; unused parameters must retain grad=None (AdamW semantics).
        self.optimizer.zero_grad(set_to_none=True)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.optimizer.zero_grad(set_to_none=False)
                with torch.autocast('cuda', dtype=self.amp_dtype,
                                    enabled=self.amp_dtype is not None):
                    loss = self.diffusion.training_losses(
                        self.model, static_target, static_t, {'y': static_y}, static_noise)['loss'].mean()
                self.scaler.scale(loss).backward()
                del loss
        torch.cuda.current_stream(self.device).wait_stream(stream)
        self.optimizer.zero_grad(set_to_none=False)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.device(self.device), torch.cuda.graph(graph, stream=stream):
            self.optimizer.zero_grad(set_to_none=False)
            with torch.autocast('cuda', dtype=self.amp_dtype,
                                enabled=self.amp_dtype is not None):
                captured_loss = self.diffusion.training_losses(
                    self.model, static_target, static_t, {'y': static_y}, static_noise)['loss'].mean()
            self.scaler.scale(captured_loss).backward()
        # Retain the output storage, not its eager autograd graph/AccumulateGrad
        # nodes, which otherwise keep the previous capture stream alive.
        captured_loss = captured_loss.detach()
        graph.replay()
        gradients = tuple(parameter.grad for parameter in self.model.parameters())
        self.graphs[self._key(target, t, noise, y)] = (
            graph, static_target, static_t, static_noise, static_y, captured_loss, gradients)
        return captured_loss

    def step(self, target, t, noise, y):
        # Validation is skipped inside capture, so check every incoming batch.
        validate_latent_conditions(y, target.shape[1])
        y = self._conditions(y)
        key = self._key(target, t, noise, y)
        if key not in self.graphs:
            # Do not retain an unbounded number of graph-private memory pools.
            # Extra signatures stay eager, avoiding eviction/recapture thrashing.
            if len(self.graphs) >= self.max_graphs:
                if not self._warned_cache_limit:
                    warnings.warn(
                        f'CUDA graph cache reached {self.max_graphs} signatures; '
                        'additional input signatures run eagerly to bound GPU memory.',
                        RuntimeWarning, stacklevel=2)
                    self._warned_cache_limit = True
                return self._eager_step(target, t, noise, y)
            loss = self._build(target, t, noise, y)
        else:
            graph, static_target, static_t, static_noise, static_y, loss, gradients = self.graphs[key]
            for parameter, gradient in zip(self.model.parameters(), gradients):
                parameter.grad = gradient
            static_target.copy_(target)
            static_t.copy_(t)
            static_noise.copy_(noise)
            self._copy_tree(static_y, y)
            graph.replay()
        return loss

    def optimizer_step(self):
        """Backward is captured; scaler state and optimizer updates stay eager."""
        previous_scale = self.scaler.get_scale()
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 1.0, error_if_nonfinite=not self.scaler.is_enabled())
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return grad_norm, self.scaler.get_scale() >= previous_scale
