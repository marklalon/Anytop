import math
import torch 
import torch.nn as nn
from typing import Optional, Union, Callable, Tuple
from torch import Tensor

from data_loaders.truebones.truebones_utils.topology_relations import (
    NUM_EDGE_EMBEDDINGS,
    NUM_TOPOLOGY_CODES,
    resolve_edge_codes,
)
import torch.nn.functional as F
CUDA_LAUNCH_BLOCKING=1


def run_in_fp32(module, tensor, *rest):
    """Run a broadcast conditioning head outside autocast, on fp32 input.

    These heads emit one vector per sample (or per joint) that is then added to
    every one of ~1e6 tokens, so their backward sums over all of those tokens and
    their weight gradients are the largest in the network -- they are the first
    tensors to overflow fp16 as the loss scale rises. They are also tiny, so
    fp32 costs under 1% of a training step (inside measurement noise) and cuts
    the global parameter-gradient error by ~37%.
    See docs/fp16_vs_bf16_precision.md.

    Defined here rather than in model.motion_diffusion because the decoder's own
    conditioning heads need it too and anytop imports this module, not the
    other way round; ``model.motion_diffusion.run_in_fp32`` re-exports this function.
    """
    with torch.autocast(device_type=tensor.device.type, enabled=False):
        return module(tensor.float(), *rest)


class QKNorm(nn.Module):
    """Per-head RMS normalization of query/key vectors before the attention
    dot product.

    Normalizes over the head_dim axis so the attention logit magnitude is
    decoupled from the Q/K projection weight norm. Without this, the linear_q /
    linear_k weights grow unbounded late in training (nothing in the loss or the
    constant 1/sqrt(d) scaling restrains them); once the logits get large the
    softmax saturates, its gradient explodes, that pushes Q/K larger still, and
    the run diverges in a positive-feedback loop. Bounding the per-head norm of
    q and k caps the logits regardless of weight magnitude, which is the
    standard cure (a.k.a. QK-LayerNorm / QK-norm). Computed in fp32 for
    stability and cast back to the input dtype.
    """

    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.normalized_shape = (head_dim,)
        self.weight = nn.Parameter(torch.ones(head_dim))

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        return F.rms_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float(),
            self.eps,
        ).to(in_dtype)


class _AttentionOutProjection(nn.Module):
    def __init__(self, embed_dim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(embed_dim, embed_dim))
        if bias:
            self.bias = nn.Parameter(torch.empty(embed_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)


# Temporal RoPE frequency range, in radians per window frame.
# The fastest pair turns half a cycle per frame (Nyquist, a two-frame period);
# the slowest has a period far beyond the window, so the low pairs read as a
# smooth distance kernel. The pairs in between are spaced geometrically.
TEMPORAL_ROPE_MAX_FREQ = math.pi
TEMPORAL_ROPE_MAX_PERIOD = 240.0


def temporal_rope_tables(length: int, head_dim: int, device: torch.device,
                         loop: bool = False) -> Tuple[Tensor, Tensor]:
    """``(cos, sin)``, each ``[length, head_dim // 2]``, for frames ``0..length-1``.

    ``loop=True`` is the table of a closed window (frame ``length`` is frame 0):
    each geometric frequency snaps to the nearest whole number of turns per
    window, so a query-key product depends on the offset modulo ``length`` and
    the seam is an ordinary one-frame step. The slowest pairs land on one turn
    per window, the circular distance kernel.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"temporal RoPE needs an even head_dim, got {head_dim}")
    pairs = head_dim // 2
    min_freq = 2.0 * math.pi / TEMPORAL_ROPE_MAX_PERIOD
    ramp = torch.arange(pairs, device=device, dtype=torch.float32) / max(pairs - 1, 1)
    freqs = TEMPORAL_ROPE_MAX_FREQ * (min_freq / TEMPORAL_ROPE_MAX_FREQ) ** ramp
    frames = torch.arange(length, device=device)
    if loop:
        turns = torch.round(freqs * length / (2.0 * math.pi)).long().clamp(1, max(length // 2, 1))
        # Integer phase (t * m mod length) keeps the period exact in fp32.
        phase = (frames[:, None] * turns[None, :]) % length
        angles = phase.float() * (2.0 * math.pi / length)
    else:
        angles = frames.float()[:, None] * freqs[None, :]
    return torch.cos(angles), torch.sin(angles)


def apply_temporal_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate ``x`` [..., T, head_dim] pairwise (first half, second half) by the
    per-frame angles, so a query-key product depends on the frame offset."""
    half = x.shape[-1] // 2
    x1 = x[..., :half].float()
    x2 = x[..., half:].float()
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)


class SelectiveMultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim must be divisible by num_heads (got embed_dim={embed_dim}, num_heads={num_heads})")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = float(dropout)
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.q_norm = QKNorm(self.head_dim)
        self.k_norm = QKNorm(self.head_dim)

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = _AttentionOutProjection(embed_dim, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)

    def _apply_attention_mask(self, scores: Tensor, attn_mask: Optional[Tensor]) -> Tensor:
        if attn_mask is None:
            return scores
        if attn_mask.dtype == torch.bool:
            if attn_mask.dim() == 2:
                return scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            if attn_mask.dim() == 3:
                batch_size, num_heads, tgt_len, src_len = scores.shape
                if attn_mask.shape[0] == batch_size * num_heads:
                    return scores.masked_fill(attn_mask.view(batch_size, num_heads, tgt_len, src_len), float('-inf'))
                if attn_mask.shape[0] == batch_size:
                    return scores.masked_fill(attn_mask.unsqueeze(1), float('-inf'))
        else:
            attn_mask = attn_mask.to(device=scores.device, dtype=scores.dtype)
            if attn_mask.dim() == 2:
                return scores + attn_mask.unsqueeze(0).unsqueeze(0)
            if attn_mask.dim() == 3:
                batch_size, num_heads, tgt_len, src_len = scores.shape
                if attn_mask.shape[0] == batch_size * num_heads:
                    return scores + attn_mask.view(batch_size, num_heads, tgt_len, src_len)
                if attn_mask.shape[0] == batch_size:
                    return scores + attn_mask.unsqueeze(1)
        raise ValueError(f"Unsupported attn_mask shape: {tuple(attn_mask.shape)}")

    def _apply_key_padding_mask(self, scores: Tensor, key_padding_mask: Optional[Tensor]) -> Tensor:
        if key_padding_mask is None:
            return scores
        if key_padding_mask.dtype == torch.bool:
            return scores.masked_fill(key_padding_mask[:, None, None, :].to(device=scores.device), float('-inf'))
        return scores + key_padding_mask.to(device=scores.device, dtype=scores.dtype)[:, None, None, :]

    def _reshape_attention_mask_for_sdpa(
        self,
        attn_mask: Tensor,
        *,
        batch_size: int,
        tgt_len: int,
        src_len: int,
    ) -> Tensor:
        if attn_mask.dim() == 2:
            return attn_mask.unsqueeze(0).unsqueeze(0)
        if attn_mask.dim() == 3:
            if attn_mask.shape[0] == batch_size * self.num_heads:
                return attn_mask.reshape(batch_size, self.num_heads, tgt_len, src_len)
            if attn_mask.shape[0] == batch_size:
                return attn_mask.unsqueeze(1)
        raise ValueError(f"Unsupported attn_mask shape: {tuple(attn_mask.shape)}")

    def _attention_mask_to_additive(
        self,
        attn_mask: Optional[Tensor],
        *,
        batch_size: int,
        tgt_len: int,
        src_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Tensor]:
        if attn_mask is None:
            return None
        attn_mask = self._reshape_attention_mask_for_sdpa(
            attn_mask,
            batch_size=batch_size,
            tgt_len=tgt_len,
            src_len=src_len,
        )
        if attn_mask.dtype == torch.bool:
            return torch.zeros(attn_mask.shape, device=device, dtype=dtype).masked_fill(
                attn_mask.to(device=device),
                float('-inf'),
            )
        return attn_mask.to(device=device, dtype=dtype)

    def _key_padding_mask_to_additive(
        self,
        key_padding_mask: Optional[Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Tensor]:
        if key_padding_mask is None:
            return None
        if key_padding_mask.dtype == torch.bool:
            mask = key_padding_mask[:, None, None, :].to(device=device)
            return torch.zeros(mask.shape, device=device, dtype=dtype).masked_fill(mask, float('-inf'))
        return key_padding_mask.to(device=device, dtype=dtype)[:, None, None, :]

    def _merged_sdpa_mask(
        self,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        *,
        batch_size: int,
        tgt_len: int,
        src_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Tensor]:
        additive_mask = self._attention_mask_to_additive(
            attn_mask,
            batch_size=batch_size,
            tgt_len=tgt_len,
            src_len=src_len,
            device=device,
            dtype=dtype,
        )
        padding_mask = self._key_padding_mask_to_additive(
            key_padding_mask,
            device=device,
            dtype=dtype,
        )
        if additive_mask is None:
            return padding_mask
        if padding_mask is None:
            return additive_mask
        return additive_mask + padding_mask

    def _scaled_dot_product_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        additive_mask = self._merged_sdpa_mask(
            attn_mask,
            key_padding_mask,
            batch_size=q.shape[0],
            tgt_len=q.shape[2],
            src_len=k.shape[2],
            device=q.device,
            dtype=q.dtype,
        )
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=additive_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
            scale=self.scaling,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        average_attn_weights: bool = True,
        rope: Optional[Tuple[Tensor, Tensor]] = None,
    ):
        # rope: temporal RoPE ``(cos, sin)``, each [batch, T, head_dim // 2]
        # (one table per attention row, see temporal_rope_tables), rotating q / k.
        tgt_len, batch_size, _ = query.shape
        src_len = key.shape[0]

        q_weight, k_weight, v_weight = self.in_proj_weight.chunk(3, dim=0)
        if self.in_proj_bias is None:
            q_bias = k_bias = v_bias = None
        else:
            q_bias, k_bias, v_bias = self.in_proj_bias.chunk(3, dim=0)

        # Under an outer torch.autocast(bf16) context these linears/matmuls run in
        # bf16 automatically; the explicit .float() casts below keep the softmax
        # and score scaling in fp32 for numerical stability (autocast also keeps
        # softmax fp32 by policy, but we make it explicit).
        q = F.linear(query, q_weight, q_bias)
        k = F.linear(key, k_weight, k_bias)
        v = F.linear(value, v_weight, v_bias)

        q = q.transpose(0, 1).reshape(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.transpose(0, 1).reshape(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.transpose(0, 1).reshape(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        # QK-norm: bound per-head q/k magnitude so logits can't run away.
        q = self.q_norm(q)
        k = self.k_norm(k)
        if rope is not None:
            # Self-attention over one window: query and key share frame indices.
            if tgt_len != src_len:
                raise ValueError(f"temporal RoPE needs tgt_len == src_len, got {tgt_len} and {src_len}")
            cos, sin = (table.unsqueeze(1) for table in rope)
            q = apply_temporal_rope(q, cos, sin)
            k = apply_temporal_rope(k, cos, sin)

        attn_weights_fp32 = None
        if need_weights:
            scores = torch.matmul(q, k.transpose(-2, -1))
            scores = scores.float() * self.scaling
            scores = self._apply_attention_mask(scores, attn_mask)
            scores = self._apply_key_padding_mask(scores, key_padding_mask)
            attn_weights_fp32 = torch.softmax(scores, dim=-1)
            attn_weights_fp32 = F.dropout(attn_weights_fp32, p=self.dropout, training=self.training)

            attn_weights = attn_weights_fp32.to(dtype=v.dtype)
            attn_output = torch.matmul(attn_weights, v)
        else:
            attn_output = self._scaled_dot_product_attention(q, k, v, attn_mask, key_padding_mask)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, tgt_len, self.embed_dim)
        attn_output = attn_output.transpose(0, 1)
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias).float()

        if not need_weights:
            return attn_output, None
        if average_attn_weights:
            return attn_output, attn_weights_fp32.mean(dim=1)
        return attn_output, attn_weights_fp32


def circular_phase_embedding(
    length: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    cycles: Optional[Tensor] = None,
) -> Tensor:
    """Sinusoidal time table closed over the window.

    Without ``cycles`` the shape is ``(length, dim)`` and frame ``t`` sits at
    phase ``2*pi*f*t / length``. With ``cycles`` (``[B]``, the loop window's
    cycle count ``k``) the shape is ``(length, B, dim)`` and the period is
    ``length / k``: phase ``2*pi*f*k*t / length``. A k-tiled loop window holds
    each copy of the cycle in exactly ``length / k`` frames (the loader's
    periodic window resample), so copies that land on the same frames get the
    same rows.

    The frame AFTER the last one is in phase with the first. That is the
    loader's convention: a loop window holds no closing key
    (``_drop_loop_closing_frame``) and is resampled periodically
    (``resample_motion_features(periodic=True)``), so its last frame is one
    ordinary step before frame 0. A period of ``(length - 1) / k`` gave the
    first and last frame the same phase and generated loops whose last frame
    copied the first -- a one-frame stall at every seam.

    The phase is reduced modulo the window on integers (``f * k * t mod
    length``) before the trig, so closure is exact and every ``k`` reads rows
    of the ``k = 1`` table: a permutation of it when ``gcd(k, length) = 1``,
    otherwise ``length / gcd`` of its rows each repeated ``gcd`` times.
    ``cycles`` must hold integers; a fractional value is truncated.

    For a loop sample this is the only frame-position signal: MotionDiffusion adds it
    at the input in place of the absolute frame PE, and every decoder layer
    adds it again (``GraphMotionDecoderLayer.temporal_phase_scale``).

    Frequencies run over ``1..dim // 2`` unconditionally, so at production
    shapes (``length < dim // 2``) the table is redundant: ``f`` and
    ``f + length`` coincide at integer ``t``, multiples of ``length`` are
    constant ``(1, 0)``, and ``f`` / ``length - f`` share a cosine with negated
    sine. Wasted capacity, not a conflict, so it is left alone: clamping to
    Nyquist would only swap them for zero channels.
    """
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    half_dim = dim // 2
    batched = cycles is not None
    if cycles is None:
        cycles = torch.ones(1, device=device, dtype=torch.long)
    cycles = torch.as_tensor(cycles, device=device).reshape(-1).to(torch.long)
    if half_dim == 0:
        table = torch.zeros((length, cycles.numel(), dim), device=device, dtype=dtype)
        return table if batched else table[:, 0]

    frame_positions = torch.arange(length, device=device, dtype=torch.long)
    frequencies = torch.arange(1, half_dim + 1, device=device, dtype=torch.long)
    # [T, B]: the frame's position inside the k-cycle window, folded back to
    # one period of the table.
    position = torch.remainder(frame_positions[:, None] * cycles[None, :], length)
    residue = torch.remainder(position[..., None] * frequencies, length)  # [T, B, F]
    phase = residue.to(torch.float32) * (2.0 * math.pi / float(length))
    motion_emb = torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1).to(dtype)
    if motion_emb.shape[-1] < dim:
        motion_emb = F.pad(motion_emb, (0, dim - motion_emb.shape[-1]))
    return motion_emb if batched else motion_emb[:, 0]


class GraphMultiHeadAttention(nn.Module):
    def __init__(self, d_model, dropout, nheads):
        super().__init__()

        self.nheads = nheads

        self.att_size = att_size = d_model // nheads
        self.scale = att_size ** -0.5
        self.q_norm = QKNorm(att_size)
        self.k_norm = QKNorm(att_size)

        self.linear_q = nn.Linear(d_model, nheads * att_size)
        self.linear_k = nn.Linear(d_model, nheads * att_size)
        self.linear_v = nn.Linear(d_model, nheads * att_size)
        self.dropout = nn.Dropout(dropout)

        self.output_layer = nn.Linear(nheads * att_size, d_model)

    def _prepare_pairwise_index(
        self,
        pairwise: Tensor,
        *,
        batch_size: int,
        sequence_length: int,
    ) -> tuple[Tensor, int, int]:
        if pairwise.dim() == 3:
            pairwise = pairwise.unsqueeze(1)
        elif pairwise.dim() != 4:
            raise ValueError(f"Unsupported pairwise index shape: {tuple(pairwise.shape)}")
        if pairwise.shape[-2:] != (sequence_length, sequence_length):
            raise ValueError(
                f"Pairwise index shape {tuple(pairwise.shape)} does not match sequence length {sequence_length}"
            )

        base_batch = pairwise.shape[0]
        if pairwise.shape[1] == 1:
            pairwise = pairwise.expand(-1, self.nheads, -1, -1)
        elif pairwise.shape[1] != self.nheads:
            raise ValueError(
                f"Pairwise index head dimension {pairwise.shape[1]} does not match num_heads {self.nheads}"
            )
        if batch_size % base_batch != 0:
            raise ValueError(
                f"Expanded batch size {batch_size} is not divisible by pairwise batch {base_batch}"
            )

        frames = batch_size // base_batch
        return pairwise.unsqueeze(0).expand(frames, -1, -1, -1, -1), frames, base_batch

    @staticmethod
    def _reshape_pairwise_tensor(tensor: Tensor, *, frames: int, base_batch: int) -> Tensor:
        return tensor.reshape(frames, base_batch, *tensor.shape[1:])

    def _apply_pairwise_mask(self, scores: Tensor, mask: Optional[Tensor]) -> Tensor:
        if mask is None:
            return scores
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        elif mask.dim() != 4:
            raise ValueError(f"Unsupported pairwise mask shape: {tuple(mask.shape)}")

        batch_size, _, tgt_len, src_len = scores.shape
        if mask.shape[-2:] != (tgt_len, src_len):
            raise ValueError(
                f"Pairwise mask shape {tuple(mask.shape)} does not match attention shape {(tgt_len, src_len)}"
            )

        base_batch = mask.shape[0]
        if mask.shape[1] == 1:
            mask = mask.expand(-1, self.nheads, -1, -1)
        elif mask.shape[1] != self.nheads:
            raise ValueError(
                f"Pairwise mask head dimension {mask.shape[1]} does not match num_heads {self.nheads}"
            )
        if batch_size % base_batch != 0:
            raise ValueError(
                f"Expanded batch size {batch_size} is not divisible by mask batch {base_batch}"
            )

        if base_batch == batch_size:
            if mask.dtype == torch.bool:
                return scores.masked_fill(mask.to(device=scores.device), float('-inf'))
            return scores + mask.to(device=scores.device, dtype=torch.float32)

        frames = batch_size // base_batch
        scores = scores.reshape(frames, base_batch, self.nheads, tgt_len, src_len)
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(mask.to(device=scores.device).unsqueeze(0), float('-inf'))
        else:
            scores = scores + mask.to(device=scores.device, dtype=torch.float32).unsqueeze(0)
        return scores.reshape(batch_size, self.nheads, tgt_len, src_len)

    def _pairwise_mask_to_additive(
        self,
        mask: Optional[Tensor],
        *,
        batch_size: int,
        tgt_len: int,
        src_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Tensor]:
        if mask is None:
            return None
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        elif mask.dim() != 4:
            raise ValueError(f"Unsupported pairwise mask shape: {tuple(mask.shape)}")

        if mask.shape[-2:] != (tgt_len, src_len):
            raise ValueError(
                f"Pairwise mask shape {tuple(mask.shape)} does not match attention shape {(tgt_len, src_len)}"
            )

        base_batch = mask.shape[0]
        if mask.shape[1] == 1:
            mask = mask.expand(-1, self.nheads, -1, -1)
        elif mask.shape[1] != self.nheads:
            raise ValueError(
                f"Pairwise mask head dimension {mask.shape[1]} does not match num_heads {self.nheads}"
            )
        if batch_size % base_batch != 0:
            raise ValueError(
                f"Expanded batch size {batch_size} is not divisible by mask batch {base_batch}"
            )

        frames = batch_size // base_batch
        if frames > 1:
            mask = mask.unsqueeze(0).expand(frames, -1, -1, -1, -1).reshape(
                batch_size, self.nheads, tgt_len, src_len
            )

        if mask.dtype == torch.bool:
            return torch.zeros(mask.shape, device=device, dtype=dtype).masked_fill(
                mask.to(device=device),
                float('-inf'),
            )
        return mask.to(device=device, dtype=dtype)

    def _key_padding_mask_to_additive(
        self,
        key_padding_mask: Optional[Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Tensor]:
        if key_padding_mask is None:
            return None
        if key_padding_mask.dtype == torch.bool:
            mask = key_padding_mask[:, None, None, :].to(device=device)
            return torch.zeros(mask.shape, device=device, dtype=dtype).masked_fill(mask, float('-inf'))
        return key_padding_mask.to(device=device, dtype=dtype)[:, None, None, :]

    def _merged_sdpa_mask(
        self,
        graph_bias: Tensor,
        pairwise_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        *,
        batch_size: int,
        tgt_len: int,
        src_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        additive_mask = graph_bias.to(device=device, dtype=dtype) * self.scale
        mask_bias = self._pairwise_mask_to_additive(
            pairwise_mask,
            batch_size=batch_size,
            tgt_len=tgt_len,
            src_len=src_len,
            device=device,
            dtype=dtype,
        )
        if mask_bias is not None:
            additive_mask = additive_mask + mask_bias
        padding_bias = self._key_padding_mask_to_additive(
            key_padding_mask,
            device=device,
            dtype=dtype,
        )
        if padding_bias is not None:
            additive_mask = additive_mask + padding_bias
        return additive_mask

    def _merged_plain_mask(self, mask, key_padding_mask, q):
        batch_size, _, length, _ = q.shape
        additive = self._key_padding_mask_to_additive(key_padding_mask, device=q.device, dtype=q.dtype)
        mask_bias = self._pairwise_mask_to_additive(
            mask, batch_size=batch_size, tgt_len=length, src_len=length,
            device=q.device, dtype=q.dtype,
        )
        if mask_bias is not None:
            additive = mask_bias if additive is None else additive + mask_bias
        return additive

    def _graph_scaled_dot_product_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        graph_bias: Tensor,
        mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        additive_mask = self._merged_sdpa_mask(
            graph_bias,
            mask,
            key_padding_mask,
            batch_size=q.shape[0],
            tgt_len=q.shape[2],
            src_len=k.shape[2],
            device=q.device,
            dtype=q.dtype,
        )
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=additive_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
            scale=self.scale,
        )

    def forward(
        self,
        q,
        k,
        v,
        query_hop_emb,
        query_edge_emb,
        key_hop_emb,
        key_edge_emb,
        value_hop_emb,
        value_edge_emb,
        distance,
        edge_attr,
        mask=None,
        key_padding_mask: Optional[Tensor] = None,
    ):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)


        q = self.linear_q(q).view(batch_size, -1, self.nheads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.nheads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.nheads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2)  # [b, h, k_len, d_k]

        # QK-norm: bound per-head q/k magnitude before they feed both the
        # content dot-product and the GRPE hop/edge bias terms, so attention
        # logits can't run away into softmax saturation.
        q = self.q_norm(q)
        k = self.k_norm(k)

        if distance is None:
            # No graph relations: plain attention, so SDPA can take its fused
            # kernels (no [B, H, J, J] bias to gather and add).
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=self._merged_plain_mask(mask, key_padding_mask, q),
                dropout_p=self.dropout.p if self.training else 0.0,
                scale=self.scale,
            )
            x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.nheads * d_v)
            return self.output_layer(x).float()

        sequence_length = v.shape[2]
        use_sdpa = value_hop_emb is None and value_edge_emb is None
        num_hop_types = query_hop_emb.shape[0]
        num_edge_types = query_edge_emb.shape[0]
        distance_index, relation_frames, relation_batch = self._prepare_pairwise_index(
            distance,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        edge_index, edge_frames, edge_batch = self._prepare_pairwise_index(
            edge_attr,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        if (edge_frames, edge_batch) != (relation_frames, relation_batch):
            raise ValueError(
                "distance and edge_attr must expand to the same (frames, batch) layout"
            )

        query_hop_emb = query_hop_emb.view(
            1, num_hop_types, self.nheads, self.att_size
        ).transpose(1, 2)
        query_edge_emb = query_edge_emb.view(
            1, -1, self.nheads, self.att_size
        ).transpose(1, 2)
        key_hop_emb = key_hop_emb.view(
            1, num_hop_types, self.nheads, self.att_size
        ).transpose(1, 2)
        key_edge_emb = key_edge_emb.view(
            1, num_edge_types, self.nheads, self.att_size
        ).transpose(1, 2)

        # Under an outer torch.autocast(bf16) context these matmuls run in bf16;
        # graph_bias is accumulated in fp32 below to avoid catastrophic
        # cancellation from summing the bf16 terms.
        query_hop = torch.matmul(q, query_hop_emb.transpose(2, 3))
        query_hop = self._reshape_pairwise_tensor(query_hop, frames=relation_frames, base_batch=relation_batch)
        query_hop = torch.gather(query_hop, 4, distance_index).reshape(
            batch_size, self.nheads, sequence_length, sequence_length
        )
        query_edge = torch.matmul(q, query_edge_emb.transpose(2, 3))
        query_edge = self._reshape_pairwise_tensor(query_edge, frames=edge_frames, base_batch=edge_batch)
        query_edge = torch.gather(query_edge, 4, edge_index).reshape(
            batch_size, self.nheads, sequence_length, sequence_length
        )

        key_hop = torch.matmul(k, key_hop_emb.transpose(2, 3))
        key_hop = self._reshape_pairwise_tensor(key_hop, frames=relation_frames, base_batch=relation_batch)
        key_hop = torch.gather(key_hop, 4, distance_index).reshape(
            batch_size, self.nheads, sequence_length, sequence_length
        )
        key_edge = torch.matmul(k, key_edge_emb.transpose(2, 3))
        key_edge = self._reshape_pairwise_tensor(key_edge, frames=edge_frames, base_batch=edge_batch)
        key_edge = torch.gather(key_edge, 4, edge_index).reshape(
            batch_size, self.nheads, sequence_length, sequence_length
        )
        if not use_sdpa:
            qk = torch.matmul(q, k.transpose(2, 3))

        graph_bias = query_hop.float() + key_hop.float() + query_edge.float() + key_edge.float()

        if use_sdpa:
            x = self._graph_scaled_dot_product_attention(
                q,
                k,
                v,
                graph_bias=graph_bias,
                mask=mask,
                key_padding_mask=key_padding_mask,
            )
        else:
            # Accumulate in fp32 to prevent catastrophic cancellation from summing bf16 terms.
            x = (qk.float() + graph_bias) * self.scale

            if mask is not None:
                x = self._apply_pairwise_mask(x, mask)
            if key_padding_mask is not None:
                if key_padding_mask.dtype == torch.bool:
                    x = x.masked_fill(key_padding_mask[:, None, None, :].to(device=x.device), float('-inf'))
                else:
                    x = x + key_padding_mask.to(device=x.device, dtype=torch.float32)[:, None, None, :]

            x = torch.softmax(x.float(), dim=3)
            x = self.dropout(x)
            value_hop_emb = value_hop_emb.view(
                1, num_hop_types, self.nheads, self.att_size
            ).transpose(1, 2)
            value_edge_emb = value_edge_emb.view(
                1, num_edge_types, self.nheads, self.att_size
            ).transpose(1, 2)

            value_hop_att = torch.zeros(
                (relation_frames, relation_batch, self.nheads, sequence_length, num_hop_types),
                device=value_hop_emb.device,
                dtype=x.dtype,
            )
            x_for_scatter = self._reshape_pairwise_tensor(x, frames=relation_frames, base_batch=relation_batch)
            value_hop_att = torch.scatter_add(
                value_hop_att, 4, distance_index, x_for_scatter
            ).reshape(batch_size, self.nheads, sequence_length, num_hop_types)
            value_edge_att = torch.zeros(
                (edge_frames, edge_batch, self.nheads, sequence_length, num_edge_types),
                device=value_hop_emb.device,
                dtype=x.dtype,
            )
            value_edge_att = torch.scatter_add(
                value_edge_att, 4, edge_index, x_for_scatter
            ).reshape(batch_size, self.nheads, sequence_length, num_edge_types)
            x = x.to(dtype=v.dtype)
            x = torch.matmul(x, v)
            x = x + torch.matmul(value_hop_att, value_hop_emb) + torch.matmul(value_edge_att, value_edge_emb)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.nheads * d_v)

        x = self.output_layer(x).float()
        assert x.size() == orig_q_size
        return x

# The two branch inputs the action AdaLN modulates, each with a scale and a
# shift: the temporal block's input and the feed-forward block's input
# (docs/conditional_modulation_upgrade.md section 3.2). The SPATIAL branch is
# deliberately left alone -- its input already carries embed_timesteps' additive
# offset, so a second injection there would be redundant.
ACTION_ADALN_BRANCHES = 2
ACTION_ADALN_PARAMS_PER_LAYER = 2 * ACTION_ADALN_BRANCHES
ADALN_TEMPORAL, ADALN_FF = 0, 1


def _modulate(x: Tensor, adaln: Optional[Tensor], branch: int) -> Tensor:
    """Scale and shift ONE branch's input: ``(1 + gamma) * x + beta``.

    ``adaln`` is this layer's (B, 4, d_model) slice of the action head's
    output, or None when the run has no action AdaLN (then this is the
    identity and costs nothing). ``x`` is
    (frames, B, njoints, d_model), so the modulation broadcasts over frames and
    joints: a per-sample, per-channel gain, which is the authority the purely
    additive condition token lacks.

    This feeds the BRANCH only. The residual stream stays the unmodulated
    post-norm activation, so the identity path through the stack is untouched
    and the gains cannot compound layer over layer.

    The heads are zero-initialised, so gamma = 0 and beta = 0 on a fresh model and
    this reproduces the unmodulated layer exactly.
    """
    if adaln is None:
        return x
    gamma = adaln[:, 2 * branch].to(x.dtype).view(1, -1, 1, x.shape[-1])
    beta = adaln[:, 2 * branch + 1].to(x.dtype).view(1, -1, 1, x.shape[-1])
    return x * (1.0 + gamma) + beta


class GraphMotionDecoder(nn.TransformerDecoder):
    def __init__(self, decoder_layer, num_layers, norm=None,
                 num_topology_codes=NUM_TOPOLOGY_CODES, num_edge_codes=NUM_EDGE_EMBEDDINGS,
                 value_emb=False, action_label_adaln=False,
                 graph_relations=True):
                # multi head attention
        super().__init__(decoder_layer, num_layers, norm)
        # graph_relations=False: spatial attention without the topology / edge
        # bias and without relation tables.
        self.graph_relations = bool(graph_relations)
        if value_emb and not self.graph_relations:
            raise ValueError('value_emb needs graph_relations')

        self.d_model = decoder_layer.d_model
        self.nheads = decoder_layer.heads
        # Table sizes come from ``topology_relations`` rather than being written
        # here: the dataset emits those indices, and a mismatch surfaces only as an
        # out-of-bounds gather deep inside the attention bias.
        if self.graph_relations:
            self.topology_key_emb = nn.Embedding(num_topology_codes, self.d_model)
            self.edge_key_emb = nn.Embedding(num_edge_codes, self.d_model)
            self.topology_query_emb = nn.Embedding(num_topology_codes, self.d_model)
            self.edge_query_emb = nn.Embedding(num_edge_codes, self.d_model)
        else:
            self.topology_key_emb = self.edge_key_emb = None
            self.topology_query_emb = self.edge_query_emb = None
        self.value_emb_flag = value_emb
        if value_emb:
            self.topology_value_emb = nn.Embedding(num_topology_codes, self.d_model)
            self.edge_value_emb = nn.Embedding(num_edge_codes, self.d_model)

        # Action AdaLN: the action condition gets a MULTIPLICATIVE
        # pathway on top of the additive token summed into timesteps_emb
        # (docs/conditional_modulation_upgrade.md section 3.2). The additive one
        # can only translate the residual stream, and every sublayer output is
        # LayerNormed right after its residual add, which pushes a uniform
        # translation back down; a per-channel gain is what lets the label
        # amplify a small input difference (two labels sharing a slot sit at
        # cosine ~0.84 by construction, e.g. 'turn, left' against
        # 'run, turn, left') into a large behavioural one, instead of leaving
        # that entirely to --action_label_cfg_scale and its quality cost.
        #
        # Allocated after the decoder's other parameters so it does not reorder
        # their initialisation.
        #
        # ONE head for every layer, not one per layer: same parameter count
        # (d^2 + 4*L*d^2 against L*(d^2 + 4*d^2)) for one matmul instead of 2L,
        # which matters on a step that is partly kernel-launch bound.
        #
        # The conditioning head uses the same width as the decoder.
        if action_label_adaln:
            self.action_adaln = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.SiLU(),
                nn.Linear(self.d_model, num_layers * ACTION_ADALN_PARAMS_PER_LAYER * self.d_model),
            )
            # Zero-init: gamma = beta = 0 on a fresh model, so training starts
            # from exactly the unmodulated architecture and the head only ever
            # has to learn a departure from it. No residual gate.
            nn.init.zeros_(self.action_adaln[-1].weight)
            nn.init.zeros_(self.action_adaln[-1].bias)
        else:
            self.action_adaln = None
        # Every layer's temporal attention (the replaced last layer included)
        # rotates q / k by frame index; parameter-free.

    def _resolve_edge_codes(self, edge_rel: Tensor) -> Tensor:
        """Resolve stored relation codes without hiding symmetry or laterality."""
        return resolve_edge_codes(edge_rel)

    def _expand_relation_heads(self, relation: Tensor) -> Tensor:
        if relation.dim() == 3:
            return relation.unsqueeze(1).expand(-1, self.nheads, -1, -1)
        if relation.dim() == 4:
            if relation.shape[1] == 1:
                return relation.expand(-1, self.nheads, -1, -1)
            if relation.shape[1] == self.nheads:
                return relation
        raise ValueError(f"Unsupported graph relation shape: {tuple(relation.shape)}")
        

        
    def forward(self, tgt: Tensor, timesteps_embs: Tensor, memory: Tensor, spatial_mask:  Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None, y=None,
            loop_phase_mask: Optional[Tensor] = None,
            loop_cycles: Optional[Tensor] = None,
            action_adaln_cond: Optional[Tensor] = None,
            temporal_key_padding_mask: Optional[Tensor] = None,
            motion_z: Optional[Tensor] = None,
            temporal_rope_override: Optional[Tuple[Tensor, Tensor]] = None,
            loop_phase_override: Optional[Tensor] = None) -> Union[Tensor , Tuple[Tensor, dict]]:
        # temporal_key_padding_mask: (B, T) bool, True = no frame may attend to
        # this frame in temporal attention (the frame still queries).
        topology_rel = edge_rel = None
        if self.graph_relations:
            topology_rel = self._expand_relation_heads(y['graph_dist'].to(device=tgt.device, dtype=torch.long))
            edge_rel = self._resolve_edge_codes(y['joints_relations'].to(device=tgt.device, dtype=torch.long))
            edge_rel = self._expand_relation_heads(edge_rel)
        output = tgt
        T, B = tgt.shape[0], tgt.shape[1]
        # The loop phase table is built once here and shared by every layer:
        # the circular phase added before temporal attention, zeroed for
        # non-loop samples. Its period is T / loop_cycles ([B], the window's
        # cycle count) -- the same table MotionDiffusion put in
        # `tgt` for loop samples in place of the absolute PE.
        loop_phase_embedding = None
        if loop_phase_mask is not None:
            loop_phase_mask = torch.as_tensor(loop_phase_mask, device=tgt.device, dtype=torch.bool).reshape(-1)
            if loop_phase_mask.numel() == 1 and B != 1:
                loop_phase_mask = loop_phase_mask.expand(B)
            elif loop_phase_mask.numel() != B:
                raise ValueError(
                    "loop_phase_mask batch dimension must match the motion batch size, got "
                    f"{loop_phase_mask.numel()} for batch {B}"
                )
            # Always the mask-zeroed table: an all-False fast path would read
            # the mask back (a stream sync, and a graph break under
            # torch.compile), and all-False zeroes the phase anyway.
            if loop_cycles is None:
                table = circular_phase_embedding(T, self.d_model, tgt.device, tgt.dtype).unsqueeze(1)
            else:
                table = circular_phase_embedding(
                    T, self.d_model, tgt.device, tgt.dtype, cycles=loop_cycles,
                )
            loop_phase_embedding = table * loop_phase_mask.view(1, B, 1)
        # Temporal RoPE tables, [B, T, head_dim // 2] each: the closed-window
        # table on loop samples, the open one elsewhere.
        if loop_phase_override is not None:
            loop_phase_embedding = loop_phase_override
        temporal_rope = temporal_rope_override
        if temporal_rope is None:
            head_dim = self.layers[0].temporal_attn.head_dim
            cos, sin = temporal_rope_tables(T, head_dim, tgt.device)
            cos, sin = cos.expand(B, -1, -1), sin.expand(B, -1, -1)
            if loop_phase_mask is not None:
                loop_cos, loop_sin = temporal_rope_tables(T, head_dim, tgt.device, loop=True)
                is_loop = loop_phase_mask.view(B, 1, 1)
                cos = torch.where(is_loop, loop_cos, cos)
                sin = torch.where(is_loop, loop_sin, sin)
            temporal_rope = (cos, sin)
        # (B, L, 4, d_model), built once and sliced per layer. The driver is the
        # SAME token the additive path uses -- one Bernoulli draw, one
        # action_repr -- which is what keeps classifier-free guidance honest: a
        # hard-dropped row carries action_label_null_emb, so its modulation is
        # the null token's modulation rather than a bypass of the head (the
        # failure mode of the retired global_energy null).
        adaln = None
        if self.action_adaln is not None and action_adaln_cond is not None:
            adaln = run_in_fp32(self.action_adaln, action_adaln_cond).view(
                B, self.num_layers, ACTION_ADALN_PARAMS_PER_LAYER, self.d_model
            )
        for layer_ind, mod in enumerate(self.layers):
            edge_value_emb = None
            topology_value_emb = None
            if self.value_emb_flag:
                edge_value_emb = self.edge_value_emb
                topology_value_emb = self.topology_value_emb
            output = mod(
                    output, timesteps_embs, topology_rel, edge_rel, self.edge_key_emb, self.edge_query_emb, edge_value_emb, self.topology_key_emb, self.topology_query_emb, topology_value_emb, spatial_mask,
                    tgt_key_padding_mask, memory_key_padding_mask, y,
                    loop_phase_embedding=loop_phase_embedding,
                    adaln=None if adaln is None else adaln[:, layer_ind],
                    temporal_key_padding_mask=temporal_key_padding_mask,
                    temporal_rope=temporal_rope, motion_z=motion_z)
        if self.norm is not None:
            output = self.norm(output)
        return output

class GraphMotionDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)
        # nn.TransformerDecoderLayer.__init__ allocates self_attn and
        # multihead_attn (each ~4*d_model^2 params). This layer overrides both
        # with spatial_attn / temporal_attn and never calls the base
        # sublayers, so those two modules are pure dead weight that would
        # still be saved into every checkpoint -- drop them.
        del self.self_attn
        del self.multihead_attn
        self.d_model= d_model
        self.heads = nhead
        self.spatial_attn = GraphMultiHeadAttention(d_model = d_model, nheads = nhead, dropout=dropout)
        self.temporal_attn = SelectiveMultiheadAttention(self.d_model, nhead, dropout=dropout)
        self.embed_timesteps = nn.Linear(d_model, d_model)
        # The action AdaLN head is owned by GraphMotionDecoder: one head emits every layer's
        # modulation in a single matmul, and each layer is handed its own
        # (B, 4, d_model) slice.
        self.temporal_phase_scale = nn.Parameter(torch.zeros(1))

    def enable_motion_latent(self, latent_dim):
        """AE-only, frame-aligned latent reader; each decoder layer owns its weights."""
        self.motion_latent_norm = nn.LayerNorm(self.d_model)
        self.motion_latent_attn = nn.MultiheadAttention(
            self.d_model, self.heads, dropout=self.dropout1.p, batch_first=True,
            kdim=latent_dim, vdim=latent_dim)

    def _read_motion_latent(self, x, z):
        frames, batch, joints, width = x.shape
        if z.shape[:2] != (batch, frames):
            raise ValueError('Motion latent and output windows must be frame aligned')
        query = self.motion_latent_norm(x).reshape(frames * batch, joints, width)
        values = z.transpose(0, 1).reshape(frames * batch, z.shape[2], z.shape[3])
        read, _ = self.motion_latent_attn(query, values, values, need_weights=False)
        return x + read.reshape_as(x)

    # spatial attention block
    def _spatial_mha_block(self, x: Tensor, topology_rel: Optional[Tensor], edge_rel: Optional[Tensor], edge_key_emb, edge_query_emb, edge_value_emb,
        topology_key_emb, topology_query_emb, topology_value_emb, attn_mask: Optional[Tensor],  key_padding_mask: Optional[Tensor], y = None) -> Tensor:
        #x.shape (frames, bs, njoints, feature_len)
        frames, bs, njoints, feature_len = x.shape
        x = x.reshape(frames * bs, njoints, feature_len)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0).expand(frames, bs, njoints).reshape(-1, njoints)
        
        def weight(emb):
            return None if emb is None else emb.weight
        attn_output = self.spatial_attn(x, x, x, weight(topology_query_emb), weight(edge_query_emb), weight(topology_key_emb), weight(edge_key_emb), weight(topology_value_emb),
        weight(edge_value_emb), topology_rel, edge_rel, attn_mask, key_padding_mask=key_padding_mask)
        attn_output = attn_output.reshape(frames, bs, njoints, feature_len) # njoints, bs, frames, feature_len
        return self.dropout1(attn_output)
    
    
        # temporal attention block
    def _temporal_mha_block_sin_joint(self, x: Tensor, key_padding_mask: Optional[Tensor], loop_phase_embedding: Optional[Tensor] = None,
                                      temporal_rope: Optional[Tuple[Tensor, Tensor]] = None) -> Tensor:
        frames, bs, njoints, feats= x.size()
        if loop_phase_embedding is not None:
            if loop_phase_embedding.shape != (frames, bs, feats):
                raise ValueError(
                    "loop_phase_embedding must have shape "
                    f"{(frames, bs, feats)}, got {tuple(loop_phase_embedding.shape)}"
                )
            # The same table MotionDiffusion already put in x for loop samples, re-added
            # per layer; zero-init scale, so a fresh model starts at the input
            # table alone.
            x = x + self.temporal_phase_scale * loop_phase_embedding.unsqueeze(2)
        # Every frame token attends over the whole window, except frames flagged
        # in key_padding_mask ((B * J, T)).
        x = x.view(frames, bs * njoints, feats)
        rope = None
        if temporal_rope is not None:
            # (B, T, P) -> (B * J, T, P), the row order of the temporal view.
            rope = tuple(table.repeat_interleave(njoints, dim=0) for table in temporal_rope)
        output_attn, _ = self.temporal_attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            rope=rope,
        )
        output_attn = output_attn.view(frames, bs ,njoints, feats)
        return self.dropout2(output_attn)
    
    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

    def forward(self,
        tgt: Tensor,
        timesteps_emb: Tensor,
        topology_rel: Tensor,
        edge_rel: Tensor,
        edge_key_emb,
        edge_query_emb,
        edge_value_emb,
        topo_key_emb,
        topo_query_emb,
        topo_value_emb,
        spatial_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None, #for future use
        y = None,
        loop_phase_embedding: Optional[Tensor] = None,
        adaln: Optional[Tensor] = None,
        temporal_key_padding_mask: Optional[Tensor] = None,
        temporal_rope: Optional[Tuple[Tensor, Tensor]] = None,
        motion_z: Optional[Tensor] = None) -> Tensor:
        x = tgt #(frames, bs, njoints, feature_len)
        bs = x.shape[1]
        x = x + self.embed_timesteps(timesteps_emb).view(1, bs, 1, self.d_model)
        spatial_attn_output = self._spatial_mha_block(x, topology_rel, edge_rel, edge_key_emb, edge_query_emb, edge_value_emb,
        topo_key_emb, topo_query_emb, topo_value_emb, spatial_mask, tgt_key_padding_mask, y)
        x = self.norm1(x + spatial_attn_output)
        if motion_z is not None:
            x = self._read_motion_latent(x, motion_z)
        # Branch input only; the residual adds to the unmodulated x. The
        # modulation lands BEFORE the circular phase, which _temporal_mha_block
        # sums in, so a loop sample's phase is never scaled by the label.
        temporal_padding = None
        if temporal_key_padding_mask is not None:
            # (B, T) -> (B * J, T), the row order of the temporal view.
            temporal_padding = temporal_key_padding_mask.to(device=x.device, dtype=torch.bool) \
                .repeat_interleave(x.shape[2], dim=0)
        x = self.norm2(x + self._temporal_mha_block_sin_joint(
            _modulate(x, adaln, ADALN_TEMPORAL), temporal_padding,
            loop_phase_embedding=loop_phase_embedding, temporal_rope=temporal_rope))
        x = self.norm3(x + self._ff_block(_modulate(x, adaln, ADALN_FF)))
        return x
