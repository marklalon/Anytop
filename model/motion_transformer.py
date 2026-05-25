import math
import torch 
import torch.nn as nn
from typing import Optional, Union, Callable, Tuple
from torch import Tensor
import torch.nn.functional as F
CUDA_LAUNCH_BLOCKING=1


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
        self.autocast_dtype: torch.dtype | None = None
        self.autocast_device_type = 'cuda'
        self.use_selective_bf16 = False

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

    def configure_precision(self, *, device_type: str, autocast_dtype: torch.dtype | None) -> bool:
        self.autocast_device_type = device_type
        self.autocast_dtype = autocast_dtype
        self.use_selective_bf16 = autocast_dtype == torch.bfloat16
        return True

    def _bf16_context(self, reference_tensor: Tensor):
        device_type = reference_tensor.device.type if torch.is_tensor(reference_tensor) else self.autocast_device_type
        if not self.use_selective_bf16:
            return torch.autocast(device_type=device_type, enabled=False)
        return torch.autocast(device_type=device_type, dtype=self.autocast_dtype)

    def _project_bf16(self, inputs: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        with self._bf16_context(inputs):
            return F.linear(inputs, weight, bias)

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

    def _softmax_fp32(self, scores: Tensor) -> Tensor:
        return torch.softmax(scores.float(), dim=-1)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        average_attn_weights: bool = True,
    ):
        tgt_len, batch_size, _ = query.shape
        src_len = key.shape[0]

        q_weight, k_weight, v_weight = self.in_proj_weight.chunk(3, dim=0)
        if self.in_proj_bias is None:
            q_bias = k_bias = v_bias = None
        else:
            q_bias, k_bias, v_bias = self.in_proj_bias.chunk(3, dim=0)

        q = self._project_bf16(query, q_weight, q_bias)
        k = self._project_bf16(key, k_weight, k_bias)
        v = self._project_bf16(value, v_weight, v_bias)

        q = q.transpose(0, 1).reshape(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.transpose(0, 1).reshape(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.transpose(0, 1).reshape(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights_fp32 = None
        if need_weights:
            with self._bf16_context(q):
                scores = torch.matmul(q, k.transpose(-2, -1))
            scores = scores.float() * self.scaling
            scores = self._apply_attention_mask(scores, attn_mask)
            scores = self._apply_key_padding_mask(scores, key_padding_mask)
            attn_weights_fp32 = self._softmax_fp32(scores)
            attn_weights_fp32 = F.dropout(attn_weights_fp32, p=self.dropout, training=self.training)

            attn_weights = attn_weights_fp32.to(dtype=v.dtype)
            with self._bf16_context(v):
                attn_output = torch.matmul(attn_weights, v)
        else:
            attn_output = self._scaled_dot_product_attention(q, k, v, attn_mask, key_padding_mask)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, tgt_len, self.embed_dim)
        attn_output = attn_output.transpose(0, 1)
        attn_output = self._project_bf16(attn_output, self.out_proj.weight, self.out_proj.bias).float()

        if not need_weights:
            return attn_output, None
        if average_attn_weights:
            return attn_output, attn_weights_fp32.mean(dim=1)
        return attn_output, attn_weights_fp32


def _sin_time_embedding(
    length: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    max_period: float = 10000.0,
) -> Tensor:
    """Return a standard sinusoidal time table with shape ``(length, dim)``."""
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")

    positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    half_dim = dim // 2
    if half_dim == 0:
        return torch.zeros((length, dim), device=device, dtype=dtype)

    scales = torch.arange(half_dim, device=device, dtype=dtype).unsqueeze(0)
    max_period_tensor = torch.full((), max_period, device=device, dtype=dtype)
    phase = positions / (max_period_tensor ** (scales / max(half_dim - 1, 1)))
    emb = torch.cat([torch.cos(phase), torch.sin(phase)], dim=1)
    if emb.shape[1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[1]))
    return emb


def _circular_phase_embedding(
    length: int,
    dim: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    emb = torch.zeros((length, batch_size, dim), device=device, dtype=dtype)
    motion_frames = max(length - 1, 0)
    half_dim = dim // 2
    if motion_frames <= 0 or half_dim == 0:
        return emb

    if lengths is None:
        lengths_t = torch.full((batch_size,), max(motion_frames - 1, 1), device=device, dtype=dtype)
    else:
        lengths_t = torch.as_tensor(lengths, device=device, dtype=dtype).reshape(-1)
        if lengths_t.numel() == 1 and batch_size != 1:
            lengths_t = lengths_t.expand(batch_size)
        elif lengths_t.numel() != batch_size:
            raise ValueError(
                "lengths batch dimension must match the motion batch size, got "
                f"{lengths_t.numel()} for batch {batch_size}"
            )
        lengths_t = (lengths_t - 1.0).clamp(min=1.0)

    frame_positions = torch.arange(motion_frames, device=device, dtype=dtype).unsqueeze(1)
    phase = (2.0 * math.pi) * frame_positions / lengths_t.unsqueeze(0)
    frequencies = torch.arange(1, half_dim + 1, device=device, dtype=dtype).view(1, 1, -1)
    phase = phase.unsqueeze(-1) * frequencies
    motion_emb = torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)
    if motion_emb.shape[-1] < dim:
        motion_emb = F.pad(motion_emb, (0, dim - motion_emb.shape[-1]))
    emb[1:] = motion_emb
    return emb


def _loop_aware_time_embedding(
    length: int,
    dim: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    loop_phase_mask: Optional[Tensor],
    lengths: Optional[Tensor],
) -> Tensor:
    absolute = _sin_time_embedding(length, dim, device, dtype).unsqueeze(1)
    if loop_phase_mask is None:
        return absolute
    loop_phase_mask = torch.as_tensor(loop_phase_mask, device=device, dtype=torch.bool).reshape(-1)
    if loop_phase_mask.numel() == 1 and batch_size != 1:
        loop_phase_mask = loop_phase_mask.expand(batch_size)
    elif loop_phase_mask.numel() != batch_size:
        raise ValueError(
            "loop_phase_mask batch dimension must match the motion batch size, got "
            f"{loop_phase_mask.numel()} for batch {batch_size}"
        )
    if not bool(loop_phase_mask.any()):
        return absolute
    absolute = absolute.expand(-1, batch_size, -1)
    circular = _circular_phase_embedding(length, dim, batch_size, device, dtype, lengths)
    return torch.where(loop_phase_mask.view(1, batch_size, 1), circular, absolute)


class CrossLimbTemporalBlock(nn.Module):
    """Perceiver-style cross-limb temporal pathway (one independent instance
    per active decoder layer; post-norm sublayer).

    The base decoder factorizes attention into spatial (per-frame, across
    joints) and temporal (per-joint, across frames). A joint's temporal
    trajectory therefore never directly attends to another joint's, so
    inter-limb frequency/phase coupling is only weak/implicit -- which is why
    an inpainted limb drifts out of sync with the clamped limbs.

    This block adds the missing path: K learned latent tokens pool ALL joints
    per frame (cross-in), attend over time with the SAME windowed temporal mask
    (latent temporal self-attention), then write a whole-body rhythm context
    back to every joint (cross-out). Reuses SelectiveMultiheadAttention; the
    final residual is post-norm to stay consistent with norm1/2/3 of the layer.

    Efficiency: the latent pathway runs at a *narrow* bottleneck width
    ``latent_width`` (Perceiver latents are meant to be a narrow information
    bottleneck, not full model width). Joints are projected d_model -> d_cl
    once (shared by cross-in K/V and cross-out Q), all three attentions run at
    d_cl, and the whole-body context is projected d_cl -> d_model on the way
    out. Cross-limb parameter cost scales with the number of active layers
    (controlled by ``cross_limb_last_n``).

    x flows as (T, B, J, d) where T = nframes+1 (index 0 is the T-pose token).
    """

    def __init__(self, d_model: int, nhead: int, num_latents: int = 8,
                 dropout: float = 0.1, latent_width: int = 64):
        super().__init__()
        self.d_model = d_model
        self.num_heads = nhead
        self.num_latents = num_latents
        # Bottleneck width: <= d_model and a multiple of nhead (>= nhead).
        d_cl = max(nhead, (min(latent_width, d_model) // nhead) * nhead)
        self.latent_dim = d_cl
        self.latents = nn.Parameter(torch.empty(num_latents, d_cl))
        nn.init.normal_(self.latents, std=0.02)
        # Shared joint <-> latent-space projection (read for cross-in K/V and
        # write-query for cross-out); identity when no bottleneck is needed.
        self.proj_in = nn.Linear(d_model, d_cl) if d_cl != d_model else nn.Identity()
        self.proj_out = nn.Linear(d_cl, d_model) if d_cl != d_model else nn.Identity()
        self.cross_in_attn = SelectiveMultiheadAttention(d_cl, nhead, dropout=dropout)
        self.temporal_attn = SelectiveMultiheadAttention(d_cl, nhead, dropout=dropout)
        self.cross_out_attn = SelectiveMultiheadAttention(d_cl, nhead, dropout=dropout)
        self.time_emb_scale = nn.Parameter(torch.zeros(1))
        self.reliability_bias = nn.Parameter(torch.zeros(1))
        self.norm_cl = nn.LayerNorm(d_model)
        self.register_buffer('_cached_time_emb', torch.empty(0), persistent=False)

    def _get_cached_time_embedding(
        self,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        cached = self._cached_time_emb
        if (
            cached.ndim == 2
            and cached.shape == (length, self.latent_dim)
            and cached.device == device
            and cached.dtype == dtype
        ):
            return cached

        cached = _sin_time_embedding(length, self.latent_dim, device, dtype)
        self._cached_time_emb = cached
        return cached

    def forward(
        self,
        x: Tensor,
        temporal_template: Tensor,
        joints_key_padding_mask: Tensor,
        unreliable_mask: Optional[Tensor] = None,
        loop_phase_mask: Optional[Tensor] = None,
        lengths: Optional[Tensor] = None,
        time_embedding: Optional[Tensor] = None,
    ) -> Tensor:
        # x: (T, B, J, d_model); temporal_template: (B*H, T, T) additive float
        # mask (windowed, NO joint repeat); joints_key_padding_mask: (B, J)
        # bool, True == padded joint.
        T, B, J, _ = x.shape
        K, H, d = self.num_latents, self.num_heads, self.latent_dim

        # Joints embedded into the narrow latent space once; reused as cross-in
        # key/value and as cross-out query.
        xd = self.proj_in(x)                                               # (T, B, J, d_cl)

        # --- Cross-in: latents (query) attend over joints (key/value), per frame.
        # Flatten so attention batch = T*B with index (t*B + b).
        kv = xd.permute(2, 0, 1, 3).reshape(J, T * B, d)                   # (J, T*B, d_cl)
        q = self.latents.unsqueeze(1).expand(K, T * B, d)                  # (K, T*B, d_cl)
        kpm = joints_key_padding_mask.unsqueeze(0).expand(T, B, J).reshape(T * B, J)  # (T*B, J)
        reliability_attn_mask = None
        if unreliable_mask is not None:
            if unreliable_mask.shape != (T, B, J):
                raise ValueError(
                    f"unreliable_mask must have shape {(T, B, J)}, got {tuple(unreliable_mask.shape)}"
                )
            reliability_attn_mask = unreliable_mask.reshape(T * B, J)
            reliability_attn_mask = reliability_attn_mask.to(device=xd.device, dtype=xd.dtype)
            reliability_attn_mask = self.reliability_bias * reliability_attn_mask
            reliability_attn_mask = reliability_attn_mask.unsqueeze(1).expand(T * B, K, J)
        bz, _ = self.cross_in_attn(
            q,
            kv,
            kv,
            attn_mask=reliability_attn_mask,
            key_padding_mask=kpm,
            need_weights=False,
        )
        bz = bz.reshape(K, T, B, d)

        # --- Latent temporal self-attention across T (same windowed mask).
        # attention batch = B*K with index (b*K + k); mask -> (B*K*H, T, T).
        zt_in = bz.permute(1, 2, 0, 3).reshape(T, B * K, d)               # (T, B*K, d_cl)
        if time_embedding is not None:
            time_embedding = time_embedding.to(device=zt_in.device, dtype=zt_in.dtype)
            if time_embedding.shape == (T, d):
                time_emb = time_embedding.unsqueeze(1)
            elif time_embedding.shape == (T, B, d):
                time_emb = time_embedding.unsqueeze(2).expand(T, B, K, d).reshape(T, B * K, d)
            else:
                raise ValueError(
                    "time_embedding must have shape "
                    f"{(T, d)} or {(T, B, d)}, got {tuple(time_embedding.shape)}"
                )
        elif loop_phase_mask is None:
            time_emb = self._get_cached_time_embedding(T, zt_in.device, zt_in.dtype).unsqueeze(1)
        else:
            time_emb = _loop_aware_time_embedding(
                T,
                d,
                B,
                zt_in.device,
                zt_in.dtype,
                loop_phase_mask,
                lengths,
            )
            time_emb = time_emb.unsqueeze(2).expand(T, B, K, d).reshape(T, B * K, d)
        zt_in = zt_in + self.time_emb_scale * time_emb
        tt = temporal_template.reshape(B, H, T, T)
        tt = tt.unsqueeze(1).expand(B, K, H, T, T).reshape(B * K * H, T, T)
        zt, _ = self.temporal_attn(zt_in, zt_in, zt_in, attn_mask=tt, need_weights=False)
        zt = zt.reshape(T, B, K, d)

        # --- Cross-out: joints (query) attend over latents (key/value), per frame.
        # Both flattened to attention batch = T*B with index (t*B + b).
        q_out = xd.permute(2, 0, 1, 3).reshape(J, T * B, d)               # (J, T*B, d_cl)
        kv_out = zt.permute(2, 0, 1, 3).reshape(K, T * B, d)              # (K, T*B, d_cl)
        delta, _ = self.cross_out_attn(q_out, kv_out, kv_out, need_weights=False)
        delta = delta.reshape(J, T, B, d).permute(1, 2, 0, 3)            # (T, B, J, d_cl)

        return self.norm_cl(x + self.proj_out(delta))


class GraphMultiHeadAttention(nn.Module):
    def __init__(self, d_model, dropout, nheads):
        super().__init__()

        self.nheads = nheads
        self.autocast_dtype: torch.dtype | None = None
        self.autocast_device_type = 'cuda'
        self.use_selective_bf16 = False

        self.att_size = att_size = d_model // nheads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(d_model, nheads * att_size)
        self.linear_k = nn.Linear(d_model, nheads * att_size)
        self.linear_v = nn.Linear(d_model, nheads * att_size)
        self.dropout = nn.Dropout(dropout)

        self.output_layer = nn.Linear(nheads * att_size, d_model)

    def configure_precision(self, *, device_type: str, autocast_dtype: torch.dtype | None) -> bool:
        self.autocast_device_type = device_type
        self.autocast_dtype = autocast_dtype
        self.use_selective_bf16 = autocast_dtype == torch.bfloat16
        return True

    def _bf16_context(self, reference_tensor: Tensor):
        device_type = reference_tensor.device.type if torch.is_tensor(reference_tensor) else self.autocast_device_type
        if not self.use_selective_bf16:
            return torch.autocast(device_type=device_type, enabled=False)
        return torch.autocast(device_type=device_type, dtype=self.autocast_dtype)

    def _project_bf16(self, inputs: Tensor, linear: nn.Linear) -> Tensor:
        with self._bf16_context(inputs):
            return F.linear(inputs, linear.weight, linear.bias)

    def _softmax_fp32(self, scores: Tensor) -> Tensor:
        return torch.softmax(scores.float(), dim=3)

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

        q = self._project_bf16(q, self.linear_q).view(batch_size, -1, self.nheads, d_k)
        k = self._project_bf16(k, self.linear_k).view(batch_size, -1, self.nheads, d_k)
        v = self._project_bf16(v, self.linear_v).view(batch_size, -1, self.nheads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2)  # [b, h, k_len, d_k]

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

        with self._bf16_context(q):
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

            x = self._softmax_fp32(x)
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
            with self._bf16_context(v):
                x = torch.matmul(x, v)
                x = x + torch.matmul(value_hop_att, value_hop_emb) + torch.matmul(value_edge_att, value_edge_emb)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.nheads * d_v)

        x = self._project_bf16(x, self.output_layer).float()
        assert x.size() == orig_q_size
        return x

class ReferenceCrossAttnBlock(nn.Module):
    """Reference cross-attention block — one per active decoder layer.

    Owned by GraphMotionDecoder so that disabled layers (reference_cond=False
    or outside the last-N range) don't carry dead weights. Mirrors the
    cross-limb pattern: the decoder allocates a ModuleList of these and
    passes the per-layer instance into GraphMotionDecoderLayer.forward.
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.reference_attn = SelectiveMultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout_ref = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        reference_memory: Tensor,
        key_padding_mask: Optional[Tensor],
        reference_batch_mask: Optional[Tensor],
    ) -> Tensor:
        frames, bs, njoints, feats = x.size()
        queries = x.reshape(frames, bs * njoints, feats)
        if reference_memory.dim() != 3 or reference_memory.shape[1] != bs or reference_memory.shape[2] != feats:
            raise ValueError(
                "reference_memory must have shape (K, B, D), got "
                f"{tuple(reference_memory.shape)} for batch={bs}, dim={feats}"
            )
        memory = reference_memory.unsqueeze(2).expand(-1, -1, njoints, -1).reshape(reference_memory.shape[0], bs * njoints, feats)
        expanded_key_padding_mask = None
        if key_padding_mask is not None:
            if key_padding_mask.dim() != 2 or key_padding_mask.shape[0] != bs:
                raise ValueError(
                    "reference_key_padding_mask must have shape (B, K), got "
                    f"{tuple(key_padding_mask.shape)} for batch={bs}"
                )
            expanded_key_padding_mask = key_padding_mask.unsqueeze(1).expand(-1, njoints, -1).reshape(bs * njoints, -1)
        attn_output, _ = self.reference_attn(
            queries,
            memory,
            memory,
            key_padding_mask=expanded_key_padding_mask,
            need_weights=False,
        )
        attn_output = attn_output.reshape(frames, bs, njoints, feats)
        if reference_batch_mask is not None:
            batch_mask = reference_batch_mask.to(device=attn_output.device, dtype=attn_output.dtype)
            attn_output = attn_output * batch_mask.view(1, bs, 1, 1)
        return self.dropout_ref(attn_output)


class GraphMotionDecoder(nn.TransformerDecoder):
    def __init__(self, decoder_layer, num_layers, norm=None, max_path_len=5, value_emb=False,
                 cross_limb=True, cross_limb_latents=8, cross_limb_dim=64,
                 cross_limb_last_n=0, reference_cond=False, reference_cond_last_n=0):
                # multi head attention
        super().__init__(decoder_layer, num_layers, norm)

        self.d_model = decoder_layer.d_model
        self.nheads = decoder_layer.heads
        # 0 -> apply at every layer; N>0 -> only the last N layers. Each active
        # layer gets its own independent block, so this also scales the
        # parameter count for both cross-limb and reference paths.
        self.cross_limb_last_n = cross_limb_last_n
        self.reference_cond_last_n = reference_cond_last_n
        if cross_limb:
            num_active = (
                cross_limb_last_n if cross_limb_last_n > 0 else num_layers
            )
            self.cross_limb_blocks = nn.ModuleList([
                CrossLimbTemporalBlock(
                    self.d_model, decoder_layer.heads,
                    num_latents=cross_limb_latents,
                    dropout=decoder_layer.dropout1.p,
                    latent_width=cross_limb_dim,
                )
                for _ in range(num_active)
            ])
        else:
            self.cross_limb_blocks = None
        if reference_cond:
            num_ref_active = (
                reference_cond_last_n if reference_cond_last_n > 0 else num_layers
            )
            self.reference_blocks = nn.ModuleList([
                ReferenceCrossAttnBlock(
                    self.d_model, decoder_layer.heads,
                    dropout=decoder_layer.dropout1.p,
                )
                for _ in range(num_ref_active)
            ])
        else:
            self.reference_blocks = None
        self.topology_key_emb = nn.Embedding(max_path_len + 1, self.d_model) # 'far': max_path_len + 1
        self.edge_key_emb = nn.Embedding(6, self.d_model) # 'self':0, 'parent':1, 'child':2, 'sibling':3, 'no_relation':4, 'end_effector':5
        self.topology_query_emb = nn.Embedding(max_path_len + 1, self.d_model) # 'far': max_path_len + 1
        self.edge_query_emb = nn.Embedding(6, self.d_model) # 'self':0, 'parent':1, 'child':2, 'sibling':3, 'no_relation':4, 'end_effector':5
        self.value_emb_flag = value_emb
        if value_emb:
            self.topology_value_emb = nn.Embedding(max_path_len + 1, self.d_model) # 'far': max_path_len + 1
            self.edge_value_emb = nn.Embedding(6, self.d_model) # 'self':0, 'parent':1, 'child':2, 'sibling':3, 'no_relation':4, 'end_effector':5

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
                temporal_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None, y=None, get_layer_activation=-1, reference_memory: Optional[Tensor] = None,
            global_energy_condition: Optional[Tensor] = None,
            reference_key_padding_mask: Optional[Tensor] = None, temporal_template: Optional[Tensor] = None,
            cross_limb_unreliable_mask: Optional[Tensor] = None,
            reference_batch_mask: Optional[Tensor] = None,
            loop_phase_mask: Optional[Tensor] = None,
            lengths: Optional[Tensor] = None) -> Union[Tensor , Tuple[Tensor, dict]]:
        topology_rel = self._expand_relation_heads(y['graph_dist'].to(device=tgt.device, dtype=torch.long))
        edge_rel = self._expand_relation_heads(y['joints_relations'].to(device=tgt.device, dtype=torch.long))
        output = tgt
        T, B = tgt.shape[0], tgt.shape[1]
        loop_phase_mask_batch = None
        loop_phase_embedding = None
        cross_limb_time_embedding = None
        if loop_phase_mask is not None:
            loop_phase_mask_batch = torch.as_tensor(loop_phase_mask, device=tgt.device, dtype=torch.bool).reshape(-1)
            if loop_phase_mask_batch.numel() == 1 and B != 1:
                loop_phase_mask_batch = loop_phase_mask_batch.expand(B)
            elif loop_phase_mask_batch.numel() != B:
                raise ValueError(
                    "loop_phase_mask batch dimension must match the motion batch size, got "
                    f"{loop_phase_mask_batch.numel()} for batch {B}"
                )
            if bool(loop_phase_mask_batch.any()):
                loop_phase_embedding = _circular_phase_embedding(
                    T,
                    self.d_model,
                    B,
                    tgt.device,
                    tgt.dtype,
                    lengths,
                )
                loop_phase_embedding = loop_phase_embedding * loop_phase_mask_batch.view(1, B, 1)
                if self.cross_limb_blocks is not None and len(self.cross_limb_blocks) > 0:
                    cross_limb_dim = self.cross_limb_blocks[0].latent_dim
                    cross_limb_time_embedding = _loop_aware_time_embedding(
                        T,
                        cross_limb_dim,
                        B,
                        tgt.device,
                        tgt.dtype,
                        loop_phase_mask_batch,
                        lengths,
                    )
            else:
                loop_phase_mask_batch = None
        if get_layer_activation > -1 and get_layer_activation < self.num_layers:
            activations=dict()
        first_cl_layer = (
            self.num_layers - self.cross_limb_last_n
            if self.cross_limb_last_n > 0 else 0
        )
        first_ref_layer = (
            self.num_layers - self.reference_cond_last_n
            if self.reference_cond_last_n > 0 else 0
        )
        for layer_ind, mod in enumerate(self.layers):
            edge_value_emb = None
            topology_value_emb = None
            if self.value_emb_flag:
                edge_value_emb = self.edge_value_emb
                topology_value_emb = self.topology_value_emb
            if self.cross_limb_blocks is not None and layer_ind >= first_cl_layer:
                cl_block = self.cross_limb_blocks[layer_ind - first_cl_layer]
            else:
                cl_block = None
            if self.reference_blocks is not None and layer_ind >= first_ref_layer:
                ref_block = self.reference_blocks[layer_ind - first_ref_layer]
            else:
                ref_block = None
            output = mod(
                    output, timesteps_embs, topology_rel, edge_rel, self.edge_key_emb, self.edge_query_emb, edge_value_emb, self.topology_key_emb, self.topology_query_emb, topology_value_emb, spatial_mask, temporal_mask,
                    tgt_key_padding_mask, memory_key_padding_mask, y, reference_memory, global_energy_condition, reference_key_padding_mask,
                    temporal_template=temporal_template, cross_limb_block=cl_block,
                    reference_block=ref_block,
                    cross_limb_unreliable_mask=cross_limb_unreliable_mask,
                    reference_batch_mask=reference_batch_mask,
                    loop_phase_mask=loop_phase_mask_batch,
                    lengths=lengths,
                    loop_phase_embedding=loop_phase_embedding,
                    cross_limb_time_embedding=cross_limb_time_embedding)
            if layer_ind == get_layer_activation:
                activations[layer_ind] = output.clone()
        if self.norm is not None:
            output = self.norm(output)
        if get_layer_activation > -1 and get_layer_activation < self.num_layers:
            return output, activations
        return output

class GraphMotionDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 reference_residual_gate: float = 1.0,
                 global_energy_cond: bool = False):
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
        self.global_energy_cond = bool(global_energy_cond)
        if self.global_energy_cond:
            self.layer_gamma_scale = nn.Parameter(torch.zeros(d_model))
            self.layer_gamma_bias = nn.Parameter(torch.zeros(d_model))
            self.layer_beta_scale = nn.Parameter(torch.zeros(d_model))
            self.layer_beta_bias = nn.Parameter(torch.zeros(d_model))
        # norm_ref is reused by both the reference-residual path and the
        # global-energy path, so it stays on every layer regardless of
        # reference_cond_last_n.
        self.norm_ref = nn.LayerNorm(d_model)
        self.reference_residual_gate = float(reference_residual_gate)
        if self.reference_residual_gate < 0.0:
            raise ValueError(
                f"reference_residual_gate must be >= 0, got {self.reference_residual_gate}"
            )
        # Both the cross-limb pathway and the reference cross-attention block
        # are owned by GraphMotionDecoder (one block per active layer) and
        # passed into forward(), not held here.
        self.temporal_phase_scale = nn.Parameter(torch.zeros(1))

    # spatial attention block
    def _spatial_mha_block(self, x: Tensor, topology_rel: Optional[Tensor], edge_rel: Optional[Tensor], edge_key_emb, edge_query_emb, edge_value_emb,
        topology_key_emb, topology_query_emb, topology_value_emb, attn_mask: Optional[Tensor],  key_padding_mask: Optional[Tensor], y = None) -> Tensor:
        #x.shape (frames, bs, njoints, feature_len)
        frames, bs, njoints, feature_len = x.shape
        x = x.reshape(frames * bs, njoints, feature_len)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0).expand(frames, bs, njoints).reshape(-1, njoints)
        
        attn_output = self.spatial_attn(x, x, x, topology_query_emb.weight, edge_query_emb.weight, topology_key_emb.weight, edge_key_emb.weight, None if topology_value_emb is None else topology_value_emb.weight, 
        None if edge_value_emb is None else edge_value_emb.weight, topology_rel, edge_rel, attn_mask, key_padding_mask=key_padding_mask)
        attn_output = attn_output.reshape(frames, bs, njoints, feature_len) # njoints, bs, frames, feature_len
        return self.dropout1(attn_output)
    
    
        # temporal attention block
    def _temporal_mha_block_sin_joint(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], loop_phase_mask: Optional[Tensor] = None, lengths: Optional[Tensor] = None, loop_phase_embedding: Optional[Tensor] = None) -> Tensor:
        frames, bs, njoints, feats= x.size() 
        # attn_mask_ = attn_mask[..., 1:, 1:]
        if loop_phase_embedding is not None:
            if loop_phase_embedding.shape != (frames, bs, feats):
                raise ValueError(
                    "loop_phase_embedding must have shape "
                    f"{(frames, bs, feats)}, got {tuple(loop_phase_embedding.shape)}"
                )
            x = x + self.temporal_phase_scale * loop_phase_embedding.unsqueeze(2)
        elif loop_phase_mask is not None:
            loop_phase_mask = torch.as_tensor(loop_phase_mask, device=x.device, dtype=torch.bool).reshape(-1)
            if loop_phase_mask.numel() == 1 and bs != 1:
                loop_phase_mask = loop_phase_mask.expand(bs)
            if bool(loop_phase_mask.any()):
                phase = _circular_phase_embedding(frames, feats, bs, x.device, x.dtype, lengths)
                phase = phase * loop_phase_mask.view(1, bs, 1)
                x = x + self.temporal_phase_scale * phase.unsqueeze(2)
        x = x.view(frames, bs * njoints, feats)
        output_attn, _ = self.temporal_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        output_attn = output_attn.view(frames, bs ,njoints, feats)
        return self.dropout2(output_attn)
    
    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

    def _apply_global_energy_cond(
        self,
        x: Tensor,
        global_energy_condition: Tensor,
    ) -> Tensor:
        if not self.global_energy_cond:
            raise RuntimeError(
                "This model was built with global_energy_cond=False; "
                "do not pass global_energy_condition."
            )
        frames, bs, _, feats = x.size()
        if global_energy_condition.dim() == 1:
            global_energy_condition = global_energy_condition.unsqueeze(0)
        if global_energy_condition.dim() != 2 or global_energy_condition.shape[1] != feats * 2:
            raise ValueError(
                "global_energy_condition must have shape (B, 2*D) or (2*D,), got "
                f"{tuple(global_energy_condition.shape)} for batch={bs}, expected={feats * 2}"
            )
        if global_energy_condition.shape[0] == 1 and bs != 1:
            global_energy_condition = global_energy_condition.expand(bs, -1)
        elif global_energy_condition.shape[0] != bs:
            raise ValueError(
                "global_energy_condition batch dimension must match the motion batch size, got "
                f"{global_energy_condition.shape[0]} for batch {bs}"
            )
        cond = global_energy_condition.to(device=x.device, dtype=x.dtype)
        gamma, beta = cond.chunk(2, dim=-1)
        gamma = gamma.view(1, bs, 1, feats)
        beta  = beta.view(1, bs, 1, feats)
        # Clamp gamma_scale to prevent collapse to -1 (which would zero out
        # the global-energy contribution). Straight-through: clamping is
        # applied in-place during forward so gradients flow through the
        # unclamped values, keeping the optimizer's dynamics intact.
        gamma = gamma * (1.0 + self.layer_gamma_scale.clamp(-0.95, 3.0)) + self.layer_gamma_bias
        beta  = beta  * (1.0 + self.layer_beta_scale.clamp(-0.95, 3.0))  + self.layer_beta_bias
        gamma = torch.tanh(gamma)
        return x * (1.0 + gamma) + beta
    
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
        temporal_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None, #for future use
        y = None,
        reference_memory: Optional[Tensor] = None,
        global_energy_condition: Optional[Tensor] = None,
        reference_key_padding_mask: Optional[Tensor] = None,
        temporal_template: Optional[Tensor] = None,
        cross_limb_block: Optional[nn.Module] = None,
        reference_block: Optional[nn.Module] = None,
        cross_limb_unreliable_mask: Optional[Tensor] = None,
        reference_batch_mask: Optional[Tensor] = None,
        loop_phase_mask: Optional[Tensor] = None,
        lengths: Optional[Tensor] = None,
        loop_phase_embedding: Optional[Tensor] = None,
        cross_limb_time_embedding: Optional[Tensor] = None) -> Tensor:
        x = tgt #(frames, bs, njoints, feature_len)
        bs = x.shape[1]
        x = x + self.embed_timesteps(timesteps_emb).view(1, bs, 1, self.d_model)
        spatial_attn_output = self._spatial_mha_block(x, topology_rel, edge_rel, edge_key_emb, edge_query_emb, edge_value_emb,
        topo_key_emb, topo_query_emb, topo_value_emb, spatial_mask, tgt_key_padding_mask, y)
        x = self.norm1(x + spatial_attn_output)
        x = self.norm2(x + self._temporal_mha_block_sin_joint(x, temporal_mask, None, loop_phase_mask=loop_phase_mask, lengths=lengths, loop_phase_embedding=loop_phase_embedding))
        if cross_limb_block is not None:
            x = cross_limb_block(
                x,
                temporal_template,
                y['joints_key_padding_mask'],
                unreliable_mask=cross_limb_unreliable_mask,
                loop_phase_mask=loop_phase_mask,
                lengths=lengths,
                time_embedding=cross_limb_time_embedding,
            )
        reference_delta = None
        conditioning_batch_mask = None
        if reference_block is not None and reference_memory is not None and self.reference_residual_gate != 0.0:
            reference_delta = reference_block(
                x,
                reference_memory,
                reference_key_padding_mask,
                reference_batch_mask,
            )
            if self.reference_residual_gate != 1.0:
                reference_delta = reference_delta * self.reference_residual_gate
            if reference_batch_mask is not None:
                conditioning_batch_mask = reference_batch_mask.to(device=x.device, dtype=torch.bool)
        if reference_delta is not None:
            reference_conditioned = x + reference_delta
        else:
            reference_conditioned = x
        if conditioning_batch_mask is not None and reference_delta is not None:
            batch_mask = conditioning_batch_mask.view(1, bs, 1, 1)
            reference_conditioned = torch.where(batch_mask, reference_conditioned, x)
        if global_energy_condition is not None:
            x = self.norm_ref(self._apply_global_energy_cond(reference_conditioned, global_energy_condition))
        elif reference_delta is not None:
            conditioned_output = self.norm_ref(x + reference_delta)
            if conditioning_batch_mask is None:
                x = conditioned_output
            else:
                batch_mask = conditioning_batch_mask.view(1, bs, 1, 1)
                x = torch.where(batch_mask, conditioned_output, x)
        x = self.norm3(x + self._ff_block(x))
        return x
