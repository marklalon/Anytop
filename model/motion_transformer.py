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

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, tgt_len, self.embed_dim)
        attn_output = attn_output.transpose(0, 1)
        attn_output = self._project_bf16(attn_output, self.out_proj.weight, self.out_proj.bias).float()

        if not need_weights:
            return attn_output, None
        if average_attn_weights:
            return attn_output, attn_weights_fp32.mean(dim=1)
        return attn_output, attn_weights_fp32

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
        num_hop_types = query_hop_emb.shape[0]
        num_edge_types = query_edge_emb.shape[0]

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
            query_hop = torch.gather(
                query_hop, 3, distance.unsqueeze(1).repeat(1, self.nheads, 1, 1)
            )
            query_edge = torch.matmul(q, query_edge_emb.transpose(2, 3))
            query_edge = torch.gather(
                query_edge, 3, edge_attr.unsqueeze(1).repeat(1, self.nheads, 1, 1)
            )

            key_hop = torch.matmul(k, key_hop_emb.transpose(2, 3))
            key_hop = torch.gather(
                key_hop, 3, distance.unsqueeze(1).repeat(1, self.nheads, 1, 1)
            )
            key_edge = torch.matmul(k, key_edge_emb.transpose(2, 3))
            key_edge = torch.gather(
                key_edge, 3, edge_attr.unsqueeze(1).repeat(1, self.nheads, 1, 1)
            )
            qk = torch.matmul(q, k.transpose(2, 3))

        # Accumulate in fp32 to prevent catastrophic cancellation from summing bf16 terms
        x = (qk.float() + query_hop.float() + key_hop.float() + query_edge.float() + key_edge.float()) * self.scale

        if mask is not None:
            x = x + mask.to(device=x.device, dtype=torch.float32)

        x = self._softmax_fp32(x)
        x = self.dropout(x)
        if value_hop_emb is not None:
            value_hop_emb = value_hop_emb.view(
                1, num_hop_types, self.nheads, self.att_size
            ).transpose(1, 2)
            value_edge_emb = value_edge_emb.view(
                1, num_edge_types, self.nheads, self.att_size
            ).transpose(1, 2)

            value_hop_att = torch.zeros(
                (batch_size, self.nheads, sequence_length, num_hop_types),
                device=value_hop_emb.device,
                dtype=x.dtype,
            )
            value_hop_att = torch.scatter_add(
                value_hop_att, 3, distance.unsqueeze(1).repeat(1, self.nheads, 1, 1), x
            )
            value_edge_att = torch.zeros(
                (batch_size, self.nheads, sequence_length, num_edge_types),
                device=value_hop_emb.device,
                dtype=x.dtype,
            )
            value_edge_att = torch.scatter_add(
                value_edge_att, 3, edge_attr.unsqueeze(1).repeat(1, self.nheads, 1, 1), x
            )
        with self._bf16_context(v):
            x = torch.matmul(x, v)
            if value_hop_emb is not None:
                x = x + torch.matmul(value_hop_att, value_hop_emb) + torch.matmul(value_edge_att, value_edge_emb)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.nheads * d_v)

        x = self._project_bf16(x, self.output_layer).float()
        assert x.size() == orig_q_size
        return x

class GraphMotionDecoder(nn.TransformerDecoder):
    def __init__(self, decoder_layer, num_layers, norm=None, max_path_len=5, value_emb=False): 
                # multi head attention
        super().__init__(decoder_layer, num_layers, norm)
        
        self.d_model = decoder_layer.d_model
        self.topology_key_emb = nn.Embedding(max_path_len + 1, self.d_model) # 'far': max_path_len + 1
        self.edge_key_emb = nn.Embedding(6, self.d_model) # 'self':0, 'parent':1, 'child':2, 'sibling':3, 'no_relation':4, 'end_effector':5
        self.topology_query_emb = nn.Embedding(max_path_len + 1, self.d_model) # 'far': max_path_len + 1
        self.edge_query_emb = nn.Embedding(6, self.d_model) # 'self':0, 'parent':1, 'child':2, 'sibling':3, 'no_relation':4, 'end_effector':5
        self.value_emb_flag = value_emb
        if value_emb:
            self.topology_value_emb = nn.Embedding(max_path_len + 1, self.d_model) # 'far': max_path_len + 1
            self.edge_value_emb = nn.Embedding(6, self.d_model) # 'self':0, 'parent':1, 'child':2, 'sibling':3, 'no_relation':4, 'end_effector':5
        

        
    def forward(self, tgt: Tensor, timesteps_embs: Tensor, memory: Tensor, spatial_mask:  Optional[Tensor] = None,
                temporal_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None, y=None, get_layer_activation=-1, reference_memory: Optional[Tensor] = None,
            reference_key_padding_mask: Optional[Tensor] = None) -> Union[Tensor , Tuple[Tensor, dict]]:
        topology_rel = y['graph_dist'].long().to(tgt.device)
        edge_rel = y['joints_relations'].long().to(tgt.device)
        output = tgt
        if get_layer_activation > -1 and get_layer_activation < self.num_layers:
            activations=dict()
        for layer_ind, mod in enumerate(self.layers):
            edge_value_emb = None
            topology_value_emb = None
            if self.value_emb_flag:
                edge_value_emb = self.edge_value_emb
                topology_value_emb = self.topology_value_emb
            output = mod(
                    output, timesteps_embs, topology_rel, edge_rel, self.edge_key_emb, self.edge_query_emb, edge_value_emb, self.topology_key_emb, self.topology_query_emb, topology_value_emb, spatial_mask, temporal_mask,
                    tgt_key_padding_mask, memory_key_padding_mask, y, reference_memory, reference_key_padding_mask)
            if layer_ind == get_layer_activation:
                activations[layer_ind] = output.clone()
        if self.norm is not None:
            output = self.norm(output)
        if get_layer_activation > -1 and get_layer_activation < self.num_layers:
            return output, activations
        return output

class GraphMotionDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)
        self.d_model= d_model
        self.heads = nhead
        self.spatial_attn = GraphMultiHeadAttention(d_model = d_model, nheads = nhead, dropout=dropout)
        self.temporal_attn = SelectiveMultiheadAttention(self.d_model, nhead, dropout=dropout)
        self.reference_attn = SelectiveMultiheadAttention(self.d_model, nhead, dropout=dropout)
        self.embed_timesteps = nn.Linear(d_model, d_model)
        self.norm_ref = nn.LayerNorm(d_model)
        self.dropout_ref = nn.Dropout(dropout)

    # spatial attention block
    def _spatial_mha_block(self, x: Tensor, topology_rel: Optional[Tensor], edge_rel: Optional[Tensor], edge_key_emb, edge_query_emb, edge_value_emb,
        topology_key_emb, topology_query_emb, topology_value_emb, attn_mask: Optional[Tensor],  key_padding_mask: Optional[Tensor], y = None) -> Tensor:
        #x.shape (frames, bs, njoints, feature_len)
        frames, bs, njoints, feature_len = x.shape
        x = x.view(frames * bs, njoints, feature_len)
        topology_rel = topology_rel.unsqueeze(0).repeat(frames, 1, 1, 1).view(-1, njoints, njoints)
        edge_rel = edge_rel.unsqueeze(0).repeat(frames, 1, 1, 1).view(-1, njoints, njoints)
        
        attn_output = self.spatial_attn(x, x, x, topology_query_emb.weight, edge_query_emb.weight, topology_key_emb.weight, edge_key_emb.weight, None if topology_value_emb is None else topology_value_emb.weight, 
        None if edge_value_emb is None else edge_value_emb.weight, topology_rel, edge_rel, attn_mask)
        attn_output = attn_output.reshape(frames, bs, njoints, feature_len) # njoints, bs, frames, feature_len
        return self.dropout1(attn_output)
    
    
        # temporal attention block
    def _temporal_mha_block_sin_joint(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        frames, bs, njoints, feats= x.size() 
        # attn_mask_ = attn_mask[..., 1:, 1:]
        x = x.view(frames, bs * njoints, feats)
        output_attn, output_scores = self.temporal_attn(x, x, x,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask)
        output_attn = output_attn.view(frames, bs ,njoints, feats)
        return self.dropout2(output_attn)
    
    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

    def _reference_mha_block(self, x: Tensor, reference_memory: Tensor, key_padding_mask: Optional[Tensor]) -> Tensor:
        frames, bs, njoints, feats = x.size()
        queries = x.reshape(frames, bs * njoints, feats)
        memory = reference_memory.reshape(reference_memory.shape[0], bs * njoints, feats)
        attn_output, _ = self.reference_attn(
            queries,
            memory,
            memory,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        attn_output = attn_output.reshape(frames, bs, njoints, feats)
        return self.dropout_ref(attn_output)
    
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
        reference_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = tgt #(frames, bs, njoints, feature_len)
        bs = x.shape[1]
        x = x + self.embed_timesteps(timesteps_emb).view(1, bs, 1, self.d_model)
        spatial_attn_output = self._spatial_mha_block(x, topology_rel, edge_rel, edge_key_emb, edge_query_emb, edge_value_emb,
        topo_key_emb, topo_query_emb, topo_value_emb, spatial_mask, tgt_key_padding_mask, y)
        x = self.norm1(x + spatial_attn_output)
        x = self.norm2(x + self._temporal_mha_block_sin_joint(x, temporal_mask, tgt_key_padding_mask))
        if reference_memory is not None:
            x = self.norm_ref(x + self._reference_mha_block(x, reference_memory, reference_key_padding_mask))
        x = self.norm3(x + self._ff_block(x))
        return x
