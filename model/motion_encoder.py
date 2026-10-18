"""Frame-aligned, joint-count independent reference motion bottleneck."""
import torch
from torch import nn

from model.motion_transformer import GraphMotionDecoder, GraphMotionDecoderLayer


class MotionEncoder(nn.Module):
    def __init__(self, t5_out_dim, slots=2, ref_dim=8, width=64, layers=2,
                 heads=4, dropout=0.1):
        super().__init__()
        # Local import: AnyTop also owns this module. These are independent
        # projections, with exactly the same 9 joint / 11 root input channels.
        from model.anytop import InputProcess
        from data_loaders.truebones.truebones_utils.canonical_features import (
            JOINT_MODEL_CHANNELS, ROOT_MODEL_CHANNELS,
        )
        if min(slots, ref_dim, width, layers, heads) < 1 or width % heads:
            raise ValueError('Invalid reference encoder dimensions / attention heads')
        if width < 4 or width % 2:
            raise ValueError('Reference encoder width must be even and >= 4')
        self.slots, self.ref_dim = slots, ref_dim
        self.input_process = InputProcess(len(JOINT_MODEL_CHANNELS), len(ROOT_MODEL_CHANNELS),
                                          width, t5_out_dim, dropout_prob=dropout)
        self.decoder = GraphMotionDecoder(
            GraphMotionDecoderLayer(width, heads, width * 4, dropout, 'gelu'),
            layers, cross_limb=False,
        )
        self.queries = nn.Parameter(torch.randn(slots, width) * width ** -0.5)
        self.pool_norm = nn.LayerNorm(width)
        self.pool = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.output = nn.Linear(width, ref_dim)

    def forward(self, y):
        feats = y['ref_feats']
        batch, joints, _, frames = feats.shape
        valid = y['ref_mask'].to(device=feats.device, dtype=torch.bool)
        if valid.shape != (batch, joints):
            raise ValueError('ref_mask must have shape [B, J_source]')
        # Null rows still execute a finite static graph. An otherwise all-masked
        # attention row gets a zero sentinel, then the decoder hard-bypasses it.
        safe_valid = valid | ((~valid.any(-1, keepdim=True)) &
                              (torch.arange(joints, device=feats.device)[None] == 0))
        feats = feats.masked_fill(~valid[:, :, None, None], 0)
        tokens = self.input_process(
            feats, y['ref_rest_pose'].unsqueeze(0), y['ref_joints_names_embs'],
            joint_valid=valid, joint_struct=y['ref_joint_struct'],
        )
        tokens = self.decoder(
            tokens, tokens.new_zeros(batch, tokens.shape[-1]), None,
            tgt_key_padding_mask=~safe_valid,
            y={'graph_dist': y['ref_graph_dist'], 'joints_relations': y['ref_joints_relations']},
        )[1:]  # Rest token is context, never a motion slot.
        values = self.pool_norm(tokens).reshape(frames * batch, joints, -1)
        queries = self.queries[None].expand(frames * batch, -1, -1)
        mask = (~safe_valid)[None].expand(frames, -1, -1).reshape(frames * batch, joints)
        pooled, _ = self.pool(queries, values, values, key_padding_mask=mask, need_weights=False)
        z = self.output(pooled).reshape(frames, batch, self.slots, self.ref_dim)
        return z.permute(1, 0, 2, 3).contiguous()  # [B, T, K, d_z], no extra time PE
