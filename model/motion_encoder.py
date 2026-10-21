"""Frame-aligned, joint-count independent reference motion bottleneck."""
import torch
from torch import nn

from model.motion_transformer import GraphMotionDecoder, GraphMotionDecoderLayer


# Structural columns the encoder reads: topology only. The rest-geometry
# columns (heights, offsets, attach height) are left out, and the lateral
# offset is reduced to its side sign, so the encoder never sees where the
# source's joints sit at rest.
REF_STRUCT_TOPOLOGY_COLUMNS = ('depth_norm', 'run_len_inv', 'is_leaf', 'subtree_n',
                               'sib_rank', 'sib_n_inv')
REF_STRUCT_SIDE_COLUMN = 'lateral_signed'
# A lateral offset within this fraction of the rig's span is the midline.
REF_SIDE_TOLERANCE = 1e-3


class MotionEncoder(nn.Module):
    """Encodes a reference from rest-free channels only.

    Every per-frame input means the same thing on every skeleton: the unit
    bone direction and the joint height (over ``L``) in the rest-facing frame,
    and the root's XZ velocity. Nothing describes the source's rest pose -- no
    rest token, no rest-centered position residual, no local rotation, no rest
    geometry -- so the encoder cannot code motion as a deviation from the
    source's own rest, which would then be replayed against the target's rest.

    ``rest_context`` adds one static per-joint input: the same pose channels
    evaluated on the source's rest (``ref_rest_pose``). The per-frame inputs
    stay absolute; the encoder may additionally read how far a joint departs
    from its rest (a tail raised far above where it hangs at rest), which the
    absolute direction alone cannot tell apart from a rig whose tail simply
    rests high.
    """

    def __init__(self, t5_out_dim, slots=2, ref_dim=8, width=64, layers=2,
                 heads=4, dropout=0.1, rest_context=False):
        super().__init__()
        from data_loaders.reference import REFERENCE_POSE_CHANNELS
        from data_loaders.truebones.truebones_utils.canonical_features import ROOT_VEL_CHANNELS
        from data_loaders.truebones.truebones_utils.joint_struct_features import (
            JOINT_STRUCT_FEATURE_NAMES,
        )
        if min(slots, ref_dim, width, layers, heads) < 1 or width % heads:
            raise ValueError('Invalid reference encoder dimensions / attention heads')
        if width < 4 or width % 2:
            raise ValueError('Reference encoder width must be even and >= 4')
        self.slots, self.ref_dim, self.width = slots, ref_dim, width
        self.root_vel_channels = tuple(ROOT_VEL_CHANNELS)
        self.struct_columns = [JOINT_STRUCT_FEATURE_NAMES.index(c) for c in REF_STRUCT_TOPOLOGY_COLUMNS]
        self.side_column = JOINT_STRUCT_FEATURE_NAMES.index(REF_STRUCT_SIDE_COLUMN)
        self.joint_embedding = nn.Linear(REFERENCE_POSE_CHANNELS, width)
        self.root_embedding = nn.Linear(REFERENCE_POSE_CHANNELS + len(ROOT_VEL_CHANNELS), width)
        self.names_dropout = nn.Dropout(p=dropout)
        self.text_embedding = nn.Linear(t5_out_dim, width)
        self.struct_embedding = nn.Sequential(
            nn.Linear(len(self.struct_columns) + 1, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.rest_embedding = nn.Linear(REFERENCE_POSE_CHANNELS, width) if rest_context else None
        # Plain spatial attention, no topology / edge bias: that bias costs as
        # much as in the shared decoder whatever the width. Skeleton identity
        # reaches the encoder through names and the topology columns.
        self.decoder = GraphMotionDecoder(
            GraphMotionDecoderLayer(width, heads, width * 4, dropout, 'gelu'),
            layers, cross_limb=False, graph_relations=False,
        )
        self.queries = nn.Parameter(torch.randn(slots, width) * width ** -0.5)
        self.pool_norm = nn.LayerNorm(width)
        self.pool = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.output = nn.Linear(width, ref_dim)

    def _structure(self, joint_struct):
        side = joint_struct[..., self.side_column:self.side_column + 1]
        side = torch.sign(side) * (side.abs() > REF_SIDE_TOLERANCE)
        return torch.cat([joint_struct[..., self.struct_columns], side], dim=-1)

    def forward(self, y):
        # Local import: model.anytop imports this module.
        from model.anytop import create_sin_embedding
        pose = y['ref_pose']
        batch, joints, _, frames = pose.shape
        valid = y['ref_mask'].to(device=pose.device, dtype=torch.bool)
        if valid.shape != (batch, joints):
            raise ValueError('ref_mask must have shape [B, J_source]')
        # Null rows still execute a finite static graph. An otherwise all-masked
        # attention row gets a zero sentinel, then the decoder hard-bypasses it.
        safe_valid = valid | ((~valid.any(-1, keepdim=True)) &
                              (torch.arange(joints, device=pose.device)[None] == 0))
        live = valid[:, :, None, None]
        pose = pose.masked_fill(~live, 0).permute(3, 0, 1, 2)  # [T, B, J, C]
        root_vel = torch.cat([y['ref_feats'][:, 0:1, c:c + 1] for c in self.root_vel_channels], dim=2)
        root_vel = root_vel.masked_fill(~live[:, :1], 0).permute(3, 0, 1, 2)
        tokens = torch.cat([
            self.root_embedding(torch.cat([pose[:, :, :1], root_vel], dim=-1)),
            self.joint_embedding(pose[:, :, 1:]),
        ], dim=2)
        names = self.text_embedding(self.names_dropout(y['ref_joints_names_embs']))
        static = names + self.struct_embedding(self._structure(y['ref_joint_struct']))
        if self.rest_embedding is not None:
            static = static + self.rest_embedding(y['ref_rest_pose'])
        # Re-zero after the projections: their biases would make padding live.
        tokens = tokens + (static * valid[..., None].to(names.dtype))[None]
        positions = torch.arange(frames, device=pose.device).view(1, -1, 1)
        tokens = tokens + create_sin_embedding(positions, self.width)[0].to(tokens.dtype)[:, None, None]
        tokens = self.decoder(
            tokens, tokens.new_zeros(batch, tokens.shape[-1]), None,
            tgt_key_padding_mask=~safe_valid, y={},
        )
        values = self.pool_norm(tokens).reshape(frames * batch, joints, -1)
        queries = self.queries[None].expand(frames * batch, -1, -1)
        mask = (~safe_valid)[None].expand(frames, -1, -1).reshape(frames * batch, joints)
        pooled, _ = self.pool(queries, values, values, key_padding_mask=mask, need_weights=False)
        z = self.output(pooled).reshape(frames, batch, self.slots, self.ref_dim)
        return z.permute(1, 0, 2, 3).contiguous()  # [B, T, K, d_z], no extra time PE
