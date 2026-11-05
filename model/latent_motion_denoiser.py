"""Fixed-window latent x0 denoiser. Graph relations belong only to skeleton tokens."""
import torch
from torch import nn

from model.anytop import InputProcess
from model.motion_conditioning import MotionConditioningMixin, create_sin_embedding, _zero_init_mlp
from model.motion_transformer import GraphMotionDecoder, GraphMotionDecoderLayer, run_in_fp32
from data_loaders.truebones.truebones_utils.action_label_conditioning_contract import projection_input_dim


class StaticSkeletonEmbedding(InputProcess):
    def __init__(self, width, t5_out_dim, species_cond, bottleneck):
        super().__init__(9, width, t5_out_dim, species_cond, bottleneck)
        del self.joint_embedding

    def forward(self, y, valid, name_drop, rest_drop, species):
        names, geometry = self.static_components(
            y['rest_geom'], y['joints_names_embs'], valid.device, torch.float32,
            valid, y['joint_struct'], name_drop, rest_drop, species)
        return (names + geometry).masked_fill(~valid[..., None], 0)


class LatentDecoderLayer(GraphMotionDecoderLayer):
    """Read static skeleton after slot interaction and before time attention."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skeleton_norm = nn.LayerNorm(self.d_model)
        self.skeleton_attention = nn.MultiheadAttention(
            self.d_model, self.heads, batch_first=True)

    def _read_motion_latent(self, x, context):
        skeleton, padding = context
        frames, batch, slots, width = x.shape
        # Combine frames and slots in the query; static keys are projected once.
        query = self.skeleton_norm(x).permute(1, 0, 2, 3).reshape(batch, frames * slots, width)
        read, _ = self.skeleton_attention(query, skeleton, skeleton,
                                          key_padding_mask=padding, need_weights=False)
        return x + read.reshape(batch, frames, slots, width).permute(1, 0, 2, 3)


class LatentMotionDenoiser(MotionConditioningMixin, nn.Module):
    def __init__(self, t5_out_dim, slots=8, latent_channels=32, frames=60,
                 width=256, layers=8, heads=4, ff_size=1024, dropout=0.05,
                 skeleton_layers=2, temporal_rope=True,
                 action_label_cond=True, action_conditioning=None,
                 action_label_cfg_drop_prob=0.2, direction_slot_drop_prob=0.0,
                 modifier_slot_drop_prob=0.0, species_cond=True,
                 species_film_bottleneck=128, joint_name_drop_prob=0.15,
                 rest_geom_joint_drop_prob=0.1,
                 temporal_rope_drop_prob=0.0):
        super().__init__()
        config = dict(locals())
        config.pop('self'); config.pop('__class__', None); config.pop('action_conditioning')
        self.config = config
        if frames != 60:
            raise ValueError('Latent diffusion currently requires exactly 60 frames')
        if min(slots, latent_channels, layers, skeleton_layers, heads) < 1 or width < 4 or width % (2 * heads):
            raise ValueError('Invalid latent denoiser dimensions')
        for key, value in config.items():
            if key.endswith('_prob') or key == 'dropout':
                if not 0 <= value <= 1:
                    raise ValueError(f'Invalid {key}: {value}')
            setattr(self, key, value)
        self.latent_dim = width  # shared condition heads use the trunk width
        self.tempo_projection = _zero_init_mlp(width)
        self.loop_period_projection = _zero_init_mlp(width)
        self.action_label_projection = self.action_label_null_emb = None
        self.action_conditioning_metadata = None
        if action_label_cond:
            self._init_action_conditioning(action_conditioning, t5_out_dim)
            self.action_label_projection = nn.Sequential(
                nn.Linear(projection_input_dim(t5_out_dim), width), nn.GELU(), nn.Linear(width, width))
            self.action_label_null_emb = nn.Parameter(torch.zeros(width))
        self.static = StaticSkeletonEmbedding(width, t5_out_dim, species_cond, species_film_bottleneck)
        self.skeleton_encoder = GraphMotionDecoder(
            GraphMotionDecoderLayer(width, heads, ff_size, dropout, 'gelu'),
            skeleton_layers, cross_limb=False, graph_relations=True)
        self.input = nn.Linear(latent_channels, width)
        self.slot_identity = nn.Parameter(torch.randn(slots, width) * width ** -0.5)
        self.trunk = GraphMotionDecoder(
            LatentDecoderLayer(width, heads, ff_size, dropout, 'gelu'), layers,
            cross_limb=False, graph_relations=False, temporal_rope=temporal_rope,
            action_label_adaln=action_label_cond)
        self.output_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, latent_channels)

    def forward(self, z, timesteps, y=None):
        from diffusion.latent_diffusion import validate_latent_conditions
        validate_latent_conditions(y, self.frames)
        batch, frames, slots, channels = z.shape
        if (frames, slots, channels) != (self.frames, self.slots, self.latent_channels):
            raise ValueError('Expected latent [B,60,K,d] matching the AE layout')
        is_loop, cycles = self._loop_condition(y, batch, z.device)
        action = self._build_action_label_token(y, batch, z.device, torch.float32)
        timestep = create_sin_embedding(timesteps.reshape(1, -1, 1), self.latent_dim)[0]
        if action is not None:
            timestep = timestep + action
        joints = y['joints_names_embs'].shape[1]
        valid = torch.arange(joints, device=z.device)[None] < y['n_joints'][:, None]
        species = self._coerce_species_code(y, batch, z.device, torch.float32) if self.species_cond else None
        skeleton = self.static(y, valid, self._sample_joint_name_drop(batch, joints, z.device),
                               self._sample_rest_geom_drop(batch, joints, z.device), species)
        skeleton = self.skeleton_encoder(skeleton[None], skeleton.new_zeros(batch, self.latent_dim),
                                          None, tgt_key_padding_mask=~valid, y=y)[0]
        x = self.input(z).permute(1, 0, 2, 3) + self.slot_identity[None, None]
        x = x + self._frame_position_embedding(frames, is_loop, cycles, z.device)[:, :, None]
        x = x + self._tempo_frame_embedding(y, batch, frames, z.device, is_loop, cycles)[:, :, None]
        off = (torch.rand(batch, device=z.device) < self.temporal_rope_drop_prob
               if self.temporal_rope and self.training and self.temporal_rope_drop_prob else None)
        # An explicit override must work even when training RoPE drop is disabled.
        explicit = y.get('temporal_rope_off')
        if explicit is not None:
            explicit = self._per_sample(explicit, batch, z.device, torch.bool, 'temporal_rope_off')
            off = explicit if off is None else off | explicit
        x = self.trunk(x, timestep, None, y={}, motion_z=(skeleton, ~valid),
                       loop_phase_mask=is_loop, loop_cycles=cycles, temporal_rope_off=off,
                       action_adaln_cond=action)
        return run_in_fp32(self.output, self.output_norm(x)).permute(1, 0, 2, 3).contiguous()
