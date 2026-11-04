"""Motion AE / diagonal-Gaussian VAE: frame-aligned latents, no skips."""
import torch
from torch import nn

from data_loaders.reference import REST_GEOM_DIM
from data_loaders.truebones.truebones_utils.canonical_features import MODEL_CHANNELS
from data_loaders.truebones.truebones_utils.joint_struct_features import JOINT_STRUCT_DIM
from model.anytop import create_sin_embedding, _zero_init_mlp
from model.motion_transformer import (
    GraphMotionDecoder, GraphMotionDecoderLayer,
    TEMPORAL_ROPE_MAX_FREQ, TEMPORAL_ROPE_MAX_PERIOD,
    run_in_fp32,
)
from utils.device_transfer import check_on_device


DECODER_CONTENT_VERSION = 'latent_time_v2'


def motion_masks(x, skeleton, frame_mask=None):
    """Return valid joint/frame masks (True = valid)."""
    batch, joints, _, frames = x.shape
    joint = torch.arange(joints, device=x.device)[None] < skeleton['n_joints'][:, None]
    if frame_mask is None:
        frame_mask = torch.ones(batch, frames, device=x.device, dtype=torch.bool)
    if frame_mask.shape != (batch, frames):
        raise ValueError('Each sample needs at least one valid joint and frame')
    if not torch.compiler.is_compiling():
        if not frame_mask.any(-1).all() or not joint.any(-1).all():
            raise ValueError('Each sample needs at least one valid joint and frame')
    return joint, frame_mask.bool()


class TemporalResidualConv(nn.Module):
    """Per-joint local refinement: depthwise temporal conv + channel mixing.

    Loop rows wrap at their valid length; open rows replicate edge frames.
    Frame masks describe right padding. Zero-init mixing starts as identity.
    """

    def __init__(self, width, kernel):
        super().__init__()
        if kernel < 1 or kernel % 2 == 0:
            raise ValueError('Temporal convolution kernel must be positive and odd')
        self.radius = kernel // 2
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(width, width, kernel, groups=width)
        self.mix = nn.Conv1d(width, width, 1)
        nn.init.zeros_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)

    def forward(self, tokens, frame, loop_metadata=None):
        frames, batch, joints, width = tokens.shape
        length = frame.sum(-1)[:, None].clamp_min(1)
        position = torch.arange(-self.radius, frames + self.radius, device=tokens.device)[None]
        is_loop = (loop_metadata or {}).get(
            'is_loop', torch.zeros(batch, device=tokens.device, dtype=torch.bool))
        indices = torch.where(is_loop[:, None], position % length,
                              torch.minimum(position.clamp_min(0), length - 1))
        values = self.norm(tokens).permute(1, 2, 3, 0)
        values = values.gather(-1, indices[:, None, None].expand(batch, joints, width, -1))
        values = values.reshape(batch * joints, width, -1)
        delta = self.mix(torch.nn.functional.gelu(self.depthwise(values)))
        delta = delta.reshape(batch, joints, width, frames).permute(3, 0, 1, 2)
        return tokens + delta.masked_fill(~frame.T[:, :, None, None], 0)


class MotionAutoencoder(nn.Module):
    """Motion-content flows only through frame-aligned z.

    Encoder/decoder read only named static skeleton fields, never reference,
    action labels, or other motion-content fields from a loader condition dictionary.
    Shared temporal metadata carries is_loop, loop_tile_count and
    resample_speed_cond (source frames / valid internal frames).
    Inputs/outputs are [B,J,9,T]; latents are [B,T,K,d_z].
    """

    def __init__(self, t5_out_dim, width=128, layers=2, heads=4, slots=8, latent_dim=32,
                 ff_size=0, last_layer_ff=0, cross_limb_dim=0, cross_limb_last_n=0,
                 temporal_rope=False, layerwise_latent=False, decoder_temporal_kernel=0, variational=False,
                 dropout_prob=0.0, mirror_twin_drop_prob=0.0, lateral_drop_prob=0.0,
                 encoder_layers=None, decoder_layers=None,
                 encoder_width=None, decoder_width=None,
                 encoder_ff_size=None, decoder_ff_size=None):
        super().__init__()
        encoder_layers = layers if encoder_layers is None else encoder_layers
        decoder_layers = layers if decoder_layers is None else decoder_layers
        encoder_width = width if encoder_width is None else encoder_width
        decoder_width = width if decoder_width is None else decoder_width
        encoder_ff_size = ff_size if encoder_ff_size is None else encoder_ff_size
        decoder_ff_size = ff_size if decoder_ff_size is None else decoder_ff_size
        if (min(width, layers, encoder_layers, decoder_layers, heads, slots, latent_dim,
                encoder_width, decoder_width) < 1
                or any(dim % heads or dim % 2 for dim in (encoder_width, decoder_width))):
            raise ValueError('Invalid AE dimensions')
        if min(encoder_ff_size, decoder_ff_size) < 0:
            raise ValueError('AE ff_size must be nonnegative (zero uses 4 * width)')
        self.config = dict(t5_out_dim=t5_out_dim, width=width, layers=layers,
                           encoder_layers=encoder_layers, decoder_layers=decoder_layers,
                           encoder_width=encoder_width, decoder_width=decoder_width,
                           encoder_ff_size=encoder_ff_size, decoder_ff_size=decoder_ff_size,
                           heads=heads, slots=slots, latent_dim=latent_dim,
                           ff_size=ff_size, last_layer_ff=last_layer_ff,
                           cross_limb_dim=cross_limb_dim, cross_limb_last_n=cross_limb_last_n,
                           temporal_rope=temporal_rope, layerwise_latent=layerwise_latent,
                           decoder_temporal_kernel=decoder_temporal_kernel, variational=variational,
                           dropout_prob=dropout_prob, mirror_twin_drop_prob=mirror_twin_drop_prob,
                           lateral_drop_prob=lateral_drop_prob)
        for key in ('dropout_prob', 'mirror_twin_drop_prob', 'lateral_drop_prob'):
            if not 0 <= self.config[key] <= 1:
                raise ValueError(f'Invalid {key}: {self.config[key]}')
        if decoder_temporal_kernel < 0 or (decoder_temporal_kernel and decoder_temporal_kernel % 2 == 0):
            raise ValueError('decoder_temporal_kernel must be zero or a positive odd number')
        if cross_limb_dim < 0 or not 0 <= cross_limb_last_n <= min(encoder_layers, decoder_layers):
            raise ValueError('Invalid cross-limb configuration')
        if temporal_rope and any((dim // heads) % 2 for dim in (encoder_width, decoder_width)):
            raise ValueError('Temporal RoPE requires an even head dimension')
        self.variational = variational
        self.temporal_rope, self.layerwise_latent = temporal_rope, layerwise_latent
        self.width, self.slots, self.latent_dim = encoder_width, slots, latent_dim
        self.encoder_width, self.decoder_width = encoder_width, decoder_width
        width = encoder_width
        self.motion_projection = nn.Linear(len(MODEL_CHANNELS), width)
        self.names = nn.Linear(t5_out_dim, width)
        self.geometry = nn.Linear(REST_GEOM_DIM, width)
        self.structure = nn.Linear(JOINT_STRUCT_DIM, width)

        def trunk(decoder=False):
            width = decoder_width if decoder else encoder_width
            ff_size = decoder_ff_size if decoder else encoder_ff_size
            return GraphMotionDecoder(
                GraphMotionDecoderLayer(width, heads, ff_size or width * 4, dropout_prob, 'gelu'),
                decoder_layers if decoder else encoder_layers,
                cross_limb=cross_limb_dim > 0, cross_limb_dim=cross_limb_dim or 64,
                cross_limb_last_n=cross_limb_last_n, graph_relations=True,
                last_layer_ff=last_layer_ff if decoder else 0, temporal_rope=temporal_rope,
                mirror_twin_drop_prob=mirror_twin_drop_prob, lateral_drop_prob=lateral_drop_prob)

        self.encoder = trunk()
        self.decoder = trunk(decoder=True)
        if layerwise_latent:
            for layer in self.decoder.layers:
                layer.enable_motion_latent(latent_dim)
        self.queries = nn.Parameter(torch.randn(slots, width) * width ** -0.5)
        self.pool = nn.MultiheadAttention(width, heads, dropout=dropout_prob, batch_first=True)
        self.to_latent = nn.Linear(width, latent_dim)
        self.from_latent = nn.Linear(latent_dim, decoder_width)
        self.read_latent = nn.MultiheadAttention(decoder_width, heads, dropout=dropout_prob, batch_first=True)
        self.output = nn.Linear(decoder_width, len(MODEL_CHANNELS))
        # As in AnyTop: physical frame times for open motion, cycle length for
        # loops. Shared by E/D; zero-init heads start as a no-op.
        self.tempo_projection = _zero_init_mlp(width)
        self.loop_period_projection = _zero_init_mlp(width)
        # Append after all shared initialization to keep the baseline weights identical.
        self.temporal_refinement = (TemporalResidualConv(decoder_width, decoder_temporal_kernel)
                                    if decoder_temporal_kernel else None)
        self.to_logvar = nn.Linear(width, latent_dim) if variational else None
        if self.to_logvar is not None:
            nn.init.zeros_(self.to_logvar.weight)
            nn.init.zeros_(self.to_logvar.bias)

        # Equal widths share skeleton and physical-time projections.
        if decoder_width != encoder_width:
            self.decoder_names = nn.Linear(t5_out_dim, decoder_width)
            self.decoder_geometry = nn.Linear(REST_GEOM_DIM, decoder_width)
            self.decoder_structure = nn.Linear(JOINT_STRUCT_DIM, decoder_width)
            self.decoder_tempo_projection = _zero_init_mlp(decoder_width)
            self.decoder_loop_period_projection = _zero_init_mlp(decoder_width)

    def _static(self, skeleton, decoder=False):
        if decoder and self.decoder_width != self.encoder_width:
            return (self.decoder_names(skeleton['joints_names_embs'])
                    + self.decoder_geometry(skeleton['rest_geom'])
                    + self.decoder_structure(skeleton['joint_struct']))
        return (self.names(skeleton['joints_names_embs'])
                + self.geometry(skeleton['rest_geom'])
                + self.structure(skeleton['joint_struct']))

    def _position(self, frame, loop_metadata):
        batch, frames = frame.shape
        positions = torch.arange(frames, device=frame.device).view(1, -1, 1)
        absolute = create_sin_embedding(positions, self.width).expand(batch, -1, -1)
        metadata = loop_metadata or {}
        is_loop = metadata.get('is_loop', torch.zeros(batch, device=frame.device, dtype=torch.bool))
        cycles = metadata.get('loop_tile_count', torch.ones(batch, device=frame.device))
        phase = (2 * torch.pi * positions * cycles[:, None, None]
                 / frame.sum(-1)[:, None, None])
        phase = phase * torch.arange(1, self.width // 2 + 1, device=frame.device)
        circular = torch.cat((phase.sin(), phase.cos()), dim=-1)
        return torch.where(is_loop[:, None, None], circular, absolute).transpose(0, 1)[:, :, None]

    def _rope(self, frame, metadata, width=None):
        """Per-sample open/closed RoPE, independent of right-padded window size."""
        width = self.width if width is None else width
        pairs = width // self.config['heads'] // 2
        ramp = torch.arange(pairs, device=frame.device) / max(pairs - 1, 1)
        frequency = TEMPORAL_ROPE_MAX_FREQ * (
            (2 * torch.pi / TEMPORAL_ROPE_MAX_PERIOD) / TEMPORAL_ROPE_MAX_FREQ) ** ramp
        position = torch.arange(frame.shape[1], device=frame.device)[None, :, None]
        length = frame.sum(-1)[:, None, None]
        turns = torch.round(frequency * length / (2 * torch.pi)).clamp_min(1)
        turns = torch.minimum(turns, (length // 2).clamp_min(1))
        closed = ((position * turns) % length) * (2 * torch.pi / length)
        is_loop = metadata.get('is_loop', torch.zeros(frame.shape[0], device=frame.device, dtype=torch.bool))
        angle = torch.where(is_loop[:, None, None], closed, position * frequency)
        return angle.cos(), angle.sin()

    def _tempo_embedding(self, frame, metadata, decoder=False):
        """[T,B,1,D] physical time, matching AnyTop's tempo semantics.

        Use valid lengths rather than padded tensor width. A missing speed in
        the low-level API means native time (L=N); training/eval pass it explicitly.
        """
        metadata = metadata or {}
        width = self.decoder_width if decoder else self.encoder_width
        separate = decoder and self.decoder_width != self.encoder_width
        tempo_projection = self.decoder_tempo_projection if separate else self.tempo_projection
        period_projection = self.decoder_loop_period_projection if separate else self.loop_period_projection
        batch = frame.shape[0]

        def per_sample(key, default, dtype):
            value = torch.as_tensor(metadata.get(key, default), device=frame.device,
                                    dtype=dtype).reshape(-1)
            if value.numel() == 1:
                value = value.expand(batch)
            elif value.numel() != batch:
                raise ValueError(f'{key} batch dimension must match the motion batch size')
            return value

        speed = per_sample('resample_speed_cond', 1.0, torch.float32)
        cycles = per_sample('loop_tile_count', 1, torch.float32)
        is_loop = per_sample('is_loop', False, torch.bool)
        if not torch.compiler.is_compiling():
            check_on_device((torch.isfinite(speed) & (speed > 0)).all(),
                            'resample_speed_cond must be finite and positive')
            check_on_device((torch.isfinite(cycles) & (cycles >= 1)
                             & (cycles == cycles.round())).all(),
                            'loop_tile_count must contain positive integers')
        length = frame.sum(-1).float()
        source_frames = speed * length
        step = torch.where(length > 1,
                           (source_frames.clamp_min(1) - 1) / (length - 1).clamp_min(1),
                           torch.ones_like(speed))
        positions = torch.arange(frame.shape[1], device=frame.device, dtype=torch.float32)
        times = positions[None, :, None] * step[:, None, None]
        open_emb = run_in_fp32(tempo_projection, create_sin_embedding(times, width))
        period = (source_frames / cycles)[:, None, None]
        loop_emb = run_in_fp32(period_projection, create_sin_embedding(period, width))
        tempo = torch.where(is_loop[:, None, None], loop_emb, open_emb)
        return tempo.masked_fill(~frame[:, :, None], 0).transpose(0, 1)[:, :, None]

    def _trunk(self, module, tokens, skeleton, joint, frame, loop_metadata, z=None):
        # No diffusion timestep. A fixed zero vector only satisfies the reused layer API.
        graph = {k: skeleton[k] for k in ('graph_dist', 'joints_relations')}
        metadata = loop_metadata or {}
        tokens = tokens + self._tempo_embedding(frame, metadata, decoder=module is self.decoder).to(tokens.dtype)
        rope = self._rope(frame, metadata, tokens.shape[-1]) if self.temporal_rope else None
        phase = None
        # Only the encoder receives phase as content. Decoder uses RoPE and
        # physical tempo above, without an extra absolute/circular phase table.
        if self.temporal_rope and module is self.encoder:
            is_loop = metadata.get('is_loop', torch.zeros(frame.shape[0], device=frame.device, dtype=torch.bool))
            phase = self._position(frame, metadata).squeeze(2) * is_loop[None, :, None]
        return module(tokens, tokens.new_zeros(tokens.shape[1], tokens.shape[-1]), None,
                      tgt_key_padding_mask=~joint, temporal_key_padding_mask=~frame,
                      y=graph, motion_z=z, temporal_rope_override=rope,
                      loop_phase_override=phase)

    def encode_posterior(self, x, skeleton, frame_mask=None, loop_metadata=None):
        if x.shape[2] != len(MODEL_CHANNELS):
            raise ValueError('AE expects the 9 MODEL_CHANNELS, without velocity')
        joint, frame = motion_masks(x, skeleton, frame_mask)
        valid = joint[:, :, None, None] & frame[:, None, None, :]
        x = x.masked_fill(~valid, 0)
        tokens = self.motion_projection(x.permute(3, 0, 1, 2))
        tokens = tokens + self._static(skeleton)[None] + self._position(frame, loop_metadata)
        tokens = self._trunk(self.encoder, tokens, skeleton, joint, frame, loop_metadata)
        frames, batch, joints, width = tokens.shape
        values = tokens.reshape(frames * batch, joints, width)
        queries = self.queries[None].expand(frames * batch, -1, -1)
        mask = (~joint)[None].expand(frames, -1, -1).reshape(frames * batch, joints)
        pooled, _ = self.pool(queries, values, values, key_padding_mask=mask, need_weights=False)
        z = self.to_latent(pooled).reshape(frames, batch, self.slots, self.latent_dim)
        mu = z.permute(1, 0, 2, 3).masked_fill(~frame[:, :, None, None], 0).contiguous()
        logvar = None
        if self.variational:
            # Keep variance math in FP32 under AMP; bound the numerical range.
            logvar = self.to_logvar(pooled).float().clamp(-20, 10)
            logvar = logvar.reshape(frames, batch, self.slots, self.latent_dim).permute(1, 0, 2, 3)
            logvar = logvar.masked_fill(~frame[:, :, None, None], 0).contiguous()
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar, frame_mask=None):
        z = mu.float() + torch.exp(0.5 * logvar.float()) * torch.randn_like(mu, dtype=torch.float32)
        if frame_mask is not None:
            z = z.masked_fill(~frame_mask[:, :, None, None], 0)
        return z

    def encode(self, x, skeleton, frame_mask=None, loop_metadata=None, sample=False):
        """Return posterior mean by default; sample=True draws a VAE latent."""
        mu, logvar = self.encode_posterior(x, skeleton, frame_mask, loop_metadata)
        return self.reparameterize(mu, logvar, frame_mask) if sample and self.variational else mu

    def decode(self, z, skeleton, frame_mask=None, loop_metadata=None):
        batch, latent_frames, slots, dim = z.shape
        if (slots, dim) != (self.slots, self.latent_dim):
            raise ValueError('Latent slot layout does not match this AE')
        static = self._static(skeleton, decoder=True)
        joints = static.shape[1]
        frames = latent_frames
        if frame_mask is not None and frame_mask.shape[1] != frames:
            raise ValueError('Latent time length does not match output frame mask')
        shape = z.new_empty(batch, joints, len(MODEL_CHANNELS), frames)
        joint, frame = motion_masks(shape, skeleton, frame_mask)
        z = z.masked_fill(~frame[:, :, None, None], 0)
        queries = static[None].expand(frames, -1, -1, -1)
        values = self.from_latent(z).transpose(0, 1).reshape(frames * batch, slots, self.decoder_width)
        read, _ = self.read_latent(queries.reshape(frames * batch, joints, self.decoder_width),
                                   values, values, need_weights=False)
        tokens = queries + read.reshape(frames, batch, joints, self.decoder_width)
        tokens = self._trunk(self.decoder, tokens, skeleton, joint, frame, loop_metadata,
                             z if self.layerwise_latent else None)
        if self.temporal_refinement is not None:
            tokens = self.temporal_refinement(tokens, frame, loop_metadata)
        out = self.output(tokens).permute(1, 2, 3, 0)
        return out.masked_fill(~(joint[:, :, None, None] & frame[:, None, None, :]), 0)

    def forward(self, x, skeleton, frame_mask=None, loop_metadata=None, return_posterior=False):
        if frame_mask is None:
            frame_mask = torch.ones(x.shape[0], x.shape[-1], device=x.device, dtype=torch.bool)
        mu, logvar = self.encode_posterior(x, skeleton, frame_mask, loop_metadata)
        z = self.reparameterize(mu, logvar, frame_mask) if self.variational and self.training else mu
        prediction = self.decode(z, skeleton, frame_mask, loop_metadata)
        return (prediction, mu, logvar) if return_posterior else prediction


def diagonal_gaussian_kl(mu, logvar, frame_mask):
    """KL(q(z|x) || N(0,I)), in nats per latent scalar, averaged per sample."""
    terms = 0.5 * (mu.float().square() + logvar.float().exp() - 1 - logvar.float())
    per_frame = terms.mean(dim=(-1, -2)).masked_fill(~frame_mask, 0)
    return (per_frame.sum(-1) / frame_mask.sum(-1).clamp_min(1)).mean()


def to_working_motion(x):
    """Restore the current 12-channel working container with zero velocity."""
    if x.shape[2] != 9:
        raise ValueError('Expected 9-channel canonical motion')
    return torch.cat((x, x.new_zeros(x.shape[0], x.shape[1], 3, x.shape[-1])), dim=2)
