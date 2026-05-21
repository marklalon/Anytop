import torch
torch.cuda.empty_cache()
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.motion_transformer import GraphMotionDecoderLayer, GraphMotionDecoder, SelectiveMultiheadAttention
from model.joint_mask_utils import sample_subtree_joint_mask_batch


def create_sin_embedding(positions: torch.Tensor, dim: int, max_period: float = 10000,
                         dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create sinusoidal positional embedding, with shape `[B, T, C]`.

    Args:
        positions (torch.Tensor): LongTensor of positions.
        dim (int): Dimension of the embedding.
        max_period (float): Maximum period of the cosine/sine functions.
        dtype (torch.dtype or str): dtype to use to generate the embedding.
    Returns:
        torch.Tensor: Sinusoidal positional embedding.
    """
    # We aim for BTC format
    assert dim % 2 == 0
    half_dim = dim // 2
    positions = positions.to(dtype)
    adim = torch.arange(half_dim, device=positions.device, dtype=dtype).view(1, 1, -1)
    max_period_tensor = torch.full([], max_period, device=positions.device, dtype=dtype)  # avoid sync point
    phase = positions / (max_period_tensor ** (adim / (half_dim - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)

class AnyTop(nn.Module):
    def __init__(self, max_joints, feature_len,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 activation="gelu", t5_out_dim = 512, root_input_feats=13,
                 **kargs):
        super().__init__()

        self.max_joints = max_joints
        self.feature_len = feature_len
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = self.feature_len
        self.root_input_feats = root_input_feats
        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.skip_t5=kargs.get('skip_t5', False)
        self.value_emb=kargs.get('value_emb', False)
        self.cross_limb=kargs.get('cross_limb', True)
        self.cross_limb_latents=kargs.get('cross_limb_latents', 8)
        self.cross_limb_dim=kargs.get('cross_limb_dim', 64)
        self.cross_limb_last_n=kargs.get('cross_limb_last_n', 0)
        self.joint_mask_prob=float(kargs.get('joint_mask_prob', 0.0))
        self.reference_cond=bool(kargs.get('reference_cond', False))
        self.reference_encoder_layers=int(kargs.get('reference_encoder_layers', 2))
        self.reference_uncond_prob=float(kargs.get('reference_uncond_prob', 0.25))
        self.reference_clean_prob=float(kargs.get('reference_clean_prob', 0.1))
        self.reference_subtree_prob=float(kargs.get('reference_subtree_prob', 0.35))
        self.reference_subtree_budget=float(kargs.get('reference_subtree_budget', 0.35))
        self.reference_noise_prob=float(kargs.get('reference_noise_prob', 0.75))
        self.reference_noise_sigma_min=float(kargs.get('reference_noise_sigma_min', 0.02))
        self.reference_noise_sigma_max=float(kargs.get('reference_noise_sigma_max', 0.12))
        self.reference_jitter_prob=float(kargs.get('reference_jitter_prob', 0.35))
        self.reference_hold_prob=float(kargs.get('reference_hold_prob', 0.25))
        self.reference_smooth_prob=float(kargs.get('reference_smooth_prob', 0.25))
        if not 0.0 <= self.joint_mask_prob <= 1.0:
            raise ValueError(f"joint_mask_prob must be in [0, 1], got {self.joint_mask_prob}")
        if self.reference_encoder_layers < 0:
            raise ValueError(
                f"reference_encoder_layers must be >= 0, got {self.reference_encoder_layers}"
            )
        for prob_name in (
            'reference_uncond_prob',
            'reference_clean_prob',
            'reference_subtree_prob',
            'reference_subtree_budget',
            'reference_noise_prob',
            'reference_jitter_prob',
            'reference_hold_prob',
            'reference_smooth_prob',
        ):
            prob_value = float(getattr(self, prob_name))
            if not 0.0 <= prob_value <= 1.0:
                raise ValueError(f"{prob_name} must be in [0, 1], got {prob_value}")
        if self.reference_noise_sigma_min < 0.0 or self.reference_noise_sigma_max < self.reference_noise_sigma_min:
            raise ValueError(
                "reference_noise_sigma_min/max must satisfy 0 <= min <= max"
            )
        self.input_process = InputProcess(self.input_feats, self.root_input_feats, self.latent_dim, t5_out_dim, skip_t5=self.skip_t5, dropout_prob=self.dropout)
        if self.reference_cond:
            self.reference_encoder = ReferenceEncoder(
                input_feats=self.input_feats,
                root_input_feats=self.root_input_feats,
                latent_dim=self.latent_dim,
                ff_size=self.ff_size,
                num_heads=self.num_heads,
                dropout=self.dropout,
                t5_out_dim=t5_out_dim,
                skip_t5=self.skip_t5,
                num_layers=self.reference_encoder_layers,
            )
        else:
            self.reference_encoder = None

        seqTransDecoderLayer = GraphMotionDecoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)
        self.seqTransDecoder = GraphMotionDecoder(seqTransDecoderLayer,
                                                        num_layers=self.num_layers, value_emb=self.value_emb,
                                                        cross_limb=self.cross_limb,
                                                        cross_limb_latents=self.cross_limb_latents,
                                                        cross_limb_dim=self.cross_limb_dim,
                                                        cross_limb_last_n=self.cross_limb_last_n)
            
        
        self.output_process = OutputProcess(self.feature_len, self.root_input_feats, self.max_joints, self.latent_dim)

    @staticmethod
    def _build_joint_key_padding_mask(njoints, n_joints, device):
        """Return the padding-only joint key mask used by attention.

        Training-time subtree perturbation deliberately stays out of the
        attention masks. Only structurally padded joints are masked here.
        """
        return torch.arange(njoints, device=device)[None, :] >= n_joints[:, None]

    @staticmethod
    def _build_reference_key_padding_mask(nframes, njoints, lengths, device):
        clipped_lengths = torch.clamp(lengths.to(device=device, dtype=torch.int64), min=0, max=nframes)
        frame_positions = torch.arange(nframes, device=device)[None, :]
        valid_frames = frame_positions < clipped_lengths[:, None]
        valid_with_tpose = torch.cat(
            [torch.ones((clipped_lengths.shape[0], 1), device=device, dtype=torch.bool), valid_frames],
            dim=1,
        )
        key_padding_mask = ~valid_with_tpose
        return key_padding_mask.unsqueeze(1).expand(-1, njoints, -1).reshape(-1, nframes + 1)

    def supports_reference_conditioning(self):
        return self.reference_cond and self.reference_encoder is not None

    def sample_subtree_joint_mask_train(self, y, njoints, device):
        """Select subtrees of joints to perturb during training (governed by
        ``joint_mask_prob``).

        Returns a bool tensor of shape ``[B, njoints]`` (True = joint selected)
        or ``None`` if no joint was selected, or if not in training mode, or
        if ``joint_mask_prob == 0`` -- so eval-mode loss reports a clean
        diffusion objective.

        Called from ``GaussianDiffusion.training_losses`` AFTER ``q_sample``
        to decide which joints' x_t slice should be re-noised with an
        independent random timestep and fresh noise, so that those joints'
        noise level disagrees with the rest of the batch sample. This trains
        the cross-joint pathway to denoise robustly against per-joint
        timestep mismatch -- the regime RePaint clamping produces at
        inference. The model's forward itself stays vanilla.
        """
        if (not self.training) or self.joint_mask_prob <= 0.0:
            return None
        n_joints_cpu = torch.as_tensor(y['n_joints'], device='cpu', dtype=torch.int64).reshape(-1)
        return self._sample_subtree_joint_mask(y, n_joints_cpu, njoints, device)

    def _sample_subtree_joint_mask(self, y, n_joints, njoints, device):
        parents_batch = y.get('parents')
        if parents_batch is None:
            return None
        candidate_roots_batch = y.get('joint_mask_candidate_roots')
        if torch.is_tensor(n_joints):
            n_joints_np = n_joints.detach().to(device='cpu', dtype=torch.int64).numpy()
        else:
            n_joints_np = np.asarray(n_joints, dtype=np.int64)

        if torch.is_tensor(parents_batch):
            parents_batch_np = parents_batch.detach().to(device='cpu', dtype=torch.int64).numpy()
        else:
            parents_batch_np = [np.asarray(parents, dtype=np.int64) for parents in parents_batch]

        if candidate_roots_batch is None:
            candidate_roots_np = None
        elif torch.is_tensor(candidate_roots_batch):
            candidate_roots_np = candidate_roots_batch.detach().to(device='cpu').numpy()
        else:
            candidate_roots_np = np.asarray(candidate_roots_batch, dtype=np.bool_)

        subtree_joint_mask_np = sample_subtree_joint_mask_batch(
            parents_batch=parents_batch_np,
            candidate_root_mask_batch=candidate_roots_np,
            n_joints=n_joints_np,
            max_joints=njoints,
            joint_mask_prob=self.joint_mask_prob,
            rng=np.random,
        )
        if subtree_joint_mask_np is None:
            return None
        return torch.from_numpy(subtree_joint_mask_np).to(device=device)

    def forward(self, x, timesteps, get_layer_activation=-1, y=None, train_step=None, **unused_kwargs):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """

        joints_padding_mask = y['joints_padding_mask'].to(x.device)
        temp_mask = y['mask'].to(x.device)
        tpos_first_frame = y['tpos_first_frame'].to(x.device).unsqueeze(0)

        bs, njoints, nfeats, nframes = x.shape
        n_joints = torch.as_tensor(y['n_joints'], device=x.device).reshape(-1)
        joint_key_padding_mask = self._build_joint_key_padding_mask(njoints, n_joints, x.device)
        reference_memory = None
        reference_key_padding_mask = None
        reference_batch_mask = None
        # joint_mask_prob-driven subtree perturbation is applied OUTSIDE this
        # forward, in diffusion.training_losses, by re-noising the selected
        # joints' x_t with q_sample(x_0, t_random, fresh_noise). The model
        # itself stays vanilla -- selected joints DO NOT enter the
        # key-padding masks and continue to participate in attention normally,
        # just with mismatched noise levels, which trains the
        # cross-joint pathway to denoise robustly against per-joint timestep
        # disagreement (matching RePaint clamp behavior at inference).
        timesteps_emb = create_sin_embedding(timesteps.view(1, -1, 1), self.latent_dim)[0]

        x = self.input_process(x, tpos_first_frame, y['joints_names_embs']) # applies linear layer on each frame to convert it to latent dim
        spatial_mask = (1.0 - joints_padding_mask[:, 0, 0, 1:, 1:].float()) * -1e4
        spatial_mask = spatial_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        temporal_template = (1.0 - temp_mask.reshape(bs, -1, nframes + 1, nframes + 1)[:, :1].float()) * -1e4
        temporal_template = temporal_template.expand(-1, self.num_heads, -1, -1)
        temporal_mask = temporal_template.unsqueeze(1).expand(-1, njoints, -1, -1, -1).reshape(-1, nframes + 1, nframes + 1)

        # Cross-limb temporal pathway needs (1) the windowed temporal mask
        # WITHOUT the per-joint repeat -> (bs*H, T, T), which the block expands
        # per-latent itself, and (2) a (bs, njoints) bool key-padding mask
        # (True == padded) derived from the real joint count.
        if self.cross_limb:
            assert 'n_joints' in y, "cross_limb requires y['n_joints'] in the batch"
            temporal_template = temporal_template.reshape(-1, nframes + 1, nframes + 1)
            y['joints_key_padding_mask'] = joint_key_padding_mask
        else:
            temporal_template = None

        if self.reference_cond:
            reference_motion = y.get('reference_motion')
            raw_reference_batch_mask = y.get('reference_cond_mask')
            if raw_reference_batch_mask is not None:
                reference_batch_mask = raw_reference_batch_mask.to(device=x.device, dtype=torch.bool).reshape(bs)
            if reference_motion is not None and (reference_batch_mask is None or bool(reference_batch_mask.any())):
                reference_motion = reference_motion.to(device=x.device, dtype=x.dtype)
                if reference_motion.shape[:3] != (bs, njoints, nfeats):
                    raise ValueError(
                        "y['reference_motion'] must have shape "
                        f"(B, J, F, T) matching x except for T, got {tuple(reference_motion.shape)}"
                    )
                if reference_motion.shape[-1] != nframes:
                    raise ValueError(
                        "y['reference_motion'] must match x in frame count, got "
                        f"{reference_motion.shape[-1]} and {nframes}"
                    )
                reference_key_padding_mask = self._build_reference_key_padding_mask(
                    nframes,
                    njoints,
                    y['lengths'],
                    x.device,
                )
                reference_memory = self.reference_encoder(
                    reference_motion,
                    tpos_first_frame,
                    y['joints_names_embs'],
                    reference_key_padding_mask,
                )

        cross_limb_unreliable_mask = None
        if self.cross_limb:
            raw_cross_limb_unreliable_mask = y.get('cross_limb_unreliable_mask')
            if raw_cross_limb_unreliable_mask is not None:
                cross_limb_unreliable_mask = raw_cross_limb_unreliable_mask.to(device=x.device, dtype=x.dtype)
                raw_expected_shape = (bs, nframes, njoints)
                prepared_expected_shape = (nframes + 1, bs, njoints)
                if cross_limb_unreliable_mask.shape == raw_expected_shape:
                    reliable_tpose = torch.zeros((bs, 1, njoints), device=x.device, dtype=x.dtype)
                    cross_limb_unreliable_mask = torch.cat([reliable_tpose, cross_limb_unreliable_mask], dim=1)
                    cross_limb_unreliable_mask = cross_limb_unreliable_mask.transpose(0, 1).contiguous()
                elif cross_limb_unreliable_mask.shape != prepared_expected_shape:
                    raise ValueError(
                        "y['cross_limb_unreliable_mask'] must have shape "
                        f"{raw_expected_shape} or {prepared_expected_shape}, got "
                        f"{tuple(cross_limb_unreliable_mask.shape)}"
                    )

        output = self.seqTransDecoder(
            tgt=x,
            timesteps_embs=timesteps_emb,
            memory=None,
            spatial_mask=spatial_mask,
            temporal_mask=temporal_mask,
            tgt_key_padding_mask=joint_key_padding_mask,
            y=y,
            get_layer_activation=get_layer_activation,
            reference_memory=reference_memory,
            reference_key_padding_mask=reference_key_padding_mask,
            temporal_template=temporal_template,
            cross_limb_unreliable_mask=cross_limb_unreliable_mask,
            reference_batch_mask=reference_batch_mask,
        )
        if get_layer_activation > -1 and get_layer_activation < self.num_layers:
            activations = output[1]
            output=output[0]
        output = self.output_process(output) # Applies linear layer on each frame to convert it back to feature len dim
        if get_layer_activation > -1 and get_layer_activation < self.num_layers:
            return output, activations
        return output


    def _apply(self, fn):
        super()._apply(fn)


    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)

# in the case of GMDM, the input process is as follows: 
# embed each joint of each frame of each motion in batch by the same MLP, separately ! 
class InputProcess(nn.Module):
    def __init__(self, input_feats, root_input_feats, latent_dim, t5_output_dim, skip_t5=False, dropout_prob=0):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.root_input_feats = root_input_feats
        self.root_embedding = nn.Linear(self.root_input_feats, self.latent_dim)
        self.tpos_root_embedding = nn.Linear(self.root_input_feats, self.latent_dim)
        self.joint_embedding = nn.Linear(self.input_feats, self.latent_dim)
        self.tpos_joint_embedding = nn.Linear(self.input_feats, self.latent_dim)
        self.skip_t5=skip_t5
        if not self.skip_t5:
            self.joints_names_dropout = nn.Dropout(p=dropout_prob)
            self.text_embedding = nn.Linear(t5_output_dim, self.latent_dim)
    def forward(self, x, tpos_first_frame, joints_embedded_names):
        # x.shape = [batch_size, joints, 13, frames]
        x = x.permute(3, 0, 1, 2) # [frames, batch_size, n_joints, features_len]
        tpos_all_joints_except_root = self.tpos_joint_embedding(tpos_first_frame[:, :, 1:])
        tpos_root_data = self.tpos_root_embedding(tpos_first_frame[:, :, 0:1])
        all_joints_except_root = self.joint_embedding(x[:, :, 1:])
        root_data = self.root_embedding(x[:, :, 0:1])
        tpos_embedded = torch.cat([tpos_root_data, tpos_all_joints_except_root], dim=2)
        x_embedded = torch.cat([root_data, all_joints_except_root], dim=2)
        x = torch.cat([tpos_embedded, x_embedded], dim=0)
        if not self.skip_t5:
            joints_embedded_names = self.text_embedding(self.joints_names_dropout(joints_embedded_names.to(x.device)))
            x = x + joints_embedded_names[None, ...]# [frames, batch_size, n_joints, d]
        positions = torch.arange(x.shape[0], device=x.device).view(1, -1, 1).repeat(x.shape[1], 1, 1)
        pos_emb = create_sin_embedding(positions, self.latent_dim)[0]
        return x + pos_emb.unsqueeze(1).unsqueeze(1)


class ReferenceTemporalEncoderLayer(nn.Module):
    def __init__(self, latent_dim, ff_size, num_heads, dropout):
        super().__init__()
        self.self_attn = SelectiveMultiheadAttention(latent_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, latent_dim),
        )

    def forward(self, x, key_padding_mask=None):
        frames, bs, njoints, feats = x.shape
        residual = x
        x_norm = self.norm1(x)
        attn_input = x_norm.reshape(frames, bs * njoints, feats)
        attn_output, _ = self.self_attn(
            attn_input,
            attn_input,
            attn_input,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        attn_output = attn_output.reshape(frames, bs, njoints, feats)
        x = residual + self.dropout1(attn_output)
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x


class ReferenceEncoder(nn.Module):
    def __init__(
        self,
        input_feats,
        root_input_feats,
        latent_dim,
        ff_size,
        num_heads,
        dropout,
        t5_out_dim,
        skip_t5,
        num_layers,
    ):
        super().__init__()
        self.input_process = InputProcess(
            input_feats,
            root_input_feats,
            latent_dim,
            t5_out_dim,
            skip_t5=skip_t5,
            dropout_prob=dropout,
        )
        self.layers = nn.ModuleList([
            ReferenceTemporalEncoderLayer(
                latent_dim=latent_dim,
                ff_size=ff_size,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, reference_motion, tpos_first_frame, joints_embedded_names, key_padding_mask=None):
        output = self.input_process(reference_motion, tpos_first_frame, joints_embedded_names)
        for layer in self.layers:
            output = layer(output, key_padding_mask=key_padding_mask)
        return self.norm(output)

class OutputProcess(nn.Module):
    def __init__(self, feature_len, root_feature_len, max_joints, latent_dim):
        super().__init__()
        self.feature_len = feature_len
        self.max_joints = max_joints
        self.latent_dim = latent_dim
        self.root_feature_len = root_feature_len
        self.root_dembedding = nn.Linear(self.latent_dim, self.root_feature_len)
        self.joint_dembedding = nn.Linear(self.latent_dim, self.feature_len)

    def forward(self, output):
        # output shape [frames, batch_size, joints, latent_dim]
        root_data = self.root_dembedding(output[:, :, 0])
        all_joints = self.joint_dembedding(output[:, :, 1:])
        output = torch.cat([root_data.unsqueeze(2), all_joints], dim=-2)
        output = output.permute(1, 2, 3, 0)[..., 1:]  # [bs, njoints, nfeats, nframes]
        return output


