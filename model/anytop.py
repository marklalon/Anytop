import torch
torch.cuda.empty_cache()
import torch.nn as nn
import numpy as np
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
        self.reference_encoder_layers=int(kargs.get('reference_encoder_layers', 1))
        self.reference_cond_prob=float(kargs.get('reference_cond_prob', 0.2))
        self.reference_residual_gate=float(kargs.get('reference_residual_gate', 1.0))
        self.reference_token_dropout_prob=float(kargs.get('reference_token_dropout_prob', 0.0))
        self.reference_token_noise_std=float(kargs.get('reference_token_noise_std', 0.0))
        if not 0.0 <= self.joint_mask_prob <= 1.0:
            raise ValueError(f"joint_mask_prob must be in [0, 1], got {self.joint_mask_prob}")
        if self.reference_encoder_layers < 0:
            raise ValueError(
                f"reference_encoder_layers must be >= 0, got {self.reference_encoder_layers}"
            )
        if self.reference_residual_gate < 0.0:
            raise ValueError(
                f"reference_residual_gate must be >= 0, got {self.reference_residual_gate}"
            )
        for prob_name in (
            'reference_cond_prob',
            'reference_token_dropout_prob',
        ):
            prob_value = float(getattr(self, prob_name))
            if not 0.0 <= prob_value <= 1.0:
                raise ValueError(f"{prob_name} must be in [0, 1], got {prob_value}")
        if self.reference_token_noise_std < 0.0:
            raise ValueError(
                f"reference_token_noise_std must be >= 0, got {self.reference_token_noise_std}"
            )
        self.input_process = InputProcess(self.input_feats, self.root_input_feats, self.latent_dim, t5_out_dim, skip_t5=self.skip_t5, dropout_prob=self.dropout)
        if self.reference_cond:
            self.reference_encoder = ReferencePriorEncoder(
                max_joints=self.max_joints,
                input_feats=self.input_feats,
                latent_dim=self.latent_dim,
                ff_size=self.ff_size,
                num_heads=self.num_heads,
                dropout=self.dropout,
                t5_out_dim=t5_out_dim,
                skip_t5=self.skip_t5,
                num_layers=self.reference_encoder_layers,
                token_dropout_prob=self.reference_token_dropout_prob,
                token_noise_std=self.reference_token_noise_std,
            )
        else:
            self.reference_encoder = None

        seqTransDecoderLayer = GraphMotionDecoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation,
                                                            reference_residual_gate=self.reference_residual_gate)
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
                if reference_motion.shape[0] != bs or reference_motion.shape[2] != nfeats:
                    raise ValueError(
                        "y['reference_motion'] must have shape (B, J, F, T) with matching batch/feature dims, got "
                        f"{tuple(reference_motion.shape)}"
                    )
                reference_n_joints = y.get('reference_n_joints', y['n_joints'])
                reference_lengths = y.get('reference_lengths', y['lengths'])
                reference_translation_root_index = y.get(
                    'reference_translation_root_index',
                    y.get('translation_root_index'),
                )
                if reference_translation_root_index is None:
                    raise ValueError("reference conditioning requires translation_root_index metadata")
                reference_joints_names_embs = y.get(
                    'reference_joints_names_embs',
                    y.get('joints_names_embs'),
                )
                if reference_joints_names_embs is None:
                    raise ValueError("reference conditioning requires joint-name embeddings")
                reference_memory = self.reference_encoder(
                    reference_motion,
                    n_joints=reference_n_joints,
                    lengths=reference_lengths,
                    translation_root_index=reference_translation_root_index,
                    joints_embedded_names=reference_joints_names_embs,
                )
                reference_key_padding_mask = None

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


class ReferencePriorTemporalLayer(nn.Module):
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
        x_norm = self.norm1(x)
        attn_output, _ = self.self_attn(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout1(attn_output)
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x


class ReferencePriorEncoder(nn.Module):
    def __init__(
        self,
        max_joints,
        input_feats,
        latent_dim,
        ff_size,
        num_heads,
        dropout,
        t5_out_dim,
        skip_t5,
        num_layers,
        num_groups=6,
        num_prior_tokens=8,
        token_dropout_prob=0.0,
        token_noise_std=0.0,
    ):
        super().__init__()
        self.input_feats = int(input_feats)
        if self.input_feats < 13:
            raise ValueError(
                "ReferencePriorEncoder expects at least the 13-dim feature schema "
                "[pos(3), rot6d(6), vel(3), contact(1)], "
                f"got input_feats={self.input_feats}"
            )
        self.latent_dim = int(latent_dim)
        self.max_joints = int(max_joints)
        if self.max_joints <= 0:
            raise ValueError(f"max_joints must be > 0, got {self.max_joints}")
        self.num_groups = int(num_groups)
        self.num_prior_tokens = int(num_prior_tokens)
        self.skip_t5 = bool(skip_t5)
        self.token_dropout_prob = float(token_dropout_prob)
        self.token_noise_std = float(token_noise_std)
        if not 0.0 <= self.token_dropout_prob <= 1.0:
            raise ValueError(f"token_dropout_prob must be in [0, 1], got {self.token_dropout_prob}")
        if self.token_noise_std < 0.0:
            raise ValueError(f"token_noise_std must be >= 0, got {self.token_noise_std}")
        # Group assignment should be phase-invariant and only summarize each
        # joint's coarse semantics/motion role across the clip.
        self.joint_motion_feature_dim = 4
        self.joint_motion_stat_dim = self.joint_motion_feature_dim * 2
        # Global/group summaries stay phase-invariant; signed limb phase is
        # fused into a direct per-joint semantic branch below.
        self.group_motion_feature_dim = self.joint_motion_feature_dim
        self.group_motion_stat_dim = self.group_motion_feature_dim * 2
        self.phase_joint_feature_dim = 4
        self.phase_joint_stat_dim = self.phase_joint_feature_dim * 3
        self.group_feature_dim = self.group_motion_stat_dim
        self.global_motion_feature_dim = self.group_motion_stat_dim
        self.per_joint_prior_dim = self.phase_joint_stat_dim
        prior_input_dim = (
            self.global_motion_feature_dim
            + self.num_groups * self.group_feature_dim
            + self.max_joints * self.per_joint_prior_dim
        )
        layer_count = int(num_layers)

        # Historical name kept for checkpoint-key stability: this projection is
        # used to build per-joint grouping cues, not framewise phase dynamics.
        self.joint_motion_projection = nn.Sequential(
            nn.LayerNorm(self.joint_motion_stat_dim),
            nn.Linear(self.joint_motion_stat_dim, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        if self.skip_t5:
            self.name_projection = None
            self.joint_fusion = None
        else:
            self.name_projection = nn.Sequential(
                nn.LayerNorm(t5_out_dim),
                nn.Linear(t5_out_dim, self.latent_dim),
                nn.GELU(),
                nn.Linear(self.latent_dim, self.latent_dim),
            )
            self.joint_fusion = nn.Linear(self.latent_dim * 2, self.latent_dim)
        self.group_queries = nn.Parameter(torch.randn(self.num_groups, self.latent_dim) * 0.02)
        self.joint_prior_projection = nn.Sequential(
            nn.LayerNorm(self.latent_dim + self.phase_joint_stat_dim),
            nn.Linear(self.latent_dim + self.phase_joint_stat_dim, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.per_joint_prior_dim),
        )
        self.sequence_projection = nn.Sequential(
            nn.LayerNorm(prior_input_dim),
            nn.Linear(prior_input_dim, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(layer_count)
        ])
        self.temporal_layers = nn.ModuleList([
            ReferencePriorTemporalLayer(
                latent_dim=self.latent_dim,
                ff_size=ff_size,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(layer_count)
        ])
        self.token_queries = nn.Parameter(torch.randn(self.num_prior_tokens, self.latent_dim) * 0.02)
        self.token_attn = SelectiveMultiheadAttention(self.latent_dim, num_heads, dropout=dropout)
        self.output_ffn = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, self.latent_dim),
        )
        self.output_norm = nn.LayerNorm(self.latent_dim)

    def _apply_token_regularization(self, prior_tokens):
        if not self.training:
            return prior_tokens

        keep_mask = None
        if self.token_dropout_prob > 0.0:
            keep_mask = torch.rand(
                prior_tokens.shape[:2],
                device=prior_tokens.device,
            ) >= self.token_dropout_prob
            all_dropped = ~keep_mask.any(dim=0)
            if bool(all_dropped.any()):
                dropped_batch_indices = torch.nonzero(all_dropped, as_tuple=False).flatten()
                rescued_tokens = torch.randint(
                    0,
                    prior_tokens.shape[0],
                    (dropped_batch_indices.numel(),),
                    device=prior_tokens.device,
                )
                keep_mask[rescued_tokens, dropped_batch_indices] = True
            prior_tokens = prior_tokens * keep_mask.unsqueeze(-1).to(dtype=prior_tokens.dtype)

        if self.token_noise_std > 0.0:
            noise = torch.randn_like(prior_tokens) * self.token_noise_std
            if keep_mask is not None:
                noise = noise * keep_mask.unsqueeze(-1).to(dtype=prior_tokens.dtype)
            prior_tokens = prior_tokens + noise

        return prior_tokens

    @staticmethod
    def _coerce_joint_name_embeddings(joints_embedded_names, batch_size, max_joints, device, dtype):
        if joints_embedded_names is None:
            raise ValueError("reference conditioning requires joint-name embeddings")
        if not torch.is_tensor(joints_embedded_names):
            joints_embedded_names = torch.as_tensor(joints_embedded_names)
        if joints_embedded_names.dim() == 2:
            joints_embedded_names = joints_embedded_names.unsqueeze(0)
        elif joints_embedded_names.dim() != 3:
            raise ValueError(
                f"reference joint-name embeddings must be rank 2 or 3, got {tuple(joints_embedded_names.shape)}"
            )
        if joints_embedded_names.shape[0] != batch_size:
            raise ValueError(
                "reference joint-name embedding batch dimension "
                f"{joints_embedded_names.shape[0]} does not match batch size {batch_size}"
            )
        joints_embedded_names = joints_embedded_names.to(device=device, dtype=dtype)
        if joints_embedded_names.shape[1] > max_joints:
            raise ValueError(
                "reference joint-name embedding joint dimension "
                f"{joints_embedded_names.shape[1]} exceeds reference motion joints {max_joints}"
            )
        if joints_embedded_names.shape[1] == max_joints:
            return joints_embedded_names
        padded = torch.zeros(
            (batch_size, max_joints, joints_embedded_names.shape[2]),
            device=device,
            dtype=dtype,
        )
        padded[:, : joints_embedded_names.shape[1]] = joints_embedded_names
        return padded

    @staticmethod
    def _masked_mean_and_std(values, weights, eps=1e-6):
        weights_sum = weights.sum(dim=2, keepdim=True).clamp_min(eps)
        mean = (values * weights.unsqueeze(-1)).sum(dim=2) / weights_sum
        second_moment = ((values * values) * weights.unsqueeze(-1)).sum(dim=2) / weights_sum
        variance = (second_moment - mean * mean).clamp_min(0.0)
        std = torch.sqrt(variance + eps)
        return mean, std

    @staticmethod
    def _masked_mean(values, weights, eps=1e-6):
        weights_sum = weights.sum(dim=2, keepdim=True).clamp_min(eps)
        return (values * weights.unsqueeze(-1)).sum(dim=2) / weights_sum

    @staticmethod
    def _build_joint_motion_frame_features(vel, rot_delta_norm, contact):
        """Return phase-invariant per-frame cues for joint grouping.

        This branch deliberately drops velocity sign by using ``vel_norm`` so a
        joint's coarse semantic role (for example foreleg vs hindleg) is driven
        by stable motion statistics rather than instantaneous swing direction.
        """
        vel_norm = torch.linalg.norm(vel, dim=-1, keepdim=True)
        energy = torch.sqrt(vel_norm.square() + rot_delta_norm.square() + 1e-6)
        joint_motion_frame_features = torch.cat([vel_norm, rot_delta_norm, energy, contact], dim=-1)
        return joint_motion_frame_features, vel_norm, energy

    @staticmethod
    def _build_phase_joint_features(vel, contact):
        """Return root-excluded signed limb dynamics for phase modeling.

        Unlike the grouping branch, this path keeps signed velocity so the
        encoder can distinguish swing direction and left/right anti-phase.
        """
        return torch.cat([vel, contact], dim=-1)

    def forward(
        self,
        reference_motion,
        n_joints,
        lengths,
        translation_root_index,
        joints_embedded_names,
    ):
        if reference_motion.dim() != 4:
            raise ValueError(f"reference_motion must have shape (B, J, F, T), got {tuple(reference_motion.shape)}")

        batch_size, max_joints, feature_dim, frame_count = reference_motion.shape
        device = reference_motion.device
        dtype = reference_motion.dtype

        if feature_dim != self.input_feats:
            raise ValueError(f"Expected reference feature dim {self.input_feats}, got {feature_dim}")
        if max_joints > self.max_joints:
            raise ValueError(
                f"reference_motion joint dimension {max_joints} exceeds configured max_joints {self.max_joints}"
            )

        n_joints = torch.as_tensor(n_joints, device=device, dtype=torch.long).reshape(batch_size)
        lengths = torch.as_tensor(lengths, device=device, dtype=torch.long).reshape(batch_size)
        translation_root_index = torch.as_tensor(translation_root_index, device=device, dtype=torch.long).reshape(batch_size)
        if bool(((n_joints < 0) | (n_joints > max_joints)).any()):
            raise ValueError(
                f"reference n_joints must be in [0, {max_joints}], got {n_joints.tolist()}"
            )
        if bool(((translation_root_index < 0) | (translation_root_index >= n_joints.clamp_min(1))).any()):
            raise ValueError(
                "reference translation_root_index must reference a valid non-padded joint, got "
                f"{translation_root_index.tolist()} for n_joints={n_joints.tolist()}"
            )
        joints_embedded_names = self._coerce_joint_name_embeddings(
            joints_embedded_names,
            batch_size,
            max_joints,
            device,
            dtype,
        )
        valid_joints = torch.arange(max_joints, device=device).unsqueeze(0) < n_joints.unsqueeze(1)
        valid_frames = torch.arange(frame_count, device=device).unsqueeze(0) < lengths.unsqueeze(1)

        motion_btjf = reference_motion.permute(0, 3, 1, 2)
        vel = motion_btjf[..., 9:12]
        rot = motion_btjf[..., 3:9]
        rot_delta = torch.zeros_like(rot)
        if frame_count > 1:
            rot_delta[:, 1:] = rot[:, 1:] - rot[:, :-1]
        rot_delta_norm = torch.linalg.norm(rot_delta, dim=-1, keepdim=True)
        contact = motion_btjf[..., 12:13]
        # Branch A: build phase-invariant per-joint statistics used only for
        # semantic grouping / soft limb assignment.
        joint_motion_frame_features, vel_norm, energy = self._build_joint_motion_frame_features(
            vel,
            rot_delta_norm,
            contact,
        )
        joint_motion_frame_features = joint_motion_frame_features * valid_joints[:, None, :, None].to(dtype)

        # Time-aggregate only the grouping branch. Despite the historical
        # ``joint_motion`` name, these stats describe a joint's coarse semantic
        # motion role across the whole clip, not its framewise phase.
        valid_frames_bj = valid_frames[:, :, None, None].to(dtype)
        masked_joint_motion_ff = joint_motion_frame_features * valid_frames_bj
        frame_count_valid = valid_frames.to(dtype).sum(dim=1, keepdim=True).unsqueeze(-1).clamp_min(1)
        joint_motion_mean = masked_joint_motion_ff.sum(dim=1) / frame_count_valid
        joint_motion_second_moment = (masked_joint_motion_ff ** 2).sum(dim=1) / frame_count_valid
        joint_motion_std = torch.sqrt(
            (joint_motion_second_moment - joint_motion_mean ** 2).clamp_min(0.0) + 1e-6
        )
        joint_motion_stats = torch.cat([joint_motion_mean, joint_motion_std], dim=-1)
        joint_motion_stats = joint_motion_stats * valid_joints.unsqueeze(-1).to(dtype)

        projected_joint_motion = self.joint_motion_projection(joint_motion_stats)
        if self.name_projection is not None:
            joint_name_latent = self.name_projection(joints_embedded_names)
            joint_latent = self.joint_fusion(
                torch.cat([joint_name_latent, projected_joint_motion], dim=-1)
            )
        else:
            joint_latent = projected_joint_motion

        # Softly assign joints to motion groups using only the semantic/grouping
        # branch (plus joint names when T5 embeddings are enabled).
        group_logits = torch.einsum('bjd,gd->bjg', joint_latent, self.group_queries)
        group_logits = group_logits.masked_fill(~valid_joints.unsqueeze(-1), -1e4)
        group_weights = torch.softmax(group_logits, dim=-1) * valid_joints.unsqueeze(-1).to(dtype)

        safe_root_index = translation_root_index.clamp(min=0, max=max_joints - 1).view(batch_size, 1)
        non_root_joints = valid_joints.clone()
        non_root_joints.scatter_(1, safe_root_index, False)
        # Root-excluded signed phase cues are fused with each joint's semantic
        # latent before entering the final prior sequence.
        phase_joint_features = self._build_phase_joint_features(vel, contact)
        phase_joint_features = phase_joint_features * non_root_joints[:, None, :, None].to(dtype)

        global_mean, global_std = self._masked_mean_and_std(
            joint_motion_frame_features,
            valid_joints[:, None, :].expand(-1, frame_count, -1).to(dtype),
        )
        global_features = torch.cat([global_mean, global_std], dim=-1)

        group_features = []
        for group_index in range(self.num_groups):
            expanded_group_weights = group_weights[:, None, :, group_index].expand(-1, frame_count, -1)
            group_mean, group_std = self._masked_mean_and_std(
                joint_motion_frame_features,
                expanded_group_weights,
            )
            group_features.append(torch.cat([group_mean, group_std], dim=-1))
        group_features = torch.cat(group_features, dim=-1)
        phase_joint_abs = phase_joint_features.abs()
        phase_joint_delta = torch.zeros_like(phase_joint_features)
        if frame_count > 1:
            phase_joint_delta[:, 1:] = phase_joint_features[:, 1:] - phase_joint_features[:, :-1]
        phase_joint_stats = torch.cat([
            phase_joint_features,
            phase_joint_abs,
            phase_joint_delta,
        ], dim=-1)
        joint_semantic_sequence = torch.cat(
            [
                joint_latent[:, None, :, :].expand(-1, frame_count, -1, -1),
                phase_joint_stats,
            ],
            dim=-1,
        )
        joint_semantic_sequence = self.joint_prior_projection(joint_semantic_sequence)
        joint_semantic_sequence = joint_semantic_sequence * valid_joints[:, None, :, None].to(dtype)
        if max_joints < self.max_joints:
            joint_semantic_sequence = torch.nn.functional.pad(
                joint_semantic_sequence,
                (0, 0, 0, self.max_joints - max_joints),
            )
        joint_semantic_features = joint_semantic_sequence.reshape(
            batch_size,
            frame_count,
            self.max_joints * self.per_joint_prior_dim,
        )
        prior_sequence = torch.cat([global_features, group_features, joint_semantic_features], dim=-1)
        # Reference priors assume the canonical 13-D motion schema:
        # pos[0:3], rot6d[3:9], vel[9:12], contact[12].
        frame_mask = valid_frames.unsqueeze(-1).to(dtype)
        prior_sequence = prior_sequence * frame_mask
        prior_sequence = self.sequence_projection(prior_sequence)

        positions = torch.arange(frame_count, device=device).view(1, -1, 1).repeat(batch_size, 1, 1)
        prior_sequence = prior_sequence + create_sin_embedding(positions, self.latent_dim, dtype=dtype)
        prior_sequence = prior_sequence * frame_mask

        conv_input = prior_sequence.transpose(1, 2)
        conv_frame_mask = valid_frames.unsqueeze(1).to(dtype)
        for conv_block in self.conv_blocks:
            conv_input = (conv_input + conv_block(conv_input)) * conv_frame_mask
        temporal_sequence = conv_input.transpose(1, 2).transpose(0, 1).contiguous()
        temporal_key_padding_mask = ~valid_frames
        for layer in self.temporal_layers:
            temporal_sequence = layer(temporal_sequence, key_padding_mask=temporal_key_padding_mask)

        token_queries = self.token_queries.unsqueeze(1).expand(-1, batch_size, -1)
        prior_tokens, _ = self.token_attn(
            token_queries,
            temporal_sequence,
            temporal_sequence,
            key_padding_mask=temporal_key_padding_mask,
            need_weights=False,
        )
        prior_tokens = prior_tokens + self.output_ffn(prior_tokens)
        prior_tokens = self.output_norm(prior_tokens)
        prior_tokens = self._apply_token_regularization(prior_tokens)
        return prior_tokens

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


