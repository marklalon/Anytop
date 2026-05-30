import torch
torch.cuda.empty_cache()
import torch.nn as nn
import numpy as np
from model.motion_transformer import GraphMotionDecoderLayer, GraphMotionDecoder
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
        self.value_emb=kargs.get('value_emb', False)
        self.cross_limb=kargs.get('cross_limb', True)
        self.cross_limb_latents=kargs.get('cross_limb_latents', 8)
        self.cross_limb_dim=kargs.get('cross_limb_dim', 64)
        self.cross_limb_last_n=kargs.get('cross_limb_last_n', 0)
        self.joint_mask_prob=float(kargs.get('joint_mask_prob', 0.5))
        self.joint_mask_budget=float(kargs.get('joint_mask_budget', 0.15))
        self.temporal_span_mask_prob=float(kargs.get('temporal_span_mask_prob', 0.0))
        self.temporal_span_mask_min_frames=int(kargs.get('temporal_span_mask_min_frames', 4))
        self.temporal_span_mask_max_frames=int(kargs.get('temporal_span_mask_max_frames', 12))
        self.global_energy_cond=bool(kargs.get('global_energy_cond', False))
        self.loop_cond_prob=float(kargs.get('loop_cond_prob', 0.0))
        self.global_energy_stats_momentum = 0.01
        if not 0.0 <= self.joint_mask_prob <= 1.0:
            raise ValueError(f"joint_mask_prob must be in [0, 1], got {self.joint_mask_prob}")
        if not 0.0 <= self.joint_mask_budget <= 1.0:
            raise ValueError(f"joint_mask_budget must be in [0, 1], got {self.joint_mask_budget}")
        if not 0.0 <= self.temporal_span_mask_prob <= 1.0:
            raise ValueError(
                f"temporal_span_mask_prob must be in [0, 1], got {self.temporal_span_mask_prob}"
            )
        if self.temporal_span_mask_min_frames < 1:
            raise ValueError(
                "temporal_span_mask_min_frames must be >= 1, got "
                f"{self.temporal_span_mask_min_frames}"
            )
        if self.temporal_span_mask_max_frames <= 0:
            raise ValueError("temporal_span_mask_max_frames must be >= 1")
        if 0 < self.temporal_span_mask_max_frames < self.temporal_span_mask_min_frames:
            raise ValueError(
                "temporal_span_mask_max_frames must be >= temporal_span_mask_min_frames "
                f"(got min={self.temporal_span_mask_min_frames}, max={self.temporal_span_mask_max_frames})"
            )

        self.input_process = InputProcess(self.input_feats, self.root_input_feats, self.latent_dim, t5_out_dim, dropout_prob=self.dropout)
        if self.global_energy_cond:
            self.global_energy_projection = nn.Sequential(
                nn.LayerNorm(2),
                nn.Linear(2, self.latent_dim),
                nn.GELU(),
                nn.Linear(self.latent_dim, self.latent_dim),
            )
            self.register_buffer('global_energy_running_mean', torch.zeros(2, dtype=torch.float32))
            self.register_buffer('global_energy_running_var', torch.ones(2, dtype=torch.float32))
            self.register_buffer('global_energy_running_count', torch.zeros((), dtype=torch.long))
        else:
            self.global_energy_projection = None
        if self.loop_cond_prob > 0.0:
            self.loop_condition_projection = nn.Sequential(
                nn.Linear(1, self.latent_dim),
                nn.GELU(),
                nn.Linear(self.latent_dim, self.latent_dim),
            )
        else:
            self.loop_condition_projection = None
        self.playspeed_projection = nn.Sequential(
            nn.Linear(1, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        seqTransDecoderLayer = GraphMotionDecoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation,
                                                            global_energy_cond=self.global_energy_cond)
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

    def _update_global_energy_running_stats(self, raw_global_energy_cond):
        if not self.global_energy_cond or raw_global_energy_cond.numel() == 0:
            return

        batch_stats = raw_global_energy_cond.detach().to(dtype=torch.float32)
        batch_mean = batch_stats.mean(dim=0)
        batch_var = batch_stats.var(dim=0, unbiased=False).clamp_min(1e-6)
        with torch.no_grad():
            if int(self.global_energy_running_count.item()) == 0:
                self.global_energy_running_mean.copy_(batch_mean)
                self.global_energy_running_var.copy_(batch_var)
            else:
                self.global_energy_running_mean.lerp_(batch_mean, self.global_energy_stats_momentum)
                self.global_energy_running_var.lerp_(batch_var, self.global_energy_stats_momentum)
            self.global_energy_running_count.add_(int(batch_stats.shape[0]))

    def _coerce_global_energy_condition(self, raw_global_energy_cond, batch_size, device, dtype):
        if raw_global_energy_cond is None:
            raw_global_energy_cond = self.global_energy_running_mean.unsqueeze(0)
        if not torch.is_tensor(raw_global_energy_cond):
            raw_global_energy_cond = torch.as_tensor(raw_global_energy_cond)
        raw_global_energy_cond = raw_global_energy_cond.to(device=device, dtype=dtype)
        if raw_global_energy_cond.dim() == 1:
            raw_global_energy_cond = raw_global_energy_cond.unsqueeze(0)
        elif raw_global_energy_cond.dim() != 2:
            raise ValueError(
                "global_energy_cond must have shape (2,) or (B, 2), got "
                f"{tuple(raw_global_energy_cond.shape)}"
            )
        if raw_global_energy_cond.shape[1] != 2:
            raise ValueError(
                "global_energy_cond must provide [energy_mean, energy_std], got "
                f"shape {tuple(raw_global_energy_cond.shape)}"
            )
        if raw_global_energy_cond.shape[0] == 1 and batch_size != 1:
            raw_global_energy_cond = raw_global_energy_cond.expand(batch_size, -1)
        elif raw_global_energy_cond.shape[0] != batch_size:
            raise ValueError(
                "global_energy_cond batch dimension must match the motion batch size, got "
                f"{raw_global_energy_cond.shape[0]} for batch {batch_size}"
            )
        if not torch.isfinite(raw_global_energy_cond).all():
            raise ValueError("global_energy_cond must be finite")
        return raw_global_energy_cond

    def _build_global_energy_token(self, raw_global_energy_cond, batch_size, device, dtype):
        if not self.global_energy_cond or self.global_energy_projection is None:
            return None

        raw_global_energy_cond = self._coerce_global_energy_condition(
            raw_global_energy_cond,
            batch_size,
            device,
            dtype,
        )
        if self.training:
            self._update_global_energy_running_stats(raw_global_energy_cond)
        running_mean = self.global_energy_running_mean.to(device=device, dtype=dtype)
        running_std = torch.sqrt(self.global_energy_running_var.to(device=device, dtype=dtype).clamp_min(1e-6))
        normalized_global_energy = (raw_global_energy_cond - running_mean.unsqueeze(0)) / running_std.unsqueeze(0)
        return self.global_energy_projection(normalized_global_energy)

    def _coerce_loop_condition(self, raw_loop_cond, batch_size, device, dtype):
        if raw_loop_cond is None:
            raw_loop_cond = torch.zeros(batch_size, device=device, dtype=dtype)
        elif not torch.is_tensor(raw_loop_cond):
            raw_loop_cond = torch.as_tensor(raw_loop_cond, device=device)
        raw_loop_cond = raw_loop_cond.to(device=device)
        if raw_loop_cond.dim() == 0:
            raw_loop_cond = raw_loop_cond.reshape(1)
        raw_loop_cond = raw_loop_cond.reshape(-1)
        if raw_loop_cond.numel() == 1 and batch_size != 1:
            raw_loop_cond = raw_loop_cond.expand(batch_size)
        elif raw_loop_cond.numel() != batch_size:
            raise ValueError(
                "is_loop batch dimension must match the motion batch size, got "
                f"{raw_loop_cond.numel()} for batch {batch_size}"
            )
        return raw_loop_cond.to(dtype=dtype).view(batch_size, 1)

    def _coerce_playspeed_cond(self, raw_playspeed_cond, batch_size, device, dtype):
        if raw_playspeed_cond is None:
            raw_playspeed_cond = torch.ones(batch_size, device=device, dtype=dtype)
        elif not torch.is_tensor(raw_playspeed_cond):
            raw_playspeed_cond = torch.as_tensor(raw_playspeed_cond, device=device)
        raw_playspeed_cond = raw_playspeed_cond.to(device=device)
        if raw_playspeed_cond.dim() == 0:
            raw_playspeed_cond = raw_playspeed_cond.reshape(1)
        raw_playspeed_cond = raw_playspeed_cond.reshape(-1)
        if raw_playspeed_cond.numel() == 1 and batch_size != 1:
            raw_playspeed_cond = raw_playspeed_cond.expand(batch_size)
        elif raw_playspeed_cond.numel() != batch_size:
            raise ValueError(
                "playspeed_cond batch dimension must match the motion batch size, got "
                f"{raw_playspeed_cond.numel()} for batch {batch_size}"
            )
        if not torch.isfinite(raw_playspeed_cond).all():
            raise ValueError("playspeed_cond must be finite")
        return raw_playspeed_cond.to(dtype=dtype).view(batch_size, 1)

    def sample_subtree_joint_mask_train(self, y, njoints, device):
        """Select subtrees of joints to perturb during training.

        Returns a bool tensor of shape ``[B, njoints]`` (True = joint selected)
        or ``None`` if no joint was selected, or if not in training mode, or
        if ``joint_mask_prob == 0`` or ``joint_mask_budget == 0`` -- so
        eval-mode loss reports a clean diffusion objective.

        Called from ``GaussianDiffusion.training_losses`` AFTER ``q_sample``
        to decide which joints' x_t slice should be re-noised with an
        independent random timestep and fresh noise, so that those joints'
        noise level disagrees with the rest of the batch sample. This trains
        the cross-joint pathway to denoise robustly against per-joint
        timestep mismatch -- the regime inpaint clamping produces at
        inference. The model's forward itself stays vanilla.
        """
        if (not self.training) or self.joint_mask_prob <= 0.0 or self.joint_mask_budget <= 0.0:
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
            joint_mask_budget=self.joint_mask_budget,
            rng=np.random,
        )
        if subtree_joint_mask_np is None:
            return None
        return torch.from_numpy(subtree_joint_mask_np).to(device=device)

    def sample_temporal_span_mask_train(self, y, njoints, nframes, device):
        """Select contiguous temporal spans to perturb during training.

        Returns a bool tensor of shape ``[B, njoints, nframes]`` (True =
        re-noise that joint/frame cell) or ``None`` if no span was selected,
        if not in training mode, or if ``temporal_span_mask_prob == 0``.

        Full-clip spans are not supported: the sampled span always leaves at
        least one frame at the start or end unmasked so seam detection never
        triggers on a clip-span boundary.  ``temporal_span_mask_max_frames``
        is therefore capped at ``min(config_max, nframes - 1)`` at runtime.
        Every real joint in the sample shares the same contiguous masked frame
        interval, while padded joints stay False throughout.
        """
        if (not self.training) or self.temporal_span_mask_prob <= 0.0:
            return None

        n_joints_batch = y.get('n_joints_cpu', y.get('n_joints'))
        if n_joints_batch is None:
            return None

        if torch.is_tensor(n_joints_batch):
            n_joints_np = n_joints_batch.detach().to(device='cpu', dtype=torch.int64).numpy().reshape(-1)
        else:
            n_joints_np = np.asarray(n_joints_batch, dtype=np.int64).reshape(-1)

        batch_size = len(n_joints_np)
        if batch_size == 0 or nframes <= 0:
            return None

        min_span = min(self.temporal_span_mask_min_frames, nframes)
        max_span = max(min_span, min(self.temporal_span_mask_max_frames, nframes - 1))

        temporal_span_mask_np = np.zeros((batch_size, njoints, nframes), dtype=np.bool_)
        any_selected = False
        for batch_index in range(batch_size):
            valid_joints = min(int(n_joints_np[batch_index]), int(njoints))
            if valid_joints <= 0:
                continue
            if np.random.random() >= self.temporal_span_mask_prob:
                continue

            span_length = int(np.random.randint(min_span, max_span + 1))
            start_hi = nframes - span_length
            span_start = 0 if start_hi <= 0 else int(np.random.randint(0, start_hi + 1))
            temporal_span_mask_np[
                batch_index,
                :valid_joints,
                span_start:span_start + span_length,
            ] = True
            any_selected = True

        if not any_selected:
            return None
        return torch.from_numpy(temporal_span_mask_np).to(device=device)

    def forward(self, x, timesteps, y=None, train_step=None, **unused_kwargs):
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
        global_energy_condition = None
        # joint-mask subtree perturbation is applied OUTSIDE this
        # forward, in diffusion.training_losses, by re-noising the selected
        # joints' x_t with q_sample(x_0, t_random, fresh_noise). The model
        # itself stays vanilla -- selected joints DO NOT enter the
        # key-padding masks and continue to participate in attention normally,
        # just with mismatched noise levels, which trains the
        # cross-joint pathway to denoise robustly against per-joint timestep
        # disagreement (matching inpaint clamp behavior at inference).
        timesteps_emb = create_sin_embedding(timesteps.view(1, -1, 1), self.latent_dim)[0]
        playspeed_condition = self._coerce_playspeed_cond(
            y.get('playspeed_cond'),
            batch_size=bs,
            device=x.device,
            dtype=x.dtype,
        )
        timesteps_emb = timesteps_emb + self.playspeed_projection(playspeed_condition)
        if self.loop_cond_prob > 0.0 and self.loop_condition_projection is not None:
            loop_condition = self._coerce_loop_condition(
                y.get('is_loop'),
                batch_size=bs,
                device=x.device,
                dtype=x.dtype,
            )
            timesteps_emb = timesteps_emb + self.loop_condition_projection(loop_condition)

        loop_phase_mask = None
        raw_loop_phase_mask = y.get('is_loop')
        if raw_loop_phase_mask is not None:
            loop_phase_mask = torch.as_tensor(raw_loop_phase_mask, device=x.device, dtype=torch.bool).reshape(-1)
            if loop_phase_mask.numel() == 1 and bs != 1:
                loop_phase_mask = loop_phase_mask.expand(bs)
            elif loop_phase_mask.numel() != bs:
                raise ValueError(
                    "is_loop batch dimension must match the motion batch size, got "
                    f"{loop_phase_mask.numel()} for batch {bs}"
                )
            raw_loop_full_cycle = y.get('loop_full_cycle')
            if raw_loop_full_cycle is not None:
                loop_full_cycle_mask = torch.as_tensor(raw_loop_full_cycle, device=x.device, dtype=torch.bool).reshape(-1)
                if loop_full_cycle_mask.numel() == 1 and bs != 1:
                    loop_full_cycle_mask = loop_full_cycle_mask.expand(bs)
                elif loop_full_cycle_mask.numel() != bs:
                    raise ValueError(
                        "loop_full_cycle batch dimension must match the motion batch size, got "
                        f"{loop_full_cycle_mask.numel()} for batch {bs}"
                    )
                loop_phase_mask = loop_phase_mask & loop_full_cycle_mask

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

        if self.global_energy_cond:
            global_energy_condition = self._build_global_energy_token(
                y.get('global_energy_cond'),
                batch_size=bs,
                device=x.device,
                dtype=x.dtype,
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

        loop_phase_lengths = y.get('loop_phase_lengths', y.get('lengths'))

        output = self.seqTransDecoder(
            tgt=x,
            timesteps_embs=timesteps_emb,
            memory=None,
            spatial_mask=spatial_mask,
            temporal_mask=temporal_mask,
            tgt_key_padding_mask=joint_key_padding_mask,
            y=y,
            global_energy_condition=global_energy_condition,
            temporal_template=temporal_template,
            cross_limb_unreliable_mask=cross_limb_unreliable_mask,
            loop_phase_mask=loop_phase_mask,
            lengths=loop_phase_lengths,
        )
        output = self.output_process(output) # Applies linear layer on each frame to convert it back to feature len dim
        return output


    def _apply(self, fn):
        super()._apply(fn)


    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)

# in the case of GMDM, the input process is as follows: 
# embed each joint of each frame of each motion in batch by the same MLP, separately ! 
class InputProcess(nn.Module):
    def __init__(self, input_feats, root_input_feats, latent_dim, t5_output_dim, dropout_prob=0):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.root_input_feats = root_input_feats
        self.root_embedding = nn.Linear(self.root_input_feats, self.latent_dim)
        self.tpos_root_embedding = nn.Linear(self.root_input_feats, self.latent_dim)
        self.joint_embedding = nn.Linear(self.input_feats, self.latent_dim)
        self.tpos_joint_embedding = nn.Linear(self.input_feats, self.latent_dim)
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
        joints_embedded_names = self.text_embedding(self.joints_names_dropout(joints_embedded_names.to(x.device)))
        x = x + joints_embedded_names[None, ...]# [frames, batch_size, n_joints, d]
        positions = torch.arange(x.shape[0], device=x.device).view(1, -1, 1).repeat(x.shape[1], 1, 1)
        pos_emb = create_sin_embedding(positions, self.latent_dim)[0]
        return x + pos_emb.unsqueeze(1).unsqueeze(1)


class GlobalEnergyExtractor:
    """Phase-invariant clip-level global energy statistics.

    Standalone helper retained after the reference/controlnet conditioning
    path was removed; the ``global_energy_cond`` feature consumes these
    statistics to derive a clip-level [energy_mean, energy_std] condition.
    Assumes the canonical 13-D motion schema: pos[0:3], rot6d[3:9],
    vel[9:12], contact[12].
    """

    @staticmethod
    def _masked_mean_and_std(values, weights, eps=1e-6):
        weights_sum = weights.sum(dim=2, keepdim=True).clamp_min(eps)
        mean = (values * weights.unsqueeze(-1)).sum(dim=2) / weights_sum
        second_moment = ((values * values) * weights.unsqueeze(-1)).sum(dim=2) / weights_sum
        variance = (second_moment - mean * mean).clamp_min(0.0)
        std = torch.sqrt(variance + eps)
        return mean, std

    @staticmethod
    def _coerce_global_energy_playspeed_cond(raw_playspeed_cond, batch_size, device, dtype):
        if raw_playspeed_cond is None:
            return None
        if not torch.is_tensor(raw_playspeed_cond):
            raw_playspeed_cond = torch.as_tensor(raw_playspeed_cond, device=device)
        raw_playspeed_cond = raw_playspeed_cond.to(device=device)
        if raw_playspeed_cond.dim() == 0:
            raw_playspeed_cond = raw_playspeed_cond.reshape(1)
        raw_playspeed_cond = raw_playspeed_cond.reshape(-1)
        if raw_playspeed_cond.numel() == 1 and batch_size != 1:
            raw_playspeed_cond = raw_playspeed_cond.expand(batch_size)
        elif raw_playspeed_cond.numel() != batch_size:
            raise ValueError(
                "playspeed_cond batch dimension must match the motion batch size, got "
                f"{raw_playspeed_cond.numel()} for batch {batch_size}"
            )
        if not torch.isfinite(raw_playspeed_cond).all():
            raise ValueError("playspeed_cond must be finite")
        if bool((raw_playspeed_cond <= 0).any()):
            raise ValueError("playspeed_cond must be positive")
        return raw_playspeed_cond.to(dtype=dtype).view(batch_size, 1, 1, 1)

    @staticmethod
    def _resample_motion_time_axis(motion, target_frame_count):
        # Fallback global-energy paths may only have the internal-window clip
        # plus playspeed_cond. Rebuild the inferred physical frame count first,
        # then measure energy on that cadence instead of the stretched window.
        source_frame_count = int(motion.shape[-1])
        target_frame_count = int(target_frame_count)
        if target_frame_count <= 0:
            raise ValueError(f"target_frame_count must be positive, got {target_frame_count}")
        if source_frame_count == target_frame_count:
            return motion

        motion_tjf = motion.permute(2, 0, 1)
        src = torch.linspace(
            0.0,
            float(source_frame_count - 1),
            target_frame_count,
            device=motion.device,
            dtype=motion.dtype,
        )
        lo = torch.floor(src).to(dtype=torch.long).clamp(0, source_frame_count - 1)
        hi = torch.minimum(lo + 1, torch.full_like(lo, source_frame_count - 1))
        w = (src - torch.floor(src)).view(-1, 1, 1).to(dtype=motion_tjf.dtype)
        resampled = motion_tjf.index_select(0, lo) * (1.0 - w) + motion_tjf.index_select(0, hi) * w

        if motion.shape[1] >= 13:
            nearest = torch.round(src).to(dtype=torch.long).clamp(0, source_frame_count - 1)
            resampled[..., 12] = (motion_tjf.index_select(0, nearest)[..., 12] >= 0.5).to(dtype=resampled.dtype)

        return resampled.permute(1, 2, 0).contiguous()

    @staticmethod
    def _build_joint_motion_frame_features(vel, rot_delta_norm, contact):
        """Return phase-invariant per-frame energy cues.

        Drops velocity sign by using ``vel_norm`` so the energy statistic is
        driven by stable motion magnitude rather than instantaneous swing
        direction.
        """
        vel_norm = torch.linalg.norm(vel, dim=-1, keepdim=True)
        energy = torch.sqrt(vel_norm.square() + rot_delta_norm.square() + 1e-6)
        joint_motion_frame_features = torch.cat([vel_norm, rot_delta_norm, energy, contact], dim=-1)
        return joint_motion_frame_features, vel_norm, energy

    @classmethod
    def _extract_joint_motion_inputs(cls, motion, n_joints):
        if motion.dim() != 4:
            raise ValueError(f"motion must have shape (B, J, F, T), got {tuple(motion.shape)}")

        batch_size, max_joints, feature_dim, frame_count = motion.shape
        if feature_dim < 13:
            raise ValueError(
                "global energy extraction expects at least the 13-dim feature schema "
                f"[pos(3), rot6d(6), vel(3), contact(1)], got feature_dim={feature_dim}"
            )

        device = motion.device
        dtype = motion.dtype
        n_joints = torch.as_tensor(n_joints, device=device, dtype=torch.long).reshape(batch_size)
        if bool(((n_joints < 0) | (n_joints > max_joints)).any()):
            raise ValueError(
                f"n_joints must be in [0, {max_joints}], got {n_joints.tolist()}"
            )

        valid_joints = torch.arange(max_joints, device=device).unsqueeze(0) < n_joints.unsqueeze(1)
        motion_btjf = motion.permute(0, 3, 1, 2)
        vel = motion_btjf[..., 9:12]
        rot = motion_btjf[..., 3:9]
        rot_delta = torch.zeros_like(rot)
        if frame_count > 1:
            rot_delta[:, 1:] = rot[:, 1:] - rot[:, :-1]
        rot_delta_norm = torch.linalg.norm(rot_delta, dim=-1, keepdim=True)
        contact = motion_btjf[..., 12:13]
        joint_motion_frame_features, vel_norm, energy = cls._build_joint_motion_frame_features(
            vel,
            rot_delta_norm,
            contact,
        )
        joint_motion_frame_features = joint_motion_frame_features * valid_joints[:, None, :, None].to(dtype)
        return {
            'dtype': dtype,
            'frame_count': frame_count,
            'valid_joints': valid_joints,
            'joint_motion_frame_features': joint_motion_frame_features,
        }

    @classmethod
    def compute_global_energy_condition(cls, motion, n_joints, playspeed_cond=None):
        if playspeed_cond is not None:
            if motion.dim() != 4:
                raise ValueError(f"motion must have shape (B, J, F, T), got {tuple(motion.shape)}")
            batch_size = motion.shape[0]
            source_length_ratio = cls._coerce_global_energy_playspeed_cond(
                playspeed_cond,
                batch_size,
                motion.device,
                motion.dtype,
            ).reshape(batch_size)
            inferred_source_frames = torch.round(source_length_ratio * float(motion.shape[-1])).to(dtype=torch.long).clamp_min(1)
            if bool((inferred_source_frames != motion.shape[-1]).any()):
                # Scaling rot_delta alone is not enough: the energy mean/std is
                # also averaged across the wrong number of frames. Resample back
                # to the inferred physical length before extracting statistics.
                n_joints_tensor = torch.as_tensor(n_joints, device=motion.device, dtype=torch.long).reshape(batch_size)
                sample_conditions = []
                for batch_index in range(batch_size):
                    sample_motion = motion[batch_index]
                    sample_source_frames = int(inferred_source_frames[batch_index].item())
                    if sample_source_frames != motion.shape[-1]:
                        sample_motion = cls._resample_motion_time_axis(sample_motion, sample_source_frames)
                    sample_conditions.append(
                        cls.compute_global_energy_condition(
                            sample_motion.unsqueeze(0),
                            n_joints=n_joints_tensor[batch_index:batch_index + 1],
                            playspeed_cond=None,
                        )
                    )
                return torch.cat(sample_conditions, dim=0)

        motion_inputs = cls._extract_joint_motion_inputs(motion, n_joints)
        dtype = motion_inputs['dtype']
        frame_count = motion_inputs['frame_count']
        valid_joints = motion_inputs['valid_joints']
        joint_motion_frame_features = motion_inputs['joint_motion_frame_features']
        global_mean, global_std = cls._masked_mean_and_std(
            joint_motion_frame_features,
            valid_joints[:, None, :].expand(-1, frame_count, -1).to(dtype),
        )
        global_energy_profile = torch.cat([global_mean[..., 2:3], global_std[..., 2:3]], dim=-1)
        return global_energy_profile.mean(dim=1)

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


