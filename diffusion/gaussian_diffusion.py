# This code is based on https://github.com/openai/guided-diffusion
"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math
import weakref
import numpy as np
import torch
import torch as th
from copy import deepcopy
from diffusion import logger
from diffusion.nn import mean_flat, sum_flat
from diffusion.losses import normal_kl, discretized_gaussian_log_likelihood, geodesic_distance
from model.anytop import ReferencePriorEncoder
from utils.rotation_conversions import rotation_6d_to_matrix_safe


_EXTRACT_TENSOR_CACHE = {}
_EXTRACT_TENSOR_CACHE_FINALIZERS = {}


def _device_cache_key(device):
    device = th.device(device)
    return device.type, device.index


def _cleanup_extract_tensor_cache(array_id):
    for key in list(_EXTRACT_TENSOR_CACHE):
        if key[0] == array_id:
            _EXTRACT_TENSOR_CACHE.pop(key, None)
    _EXTRACT_TENSOR_CACHE_FINALIZERS.pop(array_id, None)


def _cached_extract_source_tensor(arr, device):
    array_id = id(arr)
    key = (array_id, arr.shape, arr.dtype.str, _device_cache_key(device))
    cached = _EXTRACT_TENSOR_CACHE.get(key)
    if cached is not None and cached.device == th.device(device):
        return cached

    tensor = th.from_numpy(arr).to(device=device, dtype=th.float32)
    _EXTRACT_TENSOR_CACHE[key] = tensor
    if array_id not in _EXTRACT_TENSOR_CACHE_FINALIZERS:
        try:
            _EXTRACT_TENSOR_CACHE_FINALIZERS[array_id] = weakref.finalize(
                arr, _cleanup_extract_tensor_cache, array_id
            )
        except TypeError:
            pass
    return tensor


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, scale_betas=1.):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = scale_betas * 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        lambda_geo=0.,
        lambda_vel=0.,
        lambda_loop_wrap=0.,
        lambda_loop_root_xz=0.,
        temporal_span_seam_loss_weight=0.0,
        temporal_span_seam_width=0,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        self.lambda_geo = lambda_geo
        self.lambda_vel = lambda_vel
        self.lambda_loop_wrap = float(lambda_loop_wrap)
        self.lambda_loop_root_xz = float(lambda_loop_root_xz)
        self.temporal_span_seam_loss_weight = float(temporal_span_seam_loss_weight)
        self.temporal_span_seam_width = int(temporal_span_seam_width)
        if self.lambda_loop_wrap < 0.0:
            raise ValueError(f"lambda_loop_wrap must be >= 0, got {self.lambda_loop_wrap}")
        if self.lambda_loop_root_xz < 0.0:
            raise ValueError(f"lambda_loop_root_xz must be >= 0, got {self.lambda_loop_root_xz}")
        if self.temporal_span_seam_loss_weight < 0.0:
            raise ValueError(
                "temporal_span_seam_loss_weight must be >= 0, got "
                f"{self.temporal_span_seam_loss_weight}"
            )
        if self.temporal_span_seam_width < 0:
            raise ValueError(
                f"temporal_span_seam_width must be >= 0, got {self.temporal_span_seam_width}"
            )

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas = np.sqrt(alphas)
        self.sqrt_betas = np.sqrt(betas)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.l2_loss = lambda a, b: (a - b) ** 2  # th.nn.MSELoss(reduction='none')  # must be None for handling mask later on.

    def _unwrap_model(self, model):
        return getattr(model, 'model', model)

    def _fp32_math_context(self, *tensors):
        device_type = None
        for tensor in tensors:
            if isinstance(tensor, th.Tensor):
                device_type = tensor.device.type
                break
        if device_type is None:
            device_type = "cuda" if th.cuda.is_available() else "cpu"
        return th.autocast(device_type=device_type, enabled=False)

    def masked_l2(self, a, b, mask):
        # assuming a.shape == b.shape == bs, J, Jdim, seqlen
        # assuming mask.shape == bs, 1, 1, seqlen
        loss = self.l2_loss(a, b)
        loss = sum_flat(loss * mask.float())  # gives \sigma_euclidean over unmasked elements
        n_entries = a.shape[1] * a.shape[2]
        non_zero_elements = sum_flat(mask) * n_entries
        mse_loss_val = loss / non_zero_elements
        # print('mse_loss_val', mse_loss_val)
        return mse_loss_val
    
    def spatial_masked_l2(self, a, b, spat_mask, lengths, n_joints):
        # assuming a.shape == b.shape == bs, J, Jdim, seqlen
        # assuming spat_mask.shape == bs, 1, 1, max_joints

        loss = self.l2_loss(a, b)
        spat_masked_loss = (loss * spat_mask.float().transpose(1,3))
        loss = sum_flat(spat_masked_loss)  # gives \sigma_euclidean over unmasked elements
        non_zero_elements = lengths * n_joints * a.size(2) 
        mse_loss_val = loss / non_zero_elements
        return mse_loss_val

    def weighted_feature_l2(self, a, b, weights):
        if weights.dim() == 3:
            weights = weights.unsqueeze(2)
        weighted_loss = self.l2_loss(a, b) * weights.float()
        loss = sum_flat(weighted_loss)
        denom = sum_flat(weights.float()) * a.size(2)
        return loss / (denom + 1e-8)

    def _normalize_temporal_span_time_mask(self, temporal_span_mask):
        if temporal_span_mask is None:
            return None
        if temporal_span_mask.dim() == 3:
            return temporal_span_mask.any(dim=1)
        if temporal_span_mask.dim() == 2:
            return temporal_span_mask
        raise ValueError(
            "temporal_span_mask must have shape (B, J, T) or (B, T), got "
            f"{tuple(temporal_span_mask.shape)}"
        )

    def _build_temporal_span_seam_weights(self, temporal_span_mask, lengths):
        if (
            temporal_span_mask is None
            or self.temporal_span_seam_width <= 0
        ):
            return None

        temporal_span_time = self._normalize_temporal_span_time_mask(temporal_span_mask)

        batch_size, n_frames = temporal_span_time.shape
        seam_weights = th.zeros(
            (batch_size, 1, 1, n_frames),
            device=temporal_span_time.device,
            dtype=th.float32,
        )
        lengths_long = th.as_tensor(
            lengths, device=temporal_span_time.device, dtype=th.long
        ).reshape(-1)
        band = int(self.temporal_span_seam_width)
        # Treat seam_width as an approximately 2-sigma support radius so the
        # boundary frame remains dominant while nearby frames still contribute.
        sigma = max(float(band) / 2.0, 1e-6)

        for batch_index in range(batch_size):
            valid_frames = min(int(lengths_long[batch_index].item()), n_frames)
            if valid_frames <= 0:
                continue
            sample_mask = temporal_span_time[batch_index, :valid_frames]
            if not bool(sample_mask.any()):
                continue

            prev_mask = th.zeros_like(sample_mask)
            prev_mask[1:] = sample_mask[:-1]
            next_mask = th.zeros_like(sample_mask)
            next_mask[:-1] = sample_mask[1:]
            boundaries = th.cat(
                [
                    th.nonzero(sample_mask & ~prev_mask, as_tuple=False).flatten(),
                    th.nonzero(sample_mask & ~next_mask, as_tuple=False).flatten(),
                ],
                dim=0,
            )
            for boundary in boundaries.tolist():
                left = max(0, boundary - band)
                right = min(valid_frames, boundary + band + 1)
                local_indices = th.arange(left, right, device=temporal_span_time.device)
                boundary_weights = th.exp(
                    -0.5 * ((local_indices.float() - float(boundary)) / sigma) ** 2
                )
                seam_weights[batch_index, 0, 0, left:right] = th.maximum(
                    seam_weights[batch_index, 0, 0, left:right],
                    boundary_weights,
                )

        if not bool((seam_weights > 0).any()):
            return None
        return seam_weights

    def quat_to_mat(self, qs):
        r = qs[..., 0]
        i = qs[..., 1]
        j = qs[..., 2]
        k = qs[..., 3]
        two_s = 2.0 / (qs * qs).sum(-1)
        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        rotations = o.reshape(qs.shape[:-1] + (3, 3))
        return rotations
    
    def geodesic_loss(self, a, b, spat_mask, lengths, n_joints):
        # assuming a.shape == b.shape == bs, J, Jdim, seqlen
        # assuming spat_mask.shape == bs, 1, 1, max_joints
        a = a.float()
        b = b.float()
        rots_target = rotation_6d_to_matrix_safe(a.permute(0, 3, 1, 2)[..., 3:9])
        rots_pred = rotation_6d_to_matrix_safe(b.permute(0, 3, 1, 2)[..., 3:9])
        loss = geodesic_distance(rots_pred, rots_target).permute(0, 2, 3, 1)
        spat_masked_loss = (loss * spat_mask.float().transpose(1,3))
        loss = sum_flat(spat_masked_loss)  # gives \sigma_euclidean over unmasked elements
        non_zero_elements = (lengths * n_joints).float()
        loss_val = loss / non_zero_elements
        return loss_val

    def velocity_consistency_loss(self, model_output, spat_mask, lengths, n_joints):
        # model_output: [bs, njoints, nfeats, nframes] (denormalized)
        # vel[t] should equal pos[t+1] - pos[t]; enforce this across all valid frames.
        pos = model_output[:, :, 0:3, :]    # [bs, njoints, 3, nframes]
        vel = model_output[:, :, 9:12, :]   # [bs, njoints, 3, nframes]
        finite_diff = pos[:, :, :, 1:] - pos[:, :, :, :-1]  # [bs, njoints, 3, nframes-1]
        pred_vel    = vel[:, :, :, :-1]                       # [bs, njoints, 3, nframes-1]
        loss = (finite_diff - pred_vel) ** 2
        valid = spat_mask.float().transpose(1, 3)[:, :, :, :-1]        # [bs, njoints, 1, nframes-1]
        loss_val = (loss * valid).sum() / (valid.sum() * 3).clamp(min=1)
        return loss_val

    def _coerce_bool_batch(self, value, batch_size, device, default=False):
        if value is None:
            return th.full((batch_size,), bool(default), device=device, dtype=th.bool)
        value = th.as_tensor(value, device=device, dtype=th.bool).reshape(-1)
        if value.numel() == 1 and batch_size != 1:
            value = value.expand(batch_size)
        elif value.numel() != batch_size:
            raise ValueError(
                f"Boolean batch value has length {value.numel()} but expected {batch_size}."
            )
        return value

    def _coerce_index_batch(self, value, batch_size, device):
        if value is None:
            return th.zeros(batch_size, device=device, dtype=th.long)
        if torch.is_tensor(value):
            result = value.to(device=device, dtype=th.long).reshape(-1)
        else:
            result = th.as_tensor([
                0 if item is None else int(item)
                for item in value
            ], device=device, dtype=th.long).reshape(-1)
        if result.numel() == 1 and batch_size != 1:
            result = result.expand(batch_size)
        elif result.numel() != batch_size:
            raise ValueError(
                f"translation_root_index has length {result.numel()} but expected {batch_size}."
            )
        return result

    def loop_wrap_loss(self, model_output, y, lengths, n_joints):
        batch_size, max_joints, n_feats, n_frames = model_output.shape
        device = model_output.device
        is_loop = self._coerce_bool_batch(y.get('is_loop'), batch_size, device, default=False)
        loop_full_cycle = self._coerce_bool_batch(y.get('loop_full_cycle'), batch_size, device, default=False)
        active = is_loop & loop_full_cycle
        lengths_long = th.as_tensor(lengths, device=device, dtype=th.long).reshape(-1)
        n_joints_long = th.as_tensor(n_joints, device=device, dtype=th.long).reshape(-1)
        root_indices = self._coerce_index_batch(y.get('translation_root_index'), batch_size, device)
        zero = model_output.new_zeros(())

        valid_frames = lengths_long.clamp(min=0, max=n_frames)
        valid_joints = n_joints_long.clamp(min=0, max=max_joints)
        active_valid = active & (valid_frames >= 2) & (valid_joints > 0)
        active_weight = active_valid.to(dtype=model_output.dtype)
        active_denom = active_weight.sum().clamp(min=1.0)

        last_frame_index = (valid_frames.clamp(min=1) - 1).view(batch_size, 1, 1, 1)
        last_frame_index = last_frame_index.expand(-1, max_joints, n_feats, 1)
        first_frame = model_output[..., 0:1]
        last_frame = model_output.gather(dim=3, index=last_frame_index)

        joint_mask = th.arange(max_joints, device=device).view(1, max_joints) < valid_joints.view(batch_size, 1)
        joint_weight = joint_mask.to(dtype=model_output.dtype)

        pose_weight = joint_weight[:, :, None, None].expand(-1, -1, 3, -1).clone()
        batch_indices = th.arange(batch_size, device=device)
        root_valid = ((root_indices >= 0) & (root_indices < valid_joints)).to(dtype=model_output.dtype)
        root_indices_clamped = root_indices.clamp(min=0, max=max(max_joints - 1, 0))
        pose_weight[batch_indices, root_indices_clamped, 0, 0] *= 1.0 - root_valid
        pose_weight[batch_indices, root_indices_clamped, 2, 0] *= 1.0 - root_valid
        pose_denom = pose_weight.sum(dim=(1, 2, 3)).clamp(min=1.0)
        pose_per_sample = (
            (((first_frame[:, :, 0:3] - last_frame[:, :, 0:3]) ** 2) * pose_weight)
            .sum(dim=(1, 2, 3))
            / pose_denom
        )
        pose_per_sample = th.where(active_valid, pose_per_sample, th.zeros_like(pose_per_sample))
        pose_loss = (pose_per_sample * active_weight).sum() / active_denom

        rot_first = first_frame[:, :, 3:9].permute(0, 3, 1, 2)
        rot_last = last_frame[:, :, 3:9].permute(0, 3, 1, 2)
        rots_first = rotation_6d_to_matrix_safe(rot_first)
        rots_last = rotation_6d_to_matrix_safe(rot_last)
        rot_distance = geodesic_distance(rots_last, rots_first).squeeze(-1).squeeze(1)
        rot_per_sample = (rot_distance * joint_weight).sum(dim=1) / joint_weight.sum(dim=1).clamp(min=1.0)
        rot_per_sample = th.where(active_valid, rot_per_sample, th.zeros_like(rot_per_sample))
        rot_loss = (rot_per_sample * active_weight).sum() / active_denom

        terminal_vel_loss = zero
        if n_feats >= 12:
            terminal_residual = first_frame[:, :, 0:3, 0] - last_frame[:, :, 0:3, 0] - last_frame[:, :, 9:12, 0]
            terminal_per_sample = ((terminal_residual ** 2) * joint_weight[:, :, None]).sum(dim=(1, 2))
            terminal_per_sample = terminal_per_sample / (joint_weight.sum(dim=1) * 3.0).clamp(min=1.0)
            terminal_per_sample = th.where(active_valid, terminal_per_sample, th.zeros_like(terminal_per_sample))
            terminal_vel_loss = (terminal_per_sample * active_weight).sum() / active_denom

        total = pose_loss + rot_loss + terminal_vel_loss
        return {
            'loop_wrap_loss': total,
            'loop_wrap_pose': pose_loss,
            'loop_wrap_rot': rot_loss,
            'loop_wrap_terminal_vel': terminal_vel_loss,
        }

    def loop_root_xz_closure_loss(self, model_output, y, lengths, n_joints):
        batch_size, max_joints, n_feats, n_frames = model_output.shape
        device = model_output.device
        if n_feats < 12 or n_frames < 2:
            return model_output.new_zeros(())

        is_loop = self._coerce_bool_batch(y.get('is_loop'), batch_size, device, default=False)
        loop_full_cycle = self._coerce_bool_batch(y.get('loop_full_cycle'), batch_size, device, default=False)
        active = is_loop & loop_full_cycle
        lengths_long = th.as_tensor(lengths, device=device, dtype=th.long).reshape(-1)
        n_joints_long = th.as_tensor(n_joints, device=device, dtype=th.long).reshape(-1)
        root_indices = self._coerce_index_batch(y.get('translation_root_index'), batch_size, device)

        valid_frames = lengths_long.clamp(min=0, max=n_frames)
        valid_joints = n_joints_long.clamp(min=0, max=max_joints)
        root_valid = (root_indices >= 0) & (root_indices < valid_joints)
        active_valid = active & root_valid & (valid_frames >= 2)
        active_weight = active_valid.to(dtype=model_output.dtype)
        active_denom = active_weight.sum().clamp(min=1.0)

        batch_indices = th.arange(batch_size, device=device)
        root_indices_clamped = root_indices.clamp(min=0, max=max(max_joints - 1, 0))
        root_vel_xz = model_output[batch_indices, root_indices_clamped][:, [9, 11], :]
        transition_count = (valid_frames - 1).clamp(min=0)
        time_mask = th.arange(n_frames, device=device).view(1, 1, n_frames) < transition_count.view(batch_size, 1, 1)
        net_xz = (root_vel_xz * time_mask.to(dtype=model_output.dtype)).sum(dim=2)
        per_sample = (net_xz ** 2).mean(dim=1)
        per_sample = th.where(active_valid, per_sample, th.zeros_like(per_sample))
        return (per_sample * active_weight).sum() / active_denom

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the dataset for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial dataset batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values.float()
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values.float() + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            # print('model_variance', model_variance)
            # print('model_log_variance',model_log_variance)
            # print('self.posterior_variance', self.posterior_variance)
            # print('self.posterior_log_variance_clipped', self.posterior_log_variance_clipped)
            # print('self.model_var_type', self.model_var_type)


            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                # print('clip_denoised', clip_denoised)
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:  # THIS IS US!
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_mean_with_grad(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, t, p_mean_var, **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def condition_score_with_grad(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, t, p_mean_var, **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        const_noise=False,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        # print('const_noise', const_noise)
        if const_noise:
            noise = noise[[0]].repeat(x.shape[0], 1, 1, 1)

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        # print('mean', out["mean"].shape, out["mean"])
        # print('log_variance', out["log_variance"].shape, out["log_variance"])
        # print('nonzero_mask', nonzero_mask.shape, nonzero_mask)
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_with_grad(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        const_noise=False,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        with th.enable_grad():
            x = x.detach().requires_grad_()
            out = self.p_mean_variance(
                model,
                x,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            noise = th.randn_like(x)
            if const_noise:
                noise = noise[[0]].repeat(x.shape[0], 1, 1, 1)
            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
            )  # no noise when t == 0
            if cond_fn is not None:
                out["mean"] = self.condition_mean_with_grad(
                    cond_fn, out, x, t, model_kwargs=model_kwargs
                )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"].detach()}

    def _inpaint_project(self, sample, i, shape, device, inpaint_mask, inpaint_reference):
        """RePaint-style imputation: replace the known (unmasked) region of the
        just-produced x_{i-1} with the reference forward-noised to step i-1.

        inpaint_mask is 1.0 where the region is regenerated (free) and 0.0
        where it must equal the reference. At the final step (i == 0) the
        clean reference is used (no noise). Returns the projected sample so
        callers can write it back into out["sample"] BEFORE yielding (the
        loop yields before reading out["sample"], and the wrappers return
        final["sample"], so an in-place local edit would not survive).
        """
        if inpaint_mask is None or inpaint_reference is None:
            return sample
        if i == 0:
            known = inpaint_reference
        else:
            t_prev = th.tensor([i - 1] * shape[0], device=device)
            known = self.q_sample(
                inpaint_reference, t_prev, th.randn_like(inpaint_reference)
            )
        return inpaint_mask * sample + (1.0 - inpaint_mask) * known

    def _build_repaint_schedule(self, start_t, jump_length, jump_n_sample):
        """Build a RePaint jump/time-travel schedule over state timesteps x_t.

        Returns a list like [T-1, T-2, T-1, T-2, ... , 0, -1]. Consecutive
        entries differ by exactly one timestep; descending transitions are
        denoising steps, ascending transitions are forward noising (time
        travel). The terminal -1 sentinel preserves the final t=0 denoise step
        from the original monotonic sampler.
        """
        schedule = [start_t]
        if start_t <= 0 or jump_length <= 0 or jump_n_sample <= 1:
            schedule.extend(range(start_t - 1, -1, -1))
            schedule.append(-1)
            return schedule

        max_anchor = start_t - jump_length
        if max_anchor < jump_length:
            schedule.extend(range(start_t - 1, -1, -1))
            schedule.append(-1)
            return schedule

        jumps = {
            t: jump_n_sample - 1
            for t in range(jump_length, max_anchor + 1, jump_length)
        }
        t = start_t
        while t > 0:
            t -= 1
            schedule.append(t)
            repeats_left = jumps.get(t, 0)
            if repeats_left > 0:
                jumps[t] = repeats_left - 1
                for _ in range(jump_length):
                    t += 1
                    schedule.append(t)
        schedule.append(-1)
        return schedule

    def _repaint_time_travel(self, sample, t, const_noise=False):
        """Sample one exact forward diffusion step q(x_t | x_{t-1})."""
        noise = th.randn_like(sample)
        if const_noise:
            noise = noise[[0]].repeat(sample.shape[0], 1, 1, 1)
        return (
            _extract_into_tensor(self.sqrt_alphas, t, sample.shape) * sample
            + _extract_into_tensor(self.sqrt_betas, t, sample.shape) * noise
        )

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        dump_steps=None,
        const_noise=False,
        inpaint_mask=None,
        inpaint_reference=None,
        repaint_jump_length=0,
        repaint_jump_n_sample=1,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :param const_noise: If True, will noise all samples with the same noise throughout sampling
        :return: a non-differentiable batch of samples.
        """
        final = None
        if dump_steps is not None:
            dump = []
        sample_loop = self.p_sample_loop_progressive
        for i, sample in enumerate(sample_loop(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
            cond_fn_with_grad=cond_fn_with_grad,
            const_noise=const_noise,
            inpaint_mask=inpaint_mask,
            inpaint_reference=inpaint_reference,
            repaint_jump_length=repaint_jump_length,
            repaint_jump_n_sample=repaint_jump_n_sample,
        )):
            final = sample
            if dump_steps is not None and i in dump_steps:
                dump.append(deepcopy(sample["sample"]))
            
        if dump_steps is not None:
            return dump
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        const_noise=False,
        inpaint_mask=None,
        inpaint_reference=None,
        repaint_jump_length=0,
        repaint_jump_n_sample=1,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img)

        # Keep the original final t=0 denoise step in the transition-based
        # sampler by ending the monotonic schedule with a single sentinel.
        repaint_schedule = indices + [-1]
        if inpaint_mask is not None and inpaint_reference is not None:
            repaint_schedule = self._build_repaint_schedule(
                start_t=indices[0],
                jump_length=repaint_jump_length,
                jump_n_sample=repaint_jump_n_sample,
            )
        transitions = list(zip(repaint_schedule[:-1], repaint_schedule[1:]))

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            transitions = tqdm(transitions)

        for current_i, next_i in transitions:
            if next_i > current_i:
                t_forward = th.tensor([next_i] * shape[0], device=device)
                img = self._repaint_time_travel(
                    img, t_forward, const_noise=const_noise
                )
                continue

            t = th.tensor([current_i] * shape[0], device=device)
            if randomize_class and 'y' in model_kwargs:
                model_kwargs['y'] = th.randint(low=0, high=model.num_classes,
                                               size=model_kwargs['y'].shape,
                                               device=model_kwargs['y'].device)
            with th.no_grad():
                sample_fn = self.p_sample_with_grad if cond_fn_with_grad else self.p_sample
                out = sample_fn(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    const_noise=const_noise,
                )
                out["sample"] = self._inpaint_project(
                    out["sample"], current_i, shape, device,
                    inpaint_mask, inpaint_reference,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out_orig = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out_orig, x, t, model_kwargs=model_kwargs)
        else:
            out = out_orig

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out_orig["pred_xstart"]}

    def ddim_sample_with_grad(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        with th.enable_grad():
            x = x.detach().requires_grad_()
            out_orig = self.p_mean_variance(
                model,
                x,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            if cond_fn is not None:
                out = self.condition_score_with_grad(cond_fn, out_orig, x, t,
                                                     model_kwargs=model_kwargs)
            else:
                out = out_orig

        out["pred_xstart"] = out["pred_xstart"].detach()

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out_orig["pred_xstart"].detach()}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        dump_steps=None,
        const_noise=False,
        inpaint_mask=None,
        inpaint_reference=None,
        repaint_jump_length=0,
        repaint_jump_n_sample=1,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        if dump_steps is not None:
            raise NotImplementedError()
        if const_noise == True:
            raise NotImplementedError()

        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
            cond_fn_with_grad=cond_fn_with_grad,
            inpaint_mask=inpaint_mask,
            inpaint_reference=inpaint_reference,
            repaint_jump_length=repaint_jump_length,
            repaint_jump_n_sample=repaint_jump_n_sample,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        inpaint_mask=None,
        inpaint_reference=None,
        repaint_jump_length=0,
        repaint_jump_n_sample=1,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img)

        # Keep the original final t=0 denoise step in the transition-based
        # sampler by ending the monotonic schedule with a single sentinel.
        repaint_schedule = indices + [-1]
        if inpaint_mask is not None and inpaint_reference is not None:
            repaint_schedule = self._build_repaint_schedule(
                start_t=indices[0],
                jump_length=repaint_jump_length,
                jump_n_sample=repaint_jump_n_sample,
            )
        transitions = list(zip(repaint_schedule[:-1], repaint_schedule[1:]))

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            transitions = tqdm(transitions)

        for current_i, next_i in transitions:
            if next_i > current_i:
                t_forward = th.tensor([next_i] * shape[0], device=device)
                img = self._repaint_time_travel(img, t_forward)
                continue

            t = th.tensor([current_i] * shape[0], device=device)
            if randomize_class and 'y' in model_kwargs:
                model_kwargs['y'] = th.randint(low=0, high=model.num_classes,
                                               size=model_kwargs['y'].shape,
                                               device=model_kwargs['y'].device)
            with th.no_grad():
                sample_fn = self.ddim_sample_with_grad if cond_fn_with_grad else self.ddim_sample
                out = sample_fn(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                out["sample"] = self._inpaint_project(
                    out["sample"], current_i, shape, device,
                    inpaint_mask, inpaint_reference,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def _apply_joint_mask_training_perturbation(
        self, model, x_start, x_t, t, model_kwargs
    ):
        """Re-noise selected joints / spans without altering attention masks.

        Training-time structured corruption is meant to mimic RePaint's mixed
        reliability at the model input, not to hide tokens from attention.
        The selected joints / frames keep participating in attention; only
        their x_t features are replaced by q_sample(x_0, t_random).
        """
        y = model_kwargs.get('y') if model_kwargs is not None else None
        model_for_hooks = self._unwrap_model_for_training_hooks(model)

        subtree_mask = None
        if hasattr(model_for_hooks, 'sample_subtree_joint_mask_train'):
            subtree_mask = model_for_hooks.sample_subtree_joint_mask_train(
                model_kwargs.get('y', {}), x_t.shape[1], x_t.device
            )

        temporal_span_mask = None
        if hasattr(model_for_hooks, 'sample_temporal_span_mask_train'):
            temporal_span_mask = model_for_hooks.sample_temporal_span_mask_train(
                model_kwargs.get('y', {}), x_t.shape[1], x_t.shape[-1], x_t.device
            )

        corruption_mask = None
        if subtree_mask is not None:
            subtree_mask = subtree_mask.to(device=x_t.device, dtype=th.bool)
            expected_subtree_shape = (x_t.shape[0], x_t.shape[1])
            if subtree_mask.shape != expected_subtree_shape:
                raise ValueError(
                    "sample_subtree_joint_mask_train must return shape "
                    f"{expected_subtree_shape}, got {tuple(subtree_mask.shape)}"
                )
            corruption_mask = subtree_mask[:, :, None].expand(-1, -1, x_t.shape[-1])

        if temporal_span_mask is not None:
            temporal_span_mask = temporal_span_mask.to(device=x_t.device, dtype=th.bool)
            if temporal_span_mask.dim() == 2:
                temporal_span_mask = temporal_span_mask[:, None, :].expand(-1, x_t.shape[1], -1)
            expected_temporal_shape = (x_t.shape[0], x_t.shape[1], x_t.shape[-1])
            if temporal_span_mask.shape != expected_temporal_shape:
                raise ValueError(
                    "sample_temporal_span_mask_train must return shape "
                    f"{expected_temporal_shape} or (B, T), got {tuple(temporal_span_mask.shape)}"
                )
            corruption_mask = (
                temporal_span_mask
                if corruption_mask is None
                else (corruption_mask | temporal_span_mask)
            )

        if corruption_mask is None:
            if y is not None:
                y.pop('cross_limb_unreliable_mask', None)
            return x_t, temporal_span_mask

        if y is not None:
            y['cross_limb_unreliable_mask'] = corruption_mask.transpose(1, 2).to(
                dtype=x_t.dtype
            ).contiguous()

        t_random = th.randint(
            0, self.num_timesteps, t.shape, device=x_t.device, dtype=t.dtype
        )
        fresh_noise = th.randn_like(x_start)
        x_t_random = self.q_sample(x_start, t_random, noise=fresh_noise)
        return th.where(corruption_mask[:, :, None, :], x_t_random, x_t), temporal_span_mask

    @staticmethod
    def _unwrap_model_for_training_hooks(model):
        unwrapped_model = model
        while hasattr(unwrapped_model, 'model'):
            next_model = getattr(unwrapped_model, 'model')
            if next_model is None or next_model is unwrapped_model:
                break
            unwrapped_model = next_model
        return unwrapped_model

    @staticmethod
    def _sample_structured_dropout_mask(batch_size, drop_prob, device):
        if drop_prob <= 0.0 or batch_size <= 0:
            return th.zeros(batch_size, device=device, dtype=th.bool)
        expected = float(drop_prob) * float(batch_size)
        drop_count = int(math.floor(expected))
        if expected > drop_count and th.rand((), device=device).item() < (expected - drop_count):
            drop_count += 1
        drop_count = min(max(drop_count, 0), batch_size)
        if drop_count == 0:
            return th.zeros(batch_size, device=device, dtype=th.bool)
        drop_mask = th.zeros(batch_size, device=device, dtype=th.bool)
        drop_mask[th.randperm(batch_size, device=device)[:drop_count]] = True
        return drop_mask

    def _build_reference_conditioning(self, model, x_start, model_kwargs):
        y = model_kwargs.get('y') if model_kwargs is not None else None
        if y is None:
            return

        model_for_hooks = self._unwrap_model_for_training_hooks(model)
        if not getattr(model_for_hooks, 'reference_cond', False):
            y.pop('reference_motion', None)
            y.pop('reference_cond_mask', None)
            return

        batch_size = x_start.shape[0]
        device = x_start.device

        uncond_drop_prob = 1.0 - float(getattr(model_for_hooks, 'reference_cond_prob', 0.3))
        dropout_mask = self._sample_structured_dropout_mask(
            batch_size, uncond_drop_prob, device,
        )
        cond_mask = ~dropout_mask
        if not bool(cond_mask.any()):
            y['reference_motion'] = None
            y['reference_cond_mask'] = cond_mask
            return

        # Reuse x_start storage to avoid a per-step clone; callers must not
        # mutate x_start in-place after building reference conditioning.
        y['reference_motion'] = x_start.detach()
        y['reference_cond_mask'] = cond_mask

    def _build_global_energy_conditioning(self, model, x_start, model_kwargs):
        y = model_kwargs.get('y') if model_kwargs is not None else None
        if y is None:
            return

        model_for_hooks = self._unwrap_model_for_training_hooks(model)
        if not getattr(model_for_hooks, 'global_energy_cond', False):
            y.pop('global_energy_cond', None)
            return

        lengths = y.get('lengths')
        n_joints = y.get('n_joints')
        if lengths is None or n_joints is None:
            raise ValueError("global energy conditioning requires y['lengths'] and y['n_joints'] metadata")

        y['global_energy_cond'] = ReferencePriorEncoder.compute_global_energy_condition(
            x_start.detach(),
            n_joints=n_joints,
            lengths=lengths,
        )

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

        lengths = model_kwargs['y']['lengths']
        actual_joints = model_kwargs['y']['n_joints']
        joints_padding_mask = model_kwargs['y']['joints_padding_mask'][:, :, :, 1, 1:]
        mean = model_kwargs['y']['mean'][..., None]
        std = model_kwargs['y']['std'][..., None]

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        # Subtree perturbation: when the model selects a subset of joints
        # (via joint_mask_prob), replace those joints' x_t slice with
        # q_sample(x_0, t_random, fresh_noise) -- same x_0 ground truth (so
        # the loss target is unchanged) but at a random independent timestep
        # with fresh independent noise. The selected joints continue to
        # participate in attention normally; the model must learn to denoise
        # them despite their noise level disagreeing with the rest of the
        # batch sample, which is exactly what RePaint clamping produces at
        # inference (clamped joints are at a fixed reference's q_sample state
        # that's uncorrelated with the in-flight masked joint's trajectory).
        x_t, temporal_span_mask = self._apply_joint_mask_training_perturbation(
            model, x_start, x_t, t, model_kwargs
        )
        self._build_reference_conditioning(model, x_start, model_kwargs)
        self._build_global_energy_conditioning(model, x_start, model_kwargs)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1).float()
                with self._fp32_math_context(x_t, x_start, frozen_out):
                    terms["vb"] = self._vb_terms_bpd(
                        model=lambda *args, r=frozen_out: r,
                        x_start=x_start.float(),
                        x_t=x_t.float(),
                        t=t,
                        clip_denoised=False,
                    )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape  # [bs, njoints, nfeats, nframes]
            with self._fp32_math_context(model_output, target, mean, std):
                target_fp32 = target.float()
                model_output_fp32 = model_output.float()
                joints_padding_mask_fp32 = joints_padding_mask.float()
                lengths_fp32 = lengths.float()
                actual_joints_fp32 = actual_joints.float()
                mean_fp32 = mean.float()
                std_fp32 = std.float()

                terms["l_simple"] = self.spatial_masked_l2(
                    target_fp32, model_output_fp32, joints_padding_mask_fp32, lengths_fp32, actual_joints_fp32
                )
                terms["loss"] = terms["l_simple"].clone()

                temporal_span_seam_weights = self._build_temporal_span_seam_weights(
                    temporal_span_mask, lengths
                )
                if (
                    self.temporal_span_seam_loss_weight > 0.0
                    and temporal_span_seam_weights is not None
                ):
                    seam_weights = (
                        temporal_span_seam_weights
                        * joints_padding_mask_fp32.transpose(1, 3)
                    )
                    if bool(seam_weights.any()):
                        terms["temporal_span_seam_loss"] = self.weighted_feature_l2(
                            target_fp32, model_output_fp32, seam_weights
                        )
                        terms["loss"] = (
                            terms["loss"]
                            + self.temporal_span_seam_loss_weight * terms["temporal_span_seam_loss"]
                        )

                target_denorm = (target_fp32 * std_fp32) + mean_fp32
                model_output_denorm = (model_output_fp32 * std_fp32) + mean_fp32

                if self.lambda_geo > 0.:
                    terms["geodesic_loss"] = self.geodesic_loss(
                        target_denorm, model_output_denorm, joints_padding_mask_fp32, lengths_fp32, actual_joints_fp32
                    )
                    terms["loss"] = terms["loss"] + self.lambda_geo * terms["geodesic_loss"]

                if self.lambda_vel > 0.:
                    terms["vel_loss"] = self.velocity_consistency_loss(
                        model_output_denorm, joints_padding_mask_fp32, lengths_fp32, actual_joints_fp32
                    )
                    terms["loss"] = terms["loss"] + self.lambda_vel * terms["vel_loss"]

                if self.lambda_loop_wrap > 0.0:
                    y = model_kwargs.get('y', {}) if isinstance(model_kwargs, dict) else {}
                    loop_terms = self.loop_wrap_loss(
                        model_output_denorm,
                        y,
                        lengths,
                        actual_joints,
                    )
                    terms.update(loop_terms)
                    terms["loss"] = terms["loss"] + self.lambda_loop_wrap * terms["loop_wrap_loss"]

                if self.lambda_loop_root_xz > 0.0:
                    y = model_kwargs.get('y', {}) if isinstance(model_kwargs, dict) else {}
                    terms["loop_root_xz_loss"] = self.loop_root_xz_closure_loss(
                        model_output_denorm,
                        y,
                        lengths,
                        actual_joints,
                    )
                    terms["loss"] = terms["loss"] + self.lambda_loop_root_xz * terms["loop_root_xz_loss"]

        else:
            raise NotImplementedError(self.loss_type)

        return terms

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = _cached_extract_source_tensor(arr, timesteps.device)[timesteps]
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
