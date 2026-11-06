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
from utils.rotation_conversions import rotation_6d_to_matrix_safe
from data_loaders.truebones.truebones_utils.canonical_features import (
    CANONICAL_DPOS_STD_KEY,
    CANONICAL_STD_KEY,
    DPOS_LOSS_WEIGHT_KEY,
    MODEL_CHANNELS,
    PARENT_INDEX_KEY,
    ROOT_STAT_SLICE,
    ROOT_XZ_CHANNELS,
    _length_scale_from_cond,
    get_canonical_global_stats,
    _view_stat_torch,
    _check_root_stats,
    canonical_to_physical_hml,
    root_relative_positions,
)
from utils.device_transfer import host_to_device


from train.motion_reconstruction_losses import (
    MotionGeometryLosses, BONE_REST_FLOOR_FRAC, LIMB_CORR_LAG_FRACS,
    LIMB_CORR_ENERGY_FLOOR, ROTATION_STEP_SCALE_RAD,
)


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


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, scale_betas=1., shift=1.,
                            shift_min=None):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.

    ``shift`` moves every noise level of the named schedule toward noise
    (see ``shift_beta_schedule``); 1 is the named schedule itself.
    ``shift_min`` is the shift of the lowest-noise step (None = ``shift``).
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = scale_betas * 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        betas = betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")
    return shift_beta_schedule(betas, shift, shift_min=shift_min)


def shift_beta_schedule(betas, shift, max_beta=0.999, shift_min=None):
    """Divide the SNR of every step by ``shift ** 2`` (log-SNR minus ``2 ln shift``).

    The step count and the uniform training draw over steps are unchanged, so a
    shift > 1 puts more of the steps -- and more of the training samples -- at
    high noise, where a clip's global structure (its tempo, its posture) is
    decided, and fewer at the low-noise detail end. ``shift == 1`` returns
    ``betas`` untouched.

    ``shift_min`` (<= ``shift``; None = ``shift``) is the shift of the first,
    lowest-noise step. Over the steps where ``betas`` has SNR > 1, ln(shift)
    moves linearly in ln(sigma) from ``shift_min`` at the first step to
    ``shift`` at SNR 1; the noisier steps keep ``shift``. A ``shift_min``
    below 1 lowers the smallest noise level, so motion smaller than the
    plain schedule's first-step noise still has steps where it stands above
    the noise, while the high-noise half is the plain shifted schedule.
    """
    shift = float(shift)
    shift_min = shift if shift_min is None else float(shift_min)
    for name, value in (('shift', shift), ('shift_min', shift_min)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"noise schedule {name} must be finite and > 0, got {value}")
    # A shift falling with t could raise the SNR from one step to the next.
    if shift_min > shift:
        raise ValueError(f"noise schedule shift_min ({shift_min}) must be <= shift ({shift})")
    betas = np.asarray(betas, dtype=np.float64)
    if shift == 1.0 and shift_min == 1.0:
        return betas
    alphas_cumprod = np.cumprod(1.0 - betas)
    log_sigma = 0.5 * (np.log1p(-alphas_cumprod) - np.log(alphas_cumprod))
    shifts = np.full_like(alphas_cumprod, shift)
    if shift_min != shift:
        if log_sigma[0] >= 0.0:
            raise ValueError("noise schedule shift_min needs a first step with SNR > 1")
        # 1 at the first step, 0 from SNR 1 on.
        ramp = np.clip(log_sigma / log_sigma[0], 0.0, 1.0)
        shifts = np.exp(math.log(shift) + ramp * (math.log(shift_min) - math.log(shift)))
    shifted = alphas_cumprod / (alphas_cumprod + shifts ** 2 * (1.0 - alphas_cumprod))
    previous = np.concatenate(([1.0], shifted[:-1]))
    return np.minimum(1.0 - shifted / previous, max_beta)


def matching_shifted_timestep(betas, shifted_betas, t):
    """The (fractional) step of ``shifted_betas`` at the noise level step ``t``
    has in ``betas``; ``t`` itself when the two schedules are the same."""
    def log_snr(b):
        alphas_cumprod = np.cumprod(1.0 - np.asarray(b, dtype=np.float64))
        return np.log(alphas_cumprod) - np.log1p(-alphas_cumprod)
    steps = np.arange(len(betas), dtype=np.float64)
    target = np.interp(float(t), steps, log_snr(betas))
    # log-SNR falls with t; np.interp needs increasing x.
    return float(np.interp(-target, -log_snr(shifted_betas), steps))


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


class GaussianDiffusion(MotionGeometryLosses):
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
        lambda_drot=0.,
        drot_joint_boost=0.,
        lambda_dpos=0.,
        lambda_root_dpos=0.,
        lambda_root_acc=0.,
        lambda_loop_wrap=0.,
        lambda_bone=0.,
        lambda_limb_corr=0.,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        self.lambda_geo = lambda_geo
        self.lambda_drot = float(lambda_drot)
        self.drot_joint_boost = float(drot_joint_boost)
        if not math.isfinite(self.drot_joint_boost) or self.drot_joint_boost < 0.0:
            raise ValueError("drot_joint_boost must be finite and >= 0")
        if not math.isfinite(self.lambda_drot) or self.lambda_drot < 0.0:
            raise ValueError(f"lambda_drot must be finite and >= 0, got {self.lambda_drot}")
        self.lambda_dpos = float(lambda_dpos)
        self.lambda_root_dpos = float(lambda_root_dpos)
        self.lambda_root_acc = float(lambda_root_acc)
        self.lambda_loop_wrap = float(lambda_loop_wrap)
        self.lambda_bone = float(lambda_bone)
        self.lambda_limb_corr = float(lambda_limb_corr)
        if self.lambda_loop_wrap < 0.0:
            raise ValueError(f"lambda_loop_wrap must be >= 0, got {self.lambda_loop_wrap}")
        if self.lambda_bone < 0.0:
            raise ValueError(f"lambda_bone must be >= 0, got {self.lambda_bone}")
        if self.lambda_limb_corr < 0.0:
            raise ValueError(f"lambda_limb_corr must be >= 0, got {self.lambda_limb_corr}")
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
    
    def rotation_step_loss(self, pred_physical, target_physical, spat_mask, lengths, y,
                           t=None, rotations=None):
        loss = super().rotation_step_loss(
            pred_physical, target_physical, spat_mask, lengths, y, rotations=rotations)
        if t is not None:
            loss = loss * extract_into_tensor(self.alphas_cumprod, t, (pred_physical.shape[0],)).to(loss.dtype)
        return loss

    def bone_vector_consistency_loss(self, pred_physical, target_physical, spat_mask, y, t=None, rotations=None):
        signal = None
        if t is not None:
            signal = extract_into_tensor(self.alphas_cumprod, t, (pred_physical.shape[0], 1, 1)).to(pred_physical.dtype)
        return super().bone_vector_consistency_loss(
            pred_physical, target_physical, spat_mask, y, rotations=rotations, sample_weight=signal)

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(
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
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
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
                min_log = extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = extract_into_tensor(np.log(self.betas), t, x.shape)
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


            model_variance = extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = extract_into_tensor(model_log_variance, t, x.shape)

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
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

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
        alpha_bar = extract_into_tensor(self.alphas_cumprod, t, x.shape)

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
        alpha_bar = extract_into_tensor(self.alphas_cumprod, t, x.shape)

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
        noise_sampler=None,
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
        noise = th.randn_like(x) if noise_sampler is None else noise_sampler(x)
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
        noise_sampler=None,
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
            noise_sampler=noise_sampler,
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
        noise_sampler=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if noise_sampler is not None and (cond_fn_with_grad or randomize_class):
            raise ValueError('Custom noise supports plain, non-gradient sampling only')
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = (th.randn(*shape, device=device) if noise_sampler is None
                   else noise_sampler(th.empty(*shape, device=device)))

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img)

        schedule = indices + [-1]
        transitions = list(zip(schedule[:-1], schedule[1:]))

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            transitions = tqdm(transitions)

        for current_i, next_i in transitions:
            t = th.tensor([current_i] * shape[0], device=device)
            if randomize_class and 'y' in model_kwargs:
                model_kwargs['y'] = th.randint(low=0, high=model.num_classes,
                                               size=model_kwargs['y'].shape,
                                               device=model_kwargs['y'].device)
            with th.no_grad():
                sample_fn = self.p_sample_with_grad if cond_fn_with_grad else self.p_sample
                noise_kwargs = {} if noise_sampler is None else {'noise_sampler': noise_sampler}
                out = sample_fn(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    **noise_kwargs,
                    const_noise=const_noise,
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
        noise_sampler=None,
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

        alpha_bar = extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x) if noise_sampler is None else noise_sampler(x)
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

        alpha_bar = extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
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
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

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
        noise_sampler=None,
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
            noise_sampler=noise_sampler,
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
        noise_sampler=None,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if noise_sampler is not None and (cond_fn_with_grad or randomize_class):
            raise ValueError('Custom noise supports plain, non-gradient sampling only')
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = (th.randn(*shape, device=device) if noise_sampler is None
                   else noise_sampler(th.empty(*shape, device=device)))

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img)

        schedule = indices + [-1]
        transitions = list(zip(schedule[:-1], schedule[1:]))

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            transitions = tqdm(transitions)

        for current_i, next_i in transitions:
            t = th.tensor([current_i] * shape[0], device=device)
            if randomize_class and 'y' in model_kwargs:
                model_kwargs['y'] = th.randint(low=0, high=model.num_classes,
                                               size=model_kwargs['y'].shape,
                                               device=model_kwargs['y'].device)
            with th.no_grad():
                sample_fn = self.ddim_sample_with_grad if cond_fn_with_grad else self.ddim_sample
                noise_kwargs = {} if noise_sampler is None else {'noise_sampler': noise_sampler}
                out = sample_fn(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    **noise_kwargs,
                    eta=eta,
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

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

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
            with self._fp32_math_context(model_output, target):
                target_fp32 = target.float()
                model_output_fp32 = model_output.float()
                joints_padding_mask_fp32 = joints_padding_mask.float()
                lengths_fp32 = lengths.float()
                actual_joints_fp32 = actual_joints.float()

                # Position and rotation only: velocity is not a model channel.
                n_model_channels = len(MODEL_CHANNELS)
                terms["l_simple"] = self.spatial_masked_l2(
                    target_fp32[:, :, :n_model_channels], model_output_fp32[:, :, :n_model_channels],
                    joints_padding_mask_fp32, lengths_fp32, actual_joints_fp32,
                )
                terms["loss"] = terms["l_simple"].clone()

                y_for_decode = (model_kwargs or {}).get('y', {})
                if self.lambda_dpos > 0.:
                    terms["dpos_loss"] = self.position_step_loss(
                        target_fp32, model_output_fp32, joints_padding_mask_fp32,
                        actual_joints_fp32, y_for_decode,
                    )
                    terms["loss"] = terms["loss"] + self.lambda_dpos * terms["dpos_loss"]
                if self.lambda_root_dpos > 0.:
                    terms["root_dpos_loss"] = self.root_position_step_loss(
                        target_fp32, model_output_fp32, y_for_decode,
                    )
                    terms["loss"] = terms["loss"] + self.lambda_root_dpos * terms["root_dpos_loss"]
                if self.lambda_root_acc > 0.:
                    terms["root_acc_loss"] = self.root_acceleration_loss(
                        target_fp32, model_output_fp32, y_for_decode,
                    )
                    terms["loss"] = terms["loss"] + self.lambda_root_acc * terms["root_acc_loss"]

                step_rotations = None
                if self.lambda_drot > 0.:
                    target_physical, target_global_rot = canonical_to_physical_hml(
                        target_fp32, y_for_decode, return_canonical_rotations=True,
                    )
                    model_output_physical, pred_global_rot = canonical_to_physical_hml(
                        model_output_fp32, y_for_decode, return_canonical_rotations=True,
                    )
                    step_rotations = (target_global_rot, pred_global_rot)
                else:
                    target_physical = canonical_to_physical_hml(target_fp32, y_for_decode)
                    model_output_physical = canonical_to_physical_hml(model_output_fp32, y_for_decode)


                # The geodesic and bone losses share physical local matrices.
                rotations = None
                if self.lambda_geo > 0. and self.lambda_bone > 0.:
                    rotations = (
                        self._physical_rotation_matrices(target_physical),
                        self._physical_rotation_matrices(model_output_physical),
                    )

                if self.lambda_drot > 0.:
                    terms["drot_loss"] = self.rotation_step_loss(
                        model_output_physical, target_physical, joints_padding_mask_fp32,
                        lengths_fp32, y_for_decode, t=t, rotations=step_rotations,
                    )
                    terms["loss"] = terms["loss"] + self.lambda_drot * terms["drot_loss"]

                if self.lambda_geo > 0.:
                    terms["geodesic_loss"] = self.geodesic_loss(
                        target_physical, model_output_physical, joints_padding_mask_fp32,
                        lengths_fp32, actual_joints_fp32, rotations=rotations,
                    )
                    terms["loss"] = terms["loss"] + self.lambda_geo * terms["geodesic_loss"]

                if self.lambda_bone > 0.:
                    terms["bone_loss"] = self.bone_vector_consistency_loss(
                        model_output_physical, target_physical,
                        joints_padding_mask_fp32, y_for_decode, t=t, rotations=rotations,
                    )
                    terms["loss"] = terms["loss"] + self.lambda_bone * terms["bone_loss"]

                if self.lambda_limb_corr > 0.:
                    terms["limb_corr_loss"] = self.limb_correlation_loss(
                        model_output_physical, target_physical, y_for_decode, lengths_fp32,
                    )
                    terms["loss"] = terms["loss"] + self.lambda_limb_corr * terms["limb_corr_loss"]

                if self.lambda_loop_wrap > 0.0:
                    y = model_kwargs.get('y', {}) if isinstance(model_kwargs, dict) else {}
                    loop_terms = self.loop_wrap_loss(
                        model_output_physical,
                        y,
                        actual_joints,
                    )
                    terms.update(loop_terms)
                    terms["loss"] = terms["loss"] + self.lambda_loop_wrap * terms["loop_wrap_loss"]

        else:
            raise NotImplementedError(self.loss_type)

        return terms

def extract_into_tensor(arr, timesteps, broadcast_shape):
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
