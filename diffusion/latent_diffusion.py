"""Latent-only diffusion: x0 MSE, no motion losses, clipping or editing."""
import numpy as np
import torch

from diffusion.gaussian_diffusion import (
    ModelMeanType, ModelVarType, LossType, get_named_beta_schedule)
from diffusion.respace import SpacedDiffusion, space_timesteps
from utils.device_transfer import check_on_device


EDIT_KEYS = ('joint_mask', 'temporal_span_mask', 'cross_limb_unreliable_mask',
             'inpaint_mask', 'inpaint_reference', 'inpaint_joints', 'inpaint_frames',
             'outpaint', 'outpaint_frames', 'renoise', 'ref_frame_mask')


def validate_latent_conditions(y, frames=60):
    if y is None:
        raise ValueError('Latent diffusion requires skeleton conditions')
    for key in EDIT_KEYS:
        if y.get(key) is not None:
            raise ValueError(f'{key} is unsupported by latent diffusion')
    for key in y:
        if key.startswith('ref_'):
            raise ValueError(f'{key} is unsupported by latent diffusion; use the autoencoder for motion transfer')
    if frames != 60:
        raise ValueError('Latent diffusion requires a fixed 60-frame window')
    # Short-circuit before querying CUDA stream state: Dynamo cannot trace the
    # Python bool returned by is_current_stream_capturing on CUDA-enabled hosts.
    if ('lengths' in y and not torch.compiler.is_compiling()
            and not (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())):
        check_on_device((y['lengths'] == frames).all(),
                        'Latent diffusion does not support padded/invalid frames')


class LatentDiffusion(SpacedDiffusion):
    def __init__(self, *args, loss_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_weights = None
        if loss_weights is not None:
            weights = torch.as_tensor(loss_weights).detach().float()
            if (weights.ndim != 2 or not torch.isfinite(weights).all()
                    or not (weights > 0).all()
                    or not torch.isclose(weights.mean(), weights.new_tensor(1.), atol=1e-5)):
                raise ValueError('Latent loss weights must be positive [K,d] with mean one')
            self.loss_weights = weights

    def latent_error_terms(self, prediction, target):
        if prediction.shape != target.shape:
            raise ValueError('Latent prediction shape differs from target')
        error = (prediction.float() - target.float()).square()
        mse = error.mean((1, 2, 3))
        if self.loss_weights is None:
            return {'loss': mse, 'latent_mse': mse}
        if tuple(self.loss_weights.shape) != tuple(target.shape[-2:]):
            raise ValueError('Latent loss weights and target layout mismatch')
        weighted = (error * self.loss_weights.to(target.device)).mean((1, 2, 3))
        return {'loss': weighted, 'latent_mse': mse, 'latent_sensitivity_mse': weighted}

    def p_mean_variance(self, model, x, t, clip_denoised=False, **kwargs):
        if clip_denoised:
            raise ValueError('Motion-domain clipping is unsupported for standardized latents')
        return super().p_mean_variance(model, x, t, clip_denoised=False, **kwargs)

    def _inpaint_project(self, sample, i, shape, device, inpaint_mask, inpaint_reference):
        if inpaint_mask is not None or inpaint_reference is not None:
            raise ValueError('Inpaint/outpaint is unsupported by latent diffusion')
        return sample

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        y = (model_kwargs or {}).get('y')
        validate_latent_conditions(y, x_start.shape[1])
        noise = torch.randn_like(x_start) if noise is None else noise
        noisy = self.q_sample(x_start.float(), t, noise.float())
        prediction = self._wrap_model(model)(noisy, t, **(model_kwargs or {}))
        return self.latent_error_terms(prediction, x_start)

    @torch.no_grad()
    def sample(self, model, shape, y, device, noise=None, ddim=False, progress=False):
        validate_latent_conditions(y, shape[1])
        x = torch.randn(*shape, device=device) if noise is None else noise.to(device)
        if tuple(x.shape) != tuple(shape):
            raise ValueError('Sampling noise shape differs from latent layout')
        steps = range(self.num_timesteps - 1, -1, -1)
        if progress:
            from tqdm import tqdm
            steps = tqdm(steps, desc='latent diffusion')
        for index in steps:
            t = torch.full((shape[0],), index, device=device, dtype=torch.long)
            method = self.ddim_sample if ddim else self.p_sample
            x = method(model, x, t, clip_denoised=False, model_kwargs={'y': y})['sample']
        return x


def create_latent_diffusion(steps=100, noise_schedule='cosine', shift=1.0, respacing='', loss_weights=None):
    if steps < 2 or not np.isfinite(shift) or shift <= 0:
        raise ValueError('Invalid diffusion schedule')
    return LatentDiffusion(
        use_timesteps=space_timesteps(steps, respacing or [steps]),
        betas=get_named_beta_schedule(noise_schedule, steps, shift=shift),
        model_mean_type=ModelMeanType.START_X, model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE, rescale_timesteps=False, loss_weights=loss_weights)
