"""One reverse-diffusion run over a batch: plain, img2img, inpaint, outpaint.

``_sample_batch`` is the single call site of the diffusion samplers for
generation. It owns the seed, the top-level autocast context and the four
start/clamp combinations (pure noise, noised reference, reference-clamped
mask, both), so ``sample.generate`` only decides WHICH one to run.
"""
import torch

from utils.fixseed import fixseed


def _sample_batch(
    diffusion,
    model,
    model_kwargs,
    sampling_method,
    sample_shape,
    ddim_eta,
    seed,
    device,
    reference_motion=None,
    skip_timesteps=0,
    inpaint_mask=None,
    autocast_dtype=None,
):
    skip_timesteps = int(skip_timesteps) if skip_timesteps is not None else 0

    inpainting = inpaint_mask is not None
    if inpainting:
        if reference_motion is None:
            raise ValueError("inpaint_mask given without a reference_motion")
        if int(reference_motion.shape[-1]) != int(sample_shape[-1]):
            raise ValueError(
                "Motion inpainting requires reference_motion frame count to match target sample length; "
                f"got reference {reference_motion.shape[-1]} and target {sample_shape[-1]}"
            )
    def _prepared_cross_limb_unreliable_mask_from_inpaint_mask(inpaint_mask_):
        if inpaint_mask_ is None:
            return None
        if inpaint_mask_.dim() != 4 or inpaint_mask_.shape[2] != 1:
            raise ValueError(
                f"inpaint_mask must have shape [B, J, 1, T], got {tuple(inpaint_mask_.shape)}"
            )
        raw_cross_limb_unreliable_mask = inpaint_mask_.squeeze(2).permute(0, 2, 1).contiguous()
        reliable_tpose = torch.zeros(
            (
                raw_cross_limb_unreliable_mask.shape[0],
                1,
                raw_cross_limb_unreliable_mask.shape[2],
            ),
            device=raw_cross_limb_unreliable_mask.device,
            dtype=raw_cross_limb_unreliable_mask.dtype,
        )
        return torch.cat([reliable_tpose, raw_cross_limb_unreliable_mask], dim=1).transpose(0, 1).contiguous()

    def _copy_model_kwargs_for_loop(cross_limb_unreliable_mask_):
        if model_kwargs is None:
            loop_model_kwargs = {}
            loop_y = {}
        else:
            loop_model_kwargs = dict(model_kwargs)
            loop_y = dict(model_kwargs.get('y', {}))
        if cross_limb_unreliable_mask_ is None:
            loop_y.pop('cross_limb_unreliable_mask', None)
        else:
            loop_y['cross_limb_unreliable_mask'] = cross_limb_unreliable_mask_
        loop_model_kwargs['y'] = loop_y
        return loop_model_kwargs

    def _autocast_context():
        # Top-level autocast around the reverse-diffusion loop (model invoked
        # many times inside the sampler, so this is the natural call site).
        if autocast_dtype is None:
            return torch.autocast(device_type=device.type, enabled=False)
        return torch.autocast(device_type=device.type, dtype=autocast_dtype)

    def _run_loop(noise, init_image, skip_ts, inpaint_mask_, inpaint_reference_, cross_limb_unreliable_mask_):
        common_kwargs = dict(
            model=model,
            shape=sample_shape,
            noise=noise,
            clip_denoised=False,
            model_kwargs=_copy_model_kwargs_for_loop(cross_limb_unreliable_mask_),
            device=device,
            init_image=init_image,
            skip_timesteps=skip_ts,
        )
        # Only p_* / ddim_* loops accept the inpaint kwargs.
        inpaint_kwargs = dict(
            inpaint_mask=inpaint_mask_, inpaint_reference=inpaint_reference_
        )
        with _autocast_context():
            if sampling_method == 'ddim':
                return diffusion.ddim_sample_loop(
                    progress=True,
                    eta=ddim_eta,
                    **inpaint_kwargs,
                    **common_kwargs,
                )
            if sampling_method in ('p', 'ddpm'):
                return diffusion.p_sample_loop(
                    progress=True,
                    dump_steps=None,
                    const_noise=False,
                    **inpaint_kwargs,
                    **common_kwargs,
                )
        raise ValueError(f'Unknown sampling_method: {sampling_method}')

    if inpainting and skip_timesteps > 0:
        # Start from noised reference but clamp unmasked region to original reference at every step.
        ref = reference_motion.to(device, non_blocking=True)
        mask = inpaint_mask.to(device, non_blocking=True)
        prepared_cross_limb_unreliable_mask = _prepared_cross_limb_unreliable_mask_from_inpaint_mask(mask)

        fixseed(seed)
        return _run_loop(
            noise=torch.randn(sample_shape, device=device),
            init_image=ref,
            skip_ts=skip_timesteps,
            inpaint_mask_=mask,
            inpaint_reference_=ref,
            cross_limb_unreliable_mask_=prepared_cross_limb_unreliable_mask,
        )

    fixseed(seed)
    if inpainting:
        # skip_timesteps=0: start from pure noise; reference is only the per-step clamp source for unmasked region.
        mask = inpaint_mask.to(device, non_blocking=True)
        prepared_cross_limb_unreliable_mask = _prepared_cross_limb_unreliable_mask_from_inpaint_mask(mask)
        return _run_loop(
            noise=torch.randn(sample_shape, device=device),
            init_image=None,
            skip_ts=0,
            inpaint_mask_=mask,
            inpaint_reference_=reference_motion.to(device, non_blocking=True),
            cross_limb_unreliable_mask_=prepared_cross_limb_unreliable_mask,
        )
    if reference_motion is not None and skip_timesteps > 0:
        # img2img: noise the whole reference to an intermediate step (higher = more faithful).
        return _run_loop(
            noise=torch.randn(sample_shape, device=device),
            init_image=reference_motion.to(device, non_blocking=True),
            skip_ts=skip_timesteps,
            inpaint_mask_=None,
            inpaint_reference_=None,
            cross_limb_unreliable_mask_=None,
        )
    # Plain generation: full denoising from pure noise.
    return _run_loop(
        noise=torch.randn(sample_shape, device=device),
        init_image=None,
        skip_ts=0,
        inpaint_mask_=None,
        inpaint_reference_=None,
        cross_limb_unreliable_mask_=None,
    )
