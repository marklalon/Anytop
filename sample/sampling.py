"""Reverse diffusion for plain generation and whole-reference img2img."""
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
    autocast_dtype=None,
    row_seeds=None,
    progress=True,
):
    skip_timesteps = int(skip_timesteps) if skip_timesteps is not None else 0

    # Separate generators keep each row's diffusion noise independent of batching.
    noise_sampler = None
    if row_seeds is not None:
        if len(row_seeds) != sample_shape[0]:
            raise ValueError('row_seeds must contain one seed per batch row')
        if reference_motion is not None or skip_timesteps:
            raise ValueError('row_seeds supports plain generation only')
        generators = [torch.Generator(device=device).manual_seed(int(s)) for s in row_seeds]

        def noise_sampler(like):
            return torch.cat([
                torch.randn((1, *like.shape[1:]), generator=g, device=like.device, dtype=like.dtype)
                for g in generators
            ])

    def _autocast_context():
        # Top-level autocast around the reverse-diffusion loop (model invoked
        # many times inside the sampler, so this is the natural call site).
        if autocast_dtype is None:
            return torch.autocast(device_type=device.type, enabled=False)
        return torch.autocast(device_type=device.type, dtype=autocast_dtype)

    def _run_loop(noise, init_image, skip_ts):
        common_kwargs = dict(
            model=model,
            shape=sample_shape,
            noise=noise,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            device=device,
            init_image=init_image,
            skip_timesteps=skip_ts,
        )
        if noise_sampler is not None:
            common_kwargs['noise_sampler'] = noise_sampler
        with _autocast_context():
            if sampling_method == 'ddim':
                return diffusion.ddim_sample_loop(
                    progress=progress,
                    eta=ddim_eta,
                    **common_kwargs,
                )
            if sampling_method in ('p', 'ddpm'):
                return diffusion.p_sample_loop(
                    progress=progress,
                    dump_steps=None,
                    const_noise=False,
                    **common_kwargs,
                )
        raise ValueError(f'Unknown sampling_method: {sampling_method}')

    fixseed(seed)
    if reference_motion is not None and skip_timesteps > 0:
        # img2img: noise the whole reference to an intermediate step (higher = more faithful).
        return _run_loop(
            noise=torch.randn(sample_shape, device=device),
            init_image=reference_motion.to(device, non_blocking=True),
            skip_ts=skip_timesteps,
        )
    # Plain generation: full denoising from pure noise.
    return _run_loop(
        noise=(torch.randn(sample_shape, device=device) if noise_sampler is None
               else noise_sampler(torch.empty(sample_shape, device=device))),
        init_image=None,
        skip_ts=0,
    )
