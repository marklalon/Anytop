"""Classifier-free guidance wrappers used at sampling time.

Training holds the other half of this contract: ``--action_label_cfg_drop_prob``
hard-drops the action condition on a fraction of the samples (see
``AnyTop._resolve_action_label_active``), substituting the learned
``action_label_null_emb``. That is what gives the model a real unconditional
mode. Sampling then runs the denoiser twice per diffusion
step -- once with the label, once with the condition forced to that null mode --
and extrapolates away from the unconditional prediction:

    out = out_uncond + s * (out_cond - out_uncond)

AnyTop predicts x_0 rather than eps, so the extrapolation is on x_0; the algebra
is the same, and s = 1 collapses to the plain conditional prediction.
"""

import math

import torch
import torch.nn as nn


class ClassifierFreeActionModel(nn.Module):
    """Wrap an AnyTop denoiser with CFG over the action-label condition.

    Only the action condition is guided. The unconditional pass differs from the
    conditional one by exactly ``y['action_label_active'] = False``, so every
    other channel (species FiLM, the canonical output frame, loop/resample_speed, the
    skeleton graph itself) is bit-identical between the two and cancels out of
    the guidance term instead of being extrapolated along with the prompt.

    The two passes run SEQUENTIALLY rather than as one 2B batch. ``y`` carries
    per-sample python lists (parents, joint names, metadata) next to its tensors,
    so a batched pass would have to hand-duplicate every entry, and the padded
    joint/temporal masks would double in memory for no change in the amount of
    attention actually computed -- both passes here run the full model.
    """

    def __init__(self, model, guidance_scale):
        super().__init__()
        self.model = model
        guidance_scale = float(guidance_scale)
        if not math.isfinite(guidance_scale):
            raise ValueError(f"guidance_scale must be finite, got {guidance_scale}")
        if guidance_scale < 0.0:
            raise ValueError(
                f"guidance_scale must be >= 0, got {guidance_scale}. A negative scale "
                "extrapolates away from the prompt, which is not what --action_label means."
            )
        self.guidance_scale = guidance_scale

    def __getattr__(self, name):
        # Transparent stand-in for the wrapped denoiser: generate.py reads
        # model.feature_len, and unwrap_anytop_model walks `.model` down to the
        # AnyTop, so anything not defined on the wrapper resolves on the model.
        try:
            return super().__getattr__(name)
        except AttributeError:
            inner = self._modules.get('model')
            if inner is None or name == 'model':
                raise
            return getattr(inner, name)

    def forward(self, x, timesteps, y=None, **kwargs):
        cond_out = self.model(x, timesteps, y=y, **kwargs)
        if self.guidance_scale == 1.0:
            return cond_out
        uncond_y = dict(y or {})
        # Force the null condition for every row. _resolve_action_label_active
        # gives an explicit mask precedence over the training Bernoulli, and
        # _build_action_label_token routes a False row to action_label_null_emb.
        uncond_y['action_label_active'] = torch.zeros(
            x.shape[0], dtype=torch.bool, device=x.device
        )
        uncond_out = self.model(x, timesteps, y=uncond_y, **kwargs)
        return uncond_out + self.guidance_scale * (cond_out - uncond_out)


class ClassifierFreeSpeciesModel(ClassifierFreeActionModel):
    """Guide per-joint species FiLM using its trained identity bypass.

    Other conditions, including the joint names, remain active in both passes.
    """

    def forward(self, x, timesteps, y=None, **kwargs):
        cond_out = self.model(x, timesteps, y=y, **kwargs)
        if self.guidance_scale == 1.0:
            return cond_out
        uncond_y = dict(y or {})
        uncond_y['species_active'] = torch.zeros(
            x.shape[0], dtype=torch.bool, device=x.device
        )
        uncond_out = self.model(x, timesteps, y=uncond_y, **kwargs)
        return uncond_out + self.guidance_scale * (cond_out - uncond_out)


class ClassifierFreeReferenceModel(ClassifierFreeActionModel):
    """Sequential ref/text composition.

    The model bypasses the species FiLM on every reference-active row, so the
    reference and full passes run with species off. The null-reference base pass
    keeps whatever ``y['species_active']`` says; encode sets it False, which makes
    species identical across all passes.

    ``ref_scale`` applies only to the high-noise steps, where global posture
    (facing, body height) is settled; the detail steps below
    ``REF_CFG_SWITCH * num_timesteps`` use ``REF_CFG_LATE_SCALE``, so a high
    scale shapes the pose without carrying its artifacts into the details.
    ``timesteps`` are the original-schedule indices the respaced diffusion hands
    the model, so ``num_timesteps`` is the training ``--diffusion_steps``.
    """
    # Noise level t / T below which the late scale takes over.
    REF_CFG_SWITCH = 0.5
    # Reference scale on the low-noise detail steps: the plain reference prediction.
    REF_CFG_LATE_SCALE = 1.0

    def __init__(self, model, ref_scale, text_scale, use_text, num_timesteps):
        super().__init__(model, ref_scale)
        if not math.isfinite(float(text_scale)) or float(text_scale) < 0:
            raise ValueError('text_scale must be finite and >= 0')
        if int(num_timesteps) <= 0:
            raise ValueError(f'num_timesteps must be positive, got {num_timesteps}')
        self.text_scale, self.use_text = float(text_scale), bool(use_text)
        self.switch_t = self.REF_CFG_SWITCH * int(num_timesteps)

    def _ref_scale(self, timesteps):
        """Reference scale for this step: a float, or a (B, 1, 1, 1) tensor when
        rows of one batch sit on different sides of the switch."""
        early = timesteps >= self.switch_t
        if bool(early.all()):
            return self.guidance_scale
        if not bool(early.any()):
            return self.REF_CFG_LATE_SCALE
        scale = torch.where(early, self.guidance_scale, self.REF_CFG_LATE_SCALE)
        return scale.to(torch.float32)[:, None, None, None]

    def forward(self, x, timesteps, y=None, **kwargs):
        # Expand the composition so branches with zero weight need no forward:
        # (1 - ref_scale) * base + (ref_scale - text_scale) * ref
        # + text_scale * full. Without a text prompt, text_scale is zero.
        text_scale = self.text_scale if self.use_text else 0.0
        ref_scale = self._ref_scale(timesteps)
        base_weight = 1.0 - ref_scale
        ref_weight = ref_scale - text_scale
        out = None

        def nonzero(weight):
            return bool((weight != 0.0).any()) if torch.is_tensor(weight) else weight != 0.0

        null_y = dict(y or {})
        null_y['ref_active'] = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        null_y['action_label_active'] = torch.zeros_like(null_y['ref_active'])
        if nonzero(base_weight):
            base = self.model(x, timesteps, y=null_y, **kwargs)
            out = base_weight * base
        if nonzero(ref_weight):
            ref_y = dict(y or {})
            ref_y['action_label_active'] = null_y['action_label_active']
            ref = self.model(x, timesteps, y=ref_y, **kwargs)
            term = ref_weight * ref
            out = term if out is None else out + term
        if text_scale != 0.0:
            full = self.model(x, timesteps, y=y, **kwargs)
            term = text_scale * full
            out = term if out is None else out + term
        return out
