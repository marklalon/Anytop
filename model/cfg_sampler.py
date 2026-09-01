"""Classifier-free guidance wrappers used at sampling time.

Training holds the other half of this contract: ``--action_label_cfg_drop_prob``
hard-drops the action condition on a fraction of the samples (the T5 text and the
multi-hot together, under one mask -- see ``AnyTop._resolve_action_label_active``),
substituting the learned ``action_label_null_emb``. That is what gives the model
a real unconditional mode. Sampling then runs the denoiser twice per diffusion
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
    other channel (species FiLM, the canonical output frame, loop/playspeed, the
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
