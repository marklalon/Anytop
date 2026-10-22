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
    other channel (species FiLM, the canonical output frame, the loop flag, the
    skeleton graph itself) is bit-identical between the two and cancels out of
    the guidance term instead of being extrapolated along with the prompt.

    The two passes run SEQUENTIALLY rather than as one 2B batch. ``y`` carries
    per-sample python lists (parents, joint names, metadata) next to its tensors,
    so a batched pass would have to hand-duplicate every entry, and the padded
    joint/temporal masks would double in memory for no change in the amount of
    attention actually computed -- both passes here run the full model.
    """

    # Noise level, as t / T of the unshifted schedule, below which the late
    # scale takes over.
    CFG_SWITCH = 0.5
    # Scale on the low-noise detail steps: the plain conditional prediction.
    CFG_LATE_SCALE = 1.0

    def __init__(self, model, guidance_scale, num_timesteps, switch_t=None):
        """``switch_t`` is the step at the ``CFG_SWITCH`` noise level of a shifted
        schedule (``sample.conditioning.cfg_switch_timestep``); None means the
        unshifted ``CFG_SWITCH * num_timesteps``."""
        super().__init__()
        self.model = model
        if int(num_timesteps) <= 0:
            raise ValueError(f'num_timesteps must be positive, got {num_timesteps}')
        self.switch_t = (
            self.CFG_SWITCH * int(num_timesteps) if switch_t is None else float(switch_t)
        )
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

    def _staged_scale(self, scale, timesteps):
        """``scale`` on the high-noise half of the steps, ``CFG_LATE_SCALE`` below
        the switch: a float, or a (B, 1, 1, 1) tensor when rows of one batch sit
        on different sides of it."""
        early = timesteps >= self.switch_t
        if bool(early.all()):
            return scale
        if not bool(early.any()):
            return self.CFG_LATE_SCALE
        guided = torch.where(early, scale, self.CFG_LATE_SCALE)
        return guided.to(torch.float32)[:, None, None, None]

    @staticmethod
    def _is_identity(scale):
        return bool((scale == 1.0).all()) if torch.is_tensor(scale) else scale == 1.0

    def forward(self, x, timesteps, y=None, **kwargs):
        cond_out = self.model(x, timesteps, y=y, **kwargs)
        scale = self._staged_scale(self.guidance_scale, timesteps)
        if self._is_identity(scale):
            return cond_out
        uncond_y = dict(y or {})
        # Force the null condition for every row. _resolve_action_label_active
        # gives an explicit mask precedence over the training Bernoulli, and
        # _build_action_label_token routes a False row to action_label_null_emb.
        uncond_y['action_label_active'] = torch.zeros(
            x.shape[0], dtype=torch.bool, device=x.device
        )
        uncond_out = self.model(x, timesteps, y=uncond_y, **kwargs)
        return uncond_out + scale * (cond_out - uncond_out)


class ClassifierFreeSpeciesModel(ClassifierFreeActionModel):
    """Guide per-joint species FiLM using its trained identity bypass.

    Other conditions, including the joint names, remain active in both passes.
    """

    def forward(self, x, timesteps, y=None, **kwargs):
        cond_out = self.model(x, timesteps, y=y, **kwargs)
        scale = self._staged_scale(self.guidance_scale, timesteps)
        if self._is_identity(scale):
            return cond_out
        uncond_y = dict(y or {})
        uncond_y['species_active'] = torch.zeros(
            x.shape[0], dtype=torch.bool, device=x.device
        )
        uncond_out = self.model(x, timesteps, y=uncond_y, **kwargs)
        return uncond_out + scale * (cond_out - uncond_out)


class ClassifierFreeReferenceModel(ClassifierFreeActionModel):
    """Guide the encoded reference against the fully unconditional mode.

    The model bypasses the action label and the species FiLM on every
    reference-active row, so the reference pass carries neither. The
    null-reference pass forces the label off as well and keeps whatever
    ``y['species_active']`` says; encode sets it False, which makes species
    identical across both passes.

    The scale applies only to the high-noise steps, where global posture
    (facing, body height) is settled; the detail steps below
    ``CFG_SWITCH * num_timesteps`` use ``CFG_LATE_SCALE``, so a high scale shapes
    the pose without carrying its artifacts into the details (under a shifted
    noise schedule, the step at that noise level instead).
    ``timesteps`` are the original-schedule indices the respaced diffusion hands
    the model, so ``num_timesteps`` is the training ``--diffusion_steps``.
    """

    def forward(self, x, timesteps, y=None, **kwargs):
        # (1 - scale) * base + scale * ref, so a branch with zero weight needs
        # no forward.
        ref_weight = self._staged_scale(self.guidance_scale, timesteps)
        base_weight = 1.0 - ref_weight
        out = None

        def nonzero(weight):
            return bool((weight != 0.0).any()) if torch.is_tensor(weight) else weight != 0.0

        null_y = dict(y or {})
        null_y['ref_active'] = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        null_y['action_label_active'] = torch.zeros_like(null_y['ref_active'])
        if nonzero(base_weight):
            out = base_weight * self.model(x, timesteps, y=null_y, **kwargs)
        if nonzero(ref_weight):
            term = ref_weight * self.model(x, timesteps, y=y, **kwargs)
            out = term if out is None else out + term
        return out
