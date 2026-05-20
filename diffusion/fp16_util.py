"""
Helpers to train with 16-bit precision.
"""

import numpy as np
import torch as th
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from diffusion import logger

INITIAL_LOG_LOSS_SCALE = 20.0
GRAD_NORM_ABORT_THRESHOLD = 1e12


def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()


def make_master_params(param_groups_and_shapes):
    """
    Copy model parameters into a (differently-shaped) list of full-precision
    parameters.
    """
    master_params = []
    for param_group, shape in param_groups_and_shapes:
        master_param = nn.Parameter(
            _flatten_dense_tensors(
                [param.detach().float() for (_, param) in param_group]
            ).view(shape)
        )
        master_param.requires_grad = True
        master_params.append(master_param)
    return master_params


def model_grads_to_master_grads(param_groups_and_shapes, master_params):
    """
    Copy the gradients from the model parameters into the master parameters
    from make_master_params().
    """
    for master_param, (param_group, shape) in zip(
        master_params, param_groups_and_shapes
    ):
        master_param.grad = _flatten_dense_tensors(
            [param_grad_or_zeros(param) for (_, param) in param_group]
        ).view(shape)


def master_params_to_model_params(param_groups_and_shapes, master_params):
    """
    Copy the master parameter data back into the model parameters.
    """
    # Without copying to a list, if a generator is passed, this will
    # silently not copy any parameters.
    for master_param, (param_group, _) in zip(master_params, param_groups_and_shapes):
        for (_, param), unflat_master_param in zip(
            param_group, unflatten_master_params(param_group, master_param.view(-1))
        ):
            param.detach().copy_(unflat_master_param)


def unflatten_master_params(param_group, master_param):
    return _unflatten_dense_tensors(master_param, [param for (_, param) in param_group])


def get_param_groups_and_shapes(named_model_params):
    named_model_params = list(named_model_params)
    scalar_vector_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim <= 1],
        (-1),
    )
    matrix_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim > 1],
        (1, -1),
    )
    return [scalar_vector_named_params, matrix_named_params]


def master_params_to_state_dict(
    model, param_groups_and_shapes, master_params, use_fp16
):
    if use_fp16:
        state_dict = model.state_dict()
        for master_param, (param_group, _) in zip(
            master_params, param_groups_and_shapes
        ):
            for (name, _), unflat_master_param in zip(
                param_group, unflatten_master_params(param_group, master_param.view(-1))
            ):
                assert name in state_dict
                state_dict[name] = unflat_master_param
    else:
        state_dict = model.state_dict()
        for i, (name, _value) in enumerate(model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
    return state_dict


def state_dict_to_master_params(model, state_dict, use_fp16):
    if use_fp16:
        named_model_params = [
            (name, state_dict[name]) for name, _ in model.named_parameters()
        ]
        param_groups_and_shapes = get_param_groups_and_shapes(named_model_params)
        master_params = make_master_params(param_groups_and_shapes)
    else:
        master_params = [state_dict[name] for name, _ in model.named_parameters()]
    return master_params


def zero_master_grads(master_params):
    for param in master_params:
        param.grad = None


def zero_grad(model_params):
    for param in model_params:
        # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()


def param_grad_or_zeros(param):
    if param.grad is not None:
        return param.grad.data.detach()
    else:
        return th.zeros_like(param)


def _count_nonfinite_entries(tensor):
    if not th.is_tensor(tensor) or not th.is_floating_point(tensor):
        return 0, 0
    inf_count = int(th.isinf(tensor).sum().item())
    nan_count = int(th.isnan(tensor).sum().item())
    return inf_count, nan_count


def count_nonfinite_gradients(parameters):
    inf_count = 0
    nan_count = 0
    for param in parameters:
        if param.grad is None:
            continue
        grad_inf_count, grad_nan_count = _count_nonfinite_entries(param.grad)
        inf_count += grad_inf_count
        nan_count += grad_nan_count
    return {
        "inf": inf_count,
        "nan": nan_count,
        "found": (inf_count + nan_count) > 0,
    }


def inspect_optimizer_state(optimizer):
    inf_count = 0
    nan_count = 0
    by_key = {}
    for state in optimizer.state.values():
        for key, value in state.items():
            if not th.is_tensor(value) or not th.is_floating_point(value):
                continue
            inf_in_tensor, nan_in_tensor = _count_nonfinite_entries(value)
            total_nonfinite = inf_in_tensor + nan_in_tensor
            if total_nonfinite == 0:
                continue
            inf_count += inf_in_tensor
            nan_count += nan_in_tensor
            by_key[key] = by_key.get(key, 0) + total_nonfinite
    return {
        "inf": inf_count,
        "nan": nan_count,
        "by_key": by_key,
        "found": (inf_count + nan_count) > 0,
    }


def sanitize_optimizer_state(optimizer):
    stats = inspect_optimizer_state(optimizer)
    for state in optimizer.state.values():
        for key, value in state.items():
            if not th.is_tensor(value) or not th.is_floating_point(value):
                continue
            if not th.isfinite(value).all():
                value.masked_fill_(~th.isfinite(value), 0.0)
    return stats


def inspect_optimizer_slot_max(optimizer, slot_key, *, param_name_lookup=None, top_k=5):
    max_abs = 0.0
    top_entries = []
    found = False
    for param, state in optimizer.state.items():
        value = state.get(slot_key)
        if not th.is_tensor(value) or not th.is_floating_point(value):
            continue
        found = True
        abs_value = th.nan_to_num(
            value.detach().abs().float(),
            nan=float("inf"),
            posinf=float("inf"),
            neginf=float("inf"),
        )
        slot_max = float(abs_value.max().item()) if abs_value.numel() else 0.0
        max_abs = max(max_abs, slot_max)
        param_name = None if param_name_lookup is None else param_name_lookup.get(id(param))
        if not param_name:
            param_name = "<unnamed>"
        top_entries.append({"name": param_name, "max_abs": slot_max})

    top_entries.sort(key=lambda entry: entry["max_abs"], reverse=True)
    return {
        "slot_key": slot_key,
        "found": found,
        "max_abs": max_abs,
        "top": top_entries[: max(int(top_k), 0)],
    }


def format_optimizer_slot_max(stats):
    slot_key = stats.get("slot_key", "slot")
    if not stats.get("found", False):
        return f"{slot_key}_absmax=n/a"
    parts = [f"{slot_key}_absmax={float(stats.get('max_abs', 0.0)):.6e}"]
    top_entries = stats.get("top", []) or []
    if top_entries:
        summary = ", ".join(
            f"{entry['name']}:{float(entry['max_abs']):.3e}" for entry in top_entries[:3]
        )
        parts.append(f"top=[{summary}]")
    return " ".join(parts)


def format_nonfinite_stats(stats):
    parts = []
    inf_count = int(stats.get("inf", 0))
    nan_count = int(stats.get("nan", 0))
    if inf_count:
        parts.append(f"inf={inf_count}")
    if nan_count:
        parts.append(f"nan={nan_count}")
    by_key = stats.get("by_key", {}) or {}
    if by_key:
        key_summary = ", ".join(f"{key}:{count}" for key, count in sorted(by_key.items()))
        parts.append(f"keys=[{key_summary}]")
    return " ".join(parts) if parts else "none"


def should_abort_on_grad_norm(value):
    return np.isfinite(value) and value > GRAD_NORM_ABORT_THRESHOLD


class MixedPrecisionTrainer:
    def __init__(
        self,
        *,
        model,
        use_fp16=False,
        amp_dtype='fp32',
        amp_enabled=False,
        device_type='cuda',
        log_norms=True,
        fp16_scale_growth=1e-3,
        initial_lg_loss_scale=INITIAL_LOG_LOSS_SCALE,
    ):
        self.model = model
        self.use_fp16 = use_fp16
        self.amp_dtype = amp_dtype
        self.amp_enabled = amp_enabled
        self.device_type = device_type
        self.log_norms = bool(log_norms)
        self.fp16_scale_growth = fp16_scale_growth

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.param_groups_and_shapes = None
        self.lg_loss_scale = initial_lg_loss_scale
        scaler_enabled = self.amp_enabled and self.amp_dtype == 'fp16' and self.device_type == 'cuda'
        self.scaler = th.amp.GradScaler('cuda', enabled=scaler_enabled)

        if self.use_fp16:
            self.param_groups_and_shapes = get_param_groups_and_shapes(
                self.model.named_parameters()
            )
            self.master_params = make_master_params(self.param_groups_and_shapes)
            self.model.convert_to_fp16()

    def zero_grad(self):
        zero_grad(self.model_params)

    def backward(self, loss: th.Tensor):
        if self.amp_enabled:
            self.scaler.scale(loss).backward()
        elif self.use_fp16:
            loss_scale = 2 ** self.lg_loss_scale
            (loss * loss_scale).backward()
        else:
            loss.backward()

    def optimize(self, opt: th.optim.Optimizer, scheduler: th.optim.lr_scheduler.StepLR):
        if self.amp_enabled:
            return self._optimize_amp(opt, scheduler)
        if self.use_fp16:
            return self._optimize_fp16(opt, scheduler)
        else:
            return self._optimize_normal(opt, scheduler)

    def _clip_gradients_and_check_nonfinite(self, parameters, *, max_norm):
        try:
            total_norm = th.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=max_norm,
                error_if_nonfinite=True,
            )
            if th.is_tensor(total_norm):
                return float(total_norm.item())
            return float(total_norm)
        except RuntimeError as exc:
            exc_text = str(exc).lower()
            if 'non-finite' not in exc_text and 'nonfinite' not in exc_text:
                raise
            return None

    def _abort_on_large_finite_grad_norm(self, grad_norm, *, mode_label=None):
        if not should_abort_on_grad_norm(grad_norm):
            return
        mode_suffix = f" under {mode_label}" if mode_label else ""
        raise RuntimeError(
            "Detected abnormal finite grad_norm before optimizer step"
            f"{mode_suffix} (grad_norm={grad_norm:.6e}, "
            f"threshold={GRAD_NORM_ABORT_THRESHOLD:.1e})"
        )

    def _optimize_amp(self, opt: th.optim.Optimizer, scheduler: th.optim.lr_scheduler.StepLR):
        if self.scaler.is_enabled():
            self.scaler.unscale_(opt)

        clipped_norm = self._clip_gradients_and_check_nonfinite(
            self.model_params,
            max_norm=1.0,
        )
        if clipped_norm is None:
            grad_stats = count_nonfinite_gradients(self.model_params)
            logger.log(
                "Skipping optimizer step due to non-finite gradients under AMP "
                f"({format_nonfinite_stats(grad_stats)})"
            )
            if self.scaler.is_enabled():
                self.scaler.step(opt)
                self.scaler.update()
            self.zero_grad()
            return False

        self._abort_on_large_finite_grad_norm(clipped_norm, mode_label="AMP")
        if self.log_norms:
            logger.logkv_mean("grad_norm", clipped_norm)

        self.scaler.step(opt)
        self.scaler.update()
        scheduler.step()
        logger.logkv_mean("lr", scheduler.get_last_lr()[0])
        return True

    def _optimize_fp16(self, opt: th.optim.Optimizer, scheduler: th.optim.lr_scheduler.StepLR):
        if self.log_norms:
            logger.logkv_mean("lg_loss_scale", self.lg_loss_scale)
        model_grads_to_master_grads(self.param_groups_and_shapes, self.master_params)
        grad_norm, param_norm = self._compute_norms(grad_scale=2 ** self.lg_loss_scale)
        if check_overflow(grad_norm):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            zero_master_grads(self.master_params)
            return False

        self._abort_on_large_finite_grad_norm(grad_norm, mode_label="FP16")

        if self.log_norms:
            logger.logkv_mean("grad_norm", grad_norm)
            logger.logkv_mean("param_norm", param_norm)

        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        opt.step()
        scheduler.step()
        zero_master_grads(self.master_params)
        master_params_to_model_params(self.param_groups_and_shapes, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth
        return True

    def _optimize_normal(self, opt: th.optim.Optimizer, scheduler: th.optim.lr_scheduler.StepLR):
        clipped_norm = self._clip_gradients_and_check_nonfinite(
            self.model_params,
            max_norm=1.0,
        )
        if clipped_norm is None:
            grad_stats = count_nonfinite_gradients(self.master_params)
            logger.log(
                "Skipping optimizer step due to non-finite gradients "
                f"({format_nonfinite_stats(grad_stats)})"
            )
            zero_master_grads(self.master_params)
            return False

        self._abort_on_large_finite_grad_norm(clipped_norm)
        if self.log_norms:
            logger.logkv_mean("grad_norm", clipped_norm)

        opt.step()
        scheduler.step()
        logger.logkv_mean("lr", scheduler.get_last_lr()[0])
        return True

    def _compute_norms(self, grad_scale=1.0):
        grad_norm = 0.0
        param_norm = 0.0
        for p in self.master_params:
            with th.no_grad():
                param_norm += th.norm(p, p=2, dtype=th.float32).item() ** 2
                if p.grad is not None:
                    grad_norm += th.norm(p.grad, p=2, dtype=th.float32).item() ** 2
        return np.sqrt(grad_norm) / grad_scale, np.sqrt(param_norm)

    def master_params_to_state_dict(self, master_params):
        return master_params_to_state_dict(
            self.model, self.param_groups_and_shapes, master_params, self.use_fp16
        )

    def state_dict_to_master_params(self, state_dict):
        return state_dict_to_master_params(self.model, state_dict, self.use_fp16)


def check_overflow(value):
    return (value == float("inf")) or (value == -float("inf")) or (value != value)
