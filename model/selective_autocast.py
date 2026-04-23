from __future__ import annotations

import types
from functools import wraps

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.motion_transformer import GraphMultiHeadAttention, SelectiveMultiheadAttention


_ORIGINAL_TORCH_SOFTMAX = torch.softmax
_ORIGINAL_F_SOFTMAX = F.softmax

_FUNCTION_PATCHED = False


def _cast_floating_tensors_to_fp32(value):
    if torch.is_tensor(value):
        if torch.is_floating_point(value) and value.dtype != torch.float32:
            return value.float()
        return value
    if isinstance(value, tuple):
        return tuple(_cast_floating_tensors_to_fp32(item) for item in value)
    if isinstance(value, list):
        return [_cast_floating_tensors_to_fp32(item) for item in value]
    if isinstance(value, dict):
        return {key: _cast_floating_tensors_to_fp32(item) for key, item in value.items()}
    return value


def _run_fp32_op(function, *args, **kwargs):
    device_type = None
    converted_args = []
    for arg in args:
        if torch.is_tensor(arg):
            if device_type is None:
                device_type = arg.device.type
            if torch.is_floating_point(arg) and arg.dtype != torch.float32:
                arg = arg.float()
        converted_args.append(arg)

    converted_kwargs = {}
    for key, value in kwargs.items():
        if torch.is_tensor(value):
            if device_type is None:
                device_type = value.device.type
            if torch.is_floating_point(value) and value.dtype != torch.float32:
                value = value.float()
        converted_kwargs[key] = value

    if device_type is None:
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.autocast(device_type=device_type, enabled=False):
        return function(*converted_args, **converted_kwargs)


def _patch_functional_ops(*, device_type: str, autocast_dtype: torch.dtype | None):
    global _FUNCTION_PATCHED
    if _FUNCTION_PATCHED or autocast_dtype is None:
        return

    def _softmax_wrapper(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            return _run_fp32_op(function, *args, **kwargs)

        return wrapped

    torch.softmax = _softmax_wrapper(_ORIGINAL_TORCH_SOFTMAX)
    F.softmax = _softmax_wrapper(_ORIGINAL_F_SOFTMAX)
    _FUNCTION_PATCHED = True


def _wrap_module_forward(module: nn.Module, *, device_type: str, autocast_dtype: torch.dtype) -> bool:
    if getattr(module, "_selective_autocast_wrapped", False):
        return False

    original_forward = module.forward

    @wraps(original_forward)
    def wrapped_forward(self, *args, **kwargs):
        with torch.autocast(device_type=device_type, dtype=autocast_dtype):
            output = original_forward(*args, **kwargs)
        return _cast_floating_tensors_to_fp32(output)

    module.forward = types.MethodType(wrapped_forward, module)
    module._selective_autocast_wrapped = True
    return True


def enable_selective_autocast(
    model: nn.Module,
    *,
    device_type: str,
    autocast_dtype: torch.dtype | None,
) -> int:
    if autocast_dtype is None:
        return 0

    _patch_functional_ops(device_type=device_type, autocast_dtype=autocast_dtype)

    patched = 0
    target_types = (
        nn.Linear,
        GraphMultiHeadAttention,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
    )
    for module in model.modules():
        if isinstance(module, SelectiveMultiheadAttention):
            patched += int(
                module.configure_precision(
                    device_type=device_type,
                    autocast_dtype=autocast_dtype,
                )
            )
            continue
        if isinstance(module, target_types):
            patched += int(
                _wrap_module_forward(
                    module,
                    device_type=device_type,
                    autocast_dtype=autocast_dtype,
                )
            )
    return patched