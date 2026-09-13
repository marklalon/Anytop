"""Bit-reproducible fp32 mode for "did this change keep the numbers" checks.

Under bf16 autocast every matmul rounds to 7 mantissa bits, and any bit-level
perturbation upstream reshuffles that rounding. Changing the association order
of the FK loss's chained products touched 13 of 1.15M output-gradient elements
by 1.5e-8, yet moved parameter gradients by a median 2e-3 relative -- the same
size as two non-deterministic runs of unchanged code. A bf16 comparison can
neither confirm nor rule out a regression (docs/bf16_precision_issues.md 4.4).

Protocol for every before/after equivalence check (docs 5.2 D):

1. ``--amp_dtype fp32`` (no autocast) with TF32 off;
2. ``torch.use_deterministic_algorithms(True)`` with
   ``CUBLAS_WORKSPACE_CONFIG=:4096:8``;
3. SDPA restricted to the math backend;
4. dropout at 0 -- its mask changes with the SDPA backend and tensor dtype --
   or verified-identical masks on both sides;
5. torch / cuda / numpy / random reseeded before each run;
6. run the unchanged code twice and confirm bit-identical results before
   comparing old against new.

Steps 1-3 are ``enable_numerical_verification_mode``, 4 is ``disable_dropout``,
5 is ``reseed``; step 1's ``--amp_dtype`` and step 6 are the caller's. bf16 is
only for the final speed measurement. Under this mode the FK change above
measured a median 1.5e-7 (worst 8e-7) -- plain float32 rounding.
"""
import os
import random

import numpy as np
import torch
import torch.nn as nn

_CUBLAS_WORKSPACE_ENV = 'CUBLAS_WORKSPACE_CONFIG'
_DETERMINISTIC_CUBLAS_CONFIGS = (':4096:8', ':16:8')


def enable_numerical_verification_mode():
    """TF32 off, deterministic algorithms, math-only SDPA (protocol steps 1-3).

    Process-global and not undone. Call it before any CUDA work: cuBLAS sizes its
    workspace from ``CUBLAS_WORKSPACE_CONFIG`` when it first initializes, so the
    variable is only honoured if set before that.
    """
    if os.environ.get(_CUBLAS_WORKSPACE_ENV) not in _DETERMINISTIC_CUBLAS_CONFIGS:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            raise RuntimeError(
                f'enable_numerical_verification_mode() must run before CUDA is initialized, '
                f'or launch with {_CUBLAS_WORKSPACE_ENV}={_DETERMINISTIC_CUBLAS_CONFIGS[0]} set.'
            )
        os.environ[_CUBLAS_WORKSPACE_ENV] = _DETERMINISTIC_CUBLAS_CONFIGS[0]

    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def disable_dropout(model):
    """Zero every dropout rate in ``model`` in place (protocol step 4).

    Covers ``nn.Dropout`` modules (including the elementwise
    ``joints_names_dropout``) and the attention modules that keep a float
    ``dropout`` and pass it to SDPA / ``F.dropout`` directly
    (``SelectiveMultiheadAttention``). Conditioning drops -- CFG drop,
    ``joint_name_drop_prob`` whole-joint substitution, joint / temporal masks --
    draw fp32 random numbers independent of the dtype and are left alone.
    Returns the number of rates changed.
    """
    changed = 0
    for module in model.modules():
        if isinstance(module, nn.modules.dropout._DropoutNd):
            if module.p != 0.0:
                module.p = 0.0
                changed += 1
        elif isinstance(getattr(module, 'dropout', None), float) and module.dropout != 0.0:
            module.dropout = 0.0
            changed += 1
    return changed


def reseed(seed):
    """Reseed torch, every CUDA device, numpy and random (protocol step 5)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
