"""Compiler setup for Windows Triton launchers used by torch.compile."""
import os
from pathlib import Path
import sys


def configure_triton_c_compiler():
    """Require Triton's bundled TinyCC for Windows torch.compile training.

    triton-windows 3.6 emits ``CUlaunchAttribute attr = {};`` in generated C.
    MSVC 14.38 rejects those initializers in /std:c11 mode (C2059), including
    when the cluster-launch branch is never used. TinyCC compiles the same
    launcher. Set CC only in this training process and invalidate Triton's
    cached compiler selection before any lazy Inductor compilation starts.

    Override any inherited CC, even in a Visual Studio developer shell. TinyCC
    only compiles C, so disable Inductor's C++ wrapper and CPU C++ codegen too;
    CPU operations use ATen instead. CUDA kernels still use Triton/PTXAS.
    Fail if TinyCC is missing rather than selecting another compiler.
    Returns the compiler path, or None on non-Windows platforms.
    """
    if sys.platform != 'win32':
        return None
    import triton
    from triton.runtime import build

    bundled = Path(triton.__file__).resolve().parent / 'runtime' / 'tcc' / 'tcc.exe'
    if not bundled.is_file():
        raise RuntimeError(
            f'Windows torch.compile requires bundled TinyCC: {bundled}. '
            'Install triton-windows with runtime/tcc/tcc.exe; MSVC fallback is disabled.'
        )
    os.environ['CC'] = str(bundled)
    build.get_cc.cache_clear()
    from torch._inductor import config
    config.cpp_wrapper = False
    config.disable_cpp_codegen = True
    return str(bundled)
