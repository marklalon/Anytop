"""Compiler setup for Windows Triton launchers used by torch.compile."""
import os
from pathlib import Path
import sys


def configure_triton_c_compiler():
    """Prefer Triton's bundled TinyCC when its launcher compiler is MSVC.

    triton-windows 3.6 emits ``CUlaunchAttribute attr = {};`` in generated C.
    MSVC 14.38 rejects those initializers in /std:c11 mode (C2059), including
    when the cluster-launch branch is never used. TinyCC compiles the same
    launcher. Set CC only in this training process and invalidate Triton's
    cached compiler selection before any lazy Inductor compilation starts.

    PyTorch's C++ compiler (CXX) and CUDA compiler are separate. Non-Windows
    platforms and an already selected non-MSVC C compiler are left as-is.
    Returns the new compiler path, or None when no switch was made.
    """
    if sys.platform != 'win32':
        return None
    import triton
    from triton.runtime import build

    current = build.get_cc()
    if Path(current).name.lower() not in ('cl', 'cl.exe'):
        return None
    bundled = Path(triton.__file__).resolve().parent / 'runtime' / 'tcc' / 'tcc.exe'
    if not bundled.is_file():
        return None
    os.environ['CC'] = str(bundled)
    build.get_cc.cache_clear()
    return str(bundled)
