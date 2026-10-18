"""Process-local compatibility fixes for the optional compilation toolchain."""
import functools
import importlib
import os


def fix_windows_triton_launcher():
    """Use C11 zero initializers in Triton's CUDA launcher on Windows.

    Triton 3.6 emits two empty aggregate initializers, which MSVC 14.38
    rejects under /std:c11 (C2059). Rewrite only those declarations before
    Triton hashes/compiles the source; installed packages stay untouched.
    """
    if os.name != 'nt':
        return False
    try:
        driver = importlib.import_module('triton.backends.nvidia.driver')
    except ImportError:
        return False
    original = getattr(driver, 'make_launcher', None)
    if original is None or getattr(original, '_anytop_c11_initializers', False):
        return False

    @functools.wraps(original)
    def make_launcher(*args, **kwargs):
        source = original(*args, **kwargs)
        for name in ('clusterAttr', 'clusterSchedulingAttr'):
            source = source.replace(
                f'CUlaunchAttribute {name} = {{}};',
                f'CUlaunchAttribute {name} = {{0}};',
            )
        return source

    make_launcher._anytop_c11_initializers = True
    driver.make_launcher = make_launcher
    return True
