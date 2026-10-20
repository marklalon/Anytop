"""Windows launcher compiler selection without changing model/precision settings."""
from pathlib import Path
import sys
import types
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import compile_util


@pytest.fixture
def triton_install(tmp_path, monkeypatch):
    from torch._inductor import config
    monkeypatch.setattr(config, 'cpp_wrapper', True)
    monkeypatch.setattr(config, 'disable_cpp_codegen', False)
    root = tmp_path / 'triton'
    compiler = root / 'runtime' / 'tcc' / 'tcc.exe'
    compiler.parent.mkdir(parents=True)
    compiler.write_bytes(b'compiler fixture')
    fake = types.ModuleType('triton')
    fake.__file__ = str(root / '__init__.py')
    runtime = types.ModuleType('triton.runtime')
    build = types.ModuleType('triton.runtime.build')
    build.get_cc = Mock(return_value=str(tmp_path / 'MSVC' / 'cl.exe'))
    runtime.build = build
    fake.runtime = runtime
    monkeypatch.setitem(sys.modules, 'triton', fake)
    monkeypatch.setitem(sys.modules, 'triton.runtime', runtime)
    monkeypatch.setitem(sys.modules, 'triton.runtime.build', build)
    monkeypatch.setattr(compile_util.sys, 'platform', 'win32')
    monkeypatch.setenv('CC', str(tmp_path / 'MSVC' / 'cl.exe'))
    return compiler, build


def test_msvc_switches_to_bundled_tcc_and_invalidates_cached_selection(triton_install, monkeypatch):
    compiler, build = triton_install
    monkeypatch.setenv('CXX', 'cl.exe')
    monkeypatch.setenv('CUDA_HOME', 'cuda-toolkit')
    assert compile_util.configure_triton_c_compiler() == str(compiler)
    assert compile_util.os.environ['CC'] == str(compiler)
    build.get_cc.cache_clear.assert_called_once_with()
    assert compile_util.os.environ['CXX'] == 'cl.exe'
    assert compile_util.os.environ['CUDA_HOME'] == 'cuda-toolkit'


@pytest.mark.parametrize('name', ['tcc.exe', 'clang.exe', 'clang-cl.exe', 'gcc'])
def test_inherited_compilers_are_overridden(triton_install, monkeypatch, name):
    compiler, build = triton_install
    existing = str(compiler.parent / name)
    build.get_cc.return_value = existing
    monkeypatch.setenv('CC', existing)
    assert compile_util.configure_triton_c_compiler() == str(compiler)
    assert compile_util.os.environ['CC'] == str(compiler)
    build.get_cc.assert_not_called()
    build.get_cc.cache_clear.assert_called_once_with()


def test_missing_bundled_compiler_fails_without_msvc_fallback(triton_install):
    compiler, build = triton_install
    compiler.unlink()
    previous = compile_util.os.environ['CC']
    with pytest.raises(RuntimeError, match='requires bundled TinyCC'):
        compile_util.configure_triton_c_compiler()
    assert compile_util.os.environ['CC'] == previous
    build.get_cc.cache_clear.assert_not_called()


def test_unset_cc_does_not_probe_msvc_and_disables_cpp_codegen(triton_install, monkeypatch):
    from torch._inductor import config
    compiler, build = triton_install
    monkeypatch.delenv('CC')
    assert compile_util.configure_triton_c_compiler() == str(compiler)
    build.get_cc.assert_not_called()
    assert config.cpp_wrapper is False
    assert config.disable_cpp_codegen is True


def test_non_windows_does_not_import_triton(monkeypatch):
    monkeypatch.setattr(compile_util.sys, 'platform', 'linux')
    monkeypatch.setitem(sys.modules, 'triton', None)
    assert compile_util.configure_triton_c_compiler() is None


def test_training_selects_compiler_before_lazy_compile(monkeypatch, tmp_path):
    import torch
    from train.training_loop import TrainLoop
    events = []
    def configure():
        events.append('compiler')
        return str(tmp_path / 'tcc.exe')
    def compile_model(model, **kwargs):
        events.append('compile')
        assert kwargs == {'dynamic': False}
        return model
    monkeypatch.setattr(compile_util, 'configure_triton_c_compiler', configure)
    monkeypatch.setattr(torch, 'compile', compile_model)
    monkeypatch.setattr(torch, 'set_float32_matmul_precision', lambda *_: None)
    model = object()
    loop = types.SimpleNamespace(forward_model=model)
    TrainLoop._compile_forward_model(loop)
    assert events == ['compiler', 'compile']
    assert loop.forward_model is model


def test_training_does_not_swallow_compiler_setup_failure(monkeypatch):
    import torch
    from train.training_loop import TrainLoop
    def configure():
        raise RuntimeError('requires bundled TinyCC')
    compile_model = Mock()
    monkeypatch.setattr(compile_util, 'configure_triton_c_compiler', configure)
    monkeypatch.setattr(torch, 'compile', compile_model)
    loop = types.SimpleNamespace(forward_model=object())
    with pytest.raises(RuntimeError, match='requires bundled TinyCC'):
        TrainLoop._compile_forward_model(loop)
    compile_model.assert_not_called()
