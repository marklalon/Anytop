"""Dataset sidecar paths are stored relative to the ``Anytop`` module root.

Relative asset paths inside AnyTop are anchored at the module root (the same
anchor ``param_utils._ANYTOP_ROOT`` uses), so the module can be relocated or
vendored without rewriting its sidecars.  Legacy sidecars written
repo-root-relative (``Anytop/...``) must keep resolving.
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import (
    anytop_root_dir,
    load_tpose_reference_sidecar,
    save_tpose_reference_sidecar,
    repo_root_dir,
    resolve_dataset_path,
    to_portable_dataset_path,
)


def test_anytop_root_is_module_dir_not_repo_root():
    assert os.path.basename(anytop_root_dir()) == "Anytop"
    assert repo_root_dir() == os.path.dirname(anytop_root_dir())


def test_portable_path_is_anytop_relative(tmp_path):
    asset = os.path.join(anytop_root_dir(), "dataset", "truebones", "zoo", "x.glb")
    portable = to_portable_dataset_path(asset)
    assert portable == "dataset/truebones/zoo/x.glb"
    assert not portable.startswith("Anytop/")


def test_portable_path_keeps_paths_outside_anytop_absolute():
    outside = os.path.join(repo_root_dir(), "outputs", "x.glb")
    assert to_portable_dataset_path(outside) == os.path.abspath(outside)
    assert to_portable_dataset_path(None) is None
    assert to_portable_dataset_path("") is None


def test_roundtrip_resolves_back_to_the_same_file(tmp_path):
    rel_dir = os.path.join(anytop_root_dir(), "outputs", "_path_portability_test")
    os.makedirs(rel_dir, exist_ok=True)
    asset = os.path.join(rel_dir, "asset.glb")
    try:
        with open(asset, "wb") as f:
            f.write(b"x")
        portable = to_portable_dataset_path(asset)
        assert portable == "outputs/_path_portability_test/asset.glb"
        assert os.path.samefile(resolve_dataset_path(portable), asset)
        # legacy repo-root-relative form (``Anytop/`` prefix) still resolves
        assert os.path.samefile(resolve_dataset_path("Anytop/" + portable), asset)
        # absolute stored paths pass through untouched
        assert resolve_dataset_path(asset) == asset
    finally:
        if os.path.isfile(asset):
            os.remove(asset)
        os.rmdir(rel_dir)


def test_missing_path_raises_and_empty_is_none():
    assert resolve_dataset_path(None) is None
    assert resolve_dataset_path("") is None
    try:
        resolve_dataset_path("dataset/does/not/exist.glb")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")

def test_sidecar_stores_assets_in_its_own_dir_relative(tmp_path):
    """A cache dir (server skeleton cache) stays self-contained and movable."""
    cache_dir = tmp_path / "cache" / "abc123"
    cache_dir.mkdir(parents=True)
    mesh = cache_dir / "dragon_tpose.glb"
    mesh.write_bytes(b"x")
    sidecar = cache_dir / "tpose_reference_paths.jsonl"

    save_tpose_reference_sidecar(str(sidecar), {"dragon": str(mesh)})
    stored = json.loads(sidecar.read_text(encoding="utf-8").splitlines()[0])
    assert stored["path"] == "dragon_tpose.glb"

    refs = load_tpose_reference_sidecar(str(sidecar))
    assert os.path.samefile(refs["dragon"], mesh)


def test_sidecar_relative_entry_survives_moving_the_dir(tmp_path):
    src = tmp_path / "cache_a"
    src.mkdir()
    (src / "tpose.glb").write_bytes(b"x")
    sidecar = src / "tpose_reference_paths.jsonl"
    save_tpose_reference_sidecar(str(sidecar), {"dragon": str(src / "tpose.glb")})

    dst = tmp_path / "cache_b"
    shutil.move(str(src), str(dst))
    refs = load_tpose_reference_sidecar(str(dst / "tpose_reference_paths.jsonl"))
    assert os.path.samefile(refs["dragon"], dst / "tpose.glb")


def test_sidecar_roundtrip_is_idempotent_for_anytop_assets(tmp_path):
    """Dataset entries stay AnyTop-relative through load -> save."""
    sidecar = tmp_path / "tpose_reference_paths.jsonl"
    rel = "dataset/truebones/zoo/Truebone_Z-OO/Ant/Ant-TPOSE.glb"
    save_tpose_reference_sidecar(str(sidecar), {"Ant": os.path.join(anytop_root_dir(), *rel.split("/"))})
    assert json.loads(sidecar.read_text(encoding="utf-8").splitlines()[0])["path"] == rel

    save_tpose_reference_sidecar(str(sidecar), load_tpose_reference_sidecar(str(sidecar)))
    assert json.loads(sidecar.read_text(encoding="utf-8").splitlines()[0])["path"] == rel


def test_sidecar_keeps_null_entries(tmp_path):
    sidecar = tmp_path / "tpose_reference_paths.jsonl"
    save_tpose_reference_sidecar(str(sidecar), {"Ghost": None})
    assert json.loads(sidecar.read_text(encoding="utf-8").splitlines()[0])["path"] is None
    assert load_tpose_reference_sidecar(str(sidecar)) == {}
