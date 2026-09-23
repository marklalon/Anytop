"""``get_opt(inference=True)`` never reads *or stats* a dataset directory.

A checkpoint directory is meant to be self-contained: ``model####.pt`` carries
the weights and the frozen action word table, ``args.json`` the architecture,
``cond.npy`` the species, skeletons, baked T5 embeddings and baked species tags.
Two paths used to reach past that snapshot:

1. ``get_opt`` probed each cond entry's ``dataset_root`` for
   ``species_tags.jsonl`` and preferred it when found, so a dataset re-tagged
   after training would quietly describe the checkpoint's species with tags it
   never saw.
2. ``sources_from_cond`` resolved a stored ``dataset_root`` with
   ``resolve_anytop_path``, which tries the relative path under the cwd and
   takes it *if it exists*. Since a cond stores the portable relative form
   (``dataset/truebones/zoo/truebones_processed``), deriving the sources stat-ed
   the dataset directory -- and worse, the resolved root came out different
   depending on where generation was launched from.

These tests pin both, plus the training contract they must not disturb.

The watcher patches ``os``-level entry points rather than installing an
``sys.addaudithook``: **there is no ``os.stat`` audit event in CPython**, so an
audit-based watcher sees ``open()`` but is blind to exactly the probing (2) is
about. ``test_the_sidecar_path_trips_the_watcher`` and
``test_a_bare_exists_probe_trips_the_watcher`` are the guards that keep the
"nothing was touched" assertions from passing for the wrong reason.
"""
import os
from pathlib import Path
import sys

import pytest


ANYTOP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ANYTOP_ROOT.parent
for path in (REPO_ROOT, ANYTOP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from data_loaders.truebones.truebones_utils import dataset_tags as dt  # noqa: E402
from data_loaders.truebones.truebones_utils.get_opt import get_opt  # noqa: E402


# What the cond snapshot says (baked at training time) vs what the dataset
# directory says today. They differ on purpose: whichever one a run picked up is
# then unambiguous from the tags alone.
BAKED_TAGS = ("Quadruped", "Large", "Galloping")
SIDECAR_TAGS = ("Biped", "Small", "Hopping")

# A path segment that appears in no real directory, so the watcher can match a
# relative dataset_root however it ends up being resolved (under the cwd, under
# the Anytop root, or not at all).
MARKER = "__inference_isolation_probe__"
RELATIVE_ROOT = f"dataset/{MARKER}/processed"


@pytest.fixture(autouse=True)
def restore_default_configuration():
    """Each test reconfigures the process-global tag snapshot; put it back."""
    yield
    dt.configure()


def _write_sidecar(dataset_dir):
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / dt.SPECIES_TAGS_FILE).write_text(
        '{"species": "Horse", "species_tags": ["Biped", "Small", "Hopping"]}\n',
        encoding="utf-8",
    )
    return dataset_dir


def _make_cond(dataset_root):
    return {
        "truebones/zoo/Horse": {
            "dataset_namespace": "truebones/zoo",
            "dataset_root": str(dataset_root),
            "species_name": "Horse",
            "species_tags": list(BAKED_TAGS),
        }
    }


class _FsWatcher:
    """Record every filesystem call whose path matches one of *needles*.

    Patches the ``os``-level functions pathlib actually routes through --
    verified against CPython 3.11: ``Path.exists``/``Path.is_file`` -> ``os.stat``
    and ``Path.resolve`` -> ``os.path.realpath`` + ``os.stat``.
    """

    _TARGETS = (
        (os, "stat"),
        (os, "lstat"),
        (os, "listdir"),
        (os, "scandir"),
        (os.path, "realpath"),
        (os.path, "exists"),
        (os.path, "isfile"),
        (os.path, "isdir"),
    )

    def __init__(self, monkeypatch, needles):
        self.needles = tuple(os.path.normcase(str(n)) for n in needles)
        self.hits = []
        for module, name in self._TARGETS:
            self._patch(monkeypatch, module, name)
        real_open = open
        self._patch_open(monkeypatch, real_open)

    def _record(self, label, path):
        try:
            text = os.path.normcase(str(path))
        except Exception:
            return
        if any(needle in text for needle in self.needles):
            self.hits.append((label, str(path)))

    def _patch(self, monkeypatch, module, name):
        real = getattr(module, name)
        watcher = self

        def traced(path, *args, **kwargs):
            watcher._record(name, path)
            return real(path, *args, **kwargs)

        monkeypatch.setattr(module, name, traced)

    def _patch_open(self, monkeypatch, real_open):
        import builtins

        watcher = self

        def traced_open(file, *args, **kwargs):
            watcher._record("open", file)
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", traced_open)


@pytest.fixture
def watch(monkeypatch):
    def _watch(*needles):
        return _FsWatcher(monkeypatch, needles)

    return _watch


# -- the inference contract --------------------------------------------------
def test_inference_touches_no_dataset_file(tmp_path, watch):
    dataset_dir = _write_sidecar(tmp_path / "processed")
    cond_dict = _make_cond(dataset_dir)

    watcher = watch(dataset_dir)
    opt = get_opt(None, tmp_path / "cond.npy", cond_dict=cond_dict, inference=True)
    # dataset_tags() is lazy, so the snapshot has to be forced while the watcher
    # is live -- a deferred sidecar read would otherwise land after it.
    tags = dt.dataset_tags().tags_for("truebones/zoo/Horse")

    assert watcher.hits == [], f"inference touched the dataset directory: {watcher.hits!r}"
    assert tags == BAKED_TAGS
    assert opt.inference is True
    # The subsets are derived from the baked tags too, not the sidecar's.
    assert opt.subsets_dict["quadruped"] == ["truebones/zoo/Horse"]
    assert opt.subsets_dict["biped"] == []
    # The root the cond names is still reported, it is just never read.
    assert [source.root for source in opt.sources] == [str(dataset_dir)]


def test_inference_does_not_probe_a_relative_dataset_root(tmp_path, watch):
    """The regression: a relative root is what a real cond actually stores.

    ``resolve_anytop_path`` resolves one by asking whether it exists under the
    cwd, so this path stat-ed the dataset directory even though no file was
    opened -- invisible to an ``open``-only watcher.
    """
    cond_dict = _make_cond(RELATIVE_ROOT)

    watcher = watch(MARKER)
    opt = get_opt(None, tmp_path / "cond.npy", cond_dict=cond_dict, inference=True)
    dt.dataset_tags().tags_for("truebones/zoo/Horse")

    assert watcher.hits == [], f"inference probed a relative dataset_root: {watcher.hits!r}"
    assert opt.sources[0].root == str(ANYTOP_ROOT / Path(RELATIVE_ROOT))


def test_relative_dataset_root_ignores_a_shadowing_cwd(tmp_path, monkeypatch):
    """Lexical resolution makes opt.sources a function of the cond alone.

    Mechanism-independent companion to the watcher test: the probing resolver
    prefers a relative path that exists under the cwd, so launching generation
    from a directory that happens to hold its own ``dataset/...`` tree used to
    silently repoint the sources. Nothing is patched here -- only the answer is
    inspected.
    """
    shadow = _write_sidecar(tmp_path / "cwd" / Path(RELATIVE_ROOT))
    monkeypatch.chdir(tmp_path / "cwd")
    cond_dict = _make_cond(RELATIVE_ROOT)

    inference_opt = get_opt(
        None, tmp_path / "cond.npy", cond_dict=cond_dict, inference=True
    )
    assert inference_opt.sources[0].root == str(ANYTOP_ROOT / Path(RELATIVE_ROOT))
    assert dt.dataset_tags().tags_for("truebones/zoo/Horse") == BAKED_TAGS

    # Training keeps the cwd-first behaviour, shadow sidecar and all.
    training_opt = get_opt(
        None, tmp_path / "cond.npy", cond_dict=cond_dict, inference=False
    )
    assert training_opt.sources[0].root == str(shadow)
    assert dt.dataset_tags().tags_for("truebones/zoo/Horse") == SIDECAR_TAGS


def test_inference_works_when_the_dataset_directory_is_gone(tmp_path):
    """The whole point: a checkpoint generates on a machine without the data."""
    cond_dict = _make_cond(tmp_path / "no_such_dataset" / "processed")

    opt = get_opt(None, tmp_path / "cond.npy", cond_dict=cond_dict, inference=True)

    assert dt.dataset_tags().tags_for("truebones/zoo/Horse") == BAKED_TAGS
    assert opt.subsets_dict["quadruped"] == ["truebones/zoo/Horse"]


# -- the training contract, unchanged ---------------------------------------
def test_training_still_prefers_the_dataset_sidecar(tmp_path):
    dataset_dir = _write_sidecar(tmp_path / "processed")
    cond_dict = _make_cond(dataset_dir)

    opt = get_opt(None, tmp_path / "cond.npy", cond_dict=cond_dict, inference=False)

    assert dt.dataset_tags().tags_for("truebones/zoo/Horse") == SIDECAR_TAGS
    assert opt.inference is False
    assert opt.subsets_dict["biped"] == ["truebones/zoo/Horse"]


def test_training_falls_back_to_baked_tags_without_a_sidecar(tmp_path):
    cond_dict = _make_cond(tmp_path / "no_such_dataset" / "processed")

    get_opt(None, tmp_path / "cond.npy", cond_dict=cond_dict, inference=False)

    assert dt.dataset_tags().tags_for("truebones/zoo/Horse") == BAKED_TAGS


# -- negative controls -------------------------------------------------------
def test_the_sidecar_path_trips_the_watcher(tmp_path, watch):
    """A watcher that matched nothing would satisfy every assertion above."""
    dataset_dir = _write_sidecar(tmp_path / "processed")
    cond_dict = _make_cond(dataset_dir)

    watcher = watch(dataset_dir)
    get_opt(None, tmp_path / "cond.npy", cond_dict=cond_dict, inference=False)
    dt.dataset_tags().tags_for("truebones/zoo/Horse")

    assert any(
        str(path).endswith(dt.SPECIES_TAGS_FILE) for _label, path in watcher.hits
    ), f"the watcher missed the sidecar read it is meant to catch: {watcher.hits!r}"


def test_a_bare_exists_probe_trips_the_watcher(tmp_path, watch):
    """And specifically: a stat with no open must register.

    This is the coverage an ``sys.addaudithook`` watcher cannot have, since
    CPython raises no audit event for ``os.stat``.
    """
    dataset_dir = tmp_path / "processed"
    watcher = watch(dataset_dir)

    (dataset_dir / dt.SPECIES_TAGS_FILE).exists()

    assert watcher.hits, "a stat-only probe went unnoticed"
