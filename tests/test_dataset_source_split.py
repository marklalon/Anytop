"""Dataset roles travel from the manifest through the merged cond."""

import json
import sys
from pathlib import Path

import pytest

ANYTOP_ROOT = Path(__file__).resolve().parents[1]
if str(ANYTOP_ROOT) not in sys.path:
    sys.path.insert(0, str(ANYTOP_ROOT))

from data_loaders.truebones.truebones_utils.dataset_sources import (  # noqa: E402
    load_datasets_manifest,
    sources_from_cond,
)


def test_manifest_split_is_explicit_and_namespace_independent(tmp_path):
    manifest = tmp_path / "datasets.jsonl"
    rows = [
        {"namespace": "first", "path": str(tmp_path / "first"), "split": "train"},
        {"namespace": "held_out", "path": str(tmp_path / "held_out"), "split": "val"},
        {"namespace": "later", "path": str(tmp_path / "later"), "split": "val", "enabled": False},
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    sources = load_datasets_manifest(manifest)
    assert [(source.namespace, source.split) for source in sources] == [
        ("first", "train"), ("held_out", "val")
    ]

    # The merge writes dataset_split on each entry; training rebuilds sources
    # from that snapshot without reading the manifest again.
    cond = {
        f"{source.namespace}/Species": {
            "dataset_namespace": source.namespace,
            "dataset_root": source.root,
            "dataset_split": source.split,
        }
        for source in sources
    }
    restored = sources_from_cond(cond)
    assert [(source.namespace, source.split) for source in restored] == [
        ("first", "train"), ("held_out", "val")
    ]


@pytest.mark.parametrize("split", [None, "test", "training"])
def test_manifest_rejects_missing_or_unknown_split(tmp_path, split):
    manifest = tmp_path / "datasets.jsonl"
    row = {"namespace": "sample", "path": str(tmp_path / "sample")}
    if split is not None:
        row["split"] = split
    manifest.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="split"):
        load_datasets_manifest(manifest)


def test_manifest_species_filter_globs(tmp_path):
    manifest = tmp_path / "datasets.jsonl"
    row = {
        "namespace": "sample", "path": str(tmp_path / "sample"), "split": "train",
        "species_include": ["Horse", "*Dragon*"], "species_exclude": "Dragon_Baby*",
    }
    manifest.write_text(json.dumps(row), encoding="utf-8")
    (source,) = load_datasets_manifest(manifest)
    names = ["Horse", "horse", "RedDragon", "Dragon_Baby_01", "Cat"]
    assert [name for name in names if source.accepts(name)] == ["Horse", "RedDragon"]
    assert source.unmatched_includes(["Horse", "Cat"]) == ["*Dragon*"]



@pytest.mark.parametrize("include, expected", [
    ("Winged, Fluttering; Quadruped", ["Bee", "Horse"]),
    ("Fluttering; Quadruped", ["Bee", "Moth", "Horse"]),
    (["Winged, Fluttering", "Quadruped"], ["Bee", "Horse"]),
    ("Winged", ["Eagle", "Bee"]),
])
def test_manifest_species_tags_filter(tmp_path, include, expected):
    manifest = tmp_path / "datasets.jsonl"
    row = {"namespace": "sample", "path": str(tmp_path / "sample"), "split": "train",
           "species_tags_include": include}
    manifest.write_text(json.dumps(row), encoding="utf-8")
    (source,) = load_datasets_manifest(manifest)
    species = {
        "Eagle": ("Winged",), "Bee": ("Winged", "Fluttering"), "Moth": ("Multiped", "Fluttering"),
        "Horse": ("Quadruped",), "Snake": ("Serpentine",),
    }
    assert [name for name, tags in species.items() if source.accepts(name, tags)] == expected
    with pytest.raises(ValueError, match="tags"):
        source.accepts("Horse")


def test_manifest_species_tags_exclude_requires_every_tag(tmp_path):
    manifest = tmp_path / "datasets.jsonl"
    row = {"namespace": "sample", "path": str(tmp_path / "sample"), "split": "train",
           "species_tags_exclude": "Winged, Flutter*"}
    manifest.write_text(json.dumps(row), encoding="utf-8")
    (source,) = load_datasets_manifest(manifest)
    assert source.accepts("Eagle", ("Winged",))
    assert not source.accepts("Bee", ("Winged", "Fluttering"))


def test_manifest_species_tags_filter_rejects_empty_tag(tmp_path):
    manifest = tmp_path / "datasets.jsonl"
    row = {"namespace": "sample", "path": str(tmp_path / "sample"), "split": "train",
           "species_tags_include": "Winged,; Quadruped"}
    manifest.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="empty tag"):
        load_datasets_manifest(manifest)
