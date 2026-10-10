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
