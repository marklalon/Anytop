from __future__ import annotations

import json
from types import SimpleNamespace

from data_loaders.truebones.truebones_utils.motion_labels import load_action_labels
from tools import offline_retarget_augment


def test_write_labels_treats_legacy_npy_key_as_existing(monkeypatch, tmp_path):
    labels_path = tmp_path / "action_labels.jsonl"
    original = {
        "clip": "Foo.npy",
        "action_group": "locomotion",
        "action_label": "walk",
        "is_loop": True,
    }
    labels_path.write_text(json.dumps(original) + "\n", encoding="utf-8")

    target = SimpleNamespace(dataset_root=str(tmp_path))
    monkeypatch.setattr(
        offline_retarget_augment,
        "build_index",
        lambda _cond: ({}, {"target": target}, {}),
    )
    args = SimpleNamespace(cond="unused", dry_run_labels=False)
    rows = [{
        "target_species": "target",
        "target_clip": "Foo.npy",
        "action_group": "locomotion",
        "action_label": "walk",
        "source_clip": "Source_Walk",
    }]

    offline_retarget_augment._write_labels(rows, args)

    assert labels_path.read_text(encoding="utf-8").splitlines() == [json.dumps(original)]
    assert load_action_labels(tmp_path)["Foo"]["is_loop"] is True
