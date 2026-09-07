import importlib.util
import json
from pathlib import Path

import pytest


SERVE_PATH = Path(__file__).parents[1] / "dataset" / "review" / "serve.py"
SPEC = importlib.util.spec_from_file_location("action_review_serve", SERVE_PATH)
review = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(review)


def test_review_normalizer_validates_and_uses_canonical_order():
    assert review.normalize_action_label(" FAST；right, WALK, right ") == (
        "walk, right, fast"
    )
    # Head order is time order and must survive sorting; direction/modifier
    # placement is canonicalized around it.
    assert review.normalize_action_label("fast, attack, idle, left") == (
        "attack, idle, left, fast"
    )
    assert review.normalize_action_label("idle, right, turn, fast") == (
        "idle, turn, right, fast"
    )


def test_review_normalizer_rejects_unknown_words_with_suggestion():
    with pytest.raises(review.ActionLabelError) as exc_info:
        review.normalize_action_label("run, foward")
    message = str(exc_info.value)
    assert "foward" in message
    assert "forward" in message


def test_review_normalizer_enforces_label_shape():
    with pytest.raises(review.ActionLabelError, match="no head word"):
        review.normalize_action_label("fast, forward")


def test_store_sorts_valid_startup_labels_and_exposes_invalid_ones(tmp_path):
    labels = tmp_path / "action_labels.jsonl"
    input_rows = [
        {
            "clip": "valid.npy",
            "action_group": "locomotion",
            "action_label": "FAST, right, walk",
        },
        {
            "clip": "typo.npy",
            "action_group": "locomotion",
            "action_label": "run, foward",
        },
    ]
    labels.write_text(
        "".join(json.dumps(row) + "\n" for row in input_rows), encoding="utf-8"
    )

    store = review.LabelStore(labels)
    saved = [json.loads(line) for line in labels.read_text(encoding="utf-8").splitlines()]
    assert saved[0]["action_label"] == "walk, right, fast"
    assert saved[1]["action_label"] == "run, foward"

    snapshot = {row["clip"]: row for row in store.snapshot()}
    assert "label_error" not in snapshot["valid.npy"]
    assert "foward" in snapshot["typo.npy"]["label_error"]

    with pytest.raises(review.ActionLabelError):
        store.update("typo.npy", action_label="run, sidewayz")
    assert store.snapshot()[1]["action_label"] == "run, foward"

    fixed = store.update("typo.npy", action_label="FAST, forward, run")
    assert fixed["action_label"] == "run, forward, fast"
    assert "label_error" not in fixed
    assert "label_error" not in store.snapshot()[1]
