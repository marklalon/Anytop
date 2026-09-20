import importlib.util
import json
from io import BytesIO
from pathlib import Path

import pytest


SERVE_PATH = Path(__file__).parents[1] / "dataset" / "review" / "serve.py"
SPEC = importlib.util.spec_from_file_location("action_review_serve", SERVE_PATH)
review = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(review)

from data_loaders.truebones.truebones_utils.motion_labels import load_action_labels  # noqa: E402


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


def test_a_hand_edit_drops_the_prefill_flag_but_signing_it_off_keeps_it(tmp_path):
    """The ``autofill`` flag says the label on the row was written by a tool.

    Confirming that label (reviewed) leaves the flag as provenance; typing a
    different one makes the label a person's, so the flag goes.
    """
    labels = tmp_path / "action_labels.jsonl"
    row = {
        "clip": "Wolf_AtkL.npy", "action_group": "stationary",
        "action_label": "attack, left, swat", "reviewed": False,
        "autofill": True,
    }
    labels.write_text(json.dumps(row) + "\n", encoding="utf-8")
    store = review.LabelStore(labels)

    confirmed = store.update("Wolf_AtkL.npy", reviewed=True)
    assert confirmed["reviewed"] is True and "autofill" in confirmed
    # Re-saving the very same label is not a correction either.
    same = store.update("Wolf_AtkL.npy", action_label="attack, left, swat")
    assert "autofill" in same

    corrected = store.update("Wolf_AtkL.npy", action_label="attack, right, swat")
    assert corrected["action_label"] == "attack, right, swat"
    assert "autofill" not in corrected
    assert "autofill" not in json.loads(labels.read_text(encoding="utf-8").splitlines()[0])


def test_all_dataset_view_keeps_the_owner_on_duplicate_clip_names(tmp_path):
    datasets = []
    stores = {}
    for dataset_id, label in (("one", "walk, forward"), ("two", "run, forward")):
        processed = tmp_path / dataset_id
        processed.mkdir()
        labels = processed / "action_labels.jsonl"
        labels.write_text(
            json.dumps({
                "clip": "shared.npy",
                "action_group": "locomotion",
                "action_label": label,
            }) + "\n",
            encoding="utf-8",
        )
        datasets.append({
            "id": dataset_id,
            "name": dataset_id,
            "processed": str(processed),
            "labels": labels,
            "gif_dir": processed / "review" / "gif",
        })
        stores[dataset_id] = review.LabelStore(labels)

    handler = object.__new__(review.Handler)
    handler.path = "/api/labels?ds=all"
    handler.datasets = datasets
    handler.stores = stores
    handler._send_json = lambda status, payload: (status, payload)

    status, payload = handler.do_GET()

    assert status == 200
    assert payload["id"] == "all"
    assert [row["clip"] for row in payload["rows"]] == ["shared.npy", "shared.npy"]
    assert [row["_dataset"] for row in payload["rows"]] == ["one", "two"]


def test_review_search_matches_whole_words_and_keeps_substring_default():
    match = review._contains_search_term
    assert match("Bear_Walk", "walk", True)
    assert match("walk, forward", "WALK", True)
    assert not match("walking", "walk", True)
    assert not match("Walk1", "walk", True)
    assert match("walking", "walk")
    assert match("walk (left)", "(left)", True)
    assert review._row_matches_search({"action_label": "idle"}, "IDLE", whole_word=True)
    assert not review._row_matches_search({"action_label": "idle, fast"}, "idle", whole_word=True)
    assert review._row_matches_search({"action_label": "idle, fast"}, "idle")


def test_labels_api_can_filter_by_group(tmp_path):
    processed = tmp_path / "sample"
    processed.mkdir()
    labels = processed / "action_labels.jsonl"
    labels.write_text("".join(json.dumps(row) + "\n" for row in [
        {"clip": "Bear_Walk", "action_group": "locomotion", "action_label": "walk"},
        {"clip": "Bear_Stop", "action_group": "transition", "action_label": "stop"},
        {"clip": "Bear_Idle", "action_group": "stationary", "action_label": "idle"},
    ]), encoding="utf-8")
    dataset = {
        "id": "sample", "name": "sample", "processed": str(processed),
        "labels": labels, "gif_dir": processed / "review" / "gif",
    }
    handler = object.__new__(review.Handler)
    handler.datasets = [dataset]
    handler.stores = {"sample": review.LabelStore(labels)}
    handler._send_json = lambda status, payload: (status, payload)

    handler.path = "/api/labels?ds=sample&group=locomotion"
    status, payload = handler.do_GET()
    assert status == 200
    assert [row["clip"] for row in payload["rows"]] == ["Bear_Walk"]

    handler.path = "/api/labels?ds=sample"
    _, payload = handler.do_GET()
    assert len(payload["rows"]) == 3


def test_labels_api_can_filter_by_field_and_whole_word(tmp_path):
    processed = tmp_path / "sample"
    processed.mkdir()
    labels = processed / "action_labels.jsonl"
    labels.write_text("".join(json.dumps(row) + "\n" for row in [
        {"clip": "Bear_Walk", "action_group": "locomotion", "action_label": "walk, forward"},
        {"clip": "Bear_Walking", "action_group": "locomotion", "action_label": "run, forward"},
        {"clip": "Bear_Idle", "action_group": "stationary", "action_label": "idle"},
        {"clip": "Bear_IdleFast", "action_group": "stationary", "action_label": "idle, fast"},
    ]), encoding="utf-8")
    dataset = {
        "id": "sample", "name": "sample", "processed": str(processed),
        "labels": labels, "gif_dir": processed / "review" / "gif",
    }
    handler = object.__new__(review.Handler)
    handler.datasets = [dataset]
    handler.stores = {"sample": review.LabelStore(labels)}
    handler._send_json = lambda status, payload: (status, payload)

    handler.path = "/api/labels?ds=sample&q=walk&field=clip"
    status, payload = handler.do_GET()
    assert status == 200
    assert [row["clip"] for row in payload["rows"]] == ["Bear_Walk", "Bear_Walking"]

    handler.path = "/api/labels?ds=sample&q=walk&field=clip&whole_word=1"
    _, payload = handler.do_GET()
    assert [row["clip"] for row in payload["rows"]] == ["Bear_Walk"]

    handler.path = "/api/labels?ds=sample&q=walk&field=both&whole_word=1"
    _, payload = handler.do_GET()
    assert [row["clip"] for row in payload["rows"]] == ["Bear_Walk"]

    handler.path = "/api/labels?ds=sample&q=idle&field=label"
    _, payload = handler.do_GET()
    assert [row["clip"] for row in payload["rows"]] == ["Bear_Idle", "Bear_IdleFast"]

    handler.path = "/api/labels?ds=sample&q=idle&field=label&whole_word=1"
    _, payload = handler.do_GET()
    assert [row["clip"] for row in payload["rows"]] == ["Bear_Idle"]
