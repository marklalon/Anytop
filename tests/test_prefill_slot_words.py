"""The write-back the two slot-prefill tools share (``tools/prefill_common``).

``prefill_direction_words.py`` and ``prefill_hand_words.py`` measure different
things, but they hand their verdicts to one writer, and everything that can
damage the sidecar lives there: only an EMPTY slot is ever filled, a row a
person filled between the dry run and ``--apply`` wins, the row is flagged as
tool-written and unverified, and a proposal set that would put two head orders
on one word set is refused before anything is written.
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.truebones.truebones_utils.motion_labels import (
    AUTOFILL_KEY,
    DIRECTION_VOCAB,
    HANDS_VOCAB,
    load_action_labels,
)
from data_loaders.truebones.truebones_utils.param_utils import ACTION_LABELS_FILE
from tools import prefill_common
from tools.action_label_sidecar import read_action_label_rows


ROWS = [
    {"clip": "Wolf_AtkL", "action_group": "stationary", "action_label": "attack, swat",
     "is_loop": True, "reviewed": True},
    {"clip": "Wolf_AtkR", "action_group": "stationary", "action_label": "attack, swat",
     "is_loop": True, "reviewed": True},
    {"clip": "Wolf_Walk", "action_group": "locomotion", "action_label": "walk, left",
     "is_loop": True, "reviewed": True},
    {"clip": "Wolf_Dead", "action_group": "stationary", "action_label": "idle, dead",
     "is_loop": True, "reviewed": True, "pending_delete": True},
]


class _Source:
    """The one field of a dataset source the writer reads."""

    def __init__(self, root):
        self.root = str(root)


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / ACTION_LABELS_FILE
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in ROWS) + "\n",
        encoding="utf-8", newline="\n",
    )
    return tmp_path


def _clip(root, name, label, group="stationary"):
    return {"clip": name + ".npy", "key": name, "root": str(root), "species": "zoo/Wolf",
            "group": group, "label": label, "gif_path": "", "motion_path": "",
            "labels_path": str(Path(root) / ACTION_LABELS_FILE), "reviewed": True,
            "is_loop": True}


def _rows_by_clip(root):
    return {row["clip"]: row for row in read_action_label_rows(root)}


def test_a_written_row_carries_the_word_and_the_marks(dataset):
    proposal = prefill_common.Proposal(
        _clip(dataset, "Wolf_AtkL", "attack, swat"), "write", "left side energy 0.88",
        "attack, left, swat", {"measure": "side_energy", "left_share": 0.88},
    )
    written = prefill_common.apply_proposals(
        [_Source(dataset)], [proposal], slot="direction", slot_vocab=set(DIRECTION_VOCAB),
    )
    assert written[str(dataset)] == 1
    row = _rows_by_clip(dataset)["Wolf_AtkL"]
    assert row["action_label"] == "attack, left, swat"
    # The mark is false, not absent: a person signed this row off and a tool
    # then changed it, which is more than "never reviewed" says.
    assert row["reviewed"] is False
    # Just a flag: what measured it lives in the run's output, not the sidecar.
    assert row[AUTOFILL_KEY] is True
    # Everything else on the row survives, and the sidecar still loads.
    assert row["is_loop"] is True
    assert load_action_labels(dataset)["Wolf_AtkL"]["action_label"] == "attack, left, swat"


def test_only_write_proposals_reach_the_file(dataset):
    proposals = [
        prefill_common.Proposal(_clip(dataset, "Wolf_AtkL", "attack, swat"), "review", "no dominant side"),
        prefill_common.Proposal(_clip(dataset, "Wolf_AtkR", "attack, swat"), "keep", "symmetric"),
    ]
    assert prefill_common.apply_proposals(
        [_Source(dataset)], proposals, slot="direction", slot_vocab=set(DIRECTION_VOCAB),
    ) == {}
    assert [row["action_label"] for row in read_action_label_rows(dataset)] == \
        [row["action_label"] for row in ROWS]


def test_a_slot_filled_since_the_dry_run_is_left_alone(dataset, capsys):
    """The person who typed the word outranks the measurement behind it."""
    stale = _clip(dataset, "Wolf_Walk", "walk", group="locomotion")   # the row now reads 'walk, left'
    proposal = prefill_common.Proposal(
        stale, "write", "right: support foot", "walk, right", {"measure": "support_foot"},
    )
    assert prefill_common.apply_proposals(
        [_Source(dataset)], [proposal], slot="direction", slot_vocab=set(DIRECTION_VOCAB),
    ) == {str(dataset): 0}
    row = _rows_by_clip(dataset)["Wolf_Walk"]
    assert row["action_label"] == "walk, left"
    assert AUTOFILL_KEY not in row and row["reviewed"] is True
    assert "filled by hand since the dry run" in capsys.readouterr().out


def test_an_edit_in_another_slot_survives_the_write(dataset):
    """The proposal is re-spelled onto the row as it is now, not onto the snapshot."""
    proposal = prefill_common.Proposal(
        _clip(dataset, "Wolf_AtkL", "attack, swat"), "write", "hand1",
        "attack, swat, hand1", {"measure": "arm_pose_knn"},
    )
    # somebody adds a modifier to the row in the meantime
    prefill_common.rewrite_action_label_rows(
        dataset,
        lambda entry: {**entry, "action_label": "attack, left, swat"}
        if entry["clip"] == "Wolf_AtkL" else None,
    )
    prefill_common.apply_proposals(
        [_Source(dataset)], [proposal], slot="hands", slot_vocab=set(HANDS_VOCAB),
    )
    row = _rows_by_clip(dataset)["Wolf_AtkL"]
    assert row["action_label"] == "attack, left, swat, hand1"
    assert row[AUTOFILL_KEY] is True


def test_a_retiring_row_is_neither_written_nor_loaded_as_a_clip(dataset):
    proposal = prefill_common.Proposal(
        _clip(dataset, "Wolf_Dead", "idle, dead"), "write", "left", "idle, left, dead",
        {"measure": "side_energy"},
    )
    prefill_common.apply_proposals(
        [_Source(dataset)], [proposal], slot="direction", slot_vocab=set(DIRECTION_VOCAB),
    )
    row = _rows_by_clip(dataset)["Wolf_Dead"]
    assert row["action_label"] == "idle, dead" and AUTOFILL_KEY not in row


def test_head_order_conflicts_are_refused_before_anything_is_written(dataset):
    """The sidecar's cross-row rule, run over the corpus as the write would leave it."""
    prefill_common.rewrite_action_label_rows(
        dataset,
        lambda entry: {**entry, "action_label": "attack, hover"}
        if entry["clip"] == "Wolf_AtkR" else None,
    )
    conflicting = prefill_common.Proposal(
        _clip(dataset, "Wolf_AtkL", "attack, swat"), "write", "",
        "hover, attack", {"measure": "test"},
    )
    with pytest.raises(SystemExit):
        prefill_common.check_head_order([_Source(dataset)], [conflicting])
    assert _rows_by_clip(dataset)["Wolf_AtkL"]["action_label"] == "attack, swat"


def test_the_corpus_loader_drops_retiring_rows_and_carries_the_review_marks(dataset, monkeypatch):
    """What both tools see: no pending_delete clip, and each row's marks."""
    collected = [
        {"clip": "Wolf_AtkL.npy", "species": "zoo/Wolf", "group": "stationary",
         "label": "attack, swat", "motion_path": "", "gif_path": "",
         "labels_path": str(dataset / ACTION_LABELS_FILE)},
        {"clip": "Wolf_Dead.npy", "species": "zoo/Wolf", "group": "stationary",
         "label": "idle, dead", "motion_path": "", "gif_path": "",
         "labels_path": str(dataset / ACTION_LABELS_FILE)},
    ]
    monkeypatch.setattr(prefill_common, "load_cond", lambda path: {})
    monkeypatch.setattr(prefill_common, "sources_from_cond", lambda cond, path: [_Source(dataset)])
    monkeypatch.setattr(prefill_common, "collect_clips", lambda *a, **k: collected)
    _cond, _sources, clips = prefill_common.load_corpus("unused.npy")
    assert [clip["clip"] for clip in clips] == ["Wolf_AtkL.npy"]
    assert clips[0]["key"] == "Wolf_AtkL" and clips[0]["reviewed"] is True
    assert clips[0]["is_loop"] is True


def test_spell_with_puts_the_word_where_the_contract_wants_it():
    assert prefill_common.spell_with("attack, swat", ["left"]) == "attack, left, swat"
    assert prefill_common.spell_with("walk, forward", ["hand2"]) == "walk, forward, hand2"
    assert prefill_common.spell_with("jump", ["up"]) == "jump, up"
