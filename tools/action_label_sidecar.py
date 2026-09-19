"""Raw row access to ``action_labels.jsonl``, for the annotation tools.

``motion_labels.load_action_labels`` is the training-facing view of the
sidecar: validated, and reduced to the fields the loader consumes. The
annotation tools need the other one -- every row exactly as written, review
marks and all -- and they need to write rows back. Both live here rather than
in ``motion_labels`` because neither is on the training or preprocessing path:
nothing under ``data_loaders/`` reads a raw row or edits the sidecar, and the
policy in these functions (what a tool may overwrite, how it marks what it
wrote) is annotation workflow, not dataset schema.

They are one module rather than one copy per tool because the audit, the two
slot prefills and the loop-flag prefill all use them: what "rewritten in
place" means has to be defined once (抽共用而不是复制 --
``docs/action_label_direction_dropout_and_hands_default.md``).

What the sidecar IS stays in ``motion_labels``: the file name, ``clip_key``,
the vocabulary and canonical spelling, ``AUTOFILL_KEY``, ``set_loop_flag``.
This module only reads and writes the lines.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

ANYTOP_DIR = Path(__file__).resolve().parent.parent
for _candidate in (str(ANYTOP_DIR), str(ANYTOP_DIR.parent)):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from data_loaders.truebones.truebones_utils.motion_labels import (  # noqa: E402
    AUTOFILL_KEY,
)
from data_loaders.truebones.truebones_utils.param_utils import (  # noqa: E402
    ACTION_LABELS_FILE,
)


def read_action_label_rows(dataset_dir: str | Path) -> list[dict[str, object]]:
    """Every sidecar row as written, in file order, with every key kept.

    ``motion_labels.load_action_labels`` is the validated, training-facing
    view and keeps only the fields the loader consumes; the review marks
    (``reviewed``, ``pending_delete``, ``autofill``) never reach it. A tool
    that has to see those -- to skip a clip on its way out, or to tell a
    hand-verified row from a proposed one -- reads the raw rows here. Nothing
    is validated, so a row may still spell the ``.npy`` extension in ``clip``;
    use ``motion_labels.clip_key``.
    """
    labels_path = Path(dataset_dir) / ACTION_LABELS_FILE
    rows: list[dict[str, object]] = []
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        if isinstance(entry, dict):
            rows.append(entry)
    return rows


def rewrite_action_label_rows(dataset_dir: str | Path, edit) -> int:
    """Rewrite sidecar rows in place through ``edit(entry) -> entry | None``.

    *edit* receives each row as a dict (its own deep copy, so a nested value
    it mutates still reads as a change) and returns the row to write, or
    ``None`` to leave the line alone. The file keeps its line order,
    its newline style and every line *edit* declined byte for byte; a returned
    row is serialised only when it differs from the original, so a no-op edit
    changes nothing. The write goes through a temp file and ``os.replace`` so
    a crash mid-way never leaves a half-written sidecar. Returns the number of
    rows that changed.

    This is the one way a tool writes the sidecar row by row -- the loop-flag
    prefill and the direction / hands prefills all go through it, so what
    "in place" means is defined once.
    """
    labels_path = Path(dataset_dir) / ACTION_LABELS_FILE
    if not labels_path.exists():
        return 0
    raw = labels_path.read_bytes()
    newline = "\r\n" if b"\r\n" in raw else "\n"
    lines = raw.decode("utf-8").splitlines()
    changed = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        entry = json.loads(stripped)
        if not isinstance(entry, dict):
            continue
        result = edit(copy.deepcopy(entry))
        if result is None or result == entry:
            continue
        lines[index] = json.dumps(result, ensure_ascii=False)
        changed += 1
    if not changed:
        return 0
    tmp_path = labels_path.with_name(labels_path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8", newline=newline) as handle:
        handle.write("\n".join(lines) + "\n")
    os.replace(tmp_path, labels_path)
    return changed


def autofill_action_label(entry: dict, new_label: str) -> dict:
    """Return *entry* relabelled by a prefill tool, marked for re-review.

    Keeps the row's key order (the label stays where it was), sets
    ``"reviewed": false`` rather than deleting the key -- a person signed this
    row off once and a tool then changed it, which is more than "never
    reviewed" says -- and flags ``motion_labels.AUTOFILL_KEY``. A row filled
    twice (a hands word, then a direction word) carries the one flag either
    way.
    """
    rebuilt: dict[str, object] = {}
    for key, value in entry.items():
        if key == "reviewed":
            rebuilt[key] = False
        elif key == AUTOFILL_KEY:
            continue
        else:
            rebuilt[key] = value
    rebuilt["action_label"] = new_label
    rebuilt["reviewed"] = False
    rebuilt[AUTOFILL_KEY] = True
    return rebuilt
