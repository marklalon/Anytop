import json
import sys
from pathlib import Path


ANYTOP_ROOT = Path(__file__).resolve().parents[1]
for path in (ANYTOP_ROOT, ANYTOP_ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from regenerate_dataset_artifacts import _ensure_action_labels_fallback  # noqa: E402


def test_fallback_backfills_missing_clips_and_preserves_written_labels(tmp_path):
    labels_path = tmp_path / "action_labels.jsonl"
    labels_path.write_text(
        '{"clip": "Test_Walk_1.npy", "action_group": "stationary", '
        '"action_label": "rest, settles down onto its side"}\n',
        encoding="utf-8",
    )

    fallbacks = _ensure_action_labels_fallback(
        tmp_path,
        [
            tmp_path / "Test_Walk_1.npy",
            tmp_path / "Test_Jog_1.npy",
            tmp_path / "Test_Clearing_1.npy",
        ],
    )

    # The hand-written entry wins even though its label contradicts the clip name:
    # the name can never recover what a human saw in the clip.
    assert fallbacks == [
        ("Test_Jog_1.npy", "locomotion", "run"),
        ("Test_Clearing_1.npy", "stationary", "taunt"),
    ]
    entries = [json.loads(line) for line in labels_path.read_text(encoding="utf-8").splitlines()]
    assert entries == [
        {"clip": "Test_Walk_1.npy", "action_group": "stationary",
         "action_label": "rest, settles down onto its side"},
        {"clip": "Test_Clearing_1.npy", "action_group": "stationary", "action_label": "taunt"},
        {"clip": "Test_Jog_1.npy", "action_group": "locomotion", "action_label": "run"},
    ]


def test_fallback_is_a_noop_when_every_clip_is_labeled(tmp_path):
    labels_path = tmp_path / "action_labels.jsonl"
    original = (
        '{"clip": "Test_Walk_1.npy", "action_group": "locomotion", '
        '"action_label": "walk, steps forward slowly"}\n'
    )
    labels_path.write_text(original, encoding="utf-8")

    assert _ensure_action_labels_fallback(tmp_path, [tmp_path / "Test_Walk_1.npy"]) == []
    assert labels_path.read_text(encoding="utf-8") == original
