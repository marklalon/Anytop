import json
import sys
from pathlib import Path


ANYTOP_ROOT = Path(__file__).resolve().parents[1]
for path in (ANYTOP_ROOT, ANYTOP_ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from regenerate_dataset_artifacts import _ensure_action_tags_fallback  # noqa: E402


def test_fallback_replaces_unknown_but_preserves_specific_tags(tmp_path):
    tags_path = tmp_path / "action_tags.jsonl"
    tags_path.write_text(
        '{"clip": "Test_Clearing_1.npy", "action_tags": ["unknown"]}\n'
        '{"clip": "Test_Walk_1.npy", "action_tags": ["rest"]}\n',
        encoding="utf-8",
    )

    fallbacks = _ensure_action_tags_fallback(
        tmp_path,
        [
            tmp_path / "Test_Clearing_1.npy",
            tmp_path / "Test_Walk_1.npy",
            tmp_path / "Test_Jog_1.npy",
        ],
    )

    assert fallbacks == [
        ("Test_Clearing_1.npy", ["emote"]),
        ("Test_Jog_1.npy", ["locomotion"]),
    ]
    entries = [json.loads(line) for line in tags_path.read_text(encoding="utf-8").splitlines()]
    assert entries == [
        {"clip": "Test_Clearing_1.npy", "action_tags": ["emote"]},
        {"clip": "Test_Walk_1.npy", "action_tags": ["rest"]},
        {"clip": "Test_Jog_1.npy", "action_tags": ["locomotion"]},
    ]
