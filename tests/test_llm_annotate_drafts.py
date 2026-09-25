import json
import sys
from pathlib import Path


ANYTOP_ROOT = Path(__file__).resolve().parents[1]
if str(ANYTOP_ROOT) not in sys.path:
    sys.path.insert(0, str(ANYTOP_ROOT))

from dataset.review.llm_annotate import Sink  # noqa: E402


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_overwrite_invalidation_removes_old_contract_but_keeps_review_history(tmp_path):
    main = tmp_path / "action_labels.jsonl"
    review = tmp_path / "action_labels.review.jsonl"
    main.write_text(
        '{"clip": "Horse_Run", "action_label": "run, forward"}\n'
        '{"clip": "Horse_Idle", "action_label": "idle"}\n',
        encoding="utf-8",
    )
    review.write_text(
        '{"clip": "Horse_Run", "rejected": false}\n', encoding="utf-8"
    )

    sink = Sink(main, review, "clip")
    assert sink.invalidate(["Horse_Run"]) == 1

    assert _rows(main) == [{"clip": "Horse_Idle", "action_label": "idle"}]
    assert _rows(review) == [{"clip": "Horse_Run", "rejected": False}]
    assert "Horse_Run" not in sink.done
    assert "Horse_Idle" in sink.done


def test_overwrite_invalidation_is_a_noop_for_unscheduled_keys(tmp_path):
    main = tmp_path / "species_tags.jsonl"
    review = tmp_path / "species_tags.review.jsonl"
    main.write_text(
        '{"species": "Horse", "species_tags": ["Quadruped", "Galloping"]}\n',
        encoding="utf-8",
    )

    sink = Sink(main, review, "species")
    assert sink.invalidate(["Dragon"]) == 0
    assert _rows(main) == [
        {"species": "Horse", "species_tags": ["Quadruped", "Galloping"]}
    ]
