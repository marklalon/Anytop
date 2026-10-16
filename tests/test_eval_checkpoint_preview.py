"""Evaluation opens its completed report by default, with a batch-mode opt-out."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from eval import checkpoint_report as review
from eval import eval_checkpoint as evaluation


@pytest.mark.parametrize("flags,failed,expected_launches", [
    ([], False, 1), (["--no_review"], False, 0),
    (["--no-review"], False, 0), ([], True, 0),
])
def test_main_preview_is_success_only_and_can_be_disabled(tmp_path, monkeypatch, flags, failed, expected_launches):
    root = tmp_path / "report"
    task_dir = root / "Category/task1"
    task_dir.mkdir(parents=True)
    first = task_dir / "sample_0.npy"
    first.write_bytes(b"existing generation; scoring is replaced in this test")
    model = tmp_path / "model.pt"
    model.touch()
    config = tmp_path / "tasks.json"
    config.write_text("{}")
    record = dict(category="Category", index=1, task_dir=task_dir, command="--action_label walk",
                  status="harness error: failed task" if failed else "ok", first_npy=first,
                  quality=None, score=None, reference_motion=None)
    monkeypatch.setattr(evaluation, "_load_task_config", lambda _: ({"RUN_NAME": "test"}, [("Category", ["--action_label", "walk"], "walk")]))
    monkeypatch.setattr(evaluation, "_resolve_checkpoint", lambda _: model)
    monkeypatch.setattr(evaluation, "MotionQualityScorer", lambda **_: object())
    monkeypatch.setattr(evaluation, "_build_record_from_existing", lambda *args: record)
    completed = []
    monkeypatch.setattr(evaluation, "write_html_report", lambda *args: completed.append(args[0]))
    launches = []

    def launch(report_root):
        assert completed == [root / "eval_report.html"]
        launches.append(report_root)
        return "http://127.0.0.1:12345/"

    monkeypatch.setattr(review, "launch_review", launch)
    monkeypatch.setattr(sys, "argv", ["eval_checkpoint.py", "--task_config", str(config),
                        "--output_root", str(root), "--dataset_root", str(tmp_path), *flags])
    assert evaluation.main() == (1 if failed else 0)
    assert len(launches) == expected_launches
