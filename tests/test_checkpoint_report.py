"""Portable review report, durable ratings, and local HTTP contract."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest


MODULE = Path(__file__).parents[1] / "eval" / "checkpoint_report.py"
SPEC = importlib.util.spec_from_file_location("checkpoint_report", MODULE)
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)

BVH = """HIERARCHY
ROOT root.with:punctuation
{
 OFFSET 0 10 0
 CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
 JOINT child
 {
  OFFSET 0 1 0
  CHANNELS 3 Zrotation Xrotation Yrotation
  End Site
  {
   OFFSET 0 1 0
  }
 }
}
MOTION
Frames: 3
Frame Time: 0.03333333
0 1 0 0 0 0 0 0 0
1 1 0 0 0 20 0 0 30
2 1 0 0 0 0 0 0 0
"""


def build(root):
    task_dir = root / "Category" / "task1"
    task_dir.mkdir(parents=True, exist_ok=True)
    for i in (0, 1, 2, 3, 4, 10):
        (task_dir / f"test {i}.bvh").write_text(BVH, encoding="utf-8")
    for name in ("_reference_input.bvh", "_retargeted_source_to_target.bvh"):
        (task_dir / name).write_text(BVH, encoding="utf-8")
    records = [dict(category="Category", index=1, task_dir=task_dir,
                    command='--action_label "walk" </script><script>alert(1)</script>',
                    status="ok", quality=.5, score=dict(physics=.25, prior=1), score_label="walk")]
    review.write_report(root / "eval_report.html", Path("checkpoint.pt"), "run", "model", records, [.5])
    return review.RatingStore(root)


def test_report_includes_all_samples_offline_assets_stats_and_safe_json(tmp_path):
    store = build(tmp_path)
    manifest = store.manifest()
    assert [clip["name"] for clip in manifest["tasks"][0]["clips"]] == [f"test {i}.bvh" for i in (0, 1, 2, 3, 4, 10)]
    assert "%20" in manifest["tasks"][0]["clips"][0]["url"]
    assert manifest["summary"]["quality"]["mean"] == .5
    html = (tmp_path / "eval_report.html").read_text(encoding="utf-8")
    assert '</script><script>alert(1)</script>' not in html
    assert "\\u003c/script>" in html
    assert (tmp_path / "report_assets/vendor/three.module.min.js").is_file()
    for name in ("eval_review.js", "eval_review.css"):
        assert (tmp_path / "report_assets" / name).is_file()
        assert f'./report_assets/{name}' in html
    result = subprocess.run([sys.executable, str(tmp_path / "serve_report.py"), "--help"], capture_output=True)
    assert result.returncode == 0


def test_ratings_survive_rebuild_restart_and_directory_copy_but_invalidate_changed_motion(tmp_path):
    root = tmp_path / "original"
    store = build(root)
    task = store.manifest()["tasks"][0]
    store.update(task["id"], task["revision"], 8)
    build(root)  # rebuilding the same report does not reset scores
    copied = tmp_path / "copied"
    shutil.copytree(root, copied)
    restored = review.RatingStore(copied)
    assert restored.snapshot()["ratings"][task["id"]]["score"] == 8
    manifest = restored.manifest()
    manifest["tasks"][0]["revision"] = "regenerated"
    review.atomic_json(copied / "eval_report.json", manifest)
    assert restored.snapshot() == {"ratings": {}, "stale_count": 1}
    with pytest.raises(ValueError, match="刷新"):
        restored.update(task["id"], task["revision"], 7)
    restored.update(task["id"], "regenerated", 9)
    restored.update(task["id"], "regenerated", None)
    assert restored.snapshot()["ratings"] == {}


def test_motion_and_command_changes_change_revision(tmp_path):
    store = build(tmp_path)
    old_revision = store.manifest()["tasks"][0]["revision"]
    (tmp_path / "Category/task1/test 0.bvh").write_text(BVH + "\n", encoding="utf-8")
    record = dict(category="Category", index=1, task_dir=tmp_path / "Category/task1", command='--action_label "walk" </script><script>alert(1)</script>', status="ok", quality=None)
    review.write_report(tmp_path / "eval_report.html", Path("checkpoint.pt"), "run", "model", [record], [])
    changed_motion = store.manifest()["tasks"][0]["revision"]
    assert changed_motion != old_revision
    record["command"] = "--action_label run"
    review.write_report(tmp_path / "eval_report.html", Path("checkpoint.pt"), "run", "model", [record], [])
    assert store.manifest()["tasks"][0]["revision"] != changed_motion
    assert store.manifest()["summary"]["quality"] == {}


@pytest.mark.parametrize("reference_field", ["reference_bvh", "reference_motion", "last_output_resolved"])
@pytest.mark.parametrize("suffix", [".bvh", ".glb", ".npy"])
def test_reference_is_portable_and_changes_invalidate_rating(tmp_path, reference_field, suffix):
    root = tmp_path / "report"
    store = build(root)
    source = tmp_path / "external reference" / ("walk" + suffix)
    source.parent.mkdir()
    viewable = source.with_suffix(".bvh") if suffix == ".npy" else source
    content = BVH.encode() if viewable.suffix == ".bvh" else b"glTF reference bytes"
    viewable.write_bytes(content)
    record = dict(category="Category", index=1, task_dir=root / "Category/task1",
                  command="--reference_motion walk", status="ok", **{reference_field: str(source)})
    if reference_field == "last_output_resolved":
        record["reference_motion"] = "$LAST_OUTPUT"

    def rebuild():
        review.write_report(root / "eval_report.html", "checkpoint.pt", "run", "model", [record], [])
        return store.manifest()["tasks"][0]

    task = rebuild()
    assert len(task["clips"]) == 6  # reference is never a generated sample
    reference_url = task["reference_clip"]["url"]
    assert (root / reference_url).read_bytes() == content
    assert (root / "report_assets/vendor/GLTFLoader.js").is_file()
    assert (root / "report_assets/vendor/BufferGeometryUtils.js").is_file()
    store.update(task["id"], task["revision"], 8)
    assert rebuild()["revision"] == task["revision"]
    copied = tmp_path / "copied"
    shutil.copytree(root, copied)
    assert (copied / reference_url).read_bytes() == content
    viewable.write_bytes(content + b"\n")
    assert rebuild()["revision"] != task["revision"]
    assert store.snapshot()["stale_count"] == 1


@pytest.mark.parametrize("score", [0, 10, -1, 1.5, True, "8", [], {}])
def test_invalid_ratings_do_not_write(tmp_path, score):
    store = build(tmp_path)
    task = store.manifest()["tasks"][0]
    with pytest.raises(ValueError):
        store.update(task["id"], task["revision"], score)
    assert not store.path.exists()


def test_corrupt_ratings_are_not_overwritten(tmp_path):
    store = build(tmp_path)
    task = store.manifest()["tasks"][0]
    store.path.write_text("broken JSON", encoding="utf-8")
    with pytest.raises(ValueError):
        store.update(task["id"], task["revision"], 4)
    assert store.path.read_text(encoding="utf-8") == "broken JSON"


def test_http_serves_report_and_persists_rating(tmp_path):
    store = build(tmp_path)
    server = review.create_server(tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    task = store.manifest()["tasks"][0]
    payload = json.dumps(dict(task_id=task["id"], revision=task["revision"], score=6)).encode()
    try:
        with urlopen(base) as response:
            assert response.status == 200
            assert b"Checkpoint Review" in response.read()
        request = Request(base + "/api/ratings", data=payload, headers={"Content-Type": "application/json"})
        with urlopen(request) as response:
            assert json.load(response)["rating"]["score"] == 6
        with urlopen(base + "/api/ratings") as response:
            assert json.load(response)["ratings"][task["id"]]["score"] == 6
        # A different website cannot silently change the local report's scores.
        request.add_header("Origin", "https://example.com")
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 403
        assert json.loads(store.path.read_text())["ratings"][task["id"]]["score"] == 6
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_auto_preview_survives_launcher_exit_and_opens_only_after_ready(tmp_path):
    build(tmp_path)
    # Run the launcher in a separate parent process, suppressing the real browser.
    # The server must still answer after this parent has exited.
    runner = """
import importlib.util, json, sys
from pathlib import Path
from urllib.request import urlopen
root = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('portable_review', root / 'serve_report.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
original = module.subprocess.Popen
state = {}
def spawn(*args, **kwargs):
    process = original(*args, **kwargs)
    state['pid'] = process.pid
    (root / 'launched.pid').write_text(str(process.pid))
    return process
def open_browser(url):
    with urlopen(url, timeout=2) as response:
        assert response.status == 200
    state['opened_url'] = url
    return True
module.subprocess.Popen = spawn
module.webbrowser.open = open_browser
state['url'] = module.launch_review(root)
print(json.dumps(state))
"""
    try:
        result = subprocess.run([sys.executable, "-c", runner, str(tmp_path)],
                                capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stderr
        state = json.loads(result.stdout)
        assert state["opened_url"] == state["url"]
        with urlopen(state["url"] + "api/ratings", timeout=2) as response:
            assert json.load(response)["ratings"] == {}
        assert (tmp_path / "serve_report.log").is_file()
    finally:
        pid_file = tmp_path / "launched.pid"
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGTERM)
            except ProcessLookupError:
                pass
