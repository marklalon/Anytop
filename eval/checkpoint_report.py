"""Build checkpoint review reports and serve portable reports using only stdlib.

Generated reports include a copy of this file as ``serve_report.py``. Run
``python serve_report.py --open`` inside a report directory to review and rate.
"""
from __future__ import annotations

import argparse
import datetime as dt
import functools
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote, urlsplit
import webbrowser
from urllib.request import urlopen


def atomic_json(path: Path, value: dict) -> None:
    """Replace a complete JSON document, without truncating an existing rating file."""
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _finite(value):
    return float(value) if value is not None and math.isfinite(float(value)) else None


def _stats(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return {}

    def percentile(p):
        index = (len(values) - 1) * p
        lo = int(index)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)

    return dict(mean=statistics.mean(values), median=statistics.median(values),
                p25=percentile(.25), p75=percentile(.75), min=values[0], max=values[-1])


def write_report(report_path, model_path, run_name, model_name, records, all_scores,
                 dimensions=("physics", "prior"), task_config=None):
    report_path = Path(report_path).resolve()
    root = report_path.parent
    root.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parent
    assets = root / "report_assets"
    assets.mkdir(exist_ok=True)
    for name in ("eval_review.js", "eval_review.css"):
        shutil.copy2(source / "report" / name, assets / name)
    vendor = assets / "vendor"
    vendor.mkdir(exist_ok=True)
    for name in ("three.module.min.js", "OrbitControls.js", "BVHLoader.js"):
        shutil.copy2(source.parent / "dataset" / "review" / "vendor" / name, vendor / name)
    shutil.copy2(Path(__file__), root / "serve_report.py")

    tasks = []
    for record in records:
        task_dir = Path(record["task_dir"])
        task_id = f'{record["category"]}/task{record["index"]}'
        digest = hashlib.sha256(record["command"].encode("utf-8"))
        params = task_dir / "task_params.json"
        if params.is_file():
            digest.update(params.read_bytes())
        clips = []
        # Include every generated sample, including samples that failed auto scoring.
        # Match eval_checkpoint._first_output_npy: underscore-prefixed files
        # are reference/retargeting intermediates, not generated samples.
        outputs = sorted((p for p in task_dir.glob("*.bvh") if not p.name.startswith("_")), key=lambda p: [
            int(part) if part.isdigit() else part
            for part in re.split(r"(\d+)", p.name)
        ])
        for path in outputs:
            digest.update(path.name.encode("utf-8"))
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            clips.append(dict(name=path.name, url=quote(path.relative_to(root).as_posix(), safe="/")))
        # Hash NPY as well when no BVH is available, so an unviewable task's
        # old rating cannot be carried onto different generated output.
        if not outputs:
            for path in sorted(p for p in task_dir.glob("*.npy") if not p.name.startswith("_")):
                digest.update(path.name.encode("utf-8"))
                digest.update(path.read_bytes())
        score = record.get("score") or {}
        reference = record.get("reference_bvh") or record.get("last_output_resolved") or record.get("reference_motion")
        reference_link = None
        if reference and reference != "$LAST_OUTPUT":
            bvh = Path(reference).resolve()
            if bvh.suffix.lower() not in (".bvh", ".glb"):
                bvh = bvh.with_suffix(".bvh")
            if bvh.is_file():
                reference_link = "bvhview://open?--reuse&url=" + quote(bvh.as_uri(), safe="/:")
        tasks.append(dict(
            id=task_id, command=record["command"], category=record["category"],
            status=record["status"], revision=digest.hexdigest(), clips=clips,
            quality=_finite(record.get("quality")),
            dimensions={dim: _finite(score.get(dim)) for dim in dimensions},
            score_label=record.get("score_label", ""), reference_link=reference_link,
        ))
    columns = {"quality": [_finite(v) for v in all_scores]}
    columns.update({dim: [task["dimensions"][dim] for task in tasks if task["quality"] is not None]
                    for dim in dimensions})
    data = dict(version=1, report=report_path.name, run_name=run_name, model_name=model_name,
                task_config=str(task_config) if task_config is not None else None,
                task_config_name=Path(task_config).name if task_config is not None else None,
                model_path=str(model_path), generated=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                summary={name: _stats(values) for name, values in columns.items()}, tasks=tasks)
    # JSON in script elements must not contain a literal closing script tag.
    embedded = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    template = (source / "report" / "eval_review.html").read_text(encoding="utf-8")
    title = f"{run_name} / {model_name}"
    if data["task_config_name"]:
        title += f" / {data['task_config_name']}"
    doc = template.replace("__REPORT_TITLE__", html.escape(title))
    doc = doc.replace("__REPORT_DATA__", embedded)
    atomic_json(root / "eval_report.json", data)
    report_path.write_text(doc, encoding="utf-8")


class RatingStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.path = self.root / "human_scores.json"
        self.lock = threading.Lock()

    def manifest(self):
        return json.loads((self.root / "eval_report.json").read_text(encoding="utf-8"))

    def _read(self):
        if not self.path.exists():
            return {"version": 1, "ratings": {}}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("ratings"), dict):
            raise ValueError("human_scores.json 格式无效，请检查该文件后重试")
        return data

    def snapshot(self):
        with self.lock:
            tasks = {task["id"]: task for task in self.manifest()["tasks"]}
            ratings = self._read()["ratings"]
            active = {key: rating for key, rating in ratings.items()
                      if key in tasks and isinstance(rating, dict)
                      and rating.get("revision") == tasks[key]["revision"]
                      and type(rating.get("score")) is int and 1 <= rating["score"] <= 9}
            return {"ratings": active, "stale_count": len(ratings) - len(active)}

    def update(self, task_id, revision, score):
        if score is not None and (type(score) is not int or not 1 <= score <= 9):
            raise ValueError("评分必须是 1–9 的整数")
        if not isinstance(task_id, str):
            raise ValueError("task_id 必须是字符串")
        with self.lock:
            task = next((task for task in self.manifest()["tasks"] if task["id"] == task_id), None)
            if task is None:
                raise ValueError("任务不存在，请刷新报告")
            if revision != task["revision"]:
                raise ValueError("任务已重新生成，请刷新报告后重新评分")
            data = self._read()
            if score is None:
                data["ratings"].pop(task_id, None)
            else:
                data["ratings"][task_id] = dict(score=score, revision=revision,
                    updated_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
            atomic_json(self.path, data)
            return data["ratings"].get(task_id)


class ReviewHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, store, **kwargs):
        self.store = store
        super().__init__(*args, directory=str(store.root), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def _json(self, status, data):
        body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/ratings":
            try:
                self._json(200, self.store.snapshot())
            except (OSError, ValueError, KeyError) as exc:
                self._json(500, {"error": str(exc)})
        else:
            if path == "/":
                self.path = "/" + quote(self.store.manifest()["report"])
            # Do not serve symlinks pointing outside this report folder.
            if not Path(self.translate_path(self.path)).resolve().is_relative_to(self.store.root):
                self.send_error(403)
                return
            super().do_GET()

    def do_POST(self):
        if urlsplit(self.path).path != "/api/ratings":
            self._json(404, {"error": "Unknown endpoint"})
            return
        # Requests come from the page on this local server only.
        origin = self.headers.get("Origin")
        if origin and origin != "http://" + self.headers.get("Host", ""):
            self._json(403, {"error": "Origin mismatch"})
            return
        if self.headers.get_content_type() != "application/json":
            self._json(415, {"error": "Expected application/json"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16384:
                raise ValueError("Invalid request size")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict) or "score" not in payload:
                raise ValueError("Missing score")
            rating = self.store.update(payload.get("task_id"), payload.get("revision"), payload["score"])
            self._json(200, {"rating": rating})
        except (ValueError, TypeError) as exc:
            self._json(400, {"error": str(exc)})
        except OSError as exc:
            self._json(500, {"error": str(exc)})


def create_server(root, port=0):
    store = RatingStore(root)
    store.manifest()  # Fail at startup if this is not a generated review report.
    return ThreadingHTTPServer(("127.0.0.1", port), functools.partial(ReviewHandler, store=store))


def launch_review(root, *, open_browser=True, timeout=10):
    """Start an independent server on a free port, then open its ready page."""
    root = Path(root).resolve()
    log_path = root / "serve_report.log"
    options = {"start_new_session": True} if os.name != "nt" else {
        "creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
    }
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", str(root / "serve_report.py"), "--port", "0"],
            cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}, **options,
        )
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"评分服务启动失败（exit {process.returncode}），见 {log_path}")
            match = re.search(r"^Review report: (http://127\.0\.0\.1:\d+/)$",
                              log_path.read_text(encoding="utf-8", errors="replace"), re.MULTILINE)
            if match:
                url = match.group(1)
                try:
                    with urlopen(url, timeout=.5) as response:
                        response.read(1)
                except OSError:
                    time.sleep(.05)
                    continue
                if open_browser:
                    try:
                        opened = webbrowser.open(url)
                    except (webbrowser.Error, OSError):
                        opened = False
                    if not opened:
                        print(f"无法自动打开浏览器，请手动访问：{url}")
                return url
            time.sleep(.05)
        raise RuntimeError(f"评分服务启动超时，见 {log_path}")
    except BaseException:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        raise


def main():
    parser = argparse.ArgumentParser(description="预览 checkpoint 报告，人工评分自动保存到 human_scores.json")
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parent,
                        help="报告目录（默认脚本所在目录）")
    parser.add_argument("--port", type=int, default=8766, help="本地端口，0 为自动分配")
    parser.add_argument("--open", action="store_true", help="打开浏览器")
    args = parser.parse_args()
    server = create_server(args.root, args.port)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Review report: {url}\nRatings: {args.root.resolve() / 'human_scores.json'}\nCtrl+C to stop.", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
