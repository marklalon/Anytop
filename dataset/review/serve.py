#!/usr/bin/env python3
"""Local multi-dataset review server for action_labels.jsonl.

Serves review/index.html (a GIF grid) and applies label / reviewed /
pending_delete edits straight back into each dataset's action_labels.jsonl so
the page and the file never drift apart.  A dropdown in the header picks which
dataset is on screen.

Datasets are discovered from ../datasets.jsonl -- one line per processed
dataset::

    {"namespace": "unitybundles", "path": "dataset/unitybundles/processed"}
    {"namespace": "truebones/zoo", "path": "dataset/truebones/zoo/truebones_processed"}

``path`` is relative to the Anytop tree that holds this ``dataset/`` folder.
Each processed dir must carry ``action_labels.jsonl``; that dataset's review
GIFs are expected under ``<processed>/review/gif/`` (produced by that
dataset's render_gifs.py).  A dataset whose GIFs have not been rendered yet
(for example truebones/zoo_upgrade) simply shows each card's
"GIF 加载失败 · 点击重试" placeholder until the files exist -- it is listed
and fully editable either way.

``action_group`` is never blanked. A blank group makes ``load_action_labels``
exit before it reads anything else, which takes the whole sidecar -- and every
tool built on it -- down with it. Retiring a clip is spelled
``"pending_delete": true`` instead: ``load_action_labels`` ignores unknown keys,
so the marked rows stay loadable until the clip is actually removed from
``motions/`` and ``motion_metadata.json``.

    python serve.py [--port 8765] [--datasets ../datasets.jsonl] [--no-browser]
"""
import argparse
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

THIS_DIR = Path(__file__).resolve().parent      # .../dataset/review
DATASET_ROOT = THIS_DIR.parent                  # .../dataset
ANYTOP_ROOT = DATASET_ROOT.parent               # .../Anytop
INDEX = THIS_DIR / "index.html"
DEFAULT_DATASETS = DATASET_ROOT / "datasets.jsonl"


def discover_datasets(datasets_file):
    """Read datasets.jsonl into dataset descriptors (labels file must exist)."""
    out = []
    f = Path(datasets_file)
    if not f.is_file():
        return out
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        rel = entry.get("path") or entry.get("rel_path")
        if not rel:
            continue
        ns = entry.get("namespace") or entry.get("name") or Path(rel).name
        processed = (ANYTOP_ROOT / rel).resolve()
        labels = processed / "action_labels.jsonl"
        if not labels.is_file():
            continue
        gif_dir = processed / "review" / "gif"
        out.append({
            "id": ns,
            "name": ns,
            "processed": str(processed),
            "labels": labels,
            "gif_dir": gif_dir,
        })
    return out


def _bvhview_href(ds, clip):
    """Build a ``bvhview://open?--reuse&url=…`` link to a clip's ``.bvh``.

    Each processed dataset keeps its clips' BVH files under ``bvhs/`` named by
    the clip's stem (``Alligator_BigMouth.npy`` -> ``bvhs/Alligator_BigMouth.bvh``).
    Returns None when the file is missing so the page can fall back to a plain
    label. ``Path.as_uri()`` already percent-encodes ``#``, so the file URI is
    used directly -- no second quote() pass (that would double-encode it).
    """
    stem = clip[:-4] if clip.lower().endswith(".npy") else clip
    bvh = Path(ds["processed"]) / "bvhs" / f"{stem}.bvh"
    if not bvh.is_file():
        return None
    return f"bvhview://open?--reuse&url={bvh.as_uri()}"


class LabelStore:
    """One action_labels.jsonl held in memory, rewritten atomically on edits."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.lock = threading.Lock()
        self.rows = []
        self.index = {}
        self.mtime = None
        self.newline = "\n"
        self._load()

    def _load(self):
        raw = self.path.read_bytes()
        self.newline = "\r\n" if b"\r\n" in raw else "\n"
        rows = []
        for line in raw.decode("utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        self.rows = rows
        self.index = {row["clip"]: row for row in rows}
        self.mtime = self.path.stat().st_mtime_ns

    def _reload_if_stale(self):
        if self.path.stat().st_mtime_ns != self.mtime:
            self._load()

    def _write(self):
        tmp = self.path.with_name(self.path.name + ".tmp")
        with tmp.open("w", encoding="utf-8", newline=self.newline) as fh:
            for row in self.rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, self.path)
        self.mtime = self.path.stat().st_mtime_ns

    def snapshot(self):
        with self.lock:
            self._reload_if_stale()
            return [dict(row) for row in self.rows]

    def update(self, clip, action_label=None, action_group=None, reviewed=None,
               pending_delete=None):
        with self.lock:
            self._reload_if_stale()
            row = self.index.get(clip)
            if row is None:
                raise KeyError(clip)
            if action_label is not None:
                row["action_label"] = action_label
            if action_group is not None:
                if not action_group:
                    raise ValueError("action_group must not be empty")
                row["action_group"] = action_group
            if reviewed is not None:
                if reviewed:
                    row["reviewed"] = True
                else:
                    row.pop("reviewed", None)
            if pending_delete is not None:
                if pending_delete:
                    row["pending_delete"] = True
                else:
                    row.pop("pending_delete", None)
            self._write()
            return dict(row)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    datasets = []       # list of dataset descriptors (see discover_datasets)
    stores = {}         # dataset id -> LabelStore

    def log_message(self, fmt, *args):  # keep the console readable
        if self.command != "GET" or not self.path.startswith("/gif/"):
            super().log_message(fmt, *args)

    def handle_one_request(self):
        # The grid drops image src on every filter switch, so half-finished GIF
        # responses are normal; a dead client must not spew a traceback.
        try:
            super().handle_one_request()
        except (ConnectionError, TimeoutError):
            self.close_connection = True

    # -- helpers ---------------------------------------------------------
    def _query(self):
        return {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}

    def _send(self, code, body, ctype, extra=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (ConnectionError, TimeoutError):
            self.close_connection = True

    def _send_json(self, code, payload):
        self._send(code, json.dumps(payload, ensure_ascii=False), "application/json; charset=utf-8")

    def _first_dataset(self):
        return self.datasets[0]["id"] if self.datasets else None

    def _dataset(self, ds_id):
        for d in self.datasets:
            if d["id"] == ds_id:
                return d
        return None

    def _store_and_dataset(self, ds_id):
        ds = self._dataset(ds_id) or self._dataset(self._first_dataset())
        if ds is None:
            return None, None
        return ds, self.stores.get(ds["id"])

    # -- routes ----------------------------------------------------------
    def do_GET(self):
        # The browser builds GIF paths with encodeURIComponent(dataset_id), so a
        # namespace id like "truebones/zoo" arrives percent-encoded ("%2F"); decode
        # the path before routing so prefix/segment matching sees real slashes.
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            try:
                body = INDEX.read_bytes()
            except OSError:
                return self._send_json(500, {"error": f"missing {INDEX}"})
            return self._send(200, body, "text/html; charset=utf-8", {"Cache-Control": "no-store"})

        if path == "/api/datasets":
            payload = []
            for d in self.datasets:
                store = self.stores.get(d["id"])
                rows = store.snapshot() if store else []
                gifs = len(list(d["gif_dir"].glob("*.gif"))) if d["gif_dir"].is_dir() else 0
                payload.append({
                    "id": d["id"],
                    "name": d["name"],
                    "labels_path": str(d["labels"]),
                    "gif_count": gifs,
                    "total": len(rows),
                    "reviewed": sum(1 for r in rows if r.get("reviewed")),
                    "pending": sum(1 for r in rows if r.get("pending_delete")),
                })
            return self._send_json(200, {"datasets": payload})

        if path == "/api/labels":
            ds, store = self._store_and_dataset(self._query().get("ds"))
            if ds is None:
                return self._send_json(500, {"error": "no datasets configured"})
            # Each row gets a bvhview:// href to its .bvh so the grid can open the
            # motion in the BVH viewer by clicking the clip name (null when absent).
            rows = store.snapshot()
            for row in rows:
                row["bvhview"] = _bvhview_href(ds, row["clip"])
            return self._send_json(200, {
                "id": ds["id"],
                "name": ds["name"],
                "labels_path": str(ds["labels"]),
                "gif_dir": str(ds["gif_dir"]),
                "rows": rows,
            })

        if path.startswith("/gif/"):
            # /gif/<dataset id>/<clip>.gif -- ids may contain '/', so match by prefix.
            ds = None
            name = None
            for d in self.datasets:
                prefix = "/gif/" + d["id"] + "/"
                if path.startswith(prefix):
                    ds, name = d, path[len(prefix):]
                    break
            if ds is None:
                return self._send_json(404, {"error": "no such dataset"})
            name = os.path.basename(name)
            if not name.lower().endswith(".gif"):
                return self._send_json(404, {"error": "no such gif"})
            gif = ds["gif_dir"] / name
            if not gif.is_file():
                return self._send_json(404, {"error": "no such gif"})
            try:
                body = gif.read_bytes()
            except OSError as exc:
                return self._send_json(500, {"error": f"read failed: {exc}"})
            return self._send(200, body, "image/gif", {"Cache-Control": "max-age=3600"})

        return self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if urlparse(self.path).path != "/api/update":
            return self._send_json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, TypeError) as exc:
            return self._send_json(400, {"error": f"bad request: {exc}"})

        ds, store = self._store_and_dataset(payload.get("dataset") or payload.get("ds"))
        if ds is None:
            return self._send_json(400, {"error": "no datasets configured"})
        clip = payload.get("clip")
        if not clip:
            return self._send_json(400, {"error": "clip is required"})
        label = payload.get("action_label")
        if label is not None:
            label = str(label).strip()
            if not label:
                return self._send_json(400, {"error": "action_label must not be empty"})
        group = payload.get("action_group")
        if group is not None:
            group = str(group).strip()
            if not group:
                return self._send_json(400, {
                    "error": "action_group must not be empty -- a blank group makes "
                             "load_action_labels exit. Send pending_delete instead."
                })
        try:
            row = store.update(
                clip,
                action_label=label,
                action_group=group,
                reviewed=payload.get("reviewed"),
                pending_delete=payload.get("pending_delete"),
            )
        except KeyError:
            return self._send_json(404, {"error": f"clip not in labels file: {clip}"})
        except ValueError as exc:
            return self._send_json(400, {"error": str(exc)})
        except OSError as exc:
            return self._send_json(500, {"error": f"write failed: {exc}"})
        # Mirror /api/labels: include the bvhview href so the frontend's save()
        # re-render keeps the clip name clickable after an edit.
        out = dict(row)
        out["bvhview"] = _bvhview_href(ds, out["clip"])
        return self._send_json(200, {"row": out, "dataset": ds["id"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--datasets", default=str(DEFAULT_DATASETS),
                        help="datasets.jsonl manifest (default: %(default)s)")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    Handler.datasets = discover_datasets(args.datasets)
    if not Handler.datasets:
        print(f"no datasets found in {args.datasets} -- check the manifest", file=os.sys.stderr)
        raise SystemExit(1)
    for d in Handler.datasets:
        Handler.stores[d["id"]] = LabelStore(d["labels"])

    url = f"http://{args.host}:{args.port}/"
    for d in Handler.datasets:
        store = Handler.stores[d["id"]]
        rows = store.snapshot()
        done = sum(1 for r in rows if r.get("reviewed"))
        pending = sum(1 for r in rows if r.get("pending_delete"))
        gifs = len(list(d["gif_dir"].glob("*.gif"))) if d["gif_dir"].is_dir() else 0
        print(f"  {d['id']:<28} {done}/{len(rows)} reviewed, {pending} pending, {gifs} gifs")
    print(f"labels manifest : {Path(args.datasets).resolve()}")
    print(f"serving: {url}   (ctrl-c to stop)")

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 128   # a fresh page asks for ~30 GIFs at once

    httpd = Server((args.host, args.port), Handler)
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
