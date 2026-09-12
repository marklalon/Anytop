#!/usr/bin/env python3
"""Local multi-dataset review server for action_labels.jsonl.

Serves review/index.html (a GIF grid) and applies label / loop / reviewed /
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

``is_loop`` is the clip's loop verdict -- proposed by
``tools/prefill_loop_flags.py`` (the detector's reading of the source
animation, ahead of preprocessing), verified and flipped by hand here. It is
an annotation in the sidecar and nowhere else (``motion_metadata.json`` no
longer carries it, and preprocessing reads it without ever writing it), but
it also shapes the clip's tensor: preprocessing writes the last velocity row
of ``motions/<clip>.npy`` as the loop's wrap delta or, for a one-shot clip,
as a repeat of the previous frame. So flipping the flag on a clip that is
already on disk also rewrites that one row in place
(``loop_verdict.rewrite_terminal_row``), keeping the flag and the tensor in
step without a re-preprocess; a clip not built yet simply takes the flag
when it is.

``action_label`` edits are normalized and validated before being written.
Tokens are lowercased, repeated words are dropped (first occurrence kept),
checked against the training pipeline's controlled vocabulary, and put in its
canonical order.  Head words retain their written order because that is the
time direction of a transition; directions and other modifiers are sorted by
the shared conditioning contract.  Existing valid labels receive the same
canonicalization when a dataset is loaded.  Existing invalid labels are kept
verbatim and exposed to the page with an error so they can be repaired there.

The header's "clean" button turns those marks into a real removal: every
``pending_delete`` row of the active dataset has its source file (looked up in
``motion_metadata.json``) *moved* -- never unlinked -- under ``--trash``
(default ``E:\\Dataset\\Temp``), its processed ``motions/`` NPY and ``bvhs/`` BVH
deleted outright, its review GIF deleted, its row dropped from
``action_labels.jsonl``, and its entry dropped from ``motion_metadata.json``
(``total_clips`` updated) -- so the dataset is loadable immediately, not just
after the next preprocess. Every source move is appended to
``<trash>/soft_deleted.jsonl`` so it can be traced back and undone by hand.

    python serve.py [--port 8765] [--datasets ../datasets.jsonl] [--no-browser]
"""
import argparse
import difflib
import json
import os
import re
import shutil
import sys
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

THIS_DIR = Path(__file__).resolve().parent      # .../dataset/review
DATASET_ROOT = THIS_DIR.parent                  # .../dataset
ANYTOP_ROOT = DATASET_ROOT.parent               # .../Anytop
INDEX = THIS_DIR / "index.html"
DEFAULT_DATASETS = DATASET_ROOT / "datasets.jsonl"

# serve.py is commonly started by path from outside the Anytop directory.  Add
# that directory explicitly so this review UI consumes the SAME vocabulary and
# canonical ordering as training, validation and generation instead of carrying
# a copy that can drift.
if str(ANYTOP_ROOT) not in sys.path:
    sys.path.insert(0, str(ANYTOP_ROOT))

from data_loaders.truebones.truebones_utils.loop_verdict import (  # noqa: E402
    rewrite_terminal_row,
)
from data_loaders.truebones.truebones_utils.motion_labels import (  # noqa: E402
    ACTION_LABEL_MAX_HEADS,
    ACTION_LABEL_MAX_WORDS,
    CONTROLLED_VOCAB,
    DIRECTION_VOCAB,
    LOOP_FLAG_KEY,
    MOTION_METADATA_SCHEMA_VERSION,
    STATE_VOCAB,
    ActionLabelError,
    canonical_action_label,
    parse_action_label,
)

# Where "clean" parks the source file of a retired clip (--trash overrides it).
TRASH_ROOT = Path(r"E:\Dataset\Temp")

# How a source path is mirrored under the trash root. The archive drive
# ``E:\Dataset`` is the reference layout -- a file already living there keeps its
# path verbatim -- and the two in-repo mirrors map back onto the names that drive
# uses, so ``...\truebones\zoo\Truebone_Z-OO\Alligator\x.glb`` lands in
# ``<trash>\Truebone_Z-OO\Alligator\x.glb``. That reproduces exactly the
# directories that were soft-deleted by hand before this button existed.
SOURCE_MIRRORS = [
    (Path(r"E:\Dataset"), ""),
    (ANYTOP_ROOT / "dataset" / "truebones" / "zoo", ""),
    (ANYTOP_ROOT / "dataset" / "truebones" / "zoo_upgrade", "Truebones_Zoo_Upgrade"),
]


def normalize_action_label(value):
    """Validate and canonicalize an ``action_label``.

    Splits on comma-family separators (ASCII / full-width comma, CJK
    enumeration comma, semicolons), drops empty parts, lowercases every
    token, and drops repeated words (first occurrence kept).  Unknown words
    raise with spelling suggestions.  The shared motion-label contract then
    checks the head/length constraints and supplies canonical order.

    Head order is intentionally preserved: for a two-head transition it is
    semantic (``idle, attack`` is not the same as ``attack, idle``).
    """
    parts = re.split(r"[,，、;；]+", str(value))
    seen = set()
    tokens = []
    for part in parts:
        token = part.strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        tokens.append(token)

    unknown = [token for token in tokens if token not in CONTROLLED_VOCAB]
    if unknown:
        details = []
        for token in unknown:
            close = difflib.get_close_matches(token, CONTROLLED_VOCAB, n=3, cutoff=0.55)
            details.append(
                repr(token) + (f" (did you mean: {', '.join(close)})" if close else "")
            )
        raise ActionLabelError(
            "unknown action-label token(s): " + "; ".join(details)
        )

    normalized = ", ".join(tokens)
    # parse_action_label enforces the rest of the shared contract: non-empty
    # labels need 1..2 head words and may contain at most 8 controlled tokens.
    parsed = parse_action_label(normalized)
    return canonical_action_label(parsed)


def clip_stem(clip):
    """``Alligator_Bite1.npy`` -> ``Alligator_Bite1`` (the GIF / BVH basename)."""
    return clip[:-4] if clip.lower().endswith(".npy") else clip


def _archive_target(src):
    """Map a source file to its slot under the trash root.

    Returns ``(dest, mapped)``; ``mapped`` is False when no SOURCE_MIRRORS entry
    covers the path, in which case the whole path (drive letter turned into a
    folder) is kept under ``_unmapped/`` so the file is still recoverable and
    the caller can say so.
    """
    best = None
    for root, mirror in SOURCE_MIRRORS:
        try:
            rel = src.relative_to(root)
        except ValueError:
            continue
        if best is None or len(root.parts) > len(best[0].parts):
            best = (root, Path(mirror) / rel)
    if best is None:
        anchor = src.drive.replace(":", "").replace("\\", "_").strip("_") or "unc"
        return TRASH_ROOT / "_unmapped" / anchor / Path(*src.parts[1:]), False
    return TRASH_ROOT / best[1], True


def _free_name(dest):
    """``a.glb`` -> ``a (2).glb`` ... so an archived file never overwrites one."""
    if not dest.exists():
        return dest
    n = 2
    while True:
        cand = dest.with_name(f"{dest.stem} ({n}){dest.suffix}")
        if not cand.exists():
            return cand
        n += 1


def archive_source(src):
    """Move one clip's source file under the trash root.

    Returns ``(dest, note)``. ``dest`` is None when there was nothing to move --
    the file is already gone, or already sits under the trash root -- and
    ``note`` says which case it was ("" when the move was a plain success).
    Raises OSError when the move itself fails.
    """
    src = Path(src)
    try:
        if src.resolve().is_relative_to(TRASH_ROOT.resolve()):
            return None, "源文件已在回收目录里"
    except OSError:
        pass
    if not src.exists():
        return None, "源文件已不存在"
    dest, mapped = _archive_target(src)
    dest = _free_name(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    return dest, "" if mapped else "路径不在已知数据根下，已放进 _unmapped/"


def record_archive(entries):
    """Append one JSON line per archived file so a move can be traced back."""
    if not entries:
        return
    log = TRASH_ROOT / "soft_deleted.jsonl"
    try:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8", newline="\n") as fh:
            for entry in entries:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass    # the manifest is a convenience; never fail a clean over it


def _write_metadata(path, payload):
    """Atomically rewrite ``motion_metadata.json`` in the pipeline's canonical
    format so only the changed entries differ from what preprocess wrote.

    Mirrors ``data_loaders.truebones.truebones_utils.motion_labels
    .write_motion_metadata`` (``indent=2``, ``sort_keys=True``, joined action
    fields and the loop flag stripped, ``total_clips`` recomputed).  The
    ``schema_version`` is preserved from the file (default: the current one).
    """
    motions = payload.get("motions") or {}
    dropped = ("action_group", "action_label", LOOP_FLAG_KEY, "action_tags", "species_label")
    sanitized = {
        name: {k: v for k, v in entry.items() if k not in dropped}
        for name, entry in motions.items()
        if isinstance(entry, dict)
    }
    out = {
        "schema_version": payload.get("schema_version", MOTION_METADATA_SCHEMA_VERSION),
        "total_clips": len(sanitized),
        "motions": dict(sorted(sanitized.items())),
    }
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


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
            "metadata": processed / "motion_metadata.json",
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
        self.label_errors = {}
        self.mtime = None
        self.newline = "\n"
        self._load()

    def _load(self):
        raw = self.path.read_bytes()
        self.newline = "\r\n" if b"\r\n" in raw else "\n"
        rows = []
        line_numbers = []
        for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
                line_numbers.append(line_number)
        # Force-normalize every action_label on each load (in particular at
        # startup) so every VALID label is rewritten in canonical order.  An
        # invalid existing value must remain accessible in this repair UI, so
        # keep it untouched and publish its error through snapshot().
        changed = False
        label_errors = {}
        for line_number, row in zip(line_numbers, rows):
            label = row.get("action_label")
            if label is None:
                continue
            try:
                norm = normalize_action_label(label)
            except ActionLabelError as exc:
                label_errors[row.get("clip")] = f"line {line_number}: {exc}"
                continue
            if norm and norm != label:   # keep the original if it would empty out
                row["action_label"] = norm
                changed = True
        self.rows = rows
        self.index = {row["clip"]: row for row in rows}
        self.label_errors = label_errors
        if changed:
            self._write()
        else:
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
            result = []
            for row in self.rows:
                item = dict(row)
                error = self.label_errors.get(row.get("clip"))
                if error:
                    item["label_error"] = error
                result.append(item)
            return result

    def update(self, clip, action_label=None, action_group=None, reviewed=None,
               pending_delete=None, is_loop=None):
        with self.lock:
            self._reload_if_stale()
            row = self.index.get(clip)
            if row is None:
                raise KeyError(clip)
            if is_loop is not None:
                # Always written, never popped: a verdict, once made, stays a
                # verdict (a row WITHOUT the key means "not judged yet" and
                # preprocessing would propose one again). A first verdict goes
                # in right after the label, where preprocessing's own fill
                # puts it, so the file reads the same whoever judged the clip.
                if LOOP_FLAG_KEY in row:
                    row[LOOP_FLAG_KEY] = bool(is_loop)
                else:
                    ordered = {}
                    for key, value in row.items():
                        ordered[key] = value
                        if key == "action_label":
                            ordered[LOOP_FLAG_KEY] = bool(is_loop)
                    ordered.setdefault(LOOP_FLAG_KEY, bool(is_loop))
                    row.clear()
                    row.update(ordered)
            if action_label is not None:
                action_label = normalize_action_label(action_label)
                if not action_label:
                    raise ActionLabelError("action_label must not be empty")
                row["action_label"] = action_label
                self.label_errors.pop(clip, None)
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
            result = dict(row)
            error = self.label_errors.get(clip)
            if error:
                result["label_error"] = error
            return result

    def delete(self, clips):
        """Drop rows by clip name in a single rewrite; returns how many went."""
        with self.lock:
            self._reload_if_stale()
            drop = set(clips)
            keep = [row for row in self.rows if row["clip"] not in drop]
            removed = len(self.rows) - len(keep)
            if removed:
                self.rows = keep
                self.index = {row["clip"]: row for row in keep}
                self.label_errors = {
                    clip: error for clip, error in self.label_errors.items()
                    if clip not in drop
                }
                self._write()
            return removed


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
                    "invalid_labels": sum(1 for r in rows if r.get("label_error")),
                    "loops": sum(1 for r in rows if r.get(LOOP_FLAG_KEY) is True),
                    "unflagged_loops": sum(1 for r in rows if LOOP_FLAG_KEY not in r),
                })
            return self._send_json(200, {"datasets": payload,
                                         "trash_root": str(TRASH_ROOT)})

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
                "label_contract": {
                    "controlled_vocab": list(CONTROLLED_VOCAB),
                    "state_vocab": list(STATE_VOCAB),
                    "direction_vocab": list(DIRECTION_VOCAB),
                    "max_words": ACTION_LABEL_MAX_WORDS,
                    "max_heads": ACTION_LABEL_MAX_HEADS,
                },
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
        route = urlparse(self.path).path
        if route not in ("/api/update", "/api/clean"):
            return self._send_json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, TypeError) as exc:
            return self._send_json(400, {"error": f"bad request: {exc}"})
        if route == "/api/clean":
            return self._clean(payload)

        ds, store = self._store_and_dataset(payload.get("dataset") or payload.get("ds"))
        if ds is None:
            return self._send_json(400, {"error": "no datasets configured"})
        clip = payload.get("clip")
        if not clip:
            return self._send_json(400, {"error": "clip is required"})
        label = payload.get("action_label")
        if label is not None:
            try:
                label = normalize_action_label(label)
            except ActionLabelError as exc:
                return self._send_json(400, {"error": str(exc)})
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
        is_loop = payload.get(LOOP_FLAG_KEY)
        if is_loop is not None and not isinstance(is_loop, bool):
            return self._send_json(400, {"error": f"{LOOP_FLAG_KEY} must be true or false"})
        try:
            row = store.update(
                clip,
                action_label=label,
                action_group=group,
                reviewed=payload.get("reviewed"),
                pending_delete=payload.get("pending_delete"),
                is_loop=is_loop,
            )
        except KeyError:
            return self._send_json(404, {"error": f"clip not in labels file: {clip}"})
        except ValueError as exc:
            return self._send_json(400, {"error": str(exc)})
        except OSError as exc:
            return self._send_json(500, {"error": f"write failed: {exc}"})
        notes = []
        if is_loop is not None:
            # The sidecar is the truth and is already written; now the tensor.
            # A failure here is reported, not rolled back: the flag is right,
            # the row is stale, and sending the same toggle again retries it.
            tensor_note = self._sync_terminal_row(ds, clip, is_loop)
            if tensor_note:
                notes.append(tensor_note)
        # Mirror /api/labels: include the bvhview href so the frontend's save()
        # re-render keeps the clip name clickable after an edit.
        out = dict(row)
        out["bvhview"] = _bvhview_href(ds, out["clip"])
        return self._send_json(200, {"row": out, "dataset": ds["id"], "notes": notes})

    @staticmethod
    def _sync_terminal_row(ds, clip, is_loop):
        """Rewrite ``motions/<clip>``'s terminal velocity row for ``is_loop``.

        Returns a human-readable note when the tensor could NOT be brought in
        step (no NPY on disk, no ``translation_root_index`` in the metadata, a
        failed write) and "" when it was -- or already matched.
        """
        npy_name = clip if clip.lower().endswith(".npy") else clip + ".npy"
        motion_path = Path(ds["processed"]) / "motions" / npy_name
        if not motion_path.is_file():
            return f"{clip_stem(clip)}：motions/ 下没有 NPY，仅更新了标记（下次预处理时按标记生成）"
        try:
            meta_payload = json.loads(ds["metadata"].read_text(encoding="utf-8"))
            # metadata is keyed by the .npy file name, not the sidecar's stem key
            entry = (meta_payload.get("motions") or {}).get(npy_name) or {}
            root = entry.get("translation_root_index")
        except (OSError, ValueError) as exc:
            return f"{clip_stem(clip)}：读取 {ds['metadata'].name} 失败，NPY 末帧速度未更新：{exc}"
        if root is None:
            return (f"{clip_stem(clip)}：{ds['metadata'].name} 里没有 translation_root_index，"
                    f"NPY 末帧速度未更新")
        try:
            rewrite_terminal_row(motion_path, is_loop, int(root))
        except (OSError, ValueError) as exc:
            return f"{clip_stem(clip)}：NPY 末帧速度更新失败：{exc}"
        return ""

    def _clean(self, payload):
        """Retire every ``pending_delete`` clip of one dataset, for real.

        Per clip: move its source file under the trash root -- never unlinked,
        so a mistaken clean is recoverable -- delete its ``motions/`` NPY and
        ``bvhs/`` BVH outright (they are named by the clip's stem, so each is
        unique to that clip and needs no sharing check), delete its review GIF,
        drop its row from action_labels.jsonl (one rewrite for the whole batch),
        and drop its entry from motion_metadata.json (``total_clips`` updated) --
        so the dataset is loadable immediately, not just after the next
        preprocess. A clip whose source is still shared with a clip that is
        staying is skipped with a reason and keeps its mark -- dropping the row
        while leaving the source in place would only resurrect the clip on the
        next preprocess. A clip whose source cannot be located (no
        ``source_fbx_path`` in the metadata -- typically because the source file
        was deleted externally) has nothing to move, so it is retired like any
        other: the processed artifacts are deleted and the row/entry dropped.
        """
        ds, store = self._store_and_dataset(payload.get("dataset") or payload.get("ds"))
        if ds is None:
            return self._send_json(400, {"error": "no datasets configured"})
        pending = [r["clip"] for r in store.snapshot() if r.get("pending_delete")]
        result = {
            "dataset": ds["id"],
            "processed": ds["processed"],
            "trash_root": str(TRASH_ROOT),
            "cleaned": [], "moved": 0, "npy": 0, "bvh": 0, "gifs": 0, "removed": 0,
            "metadata_clips": None, "skipped": [], "notes": [],
        }
        if not pending:
            return self._send_json(200, result)

        try:
            meta_payload = json.loads(ds["metadata"].read_text(encoding="utf-8"))
            motions = meta_payload.get("motions")
            if not isinstance(motions, dict):
                return self._send_json(500, {"error": f"{ds['metadata']} 缺少 motions 字段"})
        except (OSError, ValueError) as exc:
            return self._send_json(500, {"error": f"读取 {ds['metadata']} 失败：{exc}"})

        users = {}      # source_fbx_path -> [clip, ...]
        for clip, meta in motions.items():
            src = (meta or {}).get("source_fbx_path")
            if src:
                users.setdefault(src, []).append(clip)

        marked = set(pending)
        stamp = datetime.now().isoformat(timespec="seconds")
        archived = []
        for clip in pending:
            npy_name = clip if clip.lower().endswith(".npy") else clip + ".npy"
            src = (motions.get(npy_name) or {}).get("source_fbx_path")
            if src:
                shared = [c for c in users.get(src, []) if c not in marked]
                if shared:
                    result["skipped"].append(
                        {"clip": clip,
                         "reason": f"源文件仍被保留的 {clip_stem(shared[0])} 等 {len(shared)} 个 clip 使用"})
                    continue
                try:
                    dest, note = archive_source(src)
                except OSError as exc:
                    result["skipped"].append({"clip": clip, "reason": f"移动源文件失败：{exc}"})
                    continue
                if dest is not None:
                    result["moved"] += 1
                    archived.append({"when": stamp, "dataset": ds["id"], "clip": clip,
                                     "src": str(src), "dest": str(dest)})
                if note:
                    result["notes"].append(f"{clip_stem(clip)}：{note}")
            else:
                # The source file was deleted externally, so its metadata entry
                # (and thus source_fbx_path) is gone too.  There is nothing to
                # move -- retire the processed artifacts and drop the row anyway
                # instead of leaving an un-cleanable zombie behind.
                result["notes"].append(f"{clip_stem(clip)}：源文件已不存在（外部删除），无需移动")

            # Also retire the clip's processed data out of the dataset: its motion
            # NPY and its BVH.  Both are named by the clip's stem, so each is unique
            # to this clip (no sharing check).  Unlike the source file they are
            # hard-deleted -- the archived source can rebuild them if ever needed.
            npy_name = clip if clip.lower().endswith(".npy") else clip + ".npy"
            processed = Path(ds["processed"])   # stored as str for JSON; join needs a Path
            for art, subdir, counter in (
                (processed / "motions" / npy_name, "motions", "npy"),
                (processed / "bvhs" / (clip_stem(clip) + ".bvh"), "bvhs", "bvh"),
            ):
                try:
                    art.unlink()
                    result[counter] += 1
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    # A locked artifact must not keep the row -- and its
                    # already-archived source -- in the dataset.
                    result["notes"].append(f"{clip_stem(clip)}：{subdir} 文件删除失败：{exc}")

            gif = ds["gif_dir"] / (clip_stem(clip) + ".gif")
            try:
                gif.unlink()
                result["gifs"] += 1
            except FileNotFoundError:
                pass
            except OSError as exc:
                # The GIF is a rendered artifact; a locked file must not keep the
                # row -- and its already-archived source -- in the dataset.
                result["notes"].append(f"{clip_stem(clip)}：GIF 删除失败：{exc}")
            result["cleaned"].append(clip)

        record_archive(archived)

        # Drop the cleaned clips from motion_metadata.json BEFORE dropping their
        # label rows, so a partial failure can never leave a clip "in metadata
        # but not in labels" -- the one direction that makes the dataset
        # unloadable.  A leftover label row (the reverse) is not fatal.
        if result["cleaned"]:
            for clip in result["cleaned"]:
                # metadata is keyed by the .npy file name, not the sidecar's stem key
                motions.pop(clip if clip.lower().endswith(".npy") else clip + ".npy", None)
            try:
                _write_metadata(ds["metadata"], meta_payload)
            except OSError as exc:
                return self._send_json(500, {"error": f"写入 {ds['metadata']} 失败：{exc}"})
            result["metadata_clips"] = len(motions)

        try:
            result["removed"] = store.delete(result["cleaned"])
        except OSError as exc:
            return self._send_json(500, {"error": f"写入 {ds['labels']} 失败：{exc}"})
        return self._send_json(200, result)


def main():
    global TRASH_ROOT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--datasets", default=str(DEFAULT_DATASETS),
                        help="datasets.jsonl manifest (default: %(default)s)")
    parser.add_argument("--trash", default=str(TRASH_ROOT),
                        help="where \"clean\" parks retired source files (default: %(default)s)")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    TRASH_ROOT = Path(args.trash).expanduser()

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
        invalid = sum(1 for r in rows if r.get("label_error"))
        loops = sum(1 for r in rows if r.get(LOOP_FLAG_KEY) is True)
        unflagged = sum(1 for r in rows if LOOP_FLAG_KEY not in r)
        gifs = len(list(d["gif_dir"].glob("*.gif"))) if d["gif_dir"].is_dir() else 0
        print(
            f"  {d['id']:<28} {done}/{len(rows)} reviewed, {pending} pending, "
            f"{invalid} invalid labels, {loops} loop"
            f"{f' ({unflagged} unjudged)' if unflagged else ''}, {gifs} gifs"
        )
    print(f"labels manifest : {Path(args.datasets).resolve()}")
    print(f"clean trash dir : {TRASH_ROOT}")
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
