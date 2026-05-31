#!/usr/bin/env python
"""
Checkpoint evaluation harness
=============================

Given a single checkpoint path, run a fixed battery of generation tasks
(plain generation, energy-conditioned loops, convert-to-loop, frame/joint
inpainting, outpaint), score every generated clip with the motion quality
evaluator, and write a self-contained HTML report.

Generation tasks call ``sample.generate`` in-process with a shared generation
runtime, so the checkpoint/model is loaded once for the whole battery. Tasks
still use ``batch_size=8 --amp_dtype bf16`` matching ``generate.bat``.
Generated clips are scored in-process with the motion quality scorer so the
reference bank cache is reused across tasks.

Output layout::

    Anytop/outputs/eval_checkpoint/<RUN_NAME>/<MODEL_NAME>/
        <Category>/task<N>/        # one dir per generation task
            <ObjectType>_#0.npy / .bvh ...
            generate.log
            scores.json            # machine-readable per-clip scores
        eval_report.html

By default the evaluation runs *incrementally*: existing task outputs are
kept and re-scored, and only newly-added tasks are generated.  Pass ``--force``
to wipe the output root and regenerate everything.

The task battery is loaded from a JSON config (``--task_config``, default
``eval/eval_tasks.json``) so it can be tuned without editing code. Each task is
``{"category": str, "args": [<generate.py flags>]}``; path-valued flags accept
absolute paths or paths relative to the Anytop dir.

Usage::

    python eval/eval_checkpoint.py --model_path save/quadropeds_locomotion_slim_v2/model000020000.pt
    python eval/eval_checkpoint.py --model_path .../model.pt --output_root <dir>
    python eval/eval_checkpoint.py --model_path .../model.pt --task_config my_tasks.json
    python eval/eval_checkpoint.py --model_path .../model.pt --force
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import html
import io
import json
import os
import re
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_ANYTOP_DIR = _SCRIPT_DIR.parent                  # Anytop/
_REPO_ROOT = _ANYTOP_DIR.parent                   # pcvg-skeleton-animation/
if str(_ANYTOP_DIR) not in sys.path:
    sys.path.insert(0, str(_ANYTOP_DIR))

from eval.motion_quality.scorer import DistributionMotionQualityScorer
from sample.generate import main as generate_main
from sample.generate import prepare_generation_runtime
from utils.parser_util import generate_args

# Sentinel resolved at run time to the first output .npy of the previous task.
_LAST_OUTPUT = "$LAST_OUTPUT"
_SCORE_ACTION_TAGS = "locomotion"
_SCORE_TOP_K_SPECIES = 3

# Default task battery, loaded by build_tasks() when --task_config is omitted.
_DEFAULT_TASK_CONFIG = _SCRIPT_DIR / "eval_tasks.json"
# generate.py flags whose following value is a filesystem path. Their values are
# resolved (relative → Anytop dir) when a task is loaded from the config.
_PATH_FLAGS = ("--reference_motion", "--cond_path")


# ── Task battery ────────────────────────────────────────────────────────────
# Tasks are loaded from a JSON config file so the battery can be tuned without
# editing code. Each task is ``{"category": str, "args": [str, ...]}`` where
# ``args`` are the extra generate.py flags; the common args (model_path,
# output_dir, batch_size, amp_dtype) are added per task in run_task().
#
# Path-valued flags (see ``_PATH_FLAGS``) accept either an absolute path or a
# path relative to the Anytop dir; the "$LAST_OUTPUT" sentinel passes through
# unchanged. See eval/eval_tasks.json for the default battery.
def _resolve_arg_path(value: str, base_dir: Path) -> str:
    """Resolve a path-valued task arg.

    Absolute paths and the ``$LAST_OUTPUT`` sentinel pass through unchanged;
    relative paths are resolved against ``base_dir``. ``~`` and ``$VARS`` are
    expanded for either form.
    """
    if value == _LAST_OUTPUT:
        return value
    p = Path(os.path.expanduser(os.path.expandvars(value)))
    if not p.is_absolute():
        p = base_dir / p
    return str(p)


def build_tasks(config_path: Path) -> list[tuple[str, list[str]]]:
    """Load the evaluation task battery from a JSON config file.

    The config is either a list of task objects or an object with a ``"tasks"``
    list. Each task is ``{"category": str, "args": [str, ...]}``. Path-valued
    flag arguments are resolved relative to the Anytop dir unless absolute.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_tasks = data.get("tasks", []) if isinstance(data, dict) else data
    if not isinstance(raw_tasks, list):
        raise ValueError(
            f"Task config must be a JSON list or an object with a 'tasks' list: {config_path}"
        )

    tasks: list[tuple[str, list[str]]] = []
    for i, entry in enumerate(raw_tasks):
        try:
            category = entry["category"]
            args = [str(a) for a in entry["args"]]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"Task #{i} in {config_path} must have 'category' and an 'args' list ({exc})"
            )
        # Resolve the value following each path-valued flag, in place.
        for j in range(len(args) - 1):
            if args[j] in _PATH_FLAGS:
                args[j + 1] = _resolve_arg_path(args[j + 1], _ANYTOP_DIR)
        tasks.append((category, args))

    if not tasks:
        raise ValueError(f"No tasks found in config: {config_path}")
    return tasks


def _first_output_npy(task_dir: Path) -> Path | None:
    """The 'first' generated motion: prefer ``*_#0.npy``, else the lexically
    first .npy (excluding intermediate ``_reference_*`` / ``_retargeted_*``
    helpers written by generate.py)."""
    npys = sorted(
        p for p in task_dir.glob("*.npy")
        if not p.name.startswith("_")
    )
    if not npys:
        return None
    for p in npys:
        if p.name.endswith("_#0.npy"):
            return p
    return npys[0]


def _extract_object_type(npy_path: Path) -> str | None:
    """Extract the object type from a generated .npy filename.

    Standard output files are named ``<ObjectType>_#<idx>.npy``.
    Intermediate helpers (``_reference_*``, ``_retargeted_*``) are excluded
    by the caller.

    Strategy:
    1. Match ``<name>_#<digit>.npy`` — take ``<name>``.
    2. Fallback: strip ``.npy`` suffix and use the full stem.
    """
    m = re.match(r'^(.+)_#\d+\.npy$', npy_path.name)
    if m:
        return m.group(1)
    # Fallback: just the stem (shouldn't happen for standard outputs).
    return npy_path.stem


def _bvh_href(npy_path: Path) -> str:
    """bvhview://open?url=... link to the .bvh sibling of a generated .npy.

    ``Path.as_uri()`` already percent-encodes the one character that would
    otherwise break the URL: ``#`` becomes ``%23`` so the path is not parsed as
    a fragment. The previous code wrapped this in a second ``quote(..., safe="")``
    pass, double-encoding ``%23`` into ``%2523`` so bvhview could not resolve the
    file. Use the file URI directly."""
    bvh_path = npy_path.with_suffix(".bvh")
    return f"bvhview://open?--reuse&url={bvh_path.as_uri()}"


def _extract_reference_motion(extra_args: list) -> str | None:
    """Extract the path value after --reference_motion in extra_args."""
    try:
        idx = extra_args.index("--reference_motion")
        if idx + 1 < len(extra_args):
            return extra_args[idx + 1]
    except ValueError:
        pass
    return None


def _extract_cond_path(extra_args: list) -> str | None:
    """Extract the path value after --cond_path in extra_args."""
    try:
        idx = extra_args.index("--cond_path")
        if idx + 1 < len(extra_args):
            return extra_args[idx + 1]
    except ValueError:
        pass
    return None


def _register_cond_path(scorer: DistributionMotionQualityScorer, cond_path: str) -> None:
    """Load a cond.npy and register its entries as query skeleton metadata."""
    try:
        cond_dict = np.load(cond_path, allow_pickle=True).item()
        scorer.register_cond(cond_dict)
    except Exception as exc:
        print(f"    [WARN] failed to register cond_path {cond_path}: {exc}")


def _find_reference_bvh(reference_motion: str | None) -> Path | None:
    """Find the actual .bvh file for a reference motion path.

    Priority:
    1. The path itself if it's already a .bvh and exists.
    2. Same directory: replace extension with .bvh, or append .bvh.
    3. ``../bvhs/`` relative to the reference motion's directory: look for a
       matching file by stem.
    """
    if not reference_motion:
        return None

    candidate = Path(reference_motion)

    # 1. Already a .bvh and exists
    if candidate.suffix.lower() == ".bvh" and candidate.is_file():
        return candidate

    # 2. Same directory: .bvh sibling or appended .bvh
    for alt in [candidate.with_suffix(".bvh"), candidate.with_name(candidate.name + ".bvh")]:
        if alt.is_file():
            return alt

    # 3. ../bvhs/ relative to the reference motion's directory
    bvhs_dir = candidate.parent.parent / "bvhs"
    if bvhs_dir.is_dir():
        stem = candidate.stem
        for f in sorted(bvhs_dir.iterdir()):
            if f.suffix.lower() == ".bvh" and f.stem == stem:
                return f
        # Broader match: check if stem appears in filename
        for f in sorted(bvhs_dir.iterdir()):
            if f.suffix.lower() == ".bvh" and stem in f.stem:
                return f

    return None


def _load_motion_for_scoring(path: Path) -> np.ndarray | None:
    try:
        motion = np.load(path)
    except Exception as exc:
        print(f"    [WARN] failed to load {path.name}: {exc}")
        return None
    if motion.ndim != 3 or motion.shape[-1] != 13:
        print(f"    [WARN] expected (T,J,13), got {motion.shape} - skipping {path.name}")
        return None
    return motion.astype(np.float32)


def _build_record_from_existing(
    task_dir: Path,
    category: str,
    index: int,
    scorer: DistributionMotionQualityScorer,
    root: Path,
) -> dict:
    """Build a result record by scanning an existing task directory (no generation)."""
    record = {
        "category": category,
        "index": index,
        "task_dir": task_dir,
        "command": "",
        "scores": {},
        "median": None,
        "status": "ok",
        "first_npy": None,
        "reference_motion": None,
    }

    # Try to recover the command from generate.log.
    log_path = task_dir / "generate.log"
    if log_path.is_file():
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            # The first line is the command, prefixed with "# ".
            for line in lines:
                line = line.strip()
                if line.startswith("# "):
                    record["command"] = line[2:]
                    break
        except Exception:
            pass

    # Extract --reference_motion from the recovered command so the HTML report
    # can render it as a clickable bvhview link.
    if record["command"]:
        m = re.search(r'--reference_motion\s+(?:"([^"]*)"|(\S+))', record["command"])
        if m:
            record["reference_motion"] = m.group(1) or m.group(2)
        m = re.search(r'--cond_path\s+(?:"([^"]*)"|(\S+))', record["command"])
        if m:
            _register_cond_path(scorer, m.group(1) or m.group(2))

    first_npy = _first_output_npy(task_dir)
    record["first_npy"] = first_npy
    object_type = _extract_object_type(first_npy) if first_npy else None

    # Re-score existing clips.
    if object_type:
        record["scores"] = _score_task(scorer, task_dir, object_type)
    else:
        print(f"    [WARN] {category}/task{index}: could not determine object_type; skipping scoring")

    if record["scores"]:
        record["median"] = float(np.median(list(record["scores"].values())))
        print(f"  {category}/task{index}: scored {len(record['scores'])} clip(s); median={record['median']:.4f}")
    else:
        record["status"] = "ok (no scores)"

    return record


def _score_task(
    scorer: DistributionMotionQualityScorer,
    task_dir: Path,
    object_type: str,
) -> dict[str, float]:
    """Score a task's clips in-process so the reference-bank cache is reused."""
    out_json = task_dir / "scores.json"
    motion_paths = sorted(task_dir.glob(f"{object_type}_*.npy"))
    if not motion_paths:
        print(f"    [WARN] no generated .npy files found for object_type={object_type!r}")
        return {}

    per_file: list[dict] = []
    scores: dict[str, float] = {}
    skipped = 0
    for path in motion_paths:
        motion = _load_motion_for_scoring(path)
        if motion is None:
            skipped += 1
            continue
        try:
            report = scorer.evaluate(
                motions=[motion],
                object_type=object_type,
                action_tags=_SCORE_ACTION_TAGS,
                top_k_species=_SCORE_TOP_K_SPECIES,
            )
        except (ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
            print(f"    [WARN] evaluation failed for {path.name}: {exc}")
            skipped += 1
            continue
        report_dict = report.as_dict()
        per_file.append({"file": path.name, "score": report_dict})
        scores[path.name] = float(report.overall_score)

    if skipped:
        print(f"    [WARN] skipped {skipped} invalid clip(s)")
    if per_file:
        payload = {
            "per_file": per_file,
            "average": {
                "overall_score": round(float(np.mean(list(scores.values()))), 4),
            },
        }
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return scores


def run_task(
    runtime,
    scorer: DistributionMotionQualityScorer,
    model_path: Path,
    category: str,
    index: int,
    extra_args: list[str],
    root: Path,
    prev_first_npy: Path | None,
    total: int = 0,
    current: int = 0,
) -> dict:
    """Run one generation task and return a result record."""
    task_dir = root / category / f"task{index}"
    task_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the $LAST_OUTPUT sentinel against the previous task's first clip.
    resolved_args = list(extra_args)
    last_output_unresolved = False
    last_output_resolved: Path | None = None
    if _LAST_OUTPUT in resolved_args:
        pos = resolved_args.index(_LAST_OUTPUT)
        if prev_first_npy is not None and prev_first_npy.is_file():
            last_output_resolved = prev_first_npy
            resolved_args[pos] = str(last_output_resolved)
        else:
            last_output_unresolved = True

    generate_argv = [
        "--model_path", str(model_path),
        "--output_dir", str(task_dir),
        "--batch_size", "8",
        "--amp_dtype", "bf16",
        *resolved_args,
    ]
    # Display command: only show differentiated (extra) args, not boilerplate flags.
    display_cmd = " ".join(
        f'"{a}"' if " " in a else a for a in extra_args
    )

    record = {
        "category": category,
        "index": index,
        "task_dir": task_dir,
        "command": display_cmd,
        "scores": {},
        "median": None,
        "status": "ok",
        "first_npy": None,
        "reference_motion": _extract_reference_motion(extra_args),
        "last_output_resolved": str(last_output_resolved) if last_output_resolved is not None else None,
    }

    if last_output_unresolved:
        record["status"] = "skipped ($LAST_OUTPUT unavailable — previous task produced no output)"
        print(f"  [SKIP] {category}/task{index} ({current}/{total}): {record['status']}")
        return record

    print(f"\n=== {category}/task{index} ({current}/{total}) ===")
    print(f"  {display_cmd}")

    log_path = task_dir / "generate.log"
    # Capture stdout for the per-task log; let stderr (tqdm/progress bars) pass through.
    stdout = io.StringIO()
    returncode = 0
    try:
        task_args = generate_args(generate_argv)
        with contextlib.redirect_stdout(stdout):
            generate_main(task_args, runtime=runtime)
    except SystemExit as exc:
        if exc.code not in (None, 0):
            returncode = int(exc.code) if isinstance(exc.code, int) else 1
            stdout.write(f"\n{exc.code}\n")
    except Exception:
        returncode = 1
        stdout.write("\n")
        stdout.write(traceback.format_exc())

    # Prepend the command line so incremental re-scoring can recover it.
    log_path.write_text(f"# {display_cmd}\n" + stdout.getvalue(), encoding="utf-8", errors="replace")

    if returncode != 0:
        record["status"] = f"generate.py failed (exit {returncode}) — see {log_path.name}"
        print(f"  [FAIL] {record['status']}")
        return record

    first_npy = _first_output_npy(task_dir)
    record["first_npy"] = first_npy
    object_type = _extract_object_type(first_npy) if first_npy else None

    # Register custom cond_path into the scorer so novel skeleton types
    # (e.g., 'dragon') can be resolved for query grouping and bone-length
    # scoring while reference comparisons still use the default cond baseline.
    task_cond_path = _extract_cond_path(extra_args)
    if task_cond_path:
        _register_cond_path(scorer, task_cond_path)

    # Score the generated clips via evaluate_motion_quality.py (JSON output).
    scores: dict[str, float] = {}
    if object_type:
        scores = _score_task(scorer, task_dir, object_type)
    else:
        print("  [WARN] could not determine object_type; skipping scoring")

    record["scores"] = scores
    if scores:
        record["median"] = float(np.median(list(scores.values())))
        print(f"  scored {len(scores)} clip(s); median={record['median']:.4f}")
    else:
        record["status"] = "ok (no scores)"
        print("  [WARN] no scores produced for this task")

    return record


def _pct(values: list[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else float("nan")


def write_html_report(
    report_path: Path,
    model_path: Path,
    run_name: str,
    model_name: str,
    records: list[dict],
    all_scores: list[float],
) -> None:
    root = report_path.parent
    n_tasks = len(records)
    n_ok = sum(1 for r in records if r["scores"])

    if all_scores:
        med, p25, p75 = _pct(all_scores, 50), _pct(all_scores, 25), _pct(all_scores, 75)
        overall_html = (
            f'<tr><td>median (p50)</td><td class="val">{med:.4f}</td></tr>'
            f'<tr><td>p25</td><td class="val">{p25:.4f}</td></tr>'
            f'<tr><td>p75</td><td class="val">{p75:.4f}</td></tr>'
            f'<tr><td>min</td><td class="val">{min(all_scores):.4f}</td></tr>'
            f'<tr><td>max</td><td class="val">{max(all_scores):.4f}</td></tr>'
            f'<tr><td>clips scored</td><td class="val">{len(all_scores)}</td></tr>'
        )
    else:
        overall_html = '<tr><td colspan="2">No scores produced.</td></tr>'

    def _row(rank: int, r: dict) -> str:
        label = f'{r["category"]}/task{r["index"]}'
        raw_cmd = r["command"]
        ref_motion = r.get("reference_motion")

        # Build command cell: if there is a --reference_motion path, make it a
        # clickable bvhview link directly in-place (no extra appended content).
        if ref_motion == _LAST_OUTPUT:
            # $LAST_OUTPUT sentinel — link to the resolved previous task output.
            resolved_path = r.get("last_output_resolved")
            if resolved_path:
                resolved_bvh = Path(resolved_path).with_suffix(".bvh")
                if resolved_bvh.is_file():
                    href = f"bvhview://open?--reuse&url={resolved_bvh.as_uri()}"
                    idx = raw_cmd.find(_LAST_OUTPUT)
                    if idx != -1:
                        before = html.escape(raw_cmd[:idx])
                        after = html.escape(raw_cmd[idx + len(_LAST_OUTPUT):])
                        link = f'<a href="{href}">{_LAST_OUTPUT}</a>'
                        cmd_html = before + link + after
                    else:
                        cmd_html = html.escape(raw_cmd)
                else:
                    cmd_html = html.escape(raw_cmd)
            else:
                cmd_html = html.escape(raw_cmd)
        elif ref_motion:
            ref_bvh = _find_reference_bvh(ref_motion)
            if ref_bvh is not None and ref_bvh.is_file():
                href = f"bvhview://open?--reuse&url={ref_bvh.as_uri()}"
                idx = raw_cmd.find(ref_motion)
                if idx != -1:
                    before = html.escape(raw_cmd[:idx])
                    after = html.escape(raw_cmd[idx + len(ref_motion):])
                    link = f'<a href="{href}">{html.escape(ref_motion)}</a>'
                    cmd_html = before + link + after
                else:
                    cmd_html = html.escape(raw_cmd)
            else:
                cmd_html = html.escape(raw_cmd)
        else:
            cmd_html = html.escape(raw_cmd)

        cmd_cell = f'<pre class="cmd">{cmd_html}</pre>'

        first = r["first_npy"]
        if first is not None and first.is_file():
            rel = os.path.relpath(first, root).replace(os.sep, "/")
            href = _bvh_href(first)
            motion_cell = f'<a href="{href}">{html.escape(rel)}</a>'
        else:
            motion_cell = '<span class="muted">—</span>'

        if r["median"] is not None:
            score = r["median"]
            bg = "#d4edda" if score >= 0.7 else ("#fff3cd" if score >= 0.4 else "#f8d7da")
            n = len(r["scores"])
            score_cell = (
                f'<span class="val">{score:.4f}</span>'
                f'<br><span class="path">median of {n} clip(s)</span>'
            )
        else:
            bg = "#f4f4f4"
            score_cell = '<span class="muted">—</span>'

        status = r["status"]
        status_html = "" if status.startswith("ok") and status == "ok" else \
            f'<br><span class="status">{html.escape(status)}</span>'

        return (
            f"<tr>"
            f'<td style="text-align:right">{rank}</td>'
            f"<td>{html.escape(label)}{status_html}</td>"
            f'<td>{cmd_cell}</td>'
            f"<td>{motion_cell}</td>"
            f'<td style="text-align:right;background:{bg}">{score_cell}</td>'
            f"</tr>"
        )

    rows = "\n".join(_row(i + 1, r) for i, r in enumerate(records))
    generated = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Checkpoint Eval Report — {html.escape(run_name)}/{html.escape(model_name)}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 1280px; margin: 24px auto; padding: 0 16px; color: #1e1e1e; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 4px; }}
  h2 {{ font-size: 1.1rem; margin: 24px 0 8px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
  th {{ background: #e8e8e8; position: sticky; top: 0; }}
  th, td {{ padding: 6px 10px; border: 1px solid #ddd; text-align: left; vertical-align: top; }}
  tr:hover {{ background: #f0f6ff; }}
  a {{ color: #0969da; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }}
  .stat-box {{ background: #f8f8f8; border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px; }}
  .stat-box h3 {{ font-size: 0.85rem; margin: 0 0 6px; color: #555; }}
  .val {{ font-family: 'Cascadia Code', Consolas, monospace; font-weight: 600; }}
  .path {{ color: #888; font-size: 0.75rem; font-family: Consolas, monospace; }}
  .muted {{ color: #bbb; }}
  .status {{ color: #b00; font-size: 0.78rem; }}
  pre.cmd {{ background: #f4f4f4; padding: 8px; border-radius: 6px; margin: 0;
            white-space: pre-wrap; word-break: break-all; font-size: 0.78rem; }}
  .meta {{ color: #555; font-size: 0.85rem; }}
</style>
</head>
<body>
<h1>Checkpoint Evaluation Report</h1>
<p class="meta">
  Run: <b>{html.escape(run_name)}</b> &nbsp;|&nbsp; Model: <b>{html.escape(model_name)}</b><br>
  Checkpoint: <span class="path">{html.escape(str(model_path))}</span><br>
  Tasks: {n_ok}/{n_tasks} scored &nbsp;|&nbsp; Generated: {generated}
</p>

<h2>Overall score (all generated clips)</h2>
<div class="stat-grid">
  <div class="stat-box">
    <h3>Joint-naturalness quality score — 0.0 (worst) → 1.0 (best)</h3>
    <table>{overall_html}</table>
  </div>
</div>

<h2>Task details</h2>
<table>
<thead>
<tr>
  <th>#</th>
  <th>task</th>
  <th>command</th>
  <th>first output motion</th>
  <th>score (median)</th>
</tr>
</thead>
<tbody>
{rows}
</tbody>
</table>

<p style="margin-top:32px; color:#888; font-size:0.8rem;">
  Motion links use the <code>bvhview://open?url=...</code> protocol (opens the BVH viewer app).<br>
  Per-task generate logs: <code>&lt;Category&gt;/task&lt;N&gt;/generate.log</code>.
</p>
</body>
</html>"""

    report_path.write_text(doc, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a battery of generation tasks on a checkpoint and write an HTML quality report.",
    )
    parser.add_argument(
        "--model_path", "--model-path", required=True,
        help="Path to the checkpoint .pt (absolute, or relative to the Anytop dir).",
    )
    parser.add_argument(
        "--output_root", "--output-root", default=None,
        help="Override the output root (default: Anytop/outputs/eval_checkpoint/<RUN_NAME>/<MODEL_NAME>).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Wipe the output root and regenerate all tasks from scratch. "
             "By default the evaluation is incremental: existing outputs are "
             "re-scored and only new tasks are generated.",
    )
    parser.add_argument(
        "--task_config", "--task-config", default=str(_DEFAULT_TASK_CONFIG),
        help="Path to the JSON file defining the task battery (absolute, or "
             "relative to the current working directory, falling back to the "
             "Anytop dir). Default: eval/eval_tasks.json.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = (_ANYTOP_DIR / model_path).resolve()
    if not model_path.is_file():
        print(f"ERROR: checkpoint not found: {model_path}", file=sys.stderr)
        return 1

    # Resolve the task config path: absolute as-is; relative against the cwd,
    # falling back to the Anytop dir so both invocation styles work.
    task_config = Path(os.path.expanduser(os.path.expandvars(args.task_config)))
    if not task_config.is_absolute():
        task_config = task_config if task_config.exists() else (_ANYTOP_DIR / task_config)
    task_config = task_config.resolve()
    if not task_config.is_file():
        print(f"ERROR: task config not found: {task_config}", file=sys.stderr)
        return 1

    run_name = model_path.parent.name                      # e.g. quadropeds_locomotion_slim_v2
    model_name = model_path.stem                           # e.g. model000020000

    if args.output_root:
        root = Path(args.output_root).resolve()
    else:
        root = _ANYTOP_DIR / "outputs" / "eval_checkpoint" / run_name / model_name

    # Default is incremental (keep existing outputs). --force wipes everything.
    if args.force:
        if root.exists():
            print(f"Cleaning output root: {root}")
            shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    scorer = DistributionMotionQualityScorer()
    print(f"Python      : {sys.executable}")
    print(f"Checkpoint  : {model_path}")
    print(f"Task config : {task_config}")
    print(f"Output root : {root}")

    tasks = build_tasks(task_config)
    total_tasks = len(tasks)
    # Per-category running index so dirs read task1, task2, ... within a category.
    cat_counter: dict[str, int] = {}
    records: list[dict] = []
    prev_first_npy: Path | None = None
    # The generation runtime loads the checkpoint; only prepare it if at least
    # one task actually needs generating. In incremental runs where every task
    # already has output, this is never built and we just re-score + report.
    runtime = None
    n_generated = 0
    n_skipped = 0

    def _ensure_runtime():
        nonlocal runtime
        if runtime is None:
            print("Preparing shared generation runtime (loads checkpoint once)...")
            runtime_args = generate_args([
                "--model_path", str(model_path),
                "--batch_size", "8",
                "--amp_dtype", "bf16",
            ])
            runtime = prepare_generation_runtime(runtime_args)
        return runtime

    for task_num, (category, extra_args) in enumerate(tasks, 1):
        cat_counter[category] = cat_counter.get(category, 0) + 1
        index = cat_counter[category]
        task_dir = root / category / f"task{index}"

        # ── Incremental mode (default): reuse existing output, re-score only. ──
        # Skip the expensive generation step when output already exists.
        # --force bypasses this check to regenerate everything.
        if not args.force and _first_output_npy(task_dir) is not None:
            print(f"\n=== {category}/task{index} ({task_num}/{total_tasks}) [reuse existing] ===")
            try:
                record = _build_record_from_existing(task_dir, category, index, scorer, root)
            except Exception as exc:
                print(f"  [ERROR] {category}/task{index} rescore raised: {exc}")
                record = {
                    "category": category, "index": index, "task_dir": task_dir,
                    "command": "", "scores": {}, "median": None,
                    "status": f"harness error: {exc}", "first_npy": _first_output_npy(task_dir),
                    "reference_motion": None,
                }
            if record.get("reference_motion") == _LAST_OUTPUT and prev_first_npy is not None:
                record["last_output_resolved"] = str(prev_first_npy)
            n_skipped += 1
            records.append(record)
            prev_first_npy = record["first_npy"]
            continue

        # ── Otherwise generate the task (new task, or full run). ──
        try:
            record = run_task(
                _ensure_runtime(), scorer, model_path, category, index, extra_args, root, prev_first_npy,
                total=total_tasks, current=task_num,
            )
        except Exception as exc:  # never let one task abort the whole battery
            print(f"  [ERROR] {category}/task{index} raised: {exc}")
            record = {
                "category": category, "index": index,
                "task_dir": task_dir,
                "command": "generate.py " + " ".join(extra_args),
                "scores": {}, "median": None,
                "status": f"harness error: {exc}", "first_npy": None,
                "reference_motion": _extract_reference_motion(extra_args),
            }
        n_generated += 1
        records.append(record)
        # $LAST_OUTPUT tracks the immediately preceding task's first clip.
        prev_first_npy = record["first_npy"]

    if not args.force:
        print(f"\n[increment] generated {n_generated} new task(s); reused {n_skipped} existing task(s)")

    all_scores: list[float] = []
    for r in records:
        all_scores.extend(r["scores"].values())

    report_path = root / "eval_report.html"
    write_html_report(report_path, model_path, run_name, model_name, records, all_scores)

    print("\n" + "=" * 60)
    if all_scores:
        print(
            f"Overall score  median={_pct(all_scores, 50):.4f}  "
            f"p25={_pct(all_scores, 25):.4f}  p75={_pct(all_scores, 75):.4f}  "
            f"(n={len(all_scores)} clips)"
        )
    else:
        print("No scores were produced.")
    print(f"HTML report : {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
