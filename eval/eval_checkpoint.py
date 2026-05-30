#!/usr/bin/env python
"""
Checkpoint evaluation harness
=============================

Given a single checkpoint path, run a fixed battery of generation tasks
(plain generation, energy-conditioned loops, convert-to-loop, frame/joint
inpainting, outpaint), score every generated clip with the motion quality
evaluator, and write a self-contained HTML report.

Each generation task is driven through ``sample/generate.py`` (NOT the .bat),
with ``batch_size=8 --amp_dtype bf16`` matching ``generate.bat``. Generated
clips are scored in-process with the motion quality scorer so the reference
bank cache is reused across tasks.

Output layout::

    Anytop/outputs/eval_checkpoint/<RUN_NAME>/<MODEL_NAME>/
        <Category>/task<N>/        # one dir per generation task
            <ObjectType>_#0.npy / .bvh ...
            generate.log
            scores.json            # machine-readable per-clip scores
        eval_report.html

The whole ``<RUN_NAME>/<MODEL_NAME>`` root is wiped before the run.

Usage::

    python eval/eval_checkpoint.py --model_path save/quadropeds_locomotion_slim_v2/model000020000.pt
    python eval/eval_checkpoint.py --model_path .../model.pt --output_root <dir>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import html
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_ANYTOP_DIR = _SCRIPT_DIR.parent                  # Anytop/
_REPO_ROOT = _ANYTOP_DIR.parent                   # pcvg-skeleton-animation/
if str(_ANYTOP_DIR) not in sys.path:
    sys.path.insert(0, str(_ANYTOP_DIR))

from eval.motion_quality.scorer import DistributionMotionQualityScorer

# Reference motions used by the reference-guided tasks.
_DATASET_MOTIONS = _ANYTOP_DIR / "dataset" / "truebones" / "zoo" / "truebones_processed" / "motions"
_BUFFALO_RUNLOOP = _DATASET_MOTIONS / "Buffalo_RunLoop_115.npy"
_BUFFALO_RUN_NOLOOP = _ANYTOP_DIR / "outputs" / "test" / "Buffalo_Run_Noloop.npy"

# Sentinel resolved at run time to the first output .npy of the previous task.
_LAST_OUTPUT = "$LAST_OUTPUT"
_SCORE_ACTION_TAGS = "locomotion"
_SCORE_TOP_K_SPECIES = 3


# ── Task battery ────────────────────────────────────────────────────────────
# Each task: (category, extra_args). The common args (model_path, output_dir,
# batch_size, amp_dtype) are added per task in run_task().
def build_tasks() -> list[tuple[str, list[str]]]:
    tasks: list[tuple[str, list[str]]] = []

    # Basic: plain generation + energy-conditioned loops at several lengths.
    tasks.append(("Basic", ["--object_type", "Buffalo"]))
    for frames in (30, 60, 120):
        for energy in (1, -1):
            tasks.append((
                "Basic",
                ["--object_type", "Buffalo", "--motion_frames", str(frames),
                 "--global_energy_mean", str(energy), "--loop"],
            ))

    # ConvertLoop: take a non-loop reference and convert it into a loop.
    tasks.append((
        "ConvertLoop",
        ["--reference_motion", str(_BUFFALO_RUN_NOLOOP), "--skip_timesteps", "90", "--loop"],
    ))

    # InpaintFrames: regenerate a frame window, then img2img the result.
    tasks.append((
        "InpaintFrames",
        ["--reference_motion", str(_BUFFALO_RUNLOOP), "--inpaint_frames", "20-30"],
    ))
    tasks.append((
        "InpaintFrames",
        ["--reference_motion", _LAST_OUTPUT, "--skip_timesteps", "90"],
    ))

    # InpaintJoints: regenerate selected limbs while holding the rest.
    tasks.append((
        "InpaintJoints",
        ["--reference_motion", str(_BUFFALO_RUNLOOP),
         "--inpaint_joints", "RightThigh,LeftThigh", "--loop"],
    ))

    # Outpaint: extend a reference to a longer clip.
    tasks.append((
        "Outpaint",
        ["--reference_motion", str(_BUFFALO_RUNLOOP),
         "--motion_frames", "120", "--skip_timesteps", "90"],
    ))

    return tasks


def _venv_python() -> str:
    """Resolve the project's venv python, matching generate.bat; fall back to
    the current interpreter."""
    candidate = _REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if candidate.is_file():
        return str(candidate)
    return sys.executable


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


def _bvh_href(npy_path: Path) -> str:
    """bvhview://open?url=... link to the .bvh sibling of a generated .npy.

    ``Path.as_uri()`` already percent-encodes the one character that would
    otherwise break the URL: ``#`` becomes ``%23`` so the path is not parsed as
    a fragment. The previous code wrapped this in a second ``quote(..., safe="")``
    pass, double-encoding ``%23`` into ``%2523`` so bvhview could not resolve the
    file. Use the file URI directly."""
    bvh_path = npy_path.with_suffix(".bvh")
    return f"bvhview://open?url={bvh_path.as_uri()}"


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
    python_exe: str,
    scorer: DistributionMotionQualityScorer,
    model_path: Path,
    category: str,
    index: int,
    extra_args: list[str],
    root: Path,
    prev_first_npy: Path | None,
) -> dict:
    """Run one generation task and return a result record."""
    task_dir = root / category / f"task{index}"
    task_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the $LAST_OUTPUT sentinel against the previous task's first clip.
    resolved_args = list(extra_args)
    last_output_unresolved = False
    if _LAST_OUTPUT in resolved_args:
        pos = resolved_args.index(_LAST_OUTPUT)
        if prev_first_npy is not None and prev_first_npy.is_file():
            resolved_args[pos] = str(prev_first_npy)
        else:
            last_output_unresolved = True

    cmd = [
        python_exe, str(_ANYTOP_DIR / "sample" / "generate.py"),
        "--model_path", str(model_path),
        "--output_dir", str(task_dir),
        "--batch_size", "8",
        "--amp_dtype", "bf16",
        *resolved_args,
    ]
    # Display command: only show differentiated (extra) args, not boilerplate flags.
    display_cmd = "generate.py " + " ".join(
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
    }

    if last_output_unresolved:
        record["status"] = "skipped ($LAST_OUTPUT unavailable — previous task produced no output)"
        print(f"  [SKIP] {category}/task{index}: {record['status']}")
        return record

    print(f"\n=== {category}/task{index} ===")
    print(f"  {display_cmd}")

    log_path = task_dir / "generate.log"
    # Capture stdout for the per-task log; let stderr (tqdm) pass through.
    # errors="replace": generate.py emits non-ASCII (→, ANSI) and the child's
    # stdout encoding may differ from the parent locale on Windows — never let a
    # decode error abort the battery.
    proc = subprocess.run(
        cmd, cwd=str(_ANYTOP_DIR), text=True, encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=None,
    )
    log_path.write_text(proc.stdout or "", encoding="utf-8")

    if proc.returncode != 0:
        record["status"] = f"generate.py failed (exit {proc.returncode}) — see {log_path.name}"
        print(f"  [FAIL] {record['status']}")
        return record

    first_npy = _first_output_npy(task_dir)
    record["first_npy"] = first_npy
    object_type = None
    if first_npy is not None:
        # Filenames are "<ObjectType>_#<idx>.npy".
        object_type = first_npy.name.split("_#")[0]

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
        cmd = html.escape(r["command"])
        first = r["first_npy"]
        if first is not None and first.is_file():
            rel = os.path.relpath(first, root).replace(os.sep, "/")
            href = _bvh_href(first)
            motion_cell = (
                f'<a href="{href}">{html.escape(first.stem)}</a>'
                f'<br><span class="path">{html.escape(rel)}</span>'
            )
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
            f'<td><pre class="cmd">{cmd}</pre></td>'
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
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = (_ANYTOP_DIR / model_path).resolve()
    if not model_path.is_file():
        print(f"ERROR: checkpoint not found: {model_path}", file=sys.stderr)
        return 1

    run_name = model_path.parent.name                      # e.g. quadropeds_locomotion_slim_v2
    model_name = model_path.stem                           # e.g. model000020000

    if args.output_root:
        root = Path(args.output_root).resolve()
    else:
        root = _ANYTOP_DIR / "outputs" / "eval_checkpoint" / run_name / model_name

    # Wipe the entire output root before evaluating.
    if root.exists():
        print(f"Cleaning output root: {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    python_exe = _venv_python()
    scorer = DistributionMotionQualityScorer()
    print(f"Python      : {python_exe}")
    print(f"Checkpoint  : {model_path}")
    print(f"Output root : {root}")

    tasks = build_tasks()
    # Per-category running index so dirs read task1, task2, ... within a category.
    cat_counter: dict[str, int] = {}
    records: list[dict] = []
    prev_first_npy: Path | None = None

    for category, extra_args in tasks:
        cat_counter[category] = cat_counter.get(category, 0) + 1
        index = cat_counter[category]
        try:
            record = run_task(
                python_exe, scorer, model_path, category, index, extra_args, root, prev_first_npy,
            )
        except Exception as exc:  # never let one task abort the whole battery
            print(f"  [ERROR] {category}/task{index} raised: {exc}")
            record = {
                "category": category, "index": index,
                "task_dir": root / category / f"task{index}",
                "command": "generate.py " + " ".join(extra_args),
                "scores": {}, "median": None,
                "status": f"harness error: {exc}", "first_npy": None,
            }
        records.append(record)
        # $LAST_OUTPUT tracks the immediately preceding task's first clip.
        prev_first_npy = record["first_npy"]

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
