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
use ``batch_size=8 --amp_dtype fp32`` matching ``generate.bat``.
Generated clips are scored in-process with the motion quality scorer, whose
reference prior is pooled over every dataset in ``--dataset_root`` (default
``dataset/datasets.jsonl``) and indexed once for the whole battery.

Every task must name the label its clips are scored against: its own
``--action_label``, or -- for a task that generates without one (unconditional,
or conditioned only by a reference motion) -- an ``"eval_label"`` entry naming
the action of the reference clip. A task with neither fails the config load
before anything is generated; there is no default prior.

Output layout::

    Anytop/outputs/eval_checkpoint/<RUN_NAME>/<MODEL_NAME>/
        <Category>/task<N>/        # one dir per generation task
            <ObjectType>_0.npy / .bvh ...
            generate.log
            scores.json            # machine-readable per-clip scores
        eval_report.html

By default the evaluation runs *incrementally*: existing task outputs are
kept and re-scored, and only newly-added tasks are generated.  Each task's
generate.py flags — and the contents of any referenced motion/cond files — are
checksummed into a ``task_params.json`` sidecar; if a task's parameters change
(e.g. its entry in the task config is edited, or a reference file is modified
in place), the stale output is wiped and the task is regenerated.  Pass
``--overwrite`` to wipe the output root and regenerate everything.

The checkpoint and task battery are loaded from a JSON config (``--task_config``,
default ``eval/eval_tasks_locomotion.json``) so they can be tuned without
editing code. The config must define ``checkpoint.RUN_NAME`` and may define
``checkpoint.MODEL_FILE``. Each task is
``{"category": str, "args": [<generate.py flags>], "eval_label": str?}``;
path-valued flags accept absolute paths or paths relative to the Anytop dir.

The shipped batteries are ``eval/eval_tasks_locomotion.json``,
``eval/eval_tasks_stationary.json`` and ``eval/eval_tasks_transition.json``
(one per action group, each naming its own ``checkpoint.RUN_NAME``).

Usage::

    python eval/eval_checkpoint.py --task_config eval/eval_tasks_locomotion.json
    python eval/eval_checkpoint.py --task_config my_tasks.json --output_root <dir>
    python eval/eval_checkpoint.py --task_config my_tasks.json --overwrite
    python eval/eval_checkpoint.py --model_path .../model.pt --task_config my_tasks.json
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import fnmatch
import hashlib
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

from eval.motion_quality.reference_bank import reference_prior_words
from eval.motion_quality.scorer import DistributionMotionQualityScorer
from sample.generate import main as generate_main
from sample.generation_runtime import prepare_generation_runtime
from utils.parser_util import generate_args

# Sentinel resolved at run time to the first output .npy of the previous task.
_LAST_OUTPUT = "$LAST_OUTPUT"
_SCORE_TOP_K_SPECIES = 3

# Default task battery, loaded when --task_config is omitted. The batch entry
# point requires this path explicitly; the Python default is kept for direct
# invocations. The per-action-group batteries are eval_tasks_<group>.json.
_DEFAULT_TASK_CONFIG = _SCRIPT_DIR / "eval_tasks_locomotion.json"
# The datasets the scorer's reference prior is pooled over: every processed
# dataset the training cond was merged from, not just the first one.
_DEFAULT_DATASET_ROOT = _ANYTOP_DIR / "dataset" / "datasets.jsonl"
# generate.py flags whose following value is a filesystem path. Their values are
# resolved (relative → Anytop dir) when a task is loaded from the config.
_PATH_FLAGS = ("--reference_motion", "--cond_path")
# generate.py flags every task shares (model_path / output_dir are added per
# task). fp32: bf16 rounds each frame of the x0 prediction independently, and
# that white noise inflates the Jerk / Snap / SpectralFlatness scores this
# harness reports (docs/bf16_precision_issues.md). Folded into the task checksum,
# so output generated under different common flags is regenerated, not reused.
_COMMON_GENERATE_ARGS = ("--batch_size", "8", "--amp_dtype", "fp32")


# ── Task battery ────────────────────────────────────────────────────────────
# Tasks are loaded from a JSON config file so the battery can be tuned without
# editing code. Each task is ``{"category": str, "args": [str, ...]}`` where
# ``args`` are the extra generate.py flags; model_path, output_dir and
# ``_COMMON_GENERATE_ARGS`` are added per task in run_task(). A task whose args
# carry no ``--action_label`` must set ``"eval_label"`` (scoring only).
#
# Path-valued flags (see ``_PATH_FLAGS``) accept either an absolute path or a
# path relative to the Anytop dir; the "$LAST_OUTPUT" sentinel passes through
# unchanged. See eval/eval_tasks_locomotion.json for the default battery.
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


def _load_task_config(config_path: Path) -> tuple[dict, list]:
    """Load checkpoint metadata and the evaluation task battery.

    The config is an object with a ``checkpoint`` object and a ``tasks`` list.
    ``checkpoint.RUN_NAME`` is required and ``checkpoint.MODEL_FILE`` is
    optional. Each task is ``{"category": str, "args": [str, ...]}`` plus an
    optional ``"eval_label"``; every task resolves to a scoring label
    (see :func:`_task_score_label`) or the whole config is rejected here,
    before any generation. Path-valued flag arguments are resolved relative to
    the Anytop dir unless absolute. Returns ``(checkpoint, [(category, args,
    score_label), ...])``.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"Task config must be a JSON object with 'checkpoint' and 'tasks': {config_path}"
        )

    checkpoint = data.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Task config {config_path} must define a 'checkpoint' object"
        )
    run_name = checkpoint.get("RUN_NAME")
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError(
            f"Task config {config_path} must define a non-empty checkpoint.RUN_NAME"
        )
    model_file = checkpoint.get("MODEL_FILE")
    if model_file is not None and (not isinstance(model_file, str) or not model_file.strip()):
        raise ValueError(
            f"Task config {config_path}: checkpoint.MODEL_FILE must be a non-empty string when set"
        )

    raw_tasks = data.get("tasks", [])
    if not isinstance(raw_tasks, list):
        raise ValueError(
            f"Task config {config_path} must define a 'tasks' list"
        )

    tasks: list[tuple[str, list[str], str]] = []
    for i, entry in enumerate(raw_tasks):
        try:
            category = entry["category"]
            args = [str(a) for a in entry["args"]]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"Task #{i} in {config_path} must have 'category' and an 'args' list ({exc})"
            )
        eval_label = entry.get("eval_label")
        if eval_label is not None and not isinstance(eval_label, str):
            raise ValueError(f"Task #{i} in {config_path}: eval_label must be a string")
        try:
            score_label = _task_score_label(args, eval_label)
        except ValueError as exc:
            raise ValueError(f"Task #{i} ({category}) in {config_path}: {exc}")
        # Resolve the value following each path-valued flag, in place.
        for j in range(len(args) - 1):
            if args[j] in _PATH_FLAGS:
                resolved = _resolve_arg_path(args[j + 1], _ANYTOP_DIR)
                # Fast-fail on missing paths (sentinel passes through).
                if resolved != _LAST_OUTPUT and not Path(resolved).is_file():
                    raise ValueError(
                        f"Task #{i} in {config_path}: {args[j]} points to "
                        f"non-existent file: {resolved}"
                    )
                args[j + 1] = resolved
        tasks.append((category, args, score_label))

    if not tasks:
        raise ValueError(f"No tasks found in config: {config_path}")
    return checkpoint, tasks


def _resolve_checkpoint(checkpoint: dict) -> Path:
    """Resolve a config checkpoint, choosing the newest model when omitted."""
    run_name = checkpoint["RUN_NAME"].strip()
    checkpoint_dir = _ANYTOP_DIR / "save" / run_name
    model_file = checkpoint.get("MODEL_FILE")

    if model_file:
        model_path = Path(os.path.expanduser(os.path.expandvars(model_file)))
        if not model_path.is_absolute():
            model_path = checkpoint_dir / model_path
        return model_path.resolve()

    candidates = sorted(
        checkpoint_dir.glob("model*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No model checkpoint found in {checkpoint_dir} (set checkpoint.MODEL_FILE to choose one)"
        )
    return candidates[0].resolve()


# ── Task parameter checksum (incremental change detection) ───────────────────
# Name of the per-task sidecar that records the checksum of the generate.py
# flags used to produce a task dir, so the incremental runner can detect when a
# task's parameters have changed and the output must be regenerated.
_TASK_HASH_FILE = "task_params.json"


def _file_content_hash(path: Path) -> str:
    """SHA-256 of a file's bytes, read in chunks so large motion files don't
    have to be loaded into memory at once."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _task_param_hash(extra_args: list[str]) -> str:
    """Deterministic SHA-256 checksum of a task's generate.py flags.

    Hashes ``_COMMON_GENERATE_ARGS`` followed by the task's ordered argument
    list, so any change to a task's parameters (a flag added/removed/reordered,
    or a value edited) or to the shared ``--batch_size`` / ``--amp_dtype`` yields
    a different digest and triggers a regeneration. ``--model_path`` and
    ``--output_dir`` are left out: both are fixed by the report root the task
    dir lives under.

    For path-valued flags (``_PATH_FLAGS``, e.g. ``--reference_motion`` /
    ``--cond_path``) the referenced file's *contents* are folded in as well, so
    editing a reference file in place (same path, new content) also changes the
    digest and forces a regen. The ``$LAST_OUTPUT`` sentinel is left as-is (it
    is resolved per-run from the previous task's output, not a fixed file).
    """
    hashed_args = [*_COMMON_GENERATE_ARGS, *extra_args]
    parts: list[str] = []
    for i, arg in enumerate(hashed_args):
        parts.append(arg)
        if i > 0 and hashed_args[i - 1] in _PATH_FLAGS and arg != _LAST_OUTPUT:
            p = Path(arg)
            parts.append(f"sha256:{_file_content_hash(p)}" if p.is_file() else "missing")
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_stored_hash(task_dir: Path) -> str | None:
    """Return the checksum recorded in a task dir's sidecar, or None if absent
    (legacy output predating checksums) or unreadable."""
    meta_path = task_dir / _TASK_HASH_FILE
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8")).get("hash")
    except (json.JSONDecodeError, OSError):
        return None


def _write_task_hash(task_dir: Path, extra_args: list[str], digest: str) -> None:
    """Record a task's parameter checksum (and the args it covers) so a later
    incremental run can tell whether the parameters have changed."""
    meta_path = task_dir / _TASK_HASH_FILE
    meta_path.write_text(
        json.dumps(
            {"hash": digest, "common_args": list(_COMMON_GENERATE_ARGS), "args": extra_args},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _first_output_npy(task_dir: Path) -> Path | None:
    """The 'first' generated motion: prefer ``*_0.npy``, else the lexically
    first .npy (excluding intermediate ``_reference_*`` / ``_retargeted_*``
    helpers written by generate.py)."""
    npys = sorted(
        p for p in task_dir.glob("*.npy")
        if not p.name.startswith("_")
    )
    if not npys:
        return None
    for p in npys:
        if p.name.endswith("_0.npy"):
            return p
    return npys[0]


def _extract_object_type(npy_path: Path) -> str | None:
    """Extract the object type from a generated .npy filename.

    Standard output files are named ``<SpeciesFileToken>_<idx>.npy`` -- the plain
    species name when it is unique across the cond, otherwise the qualified
    ``Horse@truebones_zoo_upgrade`` form. Either resolves back to a canonical
    cond key through ``resolve_species_key``.
    Intermediate helpers (``_reference_*``, ``_retargeted_*``) are excluded
    by the caller.

    Strategy:
    1. Match ``<name>_<digit>.npy`` — take ``<name>``.
    2. Fallback: strip ``.npy`` suffix and use the full stem.
    """
    m = re.match(r'^(.+)_\d+\.npy$', npy_path.name)
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


def _task_score_label(extra_args: list, eval_label: str | None) -> str:
    """The label a task's clips are scored with.

    A task that generates with ``--action_label`` is scored with that label. A
    task that generates without one (unconditional, or conditioned only by a
    reference motion) must say what its clips are in ``eval_label`` -- the
    action of the reference clip, e.g. ``"run"`` for a run-loop inpaint. Both
    at once is a contradiction and neither is an error: there is no default
    prior, because a wrong prior scores confidently and silently. The label is
    parsed under the generate.py contract so a typo fails here, not mid-run.
    """
    action_label = None
    try:
        idx = extra_args.index("--action_label")
    except ValueError:
        idx = -1
    if idx >= 0 and idx + 1 < len(extra_args) and extra_args[idx + 1].strip():
        action_label = extra_args[idx + 1].strip()

    eval_label = eval_label.strip() if eval_label else None
    if action_label and eval_label:
        raise ValueError(
            f"task has both --action_label {action_label!r} and eval_label {eval_label!r}; "
            "a task generated with --action_label is scored with it, so drop eval_label"
        )
    label = action_label or eval_label
    if not label:
        raise ValueError(
            "task has no --action_label and no eval_label, so there is no reference "
            "prior to score its clips against; add \"eval_label\": \"<action>\" "
            "naming the action of the reference clip (e.g. \"run\")"
        )
    # Parse under the generate.py contract so a typo fails here, not mid-run
    # (an unknown word or a head-less label raises before anything is generated).
    reference_prior_words(label)
    return label


def _register_cond_path(scorer: DistributionMotionQualityScorer, cond_path: str) -> dict | None:
    """Load a cond.npy and register its entries as query skeleton metadata.

    Returns the loaded cond dict (None when it could not be loaded) so the
    caller can resolve the task's filename token against it.
    """
    try:
        from data_loaders.truebones.truebones_utils.cond_schema import load_cond
        cond = load_cond(cond_path)
    except Exception as exc:
        print(f"    [WARN] failed to register cond_path {cond_path}: {exc}")
        return None
    scorer.register_cond(cond)
    return cond


def _task_object_type(file_token: str, task_cond: dict | None) -> str:
    """Canonical cond key for a task's generated clips.

    generate.py names clips after the cond it ran with: a bare species name
    when unique there, so a ``--cond_path`` task whose species also exists in
    the training corpus (``Elephant`` in ``new_skeleton_elephant`` vs
    ``truebones/zoo/Elephant``) writes ``Elephant_0.npy``. The scorer's bare-name
    rule takes the first corpus entry in insertion order, which is the corpus
    one -- the wrong skeleton with the wrong joint count. Resolving the token
    against the task's own cond first yields the canonical key, which the
    scorer matches exactly, so it can never be hijacked by a corpus namesake.
    """
    if task_cond:
        from data_loaders.truebones.truebones_utils.dataset_sources import resolve_species_key
        resolved = resolve_species_key(task_cond, file_token)
        if resolved is not None:
            return resolved
    return file_token


def _find_reference_bvh(reference_motion: str | None) -> Path | None:
    """Find the actual .bvh/.glb file for a reference motion path.

    Priority:
    1. The path itself if it already exists and has a directly-viewable
       extension (.bvh or .glb).
    2. Same directory: replace extension with .bvh, or append .bvh.
    3. ``../bvhs/`` relative to the reference motion's directory: look for a
       matching file by stem.
    """
    if not reference_motion:
        return None

    candidate = Path(reference_motion)

    # 1. Already a directly-viewable file (.bvh or .glb) and exists
    if candidate.suffix.lower() in (".bvh", ".glb") and candidate.is_file():
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
    from data_loaders.truebones.truebones_utils.param_utils import FEATS_LEN

    try:
        motion = np.load(path)
    except Exception as exc:
        print(f"    [WARN] failed to load {path.name}: {exc}")
        return None
    if motion.ndim != 3 or motion.shape[-1] != FEATS_LEN:
        print(f"    [WARN] expected (T,J,{FEATS_LEN}), got {motion.shape} - skipping {path.name}")
        return None
    return motion.astype(np.float32)


def _build_record_from_existing(
    task_dir: Path,
    category: str,
    index: int,
    scorer: DistributionMotionQualityScorer,
    root: Path,
    score_label: str,
) -> dict:
    """Build a result record by scanning an existing task directory (no generation).

    ``score_label`` comes from the task's current config: output reused here either
    matches its flags (checksum) or predates the checksum and is assumed to.
    """
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
        "score_label": score_label,
    }

    task_cond: dict | None = None
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
            task_cond = _register_cond_path(scorer, m.group(1) or m.group(2))

    first_npy = _first_output_npy(task_dir)
    record["first_npy"] = first_npy
    file_token = _extract_object_type(first_npy) if first_npy else None

    # Re-score existing clips.
    if file_token:
        object_type = _task_object_type(file_token, task_cond)
        record["scores"] = _score_task(scorer, task_dir, file_token, object_type, score_label)
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
    file_token: str,
    object_type: str,
    score_label: str,
) -> dict[str, float]:
    """Score a task's clips in-process so the reference-bank cache is reused.

    ``file_token`` is the species token the clips are named with (selects the
    files); ``object_type`` is the cond key they are scored as (see
    ``_task_object_type``). ``score_label`` selects the reference prior (see
    ``_task_score_label``).
    """
    out_json = task_dir / "scores.json"
    motion_paths = sorted(task_dir.glob(f"{file_token}_*.npy"))
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
                action_label=score_label,
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
    score_label: str,
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
        *_COMMON_GENERATE_ARGS,
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
        "score_label": score_label,
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
        record["status"] = f"generate.py failed (exit {returncode}) — see {log_path}"
        print(f"  [FAIL] {record['status']}")
        return record

    # Record the parameter checksum so a later incremental run can detect when
    # this task's flags change and regenerate instead of reusing stale output.
    _write_task_hash(task_dir, extra_args, _task_param_hash(extra_args))

    first_npy = _first_output_npy(task_dir)
    record["first_npy"] = first_npy
    file_token = _extract_object_type(first_npy) if first_npy else None

    # Register custom cond_path into the scorer so novel skeleton types
    # (e.g., 'dragon') can be resolved for query grouping and bone-length
    # scoring while reference comparisons still use the default cond baseline.
    task_cond: dict | None = None
    task_cond_path = _extract_cond_path(extra_args)
    if task_cond_path:
        task_cond = _register_cond_path(scorer, task_cond_path)

    # Score the generated clips in-process (scores.json per task).
    scores: dict[str, float] = {}
    if file_token:
        object_type = _task_object_type(file_token, task_cond)
        scores = _score_task(scorer, task_dir, file_token, object_type, score_label)
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
            prior = html.escape(str(r.get("score_label") or ""))
            score_cell = (
                f'<span class="val">{score:.4f}</span>'
                f'<br><span class="path">median of {n} clip(s) &middot; prior: {prior}</span>'
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
        "--model_path", "--model-path", default=None,
        help="Optional checkpoint override (absolute, or relative to the Anytop dir). "
             "By default it is read from checkpoint.RUN_NAME/MODEL_FILE in the task config.",
    )
    parser.add_argument(
        "--output_root", "--output-root", default=None,
        help="Override the output root (default: Anytop/outputs/eval_checkpoint/<RUN_NAME>/<MODEL_NAME>).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Wipe the output root and regenerate all tasks from scratch. "
             "By default the evaluation is incremental: existing outputs are "
             "re-scored and only new tasks are generated.",
    )
    parser.add_argument(
        "--task_config", "--task-config", default=str(_DEFAULT_TASK_CONFIG),
        help="Path to the JSON file defining the task battery (absolute, or "
             "relative to the current working directory, falling back to the "
             "Anytop dir). Default: eval/eval_tasks_locomotion.json.",
    )
    parser.add_argument(
        "--filter", default=None,
        help="Wildcard pattern to filter tasks by category name (e.g. 'loop_*', "
             "'convert*'). Uses fnmatch-style glob patterns. Without this flag, "
             "all tasks are run.",
    )
    parser.add_argument(
        "--dataset_root", "--dataset-root", default=str(_DEFAULT_DATASET_ROOT),
        help="Datasets the scorer's reference prior is pooled over: a processed "
             "dataset dir or a datasets.jsonl manifest (absolute, or relative to "
             "the Anytop dir). Default: dataset/datasets.jsonl.",
    )
    args = parser.parse_args()

    # Resolve the task config path: absolute as-is; relative against the cwd,
    # falling back to the Anytop dir so both invocation styles work.
    task_config = Path(os.path.expanduser(os.path.expandvars(args.task_config)))
    if not task_config.is_absolute():
        task_config = task_config if task_config.exists() else (_ANYTOP_DIR / task_config)
    task_config = task_config.resolve()
    if not task_config.is_file():
        print(f"ERROR: task config not found: {task_config}", file=sys.stderr)
        return 1

    try:
        checkpoint, tasks = _load_task_config(task_config)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: invalid task config: {exc}", file=sys.stderr)
        return 1

    if args.model_path:
        model_path = Path(os.path.expanduser(os.path.expandvars(args.model_path)))
        if not model_path.is_absolute():
            model_path = (_ANYTOP_DIR / model_path).resolve()
    else:
        try:
            model_path = _resolve_checkpoint(checkpoint)
        except (OSError, ValueError) as exc:
            print(f"ERROR: cannot resolve checkpoint: {exc}", file=sys.stderr)
            return 1
    if not model_path.is_file():
        print(f"ERROR: checkpoint not found: {model_path}", file=sys.stderr)
        return 1

    run_name = model_path.parent.name                      # e.g. quadropeds_locomotion_slim_v2
    model_name = model_path.stem                           # e.g. model000020000

    if args.output_root:
        root = Path(args.output_root).resolve()
    else:
        root = _ANYTOP_DIR / "outputs" / "eval_checkpoint" / run_name / model_name

    # Default is incremental (keep existing outputs). --overwrite wipes everything.
    if args.overwrite:
        if root.exists():
            print(f"Cleaning output root: {root}")
            shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(os.path.expanduser(os.path.expandvars(args.dataset_root)))
    if not dataset_root.is_absolute():
        dataset_root = _ANYTOP_DIR / dataset_root
    if not dataset_root.exists():
        print(f"ERROR: dataset root not found: {dataset_root}", file=sys.stderr)
        return 1
    scorer = DistributionMotionQualityScorer(dataset_root=str(dataset_root))
    print(f"Python      : {sys.executable}")
    print(f"Checkpoint  : {model_path}")
    print(f"Task config : {task_config}")
    print(f"Datasets    : {dataset_root}")
    print(f"Output root : {root}")

    # Filter tasks by category wildcard pattern
    if args.filter:
        filtered = [task for task in tasks if fnmatch.fnmatch(task[0], args.filter)]
        if not filtered:
            print(f"ERROR: no tasks match filter '{args.filter}'", file=sys.stderr)
            return 1
        print(f"Filter: {args.filter} → {len(filtered)}/{len(tasks)} tasks")
        tasks = filtered
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
                *_COMMON_GENERATE_ARGS,
            ])
            runtime = prepare_generation_runtime(runtime_args)
        return runtime

    for task_num, (category, extra_args, score_label) in enumerate(tasks, 1):
        cat_counter[category] = cat_counter.get(category, 0) + 1
        index = cat_counter[category]
        task_dir = root / category / f"task{index}"

        # ── Incremental mode (default): reuse existing output, re-score only. ──
        # Reuse the existing task dir only when output exists AND its recorded
        # parameter checksum still matches the current flags. A missing checksum
        # is legacy/pre-checksum output: reuse it and backfill the checksum so
        # later runs are guarded (we can't know the old params, so we assume the
        # output matches the current config rather than forcing a full regen).
        # --overwrite bypasses this check to regenerate everything.
        current_hash = _task_param_hash(extra_args)
        stored_hash = _read_stored_hash(task_dir)
        output_exists = _first_output_npy(task_dir) is not None
        params_match = stored_hash is None or stored_hash == current_hash

        if not args.overwrite and output_exists and params_match:
            print(f"\n=== {category}/task{index} ({task_num}/{total_tasks}) [reuse existing] ===")
            try:
                record = _build_record_from_existing(
                    task_dir, category, index, scorer, root, score_label
                )
            except Exception as exc:
                print(f"  [ERROR] {category}/task{index} rescore raised: {exc}")
                record = {
                    "category": category, "index": index, "task_dir": task_dir,
                    "command": "", "scores": {}, "median": None,
                    "status": f"harness error: {exc}", "first_npy": _first_output_npy(task_dir),
                    "reference_motion": None,
                }
            # Backfill the checksum for legacy output so subsequent runs can
            # detect parameter changes against it.
            if stored_hash is None:
                _write_task_hash(task_dir, extra_args, current_hash)
            if record.get("reference_motion") == _LAST_OUTPUT and prev_first_npy is not None:
                record["last_output_resolved"] = str(prev_first_npy)
            n_skipped += 1
            records.append(record)
            prev_first_npy = record["first_npy"]
            continue

        # Parameters changed since the recorded checksum: wipe the stale output
        # so the regenerated task dir contains only clips for the new params
        # (the new run may produce differently-named or fewer files).
        if not args.overwrite and output_exists:
            print(f"  [regen] {category}/task{index}: task parameters changed; regenerating")
            shutil.rmtree(task_dir, ignore_errors=True)

        # ── Otherwise generate the task (new task, or full run). ──
        try:
            record = run_task(
                _ensure_runtime(), scorer, model_path, category, index, extra_args, score_label,
                root, prev_first_npy, total=total_tasks, current=task_num,
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

        # A task that fails (generate.py error, harness error, or
        # unresolvable $LAST_OUTPUT) means the model / task config is broken.
        # Stop immediately rather than cascading through dependent tasks.
        if not record["status"].startswith("ok"):
            remaining = total_tasks - task_num
            if remaining:
                print(f"    {remaining} task(s) not attempted")
            break

    if not args.overwrite:
        print(f"\n[increment] generated {n_generated} new task(s); reused {n_skipped} existing task(s)")

    all_scores: list[float] = []
    for r in records:
        all_scores.extend(r["scores"].values())

    # Return non-zero if any task failed — skip report/overall scores.
    failed = sum(1 for r in records if not r["status"].startswith("ok"))
    if failed:
        print(f"\n[FAIL] {failed} task(s) failed — exiting with code 1")
        return 1

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
