"""Compute loop unclosure error for all is_loop motions in the dataset.

Measures the per-joint difference between the first and last frame of each
loop-classified motion clip, sorted from worst to best.  Small residuals
are suppressed so only meaningful discontinuities are flagged.

  ``wrap_gap`` is p75 of per-joint wrap-around gap ||pos_last - pos_first|| on
the root-relative positions stored in channels 0-2 of each *.npy feature file.
``loop_margin`` = wrap_gap - effective_tolerance, where effective_tolerance is
``clamp(LOOP_DETECTION_GAP_RATIO * transition_envelope,
LOOP_DETECTION_STEP_MIN, LOOP_DETECTION_STEP_MAX)``. The transition envelope is
the p65 of boundary-frame transition steps. ``root_xz_is_closed`` reports whether
the translation root's integrated XZ displacement stays within the absolute
tolerance ``LOOP_DETECTION_ROOT_XZ_TOLERANCE``. Both checks must pass for the
runtime loop decision to pass.

Feature layout per joint (13 channels):
  0-2  : root-relative global position (face Z+, translation-root centred)
  3-8  : 6D continuous rotation representation
  9-11 : velocity (per-frame delta, scaled for playspeed)
  12   : binary contact

Usage:
    python tools/compute_loop_unclosure_error.py
    python tools/compute_loop_unclosure_error.py --object-type Buffalo
"""

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import quote

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_ANYTOP_DIR = _SCRIPT_DIR.parent  # Anytop/

if str(_ANYTOP_DIR) not in sys.path:
    sys.path.insert(0, str(_ANYTOP_DIR))

from data_loaders.truebones.truebones_utils.animation_utils import (
    LOOP_DETECTION_GAP_RATIO,
    LOOP_DETECTION_ROOT_XZ_TOLERANCE,
    LOOP_DETECTION_STEP_MAX,
    LOOP_DETECTION_STEP_MIN,
    compute_motion_loop_diagnostics,
)


def load_motions(data_root: str) -> dict[str, dict]:
    """Return {motion_name: metadata} for all motions."""
    metadata_path = Path(data_root) / "motion_metadata.json"
    if not metadata_path.exists():
        print(f"ERROR: motion_metadata.json not found at {metadata_path}")
        sys.exit(1)

    with open(metadata_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    motions = payload.get("motions", payload)

    result = {}
    for name, meta in motions.items():
        if not isinstance(meta, dict):
            continue
        result[name] = meta

    return result


def compute_unclosure_error(motion_path: str, translation_root_index: int = 0) -> dict:
    """Compute first-vs-last frame feature-space differences.

    Returns a dict with per-joint and aggregate error metrics.
    """
    motion = np.load(motion_path).astype(np.float64)

    if motion.ndim != 3 or motion.shape[-1] < 13:
        return {"error": True}
    if motion.shape[0] < 2:
        return {"error": True}

    pos = motion[:, :, 0:3]
    vel = motion[:, :, 9:12]

    loop_diag = compute_motion_loop_diagnostics(
        pos, root_xz_velocity=vel, translation_root_index=translation_root_index,
    )

    wrap_gap = float(loop_diag["wrap_gap"])
    effective_tolerance = float(loop_diag["effective_tolerance"])

    return {
        "n_frames": motion.shape[0],
        "wrap_gap": wrap_gap,
        "loop_margin": wrap_gap - effective_tolerance,
        "runtime_is_loop": bool(loop_diag["is_loop"]),
        "root_xz_total_disp": float(loop_diag["root_xz_total_disp"]),
        "root_xz_is_closed": bool(loop_diag["root_xz_is_closed"]),
    }


def _bvh_href(name: str, bvh_dir_abs: Path) -> str:
    """Build a bvhview://open?--reuse&url=... link so the OS opens with the BVH viewer app."""
    stem = name[:-4] if name.endswith(".npy") else name
    bvh_path = bvh_dir_abs / f"{stem}.bvh"
    encoded = quote(str(bvh_path.as_uri()), safe="")
    return f"bvhview://open?--reuse&url={encoded}"


def write_html_report(
    results: list[dict],
    all_wrap_gap: list[float],
    runtime_loop_count: int,
    label: str,
    output_dir: Path,
    bvh_dir_abs: Path,
    args,
    closest: list[dict],
    closest_label: str,
    root_xz_closed_count: int = 0,
    all_root_xz_disp: list[float] | None = None,
):
    """Write an HTML report mirroring the CLI output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "loop_unclosure_report.html"

    # ── helper: format table rows ──
    def _closest_rows(motions):
        rows = []
        for rank, r in enumerate(motions):
            name = r["name"]
            href = _bvh_href(name, bvh_dir_abs)
            gap_bg = "#d4edda" if r["loop_margin"] <= 0 else "#f8d7da"
            xz_closed = r.get("root_xz_is_closed", True)
            xz_bg = "#d4edda" if xz_closed else "#f8d7da"
            xz_label = "Y" if xz_closed else "N"
            loop_flag = "F" if not r.get("runtime_is_loop", False) else "T"
            loop_bg = "#f8d7da" if not r.get("runtime_is_loop", False) else "#d4edda"
            rows.append(
                f"<tr>"
                f"<td style='text-align:right'>{rank + 1}</td>"
                f"<td><a href='{href}'>{name}</a></td>"
                f"<td style='text-align:right'>{r['wrap_gap']:.6f}</td>"
                f"<td style='text-align:right;background:{gap_bg}'>{r['loop_margin']:.6f}</td>"
                f"<td style='text-align:right'>{r.get('root_xz_total_disp', 0):.6f}</td>"
                f"<td style='text-align:center;background:{xz_bg}'>{xz_label}</td>"
                f"<td style='text-align:right'>{r['n_frames']}</td>"
                f"<td style='text-align:center;background:{loop_bg}'>{loop_flag}</td>"
                f"</tr>"
            )
        return "\n".join(rows)

    def _pct(values, p):
        return np.percentile(values, p)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Loop Unclosure Error Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 1200px; margin: 24px auto; padding: 0 16px; color: #1e1e1e; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 4px; }}
  h2 {{ font-size: 1.1rem; margin: 24px 0 8px; }}
  pre {{ background: #f4f4f4; padding: 12px; border-radius: 6px; overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
  th {{ background: #e8e8e8; position: sticky; top: 0; }}
  th, td {{ padding: 4px 10px; border: 1px solid #ddd; text-align: left; }}
  tr:hover {{ background: #f0f6ff; }}
  a {{ color: #0969da; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }}
  .stat-box {{ background: #f8f8f8; border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px; }}
  .stat-box h3 {{ font-size: 0.85rem; margin: 0 0 6px; color: #555; }}
  .val {{ font-family: 'Cascadia Code', Consolas, monospace; }}
</style>
</head>
<body>
<h1>Loop Unclosure Error Report</h1>
<p>Data root: {args.data_root or '(opt)'}  |  Motion dir: {args.motion_dir or '(opt)'}</p>

<h2>Parameters</h2>
<pre>
Runtime gap tolerance  = clamp({LOOP_DETECTION_GAP_RATIO} * transition_envelope, {LOOP_DETECTION_STEP_MIN}, {LOOP_DETECTION_STEP_MAX})
Root XZ tolerance      = {LOOP_DETECTION_ROOT_XZ_TOLERANCE}  (absolute)
Transition envelope    = p75 in-clip transition step
Object-type filter    : {args.object_type or '(none)'}
</pre>

<h2>Summary ({label})</h2>
<div class="stat-grid">

<div class="stat-box">
  <h3>Runtime Alignment</h3>
  <table>
    <tr><td>within_tolerance</td><td class="val">{runtime_loop_count}</td></tr>
    <tr><td>above_tolerance</td><td class="val">{len(results) - runtime_loop_count}</td></tr>
  </table>
</div>
<div class="stat-box">
  <h3>wrap_gap (max per-joint first-vs-last gap)</h3>
  <table>
    <tr><td>min</td><td class="val">{min(all_wrap_gap):.6f}</td></tr>
    <tr><td>p50</td><td class="val">{_pct(all_wrap_gap, 50):.6f}</td></tr>
    <tr><td>p90</td><td class="val">{_pct(all_wrap_gap, 90):.6f}</td></tr>
    <tr><td>p95</td><td class="val">{_pct(all_wrap_gap, 95):.6f}</td></tr>
    <tr><td>p99</td><td class="val">{_pct(all_wrap_gap, 99):.6f}</td></tr>
    <tr><td>max</td><td class="val">{max(all_wrap_gap):.6f}</td></tr>
  </table>
</div>
<div class="stat-box">
  <h3>root XZ displacement closure (threshold = {LOOP_DETECTION_ROOT_XZ_TOLERANCE})</h3>
  <table>
    <tr><td>closed</td><td class="val">{root_xz_closed_count}</td></tr>
    <tr><td>not_closed</td><td class="val">{len(results) - root_xz_closed_count}</td></tr>
    <tr><td>total_disp p50</td><td class="val">{_pct(all_root_xz_disp or [0], 50):.6f}</td></tr>
    <tr><td>total_disp max</td><td class="val">{max(all_root_xz_disp or [0]):.6f}</td></tr>
  </table>
</div>
</div>

<h2>{closest_label}</h2>
<table>
<thead>
<tr>
  <th>#</th>
  <th>name</th>
  <th>wrap_gap</th>
    <th>gap_margin</th>
    <th>xz_disp</th>
    <th>xz_ok</th>
  <th>frames</th>
    <th>is_loop</th>
</tr>
</thead>
<tbody>
{_closest_rows(closest)}
</tbody>
</table>

<p style="margin-top:32px; color:#888; font-size:0.8rem;">
  BVH links use <code>bvhview://open?url=...</code> protocol.<br>
  BVH root: {bvh_dir_abs}
</p>
</body>
</html>"""

    html_path.write_text(html, encoding="utf-8")
    return html_path


def main():
    parser = argparse.ArgumentParser(
        description="Compute loop unclosure error for is_loop motions"
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Override dataset root directory"
    )
    parser.add_argument(
        "--motion-dir", type=str, default=None,
        help="Override motion directory"
    )
    parser.add_argument(
        "--object-type", type=str, default=None,
        help="Filter to a specific object type (e.g. Buffalo)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for HTML report (default: Anytop/outputs/compute_loop_unclosure_error)"
    )
    parser.add_argument(
        "--bvh-dir", type=str, default=None,
        help="Directory containing BVH files for hyperlinks (default: dataset/truebones/zoo/truebones_processed/bvhs)"
    )
    args = parser.parse_args()

    # Determine paths
    from data_loaders.truebones.truebones_utils.get_opt import get_opt
    opt = get_opt(None)
    data_root = args.data_root or opt.data_root
    motion_dir = args.motion_dir or opt.motion_dir

    # Load motions
    all_motions = load_motions(data_root)

    # Filter by object type if specified
    if args.object_type:
        all_motions = {
            name: meta for name, meta in all_motions.items()
            if name.startswith(f"{args.object_type}_")
        }

    summary_label = "motions"

    # Compute errors
    results = []
    for name, meta in sorted(all_motions.items()):
        motion_path = Path(motion_dir) / name
        if not motion_path.exists():
            continue

        translation_root_index = int(meta.get("translation_root_index", 0))
        err = compute_unclosure_error(str(motion_path), translation_root_index=translation_root_index)
        if "error" in err:
            continue

        err["name"] = Path(name).stem
        err["object_type"] = meta.get("object_type", "?")
        err["metadata_is_loop"] = bool(meta.get("is_loop", False))
        results.append(err)

    # ── Summary stats (computed before filtering) ──
    all_wrap_gap = [r["wrap_gap"] for r in results]
    runtime_loop_count = sum(1 for r in results if r["runtime_is_loop"])
    root_xz_closed_count = sum(1 for r in results if r.get("root_xz_is_closed", True))
    all_root_xz_disp = [r["root_xz_total_disp"] for r in results]

    print(f"Processed {len(results)} motions  |  runtime_loop={runtime_loop_count}  not_loop={len(results) - runtime_loop_count}")

    # ── Print is_loop mismatches ──
    mismatches = [r for r in results if r["runtime_is_loop"] != r["metadata_is_loop"]]
    if mismatches:
        print(f"\n*** {len(mismatches)} is_loop mismatch(es) (runtime vs metadata): ***")
        for r in sorted(mismatches, key=lambda x: x["name"]):
            print(f"  {r['name']}:  runtime={r['runtime_is_loop']}  metadata={r['metadata_is_loop']}")

    # ── Sort: is_loop=False first, then wrap_gap descending within each group ──
    if len(results) > 0:
        sorted_results = sorted(results, key=lambda r: (r["runtime_is_loop"], -r["wrap_gap"]))
        false_count = sum(1 for r in sorted_results if not r["runtime_is_loop"])
        true_count = len(sorted_results) - false_count
        closest_label = f"all {len(sorted_results)} motions (is_loop=False: {false_count}, True: {true_count})"
    else:
        sorted_results = []
        closest_label = "(no motions)"

    # ── HTML report ──
    output_dir = Path(args.output_dir) if args.output_dir else _ANYTOP_DIR / "outputs" / "compute_loop_unclosure_error"
    # Resolve BVH dir to absolute path for file:/// links
    _bvh_dir_rel = args.bvh_dir if args.bvh_dir else "../../dataset/truebones/zoo/truebones_processed/bvhs"
    bvh_dir_abs = (output_dir / _bvh_dir_rel).resolve()
    write_html_report(
        results=results,
        all_wrap_gap=all_wrap_gap,
        runtime_loop_count=runtime_loop_count,
        label=summary_label,
        output_dir=output_dir,
        bvh_dir_abs=bvh_dir_abs,
        args=args,
        closest=sorted_results,
        closest_label=closest_label,
        root_xz_closed_count=root_xz_closed_count,
        all_root_xz_disp=all_root_xz_disp,
    )

    print(f"HTML report: {output_dir / 'loop_unclosure_report.html'}")


if __name__ == "__main__":
    main()
