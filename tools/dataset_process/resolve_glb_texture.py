"""
resolve_glb_texture.py

Re-resolve missing diffuse / alpha textures on existing GLB files and
re-export them when textures were actually applied.

For each GLB file, Blender imports it, calls
:func:`resolve_main_character_textures` to wire missing diffuse / alpha
textures, and — if the function applied any changes — re-exports the GLB
in-place.  If the character already has all textures (no-op), the GLB is left
untouched.

Incremental by default — only GLB files whose textures still need resolving
are processed.  Use ``--filter`` to restrict to specific character types.

Requires bpy (Blender as a Python module) — run with the project's .venv::

    .venv/Scripts/python.exe Anytop/tools/dataset_process/resolve_glb_texture.py [options]

Usage::

    # Resolve all characters under the default dataset directory (incremental)
    python Anytop/tools/dataset_process/resolve_glb_texture.py

    # Resolve only Buffalo and Dragon
    python Anytop/tools/dataset_process/resolve_glb_texture.py --filter Buffalo,Dragon

"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Put the Anytop package root on sys.path so data_loaders / motion_lib / utils
# imports resolve to the Anytop copies (NOT the top-level pcvg utils).
_ANYTOP_DIR = Path(__file__).resolve().parent.parent.parent
if str(_ANYTOP_DIR) not in sys.path:
    sys.path.insert(0, str(_ANYTOP_DIR))


def _list_glb_files(directory: str) -> list[str]:
    """Return sorted absolute paths of all ``.glb`` files in *directory*."""
    return sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.lower().endswith(".glb")
    )


def _get_object_dirs(dataset_dir: str) -> list[tuple[str, str]]:
    """Return ``(object_type, dir_path)`` for each character subdirectory."""
    entries: list[tuple[str, str]] = []
    for name in sorted(os.listdir(dataset_dir)):
        dir_path = os.path.join(dataset_dir, name)
        if os.path.isdir(dir_path) and not name.startswith("."):
            entries.append((name, dir_path))
    return entries


# ---------------------------------------------------------------------------
# Per-worker entry point  (must be module-level for pickle / ProcessPool)
# ---------------------------------------------------------------------------


def _resolve_one_glb(
    object_type: str,
    glb_path: str,
) -> tuple[str, str]:
    """Import *glb_path*, resolve textures, re-export if changes were made.

    Returns ``(rel_path, status)`` where *status* is one of:
        - ``"resolved"``  — textures were applied and GLB was overwritten
        - ``"already_ok"`` — no missing textures found, GLB left untouched
        - ``"failed"``    — an exception occurred
    """
    import bpy
    from motion_lib.FBX import _silence_os_std, clear_scene, import_gltf
    from utils.texture_resolve import resolve_main_character_textures

    rel = f"{object_type}/{os.path.basename(glb_path)}"
    try:
        clear_scene()
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()), \
             _silence_os_std():
            import_gltf(glb_path)

        # Find the armature.
        armature = next(
            (o for o in bpy.data.objects if o.type == "ARMATURE"), None
        )
        if armature is None:
            print(f"  [SKIP] {rel}: no armature found in GLB")
            clear_scene()
            return rel, "already_ok"

        has_changes = resolve_main_character_textures(bpy, armature, glb_path)

        if not has_changes:
            # No missing textures — nothing to do.
            clear_scene()
            return rel, "already_ok"

        # Re-export GLB, overwriting the original.  Bake animation at 30 fps
        # rather than Blender's 24 fps scene default.
        scene = bpy.context.scene
        scene.render.fps = 30
        scene.render.fps_base = 1.0

        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()), \
             _silence_os_std():
            bpy.ops.export_scene.gltf(
                filepath=glb_path,
                export_format="GLB",
            )
        clear_scene()
        return rel, "resolved"

    except Exception as exc:
        print(f"  [FAIL] {rel}: {exc}")
        try:
            clear_scene()
        except Exception:
            pass
        return rel, "failed"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_glb_texture(
    dataset_dir: str | None = None,
    only_objects: set[str] | None = None,
    workers: int = 16,
) -> int:
    """Resolve missing textures on GLB files and re-export when changed.

    Only GLB files whose textures were actually missing and applied are
    re-exported.  If a GLB already has all diffuse/alpha textures, it is
    left untouched.

    Args:
        dataset_dir: Root directory containing per-character subfolders with
            GLB files.  Defaults to ``get_raw_data_dir()`` (Truebone_Z-OO).
        only_objects: Optional set of object-type names to restrict conversion
            to (e.g. ``{"Buffalo", "Horse"}``).
        workers: Number of parallel Blender processes.  Each worker runs its
            own independent Blender instance.  Default 16.

    Returns:
        Number of GLB files that were actually overwritten (status
        ``"resolved"``).
    """
    from data_loaders.truebones.truebones_utils.param_utils import get_raw_data_dir

    # --- Collect GLB paths ---
    all_glbs: list[tuple[str, str]] = []  # (object_type, glb_path)
    dataset_dir_resolved = str(Path(get_raw_data_dir(dataset_dir)).resolve())
    object_dirs = _get_object_dirs(dataset_dir_resolved)

    if not object_dirs:
        print(f"[WARN] no character subdirectories found in {dataset_dir_resolved}")
        return 0

    # Apply --filter.
    if only_objects is not None:
        filtered = [(n, p) for n, p in object_dirs if n in only_objects]
        missing = sorted(only_objects - {n for n, _ in filtered})
        if missing:
            print(f"[WARN] --filter names not found in dataset: {', '.join(missing)}")
        object_dirs = filtered
        if not object_dirs:
            print("[ERROR] no matching object types found; nothing to process.")
            return 0

    for object_type, dir_path in object_dirs:
        for glb_path in _list_glb_files(dir_path):
            all_glbs.append((object_type, glb_path))

    if not all_glbs:
        print("[WARN] no GLB files found.")
        return 0

    # --- Prompt ---
    print(f"Dataset   : {dataset_dir_resolved}")
    print(f"GLB files : {len(all_glbs)}")
    print(f"--workers={workers}")

    # --- Parallel processing ---
    resolved_count = 0
    already_ok_count = 0
    failed_count = 0
    n_workers = min(workers, len(all_glbs))
    print(f"\nStarting with {n_workers} worker(s) ...\n")

    total = len(all_glbs)
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        fut_to_item = {
            executor.submit(_resolve_one_glb, o, p): (o, p)
            for o, p in all_glbs
        }
        for fut in as_completed(fut_to_item):
            o, p = fut_to_item[fut]
            rel = f"{o}/{os.path.basename(p)}"
            done += 1
            try:
                _rel, status = fut.result()
                if status == "resolved":
                    resolved_count += 1
                    print(f"[{done}/{total}] [OK]  {rel}  -> textures applied, re-exported")
                elif status == "already_ok":
                    already_ok_count += 1
                    print(f"[{done}/{total}] [--]  {rel}  -> textures OK, skipped")
                else:
                    failed_count += 1
                    print(f"[{done}/{total}] [FAIL]{rel}")
            except Exception as exc:
                failed_count += 1
                print(f"[{done}/{total}] [FAIL]{rel}: {exc}")

    print(f"\n[SUMMARY]")
    print(f"  Resolved & re-exported : {resolved_count}")
    print(f"  Already OK (skipped)   : {already_ok_count}")
    if failed_count:
        print(f"  Failed                 : {failed_count}")
    print(f"  Total                  : {total}")
    return resolved_count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve missing textures on GLB files and re-export when changed. "
            "If a GLB already has all diffuse/alpha textures, it is left untouched."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset-dir",
        default="",
        type=str,
        help=(
            "Root directory containing per-character subfolders with GLB files. "
            "Defaults to get_raw_data_dir() (Truebone_Z-OO)."
        ),
    )
    parser.add_argument(
        "--filter",
        default="",
        type=str,
        help=(
            "Comma/semicolon-separated object-type names to process "
            "(e.g. 'Buffalo,Horse').  Default: all characters. "
            "Only used when scanning a dataset directory."
        ),
    )
    parser.add_argument(
        "--workers", "-j",
        default=16,
        type=int,
        help="Number of parallel Blender worker processes.  Default 16.",
    )
    args = parser.parse_args()

    only_objects: set[str] | None = {
        token.strip()
        for token in args.filter.replace(";", ",").split(",")
        if token.strip()
    } or None

    try:
        resolve_glb_texture(
            dataset_dir=args.dataset_dir or None,
            only_objects=only_objects,
            workers=args.workers,
        )
        return 0
    except Exception as exc:
        print(f"ERROR: texture resolution failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
