"""
convert_fbx_2_glb.py

Convert every FBX file under the Truebones raw dataset to GLB using
Blender (bpy).  Each FBX is loaded via ``FBX.load`` (which imports it into
the Blender scene and extracts animation data) and then re-exported as a
same-named GLB next to the source FBX.

Incremental by default — only FBX files whose ``.glb`` output does **not**
yet exist are processed.  Combine ``--overwrite`` to force a full reconversion
or ``--filter`` to restrict to specific character types.

Requires bpy (Blender as a Python module) — run with the project's .venv::

    .venv/Scripts/python.exe Anytop/tools/dataset_cleanup/convert_fbx_2_glb.py [options]

Usage::

    # Convert all characters (incremental)
    python Anytop/tools/dataset_cleanup/convert_fbx_2_glb.py

    # Convert only Buffalo and Dragon, overwriting existing GLBs
    python Anytop/tools/dataset_cleanup/convert_fbx_2_glb.py --filter Buffalo,Dragon --overwrite

    # Use a custom dataset directory
    python Anytop/tools/dataset_cleanup/convert_fbx_2_glb.py --dataset-dir /path/to/fbx_root
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


def _list_fbx_files(directory: str) -> list[str]:
    """Return sorted absolute paths of all ``.fbx`` files in *directory*."""
    return sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.lower().endswith(".fbx")
    )


def _get_object_dirs(dataset_dir: str) -> list[tuple[str, str]]:
    """Return ``(object_type, dir_path)`` for each character subdirectory."""
    entries: list[tuple[str, str]] = []
    for name in sorted(os.listdir(dataset_dir)):
        dir_path = os.path.join(dataset_dir, name)
        if os.path.isdir(dir_path) and not name.startswith("."):
            entries.append((name, dir_path))
    return entries


def _confirm_yes_no(prompt: str) -> bool:
    while True:
        response = input(prompt).strip().lower()
        if response in ("yes", "y"):
            return True
        if response in ("no", "n"):
            return False
        print("Invalid response. Please enter 'yes', 'y', 'no', or 'n'.")


# ---------------------------------------------------------------------------
# Per-worker entry point  (must be module-level for pickle / ProcessPool)
# ---------------------------------------------------------------------------


def _convert_one_fbx(
    object_type: str,
    fbx_path: str,
    glb_path: str,
) -> tuple[str, bool]:
    """Import *fbx_path* into Blender and export as GLB.

    Returns ``(rel_path, success)`` where *rel_path* is
    ``"ObjectType/filename.fbx"`` for logging.
    """
    import bpy
    from motion_lib.FBX import _silence_os_std, clear_scene, import_fbx

    rel = f"{object_type}/{os.path.basename(fbx_path)}"
    try:
        clear_scene()
        import_fbx(fbx_path, use_image_search=True)

        # Wire up alpha / diffuse textures that bpy's FBX importer may have
        # skipped — without this, alpha-channel textures are missing in the GLB
        # output (matching AnimationExporter.export_glb behaviour).
        armature = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
        if armature is not None:
            from utils.texture_resolve import resolve_main_character_textures
            with contextlib.redirect_stdout(io.StringIO()):
                resolve_main_character_textures(bpy, armature, fbx_path)

        # Bake animation at 30 fps rather than Blender's 24 fps scene default.
        scene = bpy.context.scene
        scene.render.fps = 30
        scene.render.fps_base = 1.0

        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()), \
             _silence_os_std():
            bpy.ops.export_scene.gltf(
                filepath=glb_path,
                export_format="GLB"
            )
        clear_scene()
        return rel, True
    except Exception as exc:
        print(f"  [FAIL] {rel}: {exc}")
        try:
            clear_scene()
        except Exception:
            pass
        return rel, False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_fbx_2_glb(
    dataset_dir: str | None = None,
    only_objects: set[str] | None = None,
    overwrite: bool = False,
    yes: bool = False,
    workers: int = 16,
) -> int:
    """Convert all FBX files under *dataset_dir* to same-named GLB files.

    Args:
        dataset_dir: Root directory containing per-character subfolders with
            FBX files.  Defaults to ``get_raw_data_dir()`` (Truebone_Z-OO).
        only_objects: Optional set of object-type names to restrict conversion
            to (e.g. ``{"Buffalo", "Horse"}``).
        overwrite: When ``True``, re-export even if the GLB already exists.
            When ``False`` (default), skip FBX files whose GLB output exists
            (incremental mode).
        yes: When ``True``, skip all interactive confirmation prompts.
        workers: Number of parallel Blender processes.  Each worker runs its
            own independent Blender instance.  Default 16.

    Returns:
        Number of GLB files written.
    """
    from data_loaders.truebones.truebones_utils.param_utils import get_raw_data_dir

    dataset_dir_resolved = Path(get_raw_data_dir(dataset_dir)).resolve()
    object_dirs = _get_object_dirs(str(dataset_dir_resolved))

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
            print("[ERROR] no matching object types found; nothing to convert.")
            return 0

    # Collect all FBX files upfront for a summary prompt.
    all_fbx: list[tuple[str, str, str]] = []  # (object_type, fbx_path, glb_path)
    for object_type, dir_path in object_dirs:
        for fbx_path in _list_fbx_files(dir_path):
            stem = os.path.splitext(os.path.basename(fbx_path))[0]
            glb_path = os.path.join(dir_path, f"{stem}.glb")
            all_fbx.append((object_type, fbx_path, glb_path))

    if not all_fbx:
        print(f"[WARN] no FBX files found under {dataset_dir_resolved}")
        return 0

    # Separate into pending vs skipped.
    pending = [(o, f, g) for o, f, g in all_fbx if overwrite or not os.path.exists(g)]
    skipped = len(all_fbx) - len(pending)

    if not pending:
        print(f"[OK] all {len(all_fbx)} FBX file(s) already have a corresponding GLB "
              f"(use --overwrite to reconvert).")
        return 0

    print(f"Dataset : {dataset_dir_resolved}")
    print(f"Total   : {len(all_fbx)} FBX file(s) across {len(object_dirs)} character(s)")
    print(f"Pending : {len(pending)}  (--overwrite={overwrite}, --workers={workers})")
    if skipped:
        print(f"Skipped : {skipped} existing GLB(s)")

    # Prompt on destructive --overwrite; incremental is non-destructive, skip prompt.
    if overwrite and not yes:
        print(f"\nThis will overwrite {len(pending)} existing GLB file(s).")
        if not _confirm_yes_no("Proceed? [y/N] "):
            print("[ABORT]")
            return 0

    # --- Parallel conversion ---
    written = 0
    n_workers = min(workers, len(pending))
    print(f"\nStarting conversion with {n_workers} worker(s) ...\n")

    total = len(pending)
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        fut_to_item = {
            executor.submit(_convert_one_fbx, o, f, g): (o, f, g)
            for o, f, g in pending
        }
        for fut in as_completed(fut_to_item):
            o, f, g = fut_to_item[fut]
            rel = f"{o}/{os.path.basename(f)}"
            done += 1
            try:
                _rel, ok = fut.result()
                if ok:
                    written += 1
                    print(f"[{done}/{total}] [OK]     {rel} -> {os.path.basename(g)}")
                else:
                    print(f"[{done}/{total}] [FAIL]   {rel}")
            except Exception as exc:
                print(f"[{done}/{total}] [FAIL]   {rel}: {exc}")

    print(f"\n[PASS] converted {written} / {total} FBX file(s) to GLB.")
    if skipped:
        print(f"       ({skipped} existing GLB(s) skipped)")
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert all FBX files under the Truebones raw dataset to GLB "
            "using Blender (bpy).  Incremental by default."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset-dir",
        default="",
        type=str,
        help=(
            "Root directory containing per-character subfolders with FBX files. "
            "Defaults to get_raw_data_dir() (Truebone_Z-OO)."
        ),
    )
    parser.add_argument(
        "--filter",
        default="",
        type=str,
        help=(
            "Comma/semicolon-separated object-type names to convert "
            "(e.g. 'Buffalo,Horse').  Default: all characters."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-export GLB even if the output file already exists (force reconversion).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        default=False,
        help="Skip all interactive confirmation prompts (headless / automated mode).",
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
        convert_fbx_2_glb(
            dataset_dir=args.dataset_dir or None,
            only_objects=only_objects,
            overwrite=args.overwrite,
            yes=args.yes,
            workers=args.workers,
        )
        return 0
    except Exception as exc:
        print(f"ERROR: FBX-to-GLB conversion failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
