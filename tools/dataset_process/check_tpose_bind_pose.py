"""
check_tpose_bind_pose.py

Scan the Truebone_Z-OO directory tree for all *tpose*.glb files and verify
that the first animation frame matches the bind/rest pose.  Print any files
where they differ.

Usage:
    cd Anytop
    python tools/dataset_process/check_tpose_bind_pose.py

Requires bpy (Blender as a Python module) in the .venv.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
from pathlib import Path

_ANYTOP_ROOT = Path(__file__).resolve().parent.parent.parent
ROOT_DIR = str(_ANYTOP_ROOT / "dataset" / "truebones" / "zoo" / "Truebone_Z-OO")
TOLERANCE = 3e-3


# ── Minimal bpy wrappers (inlined from motion_lib.FBX) ──────────────────────

def _clear_scene(bpy) -> None:
    """Reset Blender to a fresh empty scene."""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _import_gltf(bpy, filepath: str) -> None:
    """Import a GLB/GLTF file into the current Blender scene."""
    bpy.ops.import_scene.gltf(filepath=filepath)


def _remove_lights_and_cameras(bpy) -> None:
    """Remove all LIGHT and CAMERA objects from the current scene."""
    for obj in list(bpy.data.objects):
        if obj.type in {"LIGHT", "CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)


@contextlib.contextmanager
def _silence_os_std():
    """Context manager that redirects OS-level fd 1 & 2 to NUL."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(devnull_fd)


# ── Core logic ──────────────────────────────────────────────────────────────

def _find_armature(bpy, path: str):
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError(f"No armature found in GLB: {path}")
    return max(armatures, key=lambda obj: len(obj.data.bones))


def matrices_equal(m1, m2, tol: float = TOLERANCE) -> bool:
    """Compare two 4×4 matrices element-wise with tolerance."""
    for i in range(4):
        for j in range(4):
            if not abs(float(m1[i][j]) - float(m2[i][j])) < tol:
                return False
    return True


def check_file(filepath: str, verbose: bool = False) -> list[str]:
    """Return list of bone names whose first-frame pose differs from rest pose.

    If all bones match, returns an empty list.
    """
    import bpy  # noqa: PLC0415  -- bpy replaces sys.path, import late

    _clear_scene(bpy)
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()), \
         _silence_os_std():
        _import_gltf(bpy, filepath)
    _remove_lights_and_cameras(bpy)

    armature = _find_armature(bpy, filepath)
    scene = bpy.context.scene

    # Determine the first animation frame.
    action = armature.animation_data.action if armature.animation_data else None
    if action and hasattr(action, "frame_start"):
        first_frame = int(action.frame_start)
    else:
        first_frame = int(scene.frame_start)
    scene.frame_set(first_frame)
    bpy.context.view_layer.update()

    mismatches: list[str] = []
    for pose_bone in armature.pose.bones:
        bone_name = pose_bone.name
        pose_mat = pose_bone.matrix
        rest_mat = armature.data.bones[bone_name].matrix_local

        if not matrices_equal(pose_mat, rest_mat):
            mismatches.append(bone_name)
            if verbose:
                print(f"    Bone '{bone_name}' differs")

    return mismatches


def main():
    tpose_glbs: list[str] = []
    for root, _dirs, files in os.walk(ROOT_DIR):
        for name in files:
            if not name.lower().endswith(".glb"):
                continue
            stem = Path(name).stem.lower()
            compact = stem.replace("-", "").replace("_", "").replace(" ", "")
            if "tpose" in compact:
                tpose_glbs.append(os.path.join(root, name))

    tpose_glbs.sort()
    print(f"Found {len(tpose_glbs)} *tpose*.glb files under {ROOT_DIR}\n")

    passed = 0
    failed = 0

    for fpath in tpose_glbs:
        rel = os.path.relpath(fpath, ROOT_DIR)
        sys.stdout.write(f"  Checking {rel} ... ")
        sys.stdout.flush()

        try:
            mismatches = check_file(fpath, verbose=False)
        except Exception as exc:
            print(f"ERROR: {exc}")
            failed += 1
            continue

        if not mismatches:
            print("OK")
            passed += 1
        else:
            print(f"MISMATCH ({len(mismatches)} bones)")
            for bname in mismatches:
                print(f"    - {bname}")
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"  Passed: {passed}, Failed: {failed}, Total: {len(tpose_glbs)}")


if __name__ == "__main__":
    main()
