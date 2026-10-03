"""Apply edit parameters to an Edit Package and export BVH / GLB.

Usage (from ``Anytop/``)::

    python -m motion_edit.apply_edit outputs/edit_packages/Horse_RunLoop.edit --bvh out.bvh
    python -m motion_edit.apply_edit outputs/edit_packages/Horse_RunLoop.edit --params params.json --glb out.glb
    python -m motion_edit.apply_edit outputs/edit_packages/Horse_RunLoop.edit --set tempo=1.2 --bvh out.bvh
    python -m motion_edit.apply_edit outputs/edit_packages/Horse_RunLoop.edit --sidecar exports/run.glb.json --root_motion --glb out.glb

``--params`` is a JSON object ``{"<parameter>": value}``.  ``--sidecar`` is
the file the tuning UI writes next to an export: a JSON object with
``params`` and ``events`` (the moved strike events).  ``--set`` overrides
single values on top of either.  ``--mesh`` skins the GLB on the T-pose mesh
the package was decomposed with (in the mesh's native space; see
``motion_edit.mesh``).  ``--root_motion`` moves the root
along the result's ground velocity instead of keeping the clip in place.  The
export is the skeleton-only HML path of ``tools/restore_glb_from_npy.py`` and
the generate-time BVH preview, run on the runtime's Animation.  The tuning UI
exports by running this command on its sidecar, so the same parameters give
the same file.  All-default parameters export the package's own decode (the
runtime's replay shortcut) unless ``--compose``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ANYTOP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ANYTOP_ROOT not in sys.path:
    sys.path.insert(0, ANYTOP_ROOT)

import numpy as np  # noqa: E402

from motion_edit.package import EditPackage  # noqa: E402
from motion_edit.runtime import EditResult, EditRuntime, ground_displacement  # noqa: E402

EXPORT_FORMATS = ("bvh", "glb")


def _restored(package: EditPackage, result: EditResult, names, animation=None) -> "RestoredAnimation":
    from utils.npy_restore import RestoredAnimation
    from utils.roundtrip_common import build_skeleton

    skeleton = build_skeleton(
        [str(n) for n in names],
        np.asarray(package["skeleton_offsets"], dtype=np.float32),
        np.asarray(package["parents"], dtype=np.int32),
        np.asarray(package["skeleton_rest_rotations"], dtype=np.float32),
    )
    return RestoredAnimation(animation=result.animation if animation is None else animation,
                             skeleton=skeleton, translation_root_index=package.root, fps=result.fps,
                             has_animated_pos=True)


def root_motion_animation(package: EditPackage, result: EditResult):
    """The result's Animation travelling along ``v_g'``: the ground displacement is
    added to the top-level joints' XZ, as the UI's root motion view draws it."""
    animation = result.animation.copy()
    displacement = ground_displacement(result.ground_velocity, result.fps, periodic=False)
    for joint in np.flatnonzero(np.asarray(package["parents"]) < 0):
        # a top-level joint's local translation is its world translation
        animation.positions[:, joint, 0] += displacement[:, 0]
        animation.positions[:, joint, 2] += displacement[:, 1]
    return animation


def export_bvh(package: EditPackage, result: EditResult, path: str, animation=None) -> str:
    """BVH with the cond's anatomical joint names, like the generate-time preview."""
    from utils.npy_restore import export_animation_bvh

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    export_animation_bvh(_restored(package, result, package["bvh_names"], animation), path)
    return os.path.abspath(path)


def export_glb(package: EditPackage, result: EditResult, path: str, animation=None) -> str:
    """Skeleton-only GLB in HML space, like ``restore_glb_from_npy`` without a mesh."""
    from utils.exporter import AnimationExporter, animation_to_exporter_inputs

    restored = _restored(package, result, package["names"], animation)
    joint_rotations, root_translation, root_rotation, bone_translations = (
        animation_to_exporter_inputs(restored.animation, restored.skeleton))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    AnimationExporter(restored.skeleton, fps=restored.fps).export_glb(
        joint_rotations, root_translation, root_rotation, path,
        bone_translations=bone_translations, export_mesh=False,
        rename_bones_to_canonical=True, prune_unmapped_bones=True,
    )
    return os.path.abspath(path)


def export_result(package: EditPackage, result: EditResult, path: str, *, fmt: str,
                  root_motion: bool = False, tpose_mesh: str | None = None) -> str:
    """``tpose_mesh`` skins a GLB on that mesh (``motion_edit.mesh``); a BVH has no mesh.

    The GLB path runs bpy, which only works on a process's main thread: callers on
    other threads, like the tuning UI's request handlers, run this module as a command."""
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"export format must be one of {EXPORT_FORMATS}, got {fmt!r}")
    animation = root_motion_animation(package, result) if root_motion else None
    if fmt == "glb" and tpose_mesh:
        from motion_edit.mesh import export_skinned_glb

        return export_skinned_glb(package, result.animation if animation is None else animation,
                                  result.fps, path, tpose_mesh)
    return (export_bvh if fmt == "bvh" else export_glb)(package, result, path, animation)


def read_sidecar(path: str) -> tuple[dict, dict | None]:
    """``(params, events)`` of an export sidecar file."""
    with open(path, "r", encoding="utf-8") as handle:
        sidecar = json.load(handle)
    if not isinstance(sidecar, dict) or not isinstance(sidecar.get("params", {}), dict):
        raise ValueError(f"{path}: a sidecar is a JSON object with 'params' and 'events'")
    return dict(sidecar.get("params") or {}), sidecar.get("events") or None


def parse_set(values: list[str]) -> dict:
    out = {}
    for item in values:
        name, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"--set wants name=value, got {item!r}")
        lowered = raw.strip().lower()
        out[name.strip()] = (lowered in ("1", "true", "on", "yes")) if lowered in (
            "true", "false", "on", "off", "yes", "no") else float(raw)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("package", help="<clip>.edit directory")
    parser.add_argument("--params", help="JSON file of parameter values")
    parser.add_argument("--sidecar", help="UI export sidecar JSON (params + strike events)")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--compose", action="store_true",
                        help="rebuild from the layers even at defaults (round-trip check)")
    parser.add_argument("--root_motion", action="store_true",
                        help="travel along the ground velocity instead of staying in place")
    parser.add_argument("--mesh", action="store_true",
                        help="skin the GLB on the T-pose mesh the package was decomposed with")
    parser.add_argument("--bvh", help="write a BVH here")
    parser.add_argument("--glb", help="write a GLB here (skeleton only, or skinned with --mesh)")
    args = parser.parse_args(argv)
    if not (args.bvh or args.glb):
        parser.error("nothing to write: pass --bvh and/or --glb")
    if args.mesh and not args.glb:
        parser.error("--mesh skins the GLB: pass --glb")
    if args.params and args.sidecar:
        parser.error("pass --params or --sidecar, not both")

    params, events = {}, None
    if args.params:
        with open(args.params, "r", encoding="utf-8") as handle:
            params.update(json.load(handle))
    if args.sidecar:
        params, events = read_sidecar(args.sidecar)
    params.update(parse_set(args.set))

    package = EditPackage.load(args.package)
    tpose_mesh = None
    if args.mesh:
        from motion_edit.mesh import mesh_source

        tpose_mesh = mesh_source(args.package)
        if tpose_mesh is None:
            parser.error(f"--mesh: {args.package} was decomposed without --tpose_mesh")
        if not os.path.isfile(tpose_mesh):
            parser.error(f"--mesh: the package's T-pose mesh {tpose_mesh} is gone")
    result = EditRuntime(package).apply(params, compose=args.compose, events=events)
    for item in result.diagnostics:
        print(f"  [{item['kind']}] {item['message']}")
    for fmt in EXPORT_FORMATS:
        path = getattr(args, fmt)
        if path:
            print(f"Wrote {export_result(package, result, path, fmt=fmt, root_motion=args.root_motion, tpose_mesh=tpose_mesh)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
