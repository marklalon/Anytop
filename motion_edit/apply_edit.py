"""Apply edit parameters to an Edit Package and export BVH / GLB.

Usage (from ``Anytop/``)::

    python -m motion_edit.apply_edit outputs/edit_packages/Horse_RunLoop.edit --bvh out.bvh
    python -m motion_edit.apply_edit outputs/edit_packages/Horse_RunLoop.edit --params params.json --glb out.glb
    python -m motion_edit.apply_edit outputs/edit_packages/Horse_RunLoop.edit --set tempo=1.2 --bvh out.bvh

``--params`` is a JSON object ``{"<parameter>": value}``; ``--set`` overrides
single values on top of it.  The export is the skeleton-only HML path of
``tools/restore_glb_from_npy.py`` and the generate-time BVH preview, run on
the runtime's Animation.
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
from motion_edit.runtime import EditResult, EditRuntime  # noqa: E402


def _restored(package: EditPackage, result: EditResult, names) -> "RestoredAnimation":
    from utils.npy_restore import RestoredAnimation
    from utils.roundtrip_common import build_skeleton

    skeleton = build_skeleton(
        [str(n) for n in names],
        np.asarray(package["skeleton_offsets"], dtype=np.float32),
        np.asarray(package["parents"], dtype=np.int32),
        np.asarray(package["skeleton_rest_rotations"], dtype=np.float32),
    )
    return RestoredAnimation(animation=result.animation, skeleton=skeleton,
                             translation_root_index=package.root, fps=result.fps,
                             has_animated_pos=True)


def export_bvh(package: EditPackage, result: EditResult, path: str) -> str:
    """BVH with the cond's anatomical joint names, like the generate-time preview."""
    from utils.npy_restore import export_animation_bvh

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    export_animation_bvh(_restored(package, result, package["bvh_names"]), path)
    return os.path.abspath(path)


def export_glb(package: EditPackage, result: EditResult, path: str) -> str:
    """Skeleton-only GLB in HML space, like ``restore_glb_from_npy`` without a mesh."""
    from utils.exporter import AnimationExporter, animation_to_exporter_inputs

    restored = _restored(package, result, package["names"])
    joint_rotations, root_translation, root_rotation, bone_translations = (
        animation_to_exporter_inputs(restored.animation, restored.skeleton))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    AnimationExporter(restored.skeleton, fps=restored.fps).export_glb(
        joint_rotations, root_translation, root_rotation, path,
        bone_translations=bone_translations, export_mesh=False,
        rename_bones_to_canonical=True, prune_unmapped_bones=True,
    )
    return os.path.abspath(path)


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
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--compose", action="store_true",
                        help="rebuild from the layers even at defaults (round-trip check)")
    parser.add_argument("--bvh", help="write a BVH here")
    parser.add_argument("--glb", help="write a skeleton-only GLB here")
    args = parser.parse_args(argv)
    if not (args.bvh or args.glb):
        parser.error("nothing to write: pass --bvh and/or --glb")

    params = {}
    if args.params:
        with open(args.params, "r", encoding="utf-8") as handle:
            params.update(json.load(handle))
    params.update(parse_set(args.set))

    package = EditPackage.load(args.package)
    result = EditRuntime(package).apply(params, compose=args.compose)
    for item in result.diagnostics:
        print(f"  [{item['kind']}] {item['message']}")
    if args.bvh:
        print(f"Wrote {export_bvh(package, result, args.bvh)}")
    if args.glb:
        print(f"Wrote {export_glb(package, result, args.glb)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
