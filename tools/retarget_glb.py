"""
Native-space GLB -> GLB motion retargeting.

Plays a motion from one rig on another rig, staying in the files' own
coordinate space the whole way: no HML feature encode/decode, no
``process_anim`` recentering or rescaling, and no root-XZ stripping. Global
locomotion therefore survives, and a self-retarget (same file as source and
target) round-trips to float noise.

For the feature-space path -- "what are this motion's AnyTop ``(F, J, 12)``
features on the target skeleton" -- use ``python tools/retarget_npy.py`` instead.

Usage examples:

    # Play Buffalo's walk on the Horse rig
    python tools/retarget_glb.py \\
        --source dataset/.../Buffalo-WalkLoop.glb \\
        --target dataset/.../HorseALL-TPOSE.glb \\
        --output outputs/horse_walk.glb

    # Rigs authored in different bases: sweep the 12 rigid candidates
    python tools/retarget_glb.py --source a.glb --target b.glb \\
        --output out.glb --coordinate-search on

    # Differently proportioned legs
    python tools/retarget_glb.py --source a.glb --target b.glb \\
        --output out.glb --ground

    # Skeleton-only output, first 120 frames, forced 24 fps
    python tools/retarget_glb.py --source a.glb --target b.glb \\
        --output out.glb --skeleton-only --frames 0:120 --fps 24

Requires bpy (Blender as a Python module) in the current environment.
"""

import argparse
import os
import sys

# ── Path setup ────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANYTOP_DIR = os.path.realpath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.dirname(ANYTOP_DIR)

for _p in [REPO_ROOT, ANYTOP_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


_SUPPORTED_SUFFIXES = {".glb", ".gltf", ".fbx"}


def _parse_frames(spec: str):
    """Parse a ``START:END`` frame slice into ``[start, end]``."""
    if not spec:
        return None
    if ":" not in spec:
        raise argparse.ArgumentTypeError(
            f"--frames expects START:END (either side may be empty), got {spec!r}"
        )
    raw_start, raw_end = spec.split(":", 1)
    start = int(raw_start) if raw_start.strip() else 0
    end = int(raw_end) if raw_end.strip() else None
    if end is not None and end <= start:
        raise argparse.ArgumentTypeError(
            f"--frames END must be greater than START, got {spec!r}"
        )
    return [start, end]


def _check_motion_file(parser: argparse.ArgumentParser, path: str, flag: str) -> str:
    resolved = os.path.abspath(path)
    if not os.path.isfile(resolved):
        parser.error(f"{flag} file not found: {resolved}")
    suffix = os.path.splitext(resolved)[1].lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        parser.error(
            f'{flag} has unsupported format "{suffix}". '
            f"Supported: {', '.join(sorted(_SUPPORTED_SUFFIXES))}"
        )
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retarget a motion from one rig onto another, entirely in native "
            "space (root motion preserved, no feature-space round trip)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source", required=True,
        help="Motion source (.glb/.gltf/.fbx). Only its skeleton and animation "
             "are read; its mesh is ignored.",
    )
    parser.add_argument(
        "--target", required=True,
        help="Rig that receives the motion (.glb/.gltf/.fbx). Supplies the "
             "skeleton and, unless --skeleton-only, the skin and materials. "
             "Any animation it carries is discarded.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Destination .glb path.",
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="Output frame rate. Default: the source file's own rate. The "
             "exporter writes scene fps as an int, so fractional rates truncate.",
    )
    parser.add_argument(
        "--coordinate-search", choices=("auto", "on", "off"), default="auto",
        help="Rigid 1-of-12 rest-pose alignment sweep. 'auto' (default) keeps "
             "the exporter's rule, which is off for a GLB target. Use 'on' when "
             "the two rigs are authored in different bases. A self-retarget is "
             "unaffected either way.",
    )
    parser.add_argument(
        "--ground", action="store_true",
        help="After the retarget (and --fullbody-ik), shift the target root by "
             "a constant Y so its two lowest contact joints sit at the target "
             "bind pose's contact height. Off by default: it is a real "
             "translation and breaks self-retarget "
             "idempotency. Useful when the two rigs differ in leg proportion.",
    )
    parser.add_argument(
        "--fullbody-ik", action="store_true",
        help="Re-solve the retargeted pose on the rigid target skeleton so the "
             "bone lengths the retarget stretched to reach the donor's "
             "proportions come back, with rotations carrying the motion instead. "
             "Off by default: it is a real change to the written pose and breaks "
             "self-retarget idempotency.",
    )
    parser.add_argument(
        "--ik-stretch-factor", type=float, default=None,
        help="Bone-length elasticity the IK rebuild may keep (0.1 = +/-10 %%). "
             "Default: the pipeline's own default. Only used with --fullbody-ik.",
    )
    parser.add_argument(
        "--skeleton-only", action="store_true",
        help="Write a skeleton-only GLB (no meshes).",
    )
    parser.add_argument(
        "--frames", type=_parse_frames, default=None, metavar="START:END",
        help="Slice the source animation, e.g. 0:120. Either side may be empty.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    source_path = _check_motion_file(parser, args.source, "--source")
    target_path = _check_motion_file(parser, args.target, "--target")

    output_path = os.path.abspath(args.output)
    if os.path.splitext(output_path)[1].lower() != ".glb":
        parser.error(f"--output must be a .glb path, got {output_path}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    coordinate_search = {"auto": None, "on": True, "off": False}[args.coordinate_search]

    from utils.retarget_pipeline import retarget_glb_to_glb

    retarget_glb_to_glb(
        source_path,
        target_path,
        output_path,
        fps=args.fps,
        coordinate_search=coordinate_search,
        ground=args.ground,
        export_mesh=not args.skeleton_only,
        fullbody_ik=args.fullbody_ik,
        fullbody_ik_stretch_factor=args.ik_stretch_factor,
        slice_inds=args.frames,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
