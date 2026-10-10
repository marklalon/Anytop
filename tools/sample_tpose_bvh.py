#!/usr/bin/env python3
"""
Dump the per-character canonical t-pose stored in cond.npy to single-frame BVH.

cond.npy is a dict keyed by object type; each entry carries the canonical
skeleton's rest-pose local ``offsets`` (J, 3), ``parents`` (J,), and
``joints_names`` (or ``canonical_bvh_joint_names``). The rest pose with
identity joint rotations *is* the t-pose,
so we emit one single-frame BVH per character (no source motion required).

NOTE — frame convention: these ``offsets`` are the processed bind pose stored in
``cond.npy``. Preprocessing has already applied the character orientation and
dataset scale before deriving them, so the dumped t-pose is in the dataset's
canonical training frame and units rather than the native FBX authoring frame.
The writer is ``truebones_utils.tpose_bvh``, shared with the review server's
parts page.

Output layout (next to cond.npy):
    <dataset>/cond.npy
    <dataset>/bvh_tpose/<object_type>.bvh

Usage:
    python tools/sample_tpose_bvh.py [--dataset-dir PATH] [--filter NAME[,NAME...]]

Options:
    --dataset-dir PATH   Path to dataset directory (uses default if not specified).
    --filter NAMES       Comma/semicolon-separated object names to export
                         (default: every object in cond.npy).
"""

import argparse
import sys
from pathlib import Path

ANYTOP_DIR = Path(__file__).resolve().parent.parent
_PARENT_DIR = ANYTOP_DIR.parent
sys.path.insert(0, str(_PARENT_DIR))
sys.path.insert(0, str(ANYTOP_DIR))

from data_loaders.truebones.truebones_utils.cond_schema import load_cond
from data_loaders.truebones.truebones_utils.dataset_sources import (
    build_species_file_tokens,
    resolve_species_key,
)
from data_loaders.truebones.truebones_utils.param_utils import get_dataset_dir  # noqa: E402
from data_loaders.truebones.truebones_utils.tpose_bvh import write_tpose_bvh  # noqa: E402


def sample_tpose_bvh(
    dataset_dir: str | Path | None = None,
    only_objects: set[str] | None = None,
) -> list[Path]:
    """Write one t-pose BVH per requested character; return the written paths."""
    dataset_dir_path = Path(get_dataset_dir(str(dataset_dir) if dataset_dir else None)).resolve()
    cond_path = dataset_dir_path / "cond.npy"
    if not cond_path.exists():
        raise RuntimeError(f"cond.npy not found at {cond_path}")

    cond = load_cond(cond_path)
    file_tokens = build_species_file_tokens(cond)

    object_types = sorted(cond.keys())
    if only_objects is not None:
        # --filter takes user-facing names (bare, suffixed, or canonical).
        requested = {}
        for name in only_objects:
            key = resolve_species_key(cond, name)
            if key is None:
                print(f"[WARN] --filter name not in cond.npy, ignored: {name}")
            else:
                requested[key] = name
        object_types = [obj for obj in object_types if obj in requested]
        if not object_types:
            raise RuntimeError("no requested objects found in cond.npy")

    out_dir = dataset_dir_path / "bvh_tpose"
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for object_type in object_types:
        out_path = out_dir / f"{file_tokens[object_type]}.bvh"
        order = write_tpose_bvh(cond[object_type], out_path)
        print(f"[OK] {object_type}: {len(order)} joints -> {out_path}")
        written.append(out_path)

    print(f"\n[PASS] wrote {len(written)} t-pose BVH file(s) to {out_dir}")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dump per-character t-pose from cond.npy to BVH",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset-dir",
        default="",
        type=str,
        help="Path to dataset directory. If not specified, uses default path.",
    )
    parser.add_argument(
        "--filter",
        default="",
        type=str,
        help="Comma/semicolon-separated object names to export (default: all).",
    )
    args = parser.parse_args()

    only_objects = {
        token.strip()
        for token in args.filter.replace(";", ",").split(",")
        if token.strip()
    } or None

    try:
        sample_tpose_bvh(args.dataset_dir, only_objects=only_objects)
        return 0
    except Exception as exc:
        print(f"ERROR: failed to dump t-pose BVH: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
