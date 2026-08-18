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

import numpy as np

ANYTOP_DIR = Path(__file__).resolve().parent.parent
_PARENT_DIR = ANYTOP_DIR.parent
sys.path.insert(0, str(_PARENT_DIR))
sys.path.insert(0, str(ANYTOP_DIR))

from data_loaders.truebones.truebones_utils.param_utils import get_dataset_dir  # noqa: E402

from motion_lib.Animation import Animation  # noqa: E402
from motion_lib.Quaternions import Quaternions  # noqa: E402
from motion_lib.BVH import save as bvh_save  # noqa: E402


def _build_tpose_animation(object_cond: dict) -> tuple[Animation, list[str]]:
    """Build a one-frame rest-pose Animation from a cond.npy object entry."""
    offsets = np.asarray(object_cond["offsets"], dtype=np.float64)
    parents = np.asarray(object_cond["parents"], dtype=np.int64)
    joint_names = [
        str(name).replace(" ", "_")
        for name in object_cond.get(
            "canonical_bvh_joint_names",
            object_cond.get("canonical_joint_names", object_cond["joints_names"]),
        )
    ]

    num_joints = offsets.shape[0]
    if parents.shape[0] != num_joints or len(joint_names) != num_joints:
        raise ValueError(
            f"inconsistent joint counts: offsets={num_joints}, "
            f"parents={parents.shape[0]}, names={len(joint_names)}"
        )

    # One frame, identity joint rotations -> FK reproduces the rest (t-)pose.
    rotations = Quaternions.id((1, num_joints))
    positions = offsets[np.newaxis, :, :].copy()  # (1, J, 3)
    orients = Quaternions.id(num_joints)

    anim = Animation(rotations, positions, orients, offsets, parents)
    return anim, joint_names


def sample_tpose_bvh(
    dataset_dir: str | Path | None = None,
    only_objects: set[str] | None = None,
) -> Path:
    dataset_dir_path = Path(get_dataset_dir(str(dataset_dir) if dataset_dir else None)).resolve()
    cond_path = dataset_dir_path / "cond.npy"
    if not cond_path.exists():
        raise RuntimeError(f"cond.npy not found at {cond_path}")

    cond = dict(np.load(cond_path, allow_pickle=True).item())

    object_types = sorted(cond.keys())
    if only_objects is not None:
        missing = sorted(only_objects - set(object_types))
        if missing:
            print(f"[WARN] --filter names not in cond.npy, ignored: {', '.join(missing)}")
        object_types = [obj for obj in object_types if obj in only_objects]
        if not object_types:
            raise RuntimeError("no requested objects found in cond.npy")

    out_dir = dataset_dir_path / "bvh_tpose"
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for object_type in object_types:
        anim, joint_names = _build_tpose_animation(cond[object_type])
        out_path = out_dir / f"{object_type}.bvh"
        bvh_save(str(out_path), anim, names=joint_names, positions=False)
        print(f"[OK] {object_type}: {len(joint_names)} joints -> {out_path}")
        written += 1

    print(f"\n[PASS] wrote {written} t-pose BVH file(s) to {out_dir}")
    return out_dir


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
