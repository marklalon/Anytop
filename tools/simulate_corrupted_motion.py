"""
Simulate Corrupted Motion — Freeze Specified Joint Subtrees

Loads a raw motion NPY ``(T, J, 13)``, freezes the channels of the requested
joints (and, by default, their subtrees) by setting them to the dataset's
per-joint mean — the exact corruption signal the model sees during training
when ``joint_mask_prob`` masks a subtree — and writes both the corrupted NPY
and a BVH preview.

The joint name resolution follows the same alias rules as
``--inpaint_joints`` in ``sample/generate.py`` (raw / canonical /
canonical_bvh names are all accepted), and the subtree expansion mirrors
``model/joint_mask_utils.collect_subtree_indices``.

Usage
-----
    # From the Anytop/ directory:
    python tools/simulate_corrupted_motion.py \
        --motion path/to/Horse_Attack_1.npy \
        --joints LeftThigh,RightThigh \
        --output-dir outputs/corrupted_preview

Arguments
---------
  --motion             Source raw-feature NPY (T, J, 13).
  --object-type        Species/object key in cond.npy (e.g. Horse, Ostrich).
                       Auto-inferred from the motion filename when omitted
                       (uses utils.misc.infer_object_type_from_filename).
  --joints             Comma-separated joint names to freeze. Accepts any of
                       the raw / canonical / canonical_bvh aliases.
  --no-include-subtree Freeze only the named joints (default: also freeze
                       all descendants, matching --inpaint_include_subtree).
  --cond-file          Path to cond.npy (default: dataset/truebones/zoo/
                       truebones_processed/cond.npy).
  --output-dir         Where to write the corrupted NPY + BVH preview.
  --output-stem        Override the output filename stem (default: motion
                       stem + "_frozen-<joint1>+<joint2>").
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Repo root setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_lib import BVH
from data_loaders.truebones.truebones_utils.motion_process import (
    recover_bvh_export_animation_from_motion_np,
)
from data_loaders.truebones.truebones_utils.animation_utils import (
    refresh_joint_metadata_in_cond_dict,
)
from model.joint_mask_utils import collect_subtree_indices
from utils.misc import infer_object_type_from_filename


# ---------------------------------------------------------------------------
# Joint name resolution — mirrors sample/generate._resolve_inpaint_joint_indices
# ---------------------------------------------------------------------------

def _resolve_freeze_joint_indices(
    object_cond: dict,
    names_arg: str,
    include_subtree: bool,
) -> tuple[set[int], int, list[str]]:
    """Resolve comma-separated joint names to a set of joint indices.

    Names are matched against the union of raw / canonical_joint_names /
    canonical_bvh_joint_names lists. When ``include_subtree`` is set, every
    descendant of each selected joint is added.
    """
    raw_names = list(object_cond["joints_names"])
    n_joints = len(raw_names)
    canon = list(object_cond.get("canonical_joint_names", raw_names))
    canon_bvh = list(object_cond.get("canonical_bvh_joint_names", raw_names))

    if not names_arg.strip():
        raise ValueError("--joints must be a non-empty comma-separated list")

    alias_to_index: dict[str, int] = {}
    for idx in range(n_joints):
        for alias in (raw_names[idx], canon[idx], canon_bvh[idx]):
            if alias is not None:
                alias_to_index.setdefault(str(alias), idx)

    base: set[int] = set()
    matched_names: list[str] = []
    invalid: list[str] = []
    for token in names_arg.split(","):
        token = token.strip()
        if not token:
            continue
        if token in alias_to_index:
            idx = alias_to_index[token]
            if idx not in base:
                base.add(idx)
                matched_names.append(raw_names[idx])
        else:
            invalid.append(token)

    if invalid:
        table = ["  idx | raw | canonical | canonical_bvh"]
        for idx in range(n_joints):
            table.append(
                f"  {idx:>3} | {raw_names[idx]} | {canon[idx]} | {canon_bvh[idx]}"
            )
        raise ValueError(
            f"--joints: unknown joint name(s) {invalid}.\n"
            "Accepted names (any of the three aliases):\n" + "\n".join(table)
        )

    if not include_subtree:
        return base, n_joints, matched_names

    parents = np.asarray(object_cond["parents"], dtype=np.int64)
    children: list[list[int]] = [[] for _ in range(n_joints)]
    for j in range(n_joints):
        p = int(parents[j])
        if 0 <= p < n_joints:
            children[p].append(j)

    expanded: set[int] = set()
    for root_idx in base:
        for j in collect_subtree_indices(int(root_idx), children):
            expanded.add(int(j))
    return expanded, n_joints, matched_names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Freeze specified joints (and their subtrees) in a raw motion NPY "
            "by replacing their channels with the dataset's per-joint mean — "
            "matching the corruption signal seen during training when "
            "joint_mask_prob masks a subtree."
        )
    )
    p.add_argument("--motion", required=True,
                   help="Source raw-feature NPY with shape (T, J, 13).")
    p.add_argument("--object-type", default="",
                   help="Species/object key in cond.npy (e.g. Horse, Ostrich). "
                        "Auto-inferred from the motion filename when omitted.")
    p.add_argument("--joints", required=True,
                   help="Comma-separated joint names to freeze. Accepts raw / "
                        "canonical / canonical_bvh aliases.")
    p.add_argument("--no-include-subtree", dest="include_subtree",
                   action="store_false", default=True,
                   help="Freeze only the named joints (default: also freeze "
                        "all descendants).")
    p.add_argument("--cond-file",
                   default="dataset/truebones/zoo/truebones_processed/cond.npy",
                   help="Path to cond.npy holding per-species normalization stats.")
    p.add_argument("--output-dir", default="outputs/corrupted_motion",
                   help="Directory to write the corrupted NPY + BVH preview.")
    p.add_argument("--output-stem", default="",
                   help="Override the output filename stem.")
    p.add_argument("--noise-std", type=float, default=0.0,
                   help="Std of i.i.d. Gaussian noise added to frozen joints in "
                        "normalized space (on top of the mean-fill). 0 disables "
                        "noise; 1.0 matches the unit-normal scale of the data.")
    p.add_argument("--noise-seed", type=int, default=0,
                   help="RNG seed for the noise sampler (default: 0).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    input_path = Path(args.motion).resolve()
    if not input_path.is_file():
        print(f"[ERROR] Input motion not found: {input_path}")
        return 1

    cond_path = Path(args.cond_file)
    if not cond_path.is_absolute():
        cond_path = (REPO_ROOT / cond_path).resolve()
    if not cond_path.is_file():
        print(f"[ERROR] cond.npy not found: {cond_path}")
        return 1

    # -----------------------------------------------------------------------
    # Load cond.npy and refresh joint-name aliases
    # -----------------------------------------------------------------------
    print(f"[INFO] Loading cond.npy: {cond_path}")
    cond_dict = np.load(cond_path, allow_pickle=True).item()
    cond_dict = refresh_joint_metadata_in_cond_dict(cond_dict)

    object_type = args.object_type.strip()
    if not object_type:
        inferred = infer_object_type_from_filename(
            input_path.name, valid_types=set(cond_dict.keys()),
        )
        if inferred is None:
            print(
                f"[ERROR] --object-type omitted and could not be inferred from "
                f"filename '{input_path.name}'. Pass --object-type explicitly. "
                f"Available: {sorted(cond_dict.keys())[:20]} ..."
            )
            return 1
        object_type = inferred
        print(f"[INFO] Inferred --object-type from filename: '{object_type}'")
    elif object_type not in cond_dict:
        print(
            f"[ERROR] object-type '{object_type}' not in cond.npy. "
            f"Available: {sorted(cond_dict.keys())[:20]} ..."
        )
        return 1
    object_cond = cond_dict[object_type]

    parents = np.asarray(object_cond["parents"], dtype=np.int64)
    offsets = np.asarray(object_cond["offsets"], dtype=np.float32)
    mean = np.asarray(object_cond["mean"], dtype=np.float32)
    std = np.asarray(object_cond["std"], dtype=np.float32)
    std_safe = std + 1e-6
    n_joints_cond = parents.shape[0]

    joint_names_bvh = list(
        object_cond.get(
            "canonical_bvh_joint_names",
            object_cond.get("joints_names", []),
        )
    )
    if not joint_names_bvh:
        joint_names_bvh = [f"joint_{j}" for j in range(n_joints_cond)]

    # -----------------------------------------------------------------------
    # Load input motion (T, J, 13) raw features
    # -----------------------------------------------------------------------
    motion_raw = np.load(input_path).astype(np.float32)
    if motion_raw.ndim != 3 or motion_raw.shape[2] != 13:
        print(
            f"[ERROR] Input motion must be (T, J, 13); got {motion_raw.shape}"
        )
        return 1
    T, J, _ = motion_raw.shape
    if J != n_joints_cond:
        print(
            f"[ERROR] Joint count mismatch: motion has {J}, cond expects "
            f"{n_joints_cond} for '{object_type}'."
        )
        return 1

    # -----------------------------------------------------------------------
    # Resolve freeze joints + build per-joint mask
    # -----------------------------------------------------------------------
    freeze_indices, _, matched_names = _resolve_freeze_joint_indices(
        object_cond, args.joints, args.include_subtree,
    )
    if not freeze_indices:
        print("[ERROR] No joints resolved from --joints.")
        return 1
    if 0 in freeze_indices:
        print(
            "[WARN] Freezing the root joint zeroes the global trajectory in "
            "normalized space and yields the dataset mean root pose."
        )

    freeze_mask = np.zeros((J,), dtype=bool)
    freeze_indices_sorted = sorted(int(j) for j in freeze_indices)
    freeze_mask[freeze_indices_sorted] = True
    raw_names = list(object_cond["joints_names"])
    frozen_name_list = [raw_names[i] for i in freeze_indices_sorted]
    print(
        f"[INFO] Freezing {freeze_mask.sum()}/{J} joints "
        f"(named={matched_names}, full={frozen_name_list})"
    )

    # -----------------------------------------------------------------------
    # Apply mask in normalized space → denorm (training-style corruption)
    # -----------------------------------------------------------------------
    motion_norm = np.nan_to_num(
        (motion_raw - mean[None, :, :]) / std_safe[None, :, :],
        copy=True,
    ).astype(np.float32)
    motion_norm[:, freeze_mask, :] = 0.0
    if args.noise_std > 0.0:
        rng = np.random.default_rng(int(args.noise_seed))
        n_frozen = int(freeze_mask.sum())
        noise = rng.standard_normal(
            size=(T, n_frozen, motion_norm.shape[2]),
        ).astype(np.float32) * np.float32(args.noise_std)
        motion_norm[:, freeze_mask, :] += noise
        print(
            f"[INFO] Added i.i.d. Gaussian noise (std={args.noise_std}, "
            f"seed={args.noise_seed}) on top of mean-fill for frozen joints."
        )
    motion_corrupted = (
        motion_norm * std_safe[None, :, :] + mean[None, :, :]
    ).astype(np.float32)

    # -----------------------------------------------------------------------
    # Write outputs
    # -----------------------------------------------------------------------
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = args.output_stem.strip()
    if not stem:
        # Sanitize joint names for filename use (truncate at 60 chars).
        joints_tag = "+".join(matched_names).replace("/", "_").replace(" ", "_")
        if len(joints_tag) > 60:
            joints_tag = joints_tag[:57] + "..."
        subtree_tag = "" if args.include_subtree else "_only"
        noise_tag = f"_noise{args.noise_std:g}" if args.noise_std > 0.0 else ""
        stem = f"{input_path.stem}_frozen-{joints_tag}{subtree_tag}{noise_tag}"

    npy_out = output_dir / f"{stem}.npy"
    bvh_out = output_dir / f"{stem}.bvh"

    np.save(npy_out, motion_corrupted)
    print(f"[OK ] Wrote corrupted motion NPY → {npy_out}")

    motion_metadata: dict[str, object] = {}
    trans_root = object_cond.get("translation_root_index")
    if trans_root is None:
        trans_root = object_cond.get("forward_base_joint_index")
    if trans_root is not None:
        motion_metadata["translation_root_index"] = int(trans_root)

    anim, joints_names_dfs, has_animated_pos = recover_bvh_export_animation_from_motion_np(
        motion_corrupted,
        list(parents),
        offsets,
        list(joint_names_bvh),
        motion_metadata=motion_metadata,
        allow_infer=True,
    )
    if anim is None:
        print("[ERROR] recover_bvh_export_animation_from_motion_np returned None")
        return 2

    BVH.save(str(bvh_out), anim, joints_names_dfs, positions=has_animated_pos)
    print(f"[OK ] Wrote BVH preview → {bvh_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
