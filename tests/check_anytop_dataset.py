from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_loaders.truebones.truebones_utils.param_utils import (  # noqa: E402
    FEATS_LEN,
    MAX_JOINTS,
    MOTION_DIR,
    BVHS_DIR,
    OBJECT_SUBSETS_DICT,
    get_dataset_dir,
)


class ValidationError(RuntimeError):
    pass


def _print_ok(message: str) -> None:
    print(f"[OK] {message}")


def _print_warn(message: str) -> None:
    print(f"[WARN] {message}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _resolve_dataset_dir(raw_value: str | None) -> Path:
    if raw_value:
        path = Path(raw_value)
    else:
        path = Path(get_dataset_dir())
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _read_required_artifacts(dataset_dir: Path) -> tuple[Path, Path, Path, Path, Path]:
    motions_dir = dataset_dir / MOTION_DIR
    bvhs_dir = dataset_dir / BVHS_DIR
    cond_path = dataset_dir / "cond.npy"
    metadata_path = dataset_dir / "metadata.txt"
    positions_error_path = dataset_dir / "positions_error_rate.txt"

    for path in [dataset_dir, motions_dir, bvhs_dir, cond_path, metadata_path, positions_error_path]:
        _require(path.exists(), f"missing required artifact: {path}")

    _print_ok(f"required artifacts found under {dataset_dir}")
    return motions_dir, bvhs_dir, cond_path, metadata_path, positions_error_path


def _validate_metadata(metadata_path: Path) -> None:
    content = metadata_path.read_text(encoding="utf-8").strip()
    _require(content != "", "metadata.txt is empty")
    required_markers = ["max joints:", "total frames:", "duration:"]
    for marker in required_markers:
        _require(marker in content, f"metadata.txt is missing marker: {marker}")
    _print_ok("metadata.txt contains summary fields")


def _validate_cond_file(cond_path: Path, objects_subset: str) -> dict:
    cond = np.load(cond_path, allow_pickle=True).item()
    _require(isinstance(cond, dict), "cond.npy did not load into a dictionary")
    _require(len(cond) > 0, "cond.npy is empty")

    cond_keys = set(cond.keys())
    
    # Determine which object types to validate
    if objects_subset != "all":
        objects_to_validate = set(OBJECT_SUBSETS_DICT[objects_subset])
        missing_objects = sorted(objects_to_validate - cond_keys)
        _require(not missing_objects, f"cond.npy is missing objects from subset {objects_subset}: {missing_objects}")
    else:
        objects_to_validate = cond_keys

    required_keys = {
        "tpos_first_frame",
        "joint_relations",
        "joints_graph_dist",
        "object_type",
        "parents",
        "offsets",
        "joints_names",
        "kinematic_chains",
        "mean",
        "std",
    }

    for object_type in objects_to_validate:
        object_cond = cond[object_type]
        missing = required_keys - set(object_cond.keys())
        _require(not missing, f"{object_type} is missing cond keys: {sorted(missing)}")

        parents = np.asarray(object_cond["parents"])
        offsets = np.asarray(object_cond["offsets"])
        tpos_first_frame = np.asarray(object_cond["tpos_first_frame"])
        mean = np.asarray(object_cond["mean"])
        std = np.asarray(object_cond["std"])
        joint_relations = np.asarray(object_cond["joint_relations"])
        joints_graph_dist = np.asarray(object_cond["joints_graph_dist"])
        joints_names = object_cond["joints_names"]

        n_joints = len(parents)
        _require(n_joints > 0, f"{object_type} has no joints")
        _require(offsets.shape == (n_joints, 3), f"{object_type} offsets shape mismatch: {offsets.shape}")
        _require(tpos_first_frame.shape == (n_joints, FEATS_LEN), f"{object_type} tpos_first_frame shape mismatch: {tpos_first_frame.shape}")
        _require(mean.shape == (n_joints, FEATS_LEN), f"{object_type} mean shape mismatch: {mean.shape}")
        _require(std.shape == (n_joints, FEATS_LEN), f"{object_type} std shape mismatch: {std.shape}")
        _require(joint_relations.shape == (n_joints, n_joints), f"{object_type} joint_relations shape mismatch: {joint_relations.shape}")
        _require(joints_graph_dist.shape == (n_joints, n_joints), f"{object_type} joints_graph_dist shape mismatch: {joints_graph_dist.shape}")
        _require(len(joints_names) == n_joints, f"{object_type} joints_names length mismatch: {len(joints_names)} vs {n_joints}")
        _require(np.isfinite(offsets).all(), f"{object_type} offsets contain NaN/Inf")
        _require(np.isfinite(tpos_first_frame).all(), f"{object_type} tpos_first_frame contains NaN/Inf")
        _require(np.isfinite(mean).all(), f"{object_type} mean contains NaN/Inf")
        _require(np.isfinite(std).all(), f"{object_type} std contains NaN/Inf")
        _require((std > 0).any(), f"{object_type} std is entirely non-positive")

    _print_ok(f"cond.npy validated for {len(cond)} object types")
    return cond


def _match_object_type(file_stem: str, cond: dict) -> str:
    matches = [object_type for object_type in cond.keys() if file_stem.startswith(f"{object_type}_")]
    _require(len(matches) > 0, f"could not match motion file to object type: {file_stem}")
    return max(matches, key=len)


def _validate_motion_files(motions_dir: Path, bvhs_dir: Path, cond: dict, sample_limit: int) -> None:
    motion_files = sorted(motions_dir.glob("*.npy"))
    bvh_files = sorted(bvhs_dir.glob("*.bvh"))

    _require(len(motion_files) > 0, "motions directory is empty")
    _require(len(bvh_files) > 0, "bvhs directory is empty")
    _require(len(motion_files) == len(bvh_files), f"motions/bvhs count mismatch: {len(motion_files)} vs {len(bvh_files)}")

    motion_stems = {path.stem for path in motion_files}
    bvh_stems = {path.stem for path in bvh_files}
    _require(motion_stems == bvh_stems, "motions and bvhs do not have matching stems")

    sample_files = motion_files[: min(sample_limit, len(motion_files))]
    for motion_path in sample_files:
        motion = np.load(motion_path)
        _require(motion.ndim == 3, f"{motion_path.name} must be rank-3, got {motion.ndim}")
        _require(motion.shape[0] > 0, f"{motion_path.name} has zero frames")
        _require(motion.shape[1] > 0, f"{motion_path.name} has zero joints")
        _require(motion.shape[1] <= MAX_JOINTS, f"{motion_path.name} exceeds MAX_JOINTS: {motion.shape[1]}")
        _require(motion.shape[2] == FEATS_LEN, f"{motion_path.name} feature dim mismatch: {motion.shape[2]}")
        _require(np.isfinite(motion).all(), f"{motion_path.name} contains NaN/Inf")

        object_type = _match_object_type(motion_path.stem, cond)
        expected_joints = len(cond[object_type]["parents"])
        _require(motion.shape[1] == expected_joints, f"{motion_path.name} joints mismatch: {motion.shape[1]} vs {expected_joints}")

    _print_ok(f"validated {len(sample_files)} motion tensors and {len(motion_files)} paired motion/BVH artifacts")


def _validate_positions_error_file(positions_error_path: Path) -> None:
    content = positions_error_path.read_text(encoding="utf-8").strip()
    _require(content.startswith("Position squared error per bvh file:"), "positions_error_rate.txt has unexpected header")
    if len(content.splitlines()) == 1:
        _print_warn("positions_error_rate.txt has no per-file entries")
    else:
        _print_ok("positions_error_rate.txt contains per-file error entries")


def _validate_loader(dataset_dir: Path, objects_subset: str, batch_size: int, num_frames: int, temporal_window: int) -> None:
    os.environ["ANYTOP_DATASET_DIR"] = str(dataset_dir)

    try:
        param_utils_module = importlib.import_module("data_loaders.truebones.truebones_utils.param_utils")
        get_opt_module = importlib.import_module("data_loaders.truebones.truebones_utils.get_opt")
        dataset_module = importlib.import_module("data_loaders.truebones.data.dataset")
        tensors_module = importlib.import_module("data_loaders.tensors")
        get_data_module = importlib.import_module("data_loaders.get_data")
    except ModuleNotFoundError as exc:
        raise ValidationError(
            f"loader validation could not import dependency '{exc.name}'. "
            f"Run this script with the same Python environment used for preprocessing/training, or pass --skip-validate."
        ) from exc

    importlib.reload(param_utils_module)
    importlib.reload(get_opt_module)
    importlib.reload(dataset_module)
    tensors_module = importlib.reload(tensors_module)
    get_data_module = importlib.reload(get_data_module)

    dataset = get_data_module.get_dataset(
        num_frames=num_frames,
        temporal_window=temporal_window,
        balanced=True,
        objects_subset=objects_subset,
    )
    _require(len(dataset) > 0, "dataset is empty during loader validation")
    effective_batch_size = min(batch_size, len(dataset))
    batch = [dataset[index] for index in range(effective_batch_size)]
    motion, cond = tensors_module.truebones_batch_collate(batch)

    _require(tuple(motion.shape[:3])[1:] == (MAX_JOINTS, FEATS_LEN), f"loader motion shape is unexpected: {tuple(motion.shape)}")
    _require(np.isfinite(motion.detach().cpu().numpy()).all(), "loader batch motion contains NaN/Inf")
    _require("y" in cond, "loader cond is missing 'y'")
    for required_key in ["lengths", "object_type", "tpos_first_frame", "mean", "std", "joints_relations", "graph_dist"]:
        _require(required_key in cond["y"], f"loader cond['y'] is missing key: {required_key}")

    _print_ok(f"loader smoke test passed with batch shape {tuple(motion.shape)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an AnyTop preprocessed dataset directory.")
    parser.add_argument("--dataset-dir", default=None, help="Dataset directory to validate. Defaults to ANYTOP_DATASET_DIR or repo default.")
    parser.add_argument("--objects-subset", default="all", choices=sorted(OBJECT_SUBSETS_DICT.keys()), help="Expected object subset for the dataset.")
    parser.add_argument("--sample-count", type=int, default=8, help="How many motion .npy files to inspect in detail.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for loader validation.")
    parser.add_argument("--num-frames", type=int, default=40, help="Sequence length used for loader validation.")
    parser.add_argument("--temporal-window", type=int, default=31, help="Temporal window used for loader validation.")
    parser.add_argument("--skip-validate", action="store_true", help="Skip DataLoader validation if you only want file and tensor checks.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = _resolve_dataset_dir(args.dataset_dir)

    print("=== AnyTop Dataset Validation ===")
    print(f"dataset_dir: {dataset_dir}")
    print(f"objects_subset: {args.objects_subset}")

    try:
        motions_dir, bvhs_dir, cond_path, metadata_path, positions_error_path = _read_required_artifacts(dataset_dir)
        _validate_metadata(metadata_path)
        cond = _validate_cond_file(cond_path, args.objects_subset)
        _validate_motion_files(motions_dir, bvhs_dir, cond, args.sample_count)
        _validate_positions_error_file(positions_error_path)
        if args.skip_validate:
            _print_warn("skipping loader validation by request")
        else:
            _validate_loader(dataset_dir, args.objects_subset, args.batch_size, args.num_frames, args.temporal_window)
    except ValidationError as exc:
        print(f"[FAIL] {exc}")
        return 1
    except Exception as exc:
        print(f"[FAIL] unexpected error: {exc}")
        return 1

    print("[PASS] dataset validation completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
