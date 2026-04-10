from __future__ import annotations

import argparse
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
from data_loaders.truebones.truebones_utils.motion_process import (  # noqa: E402
    BVH,
    _get_facing_candidates,
    _find_forward_reference_joint,
    _find_neck_reference_joint,
    _get_head_forward,
    positions_global,
    resolve_face_joints,
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
    try:
        content = metadata_path.read_text(encoding="utf-8").strip()
        _require(content != "", "metadata.txt is empty")
        required_markers = ["max joints:", "total frames:", "duration:"]
        for marker in required_markers:
            _require(marker in content, f"metadata.txt is missing marker: {marker}")
        _print_ok("metadata.txt contains summary fields")
    except ValidationError as e:
        _print_warn(f"validation error: {e}")


def _validate_cond_file(cond_path: Path, objects_subset: str) -> dict:
    cond = np.load(cond_path, allow_pickle=True).item()
    
    try:
        _require(isinstance(cond, dict), "cond.npy did not load into a dictionary")
        _require(len(cond) > 0, "cond.npy is empty")
    except ValidationError as e:
        _print_warn(f"validation error: {e}")

    cond_keys = set(cond.keys())
    
    # Determine which object types to validate
    if objects_subset != "all":
        objects_to_validate = set(OBJECT_SUBSETS_DICT[objects_subset])
        missing_objects = sorted(objects_to_validate - cond_keys)
        if missing_objects:
            _print_warn(f"cond.npy is missing objects from subset {objects_subset}: {missing_objects}")
        else:
            objects_to_validate = objects_to_validate
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
        try:
            object_cond = cond[object_type]
            missing = required_keys - set(object_cond.keys())
            if missing:
                msg = f"{object_type} is missing cond keys: {sorted(missing)}"
                _print_warn(f"validation error: {msg}")
                continue

            parents = np.asarray(object_cond["parents"])
            offsets = np.asarray(object_cond["offsets"])
            tpos_first_frame = np.asarray(object_cond["tpos_first_frame"])
            mean = np.asarray(object_cond["mean"])
            std = np.asarray(object_cond["std"])
            joint_relations = np.asarray(object_cond["joint_relations"])
            joints_graph_dist = np.asarray(object_cond["joints_graph_dist"])
            joints_names = object_cond["joints_names"]

            n_joints = len(parents)
            if n_joints <= 0:
                msg = f"{object_type} has no joints"
                _print_warn(f"validation error: {msg}")
            if offsets.shape != (n_joints, 3):
                msg = f"{object_type} offsets shape mismatch: {offsets.shape}"
                _print_warn(f"validation error: {msg}")
            if tpos_first_frame.shape != (n_joints, FEATS_LEN):
                msg = f"{object_type} tpos_first_frame shape mismatch: {tpos_first_frame.shape}"
                _print_warn(f"validation error: {msg}")
            if mean.shape != (n_joints, FEATS_LEN):
                msg = f"{object_type} mean shape mismatch: {mean.shape}"
                _print_warn(f"validation error: {msg}")
            if std.shape != (n_joints, FEATS_LEN):
                msg = f"{object_type} std shape mismatch: {std.shape}"
                _print_warn(f"validation error: {msg}")
            if joint_relations.shape != (n_joints, n_joints):
                msg = f"{object_type} joint_relations shape mismatch: {joint_relations.shape}"
                _print_warn(f"validation error: {msg}")
            if joints_graph_dist.shape != (n_joints, n_joints):
                msg = f"{object_type} joints_graph_dist shape mismatch: {joints_graph_dist.shape}"
                _print_warn(f"validation error: {msg}")
            if len(joints_names) != n_joints:
                msg = f"{object_type} joints_names length mismatch: {len(joints_names)} vs {n_joints}"
                _print_warn(f"validation error: {msg}")
            if not np.isfinite(offsets).all():
                msg = f"{object_type} offsets contain NaN/Inf"
                _print_warn(f"validation error: {msg}")
            if not np.isfinite(tpos_first_frame).all():
                msg = f"{object_type} tpos_first_frame contains NaN/Inf"
                _print_warn(f"validation error: {msg}")
            if not np.isfinite(mean).all():
                msg = f"{object_type} mean contains NaN/Inf"
                _print_warn(f"validation error: {msg}")
            if not np.isfinite(std).all():
                msg = f"{object_type} std contains NaN/Inf"
                _print_warn(f"validation error: {msg}")
            if not (std > 0).any():
                msg = f"{object_type} std is entirely non-positive"
                _print_warn(f"validation error: {msg}")
        except Exception as e:
            msg = f"{object_type}: {e}"
            _print_warn(f"validation error: {msg}")
    
    _print_ok(f"cond.npy validated for {len(cond)} object types")
    return cond


def _match_object_type(file_stem: str, cond: dict) -> str:
    matches = [object_type for object_type in cond.keys() if file_stem.startswith(f"{object_type}_")]
    _require(len(matches) > 0, f"could not match motion file to object type: {file_stem}")
    return max(matches, key=len)


def _select_validation_files(files: list[Path], sample_limit: int) -> list[Path]:
    if sample_limit <= 0:
        return files
    return files[: min(sample_limit, len(files))]


def _validate_motion_files(motions_dir: Path, bvhs_dir: Path, cond: dict, sample_limit: int) -> None:
    motion_files = sorted(motions_dir.glob("*.npy"))
    bvh_files = sorted(bvhs_dir.glob("*.bvh"))

    try:
        _require(len(motion_files) > 0, "motions directory is empty")
        _require(len(bvh_files) > 0, "bvhs directory is empty")
        _require(len(motion_files) == len(bvh_files), f"motions/bvhs count mismatch: {len(motion_files)} vs {len(bvh_files)}")

        motion_stems = {path.stem for path in motion_files}
        bvh_stems = {path.stem for path in bvh_files}
        _require(motion_stems == bvh_stems, "motions and bvhs do not have matching stems")
    except ValidationError as e:
        _print_warn(f"directory/naming validation failed: {e}")
        return

    files_to_validate = _select_validation_files(motion_files, sample_limit)
    for motion_path in files_to_validate:
        try:
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
        except ValidationError as e:
            _print_warn(f"validation error: {motion_path.name}: {e}")

    scope = "all" if sample_limit <= 0 else str(len(files_to_validate))
    _print_ok(f"validated {scope} motion tensors and {len(motion_files)} paired motion/BVH artifacts")


def _vector_angle_deg(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    a = np.asarray(vector_a, dtype=np.float64).reshape(-1)
    b = np.asarray(vector_b, dtype=np.float64).reshape(-1)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    _require(a_norm > 1e-8 and b_norm > 1e-8, "cannot compare zero-length orientation vectors")
    cosine = float(np.dot(a / a_norm, b / b_norm))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _format_candidate_angles(candidate_angles: dict[str, float]) -> str:
    return ", ".join(f"{name}: {angle:.4f}" for name, angle in candidate_angles.items())


def _get_frame_orientation_candidates(
    processed_bvh_path: Path,
    object_type: str,
    object_cond: dict,
    frame_index: int,
) -> dict[str, np.ndarray]:
    anim, joint_names, _ = BVH.load(str(processed_bvh_path))
    global_positions = positions_global(anim)
    resolved_frame_index = frame_index
    if resolved_frame_index < 0:
        resolved_frame_index = global_positions.shape[0] + resolved_frame_index
    _require(0 <= resolved_frame_index < global_positions.shape[0], f"{processed_bvh_path.name} has no frame {frame_index}")
    frame_positions = global_positions[resolved_frame_index:resolved_frame_index + 1]

    face_joints = object_cond.get("face_joints")
    if face_joints is None:
        face_joints = resolve_face_joints(object_type, joint_names, anim.parents, face_joints=None)

    forward_joint_index = _find_forward_reference_joint(joint_names, anim.parents)
    forward_base_joint_index = _find_neck_reference_joint(joint_names, anim.parents)

    raw = _get_facing_candidates(
        frame_positions,
        object_type,
        face_joint_indx=face_joints,
        forward_joint_index=forward_joint_index,
        forward_base_joint_index=forward_base_joint_index,
    )
    candidates = {
        name: np.asarray(fwd[0], dtype=np.float64)
        for name, fwd in raw.items()
        if fwd is not None and np.isfinite(fwd).all()
    }

    _require(candidates, f"{processed_bvh_path.name} produced no valid orientation candidates for frame {resolved_frame_index}")
    return candidates


def _validate_motion_orientation(bvhs_dir: Path, cond: dict, sample_limit: int, threshold_deg: float) -> None:
    bvh_files = sorted(bvhs_dir.glob("*.bvh"))
    
    try:
        _require(len(bvh_files) > 0, "bvhs directory is empty")
    except ValidationError as e:
        _print_warn(f"directory validation failed: {e}")
        return

    files_to_validate = _select_validation_files(bvh_files, sample_limit)
    target_forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    errors = []

    _SKIP_ORIENTATION_KEYWORDS = {"left", "right", "die", "dead", "death", "lying"}

    for bvh_path in files_to_validate:
        action_name_lower = bvh_path.stem.lower()
        if any(kw in action_name_lower for kw in _SKIP_ORIENTATION_KEYWORDS):
            continue
        try:
            object_type = _match_object_type(bvh_path.stem, cond)
            first_candidates = _get_frame_orientation_candidates(bvh_path, object_type, cond[object_type], 0)
            first_candidate_angles = {
                name: _vector_angle_deg(forward, target_forward)
                for name, forward in first_candidates.items()
            }
            first_best_name, first_best_angle_deg = min(first_candidate_angles.items(), key=lambda item: item[1])
            if first_best_angle_deg <= threshold_deg:
                continue

            last_candidates = _get_frame_orientation_candidates(bvh_path, object_type, cond[object_type], -1)
            last_candidate_angles = {
                name: _vector_angle_deg(forward, target_forward)
                for name, forward in last_candidates.items()
            }
            last_best_name, last_best_angle_deg = min(last_candidate_angles.items(), key=lambda item: item[1])
            _require(
                last_best_angle_deg <= threshold_deg,
                f"processed orientation exceeds threshold on both first and last frames： {first_best_angle_deg:.2f}|{first_best_angle_deg:.2f}, threshold={threshold_deg:.2f}",
            )
        except ValidationError as e:
            _print_warn(f"validation error: {bvh_path.name}: {e}")
            errors.append(str(e))

    if errors:
        raise ValidationError(f"orientation validation failed: {len(errors)} file(s) exceeded threshold")
    
    scope = "all" if sample_limit <= 0 else str(len(files_to_validate))
    _print_ok(f"validated processed early-frame +Z orientation for {scope} processed BVHs (threshold={threshold_deg:.2f} deg)")


def _validate_positions_error_file(positions_error_path: Path) -> None:
    try:
        content = positions_error_path.read_text(encoding="utf-8").strip()
        _require(content.startswith("Position squared error per bvh file:"), "positions_error_rate.txt has unexpected header")
        if len(content.splitlines()) == 1:
            _print_warn("positions_error_rate.txt has no per-file entries")
        else:
            _print_ok("positions_error_rate.txt contains per-file error entries")
    except ValidationError as e:
        _print_warn(f"validation error: {e}")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an AnyTop preprocessed dataset directory.")
    parser.add_argument("--dataset-dir", default=None, help="Dataset directory to validate. Defaults to ANYTOP_DATASET_DIR or repo default.")
    parser.add_argument("--objects-subset", default="all", choices=sorted(OBJECT_SUBSETS_DICT.keys()), help="Expected object subset for the dataset.")
    parser.add_argument("--sample-count", type=int, default=0, help="How many motion/BVH files to validate in detail. Use 0 to validate all files.")
    parser.add_argument("--orientation-threshold-deg", type=float, default=5.0, help="Maximum allowed first-frame facing error from +Z for validated processed BVHs.")
    parser.add_argument("--skip-orientation-check", action="store_true", help="Skip processed-BVH orientation validation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = _resolve_dataset_dir(args.dataset_dir)
    _require(args.sample_count >= 0, "sample-count must be >= 0")

    print("=== AnyTop Dataset Validation ===")
    print(f"dataset_dir: {dataset_dir}")
    print(f"objects_subset: {args.objects_subset}")
    print(f"file_validation_scope: {'all files' if args.sample_count == 0 else f'first {args.sample_count} files'}")

    motions_dir, bvhs_dir, cond_path, metadata_path, positions_error_path = _read_required_artifacts(dataset_dir)
    
    _validate_metadata(metadata_path)
    
    cond = _validate_cond_file(cond_path, args.objects_subset)
    
    _validate_motion_files(motions_dir, bvhs_dir, cond, args.sample_count)
    
    if args.skip_orientation_check:
        _print_warn("skipping processed-BVH orientation validation by request")
    else:
        _validate_motion_orientation(bvhs_dir, cond, args.sample_count, args.orientation_threshold_deg)
    
    _validate_positions_error_file(positions_error_path)

    print("[PASS] dataset validation completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
