from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from motion_lib.Quaternions import Quaternions


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT.parent))
sys.path.insert(0, str(REPO_ROOT))

from data_loaders.truebones.truebones_utils.param_utils import (  # noqa: E402
    FEATS_LEN,
    MAX_JOINTS,
    MOTION_DIR,
    BVHS_DIR,
    MOTION_METADATA_FILE,
    OBJECT_SUBSETS_DICT,
    get_dataset_dir,
)
from data_loaders.truebones.truebones_utils.motion_labels import load_motion_metadata  # noqa: E402


class ValidationError(RuntimeError):
    pass


_CARDINAL_XZ_AXES = (
    ("+x", np.array([1.0, 0.0, 0.0], dtype=np.float64)),
    ("-x", np.array([-1.0, 0.0, 0.0], dtype=np.float64)),
    ("+z", np.array([0.0, 0.0, 1.0], dtype=np.float64)),
    ("-z", np.array([0.0, 0.0, -1.0], dtype=np.float64)),
)
_CANONICAL_FORWARD_VECTOR = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)


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
        path = Path(get_dataset_dir(None))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _read_required_artifacts(dataset_dir: Path, silent: bool = False) -> tuple[Path, Path, Path, Path, Path]:
    motions_dir = dataset_dir / MOTION_DIR
    bvhs_dir = dataset_dir / BVHS_DIR
    cond_path = dataset_dir / "cond.npy"
    metadata_path = dataset_dir / "metadata.txt"
    positions_error_path = dataset_dir / "positions_error_rate.txt"

    for path in [dataset_dir, motions_dir, cond_path, metadata_path, positions_error_path]:
        _require(path.exists(), f"missing required artifact: {path}")

    if not silent:
        _print_ok(f"required artifacts found under {dataset_dir}")
    return motions_dir, bvhs_dir, cond_path, metadata_path, positions_error_path


def _validate_optional_semantic_metadata(object_type: str, object_cond: dict, n_joints: int) -> None:
    canonical_joint_names = object_cond.get("canonical_joint_names")
    if canonical_joint_names is not None and len(canonical_joint_names) != n_joints:
        _print_warn(f"validation error: {object_type} canonical_joint_names length mismatch: {len(canonical_joint_names)} vs {n_joints}")

    joint_side_labels = object_cond.get("joint_side_labels")
    if joint_side_labels is not None:
        if len(joint_side_labels) != n_joints:
            _print_warn(f"validation error: {object_type} joint_side_labels length mismatch: {len(joint_side_labels)} vs {n_joints}")
        else:
            invalid_labels = sorted({label for label in joint_side_labels if label not in {"left", "right", "center"}})
            if invalid_labels:
                _print_warn(f"validation error: {object_type} joint_side_labels contain invalid values: {invalid_labels}")

    for key in ("end_effector_joints", "contact_joints"):
        indices = object_cond.get(key)
        if indices is None:
            continue
        invalid = [index for index in indices if int(index) < 0 or int(index) >= n_joints]
        if invalid:
            _print_warn(f"validation error: {object_type} {key} contain invalid joint indices: {invalid[:8]}")

    symmetry_partner_indices = object_cond.get("symmetry_partner_indices")
    if symmetry_partner_indices is not None:
        if len(symmetry_partner_indices) != n_joints:
            _print_warn(f"validation error: {object_type} symmetry_partner_indices length mismatch: {len(symmetry_partner_indices)} vs {n_joints}")
        else:
            for joint_index, partner_index in enumerate(symmetry_partner_indices):
                partner_index = int(partner_index)
                if partner_index == -1:
                    continue
                if partner_index < 0 or partner_index >= n_joints:
                    _print_warn(f"validation error: {object_type} symmetry partner out of range at joint {joint_index}: {partner_index}")
                    break
                if int(symmetry_partner_indices[partner_index]) != joint_index:
                    _print_warn(f"validation error: {object_type} symmetry partners are not reciprocal for joints {joint_index} and {partner_index}")
                    break

    symmetric_joint_pairs = object_cond.get("symmetric_joint_pairs")
    if symmetric_joint_pairs is not None:
        for pair in symmetric_joint_pairs:
            if len(pair) != 2:
                _print_warn(f"validation error: {object_type} has malformed symmetric_joint_pairs entry: {pair}")
                break
            left_index, right_index = int(pair[0]), int(pair[1])
            if min(left_index, right_index) < 0 or max(left_index, right_index) >= n_joints:
                _print_warn(f"validation error: {object_type} has out-of-range symmetric_joint_pairs entry: {pair}")
                break

    is_symmetric = object_cond.get("is_symmetric")
    if is_symmetric is not None and not isinstance(is_symmetric, (bool, np.bool_)):
        _print_warn(f"validation error: {object_type} is_symmetric should be boolean, got {type(is_symmetric).__name__}")

    orientation_quat = object_cond.get("orientation_quat")
    if orientation_quat is not None:
        orientation_quat = np.asarray(orientation_quat, dtype=np.float64)
        if orientation_quat.ndim > 1:
            orientation_quat = orientation_quat[0]
        if orientation_quat.shape != (4,):
            _print_warn(f"validation error: {object_type} orientation_quat shape mismatch: {orientation_quat.shape}")
        elif not np.isfinite(orientation_quat).all():
            _print_warn(f"validation error: {object_type} orientation_quat contains NaN/Inf")
        else:
            quat_norm = float(np.linalg.norm(orientation_quat))
            if not np.isclose(quat_norm, 1.0, atol=1e-3):
                _print_warn(f"validation error: {object_type} orientation_quat norm mismatch: {quat_norm:.6f}")

    for key in ("forward_joint_index", "forward_base_joint_index"):
        raw_index = object_cond.get(key)
        if raw_index is None:
            continue
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            _print_warn(f"validation error: {object_type} {key} is not an integer: {raw_index}")
            continue
        if index == -1:
            continue
        if index < 0 or index >= n_joints:
            _print_warn(f"validation error: {object_type} {key} out of range: {index}")


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
        "joints_names_embs",
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
            joints_names_embs = np.asarray(object_cond["joints_names_embs"])

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
            if joints_names_embs.ndim != 2 or joints_names_embs.shape[0] != n_joints:
                msg = f"{object_type} joints_names_embs shape mismatch: {joints_names_embs.shape}"
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
            if not np.isfinite(joints_names_embs).all():
                msg = f"{object_type} joints_names_embs contain NaN/Inf"
                _print_warn(f"validation error: {msg}")
            if not (std > 0).any():
                msg = f"{object_type} std is entirely non-positive"
                _print_warn(f"validation error: {msg}")

            _validate_optional_semantic_metadata(object_type, object_cond, n_joints)
        except Exception as e:
            msg = f"{object_type}: {e}"
            _print_warn(f"validation error: {msg}")
    
    _print_ok(f"cond.npy validated for {len(cond)} object types")
    return cond


def _match_object_type(file_stem: str, cond: dict) -> str:
    from utils.misc import infer_object_type_from_filename

    result = infer_object_type_from_filename(file_stem, valid_types=cond.keys())
    _require(result is not None, f"could not match motion file to object type: {file_stem}")
    return result


def _select_validation_files(files: list[Path], sample_limit: int) -> list[Path]:
    if sample_limit <= 0:
        return files
    return files[: min(sample_limit, len(files))]


def _normalize_xz_vector(vector: np.ndarray) -> np.ndarray:
    normalized = np.asarray(vector, dtype=np.float64).reshape(-1).copy()
    _require(normalized.shape == (3,), f"expected 3D vector, got shape {normalized.shape}")
    normalized[1] = 0.0
    norm = float(np.linalg.norm(normalized))
    _require(norm > 1e-8, "XZ-projected vector is degenerate")
    return normalized / norm


def _vector_angle_xz_deg(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    a = _normalize_xz_vector(vector_a)
    b = _normalize_xz_vector(vector_b)
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _summarize_tpose_orientation_axis_alignment(object_type: str, object_cond: dict[str, object]) -> tuple[str, float]:
    raw_orientation_quat = object_cond.get("orientation_quat")
    _require(raw_orientation_quat is not None, f"{object_type} is missing orientation_quat")

    orientation_quat = np.asarray(raw_orientation_quat, dtype=np.float64)
    if orientation_quat.ndim > 1:
        orientation_quat = orientation_quat[0]
    _require(orientation_quat.shape == (4,), f"{object_type} orientation_quat shape mismatch: {orientation_quat.shape}")
    _require(np.isfinite(orientation_quat).all(), f"{object_type} orientation_quat contains NaN/Inf")

    reference_forward = ((-Quaternions(orientation_quat)) * _CANONICAL_FORWARD_VECTOR)[0]

    best_axis_label = None
    best_delta_deg = None
    for axis_label, axis_vector in _CARDINAL_XZ_AXES:
        delta_deg = _vector_angle_xz_deg(reference_forward, axis_vector)
        if best_delta_deg is None or delta_deg < best_delta_deg:
            best_axis_label = axis_label
            best_delta_deg = delta_deg

    _require(best_axis_label is not None and best_delta_deg is not None, f"{object_type} produced no valid XZ axis comparison")
    return best_axis_label, float(best_delta_deg)


def _normalize_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _collect_motion_stats(motion_files: list[Path], cond: dict | None = None) -> tuple[int, int, Counter[str], set[str]]:
    total_frames = 0
    max_joints = 0
    object_counts: Counter[str] = Counter()
    object_types: set[str] = set()

    known_object_types = tuple(cond.keys()) if cond else None
    for motion_path in motion_files:
        motion = np.load(motion_path, mmap_mode="r")
        total_frames += int(motion.shape[0])
        max_joints = max(max_joints, int(motion.shape[1]))
        if cond is not None:
            object_type = _match_object_type(motion_path.stem, cond)
        else:
            from data_loaders.truebones.truebones_utils.motion_labels import infer_motion_labels_from_motion_name

            object_type = str(
                infer_motion_labels_from_motion_name(motion_path.name, object_types=known_object_types).get("object_type")
            )
        object_counts[object_type] += 1
        object_types.add(object_type)

    return total_frames, max_joints, object_counts, object_types


def _prune_excess_joint_motions(motions_dir: Path, bvhs_dir: Path, cond: dict, sample_limit: int) -> set[str]:
    motion_files = sorted(motions_dir.glob("*.npy"))
    bvh_files = sorted(bvhs_dir.glob("*.bvh")) if bvhs_dir.exists() else []

    try:
        _require(len(motion_files) > 0, "motions directory is empty")
    except ValidationError as e:
        _print_warn(f"directory/naming validation failed before pruning: {e}")
        return set()

    if bvh_files:
        motion_stems = {path.stem for path in motion_files}
        bvh_stems = {path.stem for path in bvh_files}
        if len(motion_files) != len(bvh_files) or motion_stems != bvh_stems:
            _print_warn("optional BVH artifacts do not match motions; pruning will operate on motions only")

    files_to_scan = _select_validation_files(motion_files, sample_limit)
    excess_joints_chars: set[str] = set()
    deleted_stems: set[str] = set()

    for motion_path in files_to_scan:
        try:
            motion = np.load(motion_path, mmap_mode="r")
            if motion.ndim != 3 or motion.shape[1] <= MAX_JOINTS:
                continue
            object_type = _match_object_type(motion_path.stem, cond)
            expected_joints = len(cond[object_type]["parents"])
            excess_joints_chars.add(object_type)
            _print_warn(f"{motion_path.name} exceeds MAX_JOINTS: {motion.shape[1]}")
        except Exception as exc:
            _print_warn(f"failed to inspect {motion_path.name} during pre-validation pruning: {exc}")

    for object_type in sorted(excess_joints_chars):
        char_motions = [path for path in motion_files if path.stem.startswith(f"{object_type}_")]
        char_bvhs = [path for path in bvh_files if path.stem.startswith(f"{object_type}_")]
        for path in char_motions + char_bvhs:
            try:
                path.unlink()
                deleted_stems.add(path.stem)
            except OSError as exc:
                _print_warn(f"failed to delete {path.name}: {exc}")
        if char_bvhs:
            _print_warn(
                f"deleted {len(char_motions)} motion(s) + {len(char_bvhs)} optional BVH(s) for {object_type} "
                f"(joint count exceeds MAX_JOINTS={MAX_JOINTS})"
            )
        else:
            _print_warn(
                f"deleted {len(char_motions)} motion(s) for {object_type} "
                f"(joint count exceeds MAX_JOINTS={MAX_JOINTS})"
            )

    return deleted_stems


def _validate_motion_files(motions_dir: Path, bvhs_dir: Path, cond: dict, sample_limit: int) -> None:
    motion_files = sorted(motions_dir.glob("*.npy"))
    bvh_files = sorted(bvhs_dir.glob("*.bvh")) if bvhs_dir.exists() else []

    try:
        _require(len(motion_files) > 0, "motions directory is empty")
    except ValidationError as e:
        _print_warn(f"directory/naming validation failed: {e}")
        return

    has_paired_bvhs = False
    if bvh_files:
        motion_stems = {path.stem for path in motion_files}
        bvh_stems = {path.stem for path in bvh_files}
        if len(motion_files) == len(bvh_files) and motion_stems == bvh_stems:
            has_paired_bvhs = True
        else:
            _print_warn("optional BVH artifacts do not match motions; continuing with motion-only validation")

    files_to_validate = _select_validation_files(motion_files, sample_limit)
    for motion_path in files_to_validate:
        try:
            motion = np.load(motion_path)
            _require(motion.ndim == 3, f"{motion_path.name} must be rank-3, got {motion.ndim}")
            _require(motion.shape[0] > 0, f"{motion_path.name} has zero frames")
            _require(motion.shape[1] > 0, f"{motion_path.name} has zero joints")

            object_type = _match_object_type(motion_path.stem, cond)
            expected_joints = len(cond[object_type]["parents"])
            _require(motion.shape[1] <= MAX_JOINTS, f"{motion_path.name} exceeds MAX_JOINTS: {motion.shape[1]} (original skeleton: {expected_joints})")

            _require(motion.shape[2] == FEATS_LEN, f"{motion_path.name} feature dim mismatch: {motion.shape[2]}")
            _require(np.isfinite(motion).all(), f"{motion_path.name} contains NaN/Inf")
            _require(motion.shape[1] == expected_joints, f"{motion_path.name} joints mismatch: {motion.shape[1]} vs {expected_joints}")
        except ValidationError as e:
            _print_warn(f"validation error: {motion_path.name}: {e}")

    scope = "all" if sample_limit <= 0 else str(len(files_to_validate))
    if has_paired_bvhs:
        _print_ok(f"validated {scope} motion tensors and {len(motion_files)} paired optional BVH artifacts")
    else:
        _print_ok(f"validated {scope} motion tensors")


def _validate_tpose_orientation(cond: dict, threshold_deg: float) -> None:
    checked_count = 0
    warned_count = 0

    for object_type in sorted(str(name) for name in cond.keys()):
        try:
            best_axis_label, best_delta_deg = _summarize_tpose_orientation_axis_alignment(object_type, cond[object_type])
            checked_count += 1
            if best_delta_deg <= threshold_deg:
                continue

            reference_path = cond[object_type].get("orientation_reference_fbx_path")
            reference_label = Path(reference_path).name if isinstance(reference_path, str) and reference_path.strip() else "orientation reference"
            _print_warn(
                f"{object_type} T-pose face orientation is {best_delta_deg:.2f} deg away from the nearest cardinal XZ axis "
                f"({best_axis_label}) using {reference_label}"
            )
            warned_count += 1
        except ValidationError as e:
            _print_warn(f"validation warn: {object_type}: {e}")

    _print_ok(
        f"validated T-pose face-orientation alignment for {checked_count} object types "
        f"(threshold={threshold_deg:.2f} deg)"
    )
    if warned_count:
        _print_warn(f"T-pose face-orientation warnings: {warned_count} object type(s) exceeded threshold")


def _validate_metadata(metadata_path: Path, motion_files: list[Path], cond: dict, silent: bool = False) -> bool:
    is_valid = True
    try:
        content = metadata_path.read_text(encoding="utf-8").strip()
        _require(content != "", "metadata.txt is empty")
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        parsed: dict[str, str] = {}
        object_counts: dict[str, int] = {}
        in_object_counts = False
        for line in lines:
            if line.startswith("~~~~ objects_counts"):
                in_object_counts = True
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if in_object_counts:
                object_counts[key] = int(float(value))
            else:
                parsed[key.lower()] = value

        total_frames, max_joints, expected_counts, _ = _collect_motion_stats(motion_files, cond)
        _require(int(float(parsed.get("max joints", "-1"))) == max_joints, f"metadata.txt max joints mismatch: {parsed.get('max joints')} vs {max_joints}")
        _require(int(float(parsed.get("total frames", "-1"))) == total_frames, f"metadata.txt total frames mismatch: {parsed.get('total frames')} vs {total_frames}")
        _require(object_counts == dict(expected_counts), f"metadata.txt object counts mismatch: {object_counts} vs {dict(expected_counts)}")
        if not silent:
            _print_ok("metadata.txt summary matches motion files")
    except (ValidationError, ValueError) as e:
        _print_warn(f"validation error: {e}")
        is_valid = False
    return is_valid


def _validate_motion_metadata(dataset_dir: Path, motion_files: list[Path], cond: dict, silent: bool = False) -> bool:
    metadata_path = dataset_dir / MOTION_METADATA_FILE
    if not metadata_path.exists():
        _print_warn(f"optional artifact missing: {metadata_path}")
        return False

    is_valid = True
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        motions = payload.get("motions", payload)
        _require(isinstance(motions, dict), f"{MOTION_METADATA_FILE} must contain a motions dictionary")

        expected_motion_names = {motion_path.name for motion_path in motion_files}
        actual_motion_names = set(motions.keys())
        _require(actual_motion_names == expected_motion_names, f"{MOTION_METADATA_FILE} entries mismatch with motions directory")

        for motion_path in motion_files:
            motion_name = motion_path.name
            motion_metadata = motions[motion_name]
            _require(motion_metadata.get("motion_name") == motion_name, f"motion_name mismatch for {motion_name}")
            object_type = _match_object_type(motion_path.stem, cond)
            _require(motion_metadata.get("object_type") == object_type, f"object_type mismatch for {motion_name}")
            _require(bool(motion_metadata.get("species_label")), f"species_label missing for {motion_name}")
            _require(bool(motion_metadata.get("action_label")), f"action_label missing for {motion_name}")
            _require(bool(motion_metadata.get("action_category")), f"action_category missing for {motion_name}")

            source_fbx_path = motion_metadata.get("source_fbx_path")
            source_frame_range = motion_metadata.get("source_frame_range")
            if source_fbx_path is not None:
                _require(isinstance(source_fbx_path, str) and source_fbx_path.strip(), f"source_fbx_path invalid for {motion_name}")
            if source_frame_range is not None:
                _require(
                    isinstance(source_frame_range, (list, tuple)) and len(source_frame_range) == 2,
                    f"source_frame_range invalid for {motion_name}",
                )
                start = int(source_frame_range[0])
                end = int(source_frame_range[1])
                _require(0 <= start < end, f"source_frame_range invalid for {motion_name}: {source_frame_range}")
            if (source_fbx_path is None) != (source_frame_range is None):
                _print_warn(f"validation error: {motion_name} source FBX metadata is incomplete")

        total_clips = payload.get("total_clips")
        if total_clips is not None:
            _require(int(total_clips) == len(motion_files), f"{MOTION_METADATA_FILE} total_clips mismatch: {total_clips} vs {len(motion_files)}")

        if not silent:
            _print_ok(f"{MOTION_METADATA_FILE} matches motion files")
    except ValidationError as e:
        _print_warn(f"validation error: {e}")
        is_valid = False
    return is_valid


def _validate_generated_artifacts_consistency(dataset_dir: Path, cond: dict, objects_subset: str, silent: bool = False) -> bool:
    motions_dir = dataset_dir / MOTION_DIR
    motion_files = sorted(motions_dir.glob("*.npy"))
    if not motion_files:
        _print_warn("generated artifact consistency check skipped: motions missing")
        return False

    is_consistent = True

    try:
        _, _, _, object_types_in_motions = _collect_motion_stats(motion_files, cond)
    except ValidationError as e:
        _print_warn(f"generated artifact consistency error: {e}")
        return False

    cond_keys = set(cond.keys())
    if cond_keys != object_types_in_motions:
        _print_warn(
            f"generated artifact consistency error: cond.npy object set mismatch: {sorted(cond_keys)} vs {sorted(object_types_in_motions)}"
        )
        is_consistent = False

    if objects_subset != "all":
        expected_subset = set(OBJECT_SUBSETS_DICT[objects_subset])
        missing_subset_objects = sorted(expected_subset - object_types_in_motions)
        if missing_subset_objects:
            _print_warn(
                f"generated artifact consistency error: motions are missing objects from subset {objects_subset}: {missing_subset_objects}"
            )
            is_consistent = False

    metadata_path = dataset_dir / "metadata.txt"
    if not _validate_metadata(metadata_path, motion_files, cond, silent=silent):
        is_consistent = False

    if not _validate_motion_metadata(dataset_dir, motion_files, cond, silent=silent):
        is_consistent = False

    return is_consistent


def _prepare_dataset_for_validation(
    dataset_dir: Path,
    objects_subset: str,
    sample_count: int,
) -> None:
    motions_dir, bvhs_dir, cond_path, _metadata_path, _positions_error_path = _read_required_artifacts(dataset_dir, silent=True)
    cond = dict(np.load(cond_path, allow_pickle=True).item())

    deleted_joint_stems = _prune_excess_joint_motions(motions_dir, bvhs_dir, cond, sample_count)
    needs_regeneration = bool(deleted_joint_stems)
    if not needs_regeneration:
        needs_regeneration = not _validate_generated_artifacts_consistency(dataset_dir, cond, objects_subset, silent=True)

    if needs_regeneration:
        _print_warn("regenerating non-motion dataset artifacts to match current motions")
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from regenerate_dataset_artifacts import regenerate_dataset_artifacts

        regenerate_dataset_artifacts(str(dataset_dir))


def _validate_positions_error_file(positions_error_path: Path) -> None:
    try:
        content = positions_error_path.read_text(encoding="utf-8").strip()
        _require(content.startswith("Position squared error per source clip:"), "positions_error_rate.txt has unexpected header")
        if len(content.splitlines()) == 1:
            _print_warn("positions_error_rate.txt has no per-file entries")
        else:
            _print_ok("positions_error_rate.txt contains per-file error entries")
    except ValidationError as e:
        _print_warn(f"validation error: {e}")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an AnyTop preprocessed dataset directory.")
    parser.add_argument("--dataset-dir", default=None, help="Dataset directory to validate. If not specified, uses default path.")
    parser.add_argument("--objects-subset", default="all", choices=sorted(OBJECT_SUBSETS_DICT.keys()), help="Subset that must be present in the dataset; incremental runs may still contain additional objects.")
    parser.add_argument("--sample-count", type=int, default=0, help="How many motion files to validate in detail. Use 0 to validate all files.")
    parser.add_argument("--orientation-threshold-deg", type=float, default=5.0, help="Maximum allowed T-pose face-orientation delta from the nearest cardinal XZ axis (+x/-x/+z/-z) before warning.")
    parser.add_argument("--skip-orientation-check", action="store_true", help="Skip T-pose face-orientation validation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = _resolve_dataset_dir(args.dataset_dir)
    _require(args.sample_count >= 0, "sample-count must be >= 0")
    _require(args.orientation_threshold_deg >= 0, "orientation-threshold-deg must be >= 0")

    print("=== AnyTop Dataset Validation ===")
    print(f"dataset_dir: {dataset_dir}")
    print(f"objects_subset: {args.objects_subset}")
    print(f"file_validation_scope: {'all files' if args.sample_count == 0 else f'first {args.sample_count} files'}")

    _prepare_dataset_for_validation(
        dataset_dir,
        args.objects_subset,
        args.sample_count,
    )

    motions_dir, bvhs_dir, cond_path, metadata_path, positions_error_path = _read_required_artifacts(dataset_dir)
    cond = _validate_cond_file(cond_path, args.objects_subset)

    motion_files = sorted(motions_dir.glob("*.npy"))
    _validate_metadata(metadata_path, motion_files, cond)
    _validate_motion_metadata(dataset_dir, motion_files, cond)
    
    _validate_motion_files(motions_dir, bvhs_dir, cond, args.sample_count)
    
    if args.skip_orientation_check:
        _print_warn("skipping T-pose face-orientation validation by request")
    else:
        _validate_tpose_orientation(cond, args.orientation_threshold_deg)
    
    _validate_positions_error_file(positions_error_path)

    print("[PASS] dataset validation completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
