from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import data_loaders.truebones.truebones_utils.motion_process as motion_process
from data_loaders.truebones.offline_reference_dataset import load_cond_dict, resolve_dataset_root
from eval.motion_quality_scorer import MotionQualityScorer

SUPPORTED_SUFFIXES = {".bvh", ".npy"}
DEFAULT_CHECKPOINT_DIR = "save/motion_scorer_perceptual_v2"
METRIC_KEYS = (
    "quality_score",
    "recognizability_score",
    "density_score",
    "plausibility_score",
    "physics_score",
    "species_confidence",
    "action_confidence",
    "density_log_prob",
    "density_distance",
    "physics_distance",
    "mahal_distance",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score one motion file or all matching motion files under a directory. "
            "Supports .npy directly and .bvh via paired .npy reuse or BVH re-encoding."
        )
    )
    parser.add_argument(
        "--input_path",
        required=True,
        help="Path to a single motion file or a directory to scan recursively.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default=DEFAULT_CHECKPOINT_DIR,
        help="Motion scorer checkpoint directory or specific checkpoint path.",
    )
    parser.add_argument(
        "--dataset_dir",
        default="",
        help="Processed dataset root containing cond.npy. Empty uses the repo default dataset root.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Scoring device. Falls back to CPU when CUDA is unavailable.",
    )
    parser.add_argument(
        "--filter",
        default="*",
        help="Filename glob filter applied after suffix filtering, e.g. '*prediction.bvh'.",
    )
    parser.add_argument(
        "--object_type",
        default="",
        help="Optional object_type override. When set, all files are scored as this topology.",
    )
    parser.add_argument(
        "--output_json",
        default="",
        help="Optional output JSON report path. Defaults next to the input path.",
    )
    parser.add_argument(
        "--max_files",
        default=0,
        type=int,
        help="Optional cap on how many matched files to score. 0 means all.",
    )
    parser.add_argument(
        "--no_recursive",
        action="store_true",
        help="Disable recursive search when input_path is a directory.",
    )
    parser.add_argument(
        "--no_paired_npy",
        action="store_true",
        help="For .bvh files, do not reuse a same-stem .npy file even if it exists.",
    )
    return parser.parse_args()


def resolve_cli_path(path_str: str) -> Path:
    candidate = Path(path_str).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def sanitize_filename_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    sanitized = sanitized.strip("._")
    return sanitized or "all"


def default_output_json_path(input_path: Path, filter_pattern: str) -> Path:
    if input_path.is_file():
        return input_path.with_name(f"{input_path.name}.motion_score_report.json")
    filter_tag = sanitize_filename_component(filter_pattern)
    return input_path / f"motion_score_report_{filter_tag}.json"


def collect_candidate_paths(input_path: Path, filter_pattern: str, recursive: bool, max_files: int) -> list[Path]:
    if input_path.is_file():
        candidates = [input_path]
    else:
        iterator = input_path.rglob("*") if recursive else input_path.glob("*")
        candidates = [path for path in iterator if path.is_file()]
    matched = [
        path
        for path in sorted(candidates)
        if path.suffix.lower() in SUPPORTED_SUFFIXES and fnmatch.fnmatch(path.name, filter_pattern)
    ]
    if max_files > 0:
        matched = matched[:max_files]
    return matched


def infer_object_type(path: Path, cond_dict: dict[str, dict[str, np.ndarray]], override: str = "") -> str:
    if override:
        if override not in cond_dict:
            available = ", ".join(sorted(cond_dict.keys()))
            raise KeyError(f"Unknown object_type override '{override}'. Available types: {available}")
        return override

    known_types = sorted(cond_dict.keys(), key=len, reverse=True)
    candidate_names = [path.stem]
    candidate_names.extend(parent.name for parent in path.parents)

    sample_dir_pattern = re.compile(r"^sample_\d+_(.+)$", re.IGNORECASE)
    for candidate_name in candidate_names:
        if candidate_name in cond_dict:
            return candidate_name

        match = sample_dir_pattern.match(candidate_name)
        if match:
            object_type = match.group(1)
            if object_type in cond_dict:
                return object_type

        for object_type in known_types:
            if candidate_name.startswith(f"{object_type}_") or candidate_name.startswith(f"{object_type}__"):
                return object_type

    available = ", ".join(sorted(cond_dict.keys()))
    raise KeyError(f"Could not infer object_type for '{path}'. Available types: {available}")


def infer_foot_indices(joint_names: list[str], parents: list[int], object_type: str) -> list[int]:
    if object_type in motion_process.SNAKES:
        return list(range(len(joint_names)))
    foot_indices = [
        index
        for index, joint_name in enumerate(joint_names)
        if any(token in joint_name.lower() for token in ("toe", "foot", "phalanx", "hoof", "ashi"))
    ]
    for joint_index in list(foot_indices):
        if joint_index in parents:
            children = [child_index for child_index, parent_index in enumerate(parents) if parent_index == joint_index]
            for child_index in children:
                if child_index not in foot_indices:
                    foot_indices.append(child_index)
    return sorted(set(foot_indices))


def build_bvh_reference_cache(
    object_type: str,
    object_cond: dict[str, np.ndarray],
) -> dict[str, Any]:
    parents = [int(parent) for parent in object_cond["parents"]]
    offsets = np.asarray(object_cond["offsets"], dtype=np.float32)
    joint_names = [str(name) for name in object_cond["joints_names"]]
    face_joints = motion_process.resolve_face_joints(
        object_type,
        joint_names=joint_names,
        parents=parents,
        face_joints=object_cond.get("face_joints"),
    )
    tpose_motion = np.asarray(object_cond["tpos_first_frame"], dtype=np.float32)[None, :, :]
    tpose_anim, _ = motion_process.recover_animation_from_motion_np(tpose_motion, parents, offsets)
    if tpose_anim is None:
        raise ValueError(f"Failed to reconstruct T-pose animation for object_type '{object_type}'")

    forward_joint_index = motion_process._find_forward_reference_joint(joint_names, parents)
    forward_base_joint_index = motion_process._find_neck_reference_joint(joint_names, parents)
    orientation_quat = motion_process.get_root_quat(
        motion_process.positions_global(tpose_anim),
        object_type,
        face_joint_indx=face_joints,
        forward_joint_index=forward_joint_index,
        forward_base_joint_index=forward_base_joint_index,
    )[0]
    aligned_tpose_anim, root_pose_init_xz, scale_factor = motion_process.process_anim(
        tpose_anim,
        object_type,
        face_joints=face_joints,
        orientation_quat=orientation_quat,
        forward_joint_index=forward_joint_index,
        forward_base_joint_index=forward_base_joint_index,
    )
    return {
        "parents": parents,
        "offsets": offsets,
        "face_joints": face_joints,
        "joint_names": joint_names,
        "foot_indices": infer_foot_indices(joint_names, parents, object_type),
        "root_pose_init_xz": root_pose_init_xz,
        "scale_factor": scale_factor,
        "tpos_rots": aligned_tpose_anim.rotations,
        "orientation_quat": orientation_quat,
        "forward_joint_index": forward_joint_index,
        "forward_base_joint_index": forward_base_joint_index,
    }


def load_motion_from_bvh(
    bvh_path: Path,
    scorer: MotionQualityScorer,
    object_type: str,
    object_cond: dict[str, np.ndarray],
    bvh_cache: dict[str, dict[str, Any]],
) -> np.ndarray:
    reference = bvh_cache.get(object_type)
    if reference is None:
        reference = build_bvh_reference_cache(object_type, object_cond)
        bvh_cache[object_type] = reference

    local_errors: dict[str, float] = {}
    motion_np, _parents, _max_joints, _new_anim = motion_process.get_motion(
        str(bvh_path),
        motion_process.FOOT_CONTACT_VEL_THRESH,
        object_type,
        scorer.max_joints,
        reference["root_pose_init_xz"],
        reference["scale_factor"],
        reference["offsets"],
        reference["foot_indices"],
        reference["tpos_rots"],
        local_errors,
        face_joints=reference["face_joints"],
        orientation_quat=reference["orientation_quat"],
        forward_joint_index=reference["forward_joint_index"],
        forward_base_joint_index=reference["forward_base_joint_index"],
    )
    if motion_np is None:
        raise ValueError(f"Failed to convert BVH into motion features: {bvh_path}")
    return motion_np.astype(np.float32)


def score_motion_array(
    scorer: MotionQualityScorer,
    motion_np: np.ndarray,
    object_type: str,
    object_cond: dict[str, np.ndarray],
) -> dict[str, Any]:
    if motion_np.ndim != 3:
        raise ValueError(f"Expected motion tensor with shape [T, J, F], got {motion_np.shape}")
    if motion_np.shape[0] <= 0:
        raise ValueError("Motion has zero frames")
    if motion_np.shape[1] <= 0:
        raise ValueError("Motion has zero joints")
    if motion_np.shape[2] != scorer.feature_dim:
        raise ValueError(f"Expected feature_dim={scorer.feature_dim}, got {motion_np.shape[2]}")
    if motion_np.shape[1] > scorer.max_joints:
        raise ValueError(f"Expected at most {scorer.max_joints} joints, got {motion_np.shape[1]}")

    mean = np.asarray(object_cond["mean"], dtype=np.float32)
    std = np.asarray(object_cond["std"], dtype=np.float32) + 1e-6
    if motion_np.shape[1] != mean.shape[0]:
        raise ValueError(
            f"Object type '{object_type}' expects {mean.shape[0]} joints but motion has {motion_np.shape[1]} joints"
        )

    chunk_results: list[dict[str, float]] = []
    segment_lengths: list[int] = []
    frame_start = 0
    while frame_start < motion_np.shape[0]:
        frame_end = min(frame_start + scorer.max_frames, motion_np.shape[0])
        chunk = motion_np[frame_start:frame_end].astype(np.float32, copy=False)
        normalized = (chunk - mean[None, :, :]) / std[None, :, :]
        normalized = np.nan_to_num(normalized)
        motion_tensor = torch.from_numpy(normalized).permute(1, 2, 0).unsqueeze(0)
        result = scorer.score(
            motion_tensor,
            n_joints=[chunk.shape[1]],
            lengths=[chunk.shape[0]],
            object_types=[object_type],
        )
        chunk_result = {
            key: float(result[key].reshape(-1)[0].item())
            for key in METRIC_KEYS
        }
        chunk_results.append(chunk_result)
        segment_lengths.append(int(chunk.shape[0]))
        frame_start = frame_end

    aggregated = {
        key: float(np.mean([chunk_result[key] for chunk_result in chunk_results]))
        for key in METRIC_KEYS
    }
    aggregated["density_mode"] = scorer.density_mode
    aggregated["segment_count"] = len(chunk_results)
    aggregated["segment_lengths"] = segment_lengths
    aggregated["frame_count"] = int(motion_np.shape[0])
    aggregated["joint_count"] = int(motion_np.shape[1])
    return aggregated


def build_metric_summary(values: list[float]) -> dict[str, float]:
    values_np = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(values_np.mean()),
        "median": float(np.median(values_np)),
        "min": float(values_np.min()),
        "max": float(values_np.max()),
    }


def build_report_summary(samples: list[dict[str, Any]], failures: list[dict[str, str]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "scored_files": len(samples),
        "failed_files": len(failures),
        "object_type_counts": dict(sorted(Counter(sample["object_type"] for sample in samples).items())),
        "source_mode_counts": dict(sorted(Counter(sample["source_mode"] for sample in samples).items())),
    }
    if not samples:
        return summary

    summary["metrics"] = {
        metric_key: build_metric_summary([float(sample[metric_key]) for sample in samples])
        for metric_key in METRIC_KEYS
    }

    by_object_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_object_type[sample["object_type"]].append(sample)
    summary["by_object_type"] = {
        object_type: {
            "count": len(object_samples),
            **{
                metric_key: build_metric_summary([float(sample[metric_key]) for sample in object_samples])
                for metric_key in METRIC_KEYS
            },
        }
        for object_type, object_samples in sorted(by_object_type.items())
    }

    sorted_by_quality = sorted(samples, key=lambda sample: float(sample["quality_score"]))
    summary["lowest_quality"] = [
        {
            "path": sample["path"],
            "quality_score": float(sample["quality_score"]),
            "object_type": sample["object_type"],
        }
        for sample in sorted_by_quality[:5]
    ]
    summary["highest_quality"] = [
        {
            "path": sample["path"],
            "quality_score": float(sample["quality_score"]),
            "object_type": sample["object_type"],
        }
        for sample in reversed(sorted_by_quality[-5:])
    ]
    return summary


def score_path(
    scorer: MotionQualityScorer,
    path: Path,
    cond_dict: dict[str, dict[str, np.ndarray]],
    object_type_override: str,
    prefer_paired_npy: bool,
    bvh_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    object_type = infer_object_type(path, cond_dict, override=object_type_override)
    object_cond = cond_dict[object_type]
    paired_npy_path = path.with_suffix(".npy")

    if path.suffix.lower() == ".npy":
        motion_np = np.load(path).astype(np.float32)
        source_mode = "npy"
        scored_from = path
    elif prefer_paired_npy and paired_npy_path.exists():
        motion_np = np.load(paired_npy_path).astype(np.float32)
        source_mode = "bvh_paired_npy"
        scored_from = paired_npy_path
    else:
        motion_np = load_motion_from_bvh(path, scorer, object_type, object_cond, bvh_cache)
        source_mode = "bvh_reencoded"
        scored_from = path

    sample_report = score_motion_array(
        scorer=scorer,
        motion_np=motion_np,
        object_type=object_type,
        object_cond=object_cond,
    )
    sample_report.update(
        {
            "path": str(path),
            "scored_from": str(scored_from),
            "object_type": object_type,
            "source_mode": source_mode,
        }
    )
    return sample_report


def main() -> int:
    args = parse_args()

    input_path = resolve_cli_path(args.input_path)
    if not input_path.exists():
        print(f"Error: input_path not found: {input_path}")
        return 1

    dataset_root = resolve_dataset_root(args.dataset_dir or None)
    cond_dict = load_cond_dict(dataset_root)
    scorer = MotionQualityScorer(args.checkpoint_dir, device=args.device, dataset_dir=str(dataset_root))

    recursive = not args.no_recursive
    candidate_paths = collect_candidate_paths(
        input_path=input_path,
        filter_pattern=args.filter,
        recursive=recursive,
        max_files=max(0, int(args.max_files)),
    )
    if not candidate_paths:
        print(
            "Error: no motion files matched the requested path/filter. "
            f"input_path={input_path} filter={args.filter}"
        )
        return 1

    bvh_cache: dict[str, dict[str, Any]] = {}
    samples: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for path in candidate_paths:
        try:
            sample_report = score_path(
                scorer=scorer,
                path=path,
                cond_dict=cond_dict,
                object_type_override=args.object_type,
                prefer_paired_npy=not args.no_paired_npy,
                bvh_cache=bvh_cache,
            )
            samples.append(sample_report)
            print(
                f"[{len(samples)}/{len(candidate_paths)}] "
                f"{path.name} | object_type={sample_report['object_type']} "
                f"| quality={sample_report['quality_score']:.4f} "
                f"| source={sample_report['source_mode']}"
            )
        except Exception as exc:
            failures.append({"path": str(path), "error": str(exc)})
            print(f"[fail] {path}: {exc}")
    samples.sort(key=lambda sample: sample["path"])
    report = {
        "checkpoint": str(scorer.checkpoint_path),
        "dataset_root": str(dataset_root),
        "settings": {
            "input_path": str(input_path),
            "filter": args.filter,
            "recursive": recursive,
            "object_type_override": args.object_type,
            "device": str(scorer.device),
            "prefer_paired_npy": not args.no_paired_npy,
            "max_files": int(args.max_files),
            "matched_files": len(candidate_paths),
        },
        "summary": build_report_summary(samples, failures),
        "samples": samples,
        "failures": failures,
    }

    output_json = resolve_cli_path(args.output_json) if args.output_json else default_output_json_path(input_path, args.filter)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)

    print()
    print(f"matched files: {len(candidate_paths)}")
    print(f"scored files: {len(samples)}")
    print(f"failed files: {len(failures)}")
    if samples:
        metrics = report["summary"]["metrics"]
        print(
            "quality_score mean/median/min/max: "
            f"{metrics['quality_score']['mean']:.4f} / "
            f"{metrics['quality_score']['median']:.4f} / "
            f"{metrics['quality_score']['min']:.4f} / "
            f"{metrics['quality_score']['max']:.4f}"
        )
        print(
            "recognizability_score mean: "
            f"{metrics['recognizability_score']['mean']:.4f} | density_score mean: {metrics['density_score']['mean']:.4f}"
        )
        print(
            "plausibility_score mean: "
            f"{metrics['plausibility_score']['mean']:.4f} | physics_score mean: {metrics['physics_score']['mean']:.4f}"
        )
    print(f"saved report: {output_json}")

    return 0 if samples else 1


if __name__ == "__main__":
    raise SystemExit(main())