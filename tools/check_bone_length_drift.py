"""
check_bone_length_drift.py

Measure per-bone frame-to-frame bone-length drift relative to animation frame 0,
while also reporting a separate reference-only comparison between the character
T-pose stored in cond.npy and the animation's first frame.

Supported inputs:
    - .npy  : AnyTop motion features
    - .glb  : skinned GLB/GLTF animation sampled through Blender

The main drift report uses each bone's length in the animation's first frame as
its baseline and measures how much later frames stretch or shrink relative to
that baseline. A separate T-pose reference section keeps the old cond.npy
comparison as context only.

Examples:
    python Anytop/tools/check_bone_length_drift.py \
        --input Anytop/outputs/Fox___Run_0.npy

    python Anytop/tools/check_bone_length_drift.py \
        --input Anytop/outputs/Fox___Run_0.glb \
        --json-out outputs/Fox___Run_0.bone_length_drift.json
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANYTOP_ROOT = os.path.realpath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.dirname(ANYTOP_ROOT)

for _path in [REPO_ROOT, ANYTOP_ROOT]:
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _load_utils_module(module_name: str) -> None:
    module_path = os.path.join(ANYTOP_ROOT, "utils", f"{module_name.rsplit('.', 1)[-1]}.py")
    if not os.path.isfile(module_path) or module_name in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)


_load_utils_module("utils.rotation_conversions")
_load_utils_module("utils.npy_roundtrip_utils")
_load_utils_module("utils.misc")

from utils.misc import infer_object_type_from_filename
from utils.npy_roundtrip_utils import coerce_feature_payload, recover_from_features
from Anytop.motion_lib.Animation import positions_global
from Anytop.motion_lib.FBX import (
    _extract_armature_skeleton_data,
    _get_action_sample_times,
    _set_scene_time,
    clear_scene,
    remove_lights_and_cameras,
)
from Anytop.utils.exporter import _canonical_bone_name


_DEFAULT_COND_NPY = os.path.realpath(
    os.path.join(ANYTOP_ROOT, "dataset", "truebones", "zoo", "truebones_processed", "cond.npy")
)


@dataclass
class ReferenceSkeleton:
    object_type: str
    bone_names: list[str]
    parents: np.ndarray
    offsets: np.ndarray


@dataclass
class MotionWorldData:
    file_path: str
    file_format: str
    bone_names: list[str]
    parents: np.ndarray
    world_positions: np.ndarray
    sample_frames: list[float]

    @property
    def num_frames(self) -> int:
        return int(self.world_positions.shape[0])

    @property
    def num_joints(self) -> int:
        return int(self.world_positions.shape[1])


def _load_cond_dict(cond_npy_path: str) -> dict[str, Any]:
    if not os.path.isfile(cond_npy_path):
        raise FileNotFoundError(f"cond.npy not found: {cond_npy_path}")

    cond = np.load(cond_npy_path, allow_pickle=True)
    if isinstance(cond, np.ndarray) and cond.shape == ():
        cond = cond.item()
    if not isinstance(cond, dict) or not cond:
        raise ValueError(f"cond.npy did not load into a non-empty dict: {cond_npy_path}")
    return cond


def _resolve_object_type(input_path: str, cond: dict[str, Any], object_type: str | None) -> str:
    if object_type is not None:
        if object_type not in cond:
            raise ValueError(
                f"object_type '{object_type}' not found in cond.npy. Available: {sorted(str(k) for k in cond.keys())}"
            )
        return object_type

    inferred = infer_object_type_from_filename(input_path, valid_types=cond.keys())
    if inferred is None:
        raise ValueError(
            f"Cannot auto-detect object_type from '{os.path.basename(input_path)}'. "
            f"Pass --object-type explicitly."
        )
    return str(inferred)


def _load_reference_skeleton(object_type: str, cond_entry: dict[str, Any]) -> ReferenceSkeleton:
    required_keys = ("joints_names", "parents", "offsets")
    missing = [key for key in required_keys if key not in cond_entry]
    if missing:
        raise ValueError(f"cond entry '{object_type}' is missing keys: {missing}")

    bone_names = [str(name) for name in cond_entry["joints_names"]]
    parents = np.asarray(cond_entry["parents"], dtype=np.int32)
    offsets = np.asarray(cond_entry["offsets"], dtype=np.float64)

    if parents.shape != (len(bone_names),):
        raise ValueError(
            f"cond parents shape mismatch for '{object_type}': expected ({len(bone_names)},), got {parents.shape}"
        )
    if offsets.shape != (len(bone_names), 3):
        raise ValueError(
            f"cond offsets shape mismatch for '{object_type}': expected ({len(bone_names)}, 3), got {offsets.shape}"
        )

    return ReferenceSkeleton(
        object_type=str(object_type),
        bone_names=bone_names,
        parents=parents,
        offsets=offsets,
    )


def _load_npy_payload(npy_path: str) -> Any:
    try:
        raw = np.load(npy_path, allow_pickle=False)
    except ValueError as exc:
        if "Object arrays cannot be loaded when allow_pickle=False" not in str(exc):
            raise
        raw = np.load(npy_path, allow_pickle=True)

    if isinstance(raw, np.ndarray) and raw.shape == () and raw.dtype == object:
        raw = raw.item()
    return raw


def _collapse_redundant_root_once(
    bone_names: list[str],
    parents: np.ndarray,
    offsets: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray, str] | None:
    if len(bone_names) < 2:
        return None
    if not np.isclose(offsets[1], 0.0, atol=1e-8).all():
        return None

    parents = np.asarray(parents, dtype=np.int32)
    offsets = np.asarray(offsets, dtype=np.float64)

    if int(np.count_nonzero(parents == 1)) == 0:
        new_names = list(bone_names)
        new_names[1] = new_names[0]
        new_offsets = offsets.copy()
        new_offsets[1] = new_offsets[0]
        new_parents = parents[1:].copy() - 1
        if len(new_parents) > 1:
            new_parents[1:][new_parents[1:] < 0] = 0
        return new_names[1:], new_parents, new_offsets[1:], "removed redundant joint #1"

    if int(np.count_nonzero(parents == 0)) == 1:
        new_offsets = offsets.copy()
        new_offsets[1] = new_offsets[0] + new_offsets[1]
        new_parents = parents[1:].copy() - 1
        return list(bone_names[1:]), new_parents, new_offsets[1:], "collapsed redundant wrapper root"

    return None


def _match_reference_skeleton_to_joint_count(
    reference: ReferenceSkeleton,
    expected_joint_count: int,
) -> tuple[list[str], np.ndarray, np.ndarray, str | None]:
    bone_names = list(reference.bone_names)
    parents = np.asarray(reference.parents, dtype=np.int32)
    offsets = np.asarray(reference.offsets, dtype=np.float64)
    steps: list[str] = []

    while len(bone_names) > expected_joint_count:
        collapsed = _collapse_redundant_root_once(bone_names, parents, offsets)
        if collapsed is None:
            break
        bone_names, parents, offsets, step_note = collapsed
        steps.append(step_note)

    if len(bone_names) != expected_joint_count:
        raise ValueError(
            f"Motion joint count mismatch: features J={expected_joint_count}, reference joints={len(reference.bone_names)}"
        )

    note = None
    if steps:
        note = "; ".join(steps)
    return bone_names, parents, offsets, note


def _resolve_npy_motion_skeleton(
    raw_payload: Any,
    reference: ReferenceSkeleton,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, str | None]:
    features, payload = coerce_feature_payload(raw_payload)

    joint_names = list(reference.bone_names)
    parents = np.asarray(reference.parents, dtype=np.int32)
    offsets = np.asarray(reference.offsets, dtype=np.float64)
    fallback_note = None

    if isinstance(payload, dict):
        payload_names = payload.get("bone_names")
        if payload_names is None:
            payload_names = payload.get("joints_names")
        if payload_names is not None and "parents" in payload and "offsets" in payload:
            joint_names = [str(name) for name in payload_names]
            parents = np.asarray(payload["parents"], dtype=np.int32)
            offsets = np.asarray(payload["offsets"], dtype=np.float64)

    if features.ndim != 3:
        raise ValueError(f"Expected NPY features with shape (F, J, C), got {features.shape}")
    if len(joint_names) != features.shape[1]:
        if isinstance(payload, dict):
            raise ValueError(
                f"Motion joint count mismatch: features J={features.shape[1]}, skeleton joints={len(joint_names)}"
            )
        joint_names, parents, offsets, fallback_note = _match_reference_skeleton_to_joint_count(
            reference,
            features.shape[1],
        )
    if parents.shape != (len(joint_names),):
        raise ValueError(
            f"Motion parents shape mismatch: expected ({len(joint_names)},), got {parents.shape}"
        )
    if offsets.shape != (len(joint_names), 3):
        raise ValueError(
            f"Motion offsets shape mismatch: expected ({len(joint_names)}, 3), got {offsets.shape}"
        )

    return features, joint_names, parents, offsets, fallback_note


def _load_npy_motion(npy_path: str, reference: ReferenceSkeleton) -> MotionWorldData:
    raw = _load_npy_payload(npy_path)
    features, bone_names, parents, offsets, fallback_note = _resolve_npy_motion_skeleton(raw, reference)

    if fallback_note is not None:
        print(
            f"[Info] NPY skeleton fallback: {fallback_note}; using {len(bone_names)} joints to match the motion tensor"
        )

    recovered_anim, _has_animated_pos = recover_from_features(raw, parents, offsets)
    world_positions = np.asarray(positions_global(recovered_anim), dtype=np.float64)

    if world_positions.shape[:2] != features.shape[:2]:
        raise ValueError(
            f"Recovered world positions shape mismatch: expected {features.shape[:2]}, got {world_positions.shape[:2]}"
        )

    sample_frames = [float(frame_idx) for frame_idx in range(world_positions.shape[0])]
    return MotionWorldData(
        file_path=os.path.abspath(npy_path),
        file_format="npy",
        bone_names=bone_names,
        parents=parents,
        world_positions=world_positions,
        sample_frames=sample_frames,
    )


def _collect_armature_world_positions(armature, sample_frames: list[float], bone_names: list[str]) -> np.ndarray:
    import bpy
    from mathutils import Vector

    scene = bpy.context.scene
    num_frames = len(sample_frames)
    num_joints = len(bone_names)
    world_positions = np.zeros((num_frames, num_joints, 3), dtype=np.float64)

    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="OBJECT")

    for frame_idx, sample_frame in enumerate(sample_frames):
        _set_scene_time(scene, float(sample_frame))
        bpy.context.view_layer.update()
        for bone_idx, bone_name in enumerate(bone_names):
            pose_bone = armature.pose.bones.get(bone_name)
            if pose_bone is None:
                continue
            head_local = pose_bone.head
            head_world = armature.matrix_world @ Vector((head_local.x, head_local.y, head_local.z))
            world_positions[frame_idx, bone_idx] = (head_world.x, head_world.y, head_world.z)

    return world_positions


def _load_glb_motion(glb_path: str) -> MotionWorldData:
    import bpy

    clear_scene()

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        bpy.ops.import_scene.gltf(filepath=glb_path)

    remove_lights_and_cameras()

    armature = next((obj for obj in bpy.data.objects if obj.type == "ARMATURE"), None)
    if armature is None:
        raise RuntimeError(f"No armature found in GLB: {glb_path}")

    bone_names, parents, _offsets, _rest_rotations = _extract_armature_skeleton_data(armature)
    sample_frames = [float(frame) for frame in _get_action_sample_times(armature)]
    if not sample_frames:
        sample_frames = [float(bpy.context.scene.frame_current)]

    world_positions = _collect_armature_world_positions(armature, sample_frames, bone_names)
    return MotionWorldData(
        file_path=os.path.abspath(glb_path),
        file_format="glb",
        bone_names=bone_names,
        parents=np.asarray(parents, dtype=np.int32),
        world_positions=world_positions,
        sample_frames=sample_frames,
    )


def _load_motion(input_path: str, reference: ReferenceSkeleton) -> MotionWorldData:
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".npy":
        return _load_npy_motion(input_path, reference)
    if ext in {".glb", ".gltf"}:
        return _load_glb_motion(input_path)
    raise ValueError(f"Unsupported input format: {ext} (supported: .npy, .glb, .gltf)")


def _build_canonical_index(names: list[str]) -> dict[str, int]:
    return {_canonical_bone_name(name): idx for idx, name in enumerate(names)}


def _compute_reference_rest_positions(offsets: np.ndarray, parents: np.ndarray) -> np.ndarray:
    positions = np.zeros((len(parents), 3), dtype=np.float64)
    for joint_idx, parent_idx in enumerate(parents):
        if parent_idx >= 0:
            positions[joint_idx] = positions[parent_idx] + offsets[joint_idx]
        else:
            positions[joint_idx] = offsets[joint_idx]
    return positions


def _resolve_comparison_edges(
    reference: ReferenceSkeleton,
    motion: MotionWorldData,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    motion_index = _build_canonical_index(motion.bone_names)
    reference_canon = [_canonical_bone_name(name) for name in reference.bone_names]
    positive_reference_lengths = np.linalg.norm(
        reference.offsets[np.asarray(reference.parents) >= 0],
        axis=-1,
    )
    positive_reference_lengths = positive_reference_lengths[np.isfinite(positive_reference_lengths)]
    if positive_reference_lengths.size > 0:
        mean_reference_length = float(np.mean(positive_reference_lengths))
        min_reference_length = max(mean_reference_length * 0.1, 1e-8)
    else:
        min_reference_length = 1e-8

    edge_names: list[str] = []
    motion_parent_indices: list[int] = []
    motion_child_indices: list[int] = []
    reference_lengths: list[float] = []

    for child_idx, parent_idx in enumerate(reference.parents):
        if parent_idx < 0:
            continue

        ref_length = float(np.linalg.norm(reference.offsets[child_idx]))
        if not np.isfinite(ref_length) or ref_length <= min_reference_length:
            continue

        child_canon = reference_canon[child_idx]
        parent_canon = reference_canon[parent_idx]
        motion_child_idx = motion_index.get(child_canon)
        motion_parent_idx = motion_index.get(parent_canon)
        if motion_child_idx is None or motion_parent_idx is None:
            continue

        edge_names.append(reference.bone_names[child_idx])
        motion_parent_indices.append(int(motion_parent_idx))
        motion_child_indices.append(int(motion_child_idx))
        reference_lengths.append(ref_length)

    if not edge_names:
        raise RuntimeError("No common parent-child bones found between the input motion and the cond.npy reference")

    return (
        edge_names,
        np.asarray(motion_parent_indices, dtype=np.int32),
        np.asarray(motion_child_indices, dtype=np.int32),
        np.asarray(reference_lengths, dtype=np.float64),
    )


def _estimate_global_scale(motion_lengths: np.ndarray, reference_lengths: np.ndarray) -> float:
    per_bone_mean = np.asarray(np.mean(motion_lengths, axis=0), dtype=np.float64)
    valid_mask = np.isfinite(per_bone_mean) & np.isfinite(reference_lengths) & (per_bone_mean > 1e-8) & (reference_lengths > 1e-8)
    if not np.any(valid_mask):
        return 1.0

    mean_motion = float(np.mean(per_bone_mean[valid_mask]))
    mean_reference = float(np.mean(reference_lengths[valid_mask]))
    if mean_motion <= 1e-8 or mean_reference <= 1e-8:
        return 1.0

    scale = mean_reference / mean_motion
    if not np.isfinite(scale) or scale <= 0.0:
        return 1.0
    return float(scale)


def _summarize_length_drift(
    edge_names: list[str],
    baseline_lengths: np.ndarray,
    measured_lengths: np.ndarray,
    sample_frames: list[float],
    *,
    note: str,
) -> dict[str, Any]:
    baseline_lengths = np.asarray(baseline_lengths, dtype=np.float64)
    measured_lengths = np.asarray(measured_lengths, dtype=np.float64)

    if baseline_lengths.shape != (len(edge_names),):
        raise ValueError(
            f"baseline_lengths shape mismatch: expected ({len(edge_names)},), got {baseline_lengths.shape}"
        )
    if measured_lengths.ndim != 2 or measured_lengths.shape[1] != len(edge_names):
        raise ValueError(
            f"measured_lengths shape mismatch: expected (F, {len(edge_names)}), got {measured_lengths.shape}"
        )
    if len(sample_frames) != measured_lengths.shape[0]:
        raise ValueError(
            f"sample_frames length mismatch: expected {measured_lengths.shape[0]}, got {len(sample_frames)}"
        )

    drift_ratio = measured_lengths / baseline_lengths[np.newaxis, :] - 1.0
    drift_pct = drift_ratio * 100.0
    abs_drift_pct = np.abs(drift_pct)

    worst_flat_idx = int(np.argmax(abs_drift_pct))
    worst_frame_idx, worst_bone_idx = np.unravel_index(worst_flat_idx, abs_drift_pct.shape)

    per_bone_max_abs = np.max(abs_drift_pct, axis=0)
    per_bone_mean_abs = np.mean(abs_drift_pct, axis=0)
    per_bone_mean_signed = np.mean(drift_pct, axis=0)
    per_bone_max_stretch = np.max(drift_pct, axis=0)
    per_bone_max_compress = np.min(drift_pct, axis=0)
    per_bone_mean_length = np.mean(measured_lengths, axis=0)
    per_frame_max_abs = np.max(abs_drift_pct, axis=1)

    worst_order = np.argsort(per_bone_max_abs)[::-1]
    top_bones = []
    for bone_idx in worst_order[: min(10, len(edge_names))]:
        top_bones.append(
            {
                "name": edge_names[bone_idx],
                "baseline_length": float(baseline_lengths[bone_idx]),
                "mean_length": float(per_bone_mean_length[bone_idx]),
                "max_abs_drift_pct": float(per_bone_max_abs[bone_idx]),
                "mean_abs_drift_pct": float(per_bone_mean_abs[bone_idx]),
                "mean_signed_drift_pct": float(per_bone_mean_signed[bone_idx]),
                "max_stretch_pct": float(per_bone_max_stretch[bone_idx]),
                "max_compress_pct": float(per_bone_max_compress[bone_idx]),
            }
        )

    return {
        "note": note,
        "max_abs_drift_pct": float(np.max(abs_drift_pct)),
        "mean_abs_drift_pct": float(np.mean(abs_drift_pct)),
        "median_abs_drift_pct": float(np.median(abs_drift_pct)),
        "p95_abs_drift_pct": float(np.quantile(abs_drift_pct, 0.95)),
        "max_stretch_pct": float(np.max(drift_pct)),
        "max_compress_pct": float(np.min(drift_pct)),
        "worst_bone": edge_names[worst_bone_idx],
        "worst_frame_index": int(worst_frame_idx),
        "worst_frame_value": float(sample_frames[worst_frame_idx]),
        "worst_value_pct": float(drift_pct[worst_frame_idx, worst_bone_idx]),
        "per_frame_max_abs_drift_pct": per_frame_max_abs.tolist(),
        "per_bone": {
            "names": edge_names,
            "baseline_length": baseline_lengths.tolist(),
            "mean_length": per_bone_mean_length.tolist(),
            "max_abs_drift_pct": per_bone_max_abs.tolist(),
            "mean_abs_drift_pct": per_bone_mean_abs.tolist(),
            "mean_signed_drift_pct": per_bone_mean_signed.tolist(),
            "max_stretch_pct": per_bone_max_stretch.tolist(),
            "max_compress_pct": per_bone_max_compress.tolist(),
        },
        "top_worst_bones": top_bones,
    }


def _compute_drift_report(reference: ReferenceSkeleton, motion: MotionWorldData) -> dict[str, Any]:
    edge_names, motion_parent_idx, motion_child_idx, tpose_lengths = _resolve_comparison_edges(reference, motion)

    motion_lengths = np.linalg.norm(
        motion.world_positions[:, motion_child_idx, :] - motion.world_positions[:, motion_parent_idx, :],
        axis=-1,
    )

    if motion_lengths.shape[0] == 0:
        raise RuntimeError("Input motion did not contain any sampled frames")

    first_frame_lengths = np.asarray(motion_lengths[0], dtype=np.float64)
    valid_mask = np.isfinite(first_frame_lengths) & (first_frame_lengths > 1e-8)
    if not np.any(valid_mask):
        raise RuntimeError("No comparable bones with non-zero length in the animation's first frame")

    dropped_bones = int(valid_mask.size - np.count_nonzero(valid_mask))
    if not np.all(valid_mask):
        valid_indices = np.flatnonzero(valid_mask)
        edge_names = [edge_names[int(index)] for index in valid_indices]
        motion_lengths = motion_lengths[:, valid_mask]
        tpose_lengths = tpose_lengths[valid_mask]
        first_frame_lengths = first_frame_lengths[valid_mask]

    drift = _summarize_length_drift(
        edge_names,
        first_frame_lengths,
        motion_lengths,
        motion.sample_frames,
        note=(
            "Positive values mean stretched longer than animation frame 0 for the same bone; "
            "negative values mean compressed shorter."
        ),
    )
    drift["reference_basis"] = "animation_first_frame"
    drift["baseline_frame_index"] = 0
    drift["baseline_frame_value"] = float(motion.sample_frames[0])

    first_frame_scale = _estimate_global_scale(first_frame_lengths[np.newaxis, :], tpose_lengths)
    first_frame_lengths_aligned = first_frame_lengths * first_frame_scale
    tpose_reference = _summarize_length_drift(
        edge_names,
        tpose_lengths,
        first_frame_lengths_aligned[np.newaxis, :],
        [motion.sample_frames[0]],
        note=(
            "Reference-only comparison between cond.npy T-pose lengths and animation frame 0 "
            "after a single global scale alignment."
        ),
    )
    tpose_reference["reference_basis"] = "cond_tpose"
    tpose_reference["measured_basis"] = "animation_first_frame"
    tpose_reference["frame_index"] = 0
    tpose_reference["frame_value"] = float(motion.sample_frames[0])
    tpose_reference["global_scale"] = float(first_frame_scale)
    tpose_reference["mean_tpose_length"] = float(np.mean(tpose_lengths))
    tpose_reference["mean_first_frame_length_before"] = float(np.mean(first_frame_lengths))
    tpose_reference["mean_first_frame_length_after"] = float(np.mean(first_frame_lengths_aligned))
    tpose_reference["per_bone"]["tpose_length"] = tpose_lengths.tolist()
    tpose_reference["per_bone"]["first_frame_length_before_scale"] = first_frame_lengths.tolist()
    tpose_reference["per_bone"]["first_frame_length_after_scale"] = first_frame_lengths_aligned.tolist()

    return {
        "comparison": {
            "object_type": reference.object_type,
            "reference_bones": len(reference.bone_names),
            "motion_bones": motion.num_joints,
            "compared_bones": len(edge_names),
            "compared_frames": motion.num_frames,
            "dropped_zero_first_frame_bones": dropped_bones,
        },
        "first_frame_reference": {
            "frame_index": 0,
            "frame_value": float(motion.sample_frames[0]),
            "mean_bone_length": float(np.mean(first_frame_lengths)),
        },
        "drift": drift,
        "tpose_reference": tpose_reference,
    }


def _format_path(path: str) -> str:
    if not os.path.isabs(path):
        return path
    try:
        return os.path.relpath(path, REPO_ROOT)
    except ValueError:
        return path


def _print_summary(motion: MotionWorldData, report: dict[str, Any], top_k: int) -> None:
    comparison = report["comparison"]
    first_frame_reference = report["first_frame_reference"]
    drift = report["drift"]
    tpose_reference = report["tpose_reference"]

    print(f"[Input] {_format_path(motion.file_path)}")
    print(
        f"[Reference] object_type={comparison['object_type']}  bones={comparison['reference_bones']}  "
        f"compared_bones={comparison['compared_bones']}"
    )
    if comparison.get("dropped_zero_first_frame_bones", 0) > 0:
        print(
            f"           dropped_zero_first_frame_bones={comparison['dropped_zero_first_frame_bones']}"
        )
    print()

    print("[Frame-to-Frame Bone Length Drift]")
    print(f"  max_abs    = {drift['max_abs_drift_pct']:.2f}%")
    print(f"  mean_abs   = {drift['mean_abs_drift_pct']:.2f}%")
    print(f"  median_abs = {drift['median_abs_drift_pct']:.2f}%")
    print(f"  p95_abs    = {drift['p95_abs_drift_pct']:.2f}%")
    print(f"  stretch    = {drift['max_stretch_pct']:.2f}%")
    print(f"  compress   = {drift['max_compress_pct']:.2f}%")
    print(
        f"  worst      = bone={drift['worst_bone']}  frame_index={drift['worst_frame_index']}  "
        f"frame_value={drift['worst_frame_value']:.2f}  drift={drift['worst_value_pct']:.2f}%"
    )

    print()
    print("[T-pose vs First Frame Reference]")
    print(f"  max_abs                  = {tpose_reference['max_abs_drift_pct']:.2f}%")
    print(f"  mean_abs                 = {tpose_reference['mean_abs_drift_pct']:.2f}%")
    print(
        f"  worst                    = bone={tpose_reference['worst_bone']}  "
        f"drift={tpose_reference['worst_value_pct']:.2f}%"
    )

    top_bones = drift.get("top_worst_bones", [])
    if top_k > 0 and top_bones:
        print()
        print(f"[Top {min(top_k, len(top_bones))} Frame-to-Frame Worst Bones]")
        for rank, item in enumerate(top_bones[:top_k], start=1):
            print(
                f"  {rank:>2}. {item['name']}: max_abs={item['max_abs_drift_pct']:.2f}%  "
                f"mean_abs={item['mean_abs_drift_pct']:.2f}%  mean_signed={item['mean_signed_drift_pct']:.2f}%  "
                f"stretch={item['max_stretch_pct']:.2f}%  compress={item['max_compress_pct']:.2f}%"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure per-bone frame-to-frame bone-length drift using animation frame 0 as baseline, "
            "with a separate cond.npy T-pose reference-only comparison for NPY or GLB inputs."
        ),
    )
    parser.add_argument("--input", required=True, help="Path to the input motion (.npy or .glb/.gltf)")
    parser.add_argument(
        "--cond-npy",
        default=_DEFAULT_COND_NPY,
        help=f"Path to cond.npy. Default: {_DEFAULT_COND_NPY}",
    )
    parser.add_argument(
        "--object-type",
        default=None,
        help="Optional object_type override. By default the tool infers it from the input filename.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many worst bones to print in the console summary. Use 0 to suppress the list.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write the full JSON report.",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    cond_npy_path = os.path.abspath(args.cond_npy)

    cond = _load_cond_dict(cond_npy_path)
    object_type = _resolve_object_type(input_path, cond, args.object_type)
    reference = _load_reference_skeleton(object_type, cond[object_type])

    print(f"[Info] cond.npy     : {_format_path(cond_npy_path)}")
    print(f"[Info] object_type  : {object_type}")
    print(f"[Info] loading      : {_format_path(input_path)}")

    motion = _load_motion(input_path, reference)
    report = _compute_drift_report(reference, motion)

    print()
    _print_summary(motion, report, max(args.top_k, 0))

    if args.json_out:
        output_path = os.path.abspath(args.json_out)
        payload = {
            "input": {
                "path": input_path,
                "format": motion.file_format,
                "frames": motion.num_frames,
                "bones": motion.num_joints,
            },
            "reference": {
                "cond_npy": cond_npy_path,
                "object_type": object_type,
                "bones": len(reference.bone_names),
                "rest_positions": _compute_reference_rest_positions(reference.offsets, reference.parents).tolist(),
            },
            "report": report,
        }
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print()
        print(f"[Report] JSON saved -> {_format_path(output_path)}")


if __name__ == "__main__":
    main()