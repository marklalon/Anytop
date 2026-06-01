"""Regression checks for loop padding.

Usage:
    d:/AI/pcvg-skeleton-animation/.venv/Scripts/python.exe tests/test_dataset_loop_regression.py

This script verifies loop padding behavior.
"""

from __future__ import annotations

import glob
import os
import sys
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_loaders.truebones.data.dataset as dataset_module
from data_loaders.tensors import truebones_batch_collate
from data_loaders.truebones.data.dataset import (
    Truebones,
    _circular_roll_motion,
    resample_motion_features,
    _tile_loop_motion,
)
from data_loaders.truebones.truebones_utils.get_opt import get_opt
from data_loaders.truebones.truebones_utils.motion_process import infer_translation_root_index_from_features
from model.anytop import GlobalEnergyExtractor


def _find_motion(pattern: str) -> str:
    """Find a motion file by glob pattern (avoids hard-coded index)."""
    opt = get_opt(None)
    motion_dir = opt.motion_dir
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(motion_dir):
        motion_dir = os.path.join(repo_root, motion_dir)
    files = sorted(glob.glob(os.path.join(motion_dir, pattern)))
    assert files, f"No files matching '{pattern}' in {motion_dir}"
    return os.path.basename(files[0])


LOOP_MOTION = _find_motion("Ostrich_Run_*.npy")
LOOP_SUBSET = "bipeds_clean"
NUM_FRAMES = 60
_ENRICHED_MOTION_METADATA_LOOKUP = None


def assert_close(name: str, actual: np.ndarray, expected: np.ndarray, atol: float = 1e-6) -> None:
    max_diff = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    assert np.allclose(actual, expected, atol=atol), f"{name} mismatch: max_diff={max_diff}"


def _normalize_motion(raw: np.ndarray, cond: dict[str, np.ndarray]) -> np.ndarray:
    return np.nan_to_num((raw - cond["mean"][None, :]) / cond["std_safe"][None, :]).astype(np.float32, copy=False)


def _expected_resampled_velocity(raw: np.ndarray, target_frames: int, *, loop_terminal: bool = False) -> np.ndarray:
    source_frames = int(raw.shape[0])
    src = np.linspace(0.0, float(source_frames - 1), target_frames, endpoint=True, dtype=np.float32)
    lo = np.floor(src).astype(np.int64).clip(0, source_frames - 1)
    hi = np.minimum(lo + 1, source_frames - 1)
    w = (src - np.floor(src))[:, None, None].astype(np.float32)

    source_velocity_path = np.zeros((source_frames, raw.shape[1], 3), dtype=np.float32)
    if source_frames > 1:
        source_velocity_path[1:] = np.cumsum(
            raw[:-1, :, 9:12].astype(np.float64, copy=False),
            axis=0,
        ).astype(np.float32, copy=False)
    resampled_velocity_path = source_velocity_path[lo] * (1.0 - w) + source_velocity_path[hi] * w

    if target_frames <= 1:
        return np.zeros((target_frames, raw.shape[1], 3), dtype=np.float32)

    step_scale = float(source_frames - 1) / float(target_frames - 1)
    expected_velocity = np.zeros((target_frames, raw.shape[1], 3), dtype=np.float32)
    expected_velocity[:-1] = (resampled_velocity_path[1:] - resampled_velocity_path[:-1]) / step_scale
    if loop_terminal:
        expected_velocity[-1] = (resampled_velocity_path[0] - resampled_velocity_path[-1]) / step_scale
    else:
        expected_velocity[-1] = expected_velocity[-2]
    return expected_velocity


def _resample_raw_then_normalize(
    raw: np.ndarray,
    cond: dict[str, np.ndarray],
    target_frames: int,
    *,
    loop_terminal: bool = False,
) -> np.ndarray:
    resampled = resample_motion_features(raw, target_frames, loop_terminal=loop_terminal)
    return _normalize_motion(resampled, cond)


def _get_enriched_motion_metadata_lookup() -> dict[str, dict[str, object]]:
    global _ENRICHED_MOTION_METADATA_LOOKUP
    if _ENRICHED_MOTION_METADATA_LOOKUP is not None:
        return {name: dict(metadata) for name, metadata in _ENRICHED_MOTION_METADATA_LOOKUP.items()}

    opt = get_opt(None)
    data_root = opt.data_root
    motion_dir = opt.motion_dir
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(data_root):
        data_root = os.path.join(repo_root, data_root)
    if not os.path.isabs(motion_dir):
        motion_dir = os.path.join(repo_root, motion_dir)

    cond_dict = np.load(opt.cond_file, allow_pickle=True).item()
    motion_metadata_lookup = dataset_module.load_motion_metadata(data_root)
    enriched_lookup = {name: dict(metadata) for name, metadata in motion_metadata_lookup.items()}
    for motion_name, motion_metadata in enriched_lookup.items():
        if 'translation_root_index' in motion_metadata:
            continue
        object_type = str(motion_metadata['object_type'])
        motion = np.load(os.path.join(motion_dir, motion_name)).astype(np.float32, copy=False)
        motion_metadata['translation_root_index'] = infer_translation_root_index_from_features(
            motion,
            cond_dict[object_type]['parents'],
            cond_dict[object_type]['offsets'],
        )

    _ENRICHED_MOTION_METADATA_LOOKUP = enriched_lookup
    return {name: dict(metadata) for name, metadata in enriched_lookup.items()}


def _build_truebones(**kwargs) -> Truebones:
    enriched_lookup = _get_enriched_motion_metadata_lookup()
    with patch.object(dataset_module, 'load_motion_metadata', return_value=enriched_lookup):
        return Truebones(**kwargs)


def test_speed_resample_preserves_velocity_and_keeps_contact_binary() -> None:
    source = np.zeros((4, 2, 13), dtype=np.float32)
    source[:, :, 0] = np.array([0.0, 1.0, 3.0, 6.0], dtype=np.float32)[:, None]
    source[:, :, 1] = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float32)[:, None]
    source[:, :, 2] = np.array([0.0, -1.0, -1.5, -2.0], dtype=np.float32)[:, None]
    source[:, :, 9] = np.array([0.0, 2.0, 4.0, 8.0], dtype=np.float32)[:, None]
    source[:, :, 10] = 3.0
    source[:, :, 11] = 0.0
    source[:, :, 12] = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)[:, None]

    resampled = resample_motion_features(source, 7)

    expected_vel = _expected_resampled_velocity(source, 7)

    assert_close("resampled velocity", resampled[:, :, 9:12], expected_vel)
    assert_close("zero velocity channel", resampled[:, :, 11], np.zeros_like(resampled[:, :, 11]))
    assert set(np.unique(resampled[:, :, 12]).tolist()).issubset({0.0, 1.0})


def test_speed_resample_preserves_global_energy_magnitude() -> None:
    source = np.zeros((4, 2, 13), dtype=np.float32)
    source[:, :, 9:12] = np.array([1.25, 0.0, 0.0], dtype=np.float32)

    resampled = resample_motion_features(source, 7)
    source_tensor = torch.from_numpy(source).permute(1, 2, 0).unsqueeze(0)
    resampled_tensor = torch.from_numpy(resampled).permute(1, 2, 0).unsqueeze(0)

    source_energy = GlobalEnergyExtractor.compute_global_energy_condition(source_tensor, torch.tensor([2]))
    resampled_energy = GlobalEnergyExtractor.compute_global_energy_condition(resampled_tensor, torch.tensor([2]))

    assert_close("global energy after window stretch", resampled_energy.numpy(), source_energy.numpy())


def test_speed_resample_preserves_rotation_only_global_energy_with_playspeed() -> None:
    source = np.zeros((4, 2, 13), dtype=np.float32)
    source[:, :, 3] = np.arange(4, dtype=np.float32)[:, None]

    resampled = resample_motion_features(source, 7)
    source_tensor = torch.from_numpy(source).permute(1, 2, 0).unsqueeze(0)
    resampled_tensor = torch.from_numpy(resampled).permute(1, 2, 0).unsqueeze(0)

    source_energy = GlobalEnergyExtractor.compute_global_energy_condition(source_tensor, torch.tensor([2]))
    resampled_energy = GlobalEnergyExtractor.compute_global_energy_condition(
        resampled_tensor,
        torch.tensor([2]),
        playspeed_cond=torch.tensor([4.0 / 7.0], dtype=torch.float32),
    )

    assert_close("rotation-only energy after window stretch", resampled_energy.numpy(), source_energy.numpy(), atol=1e-5)


def test_loop_speed_resample_rebuilds_terminal_velocity_from_wrap_delta() -> None:
    source = np.zeros((4, 1, 13), dtype=np.float32)
    source[:, 0, 0:3] = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [4.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    source[:, 0, 9:12] = np.array(
        [[0.5, 0.0, 0.0], [0.5, 0.5, 0.0], [1.0, 0.5, 0.5], [-2.0, -1.0, -0.5]],
        dtype=np.float32,
    )

    resampled = resample_motion_features(source, 6, loop_terminal=True)
    expected_vel = _expected_resampled_velocity(source, 6, loop_terminal=True)

    assert_close("loop visible velocity", resampled[:-1, :, 9:12], expected_vel[:-1])
    assert_close("loop terminal velocity", resampled[-1, :, 9:12], expected_vel[-1])


def test_loop_padding_updates_effective_length() -> None:
    dataset = _build_truebones(
        split="train",
        temporal_window=31,
        num_frames=NUM_FRAMES,
        balanced=False,
        objects_subset=LOOP_SUBSET,
        motion_cache_size=2,
    )

    motion_dataset = dataset.motion_dataset
    with patch.object(motion_dataset, '_sample_loop_tile_count', return_value=1):
        sample = motion_dataset.prepare_sample_by_name(LOOP_MOTION, target_num_frames=NUM_FRAMES, loop_offset=0)
    motion, m_length, *_rest, mean, std, _max_joints, motion_metadata, name, _joint_mask_dict = sample

    assert name == LOOP_MOTION, f"unexpected sample: {name}"
    assert bool(motion_metadata.get("is_loop", False)), "loop regression sample is no longer marked loop"
    assert motion.shape[0] == NUM_FRAMES, f"expected padded motion to have {NUM_FRAMES} frames"
    assert m_length == NUM_FRAMES, f"effective length should track loop-filled frames, got {m_length}"

    data = motion_dataset.data_dict[LOOP_MOTION]
    cond = motion_dataset.cond_dict[data["object_type"]]
    raw = np.load(data["motion_path"]).astype(np.float32, copy=False)
    raw_len = raw.shape[0]
    assert raw_len < NUM_FRAMES, "loop regression sample no longer needs padding"
    expected = _resample_raw_then_normalize(raw, cond, NUM_FRAMES, loop_terminal=True)

    assert np.isclose(float(motion_metadata["loop_phase_length"]), float(NUM_FRAMES))
    assert np.isclose(float(motion_metadata["playspeed_cond"]), float(raw_len) / float(NUM_FRAMES))
    assert_close("loop-filled motion", motion, expected, atol=3e-5)


def test_loop_padding_can_tile_multiple_cycles_before_resample() -> None:
    dataset = _build_truebones(
        split="train",
        temporal_window=31,
        num_frames=NUM_FRAMES,
        balanced=False,
        objects_subset=LOOP_SUBSET,
        motion_cache_size=2,
    )

    motion_dataset = dataset.motion_dataset
    data = motion_dataset.data_dict[LOOP_MOTION]
    cond = motion_dataset.cond_dict[data["object_type"]]
    raw = np.load(data["motion_path"]).astype(np.float32, copy=False)

    with patch.object(motion_dataset, '_sample_loop_tile_count', return_value=2):
        sample = motion_dataset.prepare_sample_by_name(LOOP_MOTION, target_num_frames=NUM_FRAMES, loop_offset=0)
    motion, m_length, *_rest, mean, std, _max_joints, motion_metadata, name, _joint_mask_dict = sample

    expected = _resample_raw_then_normalize(_tile_loop_motion(raw, 2), cond, NUM_FRAMES, loop_terminal=True)

    assert name == LOOP_MOTION, f"unexpected sample: {name}"
    assert motion.shape[0] == NUM_FRAMES
    assert m_length == NUM_FRAMES
    assert np.isclose(float(motion_metadata["playspeed_cond"]), float(raw.shape[0] * 2) / float(NUM_FRAMES))
    assert np.isclose(float(motion_metadata["loop_phase_length"]), ((float(NUM_FRAMES) - 1.0) / 2.0) + 1.0)
    assert_close("loop-filled tiled motion", motion, expected, atol=3e-5)


def test_loop_padding_random_offset_wraps_without_truncation() -> None:
    dataset = _build_truebones(
        split="train",
        temporal_window=31,
        num_frames=NUM_FRAMES,
        balanced=False,
        objects_subset=LOOP_SUBSET,
        motion_cache_size=2,
    )

    motion_dataset = dataset.motion_dataset
    data = motion_dataset.data_dict[LOOP_MOTION]
    cond = motion_dataset.cond_dict[data["object_type"]]
    raw = np.load(data["motion_path"]).astype(np.float32, copy=False)
    raw_len = raw.shape[0]
    offset = raw_len - 4

    with patch.object(motion_dataset, '_sample_loop_tile_count', return_value=1):
        sample = motion_dataset.prepare_sample_by_name(
            LOOP_MOTION,
            target_num_frames=NUM_FRAMES,
            loop_offset=offset,
        )
    motion, m_length, *_rest, mean, std, _max_joints, _motion_metadata, _name, _joint_mask_dict = sample
    rolled_raw = _circular_roll_motion(raw, offset)
    expected_raw = resample_motion_features(rolled_raw, NUM_FRAMES, loop_terminal=True)
    expected = _normalize_motion(expected_raw, cond)
    raw_motion = motion * std[None, :, :] + mean[None, :, :]
    assert motion.shape[0] == NUM_FRAMES, f"expected random-offset loop fill to keep {NUM_FRAMES} frames"
    assert m_length == NUM_FRAMES, f"effective length should remain {NUM_FRAMES}, got {m_length}"
    assert_close("loop-filled motion with wraparound offset", motion, expected, atol=3e-5)
    # Pure circular roll preserves the velocity ring; the terminal velocity
    # (channels 9-12, i.e. the wrap-around delta) must equal the expected
    # terminal velocity after resample — it is NOT forced to zero/copy.
    assert_close("rolled loop terminal velocity", raw_motion[-1, :, 9:12], expected_raw[-1, :, 9:12], atol=3e-5)


def test_explicit_window_start_respects_requested_crop() -> None:
    dataset = _build_truebones(
        split="train",
        temporal_window=31,
        num_frames=NUM_FRAMES,
        balanced=False,
        objects_subset=LOOP_SUBSET,
        motion_cache_size=2,
    )

    motion_dataset = dataset.motion_dataset
    window_start = 7
    # Pick a NON-loop motion: loop motions get circular-roll + tile augmentation
    # before the crop, so their window would not match a direct raw crop.
    long_motion_name = next(
        name
        for name, length in zip(motion_dataset.name_list, motion_dataset.length_arr)
        if NUM_FRAMES + window_start <= int(length) <= NUM_FRAMES * 2
        and not bool(motion_dataset.data_dict[name].get('motion_metadata', {}).get('is_loop'))
    )

    motion, m_length, *_rest, _motion_metadata, name, _joint_mask_dict = motion_dataset.prepare_sample_by_name(
        long_motion_name,
        target_num_frames=NUM_FRAMES,
        crop_start=window_start,
    )

    data = motion_dataset.data_dict[long_motion_name]
    cond = motion_dataset.cond_dict[data["object_type"]]
    raw = np.load(data["motion_path"]).astype(np.float32, copy=False)
    expected = _normalize_motion(raw[window_start:window_start + NUM_FRAMES], cond)

    assert name == long_motion_name, f"unexpected cropped sample: {name}"
    assert m_length == NUM_FRAMES, f"cropped sample should have effective length {NUM_FRAMES}, got {m_length}"
    assert_close("explicit crop window", motion, expected)


def test_prepare_sample_aug_info_reports_actual_loop_fill() -> None:
    dataset = _build_truebones(
        split="train",
        temporal_window=31,
        num_frames=NUM_FRAMES,
        balanced=False,
        objects_subset=LOOP_SUBSET,
        motion_cache_size=2,
    )

    motion_dataset = dataset.motion_dataset

    with patch.object(motion_dataset, '_sample_loop_tile_count', return_value=1):
        sample = motion_dataset._prepare_sample(
            LOOP_MOTION,
            motion_dataset.data_dict[LOOP_MOTION],
            target_num_frames=NUM_FRAMES,
            loop_offset=0,
            return_aug_info=True,
        )
    motion, m_length, *_rest, motion_metadata, name, _joint_mask_dict, aug_info = sample

    assert name == LOOP_MOTION, f"unexpected sample: {name}"
    assert bool(motion_metadata.get("is_loop", False)), "loop regression sample is no longer marked loop"
    assert motion.shape[0] == NUM_FRAMES, f"expected loop-filled motion to have {NUM_FRAMES} frames"
    assert m_length == NUM_FRAMES, f"expected effective length {NUM_FRAMES}, got {m_length}"
    assert aug_info["loop_applied"] is True, f"expected loop_applied=True, got {aug_info}"
    assert aug_info["loop_phase_offset"] == 0, f"expected loop_phase_offset=0, got {aug_info}"
    assert aug_info["loop_tile_count"] == 1, f"expected loop_tile_count=1, got {aug_info}"
    assert np.isclose(float(aug_info["playspeed_cond"]), float(motion_dataset.data_dict[LOOP_MOTION]["length"]) / float(NUM_FRAMES))
    assert aug_info["crop_start"] == 0, f"expected crop_start=0, got {aug_info}"


def test_loop_uncond_keeps_legacy_loop_tile_but_non_loop_metadata() -> None:
    dataset = _build_truebones(
        split="train",
        temporal_window=31,
        num_frames=NUM_FRAMES,
        balanced=False,
        objects_subset=LOOP_SUBSET,
        motion_cache_size=2,
        loop_cond_prob=0.0,
    )

    motion_dataset = dataset.motion_dataset

    with patch.object(dataset_module.random, 'randint', return_value=0):
        sample = motion_dataset._prepare_sample(
            LOOP_MOTION,
            motion_dataset.data_dict[LOOP_MOTION],
            target_num_frames=NUM_FRAMES,
            return_aug_info=True,
        )
    motion, m_length, *_rest, motion_metadata, name, _joint_mask_dict, aug_info = sample

    data = motion_dataset.data_dict[LOOP_MOTION]
    cond = motion_dataset.cond_dict[data["object_type"]]
    raw = np.load(data["motion_path"]).astype(np.float32, copy=False)
    expected = _resample_raw_then_normalize(raw, cond, NUM_FRAMES, loop_terminal=True)

    assert name == LOOP_MOTION, f"unexpected sample: {name}"
    assert motion.shape[0] == NUM_FRAMES
    assert m_length == NUM_FRAMES
    assert motion_metadata["is_loop"] is False
    assert motion_metadata["loop_full_cycle"] is False
    assert aug_info["loop_applied"] is False
    assert aug_info["loop_uncond"] is True
    assert_close("loop uncond resample", motion, expected, atol=3e-5)


def test_loop_uncond_long_loop_rolls_then_crops(tmp_path) -> None:
    dataset = _build_truebones(
        split="train",
        temporal_window=31,
        num_frames=NUM_FRAMES,
        balanced=False,
        objects_subset=LOOP_SUBSET,
        motion_cache_size=0,
        loop_cond_prob=0.0,
    )

    motion_dataset = dataset.motion_dataset

    source_data = motion_dataset.data_dict[LOOP_MOTION]
    source_raw = np.load(source_data["motion_path"]).astype(np.float32, copy=False)
    repeat_count = (NUM_FRAMES * 2 + 8 + source_raw.shape[0] - 1) // source_raw.shape[0]
    long_raw = np.tile(source_raw, (repeat_count, 1, 1))[:NUM_FRAMES * 2 + 8]
    motion_path = tmp_path / "long_loop.npy"
    np.save(motion_path, long_raw.astype(np.float32, copy=False))

    long_data = dict(source_data)
    long_data["motion_path"] = str(motion_path)
    long_data["length"] = long_raw.shape[0]
    long_data["motion_metadata"] = dict(source_data["motion_metadata"])
    long_data["motion_metadata"]["is_loop"] = True

    crop_start = 5
    with patch.object(dataset_module.random, 'randint', return_value=NUM_FRAMES):
        sample = motion_dataset._prepare_sample(
            "synthetic_long_loop.npy",
            long_data,
            target_num_frames=NUM_FRAMES,
            crop_start=crop_start,
            return_aug_info=True,
        )
    motion, m_length, *_rest, motion_metadata, _name, _joint_mask_dict, aug_info = sample

    cond = motion_dataset.cond_dict[long_data["object_type"]]
    expected_augmented = _circular_roll_motion(long_raw, NUM_FRAMES)
    expected = _normalize_motion(expected_augmented[crop_start:crop_start + NUM_FRAMES], cond)

    assert motion.shape[0] == NUM_FRAMES
    assert m_length == NUM_FRAMES
    assert motion_metadata["is_loop"] is False
    assert motion_metadata["loop_full_cycle"] is False
    assert aug_info["loop_applied"] is False
    assert aug_info["loop_uncond"] is True
    assert_close("loop uncond long crop", motion, expected)


def test_loop_conditioned_long_loop_downgrades_to_non_loop(tmp_path) -> None:
    dataset = _build_truebones(
        split="train",
        temporal_window=31,
        num_frames=NUM_FRAMES,
        balanced=False,
        objects_subset=LOOP_SUBSET,
        motion_cache_size=0,
        loop_cond_prob=1.0,
    )

    motion_dataset = dataset.motion_dataset

    source_data = motion_dataset.data_dict[LOOP_MOTION]
    source_raw = np.load(source_data["motion_path"]).astype(np.float32, copy=False)
    repeat_count = (NUM_FRAMES * 2 + 8 + source_raw.shape[0] - 1) // source_raw.shape[0]
    long_raw = np.tile(source_raw, (repeat_count, 1, 1))[:NUM_FRAMES * 2 + 8]
    motion_path = tmp_path / "conditioned_long_loop.npy"
    np.save(motion_path, long_raw.astype(np.float32, copy=False))

    long_data = dict(source_data)
    long_data["motion_path"] = str(motion_path)
    long_data["length"] = long_raw.shape[0]
    long_data["motion_metadata"] = dict(source_data["motion_metadata"])
    long_data["motion_metadata"]["is_loop"] = True

    crop_start = 5
    with patch.object(dataset_module.random, 'randint', return_value=NUM_FRAMES):
        sample = motion_dataset._prepare_sample(
            "synthetic_conditioned_long_loop.npy",
            long_data,
            target_num_frames=NUM_FRAMES,
            crop_start=crop_start,
            return_aug_info=True,
        )
    motion, m_length, *_rest, motion_metadata, _name, _joint_mask_dict, aug_info = sample

    cond = motion_dataset.cond_dict[long_data["object_type"]]
    expected_augmented = _circular_roll_motion(long_raw, NUM_FRAMES)
    expected = _normalize_motion(expected_augmented[crop_start:crop_start + NUM_FRAMES], cond)

    assert motion.shape[0] == NUM_FRAMES
    assert m_length == NUM_FRAMES
    assert motion_metadata["is_loop"] is False
    assert motion_metadata["loop_full_cycle"] is False
    assert motion_metadata["loop_phase_length"] == float(NUM_FRAMES)
    assert aug_info["loop_applied"] is False
    assert aug_info["loop_uncond"] is True
    assert np.isclose(float(aug_info["playspeed_cond"]), 1.0)
    assert_close("conditioned long loop downgraded crop", motion, expected)


def test_batch_collate_preserves_translation_root_index() -> None:
    dataset = _build_truebones(
        split="train",
        temporal_window=31,
        num_frames=NUM_FRAMES,
        balanced=False,
        objects_subset=LOOP_SUBSET,
        motion_cache_size=2,
    )

    motion_dataset = dataset.motion_dataset
    motion_dataset.data_dict[LOOP_MOTION]["motion_metadata"] = dict(
        motion_dataset.data_dict[LOOP_MOTION].get("motion_metadata") or {}
    )
    motion_dataset.data_dict[LOOP_MOTION]["motion_metadata"]["translation_root_index"] = 0

    sample = motion_dataset.prepare_sample_by_name(LOOP_MOTION, target_num_frames=NUM_FRAMES)
    _motion, cond = truebones_batch_collate([sample])

    assert int(cond["y"]["translation_root_index"][0]) == 0
    assert "playspeed_cond" in cond["y"]


def main() -> None:
    test_loop_padding_updates_effective_length()
    print("loop padding regression: ok")

    test_loop_padding_random_offset_wraps_without_truncation()
    print("loop random offset regression: ok")

    test_explicit_window_start_respects_requested_crop()
    print("explicit crop regression: ok")

    test_prepare_sample_aug_info_reports_actual_loop_fill()
    print("loop aug-info regression: ok")

    print("all regression checks passed")


if __name__ == "__main__":
    main()