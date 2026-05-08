"""Regression checks for loop padding and mirror augmentation.

Usage:
    d:/AI/pcvg-skeleton-animation/.venv/Scripts/python.exe tests/test_dataset_loop_and_mirror_regression.py

This script covers two previously broken behaviors:
1. Loop padding must update effective length to max_motion_length.
2. Mirror augmentation must run in raw feature space before normalization.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.truebones.data.dataset import Truebones


LOOP_MOTION = "Ostrich_Run_548.npy"
LOOP_SUBSET = "bipeds_clean"
MIRROR_MOTION = "Jaguar_Run_264.npy"
MIRROR_SUBSET = "quadropeds_clean"
MIRROR_SAFEGUARD_SUBSET = "all"
NUM_FRAMES = 60


def assert_close(name: str, actual: np.ndarray, expected: np.ndarray, atol: float = 1e-6) -> None:
    max_diff = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    assert np.allclose(actual, expected, atol=atol), f"{name} mismatch: max_diff={max_diff}"


def find_mirror_safeguard_sample(motion_dataset):
    for name in motion_dataset.name_list:
        data = motion_dataset.data_dict[name]
        cond = motion_dataset.cond_dict[data["object_type"]]
        if cond["mirror_disabled_joint_indices"]:
            return name, data, cond

    raise AssertionError(
        "no current dataset sample exercises mirror safeguards; "
        "pick a new regression object or replace this test with synthetic coverage"
    )


def test_loop_padding_updates_effective_length() -> None:
    dataset = Truebones(
        split="train",
        temporal_window=31,
        num_frames=NUM_FRAMES,
        balanced=False,
        objects_subset=LOOP_SUBSET,
        motion_cache_size=2,
    )

    motion_dataset = dataset.motion_dataset
    sample = motion_dataset.prepare_sample_by_name(LOOP_MOTION, target_num_frames=NUM_FRAMES, loop_offset=0)
    motion, m_length, *_rest, mean, std, _max_joints, motion_metadata, name = sample

    assert name == LOOP_MOTION, f"unexpected sample: {name}"
    assert bool(motion_metadata.get("is_loop", False)), "loop regression sample is no longer marked loop"
    assert motion.shape[0] == NUM_FRAMES, f"expected padded motion to have {NUM_FRAMES} frames"
    assert m_length == NUM_FRAMES, f"effective length should track loop-filled frames, got {m_length}"

    data = motion_dataset.data_dict[LOOP_MOTION]
    cond = motion_dataset.cond_dict[data["object_type"]]
    raw = np.load(data["motion_path"]).astype(np.float32, copy=False)
    raw_norm = np.nan_to_num((raw - cond["mean"][None, :]) / cond["std_safe"][None, :]).astype(np.float32, copy=False)
    raw_len = raw_norm.shape[0]
    assert raw_len < NUM_FRAMES, "loop regression sample no longer needs padding"

    expected = np.concatenate(
        [raw_norm, np.tile(raw_norm, ((NUM_FRAMES - raw_len) // raw_len + 1, 1, 1))[: NUM_FRAMES - raw_len]],
        axis=0,
    )
    assert_close("loop-filled motion", motion, expected)


def test_loop_padding_random_offset_wraps_without_truncation() -> None:
    dataset = Truebones(
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
    raw_norm = np.nan_to_num((raw - cond["mean"][None, :]) / cond["std_safe"][None, :]).astype(np.float32, copy=False)
    raw_len = raw_norm.shape[0]
    offset = raw_len - 4

    motion, m_length, *_rest = motion_dataset.prepare_sample_by_name(
        LOOP_MOTION,
        target_num_frames=NUM_FRAMES,
        loop_offset=offset,
    )

    expected = raw_norm[(np.arange(NUM_FRAMES, dtype=np.int64) + offset) % raw_len]
    assert motion.shape[0] == NUM_FRAMES, f"expected random-offset loop fill to keep {NUM_FRAMES} frames"
    assert m_length == NUM_FRAMES, f"effective length should remain {NUM_FRAMES}, got {m_length}"
    assert_close("loop-filled motion with wraparound offset", motion, expected)


def test_explicit_window_start_respects_requested_crop() -> None:
    dataset = Truebones(
        split="train",
        temporal_window=31,
        num_frames=NUM_FRAMES,
        balanced=False,
        objects_subset=LOOP_SUBSET,
        motion_cache_size=2,
    )

    motion_dataset = dataset.motion_dataset
    window_start = 7
    long_motion_name = next(
        name
        for name, length in zip(motion_dataset.name_list, motion_dataset.length_arr)
        if int(length) >= NUM_FRAMES + window_start
    )

    motion, m_length, *_rest, _motion_metadata, name = motion_dataset.prepare_sample_by_name(
        long_motion_name,
        target_num_frames=NUM_FRAMES,
        crop_start=window_start,
    )

    data = motion_dataset.data_dict[long_motion_name]
    cond = motion_dataset.cond_dict[data["object_type"]]
    raw = np.load(data["motion_path"]).astype(np.float32, copy=False)
    raw_norm = np.nan_to_num((raw - cond["mean"][None, :]) / cond["std_safe"][None, :]).astype(np.float32, copy=False)
    expected = raw_norm[window_start:window_start + NUM_FRAMES]

    assert name == long_motion_name, f"unexpected cropped sample: {name}"
    assert m_length == NUM_FRAMES, f"cropped sample should have effective length {NUM_FRAMES}, got {m_length}"
    assert_close("explicit crop window", motion, expected)


def test_prepare_sample_aug_info_reports_actual_loop_fill() -> None:
    dataset = Truebones(
        split="train",
        temporal_window=31,
        num_frames=NUM_FRAMES,
        balanced=False,
        objects_subset=LOOP_SUBSET,
        motion_cache_size=2,
    )

    motion_dataset = dataset.motion_dataset
    motion_dataset.opt.aug_mirror_prob = 0.0
    motion_dataset.opt.aug_speed_range = 0.0

    sample = motion_dataset._prepare_sample(
        LOOP_MOTION,
        motion_dataset.data_dict[LOOP_MOTION],
        target_num_frames=NUM_FRAMES,
        loop_offset=0,
        return_aug_info=True,
    )
    motion, m_length, *_rest, motion_metadata, name, aug_info = sample

    assert name == LOOP_MOTION, f"unexpected sample: {name}"
    assert bool(motion_metadata.get("is_loop", False)), "loop regression sample is no longer marked loop"
    assert motion.shape[0] == NUM_FRAMES, f"expected loop-filled motion to have {NUM_FRAMES} frames"
    assert m_length == NUM_FRAMES, f"expected effective length {NUM_FRAMES}, got {m_length}"
    assert aug_info["loop_applied"] is True, f"expected loop_applied=True, got {aug_info}"
    assert aug_info["crop_start"] == 0, f"expected crop_start=0, got {aug_info}"
    assert aug_info["mirror_applied"] is False, f"expected mirror_applied=False, got {aug_info}"
    assert np.isclose(float(aug_info["speed_factor"]), 1.0), f"expected speed_factor=1.0, got {aug_info}"


def test_mirror_augmentation_runs_before_normalization() -> None:
    dataset = Truebones(
        split="train",
        temporal_window=31,
        num_frames=NUM_FRAMES,
        balanced=False,
        objects_subset=MIRROR_SUBSET,

        motion_cache_size=4,
    )

    motion_dataset = dataset.motion_dataset
    motion_dataset.opt.aug_mirror_prob = 1.0
    motion_dataset.opt.aug_speed_range = 0.0

    data = motion_dataset.data_dict[MIRROR_MOTION]
    motion, m_length, object_type, _parents, _graph_dist, _joint_relations, tpos_first_frame, offsets, *_ = motion_dataset.augment(data)
    cond = motion_dataset.cond_dict[object_type]

    raw = np.load(data["motion_path"]).astype(np.float32, copy=False)
    spi = cond["symmetry_partner_indices"]
    perm = list(range(len(spi)))
    for joint_index, partner in enumerate(spi):
        if partner != -1:
            perm[joint_index] = int(partner)

    manual = raw[:, perm, :].copy()
    manual[:, :, [0, 4, 5, 6, 9]] *= -1
    manual = np.nan_to_num((manual - cond["mean"][None, :]) / cond["std_safe"][None, :]).astype(np.float32, copy=False)

    manual_tpose = np.asarray(cond["tpos_first_frame"], dtype=np.float32)[perm].copy()
    manual_tpose[:, [0, 4, 5, 6, 9]] *= -1
    manual_tpose = np.nan_to_num((manual_tpose - cond["mean"]) / cond["std_safe"]).astype(np.float32, copy=False)

    manual_offsets = np.asarray(cond["offsets"], dtype=np.float32)[perm].copy()
    manual_offsets[:, 0] *= -1

    assert m_length == raw.shape[0], f"mirror augmentation changed sequence length unexpectedly: {m_length}"
    assert_close("mirrored normalized motion", motion, manual)
    assert_close("mirrored normalized tpose", tpos_first_frame, manual_tpose)
    assert_close("mirrored offsets", offsets, manual_offsets)


def test_mirror_safeguards_handle_single_frame_tpose() -> None:
    dataset = Truebones(
        split="train",
        temporal_window=31,
        num_frames=NUM_FRAMES,
        balanced=False,
        objects_subset=MIRROR_SAFEGUARD_SUBSET,
        motion_cache_size=4,
    )

    motion_dataset = dataset.motion_dataset
    motion_dataset.opt.aug_mirror_prob = 1.0
    motion_dataset.opt.aug_speed_range = 0.0

    sample_name, data, cond = find_mirror_safeguard_sample(motion_dataset)
    motion, m_length, object_type, _parents, _graph_dist, _joint_relations, tpos_first_frame, offsets, *_ = motion_dataset.augment(data)

    assert cond["mirror_disabled_joint_indices"], f"selected sample {sample_name} no longer exercises mirror safeguards"
    assert motion.ndim == 3, f"expected mirrored motion to keep frame axis, got {motion.shape}"
    assert tpos_first_frame.ndim == 2, f"expected mirrored t-pose to remain (J, C), got {tpos_first_frame.shape}"
    assert m_length == motion.shape[0], f"expected mirrored motion length to match frame axis, got {m_length} vs {motion.shape[0]}"
    assert tpos_first_frame.shape == np.asarray(cond["mean"]).shape, (
        f"expected mirrored t-pose shape {np.asarray(cond['mean']).shape}, got {tpos_first_frame.shape}"
    )
    assert np.asarray(offsets).shape == np.asarray(cond["offsets"]).shape, (
        f"expected mirrored offsets shape {np.asarray(cond['offsets']).shape}, got {np.asarray(offsets).shape}"
    )


def main() -> None:
    test_loop_padding_updates_effective_length()
    print("loop padding regression: ok")

    test_loop_padding_random_offset_wraps_without_truncation()
    print("loop random offset regression: ok")

    test_explicit_window_start_respects_requested_crop()
    print("explicit crop regression: ok")

    test_prepare_sample_aug_info_reports_actual_loop_fill()
    print("loop aug-info regression: ok")

    test_mirror_augmentation_runs_before_normalization()
    print("mirror normalization regression: ok")

    test_mirror_safeguards_handle_single_frame_tpose()
    print("mirror safeguard tpose regression: ok")

    print("all regression checks passed")


if __name__ == "__main__":
    main()