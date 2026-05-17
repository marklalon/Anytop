"""Regression checks for loop padding and mirror augmentation.

Usage:
    d:/AI/pcvg-skeleton-animation/.venv/Scripts/python.exe tests/test_dataset_loop_and_mirror_regression.py

This script covers two previously broken behaviors:
1. Loop padding must update effective length to max_motion_length.
2. Mirror augmentation must run in raw feature space before normalization.
"""

from __future__ import annotations

import glob
import os
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_loaders.truebones.data.dataset as dataset_module
from data_loaders.tensors import truebones_batch_collate
from data_loaders.truebones.data.dataset import Truebones
from data_loaders.truebones.truebones_utils.get_opt import get_opt
from data_loaders.truebones.truebones_utils.motion_process import infer_translation_root_index_from_features


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
MIRROR_MOTION = _find_motion("Jaguar_Run_*.npy")
MIRROR_SUBSET = "quadropeds_clean"
MIRROR_SAFEGUARD_SUBSET = "all"
NUM_FRAMES = 60
_ENRICHED_MOTION_METADATA_LOOKUP = None


def assert_close(name: str, actual: np.ndarray, expected: np.ndarray, atol: float = 1e-6) -> None:
    max_diff = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    assert np.allclose(actual, expected, atol=atol), f"{name} mismatch: max_diff={max_diff}"


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
    dataset = _build_truebones(
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
    raw_norm = np.nan_to_num((raw - cond["norm_mean"][None, :]) / cond["norm_std_safe"][None, :]).astype(np.float32, copy=False)
    raw_len = raw_norm.shape[0]
    assert raw_len < NUM_FRAMES, "loop regression sample no longer needs padding"

    expected = np.concatenate(
        [raw_norm, np.tile(raw_norm, ((NUM_FRAMES - raw_len) // raw_len + 1, 1, 1))[: NUM_FRAMES - raw_len]],
        axis=0,
    )
    assert_close("loop-filled motion", motion, expected)


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
    raw_norm = np.nan_to_num((raw - cond["norm_mean"][None, :]) / cond["norm_std_safe"][None, :]).astype(np.float32, copy=False)
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
    raw_norm = np.nan_to_num((raw - cond["norm_mean"][None, :]) / cond["norm_std_safe"][None, :]).astype(np.float32, copy=False)
    expected = raw_norm[window_start:window_start + NUM_FRAMES]

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
    dataset = _build_truebones(
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
    manual = np.nan_to_num((manual - cond["norm_mean"][None, :]) / cond["norm_std_safe"][None, :]).astype(np.float32, copy=False)

    # augment() now returns the raw (un-normalized) mirrored t-pose; norm_mean is
    # itself the t-pose anchor, so the conditioning t-pose is fed to the model raw.
    manual_tpose = np.asarray(cond["tpos_first_frame"], dtype=np.float32)[perm].copy()
    manual_tpose[:, [0, 4, 5, 6, 9]] *= -1

    manual_offsets = np.asarray(cond["offsets"], dtype=np.float32)[perm].copy()
    manual_offsets[:, 0] *= -1

    assert m_length == raw.shape[0], f"mirror augmentation changed sequence length unexpectedly: {m_length}"
    assert_close("mirrored normalized motion", motion, manual)
    assert_close("mirrored normalized tpose", tpos_first_frame, manual_tpose)
    assert_close("mirrored offsets", offsets, manual_offsets)


def test_mirror_augmentation_passes_motion_translation_root_index(monkeypatch) -> None:
    dataset = _build_truebones(
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
    data["motion_metadata"] = dict(data.get("motion_metadata") or {})
    data["motion_metadata"]["translation_root_index"] = 0

    seen = []

    def fake_mirror(features, object_cond, *, translation_root_index=None, motion_metadata=None, anim_pos_threshold=0.01):
        seen.append((translation_root_index, dict(motion_metadata or {})))
        return np.asarray(features).copy(), np.asarray(object_cond["offsets"], dtype=np.float32).copy()

    monkeypatch.setattr(dataset_module, "mirror_features_with_safeguards", fake_mirror)

    motion_dataset.augment(data)

    assert seen, "mirror augmentation never called mirror_features_with_safeguards"
    assert [entry[0] for entry in seen] == [None, None]
    assert all(entry[1].get("translation_root_index") == 0 for entry in seen)


def test_batch_collate_preserves_translation_root_index() -> None:
    dataset = _build_truebones(
        split="train",
        temporal_window=31,
        num_frames=NUM_FRAMES,
        balanced=False,
        objects_subset=MIRROR_SUBSET,
        motion_cache_size=2,
    )

    motion_dataset = dataset.motion_dataset
    motion_dataset.data_dict[MIRROR_MOTION]["motion_metadata"] = dict(
        motion_dataset.data_dict[MIRROR_MOTION].get("motion_metadata") or {}
    )
    motion_dataset.data_dict[MIRROR_MOTION]["motion_metadata"]["translation_root_index"] = 0

    sample = motion_dataset.prepare_sample_by_name(MIRROR_MOTION, target_num_frames=NUM_FRAMES)
    _motion, cond = truebones_batch_collate([sample])

    assert int(cond["y"]["translation_root_index"][0]) == 0


def test_mirror_safeguards_handle_single_frame_tpose() -> None:
    dataset = _build_truebones(
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
    assert tpos_first_frame.shape == np.asarray(cond["norm_mean"]).shape, (
        f"expected mirrored t-pose shape {np.asarray(cond['norm_mean']).shape}, got {tpos_first_frame.shape}"
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