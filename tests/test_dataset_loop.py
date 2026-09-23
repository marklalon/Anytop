"""Regression checks for loop padding.

Usage:
    d:/AI/pcvg-skeleton-animation/.venv/Scripts/python.exe tests/test_dataset_loop_regression.py

This script verifies loop padding behavior.
"""

from __future__ import annotations

import glob
import os
import random
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_loaders.truebones.data.dataset as dataset_module
from data_loaders.tensors import truebones_batch_collate
from data_loaders.truebones.data.dataset import (
    Truebones,
    _circular_roll_motion,
    _drop_loop_closing_frame,
    resample_motion_features,
    time_scale_motion_features,
    _tile_loop_motion,
)
from data_loaders.truebones.truebones_utils.get_opt import get_opt
from data_loaders.truebones.truebones_utils.motion_labels import loop_is_phase_free
from data_loaders.truebones.truebones_utils.param_utils import MAX_FIT_SPEEDUP, MAX_SOURCE_FRAMES_MULT
from data_loaders.truebones.truebones_utils.motion_process import infer_translation_root_index_from_features
from data_loaders.truebones.truebones_utils.canonical_features import (
    canonical_to_physical_hml,
    physical_hml_to_canonical,
)
from data_loaders.truebones.truebones_utils.cond_schema import load_cond
from data_loaders.truebones.truebones_utils.physics_joint_annotation import (
    JOINT_NAME_EMBEDDING_SCHEMA_VERSION,
    JOINT_NAME_EMBEDDING_SLIM,
)
from data_loaders.truebones.truebones_utils.dataset_sources import resolve_species_key


def _find_motion(pattern: str) -> str:
    """Find a motion file by glob pattern and return its composite clip id.

    ``data_dict`` / ``name_list`` are keyed ``"<namespace>/<file>.npy"`` so that
    two datasets holding the same filename stay distinct clips.
    """
    source = get_opt(None).sources[0]
    files = sorted(glob.glob(os.path.join(source.motion_dir, pattern)))
    assert files, f"No files matching '{pattern}' in {source.motion_dir}"
    return f"{source.namespace}/{os.path.basename(files[0])}"


LOOP_MOTION = _find_motion("Ostrich_Run.npy")
LOOP_SUBSET = "biped"
# A PHASE-FREE loop authored WITH its closing key: its last frame repeats frame
# 0, so the loader drops it and the cycle it augments is one frame shorter.
# Ostrich_Run above ends 0.6 of a step short of frame 0 and keeps all its
# frames.  It must stay phase-free (loop_is_phase_free) -- the test tiles it,
# and a phase-anchored loop is never tiled.
#
# The two halves of that premise come from different places, which is why this
# fixture drifts.  The repeated frame is in the motion, but the wrap terminal
# velocity row is written by preprocessing off the sidecar's ``is_loop``
# annotation (features.py: the annotation is the caller's verdict, the detector
# only proposes).  A clip re-annotated as a non-loop is re-stored with a
# repeat-the-last-step row instead, and ``_drop_loop_closing_frame`` then
# correctly refuses it -- so the assertion below is also the tripwire for that
# drift.  Re-point the fixture; do not relax the drop.
CLOSING_KEY_LOOP_MOTION = _find_motion("Deer_WalkForward.npy")
CLOSING_KEY_LOOP_SUBSET = "quadruped"
# A PHASE-ANCHORED loop (an attack: closed, but frame 0 is the ready pose).
ANCHORED_LOOP_MOTION = _find_motion("Spider_Attack2.npy")
ANCHORED_LOOP_SUBSET = "multiped"
NUM_FRAMES = 60
# The n*MAX_SOURCE_FRAMES_MULT source-frame budget the dataset crops over-long
# clips to (see _prepare_sample); over-long clips resample down at exactly
# resample_speed MAX_SOURCE_FRAMES_MULT.
BUDGET_FRAMES = NUM_FRAMES * MAX_SOURCE_FRAMES_MULT
_ENRICHED_MOTION_METADATA_LOOKUP = None


def assert_close(name: str, actual: np.ndarray, expected: np.ndarray, atol: float = 1e-6) -> None:
    max_diff = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    assert np.allclose(actual, expected, atol=atol), f"{name} mismatch: max_diff={max_diff}"


def _normalize_motion(raw: np.ndarray, cond: dict[str, np.ndarray]) -> np.ndarray:
    return np.nan_to_num(physical_hml_to_canonical(raw, cond)).astype(np.float32, copy=False)


def _expected_resampled_velocity(raw: np.ndarray, target_frames: int, *, periodic: bool = False) -> np.ndarray:
    """Reference velocity rows, via ``np.interp`` on the integrated path.

    The path has a node at every frame boundary, ``L + 1`` of them (the last
    one through the terminal row). An open clip samples it end to end at step
    ``(L-1)/(T-1)`` and repeats its last row; a periodic clip samples ``T + 1`` nodes
    at step ``L/T``, so its terminal row is the wrap delta.
    """
    source_frames, joints = int(raw.shape[0]), int(raw.shape[1])
    path = np.zeros((source_frames + 1, joints, 3), dtype=np.float64)
    path[1:] = np.cumsum(raw[:, :, 9:12].astype(np.float64), axis=0)
    nodes = np.arange(source_frames + 1, dtype=np.float64)

    def path_at(times):
        return np.stack(
            [np.stack([np.interp(times, nodes, path[:, j, c]) for c in range(3)], axis=-1) for j in range(joints)],
            axis=1,
        )

    if periodic:
        step_scale = source_frames / target_frames
        sampled = path_at(np.arange(target_frames + 1) * step_scale)
        return ((sampled[1:] - sampled[:-1]) / step_scale).astype(np.float32)

    step_scale = (source_frames - 1) / (target_frames - 1)
    sampled = path_at(np.linspace(0.0, source_frames - 1, target_frames))
    expected_velocity = np.zeros((target_frames, joints, 3), dtype=np.float32)
    expected_velocity[:-1] = (sampled[1:] - sampled[:-1]) / step_scale
    expected_velocity[-1] = expected_velocity[-2]
    return expected_velocity


def _resample_raw_then_normalize(
    raw: np.ndarray,
    cond: dict[str, np.ndarray],
    target_frames: int,
    *,
    periodic: bool = False,
) -> np.ndarray:
    resampled = resample_motion_features(raw, target_frames, periodic=periodic)
    return _normalize_motion(resampled, cond)


def _get_enriched_motion_metadata_lookup() -> dict[str, dict[str, object]]:
    global _ENRICHED_MOTION_METADATA_LOOKUP
    if _ENRICHED_MOTION_METADATA_LOOKUP is not None:
        return {name: dict(metadata) for name, metadata in _ENRICHED_MOTION_METADATA_LOOKUP.items()}

    opt = get_opt(None)
    source = opt.sources[0]
    data_root, motion_dir = source.root, source.motion_dir

    cond_dict = load_cond(opt.cond_file)
    motion_metadata_lookup = dataset_module.load_motion_metadata(data_root)
    enriched_lookup = {name: dict(metadata) for name, metadata in motion_metadata_lookup.items()}
    for motion_name, motion_metadata in enriched_lookup.items():
        if 'translation_root_index' in motion_metadata:
            continue
        # motion_metadata carries the bare species name; cond is canonically keyed.
        object_key = resolve_species_key(cond_dict, motion_metadata['object_type'])
        motion = np.load(os.path.join(motion_dir, motion_name)).astype(np.float32, copy=False)
        motion_metadata['translation_root_index'] = infer_translation_root_index_from_features(
            motion,
            cond_dict[object_key]['parents'],
            cond_dict[object_key]['offsets'],
        )

    _ENRICHED_MOTION_METADATA_LOOKUP = enriched_lookup
    return {name: dict(metadata) for name, metadata in enriched_lookup.items()}


def _load_cond_stamped_with_the_current_schema(*args, **kwargs):
    """``load_cond`` with the joint-name embedding schema restamped to current.

    The checked-in dataset is encoded under whatever schema it was last
    preprocessed with, and ``ensure_joint_name_embeddings`` refuses an older one
    outright -- correctly, since those vectors mean something else. These are
    loop-padding tests, though: they must exercise the temporal path, not the
    embedding contract (tests/test_joint_struct_features.py covers that), so they
    accept the cond that is on disk.
    """
    cond_dict = load_cond(*args, **kwargs)
    for entry in cond_dict.values():
        meta = dict(entry.get('joints_names_embs_meta') or {})
        meta['schema_version'] = JOINT_NAME_EMBEDDING_SCHEMA_VERSION
        meta['slim'] = JOINT_NAME_EMBEDDING_SLIM
        entry['joints_names_embs_meta'] = meta
    return cond_dict


def _build_truebones(**kwargs) -> Truebones:
    enriched_lookup = _get_enriched_motion_metadata_lookup()
    with patch.object(dataset_module, 'load_motion_metadata', return_value=enriched_lookup),             patch.object(dataset_module, 'load_cond', _load_cond_stamped_with_the_current_schema):
        return Truebones(**kwargs)


def _synthetic_cycle(period: int, joints: int = 3, closing_key: bool = False) -> np.ndarray:
    """A (T, J, 12) HML clip whose pose runs round one sine cycle over
    ``period`` frames, with velocity channels that are the true frame deltas
    and a wrap-delta terminal row -- exactly what preprocessing stores for a
    loop.  ``closing_key`` appends frame 0 again as the last frame."""
    frame_count = period + int(closing_key)
    phase = 2.0 * np.pi * (np.arange(frame_count) % period) / period
    clip = np.zeros((frame_count, joints, 12), dtype=np.float32)
    for joint in range(joints):
        clip[:, joint, 0] = 0.30 * np.sin(phase + joint)
        clip[:, joint, 1] = 1.0 + 0.10 * np.cos(phase + joint)
        clip[:, joint, 3] = np.cos(phase - joint)
        clip[:, joint, 4] = np.sin(phase - joint)
        clip[:, joint, 8] = 1.0
    positions = clip[:, :, 0:3]
    clip[:-1, :, 9:12] = positions[1:] - positions[:-1]
    clip[-1, :, 9:12] = positions[0] - positions[-1]
    return clip


def test_drop_loop_closing_frame_drops_only_a_repeated_last_frame() -> None:
    clean = _synthetic_cycle(24)
    assert _drop_loop_closing_frame(clean) is clean, "a loop without a closing key must pass through untouched"

    with_key = _synthetic_cycle(24, closing_key=True)
    assert_close("fixture closing key repeats frame 0", with_key[-1, :, :9], with_key[0, :, :9])
    kept = _drop_loop_closing_frame(with_key)
    assert kept.shape[0] == 24
    assert_close("the clean period is what remains", kept, clean)
    # No channel needs rewriting: the surviving last frame's velocity is its
    # delta to the dropped frame, i.e. the wrap delta to frame 0.
    assert_close("wrap velocity after the drop", kept[-1, :, 9:12], kept[0, :, 0:3] - kept[-1, :, 0:3])

    # Export rounding on a repeated key (well inside the ratio) is a repeat.
    noisy = with_key.copy()
    noisy[-1, :, :9] += 0.005 * float(np.median(np.linalg.norm(np.diff(with_key[:, :, :3], axis=0), axis=-1).max(axis=1)))
    assert _drop_loop_closing_frame(noisy).shape[0] == 24

    # A last frame that is real motion -- half a step short of frame 0 -- stays.
    short = _synthetic_cycle(24)
    short[-1, :, :9] = 0.5 * (short[-2, :, :9] + short[0, :, :9])
    assert _drop_loop_closing_frame(short) is short

    # An eased-out ending: the last two frames creep up on frame 0 in steps of
    # 0.5% of the way, so the last frame is within the ratio of the MEDIAN step
    # of frame 0 (the global test passes) yet a full LOCAL step from it.  That
    # is motion, not a repeated key, and it stays.
    eased = _synthetic_cycle(24)
    eased[-2, :, :9] = eased[-3, :, :9] + 0.990 * (eased[0, :, :9] - eased[-3, :, :9])
    eased[-1, :, :9] = eased[-3, :, :9] + 0.995 * (eased[0, :, :9] - eased[-3, :, :9])
    eased[:-1, :, 9:12] = eased[1:, :, 0:3] - eased[:-1, :, 0:3]
    eased[-1, :, 9:12] = eased[0, :, 0:3] - eased[-1, :, 0:3]
    pose_steps = np.linalg.norm(np.diff(eased[:, :, :9], axis=0), axis=-1).max(axis=1)
    wrap_gap = float(np.linalg.norm(eased[-1, :, :9] - eased[0, :, :9], axis=-1).max())
    assert wrap_gap <= 0.02 * float(np.median(pose_steps)), "fixture must pass the global test"
    assert wrap_gap > 0.5 * float(pose_steps[-1]), "fixture must be one local step short"
    assert _drop_loop_closing_frame(eased) is eased

    # The pose may repeat while the root travelled: the wrap velocity row
    # (world coordinates) says so, and the frame is NOT a duplicate.
    transported = with_key.copy()
    transported[-1, :, 9:12] = np.array([0.0, 0.0, -2.0], dtype=np.float32)
    assert _drop_loop_closing_frame(transported) is transported

    # A motionless clip is a held pose; its length is the duration.
    held = np.repeat(with_key[:1], 10, axis=0).copy()
    held[:, :, 9:12] = 0.0
    assert _drop_loop_closing_frame(held) is held
    # And a two-frame clip has nothing to give.
    assert _drop_loop_closing_frame(with_key[:2]).shape[0] == 2


def test_loop_with_closing_key_is_augmented_as_its_clean_period() -> None:
    dataset = _build_truebones(
        split="train",
        num_frames=NUM_FRAMES,
        objects_subset=CLOSING_KEY_LOOP_SUBSET,
        motion_cache_size=2,
    )
    motion_dataset = dataset.motion_dataset
    data = motion_dataset.data_dict[CLOSING_KEY_LOOP_MOTION]
    cond = motion_dataset.cond_dict[data["object_type"]]
    raw = np.load(data["motion_path"]).astype(np.float32, copy=False)
    period = _drop_loop_closing_frame(raw)
    assert period.shape[0] == raw.shape[0] - 1, (
        f"fixture clip {CLOSING_KEY_LOOP_MOTION} no longer ships a closing key: "
        "its is_loop annotation or its motion changed -- re-point the fixture "
        "to another annotated loop that still repeats frame 0"
    )

    # Single cycle, phase 0: the clean period is what gets resampled into the
    # window, and resample_speed_cond counts the frames the model actually sees.
    with patch.object(motion_dataset, '_sample_loop_tile_count', return_value=1):
        sample = motion_dataset._prepare_sample(
            CLOSING_KEY_LOOP_MOTION, data, target_num_frames=NUM_FRAMES, loop_offset=0, return_aug_info=True,
        )
    motion, m_length, *_rest, motion_metadata, _name, _joint_mask_dict, aug_info = sample
    expected = _resample_raw_then_normalize(period, cond, NUM_FRAMES, periodic=True)
    assert m_length == NUM_FRAMES
    assert np.isclose(float(aug_info["resample_speed_cond"]), float(period.shape[0]) / float(NUM_FRAMES))
    assert_close("closing-key loop, single cycle", motion, expected, atol=3e-5)
    # The seam is one ordinary frame step, not the ~zero wrap the stored clip carries.
    physical = canonical_to_physical_hml(motion, cond)
    seam = float(np.linalg.norm(physical[-1, :, 9:12], axis=-1).max())
    typical = float(np.median(np.linalg.norm(physical[:-1, :, 9:12], axis=-1).max(axis=1)))
    assert seam > 0.3 * typical, f"wrap velocity {seam} still reads as a stall against a typical step {typical}"

    # Rolled and tiled: the roll runs over the period and the tiles are copies
    # of it, so no seam repeats a frame.
    offset = 7
    with patch.object(motion_dataset, '_sample_loop_tile_count', return_value=2):
        sample = motion_dataset._prepare_sample(
            CLOSING_KEY_LOOP_MOTION, data, target_num_frames=NUM_FRAMES, loop_offset=offset, return_aug_info=True,
        )
    motion, _m_length, *_rest, _motion_metadata, _name, _joint_mask_dict, aug_info = sample
    tiled = _tile_loop_motion(_circular_roll_motion(period, offset), 2)
    assert np.isclose(float(aug_info["resample_speed_cond"]), float(2 * period.shape[0]) / float(NUM_FRAMES))
    assert_close("closing-key loop, rolled and tiled", motion, _resample_raw_then_normalize(tiled, cond, NUM_FRAMES, periodic=True), atol=3e-5)

    # Idempotent: the period itself has no closing key to give.
    assert _drop_loop_closing_frame(period) is period


def test_loop_is_phase_free_needs_every_head_phase_free() -> None:
    assert loop_is_phase_free("walk, forward")
    assert loop_is_phase_free("idle, sleep")
    assert loop_is_phase_free("fly, hover")
    assert not loop_is_phase_free("attack, spit")
    assert not loop_is_phase_free("roar")
    # One anchored head anchors the clip, in either position.
    assert not loop_is_phase_free("idle, rear")
    assert not loop_is_phase_free("attack, hover")
    assert not loop_is_phase_free("")
    assert not loop_is_phase_free(None)


@pytest.mark.parametrize("loop_cond_prob", [1.0, 0.0])
def test_phase_anchored_loop_is_never_rolled_or_tiled(loop_cond_prob) -> None:
    dataset = _build_truebones(
        split="train",
        num_frames=NUM_FRAMES,
        objects_subset=ANCHORED_LOOP_SUBSET,
        motion_cache_size=2,
        loop_cond_prob=loop_cond_prob,
    )
    motion_dataset = dataset.motion_dataset
    data = motion_dataset.data_dict[ANCHORED_LOOP_MOTION]
    assert data["motion_metadata"]["is_loop"] is True and not loop_is_phase_free(
        data["motion_metadata"]["action_label"]
    ), f"fixture clip {ANCHORED_LOOP_MOTION} is no longer a phase-anchored loop -- re-point it"
    cond = motion_dataset.cond_dict[data["object_type"]]
    period = _drop_loop_closing_frame(np.load(data["motion_path"]).astype(np.float32, copy=False))

    with patch.object(motion_dataset, '_sample_loop_offset', side_effect=AssertionError("rolled")), \
            patch.object(motion_dataset, '_sample_loop_tile_count', side_effect=AssertionError("tiled")):
        sample = motion_dataset._prepare_sample(
            ANCHORED_LOOP_MOTION, data, target_num_frames=NUM_FRAMES, return_aug_info=True,
        )
    motion, _m_length, *_rest, motion_metadata, _name, _joint_mask_dict, aug_info = sample

    told_loop = loop_cond_prob == 1.0
    assert aug_info["loop_phase_offset"] == 0
    assert aug_info["loop_tile_count"] == 1
    assert aug_info["loop_phase_free"] is False
    assert motion_metadata["is_loop"] is told_loop
    assert np.isclose(float(aug_info["resample_speed_cond"]), float(period.shape[0]) / float(NUM_FRAMES))
    # Frame 0 stays the ready pose: the window is the unrolled single event,
    # periodic when the model is told it is a loop, an ordinary one-shot otherwise.
    expected = _resample_raw_then_normalize(period, cond, NUM_FRAMES, periodic=told_loop)
    assert_close("phase-anchored loop window", motion, expected, atol=3e-5)


def test_speed_resample_preserves_velocity() -> None:
    source = np.zeros((4, 2, 12), dtype=np.float32)
    source[:, :, 0] = np.array([0.0, 1.0, 3.0, 6.0], dtype=np.float32)[:, None]
    source[:, :, 1] = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float32)[:, None]
    source[:, :, 2] = np.array([0.0, -1.0, -1.5, -2.0], dtype=np.float32)[:, None]
    source[:, :, 9] = np.array([0.0, 2.0, 4.0, 8.0], dtype=np.float32)[:, None]
    source[:, :, 10] = 3.0
    source[:, :, 11] = 0.0

    resampled = resample_motion_features(source, 7)

    expected_vel = _expected_resampled_velocity(source, 7)

    assert_close("resampled velocity", resampled[:, :, 9:12], expected_vel)
    assert_close("zero velocity channel", resampled[:, :, 11], np.zeros_like(resampled[:, :, 11]))


def test_loop_speed_resample_rebuilds_terminal_velocity_from_wrap_delta() -> None:
    source = np.zeros((4, 1, 12), dtype=np.float32)
    source[:, 0, 0:3] = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [4.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    source[:, 0, 9:12] = np.array(
        [[0.5, 0.0, 0.0], [0.5, 0.5, 0.0], [1.0, 0.5, 0.5], [-2.0, -1.0, -0.5]],
        dtype=np.float32,
    )

    resampled = resample_motion_features(source, 6, periodic=True)
    expected_vel = _expected_resampled_velocity(source, 6, periodic=True)

    assert_close("loop visible velocity", resampled[:-1, :, 9:12], expected_vel[:-1])
    assert_close("loop terminal velocity", resampled[-1, :, 9:12], expected_vel[-1])


def test_loop_resample_keeps_the_wrap_step_an_ordinary_step() -> None:
    # The closure ratio of docs/conditional_modulation_upgrade.md §2: the seam
    # step |x[0] - x[-1]| against the steps beside it. Resampled end to end, a
    # cycle of L frames keeps a wrap step of exactly one source frame against
    # (L-1)/(T-1) inside the window, so its seam is uneven unless L == T; that
    # unevenness is what the loop phase table and the loss were fitted to.
    period = _synthetic_cycle(45)

    def seam_ratio(window):
        pose = window[:, :, :9].astype(np.float64)
        step = lambda a, b: float(np.linalg.norm(pose[a] - pose[b]))
        return step(0, -1) / (0.5 * (step(-1, -2) + step(1, 0)))

    for target in (30, 60, 97):
        periodic = resample_motion_features(period, target, periodic=True)
        assert abs(seam_ratio(periodic) - 1.0) < 0.05, (target, seam_ratio(periodic))
        open_window = resample_motion_features(period, target)
        assert abs(seam_ratio(open_window) - 1.0) > 0.25, (target, seam_ratio(open_window))

        # Every row, the terminal one included, is the step to the next frame
        # of the clip at step L/T -- the identity loop_wrap_loss's terminal
        # term and the velocity loss read with that step scale.
        step_scale = 45.0 / float(target)
        following = np.roll(periodic[:, :, 0:3], -1, axis=0)
        assert_close(
            f"periodic velocity rows target={target}",
            periodic[:, :, 9:12] * np.float32(step_scale),
            following - periodic[:, :, 0:3],
            atol=1e-5,
        )


def test_loop_padding_updates_effective_length() -> None:
    dataset = _build_truebones(
        split="train",
        num_frames=NUM_FRAMES,
        objects_subset=LOOP_SUBSET,
        motion_cache_size=2,
    )

    motion_dataset = dataset.motion_dataset
    with patch.object(motion_dataset, '_sample_loop_tile_count', return_value=1):
        sample = motion_dataset.prepare_sample_by_name(LOOP_MOTION, target_num_frames=NUM_FRAMES, loop_offset=0)
    motion, m_length = sample[0], sample[1]
    motion_metadata, name = sample[10], sample[11]

    assert name == LOOP_MOTION, f"unexpected sample: {name}"
    assert bool(motion_metadata.get("is_loop", False)), "loop regression sample is no longer marked loop"
    assert motion.shape[0] == NUM_FRAMES, f"expected padded motion to have {NUM_FRAMES} frames"
    assert m_length == NUM_FRAMES, f"effective length should track loop-filled frames, got {m_length}"

    data = motion_dataset.data_dict[LOOP_MOTION]
    cond = motion_dataset.cond_dict[data["object_type"]]
    raw = np.load(data["motion_path"]).astype(np.float32, copy=False)
    raw_len = raw.shape[0]
    assert raw_len < NUM_FRAMES, "loop regression sample no longer needs padding"
    expected = _resample_raw_then_normalize(raw, cond, NUM_FRAMES, periodic=True)

    assert "loop_phase_length" not in motion_metadata
    assert "loop_full_cycle" not in motion_metadata
    assert np.isclose(float(motion_metadata["resample_speed_cond"]), float(raw_len) / float(NUM_FRAMES))
    assert_close("loop-filled motion", motion, expected, atol=3e-5)


def test_loop_padding_can_tile_multiple_cycles_before_resample() -> None:
    dataset = _build_truebones(
        split="train",
        num_frames=NUM_FRAMES,
        objects_subset=LOOP_SUBSET,
        motion_cache_size=2,
    )

    motion_dataset = dataset.motion_dataset
    data = motion_dataset.data_dict[LOOP_MOTION]
    cond = motion_dataset.cond_dict[data["object_type"]]
    raw = np.load(data["motion_path"]).astype(np.float32, copy=False)

    with patch.object(motion_dataset, '_sample_loop_tile_count', return_value=2):
        sample = motion_dataset.prepare_sample_by_name(LOOP_MOTION, target_num_frames=NUM_FRAMES, loop_offset=0)
    motion, m_length = sample[0], sample[1]
    motion_metadata, name = sample[10], sample[11]

    expected = _resample_raw_then_normalize(_tile_loop_motion(raw, 2), cond, NUM_FRAMES, periodic=True)

    assert name == LOOP_MOTION, f"unexpected sample: {name}"
    assert motion.shape[0] == NUM_FRAMES
    assert m_length == NUM_FRAMES
    assert np.isclose(float(motion_metadata["resample_speed_cond"]), float(raw.shape[0] * 2) / float(NUM_FRAMES))
    # The tile count is diagnostics only: the model is not told how many
    # cycles the window holds.
    assert motion_metadata["loop_tile_count"] == 2
    assert "loop_phase_length" not in motion_metadata
    assert_close("loop-filled tiled motion", motion, expected, atol=3e-5)


def test_loop_padding_random_offset_wraps_without_truncation() -> None:
    dataset = _build_truebones(
        split="train",
        num_frames=NUM_FRAMES,
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
    motion, m_length = sample[0], sample[1]
    rolled_raw = _circular_roll_motion(raw, offset)
    expected_raw = resample_motion_features(rolled_raw, NUM_FRAMES, periodic=True)
    expected = _normalize_motion(expected_raw, cond)
    raw_motion = canonical_to_physical_hml(motion, cond)
    assert motion.shape[0] == NUM_FRAMES, f"expected random-offset loop fill to keep {NUM_FRAMES} frames"
    assert m_length == NUM_FRAMES, f"effective length should remain {NUM_FRAMES}, got {m_length}"
    assert_close("loop-filled motion with wraparound offset", motion, expected, atol=3e-5)
    # Pure circular roll preserves the velocity ring; the terminal velocity
    # (channels 9-12, i.e. the wrap-around delta) must equal the expected
    # terminal velocity after resample — it is NOT forced to zero/copy.
    assert_close("rolled loop terminal velocity", raw_motion[-1, :, 9:12], expected_raw[-1, :, 9:12], atol=3e-5)


def test_long_motion_crops_fixed_length_random_window() -> None:
    dataset = _build_truebones(
        split="train",
        num_frames=NUM_FRAMES,
        objects_subset=LOOP_SUBSET,
        motion_cache_size=0,
    )

    motion_dataset = dataset.motion_dataset

    # Build a NON-loop clip longer than the n*MAX_SOURCE_FRAMES_MULT budget:
    # loop motions get circular-roll + tile augmentation before the crop, so
    # their window would not match a direct raw crop.
    source_data = motion_dataset.data_dict[LOOP_MOTION]
    source_raw = np.load(source_data["motion_path"]).astype(np.float32, copy=False)
    long_len = BUDGET_FRAMES + 37
    repeat_count = (long_len + source_raw.shape[0] - 1) // source_raw.shape[0]
    long_raw = np.tile(source_raw, (repeat_count, 1, 1))[:long_len]

    with tempfile.TemporaryDirectory() as tmp_dir:
        motion_path = os.path.join(tmp_dir, "long_non_loop.npy")
        np.save(motion_path, long_raw.astype(np.float32, copy=False))

        long_data = dict(source_data)
        long_data["motion_path"] = motion_path
        long_data["length"] = long_raw.shape[0]
        long_data["motion_metadata"] = dict(source_data["motion_metadata"])
        long_data["motion_metadata"]["is_loop"] = False

        # A non-loop clip draws exactly one randint: the crop window start.
        window_start = 13
        with patch.object(dataset_module.random, 'randint', return_value=window_start):
            sample = motion_dataset._prepare_sample(
                "synthetic_long_non_loop.npy",
                long_data,
                target_num_frames=NUM_FRAMES,
                return_aug_info=True,
            )
    motion, m_length, *_rest, _motion_metadata, _name, _joint_mask_dict, aug_info = sample

    cond = motion_dataset.cond_dict[long_data["object_type"]]
    # The crop length is always the full n*MAX_SOURCE_FRAMES_MULT budget; only
    # the start is random.
    expected = _resample_raw_then_normalize(
        long_raw[window_start:window_start + BUDGET_FRAMES], cond, NUM_FRAMES
    )

    assert m_length == NUM_FRAMES, f"cropped sample should have effective length {NUM_FRAMES}, got {m_length}"
    assert np.isclose(
        float(aug_info["resample_speed_cond"]), MAX_SOURCE_FRAMES_MULT
    ), f"expected resample_speed {MAX_SOURCE_FRAMES_MULT}, got {aug_info}"
    assert_close("fixed-length random crop window", motion, expected)


def test_prepare_sample_aug_info_reports_actual_loop_fill() -> None:
    dataset = _build_truebones(
        split="train",
        num_frames=NUM_FRAMES,
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
    assert np.isclose(float(aug_info["resample_speed_cond"]), float(motion_dataset.data_dict[LOOP_MOTION]["length"]) / float(NUM_FRAMES))


def test_loop_uncond_keeps_loop_augmentation_but_hands_over_an_open_window() -> None:
    dataset = _build_truebones(
        split="train",
        num_frames=NUM_FRAMES,
        objects_subset=LOOP_SUBSET,
        motion_cache_size=2,
        loop_cond_prob=0.0,
    )

    motion_dataset = dataset.motion_dataset

    with patch.object(motion_dataset, '_sample_loop_tile_count', return_value=1),             patch.object(dataset_module.random, 'randint', return_value=0):
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
    # The clip is still physically a loop (roll + tile applied above), but the
    # window resample runs in the open mode because the model is told it is
    # not one: absolute time table, open velocity step, no wrap losses.
    expected = _resample_raw_then_normalize(raw, cond, NUM_FRAMES, periodic=False)

    assert name == LOOP_MOTION, f"unexpected sample: {name}"
    assert motion.shape[0] == NUM_FRAMES
    assert m_length == NUM_FRAMES
    assert motion_metadata["is_loop"] is False
    assert motion_metadata["loop_data_aug_applied"] is True
    assert motion_metadata["loop_uncond"] is True
    assert aug_info["loop_applied"] is False
    assert aug_info["loop_uncond"] is True
    assert np.isclose(float(aug_info["resample_speed_cond"]), float(raw.shape[0]) / float(NUM_FRAMES))
    assert_close("loop uncond resample", motion, expected, atol=3e-5)


def test_loop_uncond_never_flips_an_explicit_loop_offset() -> None:
    dataset = _build_truebones(
        split="train",
        num_frames=NUM_FRAMES,
        objects_subset=LOOP_SUBSET,
        motion_cache_size=2,
        loop_cond_prob=0.0,
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
    motion_metadata, aug_info = sample[10], sample[-1]

    # The diagnostics path (prepare_sample_by_name) asks for one specific
    # phase of the loop; it gets the loop, whatever the training-time draw.
    assert motion_metadata["is_loop"] is True
    assert motion_metadata["loop_uncond"] is False
    assert aug_info["loop_applied"] is True
    assert aug_info["loop_uncond"] is False


def test_loop_cond_prob_out_of_range_is_refused() -> None:
    with pytest.raises(ValueError, match="loop_cond_prob"):
        _build_truebones(
            split="train",
            num_frames=NUM_FRAMES,
            objects_subset=LOOP_SUBSET,
            motion_cache_size=0,
            loop_cond_prob=1.5,
        )


def test_loop_conditioned_long_loop_downgrades_to_non_loop(tmp_path) -> None:
    dataset = _build_truebones(
        split="train",
        num_frames=NUM_FRAMES,
        objects_subset=LOOP_SUBSET,
        motion_cache_size=0,
    )

    motion_dataset = dataset.motion_dataset

    source_data = motion_dataset.data_dict[LOOP_MOTION]
    source_raw = np.load(source_data["motion_path"]).astype(np.float32, copy=False)
    repeat_count = (BUDGET_FRAMES + 8 + source_raw.shape[0] - 1) // source_raw.shape[0]
    long_raw = np.tile(source_raw, (repeat_count, 1, 1))[:BUDGET_FRAMES + 8]
    motion_path = tmp_path / "conditioned_long_loop.npy"
    np.save(motion_path, long_raw.astype(np.float32, copy=False))

    long_data = dict(source_data)
    long_data["motion_path"] = str(motion_path)
    long_data["length"] = long_raw.shape[0]
    long_data["motion_metadata"] = dict(source_data["motion_metadata"])
    long_data["motion_metadata"]["is_loop"] = True

    # Two randint draws in order: the loop phase offset, then the crop start.
    window_start = 5
    with patch.object(dataset_module.random, 'randint', side_effect=[NUM_FRAMES, window_start]):
        sample = motion_dataset._prepare_sample(
            "synthetic_conditioned_long_loop.npy",
            long_data,
            target_num_frames=NUM_FRAMES,
            return_aug_info=True,
        )
    motion, m_length, *_rest, motion_metadata, _name, _joint_mask_dict, aug_info = sample

    cond = motion_dataset.cond_dict[long_data["object_type"]]
    expected_augmented = _circular_roll_motion(long_raw, NUM_FRAMES)
    expected = _resample_raw_then_normalize(
        expected_augmented[window_start:window_start + BUDGET_FRAMES], cond, NUM_FRAMES
    )

    assert motion.shape[0] == NUM_FRAMES
    assert m_length == NUM_FRAMES
    assert motion_metadata["is_loop"] is False
    assert aug_info["loop_applied"] is False
    assert aug_info["loop_uncond"] is True
    assert np.isclose(float(aug_info["resample_speed_cond"]), MAX_SOURCE_FRAMES_MULT)
    assert_close("conditioned long loop downgraded crop", motion, expected)


def test_batch_collate_preserves_translation_root_index() -> None:
    dataset = _build_truebones(
        split="train",
        num_frames=NUM_FRAMES,
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
    assert "resample_speed_cond" in cond["y"]
    assert "loop_phase_offset" in cond["y"]
    assert "loop_tile_count" in cond["y"]
    assert "loop_data_aug_applied" in cond["y"]
    # Diagnostics only: present so training logs can report it, never read by
    # the model.
    assert cond["y"]["motion_speed_applied"].dtype == torch.float32
    assert float(cond["y"]["motion_speed_applied"][0]) == 1.0


# ── motion-speed augmentation ──────────────────────────────────────────────

def _synthetic_clip(num_frames: int, *, loop: bool) -> np.ndarray:
    """One joint following a smooth path; velocity channels are the exact
    per-frame world deltas (terminal = wrap delta for a loop)."""
    t = np.linspace(0.0, 2.0 * np.pi, num_frames, endpoint=not loop, dtype=np.float64)
    pos = np.stack([np.cos(t), 0.5 * np.sin(2.0 * t), t / (2.0 * np.pi)], axis=-1)
    clip = np.zeros((num_frames, 1, 12), dtype=np.float32)
    clip[:, 0, 0:3] = pos
    clip[:, 0, 3:9] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    clip[:-1, 0, 9:12] = (pos[1:] - pos[:-1])
    clip[-1, 0, 9:12] = (pos[0] - pos[-1]) if loop else clip[-2, 0, 9:12]
    return clip


def test_time_scale_is_identity_at_the_source_length() -> None:
    clip = _synthetic_clip(9, loop=False)
    scaled, speed = time_scale_motion_features(clip, 9)
    assert scaled is clip
    assert speed == 1.0


def test_time_scale_rescales_velocity_to_the_new_frame_step() -> None:
    # The scaled clip must be self-consistent as a source clip in its own
    # right: its velocity channel is the delta between ITS consecutive frames,
    # not the per-original-frame value resample_motion_features keeps.
    for loop in (False, True):
        clip = _synthetic_clip(13, loop=loop)
        for target in (9, 17):
            scaled, speed = time_scale_motion_features(clip, target, periodic=loop)
            assert scaled.shape[0] == target
            # A loop clip maps its 13 steps (the wrap included) onto the target's
            # `target`; an open clip its 12 onto `target - 1`.
            assert np.isclose(speed, 13.0 / float(target) if loop else 12.0 / float(target - 1))
            # resample_motion_features' velocity is (path delta) / step_scale,
            # so multiplying by step_scale hands back the plain path delta.
            expected = _expected_resampled_velocity(clip, target, periodic=loop) * np.float32(speed)
            assert_close(f"time-scaled velocity loop={loop} target={target}", scaled[:, :, 9:12], expected, atol=1e-5)
            # Positions / rotations are those of the plain resample.
            assert_close(
                f"time-scaled pose loop={loop} target={target}",
                scaled[:, :, :9],
                resample_motion_features(clip, target, periodic=loop)[:, :, :9],
            )
            if loop:
                # Closed cycle: velocities integrate back to the start.
                total = scaled[:, 0, 9:12].astype(np.float64).sum(axis=0)
                assert np.allclose(total, 0.0, atol=1e-4), total


def test_time_scale_rejects_degenerate_lengths() -> None:
    clip = _synthetic_clip(5, loop=False)
    for bad in (0, 1):
        try:
            time_scale_motion_features(clip, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"target {bad} should be rejected")


def test_motion_speed_aug_is_transparent_to_roll_tile_crop_and_resample() -> None:
    dataset = _build_truebones(
        split="train",
        num_frames=NUM_FRAMES,
        objects_subset=LOOP_SUBSET,
        motion_cache_size=2,
    )
    motion_dataset = dataset.motion_dataset
    data = motion_dataset.data_dict[LOOP_MOTION]
    cond = motion_dataset.cond_dict[data["object_type"]]
    raw = np.load(data["motion_path"]).astype(np.float32, copy=False)
    raw_len = int(raw.shape[0])
    scaled_len = int(round(raw_len / 1.15))
    assert scaled_len != raw_len

    def prepare(target_len):
        with patch.object(motion_dataset, '_sample_loop_tile_count', return_value=1),                 patch.object(motion_dataset, '_sample_motion_speed_target_length', return_value=target_len):
            sample = motion_dataset._prepare_sample(
                LOOP_MOTION, data, target_num_frames=NUM_FRAMES, loop_offset=0, return_aug_info=True,
            )
        return sample[0], sample[10], sample[-1]

    motion, motion_metadata, aug_info = prepare(scaled_len)

    # The time-scaled clip is just a shorter source clip: the window is the
    # ordinary loop path (terminal wrap kept) applied to it, and
    # resample_speed_cond reports ITS length -- the model is told nothing else.
    scaled_raw, speed = time_scale_motion_features(raw, scaled_len, periodic=True)
    expected = _resample_raw_then_normalize(scaled_raw, cond, NUM_FRAMES, periodic=True)
    assert_close("time-scaled loop window", motion, expected, atol=3e-5)
    assert np.isclose(float(aug_info["resample_speed_cond"]), float(scaled_len) / float(NUM_FRAMES))
    assert np.isclose(float(aug_info["motion_speed_applied"]), speed)
    assert np.isclose(float(motion_metadata["motion_speed_applied"]), speed)
    assert bool(motion_metadata["is_loop"]), "time-scaling must not downgrade a loop"
    assert aug_info["loop_applied"] is True
    assert aug_info["loop_tile_count"] == 1
    # The augmentation never becomes a model input.
    _motion_t, cond_batch = truebones_batch_collate([
        motion_dataset.prepare_sample_by_name(LOOP_MOTION, target_num_frames=NUM_FRAMES, loop_offset=0)
    ])
    assert "motion_speed_applied" in cond_batch["y"]
    assert not any(key.startswith("motion_speed") and key != "motion_speed_applied" for key in cond_batch["y"])

    # What the augmentation changes in the window: the pose/rotation CONTENT
    # is the same clip compressed into the same T frames (identical up to one
    # extra linear interpolation), the velocity channels scale with the speed
    # and resample_speed_cond shrinks by it. That is the whole mechanism --
    # it spreads the resample_speed a given content is seen at, it does not
    # invent new poses.
    base, _base_metadata, base_info = prepare(raw_len)
    assert np.isclose(float(base_info["motion_speed_applied"]), 1.0)
    pose_diff = np.abs(motion[..., :9] - base[..., :9]).mean()
    pose_step = np.abs(np.diff(base[..., :9], axis=0)).mean()
    assert pose_diff < 0.15 * pose_step, (pose_diff, pose_step)
    live = np.abs(base[..., 9:12]) > 1e-2
    vel_ratio = np.median(np.abs(motion[..., 9:12][live]) / np.abs(base[..., 9:12][live]))
    assert abs(vel_ratio - speed) < 0.05 * speed, (vel_ratio, speed)
    assert float(aug_info["resample_speed_cond"]) < float(base_info["resample_speed_cond"])


class _SpeedOpt:
    def __init__(self, ratio, prob=1.0, min_length=20):
        self.motion_speed_aug = ratio
        self.motion_speed_aug_prob = prob
        self.min_length = min_length


def _speed_sampler(ratio, prob=1.0, min_length=20):
    sampler = dataset_module.MotionDataset.__new__(dataset_module.MotionDataset)
    sampler.opt = _SpeedOpt(ratio, prob, min_length)
    sampler.min_length = min_length
    return sampler


def test_motion_speed_sampler_is_off_by_default_and_draws_nothing() -> None:
    sampler = _speed_sampler(1.0)
    # Off: no RNG consumed, so tests that count random draws stay valid.
    with patch.object(dataset_module.random, 'random', side_effect=AssertionError("must not draw")):
        assert sampler._sample_motion_speed_target_length(31, False, BUDGET_FRAMES) == 31
    sampler = _speed_sampler(1.2, prob=0.0)
    assert sampler._sample_motion_speed_target_length(31, False, BUDGET_FRAMES) == 31


def test_motion_speed_sampler_stays_inside_the_clip_boundaries() -> None:
    sampler = _speed_sampler(1.2)
    rng = random.Random(7)
    with patch.object(dataset_module.random, 'random', rng.random),             patch.object(dataset_module.random, 'uniform', rng.uniform):
        # Plain clip: log-uniform ratio in [1/1.2, 1.2] around 31 frames.
        draws = {sampler._sample_motion_speed_target_length(31, False, BUDGET_FRAMES) for _ in range(2000)}
        assert min(draws) == 26 and max(draws) == 37, sorted(draws)
        assert len(draws) >= 10, "the augmentation should spread the clip length, not pick a few values"
        # min_length clip: may only slow down.
        draws = {sampler._sample_motion_speed_target_length(20, False, BUDGET_FRAMES) for _ in range(2000)}
        assert min(draws) == 20 and max(draws) == 24, sorted(draws)
        # A loop that fits the source budget keeps fitting (never crop-downgraded).
        draws = {sampler._sample_motion_speed_target_length(115, True, BUDGET_FRAMES) for _ in range(2000)}
        assert max(draws) == BUDGET_FRAMES and min(draws) == 96, sorted(draws)
        # A loop already over the budget is not forced anywhere.
        draws = {sampler._sample_motion_speed_target_length(150, True, BUDGET_FRAMES) for _ in range(2000)}
        assert min(draws) == 125 and max(draws) == 180, sorted(draws)
        # A non-loop clip over the budget gets cropped either way: no ceiling.
        draws = {sampler._sample_motion_speed_target_length(115, False, BUDGET_FRAMES) for _ in range(2000)}
        assert max(draws) > BUDGET_FRAMES, sorted(draws)


def test_motion_speed_sampler_fits_phase_anchored_clips_into_the_budget() -> None:
    fit_limit = int(MAX_FIT_SPEEDUP * BUDGET_FRAMES)
    beyond = fit_limit + 20
    beyond_fitted = int(round(beyond / MAX_FIT_SPEEDUP))
    assert beyond_fitted > BUDGET_FRAMES
    # The floor is data preparation, not augmentation: it applies with the
    # augmentation off or not drawn, without consuming any randomness.
    for sampler in (_speed_sampler(1.0), _speed_sampler(1.2, prob=0.0)):
        with patch.object(dataset_module.random, 'uniform', side_effect=AssertionError("must not draw")):
            assert sampler._sample_motion_speed_target_length(
                BUDGET_FRAMES + 30, False, BUDGET_FRAMES, fit_budget=True) == BUDGET_FRAMES
            assert sampler._sample_motion_speed_target_length(
                fit_limit, True, BUDGET_FRAMES, fit_budget=True) == BUDGET_FRAMES
            # Past the fit limit: sped up by the cap, then left to the crop.
            assert sampler._sample_motion_speed_target_length(
                beyond, False, BUDGET_FRAMES, fit_budget=True) == beyond_fitted
            # A clip that already fits is untouched.
            assert sampler._sample_motion_speed_target_length(
                BUDGET_FRAMES - 5, False, BUDGET_FRAMES, fit_budget=True) == BUDGET_FRAMES - 5
    # Without the flag (a phase-free clip) nothing is forced.
    assert _speed_sampler(1.0)._sample_motion_speed_target_length(
        BUDGET_FRAMES + 30, False, BUDGET_FRAMES) == BUDGET_FRAMES + 30

    sampler = _speed_sampler(1.2)
    rng = random.Random(11)
    with patch.object(dataset_module.random, 'random', rng.random),             patch.object(dataset_module.random, 'uniform', rng.uniform):
        def draws(length, is_loop=False):
            return {
                sampler._sample_motion_speed_target_length(length, is_loop, BUDGET_FRAMES, fit_budget=True)
                for _ in range(2000)
            }
        # Floor below R: the augmentation draws in [L/budget, R], always fits.
        over = BUDGET_FRAMES + 10
        got = draws(over)
        assert max(got) == BUDGET_FRAMES and min(got) == int(round(over / 1.2)), sorted(got)
        # Floor above R: one value, exactly the budget.
        assert draws(BUDGET_FRAMES + 30) == {BUDGET_FRAMES}
        # A fitting anchored clip is never slowed back out of the budget, loop
        # or not (a phase-free non-loop clip may be).
        for is_loop in (False, True):
            got = draws(BUDGET_FRAMES - 5, is_loop)
            assert max(got) == BUDGET_FRAMES and min(got) < BUDGET_FRAMES - 5, sorted(got)
        # Past the fit limit the floor is the cap and there is no ceiling.
        assert draws(beyond) == {beyond_fitted}


def test_long_phase_anchored_clip_is_fitted_whole_not_cropped() -> None:
    dataset = _build_truebones(
        split="train",
        num_frames=NUM_FRAMES,
        objects_subset=ANCHORED_LOOP_SUBSET,
        motion_cache_size=0,
    )
    motion_dataset = dataset.motion_dataset
    source_data = motion_dataset.data_dict[ANCHORED_LOOP_MOTION]
    assert not loop_is_phase_free(source_data["motion_metadata"].get("action_label"))
    source_raw = np.load(source_data["motion_path"]).astype(np.float32, copy=False)
    cond = motion_dataset.cond_dict[source_data["object_type"]]

    def prepare(long_len, randint):
        repeat_count = (long_len + source_raw.shape[0] - 1) // source_raw.shape[0]
        long_raw = np.tile(source_raw, (repeat_count, 1, 1))[:long_len]
        with tempfile.TemporaryDirectory() as tmp_dir:
            motion_path = os.path.join(tmp_dir, "long_anchored.npy")
            np.save(motion_path, long_raw)
            long_data = dict(source_data)
            long_data["motion_path"] = motion_path
            long_data["length"] = long_len
            long_data["motion_metadata"] = dict(source_data["motion_metadata"])
            long_data["motion_metadata"]["is_loop"] = False
            with patch.object(dataset_module.random, 'randint', randint):
                sample = motion_dataset._prepare_sample(
                    "synthetic_long_anchored.npy", long_data,
                    target_num_frames=NUM_FRAMES, return_aug_info=True,
                )
        return long_raw, sample[0], sample[-1]

    # Within the fit limit: played faster into exactly the budget, no crop draw.
    long_raw, motion, aug_info = prepare(
        BUDGET_FRAMES + 30, randint=lambda *_: (_ for _ in ()).throw(AssertionError("must not crop")),
    )
    fitted_raw, speed = time_scale_motion_features(long_raw, BUDGET_FRAMES)
    assert_close("fitted anchored window", motion, _resample_raw_then_normalize(fitted_raw, cond, NUM_FRAMES), atol=3e-5)
    assert np.isclose(float(aug_info["motion_speed_applied"]), speed) and speed > 1.0
    assert np.isclose(float(aug_info["resample_speed_cond"]), MAX_SOURCE_FRAMES_MULT)

    # Past it: sped up by the cap first, then cropped to the budget.
    window_start = 3
    beyond = int(MAX_FIT_SPEEDUP * BUDGET_FRAMES) + 20
    long_raw, motion, aug_info = prepare(beyond, randint=lambda *_: window_start)
    scaled_raw, _speed = time_scale_motion_features(long_raw, int(round(beyond / MAX_FIT_SPEEDUP)))
    expected = _resample_raw_then_normalize(
        scaled_raw[window_start:window_start + BUDGET_FRAMES], cond, NUM_FRAMES
    )
    assert_close("capped anchored crop", motion, expected, atol=3e-5)
    assert np.isclose(float(aug_info["resample_speed_cond"]), MAX_SOURCE_FRAMES_MULT)


def test_motion_speed_sampler_rejects_bad_settings() -> None:
    for ratio, prob in ((0.8, 1.0), (1.2, 1.5), (1.2, -0.1)):
        sampler = _speed_sampler(ratio, prob)
        try:
            sampler._sample_motion_speed_target_length(31, False, BUDGET_FRAMES)
        except ValueError:
            pass
        else:
            raise AssertionError(f"ratio={ratio} prob={prob} should be rejected")


class _TileOpt:
    def __init__(self, single_prob):
        self.loop_tile_single_prob = single_prob


def _tile_sampler(single_prob):
    sampler = dataset_module.MotionDataset.__new__(dataset_module.MotionDataset)
    sampler.opt = _TileOpt(single_prob)
    return sampler


def test_loop_tile_sampler_floor_lifts_single_cycle_and_zero_is_uniform() -> None:
    rng = random.Random(3)
    draws = 30000
    with patch.object(dataset_module.random, 'random', rng.random),             patch.object(dataset_module.random, 'randint', rng.randint):
        # 20-frame loop in a 120-frame budget: 6 tile counts. 0.0 is the
        # plain uniform draw, one cycle 1 time in 6.
        counts = [0] * 7
        for _ in range(draws):
            counts[_tile_sampler(0.0)._sample_loop_tile_count(20, BUDGET_FRAMES)] += 1
        assert counts[0] == 0
        for k in range(1, 7):
            assert abs(counts[k] / draws - 1 / 6) < 0.01, counts
        # A 0.5 floor: one cycle half the time, the rest still uniform over 2..6.
        counts = [0] * 7
        for _ in range(draws):
            counts[_tile_sampler(0.5)._sample_loop_tile_count(20, BUDGET_FRAMES)] += 1
        assert abs(counts[1] / draws - 0.5) < 0.01, counts
        for k in range(2, 7):
            assert abs(counts[k] / draws - 0.1) < 0.01, counts
        # A floor below the uniform share changes nothing: it is a floor.
        counts = [0] * 3
        for _ in range(draws):
            counts[_tile_sampler(0.2)._sample_loop_tile_count(60, BUDGET_FRAMES)] += 1
        assert abs(counts[1] / draws - 0.5) < 0.01, counts
    # No draw at all when the clip cannot tile.
    with patch.object(dataset_module.random, 'random', side_effect=AssertionError("must not draw")),             patch.object(dataset_module.random, 'randint', side_effect=AssertionError("must not draw")):
        assert _tile_sampler(0.5)._sample_loop_tile_count(BUDGET_FRAMES + 1, BUDGET_FRAMES) == 1
        assert _tile_sampler(0.5)._sample_loop_tile_count(61, BUDGET_FRAMES) == 1


def main() -> None:
    test_loop_padding_updates_effective_length()
    print("loop padding regression: ok")

    test_loop_padding_random_offset_wraps_without_truncation()
    print("loop random offset regression: ok")

    test_long_motion_crops_fixed_length_random_window()
    print("fixed-length crop window regression: ok")

    test_prepare_sample_aug_info_reports_actual_loop_fill()
    print("loop aug-info regression: ok")

    print("all regression checks passed")


if __name__ == "__main__":
    main()
