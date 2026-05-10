import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion_lib.Animation import Animation
from motion_lib.Quaternions import Quaternions

from data_loaders.truebones.truebones_utils.get_opt import get_opt
from data_loaders.truebones.truebones_utils.motion_labels import load_motion_metadata
from data_loaders.truebones.truebones_utils.motion_process import (
    ROOT_XZ_STRIP_THRESHOLD,
    _xz_locomotion_extent,
    infer_translation_root_index_from_features,
    mirror_features_with_safeguards,
    move_xz_to_origin,
    positions_global,
    recover_animation_from_motion_np,
    recover_from_bvh_ric_np,
)


def _identity_cont6d() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)


def _load_motion_metadata_entry(opt, motion_name: str) -> dict[str, object]:
    data_root = opt.data_root
    if not os.path.isabs(data_root):
        data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), data_root)
    motion_metadata = load_motion_metadata(data_root).get(motion_name)
    if not isinstance(motion_metadata, dict):
        raise AssertionError(f"missing motion metadata for {motion_name}")
    return motion_metadata


def _with_translation_root_index(motion_metadata, motion: np.ndarray, cond) -> dict[str, object]:
    if 'translation_root_index' in motion_metadata:
        return motion_metadata
    updated_motion_metadata = dict(motion_metadata)
    updated_motion_metadata['translation_root_index'] = infer_translation_root_index_from_features(
        motion,
        cond['parents'],
        cond['offsets'],
    )
    return updated_motion_metadata


def test_recover_animation_uses_effective_translation_root_feature_row():
    parents = np.array([-1, 0, 1], dtype=np.int64)
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    )

    frames = 4
    features = np.zeros((frames, 3, 13), dtype=np.float32)
    features[:, :, 3:9] = _identity_cont6d()

    trajectory_x = np.arange(frames, dtype=np.float32)

    # Joint 2 is the effective translation root: its RIFKE XZ stays at zero, and
    # its X velocity carries the trajectory. Joint 0 remains globally fixed.
    features[:, 0, 0] = -trajectory_x
    features[:, 0, 2] = 1.0
    features[:, 1, 0] = -trajectory_x
    features[:, 1, 1] = 1.0
    features[:, 1, 2] = 1.0
    features[:, 2, 1] = 1.0
    features[:-1, 2, 9] = 1.0

    anim, has_animated_pos = recover_animation_from_motion_np(
        features,
        parents,
        offsets,
        translation_root_index=2,
    )
    global_pos = positions_global(anim)

    np.testing.assert_allclose(global_pos[:, 0], np.array([[0.0, 0.0, 1.0]] * frames, dtype=np.float32), atol=1e-5)
    np.testing.assert_allclose(global_pos[:, 1], np.array([[0.0, 1.0, 1.0]] * frames, dtype=np.float32), atol=1e-5)
    np.testing.assert_allclose(
        global_pos[:, 2],
        np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 0.0]], dtype=np.float32),
        atol=1e-5,
    )
    assert has_animated_pos is True


def test_xz_locomotion_extent_ignores_static_origin_offset_after_initial_root_centering():
    parents = np.array([-1, 0], dtype=np.int64)
    offsets = np.zeros((2, 3), dtype=np.float64)
    rotations = Quaternions(
        np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            (3, len(parents), 1),
        )
    )
    positions = np.zeros((3, 2, 3), dtype=np.float64)
    positions[:, 1, 0] = np.array([10.0, 11.0, 10.0], dtype=np.float64)

    anim = Animation(rotations, positions, Quaternions.id(len(parents)), offsets, parents)
    centered_anim, root_translation_xz = move_xz_to_origin(anim)

    np.testing.assert_allclose(root_translation_xz, np.array([10.0, 0.0, 0.0], dtype=np.float64), atol=1e-8)
    assert _xz_locomotion_extent(centered_anim, 1) == pytest.approx(1.0)


def test_xz_locomotion_extent_still_detects_true_locomotion_after_initial_root_centering():
    parents = np.array([-1, 0], dtype=np.int64)
    offsets = np.zeros((2, 3), dtype=np.float64)
    rotations = Quaternions(
        np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            (3, len(parents), 1),
        )
    )
    positions = np.zeros((3, 2, 3), dtype=np.float64)
    positions[:, 1, 0] = np.array([0.0, 2.0, 4.0], dtype=np.float64)

    anim = Animation(rotations, positions, Quaternions.id(len(parents)), offsets, parents)
    centered_anim, root_translation_xz = move_xz_to_origin(anim)

    np.testing.assert_allclose(root_translation_xz, np.array([0.0, 0.0, 0.0], dtype=np.float64), atol=1e-8)
    assert _xz_locomotion_extent(centered_anim, 1) == pytest.approx(4.0)
    assert _xz_locomotion_extent(centered_anim, 1) > ROOT_XZ_STRIP_THRESHOLD


def test_recover_animation_matches_safeguarded_horse_target_globals():
    opt = get_opt(None)
    cond = np.load(opt.cond_file, allow_pickle=True).item()['Horse']

    motion_dir = opt.motion_dir
    if not os.path.isabs(motion_dir):
        motion_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), motion_dir)

    raw = np.load(os.path.join(motion_dir, 'Horse_RunLoop_28.npy')).astype(np.float32, copy=False)
    motion_metadata = _with_translation_root_index(
        _load_motion_metadata_entry(opt, 'Horse_RunLoop_28.npy'),
        raw,
        cond,
    )
    mirrored, mirrored_offsets = mirror_features_with_safeguards(raw, cond, motion_metadata=motion_metadata)
    target_global = recover_from_bvh_ric_np(
        mirrored,
        parents=cond['parents'],
        offsets=mirrored_offsets,
        motion_metadata=motion_metadata,
    )

    anim, has_animated_pos = recover_animation_from_motion_np(
        mirrored,
        cond['parents'],
        mirrored_offsets,
        motion_metadata=motion_metadata,
    )
    recovered_global = positions_global(anim)

    np.testing.assert_allclose(recovered_global, target_global, atol=1e-4)
    assert has_animated_pos is True


def test_from_transforms_preserves_positive_trace_ninety_degree_yaw():
    matrix = np.array(
        [[[-5.2504788e-08, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, -5.2504788e-08]]],
        dtype=np.float32,
    )

    quat = Quaternions.from_transforms(matrix)

    np.testing.assert_allclose(quat.transforms(), matrix.astype(np.float64), atol=1e-6)


def test_recover_animation_hound_mirror_matches_world_x_reflection():
    opt = get_opt(None)
    cond = np.load(opt.cond_file, allow_pickle=True).item()['Hound']

    motion_dir = opt.motion_dir
    if not os.path.isabs(motion_dir):
        motion_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), motion_dir)

    raw = np.load(os.path.join(motion_dir, 'Hound_Attack_402.npy')).astype(np.float32, copy=False)
    motion_metadata = _with_translation_root_index(
        _load_motion_metadata_entry(opt, 'Hound_Attack_402.npy'),
        raw,
        cond,
    )
    mirrored, mirrored_offsets = mirror_features_with_safeguards(raw, cond, motion_metadata=motion_metadata)

    clean_anim, _ = recover_animation_from_motion_np(
        raw,
        cond['parents'],
        cond['offsets'],
        motion_metadata=motion_metadata,
    )
    mirror_anim, _ = recover_animation_from_motion_np(
        mirrored,
        cond['parents'],
        mirrored_offsets,
        motion_metadata=motion_metadata,
    )
    clean_global = positions_global(clean_anim)
    mirror_global = positions_global(mirror_anim)

    spi = np.asarray(cond['symmetry_partner_indices'], dtype=np.int64)
    perm = np.arange(len(spi), dtype=np.int64)
    perm[spi >= 0] = spi[spi >= 0]

    expected_x = clean_global[:, perm].copy()
    expected_x[..., 0] *= -1.0
    reflected_z = clean_global[:, perm].copy()
    reflected_z[..., 2] *= -1.0

    x_error = float(np.abs(expected_x - mirror_global).mean())
    z_error = float(np.abs(reflected_z - mirror_global).mean())

    assert x_error < 1e-5
    assert x_error * 1000.0 < z_error