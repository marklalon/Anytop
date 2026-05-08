import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.truebones.truebones_utils.get_opt import get_opt
from data_loaders.truebones.truebones_utils.motion_process import (
    infer_translation_root_from_features,
    mirror_features_with_safeguards,
    positions_global,
    recover_animation_from_motion_np,
    recover_from_bvh_ric_np,
)


def _identity_cont6d() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)


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

    assert infer_translation_root_from_features(features) == 2

    anim, has_animated_pos = recover_animation_from_motion_np(features, parents, offsets)
    global_pos = positions_global(anim)

    np.testing.assert_allclose(global_pos[:, 0], np.array([[0.0, 0.0, 1.0]] * frames, dtype=np.float32), atol=1e-5)
    np.testing.assert_allclose(global_pos[:, 1], np.array([[0.0, 1.0, 1.0]] * frames, dtype=np.float32), atol=1e-5)
    np.testing.assert_allclose(
        global_pos[:, 2],
        np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 0.0]], dtype=np.float32),
        atol=1e-5,
    )
    assert has_animated_pos is True


def test_recover_animation_matches_safeguarded_horse_target_globals():
    opt = get_opt(None)
    cond = np.load(opt.cond_file, allow_pickle=True).item()['Horse']

    motion_dir = opt.motion_dir
    if not os.path.isabs(motion_dir):
        motion_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), motion_dir)

    raw = np.load(os.path.join(motion_dir, 'Horse_RunLoop_28.npy')).astype(np.float32, copy=False)
    mirrored, mirrored_offsets = mirror_features_with_safeguards(raw, cond)
    target_global = recover_from_bvh_ric_np(mirrored)

    anim, has_animated_pos = recover_animation_from_motion_np(mirrored, cond['parents'], mirrored_offsets)
    recovered_global = positions_global(anim)

    np.testing.assert_allclose(recovered_global, target_global, atol=1e-4)
    assert has_animated_pos is True