import numpy as np

from data_loaders.truebones.truebones_utils.motion_process import (
    infer_translation_root_from_features,
    positions_global,
    recover_animation_from_motion_np,
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