"""Cover the prop-socket removal that runs during dataset preprocessing.

A parked weapon socket is rig furniture, not anatomy: it carries no body motion
to learn, so preprocessing drops it before the MAX_JOINTS crop. The rest pose
decides once and hands the names to every clip of that character -- these tests
pin that contract, because a rest pose and a clip are different files and a
disagreement between them would land as a joint-count mismatch much later.
"""

import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = type("Tensor", (), {})
    sys.modules["torch"] = torch_stub

from motion_lib.Animation import Animation  # noqa: E402
from motion_lib.Quaternions import Quaternions  # noqa: E402

from data_loaders.truebones.truebones_utils.animation_utils import (  # noqa: E402
    drop_prop_socket_joints,
    find_prop_socket_joints,
    reindex_animation_to_kept_joints,
)


# A chibi humanoid with both arms, both legs and a spine -- enough real bones
# that one parked socket cannot drag the 90th-percentile bone length up with it,
# which is how the detector behaves on the 20-60 bone rigs it is calibrated on.
_BODY_NAMES = [
    'Hips', 'Spine', 'Chest', 'Neck', 'Head',
    'Upper_Arm_L', 'Lower_Arm_L', 'Hand_L',
    'Upper_Arm_R', 'Lower_Arm_R', 'Hand_R',
    'Upper_Leg_L', 'Lower_Leg_L', 'Foot_L', 'Toes_L',
    'Upper_Leg_R', 'Lower_Leg_R', 'Foot_R', 'Toes_R',
]
_BODY_PARENTS = [-1, 0, 1, 2, 3, 2, 5, 6, 2, 8, 9, 0, 11, 12, 13, 0, 15, 16, 17]
_BODY_OFFSETS = [[0.0, 1.0, 0.0]] + [[0.0, 0.3, 0.0]] * (len(_BODY_NAMES) - 1)

# The socket the T-pose parks 3 units off the body, carrying a second bone.
_SOCKET_NAMES = ['Sword', 'Sword02']
_SOCKET_PARENTS = [0, 19]
_SOCKET_OFFSETS = [[3.0, 0.0, 0.0], [0.2, 0.0, 0.0]]

_NAMES = _BODY_NAMES + _SOCKET_NAMES
_PARENTS = np.array(_BODY_PARENTS + _SOCKET_PARENTS, dtype=np.int32)
_OFFSETS = np.array(_BODY_OFFSETS + _SOCKET_OFFSETS, dtype=np.float64)


def _make_anim(parents, offsets, frames=3):
    parents = np.asarray(parents, dtype=np.int32)
    n = len(parents)
    rots = Quaternions(np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (frames, n, 1)))
    orients = Quaternions(np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1)))
    positions = np.arange(frames * n * 3, dtype=np.float64).reshape(frames, n, 3)
    return Animation(rots, positions, orients, np.asarray(offsets, dtype=np.float64), parents)


def test_the_fixture_socket_is_what_the_detector_flags():
    """Guard the fixture itself: the tests below are only meaningful if it trips."""
    prop_joints = find_prop_socket_joints(_OFFSETS, _PARENTS, _NAMES)
    assert {_NAMES[j] for j in prop_joints} == {'Sword', 'Sword02'}


def test_detection_removes_the_socket_subtree_and_remaps_the_hierarchy():
    anim = _make_anim(_PARENTS, _OFFSETS)
    filtered, names, keep_indices = drop_prop_socket_joints(anim, _NAMES)

    assert names == _BODY_NAMES
    assert keep_indices == list(range(len(_BODY_NAMES)))
    assert filtered.parents.tolist() == _BODY_PARENTS
    # Every per-joint array is sliced by the same keep-set, and the untouched
    # joints keep their values rather than being recomputed.
    assert filtered.positions.shape == (3, len(_BODY_NAMES), 3)
    np.testing.assert_array_equal(filtered.positions, anim.positions[:, :len(_BODY_NAMES)])
    np.testing.assert_allclose(filtered.offsets, _OFFSETS[:len(_BODY_NAMES)])


def test_a_skeleton_with_no_socket_is_returned_untouched():
    anim = _make_anim(_BODY_PARENTS, _BODY_OFFSETS)
    filtered, names, keep_indices = drop_prop_socket_joints(anim, _BODY_NAMES)

    assert keep_indices is None
    assert filtered is anim
    assert names == _BODY_NAMES


def test_removal_is_idempotent():
    """Dropping the longest bones lowers the reference the gate is measured against.

    Nothing may be promoted over the 3.25x gate by that drop, or the rest pose
    and a clip could settle on different joint sets depending on how many times
    the filter ran.
    """
    anim = _make_anim(_PARENTS, _OFFSETS)
    filtered, names, _ = drop_prop_socket_joints(anim, _NAMES)
    assert find_prop_socket_joints(filtered.offsets, filtered.parents, names) == set()


def test_an_explicit_name_list_drops_exactly_what_detection_did():
    """The rest pose decides; each clip follows the names, never its own detector."""
    anim = _make_anim(_PARENTS, _OFFSETS)
    detected, detected_names, _ = drop_prop_socket_joints(anim, _NAMES)
    followed, followed_names, _ = drop_prop_socket_joints(
        anim, _NAMES, drop_names=('Sword', 'Sword02')
    )

    assert followed_names == detected_names
    np.testing.assert_array_equal(followed.parents, detected.parents)
    np.testing.assert_allclose(followed.offsets, detected.offsets)


def test_following_names_sweeps_in_descendants_the_list_did_not_mention():
    """A clip's rig may hang an extra bone under a socket; it goes with it."""
    anim = _make_anim(_PARENTS, _OFFSETS)
    filtered, names, _ = drop_prop_socket_joints(anim, _NAMES, drop_names=('Sword',))
    assert names == _BODY_NAMES


def test_a_name_the_rig_does_not_carry_is_an_error():
    """Silently skipping it would leave this clip one joint wider than the cond."""
    anim = _make_anim(_BODY_PARENTS, _BODY_OFFSETS)
    with pytest.raises(ValueError, match='Shield'):
        drop_prop_socket_joints(
            anim, _BODY_NAMES, drop_names=('Shield',), context='Knight Walk.fbx'
        )


def test_dropping_the_root_is_refused():
    anim = _make_anim(_BODY_PARENTS, _BODY_OFFSETS)
    with pytest.raises(ValueError, match='root joint'):
        drop_prop_socket_joints(anim, _BODY_NAMES, drop_names=('Hips',))


def test_reindex_rejects_a_keep_set_that_orphans_a_child():
    """Dropping a parent alone would remap its child to a silent second root."""
    anim = _make_anim(_BODY_PARENTS, _BODY_OFFSETS)
    keep_indices = [j for j in range(len(_BODY_NAMES)) if j != 1]  # Spine, mid-chain
    with pytest.raises(ValueError, match='parent'):
        reindex_animation_to_kept_joints(anim, _BODY_NAMES, keep_indices)
