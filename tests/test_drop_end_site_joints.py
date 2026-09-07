"""Cover the BVH end-site removal that runs during dataset preprocessing.

A tpose that round-tripped through BVH comes back with every ``End Site``
materialised as a real bone. They carry no motion, and each one steals the
``EndEffector``/``ChainEnd`` marker from the joint it hangs off, which rewrites
that joint's canonical name and so its T5 conditioning vector. Preprocessing
therefore removes them outright -- before the crop and independently of it, so
an inference build that keeps every joint still drops the punctuation. The rest
pose decides once and hands the names to every clip of that character; these
tests pin that contract, plus the name rule that keeps real anatomy safe.
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
    drop_end_site_joints,
    find_end_site_joints,
)


# A short chain plus the two end-site spellings a BVH round trip produces, and
# 'Legend' -- a real bone whose name merely ends in "end".
_BODY_NAMES = ['Hips', 'Spine', 'Head', 'Legend', 'Tail01']
_BODY_PARENTS = [-1, 0, 1, 0, 0]
_NAMES = _BODY_NAMES + ['Head_end', 'Legend_end_site', 'Tail01 End']
_PARENTS = np.array(_BODY_PARENTS + [2, 3, 4], dtype=np.int32)
_OFFSETS = np.array([[0.0, 1.0, 0.0]] + [[0.0, 0.3, 0.0]] * (len(_NAMES) - 1))


def _make_anim(parents, offsets, frames=3):
    parents = np.asarray(parents, dtype=np.int32)
    n = len(parents)
    rots = Quaternions(np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (frames, n, 1)))
    orients = Quaternions(np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1)))
    positions = np.arange(frames * n * 3, dtype=np.float64).reshape(frames, n, 3)
    return Animation(rots, positions, orients, np.asarray(offsets, dtype=np.float64), parents)


def test_every_end_site_spelling_is_detected_and_real_anatomy_is_not():
    detected = find_end_site_joints(_PARENTS, _NAMES)
    assert {_NAMES[j] for j in detected} == {'Head_end', 'Legend_end_site', 'Tail01 End'}


def test_a_bone_whose_name_merely_ends_in_end_is_kept():
    """'Legend' is a leaf here and still must survive -- the separator is the rule."""
    names = ['Hips', 'Legend']
    parents = np.array([-1, 0], dtype=np.int32)
    assert find_end_site_joints(parents, names) == set()


def test_removal_drops_the_terminators_and_remaps_the_hierarchy():
    anim = _make_anim(_PARENTS, _OFFSETS)
    filtered, names, keep_indices = drop_end_site_joints(anim, _NAMES)

    assert names == _BODY_NAMES
    assert keep_indices == list(range(len(_BODY_NAMES)))
    assert filtered.parents.tolist() == _BODY_PARENTS
    # Per-joint arrays are sliced by the same keep-set; kept joints keep their values.
    np.testing.assert_array_equal(filtered.positions, anim.positions[:, :len(_BODY_NAMES)])
    np.testing.assert_allclose(filtered.offsets, _OFFSETS[:len(_BODY_NAMES)])


def test_stacked_terminators_are_peeled_until_none_remain():
    """'Head_end' is not a leaf until its own '_end_site' child is gone."""
    names = ['Hips', 'Head', 'Head_end', 'Head_end_site']
    parents = np.array([-1, 0, 1, 2], dtype=np.int32)
    assert {names[j] for j in find_end_site_joints(parents, names)} == {
        'Head_end', 'Head_end_site'
    }


def test_a_named_joint_carrying_real_children_is_not_a_terminator():
    """Only leaves qualify, so a mid-chain joint keeps its subtree."""
    names = ['Hips', 'Weird_end', 'Foot']
    parents = np.array([-1, 0, 1], dtype=np.int32)
    assert find_end_site_joints(parents, names) == set()


def test_a_skeleton_with_no_end_sites_is_returned_untouched():
    anim = _make_anim(_BODY_PARENTS, _OFFSETS[:len(_BODY_NAMES)])
    filtered, names, keep_indices = drop_end_site_joints(anim, _BODY_NAMES)

    assert keep_indices is None
    assert names == _BODY_NAMES
    assert filtered is anim


def test_a_clip_follows_the_rest_poses_explicit_names():
    """The rest pose decides once; a clip that re-detected could disagree."""
    anim = _make_anim(_PARENTS, _OFFSETS)
    _filtered, names, keep_indices = drop_end_site_joints(
        anim, _NAMES, drop_names=('Head_end', 'Legend_end_site', 'Tail01 End')
    )
    assert names == _BODY_NAMES
    assert keep_indices == list(range(len(_BODY_NAMES)))


def test_a_name_the_clips_rig_does_not_carry_raises():
    anim = _make_anim(_BODY_PARENTS, _OFFSETS[:len(_BODY_NAMES)])
    with pytest.raises(ValueError, match='missing from this skeleton'):
        drop_end_site_joints(anim, _BODY_NAMES, drop_names=('Head_end',))


def test_dropping_a_named_joint_that_still_has_children_raises():
    """The keep-set guard is what proves an explicit list really named leaves."""
    names = ['Hips', 'Weird_end', 'Foot']
    parents = np.array([-1, 0, 1], dtype=np.int32)
    anim = _make_anim(parents, _OFFSETS[:3])
    with pytest.raises(ValueError, match='parent'):
        drop_end_site_joints(anim, names, drop_names=('Weird_end',))
