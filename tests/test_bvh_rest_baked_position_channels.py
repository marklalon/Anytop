"""Regression: BVH export must write position channels after rest-rotation baking.

``recover_bvh_export_animation_from_motion_np`` bakes the T-pose rest rotations
into the local rotations (``recover_processed_animation_from_feature_animation``)
while leaving the offsets in the rest-removed feature basis. The solved local
positions therefore deviate from the rest offsets even when the *pre-bake*
feature animation was pure rotation (``has_animated_pos == False``).

The bug: the returned ``has_animated_pos`` flag was computed on the pre-bake
feature animation, so callers exported rotation-only BVH (``positions=False``)
for pure-rotation references on skeletons with non-identity rest rotations (e.g.
GLB-derived references). Rotation-only reconstruction then produced a garbled
pose because the rotation/offset pair is not FK-consistent in that basis.

These tests lock the invariant that drives the fix:
  * baking a non-identity rest into a pure-rotation feature animation makes
    ``needs_bvh_position_channels`` True (positions deviate from offsets);
  * a BVH saved with that flag round-trips back to the same world pose.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile

import numpy as np

_ANYTOP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ANYTOP_ROOT not in sys.path:
    sys.path.insert(0, _ANYTOP_ROOT)

import pytest

from motion_lib.Animation import Animation, positions_global
from motion_lib.Quaternions import Quaternions
from data_loaders.truebones.truebones_utils.animation_utils import needs_bvh_position_channels
from data_loaders.truebones.truebones_utils.features import (
    recover_processed_animation_from_feature_animation,
    recover_bvh_export_animation_from_motion_np,
    recover_animation_from_motion_np,
    recover_from_bvh_rot_np,
    recover_root_quat_and_pos_np,
    get_rifke,
)


def _axis_angle_wxyz(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    half = angle * 0.5
    return np.array([math.cos(half), *(math.sin(half) * axis)], dtype=np.float64)


def _build_pure_rotation_feature_anim(frames=5):
    """Build a feature-space Animation: identity orients, positions == offsets.

    This mirrors the shape ``recover_animation_from_motion_np`` returns for a
    pure-rotation clip (no animated position channels), which is exactly the case
    that produced ``has_animated_pos == False`` before baking.
    """
    parents = np.array([-1, 0, 1, 2], dtype=np.int32)
    offsets = np.array(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.0, 0.0]],
        dtype=np.float64,
    )
    joints = len(parents)

    rotations = np.zeros((frames, joints, 4), dtype=np.float64)
    rotations[..., 0] = 1.0
    swing = np.linspace(0.0, math.pi * 0.3, frames)
    for f in range(frames):
        rotations[f, 1] = _axis_angle_wxyz([0.0, 0.0, 1.0], swing[f])
        rotations[f, 2] = _axis_angle_wxyz([1.0, 0.0, 0.0], swing[f] * 0.5)

    positions = np.repeat(offsets[None], frames, axis=0)
    # Root carries a little translation; non-root positions stay at rest offsets
    # (pure rotation -> needs_bvh_position_channels(feature_anim) is False).
    positions[:, 0, 1] = np.linspace(0.0, 0.5, frames)

    feature_anim = Animation(
        rotations=Quaternions(rotations),
        positions=positions,
        orients=Quaternions.id(joints),
        offsets=offsets,
        parents=parents,
    )
    return feature_anim


def test_pure_rotation_feature_anim_has_no_position_channels():
    """Sanity: the pre-bake feature animation is genuinely pure-rotation."""
    feature_anim = _build_pure_rotation_feature_anim()
    assert needs_bvh_position_channels(feature_anim) is False


def test_baking_nonidentity_rest_requires_position_channels():
    """The bug: rest baking deviates positions from offsets -> need channels."""
    feature_anim = _build_pure_rotation_feature_anim()
    joints = feature_anim.shape[1]

    # Non-identity rest rotations on the chain (the GLB-derived skeleton case).
    rest = np.zeros((joints, 4), dtype=np.float64)
    rest[:, 0] = 1.0
    rest[1] = _axis_angle_wxyz([0.0, 0.0, 1.0], math.pi * 0.5)
    rest[2] = _axis_angle_wxyz([1.0, 0.0, 0.0], math.pi * 0.5)

    baked = recover_processed_animation_from_feature_animation(feature_anim, rest)

    # Global pose is preserved by the bake (the whole point of solving positions).
    pre = positions_global(feature_anim)
    post = positions_global(baked)
    assert np.max(np.abs(pre - post)) < 1e-4

    # ...but the baked local positions now deviate from the rest offsets, so a
    # rotation-only BVH would be wrong. The export flag must reflect the baked
    # animation, not the pure-rotation feature animation.
    assert needs_bvh_position_channels(baked) is True


def test_identity_rest_keeps_pure_rotation():
    """With identity rest, baking is a no-op -> still no position channels."""
    feature_anim = _build_pure_rotation_feature_anim()
    joints = feature_anim.shape[1]
    identity_rest = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (joints, 1))
    baked = recover_processed_animation_from_feature_animation(feature_anim, identity_rest)
    assert needs_bvh_position_channels(baked) is False


def test_bvh_roundtrip_with_position_channels_after_baking():
    """A BVH saved with the (correct) flag round-trips to the same world pose."""
    from motion_lib import BVH

    feature_anim = _build_pure_rotation_feature_anim()
    joints = feature_anim.shape[1]
    rest = np.zeros((joints, 4), dtype=np.float64)
    rest[:, 0] = 1.0
    rest[1] = _axis_angle_wxyz([0.0, 0.0, 1.0], math.pi * 0.5)
    rest[2] = _axis_angle_wxyz([1.0, 0.0, 0.0], math.pi * 0.5)

    baked = recover_processed_animation_from_feature_animation(feature_anim, rest)
    needs_pos = needs_bvh_position_channels(baked)
    assert needs_pos is True

    names = ["root", "j1", "j2", "j3"]
    pre = positions_global(baked)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "rest_baked.bvh")
        BVH.save(out, baked, names, frametime=1.0 / 30.0, positions=needs_pos)
        loaded, _names, _ft = BVH.load(out)
    post = positions_global(loaded)

    j = min(pre.shape[1], post.shape[1])
    err = np.max(np.linalg.norm(pre[:, :j] - post[:, :j], axis=-1))
    assert err < 1e-3, f"BVH roundtrip world-position error too large: {err}"


_BUFFALO_NPY = os.path.join(
    _ANYTOP_ROOT, "dataset", "truebones", "zoo", "truebones_processed",
    "motions", "Buffalo_RunLoop_1.npy",
)
_COND = os.path.join(
    _ANYTOP_ROOT, "dataset", "truebones", "zoo", "truebones_processed", "cond.npy",
)


def _make_pure_rotation_feature_tensor(feats, parents, offsets, translation_root_index):
    """Rewrite the RIC channels of a real feature tensor so they are FK-consistent
    with its own rotation channels, yielding a *pure-rotation* clip (no animated
    non-root positions). This is exactly the condition under which the pre-bake
    ``has_animated_pos`` is False, so it isolates the rest-baking bug."""
    _, anim_rot = recover_from_bvh_rot_np(
        feats, parents, offsets,
        translation_root_index=translation_root_index, allow_infer=True,
    )
    glob_rot = positions_global(anim_rot)
    r_rot_quat, _r_pos = recover_root_quat_and_pos_np(
        feats, translation_root_index=translation_root_index,
        parents=parents, offsets=offsets, allow_infer=True,
    )
    ric = get_rifke(glob_rot, r_rot_quat, translation_root_index)
    pure = feats.copy()
    pure[..., 0:3] = ric.astype(pure.dtype)
    return pure


def test_export_returns_position_channels_for_pure_rotation_with_rest():
    """End-to-end guard for the fix through the public export entry point.

    A pure-rotation feature tensor has ``has_animated_pos == False`` before
    baking. With non-identity rest rotations, ``recover_bvh_export_...`` bakes the
    rest into the rotations, which moves the local positions off the offsets, so
    the returned flag MUST flip to True. Before the fix it leaked the stale
    pre-bake False, producing garbled rotation-only BVH."""
    if not (os.path.isfile(_BUFFALO_NPY) and os.path.isfile(_COND)):
        pytest.skip("Buffalo NPY / cond.npy fixtures not available")

    cond = np.load(_COND, allow_pickle=True).item()["Buffalo"]
    parents = np.asarray(cond["parents"], dtype=np.int32)
    offsets = np.asarray(cond["offsets"], dtype=np.float32)
    names = list(cond.get("canonical_bvh_joint_names", cond["joints_names"]))
    raw = np.load(_BUFFALO_NPY).astype(np.float32)
    tri = int(cond.get("translation_root_index", 0) or 0)

    feats = _make_pure_rotation_feature_tensor(raw, parents, offsets, tri)

    # Precondition: the synthesized clip is genuinely pure-rotation pre-bake.
    _anim, pre_bake_flag = recover_animation_from_motion_np(
        feats, parents, offsets, translation_root_index=tri, allow_infer=True,
    )
    assert pre_bake_flag is False, "fixture is not pure-rotation; test cannot isolate the bug"

    # Non-identity rest rotations (the GLB-derived skeleton case).
    joints = len(parents)
    rest = np.zeros((joints, 4), dtype=np.float64)
    rest[:, 0] = 1.0
    rest[1] = _axis_angle_wxyz([0.0, 0.0, 1.0], math.pi * 0.5)
    rest[2] = _axis_angle_wxyz([1.0, 0.0, 0.0], math.pi * 0.4)

    anim, _names, has_animated_pos = recover_bvh_export_animation_from_motion_np(
        feats, parents, offsets, names,
        translation_root_index=tri, allow_infer=True, tpose_rest_rotations=rest,
    )
    assert anim is not None
    # The fix: the returned flag describes the baked animation that is saved.
    assert has_animated_pos == needs_bvh_position_channels(anim)
    assert has_animated_pos is True
