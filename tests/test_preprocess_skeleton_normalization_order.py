"""The order preprocessing normalizes a loaded skeleton in.

``_load_motion_source`` and ``get_common_features_from_rest_pose`` must agree on
one order, or a clip and its rest pose disagree on the joint set:

    drop prop sockets -> drop end sites -> fold the wrapper -> crop to MAX_JOINTS

The fold sits between the drops and the crop for two independent reasons, one
pinned by each of the first two tests here. The third pins the import cache that
lets the root-confirmation loop run a second pass without a second bpy import.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion_lib.Animation import Animation
from motion_lib.Quaternions import Quaternions

from data_loaders.truebones.truebones_utils import dataset_pipeline as dataset_pipeline_mod
from data_loaders.truebones.truebones_utils.animation_utils import (
    promote_translation_root_to_hierarchy_root,
)
from data_loaders.truebones.truebones_utils.param_utils import MAX_JOINTS


def _anim(parents, offsets=None, frames=2):
    parents = np.asarray(parents, dtype=np.int32)
    n = len(parents)
    rotations = Quaternions(np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (frames, n, 1)))
    orients = Quaternions(np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1)))
    if offsets is None:
        offsets = np.tile(np.array([0.0, 1.0, 0.0]), (n, 1))
    offsets = np.asarray(offsets, dtype=np.float64)
    positions = np.tile(offsets[None], (frames, 1, 1))
    return Animation(rotations, positions, orients, offsets, parents)


def _wrapped_body(wrapper_depth, body_joints):
    """``wrapper_depth`` control nodes above a body of ``body_joints`` joints.

    The body is a hip with a fan of single-bone limbs, so cropping it has an
    obvious victim: the deepest, shortest leaf.
    """
    parents = list(range(-1, wrapper_depth - 1))          # wrapper chain
    names = [f"Ctrl{i}" for i in range(wrapper_depth)]
    hip = wrapper_depth
    parents.append(wrapper_depth - 1)
    names.append("Hips")
    for leaf in range(body_joints - 1):
        parents.append(hip)
        names.append(f"Limb{leaf:03d}")
    offsets = np.tile(np.array([0.0, 1.0, 0.0]), (len(parents), 1))
    # Shrinking limbs, so the crop's "shortest bone first" rule has a strict order.
    offsets[hip + 1:, 1] = np.linspace(1.0, 0.5, body_joints - 1)
    return _anim(parents, offsets=offsets), names


def test_the_wrapper_is_folded_before_the_crop_so_real_bones_keep_the_budget():
    """Two control nodes must not cost two anatomical joints at the cap."""
    anim, names = _wrapped_body(wrapper_depth=2, body_joints=MAX_JOINTS)
    assert len(names) == MAX_JOINTS + 2

    _folded, kept_names, _frame_time = dataset_pipeline_mod._load_motion_source(
        "/fake/Wrapped-Walk.glb",
        "Wrapped",
        promote_root_depth=2,
        raw_load_cache={os.path.realpath("/fake/Wrapped-Walk.glb"): (anim, names, 1 / 30)},
    )

    # The fold freed exactly the two slots the crop would otherwise have taken
    # out of the body, so nothing anatomical is dropped at all.
    assert len(kept_names) == MAX_JOINTS
    assert kept_names == names[2:]

    # Cropping first is what that costs: the two deepest, shortest limbs go, and
    # the wrapper is folded away afterwards regardless.
    from data_loaders.truebones.truebones_utils.animation_utils import (
        crop_animation_to_max_joints,
    )
    cropped, cropped_names, _ = crop_animation_to_max_joints(
        anim, names, max_joints=MAX_JOINTS
    )
    _, crop_first_names, _ = promote_translation_root_to_hierarchy_root(
        cropped, cropped_names, 2
    )
    assert len(crop_first_names) == MAX_JOINTS - 2
    assert set(kept_names) - set(crop_first_names) == {"Limb098", "Limb097"}


def test_the_fold_runs_after_the_drops_so_a_socket_sibling_cannot_block_it():
    """A parked weapon hanging off the wrapper is rig furniture, not a branch."""
    # Wrapper -> {Hips -> Tail, Weapon}: the wrapper looks branched until the
    # socket is dropped.
    parents = [-1, 0, 1, 0]
    names = ["Wrapper", "Hips", "Tail", "Weapon"]
    anim = _anim(parents)

    # Folding straight off the load refuses: the root still has two children.
    with pytest.raises(ValueError, match="more than one child"):
        promote_translation_root_to_hierarchy_root(anim, names, 1, context="Ogre")

    _folded, kept_names, _frame_time = dataset_pipeline_mod._load_motion_source(
        "/fake/Ogre-Idle.glb",
        "Ogre",
        prop_socket_names=("Weapon",),
        promote_root_depth=1,
        raw_load_cache={os.path.realpath("/fake/Ogre-Idle.glb"): (anim, names, 1 / 30)},
    )

    assert kept_names == ["Hips", "Tail"]


def test_the_raw_import_is_reused_across_root_confirmation_passes(monkeypatch):
    """Pass 2 re-normalizes at a new depth; it must not re-import the file."""
    anim, names = _wrapped_body(wrapper_depth=1, body_joints=4)
    loads = []

    def fake_load(path, *_args, **_kwargs):
        loads.append(path)
        return anim, list(names), 1 / 30

    monkeypatch.setattr(dataset_pipeline_mod, "FBX", SimpleNamespace(load=fake_load))

    cache = {}
    _a, pass1_names, _ = dataset_pipeline_mod._load_motion_source(
        "/fake/Wrapped-Run.glb", "Wrapped", promote_root_depth=0, raw_load_cache=cache,
    )
    _b, pass2_names, _ = dataset_pipeline_mod._load_motion_source(
        "/fake/Wrapped-Run.glb", "Wrapped", promote_root_depth=1, raw_load_cache=cache,
    )

    assert loads == ["/fake/Wrapped-Run.glb"]
    assert pass1_names == names
    assert pass2_names == names[1:]
    # The cached load itself is untouched by either normalization.
    assert cache[os.path.realpath("/fake/Wrapped-Run.glb")][1] == names

    # No cache means the old behaviour: one import per pass.
    dataset_pipeline_mod._load_motion_source("/fake/Wrapped-Run.glb", "Wrapped")
    dataset_pipeline_mod._load_motion_source("/fake/Wrapped-Run.glb", "Wrapped")
    assert len(loads) == 3
