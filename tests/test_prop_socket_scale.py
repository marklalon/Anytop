"""Cover the prop-socket exclusion in scale normalization.

A held weapon parked in its own bone far from the body (MLH_Archer's ``Bow``/
``Arrow`` at (+/-3, 1, 0) on a 1.03-tall archer) sets both scale-normalization
statistics and shrinks the character ~3x; these tests pin the exclusion and the
frame-independent rest-pose measurement it relies on.
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

from data_loaders.truebones.truebones_utils.animation_utils import (  # noqa: E402
    compute_scale_factor,
    find_prop_socket_joints,
    get_average_axial_bone_length,
    get_rest_body_max_span,
    get_scale_reference_extent,
    max_joint_span,
    rest_positions_from_offsets,
)


# MLH_Archer's real rest pose: a 1.03-tall chibi archer plus the two sockets the
# T-pose parks at (+/-3, 1, 0).  Offsets are parent-to-child deltas of the rest
# pose, so accumulating them reproduces it.
_ARCHER_NAMES = [
    'Hips', 'Spine', 'Head', 'Armor_L',
    'Upper_Arm_L', 'Lower_Arm_L', 'Hand_L', 'Index_Proximal_L',
    'Quiver', 'Armor_R', 'Upper_Arm_R', 'Rower_Arm_R',
    'Hand_R', 'Index_Proximal_R', 'Upper_Leg_L', 'Lower_Leg_L',
    'Foot_L', 'Toes_L', 'Upper_Reg_R', 'Rower_Reg_R',
    'Foot_R', 'Toes_R', 'Bow', 'Arrow',
]
_ARCHER_PARENTS = np.array([-1, 0, 1, 1, 1, 4, 5, 6, 1, 1, 1, 10, 11, 12, 0, 14, 15, 16, 0, 18, 19, 20, 0, 0], dtype=np.int32)
_ARCHER_OFFSETS = np.array([
    [ 0.0000,  0.4346, -0.0284],  # Hips
    [ 0.0000,  0.2871,  0.0939],  # Spine
    [ 0.0000,  0.3631, -0.0227],  # Head
    [ 0.5022,  0.2349, -0.0655],  # Armor_L
    [ 0.2604,  0.0844, -0.0627],  # Upper_Arm_L
    [ 0.4445, -0.0000, -0.0375],  # Lower_Arm_L
    [ 0.3060,  0.0000,  0.0346],  # Hand_L
    [ 0.2194, -0.0000,  0.0029],  # Index_Proximal_L
    [-0.2495,  0.2558, -0.4580],  # Quiver
    [-0.5022,  0.2349, -0.0655],  # Armor_R
    [-0.2604,  0.0844, -0.0627],  # Upper_Arm_R
    [-0.4445, -0.0000, -0.0375],  # Rower_Arm_R
    [-0.3060,  0.0000,  0.0346],  # Hand_R
    [-0.2194,  0.0000,  0.0029],  # Index_Proximal_R
    [ 0.1920, -0.0579, -0.0196],  # Upper_Leg_L
    [ 0.0206, -0.1941,  0.0272],  # Lower_Leg_L
    [ 0.0130, -0.1225, -0.0318],  # Foot_L
    [ 0.0001, -0.0012,  0.3041],  # Toes_L
    [-0.1920, -0.0579, -0.0196],  # Upper_Reg_R
    [-0.0206, -0.1941,  0.0272],  # Rower_Reg_R
    [-0.0130, -0.1225, -0.0318],  # Foot_R
    [-0.0001, -0.0012,  0.3041],  # Toes_R
    [ 3.0000,  0.5654,  0.0284],  # Bow
    [-3.0000,  0.5654,  0.0284],  # Arrow
], dtype=np.float64)
_ARCHER_SIDES = [
    'center', 'center', 'center', 'left',
    'left', 'left', 'left', 'left',
    'center', 'right', 'right', 'right',
    'right', 'right', 'left', 'left',
    'left', 'left', 'right', 'right',
    'right', 'right', 'center', 'center',
]

# The sockets are the last two joints, so the body is a prefix of the skeleton.
_BODY_JOINTS = len(_ARCHER_NAMES) - 2
assert _ARCHER_NAMES[_BODY_JOINTS:] == ['Bow', 'Arrow']


def _body_only(array):
    return array[:_BODY_JOINTS]


def _rest_of(offsets, parents):
    return rest_positions_from_offsets(offsets, parents)


_ARCHER_REST = _rest_of(_ARCHER_OFFSETS, _ARCHER_PARENTS)


def test_parked_sockets_are_found_and_body_bones_are_not():
    prop_joints = find_prop_socket_joints(_ARCHER_OFFSETS, _ARCHER_PARENTS, _ARCHER_NAMES)
    assert {_ARCHER_NAMES[j] for j in prop_joints} == {'Bow', 'Arrow'}


def test_a_joint_the_rig_names_for_a_body_part_is_never_a_socket():
    """The name guard, on the geometry that would otherwise flag the joint.

    Nothing about the rig changes but the two socket names: a rig that calls
    those bones a tail and a horn is taken at its word and keeps them, because
    geometry alone cannot tell a parked staff from a rat's single-bone tail.
    """
    anatomical = list(_ARCHER_NAMES)
    anatomical[_ARCHER_NAMES.index('Bow')] = 'Tail02'
    anatomical[_ARCHER_NAMES.index('Arrow')] = 'Horn_R'
    assert find_prop_socket_joints(_ARCHER_OFFSETS, _ARCHER_PARENTS, anatomical) == set()


def test_a_non_anatomical_name_alone_is_not_a_socket():
    """The geometry guard, on names that would otherwise flag the joint.

    194 joints across the 260 dataset species are named for something that is
    not a body part -- armor, fur, ponytails, saddles, backpacks -- and they sit
    on the character and belong in its size. Only the parked ones go.
    """
    worn = list(_ARCHER_NAMES)
    worn[_ARCHER_NAMES.index('Quiver')] = 'Backpack'
    assert 'Quiver' not in worn
    props = find_prop_socket_joints(_ARCHER_OFFSETS, _ARCHER_PARENTS, worn)
    assert {worn[j] for j in props} == {'Bow', 'Arrow'}


def test_socket_subtree_is_excluded_with_its_socket():
    # MLS_ElfRanger models the bow as Bow -> Bow02: the short child must not keep
    # the over-long socket bone in, and it goes out with it.
    names = _ARCHER_NAMES + ['Bow02']
    parents = np.append(_ARCHER_PARENTS, _ARCHER_NAMES.index('Bow')).astype(np.int32)
    offsets = np.vstack([_ARCHER_OFFSETS, [[0.0, 0.2, 0.0]]])
    prop_joints = find_prop_socket_joints(offsets, parents, names)
    assert {names[j] for j in prop_joints} == {'Bow', 'Arrow', 'Bow02'}


def test_scale_reference_extent_measures_the_body_not_the_sockets():
    body_span = get_rest_body_max_span(_body_only(_ARCHER_OFFSETS), _body_only(_ARCHER_PARENTS))
    assert max_joint_span(_ARCHER_REST) > 5.9  # bow to arrow
    assert get_scale_reference_extent(_ARCHER_REST, _ARCHER_PARENTS, _ARCHER_NAMES) == body_span


def test_axial_bone_length_ignores_the_socket_bones():
    with_sockets = get_average_axial_bone_length(
        _ARCHER_OFFSETS, _ARCHER_PARENTS, _ARCHER_SIDES, _ARCHER_NAMES
    )
    body_only = get_average_axial_bone_length(
        _body_only(_ARCHER_OFFSETS), _body_only(_ARCHER_PARENTS),
        _ARCHER_SIDES[:_BODY_JOINTS], _body_only(_ARCHER_NAMES)
    )
    assert with_sockets == body_only


def test_scale_factor_matches_the_same_rig_without_sockets():
    def scale_of(offsets, parents, sides, names):
        return compute_scale_factor(
            get_average_axial_bone_length(offsets, parents, sides, names),
            body_max_span=get_scale_reference_extent(_rest_of(offsets, parents), parents, names),
        )

    with_sockets = scale_of(_ARCHER_OFFSETS, _ARCHER_PARENTS, _ARCHER_SIDES, _ARCHER_NAMES)
    body_only = scale_of(
        _body_only(_ARCHER_OFFSETS), _body_only(_ARCHER_PARENTS),
        _ARCHER_SIDES[:_BODY_JOINTS], _body_only(_ARCHER_NAMES)
    )
    assert with_sockets == body_only
    # Regression guard: measured on the two sockets, both statistics point the
    # same way and together shrink the archer by more than half.
    unfiltered = compute_scale_factor(
        float(np.linalg.norm(_ARCHER_OFFSETS[1:], axis=1).mean()),
        body_max_span=max_joint_span(_ARCHER_REST),
    )
    assert with_sockets / unfiltered > 2.0


def test_long_terminal_anatomy_is_kept():
    # A rat's tail is a single bone 5.4x the median and its removal would halve
    # the body span, but it is real anatomy: nothing about it may be filtered.
    names = ['Hips', 'Spine', 'Head', 'Leg_L', 'Foot_L', 'Leg_R', 'Foot_R', 'Tail01', 'Tail02']
    parents = np.array([-1, 0, 1, 0, 3, 0, 5, 0, 7], dtype=np.int32)
    offsets = np.array([
        [0.00, 0.19, 0.00],
        [0.00, 0.11, 0.22],
        [0.00, 0.03, 0.30],
        [0.11, 0.00, 0.00],
        [-0.01, -0.14, 0.09],
        [-0.11, 0.00, 0.00],
        [0.01, -0.14, 0.09],
        [0.00, -0.05, -0.13],
        [0.00, -0.12, -0.84],
    ], dtype=np.float64)
    assert find_prop_socket_joints(offsets, parents, names) == set()
    rest = _rest_of(offsets, parents)
    assert get_scale_reference_extent(rest, parents, names) == max_joint_span(rest)


def test_degenerate_skeletons_are_left_alone():
    # Every internal bone zero-length: the percentile floor keeps the reference
    # positive so the leaves are not all swept up as sockets.
    names = ['Root', 'Spine', 'Wing_L', 'Wing_R']
    parents = np.array([-1, 0, 1, 1], dtype=np.int32)
    offsets = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, -0.5, 0.0],
    ], dtype=np.float64)
    assert find_prop_socket_joints(offsets, parents, names) == set()
    assert find_prop_socket_joints(offsets[:1], parents[:1], names[:1]) == set()


def test_scale_reference_extent_needs_the_fk_rest_pose_not_bone_local_offsets():
    """A rig whose rest rotations are non-identity must not be straightened out.

    Bone-local offsets are stated in the parent bone's frame, so summing them
    unfolds every chain into a line: this L-shaped rig is 1.0 x 1.0 with a span
    of sqrt(2), but summed raw it reads as a 2.0-long stick. Feeding those to the
    scale reference mis-sizes every character with a rotated rest pose -- 0.47x
    to 1.77x across the 260 dataset species.
    """
    from motion_lib.Animation import Animation, positions_global
    from motion_lib.Quaternions import Quaternions

    names = ['Hips', 'Spine', 'Head']
    parents = np.array([-1, 0, 1], dtype=np.int32)
    # Each bone runs 1.0 down its parent's local +Y; the second joint turns 90
    # degrees about Z, so the third bone actually leaves along -X.
    offsets = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    orients = Quaternions.from_euler(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2], [0.0, 0.0, 0.0]]), order='xyz'
    )
    rest_anim = Animation(orients[None].copy(), offsets[None].copy(), orients, offsets, parents)
    rest_positions = positions_global(rest_anim)[0]

    assert np.allclose(rest_positions[2], [-1.0, 1.0, 0.0], atol=1e-6)
    assert get_scale_reference_extent(rest_positions, parents, names) == pytest.approx(np.sqrt(2.0))
    # The straightened-out reading the raw offsets give, for contrast.
    assert get_rest_body_max_span(offsets, parents) == pytest.approx(2.0)


def test_reference_body_length_is_frame_independent():
    """The vertical clamp's band is a species property, not a per-clip one.

    ``_get_reference_body_length`` is handed a multi-frame clip by
    ``clamp_vertical_trajectory``, and ``positions_global`` reads a clip's
    per-frame rotations/positions -- never its orients/offsets. Measuring frame 0
    would tie the band to the pose the clip opens with: an Eagle's frame-0 span
    runs 59.7 on TakeOff against 126.7 on its T-pose, a 2.1x band difference
    between two clips of one species.
    """
    from motion_lib.Animation import Animation
    from motion_lib.Quaternions import Quaternions
    from data_loaders.truebones.truebones_utils.animation_utils import (
        _get_reference_body_length,
    )

    # Two "wings" 1.0 long off a root, spread flat in the rest pose.
    parents = np.array([-1, 0, 0], dtype=np.int32)
    offsets = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64)
    orients = Quaternions.id(3)

    def clip(fold_radians, frames=4):
        rotations = Quaternions.from_euler(
            np.tile(
                np.array([[0.0, 0.0, 0.0], [0.0, 0.0, fold_radians], [0.0, 0.0, -fold_radians]]),
                (frames, 1, 1),
            ),
            order='xyz',
        )
        return Animation(
            rotations, np.tile(offsets, (frames, 1, 1)), orients, offsets.copy(), parents
        )

    spread = _get_reference_body_length(clip(0.0))
    folded = _get_reference_body_length(clip(np.pi / 2))
    assert spread == pytest.approx(2.0)          # the rest pose, both times
    assert folded == pytest.approx(spread)
