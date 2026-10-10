"""Detached control roots: on the ground, with their only child well above."""
import numpy as np

from motion_lib.root_collapse import detached_root_depth

# Crow after the loader: a ground-level Pelvis under the body's Spine.
NAMES = ["Pelvis", "Spine", "Spine1", "LeftFoot", "RightFoot", "Neck"]
PARENTS = np.array([-1, 0, 1, 2, 2, 1])
REST = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.4, 0.0],
    [0.0, 0.4, 0.0],
    [0.06, 0.0, -0.1],
    [-0.06, 0.0, -0.1],
    [0.0, 0.5, 0.4],
])


def test_ground_mover_is_counted():
    assert detached_root_depth(NAMES, PARENTS, REST) == 1


def test_coincident_root_is_kept():
    # A Biped Bip01 on its pelvis: no rise to its child.
    rest = REST.copy()
    rest[0] = [0.0, 0.4, 0.0]
    assert detached_root_depth(NAMES, PARENTS, rest) == 0


def test_raised_root_is_kept():
    # A body root that sits below its child but off the ground (RMW_Spider's Body).
    rest = REST.copy()
    rest[0] = [0.0, 0.15, 0.0]
    assert detached_root_depth(NAMES, PARENTS, rest) == 0


def test_ground_root_with_child_beside_it_is_kept():
    # A chain lying on the ground (a worm, a fish's body_1): no rise.
    rest = REST.copy()
    rest[1] = rest[2] = [0.0, 0.03, 0.3]
    assert detached_root_depth(NAMES, PARENTS, rest) == 0


def test_branching_root_is_kept():
    parents = PARENTS.copy()
    parents[5] = 0
    assert detached_root_depth(NAMES, parents, REST) == 0


def test_short_body_chain_is_kept():
    # OneEyeAlien: a deforming Root under an unbranched Spine -> Head. The
    # height rule alone would fold Root as a detached control root.
    names = ["Root", "Spine", "Head"]
    parents = np.array([-1, 0, 1])
    rest = np.array([[0.0, 0.159, 0.0], [0.0, 0.568, 0.0], [0.0, 1.0, 0.0]])
    assert detached_root_depth(names, parents, rest) == 0


# Eagle after the loader: a control Hips floating far above its Bip01_Pelvis.
OUTLIER_NAMES = ["Hips", "Pelvis", "Spine", "LeftThigh", "RightThigh", "Neck", "Head"]
OUTLIER_PARENTS = np.array([-1, 0, 1, 2, 2, 2, 5])
OUTLIER_REST = np.array([
    [0.0, 1.6, 0.0],
    [0.0, 0.4, 0.0],
    [0.0, 0.45, 0.1],
    [0.1, 0.35, 0.05],
    [-0.1, 0.35, 0.05],
    [0.0, 0.55, 0.25],
    [0.0, 0.6, 0.35],
])


def test_outlier_root_is_counted():
    assert detached_root_depth(OUTLIER_NAMES, OUTLIER_PARENTS, OUTLIER_REST) == 1


def test_perched_head_root_is_kept():
    # RU01_BallRobot's RigHead: just above the body on an ordinary bone.
    rest = OUTLIER_REST.copy()
    rest[0] = [0.0, 0.62, 0.0]
    assert detached_root_depth(OUTLIER_NAMES, OUTLIER_PARENTS, rest) == 0


def test_outside_root_on_ordinary_bone_is_kept():
    # Oscafish: outside the body's bounds, but its bone is no longer than the rest.
    names = ["Root", "Body", "Tail", "FinL", "FinR"]
    parents = np.array([-1, 0, 1, 1, 1])
    rest = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, -0.2],
        [0.0, 0.0, -0.4],
        [0.1, 0.0, -0.25],
        [-0.1, 0.0, -0.25],
    ])
    assert detached_root_depth(names, parents, rest) == 0
