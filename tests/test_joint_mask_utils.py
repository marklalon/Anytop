from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


from model.joint_mask_utils import sample_subtree_joint_mask, sample_subtree_joint_mask_batch  # noqa: E402


# Root 0 with a trunk 1-2, four two-joint limbs and a three-joint tail:
#   0 -> 1 -> 2 ; limbs 3-4, 5-6 off joint 2, 7-8, 9-10 off joint 1 ; tail 11-12-13 off 0.
PARENTS = [-1, 0, 1, 2, 3, 2, 5, 1, 7, 1, 9, 0, 11, 12]
LIMB_ROOTS = (3, 5, 7, 9)
LIMBS = {3: {3, 4}, 5: {5, 6}, 7: {7, 8}, 9: {9, 10}}


def _limb_root_mask(*root_indices: int, joint_count: int = len(PARENTS)) -> np.ndarray:
    mask = np.zeros(joint_count, dtype=bool)
    mask[list(root_indices)] = True
    return mask


def _children(parents):
    children = [[] for _ in parents]
    for joint, parent in enumerate(parents):
        if parent >= 0:
            children[parent].append(joint)
    return children


def _is_whole_subtree(mask, parents) -> bool:
    """Every masked joint's children are masked too, and the root never is."""
    children = _children(parents)
    return not mask[0] and all(mask[c] for j in np.flatnonzero(mask) for c in children[j])


def test_sample_draws_whole_limbs_or_one_bounded_subtree() -> None:
    limb_draws = {1: 0, 2: 0}
    subtree_draws = 0
    for seed in range(2000):
        mask = sample_subtree_joint_mask(
            parents=PARENTS,
            limb_root_mask=_limb_root_mask(*LIMB_ROOTS),
            rng=np.random.default_rng(seed),
        )
        assert mask is not None
        assert _is_whole_subtree(mask, PARENTS)
        masked = set(np.flatnonzero(mask).tolist())
        limbs = [root for root, joints in LIMBS.items() if joints <= masked]
        if limbs and masked == set().union(*(LIMBS[root] for root in limbs)):
            limb_draws[len(limbs)] += 1
        else:
            # One subtree of a non-leaf joint, at most half the rig.
            subtree_draws += 1
            assert len(masked) <= len(PARENTS) // 2
            assert len([j for j in masked if not mask[PARENTS[j]]]) == 1

    # LIMB_MASK_SHARE of the draws go to limbs (plus the one-subtree draws that
    # happen to pick a limb root), split evenly between one and two limbs.
    assert limb_draws[1] > 0 and limb_draws[2] > 0
    assert sum(limb_draws.values()) > subtree_draws


def test_sample_without_two_limbs_draws_only_subtrees() -> None:
    for seed in range(200):
        mask = sample_subtree_joint_mask(
            parents=PARENTS,
            limb_root_mask=_limb_root_mask(3),
            rng=np.random.default_rng(seed),
        )
        assert mask is not None
        assert _is_whole_subtree(mask, PARENTS)
        roots = [j for j in np.flatnonzero(mask) if not mask[PARENTS[j]]]
        assert len(roots) == 1


def test_sample_returns_none_when_nothing_fits() -> None:
    # A chain of three: the only non-leaf subtrees are larger than half the rig.
    mask = sample_subtree_joint_mask(
        parents=[-1, 0, 1],
        limb_root_mask=_limb_root_mask(joint_count=3),
        rng=np.random.default_rng(0),
    )
    assert mask is None


def test_sample_supports_restored_global_numpy_state() -> None:
    np.random.seed(123)
    saved_state = np.random.get_state()
    first_mask = sample_subtree_joint_mask(
        parents=PARENTS, limb_root_mask=_limb_root_mask(*LIMB_ROOTS), rng=np.random,
    )

    np.random.seed(999)
    np.random.set_state(saved_state)
    second_mask = sample_subtree_joint_mask(
        parents=PARENTS, limb_root_mask=_limb_root_mask(*LIMB_ROOTS), rng=np.random,
    )

    assert first_mask is not None
    assert np.array_equal(first_mask, second_mask)


def test_sample_batch_matches_sequential_sampling() -> None:
    max_joints = len(PARENTS)
    second_parents = [-1, 0, 1, 2, 0, 0]
    parents_batch = np.zeros((2, max_joints), dtype=np.int64)
    parents_batch[0] = PARENTS
    parents_batch[1, :len(second_parents)] = second_parents
    limb_root_mask_batch = np.stack(
        [_limb_root_mask(*LIMB_ROOTS), _limb_root_mask(1, 4)], axis=0,
    )
    n_joints = np.asarray([max_joints, len(second_parents)], dtype=np.int64)

    np.random.seed(123)
    batch_mask = sample_subtree_joint_mask_batch(
        parents_batch=parents_batch,
        limb_root_mask_batch=limb_root_mask_batch,
        n_joints=n_joints,
        max_joints=max_joints,
        joint_mask_prob=1.0,
        rng=np.random,
    )

    np.random.seed(123)
    expected = np.zeros((2, max_joints), dtype=bool)
    for batch_index in range(2):
        valid_joint_count = int(n_joints[batch_index])
        per_sample_mask = sample_subtree_joint_mask(
            parents=parents_batch[batch_index, :valid_joint_count].tolist(),
            limb_root_mask=limb_root_mask_batch[batch_index, :valid_joint_count],
            rng=np.random,
        )
        if per_sample_mask is not None:
            expected[batch_index, :valid_joint_count] = per_sample_mask

    assert batch_mask is not None
    assert np.array_equal(batch_mask, expected)
    assert not np.any(batch_mask[1, int(n_joints[1]):])


def test_sample_batch_respects_probability_gate() -> None:
    mask = sample_subtree_joint_mask_batch(
        parents_batch=np.asarray([PARENTS], dtype=np.int64),
        limb_root_mask_batch=np.stack([_limb_root_mask(*LIMB_ROOTS)], axis=0),
        n_joints=np.asarray([len(PARENTS)], dtype=np.int64),
        max_joints=len(PARENTS),
        joint_mask_prob=0.0,
        rng=np.random.default_rng(0),
    )

    assert mask is None


def test_limb_root_mask_walks_up_to_the_midline() -> None:
    from data_loaders.truebones.data.dataset import _limb_root_mask

    # Hips -> Spine -> {L Thigh -> L Calf -> L Foot -> L Toe, R Thigh -> R Foot}
    cond = {
        'parents': [-1, 0, 1, 2, 3, 4, 1, 6],
        'limb_chain_id': [-1, -1, -1, -1, 0, 0, -1, 1],
        'joint_side_labels': ['center', 'center', 'left', 'left', 'left', 'left', 'right', 'right'],
    }
    assert np.flatnonzero(_limb_root_mask(cond)).tolist() == [2, 6]

    # Without side labels the limb-end chain top is the root.
    del cond['joint_side_labels']
    assert np.flatnonzero(_limb_root_mask(cond)).tolist() == [4, 7]
