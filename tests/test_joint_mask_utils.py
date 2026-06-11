from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


from model.joint_mask_utils import sample_subtree_joint_mask, sample_subtree_joint_mask_batch  # noqa: E402


def _candidate_root_mask(*root_indices: int) -> np.ndarray:
    mask = np.zeros(9, dtype=bool)
    mask[list(root_indices)] = True
    return mask


def test_sample_subtree_joint_mask_respects_budget_and_root_visibility() -> None:
    parents = [-1, 0, 1, 2, 3, 0, 5, 0, 7]

    mask = sample_subtree_joint_mask(
        parents=parents,
        candidate_root_mask=_candidate_root_mask(1, 5, 7),
        joint_mask_budget=0.5,
        rng=np.random.default_rng(0),
    )

    assert mask is not None
    assert mask.shape == (len(parents),)
    assert not bool(mask[0])
    # Single-subtree sampler: the size-4 limb {1,2,3,4} or one size-2 limb
    # ({5,6} / {7,8}) — all <= budget of 4.
    assert int(mask.sum()) in {2, 4}


def test_sample_subtree_joint_mask_biases_toward_larger_subtrees() -> None:
    parents = [-1, 0, 1, 2, 3, 0, 5, 0, 7]
    large_subtree_mask = np.zeros(9, dtype=bool)
    large_subtree_mask[[1, 2, 3, 4]] = True
    small_a_mask = np.zeros(9, dtype=bool)
    small_a_mask[[5, 6]] = True
    small_b_mask = np.zeros(9, dtype=bool)
    small_b_mask[[7, 8]] = True

    large_count = 0
    small_a_count = 0
    small_b_count = 0
    for seed in range(2000):
        mask = sample_subtree_joint_mask(
            parents=parents,
            candidate_root_mask=_candidate_root_mask(1, 5, 7),
            joint_mask_budget=0.5,
            rng=np.random.default_rng(seed),
        )

        assert mask is not None
        if np.array_equal(mask, large_subtree_mask):
            large_count += 1
        elif np.array_equal(mask, small_a_mask):
            small_a_count += 1
        elif np.array_equal(mask, small_b_mask):
            small_b_count += 1
        else:
            raise AssertionError(f"Unexpected mask at seed {seed}")

    # Weight is size**2: 16 (size-4 limb) vs 4+4 (two size-2 limbs).
    # The single large subtree should be chosen more often than both small
    # subtrees combined.
    assert large_count > small_a_count + small_b_count
    # The two small subtrees have equal weight and should appear equally often.
    total_small = small_a_count + small_b_count
    assert abs(small_a_count - small_b_count) < max(1, total_small * 0.2)


def test_sample_subtree_joint_mask_supports_restored_global_numpy_state() -> None:
    parents = [-1, 0, 1, 2, 3, 0, 5, 0, 7]
    candidate_root_mask = _candidate_root_mask(1, 5, 7)

    np.random.seed(123)
    saved_state = np.random.get_state()
    first_mask = sample_subtree_joint_mask(
        parents=parents,
        candidate_root_mask=candidate_root_mask,
        joint_mask_budget=0.5,
        rng=np.random,
    )

    np.random.seed(999)
    np.random.set_state(saved_state)
    second_mask = sample_subtree_joint_mask(
        parents=parents,
        candidate_root_mask=candidate_root_mask,
        joint_mask_budget=0.5,
        rng=np.random,
    )

    assert first_mask is not None
    assert second_mask is not None
    assert np.array_equal(first_mask, second_mask)


def test_sample_subtree_joint_mask_batch_matches_sequential_sampling() -> None:
    parents_batch = np.asarray(
        [
            [-1, 0, 1, 2, 3, 0, 5, 0, 7],
            [-1, 0, 1, 2, 0, 0, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    candidate_root_mask_batch = np.stack(
        [
            _candidate_root_mask(1, 5, 7),
            _candidate_root_mask(1, 3),
        ],
        axis=0,
    )
    n_joints = np.asarray([9, 5], dtype=np.int64)

    np.random.seed(123)
    batch_mask = sample_subtree_joint_mask_batch(
        parents_batch=parents_batch,
        candidate_root_mask_batch=candidate_root_mask_batch,
        n_joints=n_joints,
        max_joints=9,
        joint_mask_prob=1.0,
        joint_mask_budget=0.5,
        rng=np.random,
    )

    np.random.seed(123)
    expected = np.zeros((2, 9), dtype=bool)
    for batch_index in range(2):
        valid_joint_count = int(n_joints[batch_index])
        per_sample_mask = sample_subtree_joint_mask(
            parents=parents_batch[batch_index, :valid_joint_count].tolist(),
            candidate_root_mask=candidate_root_mask_batch[batch_index, :valid_joint_count],
            joint_mask_budget=0.5,
            rng=np.random,
        )
        if per_sample_mask is not None:
            expected[batch_index, :valid_joint_count] = per_sample_mask

    assert batch_mask is not None
    assert np.array_equal(batch_mask, expected)
    assert not np.any(batch_mask[1, int(n_joints[1]):])


def test_sample_subtree_joint_mask_batch_respects_probability_gate() -> None:
    parents_batch = np.asarray([[-1, 0, 1, 2, 3, 0, 5, 0, 7]], dtype=np.int64)
    candidate_root_mask_batch = np.stack([_candidate_root_mask(1, 5, 7)], axis=0)
    n_joints = np.asarray([9], dtype=np.int64)

    mask = sample_subtree_joint_mask_batch(
        parents_batch=parents_batch,
        candidate_root_mask_batch=candidate_root_mask_batch,
        n_joints=n_joints,
        max_joints=9,
        joint_mask_prob=0.0,
        joint_mask_budget=0.5,
        rng=np.random.default_rng(0),
    )

    assert mask is None