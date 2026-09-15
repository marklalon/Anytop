"""Joint-count bucketing for the training loader.

Every batch is padded to one joint count. Padding all of them to the global
``MAX_JOINTS`` (100) wastes most of the decoder: the corpus median is ~38
joints and 92% of clips fit in 64, yet spatial attention runs over 100x100
slots per frame and every per-token layer over 100 slots. Bucketing draws the
epoch's indices from the ordinary sampler, groups them by joint count and pads
each batch only to its bucket's ceiling, so the loader emits a small fixed set
of shapes -- one static ``torch.compile`` graph per bucket -- instead of one
over-padded shape.

The sampling distribution is untouched: every index the base sampler draws is
still used exactly once (dropping the same trailing partial batch per bucket
that ``drop_last`` drops today); only which indices share a batch changes.
"""

from __future__ import annotations

from typing import Iterator, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler


def resolve_joint_buckets(
    buckets: Optional[Sequence[int]], max_joints: int
) -> Optional[tuple[int, ...]]:
    """``(64,)`` / ``(48, 72, 100)`` -> ascending bucket ceilings, always ending
    with ``max_joints`` so any rig has a bucket. ``None``/empty disables
    bucketing (every batch padded to ``max_joints``)."""
    if buckets is None:
        return None
    ceilings = sorted({int(ceiling) for ceiling in buckets})
    if not ceilings:
        return None
    if ceilings[0] <= 0:
        raise ValueError(f"JOINT_BUCKETS ceilings must be positive, got {ceilings}")
    if ceilings[-1] > max_joints:
        raise ValueError(
            f"JOINT_BUCKETS ceiling {ceilings[-1]} exceeds MAX_JOINTS={max_joints}; "
            "no clip can need it."
        )
    if ceilings[-1] != max_joints:
        ceilings.append(max_joints)
    return tuple(ceilings)


def bucket_ceiling(n_joints: int, buckets: Sequence[int]) -> int:
    """Smallest bucket ceiling that holds ``n_joints`` joints."""
    for ceiling in buckets:
        if n_joints <= ceiling:
            return int(ceiling)
    raise ValueError(f"{n_joints} joints exceed the largest joint bucket {buckets[-1]}")


def bucket_ids_for_joint_counts(joint_counts: np.ndarray, buckets: Sequence[int]) -> np.ndarray:
    """Bucket index (into ``buckets``) of every entry of ``joint_counts``."""
    joint_counts = np.asarray(joint_counts, dtype=np.int64)
    ceilings = np.asarray(buckets, dtype=np.int64)
    if joint_counts.size and int(joint_counts.max()) > int(ceilings[-1]):
        raise ValueError(
            f"a rig has {int(joint_counts.max())} joints but the largest joint bucket is {int(ceilings[-1])}"
        )
    # searchsorted(side='left') = first ceiling >= n, matching bucket_ceiling.
    return np.searchsorted(ceilings, joint_counts, side='left')


class JointBucketBatchSampler(Sampler[list[int]]):
    """Regroup an index sampler's stream into joint-count-homogeneous batches.

    Indices are consumed in the order the wrapped sampler yields them (so its
    shuffle / species weighting is exactly preserved) and parked in one queue
    per bucket; a queue that fills to ``batch_size`` is emitted immediately.
    Whatever is left in the queues when the sampler is exhausted is the epoch's
    partial batches: dropped under ``drop_last`` like a plain loader's single
    trailing partial batch, otherwise emitted as short batches.
    """

    def __init__(
        self,
        index_sampler: Sampler[int],
        batch_size: int,
        bucket_ids: Sequence[int],
        num_buckets: int,
        drop_last: bool = True,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.index_sampler = index_sampler
        self.batch_size = int(batch_size)
        self.bucket_ids = np.asarray(bucket_ids, dtype=np.int64)
        self.num_buckets = int(num_buckets)
        self.drop_last = bool(drop_last)
        if self.bucket_ids.ndim != 1:
            raise ValueError("bucket_ids must be a flat sequence indexed by sampler index")
        if self.bucket_ids.size and (self.bucket_ids.min() < 0 or self.bucket_ids.max() >= self.num_buckets):
            raise ValueError("bucket_ids must lie in [0, num_buckets)")

    def __iter__(self) -> Iterator[list[int]]:
        queues: list[list[int]] = [[] for _ in range(self.num_buckets)]
        for index in self.index_sampler:
            index = int(index)
            queue = queues[self.bucket_ids[index]]
            queue.append(index)
            if len(queue) == self.batch_size:
                yield list(queue)
                queue.clear()
        if not self.drop_last:
            for queue in queues:
                if queue:
                    yield list(queue)

    def expected_bucket_counts(self) -> np.ndarray:
        """How many indices land in each bucket over one pass of the sampler.

        Exact for a permutation/sequential sampler; the expectation under the
        sampler's weights for a with-replacement weighted sampler.
        """
        weights = getattr(self.index_sampler, 'weights', None)
        if weights is not None:
            probabilities = torch.as_tensor(weights, dtype=torch.float64).reshape(-1)
            probabilities = probabilities / probabilities.sum().clamp(min=1e-12)
            if probabilities.numel() != self.bucket_ids.size:
                raise ValueError(
                    f"bucket_ids covers {self.bucket_ids.size} indices but the weighted sampler "
                    f"draws over {probabilities.numel()}"
                )
            per_bucket = np.zeros(self.num_buckets, dtype=np.float64)
            np.add.at(per_bucket, self.bucket_ids, probabilities.numpy())
            return per_bucket * float(len(self.index_sampler))
        covered = min(len(self.index_sampler), self.bucket_ids.size)
        return np.bincount(self.bucket_ids[:covered], minlength=self.num_buckets).astype(np.float64)

    def __len__(self) -> int:
        counts = self.expected_bucket_counts()
        if self.drop_last:
            return int(np.sum(np.floor(counts) // self.batch_size))
        return int(np.sum(np.ceil(counts / self.batch_size)))
