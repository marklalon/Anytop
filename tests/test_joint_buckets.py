"""Joint-count bucketing (param_utils.JOINT_BUCKETS): the loader pads each
batch to its bucket's ceiling instead of MAX_JOINTS, so the run sees one static
shape per bucket. Three things have to hold for that to be a pure speed change:

* the batch sampler regroups the base sampler's stream without changing which
  indices are drawn (only which share a batch);
* the collate pads to the bucket ceiling that fits the batch's widest rig;
* the model's output on the valid joints does not depend on how many padding
  slots follow them, so a 64-padded and a 100-padded batch train the same net.
"""
from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import RandomSampler, SequentialSampler, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_loaders.joint_buckets import (  # noqa: E402
    JointBucketBatchSampler,
    bucket_ceiling,
    bucket_ids_for_joint_counts,
    resolve_joint_buckets,
)
from data_loaders.tensors import truebones_batch_collate  # noqa: E402
from data_loaders.truebones.truebones_utils.joint_struct_features import (  # noqa: E402
    JOINT_STRUCT_DIM,
)
from data_loaders.truebones.truebones_utils.param_utils import (  # noqa: E402
    JOINT_BUCKETS,
    MAX_JOINTS,
)
from model.anytop import AnyTop  # noqa: E402


class ResolveBucketsTest(unittest.TestCase):
    def test_the_repo_constant_is_valid(self):
        resolved = resolve_joint_buckets(JOINT_BUCKETS, MAX_JOINTS)
        self.assertEqual(resolved, tuple(sorted(JOINT_BUCKETS)))
        self.assertEqual(resolved[-1], MAX_JOINTS)

    def test_empty_disables(self):
        self.assertIsNone(resolve_joint_buckets(None, 100))
        self.assertIsNone(resolve_joint_buckets((), 100))

    def test_max_joints_is_always_the_last_bucket(self):
        self.assertEqual(resolve_joint_buckets((64,), 100), (64, 100))
        self.assertEqual(resolve_joint_buckets((72, 48), 100), (48, 72, 100))
        self.assertEqual(resolve_joint_buckets((100,), 100), (100,))

    def test_rejects_ceilings_past_max_joints_or_non_positive(self):
        with self.assertRaises(ValueError):
            resolve_joint_buckets((120,), 100)
        with self.assertRaises(ValueError):
            resolve_joint_buckets((0, 64), 100)

    def test_ceiling_and_ids_agree(self):
        buckets = (48, 72, 100)
        counts = np.array([1, 48, 49, 72, 73, 100])
        ids = bucket_ids_for_joint_counts(counts, buckets)
        self.assertEqual(ids.tolist(), [0, 0, 1, 1, 2, 2])
        for n, i in zip(counts, ids):
            self.assertEqual(bucket_ceiling(int(n), buckets), buckets[i])
        with self.assertRaises(ValueError):
            bucket_ceiling(101, buckets)
        with self.assertRaises(ValueError):
            bucket_ids_for_joint_counts(np.array([101]), buckets)


class _Indexable:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


class BucketBatchSamplerTest(unittest.TestCase):
    def _joint_counts(self, n=101, seed=0):
        rng = np.random.default_rng(seed)
        return rng.choice([9, 30, 61, 100], size=n, p=[0.3, 0.4, 0.2, 0.1])

    def test_batches_are_homogeneous_and_use_every_drawn_index_once(self):
        buckets = (32, 64, 100)
        counts = self._joint_counts()
        ids = bucket_ids_for_joint_counts(counts, buckets)
        torch.manual_seed(0)
        sampler = JointBucketBatchSampler(
            RandomSampler(_Indexable(len(counts))), 8, ids, num_buckets=3, drop_last=True
        )
        batches = list(sampler)
        self.assertEqual(len(batches), len(sampler))
        seen = Counter()
        for batch in batches:
            self.assertEqual(len(batch), 8)
            self.assertEqual(len({int(ids[i]) for i in batch}), 1)
            seen.update(batch)
        self.assertTrue(all(v == 1 for v in seen.values()))
        # Exactly the per-bucket remainders are dropped -- the same rule a plain
        # drop_last loader applies to its single trailing batch.
        per_bucket = np.bincount(ids, minlength=3)
        self.assertEqual(sum(seen.values()), int(np.sum(per_bucket // 8) * 8))

    def test_without_drop_last_the_remainders_are_emitted(self):
        buckets = (32, 64, 100)
        counts = self._joint_counts()
        ids = bucket_ids_for_joint_counts(counts, buckets)
        sampler = JointBucketBatchSampler(
            SequentialSampler(_Indexable(len(counts))), 8, ids, num_buckets=3, drop_last=False
        )
        batches = list(sampler)
        self.assertEqual(len(batches), len(sampler))
        self.assertEqual(sorted(i for b in batches for i in b), list(range(len(counts))))
        short = [b for b in batches if len(b) < 8]
        per_bucket = np.bincount(ids, minlength=3)
        self.assertEqual(len(short), int(np.count_nonzero(per_bucket % 8)))

    def test_shuffle_order_matches_a_plain_random_sampler_stream(self):
        """The bucket sampler consumes the wrapped sampler verbatim: with the
        same seed the concatenated order of indices, bucket by bucket, is the
        plain RandomSampler permutation filtered to that bucket."""
        buckets = (32, 100)
        counts = self._joint_counts(n=64)
        ids = bucket_ids_for_joint_counts(counts, buckets)
        torch.manual_seed(123)
        plain = list(RandomSampler(_Indexable(64)))
        torch.manual_seed(123)
        bucketed = list(JointBucketBatchSampler(
            RandomSampler(_Indexable(64)), 4, ids, num_buckets=2, drop_last=False
        ))
        for bucket in range(2):
            expected = [i for i in plain if ids[i] == bucket]
            got = [i for b in bucketed if ids[b[0]] == bucket for i in b]
            self.assertEqual(got, expected)

    def test_weighted_sampler_length_is_the_expected_batch_count(self):
        buckets = (32, 100)
        counts = np.array([9] * 30 + [100] * 10)
        ids = bucket_ids_for_joint_counts(counts, buckets)
        weights = np.where(counts == 9, 1.0, 3.0)
        base = WeightedRandomSampler(weights, num_samples=40, replacement=True)
        sampler = JointBucketBatchSampler(base, 8, ids, num_buckets=2, drop_last=True)
        expected = sampler.expected_bucket_counts()
        self.assertAlmostEqual(float(expected.sum()), 40.0)
        # P(small) = 30 / (30 + 30) = 0.5 -> 20 draws each -> 2 + 2 batches.
        self.assertEqual(len(sampler), 4)


_BIPED_PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 10, 11, 12, 13, 14]
FEATS = 12
T5 = 32


def _item(n_joints, max_joints, frames=3, seed=0):
    rng = np.random.default_rng(seed)
    return [
        rng.standard_normal((frames, n_joints, FEATS)).astype(np.float32), frames,
        np.asarray(_BIPED_PARENTS[:n_joints], dtype=np.int64),
        rng.standard_normal((n_joints, FEATS)).astype(np.float32),
        np.zeros((n_joints, 3), dtype=np.float32),
        rng.integers(0, 4, size=(n_joints, n_joints)).astype(np.int64),
        rng.integers(0, 4, size=(n_joints, n_joints)).astype(np.int64),
        'truebones/zoo/Test',
        rng.standard_normal((n_joints, T5)).astype(np.float32),
        max_joints,
        {'translation_root_index': 0},
        'clip',
        {
            'joint_struct': rng.standard_normal((n_joints, JOINT_STRUCT_DIM)).astype(np.float32),
            'canonical_feature_mean': np.zeros(FEATS, dtype=np.float32),
            'canonical_feature_std': np.ones(FEATS, dtype=np.float32),
        },
    ]


class BucketCollateTest(unittest.TestCase):
    def test_pads_to_the_ceiling_that_fits_the_widest_rig(self):
        items = [_item(5, 100, seed=1), _item(7, 100, seed=2)]
        motion, cond = truebones_batch_collate(items, joint_buckets=(8, 16, 100))
        self.assertEqual(tuple(motion.shape), (2, 8, FEATS, 3))
        self.assertEqual(tuple(cond['y']['joints_padding_mask'].shape), (2, 1, 1, 9, 9))
        self.assertEqual(tuple(cond['y']['graph_dist'].shape), (2, 8, 8))
        self.assertEqual(tuple(cond['y']['joint_struct'].shape), (2, 8, JOINT_STRUCT_DIM))
        self.assertEqual(cond['y']['n_joints'].tolist(), [5, 7])
        motion, _ = truebones_batch_collate(items + [_item(9, 100, seed=3)], joint_buckets=(8, 16, 100))
        self.assertEqual(tuple(motion.shape), (3, 16, FEATS, 3))

    def test_no_buckets_keeps_the_global_max_joints(self):
        motion, _ = truebones_batch_collate([_item(5, 100, seed=1)])
        self.assertEqual(tuple(motion.shape), (1, 100, FEATS, 3))


class PaddingInvarianceTest(unittest.TestCase):
    """The same clips padded to 8 and to 16 slots must produce the same
    prediction on their real joints: bucketing changes nothing but speed."""

    def _model(self):
        torch.manual_seed(0)
        model = AnyTop(
            max_joints=100, feature_len=FEATS, latent_dim=16, ff_size=32, num_layers=2,
            num_heads=2, dropout=0.0, cross_limb=True, cross_limb_latents=3,
            cross_limb_dim=8, t5_out_dim=T5,
        )
        model.eval()
        return model

    def test_valid_joint_outputs_do_not_depend_on_padding_width(self):
        model = self._model()
        items = [_item(5, 100, seed=1), _item(7, 100, seed=2)]
        outputs = []
        for buckets in ((8, 100), (16, 100)):
            motion, cond = truebones_batch_collate(items, joint_buckets=buckets)
            with torch.no_grad():
                out = model(motion, torch.tensor([3, 7]), y=cond['y'])
            self.assertEqual(out.shape[1], buckets[0])
            outputs.append(out)
        narrow, wide = outputs
        for b, n in enumerate((5, 7)):
            self.assertTrue(
                torch.allclose(narrow[b, :n], wide[b, :n], atol=1e-5, rtol=1e-4),
                f"sample {b}: max abs diff {(narrow[b, :n] - wide[b, :n]).abs().max().item()}",
            )


if __name__ == '__main__':
    unittest.main()
