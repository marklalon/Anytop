import functools
import os
import queue
import threading
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from data_loaders.joint_buckets import (
    JointBucketBatchSampler,
    bucket_ids_for_joint_counts,
    resolve_joint_buckets,
)
from data_loaders.tensors import truebones_batch_collate
from data_loaders.truebones.data.dataset import Truebones
from data_loaders.truebones.truebones_utils.param_utils import JOINT_BUCKETS


class _PrefetchSentinel:
    pass


class BackgroundPrefetchLoader:
    def __init__(self, loader, max_prefetch_batches=2, batch_transform=None):
        self.loader = loader
        self.dataset = loader.dataset
        self.max_prefetch_batches = max(1, int(max_prefetch_batches))
        self.batch_transform = batch_transform

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        data_queue = queue.Queue(maxsize=self.max_prefetch_batches)
        sentinel = _PrefetchSentinel()
        errors = []

        def _producer():
            try:
                for batch in self.loader:
                    if self.batch_transform is not None:
                        batch = self.batch_transform(batch)
                    data_queue.put(batch)
            except Exception as exc:
                errors.append(exc)
            finally:
                data_queue.put(sentinel)

        worker = threading.Thread(target=_producer, daemon=True)
        worker.start()

        while True:
            item = data_queue.get()
            if item is sentinel:
                worker.join()
                if errors:
                    raise errors[0]
                break
            yield item

def get_dataset_class(name):
    return Truebones

def get_dataset(
    num_frames,
    split='train',
    balanced=False,
    objects_subset="all",
    sample_limit=0,
    action_group='',
    action_label_cond=False,
    action_conditioning=None,
    motion_cache_size=0,
    min_length=20,
    motion_speed_aug=1.0,
    motion_speed_aug_prob=1.0,
    cond_path=None,
):
    dataset = Truebones(
        split=split,
        num_frames=num_frames,
        balanced=balanced,
        objects_subset=objects_subset,
        sample_limit=sample_limit,
        action_group=action_group,
        action_label_cond=action_label_cond,
        action_conditioning=action_conditioning,
        motion_cache_size=motion_cache_size,
        min_length=min_length,
        motion_speed_aug=motion_speed_aug,
        motion_speed_aug_prob=motion_speed_aug_prob,
        cond_path=cond_path,
    )
    return dataset

def get_dataset_loader(
    batch_size,
    num_frames,
    split='train',
    balanced=True,
    objects_subset="all",
    num_workers=None,
    prefetch_factor=2,
    sample_limit=0,
    shuffle=True,
    drop_last=True,
    action_group='',
    action_label_cond=False,
    action_conditioning=None,
    motion_cache_size=0,
    min_length=20,
    main_process_prefetch_batches=0,
    batch_transform=None,
    motion_speed_aug=1.0,
    motion_speed_aug_prob=1.0,
    cond_path=None,
    joint_buckets=JOINT_BUCKETS,
):
    """``joint_buckets``: joint-count ceilings (param_utils.JOINT_BUCKETS by
    default). Batches are grouped by rig size and padded to their bucket's
    ceiling instead of MAX_JOINTS -- see data_loaders/joint_buckets. ``None`` or
    ``()`` pads every batch to MAX_JOINTS."""
    # Always use main thread (num_workers=0) - multi-worker paths removed
    dataset = get_dataset(
        num_frames=num_frames,
        split=split,
        balanced=balanced,
        objects_subset=objects_subset,
        sample_limit=sample_limit,
        action_group=action_group,
        action_label_cond=action_label_cond,
        action_conditioning=action_conditioning,
        motion_cache_size=motion_cache_size,
        min_length=min_length,
        motion_speed_aug=motion_speed_aug,
        motion_speed_aug_prob=motion_speed_aug_prob,
        cond_path=cond_path,
    )
    collate = truebones_batch_collate
    sampler = None
    # A weighted sampler is needed for species balancing (--balanced).
    if dataset.motion_dataset.use_weighted_sampler:
        from data_loaders.truebones.data.dataset import TruebonesSampler
        sampler = TruebonesSampler(dataset)
    buckets = resolve_joint_buckets(joint_buckets, dataset.motion_dataset.opt.max_joints)
    if buckets is not None:
        # The bucket sampler wraps the SAME index source a plain loader would
        # use (RandomSampler draws from the global torch RNG exactly like
        # shuffle=True does), so seeding and species weighting are unchanged;
        # it only regroups the drawn indices into bucket-homogeneous batches.
        if sampler is None:
            sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        bucket_ids = bucket_ids_for_joint_counts(
            dataset.motion_dataset.sampler_index_joint_counts(), buckets
        )
        batch_sampler = JointBucketBatchSampler(
            sampler, batch_size, bucket_ids, num_buckets=len(buckets), drop_last=drop_last
        )
        collate = functools.partial(truebones_batch_collate, joint_buckets=buckets)
        loader_kwargs = {
            'dataset': dataset,
            'batch_sampler': batch_sampler,
            'num_workers': 0,
            'collate_fn': collate,
        }
    else:
        loader_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'sampler': sampler,
            'shuffle': shuffle if sampler is None else False,
            'num_workers': 0,
            'drop_last': drop_last,
            'collate_fn': collate,
        }
    if torch.cuda.is_available():
        loader_kwargs['pin_memory'] = True
    loader = DataLoader(**loader_kwargs)
    # Use background prefetch on main thread for data loading overlap
    main_process_prefetch_batches = int(main_process_prefetch_batches) or 4
    return BackgroundPrefetchLoader(loader, max_prefetch_batches=main_process_prefetch_batches, batch_transform=batch_transform)