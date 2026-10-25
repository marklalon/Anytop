import functools
import math
import os
import pickle
import queue
import tempfile
import threading
import weakref
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from data_loaders.joint_buckets import (
    JointBucketBatchSampler,
    bucket_ids_for_joint_counts,
    resolve_joint_buckets,
)
from data_loaders.tensors import truebones_batch_collate
from data_loaders.truebones.data.dataset import Truebones
from data_loaders.truebones.truebones_utils.bone_length_aug import (
    DEFAULT_BONE_LENGTH_AUG, DEFAULT_BONE_LENGTH_AUG_MIN)
from data_loaders.truebones.truebones_utils.rest_repose_aug import DEFAULT_REST_REPOSE_MAX_DEG
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

class _TransformedCollate:
    """collate_fn followed by batch_transform, as one picklable callable so a
    worker process runs both."""

    def __init__(self, collate, batch_transform):
        self.collate = collate
        self.batch_transform = batch_transform

    def __call__(self, samples):
        return self.batch_transform(self.collate(samples))


def _load_spilled_dataset(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


class _SpilledDataset(Dataset):
    """Hands a dataset to spawned workers through a temp file.

    Spawn writes each worker's arguments into a pipe that the child reads only
    after re-importing the main module, so a large pickled dataset blocks the
    parent once per worker and the workers start one after another. Pickled as
    the file path instead, the write fits the pipe and they start together.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self._path = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]

    def __reduce__(self):
        if self._path is None:
            handle, self._path = tempfile.mkstemp(prefix='anytop_dataset_', suffix='.pkl')
            with os.fdopen(handle, 'wb') as out:
                pickle.dump(self.dataset, out, protocol=pickle.HIGHEST_PROTOCOL)
            weakref.finalize(self, _remove_quietly, self._path)
        return _load_spilled_dataset, (self._path,)


def _remove_quietly(path):
    try:
        os.remove(path)
    except OSError:
        pass


class EagerWorkerLoader:
    """A worker-process DataLoader whose workers are spawned at construction.

    Under spawn each worker re-imports the main module and unpickles the
    dataset before the next one is started, so the first ``iter()`` of a
    DataLoader blocks for num_workers times that. Starting it on a thread here
    overlaps the spawn with whatever the caller builds next (the model, its
    compile); the thread mostly waits on the worker pipes, off the GIL. The
    first ``__iter__`` hands over that iterator, later ones are the
    DataLoader's own (persistent workers, so they only reset).
    """

    def __init__(self, loader, dataset):
        self.loader = loader
        self.dataset = dataset
        self._first_iter = None
        self._error = None
        self._spawn = threading.Thread(target=self._start_workers, daemon=True)
        self._spawn.start()

    def _start_workers(self):
        try:
            self._first_iter = iter(self.loader)
        except Exception as exc:
            self._error = exc

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        if self._spawn is not None:
            self._spawn.join()
            self._spawn = None
            if self._error is not None:
                raise self._error
            first, self._first_iter = self._first_iter, None
            return first
        return iter(self.loader)


def _init_data_worker(worker_id):
    # Sample prep is small numpy work; intra-op threads in every worker would
    # only oversubscribe the cores the training process launches kernels from.
    torch.set_num_threads(1)


def get_dataset_class(name):
    return Truebones

def get_dataset(
    num_frames,
    split='train',
    head_word_weights=None,
    rare_head_word_floor=0,
    rare_head_word_max_boost=4.0,
    objects_subset="all",
    sample_limit=0,
    action_group='',
    action_label_cond=False,
    action_conditioning=None,
    motion_cache_size=0,
    min_length=20,
    loop_cond_prob=1.0,
    motion_speed_aug=1.0,
    motion_speed_aug_prob=1.0,
    loop_tile_single_prob=0.5,
    topology_aug_prob=0.0,
    bone_length_aug_prob=0.0,
    bone_length_aug_max=DEFAULT_BONE_LENGTH_AUG,
    bone_length_aug_min=DEFAULT_BONE_LENGTH_AUG_MIN,
    rest_repose_prob=0.0,
    rest_repose_max_deg=DEFAULT_REST_REPOSE_MAX_DEG,
    cond_path=None,
    species_table=None,
    ref_encoder=False,
    joint_name_drop_prob=0.0,
):
    dataset = Truebones(
        split=split,
        num_frames=num_frames,
        head_word_weights=head_word_weights,
        rare_head_word_floor=rare_head_word_floor,
        rare_head_word_max_boost=rare_head_word_max_boost,
        objects_subset=objects_subset,
        sample_limit=sample_limit,
        action_group=action_group,
        action_label_cond=action_label_cond,
        action_conditioning=action_conditioning,
        motion_cache_size=motion_cache_size,
        min_length=min_length,
        loop_cond_prob=loop_cond_prob,
        motion_speed_aug=motion_speed_aug,
        motion_speed_aug_prob=motion_speed_aug_prob,
        loop_tile_single_prob=loop_tile_single_prob,
        topology_aug_prob=topology_aug_prob,
        bone_length_aug_prob=bone_length_aug_prob,
        bone_length_aug_max=bone_length_aug_max,
        bone_length_aug_min=bone_length_aug_min,
        rest_repose_prob=rest_repose_prob,
        rest_repose_max_deg=rest_repose_max_deg,
        cond_path=cond_path,
        species_table=species_table,
        ref_encoder=ref_encoder,
        joint_name_drop_prob=joint_name_drop_prob,
    )
    return dataset

def get_dataset_loader(
    batch_size,
    num_frames,
    split='train',
    head_word_weights=None,
    rare_head_word_floor=0,
    rare_head_word_max_boost=4.0,
    objects_subset="all",
    num_workers=0,
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
    loop_cond_prob=1.0,
    motion_speed_aug=1.0,
    motion_speed_aug_prob=1.0,
    loop_tile_single_prob=0.5,
    topology_aug_prob=0.0,
    bone_length_aug_prob=0.0,
    bone_length_aug_max=DEFAULT_BONE_LENGTH_AUG,
    bone_length_aug_min=DEFAULT_BONE_LENGTH_AUG_MIN,
    rest_repose_prob=0.0,
    rest_repose_max_deg=DEFAULT_REST_REPOSE_MAX_DEG,
    cond_path=None,
    joint_buckets=JOINT_BUCKETS,
    species_table=None,
    ref_encoder=False,
    joint_name_drop_prob=0.0,
):
    """``num_workers``: 0 prepares batches on one background thread of this
    process; N > 0 prepares them in N persistent worker processes, which keeps
    the sample prep off the training process's GIL. ``main_process_prefetch_batches``
    is the number of batches kept ready in either mode.

    ``joint_buckets``: joint-count ceilings (param_utils.JOINT_BUCKETS by
    default). Batches are grouped by rig size and padded to their bucket's
    ceiling instead of MAX_JOINTS -- see data_loaders/joint_buckets. ``None`` or
    ``()`` pads every batch to MAX_JOINTS."""
    dataset = get_dataset(
        num_frames=num_frames,
        split=split,
        head_word_weights=head_word_weights,
        rare_head_word_floor=rare_head_word_floor,
        rare_head_word_max_boost=rare_head_word_max_boost,
        objects_subset=objects_subset,
        sample_limit=sample_limit,
        action_group=action_group,
        action_label_cond=action_label_cond,
        action_conditioning=action_conditioning,
        motion_cache_size=motion_cache_size,
        min_length=min_length,
        loop_cond_prob=loop_cond_prob,
        motion_speed_aug=motion_speed_aug,
        motion_speed_aug_prob=motion_speed_aug_prob,
        loop_tile_single_prob=loop_tile_single_prob,
        topology_aug_prob=topology_aug_prob,
        bone_length_aug_prob=bone_length_aug_prob,
        bone_length_aug_max=bone_length_aug_max,
        bone_length_aug_min=bone_length_aug_min,
        rest_repose_prob=rest_repose_prob,
        rest_repose_max_deg=rest_repose_max_deg,
        cond_path=cond_path,
        species_table=species_table,
        ref_encoder=ref_encoder,
        joint_name_drop_prob=joint_name_drop_prob,
    )
    collate = truebones_batch_collate
    sampler = None
    # A weighted sampler is needed once any head word can be weighted
    # (--rare_head_word_floor / --head_word_weights); the dataset owns that
    # decision so both callers agree on it.
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
    prefetch_batches = int(main_process_prefetch_batches) or 4
    num_workers = max(0, int(num_workers or 0))
    if num_workers > 0:
        if batch_transform is not None:
            loader_kwargs['collate_fn'] = _TransformedCollate(loader_kwargs['collate_fn'], batch_transform)
        # Persistent: an epoch boundary must not respawn the workers (spawn
        # re-imports torch and unpickles the dataset). Each worker seeds its own
        # python / numpy / torch RNG from the loader's base seed.
        loader_kwargs.update(
            dataset=_SpilledDataset(dataset),
            num_workers=num_workers,
            persistent_workers=True,
            prefetch_factor=max(1, math.ceil(prefetch_batches / num_workers)),
            worker_init_fn=_init_data_worker,
        )
        return EagerWorkerLoader(DataLoader(**loader_kwargs), dataset)
    loader = DataLoader(**loader_kwargs)
    return BackgroundPrefetchLoader(loader, max_prefetch_batches=prefetch_batches, batch_transform=batch_transform)
