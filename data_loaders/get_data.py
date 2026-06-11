import os
import queue
import threading
import torch
from torch.utils.data import DataLoader
from data_loaders.tensors import truebones_batch_collate
from data_loaders.truebones.data.dataset import Truebones


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
    temporal_window=31,
    balanced=False,
    objects_subset="all",
    sample_limit=0,
    action_tags='',
    motion_cache_size=0,
    min_length=20,
    loop_cond_prob=1.0,
):
    dataset = Truebones(
        split=split,
        num_frames=num_frames,
        temporal_window=temporal_window,
        balanced=balanced,
        objects_subset=objects_subset,
        sample_limit=sample_limit,
        action_tags=action_tags,
        motion_cache_size=motion_cache_size,
        min_length=min_length,
        loop_cond_prob=loop_cond_prob,
    )
    return dataset

def get_dataset_loader(
    batch_size,
    num_frames,
    split='train',
    temporal_window=31,
    balanced=True,
    objects_subset="all",
    num_workers=None,
    prefetch_factor=2,
    sample_limit=0,
    shuffle=True,
    drop_last=True,
    action_tags='',
    motion_cache_size=0,
    min_length=20,
    main_process_prefetch_batches=0,
    batch_transform=None,
    loop_cond_prob=1.0,
):
    # Always use main thread (num_workers=0) - multi-worker paths removed
    dataset = get_dataset(
        num_frames=num_frames,
        split=split,
        temporal_window=temporal_window,
        balanced=balanced,
        objects_subset=objects_subset,
        sample_limit=sample_limit,
        action_tags=action_tags,
        motion_cache_size=motion_cache_size,
        min_length=min_length,
        loop_cond_prob=loop_cond_prob,
    )
    collate = truebones_batch_collate
    sampler = None
    # A weighted sampler is needed for species balancing (--balanced) and/or for
    # the in-code per-action-tag sampling weights (ACTION_TAG_SAMPLE_WEIGHTS).
    if dataset.motion_dataset.use_weighted_sampler:
        from data_loaders.truebones.data.dataset import TruebonesSampler
        sampler = TruebonesSampler(dataset)
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