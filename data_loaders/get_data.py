import os
import queue
import threading
import torch
from torch.utils.data import DataLoader
from data_loaders.tensors import truebones_batch_collate
from data_loaders.truebones.data.dataset import Truebones


class _QueueSentinel:
    pass


class ThreadPrefetchLoader:
    def __init__(self, loader, max_prefetch_batches=2):
        self.loader = loader
        self.dataset = loader.dataset
        self.max_prefetch_batches = max(1, int(max_prefetch_batches))

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        data_queue = queue.Queue(maxsize=self.max_prefetch_batches)
        sentinel = _QueueSentinel()
        error_holder = []

        def _producer():
            try:
                for batch in self.loader:
                    data_queue.put(batch)
            except Exception as exc:
                error_holder.append(exc)
            finally:
                data_queue.put(sentinel)

        worker = threading.Thread(target=_producer, daemon=True)
        worker.start()

        while True:
            item = data_queue.get()
            if item is sentinel:
                worker.join()
                if error_holder:
                    raise error_holder[0]
                break
            yield item

def get_dataset_class(name):
    return Truebones

def get_dataset(num_frames, split='train', temporal_window=31, t5_name='t5-base', balanced=False, objects_subset="all", sample_limit=0, use_reference_conditioning=True):
    dataset = Truebones(
        split=split,
        num_frames=num_frames,
        temporal_window=temporal_window,
        t5_name=t5_name,
        balanced=balanced,
        objects_subset=objects_subset,
        sample_limit=sample_limit,
        use_reference_conditioning=use_reference_conditioning,
    )
    return dataset


def get_dataset_loader(batch_size, num_frames, split='train', temporal_window=31, t5_name='t5-base', balanced=True, objects_subset="all", num_workers=None, prefetch_factor=2, sample_limit=0, shuffle=True, drop_last=True, use_reference_conditioning=True):
    if num_workers is None or int(num_workers) < 0:
        cpu_count = os.cpu_count() or 1
        num_workers = min(4, cpu_count)
    else:
        num_workers = int(num_workers)
    use_thread_prefetch = os.name == 'nt' and num_workers > 0
    dataset = get_dataset(
        num_frames=num_frames,
        split=split,
        temporal_window=temporal_window,
        t5_name=t5_name,
        balanced=balanced,
        objects_subset=objects_subset,
        sample_limit=sample_limit,
        use_reference_conditioning=use_reference_conditioning,
    )
    collate = truebones_batch_collate
    sampler = None
    if balanced: #create batch sampler
        from data_loaders.truebones.data.dataset import TruebonesSampler
        sampler = TruebonesSampler(dataset)
    loader_kwargs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'sampler': sampler,
        'shuffle': shuffle if sampler is None else False,
        'num_workers': 0 if use_thread_prefetch else num_workers,
        'drop_last': drop_last,
        'collate_fn': collate,
    }
    if torch.cuda.is_available():
        loader_kwargs['pin_memory'] = True
    if not use_thread_prefetch and num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = max(1, int(prefetch_factor))
    loader = DataLoader(**loader_kwargs)
    if use_thread_prefetch:
        prefetch_batches = max(1, num_workers * max(1, int(prefetch_factor)))
        return ThreadPrefetchLoader(loader, max_prefetch_batches=prefetch_batches)
    return loader