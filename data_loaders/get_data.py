import os
import torch
from torch.utils.data import DataLoader
from data_loaders.tensors import truebones_batch_collate
from data_loaders.truebones.data.dataset import Truebones

def get_dataset_class(name):
    return Truebones

def get_dataset(num_frames, split='train', temporal_window=31, t5_name='t5-base', balanced=False, objects_subset="all", curriculum_stage=1, enable_topology_augmentation=False, sample_limit=0, offline_reference_samples=False, offline_reference_seed=10):
    dataset = Truebones(
        split=split,
        num_frames=num_frames,
        temporal_window=temporal_window,
        t5_name=t5_name,
        balanced=balanced,
        objects_subset=objects_subset,
        curriculum_stage=curriculum_stage,
        enable_topology_augmentation=enable_topology_augmentation,
        sample_limit=sample_limit,
        offline_reference_samples=offline_reference_samples,
        offline_reference_seed=offline_reference_seed,
    )
    return dataset
def get_dataset_loader(batch_size, num_frames, split='train', temporal_window=31, t5_name='t5-base', balanced=True, objects_subset="all", num_workers=None, curriculum_stage=1, enable_topology_augmentation=False, prefetch_factor=2, sample_limit=0, offline_reference_samples=False, offline_reference_seed=10):
    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        num_workers = min(4, cpu_count)
    if offline_reference_samples:
        num_workers = 0
    dataset = get_dataset(
        num_frames=num_frames,
        split=split,
        temporal_window=temporal_window,
        t5_name=t5_name,
        balanced=balanced,
        objects_subset=objects_subset,
        curriculum_stage=curriculum_stage,
        enable_topology_augmentation=enable_topology_augmentation,
        sample_limit=sample_limit,
        offline_reference_samples=offline_reference_samples,
        offline_reference_seed=offline_reference_seed,
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
        'shuffle': True if sampler is None else False,
        'num_workers': num_workers,
        'drop_last': True,
        'collate_fn': collate,
    }
    if torch.cuda.is_available():
        loader_kwargs['pin_memory'] = True
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = max(1, int(prefetch_factor))
    loader = DataLoader(**loader_kwargs)
    return loader