"""Dedicated held-out validation data for autoencoder training."""
from tqdm import tqdm

from data_loaders.get_data import get_dataset_loader
from data_loaders.truebones.data.dataset import EmptySplitError


def build_validation_loader(args, batch_transform=None):
    try:
        loader = get_dataset_loader(
            cond_path=args.cond_path, batch_size=args.batch_size,
            num_frames=args.num_frames, split='val',
            # Validation never inherits the training clip filter.
            sample_filter=None,
            sample_filter_seed=args.seed, shuffle=False, drop_last=False,
            action_label_cond=getattr(args, 'action_label_cond', False),
            action_conditioning=getattr(args, 'action_conditioning', None),
            species_table=getattr(args, 'species_descriptor_table', None),
            motion_cache_size=args.motion_cache_size,
            motion_speed_aug=1.0, loop_tile_single_prob=0.0, loop_cond_prob=1.0,
            topology_aug_prob=0.0, bone_length_aug_prob=0.0,
            rest_repose_prob=0.0, rest_variant_prob=0.0,
            joint_name_drop_prob=0.0, joint_buckets=None,
            num_workers=0, main_process_prefetch_batches=0,
            batch_transform=batch_transform,
        )
    except EmptySplitError as exc:
        tqdm.write(f'[val] validation disabled: {exc}')
        return None
    if len(loader.dataset) == 0:
        tqdm.write('[val] validation disabled: val split is empty')
        return None
    return loader
