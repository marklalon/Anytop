import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import os
from collections import OrderedDict, defaultdict
from os.path import join as pjoin
from pathlib import Path
import random
import re
from typing import Optional
import warnings
from torch.utils.data._utils.collate import default_collate
from data_loaders.truebones.truebones_utils.get_opt import get_opt
from data_loaders.truebones.truebones_utils.param_utils import parse_action_tags
from data_loaders.truebones.truebones_utils.motion_labels import load_motion_metadata
from data_loaders.truebones.truebones_utils.motion_process import (
    refresh_joint_metadata_in_cond_dict,
)
from data_loaders.truebones.truebones_utils.canonical_features import (
    build_canonical_rest_feature,
    canonical_to_physical_hml,
    mark_canonical_cond_entry,
    physical_hml_to_canonical,
)
from data_loaders.truebones.truebones_utils.physics_joint_annotation import (
    JOINT_NAME_EMBEDDING_SCHEMA_VERSION,
    assert_species_tags_cover,
)



DEFAULT_SPLIT_RATIOS = {"train": 0.95, "val": 0.05, "test": 0.0}
DEFAULT_SPLIT_SEED = 3407
SUPPORTED_SPLITS = tuple(DEFAULT_SPLIT_RATIOS.keys())
ALL_SPLIT_NAME = "all"


def _normalize_motion_action_tags(raw_action_tags) -> set[str]:
    if raw_action_tags is None:
        return set()
    if isinstance(raw_action_tags, str):
        values = [raw_action_tags]
    else:
        values = raw_action_tags
    return {
        str(tag).strip().lower()
        for tag in values
        if str(tag).strip()
    }


def _copy_required_motion_metadata(motion_name: str, motion_metadata) -> dict[str, object]:
    if not isinstance(motion_metadata, dict):
        raise KeyError(f"Motion '{motion_name}' is missing explicit motion metadata.")
    if 'translation_root_index' not in motion_metadata:
        raise KeyError(f"Motion '{motion_name}' metadata is missing translation_root_index.")
    copied_motion_metadata = dict(motion_metadata)
    copied_motion_metadata.setdefault('motion_name', motion_name)
    return copied_motion_metadata


def _require_motion_metadata_entry(
    motion_name: str,
    motion_metadata_lookup: dict[str, dict[str, object]],
) -> dict[str, object]:
    return _copy_required_motion_metadata(motion_name, motion_metadata_lookup.get(motion_name))


def filter_motion_names_by_action_tags(
    motion_names,
    raw_action_tags,
    motion_metadata_lookup,
):
    requested_action_tags = set(parse_action_tags(raw_action_tags))
    if not requested_action_tags:
        return motion_names

    filtered = set()
    for motion_name in motion_names:
        motion_metadata = _require_motion_metadata_entry(motion_name, motion_metadata_lookup)
        motion_action_tags = _normalize_motion_action_tags(motion_metadata.get('action_tags'))
        if motion_action_tags.intersection(requested_action_tags):
            filtered.add(motion_name)
    return filtered


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

""" extract parents based on first frame """
def get_motion_parents(motion):
    joints_num = motion.shape[1]
    parents_map = np.sum(motion[0]**2, axis=2)
    parents = [-1]
    for j in range(1, joints_num):
        j_parent = np.where(parents_map[j] != 0)[0][0]
        parents.append(j_parent)
    return parents

""" create temporal mask template for window size"""
def create_temporal_mask_for_window(window, max_len, circular=False):
    margin = window // 2
    mask = torch.zeros(max_len+1, max_len+1)
    mask[:, 0] = 1
    for i in range(max_len+1):
        if circular and i > 0 and max_len > 0:
            for delta in range(-margin, margin + 2):
                j = ((i - 1 + delta) % max_len) + 1
                mask[i, j] = 1
        else:
            mask[i, max(0, i - margin):min(max_len + 1, i + margin + 2)] = 1
    return mask


def resample_motion_features(motion, target_num_frames, *, loop_terminal=False):
    source_frames = int(motion.shape[0])
    target_num_frames = int(target_num_frames)
    if source_frames <= 0:
        raise ValueError("Cannot resample an empty motion.")
    if target_num_frames <= 0:
        raise ValueError(f"target_num_frames must be positive, got {target_num_frames}.")
    if source_frames == target_num_frames:
        return motion

    src = np.linspace(0.0, float(source_frames - 1), target_num_frames, endpoint=True, dtype=np.float32)
    lo = np.floor(src).astype(np.int64).clip(0, source_frames - 1)
    hi = np.minimum(lo + 1, source_frames - 1)
    w = (src - np.floor(src))[:, None, None].astype(np.float32)
    resampled = (motion[lo] * (1.0 - w) + motion[hi] * w).astype(motion.dtype, copy=False)

    if resampled.shape[-1] >= 12:
        # Velocity channels are stored so that `vel[t] * step_scale` recovers
        # the target-frame position delta — matching the contract that
        # velocity_consistency_loss and loop_wrap_loss multiply by step_scale
        # (reconstructed from playspeed_cond) before comparing with pos deltas.
        # Linear interpolation of source velocities would give the instantaneous
        # value at src[t], which is off whenever step_scale != 1; instead we
        # integrate to a position path, interpolate that, then take target-step
        # differences divided by step_scale.
        source_velocity_path = np.zeros((source_frames, motion.shape[1], 3), dtype=motion.dtype)
        if source_frames > 1:
            source_velocity_path[1:] = np.cumsum(
                motion[:-1, :, 9:12].astype(np.float64, copy=False),
                axis=0,
            ).astype(motion.dtype, copy=False)
        resampled_velocity_path = (
            source_velocity_path[lo] * (1.0 - w)
            + source_velocity_path[hi] * w
        ).astype(motion.dtype, copy=False)
        if target_num_frames > 1:
            step_scale = float(source_frames - 1) / float(target_num_frames - 1)
            rebuilt_visible_velocity = (
                (resampled_velocity_path[1:] - resampled_velocity_path[:-1])
                / step_scale
            ).astype(motion.dtype, copy=False)
            resampled[:-1, :, 9:12] = rebuilt_visible_velocity
            if loop_terminal:
                resampled[-1, :, 9:12] = (
                    (resampled_velocity_path[0] - resampled_velocity_path[-1])
                    / step_scale
                ).astype(motion.dtype, copy=False)
            else:
                resampled[-1, :, 9:12] = rebuilt_visible_velocity[-1]
        else:
            resampled[0, :, 9:12] = 0.0

    if resampled.shape[-1] >= 13:
        nearest = np.rint(src).astype(np.int64).clip(0, source_frames - 1)
        resampled[..., 12] = (motion[nearest, :, 12] >= 0.5).astype(resampled.dtype, copy=False)

    return resampled.astype(motion.dtype, copy=False)


def _compute_global_energy_condition_np(motion: np.ndarray, n_joints: int) -> np.ndarray:
    # Global energy must describe the clip's physical cadence before we stretch
    # or squeeze it into the fixed training window. After resampling, the model
    # only sees the windowed clip plus playspeed_cond.
    #
    # n_joints is explicit so the statistic stays aligned with the encoder's
    # valid_joints mask even if a future caller passes a joint-padded motion.
    if motion.ndim != 3 or motion.shape[-1] < 13:
        raise ValueError(f"Expected motion with shape (T, J, >=13), got {motion.shape}.")
    n_joints = int(n_joints)
    if n_joints <= 0 or n_joints > motion.shape[1]:
        raise ValueError(
            f"n_joints must be in (0, {motion.shape[1]}], got {n_joints}."
        )

    motion = motion[:, :n_joints, :]
    velocity_norm = np.linalg.norm(motion[..., 9:12], axis=-1)
    rotation_delta = np.zeros_like(motion[..., 3:9])
    if motion.shape[0] > 1:
        rotation_delta[1:] = motion[1:, :, 3:9] - motion[:-1, :, 3:9]
    rotation_delta_norm = np.linalg.norm(rotation_delta, axis=-1)
    energy = np.sqrt(velocity_norm * velocity_norm + rotation_delta_norm * rotation_delta_norm + 1e-6)

    frame_mean = energy.mean(axis=1)
    return np.asarray([frame_mean.mean()], dtype=np.float32)


def _circular_roll_motion(motion, offset):
    length = int(motion.shape[0])
    if length <= 0:
        return motion
    indices = (np.arange(length, dtype=np.int64) + int(offset)) % length
    return motion[indices]


def _tile_loop_motion(motion, repeat_count):
    """Concatenate ``repeat_count`` copies of a loop motion along the time axis.

    Feature consistency at tile boundaries is guaranteed because preprocessing
    writes the terminal velocity (channels 9-11) as the wrap-around delta
    ``pos[0] - pos[-1]`` for all motions classified as loop
    (``detect_motion_loop`` / ``_compute_terminal_local_velocity``).  Because
    the last and first frames are near-identical in a loop clip, the velocity
    at the boundary from copy *k* to copy *k+1* stays physically consistent.

    Binary contact (channel 12) and 6-D rotations (channels 3-8) are
    unaffected by tiling.  Tiling operates in whichever feature space the
    caller supplies (raw or normalized); both are linear transformations of
    each other, so the result is equivalent and the caller must only ensure
    ``playspeed_cond`` reflects the post-tile frame count.
    """
    repeat_count = int(repeat_count)
    if repeat_count <= 1:
        return motion
    return np.concatenate([motion] * repeat_count, axis=0).astype(motion.dtype, copy=False)


_JOINT_MASK_SKIP_TOKENS = {
    'mid',
    'rear',
    'front',
    'back',
    'base',
    'tip',
}
_JOINT_MASK_EXCLUDE_TOKENS = {
    'brain',
    'bip',
    'center',
    'chain',
    'cog',
    'copy',
    'ctrl',
    'dummy',
    'end',
    'fur',
    'hair',
    'helper',
    'ik',
    'joint',
    'link',
    'locator',
    'mane',
    'mesh',
    'node',
    'nub',
    'ponytail',
    'projectile',
    'reins',
    'saddle',
    'site',
    'trajectory',
    'xtra',
}


def _joint_mask_name_tokens(joint_name: str) -> list[str]:
    tokens = []
    for token in str(joint_name or '').split():
        clean_token = re.sub(r'[^a-z0-9]+', '', token.lower())
        clean_token = re.sub(r'\d+$', '', clean_token)
        if clean_token and not clean_token.isdigit():
            tokens.append(clean_token)
    return tokens


def _is_anatomical_non_helper_joint_name(joint_name: str) -> bool:
    tokens = _joint_mask_name_tokens(joint_name)
    if not tokens:
        return False
    if any(token in _JOINT_MASK_EXCLUDE_TOKENS for token in tokens):
        return False
    filtered_tokens = [token for token in tokens if token not in _JOINT_MASK_SKIP_TOKENS]
    return bool(filtered_tokens)


def _build_joint_mask_candidate_roots(object_cond) -> np.ndarray:
    parents = [int(parent_index) for parent_index in object_cond.get('parents', [])]
    joint_count = len(parents)
    candidate_roots = np.zeros((joint_count,), dtype=np.bool_)
    if joint_count <= 1:
        return candidate_roots

    helper_joint_indices = {int(joint_index) for joint_index in object_cond.get('helper_joint_indices') or []}
    canonical_joint_names = list(object_cond.get('canonical_joint_names') or object_cond.get('joints_names') or [])

    for joint_index, parent_index in enumerate(parents):
        if joint_index == 0 or parent_index < 0:
            continue
        if joint_index in helper_joint_indices:
            continue
        joint_name = canonical_joint_names[joint_index] if joint_index < len(canonical_joint_names) else ''
        if _is_anatomical_non_helper_joint_name(joint_name):
            candidate_roots[joint_index] = True

    if np.any(candidate_roots):
        return candidate_roots

    # Fallback: keep all non-root non-helper joints if the semantic filter is too strict.
    for joint_index, parent_index in enumerate(parents):
        if joint_index == 0 or parent_index < 0:
            continue
        if joint_index in helper_joint_indices:
            continue
        candidate_roots[joint_index] = True
    return candidate_roots


def _list_motion_files(motion_dir: str) -> list[str]:
    return sorted(path.name for path in Path(motion_dir).glob("*.npy"))


def _compute_split_counts(num_items: int) -> dict[str, int]:
    if num_items <= 0:
        return {split: 0 for split in SUPPORTED_SPLITS}
    if num_items == 1:
        return {"train": 1, "val": 0, "test": 0}
    if num_items == 2:
        return {"train": 1, "val": 1, "test": 0}
    if num_items == 3:
        return {"train": 1, "val": 1, "test": 1}

    raw_counts = {split: DEFAULT_SPLIT_RATIOS[split] * num_items for split in SUPPORTED_SPLITS}
    counts = {split: int(np.floor(raw_counts[split])) for split in SUPPORTED_SPLITS}
    # Minimums respect the split ratios - if a split has 0.0 ratio, it should have 0 minimum
    minimums = {split: (1 if DEFAULT_SPLIT_RATIOS[split] > 0 else 0) for split in SUPPORTED_SPLITS}

    for split, minimum in minimums.items():
        counts[split] = max(counts[split], minimum)

    while sum(counts.values()) > num_items:
        removable = [
            split for split in SUPPORTED_SPLITS
            if counts[split] > minimums[split]
        ]
        if not removable:
            break
        split_to_reduce = max(removable, key=lambda split: counts[split] - raw_counts[split])
        counts[split_to_reduce] -= 1

    while sum(counts.values()) < num_items:
        split_to_increase = max(SUPPORTED_SPLITS, key=lambda split: raw_counts[split] - counts[split])
        counts[split_to_increase] += 1

    return counts


def _compute_filtered_split_counts(num_items: int) -> dict[str, int]:
    if num_items <= 0:
        return {split: 0 for split in SUPPORTED_SPLITS}
    if num_items <= 2:
        return {"train": num_items, "val": 0, "test": 0}
    if num_items == 3:
        return {"train": 2, "val": 1, "test": 0}
    if num_items == 4:
        return {"train": 3, "val": 1, "test": 0}
    return _compute_split_counts(num_items)


def ensure_split_manifests(data_root: str, motion_dir: str) -> dict[str, Path]:
    data_root_path = Path(data_root)
    split_paths = {split: data_root_path / f"{split}.txt" for split in SUPPORTED_SPLITS}

    # Group motion names by object_type (animal character)
    grouped_motion_names: dict[str, list[str]] = defaultdict(list)
    from utils.misc import infer_object_type_from_filename
    for motion_name in _list_motion_files(motion_dir):
        grouped_motion_names[infer_object_type_from_filename(motion_name)].append(motion_name)

    # Shuffle object types and assign all their motions to the same split
    manifests = {split: [] for split in SUPPORTED_SPLITS}
    rng = random.Random(DEFAULT_SPLIT_SEED)
    object_types = sorted(grouped_motion_names.keys())
    rng.shuffle(object_types)
    split_counts = _compute_split_counts(len(object_types))
    start_index = 0
    for split in SUPPORTED_SPLITS:
        end_index = start_index + split_counts[split]
        for object_type in object_types[start_index:end_index]:
            manifests[split].extend(grouped_motion_names[object_type])
        start_index = end_index

    for split, split_path in split_paths.items():
        split_path.write_text("\n".join(sorted(manifests[split])) + "\n", encoding="utf-8")

    print(f"Generated dataset split manifests under {data_root_path}")
    return split_paths


def load_motion_names_for_split(split: str, data_root: str, motion_dir: str) -> set[str]:
    if split == ALL_SPLIT_NAME:
        motion_names = set(_list_motion_files(motion_dir))
        if not motion_names:
            raise RuntimeError(f"Split '{split}' is empty: {motion_dir}")
        return motion_names
    split_paths = ensure_split_manifests(data_root, motion_dir)
    split_path = split_paths[split]
    motion_names = {
        line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()
    }
    if not motion_names:
        raise RuntimeError(f"Split '{split}' is empty: {split_path}")
    return motion_names


def load_motion_names_for_split_with_action_tags(
    split: str,
    data_root: str,
    motion_dir: str,
    raw_action_tags,
    motion_metadata_lookup,
) -> set[str]:
    requested_action_tags = set(parse_action_tags(raw_action_tags))
    if not requested_action_tags:
        return load_motion_names_for_split(split, data_root, motion_dir)

    all_motion_names = set(_list_motion_files(motion_dir))
    filtered_motion_names = filter_motion_names_by_action_tags(
        all_motion_names,
        raw_action_tags,
        motion_metadata_lookup,
    )
    if split == ALL_SPLIT_NAME:
        return filtered_motion_names

    grouped_motion_names: dict[str, list[str]] = defaultdict(list)
    for motion_name in sorted(filtered_motion_names):
        motion_metadata = _require_motion_metadata_entry(motion_name, motion_metadata_lookup)
        object_type = str(motion_metadata.get('object_type'))
        grouped_motion_names[object_type].append(motion_name)

    # Shuffle object types and assign all their motions to the same split
    all_split_results: dict[str, set[str]] = {s: set() for s in SUPPORTED_SPLITS}
    rng = random.Random(DEFAULT_SPLIT_SEED)
    object_types_list = sorted(grouped_motion_names.keys())
    rng.shuffle(object_types_list)
    split_counts = _compute_filtered_split_counts(len(object_types_list))
    start_index = 0
    for current_split in SUPPORTED_SPLITS:
        end_index = start_index + split_counts[current_split]
        for object_type in object_types_list[start_index:end_index]:
            all_split_results[current_split].update(grouped_motion_names[object_type])
        start_index = end_index

    selected_motion_names = all_split_results[split]

    if not selected_motion_names:
        raise RuntimeError(
            f"Split '{split}' is empty after filtering action_tags={sorted(requested_action_tags)}"
        )
    
    # Generate split manifest files for manual verification
    data_root_path = Path(data_root)
    for split_name in SUPPORTED_SPLITS:
        split_path = data_root_path / f"{split_name}.txt"
        if all_split_results[split_name]:
            split_path.write_text("\n".join(sorted(all_split_results[split_name])) + "\n", encoding="utf-8")
    
    return selected_motion_names


def _motion_length_cache_path(data_root: str) -> Path:
    cache_dir = Path(data_root) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "motion_lengths.npy"


def _load_motion_length_cache(cache_path: Path) -> dict[str, dict[str, int]]:
    if not cache_path.exists():
        return {}
    try:
        payload = np.load(cache_path, allow_pickle=True).item()
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        return {}
    return entries


def _save_motion_length_cache(cache_path: Path, entries: dict[str, dict[str, int]]) -> None:
    np.save(cache_path, {"entries": entries}, allow_pickle=True)


def _read_motion_length(motion_path: str) -> int:
    motion = np.load(motion_path, mmap_mode='r')
    return len(motion)


def ensure_joint_name_embeddings(
    cond_dict: dict,
    expected_embedding_dim: Optional[int] = None,
    cond_source: str = 'cond.npy',
) -> dict:
    for object_type, cond in cond_dict.items():
        joints_names_embs = cond.get('joints_names_embs')
        if joints_names_embs is None:
            raise RuntimeError(
                f"{cond_source} for '{object_type}' is missing joints_names_embs. "
                "Re-run preprocessing with the updated pipeline."
            )

        joints_names_embs = np.asarray(joints_names_embs, dtype=np.float32)
        if joints_names_embs.ndim != 2:
            raise RuntimeError(
                f"{cond_source} for '{object_type}' has invalid joints_names_embs shape {joints_names_embs.shape}."
            )

        if expected_embedding_dim is not None and int(joints_names_embs.shape[1]) != int(expected_embedding_dim):
            raise RuntimeError(
                f"{cond_source} for '{object_type}' has embedding dim {joints_names_embs.shape[1]} "
                f"but the model expects {expected_embedding_dim}."
            )

        parents = cond.get('parents')
        joint_count = len(parents) if parents is not None else 0
        if joint_count and joints_names_embs.shape[0] != joint_count:
            raise RuntimeError(
                f"{cond_source} for '{object_type}' has {joints_names_embs.shape[0]} joint-name embeddings "
                f"but {joint_count} joints in parents."
            )

        meta = cond.get('joints_names_embs_meta')
        if isinstance(meta, dict):
            schema_version = meta.get('schema_version')
            if schema_version is not None and int(schema_version) != JOINT_NAME_EMBEDDING_SCHEMA_VERSION:
                warnings.warn(
                    f"{cond_source} for '{object_type}' uses joint-name embedding schema {schema_version}; "
                    f"current code expects {JOINT_NAME_EMBEDDING_SCHEMA_VERSION}.",
                    stacklevel=2,
                )

        cond['joints_names_embs'] = joints_names_embs
    return cond_dict

'''For use of training text motion matching model, and evaluations'''
class MotionDataset(data.Dataset):
    def __init__(self, opt, cond_dict, temporal_window, balanced, num_frames, sample_limit=0, allowed_motion_names: Optional[set[str]] = None, motion_metadata_lookup: Optional[dict[str, dict[str, object]]] = None):
        self.opt = opt
        self.temporal_window = int(temporal_window)
        self.min_length = int(getattr(opt, 'min_length', 20))
        self.pointer = 0
        self.max_motion_length = num_frames
        self.cond_dict = cond_dict
        self.balanced = balanced
        # A weighted sampler drives indexing when species are balanced; it yields
        # absolute name_list indices, so __getitem__ must skip the pointer offset
        # in that case.
        self.use_weighted_sampler = bool(self.balanced)
        self.sample_limit = max(0, int(sample_limit))
        self.motion_cache_size = max(0, int(getattr(opt, 'motion_cache_size', 0)))
        self.motion_cache = OrderedDict()
        data_dict = {}
        all_object_types = self.cond_dict.keys()
        if motion_metadata_lookup is None:
            motion_metadata_lookup = load_motion_metadata(opt.data_root)
        new_name_list = []
        length_list = []
        motion_length_cache_path = _motion_length_cache_path(opt.data_root)
        motion_length_cache = _load_motion_length_cache(motion_length_cache_path)
        cache_dirty = False

        all_motion_files = [name for name in os.listdir(opt.motion_dir) if name.endswith('.npy')]
        if allowed_motion_names is not None:
            all_motion_files = [name for name in all_motion_files if name in allowed_motion_names]

        for object_type in all_object_types:
            object_motions = [name for name in all_motion_files if name.startswith(f'{object_type}_')]
            
            for name in object_motions:
                motion_metadata = _require_motion_metadata_entry(name, motion_metadata_lookup)
                try:
                    motion_path = pjoin(opt.motion_dir, name)
                    stat = os.stat(motion_path)
                    cache_entry = motion_length_cache.get(name)
                    if cache_entry is not None and cache_entry.get('mtime_ns') == stat.st_mtime_ns and cache_entry.get('size_bytes') == stat.st_size:
                        motion_length = int(cache_entry['length'])
                    else:
                        motion_length = _read_motion_length(motion_path)
                        motion_length_cache[name] = {
                            'length': int(motion_length),
                            'mtime_ns': int(stat.st_mtime_ns),
                            'size_bytes': int(stat.st_size),
                        }
                        cache_dirty = True
                    data_dict[name] = {
                                        'motion_path': motion_path,
                                        'length': motion_length,
                                        'object_type': object_type,
                                        'motion_metadata': motion_metadata,
                                       }
                                       
                    new_name_list.append(name)
                    length_list.append(motion_length)
                except Exception:
                    pass

        if cache_dirty:
            _save_motion_length_cache(motion_length_cache_path, motion_length_cache)
                
        sorted_pairs = sorted(zip(new_name_list, length_list), key=lambda x: x[1])
        if not sorted_pairs:
            raise RuntimeError("No motion clips were found for the requested dataset subset.")

        minimum_valid_index = np.searchsorted([pair[1] for pair in sorted_pairs], self.min_length)
        if self.sample_limit > 0:
            preferred_valid_index = np.searchsorted([pair[1] for pair in sorted_pairs], self.max_motion_length)
            candidate_pairs = sorted_pairs[preferred_valid_index:]
            if len(candidate_pairs) < self.sample_limit:
                candidate_pairs = sorted_pairs[minimum_valid_index:]

            if len(candidate_pairs) > self.sample_limit:
                candidate_pairs = random.sample(candidate_pairs, self.sample_limit)

            sorted_pairs = sorted(candidate_pairs, key=lambda x: x[1])

        name_list, length_list = zip(*sorted_pairs)
        name_list = list(name_list)
        length_list = list(length_list)
        if self.sample_limit <= 0:
            name_list = name_list[minimum_valid_index:]
            length_list = length_list[minimum_valid_index:]
            data_dict = {name: data_dict[name] for name in name_list}
        else:
            data_dict = {name: data_dict[name] for name in name_list}
        self.length_arr = np.array(length_list)
        self.max_available_length = int(self.length_arr.max()) if len(self.length_arr) > 0 else 0
        self.data_dict = data_dict
        self.name_list = name_list
        self.temporal_mask_template = create_temporal_mask_for_window(self.temporal_window, self.max_motion_length)
        self.circular_temporal_mask_template = create_temporal_mask_for_window(
            self.temporal_window,
            self.max_motion_length,
            circular=True,
        )
        self.reset_min_len(self.min_length)

    def reset_min_len(self, length):
        if self.max_available_length > 0:
            length = min(length, self.max_available_length)
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        self.min_length = length
    
    def inv_transform(self, x, y):
        return canonical_to_physical_hml(x, self.cond_dict[y['object_type']])

    def _get_temporal_mask(self, target_num_frames, circular=False):
        if int(target_num_frames) == int(self.max_motion_length):
            return self.circular_temporal_mask_template if circular else self.temporal_mask_template
        return create_temporal_mask_for_window(self.temporal_window, int(target_num_frames), circular=circular)

    def _sample_loop_offset(self, length, loop_offset=None):
        length = int(length)
        if length <= 0:
            return 0
        if loop_offset is None:
            return random.randint(0, length - 1)
        offset = int(loop_offset)
        if offset < 0 or offset >= length:
            raise ValueError(f"loop_offset={offset} is invalid for loop motion with length={length}.")
        return offset

    def _sample_loop_tile_count(self, length, max_source_length):
        """Randomly select how many times to repeat a loop motion so the
        tiled length stays within ``[length, max_source_length]``.

        Returns ``1`` (no tiling) when the single-cycle length already
        exceeds the budget (``length > max_source_length``).  Otherwise
        picks uniformly from ``{1, …, max_source_length // length}``.
        """
        length = int(length)
        max_source_length = int(max_source_length)
        if length <= 0 or max_source_length <= 0:
            return 1
        max_tile_count = max_source_length // length
        if max_tile_count <= 1:
            return 1
        return int(random.randint(1, max_tile_count))

    def prepare_sample_by_name(self, name, target_num_frames=None, crop_start=None, loop_offset=None):
        if name not in self.data_dict:
            raise KeyError(f"Unknown motion sample '{name}'.")
        return self._prepare_sample(
            name,
            self.data_dict[name],
            target_num_frames=target_num_frames,
            crop_start=crop_start,
            loop_offset=loop_offset,
        )

    def _prepare_sample(self, name, data, target_num_frames=None, crop_start=None, loop_offset=None, return_aug_info=False):
        if target_num_frames is None:
            target_num_frames = self.max_motion_length
        target_num_frames = int(target_num_frames)
        if target_num_frames <= 0:
            raise ValueError(f"target_num_frames must be positive, got {target_num_frames}.")

        motion_metadata = _copy_required_motion_metadata(name, data.get('motion_metadata'))
        is_loop = bool(motion_metadata.get('is_loop'))
        loop_cond_prob = float(getattr(self.opt, 'loop_cond_prob', 1.0))
        if not 0.0 <= loop_cond_prob <= 1.0:
            raise ValueError(f"loop_cond_prob must be in [0, 1], got {loop_cond_prob}.")
        # loop_cond_prob: probability that a loop clip STAYS loop-conditioned.
        # When the random draw succeeds, loop_uncond is False (loop path active).
        # When it fails, loop_uncond is True — the motion is still physically a
        # loop, but the model is *told* it is not (is_loop=False in metadata,
        # no circular temporal mask).
        #
        # IMPORTANT: loop_uncond only controls the *label / conditioning*
        # exposed to the model.  The data-level augmentations below (circular
        # roll + tile for diverse phase coverage) apply to **all** is_loop
        # motions regardless of loop_uncond.  This keeps training-data
        # diversity high while the loop_uncond path trains the model to
        # denoise loop-shaped data WITHOUT explicit loop priors.
        loop_uncond = bool(
            is_loop
            and loop_offset is None
            and loop_cond_prob < 1.0
            and random.random() >= loop_cond_prob
        )

        motion, m_length, object_type, parents, joints_graph_dist, joints_relations, rest_pose, offsets, joints_names_embs, kinematic_chains = self._load_physical_motion(data)
        ind = 0
        loop_applied = False
        loop_full_cycle = False
        loop_phase_offset = 0
        loop_tile_count = 1
        loop_condition_active = bool(is_loop) and not loop_uncond

        max_source_length = target_num_frames * 2
        # ── Loop-aware data augmentation (applies to ALL is_loop motions) ──
        # Circular roll shifts the temporal phase so the model sees every loop
        # from a random starting frame.  Random tiling repeats the cycle up to
        # 2× target length so the subsequent resample has enough source frames
        # to produce a clean stretched/clipped result without heavy speed
        # distortion.  Both are pure temporal operations on feature arrays —
        # they preserve per-frame feature semantics (see _tile_loop_motion and
        # _circular_roll_motion for consistency guarantees).
        if is_loop:
            loop_phase_offset = self._sample_loop_offset(m_length, loop_offset=loop_offset)
            if loop_phase_offset != 0:
                motion = _circular_roll_motion(motion, loop_phase_offset)
            loop_tile_count = self._sample_loop_tile_count(m_length, max_source_length)
            motion = _tile_loop_motion(motion, loop_tile_count)
            m_length = int(motion.shape[0])

        if loop_condition_active and m_length > max_source_length:
            loop_condition_active = False
            loop_uncond = True

        if crop_start is not None and m_length > target_num_frames:
            crop_length = target_num_frames
            max_start = m_length - crop_length
            ind = int(crop_start)
            if ind < 0 or ind > max_start:
                raise ValueError(
                    f"crop_start={ind} is invalid for motion '{name}' with length={m_length} and crop_length={crop_length}."
                )
            motion = motion[ind: ind + crop_length]
            m_length = int(motion.shape[0])
            if loop_condition_active:
                loop_uncond = True
            loop_condition_active = False
        elif m_length > max_source_length:
            # Long loops have been downgraded to non-loop above, so this path
            # is always a linear crop (no circular indexing needed).
            # Sample crop_length from a normal distribution centered at
            # target_num_frames (peak probability) with std=target_num_frames/2
            # so ±2σ roughly spans [0, 2n]; clip to the valid range.
            crop_length = int(round(random.gauss(target_num_frames, target_num_frames / 2.0)))
            crop_length = max(self.min_length, min(max_source_length, crop_length))
            max_start = m_length - crop_length
            if crop_start is None:
                ind = random.randint(0, max_start)
            else:
                ind = int(crop_start)
                if ind < 0 or ind > max_start:
                    raise ValueError(
                        f"crop_start={ind} is invalid for motion '{name}' with length={m_length} and crop_length={crop_length}."
                    )
            motion = motion[ind: ind + crop_length]
            m_length = int(motion.shape[0])
            loop_condition_active = False

        source_len_for_playspeed = int(m_length)
        playspeed_cond = float(source_len_for_playspeed) / float(target_num_frames)
        # Capture clip-level energy before window resampling. The later
        # resample changes temporal density for batching, not the motion's
        # physical velocity / rotation-energy semantics. Pass the real joint
        # count explicitly so the statistic matches the encoder's valid-joint
        # mask if a future caller pre-pads the joint axis.
        global_energy_cond = _compute_global_energy_condition_np(motion, n_joints=int(motion.shape[1]))

        if m_length != target_num_frames:
            motion = resample_motion_features(
                motion,
                target_num_frames,
                loop_terminal=loop_condition_active,
            )
            m_length = target_num_frames

        motion = np.nan_to_num(
            physical_hml_to_canonical(motion, self.cond_dict[object_type])
        ).astype(np.float32, copy=False)

        if loop_condition_active:
            loop_full_cycle = True
            loop_applied = True

        motion_metadata['is_loop'] = bool(loop_condition_active)
        motion_metadata['loop_full_cycle'] = bool(loop_full_cycle)
        motion_metadata['playspeed_cond'] = float(playspeed_cond)
        motion_metadata['global_energy_cond'] = global_energy_cond
        motion_metadata['loop_data_aug_applied'] = bool(is_loop)
        motion_metadata['loop_phase_offset'] = int(loop_phase_offset)
        motion_metadata['loop_tile_count'] = int(loop_tile_count)
        # loop_phase_length: the expected single-cycle period in output frames.
        # When multiple tile copies were resampled to one target window the
        # effective cycle length is compressed proportionally.
        # Example: 32f loop tiled 2× → 64f resampled to 60f → phase_len ≈ 30.5.
        loop_phase_length = target_num_frames
        if loop_condition_active and loop_full_cycle and loop_tile_count > 1:
            loop_phase_length = ((float(target_num_frames) - 1.0) / float(loop_tile_count)) + 1.0
        motion_metadata['loop_phase_length'] = float(
            loop_phase_length if loop_condition_active and loop_full_cycle else max(int(m_length), 1)
        )
        circular_mask = bool(loop_full_cycle)
        temporal_mask = self._get_temporal_mask(target_num_frames, circular=circular_mask)

        if return_aug_info:
            return motion, m_length, parents, rest_pose, offsets, temporal_mask, joints_graph_dist, joints_relations, object_type, joints_names_embs, ind, self.opt.max_joints, motion_metadata, name, {
                'joint_mask_candidate_roots': self.cond_dict[object_type]['joint_mask_candidate_roots'],
                'species_emb': self.cond_dict[object_type].get('species_emb'),
                'rest_pose_physical': self.cond_dict[object_type]['rest_pose'],
                'rest_pos_ric_hml': self.cond_dict[object_type]['rest_pos_ric_hml'],
                'canonical_feature_mean': self.cond_dict[object_type].get('canonical_feature_mean'),
                'canonical_feature_std': self.cond_dict[object_type].get('canonical_feature_std'),
                'feature_space': self.cond_dict[object_type].get('feature_space', 'canonical_motion_v3'),
            }, {
                'crop_start': int(ind),
                'loop_applied': bool(loop_applied),
                'loop_phase_offset': int(loop_phase_offset),
                'loop_tile_count': int(loop_tile_count),
                'playspeed_cond': float(playspeed_cond),
                'loop_uncond': bool(loop_uncond),
            }
        return motion, m_length, parents, rest_pose, offsets, temporal_mask, joints_graph_dist, joints_relations, object_type, joints_names_embs, ind, self.opt.max_joints, motion_metadata, name, {
            'joint_mask_candidate_roots': self.cond_dict[object_type]['joint_mask_candidate_roots'],
            'species_emb': self.cond_dict[object_type].get('species_emb'),
            'rest_pose_physical': self.cond_dict[object_type]['rest_pose'],
            'rest_pos_ric_hml': self.cond_dict[object_type]['rest_pos_ric_hml'],
            'canonical_feature_mean': self.cond_dict[object_type].get('canonical_feature_mean'),
            'canonical_feature_std': self.cond_dict[object_type].get('canonical_feature_std'),
            'feature_space': self.cond_dict[object_type].get('feature_space', 'canonical_motion_v3'),
        }
    
    def _load_physical_motion(self, data):
        object_type = data['object_type']
        cond = self.cond_dict[object_type]
        motion_path = data['motion_path']
        if self.motion_cache_size > 0:
            motion = self.motion_cache.get(motion_path)
            if motion is None:
                motion = np.load(motion_path).astype(np.float32, copy=False)
                self.motion_cache[motion_path] = motion
                self.motion_cache.move_to_end(motion_path)
                while len(self.motion_cache) > self.motion_cache_size:
                    self.motion_cache.popitem(last=False)
            else:
                self.motion_cache.move_to_end(motion_path)
        else:
            motion = np.load(motion_path).astype(np.float32, copy=False)

        m_length = motion.shape[0]
        rest_pose = build_canonical_rest_feature(cond)
        offsets = cond['offsets']
        return (motion, m_length, object_type, cond['parents'], cond['joints_graph_dist'], cond['joint_relations'], rest_pose, offsets, cond['joints_names_embs'], cond['kinematic_chains'])
        
    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        if self.use_weighted_sampler:
            idx = item #self.pointer + item (handled in weighted sampler)
        else:
            idx = self.pointer + item
        name = self.name_list[idx]
        return self.prepare_sample_by_name(name)

class TruebonesSampler(WeightedRandomSampler):
    """Sub-balanced weighted sampler for species fairness.

    Each species' total sampling mass is proportional to the square root of its
    clip count, then normalized across all non-empty species; within a species
    the mass is split uniformly across its clips. This is a softer middle ground
    than full per-species balancing: a species with 9 clips is sampled 3x
    (=sqrt(9)) as often as a single-clip species, rather than equally (full
    balance) or 9x (uniform per-clip). The clip count is taken over the already
    split/action_tags-filtered ``name_list``, so it reflects only the clips
    actually present in this training subset.
    """
    def __init__(self, data_source):
        motion_dataset = data_source.motion_dataset
        num_samples = len(data_source)
        name_list = motion_dataset.name_list
        total_samples = len(name_list)
        pointer = motion_dataset.pointer
        weights = np.zeros(total_samples, dtype=np.float64)

        object_types = motion_dataset.cond_dict.keys()
        # Collect all object types that have samples
        non_empty_types = []
        for object_type in object_types:
            object_indices = [i for i in range(pointer, len(name_list)) if name_list[i].startswith(f'{object_type}_')]
            if len(object_indices) > 0:
                non_empty_types.append((object_type, object_indices))

        # Re-balance weights among only the non-empty object types
        if len(non_empty_types) == 0:
            raise RuntimeError(f"No samples found for any object type in split with pointer={pointer}. "
                             f"Available samples: {[name_list[i] for i in range(pointer, min(pointer+5, len(name_list)))]}")

        # Per-species mass ~ sqrt(clip count over this filtered subset).
        species_shares = [np.sqrt(len(object_indices)) for _, object_indices in non_empty_types]
        total_share = float(np.sum(species_shares))

        for (object_type, object_indices), share in zip(non_empty_types, species_shares):
            indices = np.asarray(object_indices)
            n = len(indices)
            weights[indices] = (share / total_share) / n

        super().__init__(num_samples=num_samples, weights=weights)
    
class Truebones(data.Dataset):
    def __init__(self, split="train", temporal_window=31, **kwargs):
        if split not in SUPPORTED_SPLITS and split != ALL_SPLIT_NAME:
            raise ValueError(f"Unsupported split '{split}'. Expected one of {SUPPORTED_SPLITS + (ALL_SPLIT_NAME,)}.")
        abs_base_path = f'.'
        device = None  # torch.device('cuda:4') # This param is not in use in this context
        opt = get_opt(device)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        self.opt = opt
        self.balanced = kwargs['balanced']
        self.objects_subset = kwargs['objects_subset']
        self.action_tags = kwargs.get('action_tags', '')
        self.sample_limit = kwargs.get('sample_limit', 0)
        self.motion_cache_size = kwargs.get('motion_cache_size', 0)
        self.opt.motion_cache_size = self.motion_cache_size
        self.opt.min_length = int(kwargs.get('min_length', getattr(self.opt, 'min_length', 20)))

        self.opt.loop_cond_prob = kwargs.get('loop_cond_prob', 1.0)
        cond_dict = np.load(opt.cond_file, allow_pickle=True).item()
        cond_dict = refresh_joint_metadata_in_cond_dict(cond_dict)
        # Support both predefined subsets and single species names
        if self.objects_subset in opt.subsets_dict:
            subset = opt.subsets_dict[self.objects_subset]
        else:
            # Treat as a single species name
            subset = [self.objects_subset]
        cond_dict = {k:cond_dict[k] for k in subset if k in cond_dict}
        # Fast-fail before training if any species being trained lacks a motion
        # tag (the per-species condition has no fallback).
        assert_species_tags_cover(cond_dict.keys())
        cond_dict = ensure_joint_name_embeddings(cond_dict, cond_source=opt.cond_file)
        for object_type, cond in cond_dict.items():
            mark_canonical_cond_entry(cond)
            if cond.get('canonical_feature_mean') is None or cond.get('canonical_feature_std') is None:
                raise RuntimeError(
                    f"cond entry '{object_type}' is missing canonical_feature_mean/std. "
                    "Regenerate cond.npy (python tools/regenerate_dataset_artifacts.py) to "
                    "compute the per-object_subset canonical standardization statistics."
                )
            cond['joint_mask_candidate_roots'] = _build_joint_mask_candidate_roots(cond)

        motion_metadata_lookup = load_motion_metadata(opt.data_root)
        self.split_file = pjoin(opt.data_root, f'{split}.txt') if split != ALL_SPLIT_NAME else ''
        allowed_motion_names = load_motion_names_for_split_with_action_tags(
            split,
            opt.data_root,
            opt.motion_dir,
            self.action_tags,
            motion_metadata_lookup,
        )
        self.motion_dataset = MotionDataset(
            self.opt,
            cond_dict,
            temporal_window,
            self.balanced,
            num_frames=kwargs['num_frames'],
            sample_limit=self.sample_limit,
            allowed_motion_names=allowed_motion_names,
            motion_metadata_lookup=motion_metadata_lookup,
        )
        assert len(self.motion_dataset) > 0, 'You loaded an empty dataset, ' \
                                          'it is probably because your data dir has only texts and no motions.\n' \
                                          'To train and evaluate MDM you should get the FULL data as described ' \
                                          'in the README file.'

    def __getitem__(self, item):
        return self.motion_dataset.__getitem__(item)

    def __len__(self):
        return self.motion_dataset.__len__()
