import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import os
import re
from collections import OrderedDict, defaultdict
from os.path import join as pjoin
from pathlib import Path
import random
from typing import Optional
from torch.utils.data._utils.collate import default_collate
from data_loaders.truebones.truebones_utils.get_opt import get_opt
from data_loaders.truebones.truebones_utils.param_utils import parse_action_tags
from data_loaders.truebones.truebones_utils.motion_labels import infer_motion_labels_from_motion_name, load_motion_metadata
from data_loaders.truebones.truebones_utils.motion_process import remove_joints_augmentation, add_joint_augmentation
from model.conditioners import T5Conditioner


DEFAULT_SPLIT_RATIOS = {"train": 1.0, "val": 0.0, "test": 0.0}
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


def filter_motion_names_by_action_tags(
    motion_names,
    raw_action_tags,
    motion_metadata_lookup,
    object_types,
):
    requested_action_tags = set(parse_action_tags(raw_action_tags))
    if not requested_action_tags:
        return motion_names

    filtered = set()
    for motion_name in motion_names:
        motion_metadata = motion_metadata_lookup.get(motion_name)
        if motion_metadata is None:
            motion_metadata = infer_motion_labels_from_motion_name(
                motion_name,
                object_types=object_types,
            )
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
def create_temporal_mask_for_window(window, max_len):
    margin = window // 2
    mask = torch.zeros(max_len+1, max_len+1)
    mask[:, 0] = 1
    for i in range(max_len+1):
        mask[i, max(0, i - margin):min(max_len + 1, i + margin + 2)] = 1
    return mask


def _list_motion_files(motion_dir: str) -> list[str]:
    return sorted(path.name for path in Path(motion_dir).glob("*.npy"))


def _infer_object_type_from_motion_name(name: str) -> str:
    return name.split("_", 1)[0]


def _normalize_fixed_motion_name(raw_value: str) -> str:
    value = str(raw_value or '').strip()
    if not value:
        return ''
    normalized = value.replace('\\', '/')
    base_name = Path(normalized).name
    stem, suffix = os.path.splitext(base_name)
    suffix = suffix.lower()
    if suffix == '.bvh':
        return f'{stem}.npy'
    if suffix == '.npy':
        return base_name
    if base_name.lower().endswith('.npy'):
        return base_name
    return f'{base_name}.npy'


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
    minimums = {"train": 1, "val": 1, "test": 1}

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
    if all(path.exists() for path in split_paths.values()):
        return split_paths

    grouped_motion_names: dict[str, list[str]] = defaultdict(list)
    for motion_name in _list_motion_files(motion_dir):
        grouped_motion_names[_infer_object_type_from_motion_name(motion_name)].append(motion_name)

    manifests = {split: [] for split in SUPPORTED_SPLITS}
    rng = random.Random(DEFAULT_SPLIT_SEED)
    for object_type in sorted(grouped_motion_names):
        motion_names = sorted(grouped_motion_names[object_type])
        rng.shuffle(motion_names)
        split_counts = _compute_split_counts(len(motion_names))
        start_index = 0
        for split in SUPPORTED_SPLITS:
            end_index = start_index + split_counts[split]
            manifests[split].extend(motion_names[start_index:end_index])
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
    object_types,
) -> set[str]:
    requested_action_tags = set(parse_action_tags(raw_action_tags))
    if not requested_action_tags:
        return load_motion_names_for_split(split, data_root, motion_dir)

    all_motion_names = set(_list_motion_files(motion_dir))
    filtered_motion_names = filter_motion_names_by_action_tags(
        all_motion_names,
        raw_action_tags,
        motion_metadata_lookup,
        object_types,
    )
    if split == ALL_SPLIT_NAME:
        return filtered_motion_names

    grouped_motion_names: dict[str, list[str]] = defaultdict(list)
    for motion_name in sorted(filtered_motion_names):
        motion_metadata = motion_metadata_lookup.get(motion_name)
        if motion_metadata is None:
            motion_metadata = infer_motion_labels_from_motion_name(
                motion_name,
                object_types=object_types,
            )
        object_type = str(motion_metadata.get('object_type') or _infer_object_type_from_motion_name(motion_name))
        grouped_motion_names[object_type].append(motion_name)

    selected_motion_names: set[str] = set()
    rng = random.Random(DEFAULT_SPLIT_SEED)
    for object_type in sorted(grouped_motion_names):
        motion_names = sorted(grouped_motion_names[object_type])
        rng.shuffle(motion_names)
        split_counts = _compute_filtered_split_counts(len(motion_names))
        start_index = 0
        for current_split in SUPPORTED_SPLITS:
            end_index = start_index + split_counts[current_split]
            if current_split == split:
                selected_motion_names.update(motion_names[start_index:end_index])
            start_index = end_index

    if not selected_motion_names:
        raise RuntimeError(
            f"Split '{split}' is empty after filtering action_tags={sorted(requested_action_tags)}"
        )
    return selected_motion_names


def _sanitize_cache_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return sanitized.strip("._") or "default"


def _joint_name_embedding_cache_path(data_root: str, t5_name: str) -> Path:
    cache_dir = Path(data_root) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"joint_name_t5_{_sanitize_cache_component(t5_name)}.npy"


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


def _load_cached_joint_name_embeddings(cache_path: Path, cond_file: str, expected_object_types: set[str]) -> Optional[dict[str, np.ndarray]]:
    if not cache_path.exists():
        return None

    try:
        payload = np.load(cache_path, allow_pickle=True).item()
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    metadata = payload.get("_meta", {})
    embeddings = payload.get("embeddings")
    if not isinstance(metadata, dict) or not isinstance(embeddings, dict):
        return None

    cond_mtime_ns = Path(cond_file).stat().st_mtime_ns
    if metadata.get("cond_mtime_ns") != cond_mtime_ns:
        return None

    missing_objects = [object_type for object_type in expected_object_types if object_type not in embeddings]
    if missing_objects:
        return None

    return {object_type: np.asarray(embeddings[object_type], dtype=np.float32) for object_type in expected_object_types}


def _build_joint_name_embeddings(cond_dict: dict, t5_name: str) -> dict[str, np.ndarray]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Building cached joint-name embeddings with {t5_name} on {device.upper()}")
    t5_conditioner = T5Conditioner(
        name=t5_name,
        finetune=False,
        word_dropout=0.0,
        normalize_text=False,
        device=device,
    )

    embeddings = {}
    with torch.no_grad():
        for object_type in sorted(cond_dict):
            joints_names = cond_dict[object_type]['joints_names']
            names_tokens = t5_conditioner.tokenize(joints_names)
            embs = t5_conditioner(names_tokens)
            embeddings[object_type] = embs.detach().cpu().numpy().astype(np.float32, copy=False)
    return embeddings


def attach_joint_name_embeddings(cond_dict: dict, cond_file: str, data_root: str, t5_name: str) -> dict:
    object_types = set(cond_dict.keys())
    cache_path = _joint_name_embedding_cache_path(data_root, t5_name)
    cached_embeddings = _load_cached_joint_name_embeddings(cache_path, cond_file, object_types)

    if cached_embeddings is None:
        cached_embeddings = _build_joint_name_embeddings(cond_dict, t5_name)
        payload = {
            "_meta": {
                "t5_name": t5_name,
                "cond_mtime_ns": Path(cond_file).stat().st_mtime_ns,
            },
            "embeddings": cached_embeddings,
        }
        np.save(cache_path, payload, allow_pickle=True)
        print(f"Saved joint-name embedding cache to {cache_path}")

    for object_type in object_types:
        cond_dict[object_type]['joints_names_embs'] = cached_embeddings[object_type]
    return cond_dict

'''For use of training text motion matching model, and evaluations'''
class MotionDataset(data.Dataset):
    def __init__(self, opt, cond_dict, temporal_window, t5_name, balanced, sample_limit=0, allowed_motion_names: Optional[set[str]] = None, motion_metadata_lookup: Optional[dict[str, dict[str, object]]] = None):
        self.opt = opt
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        self.cond_dict = cond_dict
        self.balanced = balanced
        self.sample_limit = max(0, int(sample_limit))
        self.fixed_motion_name = _normalize_fixed_motion_name(getattr(opt, 'fixed_motion', ''))
        self.fixed_window_start = int(getattr(opt, 'fixed_window_start', 0))
        self.fixed_motion_virtual_length = 1
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
        if self.fixed_motion_name:
            fixed_motion_path = pjoin(opt.motion_dir, self.fixed_motion_name)
            if not os.path.exists(fixed_motion_path):
                raise FileNotFoundError(
                    f"Fixed motion '{self.fixed_motion_name}' was not found under {opt.motion_dir}."
                )
            all_motion_files = [self.fixed_motion_name]
        elif allowed_motion_names is not None:
            all_motion_files = [name for name in all_motion_files if name in allowed_motion_names]

        for object_type in all_object_types:
            object_motions = [name for name in all_motion_files if name.startswith(f'{object_type}_')]
            
            for name in object_motions:
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
                                        'motion_metadata': motion_metadata_lookup.get(name) or infer_motion_labels_from_motion_name(name, object_type=object_type, object_types=all_object_types),
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

        if self.fixed_motion_name:
            fixed_entry = data_dict.get(self.fixed_motion_name)
            if fixed_entry is None:
                raise RuntimeError(
                    f"Fixed motion '{self.fixed_motion_name}' was not loaded into the dataset subset."
                )
            max_valid_start = max(0, int(fixed_entry['length']) - int(self.max_motion_length))
            self.fixed_motion_virtual_length = max_valid_start + 1
            if self.fixed_window_start != -1 and (self.fixed_window_start < 0 or self.fixed_window_start > max_valid_start):
                raise ValueError(
                    f"fixed_window_start={self.fixed_window_start} is invalid for motion '{self.fixed_motion_name}' "
                    f"with length={int(fixed_entry['length'])} and num_frames={int(self.max_motion_length)}. "
                    f"Valid range is [0, {max_valid_start}] or -1 for random cropping."
                )

        minimum_valid_index = np.searchsorted([pair[1] for pair in sorted_pairs], self.max_length)
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
        self.temporal_mask_template = create_temporal_mask_for_window(temporal_window, self.max_motion_length)
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        if self.max_available_length > 0:
            length = min(length, self.max_available_length)
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        self.max_length = length
    
    def inv_transform(self, x, y):
        mean = self.cond_dict[y['object_type']]['mean']
        std = self.cond_dict[y['object_type']]['std']
        return x * std + mean

    def _prepare_sample(self, name, data):
        motion, m_length, object_type, parents, joints_graph_dist, joints_relations, tpos_first_frame, offsets, joints_names_embs, kinematic_chains, mean, std = self.augment(data)
        motion_metadata = dict(data.get('motion_metadata') or infer_motion_labels_from_motion_name(name, object_type=object_type, object_types=self.cond_dict.keys()))
        motion_metadata.setdefault('motion_name', name)
        ind = 0
        if m_length > self.max_motion_length:
            if self.fixed_motion_name and self.fixed_window_start >= 0:
                ind = self.fixed_window_start
            else:
                ind = random.randint(0, m_length - self.max_motion_length)
            motion = motion[ind: ind + self.max_motion_length]
            m_length = self.max_motion_length

        if m_length < self.max_motion_length:
            pad_frames = self.max_motion_length - m_length
            if motion_metadata.get('is_loop') and m_length > 0:
                tiles = (pad_frames // m_length) + 1
                loop_pad = np.tile(motion, (tiles, 1, 1))[:pad_frames]
                motion = np.concatenate([motion, loop_pad], axis=0)
                m_length = self.max_motion_length
            else:
                motion = np.concatenate([
                                         motion,
                                         np.zeros((pad_frames, motion.shape[1], motion.shape[2]), dtype=motion.dtype)
                                         ], axis=0)

        return motion, m_length, parents, tpos_first_frame, offsets, self.temporal_mask_template, joints_graph_dist, joints_relations, object_type, joints_names_embs, ind, mean, std, self.opt.max_joints, motion_metadata, name
    
    def augment(self, data):
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

        speed_range = getattr(self.opt, 'aug_speed_range', 0.0)
        mirror_prob = getattr(self.opt, 'aug_mirror_prob', 0.0)

        if mirror_prob > 0.0 and random.random() < mirror_prob:
            spi = cond.get('symmetry_partner_indices')
            if spi is not None and len(spi) == motion.shape[1]:
                mirrored_cache_key = f'{motion_path}__mirrored_raw'
                if self.motion_cache_size > 0 and mirrored_cache_key in self.motion_cache:
                    motion = self.motion_cache[mirrored_cache_key]
                    self.motion_cache.move_to_end(mirrored_cache_key)
                else:
                    # Mirror in raw feature space. Doing this after normalization is wrong
                    # when mirrored channels have non-zero means.
                    perm = list(range(len(spi)))
                    for i, partner in enumerate(spi):
                        if partner != -1:
                            perm[i] = int(partner)
                    motion = motion[:, perm, :].copy()
                    motion[:, :, [0, 4, 5, 6, 9]] *= -1
                    if self.motion_cache_size > 0:
                        self.motion_cache[mirrored_cache_key] = motion
                        self.motion_cache.move_to_end(mirrored_cache_key)
                        while len(self.motion_cache) > self.motion_cache_size:
                            self.motion_cache.popitem(last=False)

        if speed_range > 0.0:
            # Resample time axis: alpha<1 speeds up (fewer frames), alpha>1 slows down (more frames).
            # The existing random-start-offset logic in _prepare_sample handles length mismatch.
            alpha = 1.0 + random.uniform(-speed_range, speed_range)
            orig_len = motion.shape[0]
            new_len = max(1, int(round(orig_len * alpha)))
            if new_len != orig_len:
                src = np.linspace(0, orig_len - 1, new_len)
                lo = np.floor(src).astype(np.int32).clip(0, orig_len - 1)
                hi = np.minimum(lo + 1, orig_len - 1)
                w = (src - lo)[:, None, None]
                motion = motion[lo] * (1.0 - w) + motion[hi] * w

        motion = np.nan_to_num((motion - cond['mean'][None, :]) / cond['std_safe'][None, :]).astype(np.float32, copy=False)

        m_length = motion.shape[0]
        mean = self.cond_dict[object_type]['mean']
        std = self.cond_dict[object_type]['std_safe']
        return motion, m_length, object_type, cond['parents'], cond['joints_graph_dist'], cond['joint_relations'], cond['tpos_first_frame_normalized'], cond['offsets'], cond['joints_names_embs'], cond['kinematic_chains'], mean, std
        
    def __len__(self):
        if self.fixed_motion_name and self.fixed_window_start == -1 and self.name_list:
            return max(1, int(self.fixed_motion_virtual_length))
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        if self.fixed_motion_name and self.fixed_window_start == -1:
            name = self.name_list[0]
            return self._prepare_sample(name, self.data_dict[name])
        if self.balanced:
            idx = item #self.pointer + item (handled in weighted sampler)
        else:
            idx = self.pointer + item
        name = self.name_list[idx]
        return self._prepare_sample(name, self.data_dict[name])

class TruebonesSampler(WeightedRandomSampler):
    def __init__(self, data_source):
        num_samples = len(data_source)
        object_types = data_source.motion_dataset.cond_dict.keys()
        name_list = data_source.motion_dataset.name_list
        total_samples = len(name_list)
        weights = np.zeros(total_samples)
        object_share = 1.0/len(object_types)
        pointer = data_source.motion_dataset.pointer
        
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
        
        object_share = 1.0 / len(non_empty_types)
        for object_type, object_indices in non_empty_types:
            object_prob = object_share / len(object_indices)
            weights[object_indices] = object_prob
        
        super().__init__(num_samples=num_samples, weights=weights)
    
class Truebones(data.Dataset):
    def __init__(self, split="train", temporal_window=31, t5_name='t5-base', **kwargs):
        if split not in SUPPORTED_SPLITS and split != ALL_SPLIT_NAME:
            raise ValueError(f"Unsupported split '{split}'. Expected one of {SUPPORTED_SPLITS + (ALL_SPLIT_NAME,)}.")
        abs_base_path = f'.'
        device = None  # torch.device('cuda:4') # This param is not in use in this context
        opt = get_opt(device)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.max_motion_length = min(opt.max_motion_length, kwargs['num_frames'])
        self.opt = opt
        self.balanced = kwargs['balanced']
        self.objects_subset = kwargs['objects_subset']
        self.action_tags = kwargs.get('action_tags', '')
        self.sample_limit = kwargs.get('sample_limit', 0)
        self.motion_cache_size = kwargs.get('motion_cache_size', 0)
        self.fixed_motion = kwargs.get('fixed_motion', '')
        self.fixed_window_start = kwargs.get('fixed_window_start', 0)
        if self.fixed_window_start == -1 and not self.fixed_motion:
            raise ValueError('fixed_window_start=-1 (random cropping) requires --fixed_motion.')
        self.opt.motion_cache_size = self.motion_cache_size
        self.opt.fixed_motion = self.fixed_motion
        self.opt.fixed_window_start = self.fixed_window_start
        cond_dict = np.load(opt.cond_file, allow_pickle=True).item()
        # Support both predefined subsets and single species names
        if self.objects_subset in opt.subsets_dict:
            subset = opt.subsets_dict[self.objects_subset]
        else:
            # Treat as a single species name
            subset = [self.objects_subset]
        cond_dict = {k:cond_dict[k] for k in subset if k in cond_dict}
        cond_dict = attach_joint_name_embeddings(cond_dict, opt.cond_file, opt.data_root, t5_name)
        for object_type, cond in cond_dict.items():
            mean = np.asarray(cond['mean'], dtype=np.float32)
            std_safe = np.asarray(cond['std'], dtype=np.float32) + 1e-6
            cond['mean'] = mean
            cond['std'] = np.asarray(cond['std'], dtype=np.float32)
            cond['std_safe'] = std_safe
            cond['tpos_first_frame_normalized'] = np.nan_to_num((np.asarray(cond['tpos_first_frame'], dtype=np.float32) - mean) / std_safe).astype(np.float32, copy=False)
            
        motion_metadata_lookup = load_motion_metadata(opt.data_root)
        self.split_file = pjoin(opt.data_root, f'{split}.txt') if split != ALL_SPLIT_NAME else ''
        allowed_motion_names = load_motion_names_for_split_with_action_tags(
            split,
            opt.data_root,
            opt.motion_dir,
            self.action_tags,
            motion_metadata_lookup,
            cond_dict.keys(),
        )
        self.motion_dataset = MotionDataset(
            self.opt,
            cond_dict,
            temporal_window,
            t5_name,
            self.balanced,
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
