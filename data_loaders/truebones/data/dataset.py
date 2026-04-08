import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import os
import re
from collections import defaultdict
from os.path import join as pjoin
from pathlib import Path
import random
from typing import Optional
from torch.utils.data._utils.collate import default_collate
from data_loaders.truebones.truebones_utils.get_opt import get_opt
from data_loaders.truebones.truebones_utils.motion_process import remove_joints_augmentation, add_joint_augmentation
from data_loaders.truebones.offline_reference_dataset import load_corrupted_reference_sample
from model.conditioners import T5Conditioner


DEFAULT_SPLIT_RATIOS = {"train": 0.9, "val": 0.05, "test": 0.05}
DEFAULT_SPLIT_SEED = 3407
SUPPORTED_SPLITS = tuple(DEFAULT_SPLIT_RATIOS.keys())


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
    split_paths = ensure_split_manifests(data_root, motion_dir)
    split_path = split_paths[split]
    motion_names = {
        line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()
    }
    if not motion_names:
        raise RuntimeError(f"Split '{split}' is empty: {split_path}")
    return motion_names


def _sanitize_cache_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return sanitized.strip("._") or "default"


def _joint_name_embedding_cache_path(data_root: str, t5_name: str) -> Path:
    cache_dir = Path(data_root) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"joint_name_t5_{_sanitize_cache_component(t5_name)}.npy"


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
    print(f"Building cached joint-name embeddings with {t5_name} on CPU")
    t5_conditioner = T5Conditioner(
        name=t5_name,
        finetune=False,
        word_dropout=0.0,
        normalize_text=False,
        device='cpu',
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
    def __init__(self, opt, cond_dict, temporal_window, t5_name, balanced, sample_limit=0, allowed_motion_names: Optional[set[str]] = None):
        self.opt = opt
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        self.cond_dict = cond_dict
        self.balanced = balanced
        self.sample_limit = max(0, int(sample_limit))
        data_dict = {}
        all_object_types = self.cond_dict.keys()
        new_name_list = []
        length_list = []

        for object_type in all_object_types:
            object_motions = [f for f in os.listdir(opt.motion_dir) if f.startswith(f'{object_type}_')]
            if allowed_motion_names is not None:
                object_motions = [name for name in object_motions if name in allowed_motion_names]
            
            for name in object_motions:
                try:
                    motion_path = pjoin(opt.motion_dir, name)
                    motion = np.load(motion_path, mmap_mode='r')
                    data_dict[name] = {
                                        'motion_path': motion_path,
                                        'length': len(motion),
                                        'object_type': object_type,
                                       }
                                       
                    new_name_list.append(name)
                    length_list.append(len(motion))
                except Exception:
                    pass
                
        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        if self.sample_limit > 0:
            name_list = name_list[:self.sample_limit]
            length_list = length_list[:self.sample_limit]
            data_dict = {name: data_dict[name] for name in name_list}
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.temporal_mask_template = create_temporal_mask_for_window(temporal_window, self.max_motion_length)
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        self.max_length = length
    
    def inv_transform(self, x, y):
        mean = self.cond_dict[y['object_type']]['mean']
        std = self.cond_dict[y['object_type']]['std']
        return x * std + mean

    def _prepare_sample(self, name, data):
        motion, m_length, object_type, parents, joints_graph_dist, joints_relations, tpos_first_frame, offsets, joints_names_embs, kinematic_chains, mean, std = self.augment(data)
        std = std + 1e-6
        motion = (motion - mean[None, :]) / std[None, :]
        motion = np.nan_to_num(motion)
        ind = 0
        tpos_first_frame = (tpos_first_frame - mean) / std
        tpos_first_frame = np.nan_to_num(tpos_first_frame)
        if m_length > self.max_motion_length:
            ind = random.randint(0, m_length - self.max_motion_length)
            motion = motion[ind: ind + self.max_motion_length]
            m_length = self.max_motion_length

        stored_sample = load_corrupted_reference_sample(name, dataset_dir=self.opt.data_root)
        reference_motion = np.nan_to_num((stored_sample['reference_motion'] - mean[None, :]) / std[None, :]).astype(np.float32)
        soft_confidence_mask = stored_sample['soft_confidence_mask'].astype(np.float32)
        corruption_metadata = stored_sample['metadata'].get('corruption_metadata')

        if reference_motion.shape[0] > m_length:
            reference_motion = reference_motion[ind: ind + m_length]
        if soft_confidence_mask.shape[0] > m_length:
            soft_confidence_mask = soft_confidence_mask[ind: ind + m_length]

        if m_length < self.max_motion_length:
            pad_frames = self.max_motion_length - m_length
            motion = np.concatenate([
                                     motion,
                                     np.zeros((pad_frames, motion.shape[1], motion.shape[2]), dtype=motion.dtype)
                                     ], axis=0)
            reference_motion = np.concatenate([
                                               reference_motion,
                                               np.zeros((pad_frames, reference_motion.shape[1], reference_motion.shape[2]), dtype=reference_motion.dtype)
                                               ], axis=0)
            soft_confidence_mask = np.concatenate([
                                                   soft_confidence_mask,
                                                   np.zeros((pad_frames, soft_confidence_mask.shape[1], soft_confidence_mask.shape[2]), dtype=soft_confidence_mask.dtype)
                                                   ], axis=0)

        return motion, m_length, parents, tpos_first_frame, offsets, self.temporal_mask_template, joints_graph_dist, joints_relations, object_type, joints_names_embs, ind, mean, std, self.opt.max_joints, reference_motion, soft_confidence_mask, corruption_metadata, name
    
    def augment(self, data):
        object_type = data['object_type']
        cond = self.cond_dict[object_type]
        motion = np.load(data['motion_path'])
        mean = self.cond_dict[object_type]['mean']
        std = self.cond_dict[object_type]['std']
        return motion, data['length'], object_type, cond['parents'], cond['joints_graph_dist'], cond['joint_relations'], cond['tpos_first_frame'], cond['offsets'], cond['joints_names_embs'], cond['kinematic_chains'], mean, std
        
    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
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
        if split not in SUPPORTED_SPLITS:
            raise ValueError(f"Unsupported split '{split}'. Expected one of {SUPPORTED_SPLITS}.")
        abs_base_path = f'.'
        device = None  # torch.device('cuda:4') # This param is not in use in this context
        opt = get_opt(device)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.max_motion_length = min(opt.max_motion_length, kwargs['num_frames'])
        self.opt = opt
        self.balanced = kwargs['balanced']
        self.objects_subset = kwargs['objects_subset']
        self.sample_limit = kwargs.get('sample_limit', 0)
        cond_dict = np.load(opt.cond_file, allow_pickle=True).item()
        subset = opt.subsets_dict[self.objects_subset] 
        cond_dict = {k:cond_dict[k] for k in subset if k in cond_dict}
        cond_dict = attach_joint_name_embeddings(cond_dict, opt.cond_file, opt.data_root, t5_name)
            
        self.split_file = pjoin(opt.data_root, f'{split}.txt')
        allowed_motion_names = load_motion_names_for_split(split, opt.data_root, opt.motion_dir)
        self.motion_dataset = MotionDataset(
            self.opt,
            cond_dict,
            temporal_window,
            t5_name,
            self.balanced,
            sample_limit=self.sample_limit,
            allowed_motion_names=allowed_motion_names,
        )
        assert len(self.motion_dataset) > 0, 'You loaded an empty dataset, ' \
                                          'it is probably because your data dir has only texts and no motions.\n' \
                                          'To train and evaluate MDM you should get the FULL data as described ' \
                                          'in the README file.'

    def __getitem__(self, item):
        return self.motion_dataset.__getitem__(item)

    def __len__(self):
        return self.motion_dataset.__len__()
