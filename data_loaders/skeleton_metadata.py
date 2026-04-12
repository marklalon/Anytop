from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from data_loaders.truebones.offline_reference_dataset import load_cond_dict, resolve_dataset_root
from data_loaders.truebones.truebones_utils.motion_labels import infer_species_label, load_motion_metadata


@dataclass(frozen=True)
class SkeletonMetadata:
    object_type: str
    parents: tuple[int, ...]
    end_effector_joints: tuple[int, ...]
    contact_joints: tuple[int, ...]
    symmetry_partner_indices: tuple[int, ...]
    symmetric_joint_pairs: tuple[tuple[int, int], ...]
    is_symmetric: bool
    n_joints: int
    joint_depths: tuple[int, ...]
    canonical_up_axis: str = "y"
    non_root_joints: tuple[int, ...] = tuple()
    edge_child_indices: tuple[int, ...] = tuple()
    edge_parent_indices: tuple[int, ...] = tuple()
    symmetry_left_indices: tuple[int, ...] = tuple()
    symmetry_right_indices: tuple[int, ...] = tuple()
    subtree_indices: tuple[tuple[int, ...], ...] = tuple()
    max_joint_depth: int = 1


@dataclass(frozen=True)
class LabelVocab:
    labels: tuple[str, ...]
    _mapping: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_mapping", {label: index for index, label in enumerate(self.labels)})

    @property
    def size(self) -> int:
        return len(self.labels)

    def to_dict(self) -> dict[str, int]:
        return dict(self._mapping)

    def encode_many(self, values: Sequence[str], *, device: torch.device) -> torch.Tensor:
        unknown_index = self._mapping.get("unknown", 0)
        encoded = [self._mapping.get(_normalize_label(value), unknown_index) for value in values]
        return torch.as_tensor(encoded, dtype=torch.long, device=device)


def _normalize_label(value: object) -> str:
    text = str(value or "").strip().lower()
    return text or "unknown"


def _to_int_tuple(values: Iterable[object] | None) -> tuple[int, ...]:
    if values is None:
        return tuple()
    return tuple(int(value) for value in values)


def _compute_joint_depths(parents: Sequence[int]) -> tuple[int, ...]:
    depths = [0] * len(parents)
    for joint_index, parent_index in enumerate(parents):
        depth = 0
        current_parent = int(parent_index)
        while current_parent >= 0:
            depth += 1
            current_parent = int(parents[current_parent])
        depths[joint_index] = depth
    return tuple(depths)


def _canonical_pair_list(values: Iterable[Iterable[object]] | None) -> tuple[tuple[int, int], ...]:
    if values is None:
        return tuple()
    pairs = []
    for value in values:
        pair = tuple(int(index) for index in value)
        if len(pair) == 2:
            pairs.append((pair[0], pair[1]))
    return tuple(pairs)


def _edge_pairs(parents: Sequence[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    child_indices: list[int] = []
    parent_indices: list[int] = []
    for child_index, parent_index in enumerate(parents):
        if parent_index < 0:
            continue
        child_indices.append(child_index)
        parent_indices.append(int(parent_index))
    return tuple(child_indices), tuple(parent_indices)


def _subtree_index_lookup(parents: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    child_lists: list[list[int]] = [[] for _ in parents]
    for child_index, parent_index in enumerate(parents):
        if parent_index >= 0:
            child_lists[int(parent_index)].append(child_index)

    subtrees: list[tuple[int, ...]] = []
    for root_index in range(len(parents)):
        stack = [root_index]
        subtree: list[int] = []
        while stack:
            current_index = stack.pop()
            subtree.append(current_index)
            stack.extend(child_lists[current_index])
        subtrees.append(tuple(subtree))
    return tuple(subtrees)


def load_skeleton_metadata(
    dataset_dir: str | Path | None = None,
    *,
    cond_dict: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, SkeletonMetadata]:
    if cond_dict is None:
        cond_dict = load_cond_dict(dataset_dir=dataset_dir)

    metadata: dict[str, SkeletonMetadata] = {}
    for object_type, object_cond in cond_dict.items():
        parents = _to_int_tuple(object_cond.get("parents"))
        symmetric_joint_pairs = _canonical_pair_list(object_cond.get("symmetric_joint_pairs"))
        edge_child_indices, edge_parent_indices = _edge_pairs(parents)
        max_joint_depth = max(_compute_joint_depths(parents) or (1,)) if parents else 1
        joint_depths = _compute_joint_depths(parents)
        metadata[object_type] = SkeletonMetadata(
            object_type=str(object_type),
            parents=parents,
            end_effector_joints=_to_int_tuple(object_cond.get("end_effector_joints")),
            contact_joints=_to_int_tuple(object_cond.get("contact_joints")),
            symmetry_partner_indices=_to_int_tuple(object_cond.get("symmetry_partner_indices")),
            symmetric_joint_pairs=symmetric_joint_pairs,
            is_symmetric=bool(object_cond.get("is_symmetric", False)),
            n_joints=len(parents),
            joint_depths=joint_depths,
            canonical_up_axis=str(object_cond.get("canonical_up_axis", "y") or "y").lower(),
            non_root_joints=tuple(index for index, parent_index in enumerate(parents) if parent_index >= 0),
            edge_child_indices=edge_child_indices,
            edge_parent_indices=edge_parent_indices,
            symmetry_left_indices=tuple(left for left, _ in symmetric_joint_pairs),
            symmetry_right_indices=tuple(right for _, right in symmetric_joint_pairs),
            subtree_indices=_subtree_index_lookup(parents),
            max_joint_depth=max_joint_depth,
        )
    return metadata


def build_label_vocabs(dataset_dir: str | Path | None = None) -> tuple[LabelVocab, LabelVocab]:
    dataset_root = resolve_dataset_root(dataset_dir)
    motion_metadata = load_motion_metadata(dataset_root)
    cond_dict = load_cond_dict(dataset_root)

    species_labels = {"unknown"}
    action_labels = {"unknown"}

    for object_type in cond_dict:
        species_labels.add(_normalize_label(infer_species_label(object_type)))

    for metadata in motion_metadata.values():
        species_labels.add(_normalize_label(metadata.get("species_label")))
        action_labels.add(_normalize_label(metadata.get("action_label")))

    return (
        LabelVocab(tuple(sorted(species_labels))),
        LabelVocab(tuple(sorted(action_labels))),
    )


def metadata_feature_dim(max_joints: int) -> int:
    return int(max_joints) * 5 + 4


def build_metadata_feature_tensor(
    object_types: Sequence[str],
    metadata_lookup: Mapping[str, SkeletonMetadata],
    *,
    max_joints: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    features = []
    for object_type in object_types:
        metadata = metadata_lookup[str(object_type)]
        features.append(build_single_metadata_feature_tensor(metadata, max_joints=max_joints, device=device, dtype=dtype))
    return torch.stack(features, dim=0)


def build_single_metadata_feature_tensor(
    metadata: SkeletonMetadata,
    *,
    max_joints: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    joint_mask = torch.zeros(max_joints, dtype=dtype, device=device)
    joint_mask[: metadata.n_joints] = 1.0

    end_effector_mask = torch.zeros(max_joints, dtype=dtype, device=device)
    if metadata.end_effector_joints:
        end_effector_mask[list(metadata.end_effector_joints)] = 1.0

    contact_mask = torch.zeros(max_joints, dtype=dtype, device=device)
    if metadata.contact_joints:
        contact_mask[list(metadata.contact_joints)] = 1.0

    symmetry_mask = torch.zeros(max_joints, dtype=dtype, device=device)
    symmetry_indices = [index for index, partner in enumerate(metadata.symmetry_partner_indices) if partner >= 0]
    if symmetry_indices:
        symmetry_mask[symmetry_indices] = 1.0

    depth_vector = torch.zeros(max_joints, dtype=dtype, device=device)
    if metadata.n_joints > 0:
        depth_values = torch.as_tensor(metadata.joint_depths, dtype=dtype, device=device) / float(max(metadata.max_joint_depth, 1))
        depth_vector[: metadata.n_joints] = depth_values

    summary = torch.as_tensor(
        [
            metadata.n_joints / max(float(max_joints), 1.0),
            len(metadata.end_effector_joints) / max(float(max_joints), 1.0),
            len(metadata.contact_joints) / max(float(max_joints), 1.0),
            1.0 if metadata.is_symmetric else 0.0,
        ],
        dtype=dtype,
        device=device,
    )
    return torch.cat([joint_mask, end_effector_mask, contact_mask, symmetry_mask, depth_vector, summary], dim=0)


def build_metadata_feature_lookup(
    metadata_lookup: Mapping[str, SkeletonMetadata],
    *,
    max_joints: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    return {
        str(object_type): build_single_metadata_feature_tensor(
            metadata,
            max_joints=max_joints,
            device=device,
            dtype=dtype,
        )
        for object_type, metadata in metadata_lookup.items()
    }
