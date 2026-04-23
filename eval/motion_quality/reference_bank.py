from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from data_loaders.truebones.offline_reference_dataset import get_motion_dir, load_cond_dict, resolve_dataset_root
from data_loaders.truebones.truebones_utils.motion_labels import (
    infer_motion_labels_from_motion_name,
    load_motion_metadata,
)


def _l2_normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= eps:
        return vector.copy()
    return vector / norm


def _resolve_lookup_key(name: str, lookup: Mapping[str, object]) -> str:
    if name in lookup:
        return name
    lowered = str(name).strip().lower()
    for key in lookup:
        if str(key).lower() == lowered:
            return str(key)
    raise KeyError(f"Unknown key: {name}")


@dataclass(frozen=True)
class ReferenceClip:
    path: str
    object_type: str
    motion_name: str
    n_frames: int
    weight: float
    motion: np.ndarray


@dataclass(frozen=True)
class ReferenceSpeciesSummary:
    object_type: str
    cosine_distance: float
    species_weight: float
    clip_count: int
    total_frames: int


@dataclass(frozen=True)
class WeightedReferenceBank:
    dataset_root: str
    object_type: str
    action_tags: str
    top_k_species: int
    clips: List[ReferenceClip]
    species: List[ReferenceSpeciesSummary]

    @property
    def clip_weights(self) -> np.ndarray:
        return np.asarray([clip.weight for clip in self.clips], dtype=np.float64)

    @property
    def total_reference_frames(self) -> int:
        return int(sum(clip.n_frames for clip in self.clips))

    @property
    def effective_reference_mass(self) -> float:
        weights = self.clip_weights
        if weights.size == 0:
            return 0.0
        denom = float(np.sum(weights * weights))
        if denom <= 0.0:
            return 0.0
        return float(1.0 / denom)


def _embedding_for_object(cond: Mapping[str, object]) -> np.ndarray:
    joint_embs = np.asarray(cond.get("joints_names_embs"), dtype=np.float64)
    if joint_embs.ndim != 2 or joint_embs.shape[0] == 0 or joint_embs.shape[1] == 0:
        raise ValueError("cond entry is missing valid joints_names_embs")
    return _l2_normalize(joint_embs.mean(axis=0))


def _normalize_motion_action_tags(raw_action_tags) -> set[str]:
    """Normalize motion action tags to a set of lowercase strings."""
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


def _collect_action_tags_paths(
    dataset_root: Path,
    cond_lookup: Mapping[str, Mapping[str, object]],
    action_tags: str,
) -> Dict[str, List[str]]:
    """Collect motion paths grouped by object type, filtered by action_tags.
    
    Args:
        dataset_root: Path to dataset root
        cond_lookup: Mapping of object types to cond data
        action_tags: Comma/semicolon-separated action tags to filter by
    
    Returns:
        Dict mapping object_type to list of motion paths that match any of the action_tags
    """
    motion_dir = get_motion_dir(dataset_root)
    metadata_lookup = load_motion_metadata(dataset_root)
    object_types = tuple(cond_lookup.keys())
    grouped: Dict[str, List[str]] = {}
    
    # Parse requested action tags
    # Support both comma and semicolon separation
    if action_tags is None:
        requested_tags = set()
    else:
        tokens = str(action_tags).replace(';', ',').split(',')
        requested_tags = {
            token.strip().lower() 
            for token in tokens 
            if str(token).strip()
        }
    
    if not requested_tags:
        raise ValueError("action_tags must contain at least one tag")

    for path in sorted(motion_dir.glob("*.npy")):
        motion_name = path.name
        metadata = metadata_lookup.get(motion_name)
        if metadata is None:
            metadata = infer_motion_labels_from_motion_name(motion_name, object_types=object_types)
        
        # Get motion's action tags and normalize to set
        motion_action_tags = _normalize_motion_action_tags(metadata.get("action_tags"))
        
        # Check if motion has any of the requested tags
        if not motion_action_tags.intersection(requested_tags):
            continue
        
        object_type = str(metadata.get("object_type") or "").strip()
        if not object_type:
            object_type = infer_motion_labels_from_motion_name(motion_name, object_types=object_types)["object_type"]
        object_type = _resolve_lookup_key(object_type, cond_lookup)
        grouped.setdefault(object_type, []).append(str(path))

    return grouped


def _select_species_weights(
    query_object_type: str,
    action_tags: str,
    action_paths_by_species: Mapping[str, Sequence[str]],
    cond_lookup: Mapping[str, Mapping[str, object]],
    top_k_species: int,
) -> List[tuple[str, float, float]]:
    if top_k_species <= 0:
        raise ValueError("top_k_species must be >= 1")
    query_key = _resolve_lookup_key(query_object_type, cond_lookup)
    query_emb = _embedding_for_object(cond_lookup[query_key])

    candidates: List[tuple[str, float]] = []
    for object_type, paths in action_paths_by_species.items():
        if not paths:
            continue
        ref_emb = _embedding_for_object(cond_lookup[object_type])
        cosine_similarity = float(np.clip(np.dot(query_emb, ref_emb), -1.0, 1.0))
        cosine_distance = float(max(0.0, 1.0 - cosine_similarity))
        candidates.append((object_type, cosine_distance))

    if not candidates:
        raise ValueError(
            f"No dataset reference motions found for action_tags={action_tags!r}"
        )

    candidates.sort(key=lambda item: (item[1], item[0]))
    selected = candidates[: min(top_k_species, len(candidates))]
    if len(selected) == 1:
        return [(selected[0][0], selected[0][1], 1.0)]

    distances = np.asarray([item[1] for item in selected], dtype=np.float64)
    positive = distances[distances > 1e-8]
    temperature = float(np.median(positive)) if positive.size else 0.03
    temperature = max(temperature, 0.03)
    logits = -distances / temperature
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()
    return [
        (species_name, float(distance), float(weight))
        for (species_name, distance), weight in zip(selected, weights)
    ]


def build_weighted_reference_bank(
    object_type: str,
    action_tags: str,
    dataset_root: Optional[str] = None,
    top_k_species: int = 5,
    min_frames: int = 8,
) -> WeightedReferenceBank:
    dataset_root_path = resolve_dataset_root(dataset_root)
    cond_lookup = load_cond_dict(dataset_root_path)
    object_key = _resolve_lookup_key(object_type, cond_lookup)
    action_tags_str = str(action_tags or "").strip()
    if not action_tags_str:
        raise ValueError("action_tags must be a non-empty string")

    action_paths_by_species = _collect_action_tags_paths(dataset_root_path, cond_lookup, action_tags_str)
    selected_species = _select_species_weights(
        object_key,
        action_tags_str,
        action_paths_by_species,
        cond_lookup,
        top_k_species,
    )

    clips: List[ReferenceClip] = []
    species_summaries: List[ReferenceSpeciesSummary] = []
    for species_name, cosine_distance, species_weight in selected_species:
        candidate_paths = action_paths_by_species.get(species_name, [])
        loaded: List[tuple[str, np.ndarray]] = []
        total_frames = 0
        for path in candidate_paths:
            motion = np.load(path)
            if motion.ndim != 3 or motion.shape[-1] != 13 or motion.shape[0] < min_frames:
                continue
            motion = motion.astype(np.float32)
            loaded.append((path, motion))
            total_frames += int(motion.shape[0])
        if not loaded or total_frames <= 0:
            continue

        for path, motion in loaded:
            motion_name = Path(path).name
            clip_weight = species_weight * (float(motion.shape[0]) / float(total_frames))
            clips.append(
                ReferenceClip(
                    path=path,
                    object_type=species_name,
                    motion_name=motion_name,
                    n_frames=int(motion.shape[0]),
                    weight=float(clip_weight),
                    motion=motion,
                )
            )

        species_summaries.append(
            ReferenceSpeciesSummary(
                object_type=species_name,
                cosine_distance=float(cosine_distance),
                species_weight=float(species_weight),
                clip_count=len(loaded),
                total_frames=int(total_frames),
            )
        )

    if not clips:
        raise ValueError(
            f"No valid reference motions found for object_type={object_key!r}, action_tags={action_tags_str!r}"
        )

    total_weight = float(sum(clip.weight for clip in clips))
    if total_weight <= 0.0:
        raise ValueError("Reference weights collapsed to zero")
    if abs(total_weight - 1.0) > 1e-6:
        clips = [
            ReferenceClip(
                path=clip.path,
                object_type=clip.object_type,
                motion_name=clip.motion_name,
                n_frames=clip.n_frames,
                weight=float(clip.weight / total_weight),
                motion=clip.motion,
            )
            for clip in clips
        ]
        species_summaries = [
            ReferenceSpeciesSummary(
                object_type=species.object_type,
                cosine_distance=species.cosine_distance,
                species_weight=float(species.species_weight / total_weight),
                clip_count=species.clip_count,
                total_frames=species.total_frames,
            )
            for species in species_summaries
        ]

    species_summaries.sort(key=lambda item: (-item.species_weight, item.cosine_distance, item.object_type))
    return WeightedReferenceBank(
        dataset_root=str(dataset_root_path),
        object_type=object_key,
        action_tags=action_tags_str,
        top_k_species=int(top_k_species),
        clips=clips,
        species=species_summaries,
    )