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
from utils.skeleton_similarity import SpeciesSimilarity, rank_species


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
    topology_distance: float = 0.0
    combined_distance: float = 0.0
    same_species_group: bool = False


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
    query_cond: Optional[Mapping[str, object]] = None,
) -> List[SpeciesSimilarity]:
    if top_k_species <= 0:
        raise ValueError("top_k_species must be >= 1")
    if query_cond is None:
        query_key = _resolve_lookup_key(query_object_type, cond_lookup)
        query_cond_obj: Mapping[str, object] = cond_lookup[query_key]
    else:
        query_cond_obj = query_cond

    candidate_conds = {
        object_type: cond_lookup[object_type]
        for object_type, paths in action_paths_by_species.items()
        if paths
    }
    if not candidate_conds:
        raise ValueError(
            f"No dataset reference motions found for action_tags={action_tags!r}"
        )

    return rank_species(
        query_cond_obj,
        candidate_conds,
        query_hint=query_object_type,
        top_k=top_k_species,
    )


_REFERENCE_BANK_CACHE: Dict[tuple, WeightedReferenceBank] = {}


def clear_reference_bank_cache() -> None:
    """Drop all memoized reference banks (frees the loaded clip arrays)."""
    _REFERENCE_BANK_CACHE.clear()


def build_weighted_reference_bank(
    object_type: str,
    action_tags: str,
    dataset_root: Optional[str] = None,
    top_k_species: int = 5,
    min_frames: int = 8,
    use_cache: bool = True,
    cond_lookup: Optional[Mapping[str, Mapping[str, object]]] = None,
    query_cond: Optional[Mapping[str, object]] = None,
) -> WeightedReferenceBank:
    """Build (or fetch from cache) the weighted reference prior.

    Assembling the bank loads every matching reference clip from disk, which is
    the dominant cost when scoring many query clips that share the same
    (object_type, action_tags, top_k_species) prior. The result is memoized on
    the resolved dataset root plus the normalized request so repeated calls
    reuse the already-loaded clips. The returned bank is treated as read-only by
    all callers; do not mutate its clips in place.
    """
    if cond_lookup is not None or query_cond is not None or not use_cache:
        return _build_weighted_reference_bank(
            object_type,
            action_tags,
            dataset_root,
            top_k_species,
            min_frames,
            cond_lookup=cond_lookup,
            query_cond=query_cond,
        )

    dataset_root_key = str(resolve_dataset_root(dataset_root))
    action_tags_key = frozenset(
        token.strip().lower()
        for token in str(action_tags or "").replace(";", ",").split(",")
        if token.strip()
    )
    cache_key = (
        dataset_root_key,
        str(object_type).strip().lower(),
        action_tags_key,
        int(top_k_species),
        int(min_frames),
    )
    cached = _REFERENCE_BANK_CACHE.get(cache_key)
    if cached is not None:
        return cached

    bank = _build_weighted_reference_bank(
        object_type, action_tags, dataset_root, top_k_species, min_frames
    )
    _REFERENCE_BANK_CACHE[cache_key] = bank
    return bank


def _build_weighted_reference_bank(
    object_type: str,
    action_tags: str,
    dataset_root: Optional[str] = None,
    top_k_species: int = 5,
    min_frames: int = 8,
    cond_lookup: Optional[Mapping[str, Mapping[str, object]]] = None,
    query_cond: Optional[Mapping[str, object]] = None,
) -> WeightedReferenceBank:
    dataset_root_path = resolve_dataset_root(dataset_root)
    if cond_lookup is None:
        cond_lookup = load_cond_dict(dataset_root_path)
    object_key = str(object_type) if query_cond is not None else _resolve_lookup_key(object_type, cond_lookup)
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
        query_cond=query_cond,
    )

    clips: List[ReferenceClip] = []
    species_summaries: List[ReferenceSpeciesSummary] = []
    for ranked in selected_species:
        species_name = ranked.name
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
            clip_weight = ranked.weight * (float(motion.shape[0]) / float(total_frames))
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
                cosine_distance=float(ranked.semantic_distance),
                species_weight=float(ranked.weight),
                clip_count=len(loaded),
                total_frames=int(total_frames),
                topology_distance=float(ranked.topology_distance),
                combined_distance=float(ranked.combined_distance),
                same_species_group=bool(ranked.same_group),
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
                topology_distance=species.topology_distance,
                combined_distance=species.combined_distance,
                same_species_group=species.same_species_group,
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