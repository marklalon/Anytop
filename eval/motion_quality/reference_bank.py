from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from data_loaders.truebones.offline_reference_dataset import (
    load_cond_dict,
    resolve_sources,
)
from data_loaders.truebones.truebones_utils.dataset_sources import (
    DatasetSource,
    resolve_species_key,
    species_lookup_map,
    split_canonical_key,
)
from data_loaders.truebones.truebones_utils.motion_labels import (
    head_words_in,
    load_motion_metadata,
    parse_action_label,
    vocab_words_in,
)
from data_loaders.truebones.truebones_utils.param_utils import FEATS_LEN
from utils.misc import infer_object_type_from_filename
from utils.skeleton_similarity import (
    SkeletonProfile,
    SpeciesSimilarity,
    assign_softmax_weights,
    rank_species,
)


# The reference prior is never narrower than this many clips: after the
# top-k species, the next-nearest species are added until the floor is met.
# Top-3 alone often yields 8-10 clips, and the IQR the scorer scales its
# deviations by is unstable at that size.
DEFAULT_MIN_REFERENCE_CLIPS = 12


def reference_prior_words(action_label) -> tuple[str, ...]:
    """The head words an ``action_label`` selects the reference prior by.

    Parsed under the contract generate.py enforces on ``--action_label``, so the
    label a clip was generated with is the label it is scored with, and a typo
    fails instead of silently narrowing the prior. Only the head words (the
    HEAD_VOCAB members: walk, run, idle, attack, ...) select reference clips;
    direction, hands and secondary words do not. A direction word is shared by
    every travelling action, so letting it match made ``walk, forward`` and
    ``run, forward`` select the same bank. The words are returned in vocabulary
    order, so the prior is keyed by the head SET, as the model's head slot is.
    An empty label returns ``()``.

    The prior is deliberately keyed by head words and not by ``action_group``:
    grouping would widen it from "the attack references" to "everything
    stationary", and a prior that loose scores almost anything as plausible.
    """
    parse_action_label(action_label)
    return tuple(head_words_in(vocab_words_in(action_label)))


def _resolve_lookup_key(name: str, lookup: Mapping[str, object]) -> str:
    """Resolve user/metadata-supplied species text to a canonical cond key.

    Shares one resolution rule with the CLI (exact key, then unique namespace
    suffix, then bare name taking the first source), so 'Horse' means the same
    species everywhere.
    """
    resolved = resolve_species_key(lookup, name)
    if resolved is None:
        raise KeyError(f"Unknown key: {name}")
    return resolved


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
    tag_distance: float
    species_weight: float
    clip_count: int
    total_frames: int
    jaccard: float = 0.0
    topology_distance: float = 0.0
    combined_distance: float = 0.0
    same_tags: bool = False


@dataclass(frozen=True)
class WeightedReferenceBank:
    dataset_root: str
    object_type: str
    action_label: str             # the prior's head words, comma-joined (not a request label)
    top_k_species: int
    clips: List[ReferenceClip]
    species: List[ReferenceSpeciesSummary]
    # Scorer-side memo of per-clip reference features keyed by nperseg. The
    # bank is otherwise read-only; the dict is the one thing that grows.
    feature_cache: dict = field(default_factory=dict, compare=False, repr=False)

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


class ReferenceCorpus:
    """The dataset side of the reference prior, indexed once per source set.

    Assembling a bank used to re-read every source's ``motion_metadata.json``,
    re-load the cond and re-resolve every clip's species on each call. The
    corpus does that once: it keeps the merged cond, a ``{species: {head
    word: [clip paths]}}`` index over every source's ``motions/`` dir, the
    similarity profile of each species, the clips already loaded from disk,
    and the banks already assembled, so a bank costs a dictionary filter plus
    the ranking, and a clip is read from disk once no matter how many banks
    share it.
    """

    def __init__(self, dataset_root=None, cond_lookup: Optional[Mapping[str, Mapping[str, object]]] = None):
        self.sources: tuple[DatasetSource, ...] = tuple(resolve_sources(dataset_root))
        self.cond_lookup: Dict[str, Mapping[str, object]] = (
            dict(cond_lookup) if cond_lookup is not None else load_cond_dict(self.sources)
        )
        self._clips_by_species: Dict[str, Dict[str, List[str]]] = self._index_clips()
        self._profiles: Dict[str, SkeletonProfile] = {}
        self._motions: Dict[str, np.ndarray] = {}
        self._banks: Dict[tuple, WeightedReferenceBank] = {}

    # ── Index ────────────────────────────────────────────────────────────────
    def _index_clips(self) -> Dict[str, Dict[str, List[str]]]:
        """``{canonical species key: {head word: [absolute clip paths]}}``.

        Paths are absolute, so the same bare filename appearing in two datasets
        stays two distinct clips. Species membership is resolved inside the
        owning source, so a bare 'Horse' from one dataset's metadata never binds
        to the other's.
        """
        grouped: Dict[str, Dict[str, List[str]]] = {}
        for source in self.sources:
            motion_dir = Path(source.motion_dir)
            metadata_lookup = load_motion_metadata(source.root)
            source_lookup = {
                key: entry for key, entry in self.cond_lookup.items()
                if str(entry.get("dataset_namespace")) == source.namespace
            }
            if not source_lookup:
                continue
            filename_lookup = species_lookup_map(source_lookup)
            species_keys: Dict[str, Optional[str]] = {}

            for path in sorted(motion_dir.glob("*.npy")):
                metadata = metadata_lookup.get(path.name)
                if metadata is None:
                    # No metadata means no action label, so the clip can never
                    # match a head word -- skip it rather than fabricate a label.
                    continue
                heads = head_words_in(vocab_words_in(str(metadata.get("action_label") or "")))
                if not heads:
                    continue

                species = str(metadata.get("object_type") or "").strip()
                if species:
                    if species not in species_keys:
                        species_keys[species] = _resolve_lookup_key(species, source_lookup)
                    object_type = species_keys[species]
                else:
                    object_type = infer_object_type_from_filename(
                        path.name, valid_types=filename_lookup
                    )
                    if object_type is None:
                        continue
                by_head = grouped.setdefault(object_type, {})
                for head in heads:
                    by_head.setdefault(head, []).append(str(path))
        return grouped

    def clip_paths(self, prior_words: Sequence[str]) -> Dict[str, List[str]]:
        """``{species: [paths]}`` of every clip whose label hits any prior word.

        A clip labelled with two of the words is listed once.
        """
        requested = set(prior_words)
        matched: Dict[str, List[str]] = {}
        for species, by_head in self._clips_by_species.items():
            seen: Dict[str, None] = {}
            for head, paths in by_head.items():
                if head in requested:
                    for path in paths:
                        seen.setdefault(path, None)
            if seen:
                matched[species] = list(seen)
        return matched

    def _load_motion(self, path: str) -> np.ndarray:
        motion = self._motions.get(path)
        if motion is None:
            motion = np.load(path)
            if motion.ndim == 3 and motion.shape[-1] == FEATS_LEN:
                motion = motion.astype(np.float32)
            self._motions[path] = motion
        return motion

    def forget_query(self, object_key: str) -> None:
        """Drop banks built for a (re-)registered custom query skeleton."""
        for key in [k for k in self._banks if k[0] == object_key]:
            del self._banks[key]

    # ── Species selection ────────────────────────────────────────────────────
    def select_species(
        self,
        query_object_type: str,
        prior_words: Sequence[str],
        paths_by_species: Mapping[str, Sequence[str]],
        top_k_species: int,
        min_reference_clips: int,
        query_cond: Optional[Mapping[str, object]] = None,
    ) -> List[SpeciesSimilarity]:
        """Nearest species with clips for ``prior_words``, softmax-weighted.

        The ``top_k_species`` nearest are always taken; the next-nearest are
        appended while the selection holds fewer than ``min_reference_clips``
        clips. Weights are the softmax over the final selection.
        """
        if top_k_species <= 0:
            raise ValueError("top_k_species must be >= 1")
        if query_cond is None:
            query_key = _resolve_lookup_key(query_object_type, self.cond_lookup)
            query_cond = self.cond_lookup[query_key]

        candidate_conds = {
            object_type: self.cond_lookup[object_type]
            for object_type, paths in paths_by_species.items()
            if paths
        }
        if not candidate_conds:
            raise ValueError(
                f"No dataset reference motions found for action words {list(prior_words)!r}"
            )

        ranked = rank_species(
            query_cond,
            candidate_conds,
            query_hint=query_object_type,
            top_k=None,
            profiles=self._profiles,
        )
        selected: List[SpeciesSimilarity] = []
        clip_total = 0
        for entry in ranked:
            if len(selected) >= top_k_species and clip_total >= min_reference_clips:
                break
            selected.append(entry)
            clip_total += len(paths_by_species[entry.name])
        assign_softmax_weights(selected)
        return selected

    # ── Bank assembly ────────────────────────────────────────────────────────
    def build_bank(
        self,
        object_type: str,
        action_label: str,
        top_k_species: int = 5,
        min_reference_clips: int = DEFAULT_MIN_REFERENCE_CLIPS,
        min_frames: int = 8,
        query_cond: Optional[Mapping[str, object]] = None,
    ) -> WeightedReferenceBank:
        """The weighted reference prior for one (species, action label) pair.

        Memoised on the resolved species key plus the normalised request -- the
        label's head words, so two spellings of the same words share one bank.
        The returned bank is treated as read-only by all callers. ``query_cond``
        stands in for a skeleton that is not in the corpus (a custom retarget
        target); ``object_type`` is then its registered key, and the caller
        owns invalidation through :meth:`forget_query`.
        """
        action_label_str = str(action_label or "").strip()
        prior_words = reference_prior_words(action_label_str)
        if not prior_words:
            raise ValueError(
                "action_label names no head word, so there is no reference prior "
                "to score against (pass the label the clips were generated with, "
                "e.g. 'run' or 'fly, forward')"
            )
        object_key = (
            str(object_type) if query_cond is not None
            else _resolve_lookup_key(object_type, self.cond_lookup)
        )
        cache_key = (object_key, prior_words, int(top_k_species), int(min_reference_clips), int(min_frames))
        cached = self._banks.get(cache_key)
        if cached is not None:
            return cached

        paths_by_species = self.clip_paths(prior_words)
        selected_species = self.select_species(
            object_key,
            prior_words,
            paths_by_species,
            top_k_species,
            min_reference_clips,
            query_cond=query_cond,
        )

        clips: List[ReferenceClip] = []
        species_summaries: List[ReferenceSpeciesSummary] = []
        for ranked in selected_species:
            species_name = ranked.name
            loaded: List[tuple[str, np.ndarray]] = []
            total_frames = 0
            for path in paths_by_species.get(species_name, []):
                motion = self._load_motion(path)
                if motion.ndim != 3 or motion.shape[-1] != FEATS_LEN or motion.shape[0] < min_frames:
                    continue
                loaded.append((path, motion))
                total_frames += int(motion.shape[0])
            if not loaded or total_frames <= 0:
                continue

            clip_namespace = split_canonical_key(species_name)[0]
            for path, motion in loaded:
                # Composite clip id: the bare filename repeats across datasets.
                motion_name = f"{clip_namespace}/{Path(path).name}" if clip_namespace else Path(path).name
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
                    tag_distance=float(ranked.tag_distance),
                    species_weight=float(ranked.weight),
                    clip_count=len(loaded),
                    total_frames=int(total_frames),
                    jaccard=float(ranked.jaccard),
                    topology_distance=float(ranked.topology_distance),
                    combined_distance=float(ranked.combined_distance),
                    same_tags=bool(ranked.same_tags),
                )
            )

        if not clips:
            raise ValueError(
                f"No valid reference motions found for object_type={object_key!r}, action_label={action_label_str!r}"
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
                    tag_distance=species.tag_distance,
                    species_weight=float(species.species_weight / total_weight),
                    clip_count=species.clip_count,
                    total_frames=species.total_frames,
                    jaccard=species.jaccard,
                    topology_distance=species.topology_distance,
                    combined_distance=species.combined_distance,
                    same_tags=species.same_tags,
                )
                for species in species_summaries
            ]

        species_summaries.sort(key=lambda item: (-item.species_weight, item.combined_distance, item.object_type))
        bank = WeightedReferenceBank(
            dataset_root=os.pathsep.join(source.root for source in self.sources),
            object_type=object_key,
            action_label=", ".join(prior_words),
            top_k_species=int(top_k_species),
            clips=clips,
            species=species_summaries,
        )
        self._banks[cache_key] = bank
        return bank
