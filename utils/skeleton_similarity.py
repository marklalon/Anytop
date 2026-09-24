"""Shared skeleton-similarity scoring.

A single place to answer "how close is this skeleton, as a *mover*, to that
one?", reused by:

  * eval/motion_quality/reference_bank.py -- pick the reference species a
    generated clip is scored against;
  * utils/clip_length_prior.py -- pool neighbouring species when a species
    has no training clip for the requested action.

Similarity blends three signals (see ``SimilarityWeights``), each a distance
in ``[0, ~1]``:

  * **Motion tags** -- the *primary* term. Every cond entry carries a baked
    ``species_tags`` pair ``(body-plan, gait)`` such as
    ``('Quadruped', 'Galloping')``; the slots are compared one by one and
    weighted (body plan above gait), so a Horse is close to a Deer and a Cavalry
    unit, and no closer to a Chicken than to a Crow.
    Grouping by *how an animal moves* is what the smoothness / spectral
    statistics the scorer compares actually depend on.
  * **Body parts** -- Jaccard over the *slim* joint tokens the joint-name
    embedding schema already normalises every rig to (``Tail``, ``Thigh``,
    ``Wing``, ``Ear HeadFeature`` ...), with the side / front / back qualifiers
    dropped. The slim vocabulary is ~370 tokens across every dataset, so two
    rigs that share body parts overlap even when their raw joint names follow
    different conventions (``Tail 01`` / ``Tail 1`` / ``Right Front Upper
    Leg`` / ``Right Thigh``).
  * **Topology descriptor** -- a permutation- and size-tolerant morphology
    vector (leaf/branch fractions, depth, kinematic-chain length stats, size);
    a weak tie-breaker only, since joint count says little about motion.

The module is intentionally numpy-only so the lightweight motion-quality
scorer does not pull in torch/motion_lib. A cond entry without ``species_tags``
(an unregistered retarget target) gets the maximum tag distance to everything
and is ranked by the two morphological terms alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, MutableMapping, Optional, Sequence

import numpy as np

# The tag sidecar is the fallback for a cond entry that predates baked
# species_tags. dataset_tags is torch-free and reads its sidecars lazily, so
# importing it keeps this module lightweight.
from data_loaders.truebones.truebones_utils.dataset_tags import dataset_tags


# ── Motion tags ──────────────────────────────────────────────────────────────
# Slot weights of the species_tags pair: (body-plan, gait). Body plan decides
# which limbs carry the motion; gait is the dynamics. The last word of the gait
# slot is compared, so the "Chibi" / "Robotic" style prefixes ("Chibi Striding"
# vs "Striding") do not separate species that move the same way.
_TAG_SLOT_WEIGHTS = (0.6, 0.4)
_TAG_ARITY = len(_TAG_SLOT_WEIGHTS)
_GAIT_SLOT = 1
# A cond baked before the size slot was removed carries (body-plan, size, gait);
# its size word is dropped on read so such a checkpoint's cond still ranks.
_LEGACY_SIZE_SLOT_ARITY = 3


def species_tags_of(object_cond: Mapping[str, object], object_type_hint: str) -> tuple[str, ...]:
    """The ``(body-plan, gait)`` pair of a cond entry, lower-cased.

    Reads the entry's baked ``species_tags`` (every cond since schema v4 has
    them, including a custom retarget cond), falling back to the tag sidecar
    by object_type. Returns ``()`` for an unregistered species.
    """
    tags = object_cond.get("species_tags")
    if not tags:
        tags = dataset_tags().tags_for(object_cond.get("object_type") or object_type_hint)
    tags = tuple(str(tag).strip().lower() for tag in (tags or ()))
    if len(tags) == _LEGACY_SIZE_SLOT_ARITY:
        tags = (tags[0], tags[2])
    return tags


def tag_distance(query_tags: Sequence[str], candidate_tags: Sequence[str]) -> float:
    """Weighted slot mismatch of two tag pairs, in ``[0, 1]``.

    A pair that is missing or short (unregistered species) is maximally far
    from everything, so ranking falls back to the morphological terms.
    """
    if len(query_tags) < _TAG_ARITY or len(candidate_tags) < _TAG_ARITY:
        return 1.0
    similarity = 0.0
    for slot, weight in enumerate(_TAG_SLOT_WEIGHTS):
        query_word, candidate_word = query_tags[slot], candidate_tags[slot]
        if slot == _GAIT_SLOT:
            query_word, candidate_word = query_word.split()[-1], candidate_word.split()[-1]
        if query_word == candidate_word:
            similarity += weight
    return float(1.0 - similarity)


# ── Body-part tokens (for Jaccard) ───────────────────────────────────────────
# Qualifiers that say *which* copy of a part a joint is, not *what* it is.
_PART_QUALIFIERS = frozenset({"left", "right", "front", "back", "rear", "fore", "hind"})


def _part_token(text: object) -> str:
    words = [w for w in str(text).lower().split() if w not in _PART_QUALIFIERS]
    # Segment numbers / helper suffixes only appear on raw names (the slim
    # tokens carry none); dropping them makes the raw fallback comparable.
    words = [w for w in words if not w.isdigit() and w != "nub"]
    return " ".join(words)


def part_token_set(object_cond: Mapping[str, object], object_type_hint: str) -> frozenset:
    """Body-part tokens of a skeleton, qualifier-free.

    Prefers the slim texts the joint-name embeddings were encoded from
    (``joints_names_embs_meta.embedding_texts``); falls back to the canonical
    (then raw) joint names with numbers and side words stripped, which is a
    coarser but compatible vocabulary.
    """
    meta = object_cond.get("joints_names_embs_meta") or {}
    texts = meta.get("embedding_texts") if isinstance(meta, Mapping) else None
    if not texts:
        texts = object_cond.get("canonical_joint_names") or object_cond.get("joints_names")
    if not texts:
        raise ValueError(f"No joint names available for {object_type_hint}")
    return frozenset(token for token in (_part_token(t) for t in texts) if token)


def require_canonical_joint_names(
    object_cond: Mapping[str, object],
    *,
    object_type_hint: str,
    joint_count: Optional[int] = None,
) -> list:
    """Canonical joint names for retarget-grade callers (raises if absent)."""
    canonical_joint_names = object_cond.get("canonical_joint_names")
    if canonical_joint_names is None:
        raise ValueError(f"Retarget requires canonical_joint_names for {object_type_hint}")
    canonical_joint_names = list(canonical_joint_names)
    if joint_count is not None and len(canonical_joint_names) < int(joint_count):
        raise ValueError(
            f"Retarget canonical_joint_names for {object_type_hint} has length "
            f"{len(canonical_joint_names)} but joint count requires at least {int(joint_count)}"
        )
    return canonical_joint_names


# ── Skeleton topology descriptor ─────────────────────────────────────────────
_TOPO_FEATURE_DIM = 8


def skeleton_parents(object_cond: Mapping[str, object]) -> np.ndarray:
    """Return the complete parent array stored for the skeleton."""
    return np.asarray(object_cond.get("parents"), dtype=np.int64).reshape(-1)


def node_depths(parents: np.ndarray) -> np.ndarray:
    """Depth of every node from the root (parent < 0), memoised, O(J)."""
    n = parents.size
    depth = np.full(n, -1, dtype=np.int64)
    for start in range(n):
        chain: List[int] = []
        j = start
        while j >= 0 and depth[j] < 0:
            chain.append(j)
            j = int(parents[j])
        base = int(depth[j]) if j >= 0 else -1  # root parent (-1) -> base -1
        for offset, node in enumerate(reversed(chain)):
            depth[node] = base + offset + 1
    return depth


def topology_descriptor(object_cond: Mapping[str, object]) -> np.ndarray:
    """Permutation- and size-tolerant morphology descriptor for a skeleton.

    Features: leaf fraction, branch fraction, root out-degree, max depth,
    mean depth, mean/std kinematic-chain length, log joint count. Mixing counts
    and fractions is fine because callers z-score each feature over the pool
    before computing distances.
    """
    parents = skeleton_parents(object_cond)
    n = parents.size
    if n <= 1:
        return np.zeros(_TOPO_FEATURE_DIM, dtype=np.float64)
    child_count = np.bincount(parents[parents >= 0], minlength=n)[:n]
    leaves = int(np.count_nonzero(child_count == 0))
    branches = int(np.count_nonzero(child_count >= 2))
    root_out = float(child_count[0])
    depths = node_depths(parents)
    chains = object_cond.get("kinematic_chains") or []
    chain_lens = np.asarray([len(ch) for ch in chains], dtype=np.float64)
    mean_chain = float(chain_lens.mean()) if chain_lens.size else 0.0
    std_chain = float(chain_lens.std()) if chain_lens.size else 0.0
    return np.asarray([
        leaves / n,
        branches / n,
        root_out,
        float(depths.max()),
        float(depths.mean()),
        mean_chain,
        std_chain,
        float(np.log(n)),
    ], dtype=np.float64)


# ── Per-skeleton profile (what the ranking reads) ────────────────────────────
@dataclass(frozen=True)
class SkeletonProfile:
    """The three similarity inputs of one skeleton, computed once."""

    tags: tuple[str, ...]
    parts: frozenset
    descriptor: np.ndarray


def skeleton_profile(object_cond: Mapping[str, object], object_type_hint: str) -> SkeletonProfile:
    return SkeletonProfile(
        tags=species_tags_of(object_cond, object_type_hint),
        parts=part_token_set(object_cond, object_type_hint),
        descriptor=topology_descriptor(object_cond),
    )


# ── Combined similarity ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class SimilarityWeights:
    """Relative weights of the three distance terms.

    They need not sum to 1: only their ratios matter. ``tags`` and ``parts``
    are bounded in ``[0, 1]``; the topology distance is pool-normalised to a
    mean of ~1 first so the blend is scale-free.
    """

    tags: float = 0.5
    parts: float = 0.3
    topology: float = 0.2


DEFAULT_WEIGHTS = SimilarityWeights()


@dataclass
class SpeciesSimilarity:
    name: str
    tag_distance: float           # weighted slot mismatch of the species_tags pairs
    jaccard: float                # body-part overlap (1 = identical part set)
    topology_distance: float      # z-scored descriptor euclidean (pool-relative)
    combined_distance: float
    same_tags: bool               # identical motion descriptor (all three slots)
    weight: float = 0.0


def _pool_scale(values: np.ndarray) -> np.ndarray:
    """Normalise a distance vector to pool mean ~1 (scale-free blend term)."""
    mean = float(np.mean(values))
    return values / mean if mean > 1e-12 else np.zeros_like(values)


def assign_softmax_weights(results: Sequence[SpeciesSimilarity]) -> None:
    """Set ``weight`` on ``results`` in place: softmax of ``-distance / T``.

    The temperature is the median positive distance of the set (floored), so
    the weights adapt to how spread out the selected neighbours are. Called by
    :func:`rank_species` over its selection and again by callers that widen or
    narrow that selection afterwards.
    """
    if not results:
        return
    if len(results) == 1:
        results[0].weight = 1.0
        return
    distances = np.asarray([r.combined_distance for r in results], dtype=np.float64)
    positive = distances[distances > 1e-8]
    temperature = max(float(np.median(positive)) if positive.size else 0.03, 0.03)
    logits = -distances / temperature
    logits -= logits.max()
    softmax = np.exp(logits)
    softmax /= softmax.sum()
    for result, weight in zip(results, softmax):
        result.weight = float(weight)


def rank_species(
    query_cond: Mapping[str, object],
    candidate_conds: Mapping[str, Mapping[str, object]],
    *,
    query_hint: str,
    top_k: Optional[int] = None,
    weights: SimilarityWeights = DEFAULT_WEIGHTS,
    profiles: Optional[MutableMapping[str, SkeletonProfile]] = None,
) -> List[SpeciesSimilarity]:
    """Rank candidate skeletons by similarity to ``query_cond`` (closest first).

    Returns one ``SpeciesSimilarity`` per selected candidate, sorted by ascending
    ``combined_distance``, with softmax ``weight`` over the selected set. ``top_k``
    None ranks every candidate. ``profiles`` is an optional memo of candidate
    profiles keyed by name, filled in as candidates are first seen, for callers
    that rank against the same pool repeatedly.
    """
    names = list(candidate_conds.keys())
    if not names:
        raise ValueError("No candidate skeletons to rank")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be >= 1 or None")

    query = skeleton_profile(query_cond, query_hint)

    tag_arr = np.zeros(len(names), dtype=np.float64)
    jaccard_arr = np.zeros(len(names), dtype=np.float64)
    descriptors: List[np.ndarray] = []
    same_tags = np.zeros(len(names), dtype=bool)
    for i, name in enumerate(names):
        profile = profiles.get(name) if profiles is not None else None
        if profile is None:
            profile = skeleton_profile(candidate_conds[name], name)
            if profiles is not None:
                profiles[name] = profile
        tag_arr[i] = tag_distance(query.tags, profile.tags)
        same_tags[i] = bool(query.tags) and query.tags == profile.tags
        union = len(query.parts | profile.parts)
        jaccard_arr[i] = (len(query.parts & profile.parts) / union) if union else 0.0
        descriptors.append(profile.descriptor)

    # Topology distance: z-score each feature over {query + candidates}, then
    # Euclidean distance in that standardised, scale-free space.
    descriptor_arr = np.asarray(descriptors, dtype=np.float64)
    stacked = np.vstack([query.descriptor[None, :], descriptor_arr])
    feature_std = stacked.std(axis=0)
    feature_std = np.where(feature_std > 1e-9, feature_std, 1.0)
    feature_mean = stacked.mean(axis=0)
    query_z = (query.descriptor - feature_mean) / feature_std
    candidate_z = (descriptor_arr - feature_mean) / feature_std
    topology_distance = np.linalg.norm(candidate_z - query_z[None, :], axis=1)

    total_weight = (weights.tags + weights.parts + weights.topology) or 1.0
    combined = (
        weights.tags * tag_arr
        + weights.parts * (1.0 - jaccard_arr)
        + weights.topology * _pool_scale(topology_distance)
    ) / total_weight

    order = sorted(range(len(names)), key=lambda i: (combined[i], names[i]))
    selected = order if top_k is None else order[: min(top_k, len(order))]

    results = [
        SpeciesSimilarity(
            name=names[i],
            tag_distance=float(tag_arr[i]),
            jaccard=float(jaccard_arr[i]),
            topology_distance=float(topology_distance[i]),
            combined_distance=float(combined[i]),
            same_tags=bool(same_tags[i]),
        )
        for i in selected
    ]
    assign_softmax_weights(results)
    return results
