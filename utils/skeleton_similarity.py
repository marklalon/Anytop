"""Shared skeleton-similarity scoring.

A single place to answer "how morphologically/semantically close are two
skeletons?", reused by:

  * eval/motion_quality/reference_bank.py -- pick reference species to score a
    generated clip against.

Similarity blends three complementary signals (see ``SimilarityWeights``):

  * **Jaccard** over synonym-normalised canonical joint names -- the *primary*
    semantic term. Counts actually-shared body parts (thigh, calf, spine,
    tail, ...); robust to naming-convention differences via the synonym map.
    Far more discriminative than mean-pooled name embeddings, which wash out
    structure (a primate can sit near a dragon in mean-embedding space while
    sharing few real body parts).
  * **Joint-name embedding** cosine -- a *secondary*, graded fallback. Gives
    partial credit to joints absent from the synonym map (e.g. a dragon's
    wings), where Jaccard sees a hard non-match.
  * **Topology descriptor** -- a permutation- and size-tolerant morphology
    vector (leaf/branch fractions, depth, kinematic-chain length stats, size)
    computed from the *biological* skeleton (helper/augmentation leaves and
    padding dropped).

Plus a graded ``group_tags`` discount: each species carries a small motion
descriptor ``(body-plan, gait/dynamics, refinement)`` tag triple (e.g. Cat ->
('Quadruped', 'Agile', 'Stalking')); the more tags two skeletons share, the
larger the fractional discount on their combined distance. Grouping by *how an
animal moves* (rather than phylogeny) matches what a motion model cares about.

The module is intentionally numpy-only so the lightweight motion-quality
scorer does not pull in torch/motion_lib. Components degrade gracefully: a
skeleton missing ``joints_names_embs`` (e.g. a freshly built retarget target)
simply drops the embedding term and the remaining weights are renormalised;
a species absent from the motion-tag table simply gets no group discount.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence

import numpy as np

# Single source of truth for the per-species motion descriptor tags
# ((body-plan, gait/dynamics, refinement) per species). dataset_tags is
# torch-free (json/pathlib/numpy only) and reads its sidecars lazily, so
# importing it keeps this module lightweight.
from data_loaders.truebones.truebones_utils.dataset_tags import dataset_tags
# Every registered species carries this many motion tags; the graded group
# discount normalises overlap by it (full overlap -> full bonus).
_GROUP_TAG_ARITY = 3


# ── Canonical joint-name normalisation (for Jaccard) ─────────────────────────
_CANONICAL_SYNONYMS: dict[str, str] = {
    "leg 1": "thigh", "leg 2": "calf", "leg ankle": "foot", "leg ball 1": "toe 0",
    "arm collarbone": "clavicle", "arm 1": "upper arm", "arm 2": "forearm",
    "arm palm": "hand", "arm ball 1": "wrist",
    "index": "finger 0", "middle": "finger 1", "ring": "finger 2", "pinky": "finger 3",
    "spine 1": "spine", "spine 2": "spine 1", "spine 3": "spine 2", "spine 4": "spine 3",
    "neck 1": "neck", "neck 2": "neck 1",
    "jaw": "chin",
}
_SYNONYM_EXACT = {k: v for k, v in _CANONICAL_SYNONYMS.items() if any(c.isdigit() for c in k)}
_SYNONYM_PREFIX = {k: v for k, v in _CANONICAL_SYNONYMS.items() if not any(c.isdigit() for c in k)}


def normalize_match_name(name: str) -> str:
    """Normalize a joint name via the canonical synonym map (for Jaccard scoring)."""
    lower = str(name).lower().strip()
    if lower in _SYNONYM_EXACT:
        return _SYNONYM_EXACT[lower]
    for side in ("left ", "right "):
        if lower.startswith(side):
            return side + normalize_match_name(lower[len(side):])
    for key, value in _SYNONYM_PREFIX.items():
        if lower == key or lower.startswith(key + " "):
            suffix = lower[len(key):].strip()
            if suffix:
                # Extract trailing digit(s) from the suffix
                digit = ""
                for ch in reversed(suffix):
                    if ch.isdigit():
                        digit = ch + digit
                    elif ch == " ":
                        continue
                    else:
                        break
                if digit:
                    # Replace the trailing digit in the canonical value
                    # e.g. "finger 0" + digit "02" -> "finger 2"
                    digit_int = str(int(digit))
                    parts = value.rsplit(" ", 1)
                    if parts[-1].isdigit():
                        return parts[0] + " " + digit_int
                    return value + " " + digit_int
            return value
    return lower


def strip_helper_names(names: Sequence[str]) -> set:
    """Joint names for skeleton similarity.

    Under own-rotation encoding no leaf rotation helpers exist.
    Returns all names unchanged for backward compatibility.
    """
    return set(names)


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


def joint_name_set(object_cond: Mapping[str, object], object_type_hint: str) -> set:
    """Synonym-normalised, helper-free joint-name set for similarity scoring.

    Prefers ``canonical_joint_names``; falls back to ``joints_names`` so that
    conds without canonical names can still be ranked (retarget mapping uses
    the strict ``require_canonical_joint_names`` instead).
    """
    raw = object_cond.get("canonical_joint_names")
    if raw is None:
        raw = object_cond.get("joints_names")
    if raw is None:
        raise ValueError(f"No joint names available for {object_type_hint}")
    return {normalize_match_name(n) for n in strip_helper_names(list(raw))}


# ── Joint-name embedding (secondary semantic term) ───────────────────────────
def _l2_normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= eps:
        return vector.copy()
    return vector / norm


def has_embedding(object_cond: Mapping[str, object]) -> bool:
    embs = object_cond.get("joints_names_embs")
    if embs is None:
        return False
    arr = np.asarray(embs)
    return arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] > 0


def embedding_for_object(object_cond: Mapping[str, object]) -> np.ndarray:
    joint_embs = np.asarray(object_cond.get("joints_names_embs"), dtype=np.float64)
    if joint_embs.ndim != 2 or joint_embs.shape[0] == 0 or joint_embs.shape[1] == 0:
        raise ValueError("cond entry is missing valid joints_names_embs")
    return _l2_normalize(joint_embs.mean(axis=0))


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


def group_tags(object_type: object) -> frozenset:
    """Motion descriptor tags ``(body-plan, gait/dynamics, refinement)`` for an object_type.

    Case-insensitive; species absent from ``species_tags.jsonl`` (e.g. a novel
    retarget target the user has not registered) return an empty set, which
    yields no group discount.
    """
    return frozenset(dataset_tags().tags_for(object_type))


# ── Combined similarity ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class SimilarityWeights:
    """Relative weights of the three distance terms plus the group discount.

    ``jaccard`` + ``embedding`` + ``topology`` need not sum to 1: only their
    ratios matter, and inactive terms (e.g. missing embeddings) are dropped
    with the rest renormalised. ``group_bonus`` is the *maximum* fractional
    discount, applied at full motion-tag overlap (identical mover); partial
    overlap (e.g. same body-plan only) gets a proportional share.
    """

    topology: float = 0.5
    jaccard: float = 0.4
    embedding: float = 0.1
    group_bonus: float = 0.3


DEFAULT_WEIGHTS = SimilarityWeights()


@dataclass
class SpeciesSimilarity:
    name: str
    jaccard: float
    semantic_distance: float      # 1 - embedding cosine (nan if embedding inactive)
    topology_distance: float      # z-scored descriptor euclidean (pool-relative)
    combined_distance: float
    same_group: bool            # identical motion descriptor (full tag overlap)
    weight: float = 0.0


def _pool_scale(values: np.ndarray) -> np.ndarray:
    """Normalise a distance vector to pool mean ~1 (scale-free blend term)."""
    mean = float(np.mean(values))
    return values / mean if mean > 1e-12 else np.zeros_like(values)


def rank_species(
    query_cond: Mapping[str, object],
    candidate_conds: Mapping[str, Mapping[str, object]],
    *,
    query_hint: str,
    top_k: Optional[int] = None,
    weights: SimilarityWeights = DEFAULT_WEIGHTS,
) -> List[SpeciesSimilarity]:
    """Rank candidate skeletons by similarity to ``query_cond`` (closest first).

    Returns one ``SpeciesSimilarity`` per selected candidate, sorted by ascending
    ``combined_distance``, with softmax ``weight`` over the selected set. ``top_k``
    None ranks every candidate.
    """
    names = list(candidate_conds.keys())
    if not names:
        raise ValueError("No candidate skeletons to rank")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be >= 1 or None")

    query_names = joint_name_set(query_cond, query_hint)
    query_desc = topology_descriptor(query_cond)
    # Motion tags drive a graded group discount. The dict key (``name`` /
    # ``query_hint``) is the object_type; prefer an explicit cond field if set.
    query_tags = group_tags(query_cond.get("object_type") or query_hint)
    query_emb = embedding_for_object(query_cond) if has_embedding(query_cond) else None

    jaccard_arr = np.zeros(len(names), dtype=np.float64)
    semantic_arr = np.full(len(names), np.nan, dtype=np.float64)
    descriptors: List[np.ndarray] = []
    group_overlap = np.zeros(len(names), dtype=np.int64)
    for i, name in enumerate(names):
        cond = candidate_conds[name]
        cand_names = joint_name_set(cond, name)
        union = len(query_names | cand_names)
        jaccard_arr[i] = (len(query_names & cand_names) / union) if union else 0.0
        descriptors.append(topology_descriptor(cond))
        cand_tags = group_tags(cond.get("object_type") or name)
        group_overlap[i] = len(query_tags & cand_tags)
        if query_emb is not None and has_embedding(cond):
            cos = float(np.clip(np.dot(query_emb, embedding_for_object(cond)), -1.0, 1.0))
            semantic_arr[i] = max(0.0, 1.0 - cos)

    jaccard_distance = 1.0 - jaccard_arr

    # Topology distance: z-score each feature over {query + candidates}, then
    # Euclidean distance in that standardised, scale-free space.
    descriptor_arr = np.asarray(descriptors, dtype=np.float64)
    stacked = np.vstack([query_desc[None, :], descriptor_arr])
    feature_std = stacked.std(axis=0)
    feature_std = np.where(feature_std > 1e-9, feature_std, 1.0)
    feature_mean = stacked.mean(axis=0)
    query_z = (query_desc - feature_mean) / feature_std
    candidate_z = (descriptor_arr - feature_mean) / feature_std
    topology_distance = np.linalg.norm(candidate_z - query_z[None, :], axis=1)

    # Embedding term is active only when the query and *every* candidate carries
    # an embedding (otherwise the comparison would be apples-to-oranges).
    embedding_active = query_emb is not None and not np.isnan(semantic_arr).any()

    # Pool-normalise each active term (mean ~1) then blend by renormalised weight.
    terms: List[tuple[float, np.ndarray]] = [
        (weights.jaccard, jaccard_distance),
        (weights.topology, topology_distance),
    ]
    if embedding_active:
        terms.append((weights.embedding, semantic_arr))
    total_weight = sum(w for w, _ in terms) or 1.0
    combined = np.zeros(len(names), dtype=np.float64)
    for weight, distance in terms:
        combined += (weight / total_weight) * _pool_scale(distance)

    # Graded motion-group discount: shared motion tags fractionally shrink the
    # combined distance. Full overlap (identical mover, e.g. Cat/Lion) gets the
    # full bonus; partial overlap (e.g. same body-plan only, Cat/Horse) gets a
    # proportional share; no shared tag (or an unregistered species) -> no
    # discount. ``same_group`` means "identical motion descriptor" (full overlap).
    same_group = group_overlap >= _GROUP_TAG_ARITY
    group_factor = 1.0 - weights.group_bonus * (group_overlap / _GROUP_TAG_ARITY)
    combined = combined * group_factor

    order = sorted(range(len(names)), key=lambda i: (combined[i], names[i]))
    selected = order if top_k is None else order[: min(top_k, len(order))]

    results = [
        SpeciesSimilarity(
            name=names[i],
            jaccard=float(jaccard_arr[i]),
            semantic_distance=float(semantic_arr[i]) if embedding_active else float("nan"),
            topology_distance=float(topology_distance[i]),
            combined_distance=float(combined[i]),
            same_group=bool(same_group[i]),
        )
        for i in selected
    ]

    if len(results) == 1:
        results[0].weight = 1.0
        return results

    distances = np.asarray([r.combined_distance for r in results], dtype=np.float64)
    positive = distances[distances > 1e-8]
    temperature = max(float(np.median(positive)) if positive.size else 0.03, 0.03)
    logits = -distances / temperature
    logits -= logits.max()
    softmax = np.exp(logits)
    softmax /= softmax.sum()
    for result, weight in zip(results, softmax):
        result.weight = float(weight)
    return results
