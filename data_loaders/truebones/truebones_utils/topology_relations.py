"""Pairwise skeleton-topology codes for the graph attention bias.

``graph_dist`` and ``joint_relations`` are the *only* path by which the actual
tree structure reaches the forward pass: ``parents`` is handed to the model but
is read solely by ``AnyTop._sample_subtree_joint_mask`` (an augmentation
sampler), never as a conditioning input. Whatever these two matrices fail to
distinguish, the network cannot recover.

The original codes distinguished very little. Measured over the 257 skeletons in
``dataset/merged/cond.npy`` (share of off-diagonal pairs landing in the single
largest code, averaged over species):

    graph_dist       61.0%   (5-hop saturation: everything >=5 hops was one code)
    joint_relations  88.3%   (direction existed only at 1 hop; the rest was
                              'no_relation')

Both are refined here without introducing an absolute hop count, an absolute
length, or an absolute coordinate -- a scale-free constraint, because absolute
graph distance is not comparable across skeletons (a human forearm is 9 hops
from a foot; a quadruped's is 1 hop from its forepaw, for the same anatomical
role). The refinements key on quantities that normalize by the skeleton's own
size: which *branch-free run* a joint sits in, and the depth of the pair's
lowest common ancestor as a fraction of the skeleton's maximum depth.

    graph_dist       61.0% -> 44.7%   (>60% on 155/257 species -> 30/257)
    joint_relations  88.3% -> 46.6%   (>60% on 252/257 species ->  45/257)

The residual is a hard floor rather than a coding weakness: on radially
symmetric many-limbed rigs (``zoo/Raptor2``, ``zoo/HermitCrab``,
``zoo/Isopetra``) leg 3 and leg 7 attach at the same branch point at the same
depth, so *no* purely topological pairwise code separates them. Telling those
apart needs geometry, which belongs in a per-joint channel, not here.

Codes 0-3 and 5 keep their historical meaning so the near field is unchanged;
only the two saturated buckets (hop >= MAX_PATH_LEN, and ``no_relation``) are
subdivided. The tables are still plain integer indices into an ``nn.Embedding``,
so nothing about the GRPE attention bias changes shape.
"""

from __future__ import annotations

import numpy as np

from data_loaders.truebones.truebones_utils.param_utils import MAX_PATH_LEN


# ---------------------------------------------------------------------------
# Edge relation codes (directed: ``edge_rel[i, j]`` is j's relation *to* i).
# ---------------------------------------------------------------------------
EDGE_CODES = {
    'self': 0,
    'parent': 1,
    'child': 2,
    'sibling': 3,
    # Index 4 was 'no_relation' and covered 88% of all pairs. It is retained as a
    # reserved, never-emitted index so codes 0-3 and 5 keep the meaning they had
    # in every earlier checkpoint's embedding table; every pair that used to land
    # here now lands in 6..11 below.
    'no_relation': 4,
    'end_effector': 5,
    # --- former 'no_relation', subdivided -----------------------------------
    'ancestor': 6,        # j is a strict ancestor of i, more than one hop up
    'descendant': 7,      # j is a strict descendant of i, more than one hop down
    'sibling_limb': 8,    # the two runs hang off the same branch point
    'cousin_shallow': 9,  # everything else, split by normalized LCA depth
    'cousin_mid': 10,
    'cousin_deep': 11,
}
NUM_EDGE_CODES = 12
EDGE_COUSIN_TIERS = 3

# Two categories that look useful and are provably empty, recorded so they do not
# get proposed again. A branch-free run is a path, so any two joints in it are
# already ancestor/descendant -- a 'same_run' code can never win. And 'one joint's
# run root is an ancestor of the other' likewise collapses: a run has no branches
# before its last joint, so anything hanging below the run root is either below
# that last joint (making the pair ancestor/descendant) or is the run root itself.
# ``test_topology_relations.EmittedCodeCoverageTest`` fails if a dead code returns.

# ---------------------------------------------------------------------------
# Graph distance codes.
#   0 .. MAX_PATH_LEN-1        exact hop count (unchanged)
#   MAX_PATH_LEN .. +3         >= MAX_PATH_LEN hops, one joint above the other,
#                              by normalized LCA depth quartile
#   MAX_PATH_LEN+4 .. +7       >= MAX_PATH_LEN hops, on separate branches,
#                              by normalized LCA depth quartile
# ---------------------------------------------------------------------------
GRAPH_DIST_FAR_BASE = int(MAX_PATH_LEN)
GRAPH_DIST_LCA_TIERS = 4
NUM_TOPOLOGY_CODES = GRAPH_DIST_FAR_BASE + 2 * GRAPH_DIST_LCA_TIERS


def _depths(parents: np.ndarray) -> np.ndarray:
    depth = np.zeros(len(parents), dtype=np.int64)
    for i in range(1, len(parents)):
        depth[i] = depth[parents[i]] + 1
    return depth


def _ancestor_or_self(parents: np.ndarray) -> np.ndarray:
    """``A[i, a]`` is True when *a* is an ancestor of *i* or is *i* itself.

    Built in topological order (``parents[i] < i`` holds for every processed
    skeleton, which is DFS-reordered), so each row is its parent's row plus the
    parent itself.
    """
    n = len(parents)
    anc = np.zeros((n, n), dtype=bool)
    for i in range(n):
        p = int(parents[i])
        if p >= 0:
            anc[i] = anc[p]
            anc[i, p] = True
        anc[i, i] = True
    return anc


def _branch_free_runs(parents: np.ndarray) -> np.ndarray:
    """Return each joint's run root: the top of its branch-free chain.

    Splits the tree into maximal chains of single-child joints.

    A run starts at the root or at any child of a branch point and continues
    while every joint on it has exactly one child. This is *not*
    ``kinematic_chains``: that one greedily merges root-to-tip paths, so a
    humanoid's third chain runs Hips -> spine -> shoulder -> a pinky tip and its
    internal depth means nothing.
    """
    n = len(parents)
    n_children = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        n_children[parents[i]] += 1
    run_root = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        p = int(parents[i])
        run_root[i] = i if (p < 0 or n_children[p] != 1) else run_root[p]
    return run_root


def create_topology_edge_relations(parents, max_path_len: int = MAX_PATH_LEN):
    """Return ``(edge_rel, topo_rel)``, both ``(n, n)`` int arrays.

    ``topo_rel`` stays symmetric; ``edge_rel`` is directed, as it always was
    (``parent``/``child`` already distinguished the two orders).
    """
    parents = np.asarray(parents, dtype=np.int64).reshape(-1)
    n = len(parents)
    max_path_len = int(max_path_len)
    if max_path_len != GRAPH_DIST_FAR_BASE:
        raise ValueError(
            f"max_path_len={max_path_len} does not match the code table built for "
            f"{GRAPH_DIST_FAR_BASE}; NUM_TOPOLOGY_CODES and the model's embedding "
            "size both derive from param_utils.MAX_PATH_LEN, so they have to move "
            "together (and the checkpoint has to be retrained)."
        )
    if n == 0:
        return np.zeros((0, 0), dtype=np.int64), np.zeros((0, 0), dtype=np.int64)
    # Single root at index 0, every other parent already seen: the invariant
    # ``reorder_animation_to_dfs`` establishes and that ``_depths`` relies on --
    # a second ``-1`` would index ``depth[-1]``, quietly reading the last joint's
    # depth instead of failing.
    if int(parents[0]) >= 0 or (n > 1 and not (parents[1:] >= 0).all()):
        raise ValueError(
            "parents must be a single-rooted, topologically ordered tree with "
            f"parents[0] < 0; got roots at {np.flatnonzero(parents < 0).tolist()}"
        )
    if n > 1 and not (parents[1:] < np.arange(1, n)).all():
        raise ValueError("parents must be topologically ordered (parents[i] < i)")

    depth = _depths(parents)
    max_depth = max(int(depth.max()), 1)
    anc = _ancestor_or_self(parents)
    run_root = _branch_free_runs(parents)
    eye = np.eye(n, dtype=bool)
    idx = np.arange(n)

    # |anc_or_self(i) & anc_or_self(j)| is the size of the LCA's own root path,
    # because both root paths are chains sharing exactly that prefix.
    anc_int = anc.astype(np.int64)
    lca_depth = anc_int @ anc_int.T - 1
    hops = depth[:, None] + depth[None, :] - 2 * lca_depth
    np.fill_diagonal(hops, 0)

    is_anc = anc & ~eye          # is_anc[i, a]: a is a strict ancestor of i
    colinear = is_anc | is_anc.T  # one joint lies above the other

    # --- graph distance ----------------------------------------------------
    lca_tier = np.minimum(
        GRAPH_DIST_LCA_TIERS - 1, (lca_depth * GRAPH_DIST_LCA_TIERS) // max_depth
    )
    far = hops >= max_path_len
    topo_rel = np.where(
        far,
        GRAPH_DIST_FAR_BASE + lca_tier + np.where(colinear, 0, GRAPH_DIST_LCA_TIERS),
        hops,
    ).astype(np.int64)
    np.fill_diagonal(topo_rel, 0)

    # --- edge relations ----------------------------------------------------
    parent_of = parents[:, None] == idx[None, :]                     # j is i's parent
    child_of = parent_of.T
    sibling = (parents[:, None] == parents[None, :]) & (parents[:, None] >= 0) & ~eye

    limb_parent = parents[run_root]
    sibling_limb = (limb_parent[:, None] == limb_parent[None, :]) & (
        limb_parent[:, None] >= 0
    )

    cousin_tier = np.minimum(EDGE_COUSIN_TIERS - 1, (lca_depth * EDGE_COUSIN_TIERS) // max_depth)
    edge_rel = (EDGE_CODES['cousin_shallow'] + cousin_tier).astype(np.int64)
    edge_rel = np.where(sibling_limb, EDGE_CODES['sibling_limb'], edge_rel)
    edge_rel = np.where(is_anc, EDGE_CODES['ancestor'], edge_rel)
    edge_rel = np.where(is_anc.T, EDGE_CODES['descendant'], edge_rel)
    # The near field is applied last so it wins over every refinement above,
    # preserving the original precedence (self > child > parent > sibling).
    edge_rel = np.where(sibling, EDGE_CODES['sibling'], edge_rel)
    edge_rel = np.where(parent_of, EDGE_CODES['parent'], edge_rel)
    edge_rel = np.where(child_of, EDGE_CODES['child'], edge_rel)
    np.fill_diagonal(edge_rel, EDGE_CODES['self'])
    is_leaf = ~np.any(child_of, axis=1)
    edge_rel[idx[is_leaf], idx[is_leaf]] = EDGE_CODES['end_effector']

    return edge_rel, topo_rel


def refresh_topology_relations_in_object_cond(object_cond) -> None:
    """Recompute both matrices from ``parents``, in place.

    Both are pure functions of ``parents``, which every cond entry already
    carries, so a code change here reaches existing datasets without a regen.
    It does mean the values on disk are advisory; the next preprocessing run
    persists the refreshed ones.
    """
    if not isinstance(object_cond, dict) or 'parents' not in object_cond:
        return
    edge_rel, topo_rel = create_topology_edge_relations(object_cond['parents'])
    object_cond['joint_relations'] = edge_rel
    object_cond['joints_graph_dist'] = topo_rel


def refresh_topology_relations_in_cond_dict(cond_dict):
    """Refresh every species in a loaded cond dict.

    Called from ``cond_schema.load_cond``, the single point through which the
    dataset, ``sample/generate.py`` and the validator all read ``cond.npy`` --
    one hook rather than one per call site, because a missed call site would mix
    old and new codes in the same training run and fail silently.
    """
    if not isinstance(cond_dict, dict):
        return cond_dict
    for object_cond in cond_dict.values():
        refresh_topology_relations_in_object_cond(object_cond)
    return cond_dict
