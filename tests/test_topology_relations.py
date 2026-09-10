"""Pairwise topology codes: the refinement must only subdivide, never relabel.

``graph_dist``/``joint_relations`` are the only route by which the tree reaches
the forward pass, and the two things that can go wrong are silent: emitting an
index past the model's embedding table (a device-side gather failure that names
nothing), and quietly changing what the *near* codes mean, which would make the
refinement a semantic change rather than a subdivision.

The reference implementation below is the original ``create_topology_edge_relations``,
kept verbatim so "codes 0-3 and 5 are untouched" is checked against the real
thing rather than against a restatement of the new code.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_loaders.truebones.truebones_utils.topology_relations import (  # noqa: E402
    EDGE_CODES,
    GRAPH_DIST_FAR_BASE,
    NUM_EDGE_CODES,
    NUM_TOPOLOGY_CODES,
    create_topology_edge_relations,
    refresh_topology_relations_in_cond_dict,
)


def legacy_topology_edge_relations(parents, max_path_len=5):
    """The pre-refinement implementation, verbatim."""
    edge_types = {'self': 0, 'parent': 1, 'child': 2, 'sibling': 3,
                  'no_relation': 4, 'end_effector': 5}
    n = len(parents)
    topo_rel = np.zeros((n, n))
    edge_rel = np.ones((n, n)) * edge_types['no_relation']
    for i in range(n):
        parent = parents[i]
        ee = True
        for j in range(n):
            parent_j = parents[j]
            edge_type = edge_types['no_relation']
            if i == j:
                edge_type = edge_types['self']
            elif parent_j == i:
                ee = False
                edge_type = edge_types['child']
            elif j == parent:
                edge_type = edge_types['parent']
            elif parent_j == parent:
                edge_type = edge_types['sibling']
            edge_rel[i, j] = edge_type

            if i == j:
                topo_rel[i, j] = 0
            elif j < i:
                topo_rel[i, j] = topo_rel[j, i]
            elif parent_j == i:
                topo_rel[i, j] = 1
            else:
                topo_rel[i, j] = topo_rel[i, parent_j] + 1
        if ee:
            edge_rel[i, i] = edge_types['end_effector']
    topo_rel[topo_rel > max_path_len] = max_path_len
    return edge_rel, topo_rel


# Root, spine of 3, two 4-joint legs and two 3-joint arms off the chest, plus a
# 2-joint tail: deep enough for the >=5-hop bucket to be populated, branchy
# enough to exercise sibling limbs.
HUMANOID_PARENTS = np.array([
    -1,          # 0  hips
    0, 1, 2,     # 1..3  spine chain (3 = chest, a branch point)
    0, 4, 5, 6,  # 4..7  left leg
    0, 8, 9, 10,  # 8..11 right leg
    3, 12, 13,   # 12..14 left arm
    3, 15, 16,   # 15..17 right arm
    3, 18,       # 18..19 neck/head
], dtype=np.int64)

# A single unbranched chain: every joint is in one run, no siblings anywhere.
CHAIN_PARENTS = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)

# Radially symmetric: eight 2-joint legs off one body node.
SPIDER_PARENTS = np.array(
    [-1] + [i for leg in range(8) for i in (0, 1 + 2 * leg)], dtype=np.int64
)

# Branch points at three different depths, so a cousin pair's LCA can land in
# any of the three normalized-depth tiers. The humanoid alone does not reach the
# upper tiers: every one of its cousin pairs meets at the hips.
DEEP_BRANCHING_PARENTS = np.array([
    -1, 0, 1, 2, 3,   # 0..4   spine, 4 branches
    4, 5, 6, 7,       # 5..8   trunk continues, 8 branches
    8, 9,             # 9..10  limb P, 10 branches
    10, 11,           # 11..12 P's first fork
    10, 13,           # 13..14 P's second fork
    8, 15,            # 15..16 limb Q
    4, 17,            # 17..18 a limb off the shallow branch point
], dtype=np.int64)

ALL_TREES = {
    'humanoid': HUMANOID_PARENTS,
    'chain': CHAIN_PARENTS,
    'spider': SPIDER_PARENTS,
    'deep_branching': DEEP_BRANCHING_PARENTS,
}


class TopologyCodeRangeTest(unittest.TestCase):
    def test_codes_stay_inside_the_embedding_tables(self):
        for name, parents in ALL_TREES.items():
            edge_rel, topo_rel = create_topology_edge_relations(parents)
            with self.subTest(tree=name):
                self.assertGreaterEqual(int(edge_rel.min()), 0)
                self.assertLess(int(edge_rel.max()), NUM_EDGE_CODES)
                self.assertGreaterEqual(int(topo_rel.min()), 0)
                self.assertLess(int(topo_rel.max()), NUM_TOPOLOGY_CODES)

    def test_no_relation_is_never_emitted(self):
        """Index 4 is reserved only to keep 0-3 and 5 at their historical values."""
        for name, parents in ALL_TREES.items():
            edge_rel, _ = create_topology_edge_relations(parents)
            with self.subTest(tree=name):
                self.assertNotIn(EDGE_CODES['no_relation'], np.unique(edge_rel))

    def test_padding_fill_still_reads_as_the_diagonal(self):
        """``create_padded_relation`` zero-fills past n_joints.

        Zero has to keep meaning 'self'/'distance 0' or a padded slot would read
        as some live relation. Padding is masked out of attention anyway, but the
        codes must not quietly depend on that.
        """
        self.assertEqual(EDGE_CODES['self'], 0)
        edge_rel, topo_rel = create_topology_edge_relations(HUMANOID_PARENTS)
        self.assertTrue((np.diag(topo_rel) == 0).all())


class MalformedParentsTest(unittest.TestCase):
    """A second root would read ``depth[-1]`` and silently return a wrong tree."""

    def test_a_second_root_is_rejected(self):
        two_roots = np.array([-1, 0, -1, 2], dtype=np.int64)
        with self.assertRaises(ValueError):
            create_topology_edge_relations(two_roots)

    def test_a_forward_reference_is_rejected(self):
        not_topological = np.array([-1, 2, 0], dtype=np.int64)
        with self.assertRaises(ValueError):
            create_topology_edge_relations(not_topological)

    def test_a_mismatched_max_path_len_is_rejected(self):
        """The code table and the model's embedding size derive from one constant."""
        with self.assertRaises(ValueError):
            create_topology_edge_relations(CHAIN_PARENTS, max_path_len=8)


class NearFieldUnchangedTest(unittest.TestCase):
    """The refinement subdivides the two saturated buckets and nothing else."""

    def test_exact_hop_codes_match_the_legacy_implementation(self):
        for name, parents in ALL_TREES.items():
            legacy_edge, legacy_topo = legacy_topology_edge_relations(parents)
            edge_rel, topo_rel = create_topology_edge_relations(parents)
            near = legacy_topo < GRAPH_DIST_FAR_BASE
            with self.subTest(tree=name):
                self.assertTrue(near.any(), 'tree has no near pairs to compare')
                np.testing.assert_array_equal(topo_rel[near], legacy_topo[near])
                # Everything the legacy code called 'far' must have moved out of
                # the exact-hop range, not stayed put.
                self.assertTrue((topo_rel[~near] >= GRAPH_DIST_FAR_BASE).all())

    def test_self_parent_child_sibling_end_effector_match_the_legacy_implementation(self):
        kept = (EDGE_CODES['self'], EDGE_CODES['parent'], EDGE_CODES['child'],
                EDGE_CODES['sibling'], EDGE_CODES['end_effector'])
        for name, parents in ALL_TREES.items():
            legacy_edge, _ = legacy_topology_edge_relations(parents)
            edge_rel, _ = create_topology_edge_relations(parents)
            mask = np.isin(legacy_edge, kept)
            with self.subTest(tree=name):
                np.testing.assert_array_equal(edge_rel[mask], legacy_edge[mask])
                # And the refinement only touched former 'no_relation' cells.
                changed = edge_rel != legacy_edge
                self.assertTrue(
                    (legacy_edge[changed] == EDGE_CODES['no_relation']).all()
                )


class RefinementSemanticsTest(unittest.TestCase):
    def test_graph_dist_stays_symmetric(self):
        for name, parents in ALL_TREES.items():
            _, topo_rel = create_topology_edge_relations(parents)
            with self.subTest(tree=name):
                np.testing.assert_array_equal(topo_rel, topo_rel.T)

    def test_ancestor_and_descendant_are_mirror_images(self):
        for name, parents in ALL_TREES.items():
            edge_rel, _ = create_topology_edge_relations(parents)
            anc = edge_rel == EDGE_CODES['ancestor']
            desc = edge_rel == EDGE_CODES['descendant']
            with self.subTest(tree=name):
                np.testing.assert_array_equal(anc, desc.T)

    def test_direction_now_exists_beyond_one_hop(self):
        """The gap the refinement exists to close."""
        edge_rel, _ = create_topology_edge_relations(HUMANOID_PARENTS)
        legacy_edge, _ = legacy_topology_edge_relations(HUMANOID_PARENTS)
        head, hips = 19, 0
        self.assertEqual(legacy_edge[head, hips], EDGE_CODES['no_relation'])
        self.assertEqual(legacy_edge[hips, head], EDGE_CODES['no_relation'])
        self.assertEqual(edge_rel[head, hips], EDGE_CODES['ancestor'])
        self.assertEqual(edge_rel[hips, head], EDGE_CODES['descendant'])

    def test_left_and_right_legs_read_as_sibling_limbs(self):
        edge_rel, _ = create_topology_edge_relations(HUMANOID_PARENTS)
        left_knee, right_knee = 5, 9
        self.assertEqual(edge_rel[left_knee, right_knee], EDGE_CODES['sibling_limb'])
        self.assertEqual(edge_rel[right_knee, left_knee], EDGE_CODES['sibling_limb'])

    def test_a_leg_and_an_arm_do_not_read_as_sibling_limbs(self):
        """Legs hang off the hips, arms off the chest -- different branch points."""
        edge_rel, _ = create_topology_edge_relations(HUMANOID_PARENTS)
        left_knee, left_elbow = 5, 13
        self.assertNotEqual(edge_rel[left_knee, left_elbow], EDGE_CODES['sibling_limb'])

    def test_an_unbranched_chain_yields_only_ancestor_and_descendant(self):
        edge_rel, _ = create_topology_edge_relations(CHAIN_PARENTS)
        refined = edge_rel[edge_rel >= EDGE_CODES['ancestor']]
        self.assertTrue(
            set(np.unique(refined)).issubset(
                {EDGE_CODES['ancestor'], EDGE_CODES['descendant']}
            ),
            f"unexpected codes on a bare chain: {np.unique(refined)}",
        )

    def test_saturation_actually_drops(self):
        """The measured claim, on a tree small enough to state exactly."""
        parents = HUMANOID_PARENTS
        n = len(parents)
        off = ~np.eye(n, dtype=bool)

        def largest_bucket(matrix):
            values, counts = np.unique(matrix[off], return_counts=True)
            return counts.max() / off.sum()

        legacy_edge, legacy_topo = legacy_topology_edge_relations(parents)
        edge_rel, topo_rel = create_topology_edge_relations(parents)
        self.assertLess(largest_bucket(topo_rel), largest_bucket(legacy_topo))
        self.assertLess(largest_bucket(edge_rel), largest_bucket(legacy_edge))


class EmittedCodeCoverageTest(unittest.TestCase):
    """Every declared code must be reachable.

    Two earlier candidates ('same_run', 'nested_limb') read as useful and were
    provably empty under the precedence rules -- a branch-free run is a path, so
    both collapse into ancestor/descendant. A dead code costs an embedding row
    and, worse, reads in the source as a distinction the model can make.
    """

    def test_every_code_except_the_reserved_one_is_emitted(self):
        emitted = set()
        for parents in ALL_TREES.values():
            edge_rel, topo_rel = create_topology_edge_relations(parents)
            emitted.update(int(v) for v in np.unique(edge_rel))
        declared = set(EDGE_CODES.values()) - {EDGE_CODES['no_relation']}
        self.assertEqual(
            declared - emitted, set(),
            f"declared but never emitted: "
            f"{sorted(k for k, v in EDGE_CODES.items() if v in declared - emitted)}",
        )

    def test_cousin_tiers_track_how_deep_the_pair_meets(self):
        """The tier is the LCA depth as a fraction of the skeleton's own depth.

        Absolute depth would not do: it is the quantity that made ``d_contact``
        unusable across skeletons of different sizes.
        """
        edge_rel, _ = create_topology_edge_relations(DEEP_BRANCHING_PARENTS)
        # (12, 18) meet at joint 4; (12, 16) meet at joint 8, twice as deep.
        self.assertEqual(edge_rel[12, 18], EDGE_CODES['cousin_mid'])
        self.assertEqual(edge_rel[12, 16], EDGE_CODES['cousin_deep'])
        # Two forks off the same branch point stay a sibling limb, not a cousin.
        self.assertEqual(edge_rel[12, 14], EDGE_CODES['sibling_limb'])

    def test_the_table_is_exactly_as_large_as_the_codes_need(self):
        self.assertEqual(NUM_EDGE_CODES, max(EDGE_CODES.values()) + 1)


class HopDistanceTest(unittest.TestCase):
    """The vectorized LCA-based distance must equal the legacy recursion."""

    def test_hops_match_a_brute_force_tree_distance(self):
        for name, parents in ALL_TREES.items():
            _, legacy_topo = legacy_topology_edge_relations(parents, max_path_len=10**6)
            n = len(parents)
            brute = np.zeros((n, n), dtype=np.int64)
            paths = []
            for i in range(n):
                path, p = [i], int(parents[i])
                while p >= 0:
                    path.append(p)
                    p = int(parents[p])
                paths.append(path)
            for i in range(n):
                for j in range(n):
                    common = set(paths[i]) & set(paths[j])
                    depth_lca = max(
                        len(paths[i]) - 1 - paths[i].index(c) for c in common
                    )
                    brute[i, j] = (len(paths[i]) - 1 - depth_lca) + (
                        len(paths[j]) - 1 - depth_lca
                    )
            with self.subTest(tree=name):
                np.testing.assert_array_equal(legacy_topo.astype(np.int64), brute)


class CondRefreshTest(unittest.TestCase):
    def test_refresh_overwrites_stale_matrices_from_parents(self):
        """Existing cond.npy files carry the old codes; the load path replaces them."""
        legacy_edge, legacy_topo = legacy_topology_edge_relations(HUMANOID_PARENTS)
        cond = {
            'some/Species': {
                'parents': HUMANOID_PARENTS,
                'joint_relations': legacy_edge,
                'joints_graph_dist': legacy_topo,
            }
        }
        refresh_topology_relations_in_cond_dict(cond)
        expected_edge, expected_topo = create_topology_edge_relations(HUMANOID_PARENTS)
        np.testing.assert_array_equal(cond['some/Species']['joint_relations'], expected_edge)
        np.testing.assert_array_equal(cond['some/Species']['joints_graph_dist'], expected_topo)

    def test_refresh_skips_entries_without_parents(self):
        cond = {'no_parents': {'rest_pose': np.zeros((3, 12))}}
        refresh_topology_relations_in_cond_dict(cond)
        self.assertNotIn('joint_relations', cond['no_parents'])


if __name__ == '__main__':
    unittest.main()
