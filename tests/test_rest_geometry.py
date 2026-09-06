"""Re-seating a species' rest bones onto the geometry its clips actually use.

The step runs inside regenerate_dataset_artifacts, so what matters most here is
that it converges (idempotent), that it never touches a joint whose rest is a
legitimate neutral, and that it keeps ``rest_pos_ric_hml == FK(offsets)``.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_loaders.truebones.truebones_utils import rest_geometry as rg  # noqa: E402


# root -> spine -> head -> {nub (leaf), ear (leaf)}; spine also -> tail (leaf).
PARENTS = np.array([-1, 0, 1, 2, 2, 1], dtype=np.int64)
NAMES = ["Root", "Spine", "Head", "HeadNub", "Ear", "Tail"]
OFFSETS = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.5, 0.0],
    [0.0, 0.4, 0.0],
    [0.0, 0.2, 0.0],   # nub: rest gives it 0.2, the clips collapse it onto Head
    [0.15, 0.1, 0.0],  # ear: rest 0.18, the clips hold it at ~0.09
    [0.0, -0.1, -0.4],  # tail: genuinely animated, rest is a legitimate neutral
], dtype=np.float32)


def _fk(offsets, parents):
    rest = np.zeros_like(offsets)
    for joint, parent in enumerate(parents):
        rest[joint] = offsets[joint] if parent < 0 else rest[parent] + offsets[joint]
    return rest


def _entry():
    rest_pos = _fk(OFFSETS, PARENTS)
    rest_pose = np.zeros((len(PARENTS), 13), dtype=np.float32)
    rest_pose[:, 0:3] = rest_pos
    return {
        "parents": PARENTS.copy(),
        "offsets": OFFSETS.copy(),
        "rest_pose": rest_pose,
        "rest_pos_ric_hml": rest_pos.astype(np.float32, copy=True),
        "joints_names": list(NAMES),
    }


def _clip(entry, frames=20, seed=0):
    """A clip that collapses the nub, halves the ear, and animates the tail."""
    rng = np.random.default_rng(seed)
    rest_pos = np.asarray(entry["rest_pos_ric_hml"], dtype=np.float64)
    motion = np.zeros((frames, len(PARENTS), 13), dtype=np.float32)
    motion[:, :, 0:3] = rest_pos[None, :, :]
    motion[:, 3, 0:3] = rest_pos[2]                                  # nub -> onto Head
    motion[:, 4, 0:3] = rest_pos[2] + (rest_pos[4] - rest_pos[2]) * 0.5  # ear -> half
    # Tail swings: its length is far from rigid, so its rest must be left alone.
    swing = rng.uniform(0.4, 1.6, size=frames)
    motion[:, 5, 0:3] = rest_pos[1] + (rest_pos[5] - rest_pos[1])[None, :] * swing[:, None]
    return motion


def _measure(entry, clips):
    acc = None
    for clip in clips:
        acc = rg.accumulate_rest_vs_clip(entry, clip, acc=acc)
    return rg.finalize_rest_vs_clip(entry, acc)


class RestGeometryTest(unittest.TestCase):
    def test_selects_only_the_rigid_disagreeing_leaves(self):
        entry = _entry()
        report = _measure(entry, [_clip(entry)])
        picked = {c["name"] for c in rg.reseat_candidates(entry, report)}
        self.assertEqual(picked, {"HeadNub", "Ear"})

    def test_animated_bone_is_never_reseated(self):
        entry = _entry()
        report = _measure(entry, [_clip(entry, seed=1), _clip(entry, seed=2)])
        # Tail's length varies 0.4x-1.6x across frames: there is no single length
        # to move its rest onto, so its rest stays the neutral the rig shipped.
        self.assertNotIn("Tail", {c["name"] for c in rg.reseat_candidates(entry, report)})

    def test_interior_joint_is_never_reseated(self):
        entry = _entry()
        # Make Head itself disagree. It carries children, so moving it would drag
        # the whole subtree and turn one bone's error into its descendants'.
        clip = _clip(entry)
        rest_pos = np.asarray(entry["rest_pos_ric_hml"], dtype=np.float64)
        clip[:, 2, 0:3] = rest_pos[1] + (rest_pos[2] - rest_pos[1]) * 0.4
        report = _measure(entry, [clip])
        self.assertNotIn("Head", {c["name"] for c in rg.reseat_candidates(entry, report)})

    def test_apply_matches_clip_length_and_keeps_fk_consistent(self):
        entry = _entry()
        report = _measure(entry, [_clip(entry)])
        candidates = rg.reseat_candidates(entry, report)
        self.assertEqual(rg.apply_reseat(entry, candidates), 2)

        offsets = np.asarray(entry["offsets"])
        rest_pos = np.asarray(entry["rest_pos_ric_hml"])
        # The nub collapses onto its parent; the ear takes half its rest length.
        np.testing.assert_allclose(offsets[3], 0.0, atol=1e-6)
        np.testing.assert_allclose(
            np.linalg.norm(offsets[4]), np.linalg.norm(OFFSETS[4]) * 0.5, rtol=1e-5)
        # Direction is preserved -- length is the rotation-invariant part.
        np.testing.assert_allclose(
            offsets[4] / np.linalg.norm(offsets[4]),
            OFFSETS[4] / np.linalg.norm(OFFSETS[4]), rtol=1e-5)
        # Untouched joints stay put, and the two rest representations still agree.
        np.testing.assert_array_equal(offsets[5], OFFSETS[5])
        np.testing.assert_allclose(rest_pos, _fk(offsets, PARENTS), atol=1e-6)
        np.testing.assert_allclose(rest_pos, np.asarray(entry["rest_pose"])[:, 0:3], atol=0)

    def test_is_idempotent(self):
        entry = _entry()
        # The SAME clips on both passes -- the motion files on disk do not change
        # when the rest moves, which is exactly why convergence has to come from
        # the rule and not from the data.
        clips = [_clip(entry)]
        rg.apply_reseat(entry, rg.reseat_candidates(entry, _measure(entry, clips)))
        after_first = np.asarray(entry["offsets"]).copy()

        second = rg.reseat_candidates(entry, _measure(entry, clips))
        self.assertEqual(second, [])
        np.testing.assert_array_equal(np.asarray(entry["offsets"]), after_first)

    def test_prop_socket_is_excluded(self):
        # A held weapon parked far from the body is the repo's calibrated
        # prop-socket case: it is already kept out of the scale statistics, and
        # re-seating it would change the rest span those statistics rest on.
        # A long spine so the socket does not become its own p90 reference --
        # find_prop_socket_joints is calibrated on real rigs of 40-100 bones.
        n_spine = 14
        parents = np.array([-1] + list(range(n_spine)) + [n_spine, n_spine],
                           dtype=np.int64)
        names = ["Root"] + [f"Spine{i}" for i in range(n_spine)] + ["HeadNub", "Weapon01"]
        offsets = np.zeros((len(parents), 3), dtype=np.float32)
        offsets[1:n_spine + 1, 1] = 0.4
        offsets[n_spine + 1] = [0.0, 0.2, 0.0]   # nub, collapsed by the clips
        offsets[n_spine + 2] = [3.0, 0.0, 0.0]   # weapon, parked off the body
        rest_pos = _fk(offsets, parents)
        rest_pose = np.zeros((len(parents), 13), dtype=np.float32)
        rest_pose[:, 0:3] = rest_pos
        entry = {
            "parents": parents, "offsets": offsets, "rest_pose": rest_pose,
            "rest_pos_ric_hml": rest_pos.astype(np.float32, copy=True),
            "joints_names": names,
        }

        clip = np.zeros((12, len(parents), 13), dtype=np.float32)
        clip[:, :, 0:3] = rest_pos[None, :, :]
        clip[:, n_spine + 1, 0:3] = rest_pos[n_spine]                       # nub collapsed
        parent_pos = rest_pos[n_spine]
        clip[:, n_spine + 2, 0:3] = parent_pos + (rest_pos[n_spine + 2] - parent_pos) * 0.1

        report = _measure(entry, [clip])
        picked = {c["name"] for c in rg.reseat_candidates(entry, report)}
        self.assertNotIn("Weapon01", picked)
        self.assertIn("HeadNub", picked)   # the rest of the rule still fires

    def test_degenerate_and_empty_inputs_are_tolerated(self):
        entry = _entry()
        self.assertIsNone(rg.finalize_rest_vs_clip(entry, None))
        # A clip with too few joints is skipped rather than raising.
        self.assertIsNone(rg.accumulate_rest_vs_clip(entry, np.zeros((4, 2, 13), np.float32)))
        self.assertEqual(rg.reseat_candidates(entry, None), [])
        self.assertEqual(rg.apply_reseat(entry, []), 0)


if __name__ == "__main__":
    unittest.main()
