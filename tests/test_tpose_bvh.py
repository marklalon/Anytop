"""The cond t-pose BVH: what the review page and tools/sample_tpose_bvh.py rely on.

A reader maps BVH joint ``k`` to cond joint ``order[k]``; the tests pin that
mapping on a cond whose joint order is not the hierarchy's DFS order (helpers
appended at the tail), and that the file reproduces the cond rest positions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT.parent))

from data_loaders.truebones.truebones_utils.tpose_bvh import (  # noqa: E402
    tpose_bvh_text,
    tpose_dfs_order,
    write_tpose_bvh,
)


def _entry():
    # 0 root -> 1 spine -> 2 head; 0 -> 3 tail; 1 -> 4 (appended helper under the spine).
    return {
        'joints_names': ['Root', 'Spine', 'Head', 'Tail', 'Helper Socket'],
        'parents': np.array([-1, 0, 1, 0, 1]),
        'offsets': np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.2, 0.3],
            [0.0, 0.0, -0.6],
            [0.1, 0.0, 0.0],
        ]),
    }


def test_order_is_dfs_not_index_order():
    assert tpose_dfs_order(_entry()['parents']) == [0, 1, 2, 4, 3]


def test_order_refuses_a_second_root():
    with pytest.raises(ValueError, match='one root'):
        tpose_dfs_order(np.array([-1, 0, -1]))


def test_bvh_names_are_single_tokens_and_frame_is_complete():
    text, order = tpose_bvh_text(_entry())
    assert 'JOINT Helper_Socket' in text
    motion = text.split('MOTION\n', 1)[1].splitlines()
    values = motion[2].split()
    assert motion[0] == 'Frames: 1'
    assert len(values) == 3 + 3 * len(order)
    assert [float(v) for v in values[:3]] == [0.0, 1.0, 0.0]


def test_bvh_reproduces_rest_positions(tmp_path):
    from motion_lib import BVH
    from motion_lib.Animation import positions_global

    entry = _entry()
    path = tmp_path / 'bvh_tpose' / 't.bvh'
    order = write_tpose_bvh(entry, path)
    anim, _, _ = BVH.load(str(path), collapse_root=False)
    parents = entry['parents']
    expected = np.zeros_like(entry['offsets'])
    for index, parent in enumerate(parents):
        expected[index] = entry['offsets'][index] + (expected[parent] if parent >= 0 else 0.0)
    np.testing.assert_allclose(positions_global(anim)[0], expected[order], atol=1e-5)
