from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.truebones.truebones_utils.get_opt import get_opt
from data_loaders.truebones.truebones_utils.physics_joint_annotation import (
    _infer_symmetry_metadata,
    _rest_positions_from_offsets,
)


def test_horse_front_helper_bones_are_paired() -> None:
    opt = get_opt(None)
    cond = np.load(opt.cond_file, allow_pickle=True).item()['Horse']

    joint_names = list(cond['joints_names'])
    parents = np.asarray(cond['parents'], dtype=np.int64)
    offsets = np.asarray(cond['offsets'], dtype=np.float64)
    rest_positions = _rest_positions_from_offsets(offsets, parents)

    joint_side_labels, symmetry_partner_indices, _pairs = _infer_symmetry_metadata(
        joint_names,
        parents,
        rest_positions,
    )

    expected_pairs = {
        'Bip01_R_Hand': ('Bip01_L_Hand', 'right', 'left'),
        'Bip01_R_Finger0': ('Bip01_L_Finger0', 'right', 'left'),
        'Bip01_Xtra02': ('Bip01_Xtra01', 'right', 'left'),
    }

    index_by_name = {name: index for index, name in enumerate(joint_names)}

    # Verify Xtra02Nub and Xtra01Nub are NOT paired (different signatures: "xtra02 nub" vs "xtra01 nub")
    for nub_name in ['Bip01_Xtra02Nub', 'Bip01_Xtra01Nub']:
        nub_index = index_by_name[nub_name]
        assert symmetry_partner_indices[nub_index] == -1, (
            f'{nub_name} should NOT be paired (different signature from its mirror counterpart), '
            f'got {symmetry_partner_indices[nub_index]}'
        )

    for source_name, (partner_name, source_side, partner_side) in expected_pairs.items():
        source_index = index_by_name[source_name]
        partner_index = index_by_name[partner_name]
        assert symmetry_partner_indices[source_index] == partner_index, (
            f'{source_name} should pair with {partner_name}, '
            f'got {symmetry_partner_indices[source_index]}'
        )
        assert symmetry_partner_indices[partner_index] == source_index, (
            f'{partner_name} should pair with {source_name}, '
            f'got {symmetry_partner_indices[partner_index]}'
        )
        assert joint_side_labels[source_index] == source_side, f'{source_name} should be labeled {source_side}'
        assert joint_side_labels[partner_index] == partner_side, f'{partner_name} should be labeled {partner_side}'


def test_conservative_fallback_rejects_non_mirrored_unique_children() -> None:
    joint_names = [
        'Root',
        'LeftShoulder',
        'RightShoulder',
        'Xtra01',
        'Xtra02',
    ]
    parents = np.asarray([-1, 0, 0, 1, 2], dtype=np.int64)
    rest_positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.4, -0.2, 0.1],
            [1.4, 0.5, 0.7],
        ],
        dtype=np.float64,
    )

    details = _infer_symmetry_metadata(joint_names, parents, rest_positions, return_details=True)

    assert details['symmetry_partner_indices'][3] == -1
    assert details['symmetry_partner_indices'][4] == -1
    assert details['mirror_disabled_joint_indices'] == [3, 4]
    assert details['mirror_disabled_warnings'], 'expected a conservative mirror warning for unresolved unique children'


def test_conservative_fallback_disables_ambiguous_child_subtrees() -> None:
    joint_names = [
        'Root',
        'LeftShoulder',
        'RightShoulder',
        'Xtra01',
        'Xtra02',
        'Xtra01',
        'Xtra02',
    ]
    parents = np.asarray([-1, 0, 0, 1, 1, 2, 2], dtype=np.int64)
    rest_positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.3, -0.1, 0.1],
            [-1.7, -0.2, 0.2],
            [1.3, -0.1, 0.1],
            [1.7, -0.2, 0.2],
        ],
        dtype=np.float64,
    )

    details = _infer_symmetry_metadata(joint_names, parents, rest_positions, return_details=True)

    for joint_index in (3, 4, 5, 6):
        assert details['symmetry_partner_indices'][joint_index] == -1
    assert details['mirror_disabled_joint_indices'] == [3, 4, 5, 6]
    assert details['mirror_disabled_warnings'], 'expected warning for ambiguous child subtrees'


def main() -> None:
    test_horse_front_helper_bones_are_paired()
    test_conservative_fallback_rejects_non_mirrored_unique_children()
    test_conservative_fallback_disables_ambiguous_child_subtrees()
    print('horse symmetry metadata regression: ok')


if __name__ == '__main__':
    main()