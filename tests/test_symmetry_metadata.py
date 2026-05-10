from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.truebones.truebones_utils.get_opt import get_opt
from data_loaders.truebones.truebones_utils.physics_joint_annotation import (
    _detect_joint_side,
    _infer_symmetry_metadata,
    _joint_signature,
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

    # Xtra01/Xtra02 and their Nub children are structurally mirrored by the
    # child-mirror fallback (geometry check passes), so all four are paired.
    expected_pairs['Bip01_Xtra02Nub'] = ('Bip01_Xtra01Nub', 'right', 'left')

    index_by_name = {name: index for index, name in enumerate(joint_names)}

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


def test_lf_rf_suffixes_drive_side_detection_and_signature_normalization() -> None:
    assert _detect_joint_side('Sabrecat_Finger4_LF04_') == 'left'
    assert _detect_joint_side('Sabrecat_Finger4_RF04_') == 'right'
    assert _detect_joint_side('Sabrecat_LeftFinger3_RF30_') == 'left'
    assert _detect_joint_side('Sabrecat_RightFinger3_LF30_') == 'right'

    assert _joint_signature('Sabrecat_LeftFinger1_LF10_') == _joint_signature('Sabrecat_RightFinger1_RF10_')
    assert _joint_signature('Sabrecat_Finger4_LF04_') == _joint_signature('Sabrecat_Finger4_RF04_')


def test_lf_rf_suffix_children_are_paired() -> None:
    joint_names = [
        'Root',
        'LeftHand',
        'RightHand',
        'Sabrecat_Finger4_LF04_',
        'Sabrecat_Finger4_RF04_',
        'Sabrecat_LeftFinger3_RF30_',
        'Sabrecat_RightFinger3_RF30_',
    ]
    parents = np.asarray([-1, 0, 0, 1, 2, 1, 2], dtype=np.int64)
    rest_positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.3, -0.1, 0.1],
            [1.3, -0.1, 0.1],
            [-1.6, -0.2, 0.2],
            [1.6, -0.2, 0.2],
        ],
        dtype=np.float64,
    )

    joint_side_labels, symmetry_partner_indices, _pairs = _infer_symmetry_metadata(
        joint_names,
        parents,
        rest_positions,
    )

    assert symmetry_partner_indices[3] == 4, f'unexpected LF/RF pair for Finger4: {symmetry_partner_indices[3]}'
    assert symmetry_partner_indices[4] == 3, f'unexpected LF/RF pair for Finger4 mirror: {symmetry_partner_indices[4]}'
    assert symmetry_partner_indices[5] == 6, f'unexpected mixed-token pair for Finger3: {symmetry_partner_indices[5]}'
    assert symmetry_partner_indices[6] == 5, f'unexpected mixed-token pair for Finger3 mirror: {symmetry_partner_indices[6]}'
    assert joint_side_labels[3] == 'left'
    assert joint_side_labels[4] == 'right'
    assert joint_side_labels[5] == 'left'
    assert joint_side_labels[6] == 'right'


def main() -> None:
    test_horse_front_helper_bones_are_paired()
    test_conservative_fallback_rejects_non_mirrored_unique_children()
    test_conservative_fallback_disables_ambiguous_child_subtrees()
    test_lf_rf_suffixes_drive_side_detection_and_signature_normalization()
    test_lf_rf_suffix_children_are_paired()
    print('horse symmetry metadata regression: ok')


if __name__ == '__main__':
    main()