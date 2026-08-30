from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.truebones.truebones_utils.get_opt import get_opt
from data_loaders.truebones.truebones_utils.cond_schema import load_cond
from data_loaders.truebones.truebones_utils.dataset_sources import resolve_species_key
from data_loaders.truebones.truebones_utils.physics_joint_annotation import (
    detect_joint_side,
    _infer_symmetry_metadata,
    _joint_signature,
    rest_positions_from_offsets,
)


def test_horse_front_helper_bones_are_paired() -> None:
    opt = get_opt(None)
    cond_dict = load_cond(opt.cond_file)
    cond = cond_dict[resolve_species_key(cond_dict, 'Horse')]

    joint_names = list(cond['joints_names'])
    parents = np.asarray(cond['parents'], dtype=np.int64)
    offsets = np.asarray(cond['offsets'], dtype=np.float64)
    rest_positions = rest_positions_from_offsets(offsets, parents)

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


def test_lf_rf_suffixes_drive_side_detection_and_signature_normalization() -> None:
    assert detect_joint_side('Sabrecat_Finger4_LF04_') == 'left'
    assert detect_joint_side('Sabrecat_Finger4_RF04_') == 'right'
    assert detect_joint_side('Sabrecat_LeftFinger3_RF30_') == 'left'
    assert detect_joint_side('Sabrecat_RightFinger3_LF30_') == 'right'

    assert _joint_signature('Sabrecat_LeftFinger1_LF10_') == _joint_signature('Sabrecat_RightFinger1_RF10_')
    assert _joint_signature('Sabrecat_Finger4_LF04_') == _joint_signature('Sabrecat_Finger4_RF04_')


def test_lb_rb_suffixes_drive_side_detection_without_crossing_fore_and_hind() -> None:
    # Only the fore codes were ever read, so every Lb*/Rb* hind-limb joint in
    # Bear, Dinosaur, Tiger, antilope and rhino came back 'center' and lost its
    # side along with its symmetry pairing.
    assert detect_joint_side('LbLeg01') == 'left'
    assert detect_joint_side('RbLeg01') == 'right'
    assert detect_joint_side('RbClaw4') == 'right'

    assert _joint_signature('LbLeg01') == _joint_signature('RbLeg01')
    # The side half of the code is dropped, the fore/hind half is not: a fore leg
    # and a hind leg must not land in one symmetry group.
    assert _joint_signature('LfLeg01') != _joint_signature('LbLeg01')


def test_lb_rb_hind_limbs_pair_with_each_other_not_with_the_fore_limbs() -> None:
    joint_names = ['Root', 'LfLeg01', 'RfLeg01', 'LbLeg01', 'RbLeg01']
    parents = np.asarray([-1, 0, 0, 0, 0], dtype=np.int64)
    rest_positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [-1.0, 0.0, -2.0],
            [1.0, 0.0, -2.0],
        ],
        dtype=np.float64,
    )

    joint_side_labels, symmetry_partner_indices, _pairs = _infer_symmetry_metadata(
        joint_names,
        parents,
        rest_positions,
    )

    assert joint_side_labels == ['center', 'left', 'right', 'left', 'right']
    assert symmetry_partner_indices[1] == 2, f'fore pair: {symmetry_partner_indices[1]}'
    assert symmetry_partner_indices[3] == 4, f'hind pair: {symmetry_partner_indices[3]}'


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


def test_swapped_limb_code_is_read_only_next_to_a_limb_word() -> None:
    # Fl/Fr/Bl/Br is the same fore/hind code with the halves swapped, and Lm/Rm
    # is the middle pair of a hexapod. A whole quadruped ("FlLeg1".."BrLegFoot2")
    # came back 'center' and formed no symmetry pairs at all without these.
    assert detect_joint_side('FlLeg1') == 'left'
    assert detect_joint_side('FrLegAnkle') == 'right'
    assert detect_joint_side('BlLegFoot1') == 'left'
    assert detect_joint_side('BrLeg2') == 'right'
    assert detect_joint_side('LmLegAnkle') == 'left'
    assert detect_joint_side('RmLeg1') == 'right'

    # The gate: without a limb word the code is ambiguous. "MouthBL" is the
    # bottom-left corner of a worm's mouth, not a back-left limb.
    assert detect_joint_side('RigMouthBL') is None
    assert detect_joint_side('RigMouthTR') is None

    # Side half dropped, fore/hind/middle half kept, same as Lf/Rf/Lb/Rb.
    assert _joint_signature('FlLeg1') == _joint_signature('FrLeg1')
    assert _joint_signature('LmLeg1') == _joint_signature('RmLeg1')
    assert _joint_signature('FlLeg1') != _joint_signature('BlLeg1')
    assert _joint_signature('FlLeg1') != _joint_signature('LmLeg1')


def test_glued_side_letter_on_a_wing_drives_side_detection() -> None:
    # "Lwing1" has no case boundary after the side letter, so neither the marker
    # list nor the compound splitter saw it and the bee's ten wing joints stayed
    # 'center' while its "LBackArm1" limbs paired normally.
    assert detect_joint_side('RigLwing1') == 'left'
    assert detect_joint_side('RigRwing5') == 'right'
    assert _joint_signature('RigLwing1') == _joint_signature('RigRwing1')


def test_mirrored_name_typo_still_pairs_with_its_twin() -> None:
    # One pack mirrored its left bones and ran a global L -> R replace over the
    # copied names, corrupting the words: "Lower_Arm_L" against "Rower_Arm_R",
    # "Upper_Leg_L" against "Upper_Reg_R". The signature is a spelling key, so
    # the two halves of one limb stopped matching and the pairs never formed.
    assert _joint_signature('Lower_Arm_L') == _joint_signature('Rower_Arm_R')
    assert _joint_signature('Upper_Leg_L') == _joint_signature('Upper_Reg_R')
    assert _joint_signature('Lower_Leg_L') == _joint_signature('Rower_Reg_R')

    joint_names = ['Hips', 'Upper_Leg_L', 'Lower_Leg_L', 'Upper_Reg_R', 'Rower_Reg_R']
    parents = np.asarray([-1, 0, 1, 0, 3], dtype=np.int64)
    rest_positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [-1.0, -1.0, 0.0],
            [-1.0, -2.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, -2.0, 0.0],
        ],
        dtype=np.float64,
    )

    _sides, symmetry_partner_indices, _pairs = _infer_symmetry_metadata(
        joint_names,
        parents,
        rest_positions,
    )

    assert symmetry_partner_indices[1] == 3, f'upper leg pair: {symmetry_partner_indices[1]}'
    assert symmetry_partner_indices[2] == 4, f'lower leg pair: {symmetry_partner_indices[2]}'


def main() -> None:
    test_horse_front_helper_bones_are_paired()
    test_conservative_fallback_rejects_non_mirrored_unique_children()
    test_conservative_fallback_disables_ambiguous_child_subtrees()
    test_lf_rf_suffixes_drive_side_detection_and_signature_normalization()
    test_lf_rf_suffix_children_are_paired()
    test_swapped_limb_code_is_read_only_next_to_a_limb_word()
    test_glued_side_letter_on_a_wing_drives_side_detection()
    test_mirrored_name_typo_still_pairs_with_its_twin()
    print('horse symmetry metadata regression: ok')


if __name__ == '__main__':
    main()