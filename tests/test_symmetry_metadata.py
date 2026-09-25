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


def test_rig_prefix_then_a_word_starting_with_r_or_l_is_not_a_side() -> None:
    # Bear "NPC_Ribcage"/"NPC_LowerFrontLip", Horse "BN_Reins_01", Monkey
    # "BN_Lip_01" are all on the midline; the bare side letter still reads.
    assert detect_joint_side('NPC_Ribcage') is None
    assert detect_joint_side('NPC_LowerFrontLip') is None
    assert detect_joint_side('BN_Reins_01') is None
    assert detect_joint_side('BN_Lip_01') is None
    assert detect_joint_side('NPC_R_Thigh') == 'right'
    assert detect_joint_side('BN_L_Arm') == 'left'
    assert detect_joint_side('Bip01_R_Thigh') == 'right'
    assert detect_joint_side('NPC_LowerLeftLip') == 'left'


def test_piers_typo_pairs_with_pliers() -> None:
    # Centipede: "BN_Piers_L_01" is the left twin of "BN_Pliers_R_01".
    assert _joint_signature('BN_Piers_L_01') == _joint_signature('BN_Pliers_R_01')


def test_lm_rm_on_a_head_reads_as_the_mouth_corners() -> None:
    # SabreToothTiger's "Sabrecat_Head_LM01_"/"_RM01_" sit mirrored beside the
    # cheeks: the side letter leads, the head word says M is the mouth.
    assert detect_joint_side('Sabrecat_Head_LM01_') == 'left'
    assert detect_joint_side('Sabrecat_Head_RM01_') == 'right'
    assert _joint_signature('Sabrecat_Head_LM01_') == _joint_signature('Sabrecat_Head_RM01_')
    # Without a head or limb word the code stays unread.
    assert detect_joint_side('RigLM01') is None


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


def test_named_sides_with_mismatched_signatures_pair_by_geometry() -> None:
    # FireAnt keeps the Biped Ponytail index in the name, so the two antennae
    # never share a signature. Same parent, same depth, mirrored rest pose:
    # the geometry fallback pairs the roots and the chain follows.
    joint_names = [
        'Bip01_Head_Brain',
        'Bip01_Ponytail2_R_Antenna1', 'Bip01_Ponytail21_R_Antenna2',
        'Bip01_Ponytail1_L_Antenna', 'Bip01_Ponytail11L_Antenna',
        'Bip01_Ponytail4R_Mandible', 'Bip01_Ponytail3L_Mandible_',
    ]
    parents = np.array([-1, 0, 1, 0, 3, 0, 0])
    rest_positions = np.array([
        [0.0, 0.0, 0.0],
        [-0.06, -0.14, 0.19], [-0.15, -0.18, 0.18],
        [0.06, -0.14, 0.19], [0.15, -0.17, 0.18],
        [-0.066, -0.167, 0.153], [0.064, -0.167, 0.153],
    ])
    _sides, partners, _pairs = _infer_symmetry_metadata(joint_names, parents, rest_positions)
    assert partners[1] == 3 and partners[3] == 1
    assert partners[2] == 4 and partners[4] == 2
    assert partners[5] == 6 and partners[6] == 5


def test_geometry_fallback_never_pairs_a_non_mirrored_or_centre_joint() -> None:
    joint_names = ['Head', 'Horn1_L', 'Horn2_R', 'Crest']
    parents = np.array([-1, 0, 0, 0])
    # Horn_R sits far off the mirror image of Horn_L; Crest names no side.
    rest_positions = np.array([
        [0.0, 0.0, 0.0], [0.1, 0.1, 0.0], [-0.1, -0.3, 0.4], [-0.1, 0.1, 0.0],
    ])
    sides, partners, _pairs = _infer_symmetry_metadata(joint_names, parents, rest_positions)
    assert partners[1] == -1 and partners[2] == -1
    assert sides[3] == 'center' and partners[3] == -1


def _leopard_like_rig(mane_names):
    # Ten named pairs vote that left is +X; the two mane joints are siblings
    # mirrored across the midline.
    parts = ['Arm', 'Leg', 'Ear', 'Toe', 'Hand', 'Foot', 'Eye', 'Wing', 'Horn', 'Fin']
    joint_names = ['Spine', *[f'{side}_{part}' for part in parts for side in 'LR'], *mane_names]
    parents = np.array([-1] + [0] * (len(joint_names) - 1))
    rest_positions = np.array([
        [0.0, 0.0, 0.0],
        *[[sign * (0.2 + 0.05 * k), 0.1 * k, 0.1] for k in range(len(parts)) for sign in (1.0, -1.0)],
        [-0.08, -0.095, 0.221], [0.08, -0.095, 0.221],
    ])
    return _infer_symmetry_metadata(joint_names, parents, rest_positions)


def test_mirror_sibling_named_on_the_wrong_side_is_relabelled() -> None:
    # Leopard: "BN_Mane_R_02" sits at +X where the rest of the rig keeps its left.
    sides, partners, _pairs = _leopard_like_rig(['BN_Mane_R_01', 'BN_Mane_R_02'])
    assert sides[-2] == 'right'
    assert sides[-1] == 'left'
    assert partners[-2] == len(partners) - 1 and partners[-1] == len(partners) - 2


def test_wrong_side_relabel_needs_a_twin_of_the_same_part() -> None:
    # Spider: a jaw is not overruled by a fang that merely mirrors it.
    sides, _partners, _pairs = _leopard_like_rig(['FangR_00_', 'R_Jaw_'])
    assert sides[-2] == 'right'
    assert sides[-1] == 'right'


def test_mirror_pair_named_the_wrong_way_round_is_swapped() -> None:
    # Spider: "R_Jaw_" sits at the rig's left, "L_Jaw_" at its right.
    sides, partners, _pairs = _leopard_like_rig(['L_Jaw_', 'R_Jaw_'])
    assert sides[-2] == 'right'
    assert sides[-1] == 'left'
    assert partners[-2] == len(partners) - 1 and partners[-1] == len(partners) - 2


def _unsided_twin_rig(unsided_name, unsided_half=1.0):
    # serpent_man: "R_Arm_Shoulder" and a plain "Arm_Shoulder" hang off the
    # spine; only the right one names its side. A blade rides in the right hand.
    joint_names = [
        'spine_high',
        unsided_name, 'L_Arm_Up', 'L_Arm_Low', 'L_Arm_Hand', 'L_Finger',
        'R_Arm_Shoulder', 'R_Arm_Up', 'R_Arm_Low', 'R_Arm_Hand', 'R_Finger', 'blade',
    ]
    parents = np.array([-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 9])
    rest_positions = np.array([
        [0.0, 0.0, 0.0],
        *[[unsided_half * x, y, z] for x, y, z in (
            (0.155, 0.72, -0.28), (0.30, 0.72, -0.30), (0.45, 0.22, -0.41),
            (0.57, -0.25, -0.13), (0.60, -0.40, -0.10),
        )],
        [-0.155, 0.72, -0.28], [-0.30, 0.72, -0.30], [-0.45, 0.22, -0.41],
        [-0.57, -0.25, -0.13], [-0.60, -0.40, -0.10], [-0.60, -0.50, 0.10],
    ])
    return _infer_symmetry_metadata(joint_names, parents, rest_positions)


def test_unsided_twin_of_a_named_side_joint_takes_the_other_side() -> None:
    sides, partners, _pairs = _unsided_twin_rig('Arm_Shoulder')
    assert sides[1] == 'left' and sides[6] == 'right'
    assert partners[1] == 6 and partners[6] == 1
    assert partners[2] == 7 and partners[5] == 10


def test_unsided_twin_needs_the_same_name_and_the_other_half() -> None:
    # A different part, or the same part on the named joint's own half, stays 'center'.
    sides, partners, _pairs = _unsided_twin_rig('Arm_Collar')
    assert sides[1] == 'center' and partners[1] == -1
    sides, partners, _pairs = _unsided_twin_rig('Arm_Shoulder', unsided_half=-1.0)
    assert sides[1] == 'center' and partners[1] == -1


def main() -> None:
    test_horse_front_helper_bones_are_paired()
    test_conservative_fallback_rejects_non_mirrored_unique_children()
    test_conservative_fallback_disables_ambiguous_child_subtrees()
    test_lf_rf_suffixes_drive_side_detection_and_signature_normalization()
    test_lf_rf_suffix_children_are_paired()
    test_swapped_limb_code_is_read_only_next_to_a_limb_word()
    test_rig_prefix_then_a_word_starting_with_r_or_l_is_not_a_side()
    test_piers_typo_pairs_with_pliers()
    test_lm_rm_on_a_head_reads_as_the_mouth_corners()
    test_glued_side_letter_on_a_wing_drives_side_detection()
    test_mirrored_name_typo_still_pairs_with_its_twin()
    test_named_sides_with_mismatched_signatures_pair_by_geometry()
    test_geometry_fallback_never_pairs_a_non_mirrored_or_centre_joint()
    test_mirror_sibling_named_on_the_wrong_side_is_relabelled()
    test_wrong_side_relabel_needs_a_twin_of_the_same_part()
    test_mirror_pair_named_the_wrong_way_round_is_swapped()
    test_unsided_twin_of_a_named_side_joint_takes_the_other_side()
    test_unsided_twin_needs_the_same_name_and_the_other_half()
    print('horse symmetry metadata regression: ok')


if __name__ == '__main__':
    main()
