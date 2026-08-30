"""Long-tail hygiene in the per-joint T5 text.

The model never sees a joint's raw name -- it sees the sentence
``build_joint_embedding_texts`` writes and T5 encodes. These cover the four
rewrites that keep a body part at *one* point in that space instead of a dozen:
species words dropped, anatomical synonyms folded onto the corpus vocabulary,
quadruped limb codes decoded, and rig props blanked outright.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.truebones.truebones_utils.physics_joint_annotation import (
    build_joint_embedding_texts,
    build_semantic_metadata,
)


def _embedding_texts(joint_names, parents, offsets=None, species_name=None):
    parents = np.asarray(parents, dtype=np.int64)
    if offsets is None:
        # A plain chain down -Y, mirrored on X for the Left/Right pairs, is
        # enough for the side/contact annotations the texts are built on.
        offsets = np.zeros((len(joint_names), 3), dtype=np.float64)
        for index, name in enumerate(joint_names):
            offsets[index] = [
                -1.0 if 'Left' in name or name.startswith('Lf') or name.startswith('Lb')
                else (1.0 if 'Right' in name or name.startswith('Rf') or name.startswith('Rb') else 0.0),
                -1.0,
                0.0,
            ]
    metadata = build_semantic_metadata(
        joint_names=joint_names,
        parents=parents,
        offsets=np.asarray(offsets, dtype=np.float64),
        species_name=species_name,
    )
    object_cond = dict(metadata)
    object_cond['joints_names'] = list(joint_names)
    object_cond['parents'] = parents
    if species_name:
        object_cond['species_name'] = species_name
    return build_joint_embedding_texts(object_cond)


def test_skeleton_wide_species_prefix_is_removed_from_embedding_text():
    texts = _embedding_texts(
        ['Caveman Pelvis', 'Caveman Spine', 'Caveman Head'],
        [-1, 0, 1],
        species_name='IAC_Caveman',
    )

    assert all('Caveman' not in text for text in texts), texts
    assert texts[0] == 'Pelvis'
    assert texts[1] == 'Spine'
    assert texts[2].startswith('Head'), texts[2]


def test_species_word_is_dropped_so_variant_heads_share_one_anatomy():
    # antilope carries four interchangeable heads on one skeleton. Keeping the
    # species word made T5 cluster them by creature -- "Moose Neck" neighboured
    # "Right Quilin Moustache" -- instead of by body part.
    texts = _embedding_texts(
        ['Root', 'Spine01', 'DeerNeck01', 'MooseNeck', 'QuilinNeck01', 'DonkeyNeck01'],
        [-1, 0, 1, 1, 1, 1],
    )
    necks = texts[2:]
    assert all(text.startswith('Neck') for text in necks), necks
    assert all('Deer' not in text and 'Moose' not in text for text in necks), necks
    # Collapsing them onto one anatomy is exactly what the instance ordinals are
    # for, so they stay individually addressable.
    assert len({text for text in necks}) == len(necks), necks


def test_species_word_is_kept_when_it_is_the_only_word_left():
    texts = _embedding_texts(['Root', 'Dragon'], [-1, 0])
    assert texts[1].startswith('Dragon'), texts[1]


def test_anatomical_synonyms_fold_onto_the_corpus_vocabulary():
    # Hyena's fore limb, which T5 otherwise neighbours by spelling: "Left Ulna"
    # landed on "Left Clavicle" and "Left Carpal" on "Left Calf".
    texts = _embedding_texts(
        ['Root', 'Ribcage', 'LeftScapula', 'LeftHumerus', 'LeftUlna', 'LeftCarpal'],
        [-1, 0, 1, 2, 3, 4],
    )
    assert texts[1] == 'Chest'
    assert texts[2] == 'Left Clavicle'
    assert texts[3] == 'Left UpperArm'
    assert texts[4] == 'Left Forearm'
    assert texts[5].startswith('Left Hand')


def test_echoed_rig_abbreviation_collapses_instead_of_doubling_the_part():
    # SabreToothTiger spells the part twice ("LeftThighLeftThi", "SpineSpn0");
    # expanding the abbreviation lets the adjacent-duplicate collapse eat it.
    texts = _embedding_texts(
        ['Hips', 'SpineSpn0', 'NeckNek0', 'LeftThighLeftThi', 'LeftCalfLeftClf'],
        [-1, 0, 1, 0, 3],
    )
    assert texts[1] == 'Spine'
    assert texts[2].startswith('Neck'), texts[2]
    assert texts[3] == 'Left Thigh'
    assert texts[4].startswith('Left Calf')


def test_limb_code_keeps_fore_hind_apart_while_side_comes_from_geometry():
    texts = _embedding_texts(
        ['Root', 'LfLeg01', 'RfLeg01', 'LbLeg01', 'RbLeg01'],
        [-1, 0, 0, 0, 0],
        offsets=[[0, 0, 0], [-1, -1, 2], [1, -1, 2], [-1, -1, -2], [1, -1, -2]],
    )
    assert texts[1].startswith('Left Front Leg'), texts[1]
    assert texts[2].startswith('Right Front Leg'), texts[2]
    assert texts[3].startswith('Left Back Leg'), texts[3]
    assert texts[4].startswith('Right Back Leg'), texts[4]


def test_horse_link_is_named_for_where_it_sits_not_for_a_horse():
    # 3ds Max Biped's extra leg link, exported by 33 species here including a
    # Cat and a Chicken. It sits Thigh -> Calf -> HorseLink -> Foot.
    texts = _embedding_texts(
        ['Hips', 'LeftThigh', 'LeftCalf', 'LeftHorseLink', 'LeftFoot'],
        [-1, 0, 1, 2, 3],
    )
    assert texts[3].startswith('Left Ankle'), texts[3]


def test_props_and_controls_are_blanked_but_anatomy_in_the_same_name_survives():
    texts = _embedding_texts(
        ['Hips', 'Spine1', 'Saddle', 'Reins01', 'Ctrl', 'MagicEffectsNode', 'XtraSpine'],
        [-1, 0, 1, 1, 0, 1, 1],
    )
    assert texts[2] == ''
    assert texts[3] == ''
    assert texts[4] == ''
    assert texts[5] == ''
    # "Xtra" is a rig marker, "Spine" is not: the joint keeps the anatomy and
    # joins the spine chain.
    assert texts[6].startswith('Spine'), texts[6]


def test_misspellings_and_standalone_abbreviations_fold_onto_the_real_word():
    texts = _embedding_texts(
        ['Hips', 'Pelv', 'LeftScap', 'LeftClav', 'LeftShin', 'Scull', 'Thouge01'],
        [-1, 0, 1, 1, 1, 1, 5],
    )
    assert texts[1].startswith('Pelvis'), texts[1]
    assert texts[2].startswith('Left Clavicle'), texts[2]
    assert texts[3].startswith('Left Clavicle'), texts[3]
    assert texts[4].startswith('Left Calf'), texts[4]
    assert texts[5].startswith('Head'), texts[5]
    assert texts[6].startswith('Tongue'), texts[6]


def test_eye_lid_pair_stays_an_eyelid_instead_of_an_eye_plus_a_lid():
    texts = _embedding_texts(['Head', 'HeadEyeLidHelt', 'GorillaEyeLids'], [-1, 0, 0])
    assert texts[1].startswith('Head Eyelid'), texts[1]
    assert texts[2].startswith('Eyelid'), texts[2]


def test_a_name_made_only_of_markers_blanks_through_punctuation_and_indices():
    # The fallback hands back the raw canonical tokens, so the blanking check
    # has to clean them the same way the table lookups do.
    texts = _embedding_texts(['Hips', 'Bip01', 'BN_P', 'All', 'Ponitail'], [-1, 0, 0, 0, 0])
    assert texts[1:] == ['', '', '', ''], texts


def test_aux_helper_keeps_the_anatomy_it_qualifies():
    texts = _embedding_texts(['Spine1', 'LeftClavicleAux', 'LeftCheekAux'], [-1, 0, 0])
    assert texts[1].startswith('Left Clavicle'), texts[1]
    assert texts[2].startswith('Left Cheek'), texts[2]


def test_anaconda_spline_trunk_is_a_spine_not_a_control():
    texts = _embedding_texts(
        ['Hips', 'Spline01', 'Spline02', 'Neck'],
        [-1, 0, 1, 2],
    )
    assert texts[1].startswith('Spine'), texts[1]
    assert texts[2].startswith('Spine'), texts[2]


def test_swapped_limb_code_decodes_only_when_the_name_spells_a_limb():
    texts = _embedding_texts(
        ['Root', 'FlLeg1', 'FrLeg1', 'BlLeg1', 'BrLeg1', 'LmLeg1', 'RmLeg1'],
        [-1, 0, 0, 0, 0, 0, 0],
        offsets=[[0, 0, 0], [-1, -1, 2], [1, -1, 2], [-1, -1, -2], [1, -1, -2],
                 [-1, -1, 0], [1, -1, 0]],
    )
    assert texts[1].startswith('Left Front Leg'), texts[1]
    assert texts[2].startswith('Right Front Leg'), texts[2]
    assert texts[3].startswith('Left Back Leg'), texts[3]
    assert texts[4].startswith('Right Back Leg'), texts[4]
    assert texts[5].startswith('Left Mid Leg'), texts[5]
    assert texts[6].startswith('Right Mid Leg'), texts[6]


def test_swapped_limb_code_is_left_alone_on_a_mouth_corner():
    # MU04_Earthworm names the corners of its mouth MouthTL/TR/BL/BR, where "BL"
    # is bottom-left. Decoding it as a hind limb would invent anatomy, so the
    # code is only read next to a limb word.
    texts = _embedding_texts(
        ['RigBase', 'RigHead', 'RigMouthBL', 'RigMouthTR'],
        [-1, 0, 1, 1],
    )
    assert 'Back' not in texts[2], texts[2]
    assert texts[2].startswith('Mouth'), texts[2]
    assert texts[3].startswith('Mouth'), texts[3]


def test_mirrored_name_typo_folds_onto_the_segment_its_twin_uses():
    # The pack mirrored its left bones and ran a global L -> R replace over the
    # copied names: the left arm is "Lower_Arm_L", the right one "Rower_Arm_R".
    texts = _embedding_texts(
        ['Hips', 'Upper_Arm_L', 'Lower_Arm_L', 'Upper_Arm_R', 'Rower_Arm_R',
         'Upper_Leg_L', 'Lower_Leg_L', 'Upper_Reg_R', 'Rower_Reg_R'],
        [-1, 0, 1, 0, 3, 0, 5, 0, 7],
    )
    assert texts[1].startswith('Left UpperArm'), texts[1]
    assert texts[2].startswith('Left Forearm'), texts[2]
    assert texts[3].startswith('Right UpperArm'), texts[3]
    assert texts[4].startswith('Right Forearm'), texts[4]
    assert texts[5].startswith('Left Thigh'), texts[5]
    assert texts[6].startswith('Left Calf'), texts[6]
    assert texts[7].startswith('Right Thigh'), texts[7]
    assert texts[8].startswith('Right Calf'), texts[8]


def test_glued_side_letter_on_a_wing_leaves_a_plain_wing():
    texts = _embedding_texts(
        ['RigBody', 'RigLwing1', 'RigRwing1'],
        [-1, 0, 0],
        offsets=[[0, 0, 0], [-1, 0, 0], [1, 0, 0]],
    )
    assert texts[1].startswith('Left Wing'), texts[1]
    assert texts[2].startswith('Right Wing'), texts[2]


def test_carried_equipment_is_blanked_but_worn_cloth_keeps_its_word():
    texts = _embedding_texts(
        ['Hips', 'Spine1', 'Sword', 'Shield', 'L_hand_container', 'Bone_Mount',
         'BackpackContainer', 'CapeBack1', 'HeadContainer'],
        [-1, 0, 1, 1, 1, 0, 1, 1, 1],
    )
    assert texts[2] == ''
    assert texts[3] == ''
    assert texts[5] == ''
    assert texts[6] == ''
    # The socket keeps the body part it hangs off; only the socket word goes.
    assert texts[4].startswith('Left Hand'), texts[4]
    assert texts[8].startswith('Head'), texts[8]
    # Cloth is qualified by a direction ("CapeBack"), so blanking the noun would
    # leave a bare "Back" claiming to be the creature's back.
    assert texts[7].startswith('Cape'), texts[7]


def test_a_name_that_is_only_a_side_and_indices_does_not_say_the_side_twice():
    # spider_tarantula names its leg segments "R4_00".."L2_03". Everything in
    # them is dropped -- the side word by the side table, the two index runs as
    # digits -- so they fall back to the raw canonical tokens, which used to put
    # the side word back and leave "Right Right 4 00". The indices stay: they are
    # the only thing telling leg 4 from leg 2.
    texts = _embedding_texts(
        ['Body', 'L2_00', 'R2_00', 'L4_03', 'R4_03'],
        [-1, 0, 0, 0, 0],
        offsets=[[0, 0, 0], [-1, 0, 1], [1, 0, 1], [-1, 0, -1], [1, 0, -1]],
    )
    assert texts[1].split().count('Left') == 1, texts[1]
    assert texts[2].split().count('Right') == 1, texts[2]
    assert texts[1].startswith('Left 2'), texts[1]
    assert texts[3].startswith('Left 4'), texts[3]
    assert texts[4].startswith('Right 4'), texts[4]


def test_digit_becomes_a_finger_or_a_toe_depending_on_the_limb_it_hangs_off():
    # "Digit" is the anatomical word for both, and this corpus uses it for both.
    # MU01_Bird is the control: wing digits under a palm, toes under an ankle,
    # one skeleton, same spelling.
    texts = _embedding_texts(
        ['Ribcage', 'LeftArm1', 'LeftArmPalm', 'LeftArmDigit21',
         'LeftLeg1', 'LeftLegAnkle', 'LeftLegDigit11'],
        [-1, 0, 1, 2, 0, 4, 5],
    )
    assert texts[3].startswith('Left Arm Finger'), texts[3]
    assert texts[6].startswith('Left Leg Toe'), texts[6]


def test_digit_keeps_the_bare_word_when_the_name_names_no_limb():
    # Only an unambiguous single side of the fork is read -- guessing between a
    # finger and a toe is worse than an uninformative "Digit".
    texts = _embedding_texts(['Body', 'Digit01', 'ArmLegDigit01'], [-1, 0, 0])
    assert texts[1].startswith('Digit'), texts[1]
    assert 'Digit' in texts[2], texts[2]
