import json
import numpy as np
import pytest
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.truebones.truebones_utils.motion_process import (
    collect_joint_name_collision_groups,
    canonical_name_for_bvh,
    refresh_joint_metadata_in_object_cond,
    write_joint_name_collision_report,
)
from data_loaders.truebones.truebones_utils.animation_utils import (
    _joint_disambiguation_tokens,
)
from data_loaders.truebones.truebones_utils.physics_joint_annotation import build_semantic_metadata
from data_loaders.truebones.truebones_utils.joint_name_canonical import (
    canonicalize_joint_name,
    infer_species_joint_name_prefixes,
    strip_joint_name_prefix,
)


def test_unity_rig_prefixes_are_removed_from_canonical_names():
    raw_names = ["RigPelvis", "RigSpine1", "RigLHand", "RigRHand"]
    metadata = build_semantic_metadata(
        joint_names=raw_names,
        parents=np.array([-1, 0, 1, 1], dtype=np.int64),
        offsets=np.zeros((len(raw_names), 3), dtype=np.float64),
    )

    assert metadata["canonical_joint_names"] == [
        "Pelvis",
        "Spine 1",
        "Left Hand",
        "Right Hand",
    ]


def test_species_prefix_is_derived_from_namespaced_pack_identifier():
    cases = [
        ("unitybundles/IAC_Caveman", ["Caveman Pelvis", "Caveman Spine1"], ["Pelvis", "Spine 1"]),
        ("IAC_Cavewoman", ["Cavewoman Pelvis", "Cavewoman Head"], ["Pelvis", "Head"]),
        ("IAC_Mammoth", ["Mammoth Pelvis", "Mammoth Trunk02"], ["Pelvis", "Trunk 02"]),
        ("IAC_Sabertooth", ["Sabertooth Pelvis", "Sabertooth L Ear"], ["Pelvis", "Left Ear"]),
        ("NEW_SpaceDragon", ["SpaceDragonRoot", "SpaceDragonWing1"], ["Root", "Wing 1"]),
    ]
    for species_name, raw_names, expected in cases:
        metadata = build_semantic_metadata(
            joint_names=raw_names,
            parents=np.array([-1, 0], dtype=np.int64),
            offsets=np.zeros((2, 3), dtype=np.float64),
            species_name=species_name,
        )
        assert metadata["canonical_joint_names"] == expected


def test_species_word_is_not_removed_when_it_is_not_a_skeleton_wide_prefix():
    raw_names = ["Hips", "HorseLink", "HorseHead"]
    metadata = build_semantic_metadata(
        joint_names=raw_names,
        parents=np.array([-1, 0, 1], dtype=np.int64),
        offsets=np.zeros((3, 3), dtype=np.float64),
        species_name="Horse",
    )

    assert metadata["canonical_joint_names"] == ["Hips", "Horse Link", "Horse Head"]


@pytest.mark.parametrize("mount, rider", [("horse", "man"), ("Mount", "Rider")])
def test_composite_body_words_are_stripped_from_rider_and_mount(mount, rider):
    raw_names = [
        "Root",
        f"{mount}Spine", f"{mount}_hand_L", f"{mount}_hand_R",
        f"{rider}_spine", f"{rider}_hand_L", f"{rider}_hand_R", f"{rider}_armor_L",
        "Spear",
    ]
    body_words = tuple(sorted((mount.lower(), rider.lower())))
    assert infer_species_joint_name_prefixes(raw_names, "MLH_Horseman") == body_words
    assert infer_species_joint_name_prefixes(raw_names, None) == body_words
    stripped = [canonicalize_joint_name(name, additional_prefixes=(mount, rider)) for name in raw_names]
    assert stripped == [
        "Root", "Spine", "Hand Left", "Hand Right", "Spine", "Hand Left", "Hand Right", "Armor Left", "Spear",
    ]


@pytest.mark.parametrize("raw_names", [
    # Limb codes: each leads one side only.
    ["Pelvis", "LF_Leg1", "LF_Leg2", "LF_Foot", "RF_Leg1", "RF_Leg2", "RF_Foot"],
    # Fingers: no midline joint.
    ["Hand_L", "Ring_01_L", "Ring_02_L", "Ring_01_R", "Ring_02_R", "Pinky_01_L", "Pinky_01_R", "Pinky_02_L"],
    # One body word is anatomy.
    ["Hips", "HorseLink", "HorseHead", "Horse_Leg_L", "Horse_Leg_R"],
])
def test_composite_body_rule_leaves_other_rigs_alone(raw_names):
    assert infer_species_joint_name_prefixes(raw_names, None) == ()


def _canonical_names(raw_names, species_name=None):
    metadata = build_semantic_metadata(
        joint_names=raw_names,
        parents=np.array([-1] + [0] * (len(raw_names) - 1), dtype=np.int64),
        offsets=np.zeros((len(raw_names), 3), dtype=np.float64),
        species_name=species_name,
    )
    return metadata["canonical_joint_names"]


def test_fbx_namespace_is_removed():
    assert strip_joint_name_prefix("mixamorig:LeftArm") == "LeftArm"
    assert strip_joint_name_prefix("mixamorig7:Hips_01") == "Hips_01"
    assert _canonical_names(["mixamorig:Hips", "mixamorig:LeftUpLeg"]) == ["Hips", "Left Up Leg"]


def test_sketchfab_armature_stamp_is_removed_at_either_end():
    raw_names = [
        "Hawksbill.armature_rootJoint", "Root_Hawksbill.armature",
        "Head.001_Hawksbill.armature", "Hind_L.001_Hawksbill.armature",
    ]
    assert infer_species_joint_name_prefixes(raw_names, "ORA_Hawksbill") == ("Hawksbill.armature",)
    assert _canonical_names(raw_names, "ORA_Hawksbill") == ["Root Joint", "Root", "Head 001", "Hind Left 001"]


def test_trailing_stamp_never_takes_anatomy_or_a_side():
    assert infer_species_joint_name_prefixes(["Arm_L", "Hand_L", "Finger_L"], None) == ()
    assert infer_species_joint_name_prefixes(["Front_Tail", "Back_Tail", "Mid_Tail"], None) == ()


@pytest.mark.parametrize("role", ["IK", "Control", "FX", "Target"])
def test_trailing_joint_role_is_not_treated_as_a_rig_stamp(role):
    raw_names = [f"Root_{role}", f"Hand_{role}", f"Foot_{role}"]
    assert infer_species_joint_name_prefixes(raw_names) == ()
    assert _canonical_names(raw_names) == [f"Root {role.capitalize()}", f"Hand {role.capitalize()}", f"Foot {role.capitalize()}"]


def test_biped_root_name_stamp_extends_the_species_word():
    raw_names = ["Cat Shorthair Pelvis", "Cat Shorthair Spine1", "Cat Shorthair L Thigh", "Cat Shorthair Head"]
    assert _canonical_names(raw_names, "ORA_Cat") == ["Pelvis", "Spine 1", "Left Thigh", "Head"]
    raw_names = ["Base HumanPelvis_01", "Base HumanLThigh_02", "Base HumanSpine1_03", "Base HumanHead_04"]
    assert _canonical_names(raw_names) == ["Pelvis 01", "Left Thigh 02", "Spine 1 03", "Head 04"]


@pytest.mark.parametrize("raw_names", [
    # One body part repeated along a chain.
    ["Root", "Tail1", "Tail2", "Tail3", "Tail4", "Tail5"],
    # A single leading word on most joints is as often anatomy as a stamp.
    ["Root", "Tentacle_A_01", "Tentacle_A_02", "Tentacle_B_01", "Tentacle_B_02", "Tentacle_C_01"],
    # A two-word anatomy run.
    ["Root", "Tail_Fin_A", "Tail_Fin_B", "Tail_Fin_C", "Tail_Fin_D"],
])
def test_multi_word_stamp_leaves_anatomy_chains_alone(raw_names):
    assert infer_species_joint_name_prefixes(raw_names, None) == ()


def test_species_stamp_tolerates_unstamped_extra_bones():
    raw_names = [f"SPARROW_ {part}" for part in ("Pelvis", "Spine", "Neck", "Head", "L Thigh", "R Thigh", "L Foot", "R Foot")]
    raw_names += ["WingLeftFeatherA", "WingRightFeatherA"]
    assert infer_species_joint_name_prefixes(raw_names, "ORA_Sparrow") == ("Sparrow",)
    assert _canonical_names(raw_names, "ORA_Sparrow")[:2] == ["Pelvis", "Spine"]


def test_rig_suffix_and_armature_prefix_are_removed():
    assert _canonical_names(["Hip_JNT", "Neck_1_JNT", "Ear_R_END_JNT"]) == ["Hip", "Neck 1", "Ear Right End"]
    assert strip_joint_name_prefix("Armature_Bone.001") == "_Bone.001"


def test_species_prefix_is_removed_before_duplicate_name_disambiguation():
    object_cond = {
        "object_type": "unitybundles/IAC_Caveman",
        "species_name": "IAC_Caveman",
        "joints_names": ["Caveman Tongue", "Caveman Tongue02"],
        "parents": np.array([-1, 0], dtype=np.int64),
        "offsets": np.zeros((2, 3), dtype=np.float64),
    }

    refresh_joint_metadata_in_object_cond(object_cond)

    assert object_cond["canonical_joint_names"] == ["Tongue", "Tongue 02"]
    assert _joint_disambiguation_tokens(
        "Caveman Tongue02",
        "Tongue",
        additional_prefixes=("Caveman",),
    ) == ["02"]


def test_short_rig_prefixes_only_match_at_identifier_boundaries():
    assert strip_joint_name_prefix("RigHead") == "Head"
    assert strip_joint_name_prefix("Rig_Head") == "_Head"

    assert strip_joint_name_prefix("RightArm") == "RightArm"
    assert strip_joint_name_prefix("RIGHT_Arm") == "RIGHT_Arm"
    assert strip_joint_name_prefix("RigidBody") == "RigidBody"
    assert strip_joint_name_prefix("Belly") == "Belly"
    assert strip_joint_name_prefix("BODY_00") == "BODY_00"


def test_tai_tokens_are_canonicalized_to_tail_bvh_names():
    metadata = build_semantic_metadata(
        joint_names=["Bip01_Pelvis", "BN_Tai01", "BN_Tai02"],
        parents=np.array([-1, 0, 1], dtype=np.int64),
        offsets=np.zeros((3, 3), dtype=np.float64),
    )

    assert metadata["canonical_joint_names"][1:] == ["Tail 01", "Tail 02"]
    assert [
        canonical_name_for_bvh(name, raw_name)
        for name, raw_name in zip(metadata["canonical_joint_names"], ["Bip01_Pelvis", "BN_Tai01", "BN_Tai02"])
    ][1:] == ["Tail01", "Tail02"]


def test_solitary_ear_indices_are_removed_but_tail_chain_indices_remain():
    metadata = build_semantic_metadata(
        joint_names=["Bip01_Head", "Bip01_R_Ear_01", "Bip01__L_Ear_01", "BN_Tail_01", "BN_Tail_02"],
        parents=np.array([-1, 0, 0, 0, 3], dtype=np.int64),
        offsets=np.zeros((5, 3), dtype=np.float64),
    )

    assert metadata["canonical_joint_names"][1:3] == ["Right Ear", "Left Ear"]
    assert metadata["canonical_joint_names"][3:] == ["Tail 01", "Tail 02"]
    assert [
        canonical_name_for_bvh(name, raw_name)
        for name, raw_name in zip(
            metadata["canonical_joint_names"],
            ["Bip01_Head", "Bip01_R_Ear_01", "Bip01__L_Ear_01", "BN_Tail_01", "BN_Tail_02"],
        )
    ][1:] == ["RightEar", "LeftEar", "Tail01", "Tail02"]


def test_toe_root_indices_are_preserved_for_parallel_digits():
    metadata = build_semantic_metadata(
        joint_names=["Bip01_Pelvis", "Bip01_L_Toe2", "Bip01_L_Toe1", "Bip01_L_Toe0"],
        parents=np.array([-1, 0, 0, 0], dtype=np.int64),
        offsets=np.zeros((4, 3), dtype=np.float64),
    )

    assert metadata["canonical_joint_names"][1:] == ["Left Toe 2", "Left Toe 1", "Left Toe 0"]
    assert [
        canonical_name_for_bvh(name, raw_name)
        for name, raw_name in zip(
            metadata["canonical_joint_names"],
            ["Bip01_Pelvis", "Bip01_L_Toe2", "Bip01_L_Toe1", "Bip01_L_Toe0"],
        )
    ][1:] == ["LeftToe2", "LeftToe1", "LeftToe0"]


def test_species_words_are_removed_from_canonical_names_and_disambiguation():
    object_cond = {
        "object_type": "Kappa_gorilla",
        "joints_names": [
            "GorillaJaw",
            "KappaJaw",
            "R_gorilla_finger1_J01",
            "R_gorilla_finger1_J02",
            "L_gorilla_finger5_J02",
            "gorilla_mouth",
            "kappa_neck",
        ],
        "parents": np.array([-1, 0, 0, 0, 0, 0, 0], dtype=np.int64),
        "offsets": np.zeros((7, 3), dtype=np.float64),
    }

    refresh_joint_metadata_in_object_cond(object_cond)

    assert object_cond["canonical_joint_names"] == [
        "Jaw",
        "Jaw Variant2",
        "Right Finger 1 01",
        "Right Finger 1 02",
        "Left Finger 5 02",
        "Mouth",
        "Neck",
    ]
    assert object_cond["canonical_bvh_joint_names"] == [
        "Jaw",
        "JawVariant2",
        "RightFinger101",
        "RightFinger102",
        "LeftFinger502",
        "Mouth",
        "Neck",
    ]


def test_refresh_joint_metadata_rewrites_stale_canonical_names():
    object_cond = {
        "object_type": "Dragon",
        "joints_names": ["Bip01_Pelvis", "Bip01_L_Toe2", "Bip01_L_Toe1"],
        "parents": np.array([-1, 0, 0], dtype=np.int64),
        "offsets": np.zeros((3, 3), dtype=np.float64),
        "canonical_joint_names": ["Pelvis", "Left Toe", "Left Toe"],
        "canonical_bvh_joint_names": ["Pelvis", "LeftToe", "LeftToe"],
    }

    refresh_joint_metadata_in_object_cond(object_cond)

    assert object_cond["canonical_joint_names"] == ["Pelvis", "Left Toe 2", "Left Toe 1"]
    assert object_cond["canonical_bvh_joint_names"] == ["Pelvis", "LeftToe2", "LeftToe1"]


def test_refresh_joint_metadata_disambiguates_duplicate_canonical_names():
    object_cond = {
        "object_type": "Scorpion-2",
        "joints_names": ["Hips", "jt_Hips_C", "jt_Tail01_C", "jt_Tail01x_C"],
        "parents": np.array([-1, 0, 1, 2], dtype=np.int64),
        "offsets": np.zeros((4, 3), dtype=np.float64),
    }

    refresh_joint_metadata_in_object_cond(object_cond)

    assert object_cond["canonical_joint_names"] == ["Hips", "Hips Joint", "Tail 01", "Tail 01 Copy"]
    assert object_cond["canonical_bvh_joint_names"] == ["Hips", "HipsJoint", "Tail01", "Tail01Copy"]


def test_joint_name_collision_report_is_empty_after_disambiguation():
    object_cond = {
        "object_type": "Scorpion-2",
        "joints_names": ["Hips", "jt_Hips_C", "jt_Tail01_C", "jt_Tail01x_C"],
        "parents": np.array([-1, 0, 1, 2], dtype=np.int64),
        "offsets": np.zeros((4, 3), dtype=np.float64),
    }
    refresh_joint_metadata_in_object_cond(object_cond)
    cond = {"Scorpion-2": object_cond}

    assert collect_joint_name_collision_groups(cond) == []

    with tempfile.TemporaryDirectory() as temp_dir:
        report_groups = write_joint_name_collision_report(cond, temp_dir)
        report_path = Path(temp_dir) / "joint_name_collision_report.json"
        assert report_groups == []
        assert report_path.exists()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["num_collision_groups"] == 0


def test_translated_romaji_is_not_reused_as_a_collision_suffix():
    # Pirrana: three anal-fin joints in two romaji spellings all canonicalize to
    # "Anal Fin". The romaji word is a translation, not a distinguishing mark;
    # appended it was translated again into the text "Anal Fin Anal Fin".
    from data_loaders.truebones.truebones_utils.joint_name_canonical import (
        refresh_joint_metadata_in_object_cond,
    )
    from data_loaders.truebones.truebones_utils.joint_embedding_text import (
        build_joint_embedding_texts,
    )
    names = ['mune', 'atama', 'ago', 'kosi', 'shippoA', 'shiribire', 'shirihireB', 'shiribireA']
    cond = {
        'joints_names': names,
        'parents': np.array([-1, 0, 1, 0, 3, 4, 5, 5]),
        'offsets': np.array([[0, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, -1],
                             [0, 0, -1], [0, -1, 0], [0, -1, -1], [0, -1, 1]], dtype=np.float64),
        'species_name': 'Pirrana',
    }
    refresh_joint_metadata_in_object_cond(cond)
    assert cond['canonical_joint_names'][5:] == ['Anal Fin', 'Anal Fin B', 'Anal Fin A']
    texts = build_joint_embedding_texts(cond)
    assert texts[5:] == ['Anal Fin', 'Anal Fin', 'Anal Fin']


if __name__ == "__main__":
    import traceback

    tests = [
        test_tai_tokens_are_canonicalized_to_tail_bvh_names,
        test_solitary_ear_indices_are_removed_but_tail_chain_indices_remain,
        test_toe_root_indices_are_preserved_for_parallel_digits,
        test_refresh_joint_metadata_rewrites_stale_canonical_names,
        test_refresh_joint_metadata_disambiguates_duplicate_canonical_names,
        test_joint_name_collision_report_is_empty_after_disambiguation,
        test_translated_romaji_is_not_reused_as_a_collision_suffix,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {test.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")


def test_annotation_keywords_match_words_not_substrings():
    from data_loaders.truebones.truebones_utils.physics_joint_annotation import _text_matches_keywords

    # A keyword sitting inside another word is not that word.
    assert not _text_matches_keywords('elk r rear hoof', ('ear',))
    assert not _text_matches_keywords('forearm left 04', ('ear',))
    assert not _text_matches_keywords('clip right 02', ('lip',))
    assert not _text_matches_keywords('head container', ('tai',))
    # A word the keyword begins still matches: plurals and glued suffixes.
    assert _text_matches_keywords('toes rear 1 right', ('toe',))
    assert _text_matches_keywords('tailio', ('tail',))
    # A side letter glued in front, for keywords long enough not to collide.
    assert _text_matches_keywords('rig rwing 5', ('wing',))
    assert not _text_matches_keywords('rear', ('ear',))
    # Multi-word keywords match consecutive words.
    assert _text_matches_keywords('toe end site', ('end site',))
    assert not _text_matches_keywords('end of site', ('end site',))


_CONTACT_CASES = [
    # 'Rear' must not read as 'ear'.
    ('truebones/zoo/Deer', {'ElkRRearHoof', 'ElkLRearHoof', 'ElkRFrontHoof', 'ElkLFrontHoof'}, set()),
    # The rear hooves are 'feet', not 'foot'.
    ('unitybundles/MLH_Horseman', {'horse_feetTip_L', 'horse_feetTip_R', 'horse_handTip_L', 'horse_handTip_R'}, set()),
    ('unitybundles/MLS_Dryad', {'horse_feetTip_L', 'horse_feetTip_R', 'horse_handTip_L', 'horse_handTip_R'}, set()),
    # The foot ends in a 'ball' leaf; the hands hang well above the floor.
    ('unitybundles/RTH_Hero', {'ball_l', 'ball_r', 'foot_l', 'foot_r'}, {'hand_l', 'hand_r', 'thumb_03_l', 'thumb_03_r'}),
    # Flippers end in a 'Wrist' / 'Ankle' leaf, all four on the floor.
    ('integrate_20260924/PP_Seal', {'Wrist_R', 'Wrist_L', 'Ankle_R', 'Ankle_L'}, set()),
]


@pytest.mark.parametrize('species, expected, forbidden', _CONTACT_CASES)
def test_contact_prefill_on_real_rigs(species, expected, forbidden):
    cond_path = Path(__file__).resolve().parents[1] / 'dataset' / 'merged' / 'cond.npy'
    if not cond_path.is_file():
        pytest.skip('merged dataset not present')
    from data_loaders.truebones.truebones_utils.cond_schema import load_cond

    from data_loaders.truebones.truebones_utils.joint_parts import prefill_contacts
    from data_loaders.truebones.truebones_utils.physics_joint_annotation import (
        rest_positions_from_offsets,
    )

    entry = load_cond(str(cond_path))[species]
    joint_names = list(entry['joints_names'])
    rest = rest_positions_from_offsets(entry['offsets'], entry['parents'])
    names = {joint_names[index] for index in prefill_contacts(joint_names, entry['parents'], rest)}
    assert expected <= names, sorted(expected - names)
    assert not names & forbidden, sorted(names & forbidden)
