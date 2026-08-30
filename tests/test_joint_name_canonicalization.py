import json
import numpy as np
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
from data_loaders.truebones.truebones_utils.physics_joint_annotation import (
    build_semantic_metadata,
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


if __name__ == "__main__":
    import traceback

    tests = [
        test_tai_tokens_are_canonicalized_to_tail_bvh_names,
        test_solitary_ear_indices_are_removed_but_tail_chain_indices_remain,
        test_toe_root_indices_are_preserved_for_parallel_digits,
        test_refresh_joint_metadata_rewrites_stale_canonical_names,
        test_refresh_joint_metadata_disambiguates_duplicate_canonical_names,
        test_joint_name_collision_report_is_empty_after_disambiguation,
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
