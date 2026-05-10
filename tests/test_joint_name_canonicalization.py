import json
import numpy as np
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.truebones.truebones_utils.motion_process import (
    _collect_joint_name_collision_groups,
    _canonical_name_for_bvh,
    _refresh_joint_metadata_in_object_cond,
    _write_joint_name_collision_report,
)
from data_loaders.truebones.truebones_utils.physics_joint_annotation import _build_semantic_metadata


def test_tai_tokens_are_canonicalized_to_tail_bvh_names():
    metadata = _build_semantic_metadata(
        joint_names=["Bip01_Pelvis", "BN_Tai01", "BN_Tai02"],
        parents=np.array([-1, 0, 1], dtype=np.int64),
        offsets=np.zeros((3, 3), dtype=np.float64),
    )

    assert metadata["canonical_joint_names"][1:] == ["Tail 01", "Tail 02"]
    assert [
        _canonical_name_for_bvh(name, raw_name)
        for name, raw_name in zip(metadata["canonical_joint_names"], ["Bip01_Pelvis", "BN_Tai01", "BN_Tai02"])
    ][1:] == ["Tail01", "Tail02"]


def test_solitary_ear_indices_are_removed_but_tail_chain_indices_remain():
    metadata = _build_semantic_metadata(
        joint_names=["Bip01_Head", "Bip01_R_Ear_01", "Bip01__L_Ear_01", "BN_Tail_01", "BN_Tail_02"],
        parents=np.array([-1, 0, 0, 0, 3], dtype=np.int64),
        offsets=np.zeros((5, 3), dtype=np.float64),
    )

    assert metadata["canonical_joint_names"][1:3] == ["Right Ear", "Left Ear"]
    assert metadata["canonical_joint_names"][3:] == ["Tail 01", "Tail 02"]
    assert [
        _canonical_name_for_bvh(name, raw_name)
        for name, raw_name in zip(
            metadata["canonical_joint_names"],
            ["Bip01_Head", "Bip01_R_Ear_01", "Bip01__L_Ear_01", "BN_Tail_01", "BN_Tail_02"],
        )
    ][1:] == ["RightEar", "LeftEar", "Tail01", "Tail02"]


def test_toe_root_indices_are_preserved_for_parallel_digits():
    metadata = _build_semantic_metadata(
        joint_names=["Bip01_Pelvis", "Bip01_L_Toe2", "Bip01_L_Toe1", "Bip01_L_Toe0"],
        parents=np.array([-1, 0, 0, 0], dtype=np.int64),
        offsets=np.zeros((4, 3), dtype=np.float64),
    )

    assert metadata["canonical_joint_names"][1:] == ["Left Toe 2", "Left Toe 1", "Left Toe 0"]
    assert [
        _canonical_name_for_bvh(name, raw_name)
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

    _refresh_joint_metadata_in_object_cond(object_cond)

    assert object_cond["canonical_joint_names"] == ["Pelvis", "Left Toe 2", "Left Toe 1"]
    assert object_cond["canonical_bvh_joint_names"] == ["Pelvis", "LeftToe2", "LeftToe1"]


def test_refresh_joint_metadata_disambiguates_duplicate_canonical_names():
    object_cond = {
        "object_type": "Scorpion-2",
        "joints_names": ["Hips", "jt_Hips_C", "jt_Tail01_C", "jt_Tail01x_C"],
        "parents": np.array([-1, 0, 1, 2], dtype=np.int64),
        "offsets": np.zeros((4, 3), dtype=np.float64),
    }

    _refresh_joint_metadata_in_object_cond(object_cond)

    assert object_cond["canonical_joint_names"] == ["Hips", "Hips Joint", "Tail 01", "Tail 01 Copy"]
    assert object_cond["canonical_bvh_joint_names"] == ["Hips", "HipsJoint", "Tail01", "Tail01Copy"]


def test_refresh_joint_metadata_disables_unpaired_truncated_helper_mirroring():
    object_cond = {
        "object_type": "SabreToothTiger",
        "joints_names": [
            "Root",
            "Sabrecat_LeftFinger1_LF11_",
            "Sabrecat_RightFinger1_RF11_",
            "Sabrecat_RightFinger1_RF11___rot_helper_2",
        ],
        "parents": np.array([-1, 0, 0, 2], dtype=np.int64),
        "offsets": np.array(
            [
                [0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        "original_joint_count": 3,
        "helper_joint_indices": [3],
        "helper_source_leaf_indices": [2],
        "helper_joint_names": ["Sabrecat_RightFinger1_RF11___rot_helper_2"],
    }

    _refresh_joint_metadata_in_object_cond(object_cond)

    assert object_cond["symmetry_partner_indices"][:3] == [-1, 2, 1]
    assert object_cond["symmetry_partner_indices"][3] == -1
    assert object_cond["mirror_disabled_joint_indices"] == [3]
    assert object_cond["mirror_disabled_joint_names"] == ["Sabrecat_RightFinger1_RF11___rot_helper_2"]
    assert object_cond["mirror_disabled_warnings"]


def test_joint_name_collision_report_is_empty_after_disambiguation():
    object_cond = {
        "object_type": "Scorpion-2",
        "joints_names": ["Hips", "jt_Hips_C", "jt_Tail01_C", "jt_Tail01x_C"],
        "parents": np.array([-1, 0, 1, 2], dtype=np.int64),
        "offsets": np.zeros((4, 3), dtype=np.float64),
    }
    _refresh_joint_metadata_in_object_cond(object_cond)
    cond = {"Scorpion-2": object_cond}

    assert _collect_joint_name_collision_groups(cond) == []

    with tempfile.TemporaryDirectory() as temp_dir:
        report_groups = _write_joint_name_collision_report(cond, temp_dir)
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
        test_refresh_joint_metadata_disables_unpaired_truncated_helper_mirroring,
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