import numpy as np

from data_loaders.truebones.truebones_utils.motion_process import _canonical_name_for_bvh
from data_loaders.truebones.truebones_utils.physics_joint_annotation import _build_semantic_metadata


def test_tai_tokens_are_canonicalized_to_tail_bvh_names():
    metadata = _build_semantic_metadata(
        object_type="Ant",
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
        object_type="Horse",
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