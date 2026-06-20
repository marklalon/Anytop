from types import SimpleNamespace

import numpy as np

from tools.validate_bind_pose import _check_one_source, _filter_motion_files


def _mock_fbx_load(monkeypatch, parents, offsets):
    from motion_lib import FBX

    anim = SimpleNamespace(
        parents=np.asarray(parents, dtype=np.int32),
        offsets=np.asarray(offsets, dtype=np.float64),
    )
    monkeypatch.setattr(
        FBX,
        "load",
        lambda _path: (anim, [f"joint_{i}" for i in range(len(parents))], 1 / 30),
    )


def test_check_source_rejects_fewer_joints_than_cond(tmp_path, monkeypatch):
    source_path = tmp_path / "source.fbx"
    source_path.touch()
    _mock_fbx_load(
        monkeypatch,
        [-1, 0],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    )
    cond_subset = {
        "parents": np.array([-1, 0, 1]),
        "offsets": np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        ),
        "scale_factor": 1.0,
    }

    result = _check_one_source(cond_subset, str(source_path))

    assert result is not None
    assert "joint count 2 differs from cond.npy reference (3)" in result


def test_check_source_accepts_matching_complete_skeleton(tmp_path, monkeypatch):
    source_path = tmp_path / "source.fbx"
    source_path.touch()
    parents = np.array([-1, 0])
    offsets = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    _mock_fbx_load(monkeypatch, parents, offsets)
    cond_subset = {
        "parents": parents,
        "offsets": offsets,
        "scale_factor": 1.0,
    }

    assert _check_one_source(cond_subset, str(source_path)) is None


def test_check_source_accepts_zero_length_bones_without_warning(
    tmp_path, monkeypatch,
):
    source_path = tmp_path / "source.fbx"
    source_path.touch()
    parents = np.array([-1, 0])
    offsets = np.zeros((2, 3))
    _mock_fbx_load(monkeypatch, parents, offsets)
    cond_subset = {
        "parents": parents,
        "offsets": offsets,
        "scale_factor": 1.0,
    }

    with np.errstate(divide="raise", invalid="raise"):
        assert _check_one_source(cond_subset, str(source_path)) is None


def test_check_source_rejects_extra_source_joints_below_crop_cap(
    tmp_path, monkeypatch,
):
    """Preprocessing does not crop merely to match a smaller cond skeleton."""
    source_path = tmp_path / "source.fbx"
    source_path.touch()
    _mock_fbx_load(
        monkeypatch,
        [-1, 0, 1, 2],
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ],
    )
    cond_subset = {
        "parents": np.array([-1, 0]),
        "offsets": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "scale_factor": 1.0,
    }

    result = _check_one_source(cond_subset, str(source_path))

    assert result is not None
    assert "joint count 4 differs from cond.npy reference (2)" in result


def test_check_source_accepts_preprocessing_crop_above_max_joints(
    tmp_path, monkeypatch,
):
    source_path = tmp_path / "source.fbx"
    source_path.touch()
    source_joint_count = 102
    source_parents = np.arange(-1, source_joint_count - 1)
    source_offsets = np.zeros((source_joint_count, 3), dtype=np.float64)
    source_offsets[1:, 0] = 1.0
    _mock_fbx_load(monkeypatch, source_parents, source_offsets)

    cond_subset = {
        "parents": source_parents[:100],
        "offsets": source_offsets[:100],
        "scale_factor": 1.0,
    }

    assert _check_one_source(cond_subset, str(source_path)) is None


def test_check_source_accepts_matching_uncropped_skeleton_above_max_joints(
    tmp_path, monkeypatch,
):
    source_path = tmp_path / "source.fbx"
    source_path.touch()
    source_joint_count = 102
    source_parents = np.arange(-1, source_joint_count - 1)
    source_offsets = np.zeros((source_joint_count, 3), dtype=np.float64)
    source_offsets[1:, 0] = 1.0
    _mock_fbx_load(monkeypatch, source_parents, source_offsets)

    cond_subset = {
        "parents": source_parents,
        "offsets": source_offsets,
        "scale_factor": 1.0,
    }

    assert _check_one_source(cond_subset, str(source_path)) is None


def test_check_source_rejects_different_topology(
    tmp_path, monkeypatch,
):
    """Equal bone lengths must not hide a different hierarchy."""
    source_path = tmp_path / "source.fbx"
    source_path.touch()
    _mock_fbx_load(
        monkeypatch,
        [-1, 0, 1],
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
    )
    cond_subset = {
        "parents": np.array([-1, 0, 0]),
        "offsets": np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        ),
        "scale_factor": 1.0,
    }

    result = _check_one_source(cond_subset, str(source_path))

    assert result is not None
    assert "parent hierarchy differs" in result


def test_check_source_compares_bone_lengths_by_joint_index(
    tmp_path, monkeypatch,
):
    """Corresponding 5%-close bones pass without greedy set matching."""
    source_path = tmp_path / "source.fbx"
    source_path.touch()
    parents = np.array([-1, 0, 1])
    _mock_fbx_load(
        monkeypatch,
        parents,
        [[0.0, 0.0, 0.0], [95.1, 0.0, 0.0], [100.0, 0.0, 0.0]],
    )
    cond_subset = {
        "parents": parents,
        "offsets": np.array(
            [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [105.0, 0.0, 0.0]]
        ),
        "scale_factor": 1.0,
    }

    assert _check_one_source(cond_subset, str(source_path)) is None


def test_check_source_rejects_mismatched_bone_lengths(
    tmp_path, monkeypatch,
):
    """Matching topology still requires corresponding bone lengths."""
    source_path = tmp_path / "source.fbx"
    source_path.touch()
    _mock_fbx_load(
        monkeypatch,
        [-1, 0],
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
    )
    cond_subset = {
        "parents": np.array([-1, 0]),
        "offsets": np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        "scale_factor": 1.0,
    }

    result = _check_one_source(cond_subset, str(source_path))
    assert result is not None
    assert "bone length" in result.lower()


def test_filter_motion_files_supports_case_insensitive_wildcards(tmp_path):
    motion_files = [
        tmp_path / "horse_walk.npy",
        tmp_path / "Raptor_Run.npy",
        tmp_path / "bear_idle.npy",
        tmp_path / "cat_jump.npy",
    ]
    metadata = {
        "horse_walk.npy": {"object_type": "Horse"},
        "Raptor_Run.npy": {"object_type": "RaptorBlue"},
        "bear_idle.npy": {"object_type": "BrownBear"},
        "cat_jump.npy": {"object_type": "Cat"},
    }

    result = _filter_motion_files(
        motion_files, metadata, "horse; RAPTOR*,*bear*",
    )

    assert result == motion_files[:3]


def test_filter_motion_files_without_filter_returns_all_files(tmp_path):
    motion_files = [tmp_path / "horse_walk.npy", tmp_path / "cat_jump.npy"]

    assert _filter_motion_files(motion_files, {}, "") == motion_files
