import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.tensors import truebones_batch_collate
from data_loaders.truebones.truebones_utils import canonical_features as cf
from data_loaders.truebones.truebones_utils.canonical_features import (
    build_canonical_rest_feature,
    canonical_to_physical_hml,
    mark_canonical_cond_entry,
    physical_hml_to_canonical,
    physical_hml_to_lnorm,
    set_canonical_global_stats,
)


def _global_stats():
    # Non-trivial per-channel mean/std so the standardization step is exercised.
    mean = (np.arange(13, dtype=np.float32) * 0.1).astype(np.float32)
    std = np.full(13, 2.0, dtype=np.float32)
    return mean, std


def _cond(with_stats=True):
    rest_pose = np.zeros((2, 13), dtype=np.float32)
    rest_pose[:, 0:3] = np.array([[1.0, 2.0, 3.0], [-2.0, 0.5, 4.0]], dtype=np.float32)
    rest_pose[:, 3:9] = np.arange(12, dtype=np.float32).reshape(2, 6)
    cond = mark_canonical_cond_entry({"rest_pose": rest_pose})
    if with_stats:
        mean, std = _global_stats()
        set_canonical_global_stats(cond, mean, std)
    return cond


def test_canonical_feature_roundtrip_numpy():
    cond = _cond()
    physical = np.random.default_rng(123).normal(size=(5, 2, 13)).astype(np.float32)

    canonical = physical_hml_to_canonical(physical, cond)
    recovered = canonical_to_physical_hml(canonical, cond)

    # Exact inversion is the contract that every call site depends on.
    np.testing.assert_allclose(recovered, physical, atol=1e-5)
    # Rotation channels (size-independent, no rest) are pure global standardization.
    mean = cond["canonical_feature_mean"]
    std = cond["canonical_feature_std"]
    np.testing.assert_allclose(
        canonical[..., 3:9], (physical[..., 3:9] - mean[3:9]) / std[3:9], atol=1e-5
    )
    # Velocity is rescaled by L and standardized -> must differ from the input.
    assert not np.allclose(canonical[..., 9:12], physical[..., 9:12])


def test_lnorm_scale_uses_per_skeleton_length():
    # The L-normalization step (physical_hml_to_lnorm, stats-free) scales
    # position/velocity by the per-skeleton length L while rotation is
    # size-independent. This is the space the global stats are calibrated in.
    small = mark_canonical_cond_entry(
        {"rest_pose": np.tile(np.array([[0.1, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32), (2, 1))}
    )
    big = mark_canonical_cond_entry(
        {"rest_pose": np.tile(np.array([[10.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32), (2, 1))}
    )
    # non-degenerate spread between the two joints
    small["rest_pos_ric_hml"] = np.array([[0.0, 0, 0], [0.2, 0, 0]], dtype=np.float32)
    big["rest_pos_ric_hml"] = np.array([[0.0, 0, 0], [20.0, 0, 0]], dtype=np.float32)

    L_small = cf._length_scale_from_rest(small["rest_pos_ric_hml"])
    L_big = cf._length_scale_from_rest(big["rest_pos_ric_hml"])
    assert L_big > L_small * 50

    phys = np.ones((3, 2, 13), dtype=np.float32)
    enc_small = physical_hml_to_lnorm(phys, small)
    enc_big = physical_hml_to_lnorm(phys, big)
    # bigger skeleton -> larger pos/vel divisor -> smaller encoded magnitude
    assert np.abs(enc_big[..., 9:12]).mean() < np.abs(enc_small[..., 9:12]).mean()
    # rotation is L-independent -> identical across skeletons
    np.testing.assert_allclose(enc_small[..., 3:9], enc_big[..., 3:9], atol=1e-6)


def test_canonical_encode_requires_global_stats():
    # Silently skipping standardization is a bug; encode/decode must raise when
    # the global stats are missing rather than return wrong-scale features.
    cond = _cond(with_stats=False)
    phys = np.ones((3, 2, 13), dtype=np.float32)
    import pytest
    with pytest.raises(KeyError):
        physical_hml_to_canonical(phys, cond)
    with pytest.raises(KeyError):
        canonical_to_physical_hml(phys, cond)


def test_canonical_rest_feature_roundtrips_to_physical_rest():
    cond = _cond()
    rest = build_canonical_rest_feature(cond)
    recovered = canonical_to_physical_hml(rest, cond)

    # The rest token decodes back to the rest pose: position == rest position,
    # rotation == rest rotation, velocity / contact == 0 (no motion).
    np.testing.assert_allclose(recovered[:, 0:3], cond["rest_pose"][:, 0:3], atol=1e-5)
    np.testing.assert_allclose(recovered[:, 3:9], cond["rest_pose"][:, 3:9], atol=1e-5)
    np.testing.assert_allclose(recovered[:, 9:13], 0.0, atol=1e-5)


def test_canonical_decode_torch_batch_layout():
    cond = _cond()
    physical = torch.randn(1, 2, 13, 4)
    mean, std = _global_stats()
    y = {
        "rest_pos_ric_hml": torch.as_tensor(cond["rest_pos_ric_hml"]).unsqueeze(0),
        "canonical_feature_mean": torch.as_tensor(mean),
        "canonical_feature_std": torch.as_tensor(std),
    }

    canonical = physical_hml_to_canonical(physical, y)
    recovered = canonical_to_physical_hml(canonical, y)

    torch.testing.assert_close(recovered, physical, atol=1e-4, rtol=1e-4)


def test_canonical_decode_torch_multi_skeleton_batch():
    # Batched [B, J, F, T] decode with per-sample skeleton lengths exercises the
    # [B] length-scale path used by the training-time aux-loss decode.
    rest = torch.stack(
        [
            torch.tensor([[0.0, 0, 0], [0.3, 0, 0]]),
            torch.tensor([[0.0, 0, 0], [3.0, 0, 0]]),
        ]
    ).float()  # [B=2, J=2, 3]
    mean, std = _global_stats()
    y = {
        "rest_pos_ric_hml": rest,
        "canonical_feature_mean": torch.as_tensor(mean),
        "canonical_feature_std": torch.as_tensor(std),
    }
    physical = torch.randn(2, 2, 13, 5)

    canonical = physical_hml_to_canonical(physical, y)
    recovered = canonical_to_physical_hml(canonical, y)

    torch.testing.assert_close(recovered, physical, atol=1e-4, rtol=1e-4)
    # different skeleton sizes -> different encoded position magnitudes per sample
    assert not torch.allclose(
        canonical[0, :, 0:3].abs().mean(), canonical[1, :, 0:3].abs().mean()
    )


def test_truebones_collate_drops_motion_stats_and_carries_global_stats():
    cond = _cond()
    motion = np.zeros((4, 2, 13), dtype=np.float32)
    item = (
        motion,
        4,
        np.array([-1, 0], dtype=np.int64),
        build_canonical_rest_feature(cond),
        np.zeros((2, 3), dtype=np.float32),
        torch.ones((5, 5), dtype=torch.float32),
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
        "TestSpecies",
        np.zeros((2, 4), dtype=np.float32),
        0,
        2,
        {"translation_root_index": 0},
        "TestSpecies_Motion_1.npy",
        {
            "rest_pose_physical": cond["rest_pose"],
            "rest_pos_ric_hml": cond["rest_pos_ric_hml"],
            "canonical_feature_mean": cond["canonical_feature_mean"],
            "canonical_feature_std": cond["canonical_feature_std"],
            "feature_space": "canonical_motion_v3",
            "joint_mask_candidate_roots": np.array([False, True]),
        },
    )

    _motion, batch_cond = truebones_batch_collate([item])
    y = batch_cond["y"]

    assert "mean" not in y
    assert "std" not in y
    assert y["feature_space"] == ["canonical_motion_v3"]
    assert tuple(y["rest_pos_ric_hml"].shape) == (1, 2, 3)
    # Per-object_subset standardization stats flow through per-sample, stacked in
    # batch order as [B, 13] so a mixed-species batch de-standardizes each sample
    # with its own object_subset's stats.
    assert tuple(y["canonical_feature_mean"].shape) == (1, 13)
    assert tuple(y["canonical_feature_std"].shape) == (1, 13)
