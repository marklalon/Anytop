import os
import sys

import numpy as np
import pytest
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
    mean = (np.arange(12, dtype=np.float32) * 0.1).astype(np.float32)
    std = np.full(12, 2.0, dtype=np.float32)
    return mean, std


def _cond(with_stats=True):
    rest_pose = np.zeros((2, 12), dtype=np.float32)
    rest_pose[:, 0:3] = np.array([[1.0, 2.0, 3.0], [-2.0, 0.5, 4.0]], dtype=np.float32)
    rest_pose[:, 3:9] = np.arange(12, dtype=np.float32).reshape(2, 6)
    cond = mark_canonical_cond_entry({"rest_pose": rest_pose})
    if with_stats:
        mean, std = _global_stats()
        set_canonical_global_stats(cond, mean, std, 0.1)
    return cond


def test_canonical_feature_roundtrip_numpy():
    cond = _cond()
    physical = np.random.default_rng(123).normal(size=(5, 2, 12)).astype(np.float32)

    canonical = physical_hml_to_canonical(physical, cond)
    recovered = canonical_to_physical_hml(canonical, cond)

    # Exact inversion on every kept channel is the contract that every call
    # site depends on; the derived velocity channels decode to zero.
    np.testing.assert_allclose(recovered, cf.zero_derived_velocity(physical), atol=1e-5)
    np.testing.assert_allclose(canonical[:, 1:, 9:12], 0.0)
    np.testing.assert_allclose(canonical[:, 0, 10], 0.0)
    # Rotation channels (size-independent, no rest) are pure global standardization.
    mean = cond["canonical_feature_mean"]
    std = cond["canonical_feature_std"]
    np.testing.assert_allclose(
        canonical[..., 3:9], (physical[..., 3:9] - mean[3:9]) / std[3:9], atol=1e-5
    )
    # The root's X/Z velocity is rescaled by L and standardized -> must differ
    # from the input.
    assert not np.allclose(canonical[:, 0, [9, 11]], physical[:, 0, [9, 11]])


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
    # non-degenerate spread between the two joints, both above the L floor
    small["rest_pos_ric_hml"] = np.array([[0.0, 0, 0], [2.0, 0, 0]], dtype=np.float32)
    big["rest_pos_ric_hml"] = np.array([[0.0, 0, 0], [200.0, 0, 0]], dtype=np.float32)

    L_small = cf._length_scale_from_rest(small["rest_pos_ric_hml"])
    L_big = cf._length_scale_from_rest(big["rest_pos_ric_hml"])
    assert L_big > L_small * 50

    phys = np.ones((3, 2, 12), dtype=np.float32)
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
    phys = np.ones((3, 2, 12), dtype=np.float32)
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
    # rotation == rest rotation, velocity == 0 (no motion).
    np.testing.assert_allclose(recovered[:, 0:3], cond["rest_pose"][:, 0:3], atol=1e-5)
    np.testing.assert_allclose(recovered[:, 3:9], cond["rest_pose"][:, 3:9], atol=1e-5)
    np.testing.assert_allclose(recovered[:, 9:12], 0.0, atol=1e-5)


def test_canonical_decode_torch_batch_layout():
    cond = _cond()
    physical = torch.randn(1, 2, 12, 4)
    mean, std = _global_stats()
    y = {
        "rest_pos_ric_hml": torch.as_tensor(cond["rest_pos_ric_hml"]).unsqueeze(0),
        "canonical_feature_mean": torch.as_tensor(mean),
        "canonical_feature_std": torch.as_tensor(std),
    }

    canonical = physical_hml_to_canonical(physical, y)
    recovered = canonical_to_physical_hml(canonical, y)

    torch.testing.assert_close(recovered, cf.zero_derived_velocity(physical), atol=1e-4, rtol=1e-4)


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
    physical = torch.randn(2, 2, 12, 5)

    canonical = physical_hml_to_canonical(physical, y)
    recovered = canonical_to_physical_hml(canonical, y)

    torch.testing.assert_close(recovered, cf.zero_derived_velocity(physical), atol=1e-4, rtol=1e-4)
    # different skeleton sizes -> different encoded position magnitudes per sample
    assert not torch.allclose(
        canonical[0, :, 0:3].abs().mean(), canonical[1, :, 0:3].abs().mean()
    )


def test_canonical_decode_torch_padded_batch_matches_unpadded_samples():
    """Padding joints must not change the per-skeleton geometric scale ``L``.

    Training collates mixed skeletons by zero-padding ``rest_pos_ric_hml`` to
    ``max_joints``.  The canonical encoder calibrated each sample with its
    unpadded rest geometry, so batched decode must use ``n_joints`` to ignore
    those padding rows and exactly match decoding each sample on its own.
    """
    rest_a = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.8, 0.0], [0.5, 1.2, 0.0], [0.8, 1.3, 0.4]],
        dtype=torch.float32,
    )
    rest_b = torch.tensor(
        [[0.0, 0.0, 0.0], [3.0, 1.0, -0.5]],
        dtype=torch.float32,
    )
    max_joints = len(rest_a)
    padded_rest = torch.zeros(2, max_joints, 3)
    padded_rest[0, :len(rest_a)] = rest_a
    padded_rest[1, :len(rest_b)] = rest_b

    mean, std = _global_stats()
    batch_y = {
        "rest_pos_ric_hml": padded_rest,
        "n_joints": torch.tensor([len(rest_a), len(rest_b)]),
        "canonical_feature_mean": torch.stack(
            [torch.as_tensor(mean), torch.as_tensor(mean + 0.25)]
        ),
        "canonical_feature_std": torch.stack(
            [torch.as_tensor(std), torch.as_tensor(std * 1.5)]
        ),
    }
    canonical = torch.randn(2, max_joints, 12, 5)
    decoded_batch = canonical_to_physical_hml(canonical, batch_y)

    for batch_index, rest in enumerate((rest_a, rest_b)):
        n_joints = len(rest)
        single_y = {
            "rest_pos_ric_hml": rest.unsqueeze(0),
            "canonical_feature_mean": batch_y["canonical_feature_mean"][batch_index],
            "canonical_feature_std": batch_y["canonical_feature_std"][batch_index],
        }
        decoded_single = canonical_to_physical_hml(
            canonical[batch_index:batch_index + 1, :n_joints], single_y
        )
        torch.testing.assert_close(
            decoded_batch[batch_index, :n_joints], decoded_single[0],
            atol=1e-5, rtol=1e-5,
        )


def test_length_scale_floor_is_half_the_reference_skeleton():
    from data_loaders.truebones.truebones_utils.param_utils import (
        HML_REF_REST_LENGTH_SCALE,
        _SMPL_REST_POSITIONS,
    )

    # The reference skeleton sits above the floor, so its L is the raw spread.
    np.testing.assert_allclose(
        cf._length_scale_from_rest(_SMPL_REST_POSITIONS), HML_REF_REST_LENGTH_SCALE, rtol=1e-12
    )
    assert cf.REST_LENGTH_SCALE_FLOOR == pytest.approx(0.5 * HML_REF_REST_LENGTH_SCALE)


def test_length_scale_floors_clustered_rigs_on_every_path():
    """A few clustered joints (MU04_Pollen: 4 joints, raw L 0.03) get the floor on
    the numpy, torch and padded-batch paths alike; a normal rig keeps its own L."""
    floor = cf.REST_LENGTH_SCALE_FLOOR
    clustered = np.array(
        [[0.0, 1.835, 0.0], [0.0, 1.835, 0.0], [0.0, 1.96, 0.0], [0.0, 1.908, 0.0]],
        dtype=np.float32,
    )
    normal = np.array([[0.0, 0.0, 0.0], [3.0, 1.0, -0.5]], dtype=np.float32)
    normal_L = float(cf._length_scale_from_rest(normal))
    assert normal_L > floor

    assert float(cf._length_scale_from_rest(clustered)) == pytest.approx(floor)
    assert float(cf._length_scale_from_rest(torch.as_tensor(clustered))) == pytest.approx(floor)
    # Collapsed and non-finite rests land on the floor as well.
    assert float(cf._length_scale_from_rest(np.zeros((3, 3)))) == pytest.approx(floor)
    assert float(cf._length_scale_from_rest(torch.zeros(3, 3))) == pytest.approx(floor)
    assert float(cf._length_scale_from_rest(np.full((2, 3), np.nan))) == pytest.approx(floor)

    padded = np.zeros((2, 4, 3), dtype=np.float32)
    padded[0] = clustered
    padded[1, :2] = normal
    counts = np.array([4, 2])
    expected = np.array([floor, normal_L])
    np.testing.assert_allclose(cf._length_scale_from_rest(padded, n_joints=counts), expected, rtol=1e-6)
    np.testing.assert_allclose(
        cf._length_scale_from_rest(torch.as_tensor(padded), n_joints=torch.as_tensor(counts)).numpy(),
        expected, rtol=1e-5,
    )


def test_canonical_roundtrip_is_exact_for_a_floored_rig():
    cond = _cond()
    cond["rest_pos_ric_hml"] = np.array([[0.0, 1.835, 0.0], [0.0, 1.9, 0.0]], dtype=np.float32)
    physical = np.random.default_rng(7).normal(size=(5, 2, 12)).astype(np.float32)

    lnorm = physical_hml_to_lnorm(physical, cond)
    np.testing.assert_allclose(
        lnorm[..., 9:12], physical[..., 9:12] / cf.REST_LENGTH_SCALE_FLOOR, rtol=1e-5
    )
    recovered = canonical_to_physical_hml(physical_hml_to_canonical(physical, cond), cond)
    np.testing.assert_allclose(recovered, cf.zero_derived_velocity(physical), atol=1e-5)


def test_length_scale_numpy_batch_ignores_padding_rows():
    rest_a = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.8, 0.0], [0.5, 1.2, 0.0]], dtype=np.float32
    )
    rest_b = np.array([[0.0, 0.0, 0.0], [3.0, 1.0, -0.5]], dtype=np.float32)
    padded_rest = np.zeros((2, len(rest_a), 3), dtype=np.float32)
    padded_rest[0] = rest_a
    padded_rest[1, :len(rest_b)] = rest_b

    batched = cf._length_scale_from_rest(
        padded_rest, n_joints=np.array([len(rest_a), len(rest_b)])
    )
    expected = np.array([
        cf._length_scale_from_rest(rest_a),
        cf._length_scale_from_rest(rest_b),
    ])
    np.testing.assert_allclose(batched, expected, atol=1e-7, rtol=1e-7)


def test_truebones_collate_drops_motion_stats_and_carries_global_stats():
    cond = _cond()
    motion = np.zeros((4, 2, 12), dtype=np.float32)
    item = (
        motion,
        4,
        np.array([-1, 0], dtype=np.int64),
        build_canonical_rest_feature(cond),
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
        "TestSpecies",
        np.zeros((2, 4), dtype=np.float32),
        2,
        {"translation_root_index": 0},
        "TestSpecies_Motion_1.npy",
        {
            "rest_pose_physical": cond["rest_pose"],
            "rest_pos_ric_hml": cond["rest_pos_ric_hml"],
            "canonical_feature_mean": cond["canonical_feature_mean"],
            "canonical_feature_std": cond["canonical_feature_std"],
            "feature_space": cf.CANONICAL_FEATURE_SPACE,
            "joint_mask_candidate_roots": np.array([False, True]),
        },
    )

    _motion, batch_cond = truebones_batch_collate([item])
    y = batch_cond["y"]

    assert "mean" not in y
    assert "std" not in y
    assert y["feature_space"] == [cf.CANONICAL_FEATURE_SPACE]
    assert tuple(y["rest_pos_ric_hml"].shape) == (1, 2, 3)
    # Per-object_subset standardization stats flow through per-sample, stacked in
    # batch order as [B, FEATS_LEN] so a mixed-species batch de-standardizes each sample
    # with its own object_subset's stats.
    assert tuple(y["canonical_feature_mean"].shape) == (1, 12)
    assert tuple(y["canonical_feature_std"].shape) == (1, 12)
    # The rest length scale is fixed at collate time from the unpadded rest, so
    # the training decode never derives it from the padded rows.
    assert tuple(y[cf.REST_LENGTH_SCALE_KEY].shape) == (1,)
    np.testing.assert_allclose(
        y[cf.REST_LENGTH_SCALE_KEY].numpy(),
        [cf._length_scale_from_rest(cond["rest_pos_ric_hml"])],
        rtol=1e-6,
    )


def _collate_item(cond, rest_pos, max_joints, name):
    n_joints = len(rest_pos)
    entry = dict(cond)
    entry["rest_pos_ric_hml"] = np.asarray(rest_pos, dtype=np.float32)
    entry["rest_pose"] = np.zeros((n_joints, 12), dtype=np.float32)
    entry["rest_pose"][:, 0:3] = entry["rest_pos_ric_hml"]
    return (
        np.zeros((4, n_joints, 12), dtype=np.float32),
        4,
        np.arange(-1, n_joints - 1, dtype=np.int64),
        build_canonical_rest_feature(entry),
        np.zeros((n_joints, 3), dtype=np.float32),
        np.zeros((n_joints, n_joints), dtype=np.float32),
        np.zeros((n_joints, n_joints), dtype=np.float32),
        name,
        np.zeros((n_joints, 4), dtype=np.float32),
        max_joints,
        {"translation_root_index": 0},
        f"{name}_Motion_1.npy",
        {
            "rest_pos_ric_hml": entry["rest_pos_ric_hml"],
            "canonical_feature_mean": entry["canonical_feature_mean"],
            "canonical_feature_std": entry["canonical_feature_std"],
            "feature_space": cf.CANONICAL_FEATURE_SPACE,
        },
    )


def test_collated_padded_batch_decodes_like_each_unpadded_sample():
    """End-to-end through the real collate: a mixed-skeleton batch padded to
    max_joints decodes every sample exactly as its own unpadded cond entry does.
    The collate supplies ``rest_length_scale`` so no padded row reaches ``L``."""
    rest_a = [[0.0, 0.0, 0.0], [0.0, 0.8, 0.0], [0.5, 1.2, 0.0], [0.8, 1.3, 0.4]]
    rest_b = [[0.0, 0.0, 0.0], [3.0, 1.0, -0.5]]
    max_joints = 7
    mean, std = _global_stats()
    cond_a = set_canonical_global_stats(mark_canonical_cond_entry({}), mean, std, 0.1)
    cond_b = set_canonical_global_stats(mark_canonical_cond_entry({}), mean + 0.25, std * 1.5, 0.1)
    _motion, batch_cond = truebones_batch_collate([
        _collate_item(cond_a, rest_a, max_joints, "A"),
        _collate_item(cond_b, rest_b, max_joints, "B"),
    ])
    y = batch_cond["y"]
    assert tuple(y["rest_pos_ric_hml"].shape) == (2, max_joints, 3)
    assert cf.REST_LENGTH_SCALE_KEY in y

    torch.manual_seed(0)
    canonical = torch.randn(2, max_joints, 12, 5)
    decoded_batch = canonical_to_physical_hml(canonical, y)
    for index, (rest, cond) in enumerate(((rest_a, cond_a), (rest_b, cond_b))):
        n_joints = len(rest)
        single = dict(cond)
        single["rest_pos_ric_hml"] = np.asarray(rest, dtype=np.float32)
        decoded_single = canonical_to_physical_hml(canonical[index:index + 1, :n_joints], single)
        torch.testing.assert_close(
            decoded_batch[index, :n_joints], decoded_single[0], atol=1e-5, rtol=1e-5
        )

    # Decoding one sample out of the collated batch must slice every per-sample
    # field; handing the decoder the whole [B, F] stat stack is an error, not a
    # broadcast that silently reads row 0's object_subset.
    with pytest.raises(ValueError, match="per-sample canonical stats"):
        canonical_to_physical_hml(
            canonical[1:2, :len(rest_b)],
            {
                "rest_pos_ric_hml": y["rest_pos_ric_hml"][1:2, :len(rest_b)],
                "canonical_feature_mean": y["canonical_feature_mean"],
                "canonical_feature_std": y["canonical_feature_std"],
            },
        )


def test_rest_length_scale_in_cond_takes_precedence_over_rest_rows():
    cond = _cond()
    feature = np.zeros((1, 2, 12, 3), dtype=np.float32)
    feature[..., 0:3, :] = 1.0
    feature[..., 9:12, :] = 1.0
    derived = float(cf._length_scale_from_rest(cond["rest_pos_ric_hml"]))
    with_key = dict(cond)
    with_key[cf.REST_LENGTH_SCALE_KEY] = np.float32(derived * 4.0)
    base = cf._apply_L_scale(feature, cond, inverse=True)
    scaled = cf._apply_L_scale(feature, with_key, inverse=True)
    np.testing.assert_allclose(scaled[..., 0:3, :], base[..., 0:3, :] * 4.0, rtol=1e-6)
    np.testing.assert_allclose(scaled[..., 9:12, :], base[..., 9:12, :] * 4.0, rtol=1e-6)
    np.testing.assert_allclose(scaled[..., 3:9, :], base[..., 3:9, :], rtol=0)


def test_length_scale_broadcasts_a_shared_rest_over_per_row_counts():
    """A single shared rest with one count per batch row gives one ``L`` per row,
    identically on numpy and torch -- no device-dependent collapse to row 0."""
    rest = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.8, 0.0], [0.5, 1.2, 0.0], [9.0, 9.0, 9.0]],
        dtype=np.float32,
    )
    expected = np.array([
        cf._length_scale_from_rest(rest[:4]),
        cf._length_scale_from_rest(rest[:3]),
    ])
    np.testing.assert_allclose(
        cf._length_scale_from_rest(rest, n_joints=np.array([4, 3])), expected, rtol=1e-6
    )
    np.testing.assert_allclose(
        cf._length_scale_from_rest(torch.as_tensor(rest), n_joints=torch.tensor([4, 3])).numpy(),
        expected, rtol=1e-5,
    )
    # A lone count against an unbatched rest stays a scalar.
    assert np.ndim(cf._length_scale_from_rest(rest, n_joints=3)) == 0
    assert cf._length_scale_from_rest(torch.as_tensor(rest), n_joints=torch.tensor([3])).dim() == 0
    # One count broadcasts over a batch; a mismatched count vector is an error.
    batched = np.stack([rest, rest])
    assert cf._length_scale_from_rest(batched, n_joints=3).shape == (2,)
    with pytest.raises(ValueError, match="one per batch row"):
        cf._length_scale_from_rest(np.stack([rest] * 3), n_joints=[4, 3])


def _raw_subset_stats():
    """Two subsets with anisotropic blocks and very different position gains.

    Mirrors the measured shape of the real table (aquatic's vertical position std
    is ~4x its horizontal one; quadruped's is ~1.8x).
    """
    quadruped_std = np.array(
        [0.468, 0.720, 0.824, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 0.5, 0.6, 0.7],
        dtype=np.float32,
    )
    aquatic_std = np.array(
        [0.687, 2.810, 1.294, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 0.8, 0.9, 1.0],
        dtype=np.float32,
    )
    return {
        "quadruped": (np.arange(12, dtype=np.float32) * 0.01, quadruped_std, np.float32(0.2)),
        "aquatic": (np.arange(12, dtype=np.float32) * -0.02, aquatic_std, np.float32(0.05)),
    }


def test_pool_global_stats_weights_subsets_equally_per_block():
    raw = _raw_subset_stats()
    mean, std, dpos_std = cf.pool_global_stats(raw)

    assert mean.shape == (12,) and std.shape == (12,)
    # The position-step RMS is pooled like a block scalar.
    np.testing.assert_allclose(dpos_std, 0.1, rtol=1e-6)
    # mean is the plain average of the subset means, whatever the clip counts.
    np.testing.assert_allclose(
        mean, (raw["quadruped"][0] + raw["aquatic"][0]) / 2.0, rtol=1e-6
    )
    # Every block is one scalar: the geometric mean of the subsets' block means.
    for block in (slice(0, 3), slice(3, 9), slice(9, 12)):
        assert len(set(std[block].tolist())) == 1
        expected = float(np.exp(np.mean(np.log([
            raw["quadruped"][1][block].mean(), raw["aquatic"][1][block].mean()
        ]))))
        np.testing.assert_allclose(std[block.start], expected, rtol=1e-6)


def test_pool_global_stats_tolerates_degenerate_blocks():
    raw = {
        "serpentine": (np.zeros(12, dtype=np.float32), np.zeros(12, dtype=np.float32), np.float32(0.0)),
        "aquatic": _raw_subset_stats()["aquatic"],
    }
    _mean, std, dpos_std = cf.pool_global_stats(raw)
    np.testing.assert_allclose(dpos_std, 0.05, rtol=1e-6)
    # The constant subset contributes nothing usable; the other subset sets every block.
    np.testing.assert_allclose(std[0], raw["aquatic"][1][0:3].mean(), rtol=1e-6)

    _mean, std, dpos_std = cf.pool_global_stats({"serpentine": raw["serpentine"]})
    # Nothing usable at all: unit std, never a zero divide.
    np.testing.assert_allclose(std, 1.0)
    np.testing.assert_allclose(dpos_std, 1.0)
    with pytest.raises(ValueError):
        cf.pool_global_stats({})


def test_assert_single_canonical_frame_rejects_mixed_frames():
    mean, std, dpos_std = cf.pool_global_stats(_raw_subset_stats())
    cond = {name: set_canonical_global_stats({}, mean, std, dpos_std) for name in ("Dog", "Bat")}
    cond["Bare"] = {}
    cf.assert_single_canonical_frame(cond)

    set_canonical_global_stats(cond["Bat"], mean + 0.1, std, dpos_std)
    with pytest.raises(ValueError, match="2 different"):
        cf.assert_single_canonical_frame(cond)

    # A different position-step RMS is a different frame too.
    set_canonical_global_stats(cond["Bat"], mean, std, dpos_std * 2.0)
    with pytest.raises(ValueError, match="2 different"):
        cf.assert_single_canonical_frame(cond)


def test_rebuild_derived_velocity_matches_the_stored_world_steps():
    # A stored clip's velocity rows are world steps (RIC removes the root's
    # world XZ from every joint). Decoding keeps the root's X/Z and rebuilds
    # the rest from positions; the result equals the stored rows.
    rng = np.random.default_rng(7)
    n_frames, n_joints, step = 6, 3, 0.75
    world = np.cumsum(rng.normal(size=(n_frames, n_joints, 3)), axis=0).astype(np.float64)
    ric = world.copy()
    ric[:, :, [0, 2]] -= world[:, 0:1, [0, 2]]
    motion = np.zeros((n_frames, n_joints, 12), dtype=np.float32)
    motion[..., 0:3] = ric
    motion[:-1, :, 9:12] = (world[1:] - world[:-1]) / step
    motion[-1, :, 9:12] = motion[-2, :, 9:12]

    rebuilt = cf.rebuild_derived_velocity(cf.zero_derived_velocity(motion), is_loop=False, step_scale=step)
    np.testing.assert_allclose(rebuilt, motion, atol=1e-5)

    # A loop's terminal row is the wrap step.
    looped = motion.copy()
    looped[-1, :, 9:12] = (world[0] - world[-1]) / step
    looped[-1, 0, [9, 11]] = (world[0, 0, [0, 2]] - world[-1, 0, [0, 2]]) / step
    rebuilt_loop = cf.rebuild_derived_velocity(cf.zero_derived_velocity(looped), is_loop=True, step_scale=step)
    np.testing.assert_allclose(rebuilt_loop, looped, atol=1e-5)


def test_species_aux_loss_weights_only_scale_down():
    mean = np.zeros(12, dtype=np.float32)
    std = np.ones(12, dtype=np.float32)
    acc = None
    slow = np.zeros((5, 2, 12), dtype=np.float32)
    slow[:, 0, 9] = 0.1
    fast = np.zeros((5, 2, 12), dtype=np.float32)
    fast[:, 0, 9] = 10.0
    fast[:, :, 0] = np.arange(5, dtype=np.float32)[:, None] * 10.0
    cond = mark_canonical_cond_entry({"rest_pose": np.zeros((2, 12), dtype=np.float32)})
    acc = cf.accumulate_lnorm_stats(slow, cond, acc=acc, species="slow")
    acc = cf.accumulate_lnorm_stats(fast, cond, acc=acc, species="fast")
    weights = cf.species_aux_loss_weights([acc], mean, std, dpos_std=1.0)
    assert weights["slow"] == (1.0, 1.0)
    root_weight, step_weight = weights["fast"]
    assert 0.0 < root_weight < 1.0 and 0.0 < step_weight < 1.0
