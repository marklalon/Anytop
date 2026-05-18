"""Unit tests for the v4 role-aware structural prior bank.

These cover the three properties the design relies on:

1. Default-bank behavior: root and non-root channels retain unit std where no
    learned structural prior is present.
2. Role-aware anisotropy: root position keeps axis-wise profiles while
    non-root joints use the stretch scalar instead of non-root position/velocity.
3. Cross-skeleton transfer + npy round-trip: the shared bank resolves onto a
    motion-free skeleton by canonical joint name and survives save/load.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.truebones.truebones_utils import dataset_pipeline as dp


FEAT = 13


def _tpose(n_joints):
    # Zero anchor so residual == motion; vel/contact get zeroed by
    # _build_structural_norm_mean anyway.
    return np.zeros((n_joints, FEAT), dtype=np.float32)


def _payload(canonical_names, motion):
    joint_count = len(canonical_names)
    return {
        'object_cond': {
            'tpos_first_frame': _tpose(joint_count),
            'canonical_joint_names': list(canonical_names),
            'offsets': np.zeros((joint_count, 3), dtype=np.float32),
            'rest_rotations': np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (joint_count, 1)),
            'canon_joint_rot': np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (joint_count, 1)),
        },
        'results': [{'motion': np.asarray(motion, dtype=np.float32)}],
    }


def _object_cond(canonical_names):
    joint_count = len(canonical_names)
    return {
        'tpos_first_frame': _tpose(joint_count),
        'canonical_joint_names': list(canonical_names),
        'offsets': np.zeros((joint_count, 3), dtype=np.float32),
        'rest_rotations': np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (joint_count, 1)),
        'canon_joint_rot': np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (joint_count, 1)),
    }


def test_sanitize_profile_normalizes_clamps_and_degrades():
    # mean-normalized to 1.0
    prof = dp._sanitize_profile([1.0, 2.0, 3.0], 3)
    assert pytest.approx(float(np.mean(prof)), rel=1e-6) == 1.0
    # clamped: a near-zero axis cannot collapse the channel toward 0
    prof = dp._sanitize_profile([1e-9, 1.0, 1.0], 3)
    assert all(np.isfinite(prof))
    assert min(prof) > 0.1, f"near-static axis collapsed: {prof}"
    assert pytest.approx(float(np.mean(prof)), rel=1e-6) == 1.0
    # bad input / length mismatch -> isotropic ones
    assert dp._sanitize_profile(None, 6) == [1.0] * 6
    assert dp._sanitize_profile([1.0, 2.0], 3) == [1.0] * 3
    assert dp._sanitize_profile([0.0, -1.0, np.nan], 3) == [1.0] * 3


def test_default_bank_is_isotropic_and_v1_equivalent():
    bank = dp._default_structural_prior_bank()
    assert bank['schema_version'] == 4
    assert bank['variance_calibration'] == {'pos': 1.0, 'rot': 1.0, 'vel': 1.0, 'stretch': 1.0}
    cond = _object_cond(['root', 'knee'])
    dp._apply_structural_stats_to_object_cond(cond, bank)
    norm_std = cond['norm_std']
    assert norm_std.shape == (2, FEAT)
    assert np.allclose(norm_std[0, 0:3], norm_std[0, 0])
    assert np.allclose(norm_std[0, 3:9], norm_std[0, 3])
    assert np.allclose(norm_std[0, 9:12], norm_std[0, 9])
    assert np.allclose(norm_std[1, 3:9], norm_std[1, 3])
    # default magnitude is 1.0 and identity calibration leaves it untouched
    assert np.allclose(norm_std[:, 0:12], 1.0)
    assert cond['norm_schema_version'] == 4
    assert cond['norm_mean'][1, 9] == pytest.approx(1.0)
    assert cond['norm_std_variance_calibration'] == {'pos': 1.0, 'rot': 1.0, 'vel': 1.0, 'stretch': 1.0}


def test_anisotropy_is_recovered_with_magnitude_preserved():
    rng = np.random.default_rng(0)
    frames = 4000
    motion = np.zeros((frames, 2, FEAT), dtype=np.float32)
    # root keeps transferable position channels, so anisotropy should still be recovered here.
    motion[:, 0, 0] = rng.normal(0.0, 0.05, size=frames)
    motion[:, 0, 1] = rng.normal(0.0, 1.00, size=frames)
    motion[:, 0, 2] = rng.normal(0.0, 0.05, size=frames)
    # non-root translation is now represented by stretch.
    motion[:, 1, 9] = rng.normal(1.0, 0.20, size=frames)

    payload = _payload(['root', 'knee'], motion)
    bank = dp._build_structural_prior_bank([payload])

    root_leaf = bank['by_canonical_name']['root']
    pos_profile = np.asarray(root_leaf['pos_profile'])
    # Y axis must dominate the profile; X/Z suppressed
    assert pos_profile[1] > pos_profile[0] * 3
    assert pos_profile[1] > pos_profile[2] * 3
    # profile is unitless / mean-normalized
    assert pytest.approx(float(np.mean(pos_profile)), rel=1e-6) == 1.0

    knee_leaf = bank['by_canonical_name']['knee']
    assert knee_leaf['stretch'] > 0.1

    cond = _object_cond(['root', 'knee'])
    dp._apply_structural_stats_to_object_cond(cond, bank)
    root_std = cond['norm_std'][0, 0:3]
    # anisotropy is preserved through the (uniform) per-group calibration
    assert root_std[1] > root_std[0] * 3
    assert root_std[1] > root_std[2] * 3
    # magnitude/shape decoupling: channel mean ~= v1 pooled scalar magnitude
    # scaled only by the shared per-group variance-calibration factor.
    cal_pos = bank['variance_calibration']['pos']
    assert pytest.approx(float(np.mean(root_std)), rel=1e-6) == float(root_leaf['pos']) * cal_pos
    assert cond['norm_std'][1, 9] == pytest.approx(float(knee_leaf['stretch']) * bank['variance_calibration']['stretch'])
    # all channels respect the floor
    assert (cond['norm_std'] >= dp._MIN_STRUCTURAL_SCALE).all()


def test_variance_calibration_makes_training_data_unit_rms():
    # Real-world failure mode: a joint's resolved std is the *population
    # median* of the shared canonical leaf, not its own scale. Two species
    # whose 'knee' moves at very different scales => the median under-estimates
    # the big mover, so the pooled normalized RMS >> 1 before calibration.
    rng = np.random.default_rng(7)
    frames = 4000

    def _species(knee_pos_std):
        m = np.zeros((frames, 2, FEAT), dtype=np.float32)
        m[:, 0, 0:3] = rng.normal(0.0, knee_pos_std, size=(frames, 3))   # root
        m[:, :, 3:9] = rng.normal(0.0, 0.3, size=(frames, 2, 6))
        m[:, 0, 9:12] = rng.normal(0.0, 0.2, size=(frames, 3))
        m[:, 1, 9] = rng.normal(1.0, 0.1, size=frames)
        return _payload(['root', 'knee'], m)

    # Median of the shared 'knee' leaf is pinned near the small movers, so
    # the big mover is left grossly under-normalized unless calibration accounts for it.
    species = [_species(0.1), _species(0.1), _species(8.0)]
    bank = dp._build_structural_prior_bank(species)

    cal = bank['variance_calibration']
    assert cal['pos'] > 3.0, f"calibration did not detect the mismatch: {cal}"

    # After calibration the diffusion model sees ~unit-RMS data, pooled over
    # the whole 'training set' (exact by construction of optimization "a").
    sq_sum = {g: 0.0 for g in ('pos', 'rot', 'vel', 'stretch')}
    n = {g: 0 for g in ('pos', 'rot', 'vel', 'stretch')}
    for payload in species:
        cond = _object_cond(['root', 'knee'])
        dp._apply_structural_stats_to_object_cond(cond, bank)
        motion = payload['results'][0]['motion'].astype(np.float64)
        normalized = (motion - cond['norm_mean'][None]) / cond['norm_std'][None]
        for joint_index in range(normalized.shape[1]):
            for g, start, stop in dp._joint_feature_groups(joint_index):
                blk = normalized[:, joint_index, start:stop]
                sq_sum[g] += float(np.sum(blk ** 2))
                n[g] += blk.size
    for g in ('pos', 'rot', 'vel', 'stretch'):
        rms = float(np.sqrt(sq_sum[g] / n[g]))
        assert rms == pytest.approx(1.0, abs=0.05), f"group {g} pooled rms={rms}"


def test_variance_calibration_streams_without_second_concatenate(monkeypatch):
    payload = {
        'object_cond': {
            'tpos_first_frame': _tpose(2),
            'canonical_joint_names': ['root', 'knee'],
        },
        'results': [
            {'motion': np.zeros((5, 2, FEAT), dtype=np.float32)},
            {'motion': np.ones((7, 2, FEAT), dtype=np.float32)},
        ],
    }

    def fail_concatenate(*args, **kwargs):
        raise AssertionError('unexpected concatenate in variance calibration')

    monkeypatch.setattr(dp.np, 'concatenate', fail_concatenate)

    calibration = dp._measure_variance_calibration([payload], dp._default_structural_prior_bank())

    assert set(calibration) == {'pos', 'rot', 'vel', 'stretch'}


def test_profile_transfers_to_motionless_skeleton_and_survives_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    frames = 3000
    motion = np.zeros((frames, 2, FEAT), dtype=np.float32)
    motion[:, 1, 9] = rng.normal(1.0, 0.25, size=frames)
    bank = dp._build_structural_prior_bank([_payload(['root', 'knee'], motion)])

    dp._save_structural_prior_bank(str(tmp_path), bank)
    loaded = dp._load_structural_prior_bank(
        os.path.join(str(tmp_path), dp.STRUCTURAL_NORM_PRIORS_FILE)
    )
    assert loaded['schema_version'] == 4
    assert set(loaded['variance_calibration']) == {'pos', 'rot', 'vel', 'stretch'}

    # A brand-new skeleton with NO motion, joint named 'knee', inherits the
    # anisotropic shape purely from the shared bank + its own (zero) T-pose.
    new_cond = _object_cond(['some_root', 'knee'])
    dp._apply_structural_stats_to_object_cond(new_cond, loaded)
    assert new_cond['norm_mean'][1, 9] == pytest.approx(1.0)
    assert new_cond['norm_std'][1, 9] == pytest.approx(
        float(loaded['by_canonical_name']['knee']['stretch']) * loaded['variance_calibration']['stretch']
    )
    assert new_cond['norm_std_source'] == 'structural_prior_bank_v4_role_aware_varcal'
    assert new_cond['norm_std_joint_sources'][1] == 'canonical:knee'
    # the motion-free skeleton inherited the shared calibration verbatim
    assert new_cond['norm_std_variance_calibration'] == loaded['variance_calibration']


@pytest.mark.parametrize('missing_field', ['offsets', 'rest_rotations', 'canon_joint_rot'])
def test_apply_structural_stats_requires_v4_skeleton_metadata(missing_field):
    bank = dp._default_structural_prior_bank()
    cond = _object_cond(['root', 'knee'])
    del cond[missing_field]

    with pytest.raises(ValueError, match='missing required v4 fields'):
        dp._apply_structural_stats_to_object_cond(cond, bank)


def test_schema_version_mismatch_raises(tmp_path):
    stale = dp._default_structural_prior_bank()
    stale['schema_version'] = 1
    path = os.path.join(str(tmp_path), dp.STRUCTURAL_NORM_PRIORS_FILE)
    np.save(path, stale)
    with pytest.raises(RuntimeError, match="schema mismatch"):
        dp._load_structural_prior_bank(path)


def test_missing_prior_bank_message_points_to_regeneration(tmp_path):
    path = os.path.join(str(tmp_path), dp.STRUCTURAL_NORM_PRIORS_FILE)
    with pytest.raises(FileNotFoundError) as exc_info:
        dp._load_structural_prior_bank(path)

    message = str(exc_info.value)
    assert dp.STRUCTURAL_NORM_PRIORS_FILE in message
    assert "tools/process_new_skeleton.py" in message
    assert "--training-cond-path" not in message


@pytest.mark.parametrize(
    ("canonical_name", "extra_cond", "expected_source", "expected_rot_std"),
    [
        (
            'Xtra 05 Nub Helper',
            {
                'helper_joint_indices': [1],
                'helper_joint_names': ['Xtra 05 Nub Helper'],
            },
            'role:nonroot',
            0.25,
        ),
        (
            'Weapon Locator',
            {},
            'canonical:weapon locator',
            dp._MIN_STRUCTURAL_SCALE,
        ),
    ],
)
def test_only_explicit_helpers_skip_canonical_priors(canonical_name, extra_cond, expected_source, expected_rot_std):
    bank = dp._default_structural_prior_bank()
    bank['by_canonical_name'][canonical_name.lower()] = dp._structural_scale_dict(
        pos=dp._MIN_STRUCTURAL_SCALE,
        rot=dp._MIN_STRUCTURAL_SCALE,
        vel=dp._MIN_STRUCTURAL_SCALE,
    )
    bank['by_semantic_group']['nonroot'] = dp._structural_scale_dict(pos=0.4, rot=0.4, vel=0.4)
    bank['by_role']['nonroot'] = dp._structural_scale_dict(pos=0.25, rot=0.25, vel=0.25)

    cond = _object_cond(['root', canonical_name])
    cond.update(extra_cond)
    dp._apply_structural_stats_to_object_cond(cond, bank)

    assert cond['norm_std_joint_sources'][1] == expected_source
    assert np.allclose(cond['norm_std'][1, 3:9], expected_rot_std)
    assert cond['norm_std'][1, 3] > dp._MIN_STRUCTURAL_SCALE
