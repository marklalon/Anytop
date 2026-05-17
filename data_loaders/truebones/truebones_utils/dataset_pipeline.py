"""Dataset pipeline & augmentation.

Top layer of the motion-processing pipeline. Orchestrates the full dataset
preprocessing workflow, statistics computation, topology analysis, data
augmentation, and skeleton-processing entry points.

Depends on: features.py, animation_utils.py
"""

from motion_lib import BVH, FBX
import numpy as np
import os
import sys
from os.path import join as pjoin
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import math
import bisect
from data_loaders.truebones.truebones_utils.param_utils import DEFAULT_DATASET_DIR, MAX_JOINTS, MAX_PATH_LEN, MOTION_DIR, FOOT_CONTACT_VEL_THRESH, BVHS_DIR, get_raw_data_dir, SNAKES
from .motion_labels import build_motion_labels, build_object_labels, infer_motion_labels_from_motion_name, write_motion_metadata
from .physics_joint_annotation import (
    _build_semantic_metadata,
    _rest_positions_from_offsets,
)
from .fbx_filename_rules import (
    find_tpose_reference_path,
    _normalize_action_name,
    _should_skip_anim,
)

from .animation_utils import (
    _canonical_name_for_bvh,
    _attach_joint_name_embeddings_to_cond,
    _extend_semantic_metadata_with_leaf_helpers,
    needs_bvh_position_channels,
    reorder_animation_to_dfs,
    _coerce_single_orientation_quat,
)

from .features import (
    get_common_features_from_T_pose,
    get_motion,
    infer_translation_root_index_from_features,
)


# Schema v2 (Tier 1): each prior-bank leaf additionally carries a unitless,
# mean-normalized per-channel anisotropy *profile* alongside the v1 scalar
# *magnitude*. Reconstructed per-channel std = magnitude * profile, so a leaf
# whose profile is all-ones reproduces v1 isotropic behavior exactly.
#
# Schema v3 (variance calibration / optimization "a"): the borrowed structural
# magnitudes systematically under/over-estimate true motion scale, so the
# normalized training data was NOT ~unit-variance and the diffusion cosine
# schedule's SNR assumption was violated (inflated, ill-conditioned loss). The
# bank now stores a per-feature-group multiplicative factor measured ONCE on
# the training set so that (motion - norm_mean) / norm_std has unit RMS. The
# factor is baked into the shared bank and applied uniformly to every skeleton
# (training and motion-free new ones alike), so the scheme stays
# motion-prior-free. Identity calibration {1,1,1} reproduces v2 exactly.
STRUCTURAL_NORM_SCHEMA_VERSION = 3
STRUCTURAL_NORM_PRIORS_FILE = "structural_norm_priors.npy"
_MIN_STRUCTURAL_SCALE = 1e-3
_STRUCTURAL_CONTACT_SCALE = 1.0
# Calibration is a std multiplier; clamp generously to bound pathological
# training-set RMS measurements without distorting the typical ~1-20x range.
_MIN_VARIANCE_CALIBRATION = 1e-2
_MAX_VARIANCE_CALIBRATION = 1e3
# Anisotropy ratios are clamped before mean-normalization so a near-static axis
# cannot collapse a channel's std toward zero (which would explode the
# normalized residual) or dominate the group.
_MIN_PROFILE_RATIO = 0.2
_MAX_PROFILE_RATIO = 5.0
# (key, start, stop) for the contiguous motion-feature groups that get an
# anisotropy profile. Contact (channel 12) stays a single scalar.
_PROFILE_GROUPS = (('pos', 0, 3), ('rot', 3, 9), ('vel', 9, 12))
_GROUP_WIDTHS = {'pos': 3, 'rot': 6, 'vel': 3}
_SEMANTIC_GROUP_KEYWORDS = (
    ('axial', ('pelvis', 'hip', 'spine', 'chest', 'torso', 'neck', 'head')),
    ('arm', ('shoulder', 'arm', 'forearm', 'elbow', 'hand', 'wrist', 'finger', 'thumb')),
    ('leg', ('thigh', 'leg', 'calf', 'shin', 'knee', 'foot', 'toe', 'ankle', 'hoof')),
    ('tail', ('tail', 'tip')),
    ('wing', ('wing', 'feather')),
    ('face', ('jaw', 'mouth', 'eye', 'ear', 'nose', 'snout', 'horn', 'brow')),
)


################## Statistics & Topology #####################

""" computes mean and std for a list of motions """
def get_mean_std(data):
    if len(data) > 0:
        Mean = data.mean(axis=0) # (Joints, 25)
        Std = data.std(axis=0) # # (Joints, 25)
        Std[0, :3] = Std[0, :3].mean() / 1.0 # all joints except root ric pos
        Std[0, 3:9] = Std[0, 3:9].mean() / 1.0 # all joints except root rotation
        Std[0, 9:12] = Std[0, 9:12].mean() / 1.0 # all joints except root local velocity

        Std[1:, :3] = Std[1:, :3].mean() / 1.0 # all joints except root ric pos
        Std[1:, 3:9] = Std[1:, 3:9].mean() / 1.0 # all joints except root rotation
        Std[1:, 9:12] = Std[1:, 9:12].mean() / 1.0 # all joints except root local velocity
        if len(Std[:, 12][Std[:, 12]!=0]) > 0:
            Std[:, 12][Std[:, 12]!=0] = Std[:, 12][Std[:, 12]!=0].mean() / 1.0 
        Std[:, 12][Std[:, 12]==0] = 1.0 # replace zeros with ones
        
        return Mean, Std


def _sanitize_profile(profile, width):
    """Coerce a raw per-channel ratio vector into a clean, unitless anisotropy
    profile: non-finite/non-positive entries -> 1, clamped to
    [_MIN_PROFILE_RATIO, _MAX_PROFILE_RATIO], then rescaled so the mean is
    exactly 1.0 (magnitude is carried by the scalar, shape by the profile).
    Returns a plain ``list[float]`` so the bank stays pickle/.item() clean.
    A length mismatch or empty input degrades to an isotropic ones-profile."""
    ones = [1.0] * int(width)
    if profile is None:
        return ones
    arr = np.asarray(profile, dtype=np.float64).reshape(-1)
    if arr.size != int(width):
        return ones
    arr = np.where(np.isfinite(arr) & (arr > 0.0), arr, 1.0)
    arr = np.clip(arr, _MIN_PROFILE_RATIO, _MAX_PROFILE_RATIO)
    mean = float(np.mean(arr))
    if not np.isfinite(mean) or mean <= 0.0:
        return ones
    return [float(v) for v in (arr / mean)]


def _structural_scale_dict(pos=1.0, rot=1.0, vel=1.0, contact=_STRUCTURAL_CONTACT_SCALE,
                           pos_profile=None, rot_profile=None, vel_profile=None):
    return {
        'pos': float(max(pos, _MIN_STRUCTURAL_SCALE)),
        'rot': float(max(rot, _MIN_STRUCTURAL_SCALE)),
        'vel': float(max(vel, _MIN_STRUCTURAL_SCALE)),
        'contact': float(max(contact, _MIN_STRUCTURAL_SCALE)),
        'pos_profile': _sanitize_profile(pos_profile, _GROUP_WIDTHS['pos']),
        'rot_profile': _sanitize_profile(rot_profile, _GROUP_WIDTHS['rot']),
        'vel_profile': _sanitize_profile(vel_profile, _GROUP_WIDTHS['vel']),
    }


def _identity_variance_calibration():
    return {'pos': 1.0, 'rot': 1.0, 'vel': 1.0}


def _default_structural_prior_bank():
    default_scales = _structural_scale_dict()
    return {
        'schema_version': STRUCTURAL_NORM_SCHEMA_VERSION,
        'feature_len': 13,
        'global_scales': dict(default_scales),
        'by_canonical_name': {},
        'by_semantic_group': {},
        'by_role': {
            'root': dict(default_scales),
            'nonroot': dict(default_scales),
        },
        # Identity until measured on a real training set -> default / single
        # object / no-data paths behave exactly like uncalibrated v2.
        'variance_calibration': _identity_variance_calibration(),
        'metadata': {
            'default_bank': True,
            'object_count': 0,
            'joint_examples': 0,
        },
    }


def _load_structural_prior_bank(priors_path=None):
    if priors_path is None:
        return _default_structural_prior_bank()
    if not os.path.isfile(priors_path):
        raise FileNotFoundError(
            f"Structural prior bank not found at {priors_path}. A new skeleton with no "
            f"motion priors requires the shared structural_norm_priors.npy that is written "
            f"next to the training dataset's cond.npy. Re-run dataset preprocessing on the "
            f"training set first so it emits {STRUCTURAL_NORM_PRIORS_FILE}, then rerun "
            f"tools/process_new_skeleton.py. Falling back to unit scales would silently "
            f"degrade the generated motion."
        )
    bank = np.load(priors_path, allow_pickle=True).item()
    if not isinstance(bank, dict):
        raise RuntimeError(f"Invalid structural prior bank: {priors_path}")
    if int(bank.get('schema_version', 0)) != STRUCTURAL_NORM_SCHEMA_VERSION:
        raise RuntimeError(
            f"Structural prior bank schema mismatch at {priors_path}: "
            f"expected {STRUCTURAL_NORM_SCHEMA_VERSION}, got {bank.get('schema_version')}. "
            f"Re-run training-set preprocessing to regenerate {STRUCTURAL_NORM_PRIORS_FILE} "
            f"with the current schema."
        )
    return bank


def _save_structural_prior_bank(save_dir, prior_bank):
    if prior_bank is None:
        return
    np.save(pjoin(save_dir, STRUCTURAL_NORM_PRIORS_FILE), prior_bank)


def _build_structural_norm_mean(tpos_first_frame):
    norm_mean = np.asarray(tpos_first_frame, dtype=np.float32).copy()
    norm_mean[:, 9:12] = 0.0
    norm_mean[:, 12] = 0.0
    return norm_mean


def _robust_scalar_scale(values):
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return _MIN_STRUCTURAL_SCALE
    median = np.median(flat)
    centered = flat - median
    mad = np.median(np.abs(centered))
    scale = float(mad * 1.4826)
    if not np.isfinite(scale) or scale < _MIN_STRUCTURAL_SCALE:
        scale = float(np.sqrt(np.mean(centered ** 2))) if centered.size else _MIN_STRUCTURAL_SCALE
    return max(scale, _MIN_STRUCTURAL_SCALE)


def _robust_per_channel_scale(values_2d):
    """Per-column robust scale of a (frames, channels) residual block, reusing
    the same MAD-based estimator as the pooled scalar path."""
    block = np.asarray(values_2d, dtype=np.float32)
    if block.ndim == 1:
        block = block[:, None]
    return np.asarray(
        [_robust_scalar_scale(block[:, c]) for c in range(block.shape[1])],
        dtype=np.float64,
    )


def _group_magnitude_and_profile(residual_block):
    """Split a residual block into (pooled magnitude, per-channel profile).

    The pooled magnitude is exactly the v1 scalar (``_robust_scalar_scale`` over
    the whole block), so magnitude transfer/fallback behavior is unchanged. The
    profile is the per-channel robust scale divided by that magnitude, i.e. the
    unitless *shape* of the joint's motion within the group."""
    magnitude = _robust_scalar_scale(residual_block)
    per_channel = _robust_per_channel_scale(residual_block)
    safe_mag = max(float(magnitude), _MIN_STRUCTURAL_SCALE)
    return magnitude, (per_channel / safe_mag)


def _joint_structural_scales(joint_motion, joint_anchor):
    residual = np.asarray(joint_motion, dtype=np.float32) - np.asarray(joint_anchor, dtype=np.float32)[None, :]
    pos_mag, pos_profile = _group_magnitude_and_profile(residual[:, 0:3])
    rot_mag, rot_profile = _group_magnitude_and_profile(residual[:, 3:9])
    vel_mag, vel_profile = _group_magnitude_and_profile(residual[:, 9:12])
    return _structural_scale_dict(
        pos=pos_mag,
        rot=rot_mag,
        vel=vel_mag,
        contact=_STRUCTURAL_CONTACT_SCALE,
        pos_profile=pos_profile,
        rot_profile=rot_profile,
        vel_profile=vel_profile,
    )


def _coarse_semantic_group(object_cond, joint_index):
    helper_joint_indices = {int(idx) for idx in object_cond.get('helper_joint_indices', [])}
    if int(joint_index) == 0:
        return 'root'
    if int(joint_index) in helper_joint_indices:
        return 'helper'

    contact_indices = {int(idx) for idx in object_cond.get('contact_joints', [])}
    end_effector_indices = {int(idx) for idx in object_cond.get('end_effector_joints', [])}
    if int(joint_index) in contact_indices and int(joint_index) in end_effector_indices:
        return 'distal_contact'
    if int(joint_index) in end_effector_indices:
        return 'distal'
    if int(joint_index) in contact_indices:
        return 'contact'

    canonical_names = object_cond.get('canonical_joint_names') or object_cond.get('joints_names') or []
    canonical_name = str(canonical_names[joint_index] if joint_index < len(canonical_names) else '').lower()
    for group_name, keywords in _SEMANTIC_GROUP_KEYWORDS:
        if any(keyword in canonical_name for keyword in keywords):
            return group_name
    return 'nonroot'


def _is_helper_like_joint(object_cond, joint_index):
    if int(joint_index) == 0:
        return False

    helper_joint_indices = {int(idx) for idx in object_cond.get('helper_joint_indices', [])}
    if int(joint_index) in helper_joint_indices:
        return True

    helper_joint_names = {
        str(name).strip().lower()
        for name in object_cond.get('helper_joint_names', [])
        if str(name).strip()
    }
    raw_names = object_cond.get('joints_names') or []
    canonical_names = object_cond.get('canonical_joint_names') or raw_names or []
    candidate_names = []
    if joint_index < len(raw_names):
        candidate_names.append(str(raw_names[joint_index]).strip().lower())
    if joint_index < len(canonical_names):
        candidate_names.append(str(canonical_names[joint_index]).strip().lower())

    for name in candidate_names:
        if not name:
            continue
        if name in helper_joint_names:
            return True
    return False


def _empty_scale_samples():
    return {
        'pos': [], 'rot': [], 'vel': [],
        'pos_profile': [], 'rot_profile': [], 'vel_profile': [],
    }


def _append_scale_sample(sample_dict, key, scales):
    bucket = sample_dict.setdefault(str(key), _empty_scale_samples())
    for group_name in ('pos', 'rot', 'vel'):
        bucket[group_name].append(float(scales[group_name]))
        bucket[f'{group_name}_profile'].append(
            _sanitize_profile(scales.get(f'{group_name}_profile'), _GROUP_WIDTHS[group_name])
        )


def _finalize_scale_bucket(bucket, fallback_scales):
    finalized = {}
    for group_name in ('pos', 'rot', 'vel'):
        samples = np.asarray(bucket.get(group_name, []), dtype=np.float32)
        samples = samples[np.isfinite(samples) & (samples > 0)]
        if samples.size == 0:
            finalized[group_name] = float(fallback_scales[group_name])
        else:
            finalized[group_name] = float(max(np.median(samples), _MIN_STRUCTURAL_SCALE))

        width = _GROUP_WIDTHS[group_name]
        profile_samples = bucket.get(f'{group_name}_profile', [])
        if profile_samples:
            stacked = np.asarray(profile_samples, dtype=np.float64)
            if stacked.ndim == 2 and stacked.shape[1] == width:
                finalized[f'{group_name}_profile'] = _sanitize_profile(
                    np.median(stacked, axis=0), width
                )
            else:
                finalized[f'{group_name}_profile'] = _sanitize_profile(
                    fallback_scales.get(f'{group_name}_profile'), width
                )
        else:
            finalized[f'{group_name}_profile'] = _sanitize_profile(
                fallback_scales.get(f'{group_name}_profile'), width
            )
    finalized['contact'] = float(fallback_scales.get('contact', _STRUCTURAL_CONTACT_SCALE))
    return finalized


def _build_structural_prior_bank(payloads):
    name_samples = {}
    semantic_samples = {}
    role_samples = {
        'root': _empty_scale_samples(),
        'nonroot': _empty_scale_samples(),
    }
    global_samples = _empty_scale_samples()
    object_count = 0
    joint_examples = 0

    for payload in payloads:
        if payload is None:
            continue
        object_cond = payload['object_cond']
        results = payload.get('results') or []
        if not results:
            continue

        motion_tensor = np.concatenate(
            [np.asarray(result['motion'], dtype=np.float32) for result in results],
            axis=0,
        )
        norm_mean = _build_structural_norm_mean(object_cond['tpos_first_frame'])
        canonical_names = object_cond.get('canonical_joint_names') or object_cond.get('joints_names') or []

        object_count += 1
        for joint_index in range(motion_tensor.shape[1]):
            scales = _joint_structural_scales(motion_tensor[:, joint_index, :], norm_mean[joint_index])
            canonical_name = str(canonical_names[joint_index] if joint_index < len(canonical_names) else '').strip().lower() or '__unknown__'
            semantic_group = _coarse_semantic_group(object_cond, joint_index)
            role_key = 'root' if int(joint_index) == 0 else 'nonroot'
            _append_scale_sample(name_samples, canonical_name, scales)
            _append_scale_sample(semantic_samples, semantic_group, scales)
            _append_scale_sample(role_samples, role_key, scales)
            global_samples['pos'].append(float(scales['pos']))
            global_samples['rot'].append(float(scales['rot']))
            global_samples['vel'].append(float(scales['vel']))
            joint_examples += 1

    default_bank = _default_structural_prior_bank()
    global_scales = _finalize_scale_bucket(global_samples, default_bank['global_scales'])
    by_role = {
        role_key: _finalize_scale_bucket(role_samples.get(role_key, _empty_scale_samples()), global_scales)
        for role_key in ('root', 'nonroot')
    }

    bank = {
        'schema_version': STRUCTURAL_NORM_SCHEMA_VERSION,
        'feature_len': 13,
        'global_scales': global_scales,
        'by_canonical_name': {
            key: _finalize_scale_bucket(bucket, global_scales)
            for key, bucket in name_samples.items()
        },
        'by_semantic_group': {
            key: _finalize_scale_bucket(bucket, by_role['root'] if key == 'root' else by_role['nonroot'])
            for key, bucket in semantic_samples.items()
        },
        'by_role': by_role,
        # Identity placeholder; measured below against the just-built bank.
        'variance_calibration': _identity_variance_calibration(),
        'metadata': {
            'default_bank': False,
            'object_count': int(object_count),
            'joint_examples': int(joint_examples),
        },
    }
    # Second pass: with magnitudes+profiles fixed, measure the residual RMS the
    # diffusion model would actually see and bake in the correction so it sees
    # ~unit variance instead.
    bank['variance_calibration'] = _measure_variance_calibration(payloads, bank)
    return bank


def _resolve_joint_structural_scales(object_cond, joint_index, prior_bank):
    canonical_names = object_cond.get('canonical_joint_names') or object_cond.get('joints_names') or []
    canonical_name = str(canonical_names[joint_index] if joint_index < len(canonical_names) else '').strip().lower() or '__unknown__'
    semantic_group = _coarse_semantic_group(object_cond, joint_index)
    role_key = 'root' if int(joint_index) == 0 else 'nonroot'
    helper_like = _is_helper_like_joint(object_cond, joint_index)

    by_name = prior_bank.get('by_canonical_name', {})
    if not helper_like and canonical_name in by_name:
        return by_name[canonical_name], f'canonical:{canonical_name}'

    by_semantic = prior_bank.get('by_semantic_group', {})
    if semantic_group in by_semantic and not (helper_like and semantic_group == 'helper'):
        return by_semantic[semantic_group], f'semantic:{semantic_group}'

    by_role = prior_bank.get('by_role', {})
    if role_key in by_role:
        return by_role[role_key], f'role:{role_key}'

    return prior_bank.get('global_scales', _default_structural_prior_bank()['global_scales']), 'global'


def _resolve_variance_calibration(prior_bank):
    """Per-feature-group std multiplier; missing/identity -> no-op (== v2)."""
    raw = prior_bank.get('variance_calibration') if isinstance(prior_bank, dict) else None
    calibration = {}
    for group_name in ('pos', 'rot', 'vel'):
        value = 1.0
        if isinstance(raw, dict) and group_name in raw:
            try:
                value = float(raw[group_name])
            except (TypeError, ValueError):
                value = 1.0
        if not np.isfinite(value) or value <= 0.0:
            value = 1.0
        calibration[group_name] = float(
            np.clip(value, _MIN_VARIANCE_CALIBRATION, _MAX_VARIANCE_CALIBRATION)
        )
    return calibration


def _compute_object_norm_std(object_cond, prior_bank, apply_calibration=True):
    """Resolve the (J, 13) structural norm_std for one object.

    With ``apply_calibration=False`` this returns the raw v2 magnitude*profile
    std (used to *measure* the training-set RMS before the calibration factor
    exists); the apply path uses ``True`` so every skeleton, including
    motion-free ones, inherits the shared per-group calibration."""
    norm_mean = _build_structural_norm_mean(object_cond['tpos_first_frame'])
    norm_std = np.ones_like(norm_mean, dtype=np.float32)
    calibration = _resolve_variance_calibration(prior_bank) if apply_calibration else None
    joint_sources = []
    for joint_index in range(norm_mean.shape[0]):
        scales, source = _resolve_joint_structural_scales(object_cond, joint_index, prior_bank)
        for group_name, start, stop in _PROFILE_GROUPS:
            magnitude = float(scales[group_name])
            # A v1 leaf (or default bank) has no *_profile -> isotropic ones,
            # so this reduces exactly to the previous scalar broadcast.
            profile = np.asarray(
                _sanitize_profile(scales.get(f'{group_name}_profile'), _GROUP_WIDTHS[group_name]),
                dtype=np.float32,
            )
            channel_std = magnitude * profile
            if calibration is not None:
                channel_std = channel_std * calibration[group_name]
            norm_std[joint_index, start:stop] = np.maximum(channel_std, _MIN_STRUCTURAL_SCALE)
        norm_std[joint_index, 12] = float(scales.get('contact', _STRUCTURAL_CONTACT_SCALE))
        joint_sources.append(source)
    return norm_mean, norm_std, joint_sources


def _measure_variance_calibration(payloads, prior_bank):
    """Pooled per-feature-group RMS of the *uncalibrated* normalized residual
    over the whole training set. Calibrating norm_std by this factor makes
    (motion - norm_mean) / norm_std unit-RMS, restoring the diffusion cosine
    schedule's variance assumption."""
    sumsq = {g: 0.0 for g, _, _ in _PROFILE_GROUPS}
    count = {g: 0 for g, _, _ in _PROFILE_GROUPS}
    for payload in payloads:
        if payload is None:
            continue
        object_cond = payload['object_cond']
        results = payload.get('results') or []
        if not results:
            continue
        norm_mean, norm_std, _ = _compute_object_norm_std(
            object_cond, prior_bank, apply_calibration=False
        )
        norm_mean64 = norm_mean.astype(np.float64, copy=False)
        norm_std64 = norm_std.astype(np.float64, copy=False)
        # Stream each motion clip to avoid a second dataset-scale concatenate.
        for result in results:
            motion_tensor = np.asarray(result['motion'], dtype=np.float64)
            residual = motion_tensor - norm_mean64[None, :, :]
            normalized = residual / norm_std64[None, :, :]
            for group_name, start, stop in _PROFILE_GROUPS:
                block = normalized[:, :, start:stop]
                block = block[np.isfinite(block)]
                if block.size:
                    sumsq[group_name] += float(np.sum(block ** 2))
                    count[group_name] += int(block.size)

    calibration = {}
    for group_name in ('pos', 'rot', 'vel'):
        if count[group_name] > 0:
            rms = float(np.sqrt(sumsq[group_name] / count[group_name]))
        else:
            rms = 1.0
        if not np.isfinite(rms) or rms <= 0.0:
            rms = 1.0
        calibration[group_name] = float(
            np.clip(rms, _MIN_VARIANCE_CALIBRATION, _MAX_VARIANCE_CALIBRATION)
        )
    print(
        "[structural_stats] variance calibration (pre-cal training RMS): "
        f"pos={calibration['pos']:.4f} rot={calibration['rot']:.4f} "
        f"vel={calibration['vel']:.4f}"
    )
    return calibration


def _apply_structural_stats_to_object_cond(object_cond, prior_bank):
    norm_mean, norm_std, joint_sources = _compute_object_norm_std(
        object_cond, prior_bank, apply_calibration=True
    )
    object_cond['tpos_first_frame'] = np.asarray(object_cond['tpos_first_frame'], dtype=np.float32)
    object_cond['norm_mean'] = norm_mean.astype(np.float32, copy=False)
    object_cond['norm_std'] = norm_std.astype(np.float32, copy=False)
    object_cond['norm_schema_version'] = int(STRUCTURAL_NORM_SCHEMA_VERSION)
    object_cond['norm_mean_source'] = 'tpose_anchor_v1'
    object_cond['norm_std_source'] = 'structural_prior_bank_v3_anisotropic_varcal'
    object_cond['norm_std_variance_calibration'] = _resolve_variance_calibration(prior_bank)
    object_cond['norm_std_joint_sources'] = joint_sources
    return object_cond


def _apply_structural_stats_to_payloads(payloads, prior_bank):
    for payload in payloads:
        if payload is None:
            continue
        _apply_structural_stats_to_object_cond(payload['object_cond'], prior_bank)


""" compures Relations and Distance marices"""
def create_topology_edge_relations(parents, max_path_len = 5): # joint j+1 contains len(j, j+1)
    edge_types = {'self':0, 'parent':1, 'child':2, 'sibling':3, 'no_relation':4, 'end_effector':5, 'ts_token_conn': 6}
    n = len(parents)
    topo_rel = np.zeros((n, n))
    edge_rel = np.ones((n, n)) * edge_types['no_relation'] 
    for i in range(n):
        parent = parents[i]
        ee = True
        for j in range(n):
            parent_j = parents[j]
            """Update edge type"""
            edge_type = edge_types['no_relation']
            if i == j: #self
                edge_type = edge_types['self'] 
            elif parent_j == i: #child
                ee=False
                edge_type = edge_types['child']
            elif j == parent: #parent
                edge_type = edge_types['parent'] 
            elif parent_j == parent: #sibling
                edge_type = edge_types['sibling']
            edge_rel[i, j] = edge_type

            """Update path length type"""
            
            if i == j:
                topo_rel[i, j] = 0      
            elif j < i:
                topo_rel[i, j] = topo_rel[j, i]
            elif parent_j == i: # parent-child relation
                topo_rel[i, j] = 1
            else: #any other 
                topo_rel[i, j] = topo_rel[i, parent_j] + 1
        if ee:
            edge_rel[i, i] = edge_types['end_effector']
            
    topo_rel[topo_rel > max_path_len] = max_path_len
    return edge_rel, topo_rel


################## Parents to kinematic chains ###################
def reverse_insort(a, x, lo=0, hi=None):
    """Insert item x in list a, and keep it reverse-sorted assuming a
    is reverse-sorted.

    If x is already in a, insert it to the right of the rightmost x.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if x > a[mid]: hi = mid
        else: lo = mid+1
    a.insert(lo, x)


def parents2kinchains(parents, policy = 'h_first'):
    chains = list()
    children_dict = {i:[] for i in range(len(parents))}
    for j,p in enumerate(parents[1: ], start=1):
        if policy == 'h_first':
            reverse_insort(children_dict[p], j)
        else:
            bisect.insort(children_dict[p], j)
    recursion_kinchains([], 0, children_dict, chains, policy)
    return chains


def recursion_kinchains(chain, j, children_dict, chains, policy):
    children = children_dict[j]
    if len(children) == 0: #ee
        chain.append(j)
        chains.append(chain) 
    elif len(children) == 1:
        chain.append(j)
        recursion_kinchains(chain, children[0], children_dict, chains, policy)
    else:
        chain.append(j)
        if policy == 'h_first':
            main_child = max(children)
        else:
            main_child = min(children)
        for child in children:
            if child == main_child:
                recursion_kinchains(chain, child, children_dict, chains, policy)
            else:
                recursion_kinchains([j], child, children_dict, chains, policy)  


""" returns policy for extracting kinematic chains from parent array, 
in attempt to divide the skeleton to meaningful kinchains. h_first mean the head joints are at the 
beggining of the parent array"""
def object_policy(obj):
    if obj in ["Mousey_m", "MouseyNoFingers", "Scorpion", "Raptor2"]:
        return "l_first"
    else:
        return "h_first"


################## Augmentations ##########################
def remove_joints_augmentation(data, removal_rate, mean, std):
    motion, m_length, object_type, parents, joints_graph_dist, joints_relations, tpos_first_frame, offsets, joints_names_embs, kinematic_chains = data['motion'], data['length'], data['object_type'], data['parents'], data['joints_graph_dist'], data['joints_relations'], data['tpos_first_frame'], data['offsets'], data['joints_names_embs'], data['kinematic_chains']
    ee = [chain[-1] for chain in kinematic_chains]
    possible_feet = np.unique(np.where(motion[..., -1] > 0)[1])
    if object_type in SNAKES:
        possible_feet=[]
    removal_options = [j for j in ee if j not in possible_feet]
    # removal_rate = min(1.0, (removal_rate*len(parents)) / len(removal_options))
    remove_joints = sorted(random.sample(removal_options, math.floor(len(removal_options) * removal_rate)), reverse=True)
    motion = np.delete(motion, remove_joints, axis=1)
    new_ee = [parents[j] for j in remove_joints if np.count_nonzero(parents == parents[j]) == 1]
    for el in new_ee:
        joints_relations[el, el] = 5    
    parents = np.delete(parents, remove_joints, axis=0)
    joints_relations = np.delete(np.delete(joints_relations, remove_joints, axis=0), remove_joints, axis=1)
        
    for rj in remove_joints:
        parents[parents > rj] -= 1
    joints_graph_dist = np.delete(np.delete(joints_graph_dist, remove_joints, axis=0), remove_joints, axis=1)
    tpos_first_frame = np.delete(tpos_first_frame, remove_joints, axis=0)
    offsets = np.delete(offsets, remove_joints, axis=0)
    joints_names_embs = np.delete(joints_names_embs, remove_joints, axis=0)
    mean = np.delete(mean, remove_joints, axis=0)
    std = np.delete(std, remove_joints, axis=0)
    object_type = f'{object_type}__remove{remove_joints}'
    return motion, m_length, object_type, parents, joints_graph_dist, joints_relations, tpos_first_frame, offsets, joints_names_embs, kinematic_chains, mean, std


def add_joint_augmentation(data, mean, std):
    motion, m_length, object_type, parents, joints_graph_dist, joints_relations, tpos_first_frame, offsets, joints_names_embs, kinematic_chains = data['motion'], data['length'], data['object_type'], data['parents'], data['joints_graph_dist'], data['joints_relations'], data['tpos_first_frame'], data['offsets'], data['joints_names_embs'], data['kinematic_chains']
    n_joints = motion.shape[1]
    n_frames = motion.shape[0]
    # added joint mut follow:
    # j has exactly 1 child 
    # j parent is not the root joint
    # j is not the root joint
    possible_joints_to_add = [j for j in range(1, n_joints) if np.count_nonzero(joints_relations[j] == 2) == 1 and joints_relations[j,0] != 1]
    if len(possible_joints_to_add) == 0:
        return motion, m_length, object_type, parents, joints_graph_dist, joints_relations, tpos_first_frame, offsets, joints_names_embs, kinematic_chains, mean, std
    add_j = random.choice(possible_joints_to_add)
    # motion features
    j_feats = motion[:, add_j].copy()
    p_feats = motion[:, parents[add_j]]
    new_feats = ((j_feats + p_feats)/2).copy()
    new_feats[..., 3:9] = j_feats[..., 3:9].copy() # rotations
    new_feats[..., 12] = j_feats[..., 12].copy() # feet 
    j_feats[..., 3:9] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])[None].repeat(n_frames, axis=0)
    
    # tpos features
    tpos_j_feats = tpos_first_frame[add_j].copy()
    tpos_p_feats = tpos_first_frame[parents[add_j]]
    tpos_new_feats = ((tpos_j_feats + tpos_p_feats)/2)
    tpos_new_feats[3:9] = tpos_j_feats[3:9].copy() # rotations
    tpos_new_feats[12] = tpos_j_feats[12] # feet 
    tpos_j_feats[3:9] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    
    # mean features
    mean_j_feats = mean[add_j].copy()
    mean_p_feats = mean[parents[add_j]]
    mean_new_feats = ((mean_j_feats + mean_p_feats)/2).copy()
    mean_new_feats[3:9] = mean_j_feats[3:9].copy() # rotations
    mean_new_feats[12] = mean_j_feats[12] # feet 
    mean_j_feats[3:9] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    
    # std features
    std_new_feats = std[add_j].copy()
    
    # joints names embs features 
    emb_j_feats = joints_names_embs[add_j]
    emb_p_feats = joints_names_embs[parents[add_j]]
    emb_new_feats = (emb_j_feats + emb_p_feats)/2
    
    # apply augmentation
    #motion
    augmented = np.concatenate([motion[:, :add_j], new_feats[:, None], j_feats[:, None], motion[:, add_j+1:]], axis=1).copy()
    #tpos_first_frame
    tpos_first_frame_augmented = np.vstack([tpos_first_frame[:add_j], tpos_new_feats[None], tpos_j_feats[None], tpos_first_frame[add_j+1:]]).copy()
    #mean TODO: AUGMENT LIKE MOTION AND TPOS 
    mean_augmented = np.vstack([mean[:add_j], mean_new_feats[None], mean_j_feats[None], mean[add_j+1:]]).copy()
    #std TODO: AUGMENT LIKE MOTION AND TPOS 
    std_augmented = np.vstack([std[:add_j], std_new_feats[None], std[add_j:]]).copy()
    #joints_names_embs
    joints_names_embs_augmented = np.vstack([joints_names_embs[:add_j], emb_new_feats[None], joints_names_embs[add_j:]]).copy()
    # parents 
    augmented_parents = parents.copy()
    augmented_parents[augmented_parents >= add_j] += 1
    augmented_parents = augmented_parents.tolist()
    augmented_parents = np.array(augmented_parents[:add_j] + [add_j] + augmented_parents[add_j:])

    # topology conditions 
    relations, graph_dist = create_topology_edge_relations(augmented_parents.tolist(), max_path_len = MAX_PATH_LEN)
    
    # all others 
    offsets = np.vstack([offsets[:add_j], offsets[add_j]/2, offsets[add_j]/2, offsets[add_j+1:]])
    object_type = f'{object_type}__add{add_j}'
    return augmented, m_length, object_type, augmented_parents, graph_dist, relations, tpos_first_frame_augmented, offsets, joints_names_embs_augmented, kinematic_chains, mean_augmented, std_augmented


################## Dataset Pipeline #####################

def _process_motion_file(file_path, object_type, max_joints,
                         offsets, foot_indices, tpos_rots, scale_factor,
                         helper_metadata, orientation_quat):
    local_errors = dict()
    # Load the animation file (FBX/GLB/GLTF) once; pass it as `preloaded` to every get_motion call so that
    raw_anim, names, frame_time = FBX.load(file_path)
    anim_len = len(raw_anim)
    begin = 0
    file_max_joints = max_joints
    file_results = []
    file_motion_errors = []

    while begin < anim_len:
        if anim_len - begin > 240:
            slice_ind = begin + 200
        else:
            slice_ind = anim_len

        motion, parents, file_max_joints, new_anim, export_anim, is_loop, translation_root_index, root_translation_xz = get_motion(
            file_path,
            FOOT_CONTACT_VEL_THRESH,
            object_type,
            file_max_joints,
            offsets,
            foot_indices,
            tpos_rots,
            local_errors,
            scale_factor=scale_factor,
            orientation_quat=orientation_quat,
            slice_inds=[begin, slice_ind],
            preloaded=(raw_anim, names),
            helper_metadata=helper_metadata,
        )
        current_begin = begin
        begin = slice_ind

        if motion is None:
            err_msg = f"[FAIL] Object '{object_type}', file: {file_path}, slice {current_begin}:{slice_ind}"
            file_motion_errors.append(err_msg)
            continue

        _, file_name = os.path.split(file_path)
        raw_action = file_name.split('.')[0]
        raw_action = _normalize_action_name(object_type, raw_action)
        file_results.append({
            'action': raw_action,
            'motion': motion,
            'parents': parents,
            'new_anim': new_anim,
            'export_anim': export_anim,
            'names': names,
            'frame_time': frame_time,
            'is_loop': is_loop,
            'translation_root_index': translation_root_index,
            'root_translation_xz': root_translation_xz,
            'source_fbx_path': file_path,
            'slice_range': (current_begin, slice_ind),
            'motion_labels': build_motion_labels(object_type, raw_action),
        })

    return {
        'errors': local_errors,
        'max_joints': file_max_joints,
        'results': file_results,
        'motion_errors': file_motion_errors,
    }


def _attach_orientation_reference_metadata(
    object_cond,
    orientation_quat,
    forward_joint_index,
    forward_base_joint_index,
    orientation_reference_fbx_path,
):
    orientation_qs = _coerce_single_orientation_quat(orientation_quat).qs[0]
    object_cond['orientation_quat'] = orientation_qs.reshape(4)
    object_cond['forward_joint_index'] = int(forward_joint_index) if forward_joint_index is not None else None
    object_cond['forward_base_joint_index'] = int(forward_base_joint_index) if forward_base_joint_index is not None else None
    object_cond['orientation_reference_fbx_path'] = (
        os.path.abspath(orientation_reference_fbx_path)
        if orientation_reference_fbx_path
        else None
    )


def _build_motion_metadata_entry(result, motion_file_name):
    motion_labels = dict(result['motion_labels'])
    motion_labels['motion_name'] = motion_file_name
    motion_labels['is_loop'] = result.get('is_loop', False)

    translation_root_index = result.get('translation_root_index')
    if translation_root_index is not None:
        motion_labels['translation_root_index'] = int(translation_root_index)

    source_fbx_path = result.get('source_fbx_path')
    if source_fbx_path:
        motion_labels['source_fbx_path'] = os.path.abspath(source_fbx_path)

    source_frame_range = result.get('slice_range')
    if source_frame_range is not None:
        motion_labels['source_frame_range'] = [
            int(source_frame_range[0]),
            int(source_frame_range[1]),
        ]

    return motion_labels


"""Load T-pose FBX, build the shared cond dict, and return all values callers need."""
def _build_tpose_cond(object_type, t_pos_path, face_joints, max_joints=MAX_JOINTS):
    squared_positions_error = dict()
    tp = get_common_features_from_T_pose(
        t_pos_path,
        object_type,
        face_joints=face_joints,
        augment_leaf_rotation_helpers=True,
        max_joints=MAX_JOINTS,
    )
    character_scale_factor = float(tp.scale_factor)
    t_pos_motion, parents, max_joints, new_anim, _export_anim, _tpos_is_loop, _tpos_translation_root_index, _tpos_root_translation_xz = get_motion(
        tp.tpos_anim,
        FOOT_CONTACT_VEL_THRESH,
        object_type,
        max_joints,
        tp.offsets,
        tp.foot_indices,
        tp.tpos_rots,
        squared_positions_error,
        scale_factor=character_scale_factor,
        orientation_quat=tp.orientation_quat,
        helper_metadata=tp.helper_metadata,
        animation_input_is_tpose_aligned=False,
    )
    rest_positions = _rest_positions_from_offsets(tp.offsets, parents)
    original_joint_count = int(tp.helper_metadata['original_joint_count'])
    base_semantic_metadata = _build_semantic_metadata(
        tp.names[:original_joint_count],
        parents[:original_joint_count],
        tp.offsets[:original_joint_count],
        rest_positions=rest_positions[:original_joint_count],
    )
    semantic_metadata = _extend_semantic_metadata_with_leaf_helpers(
        base_semantic_metadata,
        tp.names,
        tp.helper_metadata,
    )
    object_cond = dict()
    object_cond['tpos_first_frame'] = t_pos_motion[0]
    joint_relations, joints_graph_dist = create_topology_edge_relations(tp.tpos_anim.parents, max_path_len=MAX_PATH_LEN)
    object_cond['joint_relations'] = joint_relations
    object_cond['joints_graph_dist'] = joints_graph_dist
    object_cond['object_type'] = object_type
    object_cond['parents'] = parents
    object_cond['offsets'] = tp.offsets
    object_cond['joints_names'] = tp.names
    object_cond['canonical_joint_names'] = semantic_metadata['canonical_joint_names']
    object_cond['canonical_bvh_joint_names'] = [
        _canonical_name_for_bvh(canonical_name, raw_name)
        for canonical_name, raw_name in zip(semantic_metadata['canonical_joint_names'], tp.names)
    ]
    object_cond['face_joints'] = list(tp.face_joints)
    object_cond['face_joint_names'] = [tp.names[index] for index in tp.face_joints]
    _attach_orientation_reference_metadata(
        object_cond,
        tp.orientation_quat,
        tp.forward_joint_index,
        tp.forward_base_joint_index,
        t_pos_path,
    )
    object_cond['end_effector_joints'] = semantic_metadata['end_effector_joints']
    object_cond['end_effector_names'] = semantic_metadata['end_effector_names']
    object_cond['contact_joints'] = semantic_metadata['contact_joints']
    object_cond['contact_joint_names'] = semantic_metadata['contact_joint_names']
    object_cond['contact_joint_source'] = semantic_metadata['contact_joint_source']
    object_cond['joint_side_labels'] = semantic_metadata['joint_side_labels']
    object_cond['symmetry_partner_indices'] = semantic_metadata['symmetry_partner_indices']
    object_cond['symmetric_joint_pairs'] = semantic_metadata['symmetric_joint_pairs']
    object_cond['symmetric_joint_pair_names'] = semantic_metadata['symmetric_joint_pair_names']
    object_cond['mirror_disabled_joint_indices'] = semantic_metadata['mirror_disabled_joint_indices']
    object_cond['mirror_disabled_joint_names'] = semantic_metadata['mirror_disabled_joint_names']
    object_cond['mirror_disabled_warnings'] = semantic_metadata['mirror_disabled_warnings']
    object_cond['is_symmetric'] = semantic_metadata['is_symmetric']
    object_cond['original_joint_count'] = int(tp.helper_metadata['original_joint_count'])
    object_cond['original_leaf_joint_indices'] = list(tp.helper_metadata['original_leaf_joint_indices'])
    object_cond['helper_joint_indices'] = list(tp.helper_metadata['helper_joint_indices'])
    object_cond['helper_joint_names'] = list(tp.helper_metadata['helper_joint_names'])
    object_cond['helper_joint_count'] = int(tp.helper_metadata['helper_joint_count'])
    object_cond['helper_source_leaf_indices'] = list(tp.helper_metadata['helper_source_leaf_indices'])
    object_cond['unaugmented_leaf_indices'] = list(tp.helper_metadata['unaugmented_leaf_indices'])
    object_cond['leaf_rotation_helper_suffix'] = tp.helper_metadata['leaf_rotation_helper_suffix']
    object_cond['scale_factor'] = character_scale_factor
    object_cond['axial_avg_len'] = float(tp.axial_avg_len)
    object_cond['kinematic_chains'] = parents2kinchains(parents, object_policy(object_type))
    object_cond.update(build_object_labels(object_type))
    return object_cond, tp, t_pos_motion, parents, semantic_metadata, character_scale_factor, squared_positions_error, max_joints


"""Build the T-pose cond dict from a single FBX file (no motion files needed)."""
def _build_tpose_only_cond(object_type, t_pos_path, face_joints, structural_prior_bank=None):
    object_cond, *_unused, max_joints = _build_tpose_cond(
        object_type, t_pos_path, face_joints,
    )
    _apply_structural_stats_to_object_cond(
        object_cond,
        structural_prior_bank or _default_structural_prior_bank(),
    )
    return object_cond, max_joints


"""Prepare processed tensors for all the files of a given object without writing them to disk yet."""
def _prepare_object_outputs(object_type, max_joints, face_joints=None, fbxs_dir=None, t_pos_path=None, max_files=None, raw_data_dir=None):
    object_cond = dict()
    if fbxs_dir is None:
        fbxs_dir = pjoin(get_raw_data_dir(raw_data_dir), object_type)
    if not os.path.isdir(fbxs_dir):
        print(f'skipping {object_type}: raw animation directory not found at {fbxs_dir}')
        return None
    anim_files = sorted([pjoin(fbxs_dir, f) for f in os.listdir(fbxs_dir) if f.lower().endswith(('.fbx', '.glb', '.gltf'))])
    if len(anim_files) == 0:
        print(f'skipping {object_type}: no animation files (.fbx/.glb/.gltf) found in {fbxs_dir}')
        return None
    ## get a character-level orientation reference clip
    if t_pos_path is None or t_pos_path == '':
        t_pos_path = find_tpose_reference_path(anim_files)
    else:
        # removes T-pose file from anim_files, as it represents a static pose and should be used only for
        # extracting common characteristics. If this is not the case, disable this part
        anim_files.remove(t_pos_path)
    if max_files is not None:
        anim_files = anim_files[:max_files]

    # Filter out files with no inferable action name or all-in-one animation bundles
    anim_files = [f for f in anim_files if not _should_skip_anim(f, object_type)]
    if len(anim_files) == 0:
        print(f'skipping {object_type}: no valid animation files after filtering')
        return None

    squared_positions_error = dict()
    object_cond, tp, t_pos_motion, parents, semantic_metadata, character_scale_factor, _, max_joints = _build_tpose_cond(
        object_type, t_pos_path, face_joints, max_joints=max_joints,
    )
    all_tensors = list()

    # Animation loading via bpy is single-threaded inside a process because clear_scene
    # mutates global Blender state, so file-level parallelism is intentionally removed.
    print(f'processing {len(anim_files)} animation files for {object_type} (serial — bpy is single-threaded)', flush=True)

    def process_file(file_path):
        print("processing file: " + file_path, flush=True)
        return _process_motion_file(
            file_path,
            object_type,
            max_joints,
            tp.offsets,
            tp.foot_indices,
            tp.tpos_rots,
            character_scale_factor,
            tp.helper_metadata,
            orientation_quat=tp.orientation_quat,
        )

    file_outputs = [process_file(file_path) for file_path in anim_files]

    files_counter = 0
    frames_counter = 0
    prepared_results = []
    all_motion_errors = []
    for file_output in file_outputs:
        squared_positions_error.update(file_output['errors'])
        max_joints = max(max_joints, file_output['max_joints'])
        all_motion_errors.extend(file_output.get('motion_errors', []))
        for result in file_output['results']:
            result['canonical_names'] = list(object_cond['canonical_bvh_joint_names'])
            prepared_results.append(result)

    if len(prepared_results) == 0:
        print(
            f"\x1b[33m[WARN] skipping {object_type}: no valid motion tensors were produced\x1b[0m"
        )
        return None

    for result in prepared_results:
        motion = result['motion']
        all_tensors.append(motion)
        files_counter += 1
        frames_counter += motion.shape[0]

    return {
        'object_type': object_type,
        'object_cond': object_cond,
        'errors': squared_positions_error,
        'max_joints': max_joints,
        'results': prepared_results,
        'files_counter': files_counter,
        'frames_counter': frames_counter,
        'face_joints': face_joints,
        'motion_errors': all_motion_errors,
    }


"""Write a prepared object payload to disk with stable sequential clip naming."""
def _write_object_outputs(save_dir, object_payload, files_counter):
    object_type = object_payload['object_type']
    frames_counter = 0
    motion_metadata = {}

    for result in object_payload['results']:
        motion = result['motion']
        files_counter += 1
        frames_counter += motion.shape[0]
        name = object_type + "_" + result['action'] + "_" + str(files_counter)
        motion_file_name = name + '.npy'
        np.save(pjoin(save_dir, MOTION_DIR, motion_file_name), motion)
        # Export the visually faithful processed animation rather than the
        # T-pose-reparameterized training animation. The latter preserves global
        # positions under this repo's FK but can look distorted in external BVH
        # viewers because its local position/offset decomposition is training-oriented.
        anim_obj = result['export_anim']
        bvh_names = list(result.get('canonical_names', result['names']))
        anim_obj, bvh_names = reorder_animation_to_dfs(anim_obj, bvh_names)
        BVH.save(
            pjoin(save_dir, BVHS_DIR, name + '.bvh'),
            anim_obj,
            bvh_names,
            frametime=result.get('frame_time', 1.0 / 24.0),
            positions=needs_bvh_position_channels(anim_obj),
        )

        motion_labels = _build_motion_metadata_entry(result, motion_file_name)
        motion_metadata[motion_file_name] = motion_labels

    return files_counter, frames_counter, motion_metadata


def _write_dataset_artifacts(save_dir, cond, motion_metadata, objects_counter, max_joints, files_counter, frames_counter, squared_positions_error, structural_prior_bank=None):
    print('Total clips: %d, Frames: %d, Duration: %fm' %(files_counter, frames_counter, frames_counter / 12.5 / 60))
    print('max joints: %d' %(max_joints))
    text_file = open(pjoin(save_dir, 'metadata.txt'), "w")
    n = text_file.write('max joints: %d\n' %(max_joints))
    n = text_file.write('total frames: %d\n' %(frames_counter))
    n = text_file.write('duration: %d\n' %(frames_counter / 12.5 / 60))
    n = text_file.write('~~~~ objects_counts - Total: %d ~~~~\n' %(files_counter) )
    for obj in objects_counter:
        text_file.write('%s: %d\n' %(obj, objects_counter[obj]))
    text_file.close()

    error_file = open(pjoin(save_dir, 'positions_error_rate.txt'), "w")
    n = error_file.write('Position squared error per source clip:')
    for f in squared_positions_error.keys():
        error_file.write('%s: %f\n' %(f, squared_positions_error[f]))
    error_file.close()

    _attach_joint_name_embeddings_to_cond(cond, save_dir)
    np.save(pjoin(save_dir, "cond.npy"), cond)
    _save_structural_prior_bank(save_dir, structural_prior_bank)
    write_motion_metadata(save_dir, motion_metadata, files_counter)


def _resolve_preprocessing_workers(objects, object_workers=8):
    object_count = max(1, len(objects))
    return min(object_count, max(1, int(object_workers)))


def _prepare_object_outputs_worker(object_type, max_files, raw_data_dir=None):
    return _prepare_object_outputs(
        object_type,
        max_joints=23,
        max_files=max_files,
        raw_data_dir=raw_data_dir,
    )


""" creates processed tensors for all the files of a given object. Returens statistics and the object condition,
which includes tpos, relation/distances matrices, offsets, parents, joints names, kinematic chains, mean and std"""    
def process_object(object_type, files_counter, frames_counter, max_joints, squared_positions_error, save_dir = DEFAULT_DATASET_DIR, face_joints=None, fbxs_dir=None, t_pos_path=None, max_files=None, raw_data_dir=None, structural_prior_bank=None):
    object_payload = _prepare_object_outputs(
        object_type,
        max_joints,
        face_joints=face_joints,
        fbxs_dir=fbxs_dir,
        t_pos_path=t_pos_path,
        max_files=max_files,
        raw_data_dir=raw_data_dir,
    )
    if object_payload is None:
        return files_counter, frames_counter, max_joints, None, {}

    if structural_prior_bank is None:
        structural_prior_bank = _build_structural_prior_bank([object_payload])
    _apply_structural_stats_to_payloads([object_payload], structural_prior_bank)

    squared_positions_error.update(object_payload['errors'])
    max_joints = max(max_joints, object_payload['max_joints'])
    files_counter, object_frames_counter, object_motion_metadata = _write_object_outputs(
        save_dir,
        object_payload,
        files_counter,
    )
    frames_counter += object_frames_counter

    return files_counter, frames_counter, max_joints, object_payload['object_cond'], object_motion_metadata


""" create dataset """
def create_data_samples(objects=None, max_files_per_object=None, dataset_dir=None, raw_data_dir=None, object_workers=8):
    ## prepare
    target_dataset_dir = dataset_dir or DEFAULT_DATASET_DIR
    os.makedirs(pjoin(target_dataset_dir, MOTION_DIR), exist_ok=True)
    os.makedirs(pjoin(target_dataset_dir, BVHS_DIR), exist_ok=True)
    
    ## process
    if objects is None:
        resolved_raw_data_dir = get_raw_data_dir(raw_data_dir)
        objects = sorted(
            obj for obj in os.listdir(resolved_raw_data_dir)
            if os.path.isdir(pjoin(resolved_raw_data_dir, obj))
        )

    obj_workers = _resolve_preprocessing_workers(
        objects,
        object_workers=object_workers,
    )
    print(f'Preprocessing {len(objects)} characters with {obj_workers} object workers')

    payloads = [None] * len(objects)
    if obj_workers <= 1:
        for idx, object_type in enumerate(objects):
            payloads[idx] = _prepare_object_outputs(
                object_type,
                max_joints=23,
                max_files=max_files_per_object,
                raw_data_dir=raw_data_dir,
            )
    else:
        with ProcessPoolExecutor(max_workers=obj_workers) as executor:
            future_to_idx = {
                executor.submit(
                    _prepare_object_outputs_worker,
                    object_type,
                    max_files_per_object,
                    raw_data_dir,
                ): idx
                for idx, object_type in enumerate(objects)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                payloads[idx] = future.result()  # propagates exception to abort all processing

    files_counter = 0
    frames_counter = 0
    max_joints = 23
    objects_counter = dict()
    squared_positions_error = dict()
    cond = dict()
    motion_metadata = {}

    all_motion_errors = []
    for idx, object_type in enumerate(objects):
        payload = payloads[idx]
        if payload is None:
            continue
        squared_positions_error.update(payload['errors'])
        max_joints = max(max_joints, payload['max_joints'])
        all_motion_errors.extend(payload.get('motion_errors', []))
        cur_counter = files_counter
        files_counter, object_frames, object_motion_metadata = _write_object_outputs(
            target_dataset_dir,
            payload,
            files_counter,
        )
        frames_counter += object_frames
        cond[object_type] = payload['object_cond']
        objects_counter[object_type] = files_counter - cur_counter
        motion_metadata.update(object_motion_metadata)

    if all_motion_errors:
        print(f"\n{'=' * 70}")
        print(f"\x1b[31mMOTION PROCESSING ERRORS ({len(all_motion_errors)} total)\x1b[0m")
        print('=' * 70)
        for err in all_motion_errors:
            print(err)
        print(f"{'=' * 70}\n")
        sys.exit(1)

    # Write bare cond.npy with structural info only (no prior bank, norm stats,
    # T5 embeddings, or metadata).  The parent process (preprocess_and_validate.py)
    # will rebuild the prior bank from all on-disk motions and regenerate all
    # side artifacts, so this subprocess only needs to produce the raw data.
    np.save(pjoin(target_dataset_dir, "cond.npy"), cond)


########################### Tests ##############################
def process_single_object_type(object_type, save_dir):
    ## prepare
    os.makedirs(pjoin(save_dir, MOTION_DIR), exist_ok=True)
    os.makedirs(pjoin(save_dir, BVHS_DIR), exist_ok=True)
    
    payload = _prepare_object_outputs(
        object_type,
        max_joints=23,
    )
    if payload is None:
        return

    structural_prior_bank = _build_structural_prior_bank([payload])
    _apply_structural_stats_to_payloads([payload], structural_prior_bank)

    files_counter = 0
    frames_counter = 0
    max_joints = payload['max_joints']
    objects_counter = dict()
    squared_positions_error = dict(payload['errors'])
    cond = dict()
    motion_metadata = {}
    cur_counter = files_counter
    files_counter, object_frames_counter, object_motion_metadata = _write_object_outputs(
        save_dir,
        payload,
        files_counter,
    )
    frames_counter += object_frames_counter
    cond[object_type] = payload['object_cond']
    objects_counter[object_type] = files_counter - cur_counter 
    motion_metadata.update(object_motion_metadata)

    _write_dataset_artifacts(
        save_dir,
        cond,
        motion_metadata,
        objects_counter,
        max_joints,
        files_counter,
        frames_counter,
        squared_positions_error,
        structural_prior_bank=structural_prior_bank,
    )


def process_skeleton(object_name, face_joints, save_dir, tpose_path, anim_dir=None,
                     motions_from_npys=None, target_cond_partial=None, structural_prior_bank_path=None):
    ## prepare
    os.makedirs(pjoin(save_dir, MOTION_DIR), exist_ok=True)
    os.makedirs(pjoin(save_dir, BVHS_DIR), exist_ok=True)
    structural_prior_bank = _load_structural_prior_bank(structural_prior_bank_path)

    if motions_from_npys is not None:
        # Retarget branch: motions already written to save_dir/motions/ by auto_retarget_pipeline.
        # Structural stats still come from the shared prior bank.
        assert target_cond_partial is not None, "target_cond_partial required with motions_from_npys"
        all_motions = [np.load(p).astype(np.float32) for p in motions_from_npys]
        if not all_motions:
            print(f"[process_skeleton] no retargeted motions available; cond.npy not written")
            return
        object_cond = dict(target_cond_partial)
        _apply_structural_stats_to_object_cond(object_cond, structural_prior_bank)
        motion_metadata = {}
        parents = np.asarray(object_cond['parents'], dtype=np.int64)
        offsets = np.asarray(object_cond['offsets'], dtype=np.float64)
        for motion_path, motion in zip(motions_from_npys, all_motions):
            motion_name = os.path.basename(motion_path)
            motion_labels = infer_motion_labels_from_motion_name(
                motion_name,
                object_type=object_name,
            )
            motion_labels['translation_root_index'] = int(
                infer_translation_root_index_from_features(
                    motion,
                    parents,
                    offsets,
                )
            )
            motion_metadata[motion_name] = motion_labels
        n_joints = len(object_cond['parents'])
        cond = {object_name: object_cond}
        _write_dataset_artifacts(
            save_dir,
            cond,
            motion_metadata,
            {object_name: len(all_motions)},             # objects_counter
            n_joints,                                     # max_joints
            len(all_motions),                             # files_counter
            sum(m.shape[0] for m in all_motions),         # frames_counter
            {},                                           # squared_positions_error
            structural_prior_bank=structural_prior_bank,
        )
        return

    ## process
    files_counter = 0
    frames_counter = 0
    max_joints = 23
    objects_counter = dict()
    squared_positions_error = dict()
    cond = dict()
    motion_metadata = {}

    if anim_dir is None:
        # T-pose only: generate cond.npy without motion file processing
        object_cond, max_joints = _build_tpose_only_cond(
            object_name,
            tpose_path,
            face_joints,
            structural_prior_bank=structural_prior_bank,
        )
        cond[object_name] = object_cond
        _write_dataset_artifacts(
            save_dir,
            cond,
            motion_metadata,
            objects_counter,
            max_joints,
            files_counter,
            frames_counter,
            squared_positions_error,
            structural_prior_bank=structural_prior_bank,
        )
        return

    cur_counter = files_counter
    files_counter, frames_counter, max_joints, object_cond, object_motion_metadata = process_object(
        object_name,
        files_counter,
        frames_counter,
        max_joints,
        squared_positions_error,
        save_dir=save_dir,
        fbxs_dir=anim_dir,
        face_joints=face_joints,
        t_pos_path=tpose_path,
        structural_prior_bank=structural_prior_bank,
    )
    if object_cond is None:
        print(f"No valid animation data found for '{object_name}', aborting.")
        return
    cond[object_name] = object_cond
    objects_counter[object_name] = files_counter - cur_counter
    motion_metadata.update(object_motion_metadata)

    _write_dataset_artifacts(
        save_dir,
        cond,
        motion_metadata,
        objects_counter,
        max_joints,
        files_counter,
        frames_counter,
        squared_positions_error,
        structural_prior_bank=structural_prior_bank,
    )

