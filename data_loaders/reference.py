"""Stable reference conditioning tensors, shared by loading and inference."""
import numpy as np
import torch

from data_loaders.truebones.truebones_utils.canonical_features import (
    _length_scale_from_cond, physical_hml_to_canonical, rest_pos_from_cond,
)
from data_loaders.truebones.truebones_utils.joint_struct_features import build_joint_struct_features


def prepare_reference_sample(motion, cond, metadata, opt, *, augment=True, shared=None):
    """Augment the encoder input's skeleton after time sampling.

    The draws are independent of the target's own skeleton augmentation;
    ``motion`` / ``cond`` are the stored rig's window. ``shared`` holds
    channels already derived from this same motion and cond; they are reused
    while no augmentation changes the skeleton. The augmentations return new
    arrays, so the caller's motion and cond are never written.
    """
    import random
    from data_loaders.truebones.truebones_utils.leaf_drop import sample_leaf_drop, drop_joints_from_cond
    from data_loaders.truebones.truebones_utils.bone_length_aug import (
        DEFAULT_BONE_LENGTH_AUG, DEFAULT_BONE_LENGTH_AUG_MIN, augment_bone_lengths)

    metadata = dict(metadata)
    leaf_prob = getattr(opt, 'leaf_drop_prob', 0.0) if augment else 0.0
    bone_prob = getattr(opt, 'bone_length_aug_prob', 0.0) if augment else 0.0
    changed = False
    if leaf_prob and random.random() < leaf_prob:
        dropped = sample_leaf_drop(cond, motion, metadata['translation_root_index'])
        if dropped:
            keep, cond = drop_joints_from_cond(cond, dropped)
            motion = motion[:, keep]
            metadata['translation_root_index'] = int(np.searchsorted(keep, metadata['translation_root_index']))
            changed = True
    if bone_prob and random.random() < bone_prob:
        motion, cond, info = augment_bone_lengths(
            motion, cond, metadata, getattr(opt, 'bone_length_aug_max', DEFAULT_BONE_LENGTH_AUG),
            getattr(opt, 'bone_length_aug_min', DEFAULT_BONE_LENGTH_AUG_MIN))
        changed = changed or info['applied']
    names = np.asarray(cond['joints_names_embs'], dtype=np.float32)
    prob = getattr(opt, 'joint_name_drop_prob', 0.0) if augment else 0.0
    if prob:
        drop = np.array([random.random() < prob for _ in names], dtype=bool)
        if drop.any():
            names = names.copy()
            names[drop] = 0
    return reference_sample(motion, cond, names, **({} if changed else (shared or {})))


# A bone shorter than this fraction of the rig's longest rest bone is a
# coincident helper joint; its direction is noise, so it reads as zero.
BONE_DIRECTION_MIN_REST_FRACTION = 1e-3

# Per-joint encoder pose: unit bone direction (3) and height over ``L`` (1).
REFERENCE_POSE_CHANNELS = 4
# Target rest-geometry channel: the rest's pose channels plus a present flag, since an
# all-zero geometry (root, coincident helper, grounded joint) is a real value.
REST_GEOM_DIM = REFERENCE_POSE_CHANNELS + 1


def bone_directions(positions, parents, rest_positions):
    """``[T, J, 3]`` unit parent->child directions, one row per frame.

    Physical positions sit in the skeleton-canonicalized rest-facing frame
    (every rest faces +Z; per-frame root yaw is not removed) with absolute
    height, so a direction means the same world direction on every skeleton
    ("points up", "points back"), unlike the rest-centered position residual or
    the local rotation. The rest pose only decides which bones are coincident
    helpers; it is never an output row. Roots and coincident joints are zero.
    """
    positions = np.asarray(positions, np.float32)
    rest_positions = np.asarray(rest_positions, np.float32)
    parents = np.asarray(parents)
    out = np.zeros_like(positions)
    child = np.flatnonzero(parents >= 0)
    if not len(child):
        return out
    rest_norm = np.linalg.norm(rest_positions[child] - rest_positions[parents[child]], axis=-1)
    keep = rest_norm > BONE_DIRECTION_MIN_REST_FRACTION * max(float(rest_norm.max()), 1e-12)
    vec = positions[:, child] - positions[:, parents[child]]
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    # Non-finite positions are tolerated by the ``feats`` path (nan_to_num), so
    # they must not leave a NaN here: inf/inf in the divide is NaN.
    with np.errstate(invalid='ignore', over='ignore'):
        unit = np.divide(vec, norm, out=np.zeros_like(vec), where=norm > 1e-12)
        out[:, child] = unit * keep[None, :, None]
    return np.nan_to_num(out)


def reference_pose(motion, cond):
    """``[T, J, REFERENCE_POSE_CHANNELS]`` rest-free encoder pose.

    Bone directions plus each joint's absolute height over the rig's ``L``
    (ground contact and posture height, comparable across body sizes).
    """
    positions = np.asarray(motion, np.float32)[..., :3]
    height = positions[..., 1:2] / float(_length_scale_from_cond(cond))
    pose = np.concatenate([bone_directions(positions, cond['parents'], rest_pos_from_cond(cond)),
                           height], axis=-1)
    return np.nan_to_num(pose).astype(np.float32)


def rest_geom_features(cond):
    """``[J, REFERENCE_POSE_CHANNELS]`` rest geometry: the pose channels of the rest.

    The same bone-direction / height channels evaluated on the rest positions,
    so a frame's pose and the rest sit in one space and their difference is
    linear in the inputs. The single builder for both rest inputs: the reference
    encoder's static per-joint context and the denoiser's per-joint rest-geometry
    channel (``y['rest_geom']``, loader and generation alike).
    """
    return reference_pose(rest_pos_from_cond(cond)[None], cond)[0]


def reference_sample(motion, cond, names, *, feats=None, joint_struct=None):
    """One encoder input from physical features on their own skeleton.

    The single builder for both the loader and generation, so the two never
    hand the encoder differently derived channels. ``feats`` / ``joint_struct``
    may be passed in when already derived from this same motion and cond.
    ``feats`` is the canonical window; the encoder reads only its root XZ
    velocity, the one canonical channel that is not rest-centered.
    """
    if feats is None:
        feats = np.nan_to_num(physical_hml_to_canonical(motion, cond)).astype(np.float32)
    return {
        'feats': feats,
        'pose': reference_pose(motion, cond),
        'rest_pose': rest_geom_features(cond),
        'joints_names_embs': names,
        'joint_struct': build_joint_struct_features(cond) if joint_struct is None else joint_struct,
    }


def collate_reference(samples, motion, names, joint_struct):
    """Use the batch joint bucket for loader copies; it holds target and reference rigs.

    Inference passes a source-sized canvas, independent of the target rig.
    Every batch has the same key set, including ref-null batches.
    """
    batch, joints, feats, frames = motion.shape
    result = {
        'ref_feats': torch.zeros_like(motion),
        'ref_pose': motion.new_zeros(batch, joints, REFERENCE_POSE_CHANNELS, frames),
        'ref_rest_pose': motion.new_zeros(batch, joints, REFERENCE_POSE_CHANNELS),
        'ref_mask': torch.zeros(batch, joints, dtype=torch.bool, device=motion.device),
        'ref_active': torch.zeros(batch, dtype=torch.bool, device=motion.device),
        'ref_joints_names_embs': torch.zeros_like(names),
        'ref_joint_struct': torch.zeros_like(joint_struct),
    }
    for i, sample in enumerate(samples):
        if sample is None:
            continue
        data = torch.as_tensor(sample['feats'], dtype=motion.dtype, device=motion.device)
        if data.ndim != 3 or data.shape[0] != frames or data.shape[2] != feats:
            raise ValueError('Reference features must match target frames and feature width')
        count = data.shape[1]
        if not 0 < count <= joints:
            raise ValueError('Reference joints exceed their source canvas / loader bucket')
        result['ref_feats'][i, :count] = data.permute(1, 2, 0)
        pose = torch.as_tensor(sample['pose'], dtype=motion.dtype, device=motion.device)
        if pose.shape != (frames, count, REFERENCE_POSE_CHANNELS):
            raise ValueError(
                f'Reference pose must have shape [frames, source joints, {REFERENCE_POSE_CHANNELS}], '
                f'got {tuple(pose.shape)} for {frames} frames and {count} joints'
            )
        result['ref_pose'][i, :count] = pose.permute(1, 2, 0)
        result['ref_rest_pose'][i, :count] = torch.as_tensor(
            sample['rest_pose'], dtype=motion.dtype, device=motion.device)
        result['ref_mask'][i, :count] = True
        result['ref_active'][i] = True
        for key in ('joints_names_embs', 'joint_struct'):
            result['ref_' + key][i, :count] = torch.as_tensor(sample[key], device=motion.device)
    return result
