"""Stable reference conditioning tensors, shared by loading and inference."""
import numpy as np
import torch

from data_loaders.truebones.truebones_utils.canonical_features import (
    build_canonical_rest_feature, physical_hml_to_canonical, rest_pos_from_cond,
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
    from data_loaders.truebones.truebones_utils.bone_length_aug import augment_bone_lengths

    metadata = dict(metadata)
    leaf_prob = getattr(opt, 'ref_leaf_drop_prob', 0.0) if augment else 0.0
    bone_prob = getattr(opt, 'ref_bone_length_aug_prob', 0.0) if augment else 0.0
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
            motion, cond, metadata, getattr(opt, 'ref_bone_length_aug', 0.1))
        changed = changed or info['applied']
    names = np.asarray(cond['joints_names_embs'], dtype=np.float32)
    prob = getattr(opt, 'ref_joint_name_drop_prob', 0.0) if augment else 0.0
    if prob:
        drop = np.array([random.random() < prob for _ in names], dtype=bool)
        if drop.any():
            names = names.copy()
            names[drop] = 0
    return reference_sample(motion, cond, names, **({} if changed else (shared or {})))


# A bone shorter than this fraction of the rig's longest rest bone is a
# coincident helper joint; its direction is noise, so it reads as zero.
BONE_DIRECTION_MIN_REST_FRACTION = 1e-3


def bone_directions(positions, parents, rest_positions):
    """``[T + 1, J, 3]`` unit parent->child directions; row 0 is the rest pose.

    Physical positions sit in the skeleton-canonicalized rest-facing frame
    (every rest faces +Z; per-frame root yaw is not removed) with absolute
    height, so a direction means the same world direction on every skeleton
    ("points up", "points back"), unlike the rest-centered position residual or
    the local rotation. Roots and coincident joints are zero.
    """
    positions = np.concatenate([np.asarray(rest_positions, np.float32)[None],
                                np.asarray(positions, np.float32)], axis=0)
    parents = np.asarray(parents)
    out = np.zeros_like(positions)
    child = np.flatnonzero(parents >= 0)
    if not len(child):
        return out
    vec = positions[:, child] - positions[:, parents[child]]
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    rest_norm = norm[0, :, 0]
    keep = rest_norm > BONE_DIRECTION_MIN_REST_FRACTION * max(float(rest_norm.max()), 1e-12)
    # Non-finite positions are tolerated by the ``feats`` path (nan_to_num), so
    # they must not leave a NaN here: inf/inf in the divide is NaN.
    with np.errstate(invalid='ignore', over='ignore'):
        unit = np.divide(vec, norm, out=np.zeros_like(vec), where=norm > 1e-12)
        out[:, child] = unit * keep[None, :, None]
    return np.nan_to_num(out)


def reference_sample(motion, cond, names, *, feats=None, rest_pose=None, joint_struct=None):
    """One encoder input from physical features on their own skeleton.

    The single builder for both the loader and generation, so the two never
    hand the encoder differently derived channels. ``feats`` / ``rest_pose`` /
    ``joint_struct`` may be passed in when already derived from this same
    motion and cond.
    """
    if feats is None:
        feats = np.nan_to_num(physical_hml_to_canonical(motion, cond)).astype(np.float32)
    return {
        'feats': feats,
        'bone_dir': bone_directions(motion[..., :3], cond['parents'], rest_pos_from_cond(cond)),
        'rest_pose': build_canonical_rest_feature(cond) if rest_pose is None else rest_pose,
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
        'ref_bone_dir': motion.new_zeros(batch, joints, 3, frames + 1),
        'ref_mask': torch.zeros(batch, joints, dtype=torch.bool, device=motion.device),
        'ref_active': torch.zeros(batch, dtype=torch.bool, device=motion.device),
        'ref_rest_pose': motion.new_zeros(batch, joints, feats),
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
        bone_dir = torch.as_tensor(sample['bone_dir'], dtype=motion.dtype, device=motion.device)
        if bone_dir.shape != (frames + 1, count, 3):
            raise ValueError(
                'Reference bone directions must have shape [frames + 1, source joints, 3], '
                f'got {tuple(bone_dir.shape)} for {frames} frames and {count} joints'
            )
        result['ref_bone_dir'][i, :count] = bone_dir.permute(1, 2, 0)
        result['ref_mask'][i, :count] = True
        result['ref_active'][i] = True
        for key in ('rest_pose', 'joints_names_embs', 'joint_struct'):
            result['ref_' + key][i, :count] = torch.as_tensor(sample[key], device=motion.device)
    return result
