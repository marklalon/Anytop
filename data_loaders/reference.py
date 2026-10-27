"""Stable reference conditioning tensors, shared by loading and inference."""
import numpy as np
import torch

from data_loaders.truebones.truebones_utils.canonical_features import (
    _length_scale_from_cond, physical_hml_to_canonical, rest_pos_from_cond, root_relative_positions,
)
from data_loaders.truebones.truebones_utils.joint_struct_features import build_joint_struct_features


def prepare_reference_sample(motion, cond, metadata, opt, *, augment=True, shared=None):
    """One encoder input from a clip used whole: its skeleton augmentation
    (``augment_reference_skeleton``), then ``finish_reference_sample``."""
    if augment:
        motion, cond, shared, _changed = augment_reference_skeleton(motion, cond, metadata, opt, shared=shared)
    return finish_reference_sample(motion, cond, opt, augment=augment, shared=shared)


def augment_reference_skeleton(motion, cond, metadata, opt, *, shared=None):
    """Return ``(motion, cond, shared, changed)``: the encoder input's skeleton
    augmentation, drawn on the whole stored clip before time sampling, as the
    target's is, so both are grounded on the same clip.

    The draws are independent of the target's own skeleton augmentation;
    ``motion`` / ``cond`` are the stored rig's clip. ``shared`` holds channels
    already derived from this same rig; they are kept while no augmentation
    changes the skeleton, and the structural channel also across the rest
    variant, bone-length and rest re-pose stages. The augmentations return new arrays,
    so the caller's motion and cond are never written.
    """
    import random
    from data_loaders.truebones.truebones_utils.topology_aug import augment_topology
    from data_loaders.truebones.truebones_utils.bone_length_aug import (
        DEFAULT_BONE_LENGTH_AUG, DEFAULT_BONE_LENGTH_AUG_MIN, augment_bone_lengths)
    from data_loaders.truebones.truebones_utils.rest_repose_aug import (
        DEFAULT_REST_REPOSE_MAX_DEG, augment_rest_repose)
    from data_loaders.truebones.truebones_utils.rest_variants import (
        augment_rest_variant, stored_rest_struct_cond)

    metadata = dict(metadata)
    variant_prob = getattr(opt, 'rest_variant_prob', 0.0)
    topology_prob = getattr(opt, 'topology_aug_prob', 0.0)
    bone_prob = getattr(opt, 'bone_length_aug_prob', 0.0)
    repose_prob = getattr(opt, 'rest_repose_prob', 0.0)
    shared = dict(shared or {})
    changed = False
    stored_cond = cond
    variant_applied = False
    # Drawn independently of the target's, so a self-reference can see the same
    # clip under another of the species' rests.
    if variant_prob and random.random() < variant_prob:
        motion, cond, info = augment_rest_variant(motion, cond)
        variant_applied = info['applied']
    if topology_prob and random.random() < topology_prob:
        motion, cond, info = augment_topology(motion, cond, metadata['translation_root_index'])
        if info['applied']:
            metadata['translation_root_index'] = info['translation_root_index']
            struct_cond = (stored_rest_struct_cond(cond, stored_cond, info['new_index'])
                           if variant_applied else cond)
            shared = dict(joint_struct=build_joint_struct_features(struct_cond))
            changed = True
    # The structural channel names the rig's joints, not its rest geometry, so
    # the rest variant, bone-length and re-pose stages keep it (see rest_repose_aug).
    geometry_changed = variant_applied
    if bone_prob and random.random() < bone_prob:
        motion, cond, info = augment_bone_lengths(
            motion, cond, metadata, getattr(opt, 'bone_length_aug_max', DEFAULT_BONE_LENGTH_AUG),
            getattr(opt, 'bone_length_aug_min', DEFAULT_BONE_LENGTH_AUG_MIN))
        geometry_changed = geometry_changed or info['applied']
    if repose_prob and random.random() < repose_prob:
        motion, cond, info = augment_rest_repose(
            motion, cond, metadata, getattr(opt, 'rest_repose_max_deg', DEFAULT_REST_REPOSE_MAX_DEG))
        geometry_changed = geometry_changed or info['applied']
    if geometry_changed:
        struct = shared.get('joint_struct')
        shared = dict(joint_struct=build_joint_struct_features(stored_cond) if struct is None else struct)
    return motion, cond, shared, changed or geometry_changed


def finish_reference_sample(motion, cond, opt, *, augment=True, shared=None):
    """The encoder input of a time-sampled ``motion`` on its (augmented) rig
    ``cond``, with the joint-name drop when ``augment``; ``shared`` as
    ``reference_sample`` takes it."""
    import random

    names = np.asarray(cond['joints_names_embs'], dtype=np.float32)
    prob = getattr(opt, 'joint_name_drop_prob', 0.0) if augment else 0.0
    if prob:
        drop = np.array([random.random() < prob for _ in names], dtype=bool)
        if drop.any():
            names = names.copy()
            names[drop] = 0
    return reference_sample(motion, cond, names, **(shared or {}))


# A bone shorter than this fraction of the rig's longest rest bone is a
# coincident helper joint; its direction is noise, so it reads as zero.
BONE_DIRECTION_MIN_REST_FRACTION = 1e-3

# Per-joint encoder pose: unit bone direction (3) and height over ``L`` (1).
REFERENCE_POSE_CHANNELS = 4
# Rest geometry per joint (``rest_geometry``): the encoder pose channels of the
# rest, then the parent->joint bone length over ``L``, so each consumer's
# columns are one contiguous slice.
REST_GEOMETRY_DIM = REFERENCE_POSE_CHANNELS + 1
# Reference encoder: the rest in the space of its per-frame pose.
REST_POSE_COLUMNS = slice(0, REFERENCE_POSE_CHANNELS)
# Denoiser: the encoder's rest channels plus bone length, which a near
# horizontal bone's direction and height cannot recover.
REST_GEOM_COLUMNS = slice(0, REST_GEOMETRY_DIM)
# Target rest-geometry channel: the denoiser columns plus a present flag, since an
# all-zero geometry (a root on the ground) is a real value.
REST_GEOM_DIM = REST_GEOM_COLUMNS.stop - REST_GEOM_COLUMNS.start + 1


def bone_directions(positions, parents, rest_positions):
    """``[T, J, 3]`` unit parent->child directions, one row per frame.

    Physical positions sit in the skeleton-canonicalized rest-facing frame
    (every rest faces +Z; per-frame root yaw is not removed) with absolute
    height, so a direction means the same world direction on every skeleton
    ("points up", "points back"), unlike the local rotation. The rest pose only decides which bones are coincident
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
    (ground contact and posture height, comparable across body sizes), read
    off the all-RIC view of the working-layout ``motion``.
    """
    positions = root_relative_positions(np.asarray(motion, np.float32)[..., :3])
    height = positions[..., 1:2] / float(_length_scale_from_cond(cond))
    pose = np.concatenate([bone_directions(positions, cond['parents'], rest_pos_from_cond(cond)),
                           height], axis=-1)
    return np.nan_to_num(pose).astype(np.float32)


def rest_geometry(cond):
    """``[J, REST_GEOMETRY_DIM]`` rest geometry, built once for both consumers.

    The reference encoder reads ``REST_POSE_COLUMNS``: the pose channels
    evaluated on the rest, so a frame's pose and the rest sit in one space and
    their difference is linear in the inputs. The denoiser reads
    ``REST_GEOM_COLUMNS`` (``y['rest_geom']``, loader and generation alike):
    the same channels plus each joint's rest bone length over ``L`` (zero at
    roots).
    """
    rest = np.asarray(rest_pos_from_cond(cond), np.float32)
    parents = np.asarray(cond['parents']).reshape(-1)
    pose = reference_pose(rest[None], cond)[0]
    length = np.zeros((len(rest), 1), np.float32)
    child = np.flatnonzero(parents >= 0)
    length[child, 0] = np.linalg.norm(rest[child] - rest[parents[child]], axis=-1)
    length = length / float(_length_scale_from_cond(cond))
    geometry = np.concatenate([pose, length], axis=-1)
    return np.nan_to_num(geometry).astype(np.float32)


def reference_sample(motion, cond, names, *, feats=None, joint_struct=None, geometry=None):
    """One encoder input from physical features on their own skeleton.

    The single builder for both the loader and generation, so the two never
    hand the encoder differently derived channels. ``feats`` / ``joint_struct``
    / ``geometry`` (``rest_geometry``) may be passed in when already derived
    from this same motion and cond.
    ``feats`` is the canonical window; the encoder reads only its root's
    world XZ.
    """
    if feats is None:
        feats = np.nan_to_num(physical_hml_to_canonical(motion, cond)).astype(np.float32)
    return {
        'feats': feats,
        'pose': reference_pose(motion, cond),
        'rest_pose': (rest_geometry(cond) if geometry is None else geometry)[:, REST_POSE_COLUMNS],
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
        # Frames withheld from the encoder (MotionEncoder); training and
        # generation fill it in once they know what is re-drawn.
        'ref_frame_mask': torch.zeros(batch, frames, dtype=torch.bool, device=motion.device),
        # The reference is the target's own window on the target's rig, so
        # its joint j is the target's joint j (sample['same_rig']).
        'ref_same_rig': torch.zeros(batch, dtype=torch.bool, device=motion.device),
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
        result['ref_same_rig'][i] = bool(sample.get('same_rig', False))
        for key in ('joints_names_embs', 'joint_struct'):
            result['ref_' + key][i, :count] = torch.as_tensor(sample[key], device=motion.device)
    return result
