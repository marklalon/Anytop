"""Stable reference conditioning tensors, shared by loading and inference."""
import numpy as np
import torch

from data_loaders.truebones.truebones_utils.canonical_features import (
    build_canonical_rest_feature, physical_hml_to_canonical,
)
from data_loaders.truebones.truebones_utils.joint_struct_features import build_joint_struct_features


def prepare_reference_sample(motion, cond, metadata, opt, *, augment=True):
    """Copy and augment physical features only on the encoder side, after time sampling."""
    import random
    from data_loaders.truebones.truebones_utils.leaf_drop import sample_leaf_drop, drop_joints_from_cond
    from data_loaders.truebones.truebones_utils.bone_length_aug import augment_bone_lengths

    motion = np.array(motion, dtype=np.float32, copy=True)
    cond = dict(cond)
    metadata = dict(metadata)
    leaf_prob = getattr(opt, 'ref_leaf_drop_prob', 0.0) if augment else 0.0
    bone_prob = getattr(opt, 'ref_bone_length_aug_prob', 0.0) if augment else 0.0
    if leaf_prob and random.random() < leaf_prob:
        dropped = sample_leaf_drop(cond, motion, metadata['translation_root_index'])
        if dropped:
            keep, cond = drop_joints_from_cond(cond, dropped)
            motion = motion[:, keep]
            metadata['translation_root_index'] = int(np.searchsorted(keep, metadata['translation_root_index']))
    if bone_prob and random.random() < bone_prob:
        motion, cond, _ = augment_bone_lengths(
            motion, cond, metadata, getattr(opt, 'ref_bone_length_aug', 0.1))
    names = np.array(cond['joints_names_embs'], dtype=np.float32, copy=True)
    prob = getattr(opt, 'ref_joint_name_drop_prob', 0.0) if augment else 0.0
    if prob:
        names[np.array([random.random() < prob for _ in names])] = 0
    return {
        'feats': np.nan_to_num(physical_hml_to_canonical(motion, cond)).astype(np.float32),
        'rest_pose': build_canonical_rest_feature(cond),
        'joints_names_embs': names,
        'joint_struct': build_joint_struct_features(cond),
        'graph_dist': cond['joints_graph_dist'],
        'joints_relations': cond['joint_relations'],
    }


def collate_reference(samples, motion, names, joint_struct):
    """Use the target joint bucket for loader copies; source rigs only lose leaves.

    Inference passes a source-sized canvas, independent of the target rig.
    Every batch has the same key set, including ref-null batches.
    """
    batch, joints, feats, frames = motion.shape
    result = {
        'ref_feats': torch.zeros_like(motion),
        'ref_mask': torch.zeros(batch, joints, dtype=torch.bool, device=motion.device),
        'ref_active': torch.zeros(batch, dtype=torch.bool, device=motion.device),
        'ref_rest_pose': motion.new_zeros(batch, joints, feats),
        'ref_joints_names_embs': torch.zeros_like(names),
        'ref_joint_struct': torch.zeros_like(joint_struct),
        'ref_graph_dist': motion.new_zeros(batch, joints, joints),
        'ref_joints_relations': motion.new_zeros(batch, joints, joints),
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
        result['ref_mask'][i, :count] = True
        result['ref_active'][i] = True
        for key in ('rest_pose', 'joints_names_embs', 'joint_struct'):
            result['ref_' + key][i, :count] = torch.as_tensor(sample[key], device=motion.device)
        for key in ('graph_dist', 'joints_relations'):
            result['ref_' + key][i, :count, :count] = torch.as_tensor(sample[key], device=motion.device)
    return result
