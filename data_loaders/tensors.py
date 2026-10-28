import torch
import numpy as np

from data_loaders.truebones.truebones_utils.action_label_conditioning_contract import (
    SLOT_PAD_ID,
)
from data_loaders.truebones.truebones_utils.canonical_features import (
    CANONICAL_DPOS_STD_KEY,
    CANONICAL_FEATURE_SPACE,
    DPOS_LOSS_WEIGHT_KEY,
    PARENT_INDEX_KEY,
    REST_BONE_FRAME_KEY,
    REST_LENGTH_SCALE_KEY,
    _length_scale_from_rest,
    padded_rotation_context,
)
from data_loaders.joint_buckets import bucket_ceiling
from data_loaders.reference import REST_GEOM_DIM
from data_loaders.truebones.truebones_utils.joint_struct_features import JOINT_STRUCT_DIM
from data_loaders.truebones.truebones_utils.motion_labels import ACTION_LABEL_MAX_WORDS


def _build_action_slot_batch(action_slots_batch, action_labels_batch):
    """Pad the per-sample word ids / slots to ``[B, ACTION_LABEL_MAX_WORDS]``.

    Padded to the contract's word cap rather than to the batch maximum, so the
    conditioning tensors have one fixed shape for every batch: a shape that moved
    with the longest label in the batch would re-trigger compilation and rule out
    graph capture, for eight columns of saving.

    Rows with an empty label carry no words at all; their mask is all-False and
    the model routes them to its learned null embedding, which is what an empty
    label means. Padding columns get ``SLOT_PAD_ID`` so they match no slot even
    if a reader forgets the mask.
    """
    batch_size = len(action_slots_batch)
    width = ACTION_LABEL_MAX_WORDS
    word_ids = torch.zeros((batch_size, width), dtype=torch.int64)
    slot_ids = torch.full((batch_size, width), SLOT_PAD_ID, dtype=torch.int64)
    word_mask = torch.zeros((batch_size, width), dtype=torch.bool)
    valid = torch.zeros((batch_size,), dtype=torch.bool)
    any_slots = False
    for row_index, slots in enumerate(action_slots_batch):
        if slots is None:
            continue
        any_slots = True
        count = int(len(slots['word_ids']))
        if count > width:
            raise ValueError(
                f"action label has {count} words, over the contract cap {width}"
            )
        word_ids[row_index, :count] = torch.as_tensor(slots['word_ids'], dtype=torch.int64)
        slot_ids[row_index, :count] = torch.as_tensor(slots['slot_ids'], dtype=torch.int64)
        word_mask[row_index, :count] = torch.as_tensor(slots['word_mask'], dtype=torch.bool)
        valid[row_index] = bool(str(action_labels_batch[row_index] or ""))
    if not any_slots:
        return None, valid
    return {
        'action_word_ids': word_ids,
        'action_slot_ids': slot_ids,
        'action_word_mask': word_mask,
    }, valid

def n_joints_to_mask(n_joints, max_joints):
    mask = torch.arange(max_joints + 1, device=n_joints.device).expand(len(n_joints), max_joints + 1) < (n_joints.unsqueeze(1) + 1)
    mask = mask.unsqueeze(2).float() * mask.unsqueeze(1).float() 
    return mask

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas

def create_padded_relation(relation_np, max_joints, n_joints):
    # it counts on spatial attention masks!
    relation = torch.as_tensor(relation_np)
    padded_relation = torch.zeros((max_joints, max_joints)) 
    padded_relation[:n_joints, :n_joints ] = relation
    return padded_relation

def truebones_collate(batch):
    """Collate a list of motion items into a single (inp, cond) batch.

    Each item in *batch* is a dict produced by ``truebones_batch_collate``
    containing keys like ``'inp'``, ``'rest_geom'``, ``'lengths'``, etc.

    Returns
    -------
    inp : torch.Tensor
        Concatenated motion input tensor.
    cond : dict
        Conditioning dictionary with ``'y'`` containing all metadata
        (lengths, T-pose, object type, motion name, action group/label,
        loop info, …).
    """
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    restgeombatch = [b['rest_geom'] for b in notnone_batches]
    if 'n_joints' in notnone_batches[0]:
        jointsnumbatch = [b['n_joints'] for b in notnone_batches]
    else:
        jointsnumbatch = [22 for b in notnone_batches] #smpl n_joints 
        
    databatchTensor = collate_tensors(databatch)
    restgeombatchTensor = collate_tensors(restgeombatch)
    jointsnumbatchTensor = torch.as_tensor(jointsnumbatch)
    jointsmaskbatchTensor = n_joints_to_mask(jointsnumbatchTensor, databatchTensor.shape[1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    frame_count = databatchTensor.shape[-1]
    batch_size = len(databatch)
    cond = {'y': {'lengths': torch.full((batch_size,), frame_count, dtype=torch.long), 'rest_geom': restgeombatchTensor}}

    if 'object_type' in notnone_batches[0]:
        objecttypebatch = [b['object_type'] for b in notnone_batches]
        cond['y'].update({'object_type': objecttypebatch})

    if 'motion_name' in notnone_batches[0]:
        motionnamebatch = [b['motion_name'] for b in notnone_batches]
        cond['y'].update({'motion_name': motionnamebatch})

    if any('action_label' in batch_item or 'action_group' in batch_item
           for batch_item in notnone_batches):
        action_labels_batch = [batch_item.get('action_label') for batch_item in notnone_batches]
        action_groups_batch = [batch_item.get('action_group') for batch_item in notnone_batches]
        action_slots_batch = [batch_item.get('action_slots') for batch_item in notnone_batches]
        cond['y'].update({
            'action_group': action_groups_batch,
            'action_label': action_labels_batch,
        })
        slot_tensors, label_valid = _build_action_slot_batch(
            action_slots_batch, action_labels_batch
        )
        cond['y']['action_label_valid'] = label_valid
        if slot_tensors is not None:
            cond['y'].update(slot_tensors)

    if any('translation_root_index' in batch_item for batch_item in notnone_batches):
        cond['y'].update({
            'translation_root_index': [batch_item.get('translation_root_index') for batch_item in notnone_batches]
        })

    if any('rest_pos_ric_hml' in batch_item for batch_item in notnone_batches):
        cond['y'].update({
            'rest_pos_ric_hml': collate_tensors([
                batch_item['rest_pos_ric_hml'] for batch_item in notnone_batches
            ])
        })

    # Per-sample rest length scale, stacked in batch order like the canonical
    # stats below, and for the same reason: a partially-populated batch would
    # misalign sample<->L, so it is emitted only when every item carries it.
    length_scales = [batch_item.get(REST_LENGTH_SCALE_KEY) for batch_item in notnone_batches]
    if all(v is not None for v in length_scales):
        cond['y'][REST_LENGTH_SCALE_KEY] = torch.tensor(length_scales, dtype=torch.float32)

    # Structural joint descriptors. Emitted whenever the loader supplied them, so
    # the key set stays constant across every batch of a run (a key that appears
    # and disappears retriggers a torch.compile recompilation).
    if any('joint_struct' in batch_item for batch_item in notnone_batches):
        cond['y'].update({
            'joint_struct': collate_tensors([
                batch_item['joint_struct'] for batch_item in notnone_batches
            ])
        })

    # Emitted only when every item has them, like rest_length_scale: a decode
    # that found them on some rows only would misalign sample<->rig.
    if all(REST_BONE_FRAME_KEY in batch_item for batch_item in notnone_batches):
        cond['y'][REST_BONE_FRAME_KEY] = torch.stack(
            [batch_item[REST_BONE_FRAME_KEY] for batch_item in notnone_batches])
        cond['y'][PARENT_INDEX_KEY] = torch.stack(
            [batch_item[PARENT_INDEX_KEY] for batch_item in notnone_batches])

    if any('rest_pose_physical' in batch_item for batch_item in notnone_batches):
        cond['y'].update({
            'rest_pose_physical': collate_tensors([
                batch_item['rest_pose_physical'] for batch_item in notnone_batches
            ])
        })

    # Canonical standardization stats, stacked in batch order into [B, F] for the
    # y-only decoders (the training-time aux-loss decode, canonical_to_physical_hml
    # reading y). Every species carries the same global vectors; stacking keeps the
    # per-sample decode contract instead of assuming it. Only stack when every item
    # carries the stat; a partially-populated batch would misalign sample<->stat
    # (and the fail-fast loader guarantees all training items have it).
    for stat_key in ('canonical_feature_mean', 'canonical_feature_std'):
        stat_vals = [batch_item.get(stat_key) for batch_item in notnone_batches]
        if all(v is not None for v in stat_vals):
            cond['y'][stat_key] = torch.stack(
                [torch.as_tensor(v, dtype=torch.float32).reshape(-1) for v in stat_vals],
                dim=0,
            )

    # Per-sample scalars the loss reads ([B]), emitted only when every item has
    # them for the same reason as the stats above.
    for scalar_key in (CANONICAL_DPOS_STD_KEY, DPOS_LOSS_WEIGHT_KEY):
        scalar_vals = [batch_item.get(scalar_key) for batch_item in notnone_batches]
        if all(v is not None for v in scalar_vals):
            cond['y'][scalar_key] = torch.tensor([float(v) for v in scalar_vals], dtype=torch.float32)

    if any('feature_space' in batch_item for batch_item in notnone_batches):
        cond['y'].update({
            'feature_space': [batch_item.get('feature_space') for batch_item in notnone_batches]
        })

    if any('is_loop' in batch_item for batch_item in notnone_batches):
        cond['y'].update({
            'is_loop': torch.as_tensor(
                [bool(batch_item.get('is_loop', False)) for batch_item in notnone_batches],
                dtype=torch.bool,
            )
        })

    if any('loop_data_aug_applied' in batch_item for batch_item in notnone_batches):
        cond['y'].update({
            'loop_data_aug_applied': torch.as_tensor(
                [bool(batch_item.get('loop_data_aug_applied', False)) for batch_item in notnone_batches],
                dtype=torch.bool,
            )
        })

    if any('resample_speed_cond' in batch_item for batch_item in notnone_batches):
        cond['y'].update({
            'resample_speed_cond': torch.as_tensor(
                [float(batch_item.get('resample_speed_cond', 1.0)) for batch_item in notnone_batches],
                dtype=torch.float32,
            )
        })

    if any('motion_speed_applied' in batch_item for batch_item in notnone_batches):
        cond['y'].update({
            'motion_speed_applied': torch.as_tensor(
                [float(batch_item.get('motion_speed_applied', 1.0)) for batch_item in notnone_batches],
                dtype=torch.float32,
            )
        })

    if any('bone_length_aug_applied' in batch_item for batch_item in notnone_batches):
        cond['y'].update({
            'bone_length_aug_applied': torch.as_tensor(
                [bool(batch_item.get('bone_length_aug_applied', False)) for batch_item in notnone_batches],
                dtype=torch.bool,
            )
        })

    if any('loop_phase_offset' in batch_item for batch_item in notnone_batches):
        cond['y'].update({
            'loop_phase_offset': torch.as_tensor(
                [int(batch_item.get('loop_phase_offset', 0)) for batch_item in notnone_batches],
                dtype=torch.long,
            )
        })

    if any('loop_tile_count' in batch_item for batch_item in notnone_batches):
        cond['y'].update({
            'loop_tile_count': torch.as_tensor(
                [int(batch_item.get('loop_tile_count', 1)) for batch_item in notnone_batches],
                dtype=torch.long,
            )
        })

    if any('loop_cycles_active' in batch_item for batch_item in notnone_batches):
        cond['y'].update({
            'loop_cycles_active': torch.as_tensor(
                [bool(batch_item.get('loop_cycles_active', True)) for batch_item in notnone_batches],
                dtype=torch.bool,
            )
        })

    if any('joint_mask_limb_roots' in batch_item for batch_item in notnone_batches):
        cond['y']['joint_mask_limb_roots'] = torch.stack([
            batch_item.get('joint_mask_limb_roots', torch.zeros(databatchTensor.shape[1], dtype=torch.bool)).bool()
            for batch_item in notnone_batches
        ])

    if any('limb_tip_mask' in batch_item for batch_item in notnone_batches):
        cond['y']['limb_tip_mask'] = torch.stack([
            batch_item.get('limb_tip_mask', torch.zeros(databatchTensor.shape[1], dtype=torch.bool)).bool()
            for batch_item in notnone_batches
        ])

    if 'parents' in notnone_batches[0]:
        parentsbatch = [b['parents'] for b in notnone_batches]
        cond['y'].update({'parents': parentsbatch})
          
    if 'joints_names_embs' in notnone_batches[0]:
        jointsnamesembsbatch = [b['joints_names_embs'] for b in notnone_batches]
        jointsnamesembsbatchTensor = collate_tensors(jointsnamesembsbatch)
        cond['y'].update({'joints_names_embs': jointsnamesembsbatchTensor})

    if 'species_code' in notnone_batches[0]:
        speciescodebatch = [b['species_code'] for b in notnone_batches]
        cond['y'].update({'species_code': torch.stack(speciescodebatch)})
        
    if 'joints_relations' in notnone_batches[0]:
        jointsrelationsbatch = [b['joints_relations'] for b in notnone_batches]

    if 'graph_dist' in notnone_batches[0]:
        graphdistbatch = [b['graph_dist'] for b in notnone_batches]

    cond['y'].update({'joints_padding_mask': jointsmaskbatchTensor})
    cond['y'].update({'n_joints': jointsnumbatchTensor})
    cond['y'].update({'joints_relations': torch.stack(jointsrelationsbatch)})
    cond['y'].update({'graph_dist': torch.stack(graphdistbatch)})

    from data_loaders.reference import collate_reference
    # The null fields are tensors too, so compiled batches never change keys.
    struct = cond['y'].get('joint_struct', motion.new_zeros(
        motion.shape[0], motion.shape[1], JOINT_STRUCT_DIM))
    cond['y'].update(collate_reference(
        [item.get('reference') for item in notnone_batches], motion,
        cond['y']['joints_names_embs'], struct,
    ))
    return motion, cond


def _sample_joint_width(b):
    """Joints a raw sample needs on the canvas: its rig, or a wider reference.

    Target and reference skeletons are augmented independently, so the
    reference can keep leaves the target dropped.
    """
    width = b[0].shape[1]
    for extra in b[10:]:
        if isinstance(extra, dict) and extra.get('reference') is not None:
            width = max(width, np.asarray(extra['reference']['feats']).shape[1])
    return width


def truebones_batch_collate(batch, joint_buckets=None):
    """Collate a raw batch from MotionDataset into the format for truebones_collate.

    ``joint_buckets`` (ascending ceilings ending at MAX_JOINTS) pads the batch to
    the smallest ceiling that holds its widest rig instead of the global
    ``max_joints`` every sample carries at ``[9]``; the loader pairs it with
    ``JointBucketBatchSampler`` so each batch is bucket-homogeneous and the run
    sees one shape per bucket. Without it (inference, tests, the eval loader)
    every batch is padded to MAX_JOINTS as before.

    Each element ``b`` in *batch* is a tuple returned by
    ``MotionDataset._prepare_sample`` with the following layout:

        [0]  motion          – np.ndarray, (frames, n_joints, n_feats)
        [1]  m_length        – int, motion length in frames
        [2]  parents         – np.ndarray, parent indices
        [3]  rest_geom – np.ndarray, (n_joints, REST_GEOM_DIM - 1) rest_geometry's REST_GEOM_COLUMNS
        [4]  offsets         – np.ndarray, bone offsets
        [5]  joints_graph_dist – np.ndarray, graph distance matrix
        [6]  joints_relations  – np.ndarray, joint relation matrix
        [7]  object_type     – str
        [8]  joints_names_embs – np.ndarray, joint name embeddings
        [9]  max_joints      – int
        [10] motion_metadata – dict or None (action group/label, loop info, etc.)
        [11] name            – str, motion name
        [12+] extras         – dicts (joint_mask_limb_roots, aug info, …)
    """
    max_joints = batch[0][9]
    if joint_buckets is not None:
        max_joints = bucket_ceiling(max(_sample_joint_width(b) for b in batch), joint_buckets)
    adapted_batch = []
    for b in batch:  
        max_len, n_joints, n_feats = b[0].shape
        # The last channel is the present flag: live joints 1, padding 0.
        raw_rest_geom = np.asarray(b[3], dtype=np.float32)
        if raw_rest_geom.shape != (n_joints, REST_GEOM_DIM - 1):
            raise ValueError(
                f"rest_geom must have shape [{n_joints}, {REST_GEOM_DIM - 1}], got {raw_rest_geom.shape}."
            )
        rest_geom = torch.zeros((max_joints, REST_GEOM_DIM))
        rest_geom[:n_joints, :-1] = torch.from_numpy(raw_rest_geom)
        rest_geom[:n_joints, -1] = 1.0
        motion = torch.zeros((max_len, max_joints, n_feats)) # (frames, max_joints, feature_len) 
        motion[:, :b[0].shape[1], :] = torch.from_numpy(np.asarray(b[0], dtype=np.float32))
        joints_names_embs = torch.zeros((max_joints, b[8].shape[1]))
        joints_names_embs[:n_joints] = torch.from_numpy(np.asarray(b[8], dtype=np.float32))
        n_joints = b[0].shape[1]
        padded_joints_relations =  create_padded_relation(b[6], max_joints, n_joints)
        padded_graph_dist =  create_padded_relation(b[5], max_joints, n_joints)
        object_type = b[7]
        motion_metadata = None
        motion_name = None
        extra_cond = None
        for extra in b[10:]:
            if isinstance(extra, dict):
                if any(key in extra for key in ('joint_mask_limb_roots', 'rest_pos_ric_hml', 'joint_struct')):
                    extra_cond = extra
                elif any(key in extra for key in ('action_group', 'action_label', 'action_slots', 'translation_root_index', 'is_loop', 'resample_speed_cond', 'motion_speed_applied', 'loop_data_aug_applied', 'loop_phase_offset', 'loop_tile_count', 'loop_cycles_active')):
                    motion_metadata = extra
            elif isinstance(extra, str):
                motion_name = extra

        item = {
            'inp': motion.permute(1, 2, 0).float(), # [seqlen, J, F] -> [J, F, seqlen]
            'n_joints': n_joints,
            'lengths': b[1],
            'parents': b[2],
            'graph_dist' : padded_graph_dist,
            'joints_relations':  padded_joints_relations,
            'object_type': object_type,
            'joints_names_embs': joints_names_embs,
            'rest_geom': rest_geom,
        }
        if extra_cond is not None:
            item['reference'] = extra_cond.get('reference')
        if extra_cond is not None and 'joint_mask_limb_roots' in extra_cond:
            raw_roots = np.asarray(extra_cond['joint_mask_limb_roots'], dtype=np.bool_)
            padded_roots = torch.zeros((max_joints,), dtype=torch.bool)
            root_count = min(max_joints, n_joints, int(raw_roots.shape[0]))
            padded_roots[:root_count] = torch.from_numpy(raw_roots[:root_count])
            item['joint_mask_limb_roots'] = padded_roots
        if extra_cond is not None and 'limb_tip_mask' in extra_cond:
            raw_tips = np.asarray(extra_cond['limb_tip_mask'], dtype=np.bool_)
            padded_tips = torch.zeros((max_joints,), dtype=torch.bool)
            tip_count = min(max_joints, n_joints, int(raw_tips.shape[0]))
            padded_tips[:tip_count] = torch.from_numpy(raw_tips[:tip_count])
            item['limb_tip_mask'] = padded_tips
        if extra_cond is not None and extra_cond.get('species_code') is not None:
            item['species_code'] = torch.from_numpy(np.asarray(extra_cond['species_code'], dtype=np.float32))
        if extra_cond is not None and 'rest_pos_ric_hml' in extra_cond:
            rest_pos = torch.zeros((max_joints, 3), dtype=torch.float32)
            raw_rest_pos = np.asarray(extra_cond['rest_pos_ric_hml'], dtype=np.float32)
            rest_count = min(max_joints, n_joints, raw_rest_pos.shape[0])
            rest_pos[:rest_count] = torch.from_numpy(raw_rest_pos[:rest_count])
            item['rest_pos_ric_hml'] = rest_pos
            # The skeleton's length scale L and the rotation channel's rest bone
            # frames, taken from the UNPADDED rig exactly as the encoder took
            # them, once, on the CPU, rather than re-derived on every
            # training-time aux-loss decode. Padding slots are identity frames
            # without a parent.
            rig_parents = np.asarray(b[2]).reshape(-1)[:rest_count]
            item[REST_LENGTH_SCALE_KEY] = float(_length_scale_from_rest(raw_rest_pos[:rest_count], rig_parents))
            frames, parent_index = padded_rotation_context(
                raw_rest_pos[:rest_count], rig_parents, max_joints
            )
            item[REST_BONE_FRAME_KEY] = torch.from_numpy(frames)
            item[PARENT_INDEX_KEY] = torch.from_numpy(parent_index)
        if extra_cond is not None and 'joint_struct' in extra_cond:
            raw_joint_struct = np.asarray(extra_cond['joint_struct'], dtype=np.float32)
            if raw_joint_struct.ndim != 2 or raw_joint_struct.shape[1] != JOINT_STRUCT_DIM:
                raise ValueError(
                    f"joint_struct must have shape [n_joints, {JOINT_STRUCT_DIM}], got "
                    f"{raw_joint_struct.shape}."
                )
            if raw_joint_struct.shape[0] != n_joints:
                raise ValueError(
                    f"joint_struct describes {raw_joint_struct.shape[0]} joints but the "
                    f"sample has {n_joints}."
                )
            # Rows past n_joints stay zero, like every other padded channel; the
            # model re-zeroes them AFTER its projection, since a zero row does not
            # survive an MLP with biases.
            joint_struct = torch.zeros((max_joints, JOINT_STRUCT_DIM), dtype=torch.float32)
            joint_struct[:n_joints] = torch.from_numpy(raw_joint_struct)
            item['joint_struct'] = joint_struct
        if extra_cond is not None and 'rest_pose_physical' in extra_cond:
            rest_physical = torch.zeros((max_joints, n_feats), dtype=torch.float32)
            raw_rest_physical = np.asarray(extra_cond['rest_pose_physical'], dtype=np.float32)
            count = min(max_joints, n_joints, raw_rest_physical.shape[0])
            rest_physical[:count, :min(n_feats, raw_rest_physical.shape[1])] = torch.from_numpy(
                raw_rest_physical[:count, :min(n_feats, raw_rest_physical.shape[1])]
            )
            item['rest_pose_physical'] = rest_physical
        if extra_cond is not None and extra_cond.get('canonical_feature_mean') is not None:
            item['canonical_feature_mean'] = torch.from_numpy(
                np.asarray(extra_cond['canonical_feature_mean'], dtype=np.float32).reshape(-1)
            )
        if extra_cond is not None and extra_cond.get('canonical_feature_std') is not None:
            item['canonical_feature_std'] = torch.from_numpy(
                np.asarray(extra_cond['canonical_feature_std'], dtype=np.float32).reshape(-1)
            )
        if extra_cond is not None:
            for scalar_key in (CANONICAL_DPOS_STD_KEY, DPOS_LOSS_WEIGHT_KEY):
                if extra_cond.get(scalar_key) is not None:
                    item[scalar_key] = float(extra_cond[scalar_key])
            item['feature_space'] = extra_cond.get('feature_space', CANONICAL_FEATURE_SPACE)
        if motion_metadata is not None:
            for key in ('action_group', 'action_label', 'action_slots', 'translation_root_index', 'is_loop', 'resample_speed_cond', 'motion_speed_applied', 'loop_data_aug_applied', 'loop_phase_offset', 'loop_tile_count', 'loop_cycles_active', 'bone_length_aug_applied'):
                if key in motion_metadata:
                    item[key] = motion_metadata[key]
            if 'species_code' in motion_metadata:
                item['species_code'] = torch.from_numpy(np.asarray(motion_metadata['species_code'], dtype=np.float32))
        if motion_name is not None:
            item['motion_name'] = motion_name
        adapted_batch.append(item)

    return truebones_collate(adapted_batch)
