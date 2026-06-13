import torch
import numpy as np


_ACTION_TAG_VOCAB = None
_ACTION_TAG_TO_INDEX = None


def _get_action_tag_vocab():
    global _ACTION_TAG_VOCAB, _ACTION_TAG_TO_INDEX
    if _ACTION_TAG_VOCAB is None or _ACTION_TAG_TO_INDEX is None:
        from data_loaders.truebones.truebones_utils.motion_labels import ACTION_TAGS

        _ACTION_TAG_VOCAB = list(ACTION_TAGS)
        _ACTION_TAG_TO_INDEX = {tag: i for i, tag in enumerate(_ACTION_TAG_VOCAB)}
    return _ACTION_TAG_VOCAB, _ACTION_TAG_TO_INDEX


def _build_action_tag_multihot_batch(action_tags_batch):
    vocab, tag_to_index = _get_action_tag_vocab()
    multihot = torch.zeros((len(action_tags_batch), len(vocab)), dtype=torch.float32)
    for row_index, raw_tags in enumerate(action_tags_batch):
        if raw_tags is None:
            continue
        tags = [raw_tags] if isinstance(raw_tags, str) else raw_tags
        for tag in tags:
            idx = tag_to_index.get(str(tag).strip().lower())
            if idx is not None:
                multihot[row_index, idx] = 1.0
    return multihot

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
    containing keys like ``'inp'``, ``'tpos_first_frame'``, ``'mean'``,
    ``'std'``, ``'motion_start_frame'``, etc.

    Returns
    -------
    inp : torch.Tensor
        Concatenated motion input tensor.
    cond : dict
        Conditioning dictionary with ``'y'`` containing all metadata
        (mask, lengths, T-pose, normalization stats, object type,
        motion name, action tags, loop info, motion_start_frame, …).
    """
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    tposfirstframebatch = [b['tpos_first_frame'] for b in notnone_batches]
    meanbatch = [b['mean'] for b in notnone_batches]
    stdbatch = [b['std'] for b in notnone_batches]
    if 'n_joints' in notnone_batches[0]:
        jointsnumbatch = [b['n_joints'] for b in notnone_batches]
    else:
        jointsnumbatch = [22 for b in notnone_batches] #smpl n_joints 
        
    if 'temporal_mask' in notnone_batches[0]:
        temporalmasksbatch = [b['temporal_mask'] for b in notnone_batches]


    
    databatchTensor = collate_tensors(databatch)
    tposfirstframebatchTensor = collate_tensors(tposfirstframebatch)
    meanbatchTensor = collate_tensors(meanbatch)
    stdbatchTensor = collate_tensors(stdbatch)
    jointsnumbatchTensor = torch.as_tensor(jointsnumbatch)
    jointsmaskbatchTensor = n_joints_to_mask(jointsnumbatchTensor, databatchTensor.shape[1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    collated_temporalmasksbatch = collate_tensors(temporalmasksbatch)
    # All samples are resampled to a fixed frame count, so the per-sample
    # temporal mask is the windowed-attention template (no length-derived
    # padding portion).
    maskbatchTensor = collated_temporalmasksbatch.unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    frame_count = databatchTensor.shape[-1]
    batch_size = len(databatch)
    cond = {'y': {'mask': maskbatchTensor, 'lengths': torch.full((batch_size,), frame_count, dtype=torch.long), 'tpos_first_frame': tposfirstframebatchTensor, 'mean': meanbatchTensor, 'std':stdbatchTensor}}

    if 'object_type' in notnone_batches[0]:
        objecttypebatch = [b['object_type'] for b in notnone_batches]
        cond['y'].update({'object_type': objecttypebatch})

    if 'motion_name' in notnone_batches[0]:
        motionnamebatch = [b['motion_name'] for b in notnone_batches]
        cond['y'].update({'motion_name': motionnamebatch})

    if any('action_tags' in batch_item for batch_item in notnone_batches):
        action_tags_batch = [batch_item.get('action_tags') for batch_item in notnone_batches]
        cond['y'].update({
            'action_tags': action_tags_batch,
            'action_tag_multihot': _build_action_tag_multihot_batch(action_tags_batch),
        })

    if any('translation_root_index' in batch_item for batch_item in notnone_batches):
        cond['y'].update({
            'translation_root_index': [batch_item.get('translation_root_index') for batch_item in notnone_batches]
        })

    for key in ('is_loop', 'loop_full_cycle'):
        if any(key in batch_item for batch_item in notnone_batches):
            cond['y'].update({key: torch.as_tensor([bool(batch_item.get(key, False)) for batch_item in notnone_batches], dtype=torch.bool)})

    if any('loop_data_aug_applied' in batch_item for batch_item in notnone_batches):
        cond['y'].update({
            'loop_data_aug_applied': torch.as_tensor(
                [bool(batch_item.get('loop_data_aug_applied', False)) for batch_item in notnone_batches],
                dtype=torch.bool,
            )
        })

    if any('loop_phase_length' in batch_item for batch_item in notnone_batches):
        cond['y'].update({
            'loop_phase_lengths': torch.as_tensor(
                [float(batch_item.get('loop_phase_length', batch_item.get('lengths', 1))) for batch_item in notnone_batches],
                dtype=torch.float32,
            )
        })

    if any('playspeed_cond' in batch_item for batch_item in notnone_batches):
        cond['y'].update({
            'playspeed_cond': torch.as_tensor(
                [float(batch_item.get('playspeed_cond', 1.0)) for batch_item in notnone_batches],
                dtype=torch.float32,
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

    if any('motion_start_frame' in batch_item for batch_item in notnone_batches):
        cond['y'].update({
            'motion_start_frame': torch.as_tensor(
                [int(batch_item.get('motion_start_frame', 0)) for batch_item in notnone_batches],
                dtype=torch.long,
            )
        })

    if any('global_energy_cond' in batch_item for batch_item in notnone_batches):
        if not all('global_energy_cond' in batch_item for batch_item in notnone_batches):
            raise ValueError(
                "global_energy_cond must be present for all samples in a batch when any sample provides it."
            )
        cond['y'].update({
            'global_energy_cond': torch.stack([
                torch.as_tensor(batch_item['global_energy_cond'], dtype=torch.float32).reshape(1)
                for batch_item in notnone_batches
            ]),
        })

    if any('joint_mask_candidate_roots' in batch_item for batch_item in notnone_batches):
        candidate_root_batch = []
        for batch_item in notnone_batches:
            candidate_root_batch.append(batch_item.get('joint_mask_candidate_roots', torch.zeros(databatchTensor.shape[1], dtype=torch.bool)).bool())
        cond['y'].update({'joint_mask_candidate_roots': torch.stack(candidate_root_batch)})
    
    if 'parents' in notnone_batches[0]:
        parentsbatch = [b['parents'] for b in notnone_batches]
        cond['y'].update({'parents': parentsbatch})
          
    if 'joints_names_embs' in notnone_batches[0]:
        jointsnamesembsbatch = [b['joints_names_embs'] for b in notnone_batches]
        jointsnamesembsbatchTensor = collate_tensors(jointsnamesembsbatch)
        cond['y'].update({'joints_names_embs': jointsnamesembsbatchTensor})

    if 'species_emb' in notnone_batches[0]:
        speciesembbatch = [b['species_emb'] for b in notnone_batches]
        cond['y'].update({'species_emb': torch.stack(speciesembbatch)})
        
    if 'joints_relations' in notnone_batches[0]:
        jointsrelationsbatch = [b['joints_relations'] for b in notnone_batches]

    if 'graph_dist' in notnone_batches[0]:
        graphdistbatch = [b['graph_dist'] for b in notnone_batches]

    cond['y'].update({'joints_padding_mask': jointsmaskbatchTensor})
    cond['y'].update({'n_joints': jointsnumbatchTensor})
    cond['y'].update({'joints_relations': torch.stack(jointsrelationsbatch)})
    cond['y'].update({'graph_dist': torch.stack(graphdistbatch)})

    return motion, cond

def truebones_batch_collate(batch):
    """Collate a raw batch from MotionDataset into the format for truebones_collate.

    Each element ``b`` in *batch* is a tuple returned by
    ``MotionDataset._prepare_sample`` with the following layout:

        [0]  motion          – np.ndarray, (frames, n_joints, n_feats)
        [1]  m_length        – int, motion length in frames
        [2]  parents         – np.ndarray, parent indices
        [3]  tpos_first_frame – np.ndarray, T-pose first frame features
        [4]  offsets         – np.ndarray, bone offsets
        [5]  temporal_mask   – np.ndarray, temporal attention mask
        [6]  joints_graph_dist – np.ndarray, graph distance matrix
        [7]  joints_relations  – np.ndarray, joint relation matrix
        [8]  object_type     – str
        [9]  joints_names_embs – np.ndarray, joint name embeddings
        [10] crop_start      – int, starting frame index in source motion
        [11] mean            – np.ndarray, per-joint normalization mean
        [12] std             – np.ndarray, per-joint normalization std
        [13] max_joints      – int
        [14] motion_metadata – dict or None (action tags, loop info, etc.)
        [15] name            – str, motion name
        [16+] extras         – dicts (joint_mask_candidate_roots, aug info, …)
    """
    max_joints = batch[0][13]
    adapted_batch = []
    for b in batch:  
        max_len, n_joints, n_feats = b[0].shape
        tpos_first_frame = torch.zeros((max_joints, n_feats))
        tpos_first_frame[:n_joints] = torch.from_numpy(np.asarray(b[3], dtype=np.float32))
        motion = torch.zeros((max_len, max_joints, n_feats)) # (frames, max_joints, feature_len) 
        motion[:, :b[0].shape[1], :] = torch.from_numpy(np.asarray(b[0], dtype=np.float32))
        joints_names_embs = torch.zeros((max_joints, b[9].shape[1]))
        joints_names_embs[:n_joints] = torch.from_numpy(np.asarray(b[9], dtype=np.float32))
        mean = torch.zeros((max_joints, n_feats))
        mean[:n_joints] = torch.from_numpy(np.asarray(b[11], dtype=np.float32))
        std = torch.ones((max_joints, n_feats))
        std[:n_joints] = torch.from_numpy(np.asarray(b[12], dtype=np.float32))
        n_joints = b[0].shape[1]
        temporal_mask = torch.as_tensor(b[5][:max_len + 1, :max_len + 1])
        padded_joints_relations =  create_padded_relation(b[7], max_joints, n_joints)
        padded_graph_dist =  create_padded_relation(b[6], max_joints, n_joints)
        object_type = b[8]
        motion_metadata = None
        motion_name = None
        extra_cond = None
        for extra in b[14:]:
            if isinstance(extra, dict):
                if 'joint_mask_candidate_roots' in extra:
                    extra_cond = extra
                elif any(key in extra for key in ('species_label', 'action_tags', 'translation_root_index', 'is_loop', 'loop_full_cycle', 'loop_phase_length', 'playspeed_cond', 'global_energy_cond', 'loop_data_aug_applied', 'loop_phase_offset', 'loop_tile_count')):
                    motion_metadata = extra
            elif isinstance(extra, str):
                motion_name = extra

        item = {
            'inp': motion.permute(1, 2, 0).float(), # [seqlen , J, 13] -> [J, 13,  seqlen]
            'n_joints': n_joints,
            'lengths': b[1],
            'parents': b[2],
            'temporal_mask' : temporal_mask,
            'graph_dist' : padded_graph_dist,
            'joints_relations':  padded_joints_relations,
            'object_type': object_type,
            'joints_names_embs': joints_names_embs,
            'tpos_first_frame': tpos_first_frame,
            'mean': mean,
            'std': std,
            'motion_start_frame': int(b[10]),  # crop_start from _prepare_sample
        }
        if extra_cond is not None and 'joint_mask_candidate_roots' in extra_cond:
            raw_candidates = np.asarray(extra_cond['joint_mask_candidate_roots'], dtype=np.bool_)
            padded_candidate_roots = torch.zeros((max_joints,), dtype=torch.bool)
            candidate_count = min(max_joints, n_joints, int(raw_candidates.shape[0]))
            if candidate_count > 0:
                padded_candidate_roots[:candidate_count] = torch.from_numpy(raw_candidates[:candidate_count])
            item['joint_mask_candidate_roots'] = padded_candidate_roots
        if extra_cond is not None and extra_cond.get('species_emb') is not None:
            item['species_emb'] = torch.from_numpy(np.asarray(extra_cond['species_emb'], dtype=np.float32))
        if motion_metadata is not None:
            for key in ('action_tags', 'translation_root_index', 'is_loop', 'loop_full_cycle', 'loop_phase_length', 'playspeed_cond', 'global_energy_cond', 'loop_data_aug_applied', 'loop_phase_offset', 'loop_tile_count'):
                if key in motion_metadata:
                    item[key] = motion_metadata[key]
        if motion_name is not None:
            item['motion_name'] = motion_name
        adapted_batch.append(item)

    return truebones_collate(adapted_batch)
