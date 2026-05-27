import torch
import numpy as np

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

    for key in ('species_label', 'species_group', 'action_tags', 'translation_root_index'):
        if any(key in batch_item for batch_item in notnone_batches):
            cond['y'].update({key: [batch_item.get(key) for batch_item in notnone_batches]})

    for key in ('is_loop', 'loop_full_cycle'):
        if any(key in batch_item for batch_item in notnone_batches):
            cond['y'].update({key: torch.as_tensor([bool(batch_item.get(key, False)) for batch_item in notnone_batches], dtype=torch.bool)})

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

    if any('global_energy_cond' in batch_item for batch_item in notnone_batches):
        if not all('global_energy_cond' in batch_item for batch_item in notnone_batches):
            raise ValueError(
                "global_energy_cond must be present for all samples in a batch when any sample provides it."
            )
        cond['y'].update({
            'global_energy_cond': torch.stack([
                torch.as_tensor(batch_item['global_energy_cond'], dtype=torch.float32).reshape(2)
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
        
    if 'joints_relations' in notnone_batches[0]:
        jointsrelationsbatch = [b['joints_relations'] for b in notnone_batches]

    if 'graph_dist' in notnone_batches[0]:
        graphdistbatch = [b['graph_dist'] for b in notnone_batches]

    cond['y'].update({'joints_padding_mask': jointsmaskbatchTensor})
    cond['y'].update({'n_joints': jointsnumbatchTensor})
    cond['y'].update({'joints_relations': torch.stack(jointsrelationsbatch)})
    cond['y'].update({'graph_dist': torch.stack(graphdistbatch)})

    return motion, cond

""" recieves list of tuples of the form: 
 motion, m_length, parents, tpos_first_frame, offsets, self.temporal_mask_template, joints_graph_dist, joints_relations, object_type, joints_names_embs, ind, mean, std, max_joints
"""
def truebones_batch_collate(batch):
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
                elif any(key in extra for key in ('action_category', 'species_label', 'action_tags', 'translation_root_index', 'is_loop', 'loop_full_cycle', 'loop_phase_length', 'playspeed_cond', 'global_energy_cond')):
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
            'std': std
        }
        if extra_cond is not None and 'joint_mask_candidate_roots' in extra_cond:
            raw_candidates = np.asarray(extra_cond['joint_mask_candidate_roots'], dtype=np.bool_)
            padded_candidate_roots = torch.zeros((max_joints,), dtype=torch.bool)
            candidate_count = min(max_joints, n_joints, int(raw_candidates.shape[0]))
            if candidate_count > 0:
                padded_candidate_roots[:candidate_count] = torch.from_numpy(raw_candidates[:candidate_count])
            item['joint_mask_candidate_roots'] = padded_candidate_roots
        if motion_metadata is not None:
            for key in ('species_label', 'species_group', 'action_tags', 'translation_root_index', 'is_loop', 'loop_full_cycle', 'loop_phase_length', 'playspeed_cond', 'global_energy_cond'):
                if key in motion_metadata:
                    item[key] = motion_metadata[key]
        if motion_name is not None:
            item['motion_name'] = motion_name
        adapted_batch.append(item)

    return truebones_collate(adapted_batch)