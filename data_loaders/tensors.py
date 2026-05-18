import torch
import numpy as np

def lengths_to_mask(lengths, max_len):
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask

def n_joints_to_mask(n_joints, max_joints):
    mask = torch.arange(max_joints + 1, device=n_joints.device).expand(len(n_joints), max_joints + 1) < (n_joints.unsqueeze(1) + 1)
    mask = mask.unsqueeze(2).float() * mask.unsqueeze(1).float() 
    return mask

def length_to_temp_mask(max_len_mask, lengths, max_len):
    mask = torch.arange(max_len + 1, device=lengths.device).expand(len(lengths), max_len + 1) < (lengths.unsqueeze(1) + 1)
    mask = mask.unsqueeze(2).float() * mask.unsqueeze(1).float() 
    mask = mask.logical_and(max_len_mask)
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
    norm_meanbatch = [b['norm_mean'] for b in notnone_batches]
    norm_stdbatch = [b['norm_std'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]
    if 'n_joints' in notnone_batches[0]:
        jointsnumbatch = [b['n_joints'] for b in notnone_batches]
    else:
        jointsnumbatch = [22 for b in notnone_batches] #smpl n_joints 
        
    if 'temporal_mask' in notnone_batches[0]:
        temporalmasksbatch = [b['temporal_mask'] for b in notnone_batches]


    
    databatchTensor = collate_tensors(databatch)
    tposfirstframebatchTensor = collate_tensors(tposfirstframebatch)
    norm_meanbatchTensor = collate_tensors(norm_meanbatch)
    norm_stdbatchTensor = collate_tensors(norm_stdbatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    lengthsmaskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting
    jointsnumbatchTensor = torch.as_tensor(jointsnumbatch)
    jointsmaskbatchTensor = n_joints_to_mask(jointsnumbatchTensor, databatchTensor.shape[1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    collated_temporalmasksbatch = collate_tensors(temporalmasksbatch)
    maskbatchTensor = length_to_temp_mask(collated_temporalmasksbatch, lenbatchTensor, collated_temporalmasksbatch[0].size(0) - 1).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor, 'lengths_mask': lengthsmaskbatchTensor, 'tpos_first_frame': tposfirstframebatchTensor, 'norm_mean': norm_meanbatchTensor, 'norm_std':norm_stdbatchTensor}}

    if 'object_type' in notnone_batches[0]:
        objecttypebatch = [b['object_type'] for b in notnone_batches]
        cond['y'].update({'object_type': objecttypebatch})

    if 'motion_name' in notnone_batches[0]:
        motionnamebatch = [b['motion_name'] for b in notnone_batches]
        cond['y'].update({'motion_name': motionnamebatch})

    for key in ('species_label', 'species_group', 'action_tags', 'translation_root_index'):
        if any(key in batch_item for batch_item in notnone_batches):
            cond['y'].update({key: [batch_item.get(key) for batch_item in notnone_batches]})
    
    if 'parents' in notnone_batches[0]:
        parentsbatch = [b['parents'] for b in notnone_batches]
        cond['y'].update({'parents': parentsbatch})

    if 'offsets' in notnone_batches[0]:
        offsetsbatch = [b['offsets'] for b in notnone_batches]
        cond['y'].update({'offsets': collate_tensors(offsetsbatch)})

    if 'rest_rotations' in notnone_batches[0]:
        restrotbatch = [b['rest_rotations'] for b in notnone_batches]
        cond['y'].update({'rest_rotations': collate_tensors(restrotbatch)})

    if 'canon_joint_rot' in notnone_batches[0]:
        canonrotbatch = [b['canon_joint_rot'] for b in notnone_batches]
        cond['y'].update({'canon_joint_rot': collate_tensors(canonrotbatch)})

    if any('norm_schema_version' in batch_item for batch_item in notnone_batches):
        cond['y'].update({'norm_schema_version': torch.as_tensor([batch_item.get('norm_schema_version', 0) for batch_item in notnone_batches])})
          
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
 motion, m_length, parents, tpos_first_frame, offsets, self.temporal_mask_template, joints_graph_dist, joints_relations, object_type, joints_names_embs, ind, norm_mean, norm_std, max_joints
"""
def truebones_batch_collate(batch):
    if batch and isinstance(batch[0], dict):
        return truebones_collate(batch)

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
        norm_mean = torch.zeros((max_joints, n_feats))
        norm_mean[:n_joints] = torch.from_numpy(np.asarray(b[11], dtype=np.float32))
        norm_std = torch.ones((max_joints, n_feats))
        norm_std[:n_joints] = torch.from_numpy(np.asarray(b[12], dtype=np.float32))
        n_joints = b[0].shape[1]
        temporal_mask = torch.as_tensor(b[5][:max_len + 1, :max_len + 1])
        padded_joints_relations =  create_padded_relation(b[7], max_joints, n_joints)
        padded_graph_dist =  create_padded_relation(b[6], max_joints, n_joints)
        object_type = b[8]
        motion_metadata = None
        extra_cond = None
        motion_name = None
        for tail_item in b[14:]:
            if isinstance(tail_item, dict):
                if extra_cond is None and ('offsets' in tail_item or 'rest_rotations' in tail_item or 'canon_joint_rot' in tail_item):
                    extra_cond = tail_item
                    continue
                if motion_metadata is None and ('action_category' in tail_item or 'species_label' in tail_item or 'action_tags' in tail_item or 'translation_root_index' in tail_item):
                    motion_metadata = tail_item
            elif motion_name is None and isinstance(tail_item, str):
                motion_name = tail_item

        offsets = torch.zeros((max_joints, 3), dtype=torch.float32)
        offsets[:n_joints] = torch.from_numpy(np.asarray(b[4], dtype=np.float32))
        rest_rotations = torch.zeros((max_joints, 4), dtype=torch.float32)
        rest_rotations[:, 0] = 1.0
        canon_joint_rot = torch.zeros((max_joints, 4), dtype=torch.float32)
        canon_joint_rot[:, 0] = 1.0
        norm_schema_version = 0
        if extra_cond is not None:
            if 'offsets' in extra_cond:
                offsets[:n_joints] = torch.from_numpy(np.asarray(extra_cond['offsets'], dtype=np.float32))
            if 'rest_rotations' in extra_cond:
                rest_rotations[:n_joints] = torch.from_numpy(np.asarray(extra_cond['rest_rotations'], dtype=np.float32))
            if 'canon_joint_rot' in extra_cond:
                canon_joint_rot[:n_joints] = torch.from_numpy(np.asarray(extra_cond['canon_joint_rot'], dtype=np.float32))
            norm_schema_version = int(extra_cond.get('norm_schema_version', 0) or 0)

        item = {
            'inp': motion.permute(1, 2, 0).float(), # [seqlen , J, 13] -> [J, 13,  seqlen]
            'n_joints': n_joints,
            'lengths': b[1],
            'parents': b[2],
            'offsets': offsets,
            'rest_rotations': rest_rotations,
            'canon_joint_rot': canon_joint_rot,
            'norm_schema_version': norm_schema_version,
            'temporal_mask' : temporal_mask,
            'graph_dist' : padded_graph_dist,
            'joints_relations':  padded_joints_relations,
            'object_type': object_type,
            'joints_names_embs': joints_names_embs,
            'tpos_first_frame': tpos_first_frame,
            'norm_mean': norm_mean,
            'norm_std': norm_std
        }
        if motion_metadata is not None:
            for key in ('species_label', 'species_group', 'action_tags', 'translation_root_index'):
                if key in motion_metadata:
                    item[key] = motion_metadata[key]
        if motion_name is not None:
            item['motion_name'] = motion_name
        adapted_batch.append(item)

    return truebones_collate(adapted_batch)