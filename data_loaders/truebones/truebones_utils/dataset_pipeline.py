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
import bisect
from data_loaders.truebones.truebones_utils.param_utils import DEFAULT_DATASET_DIR, MAX_JOINTS, MAX_PATH_LEN, MOTION_DIR, FOOT_CONTACT_VEL_THRESH, BVHS_DIR, get_raw_data_dir
from pathlib import Path
from .motion_labels import build_motion_labels, build_object_labels, infer_motion_labels_from_motion_name, write_motion_metadata, load_motion_metadata
from .physics_joint_annotation import (
    build_semantic_metadata,
    rest_positions_from_offsets,
)
from .fbx_filename_rules import (
    find_tpose_reference_path,
    normalize_action_name,
    should_skip_anim,
)

from .animation_utils import (
    canonical_name_for_bvh,
    attach_joint_name_embeddings_to_cond,
    extend_semantic_metadata_with_leaf_helpers,
    needs_bvh_position_channels,
    reorder_animation_to_dfs,
    coerce_single_orientation_quat,
)

from .features import (
    get_common_features_from_T_pose,
    get_motion,
    infer_translation_root_index_from_features,
)


class DatasetPreprocessingError(RuntimeError):
    def __init__(self, motion_errors):
        self.motion_errors = tuple(str(err) for err in motion_errors)
        super().__init__(f"{len(self.motion_errors)} motion processing error(s)")


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


################## Dataset Pipeline ####################

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
        raw_action = normalize_action_name(object_type, raw_action)
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
    orientation_qs = coerce_single_orientation_quat(orientation_quat).qs[0]
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
    motion_labels['motion_source'] = 'anim_dir'

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
    rest_positions = rest_positions_from_offsets(tp.offsets, parents)
    original_joint_count = int(tp.helper_metadata['original_joint_count'])
    base_semantic_metadata = build_semantic_metadata(
        tp.names[:original_joint_count],
        parents[:original_joint_count],
        tp.offsets[:original_joint_count],
        rest_positions=rest_positions[:original_joint_count],
    )
    semantic_metadata = extend_semantic_metadata_with_leaf_helpers(
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
        canonical_name_for_bvh(canonical_name, raw_name)
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
def _build_tpose_only_cond(object_type, t_pos_path, face_joints):
    object_cond, tp, t_pos_motion, parents, semantic_metadata, character_scale_factor, _, max_joints = _build_tpose_cond(
        object_type, t_pos_path, face_joints,
    )
    num_joints = len(parents)

    # mean: T-pose feature vector with velocity channels (9:12) explicitly zeroed
    # to make rest-pose semantics unambiguous.
    mean = t_pos_motion[0].astype(np.float32).copy()  # (J, 13)
    mean[:, 9:12] = 0.0
    object_cond['mean'] = mean

    object_cond['std'] = np.ones_like(mean)

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
    anim_files = [f for f in anim_files if not should_skip_anim(f, object_type)]
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

    stats_tensors = np.concatenate(all_tensors, axis=0)

    mean, std = get_mean_std(stats_tensors)
    object_cond["mean"] = mean
    object_cond["std"] = std

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


def _print_dataset_summary(max_joints, files_counter, frames_counter):
    print('Total clips: %d, Frames: %d, Duration: %fm' %(files_counter, frames_counter, frames_counter / 12.5 / 60))
    print('max joints: %d' %(max_joints))


def _write_positions_error_file(save_dir, squared_positions_error):
    with open(pjoin(save_dir, 'positions_error_rate.txt'), 'w', encoding='utf-8') as error_file:
        error_file.write('Position squared error per source clip:\n')
        for source_clip, squared_error in squared_positions_error.items():
            error_file.write('%s: %f\n' %(source_clip, squared_error))


def _write_preprocess_seed_artifacts(save_dir, cond, motion_metadata, max_joints, files_counter, frames_counter, squared_positions_error):
    # Seed artifacts are the minimum inputs regeneration needs to rebuild the
    # full side-artifact set after motion export completes.
    _print_dataset_summary(max_joints, files_counter, frames_counter)
    _write_positions_error_file(save_dir, squared_positions_error)
    np.save(pjoin(save_dir, 'cond.npy'), cond)
    write_motion_metadata(save_dir, motion_metadata, files_counter)


def _write_dataset_artifacts(save_dir, cond, motion_metadata, objects_counter, max_joints, files_counter, frames_counter, squared_positions_error):
    _print_dataset_summary(max_joints, files_counter, frames_counter)
    with open(pjoin(save_dir, 'metadata.txt'), 'w', encoding='utf-8') as text_file:
        text_file.write('max joints: %d\n' %(max_joints))
        text_file.write('total frames: %d\n' %(frames_counter))
        text_file.write('duration: %d\n' %(frames_counter / 12.5 / 60))
        text_file.write('~~~~ objects_counts - Total: %d ~~~~\n' %(files_counter) )
        for obj in objects_counter:
            text_file.write('%s: %d\n' %(obj, objects_counter[obj]))

    _write_positions_error_file(save_dir, squared_positions_error)

    attach_joint_name_embeddings_to_cond(cond, save_dir)
    np.save(pjoin(save_dir, "cond.npy"), cond)
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
def process_object(object_type, files_counter, frames_counter, max_joints, squared_positions_error, save_dir = DEFAULT_DATASET_DIR, face_joints=None, fbxs_dir=None, t_pos_path=None, max_files=None, raw_data_dir=None):
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
        raise DatasetPreprocessingError(all_motion_errors)

    _write_preprocess_seed_artifacts(
        target_dataset_dir,
        cond,
        motion_metadata,
        max_joints,
        files_counter,
        frames_counter,
        squared_positions_error,
    )


"""Merge a freshly built object cond entry into an existing cond.npy in place.

Other objects already present in cond.npy are left untouched. mean/std written
here are provisional — regenerate_dataset_artifacts(recompute_stats=True) rebuilds
them over the merged clip set after an incremental --update."""
def _merge_object_into_cond(save_dir, object_name, object_cond):
    cond_path = pjoin(save_dir, 'cond.npy')
    cond = {}
    if os.path.exists(cond_path):
        cond = dict(np.load(cond_path, allow_pickle=True).item())
    cond[object_name] = object_cond
    np.save(cond_path, cond)


def _is_anim_dir_motion_entry(entry):
    return bool(entry.get('source_fbx_path')) or entry.get('motion_source') == 'anim_dir'


def _is_retarget_motion_entry(entry):
    return entry.get('motion_source') == 'retarget'


def _normalized_source_fbx_path(entry):
    source_fbx_path = entry.get('source_fbx_path')
    if not source_fbx_path:
        return None
    return os.path.realpath(str(source_fbx_path))


def validate_anim_dir_update_state(object_name, save_dir, existing_meta=None):
    """Refuse incremental anim-dir replacement when target clips are ambiguous.

    Replacing only the prior anim-dir clips is safe only when every existing
    target motion on disk is tracked in motion_metadata.json and explicitly
    identifiable as either a direct anim-dir clip or a preserved retarget clip.
    Legacy datasets without that metadata must be rebuilt instead of updated in
    place, otherwise an incoming source clip cannot reliably replace its older
    output slices without duplicating target motions."""
    motions_dir = pjoin(save_dir, MOTION_DIR)
    existing_meta = load_motion_metadata(save_dir) if existing_meta is None else existing_meta

    target_prefix = f"{object_name}_"
    target_motion_names = [
        p.name
        for p in sorted(Path(motions_dir).glob("*.npy"))
        if p.name.startswith(target_prefix)
    ]
    if not target_motion_names:
        return

    untracked_target = [
        motion_name for motion_name in target_motion_names if motion_name not in existing_meta
    ]
    if untracked_target:
        sample = ', '.join(untracked_target[:5])
        raise RuntimeError(
            f"cannot incrementally update anim-dir for {object_name}: existing target "
            f"motions are present on disk but missing from motion_metadata.json "
            f"({sample}). Rebuild this dataset without --update."
        )

    ambiguous_target = []
    for motion_name in target_motion_names:
        entry = existing_meta.get(motion_name, {})
        if str(entry.get('object_type', '')) != object_name:
            ambiguous_target.append(motion_name)
            continue
        if _is_anim_dir_motion_entry(entry) or _is_retarget_motion_entry(entry):
            continue
        ambiguous_target.append(motion_name)
    if ambiguous_target:
        sample = ', '.join(ambiguous_target[:5])
        raise RuntimeError(
            f"cannot incrementally update anim-dir for {object_name}: existing target "
            f"motions lack source metadata needed to distinguish direct clips from "
            f"preserved retarget clips ({sample}). Rebuild or regenerate this dataset "
            f"with explicit motion_source metadata."
        )


"""Incremental --update for the --anim-dir path.

Reprocesses the current --anim-dir input and merges the result into the
existing dataset. Prior anim-dir clips from the same source FBX files are
replaced in place, while untouched source clips and donor clips from a prior
--retarget-top-k run are preserved. Side artifacts are rebuilt afterwards by
regenerate_dataset_artifacts() in the caller."""
def _update_anim_dir(object_name, face_joints, save_dir, tpose_path, anim_dir):
    motions_dir = pjoin(save_dir, MOTION_DIR)
    bvhs_dir = pjoin(save_dir, BVHS_DIR)
    existing_meta = load_motion_metadata(save_dir)
    validate_anim_dir_update_state(object_name, save_dir, existing_meta)

    # Number new clips above every existing clip so they collide neither with
    # retained donor clips nor with older anim-dir outputs still on disk during
    # processing (matching-source clips are removed only after processing
    # succeeds, so a failed reprocess leaves the existing dataset intact).
    def _clip_index(name):
        tail = os.path.splitext(name)[0].rsplit('_', 1)[-1]
        return int(tail) if tail.isdigit() else 0
    start_counter = max([_clip_index(n) for n in existing_meta] + [0])

    squared_positions_error = dict()
    _, _, _, object_cond, new_meta = process_object(
        object_name,
        start_counter,
        0,
        23,
        squared_positions_error,
        save_dir=save_dir,
        fbxs_dir=anim_dir,
        face_joints=face_joints,
        t_pos_path=tpose_path,
    )
    if object_cond is None:
        print(f"[update] no valid animation data found in {anim_dir}; dataset unchanged")
        return

    replaced_sources = {
        _normalized_source_fbx_path(entry)
        for entry in new_meta.values()
    }
    replaced_sources.discard(None)

    # Processing succeeded — replace only prior direct clips that came from the
    # same source files as this update. Untouched direct clips and donor clips
    # are preserved, so A,B updated with B,C becomes A,B,C.
    kept_meta = {}
    replaced = 0
    for motion_name, entry in existing_meta.items():
        if (
            str(entry.get('object_type', '')) == object_name
            and _is_anim_dir_motion_entry(entry)
            and _normalized_source_fbx_path(entry) in replaced_sources
        ):
            npy_path = pjoin(motions_dir, motion_name)
            if os.path.exists(npy_path):
                os.remove(npy_path)
            bvh_path = pjoin(bvhs_dir, os.path.splitext(motion_name)[0] + '.bvh')
            if os.path.exists(bvh_path):
                os.remove(bvh_path)
            replaced += 1
        else:
            kept_meta[motion_name] = entry
    if replaced:
        print(f"[update] replaced {replaced} previously processed anim-dir clip(s) "
              f"from {len(replaced_sources)} updated source file(s)")

    merged_meta = dict(kept_meta)
    merged_meta.update(new_meta)
    write_motion_metadata(save_dir, merged_meta, len(merged_meta))
    _merge_object_into_cond(save_dir, object_name, object_cond)
    print(f"[update] anim-dir: {len(new_meta)} clip(s) written, "
          f"{len(merged_meta)} clip(s) total")


"""Incremental --update for the --retarget-top-k path.

Donor motions were already written to save_dir/motions/ by auto_retarget_pipeline
(deterministic `{target}_{donor}_{action}` names, so re-runs overwrite same-named
donors and add new ones). Existing clips are kept; cond.npy and motion_metadata
are merged. Side artifacts are rebuilt afterwards by the caller."""
def _update_retarget(object_name, save_dir, motions_from_npys, target_cond_partial):
    all_motions = [np.load(p).astype(np.float32) for p in motions_from_npys]
    if not all_motions:
        print("[update] no retargeted motions produced; dataset unchanged")
        return

    object_cond = dict(target_cond_partial)
    object_cond['mean'], object_cond['std'] = get_mean_std(
        np.concatenate(all_motions, axis=0)
    )

    parents = np.asarray(object_cond['parents'], dtype=np.int64)
    offsets = np.asarray(object_cond['offsets'], dtype=np.float64)
    existing_meta = load_motion_metadata(save_dir)
    new_meta = {}
    for motion_path, motion in zip(motions_from_npys, all_motions):
        motion_name = os.path.basename(motion_path)
        motion_labels = infer_motion_labels_from_motion_name(
            motion_name, object_type=object_name,
        )
        motion_labels['translation_root_index'] = int(
            infer_translation_root_index_from_features(motion, parents, offsets)
        )
        motion_labels['motion_source'] = 'retarget'
        new_meta[motion_name] = motion_labels

    merged_meta = dict(existing_meta)
    merged_meta.update(new_meta)
    write_motion_metadata(save_dir, merged_meta, len(merged_meta))
    _merge_object_into_cond(save_dir, object_name, object_cond)
    print(f"[update] retarget: {len(new_meta)} donor clip(s) written, "
          f"{len(merged_meta)} clip(s) total")


def process_skeleton(object_name, face_joints, save_dir, tpose_path, anim_dir=None,
                     motions_from_npys=None, target_cond_partial=None, update=False):
    ## prepare
    os.makedirs(pjoin(save_dir, MOTION_DIR), exist_ok=True)
    os.makedirs(pjoin(save_dir, BVHS_DIR), exist_ok=True)

    if motions_from_npys is not None:
        # Retarget branch: motions already written to save_dir/motions/ by auto_retarget_pipeline.
        # Load them, compute mean/std, then write cond.npy.
        assert target_cond_partial is not None, "target_cond_partial required with motions_from_npys"
        if update:
            _update_retarget(object_name, save_dir, motions_from_npys, target_cond_partial)
            return
        all_motions = [np.load(p).astype(np.float32) for p in motions_from_npys]
        if not all_motions:
            print(f"[process_skeleton] no retargeted motions available; cond.npy not written")
            return
        stats_tensors = np.concatenate(all_motions, axis=0)  # (total_frames, J, 13)
        mean, std = get_mean_std(stats_tensors)
        object_cond = dict(target_cond_partial)
        object_cond['mean'] = mean
        object_cond['std'] = std
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
            motion_labels['motion_source'] = 'retarget'
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
        )
        return

    if update and anim_dir is not None:
        _update_anim_dir(object_name, face_joints, save_dir, tpose_path, anim_dir)
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
    )



