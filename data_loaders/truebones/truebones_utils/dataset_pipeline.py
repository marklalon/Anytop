"""Dataset pipeline & augmentation.

Top layer of the motion-processing pipeline. Orchestrates the full dataset
preprocessing workflow, statistics computation, topology analysis, data
augmentation, and skeleton-processing entry points.

Depends on: features.py, animation_utils.py
"""

from motion_lib import BVH, FBX
import json
import numpy as np
import os
import sys
from os.path import join as pjoin
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import bisect
from data_loaders.truebones.truebones_utils.param_utils import DEFAULT_DATASET_DIR, MAX_JOINTS, MAX_PATH_LEN, MOTION_DIR, MOTION_METADATA_FILE, FOOT_CONTACT_VEL_THRESH, BVHS_DIR, get_raw_data_dir
from pathlib import Path
from .motion_labels import build_motion_labels, build_object_labels, write_motion_metadata, load_motion_metadata
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
    attach_t5_embeddings_to_cond,
    needs_bvh_position_channels,
    reorder_animation_to_dfs,
    crop_animation_to_max_joints,
    coerce_single_orientation_quat,
)

from .features import (
    get_common_features_from_rest_pose,
    get_motion,
    infer_translation_root_index_from_features,
    extract_motion_features_from_aligned_anims,
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

        # Universal guard for constant (zero-variance) channels across ALL
        # channels. The own-rotation encoding leaves the root joint's 6D
        # rotation constant, so its block-averaged std collapses to 0 above.
        # A ~zero divisor is meaningless and dangerous: it leaves the motion at
        # 0 but amplifies tpos_first_frame normalization, (tpos - mean) / std,
        # to ~1e6, which drives the spatial-attention graph bias to ~1e5 and
        # yields NaN gradients under bf16. Treat any sub-1e-5 std as unit
        # variance (mirrors the std_safe floor in data/dataset.py:Truebones).
        Std = np.where(Std < 1e-5, 1.0, Std)

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
                         orientation_quat, crop_enabled=True):
    local_errors = dict()
    _crop_max = MAX_JOINTS if crop_enabled else 2 ** 16
    # Load the animation file (FBX/GLB/GLTF) once; pass it as `preloaded` to every get_motion call so that
    raw_anim, names, frame_time = FBX.load(file_path)
    # Crop oversized skeletons to the crop cap so the loaded animation and its
    # exported names match the cropped rest-pose offsets. Leaves are peeled from
    # deepest to shallowest; ties at the same depth prefer shorter bones first,
    # while longer-than-average bones are preserved whenever possible. The cap
    # is ``_crop_max`` (defaults to MAX_JOINTS for training); the running
    # ``max_joints`` is a dataset-wide maximum used only for padding/metadata.
    # The crop stays deterministic for a given skeleton (same topology and
    # offsets), so it removes the same joints the rest pose did.
    if crop_enabled:
        raw_anim, names, _ = crop_animation_to_max_joints(
            raw_anim,
            names,
            max_joints=_crop_max,
            context=f"{object_type} '{os.path.basename(str(file_path))}'",
        )
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
            'motion_labels': build_motion_labels(object_type),
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


"""Load a reference FBX/GLB, build rest-pose-based cond, and return all caller values."""
def _build_rest_pose_cond(object_type, rest_pose_path, face_joints, max_joints=MAX_JOINTS,
                          crop_enabled=True):
    squared_positions_error = dict()
    _crop_max = MAX_JOINTS if crop_enabled else 2 ** 16
    tp = get_common_features_from_rest_pose(
        rest_pose_path,
        object_type,
        face_joints=face_joints,
        max_joints=_crop_max,
    )
    character_scale_factor = float(tp.scale_factor)
    rest_pose_motion, parents, max_joints, new_anim, _export_anim, _rest_is_loop, _rest_translation_root_index, _rest_root_translation_xz = get_motion(
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
        animation_input_is_tpose_aligned=False,
    )
    rest_positions = rest_positions_from_offsets(tp.offsets, parents)
    semantic_metadata = build_semantic_metadata(
        tp.names,
        parents,
        tp.offsets,
        rest_positions=rest_positions,
    )
    object_cond = dict()
    # Provisional translation root from the T-pose animation. Will be refreshed
    # by regenerate_dataset_artifacts._normalize_object_translation_roots, which
    # aggregates per-motion translation_root_index values and picks the consensus.
    object_cond['translation_root_index'] = int(_rest_translation_root_index)
    object_cond['tpos_first_frame'] = rest_pose_motion[0]
    object_cond['pose_base'] = 'rest_pose'
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
        rest_pose_path,
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
    object_cond['is_symmetric'] = semantic_metadata['is_symmetric']
    object_cond['scale_factor'] = character_scale_factor
    object_cond['axial_avg_len'] = float(tp.axial_avg_len)
    object_cond['kinematic_chains'] = parents2kinchains(parents, object_policy(object_type))
    object_cond.update(build_object_labels(object_type))
    return object_cond, tp, rest_pose_motion, parents, semantic_metadata, character_scale_factor, squared_positions_error, max_joints


def build_tpose_cond(*args, **kwargs):
    """Backward-compatible alias; cond is now built from the file bind/rest pose."""
    return _build_rest_pose_cond(*args, **kwargs)


"""Build the rest-pose cond dict from a single FBX/GLB file (no motion files needed)."""
def _build_rest_pose_only_cond(object_type, rest_pose_path, face_joints, crop_enabled=True):
    object_cond, tp, rest_pose_motion, parents, semantic_metadata, character_scale_factor, _, max_joints = _build_rest_pose_cond(
        object_type, rest_pose_path, face_joints, crop_enabled=crop_enabled,
    )
    num_joints = len(parents)

    # mean: rest-pose feature vector with velocity channels (9:12) explicitly zeroed
    # to make rest-pose semantics unambiguous.
    mean = rest_pose_motion[0].astype(np.float32).copy()  # (J, 13)
    mean[:, 9:12] = 0.0
    object_cond['mean'] = mean

    object_cond['std'] = np.ones_like(mean)

    return object_cond, max_joints


def _resample_animation(anim, target_len):
    """Resample Animation to target_len frames; uses slerp for rotations, linear for positions."""
    from motion_lib.Animation import Animation
    from motion_lib.Quaternions import Quaternions

    src_len = len(anim)
    if src_len == target_len:
        return anim
    t = np.linspace(0, src_len - 1, target_len)
    lo = np.floor(t).astype(int)
    hi = np.minimum(lo + 1, src_len - 1)
    w = t - lo  # (target_len,)

    # Positions: linear interpolation
    new_pos = anim.positions[lo] * (1.0 - w[:, None, None]) + anim.positions[hi] * w[:, None, None]

    # Rotations: slerp each target frame
    qs_a = anim.rotations.qs[lo]
    qs_b = anim.rotations.qs[hi]
    new_qs = np.zeros_like(qs_a)
    for i in range(target_len):
        q = Quaternions.slerp(Quaternions(qs_a[i]), Quaternions(qs_b[i]), float(w[i]))
        new_qs[i] = q.qs
    new_rot = Quaternions(new_qs)

    return Animation(new_rot, new_pos, anim.orients, anim.offsets, anim.parents)


"""Prepare processed tensors for all the files of a given object without writing them to disk yet.

``skip_source_paths`` (incremental preprocessing): realpaths of source anim files that
already produced clips on disk. Matching files are dropped from this run so only newly
added source files are (re)processed. The rest-pose reference carrier is still selected
from the full file list, so the per-object cond stays stable regardless of which clips
are new. Returns None when no source files remain to process (object fully up to date)."""
def _prepare_object_outputs(object_type, max_joints, face_joints=None, fbxs_dir=None, t_pos_path=None, max_files=None, raw_data_dir=None, filter_min_length=10, resample_min_length=20, skip_source_paths=None, crop_enabled=True):
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
    ## get a character-level rest-pose reference carrier
    if t_pos_path is None or t_pos_path == '':
        t_pos_path = find_tpose_reference_path(anim_files)
    else:
        # removes a static reference file from anim_files, as it should be used only for
        # extracting common characteristics. If this is not the case, disable this part
        anim_files.remove(t_pos_path)
    if max_files is not None:
        anim_files = anim_files[:max_files]

    # Filter out files with no inferable action name or all-in-one animation bundles
    anim_files = [f for f in anim_files if not should_skip_anim(f, object_type)]
    if len(anim_files) == 0:
        print(f'skipping {object_type}: no valid animation files after filtering')
        return None

    # Incremental: drop source files that already produced clips, so only newly added
    # animations are processed. Done before the (expensive) rest-pose cond build below.
    if skip_source_paths:
        skip_norm = {os.path.realpath(p) for p in skip_source_paths}
        kept = [f for f in anim_files if os.path.realpath(f) not in skip_norm]
        skipped = len(anim_files) - len(kept)
        if skipped:
            print(f'{object_type}: skipping {skipped} already-processed source file(s), {len(kept)} new to process')
        anim_files = kept
        if len(anim_files) == 0:
            print(f'skipping {object_type}: all source files already processed')
            return None

    squared_positions_error = dict()
    object_cond, tp, rest_pose_motion, parents, semantic_metadata, character_scale_factor, _, max_joints = _build_rest_pose_cond(
        object_type, t_pos_path, face_joints, max_joints=max_joints, crop_enabled=crop_enabled,
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
            orientation_quat=tp.orientation_quat,
            crop_enabled=crop_enabled,
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
            num_frames = result['motion'].shape[0]
            if num_frames < filter_min_length:
                continue
            if resample_min_length > 0 and num_frames < resample_min_length:
                # Resample the animation (slerp rotations / linear positions), then
                # RECOMPUTE the feature tensor from it. Interpolating the precomputed
                # feature tensor directly would corrupt every channel: velocity is
                # per-frame displacement (so it must shrink as frames are added),
                # foot contact is binary, and the 6D rotation rep is not closed under
                # linear interpolation. Re-extraction keeps vel = diff(pos), valid
                # rotations, and re-thresholded contacts — all physically consistent
                # with the resampled motion.
                result['new_anim'] = _resample_animation(result['new_anim'], resample_min_length)
                result['export_anim'] = _resample_animation(result['export_anim'], resample_min_length)
                motion, _, _, _, is_loop = extract_motion_features_from_aligned_anims(
                    result['new_anim'],
                    result['export_anim'],
                    FOOT_CONTACT_VEL_THRESH,
                    object_type,
                    max_joints,
                    tp.foot_indices,
                    tp.orientation_quat,
                    result['translation_root_index'],
                )
                result['motion'] = motion
                result['is_loop'] = is_loop
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


"""Write a prepared object payload to disk with stable per-(species, action) clip naming.

The trailing clip number is a segment index counted *within* each
`{object_type}_{action}` group (1, 2, 3, ...), not a global running counter.
This keeps a clip's name stable when unrelated species are added/removed or when
only a subset is reprocessed (e.g. --filter), so externally maintained, clip-name
keyed sidecars (action_tags.jsonl, motion_captions.jsonl) do not go stale.

`files_counter` is still threaded through purely for the dataset-wide summary
counts; it no longer participates in clip names. `action_start_counts` lets the
incremental --update path continue numbering above existing clips of the same
(object, action) group so freshly written clips never collide with retained ones.
"""
def _write_object_outputs(save_dir, object_payload, files_counter, action_start_counts=None):
    object_type = object_payload['object_type']
    frames_counter = 0
    motion_metadata = {}
    action_counter = dict(action_start_counts or {})

    for result in object_payload['results']:
        motion = result['motion']
        files_counter += 1
        frames_counter += motion.shape[0]
        action = result['action']
        action_counter[action] = action_counter.get(action, 0) + 1
        name = object_type + "_" + action + "_" + str(action_counter[action])
        motion_file_name = name + '.npy'
        np.save(pjoin(save_dir, MOTION_DIR, motion_file_name), motion)
        # Export the visually faithful processed animation rather than the
        # rest-pose-reparameterized training animation. The latter preserves global
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

    attach_t5_embeddings_to_cond(cond, save_dir)
    np.save(pjoin(save_dir, "cond.npy"), cond)
    write_motion_metadata(save_dir, motion_metadata, files_counter)


def _resolve_preprocessing_workers(objects, object_workers=8):
    object_count = max(1, len(objects))
    return min(object_count, max(1, int(object_workers)))


def _prepare_object_outputs_worker(object_type, max_files, raw_data_dir=None, filter_min_length=10, resample_min_length=20, skip_source_paths=None):
    # ── Install a local warning collector inside the worker process ──────
    # The parent's _WarnCollector monkey-patches do NOT propagate into
    # ProcessPoolExecutor children.  Capture _warn() / degenerate-facing
    # calls here and return them so the parent can print a deduplicated
    # summary via its own collector.
    from . import animation_utils as _au
    from . import face_orientation as _fo
    _warn_messages: list[str] = []
    _original_warn = _au._warn
    _original_emit = _fo._emit_degenerate_facing_warning
    _au._warn = lambda msg: _warn_messages.append(msg)
    _fo._emit_degenerate_facing_warning = lambda ot, wk, msg: _warn_messages.append(msg)
    try:
        payload = _prepare_object_outputs(
            object_type,
            max_joints=23,
            max_files=max_files,
            raw_data_dir=raw_data_dir,
            filter_min_length=filter_min_length,
            resample_min_length=resample_min_length,
            skip_source_paths=skip_source_paths,
        )
        if payload is not None:
            payload['_warn_messages'] = _warn_messages
        return payload
    finally:
        _au._warn = _original_warn
        _fo._emit_degenerate_facing_warning = _original_emit


""" creates processed tensors for all the files of a given object. Returens statistics and the object condition,
which includes rest-pose/tpos-compatible conditioning, relation/distances matrices, offsets, parents, joints names, kinematic chains, mean and std"""    
def process_object(object_type, files_counter, frames_counter, max_joints, squared_positions_error, save_dir = DEFAULT_DATASET_DIR, face_joints=None, fbxs_dir=None, t_pos_path=None, max_files=None, raw_data_dir=None, action_start_counts=None, crop_enabled=True):
    object_payload = _prepare_object_outputs(
        object_type,
        max_joints,
        face_joints=face_joints,
        fbxs_dir=fbxs_dir,
        t_pos_path=t_pos_path,
        max_files=max_files,
        raw_data_dir=raw_data_dir,
        crop_enabled=crop_enabled,
    )
    if object_payload is None:
        return files_counter, frames_counter, max_joints, None, {}

    squared_positions_error.update(object_payload['errors'])
    max_joints = max(max_joints, object_payload['max_joints'])
    files_counter, object_frames_counter, object_motion_metadata = _write_object_outputs(
        save_dir,
        object_payload,
        files_counter,
        action_start_counts=action_start_counts,
    )
    frames_counter += object_frames_counter

    return files_counter, frames_counter, max_joints, object_payload['object_cond'], object_motion_metadata


""" create dataset

``incremental``: keep already-processed clips on disk and only process source anim
files that have not produced clips yet (per-object, keyed on source_fbx_path). New
clips number above retained ones within each (object, action) group. The rewritten
cond.npy / motion_metadata.json are seeded from the existing dataset so untouched
objects survive. Provisional mean/std are computed over the newly processed clips only;
the caller must finalize them with regenerate_dataset_artifacts(recompute_stats=True).
Without ``incremental`` the prior full-build behavior is unchanged (callers wipe
outputs first). """
def create_data_samples(objects=None, max_files_per_object=None, dataset_dir=None, raw_data_dir=None, object_workers=8, filter_min_length=10, resample_min_length=20, incremental=False):
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

    # Incremental: read the existing dataset so we can skip done source files, continue
    # clip numbering above retained clips, and seed the merged cond/metadata.
    existing_cond = {}
    existing_meta = {}
    per_object_skip = {}
    per_object_action_start = {}
    if incremental:
        existing_meta = _load_motion_metadata_raw(target_dataset_dir)
        cond_path = pjoin(target_dataset_dir, 'cond.npy')
        if os.path.exists(cond_path):
            existing_cond = dict(np.load(cond_path, allow_pickle=True).item())
        for object_type in objects:
            per_object_skip[object_type] = _object_processed_sources(existing_meta, object_type)
            per_object_action_start[object_type] = _object_action_start_counts(existing_meta, object_type)

    obj_workers = _resolve_preprocessing_workers(
        objects,
        object_workers=object_workers,
    )
    print(f'Preprocessing {len(objects)} characters with {obj_workers} object workers'
          f"{' (incremental)' if incremental else ''}")

    payloads = [None] * len(objects)
    if obj_workers <= 1:
        for idx, object_type in enumerate(objects):
            payloads[idx] = _prepare_object_outputs(
                object_type,
                max_joints=23,
                max_files=max_files_per_object,
                raw_data_dir=raw_data_dir,
                filter_min_length=filter_min_length,
                resample_min_length=resample_min_length,
                skip_source_paths=per_object_skip.get(object_type),
            )
    else:
        with ProcessPoolExecutor(max_workers=obj_workers) as executor:
            future_to_idx = {
                executor.submit(
                    _prepare_object_outputs_worker,
                    object_type,
                    max_files_per_object,
                    raw_data_dir,
                    filter_min_length,
                    resample_min_length,
                    per_object_skip.get(object_type),
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
    # Seed from the existing dataset in incremental mode so objects/clips we do not touch
    # this run survive the cond.npy / motion_metadata rewrite below.
    cond = dict(existing_cond)
    motion_metadata = dict(existing_meta)

    all_motion_errors = []
    all_warn_messages: list[str] = []
    for idx, object_type in enumerate(objects):
        payload = payloads[idx]
        if payload is None:
            continue
        squared_positions_error.update(payload['errors'])
        max_joints = max(max_joints, payload['max_joints'])
        all_motion_errors.extend(payload.get('motion_errors', []))
        all_warn_messages.extend(payload.pop('_warn_messages', []))
        cur_counter = files_counter
        files_counter, object_frames, object_motion_metadata = _write_object_outputs(
            target_dataset_dir,
            payload,
            files_counter,
            action_start_counts=per_object_action_start.get(object_type),
        )
        frames_counter += object_frames
        cond[object_type] = payload['object_cond']
        objects_counter[object_type] = files_counter - cur_counter
        motion_metadata.update(object_motion_metadata)

    # ── Re-emit worker-collected warnings through _warn so the parent's
    # _WarnCollector (which patches _warn) picks them up and prints a
    # single deduplicated summary at the end of STEP 1.
    if all_warn_messages:
        from .animation_utils import _warn as _au_warn
        for msg in all_warn_messages:
            _au_warn(msg)

    if all_motion_errors:
        print(f"\n{'=' * 70}")
        print(f"\x1b[31mMOTION PROCESSING ERRORS ({len(all_motion_errors)} total)\x1b[0m")
        print('=' * 70)
        for err in all_motion_errors:
            print(err)
        print(f"{'=' * 70}\n")
        raise DatasetPreprocessingError(all_motion_errors)

    # total_clips reflects the full merged set on disk (seeded entries + freshly written).
    total_clips = len(motion_metadata) if incremental else files_counter
    _write_preprocess_seed_artifacts(
        target_dataset_dir,
        cond,
        motion_metadata,
        max_joints,
        total_clips,
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


def _load_motion_metadata_raw(dataset_dir):
    """Read motion_metadata.json's per-clip entries directly, without joining action_tags.

    Unlike load_motion_metadata this never requires the action_tags.jsonl sidecar, so the
    incremental path can read source/numbering bookkeeping on datasets that have not been
    hand-tagged yet. Returns {} when the file is absent or malformed."""
    metadata_path = pjoin(str(dataset_dir), MOTION_METADATA_FILE)
    if not os.path.exists(metadata_path):
        return {}
    with open(metadata_path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    motions = payload.get('motions', payload)
    if not isinstance(motions, dict):
        return {}
    return {name: dict(entry) for name, entry in motions.items() if isinstance(entry, dict)}


def _parse_action_and_index(object_name, motion_name):
    """Split a clip file name into its (action, per-action index) pair.

    Clip names are ``{object_name}_{action}_{index}.npy``; the trailing index is a
    per-(object, action) segment counter. Returns (None, 0) for names that do not
    belong to this object or lack the trailing numeric index."""
    stem = os.path.splitext(motion_name)[0]
    if not stem.startswith(object_name + '_'):
        return None, 0
    rest = stem[len(object_name) + 1:]
    head, _, tail = rest.rpartition('_')
    if not head or not tail.isdigit():
        return None, 0
    return head, int(tail)


def _object_processed_sources(existing_meta, object_name):
    """Realpaths of source anim files that already produced clips for this object."""
    sources = set()
    for entry in existing_meta.values():
        if str(entry.get('object_type', '')) != object_name:
            continue
        src = _normalized_source_fbx_path(entry)
        if src:
            sources.add(src)
    return sources


def _object_action_start_counts(existing_meta, object_name):
    """Highest existing clip index per action, so new clips number above retained ones."""
    action_start_counts = {}
    for motion_name, entry in existing_meta.items():
        if str(entry.get('object_type', '')) != object_name:
            continue
        action, index = _parse_action_and_index(object_name, motion_name)
        if action is None:
            continue
        action_start_counts[action] = max(action_start_counts.get(action, 0), index)
    return action_start_counts


def list_object_source_files(object_type, raw_data_dir=None):
    """Source anim files create_data_samples would consider for an object (post name-filter).

    Mirrors the enumeration in _prepare_object_outputs (same extensions + should_skip_anim
    filter) so callers can detect newly added animations without loading any geometry."""
    fbxs_dir = pjoin(get_raw_data_dir(raw_data_dir), object_type)
    if not os.path.isdir(fbxs_dir):
        return []
    anim_files = sorted(
        pjoin(fbxs_dir, f) for f in os.listdir(fbxs_dir)
        if f.lower().endswith(('.fbx', '.glb', '.gltf'))
    )
    # Mirror _prepare_object_outputs: a dedicated T-pose/rest reference file is consumed
    # as the encoding base (find_tpose_reference_path removes it from anim_files in place)
    # and never produces a clip, so it never lands in motion_metadata.json. Drop it here
    # too — otherwise every object carrying a *-TPOSE.fbx would perpetually report one
    # unprocessed source file and get needlessly reprocessed on every incremental run.
    find_tpose_reference_path(anim_files)
    return [f for f in anim_files if not should_skip_anim(f, object_type)]


def find_new_source_files(objects, dataset_dir=None, raw_data_dir=None):
    """Map each object with >=1 not-yet-processed source anim file to those new files.

    Objects whose every current source file already produced clips are omitted. Used by
    the incremental preprocessing path to decide which objects need any work at all
    (cheap: reads motion_metadata.json + lists raw dirs, no geometry loading)."""
    target_dataset_dir = dataset_dir or DEFAULT_DATASET_DIR
    existing_meta = _load_motion_metadata_raw(target_dataset_dir)
    result = {}
    for object_type in objects:
        processed = _object_processed_sources(existing_meta, object_type)
        new_files = [
            f for f in list_object_source_files(object_type, raw_data_dir)
            if os.path.realpath(f) not in processed
        ]
        if new_files:
            result[object_type] = new_files
    return result


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

    # Clip numbers are per-(object, action) segment indices, so number new clips
    # above the highest existing index *within each action group*. New clips then
    # collide neither with retained donor clips nor with older anim-dir outputs
    # still on disk during processing (matching-source clips are removed only
    # after processing succeeds, so a failed reprocess leaves the existing
    # dataset intact).
    action_start_counts = _object_action_start_counts(existing_meta, object_name)

    squared_positions_error = dict()
    _, _, _, object_cond, new_meta = process_object(
        object_name,
        0,
        0,
        23,
        squared_positions_error,
        save_dir=save_dir,
        fbxs_dir=anim_dir,
        face_joints=face_joints,
        t_pos_path=tpose_path,
        action_start_counts=action_start_counts,
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
        motion_labels = build_motion_labels(object_name, motion_name=motion_name)
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
                     motions_from_npys=None, target_cond_partial=None, update=False,
                     crop_enabled=True):
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
            motion_labels = build_motion_labels(object_name, motion_name=motion_name)
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
        # Rest-pose only: generate cond.npy without motion file processing
        object_cond, max_joints = _build_rest_pose_only_cond(
            object_name,
            tpose_path,
            face_joints,
            crop_enabled=crop_enabled,
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
        crop_enabled=crop_enabled,
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
