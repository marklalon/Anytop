"""Shared retarget helpers for the --retarget-top-k preprocessing flow.

Used by:
  tools/process_new_skeleton.py  -- builds coarse motions from similar donors
  sample/generate.py             -- cross-species reference motion retarget
"""
import glob
import os
from collections import Counter
from os.path import join as pjoin
from typing import List, Optional, Tuple

import numpy as np

from data_loaders.truebones.truebones_utils.param_utils import (
    BVHS_DIR,
    FOOT_CONTACT_VEL_THRESH,
    FPS,
    MAX_JOINTS,
    MOTION_DIR,
)
from data_loaders.truebones.truebones_utils.animation_utils import (
    find_translation_root,
)
# Skeleton-similarity primitives are shared with the motion-quality scorer.
# Imported under the existing private aliases so retarget call sites are unchanged.
from utils.skeleton_similarity import (
    normalize_match_name as _normalize_match_name,
    require_canonical_joint_names as _require_canonical_joint_names,
    strip_helper_names as _strip_helper_names,
    rank_species,
)


def _get_valid_translation_root_index(
    object_cond: Optional[dict],
    *,
    joint_count: Optional[int] = None,
) -> Optional[int]:
    if not isinstance(object_cond, dict):
        return None
    try:
        candidate = int(object_cond.get('translation_root_index'))
    except (TypeError, ValueError):
        return None
    if candidate < 0 or (joint_count is not None and candidate >= joint_count):
        return None
    return candidate


def _infer_donor_consensus_effective_root_index(
    donor_npys: List[str],
    donor_cond: dict,
) -> Optional[int]:
    from data_loaders.truebones.truebones_utils.features import (
        infer_translation_root_index_from_features,
    )

    parents = np.asarray(donor_cond['parents'], dtype=np.int32)
    offsets = np.asarray(donor_cond['offsets'], dtype=np.float64)
    counts: Counter[int] = Counter()

    for donor_npy_path in donor_npys:
        try:
            donor_features = np.load(donor_npy_path).astype(np.float32)
            counts[int(infer_translation_root_index_from_features(donor_features, parents, offsets))] += 1
        except Exception:
            continue

    if not counts:
        return None

    return int(counts.most_common(1)[0][0])


def infer_object_consensus_effective_root_index(
    motions_dir: str,
    object_type: str,
    object_cond: dict,
    *,
    max_files: int = 32,
) -> Optional[int]:
    stored_index = _get_valid_translation_root_index(object_cond)
    if stored_index is not None:
        return stored_index

    if not motions_dir or not os.path.isdir(motions_dir):
        return None

    object_npys = sorted(glob.glob(pjoin(motions_dir, f"{object_type}_*.npy")))[:max_files]
    if not object_npys:
        return None

    return _infer_donor_consensus_effective_root_index(object_npys, object_cond)


# ---------------------------------------------------------------------------
# Core retarget helper (shared between pipeline and generate.py wrapper)
# ---------------------------------------------------------------------------


def build_tpose_aligned_target_animation(retarget_result: dict, target_tp):
    """Convert a retarget result into the Animation form expected by get_motion.

    ``retarget_world_space_np`` works in target rest-composed world space so it
    can preserve target rest roll while transferring relative twist. For mapped
    joints (and gap joints on the path to a mapped descendant), the local
    rotation is recovered directly from the parent-relative world rotation. Pure
    unmapped side-branch joints stay at local identity so large FBX rest
    quaternions (e.g. dragon clavicles) do not leak into motion channels.
    Non-root local positions are reconstructed from the pose-location channel
    when present; otherwise the parent-relative world offset is taken directly.
    """
    from motion_lib.Animation import Animation
    from motion_lib.Quaternions import Quaternions
    from utils.rotation_numpy import (
        quat_conjugate_wxyz_np,
        quat_multiply_wxyz_np,
        quat_rotate_wxyz_np,
    )

    target_world_positions = np.asarray(
        retarget_result['target_world_positions'],
        dtype=np.float64,
    )
    target_world_rotations = np.asarray(
        retarget_result['target_world_rotations'],
        dtype=np.float64,
    )
    target_bone_translations = retarget_result.get('bone_translations')
    if target_bone_translations is not None:
        target_bone_translations = np.asarray(target_bone_translations, dtype=np.float64)
    target_offsets = np.asarray(target_tp.offsets, dtype=np.float64)
    target_parents = np.asarray(target_tp.tpos_anim.parents, dtype=np.int32)
    src_to_tgt = np.asarray(retarget_result['src_to_tgt'], dtype=np.int32)

    frame_count, joint_count = target_world_positions.shape[:2]
    target_rest_rotations = np.asarray(target_tp.tpos_rots[0], dtype=np.float64)
    identity_orients = np.zeros((joint_count, 4), dtype=np.float64)
    identity_orients[:, 0] = 1.0
    orient_quats = Quaternions(identity_orients)
    mapped_target_mask = np.zeros(joint_count, dtype=bool)
    for target_idx in src_to_tgt:
        if int(target_idx) >= 0:
            mapped_target_mask[int(target_idx)] = True

    has_mapped_descendant = np.zeros(joint_count, dtype=bool)
    for joint_idx in range(joint_count - 1, -1, -1):
        parent_idx = int(target_parents[joint_idx])
        if parent_idx >= 0 and (mapped_target_mask[joint_idx] or has_mapped_descendant[joint_idx]):
            has_mapped_descendant[parent_idx] = True

    local_rotations = np.zeros((frame_count, joint_count, 4), dtype=np.float64)
    local_positions = np.zeros_like(target_world_positions)
    local_rotations[:, :, 0] = 1.0

    for joint_idx in range(joint_count):
        parent_idx = int(target_parents[joint_idx])
        if parent_idx < 0:
            local_rotations[:, joint_idx] = target_world_rotations[:, joint_idx]
            local_positions[:, joint_idx] = target_world_positions[:, joint_idx]
            continue

        if mapped_target_mask[joint_idx] or has_mapped_descendant[joint_idx]:
            local_rotations[:, joint_idx] = quat_multiply_wxyz_np(
                quat_conjugate_wxyz_np(target_world_rotations[:, parent_idx]),
                target_world_rotations[:, joint_idx],
            )

        if target_bone_translations is not None:
            local_positions[:, joint_idx] = target_offsets[joint_idx] + quat_rotate_wxyz_np(
                np.repeat(target_rest_rotations[joint_idx:joint_idx + 1], frame_count, axis=0),
                target_bone_translations[:, joint_idx],
            )
        else:
            local_positions[:, joint_idx] = quat_rotate_wxyz_np(
                quat_conjugate_wxyz_np(target_world_rotations[:, parent_idx]),
                target_world_positions[:, joint_idx] - target_world_positions[:, parent_idx],
            )

    rotation_quats = Quaternions(local_rotations)

    return Animation(
        rotation_quats,
        local_positions.astype(np.float64),
        orient_quats,
        target_offsets,
        target_parents,
    )

def bake_foot_floor_offset(anim, foot_indices, up_axis: int = 1):
    """Lower the whole skeleton so foot-contact joints rest on y=0.

    Applied to the *reconstructed* target animation (after
    :func:`build_tpose_aligned_target_animation`), not to the retarget's
    world-space positions: cross-species retarget rebuilds the body from
    transferred rotations and a mostly-suppressed pose-location channel, so the
    rebuilt foot height differs from the world-space positions the retarget
    fitted. Measuring the floor here — on the geometry that is actually encoded
    and exported — is the only place the contact really lands at y=0.

    For each foot-contact joint, compute its minimum height across the entire
    clip, then take the median of those per-joint minimums. A single constant
    offset (that median) is subtracted from the hierarchy root's local translation. The root has no
    parent, so this is an exact world-space vertical shift of every joint; the
    encoded ``root_height`` feature carries it identically regardless of which
    ancestor it is baked into. Skeletons with no detected foot contact (snakes,
    fish, …) pass an empty ``foot_indices`` and are left untouched.

    Args:
        anim: target ``Animation`` (modified in place and returned).
        foot_indices: target-skeleton joint indices of foot-contact joints.
        up_axis: world axis treated as height (default 1 = Y).

    Returns:
        The same ``anim`` instance.
    """
    from motion_lib.Animation import positions_global

    if foot_indices is None or len(foot_indices) == 0:
        return anim
    joint_count = int(anim.positions.shape[1])
    foot_idx = np.asarray(foot_indices, dtype=np.int64).reshape(-1)
    foot_idx = foot_idx[(foot_idx >= 0) & (foot_idx < joint_count)]
    if foot_idx.size == 0:
        return anim

    global_positions = positions_global(anim)
    # For each foot contact joint, compute its minimum height across all frames,
    # then use the median of those per-joint minimums for floor alignment.
    per_joint_min = np.min(global_positions[:, foot_idx, up_axis], axis=0)
    floor_height = float(np.median(per_joint_min))
    # Ignore sub-centimeter drift from FK reconstruction to preserve self-retarget accuracy. 
    if abs(floor_height) <= 1e-2:
        return anim

    parents = np.asarray(anim.parents)
    root_candidates = np.flatnonzero(parents < 0)
    if root_candidates.size == 0:
        return anim
    anim.positions[:, int(root_candidates[0]), up_axis] -= floor_height
    return anim


def retarget_features_npy_to_target(
    source_features: np.ndarray,
    source_cond: dict,
    source_object_type: str,
    target_tp,
    target_object_type: str,
    max_joints: int,
    source_tp=None,
    target_cond: Optional[dict] = None,
    source_effective_root_index_override: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Retarget source skeleton's motion features to target skeleton's space.

    Args:
        source_features:    (F, J_src, 13) motion feature array, already loaded.
        source_cond:        Donor's cond.npy entry (parents / offsets /
                            orientation_reference_fbx_path, etc.).
        source_object_type: Donor object-type name (for get_common_features_from_T_pose).
        target_tp:          Pre-loaded TPoseFeatures for the target skeleton.
                            Pass the same object for every clip of a given target
                            to avoid repeated FBX parsing.
        target_object_type: Target object-type name (passed to get_motion).
        max_joints:         Maximum joint count for feature padding.
        source_tp:          Optional pre-loaded TPoseFeatures for the source donor.
                            If None, loaded lazily from
                            source_cond['orientation_reference_fbx_path'].
        target_cond:        Optional target cond entry carrying semantic
                    ``canonical_joint_names`` for name matching.

    Returns:
        (F, J_tgt, 13) retargeted feature array, or None if the retarget failed.
    """
    from utils.retarget import retarget_world_space_np
    from utils.exporter import animation_to_exporter_inputs
    from utils.roundtrip_common import build_skeleton
    from data_loaders.truebones.truebones_utils.features import (
        get_common_features_from_T_pose,
        get_motion,
        recover_animation_from_motion_np,
    )

    def _resolve_match_names(raw_names, object_cond=None, joint_count=None):
        if object_cond is None:
            raise ValueError("Retarget matching requires object_cond with canonical_joint_names")
        resolved_count = len(raw_names) if joint_count is None else int(joint_count)
        canonical_joint_names = _require_canonical_joint_names(
            object_cond,
            object_type_hint=str(object_cond.get('object_type') or '<unknown>'),
            joint_count=resolved_count,
        )
        return list(canonical_joint_names[:resolved_count])

    # Retarget against the donor's original skeleton only. Leaf helpers are a
    # training-time augmentation whose count varies with the joint budget, so
    # carrying them into retarget can desynchronize motion features from the
    # runtime T-pose skeleton.
    source_features = np.asarray(source_features, dtype=np.float32)
    source_joint_count = int(source_cond.get('original_joint_count') or source_features.shape[1])
    if source_joint_count <= 0 or source_joint_count > source_features.shape[1]:
        source_joint_count = int(source_features.shape[1])
    if source_joint_count < source_features.shape[1]:
        source_features = source_features[:, :source_joint_count, :]

    # 1. Load source T-pose metadata (once per donor via source_tp)
    if source_tp is None:
        src_tpose_fbx = source_cond.get('orientation_reference_fbx_path')
        if not src_tpose_fbx or not os.path.isfile(src_tpose_fbx):
            print(f"  [WARN] source T-pose FBX not found: {src_tpose_fbx!r}")
            return None
        source_tp = get_common_features_from_T_pose(
            src_tpose_fbx,
            source_object_type,
            augment_leaf_rotation_helpers=False,
            max_joints=max_joints,
        )
    elif len(source_tp.names) != source_joint_count:
        src_tpose_fbx = source_cond.get('orientation_reference_fbx_path')
        if not src_tpose_fbx or not os.path.isfile(src_tpose_fbx):
            print(f"  [WARN] source T-pose FBX not found: {src_tpose_fbx!r}")
            return None
        source_tp = get_common_features_from_T_pose(
            src_tpose_fbx,
            source_object_type,
            augment_leaf_rotation_helpers=False,
            max_joints=max_joints,
        )

    if len(source_tp.names) != source_joint_count:
        raise ValueError(
            f"Retarget source joint count {source_joint_count} does not match "
            f"source T-pose joint count {len(source_tp.names)}"
        )

    src_parents = np.asarray(source_tp.tpos_anim.parents, dtype=np.int32)
    src_offsets = np.asarray(source_tp.offsets, dtype=np.float32)
    target_effective_root_index = _get_valid_translation_root_index(
        target_cond,
        joint_count=len(target_tp.names),
    )

    # 2. Decode source features → Animation
    src_anim, _has_pos = recover_animation_from_motion_np(
        source_features,
        src_parents,
        src_offsets,
        translation_root_index=None,
        allow_infer=True,
    )
    if source_effective_root_index_override is None:
        source_effective_root_index = int(find_translation_root(src_anim))
    else:
        source_effective_root_index = int(source_effective_root_index_override)

    # 3. Build source skeleton
    src_skeleton = build_skeleton(
        source_tp.names,
        src_offsets,
        src_parents,
        rest_rotations=np.asarray(source_tp.tpos_rots[0], dtype=np.float32),
    )

    # 4. Source Animation → exporter inputs
    src_jr, src_rt, src_rr, src_bt = animation_to_exporter_inputs(src_anim, src_skeleton)

    # 5. World-space retarget
    retarget_result = retarget_world_space_np(
        src_parents=src_parents,
        src_rest_offsets=src_offsets.astype(np.float64),
        src_rest_rotations=np.asarray(source_tp.tpos_rots[0], dtype=np.float64),
        tgt_parents=np.asarray(target_tp.tpos_anim.parents, dtype=np.int32),
        tgt_rest_offsets=np.asarray(target_tp.offsets, dtype=np.float64),
        tgt_rest_rotations=np.asarray(target_tp.tpos_rots[0], dtype=np.float64),
        src_joint_rotations=src_jr.numpy().astype(np.float64),
        src_root_translation=src_rt.numpy().astype(np.float64),
        src_root_rotation=src_rr.numpy().astype(np.float64),
        src_effective_root_index=source_effective_root_index,
        tgt_effective_root_index=target_effective_root_index,
        src_bone_translations=src_bt.numpy().astype(np.float64) if src_bt is not None else None,
        src_match_names=_resolve_match_names(source_tp.names, source_cond, source_joint_count),
        tgt_match_names=_resolve_match_names(target_tp.names, target_cond),
        coordinate_search=False,
        verbose=False,
    )

    # 6. Build the target Animation in the feature-space local basis. The source
    # NPY path retargets decoded feature rotations, so target_world_rotations are
    # already the world rotations of that T-pose-relative representation.
    tgt_anim = build_tpose_aligned_target_animation(retarget_result, target_tp)

    # 6b. Drop the reconstructed skeleton onto the floor: lower it so the lowest
    # foot-contact joint over the whole clip rests at y=0. Must run on the
    # rebuilt geometry (the body is reconstructed from rotations, not the
    # retarget's world positions), and before re-encoding so the offset rides
    # in the encoded root height. Footless skeletons are left untouched.
    tgt_anim = bake_foot_floor_offset(tgt_anim, getattr(target_tp, 'foot_indices', None))

    # 7. Re-encode target Animation → motion features
    squared_positions_error = {}
    target_features, *_ = get_motion(
        tgt_anim,
        FOOT_CONTACT_VEL_THRESH,
        target_object_type,
        max_joints,
        np.asarray(target_tp.offsets, dtype=np.float64),
        target_tp.foot_indices,
        target_tp.tpos_rots,
        squared_positions_error,
        scale_factor=float(target_tp.scale_factor),
        orientation_quat=target_tp.orientation_quat,
        helper_metadata=target_tp.helper_metadata,
        animation_input_is_tpose_aligned=True,
    )

    if target_features is None:
        return None
    return np.asarray(target_features, dtype=np.float32)


def retarget_animation_file_to_target(
    source_motion_path: str,
    target_tp,
    target_object_type: str,
    max_joints: int,
    target_cond: dict,
    *,
    slice_inds=None,
) -> Optional[np.ndarray]:
    """Retarget a raw animation file (FBX/GLB/GLTF) onto the target skeleton.

    Unlike :func:`retarget_features_npy_to_target`, this path needs **no source
    cond entry**. The full source skeleton (topology, rest offsets, joint names)
    is read straight from the file's bind pose via ``FBX.load`` without requiring
    the source object_type to be present in the training cond.

    Facing is canonicalized exactly like the in-cond path — by a per-skeleton
    ``orientation_quat`` that rotates the skeleton to the dataset's +Z reference —
    only the quat is computed on the fly from the file's bind pose (via name-based
    face/forward-joint detection) instead of being read from cond. This is *not*
    done with the retarget's rigid ``coordinate_search``: that aligns the source
    bind pose to the target's reference pose geometrically, which is unreliable because
    different authored reference files can differ in shape — even for an
    identical skeleton it can fail to find the 90° yaw and leave the motion in its
    native (OOD) facing. ``orientation_quat`` looks at the head/face direction
    instead, which is invariant to leg configuration, so it canonicalizes robustly
    and ``coordinate_search`` is left off (source and target are both already +Z).

    The source's canonical match names are produced with the same
    ``build_semantic_metadata`` canonicalization the dataset cond uses, so name
    matching against the target behaves identically whether or not the source is
    registered in cond.

    Args:
        source_motion_path: path to an .fbx/.glb/.gltf animation.
        target_tp:          pre-loaded target ``TPoseFeatures`` (helper-augmented).
        target_object_type: target object-type name (passed to get_motion).
        max_joints:         maximum joint count for feature padding.
        target_cond:        target cond entry (for ``canonical_joint_names``).
        slice_inds:         optional ``[start, end]`` frame slice on the source.

    Returns:
        (F, J_tgt, 13) retargeted feature array, or None if the retarget failed.
    """
    from types import SimpleNamespace

    from motion_lib import FBX
    from motion_lib.Animation import Animation, positions_global, offsets_from_positions
    from motion_lib.Quaternions import Quaternions
    from data_loaders.truebones.truebones_utils.features import (
        get_motion,
        get_common_features_from_T_pose,
        calculate_root_quat,
        process_anim,
    )
    from data_loaders.truebones.truebones_utils.param_utils import FOOT_CONTACT_VEL_THRESH
    from data_loaders.truebones.truebones_utils.face_orientation import (
        resolve_face_joints,
        resolve_forward_reference_joints,
    )
    from data_loaders.truebones.truebones_utils.animation_utils import (
        get_average_axial_bone_length,
        get_rest_body_max_span,
        compute_scale_factor,
        build_leaf_rotation_helper_metadata,
    )
    from data_loaders.truebones.truebones_utils.physics_joint_annotation import (
        build_semantic_metadata,
        detect_joint_side,
        infer_contact_joints,
    )

    # 1. Load the raw animation. FBX.load returns per-joint total local rotations
    #    plus the armature's local rest rotations in ``orients``. Keeping that
    #    rest basis separate prevents bind-pose bone roll from being treated as
    #    animated twist by the cond-free retarget path.
    raw_anim, src_names, _fps = FBX.load(source_motion_path)
    if slice_inds:
        raw_anim = raw_anim[slice_inds[0]:slice_inds[1]]

    src_parents = np.asarray(raw_anim.parents, dtype=np.int32)
    src_offsets = np.asarray(raw_anim.offsets, dtype=np.float64)
    src_joint_count = len(src_parents)
    src_rest_rotations = np.asarray(raw_anim.orients.qs, dtype=np.float64)

    # 2. Source canonical match names — same canonicalization the dataset cond
    #    uses, so matching is consistent with the in-cond path.
    src_match_names = list(
        build_semantic_metadata(src_names, src_parents, src_offsets)['canonical_joint_names']
    )
    # 3. Source canonical orientation. Compute the +Z-facing quat from the bind
    #    pose (FK of the rest offsets) using name-based face/forward detection — no
    #    registered source object_type is needed (a neutral hint falls back to the
    #    name heuristics, which is what the dataset uses for the same skeleton).
    bind_anim = Animation(
        Quaternions(src_rest_rotations[None].copy()),
        src_offsets[None].copy(),
        Quaternions(src_rest_rotations.copy()),
        src_offsets.copy(),
        src_parents.copy(),
    )
    bind_positions = positions_global(bind_anim)  # (1, J, 3)
    _SRC_FACE_HINT = '__retarget_source_from_file__'
    src_face_joints = resolve_face_joints(_SRC_FACE_HINT, src_names, src_parents, None)
    src_forward_joint, src_forward_base_joint = resolve_forward_reference_joints(
        src_names, src_parents, object_type=_SRC_FACE_HINT,
    )
    src_orientation_quat = np.asarray(
        calculate_root_quat(
            bind_positions,
            _SRC_FACE_HINT,
            face_joint_indx=src_face_joints,
            forward_joint_index=src_forward_joint,
            forward_base_joint_index=src_forward_base_joint,
        )[0].qs,
        dtype=np.float64,
    ).reshape(-1)

    def _retarget_encoded_source_features(
        source_features: np.ndarray,
        source_cond: dict,
        source_object_type: str,
        source_tp,
        source_effective_root_index: int | None,
    ) -> Optional[np.ndarray]:
        return retarget_features_npy_to_target(
            np.asarray(source_features, dtype=np.float32),
            source_cond,
            source_object_type,
            target_tp,
            target_object_type,
            max_joints,
            source_tp=source_tp,
            target_cond=target_cond,
            source_effective_root_index_override=source_effective_root_index,
        )

    def _reindex_raw_animation_subset(anim, names, keep_indices):
        old_to_new = {old: new for new, old in enumerate(keep_indices)}
        new_parents = np.array(
            [
                old_to_new[int(anim.parents[i])] if int(anim.parents[i]) >= 0 else -1
                for i in keep_indices
            ],
            dtype=np.int32,
        )
        return Animation(
            Quaternions(anim.rotations.qs[:, keep_indices].copy()),
            anim.positions[:, keep_indices].copy(),
            Quaternions(anim.orients.qs[keep_indices].copy()),
            anim.offsets[keep_indices].copy(),
            new_parents,
        ), [names[i] for i in keep_indices]

    def _align_raw_to_expected_original_skeleton(anim, names, expected_names, expected_parents):
        if list(names) == list(expected_names) and np.array_equal(
            np.asarray(anim.parents, dtype=np.int32),
            expected_parents,
        ):
            return anim, list(names)

        expected_name_set = set(expected_names)
        parents = np.asarray(anim.parents, dtype=np.int32)
        has_children = np.zeros(len(names), dtype=bool)
        has_children[parents[parents >= 0]] = True
        unexpected_leaves = [
            idx for idx, name in enumerate(names)
            if name not in expected_name_set and not has_children[idx]
        ]
        if not unexpected_leaves:
            return None

        drop_set = set(unexpected_leaves)
        keep_indices = [idx for idx in range(len(names)) if idx not in drop_set]
        keep_set = set(keep_indices)
        for idx in keep_indices:
            parent_idx = int(parents[idx])
            if parent_idx >= 0 and parent_idx not in keep_set:
                return None

        aligned_anim, aligned_names = _reindex_raw_animation_subset(anim, names, keep_indices)
        if list(aligned_names) == list(expected_names) and np.array_equal(
            np.asarray(aligned_anim.parents, dtype=np.int32),
            expected_parents,
        ):
            stripped_names = [names[idx] for idx in unexpected_leaves]
            print(
                f"[retarget] stripped {len(stripped_names)} terminal bone(s) from raw reference "
                f"{os.path.basename(source_motion_path)!r}: {stripped_names[:10]}"
                f"{'...' if len(stripped_names) > 10 else ''}"
            )
            return aligned_anim, aligned_names

        return None

    target_source_basis_available = False
    target_source_tp = None
    target_aligned_raw_anim = None
    target_aligned_names = None
    target_tpose_path = target_cond.get('orientation_reference_fbx_path')
    if (
        target_tpose_path
        and os.path.isfile(target_tpose_path)
    ):
        target_source_tp = get_common_features_from_T_pose(
            target_tpose_path,
            target_object_type,
            augment_leaf_rotation_helpers=False,
            max_joints=max_joints,
        )
        expected_names = list(target_source_tp.names)
        expected_parents = np.asarray(target_source_tp.tpos_anim.parents, dtype=np.int32)
        aligned = _align_raw_to_expected_original_skeleton(
            raw_anim,
            src_names,
            expected_names,
            expected_parents,
        )
        if aligned is not None:
            target_aligned_raw_anim, target_aligned_names = aligned

        target_orientation_quat = np.asarray(
            getattr(target_tp.orientation_quat, 'qs', target_tp.orientation_quat),
            dtype=np.float64,
        ).reshape(-1)
        target_orientation_quat = target_orientation_quat / max(
            float(np.linalg.norm(target_orientation_quat)),
            1e-12,
        )
        source_orientation_unit = src_orientation_quat / max(
            float(np.linalg.norm(src_orientation_quat)),
            1e-12,
        )
        target_source_basis_available = (
            target_aligned_raw_anim is not None
            and abs(float(np.dot(source_orientation_unit, target_orientation_quat))) > 1.0 - 1e-4
        )

    if target_source_basis_available:
        squared_positions_error = {}
        source_features, *_unused, source_effective_root_index, _source_root_xz = get_motion(
            source_motion_path,
            FOOT_CONTACT_VEL_THRESH,
            target_object_type,
            max_joints,
            np.asarray(target_tp.offsets, dtype=np.float64),
            target_tp.foot_indices,
            target_tp.tpos_rots,
            squared_positions_error,
            scale_factor=float(target_tp.scale_factor),
            orientation_quat=target_tp.orientation_quat,
            slice_inds=None,
            preloaded=(target_aligned_raw_anim, target_aligned_names),
            helper_metadata=target_tp.helper_metadata,
            animation_input_is_tpose_aligned=False,
        )
        if source_features is None:
            return None
        target_helper_source_cond = dict(target_cond)
        target_helper_source_cond['original_joint_count'] = int(
            target_cond.get('original_joint_count') or len(np.asarray(target_cond['parents']))
        )
        return _retarget_encoded_source_features(
            source_features,
            target_helper_source_cond,
            target_object_type,
            target_tp,
            source_effective_root_index,
        )

    # 4. Build source rest-pose metadata from this file's own bind pose. This keeps
    #    the raw-file path cond-free while still using the same feature convention
    #    as the dataset preprocessing path before entering the shared retargeter.
    side_labels = []
    for name in src_names:
        detected = detect_joint_side(name)
        side_labels.append(detected if detected in ('left', 'right') else 'center')
    axial_avg_len = get_average_axial_bone_length(src_offsets, src_parents, side_labels)
    body_max_span = get_rest_body_max_span(src_offsets, src_parents)
    source_scale_factor = compute_scale_factor(axial_avg_len, body_max_span=body_max_span)
    source_tpose_anim, _source_root_xz_center, source_scale_factor = process_anim(
        bind_anim,
        _SRC_FACE_HINT,
        Quaternions(src_orientation_quat[None]),
        scale_factor=source_scale_factor,
    )
    source_tpose_positions = positions_global(source_tpose_anim)
    source_offsets = offsets_from_positions(source_tpose_positions[0], source_tpose_anim.parents)
    source_foot_indices, source_contact_source = infer_contact_joints(
        src_names,
        source_tpose_anim.parents,
        source_tpose_positions[0],
    )
    source_helper_metadata = build_leaf_rotation_helper_metadata(
        src_names,
        source_tpose_anim.parents,
        offsets=source_offsets,
        max_joints=src_joint_count,
    )
    source_tp = SimpleNamespace(
        scale_factor=source_scale_factor,
        offsets=source_offsets,
        foot_indices=source_foot_indices,
        tpos_rots=source_tpose_anim.rotations,
        names=list(src_names),
        tpos_anim=source_tpose_anim,
        face_joints=src_face_joints,
        orientation_quat=Quaternions(src_orientation_quat[None]),
        forward_joint_index=src_forward_joint,
        forward_base_joint_index=src_forward_base_joint,
        contact_joint_source=source_contact_source,
        axial_avg_len=axial_avg_len,
        helper_metadata=source_helper_metadata,
    )
    source_cond = {
        'object_type': _SRC_FACE_HINT,
        'parents': np.asarray(source_tpose_anim.parents, dtype=np.int32),
        'offsets': np.asarray(source_offsets, dtype=np.float32),
        'original_joint_count': src_joint_count,
        'canonical_joint_names': src_match_names,
        'orientation_quat': src_orientation_quat.astype(np.float32),
        'scale_factor': float(source_scale_factor),
    }

    squared_positions_error = {}
    source_features, *_unused, source_effective_root_index, _source_root_xz = get_motion(
        source_motion_path,
        FOOT_CONTACT_VEL_THRESH,
        _SRC_FACE_HINT,
        max_joints,
        np.asarray(source_offsets, dtype=np.float64),
        source_foot_indices,
        source_tpose_anim.rotations,
        squared_positions_error,
        scale_factor=float(source_scale_factor),
        orientation_quat=Quaternions(src_orientation_quat[None]),
        slice_inds=None,
        preloaded=(raw_anim, src_names),
        helper_metadata=source_helper_metadata,
        animation_input_is_tpose_aligned=False,
    )
    if source_features is None:
        return None

    return _retarget_encoded_source_features(
        source_features,
        source_cond,
        _SRC_FACE_HINT,
        source_tp,
        source_effective_root_index,
    )


# ---------------------------------------------------------------------------
# Donor ranking
# ---------------------------------------------------------------------------

def rank_donors(
    target_cond: dict,
    training_cond_dict: dict,
    target_object_type: str,
) -> List[Tuple[str, float]]:
    """Rank all training skeletons by similarity to the target.

    Uses the shared ``rank_species`` blend (Jaccard primary + joint-name
    embedding secondary + topology descriptor + graded lineage-tag discount). A
    freshly built target skeleton usually lacks ``joints_names_embs`` (and, if
    its object_type is unregistered, lineage tags); those terms are dropped
    automatically and the blend reduces to Jaccard + topology.

    Returns list of (donor_name, score) sorted by descending score, where
    ``score = 100 / (1 + combined_distance)`` is a monotonic higher-is-better
    transform of the combined distance (ordering matches closest-first).
    """
    ranked = rank_species(
        target_cond,
        training_cond_dict,
        query_hint=target_object_type,
        top_k=None,
    )
    return [(r.name, 100.0 / (1.0 + r.combined_distance)) for r in ranked]


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def auto_retarget_pipeline(
    target_object_type: str,
    target_tpose_path: str,
    save_dir: str,
    top_k: int,
    training_cond_path: str,
    face_joints_names=None,
    donor_skeletons_override=None,
    max_joints: int = MAX_JOINTS,
    fps: float = FPS,
) -> dict:
    """Auto-retarget motions from top-k similar training donors onto the target.

    Steps:
      1. Load training cond_dict from training_cond_path.
      2. Build target_cond + target_tp via _build_tpose_cond.
      3. Select donors (override list or auto top-k ranking).
      4. For each donor: retarget all motion .npy files, save .npy + .bvh.
      5. Return summary dict.

    Returns:
        dict with keys:
          'target_cond'      -- the built target cond entry (no mean/std yet)
          'retargeted_npys'  -- list of absolute paths to written .npy files
          'donors_used'      -- list of (name, score, n_success) tuples
    """
    from data_loaders.truebones.truebones_utils.dataset_pipeline import build_tpose_cond
    from data_loaders.truebones.truebones_utils.features import get_common_features_from_T_pose
    from data_loaders.truebones.truebones_utils.motion_process import (
        recover_bvh_export_animation_from_motion_np,
    )
    from data_loaders.truebones.truebones_utils.animation_utils import (
        needs_bvh_position_channels,
        reorder_animation_to_dfs,
    )
    from motion_lib import BVH

    # 1. Load training cond_dict
    if not os.path.isfile(training_cond_path):
        raise FileNotFoundError(
            f"Training cond.npy not found: {training_cond_path!r}. "
            "Use --training-cond-path to point to the processed dataset's cond.npy."
        )
    training_cond_dict = np.load(training_cond_path, allow_pickle=True).item()
    training_motions_dir = pjoin(os.path.dirname(os.path.abspath(training_cond_path)), MOTION_DIR)

    # 2. Build target_cond + target_tp
    print(f"\n[auto_retarget] Building target skeleton: {target_object_type}")
    (
        target_cond,
        target_tp,
        _t_pos_motion,
        target_parents,
        _sem_meta,
        _scale,
        _sq_err,
        max_joints_tgt,
    ) = build_tpose_cond(target_object_type, target_tpose_path, face_joints_names)
    max_joints = max(max_joints, max_joints_tgt)
    target_effective_root_index = infer_object_consensus_effective_root_index(
        training_motions_dir,
        target_object_type,
        target_cond,
    )
    if target_effective_root_index is not None:
        target_cond['translation_root_index'] = int(target_effective_root_index)

    n_joints = int(target_cond.get('original_joint_count') or len(target_parents))
    n_chains = len(target_cond.get('kinematic_chains', []))
    print(
        f"[auto_retarget] Target: {target_object_type} "
        f"({n_joints} joints"
    )
    if target_effective_root_index is not None:
        print(
            f"[auto_retarget] {target_object_type}: using target effective root "
            f"{target_effective_root_index} for retarget mapping"
        )

    # 3. Select donors
    if donor_skeletons_override is not None:
        missing = [d for d in donor_skeletons_override if d not in training_cond_dict]
        if missing:
            available = sorted(training_cond_dict.keys())
            raise ValueError(
                f"--donor-skeletons specified unknown donors: {missing}\n"
                f"Available: {available}"
            )
        scored_all = rank_donors(target_cond, training_cond_dict, target_object_type)
        score_map = {name: score for name, score in scored_all}
        selected_donors = [(d, score_map.get(d, 0.0)) for d in donor_skeletons_override]
    else:
        scored_all = rank_donors(target_cond, training_cond_dict, target_object_type)
        selected_donors = scored_all[:top_k]

    print(f"[auto_retarget] Top-{len(selected_donors)} donors selected:")
    t_names = _strip_helper_names(
        _require_canonical_joint_names(
            target_cond,
            object_type_hint=target_object_type,
        )
    )
    for rank, (donor_name, score) in enumerate(selected_donors, 1):
        donor_cond = training_cond_dict[donor_name]
        d_names = _strip_helper_names(
            _require_canonical_joint_names(
                donor_cond,
                object_type_hint=donor_name,
            )
        )
        t_norm = {_normalize_match_name(n) for n in t_names}
        d_norm = {_normalize_match_name(n) for n in d_names}
        union = max(1, len(t_norm | d_norm))
        jaccard = len(t_norm & d_norm) / union
        d_joints = int(donor_cond.get('original_joint_count') or len(donor_cond['parents']))
        d_chains = len(donor_cond.get('kinematic_chains', []))
        print(
            f"  {rank}. {donor_name:<22} score={score:.1f}  "
            f"(jaccard={jaccard:.2f}, "
            f"Δjoints={abs(n_joints - d_joints)}, Δchains={abs(n_chains - d_chains)})"
        )

    # Prepare output dirs
    os.makedirs(pjoin(save_dir, MOTION_DIR), exist_ok=True)
    os.makedirs(pjoin(save_dir, BVHS_DIR), exist_ok=True)

    retargeted_npys: List[str] = []
    donors_used: List[Tuple[str, float, int]] = []

    # 4. For each donor, retarget all motion files
    for donor_name, donor_score in selected_donors:
        donor_cond = training_cond_dict[donor_name]
        donor_fbx = donor_cond.get('orientation_reference_fbx_path')

        if not donor_fbx or not os.path.isfile(donor_fbx):
            print(f"\n[auto_retarget] {donor_name}: T-pose FBX not found ({donor_fbx!r}), skipping donor")
            donors_used.append((donor_name, donor_score, 0))
            continue

        donor_npys = sorted(glob.glob(pjoin(training_motions_dir, f"{donor_name}_*.npy")))
        print(f"\n[auto_retarget] {donor_name}: {len(donor_npys)} motion files found, retargeting...")

        if not donor_npys:
            print(f"  [WARN] No motion files found for {donor_name} in {training_motions_dir}")
            donors_used.append((donor_name, donor_score, 0))
            continue

        # Pre-load source T-pose once per donor
        source_tp = get_common_features_from_T_pose(
            donor_fbx,
            donor_name,
            augment_leaf_rotation_helpers=False,
            max_joints=max_joints,
        )
        donor_effective_root_index = _infer_donor_consensus_effective_root_index(
            donor_npys,
            donor_cond,
        )
        if donor_effective_root_index is not None:
            print(
                f"[auto_retarget] {donor_name}: using donor effective root "
                f"{donor_effective_root_index} for retarget mapping"
            )

        n_success = 0
        for src_npy_path in donor_npys:
            src_base = os.path.splitext(os.path.basename(src_npy_path))[0]
            # Strip leading "<donor>_" prefix to get action token
            prefix = donor_name + '_'
            action_token = src_base[len(prefix):] if src_base.startswith(prefix) else src_base

            out_name = f"{target_object_type}_{donor_name}_{action_token}"
            out_npy = pjoin(save_dir, MOTION_DIR, f"{out_name}.npy")
            out_bvh = pjoin(save_dir, BVHS_DIR, f"{out_name}.bvh")

            try:
                src_features = np.load(src_npy_path).astype(np.float32)
                tgt_features = retarget_features_npy_to_target(
                    src_features,
                    donor_cond,
                    donor_name,
                    target_tp,
                    target_object_type,
                    max_joints,
                    source_tp=source_tp,
                    target_cond=target_cond,
                    source_effective_root_index_override=donor_effective_root_index,
                )
                if tgt_features is None:
                    print(f"  ✗ {src_base} (retarget returned None, skipped)")
                    continue

                np.save(out_npy, tgt_features)
                retargeted_npys.append(os.path.abspath(out_npy))
                n_success += 1
                print(f"  ✓ {src_base} → {out_name}.npy")

                # BVH for visual inspection
                try:
                    bvh_joint_names = list(
                        target_cond.get('canonical_bvh_joint_names')
                        or target_cond.get('joints_names', [])
                    )
                    out_anim, bvh_joint_names, has_animated_pos = recover_bvh_export_animation_from_motion_np(
                        tgt_features,
                        np.asarray(target_cond['parents'], dtype=np.int32),
                        np.asarray(target_cond['offsets'], dtype=np.float32),
                        bvh_joint_names,
                        allow_infer=True,
                        tpose_rest_rotations=target_tp.tpos_rots[0],
                    )
                    if out_anim is not None:
                        out_anim, bvh_joint_names = reorder_animation_to_dfs(out_anim, bvh_joint_names)
                        BVH.save(
                            out_bvh, out_anim, bvh_joint_names,
                            frametime=1.0 / fps,
                            positions=needs_bvh_position_channels(out_anim),
                        )
                except Exception as bvh_err:
                    print(f"    [WARN] BVH write failed: {bvh_err}")

            except Exception as e:
                print(f"  ✗ {src_base} (error: {e}, skipped)")

        print(f"[auto_retarget] {donor_name}: {n_success}/{len(donor_npys)} success")
        donors_used.append((donor_name, donor_score, n_success))

    total_success = sum(n for _, _, n in donors_used)
    print(
        f"\n[auto_retarget] Total: {total_success} motions retargeted across "
        f"{len(selected_donors)} donors → {pjoin(save_dir, MOTION_DIR)}/"
    )

    if total_success == 0:
        raise RuntimeError(
            f"All retargets failed across {len(selected_donors)} donors. "
            "Try --donor-skeletons to manually specify similar skeletons, "
            "or verify that training motions are accessible."
        )

    return {
        'target_cond': target_cond,
        'retargeted_npys': retargeted_npys,
        'donors_used': donors_used,
    }
