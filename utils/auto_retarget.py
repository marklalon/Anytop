"""Shared retarget helpers for the cross-species reference retarget flow.

Used by:
  sample/generate.py             -- cross-species reference motion retarget
  utils/retarget.py              -- CLI wrapper for retargeting files
"""
import os
from typing import Optional

import numpy as np

from data_loaders.truebones.truebones_utils.param_utils import (
    FOOT_CONTACT_VEL_THRESH,
)
from data_loaders.truebones.truebones_utils.animation_utils import (
    find_translation_root,
)
# Skeleton-similarity primitive shared with the motion-quality scorer.
# Imported under the existing private alias so retarget call sites are unchanged.
from utils.skeleton_similarity import (
    require_canonical_joint_names as _require_canonical_joint_names,
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


# ---------------------------------------------------------------------------
# Core retarget helper (shared between the retarget CLI and generate.py)
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
        source_cond:        Donor's cond.npy entry (parents / offsets / rest_pose
                            / orientation_quat / scale_factor, etc.).
        source_object_type: Donor object-type name.
        target_tp:          Pre-loaded TPoseFeatures for the target skeleton.
                            Pass the same object for every clip of a given target
                            to avoid rebuilding it from cond per clip.
        target_object_type: Target object-type name (passed to get_motion).
        max_joints:         Maximum joint count for feature padding.
        source_tp:          Optional pre-loaded TPoseFeatures for the source donor.
                            If None, reconstructed from ``source_cond`` via
                            tpose_features_from_cond (no T-pose mesh read).
        target_cond:        Optional target cond entry carrying semantic
                    ``canonical_joint_names`` for name matching.

    Returns:
        (F, J_tgt, 13) retargeted feature array, or None if the retarget failed.
    """
    from utils.retarget import retarget_world_space_np
    from utils.exporter import animation_to_exporter_inputs
    from utils.roundtrip_common import build_skeleton
    from data_loaders.truebones.truebones_utils.features import (
        tpose_features_from_cond,
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

    # Current own-rotation features use exactly the joints in the runtime
    # T-pose skeleton; no leaf helper joints are appended.
    source_features = np.asarray(source_features, dtype=np.float32)
    source_joint_count = int(source_features.shape[1])

    # 1. Source rest-pose metadata reconstructed from the donor cond entry — no
    #    original T-pose FBX/GLB mesh is read. The caller reuses one source_tp
    #    across a donor's clips; rebuild from cond when absent or stale.
    if source_tp is None or len(source_tp.names) != source_joint_count:
        source_tp = tpose_features_from_cond(source_cond, source_object_type)

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
        animation_input_is_tpose_aligned=True,
    )

    if target_features is None:
        return None
    return np.asarray(target_features, dtype=np.float32)


def canonical_match_names_from_raw_skeleton(
    joint_names,
    parents,
    offsets,
    *,
    species_name: Optional[str] = None,
) -> list[str]:
    """Return dataset-compatible canonical names for a raw source skeleton.

    This deliberately uses the cond refresh path instead of calling
    ``build_semantic_metadata`` alone: refresh also applies the duplicate-name
    disambiguation used by preprocessed targets (for example ``Tongue`` and
    ``Tongue02`` must not both collapse to ``Tongue``).
    """
    from data_loaders.truebones.truebones_utils.animation_utils import (
        refresh_joint_metadata_in_object_cond,
    )

    species_name = str(species_name or '').strip()
    source_name_cond = {
        # Empty when unknown: prefix inference no-ops on an empty species name.
        'object_type': species_name,
        'species_name': species_name,
        'joints_names': [str(name) for name in joint_names],
        'parents': np.asarray(parents, dtype=np.int32),
        'offsets': np.asarray(offsets, dtype=np.float64),
    }
    refresh_joint_metadata_in_object_cond(source_name_cond)
    return list(source_name_cond['canonical_joint_names'])


def retarget_animation_file_to_target(
    source_motion_path: str,
    target_tp,
    target_object_type: str,
    max_joints: int,
    target_cond: dict,
    *,
    slice_inds=None,
    source_object_type: Optional[str] = None,
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

    The source's canonical match names are produced through the same metadata
    refresh and duplicate-name disambiguation path as dataset cond. When known,
    ``source_object_type`` also enables the same skeleton-wide species-prefix
    stripping; an unregistered source may still supply this name without needing
    a source cond entry.

    Args:
        source_motion_path: path to an .fbx/.glb/.gltf animation.
        target_tp:          pre-loaded target ``TPoseFeatures`` (helper-augmented).
        target_object_type: target object-type name (passed to get_motion).
        max_joints:         maximum joint count for feature padding.
        target_cond:        target cond entry (for ``canonical_joint_names``).
        slice_inds:         optional ``[start, end]`` frame slice on the source.
        source_object_type: optional source species/object identifier used only
                            for canonical joint-name prefix normalization.

    Returns:
        (F, J_tgt, 13) retargeted feature array, or None if the retarget failed.
    """
    from types import SimpleNamespace

    from motion_lib import FBX
    from motion_lib.Animation import Animation, positions_global, offsets_from_positions
    from motion_lib.Quaternions import Quaternions
    from data_loaders.truebones.truebones_utils.features import (
        get_motion,
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
        get_scale_reference_extent,
        compute_scale_factor,
    )
    from data_loaders.truebones.truebones_utils.physics_joint_annotation import (
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
    src_rest_rotations = np.asarray(raw_anim.orients.qs, dtype=np.float64)
    _SRC_FACE_HINT = '__retarget_source_from_file__'

    # 2. Source canonical match names. Build a minimal cond-shaped record and
    #    run the exact refresh/assignment path used by dataset cond, including
    #    skeleton-wide species-prefix stripping and duplicate-name disambiguation.
    #    A raw source does not require a cond entry: callers can pass only its
    #    object/species identifier. Dataset source files also commonly live in a
    #    species-named parent directory, which is a safe fallback because the
    #    prefix inference still requires every source joint to share the token.
    source_species_hint = str(source_object_type or '').strip()
    if not source_species_hint:
        # Only use the parent directory when the path actually carries one --
        # a bare filename would otherwise resolve to the current working
        # directory. The all-joints prefix gate keeps this a safe hint either way.
        dir_name = os.path.dirname(source_motion_path)
        if dir_name:
            source_species_hint = os.path.basename(dir_name)
    src_match_names = canonical_match_names_from_raw_skeleton(
        src_names,
        src_parents,
        src_offsets,
        species_name=source_species_hint,
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
    src_face_joints = resolve_face_joints(
        _SRC_FACE_HINT, src_names, src_parents, None, rest_positions=bind_positions
    )
    src_forward_joint, src_forward_base_joint = resolve_forward_reference_joints(
        src_names, src_parents, object_type=_SRC_FACE_HINT, rest_positions=bind_positions,
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
    target_aligned_raw_anim = None
    target_aligned_names = None
    # The target's expected raw skeleton (names + topology) comes from the
    # cond-derived target_tp — no original T-pose FBX/GLB mesh is read. The
    # source skeleton/motion below is still the user-provided animation file.
    expected_names = list(target_tp.names)
    expected_parents = np.asarray(target_tp.tpos_anim.parents, dtype=np.int32)
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
            animation_input_is_tpose_aligned=False,
        )
        if source_features is None:
            return None
        return _retarget_encoded_source_features(
            source_features,
            target_cond,
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
    axial_avg_len = get_average_axial_bone_length(src_offsets, src_parents, side_labels, src_names)
    # Must stay the same call the dataset path makes in
    # get_common_features_from_rest_pose: the retargeter cancels a uniform
    # source scale (it renormalizes by mean_len_tgt / mean_len_src), but the
    # absolute thresholds applied while encoding the source in normalized space
    # -- ROOT_Y_MIN_HEIGHT, the root-XZ strip / loop-detection bands, the
    # foot-contact velocity and height thresholds -- do not, so a raw source
    # file must land in the same normalized space its dataset clips do.
    body_max_span = get_scale_reference_extent(bind_positions[0], src_parents, src_names)
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
    )
    source_cond = {
        'object_type': _SRC_FACE_HINT,
        'parents': np.asarray(source_tpose_anim.parents, dtype=np.int32),
        'offsets': np.asarray(source_offsets, dtype=np.float32),
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
