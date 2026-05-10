"""Compatibility proxy — delegates to animation_utils, features, dataset_pipeline.

This module exists solely to preserve backward compatibility for the 23 external
files that import from ``motion_process``. All symbols are re-exported from the
three functional sub-modules.
"""

from motion_lib import BVH, FBX, Animation, Quaternions
from motion_lib import animation_from_positions
from motion_lib.Animation import positions_global, rotations_global, offsets_from_positions, offsets_global
from data_loaders.truebones.truebones_utils.param_utils import (
    HML_REF_AXIAL_BONE_LENGTH,
    FOOT_CONTACT_HEIGHT_THRESH,
    DEFAULT_DATASET_DIR,
    MAX_JOINTS,
    MAX_PATH_LEN,
    MOTION_DIR,
    FOOT_CONTACT_VEL_THRESH,
    BVHS_DIR,
    OBJECT_SUBSETS_DICT,
    get_raw_data_dir,
    SNAKES,
    CHAIN_FORWARD_JOINTS,
    FLYING,
    FISH,
    VERTICAL_CLAMP_MIN_RATIO,
    VERTICAL_CLAMP_MAX_RATIO,
)

# ── animation_utils (animation processing & joint metadata) ─────────────
from .animation_utils import (
    ROOT_XZ_STRIP_THRESHOLD,
    LOOP_DETECTION_POS_THRESHOLD,
    LEAF_ROTATION_HELPER_SUFFIX,
    _EMITTED_MIRROR_SAFEGUARD_WARNINGS,
    # Joint name canonicalization
    _canonical_name_for_bvh,
    _build_joint_name_inspection_rows,
    _remove_token_counts,
    _joint_disambiguation_tokens,
    _display_disambiguation_tokens,
    _disambiguate_duplicate_canonical_names,
    _collect_joint_name_collision_groups,
    _write_joint_name_collision_report,
    _refresh_joint_metadata_in_object_cond,
    refresh_joint_metadata_in_cond_dict,
    _joint_name_embeddings_are_current,
    _attach_joint_name_embeddings_to_cond,
    # Animation transforms
    _detect_motion_loop,
    _find_translation_root,
    _find_descendant_transport_chain,
    _bake_descendant_y_into_translation_root,
    _get_reference_body_length,
    _compress_positive_excursion,
    _compress_negative_excursion,
    _clamp_vertical_trajectory,
    _coerce_root_xz_center,
    _get_translation_root_initial_xz,
    move_xz_to_origin,
    _xz_locomotion_extent,
    strip_translation_root_xz,
    _resolve_detected_translation_root_index,
    # BVH export
    needs_bvh_position_channels,
    reorder_animation_to_dfs,
    # Scaling
    _get_average_axial_bone_length,
    compute_scale_factor,
    scale,
    # Leaf rotation helpers
    _reference_clip_needs_local_position_rebuild,
    _leaf_rotation_helper_name,
    _dfs_leaf_joint_indices,
    _build_leaf_rotation_helper_metadata,
    _append_leaf_rotation_helpers_to_animation,
    _extend_semantic_metadata_with_leaf_helpers,
    # Mirror & neutralization
    _warn_mirror_disabled_subtrees,
    neutralize_animation_subtrees,
    # FK helpers
    _coerce_single_orientation_quat,
    compute_rots_from_tpos,
    _solve_local_positions_for_target_global,
)

# ── features (feature extraction & motion recovery) ─────────────────────
from .features import (
    # Contact & feature building
    get_contact_state,
    get_terminal_contact_state,
    get_foot_contact,
    get_6d_rep,
    get_rifke,
    get_motion_features,
    get_bvh_cont6d_params,
    process_anim,
    _compute_terminal_local_velocity,
    # Translation root resolution
    _coerce_translation_root_index,
    _translation_root_index_from_motion_metadata,
    _require_translation_root_index_from_motion_metadata,
    resolve_feature_translation_root_index,
    infer_translation_root_index_from_features,
    # Mirror
    _neutralize_mirror_disabled_subtrees,
    mirror_features_with_safeguards,
    # T-Pose & motion extraction
    get_common_features_from_T_pose,
    TPoseFeatures,
    _extract_motion_features_from_aligned_anims,
    get_hml_aligned_anim,
    get_motion,
    # Motion recovery
    recover_processed_animation_from_feature_animation,
    recover_root_quat_and_pos_np,
    recover_root_quat_and_pos,
    recover_from_bvh_ric_np,
    recover_from_bvh_rot_np,
    recover_animation_from_motion_np,
    recover_bvh_export_animation_from_motion_np,
)

# ── dataset_pipeline (dataset building & augmentation) ──────────────────
from .dataset_pipeline import (
    # Statistics & topology
    get_mean_std,
    create_topology_edge_relations,
    # Kinematic chains
    reverse_insort,
    parents2kinchains,
    recursion_kinchains,
    object_policy,
    # Augmentations
    remove_joints_augmentation,
    add_joint_augmentation,
    # Dataset pipeline
    _process_motion_file,
    _attach_orientation_reference_metadata,
    _build_motion_metadata_entry,
    _prepare_object_outputs,
    _write_object_outputs,
    _write_dataset_artifacts,
    _resolve_preprocessing_workers,
    _prepare_object_outputs_worker,
    process_object,
    create_data_samples,
    # Test / entry points
    process_single_object_type,
    process_skeleton,
)
