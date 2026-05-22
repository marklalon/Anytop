"""Compatibility proxy — delegates to animation_utils, features, dataset_pipeline.

This module exists solely to preserve backward compatibility for external
files that import from ``motion_process``. All symbols are re-exported from the
three functional sub-modules.
"""

from motion_lib.Animation import positions_global
from data_loaders.truebones.truebones_utils.param_utils import (
    FOOT_CONTACT_VEL_THRESH,
)

# ── animation_utils (animation processing & joint metadata) ─────────────
from .animation_utils import (
    ROOT_XZ_STRIP_THRESHOLD,
    # Joint name canonicalization
    canonical_name_for_bvh,
    collect_joint_name_collision_groups,
    refresh_joint_metadata_in_object_cond,
    write_joint_name_collision_report,
    refresh_joint_metadata_in_cond_dict,
    attach_joint_name_embeddings_to_cond,
    # Animation transforms
    find_translation_root,
    xz_locomotion_extent,
    move_xz_to_origin,
    # BVH export
    needs_bvh_position_channels,
    reorder_animation_to_dfs,
    # Leaf rotation helpers
    build_leaf_rotation_helper_metadata,
    append_leaf_rotation_helpers_to_animation,
    resolve_mirrored_export_skeleton_metadata,
)

# ── features (feature extraction & motion recovery) ─────────────────────
from .features import (
    get_6d_rep,
    get_rifke,
    get_motion_features,
    get_bvh_cont6d_params,
    process_anim,
    infer_translation_root_index_from_features,
    mirror_features_with_safeguards,
    get_common_features_from_T_pose,
    TPoseFeatures,
    get_hml_aligned_anim,
    get_motion,
    recover_processed_animation_from_feature_animation,
    recover_root_quat_and_pos_np,
    recover_from_bvh_ric_np,
    recover_from_bvh_rot_np,
    recover_animation_from_motion_np,
    recover_bvh_export_animation_from_motion_np,
)

# ── dataset_pipeline (dataset building) ─────────────────────────────────
from .dataset_pipeline import (
    DatasetPreprocessingError,
    create_data_samples,
    process_skeleton,
    get_mean_std,
    validate_anim_dir_update_state,
)
