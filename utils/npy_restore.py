"""
Shared NPY -> Animation core behind the BVH preview and the GLB restore.

``sample/generate.py`` (the ``.bvh`` written next to every generated ``.npy``),
``tools/restore_glb_from_npy.py`` (skeleton-only / skinned GLB) and the
inspection BVHs of the retarget tools all decode the same 12-channel feature
tensor. This module is the one decode path, so the outputs can only differ in
the final writer::

    features (F, J, 12)
      -> recover_animation_from_motion_np              feature-basis Animation
      -> recover_processed_animation_from_feature_animation
                                                       bake the rest rotations
                                                       (identity => no-op)
      -> restore_space='native': invert the preprocess similarity
      -> fullbody_ik: re-solve rotations on the rigid export skeleton
      -> optional time resample
      -> build_skeleton                                exporter skeleton
      -> AnimationExporter.export_bvh / export_glb

Skeleton context
----------------
:func:`build_skeleton_only_context` reads everything from the cond entry:
identity rest rotations, feature-basis offsets, HML space. With identity rest
the bake is a no-op and the feature rotations are FK-able as they are, so the
BVH preview and the skeleton-only GLB are the same animation on the same
skeleton. :func:`build_mesh_restore_context` additionally loads a T-pose mesh
and re-expresses the motion in that rig's bind frame for the skinned export.

Translation root
----------------
Preprocessing collapses the rig so the joint that carries the locomotion is
the root; every cond entry stores ``translation_root_index == 0``. The core
therefore takes the index from the cond entry and never infers it from the
tensor (the old inference re-ran the full recovery once per candidate joint).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from motion_lib.Animation import Animation
from motion_lib.Quaternions import Quaternions
from utils.fullbody_ik import (
    DEFAULT_IK_STRETCH_FACTOR,
    rebuild_fullbody_animation_with_ik,
)
from utils.roundtrip_common import build_skeleton, identity_rest_rotations

_REQUIRED_COND_FIELDS = ("joints_names", "parents", "offsets", "scale_factor", "orientation_quat")

LogFn = Optional[Callable[[str], None]]


def _log(log: LogFn, message: str) -> None:
    if log is not None:
        log(message)


# ── Skeleton context ──────────────────────────────────────────────────────────

@dataclass
class RestoreSkeletonContext:
    """Everything the decode path needs to know about one skeleton.

    ``parents`` / ``offsets`` / ``tpose_rest_rotations`` describe the feature
    skeleton the tensor was encoded on. ``export_*`` describe the rig the
    animation is written onto; for a cond-only context they are the same
    skeleton, for a mesh context they are the mesh armature's collapsed rig.
    ``export_offsets_space`` says which units ``export_offsets`` carry:
    ``'hml'`` (normalized preprocess units, like the animation in HML mode) or
    ``'native'`` (the mesh rig's own units).
    """

    joint_names: list[str]
    export_joint_names: list[str]
    parents: np.ndarray
    offsets: np.ndarray
    tpose_rest_rotations: np.ndarray
    orientation_quat: np.ndarray
    scale_factor: float
    translation_root_index: int
    export_parents: np.ndarray
    export_offsets: np.ndarray
    export_rest_rotations: np.ndarray
    export_offsets_space: str = "hml"
    mesh_bone_names: Optional[list[str]] = None

    @property
    def joint_count(self) -> int:
        return len(self.parents)


def _require_cond_fields(cond_entry: dict, object_type: Optional[str]) -> None:
    for key in _REQUIRED_COND_FIELDS:
        if key not in cond_entry:
            raise ValueError(
                f"cond.npy entry for '{object_type or cond_entry.get('object_type')}' "
                f"is missing required field '{key}'."
            )


def cond_translation_root_index(cond_entry: dict, *, joint_count: Optional[int] = None) -> int:
    """The cond entry's translation root (0 for every collapsed rig), validated."""
    index = int(cond_entry.get("translation_root_index", 0) or 0)
    if index < 0 or (joint_count is not None and index >= int(joint_count)):
        raise ValueError(
            f"cond translation_root_index={index} out of range for {joint_count} joints"
        )
    return index


def _check_feature_joint_count(feature_joint_count: Optional[int], joint_names: list[str], object_type) -> None:
    if feature_joint_count not in (None, 0, len(joint_names)):
        raise ValueError(
            f"NPY has J={feature_joint_count} joints but cond.npy has "
            f"{len(joint_names)} joints for '{object_type}'."
        )


def build_skeleton_only_context(
    cond_entry: dict,
    *,
    object_type: Optional[str] = None,
    export_joint_names: Optional[list[str]] = None,
    feature_joint_count: Optional[int] = None,
) -> RestoreSkeletonContext:
    """Skeleton context from cond.npy alone -- no mesh, identity rest.

    The animation stays in the feature basis: with identity rest rotations the
    bake is a no-op and FK with the cond offsets reproduces the HML poses
    exactly. ``export_joint_names`` lets a caller write the BVH with the
    anatomical ``canonical_bvh_joint_names`` instead of the raw rig names.
    """
    _require_cond_fields(cond_entry, object_type)
    joint_names = list(cond_entry["joints_names"])
    _check_feature_joint_count(feature_joint_count, joint_names, object_type)
    parents = np.asarray(cond_entry["parents"], dtype=np.int32)
    offsets = np.asarray(cond_entry["offsets"], dtype=np.float32)
    if export_joint_names is None:
        export_joint_names = list(joint_names)
    elif len(export_joint_names) != len(joint_names):
        raise ValueError(
            f"export_joint_names has {len(export_joint_names)} names but the skeleton has "
            f"{len(joint_names)} joints"
        )
    identity_rest = identity_rest_rotations(len(joint_names))
    return RestoreSkeletonContext(
        joint_names=joint_names,
        export_joint_names=list(export_joint_names),
        parents=parents,
        offsets=offsets,
        tpose_rest_rotations=identity_rest,
        orientation_quat=np.asarray(cond_entry["orientation_quat"], dtype=np.float64),
        scale_factor=float(cond_entry["scale_factor"]),
        translation_root_index=cond_translation_root_index(cond_entry, joint_count=len(parents)),
        export_parents=parents.copy(),
        export_offsets=offsets.copy(),
        export_rest_rotations=identity_rest.copy(),
        export_offsets_space="hml",
        mesh_bone_names=None,
    )


def load_tpose_restore_metadata(
    tpose_mesh: str,
    object_type: str,
    *,
    expected_joint_count: Optional[int] = None,
) -> dict[str, object]:
    """Read the T-pose mesh's armature both as AnyTop features and as the raw rig."""
    from data_loaders.truebones.truebones_utils.motion_process import (
        TPoseFeatures,
        get_common_features_from_T_pose,
    )
    from motion_lib.FBX import collapse_root_skeleton
    from utils.roundtrip_common import load_fbx_skeleton_metadata

    tpose_lower = tpose_mesh.lower()
    if not tpose_lower.endswith((".fbx", ".glb", ".gltf")):
        raise ValueError(f"Unsupported T-pose mesh format: {tpose_mesh} - expected .fbx, .glb, or .gltf")

    raw_joint_names, raw_parents, raw_offsets, raw_rest_rotations = load_fbx_skeleton_metadata(tpose_mesh)
    uncropped_joint_cap = max(
        len(raw_joint_names),
        int(expected_joint_count or 0),
        1,
    )
    # Restore/inference should consume the full T-pose skeleton. The default
    # get_common_features_from_T_pose(max_joints=MAX_JOINTS) cap is a training
    # concern and would silently drop tail joints here.
    tp: TPoseFeatures = get_common_features_from_T_pose(
        tpose_mesh,
        object_type,
        max_joints=uncropped_joint_cap,
    )
    raw_parents = np.asarray(raw_parents, dtype=np.int32)
    raw_offsets = np.asarray(raw_offsets, dtype=np.float32)
    raw_rest_rotations = np.asarray(raw_rest_rotations, dtype=np.float32)
    collapsed_joint_names, collapsed_parents, collapsed_offsets, collapsed_rest_rotations = (
        collapse_root_skeleton(
            raw_joint_names,
            raw_parents,
            raw_offsets,
            raw_rest_rotations[None, ...],
            raw_offsets[None, ...],
        )
    )[:4]
    return {
        "joint_names": list(tp.names),
        "parents": np.asarray(tp.tpos_anim.parents, dtype=np.int32),
        "offsets": np.asarray(tp.offsets, dtype=np.float32),
        "tpose_rest_rotations": np.asarray(tp.tpos_rots[0], dtype=np.float32),
        "orientation_quat": np.asarray(tp.orientation_quat, dtype=np.float64),
        "scale_factor": float(tp.scale_factor),
        "raw_joint_names": list(raw_joint_names),
        "raw_parents": raw_parents,
        "raw_offsets": raw_offsets,
        "raw_rest_rotations": raw_rest_rotations,
        "collapsed_joint_names": list(collapsed_joint_names),
        "collapsed_parents": np.asarray(collapsed_parents, dtype=np.int32),
        "collapsed_offsets": np.asarray(collapsed_offsets, dtype=np.float32),
        "collapsed_rest_rotations": np.asarray(collapsed_rest_rotations[0], dtype=np.float32),
    }


def _remap_skeleton_metadata(
    source_names: list[str],
    source_parents: np.ndarray,
    source_offsets: np.ndarray,
    source_rest_rotations: np.ndarray,
    target_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if list(source_names) == list(target_names):
        return (
            np.asarray(source_parents, dtype=np.int32),
            np.asarray(source_offsets),
            np.asarray(source_rest_rotations),
        )

    source_index = {name: index for index, name in enumerate(source_names)}
    target_index = {name: index for index, name in enumerate(target_names)}
    missing = [name for name in target_names if name not in source_index]
    if missing:
        preview = missing[:10]
        suffix = "..." if len(missing) > 10 else ""
        raise ValueError(f"T-pose mesh is missing export skeleton joints: {preview}{suffix}")

    parents = np.full((len(target_names),), -1, dtype=np.int32)
    offsets = np.zeros((len(target_names), 3), dtype=np.float32)
    rest_rotations = np.zeros((len(target_names), 4), dtype=np.float32)
    for target_joint_idx, joint_name in enumerate(target_names):
        source_joint_idx = source_index[joint_name]
        offsets[target_joint_idx] = source_offsets[source_joint_idx]
        rest_rotations[target_joint_idx] = source_rest_rotations[source_joint_idx]
        parent_idx = int(source_parents[source_joint_idx])
        if parent_idx >= 0:
            parent_name = source_names[parent_idx]
            if parent_name not in target_index:
                raise ValueError(f"T-pose mesh parent '{parent_name}' for joint '{joint_name}' is missing")
            parents[target_joint_idx] = target_index[parent_name]

    return parents, offsets, rest_rotations


def build_mesh_restore_context(
    cond_entry: dict,
    tpose_mesh: str,
    object_type: str,
    *,
    feature_joint_count: Optional[int] = None,
) -> RestoreSkeletonContext:
    """Skeleton context for a skinned / mesh-rig export.

    The feature skeleton still comes from cond.npy; the T-pose mesh supplies the
    bind-pose rest rotations that re-express the motion in the rig's own local
    frames, and the collapsed mesh armature becomes the export skeleton (in the
    mesh's native units).
    """
    _require_cond_fields(cond_entry, object_type)
    joint_names = list(cond_entry["joints_names"])
    _check_feature_joint_count(feature_joint_count, joint_names, object_type)
    parents = np.asarray(cond_entry["parents"], dtype=np.int32)
    offsets = np.asarray(cond_entry["offsets"], dtype=np.float32)
    tpose_meta = load_tpose_restore_metadata(
        tpose_mesh,
        object_type,
        expected_joint_count=len(joint_names),
    )

    # ── T-pose rest rotations (from T-pose mesh), matched by joint name ────
    tpose_joint_names = list(tpose_meta["joint_names"])
    tpose_rest_src = np.asarray(tpose_meta["tpose_rest_rotations"], dtype=np.float32)
    tpose_name_index = {name: idx for idx, name in enumerate(tpose_joint_names)}
    tpose_rest_rotations = identity_rest_rotations(len(joint_names))
    for j, name in enumerate(joint_names):
        if name in tpose_name_index:
            tpose_rest_rotations[j] = tpose_rest_src[tpose_name_index[name]]

    export_joint_names = list(joint_names)
    export_parents, export_offsets, export_rest_rotations = _remap_skeleton_metadata(
        list(tpose_meta["collapsed_joint_names"]),
        np.asarray(tpose_meta["collapsed_parents"], dtype=np.int32),
        np.asarray(tpose_meta["collapsed_offsets"], dtype=np.float32),
        np.asarray(tpose_meta["collapsed_rest_rotations"], dtype=np.float32),
        export_joint_names,
    )

    return RestoreSkeletonContext(
        joint_names=joint_names,
        export_joint_names=export_joint_names,
        parents=parents,
        offsets=offsets,
        tpose_rest_rotations=tpose_rest_rotations,
        orientation_quat=np.asarray(cond_entry["orientation_quat"], dtype=np.float64),
        scale_factor=float(cond_entry["scale_factor"]),
        translation_root_index=cond_translation_root_index(cond_entry, joint_count=len(parents)),
        export_parents=np.asarray(export_parents, dtype=np.int32),
        export_offsets=np.asarray(export_offsets, dtype=np.float32),
        export_rest_rotations=np.asarray(export_rest_rotations, dtype=np.float32),
        export_offsets_space="native",
        mesh_bone_names=list(tpose_meta["raw_joint_names"]),
    )


# ── Space / time helpers ──────────────────────────────────────────────────────

def coerce_root_translation_xz(root_translation_xz: np.ndarray) -> np.ndarray:
    root_translation_xz = np.asarray(root_translation_xz, dtype=np.float64).reshape(-1)
    if root_translation_xz.size == 3:
        return root_translation_xz
    if root_translation_xz.size == 2:
        return np.array([root_translation_xz[0], 0.0, root_translation_xz[1]], dtype=np.float64)
    raise ValueError(
        f"root_translation_xz must have shape (2,) or (3,), got {root_translation_xz.shape}"
    )


def invert_preprocess_transform(
    processed_anim: Animation,
    *,
    scale_factor: float,
    root_translation_xz: Optional[np.ndarray],
    orientation_quat: np.ndarray,
) -> Animation:
    """Undo the preprocess similarity: HML (scaled, centred, oriented) -> raw rig space."""
    positions = processed_anim.positions.copy().astype(np.float64, copy=False)
    offsets = processed_anim.offsets.copy().astype(np.float64, copy=False)
    rotations = processed_anim.rotations.copy()

    scale_factor = float(scale_factor)
    if scale_factor <= 0.0:
        raise ValueError(f"scale_factor must be positive, got {scale_factor}")
    if abs(scale_factor - 1.0) > 1e-8:
        inv_scale = 1.0 / scale_factor
        positions *= inv_scale
        offsets *= inv_scale

    if root_translation_xz is not None:
        root_offset = coerce_root_translation_xz(root_translation_xz)
        positions[:, 0] += root_offset
        offsets[0] += root_offset

    orientation_quat = np.asarray(orientation_quat, dtype=np.float64)
    if orientation_quat.ndim > 1:
        orientation_quat = orientation_quat[0]
    if orientation_quat.shape != (4,):
        raise ValueError(f"orientation_quat must have shape (4,), got {orientation_quat.shape}")
    inverse_orientation = -Quaternions(orientation_quat[None, :])
    inverse_orientation = inverse_orientation.repeat(processed_anim.shape[0], axis=0)
    rotations[:, 0] = inverse_orientation * rotations[:, 0]
    positions[:, 0] = inverse_orientation * positions[:, 0]

    return Animation(
        rotations,
        positions,
        processed_anim.orients.copy(),
        offsets,
        processed_anim.parents.copy(),
    )


def resample_frame_indices(
    frame_count: int,
    src_fps: float,
    tgt_fps: float,
    min_length: Optional[int] = None,
) -> list[float]:
    """Fractional source-frame indices that resample ``frame_count`` to ``tgt_fps``.

    The indices span ``[0, frame_count - 1]`` spaced ``src_fps / tgt_fps`` frames
    apart, so a clip sampled at ``src_fps`` plays back at ``tgt_fps`` over the same
    time span (e.g. 136 frames at 30fps -> 68 indices at 15fps).  When ``min_length``
    is given and the resampled clip is shorter, the whole clip is instead
    *time-stretched* to exactly ``min_length`` frames -- ``min_length`` indices
    spread evenly across the source range -- so a short motion is interpolated
    (slowed down) to fill the minimum length with no looping/seam jump.  Returns
    plain ``range(frame_count)`` when resampling is impossible/unnecessary.
    """
    if frame_count < 1:
        return []
    if src_fps and tgt_fps and src_fps > 0 and tgt_fps > 0 and frame_count >= 2:
        step = src_fps / tgt_fps
        n = int(math.floor((frame_count - 1) / step + 1e-6)) + 1 if step > 0 else frame_count
        times = [i * step for i in range(max(n, 1))]
    else:
        times = [float(i) for i in range(frame_count)]
    if min_length and len(times) < min_length:
        times = np.linspace(0.0, float(frame_count - 1), min_length).tolist()
    return times


def resample_animation(animation: Animation, frame_times) -> Animation:
    """Resample an Animation in time at fractional ``frame_times`` (source-frame units).

    Positions are linearly interpolated and rotations are slerped between the two
    bracketing source frames; the rest pose (orients/offsets/parents) is unchanged.
    Integer-ratio downsampling (e.g. 30->15fps) lands exactly on source frames, so
    it is plain decimation with no interpolation error.
    """
    frame_count = animation.shape[0]
    times = np.clip(np.asarray(frame_times, dtype=np.float64), 0.0, frame_count - 1)
    lo = np.floor(times).astype(np.int64)
    hi = np.minimum(lo + 1, frame_count - 1)
    alpha = (times - lo)[:, None]   # (T, 1) -- broadcasts over joints in slerp/lerp

    positions = np.asarray(animation.positions, dtype=np.float64)
    new_positions = (
        positions[lo] * (1.0 - alpha[..., None]) + positions[hi] * alpha[..., None]
    )
    new_rotations = Quaternions.slerp(
        animation.rotations[lo], animation.rotations[hi], alpha
    )
    return Animation(
        new_rotations,
        new_positions,
        animation.orients.copy(),
        animation.offsets.copy(),
        animation.parents.copy(),
    )


# ── Core decode ───────────────────────────────────────────────────────────────

@dataclass
class RestoredAnimation:
    animation: Animation
    skeleton: object
    translation_root_index: int
    fps: float
    has_animated_pos: bool
    ik_error: Optional[tuple[float, float]] = None


def restore_animation_from_features(
    features: np.ndarray,
    ctx: RestoreSkeletonContext,
    *,
    restore_space: str = "hml",
    fullbody_ik: bool = False,
    stretch_factor: float = DEFAULT_IK_STRETCH_FACTOR,
    root_translation_xz: Optional[np.ndarray] = None,
    fps: float = 30.0,
    resample_fps: Optional[float] = None,
    resample_min_length: Optional[int] = None,
    anim_pos_threshold: float = 0.01,
    log: LogFn = None,
) -> RestoredAnimation:
    """Decode a feature tensor into an export-ready Animation on ``ctx``'s export rig.

    ``restore_space='hml'`` keeps the NPY's normalized / centred / oriented
    placement; ``'native'`` inverts the preprocess similarity into the mesh rig's
    own space (``root_translation_xz`` is only meaningful there).

    ``fullbody_ik`` re-solves the rotations on the rigid export skeleton so the
    position channels are honoured through rotations instead of per-joint local
    translations; bones stay within ``stretch_factor`` of their rest length. The
    translation root keeps its local pose verbatim -- that translation *is* the
    locomotion.
    """
    from data_loaders.truebones.truebones_utils.features import (
        recover_animation_from_motion_np,
        recover_processed_animation_from_feature_animation,
    )
    from data_loaders.truebones.truebones_utils.param_utils import FEATS_LEN

    if restore_space not in ("native", "hml"):
        raise ValueError(f"restore_space must be 'native' or 'hml', got {restore_space!r}")
    if stretch_factor < 0 or stretch_factor > 1.0:
        raise ValueError(f"stretch_factor must be in [0, 1], got {stretch_factor}")

    features = np.asarray(features)
    if features.ndim != 3:
        raise ValueError(f"Expected feature tensor with shape (F, J, C), got {features.shape}")
    frame_count, joint_count, channel_count = features.shape
    if joint_count != ctx.joint_count:
        raise ValueError(
            f"NPY has J={joint_count} joints but the skeleton context has {ctx.joint_count} joints."
        )
    if channel_count != FEATS_LEN:
        raise ValueError(f"Expected {FEATS_LEN} channels per joint, got {channel_count}.")

    translation_root_index = int(ctx.translation_root_index)

    # ── Feature-basis Animation ────────────────────────────────────────────
    _log(log, "Recovering feature-space animation from NPY...")
    feature_anim, has_animated_pos = recover_animation_from_motion_np(
        features,
        ctx.parents,
        ctx.offsets,
        translation_root_index=translation_root_index,
        anim_pos_threshold=anim_pos_threshold,
    )
    _log(log, f"Recovered: {feature_anim.shape[0]} frames")

    # ── Rest bake (identity rest => the feature animation unchanged) ──────
    export_anim = recover_processed_animation_from_feature_animation(
        feature_anim,
        ctx.tpose_rest_rotations,
    )

    if restore_space == "hml":
        _log(log, "HML space: staying in HML space (skipping inverse preprocess transform)")
    else:
        export_anim = invert_preprocess_transform(
            export_anim,
            scale_factor=ctx.scale_factor,
            root_translation_xz=root_translation_xz,
            orientation_quat=ctx.orientation_quat,
        )

    # ── Rigid export skeleton, in the animation's units ───────────────────
    # A mesh context carries its rig in native units; in HML mode the animation
    # is still preprocess-scaled, so the rigid skeleton IK solves against has to
    # be scaled the same way or every bone length is constrained to the wrong
    # size.
    export_offsets = np.asarray(ctx.export_offsets, dtype=np.float32)
    if restore_space == "hml" and ctx.export_offsets_space == "native":
        export_offsets = (export_offsets * ctx.scale_factor).astype(np.float32)

    ik_error = None
    if fullbody_ik:
        _log(log, f"Full-body IK reconstruction on export skeleton (stretch_factor={stretch_factor:.2f})...")
        export_anim, ik_mean_error, ik_max_error = rebuild_fullbody_animation_with_ik(
            export_anim,
            rigid_offsets=export_offsets,
            rigid_parents=np.asarray(ctx.export_parents, dtype=np.int32),
            preserved_position_indices=[translation_root_index],
            preserved_rotation_indices=[translation_root_index],
            stretch_factor=stretch_factor,
        )
        ik_error = (float(ik_mean_error), float(ik_max_error))
        _log(
            log,
            "Full-body IK residual joint error: "
            f"mean={ik_mean_error:.6f}, max={ik_max_error:.6f}",
        )
        _log(
            log,
            "Preserving translation-root local pose during IK: "
            f"{ctx.export_joint_names[translation_root_index]} (index {translation_root_index})",
        )

    # ── Resample in time (optional) ───────────────────────────────────────
    output_fps = float(fps)
    if resample_fps is not None and resample_fps > 0:
        if abs(resample_fps - fps) < 1e-6:
            _log(log, f"resample_fps ({resample_fps}) equals fps ({fps}), skipping resample.")
        else:
            src_frames = export_anim.shape[0]
            frame_times = resample_frame_indices(
                src_frames, fps, resample_fps, min_length=resample_min_length
            )
            export_anim = resample_animation(export_anim, frame_times)
            output_fps = float(resample_fps)
            _log(
                log,
                f"Resampled motion {src_frames} -> {export_anim.shape[0]} frames "
                f"({fps:g}fps -> {resample_fps:g}fps"
                + (f", min_length={resample_min_length}" if resample_min_length else "")
                + ")",
            )

    skeleton = build_skeleton(
        ctx.export_joint_names,
        export_offsets,
        np.asarray(ctx.export_parents, dtype=np.int32),
        np.asarray(ctx.export_rest_rotations, dtype=np.float32),
    )

    return RestoredAnimation(
        animation=export_anim,
        skeleton=skeleton,
        translation_root_index=translation_root_index,
        fps=output_fps,
        has_animated_pos=bool(has_animated_pos),
        ik_error=ik_error,
    )


# ── Writers ───────────────────────────────────────────────────────────────────

def export_animation_bvh(restored: RestoredAnimation, output_bvh: str) -> None:
    """Write a restored animation as BVH through the exporter's unified channel split."""
    from utils.exporter import AnimationExporter, animation_to_exporter_inputs

    joint_rotations, root_translation, root_rotation, bone_translations = (
        animation_to_exporter_inputs(restored.animation, restored.skeleton)
    )
    exporter = AnimationExporter(restored.skeleton, fps=restored.fps)
    exporter.export_bvh(
        joint_rotations,
        root_translation,
        root_rotation,
        output_bvh,
        bone_translations=bone_translations,
    )


def write_feature_bvh(
    features: np.ndarray,
    cond_entry: dict,
    output_bvh: str,
    *,
    fps: float,
    object_type: Optional[str] = None,
    joint_names: Optional[list[str]] = None,
    fullbody_ik: bool = False,
    stretch_factor: float = DEFAULT_IK_STRETCH_FACTOR,
    log: LogFn = None,
) -> RestoredAnimation:
    """BVH preview of a feature tensor on its cond skeleton (HML space, identity rest).

    This is the skeleton-only GLB's animation written as BVH: same decode, same
    rigid-skeleton IK when requested, so the preview shows what the GLB will.
    ``joint_names`` defaults to the cond's anatomical ``canonical_bvh_joint_names``.
    """
    features = np.asarray(features)
    if joint_names is None:
        joint_names = list(cond_entry.get("canonical_bvh_joint_names", cond_entry["joints_names"]))
    ctx = build_skeleton_only_context(
        cond_entry,
        object_type=object_type,
        export_joint_names=list(joint_names),
        feature_joint_count=int(features.shape[1]) if features.ndim == 3 else None,
    )
    restored = restore_animation_from_features(
        features,
        ctx,
        restore_space="hml",
        fullbody_ik=fullbody_ik,
        stretch_factor=stretch_factor,
        fps=fps,
        log=log,
    )
    export_animation_bvh(restored, output_bvh)
    return restored
