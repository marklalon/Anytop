"""
Pure-numpy cross-skeleton retargeting.

This module hosts the world-space alignment + semantic match-name remap +
inverse-FK math originally embedded inside ``AnimationExporter.export_glb``.
Lifting it out means callers other than the GLB export pipeline — e.g. the
cross-species reference-motion path in ``sample/generate.py`` — can run the
same retargeting without depending on ``bpy``.
"""
from __future__ import annotations

from typing import Optional, TypedDict

import numpy as np

from .rotation_numpy import (
    apply_rotation_to_quaternions_wxyz_np,
    quat_conjugate_wxyz_np,
    quat_multiply_wxyz_np,
    quat_rotate_wxyz_np,
)


# ---------------------------------------------------------------------------
# Canonical joint-name synonym map
# ---------------------------------------------------------------------------
# Maps common variant names to a single canonical form so that skeletons
# from different naming conventions (e.g. Truebone "Thigh" vs "Leg 1") can
# still match during retarget.  All keys and values are lower-cased.
#
# Important: keys that already contain digits (like "leg 1") are matched
# exactly.  Keys without digits (like "index") are matched as a prefix —
# the digit suffix is preserved on the canonical form.

_CANONICAL_SYNONYMS: dict[str, str] = {
    # --- Leg chain ---
    "leg 1": "thigh",
    "leg 2": "calf",
    "leg ankle": "foot",
    "leg ball 1": "toe 0",
    # --- Arm chain ---
    "arm collarbone": "clavicle",
    "arm 1": "upper arm",
    "arm 2": "forearm",
    "arm palm": "hand",
    "arm ball 1": "wrist",
    # --- Finger aliases (no-digit keys — matched as prefix) ---
    "index": "finger 0",
    "middle": "finger 1",
    "ring": "finger 2",
    "pinky": "finger 3",
    # --- Spine / neck ---
    "spine 1": "spine",
    "spine 2": "spine 1",
    "spine 3": "spine 2",
    "spine 4": "spine 3",
    "neck 1": "neck",
    "neck 2": "neck 1",
    # --- Face ---
    "jaw": "chin",
}

# Pre-compute which keys have digits (exact match only) vs no digits (prefix match).
_SYNONYM_EXACT = {k: v for k, v in _CANONICAL_SYNONYMS.items() if any(c.isdigit() for c in k)}
_SYNONYM_PREFIX = {k: v for k, v in _CANONICAL_SYNONYMS.items() if not any(c.isdigit() for c in k)}


def _normalize_match_name(name: str) -> str:
    """Return a normalized form of *name* for synonym-aware matching.

    Lookup strategy (three tiers):
      1. Exact lower-case match in the synonym map (e.g. "Leg 1" → "thigh").
      2. If the name starts with "left " or "right ", strip the side prefix,
         normalize the remainder, then re-prefix (e.g. "Right Leg 1" → "right thigh").
      3. Prefix match for digit-less keys — extracts trailing digit and preserves
         it on the canonical form (e.g. "Index 01" → "finger 1", "Middle 2" → "finger 2").
    If none match, returns the lower-cased name unchanged.
    """
    lower = name.lower().strip()
    # Exact match first
    if lower in _SYNONYM_EXACT:
        return _SYNONYM_EXACT[lower]
    # Strip side prefix and try again
    for side in ("left ", "right "):
        if lower.startswith(side):
            remainder = lower[len(side):]
            normalized_remainder = _normalize_match_name(remainder)
            return side + normalized_remainder
    # Prefix match for keys without digits (e.g. "Index 01" starts with "index")
    # Extract trailing digit and preserve it on the canonical form
    # so that "Index 02" → "finger 2", "Middle 1" → "finger 1", etc.
    for key, value in _SYNONYM_PREFIX.items():
        if lower == key or lower.startswith(key + " "):
            suffix = lower[len(key):].strip()
            if suffix:
                # Extract trailing digit(s) from the suffix
                digit = ""
                for ch in reversed(suffix):
                    if ch.isdigit():
                        digit = ch + digit
                    elif ch == " ":
                        continue
                    else:
                        break
                if digit:
                    # Replace the trailing digit in the canonical value
                    # e.g. "finger 0" + digit "02" → "finger 2"
                    digit_int = str(int(digit))
                    parts = value.rsplit(" ", 1)
                    if parts[-1].isdigit():
                        return parts[0] + " " + digit_int
                    return value + " " + digit_int
            return value
    return lower


# ---------------------------------------------------------------------------
# Shared helpers used by the retarget/exporter numpy path.
# ---------------------------------------------------------------------------


def _generate_coordinate_candidates_np():
    """Generate candidate 3x3 rotation/flip matrices for auto-detection."""
    I = np.eye(3, dtype=np.float64)

    def R_x(deg):
        c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)

    def R_y(deg):
        c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)

    def R_z(deg):
        c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

    return [
        ("identity", I),
        ("R_x(+90°)", R_x(90)),
        ("R_x(-90°)", R_x(-90)),
        ("R_y(+90°)", R_y(90)),
        ("R_y(-90°)", R_y(-90)),
        ("R_z(+90°)", R_z(90)),
        ("R_z(-90°)", R_z(-90)),
        ("R_x(+180°)", R_x(180)),
        ("R_z(+180°)", R_z(180)),
        ("flip_X", np.diag([-1, 1, 1])),
        ("flip_Y", np.diag([1, -1, 1])),
        ("flip_Z", np.diag([1, 1, -1])),
    ]


def _batch_forward_kinematics_np(
    local_rotations: np.ndarray,
    local_positions: np.ndarray,
    parents: np.ndarray,
    rest_rotations: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute world-space positions & rotations from local data.

    Args:
        local_rotations: (F, J, 4)  animated local quaternions
        local_positions: (F, J, 3)  local (parent-relative) translations
        parents:         (J,) int32 parent indices (-1 = root)
        rest_rotations:  (J, 4) or None — total local rot = rest_rot ⊗ local_rot

    Returns:
        world_positions: (F, J, 3)
        world_rotations: (F, J, 4)
    """
    F, J = local_rotations.shape[:2]

    if rest_rotations is not None:
        total_local = np.zeros((F, J, 4), dtype=np.float64)
        for j in range(J):
            total_local[:, j] = quat_multiply_wxyz_np(
                rest_rotations[j:j+1], local_rotations[:, j]
            )
    else:
        total_local = local_rotations.copy()

    world_pos = np.zeros((F, J, 3), dtype=np.float64)
    world_rot = np.zeros((F, J, 4), dtype=np.float64)

    for j in range(J):
        p = parents[j]
        if p < 0:
            world_pos[:, j] = local_positions[:, j]
            world_rot[:, j] = total_local[:, j]
        else:
            world_pos[:, j] = world_pos[:, p] + quat_rotate_wxyz_np(
                world_rot[:, p], local_positions[:, j]
            )
            world_rot[:, j] = quat_multiply_wxyz_np(world_rot[:, p], total_local[:, j])

    return world_pos, world_rot


def _batch_pose_fk_np(
    pose_rotations: np.ndarray,
    pose_locations: np.ndarray,
    parents: np.ndarray,
    rest_offsets: np.ndarray,
    rest_rotations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute world transforms using Blender pose-bone semantics.

    Local bone transform is modeled as:
        T_local = T(rest_offset) * R(rest_rotation) * T(pose_location) * R(pose_rotation)

    This matches how the exporter drives external FBX/GLB armatures through
    pose bone `location` and `rotation_quaternion` channels.
    """
    F, J = pose_rotations.shape[:2]
    world_pos = np.zeros((F, J, 3), dtype=np.float64)
    world_rot = np.zeros((F, J, 4), dtype=np.float64)

    for j in range(J):
        rest_q = np.repeat(rest_rotations[j:j+1], F, axis=0)
        total_local_rot = quat_multiply_wxyz_np(rest_q, pose_rotations[:, j])
        pose_loc_in_parent = rest_offsets[j:j+1] + quat_rotate_wxyz_np(rest_q, pose_locations[:, j])

        p = parents[j]
        if p < 0:
            world_pos[:, j] = pose_loc_in_parent
            world_rot[:, j] = total_local_rot
        else:
            world_pos[:, j] = world_pos[:, p] + quat_rotate_wxyz_np(world_rot[:, p], pose_loc_in_parent)
            world_rot[:, j] = quat_multiply_wxyz_np(world_rot[:, p], total_local_rot)

    return world_pos, world_rot


def _batch_internal_pose_fk_np(
    joint_rotations: np.ndarray,
    root_translation: np.ndarray,
    root_rotation: np.ndarray,
    pose_locations: np.ndarray | None,
    parents: np.ndarray,
    rest_offsets: np.ndarray,
    rest_rotations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute world pose with unified exporter semantics.

    External caller semantics:
      - ``joint_rotations`` carries animated local joint quaternions for all joints,
        including the root joint.
      - ``root_translation`` / ``root_rotation`` form an extra world-space wrapper
        transform applied before the skeleton hierarchy.
      - ``pose_locations`` carries optional Blender-style pose-bone location channels
        for non-root joints. The root entry is ignored; root world translation always
        comes from ``root_translation``.
    """
    F, J = joint_rotations.shape[:2]
    world_pos = np.zeros((F, J, 3), dtype=np.float64)
    world_rot = np.zeros((F, J, 4), dtype=np.float64)

    zero_loc = np.zeros((F, 3), dtype=np.float64)

    for j in range(J):
        rest_q = np.repeat(rest_rotations[j:j+1], F, axis=0)
        total_local_rot = quat_multiply_wxyz_np(rest_q, joint_rotations[:, j])

        if pose_locations is None or parents[j] < 0:
            pose_loc = zero_loc
        else:
            pose_loc = pose_locations[:, j]

        local_pos = np.repeat(rest_offsets[j:j+1], F, axis=0) + quat_rotate_wxyz_np(
            rest_q,
            pose_loc,
        )

        p = parents[j]
        if p < 0:
            world_pos[:, j] = root_translation + quat_rotate_wxyz_np(root_rotation, local_pos)
            world_rot[:, j] = quat_multiply_wxyz_np(root_rotation, total_local_rot)
        else:
            world_pos[:, j] = world_pos[:, p] + quat_rotate_wxyz_np(world_rot[:, p], local_pos)
            world_rot[:, j] = quat_multiply_wxyz_np(world_rot[:, p], total_local_rot)

    return world_pos, world_rot


# ---------------------------------------------------------------------------
# Public retargeting function
# ---------------------------------------------------------------------------

class RetargetResult(TypedDict):
    joint_rotations: np.ndarray         # (F, J_tgt, 4)
    root_translation: np.ndarray        # (F, 3)
    root_rotation: np.ndarray           # (F, 4)
    bone_translations: Optional[np.ndarray]  # (F, J_tgt, 3) or None
    target_world_positions: np.ndarray  # (F, J_tgt, 3)
    target_world_rotations: np.ndarray  # (F, J_tgt, 4)
    src_to_tgt: np.ndarray              # (J_src,) int32, -1 where no match
    common_count: int
    alignment_label: str
    alignment_error: float
    alignment_scale: float
    alignment_translation: np.ndarray   # (3,)


def retarget_world_space_np(
    *,
    src_parents: np.ndarray,
    src_rest_offsets: np.ndarray,
    src_rest_rotations: np.ndarray,
    tgt_parents: np.ndarray,
    tgt_rest_offsets: np.ndarray,
    tgt_rest_rotations: np.ndarray,
    src_joint_rotations: np.ndarray,
    src_root_translation: np.ndarray,
    src_root_rotation: np.ndarray,
    src_match_names: list[str],
    tgt_match_names: list[str],
    src_bone_translations: Optional[np.ndarray] = None,
    coordinate_search: bool = True,
    verbose: bool = True,
) -> RetargetResult:
    """Retarget an exporter-style animation from a source skeleton to a target.

    The math is identical to what ``AnimationExporter.export_glb`` performs when
    a ``mesh_path`` is supplied: canonical bone-name matching → world-space
    alignment (scale + translation + 1-of-12 rigid rotation) → inverse FK back
    to target local pose channels.

    Args:
        src_parents: (J_src,) int32 parent indices, -1 for root.
        src_rest_offsets: (J_src, 3) parent-relative rest offsets.
        src_rest_rotations: (J_src, 4) WXYZ rest rotations.
        tgt_parents: (J_tgt,) int32 parent indices.
        tgt_rest_offsets: (J_tgt, 3) parent-relative rest offsets.
        tgt_rest_rotations: (J_tgt, 4) WXYZ rest rotations.
        src_joint_rotations: (F, J_src, 4) exporter-style local pose rotations.
        src_root_translation: (F, 3) wrapper translation applied above root.
        src_root_rotation: (F, 4) wrapper rotation applied above root.
        src_bone_translations: optional (F, J_src, 3) pose-bone locations for
            non-root joints. Root entry is ignored.
        src_match_names: semantic match names for source joints.
        tgt_match_names: semantic match names for target joints.
        coordinate_search: when ``True``, sweep 12 rigid rotation/flip candidates
            to find the best alignment of rest poses. Set ``False`` when the
            source and target are known to share the same world basis (e.g.
            both are processed cond entries from the same dataset pipeline).
        verbose: print one-line summary diagnostics.

    Returns:
        A ``RetargetResult`` mapping. ``joint_rotations`` / ``root_translation``
        / ``root_rotation`` / ``bone_translations`` are exporter-input
        compatible and can be fed straight back into ``AnimationExporter`` or
        used to drive any other target-skeleton animation pipeline.
    """
    src_parents = np.asarray(src_parents, dtype=np.int32)
    tgt_parents = np.asarray(tgt_parents, dtype=np.int32)
    src_rest_offsets = np.asarray(src_rest_offsets, dtype=np.float64)
    src_rest_rotations = np.asarray(src_rest_rotations, dtype=np.float64)
    tgt_rest_offsets = np.asarray(tgt_rest_offsets, dtype=np.float64)
    tgt_rest_rotations = np.asarray(tgt_rest_rotations, dtype=np.float64)

    jr_np = np.asarray(src_joint_rotations, dtype=np.float64)
    rt_np = np.asarray(src_root_translation, dtype=np.float64)
    rr_np = np.asarray(src_root_rotation, dtype=np.float64)
    pose_locations_np = (
        np.asarray(src_bone_translations, dtype=np.float64)
        if src_bone_translations is not None else None
    )

    F = jr_np.shape[0]
    src_match_names = list(src_match_names)
    tgt_match_names = list(tgt_match_names)
    J_src = len(src_match_names)
    J_tgt = len(tgt_match_names)
    if len(src_match_names) != J_src:
        raise ValueError(
            f"Source match-name count {len(src_match_names)} does not match source joint count {J_src}"
        )
    if len(tgt_match_names) != J_tgt:
        raise ValueError(
            f"Target match-name count {len(tgt_match_names)} does not match target joint count {J_tgt}"
        )

    # ── B) Map source → target indices by semantic match name ─────────────
    # Two-pass matching:
    #   1. Exact canonical-name match (original behavior).
    #   2. Synonym-aware fuzzy match for remaining unmatched joints.
    tgt_match_to_idx = {name: i for i, name in enumerate(tgt_match_names)}
    src_to_tgt = np.full(J_src, -1, dtype=np.int32)
    matched_tgt = np.zeros(J_tgt, dtype=bool)

    # Pass 1: exact match
    for i, name in enumerate(src_match_names):
        target_index = tgt_match_to_idx.get(name)
        if target_index is not None and not matched_tgt[target_index]:
            src_to_tgt[i] = target_index
            matched_tgt[target_index] = True

    # Pass 2: synonym-aware fuzzy match
    # Build a normalized-name → target index map for unmatched targets.
    tgt_norm_to_idx = {
        _normalize_match_name(tgt_match_names[j]): j
        for j in range(J_tgt)
        if not matched_tgt[j]
    }
    for i in range(J_src):
        if src_to_tgt[i] >= 0:
            continue  # already matched
        norm = _normalize_match_name(src_match_names[i])
        target_index = tgt_norm_to_idx.get(norm)
        if target_index is not None:
            src_to_tgt[i] = target_index
            matched_tgt[target_index] = True

    # ── D) Source animation in world space ────────────────────────────────
    src_wpos, src_wrot = _batch_internal_pose_fk_np(
        jr_np, rt_np, rr_np, pose_locations_np,
        src_parents, src_rest_offsets, src_rest_rotations,
    )

    # ── E) Target rest pose in world space ────────────────────────────────
    identity_q = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    tgt_rest_local_rot = np.tile(identity_q, (J_tgt, 1))  # (J_tgt, 4)
    tgt_rest_local_pos = np.zeros((1, J_tgt, 3), dtype=np.float64)

    tgt_rest_wpos, tgt_rest_wrot = _batch_pose_fk_np(
        tgt_rest_local_rot[None],   # (1, J, 4)
        tgt_rest_local_pos,         # (1, J, 3)
        tgt_parents,
        tgt_rest_offsets,
        tgt_rest_rotations,
    )

    # ── F) Compute alignment on common bones using source REST pose ───────
    # (Not frame 0 of the animation, which may be in a running pose.)
    src_rest_local_rot = np.tile(identity_q, (J_src, 1))
    src_rest_local_pos = np.zeros((1, J_src, 3), dtype=np.float64)
    src_rest_wpos, _ = _batch_pose_fk_np(
        src_rest_local_rot[None], src_rest_local_pos,
        src_parents, src_rest_offsets, src_rest_rotations,
    )

    common_src_idx = [i for i in range(J_src) if src_to_tgt[i] >= 0]
    common_tgt_idx = [int(src_to_tgt[i]) for i in common_src_idx]

    if not common_src_idx:
        raise RuntimeError(
            "No common joints between source and target semantic match names.\n"
            f"  Source match names: {src_match_names[:10]}...\n"
            f"  Target match names: {tgt_match_names[:10]}..."
        )

    pos_src_rest = src_rest_wpos[:, common_src_idx, :]   # (1, K, 3)
    pos_tgt_rest = tgt_rest_wpos[:, common_tgt_idx, :]   # (1, K, 3)

    def _mean_bone_len(pos, local_tgt_idx):
        lengths = []
        for ci, fi in enumerate(local_tgt_idx):
            p = tgt_parents[fi]
            if p < 0:
                continue
            for ci2, fi2 in enumerate(local_tgt_idx):
                if fi2 == p:
                    diff = pos[0, ci] - pos[0, ci2]
                    lengths.append(float(np.linalg.norm(diff)))
                    break
        return float(np.mean(lengths)) if lengths else 1.0

    mean_len_src = _mean_bone_len(pos_src_rest, common_tgt_idx)
    mean_len_tgt = _mean_bone_len(pos_tgt_rest, common_tgt_idx)

    scale = mean_len_tgt / mean_len_src if mean_len_src > 1e-8 else 1.0
    if abs(scale - 1.0) < 0.001:
        scale = 1.0

    root_tgt_idx = int(np.flatnonzero(tgt_parents == -1)[0])
    root_in_common = None
    for ci, fi in enumerate(common_tgt_idx):
        if fi == root_tgt_idx:
            root_in_common = ci
            break
    if root_in_common is None:
        root_in_common = 0

    # No rest-pose translation alignment: both source and target go through
    # process_anim() (root centered at XZ origin, feet grounded at y≈0, common
    # scale_factor), so any rest-based t_align tends to introduce drift rather
    # than correct it — especially Y, where the bone-length ratio used for
    # `scale` rarely matches the root-height ratio across species.
    t_align = np.zeros(3, dtype=np.float64)
    pos_src_rest_st = pos_src_rest * scale

    candidates = _generate_coordinate_candidates_np() if coordinate_search else [
        ("identity", np.eye(3, dtype=np.float64))
    ]
    best_R = np.eye(3, dtype=np.float64)
    best_label = "identity"
    best_err = float("inf")

    for label, R in candidates:
        pos_candidate = pos_src_rest_st @ R.T
        err = float(np.mean(np.linalg.norm(pos_tgt_rest - pos_candidate, axis=-1)))
        if err < best_err:
            best_err = err
            best_label = label
            best_R = R

    if verbose:
        print(f"  [Retarget] common={len(common_src_idx)}/{J_src}, "
              f"target={J_tgt} bones, alignment error={best_err:.6f}")
        print(f"  [Retarget] alignment: scale={scale:.6f}, "
              f"rot={best_label}, "
              f"trans=({t_align[0]:.4f}, {t_align[1]:.4f}, {t_align[2]:.4f})")

    # ── G) Apply alignment and remap to target world-space ────────────────
    target_wpos = np.repeat(tgt_rest_wpos, F, axis=0)
    target_wrot = np.repeat(tgt_rest_wrot, F, axis=0)
    aligned_src_wpos = (
        src_wpos * scale + t_align[np.newaxis, np.newaxis, :]
    ) @ best_R.T
    aligned_src_wrot = apply_rotation_to_quaternions_wxyz_np(src_wrot, best_R)

    mapped_mask = np.zeros(J_tgt, dtype=bool)
    for ii, fi in enumerate(src_to_tgt):
        if fi >= 0:
            target_wpos[:, fi] = aligned_src_wpos[:, ii]
            target_wrot[:, fi] = aligned_src_wrot[:, ii]
            mapped_mask[fi] = True

    # Propagate FK for every unmatched target joint from its (already updated)
    # parent.  In practice these are leaf rotation helpers — joints appended by
    # ``augment_leaf_rotation_helpers`` whose names end with ``__rot_helper`` —
    # which never canonical-match across species.  They should follow their
    # real leaf parent's animated world transform while keeping their rest
    # local offset and rotation (identity for helpers).  tgt_parents is in
    # topological order (parent index < child index), so a single forward pass
    # suffices.
    F_q = aligned_src_wrot.shape[0]
    for j in range(J_tgt):
        if mapped_mask[j]:
            continue
        p = int(tgt_parents[j])
        if p < 0:
            continue
        rest_q_j = np.repeat(tgt_rest_rotations[j:j+1], F_q, axis=0)
        rest_off_j = np.repeat(tgt_rest_offsets[j:j+1], F_q, axis=0)
        target_wpos[:, j] = target_wpos[:, p] + quat_rotate_wxyz_np(
            target_wrot[:, p], rest_off_j,
        )
        target_wrot[:, j] = quat_multiply_wxyz_np(target_wrot[:, p], rest_q_j)

    # ── H) Inverse FK back to target local pose channels ──────────────────
    tgt_pose_rot = np.zeros((F, J_tgt, 4), dtype=np.float64)
    tgt_pose_rot[:] = identity_q
    tgt_pose_loc = np.zeros((F, J_tgt, 3), dtype=np.float64)

    identity_q_row = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    for j in range(J_tgt):
        parent_j = int(tgt_parents[j])
        if parent_j < 0:
            parent_world_rot = np.repeat(identity_q_row[np.newaxis], F, axis=0)
            parent_world_pos = np.zeros((F, 3), dtype=np.float64)
        else:
            parent_world_rot = target_wrot[:, parent_j]
            parent_world_pos = target_wpos[:, parent_j]

        rel_world_rot = quat_multiply_wxyz_np(
            quat_conjugate_wxyz_np(parent_world_rot),
            target_wrot[:, j],
        )
        tgt_pose_rot[:, j] = quat_multiply_wxyz_np(
            quat_conjugate_wxyz_np(np.repeat(tgt_rest_rotations[j:j+1], F, axis=0)),
            rel_world_rot,
        )

        rel_world_pos = target_wpos[:, j] - parent_world_pos
        rel_parent_pos = quat_rotate_wxyz_np(
            quat_conjugate_wxyz_np(parent_world_rot),
            rel_world_pos,
        )
        tgt_pose_loc[:, j] = quat_rotate_wxyz_np(
            quat_conjugate_wxyz_np(np.repeat(tgt_rest_rotations[j:j+1], F, axis=0)),
            rel_parent_pos - np.repeat(tgt_rest_offsets[j:j+1], F, axis=0),
        )

    root_mask = tgt_parents < 0
    root_indices = np.flatnonzero(root_mask)

    if root_indices.size > 0:
        out_root_rotation = tgt_pose_rot[:, root_indices[0], :].copy()
        out_root_translation = tgt_pose_loc[:, root_indices[0], :].copy()
    else:
        out_root_rotation = rr_np.copy()
        out_root_translation = rt_np.copy()

    has_nonzero_bone_translations = (
        pose_locations_np is not None
        or np.any(np.abs(tgt_pose_loc[:, ~root_mask, :]) > 1e-6)
    )
    out_bone_translations = tgt_pose_loc if has_nonzero_bone_translations else None

    if verbose:
        print(f"  [Retarget] Conversion complete: {F} frames, {J_tgt} bones")

    return RetargetResult(
        joint_rotations=tgt_pose_rot,
        root_translation=out_root_translation,
        root_rotation=out_root_rotation,
        bone_translations=out_bone_translations,
        target_world_positions=target_wpos,
        target_world_rotations=target_wrot,
        src_to_tgt=src_to_tgt,
        common_count=len(common_src_idx),
        alignment_label=best_label,
        alignment_error=best_err,
        alignment_scale=scale,
        alignment_translation=t_align,
    )
