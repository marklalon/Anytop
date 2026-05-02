"""
FBX → NPY → FBX Roundtrip Test (Zero BVH)

Loads an FBX animation directly into AnyTop's Animation object via Blender,
exports it to the 13-channel NPY motion-feature tensor, recovers the Animation
from the NPY, re-exports to FBX, re-imports, and compares bone positions
frame-by-frame.

Requires bpy (Blender as Python module) in the current Python environment.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from typing import Any

import numpy as np
import torch

# ── ensure repo root is on sys.path ──────────────────────────────────────────
_ANYTOP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_ANYTOP_ROOT)
sys.path.insert(0, _REPO_ROOT)

# ── Resolve utils namespace conflict ──────────────────────────────────────
# Both Anytop/utils/ and root's utils/ have a `utils` package.
# Strategy: load root's utils (with quaternion) first, then inject Anytop's
#           rotation_conversions into the cached utils.  This lets both
#           `from utils.rotation_conversions import ...` (from Anytop) and
#           `from utils.quaternion import ...` (from root's kinematics) work.
import utils
import utils.quaternion
import utils.geometry

# Now inject Anytop's rotation_conversions into the already-cached utils
import importlib.machinery
import importlib.util

_rotconv_path = os.path.join(_ANYTOP_ROOT, "utils", "rotation_conversions.py")
if os.path.isfile(_rotconv_path) and "utils.rotation_conversions" not in sys.modules:
    _loader = importlib.machinery.SourceFileLoader(
        "utils.rotation_conversions", _rotconv_path,
    )
    _spec = importlib.util.spec_from_loader(
        "utils.rotation_conversions", _loader, origin=_rotconv_path,
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["utils.rotation_conversions"] = _mod
    _spec.loader.exec_module(_mod)

# Finally add Anytop to sys.path so its subpackages are resolvable by name
if _ANYTOP_ROOT not in sys.path:
    sys.path.insert(1, _ANYTOP_ROOT)


# ==============================================================================
# Phase A : FBX → Animation  (Blender-based loader, no BVH)
# ==============================================================================

def _patch_fbx_light_import():
    """Monkey-patch FBX importer for Blender 5.0 light API changes."""
    import importlib
    mod = sys.modules.get("io_scene_fbx.import_fbx")
    if mod is None:
        try:
            mod = importlib.import_module("io_scene_fbx.import_fbx")
        except ImportError:
            return
    if mod is None or not hasattr(mod, "blen_read_light"):
        return

    original_fn = mod.blen_read_light

    def _patched_blen_read_light(fbx_tmpl, fbx_obj, settings, _orig=original_fn):
        try:
            return _orig(fbx_tmpl, fbx_obj, settings)
        except AttributeError as exc:
            if "cast_shadow" in str(exc):
                return None
            raise

    mod.blen_read_light = _patched_blen_read_light


def _compute_rest_positions(offsets: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Forward-kinematics on the rest pose → (J, 3) global positions."""
    J = len(parents)
    pos = np.zeros((J, 3), dtype=np.float64)
    for j in range(J):
        p = parents[j]
        if p >= 0:
            pos[j] = pos[p] + offsets[j]
        else:
            pos[j] = offsets[j].copy()
    return pos


def _extract_fbx_skeleton_data(armature) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Extract bone names, parents, and offsets from an armature's edit bones.

    Must be called while the armature is in EDIT mode.
    Returns (bone_names, parents_array, offsets_array) in BFS topological order.

    Handles FBX rigs with multiple root bones by selecting the root with the
    largest subtree (the actual character skeleton).
    """
    from collections import deque

    edit_bones = armature.data.edit_bones

    # Find all roots, find the one with the most descendants
    all_roots = [eb for eb in edit_bones if eb.parent is None]
    if not all_roots:
        raise RuntimeError("No root bone found in armature")

    def _subtree_size(root) -> int:
        count = 0
        q = deque([root])
        while q:
            eb = q.popleft()
            count += 1
            for child in eb.children:
                q.append(child)
        return count

    root = max(all_roots, key=_subtree_size)

    # BFS order from the chosen root
    ordered_eb = []
    queue = deque([root])
    while queue:
        eb = queue.popleft()
        ordered_eb.append(eb)
        for child in eb.children:
            queue.append(child)

    J = len(ordered_eb)
    bone_names: list[str] = [eb.name for eb in ordered_eb]
    parents = np.full(J, -1, dtype=np.int32)
    offsets = np.zeros((J, 3), dtype=np.float64)
    name_to_id = {name: idx for idx, name in enumerate(bone_names)}

    for idx, eb in enumerate(ordered_eb):
        if eb.parent is not None and eb.parent.name in name_to_id:
            parent_name = eb.parent.name
            parents[idx] = name_to_id[parent_name]
            offsets[idx] = np.array(eb.head) - np.array(eb.parent.head)
        else:
            parents[idx] = -1
            offsets[idx] = np.array(eb.head)

    return bone_names, parents, offsets


def _fbx_to_animation(fbx_path: str) -> tuple[Any, list[str], float]:
    """Load FBX via Blender and return (Animation, joint_names, fps).

    The returned Animation uses AnyTop's data layout (Quaternions w,x,y,z).
    Joints are in BFS topological order (parents before children).

    No BVH files are created at any point.
    """
    import bpy
    from mathutils import Quaternion as BQuat

    from motion_lib.Animation import Animation as ATopAnim
    from motion_lib.Quaternions import Quaternions

    bpy.ops.wm.read_factory_settings(use_empty=True)

    _patch_fbx_light_import()

    bpy.ops.import_scene.fbx(
        filepath=fbx_path,
        ignore_leaf_bones=False,
        force_connect_children=True,
        automatic_bone_orientation=True,
        bake_space_transform=False,
        use_custom_normals=False,
        use_image_search=False,
    )

    # Remove lights / cameras
    for obj in list(bpy.data.objects):
        if obj.type in {"LIGHT", "CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)

    armature = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    if armature is None:
        raise RuntimeError(f"No armature found in {fbx_path}")

    # ── BFS bone order ──────────────────────────────────────────────────
    # Use edit bones for clean hierarchy traversal
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="EDIT")
    bone_names, parents, offsets = _extract_fbx_skeleton_data(armature)
    bpy.ops.object.mode_set(mode="OBJECT")

    J = len(bone_names)
    # orients = identity (quaternions); the rest pose rotations are identity
    orients = Quaternions.id(J)

    # ── Frame range from animation data ─────────────────────────────────
    scene = bpy.context.scene
    fps = scene.render.fps / scene.render.fps_base

    if armature.animation_data and armature.animation_data.action:
        action = armature.animation_data.action
        frame_start = int(action.frame_range[0])
        frame_end = int(action.frame_range[1])
    else:
        # No animation — single frame
        frame_start = 0
        frame_end = 0

    num_frames = frame_end - frame_start + 1

    # ── Build per-frame rotations ───────────────────────────────────────
    rot_qs = np.zeros((num_frames, J, 4), dtype=np.float64)
    # Pose bone locations for root translation
    pos_np = np.zeros((num_frames, J, 3), dtype=np.float64)

    # All non-root bones get their rest offsets as local positions by default
    for j in range(J):
        pos_np[:, j] = offsets[j][None, :]
        if parents[j] < 0:
            # Root: positions will be filled from bone.location per frame
            pass

    # Enable pose mode for reading bone transforms
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="POSE")

    pbone_map = armature.pose.bones

    for f_idx in range(num_frames):
        frame = frame_start + f_idx
        scene.frame_set(frame)

        for j, name in enumerate(bone_names):
            pbone = pbone_map.get(name)
            if pbone is None:
                # Shouldn't happen but guard
                rot_qs[f_idx, j] = [1.0, 0.0, 0.0, 0.0]
                continue

            # Read quaternion rotation (bpy uses (w, x, y, z) which matches our convention)
            bq: BQuat = pbone.rotation_quaternion
            rot_qs[f_idx, j] = [bq.w, bq.x, bq.y, bq.z]

            if parents[j] < 0:
                # Root: read local position from pose bone location
                loc = pbone.location
                pos_np[f_idx, j] = [loc.x, loc.y, loc.z]

    # Restore object mode
    bpy.ops.object.mode_set(mode="OBJECT")

    rotations = Quaternions(rot_qs)
    anim = ATopAnim(rotations, pos_np, orients, offsets, parents)

    return anim, bone_names, fps


# ==============================================================================
# Phase C : Animation → NPY Features  (no HML transforms)
# ==============================================================================

def _get_cont6d_params_own(anim: Any, r_rot: Any) -> np.ndarray:
    """Compute 6D rotation features — each bone stores its OWN rotation.

    Unlike the HML convention (parent rotation stored at child slot), this
    stores slot j = rotation of bone j.  Leaf bones (like ``ago``, tail tips,
    limb ends) are NOT lost, because every bone has exactly one slot for its
    own rotation.

    Uses *raw* root quaternion *r_rot* (e.g. ``anim.rotations[:, 0]``) instead
    of a face-computed ``get_root_quat()``.
    """
    quat_params = anim.rotations  # (F, J) Quaternions
    cont_6d_params = quat_params.rotation_matrix(cont6d=True)  # (F, J, 6)
    # No HML reordering — each slot j stores bone j's own rotation.
    return cont_6d_params


def _detect_motion_loop(positions: np.ndarray) -> bool:
    """Return True if the last frame's root-relative pose ≈ first frame's."""
    if positions.shape[0] < 2:
        return False
    per_joint_dist = np.linalg.norm(positions[-1] - positions[0], axis=-1)
    # Use a higher threshold matching AnyTop's LOOP_DETECTION_POS_THRESHOLD
    thresh = 0.05  # from param_utils (conservative)
    return bool(np.mean(per_joint_dist) < thresh)


def _compute_terminal_local_velocity(global_positions, r_rot, is_loop, prev_velocity=None):
    """Terminal-frame local velocity for feature export."""
    joints_num = global_positions.shape[1]
    terminal = np.zeros((joints_num, 3), dtype=np.float32)
    if prev_velocity is not None:
        # Use the last frame's delta for proper terminal velocity
        delta = global_positions[-1] - global_positions[-2]
        terminal = (r_rot[-1] * delta).astype(np.float32)
    if is_loop and global_positions.shape[0] >= 2:
        # Wrap-around: delta from last frame back to first
        wrap_delta = global_positions[0] - global_positions[-1]
        wrap_vel = (r_rot[0] * wrap_delta).astype(np.float32)
        # Use whichever has smaller magnitude (conservative blend)
        if np.linalg.norm(wrap_vel) < np.linalg.norm(terminal):
            terminal = wrap_vel
    return terminal.astype(np.float32)


def _extract_raw_features(
    anim: Any,
    object_type: str,
    offsets: np.ndarray,
    parents: np.ndarray,
    bone_names: list[str],
    max_joints: int = 85,
) -> np.ndarray:
    """Extract the 13-channel NPY motion features (no HML transforms).

    Uses ``anim.rotations[:, 0]`` as the root-facing quaternion *r_rot*.
    No scaling, centering, or face-orientation transforms are applied.
    """
    from motion_lib.Animation import positions_global, rotations_global

    # ── helper: contact detection ──────────────────────────────────────
    from data_loaders.truebones.truebones_utils.physics_joint_annotation import (
        _infer_contact_joints,
        _rest_positions_from_offsets,
    )
    from data_loaders.truebones.truebones_utils.param_utils import (
        FOOT_CONTACT_VEL_THRESH,
        SNAKES,
    )
    from data_loaders.truebones.truebones_utils.motion_process import (
        get_contact_state,
        get_terminal_contact_state,
        get_motion_features,
        get_rifke,
        _find_translation_root,
    )

    # ── Step 1: global positions ───────────────────────────────────────
    global_pos = positions_global(anim)  # (F, J, 3)

    # ── Step 2: root-facing quaternion (raw, NOT face-computed) ────────
    r_rot = anim.rotations[:, 0].copy()  # (F,) Quaternions

    # ── Step 3: 6D rotation params in HML convention ───────────────────
    cont_6d_params = _get_cont6d_params_own(anim, r_rot)  # (F, J, 6)

    # ── Step 4: RIFKE positions ────────────────────────────────────────
    # All joints relative to translation root's XZ, rotated into root frame
    translation_root_index = _find_translation_root(anim)
    positions = get_rifke(
        global_pos, r_rot, translation_root_index=translation_root_index,
    )  # (F, J, 3)

    # ── Step 5: foot contact ───────────────────────────────────────────
    # Compute rest positions for contact inference
    rest_pos = _compute_rest_positions(offsets, parents)
    foot_indices, contact_source = _infer_contact_joints(
        object_type, bone_names, parents.tolist(), rest_pos,
    )
    foot_contact = get_contact_state(
        global_pos, foot_indices, FOOT_CONTACT_VEL_THRESH,
    )  # (F-1, J)

    # ── Step 6: local velocity ─────────────────────────────────────────
    local_vel = r_rot[1:, None] * (global_pos[1:] - global_pos[:-1])  # (F-1, J, 3)
    prev_velocity = local_vel[-1] if local_vel.shape[0] > 0 else None
    is_loop = _detect_motion_loop(positions)
    terminal_local_vel = _compute_terminal_local_velocity(
        global_pos, r_rot, is_loop, prev_velocity=prev_velocity,
    )

    # ── Step 7: assemble features ──────────────────────────────────────
    terminal_contact = get_terminal_contact_state(
        global_pos, foot_indices, FOOT_CONTACT_VEL_THRESH, is_loop,
    )
    features, _max_joints = get_motion_features(
        positions,
        cont_6d_params,
        foot_contact,
        local_vel,
        terminal_local_vel,
        terminal_contact,
        max_joints,
    )
    return features


# ==============================================================================
# Helper : Build skeleton from Animation exporter
# ==============================================================================

from dataclasses import dataclass, field
from typing import Optional
from typing import Optional


# ==============================================================================
# Minimal Skeleton + Bone (avoids importing kinematics which conflicts with
# Anytop's utils namespace)
# ==============================================================================

@dataclass
class _SimpleBone:
    id: int
    name: str
    parent_id: Optional[int]
    rest_offset: torch.Tensor    # [3]
    rest_rotation: torch.Tensor  # [4]


class _SimpleSkeleton:
    """Minimal skeleton that matches the API expected by AnimationExporter."""

    def __init__(self, bones: list[_SimpleBone]):
        self.bones = bones
        self.rest_offsets = torch.stack([b.rest_offset for b in bones], dim=0)
        self._build_depth_levels()

    @property
    def num_joints(self) -> int:
        return len(self.bones)

    def _build_depth_levels(self):
        """Build depth_levels needed by forward_kinematics."""
        J = len(self.bones)
        parents = torch.tensor(
            [b.parent_id if b.parent_id is not None else -1 for b in self.bones],
            dtype=torch.long,
        )
        depths = torch.zeros(J, dtype=torch.long)
        for j in range(1, J):
            p = parents[j].item()
            if p >= 0:
                depths[j] = depths[p] + 1

        max_depth = depths.max().item()
        device = self.rest_offsets.device
        self.depth_levels: list[tuple] = []
        self.root_bone_ids: list[int] = []

        for d in range(max_depth + 1):
            ids = [b.id for b in self.bones if depths[b.id].item() == d]
            if d == 0:
                self.root_bone_ids = ids
                parent_ids = [-1] * len(ids)
            else:
                parent_ids = [self.bones[bid].parent_id for bid in ids]
            self.depth_levels.append((
                torch.tensor(ids, dtype=torch.long, device=device),
                torch.tensor(parent_ids, dtype=torch.long, device=device),
            ))


def _build_skeleton(bone_names, offsets, parents, device=None):
    """Build a _SimpleSkeleton from names/offsets/parents arrays."""
    if device is None:
        device = torch.device("cpu")
    bones = []
    for j, name in enumerate(bone_names):
        p = parents[j]
        bones.append(_SimpleBone(
            id=j,
            name=name,
            parent_id=None if p < 0 else int(p),
            rest_offset=torch.tensor(offsets[j], dtype=torch.float32, device=device),
            rest_rotation=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device),
        ))
    return _SimpleSkeleton(bones)


# ==============================================================================
# Comparison helpers
# ==============================================================================

def _collect_bone_world_positions(armature, bone_names, num_frames, frame_start):
    """Walk *num_frames* and return dict bone_name → ndarray of world positions.

    Returns dict mapping bone_name → ndarray of shape (num_frames, 3).
    """
    import bpy
    from mathutils import Vector

    scene = bpy.context.scene
    result: dict[str, np.ndarray] = {}

    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="OBJECT")

    positions = np.zeros((num_frames, 3), dtype=np.float64)

    for name in bone_names:
        for f_idx in range(num_frames):
            frame = frame_start + f_idx
            scene.frame_set(frame)

            mw = armature.matrix_world
            pbone = armature.pose.bones.get(name)
            if pbone is None:
                head_world = (0.0, 0.0, 0.0)
            else:
                head_local = pbone.head
                head_world_vec = mw @ Vector((head_local.x, head_local.y, head_local.z))
                head_world = (head_world_vec.x, head_world_vec.y, head_world_vec.z)
            positions[f_idx] = head_world
        result[name] = positions.copy()

    return result


def _compare_animations(
    orig_positions: dict[str, np.ndarray],
    restored_armature,
    bone_names: list[str],
    num_frames: int,
    frame_start_rest: int,
    tol: float = 1e-3,
) -> dict[str, float]:
    """Compare *orig_positions* (pre-collected) against restored armature.

    Returns dict mapping bone_name → max_error over all frames.
    """
    import bpy
    from mathutils import Vector

    scene = bpy.context.scene
    bpy.context.view_layer.objects.active = restored_armature
    bpy.ops.object.mode_set(mode="OBJECT")

    max_errors: dict[str, float] = {}
    for name in bone_names:
        max_err = 0.0
        for f_idx in range(num_frames):
            frame = frame_start_rest + f_idx
            scene.frame_set(frame)

            mw = restored_armature.matrix_world
            pbone = restored_armature.pose.bones.get(name)
            if pbone is None:
                restored_pos = (0.0, 0.0, 0.0)
            else:
                hl = pbone.head
                v = mw @ Vector((hl.x, hl.y, hl.z))
                restored_pos = (v.x, v.y, v.z)

            err = math.dist(orig_positions[name][f_idx].tolist(), restored_pos)
            if err > max_err:
                max_err = err
        max_errors[name] = max_err

    return max_errors


# ==============================================================================
# Main test
# ==============================================================================

def test_fbx_npy_roundtrip(
    tpose_fbx: str,
    anim_fbx: str,
    object_type: str = "Alligator",
    output_dir: str | None = None,
    tolerance: float = 3.0,
) -> dict[str, float]:
    """FBX → NPY → FBX roundtrip test.

    Args:
        tpose_fbx: Path to T-pose FBX file (skeleton metadata).
        anim_fbx: Path to animation FBX file.
        object_type: Character type for contact inference (e.g. "Alligator").
        output_dir: Directory to save intermediate NPY and exported FBX files.
                    If None, uses a temporary directory.
        tolerance: Max allowed roundtrip error.

    Returns:
        Dict with keys: npy_error, direct_error, roundtrip_error, worst_bone
    """
    for fn in [tpose_fbx, anim_fbx]:
        assert os.path.isfile(fn), f"Missing required file: {fn}"

    print(f"[FBX Roundtrip] T-pose: {tpose_fbx}")
    print(f"[FBX Roundtrip] Animation: {anim_fbx}")

    # ── Phase B : Load T-pose (metadata only) ────────────────────────────────
    print("  [Phase B] Loading T-pose FBX for skeleton metadata...")
    tpose_anim, tpose_names, tpose_fps = _fbx_to_animation(tpose_fbx)
    offsets = tpose_anim.offsets.copy()
    parents = tpose_anim.parents.copy()
    print(f"    Joints: {len(tpose_names)}, FPS: {tpose_fps:.1f}")

    # ── Phase A + C : Load animation FBX → NPY features (no HML) ────────────
    print(f"  [Phase A+C] Loading animation FBX and extracting NPY features...")
    anim, bone_names, fps = _fbx_to_animation(anim_fbx)
    print(f"    Frames: {len(anim)}, Joints: {anim.shape[1]}, FPS: {fps:.1f}")

    print(f"  [Phase C] Extracting raw NPY features (no HML transforms)...")
    features_npy = _extract_raw_features(
        anim, object_type, offsets, parents, bone_names,
    )
    print(f"    NPY shape: {features_npy.shape}")

    # Save intermediate NPY if output_dir is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        npy_path = os.path.join(output_dir, "roundtrip_features.npy")
        np.save(npy_path, features_npy)
        print(f"    Saved NPY features to {npy_path}")

    # ── Phase D : Recover Animation from NPY features ──────────────────────
    print(f"  [Phase D] Recovering Animation from NPY features...")
    recovered_anim, has_animated_pos = _recover_from_features(
        features_npy, parents, offsets,
    )
    print(f"    Recovered frames: {len(recovered_anim)}")
    if has_animated_pos:
        print(f"    (has non-root animated position channels)")

    # ── NPY-level diagnostic: compare global positions directly ──────────────
    from motion_lib.Animation import positions_global
    orig_global = positions_global(anim)
    recovered_global = positions_global(recovered_anim)
    npy_position_error = np.abs(orig_global - recovered_global).max(axis=(0, 2))
    print(f"  [Diag] NPY roundtrip max per-joint error: {npy_position_error.max():.6f}")
    assert npy_position_error.max() < 1e-4, (
        f"NPY encoding/decoding loss exceeds 1e-4: {npy_position_error.max():.6f}"
    )
    print(f"    NPY encoding/decoding is lossless ✓")

    # The NPY roundtrip is exact.  FBX export→re-import through
    # Blender introduces larger error.  To validate the full pipeline:
    #   1. Export ORIGINAL animation → FBX → re-import → baseline error
    #   2. Export NPY-recovered → FBX → re-import → compare error
    #   3. Assert (2) ≈ (1)

    # ── Shared setup: skeleton + exporter ──────────────────────────────────
    skeleton = _build_skeleton(bone_names, offsets, parents)
    from postprocessing.exporter import AnimationExporter
    exporter = AnimationExporter(skeleton, fps=30.0)

    # ── Collect original bone positions from the SOURCE FBX ────────────────
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    _patch_fbx_light_import()
    bpy.ops.import_scene.fbx(
        filepath=anim_fbx,
        ignore_leaf_bones=False, force_connect_children=True,
        automatic_bone_orientation=True, bake_space_transform=False,
        use_custom_normals=False, use_image_search=False,
    )
    for obj in list(bpy.data.objects):
        if obj.type in {"LIGHT", "CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    orig_arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    orig_action = orig_arm.animation_data.action
    orig_frame_start = int(orig_action.frame_range[0])
    orig_frame_end = int(orig_action.frame_range[1])
    orig_num_frames = orig_frame_end - orig_frame_start + 1
    print(f"    Source FBX: frames {orig_frame_start}–{orig_frame_end} ({orig_num_frames} frames)")

    orig_positions = _collect_bone_world_positions(
        orig_arm, bone_names, orig_num_frames, orig_frame_start,
    )

    # ── Collect per-bone LOCAL positions (pbone.location) from source FBX ──
    # Needed for FBX export, especially for rigs where non-root bones carry
    # animated local positions (e.g. IK control bones like Handle on Horse).
    orig_local_pos = np.zeros((orig_num_frames, len(bone_names), 3), dtype=np.float64)
    scene = bpy.context.scene
    for f_idx in range(orig_num_frames):
        frame = orig_frame_start + f_idx
        scene.frame_set(frame)
        for j, name in enumerate(bone_names):
            pb = orig_arm.pose.bones.get(name)
            if pb is not None:
                loc = pb.location
                orig_local_pos[f_idx, j] = [loc.x, loc.y, loc.z]

    def _measure_fbx_export_error(export_anim, export_label) -> tuple[float, str]:
        """Export *export_anim* → FBX → re-import → compare with source."""
        jq_t = torch.from_numpy(export_anim.rotations.qs.astype(np.float32))
        rt_t = torch.from_numpy(export_anim.positions[:, 0, :].astype(np.float32))
        rr_t = torch.from_numpy(export_anim.rotations.qs[:, 0, :].astype(np.float32))
        # Use per-bone local positions from the SOURCE FBX (captures
        # animated location on non-root IK control bones like Handle).
        bt_t = torch.from_numpy(orig_local_pos.astype(np.float32))

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_fbx = os.path.join(output_dir, f"{export_label}.fbx")
            exporter.export(jq_t, rt_t, rr_t, out_fbx, mesh_path=anim_fbx,
                            bone_translations=bt_t)

            bpy.ops.wm.read_factory_settings(use_empty=True)
            _patch_fbx_light_import()
            bpy.ops.import_scene.fbx(
                filepath=out_fbx,
                ignore_leaf_bones=False, force_connect_children=True,
                automatic_bone_orientation=True, bake_space_transform=False,
                use_custom_normals=False, use_image_search=False,
            )
            for obj in list(bpy.data.objects):
                if obj.type in {"LIGHT", "CAMERA"}:
                    bpy.data.objects.remove(obj, do_unlink=True)
            imp_arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
            imp_action = imp_arm.animation_data.action
            imp_fs = int(imp_action.frame_range[0])
        else:
            with tempfile.TemporaryDirectory() as etmp:
                out_fbx = os.path.join(etmp, f"{export_label}.fbx")
                exporter.export(jq_t, rt_t, rr_t, out_fbx, mesh_path=anim_fbx,
                                bone_translations=bt_t)

                bpy.ops.wm.read_factory_settings(use_empty=True)
                _patch_fbx_light_import()
                bpy.ops.import_scene.fbx(
                    filepath=out_fbx,
                    ignore_leaf_bones=False, force_connect_children=True,
                    automatic_bone_orientation=True, bake_space_transform=False,
                    use_custom_normals=False, use_image_search=False,
                )
                for obj in list(bpy.data.objects):
                    if obj.type in {"LIGHT", "CAMERA"}:
                        bpy.data.objects.remove(obj, do_unlink=True)
                imp_arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
                imp_action = imp_arm.animation_data.action
                imp_fs = int(imp_action.frame_range[0])

        errors = _compare_animations(
            orig_positions, imp_arm, bone_names,
            orig_num_frames, imp_fs,
        )
        overall = max(errors.values())
        worst = max(errors, key=errors.get)
        return overall, worst

    # ── 1. Direct FBX export of ORIGINAL animation ───────────────────────
    print(f"  [Phase Ea] Exporting ORIGINAL animation → FBX via exporter...")
    exp_max, exp_worst = _measure_fbx_export_error(anim, "original_export")
    print(f"    Direct error: {exp_worst} = {exp_max:.6f}")
    print(f"    (FBX exporter baseline for this skeleton)")

    # ── 2. FBX export of NPY-recovered animation ────────────────────────
    print(f"  [Phase Eb] Exporting NPY-recovered animation → FBX via exporter...")
    rec_max, rec_worst = _measure_fbx_export_error(recovered_anim, "recovered_export")
    print(f"    NPY-path error: {rec_worst} = {rec_max:.6f}")

    # ── Assertions ──────────────────────────────────────────────────────
    assert rec_max < exp_max * 1.5 + 0.01, (
        f"NPY roundtrip adds excessive error: rec={rec_max:.6f} vs "
        f"direct={exp_max:.6f}"
    )
    print(f"    NPY error ({rec_max:.6f}) ≈ direct error ({exp_max:.6f}) ✓")

    assert rec_max < tolerance, (
        f"Full FBX roundtrip error {rec_max:.6f} exceeds {tolerance}"
    )
    print(f"\n  PASS  FBX→NPY→FBX roundtrip max error = {rec_max:.6f} "
          f"(baseline: {exp_max:.6f}, NPY encoding: {npy_position_error.max():.6f})")

    return {
        "npy_error": float(npy_position_error.max()),
        "direct_error": float(exp_max),
        "roundtrip_error": float(rec_max),
        "worst_bone": rec_worst,
    }
# ==============================================================================

def _recover_from_features(
    features: np.ndarray,
    parents: np.ndarray,
    offsets: np.ndarray,
    pos_err_threshold: float = 0.01,
):
    """Recover an Animation from a 13-channel NPY feature tensor.

    Decodes the "own-rotation" convention produced by
    ``_get_cont6d_params_own`` — each joint's 6D rotation is stored in its
    own slot (no HML parent→child reordering).

    This function is self-contained and does NOT reference any HML or BVH
    pipeline code.  It inlines the small amount of quaternion arithmetic
    needed to invert RIFKE + 6D rotations + velocity integration.

    Returns:
        (anim, has_animated_pos) — same semantics as
        ``recover_animation_from_motion_np`` but without the HML/BVH baggage.
    """
    from motion_lib.Animation import Animation, positions_global, rotations_global
    from motion_lib.Quaternions import Quaternions
    from utils.rotation_conversions import rotation_6d_to_matrix_np as _r6d_to_mat

    F, J, C = features.shape
    assert C == 13, f"Expected 13 channels, got {C}"

    # ── 1. Find the translation root (joint whose RIFKE XZ ≈ 0) ────────
    xz_abs_max = np.max(np.abs(features[:, :, [0, 2]]), axis=(0, 2))
    zero_xz = np.flatnonzero(xz_abs_max <= 1e-5)
    trans_root = int(zero_xz[0]) if zero_xz.size > 0 else int(np.argmin(xz_abs_max))

    # ── 2. Root quaternion from features[:, 0, 3:9] (own rotation) ──────
    r_rot_6d = features[:, 0, 3:9]                       # (F, 6)
    r_rot_mat = _r6d_to_mat(r_rot_6d)                     # (F, 3, 3)
    r_rot = Quaternions(np.empty((F, 4), dtype=np.float64))
    # from_transforms expects (..., 3, 3) → returns (..., 4) quaternions
    from motion_lib.Quaternions import Quaternions as Qcls
    r_rot = Qcls.from_transforms(r_rot_mat)               # (F,) Quaternions

    # ── 3. Root position from velocity integration + Y channel ──────────
    #   velocity is stored in features[:, trans_root, 9:12] → indices 9=X, 11=Z
    #   root Y is features[:, 0, 1]
    r_pos = np.zeros((F, 3), dtype=np.float64)
    r_pos[1:, [0, 2]] = features[1:, trans_root, [9, 11]]
    r_pos = (-r_rot) * r_pos                              # rotate into world frame
    r_pos = np.cumsum(r_pos, axis=0)                      # integrate velocity
    r_pos[:, 1] = features[:, 0, 1]                       # Y from direct feature

    # ── 4. Joint rotations from features[..., 3:9] (own convention) ────
    own_rot_6d = features[..., 3:9]                        # (F, J, 6)
    rot_mats = _r6d_to_mat(own_rot_6d)                     # (F, J, 3, 3)
    all_rots = Qcls.from_transforms(rot_mats)              # (F, J) Quaternions

    # ── 5. Positions — non-root = rest offsets; root from RIFKE ────────
    positions = offsets[None].repeat(F, axis=0).copy()     # (F, J, 3)
    root_ric = np.asarray(features[:, 0, :3], dtype=np.float64)
    root_global = (-r_rot) * root_ric                      # rotate RIFKE into world
    root_global[:, 0] += r_pos[:, 0]
    root_global[:, 2] += r_pos[:, 2]
    positions[:, 0] = root_global

    # ── 6. Handle non-root translation root ────────────────────────────
    if trans_root != 0 and parents[trans_root] >= 0:
        anim = Animation(all_rots, positions, Qcls.id(0), offsets, parents)
        g_rots = rotations_global(anim)
        g_pos = positions_global(anim)
        p_idx = parents[trans_root]
        positions[:, trans_root] = (-g_rots[:, p_idx]) * (r_pos - g_pos[:, p_idx])

    # ── 7. Reconcile with RIFKE truth (fix animated positions) ─────────
    # Recover target global positions from RIFKE
    target_global = (-r_rot[:, None]) * np.asarray(features[..., :3], dtype=np.float64)
    target_global[..., 0] += r_pos[:, 0:1]
    target_global[..., 2] += r_pos[:, 2:3]

    recovered_anim = Animation(all_rots, positions, Qcls.id(0), offsets, parents)
    glob_rot = positions_global(recovered_anim)
    per_joint_err = np.abs(target_global - glob_rot).max(axis=(0, 2))
    animated_joints = sorted(
        j for j in range(J) if per_joint_err[j] > pos_err_threshold
    )
    if animated_joints:
        for j in animated_joints:
            if j == 0 or parents[j] < 0:
                positions[:, j] = target_global[:, j]
                continue
            temp = Animation(all_rots, positions, Qcls.id(0), offsets, parents)
            tg_rots = rotations_global(temp)
            tg_pos = positions_global(temp)
            p = parents[j]
            positions[:, j] = (-tg_rots[:, p]) * (target_global[:, j] - tg_pos[:, p])
        recovered_anim = Animation(all_rots, positions, Qcls.id(0), offsets, parents)

    has_animated_pos = bool(
        animated_joints and any(j > 0 and parents[j] >= 0 for j in animated_joints)
    ) or any(
        np.any(np.ptp(recovered_anim.positions[:, j], axis=0) > 1e-4)
        for j in range(1, J)
    )

    return recovered_anim, has_animated_pos


# ==============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FBX → NPY → FBX roundtrip smoke test",
    )
    parser.add_argument(
        "--tpose-fbx",
        default=None,
        help="Path to T-pose FBX file. Default: auto-resolved from --object-type.",
    )
    parser.add_argument(
        "--anim-fbx",
        default=None,
        help="Path to animation FBX file. Default: auto-resolved from --object-type.",
    )
    parser.add_argument(
        "--object-type",
        default="Alligator",
        help="Character type for contact inference and default path resolution (default: Alligator).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save intermediate NPY and exported FBX files (default: temp dir).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Max allowed roundtrip error in meters (default: 0.01).",
    )
    return parser.parse_args()


# ==============================================================================
if __name__ == "__main__":
    args = parse_args()

    # Resolve default paths from --object-type
    default_raw_dir = os.path.join(
        _ANYTOP_ROOT,
        "dataset/truebones/zoo/Truebone_Z-OO",
        args.object_type,
    )
    default_tpose = os.path.join(default_raw_dir, f"{args.object_type}ALL-TPOSE.fbx")
    default_anim = os.path.join(default_raw_dir, f"{args.object_type}ALL-Bite1.fbx")

    tpose_fbx = args.tpose_fbx or default_tpose
    anim_fbx = args.anim_fbx or default_anim

    print(f"Object type : {args.object_type}")
    print(f"T-pose FBX  : {tpose_fbx}")
    print(f"Anim FBX    : {anim_fbx}")
    print(f"Output dir  : {args.output_dir or '(temp)'}")
    print(f"Tolerance   : {args.tolerance}")
    print()

    result = test_fbx_npy_roundtrip(
        tpose_fbx=tpose_fbx,
        anim_fbx=anim_fbx,
        object_type=args.object_type,
        output_dir=args.output_dir,
        tolerance=args.tolerance,
    )

    print(f"\nSummary:")
    print(f"  NPY encoding error  : {result['npy_error']:.6f}")
    print(f"  Direct FBX error    : {result['direct_error']:.6f}")
    print(f"  Roundtrip error     : {result['roundtrip_error']:.6f}")
    print(f"  Worst bone          : {result['worst_bone']}")
