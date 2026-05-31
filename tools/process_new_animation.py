#!/usr/bin/env python3
"""
Process new animation clips for an *existing* AnyTop skeleton.

Unlike ``process_new_skeleton.py`` (which builds a brand-new skeleton entry from
scratch, writing/overwriting ``cond.npy``), this tool consumes the character
definition that already lives in ``cond.npy`` for a given ``object_type`` and only
produces motion outputs for new animation files. It never creates a new cond entry
and never modifies any existing cond content.

The per-character constants the dataset was built with (offsets, scale factor,
orientation quaternion, foot/contact joints, leaf-rotation helpers, canonical BVH
joint names) are reproduced from the cond entry's recorded T-pose reference, so the
emitted ``.npy`` lands in exactly the same raw feature space as the training motions
and can be fed straight back in as a reference motion during inference (it is
normalized with that object_type's mean/std at inference time).

Differences from regular preprocessing:
  * Uses the existing cond entry for the requested object_type; cond.npy is read
    only, never written or modified.
  * Does NOT auto-split long animations into 200-frame windows — each input file is
    processed as a single clip of its full length.

Short-clip handling matches regular preprocessing (clips shorter than
--filter-min-length are discarded; clips shorter than --resample-min-length are
time-stretched), but a warning is printed whenever a clip is discarded or resampled
so the divergence from the raw input is never silent. Both can be disabled (=0).

Input:
  --input        A single animation file (FBX/GLB/GLTF) or a directory of them.
  --object-type  The existing cond object_type these animations belong to.
                 Inferred from filenames when omitted.

Output (created side-by-side with the input):
  <out>/motions/<object_type>_<action>_<n>.npy   processed motion features (reference motion)
  <out>/bvhs/<object_type>_<action>_<n>.bvh      BVH preview of the processed clip

  <out> defaults to the input directory (when --input is a directory) or the input
  file's parent directory (when --input is a single file). Override with --save-dir.

Usage:
  python tools/process_new_animation.py --input path/to/Dog --object-type Dog
  python tools/process_new_animation.py --input clip.glb --object-type Horse
  python tools/process_new_animation.py --input dir/ --object-type Dog --save-dir out/
"""

import argparse
import os
import sys

import numpy as np

ANYTOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ANYTOP_DIR not in sys.path:
    sys.path.insert(0, ANYTOP_DIR)

from motion_lib import BVH, FBX  # noqa: E402
from motion_lib.Animation import Animation  # noqa: E402
from motion_lib.Quaternions import Quaternions  # noqa: E402
from data_loaders.truebones.truebones_utils.motion_process import (  # noqa: E402
    get_common_features_from_T_pose,
    get_motion,
    needs_bvh_position_channels,
    reorder_animation_to_dfs,
)
# Internal helpers reused so short-clip filter/resample behaves identically to
# regular preprocessing (see _prepare_object_outputs in dataset_pipeline.py).
from data_loaders.truebones.truebones_utils.features import (  # noqa: E402
    _extract_motion_features_from_aligned_anims,
)
from data_loaders.truebones.truebones_utils.dataset_pipeline import (  # noqa: E402
    _resample_animation,
)
from data_loaders.truebones.truebones_utils.param_utils import (  # noqa: E402
    MAX_JOINTS,
    FOOT_CONTACT_VEL_THRESH,
    MOTION_DIR,
    BVHS_DIR,
    get_dataset_dir,
)
from data_loaders.truebones.truebones_utils.fbx_filename_rules import (  # noqa: E402
    normalize_action_name,
)
from utils.misc import infer_object_type_from_filename  # noqa: E402

ANIM_EXTENSIONS = ('.fbx', '.glb', '.gltf')


def _collect_input_files(input_path):
    """Return (animation_files, default_save_root) for a file or directory input."""
    if os.path.isdir(input_path):
        files = sorted(
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.lower().endswith(ANIM_EXTENSIONS)
        )
        return files, input_path
    if os.path.isfile(input_path):
        if not input_path.lower().endswith(ANIM_EXTENSIONS):
            raise SystemExit(
                f"Error: --input '{input_path}' is not an animation file "
                f"({', '.join(ANIM_EXTENSIONS)})."
            )
        return [input_path], os.path.dirname(os.path.abspath(input_path))
    raise SystemExit(f"Error: --input path does not exist: {input_path}")


def _load_object_cond(cond_path, object_type):
    if not os.path.exists(cond_path):
        raise SystemExit(
            f"Error: cond.npy not found at '{cond_path}'. Pass --dataset-dir or "
            f"--cond-path to point at the dataset whose skeleton definition to reuse."
        )
    cond = dict(np.load(cond_path, allow_pickle=True).item())
    if object_type not in cond:
        available = ', '.join(sorted(cond.keys()))
        raise SystemExit(
            f"Error: object_type '{object_type}' is not present in '{cond_path}'.\n"
            f"Available object types: {available}"
        )
    return cond[object_type]


def _resolve_tpose_reference(object_cond, object_type, tpose_override):
    """Resolve the T-pose reference file used to rebuild the character constants."""
    if tpose_override:
        if not os.path.isfile(tpose_override):
            raise SystemExit(f"Error: --tpose-path '{tpose_override}' does not exist.")
        return tpose_override

    recorded = object_cond.get('orientation_reference_fbx_path')
    if recorded and os.path.isfile(recorded):
        return recorded

    raise SystemExit(
        f"Error: cannot locate the T-pose reference for '{object_type}'. The path "
        f"recorded in cond ('{recorded}') is missing. The reference rest pose is "
        f"required to reproduce the dataset's per-character constants — pass it "
        f"explicitly with --tpose-path."
    )


def _build_character_constants(object_cond, object_type, tpose_path):
    """Reproduce the exact per-character constants the dataset motions were built with.

    These come from the recorded T-pose reference rather than being read piecemeal out
    of cond, because ``get_motion`` needs the T-pose local rotations (``tpos_rots``),
    which cond does not store. Re-deriving from the same reference with the same face
    joints reproduces every constant deterministically. cond.npy is not modified.
    """
    face_joint_names = object_cond.get('face_joint_names') or None
    tp = get_common_features_from_T_pose(
        tpose_path,
        object_type,
        face_joints=face_joint_names,
        augment_leaf_rotation_helpers=True,
        max_joints=MAX_JOINTS,
    )

    # Sanity-check the reproduced constants against what cond recorded so the user is
    # warned if the reference no longer reproduces a compatible feature space.
    cond_offsets = object_cond.get('offsets')
    if cond_offsets is not None and np.asarray(cond_offsets).shape != np.asarray(tp.offsets).shape:
        print(
            f"[process_new_animation] WARNING: reproduced offsets shape "
            f"{np.asarray(tp.offsets).shape} differs from cond offsets shape "
            f"{np.asarray(cond_offsets).shape}; emitted motions may not match the "
            f"dataset feature space.",
            flush=True,
        )
    cond_scale = object_cond.get('scale_factor')
    if cond_scale is not None and not np.isclose(float(cond_scale), float(tp.scale_factor), rtol=1e-4):
        print(
            f"[process_new_animation] WARNING: reproduced scale_factor "
            f"{tp.scale_factor:.6g} differs from cond scale_factor "
            f"{float(cond_scale):.6g}.",
            flush=True,
        )
    return tp


def _reference_skeleton_from_tpose(tp):
    """Return (names, parents) of the original (un-augmented) reference skeleton.

    ``tp.names`` / ``tp.tpos_anim`` already include the appended leaf-rotation
    helper joints. The raw input animation must match the skeleton *before* that
    augmentation, so we slice the leading ``original_joint_count`` entries (helpers
    are always appended at the end)."""
    helper_count = int(tp.helper_metadata.get('helper_joint_count', 0)) if tp.helper_metadata else 0
    original_joint_count = len(tp.names) - helper_count
    expected_names = list(tp.names[:original_joint_count])
    expected_parents = np.asarray(tp.tpos_anim.parents[:original_joint_count], dtype=np.int32)
    return expected_names, expected_parents


def _reindex_animation_subset(raw_anim, names, keep_indices):
    """Drop all joints except ``keep_indices`` and reindex parents/arrays.

    ``keep_indices`` must already be in ascending (DFS pre-order) order and must
    be closed under the parent relation (every kept joint's parent is kept), which
    the caller guarantees before calling."""
    old_to_new = {old: new for new, old in enumerate(keep_indices)}
    new_names = [names[i] for i in keep_indices]
    new_parents = np.array(
        [old_to_new[int(raw_anim.parents[i])] if int(raw_anim.parents[i]) >= 0 else -1
         for i in keep_indices],
        dtype=np.int32,
    )
    new_anim = Animation(
        Quaternions(raw_anim.rotations.qs[:, keep_indices].copy()),
        raw_anim.positions[:, keep_indices].copy(),
        Quaternions(raw_anim.orients.qs[keep_indices].copy()),
        raw_anim.offsets[keep_indices].copy(),
        new_parents,
    )
    return new_anim, new_names


def _align_input_skeleton_to_reference(raw_anim, names, expected_names, expected_parents, fname):
    """Validate/repair the raw input skeleton against the dataset reference skeleton.

    ``process_new_animation`` (via ``get_motion``) assumes the raw animation's
    joints match the dataset's canonical skeleton index-for-index, then appends
    leaf-rotation helpers to reach the feature joint count. That assumption is
    silent: an input carrying extra terminal bones (e.g. Blender ``*_end`` tip
    bones materialised on FBX/GLB export) can coincidentally match the *augmented*
    joint count and get mismatched joint-by-joint, scrambling the whole skeleton.

    Defense:
      * If the skeleton already matches the reference, return it unchanged.
      * Otherwise strip terminal (leaf) bones whose names are not in the reference
        — this removes ``*_end``-style tip bones while preserving DFS order — and
        re-validate.
      * If it still does not match (missing joints, reordered joints, or an
        *internal* extra bone that cannot be safely dropped), raise with a clear
        diff instead of silently producing corrupt motion.
    """
    expected_name_set = set(expected_names)

    if list(names) == list(expected_names):
        return raw_anim, list(names)

    # Identify leaf joints (no children) whose names are not in the reference.
    parents = np.asarray(raw_anim.parents, dtype=np.int32)
    has_children = np.zeros(len(names), dtype=bool)
    has_children[parents[parents >= 0]] = True
    unexpected_leaves = [
        i for i, name in enumerate(names)
        if name not in expected_name_set and not has_children[i]
    ]

    if unexpected_leaves:
        keep_indices = [i for i in range(len(names)) if i not in set(unexpected_leaves)]
        # Reject if dropping orphans an expected joint (its parent was removed) —
        # that means the extra bone is internal and cannot be safely stripped.
        keep_set = set(keep_indices)
        for i in keep_indices:
            p = int(parents[i])
            if p >= 0 and p not in keep_set:
                break
        else:
            stripped_names = [names[i] for i in unexpected_leaves]
            raw_anim, names = _reindex_animation_subset(raw_anim, names, keep_indices)
            print(
                f"[process_new_animation] WARNING: stripped {len(stripped_names)} terminal "
                f"bone(s) from {fname} not present in the '{','.join(expected_names[:1])}...' "
                f"reference skeleton: {stripped_names[:10]}"
                f"{'...' if len(stripped_names) > 10 else ''}. These are typically Blender "
                f"'*_end' tip bones; the clip is processed on the canonical "
                f"{len(expected_names)}-joint skeleton.",
                flush=True,
            )

    # Final structural validation (names AND parents must match exactly).
    if list(names) == list(expected_names) and np.array_equal(
        np.asarray(raw_anim.parents, dtype=np.int32), expected_parents
    ):
        return raw_anim, list(names)

    extra = [n for n in names if n not in expected_name_set]
    missing = [n for n in expected_names if n not in set(names)]
    raise SystemExit(
        f"Error: skeleton of '{fname}' does not match the dataset reference skeleton "
        f"({len(expected_names)} joints, root '{expected_names[0] if expected_names else '?'}') "
        f"and cannot be auto-aligned.\n"
        f"  input joints  : {len(names)}\n"
        f"  extra joints  ({len(extra)}): {extra[:10]}{'...' if len(extra) > 10 else ''}\n"
        f"  missing joints({len(missing)}): {missing[:10]}{'...' if len(missing) > 10 else ''}\n"
        f"  Re-export the animation on the dataset's canonical skeleton, or pass a matching "
        f"--tpose-path. (A same-count but differently-ordered skeleton would otherwise be "
        f"silently scrambled.)"
    )


def _process_one_file(file_path, object_type, tp, max_joints, filter_min_length, resample_min_length):
    """Process a single animation file as one full-length clip (no windowing).

    Short-clip filtering/resampling mirrors regular preprocessing, but emits a
    warning whenever a clip is discarded or resampled so the divergence from the
    raw input is never silent.

    Returns a list of result dicts (normally one; empty if the clip fails, is too
    short, or the animation is empty)."""
    raw_anim, names, frame_time = FBX.load(file_path)
    anim_len = len(raw_anim)
    fname = os.path.basename(file_path)
    if anim_len == 0:
        print(f"[process_new_animation] skipping empty animation: {file_path}", flush=True)
        return [], max_joints

    # Defense: make sure the raw skeleton matches the dataset's canonical skeleton
    # before get_motion blindly aligns it index-by-index against the T-pose. Strips
    # stray terminal '*_end' tip bones (or errors clearly on a real mismatch).
    expected_names, expected_parents = _reference_skeleton_from_tpose(tp)
    raw_anim, names = _align_input_skeleton_to_reference(
        raw_anim, names, expected_names, expected_parents, fname
    )

    local_errors = dict()
    # whole clip, no auto-splitting (slice_inds spans the full length)
    motion, _parents, max_joints, new_anim, export_anim, _is_loop, translation_root_index, _root_xz = get_motion(
        file_path,
        FOOT_CONTACT_VEL_THRESH,
        object_type,
        max_joints,
        tp.offsets,
        tp.foot_indices,
        tp.tpos_rots,
        local_errors,
        scale_factor=tp.scale_factor,
        orientation_quat=tp.orientation_quat,
        slice_inds=[0, anim_len],
        preloaded=(raw_anim, names),
        helper_metadata=tp.helper_metadata,
    )
    if motion is None:
        print(f"[process_new_animation] [FAIL] could not process: {file_path}", flush=True)
        return [], max_joints

    num_frames = motion.shape[0]
    if num_frames < filter_min_length:
        print(
            f"[process_new_animation] WARNING: discarding {fname}: {num_frames} frames "
            f"< filter-min-length {filter_min_length} (no output written for this clip).",
            flush=True,
        )
        return [], max_joints

    if resample_min_length > 0 and num_frames < resample_min_length:
        print(
            f"[process_new_animation] WARNING: resampling {fname} from {num_frames} to "
            f"{resample_min_length} frames to meet resample-min-length; the emitted clip "
            f"is time-stretched from the raw input.",
            flush=True,
        )
        # Mirror preprocessing: resample both aligned animations, then RECOMPUTE the
        # feature tensor (interpolating features directly would corrupt velocity,
        # contacts, and the 6D rotation rep). The resampled export_anim is what gets
        # written to BVH, matching _prepare_object_outputs.
        new_anim = _resample_animation(new_anim, resample_min_length)
        export_anim = _resample_animation(export_anim, resample_min_length)
        motion, max_joints, _motion_anim, _motion_export_anim, _is_loop = _extract_motion_features_from_aligned_anims(
            new_anim,
            export_anim,
            FOOT_CONTACT_VEL_THRESH,
            object_type,
            max_joints,
            tp.foot_indices,
            tp.orientation_quat,
            translation_root_index,
        )

    raw_action = fname.split('.')[0]
    action = normalize_action_name(object_type, raw_action)
    return [{
        'action': action,
        'motion': motion,
        'export_anim': export_anim,
        'frame_time': frame_time,
    }], max_joints


def _write_outputs(save_root, object_type, results, canonical_bvh_names):
    if not canonical_bvh_names:
        raise SystemExit(
            "Error: cond entry has no 'canonical_bvh_joint_names'; cannot label BVH "
            "previews consistently with the dataset. The cond entry appears malformed."
        )

    motions_dir = os.path.join(save_root, MOTION_DIR)
    bvhs_dir = os.path.join(save_root, BVHS_DIR)
    os.makedirs(motions_dir, exist_ok=True)
    os.makedirs(bvhs_dir, exist_ok=True)

    written = []
    for counter, result in enumerate(results, start=1):
        name = f"{object_type}_{result['action']}_{counter}"

        motion_path = os.path.join(motions_dir, name + '.npy')
        np.save(motion_path, result['motion'])

        # Export the visually faithful processed animation (export_anim), matching
        # how preprocessing writes BVH previews. Use the canonical BVH joint names
        # from cond so previews are labelled consistently with the dataset. (reorder
        # validates that the name count matches export_anim's augmented joint count.)
        anim_obj = result['export_anim']
        bvh_names = list(canonical_bvh_names)
        anim_obj, bvh_names = reorder_animation_to_dfs(anim_obj, bvh_names)
        bvh_path = os.path.join(bvhs_dir, name + '.bvh')
        BVH.save(
            bvh_path,
            anim_obj,
            bvh_names,
            frametime=result.get('frame_time', 1.0 / 24.0),
            positions=needs_bvh_position_channels(anim_obj),
        )
        written.append((motion_path, bvh_path))
        print(f"[process_new_animation] wrote {motion_path}", flush=True)
        print(f"[process_new_animation] wrote {bvh_path}", flush=True)

    return written


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process new animation clips for an existing AnyTop skeleton "
                    "without creating or modifying cond.npy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", required=True, type=str,
        help="A single animation file (FBX/GLB/GLTF) or a directory containing them.",
    )
    parser.add_argument(
        "--object-type", default=None, type=str,
        help="The existing cond object_type these animations belong to. "
             "Inferred from filenames when omitted.",
    )
    parser.add_argument(
        "--save-dir", default=None, type=str,
        help="Output root. motions/ and bvhs/ are created under it. Defaults to the "
             "input directory (or the input file's parent directory).",
    )
    parser.add_argument(
        "--dataset-dir", default=None, type=str,
        help="Dataset directory whose cond.npy holds the skeleton definition to reuse. "
             "Uses the default dataset directory when omitted.",
    )
    parser.add_argument(
        "--cond-path", default=None, type=str,
        help="Explicit path to cond.npy. Overrides --dataset-dir when set.",
    )
    parser.add_argument(
        "--tpose-path", default=None, type=str,
        help="Override the T-pose reference rest pose. Defaults to the path recorded "
             "in the cond entry. Required if that recorded path is missing.",
    )
    parser.add_argument(
        "--filter-min-length", default=10, type=int,
        help="Discard clips shorter than this many frames (matches preprocessing). "
             "A warning is printed for each discarded clip. Use 0 to disable. Default: 10.",
    )
    parser.add_argument(
        "--resample-min-length", default=20, type=int,
        help="Clips with >= filter-min-length but < this many frames are time-stretched "
             "to this length (matches preprocessing). A warning is printed for each "
             "resampled clip. Use 0 to disable. Default: 20.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.filter_min_length < 0:
        raise SystemExit("Error: --filter-min-length must be >= 0")
    if args.resample_min_length < 0:
        raise SystemExit("Error: --resample-min-length must be >= 0")
    if args.resample_min_length > 0 and args.resample_min_length <= args.filter_min_length:
        raise SystemExit("Error: --resample-min-length must be > --filter-min-length")

    anim_files, default_save_root = _collect_input_files(args.input)
    if len(anim_files) == 0:
        raise SystemExit(
            f"Error: no animation files ({', '.join(ANIM_EXTENSIONS)}) found in "
            f"--input '{args.input}'."
        )

    object_type = args.object_type
    if object_type is None:
        object_type = infer_object_type_from_filename(anim_files[0])
        if object_type is None:
            raise SystemExit(
                f"Error: could not infer --object-type from '{anim_files[0]}'. "
                f"Pass --object-type explicitly."
            )
        print(f"[process_new_animation] inferred object_type: {object_type}", flush=True)

    cond_path = args.cond_path or os.path.join(get_dataset_dir(args.dataset_dir or None), 'cond.npy')
    object_cond = _load_object_cond(cond_path, object_type)
    print(f"[process_new_animation] using skeleton definition from {cond_path}", flush=True)

    tpose_path = _resolve_tpose_reference(object_cond, object_type, args.tpose_path)
    print(f"[process_new_animation] T-pose reference: {tpose_path}", flush=True)

    tp = _build_character_constants(object_cond, object_type, tpose_path)

    save_root = args.save_dir or default_save_root
    canonical_bvh_names = object_cond.get('canonical_bvh_joint_names')

    max_joints = MAX_JOINTS
    all_results = []
    for file_path in anim_files:
        print(f"[process_new_animation] processing {file_path}", flush=True)
        results, max_joints = _process_one_file(
            file_path, object_type, tp, max_joints,
            args.filter_min_length, args.resample_min_length,
        )
        all_results.extend(results)

    if len(all_results) == 0:
        raise SystemExit("Error: no motions were produced from the input animation(s).")

    written = _write_outputs(save_root, object_type, all_results, canonical_bvh_names)

    print(
        f"\n[process_new_animation] done: {len(written)} clip(s) -> "
        f"{os.path.join(save_root, MOTION_DIR)} / {os.path.join(save_root, BVHS_DIR)}",
        flush=True,
    )


if __name__ == '__main__':
    try:
        main()
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception as exc:
        print(f"\n[process_new_animation] Error: {exc}", file=sys.stderr)
        sys.exit(1)
