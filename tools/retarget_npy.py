"""
Feature-space cross-skeleton motion retargeting.

Retargets an animation (an AnyTop ``.npy`` feature file, or a raw
``.glb`` / ``.gltf`` / ``.fbx``) onto a target skeleton and writes the
target's ``(F, J, 12)`` motion features plus an inspection ``.bvh``.

* ``.npy`` sources: the source object_type is inferred from the filename
  and must be a known species; the source's cond entry is read from the
  training cond.
* Raw ``.glb/.gltf/.fbx`` sources: cond-free on the source side — the
  source skeleton is read straight from the file's bind pose.

For the native-space path -- "play this motion on that rig, no feature
round trip" -- use ``python tools/retarget_glb.py`` instead.

Usage examples:

    # Retarget a feature-space .npy onto the Dragon skeleton
    python tools/retarget_npy.py \\
        --source dataset/.../Horse___Run_01.npy --object_type Dragon

    # Retarget a raw GLB animation (source needs no cond entry)
    python tools/retarget_npy.py \\
        --source inputs/.../walk.glb --object_type Buffalo \\
        --output_dir outputs/my_retarget
"""
import argparse
import os
import sys

# ── Path setup ────────────────────────────────────────────────────────────────
# Ensure both the Anytop dir and its parent are on sys.path so the bare
# ``utils.*`` / ``data_loaders.*`` / ``motion_lib.*`` imports resolve
# regardless of CWD.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANYTOP_DIR = os.path.realpath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.dirname(ANYTOP_DIR)

for _p in [REPO_ROOT, ANYTOP_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Cross-skeleton motion retargeting — retarget an animation '
                    '(.npy/.glb/.fbx) onto a target skeleton.'
    )
    parser.add_argument(
        '--source', required=True, type=str,
        help='Path to source animation file (.npy / .glb / .fbx / .gltf).',
    )
    parser.add_argument(
        '--object_type', required=True, type=str,
        help='Target object type name (e.g. Horse, Buffalo, Dragon). '
             'Must match a key in the cond.npy dataset.',
    )
    parser.add_argument(
        '--cond_path', default=None, type=str,
        help='Optional path to an additional cond.npy file. Entries in this '
             'file are merged into the default training cond (not replaced). '
             'Useful for providing cond entries for skeletons not in the '
             'training set.',
    )
    parser.add_argument(
        '--output_dir', default=None, type=str,
        help='Output directory for retargeted .npy and inspection .bvh files. '
             'Default: Anytop/outputs/retarget_output',
    )

    args = parser.parse_args()

    # ── Resolve paths ─────────────────────────────────────────────────────
    source_path = os.path.abspath(args.source)
    if not os.path.isfile(source_path):
        parser.error(f'--source file not found: {source_path}')

    suffix = os.path.splitext(source_path)[1].lower()
    if suffix not in {'.npy', '.glb', '.fbx', '.gltf'}:
        parser.error(
            f'Unsupported source format "{suffix}". '
            f'Supported: .npy, .glb, .fbx, .gltf'
        )

    default_cond = os.path.join(
        ANYTOP_DIR, 'dataset', 'truebones', 'zoo',
        'truebones_processed', 'cond.npy',
    )
    if args.cond_path:
        extra_cond_path = os.path.abspath(args.cond_path)
        if not os.path.isfile(extra_cond_path):
            parser.error(f'--cond_path file not found: {extra_cond_path}')
    else:
        extra_cond_path = None

    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = os.path.join(ANYTOP_DIR, 'outputs', 'retarget_output')
    os.makedirs(output_dir, exist_ok=True)

    # ── Load cond ──────────────────────────────────────────────────────────
    if not os.path.isfile(default_cond):
        parser.error(
            f'Default training cond not found at {default_cond}. '
            f'Run preprocessing first or provide --cond_path.'
        )

    from data_loaders.truebones.truebones_utils.cond_schema import load_cond
    from data_loaders.truebones.truebones_utils.dataset_sources import (
        resolve_species_key,
        species_lookup_map,
    )

    cond_dict = load_cond(default_cond)

    if extra_cond_path:
        extra_cond = load_cond(extra_cond_path)
        # Merge: extra entries override/add to default, but don't replace
        # keys that already exist in default — we only add new object_types.
        for key, val in extra_cond.items():
            if key not in cond_dict:
                cond_dict[key] = val
        print(f'[retarget CLI] Merged {len(extra_cond)} extra cond entries '
              f'(total: {len(cond_dict)}).')

    # Bare name, namespace suffix, canonical key, or filename token all resolve.
    target_type = resolve_species_key(cond_dict, args.object_type)
    if target_type is None:
        parser.error(
            f'Target object_type "{args.object_type}" not found in cond. '
            f'Available: {sorted(cond_dict.keys())}'
        )

    tgt_cond = dict(cond_dict[target_type])
    max_joints = max(
        len(np.asarray(cond_dict[k]['parents']))
        for k in cond_dict
    )

    # ── Retarget ───────────────────────────────────────────────────────────
    from data_loaders.truebones.truebones_utils.features import (
        tpose_features_from_cond,
    )
    from data_loaders.truebones.truebones_utils.motion_process import (
        recover_bvh_export_animation_from_motion_np,
    )
    from motion_lib import BVH
    from utils.misc import infer_object_type_from_filename

    base_name = os.path.splitext(os.path.basename(source_path))[0]

    # Target rest-pose skeleton reconstructed from cond — no T-pose mesh read.
    tgt_tp = tpose_features_from_cond(tgt_cond, target_type)

    # Resolve FPS from target cond (used for BVH export frametime).
    fps = float(tgt_cond.get('fps', 30.0))

    if suffix == '.npy':
        # Feature-space .npy source: infer source object_type from filename,
        # then delegate to retarget_features_npy_to_target.
        from utils.retarget_pipeline import retarget_features_npy_to_target

        src_type = infer_object_type_from_filename(
            source_path,
            valid_types=species_lookup_map(cond_dict),
        )
        if src_type is None:
            parser.error(
                f'Could not infer source object_type from filename '
                f'"{base_name}". For .npy sources the filename must contain '
                f'a known object_type (e.g. Horse___Run_01.npy).'
            )
        if src_type == target_type:
            # Same skeleton — no retarget needed; copy source features as-is.
            print(f'[retarget CLI] Source and target are the same type '
                  f'("{src_type}") — skipping retarget.')
            target_features = np.load(source_path).astype(np.float32)
        else:
            src_cond = dict(cond_dict[src_type])
            src_features = np.load(source_path).astype(np.float32)

            print(f'[retarget CLI] Retargeting {src_type} → {target_type} '
                  f'(source: {src_features.shape})')

            target_features = retarget_features_npy_to_target(
                src_features,
                src_cond,
                src_type,
                tgt_tp,
                target_type,
                max_joints,
                source_tp=None,
                target_cond=tgt_cond,
            )
    else:
        # Raw animation file (.glb/.fbx/.gltf): cond-free on the source side.
        from utils.retarget_pipeline import retarget_animation_file_to_target

        print(f'[retarget CLI] Retargeting {source_path} → {target_type}')
        raw_src_type = infer_object_type_from_filename(
            source_path,
            valid_types=species_lookup_map(cond_dict),
        )

        target_features = retarget_animation_file_to_target(
            source_path,
            tgt_tp,
            target_type,
            max_joints,
            tgt_cond,
            source_object_type=raw_src_type,
        )

    if target_features is None:
        sys.exit(
            f'ERROR: Retargeting failed ({source_path} → {target_type}). '
            f'Check joint-name overlap between source and target skeletons.'
        )

    # ── Save outputs ───────────────────────────────────────────────────────
    out_npy = os.path.join(
        output_dir,
        f'_retargeted_to_{target_type}__{base_name}.npy',
    )
    np.save(out_npy, target_features)
    print(f'[retarget CLI] Retargeted features {target_features.shape} → {out_npy}')

    # Inspection BVH
    try:
        out_bvh = out_npy[:-4] + '.bvh'
        out_anim, joint_names, has_pos = recover_bvh_export_animation_from_motion_np(
            target_features,
            np.asarray(tgt_cond['parents'], dtype=np.int32),
            np.asarray(tgt_cond['offsets'], dtype=np.float32),
            list(tgt_cond.get('canonical_bvh_joint_names',
                               tgt_cond['joints_names'])),
            allow_infer=True,
            tpose_rest_rotations=tgt_tp.tpos_rots[0],
        )
        if out_anim is not None:
            BVH.save(
                out_bvh, out_anim, joint_names,
                frametime=1.0 / fps, positions=has_pos,
                order='auto',
            )
            print(f'[retarget CLI] Inspection BVH → {out_bvh}')
    except Exception as exc:
        print(f'[retarget CLI] WARNING: Failed to write inspection BVH: {exc}')


if __name__ == '__main__':
    main()
