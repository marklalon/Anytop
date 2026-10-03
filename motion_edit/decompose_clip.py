"""Decompose a motion into an Edit Package.

Usage (from ``Anytop/``)::

    # a generated sample, on a dataset skeleton (cond key as in cond.npy)
    python -m motion_edit.decompose_clip --npy out/sample0.npy --object_type truebones/zoo/Horse \
        --action_label "walk, forward" --action_group locomotion --is_loop --out outputs/edit_packages

    # a training clip, labels and loop flag from its dataset
    python -m motion_edit.decompose_clip --clip truebones/zoo:Horse_Walk.npy --out outputs/edit_packages

Writes ``<out>/<clip>.edit/``.  The skeleton profile and the species'
``contact_overrides.json`` / ``passive_overrides.json`` rows are read from the dataset whose ``cond.npy``
holds the cond key; ``--cond`` points at another cond file (a new skeleton
from ``tools/process_new_skeleton.py``), which then has no profile unless
``--profiles`` names one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ANYTOP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ANYTOP_ROOT not in sys.path:
    sys.path.insert(0, ANYTOP_ROOT)

import numpy as np  # noqa: E402

from motion_edit.decompose import (  # noqa: E402
    DEFAULT_STRETCH_FACTOR,
    StaleProfileError,
    contact_source,
    decompose_motion,
    passive_source,
    profile_subset,
)
from motion_edit.package import PACKAGE_SUFFIX  # noqa: E402
from motion_edit.profile.data import (  # noqa: E402
    CONTACT_OVERRIDES_FILE,
    PASSIVE_OVERRIDES_FILE,
    PROFILES_FILE,
    discover_sources,
    load_cond,
    load_profiles,
    load_species_sidecar,
    skeleton_hash,
)


def _find_dataset(cond_key: str):
    for source in discover_sources():
        cond = load_cond(source)
        if cond_key in cond:
            return source, cond[cond_key]
    return None, None


def _species_override(root: str, file_name: str, cond_key: str, cond_entry: dict):
    row = load_species_sidecar(root, file_name).get(cond_key)
    if row is None:
        return None, None
    if row.stale(skeleton_hash(cond_entry["parents"], cond_entry["offsets"])):
        return None, f"{file_name} row for {cond_key} is stale (skeleton_hash changed); ignored"
    return row.entries, None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source_args = parser.add_mutually_exclusive_group(required=True)
    source_args.add_argument("--npy", help="feature tensor (F, J, 12)")
    source_args.add_argument("--clip", help="<dataset namespace>:<motion file> of a training clip")
    parser.add_argument("--object_type", help="cond key of the skeleton (with --npy)")
    parser.add_argument("--cond", help="cond.npy holding --object_type (default: search the datasets)")
    parser.add_argument("--profiles", help=f"{PROFILES_FILE} to read the profile from (default: the dataset's)")
    parser.add_argument("--no_profile", action="store_true", help="decompose without a skeleton profile")
    parser.add_argument("--is_loop", action="store_true", help="the clip is a loop (with --npy)")
    parser.add_argument("--action_group", default="", help="with --npy")
    parser.add_argument("--action_label", default="", help="with --npy")
    parser.add_argument("--name", help="package name (default: the clip / file stem)")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--stretch_factor", type=float, default=DEFAULT_STRETCH_FACTOR)
    parser.add_argument("--no_fullbody_ik", action="store_true",
                        help="decode without full-body IK: keep the per-frame bone translations "
                             "(--stretch_factor is then unused)")
    parser.add_argument("--out", default="outputs/edit_packages", help="directory the package is written under")
    args = parser.parse_args(argv)

    from data_loaders.truebones.truebones_utils.param_utils import FPS

    notes = []
    dataset_root = None
    if args.clip:
        from data_loaders.truebones.data.dataset import _drop_loop_closing_frame
        from data_loaders.truebones.truebones_utils.motion_labels import load_motion_metadata

        namespace, _, motion = args.clip.partition(":")
        sources = {s.namespace: s for s in discover_sources()}
        if namespace not in sources or not motion:
            parser.error(f"--clip wants <namespace>:<motion file>; namespaces: {sorted(sources)}")
        source = sources[namespace]
        motion = motion if motion.endswith(".npy") else motion + ".npy"
        row = load_motion_metadata(source.root, only={motion}).get(motion)
        if row is None:
            parser.error(f"{motion} is not in {namespace}'s motion_metadata.json")
        species = str(row["object_type"])
        cond = load_cond(source)
        cond_key = next((k for k, e in cond.items()
                         if str(e.get("species_name") or k.rsplit("/", 1)[-1]) == species), None)
        if cond_key is None:
            parser.error(f"no cond entry for species {species}")
        cond_entry = cond[cond_key]
        features = np.load(os.path.join(source.motion_dir, motion))
        is_loop = bool(row.get("is_loop", False))
        if is_loop:
            # the on-disk loop may carry its closing key; a generated loop never does
            features = _drop_loop_closing_frame(features)
        action_group = str(row.get("action_group") or "")
        action_label = str(row.get("action_label") or "")
        dataset_root = source.root
        name = args.name or os.path.splitext(motion)[0]
    else:
        if not args.object_type:
            parser.error("--npy needs --object_type")
        cond_key = args.object_type
        if args.cond:
            cond_entry = np.load(args.cond, allow_pickle=True).item().get(cond_key)
        else:
            source, cond_entry = _find_dataset(cond_key)
            dataset_root = source.root if source else None
        if cond_entry is None:
            parser.error(f"cond key {cond_key!r} not found")
        features = np.load(args.npy)
        is_loop = bool(args.is_loop)
        action_group, action_label = args.action_group, args.action_label
        name = args.name or os.path.splitext(os.path.basename(args.npy))[0]

    profile = None
    if not args.no_profile:
        if args.profiles:
            with open(args.profiles, "r", encoding="utf-8") as handle:
                profile = (json.load(handle).get("profiles") or {}).get(cond_key)
        elif dataset_root:
            profile = load_profiles(dataset_root).get(cond_key)
    try:
        subset = profile_subset(profile, cond_entry, action_label)
    except StaleProfileError as exc:
        print(f"error: {cond_key}: {exc}", file=sys.stderr)
        return 2

    override, note = (_species_override(dataset_root, CONTACT_OVERRIDES_FILE, cond_key, cond_entry)
                      if dataset_root else (None, None))
    input_notes = [{"kind": "contacts", "message": n} for n in notes + ([note] if note else [])]
    contacts = contact_source(cond_entry, override)
    override, note = (_species_override(dataset_root, PASSIVE_OVERRIDES_FILE, cond_key, cond_entry)
                      if dataset_root else (None, None))
    if note:
        input_notes.append({"kind": "passive", "message": note})
    passive = passive_source(cond_entry, override)

    package = decompose_motion(
        features, cond_entry, object_type=cond_key, clip_name=name, is_loop=is_loop,
        action_group=action_group, action_label=action_label,
        fps=float(args.fps or FPS), stretch_factor=args.stretch_factor,
        fullbody_ik=not args.no_fullbody_ik, profile=subset, contacts=contacts, passive=passive,
        dataset_root=os.path.abspath(dataset_root) if dataset_root else None,
        notes=input_notes)
    out_dir = package.save(os.path.join(args.out, name + PACKAGE_SUFFIX))

    m = package.manifest
    print(f"Wrote {out_dir}: {m['frame_count']} frames, {m['joint_count']} joints, "
          f"loop={m['is_loop']}, profile={m['profile']['status']}, "
          f"plants={m['diagnostics']['plants']}, period={m['events'].get('period')}")
    for item in m["diagnostics"]["items"]:
        print(f"  [{item['kind']}] {item['message']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
