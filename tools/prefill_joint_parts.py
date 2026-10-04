#!/usr/bin/env python3
"""Prefill ``joint_parts.jsonl`` -- each joint's body part and ground contact.

The sidecar is the single source of the part / contact annotation: training
reads it as an auxiliary target, and nothing about it is stored in cond.npy.
This tool proposes it from the cond skeleton, so the workflow reads:

    1. build the dataset (cond.npy exists)
    2. python tools/prefill_joint_parts.py [--dataset NS]     <- this
    3. verify the proposals in dataset/review (serve.py, the parts page)

The rules live in ``data_loaders/truebones/truebones_utils/joint_parts.py``
(names, then inheritance down the tree, then geometry; contacts from
``infer_contact_joints``), the same function the review page's re-prefill
button calls.

Rules:
    * A species with no row gets a full proposal.
    * In an existing row, joints marked ``"src": "manual"`` are never touched,
      and a ``"reviewed": true`` row whose skeleton still matches is kept
      whole. Every other joint is re-proposed.
    * A row whose ``skeleton_sig`` no longer matches cond is stale: its manual
      joints that still exist are kept, the rest is re-proposed, and it loses
      its reviewed mark.
    * A row for a species no longer in cond is kept (it is someone's work) and
      reported.

The report lists every selected species, the ones with the most inherited
and geometry-derived joints first: that is the review order.

Usage:
    python tools/prefill_joint_parts.py [--dataset NS[,NS...]] [--dataset-dir DIR]
                                        [--filter GLOB[,GLOB...]] [--dry-run]
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
from collections import Counter
from pathlib import Path

ANYTOP_DIR = Path(__file__).resolve().parent.parent
if str(ANYTOP_DIR) not in sys.path:
    sys.path.insert(0, str(ANYTOP_DIR))
if str(ANYTOP_DIR.parent) not in sys.path:
    sys.path.insert(0, str(ANYTOP_DIR.parent))

from data_loaders.truebones.truebones_utils.cond_schema import load_cond  # noqa: E402
from data_loaders.truebones.truebones_utils.dataset_sources import (  # noqa: E402
    load_datasets_manifest,
    resolve_anytop_path,
)
from data_loaders.truebones.truebones_utils.joint_parts import (  # noqa: E402
    ALL_PART_LABELS,
    JOINT_PARTS_FILE,
    merge_prefill,
    prefill_joint_parts,
    read_joint_parts_sidecar,
    skeleton_signature,
    species_of,
    write_joint_parts_sidecar,
)

DEFAULT_MANIFEST = 'dataset/datasets.jsonl'


def _split_list(value):
    return [item.strip() for item in str(value or '').replace(';', ',').split(',') if item.strip()]


def _selected_dirs(args):
    """``[(label, processed dir)]`` the run covers."""
    if args.dataset_dir:
        path = Path(resolve_anytop_path(args.dataset_dir))
        return [(path.name, path)]
    sources = load_datasets_manifest(args.manifest)
    wanted = _split_list(args.dataset)
    known = {source.namespace for source in sources}
    unknown = [name for name in wanted if name not in known]
    if unknown:
        raise SystemExit(f'unknown dataset namespace(s) {unknown}; manifest lists {sorted(known)}')
    return [
        (source.namespace, source.root_path)
        for source in sources
        if not wanted or source.namespace in wanted
    ]


def _matches(species, patterns):
    return not patterns or any(fnmatch.fnmatch(species.lower(), pattern.lower()) for pattern in patterns)


def prefill_dataset(label, processed_dir, patterns, dry_run):
    """Prefill one processed dataset; returns its report rows."""
    cond = load_cond(processed_dir / 'cond.npy')
    sidecar_path = processed_dir / JOINT_PARTS_FILE
    existing = read_joint_parts_sidecar(sidecar_path)

    rows = {}
    report = []
    cond_species = set()
    for entry in cond.values():
        species = species_of(entry)
        cond_species.add(species)
        old_row = existing.get(species)
        if not _matches(species, patterns):
            if old_row is not None:
                rows[species] = old_row
            continue
        sig = skeleton_signature(entry['joints_names'], entry['parents'])
        row, changed = merge_prefill(old_row, prefill_joint_parts(entry), species, sig)
        rows[species] = row
        if old_row is None:
            status = 'new'
        elif old_row['skeleton_sig'] != sig:
            status = 'stale'
        elif old_row['reviewed']:
            status = 'reviewed'
        else:
            status = 'updated' if changed else 'same'
        sources = Counter(joint['src'] for joint in row['joints'].values())
        parts = Counter(joint['part'] for joint in row['joints'].values())
        report.append({
            'dataset': label,
            'species': species,
            'status': status,
            'joints': len(row['joints']),
            'changed': len(changed),
            'inherit': sources['inherit'],
            'geometry': sources['geometry'],
            'manual': sources['manual'],
            'contact': sum(joint['contact'] for joint in row['joints'].values()),
            'parts': parts,
        })

    orphans = [species for species in existing if species not in cond_species]
    for species in orphans:
        rows[species] = existing[species]
    if orphans:
        print(f'[WARN] {label}: rows kept for species no longer in cond: {", ".join(orphans)}')

    if not dry_run:
        write_joint_parts_sidecar(sidecar_path, rows.values())
        print(f'[OK] {label}: wrote {len(rows)} row(s) to {sidecar_path}')
    return report


def print_report(report):
    report = sorted(report, key=lambda item: (-(item['geometry'] * 4 + item['inherit']), item['dataset'], item['species']))
    header = f'{"dataset":22s} {"species":30s} {"status":8s} {"J":>4s} {"chg":>4s} {"inh":>4s} {"geo":>4s} {"man":>4s} {"cnt":>4s}  parts'
    print(header)
    print('-' * len(header))
    for item in report:
        parts = ' '.join(
            f'{part}:{item["parts"][part]}' for part in ALL_PART_LABELS if item['parts'][part]
        )
        print(
            f'{item["dataset"][:22]:22s} {item["species"][:30]:30s} {item["status"]:8s} '
            f'{item["joints"]:4d} {item["changed"]:4d} {item["inherit"]:4d} {item["geometry"]:4d} '
            f'{item["manual"]:4d} {item["contact"]:4d}  {parts}'
        )
    statuses = Counter(item['status'] for item in report)
    print(f'\n{len(report)} species: ' + ', '.join(f'{name} {count}' for name, count in sorted(statuses.items())))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--manifest', default=DEFAULT_MANIFEST, help=f'Dataset manifest (default: {DEFAULT_MANIFEST}).')
    parser.add_argument('--dataset', default='', help='Comma-separated manifest namespace(s) (default: every enabled dataset).')
    parser.add_argument('--dataset-dir', default='', help='A processed dataset dir to use instead of the manifest.')
    parser.add_argument('--filter', dest='species_filter', default='', help='Comma/semicolon-separated case-insensitive glob(s) of species.')
    parser.add_argument('--dry-run', action='store_true', help='Propose and report, write nothing.')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    patterns = _split_list(args.species_filter)
    report = []
    for label, processed_dir in _selected_dirs(args):
        report.extend(prefill_dataset(label, processed_dir, patterns, args.dry_run))
    print_report(report)
    return 0


if __name__ == '__main__':
    sys.exit(main())
