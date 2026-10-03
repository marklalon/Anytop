"""Build ``skeleton_profiles.json`` and its review report for every processed dataset.

Usage (from ``Anytop/``)::

    python -m motion_edit.build_profiles                       # every dataset, every species
    python -m motion_edit.build_profiles --dataset truebones/zoo --species Alligator --species "Dog*"

Writes ``<processed root>/skeleton_profiles.json`` and
``<processed root>/skeleton_profiles_report.md``.  A filtered run replaces only
the species it built and keeps every other row of the existing file.

User sidecars read from the same root (rows keyed by cond key, joints by name,
each with the ``skeleton_hash`` it was made against):

* ``contact_overrides.json``    ``{"add": [...], "remove": [...]}``
* ``passive_overrides.json``    ``{"add": [...], "remove": [...]}``  (passive joints beyond the named defaults)
"""

from __future__ import annotations

import argparse
import dataclasses
import fnmatch
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ANYTOP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ANYTOP_ROOT not in sys.path:
    sys.path.insert(0, ANYTOP_ROOT)

from motion_edit.profile import build as builder  # noqa: E402
from motion_edit.profile.data import (  # noqa: E402
    CONTACT_OVERRIDES_FILE,
    PASSIVE_OVERRIDES_FILE,
    PROFILES_FILE,
    REPORT_FILE,
    discover_sources,
    load_cond,
    load_species_sidecar,
    species_motion_names,
)
from motion_edit.profile.report import render_report  # noqa: E402


def _build_job(source, cond_key, cond_entry, motion_rows, contact_override, passive_override):
    profile, findings = builder.build_species_profile(
        source, cond_key, cond_entry, motion_rows, contact_override, passive_override)
    return source.namespace, cond_key, profile, dataclasses.asdict(findings)


def _read_profiles_file(root: str) -> dict:
    path = os.path.join(root, PROFILES_FILE)
    if not os.path.isfile(path):
        return {"schema_version": builder.SCHEMA_VERSION, "profiles": {}, "findings": {}}
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("schema_version") != builder.SCHEMA_VERSION:
        return {"schema_version": builder.SCHEMA_VERSION, "profiles": {}, "findings": {}}
    data.setdefault("findings", {})
    return data


def _write_json(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    os.replace(tmp, path)


def _matches(cond_key: str, species_name: str, patterns: list[str]) -> bool:
    if not patterns:
        return True
    return any(fnmatch.fnmatchcase(species_name, p) or fnmatch.fnmatchcase(cond_key, p)
               for p in patterns)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", action="append", default=[],
                        help="namespace from dataset/datasets.jsonl (repeatable; default all)")
    parser.add_argument("--species", action="append", default=[],
                        help="species name or cond key glob (repeatable; default all)")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    args = parser.parse_args(argv)

    from data_loaders.truebones.truebones_utils.motion_labels import load_motion_metadata

    sources = discover_sources()
    if args.dataset:
        unknown = set(args.dataset) - {s.namespace for s in sources}
        if unknown:
            parser.error(f"unknown dataset namespace(s): {sorted(unknown)}")
        sources = [s for s in sources if s.namespace in args.dataset]

    jobs = []
    conds: dict[str, dict] = {}
    for source in sources:
        cond = load_cond(source)
        metadata = load_motion_metadata(source.root)
        contact_overrides = load_species_sidecar(source.root, CONTACT_OVERRIDES_FILE)
        passive_overrides = load_species_sidecar(source.root, PASSIVE_OVERRIDES_FILE)
        for cond_key, entry in cond.items():
            species = str(entry.get("species_name") or cond_key.rsplit("/", 1)[-1])
            subset = builder.cond_subset(entry)
            conds[cond_key] = subset
            if not _matches(cond_key, species, args.species):
                continue
            rows = {name: metadata[name] for name in species_motion_names(metadata, species)}
            jobs.append((source, cond_key, subset, rows,
                         contact_overrides.get(cond_key), passive_overrides.get(cond_key)))
    if not jobs:
        print("No species matched.")
        return 1

    print(f"Building {len(jobs)} profile(s) with {args.workers} worker(s)...", flush=True)
    files = {s.namespace: _read_profiles_file(s.root) for s in sources}
    started = time.time()
    results = []
    if args.workers <= 1:
        for job in jobs:
            results.append(_build_job(*job))
            print(f"  {results[-1][1]}: {results[-1][3]['clips']} clips", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_build_job, *job) for job in jobs]
            for done, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results.append(result)
                print(f"  [{done}/{len(jobs)}] {result[1]}: {result[3]['clips']} clips", flush=True)

    for namespace, cond_key, profile, findings in results:
        files[namespace]["profiles"][cond_key] = profile
        files[namespace]["findings"][cond_key] = findings

    # Clip-less species borrow from every profile with data, across all datasets
    # (a --dataset filter limits what is built, not who can donate).
    if any(profile.get("needs_fallback") for _, _, profile, _ in results):
        built = {s.namespace for s in sources}
        for source in discover_sources():
            if source.namespace not in built:
                conds.update({k: builder.cond_subset(e) for k, e in load_cond(source).items()})
        donor_files = {s.namespace: files.get(s.namespace) or _read_profiles_file(s.root)
                       for s in discover_sources()}
        donors = [(conds[key], prof) for data in donor_files.values()
                  for key, prof in data["profiles"].items()
                  if key in conds and prof.get("joints") and not prof.get("fallback")]
        for namespace, cond_key, profile, _ in results:
            if profile.get("needs_fallback"):
                builder.apply_fallback(profile, conds[cond_key], donors)

    for source in sources:
        data = files[source.namespace]
        data["profiles"] = dict(sorted(data["profiles"].items()))
        data["findings"] = dict(sorted(data["findings"].items()))
        _write_json(os.path.join(source.root, PROFILES_FILE), data)
        with open(os.path.join(source.root, REPORT_FILE), "w", encoding="utf-8", newline="\n") as handle:
            handle.write(render_report(source.namespace, data["findings"]))
        print(f"Wrote {os.path.join(source.root, PROFILES_FILE)} "
              f"({len(data['profiles'])} species) and {REPORT_FILE}")
    print(f"Done in {time.time() - started:.0f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
