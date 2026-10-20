"""Plot current species tag pairs from enabled dataset sidecars.

Run from the Anytop repository: python tools/species_tags_matrix.py
Counts are dataset-qualified species entries, not animation clips or biological taxa.
Enabled manifest sources and their species filters are respected; no cond.npy is read.
Only the absolute paths of the generated PNG and SVG are printed.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_loaders.truebones.truebones_utils.dataset_sources import load_datasets_manifest
from data_loaders.truebones.truebones_utils.dataset_tags import (
    SPECIES_BODY_PLANS,
    SPECIES_LOCOMOTIONS,
    load_species_tags,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "dataset/datasets_all.jsonl")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/species_tags_matrix")
    args = parser.parse_args()
    members = defaultdict(list)
    datasets = []
    seen = set()
    for source in load_datasets_manifest(args.manifest):
        sidecar = load_species_tags(source.root_path / "species_tags.jsonl")
        count = 0
        for name, tags in sidecar.items():
            if not source.accepts(name):
                continue
            qualified = source.key_for(name)
            if qualified in seen:
                raise ValueError(f"Duplicate species: {qualified}")
            seen.add(qualified)
            members[tags].append(qualified)
            count += 1
        datasets.append({"namespace": source.namespace, "count": count})

    rows, cols = list(SPECIES_BODY_PLANS), list(SPECIES_LOCOMOTIONS)
    matrix = np.array([[len(members[r, c]) for c in cols] for r in rows])
    total, occupied = int(matrix.sum()), int(np.count_nonzero(matrix))
    assert total == len(seen)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(17, 8.5))
    fig.patch.set_facecolor("#f8fafc")
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#e9edf2")
    norm = PowerNorm(gamma=0.55, vmin=0, vmax=max(1, int(matrix.max())))
    im = ax.imshow(np.ma.masked_equal(matrix, 0), cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(cols)), [f"{c}\n({matrix[:, j].sum()})" for j, c in enumerate(cols)],
                  rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(rows)), [f"{r}  ({matrix[i].sum()})" for i, r in enumerate(rows)])
    ax.set_xlabel("species tag2 · locomotion  (column totals)", labelpad=16)
    ax.set_ylabel("species tag1 · body plan  (row totals)", labelpad=16)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(len(rows)):
        for j in range(len(cols)):
            count = int(matrix[i, j])
            ax.text(j, i, str(count) if count else "–", ha="center", va="center",
                    color="white" if count and norm(count) > 0.56 else "#172b45" if count else "#9ba6b2",
                    fontsize=14 if count else 12, fontweight="bold" if count else "normal")
    fig.suptitle("Species tag combination matrix", x=0.06, ha="left", y=0.96,
                 fontsize=23, fontweight="bold", color="#172b45")
    fig.text(0.06, 0.89, f"{total} species entries  ·  {len(datasets)} enabled datasets  ·  "
             f"{occupied} observed combinations / {matrix.size} vocabulary pairs",
             color="#475569", fontsize=13)
    bar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.025)
    bar.set_label("Species count (nonlinear color scale)")
    fig.text(0.06, 0.055, "Gray / – = no species observed; this does not imply an invalid combination.\n"
             "One count per dataset-qualified species entry; same names across datasets count separately.",
             fontsize=10, color="#475569")
    fig.subplots_adjust(left=0.14, right=0.95, top=0.84, bottom=0.27)
    image_paths = [out_dir / f"species_tags_matrix.{suffix}" for suffix in ("png", "svg")]
    for path in image_paths:
        fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)

    report = {"counting_unit": "dataset-qualified species entry present in species_tags.jsonl",
              "tag_source": "species_tags.jsonl",
              "datasets": datasets, "total_species": total, "observed_combinations": occupied,
              "tag1": rows, "tag2": cols, "matrix": matrix.tolist(),
              "combinations": [{"tag1": r, "tag2": c, "count": len(names), "species": sorted(names)}
                               for (r, c), names in sorted(members.items()) if names]}
    (out_dir / "species_tags_matrix.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for path in image_paths:
        print(path)



if __name__ == "__main__":
    main()
