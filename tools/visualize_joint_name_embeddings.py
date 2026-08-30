"""
Joint Name Embedding Distribution Visualizer

Description:
    Loads precomputed joint-name embeddings from cond.npy, aggregates per-animal,
    then produces:
      1. t-SNE scatter plot — per-animal mean joint-name embeddings, colored by group
      2. t-SNE scatter plot — species embeddings (species_emb), colored by group
      3. t-SNE scatter plot — additive species_emb + mean joint-name embeddings,
         colored by group

    A similarity report is also written so cosine neighbors and embedding norms
    can be inspected directly instead of relying only on 2D layout.

    Biologically similar animals should cluster together if joint name embeddings
    carry meaningful anatomical signal.

Usage:
    # From Anytop/ directory:
    python tools/visualize_joint_name_embeddings.py

    # Custom paths:
    python tools/visualize_joint_name_embeddings.py \\
        --cond-path dataset/truebones/zoo/truebones_processed/cond.npy \\
        --output-dir ./joint_emb_vis \\
        --tsne-perplexity 15
"""

import argparse
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

# Add project root to path so imports from Anytop/ work when running the script directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loaders.truebones.truebones_utils.physics_joint_annotation import build_joint_embedding_texts
from data_loaders.truebones.truebones_utils.cond_schema import load_cond as _load_cond
from data_loaders.truebones.truebones_utils.dataset_sources import bare_species_name

def l2_normalize(emb: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(emb))
    if norm <= eps:
        return emb.copy()
    return emb / norm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Canonical object_subset keys (see dataset_tags.CANONICAL_OBJECT_SUBSETS):
# the lower-cased first species tag. Kept in sync with the dataset code so the
# plot grouping matches how training/``--object_subsets`` bucket species.
GROUP_COLORS = {
    "quadruped":   "#4e79a7",
    "biped":       "#f28e2b",
    "multiped":    "#59a14f",
    "winged":      "#e15759",
    "serpentine":  "#76b7b2",
    "aquatic":     "#edc948",
    "drifting":    "#8c564b",
    "unknown":     "#b07aa1",
}

GROUP_ORDER = ["quadruped", "biped", "multiped", "winged", "serpentine", "aquatic", "drifting", "unknown"]


def load_cond(path: str) -> dict:
    # Schema-normalized: entries come back keyed '<namespace>/<species>', which is
    # also what the plot labels and group lookups use.
    return _load_cond(path)


def _embedding_texts_for_object(object_cond: dict) -> list[str]:
    meta = object_cond.get("joints_names_embs_meta") or {}
    embedding_texts = meta.get("embedding_texts")
    if embedding_texts is not None:
        return [str(text) for text in embedding_texts]
    return [str(text) for text in build_joint_embedding_texts(object_cond)]


def _embedding_source_description(cond: dict) -> str:
    model_names = []
    schema_versions = []
    dims = []
    species_model_names = []
    species_schema_versions = []
    species_dims = []
    for object_cond in cond.values():
        meta = object_cond.get("joints_names_embs_meta") or {}
        if meta.get("t5_name") is not None:
            model_names.append(str(meta["t5_name"]))
        if meta.get("schema_version") is not None:
            schema_versions.append(str(meta["schema_version"]))
        if meta.get("embedding_dim") is not None:
            dims.append(str(meta["embedding_dim"]))

        smeta = object_cond.get("species_emb_meta") or {}
        if smeta.get("t5_name") is not None:
            species_model_names.append(str(smeta["t5_name"]))
        if smeta.get("schema_version") is not None:
            species_schema_versions.append(str(smeta["schema_version"]))
        if smeta.get("embedding_dim") is not None:
            species_dims.append(str(smeta["embedding_dim"]))

    lines = ["source: cond.npy precomputed embeddings"]
    if model_names:
        lines.append(f"joint model={sorted(set(model_names))} schema={sorted(set(schema_versions))} dim={sorted(set(dims))}")
    if species_model_names:
        lines.append(f"species model={sorted(set(species_model_names))} schema={sorted(set(species_schema_versions))} dim={sorted(set(species_dims))}")
    return " | ".join(lines)


def _object_subset(object_cond: dict, a: str) -> str:
    """Canonical object_subset for an entry: the lower-cased first motion tag.

    Entries with no tags are a data error (every species must be registered in
    species_tags.jsonl), so fail loudly instead of silently plotting an
    "unknown" bucket.
    """
    tags = tuple(object_cond.get("species_tags") or ())
    if not tags:
        raise ValueError(
            f"{a!r} has empty species_tags in cond.npy; register it in "
            "species_tags.jsonl and re-stamp the dataset cond."
        )
    return tags[0].strip().lower()


def per_animal_embedding(cond: dict, normalize_means: bool) -> dict:
    """
    Returns per-animal diagnostics including the precomputed semantic joint-name mean embedding.
    """
    animals = sorted(cond.keys())

    result = {}
    for a in animals:
        object_cond = cond[a]
        raw_names = [str(name) for name in (object_cond.get("joints_names") or [])]
        if not raw_names:
            continue

        joint_embs = object_cond.get("joints_names_embs")
        if joint_embs is None:
            print(f"Warning: {a} is missing joints_names_embs in cond.npy, skipping.")
            continue

        joint_embs = np.asarray(joint_embs, dtype=np.float32)
        if joint_embs.ndim != 2 or joint_embs.shape[0] == 0 or joint_embs.shape[1] == 0:
            print(f"Warning: {a} has invalid joints_names_embs shape {joint_embs.shape}, skipping.")
            continue

        embedding_texts = _embedding_texts_for_object(object_cond)
        anatomical_joint_count = sum(1 for text in embedding_texts if text)

        mean_emb = joint_embs.mean(axis=0)
        plot_emb = l2_normalize(mean_emb) if normalize_means else mean_emb
        group = _object_subset(object_cond, a)
        result[a] = {
            "mean_emb": mean_emb,
            "plot_emb": plot_emb,
            "group": group,
            "raw_joint_count": len(raw_names),
            "anatomical_joint_count": anatomical_joint_count,
            "filtered_joint_count": len(raw_names) - anatomical_joint_count,
            "mean_norm": float(np.linalg.norm(mean_emb)),
        }

    return result


def per_animal_species_embedding(cond: dict, normalize: bool) -> dict:
    """
    Returns per-animal species embeddings (species_emb) from cond.npy.

    Each animal has exactly one species_emb vector encoding its motion-relevant
    body-plan descriptor (e.g. "Quadruped Small Stalking").
    """
    animals = sorted(cond.keys())

    result = {}
    for a in animals:
        object_cond = cond[a]
        species_emb = object_cond.get("species_emb")
        if species_emb is None:
            print(f"Warning: {a} is missing species_emb in cond.npy, skipping.")
            continue

        species_emb = np.asarray(species_emb, dtype=np.float32)
        if species_emb.ndim != 1 or species_emb.shape[0] == 0:
            print(f"Warning: {a} has invalid species_emb shape {species_emb.shape}, skipping.")
            continue

        species_meta = object_cond.get("species_emb_meta") or {}
        embedding_text = str(species_meta.get("embedding_text") or "")

        plot_emb = l2_normalize(species_emb) if normalize else species_emb.copy()
        group = _object_subset(object_cond, a)
        result[a] = {
            "species_emb": species_emb,
            "plot_emb": plot_emb,
            "group": group,
            "embedding_text": embedding_text,
            "norm": float(np.linalg.norm(species_emb)),
        }

    return result


def plot_tsne(animal_embs: dict, output_dir: Path, perplexity: int, suffix: str = "joint_embeddings"):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    animals = list(animal_embs.keys())
    embs = np.stack([animal_embs[a]["plot_emb"] for a in animals])
    groups = [animal_embs[a]["group"] for a in animals]

    print(f"Running t-SNE ({suffix}) …")
    coords = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                  init="pca", learning_rate="auto").fit_transform(embs)

    fig, ax = plt.subplots(figsize=(14, 10))
    
    texts = []
    for a, (x, y), g in zip(animals, coords, groups):
        color = GROUP_COLORS.get(g, GROUP_COLORS["unknown"])
        ax.scatter(x, y, c=color, s=80, zorder=2, edgecolors="white", linewidths=0.5)

        # Label with the bare species name (strip the <namespace>/ prefix).
        label = bare_species_name(a)
        text = ax.text(x, y, label, fontsize=7, ha="center", va="bottom", zorder=4)
        texts.append(text)

    # Use adjustText to automatically reposition overlapping labels
    try:
        from adjustText import adjust_text
        # adjustText v1.3 has a stray print() in _explode_repeated; suppress it.
        with redirect_stdout(io.StringIO()):
            adjust_text(texts, arrowprops=dict(arrowstyle="-", lw=0.5, color="gray", alpha=0.4),
                       ax=ax, expand_points=(1.5, 1.5), force_points=(0.5, 0.5),
                       avoid_points=False)
    except ImportError:
        print("Warning: adjustText not installed, labels may overlap. Install with: pip install adjustText")

    legend_handles = [
        mpatches.Patch(color=GROUP_COLORS.get(g, GROUP_COLORS["unknown"]), label=g)
        for g in GROUP_ORDER if any(grp == g for grp in groups)
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=9)

    # Choose title based on suffix
    if suffix == "species_emb":
        title = "t-SNE of L2-Normalized Per-Animal Species T5 Embeddings"
    elif suffix == "add_species_joint":
        title = "t-SNE of L2-Normalized Additive [Species + Joint-Name Mean] T5 Embeddings"
    else:
        title = "t-SNE of L2-Normalized Per-Animal Semantic Joint-Text T5 Embeddings"
    ax.set_title(title, fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    out = output_dir / f"tsne_{suffix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def build_additive_embeddings(species_embs: dict, joint_embs: dict, normalize: bool) -> dict:
    """Add species_emb to per-animal mean joint-name embedding.

    Matches training behaviour in InputProcess.forward where species_emb is
    projected and added to joints_embedded_names (broadcast per-joint).
    Both halves are independently L2-normalized before addition so neither
    dominates the sum.
    """
    common = sorted(set(species_embs.keys()) & set(joint_embs.keys()))
    if not common:
        print("Warning: no animals have both species_emb and joint name embeddings.")
        return {}

    result = {}
    for a in common:
        s_emb = l2_normalize(species_embs[a]["species_emb"])
        j_emb = l2_normalize(joint_embs[a]["mean_emb"])
        additive = s_emb + j_emb
        plot_emb = l2_normalize(additive) if normalize else additive
        result[a] = {
            "plot_emb": plot_emb,
            "group": species_embs[a]["group"],
        }
    return result


def save_similarity_report(animal_embs: dict, output_dir: Path, top_k: int = 8):
    animals = list(animal_embs.keys())
    report_lines = []
    raw_embs = {animal: animal_embs[animal]["mean_emb"] for animal in animals}
    mismatch_rows = []
    gap_rows = []

    cosine_values = []
    for i, animal_a in enumerate(animals):
        for animal_b in animals[i + 1 :]:
            emb_a = raw_embs[animal_a]
            emb_b = raw_embs[animal_b]
            cosine = float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b)))
            cosine_values.append(cosine)

    if cosine_values:
        report_lines.append(
            "pairwise cosine stats: "
            f"min={min(cosine_values):.4f} mean={np.mean(cosine_values):.4f} max={max(cosine_values):.4f}"
        )
        report_lines.append("")

    for animal in animals:
        current = animal_embs[animal]
        current_group = current["group"]
        report_lines.append(
            f"[{animal}] group={current['group']} raw_joints={current['raw_joint_count']} "
            f"anatomical={current['anatomical_joint_count']} filtered={current['filtered_joint_count']} "
            f"mean_norm={current['mean_norm']:.4f}"
        )

        neighbors = []
        for other in animals:
            if other == animal:
                continue
            emb_a = raw_embs[animal]
            emb_b = raw_embs[other]
            cosine = float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b)))
            euclidean = float(np.linalg.norm(emb_a - emb_b))
            neighbors.append((cosine, euclidean, other, animal_embs[other]["group"]))
        neighbors.sort(key=lambda item: item[0], reverse=True)

        same_group = [cosine for cosine, _euclidean, _other, group in neighbors if group == current_group]
        different_group = [cosine for cosine, _euclidean, _other, group in neighbors if group != current_group]
        group_gap = float(np.mean(same_group) - np.mean(different_group)) if same_group and different_group else 0.0
        gap_rows.append((group_gap, animal, current_group))

        top_neighbor = neighbors[0]
        if top_neighbor[3] != current_group:
            mismatch_rows.append((top_neighbor[0], animal, current_group, top_neighbor[2], top_neighbor[3]))

        report_lines.append(f"  same_group_minus_diff_group={group_gap:.4f}")
        for cosine, euclidean, other, other_group in neighbors[:top_k]:
            report_lines.append(
                f"  {other:<18} group={other_group:<10} cosine={cosine:.4f} euclidean={euclidean:.4f}"
            )
        report_lines.append("")

    report_lines.append("Suspicious Cross-Group Top-1 Neighbors")
    if mismatch_rows:
        for cosine, animal, group, other, other_group in sorted(mismatch_rows, reverse=True):
            report_lines.append(
                f"  {animal:<18} group={group:<10} top1={other:<18} top1_group={other_group:<10} cosine={cosine:.4f}"
            )
    else:
        report_lines.append("  none")

    report_lines.append("")
    report_lines.append("Lowest Same-Group Separation")
    for group_gap, animal, group in sorted(gap_rows)[:12]:
        report_lines.append(
            f"  {animal:<18} group={group:<10} same_group_minus_diff_group={group_gap:.4f}"
        )

    report_path = output_dir / "animal_similarity_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"  Saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize precomputed joint-name embedding distribution across animals.")
    parser.add_argument("--cond-path", default=None,
                        help="Path to cond.npy (auto-detected if omitted)")
    parser.add_argument("--output-dir", default="outputs/joint_emb_vis",
                        help="Directory to write PNGs into")
    parser.add_argument("--tsne-perplexity", type=int, default=15,
                        help="t-SNE perplexity (default: 15; try 5-30 for <100 animals)")
    parser.add_argument("--raw-means", action="store_true",
                        help="Use unnormalized embeddings for t-SNE. By default embeddings are L2-normalized first.")
    args = parser.parse_args()

    # Auto-detect cond.npy
    if args.cond_path is None:
        script_dir = Path(__file__).parent
        candidates = [
            script_dir.parent / "dataset/truebones/zoo/truebones_processed/cond.npy",
            Path("dataset/truebones/zoo/truebones_processed/cond.npy"),
        ]
        for c in candidates:
            if c.exists():
                args.cond_path = str(c)
                break
        if args.cond_path is None:
            sys.exit("Could not auto-detect cond.npy. Pass --cond-path explicitly.")

    print(f"cond.npy : {args.cond_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cond = load_cond(args.cond_path)
    print(f"Loaded {len(cond)} animals.")
    print(_embedding_source_description(cond))

    print("\n--- Generating plots ---")

    # 1. Joint-name mean embedding t-SNE (original plot)
    animal_embs_joint = per_animal_embedding(cond, normalize_means=not args.raw_means)
    plot_tsne(animal_embs_joint, output_dir, args.tsne_perplexity, suffix="joint_embeddings")

    # 2. Species embedding t-SNE
    animal_embs_species = per_animal_species_embedding(cond, normalize=not args.raw_means)
    if animal_embs_species:
        plot_tsne(animal_embs_species, output_dir, args.tsne_perplexity, suffix="species_emb")
    else:
        print("  Skipped species_emb t-SNE: no species embeddings found in cond.npy.")

    # 3. Additive species + joint-name mean embedding t-SNE
    if animal_embs_species and animal_embs_joint:
        additive_embs = build_additive_embeddings(animal_embs_species, animal_embs_joint,
                                                  normalize=not args.raw_means)
        if additive_embs:
            plot_tsne(additive_embs, output_dir, args.tsne_perplexity, suffix="add_species_joint")
    else:
        print("  Skipped additive t-SNE: need both species_emb and joint embeddings.")

    save_similarity_report(animal_embs_joint, output_dir)

    print(f"\nDone. Outputs in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
