"""
Joint Name Embedding Distribution Visualizer

Description:
    Loads joint names from cond.npy, encodes them with T5 through the actual
    T5Conditioner pipeline, aggregates per-animal, then produces:
      1. t-SNE scatter plot — animals colored by species group

    A similarity report is also written so cosine neighbors and embedding norms
    can be inspected directly instead of relying only on 2D layout.

    Biologically similar animals should cluster together if joint name embeddings
    carry meaningful anatomical signal.

Usage:
    # From Anytop/ directory:
    python tools/visualize_joint_name_embeddings.py

    # Custom paths / model:
    python tools/visualize_joint_name_embeddings.py \\
        --cond-path dataset/truebones/zoo/truebones_processed/cond.npy \\
        --t5-model t5-small \\
        --output-dir ./joint_emb_vis \\
        --tsne-perplexity 15
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path so imports from Anytop/ work when running the script directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loaders.truebones.truebones_utils.physics_joint_annotation import build_joint_embedding_texts
from model.conditioners import T5Conditioner

def l2_normalize(emb: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(emb))
    if norm <= eps:
        return emb.copy()
    return emb / norm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

GROUP_COLORS = {
    "quadruped":  "#4e79a7",
    "biped":      "#f28e2b",
    "flying":     "#59a14f",
    "millipede":  "#e15759",
    "snake":      "#76b7b2",
    "fish":       "#edc948",
    "unknown":    "#b07aa1",
}

GROUP_ORDER = ["quadruped", "biped", "flying", "millipede", "snake", "fish", "unknown"]


def load_cond(path: str) -> dict:
    raw = np.load(path, allow_pickle=True).item()
    return raw


def per_animal_embedding(cond: dict, t5_model: str, device: str, normalize_means: bool) -> dict:
    """
    Returns per-animal diagnostics including the exact semantic joint-text mean embedding.
    """
    animals = sorted(cond.keys())
    print(f"Loading {t5_model} through T5Conditioner …")
    conditioner = T5Conditioner(
        name=t5_model,
        finetune=False,
        word_dropout=0.0,
        normalize_text=False,
        device=device,
    )

    result = {}
    for a in animals:
        raw_names = [str(name) for name in (cond[a].get("joints_names") or [])]
        if not raw_names:
            continue

        embedding_texts = build_joint_embedding_texts(cond[a])
        anatomical_joint_count = sum(1 for text in embedding_texts if text)

        with torch.no_grad():
            joint_inputs = conditioner.tokenize_entries(embedding_texts)
            joint_embs = conditioner(joint_inputs).detach().cpu().float().numpy()

        mean_emb = joint_embs.mean(axis=0)
        plot_emb = l2_normalize(mean_emb) if normalize_means else mean_emb
        group = cond[a].get("species_group", "unknown")
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


def plot_tsne(animal_embs: dict, output_dir: Path, perplexity: int):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    animals = list(animal_embs.keys())
    embs = np.stack([animal_embs[a]["plot_emb"] for a in animals])
    groups = [animal_embs[a]["group"] for a in animals]

    print("Running t-SNE …")
    coords = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                  init="pca", learning_rate="auto").fit_transform(embs)

    fig, ax = plt.subplots(figsize=(14, 10))
    
    texts = []
    for a, (x, y), g in zip(animals, coords, groups):
        color = GROUP_COLORS.get(g, GROUP_COLORS["unknown"])
        ax.scatter(x, y, c=color, s=80, zorder=2, edgecolors="white", linewidths=0.5)
        
        # Create text annotation (position will be adjusted later)
        text = ax.text(x, y, a, fontsize=7, ha="center", va="bottom", zorder=4)
        texts.append(text)

    # Use adjustText to automatically reposition overlapping labels
    try:
        from adjustText import adjust_text
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
    ax.set_title("t-SNE of L2-Normalized Per-Animal Semantic Joint-Text T5 Embeddings", fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    out = output_dir / "tsne_joint_embeddings.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


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
        description="Visualize T5 joint-name embedding distribution across animals.")
    parser.add_argument("--cond-path", default=None,
                        help="Path to cond.npy (auto-detected if omitted)")
    parser.add_argument("--t5-model", default="t5-small",
                        choices=["t5-small", "t5-base", "t5-large",
                                 "google/flan-t5-small", "google/flan-t5-base"],
                        help="T5 variant to use for encoding (default: t5-small)")
    parser.add_argument("--output-dir", default="outputs/joint_emb_vis",
                        help="Directory to write PNGs into")
    parser.add_argument("--tsne-perplexity", type=int, default=15,
                        help="t-SNE perplexity (default: 15; try 5-30 for <100 animals)")
    parser.add_argument("--device", default=None,
                        help="Torch device (auto: cuda if available, else cpu)")
    parser.add_argument("--raw-means", action="store_true",
                        help="Use unnormalized per-animal mean embeddings for t-SNE. By default means are L2-normalized first.")
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

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"cond.npy : {args.cond_path}")
    print(f"T5 model : {args.t5_model}  |  device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cond = load_cond(args.cond_path)
    print(f"Loaded {len(cond)} animals.")

    animal_embs = per_animal_embedding(cond, args.t5_model, device, normalize_means=not args.raw_means)

    print("\n--- Generating plots ---")
    plot_tsne(animal_embs, output_dir, args.tsne_perplexity)
    save_similarity_report(animal_embs, output_dir)

    print(f"\nDone. Outputs in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
