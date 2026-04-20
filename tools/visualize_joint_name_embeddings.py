"""
Joint Name Embedding Distribution Visualizer

Description:
    Loads joint names from cond.npy, encodes them with T5 (using the same
    preprocessing pipeline as the model), aggregates per-animal, then produces:
      1. t-SNE scatter plot — animals colored by species group

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
import re
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Preprocessing — mirrors T5Conditioner in model/conditioners.py exactly
# ---------------------------------------------------------------------------

REMOVE_PREFIXES = ["BN_Bip01", "Bip01", "BN", "NPC", "jt", "Sabrecat", "Elk"]
JAPANESE_WORDS = {
    "momo": "Thigh", "sippo": "Tail", "mune": "Chest", "hiza": "Knee",
    "hara": "Stomach", "ashi": "Leg", "hiji": "Elbow", "koshi": "Hips",
    "te": "Hand", "kubi": "Neck", "atama": "Head", "ago": "Jaw", "kata": "Shoulder",
}


def _remove_prefix(s: str) -> str:
    if s.startswith("Sabrecat"):
        s = s[:-6]
    for prefix in REMOVE_PREFIXES:
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s


def _split_and_replace(s: str) -> str:
    s = s.replace('ForeArm', 'Forearm')
    splitted = re.split(r"(?=[A-Z]|_)", s)
    new_splitted = []
    for part in splitted:
        clean = re.sub(r"[\d_]+", "", part)
        if not clean:
            continue
        elif clean in ("L", "l"):
            new_splitted.append("Left")
        elif clean in ("R", "r"):
            new_splitted.append("Right")
        elif len(clean) == 1:
            continue
        elif clean == "Tai":
            new_splitted.append("Tail")
        elif clean in JAPANESE_WORDS:
            new_splitted.append(JAPANESE_WORDS[clean])
        else:
            new_splitted.append(clean[0].upper() + clean[1:])
    sides = [w for w in new_splitted if w in ("Left", "Right")]
    rest  = [w for w in new_splitted if w not in ("Left", "Right")]
    return " ".join(sides + rest)


# Words that indicate non-anatomical rig/game nodes — skip joints containing any of these.
# "Nub" = structural end-effector bone; "end"+"site" = BVH end sites.
NON_ANATOMICAL = {"Dummy", "Projectile", "Brain", "Ponytail", "Node", "Nub"}


def _is_anatomical(processed: str) -> bool:
    words = set(processed.split())
    if words & NON_ANATOMICAL:
        return False
    if {"End", "Site"} <= words:
        return False
    return True


def preprocess_joint_name(name: str) -> str:
    return _split_and_replace(_remove_prefix(name))


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_names_t5(names: list[str], model_name: str, device: str) -> np.ndarray:
    """Mean-pool T5 encoder hidden states — same as T5Conditioner.forward()."""
    from transformers import T5EncoderModel, T5Tokenizer

    print(f"  Loading {model_name} …", flush=True)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name).to(device).eval()

    processed = [n if n else "" for n in names]

    BATCH = 128
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(processed), BATCH):
            batch = processed[i : i + BATCH]
            inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
            mask = inputs["attention_mask"]
            out = model(**inputs).last_hidden_state  # (B, seq, d)
            # masked mean-pool
            denom = mask.sum(dim=-1, keepdim=True).clamp(min=1).float()
            emb = (out * mask.unsqueeze(-1)).sum(dim=1) / denom
            all_embs.append(emb.cpu().float().numpy())

    return np.concatenate(all_embs, axis=0)  # (N, d)


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


def per_animal_embedding(cond: dict, t5_model: str, device: str) -> dict:
    """
    Returns {animal_name: (mean_emb, species_group)} where mean_emb is (d,).
    Encodes all joint names in a single batch for efficiency.
    """
    animals = sorted(cond.keys())

    # Preprocess and filter non-anatomical joints to mirror T5Conditioner.tokenize().
    # No deduplication: duplicate names (e.g. numbered sub-bones) each contribute
    # one embedding, consistent with how the model sees the data during training.
    animal_joint_lists: list[list[str]] = []
    for a in animals:
        raw_names = cond[a].get("joints_names") or []
        anatomical: list[str] = []
        for n in raw_names:
            proc = preprocess_joint_name(str(n))
            if _is_anatomical(proc):
                anatomical.append(proc)
        animal_joint_lists.append(anatomical)

    # Flatten for a single batched forward pass (names already preprocessed)
    flat_names = [n for lst in animal_joint_lists for n in lst]
    print(f"Encoding {len(flat_names)} anatomical joint names across {len(animals)} animals …")
    flat_embs = embed_names_t5(flat_names, t5_model, device)  # (total_joints, d)

    # Re-aggregate: mean per animal
    result = {}
    idx = 0
    for a, jlist in zip(animals, animal_joint_lists):
        k = len(jlist)
        if k == 0:
            continue
        mean_emb = flat_embs[idx : idx + k].mean(axis=0)
        group = cond[a].get("species_group", "unknown")
        result[a] = (mean_emb, group)
        idx += k

    return result


def plot_tsne(animal_embs: dict, output_dir: Path, perplexity: int):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    animals = list(animal_embs.keys())
    embs = np.stack([animal_embs[a][0] for a in animals])
    groups = [animal_embs[a][1] for a in animals]

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
    ax.set_title("t-SNE of Per-Animal Mean Joint-Name T5 Embeddings", fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    out = output_dir / "tsne_joint_embeddings.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


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

    animal_embs = per_animal_embedding(cond, args.t5_model, device)

    print("\n--- Generating plots ---")
    plot_tsne(animal_embs, output_dir, args.tsne_perplexity)

    print(f"\nDone. Outputs in: {output_dir.resolve()}")


if __name__ == "__main__":
    # Add project root to path so imports from Anytop/ work if needed
    sys.path.insert(0, str(Path(__file__).parent.parent))
    main()
