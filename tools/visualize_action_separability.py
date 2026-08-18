"""Diagnose how well an AnyTop checkpoint's internal representation separates
different action types (action_tags) and species groups (objects_subset).

Motivation
----------
AnyTop has NO explicit action-class conditioning: action type only enters
through loop / playspeed / global-energy scalars and the joint-name text
embedding. So "do different actions overlap in the same activation region?"
is tested by probing intermediate decoder-layer activations, labelling each
clip by its primary action tag, and measuring separability.

The same analysis is also performed for species groups (objects_subset, e.g.
quadruped / biped / serpentine) to assess whether different broader creature
categories are linearly separable in the latent space.

Pipeline
--------
1. Pick N random clips from the configured objects_subset.
2. Feed each clip to the model at a fixed diffusion timestep (default t=0,
   i.e. the clean normalized motion) and capture every decoder layer's
   latent via forward hooks; masked-mean-pool over valid joints + frames
   -> one D-dim vector per clip per layer.
3. Quantify separability of both action_tags and species groups:
      * leave-one-out kNN accuracy  (no model fitting; non-linear sensitive)
      * silhouette score            (cluster compactness vs separation)
   Two tables are printed (one per label type).
4. Render two UMAP (fallback t-SNE) 2-D scatters for the chosen layer:
   one coloured by action tag, one coloured by species group.

Usage
-----
    python tools/visualize_action_separability.py \
        --ckpt  save/quadropeds_final_v1/model000100000.pt \
        --objects_subset all \
        --num_clips 100 --device 0 --out_dir save/_action_separability
"""
import argparse
import os
import sys

# Make bare `data_loaders.*` / `utils.*` / `sample.*` imports work when run as
# a script (mirror sample/generate.py's path setup: repo-root then Anytop).
_ANYTOP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_ANYTOP_ROOT))
sys.path.insert(0, _ANYTOP_ROOT)

import numpy as np
import torch

from data_loaders.get_data import get_dataset
from data_loaders.tensors import truebones_batch_collate
from data_loaders.truebones.data.dataset import (
    _normalize_motion_action_tags,
)
from data_loaders.truebones.truebones_utils.dataset_tags import dataset_tags
from sample.generate import prepare_generation_runtime
from utils.model_util import unwrap_anytop_model
from utils.parser_util import generate_args

# Reverse mapping: object_type → objects_subset name (e.g. "Horse" → "quadruped")
_OBJECT_TYPE_TO_SUBSET = {}
for _subset_name, _type_set in dataset_tags().object_subsets.items():
    if _subset_name == "all":
        continue
    for _t in _type_set:
        _OBJECT_TYPE_TO_SUBSET[_t.lower()] = _subset_name


# ----------------------------------------------------------------------------
# Action-tag -> single primary label
# ----------------------------------------------------------------------------
def primary_action_tag(raw_tags):
    """Reduce a clip's (possibly multi-)action_tags to one display label.

    Returns the alphabetically first tag when multiple tags are present.
    """
    tags = _normalize_motion_action_tags(raw_tags)  # set[str], lowercased
    if not tags:
        return "unknown"
    return sorted(tags)[0]


# ----------------------------------------------------------------------------
# Activation extraction
# ----------------------------------------------------------------------------
def _move_y_to_device(y, device):
    moved = {}
    for key, value in y.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


@torch.no_grad()
def extract_layer_activations(runtime, dataset, names, batch_size, timestep, device, seed):
    """Run the `names` clips through the model, capturing per-layer pooled latents.

    Returns
    -------
    layer_feats    : dict[int, np.ndarray]   layer_idx -> (N, D)
    action_labels  : list[str]               primary action tag per clip (len N)
    species_labels : list[str]               object_type per clip (len N)
    """
    # Reseed so random crop inside prepare_sample_by_name is deterministic
    # for the given name list (same seed -> same crops -> reproducible input).
    np.random.seed(seed)
    motion_dataset = dataset.motion_dataset

    model = unwrap_anytop_model(runtime.model)
    model.eval()
    diffusion = runtime.diffusion
    decoder_layers = model.seqTransDecoder.layers
    num_layers = len(decoder_layers)

    captured = {}

    def make_hook(layer_idx):
        def hook(_module, _inp, output):
            # output: (1 + T, B, J, D); index 0 is the tpos frame.
            captured[layer_idx] = output.detach()
        return hook

    handles = [decoder_layers[i].register_forward_hook(make_hook(i)) for i in range(num_layers)]

    layer_feats = {i: [] for i in range(num_layers)}
    action_labels = []
    species_labels = []
    try:
        for start in range(0, len(names), batch_size):
            batch_names = names[start:start + batch_size]
            items = [motion_dataset.prepare_sample_by_name(nm) for nm in batch_names]
            motion, cond = truebones_batch_collate(items)

            x = motion.to(device)                       # (B, J, 13, T)
            y = _move_y_to_device(cond["y"], device)
            bs = x.shape[0]

            if timestep > 0:
                t = torch.full((bs,), int(timestep), device=device, dtype=torch.long)
                x_in = diffusion.q_sample(x, t)
            else:
                t = torch.zeros(bs, device=device, dtype=torch.long)
                x_in = x

            captured.clear()
            model(x_in, t, y=y)

            n_joints = torch.as_tensor(y["n_joints"], device=device).reshape(-1)  # (B,)
            for layer_idx in range(num_layers):
                act = captured[layer_idx]               # (1+T, B, J, D)
                joint_count = act.shape[2]
                joint_valid = (
                    torch.arange(joint_count, device=device)[None, :] < n_joints[:, None]
                ).float()                               # (B, J)
                # drop tpos frame, mean over motion frames -> (B, J, D)
                per_joint = act[1:].mean(dim=0)         # (B, J, D)
                w = joint_valid.unsqueeze(-1)           # (B, J, 1)
                pooled = (per_joint * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-6)  # (B, D)
                layer_feats[layer_idx].append(pooled.float().cpu().numpy())

            raw_tag_list = y.get("action_tags") or [None] * bs
            for raw_tags in raw_tag_list:
                action_labels.append(primary_action_tag(raw_tags))

            species_list = y.get("object_type") or ["unknown"] * bs
            for sp in species_list:
                sp_key = str(sp).strip().lower() if sp is not None else "unknown"
                species_labels.append(_OBJECT_TYPE_TO_SUBSET.get(sp_key, sp_key))
    finally:
        for h in handles:
            h.remove()

    layer_feats = {i: np.concatenate(v, axis=0) for i, v in layer_feats.items()}
    return layer_feats, action_labels, species_labels


# ----------------------------------------------------------------------------
# Separability metrics
# ----------------------------------------------------------------------------
def separability_metrics(features, labels):
    """Leave-one-out kNN accuracy + silhouette score on standardized features."""
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(labels)
    X = StandardScaler().fit_transform(features)
    n = len(y)
    classes, counts = np.unique(y, return_counts=True)

    # Leave-one-out kNN (k=5, excluding self). Majority vote of nearest others.
    k = min(5, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, neigh = nn.kneighbors(X)
    correct = 0
    for i in range(n):
        others = [j for j in neigh[i] if j != i][:k]
        votes = y[others]
        vals, cnts = np.unique(votes, return_counts=True)
        if vals[np.argmax(cnts)] == y[i]:
            correct += 1
    knn_acc = correct / n

    if len(classes) >= 2 and counts.min() >= 1 and n > len(classes):
        sil = float(silhouette_score(X, y))
    else:
        sil = float("nan")

    chance = float(counts.max() / n)  # majority-class baseline
    return {"knn_acc": knn_acc, "silhouette": sil, "chance": chance, "n_classes": len(classes)}


# ----------------------------------------------------------------------------
# 2-D embedding for the scatter plot
# ----------------------------------------------------------------------------
def embed_2d(features, seed=0):
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(features)
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(
            n_neighbors=min(15, max(2, len(X) - 1)),
            min_dist=0.1,
            n_components=2,
            random_state=seed,
        )
        return reducer.fit_transform(X), "UMAP"
    except Exception:
        from sklearn.manifold import TSNE

        reducer = TSNE(
            n_components=2,
            perplexity=min(30, max(5, len(X) // 4)),
            init="pca",
            random_state=seed,
        )
        return reducer.fit_transform(X), "t-SNE"


def plot_single(emb, labels, title, method, suptitle, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_tags = sorted(set(labels))
    cmap = plt.get_cmap("tab20" if len(all_tags) > 10 else "tab10")
    color = {tag: cmap(i % cmap.N) for i, tag in enumerate(all_tags)}

    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    labels = np.asarray(labels)
    for tag in all_tags:
        m = labels == tag
        if not m.any():
            continue
        ax.scatter(emb[m, 0], emb[m, 1], s=28, alpha=0.8, color=color[tag], label=tag)
    ax.set_title(title)
    ax.set_xlabel(f"{method}-1")
    ax.set_ylabel(f"{method}-2")
    fig.legend(loc="upper center", ncol=min(len(all_tags), 8), fontsize=9)
    fig.suptitle(suptitle, y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[plot] saved -> {out_path}")


# ----------------------------------------------------------------------------
def load_runtime(ckpt_path, device):
    args = generate_args(argv=["--model_path", ckpt_path, "--device", str(device)])
    runtime = prepare_generation_runtime(args)
    return runtime, args


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="checkpoint .pt to evaluate")
    ap.add_argument("--num_clips", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--objects_subset", default="all")
    ap.add_argument("--split", default="train")
    ap.add_argument("--timestep", type=int, default=0,
                    help="diffusion t to probe at (0 = clean motion). >0 noises via q_sample.")
    ap.add_argument("--plot_layer", type=int, default=-1, help="decoder layer to plot (-1 = last)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out_dir", default="outputs/action_separability")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # --- Load checkpoint ---
    print(f"\n=== Loading checkpoint ===\n{args.ckpt}")
    rt, model_args = load_runtime(args.ckpt, args.device)

    # --- Build the clip set ---
    rng = np.random.default_rng(args.seed)
    ds = get_dataset(
        num_frames=int(getattr(model_args, "num_frames", 60)),
        split=args.split,
        temporal_window=int(getattr(model_args, "temporal_window", 31)),
        balanced=False,
        objects_subset=args.objects_subset,
    )
    all_names = list(ds.motion_dataset.name_list)
    n = min(args.num_clips, len(all_names))
    names = list(rng.choice(np.asarray(all_names, dtype=object), size=n, replace=False))
    print(f"\nSampled {n} clips from '{args.objects_subset}' split='{args.split}'.")

    # --- Extract activations ---
    print("\n[extract] ...")
    feats, action_labels, species_labels = extract_layer_activations(
        rt, ds, names, args.batch_size, args.timestep, device, args.seed
    )

    # Report the tag / species distribution.
    tags, tag_counts = np.unique(np.asarray(action_labels), return_counts=True)
    print("\nAction-tag distribution in the probe set:")
    for tag, c in sorted(zip(tags, tag_counts), key=lambda kv: -kv[1]):
        print(f"  {tag:<24s} {c}")
    if args.objects_subset == "all":
        sps, sp_counts = np.unique(np.asarray(species_labels), return_counts=True)
        print("\nSpecies (objects_subset) distribution in the probe set:")
        for sp, c in sorted(zip(sps, sp_counts), key=lambda kv: -kv[1]):
            print(f"  {sp:<24s} {c}")

    num_layers = len(feats)

    # --- Per-layer separability table: action tags ---
    print("\n" + "=" * 55)
    print(f"{'layer':>5s} | {'kNN acc':>9s} {'silhouette':>9s}  <- ACTION tags")
    print("-" * 55)
    action_metrics = {}
    for layer in range(num_layers):
        m = separability_metrics(feats[layer], action_labels)
        action_metrics[layer] = m
        print(f"{layer:>5d} | {m['knn_acc']:>9.3f} {m['silhouette']:>9.3f}")
    print("-" * 55)
    print(f"(majority-class chance kNN baseline = {action_metrics[0]['chance']:.3f}, "
          f"n_classes = {action_metrics[0]['n_classes']})")
    print("=" * 55)

    # --- Per-layer separability table: species (objects_subset) ---
    species_metrics = {}
    if args.objects_subset == "all":
        print("\n" + "=" * 55)
        print(f"{'layer':>5s} | {'kNN acc':>9s} {'silhouette':>9s}  <- SPECIES (objects_subset)")
        print("-" * 55)
        for layer in range(num_layers):
            m = separability_metrics(feats[layer], species_labels)
            species_metrics[layer] = m
            print(f"{layer:>5d} | {m['knn_acc']:>9.3f} {m['silhouette']:>9.3f}")
        print("-" * 55)
        print(f"(majority-class chance kNN baseline = {species_metrics[0]['chance']:.3f}, "
              f"n_classes = {species_metrics[0]['n_classes']})")
        print("=" * 55)
    print(
        "\nReading: higher kNN/silhouette = labels are better separated in that layer."
    )

    # --- 2-D scatters for the chosen layer ---
    plot_layer = args.plot_layer if args.plot_layer >= 0 else num_layers - 1
    emb, method = embed_2d(feats[plot_layer], seed=args.seed)

    out_png_action = os.path.join(
        args.out_dir, f"action_separability_layer{plot_layer}_t{args.timestep}.png"
    )
    plot_single(
        emb, action_labels,
        f"layer {plot_layer}, kNN={action_metrics[plot_layer]['knn_acc']:.3f}",
        method,
        f"Action-tag separability of decoder latents ({method})",
        out_png_action,
    )

    if args.objects_subset == "all":
        out_png_species = os.path.join(
            args.out_dir, f"species_separability_layer{plot_layer}_t{args.timestep}.png"
        )
        plot_single(
            emb, species_labels,
            f"layer {plot_layer}, kNN={species_metrics[plot_layer]['knn_acc']:.3f}",
            method,
            f"Species (objects_subset) separability of decoder latents ({method})",
            out_png_species,
        )

    # --- Persist raw numbers for follow-up analysis ---
    np.savez(
        os.path.join(args.out_dir, f"action_separability_t{args.timestep}.npz"),
        action_labels=np.asarray(action_labels),
        species_labels=np.asarray(species_labels),
        **{f"layer{l}": feats[l] for l in range(num_layers)},
    )
    print(f"[data]  saved raw latents -> {args.out_dir}")


if __name__ == "__main__":
    main()
