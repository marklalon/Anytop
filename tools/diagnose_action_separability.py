"""Diagnose how well an AnyTop checkpoint's internal representation separates
different action types (action_tags), and compare two checkpoints side by side.

Motivation
----------
AnyTop has NO explicit action-class conditioning: action type only enters
through loop / playspeed / global-energy scalars and the joint-name text
embedding. So "do different actions overlap in the same activation region?"
is tested by probing intermediate decoder-layer activations, labelling each
clip by its primary action tag, and measuring separability.

Pipeline (per checkpoint)
-------------------------
1. Pick N random QUADROPEDS clips (same indices for both checkpoints).
2. Feed each clip to the model at a fixed diffusion timestep (default t=0,
   i.e. the clean normalized motion) and capture every decoder layer's
   latent via forward hooks; masked-mean-pool over valid joints + frames
   -> one D-dim vector per clip per layer.
3. Quantify separability of the action tags in that latent space:
      * leave-one-out kNN accuracy  (no model fitting; non-linear sensitive)
      * silhouette score            (cluster compactness vs separation)
   Both are reported per layer.
4. Render a UMAP (fallback t-SNE) 2-D scatter coloured by action tag for a
   chosen layer, two checkpoints side by side.

The two numbers per layer are only meaningful *relative to each other*:
compare the loco-only checkpoint against the full-action checkpoint. If the
full model's separability is markedly lower, that is evidence of
representation interference from mixing action types.

Usage
-----
    python tools/diagnose_action_separability.py \
        --ckpt1  save/quadropeds_locomotion_no_rot_helper_v1/model000100000.pt \
        --ckpt2  save/quadropeds_final_v1/model000100000.pt \
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
    ACTION_TAG_SAMPLE_WEIGHTS,
    _normalize_motion_action_tags,
)
from sample.generate import prepare_generation_runtime
from utils.model_util import unwrap_anytop_model
from utils.parser_util import generate_args


# ----------------------------------------------------------------------------
# Action-tag -> single primary label
# ----------------------------------------------------------------------------
def primary_action_tag(raw_tags):
    """Reduce a clip's (possibly multi-)action_tags to one display label.

    Mirrors the dataset's "attribute the clip to its highest-weighted tag"
    rule so the labelling matches how training actually samples the clip.
    Falls back to alphabetical first tag when no weight map is configured.
    """
    tags = _normalize_motion_action_tags(raw_tags)  # set[str], lowercased
    if not tags:
        return "unknown"
    ordered = sorted(tags)
    if ACTION_TAG_SAMPLE_WEIGHTS:
        return max(ordered, key=lambda t: ACTION_TAG_SAMPLE_WEIGHTS.get(t, 1.0))
    return ordered[0]


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
    layer_feats : dict[int, np.ndarray]   layer_idx -> (N, D)
    labels      : list[str]               primary action tag per clip (len N)
    """
    # Reseed so any random crop inside prepare_sample_by_name is identical
    # across the two checkpoints (same name order -> same crops -> same input).
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
    labels = []
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
                labels.append(primary_action_tag(raw_tags))
    finally:
        for h in handles:
            h.remove()

    layer_feats = {i: np.concatenate(v, axis=0) for i, v in layer_feats.items()}
    return layer_feats, labels


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


def plot_comparison(emb_a, labels_a, title_a, emb_b, labels_b, title_b, method, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_tags = sorted(set(labels_a) | set(labels_b))
    cmap = plt.get_cmap("tab20" if len(all_tags) > 10 else "tab10")
    color = {tag: cmap(i % cmap.N) for i, tag in enumerate(all_tags)}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=False, sharey=False)
    for ax, emb, labels, title in (
        (axes[0], emb_a, labels_a, title_a),
        (axes[1], emb_b, labels_b, title_b),
    ):
        labels = np.asarray(labels)
        for tag in all_tags:
            m = labels == tag
            if not m.any():
                continue
            ax.scatter(emb[m, 0], emb[m, 1], s=28, alpha=0.8, color=color[tag], label=tag)
        ax.set_title(title)
        ax.set_xlabel(f"{method}-1")
        ax.set_ylabel(f"{method}-2")
    handles, lbls = axes[0].get_legend_handles_labels()
    if not handles:
        handles, lbls = axes[1].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper center", ncol=min(len(all_tags), 8), fontsize=9)
    fig.suptitle(f"Action-tag separability of decoder latents ({method})", y=1.02, fontsize=13)
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
    ap.add_argument("--ckpt1", required=True, help="first checkpoint .pt")
    ap.add_argument("--ckpt2", required=True, help="second checkpoint .pt")
    ap.add_argument("--num_clips", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--objects_subset", default="quadropeds")
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

    # --- Load both checkpoints (each reads its own args.json architecture) ---
    print(f"\n=== Loading checkpoint 1 ===\n{args.ckpt1}")
    loco_rt, loco_args = load_runtime(args.ckpt1, args.device)
    print(f"\n=== Loading checkpoint 2 ===\n{args.ckpt2}")
    full_rt, full_args = load_runtime(args.ckpt2, args.device)

    # --- Build the clip set (same motion indices fed to both models) ---
    # Use each model's own num_frames/temporal_window for the dataset resample,
    # but the SAME random indices so both see the same underlying motions/tags.
    rng = np.random.default_rng(args.seed)

    def build_dataset(model_args):
        return get_dataset(
            num_frames=int(getattr(model_args, "num_frames", 60)),
            split=args.split,
            temporal_window=int(getattr(model_args, "temporal_window", 31)),
            balanced=False,
            objects_subset=args.objects_subset,
        )

    loco_ds = build_dataset(loco_args)
    full_ds = build_dataset(full_args)
    # Sample clip NAMES (not indices) so both models see identical motions even
    # if their datasets order/length differently; drop any name missing from the
    # full-action dataset (should not happen for the same subset/split).
    loco_names = list(loco_ds.motion_dataset.name_list)
    full_name_set = set(full_ds.motion_dataset.name_list)
    shared = [nm for nm in loco_names if nm in full_name_set]
    n = min(args.num_clips, len(shared))
    names = list(rng.choice(np.asarray(shared, dtype=object), size=n, replace=False))
    print(f"\nSampled {n} clips from '{args.objects_subset}' split='{args.split}' "
          f"(shared pool={len(shared)}).")

    # --- Extract activations (same names + same seed -> identical inputs) ---
    print("\n[extract] model 1 ...")
    loco_feats, loco_labels = extract_layer_activations(
        loco_rt, loco_ds, names, args.batch_size, args.timestep, device, args.seed
    )
    print("[extract] model 2 ...")
    full_feats, full_labels = extract_layer_activations(
        full_rt, full_ds, names, args.batch_size, args.timestep, device, args.seed
    )

    # labels are identical (same indices); sanity-report the tag distribution.
    tags, tag_counts = np.unique(np.asarray(loco_labels), return_counts=True)
    print("\nAction-tag distribution in the probe set:")
    for tag, c in sorted(zip(tags, tag_counts), key=lambda kv: -kv[1]):
        print(f"  {tag:<24s} {c}")

    # --- Per-layer separability table ---
    num_layers = len(loco_feats)
    print("\n" + "=" * 74)
    print(f"{'layer':>5s} | {'ckpt1 kNN':>9s} {'ckpt1 sil':>9s} | {'ckpt2 kNN':>9s} {'ckpt2 sil':>9s}")
    print("-" * 74)
    loco_metrics, full_metrics = {}, {}
    for layer in range(num_layers):
        lm = separability_metrics(loco_feats[layer], loco_labels)
        fm = separability_metrics(full_feats[layer], full_labels)
        loco_metrics[layer], full_metrics[layer] = lm, fm
        print(
            f"{layer:>5d} | {lm['knn_acc']:>9.3f} {lm['silhouette']:>9.3f} | "
            f"{fm['knn_acc']:>9.3f} {fm['silhouette']:>9.3f}"
        )
    print("-" * 74)
    print(f"(majority-class chance kNN baseline = {loco_metrics[0]['chance']:.3f}, "
          f"n_classes = {loco_metrics[0]['n_classes']})")
    print("=" * 74)
    print(
        "\nReading: higher kNN/silhouette = action tags are better separated in "
        "that layer.\nIf ckpt2 columns are markedly lower than ckpt1 at "
        "the same layer,\nthat is evidence of representation interference from "
        "mixing action types."
    )

    # --- 2-D scatter for the chosen layer ---
    plot_layer = args.plot_layer if args.plot_layer >= 0 else num_layers - 1
    emb_loco, method = embed_2d(loco_feats[plot_layer], seed=args.seed)
    emb_full, _ = embed_2d(full_feats[plot_layer], seed=args.seed)
    out_png = os.path.join(
        args.out_dir, f"action_separability_layer{plot_layer}_t{args.timestep}.png"
    )
    plot_comparison(
        emb_loco, loco_labels,
        f"ckpt1  (layer {plot_layer}, kNN={loco_metrics[plot_layer]['knn_acc']:.3f})",
        emb_full, full_labels,
        f"ckpt2  (layer {plot_layer}, kNN={full_metrics[plot_layer]['knn_acc']:.3f})",
        method, out_png,
    )

    # --- Persist raw numbers for follow-up analysis ---
    np.savez(
        os.path.join(args.out_dir, f"action_separability_t{args.timestep}.npz"),
        labels=np.asarray(loco_labels),
        **{f"loco_layer{l}": loco_feats[l] for l in range(num_layers)},
        **{f"full_layer{l}": full_feats[l] for l in range(num_layers)},
    )
    print(f"[data]  saved raw latents -> {args.out_dir}")


if __name__ == "__main__":
    main()
