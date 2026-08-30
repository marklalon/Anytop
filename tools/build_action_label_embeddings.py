#!/usr/bin/env python3
"""
Precompute the frozen-T5 embedding sidecar for ``action_labels.jsonl``.

Training conditions on the T5 mean-pool of each clip's ``action_label``. Encoding
those on the fly would mean a resident T5 in every training process for a value
that never changes, so they are baked once into ``action_label_embs.npy`` next to
the labels and looked up by text.

The sidecar covers two string sets, both needed at training time:

  * every distinct ``action_label`` in the dataset;
  * every coarse string those labels synthesize to ("stands still and growls" ->
    "idle, roar"), because ``--action_label_coarse_prob`` swaps one for the other
    per sample and both have to resolve through the same table.

Keyed by the label text rather than by clip: labels repeat heavily across clips,
and the coarse strings collapse further still.

Usage:
    python tools/build_action_label_embeddings.py [DATASET_DIR ...] [--t5-model NAME]

    # Every dataset a cond.npy references
    python tools/build_action_label_embeddings.py --cond-path dataset/.../cond.npy

Options:
    --t5-model NAME   T5 model to encode with (default: t5-base). MUST match the
                      model that built cond.npy's joints_names_embs, or the
                      label vector lands in a different space than the model's
                      t5_out_dim expects and loading fails.
    --force           Re-encode even when the sidecar is already up to date.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

ANYTOP_DIR = Path(__file__).resolve().parent.parent
_PARENT_DIR = ANYTOP_DIR.parent
sys.path.insert(0, str(_PARENT_DIR))
sys.path.insert(0, str(ANYTOP_DIR))

from data_loaders.truebones.truebones_utils.motion_labels import (  # noqa: E402
    coarse_label_from_words,
    load_action_labels,
    vocab_words_in,
)
from data_loaders.truebones.truebones_utils.param_utils import (  # noqa: E402
    ACTION_LABELS_FILE,
    ACTION_LABEL_EMBEDDINGS_FILE,
    get_dataset_dir,
)


def collect_label_strings(dataset_dir: Path) -> list[str]:
    """Every string the dataset can hand the model, full labels and coarse alike."""
    strings: set[str] = set()
    for entry in load_action_labels(dataset_dir).values():
        label = entry["action_label"]
        if not label:
            # An empty label is the unconditional state; it is routed to the
            # model's learned null embedding and must never be T5-encoded.
            continue
        strings.add(label)
        coarse = coarse_label_from_words(vocab_words_in(label))
        if coarse:
            strings.add(coarse)
    return sorted(strings)


def build_sidecar(dataset_dir: Path, t5_model: str, force: bool) -> Path:
    labels_path = dataset_dir / ACTION_LABELS_FILE
    if not labels_path.exists():
        raise FileNotFoundError(f"{ACTION_LABELS_FILE} not found at {labels_path}")

    strings = collect_label_strings(dataset_dir)
    # Source hash: a byte-identical labels file guarantees the sidecar is
    # current, and any edit (new clip, reworded label) makes it stale even when
    # every old string is still covered (e.g. a clip was removed).
    labels_md5 = hashlib.md5(labels_path.read_bytes()).hexdigest()
    sidecar_path = dataset_dir / ACTION_LABEL_EMBEDDINGS_FILE

    if sidecar_path.exists() and not force:
        payload = np.load(sidecar_path, allow_pickle=True).item()
        existing = payload.get("embeddings") or {}
        stored_md5 = payload.get("action_labels_md5")
        up_to_date = payload.get("t5_name") == t5_model and (
            (stored_md5 is not None and stored_md5 == labels_md5)
            # Legacy sidecar (pre-hash): fall back to the string-coverage check.
            or (stored_md5 is None and set(existing) >= set(strings))
        )
        if up_to_date:
            if stored_md5 is None:
                # Stamp the source hash so future runs take the fast path.
                payload["action_labels_md5"] = labels_md5
                np.save(sidecar_path, payload, allow_pickle=True)
            print(f"[skip] {sidecar_path} already covers {len(strings)} label string(s)")
            return sidecar_path

    print(f"[{dataset_dir.name}] encoding {len(strings)} label string(s) with '{t5_model}' ...")
    import torch

    from model.conditioners import T5Conditioner

    device = "cuda" if torch.cuda.is_available() else "cpu"
    conditioner = T5Conditioner(
        name=t5_model,
        finetune=False,
        word_dropout=0.0,
        normalize_text=False,
        device=device,
        autocast_dtype=None,
        local_files_only=True,
    )

    embeddings: dict[str, np.ndarray] = {}
    batch_size = 64
    with torch.no_grad():
        for start in range(0, len(strings), batch_size):
            chunk = strings[start:start + batch_size]
            # tokenize_entries, not tokenize: tokenize runs the joint-name
            # normalizer (prefix stripping, the anatomical-word gate) which would
            # blank out an ordinary sentence.
            tokens = conditioner.tokenize_entries(chunk)
            embs = conditioner(tokens).detach().cpu().numpy().astype(np.float32, copy=False)
            for text, emb in zip(chunk, embs):
                embeddings[text] = emb

    embedding_dim = int(next(iter(embeddings.values())).shape[-1]) if embeddings else 0
    np.save(sidecar_path, {
        "t5_name": t5_model,
        "embedding_dim": embedding_dim,
        "embeddings": embeddings,
        "action_labels_md5": labels_md5,
    }, allow_pickle=True)
    print(f"[OK] wrote {sidecar_path} ({len(embeddings)} strings x {embedding_dim}d)")
    return sidecar_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Precompute the frozen-T5 action_label embedding sidecar.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "dataset_dirs", nargs="*", type=str,
        help="Processed dataset directories holding action_labels.jsonl. "
             "Defaults to the standard truebones_processed directory.",
    )
    parser.add_argument(
        "--cond-path", "--cond_path", dest="cond_path", type=str, default=None,
        help="Build for every dataset source a cond.npy references, instead of "
             "naming the directories.",
    )
    parser.add_argument("--t5-model", "--t5_model", dest="t5_model", default="t5-base",
                        help="T5 model to encode with (default: t5-base).")
    parser.add_argument("--force", action="store_true",
                        help="Re-encode even when the sidecar is already up to date.")
    args = parser.parse_args()

    if args.cond_path:
        from data_loaders.truebones.truebones_utils.get_opt import get_opt

        opt = get_opt(None, args.cond_path)
        dataset_dirs = [Path(source.root) for source in opt.sources]
    elif args.dataset_dirs:
        dataset_dirs = [Path(get_dataset_dir(d)).resolve() for d in args.dataset_dirs]
    else:
        dataset_dirs = [Path(get_dataset_dir(None)).resolve()]

    for dataset_dir in dataset_dirs:
        build_sidecar(dataset_dir, args.t5_model, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
