#!/usr/bin/env python3
"""
Precompute the frozen text-encoder word table the action-label conditioner runs on.

Training conditions on the controlled vocabulary token by token: every label is
assembled at runtime from these vectors. Encoding them on the fly would mean a
resident encoder in every training process for vectors that never change, so they are
baked once into ``dataset/action_word_embeddings.npy``.

One vector per ``CONTROLLED_VOCAB`` token, in vocabulary order. A
``T5_ENCODED_VOCAB`` token is encoded from ``vocab_t5_text(token)`` -- not from
the token spelling, which reads as a milkshake for "shake" and as terrain
for "land". A ``SYNTHETIC_CODE_VOCAB`` token (the direction axis) is not
encoded at all: its row is an orthonormal code written by
``synthetic_code_rows``, because T5's geometry on that closed axis put every
member next to its own antonym. The two halves are stitched into one table by
``scatter_synthetic_code_rows``.

Keyed by WORD, not by label string. The old label-keyed sidecar had to be rebuilt
whenever anyone edited a label and could not represent an unseen combination at
all; this table depends on the vocabulary alone, so it is one global file and
relabelling never stales it.

The encoder settings are not options: pooling and EOS policy come with the
encoder (``text_encoder_pooling``), the vector postprocess is fixed by the
conditioning contract (``center_l2``), and all three are recorded in
``embedding_contract``, whose hash is the ``embedding_fingerprint`` a checkpoint
is bound to.

Usage:
    python tools/build_action_label_embeddings.py [--t5-model NAME] [--force]
    python tools/build_action_label_embeddings.py --out path/to/table.npy

Options:
    --t5-model NAME   Text encoder to encode with (default: google/embeddinggemma-300m). MUST match the
                      model that built cond.npy's joints_names_embs, or the word
                      vectors land in a different space than the model's
                      t5_out_dim expects and construction fails.
    --t5-path DIR     Local model directory (default: ``Anytop/.models/<name>``, then the HF cache).
                      Its files are hashed into the contract, so pointing this at
                      different weights re-encodes the table even when
                      --t5-model is unchanged.
    --force           Re-encode even when the table is already current.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import sys
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
ANYTOP_DIR = TOOLS_DIR.parent
_PARENT_DIR = ANYTOP_DIR.parent
sys.path.insert(0, str(_PARENT_DIR))
sys.path.insert(0, str(ANYTOP_DIR))
sys.path.insert(0, str(TOOLS_DIR))

from data_loaders.truebones.truebones_utils.action_label_conditioning_contract import (  # noqa: E402
    ACTION_WORD_EMBEDDING_DTYPE,
    ACTION_WORD_EMBEDDING_VECTOR_POSTPROCESS,
    ActionConditioningError,
    action_word_embedding_payload,
    build_action_conditioning_bundle,
    embedding_contract_payload,
    load_action_conditioning_bundle,
    ordered_token_sources,
    scatter_synthetic_code_rows,
    word_table_sha256,
)
from data_loaders.truebones.truebones_utils.motion_labels import (  # noqa: E402
    CONTROLLED_VOCAB,
    SYNTHETIC_CODE_VOCAB,
    T5_ENCODED_VOCAB,
    vocab_t5_text,
)
from data_loaders.truebones.truebones_utils.param_utils import (  # noqa: E402
    get_action_word_embeddings_path,
)
from data_loaders.truebones.truebones_utils.text_encoder import (  # noqa: E402
    DEFAULT_TEXT_ENCODER,
    is_sentence_encoder,
    text_encoder_pooling,
)


# The files whose bytes decide what a T5 produces. A sentence-transformers
# encoder spreads its pooling and dense heads over subdirectories, so every
# file in its directory is hashed instead.
_T5_HASHED_FILES = (
    "config.json", "generation_config.json", "model.safetensors",
    "spiece.model", "special_tokens_map.json", "tokenizer_config.json",
)
_UNHASHED_SUFFIXES = (".md", ".gitattributes")


def _sha256_files(root: Path, names) -> str:
    """One digest over *names* under *root*, order-independent and name-tagged."""
    digest = hashlib.sha256()
    for name in sorted(names):
        path = root / name
        if not path.is_file():
            continue
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _resolve_t5_dir(t5_path: str | None, t5_name: str) -> Path:
    if t5_path:
        path = Path(t5_path)
    else:
        from model.conditioners import _resolve_t5_local_dir

        path = Path(_resolve_t5_local_dir(t5_name))
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            f"local T5 directory not found: {path}. Pass --t5-path explicitly."
        )
    return path


def _resolve_t5_material(t5_model: str, t5_path: str | None):
    """``(directory, artifact hash)`` for the encoder this run was asked for.

    Resolved before anything decides to skip: "the table is current" is a claim
    about the WEIGHTS it was encoded from, and the model name alone does not
    identify those -- two directories both called t5-base can hold different
    bytes, which is exactly what an explicit --t5-path is for.
    """
    t5_dir = _resolve_t5_dir(t5_path, t5_model)
    if is_sentence_encoder(t5_model):
        names = [
            path.relative_to(t5_dir).as_posix() for path in t5_dir.rglob("*")
            if path.is_file() and not path.name.endswith(_UNHASHED_SUFFIXES)
        ]
    else:
        names = _T5_HASHED_FILES
    return t5_dir, _sha256_files(t5_dir, names)


def _postprocess_atoms(vectors: np.ndarray, mode: str) -> np.ndarray:
    """``center`` subtracts the mean, ``l2`` puts every row on the sphere.

    Applied to the T5-encoded rows ONLY. The synthetic code rows are already
    unit vectors on their own axes and are added afterwards: centring them would
    destroy the orthogonality they exist for, and including them in the mean
    would make every encoded vector depend on how many synthetic tokens the
    vocabulary happens to hold.
    """
    result = vectors.astype(np.float64, copy=True)
    if mode.startswith("center"):
        result -= result.mean(axis=0, keepdims=True)
    if mode.endswith("l2"):
        norm = np.linalg.norm(result, axis=1, keepdims=True)
        result = result / np.maximum(norm, 1e-12)
    return result


def _encode_vocabulary(t5_dir, t5_hash: str, t5_model: str, batch_size: int):
    """Encode every token with the encoder's own pooling, then the shared postprocess."""
    import torch

    from model.conditioners import build_text_conditioner

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = build_text_conditioner(t5_model, device, t5_path=str(t5_dir))
    tokenizer = encoder.t5_tokenizer if hasattr(encoder, "t5_tokenizer") else encoder.model.tokenizer

    texts = [vocab_t5_text(token) for token in T5_ENCODED_VOCAB]
    chunks = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            tokens = encoder.tokenize_entries(texts[start:start + batch_size])
            chunks.append(encoder(tokens).float().cpu().numpy())
    pooled = np.concatenate(chunks, axis=0)
    encoded = _postprocess_atoms(pooled, ACTION_WORD_EMBEDDING_VECTOR_POSTPROCESS)
    table = scatter_synthetic_code_rows(np.asarray(encoded, dtype=np.float32))

    pooling, eos_policy = text_encoder_pooling(t5_model)
    contract = embedding_contract_payload(
        token_sources=ordered_token_sources(),
        t5_name=t5_model,
        t5_artifact_sha256=t5_hash,
        tokenizer_class=type(tokenizer).__name__,
        tokenizer_version=importlib.metadata.version("transformers"),
        pooling=pooling,
        eos_policy=eos_policy,
        vector_postprocess=ACTION_WORD_EMBEDDING_VECTOR_POSTPROCESS,
        embedding_dim=int(table.shape[1]),
        dtype=ACTION_WORD_EMBEDDING_DTYPE,
        word_table_sha256=word_table_sha256(table),
    )
    return table, contract


def _current_bundle(path: Path):
    """The table already on disk, or None when there is nothing usable there.

    A file this code cannot validate is treated as absent rather than fatal: the
    whole point of the rebuild path is to replace it.
    """
    if not path.is_file():
        return None
    try:
        return load_action_conditioning_bundle(path)
    except (ActionConditioningError, ValueError, OSError) as exc:
        print(f"[stale] {path}: {exc}")
        return None


def build_word_table(out_path: Path, t5_model: str, t5_path: str | None,
                     force: bool, batch_size: int = 64) -> Path:
    # Resolve and hash the requested encoder before deciding anything: a table
    # counts as current only when it was encoded from THESE weights, and the
    # contract already records the hash to compare against. Matching on t5_name
    # alone let an explicit --t5-path holding different weights be silently
    # ignored, because the name is the same either way.
    t5_dir, t5_hash = _resolve_t5_material(t5_model, t5_path)
    existing = None if force else _current_bundle(out_path)
    if existing is not None:
        contract = existing.embedding_contract
        stale = []
        if contract.get("t5_name") != t5_model:
            stale.append(
                f"encoded with '{contract.get('t5_name')}', asked for '{t5_model}'"
            )
        if contract.get("t5_artifact_sha256") != t5_hash:
            stale.append(
                f"encoder artifact {str(contract.get('t5_artifact_sha256'))[:12]}..., "
                f"{t5_dir} hashes to {t5_hash[:12]}..."
            )
        # The vocabulary LIST is already checked on load (ordered_vocab), but the
        # TEXT each row was encoded from is not: editing _VOCAB_T5_TEXT, or moving
        # a token onto the synthetic-code side, leaves a table that loads cleanly
        # and holds the wrong vectors. Compare the sources so an unattended caller
        # (regenerate_dataset_artifacts) rebuilds on that edit instead of skipping.
        wanted_sources = ordered_token_sources()
        stored_sources = [dict(entry) for entry in (contract.get("ordered_token_sources") or ())]
        if stored_sources != wanted_sources:
            changed = [
                str(wanted.get("token"))
                for stored, wanted in zip(stored_sources, wanted_sources)
                if stored != wanted
            ]
            stale.append(
                "token sources changed"
                + (f" ({', '.join(changed[:6])}{' ...' if len(changed) > 6 else ''})"
                   if changed else "")
            )
        if not stale:
            print(
                f"[skip] {out_path} already holds {len(CONTROLLED_VOCAB)} word vector(s) "
                f"(embedding_fingerprint {existing.embedding_fingerprint})"
            )
            return out_path
        print(f"[rebuild] {out_path}: {'; '.join(stale)}")

    print(f"encoding {len(T5_ENCODED_VOCAB)} vocabulary token(s) with '{t5_model}' "
          f"from {t5_dir}; {len(SYNTHETIC_CODE_VOCAB)} more take orthonormal code "
          f"rows ...")
    table, contract = _encode_vocabulary(t5_dir, t5_hash, t5_model, batch_size)
    # Validate before writing: a table that the loader would refuse must never
    # reach the dataset directory, where it would fail at the start of training
    # instead of here.
    bundle = build_action_conditioning_bundle(table, contract, source=str(out_path))
    rank = bundle.slot_source_rank_report(latent_dim=bundle.embedding_dim)
    if not rank["full_rank"]:
        raise SystemExit(
            f"ERROR: the encoded word table does not have full slot-source rank "
            f"({rank['slots']}). Slot channels would not be separable for every legal "
            "label; a vocabulary token whose T5 text collides with another's is the "
            "usual cause -- change it in _VOCAB_T5_TEXT and re-encode. The direction "
            "block is orthonormal by construction and cannot be the cause."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, action_word_embedding_payload(table, contract), allow_pickle=True)
    print(
        f"[OK] wrote {out_path} ({table.shape[0]} words x {table.shape[1]}d, "
        f"pooling={contract['pooling']}, "
        f"eos={contract['eos_policy']}, "
        f"postprocess={ACTION_WORD_EMBEDDING_VECTOR_POSTPROCESS}; "
        f"{len(SYNTHETIC_CODE_VOCAB)} of them orthonormal code rows)"
    )
    print(f"     embedding_fingerprint         {bundle.embedding_fingerprint}")
    print(f"     conditioning_contract_finger. {bundle.conditioning_contract_fingerprint}")
    print(f"     slot source rank              {rank['total_rank']} "
          f"(latent_dim must be at least this)")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Precompute the frozen text-encoder action-word table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--out", dest="out", type=str, default=None,
        help="Where to write the table (default: dataset/action_word_embeddings.npy).",
    )
    parser.add_argument("--t5-model", "--t5_model", dest="t5_model", default=DEFAULT_TEXT_ENCODER,
                        help=f"Text encoder to encode with (default: {DEFAULT_TEXT_ENCODER}).")
    parser.add_argument("--t5-path", "--t5_path", dest="t5_path", default=None,
                        help="Local T5 directory, if it is not the default cache.")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=64)
    parser.add_argument("--force", action="store_true",
                        help="Re-encode even when the table is already current.")
    args = parser.parse_args()

    build_word_table(
        Path(get_action_word_embeddings_path(args.out)),
        args.t5_model,
        args.t5_path,
        args.force,
        args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
