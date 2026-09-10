#!/usr/bin/env python3
"""
Precompute the frozen-T5 word table the action-label conditioner runs on.

Training conditions on the controlled vocabulary token by token: every label is
assembled at runtime from these vectors. Encoding them on the fly would mean a
resident T5 in every training process
for 103 vectors that never change, so they are baked once into
``dataset/action_word_embeddings.npy``.

One vector per ``CONTROLLED_VOCAB`` token, in vocabulary order, encoded from
``vocab_t5_text(token)`` -- not from the token spelling, which reads as the drink
for "punch" and as terrain for "land".

Keyed by WORD, not by label string. The old label-keyed sidecar had to be rebuilt
whenever anyone edited a label and could not represent an unseen combination at
all; this table depends on the vocabulary alone, so it is one global file and
relabelling never stales it.

The encoder settings are not options: pooling, EOS policy and vector
postprocessing are fixed by the geometry preflight's selected variant
(``slot/eos_keep/center_l2``) and recorded in ``embedding_contract``, whose hash
is the ``embedding_fingerprint`` a checkpoint is bound to.

Usage:
    python tools/build_action_label_embeddings.py [--t5-model NAME] [--force]
    python tools/build_action_label_embeddings.py --out path/to/table.npy

Options:
    --t5-model NAME   T5 model to encode with (default: t5-base). MUST match the
                      model that built cond.npy's joints_names_embs, or the word
                      vectors land in a different space than the model's
                      t5_out_dim expects and construction fails.
    --t5-path DIR     Local model directory (default: the sibling t5 cache the
                      geometry preflight resolves). Its files are hashed into
                      the contract, so pointing this at different weights
                      re-encodes the table even when --t5-model is unchanged.
    --force           Re-encode even when the table is already current.
"""

from __future__ import annotations

import argparse
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
    ACTION_WORD_EMBEDDING_EOS_POLICY,
    ACTION_WORD_EMBEDDING_DTYPE,
    ACTION_WORD_EMBEDDING_POOLING,
    ACTION_WORD_EMBEDDING_VECTOR_POSTPROCESS,
    ActionConditioningError,
    action_word_embedding_payload,
    build_action_conditioning_bundle,
    embedding_contract_payload,
    load_action_conditioning_bundle,
    word_table_sha256,
)
from data_loaders.truebones.truebones_utils.motion_labels import (  # noqa: E402
    CONTROLLED_VOCAB,
    vocab_t5_text,
)
from data_loaders.truebones.truebones_utils.param_utils import (  # noqa: E402
    get_action_word_embeddings_path,
)


# The files whose bytes decide what the encoder produces.
_T5_HASHED_FILES = (
    "config.json", "generation_config.json", "model.safetensors",
    "spiece.model", "special_tokens_map.json", "tokenizer_config.json",
)


def _resolve_t5_material(t5_model: str, t5_path: str | None):
    """``(directory, artifact hash)`` for the encoder this run was asked for.

    Resolved before anything decides to skip: "the table is current" is a claim
    about the WEIGHTS it was encoded from, and the model name alone does not
    identify those -- two directories both called t5-base can hold different
    bytes, which is exactly what an explicit --t5-path is for.
    """
    from evaluate_action_label_geometry import _resolve_t5_dir, _sha256_files

    t5_dir = _resolve_t5_dir(t5_path, t5_model)
    return t5_dir, _sha256_files(t5_dir, _T5_HASHED_FILES)


def _encode_vocabulary(t5_dir, t5_hash: str, t5_model: str, batch_size: int):
    """Encode every token under the contract's pooling / EOS / postprocess.

    Deliberately calls the geometry preflight's own encoder helpers rather than
    re-implementing masked mean pooling: those are the functions that produced
    the vectors the representation was selected on, so the table shipped here is
    the table that was measured.
    """
    from evaluate_action_label_geometry import (
        _encode_both_eos_policies,
        _postprocess_atoms,
    )

    import torch
    from transformers import T5Config, T5EncoderModel, T5Tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = T5Tokenizer.from_pretrained(str(t5_dir), local_files_only=True)
    safetensors_path = t5_dir / "model.safetensors"
    if safetensors_path.is_file():
        import safetensors.torch

        config = T5Config.from_pretrained(str(t5_dir), local_files_only=True)
        encoder = T5EncoderModel(config)
        encoder.load_state_dict(
            safetensors.torch.load_file(str(safetensors_path)), strict=False
        )
    else:
        encoder = T5EncoderModel.from_pretrained(str(t5_dir), local_files_only=True)
    encoder = encoder.eval().to(device)

    texts = [vocab_t5_text(token) for token in CONTROLLED_VOCAB]
    pooled = _encode_both_eos_policies(
        tokenizer, encoder, device, texts, batch_size
    )[ACTION_WORD_EMBEDDING_EOS_POLICY]
    table = _postprocess_atoms(pooled, ACTION_WORD_EMBEDDING_VECTOR_POSTPROCESS)
    table = np.asarray(table, dtype=np.float32)

    contract = embedding_contract_payload(
        token_to_text={token: vocab_t5_text(token) for token in CONTROLLED_VOCAB},
        t5_name=t5_model,
        t5_artifact_sha256=t5_hash,
        tokenizer_class=type(tokenizer).__name__,
        tokenizer_version=importlib.metadata.version("transformers"),
        pooling=ACTION_WORD_EMBEDDING_POOLING,
        eos_policy=ACTION_WORD_EMBEDDING_EOS_POLICY,
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
    # ignored, because the name is "t5-base" either way.
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
        if not stale:
            print(
                f"[skip] {out_path} already holds {len(CONTROLLED_VOCAB)} word vector(s) "
                f"(embedding_fingerprint {existing.embedding_fingerprint})"
            )
            return out_path
        print(f"[rebuild] {out_path}: {'; '.join(stale)}")

    print(f"encoding {len(CONTROLLED_VOCAB)} vocabulary token(s) with '{t5_model}' "
          f"from {t5_dir} ...")
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
            "label; re-run tools/evaluate_action_label_geometry.py before using it."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, action_word_embedding_payload(table, contract), allow_pickle=True)
    print(
        f"[OK] wrote {out_path} ({table.shape[0]} words x {table.shape[1]}d, "
        f"pooling={ACTION_WORD_EMBEDDING_POOLING}, "
        f"eos={ACTION_WORD_EMBEDDING_EOS_POLICY}, "
        f"postprocess={ACTION_WORD_EMBEDDING_VECTOR_POSTPROCESS})"
    )
    print(f"     embedding_fingerprint         {bundle.embedding_fingerprint}")
    print(f"     conditioning_contract_finger. {bundle.conditioning_contract_fingerprint}")
    print(f"     slot source rank              {rank['total_rank']} "
          f"(latent_dim must be at least this)")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Precompute the frozen-T5 action-word table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--out", dest="out", type=str, default=None,
        help="Where to write the table (default: dataset/action_word_embeddings.npy).",
    )
    parser.add_argument("--t5-model", "--t5_model", dest="t5_model", default="t5-base",
                        help="T5 model to encode with (default: t5-base).")
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
