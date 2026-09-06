#!/usr/bin/env python3
"""Reproducible preflight for the word-keyed action-label conditioning.

The tool is deliberately independent of the diffusion model.  Two things are
measured, and they are not the same kind of thing:

* the HARD gate -- properties the model cannot repair after the fact: two
  distinct labels landing on one point, a word whose contribution vanishes as
  the label grows, a frozen table/slot source basis that has lost rank, a
  projection narrower than the complete slot source space, or a transition
  that reads the same in both directions;
* REPORTED geometry -- pairwise p95, nearest-neighbour median and effective
  rank.  These are anisotropy measures, and the first learned Linear of
  ``action_label_projection`` can rescale any subspace, so they are tracked as
  regressions and never block.  Blocking on them is what made the previous gate
  unsatisfiable: a single pooled vector has ONE detail-energy share, and axis
  retention wants it small while these want it large.

The one-vector weighted-mean family this replaced is no longer enumerated: it was
rejected on a structural argument, not on a margin, and its measured frontier is
recoverable from git history if it ever needs re-deriving.  Generated-motion
quality is a
post-training acceptance check and is not claimed here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ANYTOP_DIR = Path(__file__).resolve().parent.parent
PARENT_DIR = ANYTOP_DIR.parent
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(ANYTOP_DIR))

from data_loaders.truebones.truebones_utils.action_label_conditioning_contract import (  # noqa: E402
    ACTION_LABEL_SLOTS,
    SLOT_DIRECTION,
    SLOT_HEAD,
    SLOT_MODIFIER,
    action_label_slots,
    assemble_slot_channels,
    conditioning_contract_payload,
    embedding_contract_payload,
    fingerprint,
    role_b_material,
    slot_channel_representation,
    slot_source_rank_report,
    word_slot,
    word_table_sha256,
)
from data_loaders.truebones.truebones_utils.motion_labels import (  # noqa: E402
    ACTION_LABEL_MAX_WORDS,
    CONTROLLED_VOCAB,
    DIRECTION_VOCAB,
    STATE_VOCAB,
    load_action_labels,
    parse_action_label,
    vocab_t5_text,
)
from data_loaders.truebones.truebones_utils.param_utils import ACTION_LABELS_FILE  # noqa: E402


DEFAULT_DATASET_DIRS = (
    ANYTOP_DIR / "dataset/truebones/zoo/truebones_processed",
    ANYTOP_DIR / "dataset/truebones/zoo_upgrade/clean_processed",
    ANYTOP_DIR / "dataset/unitybundles/processed",
)

# The HARD gate.  Every entry is a property the model cannot undo downstream.
HARD_GATE = {
    # Distinct labels must not share a conditioning point, and the worst-case
    # near neighbour must be no closer than the current baseline's worst case.
    "collision_cosine": 0.999999,
    "max_worst_nearest_over_baseline": 0.0,
    # A transition must not read the same in both directions.
    "max_reverse_pair_median": 0.50,
    # Axis retention is an EQUALITY for a slot representation: a channel's input
    # is a function of its own slot, so appending modifiers changes it by zero.
    "max_channel_drift": 0.0,
    # No word or role-specific slot source may be a linear combination of the
    # others.  Full slot-source rank proves injectivity and linear membership
    # readout for every non-empty subset admitted by the total-word cap; it does
    # not need a combinatorial enumeration.
    "require_full_rank_word_table": True,
    "require_full_rank_slot_sources": True,
    # The first learned Linear must be wide enough to inject the complete
    # direct sum of the three slot source spaces.
    "require_projection_contains_slot_sources": True,
}

# Reported, never blocking: the first learned Linear can rescale any subspace,
# so these measure anisotropy, not information.  Tracked against the baseline so
# a regression is still visible.
REPORTED_METRICS = ("pairwise_cosine_p95", "nearest_cosine_median", "effective_rank")

# How much better a variant has to measure before the difference counts as a
# reason to pick it.  Below this the canonical order decides instead.
SELECTION_TOLERANCE = 0.005

def _sha256_files(root: Path, names: Iterable[str]) -> str:
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
    path = Path(t5_path) if t5_path else ANYTOP_DIR / ".models" / t5_name
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            f"local T5 directory not found: {path}. Pass --t5-path explicitly."
        )
    return path


def _load_rows(dataset_dirs: Iterable[Path]) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    labels: set[str] = set()
    for dataset_dir in dataset_dirs:
        for entry in load_action_labels(dataset_dir).values():
            label = str(entry.get("action_label") or "")
            if not label:
                continue
            parse_action_label(label)
            rows.append({"group": entry["action_group"], "label": label})
            labels.add(label)
    return rows, sorted(labels)


def _encode_both_eos_policies(
    tokenizer, encoder, device: str, texts: list[str], batch_size: int
) -> dict[str, np.ndarray]:
    import torch

    chunks: dict[str, list[np.ndarray]] = {"keep": [], "drop": []}
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("the tokenizer has no eos_token_id")
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            hidden = encoder(**inputs).last_hidden_state.float()
            base_mask = inputs["attention_mask"].bool()
            for policy, mask in (
                ("keep", base_mask),
                ("drop", base_mask & inputs["input_ids"].ne(eos_id)),
            ):
                counts = mask.sum(dim=-1, keepdim=True)
                if torch.any(counts == 0):
                    raise ValueError(f"EOS policy {policy!r} produced an empty token sequence")
                pooled = (hidden * mask.unsqueeze(-1)).sum(dim=-2) / counts
                chunks[policy].append(pooled.cpu().numpy().astype(np.float32, copy=False))
    return {policy: np.concatenate(parts, axis=0) for policy, parts in chunks.items()}


def _l2_rows(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norm, 1e-12)


def _postprocess_atoms(vectors: np.ndarray, mode: str) -> np.ndarray:
    result = vectors.astype(np.float64, copy=True)
    if mode.startswith("center"):
        result -= result.mean(axis=0, keepdims=True)
    if mode.endswith("l2"):
        result = _l2_rows(result)
    return result


def _cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    unit = _l2_rows(vectors.astype(np.float64, copy=False))
    return np.clip(unit @ unit.T, -1.0, 1.0)


def _slot_bundle(
    atoms: np.ndarray,
    keyed: Iterable[tuple[str, str]],
    role_payload: dict[str, Any],
) -> np.ndarray:
    """Concatenated role-slot channels, standardised per channel and per group.

    Per-channel standardisation is not cosmetic: each channel enters the model
    through its own block of ``action_label_projection``'s first Linear (one
    Linear over the concatenation *is* one Linear per block, summed), so any
    fixed offline budget for a channel is something the model re-learns anyway.
    Measuring the raw budget would be measuring our own arbitrary constant.
    """
    rows: list[np.ndarray] = []
    groups: list[str] = []
    for group, label in keyed:
        slots = action_label_slots(group, parse_action_label(label))
        channels, _present = assemble_slot_channels(
            atoms, slots, role_payload["perm"], role_payload["sign"]
        )
        rows.append(channels.reshape(-1))
        groups.append(group)
    bundle = np.stack(rows)
    width = atoms.shape[1]
    for group in sorted(set(groups)):
        index = np.asarray([i for i, value in enumerate(groups) if value == group])
        for slot in range(len(ACTION_LABEL_SLOTS)):
            columns = slice(slot * width, (slot + 1) * width)
            block = bundle[index, columns]
            centered = block - block.mean(axis=0, keepdims=True)
            scale = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
            bundle[np.ix_(index, range(columns.start, columns.stop))] = block / max(scale, 1e-12)
    return bundle


def _worst_nearest(vectors: np.ndarray, groups: list[str]) -> dict[str, dict[str, Any]]:
    """The closest pair per checkpoint -- the collision metric a median hides."""
    result: dict[str, dict[str, Any]] = {}
    for group in sorted(set(groups)):
        index = np.asarray([i for i, value in enumerate(groups) if value == group])
        cosine = _cosine_matrix(vectors[index])
        nearest = np.max(cosine - np.eye(len(index)) * 2.0, axis=1)
        left = int(np.argmax(nearest))
        right = int(np.argmax(cosine[left] - np.eye(len(index))[left] * 2.0))
        result[group] = {
            "worst_nearest_cosine": float(nearest.max()),
            "pair": (int(index[left]), int(index[right])),
        }
    return result


def _channel_drift(atoms: np.ndarray, role_payload: dict[str, Any]) -> dict[str, Any]:
    """Does a head/direction channel move when unrelated modifiers are appended?

    For a slot representation the answer is exactly zero by construction; the
    check exists so a future edit that reintroduces cross-slot pooling fails
    here instead of silently reinstating 1/N dilution.
    """
    fillers = ("weapon", "bow", "shield", "1hand", "gun", "hammer")
    worst = 0.0
    samples: list[dict[str, Any]] = []
    for base in (("walk", "forward"), ("run", "forward"), ("walk", "backward"), ("idle",)):
        reference, _ = assemble_slot_channels(
            atoms,
            action_label_slots("locomotion" if len(base) > 1 else "stationary", base),
            role_payload["perm"], role_payload["sign"],
        )
        for count in range(1, len(fillers) + 1):
            tokens = base + fillers[:count]
            if len(tokens) > 8:
                break
            grown, _ = assemble_slot_channels(
                atoms,
                action_label_slots("locomotion" if len(base) > 1 else "stationary", tokens),
                role_payload["perm"], role_payload["sign"],
            )
            drift = max(
                float(np.max(np.abs(grown[slot] - reference[slot])))
                for slot in (SLOT_HEAD, SLOT_DIRECTION)
            )
            worst = max(worst, drift)
            samples.append({"base": list(base), "words": len(tokens), "max_abs_drift": drift})
    return {"max_channel_drift": worst, "samples": samples}


def _worst_pair_cosine(
    matrix: np.ndarray, names: list[tuple[str, ...]], device: str
) -> tuple[float, tuple[str, str] | None]:
    """Closest pair in a set of unit rows, chunked so the Gram never materialises.

    The exhaustive scan is the only quadratic step in the tool (tens of
    thousands of configurations), so it follows --device.  float64 throughout:
    the numbers here are gate evidence, and CPU/GPU agreement to ~1e-12 keeps a
    rerun on either device comparable.
    """
    step = 4096
    best_value = -2.0
    best_pair: tuple[str, str] | None = None
    if device == "cuda":
        import torch

        gram_source = torch.as_tensor(matrix, device="cuda", dtype=torch.float64)
        for start in range(0, len(matrix), step):
            chunk = gram_source[start:start + step] @ gram_source.T
            rows = torch.arange(chunk.shape[0], device="cuda")
            chunk[rows, rows + start] = -2.0
            flat = int(torch.argmax(chunk))
            value = float(chunk.view(-1)[flat])
            if value > best_value:
                left, right = divmod(flat, chunk.shape[1])
                best_value = value
                best_pair = (",".join(names[start + left]), ",".join(names[right]))
        return best_value, best_pair
    for start in range(0, len(matrix), step):
        chunk = matrix[start:start + step] @ matrix.T
        for row in range(chunk.shape[0]):
            chunk[row, start + row] = -2.0
        left, right = np.unravel_index(np.argmax(chunk), chunk.shape)
        if chunk[left, right] > best_value:
            best_value = float(chunk[left, right])
            best_pair = (",".join(names[start + left]), ",".join(names[right]))
    return best_value, best_pair


def _ridge_readout(design: np.ndarray, targets: np.ndarray, device: str) -> np.ndarray:
    """Closed-form ridge fit, on the same device as the rest of the scan."""
    ridge_scale = 1e-6
    if device == "cuda":
        import torch

        x = torch.as_tensor(design, device="cuda", dtype=torch.float64)
        y = torch.as_tensor(targets, device="cuda", dtype=torch.float64)
        gram = x.T @ x
        ridge = ridge_scale * torch.trace(gram) / x.shape[1]
        weights = torch.linalg.solve(gram + ridge * torch.eye(x.shape[1], device="cuda", dtype=torch.float64), x.T @ y)
        return (x @ weights).cpu().numpy()
    gram = design.T @ design
    ridge = ridge_scale * np.trace(gram) / design.shape[1]
    weights = np.linalg.solve(gram + ridge * np.eye(design.shape[1]), design.T @ targets)
    return design @ weights


def _slot_configuration_margins(
    atoms: np.ndarray, role_payload: dict[str, Any], max_subset: int, device: str = "cpu"
) -> dict[str, Any]:
    """Quantitative nearest-pair/readout diagnostic over useful configurations.

    Head and direction are cheap enough to enumerate over their complete legal
    domains.  Modifier combinations are enumerated only through ``max_subset``
    (the current corpus maximum by default).  Full-domain injectivity and word
    readability are certified separately by :func:`_slot_source_rank_report`;
    trying to enumerate all modifier subsets under an eight-token total cap
    would add hundreds of millions of rows without proving anything stronger.
    """
    import itertools

    perm = np.asarray(role_payload["perm"], dtype=np.int64)
    sign = np.asarray(role_payload["sign"], dtype=np.float64)
    vocab_index = {word: index for index, word in enumerate(CONTROLLED_VOCAB)}
    modifiers = tuple(word for word in CONTROLLED_VOCAB if word_slot(word) == SLOT_MODIFIER)

    def pooled(vectors: list[np.ndarray]) -> np.ndarray:
        mean = np.mean(np.stack(vectors), axis=0)
        return mean / max(float(np.linalg.norm(mean)), 1e-12)

    configurations: dict[str, tuple[list[np.ndarray], list[tuple[str, ...]]]] = {}
    head_vectors, head_names = [], []
    for word in STATE_VOCAB:
        head_vectors.append(pooled([atoms[vocab_index[word]]]))
        head_names.append((word,))
    for first, second in itertools.permutations(STATE_VOCAB, 2):
        head_vectors.append(pooled([
            atoms[vocab_index[first]], sign * atoms[vocab_index[second]][perm]
        ]))
        head_names.append((first, second))
    configurations["head"] = (head_vectors, head_names)

    direction_vectors, direction_names = [], []
    max_directions = min(len(DIRECTION_VOCAB), ACTION_LABEL_MAX_WORDS - 1)
    for size in range(1, max_directions + 1):
        for subset in itertools.combinations(DIRECTION_VOCAB, size):
            direction_vectors.append(pooled([atoms[vocab_index[w]] for w in subset]))
            direction_names.append(subset)
    configurations["direction"] = (direction_vectors, direction_names)

    modifier_subsets = [
        subset
        for size in range(1, max_subset + 1)
        for subset in itertools.combinations(modifiers, size)
    ]
    configurations["modifier"] = (
        [pooled([atoms[vocab_index[w]] for w in subset]) for subset in modifier_subsets],
        modifier_subsets,
    )

    report: dict[str, Any] = {
        "max_modifier_subset_size": max_subset,
        "direction_domain_is_complete": True,
        "device": device,
    }
    for slot_name, (vectors, names) in configurations.items():
        matrix = _l2_rows(np.stack(vectors))
        best = _worst_pair_cosine(matrix, names, device)
        # One ridge readout per slot: can the pooled channel still name its words?
        words = sorted({word for name in names for word in name})
        column = {word: i for i, word in enumerate(words)}
        targets = np.zeros((len(names), len(words)))
        for row, name in enumerate(names):
            for word in name:
                targets[row, column[word]] = 1.0
        design = np.stack(vectors)
        design = design - design.mean(axis=0, keepdims=True)
        predicted = _ridge_readout(design, targets, device)
        margins = [
            (
                float(
                    predicted[targets[:, i] > 0, i].min()
                    - predicted[targets[:, i] == 0, i].max()
                ),
                word,
            )
            for i, word in enumerate(words)
            if targets[:, i].any() and not targets[:, i].all()
        ]
        margins.sort()
        report[slot_name] = {
            "configurations": len(names),
            "worst_pair_cosine": best[0],
            "worst_pair": best[1],
            "min_membership_margin": margins[0][0],
            "min_membership_margin_word": margins[0][1],
            "words_not_separable": sum(1 for margin, _ in margins if margin <= 0.0),
        }
    return report


def _slot_source_rank_report(
    atoms: np.ndarray,
    role_payload: dict[str, Any],
    latent_dim: int,
) -> dict[str, Any]:
    """Certify every slot subset without enumerating the power set.

    If a slot's source rows are independent, two different 0/1 membership
    vectors cannot yield proportional sums.  L2-normalising those sums therefore
    creates neither a collision nor a loss of linear membership readability.
    The head source basis includes both the ordinary and ``R_B``-transformed
    version of every state word so the same argument covers ordered transitions.

    Slots occupy disjoint blocks after concatenation, hence their ranks add.  A
    first Linear at least that wide can be injective on the entire reachable
    source space and may freely rescale its subspaces.

    The computation itself lives in the conditioning contract because model
    construction runs the same certificate against ``latent_dim`` before a run
    starts; a second copy here would be a second definition of injectivity.
    """
    return slot_source_rank_report(
        atoms, role_payload["perm"], role_payload["sign"], latent_dim
    )


def _effective_rank(vectors: np.ndarray) -> float:
    centered = vectors.astype(np.float64, copy=False) - vectors.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    eigen = singular * singular
    denominator = float(np.square(eigen).sum())
    return float(eigen.sum() ** 2 / denominator) if denominator else 0.0


def _reverse_pairs(labels: list[str], label_to_group: dict[str, str]) -> list[tuple[int, int]]:
    signatures: dict[tuple[frozenset[str], tuple[str, ...]], list[tuple[int, tuple[str, ...]]]] = {}
    for index, label in enumerate(labels):
        if label_to_group[label] != "transition":
            continue
        tokens = parse_action_label(label)
        slots = action_label_slots("transition", tokens)
        if not any(slots["order_head_mask"]):
            continue
        heads = tuple(token for token, flagged in zip(tokens, slots["order_head_mask"]) if flagged)
        modifiers = tuple(token for token, flagged in zip(tokens, slots["order_head_mask"]) if not flagged)
        key = (frozenset(heads), modifiers)
        signatures.setdefault(key, []).append((index, heads))

    pairs: list[tuple[int, int]] = []
    for variants in signatures.values():
        for left_pos, (left_index, left_heads) in enumerate(variants):
            for right_index, right_heads in variants[left_pos + 1:]:
                if left_heads == tuple(reversed(right_heads)):
                    pairs.append((left_index, right_index))
    return sorted(pairs)


def _metrics(
    vectors: np.ndarray,
    groups: list[str],
    reverse_pairs: list[tuple[int, int]],
    collision: float,
) -> dict[str, Any]:
    # Models/checkpoints are group-specific.  Pooling across groups would count
    # an intentionally shared label string as an exact collision even though
    # the two conditions can never coexist in one model.
    all_upper: list[np.ndarray] = []
    all_nearest: list[np.ndarray] = []
    by_group: dict[str, dict[str, Any]] = {}
    rank_weighted = 0.0
    rank_count = 0
    collision_pairs = 0
    for group in sorted(set(groups)):
        indices = np.asarray([index for index, value in enumerate(groups) if value == group])
        group_vectors = vectors[indices]
        cosine = _cosine_matrix(group_vectors)
        upper = cosine[np.triu_indices(len(indices), k=1)]
        nearest = np.max(cosine - np.eye(len(indices)) * 2.0, axis=1)
        rank = _effective_rank(group_vectors)
        collisions = int(np.count_nonzero(upper >= collision))
        by_group[group] = {
            "labels": int(len(indices)),
            "pairwise_cosine_p95": float(np.quantile(upper, 0.95)),
            "nearest_cosine_median": float(np.median(nearest)),
            "effective_rank": rank,
            "collision_pairs": collisions,
        }
        all_upper.append(upper)
        all_nearest.append(nearest)
        rank_weighted += rank * len(indices)
        rank_count += len(indices)
        collision_pairs += collisions

    # Reverse pairs are all transition-local; compute them against the full
    # occurrence matrix only to keep their original indices.
    unit = _l2_rows(vectors.astype(np.float64, copy=False))
    reverse = np.asarray(
        [float(unit[left] @ unit[right]) for left, right in reverse_pairs], dtype=np.float64
    )
    upper = np.concatenate(all_upper)
    nearest = np.concatenate(all_nearest)
    return {
        "pairwise_cosine_p95": float(np.quantile(upper, 0.95)),
        "nearest_cosine_median": float(np.median(nearest)),
        "effective_rank": rank_weighted / max(rank_count, 1),
        "collision_pairs": collision_pairs,
        "reverse_pair_count": int(len(reverse_pairs)),
        "reverse_pair_cosine_median": float(np.median(reverse)) if len(reverse) else None,
        "reverse_pair_cosine_max": float(np.max(reverse)) if len(reverse) else None,
        "by_group": by_group,
    }


def _hard_gate(candidate: dict[str, Any], baseline: dict[str, Any]) -> tuple[bool, list[str]]:
    """Only irreversible failures block.  Anisotropy is reported, not gated."""
    failures: list[str] = []
    if candidate["collision_pairs"]:
        failures.append("collisions")
    for group, metrics in candidate["worst_nearest"].items():
        reference = baseline["worst_nearest"][group]["worst_nearest_cosine"]
        allowed = reference + HARD_GATE["max_worst_nearest_over_baseline"]
        if metrics["worst_nearest_cosine"] > allowed:
            failures.append(f"{group}:worst_nearest")
    reverse_median = candidate["reverse_pair_cosine_median"]
    if reverse_median is None or reverse_median > HARD_GATE["max_reverse_pair_median"]:
        failures.append("reverse_pairs")
    if candidate["channel_drift"]["max_channel_drift"] > HARD_GATE["max_channel_drift"]:
        failures.append("channel_drift")
    if not candidate["word_table"]["full_rank"]:
        failures.append("word_table_rank")
    if not candidate["slot_source_rank"]["full_rank"]:
        failures.append("slot_source_rank")
    if not candidate["slot_source_rank"]["fits_projection"]:
        failures.append("action_projection_bottleneck")
    if not candidate["bundle_key_is_unique"]:
        failures.append("bundle_key_collision")
    return not failures, failures


def _bundle_keys_are_unique(keyed: Iterable[tuple[str, str]]) -> bool:
    """Distinct labels must produce distinct (group, {(word, role)}) keys.

    The head slot pools its words, so two labels whose only difference is head
    ORDER would share a channel unless a role transform separates them. The data
    contract forbids that outside ``transition`` and R_B covers the gated case;
    this is the invariant that keeps a future annotation from reopening it.
    """
    seen: set[tuple[str, frozenset]] = set()
    for group, label in keyed:
        slots = action_label_slots(group, parse_action_label(label))
        key = (group, frozenset(zip(slots["word_ids"], slots["role_ids"])))
        if key in seen:
            return False
        seen.add(key)
    return True


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dirs = tuple(Path(path).resolve() for path in (args.dataset_dirs or DEFAULT_DATASET_DIRS))
    rows, _ = _load_rows(dataset_dirs)
    label_groups: dict[str, set[str]] = {}
    for row in rows:
        label_groups.setdefault(row["label"], set()).add(row["group"])
    cross_group = {label: groups for label, groups in label_groups.items() if len(groups) > 1}
    # Geometry is checkpoint-local.  Duplicate strings in two groups must be
    # represented once per group so their role gate is never inferred globally.
    keyed_labels = sorted({(row["group"], row["label"]) for row in rows})
    keyed_names = [f"{group}\0{label}" for group, label in keyed_labels]
    role_labels = [name.split("\0", 1)[1] for name in keyed_names]
    role_group_by_occurrence = {name: name.split("\0", 1)[0] for name in keyed_names}

    t5_dir = _resolve_t5_dir(args.t5_path, args.t5_model)
    t5_files = (
        "config.json", "generation_config.json", "model.safetensors",
        "spiece.model", "special_tokens_map.json", "tokenizer_config.json",
    )
    t5_hash = _sha256_files(t5_dir, t5_files)

    import torch
    from transformers import T5Config, T5EncoderModel, T5Tokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained(str(t5_dir), local_files_only=True)
    safetensors_path = t5_dir / "model.safetensors"
    if safetensors_path.is_file():
        import safetensors.torch

        config = T5Config.from_pretrained(str(t5_dir), local_files_only=True)
        encoder = T5EncoderModel(config)
        encoder.load_state_dict(safetensors.torch.load_file(str(safetensors_path)), strict=False)
    else:
        encoder = T5EncoderModel.from_pretrained(str(t5_dir), local_files_only=True)
    encoder = encoder.eval().to(device)
    word_texts = [vocab_t5_text(token) for token in CONTROLLED_VOCAB]
    atom_by_eos = _encode_both_eos_policies(
        tokenizer, encoder, device, word_texts, args.batch_size
    )
    # Baseline faithfully encodes the current comma-separated label strings.
    unique_label_texts = sorted(set(role_labels))
    full_by_eos = _encode_both_eos_policies(
        tokenizer, encoder, device, unique_label_texts, args.batch_size
    )
    unique_index = {label: index for index, label in enumerate(unique_label_texts)}

    role_payload = role_b_material(expected_dim=atom_by_eos["keep"].shape[1])
    variants: dict[str, dict[str, Any]] = {}
    reverse_pairs = _reverse_pairs(
        role_labels,
        {label: group for label, group in zip(role_labels, (role_group_by_occurrence[n] for n in keyed_names))},
    )
    # Repeated strings across groups make the mapping above ambiguous. Rebuild
    # the pair set directly over group-qualified occurrences.
    reverse_pairs = []
    for group in ("locomotion", "stationary", "transition"):
        indices = [i for i, key in enumerate(keyed_names) if key.startswith(group + "\0")]
        local_labels = [role_labels[i] for i in indices]
        local_pairs = _reverse_pairs(local_labels, {label: group for label in local_labels})
        reverse_pairs.extend((indices[left], indices[right]) for left, right in local_pairs)

    groups = [role_group_by_occurrence[name] for name in keyed_names]
    bundle_keys_unique = _bundle_keys_are_unique(keyed_labels)

    for eos_policy in ("keep", "drop"):
        baseline_vectors = np.stack(
            [full_by_eos[eos_policy][unique_index[label]] for label in role_labels]
        )
        baseline_metrics = _metrics(
            baseline_vectors, groups, reverse_pairs, HARD_GATE["collision_cosine"]
        )
        baseline_metrics["worst_nearest"] = _worst_nearest(baseline_vectors, groups)
        variants[f"baseline/eos_{eos_policy}"] = baseline_metrics

        # --- the approved representation: one channel per role slot -----------
        raw_singular = np.linalg.svd(atom_by_eos[eos_policy], compute_uv=False)
        raw_table_rank = int(np.count_nonzero(raw_singular > raw_singular[0] * 1e-10))
        for postprocess in ("raw", "center", "l2", "center_l2"):
            atoms = _postprocess_atoms(atom_by_eos[eos_policy], postprocess)
            name = f"slot/eos_{eos_policy}/{postprocess}"
            bundle = _slot_bundle(atoms, keyed_labels, role_payload)
            metrics = _metrics(bundle, groups, reverse_pairs, HARD_GATE["collision_cosine"])
            metrics["worst_nearest"] = _worst_nearest(bundle, groups)
            metrics["channel_drift"] = _channel_drift(atoms, role_payload)
            # Mean-centering costs exactly one dimension by construction, so the
            # criterion is affine independence: no word may be a combination of
            # the others, which is rank V for a raw table and V-1 for a centered
            # one.  Anything below that means the frozen table itself lost a word.
            singular = np.linalg.svd(atoms, compute_uv=False)
            rank = int(np.count_nonzero(singular > singular[0] * 1e-10))
            expected_rank = len(CONTROLLED_VOCAB) - (1 if postprocess.startswith("center") else 0)
            metrics["word_table"] = {
                "rank": rank,
                "expected_rank": expected_rank,
                "raw_rank": raw_table_rank,
                "full_rank": rank >= expected_rank and raw_table_rank == len(CONTROLLED_VOCAB),
                "condition_number": float(singular[0] / max(singular[rank - 1], 1e-12)),
            }
            metrics["slot_source_rank"] = _slot_source_rank_report(
                atoms, role_payload, args.latent_dim
            )
            metrics["bundle_key_is_unique"] = bundle_keys_unique
            if not args.skip_exhaustive:
                metrics["slot_configurations"] = _slot_configuration_margins(
                    atoms, role_payload, args.max_subset_size, device
                )
            # Reported only: the same numbers the rejected family was blocked on.
            metrics["reported_vs_baseline"] = {
                group: {
                    key: metrics["by_group"][group][key] - baseline_metrics["by_group"][group][key]
                    for key in ("pairwise_cosine_p95", "nearest_cosine_median")
                } | {
                    "effective_rank_ratio": metrics["by_group"][group]["effective_rank"]
                    / max(baseline_metrics["by_group"][group]["effective_rank"], 1e-12)
                }
                for group in sorted(set(groups))
            }
            passed, failures = _hard_gate(metrics, baseline_metrics)
            metrics["go"] = passed
            metrics["failures"] = failures
            variants[name] = metrics

    # Feasibility is decided by the hard gate alone.  Among the survivors the
    # ranking maximises the irreversible margin (the worst-case near pair), but
    # only differences above SELECTION_TOLERANCE count as evidence: EOS and
    # vector postprocess separate by ~1e-3 here, and a fourth-decimal coin flip
    # must not be what fixes a contract fingerprint for the next training run.
    # Within the tolerance band the canonical order below decides, so a rerun on
    # another machine picks the same variant.
    canonical_order = [
        f"slot/eos_{eos}/{postprocess}"
        for eos in ("keep", "drop")
        for postprocess in ("center_l2", "center", "l2", "raw")
    ]
    eligible = [
        (name, metrics) for name, metrics in variants.items()
        if name.startswith("slot/") and metrics.get("go")
    ]

    def worst(metrics: dict[str, Any]) -> float:
        return max(g["worst_nearest_cosine"] for g in metrics["worst_nearest"].values())

    selected_name = None
    if eligible:
        best = min(worst(metrics) for _name, metrics in eligible)
        within = {
            name for name, metrics in eligible
            if worst(metrics) <= best + SELECTION_TOLERANCE
        }
        selected_name = next(name for name in canonical_order if name in within)

    embedding_contract = None
    conditioning_contract = None
    if selected_name:
        parts = selected_name.split("/")
        eos_policy = parts[1].removeprefix("eos_")
        postprocess = parts[2]
        # The selected variant's own table, not just its width: the contract now
        # commits to the vectors, so the report's fingerprint has to be the one
        # the builder reproduces when it encodes this variant.
        selected_atoms = _postprocess_atoms(atom_by_eos[eos_policy], postprocess)
        token_to_text = {token: vocab_t5_text(token) for token in CONTROLLED_VOCAB}
        embedding_contract = embedding_contract_payload(
            token_to_text=token_to_text,
            t5_name=args.t5_model,
            t5_artifact_sha256=t5_hash,
            tokenizer_class=type(tokenizer).__name__,
            tokenizer_version=importlib.metadata.version("transformers"),
            pooling="masked_mean",
            eos_policy=eos_policy,
            vector_postprocess=postprocess,
            embedding_dim=int(selected_atoms.shape[1]),
            dtype="float32",
            word_table_sha256=word_table_sha256(selected_atoms),
        )
        embedding_fp = fingerprint(embedding_contract)
        conditioning_contract = conditioning_contract_payload(
            embedding_fingerprint=embedding_fp,
            role_b_material_sha256=role_payload["material_sha256"],
            representation=slot_channel_representation(),
        )

    return {
        "schema_version": 2,
        "status": "GO" if selected_name else "NO_GO",
        "selected_variant": selected_name,
        "selection_rule": "role-slot-channel candidates passing every hard criterion; then the lowest worst-case nearest cosine outside a 0.005 tolerance band; then fixed canonical order",
        "thresholds": {
            "hard": HARD_GATE,
            "projection_latent_dim": args.latent_dim,
            "reported_only": list(REPORTED_METRICS),
        },
        "corpus": {
            "dataset_dirs": [str(path) for path in dataset_dirs],
            "action_labels_sha256": {
                str(path): hashlib.sha256(
                    (path / ACTION_LABELS_FILE).read_bytes()
                ).hexdigest()
                for path in dataset_dirs
            },
            "rows": len(rows),
            "distinct_labels": len(set(role_labels)),
            "group_qualified_labels": len(role_labels),
            "cross_group_label_strings": len(cross_group),
            "reverse_pairs": len(reverse_pairs),
        },
        "encoder": {
            "t5_name": args.t5_model,
            "local_path": str(t5_dir),
            "artifact_sha256": t5_hash,
            "transformers_version": importlib.metadata.version("transformers"),
            "torch_version": torch.__version__,
            "device": device,
        },
        "role_b": {
            "namespace": role_payload["namespace"],
            "material_sha256": role_payload["material_sha256"],
            "embedding_dim": role_payload["embedding_dim"],
        },
        "variants": variants,
        "embedding_contract": embedding_contract,
        "embedding_fingerprint": fingerprint(embedding_contract) if embedding_contract else None,
        "conditioning_contract": conditioning_contract,
        "conditioning_contract_fingerprint": fingerprint(conditioning_contract) if conditioning_contract else None,
    }


def _print_summary(report: dict[str, Any]) -> None:
    print(f"status: {report['status']}")
    print(f"selected: {report['selected_variant']}")
    print(
        "corpus: "
        f"{report['corpus']['rows']} rows, "
        f"{report['corpus']['distinct_labels']} labels, "
        f"{report['corpus']['reverse_pairs']} reverse pair(s)"
    )
    print()
    print("HARD gate -- irreversible properties only")
    print("variant                          worstNN  rev50   drift  word  src  keys  gate")
    for name, metrics in report["variants"].items():
        if not name.startswith(("slot/", "baseline/")):
            continue
        reverse = metrics["reverse_pair_cosine_median"]
        worst = max(g["worst_nearest_cosine"] for g in metrics["worst_nearest"].values())
        if "go" not in metrics:
            print(f"{name:32s} {worst:7.4f} {reverse:6.3f}     ---   ---  ---   ---  BASE")
            continue
        gate = "GO" if metrics["go"] else "NO:" + ",".join(metrics["failures"])
        table = metrics["word_table"]
        print(
            f"{name:32s} {worst:7.4f} {reverse:6.3f} "
            f"{metrics['channel_drift']['max_channel_drift']:7.1e} "
            f"{table['rank']:5d} {metrics['slot_source_rank']['total_rank']:4d} "
            f"{str(metrics['bundle_key_is_unique']):>5s}  {gate}"
        )
    selected = report["variants"].get(report["selected_variant"] or "")
    if selected:
        print()
        print("REPORTED geometry of the selected variant (never blocking)")
        for group, delta in selected["reported_vs_baseline"].items():
            print(
                f"  {group:11s} p95 {delta['pairwise_cosine_p95']:+.3f}  "
                f"nn50 {delta['nearest_cosine_median']:+.3f}  "
                f"erank ratio {delta['effective_rank_ratio']:.3f}   (vs full-string baseline)"
            )
        configurations = selected.get("slot_configurations")
        if configurations:
            print(
                "  slot diagnostic: complete head/direction domains; modifier "
                f"subsets up to {configurations['max_modifier_subset_size']} words:"
            )
            for slot in ACTION_LABEL_SLOTS:
                entry = configurations[slot]
                print(
                    f"    {slot:10s} {entry['configurations']:6d} configs  worst pair "
                    f"{entry['worst_pair_cosine']:.4f}  membership margin "
                    f"{entry['min_membership_margin']:+.3f} ({entry['min_membership_margin_word']})"
                )
    if report["embedding_fingerprint"]:
        print(f"embedding_fingerprint: {report['embedding_fingerprint']}")
        print(
            "conditioning_contract_fingerprint: "
            f"{report['conditioning_contract_fingerprint']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dirs", nargs="*", help="Processed dataset directories; defaults to all three current sources.")
    parser.add_argument("--t5-model", default="t5-base")
    parser.add_argument("--t5-path", default=None, help="Local HuggingFace model directory.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=256,
        help="Width of action_label_projection's first output (default: 256). The "
             "hard gate requires it to contain the complete slot-source space.",
    )
    parser.add_argument(
        "--skip-exhaustive",
        action="store_true",
        help="Skip the quantitative configuration diagnostic (~1 min on CPU). "
             "Full-domain injectivity remains certified by the slot-source rank gate.",
    )
    parser.add_argument(
        "--max-subset-size",
        type=int,
        default=3,
        help="Largest modifier subset used by the quantitative diagnostic. Default 3 = "
             "the current corpus maximum; full-domain correctness is covered separately "
             "by the algebraic slot-source rank gate.",
    )
    parser.add_argument("--json", action="store_true", help="Print the complete machine-readable report.")
    args = parser.parse_args()
    report = evaluate(args)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _print_summary(report)
    return 0 if report["status"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
