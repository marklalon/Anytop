#!/usr/bin/env python3
"""Count the frequency of the words of one action-label slot.

Reads ``dataset/datasets.jsonl``, merges the ``action_labels.jsonl`` sidecars
of every enabled dataset, and reports the words of the selected slot of each
``action_label`` as one pooled frequency table.  No train/val split is
applied.  A label contributes one row PER word in the slot (a head slot may
hold up to ACTION_LABEL_MAX_HEADS words, e.g. ``"attack, jump, charge"``
counts both ``attack`` and ``jump``), so per-slot rows can exceed the clip
count.

Slots are the vocabulary's own partition (``motion_labels``): ``head``
(default) = HEAD_VOCAB words (the state/event the label is about),
``modifier`` = MODIFIER_VOCAB words, ``direction`` = DIRECTION_VOCAB words.
A clip whose label holds no word of the selected slot is reported as missing
for that slot.

Usage:
    python tools/count_action_label_words.py
    python tools/count_action_label_words.py --slot modifier
    python tools/count_action_label_words.py --slot direction
    python tools/count_action_label_words.py --datasets dataset/datasets.jsonl
    python tools/count_action_label_words.py --by-dataset
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_anytop_root = Path(__file__).resolve().parent.parent
if str(_anytop_root) not in sys.path:
    sys.path.insert(0, str(_anytop_root))

from data_loaders.truebones.truebones_utils.dataset_sources import (
    load_datasets_manifest,
)
from data_loaders.truebones.truebones_utils.motion_labels import (
    HEAD_VOCAB,
    MODIFIER_VOCAB,
    DIRECTION_VOCAB,
)

_DEFAULT_MANIFEST = _anytop_root / "dataset" / "datasets.jsonl"

# The vocabulary's own slot partition (word_slots in the conditioning
# contract): a word's slot is a property of the word, never of its position.
SLOT_VOCABS = {
    "head": HEAD_VOCAB,
    "modifier": MODIFIER_VOCAB,
    "direction": DIRECTION_VOCAB,
}
SLOTS = tuple(SLOT_VOCABS)


def slot_words_of(label, slot):
    """Return the words of *label* that belong to *slot* (may be empty)."""
    label = (label or "").strip()
    if not label:
        return []
    vocab = SLOT_VOCABS[slot]
    return [w.strip() for w in label.split(",") if w.strip() in vocab]


def count_sidecar(path, slot):
    """Return (word_counter, n_clips_without_slot) for one sidecar.

    Each clip contributes one row per word it has in the slot, so the counter
    total can exceed the clip count (a label can carry two head words).
    """
    counter = Counter()
    n_missing = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            words = slot_words_of(row.get("action_label"), slot)
            if not words:
                n_missing += 1
            else:
                counter.update(words)
    return counter, n_missing


def print_table(counter, total, title, slot):
    print("=" * 55)
    print(f"{title} ({slot}-slot words: {total})")
    print("=" * 55)
    print(f"{slot + ' word':<20} {'count':>8} {'pct':>9}  bar")
    print("-" * 55)
    max_count = max(counter.values()) if counter else 1
    for word, count in counter.most_common():
        pct = 100.0 * count / total if total else 0.0
        bar = "#" * int(round(40 * count / max_count))
        print(f"{word:<20} {count:>8} {pct:>8.2f}%  {bar}")
    print(f"distinct {slot} words: {len(counter)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default=str(_DEFAULT_MANIFEST),
        help="Path to the datasets.jsonl manifest (absolute, or relative to "
        "the Anytop dir). Default: dataset/datasets.jsonl.",
    )
    parser.add_argument(
        "--slot",
        choices=SLOTS,
        default="head",
        help="Label slot to count, per the vocabulary partition: 'head' = "
        "HEAD_VOCAB words (up to two per label), 'modifier' = MODIFIER_VOCAB, "
        "'direction' = DIRECTION_VOCAB. Default: head.",
    )
    parser.add_argument(
        "--by-dataset",
        action="store_true",
        help="Also print a per-dataset breakdown after the merged table.",
    )
    args = parser.parse_args()
    slot = args.slot

    sources = load_datasets_manifest(args.datasets)  # disabled rows already dropped

    merged = Counter()
    per_dataset = {}
    total_missing = 0
    for source in sources:
        sidecar = Path(source.root) / "action_labels.jsonl"
        if not sidecar.is_file():
            print(f"warning: {sidecar} not found, skipping", file=sys.stderr)
            continue
        counter, n_missing = count_sidecar(sidecar, slot)
        merged.update(counter)
        per_dataset[source.namespace] = (source, counter)
        total_missing += n_missing

    total = sum(merged.values())
    if total == 0:
        sys.exit(f"No {slot}-slot words found in any enabled dataset.")

    print_table(merged, total, "ALL DATASETS (merged)", slot)

    if total_missing:
        print(f"\nclips with no {slot}-slot word: {total_missing}")

    if args.by_dataset:
        for namespace, (source, counter) in per_dataset.items():
            print()
            print_table(
                counter,
                sum(counter.values()),
                f"DATASET: {source.portable_root}",
                slot,
            )


if __name__ == "__main__":
    main()
