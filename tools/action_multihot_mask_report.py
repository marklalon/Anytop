#!/usr/bin/env python3
"""Recompute GROUP_MULTIHOT_MASK from the corpus, with the object_subset axis.

``GROUP_MULTIHOT_MASK`` (data_loaders/truebones/truebones_utils/motion_labels.py)
is a FROZEN CONSTANT on purpose -- adding clips must never silently redefine what
a slot means. This tool is how it is recomputed when the corpus grows: it prints
the per-(group, word) support table and the mask rows the current corpus implies,
which are then reviewed and committed by hand.

The rule (see docs/canonical_frame_and_label_transfer.md section 4.2) is::

    keep a slot in a group iff  clips >= 10  AND  species >= 5  AND  subs >= 3

where ``subs`` is the number of object_subsets in which at least ``3`` distinct
species carry the word. The body-plan axis is the one a clip or species count
cannot see: ``swim`` clears 59 clips / 11 species library-wide, yet lives on two
body plans only, at 12.5 clips per winged species -- the memorization fingerprint
(walk / run sit at 1.7-3.9). A slot fitted like that binds the word to those
species' body plan, which is exactly what cross-species transfer must not do.

Usage:
    python tools/action_multihot_mask_report.py [--datasets dataset/datasets.jsonl]
                                                [--group locomotion] [--all-words]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ANYTOP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ANYTOP_DIR.parent))
sys.path.insert(0, str(ANYTOP_DIR))

from data_loaders.truebones.truebones_utils.dataset_sources import (  # noqa: E402
    load_datasets_manifest,
)
from data_loaders.truebones.truebones_utils.dataset_tags import (  # noqa: E402
    SPECIES_TAGS_FILE,
    load_species_tags,
)
from data_loaders.truebones.truebones_utils.motion_labels import (  # noqa: E402
    ACTION_GROUPS,
    ACTION_VOCAB_CORE,
    GROUP_MULTIHOT_MASK,
    load_action_labels,
    vocab_words_in,
)

# The thresholds the committed mask is fitted with.
MIN_CLIPS = 10
MIN_SPECIES = 5
MIN_SPECIES_PER_SUBSET = 3
MIN_SUBSETS = 3


def _match_species(clip_name, species_by_name):
    """Longest-prefix match of ``{species}_...npy`` inside one source."""
    best_key, best_len = None, -1
    for species, key in species_by_name.items():
        if clip_name.startswith(species + "_") and len(species) > best_len:
            best_key, best_len = key, len(species)
    return best_key


def collect_support(manifest_path):
    """``{group: {word: {subset: {species}}}}`` plus per-word clip counts.

    Species are keyed ``<namespace>/<species>`` so two datasets' same-named
    species do not merge (they are different rigs, and the mask is fitted on the
    merged corpus a group's model actually trains on).
    """
    sources = load_datasets_manifest(manifest_path)
    support = {group: defaultdict(lambda: defaultdict(set)) for group in ACTION_GROUPS}
    clips = {group: defaultdict(int) for group in ACTION_GROUPS}
    group_clip_total = defaultdict(int)
    unmatched = 0

    for source in sources:
        root = Path(source.root)
        tags = load_species_tags(root / SPECIES_TAGS_FILE)
        subset_of = {
            source.namespace + "/" + species: species_tags[0].strip().lower()
            for species, species_tags in tags.items() if species_tags
        }
        species_by_name = {species: source.namespace + "/" + species for species in tags}
        for clip, entry in load_action_labels(root).items():
            group = entry["action_group"]
            if group not in support:
                continue
            key = _match_species(clip, species_by_name)
            subset = subset_of.get(key) if key is not None else None
            if subset is None:
                unmatched += 1
                continue
            group_clip_total[group] += 1
            for word in vocab_words_in(entry["action_label"], core_only=True):
                clips[group][word] += 1
                support[group][word][subset].add(key)

    return support, clips, group_clip_total, unmatched


def word_stats(group, word, support, clips):
    """Support of one word inside one group, and whether the rule keeps it."""
    by_subset = support[group].get(word, {})
    species = set().union(*by_subset.values()) if by_subset else set()
    qualifying = {
        subset: members for subset, members in by_subset.items()
        if len(members) >= MIN_SPECIES_PER_SUBSET
    }
    n_clips = clips[group].get(word, 0)
    keep = (
        n_clips >= MIN_CLIPS
        and len(species) >= MIN_SPECIES
        and len(qualifying) >= MIN_SUBSETS
    )
    # Densest body plan: clips-per-species there is the memorization fingerprint.
    densest, densest_members = max(
        by_subset.items(), key=lambda item: len(item[1]), default=(None, set()),
    )
    return {
        "clips": n_clips,
        "species": len(species),
        "subs": len(qualifying),
        "n_subsets": len(by_subset),
        "densest": densest,
        "densest_species": len(densest_members),
        "keep": keep,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--datasets", default="dataset/datasets.jsonl")
    parser.add_argument("--group", action="append", choices=list(ACTION_GROUPS),
                        help="Report only these groups (repeatable). Default: all.")
    parser.add_argument("--all-words", action="store_true",
                        help="Also list words with zero support in the group.")
    args = parser.parse_args()

    support, clips, totals, unmatched = collect_support(args.datasets)
    groups = args.group or list(ACTION_GROUPS)
    if unmatched:
        print("[WARN] " + str(unmatched) + " clip(s) could not be resolved to a tagged species\n")

    print("rule: clips >= {} AND species >= {} AND subs >= {}  (subs = object_subsets "
          "with >= {} species)\n".format(MIN_CLIPS, MIN_SPECIES, MIN_SUBSETS,
                                         MIN_SPECIES_PER_SUBSET))

    proposed = {}
    for group in groups:
        print("=== {}  ({} clips) ".format(group, totals[group]) + "=" * 28)
        print("{:<10}{:>7}{:>9}{:>6}{:>9}{:>12}{:>10}   keep   committed".format(
            "word", "clips", "species", "subs", "/subsets", "densest", "clips/sp"))
        row_bits = []
        for word, committed in zip(ACTION_VOCAB_CORE, GROUP_MULTIHOT_MASK[group]):
            stats = word_stats(group, word, support, clips)
            row_bits.append(1 if stats["keep"] else 0)
            if not args.all_words and stats["clips"] == 0 and not committed:
                continue
            density = (stats["clips"] / stats["densest_species"]) if stats["densest_species"] else 0.0
            flag = "" if int(stats["keep"]) == committed else "   <-- CHANGED"
            print("{:<10}{:>7}{:>9}{:>6}{:>9}{:>12}{:>10.1f}{:>7}{:>11}{}".format(
                word, stats["clips"], stats["species"], stats["subs"], stats["n_subsets"],
                str(stats["densest"]), density, int(stats["keep"]), committed, flag))
        proposed[group] = tuple(row_bits)
        print()

    print("Proposed GROUP_MULTIHOT_MASK rows (review, then commit by hand):\n")
    print("    #             " + " ".join("{:>4}".format(w[:4]) for w in ACTION_VOCAB_CORE))
    for group in groups:
        bits = " ".join("{:>4}".format(bit) for bit in proposed[group])
        print('    "{}": ({}),'.format(group, bits))
    changed = [g for g in groups if proposed[g] != tuple(GROUP_MULTIHOT_MASK[g])]
    print()
    print("committed mask is up to date" if not changed
          else "DIFFERS from the committed mask: " + ", ".join(changed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
