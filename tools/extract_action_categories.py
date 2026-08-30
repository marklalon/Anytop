import argparse
import sys
from pathlib import Path

# Add project root so we can import from data_loaders
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
from data_loaders.truebones.truebones_utils.dataset_tags import configure as configure_dataset_tags, dataset_tags
from data_loaders.truebones.truebones_utils.param_utils import get_dataset_dir
from data_loaders.truebones.truebones_utils.motion_labels import (
    ACTION_GROUPS,
    ACTION_VOCAB_CORE,
    action_multihot_words,
    group_multihot_mask,
    load_motion_metadata,
    vocab_words_in,
)

parser = argparse.ArgumentParser(
    description="Action group / controlled-word statistics over a processed dataset.",
)
parser.add_argument(
    "--objects_subset",
    type=str,
    nargs="*",
    default=None,
    help="Filter by object_type(s). Supports group names (e.g. quadruped biped winged) "
         "from species_tags.jsonl, or individual PascalCase names (e.g. Horse Buffalo Camel). "
         "If omitted, all objects are included.",
)
parser.add_argument(
    "--dataset-dir", "--dataset_dir",
    dest="dataset_dir",
    type=str,
    default=None,
    help="Processed dataset directory to read motion_metadata.json and the tag sidecars "
         "from. Defaults to the standard truebones_processed directory.",
)
parser.add_argument(
    "--action_words",
    type=str,
    nargs="*",
    default=None,
    help="For each given controlled word, print a per-species (object_type) count, "
         "sorted descending. Species with zero count are omitted.",
)
args = parser.parse_args()

# Allow comma-separated --action_words (e.g. --action_words jump,turn)
if args.action_words:
    args.action_words = [t for item in args.action_words for t in item.split(",") if t.strip()]

# Load motion metadata; action_group / action_label are merged in from
# action_labels.jsonl. This reads only per-dataset sidecars (never cond.npy), so
# object_type here is the BARE species name and the tag snapshot is configured
# for that one dataset.
dataset_dir = Path(get_dataset_dir(args.dataset_dir))
configure_dataset_tags(dataset_dir=dataset_dir)
motions = load_motion_metadata(dataset_dir)

# --- Optional filter by object_type ---
if args.objects_subset:
    filter_set: set[str] = set()
    tags = dataset_tags()
    for name in args.objects_subset:
        filter_set.update(tags.species_for(name))
    motions = {k: v for k, v in motions.items() if v.get("object_type") in filter_set}
    print(f"Filtered to {len(motions)} motions matching object_type in {sorted(filter_set)}\n")

total_motions = len(motions)
if total_motions == 0:
    print("No motions matched the filter. Nothing to report.")
    sys.exit(0)


def _frame_count(entry) -> int:
    sfr = entry.get("source_frame_range")
    return sfr[1] - sfr[0] + 1 if sfr else 0


total_frames = sum(_frame_count(entry) for entry in motions.values())

if not args.action_words:
    # --- Group distribution: this is what decides the training splits ---
    print(f"{'Action Group':<20s} {'Motions':>8s} {'%Motions':>9s} {'Frames':>10s} {'%Frames':>9s}")
    print("-" * 60)
    group_stats: dict[str, dict[str, int]] = {}
    for entry in motions.values():
        group = str(entry.get("action_group") or "?")
        stats = group_stats.setdefault(group, {"motion_count": 0, "frame_count": 0})
        stats["motion_count"] += 1
        stats["frame_count"] += _frame_count(entry)
    for group in sorted(group_stats, key=lambda g: group_stats[g]["motion_count"], reverse=True):
        s = group_stats[group]
        mpct = s["motion_count"] / total_motions * 100
        fpct = s["frame_count"] / total_frames * 100 if total_frames else 0.0
        print(f"{group:<20s} {s['motion_count']:>8d} {mpct:>8.2f}% {s['frame_count']:>10d} {fpct:>8.2f}%")
    print("-" * 60)
    print(f"{'TOTAL':<20s} {total_motions:>8d} {'100.00%':>9s} {total_frames:>10d} {'100.00%':>9s}")

    # --- Core-word support, per group ---
    # The per-group columns are the numbers GROUP_MULTIHOT_MASK is frozen from:
    # a word needs >= 10 clips AND >= 5 species inside a group to keep its slot
    # there. 'masked' marks a word this group currently holds at zero.
    print(f"\n\n{'Core word':<12s}" + "".join(f"{g:>26s}" for g in ACTION_GROUPS))
    print("-" * (12 + 26 * len(ACTION_GROUPS)))
    per_group_clips: dict[str, dict[str, int]] = {g: {} for g in ACTION_GROUPS}
    per_group_species: dict[str, dict[str, set]] = {g: {} for g in ACTION_GROUPS}
    for entry in motions.values():
        group = str(entry.get("action_group") or "")
        if group not in ACTION_GROUPS:
            continue
        species = str(entry.get("object_type") or "?")
        for word in action_multihot_words(str(entry.get("action_label") or "")):
            per_group_clips[group][word] = per_group_clips[group].get(word, 0) + 1
            per_group_species[group].setdefault(word, set()).add(species)
    for index, word in enumerate(ACTION_VOCAB_CORE):
        row = f"{word:<12s}"
        for group in ACTION_GROUPS:
            clips = per_group_clips[group].get(word, 0)
            species = len(per_group_species[group].get(word, ()))
            masked = "" if group_multihot_mask(group)[index] else "  masked"
            row += f"{clips:>10d}/{species:<3d}{masked:<9s}"[:26]
        print(row)

    # --- Labels the text path cannot reach ---
    empty = [name for name, entry in motions.items() if not entry.get("action_label")]
    no_core = [
        name for name, entry in motions.items()
        if entry.get("action_label") and not action_multihot_words(str(entry["action_label"]))
    ]
    print(f"\n\nEmpty labels (train unconditioned): {len(empty)}")
    for name in empty[:20]:
        print(f"  \033[93m{name}\033[0m")
    if len(empty) > 20:
        print(f"  ... (+{len(empty) - 20} more)")
    print(f"\nLabels with no core word (detail words only, all-zero multi-hot): {len(no_core)}")
    for name in no_core[:20]:
        print(f"  {name}: {motions[name]['action_label']!r}")
    if len(no_core) > 20:
        print(f"  ... (+{len(no_core) - 20} more)")

    # --- Core words per label ---
    print(f"\n\n{'Core words per label':<24s} {'Motions':>8s} {'%Motions':>9s}")
    print("-" * 43)
    word_count_dist: dict[int, int] = {}
    for entry in motions.values():
        n = len(action_multihot_words(str(entry.get("action_label") or "")))
        word_count_dist[n] = word_count_dist.get(n, 0) + 1
    for n in sorted(word_count_dist):
        cnt = word_count_dist[n]
        pct = cnt / total_motions * 100
        label = f"{n} word{'s' if n != 1 else ''}"
        print(f"{label:<24s} {cnt:>8d} {pct:>8.2f}%")
    multi = sum(cnt for n, cnt in word_count_dist.items() if n > 1)
    print("-" * 43)
    print(f"{'Multiple words':<24s} {multi:>8d} {multi / total_motions * 100:>8.2f}%")

# --- Per-species breakdown for requested --action_words ---
if args.action_words:
    word_set = {t.strip().lower() for t in args.action_words}

    # species -> count of motions whose label hits ANY of the requested words
    species_counts: dict[str, int] = {}
    for entry in motions.values():
        species = entry.get("object_type", "Unknown")
        hits = set(vocab_words_in(str(entry.get("action_label") or "")))
        if hits & word_set:
            species_counts[species] = species_counts.get(species, 0) + 1

    # Sort descending, drop zeros
    ranked = [(s, c) for s, c in species_counts.items() if c > 0]
    ranked.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'='*60}")
    print(f"Per-species motions matching words: {', '.join(sorted(word_set))}")
    print(f"{'='*60}")
    print(f"  {'Species':<35s} {'Motions':>8s}")
    print(f"  {'-'*45}")
    for species, cnt in ranked:
        print(f"  {species:<35s} {cnt:>8d}")
    total = sum(c for _, c in ranked)
    print(f"  {'-'*45}")
    print(f"  {'TOTAL':<35s} {total:>8d}")
