import argparse
import sys
from pathlib import Path

# Add project root so we can import from data_loaders
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
from data_loaders.truebones.truebones_utils.dataset_tags import configure as configure_dataset_tags, dataset_tags
from data_loaders.truebones.truebones_utils.param_utils import get_dataset_dir
from data_loaders.truebones.truebones_utils.motion_labels import load_motion_metadata

parser = argparse.ArgumentParser(description="Extract action category statistics from motion metadata.")
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
    "--action_tags",
    type=str,
    nargs="*",
    default=None,
    help="For each given action tag, print a per-species (object_type) count, "
         "sorted descending. Species with zero count are omitted.",
)
args = parser.parse_args()

# Allow comma-separated --action_tags (e.g. --action_tags jump,turn)
if args.action_tags:
    args.action_tags = [t for item in args.action_tags for t in item.split(",") if t.strip()]

# Load motion metadata; action_tags are merged in from action_tags.jsonl.
# This reads only per-dataset sidecars (never cond.npy), so object_type here is
# the BARE species name and the tag snapshot is configured for that one dataset.
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

# --- Compute per-tag statistics ---
tag_stats = {}  # tag -> {"motion_count": int, "frame_count": int}

for motion_name, entry in motions.items():
    raw_tags = entry.get('action_tags')
    if isinstance(raw_tags, str):
        raw_tags = [raw_tags]

    # Frame count from source_frame_range
    sfr = entry.get('source_frame_range')
    frame_count = sfr[1] - sfr[0] + 1 if sfr else 0

    for tag in raw_tags or []:
        tag_text = str(tag).strip()
        if not tag_text:
            continue
        if tag_text not in tag_stats:
            tag_stats[tag_text] = {"motion_count": 0, "frame_count": 0}
        tag_stats[tag_text]["motion_count"] += 1
        tag_stats[tag_text]["frame_count"] += frame_count

total_motions = len(motions)
total_frames = sum(
    sfr[1] - sfr[0] + 1
    for entry in motions.values()
    if (sfr := entry.get("source_frame_range"))
)

if total_motions == 0:
    print("No motions matched the filter. Nothing to report.")
    sys.exit(0)

if not args.action_tags:
    print(f"{'Action Tag':<45s} {'Motions':>8s} {'%Motions':>9s} {'Frames':>8s} {'%Frames':>9s}")
    print("-" * 79)
    for tag in sorted(tag_stats, key=lambda t: tag_stats[t]["motion_count"], reverse=True):
        s = tag_stats[tag]
        mpct = s["motion_count"] / total_motions * 100
        fpct = s["frame_count"] / total_frames * 100
        print(
            f"{tag:<45s} {s['motion_count']:>8d} {mpct:>8.2f}% "
            f"{s['frame_count']:>8d} {fpct:>8.2f}%"
        )

    print("-" * 79)
    print(f"{'TOTAL':<45s} {total_motions:>8d} {'100.00%':>9s} {total_frames:>8d} {'100.00%':>9s}")
    print(f"\nTotal unique tags: {len(tag_stats)}")
    print(f"Total motions: {total_motions}")
    print(f"Total frames: {total_frames}")

    # --- List motions with 'unknown' tag ---
    unknown_motions: list[str] = []
    for motion_name, entry in motions.items():
        raw_tags = entry.get('action_tags')
        if isinstance(raw_tags, str):
            raw_tags = [raw_tags]
        tags_clean = [str(t).strip() for t in (raw_tags or []) if str(t).strip()]
        if 'unknown' in tags_clean:
            unknown_motions.append(motion_name)

    if unknown_motions:
        print(f"\n\n\033[93mMotions with 'unknown' action tag ({len(unknown_motions)}):\033[0m")
        for name in unknown_motions:
            print(f"  \033[93m{name}\033[0m")

    # --- Multi-tag statistics ---
    print(f"\n\n{'Tags per motion':<20s} {'Motions':>8s} {'%Motions':>9s}")
    print("-" * 39)
    tag_count_dist: dict[int, int] = {}
    for motion_name, entry in motions.items():
        raw_tags = entry.get('action_tags')
        if isinstance(raw_tags, str):
            raw_tags = [raw_tags]
        tags_clean = [str(t).strip() for t in (raw_tags or []) if str(t).strip()]
        n = len(tags_clean)
        tag_count_dist[n] = tag_count_dist.get(n, 0) + 1

    for n in sorted(tag_count_dist):
        cnt = tag_count_dist[n]
        pct = cnt / total_motions * 100
        label = f"{n} tag{'s' if n > 1 else ''}"
        print(f"{label:<20s} {cnt:>8d} {pct:>8.2f}%")

    multi = sum(cnt for n, cnt in tag_count_dist.items() if n > 1)
    multi_pct = multi / total_motions * 100
    print("-" * 39)
    print(f"{'Multiple tags':<20s} {multi:>8d} {multi_pct:>8.2f}%")

# --- Per-species breakdown for requested --action_tags ---
if args.action_tags:
    tag_set = {t.strip() for t in args.action_tags}

    # species -> count of motions matching ANY of the requested tags
    species_counts: dict[str, int] = {}
    for motion_name, entry in motions.items():
        species = entry.get("object_type", "Unknown")
        raw_tags = entry.get('action_tags')
        if isinstance(raw_tags, str):
            raw_tags = [raw_tags]
        tags_clean = {str(t).strip() for t in (raw_tags or []) if str(t).strip()}
        if tags_clean & tag_set:
            species_counts[species] = species_counts.get(species, 0) + 1

    # Sort descending, drop zeros
    ranked = [(s, c) for s, c in species_counts.items() if c > 0]
    ranked.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'='*60}")
    print(f"Per-species motions matching tags: {', '.join(sorted(tag_set))}")
    print(f"{'='*60}")
    print(f"  {'Species':<35s} {'Motions':>8s}")
    print(f"  {'-'*45}")
    for species, cnt in ranked:
        print(f"  {species:<35s} {cnt:>8d}")
    total = sum(c for _, c in ranked)
    print(f"  {'-'*45}")
    print(f"  {'TOTAL':<35s} {total:>8d}")
