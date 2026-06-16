import argparse
import sys
from pathlib import Path

# Add project root so we can import from data_loaders
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
from data_loaders.truebones.truebones_utils.param_utils import OBJECT_SUBSETS_DICT
from data_loaders.truebones.truebones_utils.motion_labels import load_motion_metadata

parser = argparse.ArgumentParser(description="Extract action category statistics from motion metadata.")
parser.add_argument(
    "--objects_subset",
    type=str,
    nargs="*",
    default=None,
    help="Filter by object_type(s). Supports group names (e.g. quadruped biped winged) "
         "from OBJECT_SUBSETS_DICT, or individual PascalCase names (e.g. Horse Buffalo Camel). "
         "If omitted, all objects are included.",
)
args = parser.parse_args()

# Load motion metadata; action_tags are merged in from action_tags.jsonl.
dataset_dir = _project_root / 'dataset' / 'truebones' / 'zoo' / 'truebones_processed'
motions = load_motion_metadata(dataset_dir)

# --- Optional filter by object_type ---
if args.objects_subset:
    filter_set: set[str] = set()
    for name in args.objects_subset:
        if name in OBJECT_SUBSETS_DICT:
            filter_set.update(OBJECT_SUBSETS_DICT[name])
        else:
            filter_set.add(name)
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
