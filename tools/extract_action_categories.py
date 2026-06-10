import json
from pathlib import Path

# Load the metadata JSON file
metadata_path = Path(__file__).resolve().parent.parent / 'dataset' / 'truebones' / 'zoo' / 'truebones_processed' / 'motion_metadata.json'

with open(metadata_path, 'r', encoding='utf-8') as handle:
    payload = json.load(handle)

# Extract all action_tags values
motions = payload.get('motions', payload)

# --- Compute per-tag statistics ---
tag_stats = {}  # tag -> {"motion_count": int, "frame_count": int}

for motion_name, entry in motions.items():
    raw_tags = entry.get('action_tags')
    if raw_tags is None and 'action_category' in entry:
        raw_tags = [entry['action_category']]
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
    if raw_tags is None and 'action_category' in entry:
        raw_tags = [entry['action_category']]
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
    if raw_tags is None and 'action_category' in entry:
        raw_tags = [entry['action_category']]
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
