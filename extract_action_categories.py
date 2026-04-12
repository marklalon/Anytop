import json
from pathlib import Path

# Load the metadata JSON file
metadata_path = Path('dataset/truebones/zoo/truebones_processed/motion_metadata.json')

with open(metadata_path, 'r', encoding='utf-8') as handle:
    payload = json.load(handle)

# Extract all action_category values
motions = payload.get('motions', payload)
action_categories = set()

for motion_name, entry in motions.items():
    if 'action_category' in entry:
        action_categories.add(entry['action_category'])

# Print unique values sorted alphabetically
print("Unique action_category values:")
for category in sorted(action_categories):
    print(f"  - {category}")

print(f"\nTotal unique categories: {len(action_categories)}")
