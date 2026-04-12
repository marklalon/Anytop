#!/usr/bin/env python3
"""
Extract character condition data from cond.npy and save to JSON.

Extracts:
  - end_effector_names
  - contact_joint_names
  - is_symmetric
  
for each character and writes to a JSON file.
"""

import json
import sys
import numpy as np
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def extract_cond_to_json(cond_path, output_path=None, recompute_semantics=False, write_back_cond=False):
    """
    Load cond.npy and extract character metadata to JSON.
    
    Args:
        cond_path: Path to cond.npy file
        output_path: Output JSON path (defaults to cond_metadata.json in same dir)
        recompute_semantics: Rebuild semantic metadata from joints_names/parents/offsets before export
        write_back_cond: Persist recomputed semantic metadata into cond.npy
    """
    cond_path = Path(cond_path)
    
    if not cond_path.exists():
        raise FileNotFoundError(f"cond.npy not found at {cond_path}")
    
    # Load the condition dictionary
    cond_dict = np.load(cond_path, allow_pickle=True).item()

    if recompute_semantics:
        from data_loaders.truebones.truebones_utils.motion_process import _build_semantic_metadata

        for char_name, char_cond in cond_dict.items():
            semantic_metadata = _build_semantic_metadata(
                char_cond.get('object_type', char_name),
                char_cond['joints_names'],
                char_cond['parents'],
                char_cond['offsets'],
            )
            char_cond.update(semantic_metadata)

        if write_back_cond:
            np.save(cond_path, cond_dict, allow_pickle=True)
    
    # Extract metadata for each character
    metadata = {}
    for char_name, char_cond in cond_dict.items():
        metadata[char_name] = {
            'end_effector_names': char_cond.get('end_effector_names', []),
            'contact_joint_names': char_cond.get('contact_joint_names', []),
            'is_symmetric': bool(char_cond.get('is_symmetric', False)),
            'species_label': char_cond.get('species_label', ''),
            'species_group': char_cond.get('species_group', ''),
        }
    
    # Determine output path
    if output_path is None:
        output_path = cond_path.parent / 'cond_metadata.json'
    else:
        output_path = Path(output_path)
    
    # Write to JSON
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Extracted metadata for {len(metadata)} characters")
    print(f"✓ Saved to: {output_path}")
    
    return metadata


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract character metadata from cond.npy to JSON'
    )
    parser.add_argument(
        '--cond-path',
        default='dataset/truebones/zoo/truebones_processed/cond.npy',
        help='Path to cond.npy file (default: dataset/truebones/zoo/truebones_processed/cond.npy)'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output JSON path (default: same directory as cond.npy with name cond_metadata.json)'
    )
    parser.add_argument(
        '--recompute-semantics',
        action='store_true',
        help='Recompute end effector/contact/symmetry metadata from cond.npy before exporting JSON.'
    )
    parser.add_argument(
        '--write-back-cond',
        action='store_true',
        help='When recomputing semantics, also overwrite cond.npy with the refreshed semantic fields.'
    )
    
    args = parser.parse_args()
    
    extract_cond_to_json(
        args.cond_path,
        args.output,
        recompute_semantics=args.recompute_semantics,
        write_back_cond=args.write_back_cond,
    )
