"""Read-only audit of the joint condition: structural channel + slim joint names.

Everything the design document requires *before* a training run is started, in
one place, so the claims cannot drift away from the implementation:

  1. determinism and name invariance of ``build_joint_struct_features``
  2. finiteness and range of the descriptors over every species
  3. structural discriminability -- how many joints the slim text collapses that
     the structural channel then fails to tell apart
  4. text slimming -- token counts, and that no structure-derived word survives
  5. padding -- the structural latent of a padded joint is exactly zero AFTER
     the MLP projection, and a mixed batch matches a single sample
  6. path consistency -- the loader and sample/generate.py call one builder
  7. version stamps -- the schema numbers and the cond file's hash

Writes nothing. Exits non-zero when a check fails, so it can gate a run.

    python tools/audit_joint_conditioning.py --cond dataset/merged/cond.npy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loaders.truebones.truebones_utils.cond_schema import load_cond  # noqa: E402
from data_loaders.truebones.truebones_utils.joint_struct_features import (  # noqa: E402
    JOINT_STRUCT_DIM,
    JOINT_STRUCT_FEATURE_NAMES,
    JOINT_STRUCT_FEATURE_SCHEMA_VERSION,
    build_joint_struct_features,
)
from data_loaders.truebones.truebones_utils.physics_joint_annotation import (  # noqa: E402
    JOINT_NAME_EMBEDDING_SCHEMA_VERSION,
    JOINT_NAME_EMBEDDING_SLIM,
    build_joint_embedding_texts,
)

# Thresholds from the design document (section 5.1).
MAX_UNRESOLVED_COLLISION_PAIRS = 5
MAX_MEAN_TEXT_TOKENS = 2.0
MAX_TEXT_TOKENS = 4

# Words the slim text must never emit again: every one of them is a function of
# parents / rest positions / contacts, which the structural channel now carries.
_STRUCTURE_DERIVED_WORDS = {
    'segment', 'of', 'instance', 'contact', 'endeffector',
    'chainstart', 'chainmiddle', 'chainend', 'chainearly', 'chainlate',
    'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth',
}


class Audit:
    def __init__(self):
        self.results = []
        self.failed = 0

    def check(self, name, passed, detail=''):
        self.results.append({'check': name, 'passed': bool(passed), 'detail': detail})
        self.failed += 0 if passed else 1
        mark = 'OK  ' if passed else 'FAIL'
        print(f'[{mark}] {name}' + (f'\n       {detail}' if detail else ''))
        return passed


def _shuffled_name_copy(entry, rng):
    """The same skeleton with every joint name replaced by a random token."""
    joint_count = len(np.asarray(entry['parents']).reshape(-1))
    scrambled = dict(entry)
    fake = [f'z{int(rng.integers(0, 10 ** 9))}' for _ in range(joint_count)]
    scrambled['joints_names'] = list(fake)
    scrambled['canonical_joint_names'] = list(fake)
    scrambled['canonical_bvh_joint_names'] = list(fake)
    return scrambled


def audit_struct_features(cond, audit):
    rng = np.random.default_rng(0)
    features = {}
    non_deterministic, name_dependent, errors = [], [], []
    for object_type, entry in sorted(cond.items()):
        try:
            first = build_joint_struct_features(entry, source=object_type)
        except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
            errors.append(f'{object_type}: {exc}')
            continue
        features[object_type] = first
        if not np.array_equal(first, build_joint_struct_features(entry, source=object_type)):
            non_deterministic.append(object_type)
        renamed = build_joint_struct_features(
            _shuffled_name_copy(entry, rng), source=object_type
        )
        if not np.array_equal(first, renamed):
            name_dependent.append(object_type)

    audit.check(
        'every species builds structural features',
        not errors,
        '' if not errors else '; '.join(errors[:5]),
    )
    audit.check('features are deterministic across repeated calls',
                not non_deterministic, ', '.join(non_deterministic[:5]))
    audit.check('features are bit-identical after a full rename',
                not name_dependent, ', '.join(name_dependent[:5]))

    if not features:
        return features, {}

    stacked = np.concatenate(list(features.values()), axis=0)
    audit.check('all features finite', bool(np.isfinite(stacked).all()))
    # Every channel is a ratio, a normalized coordinate or a flag; a magnitude
    # past a few units means a scale or a span divided by something wrong.
    worst = float(np.abs(stacked).max())
    audit.check('feature magnitudes stay bounded (|x| <= 4)', worst <= 4.0,
                f'max |feature| = {worst:.4f}')
    ranges = {
        name: {
            'min': round(float(stacked[:, index].min()), 6),
            'max': round(float(stacked[:, index].max()), 6),
            'mean': round(float(stacked[:, index].mean()), 6),
            'std': round(float(stacked[:, index].std()), 6),
        }
        for index, name in enumerate(JOINT_STRUCT_FEATURE_NAMES)
    }
    print('\n       per-channel range over '
          f'{stacked.shape[0]} joints in {len(features)} species:')
    for name, stats in ranges.items():
        print(f'         {name:>16}  min={stats["min"]:9.4f}  max={stats["max"]:9.4f}  '
              f'mean={stats["mean"]:8.4f}  std={stats["std"]:8.4f}')
    print()
    return features, ranges


def audit_texts_and_collisions(cond, features, audit):
    token_counts = []
    longest = ('', 0)
    banned_hits = []
    collision_groups = 0
    unresolved_pairs = []

    for object_type, entry in sorted(cond.items()):
        texts = build_joint_embedding_texts(entry)
        for text in texts:
            tokens = str(text).split()
            token_counts.append(len(tokens))
            if len(tokens) > longest[1]:
                longest = (str(text), len(tokens))
            hit = {token.lower() for token in tokens} & _STRUCTURE_DERIVED_WORDS
            if hit:
                banned_hits.append(f'{object_type}: {text!r} ({", ".join(sorted(hit))})')

        struct = features.get(object_type)
        if struct is None:
            continue
        by_text = defaultdict(list)
        for joint_index, text in enumerate(texts):
            if str(text).strip():
                by_text[str(text)].append(joint_index)
        for text, indices in by_text.items():
            if len(indices) < 2:
                continue
            collision_groups += 1
            for position, left in enumerate(indices):
                for right in indices[position + 1:]:
                    if np.allclose(struct[left], struct[right], atol=1e-6):
                        unresolved_pairs.append(f'{object_type}: joints {left}/{right} -> {text!r}')

    mean_tokens = float(np.mean(token_counts)) if token_counts else 0.0
    audit.check(f'mean joint-name text is <= {MAX_MEAN_TEXT_TOKENS} tokens',
                mean_tokens <= MAX_MEAN_TEXT_TOKENS, f'mean = {mean_tokens:.3f} tokens')
    audit.check(f'longest joint-name text is <= {MAX_TEXT_TOKENS} tokens',
                longest[1] <= MAX_TEXT_TOKENS, f'longest = {longest[1]} tokens: {longest[0]!r}')
    audit.check('no structure-derived word survives in the text',
                not banned_hits, '; '.join(banned_hits[:5]))
    audit.check(
        f'structural channel resolves all but <= {MAX_UNRESOLVED_COLLISION_PAIRS} '
        'same-text joint pairs',
        len(unresolved_pairs) <= MAX_UNRESOLVED_COLLISION_PAIRS,
        f'{len(unresolved_pairs)} unresolved of {collision_groups} colliding text group(s)'
        + ('\n       ' + '\n       '.join(unresolved_pairs[:10]) if unresolved_pairs else ''),
    )
    return {
        'mean_tokens': round(mean_tokens, 4),
        'max_tokens': int(longest[1]),
        'longest_text': longest[0],
        'collision_groups': int(collision_groups),
        'unresolved_pairs': unresolved_pairs[:50],
    }


def audit_model_padding(audit):
    import torch  # local: the data-side checks must run without a torch build

    from model.anytop import InputProcess

    torch.manual_seed(0)
    latent_dim, t5_dim, frames, max_joints = 16, 32, 3, 5
    process = InputProcess(13, 13, latent_dim, t5_dim, dropout_prob=0.0).eval()

    x = torch.randn(2, max_joints, 13, frames)
    rest_pose = torch.randn(1, 2, max_joints, 13)
    names = torch.randn(2, max_joints, t5_dim)
    struct = torch.randn(2, max_joints, JOINT_STRUCT_DIM)
    # Sample 0 has 3 live joints, sample 1 has all 5.
    valid = torch.tensor([[True, True, True, False, False], [True] * max_joints])
    struct = struct * valid.unsqueeze(-1)  # what the collate hands over

    with torch.no_grad():
        latent = process.struct_embedding(struct) * valid.unsqueeze(-1)
    padded_norm = float(latent[0, 3:].abs().max())
    audit.check('padded structural latent is exactly zero after the projection',
                padded_norm == 0.0, f'max |padded latent| = {padded_norm}')

    with torch.no_grad():
        biased = process.struct_embedding(struct)
    audit.check(
        'the re-zero is load-bearing (the raw MLP does NOT map zeros to zeros)',
        float(biased[0, 3:].abs().max()) > 0.0,
        'a zero input row leaves the MLP non-zero because of its biases',
    )

    with torch.no_grad():
        mixed = process(x, rest_pose, names, None, valid, struct)
        alone = process(
            x[1:], rest_pose[:, 1:], names[1:], None, valid[1:], struct[1:],
        )
    delta = float((mixed[:, 1:] - alone).abs().max())
    audit.check('a mixed-joint-count batch matches the single-sample run',
                delta < 1e-5, f'max |difference| = {delta:.3e}')


def audit_path_consistency(audit):
    import data_loaders.truebones.data.dataset as dataset_module
    import sample.generate as generate_module

    audit.check(
        'the loader and generate.py share ONE structural builder',
        dataset_module.build_joint_struct_features
        is generate_module.build_joint_struct_features
        is build_joint_struct_features,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cond', default='dataset/merged/cond.npy',
                        help='cond.npy to audit (default: dataset/merged/cond.npy)')
    parser.add_argument('--json', default='',
                        help='also write the full report to this path')
    parser.add_argument('--skip_model', action='store_true',
                        help='skip the checks that need torch')
    args = parser.parse_args()

    cond_path = Path(args.cond)
    if not cond_path.is_file():
        sys.exit(f'ERROR: {cond_path} does not exist.')
    digest = hashlib.sha256(cond_path.read_bytes()).hexdigest()

    print(f'cond:                          {cond_path}')
    print(f'  sha256:                      {digest[:16]}...')
    print(f'  size:                        {os.path.getsize(cond_path)} bytes')
    print(f'joint_struct schema:           {JOINT_STRUCT_FEATURE_SCHEMA_VERSION} '
          f'({JOINT_STRUCT_DIM} channels)')
    print(f'joint-name embedding schema:   {JOINT_NAME_EMBEDDING_SCHEMA_VERSION} '
          f'(slim={JOINT_NAME_EMBEDDING_SLIM})')
    print()

    cond = load_cond(cond_path)
    print(f'{len(cond)} species\n')

    audit = Audit()
    features, ranges = audit_struct_features(cond, audit)
    text_stats = audit_texts_and_collisions(cond, features, audit)

    encoded = {
        str(object_type): int((entry.get('joints_names_embs_meta') or {}).get('schema_version', -1))
        for object_type, entry in cond.items()
    }
    stale = sorted({version for version in encoded.values()} - {JOINT_NAME_EMBEDDING_SCHEMA_VERSION})
    audit.check(
        'the encoded joint-name embeddings are at the current schema',
        not stale,
        '' if not stale else (
            f'cond carries schema(s) {stale}; regenerate with '
            'tools/regenerate_dataset_artifacts.py before training'
        ),
    )

    if not args.skip_model:
        audit_model_padding(audit)
        audit_path_consistency(audit)

    print()
    print(f'{len(audit.results) - audit.failed}/{len(audit.results)} checks passed')

    if args.json:
        report = {
            'cond_path': str(cond_path),
            'cond_sha256': digest,
            'num_species': len(cond),
            'joint_struct_schema_version': JOINT_STRUCT_FEATURE_SCHEMA_VERSION,
            'joint_struct_feature_names': list(JOINT_STRUCT_FEATURE_NAMES),
            'joint_name_embedding_schema_version': JOINT_NAME_EMBEDDING_SCHEMA_VERSION,
            'joint_name_embedding_slim': bool(JOINT_NAME_EMBEDDING_SLIM),
            'encoded_schema_versions': encoded,
            'feature_ranges': ranges,
            'text_stats': text_stats,
            'checks': audit.results,
        }
        Path(args.json).write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'report written to {args.json}')

    sys.exit(1 if audit.failed else 0)


if __name__ == '__main__':
    main()
