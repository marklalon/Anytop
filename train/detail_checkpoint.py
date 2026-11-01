"""Deterministic group scheduling and resume contracts for refiner training."""
import random
import math

import torch


def learning_rate_factor(step, total, groups, mode='constant', warmup_per_group=2,
                         min_ratio=.01):
    """One-based update schedule; derived from saved step, without hidden state."""
    if mode == 'constant':
        return 1.
    if mode != 'cosine':
        raise ValueError('Unknown learning rate schedule')
    warmup = min(warmup_per_group * groups, max(0, total - 1))
    if warmup and step <= warmup:
        return step / warmup
    progress = min(1., max(0., (step - warmup) / max(1, total - warmup)))
    return min_ratio + (1 - min_ratio) * .5 * (1 + math.cos(math.pi * progress))


class GroupSchedule:
    def __init__(self, groups, budget, seed, state=None, legacy_step=0):
        self.groups, self.budget = groups, budget
        self.rng = random.Random(seed)
        self.epoch, self.order, self.cursor = 0, [], 0
        if state is not None:
            self.epoch, self.order, self.cursor = state['epoch'], list(state['order']), state['cursor']
            if (sorted(self.order) != list(range(groups)) or not 0 <= self.cursor <= groups
                    or not 0 <= self.epoch < budget):
                raise ValueError('Invalid resume sampler state')
            self.rng.setstate(state['rng_state'])
        else:
            # Older checkpoints used the same shuffle/randrange sequence but did
            # not save it. Replaying just Python scheduling recovers it exactly.
            for _ in range(legacy_step):
                next(self)

    def __iter__(self):
        return self

    def __next__(self):
        if not self.order or self.cursor == self.groups:
            if self.order:
                self.epoch += 1
            if self.epoch >= self.budget:
                raise StopIteration
            self.order = list(range(self.groups))
            self.rng.shuffle(self.order)
            self.cursor = 0
        group = self.order[self.cursor]
        seed = self.rng.randrange(2**31)
        self.cursor += 1
        return self.epoch, group, seed

    def state_dict(self):
        return {'epoch': self.epoch, 'order': list(self.order), 'cursor': self.cursor,
                'rng_state': self.rng.getstate()}

    def counts(self):
        visited = set(self.order[:self.cursor])
        return [self.epoch + int(i in visited) for i in range(self.groups)]


def validate_resume(checkpoint, manifest, manifest_hash, config, weights, conditions, budget):
    from sample.detail_refinement import normalize_backbone_contract
    expected = {'schema': manifest['schema'], 'contract': normalize_backbone_contract(manifest['contract']),
                'cache_manifest_sha256': manifest_hash, 'architecture': config,
                'loss_weights': weights, 'conditions': conditions}
    for key, value in expected.items():
        actual = checkpoint.get(key)
        if key == 'contract' and isinstance(actual, dict):
            actual = normalize_backbone_contract(actual)
        if actual != value:
            raise ValueError(f'Cannot resume: {key} mismatch')
    counts = checkpoint.get('group_updates', [])
    if (len(counts) != len(conditions) or any(type(v) is not int or v < 0 for v in counts)
            or sum(counts) != checkpoint.get('step') or max(counts) > budget):
        raise ValueError('Cannot resume: invalid group counts or updates-per-group below completed budget')
    if 'optimizer' not in checkpoint or len(checkpoint.get('normalization', [])) != len(conditions):
        raise ValueError('Cannot resume: missing optimizer/normalization state')


def rng_state():
    return {'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    if state:
        torch.set_rng_state(state['torch'].cpu())
        if state.get('cuda') is not None and torch.cuda.is_available():
            if len(state['cuda']) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
