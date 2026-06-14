from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model.anytop import AnyTop  # noqa: E402
from model.morphology_expert import (  # noqa: E402
    MORPHOLOGY_GROUPS,
    MorphologyExpertBank,
    _normalize_layers_spec,
    resolve_morphology_ids,
    validate_morphology_registry,
)
from utils.model_util import load_model  # noqa: E402


_COMMON = dict(
    max_joints=4, feature_len=13, latent_dim=16, ff_size=32,
    num_layers=4, num_heads=2, dropout=0.0, cross_limb=True,
)


def _make_y(object_type, B=2, J=4, F=3, **extra):
    y = {
        'joints_padding_mask': torch.ones(B, 1, 1, J + 1, J + 1),
        'mask': torch.ones(B, 1, 1, F + 1, F + 1),
        'tpos_first_frame': torch.randn(B, J, 13),
        'n_joints': torch.tensor([4, 3]),
        'joints_names_embs': torch.zeros(B, J, 512),
        'lengths': torch.tensor([F, F]),
        'graph_dist': torch.zeros(B, J, J, dtype=torch.long),
        'joints_relations': torch.zeros(B, J, J, dtype=torch.long),
        'object_type': object_type,
    }
    y.update(extra)
    return y


class ResolveMorphologyIdsTest(unittest.TestCase):
    def test_registry_and_routing(self):
        groups, table = resolve_morphology_ids()
        self.assertEqual(groups, MORPHOLOGY_GROUPS)
        self.assertEqual(table['Cat'], 0)          # Quadruped
        self.assertEqual(table['Anaconda'], 4)      # Serpentine
        self.assertEqual(table['Jaws'], 5)          # Aquatic
        self.assertEqual(len(table), 74)

    def test_layers_spec(self):
        self.assertEqual(_normalize_layers_spec('last4', 8), 4)
        self.assertEqual(_normalize_layers_spec('all8', 8), 8)
        self.assertEqual(_normalize_layers_spec('all', 8), 8)
        self.assertEqual(_normalize_layers_spec('last99', 8), 8)  # clamped
        with self.assertRaises(ValueError):
            _normalize_layers_spec('bogus', 8)


class MorphologyExpertBankTest(unittest.TestCase):
    def test_zero_init_residual(self):
        bank = MorphologyExpertBank(dim=32, num_groups=6, bottleneck=8, dropout=0.0)
        x = torch.randn(5, 4, 7, 32)
        res = bank(x, torch.tensor([0, 2, 5, 1]))
        self.assertEqual(tuple(res.shape), (5, 4, 7, 32))
        self.assertEqual(float(res.abs().max()), 0.0)

    def test_dense_routing_selects_per_sample(self):
        bank = MorphologyExpertBank(dim=8, num_groups=6, bottleneck=4, dropout=0.0)
        with torch.no_grad():
            for g, a in enumerate(bank.adapters):
                a.net[-1].bias.fill_(float(g + 1))
        x = torch.zeros(2, 4, 3, 8)
        res = bank(x, torch.tensor([0, 2, 5, 1]))
        means = res.mean(dim=(0, 2, 3))
        self.assertTrue(torch.allclose(means, torch.tensor([1.0, 3.0, 6.0, 2.0])))


class AnyTopMorphologyExpertTest(unittest.TestCase):
    def test_disabled_by_default(self):
        m = AnyTop(**_COMMON)
        self.assertFalse(m.morphology_expert)
        self.assertIsNone(m.seqTransDecoder.morphology_expert_banks)

    def test_step0_equivalent_to_baseline(self):
        torch.manual_seed(0)
        m = AnyTop(morphology_expert=True, morphology_expert_layers='last2',
                   morphology_expert_bottleneck=8, morphology_expert_dropout=0.0, **_COMMON)
        m.eval()
        y = _make_y(['Cat', 'Anaconda'])
        x = torch.randn(2, 4, 13, 3)
        t = torch.tensor([5, 7])
        with torch.no_grad():
            out_expert = m(x, t, y=y)
            saved = m.seqTransDecoder.morphology_expert_banks
            m.seqTransDecoder.morphology_expert_banks = None
            out_base = m(x, t, y=y)
            m.seqTransDecoder.morphology_expert_banks = saved
        self.assertEqual(float((out_expert - out_base).abs().max()), 0.0)

    def test_routing_only_affects_routed_sample(self):
        torch.manual_seed(0)
        m = AnyTop(morphology_expert=True, morphology_expert_layers='last2',
                   morphology_expert_bottleneck=8, morphology_expert_dropout=0.0, **_COMMON)
        m.eval()
        y = _make_y(['Cat', 'Anaconda'])  # morphology 0, morphology 4
        x = torch.randn(2, 4, 13, 3)
        t = torch.tensor([5, 7])
        with torch.no_grad():
            out0 = m(x, t, y=y)
            for bank in m.seqTransDecoder.morphology_expert_banks:
                bank.adapters[4].net[-1].bias.fill_(1.0)  # Serpentine -> sample 1
            out1 = m(x, t, y=y)
        per_sample = (out1 - out0).abs().reshape(2, -1).max(dim=1).values
        self.assertAlmostEqual(float(per_sample[0]), 0.0, places=6)
        self.assertGreater(float(per_sample[1]), 0.0)

    def test_precomputed_group_ids_matches_object_type(self):
        torch.manual_seed(0)
        m = AnyTop(morphology_expert=True, morphology_expert_layers='last2',
                   morphology_expert_bottleneck=8, morphology_expert_dropout=0.0, **_COMMON)
        m.eval()
        x = torch.randn(2, 4, 13, 3)
        t = torch.tensor([5, 7])
        y_names = _make_y(['Cat', 'Anaconda'])
        y_ids = dict(y_names)
        y_ids.pop('object_type')
        y_ids['group_ids'] = torch.tensor([0, 4])
        with torch.no_grad():
            for bank in m.seqTransDecoder.morphology_expert_banks:
                bank.adapters[4].net[-1].bias.fill_(1.0)
            out_names = m(x, t, y=y_names)
            out_ids = m(x, t, y=y_ids)
        self.assertEqual(float((out_names - out_ids).abs().max()), 0.0)

    def test_resume_tolerates_missing_bank_keys(self):
        base = AnyTop(**_COMMON)
        expert = AnyTop(morphology_expert=True, morphology_expert_layers='last2',
                        morphology_expert_bottleneck=8, **_COMMON)
        # Baseline checkpoint has no bank keys; load must not raise.
        load_model(expert, base.state_dict())


class FrozenRoutingTableTest(unittest.TestCase):
    """Fix 1: a checkpoint's routing table is frozen and used verbatim."""

    def test_saved_mapping_used_not_tags_file(self):
        # A mapping that deliberately disagrees with the live species_tags.jsonl
        # (Cat is Quadruped==0 there) must be honored verbatim.
        frozen = {'Cat': 1, 'Dragon': 3}
        m = AnyTop(morphology_expert=True, morphology_expert_layers='last2',
                   morphology_expert_bottleneck=8,
                   morphology_groups=list(MORPHOLOGY_GROUPS),
                   morphology_object_type_to_group_id=frozen, **_COMMON)
        self.assertEqual(m.object_type_to_group_id, {'Cat': 1, 'Dragon': 3})

    def test_registry_reorder_rejected(self):
        reordered = ['Biped', 'Quadruped', 'Multiped', 'Winged', 'Serpentine', 'Aquatic']
        with self.assertRaises(ValueError):
            AnyTop(morphology_expert=True, morphology_expert_layers='last2',
                   morphology_expert_bottleneck=8,
                   morphology_groups=reordered,
                   morphology_object_type_to_group_id={'Cat': 0}, **_COMMON)

    def test_append_only_prefix_accepted(self):
        # A shorter (prefix) registry is the append-only case: accepted.
        self.assertEqual(validate_morphology_registry(['Quadruped', 'Biped']),
                         ('Quadruped', 'Biped'))
        m = AnyTop(morphology_expert=True, morphology_expert_layers='last2',
                   morphology_expert_bottleneck=8,
                   morphology_groups=['Quadruped', 'Biped'],
                   morphology_object_type_to_group_id={'Cat': 0, 'Human': 1}, **_COMMON)
        self.assertEqual(len(m.morphology_groups), 2)
        self.assertEqual(len(m.seqTransDecoder.morphology_expert_banks[0].adapters), 2)

    def test_out_of_range_group_id_rejected(self):
        with self.assertRaises(ValueError):
            AnyTop(morphology_expert=True, morphology_expert_layers='last2',
                   morphology_expert_bottleneck=8,
                   morphology_groups=['Quadruped', 'Biped'],
                   morphology_object_type_to_group_id={'Cat': 5}, **_COMMON)


class LoadModelMorphologyToleranceTest(unittest.TestCase):
    """Fix 2: only warm-start from a non-expert checkpoint is tolerant."""

    def _expert(self):
        return AnyTop(morphology_expert=True, morphology_expert_layers='last2',
                      morphology_expert_bottleneck=8, **_COMMON)

    def test_warm_start_from_baseline_ok(self):
        load_model(self._expert(), AnyTop(**_COMMON).state_dict())

    def test_full_expert_checkpoint_ok(self):
        src = self._expert()
        load_model(self._expert(), src.state_dict())

    def test_partial_expert_checkpoint_rejected(self):
        src = self._expert()
        sd = src.state_dict()
        dropped = next(k for k in sd if 'morphology_expert_banks' in k)
        del sd[dropped]
        with self.assertRaises(AssertionError):
            load_model(self._expert(), sd)


if __name__ == '__main__':
    unittest.main()
