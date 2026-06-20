from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


from model.anytop import AnyTop  # noqa: E402


T5_DIM = 512


class _CaptureDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_kwargs = None

    def forward(self, **kwargs):
        self.last_kwargs = kwargs
        return kwargs['tgt']


def _make_model(species_cond=False, species_cfg_drop_prob=0.15, species_joint_cond=False):
    return AnyTop(
        max_joints=4,
        feature_len=13,
        latent_dim=8,
        ff_size=32,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        cross_limb=True,
        t5_out_dim=T5_DIM,
        species_cond=species_cond,
        species_cfg_drop_prob=species_cfg_drop_prob,
        species_joint_cond=species_joint_cond,
    )


def _make_y(species_emb=None, **extra):
    y = {
        'joints_padding_mask': torch.ones(2, 1, 1, 5, 5, dtype=torch.float32),
        'mask': torch.ones(2, 1, 1, 4, 4, dtype=torch.float32),
        'rest_pose': torch.randn(2, 4, 13, dtype=torch.float32),
        'n_joints': torch.tensor([4, 3], dtype=torch.int64),
        'joints_names_embs': torch.zeros(2, 4, T5_DIM, dtype=torch.float32),
        'lengths': torch.tensor([3, 3], dtype=torch.int64),
    }
    if species_emb is not None:
        y['species_emb'] = species_emb
    y.update(extra)
    return y


class SpeciesHybridTest(unittest.TestCase):
    def test_disabled_by_default(self):
        model = AnyTop(max_joints=4, feature_len=13, latent_dim=8, ff_size=32,
                       num_layers=1, num_heads=2, dropout=0.0, cross_limb=True)
        self.assertFalse(model.species_cond)
        self.assertIsNone(model.species_film)
        self.assertFalse(model.species_joint_cond)
        self.assertFalse(model.input_process.species_joint_cond)

    def test_invalid_drop_prob_rejected(self):
        with self.assertRaises(ValueError):
            _make_model(species_cond=True, species_cfg_drop_prob=1.5)

    def test_film_is_identity_at_init(self):
        # Zero-init head -> gamma=1, beta=0 -> output == input timestep embedding.
        model = _make_model(species_cond=True)
        model.eval()
        ts = torch.randn(2, 8)
        y = _make_y(species_emb=torch.randn(2, T5_DIM))
        out = model._apply_species_film(ts.clone(), y, 2, torch.device('cpu'), torch.float32)
        self.assertTrue(torch.allclose(out, ts))

    def test_film_cfg_drop_is_identity(self):
        # With a perturbed (non-identity) head, active rows are modulated and
        # dropped rows bypass to identity (gamma=1, beta=0).
        model = _make_model(species_cond=True)
        model.eval()
        torch.nn.init.normal_(model.species_film[-1].weight, std=0.5)
        torch.nn.init.normal_(model.species_film[-1].bias, std=0.5)
        ts = torch.randn(2, 8)
        y = _make_y(species_emb=torch.randn(2, T5_DIM),
                    species_active=torch.tensor([True, False]))
        out = model._apply_species_film(ts.clone(), y, 2, torch.device('cpu'), torch.float32)
        self.assertFalse(torch.allclose(out[0], ts[0]))   # kept -> modulated
        self.assertTrue(torch.allclose(out[1], ts[1]))    # dropped -> identity

    def test_full_drop_in_training_is_identity(self):
        model = _make_model(species_cond=True, species_cfg_drop_prob=1.0)
        model.train()
        torch.nn.init.normal_(model.species_film[-1].weight, std=0.5)
        ts = torch.randn(2, 8)
        y = _make_y(species_emb=torch.randn(2, T5_DIM))
        out = model._apply_species_film(ts.clone(), y, 2, torch.device('cpu'), torch.float32)
        self.assertTrue(torch.allclose(out, ts))

    def test_film_missing_species_emb_raises(self):
        model = _make_model(species_cond=True)
        with self.assertRaises(ValueError):
            model._apply_species_film(torch.randn(2, 8), _make_y(), 2,
                                      torch.device('cpu'), torch.float32)

    def test_joint_cond_additive_fusion_keeps_text_embedding_dim(self):
        plain = _make_model()
        fused = _make_model(species_joint_cond=True)
        self.assertEqual(plain.input_process.text_embedding.in_features, T5_DIM)
        self.assertEqual(fused.input_process.text_embedding.in_features, T5_DIM)

    def test_joint_cond_requires_species_emb(self):
        model = _make_model(species_joint_cond=True)
        model.eval()
        model.seqTransDecoder = _CaptureDecoder()
        x = torch.randn(2, 4, 13, 3, dtype=torch.float32)
        ts = torch.tensor([1, 2], dtype=torch.int64)
        with self.assertRaises(ValueError):
            model(x, ts, y=_make_y())  # no species_emb -> InputProcess raises

    def test_full_forward_both_paths_on(self):
        model = _make_model(species_cond=True, species_cfg_drop_prob=0.0,
                            species_joint_cond=True)
        model.eval()
        capture = _CaptureDecoder()
        model.seqTransDecoder = capture
        x = torch.randn(2, 4, 13, 3, dtype=torch.float32)
        ts = torch.tensor([1, 2], dtype=torch.int64)
        y = _make_y(species_emb=torch.randn(2, T5_DIM))
        out = model(x, ts, y=y)
        self.assertIsNotNone(capture.last_kwargs)
        self.assertEqual(out.shape, x.shape)


if __name__ == '__main__':
    unittest.main()
