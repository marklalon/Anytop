"""The output coordinate frame as an unconditional model input.

The per-object_subset canonical (mean, std) that the features are standardized
by define WHICH affine space the model writes into. They reach the model in
``y`` already; this projection is what lets it read them. There is deliberately
no flag: every canonical_motion_v3 cond carries the two vectors and the loader
refuses to start without them, so "off" could only mean "blind to the output
space". See docs/canonical_frame_and_label_transfer.md (lever 2).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model.anytop import AnyTop  # noqa: E402
from utils.model_util import get_gmdm_args  # noqa: E402


T5_DIM = 512
LATENT = 8


def _make_model():
    return AnyTop(
        max_joints=4,
        feature_len=13,
        latent_dim=LATENT,
        ff_size=32,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        cross_limb=True,
        t5_out_dim=T5_DIM,
    )


def _stats(batch=2):
    mean = torch.arange(13, dtype=torch.float32).mul(0.01).repeat(batch, 1)
    std = torch.arange(13, dtype=torch.float32).mul(0.1).add(0.5).repeat(batch, 1)
    return {'canonical_feature_mean': mean, 'canonical_feature_std': std}


class CanonicalFrameCondTest(unittest.TestCase):
    def test_projection_is_unconditional(self):
        # No flag, no opt-in: a default model already carries the projection.
        model = _make_model()
        self.assertIsNotNone(model.canonical_frame_projection)
        self.assertIsNotNone(
            model._build_canonical_frame_token(_stats(), 2, torch.device('cpu'), torch.float32)
        )

    def test_zero_init_contributes_nothing(self):
        # Additive token, zero-init final linear -> exactly 0 at construction, so
        # the condition starts at exact identity.
        model = _make_model()
        model.eval()
        token = model._build_canonical_frame_token(
            _stats(), 2, torch.device('cpu'), torch.float32)
        self.assertEqual(tuple(token.shape), (2, LATENT))
        self.assertTrue(torch.equal(token, torch.zeros_like(token)))

    def test_token_separates_two_coordinate_frames(self):
        model = _make_model()
        model.eval()
        torch.nn.init.normal_(model.canonical_frame_projection[-1].weight, std=0.5)
        y = _stats(batch=2)
        # Row 1 gets a different subset's gain -- the two rows must not collapse.
        y['canonical_feature_std'] = y['canonical_feature_std'].clone()
        y['canonical_feature_std'][1] *= 3.0
        token = model._build_canonical_frame_token(y, 2, torch.device('cpu'), torch.float32)
        self.assertFalse(torch.allclose(token[0], token[1]))

    def test_never_cfg_dropped_in_training(self):
        # The frame is the definition of the output space, not a semantic
        # condition: there is no keep mask and training must not zero it.
        model = _make_model()
        torch.nn.init.normal_(model.canonical_frame_projection[-1].weight, std=0.5)
        torch.nn.init.normal_(model.canonical_frame_projection[-1].bias, std=0.5)
        y = _stats()
        model.train()
        train_token = model._build_canonical_frame_token(
            y, 2, torch.device('cpu'), torch.float32)
        model.eval()
        eval_token = model._build_canonical_frame_token(
            y, 2, torch.device('cpu'), torch.float32)
        self.assertTrue(torch.allclose(train_token, eval_token))
        self.assertFalse(torch.allclose(train_token, torch.zeros_like(train_token)))

    def test_bare_vector_broadcasts_over_batch(self):
        model = _make_model()
        torch.nn.init.normal_(model.canonical_frame_projection[-1].weight, std=0.5)
        y = {'canonical_feature_mean': torch.zeros(13), 'canonical_feature_std': torch.ones(13)}
        token = model._build_canonical_frame_token(y, 3, torch.device('cpu'), torch.float32)
        self.assertEqual(tuple(token.shape), (3, LATENT))
        self.assertTrue(torch.allclose(token[0], token[2]))

    def test_missing_stats_raises(self):
        model = _make_model()
        with self.assertRaises(ValueError):
            model._build_canonical_frame_token({}, 2, torch.device('cpu'), torch.float32)

    def test_batch_size_mismatch_raises(self):
        model = _make_model()
        with self.assertRaises(ValueError):
            model._build_canonical_frame_token(
                _stats(batch=3), 2, torch.device('cpu'), torch.float32)

    def test_model_args_carry_no_frame_flag(self):
        # The condition is not configurable, so nothing about it may leak into
        # args.json -- a stale flag there would read as a togglable feature.
        class _Args:
            latent_dim = 8
            layers = 1
            value_emb = False
            cross_limb_latents = 8
            t5_out_dim = T5_DIM

        self.assertNotIn('canonical_frame_cond', get_gmdm_args(_Args()))


if __name__ == '__main__':
    unittest.main()
