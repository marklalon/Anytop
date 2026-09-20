"""``--action_label_adaln``: the action label's multiplicative pathway.

The additive action token can only translate the residual stream, and every
sublayer output is LayerNormed straight after its residual add.
``--action_label_adaln`` adds a per-channel scale and shift, driven by the SAME
token, to the temporal and feed-forward BRANCH INPUTS of every decoder layer
(docs/conditional_modulation_upgrade.md section 3.2).

What these tests pin:

* zero-init -- a fresh modulated model is bit-identical to one without the flag,
  so turning it on changes nothing until training moves the head;
* once the head moves, the label reaches the output THROUGH the modulation and
  not only through the additive token;
* classifier-free guidance stays honest: a dropped row modulates by the null
  embedding rather than bypassing the head;
* it is refused without a label to drive it.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loaders.truebones.truebones_utils.joint_struct_features import (  # noqa: E402
    JOINT_STRUCT_DIM,
)
from model.anytop import AnyTop  # noqa: E402
from model.motion_transformer import (  # noqa: E402
    ACTION_ADALN_PARAMS_PER_LAYER,
    ACTION_ADALN_BRANCHES,
)
from tests.action_label_test_utils import (  # noqa: E402
    TEST_LATENT_DIM,
    TEST_T5_DIM,
    action_cond_fields,
    make_test_bundle,
)


NUM_LAYERS = 2


def _model(action_label_adaln, bundle, action_label_cond=True, seed=7):
    torch.manual_seed(seed)
    model = AnyTop(
        max_joints=4,
        feature_len=12,
        latent_dim=TEST_LATENT_DIM,
        ff_size=32,
        num_layers=NUM_LAYERS,
        num_heads=2,
        dropout=0.0,
        cross_limb=True,
        t5_out_dim=TEST_T5_DIM,
        action_label_cond=action_label_cond,
        action_label_cfg_drop_prob=0.0,
        action_label_adaln=action_label_adaln,
        action_conditioning=bundle if action_label_cond else None,
    )
    model.eval()
    return model


def _y(labels):
    # Deterministic: every field but the label has to be identical across calls,
    # or "the output changed" would not isolate the label.
    y = {
        'joints_padding_mask': torch.ones(2, 1, 1, 5, 5, dtype=torch.float32),
        'rest_pose': torch.randn(
            2, 4, 12, dtype=torch.float32,
            generator=torch.Generator().manual_seed(5),
        ),
        'n_joints': torch.tensor([4, 3], dtype=torch.int64),
        'joints_names_embs': torch.zeros(2, 4, TEST_T5_DIM, dtype=torch.float32),
        'joint_struct': torch.zeros(2, 4, JOINT_STRUCT_DIM, dtype=torch.float32),
        'graph_dist': torch.zeros(2, 4, 4, dtype=torch.int64),
        'joints_relations': torch.zeros(2, 4, 4, dtype=torch.int64),
        'lengths': torch.tensor([3, 3], dtype=torch.int64),
        'canonical_feature_mean': torch.zeros(12, dtype=torch.float32),
        'canonical_feature_std': torch.ones(12, dtype=torch.float32),
    }
    y.update(action_cond_fields(labels, ['stationary'] * len(labels)))
    return y


def _x_t():
    torch.manual_seed(11)
    return torch.randn(2, 4, 12, 3), torch.tensor([1, 2], dtype=torch.int64)


def _perturb_head(model, scale=0.05):
    """Move the zero-initialised output layer off zero, as training would."""
    head = model.seqTransDecoder.action_adaln
    torch.manual_seed(3)
    with torch.no_grad():
        head[-1].weight.normal_(0.0, scale)
        head[-1].bias.normal_(0.0, scale)


class ActionAdaLNStartsAsIdentity(unittest.TestCase):
    def test_a_fresh_modulated_model_matches_the_unmodulated_one(self):
        bundle = make_test_bundle()
        plain = _model(False, bundle)
        modulated = _model(True, bundle)
        # Give the two models the same weights everywhere but the head. The
        # head consumes RNG at construction, so a shared seed alone would
        # not do it, and the property under test is about the zero-init head,
        # not about initialisation order.
        missing, unexpected = modulated.load_state_dict(plain.state_dict(), strict=False)
        self.assertEqual(unexpected, [])
        self.assertTrue(
            all(name.startswith('seqTransDecoder.action_adaln.') for name in missing),
            missing,
        )

        x, t = _x_t()
        y = _y(['attack, bite', 'idle'])
        self.assertTrue(torch.allclose(modulated(x, t, y=y), plain(x, t, y=y), atol=1e-6))

    def test_the_output_layer_is_zero_initialised(self):
        model = _model(True, make_test_bundle())
        head = model.seqTransDecoder.action_adaln
        self.assertTrue(torch.equal(head[-1].weight, torch.zeros_like(head[-1].weight)))
        self.assertTrue(torch.equal(head[-1].bias, torch.zeros_like(head[-1].bias)))
        # One head for every layer: four parameters (scale + shift for each of
        # the two modulated branches) per layer, emitted in one matmul.
        self.assertEqual(ACTION_ADALN_PARAMS_PER_LAYER, 2 * ACTION_ADALN_BRANCHES)
        self.assertEqual(ACTION_ADALN_BRANCHES, 2)
        self.assertEqual(
            head[-1].out_features,
            NUM_LAYERS * ACTION_ADALN_PARAMS_PER_LAYER * TEST_LATENT_DIM,
        )

    def test_no_modulation_head_without_the_flag(self):
        self.assertIsNone(_model(False, make_test_bundle()).seqTransDecoder.action_adaln)


class ActionAdaLNCarriesTheLabel(unittest.TestCase):
    def test_a_trained_head_makes_the_label_reach_the_output_through_it(self):
        bundle = make_test_bundle()
        modulated = _model(True, bundle)
        _perturb_head(modulated)
        x, t = _x_t()

        attack = modulated(x, t, y=_y(['attack, bite', 'idle']))
        idle = modulated(x, t, y=_y(['idle', 'idle']))
        # Row 0's label differs, row 1's does not.
        self.assertFalse(torch.allclose(attack[0], idle[0], atol=1e-6))
        self.assertTrue(torch.allclose(attack[1], idle[1], atol=1e-6))

        # And the difference is not just the additive token: with the additive
        # path alone (same weights, head still zeroed) it comes out different.
        plain = _model(True, bundle)
        plain_gap = (plain(x, t, y=_y(['attack, bite', 'idle']))[0]
                     - plain(x, t, y=_y(['idle', 'idle']))[0]).abs().mean().detach()
        modulated_gap = (attack[0] - idle[0]).abs().mean().detach()
        self.assertNotAlmostEqual(float(modulated_gap), float(plain_gap), places=6)

    def test_a_dropped_row_modulates_by_the_null_embedding(self):
        """CFG's unconditional pass must go THROUGH the head, not around it.

        A bypass would make the guidance base a model the head never
        conditioned -- the failure the retired global_energy null had.
        """
        bundle = make_test_bundle()
        modulated = _model(True, bundle)
        _perturb_head(modulated)
        with torch.no_grad():
            # A null embedding that is not the zero vector, so "went through the
            # head" and "skipped it" cannot coincide.
            modulated.action_label_null_emb.normal_(0.0, 1.0)
        x, t = _x_t()
        y = _y(['attack, bite', 'idle'])

        inactive = torch.tensor([False, False])
        dropped = modulated(x, t, y=dict(y, action_label_active=inactive))
        other_label = modulated(
            x, t,
            y=dict(_y(['idle', 'idle']), action_label_active=inactive),
        )
        # With both rows unconditional the label no longer reaches the output.
        self.assertTrue(torch.allclose(dropped, other_label, atol=1e-6))

        # And that unconditional output is NOT what a zeroed head would give:
        # the null token drove a real modulation.
        bypass = _model(True, bundle)
        with torch.no_grad():
            bypass.action_label_null_emb.copy_(modulated.action_label_null_emb)
        self.assertFalse(torch.allclose(
            dropped, bypass(x, t, y=dict(y, action_label_active=inactive)), atol=1e-6))


class ActionAdaLNNeedsALabel(unittest.TestCase):
    def test_modulation_without_action_label_cond_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            _model(True, None, action_label_cond=False)
        self.assertIn('action_label_cond', str(caught.exception))


if __name__ == '__main__':
    unittest.main()
