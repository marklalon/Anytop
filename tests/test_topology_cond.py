"""``--topology_cond``: the global topology conditioner.

Learned queries pool the skeleton's rest-pose + structural tokens into one
summary that modulates every decoder layer, summed with the action AdaLN.

What these tests pin:

* zero-init -- a fresh conditioned model is bit-identical to one without the
  flag;
* the pool input carries no joint names, so renaming a rig cannot move it,
  while a structural change does;
* padding joints never reach the summary;
* it runs alone and alongside the action AdaLN, whose modulation it adds to.
"""

from __future__ import annotations

import argparse
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
    TopologyConditioner,
)
from tests.action_label_test_utils import (  # noqa: E402
    TEST_LATENT_DIM,
    TEST_T5_DIM,
    action_cond_fields,
    make_test_bundle,
)
from utils.parser_util import add_model_options  # noqa: E402


NUM_LAYERS = 2


def _model(topology_cond, action_label_adaln=False, seed=7):
    torch.manual_seed(seed)
    bundle = make_test_bundle() if action_label_adaln else None
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
        action_label_cond=action_label_adaln,
        action_label_cfg_drop_prob=0.0,
        action_label_adaln=action_label_adaln,
        action_conditioning=bundle,
        topology_cond=topology_cond,
    )
    model.eval()
    return model


def _y(names_seed=0, struct_seed=1, with_action=False):
    names = torch.randn(2, 4, TEST_T5_DIM, generator=torch.Generator().manual_seed(names_seed))
    struct = torch.randn(2, 4, JOINT_STRUCT_DIM, generator=torch.Generator().manual_seed(struct_seed))
    # Sample 1 has 3 live joints: its padded row arrives zeroed, as the collate hands it.
    names[1, 3] = 0.0
    struct[1, 3] = 0.0
    mask = torch.ones(2, 1, 1, 5, 5, dtype=torch.float32)
    mask[1, :, :, 4, :] = 0.0
    mask[1, :, :, :, 4] = 0.0
    y = {
        'joints_padding_mask': mask,
        'rest_pose': torch.randn(2, 4, 12, generator=torch.Generator().manual_seed(5)),
        'n_joints': torch.tensor([4, 3], dtype=torch.int64),
        'joints_names_embs': names,
        'joint_struct': struct,
        'graph_dist': torch.zeros(2, 4, 4, dtype=torch.int64),
        'joints_relations': torch.zeros(2, 4, 4, dtype=torch.int64),
        'lengths': torch.tensor([3, 3], dtype=torch.int64),
        'canonical_feature_mean': torch.zeros(12, dtype=torch.float32),
        'canonical_feature_std': torch.ones(12, dtype=torch.float32),
    }
    if with_action:
        y.update(action_cond_fields(['attack, bite', 'idle'], ['stationary'] * 2))
    return y


def _x_t():
    torch.manual_seed(11)
    return torch.randn(2, 4, 12, 3), torch.tensor([1, 2], dtype=torch.int64)


def _perturb_head(model, scale=0.05):
    """Move the zero-initialised output layer off zero, as training would."""
    head = model.topology_conditioner.head
    torch.manual_seed(3)
    with torch.no_grad():
        head[-1].weight.normal_(0.0, scale)
        head[-1].bias.normal_(0.0, scale)


def _topology_tokens(model, y):
    x, _ = _x_t()
    joint_valid = torch.diagonal(y['joints_padding_mask'][:, 0, 0, 1:, 1:], dim1=-2, dim2=-1) > 0.5
    with torch.no_grad():
        _, tokens = model.input_process(
            x, y['rest_pose'].unsqueeze(0), y['joints_names_embs'], None, joint_valid,
            y['joint_struct'], return_topology_tokens=True,
        )
    return tokens, joint_valid


class TopologyCondStartsAsIdentity(unittest.TestCase):
    def test_a_fresh_conditioned_model_matches_the_plain_one(self):
        for with_action in (False, True):
            with self.subTest(action_label_adaln=with_action):
                plain = _model(False, action_label_adaln=with_action)
                conditioned = _model(True, action_label_adaln=with_action)
                missing, unexpected = conditioned.load_state_dict(plain.state_dict(), strict=False)
                self.assertEqual(unexpected, [])
                self.assertTrue(missing)
                self.assertTrue(
                    all(name.startswith('topology_conditioner.') for name in missing), missing)

                x, t = _x_t()
                y = _y(with_action=with_action)
                self.assertTrue(torch.allclose(conditioned(x, t, y=y), plain(x, t, y=y), atol=1e-6))

    def test_the_head_is_zero_initialised_in_the_action_adaln_layout(self):
        head = _model(True).topology_conditioner.head
        self.assertTrue(torch.equal(head[-1].weight, torch.zeros_like(head[-1].weight)))
        self.assertTrue(torch.equal(head[-1].bias, torch.zeros_like(head[-1].bias)))
        self.assertEqual(
            head[-1].out_features,
            NUM_LAYERS * ACTION_ADALN_PARAMS_PER_LAYER * TEST_LATENT_DIM,
        )

    def test_no_conditioner_without_the_flag(self):
        self.assertIsNone(_model(False).topology_conditioner)


class TopologyCondReadsGeometryNotNames(unittest.TestCase):
    def test_pool_tokens_ignore_joint_names(self):
        model = _model(True)
        renamed, _ = _topology_tokens(model, _y(names_seed=0))
        original, _ = _topology_tokens(model, _y(names_seed=9))
        self.assertTrue(torch.equal(renamed, original))

    def test_pool_tokens_follow_the_structural_channel(self):
        model = _model(True)
        a, _ = _topology_tokens(model, _y(struct_seed=1))
        b, _ = _topology_tokens(model, _y(struct_seed=2))
        self.assertFalse(torch.allclose(a, b))

    def test_a_trained_head_moves_the_output_with_structure_only(self):
        model = _model(True)
        _perturb_head(model)
        x, t = _x_t()
        base = model(x, t, y=_y())
        restructured = model(x, t, y=_y(struct_seed=2))
        self.assertFalse(torch.allclose(base, restructured, atol=1e-6))

        # A rename still moves the output through the per-joint token, but the
        # modulation the conditioner emits stays exactly the same.
        with torch.no_grad():
            modulation = model.topology_conditioner(*_topology_tokens(model, _y()))
            renamed = model.topology_conditioner(*_topology_tokens(model, _y(names_seed=9)))
        self.assertTrue(torch.equal(modulation, renamed))
        self.assertGreater(float(modulation.abs().max()), 0.0)


class TopologyCondIgnoresPadding(unittest.TestCase):
    def test_padding_rows_do_not_reach_the_summary(self):
        torch.manual_seed(0)
        conditioner = TopologyConditioner(TEST_LATENT_DIM, NUM_LAYERS, num_heads=2).eval()
        tokens = torch.randn(1, 3, TEST_LATENT_DIM)
        valid = torch.ones(1, 3, dtype=torch.bool)
        padded = torch.cat([tokens, torch.randn(1, 5, TEST_LATENT_DIM) * 100.0], dim=1)
        padded_valid = torch.cat([valid, torch.zeros(1, 5, dtype=torch.bool)], dim=1)
        with torch.no_grad():
            self.assertTrue(torch.allclose(
                conditioner.summary(tokens, valid),
                conditioner.summary(padded, padded_valid),
                atol=1e-5,
            ))


class TopologyCondFlag(unittest.TestCase):
    def test_the_flag_reaches_the_model_kwargs(self):
        from utils.model_util import get_gmdm_args

        parser = argparse.ArgumentParser()
        add_model_options(parser)
        self.assertFalse(parser.parse_args([]).topology_cond)
        args = parser.parse_args(['--topology_cond'])
        self.assertTrue(args.topology_cond)
        args.t5_out_dim = TEST_T5_DIM
        self.assertTrue(get_gmdm_args(args)['topology_cond'])


if __name__ == '__main__':
    unittest.main()
