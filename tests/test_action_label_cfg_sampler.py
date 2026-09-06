"""Inference-side classifier-free guidance over the action label.

Training's half of the contract (the hard-drop that creates the unconditional
mode) is covered by test_action_label_cond.py; this file covers the sampling
half: the two-pass wrapper and the flag that turns it on.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


from model.anytop import AnyTop  # noqa: E402
from model.cfg_sampler import ClassifierFreeActionModel  # noqa: E402
from tests.action_label_test_utils import (  # noqa: E402
    TEST_LATENT_DIM,
    TEST_T5_DIM,
    action_cond_fields,
    make_test_bundle,
    sample_action_slots,
)


T5_DIM = TEST_T5_DIM


class _StubDenoiser(torch.nn.Module):
    """Returns a constant per branch, and records every y it was called with."""

    def __init__(self):
        super().__init__()
        self.feature_len = 13
        self.calls = []

    def forward(self, x, timesteps, y=None, **kwargs):
        self.calls.append(y)
        active = None if y is None else y.get('action_label_active')
        conditional = active is None or bool(torch.as_tensor(active).reshape(-1)[0])
        return torch.full_like(x, 4.0 if conditional else 1.0)


class _TimestepEchoDecoder(torch.nn.Module):
    """Stands in for the real decoder, but keeps a dependence on the timestep
    token -- which is where the action condition lives, so a decoder that ignored
    it would make the conditional and unconditional forwards identical."""

    def forward(self, **kwargs):
        return kwargs['tgt'] + kwargs['timesteps_embs'].sum()


def _make_model():
    return AnyTop(
        max_joints=4,
        feature_len=13,
        latent_dim=TEST_LATENT_DIM,
        ff_size=32,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        cross_limb=True,
        t5_out_dim=T5_DIM,
        action_label_cond=True,
        action_label_cfg_drop_prob=0.2,
        action_conditioning=make_test_bundle(),
    )


def _make_y():
    return {
        'joints_padding_mask': torch.ones(2, 1, 1, 5, 5, dtype=torch.float32),
        'rest_pose': torch.randn(2, 4, 13, dtype=torch.float32),
        'n_joints': torch.tensor([4, 3], dtype=torch.int64),
        'joints_names_embs': torch.zeros(2, 4, T5_DIM, dtype=torch.float32),
        'lengths': torch.tensor([3, 3], dtype=torch.int64),
        'canonical_feature_mean': torch.zeros(13, dtype=torch.float32),
        'canonical_feature_std': torch.ones(13, dtype=torch.float32),
        **action_cond_fields(['run, forward'] * 2, ['locomotion'] * 2),
    }


class ClassifierFreeActionModelTest(unittest.TestCase):
    def test_scale_one_is_a_single_conditional_forward(self):
        stub = _StubDenoiser()
        wrapped = ClassifierFreeActionModel(stub, 1.0)
        x = torch.zeros(2, 4, 13, 3)
        out = wrapped(x, torch.tensor([1, 1]), y={'action_label': ['run', 'run']})
        self.assertEqual(len(stub.calls), 1)
        self.assertTrue(torch.allclose(out, torch.full_like(x, 4.0)))

    def test_guidance_extrapolates_away_from_the_null_prediction(self):
        stub = _StubDenoiser()
        wrapped = ClassifierFreeActionModel(stub, 2.5)
        x = torch.zeros(2, 4, 13, 3)
        out = wrapped(x, torch.tensor([1, 1]), y={'action_label': ['run', 'run']})
        self.assertEqual(len(stub.calls), 2)
        # uncond + s * (cond - uncond) == 1 + 2.5 * (4 - 1)
        self.assertTrue(torch.allclose(out, torch.full_like(x, 8.5)))

    def test_zero_scale_is_the_pure_unconditional_prediction(self):
        stub = _StubDenoiser()
        x = torch.zeros(2, 4, 13, 3)
        out = ClassifierFreeActionModel(stub, 0.0)(
            x, torch.tensor([1, 1]), y={'action_label': ['run', 'run']})
        self.assertTrue(torch.allclose(out, torch.full_like(x, 1.0)))

    def test_uncond_pass_forces_the_null_condition_for_every_row(self):
        stub = _StubDenoiser()
        wrapped = ClassifierFreeActionModel(stub, 2.0)
        y = {'action_label': ['run', 'run']}
        wrapped(torch.zeros(2, 4, 13, 3), torch.tensor([1, 1]), y=y)
        cond_y, uncond_y = stub.calls
        # Conditional pass: untouched, so eval-mode keeps every row conditional.
        self.assertNotIn('action_label_active', cond_y)
        active = uncond_y['action_label_active']
        self.assertEqual(tuple(active.shape), (2,))
        self.assertEqual(active.dtype, torch.bool)
        self.assertFalse(bool(active.any()))
        # The label itself is NOT stripped: routing to the null embedding is the
        # mask's job, and every other channel must stay identical between passes.
        self.assertEqual(uncond_y['action_label'], y['action_label'])
        # The caller's y is left alone.
        self.assertNotIn('action_label_active', y)

    def test_invalid_scales_rejected(self):
        stub = _StubDenoiser()
        with self.assertRaises(ValueError):
            ClassifierFreeActionModel(stub, -1.0)
        with self.assertRaises(ValueError):
            ClassifierFreeActionModel(stub, float('nan'))

    def test_attributes_resolve_through_to_the_denoiser(self):
        stub = _StubDenoiser()
        wrapped = ClassifierFreeActionModel(stub, 2.0)
        self.assertEqual(wrapped.feature_len, 13)
        with self.assertRaises(AttributeError):
            _ = wrapped.no_such_attribute

    def test_matches_two_explicit_anytop_forwards(self):
        model = _make_model()
        # The null embedding is zero-init, which would make the unconditional
        # branch coincide with a zero action token; give it a real value so the
        # two branches genuinely differ.
        torch.nn.init.normal_(model.action_label_null_emb)
        model.seqTransDecoder = _TimestepEchoDecoder()
        model.eval()

        x = torch.randn(2, 4, 13, 3)
        t = torch.tensor([1, 2], dtype=torch.int64)
        y = _make_y()
        scale = 3.0

        with torch.no_grad():
            cond = model(x, t, y=dict(y, action_label_active=torch.tensor([True, True])))
            uncond = model(x, t, y=dict(y, action_label_active=torch.tensor([False, False])))
            # No explicit mask in eval == fully conditional.
            self.assertTrue(torch.allclose(model(x, t, y=y), cond, atol=1e-6))
            out = ClassifierFreeActionModel(model, scale)(x, t, y=y)

        self.assertFalse(torch.allclose(cond, uncond))
        self.assertTrue(torch.allclose(out, uncond + scale * (cond - uncond), atol=1e-5))


class ActionLabelCfgFlagTest(unittest.TestCase):
    @staticmethod
    def _args(scale, drop_prob=0.2):
        return types.SimpleNamespace(
            action_label_cfg_scale=scale, action_label_cfg_drop_prob=drop_prob,
        )

    @staticmethod
    def _condition():
        return {'action_group': 'locomotion', 'action_label': 'run',
                'action_slots': sample_action_slots('run', 'locomotion')}

    def setUp(self):
        from sample.generate import _wrap_action_label_cfg
        self.wrap = _wrap_action_label_cfg
        self.model = _StubDenoiser()

    def test_default_scale_leaves_the_model_untouched(self):
        self.assertIs(self.wrap(self.model, self._args(1.0), self._condition()), self.model)
        # ... and needs no label, since it does nothing.
        self.assertIs(self.wrap(self.model, self._args(1.0), None), self.model)

    def test_guidance_without_a_label_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.wrap(self.model, self._args(2.0), None)

    def test_guidance_on_a_checkpoint_with_no_null_mode_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.wrap(self.model, self._args(2.0, drop_prob=0.0), self._condition())

    def test_negative_scale_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.wrap(self.model, self._args(-2.0), self._condition())

    def test_wraps_when_both_halves_are_present(self):
        wrapped = self.wrap(self.model, self._args(2.5), self._condition())
        self.assertIsInstance(wrapped, ClassifierFreeActionModel)
        self.assertEqual(wrapped.guidance_scale, 2.5)
        self.assertIs(wrapped.model, self.model)


if __name__ == '__main__':
    unittest.main()
