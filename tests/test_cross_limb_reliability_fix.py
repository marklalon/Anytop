"""Model-, training- and checkpoint-level tests for the cross-limb reliability
fix (docs/cross_limb_reliability_cost_effective_fix.md, section 8).

The block-level half (frame key bias, cross-K) lives in
tests/test_cross_limb_temporal.py. This file covers what sits around the
block:

* the trunk-wide ``unreliable_embedding`` on the input tokens (AnyTop);
* the same-level / hard re-noise mixture (GaussianDiffusion);
* the AdamW no-decay groups.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_loaders.truebones.truebones_utils.joint_struct_features import (  # noqa: E402
    JOINT_STRUCT_DIM,
)
from diffusion.gaussian_diffusion import (  # noqa: E402
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
)
from model.anytop import AnyTop  # noqa: E402
from train.training_loop import (  # noqa: E402
    build_optimizer_param_groups,
    is_no_weight_decay_param,
)


B, J, F_, NFEATS = 2, 4, 3, 12


def _small_model(seed: int = 0, **overrides) -> AnyTop:
    torch.manual_seed(seed)
    kwargs = dict(
        max_joints=J,
        feature_len=NFEATS,
        latent_dim=8,
        ff_size=32,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
        cross_limb=True,
        cross_limb_latents=3,
        cross_limb_dim=8,
    )
    kwargs.update(overrides)
    model = AnyTop(**kwargs)
    model.eval()
    return model


def _inputs(seed: int = 1):
    torch.manual_seed(seed)
    x = torch.randn(B, J, NFEATS, F_, dtype=torch.float32)
    y = {
        "joints_padding_mask": torch.ones(B, 1, 1, J + 1, J + 1, dtype=torch.float32),
        "rest_pose": torch.randn(B, J, NFEATS, dtype=torch.float32),
        "n_joints": torch.full((B,), J, dtype=torch.int64),
        "joints_names_embs": torch.zeros(B, J, 512, dtype=torch.float32),
        "joint_struct": torch.zeros(B, J, JOINT_STRUCT_DIM, dtype=torch.float32),
        "graph_dist": torch.zeros(B, J, J, dtype=torch.int64),
        "joints_relations": torch.zeros(B, J, J, dtype=torch.int64),
        "canonical_feature_mean": torch.zeros(NFEATS, dtype=torch.float32),
        "canonical_feature_std": torch.ones(NFEATS, dtype=torch.float32),
    }
    t = torch.tensor([1, 2], dtype=torch.int64)
    return x, t, y


def _whole_frame_raw_mask(frames) -> torch.Tensor:
    """Loss-side (B, F, J) layout, every joint of the listed frames flagged."""
    mask = torch.zeros(B, F_, J, dtype=torch.float32)
    mask[:, frames, :] = 1.0
    return mask


class UnreliableEmbeddingTests(unittest.TestCase):
    def test_exists_zero_init_and_is_a_noop_on_a_whole_frame_mask(self):
        model = _small_model()
        self.assertEqual(model.unreliable_embedding.shape, (model.latent_dim,))
        self.assertTrue(bool((model.unreliable_embedding == 0).all()))
        x, t, y = _inputs()

        out_none = model(x, t, y=dict(y))
        out_mask = model(x, t, y=dict(y, cross_limb_unreliable_mask=_whole_frame_raw_mask([1])))

        self.assertTrue(torch.allclose(out_none, out_mask, atol=1e-6))

    def test_nonzero_embedding_makes_a_whole_frame_mask_visible(self):
        """The core defect: with every cross-limb gate at zero, a whole-frame
        mask used to be invisible to the network. The input embedding alone
        must now make it change the output."""
        model = _small_model()
        for blk in model.seqTransDecoder.cross_limb_blocks:
            self.assertEqual(blk.reliability_bias.item(), 0.0)
            self.assertEqual(blk.temporal_reliability_bias.item(), 0.0)
        with torch.no_grad():
            model.unreliable_embedding.fill_(0.5)
        x, t, y = _inputs()

        out_none = model(x, t, y=dict(y))
        out_mask = model(x, t, y=dict(y, cross_limb_unreliable_mask=_whole_frame_raw_mask([1])))

        self.assertFalse(torch.allclose(out_none, out_mask, atol=1e-6))

    def test_tpose_row_is_never_flagged_and_prepared_mask_is_not_prepended_twice(self):
        model = _small_model()
        raw = _whole_frame_raw_mask([0, 2])
        prepared = model._prepare_unreliable_mask(raw, B, F_, J, torch.device("cpu"), torch.float32)

        self.assertEqual(prepared.shape, (F_ + 1, B, J))
        self.assertTrue(bool((prepared[0] == 0).all()))
        self.assertTrue(torch.equal(prepared[1:], raw.transpose(0, 1)))
        again = model._prepare_unreliable_mask(prepared, B, F_, J, torch.device("cpu"), torch.float32)
        self.assertTrue(torch.equal(again, prepared))

    def test_masks_do_not_leak_across_batch_samples(self):
        model = _small_model()
        with torch.no_grad():
            model.unreliable_embedding.fill_(0.5)
            for blk in model.seqTransDecoder.cross_limb_blocks:
                blk.temporal_reliability_bias.fill_(-2.0)
                blk.reliability_bias.fill_(-1.0)
                blk.cross_k_scale.fill_(0.5)
        x, t, y = _inputs()
        mask_a = torch.zeros(B, F_, J)
        mask_a[0, 1, :] = 1.0           # sample 0 only
        mask_b = mask_a.clone()
        mask_b[1, 2, 1] = 1.0           # sample 1 changes

        out_a = model(x, t, y=dict(y, cross_limb_unreliable_mask=mask_a))
        out_b = model(x, t, y=dict(y, cross_limb_unreliable_mask=mask_b))

        self.assertTrue(torch.allclose(out_a[0], out_b[0], atol=1e-6))
        self.assertFalse(torch.allclose(out_a[1], out_b[1], atol=1e-6))

    def test_embedding_applies_without_cross_limb_blocks(self):
        """The map is a trunk-level signal, not a cross-limb-only one."""
        model = _small_model(cross_limb=False)
        self.assertIsNone(model.seqTransDecoder.cross_limb_blocks)
        with torch.no_grad():
            model.unreliable_embedding.fill_(0.5)
        x, t, y = _inputs()

        out_none = model(x, t, y=dict(y))
        out_mask = model(x, t, y=dict(y, cross_limb_unreliable_mask=_whole_frame_raw_mask([1])))

        self.assertFalse(torch.allclose(out_none, out_mask, atol=1e-6))


class RenoiseMixtureTests(unittest.TestCase):
    def _diffusion(self, same_level_prob: float, steps: int = 100) -> GaussianDiffusion:
        return GaussianDiffusion(
            betas=np.linspace(1e-4, 0.02, steps, dtype=np.float64),
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_LARGE,
            loss_type=LossType.MSE,
            renoise_same_level_prob=same_level_prob,
        )

    def test_default_is_same_level_and_range_is_validated(self):
        self.assertEqual(self._diffusion(0.5).renoise_same_level_prob, 0.5)
        default = GaussianDiffusion(
            betas=np.linspace(1e-4, 0.02, 10, dtype=np.float64),
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_LARGE,
            loss_type=LossType.MSE,
        )
        self.assertEqual(default.renoise_same_level_prob, 1.0)
        for bad in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                self._diffusion(bad)

    def test_never_below_t_and_never_at_or_past_T(self):
        for prob in (0.0, 0.5, 1.0):
            diffusion = self._diffusion(prob)
            torch.manual_seed(0)
            t = torch.randint(0, diffusion.num_timesteps, (4096,), dtype=torch.int64)
            t_random = diffusion._sample_renoise_timesteps(t, torch.device("cpu"))
            self.assertEqual(t_random.shape, t.shape)
            self.assertEqual(t_random.dtype, t.dtype)
            self.assertTrue(bool((t_random >= t).all()), prob)
            self.assertTrue(bool((t_random < diffusion.num_timesteps).all()), prob)

    def test_same_level_branch_returns_t_exactly(self):
        diffusion = self._diffusion(1.0)
        torch.manual_seed(0)
        t = torch.randint(0, diffusion.num_timesteps, (2048,), dtype=torch.int64)
        t_random = diffusion._sample_renoise_timesteps(t, torch.device("cpu"))
        self.assertTrue(torch.equal(t_random, t))
        self.assertIsNot(t_random, t)

    def test_hard_branch_alone_still_degrades_well_above_t(self):
        diffusion = self._diffusion(0.0)
        torch.manual_seed(0)
        t = torch.zeros(20000, dtype=torch.int64)
        gap = (diffusion._sample_renoise_timesteps(t, torch.device("cpu")) - t).float()
        # Uniform on [0, 100): mean ~49.5.
        self.assertGreater(gap.mean().item(), 45.0)
        self.assertLess(gap.mean().item(), 54.0)
        self.assertGreater(gap.max().item(), 90.0)

    def test_mixture_fraction_matches_the_configured_probability(self):
        diffusion = self._diffusion(0.5)
        torch.manual_seed(0)
        t = torch.zeros(20000, dtype=torch.int64)
        t_random = diffusion._sample_renoise_timesteps(t, torch.device("cpu"))
        same = (t_random == t).float().mean().item()
        # 0.5 from the same-level branch + 0.5 * 1/100 from a hard draw of 0.
        self.assertAlmostEqual(same, 0.505, delta=0.02)
        # The hard half still reaches far above t.
        self.assertGreater((t_random - t).float().mean().item(), 20.0)

    def test_extreme_probabilities_take_no_random_gate(self):
        """0 and 1 short-circuit: no second rand draw, so the RNG stream a
        pure-hard run consumed before this change is unchanged."""
        torch.manual_seed(7)
        t = torch.randint(0, 100, (256,), dtype=torch.int64)
        before = torch.get_rng_state()
        self._diffusion(0.0)._sample_renoise_timesteps(t, torch.device("cpu"))
        after_hard = torch.get_rng_state()
        torch.set_rng_state(before)
        _ = torch.rand(t.shape)          # exactly one rand draw of that shape
        self.assertTrue(torch.equal(torch.get_rng_state(), after_hard))


class OptimizerParamGroupTests(unittest.TestCase):
    def test_no_decay_rule_names_the_gates_and_cross_k_norm_only(self):
        model = _small_model()
        names = [n for n, _ in model.named_parameters()]
        no_decay = sorted(n for n in names if is_no_weight_decay_param(n))
        expected = sorted(
            ["unreliable_embedding"]
            + [f"seqTransDecoder.layers.{i}.temporal_phase_scale" for i in range(2)]
            + [
                f"seqTransDecoder.cross_limb_blocks.{i}.{leaf}"
                for i in range(2)
                for leaf in (
                    "reliability_bias",
                    "time_emb_scale",
                    "temporal_reliability_bias",
                    "cross_k_scale",
                    "cross_k_norm.weight",
                    "cross_k_norm.bias",
                )
            ]
        )
        self.assertEqual(no_decay, expected)
        # Other 1-D params (LayerNorm affines, biases) keep decaying: a rank
        # rule would have swept them in.
        self.assertFalse(is_no_weight_decay_param("seqTransDecoder.cross_limb_blocks.0.norm_cl.weight"))
        self.assertFalse(is_no_weight_decay_param("seqTransDecoder.layers.0.norm1.bias"))
        self.assertFalse(is_no_weight_decay_param("input_process.joint_embedding.bias"))

    def test_groups_partition_all_trainable_params_and_carry_the_decay(self):
        model = _small_model()
        groups = build_optimizer_param_groups(model.named_parameters(), 0.01)

        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["weight_decay"], 0.01)
        self.assertEqual(groups[1]["weight_decay"], 0.0)
        placed = [id(p) for g in groups for p in g["params"]]
        self.assertEqual(sorted(placed), sorted(id(p) for p in model.parameters()))
        self.assertEqual(len(placed), len(set(placed)))
        no_decay_ids = {id(p) for p in groups[1]["params"]}
        self.assertIn(id(model.unreliable_embedding), no_decay_ids)
        for blk in model.seqTransDecoder.cross_limb_blocks:
            self.assertIn(id(blk.cross_k_scale), no_decay_ids)
            self.assertIn(id(blk.cross_k_norm.weight), no_decay_ids)
            self.assertNotIn(id(blk.norm_cl.weight), no_decay_ids)

    def test_one_adamw_step_over_the_groups_with_a_mask(self):
        """Fresh AdamW over both groups on a masked batch: every new
        parameter receives a finite gradient and update."""
        model = _small_model()
        model.train()
        opt = torch.optim.AdamW(
            build_optimizer_param_groups(model.named_parameters(), 0.01), lr=1e-3
        )
        x, t, y = _inputs()
        out = model(x, t, y=dict(y, cross_limb_unreliable_mask=_whole_frame_raw_mask([1])))
        out.pow(2).mean().backward()
        opt.step()

        self.assertTrue(torch.isfinite(model.unreliable_embedding).all())
        # The gates get gradient through the (mask-dependent) paths they open.
        self.assertIsNotNone(model.unreliable_embedding.grad)
        self.assertGreater(model.unreliable_embedding.grad.abs().sum().item(), 0.0)
        for blk in model.seqTransDecoder.cross_limb_blocks:
            self.assertIsNotNone(blk.cross_k_scale.grad)
            self.assertIsNotNone(blk.temporal_reliability_bias.grad)


if __name__ == "__main__":
    unittest.main()
