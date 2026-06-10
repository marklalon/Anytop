import os
import sys
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion.fp16_util import MixedPrecisionTrainer, inspect_optimizer_state, sanitize_optimizer_state
from model.motion_transformer import GraphMultiHeadAttention, SelectiveMultiheadAttention


class MixedPrecisionTrainerTests(unittest.TestCase):
    def test_sanitize_optimizer_state_clears_nonfinite_entries(self):
        model = torch.nn.Linear(4, 2)
        opt = AdamW(model.parameters(), lr=1e-3)

        data = torch.randn(3, 4)
        target = torch.randn(3, 2)
        loss = torch.nn.functional.mse_loss(model(data), target)
        loss.backward()
        opt.step()

        first_state = next(iter(opt.state.values()))
        first_state["exp_avg_sq"].view(-1)[0] = float("inf")
        first_state["exp_avg"].view(-1)[0] = float("nan")

        stats = sanitize_optimizer_state(opt)

        self.assertTrue(stats["found"])
        self.assertGreater(stats["inf"], 0)
        self.assertGreater(stats["nan"], 0)
        for state in opt.state.values():
            for value in state.values():
                if torch.is_tensor(value) and torch.is_floating_point(value):
                    self.assertTrue(torch.isfinite(value).all())

    def test_inspect_optimizer_state_reports_nonfinite_entries_without_mutation(self):
        model = torch.nn.Linear(4, 2)
        opt = AdamW(model.parameters(), lr=1e-3)

        data = torch.randn(3, 4)
        target = torch.randn(3, 2)
        loss = torch.nn.functional.mse_loss(model(data), target)
        loss.backward()
        opt.step()

        first_state = next(iter(opt.state.values()))
        first_state["exp_avg_sq"].view(-1)[0] = float("inf")

        stats = inspect_optimizer_state(opt)

        self.assertTrue(stats["found"])
        self.assertGreater(stats["inf"], 0)
        self.assertTrue(torch.isinf(first_state["exp_avg_sq"]).any())

    def test_optimize_amp_skips_nonfinite_gradients(self):
        model = torch.nn.Linear(4, 2)
        trainer = MixedPrecisionTrainer(
            model=model,
            use_fp16=False,
            amp_dtype="bf16",
            amp_enabled=True,
            device_type="cpu",
            log_norms=False,
        )
        opt = AdamW(trainer.master_params, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)

        trainer.zero_grad()
        model.weight.grad = torch.full_like(model.weight, float("inf"))
        model.bias.grad = torch.zeros_like(model.bias)
        before_weight = model.weight.detach().clone()

        took_step = trainer.optimize(opt, scheduler)

        self.assertFalse(took_step)
        self.assertTrue(torch.equal(before_weight, model.weight.detach()))
        self.assertEqual(len(opt.state), 0)
        self.assertEqual(scheduler.last_epoch, 0)

    def test_optimize_normal_raises_on_large_finite_gradients(self):
        model = torch.nn.Linear(4, 2)
        trainer = MixedPrecisionTrainer(
            model=model,
            use_fp16=False,
            amp_enabled=False,
            log_norms=False,
        )
        opt = AdamW(trainer.master_params, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)

        trainer.zero_grad()
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 4e11)
        before_params = [parameter.detach().clone() for parameter in model.parameters()]

        with self.assertRaisesRegex(RuntimeError, "Detected abnormal finite grad_norm"):
            trainer.optimize(opt, scheduler)

        self.assertEqual(scheduler.last_epoch, 0)
        self.assertEqual(len(opt.state), 0)
        for before_param, parameter in zip(before_params, model.parameters()):
            self.assertTrue(torch.equal(before_param, parameter.detach()))

    def test_optimize_amp_raises_on_large_finite_gradients(self):
        model = torch.nn.Linear(4, 2)
        trainer = MixedPrecisionTrainer(
            model=model,
            use_fp16=False,
            amp_dtype="bf16",
            amp_enabled=True,
            device_type="cpu",
            log_norms=False,
        )
        opt = AdamW(trainer.master_params, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)

        trainer.zero_grad()
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 4e11)
        before_params = [parameter.detach().clone() for parameter in model.parameters()]

        with self.assertRaisesRegex(RuntimeError, "Detected abnormal finite grad_norm"):
            trainer.optimize(opt, scheduler)

        self.assertEqual(scheduler.last_epoch, 0)
        self.assertEqual(len(opt.state), 0)
        for before_param, parameter in zip(before_params, model.parameters()):
            self.assertTrue(torch.equal(before_param, parameter.detach()))

    def test_optimize_amp_finite_step_uses_clip_without_nonfinite_scan(self):
        model = torch.nn.Linear(4, 2)
        trainer = MixedPrecisionTrainer(
            model=model,
            use_fp16=False,
            amp_dtype="bf16",
            amp_enabled=True,
            device_type="cpu",
            log_norms=False,
        )
        opt = AdamW(trainer.master_params, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)

        trainer.zero_grad()
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        before_params = [parameter.detach().clone() for parameter in model.parameters()]
        original_clip_grad_norm = torch.nn.utils.clip_grad_norm_

        with patch(
            "diffusion.fp16_util.count_nonfinite_gradients",
            side_effect=AssertionError("finite step should not trigger non-finite scan"),
        ), patch(
            "diffusion.fp16_util.th.nn.utils.clip_grad_norm_",
            wraps=original_clip_grad_norm,
        ) as clip_grad_norm:
            took_step = trainer.optimize(opt, scheduler)

        self.assertTrue(took_step)
        clip_grad_norm.assert_called_once()
        self.assertEqual(scheduler.last_epoch, 1)
        self.assertEqual(len(opt.state), 2)
        self.assertTrue(
            any(
                not torch.equal(before_param, parameter.detach())
                for before_param, parameter in zip(before_params, model.parameters())
            )
        )

    def test_optimize_fp16_raises_on_large_finite_gradients_when_log_norms_disabled(self):
        model = _MinimalFP16Model()
        trainer = MixedPrecisionTrainer(
            model=model,
            use_fp16=True,
            amp_enabled=False,
            log_norms=False,
        )
        opt = AdamW(trainer.master_params, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)

        trainer.zero_grad()
        scaled_grad_value = 4e11 * (2 ** trainer.lg_loss_scale)
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, scaled_grad_value)
        before_params = [parameter.detach().clone() for parameter in model.parameters()]

        with self.assertRaisesRegex(RuntimeError, "Detected abnormal finite grad_norm"):
            trainer.optimize(opt, scheduler)

        self.assertEqual(scheduler.last_epoch, 0)
        self.assertEqual(len(opt.state), 0)
        for before_param, parameter in zip(before_params, model.parameters()):
            self.assertTrue(torch.equal(before_param, parameter.detach()))

    def test_optimize_fp16_skips_nonfinite_gradients_when_log_norms_disabled(self):
        model = _MinimalFP16Model()
        trainer = MixedPrecisionTrainer(
            model=model,
            use_fp16=True,
            amp_enabled=False,
            log_norms=False,
        )
        opt = AdamW(trainer.master_params, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)

        trainer.zero_grad()
        model.linear.weight.grad = torch.full_like(model.linear.weight, float("inf"))
        model.linear.bias.grad = torch.zeros_like(model.linear.bias)
        before_params = [parameter.detach().clone() for parameter in model.parameters()]

        took_step = trainer.optimize(opt, scheduler)

        self.assertFalse(took_step)
        self.assertEqual(scheduler.last_epoch, 0)
        self.assertEqual(len(opt.state), 0)
        self.assertLess(trainer.lg_loss_scale, 20.0)
        for before_param, parameter in zip(before_params, model.parameters()):
            self.assertTrue(torch.equal(before_param, parameter.detach()))


class _MinimalFP16Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, input):
        return self.linear(input)

    def convert_to_fp16(self):
        return None



class AttentionModuleTests(unittest.TestCase):
    def test_selective_multihead_attention_need_weights_false_uses_sdpa_and_matches_manual_path(self):
        attn = SelectiveMultiheadAttention(embed_dim=8, num_heads=2, dropout=0.0)

        query = torch.randn(5, 3, 8, dtype=torch.float32)
        attn_mask = torch.triu(torch.ones(3 * 2, 5, 5, dtype=torch.bool), diagonal=1)
        key_padding_mask = torch.tensor(
            [
                [False, False, False, True, True],
                [False, False, True, False, True],
                [False, False, False, False, False],
            ],
            dtype=torch.bool,
        )

        with patch(
            "model.motion_transformer.F.scaled_dot_product_attention",
            wraps=F.scaled_dot_product_attention,
        ) as sdpa:
            sdpa_output, sdpa_weights = attn(
                query,
                query,
                query,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )

        manual_output, manual_weights = attn(
            query,
            query,
            query,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )

        sdpa.assert_called_once()
        self.assertIsNone(sdpa_weights)
        self.assertIsNotNone(manual_weights)
        self.assertTrue(torch.allclose(sdpa_output, manual_output, atol=1e-5, rtol=1e-5))

    def test_graph_multihead_attention_value_embeddings_keep_manual_softmax_path(self):
        attn = GraphMultiHeadAttention(d_model=8, dropout=0.0, nheads=2)
        attn.eval()

        batch_size = 2
        sequence_length = 4
        q = torch.randn(batch_size, sequence_length, 8, dtype=torch.float32)
        distance = torch.randint(0, 3, (batch_size, sequence_length, sequence_length), dtype=torch.long)
        edge_attr = torch.randint(0, 4, (batch_size, sequence_length, sequence_length), dtype=torch.long)
        query_hop_emb = torch.randn(3, 8, dtype=torch.float32)
        query_edge_emb = torch.randn(4, 8, dtype=torch.float32)
        key_hop_emb = torch.randn(3, 8, dtype=torch.float32)
        key_edge_emb = torch.randn(4, 8, dtype=torch.float32)
        value_hop_emb = torch.zeros(3, 8, dtype=torch.float32)
        value_edge_emb = torch.zeros(4, 8, dtype=torch.float32)

        with patch(
            "model.motion_transformer.F.scaled_dot_product_attention",
            wraps=F.scaled_dot_product_attention,
        ) as sdpa:
            output = attn(
                q,
                q,
                q,
                query_hop_emb,
                query_edge_emb,
                key_hop_emb,
                key_edge_emb,
                value_hop_emb,
                value_edge_emb,
                distance,
                edge_attr,
                None,
            )

        # With value embeddings present the module must take the manual
        # scatter/gather softmax path rather than SDPA.
        sdpa.assert_not_called()
        self.assertEqual(output.dtype, torch.float32)

    def test_graph_multihead_attention_sdpa_matches_manual_path_without_value_embeddings(self):
        torch.manual_seed(0)
        attn = GraphMultiHeadAttention(d_model=8, dropout=0.0, nheads=2)
        attn.eval()

        batch_size = 2
        sequence_length = 4
        q = torch.randn(batch_size, sequence_length, 8, dtype=torch.float32)
        distance = torch.randint(0, 3, (batch_size, sequence_length, sequence_length), dtype=torch.long)
        edge_attr = torch.randint(0, 4, (batch_size, sequence_length, sequence_length), dtype=torch.long)
        spatial_mask = torch.zeros(batch_size, 2, sequence_length, sequence_length, dtype=torch.float32)
        spatial_mask[1, :, :, -1] = -1e4
        key_padding_mask = torch.tensor(
            [
                [False, False, False, True],
                [False, False, True, False],
            ],
            dtype=torch.bool,
        )
        query_hop_emb = torch.randn(3, 8, dtype=torch.float32)
        query_edge_emb = torch.randn(4, 8, dtype=torch.float32)
        key_hop_emb = torch.randn(3, 8, dtype=torch.float32)
        key_edge_emb = torch.randn(4, 8, dtype=torch.float32)
        zero_value_hop_emb = torch.zeros(3, 8, dtype=torch.float32)
        zero_value_edge_emb = torch.zeros(4, 8, dtype=torch.float32)

        with patch(
            "model.motion_transformer.F.scaled_dot_product_attention",
            wraps=F.scaled_dot_product_attention,
        ) as sdpa:
            sdpa_output = attn(
                q,
                q,
                q,
                query_hop_emb,
                query_edge_emb,
                key_hop_emb,
                key_edge_emb,
                None,
                None,
                distance,
                edge_attr,
                spatial_mask,
                key_padding_mask=key_padding_mask,
            )

        manual_output = attn(
            q,
            q,
            q,
            query_hop_emb,
            query_edge_emb,
            key_hop_emb,
            key_edge_emb,
            zero_value_hop_emb,
            zero_value_edge_emb,
            distance,
            edge_attr,
            spatial_mask,
            key_padding_mask=key_padding_mask,
        )

        sdpa.assert_called_once()
        self.assertTrue(torch.allclose(sdpa_output, manual_output, atol=1e-5, rtol=1e-5))

    def test_graph_multihead_attention_broadcast_relations_and_masks_match_materialized_inputs(self):
        torch.manual_seed(0)
        attn = GraphMultiHeadAttention(d_model=8, dropout=0.0, nheads=2)
        attn.eval()

        frames = 3
        batch_size = 2
        sequence_length = 4
        q = torch.randn(frames * batch_size, sequence_length, 8, dtype=torch.float32)
        distance = torch.randint(0, 3, (batch_size, sequence_length, sequence_length), dtype=torch.long)
        edge_attr = torch.randint(0, 4, (batch_size, sequence_length, sequence_length), dtype=torch.long)
        spatial_mask = torch.zeros(batch_size, 2, sequence_length, sequence_length, dtype=torch.float32)
        spatial_mask[1, :, :, -1] = -1e4
        query_hop_emb = torch.randn(3, 8, dtype=torch.float32)
        query_edge_emb = torch.randn(4, 8, dtype=torch.float32)
        key_hop_emb = torch.randn(3, 8, dtype=torch.float32)
        key_edge_emb = torch.randn(4, 8, dtype=torch.float32)

        materialized_output = attn(
            q,
            q,
            q,
            query_hop_emb,
            query_edge_emb,
            key_hop_emb,
            key_edge_emb,
            None,
            None,
            distance.unsqueeze(0).repeat(frames, 1, 1, 1).reshape(-1, sequence_length, sequence_length),
            edge_attr.unsqueeze(0).repeat(frames, 1, 1, 1).reshape(-1, sequence_length, sequence_length),
            spatial_mask.unsqueeze(0).repeat(frames, 1, 1, 1, 1).reshape(-1, 2, sequence_length, sequence_length),
        )
        broadcast_output = attn(
            q,
            q,
            q,
            query_hop_emb,
            query_edge_emb,
            key_hop_emb,
            key_edge_emb,
            None,
            None,
            distance.unsqueeze(1).expand(-1, 2, -1, -1),
            edge_attr.unsqueeze(1).expand(-1, 2, -1, -1),
            spatial_mask,
        )

        self.assertTrue(torch.allclose(materialized_output, broadcast_output, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()