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
from model.selective_autocast import enable_selective_autocast


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


class _RecordingLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.recorded_output_dtype = None

    def forward(self, input):
        output = super().forward(input)
        self.recorded_output_dtype = output.dtype
        return output


class _ActivationProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input_dtype = None
        self.last_output_dtype = None

    def forward(self, input):
        self.last_input_dtype = input.dtype
        output = torch.nn.functional.gelu(input)
        self.last_output_dtype = output.dtype
        return output


class _RecordingConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__(in_channels, out_channels, kernel_size)
        self.recorded_output_dtype = None

    def forward(self, input):
        output = super().forward(input)
        self.recorded_output_dtype = output.dtype
        return output


class _SoftmaxProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.recorded_output_dtype = None

    def forward(self, input):
        output = torch.softmax(input, dim=-1)
        self.recorded_output_dtype = output.dtype
        return output


class _FunctionalSoftmaxProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.recorded_output_dtype = None

    def forward(self, input):
        output = F.softmax(input, dim=-1)
        self.recorded_output_dtype = output.dtype
        return output


class _RecordingSelectiveMultiheadAttention(SelectiveMultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__(embed_dim, num_heads, dropout=dropout, bias=bias)
        self.recorded_q_dtype = None
        self.recorded_k_dtype = None
        self.recorded_v_dtype = None
        self.recorded_proj_dtype = None
        self.recorded_scores_dtype = None
        self.recorded_softmax_dtype = None

    def _project_bf16(self, inputs, weight, bias):
        output = super()._project_bf16(inputs, weight, bias)
        if weight.data_ptr() == self.in_proj_weight[: self.embed_dim].data_ptr():
            self.recorded_q_dtype = output.dtype
        elif weight.data_ptr() == self.in_proj_weight[self.embed_dim : 2 * self.embed_dim].data_ptr():
            self.recorded_k_dtype = output.dtype
        elif weight.data_ptr() == self.in_proj_weight[2 * self.embed_dim :].data_ptr():
            self.recorded_v_dtype = output.dtype
        elif weight.data_ptr() == self.out_proj.weight.data_ptr():
            self.recorded_proj_dtype = output.dtype
        return output

    def _softmax_fp32(self, scores):
        self.recorded_scores_dtype = scores.dtype
        output = super()._softmax_fp32(scores)
        self.recorded_softmax_dtype = output.dtype
        return output


class _RecordingGraphMultiHeadAttention(GraphMultiHeadAttention):
    def __init__(self, d_model, dropout, nheads):
        super().__init__(d_model, dropout, nheads)
        self.recorded_q_dtype = None
        self.recorded_k_dtype = None
        self.recorded_v_dtype = None
        self.recorded_proj_dtype = None
        self.recorded_scores_dtype = None
        self.recorded_softmax_dtype = None

    def _project_bf16(self, inputs, linear):
        output = super()._project_bf16(inputs, linear)
        if linear is self.linear_q:
            self.recorded_q_dtype = output.dtype
        elif linear is self.linear_k:
            self.recorded_k_dtype = output.dtype
        elif linear is self.linear_v:
            self.recorded_v_dtype = output.dtype
        elif linear is self.output_layer:
            self.recorded_proj_dtype = output.dtype
        return output

    def _softmax_fp32(self, scores):
        self.recorded_scores_dtype = scores.dtype
        output = super()._softmax_fp32(scores)
        self.recorded_softmax_dtype = output.dtype
        return output

    def reset_records(self):
        self.recorded_q_dtype = None
        self.recorded_k_dtype = None
        self.recorded_v_dtype = None
        self.recorded_proj_dtype = None
        self.recorded_scores_dtype = None
        self.recorded_softmax_dtype = None


class SelectiveAutocastTests(unittest.TestCase):
    def test_selective_autocast_runs_linear_and_activation_in_bf16(self):
        first_linear = _RecordingLinear(4, 8)
        activation_probe = _ActivationProbe()
        second_linear = _RecordingLinear(8, 2)
        model = nn.Sequential(first_linear, activation_probe, second_linear)

        patched = enable_selective_autocast(
            model,
            device_type='cpu',
            autocast_dtype=torch.bfloat16,
        )

        output = model(torch.randn(3, 4, dtype=torch.float32))

        self.assertEqual(patched, 2)
        # Inside autocast, Linear computes in bf16, but wrapper converts output to fp32
        # for precision preservation between layers.
        self.assertEqual(first_linear.recorded_output_dtype, torch.bfloat16)
        self.assertEqual(second_linear.recorded_output_dtype, torch.bfloat16)
        self.assertEqual(activation_probe.last_input_dtype, torch.float32)
        self.assertEqual(activation_probe.last_output_dtype, torch.float32)
        self.assertEqual(output.dtype, torch.float32)

    def test_selective_autocast_is_noop_without_bf16_dtype(self):
        linear = _RecordingLinear(4, 8)
        activation_probe = _ActivationProbe()
        model = nn.Sequential(linear, activation_probe)

        patched = enable_selective_autocast(
            model,
            device_type='cpu',
            autocast_dtype=torch.float16,
        )

        output = model(torch.randn(3, 4, dtype=torch.float32))

        self.assertEqual(patched, 0)
        self.assertEqual(linear.recorded_output_dtype, torch.float32)
        self.assertEqual(activation_probe.last_input_dtype, torch.float32)
        self.assertEqual(activation_probe.last_output_dtype, torch.float32)
        self.assertEqual(output.dtype, torch.float32)

    def test_selective_autocast_runs_conv_and_activation_in_bf16(self):
        conv = _RecordingConv1d(2, 4, kernel_size=1)
        activation_probe = _ActivationProbe()
        model = nn.Sequential(conv, activation_probe)

        patched = enable_selective_autocast(
            model,
            device_type='cpu',
            autocast_dtype=torch.bfloat16,
        )

        output = model(torch.randn(3, 2, 5, dtype=torch.float32))

        self.assertEqual(patched, 1)
        # Inside autocast, Conv computes in bf16, but wrapper converts output to fp32.
        self.assertEqual(conv.recorded_output_dtype, torch.bfloat16)
        self.assertEqual(activation_probe.last_input_dtype, torch.float32)
        self.assertEqual(activation_probe.last_output_dtype, torch.float32)
        self.assertEqual(output.dtype, torch.float32)

    def test_selective_autocast_forces_softmax_to_fp32(self):
        model = nn.Sequential(_SoftmaxProbe(), _FunctionalSoftmaxProbe())

        enable_selective_autocast(
            model,
            device_type='cpu',
            autocast_dtype=torch.bfloat16,
        )

        output = model(torch.randn(2, 4, dtype=torch.float32))

        self.assertEqual(model[0].recorded_output_dtype, torch.float32)
        self.assertEqual(model[1].recorded_output_dtype, torch.float32)
        self.assertEqual(output.dtype, torch.float32)

    def test_selective_multihead_attention_uses_bf16_qkv_proj_and_fp32_softmax(self):
        attn = _RecordingSelectiveMultiheadAttention(embed_dim=8, num_heads=2, dropout=0.0)
        attn.configure_precision(device_type='cpu', autocast_dtype=torch.bfloat16)

        query = torch.randn(5, 3, 8, dtype=torch.float32)
        attn_mask = torch.zeros(3 * 2, 5, 5, dtype=torch.float32)
        output, weights = attn(query, query, query, attn_mask=attn_mask, need_weights=True)

        self.assertEqual(attn.recorded_q_dtype, torch.bfloat16)
        self.assertEqual(attn.recorded_k_dtype, torch.bfloat16)
        self.assertEqual(attn.recorded_v_dtype, torch.bfloat16)
        self.assertEqual(attn.recorded_proj_dtype, torch.bfloat16)
        # CPU matmul of bf16 tensors returns fp32, so scores are fp32.
        self.assertEqual(attn.recorded_scores_dtype, torch.float32)
        self.assertEqual(attn.recorded_softmax_dtype, torch.float32)
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(weights.dtype, torch.float32)

    def test_selective_multihead_attention_stays_fp32_without_bf16_configuration(self):
        attn = _RecordingSelectiveMultiheadAttention(embed_dim=8, num_heads=2, dropout=0.0)
        attn.configure_precision(device_type='cpu', autocast_dtype=torch.float16)

        query = torch.randn(5, 3, 8, dtype=torch.float32)
        attn_mask = torch.zeros(3 * 2, 5, 5, dtype=torch.float32)
        output, weights = attn(query, query, query, attn_mask=attn_mask, need_weights=True)

        self.assertEqual(attn.recorded_q_dtype, torch.float32)
        self.assertEqual(attn.recorded_k_dtype, torch.float32)
        self.assertEqual(attn.recorded_v_dtype, torch.float32)
        self.assertEqual(attn.recorded_proj_dtype, torch.float32)
        # CPU bf16 matmul returns fp32, so attention scores are fp32.
        self.assertEqual(attn.recorded_scores_dtype, torch.float32)
        self.assertEqual(attn.recorded_softmax_dtype, torch.float32)
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(weights.dtype, torch.float32)

    def test_graph_multihead_attention_keeps_qkv_proj_bf16_but_uses_fp32_softmax(self):
        attn = _RecordingGraphMultiHeadAttention(d_model=8, dropout=0.0, nheads=2)
        enable_selective_autocast(
            attn,
            device_type='cpu',
            autocast_dtype=torch.bfloat16,
        )

        batch_size = 2
        sequence_length = 4
        q = torch.randn(batch_size, sequence_length, 8, dtype=torch.float32)
        distance = torch.randint(0, 3, (batch_size, sequence_length, sequence_length), dtype=torch.long)
        edge_attr = torch.randint(0, 4, (batch_size, sequence_length, sequence_length), dtype=torch.long)
        query_hop_emb = torch.randn(3, 8, dtype=torch.float32)
        query_edge_emb = torch.randn(4, 8, dtype=torch.float32)
        key_hop_emb = torch.randn(3, 8, dtype=torch.float32)
        key_edge_emb = torch.randn(4, 8, dtype=torch.float32)

        output = attn(
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
            None,
        )

        self.assertEqual(attn.recorded_q_dtype, torch.bfloat16)
        self.assertEqual(attn.recorded_k_dtype, torch.bfloat16)
        self.assertEqual(attn.recorded_v_dtype, torch.bfloat16)
        self.assertEqual(attn.recorded_proj_dtype, torch.bfloat16)
        # CPU matmul of bf16 tensors returns fp32, so scores are fp32.
        self.assertEqual(attn.recorded_scores_dtype, torch.float32)
        self.assertEqual(attn.recorded_softmax_dtype, torch.float32)
        self.assertEqual(output.dtype, torch.float32)

    def test_graph_multihead_attention_stays_fp32_without_bf16_configuration(self):
        attn = _RecordingGraphMultiHeadAttention(d_model=8, dropout=0.0, nheads=2)
        attn.configure_precision(device_type='cpu', autocast_dtype=None)
        attn.reset_records()

        batch_size = 2
        sequence_length = 4
        q = torch.randn(batch_size, sequence_length, 8, dtype=torch.float32)
        distance = torch.randint(0, 3, (batch_size, sequence_length, sequence_length), dtype=torch.long)
        edge_attr = torch.randint(0, 4, (batch_size, sequence_length, sequence_length), dtype=torch.long)
        query_hop_emb = torch.randn(3, 8, dtype=torch.float32)
        query_edge_emb = torch.randn(4, 8, dtype=torch.float32)
        key_hop_emb = torch.randn(3, 8, dtype=torch.float32)
        key_edge_emb = torch.randn(4, 8, dtype=torch.float32)

        output = attn(
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
            None,
        )

        self.assertEqual(attn.recorded_q_dtype, torch.float32)
        self.assertEqual(attn.recorded_k_dtype, torch.float32)
        self.assertEqual(attn.recorded_v_dtype, torch.float32)
        self.assertEqual(attn.recorded_proj_dtype, torch.float32)
        self.assertEqual(attn.recorded_scores_dtype, torch.float32)
        self.assertEqual(attn.recorded_softmax_dtype, torch.float32)
        self.assertEqual(output.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()