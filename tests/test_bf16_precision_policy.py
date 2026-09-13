"""Precision policy from docs/bf16_precision_issues.md 5.2 (A + B + D)."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


from eval import eval_checkpoint  # noqa: E402
from model.anytop import OutputProcess  # noqa: E402
from model.motion_transformer import SelectiveMultiheadAttention  # noqa: E402
from sample import generate  # noqa: E402
from utils import numerical_verification  # noqa: E402
from utils.fixseed import fixseed  # noqa: E402


class _MatmulPrecisionRestore(unittest.TestCase):
    def setUp(self):
        previous = torch.get_float32_matmul_precision()
        self.addCleanup(torch.set_float32_matmul_precision, previous)


class OutputProcessPrecisionTests(unittest.TestCase):
    def test_x0_prediction_is_computed_in_fp32_under_bf16_autocast(self):
        torch.manual_seed(0)
        head = OutputProcess(feature_len=12, root_feature_len=12, max_joints=4, latent_dim=16)
        latent = torch.randn(5, 2, 4, 16)  # [frames, batch, joints, latent_dim]
        expected = head(latent)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            actual = head(latent)

        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.equal(actual, expected))

    def test_bf16_residual_stream_is_upcast_before_projection(self):
        torch.manual_seed(0)
        head = OutputProcess(feature_len=12, root_feature_len=12, max_joints=4, latent_dim=16)
        latent_bf16 = torch.randn(5, 2, 4, 16).to(torch.bfloat16)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            actual = head(latent_bf16)

        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.equal(actual, head(latent_bf16.float())))


class InferencePrecisionTests(_MatmulPrecisionRestore):
    def test_fixseed_leaves_matmul_precision_alone(self):
        # generate.py reseeds before every batch; it must not switch TF32 back off.
        torch.set_float32_matmul_precision("high")
        fixseed(0)
        self.assertEqual(torch.get_float32_matmul_precision(), "high")

    def test_cuda_fp32_inference_enables_tf32(self):
        torch.set_float32_matmul_precision("highest")
        with patch.object(generate.dist_util, "dev", return_value=torch.device("cuda:0")):
            amp_dtype = generate._resolve_inference_amp_dtype(SimpleNamespace(amp_dtype="fp32"))
        self.assertEqual(amp_dtype, "fp32")
        self.assertEqual(torch.get_float32_matmul_precision(), "high")

    def test_cpu_inference_keeps_matmul_precision(self):
        torch.set_float32_matmul_precision("highest")
        with patch.object(generate.dist_util, "dev", return_value=torch.device("cpu")):
            amp_dtype = generate._resolve_inference_amp_dtype(SimpleNamespace(amp_dtype="fp32"))
        self.assertEqual(amp_dtype, "fp32")
        self.assertEqual(torch.get_float32_matmul_precision(), "highest")

    def test_eval_checkpoint_generates_in_fp32(self):
        common = eval_checkpoint._COMMON_GENERATE_ARGS
        self.assertEqual(common[common.index("--amp_dtype") + 1], "fp32")

    def test_task_checksum_changes_with_inference_precision(self):
        # Output generated under bf16 must be regenerated, not re-scored.
        task_args = ["--object_type", "Horse", "--num_frames", "90"]
        fp32_digest = eval_checkpoint._task_param_hash(task_args)
        bf16_common = ("--batch_size", "8", "--amp_dtype", "bf16")
        with patch.object(eval_checkpoint, "_COMMON_GENERATE_ARGS", bf16_common):
            bf16_digest = eval_checkpoint._task_param_hash(task_args)
        self.assertNotEqual(fp32_digest, bf16_digest)


class NumericalVerificationTests(_MatmulPrecisionRestore):
    def _restore_verification_globals(self):
        saved = {
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "cudnn_sdp": torch.backends.cuda.cudnn_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
        }

        def restore():
            torch.use_deterministic_algorithms(saved["deterministic"], warn_only=saved["warn_only"])
            torch.backends.cudnn.allow_tf32 = saved["cudnn_allow_tf32"]
            torch.backends.cudnn.benchmark = saved["cudnn_benchmark"]
            torch.backends.cudnn.deterministic = saved["cudnn_deterministic"]
            torch.backends.cuda.enable_flash_sdp(saved["flash"])
            torch.backends.cuda.enable_mem_efficient_sdp(saved["mem_efficient"])
            torch.backends.cuda.enable_cudnn_sdp(saved["cudnn_sdp"])
            torch.backends.cuda.enable_math_sdp(saved["math"])

        self.addCleanup(restore)

    def test_verification_mode_disables_tf32_nondeterminism_and_fast_sdpa(self):
        self._restore_verification_globals()
        torch.set_float32_matmul_precision("high")
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}):
            numerical_verification.enable_numerical_verification_mode()

        self.assertEqual(torch.get_float32_matmul_precision(), "highest")
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.backends.cuda.flash_sdp_enabled())
        self.assertFalse(torch.backends.cuda.mem_efficient_sdp_enabled())
        self.assertFalse(torch.backends.cuda.cudnn_sdp_enabled())
        self.assertTrue(torch.backends.cuda.math_sdp_enabled())

    def test_verification_mode_sets_cublas_workspace_before_cuda_init(self):
        self._restore_verification_globals()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            with patch.object(torch.cuda, "is_initialized", return_value=False):
                numerical_verification.enable_numerical_verification_mode()
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")

    def test_verification_mode_refuses_late_cublas_workspace(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            with patch.object(torch.cuda, "is_available", return_value=True), \
                    patch.object(torch.cuda, "is_initialized", return_value=True):
                with self.assertRaises(RuntimeError):
                    numerical_verification.enable_numerical_verification_mode()

    def test_disable_dropout_zeroes_module_and_attention_rates(self):
        model = nn.Sequential(
            nn.Linear(8, 8),
            nn.Dropout(0.1),
            SelectiveMultiheadAttention(8, 2, dropout=0.2),
            nn.Dropout(0.0),
        )

        self.assertEqual(numerical_verification.disable_dropout(model), 2)
        self.assertEqual(model[1].p, 0.0)
        self.assertEqual(model[2].dropout, 0.0)

        model.train()
        x = torch.randn(3, 4, 8)
        self.assertTrue(torch.equal(model[1](x), x))

    def test_reseed_repeats_torch_numpy_and_random_draws(self):
        import random

        import numpy as np

        numerical_verification.reseed(7)
        first = (torch.rand(3), np.random.rand(3), random.random())
        numerical_verification.reseed(7)
        second = (torch.rand(3), np.random.rand(3), random.random())

        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(np.array_equal(first[1], second[1]))
        self.assertEqual(first[2], second[2])


if __name__ == "__main__":
    unittest.main()
