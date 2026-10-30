"""--noise_schedule_shift: the shifted beta schedule and the CFG switch that
follows it to the same noise level."""

from __future__ import annotations

import math
import sys
import types
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


from diffusion.gaussian_diffusion import (  # noqa: E402
    betas_for_alpha_bar,
    get_named_beta_schedule,
    matching_shifted_timestep,
    shift_beta_schedule,
)
from model.cfg_sampler import ClassifierFreeActionModel  # noqa: E402
from sample.conditioning import cfg_switch_timestep  # noqa: E402
from utils.model_util import create_gaussian_diffusion  # noqa: E402
from utils.parser_util import ArgumentParser, add_base_options  # noqa: E402


def _log_snr(betas):
    alphas_cumprod = np.cumprod(1.0 - np.asarray(betas, dtype=np.float64))
    return np.log(alphas_cumprod) - np.log1p(-alphas_cumprod)


def _args(shift, schedule='cosine', steps=100):
    return types.SimpleNamespace(
        diffusion_steps=steps, noise_schedule=schedule, noise_schedule_shift=shift,
    )


class NoiseScheduleShiftTest(unittest.TestCase):
    def test_shift_one_is_the_named_schedule(self):
        cosine = betas_for_alpha_bar(
            100, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2)
        np.testing.assert_array_equal(get_named_beta_schedule('cosine', 100), cosine)
        np.testing.assert_array_equal(get_named_beta_schedule('cosine', 100, shift=1.0), cosine)
        np.testing.assert_array_equal(
            get_named_beta_schedule('linear', 100, shift=1.0),
            get_named_beta_schedule('linear', 100))

    def test_shift_offsets_log_snr(self):
        for schedule in ('cosine', 'linear'):
            base = get_named_beta_schedule(schedule, 100)
            for shift in (1.5, 2.0, 3.0):
                shifted = get_named_beta_schedule(schedule, 100, shift=shift)
                self.assertEqual(shifted.shape, base.shape)
                self.assertTrue(np.all(shifted > 0.0) and np.all(shifted <= 0.999))
                # The last step is clipped at max_beta on both sides.
                np.testing.assert_allclose(
                    _log_snr(shifted)[:-1], _log_snr(base)[:-1] - 2.0 * math.log(shift),
                    rtol=0, atol=1e-6)

    def test_invalid_shift_rejected(self):
        betas = get_named_beta_schedule('cosine', 100)
        for shift in (0.0, -1.0, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                shift_beta_schedule(betas, shift)

    def test_shift_min_ramps_log_snr(self):
        base = get_named_beta_schedule('cosine', 100)
        np.testing.assert_array_equal(
            get_named_beta_schedule('cosine', 100, shift=2.0, shift_min=2.0),
            get_named_beta_schedule('cosine', 100, shift=2.0))
        shifted = get_named_beta_schedule('cosine', 100, shift=2.0)
        ramped = get_named_beta_schedule('cosine', 100, shift=2.0, shift_min=0.25)
        # First step carries shift_min, the noisy half is the plain shifted schedule.
        self.assertAlmostEqual(_log_snr(ramped)[0], _log_snr(base)[0] - 2.0 * math.log(0.25), places=6)
        noisy = _log_snr(base) < 0.0
        np.testing.assert_allclose(_log_snr(ramped)[noisy], _log_snr(shifted)[noisy], rtol=0, atol=1e-9)
        self.assertTrue(np.all(np.diff(_log_snr(ramped)[:-1]) < 0))

    def test_shift_min_above_shift_rejected(self):
        betas = get_named_beta_schedule('cosine', 100)
        for shift_min in (3.0, 0.0, float('nan')):
            with self.assertRaises(ValueError):
                shift_beta_schedule(betas, 2.0, shift_min=shift_min)

    def test_matching_timestep(self):
        base = get_named_beta_schedule('cosine', 100)
        self.assertEqual(matching_shifted_timestep(base, base, 50.0), 50.0)
        shifted = get_named_beta_schedule('cosine', 100, shift=2.0)
        t = matching_shifted_timestep(base, shifted, 50.0)
        self.assertLess(t, 50.0)
        steps = np.arange(100, dtype=np.float64)
        self.assertAlmostEqual(
            float(np.interp(t, steps, _log_snr(shifted))), float(_log_snr(base)[50]), places=6)

    def test_cfg_switch_follows_noise_level(self):
        self.assertEqual(
            cfg_switch_timestep(_args(1.0)), ClassifierFreeActionModel.CFG_SWITCH * 100)
        self.assertLess(cfg_switch_timestep(_args(2.0)), cfg_switch_timestep(_args(1.5)))
        wrapped = ClassifierFreeActionModel(
            lambda x, t, y=None: x, 2.0, 100, switch_t=cfg_switch_timestep(_args(2.0)))
        self.assertAlmostEqual(wrapped.switch_t, cfg_switch_timestep(_args(2.0)))
        self.assertEqual(ClassifierFreeActionModel(lambda x, t, y=None: x, 2.0, 100).switch_t, 50.0)

    def test_parser_default_and_diffusion(self):
        parser = ArgumentParser()
        add_base_options(parser)
        self.assertEqual(parser.parse_args([]).noise_schedule_shift, 1.0)
        args = parser.parse_args(['--noise_schedule_shift', '2'])
        for name, value in dict(sigma_small=True, lambda_geo=0.0).items():
            setattr(args, name, value)
        diffusion = create_gaussian_diffusion(args)
        np.testing.assert_allclose(
            diffusion.alphas_cumprod,
            np.cumprod(1.0 - get_named_beta_schedule('cosine', 100, shift=2.0)))
        self.assertIsNone(args.noise_schedule_shift_min)
        args.noise_schedule_shift_min = 0.25
        np.testing.assert_allclose(
            create_gaussian_diffusion(args).alphas_cumprod,
            np.cumprod(1.0 - get_named_beta_schedule('cosine', 100, shift=2.0, shift_min=0.25)))
        # CFG switch reads the same ramped schedule.
        ramped = _args(2.0)
        ramped.noise_schedule_shift_min = 0.25
        self.assertNotEqual(cfg_switch_timestep(ramped), cfg_switch_timestep(_args(2.0)))


if __name__ == '__main__':
    unittest.main()
