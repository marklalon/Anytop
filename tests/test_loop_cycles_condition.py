"""The loop cycle-count condition (docs/loop_cycle_condition.md): a loop row's
frame position is the circular table of period T / k, its tempo the cycle
length L / k, plus an explicit k embedding with a null for "unknown"; and
generation's rules for picking k and the output length."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_loaders.reference import REST_GEOM_DIM  # noqa: E402
from data_loaders.truebones.truebones_utils.joint_struct_features import JOINT_STRUCT_DIM  # noqa: E402
from model.anytop import AnyTop, create_sin_embedding  # noqa: E402
from model.motion_transformer import circular_phase_embedding  # noqa: E402
from sample.output_lengths import (  # noqa: E402
    reference_loop_cycles,
    resolve_loop_cycles,
    warn_loop_cycles_range,
)
from utils.clip_length_prior import build_clip_length_prior, COND_KEY, loop_period_prior  # noqa: E402
from utils.parser_util import LOOP_CYCLES_NULL, parse_loop_cycles  # noqa: E402

CPU = torch.device('cpu')
T5_DIM = 32
LATENT = 16


class _CaptureDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_kwargs = None

    def forward(self, **kwargs):
        self.last_kwargs = kwargs
        return kwargs['tgt']


def _model(**kwargs):
    torch.manual_seed(0)
    model = AnyTop(max_joints=4, feature_len=12, latent_dim=LATENT, ff_size=32, num_layers=1,
                   num_heads=2, dropout=0.0, cross_limb=True, t5_out_dim=T5_DIM, **kwargs)
    # Trained heads: the zero-init last layers would hide every effect.
    for head in (model.tempo_projection, model.loop_period_projection, model.loop_cycles_projection):
        torch.nn.init.normal_(head[-1].weight, std=0.5)
        torch.nn.init.normal_(head[-1].bias, std=0.5)
    torch.nn.init.normal_(model.loop_cycles_null_emb, std=0.5)
    return model.eval()


def _y(batch, **extra):
    y = {
        'joints_padding_mask': torch.ones(batch, 1, 1, 5, 5),
        'rest_geom': torch.zeros(batch, 4, REST_GEOM_DIM),
        'n_joints': torch.full((batch,), 4, dtype=torch.int64),
        'joints_names_embs': torch.zeros(batch, 4, T5_DIM),
        'joint_struct': torch.zeros(batch, 4, JOINT_STRUCT_DIM),
        'resample_speed_cond': torch.ones(batch),
    }
    y.update(extra)
    return y


def _decoder_input(model, y, frames):
    capture = _CaptureDecoder()
    model.seqTransDecoder = capture
    batch = y['n_joints'].shape[0]
    with torch.no_grad():
        model(torch.zeros(batch, 4, 12, frames), torch.ones(batch, dtype=torch.long), y=y)
    return capture.last_kwargs


# -- The circular table -------------------------------------------------------
@pytest.mark.parametrize('length, dim, cycles', [(12, 16, 3), (12, 16, 4), (60, 256, 2), (60, 256, 3)])
def test_tiled_copies_get_identical_rows_when_k_divides_t(length, dim, cycles):
    table = circular_phase_embedding(length, dim, CPU, torch.float32, cycles=torch.tensor([cycles]))[:, 0]
    period = length // cycles
    for copy in range(1, cycles):
        assert torch.equal(table[copy * period:(copy + 1) * period], table[:period])


@pytest.mark.parametrize('length, dim, cycles', [(12, 16, 5), (12, 16, 7), (61, 256, 2), (61, 256, 3)])
def test_coprime_k_permutes_the_single_cycle_table(length, dim, cycles):
    base = circular_phase_embedding(length, dim, CPU, torch.float32)
    table = circular_phase_embedding(length, dim, CPU, torch.float32, cycles=torch.tensor([cycles]))[:, 0]
    rows = (torch.arange(length) * cycles) % length
    assert sorted(rows.tolist()) == list(range(length))
    assert torch.equal(table, base[rows])


def test_shared_factor_repeats_a_subset_of_rows():
    length, dim, cycles = 12, 16, 8  # gcd 4: 3 distinct rows, each 4 times
    base = circular_phase_embedding(length, dim, CPU, torch.float32)
    table = circular_phase_embedding(length, dim, CPU, torch.float32, cycles=torch.tensor([cycles]))[:, 0]
    assert torch.equal(table, base[(torch.arange(length) * cycles) % length])
    assert len({tuple(row.tolist()) for row in table}) == length // math.gcd(length, cycles)


def test_batched_table_is_per_sample():
    cycles = torch.tensor([1, 2, 3])
    table = circular_phase_embedding(6, 8, CPU, torch.float32, cycles=cycles)
    assert table.shape == (6, 3, 8)
    for row, k in enumerate(cycles.tolist()):
        expected = circular_phase_embedding(6, 8, CPU, torch.float32, cycles=torch.tensor([k]))[:, 0]
        assert torch.equal(table[:, row], expected)
    assert torch.equal(table[:, 0], circular_phase_embedding(6, 8, CPU, torch.float32))


# -- Model inputs -------------------------------------------------------------
def test_loop_rows_get_the_circular_table_instead_of_the_absolute_pe():
    model = _model()
    frames = 6
    pos = model._frame_position_embedding(
        frames, torch.tensor([True, True, False]), torch.tensor([2, 1, 3]), CPU,
    )
    absolute = create_sin_embedding(torch.arange(frames).view(1, -1, 1), LATENT)[0]
    assert torch.equal(pos[:, 0], circular_phase_embedding(frames, LATENT, CPU, torch.float32,
                                                            cycles=torch.tensor([2]))[:, 0])
    assert torch.equal(pos[:, 1], circular_phase_embedding(frames, LATENT, CPU, torch.float32))
    # An open row keeps the absolute PE whatever loop_tile_count says.
    assert torch.equal(pos[:, 2], absolute)


def test_decoder_gets_the_same_period_as_the_input():
    model = _model()
    y = _y(2, is_loop=torch.tensor([True, False]), loop_tile_count=torch.tensor([3, 2]))
    kwargs = _decoder_input(model, y, 6)
    assert kwargs['loop_cycles'].tolist() == [3, 1]
    assert kwargs['loop_phase_mask'].tolist() == [True, False]


def test_unknown_k_and_k1_give_different_inputs():
    """Both have a period-T table and L as the cycle length; only the null
    vector tells them apart."""
    model = _model()
    loop = dict(is_loop=torch.tensor([True, True]), loop_tile_count=torch.tensor([1, 1]))
    known = _decoder_input(model, _y(2, **loop, loop_cycles_active=torch.tensor([True, True])), 6)
    unknown = _decoder_input(model, _y(2, **loop, loop_cycles_active=torch.tensor([False, False])), 6)
    assert not torch.allclose(known['tgt'], unknown['tgt'])
    assert torch.equal(known['loop_cycles'], unknown['loop_cycles'])


def test_unknown_k_reads_as_one_period_per_window():
    model = _model()
    loop = dict(is_loop=torch.tensor([True]), loop_tile_count=torch.tensor([3]),
                loop_cycles_active=torch.tensor([False]))
    is_loop, cycles, known = model._loop_condition(_y(1, **loop), 1, CPU)
    assert cycles.tolist() == [1] and known.tolist() == [False]


def test_open_rows_ignore_the_cycle_count():
    model = _model()
    base = dict(is_loop=torch.tensor([False, False]))
    a = _decoder_input(model, _y(2, **base, loop_tile_count=torch.tensor([1, 1])), 6)
    b = _decoder_input(model, _y(2, **base, loop_tile_count=torch.tensor([3, 2]),
                                 loop_cycles_active=torch.tensor([False, True])), 6)
    assert torch.equal(a['tgt'], b['tgt'])


def test_cycle_count_survives_the_tempo_drop():
    """A tempo-dropped loop row loses L but keeps k."""
    model = _model()
    off = torch.zeros(2, dtype=torch.bool)
    loop = dict(is_loop=torch.tensor([True, True]), tempo_active=off)
    one = _decoder_input(model, _y(2, **loop, loop_tile_count=torch.tensor([1, 1])), 6)
    two = _decoder_input(model, _y(2, **loop, loop_tile_count=torch.tensor([2, 2])), 6)
    assert not torch.allclose(one['tgt'], two['tgt'])


def test_training_drops_the_cycle_count():
    model = _model(loop_cycles_drop_prob=1.0)
    model.train()
    y = _y(4, is_loop=torch.ones(4, dtype=torch.bool), loop_tile_count=torch.full((4,), 2))
    _, cycles, known = model._loop_condition(y, 4, CPU)
    assert not known.any() and cycles.tolist() == [1, 1, 1, 1]
    model.eval()
    _, cycles, known = model._loop_condition(y, 4, CPU)
    assert known.all() and cycles.tolist() == [2, 2, 2, 2]
    with pytest.raises(ValueError):
        _model(loop_cycles_drop_prob=1.5)


def test_reference_rows_read_the_count_as_unknown():
    """A reference-active row bypasses the cycle count, like the label."""
    model = _model()
    y = _y(3, is_loop=torch.ones(3, dtype=torch.bool), loop_tile_count=torch.tensor([2, 2, 3]))
    _, cycles, known = model._loop_condition(y, 3, CPU, ref_active=torch.tensor([True, False, True]))
    assert known.tolist() == [False, True, False]
    assert cycles.tolist() == [1, 2, 1]


def test_create_condition_maps_none_to_unknown():
    from data_loaders.tensors import truebones_batch_collate
    items = []
    for cycles in (2, None):
        metadata = {'is_loop': True, 'loop_tile_count': 1 if cycles is None else cycles,
                    'loop_cycles_active': cycles is not None, 'translation_root_index': 0}
        items.append(metadata)
    # The collate reads the keys from the motion metadata.
    from tests.test_native_loop import _make_batch_item
    batch = []
    for metadata in items:
        item = list(_make_batch_item(True))
        item[10] = dict(item[10], **metadata)
        batch.append(tuple(item))
    _, cond = truebones_batch_collate(batch)
    assert cond['y']['loop_tile_count'].tolist() == [2, 1]
    assert cond['y']['loop_cycles_active'].tolist() == [True, False]


# -- Generation: picking k and the length --------------------------------------
RUN = 'run, forward'


def test_parse_loop_cycles():
    import argparse
    assert parse_loop_cycles('3') == 3
    assert parse_loop_cycles('None') == LOOP_CYCLES_NULL
    for bad in ('0', '-1', 'two'):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_loop_cycles(bad)


def test_neither_given_is_one_cycle_at_the_period():
    assert resolve_loop_cycles(None, None, loop=True, action_label=RUN, period=(30, False),
                               verbose=False) == (1, None)


def test_k_alone_sets_the_length():
    assert resolve_loop_cycles(2, None, loop=True, action_label=RUN, period=(31, False),
                               verbose=False) == (2, 62)


def test_k_alone_without_a_period_is_refused():
    with pytest.raises(SystemExit):
        resolve_loop_cycles(2, None, loop=True, action_label=RUN, period=None, verbose=False)


@pytest.mark.parametrize('frames, expected', [(31, 1), (43, 1), (44, 2), (62, 2), (75, 2), (76, 3), (10, 1)])
def test_length_alone_rounds_in_log_space(frames, expected):
    # Boundaries at P * sqrt(k (k + 1)): 31 * 1.414 = 43.8, 31 * 2.449 = 75.9.
    cycles, out = resolve_loop_cycles(None, frames, loop=True, action_label=RUN,
                                      period=(31, False), verbose=False)
    assert (cycles, out) == (expected, frames)


@pytest.mark.parametrize('frames, expected', [(30, 1), (59, 1), (60, 2), (89, 2), (20, 1)])
def test_high_frequency_rounds_down(frames, expected):
    cycles, _ = resolve_loop_cycles(None, frames, loop=True, action_label='fly, hover',
                                    period=(30, True), verbose=False)
    assert cycles == expected


def test_both_given_are_kept():
    assert resolve_loop_cycles(3, 50, loop=True, action_label=RUN, period=(31, False),
                               verbose=False) == (3, 50)


def test_none_is_unknown():
    assert resolve_loop_cycles(LOOP_CYCLES_NULL, 50, loop=True, action_label=RUN,
                               period=(31, False), verbose=False) == (None, 50)


def test_phase_anchored_loop_takes_one_cycle_only():
    assert resolve_loop_cycles(None, 80, loop=True, action_label='attack, bite',
                               period=(30, False), verbose=False) == (1, 80)
    with pytest.raises(SystemExit):
        resolve_loop_cycles(2, None, loop=True, action_label='attack, bite', period=(30, False),
                            verbose=False)


def test_open_window_has_no_cycle_count():
    assert resolve_loop_cycles(None, 40, loop=False, action_label=RUN, verbose=False) == (None, 40)
    with pytest.raises(SystemExit):
        resolve_loop_cycles(2, 40, loop=False, action_label=RUN, verbose=False)


def test_reference_cycle_count_is_always_unknown():
    assert reference_loop_cycles(None, loop=True, verbose=False) is None
    assert reference_loop_cycles(LOOP_CYCLES_NULL, loop=True, verbose=False) is None
    with pytest.raises(SystemExit):
        reference_loop_cycles(1, loop=True, verbose=False)


def test_warns_on_a_cycle_length_between_k1_and_k2():
    # R = 1.3: k = 1 covers up to 1.3 P, k = 2 from 2 P / 1.3 = 1.54 P.
    assert warn_loop_cycles_range(1, 44, period=(31, False), internal_num_frames=60,
                                  motion_speed_aug=1.3)
    assert warn_loop_cycles_range(2, 44, period=(31, False), internal_num_frames=60,
                                  motion_speed_aug=1.3)
    assert not warn_loop_cycles_range(1, 40, period=(31, False), internal_num_frames=60,
                                      motion_speed_aug=1.3)
    assert not warn_loop_cycles_range(2, 62, period=(31, False), internal_num_frames=60,
                                      motion_speed_aug=1.3)


def test_warns_on_a_fast_high_frequency_cycle_and_its_budget():
    # High-frequency: only slowed down, so L / k < P was never drawn.
    warnings = warn_loop_cycles_range(2, 50, period=(30, True), internal_num_frames=60,
                                      motion_speed_aug=1.3)
    assert len(warnings) == 1 and 'high-frequency' in warnings[0]
    # Longer than the window: cropped open in training.
    warnings = warn_loop_cycles_range(2, 70, period=(30, True), internal_num_frames=60,
                                      motion_speed_aug=1.3)
    assert len(warnings) == 1 and 'source budget' in warnings[0]


def test_loop_period_prior_reports_high_frequency_majority():
    def entry(rows):
        return {COND_KEY: build_clip_length_prior(rows)}

    cond = {
        'ns/Bee': entry([('locomotion', 'fly, forward', True, 20, True)] * 2
                        + [('locomotion', 'fly, forward', True, 24, False)]
                        # A high-frequency loop longer than the window trains open.
                        + [('locomotion', 'fly, forward', True, 90, True)] * 3),
    }
    (period, high_frequency), _ = loop_period_prior(
        cond, 'ns/Bee', action_group='locomotion', action_label='fly, forward',
        min_frames=10, max_frames=120, window_frames=60,
    )
    assert period == 20 and high_frequency
