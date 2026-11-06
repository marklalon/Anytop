"""Contract checks for the minimal self-reconstruction AE."""
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loaders.reference import REST_GEOM_DIM
from data_loaders.truebones.truebones_utils.joint_struct_features import JOINT_STRUCT_DIM
from model.motion_autoencoder import (
    MotionAutoencoder, TemporalResidualConv, to_working_motion,
)
from train.train_autoencoder import reconstruction_mse, scaled_optimizer_step


def test_cosine_schedule_resume_matches_uninterrupted_optimizer():
    from train.train_autoencoder import create_lr_scheduler, parser
    args = parser().parse_args([
        '--save_dir', 'unused', '--lr', '1e-4', '--lr_final', '5e-5',
        '--lr_decay_start', '0', '--num_steps', '10', '--weight_decay', '.03',
        '--use_ema', '--ema_rate', '.995'])
    assert args.use_ema and args.ema_rate == .995
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = create_lr_scheduler(optimizer, args)
    rates = []
    for step in range(10):
        rates.append(optimizer.param_groups[0]['lr'])
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
        if step == 3:
            import copy
            saved = copy.deepcopy(optimizer.state_dict())
            saved_parameter = parameter.detach().clone()
    assert rates[0] == args.lr
    assert all(a > b for a, b in zip(rates, rates[1:]))
    assert optimizer.param_groups[0]['lr'] == pytest.approx(args.lr_final)
    resumed_parameter = torch.nn.Parameter(saved_parameter)
    resumed = torch.optim.AdamW([resumed_parameter], lr=args.lr)
    resumed.load_state_dict(saved)
    resumed_scheduler = create_lr_scheduler(resumed, args, completed_steps=4)
    for expected in rates[4:]:
        assert resumed.param_groups[0]['lr'] == pytest.approx(expected)
        resumed_parameter.grad = torch.ones_like(resumed_parameter)
        resumed.step()
        resumed_scheduler.step()
    torch.testing.assert_close(resumed_parameter, parameter, rtol=0, atol=0)


def test_autoencoder_ema_averages_parameters_copies_stats_and_ignores_caches():
    import copy
    from train.train_autoencoder import update_autoencoder_ema
    model = torch.nn.Linear(2, 1)
    model.register_buffer('stat', torch.tensor(1.))
    model.register_buffer('cache', torch.zeros(1), persistent=False)
    averaged = copy.deepcopy(model).eval().requires_grad_(False)
    original = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(2)
        model.stat.fill_(3)
        model.cache = torch.zeros(5)
    update_autoencoder_ema(averaged, model, .995)
    torch.testing.assert_close(averaged.weight, original + .01)
    assert averaged.stat.item() == 3
    assert averaged.cache.shape == (1,)
    restored = copy.deepcopy(averaged)
    restored.load_state_dict(averaged.state_dict())
    update_autoencoder_ema(restored, model, .995)
    update_autoencoder_ema(averaged, model, .995)
    torch.testing.assert_close(restored.weight, averaged.weight, rtol=0, atol=0)


class LoaderFixture(torch.utils.data.Dataset):
    """Small picklable dataset: both joint buckets are smaller than a batch."""

    def __init__(self):
        self.motion_dataset = self
        self.opt = SimpleNamespace(max_joints=100, sources=[])
        self.data_dict = {}
        self.use_weighted_sampler = False
        self.counts = [5, 7, 12]

    def __len__(self):
        return len(self.counts)

    def sampler_index_joint_counts(self):
        return self.counts

    def __getitem__(self, index):
        import numpy as np
        from .test_joint_buckets import _item
        item = _item(self.counts[index], 100, seed=index)
        # Simulate a topology augmentation that crosses the original bucket.
        if index == 1 and self.opt.topology_aug_prob:
            item = _item(9, 100, seed=index)
        item[0] += np.random.random()
        item[11] = f'clip_{index}'
        return item


@pytest.mark.parametrize('num_workers', [0, 2])
@pytest.mark.parametrize('topology_prob', [0, 1])
def test_shared_loader_prefetches_full_bucketed_ae_batches(num_workers, topology_prob, monkeypatch):
    from itertools import islice
    from data_loaders.get_data import BackgroundPrefetchLoader, EagerWorkerLoader
    from train.train_autoencoder import (
        create_training_data_loader, parser, training_batches, training_batch_to_device,
    )
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    args = parser().parse_args([
        '--save_dir', 'unused', '--batch_size', '4', '--num_workers', str(num_workers),
        '--main_process_prefetch_batches', '2', '--joint_buckets', '8', '16', '100',
        '--joint_name_drop_prob', '1', '--rest_geom_joint_drop_prob', '1',
        '--topology_aug_prob', str(topology_prob),
    ])
    dataset = LoaderFixture()
    loader = create_training_data_loader(args, dataset)
    assert isinstance(loader, EagerWorkerLoader if num_workers else BackgroundPrefetchLoader)
    assert loader.dataset is dataset
    names, widths = set(), set()
    try:
        batches = list(islice(training_batches(loader), 3))
        for batch in batches:
            target, cond = training_batch_to_device(batch, 'cpu')
            assert target.shape[0] == 4 and target.shape[2:] == (9, 3)
            assert target.shape[1] in (8, 16, 100)
            assert cond['graph_dist'].shape == (4, target.shape[1], target.shape[1])
            assert cond['ae_source_skeleton']['joints_names_embs'].count_nonzero() == 0
            assert cond['ae_target_skeleton']['joints_names_embs'].count_nonzero() == 0
            assert cond['ae_source_skeleton']['rest_geom'].count_nonzero() == 0
            assert torch.equal(cond['ae_target_skeleton']['rest_geom'], cond['rest_geom'])
            assert not any(k.startswith('ref_') for k in cond)
            names.update(cond['motion_name'])
            widths.add(target.shape[1])
        assert names == {'clip_0', 'clip_1', 'clip_2'}
        # Augmentation grows clip_1 into the next ceiling; no arbitrary widths.
        assert widths == ({16} if topology_prob else {8, 16})
        assert not torch.equal(batches[0][0], batches[1][0])
    finally:
        if num_workers:
            loader.loader._iterator._shutdown_workers()


def test_ae_loader_defaults_to_diffusion_joint_buckets(monkeypatch):
    import train.train_autoencoder as trainer
    from data_loaders.truebones.truebones_utils.param_utils import JOINT_BUCKETS
    captured = {}
    monkeypatch.setattr(trainer, 'build_dataset_loader',
                        lambda dataset, **kwargs: captured.update(kwargs))
    args = trainer.parser().parse_args(['--save_dir', 'unused'])
    trainer.create_training_data_loader(args, LoaderFixture())
    assert captured['joint_buckets'] is JOINT_BUCKETS
    assert captured['drop_last'] is True


def test_diffusion_loader_delegates_to_the_same_factory(monkeypatch):
    import data_loaders.get_data as shared
    dataset = LoaderFixture()
    captured = {}
    monkeypatch.setattr(shared, 'get_dataset', lambda **kwargs: dataset)

    def build(actual_dataset, **kwargs):
        assert actual_dataset is dataset
        captured.update(kwargs)
        return 'shared-loader'

    monkeypatch.setattr(shared, 'build_dataset_loader', build)
    result = shared.get_dataset_loader(
        batch_size=4, num_frames=3, num_workers=2, main_process_prefetch_batches=64,
        joint_buckets=(48, 64, 100),
    )
    assert result == 'shared-loader'
    assert captured['num_workers'] == 2
    assert captured['main_process_prefetch_batches'] == 64
    assert captured['joint_buckets'] == (48, 64, 100)


@pytest.mark.parametrize('frames', [1, 2, 60])
@pytest.mark.parametrize('is_loop', [False, True])
@pytest.mark.parametrize('cycles', [1, 3])
def test_physical_time_embedding_matches_old_diffusion(frames, is_loop, cycles):
    from model.motion_diffusion import MotionDiffusion
    model, _, _ = fixture(frames=frames)
    # Nonzero heads expose time semantics, including the K=1 30/90 distinction.
    model.tempo_projection = torch.nn.Identity()
    model.loop_period_projection = torch.nn.Identity()
    metadata = dict(resample_speed_cond=torch.tensor([.5, 1.5]),
                    is_loop=torch.full((2,), is_loop), loop_tile_count=torch.full((2,), cycles))
    old = SimpleNamespace(latent_dim=model.width, tempo_projection=model.tempo_projection,
                          loop_period_projection=model.loop_period_projection)
    old._coerce_resample_speed_cond = lambda raw, **kwargs: MotionDiffusion._coerce_resample_speed_cond(old, raw, **kwargs)
    expected = MotionDiffusion._tempo_frame_embedding(old, metadata, 2, frames, torch.device('cpu'),
                                            metadata['is_loop'], metadata['loop_tile_count'])
    actual = model._tempo_embedding(torch.ones(2, frames, dtype=torch.bool), metadata).squeeze(2)
    torch.testing.assert_close(actual, expected)
    if is_loop or frames > 1:
        assert not torch.allclose(actual[:, 0], actual[:, 1])


@pytest.mark.parametrize('temporal_rope', [False, True])
def test_encoder_and_decoder_both_learn_physical_time(temporal_rope):
    model, x, skeleton = fixture(temporal_rope=temporal_rope)
    metadata = dict(is_loop=torch.tensor([False, True]), loop_tile_count=torch.ones(2),
                    resample_speed_cond=torch.tensor([.5, 1.5]))
    # Both open/loop zero-init heads must get gradients through E and through D.
    z = model.encode(x, skeleton, loop_metadata=metadata)
    z.square().mean().backward()
    for head in (model.tempo_projection, model.loop_period_projection):
        assert head[-1].weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    model.decode(z.detach(), skeleton, loop_metadata=metadata).square().mean().backward()
    for head in (model.tempo_projection, model.loop_period_projection):
        assert head[-1].weight.grad.abs().sum() > 0
        torch.nn.init.normal_(head[-1].weight, std=.1)
    other = dict(metadata, resample_speed_cond=torch.tensor([1.5, .5]))
    assert not torch.allclose(model.encode(x, skeleton, loop_metadata=metadata),
                              model.encode(x, skeleton, loop_metadata=other))
    assert not torch.allclose(model.decode(z, skeleton, loop_metadata=metadata),
                              model.decode(z, skeleton, loop_metadata=other))


@pytest.mark.parametrize('speed', [0., -1., float('nan'), float('inf')])
def test_invalid_physical_speed_is_rejected(speed):
    model, _, _ = fixture()
    with pytest.raises(ValueError, match='finite and positive'):
        model._tempo_embedding(torch.ones(2, 5, dtype=torch.bool), {'resample_speed_cond': speed})


def fixture(joints=4, frames=5, **options):
    torch.manual_seed(42)
    model = MotionAutoencoder(t5_out_dim=6, width=16, layers=1, heads=2, slots=2, latent_dim=4,
                              **options).eval()
    x = torch.randn(2, joints, 9, frames)
    s = dict(n_joints=torch.tensor([joints, max(1, joints - 1)]),
             joints_names_embs=torch.randn(2, joints, 6), rest_geom=torch.randn(2, joints, REST_GEOM_DIM),
             joint_struct=torch.randn(2, joints, JOINT_STRUCT_DIM),
             graph_dist=torch.zeros(2, joints, joints, dtype=torch.long),
             joints_relations=torch.zeros(2, joints, joints, dtype=torch.long))
    return model, x, s


def test_static_bucket_shapes_compile_once_and_reuse_graphs():
    model, x8, skeleton8 = fixture(joints=8, temporal_rope=True, layerwise_latent=True)
    _, x16, skeleton16 = fixture(joints=16, temporal_rope=True, layerwise_latent=True)
    graphs = []

    def backend(graph, example_inputs):
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    for x, skeleton in [(x8, skeleton8), (x16, skeleton16)] * 2:
        actual = compiled(x, skeleton)
        torch.testing.assert_close(actual, model(x, skeleton))
        actual.square().mean().backward()
    assert len(graphs) == 2


def test_asymmetric_depth_reads_latent_initially_and_in_all_decoder_layers():
    model, x, skeleton = fixture(
        encoder_layers=4, decoder_layers=6, layerwise_latent=True,
        temporal_rope=True, variational=True,
    )
    assert len(model.encoder.layers) == 4
    assert len(model.decoder.layers) == 6
    assert all(hasattr(layer, 'motion_latent_attn') for layer in model.decoder.layers)
    calls = []
    handles = [model.read_latent.register_forward_hook(lambda *args: calls.append('initial'))]
    for index in range(6):
        handles.append(model.decoder.layers[index].motion_latent_attn.register_forward_hook(
            lambda *args, index=index: calls.append(index)))
    try:
        prediction = model(x, skeleton)
    finally:
        for handle in handles:
            handle.remove()
    assert calls == ['initial', 0, 1, 2, 3, 4, 5]
    prediction.square().mean().backward()
    assert model.read_latent.in_proj_weight.grad.abs().sum() > 0
    assert model.to_latent.weight.grad.abs().sum() > 0
    for layer in model.decoder.layers:
        assert layer.motion_latent_attn.q_proj_weight.grad.abs().sum() > 0
    restored = MotionAutoencoder(**model.config)
    restored.load_state_dict(model.state_dict(), strict=True)
    restored.eval()
    torch.testing.assert_close(restored(x, skeleton), prediction)


@pytest.mark.parametrize('options,match', [
    ({'encoder_layers': 0}, 'dimensions'),
    ({'decoder_layers': 0}, 'dimensions'),
    ({'encoder_width': 0}, 'dimensions'),
    ({'decoder_width': 15}, 'dimensions'),
    ({'decoder_width': 18, 'temporal_rope': True}, 'even head'),
    ({'encoder_width': 18, 'temporal_rope': True}, 'even head'),
    ({'encoder_ff_size': -1}, 'ff_size'),
    ({'decoder_ff_size': -1}, 'ff_size'),
])
def test_asymmetric_configuration_validation(options, match):
    with pytest.raises(ValueError, match=match):
        fixture(**options)


def test_asymmetric_latent_readers_support_fullgraph_training():
    model, x, skeleton = fixture(
        encoder_layers=1, decoder_layers=3, layerwise_latent=True,
        temporal_rope=True, encoder_width=8, decoder_width=24,
        encoder_ff_size=20, decoder_ff_size=40,
    )
    compiled = torch.compile(model, backend='eager', fullgraph=True, dynamic=False)
    prediction = compiled(x, skeleton)
    torch.testing.assert_close(prediction, model(x, skeleton))
    prediction.square().mean().backward()
    assert model.decoder.layers[-1].motion_latent_attn.q_proj_weight.grad.abs().sum() > 0


@pytest.mark.parametrize('encoder_width,decoder_width', [(8, 24), (24, 8)])
@pytest.mark.parametrize('rope', [False, True])
def test_independent_widths_and_ff_sizes_roundtrip(encoder_width, decoder_width, rope):
    model, x, skeleton = fixture(
        encoder_width=encoder_width, decoder_width=decoder_width,
        encoder_ff_size=20, decoder_ff_size=40, last_layer_ff=12,
        decoder_layers=2, temporal_rope=rope, variational=True,
        layerwise_latent=True, decoder_temporal_kernel=3,
        cross_limb_dim=8, cross_limb_last_n=1,
    )
    assert model.encoder.layers[0].linear1.out_features == 20
    assert model.decoder.layers[0].linear1.out_features == 40
    assert model.decoder.layers[-1].linear1.out_features == 12
    frame = torch.tensor([[True] * 5, [True] * 3 + [False] * 2])
    metadata = dict(is_loop=torch.tensor([False, True]), loop_tile_count=torch.ones(2))
    prediction, mu, logvar = model(x, skeleton, frame, metadata, return_posterior=True)
    assert prediction.shape == x.shape
    assert mu.shape == logvar.shape == (2, 5, 2, 4)
    assert prediction[1, :, :, 3:].count_nonzero() == 0
    prediction.square().mean().backward()
    for module in (model.motion_projection, model.to_latent, model.from_latent,
                   model.decoder_names, model.decoder_tempo_projection[-1],
                   model.decoder_loop_period_projection[-1]):
        assert module.weight.grad is not None
        assert torch.isfinite(module.weight.grad).all()
        assert module.weight.grad.abs().sum() > 0
    restored = MotionAutoencoder(**model.config).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored(x, skeleton, frame, metadata), prediction)


def test_independent_dimension_defaults():
    model, _, _ = fixture(ff_size=48)
    assert model.encoder_width == model.decoder_width == 16
    assert model.encoder.layers[0].linear1.out_features == 48
    assert model.decoder.layers[0].linear1.out_features == 48
    independent, _, _ = fixture(encoder_width=8, decoder_width=24,
                                ff_size=48, decoder_ff_size=0)
    assert independent.encoder.layers[0].linear1.out_features == 48
    assert independent.decoder.layers[0].linear1.out_features == 96


def test_independent_dimension_cli():
    from train.train_autoencoder import parser
    options = dict(encoder_width=64, decoder_width=128,
                   encoder_ff_size=256, decoder_ff_size=512)
    defaults = parser().parse_args(['--save_dir', 'unused'])
    assert all(getattr(defaults, key) is None for key in options)
    cli = ['--save_dir', 'unused']
    for key, value in options.items():
        cli.extend(['--' + key, str(value)])
    args = parser().parse_args(cli)
    assert all(getattr(args, key) == value for key, value in options.items())


@pytest.mark.parametrize('frames', [1, 2, 5, 6])
@pytest.mark.parametrize('variational', [False, True])
def test_frame_aligned_latent_shapes_sampling_gradients_and_checkpoint(frames, variational):
    from model.motion_autoencoder import diagonal_gaussian_kl
    model, x, s = fixture(frames=frames, variational=variational,
                          temporal_rope=True, layerwise_latent=True)
    frame = torch.ones(2, frames, dtype=torch.bool)
    loop = dict(is_loop=torch.tensor([False, True]), loop_tile_count=torch.ones(2))
    model.train()
    prediction, mu, logvar = model(x, s, frame, loop, return_posterior=True)
    assert mu.shape == (2, frames, 2, 4)
    assert prediction.shape == x.shape
    loss = prediction.square().mean()
    if variational:
        loss = loss + .01 * diagonal_gaussian_kl(mu, logvar, frame)
    loss.backward()
    assert model.to_latent.weight.grad.abs().sum() > 0
    assert torch.isfinite(model.to_latent.weight.grad).all()
    model.eval()
    restored = MotionAutoencoder(**model.config).eval()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(x, s, frame, loop), model(x, s, frame, loop))


def test_frame_aligned_padding_and_mixed_valid_lengths_match_individual_clips():
    model, x, s = fixture(frames=7, variational=True,
                          temporal_rope=True, layerwise_latent=True)
    frame = torch.tensor([[True] * 5 + [False] * 2, [True] * 7])
    loop = dict(is_loop=torch.tensor([True, False]), loop_tile_count=torch.ones(2))
    loop['resample_speed_cond'] = torch.tensor([.5, 1.5])
    # Exercise nonzero time heads, rather than only their no-op initialization.
    torch.nn.init.normal_(model.tempo_projection[-1].weight, std=.1)
    torch.nn.init.normal_(model.loop_period_projection[-1].weight, std=.1)
    expected = model(x, s, frame, loop)
    for row, length in enumerate([5, 7]):
        single_s = {key: value[row:row + 1] for key, value in s.items()}
        single_loop = {key: value[row:row + 1] for key, value in loop.items()}
        actual = model(x[row:row + 1, :, :, :length], single_s, loop_metadata=single_loop)
        torch.testing.assert_close(actual[0], expected[row, :, :, :length], rtol=1e-4, atol=2e-6)
    padded = torch.nn.functional.pad(x, (0, 4), value=123.)
    padded_frame = torch.nn.functional.pad(frame, (0, 4), value=False)
    actual = model(padded, s, padded_frame, loop)
    torch.testing.assert_close(actual[..., :7], expected, rtol=1e-4, atol=2e-6)
    assert actual[..., 7:].count_nonzero() == 0
    sampled = model.encode(padded, s, padded_frame, loop, sample=True)
    assert sampled[~padded_frame].count_nonzero() == 0


def test_frame_aligned_fullgraph_training_and_exact_odd_decode_length():
    model, x, s = fixture(variational=True, layerwise_latent=True)
    model.train()
    compiled = torch.compile(model, backend='eager', fullgraph=True)
    first, mu, logvar = compiled(x, s, return_posterior=True)
    second = compiled(x, s)
    assert first.shape == second.shape == x.shape
    assert not torch.equal(first, second)
    frame = torch.ones(2, 5, dtype=torch.bool)
    assert model.decode(mu, s, frame).shape == x.shape
    with pytest.raises(ValueError, match='Latent time length'):
        model.decode(mu, s, torch.ones(2, 3, dtype=torch.bool))


def test_temporal_stride_is_removed():
    from train.train_autoencoder import parser
    args = parser().parse_args(['--save_dir', 'unused'])
    assert not hasattr(args, 'temporal_stride')
    with pytest.raises(SystemExit):
        parser().parse_args(['--save_dir', 'unused', '--temporal_stride', '1'])
    with pytest.raises(TypeError):
        fixture(temporal_stride=1)
    model, x, skeleton = fixture(variational=True, layerwise_latent=True)
    assert 'temporal_stride' not in model.config


@pytest.mark.parametrize('joints,frames', [(1, 1), (4, 5), (7, 2)])
@pytest.mark.parametrize('kernel', [0, 5])
def test_complete_channels_latent_and_gradients(joints, frames, kernel):
    model, x, s = fixture(joints, frames, decoder_temporal_kernel=kernel)
    x.requires_grad_()
    z = model.encode(x, s)
    assert z.shape == (2, frames, 2, 4)
    pred = model.decode(z, s)
    assert pred.shape == x.shape
    pred.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad[:, 0, 3:9].abs().sum() > 0
    working = to_working_motion(pred)
    assert working.shape[2] == 12
    assert working[:, :, 9:].count_nonzero() == 0


@pytest.mark.parametrize('loop', [False, True])
@pytest.mark.parametrize('attention', [False, True])
@pytest.mark.parametrize('kernel', [0, 5])
def test_joint_and_time_padding_do_not_change_valid_output(loop, attention, kernel):
    options = dict(temporal_rope=True, cross_limb_dim=8, cross_limb_last_n=1,
                   layerwise_latent=True, ff_size=32, last_layer_ff=16) if attention else {}
    model, x, s = fixture(**options, decoder_temporal_kernel=kernel)
    if kernel:
        # Exercise a learned, non-identity residual, including boundary padding.
        torch.nn.init.normal_(model.temporal_refinement.mix.weight, std=.05)
    frame = torch.ones(2, 5, dtype=torch.bool)
    metadata = dict(is_loop=torch.full((2,), loop), loop_tile_count=torch.ones(2))
    expected = model(x, s, frame, metadata)
    padded_x = torch.nn.functional.pad(x, (0, 3, 0, 0, 0, 2), value=123)
    padded_s = dict(s)
    for key in ('joints_names_embs', 'rest_geom', 'joint_struct'):
        padded_s[key] = torch.nn.functional.pad(s[key], (0, 0, 0, 2), value=321)
    for key in ('graph_dist', 'joints_relations'):
        padded_s[key] = torch.nn.functional.pad(s[key], (0, 2, 0, 2))
    padded_frame = torch.nn.functional.pad(frame, (0, 3), value=False)
    actual = model(padded_x, padded_s, padded_frame, metadata)
    torch.testing.assert_close(actual[:, :4, :, :5], expected, rtol=1e-4, atol=2e-6)
    assert actual[:, :, :, 5:].count_nonzero() == 0
    assert actual[:, 4:].count_nonzero() == 0
    loss = reconstruction_mse(actual, padded_x, padded_s, padded_frame)
    torch.testing.assert_close(loss, reconstruction_mse(expected, x, s, frame))


def test_decoder_ignores_dynamic_condition_fields_and_reads_latent():
    model, x, s = fixture()
    z = model.encode(x, s)
    expected = model.decode(z, s)
    contaminated = dict(s, ref_feats=x * 123, ref_pose=x * -10,
                        action_label=['shake', 'idle'], target_motion=x)
    torch.testing.assert_close(model.decode(z, contaminated), expected, rtol=0, atol=0)
    assert not torch.allclose(model.decode(torch.zeros_like(z), s), expected)
    assert not torch.allclose(model.decode(z.roll(1, 0), s), expected)


@pytest.mark.parametrize('rope', [False, True])
def test_decoder_zero_init_tempo_preserves_constant_latent_and_ignores_phase(monkeypatch, rope):
    model, x, skeleton = fixture(temporal_rope=rope,
                                 layerwise_latent=True)
    frame = torch.ones(2, x.shape[-1], dtype=torch.bool)
    loop = dict(is_loop=torch.tensor([False, True]), loop_tile_count=torch.ones(2))
    # Exercise a nonzero phase scale: the zero initialization would hide a leak.
    with torch.no_grad():
        model.decoder.layers[0].temporal_phase_scale.fill_(1.)
    z = model.encode(x, skeleton, frame, loop)
    changed_loop = dict(loop, loop_tile_count=torch.full((2,), 3.))
    assert not torch.allclose(model.encode(x, skeleton, frame, changed_loop)[1], z[1])

    def unexpected_position(*args):
        raise AssertionError('Decoder must not construct absolute time or phase content')

    monkeypatch.setattr(model, '_position', unexpected_position)
    prediction = model.decode(z, skeleton, frame, loop)
    torch.testing.assert_close(model.decode(z, skeleton, frame, changed_loop), prediction,
                               rtol=0, atol=0)
    constant_z = z[:, :1].expand_as(z)
    constant = model.decode(constant_z, skeleton, frame, loop)
    torch.testing.assert_close(constant, constant[..., :1].expand_as(constant), rtol=1e-4, atol=2e-6)
    assert not torch.allclose(constant, prediction)
    restored = MotionAutoencoder(**model.config).eval()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored.decode(z, skeleton, frame, loop), prediction)


def test_decoder_time_option_is_removed():
    from train.train_autoencoder import parser
    args = parser().parse_args(['--save_dir', 'unused'])
    assert not hasattr(args, 'decoder_time_injection')
    model, x, skeleton = fixture(temporal_rope=True, layerwise_latent=True)
    assert 'decoder_time_injection' not in model.config
    for option in ('--decoder_time_injection', '--no-decoder_time_injection'):
        with pytest.raises(SystemExit):
            parser().parse_args(['--save_dir', 'unused', option])
    with pytest.raises(TypeError):
        fixture(decoder_time_injection=False)


def test_masked_loss_is_per_sample_and_ignores_padding():
    model, target, s = fixture()
    frame = torch.tensor([[True] * 5, [True, True, False, False, False]])
    prediction = target + 1
    prediction[1, 3] = 1000
    prediction[1, :, :, 2:] = 1000
    torch.testing.assert_close(reconstruction_mse(prediction, target, s, frame), torch.ones(2))


def test_layerwise_latent_reader_has_gradients_and_no_dynamic_bypass():
    model, x, s = fixture(temporal_rope=True, cross_limb_dim=8, cross_limb_last_n=1,
                          layerwise_latent=True)
    z = model.encode(x, s).detach().requires_grad_()
    pred = model.decode(z, s)
    pred.square().mean().backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
    assert model.decoder.layers[0].motion_latent_attn.q_proj_weight.grad.abs().sum() > 0
    torch.testing.assert_close(model.decode(z, dict(s, target_motion=x * 100)), pred)
    assert not torch.allclose(model.decode(torch.zeros_like(z), s), pred)


def test_attention_autoencoder_compiles_as_one_graph():
    model, x, s = fixture(temporal_rope=True, cross_limb_dim=8, cross_limb_last_n=1,
                          layerwise_latent=True, decoder_temporal_kernel=5)
    metadata = dict(is_loop=torch.tensor([True, False]), loop_tile_count=torch.ones(2),
                    resample_speed_cond=torch.tensor([.5, 1.5]))
    compiled = torch.compile(model, backend='eager', fullgraph=True)
    actual = compiled(x, s, loop_metadata=metadata)
    torch.testing.assert_close(actual, model(x, s, loop_metadata=metadata))
    actual.square().mean().backward()
    assert torch.isfinite(model.output.weight.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA AMP required')
def test_amp_unscales_before_clipping_and_preserves_weights_on_overflow():
    model = torch.nn.Linear(2, 1, bias=False).cuda()
    torch.nn.init.ones_(model.weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    scaler = torch.amp.GradScaler('cuda', init_scale=8.)
    x = torch.full((1, 2), 2., device='cuda')
    with torch.autocast('cuda', dtype=torch.float16):
        loss = model(x).float().square().mean()
    norm, updated = scaled_optimizer_step(loss, model, optimizer, scaler)
    assert updated
    torch.testing.assert_close(norm, torch.tensor(512., device='cuda').sqrt())
    expected = torch.full_like(model.weight, 1 - .1 / (2 ** .5))
    torch.testing.assert_close(model.weight, expected)
    previous = model.weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    _, updated = scaled_optimizer_step(model(x).sum() * float('inf'), model, optimizer, scaler)
    assert not updated and scaler.get_scale() == 4.
    torch.testing.assert_close(model.weight, previous, rtol=0, atol=0)
    restored = torch.amp.GradScaler('cuda')
    restored.load_state_dict(scaler.state_dict())
    assert restored.get_scale() == scaler.get_scale()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA AMP required')
def test_autoencoder_fp16_forward_backward_keeps_fp32_parameters():
    model, x, s = fixture(temporal_rope=True, layerwise_latent=True, decoder_temporal_kernel=5)
    model = model.cuda().train()
    x, s = x.cuda(), {k: v.cuda() for k, v in s.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda', init_scale=1.)
    with torch.autocast('cuda', dtype=torch.float16):
        prediction = model(x, s)
    assert prediction.dtype == torch.float16
    loss = (prediction.float() - x).square().mean()
    norm, updated = scaled_optimizer_step(loss, model, optimizer, scaler)
    assert updated and torch.isfinite(norm)
    assert all(p.dtype == torch.float32 and torch.isfinite(p).all() for p in model.parameters())


def test_temporal_residual_starts_identical_to_baseline_and_learns():
    baseline, x, s = fixture()
    model, _, _ = fixture(decoder_temporal_kernel=5)
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    expected = baseline(x, s)
    actual = model(x, s)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    assert model.temporal_refinement.mix.weight.grad.abs().sum() > 0


def test_temporal_convolution_wraps_only_loop_rows_and_does_not_mix_joints():
    block = TemporalResidualConv(2, 5)
    with torch.no_grad():
        block.depthwise.weight.fill_(1)
        block.depthwise.bias.zero_()
        block.mix.weight.copy_(torch.eye(2).reshape(2, 2, 1))
        block.mix.bias.zero_()
    tokens = torch.zeros(7, 2, 2, 2)
    tokens[-1, :, 0] = torch.tensor([1., -1.])
    actual = block(tokens, torch.ones(2, 7, dtype=torch.bool),
                   dict(is_loop=torch.tensor([True, False])))
    assert actual[0, 0, 0].abs().sum() > 0
    assert actual[0, 1, 0].count_nonzero() == 0
    assert actual[:, :, 1].count_nonzero() == 0


def test_prefetch_transform_filters_reference_and_keeps_nine_channels():
    import numpy as np
    from train.train_autoencoder import AutoencoderBatchTransform, training_batch_to_device
    motion = torch.randn(2, 7, 12, 5)
    y = dict(motion_name=['b', 'a'], ref_unused=torch.ones(1),
             joints_names_embs=torch.ones(2, 7, 6))
    transform = AutoencoderBatchTransform(joint_name_drop_prob=.5)
    np.random.seed(40)
    first = transform((motion, {'y': y}))
    np.random.seed(40)
    again = transform((motion, {'y': y}))
    target, cond = training_batch_to_device(first, 'cpu')
    torch.testing.assert_close(target, motion[:, :, :9], rtol=0, atol=0)
    assert target.is_contiguous()
    assert cond['motion_name'] == ['b', 'a'] and 'ref_unused' not in cond
    torch.testing.assert_close(cond['joints_names_embs'], again[1]['y']['joints_names_embs'])
    assert 'ref_unused' in y and y['joints_names_embs'].eq(1).all()


def test_vae_kl_matches_gaussian_formula_and_ignores_padding():
    from model.motion_autoencoder import diagonal_gaussian_kl
    mu = torch.randn(2, 4, 2, 3, requires_grad=True)
    logvar = torch.randn_like(mu, requires_grad=True)
    frame = torch.tensor([[True, True, False, False], [True, True, True, False]])
    normal = torch.distributions.Normal(mu, (logvar * .5).exp())
    expected = torch.distributions.kl_divergence(normal, torch.distributions.Normal(0., 1.))
    expected = torch.stack([expected[i, frame[i]].mean() for i in range(2)]).mean()
    actual = diagonal_gaussian_kl(mu, logvar, frame)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert mu.grad[~frame].count_nonzero() == logvar.grad[~frame].count_nonzero() == 0
    assert diagonal_gaussian_kl(torch.zeros_like(mu), torch.zeros_like(mu), frame) == 0


def test_vae_sampling_mean_eval_and_posterior_gradients():
    from model.motion_autoencoder import diagonal_gaussian_kl
    model, x, skeleton = fixture(variational=True, layerwise_latent=True)
    frame = torch.tensor([[True, True, True, False, False], [True] * 5])
    mu, logvar = model.encode_posterior(x, skeleton, frame)
    torch.testing.assert_close(model.encode(x, skeleton, frame), mu)
    sampled = model.encode(x, skeleton, frame, sample=True)
    assert not torch.equal(sampled, mu)
    assert sampled[~frame].count_nonzero() == 0
    first, second = model(x, skeleton, frame), model(x, skeleton, frame)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    model.train()
    first, mean, variance = model(x, skeleton, frame, return_posterior=True)
    second = model(x, skeleton, frame)
    assert not torch.equal(first, second)
    (first.square().mean() + .01 * diagonal_gaussian_kl(mean, variance, frame)).backward()
    for layer in (model.to_latent, model.to_logvar):
        assert layer.weight.grad.abs().sum() > 0 and torch.isfinite(layer.weight.grad).all()


def test_vae_fullgraph_forward_has_fresh_samples():
    model, x, skeleton = fixture(variational=True)
    model.train()
    compiled = torch.compile(model, backend='eager', fullgraph=True)
    first, mu, logvar = compiled(x, skeleton, return_posterior=True)
    second, mu2, logvar2 = compiled(x, skeleton, return_posterior=True)
    assert not torch.equal(first, second)
    torch.testing.assert_close(mu, mu2)
    torch.testing.assert_close(logvar, logvar2)



@pytest.mark.parametrize('prob', [0.0, 0.5, 1.0])
def test_training_name_drop_is_independent_reproducible_and_keeps_clean_inputs(prob):
    import numpy as np
    import train.train_autoencoder as trainer
    _, target, skeleton = fixture()
    clean_names = skeleton['joints_names_embs'].clone()
    conditions = dict(skeleton, lengths=torch.full((2,), target.shape[-1]),
                      is_loop=torch.tensor([False, True]), loop_tile_count=torch.ones(2),
                      resample_speed_cond=torch.tensor([.5, 1.5]))

    transform = trainer.AutoencoderBatchTransform(joint_name_drop_prob=prob)
    np.random.seed(20)
    x, wrapped = transform((target, {'y': conditions}))
    cond = wrapped['y']
    np.random.seed(20)
    _, again = transform((target, {'y': conditions}))
    again = again['y']
    torch.testing.assert_close(x, target, rtol=0, atol=0)
    torch.testing.assert_close(skeleton['joints_names_embs'], clean_names, rtol=0, atol=0)
    torch.testing.assert_close(cond['joints_names_embs'], again['joints_names_embs'], rtol=0, atol=0)
    dropped = cond['ae_source_skeleton']['joints_names_embs'].eq(0).all(-1)
    if prob == 0:
        assert not dropped.any()
    elif prob == 1:
        assert dropped.all()
    else:
        assert dropped.any() and not dropped.all()
    torch.testing.assert_close(cond['ae_source_skeleton']['joints_names_embs'][~dropped], clean_names[~dropped])
    torch.testing.assert_close(cond['joints_names_embs'], clean_names)
    for side in ('ae_source_skeleton', 'ae_target_skeleton'):
        torch.testing.assert_close(cond[side]['joints_names_embs'], again[side]['joints_names_embs'])
    if prob == .5:
        assert not torch.equal(cond['ae_source_skeleton']['joints_names_embs'], cond['ae_target_skeleton']['joints_names_embs'])
    # Time and loss contexts remain clean while E/D have separate name masks.
    shared, _, loop = trainer.model_inputs(x, cond)
    assert shared['joints_names_embs'] is cond['joints_names_embs']
    assert loop['is_loop'].tolist() == [False, True]
    assert loop['resample_speed_cond'] is conditions['resample_speed_cond']


@pytest.mark.parametrize('joint_prob', [0., .5, 1.])
def test_geometry_drop_preserves_clean_context_and_replays(joint_prob):
    import numpy as np
    import train.train_autoencoder as trainer
    _, target, skeleton = fixture(joints=32)
    geometry = torch.ones_like(skeleton['rest_geom'])
    offsets = torch.randn(2, 32, 3)
    conditions = dict(skeleton, rest_geom=geometry, offsets=offsets,
                      lengths=torch.full((2,), target.shape[-1]), is_loop=torch.zeros(2, dtype=torch.bool))

    transform = trainer.AutoencoderBatchTransform(
        rest_geom_joint_drop_prob=joint_prob)
    np.random.seed(50)
    x, wrapped = transform((target, {'y': conditions}))
    cond = wrapped['y']
    np.random.seed(50)
    _, again = transform((target, {'y': conditions}))
    again = again['y']
    torch.testing.assert_close(cond['rest_geom'], again['rest_geom'], rtol=0, atol=0)
    assert geometry.eq(1).all()  # including the original present flag
    assert cond['offsets'] is offsets
    torch.testing.assert_close(x, target, rtol=0, atol=0)
    dropped = cond['ae_source_skeleton']['rest_geom'].eq(0).all(-1)
    if joint_prob == 1:
        assert dropped.all()
    elif joint_prob == 0:
        assert not dropped.any()
    else:
        assert dropped.any() and not dropped.all()
    assert cond['ae_source_skeleton']['rest_geom'][~dropped].eq(1).all()
    assert cond['rest_geom'].eq(1).all() and cond['ae_target_skeleton']['rest_geom'].eq(1).all()


def test_regularized_autoencoder_compiles_trains_and_has_deterministic_eval():
    options = dict(dropout_prob=.1,
                   temporal_rope=True, layerwise_latent=True)
    model, x, skeleton = fixture(**options)
    for trunk in (model.encoder, model.decoder):
        assert trunk.layers[0].dropout1.p == .1
        assert trunk.layers[0].temporal_attn.dropout == .1
    assert model.pool.dropout == model.read_latent.dropout == .1
    assert all(model.config[key] == value for key, value in options.items())
    model.train()
    compiled = torch.compile(model, backend='eager', fullgraph=True)
    first = compiled(x, skeleton)
    second = compiled(x, skeleton)
    assert not torch.equal(first, second)
    first.square().mean().backward()
    assert torch.isfinite(model.output.weight.grad).all()
    model.eval()
    torch.testing.assert_close(model(x, skeleton), model(x, skeleton), rtol=0, atol=0)
    restored = MotionAutoencoder(**model.config).eval()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(x, skeleton), model(x, skeleton), rtol=0, atol=0)


@pytest.mark.parametrize('key', ['dropout_prob'])
@pytest.mark.parametrize('value', [-.1, 1.1, float('nan')])
def test_regularization_rejects_invalid_probabilities(key, value):
    with pytest.raises(ValueError, match=key):
        fixture(**{key: value})
