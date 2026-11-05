"""Global coordinates and offline decoder metric/cache regression tests."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_latent_diffusion import conditions
from utils.latent_util import LatentNormalizer, estimate_normalizer, encode_target
from utils.latent_analysis import (diagonal_sensitivity, load_or_analyze, sensitivity_features,
                                   sensitivity_loss_weights, validate_sensitivity_loss)


class LinearAE:
    def encode_posterior(self, x, skeleton, loop_metadata=None):
        return x.permute(0, 3, 1, 2), None

    def encode(self, x, skeleton, loop_metadata=None, sample=False):
        return self.encode_posterior(x, skeleton, loop_metadata)[0]

    def decode(self, z, skeleton, loop_metadata=None):
        gains = torch.tensor([0., 0., 0., 1., 2., 3., 4., 5., 6.], device=z.device)
        return (z * gains).permute(0, 2, 3, 1)


def batch():
    x = torch.zeros(2, 1, 9, 60)
    x[0] = torch.arange(60) / 60
    x[1] = 10 + 2 * torch.arange(60) / 60
    y = dict(conditions(batch=2, joints=2), n_joints=torch.ones(2, dtype=torch.long),
             object_subset=['quadruped', 'winged'])
    return x, {'y': y}


def test_global_moments_roundtrip_and_subset_labels_cannot_change_coordinates():
    x, cond = batch()
    norm = estimate_normalizer(LinearAE(), iter([(x, cond)]), 1, 'cpu')
    z = x.permute(0, 3, 1, 2)
    torch.testing.assert_close(norm.mean, z.mean((0, 1)))
    torch.testing.assert_close(norm.std, z.std((0, 1), correction=0))
    u = encode_target(LinearAE(), norm, x, cond['y'])
    torch.testing.assert_close(u.mean((0, 1)), torch.zeros_like(norm.mean), atol=1e-5, rtol=0)
    restored = LatentNormalizer.from_metadata(norm.metadata())
    torch.testing.assert_close(restored.denormalize(u), z)
    unseen = dict(cond['y'], object_subset=['aquatic', 'unknown'])
    torch.testing.assert_close(encode_target(LinearAE(), restored, x, unseen), u)
    assert norm.metadata()['version'] == 1 and 'subsets' not in norm.metadata()
    with pytest.raises(ValueError, match='Subset normalization'):
        LatentNormalizer.from_metadata(dict(norm.metadata(), version=2))


def test_random_projection_metric_includes_standardization_and_excludes_padding():
    x, cond = batch()
    # Independent rotation coordinates have an exact diagonal estimate with any probe.
    norm = LatentNormalizer(torch.zeros(1, 9), torch.full((1, 9), 2.0))
    result = diagonal_sensitivity(LinearAE(), norm, x, cond['y'], probes=2)
    gains = torch.tensor([0., 0., 0., 1., 2., 3., 4., 5., 6.])
    expected = 4 * gains.square() / (6 * 60)
    torch.testing.assert_close(result, expected[None, None].expand_as(result))
    padded = torch.randn(2, 3, 9, 60)
    padded[:, :1] = LinearAE().decode(x.permute(0, 3, 1, 2), None)
    features = sensitivity_features(padded, torch.ones(2, dtype=torch.long))
    padded[:, 1:] = 10000
    torch.testing.assert_close(features, sensitivity_features(padded, torch.ones(2, dtype=torch.long)))


def test_analysis_cache_reuse_invalidation_and_rng_isolation(tmp_path):
    calls = []
    def factory():
        calls.append(True)
        return iter([batch()])

    binding = dict(path=str(tmp_path / 'autoencoder.pt'), sha256='ae-a', weights='ema')
    kwargs = dict(ae=LinearAE(), binding=binding, identity={'clips': 'a'},
                  batch_factory=factory, batch_count=1, device='cpu', sample_count=1, probes=2)
    state = torch.get_rng_state().clone()
    norm, loss, reference = load_or_analyze(**kwargs)
    torch.testing.assert_close(torch.get_rng_state(), state)
    assert len(calls) == 1
    cached_norm, cached_loss, cached_reference = load_or_analyze(**kwargs)
    assert cached_reference == reference and len(calls) == 1
    torch.testing.assert_close(cached_norm.mean, norm.mean)
    torch.testing.assert_close(cached_loss['weights'], loss['weights'])
    report = torch.load(reference['path'], weights_only=False)
    assert report['sample_count'] == 1 and report['reference_count'] == 2
    assert torch.is_tensor(report['sensitivity']) and report['sensitivity'].shape == norm.mean.shape
    assert report['contract']['sampling'] == 'uniform_global_reservoir_v1'
    assert report['applied_to_training_loss']
    torch.testing.assert_close(validate_sensitivity_loss(loss, norm.mean.shape).mean(), torch.tensor(1.))
    _, _, changed = load_or_analyze(**dict(kwargs, identity={'clips': 'b'}))
    assert changed['fingerprint'] != reference['fingerprint'] and len(calls) == 2
    _, _, changed_ae = load_or_analyze(**dict(kwargs, binding=dict(binding, sha256='ae-b')))
    assert changed_ae['fingerprint'] != reference['fingerprint'] and len(calls) == 3


@pytest.mark.parametrize('mode', ['mean', 'sample'])
def test_stats_and_sensitivity_share_encoder_pass(tmp_path, mode):
    class CountingAE(LinearAE):
        calls = 0

        def encode_posterior(self, x, skeleton, loop_metadata=None):
            self.calls += 1
            mu, _ = super().encode_posterior(x, skeleton, loop_metadata)
            return mu, torch.zeros_like(mu) if mode == 'sample' else None

        def encode(self, *args, **kwargs):
            pytest.fail('Sensitivity must reuse the statistics pass posterior')

    ae = CountingAE()
    norm, loss, reference = load_or_analyze(
        ae, dict(path=str(tmp_path / 'ae.pt'), sha256='shared-encoder'), {},
        lambda: iter([batch()]), batch_count=3, device='cpu', mode=mode,
        sample_count=2, probes=2)
    assert ae.calls == 3  # Exactly one encoding per reference batch.
    assert norm.count == 360
    report = torch.load(reference['path'], weights_only=False)
    assert report['sample_count'] == 2 and report['reference_count'] == 6
    gains = torch.tensor([0., 0., 0., 1., 2., 3., 4., 5., 6.])
    expected = norm.std.square() * gains.square() / (6 * 60)
    torch.testing.assert_close(report['sensitivity'], expected)
    torch.testing.assert_close(loss['sensitivity'], expected)


def test_legacy_normalizer_remains_global_even_with_subset_conditions():
    norm = LatentNormalizer(torch.zeros(1, 9), torch.ones(1, 9))
    assert norm.metadata()['version'] == 1
    restored = LatentNormalizer.from_metadata(norm.metadata())
    x, cond = batch()
    z = x.permute(0, 3, 1, 2)
    torch.testing.assert_close(restored.normalize(z), z)


def test_compiled_encoder_uses_global_coordinates_for_all_subsets(monkeypatch):
    from train import train_latent_diffusion as training
    from utils import compile_util
    from test_latent_diffusion import denoiser
    from model.motion_autoencoder import MotionAutoencoder
    real_compile = torch.compile
    monkeypatch.setattr(torch, 'compile', lambda function, **kwargs:
                        real_compile(function, backend='eager', fullgraph=True))
    monkeypatch.setattr(compile_util, 'configure_triton_c_compiler', lambda: 'test')
    ae = MotionAutoencoder(t5_out_dim=4, width=16, layers=1, heads=2,
                           slots=2, latent_dim=3).eval().requires_grad_(False)
    norm = LatentNormalizer(torch.ones(2, 3), torch.full((2, 3), 2.))
    x = torch.randn(2, 5, 9, 60)
    y = dict(conditions(), object_subset=['quadruped', 'winged'])
    _, encoder = training.training_forwards(denoiser(), ae, norm, 'default')
    first = encoder(x, y)
    torch.testing.assert_close(first, encode_target(ae, norm, x, y))
    y['object_subset'].reverse()
    torch.testing.assert_close(encoder(x, y), first)


def test_global_sensitivity_is_the_mean_over_samples_without_group_balancing(monkeypatch, tmp_path):
    from utils import latent_analysis as analysis
    from train.train_autoencoder import select_batch
    x, conditions = batch()
    # Three identical motions and one different motion, all in the same reservoir.
    x, y = select_batch(x, conditions['y'], torch.tensor([0, 0, 0, 1]))
    def sensitivity(ae, norm, motion, cond, probes, latent):
        value = torch.arange(1., 10.) if latent[0, 0, 0, 0] < 5 else torch.arange(9., 0., -1.)
        return value[None, None]
    monkeypatch.setattr(analysis, 'diagonal_sensitivity', sensitivity)
    _, loss, _ = load_or_analyze(
        LinearAE(), dict(path=str(tmp_path / 'ae.pt')), {},
        lambda: iter([(x, {'y': y})]), 1, 'cpu', sample_count=4, probes=1)
    expected = (.75 * torch.arange(1., 10.) + .25 * torch.arange(9., 0., -1.))[None]
    torch.testing.assert_close(loss['sensitivity'], expected)
    torch.testing.assert_close(loss['weights'], sensitivity_loss_weights(expected))
    wrong = dict(loss, weights=torch.ones_like(loss['weights']))
    with pytest.raises(ValueError, match='loss contract'):
        validate_sensitivity_loss(wrong, expected.shape)
    with pytest.raises(ValueError, match='new save_dir'):
        validate_sensitivity_loss(None, expected.shape)


def test_global_reservoir_is_bounded_and_does_not_read_subset_labels(monkeypatch, tmp_path):
    from utils import latent_analysis as analysis
    x, cond = batch()
    calls = []
    def sensitivity(ae, norm, motion, conditions, probes, latent):
        calls.append(latent.clone())
        return (latent[0, :1] + 1).abs()
    monkeypatch.setattr(analysis, 'diagonal_sensitivity', sensitivity)
    # Reject later reservoir candidates, retaining exactly the first sample.
    monkeypatch.setattr(analysis.random, 'randrange', lambda seen: seen - 1)
    kwargs = dict(ae=LinearAE(), binding=dict(path=str(tmp_path / 'ae.pt')),
                  batch_factory=lambda: iter([(x, cond)]), batch_count=3,
                  device='cpu', sample_count=1, probes=1)
    _, loss, reference = load_or_analyze(identity={'data': 'a'}, **kwargs)
    assert len(calls) == 1
    torch.testing.assert_close(calls[0], x[:1].permute(0, 3, 1, 2))
    report = torch.load(reference['path'], weights_only=False)
    assert report['sample_count'] == 1 and report['reference_count'] == 6
    cond['y']['object_subset'] = ['new', 'completely-different']
    _, changed, _ = load_or_analyze(identity={'data': 'b'}, **kwargs)
    torch.testing.assert_close(changed['sensitivity'], loss['sensitivity'])
    torch.testing.assert_close(changed['weights'], loss['weights'])


def test_sensitivity_weights_are_scale_invariant_and_bound_gradient_anisotropy():
    sensitivity = torch.tensor([[0., 1., 2., 1000.]])
    weights = sensitivity_loss_weights(sensitivity)
    for scale in (1e-15, 1e15):
        torch.testing.assert_close(sensitivity_loss_weights(sensitivity * scale), weights)
    torch.testing.assert_close(weights.mean(), torch.tensor(1.))
    assert weights.max() / weights.min() <= 4
    for invalid in (torch.zeros_like(sensitivity), sensitivity - 1,
                    torch.full_like(sensitivity, float('nan'))):
        with pytest.raises(ValueError, match='Decoder sensitivity'):
            sensitivity_loss_weights(invalid)


def test_reference_loader_is_clean_sequential_and_accepts_dataset_contract(monkeypatch):
    import inspect
    from types import SimpleNamespace
    from train import train_latent_diffusion as training
    signature = inspect.signature(training.get_dataset)
    captured = {}
    dataset = SimpleNamespace(motion_dataset=SimpleNamespace(use_weighted_sampler=True))
    def get_dataset(**kwargs):
        signature.bind(**kwargs)
        captured.update(kwargs)
        return dataset
    def loader(data, **kwargs):
        assert isinstance(data, training.AnalysisReferenceDataset)
        assert data.motion is dataset.motion_dataset
        assert kwargs['shuffle'] is kwargs['drop_last'] is False
        assert kwargs['num_workers'] == 0
        return [batch()]
    monkeypatch.setattr(training, 'get_dataset', get_dataset)
    monkeypatch.setattr(torch.utils.data, 'DataLoader', loader)
    args = training.parser().parse_args(['--save_dir', 'unused'])
    args.action_conditioning = args.species_descriptor_table = None
    x, cond = next(iter(training.build_analysis_loader(args)()))
    assert x.shape == (2, 1, 9, 60)
    assert not dataset.motion_dataset.use_weighted_sampler
    assert captured['motion_speed_aug'] == 1.
    assert captured['loop_tile_single_prob'] == 1.
    assert captured['topology_aug_prob'] == captured['rest_variant_prob'] == 0.


def test_reference_dataset_preserves_clip_order_and_fixes_loop_phase():
    from types import SimpleNamespace
    from train.train_latent_diffusion import AnalysisReferenceDataset

    class Motion:
        name_list = ['excluded', 'first', 'second']
        pointer = 1

        def __len__(self):
            return 2

        def prepare_sample_by_name(self, name, loop_offset=None):
            return name, loop_offset

    reference = AnalysisReferenceDataset(SimpleNamespace(motion_dataset=Motion()))
    assert len(reference) == 2
    assert [reference[i] for i in range(len(reference))] == [('first', 0), ('second', 0)]


def test_reference_identity_invalidates_high_frequency_changes():
    from types import SimpleNamespace
    from train.train_latent_diffusion import analysis_reference_identity

    entry = dict(motion_metadata={'is_loop': True}, high_frequency=False)
    dataset = SimpleNamespace(motion_dataset=SimpleNamespace(
        name_list=['clip'], data_dict={'clip': entry}))
    before = analysis_reference_identity(dataset, 'same-cond')
    entry['high_frequency'] = True
    after = analysis_reference_identity(dataset, 'same-cond')
    assert before['clips_sha256'] != after['clips_sha256']
