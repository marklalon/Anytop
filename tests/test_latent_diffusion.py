"""Latent contracts, normalization, condition priority and diffusion isolation."""
import copy
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loaders.reference import REST_GEOM_DIM
from data_loaders.truebones.truebones_utils.joint_struct_features import JOINT_STRUCT_DIM
from data_loaders.truebones.truebones_utils.dataset_tags import species_code_dim
from diffusion.latent_diffusion import create_latent_diffusion, validate_latent_conditions
from model.latent_motion_denoiser import LatentMotionDenoiser
from model.motion_autoencoder import MotionAutoencoder
from utils.latent_util import LatentNormalizer, estimate_normalizer, encode_target, ae_inputs
from train.train_latent_diffusion import LatentBatchTransform, parser
from action_label_test_utils import make_test_bundle, action_cond_fields


def conditions(batch=2, joints=5, names=4):
    return dict(n_joints=torch.tensor([joints - 1] * batch), lengths=torch.full((batch,), 60),
                joints_names_embs=torch.randn(batch, joints, names),
                rest_geom=torch.randn(batch, joints, REST_GEOM_DIM),
                joint_struct=torch.randn(batch, joints, JOINT_STRUCT_DIM),
                graph_dist=torch.zeros(batch, joints, joints, dtype=torch.long),
                joints_relations=torch.zeros(batch, joints, joints, dtype=torch.long),
                is_loop=torch.tensor([True] * batch), loop_tile_count=torch.full((batch,), 3),
                resample_speed_cond=torch.full((batch,), 1.5))


def denoiser(**kwargs):
    return LatentMotionDenoiser(t5_out_dim=4, slots=2, latent_channels=3,
                                width=16, layers=1, heads=2, ff_size=32, skeleton_layers=1,
                                action_label_cond=False, species_cond=False, dropout=0,
                                joint_name_drop_prob=0, rest_geom_joint_drop_prob=0, **kwargs)


def test_forward_backward_variable_skeleton_and_slot_graph_isolation():
    model = denoiser().train()
    z = torch.randn(2, 60, 2, 3)
    y = conditions()
    result = model(z, torch.tensor([1, 50]), y)
    assert result.shape == z.shape and torch.isfinite(result).all()
    result.square().mean().backward()
    assert model.input.weight.grad.abs().sum() > 0
    assert model.skeleton_encoder.graph_relations and not model.trunk.graph_relations
    assert model.trunk.cross_limb_blocks is None
    assert model.skeleton_encoder.cross_limb_blocks is None
    assert not any('reference' in key or 'ref_' in key for key in model.state_dict())


def test_padding_joints_cannot_change_valid_prediction():
    model = denoiser().eval()
    z = torch.randn(2, 60, 2, 3)
    y = conditions()
    changed = copy.deepcopy(y)
    for key in ('joints_names_embs', 'rest_geom', 'joint_struct'):
        changed[key][:, -1] = 1000
    torch.testing.assert_close(model(z, torch.zeros(2), y), model(z, torch.zeros(2), changed))


def test_loop_cycles_and_explicit_rope_override():
    model = denoiser().train()
    y = conditions()
    _, cycles = model._loop_condition(y, 2, torch.device('cpu'))
    assert cycles.tolist() == [3, 3]
    seen = []
    handle = model.trunk.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append(kwargs['temporal_rope_off']), with_kwargs=True)
    z, t = torch.randn(2, 60, 2, 3), torch.zeros(2)
    model(z, t, y)
    assert seen[-1] is None
    model.eval()(z, t, y)
    assert seen[-1] is None
    model(z, t, dict(y, temporal_rope_off=torch.tensor([True, False])))
    assert seen[-1].tolist() == [True, False]
    handle.remove()


def test_action_null_and_shared_adaln_token():
    model = LatentMotionDenoiser(t5_out_dim=768, width=144, layers=1, heads=4,
                                 ff_size=32, skeleton_layers=1, slots=2, latent_channels=3,
                                 action_conditioning=make_test_bundle(), species_cond=False,
                                 dropout=0).eval()
    model.action_label_null_emb.data.fill_(0.5)
    y = conditions(names=768)
    y.update(action_cond_fields(['shake', 'walk'], ['stationary', 'locomotion']))
    explicit = dict(y, action_label_active=torch.zeros(2, dtype=torch.bool))
    token = model._build_action_label_token(explicit, 2, torch.device('cpu'), torch.float32)
    torch.testing.assert_close(token, model.action_label_null_emb.expand(2, -1))
    seen = {}
    def hook(module, args, kwargs):
        seen['additive'] = args[1]
        seen['adaln'] = kwargs['action_adaln_cond']
    handle = model.trunk.register_forward_pre_hook(hook, with_kwargs=True)
    model(torch.randn(2, 60, 2, 3), torch.zeros(2), explicit)
    handle.remove()
    from model.motion_conditioning import create_sin_embedding
    time = create_sin_embedding(torch.zeros(1, 2, 1), 144)[0]
    torch.testing.assert_close(seen['additive'] - time, seen['adaln'])


@pytest.mark.parametrize('key', ['joint_mask', 'temporal_span_mask', 'renoise', 'inpaint_joints',
                                'inpaint_frames', 'outpaint', 'cross_limb_unreliable_mask',
                                'ref_frame_mask', 'ref_active', 'ref_feats'])
def test_edits_are_explicitly_rejected(key):
    with pytest.raises(ValueError, match='unsupported'):
        validate_latent_conditions({key: torch.zeros(1), 'lengths': torch.tensor([60])})


def test_fixed_window_and_shared_collator_transform():
    y = conditions()
    y['lengths'][0] = 59
    with pytest.raises((ValueError, RuntimeError), match='invalid frames'):
        validate_latent_conditions(y)
    y['lengths'].fill_(60)
    y['ref_frame_mask'] = torch.ones(2, 60, dtype=torch.bool)
    y['ref_feats'] = torch.full((2, 5, 12, 60), float('nan'))
    y['ref_active'] = torch.ones(2, dtype=torch.bool)
    motion, transformed = LatentBatchTransform()((torch.zeros(2, 5, 12, 60), {'y': y}))
    assert motion.shape == (2, 5, 9, 60) and motion.is_contiguous()
    assert not any(key.startswith('ref_') for key in transformed['y'])
    assert 'ref_feats' in y  # Do not mutate the shared input batch.
    validate_latent_conditions(transformed['y'])


@pytest.mark.parametrize('flag,value', [
    ('--ref_encoder', None), ('--ref_slots', '2'), ('--ref_dim', '8'),
    ('--ref_width', '64'), ('--ref_layers', '2'), ('--ref_cfg_drop_prob', '0.4'),
    ('--temporal_rope_ref_drop_prob', '0.5'),
    ('--temporal_rope_drop_prob', '0.5'),
])
def test_training_rejects_removed_reference_options(flag, value):
    with pytest.raises(SystemExit):
        parser().parse_args(['--save_dir', 'unused', flag] + ([] if value is None else [value]))


@pytest.mark.parametrize('flag', ['--reference_motion', '--reference_object_type', '--ref_cfg_scale'])
def test_generation_rejects_removed_reference_options(flag):
    from sample.generate_latent import parser as generation_parser
    with pytest.raises(SystemExit):
        generation_parser().parse_args(['--model_path', 'unused', '--output_dir', 'unused', flag, '1'])


def test_generation_without_reference_decodes_and_exports(monkeypatch, tmp_path):
    import json
    from sample import generate_latent as generation

    model = denoiser().eval()
    ae = MotionAutoencoder(t5_out_dim=4, width=16, layers=1, heads=2,
                           slots=2, latent_dim=3).eval()
    norm = LatentNormalizer(torch.zeros(2, 3), torch.ones(2, 3))
    checkpoint = dict(step=1, diffusion_config=dict(steps=10, noise_schedule='cosine', shift=1.0))
    cond = {'Target': {'parents': [-1, 0, 0, 0, 0], 'joints_names': ['root', 'a', 'b', 'c', 'd']}}
    monkeypatch.setattr(generation, 'load_latent_checkpoint', lambda *a: (model, ae, norm, checkpoint))
    monkeypatch.setattr(generation, 'load_cond', lambda *a: cond)
    monkeypatch.setattr(generation, 'mark_canonical_cond_entry', lambda *a: None)
    monkeypatch.setattr(generation, 'ensure_joint_name_embeddings', lambda *a, **k: None)
    monkeypatch.setattr(generation, '_resolve_species_code_override', lambda *a: None)
    monkeypatch.setattr(generation, 'loop_period', lambda *a, **k: None)
    y = conditions()
    y['ref_active'] = torch.zeros(2, dtype=torch.bool)
    y['ref_frame_mask'] = torch.zeros(2, 60, dtype=torch.bool)
    monkeypatch.setattr(generation, 'create_condition', lambda *a, **k: (None, {'y': y}))

    class Sampler:
        def sample(self, guided, shape, conditions, device, **kwargs):
            assert not any(key.startswith('ref_') for key in conditions)
            return guided(torch.zeros(shape, device=device), torch.zeros(shape[0]), conditions)

    monkeypatch.setattr(generation, 'create_latent_diffusion', lambda **k: Sampler())
    monkeypatch.setattr(generation, 'canonical_to_physical_hml', lambda motion, cond: motion)
    exported = []
    monkeypatch.setattr(generation, '_export_motion', exported.append)
    out = tmp_path / 'out'
    generation.main(['--model_path', str(tmp_path / 'latent.pt'), '--cond_path', str(tmp_path / 'cond.npy'),
                     '--output_dir', str(out), '--batch_size', '2', '--num_frames', '60',
                     '--loop', 'off', '--device', '-1', '--sampling_steps', '10'])
    assert len(exported) == 2
    assert exported[0][0].shape == (60, 5, 12)
    metadata = json.loads((out / 'generation.json').read_text())
    assert metadata['object_type'] == 'Target' and metadata['frames'] == 60
    assert 'reference_motion' not in metadata


def test_training_saves_loads_and_resumes_without_reference(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from train import train_latent_diffusion as training
    from utils.latent_util import load_latent_checkpoint, LATENT_CHECKPOINT_VERSION, TIME_CONTRACT
    from model.motion_autoencoder import DECODER_CONTENT_VERSION
    from data_loaders.truebones.truebones_utils.canonical_features import CANONICAL_FEATURE_SPACE

    ae = MotionAutoencoder(t5_out_dim=4, width=16, layers=1, heads=2, slots=2, latent_dim=3)
    ae_path = tmp_path / 'ae.pt'
    torch.save(dict(kind='motion_autoencoder_7a', model_config=ae.config, model=ae.state_dict(),
                    args=dict(num_frames=60, use_ema=False), canonical_version=CANONICAL_FEATURE_SPACE,
                    decoder_content_version=DECODER_CONTENT_VERSION), ae_path)
    cond_path = tmp_path / 'cond.npy'
    cond_path.write_bytes(b'skeleton snapshot')
    dataset = SimpleNamespace(motion_dataset=SimpleNamespace(name_list=['clip_a', 'clip_b']))
    batch = (torch.randn(2, 5, 12, 60), {'y': dict(
        conditions(), ref_active=torch.zeros(2, dtype=torch.bool),
        ref_frame_mask=torch.zeros(2, 60, dtype=torch.bool))})

    def get_dataset(**kwargs):
        assert not kwargs.get('ref_encoder', False)
        return dataset

    monkeypatch.setattr(training, 'get_dataset', get_dataset)
    monkeypatch.setattr(training, 'build_dataset_loader',
                        lambda dataset, **kwargs: [kwargs['batch_transform'](batch)])
    monkeypatch.setattr(training, 'build_validation_loader', lambda *a: None)
    # Stop after the first saved step to exercise a real resumed optimizer update.
    real_forwards = training.training_forwards

    def interrupted_forwards(*args):
        model, encoder = real_forwards(*args)
        calls = 0

        def encode(x, y):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('simulated interruption')
            return encoder(x, y)

        return model, encode

    monkeypatch.setattr(training, 'training_forwards', interrupted_forwards)
    save = tmp_path / 'run'
    argv = ['--ae_checkpoint', str(ae_path), '--cond_path', str(cond_path), '--save_dir', str(save),
            '--device', 'cpu', '--no-action_label_cond', '--no-species_cond',
            '--width', '16', '--layers', '1', '--heads', '2', '--ff_size', '32',
            '--skeleton_layers', '1', '--batch_size', '2', '--stats_batches', '1',
            '--num_steps', '2', '--save_interval', '1', '--diffusion_steps', '10']
    with pytest.raises(RuntimeError, match='simulated interruption'):
        training.main(argv)
    path = save / 'latent_diffusion.pt'
    assert torch.load(path, weights_only=False)['step'] == 1
    monkeypatch.setattr(training, 'training_forwards', real_forwards)
    training.main(argv + ['--auto_resume'])
    model, _, _, checkpoint = load_latent_checkpoint(path)
    assert checkpoint['step'] == 2 and checkpoint['kind'] == LATENT_CHECKPOINT_VERSION
    assert checkpoint['time_contract'] == TIME_CONTRACT
    assert checkpoint['capabilities'] == ['generate']
    assert 'reference_contract' not in checkpoint
    for config in (checkpoint['args'], checkpoint['model_config']):
        assert not any(key.startswith('ref_') or key in (
            'temporal_rope_ref_drop_prob', 'temporal_rope_drop_prob') for key in config)
    assert not any('reference' in key or 'ref_' in key for key in model.state_dict())


def test_latent_mse_respacing_and_sampling_without_clipping():
    diffusion = create_latent_diffusion(steps=10, respacing='3')
    class Constant(nn.Module):
        def __init__(self):
            super().__init__(); self.value = nn.Parameter(torch.tensor(4.0)); self.times = []
        def forward(self, x, t, y):
            self.times.append(t.detach().clone())
            return torch.ones_like(x) * self.value
    model = Constant()
    z = torch.zeros(2, 60, 2, 3)
    terms = diffusion.training_losses(model, z, torch.tensor([0, 2]), {'y': {'lengths': torch.tensor([60, 60])}})
    torch.testing.assert_close(terms['loss'], torch.tensor([16., 16.]))
    assert model.times[-1].tolist() == [0, 9]
    terms['loss'].mean().backward()
    torch.testing.assert_close(model.value.grad, torch.tensor(8.))
    result = diffusion.sample(model, z.shape, {'lengths': torch.tensor([60, 60])}, 'cpu', ddim=True)
    torch.testing.assert_close(result, torch.full_like(z, 4))
    with pytest.raises(ValueError, match='clipping'):
        diffusion.p_mean_variance(model, z, torch.tensor([0, 0]), clip_denoised=True)
    with pytest.raises(ValueError, match='Inpaint/outpaint'):
        diffusion._inpaint_project(z, 0, z.shape, 'cpu', torch.ones_like(z), z)


def test_normalization_invertibility_and_vae_variance_contribution():
    class FakeAE:
        def encode_posterior(self, x, skeleton, loop_metadata):
            return torch.full((len(x), 60, 2, 3), 7.), torch.full((len(x), 60, 2, 3), torch.log(torch.tensor(4.)))
    batch = (torch.zeros(2, 5, 9, 60), {'y': conditions()})
    mean = estimate_normalizer(FakeAE(), iter([batch]), 1, 'cpu', 'mean')
    sample = estimate_normalizer(FakeAE(), iter([batch]), 1, 'cpu', 'sample')
    torch.testing.assert_close(mean.std, torch.full((2, 3), 1e-4))
    torch.testing.assert_close(sample.std, torch.full((2, 3), 2.))
    z = torch.randn(2, 60, 2, 3)
    torch.testing.assert_close(sample.denormalize(sample.normalize(z)), z, atol=1e-6, rtol=1e-5)
    restored = LatentNormalizer.from_metadata(sample.metadata())
    torch.testing.assert_close(restored.normalize(z), sample.normalize(z))


def test_frozen_ae_target_ignores_denoiser_drop_and_keeps_truth_time():
    ae = MotionAutoencoder(t5_out_dim=4, width=16, layers=1, heads=2, slots=2, latent_dim=3).eval().requires_grad_(False)
    y = conditions()
    norm = LatentNormalizer(torch.zeros(2, 3), torch.ones(2, 3))
    x = torch.randn(2, 5, 9, 60)
    before = encode_target(ae, norm, x, y)
    y['tempo_active'] = torch.zeros(2, dtype=torch.bool)
    y['action_label_active'] = torch.zeros(2, dtype=torch.bool)
    after = encode_target(ae, norm, x, y)
    torch.testing.assert_close(before, after)
    assert not after.requires_grad and not ae.training
    _, time = ae_inputs(y)
    assert time['loop_tile_count'].tolist() == [3, 3]
    assert 'tempo_active' not in time


@pytest.mark.parametrize('launcher', ['train_tiny_vae_diffusion.bat', 'train_all_vae_diffusion.bat'])
@pytest.mark.parametrize('stage', ['train_autoencoder.py', 'train_latent_diffusion.py'])
def test_batch_launcher_arguments_are_accepted(launcher, stage):
    from train.train_autoencoder import parser as vae_parser

    bat = (Path(__file__).resolve().parents[1] / launcher).read_text()
    bat = bat.split(f'"%PYTHON_EXE%" train/{stage}', 1)[1].split('set "EXIT_CODE=', 1)[0]
    # Check argument parsing without binding the test to a run or checkpoint file.
    bat = bat.replace('%AE_RUN_NAME%', 'unused-ae-checkpoint')
    arguments = []
    for line in bat.splitlines():
        if line.strip().startswith('--'):
            arguments.extend(line.strip().rstrip('^').strip().replace('"', '').split())
    args = (vae_parser() if stage == 'train_autoencoder.py' else parser()).parse_args(arguments)
    assert args.num_frames == 60
    if stage == 'train_autoencoder.py':
        assert args.variational
    else:
        assert args.latent_mode == 'mean'
        assert args.ae_checkpoint == 'save/unused-ae-checkpoint'
    expected_compile = ('cuda-graph' if launcher == 'train_tiny_vae_diffusion.bat'
                        and stage == 'train_latent_diffusion.py' else 'reduce-overhead')
    assert args.compile == expected_compile


def test_species_film_applies_to_every_row():
    model = LatentMotionDenoiser(t5_out_dim=4, width=16, layers=1, heads=2,
                                 ff_size=32, skeleton_layers=1, slots=2, latent_channels=3,
                                 action_label_cond=False,
                                 dropout=0, joint_name_drop_prob=0, rest_geom_joint_drop_prob=0).eval()
    torch.nn.init.normal_(model.static.species_film_j[-1].weight, std=0.1)
    y = conditions()
    y['species_code'] = torch.randn(2, species_code_dim(4))
    z = torch.randn(2, 60, 2, 3)
    before = model(z, torch.zeros(2), y)
    y['species_code'] = y['species_code'] + 20
    after = model(z, torch.zeros(2), y)
    assert not torch.allclose(before[0], after[0])
    assert not torch.allclose(before[1], after[1])


def test_checkpoint_binding_and_contract_rejection(tmp_path):
    from utils.latent_util import (
        load_frozen_autoencoder, load_latent_checkpoint, LATENT_CHECKPOINT_VERSION, TIME_CONTRACT)
    from model.motion_autoencoder import DECODER_CONTENT_VERSION
    from data_loaders.truebones.truebones_utils.canonical_features import CANONICAL_FEATURE_SPACE
    ae = MotionAutoencoder(t5_out_dim=4, width=16, layers=1, heads=2, slots=2, latent_dim=3)
    ae_path = tmp_path / 'ae.pt'
    torch.save(dict(kind='motion_autoencoder_7a', model_config=ae.config, model=ae.state_dict(),
                    args=dict(num_frames=60, use_ema=False), canonical_version=CANONICAL_FEATURE_SPACE,
                    decoder_content_version=DECODER_CONTENT_VERSION), ae_path)
    frozen, binding = load_frozen_autoencoder(ae_path)
    assert not frozen.training and all(not p.requires_grad for p in frozen.parameters())
    model = denoiser()
    payload = dict(kind=LATENT_CHECKPOINT_VERSION, time_contract=TIME_CONTRACT,
                   prediction='standardized_z0',
                   ae_binding=binding, model_config=model.config, model=model.state_dict(), model_ema=None,
                   normalization=LatentNormalizer(torch.zeros(2, 3), torch.ones(2, 3)).metadata())
    path = tmp_path / 'latent.pt'
    torch.save(payload, path)
    restored, _, _, _ = load_latent_checkpoint(path)
    torch.testing.assert_close(restored.input.weight, model.input.weight)
    legacy = dict(payload, model_config=dict(model.config, tempo_cfg_drop_prob=1.0))
    torch.save(legacy, path)
    restored, _, _, _ = load_latent_checkpoint(path)
    assert 'tempo_cfg_drop_prob' not in restored.config
    torch.testing.assert_close(restored.input.weight, model.input.weight)
    for key in ('kind', 'time_contract', 'prediction'):
        wrong = dict(payload, **{key: 'incompatible'})
        torch.save(wrong, path)
        with pytest.raises(ValueError, match='contract'):
            load_latent_checkpoint(path)
    torch.save(dict(payload, kind='latent_diffusion_v1'), path)
    with pytest.raises(ValueError, match='contract'):
        load_latent_checkpoint(path)
    wrong = dict(payload, ae_binding={**binding, 'sha256': 'wrong'})
    torch.save(wrong, path)
    with pytest.raises(ValueError, match='hash'):
        load_latent_checkpoint(path)


def test_shared_static_extraction_preserves_original_amp_input_precision():
    from model.anytop import InputProcess, create_sin_embedding
    from model.motion_transformer import run_in_fp32
    module = InputProcess(9, 16, 4)
    x = torch.randn(2, 5, 9, 60)
    y = conditions()
    valid = torch.ones(2, 5, dtype=torch.bool)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = module(x, y['rest_geom'], y['joints_names_embs'], valid, y['joint_struct'])
        expected = module.joint_embedding(x.permute(3, 0, 1, 2))
        expected = expected + run_in_fp32(module.text_embedding, y['joints_names_embs'])[None]
        structure = run_in_fp32(module.struct_embedding, y['joint_struct'].to(expected.dtype))
        rest = run_in_fp32(module.rest_geom_embedding, y['rest_geom'].to(expected.dtype))
        expected = expected + (structure + rest)[None]
        positions = torch.arange(60).view(1, -1, 1)
        expected = expected + create_sin_embedding(positions, 16)[0][:, None, None]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
