"""Latent contracts, normalization, condition priority and diffusion isolation."""
import copy
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loaders.reference import REST_GEOM_DIM, REFERENCE_POSE_CHANNELS
from data_loaders.truebones.truebones_utils.joint_struct_features import JOINT_STRUCT_DIM
from data_loaders.truebones.truebones_utils.dataset_tags import species_code_dim
from diffusion.latent_diffusion import create_latent_diffusion, validate_latent_conditions
from model.latent_motion_denoiser import LatentMotionDenoiser
from model.motion_autoencoder import MotionAutoencoder
from utils.latent_util import LatentNormalizer, estimate_normalizer, encode_target, ae_inputs
from train.train_latent_diffusion import LatentBatchTransform, parser, training_forwards
from action_label_test_utils import make_test_bundle, action_cond_fields


@pytest.mark.parametrize('is_loop', [True, False])
@pytest.mark.parametrize('suffix', ['.npy', '.glb'])
def test_latent_reference_loop_is_read_before_working_conversion(monkeypatch, tmp_path, capsys,
                                                                is_loop, suffix):
    import numpy as np
    from types import SimpleNamespace
    from sample import generate_latent as generation
    from data_loaders.truebones.truebones_utils.loop_verdict import apply_loop_verdict

    rng = np.random.default_rng(7)
    stored = apply_loop_verdict(rng.normal(size=(21, 4, 12)), is_loop, 0)
    reference_path = tmp_path / ('source_fly' + suffix)
    if suffix == '.npy':
        np.save(reference_path, stored)
    source_cond = {'translation_root_index': 0}
    model = SimpleNamespace(t5_out_dim=4, species_cond=False, ref_encoder=True)
    checkpoint = {'args': {'cond_path': str(tmp_path / 'cond.npy')}, 'cond_sha256': 'test'}
    (tmp_path / 'cond.npy').touch()
    monkeypatch.setattr(generation, 'load_latent_checkpoint', lambda *a: (model, None, None, checkpoint))
    monkeypatch.setattr(generation, 'file_sha256', lambda *a: 'test')
    monkeypatch.setattr(generation, 'load_cond', lambda *a: {'source': source_cond})
    monkeypatch.setattr(generation, 'mark_canonical_cond_entry', lambda *a: None)
    monkeypatch.setattr(generation, 'ensure_joint_name_embeddings', lambda *a, **k: None)
    monkeypatch.setattr(generation, '_resolve_action_condition', lambda *a: None)
    monkeypatch.setattr(generation, '_resolve_species_code_override', lambda *a: None)
    monkeypatch.setattr(generation, '_load_raw_encoder_reference', lambda *a: (stored, source_cond))
    converted = []
    convert = generation.stored_to_working

    def to_working(features):
        # Both input formats must reach conversion with their stored rows intact.
        np.testing.assert_array_equal(features, stored)
        converted.append(convert(features))
        return converted[-1]

    monkeypatch.setattr(generation, 'stored_to_working', to_working)

    class SamplingBoundary(Exception):
        pass

    def stop_before_sampling(*a, **k):
        raise SamplingBoundary

    monkeypatch.setattr(generation, 'loop_period', stop_before_sampling)
    with pytest.raises(SamplingBoundary):
        generation.main(['--model_path', str(tmp_path / 'latent.pt'), '--object_type', 'source',
                         '--reference_object_type', 'source', '--reference_motion', str(reference_path),
                         '--output_dir', str(tmp_path / 'output'), '--device', 'cpu'])
    assert f"loop (auto) -> {'on' if is_loop else 'off'} (the reference's stored loop verdict)" in capsys.readouterr().out
    assert len(converted) == 1
    assert np.count_nonzero(converted[0][..., 9:12]) == 0


@pytest.mark.parametrize('reference_frames,requested_frames,expected_frames', [
    (145, None, 120), (10, None, 20), (60, None, 60), (120, None, 120),
    (145, 80, 80), (145, 145, None), (145, 0, None),
])
def test_latent_reference_output_length_matches_motion_generation(
        monkeypatch, tmp_path, reference_frames, requested_frames, expected_frames):
    import numpy as np
    from types import SimpleNamespace
    from sample import generate_latent as generation

    reference = tmp_path / 'Source_walk.npy'
    np.save(reference, np.zeros((reference_frames, 3, 12), dtype=np.float32))
    (tmp_path / 'cond.npy').touch()
    source_cond = {'translation_root_index': 0, 'parents': [-1, 0, 0]}
    model = SimpleNamespace(t5_out_dim=4, species_cond=False, ref_encoder=True)
    monkeypatch.setattr(generation, 'load_latent_checkpoint',
                        lambda *a: (model, None, None, {'cond_sha256': 'hash'}))
    monkeypatch.setattr(generation, 'file_sha256', lambda *a: 'hash')
    monkeypatch.setattr(generation, 'load_cond', lambda *a: {'Source': source_cond, 'Target': source_cond})
    monkeypatch.setattr(generation, 'mark_canonical_cond_entry', lambda *a: None)
    monkeypatch.setattr(generation, 'ensure_joint_name_embeddings', lambda *a, **k: None)
    monkeypatch.setattr(generation, '_resolve_action_condition', lambda *a: None)
    monkeypatch.setattr(generation, '_resolve_species_code_override', lambda *a: None)
    monkeypatch.setattr(generation, 'resolve_loop_condition', lambda *a, **k: True)
    monkeypatch.setattr(generation, 'loop_period', lambda *a, **k: None)
    monkeypatch.setattr(generation, 'create_condition', lambda *a, **k: (None, {'y': {}}))
    resample = generation.resample_reference_to_output
    fitted = []

    class ReferenceBoundary(Exception):
        pass

    def stop(features, frames):
        fitted.append(resample(features, frames)[0])
        raise ReferenceBoundary

    monkeypatch.setattr(generation, 'resample_reference_to_output', stop)
    flags = ['--model_path', str(tmp_path / 'latent.pt'), '--object_type', 'Target',
             '--reference_motion', str(reference), '--ref_cfg_scale', '6',
             '--output_dir', str(tmp_path / 'out'), '--device', '-1']
    if requested_frames is not None:
        flags += ['--num_frames', str(requested_frames)]
    if expected_frames is None:
        with pytest.raises(SystemExit, match='outside'):
            generation.main(flags)
        assert not fitted
    else:
        with pytest.raises(ReferenceBoundary):
            generation.main(flags)
        assert fitted[0].shape == (expected_frames, 3, 12)


def conditions(batch=2, joints=5, source_joints=3, names=4):
    return dict(n_joints=torch.tensor([joints - 1] * batch), lengths=torch.full((batch,), 60),
                joints_names_embs=torch.randn(batch, joints, names),
                rest_geom=torch.randn(batch, joints, REST_GEOM_DIM),
                joint_struct=torch.randn(batch, joints, JOINT_STRUCT_DIM),
                graph_dist=torch.zeros(batch, joints, joints, dtype=torch.long),
                joints_relations=torch.zeros(batch, joints, joints, dtype=torch.long),
                is_loop=torch.tensor([True] * batch), loop_tile_count=torch.full((batch,), 3),
                resample_speed_cond=torch.full((batch,), 1.5),
                ref_active=torch.ones(batch, dtype=torch.bool),
                ref_mask=torch.ones(batch, source_joints, dtype=torch.bool),
                ref_feats=torch.randn(batch, source_joints, 12, 60),
                ref_pose=torch.randn(batch, source_joints, REFERENCE_POSE_CHANNELS, 60),
                ref_rest_pose=torch.randn(batch, source_joints, REFERENCE_POSE_CHANNELS),
                ref_joints_names_embs=torch.randn(batch, source_joints, names),
                ref_joint_struct=torch.randn(batch, source_joints, JOINT_STRUCT_DIM))


def denoiser(**kwargs):
    return LatentMotionDenoiser(t5_out_dim=4, slots=2, latent_channels=3,
                                width=16, layers=1, heads=2, ff_size=32, skeleton_layers=1,
                                action_label_cond=False, species_cond=False, ref_width=8,
                                ref_layers=1, dropout=0, joint_name_drop_prob=0,
                                rest_geom_joint_drop_prob=0, **kwargs)


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
    assert model.trunk.layers[0].ref_attn_gate.grad is not None


def test_padding_joints_cannot_change_valid_prediction():
    model = denoiser(ref_encoder=False).eval()
    z = torch.randn(2, 60, 2, 3)
    y = conditions()
    changed = copy.deepcopy(y)
    for key in ('joints_names_embs', 'rest_geom', 'joint_struct'):
        changed[key][:, -1] = 1000
    torch.testing.assert_close(model(z, torch.zeros(2), y), model(z, torch.zeros(2), changed))


def test_reference_null_never_reads_source_and_mixed_inactive_nan_is_safe():
    model = denoiser().eval()
    model.trunk.layers[0].ref_attn_gate.data.fill_(1)
    z = torch.randn(2, 60, 2, 3)
    y = conditions()
    y['ref_active'][0] = False
    baseline = model(z, torch.zeros(2), y)
    for key in ('ref_feats', 'ref_pose', 'ref_rest_pose', 'ref_joints_names_embs'):
        y[key][0] = float('nan')
    result = model(z, torch.zeros(2), y)
    torch.testing.assert_close(result, baseline)
    y['ref_active'].zero_()
    for key in list(y):
        if key.startswith('ref_') and key != 'ref_active':
            del y[key]
    assert torch.isfinite(model(z, torch.zeros(2), y)).all()


def test_reference_drop_restores_cycle_semantics():
    model = denoiser(ref_cfg_drop_prob=1).train()
    y = conditions()
    _, active = model._reference_condition(y, 2, 60, torch.device('cpu'))
    assert not active.any()
    _, cycles = model._loop_condition(y, 2, torch.device('cpu'), active)
    assert cycles.tolist() == [3, 3]
    _, cycles = model._loop_condition(y, 2, torch.device('cpu'), ~active)
    assert cycles.tolist() == [1, 1]


def test_action_null_reference_priority_and_shared_adaln_token():
    model = LatentMotionDenoiser(t5_out_dim=768, width=144, layers=1, heads=4,
                                 ff_size=32, skeleton_layers=1, slots=2, latent_channels=3,
                                 action_conditioning=make_test_bundle(), species_cond=False,
                                 ref_encoder=False, dropout=0).eval()
    model.action_label_null_emb.data.fill_(0.5)
    y = conditions(names=768)
    y.update(action_cond_fields(['shake', 'walk'], ['stationary', 'locomotion']))
    active = torch.ones(2, dtype=torch.bool)
    token = model._build_action_label_token(y, 2, torch.device('cpu'), torch.float32, active)
    torch.testing.assert_close(token, model.action_label_null_emb.expand(2, -1))
    explicit = dict(y, action_label_active=torch.zeros(2, dtype=torch.bool))
    torch.testing.assert_close(token, model._build_action_label_token(explicit, 2, torch.device('cpu'), torch.float32))
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
                                'inpaint_frames', 'outpaint', 'cross_limb_unreliable_mask', 'ref_frame_mask'])
def test_edits_are_explicitly_rejected(key):
    with pytest.raises(ValueError, match='unsupported'):
        validate_latent_conditions({key: torch.zeros(1), 'lengths': torch.tensor([60])})


def test_fixed_window_and_reference_transform():
    y = conditions()
    y['lengths'][0] = 59
    with pytest.raises((ValueError, RuntimeError), match='invalid frames'):
        validate_latent_conditions(y)
    y['lengths'].fill_(60)
    y['ref_frame_mask'] = torch.zeros(2, 60, dtype=torch.bool)
    _, transformed = LatentBatchTransform()((torch.zeros(2, 5, 12, 60), {'y': y}))
    assert 'ref_frame_mask' not in transformed['y']
    y['ref_frame_mask'][0, 0] = True
    with pytest.raises(ValueError, match='hiding'):
        LatentBatchTransform()((torch.zeros(2, 5, 12, 60), {'y': y}))


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
    y['ref_active'].zero_()
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
    assert args.compile == 'reduce-overhead'


def test_compiled_training_forwards_preserve_targets_gradients_and_checkpoint_keys(monkeypatch):
    import utils.compile_util
    monkeypatch.setattr(utils.compile_util, 'configure_triton_c_compiler', lambda: None)
    compile_fn = torch.compile
    # Trace full graphs on CPU without requiring a platform-specific C compiler.
    monkeypatch.setattr(torch, 'compile', lambda fn, **kwargs: compile_fn(
        fn, backend='eager', dynamic=kwargs['dynamic'], fullgraph=kwargs['fullgraph']))
    ae = MotionAutoencoder(t5_out_dim=4, width=16, layers=1, heads=2,
                           slots=2, latent_dim=3).eval().requires_grad_(False)
    norm = LatentNormalizer(torch.zeros(2, 3), torch.ones(2, 3))
    model = denoiser(ref_cfg_drop_prob=0)
    keys = tuple(model.state_dict())
    forward, encoder = training_forwards(model, ae, norm, 'reduce-overhead')
    x, y = torch.randn(2, 5, 9, 60), conditions()
    y['motion_name'] = ['clip_a', 'clip_b']
    expected = encode_target(ae, norm, x, y)
    target = encoder(x, y)
    torch.testing.assert_close(target, expected)
    assert not target.requires_grad
    t = torch.ones(2, dtype=torch.long)
    reference_model = copy.deepcopy(model)
    actual = forward(target, t, y=y)
    reference = reference_model(expected, t, y=y)
    torch.testing.assert_close(actual, reference)
    actual.square().mean().backward()
    reference.square().mean().backward()
    for parameter, reference_parameter in zip(model.parameters(), reference_model.parameters()):
        if reference_parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, reference_parameter.grad)
    assert tuple(model.state_dict()) == keys
    assert not any(key.startswith('_orig_mod.') for key in model.state_dict())
    eager_forward, eager_encoder = training_forwards(model, ae, norm, 'None')
    assert eager_forward is model
    torch.testing.assert_close(eager_encoder(x, y), expected)


def test_species_film_is_bypassed_only_on_reference_active_rows():
    model = LatentMotionDenoiser(t5_out_dim=4, width=16, layers=1, heads=2,
                                 ff_size=32, skeleton_layers=1, slots=2, latent_channels=3,
                                 action_label_cond=False, ref_width=8, ref_layers=1,
                                 dropout=0, joint_name_drop_prob=0, rest_geom_joint_drop_prob=0).eval()
    torch.nn.init.normal_(model.static.species_film_j[-1].weight, std=0.1)
    y = conditions()
    y['ref_active'] = torch.tensor([True, False])
    y['species_code'] = torch.randn(2, species_code_dim(4))
    z = torch.randn(2, 60, 2, 3)
    before = model(z, torch.zeros(2), y)
    y['species_code'] = y['species_code'] + 20
    after = model(z, torch.zeros(2), y)
    torch.testing.assert_close(before[0], after[0])
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
    model = denoiser(ref_encoder=False)
    payload = dict(kind=LATENT_CHECKPOINT_VERSION, time_contract=TIME_CONTRACT,
                   reference_contract='source_pose_rest_v1', prediction='standardized_z0',
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
    for key in ('kind', 'time_contract', 'reference_contract', 'prediction'):
        wrong = dict(payload, **{key: 'incompatible'})
        torch.save(wrong, path)
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
