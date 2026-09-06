"""The word-keyed conditioning path end to end: loader ids -> model channels -> checkpoint.

Covers the acceptance list that the pure-contract tests
(test_action_label_conditioning_contract.py) cannot: the
tensor mirror of the slot assembly, padding and masks, the projection width gate,
and the checkpoint contract including EMA/resume round trips and the refusals.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loaders.truebones.truebones_utils.action_label_conditioning_contract import (  # noqa: E402
    ACTION_CHECKPOINT_VERSION,
    ACTION_LABEL_SLOTS,
    ROLE_HEAD_1,
    SLOT_PAD_ID,
    ActionConditioningError,
    action_label_slots,
    assemble_slot_channels,
    assert_bundle_matches_metadata,
    build_action_conditioning_bundle,
    fingerprint,
    load_action_conditioning_bundle,
    action_word_embedding_payload,
    validate_action_conditioning_metadata,
    word_table_sha256,
)
from data_loaders.truebones.truebones_utils.motion_labels import (  # noqa: E402
    ACTION_LABEL_MAX_WORDS,
    parse_action_label,
)
from model.anytop import AnyTop  # noqa: E402
from utils.model_util import (  # noqa: E402
    bind_checkpoint_action_conditioning,
    build_checkpoint_payload,
    load_checkpoint_weights,
    load_model,
)
from tests.action_label_test_utils import (  # noqa: E402
    TEST_LATENT_DIM,
    TEST_T5_DIM,
    action_cond_fields,
    make_test_bundle,
    reference_channels,
)


def _model(bundle=None, latent_dim=TEST_LATENT_DIM, drop_prob=0.0,
           action_label_cond=True, eval_mode=True):
    # AnyTop.train() drops nn.Module.train's return value, so .eval() cannot be
    # chained onto the constructor here.
    model = AnyTop(
        max_joints=4,
        feature_len=13,
        latent_dim=latent_dim,
        ff_size=32,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        cross_limb=True,
        t5_out_dim=TEST_T5_DIM,
        action_label_cond=action_label_cond,
        action_label_cfg_drop_prob=drop_prob,
        action_conditioning=bundle if action_label_cond else None,
    )
    if eval_mode:
        model.eval()
    return model


def _channels(model, labels, groups, dtype=torch.float64):
    fields = action_cond_fields(labels, groups)
    return model._assemble_action_slot_channels(
        fields['action_word_ids'],
        fields['action_role_ids'],
        fields['action_slot_ids'],
        fields['action_word_mask'],
        dtype,
    )


# --------------------------------------------------------------------------
# The model mirrors the contract, on the contract's own slot ids
# --------------------------------------------------------------------------
def test_model_channels_equal_the_numpy_contract():
    bundle = make_test_bundle()
    model = _model(bundle)
    labels = ['idle, attack', 'attack, idle', 'walk, forward, weapon, 1hand', 'idle']
    groups = ['transition', 'transition', 'locomotion', 'stationary']
    got = _channels(model, labels, groups)
    expected = reference_channels(bundle, labels, groups)
    assert torch.allclose(got, expected, atol=1e-12)


def test_projection_consumes_one_block_per_slot():
    model = _model(make_test_bundle())
    assert model.action_label_projection[0].in_features == len(ACTION_LABEL_SLOTS) * TEST_T5_DIM


def test_head_and_direction_channels_do_not_move_when_modifiers_are_added():
    """Long-label axis retention as an equality, not a tuned constant."""
    model = _model(make_test_bundle())
    short = _channels(model, ['walk, forward'], ['locomotion'])
    longer = _channels(model, ['walk, forward, weapon, 1hand, fast'], ['locomotion'])
    width = TEST_T5_DIM
    assert torch.equal(short[:, :2 * width], longer[:, :2 * width])
    # ...and the modifier channel is exactly what moved.
    assert not torch.equal(short[:, 2 * width:], longer[:, 2 * width:])


def test_absent_slot_is_a_zero_row_and_leaves_the_others_alone():
    model = _model(make_test_bundle())
    width = TEST_T5_DIM
    with_direction = _channels(model, ['walk, forward'], ['locomotion'])
    without = _channels(model, ['walk'], ['locomotion'])
    assert torch.count_nonzero(without[:, width:2 * width]) == 0
    assert torch.count_nonzero(without[:, 2 * width:]) == 0
    # The head channel is unit norm in both, so the absent slot renormalised nothing.
    for row in (with_direction, without):
        assert float(torch.linalg.vector_norm(row[0, :width])) == pytest.approx(1.0)
    assert torch.equal(with_direction[:, :width], without[:, :width])


def test_reverse_transitions_do_not_collide():
    model = _model(make_test_bundle())
    forward = _channels(model, ['idle, attack'], ['transition'])
    backward = _channels(model, ['attack, idle'], ['transition'])
    cosine = torch.nn.functional.cosine_similarity(forward, backward).item()
    assert abs(cosine) < 0.5
    # The same two words in a group whose role gate is closed DO pool the same:
    # order is only meaningful for a transition.
    stationary = [_channels(model, [label], ['stationary'])
                  for label in ('idle, attack', 'attack, idle')]
    assert torch.allclose(stationary[0], stationary[1], atol=1e-12)


def test_padding_and_masks_take_full_effect():
    """Padding columns must be inert whichever way a reader looks at them."""
    bundle = make_test_bundle()
    model = _model(bundle)
    fields = action_cond_fields(['walk, forward'], ['locomotion'])
    clean = model._assemble_action_slot_channels(
        fields['action_word_ids'], fields['action_role_ids'],
        fields['action_slot_ids'], fields['action_word_mask'], torch.float64,
    )
    # Fill the padded columns with a real word in a real slot: the mask, not the
    # ids, is what keeps them out.
    dirty_ids = fields['action_word_ids'].clone()
    dirty_slots = fields['action_slot_ids'].clone()
    dirty_ids[:, 2:] = 5
    dirty_slots[:, 2:] = 0
    dirty = model._assemble_action_slot_channels(
        dirty_ids, fields['action_role_ids'], dirty_slots,
        fields['action_word_mask'], torch.float64,
    )
    assert torch.equal(clean, dirty)
    assert fields['action_slot_ids'][0, 2] == SLOT_PAD_ID
    assert fields['action_word_ids'].shape[1] == ACTION_LABEL_MAX_WORDS


def test_partial_condition_fields_are_refused():
    model = _model(make_test_bundle())
    fields = action_cond_fields(['walk, forward'], ['locomotion'])
    y = {key: value for key, value in fields.items() if key != 'action_slot_ids'}
    with pytest.raises(ValueError, match="action_slot_ids"):
        model._action_slot_inputs(y, 1, torch.device('cpu'))


def test_word_id_outside_the_vocabulary_is_refused():
    model = _model(make_test_bundle())
    fields = action_cond_fields(['walk, forward'], ['locomotion'])
    fields['action_word_ids'][0, 0] = 10_000
    with pytest.raises(ValueError, match="ordered vocabulary"):
        model._action_slot_inputs(fields, 1, torch.device('cpu'))


def test_loader_and_model_agree_on_the_role_gate():
    """The gate is contextual, so the ids the loader emits carry it, not the model."""
    slots = action_label_slots('transition', parse_action_label('idle, attack'))
    assert slots['role_ids'][1] == ROLE_HEAD_1
    fields = action_cond_fields(['idle, attack'], ['transition'])
    assert fields['action_role_ids'][0, :2].tolist() == [0, ROLE_HEAD_1]
    assert fields['action_order_head_mask'][0, :2].tolist() == [True, True]
    # turn-headed and single-head transitions never apply R_B.
    for label in ('turn, hover, left', 'die'):
        emitted = action_cond_fields([label], ['transition'])
        assert not bool((emitted['action_role_ids'] == ROLE_HEAD_1).any())


# --------------------------------------------------------------------------
# Construction gates
# --------------------------------------------------------------------------
def test_latent_dim_below_the_slot_source_rank_fails_at_construction():
    bundle = make_test_bundle()
    total_rank = bundle.slot_source_rank_report(TEST_LATENT_DIM)['total_rank']
    assert total_rank == 135  # 32 state words + their R_B images, 6 directions, 65 modifiers
    with pytest.raises(ValueError, match="smaller than the total slot source rank"):
        _model(bundle, latent_dim=total_rank - 1)
    _model(bundle, latent_dim=TEST_LATENT_DIM)  # the first width at or above it


def test_word_table_from_another_encoder_is_refused():
    bundle = make_test_bundle()
    with pytest.raises(ValueError, match="t5_out_dim"):
        AnyTop(
            max_joints=4, feature_len=13, latent_dim=TEST_LATENT_DIM, ff_size=32,
            num_layers=1, num_heads=2, dropout=0.0, cross_limb=True,
            t5_out_dim=TEST_T5_DIM // 2, action_label_cond=True,
            action_conditioning=bundle,
        )


def test_a_wrong_width_word_table_has_no_role_transform():
    narrow = np.zeros((103, 384), dtype=np.float32) + 1.0
    # A contract that fully describes THIS table, so the width is the only thing
    # wrong with it: otherwise the vector-hash check refuses it first and the
    # role transform never gets asked about.
    contract = dict(
        make_test_bundle().embedding_contract,
        embedding_dim=384,
        word_table_sha256=word_table_sha256(narrow),
    )
    with pytest.raises(ActionConditioningError, match="role transform"):
        build_action_conditioning_bundle(
            narrow, contract, source='narrow', check_token_text=False,
        )


# --------------------------------------------------------------------------
# Sidecar round trip
# --------------------------------------------------------------------------
def test_word_sidecar_round_trips_and_rejects_a_foreign_vocabulary(tmp_path):
    bundle = make_test_bundle()
    path = tmp_path / 'action_word_embeddings.npy'
    payload = action_word_embedding_payload(
        bundle.word_embeddings, bundle.embedding_contract
    )
    np.save(path, payload, allow_pickle=True)
    reloaded = load_action_conditioning_bundle(path)
    assert reloaded.embedding_fingerprint == bundle.embedding_fingerprint
    assert reloaded.conditioning_contract_fingerprint == bundle.conditioning_contract_fingerprint
    assert np.array_equal(reloaded.word_embeddings, bundle.word_embeddings)

    payload['ordered_vocab'] = list(payload['ordered_vocab'])[::-1]
    np.save(path, payload, allow_pickle=True)
    with pytest.raises(ActionConditioningError, match="CONTROLLED_VOCAB"):
        load_action_conditioning_bundle(path)


def test_an_edited_sidecar_fails_its_own_fingerprint(tmp_path):
    bundle = make_test_bundle()
    path = tmp_path / 'action_word_embeddings.npy'
    payload = action_word_embedding_payload(
        bundle.word_embeddings, bundle.embedding_contract
    )
    payload['embedding_contract'] = dict(payload['embedding_contract'], t5_name='other')
    np.save(path, payload, allow_pickle=True)
    with pytest.raises(ActionConditioningError, match="embedding_fingerprint"):
        load_action_conditioning_bundle(path)


# --------------------------------------------------------------------------
# Checkpoint contract
# --------------------------------------------------------------------------
def _save_and_reload(tmp_path, model, model_avg=None, name='model.pt'):
    payload = build_checkpoint_payload(
        model.state_dict(),
        None if model_avg is None else model_avg.state_dict(),
        model,
    )
    path = tmp_path / name
    torch.save(payload, path)
    return torch.load(path, map_location='cpu', weights_only=False)


def test_checkpoint_carries_both_fingerprints():
    bundle = make_test_bundle()
    model = _model(bundle)
    payload = build_checkpoint_payload(model.state_dict(), None, model)
    metadata = payload['metadata']
    assert metadata['checkpoint_version'] == ACTION_CHECKPOINT_VERSION
    action = metadata['action_conditioning']
    assert action['embedding_fingerprint'] == bundle.embedding_fingerprint
    assert action['conditioning_contract_fingerprint'] == bundle.conditioning_contract_fingerprint
    validate_action_conditioning_metadata(action, source='payload')


def test_save_load_resume_round_trip_keeps_the_condition(tmp_path):
    bundle = make_test_bundle()
    model = _model(bundle)
    torch.nn.init.normal_(model.action_label_null_emb)
    model_avg = _model(bundle)
    model_avg.load_state_dict(model.state_dict())
    model.eval()

    reloaded_payload = _save_and_reload(tmp_path, model, model_avg)
    state, state_avg, metadata = load_checkpoint_weights(
        reloaded_payload, 'model.pt', prefer_ema=False
    )
    # A fresh model with NO bundle: the buffers must arrive with the weights.
    restored = _model(None)
    load_model(restored, state)
    bind_checkpoint_action_conditioning(restored, metadata, 'model.pt')
    assert torch.equal(restored.action_word_embeddings, model.action_word_embeddings)
    assert torch.equal(restored.action_role_b_perm, model.action_role_b_perm)

    labels, groups = ['idle, attack', 'walk, forward'], ['transition', 'locomotion']
    y_fields = action_cond_fields(labels, groups)
    before = model._build_action_label_token(y_fields, 2, torch.device('cpu'), torch.float32)
    after = restored._build_action_label_token(y_fields, 2, torch.device('cpu'), torch.float32)
    assert torch.allclose(before, after, atol=1e-6)

    restored_avg = _model(None)
    load_model(restored_avg, state_avg)
    bind_checkpoint_action_conditioning(restored_avg, metadata, 'model.pt')
    assert torch.allclose(
        restored_avg._build_action_label_token(
            y_fields, 2, torch.device('cpu'), torch.float32
        ),
        before,
        atol=1e-6,
    )
    # The resume half: this run's own bundle has to be the checkpoint's table.
    assert_bundle_matches_metadata(
        bundle, metadata['action_conditioning'], source='model.pt'
    )


def test_resume_across_word_tables_is_refused():
    bundle = make_test_bundle()
    other = make_test_bundle(seed=999, t5_name='t5-other')
    metadata = build_checkpoint_payload(
        _model(bundle).state_dict(), None, _model(bundle)
    )['metadata']['action_conditioning']
    with pytest.raises(ActionConditioningError, match="different frozen word table"):
        assert_bundle_matches_metadata(other, metadata, source='model.pt')


def test_a_checkpoint_under_another_conditioning_contract_is_refused():
    bundle = make_test_bundle()
    metadata = _model(bundle).action_conditioning_metadata
    tampered = dict(metadata)
    contract = dict(tampered['conditioning_contract'])
    contract['role_gate'] = 'always'
    tampered['conditioning_contract'] = contract
    tampered['conditioning_contract_fingerprint'] = fingerprint(contract)
    with pytest.raises(ActionConditioningError, match="not the one this code implements"):
        validate_action_conditioning_metadata(tampered, source='model.pt')


def test_a_pre_v2_checkpoint_is_refused(tmp_path):
    path = tmp_path / 'legacy.pt'
    torch.save({'model': {}, 'model_avg': {}}, path)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    with pytest.raises(ActionConditioningError, match="predates checkpoint version"):
        load_checkpoint_weights(payload, str(path), prefer_ema=True)


def test_tampered_role_buffers_are_refused():
    bundle = make_test_bundle()
    model = _model(bundle)
    metadata = model.action_conditioning_metadata
    with torch.no_grad():
        model.action_role_b_sign[0] *= -1
    with pytest.raises(ActionConditioningError, match="material hash mismatch"):
        model.validate_loaded_action_conditioning(metadata, source='model.pt')


def test_a_checkpoint_whose_table_did_not_load_is_refused():
    """The placeholder buffers must never pass for a real word table."""
    bundle = make_test_bundle()
    metadata = _model(bundle).action_conditioning_metadata
    empty = _model(None)
    with pytest.raises(ActionConditioningError, match="did not load"):
        empty.validate_loaded_action_conditioning(metadata, source='model.pt')


def test_unconditioned_model_ignores_action_fields():
    model = _model(None, action_label_cond=False)
    assert model.action_conditioning_metadata is None
    assert model._build_action_label_token(
        action_cond_fields(['walk, forward'], ['locomotion']), 1,
        torch.device('cpu'), torch.float32,
    ) is None


def test_slot_assembly_has_exactly_one_definition():
    """The tensor path reads the contract's slot ids; it does not re-derive them."""
    bundle = make_test_bundle()
    model = _model(bundle)
    label, group = 'walk, forward, weapon', 'locomotion'
    slots = action_label_slots(group, parse_action_label(label))
    numpy_channels, present = assemble_slot_channels(
        bundle.word_embeddings, slots, bundle.role_b_perm, bundle.role_b_sign
    )
    assert present.tolist() == [True, True, True]
    torch_channels = _channels(model, [label], [group])
    assert torch.allclose(
        torch_channels[0], torch.as_tensor(numpy_channels.reshape(-1)), atol=1e-12
    )


# --------------------------------------------------------------------------
# The fingerprint covers the VECTORS, not only the recipe that made them
# --------------------------------------------------------------------------
def _other_vectors(bundle, seed=4242):
    """Another table under the SAME encoder metadata: only the vectors move.

    Deliberately not ``make_test_bundle(seed=...)``, which also moves
    ``t5_artifact_sha256`` -- that would prove the old metadata-only fingerprint
    still works, not that the fingerprint follows the table.
    """
    table = np.random.default_rng(seed).standard_normal(
        bundle.word_embeddings.shape
    ).astype(np.float32)
    contract = dict(
        bundle.embedding_contract, word_table_sha256=word_table_sha256(table)
    )
    return table, contract


def _recipe_only(contract):
    """The contract minus the one field that describes the vectors themselves."""
    return {key: value for key, value in contract.items() if key != 'word_table_sha256'}


def test_the_embedding_fingerprint_follows_the_vectors():
    bundle = make_test_bundle()
    table, contract = _other_vectors(bundle)
    other = build_action_conditioning_bundle(table, contract, source='other')
    # Same text, same encoder, same artifact hash, same pooling: every field the
    # contract carried before was identical for these two tables.
    assert _recipe_only(contract) == _recipe_only(bundle.embedding_contract)
    assert other.embedding_fingerprint != bundle.embedding_fingerprint
    with pytest.raises(ActionConditioningError, match="different frozen word table"):
        assert_bundle_matches_metadata(
            other, _model(bundle).action_conditioning_metadata, source='model.pt'
        )


def test_a_swapped_word_table_is_refused_by_the_sidecar_loader(tmp_path):
    """A foreign table under an untouched contract is not the contract's table."""
    bundle = make_test_bundle()
    table, _contract = _other_vectors(bundle)
    path = tmp_path / 'action_word_embeddings.npy'
    np.save(
        path,
        action_word_embedding_payload(table, bundle.embedding_contract),
        allow_pickle=True,
    )
    with pytest.raises(ActionConditioningError, match="word_table_sha256"):
        load_action_conditioning_bundle(path)


def test_a_checkpoint_whose_table_was_swapped_is_refused():
    bundle = make_test_bundle()
    model = _model(bundle)
    payload = build_checkpoint_payload(model.state_dict(), None, model)
    table, _contract = _other_vectors(bundle)
    payload['model'] = dict(
        payload['model'], action_word_embeddings=torch.from_numpy(table)
    )
    state, _avg, metadata = load_checkpoint_weights(
        payload, 'model.pt', prefer_ema=False
    )
    restored = _model(None)
    load_model(restored, state)
    with pytest.raises(ActionConditioningError, match="fitted on different word"):
        bind_checkpoint_action_conditioning(restored, metadata, 'model.pt')


def test_a_schema_2_contract_is_refused_by_name():
    """No word_table_sha256 means the fingerprint certifies nothing about the table."""
    bundle = make_test_bundle()
    metadata = dict(_model(bundle).action_conditioning_metadata)
    old = _recipe_only(metadata['embedding_contract'])
    old['schema_version'] = 2
    metadata['embedding_contract'] = old
    metadata['embedding_fingerprint'] = fingerprint(old)
    with pytest.raises(ActionConditioningError, match="schema_version"):
        validate_action_conditioning_metadata(metadata, source='old.pt')


# --------------------------------------------------------------------------
# Resume order: the bind certifies the CHECKPOINT's buffers
# --------------------------------------------------------------------------
def _resume(checkpoint_path, bundle, model, model_avg=None):
    """Drive the real TrainLoop resume path over a stub carrying just its inputs."""
    import types

    from train.training_loop import TrainLoop

    stub = types.SimpleNamespace(
        resume_checkpoint=str(checkpoint_path),
        model=model,
        model_avg=model_avg,
        args=types.SimpleNamespace(action_conditioning=bundle),
        find_resume_checkpoint=lambda: None,
        _get_checkpoint_step_numbering=lambda _path: 'completed_steps',
    )
    stub._assert_resume_action_conditioning = (
        TrainLoop._assert_resume_action_conditioning.__get__(stub)
    )
    TrainLoop._load_and_sync_parameters(stub)
    return stub


def _write_checkpoint(tmp_path, model, model_avg=None, name='model000100.pt'):
    payload = build_checkpoint_payload(
        model.state_dict(),
        None if model_avg is None else model_avg.state_dict(),
        model,
    )
    path = tmp_path / name
    torch.save(payload, path)
    return path, payload


def test_resume_certifies_the_buffers_that_actually_landed(tmp_path):
    """Binding before load_model certified this run's material, not the file's."""
    bundle = make_test_bundle()
    saved = _model(bundle)
    path, payload = _write_checkpoint(tmp_path, saved, saved)
    tampered = payload['model']['action_role_b_sign'].clone()
    tampered[0] *= -1
    payload['model'] = dict(payload['model'], action_role_b_sign=tampered)
    payload['model_avg'] = payload['model']
    torch.save(payload, path)

    with pytest.raises(ActionConditioningError, match="material hash mismatch"):
        _resume(path, bundle, _model(bundle), _model(bundle))


def test_resume_refuses_a_checkpoint_whose_table_was_swapped(tmp_path):
    bundle = make_test_bundle()
    saved = _model(bundle)
    path, payload = _write_checkpoint(tmp_path, saved, saved)
    table, _contract = _other_vectors(bundle)
    payload['model'] = dict(
        payload['model'], action_word_embeddings=torch.from_numpy(table)
    )
    payload['model_avg'] = payload['model']
    torch.save(payload, path)

    with pytest.raises(ActionConditioningError, match="fitted on different word"):
        _resume(path, bundle, _model(bundle), _model(bundle))


def test_resume_binds_the_ema_copy_too(tmp_path):
    """model_avg carries its own buffers and is what sampling and eval read."""
    bundle = make_test_bundle()
    saved = _model(bundle)
    saved_avg = _model(bundle)
    path, payload = _write_checkpoint(tmp_path, saved, saved_avg)
    table, _contract = _other_vectors(bundle)
    # Only the EMA copy is wrong. The online model loads and binds cleanly, so
    # nothing but a bind of its own can catch this one.
    payload['model_avg'] = dict(
        payload['model_avg'], action_word_embeddings=torch.from_numpy(table)
    )
    torch.save(payload, path)

    with pytest.raises(ActionConditioningError, match="fitted on different word"):
        _resume(path, bundle, _model(bundle), _model(bundle))


def test_resume_binds_the_checkpoints_own_table(tmp_path):
    """A model built with no bundle: only a post-load bind can see a table at all."""
    bundle = make_test_bundle()
    saved = _model(bundle)
    path, _payload = _write_checkpoint(tmp_path, saved, saved)

    stub = _resume(path, bundle, _model(None), _model(None))
    for restored in (stub.model, stub.model_avg):
        assert torch.equal(
            restored.action_word_embeddings, saved.action_word_embeddings
        )
        assert (
            restored.action_conditioning_metadata['embedding_fingerprint']
            == bundle.embedding_fingerprint
        )
