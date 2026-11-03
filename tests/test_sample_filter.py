"""Behavioral coverage for training clip combinations and reproducible resumes."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loaders.truebones.truebones_utils.sample_filter import (
    load_sample_filter, select_records, resolve_sample_filter, record_sample_selection,
)


def record(clip, *, tags=('Winged',), group='locomotion', heads=('fly',), action_label=None,
           object_type=None):
    namespace = clip.rsplit('/', 1)[0]
    return dict(clip_id=clip, namespace=namespace, object_type=object_type or namespace + '/Bird',
                species_tags=list(tags), action_group=group, heads=list(heads),
                action_label=action_label if action_label is not None else ', '.join(heads))


def ids(records):
    return {r['clip_id'] for r in records}


def test_local_exclusion_union_exact_add_and_global_exclusion():
    records = [record('zoo/fly.npy'), record('zoo/run.npy', heads=('run',)),
               record('other/shake.npy', group='stationary', heads=('shake',)),
               record('other/extra.npy', heads=('walk',)), record('zoo/bad.npy')]
    config = {'include': [
        {'namespaces': ['zoo'], 'species_tags_all': ['Winged'], 'heads_none': ['run', 'walk']},
        {'namespaces': ['other'], 'heads_any': ['shake']},
        {'clips': ['other/extra.npy', 'zoo/fly.npy']},
    ], 'exclude': [{'clips': ['zoo/bad.npy']}]}
    selected, counts = select_records(config, records + records[:1])
    assert ids(selected) == {'zoo/fly.npy', 'other/shake.npy', 'other/extra.npy'}
    assert len(selected) == 3  # OR never multiplies sampling weight.
    assert counts['exclude'] == [1]


@pytest.mark.parametrize('field,wanted,expected', [
    ('species_tags_all', ['Winged'], {'zoo/a.npy', 'zoo/b.npy'}),
    ('species_tags_all', ['Winged', 'Fluttering'], {'zoo/b.npy'}),
    ('species_tags_any', ['Fluttering', 'Quadruped'], {'zoo/b.npy', 'zoo/c.npy'}),
    ('species_tags_none', ['Fluttering'], {'zoo/a.npy', 'zoo/c.npy'}),
    ('species_tags_exact', ['Winged'], {'zoo/a.npy'}),
    ('species_tags_exact', ['fluttering', 'WINGED'], {'zoo/b.npy'}),
])
def test_species_tags_set_semantics(field, wanted, expected):
    records = [record('zoo/a.npy'), record('zoo/b.npy', tags=('Winged', 'Fluttering')),
               record('zoo/c.npy', tags=('Quadruped',))]
    selected, _ = select_records({'include': [{field: wanted}]}, records)
    assert ids(selected) == expected


def test_precise_namespaces_objects_globs_and_head_tokens():
    records = [record('truebones/zoo/Bird_Fly.npy'),
               record('truebones/zoo_upgrade/Bird_Fly.npy'),
               record('truebones/zoo/Bee_Run.npy', heads=('run', 'jump'),
                      object_type='truebones/zoo/Bee')]
    selected, _ = select_records({'include': [{'namespaces': ['truebones/zoo']}]}, records)
    assert len(selected) == 2  # Exact namespace does not select zoo_upgrade.
    selected, _ = select_records({'include': [{'namespace_prefixes': ['truebones'],
                                               'object_types': ['truebones/zoo/Bee'],
                                               'heads_all': ['run', 'jump'],
                                               'clip_globs': ['*/Bee_*.npy']}]}, records)
    assert ids(selected) == {'truebones/zoo/Bee_Run.npy'}


def test_tiny_selects_fly_or_hover_and_shake_across_datasets(dataset):
    cond, sources, metadata, tags = dataset
    config = load_sample_filter(Path(__file__).resolve().parents[1] / 'configs/sample_filters/tiny.json')
    config['include'][0]['namespaces'] = ['zoo']
    metadata['zoo']['Bird_1.npy'] = {'action_group': 'stationary', 'action_label': 'idle, hover'}
    metadata['zoo']['Bird_2.npy'] = {'action_group': 'locomotion', 'action_label': 'run'}
    metadata['zoo']['Bird_3.npy'] = {'action_group': 'stationary', 'action_label': 'idle'}
    metadata['other']['Bird_2.npy'] = {'action_group': 'stationary', 'action_label': 'shake'}
    tags['other/Bird'] = ('Quadruped',)
    report = resolve_sample_filter(config, cond, sources, metadata, tags)
    assert ids(report['clips']) == {'zoo/Bird_1.npy', 'zoo/Bird_4.npy', 'other/Bird_2.npy'}


@pytest.mark.parametrize('field,word', [('heads_any', 'hover'), ('words_any', 'hovering')])
def test_word_filters_keep_strict_vocabularies(field, word):
    with pytest.raises(ValueError, match='unknown'):
        load_sample_filter({'include': [{field: [word]}]})


@pytest.mark.parametrize('config', [
    {}, {'include': []}, {'include': [{}], 'other': []}, {'include': [{'species': ['Bee']}]},
    {'include': [{'tags_all': ['Winged']}]}, {'include': [{'heads_any': ['walking']}]},
    {'include': [{'heads_any': 'fly'}]}, {'include': [{'clips': []}]},
    {'include': [{'action_groups': ['all']}]}, {'include': [{}], 'exclude': {}},
])
def test_invalid_rules_fail_early(config):
    with pytest.raises(ValueError):
        load_sample_filter(config)


@pytest.mark.parametrize('field,value', [
    ('clips', 'zoo/missing.npy'), ('object_types', 'Bird'), ('namespaces', 'bad'),
    ('namespace_prefixes', 'zo'), ('species_tags_all', 'Typo'),
])
def test_unknown_exact_values_fail(field, value):
    with pytest.raises(ValueError, match='unknown'):
        select_records({'include': [{field: [value]}]}, [record('zoo/a.npy')])


def test_zero_match_is_warning_not_per_source_error():
    with pytest.warns(UserWarning, match='matches no clips'):
        selected, _ = select_records({'include': [{'heads_any': ['shake']}, {}]}, [record('zoo/a.npy')])
    assert len(selected) == 1


@pytest.fixture
def dataset(tmp_path):
    cond, metadata, tags, sources = {}, {}, {}, []
    for namespace, split in [('zoo', 'train'), ('other', 'train'), ('validation', 'val')]:
        root = tmp_path / namespace
        motions = root / 'motions'
        motions.mkdir(parents=True)
        sources.append(SimpleNamespace(namespace=namespace, split=split, motion_dir=str(motions)))
        key = namespace + '/Bird'
        cond[key] = {'dataset_namespace': namespace, 'species_name': 'Bird'}
        tags[key] = ('Winged',)
        metadata[namespace] = {}
        for index, length in enumerate([10, 30, 60, 80, 100]):
            name = f'Bird_{index}.npy'
            np.save(motions / name, np.zeros((length, 2, 3), dtype=np.float32))
            metadata[namespace][name] = {'action_group': 'locomotion', 'action_label': 'fly'}
    return cond, sources, metadata, tags


def resolve(dataset, config=None, **kwargs):
    return resolve_sample_filter(config or {'include': [{}]}, *dataset, **kwargs)


def test_split_length_and_seeded_limit_share_one_effective_pool(dataset):
    report = resolve(dataset)
    assert len(report['clips']) == 8
    assert len(report['too_short']) == 2
    assert all(not r['clip_id'].startswith('validation/') for r in report['clips'])
    limited = resolve(dataset, sample_limit=2, seed=17)
    assert limited == resolve(dataset, sample_limit=2, seed=17)
    assert all(r['length'] >= 60 for r in limited['clips'])
    assert len(limited['clips']) == 2
    assert len(resolve(dataset, split='val')['clips']) == 4
    assert len(resolve(dataset, split='all')['clips']) == 12


def test_precise_add_cannot_escape_split(dataset):
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError, match='no trainable clips'):
            resolve(dataset, {'include': [{'clips': ['validation/Bird_2.npy']}]})


def test_mixed_group_contract_and_resume_snapshot(dataset, tmp_path):
    dataset[2]['other']['Bird_2.npy'] = {'action_group': 'stationary', 'action_label': 'shake'}
    report = resolve(dataset)
    assert report['action_group'] == 'all'
    assert report['effective_action_groups'] == ['locomotion', 'stationary']
    run = tmp_path / 'run'
    run.mkdir()
    record_sample_selection(report, run)
    original = (run / 'sample_selection.json').read_text(encoding='utf-8')
    record_sample_selection(report, run, resume_directory=run)
    changed = copy.deepcopy(report)
    changed['clips'].pop()
    with pytest.raises(ValueError, match='change sample_filter'):
        record_sample_selection(changed, run, resume_directory=run)
    changed = copy.deepcopy(report)
    changed['config']['exclude'] = [{'heads_any': ['run']}]
    with pytest.raises(ValueError, match='change sample_filter'):
        record_sample_selection(changed, run, resume_directory=run)
    assert (run / 'sample_selection.json').read_text(encoding='utf-8') == original
    with pytest.raises(ValueError, match='missing'):
        record_sample_selection(report, run, resume_directory=tmp_path / 'missing')


def test_selected_corrupt_motion_is_not_silently_dropped(dataset):
    source = dataset[1][0]
    (Path(source.motion_dir) / 'Bird_2.npy').write_bytes(b'broken')
    with pytest.raises(ValueError, match='zoo/Bird_2.npy.*cannot read'):
        resolve(dataset)


def test_cli_defaults_to_all_and_rejects_removed_flags(monkeypatch):
    from utils.parser_util import train_args
    base = ['train', '--cond_path', 'cond.npy', '--sample_filter', 'tiny.json']
    monkeypatch.setattr(sys, 'argv', base)
    args = train_args()
    assert args.sample_filter == 'tiny.json'
    assert not hasattr(args, 'action_group')
    monkeypatch.setattr(sys, 'argv', ['train', '--cond_path', 'cond.npy'])
    assert train_args().sample_filter is None
    for flag, value in [('--objects_subset', 'all'), ('--dataset_namespaces', 'zoo'),
                        ('--action_group', 'all')]:
        monkeypatch.setattr(sys, 'argv', base + [flag, value])
        with pytest.raises(SystemExit):
            train_args()


def test_train_and_validation_default_to_all_without_inheriting_group(monkeypatch):
    from train import train_anytop as training
    from train import training_loop
    from utils.parser_util import train_args
    monkeypatch.setattr(sys, 'argv', ['train', '--cond_path', 'cond.npy'])
    args = train_args()
    seen = []
    monkeypatch.setattr(training, 'get_dataset_loader', lambda **kwargs: seen.append(kwargs))
    monkeypatch.setattr(training_loop, 'get_dataset_loader', lambda **kwargs: seen.append(kwargs))
    training.create_training_data_loader(args)
    args.action_group = 'locomotion'  # Derived metadata must not restrict validation.
    loop = training_loop.TrainLoop.__new__(training_loop.TrainLoop)
    loop.args, loop.batch_size = args, args.batch_size
    loop._build_val_loader()
    assert len(seen) == 2
    for options in seen:
        assert options['sample_filter'] == {'include': [{}]}
        assert not {'action_group', 'objects_subset', 'dataset_namespaces'} & options.keys()
    args.val_sample_filter = 'validation.json'
    loop._build_val_loader()
    assert seen[-1]['sample_filter'] == 'validation.json'


def test_real_loader_matches_legacy_flags_and_honours_the_config_file(monkeypatch):
    """Exercise the loader boundary as well as the pure rule engine.

    The filter must reproduce the removed CLI flags for the equivalent rule, and
    the shipped config file must drive the pool exactly as its own report says.
    Its contents are free to change, so nothing here pins them.
    """
    root = Path(__file__).resolve().parents[1]
    cond_path = root / 'dataset/cond_all.npy'
    if not cond_path.is_file():
        pytest.skip('Merged dataset absent')
    from data_loaders.get_data import get_dataset
    from data_loaders.truebones.data import dataset as module
    monkeypatch.setattr(module, '_save_motion_length_cache', lambda *args: None)
    common = dict(num_frames=60, split='train', cond_path=str(cond_path))
    old = get_dataset(**common, objects_subset='Winged', action_group='locomotion',
                      dataset_namespaces='truebones/zoo;integrate_20260924')
    legacy_equivalent = get_dataset(**common, sample_filter={
        'include': [{'namespaces': ['truebones/zoo', 'integrate_20260924'],
                     'species_tags_all': ['Winged'], 'action_groups': ['locomotion']}],
        'exclude': [],
    })
    assert set(old.motion_dataset.name_list) == set(legacy_equivalent.motion_dataset.name_list)
    configured = get_dataset(**common, sample_filter=root / 'configs/sample_filters/tiny.json')
    assert configured.motion_dataset.name_list
    assert set(configured.motion_dataset.name_list) == ids(configured.sample_filter_report['clips'])
    limited = get_dataset(**common, sample_filter=root / 'configs/sample_filters/tiny.json',
                          sample_limit=3, sample_filter_seed=37)
    assert len(limited) == 3
    assert set(limited.motion_dataset.name_list) == ids(limited.sample_filter_report['clips'])


def test_training_records_derived_group_and_refuses_changed_resume_before_overwrite(tmp_path, monkeypatch, dataset):
    from train import train_anytop as training
    from utils.parser_util import train_args
    source_cond = tmp_path / 'cond.npy'
    source_cond.write_bytes(b'fixture')
    save = tmp_path / 'run'
    save.mkdir()
    monkeypatch.setattr(sys, 'argv', ['train', '--cond_path', str(source_cond),
                                    '--save_dir', str(save), '--sample_filter', 'tiny.json'])
    args = train_args()
    args.ml_platform_type = 'NoPlatform'
    dataset[2]['other']['Bird_2.npy'] = {'action_group': 'stationary', 'action_label': 'shake'}
    report = resolve(dataset)

    class Pool(list):
        sample_filter_report = report
        motion_dataset = SimpleNamespace(cond_dict={})

    data = SimpleNamespace(dataset=Pool(report['clips']))
    monkeypatch.setattr(training, 'prepare_save_dir', lambda args: str(save))
    monkeypatch.setattr(training, 'get_opt', lambda *args: SimpleNamespace(cond_file=str(source_cond)))
    monkeypatch.setattr(training, 'resolve_t5_out_dim', lambda *args, **kwargs: None)
    monkeypatch.setattr(training.dist_util, 'setup_dist', lambda *args: None)
    monkeypatch.setattr(training.dist_util, 'dev', lambda: 'cpu')
    monkeypatch.setattr(training, 'bootstrap_action_conditioning', lambda *args: None)
    monkeypatch.setattr(training, 'bootstrap_species_descriptor_table', lambda *args: None)
    monkeypatch.setattr(training, 'create_training_data_loader', lambda *args: data)
    model = SimpleNamespace(reference_parameter_budget=lambda: {},
                            to=lambda *args: None)
    monkeypatch.setattr(training, 'create_model_and_diffusion_general_skeleton', lambda *args: (model, None))
    monkeypatch.setattr(training, 'TrainLoop', lambda *args: SimpleNamespace(run_loop=lambda: None))
    training.run_training(args)
    recorded = json.loads((save / 'args.json').read_text())
    assert recorded['action_group'] == 'all'
    assert recorded['effective_action_groups'] == ['locomotion', 'stationary']
    original = (save / 'args.json').read_bytes()
    args.resume_checkpoint = str(save / 'model000001.pt')
    training.run_training(args)  # Identical pool is accepted.
    original = (save / 'args.json').read_bytes()
    report['clips'].pop()
    with pytest.raises(ValueError, match='change sample_filter'):
        training.run_training(args)
    assert (save / 'args.json').read_bytes() == original
