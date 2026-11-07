"""AE batteries use reconstruction outputs and preserve incremental scoring."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from eval import eval_checkpoint as evaluation
from sample import generate_ae


def test_ae_task_exports_target_and_rescores_only_reconstruction(tmp_path, monkeypatch):
    source = tmp_path / 'Cat_Walk.npy'
    np.save(source, np.zeros((3, 2, 12)))
    extra = ['--npy', str(source), '--object_type', 'zoo/Dog']
    calls = []
    runtime = {}

    def reconstruct(argv, runtime=None):
        calls.append(runtime)
        assert '--batch_size' not in argv
        out = Path(argv[argv.index('--output_dir') + 1])
        for label in ('original', 'reconstruction'):
            np.save(out / f'Cat_Walk_{label}.npy', np.zeros((3, 2, 12)))
            (out / f'Cat_Walk_{label}.bvh').write_text(label)
        report = dict(cond_path=str(tmp_path / 'cond.npy'), samples=[dict(
            motion_path=str(source), target_object_type='zoo/Dog')])
        (out / 'reconstruction_metrics.json').write_text(json.dumps(report))
        return report

    monkeypatch.setattr(generate_ae, 'main', reconstruct)
    monkeypatch.setattr(evaluation, '_register_cond_path', lambda *a: {'zoo/Dog': {}})
    scored = []

    def score(_scorer, task_dir, token, species, label):
        scored.append((token, species, label))
        assert [p.name for p in task_dir.glob(f'{token}_*.npy')] == ['Dog_0.npy']
        return None

    monkeypatch.setattr(evaluation, '_score_task', score)
    record = evaluation.run_task(runtime, object(), tmp_path / 'autoencoder.pt',
                                 'T1', 1, extra, 'walk', tmp_path, None)
    assert record['status'].startswith('ok')
    assert record['first_npy'].name == 'Dog_0.npy'
    assert Path(record['reference_motion']).read_text() == 'original'
    assert calls == [runtime]
    reused = evaluation._build_record_from_existing(
        tmp_path / 'T1/task1', 'T1', 1, object(), tmp_path, 'walk')
    assert reused['first_npy'] == record['first_npy']
    assert reused['reference_motion'] == record['reference_motion']
    assert scored == [('Dog', 'zoo/Dog', 'walk')] * 2


def test_ae_config_resolves_and_hashes_all_npy_inputs(tmp_path):
    clips = [tmp_path / f'Cat_{i}.npy' for i in range(2)]
    for path in clips:
        path.write_bytes(b'original')
    config = tmp_path / 'tasks.json'
    config.write_text(json.dumps(dict(checkpoint=dict(RUN_NAME='ae'), tasks=[dict(
        category='T1', args=['--npy', *map(str, clips)], eval_label='walk')])))
    _, tasks = evaluation._load_task_config(config)
    args = tasks[0][1]
    before = evaluation._task_param_hash(args)
    clips[1].write_bytes(b'changed')
    assert evaluation._task_param_hash(args) != before


def test_ae_checkpoint_defaults_to_autoencoder_file(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, '_ANYTOP_DIR', tmp_path)
    run = tmp_path / 'save/ae'
    run.mkdir(parents=True)
    model = run / 'autoencoder.pt'
    model.touch()
    (run / 'model999.pt').touch()
    assert evaluation._resolve_checkpoint(dict(RUN_NAME='ae', TYPE='autoencoder')) == model


def test_ae_main_dispatches_without_preparing_diffusion(tmp_path, monkeypatch):
    run = tmp_path / 'save/ae'
    run.mkdir(parents=True)
    (run / 'autoencoder.pt').touch()
    clip = tmp_path / 'Cat_Walk.npy'
    clip.touch()
    config = tmp_path / 'ae.json'
    config.write_text(json.dumps(dict(checkpoint=dict(RUN_NAME='ae'), tasks=[dict(
        category='T1', args=['--npy', str(clip)], eval_label='walk') for _ in range(2)])))
    monkeypatch.setattr(evaluation, '_ANYTOP_DIR', tmp_path)
    monkeypatch.setattr(evaluation, 'MotionQualityScorer', lambda **_: object())
    monkeypatch.setattr(evaluation, 'prepare_generation_runtime',
                        lambda *_: (_ for _ in ()).throw(AssertionError('diffusion runtime')))
    runtimes = []

    def run_task(runtime, _scorer, model_path, category, index, *args, **kwargs):
        assert model_path == run / 'autoencoder.pt'
        runtimes.append(runtime)
        return dict(status='ok', quality=None, first_npy=None)

    monkeypatch.setattr(evaluation, 'run_task', run_task)
    reports = []
    monkeypatch.setattr(evaluation, 'write_html_report', lambda *a: reports.append(a))
    monkeypatch.setattr(sys, 'argv', ['eval_checkpoint.py', '--task_config', str(config),
                                    '--dataset_root', str(tmp_path), '--no_review'])
    assert evaluation.main() == 0
    assert len(runtimes) == 2 and runtimes[0] is runtimes[1]
    assert runtimes[0] == {}
    assert len(reports) == 1
