import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.truebones.truebones_utils.motion_labels import load_motion_metadata, write_motion_metadata
from data_loaders.truebones.truebones_utils import dataset_pipeline as dataset_pipeline_mod
from data_loaders.truebones.truebones_utils import motion_process as motion_process_mod

from tools import regenerate_dataset_artifacts as regenerate_dataset_artifacts_module
from tools import process_new_skeleton as process_new_skeleton_module
import preprocess_and_validate as preprocess_and_validate_module


def _make_cond_entry(object_type: str) -> dict[str, object]:
    return {
        "object_type": object_type,
        "joints_names": ["Root", "Tail"],
        "parents": np.array([-1, 0], dtype=np.int64),
        "offsets": np.zeros((2, 3), dtype=np.float32),
    }


def _write_action_tags(dataset_dir, tags_by_clip):
    """Write the hand-maintained action_tags.jsonl sidecar for a temp dataset."""
    path = Path(dataset_dir) / "action_tags.jsonl"
    lines = [
        json.dumps({"clip": clip, "action_tags": list(tags)})
        for clip, tags in tags_by_clip.items()
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_write_motion_metadata_preserves_all_fields(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True)

    write_motion_metadata(
        dataset_dir,
        {
            "Cat_Run_001.npy": {
                "object_type": "Cat",
                "action_label": "run",
                "action_category": "locomotion",
                "action_tags": ["locomotion", "attack"],
                "species_label": "cat",
                "motion_name": "Cat_Run_001.npy",
                "translation_root_index": 1,
            },
        },
        total_clips=1,
    )

    payload = json.loads((dataset_dir / "motion_metadata.json").read_text(encoding="utf-8"))
    entry = payload["motions"]["Cat_Run_001.npy"]
    assert payload["schema_version"] == 5
    assert entry["object_type"] == "Cat"
    assert entry["species_label"] == "cat"
    assert entry["translation_root_index"] == 1
    assert entry["action_label"] == "run"
    assert entry["action_category"] == "locomotion"
    assert entry["action_tags"] == ["locomotion", "attack"]
    assert entry["motion_name"] == "Cat_Run_001.npy"


def test_load_motion_metadata_merges_action_tags_from_sidecar(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "motion_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 5,
                "total_clips": 1,
                "motions": {
                    "Cat_Run_001.npy": {
                        "object_type": "Cat",
                        "species_label": "cat",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    _write_action_tags(dataset_dir, {"Cat_Run_001.npy": ["Locomotion", "attack", "locomotion"]})

    loaded = load_motion_metadata(dataset_dir)
    entry = loaded["Cat_Run_001.npy"]
    assert entry["action_tags"] == ["locomotion", "attack"]
    assert "action_label" not in entry


def test_load_motion_metadata_fast_fails_when_tag_missing(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "motion_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 5,
                "total_clips": 1,
                "motions": {
                    "Cat_Run_001.npy": {"object_type": "Cat", "species_label": "cat"},
                },
            }
        ),
        encoding="utf-8",
    )
    # Sidecar exists but is missing the clip → must fail fast.
    _write_action_tags(dataset_dir, {"Dog_Jump_002.npy": ["jump"]})

    with pytest.raises(SystemExit):
        load_motion_metadata(dataset_dir)


def test_regenerate_dataset_artifacts_full_refresh_rewrites_incremental_dataset(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "dataset"
    motions_dir = dataset_dir / "motions"
    inspection_dir = dataset_dir / "joint_name_inspection"
    motions_dir.mkdir(parents=True)
    inspection_dir.mkdir(parents=True)

    np.save(motions_dir / "Cat_Run_001.npy", np.zeros((3, 2, 3), dtype=np.float32))
    np.save(motions_dir / "Dog_Jump_002.npy", np.zeros((5, 4, 3), dtype=np.float32))
    np.save(
        dataset_dir / "cond.npy",
        {
            "Cat": _make_cond_entry("Cat"),
            "Dog": _make_cond_entry("Dog"),
            "Stale": _make_cond_entry("Stale"),
        },
    )
    write_motion_metadata(
        dataset_dir,
        {
            "Cat_Run_001.npy": {
                "object_type": "Cat",
                "action_label": "legacy cat",
                "is_loop": True,
                "translation_root_index": 1,
                "motion_source": "anim_dir",
                "source_fbx_path": "cat.fbx",
            },
            "Dog_Jump_002.npy": {
                "object_type": "Dog",
                "action_label": "legacy dog",
                "is_loop": False,
                "translation_root_index": 0,
                "motion_source": "retarget",
            },
            "Stale_Idle_003.npy": {"object_type": "Stale", "action_label": "legacy stale"},
        },
        total_clips=3,
    )
    _write_action_tags(
        dataset_dir,
        {
            "Cat_Run_001.npy": ["locomotion"],
            "Dog_Jump_002.npy": ["jump"],
            "Stale_Idle_003.npy": ["idle"],
        },
    )
    (inspection_dir / "Cat.json").write_text('{"object_type": "Cat", "stale": true}', encoding="utf-8")
    (inspection_dir / "Dog.json").write_text('{"object_type": "Dog", "stale": true}', encoding="utf-8")
    (inspection_dir / "Stale.json").write_text('{"object_type": "Stale", "stale": true}', encoding="utf-8")
    (dataset_dir / "joint_name_collision_report.json").write_text('{"stale": true}', encoding="utf-8")
    (dataset_dir / "positions_error_rate.txt").write_text(
        "\n".join(
            [
                "Position squared error per source clip: previous_run: 0.100000",
                "Cat run clip: 0.010000",
                "Dog jump clip: 0.020000",
                "Stale idle clip: 0.030000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_attach(cond, save_dir, t5_name="t5-base", write_collision_report=True):
        inspection_output_dir = Path(save_dir) / "joint_name_inspection"
        inspection_output_dir.mkdir(parents=True, exist_ok=True)
        for object_type, object_cond in cond.items():
            embedding_count = len(object_cond["joints_names"])
            object_cond["joints_names_embs"] = np.ones((embedding_count, 1), dtype=np.float32)
            object_cond["joints_names_embs_meta"] = {
                "t5_name": t5_name,
                "schema_version": 1,
                "embedding_dim": 1,
                "embedding_texts": list(object_cond["joints_names"]),
            }
            (inspection_output_dir / f"{object_type}.json").write_text(
                json.dumps({"object_type": object_type, "encoded": True}),
                encoding="utf-8",
            )

    def fake_write_collision_report(cond, save_dir):
        report_path = Path(save_dir) / "joint_name_collision_report.json"
        report_path.write_text(
            json.dumps({"num_objects": len(cond), "objects": sorted(cond)}),
            encoding="utf-8",
        )
        return []

    monkeypatch.setattr(regenerate_dataset_artifacts_module, "attach_t5_embeddings_to_cond", fake_attach)
    monkeypatch.setattr(regenerate_dataset_artifacts_module, "write_joint_name_collision_report", fake_write_collision_report)

    dataset_dir_path = regenerate_dataset_artifacts_module.regenerate_dataset_artifacts(dataset_dir, t5_model="fake-t5")

    assert dataset_dir_path == dataset_dir.resolve()

    regenerated_cond = dict(np.load(dataset_dir / "cond.npy", allow_pickle=True).item())
    assert sorted(regenerated_cond) == ["Cat", "Dog"]
    assert regenerated_cond["Cat"]["joints_names_embs_meta"]["t5_name"] == "fake-t5"
    assert regenerated_cond["Dog"]["joints_names_embs_meta"]["t5_name"] == "fake-t5"
    assert regenerated_cond["Cat"]["translation_root_index"] == 1
    assert regenerated_cond["Dog"]["translation_root_index"] == 0

    motion_metadata = load_motion_metadata(dataset_dir)
    assert sorted(motion_metadata) == ["Cat_Run_001.npy", "Dog_Jump_002.npy"]
    assert motion_metadata["Cat_Run_001.npy"]["object_type"] == "Cat"
    assert motion_metadata["Cat_Run_001.npy"]["action_tags"] == ["locomotion"]
    assert motion_metadata["Cat_Run_001.npy"]["is_loop"] is True
    assert motion_metadata["Cat_Run_001.npy"]["translation_root_index"] == 1
    assert motion_metadata["Cat_Run_001.npy"]["motion_source"] == "anim_dir"
    assert motion_metadata["Cat_Run_001.npy"]["source_fbx_path"] == "cat.fbx"
    assert motion_metadata["Dog_Jump_002.npy"]["object_type"] == "Dog"
    assert motion_metadata["Dog_Jump_002.npy"]["is_loop"] is False
    assert motion_metadata["Dog_Jump_002.npy"]["translation_root_index"] == 0
    assert motion_metadata["Dog_Jump_002.npy"]["motion_source"] == "retarget"

    assert sorted(path.stem for path in inspection_dir.glob("*.json")) == ["Cat", "Dog"]
    collision_report = json.loads((dataset_dir / "joint_name_collision_report.json").read_text(encoding="utf-8"))
    assert collision_report == {"num_objects": 2, "objects": ["Cat", "Dog"]}

    positions_error_lines = (dataset_dir / "positions_error_rate.txt").read_text(encoding="utf-8").splitlines()
    assert positions_error_lines[0] == "Position squared error per source clip:__artifact_regenerated__: 0.000000"
    assert "Cat run clip: 0.010000" in positions_error_lines
    assert "Dog jump clip: 0.020000" in positions_error_lines
    assert all("Stale" not in line for line in positions_error_lines)

    metadata_summary = (dataset_dir / "metadata.txt").read_text(encoding="utf-8")
    assert "max joints: 4" in metadata_summary
    assert "total frames: 8" in metadata_summary
    assert "~~~~ objects_counts - Total: 2 ~~~~" in metadata_summary
    assert "Cat: 1" in metadata_summary
    assert "Dog: 1" in metadata_summary


def test_regenerate_dataset_artifacts_unifies_translation_root_index_per_object(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "dataset"
    motions_dir = dataset_dir / "motions"
    motions_dir.mkdir(parents=True)

    np.save(motions_dir / "Cat_Run_001.npy", np.zeros((3, 3, 13), dtype=np.float32))
    np.save(motions_dir / "Cat_Idle_002.npy", np.zeros((5, 3, 13), dtype=np.float32))
    np.save(
        dataset_dir / "cond.npy",
        {
            "Cat": {
                "object_type": "Cat",
                "joints_names": ["Root", "Mid", "Tip"],
                "parents": np.array([-1, 0, 1], dtype=np.int64),
                "offsets": np.zeros((3, 3), dtype=np.float32),
            },
        },
    )
    write_motion_metadata(
        dataset_dir,
        {
            "Cat_Run_001.npy": {
                "object_type": "Cat",
                "translation_root_index": 2,
                "motion_source": "anim_dir",
            },
            "Cat_Idle_002.npy": {
                "object_type": "Cat",
                "translation_root_index": 1,
                "motion_source": "retarget",
            },
        },
        total_clips=2,
    )
    _write_action_tags(
        dataset_dir,
        {"Cat_Run_001.npy": ["locomotion"], "Cat_Idle_002.npy": ["idle"]},
    )

    def fake_attach(cond, save_dir, t5_name="t5-base", write_collision_report=True):
        for object_cond in cond.values():
            joint_count = len(object_cond["joints_names"])
            object_cond["joints_names_embs"] = np.ones((joint_count, 1), dtype=np.float32)
            object_cond["joints_names_embs_meta"] = {"t5_name": t5_name}

    def fake_write_collision_report(cond, save_dir):
        return []

    monkeypatch.setattr(regenerate_dataset_artifacts_module, "attach_t5_embeddings_to_cond", fake_attach)
    monkeypatch.setattr(regenerate_dataset_artifacts_module, "write_joint_name_collision_report", fake_write_collision_report)

    regenerate_dataset_artifacts_module.regenerate_dataset_artifacts(dataset_dir, t5_model="fake-t5")

    regenerated_cond = dict(np.load(dataset_dir / "cond.npy", allow_pickle=True).item())
    assert regenerated_cond["Cat"]["translation_root_index"] == 1

    motion_metadata = load_motion_metadata(dataset_dir)
    assert motion_metadata["Cat_Run_001.npy"]["translation_root_index"] == 1
    assert motion_metadata["Cat_Idle_002.npy"]["translation_root_index"] == 1


def test_regenerate_dataset_artifacts_rebuilds_translation_root_when_metadata_missing(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "dataset"
    motions_dir = dataset_dir / "motions"
    motions_dir.mkdir(parents=True)

    np.save(motions_dir / "Cat_Run_001.npy", np.zeros((3, 3, 13), dtype=np.float32))
    np.save(
        dataset_dir / "cond.npy",
        {
            "Cat": {
                "object_type": "Cat",
                "joints_names": ["Root", "Mid", "Tip"],
                "parents": np.array([-1, 0, 1], dtype=np.int64),
                "offsets": np.zeros((3, 3), dtype=np.float32),
            },
        },
    )
    write_motion_metadata(
        dataset_dir,
        {
            "Cat_Run_001.npy": {
                "object_type": "Cat",
                "translation_root_index": 2,
                "motion_source": "anim_dir",
            },
        },
        total_clips=1,
    )
    _write_action_tags(dataset_dir, {"Cat_Run_001.npy": ["locomotion"]})

    def fake_attach(cond, save_dir, t5_name="t5-base", write_collision_report=True):
        for object_cond in cond.values():
            joint_count = len(object_cond["joints_names"])
            object_cond["joints_names_embs"] = np.ones((joint_count, 1), dtype=np.float32)
            object_cond["joints_names_embs_meta"] = {"t5_name": t5_name}

    def fake_write_collision_report(cond, save_dir):
        return []

    monkeypatch.setattr(regenerate_dataset_artifacts_module, "attach_t5_embeddings_to_cond", fake_attach)
    monkeypatch.setattr(regenerate_dataset_artifacts_module, "write_joint_name_collision_report", fake_write_collision_report)

    regenerate_dataset_artifacts_module.regenerate_dataset_artifacts(dataset_dir, t5_model="fake-t5")

    regenerated_cond = dict(np.load(dataset_dir / "cond.npy", allow_pickle=True).item())
    assert regenerated_cond["Cat"]["translation_root_index"] == 2

    motion_metadata = load_motion_metadata(dataset_dir)
    assert motion_metadata["Cat_Run_001.npy"]["translation_root_index"] == 2


def test_regenerate_dataset_artifacts_uses_majority_root_not_minimum(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "dataset"
    motions_dir = dataset_dir / "motions"
    motions_dir.mkdir(parents=True)

    for idx in range(4):
        np.save(motions_dir / f"Bear_Run_{idx:03d}.npy", np.zeros((idx + 3, 3, 13), dtype=np.float32))

    np.save(
        dataset_dir / "cond.npy",
        {
            "Bear": {
                "object_type": "Bear",
                "joints_names": ["Hips", "Pelvis", "Leg"],
                "parents": np.array([-1, 0, 1], dtype=np.int64),
                "offsets": np.zeros((3, 3), dtype=np.float32),
            },
        },
    )
    write_motion_metadata(
        dataset_dir,
        {f"Bear_Run_{idx:03d}.npy": {"object_type": "Bear", "translation_root_index": 1} for idx in range(4)},
        total_clips=4,
    )
    _write_action_tags(
        dataset_dir,
        {f"Bear_Run_{idx:03d}.npy": ["locomotion"] for idx in range(4)},
    )

    def fake_attach(cond, save_dir, t5_name="t5-base", write_collision_report=True):
        for object_cond in cond.values():
            joint_count = len(object_cond["joints_names"])
            object_cond["joints_names_embs"] = np.ones((joint_count, 1), dtype=np.float32)
            object_cond["joints_names_embs_meta"] = {"t5_name": t5_name}

    def fake_write_collision_report(cond, save_dir):
        return []

    monkeypatch.setattr(regenerate_dataset_artifacts_module, "attach_t5_embeddings_to_cond", fake_attach)
    monkeypatch.setattr(regenerate_dataset_artifacts_module, "write_joint_name_collision_report", fake_write_collision_report)

    regenerate_dataset_artifacts_module.regenerate_dataset_artifacts(dataset_dir, t5_model="fake-t5")

    regenerated_cond = dict(np.load(dataset_dir / "cond.npy", allow_pickle=True).item())
    assert regenerated_cond["Bear"]["translation_root_index"] == 1

    motion_metadata = load_motion_metadata(dataset_dir)
    assert all(entry["translation_root_index"] == 1 for entry in motion_metadata.values())


def test_regenerate_dataset_artifacts_resolves_active_objects_without_label_inference(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "dataset"
    motions_dir = dataset_dir / "motions"
    motions_dir.mkdir(parents=True)

    np.save(motions_dir / "Cat_Run_001.npy", np.zeros((3, 2, 3), dtype=np.float32))
    np.save(motions_dir / "Dog_Jump_002.npy", np.zeros((5, 2, 3), dtype=np.float32))
    np.save(
        dataset_dir / "cond.npy",
        {
            "Cat": _make_cond_entry("Cat"),
            "Dog": _make_cond_entry("Dog"),
        },
    )
    write_motion_metadata(
        dataset_dir,
        {
            "Cat_Run_001.npy": {"object_type": "Cat", "translation_root_index": 0},
            "Dog_Jump_002.npy": {"object_type": "Dog", "translation_root_index": 0},
        },
        total_clips=2,
    )
    _write_action_tags(
        dataset_dir,
        {
            "Cat_Run_001.npy": ["locomotion"],
            "Dog_Jump_002.npy": ["locomotion"],
        },
    )

    def fake_attach(cond, save_dir, t5_name="t5-base", write_collision_report=True):
        for object_cond in cond.values():
            joint_count = len(object_cond["joints_names"])
            object_cond["joints_names_embs"] = np.ones((joint_count, 1), dtype=np.float32)
            object_cond["joints_names_embs_meta"] = {"t5_name": t5_name}

    def fake_write_collision_report(cond, save_dir):
        return []

    calls: list[str] = []

    def fake_build_motion_labels(object_type, motion_name=None, source_file=None):
        calls.append(motion_name)
        return {
            "object_type": object_type,
            "species_label": str(object_type).lower(),
            "motion_name": motion_name,
        }

    monkeypatch.setattr(regenerate_dataset_artifacts_module, "attach_t5_embeddings_to_cond", fake_attach)
    monkeypatch.setattr(regenerate_dataset_artifacts_module, "write_joint_name_collision_report", fake_write_collision_report)
    monkeypatch.setattr(
        regenerate_dataset_artifacts_module,
        "build_motion_labels",
        fake_build_motion_labels,
    )

    regenerate_dataset_artifacts_module.regenerate_dataset_artifacts(dataset_dir, t5_model="fake-t5")

    assert calls == ["Cat_Run_001.npy", "Dog_Jump_002.npy"]


def test_create_data_samples_writes_seed_artifacts_for_regeneration(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "dataset"

    def fake_prepare_object_outputs(object_type, max_joints, face_joints=None, fbxs_dir=None, t_pos_path=None, max_files=None, raw_data_dir=None, filter_min_length=10, resample_min_length=20, skip_source_paths=None):
        return {
            'object_type': object_type,
            'object_cond': _make_cond_entry(object_type),
            'errors': {'Cat run clip': 0.010000},
            'max_joints': 2,
            'results': [],
            'files_counter': 0,
            'frames_counter': 0,
            'face_joints': face_joints,
            'motion_errors': [],
        }

    def fake_write_object_outputs(save_dir, object_payload, files_counter, action_start_counts=None):
        motions_dir = Path(save_dir) / 'motions'
        motion_name = f"{object_payload['object_type']}_Run_001.npy"
        np.save(motions_dir / motion_name, np.zeros((3, 2, 3), dtype=np.float32))
        return files_counter + 1, 3, {
            motion_name: {
                'object_type': object_payload['object_type'],
                'action_label': 'Run',
                'motion_name': motion_name,
                'translation_root_index': 1,
            }
        }

    monkeypatch.setattr(dataset_pipeline_mod, '_prepare_object_outputs', fake_prepare_object_outputs)
    monkeypatch.setattr(dataset_pipeline_mod, '_write_object_outputs', fake_write_object_outputs)

    dataset_pipeline_mod.create_data_samples(
        objects=['Cat'],
        dataset_dir=str(dataset_dir),
        object_workers=1,
    )

    seed_cond = dict(np.load(dataset_dir / 'cond.npy', allow_pickle=True).item())
    assert sorted(seed_cond) == ['Cat']
    assert 'joints_names_embs' not in seed_cond['Cat']

    _write_action_tags(dataset_dir, {"Cat_Run_001.npy": ["locomotion"]})
    motion_metadata = load_motion_metadata(dataset_dir)
    assert motion_metadata['Cat_Run_001.npy']['translation_root_index'] == 1

    positions_error_lines = (dataset_dir / 'positions_error_rate.txt').read_text(encoding='utf-8').splitlines()
    assert positions_error_lines[0] == 'Position squared error per source clip:'
    assert 'Cat run clip: 0.010000' in positions_error_lines

    assert not (dataset_dir / 'metadata.txt').exists()
    assert not (dataset_dir / 'joint_name_inspection').exists()


def test_create_data_samples_raises_preprocess_error_instead_of_exit(monkeypatch, tmp_path):
    dataset_dir = tmp_path / 'dataset'

    def fake_prepare_object_outputs(object_type, max_joints, face_joints=None, fbxs_dir=None, t_pos_path=None, max_files=None, raw_data_dir=None, filter_min_length=10, resample_min_length=20, skip_source_paths=None):
        return {
            'object_type': object_type,
            'object_cond': _make_cond_entry(object_type),
            'errors': {},
            'max_joints': 2,
            'results': [],
            'files_counter': 0,
            'frames_counter': 0,
            'face_joints': face_joints,
            'motion_errors': ['boom'],
        }

    monkeypatch.setattr(dataset_pipeline_mod, '_prepare_object_outputs', fake_prepare_object_outputs)

    with pytest.raises(dataset_pipeline_mod.DatasetPreprocessingError) as exc_info:
        dataset_pipeline_mod.create_data_samples(
            objects=['Cat'],
            dataset_dir=str(dataset_dir),
            object_workers=1,
        )

    assert exc_info.value.motion_errors == ('boom',)


def test_run_preprocessing_calls_create_data_samples_directly(monkeypatch):
    captured: dict[str, object] = {}

    def fake_create_data_samples(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(motion_process_mod, 'create_data_samples', fake_create_data_samples)

    ret = preprocess_and_validate_module.run_preprocessing(
        ['Horse', 'Raptor'],
        object_workers=4,
        raw_data_dir='raw_dir',
        dataset_dir='dataset_dir',
        incremental=True,
    )

    assert ret == 0
    assert list(captured['objects']) == ['Horse', 'Raptor']
    assert captured['dataset_dir'] == 'dataset_dir'
    assert captured['raw_data_dir'] == 'raw_dir'
    assert captured['object_workers'] == 4
    assert captured['incremental'] is True


def test_find_new_source_files_detects_only_unprocessed_sources(monkeypatch, tmp_path):
    raw = tmp_path / 'raw'
    (raw / 'Cat').mkdir(parents=True)
    (raw / 'Dog').mkdir(parents=True)
    for name in ('Cat_Walk.fbx', 'Cat_Run.fbx'):
        (raw / 'Cat' / name).write_text('x')
    (raw / 'Dog' / 'Dog_Idle.fbx').write_text('x')

    # Bypass filename heuristics so the test exercises source-dedup, not name rules.
    monkeypatch.setattr(dataset_pipeline_mod, 'should_skip_anim', lambda f, o: False)

    dataset_dir = tmp_path / 'dataset'
    dataset_dir.mkdir()
    # Cat_Walk.fbx already produced a clip; Cat_Run.fbx is new; Dog is entirely new.
    processed_src = str(raw / 'Cat' / 'Cat_Walk.fbx')
    write_motion_metadata(
        dataset_dir,
        {'Cat_Walk_1.npy': {'object_type': 'Cat', 'source_fbx_path': processed_src}},
        1,
    )

    result = dataset_pipeline_mod.find_new_source_files(['Cat', 'Dog'], str(dataset_dir), str(raw))

    assert set(result) == {'Cat', 'Dog'}
    assert [os.path.basename(p) for p in result['Cat']] == ['Cat_Run.fbx']
    assert [os.path.basename(p) for p in result['Dog']] == ['Dog_Idle.fbx']


def test_find_new_source_files_omits_fully_processed_objects(monkeypatch, tmp_path):
    raw = tmp_path / 'raw'
    (raw / 'Cat').mkdir(parents=True)
    (raw / 'Cat' / 'Cat_Walk.fbx').write_text('x')
    monkeypatch.setattr(dataset_pipeline_mod, 'should_skip_anim', lambda f, o: False)

    dataset_dir = tmp_path / 'dataset'
    dataset_dir.mkdir()
    write_motion_metadata(
        dataset_dir,
        {'Cat_Walk_1.npy': {'object_type': 'Cat', 'source_fbx_path': str(raw / 'Cat' / 'Cat_Walk.fbx')}},
        1,
    )

    assert dataset_pipeline_mod.find_new_source_files(['Cat'], str(dataset_dir), str(raw)) == {}


def test_recompute_object_stats_scopes_to_only_objects(tmp_path):
    motions = tmp_path / 'motions'
    motions.mkdir()
    cat = motions / 'Cat_Walk_1.npy'
    dog = motions / 'Dog_Idle_1.npy'
    # Motion tensors are (frames, joints, 13); get_mean_std indexes the 13-feature layout.
    np.save(cat, np.ones((4, 2, 13), dtype=np.float32))
    np.save(dog, np.ones((4, 2, 13), dtype=np.float32) * 5.0)

    dog_sentinel = np.array([[222.0]])
    rebuilt = {
        'Cat': {'mean': np.array([[111.0]]), 'std': np.array([[111.0]])},
        'Dog': {'mean': dog_sentinel.copy(), 'std': dog_sentinel.copy()},
    }

    regenerate_dataset_artifacts_module._recompute_object_stats(
        rebuilt, [cat, dog], only_objects={'Cat'}
    )

    # Cat was recomputed over its clip (mean now has the real (J, feat) shape) ...
    assert rebuilt['Cat']['mean'].shape == (2, 13)
    # ... while untouched Dog keeps its carried-forward sentinel untouched.
    assert np.array_equal(rebuilt['Dog']['mean'], dog_sentinel)


def test_create_data_samples_incremental_skips_done_sources_and_merges(monkeypatch, tmp_path):
    dataset_dir = tmp_path / 'dataset'
    (dataset_dir / 'motions').mkdir(parents=True)
    (dataset_dir / 'bvhs').mkdir(parents=True)

    # Existing dataset: Cat (Walk_1 from Cat_Walk.fbx) and an untouched Dog object.
    done_source = str(tmp_path / 'raw' / 'Cat' / 'Cat_Walk.fbx')
    np.save(
        dataset_dir / 'cond.npy',
        {'Cat': _make_cond_entry('Cat'), 'Dog': _make_cond_entry('Dog')},
    )
    write_motion_metadata(
        dataset_dir,
        {
            'Cat_Walk_1.npy': {'object_type': 'Cat', 'source_fbx_path': done_source, 'motion_name': 'Cat_Walk_1.npy'},
            'Dog_Idle_1.npy': {'object_type': 'Dog', 'source_fbx_path': str(tmp_path / 'raw' / 'Dog' / 'Dog_Idle.fbx'), 'motion_name': 'Dog_Idle_1.npy'},
        },
        2,
    )

    captured: dict[str, object] = {}

    def fake_prepare(object_type, max_joints, face_joints=None, fbxs_dir=None, t_pos_path=None,
                     max_files=None, raw_data_dir=None, filter_min_length=10, resample_min_length=20,
                     skip_source_paths=None):
        captured['skip_source_paths'] = set(skip_source_paths or set())
        return {
            'object_type': object_type,
            'object_cond': _make_cond_entry(object_type),
            'errors': {},
            'max_joints': 2,
            'results': [],
            'files_counter': 0,
            'frames_counter': 0,
            'face_joints': face_joints,
            'motion_errors': [],
        }

    def fake_write(save_dir, payload, files_counter, action_start_counts=None):
        captured['action_start_counts'] = dict(action_start_counts or {})
        obj = payload['object_type']
        idx = (action_start_counts or {}).get('Walk', 0) + 1
        name = f"{obj}_Walk_{idx}.npy"
        np.save(Path(save_dir) / 'motions' / name, np.zeros((3, 2, 3), dtype=np.float32))
        return files_counter + 1, 3, {name: {'object_type': obj, 'motion_name': name}}

    monkeypatch.setattr(dataset_pipeline_mod, '_prepare_object_outputs', fake_prepare)
    monkeypatch.setattr(dataset_pipeline_mod, '_write_object_outputs', fake_write)

    dataset_pipeline_mod.create_data_samples(
        objects=['Cat'],
        dataset_dir=str(dataset_dir),
        object_workers=1,
        incremental=True,
    )

    # Already-processed source handed to the worker as a skip, numbered above existing clip.
    assert captured['skip_source_paths'] == {os.path.realpath(done_source)}
    assert captured['action_start_counts'] == {'Walk': 1}

    # cond.npy keeps the untouched Dog and refreshes Cat.
    merged_cond = dict(np.load(dataset_dir / 'cond.npy', allow_pickle=True).item())
    assert sorted(merged_cond) == ['Cat', 'Dog']

    # Existing clips preserved; the new clip is appended without colliding.
    merged_meta = dataset_pipeline_mod._load_motion_metadata_raw(dataset_dir)
    assert set(merged_meta) == {'Cat_Walk_1.npy', 'Cat_Walk_2.npy', 'Dog_Idle_1.npy'}


def test_process_skeleton_retarget_branch_writes_translation_root_metadata(monkeypatch, tmp_path):
    motion = np.zeros((4, 3, 13), dtype=np.float32)
    motion[:, :, 3:9] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    trajectory_x = np.arange(4, dtype=np.float32)
    motion[:, 0, 0] = -trajectory_x
    motion[:, 0, 2] = 1.0
    motion[:, 1, 0] = -trajectory_x
    motion[:, 1, 1] = 1.0
    motion[:, 1, 2] = 1.0
    motion[:, 2, 1] = 1.0
    motion[:-1, 2, 9] = 1.0

    motion_path = tmp_path / 'Dragon_RunLoop_001.npy'
    np.save(motion_path, motion)

    captured: dict[str, object] = {}

    def fake_write_dataset_artifacts(save_dir, cond, motion_metadata, objects_counter, max_joints, files_counter, frames_counter, squared_positions_error):
        captured['save_dir'] = save_dir
        captured['cond'] = cond
        captured['motion_metadata'] = motion_metadata
        captured['objects_counter'] = objects_counter
        captured['max_joints'] = max_joints
        captured['files_counter'] = files_counter
        captured['frames_counter'] = frames_counter

    monkeypatch.setattr(dataset_pipeline_mod, '_write_dataset_artifacts', fake_write_dataset_artifacts)

    dataset_pipeline_mod.process_skeleton(
        'Dragon',
        None,
        str(tmp_path / 'out'),
        'unused',
        motions_from_npys=[str(motion_path)],
        target_cond_partial={
            'object_type': 'Dragon',
            'joints_names': ['Root', 'Mid', 'Tip'],
            'parents': np.array([-1, 0, 1], dtype=np.int64),
            'offsets': np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, -1.0],
                ],
                dtype=np.float32,
            ),
        },
    )

    motion_metadata = captured['motion_metadata']
    assert 'Dragon_RunLoop_001.npy' in motion_metadata
    assert motion_metadata['Dragon_RunLoop_001.npy']['motion_name'] == 'Dragon_RunLoop_001.npy'
    assert motion_metadata['Dragon_RunLoop_001.npy']['object_type'] == 'Dragon'
    assert motion_metadata['Dragon_RunLoop_001.npy']['translation_root_index'] == 2
    assert motion_metadata['Dragon_RunLoop_001.npy']['motion_source'] == 'retarget'


def test_update_anim_dir_preserves_other_objects(monkeypatch, tmp_path):
    dataset_dir = tmp_path / 'dataset'
    motions_dir = dataset_dir / 'motions'
    bvhs_dir = dataset_dir / 'bvhs'
    motions_dir.mkdir(parents=True)
    bvhs_dir.mkdir(parents=True)

    np.save(dataset_dir / 'cond.npy', {
        'Cat': _make_cond_entry('Cat'),
        'Dog': _make_cond_entry('Dog'),
    })
    np.save(motions_dir / 'Cat_Run_001.npy', np.zeros((1, 1, 1), dtype=np.float32))
    np.save(motions_dir / 'Dog_Run_001.npy', np.zeros((1, 1, 1), dtype=np.float32))
    (bvhs_dir / 'Cat_Run_001.bvh').write_text('cat', encoding='utf-8')
    (bvhs_dir / 'Dog_Run_001.bvh').write_text('dog', encoding='utf-8')
    write_motion_metadata(
        dataset_dir,
        {
            'Cat_Run_001.npy': {'object_type': 'Cat', 'source_fbx_path': 'cat.fbx'},
            'Dog_Run_001.npy': {'object_type': 'Dog', 'source_fbx_path': 'dog.fbx'},
        },
        total_clips=2,
    )
    _write_action_tags(
        dataset_dir,
        {
            'Cat_Run_001.npy': ['locomotion'],
            'Dog_Run_001.npy': ['locomotion'],
            'Cat_New_002.npy': ['locomotion'],
        },
    )

    def fake_process_object(object_name, *args, **kwargs):
        np.save(motions_dir / 'Cat_New_002.npy', np.zeros((1, 1, 1), dtype=np.float32))
        (bvhs_dir / 'Cat_New_002.bvh').write_text('new', encoding='utf-8')
        return 0, 0, 0, _make_cond_entry(object_name), {
            'Cat_New_002.npy': {
                'object_type': object_name,
                'source_fbx_path': 'cat-new.fbx',
                'motion_source': 'anim_dir',
            }
        }

    monkeypatch.setattr(dataset_pipeline_mod, 'process_object', fake_process_object)

    dataset_pipeline_mod._update_anim_dir('Cat', None, str(dataset_dir), 'tpose', 'anim_dir')

    assert sorted(path.name for path in motions_dir.glob('*.npy')) == ['Cat_New_002.npy', 'Cat_Run_001.npy', 'Dog_Run_001.npy']
    assert sorted(path.name for path in bvhs_dir.glob('*.bvh')) == ['Cat_New_002.bvh', 'Cat_Run_001.bvh', 'Dog_Run_001.bvh']

    motion_metadata = load_motion_metadata(dataset_dir)
    assert sorted(motion_metadata) == ['Cat_New_002.npy', 'Cat_Run_001.npy', 'Dog_Run_001.npy']
    cond = dict(np.load(dataset_dir / 'cond.npy', allow_pickle=True).item())
    assert sorted(cond) == ['Cat', 'Dog']


def test_update_anim_dir_replaces_only_matching_sources(monkeypatch, tmp_path):
    dataset_dir = tmp_path / 'dataset'
    motions_dir = dataset_dir / 'motions'
    bvhs_dir = dataset_dir / 'bvhs'
    motions_dir.mkdir(parents=True)
    bvhs_dir.mkdir(parents=True)

    np.save(dataset_dir / 'cond.npy', {'Dragon': _make_cond_entry('Dragon')})
    np.save(motions_dir / 'Dragon_A_001.npy', np.zeros((1, 1, 1), dtype=np.float32))
    np.save(motions_dir / 'Dragon_B_001.npy', np.zeros((1, 1, 1), dtype=np.float32))
    (bvhs_dir / 'Dragon_A_001.bvh').write_text('A', encoding='utf-8')
    (bvhs_dir / 'Dragon_B_001.bvh').write_text('B', encoding='utf-8')
    write_motion_metadata(
        dataset_dir,
        {
            'Dragon_A_001.npy': {
                'object_type': 'Dragon',
                'source_fbx_path': str((tmp_path / 'A.fbx').resolve()),
                'motion_source': 'anim_dir',
            },
            'Dragon_B_001.npy': {
                'object_type': 'Dragon',
                'source_fbx_path': str((tmp_path / 'B.fbx').resolve()),
                'motion_source': 'anim_dir',
            },
        },
        total_clips=2,
    )
    _write_action_tags(
        dataset_dir,
        {
            'Dragon_A_001.npy': ['locomotion'],
            'Dragon_B_001.npy': ['locomotion'],
            'Dragon_B_002.npy': ['locomotion'],
            'Dragon_C_003.npy': ['locomotion'],
        },
    )

    def fake_process_object(object_name, *args, **kwargs):
        np.save(motions_dir / 'Dragon_B_002.npy', np.zeros((1, 1, 1), dtype=np.float32))
        np.save(motions_dir / 'Dragon_C_003.npy', np.zeros((1, 1, 1), dtype=np.float32))
        (bvhs_dir / 'Dragon_B_002.bvh').write_text('B2', encoding='utf-8')
        (bvhs_dir / 'Dragon_C_003.bvh').write_text('C', encoding='utf-8')
        return 0, 0, 0, _make_cond_entry(object_name), {
            'Dragon_B_002.npy': {
                'object_type': object_name,
                'source_fbx_path': str((tmp_path / 'B.fbx').resolve()),
                'motion_source': 'anim_dir',
            },
            'Dragon_C_003.npy': {
                'object_type': object_name,
                'source_fbx_path': str((tmp_path / 'C.fbx').resolve()),
                'motion_source': 'anim_dir',
            },
        }

    monkeypatch.setattr(dataset_pipeline_mod, 'process_object', fake_process_object)

    dataset_pipeline_mod._update_anim_dir('Dragon', None, str(dataset_dir), 'tpose', 'anim_dir')

    assert sorted(path.name for path in motions_dir.glob('*.npy')) == [
        'Dragon_A_001.npy',
        'Dragon_B_002.npy',
        'Dragon_C_003.npy',
    ]
    assert sorted(path.name for path in bvhs_dir.glob('*.bvh')) == [
        'Dragon_A_001.bvh',
        'Dragon_B_002.bvh',
        'Dragon_C_003.bvh',
    ]

    motion_metadata = load_motion_metadata(dataset_dir)
    assert sorted(motion_metadata) == [
        'Dragon_A_001.npy',
        'Dragon_B_002.npy',
        'Dragon_C_003.npy',
    ]


def test_update_anim_dir_full_rerun_replaces_old_versions(monkeypatch, tmp_path):
    dataset_dir = tmp_path / 'dataset'
    motions_dir = dataset_dir / 'motions'
    bvhs_dir = dataset_dir / 'bvhs'
    motions_dir.mkdir(parents=True)
    bvhs_dir.mkdir(parents=True)

    anim_dir = tmp_path / 'anim_dir'
    anim_dir.mkdir()
    source_a = str((anim_dir / 'A.fbx').resolve())
    source_b = str((anim_dir / 'B.fbx').resolve())
    source_c = str((anim_dir / 'C.fbx').resolve())
    for source_path in (source_a, source_b, source_c):
        Path(source_path).write_text('fbx', encoding='utf-8')

    np.save(dataset_dir / 'cond.npy', {'Dragon': _make_cond_entry('Dragon')})
    np.save(motions_dir / 'Dragon_A_001.npy', np.zeros((1, 1, 1), dtype=np.float32))
    np.save(motions_dir / 'Dragon_B_001.npy', np.zeros((1, 1, 1), dtype=np.float32))
    (bvhs_dir / 'Dragon_A_001.bvh').write_text('A-old', encoding='utf-8')
    (bvhs_dir / 'Dragon_B_001.bvh').write_text('B-old', encoding='utf-8')
    write_motion_metadata(
        dataset_dir,
        {
            'Dragon_A_001.npy': {
                'object_type': 'Dragon',
                'source_fbx_path': source_a,
                'motion_source': 'anim_dir',
            },
            'Dragon_B_001.npy': {
                'object_type': 'Dragon',
                'source_fbx_path': source_b,
                'motion_source': 'anim_dir',
            },
        },
        total_clips=2,
    )
    _write_action_tags(
        dataset_dir,
        {
            'Dragon_A_001.npy': ['locomotion'],
            'Dragon_B_001.npy': ['locomotion'],
            'Dragon_A_003.npy': ['locomotion'],
            'Dragon_B_004.npy': ['locomotion'],
            'Dragon_C_005.npy': ['locomotion'],
        },
    )

    def fake_process_object(object_name, *args, **kwargs):
        np.save(motions_dir / 'Dragon_A_003.npy', np.zeros((1, 1, 1), dtype=np.float32))
        np.save(motions_dir / 'Dragon_B_004.npy', np.zeros((1, 1, 1), dtype=np.float32))
        np.save(motions_dir / 'Dragon_C_005.npy', np.zeros((1, 1, 1), dtype=np.float32))
        (bvhs_dir / 'Dragon_A_003.bvh').write_text('A-new', encoding='utf-8')
        (bvhs_dir / 'Dragon_B_004.bvh').write_text('B-new', encoding='utf-8')
        (bvhs_dir / 'Dragon_C_005.bvh').write_text('C-new', encoding='utf-8')
        return 0, 0, 0, _make_cond_entry(object_name), {
            'Dragon_A_003.npy': {
                'object_type': object_name,
                'source_fbx_path': source_a,
                'motion_source': 'anim_dir',
            },
            'Dragon_B_004.npy': {
                'object_type': object_name,
                'source_fbx_path': source_b,
                'motion_source': 'anim_dir',
            },
            'Dragon_C_005.npy': {
                'object_type': object_name,
                'source_fbx_path': source_c,
                'motion_source': 'anim_dir',
            },
        }

    monkeypatch.setattr(dataset_pipeline_mod, 'process_object', fake_process_object)

    dataset_pipeline_mod._update_anim_dir('Dragon', None, str(dataset_dir), 'tpose', str(anim_dir))

    assert sorted(path.name for path in motions_dir.glob('*.npy')) == [
        'Dragon_A_003.npy',
        'Dragon_B_004.npy',
        'Dragon_C_005.npy',
    ]
    assert sorted(path.name for path in bvhs_dir.glob('*.bvh')) == [
        'Dragon_A_003.bvh',
        'Dragon_B_004.bvh',
        'Dragon_C_005.bvh',
    ]

    motion_metadata = load_motion_metadata(dataset_dir)
    assert sorted(motion_metadata) == [
        'Dragon_A_003.npy',
        'Dragon_B_004.npy',
        'Dragon_C_005.npy',
    ]


def test_update_anim_dir_rejects_untracked_target_motions(monkeypatch, tmp_path):
    dataset_dir = tmp_path / 'dataset'
    motions_dir = dataset_dir / 'motions'
    bvhs_dir = dataset_dir / 'bvhs'
    motions_dir.mkdir(parents=True)
    bvhs_dir.mkdir(parents=True)

    np.save(motions_dir / 'Cat_Old_001.npy', np.zeros((1, 1, 1), dtype=np.float32))
    (bvhs_dir / 'Cat_Old_001.bvh').write_text('old', encoding='utf-8')

    def fail_process_object(*args, **kwargs):
        raise AssertionError('process_object should not run when update metadata is unsafe')

    monkeypatch.setattr(dataset_pipeline_mod, 'process_object', fail_process_object)

    with pytest.raises(RuntimeError, match='missing from motion_metadata.json'):
        dataset_pipeline_mod._update_anim_dir('Cat', None, str(dataset_dir), 'tpose', 'anim_dir')

    assert (motions_dir / 'Cat_Old_001.npy').exists()
    assert (bvhs_dir / 'Cat_Old_001.bvh').exists()


def test_process_new_skeleton_rejects_unsafe_anim_dir_update(monkeypatch, tmp_path):
    dataset_dir = tmp_path / 'dataset'
    motions_dir = dataset_dir / 'motions'
    motions_dir.mkdir(parents=True)

    np.save(dataset_dir / 'cond.npy', {'Cat': _make_cond_entry('Cat')})
    np.save(motions_dir / 'Cat_Old_001.npy', np.zeros((1, 1, 1), dtype=np.float32))

    args = SimpleNamespace(
        save_dir=str(dataset_dir),
        update=True,
        tpos_path='dragon.glb',
        anim_dir='anim_dir',
        object_type='Cat',
        face_joints_names=None,
        retarget_top_k=None,
        donor_skeletons=None,
        training_cond_path='unused',
        crop_enabled=False,
        species_tags=None,
    )

    monkeypatch.setattr(process_new_skeleton_module, 'process_new_skeleton_args', lambda: args)
    monkeypatch.setattr(
        process_new_skeleton_module,
        'process_skeleton',
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('process_skeleton should not run')),
    )

    with pytest.raises(SystemExit, match='motion_metadata.json'):
        process_new_skeleton_module.main()
