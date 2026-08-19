import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.truebones.truebones_utils.motion_labels import load_motion_metadata, write_motion_metadata
from data_loaders.truebones.truebones_utils import dataset_pipeline as dataset_pipeline_mod
from data_loaders.truebones.truebones_utils import motion_process as motion_process_mod

from tools import regenerate_dataset_artifacts as regenerate_dataset_artifacts_module
import preprocess_and_validate as preprocess_and_validate_module


def _make_cond_entry(object_type: str) -> dict[str, object]:
    return {
        "object_type": object_type,
        "joints_names": ["Root", "Tail"],
        "parents": np.array([-1, 0], dtype=np.int64),
        "offsets": np.zeros((2, 3), dtype=np.float32),
        "rest_pose": np.zeros((2, 13), dtype=np.float32),
    }


def _write_action_tags(dataset_dir, tags_by_clip):
    """Write the hand-maintained action_tags.jsonl sidecar for a temp dataset."""
    path = Path(dataset_dir) / "action_tags.jsonl"
    lines = [
        json.dumps({"clip": clip, "action_tags": list(tags)})
        for clip, tags in tags_by_clip.items()
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_species_tags(dataset_dir, species=("Cat", "Dog", "Stale")):
    """Write the species_tags.jsonl sidecar a temp dataset needs to regenerate.

    ``regenerate_dataset_artifacts`` reads the tag sidecar of the dataset it is
    pointed at (there is no in-code fallback), so a synthetic dataset must carry
    one for its species.
    """
    path = Path(dataset_dir) / "species_tags.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"species": name, "species_tags": ["Quadruped", "Medium", "Striding"]})
        for name in species
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

    _write_species_tags(dataset_dir)
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

    _write_species_tags(dataset_dir)
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

    _write_species_tags(dataset_dir)
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

    _write_species_tags(dataset_dir)
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

    _write_species_tags(dataset_dir)
    regenerate_dataset_artifacts_module.regenerate_dataset_artifacts(dataset_dir, t5_model="fake-t5")

    assert calls == ["Cat_Run_001.npy", "Dog_Jump_002.npy"]


def test_create_data_samples_writes_seed_artifacts_for_regeneration(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "dataset"

    def fake_prepare_object_outputs(object_type, max_joints, face_joints=None, fbxs_dir=None, t_pos_path=None, max_files=None, raw_data_dir=None, filter_min_length=10, resample_min_length=20, skip_source_paths=None):
        return {
            'object_type': object_type,
            'object_cond': _make_cond_entry(object_type),
            'tpose_reference_path': None,
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
            'tpose_reference_path': None,
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


def test_mark_object_feature_spaces():
    rebuilt = {
        'Cat': _make_cond_entry('Cat'),
        'Dog': _make_cond_entry('Dog'),
    }

    regenerate_dataset_artifacts_module._mark_object_feature_spaces(rebuilt)

    for object_type in ('Cat', 'Dog'):
        assert rebuilt[object_type]['feature_space'] == 'canonical_motion_v3'
        assert rebuilt[object_type]['physical_feature_space'] == 'hml_like_v_current'
        assert rebuilt[object_type]['rest_pos_ric_hml'].shape == (2, 3)


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
            'tpose_reference_path': None,
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


def _cond_entry_with_stats(object_type, mean_fill, std_fill):
    entry = _make_cond_entry(object_type)
    entry["canonical_feature_mean"] = np.full((13,), mean_fill, dtype=np.float32)
    entry["canonical_feature_std"] = np.full((13,), std_fill, dtype=np.float32)
    return entry


def test_merge_inherits_canonical_stats_from_same_object_subset(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True)
    # Existing quadruped (Cat) carries stats; add a new quadruped (Dog) without.
    np.save(dataset_dir / "cond.npy", {"Cat": _cond_entry_with_stats("Cat", 0.5, 2.0)})

    dataset_pipeline_mod._merge_object_into_cond(
        str(dataset_dir), "Dog", _make_cond_entry("Dog")
    )

    merged = dict(np.load(dataset_dir / "cond.npy", allow_pickle=True).item())
    np.testing.assert_allclose(merged["Dog"]["canonical_feature_mean"], np.full((13,), 0.5, dtype=np.float32))
    np.testing.assert_allclose(merged["Dog"]["canonical_feature_std"], np.full((13,), 2.0, dtype=np.float32))


def test_merge_fast_fails_when_no_same_object_subset_donor(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True)
    # Only a quadruped carries stats; a new winged species (Dragon) has no
    # same-object_subset donor, so borrowing would be OOD -> fast-fail.
    np.save(dataset_dir / "cond.npy", {"Cat": _cond_entry_with_stats("Cat", 0.5, 2.0)})

    with pytest.raises(ValueError, match="winged"):
        dataset_pipeline_mod._merge_object_into_cond(
            str(dataset_dir), "Dragon", _make_cond_entry("Dragon")
        )


def test_merge_update_preserves_species_own_prior_stats(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True)
    # Single-species dataset: rebuilding Dragon's cond without stats must still
    # preserve its own prior stats (no sibling to inherit from).
    np.save(dataset_dir / "cond.npy", {"Dragon": _cond_entry_with_stats("Dragon", 0.3, 1.5)})

    dataset_pipeline_mod._merge_object_into_cond(
        str(dataset_dir), "Dragon", _make_cond_entry("Dragon")
    )

    merged = dict(np.load(dataset_dir / "cond.npy", allow_pickle=True).item())
    np.testing.assert_allclose(merged["Dragon"]["canonical_feature_mean"], np.full((13,), 0.3, dtype=np.float32))
    np.testing.assert_allclose(merged["Dragon"]["canonical_feature_std"], np.full((13,), 1.5, dtype=np.float32))


