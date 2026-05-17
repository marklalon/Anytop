import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.truebones.truebones_utils.motion_labels import load_motion_metadata, write_motion_metadata
from data_loaders.truebones.truebones_utils import dataset_pipeline as dataset_pipeline_mod


_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "regenerate_dataset_artifacts.py"
_SPEC = importlib.util.spec_from_file_location("regenerate_dataset_artifacts_module", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
regenerate_dataset_artifacts_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(regenerate_dataset_artifacts_module)


def _make_cond_entry(object_type: str) -> dict[str, object]:
    return {
        "object_type": object_type,
        "joints_names": ["Root", "Tail"],
        "parents": np.array([-1, 0], dtype=np.int64),
        "offsets": np.zeros((2, 3), dtype=np.float32),
    }


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
            "Cat_Run_001.npy": {"object_type": "Cat", "action_label": "legacy cat", "is_loop": True, "translation_root_index": 1},
            "Dog_Jump_002.npy": {"object_type": "Dog", "action_label": "legacy dog", "translation_root_index": 0},
            "Stale_Idle_003.npy": {"object_type": "Stale", "action_label": "legacy stale"},
        },
        total_clips=3,
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

    def fake_attach(cond, save_dir, t5_name="t5-base", write_collision_report=True, force_reencode=True):
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

    monkeypatch.setattr(regenerate_dataset_artifacts_module, "_attach_joint_name_embeddings_to_cond", fake_attach)
    monkeypatch.setattr(regenerate_dataset_artifacts_module, "_write_joint_name_collision_report", fake_write_collision_report)

    dataset_dir_path = regenerate_dataset_artifacts_module.regenerate_dataset_artifacts(dataset_dir, t5_model="fake-t5")

    assert dataset_dir_path == dataset_dir.resolve()

    regenerated_cond = dict(np.load(dataset_dir / "cond.npy", allow_pickle=True).item())
    assert sorted(regenerated_cond) == ["Cat", "Dog"]
    assert regenerated_cond["Cat"]["joints_names_embs_meta"]["t5_name"] == "fake-t5"
    assert regenerated_cond["Dog"]["joints_names_embs_meta"]["t5_name"] == "fake-t5"

    motion_metadata = load_motion_metadata(dataset_dir)
    assert sorted(motion_metadata) == ["Cat_Run_001.npy", "Dog_Jump_002.npy"]
    assert motion_metadata["Cat_Run_001.npy"]["object_type"] == "Cat"
    assert motion_metadata["Cat_Run_001.npy"]["motion_name"] == "Cat_Run_001.npy"
    assert motion_metadata["Cat_Run_001.npy"]["is_loop"] is True
    assert motion_metadata["Cat_Run_001.npy"]["translation_root_index"] == 1
    assert motion_metadata["Dog_Jump_002.npy"]["object_type"] == "Dog"
    assert motion_metadata["Dog_Jump_002.npy"]["motion_name"] == "Dog_Jump_002.npy"
    assert motion_metadata["Dog_Jump_002.npy"]["is_loop"] is False
    assert motion_metadata["Dog_Jump_002.npy"]["translation_root_index"] == 0

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

    def fake_write_dataset_artifacts(save_dir, cond, motion_metadata, objects_counter, max_joints, files_counter, frames_counter, squared_positions_error, structural_prior_bank=None):
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
            'tpos_first_frame': np.zeros((3, 13), dtype=np.float32),
        },
    )

    motion_metadata = captured['motion_metadata']
    assert 'Dragon_RunLoop_001.npy' in motion_metadata
    assert motion_metadata['Dragon_RunLoop_001.npy']['motion_name'] == 'Dragon_RunLoop_001.npy'
    assert motion_metadata['Dragon_RunLoop_001.npy']['object_type'] == 'Dragon'
    assert motion_metadata['Dragon_RunLoop_001.npy']['translation_root_index'] == 2


# ---------------------------------------------------------------------------
# structural_norm_priors.npy rebuild from on-disk data
# ---------------------------------------------------------------------------

_MODULE_PREPROCESS_PATH = Path(__file__).resolve().parents[1] / "preprocess_and_validate.py"
_SPEC_PREPROCESS = importlib.util.spec_from_file_location("preprocess_and_validate_module", _MODULE_PREPROCESS_PATH)
assert _SPEC_PREPROCESS is not None and _SPEC_PREPROCESS.loader is not None
preprocess_module = importlib.util.module_from_spec(_SPEC_PREPROCESS)
_SPEC_PREPROCESS.loader.exec_module(preprocess_module)
PRIORS_FILE = preprocess_module.STRUCTURAL_NORM_PRIORS_FILE


def _dummy_subprocess_run(*args, **kwargs):
    """Fake subprocess.run that does nothing but succeed."""
    import subprocess
    return subprocess.CompletedProcess(args=[], returncode=0)


def test_rebuild_structural_prior_bank_normal(monkeypatch, tmp_path):
    """_rebuild_structural_prior_bank loads all motions from disk, builds the
    prior bank from scratch, saves it, and re-applies to all cond objects."""
    dataset_dir = tmp_path / "dataset"
    motions_dir = dataset_dir / "motions"
    motions_dir.mkdir(parents=True)

    # Write cond.npy with two objects.
    cond = {
        'Cat': {
            'joints_names': ['Root', 'Tail'],
            'parents': np.array([-1, 0], dtype=np.int64),
            'offsets': np.zeros((2, 3), dtype=np.float32),
            'tpos_first_frame': np.zeros((2, 13), dtype=np.float32),
        },
        'Dog': {
            'joints_names': ['Root', 'Spine', 'Head'],
            'parents': np.array([-1, 0, 1], dtype=np.int64),
            'offsets': np.zeros((3, 3), dtype=np.float32),
            'tpos_first_frame': np.zeros((3, 13), dtype=np.float32),
        },
    }
    np.save(str(dataset_dir / "cond.npy"), cond)

    # Write some motion .npy files.
    np.save(str(motions_dir / "Cat_Run_001.npy"), np.zeros((3, 2, 13), dtype=np.float32))
    np.save(str(motions_dir / "Cat_Jump_002.npy"), np.zeros((5, 2, 13), dtype=np.float32))
    np.save(str(motions_dir / "Dog_Walk_003.npy"), np.zeros((4, 3, 13), dtype=np.float32))

    # Monkeypatch _build_structural_prior_bank to capture inputs.
    captured_payloads: list[dict] = []

    def fake_build(payloads):
        captured_payloads.extend(payloads)
        # Return a realistic-looking bank.
        return {
            'schema_version': 3, 'feature_len': 13,
            'global_scales': {'pos': 0.5, 'rot': 0.3, 'vel': 0.2},
            'by_role': {}, 'by_semantic_group': {}, 'by_canonical_name': {},
            'variance_calibration': {'pos': 1.0, 'rot': 1.0, 'vel': 1.0},
            'metadata': {'object_count': 2, 'joint_examples': 5},
        }

    monkeypatch.setattr(dataset_pipeline_mod, '_build_structural_prior_bank', fake_build)

    # Monkeypatch _apply_structural_stats_to_object_cond to set a marker.
    def fake_apply(object_cond, prior_bank):
        object_cond['norm_mean'] = np.zeros_like(
            np.asarray(object_cond['tpos_first_frame'], dtype=np.float32))
        object_cond['norm_std'] = np.ones_like(object_cond['norm_mean'])
        object_cond['norm_schema_version'] = 3
        object_cond['norm_mean_source'] = 'tpose_anchor_v1'

    monkeypatch.setattr(dataset_pipeline_mod, '_apply_structural_stats_to_object_cond', fake_apply)

    preprocess_module._rebuild_structural_prior_bank(str(dataset_dir))

    # Verify payloads: 2 payloads (Cat, Dog).
    assert len(captured_payloads) == 2
    payload_types = sorted(p['object_cond']['joints_names'][0]
                           for p in captured_payloads)
    # 2 Cat motions, 1 Dog motion.
    cat_payload = next(p for p in captured_payloads
                       if len(p['object_cond']['joints_names']) == 2)
    dog_payload = next(p for p in captured_payloads
                       if len(p['object_cond']['joints_names']) == 3)
    assert len(cat_payload['results']) == 2  # Cat_Run + Cat_Jump
    assert len(dog_payload['results']) == 1  # Dog_Walk

    # Verify prior bank was saved.
    prior_bank_path = dataset_dir / PRIORS_FILE
    assert prior_bank_path.exists()
    bank = dict(np.load(str(prior_bank_path), allow_pickle=True).item())
    assert bank['metadata']['object_count'] == 2

    # Verify cond.npy was updated with structural stats for both objects.
    updated = dict(np.load(str(dataset_dir / "cond.npy"), allow_pickle=True).item())
    assert updated['Cat']['norm_mean_source'] == 'tpose_anchor_v1'
    assert updated['Cat']['norm_schema_version'] == 3
    assert updated['Dog']['norm_mean_source'] == 'tpose_anchor_v1'
    assert updated['Dog']['norm_schema_version'] == 3


def test_rebuild_structural_prior_bank_rejects_unknown_object_type(tmp_path):
    """_rebuild_structural_prior_bank must skip motion files whose object_type
    is not in cond.npy (e.g. stale files from a previous object type)."""
    dataset_dir = tmp_path / "dataset"
    motions_dir = dataset_dir / "motions"
    motions_dir.mkdir(parents=True)

    cond = {
        'Cat': {
            'joints_names': ['Root'],
            'parents': np.array([-1], dtype=np.int64),
            'offsets': np.zeros((1, 3), dtype=np.float32),
            'tpos_first_frame': np.zeros((1, 13), dtype=np.float32),
        },
    }
    np.save(str(dataset_dir / "cond.npy"), cond)
    # A known motion and an unidentifiable one.
    np.save(str(motions_dir / "Cat_Run_001.npy"), np.zeros((2, 1, 13), dtype=np.float32))
    np.save(str(motions_dir / "random_garbage.npy"), np.zeros((2, 1, 13), dtype=np.float32))

    captured: list[dict] = []

    def fake_build(payloads):
        captured.extend(payloads)
        return {
            'schema_version': 3, 'feature_len': 13,
            'global_scales': {'pos': 0.5, 'rot': 0.3, 'vel': 0.2},
            'by_role': {}, 'by_semantic_group': {}, 'by_canonical_name': {},
            'variance_calibration': {'pos': 1.0, 'rot': 1.0, 'vel': 1.0},
            'metadata': {'object_count': 1, 'joint_examples': 1},
        }

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(dataset_pipeline_mod, '_build_structural_prior_bank', fake_build)

    preprocess_module._rebuild_structural_prior_bank(str(dataset_dir))

    assert len(captured) == 1  # Only Cat payload
    monkeypatch.undo()


def test_rebuild_structural_prior_bank_skips_objects_missing_tpose(monkeypatch, tmp_path):
    """Objects without tpos_first_frame must be skipped during payload build
    instead of crashing the whole rebuild."""
    dataset_dir = tmp_path / "dataset"
    motions_dir = dataset_dir / "motions"
    motions_dir.mkdir(parents=True)

    cond = {
        'Cat': {
            'joints_names': ['Root'],
            'parents': np.array([-1], dtype=np.int64),
            'offsets': np.zeros((1, 3), dtype=np.float32),
            'tpos_first_frame': np.zeros((1, 13), dtype=np.float32),
        },
        'Dog': {
            'joints_names': ['Root'],
            'parents': np.array([-1], dtype=np.int64),
            'offsets': np.zeros((1, 3), dtype=np.float32),
        },
    }
    np.save(str(dataset_dir / "cond.npy"), cond)
    np.save(str(motions_dir / "Cat_Run_001.npy"), np.zeros((2, 1, 13), dtype=np.float32))
    np.save(str(motions_dir / "Dog_Run_001.npy"), np.zeros((2, 1, 13), dtype=np.float32))

    captured_payloads: list[dict] = []

    def fake_build(payloads):
        captured_payloads.extend(payloads)
        return {
            'schema_version': 3, 'feature_len': 13,
            'global_scales': {'pos': 0.5, 'rot': 0.3, 'vel': 0.2},
            'by_role': {}, 'by_semantic_group': {}, 'by_canonical_name': {},
            'variance_calibration': {'pos': 1.0, 'rot': 1.0, 'vel': 1.0},
            'metadata': {'object_count': 1, 'joint_examples': 1},
        }

    def fake_apply(object_cond, prior_bank):
        object_cond['norm_mean'] = np.zeros_like(
            np.asarray(object_cond['tpos_first_frame'], dtype=np.float32)
        )
        object_cond['norm_std'] = np.ones_like(object_cond['norm_mean'])

    monkeypatch.setattr(dataset_pipeline_mod, '_build_structural_prior_bank', fake_build)
    monkeypatch.setattr(dataset_pipeline_mod, '_apply_structural_stats_to_object_cond', fake_apply)

    preprocess_module._rebuild_structural_prior_bank(str(dataset_dir))

    assert len(captured_payloads) == 1
    assert captured_payloads[0]['object_cond']['joints_names'] == ['Root']

    updated = dict(np.load(str(dataset_dir / "cond.npy"), allow_pickle=True).item())
    assert 'norm_mean' in updated['Cat']
    assert 'norm_mean' not in updated['Dog']


def test_rebuild_structural_prior_bank_skips_when_no_motions(tmp_path):
    """_rebuild_structural_prior_bank must silently skip if motions/
    does not exist or is empty."""
    dataset_dir = tmp_path / "dataset_not_exists"
    preprocess_module._rebuild_structural_prior_bank(str(dataset_dir))
    # No exception.

    dataset_dir2 = tmp_path / "dataset"
    dataset_dir2.mkdir()
    np.save(str(dataset_dir2 / "cond.npy"), {'Fake': {}})
    # No motions/ dir → skip.
    preprocess_module._rebuild_structural_prior_bank(str(dataset_dir2))
    # No exception.


def test_main_returns_sidecar_failure_code(monkeypatch, tmp_path):
    """Preprocess workflow must stop when sidecar regeneration fails."""
    monkeypatch.setattr(
        preprocess_module,
        'parse_args',
        lambda: argparse.Namespace(
            sample_count=0,
            orientation_threshold_deg=15.0,
            re_encode_joint_names_only=False,
            validate_only=False,
            objects_subset='all',
            object_workers=1,
            raw_data_dir='',
            dataset_dir=str(tmp_path),
            skip_validate=False,
            skip_orientation_check=False,
        ),
    )
    monkeypatch.setattr(preprocess_module, 'check_and_clean_old_data', lambda *args, **kwargs: (True, preprocess_module.PreservedSideArtifacts()))
    monkeypatch.setattr(preprocess_module, 'run_preprocessing', lambda *args, **kwargs: 0)
    monkeypatch.setattr(preprocess_module, 'get_dataset_dir', lambda raw_value=None: str(tmp_path))
    monkeypatch.setattr(preprocess_module, '_merge_preserved_side_artifacts', lambda *args, **kwargs: None)
    monkeypatch.setattr(preprocess_module, '_rebuild_structural_prior_bank', lambda *args, **kwargs: None)
    monkeypatch.setattr(preprocess_module, 'run_re_encode_joint_names_only', lambda *args, **kwargs: 7)

    def unexpected_validation(*args, **kwargs):
        raise AssertionError('validation should not run after sidecar failure')

    monkeypatch.setattr(preprocess_module, 'run_validation', unexpected_validation)

    assert preprocess_module.main() == 7