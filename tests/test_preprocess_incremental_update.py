import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.truebones.truebones_utils.motion_labels import (
    MOTION_METADATA_SCHEMA_VERSION,
    load_motion_metadata,
    write_motion_metadata,
)
from data_loaders.truebones.truebones_utils.canonical_features import CANONICAL_FEATURE_SPACE
from data_loaders.truebones.truebones_utils.param_utils import FEATS_LEN
from data_loaders.truebones.truebones_utils.cond_schema import load_cond
from data_loaders.truebones.truebones_utils.dataset_sources import build_species_file_tokens
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
        "rest_pose": np.zeros((2, 12), dtype=np.float32),
    }


def _write_action_labels(dataset_dir, labels_by_clip, is_loop=False):
    """Write the hand-maintained action_labels.jsonl sidecar for a temp dataset.

    Values are ``(action_group, action_label)`` pairs, or
    ``(action_group, action_label, is_loop)`` to carry a loop verdict; ``is_loop``
    is the default verdict for pairs. Every clip on disk needs one for the strict
    join (load_motion_metadata) to pass, and every clip a build targets needs
    one up front; ``is_loop=None`` leaves pairs unjudged.
    """
    path = Path(dataset_dir) / "action_labels.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for clip, value in labels_by_clip.items():
        group, label = value[0], value[1]
        row = {"clip": clip, "action_group": group, "action_label": label}
        verdict = value[2] if len(value) > 2 else is_loop
        if verdict is not None:
            row["is_loop"] = bool(verdict)
        lines.append(json.dumps(row))
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



def _cond_by_species(dataset_dir):
    """Read a written cond.npy back, re-indexed by bare species name.

    cond.npy is keyed ``<namespace>/<species>`` on disk (schema v4); these tests
    assert on species, not on which dataset directory pytest happened to create,
    so the namespace is dropped here.
    """
    cond = load_cond(Path(dataset_dir) / "cond.npy")
    return {str(entry["species_name"]): entry for entry in cond.values()}


def test_write_motion_metadata_preserves_all_fields(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True)

    write_motion_metadata(
        dataset_dir,
        {
            "Cat_Run_001.npy": {
                "object_type": "Cat",
                "action_label": "run, gallops forward",
                "action_group": "locomotion",
                "species_label": "cat",
                "motion_name": "Cat_Run_001.npy",
                "translation_root_index": 1,
            },
        },
        total_clips=1,
    )

    payload = json.loads((dataset_dir / "motion_metadata.json").read_text(encoding="utf-8"))
    entry = payload["motions"]["Cat_Run_001.npy"]
    assert payload["schema_version"] == MOTION_METADATA_SCHEMA_VERSION
    assert entry["object_type"] == "Cat"
    assert entry["translation_root_index"] == 1
    assert entry["motion_name"] == "Cat_Run_001.npy"
    # The action fields are joined in from action_labels.jsonl at load time and
    # must NOT be persisted here: a stored copy diverges the moment the sidecar
    # is edited, and every rebuild path round-trips loaded entries through this
    # writer.
    assert "action_group" not in entry
    assert "action_label" not in entry
    # is_loop moved into the sidecar with schema 7: a copy here would go stale
    # the first time a verdict is corrected in the review UI.
    assert "is_loop" not in entry
    # species_label was a derived (lower-cased object_type) text label, removed
    # from the schema; the writer must strip the stale copies carried over from
    # older metadata files.
    assert "species_label" not in entry


def test_load_motion_metadata_merges_action_labels_from_sidecar(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "motion_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 6,
                "total_clips": 1,
                "motions": {
                    "Cat_Run_001.npy": {
                        "object_type": "Cat",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    _write_action_labels(
        dataset_dir,
        {"Cat_Run_001.npy": ("Locomotion", "  run,  forward ", True)},
    )

    loaded = load_motion_metadata(dataset_dir)
    entry = loaded["Cat_Run_001.npy"]
    assert entry["action_group"] == "locomotion"
    assert entry["action_label"] == "run, forward"
    assert entry["is_loop"] is True


def test_load_motion_metadata_fast_fails_when_label_missing(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "motion_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 6,
                "total_clips": 1,
                "motions": {
                    "Cat_Run_001.npy": {"object_type": "Cat"},
                },
            }
        ),
        encoding="utf-8",
    )
    # Sidecar exists but is missing the clip → must fail fast.
    _write_action_labels(dataset_dir, {"Dog_Jump_002.npy": ("transition", "jump")})

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
    # The verdicts live in the sidecar; the metadata's is_loop above is a stale
    # copy from an older build that the regeneration must not resurrect.
    _write_action_labels(
        dataset_dir,
        {
            "Cat_Run_001.npy": ("locomotion", "run", True),
            "Dog_Jump_002.npy": ("transition", "jump", False),
            "Stale_Idle_003.npy": ("stationary", "idle", False),
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
        # Mirrors the real encoder: cond keys carry '/', so inspection files are
        # named by the species file token.
        file_tokens = build_species_file_tokens(cond)
        for object_type, object_cond in cond.items():
            embedding_count = len(object_cond["joints_names"])
            object_cond["joints_names_embs"] = np.ones((embedding_count, 1), dtype=np.float32)
            object_cond["joints_names_embs_meta"] = {
                "t5_name": t5_name,
                "schema_version": 1,
                "embedding_dim": 1,
                "embedding_texts": list(object_cond["joints_names"]),
            }
            (inspection_output_dir / f"{file_tokens[object_type]}.json").write_text(
                json.dumps({"object_type": object_type, "encoded": True}),
                encoding="utf-8",
            )

    def fake_write_collision_report(cond, save_dir):
        report_path = Path(save_dir) / "joint_name_collision_report.json"
        tokens = build_species_file_tokens(cond)
        report_path.write_text(
            json.dumps({"num_objects": len(cond), "objects": sorted(tokens.values())}),
            encoding="utf-8",
        )
        return []

    monkeypatch.setattr(regenerate_dataset_artifacts_module, "attach_t5_embeddings_to_cond", fake_attach)
    monkeypatch.setattr(regenerate_dataset_artifacts_module, "write_joint_name_collision_report", fake_write_collision_report)

    _write_species_tags(dataset_dir)
    dataset_dir_path = regenerate_dataset_artifacts_module.regenerate_dataset_artifacts(dataset_dir, t5_model="fake-t5")

    assert dataset_dir_path == dataset_dir.resolve()

    regenerated_cond = _cond_by_species(dataset_dir)
    assert sorted(regenerated_cond) == ["Cat", "Dog"]
    assert regenerated_cond["Cat"]["joints_names_embs_meta"]["t5_name"] == "fake-t5"
    assert regenerated_cond["Dog"]["joints_names_embs_meta"]["t5_name"] == "fake-t5"
    assert regenerated_cond["Cat"]["translation_root_index"] == 1
    assert regenerated_cond["Dog"]["translation_root_index"] == 0

    motion_metadata = load_motion_metadata(dataset_dir)
    assert sorted(motion_metadata) == ["Cat_Run_001.npy", "Dog_Jump_002.npy"]
    assert motion_metadata["Cat_Run_001.npy"]["object_type"] == "Cat"
    assert motion_metadata["Cat_Run_001.npy"]["action_group"] == "locomotion"
    assert motion_metadata["Cat_Run_001.npy"]["action_label"] == "run"
    assert motion_metadata["Cat_Run_001.npy"]["is_loop"] is True
    assert motion_metadata["Cat_Run_001.npy"]["translation_root_index"] == 1
    assert motion_metadata["Cat_Run_001.npy"]["motion_source"] == "anim_dir"
    assert motion_metadata["Cat_Run_001.npy"]["source_fbx_path"] == "cat.fbx"
    assert motion_metadata["Dog_Jump_002.npy"]["object_type"] == "Dog"
    assert motion_metadata["Dog_Jump_002.npy"]["is_loop"] is False
    assert motion_metadata["Dog_Jump_002.npy"]["translation_root_index"] == 0
    assert motion_metadata["Dog_Jump_002.npy"]["motion_source"] == "retarget"
    rewritten = json.loads((dataset_dir / "motion_metadata.json").read_text(encoding="utf-8"))
    assert all("is_loop" not in entry for entry in rewritten["motions"].values())

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


def test_regenerate_dataset_artifacts_rejects_inconsistent_translation_roots(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "dataset"
    motions_dir = dataset_dir / "motions"
    motions_dir.mkdir(parents=True)

    np.save(motions_dir / "Cat_Run_001.npy", np.zeros((3, 3, 12), dtype=np.float32))
    np.save(motions_dir / "Cat_Idle_002.npy", np.zeros((5, 3, 12), dtype=np.float32))
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
    _write_action_labels(
        dataset_dir,
        {"Cat_Run_001.npy": ("locomotion", "run"), "Cat_Idle_002.npy": ("stationary", "idle")},
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
    with pytest.raises(RuntimeError, match="motion metadata disagrees"):
        regenerate_dataset_artifacts_module.regenerate_dataset_artifacts(
            dataset_dir,
            t5_model="fake-t5",
        )

    # Side-artifact regeneration is read-only with respect to feature provenance.
    motion_metadata = load_motion_metadata(dataset_dir)
    assert motion_metadata["Cat_Run_001.npy"]["translation_root_index"] == 2
    assert motion_metadata["Cat_Idle_002.npy"]["translation_root_index"] == 1


def _travel(*per_joint):
    return {joint: float(value) for joint, value in enumerate(per_joint)}


def test_species_translation_root_takes_the_deepest_carrier_not_the_majority():
    """``MB_TigerDrago``: 71 clips move the CG wrapper, 40 move the pelvis below it.

    The pelvis clips are the ones that travel -- Run, RunJump, DogdeLeftG -- and
    a root above their carrier sees none of it. A joint's global position already
    contains every ancestor's translation, so the deepest carrier is the only
    choice that covers the whole species.
    """
    selected, counts = dataset_pipeline_mod._select_species_translation_root(
        'MB_TigerDrago',
        [0] * 71 + [1] * 40,
        [_travel(3.66, 0.0, 0.0)] * 71 + [_travel(0.0, 4.52, 0.0)] * 40,
    )

    assert selected == 1
    assert counts == {0: 71, 1: 40}


def test_motionless_clips_do_not_drag_the_root_back_to_the_hierarchy_head():
    """``Tukan``: five clips never move, two fly.

    A clip that goes nowhere still has to answer with an index, and it answers
    with the chain head because something must be returned. The old rule counted
    those five answers as votes and they outnumbered the two that fly, leaving
    ``Tukan_Fly`` with 5.08 of travel on a root that never moved.
    """
    selected, counts = dataset_pipeline_mod._select_species_translation_root(
        'Tukan',
        [0] * 5 + [1] * 2,
        [_travel(0.0, 0.0, 0.0)] * 5 + [_travel(0.0, 5.10, 0.0)] * 2,
    )

    assert selected == 1
    assert counts == {0: 5, 1: 2}


def test_a_body_joint_that_only_sways_does_not_become_the_species_root():
    """``Crow``: the pelvis flies 0.809, the spine below it drifts 0.026.

    Counting any motion as a carrier would read the whole species' trajectory
    off a spine. Counting clips cannot separate this from MB_TigerDrago either --
    Crow's spine clips are 40% of the species and must lose, TigerDrago's pelvis
    clips are 36% and must win. Only the magnitudes tell them apart.
    """
    selected, _counts = dataset_pipeline_mod._select_species_translation_root(
        'Crow',
        [1] * 5 + [2] * 4,
        [_travel(0.0, 0.809, 0.0)] * 5 + [_travel(0.0, 0.0, 0.026)] * 4,
    )

    assert selected == 1


def test_one_clip_authored_on_the_wrong_joint_still_moves_the_species_root():
    """``Bear``: every clip travels on Root except ``AtkStand``, which uses Pelvis.

    Its whole skeleton -- all four ankles included -- shifts 0.62 while Root
    stays at 0.002, so that travel is real and a root above the pelvis cannot
    see it. Rooting at the pelvis costs nothing: the pelvis never moves relative
    to Root anywhere else in the species.
    """
    selected, _counts = dataset_pipeline_mod._select_species_translation_root(
        'Bear',
        [0] * 28 + [1],
        [_travel(5.70, 0.0)] * 28 + [_travel(0.002, 0.708)],
    )

    assert selected == 1


def test_a_species_that_never_moves_still_gets_a_deterministic_root():
    selected, counts = dataset_pipeline_mod._select_species_translation_root(
        'Statue',
        [0, 0],
        [_travel(0.0, 0.0), _travel(0.001, 0.0)],
    )

    assert selected == 0
    assert counts == {0: 2}


def test_incremental_prepare_scans_only_new_source_and_reuses_alignment(monkeypatch, tmp_path):
    raw_dir = tmp_path / 'Cat'
    raw_dir.mkdir()
    old_source = raw_dir / 'Cat-Old.glb'
    new_source = raw_dir / 'Cat-New.glb'
    tpose_source = raw_dir / 'Cat-TPOSE.glb'
    for source in (old_source, new_source, tpose_source):
        source.touch()

    parents = np.array([-1, 0], dtype=np.int64)
    tp = SimpleNamespace(
        offsets=np.zeros((2, 3), dtype=np.float32),
        foot_indices=[],
        tpos_rots=object(),
        orientation_quat=object(),
        prop_socket_names=(),
        end_site_names=(),
        names=['Root', 'Tail'],
        tpos_anim=object(),
    )
    object_cond = {
        **_make_cond_entry('Cat'),
        'canonical_bvh_joint_names': ['Root', 'Tail'],
    }
    monkeypatch.setattr(dataset_pipeline_mod, 'should_skip_anim', lambda *_args: False)
    monkeypatch.setattr(
        dataset_pipeline_mod,
        '_build_rest_pose_cond',
        lambda *_args, **_kwargs: (
            object_cond,
            tp,
            np.zeros((1, 2, 12), dtype=np.float32),
            parents,
            {},
            1.0,
            {},
            2,
            None,
        ),
    )
    monkeypatch.setattr(
        dataset_pipeline_mod,
        'get_motion',
        lambda *_args, **_kwargs: (
            np.zeros((1, 2, 12), dtype=np.float32),
            parents,
            2,
            None,
            None,
            False,
            0,
            None,
            False,
        ),
    )

    scanned = []
    prepared_payload = {
        'file_path': str(new_source),
        'translation_root_index': 0,
        'chain_xz_travel': {0: 1.5, 1: 0.0},
    }

    def fake_prepare(file_path, *_args, **_kwargs):
        scanned.append(file_path)
        return prepared_payload

    encoded = []

    def fake_encode(prepared, *_args, **_kwargs):
        encoded.append(prepared)
        return {'errors': {}, 'max_joints': 2, 'results': [], 'motion_errors': []}

    monkeypatch.setattr(
        dataset_pipeline_mod,
        '_prepare_motion_file_for_root_detection',
        fake_prepare,
    )
    monkeypatch.setattr(dataset_pipeline_mod, '_encode_prepared_motion_file', fake_encode)

    result = dataset_pipeline_mod._prepare_object_outputs(
        'Cat',
        2,
        fbxs_dir=str(raw_dir),
        t_pos_path=str(tpose_source),
        skip_source_paths={str(old_source)},
        resample_min_length=0,
        frozen_translation_root_index=0,
    )

    assert result is None  # fake encoder intentionally emitted no motion result
    assert scanned == [str(new_source)]
    assert encoded == [prepared_payload]


def test_incremental_prepare_rejects_new_source_root_mismatch(monkeypatch, tmp_path):
    raw_dir = tmp_path / 'Cat'
    raw_dir.mkdir()
    new_source = raw_dir / 'Cat-New.glb'
    tpose_source = raw_dir / 'Cat-TPOSE.glb'
    new_source.touch()
    tpose_source.touch()

    parents = np.array([-1, 0], dtype=np.int64)
    tp = SimpleNamespace(
        offsets=np.zeros((2, 3), dtype=np.float32),
        foot_indices=[],
        tpos_rots=object(),
        orientation_quat=object(),
        prop_socket_names=(),
        end_site_names=(),
        names=['Root', 'Tail'],
        tpos_anim=object(),
    )
    monkeypatch.setattr(dataset_pipeline_mod, 'should_skip_anim', lambda *_args: False)
    monkeypatch.setattr(
        dataset_pipeline_mod,
        '_build_rest_pose_cond',
        lambda *_args, **_kwargs: (
            {**_make_cond_entry('Cat'), 'canonical_bvh_joint_names': ['Root', 'Tail']},
            tp,
            np.zeros((1, 2, 12), dtype=np.float32),
            parents,
            {},
            1.0,
            {},
            2,
            None,
        ),
    )
    monkeypatch.setattr(
        dataset_pipeline_mod,
        '_prepare_motion_file_for_root_detection',
        lambda file_path, *_args, **_kwargs: {
            'file_path': file_path,
            'translation_root_index': 1,
            'chain_xz_travel': {0: 0.0, 1: 1.5},
        },
    )

    with pytest.raises(dataset_pipeline_mod.DatasetPreprocessingError) as exc_info:
        dataset_pipeline_mod._prepare_object_outputs(
            'Cat',
            2,
            fbxs_dir=str(raw_dir),
            t_pos_path=str(tpose_source),
            resample_min_length=0,
            frozen_translation_root_index=0,
        )
    assert 'frozen cond' in exc_info.value.motion_errors[0]


def test_incremental_prepare_accepts_a_new_source_rooted_above_the_frozen_joint(monkeypatch, tmp_path):
    """A carrier ABOVE the frozen root is not a mismatch.

    FK folds an ancestor's translation into the frozen root's global position,
    so the features still see it -- which is the whole reason the species vote
    picks the deepest carrier. Only a carrier below the frozen root is lost.
    ``MB_TigerDrago`` needs both: 40 clips carry on the pelvis and 71 on the CG
    wrapper above it, and the species is rooted at the pelvis.
    """
    raw_dir = tmp_path / 'Cat'
    raw_dir.mkdir()
    new_source = raw_dir / 'Cat-New.glb'
    tpose_source = raw_dir / 'Cat-TPOSE.glb'
    new_source.touch()
    tpose_source.touch()

    parents = np.array([-1, 0, 1], dtype=np.int64)
    tp = SimpleNamespace(
        offsets=np.zeros((3, 3), dtype=np.float32),
        foot_indices=[],
        tpos_rots=object(),
        orientation_quat=object(),
        prop_socket_names=(),
        end_site_names=(),
        names=['Cg', 'Pelvis', 'Spine'],
        tpos_anim=object(),
    )
    object_cond = {
        **_make_cond_entry('Cat'),
        'canonical_bvh_joint_names': ['Cg', 'Pelvis', 'Spine'],
    }
    monkeypatch.setattr(dataset_pipeline_mod, 'should_skip_anim', lambda *_args: False)
    monkeypatch.setattr(
        dataset_pipeline_mod,
        '_build_rest_pose_cond',
        lambda *_args, **_kwargs: (
            object_cond, tp, np.zeros((1, 3, 12), dtype=np.float32), parents,
            {}, 1.0, {}, 3, None,
        ),
    )
    monkeypatch.setattr(
        dataset_pipeline_mod,
        'get_motion',
        lambda *_args, **_kwargs: (
            np.zeros((1, 3, 12), dtype=np.float32), parents, 3, None, None,
            False, 1, None, False,
        ),
    )
    monkeypatch.setattr(
        dataset_pipeline_mod,
        '_prepare_motion_file_for_root_detection',
        lambda file_path, *_args, **_kwargs: {
            'file_path': file_path,
            'translation_root_index': 0,
            'chain_xz_travel': {0: 1.5, 1: 0.0, 2: 0.0},
        },
    )
    monkeypatch.setattr(
        dataset_pipeline_mod,
        '_encode_prepared_motion_file',
        lambda prepared, *_args, **_kwargs: {
            'errors': {}, 'max_joints': 3, 'results': [], 'motion_errors': [],
        },
    )

    assert dataset_pipeline_mod._prepare_object_outputs(
        'Cat',
        3,
        fbxs_dir=str(raw_dir),
        t_pos_path=str(tpose_source),
        resample_min_length=0,
        frozen_translation_root_index=1,
    ) is None  # the fake encoder intentionally emits no motion result


def test_regenerate_dataset_artifacts_rebuilds_translation_root_when_metadata_missing(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "dataset"
    motions_dir = dataset_dir / "motions"
    motions_dir.mkdir(parents=True)

    np.save(motions_dir / "Cat_Run_001.npy", np.zeros((3, 3, 12), dtype=np.float32))
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
    _write_action_labels(dataset_dir, {"Cat_Run_001.npy": ("locomotion", "run")})

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

    regenerated_cond = _cond_by_species(dataset_dir)
    assert regenerated_cond["Cat"]["translation_root_index"] == 2

    motion_metadata = load_motion_metadata(dataset_dir)
    assert motion_metadata["Cat_Run_001.npy"]["translation_root_index"] == 2


def test_regenerate_dataset_artifacts_backfills_missing_cond_root_from_unanimous_clips(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "dataset"
    motions_dir = dataset_dir / "motions"
    motions_dir.mkdir(parents=True)

    for idx in range(4):
        np.save(motions_dir / f"Bear_Run_{idx:03d}.npy", np.zeros((idx + 3, 3, FEATS_LEN), dtype=np.float32))

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
    _write_action_labels(
        dataset_dir,
        {f"Bear_Run_{idx:03d}.npy": ("locomotion", "run") for idx in range(4)},
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

    regenerated_cond = _cond_by_species(dataset_dir)
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
    _write_action_labels(
        dataset_dir,
        {
            "Cat_Run_001.npy": ("locomotion", "run"),
            "Dog_Jump_002.npy": ("locomotion", "run"),
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

    def fake_prepare_object_outputs(object_type, max_joints, face_joints=None, fbxs_dir=None, t_pos_path=None, max_files=None, raw_data_dir=None, filter_min_length=10, resample_min_length=20, skip_source_paths=None, frozen_translation_root_index=None, frozen_promote_root_depth=None, locomotion_clips=frozenset(), loop_verdicts=None):
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

    def fake_write_object_outputs(save_dir, object_payload, files_counter, existing_clip_sources=None):
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

    # Preprocessing prerequisites: the hand-maintained sidecars must exist and
    # be valid before any clip is encoded.
    _write_action_labels(dataset_dir, {"Cat_Run_001.npy": ("locomotion", "run")})
    _write_species_tags(dataset_dir, species=("Cat",))
    (tmp_path / 'raw').mkdir()

    dataset_pipeline_mod.create_data_samples(
        objects=['Cat'],
        dataset_dir=str(dataset_dir),
        raw_data_dir=str(tmp_path / 'raw'),
        object_workers=1,
    )

    seed_cond = _cond_by_species(dataset_dir)
    assert sorted(seed_cond) == ['Cat']
    assert 'joints_names_embs' not in seed_cond['Cat']

    _write_action_labels(dataset_dir, {"Cat_Run_001.npy": ("locomotion", "run")})
    motion_metadata = load_motion_metadata(dataset_dir)
    assert motion_metadata['Cat_Run_001.npy']['translation_root_index'] == 1

    positions_error_lines = (dataset_dir / 'positions_error_rate.txt').read_text(encoding='utf-8').splitlines()
    assert positions_error_lines[0] == 'Position squared error per source clip:'
    assert 'Cat run clip: 0.010000' in positions_error_lines

    assert not (dataset_dir / 'metadata.txt').exists()
    assert not (dataset_dir / 'joint_name_inspection').exists()


def test_create_data_samples_raises_preprocess_error_instead_of_exit(monkeypatch, tmp_path):
    dataset_dir = tmp_path / 'dataset'

    def fake_prepare_object_outputs(object_type, max_joints, face_joints=None, fbxs_dir=None, t_pos_path=None, max_files=None, raw_data_dir=None, filter_min_length=10, resample_min_length=20, skip_source_paths=None, frozen_translation_root_index=None, frozen_promote_root_depth=None, locomotion_clips=frozenset(), loop_verdicts=None):
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

    # Preprocessing prerequisites: the hand-maintained sidecars must exist and
    # be valid before any clip is encoded.
    _write_action_labels(dataset_dir, {"Cat_Run_001.npy": ("locomotion", "run")})
    _write_species_tags(dataset_dir, species=("Cat",))
    (tmp_path / 'raw').mkdir()

    with pytest.raises(dataset_pipeline_mod.DatasetPreprocessingError) as exc_info:
        dataset_pipeline_mod.create_data_samples(
            objects=['Cat'],
            dataset_dir=str(dataset_dir),
            raw_data_dir=str(tmp_path / 'raw'),
            object_workers=1,
        )

    assert exc_info.value.motion_errors == ('boom',)


def test_create_data_samples_refuses_a_target_clip_without_a_loop_verdict(monkeypatch, tmp_path):
    """The loop flag is a prerequisite: a build never proposes it, and a target
    clip whose row has none stops the run before any source is loaded."""
    dataset_dir = tmp_path / 'dataset'
    (tmp_path / 'raw' / 'Cat').mkdir(parents=True)
    for source in ('Walk.fbx', 'Run.fbx'):
        (tmp_path / 'raw' / 'Cat' / source).write_bytes(b'')
    _write_action_labels(
        dataset_dir,
        {'Cat_Walk': ('locomotion', 'walk', True), 'Cat_Run': ('locomotion', 'run')},
        is_loop=None,
    )
    _write_species_tags(dataset_dir, species=("Cat",))
    labels_before = (dataset_dir / 'action_labels.jsonl').read_bytes()

    def fake_prepare_object_outputs(*args, **kwargs):
        raise AssertionError("no source may be loaded before the loop-flag gate passes")

    monkeypatch.setattr(dataset_pipeline_mod, '_prepare_object_outputs', fake_prepare_object_outputs)

    with pytest.raises(dataset_pipeline_mod.DatasetPreprocessingError) as exc_info:
        dataset_pipeline_mod.create_data_samples(
            objects=['Cat'],
            dataset_dir=str(dataset_dir),
            raw_data_dir=str(tmp_path / 'raw'),
            object_workers=1,
        )

    message = ' '.join(exc_info.value.motion_errors)
    assert 'Cat_Run' in message and 'Cat_Walk' not in message
    assert 'prefill_loop_flags.py' in message
    assert (dataset_dir / 'action_labels.jsonl').read_bytes() == labels_before


def test_create_data_samples_fast_fails_without_prerequisite_sidecars(monkeypatch, tmp_path):
    """No sidecars at all: the gate aborts before any object is prepared.

    species_tags.jsonl is checked first (it is loaded through the dataset_tags
    snapshot), so a bare dataset fails on that one."""
    dataset_dir = tmp_path / "dataset"

    def fake_prepare_object_outputs(*args, **kwargs):
        raise AssertionError("no preprocessing work may start before the sidecar gate passes")

    monkeypatch.setattr(dataset_pipeline_mod, '_prepare_object_outputs', fake_prepare_object_outputs)

    with pytest.raises(FileNotFoundError, match="species_tags.jsonl"):
        dataset_pipeline_mod.create_data_samples(
            objects=['Cat'],
            dataset_dir=str(dataset_dir),
            object_workers=1,
        )


def test_create_data_samples_fast_fails_when_action_labels_missing(monkeypatch, tmp_path):
    """Species tags present but action_labels.jsonl absent: the gate still
    aborts up front, and nothing is inferred or back-filled."""
    dataset_dir = tmp_path / "dataset"
    _write_species_tags(dataset_dir, species=("Cat",))

    def fake_prepare_object_outputs(*args, **kwargs):
        raise AssertionError("no preprocessing work may start before the sidecar gate passes")

    monkeypatch.setattr(dataset_pipeline_mod, '_prepare_object_outputs', fake_prepare_object_outputs)

    with pytest.raises(FileNotFoundError, match="action_labels.jsonl"):
        dataset_pipeline_mod.create_data_samples(
            objects=['Cat'],
            dataset_dir=str(dataset_dir),
            object_workers=1,
        )


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
        assert rebuilt[object_type]['feature_space'] == CANONICAL_FEATURE_SPACE
        assert rebuilt[object_type]['physical_feature_space'] == 'hml_like_v_current'
        assert rebuilt[object_type]['rest_pos_ric_hml'].shape == (2, 3)


def test_create_data_samples_incremental_skips_done_sources_and_merges(monkeypatch, tmp_path):
    dataset_dir = tmp_path / 'dataset'
    (dataset_dir / 'motions').mkdir(parents=True)
    (dataset_dir / 'bvhs').mkdir(parents=True)

    # Existing dataset: Cat (Walk from Cat_Walk.fbx) and an untouched Dog object.
    # The source files exist (empty) so the loop-flag prerequisite gate, which
    # enumerates them by name, sees the same clips the fake below produces.
    (tmp_path / 'raw' / 'Cat').mkdir(parents=True)
    (tmp_path / 'raw' / 'Dog').mkdir(parents=True)
    for source in ('Cat/Cat_Walk.fbx', 'Cat/Cat_Run.fbx', 'Dog/Dog_Idle.fbx'):
        (tmp_path / 'raw' / source).write_bytes(b'')
    done_source = str(tmp_path / 'raw' / 'Cat' / 'Cat_Walk.fbx')
    cat_cond = _make_cond_entry('Cat')
    dog_cond = _make_cond_entry('Dog')
    cat_cond['translation_root_index'] = 0
    dog_cond['translation_root_index'] = 0
    np.save(dataset_dir / 'cond.npy', {'Cat': cat_cond, 'Dog': dog_cond})
    write_motion_metadata(
        dataset_dir,
        {
            'Cat_Walk.npy': {'object_type': 'Cat', 'source_fbx_path': done_source, 'motion_name': 'Cat_Walk.npy'},
            'Dog_Idle.npy': {'object_type': 'Dog', 'source_fbx_path': str(tmp_path / 'raw' / 'Dog' / 'Dog_Idle.fbx'), 'motion_name': 'Dog_Idle.npy'},
        },
        2,
    )
    # Preprocessing prerequisites: the hand-maintained sidecars must exist and
    # be valid before any clip is encoded. Cat_Run's verdict was proposed ahead
    # of the build (prefill_loop_flags.py); the build only reads it.
    _write_action_labels(
        dataset_dir,
        {
            'Cat_Walk.npy': ('locomotion', 'walk', True),
            'Cat_Run.npy': ('locomotion', 'run', True),
            'Dog_Idle.npy': ('stationary', 'idle', False),
        },
        is_loop=None,
    )
    _write_species_tags(dataset_dir, species=("Cat", "Dog"))
    labels_before = (dataset_dir / 'action_labels.jsonl').read_bytes()

    captured: dict[str, object] = {}

    def fake_prepare(object_type, max_joints, face_joints=None, fbxs_dir=None, t_pos_path=None,
                     max_files=None, raw_data_dir=None, filter_min_length=10, resample_min_length=20,
                     skip_source_paths=None, frozen_translation_root_index=None,
                     frozen_promote_root_depth=None, locomotion_clips=frozenset(),
                     loop_verdicts=None):
        captured['skip_source_paths'] = set(skip_source_paths or set())
        captured['frozen_translation_root_index'] = frozen_translation_root_index
        captured['frozen_promote_root_depth'] = frozen_promote_root_depth
        captured['loop_verdicts'] = dict(loop_verdicts or {})
        return {
            'object_type': object_type,
            'object_cond': {**_make_cond_entry(object_type), 'translation_root_index': 0},
            'tpose_reference_path': None,
            'errors': {},
            'max_joints': 2,
            'results': [],
            'files_counter': 0,
            'frames_counter': 0,
            'face_joints': face_joints,
            'motion_errors': [],
        }

    def fake_write(save_dir, payload, files_counter, existing_clip_sources=None):
        captured['existing_clip_sources'] = dict(existing_clip_sources or {})
        obj = payload['object_type']
        # 1:1 naming: the new source file Cat_Run.fbx yields exactly one new clip.
        name = f"{obj}_Run.npy"
        np.save(Path(save_dir) / 'motions' / name, np.zeros((3, 2, 3), dtype=np.float32))
        return files_counter + 1, 3, {name: {'object_type': obj, 'motion_name': name}}

    monkeypatch.setattr(dataset_pipeline_mod, '_prepare_object_outputs', fake_prepare)
    monkeypatch.setattr(dataset_pipeline_mod, '_write_object_outputs', fake_write)

    dataset_pipeline_mod.create_data_samples(
        objects=['Cat'],
        dataset_dir=str(dataset_dir),
        raw_data_dir=str(tmp_path / 'raw'),
        object_workers=1,
        incremental=True,
    )

    # Already-processed source handed to the worker as a skip; the retained clip's
    # (name -> source) pair is handed to the writer so a new clip cannot silently
    # overwrite it.
    assert captured['skip_source_paths'] == {os.path.realpath(done_source)}
    assert captured['frozen_translation_root_index'] == 0
    assert captured['existing_clip_sources'] == {'Cat_Walk.npy': os.path.realpath(done_source)}
    # Every verdict the sidecar holds reaches the worker, keyed by the
    # extension-less clip name -- and the build never writes the sidecar.
    assert captured['loop_verdicts'] == {'Cat_Walk': True, 'Cat_Run': True, 'Dog_Idle': False}
    assert (dataset_dir / 'action_labels.jsonl').read_bytes() == labels_before

    # cond.npy keeps the untouched Dog and refreshes Cat.
    merged_cond = _cond_by_species(dataset_dir)
    assert sorted(merged_cond) == ['Cat', 'Dog']

    # Existing clips preserved; the new clip is appended without colliding.
    merged_meta = dataset_pipeline_mod._load_motion_metadata_raw(dataset_dir)
    assert set(merged_meta) == {'Cat_Walk.npy', 'Cat_Run.npy', 'Dog_Idle.npy'}


def test_write_object_outputs_rejects_clip_name_collision(tmp_path):
    """Two source files normalizing to the same action must fail loudly.

    With 1:1 clip naming there is no trailing index left to disambiguate them, so
    the second one would otherwise overwrite the first one's .npy in place."""
    payload = {
        'object_type': 'Cat',
        'results': [{
            'motion': np.zeros((3, 2, 3), dtype=np.float32),
            'action': 'Walk1',
            'source_fbx_path': str(tmp_path / 'raw' / 'Walk1.fbx'),
        }],
    }
    existing = {'Cat_Walk1.npy': os.path.realpath(str(tmp_path / 'raw' / 'Walk_1.fbx'))}
    with pytest.raises(ValueError, match='clip name collision'):
        dataset_pipeline_mod._write_object_outputs(
            str(tmp_path), payload, 0, existing_clip_sources=existing,
        )


def _cond_entry_with_stats(object_type, mean_fill, std_fill):
    entry = _make_cond_entry(object_type)
    entry["canonical_feature_mean"] = np.full((FEATS_LEN,), mean_fill, dtype=np.float32)
    entry["canonical_feature_std"] = np.full((FEATS_LEN,), std_fill, dtype=np.float32)
    return entry


def test_merge_inherits_canonical_stats_from_same_object_subset(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True)
    # Existing quadruped (Cat) carries stats; add a new quadruped (Dog) without.
    np.save(dataset_dir / "cond.npy", {"Cat": _cond_entry_with_stats("Cat", 0.5, 2.0)})

    dataset_pipeline_mod._merge_object_into_cond(
        str(dataset_dir), "Dog", _make_cond_entry("Dog")
    )

    merged = _cond_by_species(dataset_dir)
    np.testing.assert_allclose(merged["Dog"]["canonical_feature_mean"], np.full((FEATS_LEN,), 0.5, dtype=np.float32))
    np.testing.assert_allclose(merged["Dog"]["canonical_feature_std"], np.full((FEATS_LEN,), 2.0, dtype=np.float32))


def test_merge_warns_and_borrows_when_no_same_object_subset_donor(tmp_path, capsys):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True)
    # Only a quadruped carries stats; a new winged species (Dragon) has no
    # same-object_subset donor. This used to fast-fail as out-of-distribution.
    # Since the position gain became one globally shared constant, the whole
    # cross-subset delta is a mean (a rigid translation) plus rot/vel gains (an
    # amplitude bias) -- neither can deform the skeleton -- so a brand-new body
    # plan now builds, with a warning naming the donor.
    np.save(dataset_dir / "cond.npy", {"Cat": _cond_entry_with_stats("Cat", 0.5, 2.0)})

    dataset_pipeline_mod._merge_object_into_cond(
        str(dataset_dir), "Dragon", _make_cond_entry("Dragon")
    )

    warning = capsys.readouterr().out
    assert "winged" in warning and "Cat" in warning
    merged = _cond_by_species(dataset_dir)
    np.testing.assert_allclose(merged["Dragon"]["canonical_feature_mean"], np.full((FEATS_LEN,), 0.5, dtype=np.float32))
    np.testing.assert_allclose(merged["Dragon"]["canonical_feature_std"], np.full((FEATS_LEN,), 2.0, dtype=np.float32))


def test_merge_still_fast_fails_when_no_species_carries_stats(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True)
    # Nothing to borrow from at all is still unrecoverable: there is no
    # standardization space to land in, whatever the body plan.
    np.save(dataset_dir / "cond.npy", {"Cat": _make_cond_entry("Cat")})

    with pytest.raises(ValueError, match="canonical standardization stats"):
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

    merged = _cond_by_species(dataset_dir)
    np.testing.assert_allclose(merged["Dragon"]["canonical_feature_mean"], np.full((FEATS_LEN,), 0.3, dtype=np.float32))
    np.testing.assert_allclose(merged["Dragon"]["canonical_feature_std"], np.full((FEATS_LEN,), 1.5, dtype=np.float32))


