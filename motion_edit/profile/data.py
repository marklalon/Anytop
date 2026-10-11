"""Dataset access for the profile extractor: sources, decoded clips, user sidecars."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from data_loaders.truebones.truebones_utils.param_utils import FPS

ANYTOP_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASETS_MANIFEST = os.path.join(ANYTOP_ROOT, "dataset", "datasets.jsonl")

PROFILES_FILE = "skeleton_profiles.json"
# Bumped whenever a profile's content changes meaning; files of another schema are
# rebuilt by build_profiles and refused by load_profiles.
SCHEMA_VERSION = 3
REPORT_FILE = "skeleton_profiles_report.md"


@dataclass(frozen=True)
class DatasetSource:
    namespace: str
    root: str

    @property
    def motion_dir(self) -> str:
        return os.path.join(self.root, "motions")


def discover_sources(manifest_path: str = DATASETS_MANIFEST,
                     anytop_root: str = ANYTOP_ROOT) -> list[DatasetSource]:
    """The processed trees ``dataset/datasets.jsonl`` lists."""
    sources = []
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            entry = json.loads(line)
            rel = entry.get("path")
            if not rel:
                continue
            sources.append(DatasetSource(
                namespace=str(entry.get("namespace") or os.path.basename(rel)),
                root=os.path.normpath(os.path.join(anytop_root, rel)),
            ))
    return sources


def load_cond(source: DatasetSource) -> dict:
    return np.load(os.path.join(source.root, "cond.npy"), allow_pickle=True).item()


def skeleton_hash(parents, offsets) -> str:
    """Identity of a skeleton's topology and rest offsets (offsets rounded to 1e-6)."""
    digest = hashlib.sha1()
    digest.update(np.asarray(parents, dtype=np.int64).tobytes())
    digest.update(np.round(np.asarray(offsets, dtype=np.float64), 6).astype(np.float64).tobytes())
    return digest.hexdigest()


@dataclass
class Clip:
    name: str
    action_group: str
    action_label: str
    is_loop: bool
    fps: float
    local_rotations: np.ndarray   # (F, J, 4) wxyz, parent-relative
    global_rotations: np.ndarray  # (F, J, 4)
    global_positions: np.ndarray  # (F, J, 3), HML space, Y up

    @property
    def frame_count(self) -> int:
        return int(self.local_rotations.shape[0])

    @property
    def head_word(self) -> str:
        return self.action_label.split(",")[0].strip()

    @property
    def family(self) -> str:
        """Action family the statistics are balanced over: group + head word."""
        return f"{self.action_group}|{self.head_word}"


def species_motion_names(metadata: dict, species_name: str) -> list[str]:
    """Motion files whose metadata names this species (not a name-prefix match:
    ``Dog_`` would also catch ``Dog_Big_*`` of another species)."""
    return sorted(name for name, row in metadata.items()
                  if str(row.get("object_type")) == species_name)


def decode_clip(source: DatasetSource, motion_name: str, cond_entry: dict, meta_row: dict,
                *, cond_key: Optional[str] = None) -> Clip:
    from data_loaders.truebones.data.dataset import _drop_loop_closing_frame
    from motion_lib.Animation import positions_global, rotations_global
    from utils.npy_restore import build_skeleton_only_context, restore_animation_from_features

    features = np.load(os.path.join(source.motion_dir, motion_name))
    is_loop = bool(meta_row.get("is_loop", False))
    if is_loop:
        # The on-disk loop may still carry the closing key (last frame == frame 0);
        # the loader drops it, and so must every periodic statistic here.
        features = _drop_loop_closing_frame(features)
    ctx = build_skeleton_only_context(cond_entry, object_type=cond_key,
                                      feature_joint_count=features.shape[1])
    restored = restore_animation_from_features(features, ctx, restore_space="hml", fps=FPS)
    anim = restored.animation
    return Clip(
        name=os.path.splitext(motion_name)[0],
        action_group=str(meta_row.get("action_group") or ""),
        action_label=str(meta_row.get("action_label") or ""),
        is_loop=is_loop,
        fps=float(restored.fps),
        local_rotations=np.asarray(anim.rotations.qs, dtype=np.float64),
        global_rotations=np.asarray(rotations_global(anim).qs, dtype=np.float64),
        global_positions=np.asarray(positions_global(anim), dtype=np.float64),
    )


# ── user sidecars ─────────────────────────────────────────────────────────────

def _read_json(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass
class SpeciesOverride:
    """One species' row of a name-keyed sidecar, with the hash it was made against."""

    entries: dict = field(default_factory=dict)
    skeleton_hash: Optional[str] = None

    def stale(self, current_hash: str) -> bool:
        return self.skeleton_hash is not None and self.skeleton_hash != current_hash


def load_species_sidecar(root: str, file_name: str) -> dict[str, SpeciesOverride]:
    """``{cond key: SpeciesOverride}`` from ``<root>/<file_name>``.

    Rows are ``{"<cond key>": {"skeleton_hash": ..., "joints": {name: {...}}}}``.
    """
    rows = _read_json(os.path.join(root, file_name))
    out = {}
    for key, row in rows.items():
        row = dict(row or {})
        stored_hash = row.pop("skeleton_hash", None)
        out[key] = SpeciesOverride(entries=row, skeleton_hash=stored_hash)
    return out


class ProfileSchemaError(ValueError):
    pass


def read_profiles_file(path: str) -> dict:
    """``{cond key: profile}`` of a ``skeleton_profiles.json``; empty when it does not
    exist, ``ProfileSchemaError`` when it was written with another ``SCHEMA_VERSION``."""
    data = _read_json(path)
    if not data:
        return {}
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ProfileSchemaError(
            f"{path} has profile schema {version}, expected {SCHEMA_VERSION}; "
            "rebuild it with python -m motion_edit.build_profiles")
    return data.get("profiles", {})


def load_profiles(root: str) -> dict:
    return read_profiles_file(os.path.join(root, PROFILES_FILE))


def write_species_override(root: str, file_name: str, cond_key: str, joints: dict,
                           current_hash: str) -> str:
    """Set (or, with no joint, drop) one species' row of a name-keyed override sidecar
    (``joint_parts_overrides.json``)."""
    path = os.path.join(root, file_name)
    rows = _read_json(path)
    if joints:
        rows[cond_key] = {"joints": dict(sorted(joints.items())), "skeleton_hash": current_hash}
    else:
        rows.pop(cond_key, None)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(dict(sorted(rows.items())), handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    os.replace(tmp, path)
    return path
