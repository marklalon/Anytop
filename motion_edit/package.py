"""Edit Package: a decomposed generation result, self-contained on disk.

A package is a directory ``<clip>.edit/`` holding ``manifest.json`` (metadata,
parameter specs, contact provenance, diagnostics) and ``data.npz`` (the arrays
the runtime composes from, plus the source features and cond subset the
server re-decomposes from).  Layout: section 4.2 of
``docs/skeleton_profile_and_motion_edit_runtime.md``.

This module only reads and writes packages; it imports neither torch nor the
decode path, so the runtime can load a package without either.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

# Bump when the package arrays or their meaning change; the runtime refuses
# every other version.
RUNTIME_VERSION = 2

PACKAGE_SUFFIX = ".edit"
MANIFEST_FILE = "manifest.json"
DATA_FILE = "data.npz"

# Amplitude groups of section 5.1 (``amp.<group>``), plus the joints no
# amplitude slider reaches.
CHAIN_GROUPS = ("root", "legs", "arms", "axial", "tail", "wings", "other")

# A chain offset this close to pi flips sides when scaled (section 4.1 step 4).
NEAR_PI = 0.8 * np.pi


class PackageVersionError(ValueError):
    pass


@dataclass
class EditPackage:
    manifest: dict
    arrays: dict = field(default_factory=dict)

    # ── convenience ──────────────────────────────────────────────────────
    @property
    def frame_count(self) -> int:
        return int(self.arrays["base_rot"].shape[0])

    @property
    def joint_count(self) -> int:
        return int(self.arrays["parents"].shape[0])

    @property
    def fps(self) -> float:
        return float(self.manifest["fps"])

    @property
    def is_loop(self) -> bool:
        return bool(self.manifest["is_loop"])

    @property
    def root(self) -> int:
        return int(self.manifest["translation_root_index"])

    def __getitem__(self, key: str) -> np.ndarray:
        return self.arrays[key]

    # ── disk ─────────────────────────────────────────────────────────────
    def save(self, directory: str) -> str:
        os.makedirs(directory, exist_ok=True)
        data_path = os.path.join(directory, DATA_FILE)
        tmp = data_path + ".tmp.npz"
        np.savez_compressed(tmp, **self.arrays)
        os.replace(tmp, data_path)
        manifest_path = os.path.join(directory, MANIFEST_FILE)
        tmp = manifest_path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.manifest, handle, ensure_ascii=False, indent=1)
            handle.write("\n")
        os.replace(tmp, manifest_path)
        return directory

    @classmethod
    def load(cls, directory: str) -> "EditPackage":
        with open(os.path.join(directory, MANIFEST_FILE), "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        version = manifest.get("runtime_version")
        if version != RUNTIME_VERSION:
            raise PackageVersionError(
                f"{directory}: package runtime_version {version} != runtime {RUNTIME_VERSION}; "
                "re-decompose it")
        with np.load(os.path.join(directory, DATA_FILE), allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        return cls(manifest, arrays)


def is_package_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, MANIFEST_FILE)) and os.path.isfile(os.path.join(path, DATA_FILE))


def find_packages(root: str) -> list[str]:
    """Package directories under ``root`` (recursive), as paths relative to it, sorted."""
    found = []
    for current, dirs, _ in os.walk(root):
        for name in list(dirs):
            path = os.path.join(current, name)
            if is_package_dir(path):
                found.append(os.path.relpath(path, root).replace(os.sep, "/"))
                dirs.remove(name)
    return sorted(found)


# ── cond subset (JSON inside data.npz) ───────────────────────────────────────

def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def encode_json(value) -> np.ndarray:
    """A JSON document as a 0-d unicode array (npz without pickle)."""
    return np.array(json.dumps(_jsonable(value), ensure_ascii=False))


def decode_json(array: np.ndarray):
    return json.loads(str(array[()]))
