#!/usr/bin/env python3
"""
precheck_dataset.py

Pre-flight check for a new raw motion dataset, run BEFORE rendering, labelling
and preprocessing (step 1 of dataset/readme.txt). It catches the problems that
otherwise surface only deep inside preprocess_and_validate -- or never, as a
silently skipped file or a joint whose name reaches T5 as noise.

No Blender is needed: GLB/GLTF files are read straight from their JSON and
binary chunks, and every name-derived verdict comes from the preprocessing
pipeline's own rules (fbx_filename_rules, joint_name_canonical,
joint_embedding_text, physics_joint_annotation), so the precheck cannot drift
from what preprocessing actually does.

Checks
------
Layout      one sub-directory per species, species names usable as cond keys
            (a '_' in the name makes '<Species>_<Action>' stems unusable), FBX
            flagged for conversion; other files and sub-folders are ignored, as
            preprocessing ignores them.
Filenames   rest-pose reference file present; every file survives
            ``should_skip_anim`` (an ``Action_Suffix`` stem reads as a variant
            codename and is SKIPPED); no two files normalize to one clip name;
            no rig noise words ("Ani", "Anim") or foreign prefixes in the clip
            name.  A rename that passes the pipeline rules is suggested.
Content     a skin exists; animation present, one per file, at 30 fps, long
            enough, not a still pose; MAX_JOINTS after the structural drops.
            Like the Blender loader, only the largest root subtree is kept
            ("null" wrapper roots skipped, "*mesh*" children cut); everything
            else in the armature is ignored by every check.
Consistency every file of a species carries the same joints, hierarchy and
            bind pose.
Joint names (the main one) duplicate / empty / non-ASCII / over-long names;
            each joint is run through the pipeline canonicalizer and its T5
            embedding text is checked against the words the training corpus
            carries.  An unseen word is sorted by how T5 tokenizes it: a
            fragmented word next to a known one is a misspelling ("Lag" ->
            "Leg", "Feller" -> "Feeler"), a short fragmented one is an opaque
            code ("Ksb", "Dm"), a whole word is new vocabulary.  Also: codes
            stamped on most joints of a rig, texts left with no body-part word,
            side codes the side detector does not read ("LegMR" stays
            'center'), name side vs geometry side conflicts, unpaired
            left/right names, blanked joints that parent anatomy, IK helpers
            that pass for a body part ("Foot_IK" -> "Foot"), canonical-name
            collisions, and Japanese romaji words whose gated mapping stays off.

The per-joint table (raw -> canonical -> embedding text) is printed with
``--joints`` and always written into the ``--report`` JSON.

Exit status is 1 when any ERROR is found, else 0.

Usage::

    .venv/Scripts/python.exe Anytop/tools/precheck_dataset.py E:/Dataset/Taobao_20260924
    .venv/Scripts/python.exe Anytop/tools/precheck_dataset.py E:/Dataset/Taobao_20260924 \\
        --filter "Pet_*" --joints --report precheck.json
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import fnmatch
import io
import json
import re
import struct
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Put the Anytop package root on sys.path so data_loaders / utils imports
# resolve to the Anytop copies (NOT the top-level pcvg utils).
_ANYTOP_DIR = Path(__file__).resolve().parent.parent
if str(_ANYTOP_DIR) not in sys.path:
    sys.path.insert(0, str(_ANYTOP_DIR))

from data_loaders.truebones.truebones_utils.animation_utils import (  # noqa: E402
    find_end_site_joints,
    find_prop_socket_joints,
)
from data_loaders.truebones.truebones_utils.dataset_pipeline import source_clip_name  # noqa: E402
from motion_lib.root_collapse import collapse_root_skeleton  # noqa: E402
from data_loaders.truebones.truebones_utils.fbx_filename_rules import (  # noqa: E402
    _is_tpose_reference_path,
    _strip_leading_object_prefix,
    find_tpose_reference_path,
    should_skip_anim,
)
from data_loaders.truebones.truebones_utils.joint_name_canonical import (  # noqa: E402
    JAPANESE_GATED_REPLACEMENTS,
    is_japanese_style_naming,
    normalize_joint_name,
    refresh_joint_metadata_in_object_cond,
)
from data_loaders.truebones.truebones_utils.param_utils import MAX_JOINTS  # noqa: E402
from data_loaders.truebones.truebones_utils.face_orientation import (  # noqa: E402
    calculate_root_quat,
    resolve_face_joints,
    resolve_forward_reference_joints,
)
from data_loaders.truebones.truebones_utils.ignore_warnings import skip_orientation_detection  # noqa: E402
from data_loaders.truebones.truebones_utils.physics_joint_annotation import (  # noqa: E402
    detect_joint_side,
)
from data_loaders.truebones.truebones_utils.joint_embedding_text import (  # noqa: E402
    clean_embedding_token,
    build_joint_embedding_texts,
)

DEFAULT_REFERENCE_COND = _ANYTOP_DIR / "dataset" / "merged" / "cond.npy"
MOTION_EXTENSIONS = {".glb", ".gltf", ".fbx"}
TARGET_FPS = 30.0
FPS_TOLERANCE = 0.1
# Mirrors the preprocessing defaults: shorter clips are filtered out.
MIN_FRAMES = 10
# Blender stores a bone name in a fixed 64-byte buffer (63 + NUL); a longer
# name is truncated on import, which can make two bones collide.
BLENDER_NAME_MAX_BYTES = 63
# A still clip moves no channel further than this (radians / rig-relative units).
STILL_ROTATION_EPS = 1e-3
STILL_TRANSLATION_EPS = 1e-3
# Bind pose of two files of one species differs when a joint moves further than
# this fraction of the rig's rest extent.
BIND_POSE_RELATIVE_TOLERANCE = 1e-3
# Cond keys are '<namespace>/<species>', and species names also become file
# tokens; keep them to the characters dataset_sources accepts.
SPECIES_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
# Words that tag a file as an animation without naming the action. They end
# up verbatim inside the clip name, which is the action label's only source.
CLIP_NOISE_SEGMENT_RE = re.compile(r"^(ani|anim|anims|animation|animations)(\d*)$", re.IGNORECASE)
# A short all-caps leading segment that is not the species name is almost always
# a pack or species code ("DF_Fly" on Dragonfly).
CODE_PREFIX_RE = re.compile(r"^[A-Z]{2,4}$")

# Unseen embedding words this short are rig codes ("Ksb", "Dm", "Fx"), not words.
SHORT_CODE_MAX_LEN = 3
# An unseen word on at least this share of a rig's joints is a stamped code.
RIG_STAMP_RATIO = 0.5
# Left/right name counts may differ by this much before it is worth a warning.
SIDE_COUNT_SLACK = 2
# Blanked joints this close to the root are rig roots / centre-of-gravity nodes.
ROOT_HELPER_MAX_DEPTH = 2
# Name tokens that mark an IK / control helper rather than a body part.
HELPER_MARKER_TOKENS = frozenset({"ik", "target", "pole", "ctrl", "control", "controler"})
# Names made only of these are rig roots / centre-of-gravity nodes wherever
# they sit (Unity packs hang "Bip001" under a mount and a dummy).
ROOT_MARKER_TOKENS = frozenset({
    "bip", "dummy", "root", "cg", "cog", "com", "hub", "main", "base", "center",
    "all", "locator", "null", "rig", "mount", "bone", "joint", "point",
})
# Two-letter codes that carry a side: a limb quadrant (front/back/middle x
# left/right) or a face corner (top/bottom x left/right).
SIDE_CODE_TOKENS = frozenset({
    "lf", "rf", "lb", "rb", "lm", "rm", "fl", "fr", "bl", "br", "ml", "mr", "tl", "tr",
})

ERROR = "ERROR"
WARN = "WARN"
INFO = "INFO"
_LEVEL_ORDER = {ERROR: 0, WARN: 1, INFO: 2}
_LEVEL_COLOR = {ERROR: "\033[91m", WARN: "\033[93m", INFO: "\033[96m"}
_RESET = "\033[0m"


@dataclass
class Finding:
    level: str
    species: str
    check: str
    message: str
    file: str = ""

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v != ""}


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    joint_tables: dict[str, list[dict]] = field(default_factory=dict)
    clips: dict[str, list[dict]] = field(default_factory=dict)

    def add(self, level: str, species: str, check: str, message: str, file: str = "") -> None:
        self.findings.append(Finding(level, species, check, message, file))


# ---------------------------------------------------------------------------
# glTF reading
# ---------------------------------------------------------------------------

_COMPONENT_DTYPE = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_TYPE_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


class GltfFile:
    """The JSON document of a GLB/GLTF plus lazy typed accessor reads."""

    def __init__(self, path: Path):
        self.path = path
        self._buffers: dict[int, bytes] = {}
        if path.suffix.lower() == ".glb":
            data = path.read_bytes()
            magic, _version, _length = struct.unpack_from("<III", data, 0)
            if magic != 0x46546C67:
                raise ValueError("not a GLB file (bad magic)")
            offset = 12
            self.doc = None
            while offset < len(data):
                chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
                chunk = data[offset + 8: offset + 8 + chunk_length]
                if chunk_type == 0x4E4F534A:
                    self.doc = json.loads(chunk.decode("utf-8"))
                elif chunk_type == 0x004E4942:
                    self._buffers[0] = chunk
                offset += 8 + chunk_length
            if self.doc is None:
                raise ValueError("GLB has no JSON chunk")
        else:
            self.doc = json.loads(path.read_text(encoding="utf-8"))

    def _buffer(self, index: int) -> bytes:
        if index not in self._buffers:
            uri = self.doc["buffers"][index].get("uri")
            if not uri or uri.startswith("data:"):
                raise ValueError(f"buffer {index} is embedded/missing; unsupported")
            self._buffers[index] = (self.path.parent / uri).read_bytes()
        return self._buffers[index]

    def accessor(self, index: int) -> np.ndarray:
        acc = self.doc["accessors"][index]
        width = _TYPE_WIDTH[acc["type"]]
        dtype = np.dtype(_COMPONENT_DTYPE[acc["componentType"]])
        count = int(acc["count"])
        if "bufferView" not in acc:
            return np.zeros((count, width), dtype=np.float64)
        view = self.doc["bufferViews"][acc["bufferView"]]
        buffer = self._buffer(int(view.get("buffer", 0)))
        start = int(view.get("byteOffset", 0)) + int(acc.get("byteOffset", 0))
        element_size = dtype.itemsize * width
        stride = int(view.get("byteStride", 0)) or element_size
        if stride == element_size:
            flat = np.frombuffer(buffer, dtype=dtype, count=count * width, offset=start)
            out = flat.reshape(count, width)
        else:
            rows = [
                np.frombuffer(buffer, dtype=dtype, count=width, offset=start + i * stride)
                for i in range(count)
            ]
            out = np.stack(rows) if rows else np.zeros((0, width), dtype=dtype)
        out = out.astype(np.float64)
        if acc.get("normalized") and dtype.kind in "iu":
            out /= float(np.iinfo(dtype).max)
        return out


def _node_local_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T
    t = np.asarray(node.get("translation", (0.0, 0.0, 0.0)), dtype=np.float64)
    x, y, z, w = node.get("rotation", (0.0, 0.0, 0.0, 1.0))
    s = np.asarray(node.get("scale", (1.0, 1.0, 1.0)), dtype=np.float64)
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    m = np.eye(4)
    m[:3, :3] = rot * s[None, :]
    m[:3, 3] = t
    return m


@dataclass
class Skeleton:
    """The joint set the Blender loader would keep, in its depth-first order."""
    names: list[str]
    parents: np.ndarray
    rest_positions: np.ndarray            # scene space, node default pose
    bind_positions: np.ndarray | None     # from inverse bind matrices
    node_indices: list[int]


@dataclass
class FileFacts:
    path: Path
    skeleton: Skeleton | None = None
    skin_count: int = 0
    distinct_skin_rigs: int = 0
    animations: list[dict] = field(default_factory=list)
    all_joint_names: list[str] = field(default_factory=list)  # every armature bone, kept or not
    error: str = ""


def _read_file_facts(path: Path) -> FileFacts:
    facts = FileFacts(path=path)
    if path.suffix.lower() == ".fbx":
        return facts
    try:
        gltf = GltfFile(path)
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        facts.error = f"cannot parse: {exc}"
        return facts

    doc = gltf.doc
    nodes = doc.get("nodes", [])
    skins = doc.get("skins", [])
    facts.skin_count = len(skins)
    if not skins:
        return facts

    parent_of = {}
    for index, node in enumerate(nodes):
        for child in node.get("children", ()):
            parent_of[int(child)] = index

    # Scene-space matrices of every node (the default pose).
    world: dict[int, np.ndarray] = {}

    def world_matrix(index: int) -> np.ndarray:
        if index not in world:
            local = _node_local_matrix(nodes[index])
            parent = parent_of.get(index)
            world[index] = local if parent is None else world_matrix(parent) @ local
        return world[index]

    # Blender merges skins that share joints into one armature; the loader
    # then reads only one armature. Joint sets that are pairwise disjoint are
    # separate rigs.
    joint_sets = [frozenset(int(j) for j in skin.get("joints", ())) for skin in skins]
    merged: list[set[int]] = []
    for joint_set in joint_sets:
        overlapping = [group for group in merged if group & joint_set]
        combined = set(joint_set)
        for group in overlapping:
            combined |= group
            merged.remove(group)
        merged.append(combined)
    facts.distinct_skin_rigs = len(merged)
    joints = max(merged, key=len)

    # The armature holds the joints plus any non-joint node sitting between two
    # joints (the importer turns those into bones too).
    bone_nodes = set(joints)
    for joint in list(joints):
        chain = []
        cursor = parent_of.get(joint)
        while cursor is not None and cursor not in joints:
            chain.append(cursor)
            cursor = parent_of.get(cursor)
        if cursor is not None:
            bone_nodes.update(chain)
    facts.all_joint_names = [nodes[i].get("name", "") for i in sorted(bone_nodes)]

    inverse_bind: dict[int, np.ndarray] = {}
    for skin in skins:
        if "inverseBindMatrices" not in skin:
            continue
        matrices = gltf.accessor(int(skin["inverseBindMatrices"]))
        for joint, flat in zip(skin["joints"], matrices):
            inverse_bind.setdefault(int(joint), flat.reshape(4, 4).T)

    children: dict[int, list[int]] = defaultdict(list)
    roots = []
    for node_index in sorted(bone_nodes):
        parent = parent_of.get(node_index)
        if parent in bone_nodes:
            children[parent].append(node_index)
        else:
            roots.append(node_index)
    # Keep glTF child order, which the importer preserves.
    for parent in list(children):
        order = [int(c) for c in nodes[parent].get("children", ()) if int(c) in bone_nodes]
        children[parent] = order

    def subtree(index: int) -> list[int]:
        out, stack = [], [index]
        while stack:
            current = stack.pop()
            out.append(current)
            stack.extend(reversed(children.get(current, [])))
        return out

    # extract_armature_skeleton_data: skip 'null' wrapper roots (promoting their
    # children), keep only the largest root subtree, cut '*mesh*' children.
    candidate_roots = []
    for root in roots:
        if nodes[root].get("name", "").lower() == "null":
            candidate_roots.extend(children.get(root, []))
        else:
            candidate_roots.append(root)
    if not candidate_roots:
        facts.error = "no usable root joint after skipping 'null' roots"
        return facts
    main_root = max(candidate_roots, key=lambda r: len(subtree(r)))

    ordered: list[int] = []

    def append_preorder(index: int) -> None:
        ordered.append(index)
        for child in children.get(index, []):
            if "mesh" in nodes[child].get("name", "").lower():
                continue
            append_preorder(child)

    append_preorder(main_root)
    position = {index: i for i, index in enumerate(ordered)}
    parents = np.array([position.get(parent_of.get(index), -1) for index in ordered], dtype=np.int64)
    rest = np.array([world_matrix(index)[:3, 3] for index in ordered], dtype=np.float64)
    # The loader reads bones in armature-OBJECT space: the object's scale (a
    # Truebones 0.01 on one file of a species and not the next), rotation and
    # placement never reach the bone data. Measure in the same space -- the
    # armature for the rest pose and the animated channels, the root joint's
    # own bind frame for the bind pose, which cancels whatever sits outside the
    # skeleton (object transform, or a scale baked into the inverse binds).
    armature = parent_of.get(main_root)
    armature_inv = np.linalg.inv(world_matrix(armature)) if armature is not None else np.eye(4)
    rest_local = np.array([(armature_inv @ world_matrix(index))[:3, 3] for index in ordered], dtype=np.float64)
    local_extent = float(np.ptp(rest_local, axis=0).max()) if len(rest_local) else 1.0
    bind = None
    if all(index in inverse_bind for index in ordered):
        root_frame = inverse_bind[ordered[0]]
        bind = np.array(
            [(root_frame @ np.linalg.inv(inverse_bind[index]))[:3, 3] for index in ordered],
            dtype=np.float64,
        )
    facts.skeleton = Skeleton(
        names=[nodes[i].get("name", "") for i in ordered],
        parents=parents,
        rest_positions=rest,
        bind_positions=bind,
        node_indices=ordered,
    )

    ordered_set = set(ordered)
    for anim_index, animation in enumerate(doc.get("animations", [])):
        samplers = animation.get("samplers", [])
        times: list[np.ndarray] = []
        moving = False
        targets_skeleton = 0
        for channel in animation.get("channels", []):
            target = channel.get("target", {})
            node = target.get("node")
            if node is None or int(node) not in ordered_set:
                continue
            targets_skeleton += 1
            sampler = samplers[int(channel["sampler"])]
            key_times = gltf.accessor(int(sampler["input"]))[:, 0]
            times.append(key_times)
            if moving:
                continue
            values = gltf.accessor(int(sampler["output"]))
            if sampler.get("interpolation") == "CUBICSPLINE":
                values = values.reshape(-1, 3, values.shape[-1])[:, 1]
            if len(values) < 2:
                continue
            spread = np.abs(values - values[:1]).max()
            if target.get("path") == "rotation":
                moving = spread > STILL_ROTATION_EPS
            elif target.get("path") == "translation":
                moving = spread > STILL_TRANSLATION_EPS * max(local_extent, 1e-6)
            elif target.get("path") == "scale":
                moving = spread > STILL_ROTATION_EPS
        all_times = np.unique(np.concatenate(times)) if times else np.zeros(0)
        deltas = np.diff(all_times)
        deltas = deltas[deltas > 1e-6]
        fps = float(1.0 / np.median(deltas)) if deltas.size else 0.0
        facts.animations.append({
            "index": anim_index,
            "name": animation.get("name", ""),
            "skeleton_channels": targets_skeleton,
            "frames": int(all_times.size),
            "duration": float(all_times[-1] - all_times[0]) if all_times.size else 0.0,
            "fps": fps,
            "moving": moving,
        })
    return facts


# ---------------------------------------------------------------------------
# Reference vocabulary
# ---------------------------------------------------------------------------

def _load_reference_vocabulary(path: Path) -> tuple[Counter, set[str]]:
    """Embedding words the training corpus carries -> number of species using each."""
    cond = np.load(path, allow_pickle=True).item()
    word_species: Counter = Counter()
    species_names = set()
    for key, entry in cond.items():
        species_names.add(str(key).rsplit("/", 1)[-1].lower())
        texts = (entry.get("joints_names_embs_meta") or {}).get("embedding_texts") or ()
        words = {word for text in texts for word in str(text).split()}
        word_species.update(words)
    return word_species, species_names


def _load_reference_root_promote_depths(path: Path, dataset_dir: Path) -> dict[str, int]:
    """Species (lower-case) -> ``root_promote_depth`` the reference cond recorded
    for THIS raw dataset.

    Preprocessing folds wrapper roots by measuring which joint the clips travel
    on, which a precheck cannot redo; for a species it has already processed,
    that measured depth is the answer. An entry belongs to this raw directory
    when its processed ``dataset_root`` is a sibling of it
    ("dataset/truebones/zoo/truebones_processed" next to ".../Truebone_Z-OO").
    """
    cond = np.load(path, allow_pickle=True).item()
    raw_parent = dataset_dir.resolve().parent
    depths = {}
    for entry in cond.values():
        dataset_root = entry.get("dataset_root")
        if not dataset_root:
            continue
        processed = Path(dataset_root)
        if not processed.is_absolute():
            processed = _ANYTOP_DIR / processed
        if processed.resolve().parent != raw_parent:
            continue
        depths[str(entry.get("species_name", "")).lower()] = int(entry.get("root_promote_depth") or 0)
    return depths


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _strip_species_prefix(species: str, stem: str) -> str:
    """The stem without a leading species name, however it is separated."""
    compact_species = re.sub(r"[^0-9a-z]+", "", species.lower())
    segments = [s for s in re.split(r"[^0-9A-Za-z]+", stem) if s]
    for count in range(len(segments) - 1, 0, -1):
        if "".join(segments[:count]).lower() == compact_species:
            return "_".join(segments[count:])
    return _strip_leading_object_prefix(species, stem)


def _suggest_stem(species: str, stem: str) -> str | None:
    """A stem the pipeline rules keep and turn into ``<Species>_<Action>``, or None.

    ``<Species>_<Action>`` is preferred. A species name containing '_' cannot
    use it -- the prefix rule strips only its first segment and the remainder
    reads as a variant codename -- so the bare ``<Action>`` is offered instead.
    """
    action = _strip_species_prefix(species, stem)
    segments = [s for s in re.split(r"[^0-9A-Za-z]+", action) if s]
    if action == stem and len(segments) > 1 and CODE_PREFIX_RE.match(segments[0]):
        segments = segments[1:]
    cleaned: list[str] = []
    for segment in segments:
        noise = CLIP_NOISE_SEGMENT_RE.match(segment)
        if noise:
            if noise.group(2) and cleaned:
                cleaned[-1] += noise.group(2)
            continue
        cleaned.append(segment[0].upper() + segment[1:])
    if not cleaned:
        return None
    action_name = "".join(cleaned)
    for candidate in (f"{species}_{action_name}", action_name):
        with contextlib.redirect_stdout(io.StringIO()):
            skipped = should_skip_anim(f"{candidate}.glb", species)
        if not skipped and source_clip_name(species, f"{candidate}.glb") == f"{species}_{action_name}":
            return candidate
    return None


def _check_layout(root: Path, species_dirs: list[Path], report: Report, reference_species: set[str]) -> None:
    # Readmes, licences and shortcuts in the root are expected; only a motion
    # file there is a misplaced clip that no species will pick up.
    for entry in sorted(root.iterdir()):
        if entry.is_file() and entry.suffix.lower() in MOTION_EXTENSIONS:
            report.add(WARN, "-", "layout", f"motion file in dataset root belongs to no species: {entry.name}")
    for species_dir in species_dirs:
        name = species_dir.name
        if not SPECIES_NAME_RE.match(name):
            report.add(ERROR, name, "layout",
                       "species directory name must match [A-Za-z0-9_-]+ (it becomes the cond key and file token)")
        for entry in sorted(species_dir.iterdir()):
            if entry.is_dir():
                # Texture folders and the like; preprocessing never looks inside.
                continue
            if entry.suffix.lower() == ".fbx":
                report.add(WARN, name, "layout",
                           "FBX source; convert to GLB first (tools/dataset_cleanup/convert_fbx_2_glb.py) -- "
                           "content checks skip it", entry.name)
    # Same bare name in another dataset is legal (the namespace keeps the cond
    # keys apart) but makes bare-name --filter / species lookups ambiguous.
    shared = sorted(d.name for d in species_dirs if d.name.lower() in reference_species)
    if shared:
        report.add(INFO, "-", "layout",
                   f"{len(shared)} species name(s) already exist in the reference cond (fine across namespaces; "
                   f"bare-name lookups become ambiguous): {', '.join(shared[:12])}{' ...' if len(shared) > 12 else ''}")


def _check_filenames(species: str, files: list[Path], report: Report) -> tuple[Path | None, list[Path]]:
    """Returns (rest-pose reference file, files that would become clips)."""
    if not files:
        report.add(ERROR, species, "filenames", "no GLB/GLTF/FBX files")
        return None, []
    for path in files:
        if not path.name.isascii():
            report.add(WARN, species, "filenames",
                       "file name has non-ASCII characters; they are dropped from the clip name", path.name)

    anim_files = [str(p) for p in files]
    reference = Path(find_tpose_reference_path(anim_files))
    if not _is_tpose_reference_path(str(reference)):
        report.add(WARN, species, "filenames",
                   f"no T-pose/bind-pose file (stem containing 'TPose'); the rest pose falls back to "
                   f"'{reference.name}'")

    clip_sources: dict[str, list[str]] = defaultdict(list)
    kept: list[Path] = []
    clips = []
    for path_str in anim_files:
        path = Path(path_str)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            skipped = should_skip_anim(path_str, species)
        clip = source_clip_name(species, path_str)
        stem = path.stem
        suggestion = _suggest_stem(species, stem)
        if skipped:
            reason = buffer.getvalue().strip().split(": ", 1)[-1] or "skipped"
            hint = f"; rename e.g. to '{suggestion}{path.suffix}'" if suggestion else ""
            if "_" in species and _strip_species_prefix(species, stem) != stem:
                hint += (f" (a '{species}_' prefix never works: only its first segment is stripped, "
                         f"use '<Action>' or '{species.replace('_', '')}_<Action>')")
            report.add(ERROR, species, "filenames",
                       f"preprocessing will SKIP this file ({reason}){hint}", path.name)
            clips.append({"file": path.name, "clip": None, "skipped": reason, "suggested_stem": suggestion})
            continue
        kept.append(path)
        clip_sources[clip].append(path.name)
        clips.append({"file": path.name, "clip": clip, "suggested_stem": suggestion})

        action = _strip_species_prefix(species, stem)
        segments = [s for s in re.split(r"[^0-9A-Za-z]+", action) if s]
        issues = []
        if any(CLIP_NOISE_SEGMENT_RE.match(s) for s in segments):
            issues.append("an animation marker ('Ani'/'Anim') that is not an action word")
        if action == stem and segments and CODE_PREFIX_RE.match(segments[0]):
            issues.append(f"a leading code '{segments[0]}' that is not the species name")
        if issues:
            hint = f"; suggested '{suggestion}{path.suffix}'" if suggestion and suggestion != stem else ""
            report.add(WARN, species, "filenames",
                       f"clip name '{clip}' carries {' and '.join(issues)}{hint}", path.name)
    report.clips[species] = clips

    for clip, sources in clip_sources.items():
        if len(sources) > 1:
            report.add(ERROR, species, "filenames",
                       f"{len(sources)} files normalize to the same clip '{clip}': {', '.join(sources)}")
    if not kept:
        report.add(ERROR, species, "filenames", "no file survives the filename rules; the species produces no clips")
    return reference, kept


def _check_file_content(species: str, facts: FileFacts, is_reference: bool, report: Report) -> None:
    name = facts.path.name
    if facts.path.suffix.lower() == ".fbx":
        return
    if facts.error:
        report.add(ERROR, species, "content", facts.error, name)
        return
    if not facts.skin_count or facts.skeleton is None:
        report.add(ERROR, species, "content", "no skin: Blender imports no armature from this file", name)
        return
    if facts.distinct_skin_rigs > 1:
        report.add(WARN, species, "content",
                   f"{facts.distinct_skin_rigs} disjoint skinned rigs; the loader reads only one armature", name)

    animations = facts.animations
    if is_reference:
        return
    if not animations:
        report.add(ERROR, species, "content", "no animation", name)
        return
    if len(animations) > 1:
        report.add(WARN, species, "content",
                   f"{len(animations)} animations in one file; preprocessing reads one -- split to one GLB per clip "
                   f"({', '.join(a['name'] or str(a['index']) for a in animations[:6])})", name)
    anim = animations[0]
    if anim["skeleton_channels"] == 0:
        report.add(ERROR, species, "content", "the animation drives no skeleton joint", name)
        return
    if anim["fps"] and abs(anim["fps"] - TARGET_FPS) > FPS_TOLERANCE:
        report.add(WARN, species, "content", f"{anim['fps']:.2f} fps, expected {TARGET_FPS:.0f}", name)
    if anim["frames"] < MIN_FRAMES:
        report.add(ERROR, species, "content",
                   f"only {anim['frames']} key frame(s); clips under {MIN_FRAMES} frames are filtered out", name)
    if not anim["moving"]:
        report.add(WARN, species, "content", "still pose: no joint channel moves", name)


def _check_species_consistency(species: str, facts_list: list[FileFacts], reference: Path | None,
                               report: Report) -> FileFacts | None:
    usable = [f for f in facts_list if f.skeleton is not None]
    if not usable:
        return None
    base = next((f for f in usable if reference is not None and f.path == reference), usable[0])
    base_skeleton = base.skeleton
    # Extent in the same root-bind frame the bind poses are compared in.
    base_bind = base_skeleton.bind_positions
    extent = float(np.ptp(base_bind, axis=0).max()) if base_bind is not None and len(base_bind) else 1.0
    for facts in usable:
        if facts is base:
            continue
        skeleton = facts.skeleton
        if skeleton.names != base_skeleton.names:
            missing = [n for n in base_skeleton.names if n not in skeleton.names]
            extra = [n for n in skeleton.names if n not in base_skeleton.names]
            detail = []
            if missing:
                detail.append(f"missing {missing[:6]}")
            if extra:
                detail.append(f"extra {extra[:6]}")
            if not detail:
                detail.append("same joints in a different order")
            report.add(ERROR, species, "consistency",
                       f"joint set differs from {base.path.name}: {'; '.join(detail)}", facts.path.name)
            continue
        if not np.array_equal(skeleton.parents, base_skeleton.parents):
            report.add(ERROR, species, "consistency", f"hierarchy differs from {base.path.name}", facts.path.name)
            continue
        if skeleton.bind_positions is not None and base_skeleton.bind_positions is not None:
            error = float(np.abs(skeleton.bind_positions - base_skeleton.bind_positions).max())
            if error > BIND_POSE_RELATIVE_TOLERANCE * max(extent, 1e-6):
                worst = int(np.abs(skeleton.bind_positions - base_skeleton.bind_positions).max(axis=1).argmax())
                report.add(WARN, species, "consistency",
                           f"bind pose differs from {base.path.name} (max {error:.4g}, "
                           f"{error / max(extent, 1e-6):.1%} of rig extent, at '{skeleton.names[worst]}'); "
                           "see tools/dataset_cleanup/validate_tpose_bind_pose.py", facts.path.name)
    return base


def _edit_distance(left: str, right: str) -> int:
    """Levenshtein distance with adjacent transpositions (optimal string alignment)."""
    rows = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i in range(len(left) + 1):
        rows[i][0] = i
    for j in range(len(right) + 1):
        rows[0][j] = j
    for i in range(1, len(left) + 1):
        for j in range(1, len(right) + 1):
            cost = 0 if left[i - 1] == right[j - 1] else 1
            rows[i][j] = min(rows[i - 1][j] + 1, rows[i][j - 1] + 1, rows[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and left[i - 1] == right[j - 2] and left[i - 2] == right[j - 1]:
                rows[i][j] = min(rows[i][j], rows[i - 2][j - 2] + 1)
    return rows[-1][-1]


class T5Pieces:
    """How the pipeline's T5 tokenizer splits one embedding word.

    A word T5 holds as one whole piece ("Hat", "Flower") is a real word even
    when it sits one edit away from a known one; a word it has to assemble
    from fragments ("La"+"g", "K"+"s"+"b") is a misspelling or a code.
    Without a local tokenizer every word counts as fragmented.
    """

    def __init__(self, t5_name: str = "t5-base"):
        self._tokenizer = None
        try:
            import logging
            import warnings

            from transformers import T5Tokenizer

            from model.conditioners import _resolve_t5_local_dir

            previous = logging.root.manager.disable
            logging.disable(logging.ERROR)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._tokenizer = T5Tokenizer.from_pretrained(
                        _resolve_t5_local_dir(t5_name), local_files_only=True,
                    )
            finally:
                logging.disable(previous)
        except Exception as exc:  # noqa: BLE001 - optional refinement
            print(f"[WARN] T5 tokenizer unavailable ({exc}); word/code split falls back to spelling only",
                  file=sys.stderr)

    @property
    def available(self) -> bool:
        return self._tokenizer is not None

    def pieces(self, word: str) -> list[str]:
        if self._tokenizer is None:
            return [word, ""]
        return [piece.lstrip("▁") for piece in self._tokenizer.tokenize(word) if piece != "▁"]

    def is_whole_word(self, word: str) -> bool:
        return len(self.pieces(word)) == 1


def _likely_typo_of(word: str, known: list[str]) -> str | None:
    """The known word *word* is one keystroke (two, for long words) away from."""
    if len(word) < 3:
        return None
    budget = 1 if len(word) < 6 else 2
    lowered = word.lower()
    best = None
    for candidate in known:
        if candidate.isdigit() or len(candidate) < 3 or abs(len(candidate) - len(word)) > budget:
            continue
        # A plural or a code that merely extends a word is not a misspelling.
        if candidate.lower().startswith(lowered) or lowered.startswith(candidate.lower()):
            continue
        distance = _edit_distance(lowered, candidate.lower())
        if distance <= budget and (best is None or distance < best[0]):
            best = (distance, candidate)
    return best[1] if best else None


def _check_raw_joint_names(species: str, facts: FileFacts, report: Report) -> None:
    """Spelling problems that break Blender's import or the canonicalizer.

    Only the joints the loader keeps are judged. A duplicate still counts when
    its twin sits in a discarded subtree: Blender renames at import, before the
    loader picks its subtree, so the kept joint may be the one that becomes
    '.001'.
    """
    if facts.skeleton is None:
        return
    source = facts.path.name
    kept = facts.skeleton.names
    counts = Counter(facts.all_joint_names)
    duplicates = sorted({n for n in kept if counts[n] > 1})
    if duplicates:
        report.add(ERROR, species, "joint-names",
                   f"duplicate joint names (Blender renames them '.001'): {', '.join(duplicates)}", source)
    for name in kept:
        if not name.strip():
            report.add(ERROR, species, "joint-names", "a joint has no name (Blender invents one)", source)
        elif not normalize_joint_name(name):
            report.add(ERROR, species, "joint-names",
                       f"'{name}' has no ASCII letters or digits; the canonicalizer reduces it to nothing", source)
        elif not name.isascii():
            report.add(ERROR, species, "joint-names",
                       f"'{name}' has non-ASCII characters; normalize_joint_name drops them, so T5 never sees "
                       "that part of the name", source)
        if len(name.encode("utf-8")) > BLENDER_NAME_MAX_BYTES:
            report.add(ERROR, species, "joint-names",
                       f"'{name}' is longer than {BLENDER_NAME_MAX_BYTES} bytes; Blender truncates it", source)


def _parent_offsets(skeleton: Skeleton) -> np.ndarray:
    """Per-joint offset from the parent; a root keeps its own placement, as in
    the loader, whose root-collapse rule reads a zero root offset as a wrapper."""
    offsets = skeleton.rest_positions.copy()
    for index, parent in enumerate(skeleton.parents):
        if parent >= 0:
            offsets[index] = skeleton.rest_positions[index] - skeleton.rest_positions[parent]
    return offsets


def _facing_aligned_offsets(skeleton: Skeleton, species: str) -> np.ndarray:
    """``_parent_offsets`` turned into the frame preprocessing builds the joint
    metadata in: the rest pose rotated by the same facing correction
    (``calculate_root_quat``) and its root moved onto the vertical axis.

    Side and mirror-pairing geometry assumes the rig mirrors across X, which
    holds only after that correction; Leopard's GLB mirrors across Z, and read
    raw its mane joints pair differently here than in the preprocessed cond.
    """
    positions = skeleton.rest_positions[None].astype(np.float64)
    if not skip_orientation_detection():
        # The facing fallbacks print their own [WARN] lines; preprocessing
        # reports them in its run summary, so here they are only noise.
        with contextlib.redirect_stdout(io.StringIO()):
            face_joints = resolve_face_joints(
                species, skeleton.names, skeleton.parents, rest_positions=positions,
            )
            forward_joint, forward_base_joint = resolve_forward_reference_joints(
                skeleton.names, skeleton.parents, object_type=species, rest_positions=positions,
            )
            quat = calculate_root_quat(
                positions, species,
                face_joint_indx=face_joints,
                forward_joint_index=forward_joint,
                forward_base_joint_index=forward_base_joint,
                emit_warnings=False,
                joint_names=skeleton.names,
                parents=skeleton.parents,
            )
        positions = np.asarray(quat[0:1] * positions[0])[None]
    rest_positions = positions[0]
    root = int(np.flatnonzero(skeleton.parents < 0)[0])
    rest_positions = rest_positions - np.array([rest_positions[root, 0], 0.0, rest_positions[root, 2]])
    return _parent_offsets(Skeleton(
        names=skeleton.names,
        parents=skeleton.parents,
        rest_positions=rest_positions,
        bind_positions=None,
        node_indices=skeleton.node_indices,
    ))


def _without_joints(skeleton: Skeleton, drop: set[int]) -> Skeleton:
    keep = [i for i in range(len(skeleton.names)) if i not in drop]
    remap = {old: new for new, old in enumerate(keep)}
    parents = []
    for old in keep:
        parent = int(skeleton.parents[old])
        while parent >= 0 and parent in drop:
            parent = int(skeleton.parents[parent])
        parents.append(remap.get(parent, -1))
    return Skeleton(
        names=[skeleton.names[i] for i in keep],
        parents=np.asarray(parents, dtype=np.int64),
        rest_positions=skeleton.rest_positions[keep],
        bind_positions=None if skeleton.bind_positions is None else skeleton.bind_positions[keep],
        node_indices=[skeleton.node_indices[i] for i in keep],
    )


def _pipeline_pruned_skeleton(skeleton: Skeleton, promote_depth: int = 0) -> tuple[Skeleton, dict[str, list[str]]]:
    """The skeleton after the structural drops preprocessing applies before naming.

    Same helpers, same order as the loader and ``_load_motion_source``: wrapper
    roots collapse, then detached prop sockets and End-Site leaves are dropped,
    then ``promote_depth`` wrapper joints above the translation root are folded
    away (``promote_translation_root_to_hierarchy_root``). That depth is measured
    from the motion, so the caller passes the one the reference cond recorded,
    or 0 for a species not yet processed.
    """
    # Only the prop-socket drops are reported; wrapper-root and End-Site removal are routine.
    removed: dict[str, list[str]] = {}
    joint_count = len(skeleton.names)
    offsets = _parent_offsets(skeleton)
    identity = np.zeros((1, joint_count, 4))
    identity[..., 0] = 1.0
    collapsed_names = collapse_root_skeleton(
        list(skeleton.names), skeleton.parents, offsets, identity, offsets[None].copy(),
    )[0]
    # Collapsing only ever removes joints; recover which by walking both lists.
    drop, cursor = set(), 0
    for index, name in enumerate(skeleton.names):
        if cursor < len(collapsed_names) and collapsed_names[cursor] == name:
            cursor += 1
        else:
            drop.add(index)
    if cursor < len(collapsed_names):
        # A redundant joint 1 hands its slot to the root's name: joint 1 is the one gone.
        drop = {1} if len(collapsed_names) == joint_count - 1 else set()
    if drop:
        skeleton = _without_joints(skeleton, drop)

    props = find_prop_socket_joints(_parent_offsets(skeleton), skeleton.parents, skeleton.names)
    if props:
        removed["detached prop socket"] = [skeleton.names[i] for i in sorted(props)]
        skeleton = _without_joints(skeleton, set(props))

    end_sites = find_end_site_joints(skeleton.parents, skeleton.names)
    if end_sites:
        skeleton = _without_joints(skeleton, set(end_sites))

    for _ in range(promote_depth):
        root = int(np.flatnonzero(skeleton.parents < 0)[0])
        if int(np.count_nonzero(skeleton.parents == root)) != 1 or len(skeleton.names) < 2:
            break
        skeleton = _without_joints(skeleton, {root})
    return skeleton, removed


def _check_joint_names(species: str, facts: FileFacts, vocabulary: Counter | None, t5: T5Pieces,
                       report: Report, promote_depth: int = 0) -> list[dict]:
    source = facts.path.name
    skeleton, removed = _pipeline_pruned_skeleton(facts.skeleton, promote_depth)
    names = skeleton.names
    if len(names) > MAX_JOINTS:
        report.add(INFO, species, "content",
                   f"{len(names)} joints after the structural drops > MAX_JOINTS={MAX_JOINTS}; "
                   "the deepest leaves get cropped", source)
    for reason, joints in removed.items():
        report.add(INFO, species, "joint-names",
                   f"preprocessing removes {len(joints)} joint(s) ({reason}); their names are not checked: "
                   f"{', '.join(joints[:10])}{' ...' if len(joints) > 10 else ''}", source)

    offsets = _facing_aligned_offsets(skeleton, species)
    cond = {
        "joints_names": list(names),
        "parents": skeleton.parents,
        "offsets": offsets,
        "object_type": species,
        "species_name": species,
    }
    refresh_joint_metadata_in_object_cond(cond)
    texts = build_joint_embedding_texts(cond)
    canonical = cond["canonical_joint_names"]
    side_labels = cond["joint_side_labels"]
    partners = cond["symmetry_partner_indices"]

    rows = []
    unseen: dict[str, list[str]] = defaultdict(list)
    opaque: list[str] = []
    blanked: list[str] = []
    side_conflicts: list[str] = []
    for index, raw in enumerate(names):
        text = texts[index]
        name_side = detect_joint_side(raw)
        geometry_side = side_labels[index]
        row = {
            "index": index,
            "raw_name": raw,
            "parent": names[int(skeleton.parents[index])] if skeleton.parents[index] >= 0 else None,
            "canonical_name": canonical[index],
            "embedding_text": text,
            "side": geometry_side,
            "name_side": name_side,
            "partner": names[int(partners[index])] if int(partners[index]) >= 0 else None,
            "flags": [],
        }
        if not text.strip():
            blanked.append(raw)
            row["flags"].append("blank")
        else:
            words = [w for w in text.split() if w not in ("Left", "Right")]
            if words and all(w.isdigit() or len(w) <= 2 for w in words):
                opaque.append(f"{raw} -> '{text}'")
                row["flags"].append("opaque")
            if vocabulary is not None:
                for word in words:
                    if word.isdigit() or word in vocabulary:
                        continue
                    unseen[word].append(raw)
                    row["flags"].append(f"unseen:{word}")
        if name_side in ("left", "right") and geometry_side in ("left", "right") and name_side != geometry_side:
            side_conflicts.append(f"{raw} (name {name_side}, geometry {geometry_side})")
            row["flags"].append("side-conflict")
        rows.append(row)

    known = sorted(vocabulary) if vocabulary is not None else []
    for word, raws in sorted(unseen.items(), key=lambda item: -len(item[1])):
        where = f"{len(raws)} joint(s): {', '.join(raws[:4])}{' ...' if len(raws) > 4 else ''}"
        if len(raws) >= RIG_STAMP_RATIO * len(names):
            where += f" -- stamped on {len(raws)}/{len(names)} joints, a rig/pack code rather than anatomy"
        whole = t5.is_whole_word(word)
        split = "+".join(t5.pieces(word)) if t5.available and not whole else ""
        typo_of = None if whole else _likely_typo_of(word, known)
        if word.lower() in SIDE_CODE_TOKENS:
            report.add(WARN, species, "joint-names",
                       f"'{word}' is a side/position code left in the text as a word ({where})", source)
        elif typo_of:
            report.add(WARN, species, "joint-names",
                       f"'{word}' looks like a misspelling of the known word '{typo_of}'"
                       f"{f' (T5 reads {split})' if split else ''} ({where})", source)
        elif not whole and len(word) <= SHORT_CODE_MAX_LEN:
            report.add(WARN, species, "joint-names",
                       f"'{word}' is an opaque code, not a word{f' (T5 reads {split})' if split else ''} "
                       f"({where})", source)
        elif len(raws) >= RIG_STAMP_RATIO * len(names):
            report.add(WARN, species, "joint-names",
                       f"'{word}' is unseen in the training corpus ({where})", source)
        else:
            close = difflib.get_close_matches(word, known, n=3, cutoff=0.6)
            hint = f"; nearest known spelling: {', '.join(close)}" if close else ""
            report.add(INFO, species, "joint-names",
                       f"new word '{word}' unseen in the training corpus"
                       f"{f' (T5 reads {split})' if split else ''} ({where}){hint}", source)
    if opaque:
        report.add(WARN, species, "joint-names",
                   f"{len(opaque)} joint(s) reduce to codes / digits only, no body-part word: "
                   f"{'; '.join(opaque[:6])}{' ...' if len(opaque) > 6 else ''}", source)
    # Preprocessing relabels these from the mirror geometry, so the cond is
    # right. Face-orientation detection alone still reads the name's side; it
    # prefers hip/shoulder pairs, which is why a mislabelled leaf (Leopard's
    # mane, Spider's jaws) does not reach it.
    if side_conflicts:
        report.add(INFO, species, "joint-names",
                   f"side named on the wrong half of the rig, relabelled by preprocessing from the mirror "
                   f"geometry: {'; '.join(side_conflicts[:8])}", source)

    # A two-letter side code the side detector does not read leaves the joint
    # 'center' when geometry cannot pair it. It reads a limb quadrant only next
    # to arm/leg/wing and a top/bottom corner only next to a face word.
    unread_codes = [
        raw for index, raw in enumerate(names)
        if side_labels[index] not in ("left", "right")
        and set(normalize_joint_name(raw).split()) & SIDE_CODE_TOKENS
    ]
    if unread_codes:
        report.add(WARN, species, "joint-names",
                   f"{len(unread_codes)} joint(s) spell a side code the pipeline does not read and end up "
                   f"'center': {', '.join(unread_codes[:10])}{' ...' if len(unread_codes) > 10 else ''}", source)

    # Blanked joints get a zero embedding. Fine for props, helpers and the
    # root / centre-of-gravity nodes at the top of the tree ("Bip001", "Hub");
    # deeper down, a blanked joint that parents anatomy usually means a real
    # body part was spelled only with marker words ("Point_wingRoot").
    child_count = Counter(int(p) for p in skeleton.parents if p >= 0)
    depths = np.zeros(len(names), dtype=np.int64)
    for index, parent in enumerate(skeleton.parents):
        if parent >= 0:
            depths[index] = depths[parent] + 1
    anatomical_parents = []
    for index, raw in enumerate(names):
        if texts[index].strip() or depths[index] <= ROOT_HELPER_MAX_DEPTH:
            continue
        tokens = {clean_embedding_token(t) for t in normalize_joint_name(raw).split()} - {''}
        if tokens and tokens <= ROOT_MARKER_TOKENS:
            continue
        if child_count.get(index, 0) and any(
            texts[child].strip() for child in np.flatnonzero(skeleton.parents == index)
        ):
            anatomical_parents.append(raw)
    if blanked:
        report.add(INFO, species, "joint-names",
                   f"{len(blanked)} joint(s) blanked as non-anatomical (zero name embedding): "
                   f"{', '.join(blanked[:14])}{' ...' if len(blanked) > 14 else ''}", source)
    if anatomical_parents:
        report.add(WARN, species, "joint-names",
                   "blanked joint(s) that parent named anatomy -- check they are not a real body part: "
                   f"{', '.join(anatomical_parents)}", source)

    # Counted on the sides preprocessing settles on, not the names: a side named
    # on the wrong half is reported above and would otherwise show here as a
    # phantom imbalance of two. Blanked joints (a one-handed prop socket) are
    # left out, as they are from the mirror-partner check below.
    # Partners are always one left and one right, so an imbalance implies
    # unpaired joints; when those are listed below, the count is only context.
    # It stays a WARN on its own when a pair has one half blanked.
    left = [raw for index, raw in enumerate(names) if side_labels[index] == "left" and texts[index].strip()]
    right = [raw for index, raw in enumerate(names) if side_labels[index] == "right" and texts[index].strip()]
    unpaired = [
        raw for index, raw in enumerate(names)
        if side_labels[index] in ("left", "right") and int(partners[index]) < 0 and texts[index].strip()
    ]
    if len(left) != len(right):
        level = WARN if abs(len(left) - len(right)) > SIDE_COUNT_SLACK and not unpaired else INFO
        report.add(level, species, "joint-names",
                   f"{len(left)} left vs {len(right)} right joints", source)
    if unpaired:
        report.add(WARN, species, "joint-names",
                   f"{len(unpaired)} sided joint(s) found no mirror partner: "
                   f"{', '.join(unpaired[:10])}{' ...' if len(unpaired) > 10 else ''}", source)

    collisions = [c for c in canonical if " Variant" in c]
    if collisions:
        report.add(WARN, species, "joint-names",
                   f"canonical-name collisions resolved with a Variant suffix: {', '.join(collisions[:8])}", source)

    if not is_japanese_style_naming(names):
        gated = sorted({
            token for raw in names for token in normalize_joint_name(raw).split()
            if token in JAPANESE_GATED_REPLACEMENTS and len(token) > 1
        })
        if gated:
            report.add(WARN, species, "joint-names",
                       f"romaji word(s) {gated} stay unmapped: the rig shows too little Japanese naming to "
                       "enable the gated replacements", source)

    # IK / control helpers are animated like any other joint. Blanked, they are
    # merely dead weight; with a body-part word beside the marker ("Foot_IK")
    # the marker is dropped and the helper passes for that body part. A
    # "helper" that parents named anatomy is that chain's root bone under a rig
    # author's label (Dog "TungeControler" -> the tongue chain, Slime "EyeCTRL"
    # -> eyeball and lids), so reading it as the body part is right.
    helpers = [
        index for index, raw in enumerate(names)
        if {clean_embedding_token(t) for t in normalize_joint_name(raw).split()} & HELPER_MARKER_TOKENS
    ]
    chain_roots = {
        i for i in helpers
        if any(texts[child].strip() for child in np.flatnonzero(skeleton.parents == i))
    }
    posing = [f"{names[i]} -> '{texts[i]}'" for i in helpers if texts[i].strip() and i not in chain_roots]
    if posing:
        report.add(WARN, species, "joint-names",
                   f"IK/control helper joint(s) read as anatomy: {', '.join(posing)} -- remove the helper from "
                   "the skeleton or rename it without the body-part word", source)
    rooting = [f"{names[i]} -> '{texts[i]}'" for i in helpers if texts[i].strip() and i in chain_roots]
    if rooting:
        report.add(INFO, species, "joint-names",
                   f"helper-named joint(s) parent named anatomy and are read as that chain's root: "
                   f"{', '.join(rooting)}", source)
    silent = [names[i] for i in helpers if not texts[i].strip()]
    if silent:
        report.add(INFO, species, "joint-names",
                   f"IK/control helper joint(s) stay in the skeleton and are animated as joints: "
                   f"{', '.join(silent)}", source)
    return rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _print_report(report: Report, show_info: bool) -> None:
    by_species: dict[str, list[Finding]] = defaultdict(list)
    for finding in report.findings:
        by_species[finding.species].append(finding)
    for species in sorted(by_species, key=lambda s: (s != "-", s.lower())):
        findings = [f for f in by_species[species] if show_info or f.level != INFO]
        if not findings:
            continue
        print(f"\n== {species}")
        # One line per distinct problem: a rig spelled wrong is wrong in every
        # file that carries it.
        grouped: dict[tuple[str, str, str], list[str]] = {}
        for f in sorted(findings, key=lambda f: (_LEVEL_ORDER[f.level], f.check, f.file)):
            grouped.setdefault((f.level, f.check, f.message), []).append(f.file)
        for (level, check, message), files in grouped.items():
            files = [name for name in files if name]
            if not files:
                where = ""
            elif len(files) <= 2:
                where = f" [{', '.join(files)}]"
            else:
                where = f" [{files[0]} +{len(files) - 1} more]"
            print(f"  {_LEVEL_COLOR[level]}{level:<5}{_RESET} {check}{where}: {message}")


def _print_joint_tables(report: Report) -> None:
    for species, rows in report.joint_tables.items():
        print(f"\n-- joints of {species}")
        width = max((len(r["raw_name"]) for r in rows), default=10)
        for row in rows:
            flags = f"  <{', '.join(row['flags'])}>" if row["flags"] else ""
            print(f"  {row['index']:>3} {row['raw_name']:<{width}}  {row['canonical_name']:<28} "
                  f"'{row['embedding_text']}'{flags}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", type=Path, help="raw dataset root: one sub-directory per species")
    parser.add_argument("--filter", default="",
                        help="comma-separated case-insensitive globs restricting the species checked")
    parser.add_argument("--reference-cond", type=Path, default=DEFAULT_REFERENCE_COND,
                        help="cond.npy whose joint embedding texts define the known vocabulary "
                             "(default: dataset/merged/cond.npy); 'none' disables the unseen-word check")
    parser.add_argument("--joints", action="store_true", help="print the per-joint name table of every species")
    parser.add_argument("--no-info", action="store_true", help="hide INFO findings")
    parser.add_argument("--report", type=Path, help="write the full report (findings + joint tables) as JSON")
    args = parser.parse_args()
    # Non-ASCII joint/file names must reach the report, not crash a cp1252 console.
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(errors="backslashreplace")

    root = args.dataset_dir
    if not root.is_dir():
        parser.error(f"not a directory: {root}")

    vocabulary = None
    reference_species: set[str] = set()
    promote_depths: dict[str, int] = {}
    if str(args.reference_cond).lower() != "none":
        if args.reference_cond.is_file():
            vocabulary, reference_species = _load_reference_vocabulary(args.reference_cond)
            promote_depths = _load_reference_root_promote_depths(args.reference_cond, root)
        else:
            print(f"[WARN] reference cond not found: {args.reference_cond}; unseen-word check disabled")

    patterns = [p.strip().lower() for p in re.split(r"[,;]", args.filter) if p.strip()]
    species_dirs = [
        d for d in sorted(root.iterdir(), key=lambda p: p.name.lower())
        if d.is_dir() and not d.name.startswith(".")
        and (not patterns or any(fnmatch.fnmatch(d.name.lower(), p) for p in patterns))
    ]
    if not species_dirs:
        parser.error("no species directories matched")

    t5 = T5Pieces()
    report = Report()
    _check_layout(root, species_dirs, report, reference_species)
    total_files = 0
    for species_dir in species_dirs:
        species = species_dir.name
        files = sorted(
            (p for p in species_dir.iterdir() if p.is_file() and p.suffix.lower() in MOTION_EXTENSIONS),
            key=lambda p: p.name.lower(),
        )
        total_files += len(files)
        reference, _kept = _check_filenames(species, files, report)
        facts_list = [_read_file_facts(path) for path in files]
        for facts in facts_list:
            _check_file_content(species, facts, facts.path == reference, report)
            _check_raw_joint_names(species, facts, report)
        base = _check_species_consistency(species, facts_list, reference, report)
        if base is not None:
            report.joint_tables[species] = _check_joint_names(
                species, base, vocabulary, t5, report, promote_depths.get(species.lower(), 0),
            )
        print(f"checked {species} ({len(files)} files)", file=sys.stderr)

    _print_report(report, show_info=not args.no_info)
    if args.joints:
        _print_joint_tables(report)

    levels = Counter(f.level for f in report.findings)
    blocked = sorted({f.species for f in report.findings if f.level == ERROR})
    print(f"\nSummary: {len(species_dirs)} species, {total_files} files -- "
          f"{levels.get(ERROR, 0)} error(s), {levels.get(WARN, 0)} warning(s), {levels.get(INFO, 0)} info")
    if blocked:
        print(f"Species with errors: {', '.join(blocked)}")

    if args.report:
        payload = {
            "dataset_dir": str(root),
            "reference_cond": str(args.reference_cond),
            "summary": dict(levels),
            "findings": [f.as_dict() for f in report.findings],
            "clips": report.clips,
            "joints": report.joint_tables,
        }
        args.report.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Report written to {args.report}")
    return 1 if levels.get(ERROR) else 0


if __name__ == "__main__":
    sys.exit(main())
