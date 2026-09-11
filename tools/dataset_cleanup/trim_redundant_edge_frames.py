"""
trim_redundant_edge_frames.py

Drop the redundant edge frame of every raw GLB clip under a dataset directory:
frame 0 goes when it repeats frame 1, frame N when it repeats frame N-1.  Only
those two ADJACENT pairs are tested -- the wrap pair (N, 0) is not, so a loop's
duplicated closing key is left alone.  At most one frame per end is dropped per
run; a triple key needs a second pass.

The edit is a lossless in-place surgery on the GLB binary: the animation
accessors are re-sliced (byteOffset / count) and the remaining key times are
re-anchored so the clip still starts where it did.  Mesh, skin, textures and
every other byte are untouched -- no Blender round trip.  Every modified file is
first backed up next to itself as ``<name>.glb.bak``; an existing ``.bak`` is
kept, because it is the untouched original.  ``--restore`` moves the backups
back.

"Frame" here means what the preprocessing loader sees: ``FBX.load`` samples the
action at the union of its keyframe times, so frame i is the i-th distinct key
time across the animation's samplers.  Blender's exporter writes two kinds of
time accessor -- a dense one (every frame) and a 2-key ``[first, last]`` one for
every channel whose motion is noise-level (a tiny LINEAR ramp or a STEP hold)
-- and both are edited together, otherwise the stale endpoint would still be
sampled as a frame.  The 2-key channel's values are re-interpolated at the new
end frames, so every kept frame evaluates as before.  A file whose samplers are
laid out any other way is reported and left untouched.

Redundancy is judged on the FK'd GLOBAL node positions (a root-relative pose
can repeat while the root travels) AND on the local rotations (a spin about a
bone's own axis moves no position).  Both stacks must call the pair a repeat:
the largest per-node delta across the pair at most ``--ratio`` of the clip's
own median frame step.  The boundary-step ratio over a dataset is a slope, not
a valley, so this is meant to catch a repeated key and never a merely slow
frame -- a deleted frame is unrecoverable, a missed one only keeps a hitch that
was already there.

Pure Python + numpy, no bpy.

Usage::

    # Report what would be trimmed in the default dataset (truebones zoo), touch nothing
    python Anytop/tools/dataset_cleanup/trim_redundant_edge_frames.py --dry-run

    # Trim (backs each modified file up as <name>.glb.bak first)
    python Anytop/tools/dataset_cleanup/trim_redundant_edge_frames.py

    # Another raw dataset (resolved like preprocess_and_validate.py --raw-data-dir)
    python Anytop/tools/dataset_cleanup/trim_redundant_edge_frames.py --raw-data-dir dataset/truebones/zoo_upgrade/clean

    # Only some species
    python Anytop/tools/dataset_cleanup/trim_redundant_edge_frames.py --raw-data-dir <dir> --filter Camel,Dragon*

    # Put the backups back
    python Anytop/tools/dataset_cleanup/trim_redundant_edge_frames.py --raw-data-dir <dir> --restore

Incremental preprocessing keys on the source path, not its content, so a
trimmed species has to be rebuilt with ``preprocess_and_validate.py --overwrite``.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import struct
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Put the Anytop package root on sys.path so data_loaders imports resolve to the
# Anytop copies (NOT the top-level pcvg utils).
_ANYTOP_DIR = Path(__file__).resolve().parent.parent.parent
if str(_ANYTOP_DIR) not in sys.path:
    sys.path.insert(0, str(_ANYTOP_DIR))

from data_loaders.truebones.truebones_utils.param_utils import get_raw_data_dir  # noqa: E402


# Only a repeated key is dropped, never a merely slow frame: the boundary-step
# ratio over the dataset is a slope, not a valley, so any visible-stall threshold
# would also catch real motion, and a deleted frame is unrecoverable while a
# missed one only leaves a hitch that was already there.  This sits far above
# float32 FK noise on a unit-scale rig and far below the slowest authored
# ease-out seen (a lip settling at 3e-3 of a step); a near-miss is reported.
REDUNDANT_FRAME_RATIO = 1e-4
# Kept edges closer than this many times the ratio are listed for a human eye.
NEAR_MISS_FACTOR = 10.0
# Arithmetic floor: at two frames the head pair and the tail pair are the same
# pair, and dropping both edges of a three-frame clip leaves nothing readable.
TRIM_MIN_FRAMES = 3
# Absolute floor under the ratio; also the gate that leaves a motionless clip
# alone, where every frame repeats every other and its length IS the held duration.
TRIM_MIN_MOTION = 1e-9
# Key times written by one exporter for the same frame are bit-identical; this
# only absorbs float32 noise when deciding two samplers share a frame.
TIME_TOLERANCE = 1e-7

BACKUP_SUFFIX = ".bak"
GLB_EXTENSION = ".glb"

GLB_MAGIC = 0x46546C67
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942

_COMPONENT_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_TYPE_COMPONENTS = {
    "SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16,
}
_NODE_TRS_PATHS = ("translation", "rotation", "scale")


class UnsupportedLayout(Exception):
    """The file is valid but not laid out in a way this tool can edit exactly."""


# ── GLB container ─────────────────────────────────────────────────────────

@dataclass
class Glb:
    version: int
    json: dict
    bin: bytearray
    # Chunks other than the first JSON and first BIN, carried through verbatim.
    extra_chunks: list[tuple[int, bytes]] = field(default_factory=list)


def read_glb(path: str | os.PathLike) -> Glb:
    data = Path(path).read_bytes()
    if len(data) < 12:
        raise ValueError(f"{path}: too short to be a GLB")
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC:
        raise ValueError(f"{path}: not a GLB (magic {magic:#x})")
    if length > len(data):
        raise ValueError(f"{path}: header declares {length} bytes, file holds {len(data)}")

    json_chunk = None
    bin_chunk = None
    extra: list[tuple[int, bytes]] = []
    offset = 12
    while offset + 8 <= length:
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        payload = data[offset + 8: offset + 8 + chunk_length]
        if len(payload) != chunk_length:
            raise ValueError(f"{path}: chunk at byte {offset} is truncated")
        if chunk_type == CHUNK_JSON and json_chunk is None:
            json_chunk = payload
        elif chunk_type == CHUNK_BIN and bin_chunk is None:
            bin_chunk = payload
        else:
            extra.append((chunk_type, bytes(payload)))
        offset += 8 + chunk_length
    if json_chunk is None:
        raise ValueError(f"{path}: no JSON chunk")
    return Glb(version, json.loads(json_chunk), bytearray(bin_chunk or b""), extra)


def write_glb(path: str | os.PathLike, glb: Glb) -> None:
    json_bytes = json.dumps(glb.json, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * (-len(json_bytes) % 4)
    chunks = [(CHUNK_JSON, json_bytes)]
    if glb.bin:
        chunks.append((CHUNK_BIN, bytes(glb.bin) + b"\0" * (-len(glb.bin) % 4)))
    chunks.extend(glb.extra_chunks)
    body = b"".join(struct.pack("<II", len(payload), chunk_type) + payload for chunk_type, payload in chunks)
    Path(path).write_bytes(struct.pack("<III", GLB_MAGIC, glb.version, 12 + len(body)) + body)


# ── accessors ─────────────────────────────────────────────────────────────

def _accessor_stride(glb: Glb, index: int) -> tuple[np.dtype, int, int]:
    """Return ``(component dtype, components per element, byte stride)``."""
    accessor = glb.json["accessors"][index]
    dtype = np.dtype(_COMPONENT_DTYPES[accessor["componentType"]])
    components = _TYPE_COMPONENTS[accessor["type"]]
    view = glb.json["bufferViews"][accessor["bufferView"]]
    return dtype, components, int(view.get("byteStride", dtype.itemsize * components))


def _accessor_view(glb: Glb, index: int) -> np.ndarray:
    """A writable ``(count, components)`` view of an accessor's RAW components in the BIN chunk."""
    accessor = glb.json["accessors"][index]
    if "sparse" in accessor or "bufferView" not in accessor:
        raise UnsupportedLayout(f"accessor {index} is sparse or has no bufferView")
    view = glb.json["bufferViews"][accessor["bufferView"]]
    if view.get("buffer", 0) != 0 or "uri" in glb.json["buffers"][0]:
        raise UnsupportedLayout(f"accessor {index} does not live in the GLB BIN chunk")
    dtype, components, stride = _accessor_stride(glb, index)
    count = int(accessor["count"])
    view_start = int(view.get("byteOffset", 0))
    base = view_start + int(accessor.get("byteOffset", 0))
    end = base + (count - 1) * stride + dtype.itemsize * components if count else base
    if end > view_start + int(view["byteLength"]) or end > len(glb.bin):
        raise ValueError(f"accessor {index} overruns its bufferView")
    return np.ndarray((count, components), dtype=dtype, buffer=glb.bin, offset=base,
                      strides=(stride, dtype.itemsize))


def _accessor_values(glb: Glb, index: int) -> np.ndarray:
    """Accessor elements as float64, integer components de-normalized per the glTF spec."""
    raw = _accessor_view(glb, index)
    values = raw.astype(np.float64)
    if raw.dtype == np.int8:
        values = np.maximum(values / 127.0, -1.0)
    elif raw.dtype == np.uint8:
        values = values / 255.0
    elif raw.dtype == np.int16:
        values = np.maximum(values / 32767.0, -1.0)
    elif raw.dtype == np.uint16:
        values = values / 65535.0
    return values


# ── animation layout ──────────────────────────────────────────────────────

@dataclass
class SamplerLayout:
    channel: int
    input: int
    output: int
    interpolation: str
    per_key: int      # output elements per key: 1, 3 for CUBICSPLINE, x targets for weights
    kind: str         # 'dense' (a key on every frame) | 'endpoints' (2-key ramp/hold keyed at [first, last])


@dataclass
class AnimationLayout:
    index: int
    times: np.ndarray   # (frames,) float64 -- the frame timeline the loader samples
    samplers: list[SamplerLayout]


def _merge_times(stacks: list[np.ndarray]) -> np.ndarray:
    times = np.sort(np.concatenate(stacks))
    keep = np.ones(times.shape[0], dtype=bool)
    keep[1:] = np.diff(times) > TIME_TOLERANCE
    return times[keep]


def analyze_animation(glb: Glb, anim_index: int) -> AnimationLayout:
    """Classify every sampler of one animation against the frame timeline.

    Raises :class:`UnsupportedLayout` for anything the trim could not apply
    exactly: keys on a subset of frames, a cubic or non-float endpoint-only
    channel, sparse accessors, pointer targets.
    """
    animation = glb.json["animations"][anim_index]
    samplers = animation["samplers"]
    channels = animation["channels"]
    if not channels:
        raise UnsupportedLayout("animation has no channels")

    # Only samplers a channel references make frames: those are the fcurves the
    # importer creates and the loader unions the key times of.
    input_times: dict[int, np.ndarray] = {}
    for sampler_index in sorted({channel["sampler"] for channel in channels}):
        sampler = samplers[sampler_index]
        accessor = glb.json["accessors"][sampler["input"]]
        if accessor["componentType"] != 5126 or accessor["type"] != "SCALAR":
            raise UnsupportedLayout(f"sampler {sampler_index}: input accessor is not float SCALAR")
        times = _accessor_values(glb, sampler["input"])[:, 0]
        if times.shape[0] == 0 or not np.all(np.diff(times) > 0):
            raise UnsupportedLayout(f"sampler {sampler_index}: key times are not strictly increasing")
        input_times[sampler["input"]] = times

    timeline = _merge_times(list(input_times.values()))
    layouts: list[SamplerLayout] = []
    for channel_index, channel in enumerate(channels):
        target = channel.get("target", {})
        if "node" not in target or target.get("path") not in _NODE_TRS_PATHS + ("weights",):
            raise UnsupportedLayout(f"channel {channel_index}: target is not a node TRS/weights path")
        sampler_index = channel["sampler"]
        sampler = samplers[sampler_index]
        interpolation = sampler.get("interpolation", "LINEAR")
        times = input_times[sampler["input"]]
        output_count = int(glb.json["accessors"][sampler["output"]]["count"])
        if output_count % times.shape[0]:
            raise UnsupportedLayout(f"sampler {sampler_index}: output count {output_count} is not a multiple of {times.shape[0]} keys")
        per_key = output_count // times.shape[0]
        if interpolation == "CUBICSPLINE" and per_key % 3:
            raise UnsupportedLayout(f"sampler {sampler_index}: CUBICSPLINE output is not in triples")

        if times.shape[0] == timeline.shape[0] and np.all(np.abs(times - timeline) <= TIME_TOLERANCE):
            kind = "dense"
        elif (times.shape[0] == 2
              and abs(times[0] - timeline[0]) <= TIME_TOLERANCE
              and abs(times[1] - timeline[-1]) <= TIME_TOLERANCE):
            # Blender's exporter collapses a channel whose motion is noise-level
            # into its first and last key: a LINEAR one is a tiny ramp, a STEP
            # one holds the first value until the last frame.  Its two values
            # get re-interpolated on trim, so it must be float and not cubic.
            if interpolation not in ("LINEAR", "STEP"):
                raise UnsupportedLayout(f"sampler {sampler_index}: {interpolation} endpoint-only channel")
            if glb.json["accessors"][sampler["output"]]["componentType"] != 5126:
                raise UnsupportedLayout(f"sampler {sampler_index}: endpoint-only channel output is not float")
            kind = "endpoints"
        else:
            raise UnsupportedLayout(
                f"sampler {sampler_index}: {times.shape[0]} keys are neither every frame "
                f"({timeline.shape[0]}) nor the two endpoints"
            )
        layouts.append(SamplerLayout(channel_index, sampler["input"], sampler["output"], interpolation, per_key, kind))
    return AnimationLayout(anim_index, timeline, layouts)


# ── pose evaluation ───────────────────────────────────────────────────────

def _quaternion_to_matrix(quaternions: np.ndarray) -> np.ndarray:
    """(..., 4) glTF ``(x, y, z, w)`` quaternions -> (..., 3, 3) rotation matrices."""
    q = quaternions / np.linalg.norm(quaternions, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    matrix = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1 - 2 * (yy + zz)
    matrix[..., 0, 1] = 2 * (xy - wz)
    matrix[..., 0, 2] = 2 * (xz + wy)
    matrix[..., 1, 0] = 2 * (xy + wz)
    matrix[..., 1, 1] = 1 - 2 * (xx + zz)
    matrix[..., 1, 2] = 2 * (yz - wx)
    matrix[..., 2, 0] = 2 * (xz - wy)
    matrix[..., 2, 1] = 2 * (yz + wx)
    matrix[..., 2, 2] = 1 - 2 * (xx + yy)
    return matrix


def _node_parents(nodes: list[dict]) -> np.ndarray:
    parents = np.full(len(nodes), -1, dtype=np.int64)
    for node_index, node in enumerate(nodes):
        for child in node.get("children", []):
            parents[child] = node_index
    return parents


def _topological_order(parents: np.ndarray) -> list[int]:
    children: dict[int, list[int]] = {}
    for node_index, parent in enumerate(parents.tolist()):
        children.setdefault(parent, []).append(node_index)
    order: list[int] = []
    frontier = list(children.get(-1, []))
    while frontier:
        node_index = frontier.pop(0)
        order.append(node_index)
        frontier.extend(children.get(node_index, []))
    if len(order) != parents.shape[0]:
        raise UnsupportedLayout("node graph is not a forest")
    return order


def _endpoint_values(first: np.ndarray, last: np.ndarray, interpolation: str, times: np.ndarray) -> np.ndarray:
    """Sample a 2-key ``[first, last]`` channel at every frame time.

    Component-wise lerp for LINEAR, which is how Blender evaluates the imported
    fcurves (a quaternion channel is four independent curves); STEP holds the
    first value and switches on the last frame.
    """
    first = np.asarray(first, dtype=np.float64)
    last = np.asarray(last, dtype=np.float64)
    if interpolation == "STEP":
        values = np.repeat(first[None], times.shape[0], axis=0)
        values[-1] = last
        return values
    fraction = (times - times[0]) / (times[-1] - times[0])
    return first[None] + (last - first)[None] * fraction.reshape((-1,) + (1,) * first.ndim)


def evaluate_animation(glb: Glb, layout: AnimationLayout) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(global positions (T, J, 3), local rotations as 6D (T, J, 6))`` over every node.

    Sampled at the frame timeline, so a dense sampler contributes its key values
    directly and an endpoint-only one its ramp or hold at those same times.
    """
    nodes = glb.json.get("nodes", [])
    animation = glb.json["animations"][layout.index]
    n_frames, n_nodes = layout.times.shape[0], len(nodes)

    translation = np.zeros((n_frames, n_nodes, 3))
    rotation = np.zeros((n_frames, n_nodes, 4))
    rotation[..., 3] = 1.0
    scale = np.ones((n_frames, n_nodes, 3))
    for node_index, node in enumerate(nodes):
        translation[:, node_index] = node.get("translation", (0.0, 0.0, 0.0))
        rotation[:, node_index] = node.get("rotation", (0.0, 0.0, 0.0, 1.0))
        scale[:, node_index] = node.get("scale", (1.0, 1.0, 1.0))

    animated = np.zeros(n_nodes, dtype=bool)
    for sampler in layout.samplers:
        channel = animation["channels"][sampler.channel]
        path = channel["target"]["path"]
        if path == "weights":
            continue
        values = _accessor_values(glb, sampler.output).reshape(-1, sampler.per_key, _TYPE_COMPONENTS[glb.json["accessors"][sampler.output]["type"]])
        # A CUBICSPLINE key is (in-tangent, value, out-tangent); the value is the middle one.
        values = values[:, 1 if sampler.interpolation == "CUBICSPLINE" else 0, :]
        if sampler.kind == "endpoints":
            values = _endpoint_values(values[0], values[1], sampler.interpolation, layout.times)
        node_index = channel["target"]["node"]
        {"translation": translation, "rotation": rotation, "scale": scale}[path][:, node_index] = values
        animated[node_index] = True

    local = np.tile(np.eye(4), (n_frames, n_nodes, 1, 1))
    rotation_matrices = _quaternion_to_matrix(rotation)
    local[..., :3, :3] = rotation_matrices * scale[..., None, :]
    local[..., :3, 3] = translation
    for node_index, node in enumerate(nodes):
        if "matrix" in node and not animated[node_index]:
            local[:, node_index] = np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T

    parents = _node_parents(nodes)
    world = np.empty_like(local)
    for node_index in _topological_order(parents):
        parent = parents[node_index]
        world[:, node_index] = local[:, node_index] if parent < 0 else world[:, parent] @ local[:, node_index]

    positions = world[..., :3, 3]
    rotation_6d = rotation_matrices[..., :, :2].reshape(n_frames, n_nodes, 6)
    return positions, rotation_6d


# ── the verdict ───────────────────────────────────────────────────────────

def _pair_gaps(frames: np.ndarray):
    """Return ``(median frame step, gap(a, b))`` for a (T, J, D) frame stack.

    The gap is the LARGEST per-node delta magnitude: a frame on which any node
    moves is not a repeat.  A percentile over nodes would be dragged to zero by
    the static ones (a mesh node, a nub, a parked prop -- one Truebones dog
    carries 120 static mesh nodes next to 50 moving joints) and call a frame
    redundant while half the skeleton moves.  The reference is the median over
    frames of the same statistic, so the two are commensurable.
    """
    frames = np.asarray(frames, dtype=np.float64)
    frames = frames.reshape(frames.shape[0], frames.shape[1], -1)
    step_gaps = np.linalg.norm(np.diff(frames, axis=0), axis=-1).max(axis=1)
    reference = float(np.median(step_gaps)) if step_gaps.size else 0.0

    def gap(a: int, b: int) -> float:
        return float(np.linalg.norm(frames[b] - frames[a], axis=-1).max())

    return reference, gap


@dataclass
class EdgeVerdict:
    drop_first: bool = False
    drop_last: bool = False
    reason: str = ""                          # why nothing is dropped, when structural
    ratios: dict[str, float] = field(default_factory=dict)  # 'head_pos', 'head_rot', 'tail_pos', 'tail_rot'


def find_redundant_edge_frames(global_positions: np.ndarray, rotations: np.ndarray,
                               ratio: float = REDUNDANT_FRAME_RATIO) -> EdgeVerdict:
    """Decide the two adjacent edge pairs of a clip.

    Frame 0 is redundant when it repeats frame 1; frame N when it repeats frame
    N-1.  Positions are the FK'd GLOBAL ones on purpose: a root-relative pose
    can be identical across two frames the root actually travelled or turned
    between.  Rotations cover motion a position stack cannot see (a spin about
    a bone's own axis, a zero-length bone); both stacks must call the pair
    redundant.  A stack that never changes reports a zero gap for every pair
    and so abstains from the AND rather than driving it.
    """
    positions = np.asarray(global_positions, dtype=np.float64)
    frame_count = int(positions.shape[0])
    verdict = EdgeVerdict()
    if frame_count < TRIM_MIN_FRAMES + 1:
        verdict.reason = f"{frame_count} frames, fewer than {TRIM_MIN_FRAMES + 1}"
        return verdict
    if rotations.shape[0] != frame_count:
        raise ValueError(f"rotations frame count {rotations.shape[0]} != positions frame count {frame_count}")

    stacks = {"pos": _pair_gaps(positions), "rot": _pair_gaps(rotations)}
    if stacks["pos"][0] <= TRIM_MIN_MOTION:
        verdict.reason = "motionless clip"
        return verdict

    last = frame_count - 1
    for edge, (a, b) in (("head", (0, 1)), ("tail", (last - 1, last))):
        for name, (reference, gap) in stacks.items():
            distance = gap(a, b)
            verdict.ratios[f"{edge}_{name}"] = (
                distance / reference if reference > TRIM_MIN_MOTION
                else (0.0 if distance <= TRIM_MIN_MOTION else float("inf"))
            )

    def is_redundant(a: int, b: int) -> bool:
        return all(
            gap(a, b) <= max(ratio * reference, TRIM_MIN_MOTION)
            for reference, gap in stacks.values()
        )

    verdict.drop_first = is_redundant(0, 1)
    verdict.drop_last = is_redundant(last - 1, last)
    if frame_count - int(verdict.drop_first) - int(verdict.drop_last) < TRIM_MIN_FRAMES:
        verdict.drop_first = verdict.drop_last = False
        verdict.reason = f"trim would leave fewer than {TRIM_MIN_FRAMES} frames"
    return verdict


# ── the edit ──────────────────────────────────────────────────────────────

def apply_trim(glb: Glb, layout: AnimationLayout, drop_first: bool, drop_last: bool,
               edited: dict[int, tuple] | None = None) -> np.ndarray:
    """Re-slice one animation's accessors in place; return the new frame timeline.

    ``edited`` remembers every accessor already touched (shared across the
    animations of one file), so an accessor two samplers share is sliced once and
    one two animations disagree about is refused.
    """
    edited = {} if edited is None else edited
    times = layout.times
    start = 1 if drop_first else 0
    stop = times.shape[0] - 1 if drop_last else times.shape[0]
    # Re-anchor so the clip still starts where it did: the importer maps
    # frame = time * fps with no offset of its own.
    new_times = (times[start:stop] - (times[start] - times[0])).astype(np.float32)
    if not (drop_first or drop_last):
        return new_times.astype(np.float64)

    accessors = glb.json["accessors"]

    def reseat_accessor(index: int, signature: tuple) -> bool:
        """Claim an accessor for one edit; False when the same edit was already applied to it."""
        if index in edited:
            if edited[index] != signature:
                raise UnsupportedLayout(f"accessor {index} is shared by samplers that need different edits")
            return False
        edited[index] = signature
        return True

    def slice_accessor(index: int, per_key: int, signature: tuple) -> bool:
        """Shrink an accessor to keys ``[start, stop)``; False when it was already done."""
        if not reseat_accessor(index, signature):
            return False
        accessor = accessors[index]
        _dtype, _components, stride = _accessor_stride(glb, index)
        if drop_first:
            accessor["byteOffset"] = int(accessor.get("byteOffset", 0)) + per_key * stride
        accessor["count"] = int(accessor["count"]) - per_key * (int(drop_first) + int(drop_last))
        return True

    for sampler in layout.samplers:
        if sampler.kind == "dense":
            if slice_accessor(sampler.input, 1, ("dense-input", start, stop)):
                view = _accessor_view(glb, sampler.input)
                view[:, 0] = new_times
                accessors[sampler.input]["min"] = [float(view[0, 0])]
                accessors[sampler.input]["max"] = [float(view[-1, 0])]
            if slice_accessor(sampler.output, sampler.per_key, ("dense-output", sampler.per_key, start, stop)):
                accessor = accessors[sampler.output]
                if "min" in accessor or "max" in accessor:
                    view = _accessor_view(glb, sampler.output)
                    accessor["min"] = [float(v) for v in view.min(axis=0)]
                    accessor["max"] = [float(v) for v in view.max(axis=0)]
        else:
            if reseat_accessor(sampler.input, ("endpoints-input", start, stop)):
                view = _accessor_view(glb, sampler.input)
                # Bit-identical to the dense accessor's new last time: same float32 value.
                view[1, 0] = new_times[-1]
                accessors[sampler.input]["min"] = [float(view[0, 0])]
                accessors[sampler.input]["max"] = [float(view[1, 0])]
            if reseat_accessor(sampler.output, ("endpoints-output", start, stop)):
                view = _accessor_view(glb, sampler.output)
                first, last = view[:sampler.per_key].astype(np.float64), view[sampler.per_key:].astype(np.float64)
                # The channel's values at the frames that become the new ends,
                # so the ramp (or hold) through every kept frame is unchanged.
                sampled = _endpoint_values(first, last, sampler.interpolation, times)
                view[:sampler.per_key] = sampled[start]
                view[sampler.per_key:] = sampled[stop - 1]
                accessor = accessors[sampler.output]
                if "min" in accessor or "max" in accessor:
                    accessor["min"] = [float(v) for v in view.min(axis=0)]
                    accessor["max"] = [float(v) for v in view.max(axis=0)]
    return new_times.astype(np.float64)


# ── per-file driver ───────────────────────────────────────────────────────

@dataclass
class AnimationResult:
    dropped: tuple[str, ...] = ()        # subset of ('first', 'last')
    reason: str = ""                     # structural reason nothing was dropped
    ratios: dict[str, float] = field(default_factory=dict)


@dataclass
class FileResult:
    path: str
    status: str                          # 'trimmed' | 'clean' | 'skipped' | 'error'
    detail: str = ""
    frames_before: int = 0
    frames_after: int = 0
    animations: list[AnimationResult] = field(default_factory=list)

    @property
    def dropped(self) -> tuple[str, ...]:
        return tuple(edge for animation in self.animations for edge in animation.dropped)


def _animation_accessors_are_private(glb: Glb) -> bool:
    """True when no animation accessor is also referenced by a mesh, skin or anything else."""
    animation_accessors = {
        sampler[key]
        for animation in glb.json.get("animations", [])
        for sampler in animation["samplers"]
        for key in ("input", "output")
    }
    for mesh in glb.json.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            referenced = set(primitive.get("attributes", {}).values())
            if "indices" in primitive:
                referenced.add(primitive["indices"])
            for target in primitive.get("targets", []):
                referenced.update(target.values())
            if referenced & animation_accessors:
                return False
    for skin in glb.json.get("skins", []):
        if skin.get("inverseBindMatrices") in animation_accessors:
            return False
    return True


def _format_ratios(ratios: dict[str, float]) -> str:
    def fmt(key: str) -> str:
        value = ratios.get(key)
        return "n/a" if value is None else f"{value:.1e}"
    return (f"head pos={fmt('head_pos')} rot={fmt('head_rot')} | "
            f"tail pos={fmt('tail_pos')} rot={fmt('tail_rot')}")


def describe_animations(result: FileResult, verb: str) -> str:
    """One clause per animation; the ``anim[i]`` prefix only appears for a multi-animation file."""
    clauses = []
    for index, animation in enumerate(result.animations):
        prefix = f"anim[{index}] " if len(result.animations) > 1 else ""
        what = f"{verb} {' + '.join(animation.dropped)} frame" if animation.dropped else (animation.reason or "nothing to drop")
        clauses.append(f"{prefix}{what} ({_format_ratios(animation.ratios)})")
    return "; ".join(clauses)


def process_file(path: str, ratio: float = REDUNDANT_FRAME_RATIO, dry_run: bool = False) -> FileResult:
    try:
        glb = read_glb(path)
        animations = glb.json.get("animations", [])
        if not animations:
            return FileResult(path, "skipped", "no animation")
        if not _animation_accessors_are_private(glb):
            return FileResult(path, "skipped", "an animation accessor is shared with mesh/skin data")

        layouts = [analyze_animation(glb, index) for index in range(len(animations))]
        verdicts = []
        for layout in layouts:
            positions, rotations = evaluate_animation(glb, layout)
            verdicts.append(find_redundant_edge_frames(positions, rotations, ratio=ratio))

        frames_before = sum(layout.times.shape[0] for layout in layouts)
        animation_results = [
            AnimationResult(
                tuple(edge for edge, taken in (("first", verdict.drop_first), ("last", verdict.drop_last)) if taken),
                verdict.reason,
                verdict.ratios,
            )
            for verdict in verdicts
        ]
        if not any(animation.dropped for animation in animation_results):
            return FileResult(path, "clean", "", frames_before, frames_before, animation_results)

        edited: dict[int, tuple] = {}
        frames_after = 0
        for layout, verdict in zip(layouts, verdicts):
            new_times = apply_trim(glb, layout, verdict.drop_first, verdict.drop_last, edited)
            frames_after += new_times.shape[0]
            # The edited animation must read back as the same layout on the shorter timeline.
            check = analyze_animation(glb, layout.index)
            if check.times.shape[0] != new_times.shape[0] or not np.all(np.abs(check.times - new_times) <= TIME_TOLERANCE):
                raise RuntimeError(f"edited timeline does not read back: {check.times.shape[0]} frames vs {new_times.shape[0]}")

        result = FileResult(path, "trimmed", "", frames_before, frames_after, animation_results)
        if dry_run:
            return result

        backup_path = path + BACKUP_SUFFIX
        if not os.path.exists(backup_path):
            shutil.copy2(path, backup_path)
        temp_path = path + ".tmp"
        write_glb(temp_path, glb)
        reread = read_glb(temp_path)
        reread_frames = sum(analyze_animation(reread, index).times.shape[0] for index in range(len(animations)))
        if reread_frames != frames_after:
            os.remove(temp_path)
            raise RuntimeError(f"written file reads back with {reread_frames} frames, expected {frames_after}")
        os.replace(temp_path, path)
        return result
    except UnsupportedLayout as err:
        return FileResult(path, "skipped", f"unsupported layout: {err}")
    except Exception as err:  # noqa: BLE001 -- one bad file must not stop the sweep
        return FileResult(path, "error", f"{type(err).__name__}: {err}")


# ── directory walk / CLI ──────────────────────────────────────────────────

def _species_of(path: str, raw_data_dir: str) -> str:
    relative = os.path.relpath(path, raw_data_dir)
    head = relative.split(os.sep)[0]
    return head if head != os.path.basename(path) else os.path.basename(os.path.normpath(raw_data_dir))


def _matches_filter(species: str, patterns: list[str]) -> bool:
    if not patterns:
        return True
    species = species.lower()
    return any(fnmatch.fnmatchcase(species, pattern.lower()) for pattern in patterns)


def list_glb_files(raw_data_dir: str, patterns: list[str], suffix: str = GLB_EXTENSION) -> list[str]:
    """Sorted ``*.glb`` (or ``*.glb.bak``) paths under *raw_data_dir* whose species matches."""
    found: list[str] = []
    for root, dirs, files in os.walk(raw_data_dir):
        dirs.sort()
        for name in sorted(files):
            if not name.lower().endswith(suffix):
                continue
            path = os.path.join(root, name)
            if _matches_filter(_species_of(path, raw_data_dir), patterns):
                found.append(path)
    return found


def restore_backups(raw_data_dir: str, patterns: list[str]) -> int:
    backups = list_glb_files(raw_data_dir, patterns, suffix=GLB_EXTENSION + BACKUP_SUFFIX)
    for backup_path in backups:
        os.replace(backup_path, backup_path[: -len(BACKUP_SUFFIX)])
        print(f"[restored] {backup_path[: -len(BACKUP_SUFFIX)]}")
    return len(backups)


def _parse_patterns(text: str) -> list[str]:
    return [token.strip() for token in text.replace(";", ",").split(",") if token.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Drop the redundant edge frame (0 == 1, N-1 == N) of every raw GLB clip, in place, with a .bak backup.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--raw-data-dir", default="", type=str,
                        help="Raw dataset root holding per-species subfolders of GLB clips (walked recursively). "
                             "Defaults to the truebones zoo raw dataset (Truebone_Z-OO), like preprocess_and_validate.py.")
    parser.add_argument("--filter", default="", type=str,
                        help="Comma/semicolon-separated case-insensitive glob(s) on the species folder name (e.g. 'Camel,Dragon*'). Default: every species.")
    parser.add_argument("--ratio", default=REDUNDANT_FRAME_RATIO, type=float,
                        help=f"An edge pair is redundant when its step is at most this fraction of the clip's median frame step. Default {REDUNDANT_FRAME_RATIO:g}.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be trimmed and write nothing.")
    parser.add_argument("--restore", action="store_true",
                        help="Move every <name>.glb.bak back over its <name>.glb and exit.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Also print every clean file with its measured edge ratios.")
    parser.add_argument("--workers", "-j", default=8, type=int,
                        help="Parallel worker processes. Default 8.")
    args = parser.parse_args(argv)

    try:
        raw_data_dir = get_raw_data_dir(args.raw_data_dir or None)
    except FileNotFoundError as err:
        parser.error(str(err))
    patterns = _parse_patterns(args.filter)

    if args.restore:
        count = restore_backups(raw_data_dir, patterns)
        print(f"restored {count} backup(s) under {raw_data_dir}")
        return 0

    files = list_glb_files(raw_data_dir, patterns)
    if not files:
        print(f"no .glb files under {raw_data_dir}" + (f" matching {args.filter!r}" if patterns else ""))
        return 0
    mode = "dry-run" if args.dry_run else "trim"
    print(f"[{mode}] {len(files)} GLB file(s) under {raw_data_dir}, ratio {args.ratio:g}")

    results: list[FileResult] = []
    if args.workers > 1 and len(files) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_file, path, args.ratio, args.dry_run): path for path in files}
            for future in as_completed(futures):
                results.append(future.result())
    else:
        results = [process_file(path, args.ratio, args.dry_run) for path in files]
    results.sort(key=lambda result: result.path)

    verb = "would drop" if args.dry_run else "dropped"
    for result in results:
        relative = os.path.relpath(result.path, raw_data_dir)
        if result.status == "trimmed":
            print(f"[{result.status}] {relative}: {result.frames_before} -> {result.frames_after} frames, "
                  f"{describe_animations(result, verb)}")
        elif result.status == "clean" and args.verbose:
            print(f"[{result.status}] {relative}: {result.frames_before} frames, {describe_animations(result, verb)}")
        elif result.status in ("skipped", "error"):
            print(f"[{result.status}] {relative}: {result.detail}")

    near_misses = []
    for result in results:
        if result.status not in ("trimmed", "clean"):
            continue
        for animation in result.animations:
            for edge in ("head", "tail"):
                if ("first" if edge == "head" else "last") in animation.dropped:
                    continue
                value = max(animation.ratios.get(f"{edge}_pos", 0.0), animation.ratios.get(f"{edge}_rot", 0.0))
                if args.ratio < value <= NEAR_MISS_FACTOR * args.ratio:
                    near_misses.append((os.path.relpath(result.path, raw_data_dir), edge, value))
    if near_misses:
        print()
        print(f"{len(near_misses)} kept edge(s) within {NEAR_MISS_FACTOR:g}x of --ratio {args.ratio:g} "
              f"(re-run that species with --filter and a larger --ratio if it is a repeat):")
        for relative_path, edge, value in near_misses:
            print(f"[near-miss] {relative_path}: {edge} {value:.1e}")

    counts = {status: sum(1 for r in results if r.status == status) for status in ("trimmed", "clean", "skipped", "error")}
    frames_dropped = sum(r.frames_before - r.frames_after for r in results if r.status == "trimmed")
    print(
        f"\n[{mode}] {counts['trimmed']} trimmed ({frames_dropped} frame(s) {verb}), "
        f"{counts['clean']} clean, {counts['skipped']} skipped, {counts['error']} error(s)"
    )
    if counts["trimmed"] and not args.dry_run:
        print("Backups sit next to each modified file as <name>.glb.bak; --restore puts them back.\n"
              "Incremental preprocessing keys on the source path, so rebuild the trimmed species with --overwrite.")
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
