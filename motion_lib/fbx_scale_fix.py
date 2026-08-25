# -*- coding: utf-8 -*-
"""Repair FBX rigs whose bone nodes carry a non-unit ``Lcl Scaling``.

A Blender armature cannot store a rest *scale* per bone, so a bone node that
carries one has to be folded into the geometry of the rest pose.  Blender's FBX
importer only does half of that job:

* rest positions come from the FBX bind pose, which already has the scale baked
  into its world matrices, so they land correctly;
* bone *lengths* come from the limb ``Size`` property, which is read verbatim;
* the animated ``Lcl Translation`` curves are replayed verbatim as pose-bone
  ``location`` channels, so every animated translation ends up ``1 / scale``
  too large;
* a bone that is missing from the bind pose gets its whole rest offset from the
  raw local transform, so it lands ``1 / parent scale`` too far from its parent.

Unity asset packs hit this routinely: the exporter parks a ``0.01`` scale on the
root bone so the rest of the rig can stay authored in centimetres.  The rig is
self-consistent and renders correctly in Unity, but Blender blows the clip up by
100x -- the character's limbs fly hundreds of units away from the mesh.

:func:`repair_scaled_bone_import` reads the offending numbers straight out of
the FBX and repairs the imported armature.  It is a no-op for the
overwhelmingly common case where every bone node is unit-scaled.
"""

from __future__ import annotations

import bisect
import struct
import zlib
from typing import Any

# Array payloads are zlib-compressed; only decode the two that carry key data.
_ARRAY_NODES = frozenset({"KeyTime", "KeyValueFloat"})

_FBX_MAGIC = b"Kaydara FBX Binary  \x00"
_FBX_TIME_UNITS_PER_SECOND = 46186158000.0

# ``InheritType`` 2 (eInheritRrs) is the one mode where a child does *not* pick
# up its parent's scale.
_INHERIT_NO_SCALE = 2

# Scales this close to 1 are float noise in the exporter, not authored intent.
_SCALE_EPS = 1e-4


# ── minimal binary FBX reader ────────────────────────────────────────────────


def _read_props(data: bytes, off: int, end: int, decode_arrays: bool) -> list:
    props: list[Any] = []
    while off < end:
        kind = data[off : off + 1].decode("ascii", "replace")
        off += 1
        if kind == "Y":
            props.append(struct.unpack_from("<h", data, off)[0]); off += 2
        elif kind == "C":
            props.append(bool(data[off])); off += 1
        elif kind == "I":
            props.append(struct.unpack_from("<i", data, off)[0]); off += 4
        elif kind == "F":
            props.append(struct.unpack_from("<f", data, off)[0]); off += 4
        elif kind == "D":
            props.append(struct.unpack_from("<d", data, off)[0]); off += 8
        elif kind == "L":
            props.append(struct.unpack_from("<q", data, off)[0]); off += 8
        elif kind in "fdlib":
            count, encoding, length = struct.unpack_from("<III", data, off)
            off += 12
            if decode_arrays:
                raw = data[off : off + length]
                if encoding:
                    raw = zlib.decompress(raw)
                fmt = {"f": "f", "d": "d", "l": "q", "i": "i", "b": "b"}[kind]
                item = struct.calcsize("<" + fmt)
                props.append(
                    list(struct.unpack_from("<%d%s" % (count, fmt), raw, 0))
                    if len(raw) >= count * item
                    else []
                )
            else:
                props.append(None)
            off += length
        elif kind in "SR":
            length = struct.unpack_from("<I", data, off)[0]
            off += 4
            blob = data[off : off + length]
            off += length
            props.append(blob.decode("utf-8", "replace") if kind == "S" else None)
        else:
            raise ValueError("unknown FBX property type %r" % kind)
    return props


def _read_node(data: bytes, off: int, is64: bool, descend) -> tuple:
    """Read one node; ``descend(name)`` decides whether to walk its children."""
    if is64:
        end, _nprops, prop_len = struct.unpack_from("<QQQ", data, off)
        off += 24
        sentinel = 25
    else:
        end, _nprops, prop_len = struct.unpack_from("<III", data, off)
        off += 12
        sentinel = 13
    name_len = data[off]
    off += 1
    name = data[off : off + name_len].decode("utf-8", "replace")
    off += name_len
    props = _read_props(data, off, off + prop_len, name in _ARRAY_NODES)
    off += prop_len
    children = []
    if descend(name):
        while off < end:
            if data[off : off + sentinel] == b"\x00" * sentinel:
                break
            child, off = _read_node(data, off, is64, descend)
            children.append(child)
    return (name, props, children), end


def _parse_fbx(path: str, with_curves: bool):
    """Return the top-level nodes we care about, or ``None`` if unreadable.

    Unless ``with_curves`` is set, ``AnimationCurve`` bodies are skipped by
    their end offset -- decompressing every key array is by far the most
    expensive part of the scan, and only the rare bind-pose repair needs it.
    """
    with open(path, "rb") as handle:
        data = handle.read()
    if not data.startswith(_FBX_MAGIC):
        return None  # ASCII FBX or FBX 6.x -- nothing we can repair
    version = struct.unpack_from("<I", data, 23)[0]
    is64 = version >= 7500
    sentinel = 25 if is64 else 13

    # Subtrees with nothing we need; skipped wholesale by their end offset.
    skip = {"Geometry", "Video", "Texture", "Deformer"}
    if not with_curves:
        skip.update(("AnimationCurve", "AnimationCurveNode"))

    def descend(name: str) -> bool:
        return name not in skip

    roots = {}
    off = 27
    while off < len(data) - sentinel:
        if data[off : off + sentinel] == b"\x00" * sentinel:
            break
        node, off = _read_node(data, off, is64, descend)
        if node[0] in ("Objects", "Connections"):
            roots[node[0]] = node
        if node[0] == "":
            break
    return roots


def _find(children, name):
    for child in children:
        if child[0] == name:
            return child
    return None


# ── FBX scale model ──────────────────────────────────────────────────────────


class FbxScaleInfo:
    """Per-node scale facts needed to undo the importer's half-applied scale."""

    def __init__(self):
        self.scale: dict[str, float] = {}        # accumulated world scale
        self.parent_scale: dict[str, float] = {}  # accumulated scale of parent
        self.local_translation: dict[str, tuple] = {}
        self.bind_posed: set[str] = set()
        self.translation_curve: dict[str, list] = {}
        self.has_scaled_bone = False


def read_scale_info(path: str, with_curves: bool = False) -> FbxScaleInfo | None:
    """Collect the scale facts for every ``Model`` node in ``path``.

    ``with_curves`` additionally decodes the ``Lcl Translation`` key arrays,
    which only the bind-pose rest repair needs.
    """
    try:
        roots = _parse_fbx(path, with_curves)
    except Exception:
        return None
    if not roots or "Objects" not in roots or "Connections" not in roots:
        return None

    models: dict[int, tuple] = {}
    poses: list[tuple] = []
    curve_nodes: dict[int, dict] = {}
    curves: dict[int, tuple] = {}
    for node in roots["Objects"][2]:
        name, props, children = node
        if name == "Model" and len(props) >= 3:
            attrs = {}
            props70 = _find(children, "Properties70")
            if props70:
                for entry in props70[2]:
                    if entry[1]:
                        attrs[entry[1][0]] = entry[1][4:]
            models[props[0]] = (props[1].split("\x00")[0], props[2], attrs)
        elif name == "Pose" and "BindPose" in str(props):
            poses.append(node)
        elif name == "AnimationCurveNode" and props:
            defaults = {}
            props70 = _find(children, "Properties70")
            if props70:
                for entry in props70[2]:
                    if len(entry[1]) >= 5:
                        defaults[entry[1][0]] = entry[1][4]
            curve_nodes[props[0]] = defaults
        elif name == "AnimationCurve" and props:
            times = _find(children, "KeyTime")
            values = _find(children, "KeyValueFloat")
            curves[props[0]] = (
                times[1][0] if times and times[1] and times[1][0] else [],
                values[1][0] if values and values[1] and values[1][0] else [],
            )

    if not models:
        return None

    parent: dict[int, int] = {}
    curve_of_node: dict[int, dict] = {}
    node_channel: dict[int, tuple] = {}
    for conn in roots["Connections"][2]:
        props = conn[1]
        if len(props) < 3:
            continue
        if props[0] == "OO":
            child, par = props[1], props[2]
            # A model has several OO parents (its node attribute, collections);
            # only the one that is itself a model is the hierarchy parent.
            if child in models and (par in models or par == 0):
                parent[child] = par
        elif props[0] == "OP" and len(props) >= 4:
            src, dst, channel = props[1], props[2], props[3]
            if src in curves:
                curve_of_node.setdefault(dst, {})[channel] = curves[src]
            elif src in curve_nodes and dst in models:
                node_channel[src] = (dst, channel)

    info = FbxScaleInfo()
    accumulated: dict[int, tuple] = {}

    def accumulate(uid: int, guard: frozenset = frozenset()) -> tuple:
        cached = accumulated.get(uid)
        if cached is not None:
            return cached
        if uid in guard:  # cyclic hierarchy; treat as unscaled
            return (1.0, 1.0, 1.0)
        own = models[uid][2].get("Lcl Scaling") or (1.0, 1.0, 1.0)
        inherit = models[uid][2].get("InheritType") or (1,)
        par = parent.get(uid, 0)
        if par in models and inherit[0] != _INHERIT_NO_SCALE:
            outer = accumulate(par, guard | {uid})
        else:
            outer = (1.0, 1.0, 1.0)
        value = tuple(outer[i] * own[i] for i in range(3))
        accumulated[uid] = value
        return value

    def uniform(triple) -> float:
        magnitude = abs(triple[0]) * abs(triple[1]) * abs(triple[2])
        return magnitude ** (1.0 / 3.0) if magnitude > 0.0 else 1.0

    for uid, (name, kind, attrs) in models.items():
        own = uniform(accumulate(uid))
        par = parent.get(uid, 0)
        outer = uniform(accumulate(par)) if par in models else 1.0
        # A duplicated name is only a problem when the duplicates disagree.
        if name in info.scale and (
            abs(info.scale[name] - own) > _SCALE_EPS
            or abs(info.parent_scale[name] - outer) > _SCALE_EPS
        ):
            info.scale[name] = 1.0
            info.parent_scale[name] = 1.0
            continue
        info.scale[name] = own
        info.parent_scale[name] = outer
        info.local_translation[name] = tuple(
            attrs.get("Lcl Translation") or (0.0, 0.0, 0.0)
        )
        if kind != "Mesh" and abs(own - 1.0) > _SCALE_EPS:
            info.has_scaled_bone = True

    for pose in poses:
        for child in pose[2]:
            if child[0] != "PoseNode":
                continue
            node = _find(child[2], "Node")
            if node and node[1] and node[1][0] in models:
                info.bind_posed.add(models[node[1][0]][0])

    # The cheap pass never read the key arrays, so it must not claim to have
    # curves -- an empty one would silently read as "translation never moves".
    if with_curves:
        for curve_node_uid, (model_uid, channel) in node_channel.items():
            if channel != "Lcl Translation":
                continue
            axes = curve_of_node.get(curve_node_uid)
            if not axes:
                continue
            defaults = curve_nodes.get(curve_node_uid, {})
            info.translation_curve[models[model_uid][0]] = [
                axes.get("d|X"), axes.get("d|Y"), axes.get("d|Z"), defaults
            ]

    return info


# ── repair ───────────────────────────────────────────────────────────────────


def _bone_key(info: FbxScaleInfo, name: str) -> str:
    """Map a Blender bone name back to its FBX node name.

    Blender appends ``.001`` to the second of two identically named nodes.
    """
    if name in info.scale:
        return name
    base = name.rsplit(".", 1)[0]
    return base if base in info.scale else name


def _action_fcurves(action) -> list:
    curves = []
    layers = getattr(action, "layers", None)
    if layers is not None:
        for layer in layers:
            for strip in layer.strips:
                for bag in getattr(strip, "channelbags", ()):
                    curves.extend(bag.fcurves)
    if not curves and hasattr(action, "fcurves"):
        curves.extend(action.fcurves)
    return curves


def _actions_of(obj) -> list:
    anim = getattr(obj, "animation_data", None)
    if anim is None:
        return []
    found = [anim.action] if anim.action else []
    for track in getattr(anim, "nla_tracks", ()):
        for strip in track.strips:
            if strip.action and strip.action not in found:
                found.append(strip.action)
    return found


def _sample_curve(curve, frame: float, fps: float, fallback: float) -> float:
    times, values = curve
    if not times or not values:
        return fallback
    target = frame / fps * _FBX_TIME_UNITS_PER_SECOND
    if target <= times[0]:
        return values[0]
    if target >= times[-1]:
        return values[-1]
    index = bisect.bisect_left(times, target)
    if index >= len(times):
        return values[-1]
    if times[index] == target:
        return values[index]
    lo_t, hi_t = times[index - 1], times[index]
    lo_v, hi_v = values[index - 1], values[index]
    if hi_t <= lo_t:
        return lo_v
    return lo_v + (hi_v - lo_v) * (target - lo_t) / (hi_t - lo_t)


def _repair_rest(bpy, armature, info: FbxScaleInfo) -> dict:
    """Rescale the rest offset of bones the FBX bind pose forgot.

    Also restores every bone length, which the importer reads in raw FBX units.
    Returns ``{bone_name: parent_scale}`` for the bones whose rest was moved --
    their animation channels have to be rebuilt rather than merely rescaled.
    """
    previous_mode = bpy.context.object.mode if bpy.context.object else "OBJECT"
    previous_active = bpy.context.view_layer.objects.active
    moved: dict[str, float] = {}
    try:
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode="EDIT")
        for bone in armature.data.edit_bones:
            key = _bone_key(info, bone.name)
            own = info.scale.get(key, 1.0)
            if bone.parent is not None:
                outer = info.parent_scale.get(key, 1.0)
                offset = bone.head - bone.parent.head
                local = info.local_translation.get(key)
                if (
                    local is not None
                    and abs(outer - 1.0) > _SCALE_EPS
                    and key not in info.bind_posed
                ):
                    raw = (local[0] ** 2 + local[1] ** 2 + local[2] ** 2) ** 0.5
                    want = raw * abs(outer)
                    got = offset.length
                    tolerance = 0.01 * want
                    # Only move it when the offset really is the unscaled one:
                    # rescaling it lands on the FBX offset, leaving it does not.
                    unscaled = abs(got * abs(outer) - want) <= tolerance
                    already_right = abs(got - want) <= tolerance
                    if want > 1e-9 and unscaled and not already_right:
                        direction = bone.tail - bone.head
                        bone.head = bone.parent.head + offset * outer
                        bone.tail = bone.head + direction
                        moved[bone.name] = outer
            if abs(own - 1.0) > _SCALE_EPS:
                bone.length = bone.length * abs(own)
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.context.view_layer.objects.active = previous_active
        if previous_active is not None and previous_mode != "OBJECT":
            try:
                bpy.ops.object.mode_set(mode=previous_mode)
            except RuntimeError:
                pass
    return moved


def _rebuild_location(bpy, armature, action, bone_name, outer, info, fps):
    """Rewrite one bone's ``location`` channel from the FBX translation curve.

    Used for bones whose rest we just moved: the importer derived their channel
    against the unscaled rest, so the values are meaningless rather than merely
    mis-scaled.
    """
    import mathutils

    bone = armature.data.bones.get(bone_name)
    if bone is None or bone.parent is None:
        return
    key = _bone_key(info, bone_name)
    static = info.local_translation.get(key, (0.0, 0.0, 0.0))
    curve = info.translation_curve.get(key)
    # bone-relative rest rotation: basis translation lives in this frame
    rest = bone.parent.matrix_local.to_3x3().inverted() @ bone.matrix_local.to_3x3()
    to_basis = rest.inverted()

    targets = [
        fcurve
        for fcurve in _action_fcurves(action)
        if fcurve.data_path == 'pose.bones["%s"].location' % bone_name
    ]
    for fcurve in targets:
        for key_point in fcurve.keyframe_points:
            frame = key_point.co[0]
            if curve is None:
                delta = mathutils.Vector((0.0, 0.0, 0.0))
            else:
                animated = [
                    _sample_curve(
                        curve[axis] or ([], []),
                        frame - 1.0,  # the importer offsets FBX frames by +1
                        fps,
                        (curve[3] or {}).get("d|" + "XYZ"[axis], static[axis]),
                    )
                    for axis in range(3)
                ]
                delta = mathutils.Vector(
                    tuple((animated[i] - static[i]) * outer for i in range(3))
                )
            value = (to_basis @ delta)[fcurve.array_index]
            shift = value - key_point.co[1]
            key_point.co[1] = value
            key_point.handle_left[1] += shift
            key_point.handle_right[1] += shift
        fcurve.update()


def repair_scaled_bone_import(bpy, filepath: str, objects=None) -> bool:
    """Undo Blender's half-applied bone scale on a freshly imported FBX.

    ``objects`` limits the repair to the objects created by this import; it
    defaults to every object in the file.  Returns ``True`` when something was
    repaired.
    """
    info = read_scale_info(filepath)
    if info is None or not info.has_scaled_bone:
        return False

    pool = objects if objects is not None else bpy.data.objects
    armatures = [obj for obj in pool if obj.type == "ARMATURE"]
    if not armatures:
        return False

    scene = bpy.context.scene
    fps = scene.render.fps / max(scene.render.fps_base, 1e-9)
    repaired = False
    curve_info = None
    for armature in armatures:
        moved = _repair_rest(bpy, armature, info)
        if moved and curve_info is None:
            # Rebuilding a channel needs the key arrays the cheap scan skipped.
            curve_info = read_scale_info(filepath, with_curves=True) or info
        for action in _actions_of(armature):
            for fcurve in _action_fcurves(action):
                if not fcurve.data_path.endswith(".location"):
                    continue
                try:
                    bone_name = fcurve.data_path.split('"')[1]
                except IndexError:
                    continue
                if bone_name in moved:
                    continue  # rebuilt below instead of rescaled
                factor = info.scale.get(_bone_key(info, bone_name), 1.0)
                if abs(factor - 1.0) <= _SCALE_EPS:
                    continue
                for key_point in fcurve.keyframe_points:
                    key_point.co[1] *= factor
                    key_point.handle_left[1] *= factor
                    key_point.handle_right[1] *= factor
                fcurve.update()
                repaired = True
            for bone_name, outer in moved.items():
                _rebuild_location(
                    bpy, armature, action, bone_name, outer, curve_info or info, fps
                )
        repaired = repaired or bool(moved)
    return repaired
