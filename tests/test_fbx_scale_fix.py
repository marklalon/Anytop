"""Cover the FBX bone-scale repair's read side.

Unity packs (Mini Legion, TT_RTS, ...) park a ``0.01`` ``Lcl Scaling`` on the
root bone and author the rest of the rig in centimetres.  Blender folds that
scale into the rest pose but not into the animation, so the clip explodes by
100x.  The repair reads the scale straight out of the FBX, so these tests build
tiny binary FBX files and assert the numbers that drive it.
"""

from __future__ import annotations

import os
import struct
import sys

import pytest

_ANYTOP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_ANYTOP_ROOT)
for _p in [_REPO_ROOT, _ANYTOP_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from motion_lib.fbx_scale_fix import (  # noqa: E402
    _FBX_TIME_UNITS_PER_SECOND,
    _bone_key,
    _sample_curve,
    read_scale_info,
)


# ── minimal binary FBX writer (version 7400, 32-bit offsets) ────────────────


def _prop(value, kind=None):
    if kind == "L" or isinstance(value, int) and kind is None:
        return b"L" + struct.pack("<q", value)
    if kind == "I":
        return b"I" + struct.pack("<i", value)
    if kind == "D" or isinstance(value, float):
        return b"D" + struct.pack("<d", value)
    if kind == "S" or isinstance(value, str):
        raw = value.encode("utf-8")
        return b"S" + struct.pack("<I", len(raw)) + raw
    if kind in ("d", "f", "l"):
        fmt = {"d": "d", "f": "f", "l": "q"}[kind]
        raw = struct.pack("<%d%s" % (len(value), fmt), *value)
        return (
            kind.encode()
            + struct.pack("<III", len(value), 0, len(raw))
            + raw
        )
    raise TypeError(kind)


def _node(name, props=(), children=()):
    return (name, tuple(props), tuple(children))


def _render(node, base):
    """Serialize one node; FBX ``EndOffset`` is absolute, hence ``base``."""
    name, props, children = node
    raw_name = name.encode("utf-8")
    body = b"".join(props)
    cursor = base + 12 + 1 + len(raw_name) + len(body)
    kids = b""
    for child in children:
        blob = _render(child, cursor)
        kids += blob
        cursor += len(blob)
    if children:
        kids += b"\x00" * 13  # null record closing a child list
        cursor += 13
    return (
        struct.pack("<III", cursor, len(props), len(body))
        + bytes([len(raw_name)])
        + raw_name
        + body
        + kids
    )


def _serialize(nodes):
    """Lay top-level nodes out after the 27-byte file header."""
    out = bytearray(b"Kaydara FBX Binary  \x00\x1a\x00" + struct.pack("<I", 7400))
    for node in nodes:
        out += _render(node, len(out))
    out += b"\x00" * 13
    return bytes(out)


def _p70(name, *values, kind=None):
    props = [_prop(name, "S"), _prop("", "S"), _prop("", "S"), _prop("", "S")]
    props += [_prop(v, kind) for v in values]
    return _node("P", props)


def _model(uid, name, kind, props70):
    return _node(
        "Model",
        [_prop(uid, "L"), _prop(name + "\x00\x01Model", "S"), _prop(kind, "S")],
        [_node("Properties70", (), props70)],
    )


def _connection(*props):
    return _node("C", [_prop(p, "S" if isinstance(p, str) else "L") for p in props])


def _write(tmp_path, name, models, connections, poses=(), anim=()):
    objects = _node("Objects", (), list(models) + list(poses) + list(anim))
    conns = _node("Connections", (), list(connections))
    path = os.path.join(str(tmp_path), name)
    with open(path, "wb") as handle:
        handle.write(_serialize([objects, conns]))
    return path


# ── fixtures ────────────────────────────────────────────────────────────────

ROOT, CHILD, LOOSE, MESH = 100, 101, 102, 103
CURVE_NODE, CURVE = 300, 400


def _scaled_rig(tmp_path, name="rig.fbx", inherit_child=1, root_scale=0.01):
    """Root bone scaled by ``root_scale``; ``Loose`` is missing from the bind pose."""
    models = [
        _model(ROOT, "Root", "LimbNode", [
            _p70("Lcl Translation", 0.0, 1.0, 0.0),
            _p70("Lcl Scaling", root_scale, root_scale, root_scale),
            _p70("InheritType", 1, kind="I"),
        ]),
        _model(CHILD, "Child", "LimbNode", [
            _p70("Lcl Translation", 30.0, 0.0, 0.0),
            _p70("InheritType", inherit_child, kind="I"),
        ]),
        _model(LOOSE, "Loose", "LimbNode", [
            _p70("Lcl Translation", 20.0, 0.0, 0.0),
        ]),
        _model(MESH, "Skin", "Mesh", [
            _p70("Lcl Scaling", 0.01, 0.01, 0.01),
        ]),
    ]
    poses = [_node(
        "Pose",
        [_prop(200, "L"), _prop("BindPose\x00\x01Pose", "S"), _prop("BindPose", "S")],
        [
            _node("PoseNode", (), [
                _node("Node", [_prop(uid, "L")]),
                _node("Matrix", [_prop([0.0] * 16, "d")]),
            ])
            for uid in (ROOT, CHILD, MESH)
        ],
    )]
    times = [int(t / 30.0 * _FBX_TIME_UNITS_PER_SECOND) for t in (0, 1, 2)]
    anim = [
        _node(
            "AnimationCurveNode",
            [_prop(CURVE_NODE, "L"), _prop("T\x00\x01AnimCurveNode", "S"), _prop("", "S")],
            [_node("Properties70", (), [
                _p70("d|X", 20.0), _p70("d|Y", 0.0), _p70("d|Z", 0.0),
            ])],
        ),
        _node(
            "AnimationCurve",
            [_prop(CURVE, "L"), _prop("\x00\x01AnimCurve", "S"), _prop("", "S")],
            [
                _node("KeyTime", [_prop(times, "l")]),
                _node("KeyValueFloat", [_prop([20.0, 24.0, 28.0], "f")]),
            ],
        ),
    ]
    connections = [
        _connection("OO", ROOT, 0),
        _connection("OO", MESH, 0),
        _connection("OO", CHILD, ROOT),
        _connection("OO", LOOSE, CHILD),
        _connection("OP", CURVE_NODE, LOOSE, "Lcl Translation"),
        _connection("OP", CURVE, CURVE_NODE, "d|X"),
    ]
    return _write(tmp_path, name, models, connections, poses, anim)


# ── tests ───────────────────────────────────────────────────────────────────


def test_root_scale_propagates_to_every_descendant(tmp_path):
    info = read_scale_info(_scaled_rig(tmp_path))
    assert info is not None
    assert info.has_scaled_bone
    # Every bone under the scaled root carries the accumulated 0.01 ...
    assert info.scale["Root"] == pytest.approx(0.01)
    assert info.scale["Child"] == pytest.approx(0.01)
    assert info.scale["Loose"] == pytest.approx(0.01)
    # ... while the parent scale is 1.0 only for the node that introduces it.
    assert info.parent_scale["Root"] == pytest.approx(1.0)
    assert info.parent_scale["Child"] == pytest.approx(0.01)
    assert info.parent_scale["Loose"] == pytest.approx(0.01)


def test_bind_pose_membership_is_reported(tmp_path):
    info = read_scale_info(_scaled_rig(tmp_path))
    assert {"Root", "Child"} <= info.bind_posed
    # ``Loose`` is the case that needs its rest offset rescaled by hand.
    assert "Loose" not in info.bind_posed
    assert info.local_translation["Loose"] == pytest.approx((20.0, 0.0, 0.0))


def test_inherit_type_2_blocks_the_parent_scale(tmp_path):
    info = read_scale_info(_scaled_rig(tmp_path, "rrs.fbx", inherit_child=2))
    assert info.scale["Root"] == pytest.approx(0.01)
    # eInheritRrs: the child does not pick the parent's scale up.
    assert info.scale["Child"] == pytest.approx(1.0)
    assert info.scale["Loose"] == pytest.approx(1.0)


def test_unit_scaled_rig_needs_no_repair(tmp_path):
    info = read_scale_info(_scaled_rig(tmp_path, "unit.fbx", root_scale=1.0))
    assert info is not None
    # A mesh object may still be scaled -- Blender stores that fine.
    assert info.scale["Skin"] == pytest.approx(0.01)
    assert not info.has_scaled_bone


def test_translation_curves_are_only_decoded_on_demand(tmp_path):
    path = _scaled_rig(tmp_path)
    assert read_scale_info(path).translation_curve == {}
    curves = read_scale_info(path, with_curves=True).translation_curve
    assert "Loose" in curves
    times, values = curves["Loose"][0]
    assert len(times) == len(values) == 3
    assert values == pytest.approx([20.0, 24.0, 28.0])


def test_non_binary_fbx_is_skipped(tmp_path):
    path = os.path.join(str(tmp_path), "ascii.fbx")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("; FBX 7.4.0 project file\nObjects:  {\n}\n")
    assert read_scale_info(path) is None


def test_sample_curve_interpolates_between_keys():
    times = [int(t / 30.0 * _FBX_TIME_UNITS_PER_SECOND) for t in (0, 2)]
    curve = (times, [0.0, 4.0])
    assert _sample_curve(curve, 0.0, 30.0, -1.0) == pytest.approx(0.0)
    assert _sample_curve(curve, 1.0, 30.0, -1.0) == pytest.approx(2.0)
    assert _sample_curve(curve, 2.0, 30.0, -1.0) == pytest.approx(4.0)
    # Outside the key range the curve holds, matching FBX constant extrapolation.
    assert _sample_curve(curve, -5.0, 30.0, -1.0) == pytest.approx(0.0)
    assert _sample_curve(curve, 99.0, 30.0, -1.0) == pytest.approx(4.0)
    assert _sample_curve(([], []), 1.0, 30.0, -1.0) == pytest.approx(-1.0)


def test_bone_key_falls_back_to_the_deduplicated_name(tmp_path):
    info = read_scale_info(_scaled_rig(tmp_path))
    # Blender renames the second of two identically named FBX nodes.
    assert _bone_key(info, "Child") == "Child"
    assert _bone_key(info, "Child.001") == "Child"
    assert _bone_key(info, "Nothing.001") == "Nothing.001"
