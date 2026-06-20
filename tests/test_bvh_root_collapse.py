from __future__ import annotations

import os
import sys
import textwrap


_ANYTOP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_ANYTOP_ROOT)
for _p in [_REPO_ROOT, _ANYTOP_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


from motion_lib import BVH


def _write_bvh(path, root_name):
    path.write_text(
        textwrap.dedent(
            f"""\
            HIERARCHY
            ROOT {root_name}
            {{
                OFFSET 1.000000 2.000000 3.000000
                CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
                JOINT Pelvis
                {{
                    OFFSET 0.000000 0.000000 0.000000
                    CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
                    JOINT Tail
                    {{
                        OFFSET 0.000000 1.000000 0.000000
                        CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
                    }}
                }}
            }}
            MOTION
            Frames: 1
            Frame Time: 0.033333
            0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
            """
        ),
        encoding="utf-8",
    )


def test_bvh_load_keeps_semantic_root_with_zero_offset_child(tmp_path):
    bvh_path = tmp_path / "semantic_root.bvh"
    _write_bvh(bvh_path, "Hips")

    _anim, names, _frametime = BVH.load(str(bvh_path))

    assert names == ["Hips", "Pelvis", "Tail"]


def test_bvh_load_collapses_nonsemantic_wrapper_root(tmp_path):
    bvh_path = tmp_path / "wrapper_root.bvh"
    _write_bvh(bvh_path, "Armature")

    _anim, names, _frametime = BVH.load(str(bvh_path))
    _raw_anim, raw_names, _raw_frametime = BVH.load(str(bvh_path), collapse_root=False)

    assert names == ["Pelvis", "Tail"]
    assert raw_names == ["Armature", "Pelvis", "Tail"]
