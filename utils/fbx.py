"""
Shared FBX import utilities.
"""
from __future__ import annotations

from typing import Optional


def patch_fbx_light_import():
    """Monkey-patch the FBX importer's blen_read_light to handle Blender 5.0.

    In Blender 5.0, CyclesLightSettings.cast_shadow was removed, but the
    io_scene_fbx addon still tries to set it during FBX import, causing a
    crash. We wrap blen_read_light to swallow that specific AttributeError.
    """
    import sys
    import importlib

    mod = sys.modules.get("io_scene_fbx.import_fbx")
    if mod is None:
        try:
            mod = importlib.import_module("io_scene_fbx.import_fbx")
        except ImportError:
            return
    if mod is None or not hasattr(mod, "blen_read_light"):
        return

    original_fn = mod.blen_read_light

    def _patched_blen_read_light(fbx_tmpl, fbx_obj, settings, _orig=original_fn):
        try:
            return _orig(fbx_tmpl, fbx_obj, settings)
        except AttributeError as exc:
            if "cast_shadow" in str(exc):
                return None  # Skip the light; it will be deleted after import anyway
            raise

    mod.blen_read_light = _patched_blen_read_light


def import_fbx(filepath: str, ignore_leaf_bones: bool = True) -> None:
    """Import an FBX file into the current Blender scene.

    This wraps the common pattern of patching the light import and calling
    ``bpy.ops.import_scene.fbx`` with our standard parameters.

    ``force_connect_children`` is hardcoded to False to preserve the source
    rig's original head/tail connectivity semantics.
    ``automatic_bone_orientation`` is hardcoded to True for consistent
    bone axis orientation across imports.

    Raises ``RuntimeError`` if bpy is not available.
    """
    import bpy

    patch_fbx_light_import()
    bpy.ops.import_scene.fbx(
        filepath=filepath,
        ignore_leaf_bones=ignore_leaf_bones,
        force_connect_children=False,
        automatic_bone_orientation=True,
        bake_space_transform=False,
        use_custom_normals=False,
        use_image_search=False,
    )


def clear_scene() -> None:
    """Reset Blender to a fresh empty scene."""
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)


def remove_lights_and_cameras() -> None:
    """Remove all LIGHT and CAMERA objects from the current scene."""
    import bpy

    for obj in list(bpy.data.objects):
        if obj.type in {"LIGHT", "CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)
