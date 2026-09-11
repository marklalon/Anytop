#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render every processed Truebones clip to a 224x224 review GIF.

Input is the untouched source tree ``dataset/truebones/zoo/Truebone_Z-OO`` --
one directory per species, one textured GLB per animation -- and the output is
one GIF per *processed clip*::

    truebones_processed/review/gif/<Species>_<Action>.gif

The names are the ``bvhs/`` names, exactly: a GIF is the picture of the clip
that sits next to it in ``bvhs/`` and ``motions/``, so ``Alligator_BigMouth.gif``
shows what ``Alligator_BigMouth.bvh`` contains and nothing has to be re-derived
to line the two up.  Which source GLB that is comes straight out of
``motion_metadata.json`` (``motions[<clip>.npy].source_fbx_path``), the only
record of the mapping -- the preprocessor's filename rules strip species
prefixes, drop all-in-one bundles and CamelCase the rest, and re-implementing
them here would be a second copy free to drift.

The scene is the one ``E:/Dataset/UnityBundles/build/render_for_llm.py`` renders
the UnityBundles library with, so the two review galleries read the same way: a
fixed 1x1 see-through grid at ``z = 0`` under a shadow-casting sun, a 50mm lens
on a 45-degree oblique 25 degrees up, and a camera that holds still until the
silhouette leaves a centre safe box and then trucks the minimum amount that puts
it back (:func:`_solve_follow`).  The camera solver, the ground and the framing
constants below are lifted from that script; the passes it has that this one
does not (per-species stills, mp4 mux, a render index) exist for a VLM, and the
reader here is a human with a browser.

Unlike that library, Truebones species do not share a facing: 58 of the 74 need a
quarter turn and 9 need a half turn before they look at the camera.  That angle
is already known -- ``cond.npy`` carries the per-species ``orientation_quat`` the
preprocessor rotates each skeleton by -- so it is read from there rather than
guessed or hand-listed (see :func:`_load_facing`).

Run with the project venv so ``bpy`` resolves::

    .venv/Scripts/python.exe Anytop/dataset/truebones/zoo/truebones_processed/review/render_gifs.py
    .venv/Scripts/python.exe .../render_gifs.py --filter Dog --overwrite
    .venv/Scripts/python.exe .../render_gifs.py --clip 'Horse_*' -j 4 --dry-run
"""
from __future__ import annotations

import argparse
import fnmatch
import io
import json
import math
import multiprocessing
import os
import shutil
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROCESSED_ROOT = os.path.dirname(HERE)                       # truebones_processed
ZOO_ROOT = os.path.dirname(PROCESSED_ROOT)                   # zoo

# -- paths -------------------------------------------------------------------
RAW_ROOT = os.path.join(ZOO_ROOT, "Truebone_Z-OO")
GIF_ROOT = os.path.join(HERE, "gif")
METADATA_PATH = os.path.join(PROCESSED_ROOT, "motion_metadata.json")
COND_PATH = os.path.join(PROCESSED_ROOT, "cond.npy")

# -- gif ---------------------------------------------------------------------
GIF_SIZE = 224           # the review grid's card, so the browser scales nothing
GIF_COLORS = 128
SUPERSAMPLE = 2          # render 448 and Lanczos down: at 224 an aliased limb
#                          reads as noise, and noise is what the reader must not
#                          mistake for motion
DURATION_CLAMP = (40, 250)   # ms per frame, applied to the real clip timing
TARGET_FPS = 10.0
MAX_FRAMES = 32          # cap for the few 250-frame clips
MIN_FRAMES = 4           # a short clip still gets four frames

# -- image / engine ----------------------------------------------------------
# Square is a precondition, not a preference: the projection solver below uses
# the square pinhole model Blender renders with under sensor_fit AUTO.
TAA_RENDER_SAMPLES = 16
CYCLES_SAMPLES = 16

# -- camera ------------------------------------------------------------------
CAMERA_LENS = 50.0       # the wide lens is what makes the ground perspective read
CAMERA_SENSOR = 36.0
CAMERA_CLIP = (0.01, 10000.0)
ELEVATION_DEG = 25.0
# Azimuth 0 stands in front of the creature; the oblique in between reads gait
# (a side view's strength) and the front of the creature in the same frame.
CLIP_AZIMUTH_DEG = 45.0
# Two different halves, and they are not the same knob.  FILL_HALF is the
# framing target -- how big the widest frame of the clip is drawn.  SAFE_HALF is
# the dead zone -- how far the silhouette may wander before the camera moves.
FILL_HALF = 0.28         # widest frame fills the centre 56% of the frame
SAFE_HALF = 0.34         # safe box = centre 68% of the frame
DEPTH_DEAD = 0.08        # distance drift the safe stage tolerates, as a fraction
#                          of the solved distance: the oblique azimuth puts part
#                          of every forward travel along the view axis, where a
#                          lateral truck cannot reach it
HOLD_RATIO = 1.45        # a moving camera has to earn itself: hold the clip on
#                          one fixed position unless the creature *ends up*
#                          somewhere else, by more than this much framing cost
PEAK_RATIO = 2.5         # ... or unless it goes so far and comes back that a
#                          fixed camera would draw it under 40% size
DESIRE_SIGMA = 3.0       # frames of blur on the ideal path (see _solve_follow)
SMOOTH_SIGMA = 1.2       # frames of zero-phase blur on the solved camera path
SMOOTH_ITERS = 3         # smooth / re-clamp rounds (see _smooth_path)
HARD_HALF = 0.46         # nothing may leave the centre 92% of frame, ever
HARD_DEPTH = 0.20        # nor come 20% nearer than the solved distance
FIT_ITERS = 8

# -- scene -------------------------------------------------------------------
GROUND_Z = 0.0           # one height for the whole library, no per-species data
GROUND_SIZE = 200.0      # edge is off-camera at every distance we solve for
GROUND_REACH = 0.25      # how far below itself a creature drags the ground into
#                          frame, in its own heights (see _ground_anchor)
GRID_CELL_SIZE = 1.0     # cell edge in world units: a fixed cell is what makes
#                          displacement comparable across clips and species
GRID_LINE_COLOR = (0.05, 0.05, 0.07)
GRID_LINE_HALF = 0.025   # line half-width, as a fraction of a cell
GRID_CELL_ALPHA = 0.5    # how solid the gaps are: enough floor for the contact
#                          shadow to land on, sheer enough to see through
GROUND_ROUGHNESS = 0.9
GROUND_NAME = "ReviewGround"
WORLD_COLOR = (0.55, 0.72, 0.95)
WORLD_STRENGTH = 0.4     # the sky is the fill light, so how deep a shadow reads
SUN_ENERGY = 6.0         # is the sun-to-sky ratio: little sky, a lot of sun
SUN_TILT_DEG = 50.0      # 40 deg above the horizon -> shadow ~1.2x the height
# The sun's azimuth is measured from the *camera's*, so the shadow always falls
# across frame instead of hiding behind the creature at some view angles.
SUN_AZIMUTH_OFFSET_DEG = -115.0
SUN_ANGLE_DEG = 3.0      # soft edge, so contact shadows stay readable
# Standard, not AgX: AgX spends its contrast budget on highlight rolloff, which
# costs ~40% of the floor's tonal spread -- the grid and the shadow both.
VIEW_TRANSFORM = "Standard"
# The Truebones GLBs come out of the FBX conversion with a metallic sheen they
# were never authored for; the texture-linked Base Color is left alone.
MATERIAL_METALLIC = 0.4
MATERIAL_ROUGHNESS = 0.8

SCENE_FPS = 30           # every Truebones clip was authored at 30 (checked on
#                          all 1039 exported BVHs); this is the grid the glTF
#                          importer converts the clip's key times onto


# ============================ job planning ==================================
def _load_clip_sources(metadata_path=METADATA_PATH):
    """``{clip stem: (species, source GLB path)}`` from ``motion_metadata.json``.

    The clip stem *is* the BVH stem (``Alligator_BigMouth``), which is what the
    GIF is named after.  Nothing else in the tree records which source file a
    clip came from, and the filename rules that produced it are lossy.
    """
    with open(metadata_path, "r", encoding="utf-8") as fh:
        motions = json.load(fh).get("motions", {})
    clips = {}
    for motion_name, entry in motions.items():
        stem = os.path.splitext(motion_name)[0]
        source = entry.get("source_fbx_path")
        if source:
            clips[stem] = (str(entry.get("object_type") or stem.split("_")[0]),
                           source)
    return clips


def _load_facing(cond_path=COND_PATH):
    """``{species: yaw deg}`` putting each species' front towards azimuth 0.

    ``cond.npy`` stores, per species, the ``orientation_quat`` the preprocessor
    rotates the skeleton by to make it face the canonical +Z; it is always a pure
    +Y turn snapped to a quarter (measured: -90 for 58 species, +-180 for 9, 0
    for 7).  glTF's +Y is Blender's +Z and glTF's +Z is Blender's -Y (the
    importer's Y-up conversion is a rotation, so a turn about +Y keeps its
    angle), and Blender's -Y is the axis :func:`_camera_basis` measures azimuth
    from -- so the quat's angle is the azimuth correction, up to its sign.

    The sign is the inverse: the quat turns the creature onto the canonical
    front, and what the camera needs is where the creature is pointing *now*.
    Verified rather than argued, because it is easy to get backwards and a
    180-degree error renders the whole library from behind -- for each species
    the head and hips named by ``canonical_joint_names`` were read out of the
    Blender armature and the real forward measured off them.  All 70 species
    whose head has any horizontal offset from their hips land on ``-angle``
    within a few degrees.  The four that do not are unmeasurable, not
    counterexamples: Spider and Scorpion carry their head directly above their
    hips, and Crab and Pigeon have no head joint at all.

    A missing or unreadable cond costs nothing but the correction.
    """
    try:
        cond = np.load(cond_path, allow_pickle=True).item()
    except (OSError, ValueError, AttributeError) as exc:
        print("[WARN] no per-species facing (%s): %s" % (cond_path, exc))
        return {}
    facing = {}
    for key, entry in cond.items():
        quat = entry.get("orientation_quat") if isinstance(entry, dict) else None
        if quat is None:
            continue
        quat = np.asarray(quat, dtype=np.float64).reshape(4)   # (w, x, y, z)
        facing[str(key).rsplit("/", 1)[-1]] = -math.degrees(
            2.0 * math.atan2(float(quat[2]), float(quat[0])))
    return facing


def _load_scales(cond_path=COND_PATH):
    """``{species: scale_factor}`` -- cond's own canonical size normalisation.

    The GLBs are not on a shared scale: as imported, a Rat spans 0.35 world
    units and a Crab spans 160, a 457x spread, because most assets carry a 0.01
    node scale and the Crab carries 1.0.  The grid is what pays for that.  Its
    cell is a fixed world unit, so at the small end a whole creature sits inside
    one cell and at the large end the lines converge into flat grey and the
    200-unit ground plane's own edge comes into frame -- which is what the Crab
    renders as.

    ``scale_factor`` is the number the preprocessor multiplies each species by
    to reach the canonical size the model trains on, so it is both the fix and
    the right unit: applied, the library lands between 1.4 and 2.4 units and one
    cell means the same fraction of a body everywhere.  Note it multiplies the
    *armature-local* coordinates, so the node scale has to be divided back out
    (see :func:`_rescale_to_canonical`).
    """
    try:
        cond = np.load(cond_path, allow_pickle=True).item()
    except (OSError, ValueError, AttributeError):
        return {}                      # _load_facing already warned
    scales = {}
    for key, entry in cond.items():
        if not isinstance(entry, dict):
            continue
        scale = entry.get("scale_factor")
        if scale:
            scales[str(key).rsplit("/", 1)[-1]] = float(scale)
    return scales


def _plan(args, clips, facing, scales):
    """``(jobs, skipped)`` -- one job per selected clip whose GIF is missing."""
    jobs, skipped = [], 0
    base_opts = {"engine": args.engine, "size": args.size,
                 "supersample": args.supersample, "colors": args.colors,
                 "checker_size": args.checker_size, "scene_fps": SCENE_FPS,
                 "follow": args.follow == "on", "smooth_sigma": args.smooth_sigma,
                 "target_fps": args.fps, "max_frames": args.max_frames,
                 "dither": args.dither, "frames_dir": args.frames_dir}
    for clip in sorted(clips):
        species, source = clips[clip]
        if not fnmatch.fnmatch(species, args.filter):
            continue
        if not fnmatch.fnmatch(clip, args.clip):
            continue
        gif_path = os.path.join(args.gif_root, clip + ".gif")
        if not args.overwrite and os.path.isfile(gif_path) \
                and os.path.getsize(gif_path):
            skipped += 1
            continue
        if not os.path.isfile(source):
            print("  [WARN] %s: source is gone (%s)" % (clip, source))
            continue
        opts = dict(base_opts, facing_yaw_deg=facing.get(species, 0.0),
                    canonical_scale=scales.get(species))
        jobs.append((clip, species, source, gif_path, opts))
    return jobs, skipped


# ============================ bpy: scene parts ==============================
def _action_fcurves(action):
    """Every FCurve *action* holds, slotted (Blender 4.4+) or legacy."""
    if action is None:
        return []
    legacy = getattr(action, "fcurves", None)
    if legacy is not None and len(legacy):
        return list(legacy)
    out = []
    for layer in getattr(action, "layers", ()):
        for strip in layer.strips:
            for bag in getattr(strip, "channelbags", ()):
                out.extend(bag.fcurves)
    return out


def _sample_times(armature):
    """Sorted keyframe times of the armature's action, in scene-frame units."""
    anim = armature.animation_data
    action = anim.action if anim else None
    times = sorted({round(float(key.co[0]), 6)
                    for fcurve in _action_fcurves(action)
                    for key in fcurve.keyframe_points})
    return times or [0.0]


def _set_time(scene, sample_time):
    frame = math.floor(sample_time)
    scene.frame_set(frame, subframe=float(sample_time - frame))


def _source_fps(times, scene_fps):
    """Playback rate of the source clip, read off its own key spacing.

    Every Truebones clip was authored at 30 fps, so this is a constant in
    practice -- but it is the number both the frame stride and the GIF's frame
    delay are derived from, and reading it beats asserting it.
    """
    if len(times) < 2:
        return float(scene_fps)
    deltas = np.diff(np.asarray(times, dtype=np.float64))
    positive = deltas[deltas > 1e-9]
    step = float(np.median(positive)) if positive.size else 1.0
    return float(scene_fps) / max(step, 1e-6)


def _pick_times(times, stride, max_frames=MAX_FRAMES, min_frames=MIN_FRAMES):
    """The frames to render: every *stride*-th key, endpoints kept, capped."""
    if len(times) <= min_frames:
        return list(times)
    picked = list(times[::max(1, stride)])
    if picked[-1] != times[-1]:
        picked.append(times[-1])
    if len(picked) > max_frames:
        idx = np.linspace(0, len(picked) - 1, max_frames).round().astype(int)
        picked = [picked[i] for i in sorted(set(idx.tolist()))]
    if len(picked) < min_frames:
        idx = np.linspace(0, len(times) - 1, min_frames).round().astype(int)
        picked = [times[i] for i in sorted(set(idx.tolist()))]
    return picked


def _durations_ms(times, source_fps):
    """Per-frame GIF delays, so the GIF runs at the speed the clip does.

    A uniform delay would be a lie for any clip long enough to hit
    ``--max-frames``: those are decimated by :func:`_pick_times`, and playing
    them at the short clips' rate speeds them up by however much was dropped.
    The clamp keeps a very long clip watchable and a very short one from
    strobing.
    """
    low, high = DURATION_CLAMP
    if len(times) < 2:
        return [high]
    deltas = [(times[i + 1] - times[i]) / source_fps
              for i in range(len(times) - 1)]
    deltas.append(deltas[-1])      # the last frame holds as long as the one before
    return [int(min(high, max(low, round(delta * 1000.0)))) for delta in deltas]


def _setup_scene(bpy, resolution, engine, taa_samples):
    scene = bpy.context.scene
    render = scene.render
    if engine == "cycles":
        _enable_cycles(bpy)
        render.engine = "CYCLES"
        cycles = scene.cycles
        cycles.samples = CYCLES_SAMPLES
        cycles.use_denoising = True
        cycles.device = "CPU"
    else:
        render.engine = "BLENDER_EEVEE"    # 5.x id; this is EEVEE Next
        eevee = scene.eevee
        eevee.taa_render_samples = taa_samples
        if hasattr(eevee, "use_shadows"):
            eevee.use_shadows = True
    render.resolution_x = resolution
    render.resolution_y = resolution
    render.resolution_percentage = 100
    render.film_transparent = False        # the ground is the point; keep it opaque
    render.image_settings.file_format = "PNG"   # lossless into the quantizer
    render.image_settings.color_mode = "RGB"
    render.image_settings.compression = 15      # these frames die in a temp dir
    render.use_persistent_data = True
    scene.view_settings.view_transform = VIEW_TRANSFORM
    scene.view_settings.look = "None"

    world = bpy.data.worlds.new("ReviewWorld")
    if world.node_tree is None:      # 5.x hands one over already; setting the
        world.use_nodes = True       # flag there is deprecated
    background = world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs[0].default_value = (WORLD_COLOR[0], WORLD_COLOR[1],
                                              WORLD_COLOR[2], 1.0)
        background.inputs[1].default_value = WORLD_STRENGTH
    scene.world = world
    return scene


def _enable_cycles(bpy):
    if "CYCLES" in {item.identifier for item
                    in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}:
        return
    import addon_utils
    addon_utils.enable("cycles", default_set=True, persistent=True)


def _add_sun(bpy, scene):
    """One sun, side-above, casting shadows.  Aim it with :func:`_aim_sun`.

    The contact shadow is the reference that answers "is this foot on the
    ground" -- more useful here than the ground plane on its own.
    """
    data = bpy.data.lights.new("ReviewSun", type="SUN")
    data.energy = SUN_ENERGY
    data.use_shadow = True
    data.angle = math.radians(SUN_ANGLE_DEG)
    sun = bpy.data.objects.new("ReviewSun", data)
    scene.collection.objects.link(sun)
    return sun


def _aim_sun(sun, azimuth_deg):
    """Point the sun over the camera's shoulder for a camera at *azimuth_deg*."""
    sun.rotation_euler = (math.radians(SUN_TILT_DEG), 0.0,
                          math.radians(azimuth_deg + SUN_AZIMUTH_OFFSET_DEG))


def _add_ground(bpy, scene, cell_size):
    """A fixed-size grid at ``z = 0``, built from data (no operators).

    Lines and not filled cells, because an opaque floor hides whatever sinks
    below it, and this library is full of deaths, digs and swims that do exactly
    that.  A grid keeps the same reading -- fixed cell size, so displacement is
    still "count the squares" -- while the gaps stay see-through.  The cells are
    not fully transparent: ``GRID_CELL_ALPHA`` of floor is what the contact
    shadow lands on.  Object texture coordinates put the grid in world units, so
    a cell is exactly *cell_size* units wide.
    """
    half = GROUND_SIZE * 0.5
    mesh = bpy.data.meshes.new(GROUND_NAME)
    mesh.from_pydata([(-half, -half, 0.0), (half, -half, 0.0),
                      (half, half, 0.0), (-half, half, 0.0)], [], [(0, 1, 2, 3)])
    mesh.update()
    ground = bpy.data.objects.new(GROUND_NAME, mesh)
    ground.location = (0.0, 0.0, GROUND_Z)
    scene.collection.objects.link(ground)

    mat = bpy.data.materials.new(GROUND_NAME)
    if mat.node_tree is None:        # see _setup_scene
        mat.use_nodes = True
    tree, links = mat.node_tree, mat.node_tree.links
    bsdf = tree.nodes.get("Principled BSDF")

    def math_node(op, a=None, b=None):
        node = tree.nodes.new("ShaderNodeMath")
        node.operation = op
        for i, operand in enumerate((a, b)):
            if operand is None:
                continue
            if hasattr(operand, "default_value") or hasattr(operand, "links"):
                links.new(operand, node.inputs[i])
            else:
                node.inputs[i].default_value = float(operand)
        return node.outputs[0]

    coords = tree.nodes.new("ShaderNodeTexCoord")
    mapping = tree.nodes.new("ShaderNodeMapping")
    scale = 1.0 / max(float(cell_size), 1e-6)
    mapping.inputs["Scale"].default_value = (scale, scale, scale)
    links.new(coords.outputs["Object"], mapping.inputs["Vector"])
    split = tree.nodes.new("ShaderNodeSeparateXYZ")
    links.new(mapping.outputs["Vector"], split.inputs["Vector"])

    # Distance to the nearest cell line, per axis: min(fract(u), 1 - fract(u)).
    edges = []
    for axis in ("X", "Y"):
        frac = math_node("FRACT", split.outputs[axis])
        edges.append(math_node("MINIMUM", frac, math_node("SUBTRACT", 1.0, frac)))
    nearest = math_node("MINIMUM", *edges)
    on_line = math_node("LESS_THAN", nearest, GRID_LINE_HALF)

    bsdf.inputs["Base Color"].default_value = (*GRID_LINE_COLOR, 1.0)
    links.new(math_node("MAXIMUM", on_line, GRID_CELL_ALPHA), bsdf.inputs["Alpha"])
    bsdf.inputs["Roughness"].default_value = GROUND_ROUGHNESS
    bsdf.inputs["Metallic"].default_value = 0.0
    # Blended, not dithered: at 16 TAA samples a dithered part-alpha floor comes
    # out as noise.
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "BLENDED"
    else:
        mat.blend_method = "BLEND"
    mesh.materials.append(mat)
    return ground


def _first_upstream(socket, node_type, seen=None):
    """The nearest node of *node_type* feeding *socket* (depth-first), or None."""
    if not socket.is_linked:
        return None
    seen = set() if seen is None else seen
    node = socket.links[0].from_node
    if node.as_pointer() in seen:
        return None
    seen.add(node.as_pointer())
    if node.type == node_type:
        return node
    for candidate in node.inputs:
        found = _first_upstream(candidate, node_type, seen)
        if found is not None:
            return found
    return None


def _solidify_alpha(mat, bsdf):
    """Make the creature opaque again, keeping any genuine texture cutout.

    Truebones GLBs come out of the FBX conversion with ``alphaMode: BLEND`` and
    an alpha chain of ``base colour texture alpha x COLOR_0 vertex alpha``.  The
    vertex alpha is not a mask -- it is whatever the conversion left in the
    attribute -- and it renders the whole creature see-through: the ground grid
    shows straight through a dog.  Dropping the chain back to the texture's own
    alpha keeps the cutouts that are real (feather cards, wing membranes, the
    hair and eye masks) and throws away the blanket translucency; a material
    whose alpha never came from a texture is simply opaque.

    Alpha-hashed rather than blended, so the surviving cutouts do not need the
    depth sort blending would: at 16 TAA samples the hash resolves to a clean
    edge, and a fully opaque texel stays fully opaque either way.
    """
    alpha = bsdf.inputs.get("Alpha")
    if alpha is None:
        return
    texture = _first_upstream(alpha, "TEX_IMAGE")
    for link in list(alpha.links):
        mat.node_tree.links.remove(link)
    if texture is not None and "Alpha" in texture.outputs:
        mat.node_tree.links.new(texture.outputs["Alpha"], alpha)
    else:
        alpha.default_value = 1.0
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "DITHERED"
    else:
        mat.blend_method = "HASHED"


def _normalize_materials(bpy):
    """Kill the imported Metallic sheen and the blanket alpha; keep textures.

    Called before the ground exists, so the grid keeps its own values.
    """
    for mat in bpy.data.materials:
        if mat.node_tree is None:
            continue
        for node in mat.node_tree.nodes:
            if node.type != "BSDF_PRINCIPLED":
                continue
            metallic = node.inputs.get("Metallic")
            if metallic is not None and not metallic.is_linked:
                metallic.default_value = MATERIAL_METALLIC
            rough = node.inputs.get("Roughness")
            if rough is not None and not rough.is_linked:
                rough.default_value = MATERIAL_ROUGHNESS
            _solidify_alpha(mat, node)


def _add_camera(bpy, scene, basis):
    from mathutils import Matrix

    right, up, view = basis
    back = -view
    data = bpy.data.cameras.new("ReviewCamera")
    data.type = "PERSP"
    data.lens = CAMERA_LENS
    data.sensor_width = CAMERA_SENSOR
    data.sensor_fit = "AUTO"
    data.clip_start, data.clip_end = CAMERA_CLIP
    cam = bpy.data.objects.new("ReviewCamera", data)
    cam.rotation_euler = Matrix(((right[0], up[0], back[0]),
                                 (right[1], up[1], back[1]),
                                 (right[2], up[2], back[2]))).to_euler()
    scene.collection.objects.link(cam)
    scene.camera = cam
    return cam


def _visible_meshes(bpy):
    """The meshes that actually render -- what the framing may be solved on.

    Every Truebones GLB carries a hidden unit ``Icosphere`` origin marker whose
    ``hide_render`` is False but whose ``visible_get()`` is not; framing on its
    +-1 bounds squashes the creature to a dot and drags the safe box around after
    it.  Faces are checked too: a mesh with no polygons contributes nothing.
    """
    return [obj for obj in bpy.data.objects
            if obj.type == "MESH" and not obj.hide_render
            and obj.visible_get() and len(obj.data.polygons) > 0]


# =========================== camera solving =================================
def _camera_basis(azimuth_deg, elevation_deg):
    """World right / up / view-direction for a fixed orbit angle.

    Azimuth 0 stands where the creature is looking.  A GLB faces glTF +Z, which
    the importer's Y-up-to-Z-up conversion lands on Blender -Y, so that is the
    axis the azimuth is measured from -- and the axis ``facing_yaw_deg`` turns
    the whole orbit around when a species was authored facing elsewhere.
    """
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    horizontal = np.array([math.sin(azimuth), -math.cos(azimuth), 0.0])
    offset = np.array([horizontal[0] * math.cos(elevation),
                       horizontal[1] * math.cos(elevation),
                       math.sin(elevation)])
    view = -offset                                  # camera looks back at the target
    right = np.cross(view, np.array([0.0, 0.0, 1.0]))
    norm = np.linalg.norm(right)
    right = right / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])
    up = np.cross(right, view)
    return right, up / np.linalg.norm(up), view


def _ground_anchor(points):
    """The cloud again, flattened towards the ground, as a framing anchor.

    Framing the creature alone lets its contact shadow fall out of shot, and the
    shadow is what answers "is that foot on the ground".  The drop is clamped to
    ``GROUND_REACH`` of the creature's own *height*, so a walker lands exactly on
    ``z = 0`` and keeps its shadow while a flier (this library has bats, eagles
    and dragons) shows at most a quarter of its own height of air beneath it
    instead of shrinking to a speck above a mandatory floor.
    """
    lowest, highest = float(points[:, 2].min()), float(points[:, 2].max())
    anchor = points.copy()
    anchor[:, 2] = max(GROUND_Z, lowest - GROUND_REACH * (highest - lowest))
    return anchor


def _anchored(cloud):
    """A framing cloud: the creature plus its ground anchor.

    Kept separate from the creature itself because the two answer different
    questions -- what has to fit in the picture (anchored) versus where the
    creature actually is (not).  Mixing them makes a take-off read as a creature
    standing still; see :func:`_solve_follow`.
    """
    return np.concatenate([cloud, _ground_anchor(cloud)], 0)


def _frame_cloud(bpy, scene, meshes, sample_time):
    """World-space deformed vertices at one frame (creature only)."""
    _set_time(scene, sample_time)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    parts = []
    for obj in meshes:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.data
        count = len(mesh.vertices)
        if not count:
            continue
        coords = np.empty(count * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", coords)
        matrix = np.asarray(evaluated.matrix_world, dtype=np.float64)
        parts.append(coords.reshape(count, 3) @ matrix[:3, :3].T + matrix[:3, 3])
    if not parts:
        return np.empty((0, 3))
    return np.concatenate(parts, 0)


def _joint_cloud(bpy, scene, armature, sample_time):
    """Fallback cloud for a meshless GLB: every pose-bone head."""
    _set_time(scene, sample_time)
    bpy.context.view_layer.update()
    matrix = armature.matrix_world
    heads = np.array([tuple(matrix @ bone.head) for bone in armature.pose.bones])
    return heads if len(heads) else np.empty((0, 3))


def _project(points, cam_pos, basis, k):
    """Blender's square pinhole projection: NDC offset from centre is k*X/depth."""
    right, up, view = basis
    rel = points - cam_pos
    depth = np.maximum(rel @ view, 1e-6)
    return 0.5 + k * (rel @ right) / depth, 0.5 + k * (rel @ up) / depth, depth


def _centre_on(cloud, centre, distance, basis, k, iters=2):
    """Pan *centre* so the cloud's projected bbox sits at the frame centre.

    Perspective divides each point by its own depth, so centring the geometric
    centre does not centre the silhouette; the offset can only be read after a
    projection, hence the loop.
    """
    right, up, view = basis
    for _ in range(iters):
        xs, ys, depth = _project(cloud, centre - view * distance, basis, k)
        per_ndc = float(depth.mean()) / k
        shift_x = 0.5 - (float(xs.min()) + float(xs.max())) * 0.5
        shift_y = 0.5 - (float(ys.min()) + float(ys.max())) * 0.5
        centre = centre - right * (shift_x * per_ndc) - up * (shift_y * per_ndc)
    return centre


def _frame_at(clouds, distance, basis, k):
    """Per-frame look-at anchors that centre each cloud at a *given* distance."""
    return [_centre_on(cloud, (cloud.min(0) + cloud.max(0)) * 0.5, distance,
                       basis, k) for cloud in clouds]


def _solve_distance(clouds, basis, k, fill_half):
    """Distance at which the *widest* frame of the clip just fills ``fill_half``.

    Solved over every frame that will be rendered, not over the union of them:
    the union is what a fixed camera framing a whole gallop would have to hold,
    and that is exactly what shrinks the creature to a speck.  One distance for
    the whole clip means on-screen size does not depend on how far it travels.
    """
    _, _, view = basis
    centres = [(cloud.min(0) + cloud.max(0)) * 0.5 for cloud in clouds]
    diag = max(float(np.linalg.norm(cloud.max(0) - cloud.min(0))) for cloud in clouds)
    distance = 2.0 * (diag or 1.0)
    for _ in range(FIT_ITERS):
        worst = 0.0
        for i, cloud in enumerate(clouds):
            centres[i] = _centre_on(cloud, centres[i], distance, basis, k)
            xs, ys, _ = _project(cloud, centres[i] - view * distance, basis, k)
            half = max(float(xs.max() - xs.min()), float(ys.max() - ys.min())) * 0.5
            worst = max(worst, half)
        distance *= max(worst, 1e-6) / fill_half
    # The centres above were panned at the previous iteration's distance; re-pan
    # them at the one being returned, since the caller frames on them.
    centres = [_centre_on(cloud, centre, distance, basis, k)
               for cloud, centre in zip(clouds, centres)]
    return distance, centres


def _gaussian_smooth(path, sigma):
    """Blur *path* along the frame axis with a symmetric (zero-phase) kernel.

    Symmetric matters: a causal filter would make the camera lag the creature by
    its own time constant.  Solving the whole path before rendering a frame is
    what buys the non-causal filter -- the camera can start easing *before* the
    creature reaches the boundary.  Edges are replicated, so a clip that starts
    or ends still stays still.
    """
    if sigma <= 0.0 or len(path) < 3:
        return path
    radius = max(1, int(math.ceil(3.0 * sigma)))
    taps = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    taps /= taps.sum()
    padded = np.concatenate([np.repeat(path[:1], radius, 0), path,
                             np.repeat(path[-1:], radius, 0)], 0)
    out = np.empty_like(path)
    for axis in range(path.shape[1]):
        out[:, axis] = np.convolve(padded[:, axis], taps, mode="valid")
    return out


def _clamp_path(positions, clouds, basis, k, distance, half, depth_dead):
    """Move the frames that fall outside *half* / *depth_dead*, and only those.

    Used twice with different bounds.  With the safe box it is the dead zone
    itself.  With the hard box it is the bound that is not a preference -- the
    creature has to be inside the picture and in front of the lens -- and the
    blur is free to overrun the dead zone in between.  The correction is per
    frame and minimal, so a frame already inside contributes no camera motion.
    """
    right, up, view = basis
    low, high = 0.5 - half, 0.5 + half
    out = positions.copy()
    for i, cloud in enumerate(clouds):
        mean_depth = float(((cloud - out[i]) @ view).mean())
        drift = mean_depth / distance - 1.0
        if abs(drift) > depth_dead:
            keep = math.copysign(depth_dead, drift)
            out[i] = out[i] + view * (mean_depth - distance * (1.0 + keep))
        xs, ys, depth = _project(cloud, out[i], basis, k)
        per_ndc = float(depth.mean()) / k
        shifts = []
        for lo_v, hi_v in ((float(xs.min()), float(xs.max())),
                           (float(ys.min()), float(ys.max()))):
            if hi_v - lo_v >= 2.0 * half:
                shifts.append(0.5 - (lo_v + hi_v) * 0.5)   # too wide: centre it
            elif lo_v < low:
                shifts.append(low - lo_v)
            elif hi_v > high:
                shifts.append(high - hi_v)
            else:
                shifts.append(0.0)
        out[i] = out[i] - right * (shifts[0] * per_ndc) - up * (shifts[1] * per_ndc)
    return out


def _smooth_path(positions, clouds, basis, k, distance, sigma):
    """Blur the staircase into a ramp, then put back whatever left the picture.

    The raw solve is a staircase by construction: the camera holds until the dead
    zone is breached and then corrects the whole error in one frame, which at 10
    fps reads as a jump.  Blurring spreads that correction over its neighbours,
    at the cost of letting the creature sit outside the dead zone for a frame or
    two -- the trade this function exists to make.  Blur and clamp are iterated
    because the clamp is itself a step; the last round ends on a clamp, so the
    bound holds exactly.
    """
    if sigma <= 0.0 or len(positions) < 3:
        return positions
    for _ in range(SMOOTH_ITERS):
        positions = _gaussian_smooth(positions, sigma)
        positions = _clamp_path(positions, clouds, basis, k, distance,
                                HARD_HALF, HARD_DEPTH)
    return positions


def _solve_follow(clouds, basis, k, fill_half, safe_half, follow, sigma=0.0):
    """Per-frame camera positions: as close to standing still as the clip allows.

    Solved over the whole clip at once, not frame by frame, and that is the
    point.  The obvious causal rule -- hold until the silhouette leaves the safe
    box, then correct -- is a one-way ratchet: it answers the frame that breached
    and never comes back, so every transient (a rear-up, a death, a raised head)
    permanently displaces the camera.  So instead:

    1. **ideal** -- where the camera would stand to centre *this* frame, at the
       one solved distance.  Being at that distance by construction, this path
       carries the depth too; there is no separate dolly rule.
    2. **desired** -- ideal blurred with ``DESIRE_SIGMA``.  A bob averages out
       and moves nothing; a symmetric kernel reproduces a linear ramp exactly, so
       a constant-velocity gallop is still tracked in full.
    3. **safe box** -- pull back only the frames the box actually rejects.
    4. **smooth to the hard bound** (:func:`_smooth_path`).

    Trucking (translating in the camera's right/up plane) and not panning: the
    view direction stays put, so the creature's size, the ground perspective and
    the run of the grid are identical in every frame and only the squares slide
    underfoot.  With ``follow`` off the camera is solved once over every frame's
    cloud and then never moves -- the fixed-tripod control.
    """
    _, _, view = basis
    framed = [_anchored(cloud) for cloud in clouds]
    # How big the creature is drawn is decided by the creature; the ground band
    # only decides where the frame sits.  Letting the band into the *distance*
    # charges every flier a third of the picture to keep a strip of floor under
    # something that is nowhere near it.
    peak, _ = _solve_distance([np.concatenate(clouds, 0)], basis, k, fill_half)
    hold = np.repeat(
        (_frame_at([np.concatenate(framed, 0)], peak, basis, k)[0]
         - view * peak)[None, :], len(clouds), 0)
    if not follow:
        return hold

    # Does this clip actually travel?  Tracking the *silhouette* is not the same
    # question as tracking the creature: a death that folds a standing body onto
    # the floor and a lunging bite both move their bounding box a long way while
    # going nowhere.  The honest test is what a fixed camera would cost -- and
    # measured on where the creature *ends up*, not on how far it swung, because
    # peak excursion cannot tell a lunge from a departure.  Below HOLD_RATIO the
    # camera is nailed down for the whole clip and the creature is merely drawn a
    # little smaller; that is the trade, and it is the right way round.
    distance, _ = _solve_distance(clouds, basis, k, fill_half)
    edge = max(1, len(clouds) // 5)
    net, _ = _solve_distance(
        [np.concatenate(clouds[:edge] + clouds[-edge:], 0)], basis, k, fill_half)
    if net <= HOLD_RATIO * distance and peak <= PEAK_RATIO * distance:
        return hold

    centres = _frame_at(framed, distance, basis, k)
    ideal = np.array([centre - view * distance for centre in centres])
    path = _gaussian_smooth(ideal, DESIRE_SIGMA)
    path = _clamp_path(path, framed, basis, k, distance, safe_half, DEPTH_DEAD)
    return _smooth_path(path, framed, basis, k, distance, sigma)


# ============================== rendering ===================================
class _quiet_fds:
    """Send OS-level stdout/stderr to the null device for the block.

    The glTF importer logs and the render operator's ``Saved: '<path>'`` are
    written from C on the file descriptors, past anything Python can patch.
    """

    def __enter__(self):
        sys.stdout.flush()
        sys.stderr.flush()
        self._saved = (os.dup(1), os.dup(2))
        self._null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._null, 1)
        os.dup2(self._null, 2)
        return self

    def __exit__(self, *exc):
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self._saved[0], 1)
        os.dup2(self._saved[1], 2)
        os.close(self._null)
        for fd in self._saved:
            os.close(fd)
        return False


def _rescale_to_canonical(bpy, canonical_scale):
    """Scale the imported creature onto the dataset's canonical size.

    ``scale_factor`` is defined against the *armature-local* coordinates, while
    what is in the scene has the glTF node transform already applied (0.01 on
    almost every asset, 1.0 on the Crab), so the node scale is divided back out.
    Only the roots are touched; children ride along on the parent transform.

    Scaling about the world origin rather than the creature is deliberate and
    free: the camera solver frames on the geometry wherever it lands, so the
    picture is identical and only the grid underneath it changes density.
    """
    from mathutils import Matrix

    armature = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    if armature is None or not canonical_scale:
        return
    node = float(sum(abs(v) for v in armature.matrix_world.to_scale())) / 3.0
    if node <= 1e-12:
        return
    factor = canonical_scale / node
    if not 1e-9 < factor < 1e9 or abs(factor - 1.0) < 1e-6:
        return
    for obj in bpy.data.objects:
        if obj.parent is None:
            obj.matrix_world = Matrix.Scale(factor, 4) @ obj.matrix_world


def _load_glb(bpy, path, opts):
    """Fresh scene, GLB imported, lights/cameras of its own dropped.

    The fps is set *before* the import: the glTF importer converts the clip's
    keyframe times from seconds using the scene fps, and every Truebones clip was
    authored at 30, so anything else lands the keys off the frame grid.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = int(round(opts["scene_fps"]))
    scene.render.fps_base = 1.0
    with _quiet_fds():
        bpy.ops.import_scene.gltf(filepath=path)
    for obj in list(bpy.data.objects):
        if obj.type in {"LIGHT", "CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    _rescale_to_canonical(bpy, opts.get("canonical_scale"))

    meshes = _visible_meshes(bpy)          # before the ground is added
    _normalize_materials(bpy)              # before the ground has a material
    resolution = opts["size"] * opts["supersample"]
    scene = _setup_scene(bpy, resolution, opts["engine"], TAA_RENDER_SAMPLES)
    sun = _add_sun(bpy, scene)
    _add_ground(bpy, scene, opts["checker_size"])
    armature = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    return scene, armature, meshes, sun


def _clouds_for(bpy, scene, meshes, armature, times):
    clouds = []
    for sample_time in times:
        cloud = _frame_cloud(bpy, scene, meshes, sample_time) if meshes \
            else np.empty((0, 3))
        if cloud.size == 0 and armature is not None:
            cloud = _joint_cloud(bpy, scene, armature, sample_time)
        if cloud.size == 0:
            raise RuntimeError("nothing renderable to frame the camera on")
        clouds.append(cloud)
    return clouds


def _render_frames(bpy, glb_path, frame_dir, opts):
    """Render one clip into ``frame_dir/NNNNN.png``.

    Returns ``(frame paths, per-frame delays in ms, source frame count)``.
    """
    from mathutils import Vector

    scene, armature, meshes, sun = _load_glb(bpy, glb_path, opts)
    if armature is None:
        raise RuntimeError("no armature in %s" % os.path.basename(glb_path))
    azimuth = CLIP_AZIMUTH_DEG + opts["facing_yaw_deg"]
    _aim_sun(sun, azimuth)

    src_times = _sample_times(armature)
    source_fps = _source_fps(src_times, scene.render.fps)
    stride = max(1, int(round(source_fps / max(opts["target_fps"], 1e-6))))
    times = _pick_times(src_times, stride, opts["max_frames"])

    basis = _camera_basis(azimuth, ELEVATION_DEG)
    cam = _add_camera(bpy, scene, basis)
    k = CAMERA_LENS / CAMERA_SENSOR
    clouds = _clouds_for(bpy, scene, meshes, armature, times)
    positions = _solve_follow(clouds, basis, k, FILL_HALF, SAFE_HALF,
                              opts["follow"], opts["smooth_sigma"])

    os.makedirs(frame_dir, exist_ok=True)
    paths = []
    with _quiet_fds():
        for index, (sample_time, position) in enumerate(zip(times, positions), 1):
            _set_time(scene, sample_time)
            cam.location = Vector(position.tolist())   # per frame, no keyframes
            frame_path = os.path.join(frame_dir, "%05d.png" % index)
            scene.render.filepath = frame_path
            bpy.ops.render.render(write_still=True)
            paths.append(frame_path)
    return paths, _durations_ms(times, source_fps), len(src_times)


def _write_gif(frame_paths, gif_path, opts):
    """Quantize the rendered frames onto one palette and pack them into a GIF.

    One palette for the whole clip, not one per frame: a per-frame palette makes
    the sky and the floor shimmer between frames (the quantizer re-picks its
    colours as the creature moves through the picture), and every frame then has
    to carry its own colour table.  The palette is cut from all the frames at
    once, so a colour that only appears in the last frame is still represented.
    """
    from PIL import Image

    size = opts["size"]
    frames = []
    for path in frame_paths:
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        if image.size != (size, size):
            image = image.resize((size, size), Image.LANCZOS)
        frames.append(image)
    if not frames:
        raise RuntimeError("no frames rendered")

    strip = Image.new("RGB", (size, size * len(frames)))
    for index, image in enumerate(frames):
        strip.paste(image, (0, index * size))
    palette = strip.quantize(colors=opts["colors"], method=Image.Quantize.MEDIANCUT)

    dither = Image.Dither.FLOYDSTEINBERG if opts["dither"] == "fs" \
        else Image.Dither.NONE
    packed = [image.quantize(palette=palette, dither=dither) for image in frames]
    os.makedirs(os.path.dirname(gif_path) or ".", exist_ok=True)
    packed[0].save(gif_path, save_all=True, append_images=packed[1:],
                   duration=opts["durations"], loop=0, optimize=True, disposal=1)
    return os.path.getsize(gif_path)


# ============================= worker / jobs ================================
def _run_job(job):
    """Render one clip and pack its GIF.

    Runs in a pooled worker process; ``import bpy`` is amortised because the pool
    reuses its processes across many jobs.  The frames are scratch -- the GIF is
    the deliverable -- so they go to a temp directory that is removed on the way
    out unless ``--frames-dir`` asked to keep them.
    """
    import bpy   # heavy; kept in the worker and cached after the first job

    clip, species, glb_path, gif_path, opts = job
    keep = opts.get("frames_dir")
    frame_dir = os.path.join(keep, clip) if keep else tempfile.mkdtemp(prefix="tbgif_")
    try:
        paths, durations, src_frames = _render_frames(bpy, glb_path, frame_dir, opts)
        size = _write_gif(paths, gif_path, dict(opts, durations=durations))
        return {"clip": clip, "species": species, "frames": len(paths),
                "src_frames": src_frames, "bytes": size, "error": None}
    except Exception as exc:    # noqa: BLE001 -- one bad clip must not end the run
        return {"clip": clip, "species": species, "frames": 0, "src_frames": 0,
                "bytes": 0, "error": "%s: %s\n%s" % (type(exc).__name__, exc,
                                                     traceback.format_exc())}
    finally:
        if not keep:
            shutil.rmtree(frame_dir, ignore_errors=True)


# --- worker stdout plumbing (mirrors render_for_llm.py) ---------------------
# bpy's exit audit ("Not freed memory blocks", a handful of bytes Blender leaks
# on a few files) is written on the workers' file descriptors at process exit,
# and a line reading "Error:" in an otherwise clean run costs a second look every
# time.  Each worker stream gets a private pipe the parent drains and filters;
# the parent keeps writing to the real streams, because on Windows its
# console-backed sys.stdout raises WinError 1 the moment its fd is a pipe.
_AUDIT_NOISE = "Not freed memory blocks"


def _drain_worker_output(read_fd, real_fd):
    import threading

    def _pump():
        pending = ""
        try:
            while True:
                chunk = os.read(read_fd, 4096)
                if not chunk:
                    break
                lines = (pending + chunk.decode("utf-8", "replace")).split("\n")
                pending = lines.pop()
                for line in lines:
                    if _AUDIT_NOISE in line:
                        continue
                    os.write(real_fd, (line + "\n").encode("utf-8", "replace"))
            if pending and _AUDIT_NOISE not in pending:
                os.write(real_fd, pending.encode("utf-8", "replace"))
        except (OSError, ValueError):
            pass
        finally:
            try:
                os.close(read_fd)
            except OSError:
                pass

    thread = threading.Thread(target=_pump, daemon=True)
    thread.start()
    return thread


def _filter_fd(fd, stream):
    original = getattr(sys, stream)
    original.flush()
    saved = os.dup(fd)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, fd)
    os.close(write_fd)
    pump = _drain_worker_output(read_fd, saved)
    direct = io.TextIOWrapper(io.FileIO(saved, "w", closefd=False),
                              encoding="utf-8", errors="replace",
                              line_buffering=True)
    setattr(sys, stream, direct)

    def restore():
        direct.flush()
        setattr(sys, stream, original)
        os.dup2(saved, fd)
        pump.join()
        os.close(saved)

    return restore


def _filtered_worker_output():
    put_back = [_filter_fd(1, "stdout"), _filter_fd(2, "stderr")]

    def restore():
        for restore_one in put_back:
            restore_one()

    return restore


# ================================== CLI =====================================
def main():
    ap = argparse.ArgumentParser(
        description="Render every processed Truebones clip to a review GIF.")
    ap.add_argument("--filter", default="*",
                    help="species glob, e.g. 'Dog*' (default: %(default)s)")
    ap.add_argument("--clip", default="*",
                    help="glob on the clip name, e.g. 'Horse_Walk*'")
    ap.add_argument("--raw-root", default=RAW_ROOT)
    ap.add_argument("--gif-root", default=GIF_ROOT)
    ap.add_argument("--metadata", default=METADATA_PATH,
                    help="motion_metadata.json: the clip -> source GLB mapping")
    ap.add_argument("--cond", default=COND_PATH,
                    help="cond.npy, read for the per-species facing correction")
    ap.add_argument("--workers", "-j", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="limit scheduled jobs")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-render everything selected (default: update mode)")
    ap.add_argument("--engine", default="eevee", choices=("eevee", "cycles"))
    ap.add_argument("--size", type=int, default=GIF_SIZE,
                    help="GIF edge in pixels (default: %(default)s)")
    ap.add_argument("--supersample", type=int, default=SUPERSAMPLE,
                    help="render at size*this and downsample (default: %(default)s)")
    ap.add_argument("--colors", type=int, default=GIF_COLORS,
                    help="palette size, 2..256 (default: %(default)s)")
    ap.add_argument("--dither", default="fs", choices=("fs", "none"),
                    help="Floyd-Steinberg keeps the sky and the floor smooth; "
                         "'none' packs smaller because unchanged pixels stay "
                         "identical between frames (default: %(default)s)")
    ap.add_argument("--checker-size", type=float, default=GRID_CELL_SIZE,
                    help="ground cell size in world units (default: %(default)s)")
    ap.add_argument("--follow", default="on", choices=("on", "off"),
                    help="'off' pins the camera for the whole clip (control)")
    ap.add_argument("--smooth-sigma", type=float, default=SMOOTH_SIGMA,
                    help="frames of zero-phase blur on the camera path; "
                         "0 disables (default: %(default)s)")
    ap.add_argument("--fps", type=float, default=TARGET_FPS,
                    help="frames sampled per second of clip (default: %(default)s)")
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    ap.add_argument("--no-facing", dest="facing", action="store_false",
                    help="ignore cond.npy's orientation_quat; most species then "
                         "render side-on or from behind")
    ap.add_argument("--no-canonical-scale", dest="canonical_scale",
                    action="store_false",
                    help="import the GLBs at their own scale; the grid cell then "
                         "means a different fraction of a body per species, and "
                         "the Crab loses its grid entirely")
    ap.add_argument("--frames-dir", default=None,
                    help="keep the rendered PNG frames under this directory "
                         "(default: a temp dir, removed per clip)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be rendered and stop")
    args = ap.parse_args()

    if not 2 <= args.colors <= 256:
        raise SystemExit("--colors must be between 2 and 256")
    if args.supersample < 1:
        raise SystemExit("--supersample must be at least 1")
    if not os.path.isfile(args.metadata):
        raise SystemExit("no clip index: %s" % args.metadata)
    if not os.path.isdir(args.raw_root):
        raise SystemExit("raw tree not found: %s" % args.raw_root)

    clips = _load_clip_sources(args.metadata)
    if not clips:
        raise SystemExit("%s lists no clips" % args.metadata)
    facing = _load_facing(args.cond) if args.facing else {}
    scales = _load_scales(args.cond) if args.canonical_scale else {}
    jobs, skipped = _plan(args, clips, facing, scales)
    if args.limit:
        jobs = jobs[:args.limit]

    if args.dry_run:
        for clip, _species, source, _gif, opts in jobs:
            print("%-42s yaw %+7.1f  %s"
                  % (clip, opts["facing_yaw_deg"], os.path.basename(source)))
        print("\n%d job(s), %d already rendered, %d clip(s) indexed"
              % (len(jobs), skipped, len(clips)))
        return 0
    if not jobs:
        print("[OK] nothing to do (%d already rendered)" % skipped)
        return 0

    os.makedirs(args.gif_root, exist_ok=True)
    if args.frames_dir:
        os.makedirs(args.frames_dir, exist_ok=True)
    workers = max(1, min(args.workers, len(jobs)))
    print("Rendering %d clip(s) with %d worker(s) -> %s"
          % (len(jobs), workers, args.gif_root), flush=True)
    print("engine %s  %dpx (x%d)  %d colors  dither %s  %.0f fps  <=%d frames  "
          "follow %s" % (args.engine, args.size, args.supersample, args.colors,
                         args.dither, args.fps, args.max_frames, args.follow),
          flush=True)

    started = time.time()
    ok = fail = frames = written = 0
    failures = []

    def _report(i, result):
        nonlocal ok, fail, frames, written
        if result["error"]:
            fail += 1
            failures.append((result["clip"], result["error"]))
            print("  [FAIL] [%d/%d] %s -- %s"
                  % (i, len(jobs), result["clip"],
                     result["error"].splitlines()[0]), flush=True)
            return
        ok += 1
        frames += result["frames"]
        written += result["bytes"]
        print("  [OK]   [%d/%d] %s -- %d/%d frame(s), %.0f KB"
              % (i, len(jobs), result["clip"], result["frames"],
                 result["src_frames"], result["bytes"] / 1024.0), flush=True)

    if workers == 1:
        for i, job in enumerate(jobs, 1):
            _report(i, _run_job(job))
    else:
        context = multiprocessing.get_context("spawn")
        restore_output = _filtered_worker_output()
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                futures = {pool.submit(_run_job, job): job for job in jobs}
                for i, future in enumerate(as_completed(futures), 1):
                    job = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:      # noqa: BLE001
                        result = {"clip": job[0], "species": job[1], "frames": 0,
                                  "src_frames": 0, "bytes": 0,
                                  "error": "worker crashed: %s: %s"
                                           % (type(exc).__name__, exc)}
                    _report(i, result)
        finally:
            restore_output()

    print("\ndone: %d ok, %d failed, %d skipped, %d frames, %.1f MB in %.0fs"
          % (ok, fail, skipped, frames, written / 1048576.0, time.time() - started))
    for clip, error in failures:
        print("\n[FAIL] %s\n%s" % (clip, error))
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
