"""
Auto texture resolution for restored skinned meshes (main character only).
============================================================================

Truebones FBX rigs frequently embed texture paths that do not exist on this
machine (they were authored elsewhere). The textures actually ship alongside
the mesh, under ``<character>/tex/`` with a loose naming convention inferred
from the Truebone_Z-OO dataset:

    *_D_*  / *diffuse* / *albedo*     -> base color (diffuse)
    *_N_*  / *_n*    / *normal*       -> normal map        (never diffuse)
    *_SP_* / *spec*                   -> specular/gloss    (never diffuse)
    *_C_*  / *_C2_* / *alpha* / *_a*  -> alpha / opacity (cutout mask)
    *color* / *colour* / *col*        -> color  (secondary diffuse candidate)
    <plain name> (e.g. ``bear.JPG``)  -> neutral diffuse candidate

In the Truebone_Z-OO naming scheme the per-character ``_C_`` / ``_C2_`` map
is the cutout/coverage (alpha) mask that ships beside the ``_D_`` diffuse —
e.g. ``CH_NPC_MOB_Camel_A01_C2_YNG.jpg`` alongside ``..._Camel_A01_D_YNG.jpg``
(same trailing artist code) — not a secondary colour image, so the single
letter ``c`` / ``c2`` codes classify as alpha.

:func:`resolve_main_character_textures` operates only on meshes bound to the
armature (never a helper primitive like Icosphere/Cube). For each such mesh it:

  * wires a matching diffuse texture onto the Principled BSDF base color when
    the mesh has no material, or its texture is empty / points at a missing
    file (the importer's own ``use_image_search`` is expected to handle the
    common case first; this is the fallback for stale/renamed paths), and
  * connects a clearly-named alpha/opacity texture to the BSDF alpha channel
    whenever one exists in the ``tex/`` folder and is not already connected —
    independent of whether the diffuse needed fixing. A name match alone is not
    enough: the candidate's pixels are read and it is accepted only if it is a
    near-binary mask (mostly black/white). A file misnamed ``*_alpha*`` that is
    really a colour/grayscale image (too many mid-tone pixels) is rejected.

Best-effort: every failure is swallowed so export never breaks over textures.
"""
from __future__ import annotations

import os
import re

import numpy as np

__all__ = ["resolve_main_character_textures"]

# A candidate alpha texture is rejected when more than this fraction of its
# pixels are mid-tone (neither near-black nor near-white) — a sign it is a
# regular image rather than a coverage mask. A genuine cutout mask is sharply
# binary: across the Truebone_Z-OO dataset every real ``*_a*`` mask measures
# <=0.03 mid-tone, while desaturated body/AO maps misnamed ``*_a*`` (e.g.
# Deer/elkbody_a, which is ~84 % black and would make the mesh transparent)
# sit at >=0.10. 0.05 splits the two with headroom.
_ALPHA_MID_TONE_FRACTION_LIMIT = 0.05
# Pixels with gray value strictly inside this band count as "mid-tone".
_ALPHA_MID_TONE_LO = 0.15
_ALPHA_MID_TONE_HI = 0.85


_TEXTURE_EXTS = {".png", ".jpg", ".jpeg", ".tga", ".bmp", ".tif", ".tiff", ".dds"}

# Object names that are scene helpers / primitives, never the skinned character.
_EXCLUDED_MESH_PREFIXES = (
    "icosphere", "sphere", "cube", "plane", "cylinder", "cone", "torus",
    "grid", "circle", "suzanne", "empty",
)

# Token sets used to classify a texture file from its (underscore/dot
# separated) name tokens. A token like ``_N_`` shows up as a standalone "n".
_NORMAL_TOKENS = {"normal", "norm", "nrm", "n", "nm"}
_SPEC_TOKENS = {"spec", "specular", "sp", "s", "gloss", "glossiness"}
_ROUGH_METAL_TOKENS = {"rough", "roughness", "rgh", "metal", "metallic", "metalness", "mtl"}
_MISC_NON_DIFFUSE_TOKENS = {
    "ao", "occlusion", "emiss", "emissive", "emission", "glow",
    "disp", "displacement", "height", "bump", "cavity", "id",
}
# ``c`` / ``c2`` are the Truebone_Z-OO single-letter codes for the cutout /
# coverage (alpha) mask — see module docstring.
_ALPHA_TOKENS = {"alpha", "opacity", "transparency", "a", "c", "c2"}
# Strong / weak positive diffuse signals (checked separately for scoring).
_DIFFUSE_STRONG_TOKENS = {"diffuse", "diff", "albedo", "basecolor", "bc", "dif", "d"}
_COLOR_TOKENS = {"color", "colour", "col"}


# ── Texture file discovery / classification ─────────────────────────────────

def _texture_tokens(stem: str) -> set[str]:
    return {tok for tok in re.split(r"[^a-z0-9]+", stem.lower()) if tok}


def _texture_kind(filename: str) -> str:
    """Classify a texture file as one of: normal/spec/rough_metal/misc/alpha/
    diffuse/color/neutral.  Non-diffuse maps are detected first so they are
    never mistaken for a base-color texture."""
    tokens = _texture_tokens(os.path.splitext(filename)[0])
    if tokens & _NORMAL_TOKENS:
        return "normal"
    if tokens & _SPEC_TOKENS:
        return "spec"
    if tokens & _ROUGH_METAL_TOKENS:
        return "rough_metal"
    if tokens & _MISC_NON_DIFFUSE_TOKENS:
        return "misc"
    if tokens & _ALPHA_TOKENS:
        return "alpha"
    if tokens & _DIFFUSE_STRONG_TOKENS:
        return "diffuse"
    if tokens & _COLOR_TOKENS:
        return "color"
    return "neutral"


def _discover_texture_files(mesh_path: str) -> list[str]:
    """Collect candidate texture files near the mesh source.

    Looks in the mesh's own directory and a ``tex``/``textures`` subfolder
    (the layout used throughout the Truebone_Z-OO dataset).
    """
    base_dir = os.path.dirname(os.path.abspath(mesh_path))
    search_dirs = [
        base_dir,
        os.path.join(base_dir, "tex"),
        os.path.join(base_dir, "Tex"),
        os.path.join(base_dir, "textures"),
        os.path.join(base_dir, "Textures"),
    ]
    found: list[str] = []
    seen: set[str] = set()
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            full = os.path.join(d, name)
            if not os.path.isfile(full):
                continue
            if os.path.splitext(name)[1].lower() not in _TEXTURE_EXTS:
                continue
            key = os.path.normcase(full)
            if key not in seen:
                seen.add(key)
                found.append(full)
    return found


def _select_diffuse_texture(
    tex_files: list[str], char_name: str, name_hint: str
) -> str | None:
    """Pick the best base-color texture among *tex_files*.

    Scoring prefers explicit diffuse tokens, then color tokens, then plain
    names, and rewards files whose name references the character (folder name)
    or the mesh/material (*name_hint*).
    """
    char = char_name.lower()
    hint_tokens = _texture_tokens(name_hint)
    best: tuple[float, str] | None = None
    for path in tex_files:
        stem = os.path.splitext(os.path.basename(path))[0]
        kind = _texture_kind(os.path.basename(path))
        if kind in ("normal", "spec", "rough_metal", "misc", "alpha"):
            continue
        if kind == "diffuse":
            score = 50.0
        elif kind == "color":
            score = 25.0
        else:  # neutral
            score = 10.0
        stem_lower = stem.lower()
        if char and char in stem_lower:
            score += 15.0
        if hint_tokens & _texture_tokens(stem):
            score += 8.0
        score -= 0.01 * len(stem_lower)  # tie-break toward shorter names
        if best is None or score > best[0]:
            best = (score, path)
    return best[1] if best is not None else None


def _select_alpha_texture(tex_files: list[str], char_name: str) -> str | None:
    """Pick an alpha/opacity texture if one is clearly present."""
    char = char_name.lower()
    best: tuple[float, str] | None = None
    for path in tex_files:
        if _texture_kind(os.path.basename(path)) != "alpha":
            continue
        stem = os.path.splitext(os.path.basename(path))[0].lower()
        score = 0.0
        if char and char in stem:
            score += 15.0
        score -= 0.01 * len(stem)
        if best is None or score > best[0]:
            best = (score, path)
    return best[1] if best is not None else None


def _is_alpha_mask_image(bpy, path: str) -> bool:
    """Confirm a candidate alpha texture really is a coverage mask by its pixels.

    A genuine alpha/opacity mask is near-binary: almost every pixel is black or
    white. The image is read (and subsampled for speed); if more than
    :data:`_ALPHA_MID_TONE_FRACTION_LIMIT` of the pixels are mid-tone, the file
    is treated as a regular image that merely happens to be named ``*_alpha*``
    and is rejected. Conservative: any read failure also rejects, so a doubtful
    file never makes the mesh transparent.
    """
    try:
        image = bpy.data.images.load(path, check_existing=True)
        width, height = image.size
        pixel_count = width * height
        if pixel_count == 0:
            return False
        buffer = np.empty(pixel_count * 4, dtype=np.float32)
        image.pixels.foreach_get(buffer)
        rgba = buffer.reshape(-1, 4)
        # Stride-subsample (no interpolation, so binary edges stay binary).
        stride = max(1, rgba.shape[0] // 50000)
        gray = rgba[::stride, :3].mean(axis=1)
        mid = np.count_nonzero(
            (gray > _ALPHA_MID_TONE_LO) & (gray < _ALPHA_MID_TONE_HI)
        )
        return (mid / gray.shape[0]) <= _ALPHA_MID_TONE_FRACTION_LIMIT
    except Exception:  # noqa: BLE001 — unreadable -> not a confirmed mask
        return False


# ── Blender material inspection / wiring ────────────────────────────────────

def _image_is_valid(bpy, image) -> bool:
    """True if *image* has usable pixel data (packed, on-disk, or loaded)."""
    if image is None:
        return False
    if getattr(image, "packed_file", None) is not None:
        return True
    filepath = image.filepath_raw or image.filepath
    if filepath:
        abspath = bpy.path.abspath(filepath)
        if abspath and os.path.isfile(abspath):
            return True
    try:
        if image.has_data and tuple(image.size) != (0, 0):
            return True
    except (RuntimeError, AttributeError):
        pass
    return False


def _mesh_has_valid_diffuse(bpy, mesh) -> bool:
    """True if any material slot already carries a usable image texture."""
    if not mesh.data.materials:
        return False
    for mat in mesh.data.materials:
        if mat is None or not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type == "TEX_IMAGE" and _image_is_valid(bpy, node.image):
                return True
    return False


def _mesh_has_alpha_connected(mesh) -> bool:
    """True if any Principled BSDF on the mesh already has its Alpha input linked."""
    for mat in mesh.data.materials:
        if mat is None or not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type != "BSDF_PRINCIPLED":
                continue
            alpha_input = node.inputs.get("Alpha")
            if alpha_input is not None and alpha_input.is_linked:
                return True
    return False


def _armature_bound_meshes(bpy, armature) -> list:
    """Meshes skinned to / parented under *armature* — the main character."""
    meshes = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if obj.parent == armature or any(
            mod.type == "ARMATURE" and mod.object == armature for mod in obj.modifiers
        ):
            meshes.append(obj)
    return meshes


def _is_excluded_primitive(name: str) -> bool:
    base = name.lower().lstrip("_")
    return base.startswith(_EXCLUDED_MESH_PREFIXES)


def _load_texture_image(bpy, path: str, *, non_color: bool):
    """Load (or reuse) an image and tag its colorspace."""
    image = bpy.data.images.load(path, check_existing=True)
    try:
        image.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
    except (RuntimeError, TypeError):
        pass
    return image


def _get_or_create_material_bsdf(bpy, mesh):
    """Return the (material, principled BSDF) for *mesh*, creating either if
    absent so a texture can be wired on."""
    if mesh.data.materials and mesh.data.materials[0] is not None:
        mat = mesh.data.materials[0]
    else:
        mat = bpy.data.materials.new(name=f"{mesh.name}_mat")
        if mesh.data.materials:
            mesh.data.materials[0] = mat
        else:
            mesh.data.materials.append(mat)
    mat.use_nodes = True
    node_tree = mat.node_tree

    bsdf = next((n for n in node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        bsdf = node_tree.nodes.new("ShaderNodeBsdfPrincipled")
        output = next(
            (n for n in node_tree.nodes if n.type == "OUTPUT_MATERIAL"), None
        ) or node_tree.nodes.new("ShaderNodeOutputMaterial")
        node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return mat, bsdf


def _connect_diffuse(bpy, mat, bsdf, diffuse_path: str) -> None:
    node_tree = mat.node_tree
    diffuse_node = node_tree.nodes.new("ShaderNodeTexImage")
    diffuse_node.image = _load_texture_image(bpy, diffuse_path, non_color=False)
    node_tree.links.new(diffuse_node.outputs["Color"], bsdf.inputs["Base Color"])


def _connect_alpha(bpy, mat, bsdf, alpha_path: str) -> None:
    if "Alpha" not in bsdf.inputs:
        return
    node_tree = mat.node_tree
    alpha_node = node_tree.nodes.new("ShaderNodeTexImage")
    alpha_node.image = _load_texture_image(bpy, alpha_path, non_color=True)
    node_tree.links.new(alpha_node.outputs["Color"], bsdf.inputs["Alpha"])
    if hasattr(mat, "blend_method"):  # removed in Blender 4.2+
        mat.blend_method = "HASHED"


# ── Public entry point ──────────────────────────────────────────────────────

def resolve_main_character_textures(bpy, armature, mesh_path: str) -> None:
    """Resolve missing diffuse / alpha textures on the skinned main character.

    Only meshes bound to *armature* are touched (helper primitives such as
    Icosphere/Cube are skipped). A diffuse is wired on only when the mesh lacks
    a usable one; an alpha map is wired on whenever a clearly-named alpha
    texture exists and is not already connected. Best-effort: every failure is
    swallowed so export never breaks over this.
    """
    try:
        meshes = [
            m for m in _armature_bound_meshes(bpy, armature)
            if not _is_excluded_primitive(m.name)
        ]
        if not meshes:
            return
        tex_files = _discover_texture_files(mesh_path)
        if not tex_files:
            return

        char_name = os.path.basename(os.path.dirname(os.path.abspath(mesh_path)))
        alpha_path = _select_alpha_texture(tex_files, char_name)
        # Name match is not enough: confirm the candidate is a real near-binary
        # mask before using it, so a misnamed colour image never wires alpha.
        if alpha_path is not None and not _is_alpha_mask_image(bpy, alpha_path):
            alpha_path = None

        for mesh in meshes:
            need_diffuse = not _mesh_has_valid_diffuse(bpy, mesh)
            diffuse_path = (
                _select_diffuse_texture(tex_files, char_name, mesh.name)
                if need_diffuse else None
            )
            need_alpha = alpha_path is not None and not _mesh_has_alpha_connected(mesh)
            if diffuse_path is None and not need_alpha:
                continue

            mat, bsdf = _get_or_create_material_bsdf(bpy, mesh)
            applied = []
            if diffuse_path is not None:
                _connect_diffuse(bpy, mat, bsdf, diffuse_path)
                applied.append(f"diffuse={os.path.basename(diffuse_path)}")
            if need_alpha:
                _connect_alpha(bpy, mat, bsdf, alpha_path)
                applied.append(f"alpha={os.path.basename(alpha_path)}")
            if applied:
                print(f"[texture] {mesh.name}: {', '.join(applied)}")
    except Exception:  # noqa: BLE001 — texture fix is best-effort, never fatal
        pass
