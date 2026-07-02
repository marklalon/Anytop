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
  * composes a matching diffuse + alpha/opacity texture into a temporary RGBA
    WebP texture whenever one exists in the ``tex/`` folder and is not already
    connected, then wires that WebP's alpha channel to the BSDF alpha input so
    glTF exports ``alphaMode=BLEND``. A name match alone is not enough: the
    candidate's pixels are read and it is accepted only if it is a near-binary
    mask (mostly black/white). A file misnamed ``*_alpha*`` that is really a
    colour/grayscale image (too many mid-tone pixels) is rejected.

"""
from __future__ import annotations

import atexit
import hashlib
import os
import re
import shutil
import tempfile

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
_GENERIC_MATCH_TOKENS = {
    "ch", "npc", "mob", "mesh", "mat", "mi", "tex", "texture", "map",
}
_TRACEABLE_IMAGE_NODES = {
    "MIX", "MIX_RGB", "RGB_CURVE", "CURVE_RGB", "GAMMA",
    "BRIGHTCONTRAST", "HUE_SAT", "SEPHSV", "COMBHSV",
    "SEPRGB", "COMBRGB", "MATH", "MAPPING",
}


# ── Texture file discovery / classification ─────────────────────────────────

def _texture_token_list(stem: str) -> list[str]:
    return [tok for tok in re.split(r"[^a-z0-9]+", stem.lower()) if tok]


def _texture_tokens(stem: str) -> set[str]:
    return set(_texture_token_list(stem))


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


def _normalized_stem(path_or_name: str) -> str:
    return os.path.splitext(os.path.basename(path_or_name))[0].lower()


def _alpha_replacement_stems(diffuse_stem: str) -> list[str]:
    """Return likely alpha stems for a matching diffuse/base-color stem."""
    replacements = ("c2", "c", "a", "alpha", "opacity")
    pattern = re.compile(
        r"(^|[^a-z0-9])("
        + "|".join(sorted(map(re.escape, _DIFFUSE_STRONG_TOKENS), key=len, reverse=True))
        + r")(?=$|[^a-z0-9])",
        re.IGNORECASE,
    )
    stems: list[str] = []
    seen: set[str] = set()
    for repl in replacements:
        candidate = pattern.sub(lambda m: f"{m.group(1)}{repl}", diffuse_stem)
        key = candidate.lower()
        if key != diffuse_stem.lower() and key not in seen:
            seen.add(key)
            stems.append(key)
    return stems


_NON_ALPHA_MATCH_TOKENS = (
    _ALPHA_TOKENS
    | _DIFFUSE_STRONG_TOKENS
    | _COLOR_TOKENS
    | _NORMAL_TOKENS
    | _SPEC_TOKENS
    | _ROUGH_METAL_TOKENS
    | _MISC_NON_DIFFUSE_TOKENS
    | _GENERIC_MATCH_TOKENS
)


def _alpha_match_tokens(stem: str) -> set[str]:
    return _texture_tokens(stem) - _NON_ALPHA_MATCH_TOKENS


def _alpha_match_tokens_without_trailing_code(stem: str) -> set[str]:
    """Return identity tokens while ignoring a final author/batch code.

    Truebone names commonly end with short artist tags after the texture-kind
    token (``..._D_BYN``, ``..._C2_KSW``). Some valid alpha pairs disagree on
    that final tag, so the relaxed matcher removes it while keeping variant
    tokens such as ``A01`` / ``B01``.
    """
    tokens = _texture_token_list(stem)
    kind_tokens = _ALPHA_TOKENS | _DIFFUSE_STRONG_TOKENS | _COLOR_TOKENS
    if len(tokens) >= 2 and tokens[-2] in kind_tokens:
        tokens = tokens[:-1]
    return set(tokens) - _NON_ALPHA_MATCH_TOKENS


def _variant_tokens(tokens: set[str]) -> set[str]:
    """Return standalone variant tags such as ``a01`` / ``b01``."""
    return {tok for tok in tokens if re.match(r"^[a-z]\d{2,}$", tok)}


def _has_variant_conflict(hint_tokens: set[str], candidate_tokens: set[str]) -> bool:
    hint_variants = _variant_tokens(hint_tokens)
    candidate_variants = _variant_tokens(candidate_tokens)
    return bool(
        hint_variants
        and candidate_variants
        and not (hint_variants & candidate_variants)
    )


def _select_alpha_texture(tex_files: list[str], diffuse_hint: str) -> str | None:
    """Pick the alpha/opacity texture paired with *diffuse_hint*.

    The match is intentionally strict: folders such as ``Bird/tex`` can contain
    multiple variants/species, so a generic character-folder match is unsafe.
    """
    if not diffuse_hint:
        return None

    alpha_files = [
        path for path in tex_files
        if _texture_kind(os.path.basename(path)) == "alpha"
    ]
    if not alpha_files:
        return None

    hint_stem = _normalized_stem(diffuse_hint)
    by_stem = {_normalized_stem(path): path for path in alpha_files}
    for candidate_stem in _alpha_replacement_stems(hint_stem):
        path = by_stem.get(candidate_stem)
        if path is not None:
            return path

    hint_tokens = _alpha_match_tokens(hint_stem)
    if not hint_tokens:
        return None

    best: tuple[float, str] | None = None
    for path in alpha_files:
        stem = _normalized_stem(path)
        tokens = _alpha_match_tokens(stem)
        if _has_variant_conflict(hint_tokens, tokens):
            continue
        overlap = hint_tokens & tokens
        min_overlap = min(2, len(hint_tokens))
        if len(overlap) < min_overlap:
            continue
        union = hint_tokens | tokens
        score = 100.0 * len(overlap) + 10.0 * (len(overlap) / max(1, len(union)))
        score -= 0.01 * len(stem)
        if best is None or score > best[0]:
            best = (score, path)
    if best is not None:
        return best[1]

    # Relaxed fallback for pairs whose final author/batch code differs, e.g.
    # ``Skunk_D_BYN`` with ``Skunk_C2_KSW``. Keep it ambiguity-aware: when the
    # relaxed identity is only one token, accept only a single matching alpha.
    hint_tokens = _alpha_match_tokens_without_trailing_code(hint_stem)
    if not hint_tokens:
        return None

    candidates: list[tuple[float, str]] = []
    for path in alpha_files:
        stem = _normalized_stem(path)
        tokens = _alpha_match_tokens_without_trailing_code(stem)
        if _has_variant_conflict(hint_tokens, tokens):
            continue
        overlap = hint_tokens & tokens
        min_overlap = min(2, len(hint_tokens))
        if len(overlap) < min_overlap:
            continue
        union = hint_tokens | tokens
        score = 100.0 * len(overlap) + 10.0 * (len(overlap) / max(1, len(union)))
        score -= 0.01 * len(stem)
        candidates.append((score, path))

    if not candidates:
        return None
    if min(2, len(hint_tokens)) == 1 and len(candidates) != 1:
        return None
    return max(candidates, key=lambda item: item[0])[1]


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


def _iter_material_bsdfs(mesh):
    """Yield existing ``(material, Principled BSDF)`` pairs on *mesh*."""
    for mat in mesh.data.materials:
        if mat is None or not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                yield mat, node


def _trace_to_image(node, visited: set | None = None):
    """Walk backwards from *node* through intermediate shader nodes to find a TEX_IMAGE."""
    if visited is None:
        visited = set()
    if node in visited or len(visited) > 20:
        return None
    visited.add(node)

    if node.type == "TEX_IMAGE" and node.image is not None:
        return node.image

    if node.type in _TRACEABLE_IMAGE_NODES:
        for inp in node.inputs:
            if inp.is_linked:
                for link in inp.links:
                    result = _trace_to_image(link.from_node, visited)
                    if result is not None:
                        return result
    return None


def _base_color_image(bsdf):
    """Return the image wired to a BSDF Base Color input, if any."""
    base_color = bsdf.inputs.get("Base Color")
    if base_color is None or not base_color.is_linked:
        return None
    for link in base_color.links:
        result = _trace_to_image(link.from_node)
        if result is not None:
            return result
    return None


def _image_path_or_name(bpy, image) -> str | None:
    if image is None:
        return None
    filepath = image.filepath_raw or image.filepath
    if filepath:
        abspath = bpy.path.abspath(filepath)
        if abspath:
            return abspath
    return image.name


def _base_color_image_hint(bpy, bsdf) -> str | None:
    """Return the image path/name wired to a BSDF Base Color input, if any."""
    return _image_path_or_name(bpy, _base_color_image(bsdf))


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


def _texture_resolve_temp_dir() -> str:
    """Return the PID-scoped temporary directory for composited WebP files."""
    return os.path.join(tempfile.gettempdir(), "anytop_texture_resolve", str(os.getpid()))


def _cleanup_temp_dir() -> None:
    """Remove the PID-scoped temp directory, if present."""
    tmp = _texture_resolve_temp_dir()
    if os.path.isdir(tmp):
        shutil.rmtree(tmp, ignore_errors=True)


atexit.register(_cleanup_temp_dir)


def _compose_alpha_webp(diffuse_path: str, alpha_path: str) -> str | None:
    """Create an RGBA WebP base-color texture from diffuse RGB + alpha mask."""
    if not os.path.isfile(diffuse_path) or not os.path.isfile(alpha_path):
        return None
    diffuse_abspath = os.path.abspath(diffuse_path)
    alpha_abspath = os.path.abspath(alpha_path)
    digest = hashlib.sha1(
        f"{diffuse_abspath}\0{alpha_abspath}".encode("utf-8", errors="ignore")
    ).hexdigest()[:16]
    diffuse_stem = os.path.splitext(os.path.basename(diffuse_path))[0]
    alpha_stem = os.path.splitext(os.path.basename(alpha_path))[0]
    out_dir = _texture_resolve_temp_dir()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{diffuse_stem}-{alpha_stem}-{digest}.webp")
    if os.path.isfile(out_path):
        return out_path
    try:
        from PIL import Image

        with Image.open(diffuse_path) as diffuse_img, Image.open(alpha_path) as alpha_img:
            rgba = diffuse_img.convert("RGBA")
            mask = alpha_img.convert("L")
            if mask.size != rgba.size:
                mask = mask.resize(rgba.size, Image.Resampling.NEAREST)
            rgba.putalpha(mask)
            rgba.save(out_path, format="WEBP", quality=75, method=4)
        return out_path
    except Exception:  # noqa: BLE001 - texture resolution is best-effort
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except OSError:
            pass
        return None


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
    base_color = bsdf.inputs["Base Color"]
    for link in list(base_color.links):
        node_tree.links.remove(link)
    node_tree.links.new(diffuse_node.outputs["Color"], bsdf.inputs["Base Color"])


def _connect_alpha_webp_blend(bpy, mat, bsdf, diffuse_path: str, alpha_path: str) -> bool:
    if "Alpha" not in bsdf.inputs:
        return False
    webp_path = _compose_alpha_webp(diffuse_path, alpha_path)
    if webp_path is None:
        return False

    node_tree = mat.node_tree
    webp_node = node_tree.nodes.new("ShaderNodeTexImage")
    webp_node.image = _load_texture_image(bpy, webp_path, non_color=False)

    base_color = bsdf.inputs["Base Color"]
    for link in list(base_color.links):
        node_tree.links.remove(link)
    node_tree.links.new(webp_node.outputs["Color"], base_color)

    alpha_input = bsdf.inputs["Alpha"]
    for link in list(alpha_input.links):
        node_tree.links.remove(link)
    node_tree.links.new(webp_node.outputs["Alpha"], alpha_input)

    if hasattr(mat, "blend_method"):
        mat.blend_method = "HASHED"
    return True


# ── Public entry point ──────────────────────────────────────────────────────

def resolve_main_character_textures(
    bpy,
    armature,
    mesh_path: str,
    *,
    force: bool = False,
) -> bool:
    """Resolve missing diffuse / alpha textures on the skinned main character.

    Only meshes bound to *armature* are touched (helper primitives such as
    Icosphere/Cube are skipped). A diffuse is wired on only when the mesh lacks
    a usable one; a matching alpha map is composed with the diffuse into RGBA
    WebP only when a confirmed mask exists. When ``force`` is true, existing
    diffuse and alpha links are replaced from the discovered texture files.

    Returns:
        ``True`` if at least one texture (diffuse or alpha) was applied,
        ``False`` if everything was already complete (no-op).
    """
    any_applied = False
    meshes = [
        m for m in _armature_bound_meshes(bpy, armature)
        if not _is_excluded_primitive(m.name)
    ]
    if not meshes:
        return False
    tex_files = _discover_texture_files(mesh_path)
    if not tex_files:
        return False

    char_name = os.path.basename(os.path.dirname(os.path.abspath(mesh_path)))
    for mesh in meshes:
        targets = list(_iter_material_bsdfs(mesh))
        has_diffuse = any(
            _image_is_valid(bpy, _base_color_image(bsdf))
            for _mat, bsdf in targets
        )
        diffuse_path = (
            _select_diffuse_texture(tex_files, char_name, mesh.name)
            if force or not has_diffuse else None
        )
        if diffuse_path is None and not targets:
            continue

        mat = bsdf = None
        applied = []
        if diffuse_path is not None:
            mat, bsdf = _get_or_create_material_bsdf(bpy, mesh)
            _connect_diffuse(bpy, mat, bsdf, diffuse_path)
            targets = [(mat, bsdf)]
            applied.append(f"diffuse={os.path.basename(diffuse_path)}")

        for target_mat, target_bsdf in targets:
            alpha_input = target_bsdf.inputs.get("Alpha")
            if alpha_input is None or (alpha_input.is_linked and not force):
                continue

            diffuse_for_alpha = diffuse_path
            if diffuse_for_alpha is None:
                hint = _base_color_image_hint(bpy, target_bsdf)
                if hint:
                    resolved = bpy.path.abspath(hint)
                    if os.path.isfile(resolved):
                        diffuse_for_alpha = resolved
                if diffuse_for_alpha is None:
                    diffuse_for_alpha = _select_diffuse_texture(
                        tex_files, char_name, target_mat.name or mesh.name
                    )
            if diffuse_for_alpha is None:
                continue

            alpha_path = _select_alpha_texture(tex_files, diffuse_for_alpha)
            if alpha_path is None or not _is_alpha_mask_image(bpy, alpha_path):
                continue
            if _connect_alpha_webp_blend(
                bpy, target_mat, target_bsdf, diffuse_for_alpha, alpha_path
            ):
                applied.append(f"alpha={os.path.basename(alpha_path)}")

        if applied:
            any_applied = True
    return any_applied
