"""
Pure-numpy cross-skeleton retargeting.

This module hosts the world-space alignment + semantic match-name remap +
inverse-FK math originally embedded inside ``AnimationExporter.export_glb``.
Lifting it out means callers other than the GLB export pipeline — e.g. the
cross-species reference-motion path in ``sample/generate.py`` — can run the
same retargeting without depending on ``bpy``.
"""
from __future__ import annotations

import json
import os
import warnings
from typing import Optional, TypedDict

import numpy as np

from .rotation_numpy import (
    apply_rotation_to_quaternions_wxyz_np,
    quat_conjugate_wxyz_np,
    quat_multiply_wxyz_np,
    quat_rotate_wxyz_np,
)


# ---------------------------------------------------------------------------
# LLM-based joint mapping
# ---------------------------------------------------------------------------

_LLM_CACHE: dict[tuple[tuple[str, ...], tuple[str, ...]], dict[str, str | None]] = {}
_LLM_CLIENT = None   # openai.OpenAI instance, created on first use
_LLM_MODEL: str | None = None  # resolved on first use


def _get_llm_client_and_model() -> tuple:
    """Return (client, model_name), initialising lazily on first call."""
    global _LLM_CLIENT, _LLM_MODEL
    if _LLM_CLIENT is not None:
        return _LLM_CLIENT, _LLM_MODEL

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "The 'openai' package is required for LLM joint mapping. "
            "Install it with: pip install openai"
        )

    base_url = os.environ.get("RETARGET_LLM_BASE_URL", "http://127.0.0.1:8066/v1")
    api_key = os.environ.get("RETARGET_LLM_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    _LLM_CLIENT = OpenAI(base_url=base_url, api_key=api_key or "sk-dummy")

    model_override = os.environ.get("RETARGET_LLM_MODEL", "")
    if model_override:
        _LLM_MODEL = model_override
        print(f"[retarget] LLM model set from env: {_LLM_MODEL}  endpoint: {base_url}")
    else:
        models = list(_LLM_CLIENT.models.list())
        if not models:
            raise RuntimeError(
                f"LLM endpoint {base_url} returned no models. "
                "Set RETARGET_LLM_MODEL to specify a model explicitly."
            )
        _LLM_MODEL = models[0].id
        print(f"[retarget] LLM model auto-discovered: {_LLM_MODEL}  endpoint: {base_url}")

    return _LLM_CLIENT, _LLM_MODEL


def _build_skeleton_text(
    names: list[str],
    parents: np.ndarray,
    rest_offsets: np.ndarray | None = None,
) -> str:
    """Format a skeleton as a flat list with parent, normalized bone length, children count.

    bone_len is the parent-relative offset magnitude normalized by the skeleton's
    max bone length (so values are in [0, 1] and comparable across skeletons of
    different units/scale).
    """
    J = len(names)
    children_count = np.zeros(J, dtype=np.int32)
    for p in parents:
        if p >= 0:
            children_count[int(p)] += 1

    bone_len_norm: np.ndarray | None = None
    if rest_offsets is not None:
        raw = np.linalg.norm(rest_offsets, axis=-1)
        max_len = float(raw.max()) if raw.size else 0.0
        bone_len_norm = raw / max_len if max_len > 1e-8 else raw

    lines = []
    for i, name in enumerate(names):
        p = int(parents[i])
        parent_name = "root" if p < 0 else names[p]
        extras = []
        if bone_len_norm is not None and p >= 0:
            extras.append(f"bone_len: {bone_len_norm[i]:.2f}")
        extras.append(f"children: {int(children_count[i])}")
        lines.append(f"- {name} (parent: {parent_name}, {', '.join(extras)})")
    return "\n".join(lines)


def _llm_joint_mapping(
    src_names: list[str],
    tgt_names: list[str],
    src_parents: np.ndarray,
    tgt_parents: np.ndarray,
    src_rest_offsets: np.ndarray | None = None,
    tgt_rest_offsets: np.ndarray | None = None,
) -> dict[str, str | None]:
    """Call LLM to map every src joint name to a tgt joint name (or None).

    Results are cached by (src_names, tgt_names) tuple pair so repeated
    calls for the same skeleton pair skip the API entirely.
    """
    cache_key = (tuple(src_names), tuple(tgt_names))
    if cache_key in _LLM_CACHE:
        return _LLM_CACHE[cache_key]

    client, model = _get_llm_client_and_model()

    tgt_name_set = set(tgt_names)
    src_text = _build_skeleton_text(src_names, src_parents, src_rest_offsets)
    tgt_text = _build_skeleton_text(tgt_names, tgt_parents, tgt_rest_offsets)

    has_geom = src_rest_offsets is not None and tgt_rest_offsets is not None
    geom_note = (
        "Each joint shows `bone_len` (parent-relative offset magnitude, normalized "
        "to [0, 1] by each skeleton's longest bone — so it is scale-invariant and "
        "comparable across skeletons) and `children` (number of direct child joints; "
        "0 = leaf/end-effector). Use these to distinguish long limbs from short "
        "fingers, and internal junctions (e.g. hip with 3 children for legs+spine) "
        "from chain joints.\n\n"
    ) if has_geom else ""

    system_msg = (
        "You are a skeleton joint mapping expert. "
        "Given a source skeleton and a target skeleton, return a JSON object "
        "mapping each source joint name to the best-matching target joint name, "
        "or null if no suitable match exists. "
        "Use anatomical knowledge, hierarchy, and rest-pose geometry. "
        "Return ONLY valid JSON — no explanation, no markdown fences."
    )
    user_msg = (
        f"{geom_note}"
        f"Source skeleton:\n{src_text}\n\n"
        f"Target skeleton:\n{tgt_text}\n\n"
        'Return JSON: {"src_joint_name": "tgt_joint_name_or_null", ...}'
    )

    messages: list[dict] = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    print(f"[retarget] Calling LLM for joint mapping  "
          f"model={model}  src={len(src_names)} joints  tgt={len(tgt_names)} joints")

    _MAX_RETRIES = 2
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        if attempt > 0:
            print(f"[retarget] LLM retry {attempt}/{_MAX_RETRIES} after parse error")

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            temperature=0,
            top_p=0.95,
            max_tokens=8192,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = response.choices[0].message.content or ""

        # Strip optional markdown fences the model may emit despite instructions
        stripped = raw.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("\n", 1)[-1]
            stripped = stripped.rsplit("```", 1)[0]

        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES:
                print(f"[retarget] LLM response parse error (attempt {attempt}): {exc}")
                # Feed parse error back so the model can self-correct
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your response could not be parsed as JSON. "
                        f"Error: {exc}. "
                        "Please return ONLY valid JSON, nothing else."
                    ),
                })
                continue
            raise RuntimeError(
                f"LLM returned unparseable JSON after {_MAX_RETRIES + 1} attempts. "
                f"Last error: {exc}\nLast response:\n{raw}"
            ) from exc

        if not isinstance(parsed, dict):
            # LLM returned valid JSON but wrong type (e.g. array) — treat as parse error
            type_err = f"expected JSON object, got {type(parsed).__name__}"
            last_exc = ValueError(type_err)
            if attempt < _MAX_RETRIES:
                print(f"[retarget] LLM response wrong type (attempt {attempt}): {type_err}")
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your response was valid JSON but had the wrong structure: {type_err}. "
                        "Return a JSON object {{src_name: tgt_name_or_null, ...}}, nothing else."
                    ),
                })
                continue
            raise RuntimeError(
                f"LLM returned wrong JSON type after {_MAX_RETRIES + 1} attempts: {type_err}\n"
                f"Last response:\n{raw}"
            ) from last_exc

        # Validate and clean the parsed mapping
        result: dict[str, str | None] = {}
        invalid_mappings: list[str] = []
        for src_name in src_names:
            val = parsed.get(src_name)
            if val is None or val not in tgt_name_set:
                if val is not None:
                    invalid_mappings.append(f"'{src_name}' → '{val}'")
                result[src_name] = None
            else:
                result[src_name] = val

        if invalid_mappings:
            warnings.warn(
                f"[retarget] LLM returned {len(invalid_mappings)} invalid target joint(s) "
                f"(not in target skeleton), ignored: {', '.join(invalid_mappings[:5])}"
                + (" ..." if len(invalid_mappings) > 5 else ""),
                stacklevel=2,
            )

        matched_count = sum(1 for v in result.values() if v is not None)
        print(f"[retarget] LLM mapping result: {matched_count}/{len(src_names)} src joints matched")
        for src_name, tgt_name in result.items():
            status = f"→ {tgt_name}" if tgt_name else "→ (no match)"
            print(f"[retarget]   {src_name}  {status}")

        _LLM_CACHE[cache_key] = result
        return result

    # Unreachable: every iteration either returns or raises inside the loop.
    raise RuntimeError(f"LLM joint mapping failed: {last_exc}")


# ---------------------------------------------------------------------------
# Shared helpers used by the retarget/exporter numpy path.
# ---------------------------------------------------------------------------


def _generate_coordinate_candidates_np():
    """Generate candidate 3x3 rotation/flip matrices for auto-detection."""
    I = np.eye(3, dtype=np.float64)

    def R_x(deg):
        c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)

    def R_y(deg):
        c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)

    def R_z(deg):
        c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

    return [
        ("identity", I),
        ("R_x(+90°)", R_x(90)),
        ("R_x(-90°)", R_x(-90)),
        ("R_y(+90°)", R_y(90)),
        ("R_y(-90°)", R_y(-90)),
        ("R_z(+90°)", R_z(90)),
        ("R_z(-90°)", R_z(-90)),
        ("R_x(+180°)", R_x(180)),
        ("R_z(+180°)", R_z(180)),
        ("flip_X", np.diag([-1, 1, 1])),
        ("flip_Y", np.diag([1, -1, 1])),
        ("flip_Z", np.diag([1, 1, -1])),
    ]


def _batch_forward_kinematics_np(
    local_rotations: np.ndarray,
    local_positions: np.ndarray,
    parents: np.ndarray,
    rest_rotations: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute world-space positions & rotations from local data.

    Args:
        local_rotations: (F, J, 4)  animated local quaternions
        local_positions: (F, J, 3)  local (parent-relative) translations
        parents:         (J,) int32 parent indices (-1 = root)
        rest_rotations:  (J, 4) or None — total local rot = rest_rot ⊗ local_rot

    Returns:
        world_positions: (F, J, 3)
        world_rotations: (F, J, 4)
    """
    F, J = local_rotations.shape[:2]

    if rest_rotations is not None:
        total_local = np.zeros((F, J, 4), dtype=np.float64)
        for j in range(J):
            total_local[:, j] = quat_multiply_wxyz_np(
                rest_rotations[j:j+1], local_rotations[:, j]
            )
    else:
        total_local = local_rotations.copy()

    world_pos = np.zeros((F, J, 3), dtype=np.float64)
    world_rot = np.zeros((F, J, 4), dtype=np.float64)

    for j in range(J):
        p = parents[j]
        if p < 0:
            world_pos[:, j] = local_positions[:, j]
            world_rot[:, j] = total_local[:, j]
        else:
            world_pos[:, j] = world_pos[:, p] + quat_rotate_wxyz_np(
                world_rot[:, p], local_positions[:, j]
            )
            world_rot[:, j] = quat_multiply_wxyz_np(world_rot[:, p], total_local[:, j])

    return world_pos, world_rot


def _batch_pose_fk_np(
    pose_rotations: np.ndarray,
    pose_locations: np.ndarray,
    parents: np.ndarray,
    rest_offsets: np.ndarray,
    rest_rotations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute world transforms using Blender pose-bone semantics.

    Local bone transform is modeled as:
        T_local = T(rest_offset) * R(rest_rotation) * T(pose_location) * R(pose_rotation)

    This matches how the exporter drives external FBX/GLB armatures through
    pose bone `location` and `rotation_quaternion` channels.
    """
    F, J = pose_rotations.shape[:2]
    world_pos = np.zeros((F, J, 3), dtype=np.float64)
    world_rot = np.zeros((F, J, 4), dtype=np.float64)

    for j in range(J):
        rest_q = np.repeat(rest_rotations[j:j+1], F, axis=0)
        total_local_rot = quat_multiply_wxyz_np(rest_q, pose_rotations[:, j])
        pose_loc_in_parent = rest_offsets[j:j+1] + quat_rotate_wxyz_np(rest_q, pose_locations[:, j])

        p = parents[j]
        if p < 0:
            world_pos[:, j] = pose_loc_in_parent
            world_rot[:, j] = total_local_rot
        else:
            world_pos[:, j] = world_pos[:, p] + quat_rotate_wxyz_np(world_rot[:, p], pose_loc_in_parent)
            world_rot[:, j] = quat_multiply_wxyz_np(world_rot[:, p], total_local_rot)

    return world_pos, world_rot


def _batch_internal_pose_fk_np(
    joint_rotations: np.ndarray,
    root_translation: np.ndarray,
    root_rotation: np.ndarray,
    pose_locations: np.ndarray | None,
    parents: np.ndarray,
    rest_offsets: np.ndarray,
    rest_rotations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute world pose with unified exporter semantics.

    External caller semantics:
      - ``joint_rotations`` carries animated local joint quaternions for all joints,
        including the root joint.
      - ``root_translation`` / ``root_rotation`` form an extra world-space wrapper
        transform applied before the skeleton hierarchy.
      - ``pose_locations`` carries optional Blender-style pose-bone location channels
        for non-root joints. The root entry is ignored; root world translation always
        comes from ``root_translation``.
    """
    F, J = joint_rotations.shape[:2]
    world_pos = np.zeros((F, J, 3), dtype=np.float64)
    world_rot = np.zeros((F, J, 4), dtype=np.float64)

    zero_loc = np.zeros((F, 3), dtype=np.float64)

    for j in range(J):
        rest_q = np.repeat(rest_rotations[j:j+1], F, axis=0)
        total_local_rot = quat_multiply_wxyz_np(rest_q, joint_rotations[:, j])

        if pose_locations is None or parents[j] < 0:
            pose_loc = zero_loc
        else:
            pose_loc = pose_locations[:, j]

        local_pos = np.repeat(rest_offsets[j:j+1], F, axis=0) + quat_rotate_wxyz_np(
            rest_q,
            pose_loc,
        )

        p = parents[j]
        if p < 0:
            world_pos[:, j] = root_translation + quat_rotate_wxyz_np(root_rotation, local_pos)
            world_rot[:, j] = quat_multiply_wxyz_np(root_rotation, total_local_rot)
        else:
            world_pos[:, j] = world_pos[:, p] + quat_rotate_wxyz_np(world_rot[:, p], local_pos)
            world_rot[:, j] = quat_multiply_wxyz_np(world_rot[:, p], total_local_rot)

    return world_pos, world_rot


# ---------------------------------------------------------------------------
# Public retargeting function
# ---------------------------------------------------------------------------

class RetargetResult(TypedDict):
    joint_rotations: np.ndarray         # (F, J_tgt, 4)
    root_translation: np.ndarray        # (F, 3)
    root_rotation: np.ndarray           # (F, 4)
    bone_translations: Optional[np.ndarray]  # (F, J_tgt, 3) or None
    target_world_positions: np.ndarray  # (F, J_tgt, 3)
    target_world_rotations: np.ndarray  # (F, J_tgt, 4)
    src_to_tgt: np.ndarray              # (J_src,) int32, -1 where no match
    common_count: int
    alignment_label: str
    alignment_error: float
    alignment_scale: float
    alignment_translation: np.ndarray   # (3,)


def retarget_world_space_np(
    *,
    src_parents: np.ndarray,
    src_rest_offsets: np.ndarray,
    src_rest_rotations: np.ndarray,
    tgt_parents: np.ndarray,
    tgt_rest_offsets: np.ndarray,
    tgt_rest_rotations: np.ndarray,
    src_joint_rotations: np.ndarray,
    src_root_translation: np.ndarray,
    src_root_rotation: np.ndarray,
    src_match_names: list[str],
    tgt_match_names: list[str],
    src_bone_translations: Optional[np.ndarray] = None,
    coordinate_search: bool = True,
    verbose: bool = True,
) -> RetargetResult:
    """Retarget an exporter-style animation from a source skeleton to a target.

    The math is identical to what ``AnimationExporter.export_glb`` performs when
    a ``mesh_path`` is supplied: canonical bone-name matching → world-space
    alignment (scale + translation + 1-of-12 rigid rotation) → inverse FK back
    to target local pose channels.

    Args:
        src_parents: (J_src,) int32 parent indices, -1 for root.
        src_rest_offsets: (J_src, 3) parent-relative rest offsets.
        src_rest_rotations: (J_src, 4) WXYZ rest rotations.
        tgt_parents: (J_tgt,) int32 parent indices.
        tgt_rest_offsets: (J_tgt, 3) parent-relative rest offsets.
        tgt_rest_rotations: (J_tgt, 4) WXYZ rest rotations.
        src_joint_rotations: (F, J_src, 4) exporter-style local pose rotations.
        src_root_translation: (F, 3) wrapper translation applied above root.
        src_root_rotation: (F, 4) wrapper rotation applied above root.
        src_bone_translations: optional (F, J_src, 3) pose-bone locations for
            non-root joints. Root entry is ignored.
        src_match_names: semantic match names for source joints.
        tgt_match_names: semantic match names for target joints.
        coordinate_search: when ``True``, sweep 12 rigid rotation/flip candidates
            to find the best alignment of rest poses. Set ``False`` when the
            source and target are known to share the same world basis (e.g.
            both are processed cond entries from the same dataset pipeline).
        verbose: print one-line summary diagnostics.

    Returns:
        A ``RetargetResult`` mapping. ``joint_rotations`` / ``root_translation``
        / ``root_rotation`` / ``bone_translations`` are exporter-input
        compatible and can be fed straight back into ``AnimationExporter`` or
        used to drive any other target-skeleton animation pipeline.
    """
    src_parents = np.asarray(src_parents, dtype=np.int32)
    tgt_parents = np.asarray(tgt_parents, dtype=np.int32)
    src_rest_offsets = np.asarray(src_rest_offsets, dtype=np.float64)
    src_rest_rotations = np.asarray(src_rest_rotations, dtype=np.float64)
    tgt_rest_offsets = np.asarray(tgt_rest_offsets, dtype=np.float64)
    tgt_rest_rotations = np.asarray(tgt_rest_rotations, dtype=np.float64)

    jr_np = np.asarray(src_joint_rotations, dtype=np.float64)
    rt_np = np.asarray(src_root_translation, dtype=np.float64)
    rr_np = np.asarray(src_root_rotation, dtype=np.float64)
    pose_locations_np = (
        np.asarray(src_bone_translations, dtype=np.float64)
        if src_bone_translations is not None else None
    )

    F = jr_np.shape[0]
    src_match_names = list(src_match_names)
    tgt_match_names = list(tgt_match_names)
    J_src = len(src_match_names)
    J_tgt = len(tgt_match_names)
    if len(src_match_names) != J_src:
        raise ValueError(
            f"Source match-name count {len(src_match_names)} does not match source joint count {J_src}"
        )
    if len(tgt_match_names) != J_tgt:
        raise ValueError(
            f"Target match-name count {len(tgt_match_names)} does not match target joint count {J_tgt}"
        )

    # ── B) Map source → target indices by semantic match name ─────────────
    # Two-pass matching:
    #   1. Exact name match.
    #   2. LLM-based mapping for remaining unmatched joints (skipped when
    #      src is a subset of tgt names or tgt is fully covered by src).
    tgt_match_to_idx = {name: i for i, name in enumerate(tgt_match_names)}
    src_to_tgt = np.full(J_src, -1, dtype=np.int32)
    matched_tgt = np.zeros(J_tgt, dtype=bool)

    # Pass 1: exact match
    for i, name in enumerate(src_match_names):
        target_index = tgt_match_to_idx.get(name)
        if target_index is not None and not matched_tgt[target_index]:
            src_to_tgt[i] = target_index
            matched_tgt[target_index] = True

    # Pass 2: LLM-based mapping for remaining unmatched joints.
    # Subset/superset short-circuit: if all src joints are already mapped,
    # or all tgt joints are already covered, there is nothing left to do.
    unmatched_src_idx = [i for i in range(J_src) if src_to_tgt[i] < 0]
    unmatched_tgt_idx = [j for j in range(J_tgt) if not matched_tgt[j]]

    if not unmatched_src_idx:
        pass
    elif not unmatched_tgt_idx:
        print(f"[retarget] Pass 2 skipped: all tgt joints covered by exact names "
              f"(tgt ⊆ src, {len(unmatched_src_idx)} src joints unmatched/ignored)")
    else:
        tgt_name_to_unmatched_idx = {tgt_match_names[j]: j for j in unmatched_tgt_idx}

        llm_result = _llm_joint_mapping(
            src_match_names, tgt_match_names,
            src_parents, tgt_parents,
            src_rest_offsets, tgt_rest_offsets,
        )

        for i in unmatched_src_idx:
            tgt_name = llm_result.get(src_match_names[i])
            if tgt_name is not None:
                j = tgt_name_to_unmatched_idx.get(tgt_name)
                if j is not None and not matched_tgt[j]:
                    src_to_tgt[i] = j
                    matched_tgt[j] = True


    # ── D) Source animation in world space ────────────────────────────────
    src_wpos, src_wrot = _batch_internal_pose_fk_np(
        jr_np, rt_np, rr_np, pose_locations_np,
        src_parents, src_rest_offsets, src_rest_rotations,
    )

    # ── E) Target rest pose in world space ────────────────────────────────
    identity_q = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    tgt_rest_local_rot = np.tile(identity_q, (J_tgt, 1))  # (J_tgt, 4)
    tgt_rest_local_pos = np.zeros((1, J_tgt, 3), dtype=np.float64)

    tgt_rest_wpos, tgt_rest_wrot = _batch_pose_fk_np(
        tgt_rest_local_rot[None],   # (1, J, 4)
        tgt_rest_local_pos,         # (1, J, 3)
        tgt_parents,
        tgt_rest_offsets,
        tgt_rest_rotations,
    )

    # ── F) Compute alignment on common bones using source REST pose ───────
    # (Not frame 0 of the animation, which may be in a running pose.)
    src_rest_local_rot = np.tile(identity_q, (J_src, 1))
    src_rest_local_pos = np.zeros((1, J_src, 3), dtype=np.float64)
    src_rest_wpos, _ = _batch_pose_fk_np(
        src_rest_local_rot[None], src_rest_local_pos,
        src_parents, src_rest_offsets, src_rest_rotations,
    )

    common_src_idx = [i for i in range(J_src) if src_to_tgt[i] >= 0]
    common_tgt_idx = [int(src_to_tgt[i]) for i in common_src_idx]

    if not common_src_idx:
        raise RuntimeError(
            "No common joints between source and target semantic match names.\n"
            f"  Source match names: {src_match_names[:10]}...\n"
            f"  Target match names: {tgt_match_names[:10]}..."
        )

    pos_src_rest = src_rest_wpos[:, common_src_idx, :]   # (1, K, 3)
    pos_tgt_rest = tgt_rest_wpos[:, common_tgt_idx, :]   # (1, K, 3)

    def _mean_bone_len(pos, local_tgt_idx):
        lengths = []
        for ci, fi in enumerate(local_tgt_idx):
            p = tgt_parents[fi]
            if p < 0:
                continue
            for ci2, fi2 in enumerate(local_tgt_idx):
                if fi2 == p:
                    diff = pos[0, ci] - pos[0, ci2]
                    lengths.append(float(np.linalg.norm(diff)))
                    break
        return float(np.mean(lengths)) if lengths else 1.0

    mean_len_src = _mean_bone_len(pos_src_rest, common_tgt_idx)
    mean_len_tgt = _mean_bone_len(pos_tgt_rest, common_tgt_idx)

    scale = mean_len_tgt / mean_len_src if mean_len_src > 1e-8 else 1.0
    if abs(scale - 1.0) < 0.001:
        scale = 1.0

    root_tgt_idx = int(np.flatnonzero(tgt_parents == -1)[0])
    root_in_common = None
    for ci, fi in enumerate(common_tgt_idx):
        if fi == root_tgt_idx:
            root_in_common = ci
            break
    if root_in_common is None:
        root_in_common = 0

    # No rest-pose translation alignment: both source and target go through
    # process_anim() (root centered at XZ origin, feet grounded at y≈0, common
    # scale_factor), so any rest-based t_align tends to introduce drift rather
    # than correct it — especially Y, where the bone-length ratio used for
    # `scale` rarely matches the root-height ratio across species.
    t_align = np.zeros(3, dtype=np.float64)
    pos_src_rest_st = pos_src_rest * scale

    candidates = _generate_coordinate_candidates_np() if coordinate_search else [
        ("identity", np.eye(3, dtype=np.float64))
    ]
    best_R = np.eye(3, dtype=np.float64)
    best_label = "identity"
    best_err = float("inf")

    for label, R in candidates:
        pos_candidate = pos_src_rest_st @ R.T
        err = float(np.mean(np.linalg.norm(pos_tgt_rest - pos_candidate, axis=-1)))
        if err < best_err:
            best_err = err
            best_label = label
            best_R = R

    if verbose:
        print(f"  [Retarget] common={len(common_src_idx)}/{J_src}, "
              f"target={J_tgt} bones, alignment error={best_err:.6f}")
        print(f"  [Retarget] alignment: scale={scale:.6f}, "
              f"rot={best_label}, "
              f"trans=({t_align[0]:.4f}, {t_align[1]:.4f}, {t_align[2]:.4f})")

    # ── G) Apply alignment and remap to target world-space ────────────────
    target_wpos = np.repeat(tgt_rest_wpos, F, axis=0)
    target_wrot = np.repeat(tgt_rest_wrot, F, axis=0)
    aligned_src_wpos = (
        src_wpos * scale + t_align[np.newaxis, np.newaxis, :]
    ) @ best_R.T
    aligned_src_wrot = apply_rotation_to_quaternions_wxyz_np(src_wrot, best_R)

    mapped_mask = np.zeros(J_tgt, dtype=bool)
    for ii, fi in enumerate(src_to_tgt):
        if fi >= 0:
            target_wpos[:, fi] = aligned_src_wpos[:, ii]
            target_wrot[:, fi] = aligned_src_wrot[:, ii]
            mapped_mask[fi] = True

    # Propagate FK for every unmatched target joint from its (already updated)
    # parent.  In practice these are leaf rotation helpers — joints appended by
    # ``augment_leaf_rotation_helpers`` whose names end with ``__rot_helper`` —
    # which never canonical-match across species.  They should follow their
    # real leaf parent's animated world transform while keeping their rest
    # local offset and rotation (identity for helpers).  tgt_parents is in
    # topological order (parent index < child index), so a single forward pass
    # suffices.
    F_q = aligned_src_wrot.shape[0]
    for j in range(J_tgt):
        if mapped_mask[j]:
            continue
        p = int(tgt_parents[j])
        if p < 0:
            continue
        rest_q_j = np.repeat(tgt_rest_rotations[j:j+1], F_q, axis=0)
        rest_off_j = np.repeat(tgt_rest_offsets[j:j+1], F_q, axis=0)
        target_wpos[:, j] = target_wpos[:, p] + quat_rotate_wxyz_np(
            target_wrot[:, p], rest_off_j,
        )
        target_wrot[:, j] = quat_multiply_wxyz_np(target_wrot[:, p], rest_q_j)

    # ── H) Inverse FK back to target local pose channels ──────────────────
    tgt_pose_rot = np.zeros((F, J_tgt, 4), dtype=np.float64)
    tgt_pose_rot[:] = identity_q
    tgt_pose_loc = np.zeros((F, J_tgt, 3), dtype=np.float64)

    identity_q_row = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    for j in range(J_tgt):
        parent_j = int(tgt_parents[j])
        if parent_j < 0:
            parent_world_rot = np.repeat(identity_q_row[np.newaxis], F, axis=0)
            parent_world_pos = np.zeros((F, 3), dtype=np.float64)
        else:
            parent_world_rot = target_wrot[:, parent_j]
            parent_world_pos = target_wpos[:, parent_j]

        rel_world_rot = quat_multiply_wxyz_np(
            quat_conjugate_wxyz_np(parent_world_rot),
            target_wrot[:, j],
        )
        tgt_pose_rot[:, j] = quat_multiply_wxyz_np(
            quat_conjugate_wxyz_np(np.repeat(tgt_rest_rotations[j:j+1], F, axis=0)),
            rel_world_rot,
        )

        rel_world_pos = target_wpos[:, j] - parent_world_pos
        rel_parent_pos = quat_rotate_wxyz_np(
            quat_conjugate_wxyz_np(parent_world_rot),
            rel_world_pos,
        )
        tgt_pose_loc[:, j] = quat_rotate_wxyz_np(
            quat_conjugate_wxyz_np(np.repeat(tgt_rest_rotations[j:j+1], F, axis=0)),
            rel_parent_pos - np.repeat(tgt_rest_offsets[j:j+1], F, axis=0),
        )

    root_mask = tgt_parents < 0
    root_indices = np.flatnonzero(root_mask)

    if root_indices.size > 0:
        out_root_rotation = tgt_pose_rot[:, root_indices[0], :].copy()
        out_root_translation = tgt_pose_loc[:, root_indices[0], :].copy()
    else:
        out_root_rotation = rr_np.copy()
        out_root_translation = rt_np.copy()

    has_nonzero_bone_translations = (
        pose_locations_np is not None
        or np.any(np.abs(tgt_pose_loc[:, ~root_mask, :]) > 1e-6)
    )
    out_bone_translations = tgt_pose_loc if has_nonzero_bone_translations else None

    if verbose:
        print(f"  [Retarget] Conversion complete: {F} frames, {J_tgt} bones")

    return RetargetResult(
        joint_rotations=tgt_pose_rot,
        root_translation=out_root_translation,
        root_rotation=out_root_rotation,
        bone_translations=out_bone_translations,
        target_world_positions=target_wpos,
        target_world_rotations=target_wrot,
        src_to_tgt=src_to_tgt,
        common_count=len(common_src_idx),
        alignment_label=best_label,
        alignment_error=best_err,
        alignment_scale=scale,
        alignment_translation=t_align,
    )
