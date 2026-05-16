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
from .retarget_cache import (
    get_from_memory,
    set_in_memory,
    load_from_disk,
    save_to_disk,
)


# ---------------------------------------------------------------------------
# LLM-based joint mapping
# ---------------------------------------------------------------------------

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

    Results are cached (in-memory + on-disk) by a SHA-256 hash of the
    prompt content so repeated calls for the same skeleton pair skip
    the API entirely.
    """
    # --- Build messages (needed for cache lookup and LLM call) ---
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
        "Use anatomical knowledge, hierarchy, and rest-pose geometry.\n"
        "\n"
        "CRITICAL RULES — name similarity alone is NEVER sufficient:\n"
        "1. STRUCTURE OVER NAME. Two joints sharing a name (e.g. both called "
        "`Tail 06`) can have completely different topology. Always verify the "
        "candidate's parent (and chain ancestry) matches the source joint's role. "
        "If the parents differ structurally, do NOT map them together just because "
        "the names match.\n"
        "2. NUMBERED CHAINS (Tail/Spine/Neck/Finger/etc., joints with numeric "
        "suffixes). The parent of joint N in a chain MUST be joint N-1 of the "
        "same chain. Before mapping `<Chain> N_src` → `<Chain> M_tgt`, confirm "
        "the target's parent is the previous joint of the same chain. If the "
        "target's same-named joint is actually a sibling/helper hanging off an "
        "earlier joint (e.g. target `Tail 06`.parent == `Tail 02` instead of "
        "`Tail 05`), it is NOT a chain continuation — return null, or map to "
        "the structurally-correct position further down the chain.\n"
        "3. HELPER / SIBLING BONES — a joint that shares a parent with a chain "
        "joint but is not itself in the chain (typically children=0, attached "
        "mid-chain). These map only to similarly-positioned siblings, never to "
        "a sequential chain slot. If no sibling exists on the other side, return "
        "null.\n"
        "4. CHAIN LENGTH MISMATCH. If src chain has N joints and tgt chain has "
        "M with N != M, distribute proportionally along the chain (e.g. "
        "src[i] → tgt[round(i * (M-1) / (N-1))]). Do not blindly pair by index "
        "when chain shapes differ.\n"
        "5. Use bone_len and children count as tiebreakers — a leaf (children=0) "
        "should not map to a junction with multiple children, and vice versa.\n"
        "\n"
        "Return ONLY valid JSON — no explanation, no markdown fences."
    )
    user_msg = (
        f"{geom_note}"
        f"Source skeleton:\n{src_text}\n\n"
        f"Target skeleton:\n{tgt_text}\n\n"
        'Return JSON: {"src_joint_name": "tgt_joint_name_or_null", ...}'
    )

    # --- Try in-memory cache (fast, process-local) ---
    mem_result = get_from_memory(system_msg, user_msg)
    if mem_result is not None:
        return mem_result

    # --- Try disk cache (persistent, cross-process) ---
    disk_result = load_from_disk(system_msg, user_msg)
    if disk_result is not None:
        set_in_memory(system_msg, user_msg, disk_result)
        print(f"[retarget] Disk cache hit for skeleton pair "
              f"src={len(src_names)} joints  tgt={len(tgt_names)} joints")
        return disk_result

    client, model = _get_llm_client_and_model()

    tgt_name_set = set(tgt_names)

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

        set_in_memory(system_msg, user_msg, result)
        save_to_disk(system_msg, user_msg, result)
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


def _build_target_bridge_source_indices(
    src_for_tgt: np.ndarray,
    src_to_tgt: np.ndarray,
    src_parents: np.ndarray,
    tgt_parents: np.ndarray,
    tgt_children_count: np.ndarray,
) -> np.ndarray:
    """Return source-bone indices for target chains that subdivide one source bone.

    When a mapped source joint's parent also maps to the target, but lands on a
    non-parent target ancestor, the target joints in between represent extra
    segments inserted into a single source bone. Distribute that one source-bone
    direction across the whole target span instead of letting the intermediate
    joints drift with the transport frame.
    """
    bridge_src_for_tgt = np.full(len(tgt_parents), -1, dtype=np.int32)
    mapped_mask = src_for_tgt >= 0

    for tgt_joint_idx in range(len(tgt_parents)):
        src_joint_idx = int(src_for_tgt[tgt_joint_idx])
        if src_joint_idx < 0:
            continue

        src_parent_idx = int(src_parents[src_joint_idx])
        if src_parent_idx < 0:
            continue

        anchor_tgt_idx = int(src_to_tgt[src_parent_idx])
        if anchor_tgt_idx < 0:
            continue

        direct_parent_idx = int(tgt_parents[tgt_joint_idx])
        if anchor_tgt_idx == direct_parent_idx:
            continue

        path_to_anchor: list[int] = []
        cursor = tgt_joint_idx
        valid_path = True
        while cursor != anchor_tgt_idx:
            path_to_anchor.append(cursor)
            cursor_parent = int(tgt_parents[cursor])
            if cursor_parent < 0:
                valid_path = False
                break
            if cursor != tgt_joint_idx and mapped_mask[cursor]:
                valid_path = False
                break
            cursor = cursor_parent

        if not valid_path or cursor != anchor_tgt_idx or len(path_to_anchor) <= 1:
            continue

        path_to_anchor.reverse()
        intermediate_joints = path_to_anchor[:-1]
        if any(mapped_mask[joint_idx] for joint_idx in intermediate_joints):
            continue
        if any(int(tgt_children_count[joint_idx]) != 1 for joint_idx in intermediate_joints):
            continue
        if any(int(bridge_src_for_tgt[joint_idx]) >= 0 for joint_idx in path_to_anchor):
            continue

        for joint_idx in path_to_anchor:
            bridge_src_for_tgt[joint_idx] = src_joint_idx

    return bridge_src_for_tgt


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
    src_effective_root_index: int | None = None,
    src_bone_translations: Optional[np.ndarray] = None,
    coordinate_search: bool = True,
    verbose: bool = True,
) -> RetargetResult:
    """Retarget an exporter-style animation from a source skeleton to a target.

    Bone-vector direction-transfer with rigid target skeleton: each mapped
    bone is placed at ``K2 = P2 + L · dir(K − P1)`` where ``dir`` is the
    source bone's unit world direction (aligned by a 1-of-12 rigid coordinate
    match + scale), ``P2`` is the target-skeleton parent's world position, and
    ``L = ‖tgt_rest_offsets[j]‖ · (src_anim_len / src_rest_len)`` is the
    target rest bone length modulated by the source bone's *relative*
    squash/stretch — so the target keeps its own proportions while the
    source's stretch carries through (the ratio is 1 when the source has no
    bone-translation channel, making pure-rotation behavior unchanged and
    self-retarget exactly idempotent). Unmapped target joints stay rigid
    relative to a
    transport frame (rest-composed, source-aligned at mapped ancestors). Only
    the root joint carries source translation (scaled), so global locomotion
    transfers. Target world rotation is the source's aligned world rotation
    for mapped joints (rest-composed off the parent for unmapped ones): the
    bone-vector transfer carries position, so self-retarget reproduces the
    source rotation exactly (twist included) with no skeleton-equality
    shortcut. The rotation is intentionally NOT re-fit to the realized
    positions — those carry the source's per-bone translation channel
    (squash/stretch / IK), a translation that a position-derived rotation
    would chase and thereby break rotation idempotency; that position vs.
    rest-rotation residual is represented exactly by the inverse-FK
    pose-location channel instead. This also avoids the chain-skip position
    discontinuity that arises when source and target chains differ in length.

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
        src_effective_root_index: optional source joint index that carries the
            locomotion translation in local position channels (for example Horse
            ``Bip01`` beneath a static wrapper root). When that joint is left
            unmatched by semantic mapping, it may replace a mapped wrapper root
            as the target root anchor.
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

    def _is_source_ancestor(ancestor_idx: int, descendant_idx: int) -> bool:
        current_idx = int(descendant_idx)
        while current_idx >= 0:
            if current_idx == int(ancestor_idx):
                return True
            current_idx = int(src_parents[current_idx])
        return False

    # ── B) Map source → target indices by semantic match name ─────────────
    # Strategy:
    #   1. First try exact same-name matching.
    #   2. If all non-leaf joints match and only leaf joints differ, use it directly.
    #   3. Otherwise discard the exact-match result and fall through to LLM mapping.
    #
    # The LLM call sees the full source and target skeletons (names + parent
    # + normalized bone length + children count) and is authoritative: its
    # answer for each src joint — including explicit null — is taken as final,
    # because the LLM is the only step that reasons over topology and can
    # reject a coincidental same-name match (e.g. a helper joint sharing a
    # name with an unrelated chain joint on the other skeleton).
    tgt_match_to_idx = {name: i for i, name in enumerate(tgt_match_names)}
    src_to_tgt = np.full(J_src, -1, dtype=np.int32)

    # -- Step 1: try exact same-name matching --
    # Two skeletons are considered "the same" if every src joint finds a same-name
    # tgt joint, and any unmatched joints on either side are only leaf joints
    # (no children).  This allows skeletons that differ only in the number of
    # leaf helpers / end-site joints to skip the LLM call.
    exact_mapping: dict[int, int] = {}
    matched_tgt = np.zeros(J_tgt, dtype=bool)

    for i, src_name in enumerate(src_match_names):
        j = tgt_match_to_idx.get(src_name)
        if j is not None and not matched_tgt[j]:
            exact_mapping[i] = j
            matched_tgt[j] = True

    # Build children count for both skeletons
    src_children_count = np.zeros(J_src, dtype=np.int32)
    for p in src_parents:
        if p >= 0:
            src_children_count[int(p)] += 1
    tgt_children_count = np.zeros(J_tgt, dtype=np.int32)
    for p in tgt_parents:
        if p >= 0:
            tgt_children_count[int(p)] += 1

    # Unmatched src joints are leaves?
    all_src_unmatched_are_leaves = all(
        src_children_count[i] == 0 for i in range(J_src) if i not in exact_mapping
    )
    # Unmatched tgt joints are leaves?
    all_tgt_unmatched_are_leaves = all(
        tgt_children_count[j] == 0 for j in range(J_tgt) if not matched_tgt[j]
    )

    is_exact_match = (
        len(exact_mapping) > 0
        and all_src_unmatched_are_leaves
        and all_tgt_unmatched_are_leaves
    )

    if is_exact_match:
        # Exact match (possibly with leaf-only differences) — use it directly
        unmatched_src_count = J_src - len(exact_mapping)
        unmatched_tgt_count = int((~matched_tgt).sum())
        if verbose:
            leaf_note = ""
            if unmatched_src_count or unmatched_tgt_count:
                leaf_note = f" ({unmatched_src_count} src leaves, {unmatched_tgt_count} tgt leaves skipped)"
            print(f"[retarget] Exact same-name matching: {len(exact_mapping)}/{J_src} joints matched{leaf_note}")
        for i, j in exact_mapping.items():
            src_to_tgt[i] = j
    else:
        # Partial match — discard and use LLM instead
        if verbose:
            print(f"[retarget] Exact same-name matching: {len(exact_mapping)}/{J_src} matched, "
                  f"switching to LLM mapping")
        matched_tgt = np.zeros(J_tgt, dtype=bool)
        llm_result = _llm_joint_mapping(
            src_match_names, tgt_match_names,
            src_parents, tgt_parents,
            src_rest_offsets, tgt_rest_offsets,
        )
        for i, src_name in enumerate(src_match_names):
            tgt_name = llm_result.get(src_name)
            if tgt_name is None:
                continue
            j = tgt_match_to_idx.get(tgt_name)
            if j is not None and not matched_tgt[j]:
                src_to_tgt[i] = j
                matched_tgt[j] = True

    root_tgt_indices = np.flatnonzero(tgt_parents < 0)
    root_tgt_idx = int(root_tgt_indices[0]) if root_tgt_indices.size > 0 else -1
    if (
        src_effective_root_index is not None
        and root_tgt_idx >= 0
    ):
        src_effective_root_index = int(src_effective_root_index)
        if 0 <= src_effective_root_index < J_src and int(src_to_tgt[src_effective_root_index]) < 0:
            current_root_src = np.flatnonzero(src_to_tgt == root_tgt_idx)
            current_root_src_idx = int(current_root_src[0]) if current_root_src.size > 0 else -1
            should_promote_effective_root = (
                current_root_src_idx < 0
                or _is_source_ancestor(current_root_src_idx, src_effective_root_index)
            )
            if should_promote_effective_root:
                if current_root_src_idx >= 0:
                    src_to_tgt[current_root_src_idx] = -1
                src_to_tgt[src_effective_root_index] = root_tgt_idx
                if verbose:
                    replaced_name = (
                        src_match_names[current_root_src_idx]
                        if current_root_src_idx >= 0 else '<none>'
                    )
                    print(
                        f"[retarget] Promoting unmapped source effective root "
                        f"{src_match_names[src_effective_root_index]!r} to target root "
                        f"(replacing {replaced_name!r})"
                    )


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

    # ── G) Bone-vector direction-transfer with rigid target skeleton ──────
    #
    # For a mapped bone K (target joint j) whose source-skeleton parent is P1
    # and target-skeleton parent is P2, the source's world-space bone
    # direction is transferred:
    #
    #     K2 = P2 + L · dir(K − P1)
    #
    # where ``dir`` is the unit world direction of the source bone in the
    # aligned (target) coordinate basis and ``L = ‖tgt_rest_offsets[j]‖ ·
    # (src_anim_len / src_rest_len)`` — the target's own rest bone length
    # scaled by the source bone's relative squash/stretch. So the target keeps
    # its own proportions while the source's stretch carries through; with no
    # source bone-translation channel the ratio is 1 and behavior is
    # unchanged, and self-retarget stays exactly idempotent. The root joint, if
    # mapped, takes the source's (scaled + axis-aligned) world translation so
    # global locomotion transfers. Unmapped joints stay rigid relative to a
    # transport frame (rest-composed, source-aligned at mapped ancestors),
    # which keeps the old gap/side-branch behavior and avoids the
    # "mapped → unmapped span → mapped" discontinuity across chains with
    # different joint counts (e.g. Parrot 2-joint neck vs Dragon 5-joint neck).
    #
    # A bone vector fixes position but not twist; Pass G2 takes the rotation
    # directly from the source's aligned world rotation (rest-composed for
    # unmapped joints) — exact for self-retarget, with the position vs.
    # rest-rotation residual absorbed by the Pass-H pose-location channel.
    aligned_src_wpos = (
        src_wpos * scale + t_align[np.newaxis, np.newaxis, :]
    ) @ best_R.T
    aligned_src_wrot = apply_rotation_to_quaternions_wxyz_np(src_wrot, best_R)
    F_q = aligned_src_wrot.shape[0]

    src_for_tgt = np.full(J_tgt, -1, dtype=np.int32)
    for ii in range(J_src):
        fi = int(src_to_tgt[ii])
        if fi >= 0:
            src_for_tgt[fi] = ii
    mapped_mask = src_for_tgt >= 0
    bridge_src_for_tgt = _build_target_bridge_source_indices(
        src_for_tgt,
        src_to_tgt,
        src_parents,
        tgt_parents,
        tgt_children_count,
    )

    target_wrot = np.zeros((F_q, J_tgt, 4), dtype=np.float64)
    target_wpos = np.zeros((F_q, J_tgt, 3), dtype=np.float64)

    _EPS = 1e-8

    # Pre-compute source-side bone vectors and animated lengths (vectorized)
    # For root joints (parent < 0) these stay zero — they won't be used.
    parent_idx = np.where(src_parents >= 0, src_parents, 0)  # safe index
    src_bv = src_wpos - src_wpos[:, parent_idx]               # (F, J_src, 3)  child - parent
    src_bv_aligned = aligned_src_wpos - aligned_src_wpos[:, parent_idx]
    src_anim_len = np.linalg.norm(src_bv, axis=-1)           # (F, J_src)

    # ── Pass G1: world positions + a transport frame for unmapped joints ──
    # tgt_parents is topologically ordered (parent index < child index), so a
    # single forward pass places every joint after its parent. ``transport``
    # carries unmapped (gap / side-branch) joints rigidly; it equals the old
    # rotation-only ``target_wrot`` and is NOT the final target rotation.
    transport = np.zeros((F_q, J_tgt, 4), dtype=np.float64)
    for j in range(J_tgt):
        p = int(tgt_parents[j])
        ii = int(src_for_tgt[j])
        bridge_src_idx = int(bridge_src_for_tgt[j])

        if p < 0:
            if ii >= 0:
                target_wpos[:, j] = aligned_src_wpos[:, ii]
                transport[:, j] = aligned_src_wrot[:, ii]
            else:
                target_wpos[:, j] = tgt_rest_wpos[0, j]
                transport[:, j] = tgt_rest_wrot[0, j]
            continue

        rest_off_j = tgt_rest_offsets[j]  # (3,) — broadcast, no repeat needed

        if bridge_src_idx >= 0:
            # Chain-length mismatch: one source bone spans several target joints
            # (e.g. Parrot Neck1 -> Head vs Dragon Neck2/3/4 -> Head). Spread
            # that source-bone direction across the whole target chain so the
            # inserted gap joints do not peel sideways with the transport frame.
            tgt_rest_len = float(np.linalg.norm(tgt_rest_offsets[j]))
            bn = np.linalg.norm(src_bv_aligned[:, bridge_src_idx], axis=-1)
            valid = bn > _EPS
            d = src_bv_aligned[:, bridge_src_idx] / np.where(valid, bn, 1.0)[:, None]

            src_rest_len = float(np.linalg.norm(src_rest_offsets[bridge_src_idx]))
            if src_rest_len > _EPS:
                stretch = src_anim_len[:, bridge_src_idx] / src_rest_len
            else:
                stretch = np.ones(F_q, dtype=np.float64)
            L = tgt_rest_len * stretch

            dir_pos = target_wpos[:, p] + L[:, None] * d
            rest_pos = target_wpos[:, p] + quat_rotate_wxyz_np(
                transport[:, p], rest_off_j,
            )
            target_wpos[:, j] = np.where(valid[:, None], dir_pos, rest_pos)

            if ii >= 0:
                transport[:, j] = aligned_src_wrot[:, ii]
            else:
                transport[:, j] = quat_multiply_wxyz_np(
                    transport[:, p],
                    np.repeat(tgt_rest_rotations[j:j + 1], F_q, axis=0),
                )
            continue

        if ii >= 0:
            # Mapped non-root: bone-vector direction transfer.
            tgt_rest_len = float(np.linalg.norm(tgt_rest_offsets[j]))
            p1 = int(src_parents[ii])
            if p1 >= 0:
                bn = np.linalg.norm(src_bv_aligned[:, ii], axis=-1)  # (F,)
                valid = bn > _EPS
                d = src_bv_aligned[:, ii] / np.where(valid, bn, 1.0)[:, None]

                src_rest_len = float(np.linalg.norm(src_rest_offsets[ii]))
                if src_rest_len > _EPS:
                    stretch = src_anim_len[:, ii] / src_rest_len      # (F,)
                else:
                    stretch = np.ones(F_q, dtype=np.float64)
                L = tgt_rest_len * stretch                     # (F,)

                # Per-frame: use direction transfer where valid, rest fallback where degenerate
                dir_pos = target_wpos[:, p] + L[:, None] * d
                rest_pos = target_wpos[:, p] + quat_rotate_wxyz_np(
                    transport[:, p], rest_off_j,
                )
                target_wpos[:, j] = np.where(valid[:, None], dir_pos, rest_pos)
            if p1 < 0:
                # Source bone is the root edge: no parent to form a bone vector.
                target_wpos[:, j] = target_wpos[:, p] + quat_rotate_wxyz_np(
                    transport[:, p], rest_off_j,
                )
            transport[:, j] = aligned_src_wrot[:, ii]
        else:
            # Unmapped non-root: rigid rest relative to the transport frame.
            target_wpos[:, j] = target_wpos[:, p] + quat_rotate_wxyz_np(
                transport[:, p], rest_off_j,
            )
            transport[:, j] = quat_multiply_wxyz_np(
                transport[:, p],
                np.repeat(tgt_rest_rotations[j:j + 1], F_q, axis=0),
            )

    # ── Pass G2: target world rotation = source world rotation ────────────
    # Pass G1 already places every joint by the bone-vector transfer. The
    # rotation is simply the source's aligned world rotation for mapped
    # joints, and the rest-composed rotation off the finalized parent for
    # unmapped (gap / side-branch) joints. This is exact for a self-retarget
    # — target_wrot == source world rotation, twist included — with no
    # skeleton-equality shortcut.
    #
    # It deliberately does NOT re-fit the rotation to the realized positions:
    # those positions carry the source's per-bone translation channel
    # (squash/stretch / IK offset), which is a translation, not a rotation —
    # any position-derived rotation would chase it and break rotation
    # idempotency. The position vs. rest-rotation residual is instead
    # represented exactly by the inverse-FK pose-location channel (Pass H).
    for p_idx in range(J_tgt):
        p_par = int(tgt_parents[p_idx])
        ii = int(src_for_tgt[p_idx])
        if ii >= 0:
            target_wrot[:, p_idx] = aligned_src_wrot[:, ii]
        elif p_par < 0:
            target_wrot[:, p_idx] = np.repeat(
                tgt_rest_wrot[0, p_idx][None], F_q, axis=0
            )
        else:
            target_wrot[:, p_idx] = quat_multiply_wxyz_np(
                target_wrot[:, p_par],
                np.repeat(tgt_rest_rotations[p_idx:p_idx + 1], F_q, axis=0),
            )

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

    # Bone-vector transfer: the position a fitted world rotation cannot
    # reproduce from rest offsets falls into the per-bone translation channel,
    # so ``tgt_pose_loc`` is generally non-zero and is kept when so. This now
    # also carries the source's *relative* squash/stretch (folded into the
    # transferred bone length L), so a stretched source bone reproduces a
    # proportionally stretched target bone — and self-retarget round-trips the
    # bone-translation channel exactly.
    has_nonzero_bone_translations = bool(
        np.any(np.abs(tgt_pose_loc[:, ~root_mask, :]) > 1e-6)
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
