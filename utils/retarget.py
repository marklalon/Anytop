"""
Pure-numpy cross-skeleton retargeting.

This module hosts the world-space alignment + semantic match-name remap +
inverse-FK math originally embedded inside ``AnimationExporter.export_glb``.
Lifting it out means callers other than the GLB export pipeline — e.g. the
cross-species reference-motion path in ``sample/generate.py`` — can run the
same retargeting without depending on ``bpy``.

Usage as CLI::

    python -m Anytop.utils.retarget --source <path> --object_type <Type>
"""
from __future__ import annotations

# When run directly (``python utils/retarget.py``), relative imports fail
# because __package__ is None.  Auto-relaunch via ``python -m`` so the
# package context is correct.
import sys as _sys
if __name__ == '__main__' and __package__ is None:
    import os as __os
    import subprocess as __sp
    _ANYTOP_PARENT = __os.path.dirname(__os.path.dirname(__os.path.abspath(__file__)))
    _relaunch_env = __os.environ.copy()
    _existing_pythonpath = _relaunch_env.get('PYTHONPATH', '')
    _parts = [p for p in (_ANYTOP_PARENT, _existing_pythonpath) if p]
    _relaunch_env['PYTHONPATH'] = __os.pathsep.join(_parts)
    _sys.exit(__sp.call(
        [_sys.executable, '-m', 'Anytop.utils.retarget'] + _sys.argv[1:],
        cwd=_ANYTOP_PARENT,
        env=_relaunch_env,
    ))
del _sys

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

    base_url = os.environ.get("PCVG_LLM_ENDPOINT", "http://127.0.0.1:8066/v1")
    api_key = os.environ.get("PCVG_LLM_API_KEY", "")
    _LLM_CLIENT = OpenAI(base_url=base_url, api_key=api_key or "sk-dummy")

    model_override = os.environ.get(
        "PCVG_LLM_MODEL",
        os.environ.get("RETARGET_LLM_MODEL", ""),
    )
    if model_override:
        _LLM_MODEL = model_override
        print(f"[retarget] LLM model set from env: {_LLM_MODEL}  endpoint: {base_url}")
    else:
        models = list(_LLM_CLIENT.models.list())
        if not models:
            raise RuntimeError(
                f"LLM endpoint {base_url} returned no models. "
                "Set PCVG_LLM_MODEL or RETARGET_LLM_MODEL to specify a model explicitly."
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
            # Use 4 decimals (was .2f): .2f was too coarse and collapsed genuinely
            # distinct-but-similar bones onto the same string, producing false cache
            # hits and wrong joint mappings. 4 decimals keeps such bones distinct
            # in the LLM prompt / cache key.
            extras.append(f"bone_len: {bone_len_norm[i]:.4f}")
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
        matched = sum(1 for v in disk_result.values() if v is not None)
        print(f"[retarget] Disk cache hit for skeleton pair "
              f"src={len(src_names)} joints  tgt={len(tgt_names)} joints  "
              f"matched={matched}")
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

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                temperature=0,
                top_p=1.0,
                max_tokens=8192,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        except Exception as conn_err:
            print(f"[retarget] LLM connection failed: {conn_err}")
            print("[retarget] Aborting — fix the LLM endpoint and retry.")
            _sys.exit(1)
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


def generate_coordinate_candidates_np():
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


def batch_forward_kinematics_np(
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
    tgt_effective_root_index: int | None = None,
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

    def _is_target_ancestor(ancestor_idx: int, descendant_idx: int) -> bool:
        current_idx = int(descendant_idx)
        while current_idx >= 0:
            if current_idx == int(ancestor_idx):
                return True
            current_idx = int(tgt_parents[current_idx])
        return False

    def _chain_root(name: str) -> str:
        normalized = str(name).replace('_', ' ').strip().lower()
        for prefix in ('left ', 'right ', 'l ', 'r '):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        tokens = normalized.split()
        while tokens and tokens[-1].isdigit():
            tokens.pop()
        return ' '.join(tokens)

    def _path_from_ancestor_to_descendant(parents: np.ndarray, ancestor_idx: int, descendant_idx: int) -> list[int]:
        current_idx = int(descendant_idx)
        reversed_path: list[int] = []
        while current_idx >= 0 and current_idx != int(ancestor_idx):
            reversed_path.append(current_idx)
            current_idx = int(parents[current_idx])
        if current_idx != int(ancestor_idx):
            return []
        reversed_path.reverse()
        return reversed_path

    def _fill_target_gap_from_descendant_mapping(current_src_idx: int, gap_tgt_idx: int) -> bool:
        search_queue = np.flatnonzero(tgt_parents == int(gap_tgt_idx)).tolist()
        best_alignment: tuple[list[int], list[int]] | None = None
        best_target_path_len: int | None = None

        while search_queue:
            descendant_tgt_idx = int(search_queue.pop(0))
            search_queue.extend(np.flatnonzero(tgt_parents == descendant_tgt_idx).tolist())

            descendant_src_candidates = np.flatnonzero(src_to_tgt == descendant_tgt_idx)
            if descendant_src_candidates.size != 1:
                continue

            descendant_src_idx = int(descendant_src_candidates[0])
            if descendant_src_idx == int(current_src_idx):
                continue
            if not _is_source_ancestor(current_src_idx, descendant_src_idx):
                continue
            if _chain_root(src_match_names[descendant_src_idx]) != _chain_root(tgt_match_names[descendant_tgt_idx]):
                continue

            target_path = [int(gap_tgt_idx)] + _path_from_ancestor_to_descendant(
                tgt_parents,
                gap_tgt_idx,
                descendant_tgt_idx,
            )
            source_path = _path_from_ancestor_to_descendant(src_parents, current_src_idx, descendant_src_idx)
            if not target_path or not source_path:
                continue
            if len(source_path) < len(target_path):
                continue

            aligned_source_nodes = source_path[-len(target_path):]
            target_path_len = len(target_path)
            if best_target_path_len is None or target_path_len < best_target_path_len:
                best_alignment = (target_path, aligned_source_nodes)
                best_target_path_len = target_path_len

        if best_alignment is None:
            return False

        target_path, aligned_source_nodes = best_alignment
        for target_idx, source_idx in zip(target_path, aligned_source_nodes):
            previous_target_idx = int(src_to_tgt[source_idx])
            if previous_target_idx >= 0 and previous_target_idx != int(target_idx):
                src_to_tgt[source_idx] = -1

            existing_source = np.flatnonzero(src_to_tgt == int(target_idx))
            if existing_source.size == 1 and int(existing_source[0]) != int(source_idx):
                src_to_tgt[int(existing_source[0])] = -1

            src_to_tgt[source_idx] = int(target_idx)

        return True

    def _redistribute_mapped_path_along_longer_target_chain(anchor_src_idx: int, anchor_tgt_idx: int) -> None:
        search_queue = np.flatnonzero(tgt_parents == int(anchor_tgt_idx)).tolist()
        best_candidate: tuple[list[int], list[int]] | None = None
        best_target_path_len = -1

        while search_queue:
            descendant_tgt_idx = int(search_queue.pop(0))
            search_queue.extend(np.flatnonzero(tgt_parents == descendant_tgt_idx).tolist())

            descendant_src_candidates = np.flatnonzero(src_to_tgt == descendant_tgt_idx)
            if descendant_src_candidates.size != 1:
                continue

            descendant_src_idx = int(descendant_src_candidates[0])
            if descendant_src_idx == int(anchor_src_idx):
                continue
            if not _is_source_ancestor(anchor_src_idx, descendant_src_idx):
                continue

            target_path = [int(anchor_tgt_idx)] + _path_from_ancestor_to_descendant(
                tgt_parents,
                anchor_tgt_idx,
                descendant_tgt_idx,
            )
            source_path = [int(anchor_src_idx)] + _path_from_ancestor_to_descendant(
                src_parents,
                anchor_src_idx,
                descendant_src_idx,
            )
            if len(target_path) <= len(source_path):
                continue
            if not target_path or not source_path:
                continue
            if any(int(tgt_children_count[joint_idx]) != 1 for joint_idx in target_path[1:-1]):
                continue

            current_target_indices = []
            mapped_source_nodes: list[int] = []
            for source_idx in source_path:
                target_idx = int(src_to_tgt[source_idx])
                if target_idx < 0:
                    continue
                if target_idx not in target_path:
                    current_target_indices = []
                    mapped_source_nodes = []
                    break
                current_target_indices.append(target_path.index(target_idx))
                mapped_source_nodes.append(int(source_idx))

            if len(mapped_source_nodes) < 2:
                continue
            if current_target_indices != sorted(current_target_indices):
                continue
            if current_target_indices[0] != 0 or current_target_indices[-1] != len(target_path) - 1:
                continue

            if len(target_path) > best_target_path_len:
                best_candidate = (target_path, mapped_source_nodes)
                best_target_path_len = len(target_path)

        if best_candidate is None:
            return

        target_path, mapped_source_nodes = best_candidate
        if len(mapped_source_nodes) < 2:
            return

        desired_target_indices = [
            int(round(src_idx * (len(target_path) - 1) / (len(mapped_source_nodes) - 1)))
            for src_idx in range(len(mapped_source_nodes))
        ]
        desired_targets = [int(target_path[target_idx]) for target_idx in desired_target_indices]
        current_targets = [int(src_to_tgt[source_idx]) for source_idx in mapped_source_nodes]
        if desired_targets == current_targets:
            return

        for source_idx, target_idx in zip(mapped_source_nodes, desired_targets):
            existing_source = np.flatnonzero(src_to_tgt == int(target_idx))
            if existing_source.size == 1 and int(existing_source[0]) != int(source_idx):
                src_to_tgt[int(existing_source[0])] = -1

        for source_idx, target_idx in zip(mapped_source_nodes, desired_targets):
            src_to_tgt[int(source_idx)] = int(target_idx)

    def _shift_descendant_chain_after_root_promotion(current_src_idx: int, displaced_tgt_idx: int) -> None:
        target_cursor = int(displaced_tgt_idx)
        source_cursor = int(current_src_idx)
        while target_cursor >= 0:
            child_tgt_candidates = np.flatnonzero(tgt_parents == target_cursor)
            if child_tgt_candidates.size == 0:
                break

            matching_pairs: list[tuple[int, int]] = []
            for child_tgt_idx in child_tgt_candidates.tolist():
                child_src_candidates = np.flatnonzero(src_to_tgt == int(child_tgt_idx))
                if child_src_candidates.size != 1:
                    continue
                child_src_idx = int(child_src_candidates[0])
                if int(src_parents[child_src_idx]) != source_cursor:
                    continue
                matching_pairs.append((int(child_tgt_idx), child_src_idx))

            if len(matching_pairs) > 1:
                semantic_pairs = [
                    (child_tgt_idx, child_src_idx)
                    for child_tgt_idx, child_src_idx in matching_pairs
                    if _chain_root(src_match_names[child_src_idx]) == _chain_root(tgt_match_names[child_tgt_idx])
                ]
                if len(semantic_pairs) == 1:
                    matching_pairs = semantic_pairs

            if len(matching_pairs) != 1:
                if _fill_target_gap_from_descendant_mapping(source_cursor, target_cursor):
                    mapped_source_after_fill = np.flatnonzero(src_to_tgt == target_cursor)
                    if mapped_source_after_fill.size == 1:
                        _redistribute_mapped_path_along_longer_target_chain(
                            int(mapped_source_after_fill[0]),
                            target_cursor,
                        )
                    break
                break

            child_tgt_idx, child_src_idx = matching_pairs[0]

            src_to_tgt[child_src_idx] = target_cursor
            source_cursor = child_src_idx
            target_cursor = child_tgt_idx

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
    unmatched_tgt_indices = np.flatnonzero(~matched_tgt)
    all_tgt_unmatched_are_leaves = all(
        tgt_children_count[j] == 0 for j in unmatched_tgt_indices
    )

    exact_match_allows_target_root_wrappers = False
    src_root_exact_tgt_indices = [
        int(exact_mapping[i])
        for i in range(J_src)
        if src_parents[i] < 0 and i in exact_mapping
    ]
    if (
        len(exact_mapping) == J_src
        and unmatched_tgt_indices.size > 0
        and src_root_exact_tgt_indices
    ):
        allowed_wrapper_indices: set[int] = set()
        for mapped_root_tgt_idx in src_root_exact_tgt_indices:
            current_tgt_idx = int(tgt_parents[mapped_root_tgt_idx])
            while current_tgt_idx >= 0:
                allowed_wrapper_indices.add(current_tgt_idx)
                current_tgt_idx = int(tgt_parents[current_tgt_idx])

        exact_match_allows_target_root_wrappers = bool(allowed_wrapper_indices) and all(
            int(tgt_idx) in allowed_wrapper_indices and tgt_children_count[int(tgt_idx)] == 1
            for tgt_idx in unmatched_tgt_indices
        )

    is_exact_match = (
        len(exact_mapping) > 0
        and all_src_unmatched_are_leaves
        and (all_tgt_unmatched_are_leaves or exact_match_allows_target_root_wrappers)
    )

    if is_exact_match:
        # Exact match (possibly with leaf-only differences) — use it directly
        unmatched_src_count = J_src - len(exact_mapping)
        unmatched_tgt_count = int(unmatched_tgt_indices.size)
        if verbose:
            leaf_note = ""
            if unmatched_src_count or unmatched_tgt_count:
                if exact_match_allows_target_root_wrappers and unmatched_tgt_count:
                    leaf_note = f" ({unmatched_tgt_count} tgt root wrappers skipped)"
                else:
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

    # ── C) Promote effective translation root to target root ──────────────
    #
    # Some source skeletons have a static wrapper root (e.g. Horse ``Hips``)
    # above the actual locomotion joint (e.g. ``Bip01``).  Semantic mapping
    # will match ``Hips`` → target root, leaving ``Bip01`` unmapped.  But the
    # animated translation lives in ``Bip01``'s local position channel, not
    # in the wrapper root.  If we leave the wrapper root mapped, the target
    # will receive zero translation (the wrapper never moves).
    #
    # When ``src_effective_root_index`` is provided and is currently unmapped,
    # we promote it to the target root, demoting the previously-mapped wrapper
    # root (only if it is an ancestor of the effective root, so we don't
    # accidentally swap unrelated joints).
    root_tgt_indices = np.flatnonzero(tgt_parents < 0)
    root_tgt_idx = int(root_tgt_indices[0]) if root_tgt_indices.size > 0 else -1
    locomotion_tgt_idx = root_tgt_idx
    if tgt_effective_root_index is not None:
        tgt_effective_root_index = int(tgt_effective_root_index)
        if 0 <= tgt_effective_root_index < J_tgt:
            locomotion_tgt_idx = tgt_effective_root_index
    if (
        src_effective_root_index is not None
        and locomotion_tgt_idx >= 0
    ):
        src_effective_root_index = int(src_effective_root_index)
        if 0 <= src_effective_root_index < J_src:
            current_locomotion_src = np.flatnonzero(src_to_tgt == locomotion_tgt_idx)
            current_locomotion_src_idx = (
                int(current_locomotion_src[0]) if current_locomotion_src.size > 0 else -1
            )
            should_promote_effective_root = (
                current_locomotion_src_idx < 0
                or _is_source_ancestor(current_locomotion_src_idx, src_effective_root_index)
                or _is_source_ancestor(src_effective_root_index, current_locomotion_src_idx)
            )
            if (
                should_promote_effective_root
                and current_locomotion_src_idx != src_effective_root_index
            ):
                previous_tgt_for_effective = int(src_to_tgt[src_effective_root_index])
                if current_locomotion_src_idx >= 0:
                    src_to_tgt[current_locomotion_src_idx] = -1
                src_to_tgt[src_effective_root_index] = locomotion_tgt_idx
                should_shift_displaced_target = (
                    previous_tgt_for_effective >= 0
                    and previous_tgt_for_effective != locomotion_tgt_idx
                    and not _is_target_ancestor(previous_tgt_for_effective, locomotion_tgt_idx)
                )
                if should_shift_displaced_target:
                    _shift_descendant_chain_after_root_promotion(
                        src_effective_root_index,
                        previous_tgt_for_effective,
                    )
                if verbose:
                    replaced_name = (
                        src_match_names[current_locomotion_src_idx]
                        if current_locomotion_src_idx >= 0 else '<none>'
                    )
                    displaced_name = (
                        tgt_match_names[previous_tgt_for_effective]
                        if (
                            previous_tgt_for_effective >= 0
                            and previous_tgt_for_effective != locomotion_tgt_idx
                        ) else '<none>'
                    )
                    target_label = (
                        'target effective root'
                        if locomotion_tgt_idx != root_tgt_idx else 'target root'
                    )
                    print(
                        f"[retarget] Promoting source effective root "
                        f"{src_match_names[src_effective_root_index]!r} to {target_label} "
                        f"(replacing {replaced_name!r}, displaced target match {displaced_name!r})"
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

    # No rest-pose translation alignment: both source and target go through
    # process_anim() (root centered at XZ origin, feet grounded at y≈0, common
    # scale_factor), so any rest-based t_align tends to introduce drift rather
    # than correct it — especially Y, where the bone-length ratio used for
    # `scale` rarely matches the root-height ratio across species.
    t_align = np.zeros(3, dtype=np.float64)
    pos_src_rest_st = pos_src_rest * scale

    candidates = generate_coordinate_candidates_np() if coordinate_search else [
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
    bridge_src_for_tgt = _build_target_bridge_source_indices(
        src_for_tgt,
        src_to_tgt,
        src_parents,
        tgt_parents,
        tgt_children_count,
    )

    def _bridge_parent_world_rotation(bridge_src_idx: int) -> np.ndarray | None:
        src_parent_idx = int(src_parents[bridge_src_idx])
        if src_parent_idx < 0:
            return None
        return aligned_src_wrot[:, src_parent_idx]

    target_wrot = np.zeros((F_q, J_tgt, 4), dtype=np.float64)
    target_wpos = np.zeros((F_q, J_tgt, 3), dtype=np.float64)

    _EPS = 1e-8

    def _stable_valid(bn):
        """Per-frame direction validity, made temporally stable against flicker.

        ``bn`` is the per-frame aligned source-bone length; a frame is normally
        valid (direction transfer) when ``bn > _EPS`` and otherwise falls back to
        the rest offset. A structurally near-zero source bone (e.g. a zero-length
        helper like Bip01_Pelvis driving the dragon Spine) has a length that sits
        in the numerical noise floor and STRADDLES ``_EPS`` — some frames above,
        some below — so the per-frame choice flips every frame, snapping the mapped
        target joint between two placements (visible as ~1-3 Hz jitter). When the
        mask is mixed like that, the bone provides no reliable direction in any
        frame, so we use the rest fallback consistently for the whole clip. A real
        (even very short) bone stays consistently above ``_EPS`` and is untouched.
        """
        valid = bn > _EPS
        if valid.any() and not valid.all():
            valid[:] = False
        return valid

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
            valid = _stable_valid(bn)
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
                bridge_parent_rot = _bridge_parent_world_rotation(bridge_src_idx)
                if bridge_parent_rot is not None:
                    transport[:, j] = bridge_parent_rot
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
                src_rest_len = float(np.linalg.norm(src_rest_offsets[ii]))

                if src_rest_len > _EPS:
                    # Structurally real source bone: always transfer the source's
                    # animated direction × stretch, with no rest fallback. The
                    # transferred length L = tgt_rest_len · (src_anim_len /
                    # src_rest_len) goes to zero exactly when the source bone is
                    # animated onto its parent (e.g. a control bone like C_ctrl
                    # pulled onto Hips), so the target joint correctly coincides
                    # with its parent. Falling back to the rest offset here would
                    # discard that pure-translation animation and lift the whole
                    # subtree off the ground. Normalize defensively on the rare
                    # exactly-degenerate frame; L≈0 there makes the direction
                    # irrelevant.
                    safe = bn > _EPS
                    d = src_bv_aligned[:, ii] / np.where(safe, bn, 1.0)[:, None]
                    stretch = src_anim_len[:, ii] / src_rest_len      # (F,)
                    L = tgt_rest_len * stretch                     # (F,)
                    target_wpos[:, j] = target_wpos[:, p] + L[:, None] * d
                else:
                    # Structurally degenerate source bone (≈0 rest length, e.g. a
                    # zero-length locator / control bone). Two sub-cases:
                    #   (a) the bone is also ≈0 animated, or its length flickers
                    #       across the _EPS noise floor frame-to-frame — the
                    #       direction is meaningless, so fall back to the target's
                    #       rest offset consistently for the whole clip (the
                    #       _stable_valid jitter fix; e.g. a zero-len Bip01_Pelvis
                    #       driving the dragon Spine).
                    #   (b) the bone has a CONSISTENT, real animated length despite
                    #       its ≈0 rest length — e.g. a zero-rest control bone
                    #       ('C_ctrl' under Hips) that carries a pure *translation*
                    #       lifting the whole body during locomotion. That motion
                    #       lives only in the animated offset, not in any rest bone
                    #       length, so it cannot be expressed as tgt_rest_len·dir:
                    #       tgt_rest_len ≈ 0 would collapse the joint onto its parent
                    #       and sink the whole subtree below the floor (manifested as
                    #       restored Run-gait feet punching through the ground while
                    #       calm clips, whose control bone stays inert, were fine).
                    #       Transfer the source's animated world offset verbatim —
                    #       exact for self-retarget, and the only faithful option
                    #       when there is no rest proportion to scale from.
                    valid = _stable_valid(bn)
                    dir_pos = target_wpos[:, p] + src_bv_aligned[:, ii]
                    rest_pos = target_wpos[:, p] + quat_rotate_wxyz_np(
                        transport[:, p], rest_off_j,
                    )
                    target_wpos[:, j] = np.where(valid[:, None], dir_pos, rest_pos)
            if p1 < 0:
                # Source joint is the root but the target joint sits under one or
                # more synthetic wrapper roots. Preserve the source root's world
                # position directly instead of snapping back to the target rest
                # offset under the wrapper chain.
                target_wpos[:, j] = aligned_src_wpos[:, ii]
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
        bridge_src_idx = int(bridge_src_for_tgt[p_idx])
        if ii >= 0:
            target_wrot[:, p_idx] = aligned_src_wrot[:, ii]
        elif bridge_src_idx >= 0:
            bridge_parent_rot = _bridge_parent_world_rotation(bridge_src_idx)
            if bridge_parent_rot is not None:
                target_wrot[:, p_idx] = bridge_parent_rot
            elif p_par < 0:
                target_wrot[:, p_idx] = np.repeat(
                    tgt_rest_wrot[0, p_idx][None], F_q, axis=0
                )
            else:
                target_wrot[:, p_idx] = quat_multiply_wxyz_np(
                    target_wrot[:, p_par],
                    np.repeat(tgt_rest_rotations[p_idx:p_idx + 1], F_q, axis=0),
                )
        elif p_par < 0:
            target_wrot[:, p_idx] = np.repeat(
                tgt_rest_wrot[0, p_idx][None], F_q, axis=0
            )
        else:
            target_wrot[:, p_idx] = quat_multiply_wxyz_np(
                target_wrot[:, p_par],
                np.repeat(tgt_rest_rotations[p_idx:p_idx + 1], F_q, axis=0),
            )

    # When a mapped source root lands under one or more unmatched single-child
    # target wrappers, the realized Blender channels must move that wrapper
    # chain, not just the mapped child itself. Back-propagate the mapped root's
    # desired world transform through the wrapper rest chain so inverse-FK can
    # emit a non-zero target root wrapper transform that Blender can reproduce.
    # Feature-space retarget callers can opt out by naming a non-root target
    # effective root: in that case locomotion should stay on that joint's local
    # translation, leaving wrapper ancestors static.
    for tgt_joint_idx in range(J_tgt):
        src_joint_idx = int(src_for_tgt[tgt_joint_idx])
        if src_joint_idx < 0 or int(src_parents[src_joint_idx]) >= 0:
            continue
        if (
            tgt_effective_root_index is not None
            and int(tgt_joint_idx) == int(tgt_effective_root_index)
            and int(tgt_parents[tgt_joint_idx]) >= 0
        ):
            continue

        current_child_idx = int(tgt_joint_idx)
        current_world_pos = target_wpos[:, current_child_idx].copy()
        current_world_rot = target_wrot[:, current_child_idx].copy()
        parent_idx = int(tgt_parents[current_child_idx])

        while (
            parent_idx >= 0
            and int(src_for_tgt[parent_idx]) < 0
            and int(bridge_src_for_tgt[parent_idx]) < 0
            and int(tgt_children_count[parent_idx]) == 1
        ):
            child_rest_rot = np.repeat(
                tgt_rest_rotations[current_child_idx:current_child_idx + 1],
                F_q,
                axis=0,
            )
            child_rest_offset = np.repeat(
                tgt_rest_offsets[current_child_idx:current_child_idx + 1],
                F_q,
                axis=0,
            )
            parent_world_rot = quat_multiply_wxyz_np(
                current_world_rot,
                quat_conjugate_wxyz_np(child_rest_rot),
            )
            parent_world_pos = current_world_pos - quat_rotate_wxyz_np(
                parent_world_rot,
                child_rest_offset,
            )

            target_wrot[:, parent_idx] = parent_world_rot
            target_wpos[:, parent_idx] = parent_world_pos

            current_child_idx = parent_idx
            current_world_pos = parent_world_pos
            current_world_rot = parent_world_rot
            parent_idx = int(tgt_parents[current_child_idx])

    # When tgt_effective_root_index is a non-root joint, pin all ancestor
    # wrapper nodes to the hierarchy root's rest world position so that the
    # effective root's local translation carries the full locomotion without
    # being offset by wrapper rest offsets.  This keeps the wrapper chain
    # semantically inert while preserving the FK relationship.
    if (
        tgt_effective_root_index is not None
        and int(tgt_effective_root_index) >= 0
        and int(tgt_parents[int(tgt_effective_root_index)]) >= 0
    ):
        _anchor_pos = tgt_rest_wpos[0, root_tgt_idx]  # (3,)
        _anchor_rot = tgt_rest_wrot[0, root_tgt_idx]   # (4,)
        _ancestor_list: list[int] = []
        _cur = int(tgt_parents[int(tgt_effective_root_index)])
        while _cur >= 0:
            _ancestor_list.append(_cur)
            _cur = int(tgt_parents[_cur])
        for _aj in _ancestor_list:
            target_wpos[:, _aj] = _anchor_pos[np.newaxis, :]
            target_wrot[:, _aj] = np.repeat(
                _anchor_rot[np.newaxis], F_q, axis=0
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

    # Inverse-FK can emit synthetic pose-location residuals on both mapped and
    # bridge joints when the target skeleton needs extra segments or different
    # proportions to realize the transferred world-space pose. Preserve only
    # source-driven local translations: if the controlling source bone has no
    # animated local translation, keep the target joint rigid and let rotation
    # carry the motion instead of baking the residual into pose translation.
    _POSE_LOCATION_EPS = 1e-5
    for j in range(J_tgt):
        if int(tgt_parents[j]) < 0:
            continue

        # The locomotion joint (an effective root sitting under static wrapper
        # joints, e.g. a Bip01 hierarchy's Bip01 below Hips→Ctrl) must keep its
        # full inverse-FK local translation: that translation IS the character's
        # global locomotion. The source carries locomotion in its root-translation
        # channel, not a per-joint pose-location channel, so the suppression test
        # below (which only inspects the source pose-location channel) sees ~0 and
        # would wrongly zero it — freezing the whole character at its rest offset
        # (sinking it below the floor). When the effective root is itself the
        # hierarchy root the ``tgt_parents[j] < 0`` guard already skips it, so this
        # only matters for the wrapped case and is otherwise a no-op.
        if j == locomotion_tgt_idx:
            continue

        controller_src_idx = int(src_for_tgt[j])
        if controller_src_idx < 0:
            controller_src_idx = int(bridge_src_for_tgt[j])
        if controller_src_idx < 0:
            continue

        if pose_locations_np is None:
            tgt_pose_loc[:, j] = 0.0
            continue

        source_pose_loc = np.asarray(pose_locations_np[:, controller_src_idx], dtype=np.float64)
        if np.max(np.linalg.norm(source_pose_loc, axis=-1)) > _POSE_LOCATION_EPS:
            continue

        tgt_pose_loc[:, j] = 0.0

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
    if pose_locations_np is not None:
        out_bone_translations = tgt_pose_loc
    else:
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

# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    import sys
    import os as _os

    # Ensure both the Anytop dir and its parent are on sys.path so the
    # bare ``utils.*`` / ``data_loaders.*`` / ``motion_lib.*`` imports
    # resolve regardless of CWD.
    _ANYTOP_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    _REPO_ROOT = _os.path.dirname(_ANYTOP_ROOT)
    sys.path.insert(0, _REPO_ROOT)
    sys.path.insert(0, _ANYTOP_ROOT)

    _PARSER = argparse.ArgumentParser(
        description='Cross-skeleton motion retargeting — retarget an animation '
                    '(.npy/.glb/.fbx) onto a target skeleton.'
    )
    _PARSER.add_argument(
        '--source', required=True, type=str,
        help='Path to source animation file (.npy / .glb / .fbx).',
    )
    _PARSER.add_argument(
        '--object_type', required=True, type=str,
        help='Target object type name (e.g. Horse, Buffalo, Dragon). '
             'Must match a key in the cond.npy dataset.',
    )
    _PARSER.add_argument(
        '--cond_path', default=None, type=str,
        help='Optional path to an additional cond.npy file. Entries in this '
             'file are merged into the default training cond (not replaced). '
             'Useful for providing cond entries for skeletons not in the '
             'training set.',
    )
    _PARSER.add_argument(
        '--output_dir', default=None, type=str,
        help='Output directory for retargeted .npy and inspection .bvh files. '
             'Default: Anytop/outputs/retarget_output',
    )

    _cli_args = _PARSER.parse_args()

    # ── Resolve paths ─────────────────────────────────────────────────────
    _source_path = _os.path.abspath(_cli_args.source)
    if not _os.path.isfile(_source_path):
        _PARSER.error(f'--source file not found: {_source_path}')

    _suffix = _os.path.splitext(_source_path)[1].lower()
    if _suffix not in {'.npy', '.glb', '.fbx', '.gltf'}:
        _PARSER.error(
            f'Unsupported source format "{_suffix}". '
            f'Supported: .npy, .glb, .fbx, .gltf'
        )

    _default_cond = _os.path.join(
        _ANYTOP_ROOT, 'dataset', 'truebones', 'zoo',
        'truebones_processed', 'cond.npy',
    )
    if _cli_args.cond_path:
        _extra_cond_path = _os.path.abspath(_cli_args.cond_path)
        if not _os.path.isfile(_extra_cond_path):
            _PARSER.error(f'--cond_path file not found: {_extra_cond_path}')
    else:
        _extra_cond_path = None

    if _cli_args.output_dir:
        _output_dir = _os.path.abspath(_cli_args.output_dir)
    else:
        _output_dir = _os.path.join(_ANYTOP_ROOT, 'outputs', 'retarget_output')
    _os.makedirs(_output_dir, exist_ok=True)

    # ── Load cond ──────────────────────────────────────────────────────────
    if not _os.path.isfile(_default_cond):
        _PARSER.error(
            f'Default training cond not found at {_default_cond}. '
            f'Run preprocessing first or provide --cond_path.'
        )

    _cond_dict = dict(np.load(_default_cond, allow_pickle=True).item())

    if _extra_cond_path:
        _extra_cond = dict(np.load(_extra_cond_path, allow_pickle=True).item())
        # Merge: extra entries override/add to default, but don't replace
        # keys that already exist in default — we only add new object_types.
        for _key, _val in _extra_cond.items():
            if _key not in _cond_dict:
                _cond_dict[_key] = _val
        print(f'[retarget CLI] Merged {len(_extra_cond)} extra cond entries '
              f'(total: {len(_cond_dict)}).')

    _target_type = _cli_args.object_type
    if _target_type not in _cond_dict:
        _PARSER.error(
            f'Target object_type "{_target_type}" not found in cond. '
            f'Available: {sorted(_cond_dict.keys())}'
        )

    _tgt_cond = dict(_cond_dict[_target_type])
    _max_joints = max(
        len(np.asarray(_cond_dict[_k]['parents']))
        for _k in _cond_dict
    )

    # ── Retarget ───────────────────────────────────────────────────────────
    from data_loaders.truebones.truebones_utils.features import (
        get_common_features_from_T_pose,
    )
    from data_loaders.truebones.truebones_utils.motion_process import (
        recover_bvh_export_animation_from_motion_np,
    )
    from motion_lib import BVH

    _base_name = _os.path.splitext(_os.path.basename(_source_path))[0]

    from utils.misc import resolve_dataset_path as _resolve_dataset_path
    _tgt_tpose_path = _resolve_dataset_path(_tgt_cond.get('orientation_reference_fbx_path'))
    if not _tgt_tpose_path or not _os.path.isfile(_tgt_tpose_path):
        _PARSER.error(
            f'Target T-pose file not found for "{_target_type}": '
            f'{_tgt_tpose_path!r}'
        )

    _tgt_tp = get_common_features_from_T_pose(
        _tgt_tpose_path, _target_type,
        max_joints=_max_joints,
    )

    # Resolve FPS from target cond (used for BVH export frametime).
    _fps = float(_tgt_cond.get('fps', 30.0))

    if _suffix == '.npy':
        # Feature-space .npy source: infer source object_type from filename,
        # then delegate to retarget_features_npy_to_target.
        from utils.misc import infer_object_type_from_filename
        from Anytop.utils.auto_retarget import retarget_features_npy_to_target

        _src_type = infer_object_type_from_filename(
            _source_path,
            valid_types=set(_cond_dict.keys()),
        )
        if _src_type is None:
            _PARSER.error(
                f'Could not infer source object_type from filename '
                f'"{_base_name}". For .npy sources the filename must contain '
                f'a known object_type (e.g. Horse___Run_01.npy).'
            )
        if _src_type == _target_type:
            # Same skeleton — no retarget needed; copy source features as-is.
            print(f'[retarget CLI] Source and target are the same type '
                  f'("{_src_type}") — skipping retarget.')
            _target_features = np.load(_source_path).astype(np.float32)
        else:
            _src_cond = dict(_cond_dict[_src_type])
            _src_features = np.load(_source_path).astype(np.float32)

            print(f'[retarget CLI] Retargeting {_src_type} → {_target_type} '
                  f'(source: {_src_features.shape})')

            _target_features = retarget_features_npy_to_target(
                _src_features,
                _src_cond,
                _src_type,
                _tgt_tp,
                _target_type,
                _max_joints,
                source_tp=None,
                target_cond=_tgt_cond,
            )
    else:
        # Raw animation file (.glb/.fbx/.gltf): cond-free on the source side.
        from Anytop.utils.auto_retarget import retarget_animation_file_to_target

        print(f'[retarget CLI] Retargeting {_source_path} → {_target_type}')

        _target_features = retarget_animation_file_to_target(
            _source_path,
            _tgt_tp,
            _target_type,
            _max_joints,
            _tgt_cond,
        )

    if _target_features is None:
        sys.exit(
            f'ERROR: Retargeting failed ({_source_path} → {_target_type}). '
            f'Check joint-name overlap between source and target skeletons.'
        )

    # ── Save outputs ───────────────────────────────────────────────────────
    _out_npy = _os.path.join(
        _output_dir,
        f'_retargeted_to_{_target_type}__{_base_name}.npy',
    )
    np.save(_out_npy, _target_features)
    print(f'[retarget CLI] Retargeted features {_target_features.shape} → {_out_npy}')

    # Inspection BVH
    try:
        _out_bvh = _out_npy[:-4] + '.bvh'
        _out_anim, _joint_names, _has_pos = recover_bvh_export_animation_from_motion_np(
            _target_features,
            np.asarray(_tgt_cond['parents'], dtype=np.int32),
            np.asarray(_tgt_cond['offsets'], dtype=np.float32),
            list(_tgt_cond.get('canonical_bvh_joint_names',
                               _tgt_cond['joints_names'])),
            allow_infer=True,
            tpose_rest_rotations=_tgt_tp.tpos_rots[0],
        )
        if _out_anim is not None:
            BVH.save(
                _out_bvh, _out_anim, _joint_names,
                frametime=1.0 / _fps, positions=_has_pos,
                order='auto',
            )
            print(f'[retarget CLI] Inspection BVH → {_out_bvh}')
    except Exception as _exc:
        print(f'[retarget CLI] WARNING: Failed to write inspection BVH: {_exc}')
